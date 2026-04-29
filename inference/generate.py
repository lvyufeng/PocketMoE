import os
import json
import sys
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors import safe_open
from safetensors.torch import load_model

from model import (
    Transformer,
    ModelArgs,
    ParallelEmbedding,
    ParallelHead,
    ColumnParallelLinear,
    RowParallelLinear,
    Attention,
)
current_dir = os.path.dirname(os.path.abspath(__file__))
encoding_dir = os.path.join(current_dir, '../encoding')
sys.path.insert(0, os.path.abspath(encoding_dir))
from encoding_dsv4 import encode_messages, parse_message_from_completion_text


def _cpu_affinity_for_rank(local_rank: int, world_size: int) -> list[int] | None:
    topo_root = "/sys/devices/system/cpu"
    if not hasattr(os, "sched_setaffinity") or not os.path.isdir(topo_root):
        return None
    core_map: dict[tuple[int, int], list[int]] = {}
    for name in os.listdir(topo_root):
        if not name.startswith("cpu") or not name[3:].isdigit():
            continue
        cpu = int(name[3:])
        try:
            with open(os.path.join(topo_root, name, "topology/core_id")) as f:
                core_id = int(f.read())
            with open(os.path.join(topo_root, name, "topology/physical_package_id")) as f:
                package_id = int(f.read())
        except OSError:
            continue
        core_map.setdefault((package_id, core_id), []).append(cpu)
    physical_cores = [sorted(v) for _, v in sorted(core_map.items())]
    if not physical_cores:
        return None
    base = len(physical_cores) // world_size
    extra = len(physical_cores) % world_size
    start = local_rank * base + min(local_rank, extra)
    count = base + (1 if local_rank < extra else 0)
    selected = physical_cores[start:start + count]
    cpus: list[int] = []
    for siblings in selected:
        cpus.extend(siblings)
    return sorted(cpus)


def _full_tensor_name(module_name: str, param_name: str) -> str:
    return f"{module_name}.{param_name}" if module_name else param_name


def _is_column_parallel(module) -> bool:
    return isinstance(module, (ParallelEmbedding, ParallelHead, ColumnParallelLinear))


def _is_row_parallel(module) -> bool:
    return isinstance(module, RowParallelLinear)


def _shard_tensor_for_rank(name: str, tensor: torch.Tensor, module, world_size: int, rank: int) -> torch.Tensor:
    if "experts." in name and "shared_experts" not in name:
        parts = name.split('.')
        expert_pos = parts.index("experts") + 1
        expert_idx = int(parts[expert_pos])
        local_experts = module.n_local_experts if hasattr(module, 'n_local_experts') else None
        if local_experts is None:
            return tensor
        start = rank * local_experts
        end = start + local_experts
        if expert_idx < start or expert_idx >= end:
            return None
        return tensor

    if _is_column_parallel(module):
        if tensor.ndim == 2:
            shard_dim = 0
        elif tensor.ndim == 1:
            shard_dim = 0
        else:
            return tensor
        assert tensor.size(shard_dim) % world_size == 0, f"{name} not divisible on dim {shard_dim}"
        shard = tensor.size(shard_dim) // world_size
        return tensor.narrow(shard_dim, rank * shard, shard).contiguous()

    if _is_row_parallel(module):
        if tensor.ndim == 2:
            shard_dim = 1
        elif tensor.ndim == 1:
            return tensor
        else:
            return tensor
        assert tensor.size(shard_dim) % world_size == 0, f"{name} not divisible on dim {shard_dim}"
        shard = tensor.size(shard_dim) // world_size
        return tensor.narrow(shard_dim, rank * shard, shard).contiguous()

    if isinstance(module, Attention) and name.endswith("attn_sink"):
        assert tensor.size(0) % world_size == 0, f"{name} not divisible on dim 0"
        shard = tensor.size(0) // world_size
        return tensor.narrow(0, rank * shard, shard).contiguous()

    return tensor


def load_original_hf_model(model: Transformer, ckpt_path: str, world_size: int, rank: int) -> None:
    state_dict = model.state_dict()
    name_to_module = dict(model.named_modules())
    loaded = set()
    weight_map_path = os.path.join(ckpt_path, "model.safetensors.index.json")
    with open(weight_map_path) as f:
        weight_map = json.load(f)["weight_map"]

    file_to_keys = {}
    for key, file_name in weight_map.items():
        file_to_keys.setdefault(file_name, []).append(key)

    for file_name, keys in file_to_keys.items():
        file_path = os.path.join(ckpt_path, file_name)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in keys:
                if key not in state_dict:
                    continue
                module_name, _, _ = key.rpartition('.')
                module = name_to_module.get(module_name)
                if module is None:
                    continue

                target = state_dict[key]

                if key.endswith("wo_a.weight"):
                    scale_key = key.replace("weight", "scale")
                    weight = _shard_tensor_for_rank(key, f.get_tensor(key), module, world_size, rank)
                    scale = _shard_tensor_for_rank(scale_key, f.get_tensor(scale_key), module, world_size, rank)
                    if weight is None or scale is None:
                        continue
                    tensor = weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float()
                    tensor = (tensor * scale[:, None, :, None].float()).flatten(2, 3).flatten(0, 1)
                    if tensor.shape != target.shape:
                        raise ValueError(f"Shape mismatch for {key}: got {tuple(tensor.shape)}, expected {tuple(target.shape)}")
                    target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                    loaded.add(key)
                    continue

                tensor = f.get_tensor(key)
                tensor = _shard_tensor_for_rank(key, tensor, module, world_size, rank)
                if tensor is None:
                    continue
                if tensor.shape != target.shape:
                    raise ValueError(f"Shape mismatch for {key}: got {tuple(tensor.shape)}, expected {tuple(target.shape)}")

                if target.dtype == torch.float4_e2m1fn_x2:
                    target.view(torch.uint8).copy_(tensor.view(torch.uint8).to(device=target.device))
                else:
                    target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                loaded.add(key)

    loaded.update({"mtp.0.embed.weight", "mtp.0.head.weight"})
    missing = sorted(set(state_dict.keys()) - loaded)
    if missing:
        raise ValueError(f"Missing {len(missing)} parameters from original HF checkpoint load, e.g. {missing[:10]}")

def sample(logits, temperature: float = 1.0):
    """Gumbel-max trick: equivalent to multinomial sampling but faster on GPU,
    since it avoids the GPU-to-CPU sync in torch.multinomial."""
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens))
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        toks.append(eos_id)
        completion_tokens.append(toks)
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    routed_experts_device: str = "gpu",
) -> None:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    if routed_experts_device == "cpu":
        import cpu_routed_backend
        affinity_cpus = _cpu_affinity_for_rank(local_rank, world_size)
        if affinity_cpus is not None:
            os.sched_setaffinity(0, affinity_cpus)
            cpu_routed_backend.configure_cpu_routed_runtime(omp_threads=max(len(affinity_cpus), 1))
        else:
            cpu_routed_backend.configure_cpu_routed_runtime(omp_threads=max((os.cpu_count() or 1) // world_size, 1))
        torch.set_num_threads(1)
    else:
        torch.set_num_threads(8)
    torch.manual_seed(33377335)
    with open(config) as f:
        config_data = json.load(f)
    config_data["routed_experts_device"] = routed_experts_device
    args = ModelArgs(**config_data)
    if interactive:
        args.max_batch_size = 1
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    print("load model")
    mp_ckpt_paths = [
        os.path.join(ckpt_path, f"model-{rank + 1:05d}-of-{world_size:05d}.safetensors"),
        os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"),
    ]
    mp_ckpt_path = next((path for path in mp_ckpt_paths if os.path.exists(path)), None)
    if mp_ckpt_path is not None:
        load_model(model, mp_ckpt_path, strict=False)
    else:
        load_original_hf_model(model, ckpt_path, world_size, rank)
    if routed_experts_device == "cpu":
        model.prepare_cpu_expert_int8()
    torch.set_default_device("cuda")
    print("I'm DeepSeek 👋")

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.encode(encode_messages(messages, thinking_mode="chat"))
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0])
            print(completion)
            messages.append(parse_message_from_completion_text(completion, thinking_mode="chat"))
    else:
        with open(input_file) as f:
            prompts = f.read().split("\n\n")
        prompt_tokens = [tokenizer.encode(encode_messages([{"role": "user", "content": prompt}], thinking_mode="chat")) for prompt in prompts]
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--routed-experts-device", type=str, choices=["gpu", "cpu"], default="gpu")
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature, args.routed_experts_device)
