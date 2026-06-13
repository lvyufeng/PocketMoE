from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from src.loader.gguf.bundle import read_gguf_bundle
from src.models.minimax_m2.loader import load_minimax_m2_gguf_model
from src.models.minimax_m2.moe_planning import build_minimax_m2_tp_routed_resident_plan, minimax_m2_tp_expert_range


def setup_dist() -> tuple[int, int, int, torch.device]:
    world = int(__import__("os").environ.get("WORLD_SIZE", "1"))
    rank = int(__import__("os").environ.get("RANK", "0"))
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", str(rank)))
    if not torch.cuda.is_available():
        raise RuntimeError("MiniMax-M2 GGUF runtime requires CUDA")
    torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    return world, rank, local_rank, torch.device("cuda", local_rank)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def read_seed_tokens(path: str | Path) -> list[int]:
    data = Path(path).read_bytes()
    if len(data) == 0:
        raise ValueError(f"empty seed-file: {path}")
    if len(data) % 8 == 0:
        arr = np.frombuffer(data, dtype="<i8")
        if arr.size and int(arr.min()) >= 0 and int(arr.max()) < 2**31:
            return [int(x) for x in arr]
    if len(data) % 4 == 0:
        arr = np.frombuffer(data, dtype="<i4")
        return [int(x) for x in arr]
    raise ValueError(f"seed-file byte length must be divisible by 4 or 8, got {len(data)}")


def parse_tokens_csv(text: str) -> list[int]:
    vals = [item.strip() for item in text.replace("\n", ",").split(",") if item.strip()]
    if not vals:
        raise ValueError("--tokens did not contain any token ids")
    return [int(v) for v in vals]


@torch.inference_mode()
def greedy_generate_token_ids(
    model,
    prompt_tokens: list[int],
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> tuple[list[int], dict[str, float]]:
    if not prompt_tokens:
        raise ValueError("prompt_tokens must not be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    device = model.device
    total_len = len(prompt_tokens) + max_new_tokens
    model.reset_cache(batch_size=1, max_seq_len=max(1, total_len))

    generated: list[int] = []
    prefill_time = 0.0
    decode_time = 0.0
    if max_new_tokens == 0:
        return generated, {"prefill_seconds": 0.0, "decode_seconds": 0.0, "prefill_tokens": 0.0, "decode_tokens": 0.0}

    prompt = torch.tensor([prompt_tokens], device=device, dtype=torch.long)
    sync()
    t0 = time.perf_counter()
    next_token = model.forward(prompt, 0, return_next_token=True)
    sync()
    prefill_time = time.perf_counter() - t0
    token = int(next_token.reshape(-1)[0].item())
    generated.append(token)

    for step in range(1, max_new_tokens):
        if eos_token_id is not None and generated[-1] == int(eos_token_id):
            break
        pos = len(prompt_tokens) + step - 1
        inp = torch.tensor([[generated[-1]]], device=device, dtype=torch.long)
        sync()
        t0 = time.perf_counter()
        next_token = model.forward(inp, pos, return_next_token=True)
        sync()
        decode_time += time.perf_counter() - t0
        generated.append(int(next_token.reshape(-1)[0].item()))

    stats = {
        "prefill_seconds": float(prefill_time),
        "decode_seconds": float(decode_time),
        "prefill_tokens": float(len(prompt_tokens)),
        "decode_tokens": float(max(0, len(generated) - 1)),
    }
    return generated, stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniMax-M2.7 GGUF raw-block CUDA greedy generation smoke")
    parser.add_argument("--gguf-path", type=str, required=True)
    parser.add_argument("--seed-file", type=str, default="")
    parser.add_argument("--tokens", type=str, default="", help="comma-separated token ids when --seed-file is not used")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--layers", type=int, default=0, help="debug: limit layer count; 0 means full model")
    parser.add_argument("--gpu-memory-gib", type=float, default=22.0)
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16"], default="float16")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    world, rank, local_rank, device = setup_dist()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    if args.seed_file:
        prompt_tokens = read_seed_tokens(args.seed_file)
    elif args.tokens:
        prompt_tokens = parse_tokens_csv(args.tokens)
    else:
        raise ValueError("provide --seed-file or --tokens")

    bundle = read_gguf_bundle(args.gguf_path)
    tp_plan = build_minimax_m2_tp_routed_resident_plan(bundle, tp_world=world, gpu_memory_gib=float(args.gpu_memory_gib))
    if world > 1:
        if not tp_plan.ok:
            raise RuntimeError(f"MiniMax-M2 TP plan is not valid: {tp_plan.errors[:5]}")
        rplan = tp_plan.ranks[rank]
        expert_start, expert_count = rplan.expert_start, rplan.expert_count
    else:
        expert_start, expert_count = minimax_m2_tp_expert_range(
            int(bundle.metadata.get("minimax-m2.expert_count", 256)), 1, 0
        )
    n_layers = None if int(args.layers) <= 0 else int(args.layers)
    if rank == 0:
        print(
            f"minimax_load_start world={world} layers={n_layers or 'full'} prompt_tokens={len(prompt_tokens)} "
            f"max_new_tokens={args.max_new_tokens} dtype={args.dtype}",
            flush=True,
        )
    t_load = time.perf_counter()
    model, info = load_minimax_m2_gguf_model(
        bundle,
        device=device,
        dtype=dtype,
        n_layers=n_layers,
        expert_start=expert_start,
        expert_count=expert_count,
        preload_moe=True,
    )
    sync()
    load_seconds = time.perf_counter() - t_load
    alloc_gib = torch.cuda.memory_allocated(device) / 1024**3
    reserved_gib = torch.cuda.memory_reserved(device) / 1024**3
    print(
        f"rank={rank} local_rank={local_rank} expert_range=[{expert_start},{expert_start + expert_count}) "
        f"load_seconds={load_seconds:.3f} cuda_alloc_gib={alloc_gib:.3f} cuda_reserved_gib={reserved_gib:.3f}",
        flush=True,
    )

    eos = bundle.metadata.get("tokenizer.ggml.eos_token_id")
    eos_id = int(eos) if isinstance(eos, int) else None
    generated, stats = greedy_generate_token_ids(
        model,
        prompt_tokens,
        max_new_tokens=int(args.max_new_tokens),
        eos_token_id=eos_id,
    )
    if rank == 0:
        prefill_tps = stats["prefill_tokens"] / max(stats["prefill_seconds"], 1.0e-9)
        decode_tps = stats["decode_tokens"] / max(stats["decode_seconds"], 1.0e-9) if stats["decode_tokens"] > 0 else 0.0
        print(
            f"RESULT prefill_seconds={stats['prefill_seconds']:.6f} prefill_tokens={int(stats['prefill_tokens'])} "
            f"prefill_tps={prefill_tps:.2f} decode_seconds={stats['decode_seconds']:.6f} "
            f"decode_tokens={int(stats['decode_tokens'])} decode_tps={decode_tps:.2f}",
            flush=True,
        )
        print("generated_token_ids=" + ",".join(str(t) for t in generated), flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
