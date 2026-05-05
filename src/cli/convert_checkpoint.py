import json
import os
import shutil
from argparse import ArgumentParser

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm


FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def dequant_e2m1fn(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.int8
    assert weight.ndim == 2
    out_dim, packed_in_dim = weight.shape
    in_dim = packed_in_dim * 2
    fp4_block_size = 32
    assert scale.shape == (out_dim, in_dim // fp4_block_size)

    raw = weight.view(torch.uint8)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    weight = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(1)
    scale = scale.float().repeat_interleave(fp4_block_size, dim=1)
    return weight * scale


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    assert weight.dtype == torch.float8_e4m3fn
    scale = scale.float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    scale = scale[:weight.shape[0], :weight.shape[1]]
    return weight.float() * scale


def quantize_int8_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = weight.float().contiguous()
    scale = weight.abs().amax(dim=1).clamp_min(1e-6) / 127.0
    weight = torch.clamp(torch.round(weight / scale.unsqueeze(1)), -127, 127).to(torch.int8)
    return weight.contiguous(), scale.float().contiguous()


def convert_routed_expert(weight: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if scale.ndim == 1:
        return weight.contiguous(), scale.float().contiguous()
    return quantize_int8_weight(dequant_e2m1fn(weight, scale))


# Modelslim's V4-Flash W8A8 policy (from lab_practice/deepseek_v4/deepseek_w8a8.yaml):
#
# INCLUDE (quantize to INT8):
#   *attn* except wo_a, wo_b, compressor.wgate, compressor.wkv,
#           indexer.weights_proj, indexer.compressor.wgate, indexer.compressor.wkv
#   *ffn*  except gate
#
# So quantized: wq_a, wq_b, wkv, experts.w1/w2/w3, shared_experts.w1/w2/w3
# NOT quantized: wo_a, wo_b, gate, lm_head, embed, compressor.*, indexer.specials
#

_ATTN_QUANTIZED_PROJECTIONS = (".wq_a.", ".wq_b.", ".wkv.")
_ATTN_EXCLUDED_PROJECTIONS = (".wo_a.", ".wo_b.",
                              ".compressor.wgate.", ".compressor.wkv.",
                              ".indexer.weights_proj.",
                              ".indexer.compressor.wgate.", ".indexer.compressor.wkv.")
_FFN_EXCLUDED = (".gate.",)


def should_convert_int8(name: str, tensor: torch.Tensor) -> bool:
    if not name.endswith(".weight"):
        return False

    # Routed experts: INT8 in → INT8 out (already quantized upstream)
    if "ffn.experts." in name and "shared_experts" not in name:
        return tensor.dtype == torch.int8

    # Shared experts w1/w2/w3: FP8 → INT8
    if "ffn.shared_experts." in name:
        if any(ex in name for ex in _FFN_EXCLUDED):
            return False
        return tensor.dtype == torch.float8_e4m3fn

    # Attention: only wq_a, wq_b, wkv → INT8; wo_a, wo_b, compressor, indexer specials excluded
    if ".attn." in name:
        if any(exc in name for exc in _ATTN_EXCLUDED_PROJECTIONS):
            return False
        if any(proj in name for proj in _ATTN_QUANTIZED_PROJECTIONS):
            return tensor.dtype == torch.float8_e4m3fn
        return False

    return False


def convert_weight(name: str, weight: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if "ffn.experts." in name and "shared_experts" not in name:
        return convert_routed_expert(weight, scale)
    return quantize_int8_weight(dequant_fp8(weight, scale))


def copy_auxiliary_files(hf_ckpt_path: str, save_path: str) -> None:
    for file_name in [
        "config.json",
        "configuration.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        src = os.path.join(hf_ckpt_path, file_name)
        dst = os.path.join(save_path, file_name)
        if os.path.exists(src):
            shutil.copyfile(src, dst)


def convert(hf_ckpt_path: str, save_path: str) -> None:
    index_path = os.path.join(hf_ckpt_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]
    file_names = sorted(dict.fromkeys(weight_map.values()))
    os.makedirs(save_path, exist_ok=True)

    total_size = 0
    for file_name in tqdm(file_names):
        src = os.path.join(hf_ckpt_path, file_name)
        dst = os.path.join(save_path, file_name)
        state_dict = {}
        consumed = set()

        with safe_open(src, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name in consumed:
                    continue

                tensor = f.get_tensor(name)
                if should_convert_int8(name, tensor):
                    scale_name = name.replace(".weight", ".scale")
                    scale = f.get_tensor(scale_name)
                    tensor, scale = convert_weight(name, tensor, scale)
                    state_dict[name] = tensor
                    state_dict[scale_name] = scale
                    consumed.add(scale_name)
                elif name.endswith(".scale") and name.replace(".scale", ".weight") in consumed:
                    continue
                else:
                    state_dict[name] = tensor

        save_file(state_dict, dst)
        total_size += sum(t.numel() * t.element_size() for t in state_dict.values())

    index_data["metadata"] = {"total_size": total_size}
    with open(os.path.join(save_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index_data, f, indent=2)
        f.write("\n")

    copy_auxiliary_files(hf_ckpt_path, save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()
    convert(args.hf_ckpt_path, args.save_path)
