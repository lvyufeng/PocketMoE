import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gguf.tensor_reader import GGUFTensorDataReader, get_iq1_grid_tensor
from src.kernels.cuda_loader import load_cuda_kernel

DEFAULT_GGUF = "/mnt/data1/dsv4_inference/.tmp/dsv4-dynamic-iq1m-antirez.gguf"


def _type_id(type_name: str) -> int:
    if type_name == "iq2_xxs":
        return 0
    if type_name == "q2_k":
        return 1
    if type_name == "iq1_m":
        return 2
    raise ValueError(f"unsupported type {type_name!r}")


def _grid_arg(type_name: str, device: torch.device) -> torch.Tensor:
    if type_name == "iq1_m":
        return get_iq1_grid_tensor().to(device=device, non_blocking=False).contiguous()
    return torch.empty(0, dtype=torch.int8, device=device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--layer", type=int, default=6, help="layer whose w1/w3/w2 are all expected to be iq1_m")
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--loops", type=int, default=20)
    args = ap.parse_args()

    cuda_mod = load_cuda_kernel()
    assert cuda_mod is not None and hasattr(cuda_mod, "gguf_quant_gemm_forward")
    torch.manual_seed(1234)
    device = torch.device("cuda", 0)
    names = [
        f"blk.{args.layer}.ffn_gate_exps.weight",
        f"blk.{args.layer}.ffn_up_exps.weight",
        f"blk.{args.layer}.ffn_down_exps.weight",
    ]
    with GGUFTensorDataReader(args.gguf) as reader:
        for name in names:
            blocks, type_name, in_dim = reader.read_routed_expert_blocks(name, args.expert, 0, args.rows)
            if type_name != "iq1_m":
                raise AssertionError(f"{name} type={type_name}, expected iq1_m for this bench")
            x = torch.randn(args.batch, int(in_dim), dtype=torch.bfloat16, device=device)
            blocks_gpu = blocks.to(device=device, non_blocking=True).contiguous()
            grid = _grid_arg(type_name, device)
            weight = reader.read_routed_expert(name, args.expert, 0, args.rows).to(device=device, dtype=torch.float32)
            ref = torch.nn.functional.linear(x.float(), weight, None)
            out = cuda_mod.gguf_quant_gemm_forward(x, blocks_gpu, int(in_dim), _type_id(type_name), grid).float()
            torch.cuda.synchronize(device)
            diff = (out - ref).abs()
            max_abs = float(diff.max().item())
            mean_abs = float(diff.mean().item())
            rel = float((diff / ref.abs().clamp_min(1e-6)).mean().item())
            t0 = time.perf_counter()
            for _ in range(args.loops):
                out = cuda_mod.gguf_quant_gemm_forward(x, blocks_gpu, int(in_dim), _type_id(type_name), grid)
            torch.cuda.synchronize(device)
            print(
                name,
                type_name,
                "in_dim", int(in_dim),
                "rows", args.rows,
                "max_abs", max_abs,
                "mean_abs", mean_abs,
                "mean_rel", rel,
                "cuda_s", (time.perf_counter() - t0) / max(args.loops, 1),
                flush=True,
            )
            if not torch.isfinite(out.float()).all().item():
                raise AssertionError(f"non-finite CUDA output for {name}")
            if max_abs > 0.25 or mean_abs > 0.02:
                raise AssertionError(f"large CUDA/reference diff for {name}: max_abs={max_abs} mean_abs={mean_abs}")


if __name__ == "__main__":
    main()
