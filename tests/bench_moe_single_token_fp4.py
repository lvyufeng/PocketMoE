import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.kernels.cuda_loader import load_cuda_kernel


def _make_int8_experts(n_experts, out_dim, in_dim, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    wq = torch.randint(-32, 32, (n_experts, out_dim, in_dim), generator=g, device=device, dtype=torch.int8).contiguous()
    ws = (torch.rand(n_experts, out_dim, generator=g, device=device) * 0.005 + 0.001).contiguous()
    return wq, ws


def _make_fp4_experts(n_experts, out_dim, in_dim, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    wq = torch.randint(0, 256, (n_experts, out_dim, in_dim // 2), generator=g, device=device, dtype=torch.uint8).contiguous()
    ws = torch.randint(124, 131, (n_experts, out_dim, in_dim // 32), generator=g, device=device, dtype=torch.uint8).contiguous()
    return wq, ws


def _benchmark_cuda(name, fn, *, rounds=7, iters=400, warmup=80):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / iters)
    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    best = times_sorted[0]
    print(f"{name:>10s}: best={best:8.2f} us  median={median:8.2f} us  all={[round(t, 2) for t in times]}")
    return best, median


def main():
    device = torch.device("cuda:0")
    ext = load_cuda_kernel()
    if ext is None:
        raise RuntimeError("cuda_kernel ext missing")
    if not hasattr(ext, "moe_single_token_int8_forward"):
        raise RuntimeError("missing int8 v1")
    if not hasattr(ext, "moe_single_token_fp4_forward"):
        raise RuntimeError("missing fp4")

    dim = 4096
    inter_dim = 2048
    n_local_experts = 64
    experts_start_idx = 0
    topk = 6
    swiglu_limit = 10.0

    g = torch.Generator(device=device).manual_seed(1)
    x = (torch.randn(1, dim, generator=g, device=device) * 0.5).to(torch.bfloat16).contiguous()
    indices = torch.randperm(n_local_experts, generator=g, device=device)[:topk].to(torch.int64).contiguous()
    weights = (torch.rand(topk, generator=g, device=device) * 0.4 + 0.05).contiguous()

    w1q_i8, w1s_i8 = _make_int8_experts(n_local_experts, inter_dim, dim, device, 11)
    w3q_i8, w3s_i8 = _make_int8_experts(n_local_experts, inter_dim, dim, device, 12)
    w2q_i8, w2s_i8 = _make_int8_experts(n_local_experts, dim, inter_dim, device, 13)

    w1q_f4, w1s_f4 = _make_fp4_experts(n_local_experts, inter_dim, dim, device, 11)
    w3q_f4, w3s_f4 = _make_fp4_experts(n_local_experts, inter_dim, dim, device, 12)
    w2q_f4, w2s_f4 = _make_fp4_experts(n_local_experts, dim, inter_dim, device, 13)

    def run_int8():
        ext.moe_single_token_int8_forward(
            x, indices, weights,
            w1q_i8, w1s_i8, w2q_i8, w2s_i8, w3q_i8, w3s_i8,
            int(experts_start_idx), float(swiglu_limit),
        )

    def run_fp4():
        ext.moe_single_token_fp4_forward(
            x, indices, weights,
            w1q_f4, w1s_f4, w2q_f4, w2s_f4, w3q_f4, w3s_f4,
            int(experts_start_idx), float(swiglu_limit),
        )

    bytes_int8 = (
        w1q_i8.numel() + w2q_i8.numel() + w3q_i8.numel() +
        w1s_i8.numel() * 4 + w2s_i8.numel() * 4 + w3s_i8.numel() * 4
    )
    bytes_fp4 = (
        w1q_f4.numel() + w2q_f4.numel() + w3q_f4.numel() +
        w1s_f4.numel() + w2s_f4.numel() + w3s_f4.numel()
    )
    print(f"int8 weight bytes total : {bytes_int8 / 1024 / 1024:.2f} MiB")
    print(f"fp4  weight bytes total : {bytes_fp4 / 1024 / 1024:.2f} MiB  (ratio {bytes_fp4 / bytes_int8:.3f})")

    best_i8, med_i8 = _benchmark_cuda("int8 v1", run_int8)
    best_f4, med_f4 = _benchmark_cuda("fp4", run_fp4)
    print(f"speedup fp4 vs int8: best={best_i8 / best_f4:.3f}x median={med_i8 / med_f4:.3f}x")


if __name__ == "__main__":
    main()
