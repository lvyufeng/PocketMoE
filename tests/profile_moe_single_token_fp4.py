import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.kernels.cuda_loader import load_cuda_kernel
from tests.bench_moe_single_token_fp4 import _make_int8_experts, _make_fp4_experts


def main():
    device = torch.device("cuda:0")
    ext = load_cuda_kernel()
    dim = 4096
    inter_dim = 2048
    n_local_experts = 64
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
        ext.moe_single_token_int8_forward(x, indices, weights, w1q_i8, w1s_i8, w2q_i8, w2s_i8, w3q_i8, w3s_i8, 0, swiglu_limit)

    def run_fp4():
        ext.moe_single_token_fp4_forward(x, indices, weights, w1q_f4, w1s_f4, w2q_f4, w2s_f4, w3q_f4, w3s_f4, 0, swiglu_limit)

    for _ in range(20):
        run_int8(); run_fp4()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(50):
            run_int8()
        for _ in range(50):
            run_fp4()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))


if __name__ == "__main__":
    main()
