import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.kernels.cuda_loader import load_cuda_kernel
from src.moe.cpu_backend import _quantize_int8_weight_torch
from tests.bench_moe_prefill_fp4_grouped import _group_routes, _benchmark_cuda
from tests.test_moe_single_token_fp4 import _dequant_fp4, _make_random_fp4
from tests.bench_moe_single_token_fp4 import _make_int8_experts


def _fp4_to_int8_weights(wq: torch.Tensor, ws: torch.Tensor):
    n_experts = wq.shape[0]
    qs = []
    ss = []
    for e in range(n_experts):
        w = _dequant_fp4(wq[e], ws[e]).cpu().contiguous()
        q, s = _quantize_int8_weight_torch(w)
        qs.append(q)
        ss.append(s)
    return torch.stack(qs).cuda(), torch.stack(ss).cuda()


def main():
    device = torch.device("cuda:0")
    ext = load_cuda_kernel()
    dim = 4096
    inter_dim = 2048
    n_experts = 64
    tokens = 64
    topk = 6
    swiglu_limit = 7.0
    g = torch.Generator(device=device).manual_seed(1)
    x = (torch.randn(tokens, dim, generator=g, device=device) * 0.5).to(torch.bfloat16).contiguous()
    indices = torch.randint(0, n_experts, (tokens, topk), generator=g, device=device, dtype=torch.long)
    weights = (torch.rand(tokens, topk, generator=g, device=device) * 0.4 + 0.05).contiguous()
    route_tokens, route_weights, seg_starts = _group_routes(indices, weights, 0, n_experts)

    # Independent int8 baseline (current w8a8 path)
    w1q_i8, w1s_i8 = _make_int8_experts(n_experts, inter_dim, dim, device, 21)
    w3q_i8, w3s_i8 = _make_int8_experts(n_experts, inter_dim, dim, device, 22)
    w2q_i8, w2s_i8 = _make_int8_experts(n_experts, dim, inter_dim, device, 23)

    # Official-FP4 source, converted once to the same [E,N,K] int8+[E,N] scale format.
    w1q_f4, w1s_f4 = _make_random_fp4(n_experts, inter_dim, dim, 21, device)
    w3q_f4, w3s_f4 = _make_random_fp4(n_experts, inter_dim, dim, 22, device)
    w2q_f4, w2s_f4 = _make_random_fp4(n_experts, dim, inter_dim, 23, device)
    print("converting fp4 -> int8 arena on host for upper-bound test...", flush=True)
    w1q_conv, w1s_conv = _fp4_to_int8_weights(w1q_f4, w1s_f4)
    w3q_conv, w3s_conv = _fp4_to_int8_weights(w3q_f4, w3s_f4)
    w2q_conv, w2s_conv = _fp4_to_int8_weights(w2q_f4, w2s_f4)
    torch.cuda.synchronize()

    def run_int8_base():
        ext.moe_prefill_int8_grouped_gemm_forward(
            x, route_tokens, route_weights, seg_starts,
            w1q_i8, w1s_i8, w2q_i8, w2s_i8, w3q_i8, w3s_i8,
            float(swiglu_limit),
        )

    def run_fp4_as_int8():
        ext.moe_prefill_int8_grouped_gemm_forward(
            x, route_tokens, route_weights, seg_starts,
            w1q_conv, w1s_conv, w2q_conv, w2s_conv, w3q_conv, w3s_conv,
            float(swiglu_limit),
        )

    bi, mi = _benchmark_cuda("int8 baseline", run_int8_base, iters=20)
    bf, mf = _benchmark_cuda("fp4->int8", run_fp4_as_int8, iters=20)
    print(f"speedup converted fp4 vs int8: best={bi / bf:.3f}x median={mi / mf:.3f}x")


if __name__ == "__main__":
    main()
