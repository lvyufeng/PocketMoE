import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.kernels.cuda_loader import load_cuda_kernel
from tests.test_moe_single_token_fp4 import _dequant_fp4, _make_random_fp4
from tests.bench_moe_single_token_fp4 import _make_int8_experts


def _group_routes(indices: torch.Tensor, weights: torch.Tensor, experts_start_idx: int, n_local_experts: int):
    tokens, topk = indices.shape
    flat_ids = indices.reshape(-1)
    flat_weights = weights.reshape(-1)
    flat_tokens = torch.arange(tokens, device=indices.device, dtype=torch.long).repeat_interleave(topk)
    local_ids = flat_ids - experts_start_idx
    mask = (local_ids >= 0) & (local_ids < n_local_experts)
    local_ids = local_ids[mask].to(torch.long)
    route_weights = flat_weights[mask].to(torch.float32)
    route_tokens = flat_tokens[mask]
    order = torch.argsort(local_ids)
    local_ids = local_ids[order]
    route_weights = route_weights[order].contiguous()
    route_tokens = route_tokens[order].contiguous()
    counts = torch.bincount(local_ids, minlength=n_local_experts).to(torch.int32)
    seg_starts = torch.empty(n_local_experts + 1, device=indices.device, dtype=torch.int32)
    seg_starts[0] = 0
    torch.cumsum(counts, dim=0, out=seg_starts[1:])
    return route_tokens, route_weights, seg_starts


def _reference_forward(x, indices, route_weights, w1q, w1s, w2q, w2s, w3q, w3s, experts_start_idx, swiglu_limit):
    tokens, dim = x.shape
    topk = indices.shape[1]
    y = torch.zeros((tokens, dim), device=x.device, dtype=torch.float32)
    x_f = x.to(torch.float32)
    row_max = x_f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    x_scale = row_max / 127.0
    x_q = torch.clamp(torch.round(x_f / x_scale), -127, 127)
    x_dq = x_q * x_scale
    n_local = w1q.shape[0]
    for t in range(tokens):
        for k in range(topk):
            local = int(indices[t, k].item()) - experts_start_idx
            if local < 0 or local >= n_local:
                continue
            w1 = _dequant_fp4(w1q[local], w1s[local])
            w3 = _dequant_fp4(w3q[local], w3s[local])
            gate = x_dq[t:t + 1] @ w1.T
            up = x_dq[t:t + 1] @ w3.T
            if swiglu_limit > 0.0:
                up = up.clamp(-swiglu_limit, swiglu_limit)
                gate = gate.clamp(max=swiglu_limit)
            hidden = gate / (1.0 + torch.exp(-gate)) * up * route_weights[t, k]
            h_max = hidden.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
            h_scale = h_max / 127.0
            h_q = torch.clamp(torch.round(hidden / h_scale), -127, 127)
            h_dq = h_q * h_scale
            w2 = _dequant_fp4(w2q[local], w2s[local])
            y[t:t + 1] += h_dq @ w2.T
    return y


def _benchmark_cuda(name, fn, *, rounds=7, iters=50, warmup=10):
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
    best = times_sorted[0]
    median = times_sorted[len(times_sorted) // 2]
    print(f"{name:>14s}: best={best:9.2f} us  median={median:9.2f} us  all={[round(t, 2) for t in times]}")
    return best, median


def main():
    device = torch.device("cuda:0")
    ext = load_cuda_kernel()
    if ext is None:
        raise RuntimeError("cuda_kernel extension missing")
    if not hasattr(ext, "moe_prefill_fp4_grouped_gemm_forward"):
        raise RuntimeError("moe_prefill_fp4_grouped_gemm_forward missing")

    dim = 1024
    inter_dim = 512
    n_experts = 8
    tokens = 32
    topk = 4
    experts_start_idx = 0
    swiglu_limit = 7.0

    g = torch.Generator(device=device).manual_seed(123)
    x = (torch.randn(tokens, dim, generator=g, device=device) * 0.5).to(torch.bfloat16).contiguous()
    indices = torch.randint(0, n_experts, (tokens, topk), generator=g, device=device, dtype=torch.long)
    weights = (torch.rand(tokens, topk, generator=g, device=device) * 0.4 + 0.05).contiguous()
    route_tokens, route_weights, seg_starts = _group_routes(indices, weights, experts_start_idx, n_experts)

    w1q_f4, w1s_f4 = _make_random_fp4(n_experts, inter_dim, dim, 11, device)
    w3q_f4, w3s_f4 = _make_random_fp4(n_experts, inter_dim, dim, 12, device)
    w2q_f4, w2s_f4 = _make_random_fp4(n_experts, dim, inter_dim, 13, device)

    y_kernel = ext.moe_prefill_fp4_grouped_gemm_forward(
        x, route_tokens, route_weights, seg_starts,
        w1q_f4.contiguous(), w1s_f4.contiguous(),
        w2q_f4.contiguous(), w2s_f4.contiguous(),
        w3q_f4.contiguous(), w3s_f4.contiguous(),
        float(swiglu_limit),
    )
    y_ref = torch.cat([
        ext.moe_single_token_fp4_forward(
            x[t:t + 1].contiguous(),
            indices[t].contiguous(),
            weights[t].contiguous(),
            w1q_f4.contiguous(), w1s_f4.contiguous(),
            w2q_f4.contiguous(), w2s_f4.contiguous(),
            w3q_f4.contiguous(), w3s_f4.contiguous(),
            int(experts_start_idx),
            float(swiglu_limit),
        )
        for t in range(tokens)
    ], dim=0)
    torch.cuda.synchronize()
    diff = (y_kernel - y_ref).abs()
    rel = diff / y_ref.abs().clamp_min(1e-2)
    print("max_abs", float(diff.max().item()))
    print("mean_abs", float(diff.mean().item()))
    print("max_rel", float(rel.max().item()))
    assert float(rel.max().item()) < 2e-3

    # DS4-ish performance shape.
    dim = 4096
    inter_dim = 2048
    n_experts = 64
    tokens = 64
    topk = 6
    g = torch.Generator(device=device).manual_seed(1)
    x = (torch.randn(tokens, dim, generator=g, device=device) * 0.5).to(torch.bfloat16).contiguous()
    indices = torch.randint(0, n_experts, (tokens, topk), generator=g, device=device, dtype=torch.long)
    weights = (torch.rand(tokens, topk, generator=g, device=device) * 0.4 + 0.05).contiguous()
    route_tokens, route_weights, seg_starts = _group_routes(indices, weights, 0, n_experts)
    w1q_i8, w1s_i8 = _make_int8_experts(n_experts, inter_dim, dim, device, 21)
    w3q_i8, w3s_i8 = _make_int8_experts(n_experts, inter_dim, dim, device, 22)
    w2q_i8, w2s_i8 = _make_int8_experts(n_experts, dim, inter_dim, device, 23)
    w1q_f4, w1s_f4 = _make_random_fp4(n_experts, inter_dim, dim, 21, device)
    w3q_f4, w3s_f4 = _make_random_fp4(n_experts, inter_dim, dim, 22, device)
    w2q_f4, w2s_f4 = _make_random_fp4(n_experts, dim, inter_dim, 23, device)

    def run_int8():
        ext.moe_prefill_int8_grouped_gemm_forward(
            x, route_tokens, route_weights, seg_starts,
            w1q_i8, w1s_i8, w2q_i8, w2s_i8, w3q_i8, w3s_i8,
            float(swiglu_limit),
        )

    def run_fp4():
        ext.moe_prefill_fp4_grouped_gemm_forward(
            x, route_tokens, route_weights, seg_starts,
            w1q_f4, w1s_f4, w2q_f4, w2s_f4, w3q_f4, w3s_f4,
            float(swiglu_limit),
        )

    bi, mi = _benchmark_cuda("int8 grouped", run_int8, iters=20)
    bf, mf = _benchmark_cuda("fp4 grouped", run_fp4, iters=20)
    print(f"speedup fp4 vs int8: best={bi / bf:.3f}x median={mi / mf:.3f}x")


if __name__ == "__main__":
    main()
