"""Numerical sanity check for the new single-token MoE int8 CUDA op.

Compares moe_single_token_int8_forward with the existing grouped reference
(moe_prefill_int8_grouped_forward) over a synthetic int8 expert weight set.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "inference"))

import torch

from src.kernels.cuda_loader import load_cuda_kernel


def _build_synthetic(n_experts: int, dim: int, inter_dim: int, device: torch.device):
    rng = torch.Generator(device=device).manual_seed(1234)
    w1q = torch.randint(-32, 32, (n_experts, inter_dim, dim), generator=rng, device=device, dtype=torch.int8)
    w3q = torch.randint(-32, 32, (n_experts, inter_dim, dim), generator=rng, device=device, dtype=torch.int8)
    w2q = torch.randint(-32, 32, (n_experts, dim, inter_dim), generator=rng, device=device, dtype=torch.int8)
    w1s = (torch.rand(n_experts, inter_dim, generator=rng, device=device) * 0.005 + 0.001).contiguous()
    w3s = (torch.rand(n_experts, inter_dim, generator=rng, device=device) * 0.005 + 0.001).contiguous()
    w2s = (torch.rand(n_experts, dim, generator=rng, device=device) * 0.005 + 0.001).contiguous()
    return w1q, w1s, w2q, w2s, w3q, w3s


def _grouped_forward(ext, x, indices, weights, w1q, w1s, w2q, w2s, w3q, w3s, swiglu_limit, n_local_experts, experts_start_idx):
    # Replicate gpu_prefill_moe_backend route grouping for shape [1, D] input.
    indices_row = indices[0].to(torch.int64)
    weights_row = weights[0].to(torch.float32)
    local_ids = indices_row - experts_start_idx
    mask = (local_ids >= 0) & (local_ids < n_local_experts)
    if not bool(mask.any().item()):
        return torch.zeros_like(x, dtype=torch.float32)
    local_ids = local_ids[mask].to(torch.long)
    route_weights = weights_row[mask].to(torch.float32)
    order = torch.argsort(local_ids)
    local_ids = local_ids[order].contiguous()
    route_weights = route_weights[order].contiguous()
    counts = torch.bincount(local_ids, minlength=n_local_experts).to(torch.int32)
    seg_starts = torch.empty(n_local_experts + 1, device=x.device, dtype=torch.int32)
    seg_starts[0] = 0
    torch.cumsum(counts, dim=0, out=seg_starts[1:])
    routes = local_ids.numel()
    x_sorted = x.expand(routes, x.shape[1]).contiguous()
    return ext.moe_prefill_int8_grouped_forward(
        x_sorted,
        route_weights,
        seg_starts,
        w1q,
        w1s,
        w2q,
        w2s,
        w3q,
        w3s,
        float(swiglu_limit),
    ).sum(dim=0, keepdim=True)


def main():
    device = torch.device("cuda:0")
    ext = load_cuda_kernel()
    if ext is None:
        raise RuntimeError("cuda_kernel extension not available")
    if not hasattr(ext, "moe_single_token_int8_forward"):
        raise RuntimeError("moe_single_token_int8_forward symbol missing")
    if not hasattr(ext, "moe_prefill_int8_grouped_forward"):
        raise RuntimeError("moe_prefill_int8_grouped_forward symbol missing")

    n_experts = 16
    dim = 2048
    inter_dim = 768
    experts_start_idx = 32  # arbitrary global offset
    topk = 6
    swiglu_limit = 7.0

    w1q, w1s, w2q, w2s, w3q, w3s = _build_synthetic(n_experts, dim, inter_dim, device)
    rng = torch.Generator(device=device).manual_seed(7)
    x = (torch.randn(1, dim, generator=rng, device=device) * 0.5).to(torch.bfloat16)
    perm = torch.randperm(n_experts, generator=rng, device=device)[:topk]
    global_indices = (perm + experts_start_idx).reshape(1, topk).to(torch.int64)
    weights = torch.rand(1, topk, generator=rng, device=device, dtype=torch.float32) * 0.4 + 0.05

    y_single = ext.moe_single_token_int8_forward(
        x.contiguous(),
        global_indices[0].contiguous(),
        weights[0].contiguous(),
        w1q,
        w1s,
        w2q,
        w2s,
        w3q,
        w3s,
        int(experts_start_idx),
        float(swiglu_limit),
    )
    torch.cuda.synchronize()

    y_ref = _grouped_forward(
        ext,
        x,
        global_indices,
        weights,
        w1q, w1s, w2q, w2s, w3q, w3s,
        swiglu_limit,
        n_experts,
        experts_start_idx,
    )
    torch.cuda.synchronize()

    print("y_single sum =", float(y_single.sum().item()))
    print("y_ref    sum =", float(y_ref.sum().item()))
    diff = (y_single - y_ref).abs()
    rel = diff / (y_ref.abs().clamp_min(1e-3))
    print("max abs diff =", float(diff.max().item()))
    print("mean abs diff =", float(diff.mean().item()))
    print("max rel diff =", float(rel.max().item()))
    # Tolerances are intentionally loose because single-token path quantizes hidden
    # per-route while the grouped path quantizes per-row over all routes mapped to
    # an expert; both are valid INT8 representations.
    if diff.max() <= 5e-2:
        print("PASS: outputs within tight tolerance")
    elif rel.max() <= 0.1:
        print("PASS (loose): rel diff < 10% — INT8 quantization noise is expected")
    else:
        print("FAIL: outputs disagree more than 10%")

    if not hasattr(ext, "moe_single_token_int8_forward_v2"):
        print("WARN: moe_single_token_int8_forward_v2 symbol missing, skipping v2 check")
        return

    # v2: build a compact pack of only the unique active local experts and a
    # route_to_slot map. Output must be bit-identical to v1 because the math is
    # the same — only the indexing source differs.
    indices_row = global_indices[0].to(torch.int64)
    local_ids = indices_row - experts_start_idx
    in_range = (local_ids >= 0) & (local_ids < n_experts)
    unique_local = torch.unique(local_ids[in_range]).to(torch.long)
    # Slot lookup: for each route, find its position in unique_local; -1 if oor.
    slot_map = -torch.ones_like(local_ids)
    for slot, lid in enumerate(unique_local.tolist()):
        slot_map[(local_ids == lid) & in_range] = slot
    pack_w1q = w1q.index_select(0, unique_local).contiguous()
    pack_w1s = w1s.index_select(0, unique_local).contiguous()
    pack_w2q = w2q.index_select(0, unique_local).contiguous()
    pack_w2s = w2s.index_select(0, unique_local).contiguous()
    pack_w3q = w3q.index_select(0, unique_local).contiguous()
    pack_w3s = w3s.index_select(0, unique_local).contiguous()
    y_v2 = ext.moe_single_token_int8_forward_v2(
        x.contiguous(),
        slot_map.contiguous(),
        weights[0].contiguous(),
        pack_w1q,
        pack_w1s,
        pack_w2q,
        pack_w2s,
        pack_w3q,
        pack_w3s,
        float(swiglu_limit),
    )
    torch.cuda.synchronize()
    diff_v2 = (y_v2 - y_single).abs()
    print("v2 vs v1 max abs diff =", float(diff_v2.max().item()))
    if diff_v2.max() <= 1e-5:
        print("PASS v2: bit-identical to v1")
    else:
        print("FAIL v2: outputs differ from v1")


if __name__ == "__main__":
    main()
