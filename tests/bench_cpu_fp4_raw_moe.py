import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.kernels.ops import Packed4BitWeightAlongK
from src.moe.cpu_backend import _load_native_mod
from src.runtime.transformer import _pack_fp4_weight_rows_for_tile_decode, fp4_block_size


def _raw_to_fp4(raw: torch.Tensor) -> torch.Tensor:
    return raw.view(torch.int8).view(torch.float4_e2m1fn_x2)


def _e8m0_to_f32(scale_u8: torch.Tensor) -> torch.Tensor:
    return scale_u8.view(torch.float8_e8m0fnu).to(torch.float32)


def _make_ptrs(xs: list[torch.Tensor]) -> torch.Tensor:
    ptrs = torch.zeros(len(xs), dtype=torch.long)
    for i, tensor in enumerate(xs):
        ptrs[i] = tensor.data_ptr()
    return ptrs


def _benchmark_cpu(name, fn, *, rounds=7, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        times.append((time.perf_counter() - t0) * 1e6 / iters)
    times_sorted = sorted(times)
    best = times_sorted[0]
    median = times_sorted[len(times_sorted) // 2]
    print(f"{name:>12s}: best={best:9.2f} us  median={median:9.2f} us  all={[round(t, 2) for t in times]}")
    return best, median


def main() -> None:
    native = _load_native_mod()
    if native is None or not hasattr(native, "routed_fp4_moe_forward_raw"):
        raise RuntimeError("routed_fp4_moe_forward_raw missing")

    torch.manual_seed(0)
    num_experts = 64
    tokens = 64
    hidden_dim = 4096
    inter_dim = 2048
    topk = 6
    swiglu_limit = 10.0

    w1_raw = torch.randint(0, 256, (num_experts, inter_dim, hidden_dim // 2), dtype=torch.uint8)
    w2_raw = torch.randint(0, 256, (num_experts, hidden_dim, inter_dim // 2), dtype=torch.uint8)
    w3_raw = torch.randint(0, 256, (num_experts, inter_dim, hidden_dim // 2), dtype=torch.uint8)
    s1_raw = torch.randint(120, 128, (num_experts, inter_dim, hidden_dim // 32), dtype=torch.uint8)
    s2_raw = torch.randint(120, 128, (num_experts, hidden_dim, inter_dim // 32), dtype=torch.uint8)
    s3_raw = torch.randint(120, 128, (num_experts, inter_dim, hidden_dim // 32), dtype=torch.uint8)

    x = torch.randn(tokens, hidden_dim, dtype=torch.float32)
    ids = torch.randint(0, num_experts, (tokens, topk), dtype=torch.long)
    weights = torch.rand(tokens, topk, dtype=torch.float32)
    out_legacy = torch.empty(tokens, hidden_dim, dtype=torch.float32)
    out_raw = torch.empty_like(out_legacy)

    w1_legacy = []
    w2_legacy = []
    w3_legacy = []
    s1_legacy = []
    s2_legacy = []
    s3_legacy = []
    for expert_idx in range(num_experts):
        w1_packed = Packed4BitWeightAlongK.convert_from(_raw_to_fp4(w1_raw[expert_idx]))
        w2_packed = Packed4BitWeightAlongK.convert_from(_raw_to_fp4(w2_raw[expert_idx]))
        w3_packed = Packed4BitWeightAlongK.convert_from(_raw_to_fp4(w3_raw[expert_idx]))
        s1 = _e8m0_to_f32(s1_raw[expert_idx]).contiguous()
        s2 = _e8m0_to_f32(s2_raw[expert_idx]).contiguous()
        s3 = _e8m0_to_f32(s3_raw[expert_idx]).contiguous()
        w2_tiled, s2_tiled = _pack_fp4_weight_rows_for_tile_decode(
            w2_packed,
            s2,
            tile_rows=64,
            block_size=fp4_block_size,
        )
        w1_legacy.append(w1_packed.layout_tensor.contiguous())
        w2_legacy.append(w2_tiled.contiguous())
        w3_legacy.append(w3_packed.layout_tensor.contiguous())
        s1_legacy.append(s1)
        s2_legacy.append(s2_tiled.contiguous())
        s3_legacy.append(s3)

    p_w1 = _make_ptrs(w1_legacy)
    p_w2 = _make_ptrs(w2_legacy)
    p_w3 = _make_ptrs(w3_legacy)
    p_s1 = _make_ptrs(s1_legacy)
    p_s2 = _make_ptrs(s2_legacy)
    p_s3 = _make_ptrs(s3_legacy)

    raw_w1 = [w1_raw[i] for i in range(num_experts)]
    raw_w2 = [w2_raw[i] for i in range(num_experts)]
    raw_w3 = [w3_raw[i] for i in range(num_experts)]
    raw_s1 = [s1_raw[i] for i in range(num_experts)]
    raw_s2 = [s2_raw[i] for i in range(num_experts)]
    raw_s3 = [s3_raw[i] for i in range(num_experts)]
    p_rw1 = _make_ptrs(raw_w1)
    p_rw2 = _make_ptrs(raw_w2)
    p_rw3 = _make_ptrs(raw_w3)
    p_rs1 = _make_ptrs(raw_s1)
    p_rs2 = _make_ptrs(raw_s2)
    p_rs3 = _make_ptrs(raw_s3)

    def run_legacy():
        native.routed_fp4_moe_forward(
            x.data_ptr(), ids.data_ptr(), weights.data_ptr(), out_legacy.data_ptr(),
            tokens, hidden_dim, topk, inter_dim, num_experts, 0, num_experts,
            p_w1.data_ptr(), p_w2.data_ptr(), p_w3.data_ptr(),
            p_s1.data_ptr(), p_s2.data_ptr(), p_s3.data_ptr(),
            swiglu_limit,
        )

    def run_raw():
        native.routed_fp4_moe_forward_raw(
            x.data_ptr(), ids.data_ptr(), weights.data_ptr(), out_raw.data_ptr(),
            tokens, hidden_dim, topk, inter_dim, num_experts, 0, num_experts,
            p_rw1.data_ptr(), p_rw2.data_ptr(), p_rw3.data_ptr(),
            p_rs1.data_ptr(), p_rs2.data_ptr(), p_rs3.data_ptr(),
            swiglu_limit,
        )

    run_legacy()
    run_raw()
    diff = (out_raw - out_legacy).abs()
    print(f"max_abs={float(diff.max().item())} mean_abs={float(diff.mean().item())}")

    best_legacy, med_legacy = _benchmark_cpu("legacy fp4", run_legacy)
    best_raw, med_raw = _benchmark_cpu("raw fp4", run_raw)
    print(f"speedup raw vs legacy: best={best_legacy / best_raw:.3f}x median={med_legacy / med_raw:.3f}x")


if __name__ == "__main__":
    main()
