import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.kernels.ops import Packed4BitWeightAlongK
from src.runtime.moe.cpu_backend import _load_native_mod
from src.models.deepseek_v4.runtime import _pack_fp4_weight_rows_for_tile_decode, fp4_block_size


def _raw_to_fp4(raw: torch.Tensor) -> torch.Tensor:
    return raw.view(torch.int8).view(torch.float4_e2m1fn_x2)


def _e8m0_to_f32(scale_u8: torch.Tensor) -> torch.Tensor:
    return scale_u8.view(torch.float8_e8m0fnu).to(torch.float32)


def _make_ptrs(xs: list[torch.Tensor]) -> torch.Tensor:
    ptrs = torch.zeros(len(xs), dtype=torch.long)
    for i, tensor in enumerate(xs):
        ptrs[i] = tensor.data_ptr()
    return ptrs


def main() -> None:
    native = _load_native_mod()
    if native is None or not hasattr(native, "routed_fp4_moe_forward_raw"):
        raise RuntimeError("routed_fp4_moe_forward_raw missing")

    torch.manual_seed(0)
    num_experts = 4
    tokens = 3
    hidden_dim = 64
    inter_dim = 64
    topk = 2
    swiglu_limit = 10.0

    w1_raw = torch.randint(0, 256, (num_experts, inter_dim, hidden_dim // 2), dtype=torch.uint8)
    w2_raw = torch.randint(0, 256, (num_experts, hidden_dim, inter_dim // 2), dtype=torch.uint8)
    w3_raw = torch.randint(0, 256, (num_experts, inter_dim, hidden_dim // 2), dtype=torch.uint8)
    s1_raw = torch.randint(120, 128, (num_experts, inter_dim, hidden_dim // 32), dtype=torch.uint8)
    s2_raw = torch.randint(120, 128, (num_experts, hidden_dim, inter_dim // 32), dtype=torch.uint8)
    s3_raw = torch.randint(120, 128, (num_experts, inter_dim, hidden_dim // 32), dtype=torch.uint8)

    x = torch.randn(tokens, hidden_dim, dtype=torch.float32)
    ids = torch.tensor([[0, 2], [1, 3], [2, 0]], dtype=torch.long)
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
    native.routed_fp4_moe_forward(
        x.data_ptr(), ids.data_ptr(), weights.data_ptr(), out_legacy.data_ptr(),
        tokens, hidden_dim, topk, inter_dim, num_experts, 0, num_experts,
        p_w1.data_ptr(), p_w2.data_ptr(), p_w3.data_ptr(),
        p_s1.data_ptr(), p_s2.data_ptr(), p_s3.data_ptr(),
        swiglu_limit,
    )

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
    native.routed_fp4_moe_forward_raw(
        x.data_ptr(), ids.data_ptr(), weights.data_ptr(), out_raw.data_ptr(),
        tokens, hidden_dim, topk, inter_dim, num_experts, 0, num_experts,
        p_rw1.data_ptr(), p_rw2.data_ptr(), p_rw3.data_ptr(),
        p_rs1.data_ptr(), p_rs2.data_ptr(), p_rs3.data_ptr(),
        swiglu_limit,
    )

    diff = (out_raw - out_legacy).abs()
    ref_abs = out_legacy.abs().mean().item()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    print("max_abs", max_abs)
    print("mean_abs", mean_abs)
    print("legacy_abs_mean", ref_abs)
    assert torch.isfinite(out_raw).all()
    assert max_abs <= max(1e-3, 1e-3 * ref_abs), (max_abs, ref_abs)
    assert mean_abs <= max(1e-4, 1e-4 * ref_abs), (mean_abs, ref_abs)
    print("PASS")


if __name__ == "__main__":
    main()
