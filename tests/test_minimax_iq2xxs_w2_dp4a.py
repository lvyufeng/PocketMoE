"""Test iq2_xxs w2 DP4A grouped prefill kernel numerical correctness vs float baseline."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from src.loader.gguf.bundle import read_gguf_bundle
from src.models.minimax_m2.moe_runtime import MiniMaxM2DeviceResidentCache
from src.models.minimax_m2.moe_planning import build_minimax_m2_moe_runtime_plan


REAL_MINIMAX_PATH = Path("/mnt/data1/dsv4_inference/gguf_hfd/MiniMax-M2.7-GGUF/UD-IQ1_M")


def _cuda_gguf_ext_available() -> bool:
    if not torch.cuda.is_available():
        return False
    from src.kernels.cuda_loader import load_cuda_kernel

    cuda_mod = load_cuda_kernel()
    return cuda_mod is not None and hasattr(cuda_mod, "gguf_moe_prefill_grouped_forward")


@pytest.mark.skipif(not (REAL_MINIMAX_PATH.exists() and _cuda_gguf_ext_available()), reason="real MiniMax-M2.7 GGUF bundle or CUDA extension not available")
def test_iq2_xxs_w2_dp4a_vs_float_small_fixture() -> None:
    """Numerical correctness of iq2_xxs w2 DP4A grouped kernel vs float baseline."""
    from src.kernels.cuda_loader import load_cuda_kernel

    cuda_mod = load_cuda_kernel()
    bundle = read_gguf_bundle(REAL_MINIMAX_PATH)
    plan = build_minimax_m2_moe_runtime_plan(bundle)
    assert plan.ok, plan.errors[:5]

    # Use first 2 experts from layer 0 as test slice
    device = torch.device("cuda:0")
    with MiniMaxM2DeviceResidentCache(bundle, plan=plan, device=device, expert_start=0, expert_count=2) as cache:
        layer = cache.layer(0)
        dim = int(layer.w1.in_dim)  # 3072
        inter_dim = int(layer.w1.out_dim)  # 1536
        n_experts = int(layer.w1.expert_count)  # 2
        assert layer.w1.type_name == "iq2_xxs"
        assert layer.w2.type_name == "iq2_xxs"

        # Synthetic test case: 4 tokens, each routes top-2, all 8 routes hit our 2 experts
        tokens = 4
        topk = 2
        # Route pattern: token0->[expert0,expert1], token1->[expert1,expert0], ...
        # to exercise both experts in mixed order
        indices = torch.tensor([
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
        ], device=device, dtype=torch.long)
        weights = torch.rand(tokens, topk, device=device, dtype=torch.float32)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Group routes
        grouped = cuda_mod.moe_group_routes(indices, weights, 0, n_experts)
        _local_ids, route_tokens, route_weights, seg_starts = grouped
        routes = int(route_tokens.numel())
        assert routes == tokens * topk

        # Input: random float16
        torch.manual_seed(1234)
        x = torch.randn(tokens, dim, device=device, dtype=torch.float16)
        grid = cache._quant_grid("iq2_xxs")

        # --- Float baseline: env var OFF ---
        env_backup = os.environ.get("DEEPSEEK_GGUF_IQ2_XXS_W2_DP4A")
        os.environ.pop("DEEPSEEK_GGUF_IQ2_XXS_W2_DP4A", None)
        y_float = cuda_mod.gguf_moe_prefill_grouped_forward(
            x,
            route_tokens,
            route_weights,
            seg_starts,
            layer.w1.blocks,
            layer.w3.blocks,
            layer.w2.blocks,
            int(dim),
            int(layer.w1.type_id),
            int(dim),
            int(layer.w3.type_id),
            int(inter_dim),
            int(layer.w2.type_id),
            grid,
            0.0,
        )

        # --- iq2_xxs w2 DP4A: env var ON ---
        os.environ["DEEPSEEK_GGUF_IQ2_XXS_W2_DP4A"] = "1"
        y_dp4a = cuda_mod.gguf_moe_prefill_grouped_forward(
            x,
            route_tokens,
            route_weights,
            seg_starts,
            layer.w1.blocks,
            layer.w3.blocks,
            layer.w2.blocks,
            int(dim),
            int(layer.w1.type_id),
            int(dim),
            int(layer.w3.type_id),
            int(inter_dim),
            int(layer.w2.type_id),
            grid,
            0.0,
        )

        # Restore env
        if env_backup is not None:
            os.environ["DEEPSEEK_GGUF_IQ2_XXS_W2_DP4A"] = env_backup
        else:
            os.environ.pop("DEEPSEEK_GGUF_IQ2_XXS_W2_DP4A", None)

        # --- Compare ---
        assert y_float.shape == y_dp4a.shape == (tokens, dim)
        assert y_float.dtype == y_dp4a.dtype == torch.float32

        # int8 activation quantization + DP4A accumulation introduces small
        # approximation error versus the float hidden baseline.  Avoid max-relative
        # on near-zero outputs; gate on absolute error and p99 relative error.
        abs_diff = (y_dp4a - y_float).abs()
        max_abs = float(abs_diff.max().item())
        mean_abs = float(abs_diff.mean().item())
        mask = y_float.abs() > 1e-2
        rel_err = abs_diff[mask] / (y_float[mask].abs() + 1e-8)
        p99_rel = float(torch.quantile(rel_err, 0.99).item()) if rel_err.numel() else 0.0
        mean_rel = float(rel_err.mean().item()) if rel_err.numel() else 0.0

        # Sanity: outputs are finite and not all-zero
        assert torch.isfinite(y_float).all(), "float baseline has non-finite values"
        assert torch.isfinite(y_dp4a).all(), "DP4A output has non-finite values"
        assert y_float.abs().sum() > 0.0, "float baseline is all-zero"
        assert y_dp4a.abs().sum() > 0.0, "DP4A output is all-zero"

        assert max_abs < 0.35, f"max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}, p99_rel={p99_rel:.4e}, mean_rel={mean_rel:.4e}"
        assert mean_abs < 2.0e-2, f"max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}, p99_rel={p99_rel:.4e}, mean_rel={mean_rel:.4e}"
        assert p99_rel < 0.30, f"max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}, p99_rel={p99_rel:.4e}, mean_rel={mean_rel:.4e}"

        print(
            "✓ iq2_xxs w2 DP4A numerical match: "
            f"max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}, p99_rel={p99_rel:.4e}, mean_rel={mean_rel:.4e}"
        )


@pytest.mark.skipif(not (REAL_MINIMAX_PATH.exists() and _cuda_gguf_ext_available()), reason="real MiniMax-M2.7 GGUF bundle or CUDA extension not available")
def test_iq2_xxs_w2_dp4a_env_gate() -> None:
    """Verify DEEPSEEK_GGUF_IQ2_XXS_W2_DP4A env gate toggles the DP4A branch."""
    from src.kernels.cuda_loader import load_cuda_kernel

    cuda_mod = load_cuda_kernel()
    bundle = read_gguf_bundle(REAL_MINIMAX_PATH)
    plan = build_minimax_m2_moe_runtime_plan(bundle)
    device = torch.device("cuda:0")

    with MiniMaxM2DeviceResidentCache(bundle, plan=plan, device=device, expert_start=0, expert_count=1) as cache:
        layer = cache.layer(0)
        dim = int(layer.w1.in_dim)
        inter_dim = int(layer.w1.out_dim)

        tokens = 2
        indices = torch.tensor([[0], [0]], device=device, dtype=torch.long)
        weights = torch.ones(tokens, 1, device=device, dtype=torch.float32)
        grouped = cuda_mod.moe_group_routes(indices, weights, 0, 1)
        _local_ids, route_tokens, route_weights, seg_starts = grouped

        x = torch.randn(tokens, dim, device=device, dtype=torch.float16)
        grid = cache._quant_grid("iq2_xxs")

        # OFF
        env_backup = os.environ.get("DEEPSEEK_GGUF_IQ2_XXS_W2_DP4A")
        os.environ.pop("DEEPSEEK_GGUF_IQ2_XXS_W2_DP4A", None)
        y_off = cuda_mod.gguf_moe_prefill_grouped_forward(
            x, route_tokens, route_weights, seg_starts,
            layer.w1.blocks, layer.w3.blocks, layer.w2.blocks,
            int(dim), int(layer.w1.type_id), int(dim), int(layer.w3.type_id),
            int(inter_dim), int(layer.w2.type_id), grid, 0.0)

        # ON
        os.environ["DEEPSEEK_GGUF_IQ2_XXS_W2_DP4A"] = "1"
        y_on = cuda_mod.gguf_moe_prefill_grouped_forward(
            x, route_tokens, route_weights, seg_starts,
            layer.w1.blocks, layer.w3.blocks, layer.w2.blocks,
            int(dim), int(layer.w1.type_id), int(dim), int(layer.w3.type_id),
            int(inter_dim), int(layer.w2.type_id), grid, 0.0)

        if env_backup is not None:
            os.environ["DEEPSEEK_GGUF_IQ2_XXS_W2_DP4A"] = env_backup
        else:
            os.environ.pop("DEEPSEEK_GGUF_IQ2_XXS_W2_DP4A", None)

        # OFF = float baseline, ON = DP4A; outputs differ but both finite
        assert torch.isfinite(y_off).all()
        assert torch.isfinite(y_on).all()
        # They should differ (DP4A quantization path vs float)
        # but be close enough (already covered by numerical test)
        diff_norm = (y_on - y_off).norm().item()
        assert diff_norm > 0.0, "DP4A and float baseline are identical (env gate not working)"


@pytest.mark.skipif(not (REAL_MINIMAX_PATH.exists() and _cuda_gguf_ext_available()), reason="real MiniMax-M2.7 GGUF bundle or CUDA extension not available")
def test_iq2_xxs_single_token_decode_dp4a_vs_grouped_float_baseline() -> None:
    """Single-token iq2_xxs w13+w2 DP4A decode path matches grouped float baseline."""
    from src.kernels.cuda_loader import load_cuda_kernel

    cuda_mod = load_cuda_kernel()
    bundle = read_gguf_bundle(REAL_MINIMAX_PATH)
    plan = build_minimax_m2_moe_runtime_plan(bundle)
    device = torch.device("cuda:0")

    with MiniMaxM2DeviceResidentCache(bundle, plan=plan, device=device, expert_start=0, expert_count=2) as cache:
        layer = cache.layer(0)
        dim = int(layer.w1.in_dim)
        inter_dim = int(layer.w1.out_dim)

        # Single decode token routes to both local experts.
        indices = torch.tensor([[0, 1]], device=device, dtype=torch.long)
        weights = torch.tensor([[0.625, 0.375]], device=device, dtype=torch.float32)
        grouped = cuda_mod.moe_group_routes(indices, weights, 0, 2)
        _local_ids, route_tokens, route_weights, seg_starts = grouped

        torch.manual_seed(5678)
        x = torch.randn(1, dim, device=device, dtype=torch.float16)
        grid = cache._quant_grid("iq2_xxs")

        env_w13 = os.environ.get("DEEPSEEK_GGUF_IQ2_XXS_W13_DP4A")
        env_w2 = os.environ.get("DEEPSEEK_GGUF_IQ2_XXS_W2_DP4A")
        try:
            # Grouped float baseline: both env gates OFF.
            os.environ.pop("DEEPSEEK_GGUF_IQ2_XXS_W13_DP4A", None)
            os.environ.pop("DEEPSEEK_GGUF_IQ2_XXS_W2_DP4A", None)
            y_float = cuda_mod.gguf_moe_prefill_grouped_forward(
                x,
                route_tokens,
                route_weights,
                seg_starts,
                layer.w1.blocks,
                layer.w3.blocks,
                layer.w2.blocks,
                int(dim),
                int(layer.w1.type_id),
                int(dim),
                int(layer.w3.type_id),
                int(inter_dim),
                int(layer.w2.type_id),
                grid,
                0.0,
            )

            # Single-token decode path: iq2_xxs w13+w2 DP4A.
            os.environ["DEEPSEEK_GGUF_IQ2_XXS_W2_DP4A"] = "1"
            route_slots = indices.reshape(-1).contiguous()
            route_weights_flat = weights.reshape(-1).contiguous()
            y_decode = cuda_mod.gguf_moe_single_token_iq2_q2k_forward(
                x,
                route_slots,
                route_weights_flat,
                layer.w1.blocks,
                layer.w3.blocks,
                layer.w2.blocks,
                grid,
                0.0,
            )
        finally:
            if env_w13 is None:
                os.environ.pop("DEEPSEEK_GGUF_IQ2_XXS_W13_DP4A", None)
            else:
                os.environ["DEEPSEEK_GGUF_IQ2_XXS_W13_DP4A"] = env_w13
            if env_w2 is None:
                os.environ.pop("DEEPSEEK_GGUF_IQ2_XXS_W2_DP4A", None)
            else:
                os.environ["DEEPSEEK_GGUF_IQ2_XXS_W2_DP4A"] = env_w2

        assert y_float.shape == y_decode.shape == (1, dim)
        assert torch.isfinite(y_float).all()
        assert torch.isfinite(y_decode).all()

        diff = (y_decode - y_float).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        mask = y_float.abs() > 1e-2
        rel_err = diff[mask] / (y_float[mask].abs() + 1e-8)
        p99_rel = float(torch.quantile(rel_err, 0.99).item()) if rel_err.numel() else 0.0
        mean_rel = float(rel_err.mean().item()) if rel_err.numel() else 0.0

        # This includes both w13 input q8 quantization and w2 hidden q8 quantization.
        assert max_abs < 0.75, f"max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}, p99_rel={p99_rel:.4e}, mean_rel={mean_rel:.4e}"
        assert mean_abs < 5.0e-2, f"max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}, p99_rel={p99_rel:.4e}, mean_rel={mean_rel:.4e}"
        assert p99_rel < 0.50, f"max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}, p99_rel={p99_rel:.4e}, mean_rel={mean_rel:.4e}"
        print(
            "✓ iq2_xxs single-token decode DP4A match: "
            f"max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}, p99_rel={p99_rel:.4e}, mean_rel={mean_rel:.4e}"
        )
