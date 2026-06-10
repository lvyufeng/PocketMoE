from __future__ import annotations

from pathlib import Path

import pytest

from src.gguf.bundle import read_gguf_bundle
from src.runtime.minimax_m2_moe import (
    MiniMaxM2RoutedBlockLoader,
    ROUTED_ROLES,
    build_minimax_m2_moe_runtime_plan,
)
from tests.gguf_test_utils import write_minimax_bundle


REAL_MINIMAX_PATH = Path("/mnt/data1/dsv4_inference/gguf_hfd/MiniMax-M2.7-GGUF/UD-IQ1_M")


def test_minimax_moe_runtime_plan_validates_tiny_bundle(tmp_path: Path) -> None:
    # Use inter=256 so IQ2_XXS routed rows have one full 256-element block and
    # can be used by raw block smoke tests without requiring huge fixtures.
    root = write_minimax_bundle(tmp_path / "bundle", n_layers=2, hidden=256, inter=256)
    bundle = read_gguf_bundle(root)

    plan = build_minimax_m2_moe_runtime_plan(bundle)

    assert plan.ok, plan.errors
    assert plan.status == "candidate"
    assert plan.params.n_layers == 2
    assert plan.moe_tensor_count == 12
    assert plan.expected_moe_tensor_count == 12
    assert plan.routed_tensor_count == 6
    assert plan.expected_routed_tensor_count == 6
    assert plan.tensor_role_counts["gate"] == 2
    assert plan.tensor_role_counts["gate_bias"] == 2
    assert plan.tensor_role_counts["ffn_norm"] == 2
    assert plan.routed_type_counts == {"iq2_xxs": 6}
    assert plan.placements[0].name == "all_device_lowbit"
    assert plan.placements[0].status == "candidate"


def test_minimax_moe_runtime_plan_reports_dense_skips(tmp_path: Path) -> None:
    root = write_minimax_bundle(tmp_path / "bundle", n_layers=1, hidden=256, inter=256)
    bundle = read_gguf_bundle(root)
    plan = build_minimax_m2_moe_runtime_plan(bundle)

    skipped = {(item.role, item.type_name): item.reason for item in plan.skipped_tensors}

    assert plan.ok, plan.errors
    assert ("embedding", "q4_k") in skipped
    assert ("lm_head", "q4_k") in skipped
    assert ("attn_q", "q5_k") in skipped
    assert ("attn_k", "q5_k") in skipped
    assert ("attn_v", "q5_k") in skipped
    assert ("attn_o", "q5_k") in skipped
    assert "deferred" in skipped[("embedding", "q4_k")]
    assert "GQA runtime" in skipped[("attn_q", "q5_k")]


def test_minimax_routed_block_loader_reads_small_slice(tmp_path: Path) -> None:
    root = write_minimax_bundle(tmp_path / "bundle", n_layers=1, hidden=256, inter=256)
    bundle = read_gguf_bundle(root)
    plan = build_minimax_m2_moe_runtime_plan(bundle)

    with MiniMaxM2RoutedBlockLoader(bundle, plan=plan) as loader:
        for role in sorted(ROUTED_ROLES):
            blocks, type_name, in_dim = loader.read_expert_role_blocks(0, role, expert=0, row_count=1)
            assert type_name == "iq2_xxs"
            assert in_dim == 256
            assert tuple(blocks.shape) == (1, 1, 66)

            layer_blocks, layer_type_name, layer_in_dim = loader.read_layer_role_blocks(0, role, expert_start=0, expert_count=1)
            assert layer_type_name == "iq2_xxs"
            assert layer_in_dim == in_dim
            assert tuple(layer_blocks.shape) == (1, 256, 1, 66)


@pytest.mark.skipif(not REAL_MINIMAX_PATH.exists(), reason="local MiniMax-M2.7 GGUF bundle not present")
def test_real_minimax_moe_runtime_plan_does_not_read_payload() -> None:
    bundle = read_gguf_bundle(REAL_MINIMAX_PATH)
    plan = build_minimax_m2_moe_runtime_plan(bundle)

    assert plan.ok, plan.errors[:20]
    assert plan.params.n_layers == 62
    assert plan.routed_tensor_count == 62 * 3
    assert plan.expected_routed_tensor_count == 62 * 3
    assert plan.tensor_role_counts["gate"] == 62
    assert plan.tensor_role_counts["gate_bias"] == 62
    assert plan.tensor_role_counts["ffn_norm"] == 62
    assert plan.routed_type_counts == {"iq2_xxs": 62 * 3}
    assert plan.routed_bytes > 50 * 1024**3
    assert any(item.type_name == "q4_k" for item in plan.skipped_tensors)
    assert any(item.type_name == "q5_k" for item in plan.skipped_tensors)


@pytest.mark.skipif(not REAL_MINIMAX_PATH.exists(), reason="local MiniMax-M2.7 GGUF bundle not present")
def test_real_minimax_routed_block_loader_reads_tiny_slice() -> None:
    bundle = read_gguf_bundle(REAL_MINIMAX_PATH)
    plan = build_minimax_m2_moe_runtime_plan(bundle)

    with MiniMaxM2RoutedBlockLoader(bundle, plan=plan) as loader:
        blocks, type_name, in_dim = loader.read_expert_role_blocks(0, "routed_w1", expert=0, row_count=1)

    assert type_name == "iq2_xxs"
    assert in_dim == 3072
    assert tuple(blocks.shape) == (1, 12, 66)
