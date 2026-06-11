import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.deepseek_v4 import runtime as tr
from src.runtime.deepseek_v4.partition import (
    POLICY_BASELINE_4GPU,
    POLICY_LEGACY,
    assert_baseline_compatible_env,
    normalize_policy,
    partition_rule_kind,
    shard_q8_0_blocks_for_rank,
    shard_tensor_for_rank,
)
from src.models.deepseek_v4.runtime import ColumnParallelLinear, ModelArgs, RowParallelLinear, Transformer


class _DummyExpertOwner:
    def __init__(self, start: int, end: int, n_local: int):
        self.experts_start_idx = start
        self.experts_end_idx = end
        self.n_local_experts = n_local


@contextmanager
def _temp_env(**updates):
    old = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _fake_tp4_dist(rank: int):
    old_is_initialized = tr.dist.is_initialized
    old_get_world_size = tr.dist.get_world_size
    old_get_rank = tr.dist.get_rank
    try:
        tr.dist.is_initialized = lambda: True
        tr.dist.get_world_size = lambda: 4
        tr.dist.get_rank = lambda: rank
        yield
    finally:
        tr.dist.is_initialized = old_is_initialized
        tr.dist.get_world_size = old_get_world_size
        tr.dist.get_rank = old_get_rank


def _assert_raises(exc_type, fn, match: str | None = None):
    try:
        fn()
    except exc_type as exc:
        if match is not None and match not in str(exc):
            raise AssertionError(f"expected {match!r} in {exc!r}") from exc
        return exc
    except Exception as exc:
        raise AssertionError(f"expected {exc_type.__name__}, got {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"expected {exc_type.__name__} to be raised")


def _tiny_transformer(rank: int) -> Transformer:
    with _fake_tp4_dist(rank):
        args = ModelArgs(
            dtype="bf16",
            scale_dtype="fp32",
            scale_fmt=None,
            routed_experts_device="cpu",
            partition_policy=POLICY_BASELINE_4GPU,
            vocab_size=128,
            dim=16,
            moe_inter_dim=16,
            n_layers=1,
            n_hash_layers=0,
            n_mtp_layers=0,
            n_heads=4,
            n_routed_experts=8,
            n_shared_experts=1,
            n_activated_experts=2,
            q_lora_rank=8,
            head_dim=8,
            rope_head_dim=4,
            o_groups=4,
            o_lora_rank=8,
            window_size=8,
            compress_ratios=(0, 0, 1, 1, 1, 1, 1, 0),
            index_n_heads=4,
            index_head_dim=4,
            index_topk=4,
            hc_mult=1,
            hc_sinkhorn_iters=2,
            max_batch_size=1,
            max_seq_len=16,
        )
        return Transformer(args)


def test_normalize_policy_defaults_to_legacy():
    assert normalize_policy(None) == POLICY_LEGACY
    assert normalize_policy(POLICY_BASELINE_4GPU) == POLICY_BASELINE_4GPU


def test_baseline_guard_rejects_non_4gpu():
    with _temp_env(
        DEEPSEEK_CPU_MOE_INPROC_SERVER=None,
        DEEPSEEK_CPU_MOE_RANK0_SERVER=None,
        DEEPSEEK_CPU_MOE_EXTERNAL_SERVER=None,
        DEEPSEEK_CPU_MOE_EXTERNAL_PREFILL_LOCAL=None,
        DEEPSEEK_REPLICATED_C4_INDEXER=None,
        DEEPSEEK_ACTIVE_CPU_MOE_INPROC_SERVER=None,
        DEEPSEEK_ACTIVE_CPU_MOE_RANK0_SERVER=None,
        DEEPSEEK_ACTIVE_CPU_MOE_EXTERNAL_SERVER=None,
        DEEPSEEK_ACTIVE_CPU_MOE_EXTERNAL_PREFILL_LOCAL=None,
        DEEPSEEK_ACTIVE_REPLICATED_C4_INDEXER=None,
    ):
        _assert_raises(RuntimeError, lambda: assert_baseline_compatible_env(POLICY_BASELINE_4GPU, 1), "world_size=1")


def test_baseline_guard_rejects_forbidden_env():
    with _temp_env(DEEPSEEK_CPU_MOE_RANK0_SERVER="1"):
        _assert_raises(RuntimeError, lambda: assert_baseline_compatible_env(POLICY_BASELINE_4GPU, 4), "DEEPSEEK_CPU_MOE_RANK0_SERVER")


def test_baseline_guard_honors_active_env_override():
    with _temp_env(DEEPSEEK_ACTIVE_CPU_MOE_INPROC_SERVER="1"):
        _assert_raises(RuntimeError, lambda: assert_baseline_compatible_env(POLICY_BASELINE_4GPU, 4), "DEEPSEEK_CPU_MOE_INPROC_SERVER")


def test_baseline_guard_rejects_remote_external_prefill():
    with _temp_env(DEEPSEEK_CPU_MOE_EXTERNAL_SERVER="1", DEEPSEEK_CPU_MOE_EXTERNAL_PREFILL_LOCAL="0"):
        _assert_raises(
            RuntimeError,
            lambda: assert_baseline_compatible_env(POLICY_BASELINE_4GPU, 4),
            "DEEPSEEK_CPU_MOE_EXTERNAL_SERVER=1 with DEEPSEEK_CPU_MOE_EXTERNAL_PREFILL_LOCAL=0",
        )


def test_expert_owned_tensor_returns_none_for_non_owner():
    owner = _DummyExpertOwner(start=64, end=128, n_local=64)
    tensor = torch.empty(8, 8)
    assert shard_tensor_for_rank("layers.0.ffn.experts.12.w1.weight", tensor, owner, 4, 1) is None
    assert shard_tensor_for_rank("layers.0.ffn.experts.90.w1.weight", tensor, owner, 4, 1) is tensor


def test_partition_rule_kind_marks_shared_expert_as_replicated_baseline():
    with _fake_tp4_dist(0):
        module = ColumnParallelLinear(8, 16, dtype=torch.bfloat16)
        assert partition_rule_kind("layers.0.ffn.shared_experts.w1.weight", object()) == "replicated_shared_expert"
        assert partition_rule_kind("layers.0.attn.wq_b.weight", module) == "column_parallel"


def test_dense_shard_helpers_for_parallel_linears():
    with _fake_tp4_dist(0):
        col = ColumnParallelLinear(8, 16, dtype=torch.bfloat16)
        row = RowParallelLinear(16, 8, dtype=torch.bfloat16)
        col_weight = torch.arange(16 * 8, dtype=torch.float32).view(16, 8)
        row_weight = torch.arange(8 * 16, dtype=torch.float32).view(8, 16)
        assert tuple(shard_tensor_for_rank("x.weight", col_weight, col, 4, 2).shape) == (4, 8)
        assert tuple(shard_tensor_for_rank("x.weight", row_weight, row, 4, 2).shape) == (8, 4)


def test_q8_0_block_shard_helpers_for_parallel_linears():
    with _fake_tp4_dist(0):
        col = ColumnParallelLinear(8, 16, dtype=torch.bfloat16)
        row = RowParallelLinear(16, 8, dtype=torch.bfloat16)
        col_blocks = torch.empty(16, 1, 34, dtype=torch.uint8)
        row_blocks = torch.empty(8, 4, 34, dtype=torch.uint8)
        assert tuple(shard_q8_0_blocks_for_rank("x.weight", col_blocks, col, 4, 2).shape) == (4, 1, 34)
        assert tuple(shard_q8_0_blocks_for_rank("x.weight", row_blocks, row, 4, 2).shape) == (8, 1, 34)


def test_tiny_transformer_baseline_layout_is_contiguous():
    expected = {
        0: ((0, 32), (0, 2)),
        1: ((32, 64), (2, 4)),
        2: ((64, 96), (4, 6)),
        3: ((96, 128), (6, 8)),
    }
    for rank, (vocab_range, expert_range) in expected.items():
        model = _tiny_transformer(rank)
        assert (model.embed.vocab_start_idx, model.embed.vocab_end_idx) == vocab_range
        assert (model.layers[0].ffn.experts_start_idx, model.layers[0].ffn.experts_end_idx) == expert_range
        assert model.partition_policy == POLICY_BASELINE_4GPU


if __name__ == "__main__":
    tests = [
        test_normalize_policy_defaults_to_legacy,
        test_baseline_guard_rejects_non_4gpu,
        test_baseline_guard_rejects_forbidden_env,
        test_baseline_guard_honors_active_env_override,
        test_baseline_guard_rejects_remote_external_prefill,
        test_expert_owned_tensor_returns_none_for_non_owner,
        test_partition_rule_kind_marks_shared_expert_as_replicated_baseline,
        test_dense_shard_helpers_for_parallel_linears,
        test_q8_0_block_shard_helpers_for_parallel_linears,
        test_tiny_transformer_baseline_layout_is_contiguous,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"All {len(tests)} partition policy tests passed")
