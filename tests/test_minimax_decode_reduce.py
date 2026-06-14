from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.models.minimax_m2 import architecture as minimax_arch
from src.models.minimax_m2.architecture import MiniMaxMoE


class _FakeLayer:
    w1 = SimpleNamespace(type_id=99, type_name="iq2_xxs", blocks=torch.zeros(1, 1, 1, 66, dtype=torch.uint8))
    w2 = SimpleNamespace(type_id=99, type_name="iq2_xxs", blocks=torch.zeros(1, 1, 1, 66, dtype=torch.uint8))
    w3 = SimpleNamespace(type_id=99, type_name="iq2_xxs", blocks=torch.zeros(1, 1, 1, 66, dtype=torch.uint8))


class _FakeCache:
    expert_start = 0
    expert_count = 1

    def layer(self, _layer_id: int) -> _FakeLayer:
        return _FakeLayer()

    def _quant_grid(self, _type_name: str) -> torch.Tensor:
        return torch.zeros(256, dtype=torch.int8)


class _FakeCuda:
    def moe_group_routes(self, indices, weights, expert_start: int, expert_count: int):
        device = indices.device
        return (
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.float32),
            torch.zeros(expert_count + 1, device=device, dtype=torch.int32),
        )

    def gguf_moe_single_token_iq2_q2k_forward(self, *args, **kwargs):
        # Return zeros matching expected [1, dim] shape
        dim = args[0].size(1) if len(args) > 0 else 4
        device = args[0].device if len(args) > 0 else "cpu"
        return torch.zeros((1, dim), device=device, dtype=torch.float32)


def _make_fake_moe(dim: int = 4) -> MiniMaxMoE:
    moe = object.__new__(MiniMaxMoE)
    moe.args = SimpleNamespace(dim=dim, top_k=1)
    moe.layer_id = 0
    moe.cache = _FakeCache()
    moe.dtype = torch.float16
    moe._cuda = _FakeCuda()

    def route(x_flat: torch.Tensor):
        rows = x_flat.size(0)
        return (
            torch.zeros((rows, 1), device=x_flat.device, dtype=torch.long),
            torch.ones((rows, 1), device=x_flat.device, dtype=torch.float32),
        )

    moe.route = route
    return moe


@pytest.fixture
def fake_dist(monkeypatch):
    seen: list[torch.dtype] = []

    monkeypatch.setattr(minimax_arch.dist, "is_available", lambda: True)
    monkeypatch.setattr(minimax_arch.dist, "is_initialized", lambda: True)

    def fake_all_reduce(tensor: torch.Tensor, *args, **kwargs):
        seen.append(tensor.dtype)
        return None

    monkeypatch.setattr(minimax_arch.dist, "all_reduce", fake_all_reduce)
    return seen


@pytest.mark.parametrize(
    ("env_value", "expected_dtype"),
    [
        (None, torch.float32),
        ("fp32", torch.float32),
        ("bf16", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
        ("fp16", torch.float16),
        ("float16", torch.float16),
        ("half", torch.float16),
    ],
)
def test_minimax_decode_reduce_dtype_gate(monkeypatch, fake_dist, env_value, expected_dtype):
    if env_value is None:
        monkeypatch.delenv("MINIMAX_M2_DECODE_REDUCE_DTYPE", raising=False)
    else:
        monkeypatch.setenv("MINIMAX_M2_DECODE_REDUCE_DTYPE", env_value)
    monkeypatch.delenv("DEEPSEEK_GGUF_IQ2_XXS_W2_DP4A", raising=False)

    moe = _make_fake_moe()
    x = torch.randn(1, 4, dtype=torch.float16)
    y = moe(x)

    assert fake_dist == [expected_dtype]
    assert y.shape == x.shape
    assert y.dtype == torch.float16


def test_minimax_prefill_reduce_stays_fp32(monkeypatch, fake_dist):
    monkeypatch.setenv("MINIMAX_M2_DECODE_REDUCE_DTYPE", "bf16")
    monkeypatch.delenv("DEEPSEEK_GGUF_IQ2_XXS_W2_DP4A", raising=False)

    moe = _make_fake_moe()
    x = torch.randn(2, 4, dtype=torch.float16)
    y = moe(x)

    assert fake_dist == [torch.float32]
    assert y.shape == x.shape
    assert y.dtype == torch.float16
