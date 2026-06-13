from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.components.gguf.tp_logits import tp_vocab_row_range
from src.kernels.cuda_loader import load_cuda_kernel
from src.models.minimax_m2.gguf_model import MiniMaxM2GGUFModelLoader


REAL_MINIMAX_PATH = Path("/mnt/data1/dsv4_inference/gguf_hfd/MiniMax-M2.7-GGUF/UD-IQ1_M")


def _cuda_q4_q5_available() -> bool:
    if not torch.cuda.is_available():
        return False
    mod = load_cuda_kernel()
    return mod is not None and hasattr(mod, "gguf_quant_gemm_forward") and hasattr(mod, "gguf_quant_embedding_forward")


@pytest.mark.skipif(not (REAL_MINIMAX_PATH.exists() and _cuda_q4_q5_available()), reason="real MiniMax GGUF or CUDA q4/q5 extension not available")
def test_minimax_gguf_model_loader_keeps_dense_weights_as_raw_blocks() -> None:
    loader = MiniMaxM2GGUFModelLoader(
        REAL_MINIMAX_PATH,
        device="cuda:0",
        n_layers=1,
        expert_start=0,
        expert_count=1,
        preload_moe=False,
    )
    try:
        model = loader.load()
    finally:
        loader.close()
    assert model.embedding.tensor.blocks.dtype == torch.uint8
    assert model.embedding.tensor.type_name == "q4_k"
    assert model.lm_head.tensor.type_name == "q4_k"
    assert model.layers[0].attention.q_proj.tensor.type_name == "q5_k"
    assert model.layers[0].attention.k_proj.tensor.type_name == "q5_k"
    assert model.layers[0].attention.v_proj.tensor.type_name == "q5_k"
    assert model.layers[0].attention.o_proj.tensor.type_name == "q5_k"
    assert tuple(model.embedding.tensor.blocks.shape[1:]) == (12, 144)
    assert tuple(model.layers[0].attention.q_proj.tensor.blocks.shape[1:]) == (12, 176)


@pytest.mark.skipif(not (REAL_MINIMAX_PATH.exists() and _cuda_q4_q5_available()), reason="real MiniMax GGUF or CUDA q4/q5 extension not available")
def test_minimax_gguf_one_layer_forward_tp1_smoke() -> None:
    loader = MiniMaxM2GGUFModelLoader(
        REAL_MINIMAX_PATH,
        device="cuda:0",
        n_layers=1,
        expert_start=0,
        expert_count=256,
        preload_moe=True,
    )
    try:
        model = loader.load()
    finally:
        loader.close()
    model.reset_cache(batch_size=1, max_seq_len=4)
    tokens = torch.tensor([[1, 2, 3]], device="cuda:0", dtype=torch.long)
    next_token = model.forward(tokens, 0, return_next_token=True)
    assert next_token.shape == (1,)
    assert bool(torch.isfinite(next_token.float()).all().item())


def test_tp_vocab_row_range_splits_vocab_rows() -> None:
    assert tp_vocab_row_range(16, 4, 0) == (0, 4)
    assert tp_vocab_row_range(16, 4, 3) == (12, 4)
    assert tp_vocab_row_range(10, 4, 0) == (0, 2)
    assert tp_vocab_row_range(10, 4, 1) == (2, 3)
    assert tp_vocab_row_range(10, 4, 2) == (5, 2)
    assert tp_vocab_row_range(10, 4, 3) == (7, 3)
