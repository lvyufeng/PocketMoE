from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.loader.gguf.bundle import read_gguf_bundle
from src.loader.gguf.tensor_reader import GGUFTensorDataReader
from src.kernels.cuda_loader import load_cuda_kernel
from src.models.minimax_m2.loader import MiniMaxM2GGUFLoader


REAL_MINIMAX_PATH = Path("/mnt/data1/dsv4_inference/gguf_hfd/MiniMax-M2.7-GGUF/UD-IQ1_M")


def _cuda_q4_q5_available() -> bool:
    if not torch.cuda.is_available():
        return False
    mod = load_cuda_kernel()
    return mod is not None and hasattr(mod, "gguf_quant_gemm_forward") and hasattr(mod, "gguf_quant_embedding_forward")


def _read_rows(bundle, name: str, rows: int):
    tensor = bundle.tensors_by_name[name]
    reader = GGUFTensorDataReader(tensor.shard_path)
    try:
        blocks, type_name, row_elems = reader.read_quantized_matrix_block_rows(tensor.name, 0, rows)
        ref = reader.read_quantized_matrix_rows_reference(tensor.name, 0, rows).float()
    finally:
        reader.close()
    return blocks, type_name, row_elems, ref


@pytest.mark.skipif(not (REAL_MINIMAX_PATH.exists() and _cuda_q4_q5_available()), reason="real MiniMax GGUF or CUDA q4/q5 extension not available")
def test_real_minimax_q4_embedding_selected_rows_matches_reference() -> None:
    bundle = read_gguf_bundle(REAL_MINIMAX_PATH)
    blocks, type_name, row_elems, ref = _read_rows(bundle, "token_embd.weight", 8)
    assert type_name == "q4_k"
    mod = load_cuda_kernel()
    empty_grid = torch.empty((0,), device="cuda", dtype=torch.int8)
    ids = torch.tensor([0, 1, 7, 3], device="cuda", dtype=torch.long)
    out = mod.gguf_quant_embedding_forward(ids, blocks.cuda(), row_elems, 3, empty_grid).float()
    err = (out - ref.cuda()[ids]).abs()
    assert bool(torch.isfinite(out).all().item())
    assert float(err.max().item()) < 1.0e-3


@pytest.mark.skipif(not (REAL_MINIMAX_PATH.exists() and _cuda_q4_q5_available()), reason="real MiniMax GGUF or CUDA q4/q5 extension not available")
@pytest.mark.parametrize(
    ("name", "type_id"),
    [
        ("token_embd.weight", 3),
        ("blk.0.attn_q.weight", 4),
    ],
)
def test_real_minimax_q4_q5_gemm_matches_reference(name: str, type_id: int) -> None:
    bundle = read_gguf_bundle(REAL_MINIMAX_PATH)
    blocks, type_name, row_elems, ref = _read_rows(bundle, name, 8)
    assert (type_name, type_id) in {("q4_k", 3), ("q5_k", 4)}
    mod = load_cuda_kernel()
    empty_grid = torch.empty((0,), device="cuda", dtype=torch.int8)
    blocks_cuda = blocks.cuda()
    ref_cuda = ref.cuda()
    x = torch.randn((5, row_elems), device="cuda", dtype=torch.float16)
    expected = x.float() @ ref_cuda.t()
    y = mod.gguf_quant_gemm_forward(x, blocks_cuda, row_elems, type_id, empty_grid).float()
    yp = mod.gguf_quant_gemm_prefill_forward(x, blocks_cuda, row_elems, type_id, empty_grid).float()
    assert bool(torch.isfinite(y).all().item())
    assert bool(torch.isfinite(yp).all().item())
    assert float((y - expected).abs().max().item()) < 2.0e-2
    assert float((yp - expected).abs().max().item()) < 2.0e-2

@pytest.mark.skipif(not (REAL_MINIMAX_PATH.exists() and _cuda_q4_q5_available()), reason="real MiniMax GGUF or CUDA q4/q5 extension not available")
def test_real_minimax_q5_pair_gemm_matches_separate_outputs() -> None:
    bundle = read_gguf_bundle(REAL_MINIMAX_PATH)
    k_blocks, k_type_name, row_elems, _ = _read_rows(bundle, "blk.0.attn_k.weight", 8)
    v_blocks, v_type_name, v_row_elems, _ = _read_rows(bundle, "blk.0.attn_v.weight", 8)
    assert k_type_name == v_type_name == "q5_k"
    assert row_elems == v_row_elems
    mod = load_cuda_kernel()
    empty_grid = torch.empty((0,), device="cuda", dtype=torch.int8)
    x = torch.randn((3, row_elems), device="cuda", dtype=torch.float16)
    k_blocks_cuda = k_blocks.cuda()
    v_blocks_cuda = v_blocks.cuda()
    yk = mod.gguf_quant_gemm_forward(x, k_blocks_cuda, row_elems, 4, empty_grid).float()
    yv = mod.gguf_quant_gemm_forward(x, v_blocks_cuda, row_elems, 4, empty_grid).float()
    pk, pv = mod.gguf_quant_gemm_pair_forward(x, k_blocks_cuda, row_elems, 4, v_blocks_cuda, row_elems, 4, empty_grid)
    assert bool(torch.isfinite(pk).all().item())
    assert bool(torch.isfinite(pv).all().item())
    assert float((pk.float() - yk).abs().max().item()) == 0.0
    assert float((pv.float() - yv).abs().max().item()) == 0.0


@pytest.mark.skipif(not (REAL_MINIMAX_PATH.exists() and _cuda_q4_q5_available()), reason="real MiniMax GGUF or CUDA q4/q5 extension not available")
def test_minimax_loader_keeps_dense_weights_as_raw_blocks() -> None:
    loader = MiniMaxM2GGUFLoader(
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
def test_minimax_one_layer_forward_tp1_smoke() -> None:
    loader = MiniMaxM2GGUFLoader(
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


def test_minimax_loader_rank_row_range_splits_vocab_rows() -> None:
    assert MiniMaxM2GGUFLoader._rank_row_range(16, 4, 0) == (0, 4)
    assert MiniMaxM2GGUFLoader._rank_row_range(16, 4, 3) == (12, 4)
    assert MiniMaxM2GGUFLoader._rank_row_range(10, 4, 0) == (0, 2)
    assert MiniMaxM2GGUFLoader._rank_row_range(10, 4, 1) == (2, 3)
    assert MiniMaxM2GGUFLoader._rank_row_range(10, 4, 2) == (5, 2)
    assert MiniMaxM2GGUFLoader._rank_row_range(10, 4, 3) == (7, 3)
