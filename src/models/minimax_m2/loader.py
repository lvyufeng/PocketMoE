from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch

from src.loader.gguf.bundle import GGUFBundle, GGUFTensorRef, read_gguf_bundle
from src.loader.gguf.tensor_reader import GGUFTensorDataReader
from src.models.minimax_m2.moe_runtime import MiniMaxM2DeviceResidentCache
from src.models.minimax_m2.runtime import (
    GGUF_DENSE_TYPE_IDS,
    MiniMaxAttention,
    MiniMaxBlock,
    MiniMaxM2Args,
    MiniMaxMoE,
    MiniMaxTransformer,
    QuantizedGGUFEmbedding,
    QuantizedGGUFLinear,
    QuantizedGGUFTensor,
)


class MiniMaxM2GGUFLoader:
    """Load MiniMax-M2 GGUF tensors for the raw-block CUDA runtime.

    Dense q4_k/q5_k tensors are copied as uint8 GGUF blocks.  This loader never
    expands q4_k/q5_k into resident fp16/fp32 matrices; CPU dequant remains only
    a test/reference helper in tensor_reader.py.
    """

    def __init__(
        self,
        bundle_or_path: GGUFBundle | str | Path,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        n_layers: int | None = None,
        expert_start: int = 0,
        expert_count: int | None = None,
        preload_moe: bool = True,
    ):
        self.bundle = read_gguf_bundle(bundle_or_path) if not isinstance(bundle_or_path, GGUFBundle) else bundle_or_path
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError(f"MiniMax-M2 runtime requires CUDA device, got {resolved}")
        if resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        self.device = resolved
        self.dtype = dtype
        self.args = MiniMaxM2Args.from_bundle(self.bundle, n_layers=n_layers)
        self.expert_start = int(expert_start)
        self.expert_count = expert_count
        self.preload_moe = bool(preload_moe)
        self._readers: dict[str, GGUFTensorDataReader] = {}

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> "MiniMaxM2GGUFLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _tensor_ref(self, name: str) -> GGUFTensorRef:
        try:
            return self.bundle.tensors_by_name[name]
        except KeyError as exc:
            raise KeyError(f"MiniMax-M2 GGUF tensor not found: {name}") from exc

    def _reader_for(self, tensor: GGUFTensorRef) -> GGUFTensorDataReader:
        reader = self._readers.get(tensor.shard_path)
        if reader is None:
            reader = GGUFTensorDataReader(tensor.shard_path)
            self._readers[tensor.shard_path] = reader
        return reader

    def _read_dense(self, name: str) -> torch.Tensor:
        tensor = self._tensor_ref(name)
        values = self._reader_for(tensor).read_tensor(tensor.name)
        return values.to(device=self.device, dtype=torch.float32, non_blocking=False).contiguous()

    def _read_quant(self, name: str, expected_type: str) -> QuantizedGGUFTensor:
        tensor = self._tensor_ref(name)
        if tensor.type_name != expected_type:
            raise ValueError(f"{name} expected {expected_type}, got {tensor.type_name}")
        blocks, type_name, row_elems = self._reader_for(tensor).read_quantized_matrix_blocks(tensor.name)
        if type_name != expected_type:
            raise RuntimeError(f"{name} reader type mismatch: expected={expected_type} got={type_name}")
        try:
            type_id = GGUF_DENSE_TYPE_IDS[type_name]
        except KeyError as exc:
            raise NotImplementedError(f"GGUF type {type_name!r} is not supported by the MiniMax runtime") from exc
        cuda_blocks = blocks.to(device=self.device, non_blocking=False).contiguous()
        return QuantizedGGUFTensor(
            source_name=name,
            blocks=cuda_blocks,
            type_name=type_name,
            type_id=int(type_id),
            row_elems=int(row_elems),
            out_dim=int(cuda_blocks.shape[0]),
        )

    def _read_quant_rows(
        self,
        name: str,
        expected_type: str,
        row_start: int,
        row_count: int,
    ) -> QuantizedGGUFTensor:
        tensor = self._tensor_ref(name)
        if tensor.type_name != expected_type:
            raise ValueError(f"{name} expected {expected_type}, got {tensor.type_name}")
        blocks, type_name, row_elems = self._reader_for(tensor).read_quantized_matrix_block_rows(
            tensor.name,
            int(row_start),
            int(row_count),
        )
        if type_name != expected_type:
            raise RuntimeError(f"{name} reader type mismatch: expected={expected_type} got={type_name}")
        try:
            type_id = GGUF_DENSE_TYPE_IDS[type_name]
        except KeyError as exc:
            raise NotImplementedError(f"GGUF type {type_name!r} is not supported by the MiniMax runtime") from exc
        cuda_blocks = blocks.to(device=self.device, non_blocking=False).contiguous()
        return QuantizedGGUFTensor(
            source_name=name,
            blocks=cuda_blocks,
            type_name=type_name,
            type_id=int(type_id),
            row_elems=int(row_elems),
            out_dim=int(cuda_blocks.shape[0]),
            row_start=int(row_start),
        )

    @staticmethod
    def _rank_row_range(total_rows: int, world: int, rank: int) -> tuple[int, int]:
        total_rows = int(total_rows)
        world = int(world)
        rank = int(rank)
        if total_rows <= 0:
            raise ValueError(f"total_rows must be positive, got {total_rows}")
        if world <= 0:
            raise ValueError(f"world must be positive, got {world}")
        if rank < 0 or rank >= world:
            raise ValueError(f"rank {rank} outside [0, {world})")
        start = (total_rows * rank) // world
        end = (total_rows * (rank + 1)) // world
        return start, end - start

    def _read_lm_head_quant(self) -> QuantizedGGUFTensor:
        tensor = self._tensor_ref("output.weight")
        total_rows = int(tensor.dimensions[1])
        shard_head = os.getenv("MINIMAX_M2_TP_SHARD_LM_HEAD", "1").lower() not in {"0", "false", "no"}
        dist_ready = torch.distributed.is_available() and torch.distributed.is_initialized()
        world = torch.distributed.get_world_size() if dist_ready else 1
        rank = torch.distributed.get_rank() if dist_ready else 0
        if shard_head and world > 1:
            row_start, row_count = self._rank_row_range(total_rows, world, rank)
            return self._read_quant_rows("output.weight", "q4_k", row_start, row_count)
        return self._read_quant("output.weight", "q4_k")

    def load(self) -> MiniMaxTransformer:
        cache = MiniMaxM2DeviceResidentCache(
            self.bundle,
            device=self.device,
            expert_start=self.expert_start,
            expert_count=self.expert_count,
        )
        if self.preload_moe:
            cache.preload_layers(0, self.args.n_layers)

        embedding = QuantizedGGUFEmbedding(self._read_quant("token_embd.weight", "q4_k"), out_dtype=self.dtype)
        lm_head = QuantizedGGUFLinear(self._read_lm_head_quant(), out_dtype=self.dtype)
        final_norm = self._read_dense("output_norm.weight")

        layers: list[MiniMaxBlock] = []
        for layer_id in range(self.args.n_layers):
            prefix = f"blk.{layer_id}"
            q_proj = QuantizedGGUFLinear(self._read_quant(f"{prefix}.attn_q.weight", "q5_k"), out_dtype=self.dtype)
            k_proj = QuantizedGGUFLinear(self._read_quant(f"{prefix}.attn_k.weight", "q5_k"), out_dtype=self.dtype)
            v_proj = QuantizedGGUFLinear(self._read_quant(f"{prefix}.attn_v.weight", "q5_k"), out_dtype=self.dtype)
            o_proj = QuantizedGGUFLinear(self._read_quant(f"{prefix}.attn_output.weight", "q5_k"), out_dtype=self.dtype)
            attention = MiniMaxAttention(
                self.args,
                layer_id,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                self._read_dense(f"{prefix}.attn_q_norm.weight"),
                self._read_dense(f"{prefix}.attn_k_norm.weight"),
                device=self.device,
                dtype=self.dtype,
            )
            moe = MiniMaxMoE(
                self.args,
                layer_id,
                self._read_dense(f"{prefix}.ffn_gate_inp.weight"),
                self._read_dense(f"{prefix}.exp_probs_b.bias"),
                cache,
                dtype=self.dtype,
            )
            layers.append(
                MiniMaxBlock(
                    self.args,
                    layer_id,
                    self._read_dense(f"{prefix}.attn_norm.weight"),
                    self._read_dense(f"{prefix}.ffn_norm.weight"),
                    attention,
                    moe,
                    dtype=self.dtype,
                )
            )

        return MiniMaxTransformer(
            self.args,
            embedding,
            layers,
            final_norm,
            lm_head,
            device=self.device,
            dtype=self.dtype,
        )


def load_minimax_m2_gguf_model(
    gguf_path: str | Path | GGUFBundle,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    n_layers: int | None = None,
    expert_start: int = 0,
    expert_count: int | None = None,
    preload_moe: bool = True,
) -> tuple[MiniMaxTransformer, dict[str, Any]]:
    start = time.perf_counter()
    loader = MiniMaxM2GGUFLoader(
        gguf_path,
        device=device,
        dtype=dtype,
        n_layers=n_layers,
        expert_start=expert_start,
        expert_count=expert_count,
        preload_moe=preload_moe,
    )
    try:
        model = loader.load()
    finally:
        loader.close()
    elapsed = time.perf_counter() - start
    info = {
        "load_seconds": elapsed,
        "layers": model.args.n_layers,
        "dim": model.args.dim,
        "vocab_size": model.args.vocab_size,
        "device": str(model.device),
        "dtype": str(dtype),
        "expert_start": int(expert_start),
        "expert_count": int(expert_count) if expert_count is not None else None,
    }
    return model, info
