from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch

from src.components.gguf.quantized_ops import QuantizedGGUFEmbedding, QuantizedGGUFLinear
from src.components.gguf.tp_logits import tp_vocab_row_range
from src.loader.gguf.bundle import GGUFBundle, read_gguf_bundle
from src.loader.gguf.quantized_loader import GGUFQuantizedTensorLoader
from src.loader.gguf.quantized_tensor import QuantizedGGUFTensor
from src.models.minimax_m2.architecture import (
    MiniMaxAttention,
    MiniMaxBlock,
    MiniMaxM2Args,
    MiniMaxMoE,
    MiniMaxTransformer,
)
from src.models.minimax_m2.moe_runtime import MiniMaxM2DeviceResidentCache


class MiniMaxM2GGUFModelLoader:
    """Assemble MiniMax-M2 modules from a GGUF checkpoint.

    GGUF file/block loading is delegated to ``src.loader.gguf``.  CUDA kernel
    wrappers are delegated to ``src.components.gguf``.  This class only maps
    MiniMax tensor names into model modules.
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
        self._gguf_loader = GGUFQuantizedTensorLoader(self.bundle, device=self.device)

    def close(self) -> None:
        self._gguf_loader.close()

    def __enter__(self) -> "MiniMaxM2GGUFModelLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _read_dense(self, name: str) -> torch.Tensor:
        return self._gguf_loader.read_dense(name, dtype=torch.float32)

    def _read_quant(self, name: str, expected_type: str) -> QuantizedGGUFTensor:
        return self._gguf_loader.read_quant(name, expected_type)

    def _read_lm_head_quant(self) -> QuantizedGGUFTensor:
        tensor = self._gguf_loader.tensor_ref("output.weight")
        total_rows = int(tensor.dimensions[1])
        shard_head = os.getenv("MINIMAX_M2_TP_SHARD_LM_HEAD", "1").lower() not in {"0", "false", "no"}
        dist_ready = torch.distributed.is_available() and torch.distributed.is_initialized()
        world = torch.distributed.get_world_size() if dist_ready else 1
        rank = torch.distributed.get_rank() if dist_ready else 0
        if shard_head and world > 1:
            row_start, row_count = tp_vocab_row_range(total_rows, world, rank)
            return self._gguf_loader.read_quant_rows("output.weight", "q4_k", row_start, row_count)
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


# Backward-compatible alias for early PR revisions and local scripts.
MiniMaxM2GGUFLoader = MiniMaxM2GGUFModelLoader


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
    loader = MiniMaxM2GGUFModelLoader(
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
