from __future__ import annotations

from pathlib import Path

import torch

from src.loader.gguf.bundle import GGUFBundle, GGUFTensorRef, read_gguf_bundle
from src.loader.gguf.quant_types import GGUF_DENSE_TYPE_IDS
from src.loader.gguf.quantized_tensor import QuantizedGGUFTensor
from src.loader.gguf.tensor_reader import GGUFTensorDataReader


class GGUFQuantizedTensorLoader:
    """Read dense and raw quantized GGUF tensors into device memory.

    This class owns GGUF file readers and format/type checks only.  CUDA kernel
    invocation lives under src.components.gguf.
    """

    def __init__(self, bundle_or_path: GGUFBundle | str | Path, *, device: str | torch.device = "cuda"):
        self.bundle = read_gguf_bundle(bundle_or_path) if not isinstance(bundle_or_path, GGUFBundle) else bundle_or_path
        self.device = torch.device(device)
        self._readers: dict[str, GGUFTensorDataReader] = {}

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> "GGUFQuantizedTensorLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def tensor_ref(self, name: str) -> GGUFTensorRef:
        try:
            return self.bundle.tensors_by_name[name]
        except KeyError as exc:
            raise KeyError(f"GGUF tensor not found: {name}") from exc

    def reader_for(self, tensor: GGUFTensorRef) -> GGUFTensorDataReader:
        reader = self._readers.get(tensor.shard_path)
        if reader is None:
            reader = GGUFTensorDataReader(tensor.shard_path)
            self._readers[tensor.shard_path] = reader
        return reader

    def read_dense(self, name: str, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        tensor = self.tensor_ref(name)
        values = self.reader_for(tensor).read_tensor(tensor.name)
        return values.to(device=self.device, dtype=dtype, non_blocking=False).contiguous()

    def read_quant(self, name: str, expected_type: str) -> QuantizedGGUFTensor:
        tensor = self.tensor_ref(name)
        if tensor.type_name != expected_type:
            raise ValueError(f"{name} expected {expected_type}, got {tensor.type_name}")
        blocks, type_name, row_elems = self.reader_for(tensor).read_quantized_matrix_blocks(tensor.name)
        return self._to_quantized_tensor(name, blocks, type_name, row_elems, expected_type, row_start=0)

    def read_quant_rows(
        self,
        name: str,
        expected_type: str,
        row_start: int,
        row_count: int,
    ) -> QuantizedGGUFTensor:
        tensor = self.tensor_ref(name)
        if tensor.type_name != expected_type:
            raise ValueError(f"{name} expected {expected_type}, got {tensor.type_name}")
        blocks, type_name, row_elems = self.reader_for(tensor).read_quantized_matrix_block_rows(
            tensor.name,
            int(row_start),
            int(row_count),
        )
        return self._to_quantized_tensor(name, blocks, type_name, row_elems, expected_type, row_start=int(row_start))

    def _to_quantized_tensor(
        self,
        name: str,
        blocks: torch.Tensor,
        type_name: str,
        row_elems: int,
        expected_type: str,
        *,
        row_start: int,
    ) -> QuantizedGGUFTensor:
        if type_name != expected_type:
            raise RuntimeError(f"{name} reader type mismatch: expected={expected_type} got={type_name}")
        try:
            type_id = GGUF_DENSE_TYPE_IDS[type_name]
        except KeyError as exc:
            raise NotImplementedError(f"GGUF type {type_name!r} is not supported by the GGUF raw-block runtime") from exc
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
