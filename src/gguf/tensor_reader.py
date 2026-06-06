from __future__ import annotations

import math
import mmap
import os
import time
from functools import lru_cache
from typing import Iterable

import numpy as np
import torch

from src.gguf.reader import GGUFFile, GGUFReader, GGUFTensorInfo


_GGUF_READER_PROFILE = os.getenv("DEEPSEEK_GGUF_READER_PROFILE", "0").lower() in {"1", "true", "yes"}
_GGUF_READER_PROFILE_LIMIT = int(os.getenv("DEEPSEEK_GGUF_READER_PROFILE_LIMIT", "64"))
_GGUF_READER_PROFILE_COUNT = 0


_IQ2XXS_GRID = (
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
)

_DENSE_DTYPES = {
    "f32": ("<f4", torch.float32, 4),
    "f16": ("<f2", torch.float16, 2),
    "i32": ("<i4", torch.int32, 4),
}

# Quantized routed/matrix block geometry: type_name -> (block_elems, block_bytes).
# Mirrors GGML_QUANT_SIZES for the types we decode in pure Python.
_QUANT_BLOCK_META = {
    "q2_k": (256, 84),
    "iq2_xxs": (256, 66),
    "iq1_m": (256, 56),
}


def _quant_block_meta(type_name: str) -> tuple[int, int]:
    try:
        return _QUANT_BLOCK_META[type_name]
    except KeyError as exc:
        raise NotImplementedError(f"no block geometry for quant type {type_name}") from exc


def _product(values: Iterable[int]) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


def _storage_shape(dimensions: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(reversed(tuple(int(dim) for dim in dimensions)))


def _f16_bytes_to_f32(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(data).view("<f2").astype(np.float32).reshape(data.shape[:-1])


@lru_cache(maxsize=1)
def _iq2xxs_signed_grid() -> np.ndarray:
    base = np.empty((256, 8), dtype=np.int8)
    for idx, value in enumerate(_IQ2XXS_GRID):
        base[idx] = np.frombuffer(int(value).to_bytes(8, "little"), dtype=np.uint8).astype(np.int8)
    signs = np.empty((128, 8), dtype=np.int8)
    masks = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    for idx in range(128):
        sign_mask = idx | ((int(idx).bit_count() & 1) << 7)
        signs[idx] = np.where((sign_mask & masks) != 0, -1, 1).astype(np.int8)
    return (base[:, None, :].astype(np.int16) * signs[None, :, :].astype(np.int16)).astype(np.int8)


@lru_cache(maxsize=1)
def get_iq2xxs_signed_grid_tensor() -> torch.Tensor:
    return torch.from_numpy(_iq2xxs_signed_grid().copy()).contiguous().to(device="cpu")


@lru_cache(maxsize=1)
def get_iq1_grid_tensor() -> torch.Tensor:
    from src.gguf.iq1_grid import iq1_grid_i8

    return torch.from_numpy(iq1_grid_i8().copy()).contiguous().to(device="cpu")


@lru_cache(maxsize=4)
def get_cached_gguf_tensor_reader(path: str) -> GGUFTensorDataReader:
    return GGUFTensorDataReader(path)


class GGUFTensorDataReader:
    def __init__(self, gguf: GGUFFile | str):
        self.gguf = GGUFReader(gguf).read() if isinstance(gguf, str) else gguf
        self._fd = os.open(self.gguf.path, os.O_RDONLY)
        self._mmap = mmap.mmap(self._fd, 0, access=mmap.ACCESS_COPY)

    def close(self) -> None:
        mapped = getattr(self, "_mmap", None)
        if mapped is not None:
            try:
                mapped.close()
            except BufferError:
                pass
            self._mmap = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> "GGUFTensorDataReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _tensor(self, name: str | GGUFTensorInfo) -> GGUFTensorInfo:
        if isinstance(name, GGUFTensorInfo):
            return name
        try:
            return self.gguf.tensors_by_name[name]
        except KeyError as exc:
            raise KeyError(f"GGUF tensor not found: {name}") from exc

    def _read_at(self, offset: int, nbytes: int) -> bytes:
        data = os.pread(self._fd, int(nbytes), int(offset))
        if len(data) != int(nbytes):
            raise EOFError(f"short GGUF read at offset {offset}: got {len(data)}, expected {nbytes}")
        return data

    def read_tensor(self, name: str | GGUFTensorInfo) -> torch.Tensor:
        tensor = self._tensor(name)
        if tensor.type_name in _DENSE_DTYPES or tensor.type_name == "bf16":
            return self._read_dense_tensor(tensor)
        if tensor.type_name == "q8_0":
            return self._read_q8_0_tensor(tensor)
        if tensor.type_name in {"q2_k", "iq2_xxs", "iq1_m"} and len(tensor.dimensions) == 2:
            return self._read_quantized_matrix(tensor, tensor.absolute_offset, int(tensor.dimensions[0]), int(tensor.dimensions[1]), tensor.type_name)
        raise NotImplementedError(f"payload decode for {tensor.name} ({tensor.type_name}) is not supported by read_tensor")

    def read_tensor_rows(self, name: str | GGUFTensorInfo, row_start: int, row_count: int) -> torch.Tensor:
        tensor = self._tensor(name)
        row_elems = int(tensor.dimensions[0])
        rows = _product(tensor.dimensions[1:])
        if row_start < 0 or row_count < 0 or row_start + row_count > rows:
            raise ValueError(f"row range [{row_start}, {row_start + row_count}) is outside {tensor.name} rows={rows}")
        if tensor.type_name in _DENSE_DTYPES or tensor.type_name == "bf16":
            return self._read_dense_rows(tensor, row_start, row_count)
        if tensor.type_name == "q8_0":
            return self._read_q8_0_rows(tensor, row_start, row_count)
        raise NotImplementedError(f"row decode for {tensor.name} ({tensor.type_name}) is not supported")

    def read_routed_expert(
        self,
        name: str | GGUFTensorInfo,
        expert: int,
        row_start: int = 0,
        row_count: int | None = None,
    ) -> torch.Tensor:
        tensor = self._tensor(name)
        if len(tensor.dimensions) != 3:
            raise ValueError(f"{tensor.name} is not a routed expert tensor")
        in_dim, out_dim, n_experts = (int(dim) for dim in tensor.dimensions)
        if expert < 0 or expert >= n_experts:
            raise ValueError(f"expert {expert} is outside {tensor.name} expert count {n_experts}")
        if tensor.type_name not in _QUANT_BLOCK_META:
            raise NotImplementedError(f"routed expert decode for {tensor.type_name} is not supported")
        block_elems, block_bytes = _quant_block_meta(tensor.type_name)
        blocks_per_row = math.ceil(in_dim / block_elems)
        row_bytes = blocks_per_row * block_bytes
        row_count = out_dim - row_start if row_count is None else int(row_count)
        if row_start < 0 or row_count < 0 or row_start + row_count > out_dim:
            raise ValueError(f"row range [{row_start}, {row_start + row_count}) is outside {tensor.name} out_dim={out_dim}")
        expert_bytes = out_dim * row_bytes
        offset = tensor.absolute_offset + expert * expert_bytes + row_start * row_bytes
        return self._read_quantized_matrix(tensor, offset, in_dim, row_count, tensor.type_name)

    def _routed_expert_block_meta(
        self,
        name: str | GGUFTensorInfo,
        expert: int,
        row_start: int = 0,
        row_count: int | None = None,
    ) -> tuple[GGUFTensorInfo, int, int, int, int, int, int, int]:
        tensor = self._tensor(name)
        if len(tensor.dimensions) != 3:
            raise ValueError(f"{tensor.name} is not a routed expert tensor")
        in_dim, out_dim, n_experts = (int(dim) for dim in tensor.dimensions)
        if expert < 0 or expert >= n_experts:
            raise ValueError(f"expert {expert} is outside {tensor.name} expert count {n_experts}")
        if tensor.type_name not in _QUANT_BLOCK_META:
            raise NotImplementedError(f"routed expert raw blocks for {tensor.type_name} are not supported")
        block_elems, block_bytes = _quant_block_meta(tensor.type_name)
        if in_dim % block_elems != 0:
            raise ValueError(f"{tensor.name} in_dim={in_dim} is not divisible by {block_elems}")
        blocks_per_row = in_dim // block_elems
        row_bytes = blocks_per_row * block_bytes
        row_count = out_dim - row_start if row_count is None else int(row_count)
        if row_start < 0 or row_count < 0 or row_start + row_count > out_dim:
            raise ValueError(f"row range [{row_start}, {row_start + row_count}) is outside {tensor.name} out_dim={out_dim}")
        expert_bytes = out_dim * row_bytes
        offset = tensor.absolute_offset + expert * expert_bytes + row_start * row_bytes
        nbytes = row_count * row_bytes
        return tensor, in_dim, out_dim, blocks_per_row, block_bytes, offset, nbytes, row_count

    def routed_expert_blocks_ptr(
        self,
        name: str | GGUFTensorInfo,
        expert: int,
        row_start: int = 0,
        row_count: int | None = None,
    ) -> tuple[int, str, int, int, int, memoryview]:
        tensor, in_dim, _out_dim, blocks_per_row, block_bytes, offset, nbytes, _row_count = self._routed_expert_block_meta(
            name,
            expert,
            row_start,
            row_count,
        )
        view = memoryview(self._mmap)[offset:offset + nbytes]
        return int(offset), tensor.type_name, in_dim, blocks_per_row, block_bytes, view

    def read_routed_layer_blocks(
        self,
        name: str | GGUFTensorInfo,
        expert_start: int = 0,
        expert_count: int | None = None,
    ) -> tuple[torch.Tensor, str, int]:
        tensor = self._tensor(name)
        if len(tensor.dimensions) != 3:
            raise ValueError(f"{tensor.name} is not a routed expert tensor")
        in_dim, out_dim, n_experts = (int(dim) for dim in tensor.dimensions)
        if tensor.type_name not in _QUANT_BLOCK_META:
            raise NotImplementedError(f"routed expert raw blocks for {tensor.type_name} are not supported")
        block_elems, block_bytes = _quant_block_meta(tensor.type_name)
        if in_dim % block_elems != 0:
            raise ValueError(f"{tensor.name} in_dim={in_dim} is not divisible by {block_elems}")
        blocks_per_row = in_dim // block_elems
        row_bytes = blocks_per_row * block_bytes
        expert_start = int(expert_start)
        expert_count = n_experts - expert_start if expert_count is None else int(expert_count)
        if expert_start < 0 or expert_count < 0 or expert_start + expert_count > n_experts:
            raise ValueError(f"expert range [{expert_start}, {expert_start + expert_count}) is outside {tensor.name} experts={n_experts}")
        expert_bytes = out_dim * row_bytes
        nbytes = expert_count * expert_bytes
        offset = tensor.absolute_offset + expert_start * expert_bytes
        view = memoryview(self._mmap)[offset:offset + nbytes]
        blocks = torch.frombuffer(view, dtype=torch.uint8, count=nbytes).view(expert_count, out_dim, blocks_per_row, block_bytes)
        return blocks, tensor.type_name, in_dim

    def read_routed_expert_blocks(
        self,
        name: str | GGUFTensorInfo,
        expert: int,
        row_start: int = 0,
        row_count: int | None = None,
    ) -> tuple[torch.Tensor, str, int]:
        tensor, in_dim, _out_dim, blocks_per_row, block_bytes, offset, nbytes, row_count = self._routed_expert_block_meta(
            name,
            expert,
            row_start,
            row_count,
        )
        view = memoryview(self._mmap)[offset:offset + nbytes]
        blocks = torch.frombuffer(view, dtype=torch.uint8, count=nbytes).view(row_count, blocks_per_row, block_bytes)
        return blocks, tensor.type_name, in_dim

    def _read_dense_tensor(self, tensor: GGUFTensorInfo) -> torch.Tensor:
        data = self._read_at(tensor.absolute_offset, tensor.nbytes or 0)
        return self._dense_from_bytes(data, tensor.type_name, _storage_shape(tensor.dimensions))

    def _read_dense_rows(self, tensor: GGUFTensorInfo, row_start: int, row_count: int) -> torch.Tensor:
        row_elems = int(tensor.dimensions[0])
        if tensor.type_name == "bf16":
            elem_size = 2
        else:
            elem_size = _DENSE_DTYPES[tensor.type_name][2]
        row_bytes = row_elems * elem_size
        data = self._read_at(tensor.absolute_offset + row_start * row_bytes, row_count * row_bytes)
        return self._dense_from_bytes(data, tensor.type_name, (row_count, row_elems))

    def _dense_from_bytes(self, data: bytes, type_name: str, shape: tuple[int, ...]) -> torch.Tensor:
        if type_name == "bf16":
            raw = np.frombuffer(data, dtype="<u2").astype(np.uint32)
            array = (raw << 16).view(np.float32).reshape(shape).copy()
            return torch.from_numpy(array).to(torch.bfloat16)
        dtype, _torch_dtype, _elem_size = _DENSE_DTYPES[type_name]
        array = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        return torch.from_numpy(array)

    def _read_q8_0_tensor(self, tensor: GGUFTensorInfo) -> torch.Tensor:
        row_elems = int(tensor.dimensions[0])
        rows = _product(tensor.dimensions[1:])
        values = self._read_q8_0_rows_array(tensor.absolute_offset, row_elems, rows)
        return torch.from_numpy(values.reshape(_storage_shape(tensor.dimensions)).copy())

    def read_q8_0_blocks(self, name: str | GGUFTensorInfo) -> torch.Tensor:
        tensor = self._tensor(name)
        if tensor.type_name != "q8_0":
            raise NotImplementedError(f"raw q8_0 blocks for {tensor.name} ({tensor.type_name}) are not supported")
        row_elems = int(tensor.dimensions[0])
        rows = _product(tensor.dimensions[1:])
        return self._read_q8_0_block_rows(tensor.absolute_offset, row_elems, rows)

    def read_q8_0_block_rows(self, name: str | GGUFTensorInfo, row_start: int, row_count: int) -> torch.Tensor:
        tensor = self._tensor(name)
        if tensor.type_name != "q8_0":
            raise NotImplementedError(f"raw q8_0 block rows for {tensor.name} ({tensor.type_name}) are not supported")
        rows = _product(tensor.dimensions[1:])
        if row_start < 0 or row_count < 0 or row_start + row_count > rows:
            raise ValueError(f"row range [{row_start}, {row_start + row_count}) is outside {tensor.name} rows={rows}")
        row_elems = int(tensor.dimensions[0])
        blocks_per_row = math.ceil(row_elems / 32)
        row_bytes = blocks_per_row * 34
        return self._read_q8_0_block_rows(tensor.absolute_offset + row_start * row_bytes, row_elems, row_count)

    def _read_q8_0_rows(self, tensor: GGUFTensorInfo, row_start: int, row_count: int) -> torch.Tensor:
        row_elems = int(tensor.dimensions[0])
        blocks_per_row = math.ceil(row_elems / 32)
        row_bytes = blocks_per_row * 34
        values = self._read_q8_0_rows_array(tensor.absolute_offset + row_start * row_bytes, row_elems, row_count)
        return torch.from_numpy(values.copy())

    def _read_q8_0_rows_array(self, offset: int, row_elems: int, rows: int) -> np.ndarray:
        blocks_per_row = math.ceil(row_elems / 32)
        data = self._read_at(offset, rows * blocks_per_row * 34)
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, blocks_per_row, 34)
        d = _f16_bytes_to_f32(blocks[:, :, 0:2])
        qs = blocks[:, :, 2:34].view(np.int8).astype(np.float32)
        values = qs * d[:, :, None]
        return values.reshape(rows, blocks_per_row * 32)[:, :row_elems]

    def _read_q8_0_block_rows(self, offset: int, row_elems: int, rows: int) -> torch.Tensor:
        blocks_per_row = math.ceil(row_elems / 32)
        data = self._read_at(offset, rows * blocks_per_row * 34)
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, blocks_per_row, 34).copy()
        return torch.from_numpy(blocks).to(device="cpu")

    def _read_quantized_matrix(self, tensor: GGUFTensorInfo, offset: int, in_dim: int, out_dim: int, type_name: str) -> torch.Tensor:
        if in_dim % 256 != 0:
            raise ValueError(f"{tensor.name} in_dim={in_dim} is not divisible by 256")
        blocks_per_row = in_dim // 256
        if type_name == "q2_k":
            values = self._read_q2_k_rows(offset, out_dim, blocks_per_row)
        elif type_name == "iq2_xxs":
            values = self._read_iq2_xxs_rows(offset, out_dim, blocks_per_row)
        elif type_name == "iq1_m":
            values = self._read_iq1_m_rows(offset, out_dim, blocks_per_row)
        else:
            raise NotImplementedError(type_name)
        return torch.from_numpy(values.reshape(out_dim, blocks_per_row * 256)[:, :in_dim].copy())

    def _read_q2_k_rows(self, offset: int, rows: int, blocks_per_row: int) -> np.ndarray:
        global _GGUF_READER_PROFILE_COUNT
        profile = _GGUF_READER_PROFILE and _GGUF_READER_PROFILE_COUNT < _GGUF_READER_PROFILE_LIMIT
        t0 = time.perf_counter() if profile else 0.0
        nbytes = rows * blocks_per_row * 84
        data = self._read_at(offset, nbytes)
        if profile:
            t_read = time.perf_counter()
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, blocks_per_row, 84)
        scales = blocks[:, :, :16]
        qs = blocks[:, :, 16:80]
        d = _f16_bytes_to_f32(blocks[:, :, 80:82])
        dmin = _f16_bytes_to_f32(blocks[:, :, 82:84])
        out = np.empty((rows, blocks_per_row, 256), dtype=np.float32)
        for group in range(16):
            half_block = group // 8
            group_in_half = group % 8
            shift = (group_in_half // 2) * 2
            byte_start = half_block * 32 + (group_in_half % 2) * 16
            q = ((qs[:, :, byte_start:byte_start + 16] >> shift) & 0x03).astype(np.float32)
            scale = (scales[:, :, group] & 0x0F).astype(np.float32)
            minv = (scales[:, :, group] >> 4).astype(np.float32)
            out[:, :, group * 16:(group + 1) * 16] = d[:, :, None] * scale[:, :, None] * q - dmin[:, :, None] * minv[:, :, None]
        if profile:
            t_done = time.perf_counter()
            _GGUF_READER_PROFILE_COUNT += 1
            print(
                f"gguf_reader_profile type=q2_k rows={rows} blocks_per_row={blocks_per_row} bytes={nbytes} "
                f"read={t_read - t0:.6f}s decode={t_done - t_read:.6f}s total={t_done - t0:.6f}s",
                flush=True,
            )
        return out

    def _read_iq2_xxs_rows(self, offset: int, rows: int, blocks_per_row: int) -> np.ndarray:
        global _GGUF_READER_PROFILE_COUNT
        profile = _GGUF_READER_PROFILE and _GGUF_READER_PROFILE_COUNT < _GGUF_READER_PROFILE_LIMIT
        t0 = time.perf_counter() if profile else 0.0
        nbytes = rows * blocks_per_row * 66
        data = self._read_at(offset, nbytes)
        if profile:
            t_read = time.perf_counter()
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, blocks_per_row, 66)
        d = _f16_bytes_to_f32(blocks[:, :, 0:2])
        qs = blocks[:, :, 2:66]
        signed_grid = _iq2xxs_signed_grid()
        out = np.empty((rows, blocks_per_row, 256), dtype=np.float32)
        for sub in range(8):
            chunk = qs[:, :, sub * 8:(sub + 1) * 8]
            aux1 = (
                chunk[:, :, 4].astype(np.uint32)
                | (chunk[:, :, 5].astype(np.uint32) << 8)
                | (chunk[:, :, 6].astype(np.uint32) << 16)
                | (chunk[:, :, 7].astype(np.uint32) << 24)
            )
            ls = (2 * (aux1 >> 28) + 1).astype(np.float32)
            for part in range(4):
                grid_ids = chunk[:, :, part].astype(np.int64)
                sign_idx = ((aux1 >> (7 * part)) & 127).astype(np.int64)
                values = signed_grid[grid_ids, sign_idx].astype(np.float32)
                start = sub * 32 + part * 8
                out[:, :, start:start + 8] = 0.125 * d[:, :, None] * ls[:, :, None] * values
        if profile:
            t_done = time.perf_counter()
            _GGUF_READER_PROFILE_COUNT += 1
            print(
                f"gguf_reader_profile type=iq2_xxs rows={rows} blocks_per_row={blocks_per_row} bytes={nbytes} "
                f"read={t_read - t0:.6f}s decode={t_done - t_read:.6f}s total={t_done - t0:.6f}s",
                flush=True,
            )
        return out

    def _read_iq1_m_rows(self, offset: int, rows: int, blocks_per_row: int) -> np.ndarray:
        """Decode IQ1_M rows to float32.

        IQ1_M block layout is 56 bytes for 256 values:
        ``qs[32] + qh[16] + scales[8]``.  The formula mirrors llama.cpp
        gguf-py ``IQ1_M.dequantize_blocks``; imatrix is only needed while
        producing IQ1_M, not while decoding it.
        """
        global _GGUF_READER_PROFILE_COUNT
        profile = _GGUF_READER_PROFILE and _GGUF_READER_PROFILE_COUNT < _GGUF_READER_PROFILE_LIMIT
        t0 = time.perf_counter() if profile else 0.0
        nbytes = rows * blocks_per_row * 56
        data = self._read_at(offset, nbytes)
        if profile:
            t_read = time.perf_counter()

        n_blocks = rows * blocks_per_row
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, 56)
        qs = blocks[:, :32]
        qh = blocks[:, 32:48]
        scales = blocks[:, 48:56].view(np.uint16)

        # Reconstruct the shared f16 super-block scale from the high nibbles of
        # four uint16 scale words.
        d = (scales.reshape((n_blocks, 4)) & np.uint16(0xF000)) >> np.array([12, 8, 4, 0], dtype=np.uint16).reshape((1, 4))
        d = d[:, 0] | d[:, 1] | d[:, 2] | d[:, 3]
        d = d.view(np.float16).astype(np.float32).reshape((n_blocks, 1))

        # Low 12 bits of the scale words contain 4 packed 3-bit local scales.
        local_scales = scales.reshape(n_blocks, -1, 1) >> np.array([0, 3, 6, 9], dtype=np.uint16).reshape((1, 1, 4))
        local_scales = (local_scales & 0x07).reshape((n_blocks, -1))
        dl = d * (2 * local_scales + 1)
        dl = dl.reshape((n_blocks, -1, 2, 1, 1))

        qh_parts = qh.reshape((n_blocks, -1, 1)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2))
        qidx = qs.astype(np.uint16) | ((qh_parts & 0x07).astype(np.uint16) << 8).reshape((n_blocks, -1))

        delta = np.where(qh_parts & 0x08 == 0, np.float32(0.125), np.float32(-0.125))
        delta = delta.reshape((n_blocks, -1, 2, 2, 1))

        from src.gguf.iq1_grid import iq1_grid_i8

        grid = iq1_grid_i8().astype(np.float32, copy=False)[qidx.reshape(-1)].reshape((n_blocks, -1, 2, 2, 8))
        out = (dl * (grid + delta)).reshape((rows, blocks_per_row, 256))
        if profile:
            t_done = time.perf_counter()
            _GGUF_READER_PROFILE_COUNT += 1
            print(
                f"gguf_reader_profile type=iq1_m rows={rows} blocks_per_row={blocks_per_row} bytes={nbytes} "
                f"read={t_read - t0:.6f}s decode={t_done - t_read:.6f}s total={t_done - t0:.6f}s",
                flush=True,
            )
        return out
