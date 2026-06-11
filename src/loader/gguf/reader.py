from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import BinaryIO, Any


GGUF_MAGIC = b"GGUF"

_METADATA_TYPES = {
    0: ("uint8", "<B", 1),
    1: ("int8", "<b", 1),
    2: ("uint16", "<H", 2),
    3: ("int16", "<h", 2),
    4: ("uint32", "<I", 4),
    5: ("int32", "<i", 4),
    6: ("float32", "<f", 4),
    7: ("bool", "<?", 1),
    8: ("string", None, None),
    9: ("array", None, None),
    10: ("uint64", "<Q", 8),
    11: ("int64", "<q", 8),
    12: ("float64", "<d", 8),
}

GGML_TYPES = {
    0: ("f32", 1, 4),
    1: ("f16", 1, 2),
    2: ("q4_0", 32, 18),
    3: ("q4_1", 32, 20),
    6: ("q5_0", 32, 22),
    7: ("q5_1", 32, 24),
    8: ("q8_0", 32, 34),
    9: ("q8_1", 32, 40),
    10: ("q2_k", 256, 84),
    11: ("q3_k", 256, 110),
    12: ("q4_k", 256, 144),
    13: ("q5_k", 256, 176),
    14: ("q6_k", 256, 210),
    15: ("q8_k", 256, 292),
    16: ("iq2_xxs", 256, 66),
    17: ("iq2_xs", 256, 74),
    18: ("iq3_xxs", 256, 98),
    19: ("iq1_s", 256, 50),
    20: ("iq4_nl", 32, 18),
    21: ("iq3_s", 256, 110),
    22: ("iq2_s", 256, 82),
    23: ("iq4_xs", 256, 136),
    24: ("i8", 1, 1),
    25: ("i16", 1, 2),
    26: ("i32", 1, 4),
    27: ("i64", 1, 8),
    28: ("f64", 1, 8),
    29: ("iq1_m", 256, 56),
    30: ("bf16", 1, 2),
}


@dataclass(frozen=True)
class GGUFArraySummary:
    value_type: int
    value_type_name: str
    length: int


@dataclass(frozen=True)
class GGUFTensorInfo:
    name: str
    dimensions: tuple[int, ...]
    type_id: int
    offset: int
    absolute_offset: int
    nbytes: int | None

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.type_id, (f"unknown_{self.type_id}", 0, 0))[0]

    @property
    def elements(self) -> int:
        total = 1
        for dim in self.dimensions:
            total *= int(dim)
        return total


@dataclass(frozen=True)
class GGUFFile:
    path: str
    version: int
    tensor_count: int
    metadata_count: int
    metadata: dict[str, Any]
    tensors: list[GGUFTensorInfo]
    data_start: int
    alignment: int
    size: int

    @property
    def tensors_by_name(self) -> dict[str, GGUFTensorInfo]:
        return {tensor.name: tensor for tensor in self.tensors}


def align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def tensor_nbytes(type_id: int, dimensions: tuple[int, ...]) -> int | None:
    info = GGML_TYPES.get(type_id)
    if info is None:
        return None
    _name, block_elems, block_bytes = info
    total = 1
    for dim in dimensions:
        total *= int(dim)
    blocks = (total + block_elems - 1) // block_elems
    return blocks * block_bytes


class GGUFReader:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)

    def read(self) -> GGUFFile:
        with open(self.path, "rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError(f"{self.path} is not a GGUF file")
            version = _read_struct(f, "<I")
            tensor_count = _read_struct(f, "<Q")
            metadata_count = _read_struct(f, "<Q")
            metadata = self._read_metadata(f, metadata_count)
            tensor_records = self._read_tensor_records(f, tensor_count)
            alignment = int(metadata.get("general.alignment", 32))
            data_start = align_up(f.tell(), alignment)
            tensors = [
                GGUFTensorInfo(
                    name=name,
                    dimensions=dimensions,
                    type_id=type_id,
                    offset=offset,
                    absolute_offset=data_start + offset,
                    nbytes=tensor_nbytes(type_id, dimensions),
                )
                for name, dimensions, type_id, offset in tensor_records
            ]
        return GGUFFile(
            path=self.path,
            version=version,
            tensor_count=tensor_count,
            metadata_count=metadata_count,
            metadata=metadata,
            tensors=tensors,
            data_start=data_start,
            alignment=alignment,
            size=os.path.getsize(self.path),
        )

    def _read_metadata(self, f: BinaryIO, count: int) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for _ in range(count):
            key = _read_string(f)
            value_type = _read_struct(f, "<I")
            metadata[key] = self._read_value(f, value_type)
        return metadata

    def _read_tensor_records(self, f: BinaryIO, count: int) -> list[tuple[str, tuple[int, ...], int, int]]:
        tensors = []
        for _ in range(count):
            name = _read_string(f)
            n_dims = _read_struct(f, "<I")
            dimensions = tuple(_read_struct(f, "<Q") for _ in range(n_dims))
            type_id = _read_struct(f, "<I")
            offset = _read_struct(f, "<Q")
            tensors.append((name, dimensions, type_id, offset))
        return tensors

    def _read_value(self, f: BinaryIO, value_type: int) -> Any:
        type_name, fmt, size = _metadata_type(value_type)
        if type_name == "string":
            return _read_string(f)
        if type_name == "array":
            item_type = _read_struct(f, "<I")
            length = _read_struct(f, "<Q")
            self._skip_array(f, item_type, length)
            item_name = _metadata_type(item_type)[0]
            return GGUFArraySummary(item_type, item_name, length)
        if fmt is None or size is None:
            raise ValueError(f"unsupported GGUF metadata type {value_type}")
        return _read_struct(f, fmt)

    def _skip_array(self, f: BinaryIO, item_type: int, length: int) -> None:
        item_name, _fmt, size = _metadata_type(item_type)
        if item_name == "string":
            for _ in range(length):
                n = _read_struct(f, "<Q")
                f.seek(n, os.SEEK_CUR)
            return
        if item_name == "array":
            raise ValueError("nested GGUF metadata arrays are not supported")
        if size is None:
            raise ValueError(f"unsupported GGUF array item type {item_type}")
        f.seek(int(length) * int(size), os.SEEK_CUR)


def _metadata_type(type_id: int) -> tuple[str, str | None, int | None]:
    try:
        return _METADATA_TYPES[type_id]
    except KeyError as exc:
        raise ValueError(f"unsupported GGUF metadata type {type_id}") from exc


def _read_struct(f: BinaryIO, fmt: str):
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise EOFError("unexpected end of GGUF file")
    values = struct.unpack(fmt, data)
    return values[0] if len(values) == 1 else values


def _read_string(f: BinaryIO) -> str:
    n = _read_struct(f, "<Q")
    data = f.read(n)
    if len(data) != n:
        raise EOFError("unexpected end of GGUF string")
    return data.decode("utf-8", errors="replace")
