from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch

from src.gguf.iq1_grid import iq1_grid_i8
from src.gguf.reader import GGUF_MAGIC, GGUFReader, align_up, tensor_nbytes
from src.gguf.tensor_reader import GGUFTensorDataReader


ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "data" / "iq1m_oracle.json"
GGUF_VERSION = 3
GGML_TYPE_IQ1_M = 29


def _load_fixture() -> tuple[bytes, np.ndarray]:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    blocks = bytes.fromhex(raw["blocks_u8"])
    values = np.frombuffer(bytes.fromhex(raw["values_f32"]), dtype="<f4").reshape(raw["n_blocks"], raw["block_elems"])
    assert len(blocks) == raw["n_blocks"] * raw["block_bytes"]
    return blocks, values.copy()


def _pack_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _pad(buf: bytearray, alignment: int = 32) -> None:
    padded = align_up(len(buf), alignment)
    if padded > len(buf):
        buf.extend(b"\0" * (padded - len(buf)))


def _write_minimal_gguf(path: Path, *, name: str, dims: tuple[int, ...], type_id: int, payload: bytes) -> None:
    buf = bytearray()
    buf.extend(GGUF_MAGIC)
    buf.extend(struct.pack("<I", GGUF_VERSION))
    buf.extend(struct.pack("<Q", 1))  # tensor_count
    buf.extend(struct.pack("<Q", 0))  # metadata_count

    buf.extend(_pack_string(name))
    buf.extend(struct.pack("<I", len(dims)))
    for dim in dims:
        buf.extend(struct.pack("<Q", int(dim)))
    buf.extend(struct.pack("<I", int(type_id)))
    buf.extend(struct.pack("<Q", 0))  # tensor offset from data_start
    _pad(buf)

    assert tensor_nbytes(type_id, dims) == len(payload)
    buf.extend(payload)
    path.write_bytes(bytes(buf))


def test_iq1_grid_is_self_contained() -> None:
    grid = iq1_grid_i8()
    assert grid.shape == (2048, 8)
    assert grid.dtype == np.int8
    assert set(np.unique(grid).tolist()) == {-1, 0, 1}


def test_iq1m_reader_decodes_matrix_against_gguf_py_oracle(tmp_path: Path) -> None:
    blocks, expected = _load_fixture()
    gguf_path = tmp_path / "iq1m-matrix.gguf"
    _write_minimal_gguf(gguf_path, name="w", dims=(256, 4), type_id=GGML_TYPE_IQ1_M, payload=blocks)

    gguf = GGUFReader(str(gguf_path)).read()
    assert gguf.tensors[0].type_name == "iq1_m"
    assert gguf.tensors[0].nbytes == 4 * 56

    with GGUFTensorDataReader(str(gguf_path)) as reader:
        got = reader.read_tensor("w")

    assert got.dtype == torch.float32
    assert got.shape == (4, 256)
    np.testing.assert_array_equal(got.numpy(), expected)


def test_iq1m_reader_decodes_routed_expert_and_raw_blocks(tmp_path: Path) -> None:
    blocks, expected = _load_fixture()
    gguf_path = tmp_path / "iq1m-routed.gguf"
    # Payload order for routed tensors is expert-major. With dims=(in_dim, out_dim,
    # n_experts), four blocks map to expert0 rows 0..1 then expert1 rows 0..1.
    _write_minimal_gguf(gguf_path, name="blk.0.ffn_gate_exps.weight", dims=(256, 2, 2), type_id=GGML_TYPE_IQ1_M, payload=blocks)

    with GGUFTensorDataReader(str(gguf_path)) as reader:
        expert1 = reader.read_routed_expert("blk.0.ffn_gate_exps.weight", expert=1)
        raw, type_name, in_dim = reader.read_routed_expert_blocks("blk.0.ffn_gate_exps.weight", expert=1)
        layer_raw, layer_type_name, layer_in_dim = reader.read_routed_layer_blocks("blk.0.ffn_gate_exps.weight")

    assert expert1.shape == (2, 256)
    np.testing.assert_array_equal(expert1.numpy(), expected[2:4])

    assert type_name == "iq1_m"
    assert in_dim == 256
    assert raw.shape == (2, 1, 56)
    assert bytes(raw.numpy().reshape(-1).tolist()) == blocks[2 * 56:]

    assert layer_type_name == "iq1_m"
    assert layer_in_dim == 256
    assert layer_raw.shape == (2, 2, 1, 56)


def test_gguf_type_helper_never_maps_iq1m_to_q2k() -> None:
    # IQ1_M now has its own CUDA/native type id and must never be silently
    # treated as Q2_K (type id 1).
    from src.models.deepseek_v4.runtime import Expert

    expert = Expert.__new__(Expert)
    assert expert._gguf_type_id("iq2_xxs") == 0
    assert expert._gguf_type_id("q2_k") == 1
    assert expert._gguf_type_id("iq1_m") == 2
    assert expert._gguf_type_id("iq1_m") != expert._gguf_type_id("q2_k")
