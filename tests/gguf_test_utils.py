from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from src.gguf.reader import GGUF_MAGIC, align_up, tensor_nbytes

GGUF_VERSION = 3
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_IQ2_XXS = 16
GGML_IQ1_M = 29
GGML_BF16 = 30


def pack_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def pack_metadata_value(value: Any) -> bytes:
    if isinstance(value, str):
        return struct.pack("<I", 8) + pack_string(value)
    if isinstance(value, bool):
        return struct.pack("<I", 7) + struct.pack("<?", value)
    if isinstance(value, int):
        return struct.pack("<I", 5) + struct.pack("<i", int(value))
    if isinstance(value, float):
        return struct.pack("<I", 6) + struct.pack("<f", float(value))
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "array_strings":
        values = list(value[1])
        buf = bytearray()
        buf.extend(struct.pack("<I", 9))  # array
        buf.extend(struct.pack("<I", 8))  # string item type
        buf.extend(struct.pack("<Q", len(values)))
        for item in values:
            buf.extend(pack_string(str(item)))
        return bytes(buf)
    raise TypeError(f"unsupported GGUF test metadata value: {value!r}")


def write_gguf(
    path: Path,
    *,
    metadata: dict[str, Any] | None = None,
    tensors: list[tuple[str, tuple[int, ...], int]] | None = None,
    alignment: int = 32,
) -> None:
    """Write a tiny valid GGUF file for header-only tests.

    Tensor payloads are zero-filled and only sized according to GGML block
    metadata.  This helper intentionally implements the subset needed by the
    spec/bundle tests, not a general GGUF writer.
    """

    metadata = dict(metadata or {})
    tensors = list(tensors or [])
    if alignment != 32:
        metadata.setdefault("general.alignment", alignment)

    payloads: list[bytes] = []
    offsets: list[int] = []
    cursor = 0
    for _name, dims, type_id in tensors:
        nbytes = tensor_nbytes(type_id, dims)
        if nbytes is None:
            raise ValueError(f"unknown test tensor type id {type_id}")
        offsets.append(cursor)
        payloads.append(b"\0" * nbytes)
        cursor += nbytes

    buf = bytearray()
    buf.extend(GGUF_MAGIC)
    buf.extend(struct.pack("<I", GGUF_VERSION))
    buf.extend(struct.pack("<Q", len(tensors)))
    buf.extend(struct.pack("<Q", len(metadata)))

    for key, value in metadata.items():
        buf.extend(pack_string(key))
        buf.extend(pack_metadata_value(value))

    for (name, dims, type_id), offset in zip(tensors, offsets):
        buf.extend(pack_string(name))
        buf.extend(struct.pack("<I", len(dims)))
        for dim in dims:
            buf.extend(struct.pack("<Q", int(dim)))
        buf.extend(struct.pack("<I", int(type_id)))
        buf.extend(struct.pack("<Q", int(offset)))

    padded = align_up(len(buf), alignment)
    if padded > len(buf):
        buf.extend(b"\0" * (padded - len(buf)))
    for payload in payloads:
        buf.extend(payload)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buf))


def minimax_metadata(
    *,
    n_layers: int = 2,
    hidden: int = 8,
    vocab: int = 16,
    experts: int = 4,
    inter: int = 4,
) -> dict[str, Any]:
    return {
        "general.architecture": "minimax-m2",
        "minimax-m2.block_count": n_layers,
        "minimax-m2.embedding_length": hidden,
        "minimax-m2.vocab_size": vocab,
        "minimax-m2.context_length": 128,
        "minimax-m2.attention.head_count": 2,
        "minimax-m2.attention.head_count_kv": 1,
        "minimax-m2.attention.key_length": 4,
        "minimax-m2.attention.value_length": 4,
        "minimax-m2.rope.dimension_count": 4,
        "minimax-m2.rope.freq_base": 5000000.0,
        "minimax-m2.expert_count": experts,
        "minimax-m2.expert_used_count": 2,
        "minimax-m2.expert_feed_forward_length": inter,
        "minimax-m2.expert_gating_func": 2,
        "minimax-m2.attention.layer_norm_rms_epsilon": 0.00001,
    }


def minimax_tensors(
    *,
    n_layers: int = 2,
    hidden: int = 8,
    vocab: int = 16,
    experts: int = 4,
    inter: int = 4,
) -> list[tuple[str, tuple[int, ...], int]]:
    q = 8
    kv = 4
    tensors: list[tuple[str, tuple[int, ...], int]] = [
        ("token_embd.weight", (hidden, vocab), GGML_Q4_K),
        ("output.weight", (hidden, vocab), GGML_Q4_K),
        ("output_norm.weight", (hidden,), GGML_F32),
    ]
    for layer in range(n_layers):
        prefix = f"blk.{layer}"
        tensors.extend(
            [
                (f"{prefix}.attn_q.weight", (hidden, q), GGML_Q5_K),
                (f"{prefix}.attn_q_norm.weight", (q,), GGML_F32),
                (f"{prefix}.attn_k.weight", (hidden, kv), GGML_Q5_K),
                (f"{prefix}.attn_k_norm.weight", (kv,), GGML_F32),
                (f"{prefix}.attn_v.weight", (hidden, kv), GGML_Q5_K),
                (f"{prefix}.attn_output.weight", (q, hidden), GGML_Q5_K),
                (f"{prefix}.attn_norm.weight", (hidden,), GGML_F32),
                (f"{prefix}.ffn_gate_inp.weight", (hidden, experts), GGML_F32),
                (f"{prefix}.exp_probs_b.bias", (experts,), GGML_F32),
                (f"{prefix}.ffn_gate_exps.weight", (hidden, inter, experts), GGML_IQ2_XXS),
                (f"{prefix}.ffn_up_exps.weight", (hidden, inter, experts), GGML_IQ2_XXS),
                (f"{prefix}.ffn_down_exps.weight", (inter, hidden, experts), GGML_IQ2_XXS),
                (f"{prefix}.ffn_norm.weight", (hidden,), GGML_F32),
            ]
        )
    return tensors


def write_minimax_bundle(
    root: Path,
    *,
    n_layers: int = 2,
    hidden: int = 8,
    vocab: int = 16,
    experts: int = 4,
    inter: int = 4,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    write_gguf(
        root / "tiny-minimax-00001-of-00002.gguf",
        metadata={
            **minimax_metadata(n_layers=n_layers, hidden=hidden, vocab=vocab, experts=experts, inter=inter),
            "split.no": 0,
            "split.count": 2,
        },
        tensors=[],
    )
    write_gguf(
        root / "tiny-minimax-00002-of-00002.gguf",
        metadata={"split.no": 1, "split.count": 2},
        tensors=minimax_tensors(n_layers=n_layers, hidden=hidden, vocab=vocab, experts=experts, inter=inter),
    )
    return root
