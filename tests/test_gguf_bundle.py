from __future__ import annotations

from pathlib import Path

import pytest

from src.loader.gguf.bundle import read_gguf_bundle, resolve_gguf_bundle
from tests.gguf_test_utils import GGML_F32, write_gguf, write_minimax_bundle


def test_single_file_bundle(tmp_path: Path) -> None:
    path = tmp_path / "single.gguf"
    write_gguf(
        path,
        metadata={"general.architecture": "deepseek4", "deepseek4.block_count": 1},
        tensors=[("token_embd.weight", (4, 8), GGML_F32)],
    )

    bundle = read_gguf_bundle(path)

    assert bundle.paths == (str(path.resolve()),)
    assert bundle.path == str(path.resolve())
    assert bundle.version == 3
    assert bundle.tensor_count == 1
    assert bundle.metadata["general.architecture"] == "deepseek4"
    assert bundle.tensors_by_name["token_embd.weight"].shard_index == 0


def test_directory_shard_resolution_and_metadata_only_primary(tmp_path: Path) -> None:
    root = write_minimax_bundle(tmp_path / "bundle", n_layers=1)

    paths = resolve_gguf_bundle(root)
    bundle = read_gguf_bundle(root)

    assert len(paths) == 2
    assert paths[0].endswith("tiny-minimax-00001-of-00002.gguf")
    assert bundle.primary_index == 0
    assert bundle.metadata["general.architecture"] == "minimax-m2"
    assert bundle.tensor_count == 16
    assert "blk.0.ffn_gate_exps.weight" in bundle.tensors_by_name
    assert bundle.tensors_by_name["blk.0.ffn_gate_exps.weight"].shard_index == 1


def test_shard_path_resolves_all_shards(tmp_path: Path) -> None:
    root = write_minimax_bundle(tmp_path / "bundle", n_layers=1)
    second = root / "tiny-minimax-00002-of-00002.gguf"

    paths = resolve_gguf_bundle(second)

    assert len(paths) == 2
    assert paths[0].endswith("tiny-minimax-00001-of-00002.gguf")
    assert paths[1].endswith("tiny-minimax-00002-of-00002.gguf")


def test_missing_shard_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    write_gguf(
        root / "broken-00001-of-00002.gguf",
        metadata={"general.architecture": "minimax-m2", "split.no": 0, "split.count": 2},
        tensors=[],
    )

    with pytest.raises(FileNotFoundError, match="missing GGUF shard"):
        resolve_gguf_bundle(root)


def test_duplicate_tensor_detection(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    for idx in (1, 2):
        write_gguf(
            root / f"dup-{idx:05d}-of-00002.gguf",
            metadata={"general.architecture": "minimax-m2", "split.no": idx - 1, "split.count": 2},
            tensors=[("same.weight", (4,), GGML_F32)],
        )

    with pytest.raises(ValueError, match="duplicate GGUF tensor"):
        read_gguf_bundle(root)
