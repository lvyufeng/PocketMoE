from __future__ import annotations

import numpy as np

from src.gguf.imatrix import _rank_output_path, _write_imatrix_gguf, merge_imatrix_files
from src.gguf.tensor_reader import GGUFTensorDataReader


def test_layer_capture_helper_respects_collector_filters(monkeypatch) -> None:
    import src.gguf.imatrix as imatrix
    from src.runtime.moe.cpu_backend import _layer_imatrix_capture_enabled

    monkeypatch.setenv("DEEPSEEK_IMATRIX_CAPTURE", "1")
    monkeypatch.setenv("DEEPSEEK_IMATRIX_LAYERS", "0")
    monkeypatch.setenv("DEEPSEEK_IMATRIX_ROLES", "ffn_gate_exps")
    monkeypatch.setattr(imatrix, "_COLLECTOR", None)

    assert _layer_imatrix_capture_enabled(0)
    assert not _layer_imatrix_capture_enabled(1)
    assert not _layer_imatrix_capture_enabled(None)


def test_rank_output_path_adds_rank_suffix_for_tp(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")

    assert _rank_output_path("/tmp/calib.gguf") == "/tmp/calib.rank2.gguf"
    assert _rank_output_path("/tmp/calib") == "/tmp/calib.rank2.gguf"
    assert _rank_output_path("/tmp/calib.{rank}.gguf") == "/tmp/calib.2.gguf"
    assert _rank_output_path("/tmp/calib.local{local_rank}.gguf") == "/tmp/calib.local2.gguf"


def test_merge_imatrix_files_sums_rank_shards(tmp_path) -> None:
    name = "blk.0.ffn_gate_exps.weight"
    sum_a = np.zeros((4, 8), dtype=np.float32)
    sum_b = np.zeros((4, 8), dtype=np.float32)
    counts_a = np.zeros((4,), dtype=np.float32)
    counts_b = np.zeros((4,), dtype=np.float32)

    sum_a[0] = np.arange(8, dtype=np.float32)
    sum_a[2] = 1.5
    counts_a[0] = 3
    counts_a[2] = 5

    sum_b[1] = 2.0
    sum_b[2] = 0.5
    counts_b[1] = 7
    counts_b[2] = 11

    shard_a = tmp_path / "calib.rank0.gguf"
    shard_b = tmp_path / "calib.rank1.gguf"
    merged = tmp_path / "calib.gguf"
    _write_imatrix_gguf(str(shard_a), [(name, sum_a, counts_a)], dataset="rank0", chunk_size=4, chunk_count=1)
    _write_imatrix_gguf(str(shard_b), [(name, sum_b, counts_b)], dataset="rank1", chunk_size=4, chunk_count=2)

    merge_imatrix_files([str(shard_a), str(shard_b)], str(merged), dataset="merged")

    with GGUFTensorDataReader(str(merged)) as reader:
        got_sum = reader.read_tensor(f"{name}.in_sum2").numpy()
        got_counts = reader.read_tensor(f"{name}.counts").numpy().reshape(-1)
        assert reader.gguf.metadata["general.type"] == "imatrix"
        assert reader.gguf.metadata["imatrix.chunk_size"] == 4
        assert reader.gguf.metadata["imatrix.chunk_count"] == 4

    np.testing.assert_allclose(got_sum, sum_a + sum_b)
    np.testing.assert_allclose(got_counts, counts_a + counts_b)


def test_merge_imatrix_files_uses_nonzero_chunk_count_for_partial_chunk(tmp_path) -> None:
    name = "blk.0.ffn_gate_exps.weight"
    sum_a = np.ones((2, 4), dtype=np.float32)
    counts_a = np.array([1, 3], dtype=np.float32)

    shard = tmp_path / "calib.rank0.gguf"
    merged = tmp_path / "calib.gguf"
    _write_imatrix_gguf(str(shard), [(name, sum_a, counts_a)], dataset="rank0", chunk_size=4, chunk_count=1)

    merge_imatrix_files([str(shard)], str(merged), dataset="merged")

    with GGUFTensorDataReader(str(merged)) as reader:
        assert reader.gguf.metadata["imatrix.chunk_size"] == 4
        assert reader.gguf.metadata["imatrix.chunk_count"] == 1
