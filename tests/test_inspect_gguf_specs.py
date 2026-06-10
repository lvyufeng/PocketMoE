from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.gguf_test_utils import write_gguf, write_minimax_bundle


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_inspect(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "src.cli.inspect_gguf", *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_inspect_minimax_spec_summary_validation_and_reports(tmp_path: Path) -> None:
    root = write_minimax_bundle(tmp_path / "bundle", n_layers=1)

    result = _run_inspect(
        "--gguf-path",
        str(root),
        "--architecture",
        "auto",
        "--spec-summary",
        "--validate-spec",
        "--capability-report",
        "--placement-report",
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "architecture: minimax-m2" in result.stdout
    assert "layers: 1" in result.stdout
    assert "hidden_size: 8" in result.stdout
    assert "n_routed_experts: 4" in result.stdout
    assert "top_k: 2" in result.stdout
    assert "ok: True" in result.stdout
    assert "[deferred] embedding:q4_k" in result.stdout
    assert "[deferred] attn_q:q5_k" in result.stdout
    assert "[deferred] routed_w1:iq2_xxs" in result.stdout
    assert "[candidate] all_device_lowbit" in result.stdout
    assert "[candidate] heterogeneous_routed_experts" in result.stdout


def test_inspect_minimax_runtime_mapping_fails_clearly(tmp_path: Path) -> None:
    root = write_minimax_bundle(tmp_path / "bundle", n_layers=1)

    result = _run_inspect("--gguf-path", str(root), "--validate-runtime-mapping")

    assert result.returncode == 1
    assert "runtime mapping: FAILED" in result.stdout
    assert "MiniMax" in result.stdout or "minimax" in result.stdout
    assert "--validate-spec" in result.stdout




def test_inspect_minimax_moe_runtime_report(tmp_path: Path) -> None:
    root = write_minimax_bundle(tmp_path / "bundle", n_layers=1)

    result = _run_inspect(
        "--gguf-path",
        str(root),
        "--architecture",
        "minimax-m2",
        "--moe-runtime-report",
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "moe runtime report:" in result.stdout
    assert "minimax_moe_runtime: candidate" in result.stdout
    assert "layers: 1" in result.stdout
    assert "routed tensors: 3 / expected 3" in result.stdout
    assert "moe tensors: 6 / expected 6" in result.stdout
    assert "q4_k embedding/head" in result.stdout
    assert "q5_k attention" in result.stdout
    assert "generation: deferred" in result.stdout


def test_inspect_minimax_check_routed_blocks(tmp_path: Path) -> None:
    root = write_minimax_bundle(tmp_path / "bundle", n_layers=1, hidden=256, inter=256)

    result = _run_inspect(
        "--gguf-path",
        str(root),
        "--architecture",
        "minimax-m2",
        "--check-routed-blocks",
        "--check-layer-limit",
        "1",
        "--check-row-count",
        "1",
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "routed block check:" in result.stdout
    assert "layer=0 role=routed_w1 type=iq2_xxs" in result.stdout
    assert "layer=0 role=routed_w2 type=iq2_xxs" in result.stdout
    assert "layer=0 role=routed_w3 type=iq2_xxs" in result.stdout
    assert "blocks_shape=(1, 1, 66)" in result.stdout
    assert "routed block check: OK" in result.stdout


def test_inspect_legacy_ds4_summary_and_q2_validation_still_work(tmp_path: Path) -> None:
    path = tmp_path / "ds4-empty.gguf"
    write_gguf(
        path,
        metadata={
            "general.architecture": "deepseek4",
            "deepseek4.block_count": 0,
            "deepseek4.embedding_length": 8,
            "deepseek4.expert_count": 4,
            "deepseek4.expert_used_count": 2,
            "deepseek4.expert_feed_forward_length": 4,
        },
        tensors=[],
    )

    result = _run_inspect("--gguf-path", str(path), "--summary", "--validate-ds4-q2")

    assert result.returncode == 0, result.stderr + result.stdout
    assert "metadata.general.architecture: 'deepseek4'" in result.stdout
    assert "tensor_count: 0" in result.stdout
    assert "validation: OK" in result.stdout
