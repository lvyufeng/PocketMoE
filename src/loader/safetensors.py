"""Safetensors checkpoint index and shard reader helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Iterator

from safetensors import safe_open


@dataclass(frozen=True)
class SafetensorsIndex:
    root: str
    weight_map: dict[str, str]
    file_to_keys: dict[str, list[str]]

    def shard_path(self, file_name: str) -> str:
        return os.path.join(self.root, file_name)


def group_weight_map_by_file(weight_map: dict[str, str]) -> dict[str, list[str]]:
    file_to_keys: dict[str, list[str]] = {}
    for key, file_name in weight_map.items():
        file_to_keys.setdefault(file_name, []).append(key)
    return file_to_keys


def read_safetensors_index(root: str) -> SafetensorsIndex:
    root = os.path.abspath(root)
    weight_map_path = os.path.join(root, "model.safetensors.index.json")
    with open(weight_map_path) as f:
        weight_map = json.load(f)["weight_map"]
    return SafetensorsIndex(root=root, weight_map=weight_map, file_to_keys=group_weight_map_by_file(weight_map))


class SafetensorsShardReader:
    def __init__(self, path: str, *, device: str = "cpu") -> None:
        self.path = path
        self.device = device
        self._reader = None

    def __enter__(self):
        self._reader = safe_open(self.path, framework="pt", device=self.device)
        return self._reader.__enter__()

    def __exit__(self, exc_type, exc, tb):
        assert self._reader is not None
        return self._reader.__exit__(exc_type, exc, tb)


def iter_safetensors_shards(
    index: SafetensorsIndex,
    file_to_keys: dict[str, list[str]] | None = None,
    *,
    device: str = "cpu",
) -> Iterator[tuple[int, int, str, list[str], object]]:
    selected = file_to_keys if file_to_keys is not None else index.file_to_keys
    total_files = len(selected)
    for file_idx, (file_name, keys) in enumerate(selected.items(), 1):
        file_path = index.shard_path(file_name)
        with safe_open(file_path, framework="pt", device=device) as reader:
            yield file_idx, total_files, file_name, keys, reader


def filter_file_to_keys(
    file_to_keys: dict[str, list[str]],
    predicate,
) -> dict[str, list[str]]:
    filtered: dict[str, list[str]] = {}
    for file_name, keys in file_to_keys.items():
        selected = [key for key in keys if predicate(key)]
        if selected:
            filtered[file_name] = selected
    return filtered
