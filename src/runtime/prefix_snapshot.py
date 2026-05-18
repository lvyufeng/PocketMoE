"""Coarse-grained KV prefix snapshot cache (fastllm-style).

Stores a process-local LRU of per-layer KV state at fixed token boundaries.
The cache is off by default.
"""

from __future__ import annotations

import hashlib
import os
import struct
import threading
import time
from array import array
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


_DEF_BLOCK = 256
_DEF_MAX_ENTRIES = 8


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class SnapshotEntry:
    pos: int
    token_ids: list[int]
    buffers: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    age: int = 0


class PrefixSnapshotCache:
    _instance: Optional["PrefixSnapshotCache"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[bytes, SnapshotEntry] = OrderedDict()
        self._tick = 0
        self.block = max(1, _env_int("DEEPSEEK_PREFIX_SNAPSHOT_BLOCK", _DEF_BLOCK))
        self.max_entries = max(1, _env_int("DEEPSEEK_PREFIX_SNAPSHOT_MAX_ENTRIES", _DEF_MAX_ENTRIES))
        self.min_tokens = max(self.block, _env_int("DEEPSEEK_PREFIX_SNAPSHOT_MIN_TOKENS", self.block))
        self.rank = int(os.getenv("RANK", "0"))

    @classmethod
    def enabled(cls) -> bool:
        return _env_flag("DEEPSEEK_PREFIX_SNAPSHOT")

    @classmethod
    def instance(cls) -> "PrefixSnapshotCache":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = PrefixSnapshotCache()
            return cls._instance

    def _log(self, msg: str) -> None:
        if self.rank == 0:
            print(f"prefix_snapshot {msg}", flush=True)

    @staticmethod
    def _hash(token_ids: list[int], n: int) -> bytes:
        h = hashlib.blake2b(digest_size=24)
        h.update(struct.pack("<Q", n))
        h.update(array("I", token_ids[:n]).tobytes())
        return h.digest()

    def lookup_hint(self, token_ids: list[int]) -> Optional[dict[str, Any]]:
        if not token_ids:
            return None
        prompt_len = len(token_ids)
        capture_pos = (prompt_len // self.block) * self.block
        if capture_pos < self.min_tokens:
            return None
        restore_start = ((prompt_len - 1) // self.block) * self.block
        match_pos = 0
        match_key: Optional[bytes] = None
        with self._lock:
            n = restore_start
            while n >= self.block:
                key = self._hash(token_ids, n)
                entry = self._entries.get(key)
                if entry is not None and entry.token_ids == token_ids[:n]:
                    self._entries.move_to_end(key)
                    self._tick += 1
                    entry.age = self._tick
                    match_pos = n
                    match_key = key
                    break
                n -= self.block
        if match_pos > 0:
            self._log(f"hit pos={match_pos} prompt_len={prompt_len} capture_pos={capture_pos}")
        else:
            self._log(f"miss prompt_len={prompt_len} longest_match=0 capture_pos={capture_pos}")
        capture_all = _env_flag("DEEPSEEK_PREFIX_SNAPSHOT_CAPTURE_ALL", "0")
        capture_positions = list(range(self.block, capture_pos + 1, self.block)) if capture_all else [capture_pos]
        return {
            "match_key": match_key.hex() if match_key else None,
            "match_pos": match_pos,
            "capture_positions": capture_positions,
        }

    def restore(self, model: Any, hint: Optional[dict[str, Any]], token_ids: list[int]) -> int:
        if not hint:
            return 0
        match_key_hex = hint.get("match_key")
        match_pos = int(hint.get("match_pos") or 0)
        if not match_key_hex or match_pos <= 0:
            return 0
        try:
            key = bytes.fromhex(match_key_hex)
        except ValueError:
            return 0
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.pos != match_pos or entry.pos >= len(token_ids) or entry.token_ids != token_ids[:entry.pos]:
                return 0
            self._entries.move_to_end(key)
            self._tick += 1
            entry.age = self._tick
        self._apply_buffers(model, entry.buffers)
        return entry.pos

    def capture(self, model: Any, token_ids: list[int], pos: int) -> None:
        if pos <= 0 or pos > len(token_ids) or pos % self.block != 0:
            return
        if pos < self.min_tokens:
            return
        key = self._hash(token_ids, pos)
        token_prefix = list(token_ids[:pos])
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None and existing.token_ids == token_prefix:
                self._entries.move_to_end(key)
                self._tick += 1
                existing.age = self._tick
                return
        buffers = self._clone_buffers(model, pos)
        entry = SnapshotEntry(pos=pos, token_ids=token_prefix, buffers=buffers)
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None and existing.token_ids == token_prefix:
                self._entries.move_to_end(key)
                self._tick += 1
                existing.age = self._tick
                return
            self._tick += 1
            entry.age = self._tick
            self._entries[key] = entry
            while len(self._entries) > self.max_entries:
                _, evicted_entry = self._entries.popitem(last=False)
                self._log(f"evict pos={evicted_entry.pos} age={evicted_entry.age}")
        self._log(f"capture pos={pos} prompt_len={len(token_ids)} entries={len(self._entries)}")

    @staticmethod
    def _layer_buffers(layer: Any) -> list[tuple[str, torch.Tensor]]:
        attn = getattr(layer, "attn", None)
        if attn is None:
            return []
        buffers: list[tuple[str, torch.Tensor]] = []
        kv_cache = getattr(attn, "kv_cache", None)
        if isinstance(kv_cache, torch.Tensor):
            buffers.append(("kv_cache", kv_cache))
        compressor = getattr(attn, "compressor", None)
        if compressor is not None:
            for name in ("kv_state", "score_state"):
                tensor = getattr(compressor, name, None)
                if isinstance(tensor, torch.Tensor):
                    buffers.append((f"compressor.{name}", tensor))
        indexer = getattr(attn, "indexer", None)
        if indexer is not None:
            kv_cache = getattr(indexer, "kv_cache", None)
            if isinstance(kv_cache, torch.Tensor):
                buffers.append(("indexer.kv_cache", kv_cache))
            indexer_compressor = getattr(indexer, "compressor", None)
            if indexer_compressor is not None:
                for name in ("kv_state", "score_state"):
                    tensor = getattr(indexer_compressor, name, None)
                    if isinstance(tensor, torch.Tensor):
                        buffers.append((f"indexer.compressor.{name}", tensor))
        return buffers

    @staticmethod
    def _snapshot_view(name: str, tensor: torch.Tensor, pos: int, layer: Any) -> torch.Tensor:
        if tensor.dim() < 2:
            return tensor.detach().clone()
        if name == "kv_cache":
            attn = getattr(layer, "attn", None)
            win = int(getattr(attn, "window_size", 0) or 0)
            compress_ratio = int(getattr(attn, "compress_ratio", 0) or 0)
            length = win + (pos // compress_ratio if compress_ratio > 0 else 0)
            length = min(max(length, 0), tensor.size(1))
            return tensor[:1, :length].detach().clone()
        if name == "indexer.kv_cache":
            attn = getattr(layer, "attn", None)
            indexer = getattr(attn, "indexer", None)
            compress_ratio = int(getattr(indexer, "compress_ratio", 0) or 0)
            length = pos // compress_ratio if compress_ratio > 0 else tensor.size(1)
            length = min(max(length, 0), tensor.size(1))
            return tensor[:1, :length].detach().clone()
        return tensor[:1].detach().clone()

    def _clone_buffers(self, model: Any, pos: int) -> dict[int, dict[str, torch.Tensor]]:
        clones: dict[int, dict[str, torch.Tensor]] = {}
        layers = getattr(model, "layers", None)
        if layers is None:
            return clones
        profile = _env_flag("DEEPSEEK_PREFIX_SNAPSHOT_PROFILE", "0")
        timings: dict[str, float] = {}
        sizes: dict[str, int] = {}
        for idx, layer in enumerate(layers):
            if layer is None:
                continue
            tensors = self._layer_buffers(layer)
            if not tensors:
                continue
            layer_clones: dict[str, torch.Tensor] = {}
            for name, tensor in tensors:
                if profile and tensor.is_cuda:
                    torch.cuda.synchronize(tensor.device)
                t0 = time.perf_counter()
                clone = self._snapshot_view(name, tensor, pos, layer)
                if profile and clone.is_cuda:
                    torch.cuda.synchronize(clone.device)
                dt = time.perf_counter() - t0
                layer_clones[name] = clone
                if profile:
                    timings[name] = timings.get(name, 0.0) + dt
                    sizes[name] = sizes.get(name, 0) + clone.numel() * clone.element_size()
            clones[idx] = layer_clones
        if profile:
            parts = [f"{name}={timings[name]:.3f}s/{sizes.get(name, 0)/1048576:.1f}MiB" for name in sorted(timings)]
            self._log(f"capture_profile pos={pos} " + " ".join(parts))
        return clones

    @staticmethod
    def _apply_buffers(model: Any, buffers: dict[int, dict[str, torch.Tensor]]) -> None:
        layers = getattr(model, "layers", None)
        if layers is None:
            return
        for idx, snapshot in buffers.items():
            if idx >= len(layers):
                continue
            layer = layers[idx]
            if layer is None:
                continue
            for name, src in snapshot.items():
                dst = _resolve_buffer(layer, name)
                if dst is None or dst.dim() != src.dim():
                    continue
                if any(src_size > dst_size for src_size, dst_size in zip(src.shape, dst.shape)):
                    continue
                slices = tuple(slice(0, size) for size in src.shape)
                dst[slices].copy_(src)
            _refresh_aliases(layer)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def _resolve_buffer(layer: Any, name: str) -> Optional[torch.Tensor]:
    attn = getattr(layer, "attn", None)
    if attn is None:
        return None
    parts = name.split(".")
    obj: Any = attn
    for part in parts[:-1]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    tensor = getattr(obj, parts[-1], None)
    return tensor if isinstance(tensor, torch.Tensor) else None


def _refresh_aliases(layer: Any) -> None:
    attn = getattr(layer, "attn", None)
    if attn is None:
        return
    compressor = getattr(attn, "compressor", None)
    if compressor is not None and getattr(compressor, "kv_cache", None) is None:
        win = int(getattr(attn, "window_size", 0) or 0)
        if win > 0:
            compressor.kv_cache = attn.kv_cache[:, win:]
            compressor.freqs_cis = getattr(attn, "freqs_cis", None)
    indexer = getattr(attn, "indexer", None)
    if indexer is None:
        return
    if getattr(indexer, "freqs_cis", None) is None:
        indexer.freqs_cis = getattr(attn, "freqs_cis", None)
    indexer_compressor = getattr(indexer, "compressor", None)
    if indexer_compressor is not None and getattr(indexer_compressor, "kv_cache", None) is None:
        indexer_compressor.kv_cache = indexer.kv_cache
        indexer_compressor.freqs_cis = getattr(indexer, "freqs_cis", None)
