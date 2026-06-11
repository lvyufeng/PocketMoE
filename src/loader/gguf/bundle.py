from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.loader.gguf.reader import GGML_TYPES, GGUFFile, GGUFReader, GGUFTensorInfo


_SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$")


@dataclass(frozen=True)
class GGUFTensorRef:
    """Tensor header plus the shard that owns its payload.

    Offsets remain shard-local: `absolute_offset` is valid only within
    `shard_path`, exactly like `GGUFTensorInfo.absolute_offset` is valid within a
    single `GGUFFile`.  The bundle is an index, not a concatenated virtual file.
    """

    name: str
    dimensions: tuple[int, ...]
    type_id: int
    offset: int
    absolute_offset: int
    nbytes: int | None
    shard_index: int
    shard_path: str

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
class GGUFShard:
    index: int
    path: str
    file: GGUFFile


@dataclass(frozen=True)
class GGUFBundle:
    """Header-only unified view of one GGUF file or a sharded GGUF bundle."""

    shards: tuple[GGUFShard, ...]
    primary_index: int
    metadata: dict[str, Any]
    tensors: tuple[GGUFTensorRef, ...]
    size: int

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(shard.path for shard in self.shards)

    @property
    def path(self) -> str:
        if len(self.shards) == 1:
            return self.shards[0].path
        return str(Path(self.shards[0].path).parent)

    @property
    def primary(self) -> GGUFFile:
        return self.shards[self.primary_index].file

    @property
    def version(self) -> int:
        return self.primary.version

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    @property
    def metadata_count(self) -> int:
        return len(self.metadata)

    @property
    def tensors_by_name(self) -> dict[str, GGUFTensorRef]:
        return {tensor.name: tensor for tensor in self.tensors}


def _match_shard_name(path: Path) -> re.Match[str] | None:
    return _SHARD_RE.match(path.name)


def _sort_paths(paths: Iterable[Path]) -> list[Path]:
    def key(path: Path):
        match = _match_shard_name(path)
        if match is not None:
            return (match.group("prefix"), int(match.group("index")), path.name)
        return (path.stem, 0, path.name)

    return sorted(paths, key=key)


def _expected_shard_paths(path: Path) -> list[Path] | None:
    match = _match_shard_name(path)
    if match is None:
        return None
    prefix = match.group("prefix")
    count = int(match.group("count"))
    return [path.with_name(f"{prefix}-{idx:05d}-of-{count:05d}.gguf") for idx in range(1, count + 1)]


def resolve_gguf_bundle(path: str | Path) -> tuple[str, ...]:
    """Resolve a single GGUF file, shard path, or directory into shard paths."""

    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(str(root))

    if root.is_dir():
        ggufs = _sort_paths(p for p in root.iterdir() if p.is_file() and p.suffix == ".gguf")
        if not ggufs:
            raise FileNotFoundError(f"no .gguf files found in {root}")
        shard_like = [p for p in ggufs if _match_shard_name(p) is not None]
        if shard_like:
            groups: dict[tuple[str, int], list[Path]] = {}
            for item in shard_like:
                match = _match_shard_name(item)
                assert match is not None
                groups.setdefault((match.group("prefix"), int(match.group("count"))), []).append(item)
            if len(groups) != 1:
                details = ", ".join(f"{prefix} of {count}" for prefix, count in sorted(groups))
                raise ValueError(f"multiple GGUF shard groups found in {root}: {details}")
            (prefix, count), _items = next(iter(groups.items()))
            expected = [root / f"{prefix}-{idx:05d}-of-{count:05d}.gguf" for idx in range(1, count + 1)]
            missing = [str(p) for p in expected if not p.exists()]
            if missing:
                raise FileNotFoundError("missing GGUF shard(s): " + ", ".join(missing))
            return tuple(str(p) for p in expected)
        return tuple(str(p) for p in ggufs)

    if root.suffix != ".gguf":
        raise ValueError(f"expected a .gguf file or directory, got {root}")

    expected = _expected_shard_paths(root)
    if expected is None:
        return (str(root),)
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError("missing GGUF shard(s): " + ", ".join(missing))
    return tuple(str(p) for p in expected)


def _primary_shard_index(shards: list[GGUFShard]) -> int:
    for idx, shard in enumerate(shards):
        if shard.file.metadata.get("split.no") == 0:
            return idx
    for idx, shard in enumerate(shards):
        if "general.architecture" in shard.file.metadata:
            return idx
    return max(range(len(shards)), key=lambda i: shards[i].file.metadata_count)


def read_gguf_bundle(path: str | Path) -> GGUFBundle:
    paths = resolve_gguf_bundle(path)
    shards: list[GGUFShard] = []
    refs: list[GGUFTensorRef] = []
    seen: dict[str, str] = {}
    total_size = 0

    for shard_index, shard_path in enumerate(paths):
        gguf = GGUFReader(shard_path).read()
        shards.append(GGUFShard(shard_index, shard_path, gguf))
        total_size += int(gguf.size)
        for tensor in gguf.tensors:
            if tensor.name in seen:
                raise ValueError(f"duplicate GGUF tensor {tensor.name!r} in {shard_path} and {seen[tensor.name]}")
            seen[tensor.name] = shard_path
            refs.append(
                GGUFTensorRef(
                    name=tensor.name,
                    dimensions=tuple(int(dim) for dim in tensor.dimensions),
                    type_id=int(tensor.type_id),
                    offset=int(tensor.offset),
                    absolute_offset=int(tensor.absolute_offset),
                    nbytes=tensor.nbytes,
                    shard_index=shard_index,
                    shard_path=shard_path,
                )
            )

    if not shards:
        raise ValueError(f"no GGUF shards resolved from {path}")
    primary_index = _primary_shard_index(shards)
    metadata = dict(shards[primary_index].file.metadata)
    return GGUFBundle(
        shards=tuple(shards),
        primary_index=primary_index,
        metadata=metadata,
        tensors=tuple(refs),
        size=total_size,
    )


def bundle_from_file(file: GGUFFile) -> GGUFBundle:
    refs = tuple(
        GGUFTensorRef(
            name=tensor.name,
            dimensions=tuple(int(dim) for dim in tensor.dimensions),
            type_id=int(tensor.type_id),
            offset=int(tensor.offset),
            absolute_offset=int(tensor.absolute_offset),
            nbytes=tensor.nbytes,
            shard_index=0,
            shard_path=file.path,
        )
        for tensor in file.tensors
    )
    return GGUFBundle(
        shards=(GGUFShard(0, file.path, file),),
        primary_index=0,
        metadata=dict(file.metadata),
        tensors=refs,
        size=int(file.size),
    )
