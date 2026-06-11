from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.gguf.bundle import GGUFBundle


@dataclass(frozen=True)
class MoEArchitectureParams:
    architecture: str
    n_layers: int
    hidden_size: int
    vocab_size: int
    context_length: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rope_dim: int | None
    rope_base: float | None
    n_routed_experts: int
    top_k: int
    expert_intermediate_size: int
    n_shared_experts: int = 0
    gate_function: str | None = None
    attention_kind: str = "unknown"
    routed_expert_layout: str = "packed_3d"
    norm_eps: float | None = None


@dataclass(frozen=True)
class TensorMapping:
    source_name: str
    logical_name: str
    role: str
    transform: str = "direct"
    layer: int | None = None
    expert: int | None = None
    required: bool = True


@dataclass(frozen=True)
class SpecValidation:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mapped_sources: int = 0
    unmapped_sources: list[str] = field(default_factory=list)
    role_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityItem:
    name: str
    status: str
    reason: str


@dataclass(frozen=True)
class PlacementDecision:
    name: str
    status: str
    reason: str
    estimated_bytes: int | None = None
    estimated_bytes_per_gpu: int | None = None


@dataclass(frozen=True)
class CapabilityReport:
    architecture: str
    params: MoEArchitectureParams
    tensor_type_counts: dict[str, int]
    tensor_role_counts: dict[str, int]
    bytes_by_type: dict[str, int]
    bytes_by_role: dict[str, int]
    capabilities: list[CapabilityItem]
    placements: list[PlacementDecision]


class MoEModelSpec(Protocol):
    architecture: str
    display_name: str

    def parse_params(self, bundle: GGUFBundle) -> MoEArchitectureParams:
        ...

    def classify_tensor(self, name: str) -> str:
        ...

    def build_tensor_mappings(self, bundle: GGUFBundle) -> list[TensorMapping]:
        ...

    def validate_bundle(self, bundle: GGUFBundle) -> SpecValidation:
        ...

    def capability_report(self, bundle: GGUFBundle, *, gpu_count: int = 4, gpu_memory_gib: float = 22.0) -> CapabilityReport:
        ...


def metadata_value(metadata: dict[str, Any], key: str, default: Any = None) -> Any:
    return metadata.get(key, default)


def metadata_int(metadata: dict[str, Any], key: str, default: int = 0) -> int:
    value = metadata.get(key, default)
    try:
        if hasattr(value, "length"):
            return int(value.length)
        return int(value)
    except Exception:
        return int(default)


def metadata_float(metadata: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metadata.get(key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def bytes_by(items, key_fn) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        key = str(key_fn(item))
        result[key] = result.get(key, 0) + int(item.nbytes or 0)
    return result


def counts_by(items, key_fn) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        key = str(key_fn(item))
        result[key] = result.get(key, 0) + 1
    return result
