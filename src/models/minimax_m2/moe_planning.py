from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from src.loader.gguf.bundle import GGUFBundle
from src.models.minimax_m2.spec import MiniMaxM2Spec
from src.components.moe.placement import HardwareProfile, heterogeneous_expert_decision, lowbit_device_resident_decision
from src.components.moe.spec import MoEArchitectureParams, PlacementDecision


MINIMAX_M2_ARCHITECTURE = "minimax-m2"
ROUTED_ROLES = frozenset({"routed_w1", "routed_w2", "routed_w3"})
MOE_RUNTIME_ROLES = frozenset({"gate", "gate_bias", "ffn_norm", *ROUTED_ROLES})
_REQUIRED_LAYER_ROLES = tuple(sorted(MOE_RUNTIME_ROLES))
_EXPECTED_ROUTED_TYPE = "iq2_xxs"
_EXPECTED_DENSE_MOE_TYPE = "f32"

@dataclass(frozen=True)
class MiniMaxM2MoETensorPlan:
    source_name: str
    role: str
    layer: int | None
    type_name: str
    dimensions: tuple[int, ...]
    nbytes: int | None
    shard_path: str
    status: str = "candidate"
    reason: str = "available for MiniMax-M2 MoE-only runtime"


@dataclass(frozen=True)
class MiniMaxM2SkippedTensorPlan:
    source_name: str
    role: str
    layer: int | None
    type_name: str
    dimensions: tuple[int, ...]
    nbytes: int | None
    shard_path: str
    status: str = "deferred"
    reason: str = "not part of MiniMax-M2 MoE-only runtime readiness check"


@dataclass(frozen=True)
class MiniMaxM2MoERuntimePlan:
    architecture: str
    status: str
    ok: bool
    params: MoEArchitectureParams
    tensor_plans: tuple[MiniMaxM2MoETensorPlan, ...]
    skipped_tensors: tuple[MiniMaxM2SkippedTensorPlan, ...]
    tensor_role_counts: dict[str, int]
    tensor_type_counts: dict[str, int]
    routed_type_counts: dict[str, int]
    bytes_by_role: dict[str, int]
    bytes_by_type: dict[str, int]
    placements: tuple[PlacementDecision, ...]
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def moe_tensor_count(self) -> int:
        return len(self.tensor_plans)

    @property
    def expected_moe_tensor_count(self) -> int:
        return int(self.params.n_layers) * len(_REQUIRED_LAYER_ROLES)

    @property
    def routed_tensor_count(self) -> int:
        return sum(1 for item in self.tensor_plans if item.role in ROUTED_ROLES)

    @property
    def expected_routed_tensor_count(self) -> int:
        return int(self.params.n_layers) * len(ROUTED_ROLES)

    @property
    def routed_bytes(self) -> int:
        return sum(int(item.nbytes or 0) for item in self.tensor_plans if item.role in ROUTED_ROLES)

    @property
    def moe_bytes(self) -> int:
        return sum(int(item.nbytes or 0) for item in self.tensor_plans)

    def tensor_for(self, layer: int, role: str) -> MiniMaxM2MoETensorPlan:
        for item in self.tensor_plans:
            if item.layer == layer and item.role == role:
                return item
        raise KeyError(f"MiniMax-M2 MoE tensor not planned for layer={layer} role={role}")


@dataclass(frozen=True)
class MiniMaxM2TPRankResidentPlan:
    tp_rank: int
    expert_start: int
    expert_count: int
    routed_resident_bytes: int
    replicated_non_routed_moe_bytes: int
    estimated_bytes_per_gpu: int
    usable_bytes_per_gpu: int
    fits: bool


@dataclass(frozen=True)
class MiniMaxM2TPRoutedResidentPlan:
    architecture: str
    status: str
    ok: bool
    tp_world: int
    gpu_memory_gib: float
    total_tensor_bytes: int
    moe_bytes: int
    routed_bytes: int
    non_routed_moe_bytes: int
    usable_bytes_per_gpu: int
    ranks: tuple[MiniMaxM2TPRankResidentPlan, ...]
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

def _skip_reason(role: str, type_name: str) -> str:
    if role in {"embedding", "lm_head"} and type_name == "q4_k":
        return "q4_k embedding/head payload is handled by the full MiniMax raw-block CUDA runtime"
    if role in {"attn_q", "attn_k", "attn_v", "attn_o"} and type_name == "q5_k":
        return "q5_k attention payload is handled by the full MiniMax raw-block CUDA runtime"
    if role == "attn_norm":
        return "attention normalization belongs to deferred MiniMax GQA runtime"
    if role == "final_norm":
        return "final norm belongs to deferred MiniMax generation runtime"
    return "not required by the MiniMax-M2 MoE-only routed expert readiness path"


def _counts(items: Iterable[MiniMaxM2MoETensorPlan], attr: str) -> dict[str, int]:
    counter: Counter[str] = Counter(str(getattr(item, attr)) for item in items)
    return dict(counter)


def _bytes_by(items: Iterable[MiniMaxM2MoETensorPlan], attr: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        key = str(getattr(item, attr))
        result[key] = result.get(key, 0) + int(item.nbytes or 0)
    return result


def _layer_role_counts(items: Iterable[MiniMaxM2MoETensorPlan]) -> dict[tuple[int, str], int]:
    counts: Counter[tuple[int, str]] = Counter()
    for item in items:
        if item.layer is not None:
            counts[(int(item.layer), item.role)] += 1
    return dict(counts)


def minimax_m2_tp_expert_range(n_experts: int, tp_world: int, tp_rank: int) -> tuple[int, int]:
    n_experts = int(n_experts)
    tp_world = int(tp_world)
    tp_rank = int(tp_rank)
    if n_experts <= 0:
        raise ValueError(f"n_experts must be positive, got {n_experts}")
    if tp_world <= 0:
        raise ValueError(f"tp_world must be positive, got {tp_world}")
    if tp_rank < 0 or tp_rank >= tp_world:
        raise ValueError(f"tp_rank {tp_rank} outside [0, {tp_world})")
    start = (n_experts * tp_rank) // tp_world
    end = (n_experts * (tp_rank + 1)) // tp_world
    return start, end - start



def build_minimax_m2_moe_runtime_plan(
    bundle: GGUFBundle,
    *,
    gpu_count: int = 4,
    gpu_memory_gib: float = 22.0,
) -> MiniMaxM2MoERuntimePlan:
    spec = MiniMaxM2Spec()
    params = spec.parse_params(bundle)
    errors: list[str] = []
    warnings: list[str] = []
    tensor_plans: list[MiniMaxM2MoETensorPlan] = []
    skipped: list[MiniMaxM2SkippedTensorPlan] = []
    names = bundle.tensors_by_name

    if bundle.metadata.get("general.architecture") != MINIMAX_M2_ARCHITECTURE:
        errors.append(
            f"general.architecture expected {MINIMAX_M2_ARCHITECTURE!r}, got {bundle.metadata.get('general.architecture')!r}"
        )

    for mapping in spec.build_tensor_mappings(bundle):
        tensor = names.get(mapping.source_name)
        if tensor is None:
            if mapping.role in MOE_RUNTIME_ROLES:
                errors.append(f"missing MiniMax-M2 MoE tensor {mapping.source_name}")
            continue
        if mapping.role in MOE_RUNTIME_ROLES:
            if mapping.layer is None:
                errors.append(f"MiniMax-M2 MoE tensor {mapping.source_name} has no layer id")
            expected_type = _EXPECTED_ROUTED_TYPE if mapping.role in ROUTED_ROLES else _EXPECTED_DENSE_MOE_TYPE
            status = "candidate"
            reason = "available for MiniMax-M2 MoE-only runtime"
            if tensor.type_name != expected_type:
                status = "unsupported"
                reason = f"expected {expected_type}, got {tensor.type_name}"
                errors.append(f"{mapping.source_name} expected {expected_type}, got {tensor.type_name}")
            tensor_plans.append(
                MiniMaxM2MoETensorPlan(
                    source_name=mapping.source_name,
                    role=mapping.role,
                    layer=mapping.layer,
                    type_name=tensor.type_name,
                    dimensions=tuple(int(dim) for dim in tensor.dimensions),
                    nbytes=tensor.nbytes,
                    shard_path=tensor.shard_path,
                    status=status,
                    reason=reason,
                )
            )
        else:
            skipped.append(
                MiniMaxM2SkippedTensorPlan(
                    source_name=mapping.source_name,
                    role=mapping.role,
                    layer=mapping.layer,
                    type_name=tensor.type_name,
                    dimensions=tuple(int(dim) for dim in tensor.dimensions),
                    nbytes=tensor.nbytes,
                    shard_path=tensor.shard_path,
                    reason=_skip_reason(mapping.role, tensor.type_name),
                )
            )

    layer_counts = _layer_role_counts(tensor_plans)
    for layer in range(params.n_layers):
        for role in _REQUIRED_LAYER_ROLES:
            count = layer_counts.get((layer, role), 0)
            if count != 1:
                errors.append(f"layer {layer} role {role} expected 1 tensor, got {count}")

    if params.n_layers <= 0:
        errors.append("minimax-m2.block_count must be positive for MoE runtime planning")
    if params.n_routed_experts <= 0:
        errors.append("minimax-m2.expert_count must be positive for MoE runtime planning")
    if params.top_k <= 0:
        errors.append("minimax-m2.expert_used_count must be positive for MoE runtime planning")

    role_counts = _counts(tensor_plans, "role")
    type_counts = _counts(tensor_plans, "type_name")
    routed_type_counts = dict(Counter(item.type_name for item in tensor_plans if item.role in ROUTED_ROLES))
    bytes_by_role = _bytes_by(tensor_plans, "role")
    bytes_by_type = _bytes_by(tensor_plans, "type_name")
    moe_bytes = sum(int(item.nbytes or 0) for item in tensor_plans)
    routed_bytes = sum(int(item.nbytes or 0) for item in tensor_plans if item.role in ROUTED_ROLES)
    hardware = HardwareProfile(gpu_count=gpu_count, gpu_memory_gib=gpu_memory_gib)
    placements = (
        lowbit_device_resident_decision(moe_bytes, hardware),
        heterogeneous_expert_decision(routed_bytes),
    )
    status = "candidate" if not errors else "failed"
    if skipped:
        warnings.append(
            f"{len(skipped)} dense/non-MoE tensors are intentionally skipped for MiniMax-M2 MoE-only readiness"
        )
    return MiniMaxM2MoERuntimePlan(
        architecture=MINIMAX_M2_ARCHITECTURE,
        status=status,
        ok=not errors,
        params=params,
        tensor_plans=tuple(tensor_plans),
        skipped_tensors=tuple(skipped),
        tensor_role_counts=role_counts,
        tensor_type_counts=type_counts,
        routed_type_counts=routed_type_counts,
        bytes_by_role=bytes_by_role,
        bytes_by_type=bytes_by_type,
        placements=placements,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )



def build_minimax_m2_tp_routed_resident_plan(
    bundle: GGUFBundle,
    *,
    tp_world: int = 4,
    gpu_memory_gib: float = 22.0,
    reserve_fraction: float = 0.15,
) -> MiniMaxM2TPRoutedResidentPlan:
    runtime_plan = build_minimax_m2_moe_runtime_plan(
        bundle,
        gpu_count=tp_world,
        gpu_memory_gib=gpu_memory_gib,
    )
    errors = list(runtime_plan.errors)
    warnings = list(runtime_plan.warnings)
    tp_world = int(tp_world)
    if tp_world <= 0:
        errors.append(f"tp_world must be positive, got {tp_world}")
        tp_world = 1
    if runtime_plan.params.n_routed_experts % tp_world != 0:
        warnings.append(
            f"n_routed_experts={runtime_plan.params.n_routed_experts} is not divisible by tp_world={tp_world}; "
            "expert ranges will differ by at most one expert"
        )
    per_gpu_bytes = int(float(gpu_memory_gib) * (1024 ** 3))
    usable_bytes = int(per_gpu_bytes * (1.0 - float(reserve_fraction)))
    routed_bytes = runtime_plan.routed_bytes
    moe_bytes = runtime_plan.moe_bytes
    non_routed_moe_bytes = max(0, moe_bytes - routed_bytes)
    ranks: list[MiniMaxM2TPRankResidentPlan] = []
    for tp_rank in range(tp_world):
        expert_start, expert_count = minimax_m2_tp_expert_range(runtime_plan.params.n_routed_experts, tp_world, tp_rank)
        routed_rank_bytes = 0
        for item in runtime_plan.tensor_plans:
            if item.role not in ROUTED_ROLES:
                continue
            if item.nbytes is None:
                continue
            routed_rank_bytes += (int(item.nbytes) * int(expert_count) + runtime_plan.params.n_routed_experts - 1) // runtime_plan.params.n_routed_experts
        estimated = routed_rank_bytes + non_routed_moe_bytes
        ranks.append(
            MiniMaxM2TPRankResidentPlan(
                tp_rank=tp_rank,
                expert_start=expert_start,
                expert_count=expert_count,
                routed_resident_bytes=routed_rank_bytes,
                replicated_non_routed_moe_bytes=non_routed_moe_bytes,
                estimated_bytes_per_gpu=estimated,
                usable_bytes_per_gpu=usable_bytes,
                fits=estimated <= usable_bytes,
            )
        )
    if any(not rank.fits for rank in ranks):
        errors.append(
            f"MiniMax-M2 TP routed-resident plan exceeds usable per-GPU memory budget {usable_bytes} bytes"
        )
    status = "candidate" if not errors else "failed"
    return MiniMaxM2TPRoutedResidentPlan(
        architecture=MINIMAX_M2_ARCHITECTURE,
        status=status,
        ok=not errors,
        tp_world=tp_world,
        gpu_memory_gib=float(gpu_memory_gib),
        total_tensor_bytes=sum(int(t.nbytes or 0) for t in bundle.tensors),
        moe_bytes=moe_bytes,
        routed_bytes=routed_bytes,
        non_routed_moe_bytes=non_routed_moe_bytes,
        usable_bytes_per_gpu=usable_bytes,
        ranks=tuple(ranks),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
