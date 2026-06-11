from __future__ import annotations

from dataclasses import dataclass

from src.models.moe.spec import PlacementDecision


@dataclass(frozen=True)
class HardwareProfile:
    gpu_count: int = 4
    gpu_memory_gib: float = 22.0
    name: str = "consumer-gpu-box"

    @property
    def total_gpu_bytes(self) -> int:
        return int(self.gpu_count * self.gpu_memory_gib * (1024 ** 3))

    @property
    def per_gpu_bytes(self) -> int:
        return int(self.gpu_memory_gib * (1024 ** 3))


def estimate_even_shard(total_bytes: int, gpu_count: int) -> int:
    if gpu_count <= 0:
        return int(total_bytes)
    return (int(total_bytes) + int(gpu_count) - 1) // int(gpu_count)


def lowbit_device_resident_decision(total_bytes: int, hardware: HardwareProfile, *, reserve_fraction: float = 0.15) -> PlacementDecision:
    per_gpu = estimate_even_shard(total_bytes, hardware.gpu_count)
    usable = int(hardware.per_gpu_bytes * (1.0 - reserve_fraction))
    if per_gpu <= usable:
        return PlacementDecision(
            name="all_device_lowbit",
            status="candidate",
            reason=f"even sharding uses {per_gpu} bytes/GPU within reserved budget {usable} bytes/GPU",
            estimated_bytes=int(total_bytes),
            estimated_bytes_per_gpu=per_gpu,
        )
    return PlacementDecision(
        name="all_device_lowbit",
        status="deferred",
        reason=f"even sharding uses {per_gpu} bytes/GPU above reserved budget {usable} bytes/GPU",
        estimated_bytes=int(total_bytes),
        estimated_bytes_per_gpu=per_gpu,
    )


def heterogeneous_expert_decision(routed_bytes: int) -> PlacementDecision:
    return PlacementDecision(
        name="heterogeneous_routed_experts",
        status="candidate",
        reason="routed experts can be kept in CPU pinned/NUMA memory and staged by active routes when all-device placement is not practical",
        estimated_bytes=int(routed_bytes),
        estimated_bytes_per_gpu=None,
    )
