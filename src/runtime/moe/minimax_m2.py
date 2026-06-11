from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.gguf.bundle import GGUFBundle, read_gguf_bundle
from src.gguf.tensor_reader import GGUFTensorDataReader
from src.models.minimax_m2.moe_planning import (
    MINIMAX_M2_ARCHITECTURE,
    ROUTED_ROLES,
    MiniMaxM2MoERuntimePlan,
    MiniMaxM2MoETensorPlan,
    build_minimax_m2_moe_runtime_plan,
)


GGUF_DEVICE_TYPE_IDS = {
    "iq2_xxs": 0,
    "q2_k": 1,
    "iq1_m": 2,
}
SUPPORTED_DEVICE_ROUTED_TYPES = frozenset(GGUF_DEVICE_TYPE_IDS)
_ROUTED_ROLE_ATTRS = {
    "routed_w1": "w1",
    "routed_w2": "w2",
    "routed_w3": "w3",
}

@dataclass(frozen=True)
class MiniMaxM2DeviceRoutedTensor:
    source_name: str
    role: str
    layer: int
    blocks: torch.Tensor
    type_name: str
    type_id: int
    in_dim: int
    out_dim: int
    expert_start: int
    expert_count: int
    device: torch.device

    @property
    def nbytes(self) -> int:
        return int(self.blocks.numel() * self.blocks.element_size())


@dataclass(frozen=True)
class MiniMaxM2DeviceRoutedLayer:
    layer: int
    w1: MiniMaxM2DeviceRoutedTensor
    w3: MiniMaxM2DeviceRoutedTensor
    w2: MiniMaxM2DeviceRoutedTensor
    device: torch.device


@dataclass(frozen=True)
class MiniMaxM2CudaGemmSmokeResult:
    layer: int
    role: str
    expert: int
    type_name: str
    type_id: int
    device: torch.device
    input_shape: tuple[int, ...]
    blocks_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_dtype: torch.dtype
    finite: bool
    resident_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "role": self.role,
            "expert": self.expert,
            "type_name": self.type_name,
            "type_id": self.type_id,
            "device": str(self.device),
            "input_shape": self.input_shape,
            "blocks_shape": self.blocks_shape,
            "output_shape": self.output_shape,
            "output_dtype": str(self.output_dtype),
            "finite": self.finite,
            "resident_bytes": self.resident_bytes,
        }

class MiniMaxM2RoutedBlockLoader:
    """Lazy raw-block reader for MiniMax-M2 routed expert tensors.

    This loader is intentionally MoE-only.  It never attempts to read q4_k/q5_k
    dense tensors and it opens each GGUF shard lazily only when a routed block
    read is requested.
    """

    def __init__(self, bundle_or_path: GGUFBundle | str | Path, *, plan: MiniMaxM2MoERuntimePlan | None = None):
        self.bundle = read_gguf_bundle(bundle_or_path) if not isinstance(bundle_or_path, GGUFBundle) else bundle_or_path
        self.plan = plan or build_minimax_m2_moe_runtime_plan(self.bundle)
        if not self.plan.ok:
            raise ValueError(f"MiniMax-M2 MoE runtime plan is not valid: {list(self.plan.errors[:10])}")
        self._readers: dict[str, GGUFTensorDataReader] = {}

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> "MiniMaxM2RoutedBlockLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _reader(self, shard_path: str) -> GGUFTensorDataReader:
        reader = self._readers.get(shard_path)
        if reader is None:
            reader = GGUFTensorDataReader(shard_path)
            self._readers[shard_path] = reader
        return reader

    def _tensor_plan(self, layer: int, role: str) -> MiniMaxM2MoETensorPlan:
        if role not in ROUTED_ROLES:
            raise ValueError(f"role must be one of {sorted(ROUTED_ROLES)}, got {role!r}")
        if layer < 0 or layer >= self.plan.params.n_layers:
            raise ValueError(f"layer {layer} outside MiniMax-M2 range [0, {self.plan.params.n_layers})")
        return self.plan.tensor_for(layer, role)

    def read_layer_role_blocks(
        self,
        layer: int,
        role: str,
        *,
        expert_start: int = 0,
        expert_count: int | None = None,
    ):
        item = self._tensor_plan(int(layer), role)
        return self._reader(item.shard_path).read_routed_layer_blocks(
            item.source_name,
            expert_start=int(expert_start),
            expert_count=expert_count,
        )

    def read_expert_role_blocks(
        self,
        layer: int,
        role: str,
        *,
        expert: int,
        row_start: int = 0,
        row_count: int | None = None,
    ):
        item = self._tensor_plan(int(layer), role)
        return self._reader(item.shard_path).read_routed_expert_blocks(
            item.source_name,
            expert=int(expert),
            row_start=int(row_start),
            row_count=row_count,
        )


def _canonical_cuda_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"MiniMax-M2 device-resident cache requires a CUDA device, got {resolved}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for MiniMax-M2 device-resident cache")
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _gguf_cuda_quant_grid(type_name: str, device: torch.device) -> torch.Tensor:
    if type_name == "iq2_xxs":
        from src.gguf.tensor_reader import get_iq2xxs_signed_grid_tensor

        return get_iq2xxs_signed_grid_tensor().to(device=device, non_blocking=False).contiguous()
    if type_name == "iq1_m":
        from src.gguf.tensor_reader import get_iq1_grid_tensor

        return get_iq1_grid_tensor().to(device=device, non_blocking=False).contiguous()
    return torch.empty(0, dtype=torch.int8, device=device)


class MiniMaxM2DeviceResidentCache:
    """CUDA-resident MiniMax-M2 routed expert block cache.

    This is a deliberately small all-device building block: it copies raw GGUF
    routed expert blocks into CUDA memory and exposes a smoke path that feeds
    those blocks into the existing raw GGUF GEMM extension.  It does not run
    MiniMax attention, routing, or generation.
    """

    def __init__(
        self,
        bundle_or_path: GGUFBundle | str | Path,
        *,
        device: str | torch.device = "cuda",
        plan: MiniMaxM2MoERuntimePlan | None = None,
        expert_start: int = 0,
        expert_count: int | None = None,
    ):
        self.device = _canonical_cuda_device(device)
        self.bundle = read_gguf_bundle(bundle_or_path) if not isinstance(bundle_or_path, GGUFBundle) else bundle_or_path
        self.plan = plan or build_minimax_m2_moe_runtime_plan(self.bundle)
        if not self.plan.ok:
            raise ValueError(f"MiniMax-M2 MoE runtime plan is not valid: {list(self.plan.errors[:10])}")
        self.expert_start = int(expert_start)
        if self.expert_start < 0:
            raise ValueError(f"expert_start must be non-negative, got {expert_start}")
        available = int(self.plan.params.n_routed_experts) - self.expert_start
        if available < 0:
            raise ValueError(
                f"expert_start {self.expert_start} outside MiniMax-M2 expert count {self.plan.params.n_routed_experts}"
            )
        self.expert_count = available if expert_count is None else int(expert_count)
        if self.expert_count <= 0 or self.expert_count > available:
            raise ValueError(
                f"expert range [{self.expert_start}, {self.expert_start + self.expert_count}) is outside "
                f"MiniMax-M2 expert count {self.plan.params.n_routed_experts}"
            )
        self._loader = MiniMaxM2RoutedBlockLoader(self.bundle, plan=self.plan)
        self._cache: dict[tuple[int, str], MiniMaxM2DeviceRoutedTensor] = {}
        self._grid_cache: dict[str, torch.Tensor] = {}

    @classmethod
    def from_path(
        cls,
        bundle_or_path: GGUFBundle | str | Path,
        *,
        device: str | torch.device = "cuda",
        plan: MiniMaxM2MoERuntimePlan | None = None,
        expert_start: int = 0,
        expert_count: int | None = None,
    ) -> "MiniMaxM2DeviceResidentCache":
        return cls(
            bundle_or_path,
            device=device,
            plan=plan,
            expert_start=expert_start,
            expert_count=expert_count,
        )

    def close(self) -> None:
        self.clear()
        self._loader.close()

    def __enter__(self) -> "MiniMaxM2DeviceResidentCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def clear(self) -> None:
        self._cache.clear()
        self._grid_cache.clear()

    def _type_id(self, type_name: str) -> int:
        try:
            return GGUF_DEVICE_TYPE_IDS[type_name]
        except KeyError as exc:
            raise NotImplementedError(f"MiniMax-M2 device cache does not support routed type {type_name!r}") from exc

    def _quant_grid(self, type_name: str) -> torch.Tensor:
        cached = self._grid_cache.get(type_name)
        if cached is None or cached.device != self.device:
            cached = _gguf_cuda_quant_grid(type_name, self.device)
            self._grid_cache[type_name] = cached
        return cached

    def load_role(self, layer: int, role: str) -> MiniMaxM2DeviceRoutedTensor:
        layer = int(layer)
        if role not in ROUTED_ROLES:
            raise ValueError(f"role must be one of {sorted(ROUTED_ROLES)}, got {role!r}")
        key = (layer, role)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        item = self.plan.tensor_for(layer, role)
        if item.type_name not in SUPPORTED_DEVICE_ROUTED_TYPES:
            raise NotImplementedError(f"MiniMax-M2 routed type {item.type_name!r} is not supported by the CUDA device cache")
        cpu_blocks, type_name, in_dim = self._loader.read_layer_role_blocks(
            layer,
            role,
            expert_start=self.expert_start,
            expert_count=self.expert_count,
        )
        if type_name != item.type_name:
            raise RuntimeError(f"routed type mismatch for {item.source_name}: plan={item.type_name}, reader={type_name}")
        blocks = cpu_blocks.to(device=self.device, non_blocking=False).contiguous()
        device_tensor = MiniMaxM2DeviceRoutedTensor(
            source_name=item.source_name,
            role=role,
            layer=layer,
            blocks=blocks,
            type_name=type_name,
            type_id=self._type_id(type_name),
            in_dim=int(in_dim),
            out_dim=int(blocks.shape[1]),
            expert_start=self.expert_start,
            expert_count=self.expert_count,
            device=self.device,
        )
        self._cache[key] = device_tensor
        return device_tensor

    def layer(self, layer: int) -> MiniMaxM2DeviceRoutedLayer:
        layer = int(layer)
        return MiniMaxM2DeviceRoutedLayer(
            layer=layer,
            w1=self.load_role(layer, "routed_w1"),
            w3=self.load_role(layer, "routed_w3"),
            w2=self.load_role(layer, "routed_w2"),
            device=self.device,
        )

    def preload_layers(self, layer_start: int = 0, layer_count: int | None = None) -> int:
        layer_start = int(layer_start)
        layer_count = self.plan.params.n_layers - layer_start if layer_count is None else int(layer_count)
        if layer_start < 0 or layer_count < 0 or layer_start + layer_count > self.plan.params.n_layers:
            raise ValueError(
                f"layer range [{layer_start}, {layer_start + layer_count}) is outside MiniMax-M2 range [0, {self.plan.params.n_layers})"
            )
        loaded = 0
        for layer_id in range(layer_start, layer_start + layer_count):
            before = len(self._cache)
            self.layer(layer_id)
            loaded += len(self._cache) - before
        return loaded

    def memory_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self._cache.values())

    def summary(self) -> dict[str, Any]:
        return {
            "architecture": MINIMAX_M2_ARCHITECTURE,
            "device": str(self.device),
            "expert_start": self.expert_start,
            "expert_count": self.expert_count,
            "cached_tensors": len(self._cache),
            "resident_bytes": self.memory_bytes(),
            "cached_roles": sorted(f"{layer}:{role}" for (layer, role) in self._cache),
        }

    def cuda_gemm_smoke(
        self,
        *,
        layer: int = 0,
        role: str = "routed_w1",
        expert: int | None = None,
        tokens: int = 1,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[torch.Tensor, MiniMaxM2CudaGemmSmokeResult]:
        from src.kernels.cuda_loader import load_cuda_kernel

        tensor = self.load_role(int(layer), role)
        expert = self.expert_start if expert is None else int(expert)
        local_expert = expert - tensor.expert_start
        if local_expert < 0 or local_expert >= tensor.expert_count:
            raise ValueError(
                f"expert {expert} is outside cached range [{tensor.expert_start}, {tensor.expert_start + tensor.expert_count})"
            )
        tokens = int(tokens)
        if tokens <= 0:
            raise ValueError(f"tokens must be positive, got {tokens}")
        cuda_mod = load_cuda_kernel()
        if cuda_mod is None or not hasattr(cuda_mod, "gguf_quant_gemm_forward"):
            raise RuntimeError("built CUDA extension with gguf_quant_gemm_forward is not available")
        expert_blocks = tensor.blocks[local_expert].contiguous()
        x = torch.zeros((tokens, tensor.in_dim), device=self.device, dtype=dtype)
        grid = self._quant_grid(tensor.type_name)
        y = cuda_mod.gguf_quant_gemm_forward(x.contiguous(), expert_blocks, int(tensor.in_dim), int(tensor.type_id), grid)
        finite = bool(torch.isfinite(y).all().item())
        result = MiniMaxM2CudaGemmSmokeResult(
            layer=int(layer),
            role=role,
            expert=expert,
            type_name=tensor.type_name,
            type_id=tensor.type_id,
            device=self.device,
            input_shape=tuple(int(dim) for dim in x.shape),
            blocks_shape=tuple(int(dim) for dim in expert_blocks.shape),
            output_shape=tuple(int(dim) for dim in y.shape),
            output_dtype=y.dtype,
            finite=finite,
            resident_bytes=self.memory_bytes(),
        )
        if not finite:
            raise RuntimeError(f"MiniMax-M2 CUDA GEMM smoke produced non-finite values: {result.as_dict()}")
        return y, result
