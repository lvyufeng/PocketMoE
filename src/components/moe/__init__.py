"""Common MoE model spec framework."""

from src.components.moe.registry import detect_spec, get_spec, known_architectures
from src.components.moe.spec import (
    CapabilityItem,
    CapabilityReport,
    MoEArchitectureParams,
    MoEModelSpec,
    PlacementDecision,
    SpecValidation,
    TensorMapping,
)

__all__ = [
    "CapabilityItem",
    "CapabilityReport",
    "MoEArchitectureParams",
    "MoEModelSpec",
    "PlacementDecision",
    "SpecValidation",
    "TensorMapping",
    "detect_spec",
    "get_spec",
    "known_architectures",
]
