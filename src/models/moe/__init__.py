"""Common MoE model spec framework."""

from src.models.moe.registry import detect_spec, get_spec, known_architectures
from src.models.moe.spec import (
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
