from __future__ import annotations

import importlib

import pytest


_REMOVED_MODULES = (
    "src.moe",
    "src.moe.cpu_backend",
    "src.moe_model",
    "src.moe_model.registry",
    "src.runtime.transformer",
    "src.runtime.gguf_loader",
    "src.runtime.partition_policy",
    "src.models.deepseek_v4.transformer",
    "src.models.deepseek_v4.gguf_loader",
    "src.models.deepseek_v4.partition_policy",
)


def test_deepseek_canonical_namespaces_import() -> None:
    model_runtime = importlib.import_module("src.models.deepseek_v4.runtime")
    model_spec = importlib.import_module("src.models.deepseek_v4.spec")
    runtime_loader = importlib.import_module("src.runtime.deepseek_v4.loader")
    runtime_partition = importlib.import_module("src.runtime.deepseek_v4.partition")

    assert hasattr(model_runtime, "ModelArgs")
    assert hasattr(model_runtime, "Transformer")
    assert hasattr(model_spec, "DeepSeekV4Spec")
    assert callable(runtime_loader.load_gguf_model)
    assert callable(runtime_partition.normalize_policy)


def test_deepseek_runtime_module_identity_is_single_canonical_module() -> None:
    runtime_a = importlib.import_module("src.models.deepseek_v4.runtime")
    runtime_b = importlib.import_module("src.models.deepseek_v4.runtime")

    assert runtime_a is runtime_b
    assert runtime_a.__dict__ is runtime_b.__dict__
    assert runtime_a.world_size is runtime_b.world_size
    assert runtime_a.rank is runtime_b.rank


def test_moe_backend_canonical_namespace_imports() -> None:
    cpu_backend = importlib.import_module("src.runtime.moe.cpu_backend")
    gpu_backend = importlib.import_module("src.runtime.moe.gpu_prefill_backend")
    shared_weights = importlib.import_module("src.runtime.moe.shared_weights")

    assert hasattr(cpu_backend, "CPURoutedExpertsBackend")
    assert hasattr(gpu_backend, "GPUPrefillMoEBackend")
    assert hasattr(shared_weights, "SharedCPUMoEWeightArena")


def test_moe_spec_canonical_namespace_imports() -> None:
    registry = importlib.import_module("src.models.moe.registry")
    spec = importlib.import_module("src.models.moe.spec")

    assert callable(registry.detect_spec)
    assert callable(registry.known_architectures)
    assert hasattr(spec, "MoEModelSpec")


def test_minimax_canonical_namespaces_import() -> None:
    minimax_spec = importlib.import_module("src.models.minimax_m2.spec")
    planning = importlib.import_module("src.models.minimax_m2.moe_planning")
    runtime = importlib.import_module("src.runtime.moe.minimax_m2")

    assert hasattr(minimax_spec, "MiniMaxM2Spec")
    assert callable(planning.build_minimax_m2_moe_runtime_plan)
    assert hasattr(runtime, "MiniMaxM2RoutedBlockLoader")
    assert hasattr(runtime, "MiniMaxM2DeviceResidentCache")


def test_removed_legacy_namespace_imports_fail() -> None:
    for module_name in _REMOVED_MODULES:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_model_registry_still_detects_deepseek_and_minimax() -> None:
    registry = importlib.import_module("src.models.moe.registry")

    assert "deepseek4" in registry.known_architectures()
    assert "minimax-m2" in registry.known_architectures()
    assert registry.get_spec("deepseek4").architecture == "deepseek4"
    assert registry.get_spec("minimax-m2").architecture == "minimax-m2"
