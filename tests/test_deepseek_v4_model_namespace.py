from __future__ import annotations

import importlib

import pytest


_REMOVED_MODULES = (
    "src.moe",
    "src.moe.cpu_backend",
    "src.moe_model",
    "src.moe_model.registry",
    "src.gguf",
    "src.gguf.reader",
    "src.gguf.bundle",
    "src.gguf.tensor_reader",
    "src.gguf.ds4_mapping",
    "src.runtime.deepseek_v4",
    "src.runtime.deepseek_v4.loader",
    "src.runtime.deepseek_v4.partition",
    "src.runtime.moe",
    "src.runtime.moe.cpu_backend",
    "src.runtime.moe.gpu_prefill_backend",
    "src.runtime.moe.shared_weights",
    "src.runtime.moe.cpu_server",
    "src.runtime.moe.minimax_m2",
    "src.models.moe",
    "src.models.moe.registry",
    "src.models.moe.spec",
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
    model_loader = importlib.import_module("src.models.deepseek_v4.loader")
    model_partition = importlib.import_module("src.models.deepseek_v4.partition")
    model_generation = importlib.import_module("src.models.deepseek_v4.generation")

    assert hasattr(model_runtime, "ModelArgs")
    assert hasattr(model_runtime, "Transformer")
    assert hasattr(model_spec, "DeepSeekV4Spec")
    assert callable(model_loader.load_gguf_model)
    assert callable(model_loader.load_model)
    assert callable(model_partition.normalize_policy)
    assert callable(model_generation.main)


def test_deepseek_runtime_module_identity_is_single_canonical_module() -> None:
    runtime_a = importlib.import_module("src.models.deepseek_v4.runtime")
    runtime_b = importlib.import_module("src.models.deepseek_v4.runtime")

    assert runtime_a is runtime_b
    assert runtime_a.__dict__ is runtime_b.__dict__
    assert runtime_a.world_size is runtime_b.world_size
    assert runtime_a.rank is runtime_b.rank


def test_loader_canonical_namespace_imports() -> None:
    gguf_reader = importlib.import_module("src.loader.gguf.reader")
    gguf_bundle = importlib.import_module("src.loader.gguf.bundle")
    tensor_reader = importlib.import_module("src.loader.gguf.tensor_reader")
    deepseek_mapping = importlib.import_module("src.loader.mappings.deepseek_v4")
    safetensors_loader = importlib.import_module("src.loader.safetensors")

    assert hasattr(gguf_reader, "GGUFReader")
    assert callable(gguf_bundle.read_gguf_bundle)
    assert hasattr(tensor_reader, "GGUFTensorDataReader")
    assert callable(deepseek_mapping.validate_ds4_tensor_mappings)
    assert callable(safetensors_loader.read_safetensors_index)


def test_moe_backend_canonical_namespace_imports() -> None:
    cpu_backend = importlib.import_module("src.components.moe.cpu_backend")
    gpu_backend = importlib.import_module("src.components.moe.gpu_prefill_backend")
    shared_weights = importlib.import_module("src.components.moe.shared_weights")
    ipc = importlib.import_module("src.components.moe.ipc")

    assert hasattr(cpu_backend, "CPURoutedExpertsBackend")
    assert hasattr(gpu_backend, "GPUPrefillMoEBackend")
    assert hasattr(shared_weights, "SharedCPUMoEWeightArena")
    assert hasattr(ipc, "CPUMoESharedMemory")


def test_moe_spec_canonical_namespace_imports() -> None:
    registry = importlib.import_module("src.components.moe.registry")
    spec = importlib.import_module("src.components.moe.spec")

    assert callable(registry.detect_spec)
    assert callable(registry.known_architectures)
    assert hasattr(spec, "MoEModelSpec")


def test_minimax_canonical_namespaces_import() -> None:
    minimax_spec = importlib.import_module("src.models.minimax_m2.spec")
    planning = importlib.import_module("src.models.minimax_m2.moe_planning")
    runtime = importlib.import_module("src.models.minimax_m2.moe_runtime")

    assert hasattr(minimax_spec, "MiniMaxM2Spec")
    assert callable(planning.build_minimax_m2_moe_runtime_plan)
    assert hasattr(runtime, "MiniMaxM2RoutedBlockLoader")
    assert hasattr(runtime, "MiniMaxM2DeviceResidentCache")


def test_removed_legacy_namespace_imports_fail() -> None:
    for module_name in _REMOVED_MODULES:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_model_registry_still_detects_deepseek_and_minimax() -> None:
    registry = importlib.import_module("src.components.moe.registry")

    assert "deepseek4" in registry.known_architectures()
    assert "minimax-m2" in registry.known_architectures()
    assert registry.get_spec("deepseek4").architecture == "deepseek4"
    assert registry.get_spec("minimax-m2").architecture == "minimax-m2"
