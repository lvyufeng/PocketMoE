from __future__ import annotations

from collections import defaultdict

import torch

from src.gguf.ds4_mapping import validate_ds4_tensor_mappings
from src.gguf.reader import GGUFReader
from src.gguf.tensor_reader import GGUFTensorDataReader
from src.kernels.ops import soft_fp8_blockfp8_weight_dequant
from src.runtime.partition_policy import is_layer_pp_policy, shard_q8_0_blocks_for_rank, shard_tensor_for_rank
from src.runtime.transformer import Transformer


_ROUTED_ORIGINAL_FORMAT_ROLES = {"routed_w1", "routed_w2", "routed_w3"}
# Quantized GGUF types we bind directly as raw routed-expert blocks.  iq1_m joins
# the existing q2_k/iq2_xxs routed path; non-routed iq1_m is still rejected by
# _unsupported_non_routed_quantized_mappings so it can never silently load.
_ORIGINAL_QUANTIZED_TYPES = {"q8_0", "q2_k", "iq2_xxs", "iq1_m"}
_IGNORED_RUNTIME_TARGET_PREFIXES = ("mtp.",)


def _module_for_key(name_to_module: dict[str, torch.nn.Module], key: str):
    module_name, _, _ = key.rpartition(".")
    return name_to_module.get(module_name)


def _dequant_int8_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (weight.to(torch.float32) * scale.to(torch.float32).unsqueeze(1)).contiguous()


def _quantize_int8_per_row(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight_f = weight.float().contiguous()
    row_scale = weight_f.abs().amax(dim=1).clamp_min(1e-6) / 127.0
    weight_q = torch.clamp(torch.round(weight_f / row_scale.unsqueeze(1)), -127, 127).to(torch.int8).contiguous()
    return weight_q, row_scale.contiguous()


def _quantize_fp8_block_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight_f = weight.float().contiguous()
    rows, cols = weight_f.shape
    block = 128
    padded_rows = ((rows + block - 1) // block) * block
    padded_cols = ((cols + block - 1) // block) * block
    padded = torch.zeros((padded_rows, padded_cols), dtype=torch.float32, device=weight_f.device)
    padded[:rows, :cols] = weight_f
    row_blocks = padded_rows // block
    col_blocks = padded_cols // block
    blocks = padded.view(row_blocks, block, col_blocks, block).permute(0, 2, 1, 3)
    scales = blocks.abs().amax(dim=(2, 3)).clamp_min(1e-6) / 448.0
    scales = torch.pow(2.0, torch.ceil(torch.log2(scales)))
    quant = torch.clamp(blocks / scales[:, :, None, None], -448.0, 448.0).to(torch.float8_e4m3fn)
    quant = quant.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)[:rows, :cols].contiguous()
    return quant, scales.contiguous()


def _copy_scale_tensor(key: str, state_dict: dict[str, torch.Tensor], module, scale: torch.Tensor) -> None:
    scale_key = f"{key[:-7]}.scale" if key.endswith(".weight") else "scale"
    scale_target = state_dict.get(scale_key)
    if scale_target is not None:
        if scale.shape != scale_target.shape:
            raise ValueError(f"Scale shape mismatch for {scale_key}: got {tuple(scale.shape)}, expected {tuple(scale_target.shape)}")
        scale_target.copy_(scale.to(device=scale_target.device, dtype=scale_target.dtype))
        return
    module_scale = getattr(module, "scale", None)
    if module_scale is None:
        raise ValueError(f"Missing scale target for {key}")
    if scale.shape != module_scale.shape:
        raise ValueError(f"Scale shape mismatch for {scale_key}: got {tuple(scale.shape)}, expected {tuple(module_scale.shape)}")
    module_scale.copy_(scale.to(device=module_scale.device, dtype=module_scale.dtype))


def _copy_int8_weight_and_scale(
    key: str,
    state_dict: dict[str, torch.Tensor],
    module,
    target: torch.Tensor,
    weight_q: torch.Tensor,
    weight_s: torch.Tensor,
) -> None:
    if weight_q.shape != target.shape:
        raise ValueError(f"Shape mismatch for {key}: got {tuple(weight_q.shape)}, expected {tuple(target.shape)}")
    target.copy_(weight_q.to(device=target.device, dtype=target.dtype))
    _copy_scale_tensor(key, state_dict, module, weight_s)


def _copy_float_weight_to_target(
    key: str,
    state_dict: dict[str, torch.Tensor],
    module,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    if target.dtype == torch.float8_e4m3fn:
        weight_q, weight_s = _quantize_fp8_block_weight(weight)
        if weight_q.shape != target.shape:
            raise ValueError(f"Shape mismatch for {key}: got {tuple(weight_q.shape)}, expected {tuple(target.shape)}")
        target.copy_(weight_q.to(device=target.device, dtype=target.dtype))
        _copy_scale_tensor(key, state_dict, module, weight_s)
        return
    if weight.shape != target.shape:
        raise ValueError(f"Shape mismatch for {key}: got {tuple(weight.shape)}, expected {tuple(target.shape)}")
    target.copy_(weight.to(device=target.device, dtype=target.dtype))


def _maybe_bind_routed_int8_arena(
    key: str,
    state_dict: dict[str, torch.Tensor],
    name_to_module: dict[str, torch.nn.Module],
) -> None:
    if "experts." not in key or "shared_experts" in key or not key.endswith(".weight"):
        return
    parts = key.split('.')
    if len(parts) < 2 or parts[-2] not in {"w1", "w2", "w3"}:
        return
    module_name, _, _ = key.rpartition('.')
    linear = name_to_module.get(module_name)
    if linear is None or getattr(linear, "weight", None) is None or linear.weight.dtype != torch.int8:
        return
    owner_name = '.'.join(parts[:parts.index("experts")])
    owner = name_to_module.get(owner_name)
    backend = getattr(owner, "cpu_backend", None)
    if backend is None or not hasattr(backend, "_reserve_arena_slot_int8"):
        return
    expert_idx = int(parts[parts.index("experts") + 1])
    expert = name_to_module.get('.'.join(parts[:-2]))
    if expert is None:
        return
    slot = backend._reserve_arena_slot_int8(expert, expert_idx)
    if slot is None:
        return
    w1q, w1s, w2q, w2s, w3q, w3s = slot
    expert.w1.set_int8_storage(w1q, w1s)
    expert.w2.set_int8_storage(w2q, w2s)
    expert.w3.set_int8_storage(w3q, w3s)
    backend._expert_int8_cache[expert_idx] = (w1q, w1s, w2q, w2s, w3q, w3s)
    state_dict[f"{owner_name}.experts.{expert_idx}.w1.weight"] = w1q
    state_dict[f"{owner_name}.experts.{expert_idx}.w1.scale"] = w1s
    state_dict[f"{owner_name}.experts.{expert_idx}.w2.weight"] = w2q
    state_dict[f"{owner_name}.experts.{expert_idx}.w2.scale"] = w2s
    state_dict[f"{owner_name}.experts.{expert_idx}.w3.weight"] = w3q
    state_dict[f"{owner_name}.experts.{expert_idx}.w3.scale"] = w3s


def _maybe_bind_routed_fp4_arena(
    key: str,
    state_dict: dict[str, torch.Tensor],
    name_to_module: dict[str, torch.nn.Module],
) -> None:
    if "experts." not in key or "shared_experts" in key or not key.endswith(".weight"):
        return
    parts = key.split('.')
    if len(parts) < 2 or parts[-2] not in {"w1", "w2", "w3"}:
        return
    module_name, _, _ = key.rpartition('.')
    linear = name_to_module.get(module_name)
    if linear is None or getattr(linear, "weight", None) is None or linear.weight.dtype != torch.float4_e2m1fn_x2:
        return
    owner_name = '.'.join(parts[:parts.index("experts")])
    owner = name_to_module.get(owner_name)
    backend = getattr(owner, "cpu_backend", None)
    if backend is None or not hasattr(backend, "_reserve_arena_slot_fp4"):
        return
    expert_idx = int(parts[parts.index("experts") + 1])
    expert = name_to_module.get('.'.join(parts[:-2]))
    if expert is None:
        return
    slot = backend._reserve_arena_slot_fp4(expert, expert_idx)
    if slot is None:
        return
    w1q, w1s, w2q, w2s, w3q, w3s = slot
    expert._cpu_w1 = w1q
    expert._cpu_w2 = w2q
    expert._cpu_w3 = w3q
    expert._cpu_w1_scale = w1s
    expert._cpu_w2_scale = w2s
    expert._cpu_w3_scale = w3s
    expert._cpu_w2_tiled = None
    expert._cpu_w2_scale_tiled = None
    expert._cpu_materialized = True
    backend._expert_fp4_cache[expert_idx] = (w1q, w1s, w2q, w2s, w3q, w3s)
    state_dict[f"{owner_name}.experts.{expert_idx}.w1.weight"] = w1q.view(torch.float4_e2m1fn_x2)
    state_dict[f"{owner_name}.experts.{expert_idx}.w1.scale"] = w1s.view(torch.float8_e8m0fnu)
    state_dict[f"{owner_name}.experts.{expert_idx}.w2.weight"] = w2q.view(torch.float4_e2m1fn_x2)
    state_dict[f"{owner_name}.experts.{expert_idx}.w2.scale"] = w2s.view(torch.float8_e8m0fnu)
    state_dict[f"{owner_name}.experts.{expert_idx}.w3.weight"] = w3q.view(torch.float4_e2m1fn_x2)
    state_dict[f"{owner_name}.experts.{expert_idx}.w3.scale"] = w3s.view(torch.float8_e8m0fnu)
    backend._release_expert_parameter_storage(expert)


def _cuda_quant_device() -> torch.device | None:
    if not torch.cuda.is_available():
        return None
    return torch.device("cuda", torch.cuda.current_device())


def _scale_key_for_weight(key: str, state_dict: dict[str, torch.Tensor]) -> str | None:
    if not key.endswith(".weight"):
        return None
    scale_key = f"{key[:-7]}.scale"
    return scale_key if scale_key in state_dict else None


def _bind_gguf_raw_routed_experts(
    gguf_path: str,
    mappings,
    state_dict: dict[str, torch.Tensor],
    name_to_module: dict[str, torch.nn.Module],
) -> set[str]:
    bindings: dict[str, dict[str, object]] = {}
    loaded: set[str] = set()
    for mapping in mappings:
        if mapping.role not in _ROUTED_ORIGINAL_FORMAT_ROLES:
            continue
        parts = mapping.target_key.split('.')
        expert_name = '.'.join(parts[:-2])
        expert = name_to_module.get(expert_name)
        if expert is None:
            raise ValueError(f"Missing expert module for {mapping.target_key}")
        record = bindings.setdefault(
            expert_name,
            {
                "expert": expert,
                "expert_id": mapping.expert,
                "w1_name": None,
                "w2_name": None,
                "w3_name": None,
            },
        )
        if mapping.role == "routed_w1":
            record["w1_name"] = mapping.gguf_name
        elif mapping.role == "routed_w2":
            record["w2_name"] = mapping.gguf_name
        elif mapping.role == "routed_w3":
            record["w3_name"] = mapping.gguf_name
        loaded.add(mapping.target_key)
        scale_key = _scale_key_for_weight(mapping.target_key, state_dict)
        if scale_key is not None:
            loaded.add(scale_key)

    for expert_name, record in bindings.items():
        if record["expert_id"] is None:
            raise ValueError(f"Missing expert id for {expert_name}")
        if record["w1_name"] is None or record["w2_name"] is None or record["w3_name"] is None:
            raise ValueError(f"Incomplete GGUF raw binding for {expert_name}")
        record["expert"].set_gguf_raw_storage(
            gguf_path,
            record["w1_name"],
            record["w2_name"],
            record["w3_name"],
            int(record["expert_id"]),
        )
    return loaded


def _unsupported_non_routed_quantized_mappings(ds4, mappings) -> list[str]:
    examples: list[str] = []
    seen: set[str] = set()
    for mapping in mappings:
        if mapping.role in _ROUTED_ORIGINAL_FORMAT_ROLES:
            continue
        tensor = ds4.tensors_by_name.get(mapping.gguf_name)
        if tensor is None or tensor.type_name not in _ORIGINAL_QUANTIZED_TYPES:
            continue
        if tensor.type_name == "q8_0":
            continue
        example = f"{mapping.gguf_name} ({tensor.type_name}) -> {mapping.target_key}"
        if example in seen:
            continue
        seen.add(example)
        examples.append(example)
    return examples


def _copy_loaded_tensor(
    key: str,
    tensor: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    name_to_module: dict[str, torch.nn.Module],
    world_size: int,
    rank: int,
    partition_policy: str = "legacy",
) -> set[str]:
    module = _module_for_key(name_to_module, key)
    if module is None:
        return set()
    _maybe_bind_routed_int8_arena(key, state_dict, name_to_module)
    _maybe_bind_routed_fp4_arena(key, state_dict, name_to_module)
    target = state_dict[key]
    if not is_layer_pp_policy(partition_policy):
        tensor = shard_tensor_for_rank(key, tensor, module, world_size, rank)
    if tensor is None:
        return set()
    loaded = {key}
    scale_key = _scale_key_for_weight(key, state_dict)
    if key.endswith(".weight") and target.dtype == torch.int8:
        weight_q, weight_s = _quantize_int8_per_row(tensor)
        _copy_int8_weight_and_scale(key, state_dict, module, target, weight_q, weight_s)
        if scale_key is not None:
            loaded.add(scale_key)
        return loaded
    if key.endswith("wo_a.weight") and tensor.dtype == torch.float8_e4m3fn and target.dtype != torch.int8:
        scale = state_dict.get(scale_key) if scale_key is not None else None
        if scale is not None:
            tensor = soft_fp8_blockfp8_weight_dequant(tensor, scale)
    if target.dtype == torch.float8_e4m3fn:
        quant_device = _cuda_quant_device()
        if quant_device is not None:
            tensor = tensor.to(device=quant_device, non_blocking=True)
    if key.endswith(".weight"):
        _copy_float_weight_to_target(key, state_dict, module, target, tensor)
        if scale_key is not None:
            loaded.add(scale_key)
    else:
        if tensor.shape != target.shape:
            raise ValueError(f"Shape mismatch for {key}: got {tuple(tensor.shape)}, expected {tuple(target.shape)}")
        target.copy_(tensor.to(device=target.device, dtype=target.dtype))
    return loaded


def _load_q8_0_tensor(
    reader: GGUFTensorDataReader,
    gguf_name: str,
    mappings,
    state_dict: dict[str, torch.Tensor],
    name_to_module: dict[str, torch.nn.Module],
    world_size: int,
    rank: int,
    partition_policy: str = "legacy",
) -> set[str]:
    blocks = reader.read_q8_0_blocks(gguf_name)
    loaded: set[str] = set()
    for mapping in mappings:
        module = _module_for_key(name_to_module, mapping.target_key)
        if module is None:
            continue
        if is_layer_pp_policy(partition_policy):
            blocks_for_rank = blocks
        else:
            blocks_for_rank = shard_q8_0_blocks_for_rank(mapping.target_key, blocks, module, world_size, rank)
        if blocks_for_rank is None:
            continue
        if not hasattr(module, "set_q8_0_storage"):
            raise NotImplementedError(f"raw q8_0 runtime binding is not implemented for {mapping.target_key}")
        row_elems = getattr(module, "in_features", None)
        if row_elems is None:
            row_elems = getattr(module, "dim", None)
        if row_elems is None:
            raise NotImplementedError(f"unable to infer raw q8_0 row size for {mapping.target_key}")
        module.set_q8_0_storage(blocks_for_rank, row_elems=int(row_elems))
        loaded.add(mapping.target_key)
        scale_key = _scale_key_for_weight(mapping.target_key, state_dict)
        if scale_key is not None:
            loaded.add(scale_key)
    return loaded


def load_gguf_model(model: Transformer, gguf_path: str, world_size: int, rank: int) -> None:
    partition_policy = getattr(model, "partition_policy", "legacy")
    ds4 = GGUFReader(gguf_path).read()
    print("gguf load: metadata parsed", flush=True)
    state_dict = model.state_dict()
    state_shapes = {key: tuple(value.shape) for key, value in state_dict.items()}
    validation = validate_ds4_tensor_mappings(ds4, state_shapes)
    unmapped_sources = [] if is_layer_pp_policy(partition_policy) else validation.unmapped_sources
    if validation.missing_sources or validation.missing_targets or unmapped_sources or validation.unmapped_targets:
        problems = (
            validation.missing_sources
            + validation.missing_targets
            + [f"unmapped GGUF tensor {name}" for name in unmapped_sources]
            + [f"unmapped runtime tensor {name}" for name in validation.unmapped_targets]
        )
        raise ValueError(f"GGUF mapping validation failed with {len(problems)} issues, e.g. {problems[:10]}")
    if validation.shape_errors:
        print(f"gguf load: deferred {len(validation.shape_errors)} global/local shape checks to rank-local load", flush=True)

    name_to_module = dict(model.named_modules())
    grouped: dict[str, list] = defaultdict(list)
    generated_scale_keys: set[str] = set()
    routed_mappings = [mapping for mapping in validation.mappings if mapping.role in _ROUTED_ORIGINAL_FORMAT_ROLES]
    loaded = _bind_gguf_raw_routed_experts(gguf_path, routed_mappings, state_dict, name_to_module)
    print("gguf load: routed raw experts bound", flush=True)
    unsupported = _unsupported_non_routed_quantized_mappings(ds4, validation.mappings)
    if unsupported:
        raise NotImplementedError(
            "GGUF quantized tensors must stay in their original block format when loaded to device; "
            "raw non-routed Q8_0 runtime support is required before full GGUF load. "
            f"Examples: {unsupported[:5]}"
        )
    for mapping in validation.mappings:
        if mapping.transform == "generated_scale":
            generated_scale_keys.add(mapping.target_key)
            continue
        if mapping.role in _ROUTED_ORIGINAL_FORMAT_ROLES:
            continue
        grouped[mapping.gguf_name].append(mapping)

    total_sources = len(grouped)
    with GGUFTensorDataReader(ds4) as reader:
        for source_idx, gguf_name in enumerate(sorted(grouped), 1):
            if source_idx == 1 or source_idx == total_sources or source_idx % 50 == 0:
                print(f"load gguf tensor {source_idx}/{total_sources}: {gguf_name}", flush=True)
            source = ds4.tensors_by_name[gguf_name]
            if source.type_name == "q8_0":
                loaded.update(_load_q8_0_tensor(reader, gguf_name, grouped[gguf_name], state_dict, name_to_module, world_size, rank, partition_policy))
                continue
            tensor = reader.read_tensor(gguf_name)
            for mapping in grouped[gguf_name]:
                loaded.update(_copy_loaded_tensor(mapping.target_key, tensor, state_dict, name_to_module, world_size, rank, partition_policy))

    loaded.update(generated_scale_keys)
    missing = sorted(
        key for key in set(state_dict.keys()) - loaded
        if not any(key.startswith(prefix) for prefix in _IGNORED_RUNTIME_TARGET_PREFIXES)
    )
    if missing:
        raise ValueError(f"Missing {len(missing)} parameters from GGUF load, e.g. {missing[:10]}")
