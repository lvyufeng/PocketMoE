"""Single source of truth for the 4-GPU baseline partition rules.

The same rules apply to both safetensors (`load_original_hf_model`) and GGUF
(`load_gguf_model`) loaders so a given weight always lands on the same rank
regardless of checkpoint format. Helpers are intentionally side-effect free;
loaders inject `world_size` and `rank`.
"""

from __future__ import annotations

import os
from typing import Iterable

import torch


POLICY_LEGACY = "legacy"
POLICY_BASELINE_4GPU = "baseline_4gpu"
POLICY_LAYER_PP_4GPU = "layer_pp_4gpu"
SUPPORTED_POLICIES = (POLICY_LEGACY, POLICY_BASELINE_4GPU, POLICY_LAYER_PP_4GPU)

# When baseline_4gpu is active these env-driven runtime modes break the
# guarantee that each rank owns a contiguous, balanced slice of the model.
# We refuse to start rather than silently fall back to a degenerate layout.
_BASELINE_FORBIDDEN_TRUE_ENV = (
    "DEEPSEEK_CPU_MOE_INPROC_SERVER",
    "DEEPSEEK_CPU_MOE_RANK0_SERVER",
    "DEEPSEEK_REPLICATED_C4_INDEXER",
)


def _env_truthy(name: str, default: str = "0") -> bool:
    active_name = name.replace("DEEPSEEK_", "DEEPSEEK_ACTIVE_", 1)
    if active_name in os.environ:
        return os.getenv(active_name, "0").lower() in {"1", "true", "yes"}
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


def assert_baseline_compatible_env(policy: str, world_size: int | None = None) -> None:
    if policy not in {POLICY_BASELINE_4GPU, POLICY_LAYER_PP_4GPU}:
        return
    violations = []
    if world_size is not None:
        if policy == POLICY_BASELINE_4GPU and world_size != 4:
            violations.append(f"world_size={world_size}")
        if policy == POLICY_LAYER_PP_4GPU and world_size not in {2, 4}:
            violations.append(f"world_size={world_size}")
    violations.extend(name for name in _BASELINE_FORBIDDEN_TRUE_ENV if _env_truthy(name))
    if _env_truthy("DEEPSEEK_CPU_MOE_EXTERNAL_SERVER") and os.getenv("DEEPSEEK_CPU_MOE_EXTERNAL_PREFILL_LOCAL", "1").lower() not in {"1", "true", "yes"}:
        violations.append("DEEPSEEK_CPU_MOE_EXTERNAL_SERVER=1 with DEEPSEEK_CPU_MOE_EXTERNAL_PREFILL_LOCAL=0")
    if policy == POLICY_LAYER_PP_4GPU and _env_truthy("DEEPSEEK_DECODE_MTP_SPEC"):
        violations.append("DEEPSEEK_DECODE_MTP_SPEC")
    if violations:
        raise RuntimeError(
            f"partition_policy={policy} is incompatible with: "
            + ", ".join(violations)
            + ". Disable these envs or run with partition_policy=legacy."
        )


def normalize_policy(policy: str | None) -> str:
    if policy is None:
        return POLICY_LEGACY
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"unsupported partition_policy: {policy!r}; expected one of {SUPPORTED_POLICIES}")
    return policy


def expert_idx_from_name(name: str) -> int | None:
    if "experts." not in name or "shared_experts" in name:
        return None
    parts = name.split('.')
    try:
        return int(parts[parts.index("experts") + 1])
    except (ValueError, IndexError):
        return None


def module_owns_expert(module, expert_idx: int) -> bool:
    return getattr(module, "experts_start_idx", 0) <= expert_idx < getattr(module, "experts_end_idx", 0)


def is_replicated_c4_indexer_param(name: str, module) -> bool:
    return getattr(module, "replicated_c4_indexer", False) and (
        name.endswith("indexer.wq_b.weight")
        or name.endswith("indexer.weights_proj.weight")
        or name.endswith("indexer.wq_b.scale")
        or name.endswith("indexer.weights_proj.scale")
    )


def is_column_parallel(module) -> bool:
    from src.models.deepseek_v4.runtime import ColumnParallelLinear, ParallelEmbedding, ParallelHead
    return isinstance(module, (ParallelEmbedding, ParallelHead, ColumnParallelLinear))


def is_row_parallel(module) -> bool:
    from src.models.deepseek_v4.runtime import RowParallelLinear
    return isinstance(module, RowParallelLinear)


def partition_rule_kind(name: str, module) -> str:
    if "experts." in name and "shared_experts" not in name:
        return "expert_owned"
    if is_replicated_c4_indexer_param(name, module):
        return "replicated_indexer"
    if is_column_parallel(module):
        return "column_parallel"
    if is_row_parallel(module):
        return "row_parallel"
    from src.models.deepseek_v4.runtime import Attention
    if isinstance(module, Attention) and name.endswith("attn_sink"):
        return "attn_sink"
    if "shared_experts" in name:
        return "replicated_shared_expert"
    return "replicated_baseline"


def shard_shape_for_rank(shape: tuple[int, ...], rule_kind: str, world_size: int, rank: int) -> tuple[int | None, int | None, int | None]:
    """Return (shard_dim, start, shard_size) for lazy loaders, or (None, None, None).

    This mirrors `shard_tensor_for_rank()` without materializing the full tensor.
    """
    if rule_kind == "column_parallel":
        if len(shape) in (1, 2):
            shard_dim = 0
            assert shape[shard_dim] % world_size == 0, f"shape {shape} not divisible on dim {shard_dim}"
            shard = shape[shard_dim] // world_size
            return shard_dim, rank * shard, shard
        return None, None, None
    if rule_kind == "row_parallel":
        if len(shape) == 2:
            shard_dim = 1
            assert shape[shard_dim] % world_size == 0, f"shape {shape} not divisible on dim {shard_dim}"
            shard = shape[shard_dim] // world_size
            return shard_dim, rank * shard, shard
        return None, None, None
    if rule_kind == "attn_sink":
        assert shape[0] % world_size == 0, f"shape {shape} not divisible on dim 0"
        shard = shape[0] // world_size
        return 0, rank * shard, shard
    return None, None, None


def _attn_sink_shard(name: str, tensor: torch.Tensor, module, world_size: int, rank: int) -> torch.Tensor:
    from src.models.deepseek_v4.runtime import Attention
    if not (isinstance(module, Attention) and name.endswith("attn_sink")):
        return tensor
    assert tensor.size(0) % world_size == 0, f"{name} not divisible on dim 0"
    shard = tensor.size(0) // world_size
    return tensor.narrow(0, rank * shard, shard).contiguous()


def shard_tensor_for_rank(
    name: str,
    tensor: torch.Tensor,
    module,
    world_size: int,
    rank: int,
) -> torch.Tensor | None:
    """Return the rank-local slice of `tensor`, or None when this rank does not own it.

    The same dispatch is used by GGUF and safetensors loaders so the per-rank
    layout is identical across checkpoint formats.
    """
    if "experts." in name and "shared_experts" not in name:
        parts = name.split('.')
        expert_idx = int(parts[parts.index("experts") + 1])
        local_experts = getattr(module, "n_local_experts", None)
        if local_experts is None:
            return tensor
        start = getattr(module, "experts_start_idx", rank * local_experts)
        end = getattr(module, "experts_end_idx", start + local_experts)
        if expert_idx < start or expert_idx >= end:
            return None
        return tensor

    if is_replicated_c4_indexer_param(name, module):
        return tensor

    if is_column_parallel(module):
        if tensor.ndim in (1, 2):
            shard_dim = 0
            assert tensor.size(shard_dim) % world_size == 0, f"{name} not divisible on dim {shard_dim}"
            shard = tensor.size(shard_dim) // world_size
            return tensor.narrow(shard_dim, rank * shard, shard).contiguous()
        return tensor

    if is_row_parallel(module):
        if tensor.ndim == 2:
            shard_dim = 1
            assert tensor.size(shard_dim) % world_size == 0, f"{name} not divisible on dim {shard_dim}"
            shard = tensor.size(shard_dim) // world_size
            return tensor.narrow(shard_dim, rank * shard, shard).contiguous()
        return tensor

    return _attn_sink_shard(name, tensor, module, world_size, rank)


def shard_q8_0_blocks_for_rank(
    name: str,
    blocks: torch.Tensor,
    module,
    world_size: int,
    rank: int,
) -> torch.Tensor | None:
    if "experts." in name and "shared_experts" not in name:
        return shard_tensor_for_rank(name, blocks, module, world_size, rank)
    if is_column_parallel(module):
        assert blocks.size(0) % world_size == 0, f"{name} not divisible on q8_0 rows"
        shard = blocks.size(0) // world_size
        return blocks.narrow(0, rank * shard, shard).contiguous()
    if is_row_parallel(module):
        assert blocks.size(1) % world_size == 0, f"{name} not divisible on q8_0 block columns"
        shard = blocks.size(1) // world_size
        return blocks.narrow(1, rank * shard, shard).contiguous()
    return _attn_sink_shard(name, blocks, module, world_size, rank)


def checkpoint_key_is_needed_for_rank(
    key: str,
    state_keys: set[str],
    name_to_module: dict[str, torch.nn.Module],
) -> bool:
    if key.endswith(".scale") and f"{key[:-6]}.weight" in state_keys:
        return False
    if key not in state_keys:
        return False
    module_name, _, _ = key.rpartition('.')
    module = name_to_module.get(module_name)
    if module is None:
        return False
    expert_idx = expert_idx_from_name(key)
    if expert_idx is None:
        return True
    parts = key.split('.')
    owner_module = name_to_module.get('.'.join(parts[:parts.index("experts")]))
    return owner_module is not None and module_owns_expert(owner_module, expert_idx)


def is_layer_pp_policy(policy: str) -> bool:
    return policy == POLICY_LAYER_PP_4GPU


def _layer_range_for_rank(n_layers: int, pp_size: int, pp_rank: int) -> tuple[int, int]:
    start = (pp_rank * n_layers) // pp_size
    end = ((pp_rank + 1) * n_layers) // pp_size
    return start, end


def layer_owner_rank(layer_idx: int, n_layers: int, pp_size: int) -> int:
    for pp_rank in range(pp_size):
        start, end = _layer_range_for_rank(n_layers, pp_size, pp_rank)
        if start <= layer_idx < end:
            return pp_rank
    raise ValueError(f"layer {layer_idx} is out of range for n_layers={n_layers}")


def parameter_owner_rank(name: str, n_layers: int, pp_size: int) -> int | None:
    if name.startswith("embed."):
        return 0
    if name.startswith("norm.") or name.startswith("head.") or name.startswith("hc_head_"):
        return pp_size - 1
    if name.startswith("layers."):
        parts = name.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            return layer_owner_rank(int(parts[1]), n_layers, pp_size)
    if name.startswith("mtp."):
        return pp_size - 1
    return None


def checkpoint_key_is_needed_for_policy(
    key: str,
    state_keys: set[str],
    name_to_module: dict[str, torch.nn.Module],
    policy: str,
    world_size: int,
    rank: int,
    n_layers: int | None = None,
) -> bool:
    if key.endswith(".scale") and f"{key[:-6]}.weight" in state_keys:
        return False
    if key not in state_keys:
        return False
    if is_layer_pp_policy(policy):
        if n_layers is None:
            raise ValueError("n_layers is required for layer_pp_4gpu ownership checks")
        owner = parameter_owner_rank(key, n_layers, world_size)
        return owner is None or owner == rank
    return checkpoint_key_is_needed_for_rank(key, state_keys, name_to_module)


def _iter_moe_modules(model) -> Iterable:
    for layer in getattr(model, "layers", []):
        ffn = getattr(layer, "ffn", None)
        if ffn is not None:
            yield "main", layer.layer_id, ffn
    for layer in getattr(model, "mtp", []):
        ffn = getattr(layer, "ffn", None)
        if ffn is not None:
            yield "mtp", layer.layer_id, ffn


def log_partition_layout(model, world_size: int, rank: int, policy: str) -> None:
    """Print one line per rank summarizing the realized partition layout.

    Called once after `Transformer(args)` is built. Output is intentionally
    short and machine-grepable so we can diff GGUF vs safetensors runs.
    """
    embed = getattr(model, "embed", None)
    head = getattr(model, "head", None)
    vocab_start = getattr(embed, "vocab_start_idx", 0)
    vocab_end = getattr(embed, "vocab_end_idx", 0)
    head_part = getattr(head, "part_vocab_size", 0)

    layers = getattr(model, "layers", None)
    n_local_heads = 0
    n_local_groups = 0
    expert_ranges: list[tuple[int, int]] = []
    if layers is not None and len(layers) > 0:
        local_layers = [layer for layer in layers if layer is not None]
        attn0 = getattr(local_layers[0], "attn", None) if local_layers else None
        if attn0 is not None:
            n_local_heads = int(getattr(attn0, "n_local_heads", 0))
            n_local_groups = int(getattr(attn0, "n_local_groups", 0))
        for _, _, ffn in _iter_moe_modules(model):
            expert_ranges.append((int(getattr(ffn, "experts_start_idx", 0)), int(getattr(ffn, "experts_end_idx", 0))))

    expert_summary = "none"
    if expert_ranges:
        first_range = expert_ranges[0]
        all_same = all(r == first_range for r in expert_ranges)
        if all_same:
            expert_summary = f"{first_range[0]}:{first_range[1]}"
        else:
            expert_summary = "mixed"

    inproc = _env_truthy("DEEPSEEK_CPU_MOE_INPROC_SERVER")
    rank0srv = _env_truthy("DEEPSEEK_CPU_MOE_RANK0_SERVER")
    extsrv = _env_truthy("DEEPSEEK_CPU_MOE_EXTERNAL_SERVER")
    extlocal = _env_truthy("DEEPSEEK_CPU_MOE_EXTERNAL_PREFILL_LOCAL", "1")
    repidx = _env_truthy("DEEPSEEK_REPLICATED_C4_INDEXER")
    forbidden_active = []
    if inproc:
        forbidden_active.append("INPROC_SERVER")
    if rank0srv:
        forbidden_active.append("RANK0_SERVER")
    if extsrv and not extlocal:
        forbidden_active.append("EXTERNAL_SERVER_PREFILL_REMOTE")
    if repidx:
        forbidden_active.append("REPLICATED_C4_INDEXER")
    forbidden = ",".join(forbidden_active) if forbidden_active else "none"

    phase = os.getenv("DEEPSEEK_PD_ACTIVE_PHASE", "none") or "none"
    if policy == POLICY_LAYER_PP_4GPU:
        layer_start = getattr(model, "layer_start", 0)
        layer_end = getattr(model, "layer_end", 0)
        shared_policy = "stage_local_full"
        print(
            f"partition_layout policy={policy} world_size={world_size} rank={rank} "
            f"pp_rank={rank} pp_size={world_size} tp_rank=0 tp_size=1 "
            f"layers=[{layer_start},{layer_end}) owns_embed={getattr(model, 'owns_embedding', False)} "
            f"owns_head={getattr(model, 'owns_head', False)} local_heads={n_local_heads} "
            f"local_groups={n_local_groups} expert_range={expert_summary} "
            f"shared_expert_policy={shared_policy} phase={phase} forbidden_modes={forbidden}",
            flush=True,
        )
        return

    shared_policy = "replicated_baseline" if policy == POLICY_BASELINE_4GPU else "legacy"
    print(
        f"partition_layout policy={policy} world_size={world_size} rank={rank} "
        f"vocab=[{vocab_start},{vocab_end}) head_part={head_part} "
        f"local_heads={n_local_heads} local_groups={n_local_groups} "
        f"expert_range={expert_summary} shared_expert_policy={shared_policy} "
        f"phase={phase} forbidden_modes={forbidden}",
        flush=True,
    )
