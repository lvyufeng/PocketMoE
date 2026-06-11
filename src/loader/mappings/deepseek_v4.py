from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class DS4TensorMapping:
    gguf_name: str
    target_key: str
    transform: str
    expert: int | None = None
    role: str = ""


@dataclass(frozen=True)
class DS4MappingValidation:
    mappings: list[DS4TensorMapping]
    missing_sources: list[str]
    missing_targets: list[str]
    shape_errors: list[str]
    unmapped_sources: list[str]
    unmapped_targets: list[str]

    @property
    def ok(self) -> bool:
        return not (
            self.missing_sources
            or self.missing_targets
            or self.shape_errors
            or self.unmapped_sources
            or self.unmapped_targets
        )


def _metadata_int(metadata: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(metadata.get(key, default))
    except Exception:
        return default


def _has_state_key(state_keys: set[str] | None, key: str) -> bool:
    return state_keys is None or key in state_keys


def build_ds4_tensor_mappings(ds4, state_keys: Iterable[str] | None = None) -> list[DS4TensorMapping]:
    state_key_set = set(state_keys) if state_keys is not None else None
    source_names = set(ds4.tensors_by_name)
    n_layers = _metadata_int(ds4.metadata, "deepseek4.block_count", 43)
    n_hash_layers = _metadata_int(ds4.metadata, "deepseek4.hash_layer_count", 3)
    n_experts = _metadata_int(ds4.metadata, "deepseek4.expert_count", 256)
    mappings: list[DS4TensorMapping] = []

    def add(gguf_name: str, target_key: str, transform: str = "direct", *, expert: int | None = None, role: str = "") -> None:
        if _has_state_key(state_key_set, target_key):
            mappings.append(DS4TensorMapping(gguf_name, target_key, transform, expert, role))

    def add_weight(gguf_name: str, target_key: str, transform: str = "direct", *, expert: int | None = None, role: str = "") -> None:
        add(gguf_name, target_key, transform, expert=expert, role=role)

    add_weight("token_embd.weight", "embed.weight", "transpose", role="embedding")
    add_weight("output.weight", "head.weight", "transpose", role="lm_head")
    add("output_norm.weight", "norm.weight", role="norm")
    add_weight("output_hc_fn.weight", "hc_head_fn", "transpose", role="hc_head")
    add("output_hc_base.weight", "hc_head_base", role="hc_head")
    add("output_hc_scale.weight", "hc_head_scale", role="hc_head")

    for layer in range(n_layers):
        gguf_prefix = f"blk.{layer}"
        target_prefix = f"layers.{layer}"

        add("%s.attn_sinks.weight" % gguf_prefix, f"{target_prefix}.attn.attn_sink", role="attn")
        add_weight("%s.attn_q_a.weight" % gguf_prefix, f"{target_prefix}.attn.wq_a.weight", "transpose", role="attn")
        add("%s.attn_q_a_norm.weight" % gguf_prefix, f"{target_prefix}.attn.q_norm.weight", role="attn")
        add_weight("%s.attn_q_b.weight" % gguf_prefix, f"{target_prefix}.attn.wq_b.weight", "transpose", role="attn")
        add_weight("%s.attn_kv.weight" % gguf_prefix, f"{target_prefix}.attn.wkv.weight", "transpose", role="attn")
        add("%s.attn_kv_a_norm.weight" % gguf_prefix, f"{target_prefix}.attn.kv_norm.weight", role="attn")
        add_weight("%s.attn_output_a.weight" % gguf_prefix, f"{target_prefix}.attn.wo_a.weight", "transpose", role="attn")
        add_weight("%s.attn_output_b.weight" % gguf_prefix, f"{target_prefix}.attn.wo_b.weight", "transpose", role="attn")
        add("%s.attn_norm.weight" % gguf_prefix, f"{target_prefix}.attn_norm.weight", role="attn_norm")

        add_weight("%s.ffn_gate_inp.weight" % gguf_prefix, f"{target_prefix}.ffn.gate.weight", "transpose", role="gate")
        if layer < n_hash_layers:
            add_weight("%s.ffn_gate_tid2eid.weight" % gguf_prefix, f"{target_prefix}.ffn.gate.tid2eid", "transpose", role="gate_hash")
        else:
            add("%s.exp_probs_b.bias" % gguf_prefix, f"{target_prefix}.ffn.gate.bias", role="gate_bias")

        for expert in range(n_experts):
            add_weight(
                "%s.ffn_gate_exps.weight" % gguf_prefix,
                f"{target_prefix}.ffn.experts.{expert}.w1.weight",
                "routed_expert_transpose",
                expert=expert,
                role="routed_w1",
            )
            add_weight(
                "%s.ffn_down_exps.weight" % gguf_prefix,
                f"{target_prefix}.ffn.experts.{expert}.w2.weight",
                "routed_expert_transpose",
                expert=expert,
                role="routed_w2",
            )
            add_weight(
                "%s.ffn_up_exps.weight" % gguf_prefix,
                f"{target_prefix}.ffn.experts.{expert}.w3.weight",
                "routed_expert_transpose",
                expert=expert,
                role="routed_w3",
            )

        add_weight("%s.ffn_gate_shexp.weight" % gguf_prefix, f"{target_prefix}.ffn.shared_experts.w1.weight", "transpose", role="shared_w1")
        add_weight("%s.ffn_down_shexp.weight" % gguf_prefix, f"{target_prefix}.ffn.shared_experts.w2.weight", "transpose", role="shared_w2")
        add_weight("%s.ffn_up_shexp.weight" % gguf_prefix, f"{target_prefix}.ffn.shared_experts.w3.weight", "transpose", role="shared_w3")
        add("%s.ffn_norm.weight" % gguf_prefix, f"{target_prefix}.ffn_norm.weight", role="ffn_norm")

        add_weight("%s.hc_attn_fn.weight" % gguf_prefix, f"{target_prefix}.hc_attn_fn", "transpose", role="hc")
        add("%s.hc_attn_base.weight" % gguf_prefix, f"{target_prefix}.hc_attn_base", role="hc")
        add("%s.hc_attn_scale.weight" % gguf_prefix, f"{target_prefix}.hc_attn_scale", role="hc")
        add_weight("%s.hc_ffn_fn.weight" % gguf_prefix, f"{target_prefix}.hc_ffn_fn", "transpose", role="hc")
        add("%s.hc_ffn_base.weight" % gguf_prefix, f"{target_prefix}.hc_ffn_base", role="hc")
        add("%s.hc_ffn_scale.weight" % gguf_prefix, f"{target_prefix}.hc_ffn_scale", role="hc")

        if f"{gguf_prefix}.attn_compressor_ape.weight" in source_names or _has_state_key(state_key_set, f"{target_prefix}.attn.compressor.ape"):
            add_weight("%s.attn_compressor_ape.weight" % gguf_prefix, f"{target_prefix}.attn.compressor.ape", "transpose", role="attn_compressor")
            add_weight("%s.attn_compressor_kv.weight" % gguf_prefix, f"{target_prefix}.attn.compressor.wkv.weight", "transpose", role="attn_compressor")
            add_weight("%s.attn_compressor_gate.weight" % gguf_prefix, f"{target_prefix}.attn.compressor.wgate.weight", "transpose", role="attn_compressor")
            add("%s.attn_compressor_norm.weight" % gguf_prefix, f"{target_prefix}.attn.compressor.norm.weight", role="attn_compressor")

        if f"{gguf_prefix}.indexer.attn_q_b.weight" in source_names or _has_state_key(state_key_set, f"{target_prefix}.attn.indexer.wq_b.weight"):
            add_weight("%s.indexer.attn_q_b.weight" % gguf_prefix, f"{target_prefix}.attn.indexer.wq_b.weight", "transpose", role="indexer")
            add_weight("%s.indexer.proj.weight" % gguf_prefix, f"{target_prefix}.attn.indexer.weights_proj.weight", "transpose", role="indexer")
            add_weight("%s.indexer_compressor_ape.weight" % gguf_prefix, f"{target_prefix}.attn.indexer.compressor.ape", "transpose", role="indexer_compressor")
            add_weight("%s.indexer_compressor_kv.weight" % gguf_prefix, f"{target_prefix}.attn.indexer.compressor.wkv.weight", "transpose", role="indexer_compressor")
            add_weight("%s.indexer_compressor_gate.weight" % gguf_prefix, f"{target_prefix}.attn.indexer.compressor.wgate.weight", "transpose", role="indexer_compressor")
            add("%s.indexer_compressor_norm.weight" % gguf_prefix, f"{target_prefix}.attn.indexer.compressor.norm.weight", role="indexer_compressor")

    if state_key_set is not None:
        existing_targets = {mapping.target_key for mapping in mappings}
        generated: list[DS4TensorMapping] = []
        for mapping in mappings:
            if not mapping.target_key.endswith(".weight"):
                continue
            scale_key = f"{mapping.target_key[:-7]}.scale"
            if scale_key in state_key_set and scale_key not in existing_targets:
                generated.append(
                    DS4TensorMapping(
                        mapping.gguf_name,
                        scale_key,
                        "generated_scale",
                        mapping.expert,
                        mapping.role,
                    )
                )
                existing_targets.add(scale_key)
        mappings.extend(generated)

    return mappings


def _mapped_shape(mapping: DS4TensorMapping, source_shape: tuple[int, ...]) -> tuple[int, ...] | None:
    if mapping.transform == "generated_scale":
        return None
    if mapping.transform == "direct":
        return source_shape
    if mapping.transform == "transpose":
        if len(source_shape) != 2:
            return source_shape
        return (source_shape[1], source_shape[0])
    if mapping.transform == "routed_expert_transpose":
        if len(source_shape) != 3:
            return source_shape
        return (source_shape[1], source_shape[0])
    raise ValueError(f"unknown DS4 GGUF transform {mapping.transform!r}")


def validate_ds4_tensor_mappings(
    ds4,
    state_shapes: dict[str, tuple[int, ...]],
    *,
    ignored_target_prefixes: tuple[str, ...] = ("mtp.",),
) -> DS4MappingValidation:
    mappings = build_ds4_tensor_mappings(ds4, state_shapes.keys())
    source_by_name = ds4.tensors_by_name
    mapped_sources = {mapping.gguf_name for mapping in mappings}
    mapped_targets = {mapping.target_key for mapping in mappings}

    missing_sources = sorted({mapping.gguf_name for mapping in mappings if mapping.gguf_name not in source_by_name})
    missing_targets = sorted({mapping.target_key for mapping in mappings if mapping.target_key not in state_shapes})
    shape_errors: list[str] = []

    for mapping in mappings:
        source = source_by_name.get(mapping.gguf_name)
        if source is None or mapping.target_key not in state_shapes:
            continue
        expected_shape = _mapped_shape(mapping, tuple(source.dimensions))
        if expected_shape is None:
            continue
        target_shape = tuple(state_shapes[mapping.target_key])
        if expected_shape != target_shape:
            shape_errors.append(
                f"{mapping.gguf_name} -> {mapping.target_key}: got {expected_shape}, expected {target_shape}"
            )

    unmapped_sources = sorted(set(source_by_name) - mapped_sources)
    unmapped_targets = sorted(
        key for key in set(state_shapes) - mapped_targets
        if not any(key.startswith(prefix) for prefix in ignored_target_prefixes)
    )

    return DS4MappingValidation(
        mappings=mappings,
        missing_sources=missing_sources,
        missing_targets=missing_targets,
        shape_errors=shape_errors,
        unmapped_sources=unmapped_sources,
        unmapped_targets=unmapped_targets,
    )
