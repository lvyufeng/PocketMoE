from __future__ import annotations

from collections import Counter

from src.gguf.bundle import GGUFBundle
from src.models.moe.capability import capability_status_for_role
from src.models.moe.placement import HardwareProfile, heterogeneous_expert_decision, lowbit_device_resident_decision
from src.models.moe.spec import (
    CapabilityItem,
    CapabilityReport,
    MoEArchitectureParams,
    PlacementDecision,
    SpecValidation,
    TensorMapping,
    bytes_by,
    counts_by,
    metadata_float,
    metadata_int,
)


class MiniMaxM2Spec:
    architecture = "minimax-m2"
    display_name = "MiniMax-M2"

    _GLOBAL_TENSORS = {
        "token_embd.weight": ("embed_tokens.weight", "embedding", "transpose", "q4_k"),
        "output.weight": ("lm_head.weight", "lm_head", "transpose", "q4_k"),
        "output_norm.weight": ("final_norm.weight", "final_norm", "direct", "f32"),
    }

    _LAYER_TENSORS = {
        "attn_q.weight": ("self_attn.q_proj.weight", "attn_q", "transpose", "q5_k"),
        "attn_q_norm.weight": ("self_attn.q_norm.weight", "attn_norm", "direct", "f32"),
        "attn_k.weight": ("self_attn.k_proj.weight", "attn_k", "transpose", "q5_k"),
        "attn_k_norm.weight": ("self_attn.k_norm.weight", "attn_norm", "direct", "f32"),
        "attn_v.weight": ("self_attn.v_proj.weight", "attn_v", "transpose", "q5_k"),
        "attn_output.weight": ("self_attn.o_proj.weight", "attn_o", "transpose", "q5_k"),
        "attn_norm.weight": ("input_layernorm.weight", "attn_norm", "direct", "f32"),
        "ffn_gate_inp.weight": ("mlp.router.weight", "gate", "transpose", "f32"),
        "exp_probs_b.bias": ("mlp.router.bias", "gate_bias", "direct", "f32"),
        "ffn_gate_exps.weight": ("mlp.experts.routed.w1", "routed_w1", "routed_expert_transpose", "iq2_xxs"),
        "ffn_up_exps.weight": ("mlp.experts.routed.w3", "routed_w3", "routed_expert_transpose", "iq2_xxs"),
        "ffn_down_exps.weight": ("mlp.experts.routed.w2", "routed_w2", "routed_expert_transpose", "iq2_xxs"),
        "ffn_norm.weight": ("post_attention_layernorm.weight", "ffn_norm", "direct", "f32"),
    }

    def parse_params(self, bundle: GGUFBundle) -> MoEArchitectureParams:
        md = bundle.metadata
        head_dim = metadata_int(md, "minimax-m2.attention.key_length", 0)
        return MoEArchitectureParams(
            architecture=self.architecture,
            n_layers=metadata_int(md, "minimax-m2.block_count", 0),
            hidden_size=metadata_int(md, "minimax-m2.embedding_length", 0),
            vocab_size=metadata_int(md, "tokenizer.ggml.tokens", metadata_int(md, "minimax-m2.vocab_size", 0)),
            context_length=metadata_int(md, "minimax-m2.context_length", 0),
            n_heads=metadata_int(md, "minimax-m2.attention.head_count", 0),
            n_kv_heads=metadata_int(md, "minimax-m2.attention.head_count_kv", 0),
            head_dim=head_dim,
            rope_dim=metadata_int(md, "minimax-m2.rope.dimension_count", 0) or None,
            rope_base=metadata_float(md, "minimax-m2.rope.freq_base", 0.0) or None,
            n_routed_experts=metadata_int(md, "minimax-m2.expert_count", 0),
            top_k=metadata_int(md, "minimax-m2.expert_used_count", 0),
            expert_intermediate_size=metadata_int(md, "minimax-m2.expert_feed_forward_length", 0),
            n_shared_experts=0,
            gate_function=f"gguf_enum:{metadata_int(md, 'minimax-m2.expert_gating_func', 0)}",
            attention_kind="gqa_separate_qkv",
            routed_expert_layout="packed_3d",
            norm_eps=metadata_float(md, "minimax-m2.attention.layer_norm_rms_epsilon", 0.0) or None,
        )

    def classify_tensor(self, name: str) -> str:
        if name in self._GLOBAL_TENSORS:
            return self._GLOBAL_TENSORS[name][1]
        if not name.startswith("blk."):
            return "other"
        suffix = name.split(".", 2)[2] if len(name.split(".", 2)) == 3 else ""
        return self._LAYER_TENSORS.get(suffix, ("", "other", "", ""))[1]

    def build_tensor_mappings(self, bundle: GGUFBundle) -> list[TensorMapping]:
        params = self.parse_params(bundle)
        mappings: list[TensorMapping] = []
        for source_name, (logical, role, transform, _expected_type) in self._GLOBAL_TENSORS.items():
            mappings.append(TensorMapping(source_name, logical, role, transform))
        for layer in range(params.n_layers):
            for suffix, (logical_suffix, role, transform, _expected_type) in self._LAYER_TENSORS.items():
                source = f"blk.{layer}.{suffix}"
                logical = f"layers.{layer}.{logical_suffix}"
                if role in {"routed_w1", "routed_w2", "routed_w3"}:
                    mappings.append(TensorMapping(source, logical, role, transform, layer=layer))
                else:
                    mappings.append(TensorMapping(source, logical, role, transform, layer=layer))
        return mappings

    def _expected_shape_ok(self, source_name: str, dims: tuple[int, ...], params: MoEArchitectureParams) -> bool:
        h = params.hidden_size
        e = params.n_routed_experts
        i = params.expert_intermediate_size
        q = params.n_heads * params.head_dim
        kv = params.n_kv_heads * params.head_dim
        v = params.vocab_size
        if source_name == "token_embd.weight" or source_name == "output.weight":
            return dims == (h, v)
        if source_name == "output_norm.weight":
            return dims == (h,)
        suffix = source_name.split(".", 2)[2]
        expected = {
            "attn_q.weight": (h, q),
            "attn_q_norm.weight": (q,),
            "attn_k.weight": (h, kv),
            "attn_k_norm.weight": (kv,),
            "attn_v.weight": (h, kv),
            "attn_output.weight": (q, h),
            "attn_norm.weight": (h,),
            "ffn_gate_inp.weight": (h, e),
            "exp_probs_b.bias": (e,),
            "ffn_gate_exps.weight": (h, i, e),
            "ffn_up_exps.weight": (h, i, e),
            "ffn_down_exps.weight": (i, h, e),
            "ffn_norm.weight": (h,),
        }.get(suffix)
        return expected is None or dims == expected

    def validate_bundle(self, bundle: GGUFBundle) -> SpecValidation:
        params = self.parse_params(bundle)
        errors: list[str] = []
        warnings: list[str] = []
        names = bundle.tensors_by_name
        mapped_sources: set[str] = set()
        role_counts: Counter[str] = Counter()

        if bundle.metadata.get("general.architecture") != self.architecture:
            errors.append(f"general.architecture expected {self.architecture!r}, got {bundle.metadata.get('general.architecture')!r}")
        for key in (
            "minimax-m2.block_count",
            "minimax-m2.embedding_length",
            "minimax-m2.expert_count",
            "minimax-m2.expert_used_count",
            "minimax-m2.expert_feed_forward_length",
        ):
            if key not in bundle.metadata:
                errors.append(f"missing metadata {key}")
        if params.n_layers <= 0:
            errors.append("minimax-m2.block_count must be positive")

        for source_name, (_logical, role, _transform, expected_type) in self._GLOBAL_TENSORS.items():
            tensor = names.get(source_name)
            if tensor is None:
                errors.append(f"missing tensor {source_name}")
                continue
            mapped_sources.add(source_name)
            role_counts[role] += 1
            if tensor.type_name != expected_type:
                errors.append(f"{source_name} expected {expected_type}, got {tensor.type_name}")
            if not self._expected_shape_ok(source_name, tuple(tensor.dimensions), params):
                errors.append(f"{source_name} unexpected shape {tuple(tensor.dimensions)}")

        for layer in range(params.n_layers):
            for suffix, (_logical, role, _transform, expected_type) in self._LAYER_TENSORS.items():
                source_name = f"blk.{layer}.{suffix}"
                tensor = names.get(source_name)
                if tensor is None:
                    errors.append(f"missing tensor {source_name}")
                    continue
                mapped_sources.add(source_name)
                role_counts[role] += 1
                if tensor.type_name != expected_type:
                    errors.append(f"{source_name} expected {expected_type}, got {tensor.type_name}")
                if not self._expected_shape_ok(source_name, tuple(tensor.dimensions), params):
                    errors.append(f"{source_name} unexpected shape {tuple(tensor.dimensions)}")

        unmapped = sorted(set(names) - mapped_sources)
        if unmapped:
            warnings.append(f"{len(unmapped)} unmapped tensors, e.g. {unmapped[:10]}")
        return SpecValidation(
            ok=not errors,
            errors=errors,
            warnings=warnings,
            mapped_sources=len(mapped_sources),
            unmapped_sources=unmapped,
            role_counts=dict(role_counts),
        )

    def capability_report(self, bundle: GGUFBundle, *, gpu_count: int = 4, gpu_memory_gib: float = 22.0) -> CapabilityReport:
        params = self.parse_params(bundle)
        role_by_name = {t.name: self.classify_tensor(t.name) for t in bundle.tensors}
        tensor_role_counts = counts_by(bundle.tensors, lambda t: role_by_name[t.name])
        tensor_type_counts = counts_by(bundle.tensors, lambda t: t.type_name)
        bytes_by_type = bytes_by(bundle.tensors, lambda t: t.type_name)
        bytes_by_role = bytes_by(bundle.tensors, lambda t: role_by_name[t.name])

        caps: list[CapabilityItem] = []
        seen: set[tuple[str, str]] = set()
        for tensor in bundle.tensors:
            role = role_by_name[tensor.name]
            key = (role, tensor.type_name)
            if key in seen:
                continue
            seen.add(key)
            status, reason = capability_status_for_role(role, tensor.type_name, architecture=self.architecture)
            caps.append(CapabilityItem(f"{role}:{tensor.type_name}", status, reason))
        caps.append(CapabilityItem("generation", "deferred", "MiniMax-M2 generation is not implemented yet"))

        tensor_bytes = sum(int(t.nbytes or 0) for t in bundle.tensors)
        routed_bytes = sum(
            int(t.nbytes or 0)
            for t in bundle.tensors
            if role_by_name[t.name] in {"routed_w1", "routed_w2", "routed_w3"}
        )
        hardware = HardwareProfile(gpu_count=gpu_count, gpu_memory_gib=gpu_memory_gib)
        placements: list[PlacementDecision] = [
            lowbit_device_resident_decision(tensor_bytes, hardware),
            heterogeneous_expert_decision(routed_bytes),
        ]
        return CapabilityReport(
            architecture=self.architecture,
            params=params,
            tensor_type_counts=tensor_type_counts,
            tensor_role_counts=tensor_role_counts,
            bytes_by_type=bytes_by_type,
            bytes_by_role=bytes_by_role,
            capabilities=caps,
            placements=placements,
        )
