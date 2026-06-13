from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from src.loader.gguf.bundle import GGUFBundle
from src.components.moe.capability import capability_status_for_role
from src.components.moe.placement import HardwareProfile, heterogeneous_expert_decision, lowbit_device_resident_decision
from src.loader.mappings.minimax_m2 import GLOBAL_TENSORS, LAYER_TENSORS, build_tensor_mappings, classify_tensor_name
from src.components.moe.spec import (
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

if TYPE_CHECKING:
    import torch

    from src.runtime.generation import GGUFTokenRuntime


class MiniMaxM2Spec:
    architecture = "minimax-m2"
    display_name = "MiniMax-M2"

    _GLOBAL_TENSORS = GLOBAL_TENSORS
    _LAYER_TENSORS = LAYER_TENSORS

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
        return classify_tensor_name(name)

    def build_tensor_mappings(self, bundle: GGUFBundle) -> list[TensorMapping]:
        return build_tensor_mappings(self.parse_params(bundle).n_layers)

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
        caps.append(CapabilityItem("generation", "candidate", "MiniMax-M2 TP4 raw-block CUDA greedy generation is implemented"))

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

    def build_token_runtime(
        self,
        bundle: GGUFBundle,
        *,
        world: int,
        rank: int,
        device: "torch.device",
        dtype: "torch.dtype",
        n_layers: int | None,
        gpu_memory_gib: float,
    ) -> "GGUFTokenRuntime":
        """Build the TP rank-local MiniMax-M2 model for raw-block CUDA decode.

        Deferred imports keep this spec cheap to import for inspection/registry
        use; the heavy MiniMax runtime is only pulled in when actually loading.
        """
        import time

        from src.runtime.generation import GGUFTokenRuntime
        from src.models.minimax_m2.gguf_model import load_minimax_m2_gguf_model
        from src.models.minimax_m2.moe_planning import (
            build_minimax_m2_tp_routed_resident_plan,
            minimax_m2_tp_expert_range,
        )

        if world > 1:
            tp_plan = build_minimax_m2_tp_routed_resident_plan(
                bundle, tp_world=world, gpu_memory_gib=float(gpu_memory_gib)
            )
            if not tp_plan.ok:
                raise RuntimeError(f"MiniMax-M2 TP plan is not valid: {tp_plan.errors[:5]}")
            rplan = tp_plan.ranks[rank]
            expert_start, expert_count = rplan.expert_start, rplan.expert_count
        else:
            expert_start, expert_count = minimax_m2_tp_expert_range(
                metadata_int(bundle.metadata, "minimax-m2.expert_count", 256), 1, 0
            )

        t_load = time.perf_counter()
        model, _info = load_minimax_m2_gguf_model(
            bundle,
            device=device,
            dtype=dtype,
            n_layers=n_layers,
            expert_start=expert_start,
            expert_count=expert_count,
            preload_moe=True,
        )
        load_seconds = time.perf_counter() - t_load

        eos = bundle.metadata.get("tokenizer.ggml.eos_token_id")
        eos_id = int(eos) if isinstance(eos, int) else None
        return GGUFTokenRuntime(
            model=model,
            expert_start=int(expert_start),
            expert_count=int(expert_count),
            eos_token_id=eos_id,
            load_seconds=float(load_seconds),
        )
