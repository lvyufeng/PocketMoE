"""MiniMax-M2 checkpoint source-name mapping tables."""

from __future__ import annotations

from src.components.moe.spec import TensorMapping

GLOBAL_TENSORS = {
    "token_embd.weight": ("embed_tokens.weight", "embedding", "transpose", "q4_k"),
    "output.weight": ("lm_head.weight", "lm_head", "transpose", "q4_k"),
    "output_norm.weight": ("final_norm.weight", "final_norm", "direct", "f32"),
}

LAYER_TENSORS = {
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


def classify_tensor_name(name: str) -> str:
    if name in GLOBAL_TENSORS:
        return GLOBAL_TENSORS[name][1]
    if not name.startswith("blk."):
        return "other"
    suffix = name.split(".", 2)[2] if len(name.split(".", 2)) == 3 else ""
    return LAYER_TENSORS.get(suffix, ("", "other", "", ""))[1]


def build_tensor_mappings(n_layers: int) -> list[TensorMapping]:
    mappings: list[TensorMapping] = []
    for source_name, (logical, role, transform, _expected_type) in GLOBAL_TENSORS.items():
        mappings.append(TensorMapping(source_name, logical, role, transform))
    for layer in range(int(n_layers)):
        for suffix, (logical_suffix, role, transform, _expected_type) in LAYER_TENSORS.items():
            source = f"blk.{layer}.{suffix}"
            logical = f"layers.{layer}.{logical_suffix}"
            mappings.append(TensorMapping(source, logical, role, transform, layer=layer))
    return mappings
