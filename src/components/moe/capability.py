from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantCapability:
    type_name: str
    header_supported: bool
    payload_reader_supported: bool
    routed_raw_supported: bool
    runtime_supported: bool
    notes: str


_CAPABILITIES = {
    "f32": QuantCapability("f32", True, True, False, True, "dense scalar tensor payload supported"),
    "f16": QuantCapability("f16", True, True, False, True, "dense scalar tensor payload supported"),
    "bf16": QuantCapability("bf16", True, True, False, True, "dense scalar tensor payload supported"),
    "i32": QuantCapability("i32", True, True, False, True, "dense scalar tensor payload supported"),
    "q8_0": QuantCapability("q8_0", True, True, True, True, "raw q8_0 binding exists for supported modules"),
    "q2_k": QuantCapability("q2_k", True, True, True, True, "routed expert raw blocks supported in existing DSV4 paths"),
    "iq2_xxs": QuantCapability("iq2_xxs", True, True, True, True, "routed expert raw blocks supported; MiniMax full runtime is deferred"),
    "iq1_m": QuantCapability("iq1_m", True, True, True, True, "routed expert raw blocks supported in existing DSV4 paths"),
    "q4_k": QuantCapability("q4_k", True, False, False, False, "header known; payload/runtime kernels deferred"),
    "q5_k": QuantCapability("q5_k", True, False, False, False, "header known; payload/runtime kernels deferred"),
}


def quant_capability(type_name: str) -> QuantCapability:
    return _CAPABILITIES.get(
        type_name,
        QuantCapability(type_name, False, False, False, False, "unknown GGUF tensor type"),
    )


def capability_status_for_role(role: str, type_name: str, *, architecture: str) -> tuple[str, str]:
    cap = quant_capability(type_name)
    if architecture == "minimax-m2":
        if role in {"routed_w1", "routed_w2", "routed_w3"} and type_name == "iq2_xxs":
            return "deferred", "iq2_xxs routed blocks are readable and low-bit resident candidates; MiniMax MoE runtime/fast path is not implemented yet"
        if role in {"attn_q", "attn_k", "attn_v", "attn_o"} and type_name == "q5_k":
            return "deferred", "q5_k attention kernels and MiniMax GQA runtime are deferred"
        if role in {"embedding", "lm_head"} and type_name == "q4_k":
            return "deferred", "q4_k embedding/head kernels are deferred"
        if type_name == "f32":
            return "deferred", "f32 tensor payload is supported, but full MiniMax runtime/generation is deferred"
    if cap.runtime_supported:
        return "supported", cap.notes
    if cap.header_supported:
        return "deferred", cap.notes
    return "unsupported", cap.notes
