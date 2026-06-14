from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.components.gguf.quantized_ops import QuantizedGGUFEmbedding, QuantizedGGUFLinear
from src.components.gguf.tp_logits import distributed_argmax_local_logits, gather_sharded_logits
from src.kernels.cuda_loader import load_cuda_kernel
from src.loader.gguf.bundle import GGUFBundle
from src.models.minimax_m2.moe_runtime import MiniMaxM2DeviceResidentCache
from src.models.minimax_m2.spec import MiniMaxM2Spec


@dataclass(frozen=True)
class MiniMaxM2Args:
    n_layers: int
    dim: int
    vocab_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rope_dim: int
    rope_base: float
    n_routed_experts: int
    top_k: int
    moe_inter_dim: int
    norm_eps: float
    context_length: int
    gate_function: str = "gguf_enum:2"

    @classmethod
    def from_bundle(cls, bundle: GGUFBundle, *, n_layers: int | None = None) -> "MiniMaxM2Args":
        params = MiniMaxM2Spec().parse_params(bundle)
        return cls(
            n_layers=int(params.n_layers if n_layers is None else n_layers),
            dim=int(params.hidden_size),
            vocab_size=int(params.vocab_size),
            n_heads=int(params.n_heads),
            n_kv_heads=int(params.n_kv_heads),
            head_dim=int(params.head_dim),
            rope_dim=int(params.rope_dim or params.head_dim),
            rope_base=float(params.rope_base or 10000.0),
            n_routed_experts=int(params.n_routed_experts),
            top_k=int(params.top_k),
            moe_inter_dim=int(params.expert_intermediate_size),
            norm_eps=float(params.norm_eps or 1.0e-6),
            context_length=int(params.context_length),
            gate_function=str(params.gate_function),
        )


class RMSNorm:
    def __init__(self, weight: torch.Tensor, eps: float, *, out_dtype: torch.dtype = torch.float16):
        self.weight = weight.float().contiguous()
        self.eps = float(eps)
        self.out_dtype = out_dtype

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = xf * inv * self.weight
        return y.to(self.out_dtype)


class MiniMaxAttention:
    def __init__(
        self,
        args: MiniMaxM2Args,
        layer_id: int,
        q_proj: QuantizedGGUFLinear,
        k_proj: QuantizedGGUFLinear,
        v_proj: QuantizedGGUFLinear,
        o_proj: QuantizedGGUFLinear,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.args = args
        self.layer_id = int(layer_id)
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.q_norm = RMSNorm(q_norm_weight, args.norm_eps, out_dtype=dtype)
        self.k_norm = RMSNorm(k_norm_weight, args.norm_eps, out_dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.cache_k: torch.Tensor | None = None
        self.cache_v: torch.Tensor | None = None
        self.cache_batch = 0
        self.cache_len = 0

    def reset_cache(self, batch_size: int, max_seq_len: int) -> None:
        self.cache_batch = int(batch_size)
        self.cache_len = int(max_seq_len)
        self.cache_k = torch.empty(
            (self.cache_batch, self.cache_len, self.args.n_kv_heads, self.args.head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self.cache_v = torch.empty_like(self.cache_k)

    def _ensure_cache(self, batch_size: int, needed_len: int) -> None:
        if (
            self.cache_k is None
            or self.cache_v is None
            or self.cache_batch < batch_size
            or self.cache_len < needed_len
        ):
            new_len = max(int(needed_len), max(1, self.cache_len) * 2)
            self.reset_cache(batch_size, new_len)

    def _apply_rope(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        # fastllm/llama.cpp MiniMax uses half-split RoPE over the first rope_dim:
        # [0:rope_dim/2] rotates with [rope_dim/2:rope_dim], the tail is unchanged.
        rope_dim = int(self.args.rope_dim)
        if rope_dim <= 0:
            return x
        half = rope_dim // 2
        positions = torch.arange(start_pos, start_pos + x.size(1), device=x.device, dtype=torch.float32)
        j = torch.arange(half, device=x.device, dtype=torch.float32)
        inv = torch.pow(torch.full_like(j, float(self.args.rope_base)), -(2.0 * j) / float(rope_dim))
        freqs = positions[:, None] * inv[None, :]
        sin = torch.sin(freqs).to(x.dtype)[None, :, None, :]
        cos = torch.cos(freqs).to(x.dtype)[None, :, None, :]
        x1 = x[..., :half]
        x2 = x[..., half:rope_dim]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        if rope_dim < x.size(-1):
            return torch.cat((y1, y2, x[..., rope_dim:]), dim=-1)
        return torch.cat((y1, y2), dim=-1)

    def __call__(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        end_pos = int(start_pos) + int(seqlen)
        self._ensure_cache(bsz, end_pos)
        assert self.cache_k is not None and self.cache_v is not None

        q = self.q_norm(self.q_proj(x)).view(bsz, seqlen, self.args.n_heads, self.args.head_dim)
        use_pair_kv = os.getenv("MINIMAX_M2_PAIR_KV_GEMM", "0").lower() in {"1", "true", "yes"}
        if use_pair_kv:
            k_raw, v_raw = QuantizedGGUFLinear.pair(self.k_proj, self.v_proj, x)
        else:
            k_raw = self.k_proj(x)
            v_raw = self.v_proj(x)
        k = self.k_norm(k_raw).view(bsz, seqlen, self.args.n_kv_heads, self.args.head_dim)
        v = v_raw.view(bsz, seqlen, self.args.n_kv_heads, self.args.head_dim).to(self.dtype)
        q = self._apply_rope(q, int(start_pos))
        k = self._apply_rope(k, int(start_pos))

        self.cache_k[:bsz, start_pos:end_pos].copy_(k)
        self.cache_v[:bsz, start_pos:end_pos].copy_(v)
        k_full = self.cache_k[:bsz, :end_pos]
        v_full = self.cache_v[:bsz, :end_pos]

        q_t = q.transpose(1, 2).contiguous()  # [B, Hq, S, D]
        k_t = k_full.transpose(1, 2).contiguous()  # [B, Hkv, T, D]
        v_t = v_full.transpose(1, 2).contiguous()
        repeat = self.args.n_heads // self.args.n_kv_heads
        if repeat != 1:
            k_t = k_t.repeat_interleave(repeat, dim=1)
            v_t = v_t.repeat_interleave(repeat, dim=1)

        if seqlen == 1:
            attn_mask = None
            is_causal = False
        elif int(start_pos) == 0:
            attn_mask = None
            is_causal = True
        else:
            q_pos = torch.arange(start_pos, end_pos, device=x.device)
            k_pos = torch.arange(0, end_pos, device=x.device)
            allowed = k_pos[None, :] <= q_pos[:, None]
            attn_mask = torch.zeros((seqlen, end_pos), device=x.device, dtype=q_t.dtype)
            attn_mask.masked_fill_(~allowed, float("-inf"))
            is_causal = False
        out = F.scaled_dot_product_attention(
            q_t.to(self.dtype),
            k_t.to(self.dtype),
            v_t.to(self.dtype),
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=1.0 / math.sqrt(float(self.args.head_dim)),
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.args.n_heads * self.args.head_dim)
        return self.o_proj(out)


class MiniMaxMoE:
    def __init__(
        self,
        args: MiniMaxM2Args,
        layer_id: int,
        gate_weight: torch.Tensor,
        gate_bias: torch.Tensor | None,
        cache: MiniMaxM2DeviceResidentCache,
        *,
        dtype: torch.dtype = torch.float16,
    ):
        self.args = args
        self.layer_id = int(layer_id)
        self.gate_weight = gate_weight.float().contiguous()
        self.gate_bias = gate_bias.float().contiguous() if gate_bias is not None else None
        self.cache = cache
        self.dtype = dtype
        self._cuda = load_cuda_kernel()
        if self._cuda is None:
            raise RuntimeError("CUDA extension is required for MiniMax MoE")

    def route(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = x_flat.float() @ self.gate_weight.t()
        probs = torch.sigmoid(logits)
        select_scores = probs if self.gate_bias is None else probs + self.gate_bias
        _, indices = torch.topk(select_scores, k=int(self.args.top_k), dim=-1)
        weights = torch.gather(probs, dim=-1, index=indices)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
        return indices.to(torch.long).contiguous(), weights.float().contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1]).contiguous()
        indices, weights = self.route(x_flat)
        layer = self.cache.layer(self.layer_id)

        # Separate prefill (T>1) and decode (T=1) paths
        num_tokens = x_flat.size(0)
        if num_tokens == 1:
            # Decode: single-token DP4A path (iq2_xxs w1/w3/w2)
            # route_slots: [top_k] expert IDs for the single token
            # route_weights: [top_k] normalized weights
            route_slots = indices[0, :].contiguous()  # [top_k]
            route_weights_1d = weights[0, :].contiguous()  # [top_k]

            # Filter to local experts
            expert_start = int(self.cache.expert_start)
            expert_end = expert_start + int(self.cache.expert_count)
            local_mask = (route_slots >= expert_start) & (route_slots < expert_end)
            local_slots = route_slots[local_mask] - expert_start  # map to [0, expert_count)
            local_weights = route_weights_1d[local_mask]

            if local_slots.numel() == 0:
                y = torch.zeros((1, self.args.dim), device=x.device, dtype=torch.float32)
            else:
                grid = self.cache._quant_grid(layer.w1.type_name)
                y = self._cuda.gguf_moe_single_token_iq2_q2k_forward(
                    x_flat,
                    local_slots,
                    local_weights,
                    layer.w1.blocks,
                    layer.w3.blocks,
                    layer.w2.blocks,
                    grid,
                    0.0,
                )
        else:
            # Prefill: grouped batched path
            grouped = self._cuda.moe_group_routes(
                indices,
                weights,
                int(self.cache.expert_start),
                int(self.cache.expert_count),
            )
            _local_ids, route_tokens, route_weights, seg_starts = grouped
            if int(route_tokens.numel()) == 0:
                y = torch.zeros((x_flat.size(0), self.args.dim), device=x.device, dtype=torch.float32)
            else:
                grid = self.cache._quant_grid(layer.w1.type_name)
                y = self._cuda.gguf_moe_prefill_grouped_forward(
                    x_flat,
                    route_tokens,
                    route_weights,
                    seg_starts,
                    layer.w1.blocks,
                    layer.w3.blocks,
                    layer.w2.blocks,
                    int(layer.w1.in_dim),
                    int(layer.w1.type_id),
                    int(layer.w3.in_dim),
                    int(layer.w3.type_id),
                    int(layer.w2.in_dim),
                    int(layer.w2.type_id),
                    grid,
                    0.0,
                )

        if dist.is_available() and dist.is_initialized():
            # Decode (T=1) all_reduce is latency-bound on a tiny [1, dim] message
            # (e.g. 12 KB for fp32). Down-casting the reduce to bf16/fp16 halves the
            # payload and shaves latency off the 62 per-token collectives. Prefill
            # keeps fp32 for numerical fidelity across many tokens.
            reduce_dtype = torch.float32
            if num_tokens == 1:
                gate = os.environ.get("MINIMAX_M2_DECODE_REDUCE_DTYPE", "").strip().lower()
                if gate in {"bf16", "bfloat16"}:
                    reduce_dtype = torch.bfloat16
                elif gate in {"fp16", "float16", "half"}:
                    reduce_dtype = torch.float16
            if reduce_dtype != torch.float32:
                y = y.to(reduce_dtype)
                dist.all_reduce(y)
                y = y.to(torch.float32)
            else:
                dist.all_reduce(y)
        return y.reshape(original_shape).to(self.dtype)


class MiniMaxBlock:
    def __init__(
        self,
        args: MiniMaxM2Args,
        layer_id: int,
        attn_norm_weight: torch.Tensor,
        ffn_norm_weight: torch.Tensor,
        attention: MiniMaxAttention,
        moe: MiniMaxMoE,
        *,
        dtype: torch.dtype = torch.float16,
    ):
        self.args = args
        self.layer_id = int(layer_id)
        self.attn_norm = RMSNorm(attn_norm_weight, args.norm_eps, out_dtype=dtype)
        self.ffn_norm = RMSNorm(ffn_norm_weight, args.norm_eps, out_dtype=dtype)
        self.attention = attention
        self.moe = moe
        self.dtype = dtype

    def reset_cache(self, batch_size: int, max_seq_len: int) -> None:
        self.attention.reset_cache(batch_size, max_seq_len)

    def __call__(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        x = (x + self.attention(self.attn_norm(x), start_pos)).to(self.dtype)
        x = (x + self.moe(self.ffn_norm(x))).to(self.dtype)
        return x


class MiniMaxTransformer:
    def __init__(
        self,
        args: MiniMaxM2Args,
        embedding: QuantizedGGUFEmbedding,
        layers: list[MiniMaxBlock],
        final_norm_weight: torch.Tensor,
        lm_head: QuantizedGGUFLinear,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.args = args
        self.embedding = embedding
        self.layers = layers
        self.final_norm = RMSNorm(final_norm_weight, args.norm_eps, out_dtype=dtype)
        self.lm_head = lm_head
        self.device = device
        self.dtype = dtype
        self.max_seq_len = args.context_length

    def reset_cache(self, batch_size: int, max_seq_len: int) -> None:
        for layer in self.layers:
            layer.reset_cache(batch_size, max_seq_len)

    @torch.inference_mode()
    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        *,
        return_next_token: bool = False,
        return_hidden: bool = False,
        keep_all_positions: bool = False,
    ):
        if tokens.device != self.device:
            tokens = tokens.to(self.device)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        h = self.embedding(tokens).to(self.dtype)
        for layer in self.layers:
            h = layer(h, int(start_pos))
        h = self.final_norm(h)
        logits_input = h if keep_all_positions else h[:, -1:, :]
        logits = self.lm_head(logits_input).float()
        if return_next_token:
            next_token = distributed_argmax_local_logits(logits, row_start=int(self.lm_head.tensor.row_start))
            if not keep_all_positions:
                next_token = next_token[:, -1]
            if return_hidden:
                return next_token, (h if keep_all_positions else h[:, -1:, :])
            return next_token
        logits = gather_sharded_logits(logits, full_out_dim=int(self.args.vocab_size), row_start=int(self.lm_head.tensor.row_start))
        if return_hidden:
            return logits, h
        return logits

    __call__ = forward

    def resident_bytes(self) -> int:
        total = self.embedding.tensor.nbytes + self.lm_head.tensor.nbytes
        for layer in self.layers:
            total += layer.attention.q_proj.tensor.nbytes
            total += layer.attention.k_proj.tensor.nbytes
            total += layer.attention.v_proj.tensor.nbytes
            total += layer.attention.o_proj.tensor.nbytes
            total += layer.moe.cache.memory_bytes() // max(1, len(self.layers))
        return int(total)
