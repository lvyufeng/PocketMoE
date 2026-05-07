import math
import os
import time
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from functools import lru_cache
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd.profiler import record_function

from src.moe.cpu_backend import CPURoutedExpertsBackend, start_in_process_cpu_moe_server
from src.moe.gpu_prefill_backend import GPUPrefillMoEBackend
from src.kernels.ops import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn, Packed4BitWeightAlongK, _quantize_int8_weight_torch, soft_bf16_weight_gemm_int8, soft_bf16_weight_gemm_int8_pair_cuda_ext, _SHARED_EXPERT_PAIR_INT8_CUDA, _dequant_fp4_weight_torch, soft_fp8_blockfp8_weight_dequant
from src.kernels.cuda_loader import load_cuda_kernel


world_size = 1
rank = 0
_cpu_moe_server_ipc = None
_cpu_moe_server_ipc_name = None
_cpu_moe_server_last_seq = 0
_cpu_moe_server_predict_seq = 0
block_size = 128
fp4_block_size = 32
default_dtype = torch.bfloat16
scale_fmt = None
scale_dtype = torch.float32


def _get_cpu_moe_server_shm_name() -> str:
    global _cpu_moe_server_ipc_name
    if _cpu_moe_server_ipc_name is not None:
        return _cpu_moe_server_ipc_name
    default_name = os.getenv("DEEPSEEK_CPU_MOE_SERVER_SHM", "dsv4_cpu_moe_server")
    if dist.is_initialized() and _env_enabled("DEEPSEEK_CPU_MOE_INPROC_SERVER") and world_size > 1:
        objects = [default_name if rank == 0 else None]
        dist.broadcast_object_list(objects, src=0)
        _cpu_moe_server_ipc_name = str(objects[0])
    else:
        _cpu_moe_server_ipc_name = default_name
    return _cpu_moe_server_ipc_name


def _get_cpu_moe_server_ipc(dim: int, topk: int):
    global _cpu_moe_server_ipc, _cpu_moe_server_ipc_name
    from src.moe.ipc import CPUMoESharedMemory
    name = _get_cpu_moe_server_shm_name()
    if _cpu_moe_server_ipc is None or _cpu_moe_server_ipc_name != name:
        _cpu_moe_server_ipc = CPUMoESharedMemory(name, dim, topk, create=False)
        _cpu_moe_server_ipc_name = name
    return _cpu_moe_server_ipc


def _env_enabled(name: str) -> bool:
    active_name = name.replace("DEEPSEEK_", "DEEPSEEK_ACTIVE_", 1)
    if active_name in os.environ:
        return os.getenv(active_name, "0").lower() in {"1", "true", "yes"}
    return os.getenv(name, "0").lower() in {"1", "true", "yes"}


def _any_phase_env_enabled(suffix: str) -> bool:
    return any(
        os.getenv(f"DEEPSEEK_PD_{phase}_{suffix}", "0").lower() in {"1", "true", "yes"}
        for phase in ("PREFILL", "DECODE")
    )

def _phase_env_enabled(suffix: str, default: bool = False) -> bool:
    phase = os.getenv("DEEPSEEK_PD_ACTIVE_PHASE", "")
    active_key = f"DEEPSEEK_ACTIVE_{suffix}"
    if active_key in os.environ:
        return _env_enabled(active_key)
    if phase in {"prefill", "decode"}:
        phase_key = f"DEEPSEEK_PD_{phase.upper()}_{suffix}"
        if phase_key in os.environ:
            return _env_enabled(phase_key)
    global_key = f"DEEPSEEK_{suffix}"
    if global_key in os.environ:
        return _env_enabled(global_key)
    return default


def _pack_fp4_weight_rows_for_tile_decode(
    weight: Packed4BitWeightAlongK,
    scale: torch.Tensor,
    tile_rows: int = 64,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = weight.layout_tensor.contiguous()
    out_features, packed_k = raw.shape
    bytes_per_block = block_size // 2
    out_blocks = packed_k // bytes_per_block
    num_tiles = (out_features + tile_rows - 1) // tile_rows
    packed = torch.zeros((num_tiles, out_blocks, bytes_per_block, tile_rows), dtype=torch.uint8, device="cpu")
    packed_scale = torch.zeros((num_tiles, out_blocks, tile_rows), dtype=torch.float32, device="cpu")
    raw_blocks = raw.view(out_features, out_blocks, bytes_per_block)
    scale = scale.contiguous().view(out_features, out_blocks)
    for tile_idx in range(num_tiles):
        start = tile_idx * tile_rows
        end = min(start + tile_rows, out_features)
        rows = end - start
        packed[tile_idx, :, :, :rows] = raw_blocks[start:end].permute(1, 2, 0)
        packed_scale[tile_idx, :, :rows] = scale[start:end].transpose(0, 1)
    return packed.view(-1), packed_scale.view(-1)


FP4_CPU_TILE_ROWS = 64


@contextmanager
def set_dtype(dtype):
    """Temporarily override torch default dtype, restoring it on exit (even if an exception occurs)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)

@dataclass
class ModelArgs:
    """Model hyperparameters. Field names match the config JSON keys."""
    max_batch_size: int = 4
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"
    expert_dtype: Literal[None, "fp4", "int8"] = None
    scale_dtype: Literal["fp32", "fp8"] = "fp8"
    routed_experts_device: Literal["gpu", "cpu"] = "gpu"
    attn_int8: bool = False
    shared_expert_int8: bool = False
    mtp_int8: bool = False
    preloaded_attn_int8: bool = False
    preloaded_shared_expert_int8: bool = False
    preload_wq_a_int8: bool = False
    preload_wq_b_int8: bool = False
    preload_wkv_int8: bool = False
    preload_wo_a_int8: bool = False
    preload_wo_b_int8: bool = False
    preload_indexer_wq_b_int8: bool = False
    preload_shared_w1_int8: bool = False
    preload_shared_w2_int8: bool = False
    preload_shared_w3_int8: bool = False
    preload_mtp_e_proj_int8: bool = False
    preload_mtp_h_proj_int8: bool = False
    preload_routed_fp4_dequant: bool = False
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 4096
    n_layers: int = 7
    n_hash_layers: int = 0
    n_mtp_layers: int = 1
    n_heads: int = 64
    # moe
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.
    swiglu_limit: float = 0.
    # mqa
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
    # yarn
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    # index
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    # hc
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6


class ParallelEmbedding(nn.Module):
    """Embedding sharded along the vocab dimension. Each rank holds vocab_size // world_size rows.
    Out-of-range indices are zero-masked before all_reduce to combine partial embeddings."""
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Dispatches to fp4_gemm / fp8_gemm / F.linear based on weight dtype.
    For quantized weights, x is first quantized to FP8 via act_quant."""
    assert bias is None

    if weight.dtype == torch.float4_e2m1fn_x2:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp4_gemm(x, s, weight, weight.scale, scale_dtype)
    elif weight.dtype == torch.float8_e4m3fn:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp8_gemm(x, s, weight, weight.scale, scale_dtype)
    elif weight.dtype == torch.int8:
        return soft_bf16_weight_gemm_int8(x, weight, weight.scale)
    else:
        return F.linear(x, weight)


class Linear(nn.Module):
    """Linear layer supporting BF16, FP8, and FP4 weight formats with per-block scaling."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.online_int8_enabled = False
        self._online_int8_ready = False
        self.phase_env_suffix: str | None = None
        self.online_int8_chunk_tokens = max(0, int(os.getenv("DEEPSEEK_ONLINE_INT8_CHUNK_TOKENS", "4096")))
        dtype = dtype or default_dtype
        if dtype == torch.float4_e2m1fn_x2:
            # FP4: weight is [out, in//2] in float4_e2m1fn_x2, logically [out, in] in fp4
            # Scale is [out, in//32] in float8_e8m0fnu (1 scale per 32 fp4 elements along K)
            self.weight = nn.Parameter(torch.empty(out_features, in_features // 2, dtype=torch.float4_e2m1fn_x2))
            scale_out_features = out_features
            scale_in_features = in_features // fp4_block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float8_e8m0fnu))
        elif dtype == torch.float8_e4m3fn:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float8_e8m0fnu))
        elif dtype == torch.int8:
            self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8))
            self.register_buffer("scale", torch.empty(out_features, dtype=torch.float32))
            self.weight.scale = self.scale
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def set_int8_storage(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        if weight.dtype != torch.int8:
            raise TypeError(f"expected int8 weight, got {weight.dtype}")
        if scale.dtype != torch.float32:
            raise TypeError(f"expected float32 scale, got {scale.dtype}")
        if weight.shape != self.weight.shape:
            raise ValueError(f"weight shape mismatch: got {tuple(weight.shape)}, expected {tuple(self.weight.shape)}")
        if scale.shape != self.scale.shape:
            raise ValueError(f"scale shape mismatch: got {tuple(scale.shape)}, expected {tuple(self.scale.shape)}")
        self._buffers["weight"] = weight
        self._buffers["scale"] = scale
        self.weight = weight
        self.scale = scale
        self.weight.scale = self.scale
        self._online_int8_ready = False

    def enable_online_int8(self) -> None:
        self.online_int8_enabled = True

    def set_preloaded_int8(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        self.online_int8_enabled = True
        self.register_buffer("online_int8_weight", weight.detach().to(device=self.weight.device, dtype=torch.int8), persistent=False)
        self.register_buffer("online_int8_scale", scale.detach().to(device=self.weight.device, dtype=torch.float32), persistent=False)
        self._online_int8_ready = True

    def _online_int8_active(self) -> bool:
        if self.phase_env_suffix is not None:
            return _phase_env_enabled(self.phase_env_suffix, self.online_int8_enabled)
        return self.online_int8_enabled

    def _ensure_online_int8(self) -> None:
        if not self._online_int8_active() or self._online_int8_ready:
            return
        if self.weight.dtype == torch.int8:
            self.register_buffer("online_int8_weight", self.weight.detach(), persistent=False)
            self.register_buffer("online_int8_scale", self.weight.scale.detach(), persistent=False)
            self._online_int8_ready = True
            return
        if self.weight.dtype == torch.float8_e4m3fn:
            weight = soft_fp8_blockfp8_weight_dequant(self.weight.detach(), self.weight.scale.detach()).float()
        else:
            weight = self.weight.detach().float()
        weight_q, weight_s = _quantize_int8_weight_torch(weight)
        self.register_buffer("online_int8_weight", weight_q, persistent=False)
        self.register_buffer("online_int8_scale", weight_s, persistent=False)
        self._online_int8_ready = True

    def _online_int8_forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_online_int8()
        chunk = self.online_int8_chunk_tokens
        if chunk > 0 and x.numel() // x.shape[-1] > chunk:
            flat = x.reshape(-1, x.shape[-1])
            y = torch.empty((flat.size(0), self.out_features), device=x.device, dtype=torch.get_default_dtype())
            for start in range(0, flat.size(0), chunk):
                end = min(start + chunk, flat.size(0))
                y[start:end].copy_(soft_bf16_weight_gemm_int8(
                    flat[start:end].contiguous(),
                    self.online_int8_weight,
                    self.online_int8_scale,
                ))
            return y.view(*x.shape[:-1], self.out_features)
        return soft_bf16_weight_gemm_int8(x, self.online_int8_weight, self.online_int8_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._online_int8_active():
            return self._online_int8_forward(x)
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """Shards output dim across TP ranks. No all-reduce needed on output."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def _online_int8_forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_online_int8()
        chunk = self.online_int8_chunk_tokens
        if chunk > 0 and x.numel() // x.shape[-1] > chunk:
            flat = x.reshape(-1, x.shape[-1])
            y = torch.empty((flat.size(0), self.out_features), device=x.device, dtype=torch.get_default_dtype())
            for start in range(0, flat.size(0), chunk):
                end = min(start + chunk, flat.size(0))
                y[start:end].copy_(soft_bf16_weight_gemm_int8(
                    flat[start:end].contiguous(),
                    self.online_int8_weight,
                    self.online_int8_scale,
                ))
            return y.view(*x.shape[:-1], self.out_features)
        return soft_bf16_weight_gemm_int8(x, self.online_int8_weight, self.online_int8_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._online_int8_active():
            return self._online_int8_forward(x)
        return linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    """Shards input dim across TP ranks. All-reduce on output to sum partial results."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._online_int8_active():
            y = self._online_int8_forward(x)
        else:
            y = linear(x, self.weight, None)
        if world_size > 1:
            reduce_dtype = x.dtype
            y_reduce = y.to(reduce_dtype)
            dist.all_reduce(y_reduce)
            y = y_reduce.to(torch.float32)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # rmsnorm in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """Precomputes complex exponentials for rotary embeddings with YaRN scaling.
    When original_seq_len > 0, applies frequency interpolation with a smooth
    linear ramp between beta_fast and beta_slow correction ranges."""

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Applies rotary positional embeddings in-place. Uses conjugate for inverse (de-rotation)."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    ext = load_cuda_kernel() if x.is_cuda and x.size(-1) == 128 else None
    if ext is not None and hasattr(ext, "hadamard128_forward"):
        return ext.hadamard128_forward(x)
    return x


@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos == 0:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
        return matrix.unsqueeze(0).expand(bsz, -1, -1)
    # start_pos > 0: produce one window-row per query position. Matrix shape
    # is [seqlen, window]. For seqlen=1 this collapses to the original
    # single-row implementation. For seqlen>1 (speculative-verify decode)
    # we additionally mask out slots that the kv_cache write for a *later*
    # position in the same chunk has already overwritten — for query at
    # local index i, those are slots holding positions
    # (start_pos + i+1, ..., start_pos + seqlen - 1).
    rows = []
    for i in range(seqlen):
        cur_pos = start_pos + i
        if cur_pos >= window_size - 1:
            cur = cur_pos % window_size
            row = torch.cat([torch.arange(cur + 1, window_size), torch.arange(0, cur + 1)], dim=0)
        else:
            row = F.pad(torch.arange(cur_pos + 1), (0, window_size - cur_pos - 1), value=-1)
        if seqlen > 1:
            for j in range(i + 1, seqlen):
                future_slot = (start_pos + j) % window_size
                row = torch.where(row == future_slot, torch.full_like(row, -1), row)
        rows.append(row)
    matrix = torch.stack(rows, dim=0)  # [seqlen, window]
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(2)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    if start_pos == 0:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
        return matrix.unsqueeze(0).expand(bsz, -1, -1)
    # start_pos > 0
    if seqlen == 1:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
        return matrix.unsqueeze(0).expand(bsz, -1, -1)
    rows = []
    for i in range(seqlen):
        cur_pos = start_pos + i
        rows.append(torch.arange(0, (cur_pos + 1) // ratio) + offset)
    # Pad rows to the same length so we can stack.
    max_len = max(r.numel() for r in rows)
    padded = []
    for r in rows:
        if r.numel() < max_len:
            pad = torch.full((max_len - r.numel(),), -1, dtype=r.dtype)
            r = torch.cat([r, pad], dim=0)
        padded.append(r)
    matrix = torch.stack(padded, dim=0)  # [seqlen, max_len]
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


class Compressor(nn.Module):
    """Compresses KV cache via learned gated pooling over `compress_ratio` consecutive tokens.
    When overlap=True (ratio==4), uses overlapping windows for smoother compression boundaries."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4, head_dim: int = 512, rotate: bool = False):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap

        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        # wkv and wgate in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.
        # When overlap, the first half of dims is for overlapping compression, second half for normal.
        self.wkv = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.wgate = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.kv_cache: torch.Tensor = None  # assigned lazily from Attention.kv_cache
        # State buffers for decode-phase incremental compression.
        # With overlap: state[:, :ratio] = overlapping window, state[:, ratio:] = current window.
        self.register_buffer("kv_state", torch.zeros(args.max_batch_size, coff * compress_ratio, coff * self.head_dim, dtype=torch.float32), persistent=False)
        self.register_buffer("score_state", torch.full((args.max_batch_size, coff * compress_ratio, coff * self.head_dim), float("-inf"), dtype=torch.float32), persistent=False)
        self.freqs_cis: torch.Tensor = None

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        # tensor: [b,s,r,2d]
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        # compression need fp32
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff-ratio : cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff-ratio : cutoff] + self.ape
            if remainder > 0:
                kv, self.kv_state[:bsz, offset : offset+remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:bsz, offset : offset+remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return
        kv = self.norm(kv.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        if self.rotate:
            kv = rotate_activation(kv)
            fp4_act_quant(kv, fp4_block_size, True)
        else:
            act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


class Indexer(torch.nn.Module):
    """Selects top-k compressed KV positions for sparse attention via learned scoring.
    Has its own Compressor (with Hadamard rotation) to build compressed KV for scoring."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.replicated_c4_indexer = _env_enabled("DEEPSEEK_REPLICATED_C4_INDEXER")
        self.n_local_heads = self.n_heads if self.replicated_c4_indexer else self.n_heads // world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        indexer_linear = Linear if self.replicated_c4_indexer else ColumnParallelLinear
        self.wq_b = indexer_linear(self.q_lora_rank, self.n_heads * self.head_dim, dtype=torch.int8 if args.attn_int8 else None)
        self.wq_b.phase_env_suffix = "INDEXER_WQ_B_INT8"
        self.wq_b.replicated_c4_indexer = self.replicated_c4_indexer
        preload_indexer_wq_b = args.preloaded_attn_int8 or args.preload_indexer_wq_b_int8
        preload_indexer_wq_b = preload_indexer_wq_b or _any_phase_env_enabled("INDEXER_WQ_B_INT8")
        if preload_indexer_wq_b:
            self.wq_b.enable_online_int8()
        if _env_enabled("DEEPSEEK_INDEXER_WQ_B_INT8"):
            self.wq_b.enable_online_int8()
        self.weights_proj = indexer_linear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.weights_proj.replicated_c4_indexer = self.replicated_c4_indexer
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio

        self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len // compress_ratio, self.head_dim), persistent=False)
        self.freqs_cis = None
        self.fused_c4_indexer_enabled = _env_enabled("DEEPSEEK_FUSED_C4_INDEXER_CUDA")
        self._fused_c4_indexer_ext = load_cuda_kernel() if self.fused_c4_indexer_enabled else None
        self._prefill_chunk_tokens = max(
            1, int(os.getenv("DEEPSEEK_INDEXER_PREFILL_CHUNK_TOKENS", "512"))
        )

    def _fused_decode_forward(self, x: torch.Tensor, q: torch.Tensor, start_pos: int, offset: int):
        if self._fused_c4_indexer_ext is None or start_pos == 0 or x.size(1) != 1 or self.compress_ratio != 4:
            return None
        freqs_cis = self.freqs_cis[start_pos:start_pos + 1]
        rd = self.rope_head_dim
        q = q.contiguous()
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        kv = self.compressor.wkv(x.float()).contiguous()
        score = self.compressor.wgate(x.float()).contiguous()
        weights = (self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)).contiguous()
        freqs = torch.view_as_real(self.freqs_cis[start_pos + 1 - self.compress_ratio]).flatten().contiguous()
        result = self._fused_c4_indexer_ext.fused_c4_indexer_decode_forward(
            q,
            kv,
            score,
            weights,
            self.compressor.ape,
            self.compressor.norm.weight,
            freqs,
            self.compressor.kv_state,
            self.compressor.score_state,
            self.kv_cache,
            int(start_pos),
            int(offset),
            int(self.index_topk),
            float(self.compressor.norm.eps),
            world_size > 1 and not self.replicated_c4_indexer,
        )
        if world_size > 1 and not self.replicated_c4_indexer:
            dist.all_reduce(result)
            return self._fused_c4_indexer_ext.c4_topk_from_scores(
                result.contiguous(),
                int(offset),
                int(min(self.index_topk, (start_pos + 1) // self.compress_ratio)),
            )
        return result

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        if self.fused_c4_indexer_enabled:
            fused = self._fused_decode_forward(x, q, start_pos, offset)
            if fused is not None:
                return fused
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        # use fp4 simulation for q and kv in indexer
        fp4_act_quant(q, fp4_block_size, True)
        if start_pos > 0 and seqlen > 1:
            # Decode-time speculative verify: advance compressor state position
            # by position. The compressor's stateful kv_state/score_state
            # depend on `start_pos % ratio`, so feeding a length-2 chunk
            # directly would confuse it.
            for i in range(seqlen):
                self.compressor(x[:, i:i+1], start_pos + i)
        else:
            self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
        kv_window = self.kv_cache[:bsz, :end_pos // ratio]
        if seqlen > self._prefill_chunk_tokens:
            t_dim = kv_window.size(1)
            index_score = q.new_empty((bsz, seqlen, t_dim))
            chunk = self._prefill_chunk_tokens
            for qs in range(0, seqlen, chunk):
                qe = min(qs + chunk, seqlen)
                chunk_score = torch.einsum("bshd,btd->bsht", q[:, qs:qe], kv_window)
                chunk_score = (chunk_score.relu_() * weights[:, qs:qe].unsqueeze(-1)).sum(dim=2)
                index_score[:, qs:qe] = chunk_score
                del chunk_score
        else:
            index_score = torch.einsum("bshd,btd->bsht", q, kv_window)
            index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if world_size > 1 and not self.replicated_c4_indexer:
            dist.all_reduce(index_score)
        if start_pos == 0:
            t_dim = index_score.size(-1)
            t_idx = torch.arange(t_dim, device=index_score.device).unsqueeze(0)
            chunk = self._prefill_chunk_tokens
            for qs in range(0, seqlen, chunk):
                qe = min(qs + chunk, seqlen)
                valid_per_row = ((torch.arange(qs + 1, qe + 1, device=index_score.device)) // ratio).unsqueeze(-1)
                mask = t_idx >= valid_per_row
                index_score[:, qs:qe].masked_fill_(mask.unsqueeze(0), float("-inf"))
        elif seqlen > 1:
            # Speculative-verify decode: each query position `i` may only attend
            # to compressor cells `[0 .. (start_pos+i+1)//ratio)`. Mask the
            # later ones (which exist in the slice because end_pos//ratio
            # rounds up over the seqlen=2 window).
            t_dim = index_score.size(-1)
            t_idx = torch.arange(t_dim, device=index_score.device).unsqueeze(0)  # [1, t]
            chunk = self._prefill_chunk_tokens
            for qs in range(0, seqlen, chunk):
                qe = min(qs + chunk, seqlen)
                valid_per_row = torch.tensor(
                    [min(t_dim, (start_pos + i + 1) // ratio) for i in range(qs, qe)],
                    device=index_score.device,
                )
                mask = t_idx >= valid_per_row.unsqueeze(-1)  # [chunk, t]
                index_score[:, qs:qe].masked_fill_(mask.unsqueeze(0), float("-inf"))
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            chunk = self._prefill_chunk_tokens
            for qs in range(0, seqlen, chunk):
                qe = min(qs + chunk, seqlen)
                part = topk_idxs[:, qs:qe]
                valid_per_row = ((torch.arange(qs + 1, qe + 1, device=part.device)) // ratio).view(1, qe - qs, 1)
                invalid = part >= valid_per_row
                part.add_(offset)
                part.masked_fill_(invalid, -1)
        elif seqlen > 1:
            # Per-row masking: position i can only validly index cells up to
            # (start_pos+i+1)//ratio. Beyond that, mark as -1.
            chunk = self._prefill_chunk_tokens
            for qs in range(0, seqlen, chunk):
                qe = min(qs + chunk, seqlen)
                part = topk_idxs[:, qs:qe]
                valid_per_row = torch.tensor(
                    [(start_pos + i + 1) // ratio for i in range(qs, qe)],
                    device=part.device,
                ).view(1, qe - qs, 1)
                invalid = part >= valid_per_row
                part.add_(offset)
                part.masked_fill_(invalid, -1)
        else:
            topk_idxs += offset
        return topk_idxs


class Attention(nn.Module):
    """Multi-head Latent Attention (MLA) with sliding window + optional KV compression.
    Uses low-rank Q projection (wq_a -> q_norm -> wq_b) and grouped low-rank O projection."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups // world_size
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        preload_wq_a = args.preloaded_attn_int8 or args.preload_wq_a_int8
        preload_wq_b = args.preloaded_attn_int8 or args.preload_wq_b_int8
        preload_wkv = args.preloaded_attn_int8 or args.preload_wkv_int8
        preload_wo_a = args.preloaded_attn_int8 or args.preload_wo_a_int8
        preload_wo_b = args.preloaded_attn_int8 or args.preload_wo_b_int8
        preload_wq_a = preload_wq_a or _any_phase_env_enabled("WQ_A_INT8")
        preload_wq_b = preload_wq_b or _any_phase_env_enabled("WQ_B_INT8")
        preload_wkv = preload_wkv or _any_phase_env_enabled("WKV_INT8")
        preload_wo_a = preload_wo_a or _any_phase_env_enabled("WO_A_INT8")
        preload_wo_b = preload_wo_b or _any_phase_env_enabled("WO_B_INT8")

        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
        self.wq_a = Linear(self.dim, self.q_lora_rank, dtype=torch.int8 if args.attn_int8 else None)
        self.wq_a.phase_env_suffix = "WQ_A_INT8"
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim, dtype=torch.int8 if args.attn_int8 else None)
        self.wq_b.phase_env_suffix = "WQ_B_INT8"
        self.wkv = Linear(self.dim, self.head_dim, dtype=torch.int8 if args.attn_int8 else None)
        self.wkv.phase_env_suffix = "WKV_INT8"
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(self.n_heads * self.head_dim // self.n_groups, self.n_groups * args.o_lora_rank, dtype=torch.int8 if args.attn_int8 else torch.bfloat16)
        self.wo_a_int8_enabled = args.attn_int8 or _env_enabled("DEEPSEEK_WO_A_INT8") or preload_wo_a
        self.wo_a_fp16_enabled = _env_enabled("DEEPSEEK_WO_A_FP16")
        self.wo_a_bmm_enabled = _env_enabled("DEEPSEEK_WO_A_BMM")
        self._wo_a_int8_ready = False
        self._wo_a_phase_env_enabled = preload_wo_a
        self._wo_a_cuda_ext = load_cuda_kernel() if self.wo_a_int8_enabled else None
        self._wo_a_cuda_enabled = self._wo_a_cuda_ext is not None
        self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim, dtype=torch.int8 if args.attn_int8 else None)
        self.wo_b.phase_env_suffix = "WO_B_INT8"
        if preload_wq_a:
            self.wq_a.enable_online_int8()
        if preload_wq_b:
            self.wq_b.enable_online_int8()
        if preload_wkv:
            self.wkv.enable_online_int8()
        if preload_wo_b:
            self.wo_b.enable_online_int8()
        if _env_enabled("DEEPSEEK_WQ_A_INT8"):
            self.wq_a.enable_online_int8()
        if _env_enabled("DEEPSEEK_WQ_B_INT8"):
            self.wq_b.enable_online_int8()
        if _env_enabled("DEEPSEEK_WKV_INT8"):
            self.wkv.enable_online_int8()
        if _env_enabled("DEEPSEEK_WO_B_INT8"):
            self.wo_b.enable_online_int8()
        self.softmax_scale = self.head_dim ** -0.5
        self.attn_profile_enabled = _env_enabled("DEEPSEEK_ATTN_PROFILE")
        # Plan B-小-v2: fused attention decode prefuse kernels (q rmsnorm+rope; kv norm+rope+actquant).
        self._fused_attn_prefuse_enabled = _env_enabled("DEEPSEEK_FUSED_ATTN_PREFUSE")
        self._fused_attn_prefuse_ext = load_cuda_kernel() if self._fused_attn_prefuse_enabled else None
        if self._fused_attn_prefuse_ext is not None and (
            not hasattr(self._fused_attn_prefuse_ext, "fused_q_rmsnorm_rope_inplace")
            or not hasattr(self._fused_attn_prefuse_ext, "fused_kv_rope_actquant_inplace")
        ):
            # Built extension does not contain the new ops; disable rather than crash.
            self._fused_attn_prefuse_enabled = False
            self._fused_attn_prefuse_ext = None
        # Plan B-小-v3 gate: fold the per-step kv_cache write into the fused_kv
        # kernel. The kernel writes the produced row directly into
        # kv_cache[:, start_pos % win, :] in addition to the inplace kv buffer,
        # so the Python-side `self.kv_cache[:bsz, slot] = kv.squeeze(1)` and its
        # 80+ select/copy_/as_strided dispatcher ops per layer per step are
        # skipped. Default OFF: A/B over 7 long_long runs (governor pinned)
        # gave baseline {2.049, 2.054, 2.014} vs fold {2.036, 2.103, 1.797,
        # 2.061} -- mean delta -2.0% (1.999 vs 2.039), inside +/-10% jitter.
        # Same dispatcher-bound failure pattern as IMMA / inv-rope: GPU is
        # ~92% idle, so per-op fusion does not move wallclock. Re-test once
        # CUDA Graph capture or async double-buffered MoE lands.
        self._fused_kv_cache_fold_enabled = (
            self._fused_attn_prefuse_enabled
            and os.environ.get("DEEPSEEK_FUSED_KV_CACHE_FOLD", "0") == "1"
        )

        # Separate gate for the inverse-rope back-end fuse so we can A/B it
        # without disabling the prefuse front-end. Default OFF: kernel is
        # bit-exact but A/B over 6 long_long runs (governor pinned) showed mean
        # delta +1.7% (baseline {1.761, 2.074, 2.002} vs inv-rope {2.089, 2.007,
        # 1.841}), well inside the ±10% per-run jitter. Same pattern as IMMA:
        # GPU is ~92% idle so saving 5 dispatches/layer does not move wallclock.
        # Re-test once CUDA Graph capture lands.
        self._fused_attn_inv_rope_enabled = (
            self._fused_attn_prefuse_enabled
            and os.environ.get("DEEPSEEK_FUSED_ATTN_INV_ROPE", "0") == "1"
        )

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)
            else:
                self.indexer = None

        kv_cache_size = args.window_size + (args.max_seq_len // self.compress_ratio if self.compress_ratio else 0)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, kv_cache_size, self.head_dim), persistent=False)
        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            # disable YaRN and use base rope_theta in pure sliding-window attention
            original_seq_len, rope_theta = 0, args.rope_theta
        freqs_cis = precompute_freqs_cis(self.rope_head_dim, args.max_seq_len, original_seq_len,
                                         rope_theta, args.rope_factor, args.beta_fast, args.beta_slow)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def set_preloaded_wo_a_int8(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        self.wo_a_int8_enabled = True
        self.register_buffer(
            "wo_a_int8_weight",
            weight.detach().to(device=self.wo_a.weight.device, dtype=torch.int8).view(self.n_local_groups, self.o_lora_rank, -1),
            persistent=False,
        )
        self.register_buffer(
            "wo_a_int8_scale",
            scale.detach().to(device=self.wo_a.weight.device, dtype=torch.float32).view(self.n_local_groups, self.o_lora_rank),
            persistent=False,
        )
        self._wo_a_int8_ready = True

    def _wo_a_int8_active(self) -> bool:
        return _phase_env_enabled("WO_A_INT8", self.wo_a_int8_enabled or self._wo_a_phase_env_enabled)

    def _get_prefuse_freqs(self, start_pos: int, seqlen: int):
        # Returns (real, imag) fp32 contiguous tensors of shape [S, rd/2] on the freqs_cis device.
        # Builds two persistent fp32 buffers once (full sequence length), then narrows per call so
        # decode steps only pay one .narrow() each (no copy, no dtype cast).
        full_r = getattr(self, "_freqs_real_full", None)
        full_i = getattr(self, "_freqs_imag_full", None)
        if full_r is None or full_r.size(0) != self.freqs_cis.size(0) or full_r.device != self.freqs_cis.device:
            full_r = self.freqs_cis.real.to(torch.float32).contiguous()
            full_i = self.freqs_cis.imag.to(torch.float32).contiguous()
            self._freqs_real_full = full_r
            self._freqs_imag_full = full_i
        return (
            full_r.narrow(0, start_pos, seqlen),
            full_i.narrow(0, start_pos, seqlen),
        )

    def _wo_a_fp16_weight(self) -> torch.Tensor:
        weight = getattr(self, "wo_a_fp16_weight", None)
        if weight is None or weight.device != self.wo_a.weight.device:
            weight = self.wo_a.weight.detach().to(dtype=torch.float16).view(self.n_local_groups, self.o_lora_rank, -1)
            self.register_buffer("wo_a_fp16_weight", weight, persistent=False)
        return weight

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        profile = self.attn_profile_enabled
        def mark():
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter() if profile else 0.0
        t0 = mark()
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis
        # Decide once per call whether the prefuse kernels apply (decode bf16 path only).
        use_prefuse = (
            self._fused_attn_prefuse_enabled
            and self._fused_attn_prefuse_ext is not None
            and seqlen == 1
            and x.is_cuda
            and x.dtype == torch.bfloat16
        )
        # q
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        if use_prefuse:
            fr, fi = self._get_prefuse_freqs(start_pos, seqlen)
            self._fused_attn_prefuse_ext.fused_q_rmsnorm_rope_inplace(q, fr, fi, float(self.eps))
        else:
            q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
            apply_rotary_emb(q[..., -rd:], freqs_cis)
        t_q = mark()

        # win kv & topk_idxs
        # Decide if the fused-kv kernel can also fold in the kv_cache slot
        # write (decode-only fast path: seqlen == 1 and start_pos != 0).
        kv_cache_fold = (
            use_prefuse
            and self._fused_kv_cache_fold_enabled
            and seqlen == 1
            and start_pos != 0
            and bsz <= self.kv_cache.size(0)
        )
        kv = self.wkv(x)
        if use_prefuse:
            fr, fi = self._get_prefuse_freqs(start_pos, seqlen)
            if kv_cache_fold:
                self._fused_attn_prefuse_ext.fused_kv_rope_actquant_inplace(
                    kv, self.kv_norm.weight, fr, fi, 64,
                    float(self.kv_norm.eps),
                    self.kv_cache, int(start_pos % win),
                )
            else:
                self._fused_attn_prefuse_ext.fused_kv_rope_actquant_inplace(
                    kv, self.kv_norm.weight, fr, fi, 64, float(self.kv_norm.eps)
                )
        else:
            kv = self.kv_norm(kv)
            apply_rotary_emb(kv[..., -rd:], freqs_cis)
            # FP8-simulate non-rope dims to match QAT; rope dims stay bf16 for positional precision
            act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        t_kv = mark()
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
        t_window_idx = mark()
        t_compress_idx = t_window_idx
        t_compressor = t_window_idx
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
            t_compress_idx = mark()
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # compress kv & attn
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                if (kv_compress := self.compressor(x, start_pos)) is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
                t_compressor = mark()
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            # Decode path. seqlen==1 is the standard fast path; seqlen>1 is
            # the speculative-verify path used when speculative decoding is
            # active. For seqlen>1 we write `seqlen` consecutive slots into
            # the windowed kv_cache (with wrap-around), call the per-position
            # compressor seqlen times, and let sparse_attn handle a multi-q
            # batch.
            if not kv_cache_fold:
                if seqlen == 1:
                    self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
                else:
                    for i in range(seqlen):
                        self.kv_cache[:bsz, (start_pos + i) % win] = kv[:, i]
            if self.compress_ratio:
                if seqlen == 1:
                    self.compressor(x, start_pos)
                else:
                    for i in range(seqlen):
                        self.compressor(x[:, i:i+1], start_pos + i)
                t_compressor = mark()
            o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
        t_sparse = mark()
        if use_prefuse and self._fused_attn_inv_rope_enabled and o.is_cuda and o.dtype == torch.bfloat16 and hasattr(self._fused_attn_prefuse_ext, "fused_o_inverse_rope_inplace"):
            fr, fi = self._get_prefuse_freqs(start_pos, seqlen)
            self._fused_attn_prefuse_ext.fused_o_inverse_rope_inplace(o, fr, fi)
        else:
            apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # o
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        if self._wo_a_int8_active():
            if not self._wo_a_int8_ready:
                if getattr(self.wo_a, "_online_int8_ready", False) and hasattr(self.wo_a, "online_int8_weight"):
                    self.register_buffer(
                        "wo_a_int8_weight",
                        self.wo_a.online_int8_weight.detach().view(self.n_local_groups, self.o_lora_rank, -1),
                        persistent=False,
                    )
                    self.register_buffer(
                        "wo_a_int8_scale",
                        self.wo_a.online_int8_scale.detach().view(self.n_local_groups, self.o_lora_rank),
                        persistent=False,
                    )
                else:
                    weight = self.wo_a.weight.detach().view(self.n_local_groups, self.o_lora_rank, -1)
                    if self.wo_a.weight.dtype == torch.int8:
                        scale = self.wo_a.weight.scale.detach()
                        self.register_buffer("wo_a_int8_weight", weight, persistent=False)
                        self.register_buffer("wo_a_int8_scale", scale.view(self.n_local_groups, -1), persistent=False)
                    else:
                        q_chunks = []
                        s_chunks = []
                        for g in range(self.n_local_groups):
                            weight_q, weight_s = _quantize_int8_weight_torch(weight[g])
                            q_chunks.append(weight_q)
                            s_chunks.append(weight_s)
                        self.register_buffer("wo_a_int8_weight", torch.stack(q_chunks, dim=0), persistent=False)
                        self.register_buffer("wo_a_int8_scale", torch.stack(s_chunks, dim=0), persistent=False)
                self._wo_a_int8_ready = True
            if self._wo_a_cuda_enabled and o.is_cuda:
                o = self._wo_a_cuda_ext.wo_a_int8_forward(
                    o.contiguous(),
                    self.wo_a_int8_weight.contiguous(),
                    self.wo_a_int8_scale.contiguous(),
                ).to(torch.get_default_dtype())
            else:
                proj_chunks = []
                for g in range(self.n_local_groups):
                    proj_g = soft_bf16_weight_gemm_int8(o[:, :, g, :], self.wo_a_int8_weight[g], self.wo_a_int8_scale[g])
                    proj_chunks.append(proj_g.unsqueeze(2))
                o = torch.cat(proj_chunks, dim=2)
        else:
            if self.wo_a_fp16_enabled and o.is_cuda and self.wo_a.weight.is_cuda:
                wo_a = self._wo_a_fp16_weight()
                o = torch.einsum("bsgd,grd->bsgr", o.to(torch.float16), wo_a).to(torch.get_default_dtype())
            else:
                wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
                if self.wo_a_bmm_enabled and o.is_cuda:
                    o_shape = o.shape
                    o = torch.bmm(
                        o.permute(2, 0, 1, 3).reshape(self.n_local_groups, -1, o_shape[-1]),
                        wo_a.transpose(1, 2),
                    ).view(self.n_local_groups, o_shape[0], o_shape[1], self.o_lora_rank).permute(1, 2, 0, 3)
                else:
                    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        t_wo_a = mark()
        x = self.wo_b(o.flatten(2))
        t_wo_b = mark()
        if profile:
            print(
                f"attn_detail layer={self.layer_id} pos={start_pos} batch={bsz} seqlen={seqlen} ratio={ratio} topk={topk_idxs.size(-1)} "
                f"q={t_q - t0:.6f}s kv={t_kv - t_q:.6f}s win_idx={t_window_idx - t_kv:.6f}s "
                f"cmp_idx={t_compress_idx - t_window_idx:.6f}s compressor={t_compressor - t_compress_idx:.6f}s "
                f"sparse={t_sparse - t_compressor:.6f}s wo_a={t_wo_a - t_sparse:.6f}s wo_b={t_wo_b - t_wo_a:.6f}s",
                flush=True,
            )
        return x


class Gate(nn.Module):
    """MoE gating: computes expert routing scores and selects top-k experts.
    Supports hash-based routing (first n_hash_layers) where expert indices are
    predetermined per token ID, and score-based routing (remaining layers)."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        if self.hash:
            self.tid2eid = nn.Parameter(torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int32), requires_grad=False)
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        # Bias shifts scores for expert selection (topk) but does not affect routing weights.
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """Single MoE expert: SwiGLU FFN (w1, w2, w3). Computation in float32 for stability."""
    def __init__(self, dim: int, inter_dim: int, dtype=None, swiglu_limit=0, shared_int8_enabled: bool = False):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, dtype=dtype)
        self.w2 = Linear(inter_dim, dim, dtype=dtype)
        self.w3 = Linear(dim, inter_dim, dtype=dtype)
        self.swiglu_limit = swiglu_limit
        self.preload_fp4_dequant = False
        self._cpu_predequantized = False
        self._cpu_materialized = False
        self._cpu_w1 = None
        self._cpu_w2 = None
        self._cpu_w3 = None
        self._cpu_w1_scale = None
        self._cpu_w2_scale = None
        self._cpu_w3_scale = None
        self._cpu_w2_tiled = None
        self._cpu_w2_scale_tiled = None
        self._cpu_tile_rows = FP4_CPU_TILE_ROWS
        self.shared_expert_int8_enabled = shared_int8_enabled
        self.shared_expert_fp16_enabled = _env_enabled("DEEPSEEK_SHARED_EXPERT_FP16")
        self._shared_fp16_ready = False
        self._shared_w13_fp16 = None
        self._shared_w2_fp16 = None
        self._shared_int8_ready = False
        self.shared_expert_chunk_tokens = max(0, int(os.getenv("DEEPSEEK_SHARED_EXPERT_CHUNK_TOKENS", "8192")))

    def _predequantize_fp4_weights_on_gpu(self):
        if self._cpu_predequantized or self.w1.weight.dtype != torch.float4_e2m1fn_x2:
            return
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else self.w1.weight.device
        w1 = _dequant_fp4_weight_torch(
            Packed4BitWeightAlongK.convert_from(self.w1.weight.detach().to(device)),
            self.w1.weight.scale.detach().to(device, dtype=torch.float32),
            block_size=fp4_block_size,
        ).cpu().contiguous()
        w2 = _dequant_fp4_weight_torch(
            Packed4BitWeightAlongK.convert_from(self.w2.weight.detach().to(device)),
            self.w2.weight.scale.detach().to(device, dtype=torch.float32),
            block_size=fp4_block_size,
        ).cpu().contiguous()
        w3 = _dequant_fp4_weight_torch(
            Packed4BitWeightAlongK.convert_from(self.w3.weight.detach().to(device)),
            self.w3.weight.scale.detach().to(device, dtype=torch.float32),
            block_size=fp4_block_size,
        ).cpu().contiguous()
        self._cpu_w1 = w1
        self._cpu_w2 = w2
        self._cpu_w3 = w3
        self._cpu_predequantized = True
        self._cpu_materialized = True

    def _materialize_cpu_weights(self):
        if self._cpu_materialized:
            return
        if self.preload_fp4_dequant:
            self._predequantize_fp4_weights_on_gpu()
            if self._cpu_materialized:
                return
        if self.w1.weight.dtype == torch.float4_e2m1fn_x2:
            self._cpu_w1 = Packed4BitWeightAlongK.convert_from(self.w1.weight.detach().cpu())
            self._cpu_w2 = Packed4BitWeightAlongK.convert_from(self.w2.weight.detach().cpu())
            self._cpu_w3 = Packed4BitWeightAlongK.convert_from(self.w3.weight.detach().cpu())
            self._cpu_w1_scale = self.w1.weight.scale.detach().to(device="cpu", dtype=torch.float32).clone()
            self._cpu_w2_scale = self.w2.weight.scale.detach().to(device="cpu", dtype=torch.float32).clone()
            self._cpu_w3_scale = self.w3.weight.scale.detach().to(device="cpu", dtype=torch.float32).clone()
            self._cpu_w2_tiled, self._cpu_w2_scale_tiled = _pack_fp4_weight_rows_for_tile_decode(
                self._cpu_w2,
                self._cpu_w2_scale,
                tile_rows=self._cpu_tile_rows,
                block_size=fp4_block_size,
            )
        elif self.w1.weight.dtype == torch.int8:
            self._cpu_w1 = self.w1.weight.detach().cpu().contiguous()
            self._cpu_w2 = self.w2.weight.detach().cpu().contiguous()
            self._cpu_w3 = self.w3.weight.detach().cpu().contiguous()
            self._cpu_w1_scale = self.w1.weight.scale.detach().cpu().float().contiguous()
            self._cpu_w2_scale = self.w2.weight.scale.detach().cpu().float().contiguous()
            self._cpu_w3_scale = self.w3.weight.scale.detach().cpu().float().contiguous()
        else:
            self._cpu_w1 = self.w1.weight.detach().cpu().to(torch.float32)
            self._cpu_w2 = self.w2.weight.detach().cpu().to(torch.float32)
            self._cpu_w3 = self.w3.weight.detach().cpu().to(torch.float32)
        self._cpu_materialized = True

    def _ensure_shared_fp16(self):
        if not self.shared_expert_fp16_enabled or self._shared_fp16_ready:
            return
        if self.w1.weight.dtype == torch.float8_e4m3fn:
            w1 = soft_fp8_blockfp8_weight_dequant(self.w1.weight.detach(), self.w1.weight.scale.detach()).float()
            w2 = soft_fp8_blockfp8_weight_dequant(self.w2.weight.detach(), self.w2.weight.scale.detach()).float()
            w3 = soft_fp8_blockfp8_weight_dequant(self.w3.weight.detach(), self.w3.weight.scale.detach()).float()
        elif self.w1.weight.dtype == torch.int8:
            w1 = self.w1.weight.detach().float() * self.w1.weight.scale.detach().float().unsqueeze(1)
            w2 = self.w2.weight.detach().float() * self.w2.weight.scale.detach().float().unsqueeze(1)
            w3 = self.w3.weight.detach().float() * self.w3.weight.scale.detach().float().unsqueeze(1)
        else:
            w1 = self.w1.weight.detach().float()
            w2 = self.w2.weight.detach().float()
            w3 = self.w3.weight.detach().float()
        self.register_buffer("shared_w13_fp16", torch.cat([w1, w3], dim=0).to(device=self.w1.weight.device, dtype=torch.float16).contiguous(), persistent=False)
        self.register_buffer("shared_w2_fp16", w2.to(device=self.w1.weight.device, dtype=torch.float16).contiguous(), persistent=False)
        self._shared_fp16_ready = True

    def _ensure_shared_int8(self):
        if not self.shared_expert_int8_enabled or self._shared_int8_ready:
            return
        if self.w1.weight.dtype == torch.int8:
            self.register_buffer("shared_int8_w1", self.w1.weight.detach(), persistent=False)
            self.register_buffer("shared_int8_s1", self.w1.weight.scale.detach(), persistent=False)
            self.register_buffer("shared_int8_w2", self.w2.weight.detach(), persistent=False)
            self.register_buffer("shared_int8_s2", self.w2.weight.scale.detach(), persistent=False)
            self.register_buffer("shared_int8_w3", self.w3.weight.detach(), persistent=False)
            self.register_buffer("shared_int8_s3", self.w3.weight.scale.detach(), persistent=False)
            self._shared_int8_ready = True
            return
        if self.w1.weight.dtype == torch.float8_e4m3fn:
            w1 = soft_fp8_blockfp8_weight_dequant(self.w1.weight.detach(), self.w1.weight.scale.detach()).float()
            w2 = soft_fp8_blockfp8_weight_dequant(self.w2.weight.detach(), self.w2.weight.scale.detach()).float()
            w3 = soft_fp8_blockfp8_weight_dequant(self.w3.weight.detach(), self.w3.weight.scale.detach()).float()
        else:
            w1 = self.w1.weight.detach().float()
            w2 = self.w2.weight.detach().float()
            w3 = self.w3.weight.detach().float()
        w1_q, w1_s = _quantize_int8_weight_torch(w1)
        w2_q, w2_s = _quantize_int8_weight_torch(w2)
        w3_q, w3_s = _quantize_int8_weight_torch(w3)
        self.register_buffer("shared_int8_w1", w1_q, persistent=False)
        self.register_buffer("shared_int8_s1", w1_s, persistent=False)
        self.register_buffer("shared_int8_w2", w2_q, persistent=False)
        self.register_buffer("shared_int8_s2", w2_s, persistent=False)
        self.register_buffer("shared_int8_w3", w3_q, persistent=False)
        self.register_buffer("shared_int8_s3", w3_s, persistent=False)
        self._shared_int8_ready = True

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        chunk = self.shared_expert_chunk_tokens if self.shared_expert_int8_enabled or self.shared_expert_fp16_enabled else 0
        if chunk > 0 and x.dim() == 2 and x.size(0) > chunk:
            parts = []
            for start in range(0, x.size(0), chunk):
                end = min(start + chunk, x.size(0))
                weight_part = weights[start:end] if weights is not None else None
                parts.append(self.forward(x[start:end].contiguous(), weight_part))
            return torch.cat(parts, dim=0).to(dtype)
        if self.shared_expert_fp16_enabled and x.is_cuda:
            self._ensure_shared_fp16()
            gate_up = F.linear(x.to(torch.float16), self.shared_w13_fp16).float()
            gate, up = gate_up.chunk(2, dim=-1)
        elif self.shared_expert_int8_enabled:
            self._ensure_shared_int8()
            if _SHARED_EXPERT_PAIR_INT8_CUDA and x.is_cuda:
                try:
                    gate, up = soft_bf16_weight_gemm_int8_pair_cuda_ext(
                        x,
                        self.shared_int8_w1,
                        self.shared_int8_s1,
                        self.shared_int8_w3,
                        self.shared_int8_s3,
                    )
                    gate = gate.float()
                    up = up.float()
                except Exception:
                    gate = soft_bf16_weight_gemm_int8(x, self.shared_int8_w1, self.shared_int8_s1).float()
                    up = soft_bf16_weight_gemm_int8(x, self.shared_int8_w3, self.shared_int8_s3).float()
            else:
                gate = soft_bf16_weight_gemm_int8(x, self.shared_int8_w1, self.shared_int8_s1).float()
                up = soft_bf16_weight_gemm_int8(x, self.shared_int8_w3, self.shared_int8_s3).float()
        else:
            gate = self.w1(x).float()
            up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        if self.shared_expert_fp16_enabled and x.is_cuda:
            return F.linear(x.to(torch.float16), self.shared_w2_fp16).to(dtype)
        if self.shared_expert_int8_enabled:
            return soft_bf16_weight_gemm_int8(x.to(dtype), self.shared_int8_w2, self.shared_int8_s2).to(dtype)
        return self.w2(x.to(dtype))

    def forward_cpu(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None, x_cpu: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._materialize_cpu_weights()
        if x_cpu is None:
            x_cpu = x.detach().cpu().to(torch.float32)
        if isinstance(self._cpu_w1, Packed4BitWeightAlongK):
            w1 = _dequant_fp4_weight_torch(self._cpu_w1, self._cpu_w1_scale, block_size=fp4_block_size)
            w2 = _dequant_fp4_weight_torch(self._cpu_w2, self._cpu_w2_scale, block_size=fp4_block_size)
            w3 = _dequant_fp4_weight_torch(self._cpu_w3, self._cpu_w3_scale, block_size=fp4_block_size)
        else:
            w1 = self._cpu_w1
            w2 = self._cpu_w2
            w3 = self._cpu_w3
        gate = F.linear(x_cpu, w1, None)
        up = F.linear(x_cpu, w3, None)
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        y = F.silu(gate) * up
        if weights is not None:
            y = weights.detach().cpu().to(torch.float32) * y
        y = F.linear(y, w2, None)
        return y


class MoE(nn.Module):
    """Mixture-of-Experts: gate routes each token to top-k routed experts + 1 shared expert.
    Experts are sharded across TP ranks; each rank handles n_routed_experts // world_size experts."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.routed_experts_device = args.routed_experts_device
        self.pd_phase_auto_select_enabled = _env_enabled("DEEPSEEK_PD_PHASE_AUTO_SELECT")
        self.pd_single_cpu_moe_weights_enabled = _env_enabled("DEEPSEEK_PD_SINGLE_CPU_MOE_WEIGHTS")
        self.cpu_moe_inproc_server_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_CPU_MOE_INPROC_SERVER")
        self.cpu_moe_rank0_server_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_CPU_MOE_RANK0_SERVER")
        self.cpu_moe_external_server_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_CPU_MOE_EXTERNAL_SERVER")
        self.cpu_moe_external_prefill_local_enabled = self.cpu_moe_external_server_enabled and os.getenv("DEEPSEEK_CPU_MOE_EXTERNAL_PREFILL_LOCAL", "1").lower() in {"1", "true", "yes"}
        self.cpu_moe_predict_seq_enabled = (self.cpu_moe_external_server_enabled or self.cpu_moe_inproc_server_enabled) and _env_enabled("DEEPSEEK_CPU_MOE_PREDICT_SEQ")
        if self.cpu_moe_external_server_enabled and not self.cpu_moe_external_prefill_local_enabled and not self.pd_single_cpu_moe_weights_enabled:
            self.experts_start_idx = 0
            self.experts_end_idx = 0
        elif self.cpu_moe_inproc_server_enabled:
            # The rank-0 in-process daemon thread runs the full-routes int8
            # CPU MoE kernel; it requires the full per-layer pointer table to
            # cover all routed experts. We therefore have rank 0 own all
            # experts (OOM caveat: increases per-rank GPU prefill MoE staging
            # cache; lower DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS or
            # disable DEEPSEEK_GPU_PREFILL_MOE for long prefill if needed).
            # Ranks 1-3 own no experts and only contribute the shared expert
            # plus the post-MoE allreduce skip.
            if rank == 0:
                self.experts_start_idx = 0
                self.experts_end_idx = self.n_routed_experts
            else:
                self.experts_start_idx = 0
                self.experts_end_idx = 0
        elif self.cpu_moe_rank0_server_enabled and rank == 0:
            self.experts_start_idx = 0
            self.experts_end_idx = self.n_routed_experts
        else:
            self.experts_start_idx = rank * self.n_local_experts
            self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(layer_id, args)
        expert_dtype = None
        if args.expert_dtype == "fp4":
            expert_dtype = torch.float4_e2m1fn_x2
        elif args.expert_dtype == "int8":
            expert_dtype = torch.int8
        if self.routed_experts_device == "cpu":
            if self.cpu_moe_external_server_enabled and not self.cpu_moe_external_prefill_local_enabled:
                self.experts = [None for _ in range(self.n_routed_experts)]
                self.cpu_backend = None
            else:
                with torch.device("cpu"):
                    self.experts = nn.ModuleList([
                        Expert(args.dim, args.moe_inter_dim, dtype=expert_dtype, swiglu_limit=args.swiglu_limit, shared_int8_enabled=False)
                        if self.experts_start_idx <= i < self.experts_end_idx else None
                        for i in range(self.n_routed_experts)
                    ])
                    for expert in self.experts:
                        if expert is not None:
                            expert.preload_fp4_dequant = args.preload_routed_fp4_dequant
                self.cpu_backend = CPURoutedExpertsBackend(
                    layer_idx=self.layer_id,
                    experts=self.experts,
                    experts_start_idx=self.experts_start_idx,
                    experts_end_idx=self.experts_end_idx,
                    num_experts_per_tok=self.n_activated_experts,
                    output_dim=self.dim,
                )
        else:
            self.experts = nn.ModuleList([
                Expert(args.dim, args.moe_inter_dim, dtype=expert_dtype, swiglu_limit=args.swiglu_limit)
                if self.experts_start_idx <= i < self.experts_end_idx else None
                for i in range(self.n_routed_experts)
            ])
            self.cpu_backend = None
        assert args.n_shared_experts == 1
        self.shared_experts = Expert(
            args.dim,
            args.moe_inter_dim,
            dtype=torch.int8 if args.shared_expert_int8 else None,
            shared_int8_enabled=args.shared_expert_int8 or _env_enabled("DEEPSEEK_SHARED_EXPERT_INT8"),
        )
        if args.preloaded_shared_expert_int8 or args.preload_shared_w1_int8:
            self.shared_experts.w1.enable_online_int8()
        if args.preloaded_shared_expert_int8 or args.preload_shared_w2_int8:
            self.shared_experts.w2.enable_online_int8()
        if args.preloaded_shared_expert_int8 or args.preload_shared_w3_int8:
            self.shared_experts.w3.enable_online_int8()
        self.async_allreduce_enabled = os.getenv("DEEPSEEK_MOE_ASYNC_ALLREDUCE", "0").lower() in {"1", "true", "yes"}
        self.cpu_host_reduce_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_CPU_MOE_HOST_REDUCE")
        self.reduce_fp16_enabled = _env_enabled("DEEPSEEK_MOE_REDUCE_FP16")
        self._profile_enabled = os.getenv("DEEPSEEK_MOE_PROFILE", "0").lower() in {"1", "true", "yes"}
        self._rank_route_profile_enabled = _env_enabled("DEEPSEEK_RANK_ROUTE_PROFILE")
        self._allreduce_stream = torch.cuda.Stream() if self.async_allreduce_enabled and torch.cuda.is_available() else None
        self.gpu_prefill_moe_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_GPU_PREFILL_MOE")
        self.gpu_decode_active_moe_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_GPU_MOE_DECODE_ACTIVE")
        self.gpu_spec_token2_moe_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_GPU_MOE_SPEC_TOKEN2")
        self.gpu_prefill_moe_min_tokens = int(os.getenv("DEEPSEEK_GPU_PREFILL_MOE_MIN_TOKENS", "64"))
        self.gpu_prefill_moe_chunk_tokens = max(0, int(os.getenv("DEEPSEEK_GPU_PREFILL_MOE_CHUNK_TOKENS", "0")))
        self.moe_reduce_cast_chunk_tokens = max(0, int(os.getenv("DEEPSEEK_MOE_REDUCE_CAST_CHUNK_TOKENS", "8192")))
        self.gpu_prefill_backend = None

    def _finalize_reduced_moe(self, y_reduce: torch.Tensor, shared: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        chunk = self.moe_reduce_cast_chunk_tokens
        if chunk > 0 and y_reduce.dim() == 2 and y_reduce.size(0) > chunk:
            out = torch.empty((y_reduce.size(0), y_reduce.size(1)), device=y_reduce.device, dtype=out_dtype)
            for start in range(0, y_reduce.size(0), chunk):
                end = min(start + chunk, y_reduce.size(0))
                part = y_reduce[start:end].to(torch.float32)
                part.add_(shared[start:end])
                out[start:end].copy_(part)
                del part
            return out
        y = y_reduce.to(torch.float32)
        y += shared
        return y.to(out_dtype)

    def _pd_active_phase(self) -> Optional[str]:
        if not self.pd_phase_auto_select_enabled:
            return None
        phase = os.getenv("DEEPSEEK_PD_ACTIVE_PHASE", "")
        return phase if phase in {"prefill", "decode"} else None

    def _should_use_gpu_prefill_moe(self, x: torch.Tensor) -> bool:
        phase = self._pd_active_phase()
        return (
            phase != "decode"
            and self.gpu_prefill_moe_enabled
            and self.cpu_backend is not None
            and self.experts_end_idx > self.experts_start_idx
            and x.is_cuda
            and x.shape[0] >= self.gpu_prefill_moe_min_tokens
        )

    def _should_use_gpu_decode_active_moe(self, x: torch.Tensor) -> bool:
        return (
            self._pd_active_phase() == "decode"
            and self.gpu_decode_active_moe_enabled
            and self.cpu_backend is not None
            and self.experts_end_idx > self.experts_start_idx
            and x.is_cuda
        )

    def _should_use_gpu_spec_token2_moe(self, x: torch.Tensor) -> bool:
        return (
            self._pd_active_phase() == "decode"
            and self.gpu_spec_token2_moe_enabled
            and self.cpu_backend is not None
            and self.experts_end_idx > self.experts_start_idx
            and x.is_cuda
            and x.shape[0] == 2
        )

    def _ensure_gpu_prefill_backend(self) -> GPUPrefillMoEBackend:
        if self.gpu_prefill_backend is None:
            self.gpu_prefill_backend = GPUPrefillMoEBackend(
                self.cpu_backend,
                dim=self.dim,
                num_experts=self.n_routed_experts,
                experts_start_idx=self.experts_start_idx,
                experts_end_idx=self.experts_end_idx,
            )
        return self.gpu_prefill_backend


    def prefetch_gpu_prefill_moe(self, device: torch.device, token_count: int) -> None:
        if self.gpu_decode_active_moe_enabled and self._pd_active_phase() == "decode":
            return
        if (
            self.gpu_prefill_moe_enabled
            and self.cpu_backend is not None
            and self.experts_end_idx > self.experts_start_idx
            and device.type == "cuda"
            and token_count >= self.gpu_prefill_moe_min_tokens
        ):
            self._ensure_gpu_prefill_backend().prefetch(device)

    def _should_use_in_process_cpu_server(self, x: torch.Tensor) -> bool:
        if not self.cpu_moe_inproc_server_enabled or world_size <= 1 or x.shape[0] != 1:
            return False
        return self._pd_active_phase() != "prefill"

    def _should_use_rank0_cpu_server(self, x: torch.Tensor) -> bool:
        return self.cpu_moe_rank0_server_enabled and world_size > 1 and x.shape[0] == 1


    def _should_use_external_cpu_server(self, x: torch.Tensor) -> bool:
        if not self.cpu_moe_external_server_enabled or world_size <= 1 or x.shape[0] != 1:
            return False
        return self._pd_active_phase() != "prefill"

    def _external_cpu_server_submit_decode(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> int:
        global _cpu_moe_server_last_seq, _cpu_moe_server_predict_seq
        if rank != 0:
            if self.cpu_moe_predict_seq_enabled:
                _cpu_moe_server_predict_seq += 1
                return _cpu_moe_server_predict_seq
            return -1
        ipc = _get_cpu_moe_server_ipc(self.dim, self.n_activated_experts)
        req, _resp, _layer, _stop = ipc.read_header()
        seq_to_reuse = req + 1 - ipc.output_slots
        if seq_to_reuse > 0:
            ipc.wait_slot_acks(seq_to_reuse, world_size)
        seq = ipc.submit(self.layer_id, x, indices, weights)
        _cpu_moe_server_last_seq = seq
        _cpu_moe_server_predict_seq = seq
        return seq

    def _external_cpu_server_wait_decode(self, seq: int, x: torch.Tensor) -> torch.Tensor:
        ipc = _get_cpu_moe_server_ipc(self.dim, self.n_activated_experts)
        if self.cpu_moe_predict_seq_enabled:
            if rank == 0:
                global _cpu_moe_server_predict_seq
                _cpu_moe_server_predict_seq = seq
        else:
            seq_tensor = torch.empty(1, device=x.device, dtype=torch.long)
            if rank == 0:
                seq_tensor.fill_(seq)
            dist.broadcast(seq_tensor, src=0)
            seq = int(seq_tensor.item())
        ipc.wait_response(seq)
        y = ipc.output_tensor(seq).to(device=x.device, dtype=torch.float32, non_blocking=False)
        if rank != 0:
            ipc.ack(rank, seq)
        return y


    def _external_cpu_server_sync(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        global _cpu_moe_server_last_seq, _cpu_moe_server_predict_seq
        profile = _env_enabled("DEEPSEEK_CPU_MOE_SERVER_PROFILE")
        ipc = _get_cpu_moe_server_ipc(self.dim, self.n_activated_experts)
        outputs = []
        seq_tensor = torch.empty(1, device=x.device, dtype=torch.long)
        t_total = time.perf_counter() if profile else 0.0
        for row in range(x.shape[0]):
            if rank == 0:
                req, _resp, _layer, _stop = ipc.read_header()
                seq_to_reuse = req + 1 - ipc.output_slots
                if seq_to_reuse > 0:
                    ipc.wait_slot_acks(seq_to_reuse, world_size)
                seq = ipc.submit(self.layer_id, x[row:row + 1], indices[row:row + 1], weights[row:row + 1])
                _cpu_moe_server_last_seq = seq
                _cpu_moe_server_predict_seq = seq
                seq_tensor.fill_(seq)
            dist.broadcast(seq_tensor, src=0)
            seq = int(seq_tensor.item())
            if self.cpu_moe_predict_seq_enabled and rank != 0:
                _cpu_moe_server_predict_seq = seq
            ipc.wait_response(seq)
            y_row = ipc.output_tensor(seq).to(device=x.device, dtype=torch.float32, non_blocking=False)
            if rank != 0:
                ipc.ack(rank, seq)
            outputs.append(y_row)
        y = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
        if profile:
            print(
                f"cpu_moe_external_profile layer={self.layer_id} rank={rank} batch={x.shape[0]} total={time.perf_counter() - t_total:.6f}s",
                flush=True,
            )
        return y

    def _rank0_cpu_server_sync(self, x: torch.Tensor) -> torch.Tensor:
        if rank == 0:
            y = self.cpu_backend.sync_forward(x).to(torch.float32)
        else:
            y = torch.empty_like(x, dtype=torch.float32)
        dist.broadcast(y, src=0)
        return y

    def _should_use_cpu_host_reduce(self, x: torch.Tensor) -> bool:
        phase = self._pd_active_phase()
        return phase != "prefill" and self.cpu_host_reduce_enabled and world_size > 1 and x.shape[0] == 1

    def _cpu_host_reduce(self, x: torch.Tensor) -> tuple[torch.Tensor, float, float, float]:
        t0 = time.perf_counter() if self._profile_enabled else 0.0
        y_cpu, current_slot, batch_size, device = self.cpu_backend.sync_forward_cpu(x)
        t_sync = time.perf_counter() if self._profile_enabled else 0.0
        if not hasattr(self.cpu_backend, "host_float_allreduce"):
            raise RuntimeError("DEEPSEEK_CPU_MOE_HOST_REDUCE requires native host_float_allreduce")
        self.cpu_backend.host_float_allreduce(y_cpu, rank, world_size)
        t_reduce = time.perf_counter() if self._profile_enabled else 0.0
        y = self.cpu_backend.copy_cpu_output_to_device(y_cpu, current_slot, batch_size, device, x).to(torch.float32)
        t_h2d = time.perf_counter() if self._profile_enabled else 0.0
        return y, t_sync - t0, t_reduce - t_sync, t_h2d - t_reduce

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        if self._rank_route_profile_enabled and x.shape[0] == 1:
            local = ((indices >= self.experts_start_idx) & (indices < self.experts_end_idx)).sum().item()
            print(f"rank_route layer={self.layer_id} rank={rank} local={int(local)} ids={indices.flatten().tolist()}", flush=True)
        used_gpu_prefill_moe = False
        reduced_moe_ready = False
        used_remote_only = False
        if self.routed_experts_device == "cpu" and self._should_use_gpu_prefill_moe(x):
            shared = self.shared_experts(x)
            backend = self._ensure_gpu_prefill_backend()
            chunk = self.gpu_prefill_moe_chunk_tokens
            if chunk > 0 and x.shape[0] > chunk:
                y = torch.empty((x.shape[0], self.dim), device=x.device, dtype=torch.float32)
                for start in range(0, x.shape[0], chunk):
                    end = min(start + chunk, x.shape[0])
                    y_part = backend.forward(
                        x[start:end].contiguous(),
                        indices[start:end].contiguous(),
                        weights[start:end].contiguous(),
                        self.cpu_backend._swiglu_limit,
                    )
                    y[start:end].copy_(y_part)
                    del y_part
            else:
                y = backend.forward(
                    x,
                    indices,
                    weights,
                    self.cpu_backend._swiglu_limit,
                )
            used_gpu_prefill_moe = True
        elif self.routed_experts_device == "cpu" and self._should_use_gpu_spec_token2_moe(x):
            # Speculative verify mixed path: keep the first token on the normal
            # CPU MoE path, and run only the draft/second token's routed experts
            # on GPU with route-aware active-expert staging. This is the only
            # form that can help on the heterogeneous CPU-MoE architecture: it
            # avoids doubling CPU MoE work while using otherwise-idle GPU time.
            self.cpu_backend.submit_forward(x[:1], indices[:1], weights[:1])
            backend = self._ensure_gpu_prefill_backend()
            backend.prefetch_active_experts(x.device, indices[1:2])
            shared = self.shared_experts(x)
            y2 = backend.forward(
                x[1:2],
                indices[1:2],
                weights[1:2],
                self.cpu_backend._swiglu_limit,
            )
            y1 = self.cpu_backend.sync_forward(x[:1])
            y = torch.empty((2, self.dim), device=x.device, dtype=torch.float32)
            y[:1] = y1
            y[1:2] = y2
            used_gpu_prefill_moe = True
        elif self.routed_experts_device == "cpu" and self._should_use_gpu_decode_active_moe(x):
            backend = self._ensure_gpu_prefill_backend()
            backend.prefetch_active_experts(x.device, indices)
            if hasattr(backend, "prefetch_active_experts_to_cache"):
                backend.prefetch_active_experts_to_cache(x.device, indices)
            shared = self.shared_experts(x)
            y = backend.forward(
                x,
                indices,
                weights,
                self.cpu_backend._swiglu_limit,
            )
            used_gpu_prefill_moe = True
        elif self.routed_experts_device == "cpu" and self._should_use_in_process_cpu_server(x):
            profile = self._profile_enabled
            t0 = time.perf_counter() if profile else 0.0
            seq = self._external_cpu_server_submit_decode(x, indices, weights)
            t_submit = time.perf_counter() if profile else 0.0
            shared = self.shared_experts(x)
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_shared = time.perf_counter() if profile else 0.0
            y = self._external_cpu_server_wait_decode(seq, x)
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_wait = time.perf_counter() if profile else 0.0
            if profile:
                print(
                    f"moe_inproc_overlap_profile layer={self.layer_id} rank={rank} "
                    f"submit={t_submit - t0:.6f}s shared={t_shared - t_submit:.6f}s wait_h2d={t_wait - t_shared:.6f}s total={t_wait - t0:.6f}s",
                    flush=True,
                )
        elif self.routed_experts_device == "cpu" and self._should_use_external_cpu_server(x):
            if x.shape[0] == 1:
                profile = self._profile_enabled
                t0 = time.perf_counter() if profile else 0.0
                seq = self._external_cpu_server_submit_decode(x, indices, weights)
                t_submit = time.perf_counter() if profile else 0.0
                shared = self.shared_experts(x)
                if profile and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_shared = time.perf_counter() if profile else 0.0
                y = self._external_cpu_server_wait_decode(seq, x)
                if profile and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_wait = time.perf_counter() if profile else 0.0
                if profile:
                    print(
                        f"moe_external_overlap_profile layer={self.layer_id} rank={rank} "
                        f"submit={t_submit - t0:.6f}s shared={t_shared - t_submit:.6f}s wait_h2d={t_wait - t_shared:.6f}s total={t_wait - t0:.6f}s",
                        flush=True,
                    )
            else:
                shared = self.shared_experts(x)
                y = self._external_cpu_server_sync(x, indices, weights)
        elif self.routed_experts_device == "cpu" and self._should_use_rank0_cpu_server(x):
            if rank == 0:
                self.cpu_backend.submit_forward(x, indices, weights)
            shared = self.shared_experts(x)
            y = self._rank0_cpu_server_sync(x)
        elif self.routed_experts_device == "cpu" and self.cpu_moe_inproc_server_enabled and rank != 0:
            y = torch.zeros_like(x, dtype=torch.float32)
            shared = self.shared_experts(x)
            used_remote_only = True
        elif self.routed_experts_device == "cpu" and self.async_allreduce_enabled and world_size > 1 and self._allreduce_stream is not None:
            profile = self._profile_enabled
            t0 = time.perf_counter() if profile else 0.0
            self.cpu_backend.submit_forward(x, indices, weights)
            t_submit = time.perf_counter() if profile else 0.0
            shared = self.shared_experts(x)
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_shared = time.perf_counter() if profile else 0.0
            with torch.cuda.stream(self._allreduce_stream):
                if self._should_use_cpu_host_reduce(x):
                    y, host_sync_time, host_reduce_time, host_h2d_time = self._cpu_host_reduce(x)
                else:
                    y = self.cpu_backend.sync_forward(x)
                    if profile and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_sync = time.perf_counter() if profile else 0.0
                    y_reduce = y.to(torch.float16 if self.reduce_fp16_enabled else x.dtype)
                    chunked_finalize = (
                        self.moe_reduce_cast_chunk_tokens > 0
                        and y_reduce.dim() == 2
                        and y_reduce.size(0) > self.moe_reduce_cast_chunk_tokens
                    )
                    if chunked_finalize:
                        del y
                    dist.all_reduce(y_reduce, async_op=False)
                    if profile and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_reduce = time.perf_counter() if profile else 0.0
                    if chunked_finalize:
                        y = y_reduce
                        reduced_moe_ready = True
                    else:
                        y = y_reduce.to(torch.float32)
            torch.cuda.current_stream(y.device).wait_stream(self._allreduce_stream)
            if 'host_sync_time' in locals():
                if profile and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_sync = time.perf_counter() if profile else 0.0
                t_reduce = t_sync
                if profile:
                    print(
                        f"moe_host_reduce_profile layer={self.layer_id} batch={x.shape[0]} cpu_sync={host_sync_time:.4f}s host_reduce={host_reduce_time:.4f}s h2d={host_h2d_time:.4f}s",
                        flush=True,
                    )
            if profile:
                print(
                    f"moe_profile layer={self.layer_id} batch={x.shape[0]} submit={t_submit - t0:.4f}s shared={t_shared - t_submit:.4f}s sync={t_sync - t_shared:.4f}s reduce={t_reduce - t_sync:.4f}s",
                    flush=True,
                )
        elif self.routed_experts_device == "cpu":
            self.cpu_backend.submit_forward(x, indices, weights)
            shared = self.shared_experts(x)
            if self._should_use_cpu_host_reduce(x):
                y, _host_sync_time, _host_reduce_time, _host_h2d_time = self._cpu_host_reduce(x)
            else:
                y = self.cpu_backend.sync_forward(x)
        else:
            y = torch.zeros_like(x, dtype=torch.float32)
            counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
            for i in range(self.experts_start_idx, self.experts_end_idx):
                if counts[i] == 0:
                    continue
                expert = self.experts[i]
                idx, top = torch.where(indices == i)
                y[idx] += expert(x[idx], weights[idx, top, None])
            shared = self.shared_experts(x)
        reduced_moe = False
        if not reduced_moe_ready and world_size > 1 and not (
            self.routed_experts_device == "cpu"
            and (
                (self.async_allreduce_enabled and not used_gpu_prefill_moe and not used_remote_only)
                or self._should_use_cpu_host_reduce(x)
                or self._should_use_in_process_cpu_server(x)
                or self._should_use_external_cpu_server(x)
                or self._should_use_rank0_cpu_server(x)
            )
        ):
            reduce_dtype = torch.float16 if self.reduce_fp16_enabled else x.dtype
            y_reduce = y.to(reduce_dtype)
            chunked_finalize = (
                self.moe_reduce_cast_chunk_tokens > 0
                and y_reduce.dim() == 2
                and y_reduce.size(0) > self.moe_reduce_cast_chunk_tokens
            )
            if chunked_finalize:
                del y
            dist.all_reduce(y_reduce)
            if chunked_finalize:
                y = self._finalize_reduced_moe(y_reduce, shared, x.dtype)
                reduced_moe = True
            else:
                y = y_reduce.to(torch.float32)
        if reduced_moe_ready and not reduced_moe:
            y = self._finalize_reduced_moe(y, shared, x.dtype)
            reduced_moe = True
        if not reduced_moe:
            y += shared
            y = y.type_as(x)
        return y.view(shape)


class Block(nn.Module):
    """Transformer block with Hyper-Connections (HC) mixing.
    Instead of a simple residual, HC maintains `hc_mult` copies of the hidden state.
    hc_pre: reduces hc copies -> 1 via learned weighted sum (pre-weights from Sinkhorn).
    hc_post: expands 1 -> hc copies via learned post-weights + combination matrix."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.empty(3))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3))
        self.hc_int8_enabled = _env_enabled("DEEPSEEK_HC_INT8")
        self.hc_pre_cuda_enabled = _env_enabled("DEEPSEEK_HC_PRE_CUDA")
        self.hc_post_cuda_enabled = _env_enabled("DEEPSEEK_HC_POST_CUDA")
        self.hc_fp16_mode = os.getenv("DEEPSEEK_HC_FP16", "0").lower()
        self._hc_int8_ready = False
        self._hc_cuda_ext = load_cuda_kernel() if self.hc_int8_enabled or self.hc_pre_cuda_enabled or self.hc_post_cuda_enabled else None
        self._hc_cuda_enabled = self._hc_cuda_ext is not None
        self._hc_pre_cuda_available = self._hc_cuda_enabled and hasattr(self._hc_cuda_ext, "hc_split_pre_forward")
        self._hc_post_cuda_available = self._hc_cuda_enabled and hasattr(self._hc_cuda_ext, "hc_post_forward")
        self.layer_profile_enabled = _env_enabled("DEEPSEEK_LAYER_PROFILE")
        self.prefetch_moe_before_ffn = _env_enabled("DEEPSEEK_GPU_PREFILL_MOE_PREFETCH_BEFORE_FFN")
        self.hc_pre_chunk_tokens = max(0, int(os.getenv("DEEPSEEK_HC_PRE_CHUNK_TOKENS", "16384")))

    def prefetch_gpu_prefill_moe(self, device: torch.device, token_count: int) -> None:
        self.ffn.prefetch_gpu_prefill_moe(device, token_count)

    def release_gpu_prefill_moe_cache(self) -> None:
        backend = getattr(self.ffn, "gpu_prefill_backend", None)
        if backend is not None:
            backend.release_cache()

    def _ensure_hc_int8(self):
        if self._hc_int8_ready:
            return
        attn_q, attn_s = _quantize_int8_weight_torch(self.hc_attn_fn.detach())
        ffn_q, ffn_s = _quantize_int8_weight_torch(self.hc_ffn_fn.detach())
        self.register_buffer("hc_attn_int8_weight", attn_q.unsqueeze(0), persistent=False)
        self.register_buffer("hc_attn_int8_scale", attn_s.unsqueeze(0), persistent=False)
        self.register_buffer("hc_ffn_int8_weight", ffn_q.unsqueeze(0), persistent=False)
        self.register_buffer("hc_ffn_int8_scale", ffn_s.unsqueeze(0), persistent=False)
        self._hc_int8_ready = True

    def _hc_linear_int8(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        self._ensure_hc_int8()
        if kind == "attn":
            weight_q = self.hc_attn_int8_weight
            weight_s = self.hc_attn_int8_scale
        else:
            weight_q = self.hc_ffn_int8_weight
            weight_s = self.hc_ffn_int8_scale
        if self._hc_cuda_enabled and x.is_cuda:
            y = self._hc_cuda_ext.wo_a_int8_forward(
                x.unsqueeze(2).contiguous(),
                weight_q.contiguous(),
                weight_s.contiguous(),
            ).squeeze(2)
            return y.to(torch.float32)
        return soft_bf16_weight_gemm_int8(x, weight_q[0], weight_s[0]).to(torch.float32)

    def _hc_linear_fp16(self, x: torch.Tensor, hc_fn: torch.Tensor) -> torch.Tensor:
        weight_name = "hc_attn_fp16_weight" if hc_fn is self.hc_attn_fn else "hc_ffn_fp16_weight"
        weight = getattr(self, weight_name, None)
        if weight is None or weight.device != hc_fn.device:
            weight = hc_fn.detach().to(dtype=torch.float16)
            self.register_buffer(weight_name, weight, persistent=False)
        return F.linear(x.to(torch.float16), weight).to(torch.float32)

    def _hc_should_use_fp16(self, kind: str, x: torch.Tensor, hc_fn: torch.Tensor) -> bool:
        if not x.is_cuda or not hc_fn.is_cuda:
            return False
        return self.hc_fp16_mode in {"1", "true", "yes", "all", kind}

    def _hc_pre_impl(self, x: torch.Tensor, shape: torch.Size, dtype: torch.dtype, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        kind = "attn" if hc_fn is self.hc_attn_fn else "ffn"
        if self.hc_int8_enabled:
            mixes = self._hc_linear_int8(x, kind) * rsqrt
        elif self._hc_should_use_fp16(kind, x, hc_fn):
            mixes = self._hc_linear_fp16(x, hc_fn) * rsqrt
        else:
            mixes = F.linear(x, hc_fn) * rsqrt
        if self.hc_pre_cuda_enabled and self._hc_pre_cuda_available and x.is_cuda and self.hc_mult == 4:
            y, _pre, post, comb = self._hc_cuda_ext.hc_split_pre_forward(
                mixes.contiguous(),
                x.view(shape).contiguous(),
                hc_scale.contiguous(),
                hc_base.contiguous(),
                self.hc_mult,
                self.hc_sinkhorn_iters,
                self.hc_eps,
            )
            return y.to(dtype), post, comb
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        chunk = self.hc_pre_chunk_tokens
        if chunk > 0 and x.size(1) > chunk:
            y = torch.empty((shape[0], shape[1], shape[3]), device=x.device, dtype=dtype)
            post = torch.empty((shape[0], shape[1], self.hc_mult), device=x.device, dtype=torch.float32)
            comb = torch.empty((shape[0], shape[1], self.hc_mult, self.hc_mult), device=x.device, dtype=torch.float32)
            for start in range(0, x.size(1), chunk):
                end = min(start + chunk, x.size(1))
                part = x[:, start:end].flatten(2).float().contiguous()
                part_shape = torch.Size((shape[0], end - start, shape[2], shape[3]))
                y_part, post_part, comb_part = self._hc_pre_impl(
                    part,
                    part_shape,
                    dtype,
                    hc_fn,
                    hc_scale,
                    hc_base,
                )
                y[:, start:end].copy_(y_part)
                post[:, start:end].copy_(post_part)
                comb[:, start:end].copy_(comb_part)
                del part, y_part, post_part, comb_part
            return y, post, comb
        return self._hc_pre_impl(x.flatten(2).float(), shape, dtype, hc_fn, hc_scale, hc_base)

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        if (
            self.hc_post_cuda_enabled
            and self._hc_post_cuda_available
            and x.is_cuda
            and residual.dim() == 4
            and residual.size(2) == 4
            and residual.size(0) == x.size(0)
            and residual.size(1) == x.size(1)
            and residual.size(3) == x.size(2)
        ):
            return self._hc_cuda_ext.hc_post_forward(
                x.contiguous(),
                residual.contiguous(),
                post.contiguous(),
                comb.contiguous(),
            )
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def _profile_sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor], prefetch_next: Optional[nn.Module] = None) -> torch.Tensor:
        profile = self.layer_profile_enabled
        t0 = self._profile_sync() if profile else 0.0
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        t_hc_attn_pre = self._profile_sync() if profile else 0.0
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)
        t_attn = self._profile_sync() if profile else 0.0
        x = self.hc_post(x, residual, post, comb)
        t_hc_attn_post = self._profile_sync() if profile else 0.0

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        t_hc_ffn_pre = self._profile_sync() if profile else 0.0
        x = self.ffn_norm(x)
        if self.prefetch_moe_before_ffn and prefetch_next is not None and input_ids is not None:
            prefetch_next.prefetch_gpu_prefill_moe(x.device, input_ids.numel())
        x = self.ffn(x, input_ids)
        if not self.prefetch_moe_before_ffn and prefetch_next is not None and input_ids is not None:
            prefetch_next.prefetch_gpu_prefill_moe(x.device, input_ids.numel())
        t_moe = self._profile_sync() if profile else 0.0
        x = self.hc_post(x, residual, post, comb)
        t_hc_ffn_post = self._profile_sync() if profile else 0.0
        if profile:
            print(
                f"layer_profile layer={self.layer_id} pos={start_pos} batch={x.shape[0]} "
                f"hc_attn_pre={t_hc_attn_pre - t0:.4f}s "
                f"attn={t_attn - t_hc_attn_pre:.4f}s "
                f"hc_attn_post={t_hc_attn_post - t_attn:.4f}s "
                f"hc_ffn_pre={t_hc_ffn_pre - t_hc_attn_post:.4f}s "
                f"moe={t_moe - t_hc_ffn_pre:.4f}s "
                f"hc_ffn_post={t_hc_ffn_post - t_moe:.4f}s",
                flush=True,
            )
        return x


class ParallelHead(nn.Module):

    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.part_vocab_size = (vocab_size // world_size)
        self.hc_fp16_enabled = _env_enabled("DEEPSEEK_HEAD_HC_FP16")
        self.lm_head_fp16_enabled = _env_enabled("DEEPSEEK_LM_HEAD_FP16")
        # lm_head in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for easier computation of logits later.
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim, dtype=torch.float32))

    def _lm_head_fp16_weight(self) -> torch.Tensor:
        weight = getattr(self, "lm_head_fp16_weight", None)
        if weight is None or weight.device != self.weight.device:
            weight = self.weight.detach().to(dtype=torch.float16)
            self.register_buffer("lm_head_fp16_weight", weight, persistent=False)
        return weight

    def get_logits(self, x, keep_all_positions: bool = False):
        if not keep_all_positions:
            x = x[:, -1]
        if self.lm_head_fp16_enabled and x.is_cuda and self.weight.is_cuda:
            return F.linear(x.to(torch.float16), self._lm_head_fp16_weight()).to(torch.float32)
        return F.linear(x.float(), self.weight)

    def forward(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, norm: RMSNorm, keep_all_positions: bool = False):
        if not keep_all_positions:
            x = x[:, -1:]
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        logits = self.get_logits(norm(x), keep_all_positions=keep_all_positions)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits

    def next_token(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, norm: RMSNorm, keep_all_positions: bool = False) -> torch.Tensor:
        if not keep_all_positions:
            x = x[:, -1:]
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        logits = self.get_logits(norm(x), keep_all_positions=keep_all_positions)
        values, indices = logits.max(dim=-1)
        if world_size > 1:
            all_values = [torch.empty_like(values) for _ in range(world_size)]
            all_indices = [torch.empty_like(indices) for _ in range(world_size)]
            dist.all_gather(all_values, values)
            dist.all_gather(all_indices, indices)
            values = torch.stack(all_values, dim=0)
            indices = torch.stack(all_indices, dim=0)
            best_rank = values.argmax(dim=0)
            next_token = indices.gather(0, best_rank.unsqueeze(0)).squeeze(0)
            return next_token + best_rank.to(next_token.dtype) * self.part_vocab_size
        return indices

    def _hc_linear_fp16(self, x: torch.Tensor, hc_fn: torch.Tensor) -> torch.Tensor:
        weight = getattr(self, "hc_fp16_weight", None)
        if weight is None or weight.device != hc_fn.device:
            weight = hc_fn.detach().to(dtype=torch.float16)
            self.register_buffer("hc_fp16_weight", weight, persistent=False)
        return F.linear(x.to(torch.float16), weight).to(torch.float32)

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        if self.hc_fp16_enabled and x.is_cuda and hc_fn.is_cuda:
            mixes = self._hc_linear_fp16(x, hc_fn) * rsqrt
        else:
            mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)


class MTPBlock(Block):

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        self.e_proj = Linear(args.dim, args.dim, dtype=torch.int8 if args.mtp_int8 else None)
        self.h_proj = Linear(args.dim, args.dim, dtype=torch.int8 if args.mtp_int8 else None)
        if args.preload_mtp_e_proj_int8:
            self.e_proj.enable_online_int8()
        if args.preload_mtp_h_proj_int8:
            self.h_proj.enable_online_int8()
        self.enorm = RMSNorm(args.dim, args.norm_eps)
        self.hnorm = RMSNorm(args.dim, args.norm_eps)
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))
        self.embed: ParallelEmbedding = None
        self.head: ParallelHead = None

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, start_pos: int, input_ids: torch.Tensor) -> torch.Tensor:
        # x: [b,s,hc,d]
        assert self.embed is not None and self.head is not None
        e = self.embed(input_ids)
        e = self.enorm(e)
        x = self.hnorm(x)
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
        x = super().forward(x, start_pos, input_ids)
        logits = self.head(x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
        return logits


class Transformer(nn.Module):
    """Full DeepSeek-V4 model: embed -> HC-expand -> N blocks -> HC-head -> logits.
    Sets global state (world_size, rank, default_dtype, scale_fmt, scale_dtype) in __init__."""
    def __init__(self, args: ModelArgs):
        global world_size, rank, default_dtype, scale_fmt, scale_dtype
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
        scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim, self.norm_eps)
        self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps)
        self.mtp = torch.nn.ModuleList()
        for layer_id in range(args.n_mtp_layers):
            self.mtp.append(MTPBlock(args.n_layers + layer_id, args))
            self.mtp[-1].embed = self.embed
            self.mtp[-1].head = self.head
        self.hc_mult = hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))

    def prepare_cpu_expert_int8(self) -> None:
        for layer in self.layers:
            backend = getattr(layer.ffn, "cpu_backend", None)
            if backend is not None:
                backend.prepare_int8_weights()
        for layer in self.mtp:
            backend = getattr(layer.ffn, "cpu_backend", None)
            if backend is not None:
                backend.prepare_int8_weights()
        if (
            _env_enabled("DEEPSEEK_CPU_MOE_INPROC_SERVER")
            and rank == 0
            and world_size > 1
            and len(self.layers) > 0
        ):
            backends = []
            template_ffn = None
            for layer in self.layers:
                ffn = getattr(layer, "ffn", None)
                backend = getattr(ffn, "cpu_backend", None) if ffn is not None else None
                if backend is None:
                    raise RuntimeError(f"layer {layer.layer_id} missing cpu_backend; in-process CPU MoE server requires routed-experts-device=cpu")
                backends.append(backend)
                if template_ffn is None:
                    template_ffn = ffn
            if template_ffn is None:
                return
            shm_name = start_in_process_cpu_moe_server(
                backends=backends,
                dim=template_ffn.dim,
                topk=template_ffn.n_activated_experts,
                inter_dim=backends[0]._inter_dim,
                n_routed_experts=template_ffn.n_routed_experts,
                swiglu_limit=backends[0]._swiglu_limit,
                shm_name=os.getenv("DEEPSEEK_CPU_MOE_SERVER_SHM"),
                use_v2=True,
            )
            os.environ["DEEPSEEK_CPU_MOE_SERVER_SHM"] = shm_name
            print(f"deepseek inproc cpu moe server started shm={shm_name}", flush=True)

    def release_gpu_prefill_moe_cache(self) -> None:
        for layer in self.layers:
            layer.release_gpu_prefill_moe_cache()
        for layer in self.mtp:
            layer.release_gpu_prefill_moe_cache()

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0, return_next_token: bool = False, return_hidden: bool = False, keep_all_positions: bool = False):
        h = self.embed(input_ids)
        # Expand to hc_mult copies for Hyper-Connections
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        if self.layers:
            self.layers[0].prefetch_gpu_prefill_moe(h.device, input_ids.numel())
        for layer_idx, layer in enumerate(self.layers):
            prefetch_next = self.layers[layer_idx + 1] if layer_idx + 1 < len(self.layers) else None
            h = layer(h, start_pos, input_ids, prefetch_next=prefetch_next)
        if return_next_token:
            next_token = self.head.next_token(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm, keep_all_positions=keep_all_positions)
            if return_hidden:
                return next_token, h
            return next_token
        logits = self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm, keep_all_positions=keep_all_positions)
        if return_hidden:
            return logits, h
        return logits

    @torch.inference_mode()
    def draft_with_mtp(self, h: torch.Tensor, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Run a single MTP block over (h, input_ids) and return the greedy draft token.

        h:         [b, s, hc_mult, dim] — last main backbone hidden state at positions
                   [start_pos-s+1 .. start_pos]. Caller typically slices h[:, -1:] when
                   only the most recent position matters.
        input_ids: [b, s] — the tokens at the same positions as h.
        start_pos: position of the FIRST entry in input_ids (matches the convention used
                   by Transformer.forward / Block.forward).

        Returns a token tensor of shape [b] — argmax of the MTP logits at the last
        position, i.e. the draft token for position (start_pos + s).
        """
        if not self.mtp:
            raise RuntimeError("draft_with_mtp called but Transformer has no MTP layers")
        logits = self.mtp[0](h, start_pos, input_ids)  # MTPBlock already calls head; returns [b, vocab]
        # head.get_logits already slices x[:, -1] internally and head.forward all-gathers
        # across TP world_size, so logits is [b, vocab] for the LAST position only.
        return logits.argmax(dim=-1)


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs(n_hash_layers=0)
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)

    print(model(x).size())
    for i in range(128, 150):
        print(i, model(x[:, 0:1], i).size())

    h = torch.randn(2, 128, args.hc_mult, args.dim)
    mtp = model.mtp[0]
    print(mtp(h, 0, x).size())
    print(mtp(h[:, 0:1], 1, x[:, 0:1]).size())
