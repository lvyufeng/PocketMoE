import os
import struct
from dataclasses import dataclass
from typing import Optional

import torch

from src.kernels.cuda_loader import load_cuda_kernel


_FORCE_FALLBACK = os.getenv("DEEPSEEK_FORCE_FALLBACK", "0").lower() in {"1", "true", "yes"}
_FORCE_TORCH_GEMM = os.getenv("DEEPSEEK_FORCE_TORCH_GEMM", "0").lower() in {"1", "true", "yes"}
_FP8_IMPL = os.getenv("DEEPSEEK_FP8_IMPL", "auto")
_FP4_IMPL = os.getenv("DEEPSEEK_FP4_IMPL", "auto")
_INT8_IMPL = os.getenv("DEEPSEEK_INT8_IMPL", "torch")
_FUSED_DECODE_ATTN = os.getenv("DEEPSEEK_FUSED_DECODE_ATTN", "0").lower() in {"1", "true", "yes"}
_FUSED_DECODE_ATTN_CUDA = os.getenv("DEEPSEEK_FUSED_DECODE_ATTN_CUDA", "0").lower() in {"1", "true", "yes"}
_TENSOR_CORE_ATTN = os.getenv("DEEPSEEK_TENSOR_CORE_ATTN", "0").lower() in {"1", "true", "yes"}
_TENSOR_CORE_ATTN_CUDA = os.getenv("DEEPSEEK_TENSOR_CORE_ATTN_CUDA", "0").lower() in {"1", "true", "yes"}
_FLASHINFER_STYLE_ATTN_CUDA = os.getenv("DEEPSEEK_FLASHINFER_STYLE_ATTN_CUDA", "0").lower() in {"1", "true", "yes"}
_PREFILL_SPARSE_ATTN_CUDA = os.getenv("DEEPSEEK_PREFILL_SPARSE_ATTN_CUDA", "1").lower() in {"1", "true", "yes"}
_PREFILL_SPARSE_ATTN_HEADPAIR_CUDA = os.getenv("DEEPSEEK_PREFILL_SPARSE_ATTN_HEADPAIR_CUDA", "1").lower() in {"1", "true", "yes"}
_SHARED_EXPERT_PAIR_INT8_CUDA = os.getenv("DEEPSEEK_SHARED_EXPERT_PAIR_INT8_CUDA", "0").lower() in {"1", "true", "yes"}
_INT8_CUDA_EXT = None



_USE_TRITON = False
if not _FORCE_TORCH_GEMM:
    try:
        import triton
        import triton.language as tl
        _USE_TRITON = True
    except Exception:
        _USE_TRITON = False


_FP8_MAX = 448.0
_FP8_MIN = -448.0
_FP4_MAX = 6.0
_FP4_MIN = -6.0
_FP4_MIN_SCALE = 6 * (2 ** -126)
_FP4_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_SIGNED_INT32_0X87F00000 = -2014314496


@dataclass
class Packed4BitWeightAlongK:
    plain_shape: tuple[int, ...]
    layout_tensor: torch.Tensor
    k_stride: int = 1

    @classmethod
    def convert_from(cls, tensor: torch.Tensor, *, k_stride: int = 1) -> "Packed4BitWeightAlongK":
        raw = tensor.view(torch.uint8).contiguous()
        *batch_dims, k_half = raw.shape
        return cls(plain_shape=(*batch_dims, k_half * 2), layout_tensor=raw, k_stride=k_stride)


def _auto_impl(kind: str) -> str:
    if kind == "fp8_quant":
        return "torch"
    if kind == "fp8":
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major < 8:
                return "torch"
        if _USE_TRITON:
            return "triton"
        return "torch"
    if kind == "fp4":
        if _USE_TRITON:
            return "triton"
        return "torch"
    if kind == "int8":
        if _USE_TRITON:
            return "triton"
        return "torch"
    return "torch"


def _resolve_impl(kind: str, impl: str) -> str:
    if impl == "auto":
        return _auto_impl(kind)
    if impl == "triton" and not _USE_TRITON:
        return "torch"
    return impl


def _round_scale_pow2(x: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(torch.float32).tiny
    return torch.pow(2.0, torch.ceil(torch.log2(torch.clamp(x, min=tiny))))


def _repeat_last_dim(scales: torch.Tensor, block_size: int, target: int) -> torch.Tensor:
    return scales.repeat_interleave(block_size, dim=-1)[..., :target]


def _repeat_block_scales_2d(scales: torch.Tensor, block_m: int, block_k: int, rows: int, cols: int) -> torch.Tensor:
    out = scales.to(torch.float32).repeat_interleave(block_m, dim=0)
    out = out[:rows].repeat_interleave(block_k, dim=1)
    return out[:, :cols]


def _fp4_levels(device: torch.device) -> torch.Tensor:
    return _FP4_LEVELS.to(device=device)


def _quantize_fp4_codes(x: torch.Tensor) -> torch.Tensor:
    levels = _fp4_levels(x.device)
    distances = (x.unsqueeze(-1) - levels).abs()
    return distances.argmin(dim=-1).to(torch.uint8)


def _pack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    low = codes[..., 0::2]
    high = codes[..., 1::2] << 4
    packed = (low | high).contiguous()
    return packed.view(torch.int8).view(torch.float4_e2m1fn_x2)


def _unpack_fp4_tensor(x: torch.Tensor) -> torch.Tensor:
    raw = x.contiguous().view(torch.uint8)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    codes = torch.empty(raw.shape[0], raw.shape[1] * 2, dtype=torch.uint8, device=x.device)
    codes[:, 0::2] = low
    codes[:, 1::2] = high
    levels = _fp4_levels(x.device)
    return levels[codes.long()]


def _dequant_fp8_acts(a: torch.Tensor, a_s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    k = a.size(-1)
    scales = _repeat_last_dim(a_s.to(torch.float32), block_size, k)
    return a.to(torch.float32) * scales


def _dequant_fp8_weight_torch(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    n, k = weight.shape
    scales = _repeat_block_scales_2d(scale, block_size, block_size, n, k)
    return weight.to(torch.float32) * scales


def _dequant_fp4_weight_torch(weight: Packed4BitWeightAlongK, scale: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    unpacked = _unpack_fp4_tensor(weight.layout_tensor)
    k = unpacked.size(-1)
    scales = _repeat_last_dim(scale.to(torch.float32), block_size, k)
    return unpacked * scales


def _choose_compute_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        if major < 8:
            return torch.float16
        if torch.get_default_dtype() == torch.bfloat16:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _to_output_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        if major < 8:
            return torch.float16
    return torch.get_default_dtype()


if _USE_TRITON:
    @triton.jit
    def fast_log2_ceil(x):
        bits_x = tl.cast(x, tl.uint32, bitcast=True)
        exp_x = (bits_x >> 23) & 0xFF
        man_bits = bits_x & ((1 << 23) - 1)
        return tl.cast(exp_x - 127 + tl.where(man_bits != 0, 1, 0), tl.int32)

    @triton.jit
    def fast_pow2(x):
        bits_x = (x + 127) << 23
        return tl.cast(bits_x, tl.float32, bitcast=True)

    @triton.jit
    def fast_round_scale(amax, fp8_max_inv):
        return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))

    @triton.jit
    def _blockfp8_act_quant_kernel(
        x_ptr,
        y_ptr,
        s_ptr,
        stride_x0,
        stride_y0,
        hidden_dim,
        BLOCK_SIZE: tl.constexpr,
        SCALE_IS_UE8M0: tl.constexpr,
        EPS: tl.constexpr,
        INDEX_DTYPE: tl.constexpr,
    ):
        row_id = tl.program_id(axis=0)
        block_id = tl.program_id(axis=1)

        x_offs = (
            tl.cast(row_id, INDEX_DTYPE) * stride_x0
            + tl.cast(block_id, INDEX_DTYPE) * BLOCK_SIZE
            + tl.arange(0, BLOCK_SIZE)
        )
        y_offs = (
            tl.cast(row_id, INDEX_DTYPE) * stride_y0
            + tl.cast(block_id, INDEX_DTYPE) * BLOCK_SIZE
            + tl.arange(0, BLOCK_SIZE)
        )
        s_offs = row_id * (hidden_dim // BLOCK_SIZE) + block_id

        x = tl.load(x_ptr + x_offs).to(tl.float32)
        s = tl.maximum(tl.max(tl.abs(x)), EPS)
        if SCALE_IS_UE8M0:
            s = fast_round_scale(s, 1.0 / 448.0)
        else:
            s /= 448.0
        y = (x / s).to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + y_offs, y)
        tl.store(s_ptr + s_offs, s)

    @triton.jit
    def _soft_fp8_blockfp8_weight_dequant_kernel_step_1(
        x_ptr,
        y_ptr,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask)
        x = x.to(tl.int8, bitcast=True).to(tl.int32)
        x = (x << 20) & -2014314496
        y = x.to(tl.uint32, bitcast=True)
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.jit
    def _soft_fp8_blockfp8_weight_dequant_kernel_step_2(
        x_ptr,
        s_ptr,
        y_ptr,
        M,
        N,
        BLOCK_SIZE: tl.constexpr,
        fp8_to_fp32_scale: tl.constexpr,
    ):
        pid_b = tl.program_id(axis=0)
        pid_m = tl.program_id(axis=1)
        pid_n = tl.program_id(axis=2)
        n = tl.cdiv(N, BLOCK_SIZE)
        m = tl.cdiv(M, BLOCK_SIZE)
        offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs = pid_b * M * N + offs_m[:, None] * N + offs_n[None, :]
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs, mask=mask)
        s = tl.load(s_ptr + pid_b * m * n + pid_m * n + pid_n)
        y = x * (s * fp8_to_fp32_scale)
        tl.store(y_ptr + offs, y, mask=mask)

    _soft_fp8_blockfp8_gemm_configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": group_m,
            },
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_m in [16, 32]
        for block_n in [64, 128]
        for group_m in [1, 32]
        for num_stages in [3, 4]
        for num_warps in [4, 8]
    ]

    @triton.autotune(configs=_soft_fp8_blockfp8_gemm_configs, key=["N", "K"])
    @triton.jit
    def _soft_fp8_blockfp8_gemm_kernel(
        A,
        B,
        C,
        Bs,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        group_n: tl.constexpr,
        group_k: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        fp8_to_fp32_scale: tl.constexpr,
        compute_dtype: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = A + (offs_am[:, None] * K + offs_k[None, :])
        b_ptrs = B + (offs_k[:, None] + offs_bn[None, :] * K)

        offs_bsn = offs_bn // group_n
        Bs_ptrs = Bs + offs_bsn * tl.cdiv(K, BLOCK_SIZE_K)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // group_k
            b_s = tl.load(Bs_ptrs + offs_ks)

            t = b.to(tl.int8, bitcast=True).to(tl.int32)
            t = (t << 20) & -2014314496
            b_unscaled_fp32 = t.to(tl.float32, bitcast=True)
            b_scaled_fp32 = (b_unscaled_fp32 * (b_s * fp8_to_fp32_scale)).to(compute_dtype)
            accumulator += tl.dot(a, b_scaled_fp32)

            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K

        c = accumulator.to(compute_dtype)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = C + N * offs_cm[:, None] + offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    @triton.jit
    def _decode_sparse_attn_kernel(
        q_ptr,
        kv_ptr,
        sink_ptr,
        idx_ptr,
        out_ptr,
        topk: tl.constexpr,
        kv_len: tl.constexpr,
        heads: tl.constexpr,
        dim: tl.constexpr,
        stride_q_b: tl.constexpr,
        stride_q_h: tl.constexpr,
        stride_kv_b: tl.constexpr,
        stride_kv_t: tl.constexpr,
        stride_out_b: tl.constexpr,
        stride_out_h: tl.constexpr,
        softmax_scale: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_d = tl.program_id(2)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = offs_d < dim

        max_score = tl.load(sink_ptr + pid_h).to(tl.float32)
        denom = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start in range(0, topk, BLOCK_T):
            offs_t = start + tl.arange(0, BLOCK_T)
            valid_t = offs_t < topk
            idx = tl.load(idx_ptr + pid_b * topk + offs_t, mask=valid_t, other=-1)
            valid = valid_t & (idx >= 0) & (idx < kv_len)
            scores = tl.zeros((BLOCK_T,), dtype=tl.float32)
            for d0 in range(0, dim, BLOCK_D):
                kd = d0 + tl.arange(0, BLOCK_D)
                mask_d = kd < dim
                q_blk = tl.load(q_ptr + pid_b * stride_q_b + pid_h * stride_q_h + kd, mask=mask_d, other=0.0).to(tl.float32)
                kv = tl.load(kv_ptr + pid_b * stride_kv_b + idx[:, None] * stride_kv_t + kd[None, :], mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
                scores += tl.sum(kv * q_blk[None, :], axis=1)
            scores = tl.where(valid, scores * softmax_scale, -float("inf"))
            max_score = tl.maximum(max_score, tl.max(scores, axis=0))

        sink_score = tl.load(sink_ptr + pid_h).to(tl.float32)
        denom += tl.exp(sink_score - max_score)
        for start in range(0, topk, BLOCK_T):
            offs_t = start + tl.arange(0, BLOCK_T)
            valid_t = offs_t < topk
            idx = tl.load(idx_ptr + pid_b * topk + offs_t, mask=valid_t, other=-1)
            valid = valid_t & (idx >= 0) & (idx < kv_len)
            scores = tl.zeros((BLOCK_T,), dtype=tl.float32)
            for d0 in range(0, dim, BLOCK_D):
                kd = d0 + tl.arange(0, BLOCK_D)
                mask_d = kd < dim
                q_blk = tl.load(q_ptr + pid_b * stride_q_b + pid_h * stride_q_h + kd, mask=mask_d, other=0.0).to(tl.float32)
                kv = tl.load(kv_ptr + pid_b * stride_kv_b + idx[:, None] * stride_kv_t + kd[None, :], mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
                scores += tl.sum(kv * q_blk[None, :], axis=1)
            scores = tl.where(valid, scores * softmax_scale, -float("inf"))
            weights = tl.exp(scores - max_score)
            denom += tl.sum(weights, axis=0)
            kv_out = tl.load(kv_ptr + pid_b * stride_kv_b + idx[:, None] * stride_kv_t + offs_d[None, :], mask=valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(weights[:, None] * kv_out, axis=0)
        acc = acc / denom
        tl.store(out_ptr + pid_b * stride_out_b + pid_h * stride_out_h + offs_d, acc, mask=d_mask)

    @triton.jit
    def _matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc = tl.dot(a, b, acc)
            offs_k += BLOCK_K
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)


def _triton_matmul(a: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    k2, n = b_t.shape
    assert k == k2
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(m, 64), triton.cdiv(n, 64))
    _matmul_kernel[grid](
        a,
        b_t,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b_t.stride(0),
        b_t.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return c


def _soft_gemm(a_ref: torch.Tensor, b_ref: torch.Tensor, impl: str) -> torch.Tensor:
    impl = _resolve_impl("fp8", impl)
    if impl == "triton" and a_ref.is_cuda and b_ref.is_cuda:
        compute_dtype = _choose_compute_dtype(a_ref.device)
        a_comp = a_ref.to(compute_dtype).contiguous()
        b_t = b_ref.transpose(0, 1).to(compute_dtype).contiguous()
        return _triton_matmul(a_comp, b_t)
    return torch.matmul(a_ref, b_ref.transpose(0, 1))


def blockfp8_act_quant_torch(
    x: torch.Tensor,
    *,
    block_size: int = 128,
    round_scale_to_pow2: bool = False,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.shape[-1] % block_size == 0
    z = x.contiguous()
    flat = z.view(-1, z.shape[-1]).to(torch.float32)
    scales = flat.abs().view(-1, z.shape[-1] // block_size, block_size).amax(dim=-1).clamp_min(eps) / _FP8_MAX
    if round_scale_to_pow2:
        scales = _round_scale_pow2(scales)
    scale_full = scales.repeat_interleave(block_size, dim=-1)
    quant = torch.clamp(flat / scale_full, _FP8_MIN, _FP8_MAX).to(torch.float8_e4m3fn)
    return quant.view_as(z), scales.to(torch.float32).view(*z.shape[:-1], z.shape[-1] // block_size)


def blockfp8_act_quant_triton(
    x: torch.Tensor,
    *,
    block_size: int = 128,
    round_scale_to_pow2: bool = False,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert _USE_TRITON
    assert x.is_contiguous()
    assert x.shape[-1] % block_size == 0
    y = torch.empty(*x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(*x.shape[:-1], x.shape[-1] // block_size, dtype=torch.float32, device=x.device)
    index_dtype = tl.int64 if x.numel() >= 2147483648 else tl.int32
    grid = (x.view(-1, x.shape[-1]).shape[0], triton.cdiv(x.shape[-1], block_size))
    _blockfp8_act_quant_kernel[grid](
        x,
        y,
        s,
        stride_x0=x.view(-1, x.shape[-1]).stride(0),
        stride_y0=y.view(-1, y.shape[-1]).stride(0),
        hidden_dim=x.shape[-1],
        BLOCK_SIZE=block_size,
        SCALE_IS_UE8M0=round_scale_to_pow2,
        EPS=eps,
        INDEX_DTYPE=index_dtype,
    )
    return y, s


def blockfp8_act_quant(
    x: torch.Tensor,
    *,
    block_size: int = 128,
    round_scale_to_pow2: bool = False,
    eps: float = 1e-4,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    impl = _resolve_impl("fp8_quant", impl)
    if impl == "triton" and _USE_TRITON:
        return blockfp8_act_quant_triton(
            x,
            block_size=block_size,
            round_scale_to_pow2=round_scale_to_pow2,
            eps=eps,
        )
    return blockfp8_act_quant_torch(
        x,
        block_size=block_size,
        round_scale_to_pow2=round_scale_to_pow2,
        eps=eps,
    )


def soft_fp8_blockfp8_weight_dequant_torch(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    return _dequant_fp8_weight_torch(weight, scale, block_size)


def soft_fp8_blockfp8_weight_dequant_triton(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    assert _USE_TRITON
    assert weight.is_contiguous() and scale.is_contiguous()
    assert weight.dim() == 2 and scale.dim() == 2
    m, n = weight.shape
    raw = weight.view(torch.uint8)
    bit_reordered = torch.empty_like(raw, dtype=torch.uint32 if hasattr(torch, "uint32") else torch.int32)
    grid = (triton.cdiv(raw.numel(), 1024),)
    _soft_fp8_blockfp8_weight_dequant_kernel_step_1[grid](raw, bit_reordered, raw.numel(), BLOCK_SIZE=1024)
    bit_reordered = bit_reordered.view(torch.float32)
    out = torch.empty_like(weight, dtype=_to_output_dtype(weight.device))
    fp8_to_fp32_scale = struct.unpack(">f", bytes.fromhex("7b800000"))[0]
    grid = (1, triton.cdiv(m, block_size), triton.cdiv(n, block_size))
    _soft_fp8_blockfp8_weight_dequant_kernel_step_2[grid](
        bit_reordered,
        scale,
        out,
        m,
        n,
        BLOCK_SIZE=block_size,
        fp8_to_fp32_scale=fp8_to_fp32_scale,
    )
    return out


def soft_fp8_blockfp8_weight_dequant(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
    impl: str = "auto",
) -> torch.Tensor:
    impl = _resolve_impl("fp8", impl)
    if impl == "triton" and _USE_TRITON:
        return soft_fp8_blockfp8_weight_dequant_triton(weight, scale, block_size)
    return soft_fp8_blockfp8_weight_dequant_torch(weight, scale, block_size)


def soft_fp4_blockfp4_weight_dequant_torch(
    weight: Packed4BitWeightAlongK,
    scale: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    return _dequant_fp4_weight_torch(weight, scale, block_size)


def soft_fp4_blockfp4_weight_dequant(
    weight: Packed4BitWeightAlongK,
    scale: torch.Tensor,
    block_size: int = 32,
    impl: str = "auto",
) -> torch.Tensor:
    impl = _resolve_impl("fp4", impl)
    if impl in {"triton", "torch"}:
        return soft_fp4_blockfp4_weight_dequant_torch(weight, scale, block_size)
    raise NotImplementedError(f"Unsupported fp4 dequant impl: {impl}")


def soft_fp8_blockfp8_gemm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    weight_dequant = soft_fp8_blockfp8_weight_dequant_torch(weight, scale, block_size=128)
    out = torch.matmul(x.to(torch.float32), weight_dequant.transpose(0, 1))
    return out.to(torch.get_default_dtype())


def soft_bf16_weight_gemm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    out = torch.matmul(x.to(torch.float32), weight.transpose(0, 1).to(torch.float32))
    return out.to(torch.get_default_dtype())


def _quantize_int8_weight_torch(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight_f32 = weight.to(torch.float32).contiguous()
    scales = weight_f32.abs().amax(dim=1).clamp_min(1e-6) / 127.0
    quant = torch.clamp(torch.round(weight_f32 / scales.unsqueeze(1)), -127, 127).to(torch.int8)
    return quant.contiguous(), scales.to(torch.float32).contiguous()


if _USE_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_M": 1, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_M": 2, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _wo_a_int8_gemm_kernel(
        X,
        W,
        Ws,
        Y,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_ws,
        stride_ym,
        stride_yn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        x_ptrs = X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_scale = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, tl.cdiv(K, BLOCK_K)):
            k_offs = k0 * BLOCK_K + offs_k
            x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
            x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
            x_scale = tl.maximum(x_scale, tl.max(tl.abs(x), axis=1))
            x_ptrs += BLOCK_K * stride_xk
        x_scale = tl.maximum(x_scale, 1e-6) / 127.0

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        x_ptrs = X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = W + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        for k0 in range(0, tl.cdiv(K, BLOCK_K)):
            k_offs = k0 * BLOCK_K + offs_k
            x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
            w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
            x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptrs, mask=w_mask, other=0).to(tl.int8)
            xq = tl.clamp(tl.math.floor(x / x_scale[:, None] + 0.5), -127.0, 127.0).to(tl.int8)
            acc += tl.dot(xq, w, out_dtype=tl.int32)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        w_scale = tl.load(Ws + offs_n * stride_ws, mask=offs_n < N, other=0.0).to(tl.float32)
        y = acc.to(tl.float32) * x_scale[:, None] * w_scale[None, :]
        y = y.to(Y.dtype.element_ty)
        y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(y_ptrs, y, mask=y_mask)


def soft_bf16_weight_gemm_int8_torch(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    weight_s: torch.Tensor,
) -> torch.Tensor:
    x_f32 = x.to(torch.float32).contiguous()
    x_scales = x_f32.abs().amax(dim=-1).clamp_min(1e-6) / 127.0
    x_q = torch.clamp(torch.round(x_f32 / x_scales.unsqueeze(-1)), -127, 127).to(torch.int8).contiguous()
    flat = x_q.view(-1, x_q.shape[-1])
    m = flat.size(0)
    pad_rows = max(0, 24 - m)
    rem = (m + pad_rows) % 8
    if rem:
        pad_rows += 8 - rem
    if pad_rows:
        flat = torch.cat([flat, torch.zeros((pad_rows, flat.size(1)), dtype=torch.int8, device=flat.device)], dim=0)
        x_scales_flat = torch.cat([x_scales.view(-1), torch.zeros((pad_rows,), dtype=torch.float32, device=x_scales.device)], dim=0)
    else:
        x_scales_flat = x_scales.view(-1)
    prod_i32 = torch._int_mm(flat, weight_q.transpose(0, 1).contiguous())
    prod = prod_i32[:m].to(torch.float32)
    prod = prod * x_scales_flat[:m].unsqueeze(1) * weight_s.unsqueeze(0)
    return prod.view(*x.shape[:-1], weight_q.size(0)).to(torch.get_default_dtype())


def soft_bf16_weight_gemm_int8_triton(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    weight_s: torch.Tensor,
) -> torch.Tensor:
    assert _USE_TRITON
    m, k = x.view(-1, x.shape[-1]).shape
    n = weight_q.size(0)
    x2 = x.view(m, k).contiguous()
    y = torch.empty((m, n), device=x.device, dtype=_to_output_dtype(x.device))
    grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]), triton.cdiv(n, META["BLOCK_N"]))
    _wo_a_int8_gemm_kernel[grid](
        x2,
        weight_q.transpose(0, 1).contiguous(),
        weight_s,
        y,
        m,
        n,
        k,
        x2.stride(0),
        x2.stride(1),
        weight_q.transpose(0, 1).contiguous().stride(1),
        weight_q.transpose(0, 1).contiguous().stride(0),
        weight_s.stride(0),
        y.stride(0),
        y.stride(1),
    )
    return y.view(*x.shape[:-1], n).to(torch.get_default_dtype())


def soft_bf16_weight_gemm_int8_cuda_ext(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    weight_s: torch.Tensor,
) -> torch.Tensor:
    global _INT8_CUDA_EXT
    if _INT8_CUDA_EXT is None:
        _INT8_CUDA_EXT = load_cuda_kernel()
    if _INT8_CUDA_EXT is None:
        raise RuntimeError("INT8 CUDA extension is unavailable")
    y = _INT8_CUDA_EXT.int8_gemm_forward(
        x.contiguous(),
        weight_q.contiguous(),
        weight_s.contiguous(),
    )
    return y.to(torch.get_default_dtype())


def soft_bf16_weight_gemm_int8_pair_cuda_ext(
    x: torch.Tensor,
    weight_q0: torch.Tensor,
    weight_s0: torch.Tensor,
    weight_q1: torch.Tensor,
    weight_s1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    global _INT8_CUDA_EXT
    if _INT8_CUDA_EXT is None:
        _INT8_CUDA_EXT = load_cuda_kernel()
    if _INT8_CUDA_EXT is None or not hasattr(_INT8_CUDA_EXT, "int8_gemm_pair_forward"):
        raise RuntimeError("paired INT8 CUDA extension is unavailable")
    y = _INT8_CUDA_EXT.int8_gemm_pair_forward(
        x.contiguous(),
        weight_q0.contiguous(),
        weight_s0.contiguous(),
        weight_q1.contiguous(),
        weight_s1.contiguous(),
    )
    return y[0].to(torch.get_default_dtype()), y[1].to(torch.get_default_dtype())


def soft_bf16_weight_gemm_int8(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    weight_s: torch.Tensor,
    impl: str = "auto",
) -> torch.Tensor:
    if impl == "auto":
        impl = _INT8_IMPL
    flat_m = x.numel() // x.shape[-1]
    if impl == "cuda_ext" and x.is_cuda and weight_q.is_cuda and weight_s.is_cuda and flat_m <= 4:
        try:
            return soft_bf16_weight_gemm_int8_cuda_ext(x, weight_q, weight_s)
        except Exception:
            pass
    if impl == "triton" and _USE_TRITON:
        return soft_bf16_weight_gemm_int8_triton(x, weight_q, weight_s)
    return soft_bf16_weight_gemm_int8_torch(x, weight_q, weight_s)


def soft_fp8_blockfp8_gemm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    assert _USE_TRITON
    assert x.is_contiguous() and weight.is_contiguous() and scale.is_contiguous()
    k = x.size(-1)
    m = x.numel() // k
    n = weight.size(0)
    compute_dtype = _choose_compute_dtype(x.device)
    out_dtype = _to_output_dtype(x.device)
    x_comp = x.to(compute_dtype).contiguous().view(m, k)
    c = torch.empty((m, n), device=x.device, dtype=out_dtype)
    fp8_to_fp32_scale = struct.unpack(">f", bytes.fromhex("7b800000"))[0]

    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_SIZE_M"]) * triton.cdiv(n, META["BLOCK_SIZE_N"]),
    )
    _soft_fp8_blockfp8_gemm_kernel[grid](
        x_comp,
        weight.view(torch.uint8),
        c,
        scale,
        m,
        n,
        k,
        group_n=128,
        group_k=128,
        fp8_to_fp32_scale=fp8_to_fp32_scale,
        compute_dtype=tl.float16 if compute_dtype == torch.float16 else tl.bfloat16,
    )
    return c.view(*x.shape[:-1], n).to(torch.get_default_dtype())


def soft_fp8_blockfp8_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    impl: str = "auto",
) -> torch.Tensor:
    impl = _resolve_impl("fp8", impl)
    if impl == "triton" and _USE_TRITON:
        return soft_fp8_blockfp8_gemm_triton(x, weight, scale)
    return soft_fp8_blockfp8_gemm_torch(x, weight, scale)


def soft_fp4_raise_to_bf16_blockfp4_gemm(
    x: torch.Tensor,
    weight: Packed4BitWeightAlongK,
    scale: torch.Tensor,
    impl: str = "auto",
) -> torch.Tensor:
    weight_dequant = soft_fp4_blockfp4_weight_dequant(weight, scale, block_size=32, impl=impl)
    out = _soft_gemm(x.to(torch.float32), weight_dequant, impl)
    return out.to(torch.get_default_dtype())


def soft_fp4_raise_to_fp8_blockfp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: Packed4BitWeightAlongK,
    b_s: torch.Tensor,
    act_block_size: int = 128,
    impl: str = "auto",
) -> torch.Tensor:
    a_ref = _dequant_fp8_acts(a, a_s, act_block_size)
    weight_dequant = soft_fp4_blockfp4_weight_dequant(b, b_s, block_size=32, impl=impl)
    out = _soft_gemm(a_ref, weight_dequant, impl)
    return out.to(torch.get_default_dtype())


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    quant, scales = blockfp8_act_quant(
        x,
        block_size=block_size,
        round_scale_to_pow2=scale_fmt is not None,
        eps=1e-4,
        impl=_FP8_IMPL,
    )
    if inplace:
        scale_full = scales.repeat_interleave(block_size, dim=-1)
        dequant = (quant.to(torch.float32) * scale_full).to(x.dtype).view_as(x)
        x.copy_(dequant)
        return x
    return quant, scales.to(scale_dtype)


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
):
    n = x.size(-1)
    assert n % block_size == 0
    assert n % 2 == 0
    z = x.contiguous()
    flat = z.view(-1, n).to(torch.float32)
    scales = flat.abs().view(-1, n // block_size, block_size).amax(dim=-1).clamp_min(_FP4_MIN_SCALE) / _FP4_MAX
    scales = _round_scale_pow2(scales)
    scale_full = scales.repeat_interleave(block_size, dim=-1)
    normalized = torch.clamp(flat / scale_full, _FP4_MIN, _FP4_MAX)
    quant_chunk_rows = max(1, int(os.getenv("DEEPSEEK_FP4_QUANT_CHUNK_ROWS", "8192")))
    if flat.size(0) > quant_chunk_rows:
        levels = _fp4_levels(z.device)
        if inplace:
            for start in range(0, flat.size(0), quant_chunk_rows):
                end = min(start + quant_chunk_rows, flat.size(0))
                codes = _quantize_fp4_codes(normalized[start:end])
                flat[start:end].copy_(levels[codes.long()] * scale_full[start:end])
            x.copy_(flat.to(z.dtype).view_as(z))
            return x
        codes = torch.empty_like(flat, dtype=torch.uint8)
        for start in range(0, flat.size(0), quant_chunk_rows):
            end = min(start + quant_chunk_rows, flat.size(0))
            codes[start:end] = _quantize_fp4_codes(normalized[start:end])
    else:
        codes = _quantize_fp4_codes(normalized)
        if inplace:
            dequant = (_fp4_levels(z.device)[codes.long()] * scale_full).to(z.dtype).view_as(z)
            x.copy_(dequant)
            return x
    packed = _pack_fp4_codes(codes).view(*z.shape[:-1], n // 2)
    out_scales = scales.to(torch.float8_e8m0fnu).view(*z.shape[:-1], n // block_size)
    return packed, out_scales


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    a_ref = _dequant_fp8_acts(a, a_s, 128)
    return soft_fp8_blockfp8_gemm(a_ref, b, b_s, impl=_FP8_IMPL)


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    packed = Packed4BitWeightAlongK.convert_from(b)
    return soft_fp4_raise_to_fp8_blockfp4_gemm(a, a_s, packed, b_s, act_block_size=128, impl=_FP4_IMPL)


def _fused_decode_sparse_attn_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    assert _USE_TRITON
    b, s, h, d = q.shape
    assert s == 1
    topk = topk_idxs.size(-1)
    out = torch.empty((b, h, d), device=q.device, dtype=q.dtype)
    _decode_sparse_attn_kernel[(b, h, triton.cdiv(d, 64))](
        q.contiguous(),
        kv.contiguous(),
        attn_sink.contiguous(),
        topk_idxs.contiguous(),
        out,
        topk,
        kv.size(1),
        h,
        d,
        h * d,
        d,
        kv.size(1) * d,
        d,
        h * d,
        d,
        float(softmax_scale),
        BLOCK_T=1024,
        BLOCK_D=64,
        num_warps=8,
        num_stages=3,
    )
    return out.view(b, 1, h, d)


def _tensor_core_decode_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    b, s, h, d = q.shape
    assert s == 1
    topk = topk_idxs.size(-1)
    safe_idxs = topk_idxs.clamp_min(0)
    batch_idx = torch.arange(b, device=q.device).view(b, 1, 1).expand_as(safe_idxs)
    gathered = kv[batch_idx, safe_idxs]
    valid = topk_idxs >= 0
    gathered_h = gathered[:, 0].unsqueeze(1).expand(b, h, topk, d)
    q_tc = q[:, 0].reshape(b, h, 1, d).reshape(b * h, 1, d).to(torch.float16)
    kv_tc = gathered_h.reshape(b * h, topk, d).transpose(1, 2).contiguous().to(torch.float16)
    scores = torch.bmm(q_tc, kv_tc).view(b, h, 1, topk).permute(0, 2, 1, 3).to(torch.float32) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink = attn_sink.to(torch.float32).view(1, 1, h, 1).expand(b, 1, h, 1)
    weights = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :topk]
    weights_tc = weights.permute(0, 2, 1, 3).reshape(b * h, 1, topk).to(torch.float16)
    v_tc = gathered_h.reshape(b * h, topk, d).contiguous().to(torch.float16)
    out = torch.bmm(weights_tc, v_tc).view(b, h, 1, d).permute(0, 2, 1, 3)
    return out.to(q.dtype)


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    global _INT8_CUDA_EXT
    if _PREFILL_SPARSE_ATTN_CUDA and q.is_cuda and q.size(1) > 1 and q.size(-1) == 512:
        if _INT8_CUDA_EXT is None:
            _INT8_CUDA_EXT = load_cuda_kernel()
        if (
            _PREFILL_SPARSE_ATTN_HEADPAIR_CUDA
            and _INT8_CUDA_EXT is not None
            and hasattr(_INT8_CUDA_EXT, "prefill_sparse_attn_headpair_forward")
        ):
            return _INT8_CUDA_EXT.prefill_sparse_attn_headpair_forward(
                q.contiguous(),
                kv.contiguous(),
                attn_sink.contiguous(),
                topk_idxs.contiguous().to(torch.int32),
                float(softmax_scale),
            )
        if _INT8_CUDA_EXT is not None and hasattr(_INT8_CUDA_EXT, "prefill_sparse_attn_forward"):
            return _INT8_CUDA_EXT.prefill_sparse_attn_forward(
                q.contiguous(),
                kv.contiguous(),
                attn_sink.contiguous(),
                topk_idxs.contiguous().to(torch.int32),
                float(softmax_scale),
            )
    if _FLASHINFER_STYLE_ATTN_CUDA and q.is_cuda and q.size(1) == 1 and q.size(-1) == 512 and topk_idxs.size(-1) <= 256:
        if _INT8_CUDA_EXT is None:
            _INT8_CUDA_EXT = load_cuda_kernel()
        if _INT8_CUDA_EXT is not None and hasattr(_INT8_CUDA_EXT, "flashinfer_style_sparse_attn_forward"):
            return _INT8_CUDA_EXT.flashinfer_style_sparse_attn_forward(
                q.contiguous(),
                kv.contiguous(),
                attn_sink.contiguous(),
                topk_idxs.contiguous().to(torch.int32),
                float(softmax_scale),
            )
    if _TENSOR_CORE_ATTN_CUDA and q.is_cuda and q.size(1) == 1 and q.size(-1) == 512 and topk_idxs.size(-1) <= 256:
        if _INT8_CUDA_EXT is None:
            _INT8_CUDA_EXT = load_cuda_kernel()
        if _INT8_CUDA_EXT is not None and hasattr(_INT8_CUDA_EXT, "fused_decode_sparse_attn_wmma_forward"):
            return _INT8_CUDA_EXT.fused_decode_sparse_attn_wmma_forward(
                q.contiguous().to(torch.float16),
                kv.contiguous().to(torch.float16),
                attn_sink.contiguous(),
                topk_idxs.contiguous().to(torch.int32),
                float(softmax_scale),
            ).to(q.dtype)
    if _TENSOR_CORE_ATTN and q.is_cuda and q.size(1) == 1 and q.size(-1) == 512:
        return _tensor_core_decode_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)
    if _FUSED_DECODE_ATTN_CUDA and q.is_cuda and q.size(1) == 1:
        if _INT8_CUDA_EXT is None:
            _INT8_CUDA_EXT = load_cuda_kernel()
        if _INT8_CUDA_EXT is not None and hasattr(_INT8_CUDA_EXT, "fused_decode_sparse_attn_forward"):
            return _INT8_CUDA_EXT.fused_decode_sparse_attn_forward(
                q.contiguous(),
                kv.contiguous(),
                attn_sink.contiguous(),
                topk_idxs.contiguous().to(torch.int32),
                float(softmax_scale),
            )
    if _FUSED_DECODE_ATTN and _USE_TRITON and q.is_cuda and q.size(1) == 1 and q.size(-1) == 512 and topk_idxs.size(-1) <= 1024:
        return _fused_decode_sparse_attn_triton(q, kv, attn_sink, topk_idxs, softmax_scale)

    b, s, h, d = q.shape
    topk = topk_idxs.size(-1)
    safe_idxs = topk_idxs.clamp_min(0)
    batch_idx = torch.arange(b, device=q.device).view(b, 1, 1).expand_as(safe_idxs)
    gathered = kv[batch_idx, safe_idxs]
    valid = topk_idxs >= 0

    scores = torch.einsum("bshd,bstd->bsht", q.to(torch.float32), gathered.to(torch.float32)) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink = attn_sink.to(torch.float32).view(1, 1, h, 1).expand(b, s, h, 1)
    weights = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :topk]
    out = torch.einsum("bsht,bstd->bshd", weights, gathered.to(torch.float32))
    return out.to(q.dtype)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    mix_hc = (2 + hc_mult) * hc_mult
    flat = mixes.view(-1, mix_hc).to(torch.float32)
    base = hc_base.to(torch.float32)
    scale = hc_scale.to(torch.float32)

    pre = torch.sigmoid(flat[:, :hc_mult] * scale[0] + base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(flat[:, hc_mult : 2 * hc_mult] * scale[1] + base[hc_mult : 2 * hc_mult])

    comb_start = 2 * hc_mult
    comb = flat[:, comb_start:] * scale[2] + base[comb_start:]
    comb = comb.view(-1, hc_mult, hc_mult)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    shape = mixes.shape[:-1]
    return pre.view(*shape, hc_mult), post.view(*shape, hc_mult), comb.view(*shape, hc_mult, hc_mult)
