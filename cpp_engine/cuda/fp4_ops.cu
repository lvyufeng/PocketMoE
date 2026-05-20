#include "cuda_ops.hpp"

#include <cuda_runtime.h>
#include <sm_61_intrinsics.h>

namespace dsv4 {
namespace {

constexpr int kQuantThreads = 256;
constexpr int kGemmThreads = 128;

__host__ __device__ __forceinline__ int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ int fp4_unpack_4codes_prmt(uint32_t two_bytes) {
    const uint32_t control_mags = two_bytes & 0x7777u;
    const uint32_t pos = __byte_perm(0x03020100u, 0x0c080604u, control_mags);
    const uint32_t neg = __byte_perm(0xfdfeff00u, 0xf4f8fafcu, control_mags);
    const uint32_t control_sign = (two_bytes >> 3) & 0x1111u;
    const uint32_t mask = __byte_perm(0x0000ff00u, 0x00000000u, control_sign);
    return static_cast<int>((pos & ~mask) | (neg & mask));
}

__device__ __forceinline__ int dot_i8x4(int a, int b, int acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, acc);
#else
    const int8_t a0 = static_cast<int8_t>(a & 0xff);
    const int8_t a1 = static_cast<int8_t>((a >> 8) & 0xff);
    const int8_t a2 = static_cast<int8_t>((a >> 16) & 0xff);
    const int8_t a3 = static_cast<int8_t>((a >> 24) & 0xff);
    const int8_t b0 = static_cast<int8_t>(b & 0xff);
    const int8_t b1 = static_cast<int8_t>((b >> 8) & 0xff);
    const int8_t b2 = static_cast<int8_t>((b >> 16) & 0xff);
    const int8_t b3 = static_cast<int8_t>((b >> 24) & 0xff);
    return acc + static_cast<int>(a0) * b0 + static_cast<int>(a1) * b1 + static_cast<int>(a2) * b2 + static_cast<int>(a3) * b3;
#endif
}

__device__ __forceinline__ float fp4_block_scale(uint8_t scale_byte) {
    const int exponent = max(0, static_cast<int>(scale_byte) - 1);
    return __int_as_float(exponent << 23);
}

__global__ void quantize_float_row_kernel(
    const float* __restrict__ x,
    int8_t* __restrict__ x_q,
    float* __restrict__ x_scale,
    int k) {
    const int tid = threadIdx.x;
    __shared__ float sdata[kQuantThreads];
    float local_max = 0.0f;
    for (int idx = tid; idx < k; idx += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(x[idx]));
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    const float scale = fmaxf(sdata[0], 1.0e-6f) / 127.0f;
    if (tid == 0) x_scale[0] = scale;
    __syncthreads();
    const float inv_scale = 1.0f / scale;
    for (int idx = tid; idx < k; idx += blockDim.x) {
        int q = __float2int_rn(x[idx] * inv_scale);
        q = max(-127, min(127, q));
        x_q[idx] = static_cast<int8_t>(q);
    }
}

__device__ __forceinline__ float fp4_e2m1_value(uint8_t code) {
    constexpr float table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return table[code & 0x0f];
}

__device__ __forceinline__ float e8m0_value(uint8_t code) {
    return exp2f(static_cast<float>(static_cast<int>(code) - 127));
}

__global__ void fp4_e2m1_e8m0_matvec_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale,
    float* __restrict__ y,
    int rows,
    int cols,
    int packed_cols,
    int scale_cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const uint8_t packed = weight[static_cast<size_t>(row) * packed_cols + col / 2];
        const uint8_t code = (col & 1) ? static_cast<uint8_t>(packed >> 4) : static_cast<uint8_t>(packed & 0x0f);
        const float s = e8m0_value(scale[static_cast<size_t>(row) * scale_cols + col / 32]);
        sum += fp4_e2m1_value(code) * s * x[col];
    }

    extern __shared__ float scratch[];
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) y[row] = scratch[0];
}

__global__ void moe_single_w1w3_fp4_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const int64_t* __restrict__ indices,
    const uint8_t* __restrict__ w1q,
    const uint8_t* __restrict__ w1s,
    const uint8_t* __restrict__ w3q,
    const uint8_t* __restrict__ w3s,
    float* __restrict__ gate_f32,
    float* __restrict__ up_f32,
    int topk,
    int experts_start_idx,
    int n_local_experts,
    int dim,
    int inter_dim) {
    const int route = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= topk) return;
    const int local = static_cast<int>(indices[route]) - experts_start_idx;
    const int dim_packs = dim / 4;
    const int blocks_k = dim / 32;
    extern __shared__ int x_shared[];
    const int* x_i32 = reinterpret_cast<const int*>(x_q);
    for (int idx = threadIdx.x; idx < dim_packs; idx += blockDim.x) x_shared[idx] = x_i32[idx];
    __syncthreads();
    if (col >= inter_dim) return;
    if (local < 0 || local >= n_local_experts) {
        gate_f32[route * inter_dim + col] = 0.0f;
        up_f32[route * inter_dim + col] = 0.0f;
        return;
    }
    const uint8_t* w1_row_bytes = w1q + (static_cast<int64_t>(local) * inter_dim + col) * (dim / 2);
    const uint8_t* w3_row_bytes = w3q + (static_cast<int64_t>(local) * inter_dim + col) * (dim / 2);
    const uint8_t* w1_scale_row = w1s + (static_cast<int64_t>(local) * inter_dim + col) * blocks_k;
    const uint8_t* w3_scale_row = w3s + (static_cast<int64_t>(local) * inter_dim + col) * blocks_k;
    const uint16_t* w1_pack_base = reinterpret_cast<const uint16_t*>(w1_row_bytes);
    const uint16_t* w3_pack_base = reinterpret_cast<const uint16_t*>(w3_row_bytes);
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int kb = 0; kb < blocks_k; ++kb) {
        const uint16_t* w1_pack = w1_pack_base + kb * 8;
        const uint16_t* w3_pack = w3_pack_base + kb * 8;
        const int* x_pack = x_shared + kb * 8;
        int gate_block = 0;
        int up_block = 0;
        #pragma unroll
        for (int ip = 0; ip < 8; ++ip) {
            const int w1_p = fp4_unpack_4codes_prmt(static_cast<uint32_t>(w1_pack[ip]));
            const int w3_p = fp4_unpack_4codes_prmt(static_cast<uint32_t>(w3_pack[ip]));
            const int x_p = x_pack[ip];
            gate_block = dot_i8x4(x_p, w1_p, gate_block);
            up_block = dot_i8x4(x_p, w3_p, up_block);
        }
        gate_acc += static_cast<float>(gate_block) * fp4_block_scale(w1_scale_row[kb]);
        up_acc += static_cast<float>(up_block) * fp4_block_scale(w3_scale_row[kb]);
    }
    const float xs = x_scale[0];
    gate_f32[route * inter_dim + col] = gate_acc * xs;
    up_f32[route * inter_dim + col] = up_acc * xs;
}

__global__ void moe_single_swiglu_quant_fp4_kernel(
    const float* __restrict__ gate_f32,
    const float* __restrict__ up_f32,
    const int64_t* __restrict__ indices,
    const float* __restrict__ weights,
    int8_t* __restrict__ hidden_q,
    float* __restrict__ hidden_scale,
    int topk,
    int experts_start_idx,
    int n_local_experts,
    int inter_dim,
    float swiglu_limit) {
    const int route = blockIdx.x;
    const int tid = threadIdx.x;
    if (route >= topk) return;
    const int local = static_cast<int>(indices[route]) - experts_start_idx;
    if (local < 0 || local >= n_local_experts) {
        if (tid == 0) hidden_scale[route] = 0.0f;
        return;
    }
    __shared__ float sdata[kQuantThreads];
    float local_max = 0.0f;
    const float route_weight = weights[route];
    for (int col = tid; col < inter_dim; col += blockDim.x) {
        float gate = gate_f32[route * inter_dim + col];
        float up = up_f32[route * inter_dim + col];
        if (swiglu_limit > 0.0f) {
            up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
            gate = fminf(gate, swiglu_limit);
        }
        const float silu = gate / (1.0f + expf(-gate));
        const float v = silu * up * route_weight;
        local_max = fmaxf(local_max, fabsf(v));
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    const float scale = fmaxf(sdata[0], 1.0e-6f) / 127.0f;
    if (tid == 0) hidden_scale[route] = scale;
    __syncthreads();
    const float inv_scale = 1.0f / scale;
    for (int col = tid; col < inter_dim; col += blockDim.x) {
        float gate = gate_f32[route * inter_dim + col];
        float up = up_f32[route * inter_dim + col];
        if (swiglu_limit > 0.0f) {
            up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
            gate = fminf(gate, swiglu_limit);
        }
        const float silu = gate / (1.0f + expf(-gate));
        const float v = silu * up * route_weight;
        int q = __float2int_rn(v * inv_scale);
        q = max(-127, min(127, q));
        hidden_q[route * inter_dim + col] = static_cast<int8_t>(q);
    }
}

__global__ void moe_single_w2_accum_fp4_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ indices,
    const uint8_t* __restrict__ w2q,
    const uint8_t* __restrict__ w2s,
    float* __restrict__ y,
    int topk,
    int experts_start_idx,
    int n_local_experts,
    int dim,
    int inter_dim) {
    const int route = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= topk) return;
    const int local = static_cast<int>(indices[route]) - experts_start_idx;
    if (local < 0 || local >= n_local_experts) return;
    const int packs = inter_dim / 4;
    const int blocks_k = inter_dim / 32;
    extern __shared__ int h_shared[];
    const int* h_i32 = reinterpret_cast<const int*>(hidden_q + route * inter_dim);
    for (int idx = threadIdx.x; idx < packs; idx += blockDim.x) h_shared[idx] = h_i32[idx];
    __syncthreads();
    if (col >= dim) return;
    const uint8_t* w2_row_bytes = w2q + (static_cast<int64_t>(local) * dim + col) * (inter_dim / 2);
    const uint8_t* w2_scale_row = w2s + (static_cast<int64_t>(local) * dim + col) * blocks_k;
    const uint16_t* w2_pack_base = reinterpret_cast<const uint16_t*>(w2_row_bytes);
    float row_acc = 0.0f;
    for (int kb = 0; kb < blocks_k; ++kb) {
        const uint16_t* w2_pack = w2_pack_base + kb * 8;
        const int* h_pack = h_shared + kb * 8;
        int blk = 0;
        #pragma unroll
        for (int ip = 0; ip < 8; ++ip) {
            const int w_p = fp4_unpack_4codes_prmt(static_cast<uint32_t>(w2_pack[ip]));
            blk = dot_i8x4(h_pack[ip], w_p, blk);
        }
        row_acc += static_cast<float>(blk) * fp4_block_scale(w2_scale_row[kb]);
    }
    atomicAdd(y + col, row_acc * hidden_scale[route]);
}

}  // namespace

bool fp4_e2m1_e8m0_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int rows,
    int cols,
    void* stream) {
    if (d_x == nullptr || d_weight == nullptr || d_scale == nullptr || d_y == nullptr) return false;
    if (rows <= 0 || cols <= 0 || (cols % 32) != 0) return false;
    const int threads = 256;
    const int packed_cols = cols / 2;
    const int scale_cols = cols / 32;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fp4_e2m1_e8m0_matvec_kernel<<<rows, threads, threads * sizeof(float), cuda_stream>>>(
        d_x, d_weight, d_scale, d_y, rows, cols, packed_cols, scale_cols);
    return cudaGetLastError() == cudaSuccess;
}

bool moe_single_token_fp4_cuda(
    const float* d_x,
    const int64_t* d_indices,
    const float* d_weights,
    const uint8_t* d_w1q,
    const uint8_t* d_w1s,
    const uint8_t* d_w2q,
    const uint8_t* d_w2s,
    const uint8_t* d_w3q,
    const uint8_t* d_w3s,
    float* d_y,
    int topk,
    int experts_start_idx,
    int n_local_experts,
    int dim,
    int inter_dim,
    float swiglu_limit,
    void* stream) {
    if (d_x == nullptr || d_indices == nullptr || d_weights == nullptr || d_w1q == nullptr || d_w1s == nullptr || d_w2q == nullptr || d_w2s == nullptr || d_w3q == nullptr || d_w3s == nullptr || d_y == nullptr) return false;
    if (topk <= 0 || n_local_experts <= 0 || dim <= 0 || inter_dim <= 0 || (dim % 32) != 0 || (inter_dim % 32) != 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    int8_t* d_x_q = nullptr;
    float* d_x_scale = nullptr;
    float* d_gate = nullptr;
    float* d_up = nullptr;
    int8_t* d_hidden_q = nullptr;
    float* d_hidden_scale = nullptr;
    if (cudaMalloc(&d_x_q, static_cast<size_t>(dim)) != cudaSuccess) return false;
    if (cudaMalloc(&d_x_scale, sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&d_gate, static_cast<size_t>(topk) * inter_dim * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&d_up, static_cast<size_t>(topk) * inter_dim * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&d_hidden_q, static_cast<size_t>(topk) * inter_dim) != cudaSuccess) return false;
    if (cudaMalloc(&d_hidden_scale, static_cast<size_t>(topk) * sizeof(float)) != cudaSuccess) return false;
    cudaMemsetAsync(d_y, 0, static_cast<size_t>(dim) * sizeof(float), cuda_stream);
    quantize_float_row_kernel<<<1, kQuantThreads, 0, cuda_stream>>>(d_x, d_x_q, d_x_scale, dim);
    const dim3 w1w3_grid(ceil_div_int(inter_dim, kGemmThreads), topk);
    const dim3 gemm_block(kGemmThreads);
    moe_single_w1w3_fp4_kernel<<<w1w3_grid, gemm_block, static_cast<size_t>(dim / 4) * sizeof(int), cuda_stream>>>(
        d_x_q, d_x_scale, d_indices, d_w1q, d_w1s, d_w3q, d_w3s, d_gate, d_up, topk, experts_start_idx, n_local_experts, dim, inter_dim);
    moe_single_swiglu_quant_fp4_kernel<<<topk, kQuantThreads, 0, cuda_stream>>>(
        d_gate, d_up, d_indices, d_weights, d_hidden_q, d_hidden_scale, topk, experts_start_idx, n_local_experts, inter_dim, swiglu_limit);
    const dim3 w2_grid(ceil_div_int(dim, kGemmThreads), topk);
    moe_single_w2_accum_fp4_kernel<<<w2_grid, gemm_block, static_cast<size_t>(inter_dim / 4) * sizeof(int), cuda_stream>>>(
        d_hidden_q, d_hidden_scale, d_indices, d_w2q, d_w2s, d_y, topk, experts_start_idx, n_local_experts, dim, inter_dim);
    const cudaError_t launch_err = cudaGetLastError();
    cudaFree(d_x_q);
    cudaFree(d_x_scale);
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_hidden_q);
    cudaFree(d_hidden_scale);
    return launch_err == cudaSuccess;
}

}  // namespace dsv4
