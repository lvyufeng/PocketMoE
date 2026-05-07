#include <torch/extension.h>

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kQuantThreads = 256;
constexpr int kGemmThreads = 128;
constexpr int kAttnThreads = 256;

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template <typename scalar_t>
__global__ void hc_post_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ residual,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    scalar_t* __restrict__ y,
    int tokens,
    int dim) {
    const int token = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    if (token >= tokens || h >= 4) return;
    const scalar_t* x_base = x + token * dim;
    const scalar_t* residual_base = residual + token * 4 * dim;
    scalar_t* y_base = y + (token * 4 + h) * dim;
    const float post_h = post[token * 4 + h];
    const float* comb_base = comb + token * 16;
    for (int d = tid; d < dim; d += blockDim.x) {
        float acc = post_h * static_cast<float>(x_base[d]);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc += comb_base[i * 4 + h] * static_cast<float>(residual_base[i * dim + d]);
        }
        y_base[d] = static_cast<scalar_t>(acc);
    }
}

template <typename scalar_t>
__global__ void hc_split_pre_kernel(
    const float* __restrict__ mixes,
    const scalar_t* __restrict__ x,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    scalar_t* __restrict__ y,
    float* __restrict__ pre_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    int tokens,
    int dim,
    int sinkhorn_iters,
    float eps) {
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    if (token >= tokens) return;

    const float* m = mixes + token * 24;
    __shared__ float pre[4];
    __shared__ float post[4];
    __shared__ float comb[16];

    if (tid < 4) {
        pre[tid] = sigmoidf_fast(m[tid] * hc_scale[0] + hc_base[tid]) + eps;
        post[tid] = 2.0f * sigmoidf_fast(m[4 + tid] * hc_scale[1] + hc_base[4 + tid]);
    }
    if (tid < 16) {
        float vals[4];
        const int row = tid / 4;
        const int col = tid - row * 4;
        const float* cm = m + 8 + row * 4;
        float max_v = cm[0] * hc_scale[2] + hc_base[8 + row * 4];
        #pragma unroll
        for (int c = 1; c < 4; ++c) {
            max_v = fmaxf(max_v, cm[c] * hc_scale[2] + hc_base[8 + row * 4 + c]);
        }
        float sum = 0.0f;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            vals[c] = expf(cm[c] * hc_scale[2] + hc_base[8 + row * 4 + c] - max_v);
            sum += vals[c];
        }
        comb[tid] = vals[col] / sum + eps;
    }
    __syncthreads();

    if (tid < 4) {
        float col_sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < 4; ++r) col_sum += comb[r * 4 + tid];
        #pragma unroll
        for (int r = 0; r < 4; ++r) comb[r * 4 + tid] = comb[r * 4 + tid] / (col_sum + eps);
    }
    __syncthreads();

    for (int iter = 0; iter < sinkhorn_iters - 1; ++iter) {
        if (tid < 4) {
            float row_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < 4; ++c) row_sum += comb[tid * 4 + c];
            #pragma unroll
            for (int c = 0; c < 4; ++c) comb[tid * 4 + c] = comb[tid * 4 + c] / (row_sum + eps);
        }
        __syncthreads();
        if (tid < 4) {
            float col_sum = 0.0f;
            #pragma unroll
            for (int r = 0; r < 4; ++r) col_sum += comb[r * 4 + tid];
            #pragma unroll
            for (int r = 0; r < 4; ++r) comb[r * 4 + tid] = comb[r * 4 + tid] / (col_sum + eps);
        }
        __syncthreads();
    }

    if (tid < 4) {
        pre_out[token * 4 + tid] = pre[tid];
        post_out[token * 4 + tid] = post[tid];
    }
    if (tid < 16) comb_out[token * 16 + tid] = comb[tid];

    const scalar_t* x_base = x + token * 4 * dim;
    scalar_t* y_base = y + token * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        #pragma unroll
        for (int h = 0; h < 4; ++h) {
            acc += pre[h] * static_cast<float>(x_base[h * dim + d]);
        }
        y_base[d] = static_cast<scalar_t>(acc);
    }
}

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__global__ void route_swiglu_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ route_weights,
    float* __restrict__ out,
    int rows,
    int cols,
    float swiglu_limit) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;
    const int row = idx / cols;
    float g = gate[idx];
    float u = up[idx];
    if (swiglu_limit > 0.0f) {
        u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
        g = fminf(g, swiglu_limit);
    }
    const float silu = g / (1.0f + expf(-g));
    out[idx] = silu * u * route_weights[row];
}
template <typename scalar_t>
__global__ void gather_routes_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ route_tokens,
    scalar_t* __restrict__ x_sorted,
    int routes,
    int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = routes * dim;
    if (idx >= total) return;
    const int route = idx / dim;
    const int d = idx - route * dim;
    const int64_t token = route_tokens[route];
    x_sorted[idx] = x[token * dim + d];
}

template <typename scalar_t>
__global__ void scatter_routes_kernel(
    const float* __restrict__ y_sorted,
    const int64_t* __restrict__ route_tokens,
    scalar_t* __restrict__ y,
    int routes,
    int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = routes * dim;
    if (idx >= total) return;
    const int route = idx / dim;
    const int d = idx - route * dim;
    const int64_t token = route_tokens[route];
    atomicAdd(y + token * dim + d, static_cast<scalar_t>(y_sorted[idx]));
}

template <typename index_t>
__global__ void moe_route_count_kernel(
    const index_t* __restrict__ indices,
    int32_t* __restrict__ counts,
    int tokens,
    int topk,
    int experts_start_idx,
    int n_local_experts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * topk;
    if (idx >= total) return;
    const int expert = static_cast<int>(indices[idx]);
    const int local = expert - experts_start_idx;
    if (local >= 0 && local < n_local_experts) {
        atomicAdd(counts + local, 1);
    }
}

__global__ void moe_route_prefix_kernel(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ seg_starts,
    int32_t* __restrict__ offsets,
    int32_t* __restrict__ total_routes,
    int n_local_experts) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int32_t total = 0;
    seg_starts[0] = 0;
    for (int e = 0; e < n_local_experts; ++e) {
        offsets[e] = total;
        total += counts[e];
        seg_starts[e + 1] = total;
    }
    total_routes[0] = total;
}

template <typename index_t>
__global__ void moe_route_fill_kernel(
    const index_t* __restrict__ indices,
    const float* __restrict__ weights,
    int32_t* __restrict__ offsets,
    int64_t* __restrict__ route_tokens,
    float* __restrict__ route_weights,
    int tokens,
    int topk,
    int experts_start_idx,
    int n_local_experts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * topk;
    if (idx >= total) return;
    const int expert = static_cast<int>(indices[idx]);
    const int local = expert - experts_start_idx;
    if (local < 0 || local >= n_local_experts) return;
    const int token = idx / topk;
    const int out = atomicAdd(offsets + local, 1);
    route_tokens[out] = static_cast<int64_t>(token);
    route_weights[out] = weights[idx];
}

template <typename scalar_t>
__global__ void quantize_rows_kernel(
    const scalar_t* __restrict__ x,
    int8_t* __restrict__ x_q,
    float* __restrict__ x_scale,
    int rows,
    int groups,
    int k) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    const int tid = threadIdx.x;
    const int base = (row * groups + group) * k;

    __shared__ float sdata[kQuantThreads];
    float local_max = 0.0f;
    for (int idx = tid; idx < k; idx += blockDim.x) {
        const float v = static_cast<float>(x[base + idx]);
        local_max = fmaxf(local_max, fabsf(v));
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    const float scale = fmaxf(sdata[0], 1.0e-6f) / 127.0f;
    if (tid == 0) {
        x_scale[row * groups + group] = scale;
    }
    __syncthreads();

    const float inv_scale = 1.0f / scale;
    for (int idx = tid; idx < k; idx += blockDim.x) {
        const float v = static_cast<float>(x[base + idx]);
        int q = __float2int_rn(v * inv_scale);
        q = max(-127, min(127, q));
        x_q[base + idx] = static_cast<int8_t>(q);
    }
}

__global__ void int8_gemm_rows_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ weight_q,
    const float* __restrict__ weight_s,
    float* __restrict__ out,
    int rows,
    int groups,
    int n,
    int k) {
    const int row = blockIdx.y;
    const int group = blockIdx.z;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int packs = k / 4;

    extern __shared__ int x_shared[];
    const int8_t* x_row = x_q + (row * groups + group) * k;
    const int* x_row_i32 = reinterpret_cast<const int*>(x_row);
    for (int idx = threadIdx.x; idx < packs; idx += blockDim.x) {
        x_shared[idx] = x_row_i32[idx];
    }
    __syncthreads();

    if (col >= n) {
        return;
    }

    const int8_t* w_row = weight_q + (group * n + col) * k;
    const int* w_row_i32 = reinterpret_cast<const int*>(w_row);
    int acc = 0;
    for (int idx = 0; idx < packs; ++idx) {
        acc = __dp4a(x_shared[idx], w_row_i32[idx], acc);
    }

    const float scale = x_scale[row * groups + group] * weight_s[group * n + col];
    out[(row * groups + group) * n + col] = static_cast<float>(acc) * scale;
}
__global__ void moe_single_w1w3_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const int64_t* __restrict__ indices,
    const int8_t* __restrict__ w1q,
    const int8_t* __restrict__ w3q,
    int32_t* __restrict__ gate_i32,
    int32_t* __restrict__ up_i32,
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
    const int packs = dim / 4;
    extern __shared__ int x_shared[];
    const int* x_i32 = reinterpret_cast<const int*>(x_q);
    for (int idx = threadIdx.x; idx < packs; idx += blockDim.x) {
        x_shared[idx] = x_i32[idx];
    }
    __syncthreads();
    if (col >= inter_dim) return;
    const int8_t* w1_row = w1q + (static_cast<int64_t>(local) * inter_dim + col) * dim;
    const int8_t* w3_row = w3q + (static_cast<int64_t>(local) * inter_dim + col) * dim;
    const int* w1_i32 = reinterpret_cast<const int*>(w1_row);
    const int* w3_i32 = reinterpret_cast<const int*>(w3_row);
    int acc1 = 0;
    int acc3 = 0;
    for (int idx = 0; idx < packs; ++idx) {
        const int xv = x_shared[idx];
        acc1 = __dp4a(xv, w1_i32[idx], acc1);
        acc3 = __dp4a(xv, w3_i32[idx], acc3);
    }
    gate_i32[route * inter_dim + col] = acc1;
    up_i32[route * inter_dim + col] = acc3;
}

__global__ void moe_single_swiglu_quant_kernel(
    const int32_t* __restrict__ gate_i32,
    const int32_t* __restrict__ up_i32,
    const float* __restrict__ x_scale,
    const int64_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ w1s,
    const float* __restrict__ w3s,
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
    const float xs = x_scale[0];
    const float route_weight = weights[route];
    for (int col = tid; col < inter_dim; col += blockDim.x) {
        float gate = static_cast<float>(gate_i32[route * inter_dim + col]) * xs * w1s[local * inter_dim + col];
        float up = static_cast<float>(up_i32[route * inter_dim + col]) * xs * w3s[local * inter_dim + col];
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
        float gate = static_cast<float>(gate_i32[route * inter_dim + col]) * xs * w1s[local * inter_dim + col];
        float up = static_cast<float>(up_i32[route * inter_dim + col]) * xs * w3s[local * inter_dim + col];
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

__global__ void moe_single_w2_accum_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ indices,
    const int8_t* __restrict__ w2q,
    const float* __restrict__ w2s,
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
    extern __shared__ int h_shared[];
    const int* h_i32 = reinterpret_cast<const int*>(hidden_q + route * inter_dim);
    for (int idx = threadIdx.x; idx < packs; idx += blockDim.x) {
        h_shared[idx] = h_i32[idx];
    }
    __syncthreads();
    if (col >= dim) return;
    const int8_t* w2_row = w2q + (static_cast<int64_t>(local) * dim + col) * inter_dim;
    const int* w2_i32 = reinterpret_cast<const int*>(w2_row);
    int acc = 0;
    for (int idx = 0; idx < packs; ++idx) {
        acc = __dp4a(h_shared[idx], w2_i32[idx], acc);
    }
    const float value = static_cast<float>(acc) * hidden_scale[route] * w2s[local * dim + col];
    atomicAdd(y + col, value);
}


__global__ void moe_swiglu_hidden_grouped_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const int32_t* __restrict__ seg_starts,
    const float* __restrict__ route_weights,
    const int8_t* __restrict__ w1q,
    const float* __restrict__ w1s,
    const int8_t* __restrict__ w3q,
    const float* __restrict__ w3s,
    float* __restrict__ hidden,
    int n_experts,
    int dim,
    int inter_dim,
    int max_count,
    float swiglu_limit) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_in_expert = blockIdx.y;
    const int expert = blockIdx.z;
    if (expert >= n_experts) return;
    const int start = seg_starts[expert];
    const int count = seg_starts[expert + 1] - start;
    if (row_in_expert >= count || row_in_expert >= max_count) return;
    const int route = start + row_in_expert;
    const int packs = dim / 4;

    extern __shared__ int x_shared[];
    const int* x_row_i32 = reinterpret_cast<const int*>(x_q + route * dim);
    for (int idx = threadIdx.x; idx < packs; idx += blockDim.x) {
        x_shared[idx] = x_row_i32[idx];
    }
    __syncthreads();

    if (col >= inter_dim) return;
    const int8_t* w1_row = w1q + (expert * inter_dim + col) * dim;
    const int8_t* w3_row = w3q + (expert * inter_dim + col) * dim;
    const int* w1_i32 = reinterpret_cast<const int*>(w1_row);
    const int* w3_i32 = reinterpret_cast<const int*>(w3_row);
    int acc1 = 0;
    int acc3 = 0;
    for (int idx = 0; idx < packs; ++idx) {
        const int xv = x_shared[idx];
        acc1 = __dp4a(xv, w1_i32[idx], acc1);
        acc3 = __dp4a(xv, w3_i32[idx], acc3);
    }
    float gate = static_cast<float>(acc1) * x_scale[route] * w1s[expert * inter_dim + col];
    float up = static_cast<float>(acc3) * x_scale[route] * w3s[expert * inter_dim + col];
    if (swiglu_limit > 0.0f) {
        up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
        gate = fminf(gate, swiglu_limit);
    }
    const float silu = gate / (1.0f + expf(-gate));
    hidden[route * inter_dim + col] = silu * up * route_weights[route];
}

__global__ void moe_w2_scatter_grouped_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const int8_t* __restrict__ w2q,
    const float* __restrict__ w2s,
    float* __restrict__ y,
    int n_experts,
    int dim,
    int inter_dim,
    int max_count) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_in_expert = blockIdx.y;
    const int expert = blockIdx.z;
    if (expert >= n_experts) return;
    const int start = seg_starts[expert];
    const int count = seg_starts[expert + 1] - start;
    if (row_in_expert >= count || row_in_expert >= max_count) return;
    const int route = start + row_in_expert;
    const int packs = inter_dim / 4;

    extern __shared__ int h_shared[];
    const int* h_row_i32 = reinterpret_cast<const int*>(hidden_q + route * inter_dim);
    for (int idx = threadIdx.x; idx < packs; idx += blockDim.x) {
        h_shared[idx] = h_row_i32[idx];
    }
    __syncthreads();

    if (col >= dim) return;
    const int8_t* w2_row = w2q + (expert * dim + col) * inter_dim;
    const int* w2_i32 = reinterpret_cast<const int*>(w2_row);
    int acc = 0;
    for (int idx = 0; idx < packs; ++idx) {
        acc = __dp4a(h_shared[idx], w2_i32[idx], acc);
    }
    const float value = static_cast<float>(acc) * hidden_scale[route] * w2s[expert * dim + col];
    const int64_t token = route_tokens[route];
    atomicAdd(y + token * dim + col, value);
}

__global__ void moe_pad_q_rows_kernel(
    const int8_t* __restrict__ src_q,
    const float* __restrict__ src_scale,
    const int32_t* __restrict__ seg_starts,
    int8_t* __restrict__ dst_q,
    float* __restrict__ dst_scale,
    int n_experts,
    int rows_per_expert,
    int k) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_experts * rows_per_expert * k;
    if (idx >= total) return;
    const int col = idx % k;
    const int row = (idx / k) % rows_per_expert;
    const int expert = idx / (rows_per_expert * k);
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    const int dst = (expert * rows_per_expert + row) * k + col;
    if (row < count) {
        const int route = seg_starts[expert] + row;
        dst_q[dst] = src_q[route * k + col];
        if (col == 0) dst_scale[expert * rows_per_expert + row] = src_scale[route];
    } else {
        dst_q[dst] = 0;
        if (col == 0) dst_scale[expert * rows_per_expert + row] = 0.0f;
    }
}

__global__ void moe_pair_scale_swiglu_padded_kernel(
    const int32_t* __restrict__ gate_i32,
    const int32_t* __restrict__ up_i32,
    const float* __restrict__ x_scale,
    const int32_t* __restrict__ seg_starts,
    const float* __restrict__ route_weights,
    const float* __restrict__ w1s,
    const float* __restrict__ w3s,
    float* __restrict__ hidden,
    int n_experts,
    int rows_per_expert,
    int inter_dim,
    float swiglu_limit) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_experts * rows_per_expert * inter_dim;
    if (idx >= total) return;
    const int col = idx % inter_dim;
    const int row = (idx / inter_dim) % rows_per_expert;
    const int expert = idx / (rows_per_expert * inter_dim);
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    if (row >= count) return;
    const int route = seg_starts[expert] + row;
    const int base = (expert * rows_per_expert + row) * inter_dim + col;
    float gate = static_cast<float>(gate_i32[base]) * x_scale[expert * rows_per_expert + row] * w1s[expert * inter_dim + col];
    float up = static_cast<float>(up_i32[base]) * x_scale[expert * rows_per_expert + row] * w3s[expert * inter_dim + col];
    if (swiglu_limit > 0.0f) {
        up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
        gate = fminf(gate, swiglu_limit);
    }
    const float silu = gate / (1.0f + expf(-gate));
    hidden[route * inter_dim + col] = silu * up * route_weights[route];
}

__global__ void moe_pair_swiglu_quantize_padded_kernel(
    const int32_t* __restrict__ gate_i32,
    const int32_t* __restrict__ up_i32,
    const float* __restrict__ x_scale,
    const int32_t* __restrict__ seg_starts,
    const float* __restrict__ route_weights,
    const float* __restrict__ w1s,
    const float* __restrict__ w3s,
    int8_t* __restrict__ hidden_q,
    float* __restrict__ hidden_scale,
    int n_experts,
    int rows_per_expert,
    int inter_dim,
    float swiglu_limit) {
    const int expert = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    const int padded_row = expert * rows_per_expert + row;
    const int base = padded_row * inter_dim;
    __shared__ float sdata[kQuantThreads];
    if (row >= count) {
        for (int col = tid; col < inter_dim; col += blockDim.x) {
            hidden_q[base + col] = 0;
        }
        if (tid == 0) hidden_scale[padded_row] = 0.0f;
        return;
    }
    const int route = seg_starts[expert] + row;
    float local_max = 0.0f;
    for (int col = tid; col < inter_dim; col += blockDim.x) {
        const int idx = base + col;
        float gate = static_cast<float>(gate_i32[idx]) * x_scale[padded_row] * w1s[expert * inter_dim + col];
        float up = static_cast<float>(up_i32[idx]) * x_scale[padded_row] * w3s[expert * inter_dim + col];
        if (swiglu_limit > 0.0f) {
            up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
            gate = fminf(gate, swiglu_limit);
        }
        const float silu = gate / (1.0f + expf(-gate));
        const float value = silu * up * route_weights[route];
        local_max = fmaxf(local_max, fabsf(value));
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    const float scale = fmaxf(sdata[0], 1.0e-6f) / 127.0f;
    if (tid == 0) hidden_scale[padded_row] = scale;
    const float inv_scale = 1.0f / scale;
    for (int col = tid; col < inter_dim; col += blockDim.x) {
        const int idx = base + col;
        float gate = static_cast<float>(gate_i32[idx]) * x_scale[padded_row] * w1s[expert * inter_dim + col];
        float up = static_cast<float>(up_i32[idx]) * x_scale[padded_row] * w3s[expert * inter_dim + col];
        if (swiglu_limit > 0.0f) {
            up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
            gate = fminf(gate, swiglu_limit);
        }
        const float silu = gate / (1.0f + expf(-gate));
        const float value = silu * up * route_weights[route];
        int q = __float2int_rn(value * inv_scale);
        q = max(-127, min(127, q));
        hidden_q[idx] = static_cast<int8_t>(q);
    }
}

__global__ void moe_depad_scatter_i32_kernel(
    const int32_t* __restrict__ out_i32,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const float* __restrict__ w2s,
    float* __restrict__ y,
    int n_experts,
    int rows_per_expert,
    int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_experts * rows_per_expert * dim;
    if (idx >= total) return;
    const int col = idx % dim;
    const int row = (idx / dim) % rows_per_expert;
    const int expert = idx / (rows_per_expert * dim);
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    if (row >= count) return;
    const int route = seg_starts[expert] + row;
    const float value = static_cast<float>(out_i32[idx]) * hidden_scale[expert * rows_per_expert + row] * w2s[expert * dim + col];
    const int64_t token = route_tokens[route];
    atomicAdd(y + token * dim + col, value);
}

template <typename scalar_t>
__global__ void prefill_sparse_attn_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ kv,
    const float* __restrict__ attn_sink,
    const int32_t* __restrict__ topk_idxs,
    scalar_t* __restrict__ out,
    int bsz,
    int seqlen,
    int heads,
    int kv_len,
    int topk,
    int dim,
    float softmax_scale) {
    const int bsh = blockIdx.x;
    const int h = bsh % heads;
    const int s = (bsh / heads) % seqlen;
    const int b = bsh / (seqlen * heads);
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* scores = smem;
    float* q_shared = smem + topk;
    int32_t* idx_shared = reinterpret_cast<int32_t*>(q_shared + dim);
    float* reduce = reinterpret_cast<float*>(idx_shared + topk);

    const scalar_t* q_ptr = q + (((b * seqlen + s) * heads + h) * dim);
    const scalar_t* kv_base = kv + b * kv_len * dim;
    const int32_t* idx_base = topk_idxs + (b * seqlen + s) * topk;
    for (int d = tid; d < dim; d += blockDim.x) {
        q_shared[d] = static_cast<float>(q_ptr[d]);
    }
    for (int t = tid; t < topk; t += blockDim.x) {
        idx_shared[t] = idx_base[t];
    }
    __syncthreads();

    for (int t = tid; t < topk; t += blockDim.x) {
        const int idx = idx_shared[t];
        float score = -INFINITY;
        if (idx >= 0 && idx < kv_len) {
            float acc = 0.0f;
            const scalar_t* kv_ptr = kv_base + idx * dim;
            for (int d = 0; d < dim; ++d) {
                acc += q_shared[d] * static_cast<float>(kv_ptr[d]);
            }
            score = acc * softmax_scale;
        }
        scores[t] = score;
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int t = tid; t < topk; t += blockDim.x) {
        local_max = fmaxf(local_max, scores[t]);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
        __syncthreads();
    }
    const float max_score = fmaxf(reduce[0], attn_sink[h]);

    float local_denom = 0.0f;
    for (int t = tid; t < topk; t += blockDim.x) {
        const float w = expf(scores[t] - max_score);
        scores[t] = w;
        local_denom += w;
    }
    if (tid == 0) local_denom += expf(attn_sink[h] - max_score);
    reduce[tid] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    const float denom = reduce[0];

    scalar_t* out_ptr = out + (((b * seqlen + s) * heads + h) * dim);
    for (int d = tid; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < topk; ++t) {
            const int idx = idx_shared[t];
            if (idx >= 0 && idx < kv_len) {
                acc += scores[t] * static_cast<float>(kv_base[idx * dim + d]);
            }
        }
        out_ptr[d] = static_cast<scalar_t>(acc / denom);
    }
}

template <typename scalar_t>
__global__ void prefill_sparse_attn_headpair_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ kv,
    const float* __restrict__ attn_sink,
    const int32_t* __restrict__ topk_idxs,
    scalar_t* __restrict__ out,
    int bsz,
    int seqlen,
    int heads,
    int kv_len,
    int topk,
    int dim,
    float softmax_scale) {
    const int head_pairs = (heads + 1) / 2;
    const int bsp = blockIdx.x;
    const int pair = bsp % head_pairs;
    const int s = (bsp / head_pairs) % seqlen;
    const int b = bsp / (seqlen * head_pairs);
    const int h0 = pair * 2;
    const int h1 = h0 + 1;
    const bool has_h1 = h1 < heads;
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* scores0 = smem;
    float* scores1 = scores0 + topk;
    float* q0_shared = scores1 + topk;
    float* q1_shared = q0_shared + dim;
    int32_t* idx_shared = reinterpret_cast<int32_t*>(q1_shared + dim);
    float* reduce0 = reinterpret_cast<float*>(idx_shared + topk);
    float* reduce1 = reduce0 + blockDim.x;

    const scalar_t* q0_ptr = q + (((b * seqlen + s) * heads + h0) * dim);
    const scalar_t* q1_ptr = has_h1 ? q + (((b * seqlen + s) * heads + h1) * dim) : q0_ptr;
    const scalar_t* kv_base = kv + b * kv_len * dim;
    const int32_t* idx_base = topk_idxs + (b * seqlen + s) * topk;
    for (int d = tid; d < dim; d += blockDim.x) {
        q0_shared[d] = static_cast<float>(q0_ptr[d]);
        if (has_h1) q1_shared[d] = static_cast<float>(q1_ptr[d]);
    }
    for (int t = tid; t < topk; t += blockDim.x) {
        idx_shared[t] = idx_base[t];
    }
    __syncthreads();

    for (int t = tid; t < topk; t += blockDim.x) {
        const int idx = idx_shared[t];
        float score0 = -INFINITY;
        float score1 = -INFINITY;
        if (idx >= 0 && idx < kv_len) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            const scalar_t* kv_ptr = kv_base + idx * dim;
            for (int d = 0; d < dim; ++d) {
                const float v = static_cast<float>(kv_ptr[d]);
                acc0 += q0_shared[d] * v;
                if (has_h1) acc1 += q1_shared[d] * v;
            }
            score0 = acc0 * softmax_scale;
            if (has_h1) score1 = acc1 * softmax_scale;
        }
        scores0[t] = score0;
        if (has_h1) scores1[t] = score1;
    }
    __syncthreads();

    float local_max0 = -INFINITY;
    float local_max1 = -INFINITY;
    for (int t = tid; t < topk; t += blockDim.x) {
        local_max0 = fmaxf(local_max0, scores0[t]);
        if (has_h1) local_max1 = fmaxf(local_max1, scores1[t]);
    }
    reduce0[tid] = local_max0;
    if (has_h1) reduce1[tid] = local_max1;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce0[tid] = fmaxf(reduce0[tid], reduce0[tid + stride]);
            if (has_h1) reduce1[tid] = fmaxf(reduce1[tid], reduce1[tid + stride]);
        }
        __syncthreads();
    }
    const float max_score0 = fmaxf(reduce0[0], attn_sink[h0]);
    const float max_score1 = has_h1 ? fmaxf(reduce1[0], attn_sink[h1]) : -INFINITY;

    float local_denom0 = 0.0f;
    float local_denom1 = 0.0f;
    for (int t = tid; t < topk; t += blockDim.x) {
        const float w0 = expf(scores0[t] - max_score0);
        scores0[t] = w0;
        local_denom0 += w0;
        if (has_h1) {
            const float w1 = expf(scores1[t] - max_score1);
            scores1[t] = w1;
            local_denom1 += w1;
        }
    }
    if (tid == 0) {
        local_denom0 += expf(attn_sink[h0] - max_score0);
        if (has_h1) local_denom1 += expf(attn_sink[h1] - max_score1);
    }
    reduce0[tid] = local_denom0;
    if (has_h1) reduce1[tid] = local_denom1;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce0[tid] += reduce0[tid + stride];
            if (has_h1) reduce1[tid] += reduce1[tid + stride];
        }
        __syncthreads();
    }
    const float denom0 = reduce0[0];
    const float denom1 = has_h1 ? reduce1[0] : 1.0f;

    scalar_t* out0_ptr = out + (((b * seqlen + s) * heads + h0) * dim);
    scalar_t* out1_ptr = has_h1 ? out + (((b * seqlen + s) * heads + h1) * dim) : out0_ptr;
    for (int d = tid; d < dim; d += blockDim.x) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (int t = 0; t < topk; ++t) {
            const int idx = idx_shared[t];
            if (idx >= 0 && idx < kv_len) {
                const float v = static_cast<float>(kv_base[idx * dim + d]);
                acc0 += scores0[t] * v;
                if (has_h1) acc1 += scores1[t] * v;
            }
        }
        out0_ptr[d] = static_cast<scalar_t>(acc0 / denom0);
        if (has_h1) out1_ptr[d] = static_cast<scalar_t>(acc1 / denom1);
    }
}

template <typename scalar_t>
__global__ void fused_decode_sparse_attn_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ kv,
    const float* __restrict__ attn_sink,
    const int32_t* __restrict__ topk_idxs,
    scalar_t* __restrict__ out,
    int bsz,
    int heads,
    int kv_len,
    int topk,
    int dim,
    float softmax_scale) {
    const int bh = blockIdx.x;
    const int b = bh / heads;
    const int h = bh - b * heads;
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* scores = smem;
    float* q_shared = smem + topk;
    int32_t* idx_shared = reinterpret_cast<int32_t*>(q_shared + dim);
    float* reduce = reinterpret_cast<float*>(idx_shared + topk);

    const scalar_t* q_ptr = q + ((b * heads + h) * dim);
    const scalar_t* kv_base = kv + b * kv_len * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        q_shared[d] = static_cast<float>(q_ptr[d]);
    }
    for (int t = tid; t < topk; t += blockDim.x) {
        idx_shared[t] = topk_idxs[b * topk + t];
    }
    __syncthreads();

    for (int t = tid; t < topk; t += blockDim.x) {
        const int idx = idx_shared[t];
        float score = -INFINITY;
        if (idx >= 0 && idx < kv_len) {
            float acc = 0.0f;
            const scalar_t* kv_ptr = kv_base + idx * dim;
            for (int d = 0; d < dim; ++d) {
                acc += q_shared[d] * static_cast<float>(kv_ptr[d]);
            }
            score = acc * softmax_scale;
        }
        scores[t] = score;
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int t = tid; t < topk; t += blockDim.x) {
        local_max = fmaxf(local_max, scores[t]);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
        __syncthreads();
    }
    const float max_score = fmaxf(reduce[0], attn_sink[h]);

    float local_denom = 0.0f;
    for (int t = tid; t < topk; t += blockDim.x) {
        const float w = expf(scores[t] - max_score);
        scores[t] = w;
        local_denom += w;
    }
    if (tid == 0) local_denom += expf(attn_sink[h] - max_score);
    reduce[tid] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    const float denom = reduce[0];

    scalar_t* out_ptr = out + ((b * heads + h) * dim);
    for (int d = tid; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < topk; ++t) {
            const int idx = idx_shared[t];
            if (idx >= 0 && idx < kv_len) {
                acc += scores[t] * static_cast<float>(kv_base[idx * dim + d]);
            }
        }
        out_ptr[d] = static_cast<scalar_t>(acc / denom);
    }
}

template <typename scalar_t>
__global__ void fused_decode_sparse_attn_wmma_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ kv,
    const float* __restrict__ attn_sink,
    const int32_t* __restrict__ topk_idxs,
    scalar_t* __restrict__ out,
    int bsz,
    int heads,
    int kv_len,
    int topk,
    float softmax_scale) {
    using namespace nvcuda;
    constexpr int DIM = 512;
    constexpr int TILE = 16;
    const int bh = blockIdx.x;
    const int b = bh / heads;
    const int h = bh - b * heads;
    const int tid = threadIdx.x;

    extern __shared__ char smem_raw[];
    half* a_tile = reinterpret_cast<half*>(smem_raw);
    half* b_tile = a_tile + TILE * TILE;
    float* c_tile = reinterpret_cast<float*>(b_tile + TILE * TILE);
    float* scores = c_tile + TILE * TILE;
    float* reduce = scores + topk;

    const scalar_t* q_ptr = q + ((b * heads + h) * DIM);
    const scalar_t* kv_base = kv + b * kv_len * DIM;

    for (int t0 = 0; t0 < topk; t0 += TILE) {
        wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        for (int k0 = 0; k0 < DIM; k0 += TILE) {
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int row = i / TILE;
                const int col = i - row * TILE;
                a_tile[i] = row == 0 ? __float2half(static_cast<float>(q_ptr[k0 + col])) : __float2half(0.0f);
                const int t = t0 + col;
                const int idx = t < topk ? topk_idxs[b * topk + t] : -1;
                b_tile[row + col * TILE] = (idx >= 0 && idx < kv_len) ? __float2half(static_cast<float>(kv_base[idx * DIM + k0 + row])) : __float2half(0.0f);
            }
            __syncthreads();
            wmma::load_matrix_sync(a_frag, a_tile, TILE);
            wmma::load_matrix_sync(b_frag, b_tile, TILE);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads();
        }
        if (tid < TILE * TILE) {
            c_tile[tid] = 0.0f;
        }
        __syncthreads();
        wmma::store_matrix_sync(c_tile, c_frag, TILE, wmma::mem_row_major);
        __syncthreads();
        if (tid < TILE) {
            const int t = t0 + tid;
            if (t < topk) {
                const int idx = topk_idxs[b * topk + t];
                scores[t] = (idx >= 0 && idx < kv_len) ? c_tile[tid] * softmax_scale : -INFINITY;
            }
        }
        __syncthreads();
    }

    float local_max = -INFINITY;
    for (int t = tid; t < topk; t += blockDim.x) {
        local_max = fmaxf(local_max, scores[t]);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
        __syncthreads();
    }
    const float max_score = fmaxf(reduce[0], attn_sink[h]);

    float local_denom = 0.0f;
    for (int t = tid; t < topk; t += blockDim.x) {
        const float w = expf(scores[t] - max_score);
        scores[t] = w;
        local_denom += w;
    }
    if (tid == 0) local_denom += expf(attn_sink[h] - max_score);
    reduce[tid] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    const float denom = reduce[0];

    scalar_t* out_ptr = out + ((b * heads + h) * DIM);
    for (int d = tid; d < DIM; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < topk; ++t) {
            const int idx = topk_idxs[b * topk + t];
            if (idx >= 0 && idx < kv_len) {
                acc += scores[t] * static_cast<float>(kv_base[idx * DIM + d]);
            }
        }
        out_ptr[d] = static_cast<scalar_t>(acc / denom);
    }
}

template <typename scalar_t>
__global__ void flashinfer_style_sparse_attn_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ kv,
    const float* __restrict__ attn_sink,
    const int32_t* __restrict__ topk_idxs,
    scalar_t* __restrict__ out,
    int bsz,
    int heads,
    int kv_len,
    int topk,
    int dim,
    float softmax_scale) {
    const int bh = blockIdx.x;
    const int b = bh / heads;
    const int h = bh - b * heads;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int n_warps = blockDim.x >> 5;

    extern __shared__ float smem[];
    float* scores = smem;
    float* q_shared = scores + topk;
    int32_t* idx_shared = reinterpret_cast<int32_t*>(q_shared + dim);
    float* reduce = reinterpret_cast<float*>(idx_shared + topk);

    const scalar_t* q_ptr = q + ((b * heads + h) * dim);
    const scalar_t* kv_base = kv + b * kv_len * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        q_shared[d] = static_cast<float>(q_ptr[d]);
    }
    for (int t = tid; t < topk; t += blockDim.x) {
        idx_shared[t] = topk_idxs[b * topk + t];
    }
    __syncthreads();

    for (int t = warp_id; t < topk; t += n_warps) {
        const int idx = idx_shared[t];
        float acc = 0.0f;
        if (idx >= 0 && idx < kv_len) {
            const scalar_t* kv_ptr = kv_base + idx * dim;
            for (int d = lane; d < dim; d += 32) {
                acc += q_shared[d] * static_cast<float>(kv_ptr[d]);
            }
        } else {
            acc = -INFINITY;
        }
        if (idx >= 0 && idx < kv_len) {
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc += __shfl_down_sync(0xffffffff, acc, offset);
            }
            if (lane == 0) {
                scores[t] = acc * softmax_scale;
            }
        } else if (lane == 0) {
            scores[t] = -INFINITY;
        }
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int t = tid; t < topk; t += blockDim.x) {
        local_max = fmaxf(local_max, scores[t]);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
        __syncthreads();
    }
    const float max_score = fmaxf(reduce[0], attn_sink[h]);

    float local_denom = 0.0f;
    for (int t = tid; t < topk; t += blockDim.x) {
        const float w = expf(scores[t] - max_score);
        scores[t] = w;
        local_denom += w;
    }
    if (tid == 0) local_denom += expf(attn_sink[h] - max_score);
    reduce[tid] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    const float denom = reduce[0];

    scalar_t* out_ptr = out + ((b * heads + h) * dim);
    for (int d = tid; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < topk; ++t) {
            const int idx = idx_shared[t];
            if (idx >= 0 && idx < kv_len) {
                acc += scores[t] * static_cast<float>(kv_base[idx * dim + d]);
            }
        }
        out_ptr[d] = static_cast<scalar_t>(acc / denom);
    }
}

__device__ __forceinline__ float fp4_fake_quant_value(float x) {
    const float ax = fabsf(x);
    float mag;
    if (ax <= 0.25f) {
        mag = 0.0f;
    } else if (ax <= 0.75f) {
        mag = 0.5f;
    } else if (ax <= 1.25f) {
        mag = 1.0f;
    } else if (ax <= 1.75f) {
        mag = 1.5f;
    } else if (ax <= 2.5f) {
        mag = 2.0f;
    } else if (ax <= 3.5f) {
        mag = 3.0f;
    } else if (ax <= 5.0f) {
        mag = 4.0f;
    } else {
        mag = 6.0f;
    }
    return copysignf(mag, x);
}

__device__ __forceinline__ float fp4_pow2_scale(float amax) {
    const float min_scale = 1.1754943508222875e-38f;
    const float raw = fmaxf(amax / 6.0f, min_scale);
    return exp2f(ceilf(log2f(raw) - 1.0e-6f));
}

__device__ void hadamard128_inplace(float* x, int tid) {
    for (int stride = 1; stride < 128; stride <<= 1) {
        if (tid < 128 && (tid & stride) == 0) {
            const float a = x[tid];
            const float b = x[tid + stride];
            x[tid] = a + b;
            x[tid + stride] = a - b;
        }
        __syncthreads();
    }
    if (tid < 128) {
        x[tid] *= 0.08838834764831845f;
    }
    __syncthreads();
}

__device__ void fp4_fake_quant128_inplace(float* x, int tid) {
    __shared__ float scales[4];
    if (tid < 4) {
        float amax = 0.0f;
        const int base = tid * 32;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            amax = fmaxf(amax, fabsf(x[base + i]));
        }
        scales[tid] = fp4_pow2_scale(amax);
    }
    __syncthreads();
    if (tid < 128) {
        const float scale = scales[tid >> 5];
        const float y = fminf(fmaxf(x[tid] / scale, -6.0f), 6.0f);
        x[tid] = fp4_fake_quant_value(y) * scale;
    }
    __syncthreads();
}

template <typename scalar_t>
__global__ void c4_update_cache_kernel(
    const float* __restrict__ kv,
    const float* __restrict__ score,
    const float* __restrict__ ape,
    const float* __restrict__ norm_weight,
    const float* __restrict__ freqs,
    float* __restrict__ kv_state,
    float* __restrict__ score_state,
    scalar_t* __restrict__ kv_cache,
    int bsz,
    int max_cache_len,
    int start_pos,
    float norm_eps) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    constexpr int D = 128;
    constexpr int D2 = 256;
    constexpr int R = 4;
    constexpr int STATE = 8;
    __shared__ float comp[D];
    __shared__ float reduce[256];

    if (b >= bsz) return;
    const int pos = start_pos & 3;
    float* kv_state_b = kv_state + b * STATE * D2;
    float* score_state_b = score_state + b * STATE * D2;
    const float* kv_b = kv + b * D2;
    const float* score_b = score + b * D2;
    for (int d = tid; d < D2; d += blockDim.x) {
        kv_state_b[(R + pos) * D2 + d] = kv_b[d];
        score_state_b[(R + pos) * D2 + d] = score_b[d] + ape[pos * D2 + d];
    }
    __syncthreads();

    if (((start_pos + 1) & 3) != 0) {
        return;
    }

    if (tid < D) {
        float max_s = -INFINITY;
        float svals[STATE];
        float kvals[STATE];
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            const float s = score_state_b[i * D2 + tid];
            svals[i] = s;
            kvals[i] = kv_state_b[i * D2 + tid];
            max_s = fmaxf(max_s, s);
        }
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            const int slot = R + i;
            const float s = score_state_b[slot * D2 + D + tid];
            svals[slot] = s;
            kvals[slot] = kv_state_b[slot * D2 + D + tid];
            max_s = fmaxf(max_s, s);
        }
        float denom = 0.0f;
        float out = 0.0f;
        #pragma unroll
        for (int i = 0; i < STATE; ++i) {
            const float e = expf(svals[i] - max_s);
            denom += e;
            out += kvals[i] * e;
        }
        out /= denom;
        comp[tid] = out;
        reduce[tid] = out * out;
    } else if (tid < 256) {
        reduce[tid] = 0.0f;
    }
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    const float rms = rsqrtf(reduce[0] / static_cast<float>(D) + norm_eps);
    if (tid < D) {
        comp[tid] = static_cast<float>(static_cast<scalar_t>(comp[tid] * rms * norm_weight[tid]));
    }
    __syncthreads();

    if (tid < 32) {
        const int base = 64 + tid * 2;
        const float x0 = comp[base];
        const float x1 = comp[base + 1];
        const float c = freqs[tid * 2];
        const float s = freqs[tid * 2 + 1];
        comp[base] = x0 * c - x1 * s;
        comp[base + 1] = x0 * s + x1 * c;
    }
    __syncthreads();

    hadamard128_inplace(comp, tid);
    if (tid < D) {
        comp[tid] = static_cast<float>(static_cast<scalar_t>(comp[tid]));
    }
    __syncthreads();
    fp4_fake_quant128_inplace(comp, tid);

    const int cache_pos = start_pos >> 2;
    if (cache_pos < max_cache_len && tid < D) {
        kv_cache[(b * max_cache_len + cache_pos) * D + tid] = static_cast<scalar_t>(comp[tid]);
    }
    __syncthreads();

    for (int i = tid; i < R * D2; i += blockDim.x) {
        kv_state_b[i] = kv_state_b[R * D2 + i];
        score_state_b[i] = score_state_b[R * D2 + i];
    }
}

template <typename scalar_t>
__global__ void hadamard128_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    int rows) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float vec[128];
    if (row >= rows) return;
    if (tid < 128) {
        vec[tid] = static_cast<float>(x[row * 128 + tid]);
    }
    __syncthreads();
    hadamard128_inplace(vec, tid);
    if (tid < 128) {
        out[row * 128 + tid] = static_cast<scalar_t>(vec[tid]);
    }
}

template <typename scalar_t>
__global__ void c4_quant_q_kernel(
    const scalar_t* __restrict__ q,
    float* __restrict__ q_work,
    int rows) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float vec[128];
    if (row >= rows) return;
    if (tid < 128) {
        vec[tid] = static_cast<float>(q[row * 128 + tid]);
    }
    __syncthreads();
    hadamard128_inplace(vec, tid);
    if (tid < 128) {
        vec[tid] = static_cast<float>(static_cast<scalar_t>(vec[tid]));
    }
    __syncthreads();
    fp4_fake_quant128_inplace(vec, tid);
    if (tid < 128) {
        q_work[row * 128 + tid] = static_cast<float>(static_cast<scalar_t>(vec[tid]));
    }
}

template <typename score_t>
__global__ void c4_topk_kernel(
    score_t* __restrict__ scores,
    int64_t* __restrict__ topk,
    int bsz,
    int kv_len,
    int k,
    int offset) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= bsz) return;
    extern __shared__ char smem_raw[];
    float* best_vals = reinterpret_cast<float*>(smem_raw);
    int* best_idxs = reinterpret_cast<int*>(best_vals + blockDim.x);
    score_t* row = scores + b * kv_len;
    int64_t* out = topk + b * k;

    for (int out_i = 0; out_i < k; ++out_i) {
        float local_best = -INFINITY;
        int local_idx = kv_len;
        for (int t = tid; t < kv_len; t += blockDim.x) {
            const float v = static_cast<float>(row[t]);
            if (v > local_best || (v == local_best && t < local_idx)) {
                local_best = v;
                local_idx = t;
            }
        }
        best_vals[tid] = local_best;
        best_idxs[tid] = local_idx;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other_v = best_vals[tid + stride];
                const int other_i = best_idxs[tid + stride];
                if (other_v > best_vals[tid] || (other_v == best_vals[tid] && other_i < best_idxs[tid])) {
                    best_vals[tid] = other_v;
                    best_idxs[tid] = other_i;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            const int idx = best_idxs[0];
            out[out_i] = static_cast<int64_t>(idx + offset);
            if (idx >= 0 && idx < kv_len) {
                row[idx] = static_cast<score_t>(-INFINITY);
            }
        }
        __syncthreads();
    }
}

}  // namespace

std::vector<torch::Tensor> hc_split_pre_forward_cuda(
    const torch::Tensor& mixes,
    const torch::Tensor& x,
    const torch::Tensor& hc_scale,
    const torch::Tensor& hc_base,
    int64_t hc_mult,
    int64_t sinkhorn_iters,
    double eps) {
    c10::cuda::CUDAGuard device_guard(x.device());
    const int tokens = static_cast<int>(x.size(0) * x.size(1));
    const int dim = static_cast<int>(x.size(3));
    auto y = torch::empty({x.size(0), x.size(1), dim}, x.options());
    auto pre = torch::empty({x.size(0), x.size(1), hc_mult}, mixes.options());
    auto post = torch::empty({x.size(0), x.size(1), hc_mult}, mixes.options());
    auto comb = torch::empty({x.size(0), x.size(1), hc_mult, hc_mult}, mixes.options());
    const int block = 256;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "hc_split_pre_forward", [&] {
        hc_split_pre_kernel<scalar_t><<<tokens, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            mixes.data_ptr<float>(),
            x.data_ptr<scalar_t>(),
            hc_scale.data_ptr<float>(),
            hc_base.data_ptr<float>(),
            y.data_ptr<scalar_t>(),
            pre.data_ptr<float>(),
            post.data_ptr<float>(),
            comb.data_ptr<float>(),
            tokens,
            dim,
            static_cast<int>(sinkhorn_iters),
            static_cast<float>(eps));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, pre, post, comb};
}

torch::Tensor hc_post_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb) {
    c10::cuda::CUDAGuard device_guard(x.device());
    const int tokens = static_cast<int>(x.size(0) * x.size(1));
    const int dim = static_cast<int>(x.size(2));
    auto y = torch::empty({x.size(0), x.size(1), residual.size(2), dim}, x.options());
    const dim3 grid(tokens, 4);
    const int block = 256;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "hc_post_forward", [&] {
        hc_post_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(),
            residual.data_ptr<scalar_t>(),
            post.data_ptr<float>(),
            comb.data_ptr<float>(),
            y.data_ptr<scalar_t>(),
            tokens,
            dim);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor int8_gemm_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_s) {
    c10::cuda::CUDAGuard device_guard(x.device());

    auto x_contig = x.contiguous();
    auto wq_contig = weight_q.contiguous();
    auto ws_contig = weight_s.contiguous();

    const auto k = static_cast<int>(x_contig.size(-1));
    const auto n = static_cast<int>(weight_q.size(0));
    const auto rows = static_cast<int>(x_contig.numel() / k);

    auto x_q = torch::empty({rows, 1, k}, x.options().dtype(torch::kInt8));
    auto x_scale = torch::empty({rows, 1}, x.options().dtype(torch::kFloat32));
    auto out = torch::empty({rows, 1, n}, x.options().dtype(torch::kFloat32));

    const dim3 quant_grid(rows, 1);
    const dim3 quant_block(kQuantThreads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "int8_gemm_quantize", [&] {
        quantize_rows_kernel<scalar_t><<<quant_grid, quant_block, 0, at::cuda::getCurrentCUDAStream()>>>(
            x_contig.data_ptr<scalar_t>(),
            x_q.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(),
            rows,
            1,
            k);
    });

    const dim3 gemm_grid(ceil_div(n, kGemmThreads), rows, 1);
    const dim3 gemm_block(kGemmThreads);
    const size_t shared_bytes = static_cast<size_t>(k / 4) * sizeof(int);
    int8_gemm_rows_kernel<<<gemm_grid, gemm_block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        x_q.data_ptr<int8_t>(),
        x_scale.data_ptr<float>(),
        wq_contig.data_ptr<int8_t>(),
        ws_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        1,
        n,
        k);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto out_shape = x.sizes().vec();
    out_shape.back() = n;
    return out.view(out_shape);
}

torch::Tensor int8_gemm_pair_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_q0,
    const torch::Tensor& weight_s0,
    const torch::Tensor& weight_q1,
    const torch::Tensor& weight_s1) {
    c10::cuda::CUDAGuard device_guard(x.device());

    auto x_contig = x.contiguous();
    auto wq0_contig = weight_q0.contiguous();
    auto ws0_contig = weight_s0.contiguous();
    auto wq1_contig = weight_q1.contiguous();
    auto ws1_contig = weight_s1.contiguous();

    const auto k = static_cast<int>(x_contig.size(-1));
    const auto n = static_cast<int>(weight_q0.size(0));
    const auto rows = static_cast<int>(x_contig.numel() / k);

    auto x_q = torch::empty({rows, 1, k}, x.options().dtype(torch::kInt8));
    auto x_scale = torch::empty({rows, 1}, x.options().dtype(torch::kFloat32));
    auto out0 = torch::empty({rows, 1, n}, x.options().dtype(torch::kFloat32));
    auto out1 = torch::empty({rows, 1, n}, x.options().dtype(torch::kFloat32));

    const dim3 quant_grid(rows, 1);
    const dim3 quant_block(kQuantThreads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "int8_gemm_pair_quantize", [&] {
        quantize_rows_kernel<scalar_t><<<quant_grid, quant_block, 0, at::cuda::getCurrentCUDAStream()>>>(
            x_contig.data_ptr<scalar_t>(),
            x_q.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(),
            rows,
            1,
            k);
    });

    const dim3 gemm_grid(ceil_div(n, kGemmThreads), rows, 1);
    const dim3 gemm_block(kGemmThreads);
    const size_t shared_bytes = static_cast<size_t>(k / 4) * sizeof(int);
    int8_gemm_rows_kernel<<<gemm_grid, gemm_block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        x_q.data_ptr<int8_t>(),
        x_scale.data_ptr<float>(),
        wq0_contig.data_ptr<int8_t>(),
        ws0_contig.data_ptr<float>(),
        out0.data_ptr<float>(),
        rows,
        1,
        n,
        k);
    int8_gemm_rows_kernel<<<gemm_grid, gemm_block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        x_q.data_ptr<int8_t>(),
        x_scale.data_ptr<float>(),
        wq1_contig.data_ptr<int8_t>(),
        ws1_contig.data_ptr<float>(),
        out1.data_ptr<float>(),
        rows,
        1,
        n,
        k);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto out_shape = x.sizes().vec();
    out_shape.back() = n;
    return torch::stack({out0.view(out_shape), out1.view(out_shape)}, 0);
}


torch::Tensor moe_single_token_int8_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    int64_t experts_start_idx,
    double swiglu_limit) {
    c10::cuda::CUDAGuard device_guard(x.device());
    const int topk = static_cast<int>(indices.size(0));
    const int dim = static_cast<int>(x.size(1));
    const int n_local_experts = static_cast<int>(w1q.size(0));
    const int inter_dim = static_cast<int>(w1q.size(1));
    auto x_contig = x.contiguous();
    auto indices_contig = indices.contiguous();
    auto weights_contig = weights.contiguous();
    auto y = torch::zeros({1, dim}, x.options().dtype(torch::kFloat32));
    auto x_q = torch::empty({1, 1, dim}, x.options().dtype(torch::kInt8));
    auto x_scale = torch::empty({1, 1}, x.options().dtype(torch::kFloat32));
    auto gate_i32 = torch::empty({topk, inter_dim}, x.options().dtype(torch::kInt32));
    auto up_i32 = torch::empty({topk, inter_dim}, x.options().dtype(torch::kInt32));
    auto hidden_q = torch::empty({topk, inter_dim}, x.options().dtype(torch::kInt8));
    auto hidden_scale = torch::empty({topk}, x.options().dtype(torch::kFloat32));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "moe_single_quantize_x", [&] {
        quantize_rows_kernel<scalar_t><<<1, kQuantThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x_contig.data_ptr<scalar_t>(),
            x_q.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(),
            1,
            1,
            dim);
    });

    const dim3 w1w3_grid(ceil_div(inter_dim, kGemmThreads), topk);
    const dim3 gemm_block(kGemmThreads);
    const size_t x_shared_bytes = static_cast<size_t>(dim / 4) * sizeof(int);
    moe_single_w1w3_kernel<<<w1w3_grid, gemm_block, x_shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        x_q.data_ptr<int8_t>(),
        x_scale.data_ptr<float>(),
        indices_contig.data_ptr<int64_t>(),
        w1q.data_ptr<int8_t>(),
        w3q.data_ptr<int8_t>(),
        gate_i32.data_ptr<int32_t>(),
        up_i32.data_ptr<int32_t>(),
        topk,
        static_cast<int>(experts_start_idx),
        n_local_experts,
        dim,
        inter_dim);

    moe_single_swiglu_quant_kernel<<<topk, kQuantThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        gate_i32.data_ptr<int32_t>(),
        up_i32.data_ptr<int32_t>(),
        x_scale.data_ptr<float>(),
        indices_contig.data_ptr<int64_t>(),
        weights_contig.data_ptr<float>(),
        w1s.data_ptr<float>(),
        w3s.data_ptr<float>(),
        hidden_q.data_ptr<int8_t>(),
        hidden_scale.data_ptr<float>(),
        topk,
        static_cast<int>(experts_start_idx),
        n_local_experts,
        inter_dim,
        static_cast<float>(swiglu_limit));

    const dim3 w2_grid(ceil_div(dim, kGemmThreads), topk);
    const size_t h_shared_bytes = static_cast<size_t>(inter_dim / 4) * sizeof(int);
    moe_single_w2_accum_kernel<<<w2_grid, gemm_block, h_shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        hidden_q.data_ptr<int8_t>(),
        hidden_scale.data_ptr<float>(),
        indices_contig.data_ptr<int64_t>(),
        w2q.data_ptr<int8_t>(),
        w2s.data_ptr<float>(),
        y.data_ptr<float>(),
        topk,
        static_cast<int>(experts_start_idx),
        n_local_experts,
        dim,
        inter_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// ============================================================================
// v2 single-token MoE: compact buffers + route_to_slot indirection.
// w*q / w*s are packed [k_active, ...] (one entry per *active* unique local
// expert). route_to_slot[topk] maps each top-k route to a slot in those packed
// buffers, or -1 to skip (route lands on a foreign rank).
// ============================================================================

__global__ void moe_single_w1w3_v2_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const int64_t* __restrict__ route_to_slot,
    const int8_t* __restrict__ w1q,
    const int8_t* __restrict__ w3q,
    int32_t* __restrict__ gate_i32,
    int32_t* __restrict__ up_i32,
    int topk,
    int dim,
    int inter_dim) {
    const int route = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= topk) return;
    const int slot = static_cast<int>(route_to_slot[route]);
    if (slot < 0) return;
    const int packs = dim / 4;
    extern __shared__ int x_shared_v2[];
    const int* x_i32 = reinterpret_cast<const int*>(x_q);
    for (int idx = threadIdx.x; idx < packs; idx += blockDim.x) {
        x_shared_v2[idx] = x_i32[idx];
    }
    __syncthreads();
    if (col >= inter_dim) return;
    const int8_t* w1_row = w1q + (static_cast<int64_t>(slot) * inter_dim + col) * dim;
    const int8_t* w3_row = w3q + (static_cast<int64_t>(slot) * inter_dim + col) * dim;
    const int* w1_i32 = reinterpret_cast<const int*>(w1_row);
    const int* w3_i32 = reinterpret_cast<const int*>(w3_row);
    int acc1 = 0;
    int acc3 = 0;
    for (int idx = 0; idx < packs; ++idx) {
        const int xv = x_shared_v2[idx];
        acc1 = __dp4a(xv, w1_i32[idx], acc1);
        acc3 = __dp4a(xv, w3_i32[idx], acc3);
    }
    gate_i32[route * inter_dim + col] = acc1;
    up_i32[route * inter_dim + col] = acc3;
}

__global__ void moe_single_swiglu_quant_v2_kernel(
    const int32_t* __restrict__ gate_i32,
    const int32_t* __restrict__ up_i32,
    const float* __restrict__ x_scale,
    const int64_t* __restrict__ route_to_slot,
    const float* __restrict__ weights,
    const float* __restrict__ w1s,
    const float* __restrict__ w3s,
    int8_t* __restrict__ hidden_q,
    float* __restrict__ hidden_scale,
    int topk,
    int inter_dim,
    float swiglu_limit) {
    const int route = blockIdx.x;
    const int tid = threadIdx.x;
    if (route >= topk) return;
    const int slot = static_cast<int>(route_to_slot[route]);
    if (slot < 0) {
        if (tid == 0) hidden_scale[route] = 0.0f;
        return;
    }
    __shared__ float sdata[kQuantThreads];
    float local_max = 0.0f;
    const float xs = x_scale[0];
    const float route_weight = weights[route];
    for (int col = tid; col < inter_dim; col += blockDim.x) {
        float gate = static_cast<float>(gate_i32[route * inter_dim + col]) * xs * w1s[slot * inter_dim + col];
        float up = static_cast<float>(up_i32[route * inter_dim + col]) * xs * w3s[slot * inter_dim + col];
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
        float gate = static_cast<float>(gate_i32[route * inter_dim + col]) * xs * w1s[slot * inter_dim + col];
        float up = static_cast<float>(up_i32[route * inter_dim + col]) * xs * w3s[slot * inter_dim + col];
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

__global__ void moe_single_w2_accum_v2_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_to_slot,
    const int8_t* __restrict__ w2q,
    const float* __restrict__ w2s,
    float* __restrict__ y,
    int topk,
    int dim,
    int inter_dim) {
    const int route = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= topk) return;
    const int slot = static_cast<int>(route_to_slot[route]);
    if (slot < 0) return;
    const int packs = inter_dim / 4;
    extern __shared__ int h_shared_v2[];
    const int* h_i32 = reinterpret_cast<const int*>(hidden_q + route * inter_dim);
    for (int idx = threadIdx.x; idx < packs; idx += blockDim.x) {
        h_shared_v2[idx] = h_i32[idx];
    }
    __syncthreads();
    if (col >= dim) return;
    const int8_t* w2_row = w2q + (static_cast<int64_t>(slot) * dim + col) * inter_dim;
    const int* w2_i32 = reinterpret_cast<const int*>(w2_row);
    int acc = 0;
    for (int idx = 0; idx < packs; ++idx) {
        acc = __dp4a(h_shared_v2[idx], w2_i32[idx], acc);
    }
    const float value = static_cast<float>(acc) * hidden_scale[route] * w2s[slot * dim + col];
    atomicAdd(y + col, value);
}

torch::Tensor moe_single_token_int8_forward_v2_cuda(
    const torch::Tensor& x,
    const torch::Tensor& route_to_slot,
    const torch::Tensor& weights,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    double swiglu_limit) {
    c10::cuda::CUDAGuard device_guard(x.device());
    const int topk = static_cast<int>(route_to_slot.size(0));
    const int dim = static_cast<int>(x.size(1));
    const int inter_dim = static_cast<int>(w1q.size(1));
    auto x_contig = x.contiguous();
    auto route_slot_contig = route_to_slot.contiguous();
    auto weights_contig = weights.contiguous();
    auto y = torch::zeros({1, dim}, x.options().dtype(torch::kFloat32));
    auto x_q = torch::empty({1, 1, dim}, x.options().dtype(torch::kInt8));
    auto x_scale = torch::empty({1, 1}, x.options().dtype(torch::kFloat32));
    auto gate_i32 = torch::empty({topk, inter_dim}, x.options().dtype(torch::kInt32));
    auto up_i32 = torch::empty({topk, inter_dim}, x.options().dtype(torch::kInt32));
    auto hidden_q = torch::empty({topk, inter_dim}, x.options().dtype(torch::kInt8));
    auto hidden_scale = torch::empty({topk}, x.options().dtype(torch::kFloat32));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "moe_single_quantize_x_v2", [&] {
        quantize_rows_kernel<scalar_t><<<1, kQuantThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x_contig.data_ptr<scalar_t>(),
            x_q.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(),
            1,
            1,
            dim);
    });

    const dim3 w1w3_grid(ceil_div(inter_dim, kGemmThreads), topk);
    const dim3 gemm_block(kGemmThreads);
    const size_t x_shared_bytes = static_cast<size_t>(dim / 4) * sizeof(int);
    moe_single_w1w3_v2_kernel<<<w1w3_grid, gemm_block, x_shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        x_q.data_ptr<int8_t>(),
        x_scale.data_ptr<float>(),
        route_slot_contig.data_ptr<int64_t>(),
        w1q.data_ptr<int8_t>(),
        w3q.data_ptr<int8_t>(),
        gate_i32.data_ptr<int32_t>(),
        up_i32.data_ptr<int32_t>(),
        topk,
        dim,
        inter_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    moe_single_swiglu_quant_v2_kernel<<<topk, kQuantThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        gate_i32.data_ptr<int32_t>(),
        up_i32.data_ptr<int32_t>(),
        x_scale.data_ptr<float>(),
        route_slot_contig.data_ptr<int64_t>(),
        weights_contig.data_ptr<float>(),
        w1s.data_ptr<float>(),
        w3s.data_ptr<float>(),
        hidden_q.data_ptr<int8_t>(),
        hidden_scale.data_ptr<float>(),
        topk,
        inter_dim,
        static_cast<float>(swiglu_limit));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const dim3 w2_grid(ceil_div(dim, kGemmThreads), topk);
    const size_t h_shared_bytes = static_cast<size_t>(inter_dim / 4) * sizeof(int);
    moe_single_w2_accum_v2_kernel<<<w2_grid, gemm_block, h_shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        hidden_q.data_ptr<int8_t>(),
        hidden_scale.data_ptr<float>(),
        route_slot_contig.data_ptr<int64_t>(),
        w2q.data_ptr<int8_t>(),
        w2s.data_ptr<float>(),
        y.data_ptr<float>(),
        topk,
        dim,
        inter_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor moe_prefill_int8_grouped_forward_cuda(
    const torch::Tensor& x_sorted,
    const torch::Tensor& route_weights_sorted,
    const torch::Tensor& seg_starts,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    double swiglu_limit) {
    c10::cuda::CUDAGuard device_guard(x_sorted.device());
    const int64_t routes = x_sorted.size(0);
    const int64_t dim = x_sorted.size(1);
    const int64_t n_experts = w1q.size(0);
    const int64_t inter_dim = w1q.size(1);
    auto out = torch::empty({routes, dim}, x_sorted.options().dtype(torch::kFloat32));
    auto seg_cpu = seg_starts.to(torch::kCPU, false);
    for (int64_t e = 0; e < n_experts; ++e) {
        const int64_t start = seg_cpu.scalar_type() == torch::kInt32
            ? static_cast<int64_t>(seg_cpu.data_ptr<int32_t>()[e])
            : seg_cpu.data_ptr<int64_t>()[e];
        const int64_t end = seg_cpu.scalar_type() == torch::kInt32
            ? static_cast<int64_t>(seg_cpu.data_ptr<int32_t>()[e + 1])
            : seg_cpu.data_ptr<int64_t>()[e + 1];
        const int64_t count = end - start;
        if (count <= 0) continue;
        auto x_e = x_sorted.narrow(0, start, count).contiguous();
        auto pair = int8_gemm_pair_forward_cuda(x_e, w1q[e], w1s[e], w3q[e], w3s[e]);
        auto gate = pair[0].contiguous();
        auto up = pair[1].contiguous();
        auto hidden = torch::empty({count, inter_dim}, x_sorted.options().dtype(torch::kFloat32));
        const int threads = 256;
        const int blocks = static_cast<int>((count * inter_dim + threads - 1) / threads);
        auto route_e = route_weights_sorted.narrow(0, start, count).contiguous();
        route_swiglu_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            gate.data_ptr<float>(),
            up.data_ptr<float>(),
            route_e.data_ptr<float>(),
            hidden.data_ptr<float>(),
            static_cast<int>(count),
            static_cast<int>(inter_dim),
            static_cast<float>(swiglu_limit));
        auto y = int8_gemm_forward_cuda(hidden, w2q[e], w2s[e]);
        out.narrow(0, start, count).copy_(y.view({count, dim}));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> moe_group_routes_cuda(
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    int64_t experts_start_idx,
    int64_t n_local_experts) {
    c10::cuda::CUDAGuard device_guard(indices.device());
    const int tokens = static_cast<int>(indices.size(0));
    const int topk = static_cast<int>(indices.size(1));
    const int experts = static_cast<int>(n_local_experts);
    const int total = tokens * topk;
    auto counts = torch::zeros({experts}, indices.options().dtype(torch::kInt32));
    auto seg_starts = torch::empty({experts + 1}, indices.options().dtype(torch::kInt32));
    auto offsets = torch::empty({experts}, indices.options().dtype(torch::kInt32));
    auto total_routes = torch::empty({1}, indices.options().dtype(torch::kInt32));
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    if (indices.scalar_type() == torch::kInt64) {
        moe_route_count_kernel<int64_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            indices.data_ptr<int64_t>(),
            counts.data_ptr<int32_t>(),
            tokens,
            topk,
            static_cast<int>(experts_start_idx),
            experts);
    } else {
        moe_route_count_kernel<int32_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            indices.data_ptr<int32_t>(),
            counts.data_ptr<int32_t>(),
            tokens,
            topk,
            static_cast<int>(experts_start_idx),
            experts);
    }
    moe_route_prefix_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        counts.data_ptr<int32_t>(),
        seg_starts.data_ptr<int32_t>(),
        offsets.data_ptr<int32_t>(),
        total_routes.data_ptr<int32_t>(),
        experts);
    auto total_cpu = total_routes.to(torch::kCPU, false);
    const int routes = static_cast<int>(total_cpu.data_ptr<int32_t>()[0]);
    auto route_tokens = torch::empty({routes}, indices.options().dtype(torch::kInt64));
    auto route_weights = torch::empty({routes}, weights.options().dtype(torch::kFloat32));
    if (routes > 0) {
        if (indices.scalar_type() == torch::kInt64) {
            moe_route_fill_kernel<int64_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                indices.data_ptr<int64_t>(),
                weights.data_ptr<float>(),
                offsets.data_ptr<int32_t>(),
                route_tokens.data_ptr<int64_t>(),
                route_weights.data_ptr<float>(),
                tokens,
                topk,
                static_cast<int>(experts_start_idx),
                experts);
        } else {
            moe_route_fill_kernel<int32_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                indices.data_ptr<int32_t>(),
                weights.data_ptr<float>(),
                offsets.data_ptr<int32_t>(),
                route_tokens.data_ptr<int64_t>(),
                route_weights.data_ptr<float>(),
                tokens,
                topk,
                static_cast<int>(experts_start_idx),
                experts);
        }
    }
    auto local_ids = counts.nonzero().flatten().to(torch::kLong);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {local_ids, route_tokens, route_weights, seg_starts};
}
torch::Tensor moe_prefill_int8_grouped_gemm_range_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& route_tokens,
    const torch::Tensor& route_weights_sorted,
    const torch::Tensor& seg_starts,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    double swiglu_limit,
    int expert_begin,
    int expert_end,
    torch::Tensor* y_out) {

    c10::cuda::CUDAGuard device_guard(x.device());
    const int routes = static_cast<int>(route_tokens.size(0));
    const int dim = static_cast<int>(x.size(1));
    const int n_experts_total = static_cast<int>(w1q.size(0));
    const int inter_dim = static_cast<int>(w1q.size(1));
    expert_begin = std::max(0, expert_begin);
    expert_end = std::min(n_experts_total, expert_end);
    const int n_experts = expert_end - expert_begin;
    if (routes == 0 || n_experts <= 0) {
        return y_out == nullptr ? torch::zeros({x.size(0), x.size(1)}, x.options().dtype(torch::kFloat32)) : *y_out;
    }
    auto seg_i32_full = seg_starts.scalar_type() == torch::kInt32 ? seg_starts.contiguous() : seg_starts.to(torch::kInt32);
    auto seg_cpu = seg_i32_full.to(torch::kCPU, false);
    const int32_t* seg_cpu_ptr = seg_cpu.data_ptr<int32_t>();
    auto seg_i32 = seg_i32_full.slice(0, expert_begin, expert_end + 1);
    const int route_begin = static_cast<int>(seg_cpu_ptr[expert_begin]);
    const int route_end = static_cast<int>(seg_cpu_ptr[expert_end]);
    const int range_routes = route_end - route_begin;
    if (range_routes <= 0) {
        return y_out == nullptr ? torch::zeros({x.size(0), x.size(1)}, x.options().dtype(torch::kFloat32)) : *y_out;
    }
    torch::Tensor seg_local;
    if (route_begin == 0) {
        seg_local = seg_i32;
    } else {
        seg_local = (seg_i32 - route_begin).contiguous();
    }
    auto counts_i32 = seg_local.slice(0, 1, n_experts + 1) - seg_local.slice(0, 0, n_experts);
    const int max_count = counts_i32.max().item<int>();
    if (max_count <= 0) {
        return y_out == nullptr ? torch::zeros({x.size(0), x.size(1)}, x.options().dtype(torch::kFloat32)) : *y_out;
    }

    auto route_tokens_range = route_tokens.slice(0, route_begin, route_end).contiguous();
    auto route_weights_range = route_weights_sorted.slice(0, route_begin, route_end).contiguous();
    auto w1q_range = w1q.slice(0, expert_begin, expert_end);
    auto w1s_range = w1s.slice(0, expert_begin, expert_end);
    auto w2q_range = w2q.slice(0, expert_begin, expert_end);
    auto w2s_range = w2s.slice(0, expert_begin, expert_end);
    auto w3q_range = w3q.slice(0, expert_begin, expert_end);
    auto w3s_range = w3s.slice(0, expert_begin, expert_end);

    auto x_sorted = torch::empty({range_routes, dim}, x.options());
    auto x_q = torch::empty({range_routes, 1, dim}, x.options().dtype(torch::kInt8));
    auto x_scale = torch::empty({range_routes, 1}, x.options().dtype(torch::kFloat32));
    auto x_pad = torch::empty({n_experts, max_count, dim}, x.options().dtype(torch::kInt8));
    auto x_scale_pad = torch::empty({n_experts, max_count}, x.options().dtype(torch::kFloat32));
    auto gate_i32 = torch::empty({n_experts, max_count, inter_dim}, x.options().dtype(torch::kInt32));
    auto up_i32 = torch::empty({n_experts, max_count, inter_dim}, x.options().dtype(torch::kInt32));
    auto hidden_pad = torch::empty({n_experts, max_count, inter_dim}, x.options().dtype(torch::kInt8));
    auto hidden_scale_pad = torch::empty({n_experts, max_count}, x.options().dtype(torch::kFloat32));
    auto out_i32 = torch::empty({n_experts, max_count, dim}, x.options().dtype(torch::kInt32));
    torch::Tensor y = y_out == nullptr ? torch::zeros({x.size(0), x.size(1)}, x.options().dtype(torch::kFloat32)) : *y_out;
    const int threads = 256;
    const int gather_blocks = static_cast<int>((static_cast<int64_t>(range_routes) * dim + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "moe_grouped_gemm_gather_range", [&] {
        gather_routes_kernel<scalar_t><<<gather_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(),
            route_tokens_range.data_ptr<int64_t>(),
            x_sorted.data_ptr<scalar_t>(),
            range_routes,
            dim);
    });
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "moe_grouped_gemm_quantize_x_range", [&] {
        quantize_rows_kernel<scalar_t><<<range_routes, kQuantThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x_sorted.data_ptr<scalar_t>(),
            x_q.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(),
            range_routes,
            1,
            dim);
    });
    const int pad_x_total = n_experts * max_count * dim;
    moe_pad_q_rows_kernel<<<ceil_div(pad_x_total, threads), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x_q.data_ptr<int8_t>(),
        x_scale.data_ptr<float>(),
        seg_local.data_ptr<int32_t>(),
        x_pad.data_ptr<int8_t>(),
        x_scale_pad.data_ptr<float>(),
        n_experts,
        max_count,
        dim);
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream());
    const int alpha = 1;
    const int beta = 0;
    auto status = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        inter_dim,
        max_count,
        dim,
        &alpha,
        w1q_range.data_ptr<int8_t>(),
        CUDA_R_8I,
        dim,
        static_cast<long long>(inter_dim) * dim,
        x_pad.data_ptr<int8_t>(),
        CUDA_R_8I,
        dim,
        static_cast<long long>(max_count) * dim,
        &beta,
        gate_i32.data_ptr<int32_t>(),
        CUDA_R_32I,
        inter_dim,
        static_cast<long long>(max_count) * inter_dim,
        n_experts,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "w1 range cublasGemmStridedBatchedEx failed");
    status = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        inter_dim,
        max_count,
        dim,
        &alpha,
        w3q_range.data_ptr<int8_t>(),
        CUDA_R_8I,
        dim,
        static_cast<long long>(inter_dim) * dim,
        x_pad.data_ptr<int8_t>(),
        CUDA_R_8I,
        dim,
        static_cast<long long>(max_count) * dim,
        &beta,
        up_i32.data_ptr<int32_t>(),
        CUDA_R_32I,
        inter_dim,
        static_cast<long long>(max_count) * inter_dim,
        n_experts,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "w3 range cublasGemmStridedBatchedEx failed");
    moe_pair_swiglu_quantize_padded_kernel<<<dim3(n_experts, max_count), kQuantThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        gate_i32.data_ptr<int32_t>(),
        up_i32.data_ptr<int32_t>(),
        x_scale_pad.data_ptr<float>(),
        seg_local.data_ptr<int32_t>(),
        route_weights_range.data_ptr<float>(),
        w1s_range.data_ptr<float>(),
        w3s_range.data_ptr<float>(),
        hidden_pad.data_ptr<int8_t>(),
        hidden_scale_pad.data_ptr<float>(),
        n_experts,
        max_count,
        inter_dim,
        static_cast<float>(swiglu_limit));
    status = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        dim,
        max_count,
        inter_dim,
        &alpha,
        w2q_range.data_ptr<int8_t>(),
        CUDA_R_8I,
        inter_dim,
        static_cast<long long>(dim) * inter_dim,
        hidden_pad.data_ptr<int8_t>(),
        CUDA_R_8I,
        inter_dim,
        static_cast<long long>(max_count) * inter_dim,
        &beta,
        out_i32.data_ptr<int32_t>(),
        CUDA_R_32I,
        dim,
        static_cast<long long>(max_count) * dim,
        n_experts,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "w2 range cublasGemmStridedBatchedEx failed");
    const int out_total = n_experts * max_count * dim;
    moe_depad_scatter_i32_kernel<<<ceil_div(out_total, threads), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        out_i32.data_ptr<int32_t>(),
        hidden_scale_pad.data_ptr<float>(),
        route_tokens_range.data_ptr<int64_t>(),
        seg_local.data_ptr<int32_t>(),
        w2s_range.data_ptr<float>(),
        y.data_ptr<float>(),
        n_experts,
        max_count,
        dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor moe_prefill_int8_grouped_gemm_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& route_tokens,
    const torch::Tensor& route_weights_sorted,
    const torch::Tensor& seg_starts,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    double swiglu_limit) {
    auto y = moe_prefill_int8_grouped_gemm_range_forward_cuda(
        x,
        route_tokens,
        route_weights_sorted,
        seg_starts,
        w1q,
        w1s,
        w2q,
        w2s,
        w3q,
        w3s,
        swiglu_limit,
        0,
        static_cast<int>(w1q.size(0)),
        nullptr);
    if (const char* v = std::getenv("DEEPSEEK_GPU_PREFILL_MOE_PAD_PROFILE")) {
        if (std::strcmp(v, "0") != 0 && std::strcmp(v, "false") != 0 && std::strcmp(v, "False") != 0) {
            auto seg_i32 = seg_starts.scalar_type() == torch::kInt32 ? seg_starts.contiguous() : seg_starts.to(torch::kInt32);
            auto counts_i32 = seg_i32.slice(0, 1, seg_i32.size(0)) - seg_i32.slice(0, 0, seg_i32.size(0) - 1);
            const int max_count = counts_i32.max().item<int>();
            const int n_experts = static_cast<int>(w1q.size(0));
            const int routes = static_cast<int>(route_tokens.size(0));
            const long long padded_rows = static_cast<long long>(n_experts) * max_count;
            const double ratio = routes > 0 ? static_cast<double>(padded_rows) / static_cast<double>(routes) : 0.0;
            std::printf("gpu_prefill_moe_pad_profile routes=%d experts=%d max_count=%d padded_rows=%lld padding_ratio=%.3f\n", routes, n_experts, max_count, padded_rows, ratio);
            std::fflush(stdout);
        }
    }
    return y;
}

torch::Tensor moe_prefill_int8_grouped_gemm_bucketed_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& route_tokens,
    const torch::Tensor& route_weights_sorted,
    const torch::Tensor& seg_starts,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    double swiglu_limit) {
    c10::cuda::CUDAGuard device_guard(x.device());
    const int n_experts = static_cast<int>(w1q.size(0));
    auto y = torch::zeros({x.size(0), x.size(1)}, x.options().dtype(torch::kFloat32));
    if (route_tokens.size(0) == 0 || n_experts <= 0) {
        return y;
    }
    const int bucket = std::max(1, std::min(n_experts, static_cast<int>(std::atoi(std::getenv("DEEPSEEK_GPU_PREFILL_MOE_BUCKET_EXPERTS") ? std::getenv("DEEPSEEK_GPU_PREFILL_MOE_BUCKET_EXPERTS") : "16"))));
    for (int begin = 0; begin < n_experts; begin += bucket) {
        const int end = std::min(n_experts, begin + bucket);
        moe_prefill_int8_grouped_gemm_range_forward_cuda(
            x,
            route_tokens,
            route_weights_sorted,
            seg_starts,
            w1q,
            w1s,
            w2q,
            w2s,
            w3q,
            w3s,
            swiglu_limit,
            begin,
            end,
            &y);
    }
    return y;
}

torch::Tensor moe_prefill_int8_fused_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& route_tokens,
    const torch::Tensor& route_weights_sorted,
    const torch::Tensor& seg_starts,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    double swiglu_limit) {
    c10::cuda::CUDAGuard device_guard(x.device());
    const int routes = static_cast<int>(route_tokens.size(0));
    const int dim = static_cast<int>(x.size(1));
    auto x_sorted = torch::empty({routes, dim}, x.options());
    const int threads = 256;
    const int gather_blocks = static_cast<int>((static_cast<int64_t>(routes) * dim + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "moe_route_gather", [&] {
        gather_routes_kernel<scalar_t><<<gather_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(),
            route_tokens.data_ptr<int64_t>(),
            x_sorted.data_ptr<scalar_t>(),
            routes,
            dim);
    });
    auto y_sorted = moe_prefill_int8_grouped_forward_cuda(
        x_sorted,
        route_weights_sorted,
        seg_starts,
        w1q,
        w1s,
        w2q,
        w2s,
        w3q,
        w3s,
        swiglu_limit);
    auto y = torch::zeros({x.size(0), x.size(1)}, x.options().dtype(torch::kFloat32));
    const int scatter_blocks = static_cast<int>((static_cast<int64_t>(routes) * dim + threads - 1) / threads);
    scatter_routes_kernel<float><<<scatter_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        y_sorted.data_ptr<float>(),
        route_tokens.data_ptr<int64_t>(),
        y.data_ptr<float>(),
        routes,
        dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
torch::Tensor wo_a_int8_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_s) {
    c10::cuda::CUDAGuard device_guard(x.device());

    const auto bsz = static_cast<int>(x.size(0));
    const auto seqlen = static_cast<int>(x.size(1));
    const auto groups = static_cast<int>(x.size(2));
    const auto k = static_cast<int>(x.size(3));
    const auto n = static_cast<int>(weight_q.size(1));
    const auto rows = bsz * seqlen;

    auto x_contig = x.contiguous();
    auto wq_contig = weight_q.contiguous();
    auto ws_contig = weight_s.contiguous();

    auto x_q = torch::empty({rows, groups, k}, x.options().dtype(torch::kInt8));
    auto x_scale = torch::empty({rows, groups}, x.options().dtype(torch::kFloat32));
    auto out = torch::empty({rows, groups, n}, x.options().dtype(torch::kFloat32));

    const dim3 quant_grid(rows, groups);
    const dim3 quant_block(kQuantThreads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "wo_a_int8_quantize", [&] {
        quantize_rows_kernel<scalar_t><<<quant_grid, quant_block, 0, at::cuda::getCurrentCUDAStream()>>>(
            x_contig.data_ptr<scalar_t>(),
            x_q.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(),
            rows,
            groups,
            k);
    });

    // CUDA grid Y/Z dim is capped at 65535 (gridDimY = blockIdx.y bound). For very
    // long prefill (e.g. seqlen >= 64k) `rows` exceeds that limit and the launch
    // returns cudaErrorInvalidConfiguration. Slice rows into chunks <= 65535 and
    // dispatch one GEMM launch per chunk; the kernel itself reads `rows` only via
    // pointer arithmetic on `(row * groups + group)`, so a row offset on the input
    // pointers is sufficient.
    const int kMaxGridY = 65535;
    const dim3 gemm_block(kGemmThreads);
    const size_t shared_bytes = static_cast<size_t>(k / 4) * sizeof(int);
    int row_offset = 0;
    while (row_offset < rows) {
        const int chunk_rows = std::min(rows - row_offset, kMaxGridY);
        const dim3 gemm_grid(ceil_div(n, kGemmThreads), chunk_rows, groups);
        int8_gemm_rows_kernel<<<gemm_grid, gemm_block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            x_q.data_ptr<int8_t>() + static_cast<size_t>(row_offset) * groups * k,
            x_scale.data_ptr<float>() + static_cast<size_t>(row_offset) * groups,
            wq_contig.data_ptr<int8_t>(),
            ws_contig.data_ptr<float>(),
            out.data_ptr<float>() + static_cast<size_t>(row_offset) * groups * n,
            chunk_rows,
            groups,
            n,
            k);
        row_offset += chunk_rows;
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.view({bsz, seqlen, groups, n});
}

torch::Tensor flashinfer_style_sparse_attn_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& attn_sink,
    const torch::Tensor& topk_idxs,
    double softmax_scale) {
    c10::cuda::CUDAGuard device_guard(q.device());

    const auto bsz = static_cast<int>(q.size(0));
    const auto heads = static_cast<int>(q.size(2));
    const auto dim = static_cast<int>(q.size(3));
    const auto kv_len = static_cast<int>(kv.size(1));
    const auto topk = static_cast<int>(topk_idxs.size(2));

    auto q_contig = q.contiguous().view({bsz, heads, dim});
    auto kv_contig = kv.contiguous();
    auto sink_contig = attn_sink.contiguous();
    auto idx_contig = topk_idxs.contiguous().view({bsz, topk});
    auto out = torch::empty({bsz, heads, dim}, q.options());

    const dim3 grid(bsz * heads);
    const dim3 block(kAttnThreads);
    const size_t shared_bytes =
        static_cast<size_t>(topk) * sizeof(float) +
        static_cast<size_t>(dim) * sizeof(float) +
        static_cast<size_t>(topk) * sizeof(int32_t) +
        static_cast<size_t>(kAttnThreads) * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q_contig.scalar_type(), "flashinfer_style_sparse_attn", [&] {
        flashinfer_style_sparse_attn_kernel<scalar_t><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            q_contig.data_ptr<scalar_t>(),
            kv_contig.data_ptr<scalar_t>(),
            sink_contig.data_ptr<float>(),
            idx_contig.data_ptr<int32_t>(),
            out.data_ptr<scalar_t>(),
            bsz,
            heads,
            kv_len,
            topk,
            dim,
            static_cast<float>(softmax_scale));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.view({bsz, 1, heads, dim});
}

torch::Tensor prefill_sparse_attn_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& attn_sink,
    const torch::Tensor& topk_idxs,
    double softmax_scale) {
    c10::cuda::CUDAGuard device_guard(q.device());
    const auto bsz = static_cast<int>(q.size(0));
    const auto seqlen = static_cast<int>(q.size(1));
    const auto heads = static_cast<int>(q.size(2));
    const auto dim = static_cast<int>(q.size(3));
    const auto kv_len = static_cast<int>(kv.size(1));
    const auto topk = static_cast<int>(topk_idxs.size(2));

    auto q_contig = q.contiguous();
    auto kv_contig = kv.contiguous();
    auto sink_contig = attn_sink.contiguous();
    auto idx_contig = topk_idxs.contiguous();
    auto out = torch::empty_like(q_contig);

    const dim3 grid(bsz * seqlen * heads);
    const dim3 block(kAttnThreads);
    const size_t shared_bytes =
        static_cast<size_t>(topk) * sizeof(float) +
        static_cast<size_t>(dim) * sizeof(float) +
        static_cast<size_t>(topk) * sizeof(int32_t) +
        static_cast<size_t>(kAttnThreads) * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q_contig.scalar_type(), "prefill_sparse_attn", [&] {
        prefill_sparse_attn_kernel<scalar_t><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            q_contig.data_ptr<scalar_t>(),
            kv_contig.data_ptr<scalar_t>(),
            sink_contig.data_ptr<float>(),
            idx_contig.data_ptr<int32_t>(),
            out.data_ptr<scalar_t>(),
            bsz,
            seqlen,
            heads,
            kv_len,
            topk,
            dim,
            static_cast<float>(softmax_scale));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor prefill_sparse_attn_headpair_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& attn_sink,
    const torch::Tensor& topk_idxs,
    double softmax_scale) {
    c10::cuda::CUDAGuard device_guard(q.device());
    const auto bsz = static_cast<int>(q.size(0));
    const auto seqlen = static_cast<int>(q.size(1));
    const auto heads = static_cast<int>(q.size(2));
    const auto dim = static_cast<int>(q.size(3));
    const auto kv_len = static_cast<int>(kv.size(1));
    const auto topk = static_cast<int>(topk_idxs.size(2));

    auto q_contig = q.contiguous();
    auto kv_contig = kv.contiguous();
    auto sink_contig = attn_sink.contiguous();
    auto idx_contig = topk_idxs.contiguous();
    auto out = torch::empty_like(q_contig);

    const int head_pairs = (heads + 1) / 2;
    const dim3 grid(bsz * seqlen * head_pairs);
    const dim3 block(kAttnThreads);
    const size_t shared_bytes =
        static_cast<size_t>(topk * 2) * sizeof(float) +
        static_cast<size_t>(dim * 2) * sizeof(float) +
        static_cast<size_t>(topk) * sizeof(int32_t) +
        static_cast<size_t>(kAttnThreads * 2) * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q_contig.scalar_type(), "prefill_sparse_attn_headpair", [&] {
        prefill_sparse_attn_headpair_kernel<scalar_t><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            q_contig.data_ptr<scalar_t>(),
            kv_contig.data_ptr<scalar_t>(),
            sink_contig.data_ptr<float>(),
            idx_contig.data_ptr<int32_t>(),
            out.data_ptr<scalar_t>(),
            bsz,
            seqlen,
            heads,
            kv_len,
            topk,
            dim,
            static_cast<float>(softmax_scale));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_decode_sparse_attn_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& attn_sink,
    const torch::Tensor& topk_idxs,
    double softmax_scale) {
    c10::cuda::CUDAGuard device_guard(q.device());

    const auto bsz = static_cast<int>(q.size(0));
    const auto heads = static_cast<int>(q.size(2));
    const auto dim = static_cast<int>(q.size(3));
    const auto kv_len = static_cast<int>(kv.size(1));
    const auto topk = static_cast<int>(topk_idxs.size(2));

    auto q_contig = q.contiguous().view({bsz, heads, dim});
    auto kv_contig = kv.contiguous();
    auto sink_contig = attn_sink.contiguous();
    auto idx_contig = topk_idxs.contiguous().view({bsz, topk});
    auto out = torch::empty({bsz, heads, dim}, q.options());

    const dim3 grid(bsz * heads);
    const dim3 block(kAttnThreads);
    const size_t shared_bytes =
        static_cast<size_t>(topk) * sizeof(float) +
        static_cast<size_t>(dim) * sizeof(float) +
        static_cast<size_t>(topk) * sizeof(int32_t) +
        static_cast<size_t>(kAttnThreads) * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q_contig.scalar_type(), "fused_decode_sparse_attn", [&] {
        fused_decode_sparse_attn_kernel<scalar_t><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            q_contig.data_ptr<scalar_t>(),
            kv_contig.data_ptr<scalar_t>(),
            sink_contig.data_ptr<float>(),
            idx_contig.data_ptr<int32_t>(),
            out.data_ptr<scalar_t>(),
            bsz,
            heads,
            kv_len,
            topk,
            dim,
            static_cast<float>(softmax_scale));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.view({bsz, 1, heads, dim});
}

torch::Tensor fused_decode_sparse_attn_wmma_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& attn_sink,
    const torch::Tensor& topk_idxs,
    double softmax_scale) {
    c10::cuda::CUDAGuard device_guard(q.device());

    const auto bsz = static_cast<int>(q.size(0));
    const auto heads = static_cast<int>(q.size(2));
    const auto dim = static_cast<int>(q.size(3));
    const auto kv_len = static_cast<int>(kv.size(1));
    const auto topk = static_cast<int>(topk_idxs.size(2));
    TORCH_CHECK(dim == 512, "WMMA attention currently requires head_dim=512");
    TORCH_CHECK(topk <= 256, "WMMA attention currently requires topk<=256");

    auto q_contig = q.contiguous().view({bsz, heads, dim});
    auto kv_contig = kv.contiguous();
    auto sink_contig = attn_sink.contiguous();
    auto idx_contig = topk_idxs.contiguous().view({bsz, topk});
    auto out = torch::empty({bsz, heads, dim}, q.options());

    const dim3 grid(bsz * heads);
    const dim3 block(kAttnThreads);
    const size_t shared_bytes =
        static_cast<size_t>(16 * 16) * sizeof(half) +
        static_cast<size_t>(16 * 16) * sizeof(half) +
        static_cast<size_t>(16 * 16) * sizeof(float) +
        static_cast<size_t>(topk) * sizeof(float) +
        static_cast<size_t>(kAttnThreads) * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q_contig.scalar_type(), "fused_decode_sparse_attn_wmma", [&] {
        fused_decode_sparse_attn_wmma_kernel<scalar_t><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            q_contig.data_ptr<scalar_t>(),
            kv_contig.data_ptr<scalar_t>(),
            sink_contig.data_ptr<float>(),
            idx_contig.data_ptr<int32_t>(),
            out.data_ptr<scalar_t>(),
            bsz,
            heads,
            kv_len,
            topk,
            static_cast<float>(softmax_scale));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.view({bsz, 1, heads, dim});
}


torch::Tensor hadamard128_forward_cuda(const torch::Tensor& x) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto x_contig = x.contiguous();
    const auto rows = static_cast<int>(x_contig.numel() / 128);
    auto out = torch::empty_like(x_contig);
    const dim3 grid(rows);
    const dim3 block(128);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "hadamard128_forward", [&] {
        hadamard128_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            x_contig.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            rows);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.view(x_contig.sizes());
}

torch::Tensor c4_topk_from_scores_cuda(
    const torch::Tensor& scores,
    int64_t offset,
    int64_t index_topk) {
    c10::cuda::CUDAGuard device_guard(scores.device());
    auto scores_contig = scores.contiguous();
    const auto bsz = static_cast<int>(scores_contig.size(0));
    const auto kv_len = static_cast<int>(scores_contig.size(2));
    const int k = std::min<int>(static_cast<int>(index_topk), kv_len);
    auto topk = torch::empty({bsz, std::max(k, 1)}, scores.options().dtype(torch::kInt64));
    if (k > 0) {
        auto scores_flat = scores_contig.view({bsz, kv_len});
        const size_t topk_shared_bytes = 256 * (sizeof(float) + sizeof(int));
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores_flat.scalar_type(), "c4_topk_from_scores", [&] {
            c4_topk_kernel<scalar_t><<<bsz, 256, topk_shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                scores_flat.data_ptr<scalar_t>(),
                topk.data_ptr<int64_t>(),
                bsz,
                kv_len,
                k,
                static_cast<int>(offset));
        });
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return topk.view({bsz, 1, std::max(k, 1)});
}

torch::Tensor fused_c4_indexer_decode_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& score,
    const torch::Tensor& weights,
    const torch::Tensor& ape,
    const torch::Tensor& norm_weight,
    const torch::Tensor& freqs,
    const torch::Tensor& kv_state,
    const torch::Tensor& score_state,
    const torch::Tensor& kv_cache,
    int64_t start_pos,
    int64_t offset,
    int64_t index_topk,
    double norm_eps,
    bool return_scores) {
    c10::cuda::CUDAGuard device_guard(q.device());
    const auto bsz = static_cast<int>(q.size(0));
    const auto heads = static_cast<int>(q.size(2));
    const auto max_cache_len = static_cast<int>(kv_cache.size(1));
    const int kv_len = static_cast<int>((start_pos + 1) / 4);
    const int k = std::min<int>(static_cast<int>(index_topk), kv_len);

    auto q_contig = q.contiguous().view({bsz * heads, 128});
    auto kv_contig = kv.contiguous().view({bsz, 256});
    auto score_contig = score.contiguous().view({bsz, 256});
    auto weights_contig = weights.contiguous().view({bsz, heads}).to(torch::kFloat32);
    auto freqs_contig = freqs.contiguous().view({64});
    auto q_work = torch::empty({bsz * heads, 128}, q.options().dtype(torch::kFloat32));

    const dim3 update_grid(bsz);
    const dim3 update_block(256);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, kv_cache.scalar_type(), "c4_update_cache", [&] {
        c4_update_cache_kernel<scalar_t><<<update_grid, update_block, 0, at::cuda::getCurrentCUDAStream()>>>(
            kv_contig.data_ptr<float>(),
            score_contig.data_ptr<float>(),
            ape.data_ptr<float>(),
            norm_weight.data_ptr<float>(),
            freqs_contig.data_ptr<float>(),
            kv_state.data_ptr<float>(),
            score_state.data_ptr<float>(),
            kv_cache.data_ptr<scalar_t>(),
            bsz,
            max_cache_len,
            static_cast<int>(start_pos),
            static_cast<float>(norm_eps));
    });

    const dim3 q_grid(bsz * heads);
    const dim3 q_block(128);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q_contig.scalar_type(), "c4_quant_q", [&] {
        c4_quant_q_kernel<scalar_t><<<q_grid, q_block, 0, at::cuda::getCurrentCUDAStream()>>>(
            q_contig.data_ptr<scalar_t>(),
            q_work.data_ptr<float>(),
            bsz * heads);
    });

    auto q_work_bf16 = q_work.view({bsz, heads, 128}).to(q.scalar_type());
    auto weights_score = weights.contiguous().view({bsz, 1, heads}).to(q.scalar_type());
    auto scores_bf16 = torch::empty({bsz, 1, std::max(kv_len, 1)}, q.options().dtype(q.scalar_type()));
    if (kv_len > 0) {
        auto logits = torch::bmm(q_work_bf16, kv_cache.narrow(0, 0, bsz).narrow(1, 0, kv_len).transpose(1, 2)).unsqueeze(1);
        scores_bf16 = (logits.relu() * weights_score.unsqueeze(-1)).sum(2);
    } else {
        scores_bf16.zero_();
    }

    if (return_scores) {
        return scores_bf16.to(torch::kFloat32);
    }
    auto topk = torch::empty({bsz, std::max(k, 1)}, q.options().dtype(torch::kInt64));
    if (k > 0) {
        auto scores_flat = scores_bf16.contiguous().view({bsz, kv_len});
        const size_t topk_shared_bytes = 256 * (sizeof(float) + sizeof(int));
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores_flat.scalar_type(), "c4_topk_from_fused_scores", [&] {
            c4_topk_kernel<scalar_t><<<bsz, 256, topk_shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                scores_flat.data_ptr<scalar_t>(),
                topk.data_ptr<int64_t>(),
                bsz,
                kv_len,
                k,
                static_cast<int>(offset));
        });
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return topk.view({bsz, 1, std::max(k, 1)});
}

// =====================================================================
// Fused attention decode prefuse kernels (Plan B-小-v2)
//
// Two opaque ops that replace the high-dispatch portion of
// Attention.forward (model.py:798-812) for bf16 / decode (seqlen=1) inputs.
//
//  1. fused_q_rmsnorm_rope_inplace(q, freqs_real, freqs_imag, eps)
//     Replaces:
//         q *= rsqrt(q.square().mean(-1, keepdim=True) + eps)
//         apply_rotary_emb(q[..., -rd:], freqs_cis)
//
//  2. fused_kv_rope_actquant_inplace(kv, kv_norm_weight,
//                                    freqs_real, freqs_imag,
//                                    block_size, norm_eps)
//     Replaces:
//         kv = self.kv_norm(kv)
//         apply_rotary_emb(kv[..., -rd:], freqs_cis)
//         act_quant(kv[..., :-rd], 64, "ue8m0", fp8_e8m0, inplace=True)
// =====================================================================

namespace {

constexpr float kFp8E4m3Max = 448.0f;
constexpr float kFp8E4m3Min = -448.0f;
constexpr float kBlockFp8Eps = 1e-4f;

__device__ __forceinline__ float bf16_to_float(c10::BFloat16 v) {
    return static_cast<float>(v);
}

__device__ __forceinline__ c10::BFloat16 float_to_bf16(float v) {
    return c10::BFloat16(v);
}

// Mirror _round_scale_pow2 in kernel.py: 2^ceil(log2(clamp(x, min=tiny))).
__device__ __forceinline__ float round_scale_pow2(float x) {
    constexpr float kTiny = 1.175494351e-38f;
    if (!(x > kTiny)) x = kTiny;
    int e;
    float m = frexpf(x, &e);  // x = m * 2^e, m in [0.5, 1)
    float candidate = ldexpf(1.0f, e - 1);
    if (candidate >= x) return candidate;
    return ldexpf(1.0f, e);
}

// Round-trip fp32 -> fp8 e4m3 (RNE, finite saturation at +/-448).
__device__ __forceinline__ float fp8_e4m3_round_trip(float v) {
    if (v > kFp8E4m3Max) v = kFp8E4m3Max;
    else if (v < kFp8E4m3Min) v = kFp8E4m3Min;
    if (v == 0.0f) return 0.0f;

    union { float f; uint32_t u; } in;
    in.f = v;
    uint32_t sign = in.u & 0x80000000u;
    uint32_t mag = in.u & 0x7fffffffu;
    int exp32 = static_cast<int>((mag >> 23) & 0xff);
    uint32_t mant23 = mag & 0x7fffffu;

    int unbiased = exp32 - 127;
    float result_mag;
    if (unbiased >= -6) {
        constexpr int shift = 23 - 3;  // 20
        uint32_t round_bit = (mant23 >> (shift - 1)) & 1u;
        uint32_t sticky = (mant23 & ((1u << (shift - 1)) - 1u)) != 0u ? 1u : 0u;
        uint32_t mant3 = mant23 >> shift;
        uint32_t lsb = mant3 & 1u;
        if (round_bit && (sticky || lsb)) {
            mant3 += 1u;
            if (mant3 == 8u) {
                mant3 = 0u;
                unbiased += 1;
            }
        }
        if (unbiased > 8) {
            result_mag = kFp8E4m3Max;
        } else {
            result_mag = ldexpf(1.0f + static_cast<float>(mant3) / 8.0f, unbiased);
        }
    } else if (unbiased >= -9) {
        uint32_t mant_full_bits = (1u << 23) | mant23;
        float mant_full = static_cast<float>(mant_full_bits) / static_cast<float>(1u << 23);
        float k_real = ldexpf(mant_full, unbiased + 9);
        float k_floor = floorf(k_real);
        float frac = k_real - k_floor;
        uint32_t k_int = static_cast<uint32_t>(k_floor);
        if (frac > 0.5f || (frac == 0.5f && (k_int & 1u))) k_int += 1u;
        if (k_int > 7u) k_int = 7u;
        result_mag = ldexpf(static_cast<float>(k_int), -9);
    } else {
        result_mag = 0.0f;
    }

    union { float f; uint32_t u; } out;
    out.f = result_mag;
    out.u |= sign;
    return out.f;
}

template <int kThreads>
__global__ void fused_q_rmsnorm_rope_kernel(
    c10::BFloat16* __restrict__ q,
    const float* __restrict__ freqs_real,
    const float* __restrict__ freqs_imag,
    int total_rows,
    int H,
    int S,
    int D,
    int rd,
    float eps) {
    const int row = blockIdx.x;
    if (row >= total_rows) return;
    const int s_idx = (row / H) % S;
    const int tid = threadIdx.x;

    extern __shared__ float smem_q[];
    float* qbuf = smem_q;
    float* aux = smem_q + D;

    c10::BFloat16* q_row = q + row * D;

    float local_sum = 0.0f;
    for (int i = tid; i < D; i += kThreads) {
        float v = bf16_to_float(q_row[i]);
        qbuf[i] = v;
        local_sum += v * v;
    }
    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) local_sum += __shfl_xor_sync(mask, local_sum, off);
    int lane = tid & 31;
    int warp = tid >> 5;
    constexpr int kWarps = kThreads / 32;
    if (lane == 0) aux[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < kWarps) ? aux[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) t += __shfl_xor_sync(mask, t, off);
        if (lane == 0) aux[0] = t;
    }
    __syncthreads();
    float mean = aux[0] / static_cast<float>(D);
    float scale = rsqrtf(mean + eps);

    for (int i = tid; i < D; i += kThreads) qbuf[i] *= scale;
    __syncthreads();

    const int rope_start = D - rd;
    const int npairs = rd >> 1;
    for (int p = tid; p < npairs; p += kThreads) {
        int re_idx = rope_start + 2 * p;
        int im_idx = re_idx + 1;
        float a = qbuf[re_idx];
        float b = qbuf[im_idx];
        float cr = freqs_real[s_idx * npairs + p];
        float ci = freqs_imag[s_idx * npairs + p];
        qbuf[re_idx] = a * cr - b * ci;
        qbuf[im_idx] = a * ci + b * cr;
    }
    __syncthreads();

    for (int i = tid; i < D; i += kThreads) q_row[i] = float_to_bf16(qbuf[i]);
}

template <int kThreads>
__global__ void fused_kv_rope_actquant_kernel(
    c10::BFloat16* __restrict__ kv,
    const float* __restrict__ norm_w,           // [kv_dim], fp32 (matches RMSNorm.weight)
    const float* __restrict__ freqs_real,
    const float* __restrict__ freqs_imag,
    c10::BFloat16* __restrict__ kv_cache_out,   // optional [B, cache_len, kv_dim] dest
    int cache_stride_batch,                     // kv_cache.stride(0) (== cache_len * kv_dim)
    int cache_slot,                             // slot inside cache (= start_pos % win)
    int batches_per_kvcache,                    // == B; row -> b = row / S
    int total_rows,
    int S,
    int kv_dim,
    int rd,
    int block_size,
    float norm_eps) {
    const int row = blockIdx.x;
    if (row >= total_rows) return;
    const int s_idx = row % S;
    const int tid = threadIdx.x;

    extern __shared__ float smem_kv[];
    float* buf = smem_kv;
    float* aux = smem_kv + kv_dim;
    constexpr int kWarps = kThreads / 32;
    float* block_amax = smem_kv + kv_dim + kWarps;

    c10::BFloat16* kv_row = kv + row * kv_dim;

    float local_sum = 0.0f;
    for (int i = tid; i < kv_dim; i += kThreads) {
        float v = bf16_to_float(kv_row[i]);
        buf[i] = v;
        local_sum += v * v;
    }
    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) local_sum += __shfl_xor_sync(mask, local_sum, off);
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) aux[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < kWarps) ? aux[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) t += __shfl_xor_sync(mask, t, off);
        if (lane == 0) aux[0] = t;
    }
    __syncthreads();
    float var = aux[0] / static_cast<float>(kv_dim);
    float inv_rms = rsqrtf(var + norm_eps);

    for (int i = tid; i < kv_dim; i += kThreads) {
        float w = norm_w[i];
        // Round-trip through bf16 to match RMSNorm.forward in model.py:RMSNorm,
        // which casts the (weight * x_normalized) result back to bf16 before
        // RoPE/act_quant see it.
        buf[i] = bf16_to_float(float_to_bf16((buf[i] * inv_rms) * w));
    }
    __syncthreads();

    const int rope_start = kv_dim - rd;
    const int npairs = rd >> 1;
    for (int p = tid; p < npairs; p += kThreads) {
        int re_idx = rope_start + 2 * p;
        int im_idx = re_idx + 1;
        float a = buf[re_idx];
        float b = buf[im_idx];
        float cr = freqs_real[s_idx * npairs + p];
        float ci = freqs_imag[s_idx * npairs + p];
        // Match apply_rotary_emb: complex multiply in fp32 then bf16 round-trip on copy_.
        buf[re_idx] = bf16_to_float(float_to_bf16(a * cr - b * ci));
        buf[im_idx] = bf16_to_float(float_to_bf16(a * ci + b * cr));
    }
    __syncthreads();

    const int q_dim = kv_dim - rd;
    const int n_blocks = q_dim / block_size;
    for (int b = tid; b < n_blocks; b += kThreads) block_amax[b] = 0.0f;
    __syncthreads();

    for (int i = tid; i < q_dim; i += kThreads) {
        float v = buf[i];
        float a = fabsf(v);
        int b = i / block_size;
        atomicMax(reinterpret_cast<int*>(&block_amax[b]), __float_as_int(a));
    }
    __syncthreads();

    for (int i = tid; i < q_dim; i += kThreads) {
        int b = i / block_size;
        float amax = block_amax[b];
        if (amax < kBlockFp8Eps) amax = kBlockFp8Eps;
        float scale = round_scale_pow2(amax / kFp8E4m3Max);
        float v = buf[i] / scale;
        if (v > kFp8E4m3Max) v = kFp8E4m3Max;
        else if (v < kFp8E4m3Min) v = kFp8E4m3Min;
        float qv = fp8_e4m3_round_trip(v);
        buf[i] = qv * scale;
    }
    __syncthreads();

    for (int i = tid; i < kv_dim; i += kThreads) kv_row[i] = float_to_bf16(buf[i]);

    // Optional: also write the produced row into kv_cache_out[b, cache_slot, :].
    // Decode path uses S == 1 and a single slot per batch, so we just pick row->b
    // and write to the same offset inside the cache buffer. This eliminates the
    // Python-side `self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)` and its
    // 80+ select/copy_/as_strided dispatcher ops per layer per step.
    if (kv_cache_out != nullptr) {
        const int b = row / S;
        c10::BFloat16* cache_row =
            kv_cache_out + b * cache_stride_batch + cache_slot * kv_dim;
        for (int i = tid; i < kv_dim; i += kThreads) cache_row[i] = float_to_bf16(buf[i]);
    }
}

}  // namespace (fused attn prefuse)

void fused_q_rmsnorm_rope_inplace_cuda(
    torch::Tensor& q,
    const torch::Tensor& freqs_real,
    const torch::Tensor& freqs_imag,
    double eps) {
    c10::cuda::CUDAGuard device_guard(q.device());
    const int B = static_cast<int>(q.size(0));
    const int S = static_cast<int>(q.size(1));
    const int H = static_cast<int>(q.size(2));
    const int D = static_cast<int>(q.size(3));
    const int rd = static_cast<int>(freqs_real.size(-1)) * 2;
    const int total_rows = B * S * H;
    constexpr int kThreads = 128;
    constexpr int kWarps = kThreads / 32;
    const size_t smem_bytes = sizeof(float) * (D + kWarps);
    fused_q_rmsnorm_rope_kernel<kThreads>
        <<<total_rows, kThreads, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<c10::BFloat16>(),
            freqs_real.data_ptr<float>(),
            freqs_imag.data_ptr<float>(),
            total_rows, H, S, D, rd, static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_kv_rope_actquant_inplace_cuda(
    torch::Tensor& kv,
    const torch::Tensor& norm_weight,
    const torch::Tensor& freqs_real,
    const torch::Tensor& freqs_imag,
    int64_t block_size,
    double norm_eps,
    const c10::optional<torch::Tensor>& kv_cache_out,
    int64_t cache_slot) {
    c10::cuda::CUDAGuard device_guard(kv.device());
    const int B = static_cast<int>(kv.size(0));
    const int S = static_cast<int>(kv.size(1));
    const int kv_dim = static_cast<int>(kv.size(2));
    const int rd = static_cast<int>(freqs_real.size(-1)) * 2;
    const int total_rows = B * S;
    constexpr int kThreads = 128;
    constexpr int kWarps = kThreads / 32;
    const int q_dim = kv_dim - rd;
    const int n_blocks = q_dim / static_cast<int>(block_size);
    const size_t smem_bytes = sizeof(float) * (kv_dim + kWarps + n_blocks);

    c10::BFloat16* cache_ptr = nullptr;
    int cache_stride_batch = 0;
    int slot = 0;
    if (kv_cache_out.has_value() && kv_cache_out->defined()) {
        const auto& cache = *kv_cache_out;
        cache_ptr = cache.data_ptr<c10::BFloat16>();
        cache_stride_batch = static_cast<int>(cache.stride(0));
        slot = static_cast<int>(cache_slot);
    }

    fused_kv_rope_actquant_kernel<kThreads>
        <<<total_rows, kThreads, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            kv.data_ptr<c10::BFloat16>(),
            norm_weight.data_ptr<float>(),
            freqs_real.data_ptr<float>(),
            freqs_imag.data_ptr<float>(),
            cache_ptr,
            cache_stride_batch,
            slot,
            B,
            total_rows, S, kv_dim, rd,
            static_cast<int>(block_size),
            static_cast<float>(norm_eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// =====================================================================
// cuBLAS IMMA INT8 GEMM (Plan #1)
//
// Replaces int8_gemm_rows_kernel for the per-token decode INT8 GEMMs
// (wq_a / wq_b / wkv / wo_b / indexer.wq_b / shared_expert pair). cuBLAS
// IMMA on Turing SM75 needs M/N/K aligned to 16; all decode shapes here
// satisfy that. Output is dequantized straight to bf16, which also
// removes the outer .to(bf16) the Python caller used to do after the
// dp4a path returned fp32.
//
// quantize_rows_kernel above already produces x_q [rows, K] int8 +
// x_scale [rows] fp32; we reuse it. cuBLAS produces int32 [rows, N];
// then dequant_int32_to_bf16_kernel multiplies by x_scale[row] *
// weight_s[col] and writes bf16.
// =====================================================================

namespace {

__global__ void dequant_int32_to_bf16_kernel(
    const int32_t* __restrict__ acc,        // [rows, n]
    const float* __restrict__ x_scale,      // [rows]
    const float* __restrict__ weight_s,     // [n]
    c10::BFloat16* __restrict__ out,        // [rows, n]
    int rows,
    int n) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n || row >= rows) return;
    const float xs = x_scale[row];
    const float ws = weight_s[col];
    const float v = static_cast<float>(acc[row * n + col]) * xs * ws;
    out[row * n + col] = c10::BFloat16(v);
}

}  // namespace

torch::Tensor int8_gemm_imma_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_s) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto x_contig = x.contiguous();
    auto wq_contig = weight_q.contiguous();
    auto ws_contig = weight_s.contiguous();

    const auto k = static_cast<int>(x_contig.size(-1));
    const auto n = static_cast<int>(weight_q.size(0));
    const auto rows = static_cast<int>(x_contig.numel() / k);

    auto x_q = torch::empty({rows, k}, x.options().dtype(torch::kInt8));
    auto x_scale = torch::empty({rows}, x.options().dtype(torch::kFloat32));
    auto acc = torch::empty({rows, n}, x.options().dtype(torch::kInt32));

    auto stream = at::cuda::getCurrentCUDAStream();

    const dim3 quant_grid(rows, 1);
    const dim3 quant_block(kQuantThreads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "imma_quantize", [&] {
        quantize_rows_kernel<scalar_t><<<quant_grid, quant_block, 0, stream>>>(
            x_contig.data_ptr<scalar_t>(),
            x_q.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(),
            rows,
            1,
            k);
    });

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, stream);
    const int alpha = 1;
    const int beta = 0;
    // cuBLAS is column-major. We want C [rows, n] = X [rows, k] * W^T [k, n]
    // with X row-major and W stored row-major as [N, K]. Treat both as
    // column-major transposed: compute (W * X^T)^T = X * W^T by issuing
    //   GEMM(op_a=T, op_b=N, m=n, n=rows, k=k, A=W [n,k]^T row-major == col-major [k,n]^T,
    //        B=X^T row-major == col-major [k,rows], ldc=n).
    // The simpler/equivalent formulation used elsewhere in this file is
    //   cublasGemmEx(OP_T, OP_N, n, m=rows, k, A=W row-major, lda=k,
    //                B=X row-major (but logically X^T col-major), ldb=k, ldc=n).
    auto status = cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n,
        rows,
        k,
        &alpha,
        wq_contig.data_ptr<int8_t>(),
        CUDA_R_8I,
        k,
        x_q.data_ptr<int8_t>(),
        CUDA_R_8I,
        k,
        &beta,
        acc.data_ptr<int32_t>(),
        CUDA_R_32I,
        n,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "int8_gemm_imma cublasGemmEx failed: ", status);

    auto out = torch::empty({rows, n}, x.options().dtype(torch::kBFloat16));
    constexpr int kThreads = 128;
    const dim3 deq_grid(ceil_div(n, kThreads), rows);
    const dim3 deq_block(kThreads);
    dequant_int32_to_bf16_kernel<<<deq_grid, deq_block, 0, stream>>>(
        acc.data_ptr<int32_t>(),
        x_scale.data_ptr<float>(),
        ws_contig.data_ptr<float>(),
        out.data_ptr<c10::BFloat16>(),
        rows,
        n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto out_shape = x.sizes().vec();
    out_shape.back() = n;
    return out.view(out_shape);
}

torch::Tensor int8_gemm_pair_imma_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_q0,
    const torch::Tensor& weight_s0,
    const torch::Tensor& weight_q1,
    const torch::Tensor& weight_s1) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto x_contig = x.contiguous();
    auto wq0_contig = weight_q0.contiguous();
    auto ws0_contig = weight_s0.contiguous();
    auto wq1_contig = weight_q1.contiguous();
    auto ws1_contig = weight_s1.contiguous();

    const auto k = static_cast<int>(x_contig.size(-1));
    const auto n = static_cast<int>(weight_q0.size(0));
    const auto rows = static_cast<int>(x_contig.numel() / k);

    auto x_q = torch::empty({rows, k}, x.options().dtype(torch::kInt8));
    auto x_scale = torch::empty({rows}, x.options().dtype(torch::kFloat32));
    auto acc0 = torch::empty({rows, n}, x.options().dtype(torch::kInt32));
    auto acc1 = torch::empty({rows, n}, x.options().dtype(torch::kInt32));

    auto stream = at::cuda::getCurrentCUDAStream();

    const dim3 quant_grid(rows, 1);
    const dim3 quant_block(kQuantThreads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "imma_pair_quantize", [&] {
        quantize_rows_kernel<scalar_t><<<quant_grid, quant_block, 0, stream>>>(
            x_contig.data_ptr<scalar_t>(),
            x_q.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(),
            rows,
            1,
            k);
    });

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, stream);
    const int alpha = 1;
    const int beta = 0;

    auto run_gemm = [&](int8_t* w_ptr, int32_t* out_ptr) {
        auto status = cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            n,
            rows,
            k,
            &alpha,
            w_ptr,
            CUDA_R_8I,
            k,
            x_q.data_ptr<int8_t>(),
            CUDA_R_8I,
            k,
            &beta,
            out_ptr,
            CUDA_R_32I,
            n,
            CUBLAS_COMPUTE_32I,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "int8_gemm_pair_imma cublasGemmEx failed: ", status);
    };
    run_gemm(wq0_contig.data_ptr<int8_t>(), acc0.data_ptr<int32_t>());
    run_gemm(wq1_contig.data_ptr<int8_t>(), acc1.data_ptr<int32_t>());

    auto out_shape = x.sizes().vec();
    out_shape.back() = n;
    auto out0 = torch::empty({rows, n}, x.options().dtype(torch::kBFloat16));
    auto out1 = torch::empty({rows, n}, x.options().dtype(torch::kBFloat16));

    constexpr int kThreads = 128;
    const dim3 deq_grid(ceil_div(n, kThreads), rows);
    const dim3 deq_block(kThreads);
    dequant_int32_to_bf16_kernel<<<deq_grid, deq_block, 0, stream>>>(
        acc0.data_ptr<int32_t>(),
        x_scale.data_ptr<float>(),
        ws0_contig.data_ptr<float>(),
        out0.data_ptr<c10::BFloat16>(),
        rows,
        n);
    dequant_int32_to_bf16_kernel<<<deq_grid, deq_block, 0, stream>>>(
        acc1.data_ptr<int32_t>(),
        x_scale.data_ptr<float>(),
        ws1_contig.data_ptr<float>(),
        out1.data_ptr<c10::BFloat16>(),
        rows,
        n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return torch::stack({out0.view(out_shape), out1.view(out_shape)}, 0);
}

// =====================================================================
// Inverse RoPE on the trailing rd dims of o [B, S, H, D] (Plan #2).
//
// Replaces apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True) which
// is 6 PyTorch dispatches per layer (unflatten/view_as_complex/conj/mul/
// view_as_real/flatten + copy_). Inverse rope uses conj(freqs_cis), i.e.
// (a + bi) * (cr - ci*i) = (a*cr + b*ci) + (b*cr - a*ci) i.
//
// One block per (token, head); blockDim covers rd/2 pairs.
// =====================================================================

namespace {

template <int kThreads>
__global__ void fused_o_inverse_rope_kernel(
    c10::BFloat16* __restrict__ o,
    const float* __restrict__ freqs_real,
    const float* __restrict__ freqs_imag,
    int total_rows,
    int H,
    int S,
    int D,
    int rd) {
    const int row = blockIdx.x;
    if (row >= total_rows) return;
    const int s_idx = (row / H) % S;
    const int tid = threadIdx.x;

    c10::BFloat16* o_row = o + row * D;
    const int rope_start = D - rd;
    const int npairs = rd >> 1;
    for (int p = tid; p < npairs; p += kThreads) {
        int re_idx = rope_start + 2 * p;
        int im_idx = re_idx + 1;
        float a = static_cast<float>(o_row[re_idx]);
        float b = static_cast<float>(o_row[im_idx]);
        float cr = freqs_real[s_idx * npairs + p];
        float ci = freqs_imag[s_idx * npairs + p];
        // inverse rope == multiply by conjugate(freqs):
        // (a + bi)(cr - ci*i) = (a*cr + b*ci) + (b*cr - a*ci) i.
        // Match apply_rotary_emb: complex multiply in fp32 then bf16
        // round-trip on copy_.
        o_row[re_idx] = c10::BFloat16(a * cr + b * ci);
        o_row[im_idx] = c10::BFloat16(b * cr - a * ci);
    }
}

}  // namespace (inverse rope)

void fused_o_inverse_rope_inplace_cuda(
    torch::Tensor& o,
    const torch::Tensor& freqs_real,
    const torch::Tensor& freqs_imag) {
    c10::cuda::CUDAGuard device_guard(o.device());
    const int B = static_cast<int>(o.size(0));
    const int S = static_cast<int>(o.size(1));
    const int H = static_cast<int>(o.size(2));
    const int D = static_cast<int>(o.size(3));
    const int rd = static_cast<int>(freqs_real.size(-1)) * 2;
    const int total_rows = B * S * H;
    constexpr int kThreads = 32;
    fused_o_inverse_rope_kernel<kThreads>
        <<<total_rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            o.data_ptr<c10::BFloat16>(),
            freqs_real.data_ptr<float>(),
            freqs_imag.data_ptr<float>(),
            total_rows, H, S, D, rd);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
