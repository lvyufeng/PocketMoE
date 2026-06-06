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
#include <climits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <atomic>

namespace {

constexpr int kQuantThreads = 256;
constexpr int kGemmThreads = 128;
constexpr int kAttnThreads = 256;
constexpr int kCustomAllreduceWorldSize = 4;

struct CustomAllreduceHandle {
    int rank = 0;
    int world_size = 0;
    int dim = 0;
    at::ScalarType dtype = at::kFloat;
    void* peer_data[kCustomAllreduceWorldSize] = {nullptr, nullptr, nullptr, nullptr};
    int32_t* peer_flags[kCustomAllreduceWorldSize] = {nullptr, nullptr, nullptr, nullptr};
};

std::mutex g_custom_allreduce_mutex;
std::unordered_map<int64_t, CustomAllreduceHandle> g_custom_allreduce_handles;
int64_t g_custom_allreduce_next_handle = 1;
std::atomic<int> g_gguf_prefill_profile_count{0};

int env_int_default(const char* name, int default_value) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return default_value;
    return std::atoi(v);
}


int custom_dtype_code(at::ScalarType dtype) {
    if (dtype == at::kBFloat16) return 1;
    if (dtype == at::kHalf) return 2;
    if (dtype == at::kFloat) return 3;
    return 0;
}

void check_cuda(cudaError_t err, const char* msg) {
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

bool env_enabled_explicit(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return false;
    return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
           std::strcmp(v, "True") == 0 || std::strcmp(v, "yes") == 0 ||
           std::strcmp(v, "on") == 0;
}

bool c4_indexer_score_cuda_enabled() {
    return env_enabled_explicit("DEEPSEEK_FUSED_C4_INDEXER_SCORE_CUDA");
}

bool c4_indexer_post_gemm_enabled() {
    return env_enabled_explicit("DEEPSEEK_C4_INDEXER_FUSED_POST_GEMM");
}

bool c4_topk_tile_merge_enabled() {
    return env_enabled_explicit("DEEPSEEK_C4_TOPK_TILE_MERGE_CUDA");
}


__device__ __forceinline__ bool c4_pair_better(float a_val, int a_idx, float b_val, int b_idx) {
    return a_val > b_val || (a_val == b_val && a_idx < b_idx);
}

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
__global__ void route_swiglu_cast_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    const float* __restrict__ route_weights,
    float* __restrict__ out,
    int rows,
    int cols,
    float swiglu_limit) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;
    const int row = idx / cols;
    float g = static_cast<float>(gate[idx]);
    float u = static_cast<float>(up[idx]);
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

template <typename scalar_t>
__global__ void flashinfer_style_sparse_attn_headpair_kernel(
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
    const int pair_count = (heads + 1) >> 1;
    const int bp = blockIdx.x;
    const int b = bp / pair_count;
    const int pair = bp - b * pair_count;
    const int h0 = pair << 1;
    const int h1 = h0 + 1;
    const bool has_h1 = h1 < heads;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int n_warps = blockDim.x >> 5;

    extern __shared__ float smem[];
    float* scores0 = smem;
    float* scores1 = scores0 + topk;
    float* q0_shared = scores1 + topk;
    float* q1_shared = q0_shared + dim;
    int32_t* idx_shared = reinterpret_cast<int32_t*>(q1_shared + dim);
    float* reduce0 = reinterpret_cast<float*>(idx_shared + topk);
    float* reduce1 = reduce0 + blockDim.x;

    const scalar_t* q0_ptr = q + ((b * heads + h0) * dim);
    const scalar_t* q1_ptr = has_h1 ? q + ((b * heads + h1) * dim) : q0_ptr;
    const scalar_t* kv_base = kv + b * kv_len * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        q0_shared[d] = static_cast<float>(q0_ptr[d]);
        q1_shared[d] = has_h1 ? static_cast<float>(q1_ptr[d]) : 0.0f;
    }
    for (int t = tid; t < topk; t += blockDim.x) {
        idx_shared[t] = topk_idxs[b * topk + t];
    }
    __syncthreads();

    for (int t = warp_id; t < topk; t += n_warps) {
        const int idx = idx_shared[t];
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        if (idx >= 0 && idx < kv_len) {
            const scalar_t* kv_ptr = kv_base + idx * dim;
            for (int d = lane; d < dim; d += 32) {
                const float kvv = static_cast<float>(kv_ptr[d]);
                acc0 += q0_shared[d] * kvv;
                if (has_h1) acc1 += q1_shared[d] * kvv;
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc0 += __shfl_down_sync(0xffffffff, acc0, offset);
                acc1 += __shfl_down_sync(0xffffffff, acc1, offset);
            }
            if (lane == 0) {
                scores0[t] = acc0 * softmax_scale;
                scores1[t] = has_h1 ? acc1 * softmax_scale : -INFINITY;
            }
        } else if (lane == 0) {
            scores0[t] = -INFINITY;
            scores1[t] = -INFINITY;
        }
    }
    __syncthreads();

    float local_max0 = -INFINITY;
    float local_max1 = -INFINITY;
    for (int t = tid; t < topk; t += blockDim.x) {
        local_max0 = fmaxf(local_max0, scores0[t]);
        local_max1 = fmaxf(local_max1, scores1[t]);
    }
    reduce0[tid] = local_max0;
    reduce1[tid] = local_max1;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce0[tid] = fmaxf(reduce0[tid], reduce0[tid + stride]);
            reduce1[tid] = fmaxf(reduce1[tid], reduce1[tid + stride]);
        }
        __syncthreads();
    }
    const float max0 = fmaxf(reduce0[0], attn_sink[h0]);
    const float max1 = has_h1 ? fmaxf(reduce1[0], attn_sink[h1]) : -INFINITY;

    float denom0_local = 0.0f;
    float denom1_local = 0.0f;
    for (int t = tid; t < topk; t += blockDim.x) {
        const float w0 = expf(scores0[t] - max0);
        scores0[t] = w0;
        denom0_local += w0;
        if (has_h1) {
            const float w1 = expf(scores1[t] - max1);
            scores1[t] = w1;
            denom1_local += w1;
        }
    }
    if (tid == 0) {
        denom0_local += expf(attn_sink[h0] - max0);
        if (has_h1) denom1_local += expf(attn_sink[h1] - max1);
    }
    reduce0[tid] = denom0_local;
    reduce1[tid] = denom1_local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce0[tid] += reduce0[tid + stride];
            reduce1[tid] += reduce1[tid + stride];
        }
        __syncthreads();
    }
    const float denom0 = reduce0[0];
    const float denom1 = reduce1[0];

    scalar_t* out0 = out + ((b * heads + h0) * dim);
    scalar_t* out1 = has_h1 ? out + ((b * heads + h1) * dim) : out0;
    for (int d = tid; d < dim; d += blockDim.x) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (int t = 0; t < topk; ++t) {
            const int idx = idx_shared[t];
            if (idx >= 0 && idx < kv_len) {
                const float kvv = static_cast<float>(kv_base[idx * dim + d]);
                acc0 += scores0[t] * kvv;
                if (has_h1) acc1 += scores1[t] * kvv;
            }
        }
        out0[d] = static_cast<scalar_t>(acc0 / denom0);
        if (has_h1) out1[d] = static_cast<scalar_t>(acc1 / denom1);
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

template <typename scalar_t>
__global__ void c4_score_kernel(
    const float* __restrict__ q_work,
    const scalar_t* __restrict__ kv_cache,
    const float* __restrict__ weights,
    float* __restrict__ scores,
    int bsz,
    int heads,
    int max_cache_len,
    int kv_len) {
    const int b = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;
    if (b >= bsz || t >= kv_len) return;
    __shared__ float reduce[256];
    float total = 0.0f;
    const scalar_t* kv_row = kv_cache + (static_cast<int64_t>(b) * max_cache_len + t) * 128;
    for (int h = 0; h < heads; ++h) {
        float partial = 0.0f;
        if (tid < 128) {
            const float qv = q_work[(static_cast<int64_t>(b) * heads + h) * 128 + tid];
            const float kv = static_cast<float>(kv_row[tid]);
            partial = qv * kv;
        }
        reduce[tid] = partial;
        __syncthreads();
        for (int stride = 128 >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            const float dot = static_cast<float>(static_cast<scalar_t>(reduce[0]));
            if (dot > 0.0f) {
                const float weighted = dot * static_cast<float>(static_cast<scalar_t>(weights[b * heads + h]));
                total = static_cast<float>(static_cast<scalar_t>(total + weighted));
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        scores[b * kv_len + t] = static_cast<float>(static_cast<scalar_t>(total));
    }
}

template <typename scalar_t, typename weight_t>
__global__ void c4_post_gemm_score_kernel(
    const scalar_t* __restrict__ logits,
    const weight_t* __restrict__ weights,
    float* __restrict__ scores,
    int bsz,
    int heads,
    int kv_len) {
    const int b = blockIdx.x;
    const int t = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= bsz || t >= kv_len) return;
    float acc = 0.0f;
    for (int h = 0; h < heads; ++h) {
        const float v = static_cast<float>(logits[(static_cast<int64_t>(b) * heads + h) * kv_len + t]);
        if (v > 0.0f) {
            acc += v * static_cast<float>(weights[b * heads + h]);
        }
    }
    scores[b * kv_len + t] = acc;
}

constexpr int kC4TopkTileSize = 1024;

template <typename score_t>
__global__ void c4_topk_tile_candidates_kernel(
    const score_t* __restrict__ scores,
    float* __restrict__ cand_vals,
    int32_t* __restrict__ cand_idxs,
    int bsz,
    int kv_len,
    int k,
    int emit_per_tile,
    int num_tiles) {
    const int b = blockIdx.x;
    const int tile = blockIdx.y;
    const int tid = threadIdx.x;
    if (b >= bsz || tile >= num_tiles) return;
    __shared__ float vals[kC4TopkTileSize];
    __shared__ int idxs[kC4TopkTileSize];
    const int tile_start = tile * kC4TopkTileSize;
    for (int i = tid; i < kC4TopkTileSize; i += blockDim.x) {
        const int idx = tile_start + i;
        if (idx < kv_len) {
            vals[i] = static_cast<float>(scores[static_cast<int64_t>(b) * kv_len + idx]);
            idxs[i] = idx;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = INT_MAX;
        }
    }
    __syncthreads();

    for (int size = 2; size <= kC4TopkTileSize; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < kC4TopkTileSize; i += blockDim.x) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool ascending = (i & size) != 0;
                    const float vi = vals[i];
                    const int ii = idxs[i];
                    const float vj = vals[j];
                    const int ij = idxs[j];
                    const bool j_better = c4_pair_better(vj, ij, vi, ii);
                    if ((!ascending && j_better) || (ascending && !j_better)) {
                        vals[i] = vj;
                        idxs[i] = ij;
                        vals[j] = vi;
                        idxs[j] = ii;
                    }
                }
            }
            __syncthreads();
        }
    }

    const int actual_tile_len = max(0, min(kC4TopkTileSize, kv_len - tile_start));
    const int emit_count = min(k, actual_tile_len);
    const int64_t base = (static_cast<int64_t>(b) * num_tiles + tile) * emit_per_tile;
    for (int i = tid; i < emit_per_tile; i += blockDim.x) {
        if (i < emit_count) {
            cand_vals[base + i] = vals[i];
            cand_idxs[base + i] = idxs[i];
        } else {
            cand_vals[base + i] = -INFINITY;
            cand_idxs[base + i] = INT_MAX;
        }
    }
}

template <typename score_t>
__global__ void c4_topk_single_tile_sort_kernel(
    const score_t* __restrict__ scores,
    int64_t* __restrict__ topk,
    int bsz,
    int kv_len,
    int k,
    int offset) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= bsz) return;
    __shared__ float vals[kC4TopkTileSize];
    __shared__ int idxs[kC4TopkTileSize];
    for (int i = tid; i < kC4TopkTileSize; i += blockDim.x) {
        if (i < kv_len) {
            vals[i] = static_cast<float>(scores[static_cast<int64_t>(b) * kv_len + i]);
            idxs[i] = i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = INT_MAX;
        }
    }
    __syncthreads();

    for (int size = 2; size <= kC4TopkTileSize; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < kC4TopkTileSize; i += blockDim.x) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool ascending = (i & size) != 0;
                    const float vi = vals[i];
                    const int ii = idxs[i];
                    const float vj = vals[j];
                    const int ij = idxs[j];
                    const bool j_better = c4_pair_better(vj, ij, vi, ii);
                    if ((!ascending && j_better) || (ascending && !j_better)) {
                        vals[i] = vj;
                        idxs[i] = ij;
                        vals[j] = vi;
                        idxs[j] = ii;
                    }
                }
            }
            __syncthreads();
        }
    }

    int64_t* out = topk + static_cast<int64_t>(b) * k;
    for (int i = tid; i < k; i += blockDim.x) {
        out[i] = static_cast<int64_t>(idxs[i] + offset);
    }
}


__global__ void c4_topk_merge_candidates_kernel(
    const float* __restrict__ cand_vals,
    const int32_t* __restrict__ cand_idxs,
    int64_t* __restrict__ topk,
    int bsz,
    int k,
    int emit_per_tile,
    int num_tiles,
    int offset) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= bsz) return;
    extern __shared__ char smem_raw[];
    int* pos = reinterpret_cast<int*>(smem_raw);
    float* vals = reinterpret_cast<float*>(pos + num_tiles);
    int* idxs = reinterpret_cast<int*>(vals + blockDim.x);
    int* tile_ids = idxs + blockDim.x;
    for (int tile = tid; tile < num_tiles; tile += blockDim.x) {
        pos[tile] = 0;
    }
    __syncthreads();
    const int64_t cand_base = static_cast<int64_t>(b) * num_tiles * emit_per_tile;
    int64_t* out = topk + static_cast<int64_t>(b) * k;
    for (int out_i = 0; out_i < k; ++out_i) {
        float best_v = -INFINITY;
        int best_idx = INT_MAX;
        int best_tile = INT_MAX;
        for (int tile = tid; tile < num_tiles; tile += blockDim.x) {
            const int p = pos[tile];
            if (p < emit_per_tile) {
                const int64_t ci = cand_base + static_cast<int64_t>(tile) * emit_per_tile + p;
                const float v = cand_vals[ci];
                const int idx = cand_idxs[ci];
                if (c4_pair_better(v, idx, best_v, best_idx)) {
                    best_v = v;
                    best_idx = idx;
                    best_tile = tile;
                }
            }
        }
        vals[tid] = best_v;
        idxs[tid] = best_idx;
        tile_ids[tid] = best_tile;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float ov = vals[tid + stride];
                const int oi = idxs[tid + stride];
                if (c4_pair_better(ov, oi, vals[tid], idxs[tid])) {
                    vals[tid] = ov;
                    idxs[tid] = oi;
                    tile_ids[tid] = tile_ids[tid + stride];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            out[out_i] = static_cast<int64_t>(idxs[0] + offset);
            if (tile_ids[0] >= 0 && tile_ids[0] < num_tiles) {
                pos[tile_ids[0]] += 1;
            }
        }
        __syncthreads();
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

template <typename score_t>
void launch_c4_topk_kernel_or_tile_merge(
    score_t* scores_flat,
    int64_t* topk,
    int bsz,
    int kv_len,
    int k,
    int offset,
    const torch::TensorOptions& options) {
    const bool use_tile = c4_topk_tile_merge_enabled() && k > 0;
    const int num_tiles = k > 0 ? (kv_len + kC4TopkTileSize - 1) / kC4TopkTileSize : 0;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (!use_tile) {
        const size_t topk_shared_bytes = 256 * (sizeof(float) + sizeof(int));
        c4_topk_kernel<score_t><<<bsz, 256, topk_shared_bytes, stream>>>(
            scores_flat,
            topk,
            bsz,
            kv_len,
            k,
            offset);
    } else if (num_tiles == 1) {
        c4_topk_single_tile_sort_kernel<score_t><<<bsz, 256, 0, stream>>>(
            scores_flat,
            topk,
            bsz,
            kv_len,
            k,
            offset);
    } else {
        const int emit_per_tile = std::min(k, kC4TopkTileSize);
        auto cand_vals = torch::empty({bsz, num_tiles, emit_per_tile}, options.dtype(torch::kFloat32));
        auto cand_idxs = torch::empty({bsz, num_tiles, emit_per_tile}, options.dtype(torch::kInt32));
        const dim3 cand_grid(bsz, num_tiles);
        const dim3 cand_block(256);
        c4_topk_tile_candidates_kernel<score_t><<<cand_grid, cand_block, 0, stream>>>(
            scores_flat,
            cand_vals.data_ptr<float>(),
            cand_idxs.data_ptr<int32_t>(),
            bsz,
            kv_len,
            k,
            emit_per_tile,
            num_tiles);
        const dim3 merge_grid(bsz);
        const dim3 merge_block(256);
        const size_t merge_shared_bytes = static_cast<size_t>(num_tiles) * sizeof(int) +
            static_cast<size_t>(merge_block.x) * (sizeof(float) + sizeof(int) + sizeof(int));
        c4_topk_merge_candidates_kernel<<<merge_grid, merge_block, merge_shared_bytes, stream>>>(
            cand_vals.data_ptr<float>(),
            cand_idxs.data_ptr<int32_t>(),
            topk,
            bsz,
            k,
            emit_per_tile,
            num_tiles,
            offset);
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
// FP4 single-token MoE: routed expert weights stay packed FP4 (e2m1fn_x2)
// + e8m0 per-32 block scales all the way to GPU. Activations are int8 like the
// int8 path; weight nibbles unpack into int8 values multiplied by 2 (FP4 e2m1
// levels are {0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3, +/-4, +/-6}, all in [-12,12]
// after *2). __dp4a accumulates over 32-K blocks; each block's int32 acc is
// scaled by 2^(scale_byte - 128) (the -1 absorbs the LUT *2). The w1w3 kernel
// emits fp32 (already weight-scaled), unlike the int8 v1 kernel which emits
// int32 and scales in the SwiGLU stage.
// ============================================================================

__device__ __constant__ int8_t fp4_lut_x2[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12
};

__device__ __forceinline__ void fp4_init_byte_lut(uint16_t* lut_pair, int tid, int block_threads) {
    // Each entry maps a byte (= two FP4 nibbles) to a uint16 holding the two
    // dequantized int8 values (already *2 from the e2m1 LUT) in low/high byte.
    // Lives in shared memory: byte LDS broadcasts and avoids the warp-level
    // serialization that __constant__ would suffer on divergent nibble values.
    for (int i = tid; i < 256; i += block_threads) {
        const uint32_t b = static_cast<uint32_t>(i);
        const int8_t lo = fp4_lut_x2[b & 0xF];
        const int8_t hi = fp4_lut_x2[(b >> 4) & 0xF];
        lut_pair[i] = static_cast<uint16_t>(
            (static_cast<uint16_t>(static_cast<uint8_t>(lo)) << 0) |
            (static_cast<uint16_t>(static_cast<uint8_t>(hi)) << 8));
    }
}

__device__ __forceinline__ int fp4_unpack_4codes_via_lut(const uint16_t* lut_pair,
                                                         uint32_t two_bytes) {
    // two_bytes: low 16 bits hold weight byte0 (K0,K1) and byte1 (K2,K3).
    // Returns int8x4 [K0, K1, K2, K3] suitable for __dp4a (signed int8 lanes).
    const uint32_t pair0 = lut_pair[two_bytes & 0xFFu];
    const uint32_t pair1 = lut_pair[(two_bytes >> 8) & 0xFFu];
    return static_cast<int>(pair0 | (pair1 << 16));
}

__device__ __forceinline__ int fp4_unpack_4codes_prmt(uint32_t two_bytes) {
    const uint32_t control_mags = two_bytes & 0x7777u;
    const uint32_t pos = __byte_perm(0x03020100u, 0x0c080604u, control_mags);
    const uint32_t neg = __byte_perm(0xfdfeff00u, 0xf4f8fafcu, control_mags);
    const uint32_t control_sign = (two_bytes >> 3) & 0x1111u;
    const uint32_t mask = __byte_perm(0x0000ff00u, 0x00000000u, control_sign);
    return static_cast<int>((pos & ~mask) | (neg & mask));
}

__device__ __forceinline__ float fp4_block_scale(uint8_t scale_byte) {
    const int exponent = max(0, static_cast<int>(scale_byte) - 1);
    return __int_as_float(exponent << 23);
}

__device__ __forceinline__ float fp4_value_from_code_scale(uint8_t code, uint8_t scale_byte) {
    return static_cast<float>(fp4_lut_x2[code & 0x0f]) * fp4_block_scale(scale_byte);
}

__global__ void fp4_weight_to_int8_row_kernel(
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ ws,
    int8_t* __restrict__ out_q,
    float* __restrict__ out_s,
    int rows,
    int k) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;
    const int k_half = k / 2;
    const int k_blocks = k / 32;
    const uint8_t* row_q = wq + static_cast<int64_t>(row) * k_half;
    const uint8_t* row_s = ws + static_cast<int64_t>(row) * k_blocks;
    int8_t* dst_q = out_q + static_cast<int64_t>(row) * k;
    __shared__ float sdata[kQuantThreads];
    float local_max = 0.0f;
    for (int idx = tid; idx < k; idx += blockDim.x) {
        const uint8_t packed = row_q[idx >> 1];
        const uint8_t code = (idx & 1) ? (packed >> 4) : (packed & 0x0f);
        const float v = fp4_value_from_code_scale(code, row_s[idx >> 5]);
        local_max = fmaxf(local_max, fabsf(v));
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    const float scale = fmaxf(sdata[0], 1.0e-6f) / 127.0f;
    if (tid == 0) out_s[row] = scale;
    const float inv_scale = 1.0f / scale;
    for (int idx = tid; idx < k; idx += blockDim.x) {
        const uint8_t packed = row_q[idx >> 1];
        const uint8_t code = (idx & 1) ? (packed >> 4) : (packed & 0x0f);
        const float v = fp4_value_from_code_scale(code, row_s[idx >> 5]);
        int q = __float2int_rn(v * inv_scale);
        q = max(-127, min(127, q));
        dst_q[idx] = static_cast<int8_t>(q);
    }
}

std::vector<torch::Tensor> fp4_weight_to_int8_forward_cuda(
    const torch::Tensor& wq,
    const torch::Tensor& ws) {
    c10::cuda::CUDAGuard device_guard(wq.device());
    const int rows = static_cast<int>(wq.size(0) * wq.size(1));
    const int k = static_cast<int>(wq.size(2) * 2);
    auto out_q = torch::empty({wq.size(0), wq.size(1), k}, wq.options().dtype(torch::kInt8));
    auto out_s = torch::empty({wq.size(0), wq.size(1)}, wq.options().dtype(torch::kFloat32));
    fp4_weight_to_int8_row_kernel<<<rows, kQuantThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        wq.data_ptr<uint8_t>(),
        ws.data_ptr<uint8_t>(),
        out_q.data_ptr<int8_t>(),
        out_s.data_ptr<float>(),
        rows,
        k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_q, out_s};
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
    const int dim_packs = dim / 4;       // x_q packed as int8x4
    const int blocks_k = dim / 32;       // one e8m0 byte per 32 K elems
    extern __shared__ int x_shared[];
    const int* x_i32 = reinterpret_cast<const int*>(x_q);
    for (int idx = threadIdx.x; idx < dim_packs; idx += blockDim.x) {
        x_shared[idx] = x_i32[idx];
    }
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
    // Each K-block of 32 elems: 16 weight bytes = 8 uint16 (each uint16 = 4
    // codes). 8 activation int8x4 packs (32 int8). Two __dp4a per uint16.
    for (int kb = 0; kb < blocks_k; ++kb) {
        const uint16_t* w1_pack = w1_pack_base + kb * 8;
        const uint16_t* w3_pack = w3_pack_base + kb * 8;
        const int* x_pack = x_shared + kb * 8;
        int gate_block = 0;
        int up_block = 0;
        #pragma unroll
        for (int ip = 0; ip < 8; ++ip) {
            const uint32_t w1_bits = static_cast<uint32_t>(w1_pack[ip]);
            const uint32_t w3_bits = static_cast<uint32_t>(w3_pack[ip]);
            const int w1_p = fp4_unpack_4codes_prmt(w1_bits);
            const int w3_p = fp4_unpack_4codes_prmt(w3_bits);
            const int x_p = x_pack[ip];
            gate_block = __dp4a(x_p, w1_p, gate_block);
            up_block = __dp4a(x_p, w3_p, up_block);
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
    for (int idx = threadIdx.x; idx < packs; idx += blockDim.x) {
        h_shared[idx] = h_i32[idx];
    }
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
            const uint32_t w_bits = static_cast<uint32_t>(w2_pack[ip]);
            const int w_p = fp4_unpack_4codes_prmt(w_bits);
            blk = __dp4a(h_pack[ip], w_p, blk);
        }
        row_acc += static_cast<float>(blk) * fp4_block_scale(w2_scale_row[kb]);
    }
    atomicAdd(y + col, row_acc * hidden_scale[route]);
}

torch::Tensor moe_single_token_fp4_forward_cuda(
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
    auto gate_f32 = torch::empty({topk, inter_dim}, x.options().dtype(torch::kFloat32));
    auto up_f32 = torch::empty({topk, inter_dim}, x.options().dtype(torch::kFloat32));
    auto hidden_q = torch::empty({topk, inter_dim}, x.options().dtype(torch::kInt8));
    auto hidden_scale = torch::empty({topk}, x.options().dtype(torch::kFloat32));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "moe_single_quantize_x_fp4", [&] {
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
    moe_single_w1w3_fp4_kernel<<<w1w3_grid, gemm_block, x_shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        x_q.data_ptr<int8_t>(),
        x_scale.data_ptr<float>(),
        indices_contig.data_ptr<int64_t>(),
        w1q.data_ptr<uint8_t>(),
        w1s.data_ptr<uint8_t>(),
        w3q.data_ptr<uint8_t>(),
        w3s.data_ptr<uint8_t>(),
        gate_f32.data_ptr<float>(),
        up_f32.data_ptr<float>(),
        topk,
        static_cast<int>(experts_start_idx),
        n_local_experts,
        dim,
        inter_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    moe_single_swiglu_quant_fp4_kernel<<<topk, kQuantThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        gate_f32.data_ptr<float>(),
        up_f32.data_ptr<float>(),
        indices_contig.data_ptr<int64_t>(),
        weights_contig.data_ptr<float>(),
        hidden_q.data_ptr<int8_t>(),
        hidden_scale.data_ptr<float>(),
        topk,
        static_cast<int>(experts_start_idx),
        n_local_experts,
        inter_dim,
        static_cast<float>(swiglu_limit));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const dim3 w2_grid(ceil_div(dim, kGemmThreads), topk);
    const size_t h_shared_bytes = static_cast<size_t>(inter_dim / 4) * sizeof(int);
    moe_single_w2_accum_fp4_kernel<<<w2_grid, gemm_block, h_shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        hidden_q.data_ptr<int8_t>(),
        hidden_scale.data_ptr<float>(),
        indices_contig.data_ptr<int64_t>(),
        w2q.data_ptr<uint8_t>(),
        w2s.data_ptr<uint8_t>(),
        y.data_ptr<float>(),
        topk,
        static_cast<int>(experts_start_idx),
        n_local_experts,
        dim,
        inter_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

constexpr int kFp4GroupedCols = 128;
constexpr int kFp4GroupedRows = 4;

__global__ void moe_fp4_grouped_w13_padded_kernel(
    const int8_t* __restrict__ x_q_pad,
    const float* __restrict__ x_scale_pad,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w1q,
    const uint8_t* __restrict__ w1s,
    const uint8_t* __restrict__ w3q,
    const uint8_t* __restrict__ w3s,
    float* __restrict__ gate_f32,
    float* __restrict__ up_f32,
    int rows_per_expert,
    int dim,
    int inter_dim) {
    const int expert = blockIdx.z;
    const int row_base = blockIdx.y * kFp4GroupedRows;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim_packs = dim / 4;
    const int blocks_k = dim / 32;
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    extern __shared__ int x_shared[];
    for (int idx = threadIdx.x; idx < kFp4GroupedRows * dim_packs; idx += blockDim.x) {
        const int r = idx / dim_packs;
        const int pack = idx - r * dim_packs;
        const int row = row_base + r;
        int value = 0;
        if (row < count && row < rows_per_expert) {
            const int* x_row = reinterpret_cast<const int*>(x_q_pad + (static_cast<int64_t>(expert) * rows_per_expert + row) * dim);
            value = x_row[pack];
        }
        x_shared[idx] = value;
    }
    __syncthreads();
    if (col >= inter_dim) return;

    const uint8_t* w1_row_bytes = w1q + (static_cast<int64_t>(expert) * inter_dim + col) * (dim / 2);
    const uint8_t* w3_row_bytes = w3q + (static_cast<int64_t>(expert) * inter_dim + col) * (dim / 2);
    const uint8_t* w1_scale_row = w1s + (static_cast<int64_t>(expert) * inter_dim + col) * blocks_k;
    const uint8_t* w3_scale_row = w3s + (static_cast<int64_t>(expert) * inter_dim + col) * blocks_k;
    const uint16_t* w1_pack_base = reinterpret_cast<const uint16_t*>(w1_row_bytes);
    const uint16_t* w3_pack_base = reinterpret_cast<const uint16_t*>(w3_row_bytes);
    float gate_acc[kFp4GroupedRows] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_acc[kFp4GroupedRows] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int kb = 0; kb < blocks_k; ++kb) {
        const uint16_t* w1_pack = w1_pack_base + kb * 8;
        const uint16_t* w3_pack = w3_pack_base + kb * 8;
        int gate_block[kFp4GroupedRows] = {0, 0, 0, 0};
        int up_block[kFp4GroupedRows] = {0, 0, 0, 0};
        #pragma unroll
        for (int ip = 0; ip < 8; ++ip) {
            const int w1_p = fp4_unpack_4codes_prmt(static_cast<uint32_t>(w1_pack[ip]));
            const int w3_p = fp4_unpack_4codes_prmt(static_cast<uint32_t>(w3_pack[ip]));
            #pragma unroll
            for (int r = 0; r < kFp4GroupedRows; ++r) {
                const int x_p = x_shared[r * dim_packs + kb * 8 + ip];
                gate_block[r] = __dp4a(x_p, w1_p, gate_block[r]);
                up_block[r] = __dp4a(x_p, w3_p, up_block[r]);
            }
        }
        const float s1 = fp4_block_scale(w1_scale_row[kb]);
        const float s3 = fp4_block_scale(w3_scale_row[kb]);
        #pragma unroll
        for (int r = 0; r < kFp4GroupedRows; ++r) {
            gate_acc[r] += static_cast<float>(gate_block[r]) * s1;
            up_acc[r] += static_cast<float>(up_block[r]) * s3;
        }
    }
    #pragma unroll
    for (int r = 0; r < kFp4GroupedRows; ++r) {
        const int row = row_base + r;
        if (row < count && row < rows_per_expert) {
            const int64_t base = (static_cast<int64_t>(expert) * rows_per_expert + row) * inter_dim + col;
            const float xs = x_scale_pad[expert * rows_per_expert + row];
            gate_f32[base] = gate_acc[r] * xs;
            up_f32[base] = up_acc[r] * xs;
        }
    }
}

__global__ void moe_pair_swiglu_quantize_padded_fp32_kernel(
    const float* __restrict__ gate_f32,
    const float* __restrict__ up_f32,
    const int32_t* __restrict__ seg_starts,
    const float* __restrict__ route_weights,
    int8_t* __restrict__ hidden_q,
    float* __restrict__ hidden_scale,
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
    const float route_weight = route_weights[route];
    float local_max = 0.0f;
    for (int col = tid; col < inter_dim; col += blockDim.x) {
        const int idx = base + col;
        float gate = gate_f32[idx];
        float up = up_f32[idx];
        if (swiglu_limit > 0.0f) {
            up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
            gate = fminf(gate, swiglu_limit);
        }
        const float silu = gate / (1.0f + expf(-gate));
        const float value = silu * up * route_weight;
        local_max = fmaxf(local_max, fabsf(value));
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    const float scale = fmaxf(sdata[0], 1.0e-6f) / 127.0f;
    if (tid == 0) hidden_scale[padded_row] = scale;
    __syncthreads();
    const float inv_scale = 1.0f / scale;
    for (int col = tid; col < inter_dim; col += blockDim.x) {
        const int idx = base + col;
        float gate = gate_f32[idx];
        float up = up_f32[idx];
        if (swiglu_limit > 0.0f) {
            up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
            gate = fminf(gate, swiglu_limit);
        }
        const float silu = gate / (1.0f + expf(-gate));
        const float value = silu * up * route_weight;
        int q = __float2int_rn(value * inv_scale);
        q = max(-127, min(127, q));
        hidden_q[idx] = static_cast<int8_t>(q);
    }
}

__global__ void moe_fp4_grouped_w2_scatter_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w2q,
    const uint8_t* __restrict__ w2s,
    float* __restrict__ y,
    int rows_per_expert,
    int dim,
    int inter_dim) {
    const int expert = blockIdx.z;
    const int row_base = blockIdx.y * kFp4GroupedRows;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int inter_packs = inter_dim / 4;
    const int blocks_k = inter_dim / 32;
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    extern __shared__ int h_shared[];
    for (int idx = threadIdx.x; idx < kFp4GroupedRows * inter_packs; idx += blockDim.x) {
        const int r = idx / inter_packs;
        const int pack = idx - r * inter_packs;
        const int row = row_base + r;
        int value = 0;
        if (row < count && row < rows_per_expert) {
            const int* h_row = reinterpret_cast<const int*>(hidden_q + (static_cast<int64_t>(expert) * rows_per_expert + row) * inter_dim);
            value = h_row[pack];
        }
        h_shared[idx] = value;
    }
    __syncthreads();
    if (col >= dim) return;

    const uint8_t* w2_row_bytes = w2q + (static_cast<int64_t>(expert) * dim + col) * (inter_dim / 2);
    const uint8_t* w2_scale_row = w2s + (static_cast<int64_t>(expert) * dim + col) * blocks_k;
    const uint16_t* w2_pack_base = reinterpret_cast<const uint16_t*>(w2_row_bytes);
    float acc[kFp4GroupedRows] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int kb = 0; kb < blocks_k; ++kb) {
        const uint16_t* w2_pack = w2_pack_base + kb * 8;
        int block_acc[kFp4GroupedRows] = {0, 0, 0, 0};
        #pragma unroll
        for (int ip = 0; ip < 8; ++ip) {
            const int w_p = fp4_unpack_4codes_prmt(static_cast<uint32_t>(w2_pack[ip]));
            #pragma unroll
            for (int r = 0; r < kFp4GroupedRows; ++r) {
                const int h_p = h_shared[r * inter_packs + kb * 8 + ip];
                block_acc[r] = __dp4a(h_p, w_p, block_acc[r]);
            }
        }
        const float s = fp4_block_scale(w2_scale_row[kb]);
        #pragma unroll
        for (int r = 0; r < kFp4GroupedRows; ++r) {
            acc[r] += static_cast<float>(block_acc[r]) * s;
        }
    }
    #pragma unroll
    for (int r = 0; r < kFp4GroupedRows; ++r) {
        const int row = row_base + r;
        if (row < count && row < rows_per_expert) {
            const int route = seg_starts[expert] + row;
            const int64_t token = route_tokens[route];
            atomicAdd(y + token * dim + col, acc[r] * hidden_scale[expert * rows_per_expert + row]);
        }
    }
}

__device__ __forceinline__ int8_t fp4_decode_code_x2(uint8_t code) {
    return fp4_lut_x2[code & 0x0f];
}

__global__ void moe_fp4_grouped_w13_wmma_kernel(
    const int8_t* __restrict__ x_q_pad,
    const float* __restrict__ x_scale_pad,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w1q,
    const uint8_t* __restrict__ w1s,
    const uint8_t* __restrict__ w3q,
    const uint8_t* __restrict__ w3s,
    float* __restrict__ gate_f32,
    float* __restrict__ up_f32,
    int rows_per_expert,
    int dim,
    int inter_dim) {
    using namespace nvcuda;
    constexpr int TILE = 16;
    const int expert = blockIdx.z;
    const int row_base = blockIdx.y * TILE;
    const int col_base = blockIdx.x * TILE;
    const int tid = threadIdx.x;
    const int blocks_k = dim / 32;
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    extern __shared__ __align__(16) unsigned char smem[];
    signed char* a_tile = reinterpret_cast<signed char*>(smem);
    signed char* b_tile = a_tile + TILE * TILE;
    int* c_tile = reinterpret_cast<int*>(b_tile + 2 * TILE * TILE);
    float* gate_acc = reinterpret_cast<float*>(c_tile + TILE * TILE);
    float* up_acc = gate_acc + TILE * TILE;

    for (int i = tid; i < TILE * TILE; i += blockDim.x) {
        gate_acc[i] = 0.0f;
        up_acc[i] = 0.0f;
    }
    __syncthreads();

    for (int kb = 0; kb < blocks_k; ++kb) {
        wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, signed char, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, signed char, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> c1_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> c3_frag;
        wmma::fill_fragment(c1_frag, 0);
        wmma::fill_fragment(c3_frag, 0);
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int k0 = kb * 32 + half * TILE;
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int r = i / TILE;
                const int k = i - r * TILE;
                const int row = row_base + r;
                a_tile[i] = (row < count && row < rows_per_expert)
                    ? x_q_pad[(static_cast<int64_t>(expert) * rows_per_expert + row) * dim + k0 + k]
                    : 0;
            }
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int k = i & (TILE - 1);
                const int n = i >> 4;
                const int col = col_base + n;
                signed char v1 = 0;
                signed char v3 = 0;
                if (col < inter_dim) {
                    const int kk = k0 + k;
                    const uint8_t* w1_row = w1q + (static_cast<int64_t>(expert) * inter_dim + col) * (dim / 2);
                    const uint8_t* w3_row = w3q + (static_cast<int64_t>(expert) * inter_dim + col) * (dim / 2);
                    const uint8_t b1 = w1_row[kk >> 1];
                    const uint8_t b3 = w3_row[kk >> 1];
                    const uint8_t c1 = (kk & 1) ? (b1 >> 4) : (b1 & 0x0f);
                    const uint8_t c3 = (kk & 1) ? (b3 >> 4) : (b3 & 0x0f);
                    v1 = fp4_decode_code_x2(c1);
                    v3 = fp4_decode_code_x2(c3);
                }
                b_tile[k + n * TILE] = v1;
                b_tile[TILE * TILE + k + n * TILE] = v3;
            }
            __syncthreads();
            wmma::load_matrix_sync(a_frag, a_tile, TILE);
            wmma::load_matrix_sync(b_frag, b_tile, TILE);
            wmma::mma_sync(c1_frag, a_frag, b_frag, c1_frag);
            wmma::load_matrix_sync(b_frag, b_tile + TILE * TILE, TILE);
            wmma::mma_sync(c3_frag, a_frag, b_frag, c3_frag);
            __syncthreads();
        }
        wmma::store_matrix_sync(c_tile, c1_frag, TILE, wmma::mem_row_major);
        __syncthreads();
        for (int i = tid; i < TILE * TILE; i += blockDim.x) {
            const int n = i & (TILE - 1);
            const int col = col_base + n;
            if (col < inter_dim) {
                const float s = fp4_block_scale(w1s[(static_cast<int64_t>(expert) * inter_dim + col) * blocks_k + kb]);
                gate_acc[i] += static_cast<float>(c_tile[i]) * s;
            }
        }
        __syncthreads();
        wmma::store_matrix_sync(c_tile, c3_frag, TILE, wmma::mem_row_major);
        __syncthreads();
        for (int i = tid; i < TILE * TILE; i += blockDim.x) {
            const int n = i & (TILE - 1);
            const int col = col_base + n;
            if (col < inter_dim) {
                const float s = fp4_block_scale(w3s[(static_cast<int64_t>(expert) * inter_dim + col) * blocks_k + kb]);
                up_acc[i] += static_cast<float>(c_tile[i]) * s;
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < TILE * TILE; i += blockDim.x) {
        const int r = i / TILE;
        const int n = i - r * TILE;
        const int row = row_base + r;
        const int col = col_base + n;
        if (row < count && row < rows_per_expert && col < inter_dim) {
            const int64_t out = (static_cast<int64_t>(expert) * rows_per_expert + row) * inter_dim + col;
            const float xs = x_scale_pad[expert * rows_per_expert + row];
            gate_f32[out] = gate_acc[i] * xs;
            up_f32[out] = up_acc[i] * xs;
        }
    }
}

__global__ void moe_fp4_grouped_w2_wmma_scatter_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w2q,
    const uint8_t* __restrict__ w2s,
    float* __restrict__ y,
    int rows_per_expert,
    int dim,
    int inter_dim) {
    using namespace nvcuda;
    constexpr int TILE = 16;
    const int expert = blockIdx.z;
    const int row_base = blockIdx.y * TILE;
    const int col_base = blockIdx.x * TILE;
    const int tid = threadIdx.x;
    const int blocks_k = inter_dim / 32;
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    extern __shared__ __align__(16) unsigned char smem[];
    signed char* a_tile = reinterpret_cast<signed char*>(smem);
    signed char* b_tile = a_tile + TILE * TILE;
    int* c_tile = reinterpret_cast<int*>(b_tile + TILE * TILE);
    float* acc = reinterpret_cast<float*>(c_tile + TILE * TILE);

    for (int i = tid; i < TILE * TILE; i += blockDim.x) acc[i] = 0.0f;
    __syncthreads();

    for (int kb = 0; kb < blocks_k; ++kb) {
        wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, signed char, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, signed char, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> c_frag;
        wmma::fill_fragment(c_frag, 0);
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int k0 = kb * 32 + half * TILE;
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int r = i / TILE;
                const int k = i - r * TILE;
                const int row = row_base + r;
                a_tile[i] = (row < count && row < rows_per_expert)
                    ? hidden_q[(static_cast<int64_t>(expert) * rows_per_expert + row) * inter_dim + k0 + k]
                    : 0;
            }
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int k = i & (TILE - 1);
                const int n = i >> 4;
                const int col = col_base + n;
                signed char v = 0;
                if (col < dim) {
                    const int kk = k0 + k;
                    const uint8_t* w2_row = w2q + (static_cast<int64_t>(expert) * dim + col) * (inter_dim / 2);
                    const uint8_t b = w2_row[kk >> 1];
                    const uint8_t c = (kk & 1) ? (b >> 4) : (b & 0x0f);
                    v = fp4_decode_code_x2(c);
                }
                b_tile[k + n * TILE] = v;
            }
            __syncthreads();
            wmma::load_matrix_sync(a_frag, a_tile, TILE);
            wmma::load_matrix_sync(b_frag, b_tile, TILE);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads();
        }
        wmma::store_matrix_sync(c_tile, c_frag, TILE, wmma::mem_row_major);
        __syncthreads();
        for (int i = tid; i < TILE * TILE; i += blockDim.x) {
            const int n = i & (TILE - 1);
            const int col = col_base + n;
            if (col < dim) {
                const float s = fp4_block_scale(w2s[(static_cast<int64_t>(expert) * dim + col) * blocks_k + kb]);
                acc[i] += static_cast<float>(c_tile[i]) * s;
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < TILE * TILE; i += blockDim.x) {
        const int r = i / TILE;
        const int n = i - r * TILE;
        const int row = row_base + r;
        const int col = col_base + n;
        if (row < count && row < rows_per_expert && col < dim) {
            const int route = seg_starts[expert] + row;
            const int64_t token = route_tokens[route];
            atomicAdd(y + token * dim + col, acc[i] * hidden_scale[expert * rows_per_expert + row]);
        }
    }
}

__global__ void moe_fp4_grouped_w13_wmma64_kernel(
    const int8_t* __restrict__ x_q_pad,
    const float* __restrict__ x_scale_pad,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w1q,
    const uint8_t* __restrict__ w1s,
    const uint8_t* __restrict__ w3q,
    const uint8_t* __restrict__ w3s,
    float* __restrict__ gate_f32,
    float* __restrict__ up_f32,
    int rows_per_expert,
    int dim,
    int inter_dim) {
    using namespace nvcuda;
    constexpr int TILE = 16;
    constexpr int TN = 64;
    constexpr int NF = TN / TILE;
    const int expert = blockIdx.z;
    const int row_base = blockIdx.y * TILE;
    const int col_base = blockIdx.x * TN;
    const int tid = threadIdx.x;
    const int blocks_k = dim / 32;
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    extern __shared__ __align__(16) unsigned char smem[];
    signed char* a_tile = reinterpret_cast<signed char*>(smem);
    signed char* b_tile = a_tile + TILE * TILE;
    int* c_tile = reinterpret_cast<int*>(b_tile + 2 * NF * TILE * TILE);
    float* gate_acc = reinterpret_cast<float*>(c_tile + TILE * TILE);
    float* up_acc = gate_acc + TILE * TN;

    for (int i = tid; i < TILE * TN; i += blockDim.x) {
        gate_acc[i] = 0.0f;
        up_acc[i] = 0.0f;
    }
    __syncthreads();

    for (int kb = 0; kb < blocks_k; ++kb) {
        wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, signed char, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, signed char, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> c1_frag[NF];
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> c3_frag[NF];
        #pragma unroll
        for (int f = 0; f < NF; ++f) {
            wmma::fill_fragment(c1_frag[f], 0);
            wmma::fill_fragment(c3_frag[f], 0);
        }
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int k0 = kb * 32 + half * TILE;
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int r = i / TILE;
                const int k = i - r * TILE;
                const int row = row_base + r;
                a_tile[i] = (row < count && row < rows_per_expert)
                    ? x_q_pad[(static_cast<int64_t>(expert) * rows_per_expert + row) * dim + k0 + k]
                    : 0;
            }
            for (int i = tid; i < NF * TILE * TILE; i += blockDim.x) {
                const int f = i / (TILE * TILE);
                const int j = i - f * TILE * TILE;
                const int k = j & (TILE - 1);
                const int n = j >> 4;
                const int col = col_base + f * TILE + n;
                signed char v1 = 0;
                signed char v3 = 0;
                if (col < inter_dim) {
                    const int kk = k0 + k;
                    const uint8_t* w1_row = w1q + (static_cast<int64_t>(expert) * inter_dim + col) * (dim / 2);
                    const uint8_t* w3_row = w3q + (static_cast<int64_t>(expert) * inter_dim + col) * (dim / 2);
                    const uint8_t b1 = w1_row[kk >> 1];
                    const uint8_t b3 = w3_row[kk >> 1];
                    v1 = fp4_decode_code_x2((kk & 1) ? (b1 >> 4) : (b1 & 0x0f));
                    v3 = fp4_decode_code_x2((kk & 1) ? (b3 >> 4) : (b3 & 0x0f));
                }
                b_tile[(2 * f) * TILE * TILE + k + n * TILE] = v1;
                b_tile[(2 * f + 1) * TILE * TILE + k + n * TILE] = v3;
            }
            __syncthreads();
            wmma::load_matrix_sync(a_frag, a_tile, TILE);
            #pragma unroll
            for (int f = 0; f < NF; ++f) {
                wmma::load_matrix_sync(b_frag, b_tile + (2 * f) * TILE * TILE, TILE);
                wmma::mma_sync(c1_frag[f], a_frag, b_frag, c1_frag[f]);
                wmma::load_matrix_sync(b_frag, b_tile + (2 * f + 1) * TILE * TILE, TILE);
                wmma::mma_sync(c3_frag[f], a_frag, b_frag, c3_frag[f]);
            }
            __syncthreads();
        }
        #pragma unroll
        for (int f = 0; f < NF; ++f) {
            wmma::store_matrix_sync(c_tile, c1_frag[f], TILE, wmma::mem_row_major);
            __syncthreads();
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int n = i & (TILE - 1);
                const int col = col_base + f * TILE + n;
                if (col < inter_dim) {
                    const float s = fp4_block_scale(w1s[(static_cast<int64_t>(expert) * inter_dim + col) * blocks_k + kb]);
                    const int r = i >> 4;
                    gate_acc[r * TN + f * TILE + n] += static_cast<float>(c_tile[i]) * s;
                }
            }
            __syncthreads();
            wmma::store_matrix_sync(c_tile, c3_frag[f], TILE, wmma::mem_row_major);
            __syncthreads();
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int n = i & (TILE - 1);
                const int col = col_base + f * TILE + n;
                if (col < inter_dim) {
                    const float s = fp4_block_scale(w3s[(static_cast<int64_t>(expert) * inter_dim + col) * blocks_k + kb]);
                    const int r = i >> 4;
                    up_acc[r * TN + f * TILE + n] += static_cast<float>(c_tile[i]) * s;
                }
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < TILE * TN; i += blockDim.x) {
        const int r = i / TN;
        const int n = i - r * TN;
        const int row = row_base + r;
        const int col = col_base + n;
        if (row < count && row < rows_per_expert && col < inter_dim) {
            const int64_t out = (static_cast<int64_t>(expert) * rows_per_expert + row) * inter_dim + col;
            const float xs = x_scale_pad[expert * rows_per_expert + row];
            gate_f32[out] = gate_acc[i] * xs;
            up_f32[out] = up_acc[i] * xs;
        }
    }
}

__global__ void moe_fp4_grouped_w2_wmma64_scatter_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w2q,
    const uint8_t* __restrict__ w2s,
    float* __restrict__ y,
    int rows_per_expert,
    int dim,
    int inter_dim) {
    using namespace nvcuda;
    constexpr int TILE = 16;
    constexpr int TN = 64;
    constexpr int NF = TN / TILE;
    const int expert = blockIdx.z;
    const int row_base = blockIdx.y * TILE;
    const int col_base = blockIdx.x * TN;
    const int tid = threadIdx.x;
    const int blocks_k = inter_dim / 32;
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    extern __shared__ __align__(16) unsigned char smem[];
    signed char* a_tile = reinterpret_cast<signed char*>(smem);
    signed char* b_tile = a_tile + TILE * TILE;
    int* c_tile = reinterpret_cast<int*>(b_tile + NF * TILE * TILE);
    float* acc = reinterpret_cast<float*>(c_tile + TILE * TILE);

    for (int i = tid; i < TILE * TN; i += blockDim.x) acc[i] = 0.0f;
    __syncthreads();

    for (int kb = 0; kb < blocks_k; ++kb) {
        wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, signed char, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, signed char, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> c_frag[NF];
        #pragma unroll
        for (int f = 0; f < NF; ++f) wmma::fill_fragment(c_frag[f], 0);
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int k0 = kb * 32 + half * TILE;
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int r = i / TILE;
                const int k = i - r * TILE;
                const int row = row_base + r;
                a_tile[i] = (row < count && row < rows_per_expert)
                    ? hidden_q[(static_cast<int64_t>(expert) * rows_per_expert + row) * inter_dim + k0 + k]
                    : 0;
            }
            for (int i = tid; i < NF * TILE * TILE; i += blockDim.x) {
                const int f = i / (TILE * TILE);
                const int j = i - f * TILE * TILE;
                const int k = j & (TILE - 1);
                const int n = j >> 4;
                const int col = col_base + f * TILE + n;
                signed char v = 0;
                if (col < dim) {
                    const int kk = k0 + k;
                    const uint8_t* w2_row = w2q + (static_cast<int64_t>(expert) * dim + col) * (inter_dim / 2);
                    const uint8_t b = w2_row[kk >> 1];
                    v = fp4_decode_code_x2((kk & 1) ? (b >> 4) : (b & 0x0f));
                }
                b_tile[f * TILE * TILE + k + n * TILE] = v;
            }
            __syncthreads();
            wmma::load_matrix_sync(a_frag, a_tile, TILE);
            #pragma unroll
            for (int f = 0; f < NF; ++f) {
                wmma::load_matrix_sync(b_frag, b_tile + f * TILE * TILE, TILE);
                wmma::mma_sync(c_frag[f], a_frag, b_frag, c_frag[f]);
            }
            __syncthreads();
        }
        #pragma unroll
        for (int f = 0; f < NF; ++f) {
            wmma::store_matrix_sync(c_tile, c_frag[f], TILE, wmma::mem_row_major);
            __syncthreads();
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int n = i & (TILE - 1);
                const int col = col_base + f * TILE + n;
                if (col < dim) {
                    const float s = fp4_block_scale(w2s[(static_cast<int64_t>(expert) * dim + col) * blocks_k + kb]);
                    const int r = i >> 4;
                    acc[r * TN + f * TILE + n] += static_cast<float>(c_tile[i]) * s;
                }
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < TILE * TN; i += blockDim.x) {
        const int r = i / TN;
        const int n = i - r * TN;
        const int row = row_base + r;
        const int col = col_base + n;
        if (row < count && row < rows_per_expert && col < dim) {
            const int route = seg_starts[expert] + row;
            const int64_t token = route_tokens[route];
            atomicAdd(y + token * dim + col, acc[i] * hidden_scale[expert * rows_per_expert + row]);
        }
    }
}

torch::Tensor moe_prefill_fp4_grouped_gemm_forward_cuda(
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
    const int n_experts = static_cast<int>(w1q.size(0));
    const int inter_dim = static_cast<int>(w1q.size(1));
    auto y = torch::zeros({x.size(0), x.size(1)}, x.options().dtype(torch::kFloat32));
    if (routes == 0 || n_experts <= 0) {
        return y;
    }
    auto seg_i32 = seg_starts.scalar_type() == torch::kInt32 ? seg_starts.contiguous() : seg_starts.to(torch::kInt32);
    auto counts_i32 = seg_i32.slice(0, 1, n_experts + 1) - seg_i32.slice(0, 0, n_experts);
    const int max_count = counts_i32.max().item<int>();
    if (max_count <= 0) {
        return y;
    }

    auto x_sorted = torch::empty({routes, dim}, x.options());
    auto x_q = torch::empty({routes, 1, dim}, x.options().dtype(torch::kInt8));
    auto x_scale = torch::empty({routes, 1}, x.options().dtype(torch::kFloat32));
    auto x_pad = torch::empty({n_experts, max_count, dim}, x.options().dtype(torch::kInt8));
    auto x_scale_pad = torch::empty({n_experts, max_count}, x.options().dtype(torch::kFloat32));
    auto gate_f32 = torch::empty({n_experts, max_count, inter_dim}, x.options().dtype(torch::kFloat32));
    auto up_f32 = torch::empty({n_experts, max_count, inter_dim}, x.options().dtype(torch::kFloat32));
    auto hidden_pad = torch::empty({n_experts, max_count, inter_dim}, x.options().dtype(torch::kInt8));
    auto hidden_scale_pad = torch::empty({n_experts, max_count}, x.options().dtype(torch::kFloat32));
    const int threads = 256;
    const int gather_blocks = static_cast<int>((static_cast<int64_t>(routes) * dim + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "moe_fp4_grouped_gather", [&] {
        gather_routes_kernel<scalar_t><<<gather_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(),
            route_tokens.data_ptr<int64_t>(),
            x_sorted.data_ptr<scalar_t>(),
            routes,
            dim);
    });
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "moe_fp4_grouped_quantize_x", [&] {
        quantize_rows_kernel<scalar_t><<<routes, kQuantThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x_sorted.data_ptr<scalar_t>(),
            x_q.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(),
            routes,
            1,
            dim);
    });
    const int pad_x_total = n_experts * max_count * dim;
    moe_pad_q_rows_kernel<<<ceil_div(pad_x_total, threads), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x_q.data_ptr<int8_t>(),
        x_scale.data_ptr<float>(),
        seg_i32.data_ptr<int32_t>(),
        x_pad.data_ptr<int8_t>(),
        x_scale_pad.data_ptr<float>(),
        n_experts,
        max_count,
        dim);

    const dim3 fp4_block(128);
    const dim3 w13_grid(ceil_div(inter_dim, 16), ceil_div(max_count, 16), n_experts);
    const size_t x_shared_bytes = 4096;
    moe_fp4_grouped_w13_wmma_kernel<<<w13_grid, fp4_block, x_shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        x_pad.data_ptr<int8_t>(),
        x_scale_pad.data_ptr<float>(),
        seg_i32.data_ptr<int32_t>(),
        w1q.data_ptr<uint8_t>(),
        w1s.data_ptr<uint8_t>(),
        w3q.data_ptr<uint8_t>(),
        w3s.data_ptr<uint8_t>(),
        gate_f32.data_ptr<float>(),
        up_f32.data_ptr<float>(),
        max_count,
        dim,
        inter_dim);

    moe_pair_swiglu_quantize_padded_fp32_kernel<<<dim3(n_experts, max_count), kQuantThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        gate_f32.data_ptr<float>(),
        up_f32.data_ptr<float>(),
        seg_i32.data_ptr<int32_t>(),
        route_weights_sorted.data_ptr<float>(),
        hidden_pad.data_ptr<int8_t>(),
        hidden_scale_pad.data_ptr<float>(),
        max_count,
        inter_dim,
        static_cast<float>(swiglu_limit));

    const dim3 w2_grid(ceil_div(dim, 16), ceil_div(max_count, 16), n_experts);
    const size_t h_shared_bytes = 4096;
    moe_fp4_grouped_w2_wmma_scatter_kernel<<<w2_grid, fp4_block, h_shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        hidden_pad.data_ptr<int8_t>(),
        hidden_scale_pad.data_ptr<float>(),
        route_tokens.data_ptr<int64_t>(),
        seg_i32.data_ptr<int32_t>(),
        w2q.data_ptr<uint8_t>(),
        w2s.data_ptr<uint8_t>(),
        y.data_ptr<float>(),
        max_count,
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

template <typename scalar_t>
__device__ __forceinline__ float custom_allreduce_load(const void* ptr, int idx) {
    return static_cast<float>(reinterpret_cast<const scalar_t*>(ptr)[idx]);
}

template <typename scalar_t>
__device__ __forceinline__ void custom_allreduce_store(void* ptr, int idx, float value) {
    reinterpret_cast<scalar_t*>(ptr)[idx] = static_cast<scalar_t>(value);
}

__global__ void custom_allreduce_bf16_kernel(
    const c10::BFloat16* __restrict__ input,
    c10::BFloat16* __restrict__ output,
    c10::BFloat16* peer0,
    c10::BFloat16* peer1,
    c10::BFloat16* peer2,
    c10::BFloat16* peer3,
    volatile int32_t* flag0,
    volatile int32_t* flag1,
    volatile int32_t* flag2,
    volatile int32_t* flag3,
    int rank,
    int dim,
    int seq,
    unsigned long long max_ticks,
    int32_t* status) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    c10::BFloat16* peers[4] = {peer0, peer1, peer2, peer3};
    volatile int32_t* flags[4] = {flag0, flag1, flag2, flag3};
    for (int i = tid; i < dim; i += stride) {
        peers[rank][i] = input[i];
    }
    __threadfence_system();
    if (tid == 0) {
        flags[rank][0] = seq;
        __threadfence_system();
        const unsigned long long start = clock64();
        bool ok = false;
        while (clock64() - start < max_ticks) {
            ok = flags[0][0] >= seq && flags[1][0] >= seq && flags[2][0] >= seq && flags[3][0] >= seq;
            if (ok) break;
        }
        if (!ok) status[0] = 1;
    }
    __syncthreads();
    if (status[0] != 0) return;
    for (int i = tid; i < dim; i += stride) {
        float acc = static_cast<float>(peer0[i]) + static_cast<float>(peer1[i]) + static_cast<float>(peer2[i]) + static_cast<float>(peer3[i]);
        output[i] = static_cast<c10::BFloat16>(acc);
    }
}

__global__ void custom_allreduce_half_kernel(
    const at::Half* __restrict__ input,
    at::Half* __restrict__ output,
    at::Half* peer0,
    at::Half* peer1,
    at::Half* peer2,
    at::Half* peer3,
    volatile int32_t* flag0,
    volatile int32_t* flag1,
    volatile int32_t* flag2,
    volatile int32_t* flag3,
    int rank,
    int dim,
    int seq,
    unsigned long long max_ticks,
    int32_t* status) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    at::Half* peers[4] = {peer0, peer1, peer2, peer3};
    volatile int32_t* flags[4] = {flag0, flag1, flag2, flag3};
    for (int i = tid; i < dim; i += stride) {
        peers[rank][i] = input[i];
    }
    __threadfence_system();
    if (tid == 0) {
        flags[rank][0] = seq;
        __threadfence_system();
        const unsigned long long start = clock64();
        bool ok = false;
        while (clock64() - start < max_ticks) {
            ok = flags[0][0] >= seq && flags[1][0] >= seq && flags[2][0] >= seq && flags[3][0] >= seq;
            if (ok) break;
        }
        if (!ok) status[0] = 1;
    }
    __syncthreads();
    if (status[0] != 0) return;
    for (int i = tid; i < dim; i += stride) {
        float acc = static_cast<float>(peer0[i]) + static_cast<float>(peer1[i]) + static_cast<float>(peer2[i]) + static_cast<float>(peer3[i]);
        output[i] = static_cast<at::Half>(acc);
    }
}

__global__ void custom_allreduce_float_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* peer0,
    float* peer1,
    float* peer2,
    float* peer3,
    volatile int32_t* flag0,
    volatile int32_t* flag1,
    volatile int32_t* flag2,
    volatile int32_t* flag3,
    int rank,
    int dim,
    int seq,
    unsigned long long max_ticks,
    int32_t* status) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    float* peers[4] = {peer0, peer1, peer2, peer3};
    volatile int32_t* flags[4] = {flag0, flag1, flag2, flag3};
    for (int i = tid; i < dim; i += stride) {
        peers[rank][i] = input[i];
    }
    __threadfence_system();
    if (tid == 0) {
        flags[rank][0] = seq;
        __threadfence_system();
        const unsigned long long start = clock64();
        bool ok = false;
        while (clock64() - start < max_ticks) {
            ok = flags[0][0] >= seq && flags[1][0] >= seq && flags[2][0] >= seq && flags[3][0] >= seq;
            if (ok) break;
        }
        if (!ok) status[0] = 1;
    }
    __syncthreads();
    if (status[0] != 0) return;
    for (int i = tid; i < dim; i += stride) {
        output[i] = peer0[i] + peer1[i] + peer2[i] + peer3[i];
    }
}

template <typename reduce_t, typename shared_t, typename out_t>
__global__ void moe_finalize_reduce_kernel(
    const reduce_t* __restrict__ y_reduce,
    const shared_t* __restrict__ shared,
    out_t* __restrict__ out,
    int64_t n) {
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = tid; i < n; i += stride) {
        float v = static_cast<float>(y_reduce[i]) + static_cast<float>(shared[i]);
        out[i] = static_cast<out_t>(v);
    }
}

template <typename reduce_t, typename shared_t>
torch::Tensor moe_finalize_reduce_dispatch_out(
    const torch::Tensor& y_reduce,
    const torch::Tensor& shared,
    int64_t out_dtype_code) {
    auto options = y_reduce.options();
    if (out_dtype_code == 1) options = options.dtype(torch::kBFloat16);
    else if (out_dtype_code == 2) options = options.dtype(torch::kFloat16);
    else options = options.dtype(torch::kFloat32);
    auto out = torch::empty_like(y_reduce, options);
    const int64_t n = y_reduce.numel();
    const int threads = 256;
    const int blocks = std::max<int64_t>(1, std::min<int64_t>(128, (n + threads - 1) / threads));
    auto stream = at::cuda::getCurrentCUDAStream();
    if (out_dtype_code == 1) {
        moe_finalize_reduce_kernel<reduce_t, shared_t, c10::BFloat16><<<blocks, threads, 0, stream>>>(
            y_reduce.data_ptr<reduce_t>(), shared.data_ptr<shared_t>(), out.data_ptr<c10::BFloat16>(), n);
    } else if (out_dtype_code == 2) {
        moe_finalize_reduce_kernel<reduce_t, shared_t, at::Half><<<blocks, threads, 0, stream>>>(
            y_reduce.data_ptr<reduce_t>(), shared.data_ptr<shared_t>(), out.data_ptr<at::Half>(), n);
    } else {
        moe_finalize_reduce_kernel<reduce_t, shared_t, float><<<blocks, threads, 0, stream>>>(
            y_reduce.data_ptr<reduce_t>(), shared.data_ptr<shared_t>(), out.data_ptr<float>(), n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

template <typename reduce_t>
torch::Tensor moe_finalize_reduce_dispatch_shared(
    const torch::Tensor& y_reduce,
    const torch::Tensor& shared,
    int64_t out_dtype_code) {
    if (shared.scalar_type() == torch::kBFloat16) {
        return moe_finalize_reduce_dispatch_out<reduce_t, c10::BFloat16>(y_reduce, shared, out_dtype_code);
    }
    if (shared.scalar_type() == torch::kFloat16) {
        return moe_finalize_reduce_dispatch_out<reduce_t, at::Half>(y_reduce, shared, out_dtype_code);
    }
    return moe_finalize_reduce_dispatch_out<reduce_t, float>(y_reduce, shared, out_dtype_code);
}

torch::Tensor moe_finalize_reduce_forward_cuda(
    const torch::Tensor& y_reduce,
    const torch::Tensor& shared,
    int64_t out_dtype_code) {
    c10::cuda::CUDAGuard device_guard(y_reduce.device());
    if (y_reduce.scalar_type() == torch::kBFloat16) {
        return moe_finalize_reduce_dispatch_shared<c10::BFloat16>(y_reduce, shared, out_dtype_code);
    }
    if (y_reduce.scalar_type() == torch::kFloat16) {
        return moe_finalize_reduce_dispatch_shared<at::Half>(y_reduce, shared, out_dtype_code);
    }
    return moe_finalize_reduce_dispatch_shared<float>(y_reduce, shared, out_dtype_code);
}

std::vector<std::string> custom_allreduce_ipc_handle_cuda(
    const torch::Tensor& buffer,
    const torch::Tensor& flags) {
    c10::cuda::CUDAGuard device_guard(buffer.device());
    TORCH_CHECK(flags.device() == buffer.device(), "custom allreduce buffer/flags device mismatch");
    cudaIpcMemHandle_t buffer_handle;
    cudaIpcMemHandle_t flag_handle;
    check_cuda(cudaIpcGetMemHandle(&buffer_handle, buffer.data_ptr()), "cudaIpcGetMemHandle(buffer) failed");
    check_cuda(cudaIpcGetMemHandle(&flag_handle, flags.data_ptr()), "cudaIpcGetMemHandle(flags) failed");
    return {
        std::string(reinterpret_cast<const char*>(&buffer_handle), sizeof(buffer_handle)),
        std::string(reinterpret_cast<const char*>(&flag_handle), sizeof(flag_handle)),
    };
}

int64_t custom_allreduce_open_cuda(
    const torch::Tensor& local_buffer,
    const torch::Tensor& local_flags,
    const std::vector<std::string>& buffer_handles,
    const std::vector<std::string>& flag_handles,
    int64_t rank,
    int64_t world_size,
    int64_t dim,
    int64_t dtype_code) {
    TORCH_CHECK(world_size == kCustomAllreduceWorldSize, "custom allreduce world_size must be 4");
    TORCH_CHECK(buffer_handles.size() == kCustomAllreduceWorldSize && flag_handles.size() == kCustomAllreduceWorldSize,
                "custom allreduce requires four buffer and flag handles");
    CustomAllreduceHandle handle;
    handle.rank = static_cast<int>(rank);
    handle.world_size = static_cast<int>(world_size);
    handle.dim = static_cast<int>(dim);
    if (dtype_code == 1) handle.dtype = at::kBFloat16;
    else if (dtype_code == 2) handle.dtype = at::kHalf;
    else if (dtype_code == 3) handle.dtype = at::kFloat;
    else TORCH_CHECK(false, "unsupported custom allreduce dtype_code");
    for (int i = 0; i < kCustomAllreduceWorldSize; ++i) {
        if (i == handle.rank) {
            handle.peer_data[i] = local_buffer.data_ptr();
            handle.peer_flags[i] = local_flags.data_ptr<int32_t>();
            continue;
        }
        TORCH_CHECK(buffer_handles[i].size() == sizeof(cudaIpcMemHandle_t), "invalid buffer IPC handle size");
        TORCH_CHECK(flag_handles[i].size() == sizeof(cudaIpcMemHandle_t), "invalid flag IPC handle size");
        cudaIpcMemHandle_t buffer_ipc;
        cudaIpcMemHandle_t flag_ipc;
        std::memcpy(&buffer_ipc, buffer_handles[i].data(), sizeof(buffer_ipc));
        std::memcpy(&flag_ipc, flag_handles[i].data(), sizeof(flag_ipc));
        check_cuda(cudaIpcOpenMemHandle(&handle.peer_data[i], buffer_ipc, cudaIpcMemLazyEnablePeerAccess), "cudaIpcOpenMemHandle(buffer) failed");
        void* flag_ptr = nullptr;
        check_cuda(cudaIpcOpenMemHandle(&flag_ptr, flag_ipc, cudaIpcMemLazyEnablePeerAccess), "cudaIpcOpenMemHandle(flags) failed");
        handle.peer_flags[i] = reinterpret_cast<int32_t*>(flag_ptr);
    }
    std::lock_guard<std::mutex> lock(g_custom_allreduce_mutex);
    const int64_t id = g_custom_allreduce_next_handle++;
    g_custom_allreduce_handles.emplace(id, handle);
    return id;
}

void custom_allreduce_close_cuda(int64_t handle_id) {
    CustomAllreduceHandle handle;
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(g_custom_allreduce_mutex);
        auto it = g_custom_allreduce_handles.find(handle_id);
        if (it != g_custom_allreduce_handles.end()) {
            handle = it->second;
            g_custom_allreduce_handles.erase(it);
            found = true;
        }
    }
    if (!found) return;
    for (int i = 0; i < handle.world_size; ++i) {
        if (i == handle.rank) continue;
        if (handle.peer_data[i] != nullptr) cudaIpcCloseMemHandle(handle.peer_data[i]);
        if (handle.peer_flags[i] != nullptr) cudaIpcCloseMemHandle(handle.peer_flags[i]);
    }
}

bool custom_allreduce_inplace_cuda(
    int64_t handle_id,
    torch::Tensor& tensor,
    int64_t seq,
    int64_t timeout_us) {
    CustomAllreduceHandle handle;
    {
        std::lock_guard<std::mutex> lock(g_custom_allreduce_mutex);
        auto it = g_custom_allreduce_handles.find(handle_id);
        TORCH_CHECK(it != g_custom_allreduce_handles.end(), "invalid custom allreduce handle");
        handle = it->second;
    }
    c10::cuda::CUDAGuard device_guard(tensor.device());
    TORCH_CHECK(tensor.numel() == handle.dim, "custom allreduce tensor dim mismatch");
    TORCH_CHECK(tensor.scalar_type() == handle.dtype, "custom allreduce tensor dtype mismatch");
    auto status = torch::zeros({1}, tensor.options().dtype(torch::kInt32));
    const int threads = 256;
    const int blocks = std::max(1, std::min(32, static_cast<int>((handle.dim + threads - 1) / threads)));
    const unsigned long long max_ticks = static_cast<unsigned long long>(std::max<int64_t>(timeout_us, 1)) * 2000ULL;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (handle.dtype == at::kBFloat16) {
        custom_allreduce_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const c10::BFloat16*>(tensor.data_ptr()),
            reinterpret_cast<c10::BFloat16*>(tensor.data_ptr()),
            reinterpret_cast<c10::BFloat16*>(handle.peer_data[0]),
            reinterpret_cast<c10::BFloat16*>(handle.peer_data[1]),
            reinterpret_cast<c10::BFloat16*>(handle.peer_data[2]),
            reinterpret_cast<c10::BFloat16*>(handle.peer_data[3]),
            reinterpret_cast<volatile int32_t*>(handle.peer_flags[0]),
            reinterpret_cast<volatile int32_t*>(handle.peer_flags[1]),
            reinterpret_cast<volatile int32_t*>(handle.peer_flags[2]),
            reinterpret_cast<volatile int32_t*>(handle.peer_flags[3]),
            handle.rank,
            handle.dim,
            static_cast<int>(seq),
            max_ticks,
            status.data_ptr<int32_t>());
    } else if (handle.dtype == at::kHalf) {
        custom_allreduce_half_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const at::Half*>(tensor.data_ptr()),
            reinterpret_cast<at::Half*>(tensor.data_ptr()),
            reinterpret_cast<at::Half*>(handle.peer_data[0]),
            reinterpret_cast<at::Half*>(handle.peer_data[1]),
            reinterpret_cast<at::Half*>(handle.peer_data[2]),
            reinterpret_cast<at::Half*>(handle.peer_data[3]),
            reinterpret_cast<volatile int32_t*>(handle.peer_flags[0]),
            reinterpret_cast<volatile int32_t*>(handle.peer_flags[1]),
            reinterpret_cast<volatile int32_t*>(handle.peer_flags[2]),
            reinterpret_cast<volatile int32_t*>(handle.peer_flags[3]),
            handle.rank,
            handle.dim,
            static_cast<int>(seq),
            max_ticks,
            status.data_ptr<int32_t>());
    } else {
        custom_allreduce_float_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float*>(tensor.data_ptr()),
            reinterpret_cast<float*>(tensor.data_ptr()),
            reinterpret_cast<float*>(handle.peer_data[0]),
            reinterpret_cast<float*>(handle.peer_data[1]),
            reinterpret_cast<float*>(handle.peer_data[2]),
            reinterpret_cast<float*>(handle.peer_data[3]),
            reinterpret_cast<volatile int32_t*>(handle.peer_flags[0]),
            reinterpret_cast<volatile int32_t*>(handle.peer_flags[1]),
            reinterpret_cast<volatile int32_t*>(handle.peer_flags[2]),
            reinterpret_cast<volatile int32_t*>(handle.peer_flags[3]),
            handle.rank,
            handle.dim,
            static_cast<int>(seq),
            max_ticks,
            status.data_ptr<int32_t>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    check_cuda(cudaStreamSynchronize(stream), "custom allreduce synchronize failed");
    return status.cpu().item<int32_t>() == 0;
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

torch::Tensor flashinfer_style_sparse_attn_headpair_forward_cuda(
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

    const dim3 grid(bsz * ((heads + 1) / 2));
    const dim3 block(kAttnThreads);
    const size_t shared_bytes =
        static_cast<size_t>(topk) * sizeof(float) * 2 +
        static_cast<size_t>(dim) * sizeof(float) * 2 +
        static_cast<size_t>(topk) * sizeof(int32_t) +
        static_cast<size_t>(kAttnThreads) * sizeof(float) * 2;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q_contig.scalar_type(), "flashinfer_style_sparse_attn_headpair", [&] {
        flashinfer_style_sparse_attn_headpair_kernel<scalar_t><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
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
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores_flat.scalar_type(), "c4_topk_from_scores", [&] {
            launch_c4_topk_kernel_or_tile_merge<scalar_t>(
                scores_flat.data_ptr<scalar_t>(),
                topk.data_ptr<int64_t>(),
                bsz,
                kv_len,
                k,
                static_cast<int>(offset),
                scores.options());
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

    if (c4_indexer_score_cuda_enabled()) {
        auto scores = torch::empty({bsz, 1, std::max(kv_len, 1)}, q.options().dtype(torch::kFloat32));
        if (kv_len > 0) {
            auto scores_flat = scores.view({bsz, kv_len});
            const dim3 score_grid(bsz, kv_len);
            const dim3 score_block(256);
            AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, kv_cache.scalar_type(), "c4_score", [&] {
                c4_score_kernel<scalar_t><<<score_grid, score_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                    q_work.data_ptr<float>(),
                    kv_cache.data_ptr<scalar_t>(),
                    weights_contig.data_ptr<float>(),
                    scores_flat.data_ptr<float>(),
                    bsz,
                    heads,
                    max_cache_len,
                    kv_len);
            });
        } else {
            scores.zero_();
        }
        if (return_scores) {
            return scores;
        }
        auto topk = torch::empty({bsz, std::max(k, 1)}, q.options().dtype(torch::kInt64));
        if (k > 0) {
            auto scores_flat = scores.view({bsz, kv_len});
            launch_c4_topk_kernel_or_tile_merge<float>(
                scores_flat.data_ptr<float>(),
                topk.data_ptr<int64_t>(),
                bsz,
                kv_len,
                k,
                static_cast<int>(offset),
                q.options());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return topk.view({bsz, 1, std::max(k, 1)});
    }

    auto q_work_bf16 = q_work.view({bsz, heads, 128}).to(q.scalar_type());
    if (c4_indexer_post_gemm_enabled()) {
        auto scores = torch::empty({bsz, std::max(kv_len, 1)}, q.options().dtype(torch::kFloat32));
        if (kv_len > 0) {
            auto logits = torch::bmm(q_work_bf16, kv_cache.narrow(0, 0, bsz).narrow(1, 0, kv_len).transpose(1, 2));
            auto weights_post = weights.contiguous().view({bsz, heads});
            const dim3 post_grid(bsz, (kv_len + 255) / 256);
            const dim3 post_block(256);
            AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, logits.scalar_type(), "c4_post_gemm_score", [&] {
                using logit_t = scalar_t;
                AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, weights_post.scalar_type(), "c4_post_gemm_weight", [&] {
                    c4_post_gemm_score_kernel<logit_t, scalar_t><<<post_grid, post_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                        logits.data_ptr<logit_t>(),
                        weights_post.data_ptr<scalar_t>(),
                        scores.data_ptr<float>(),
                        bsz,
                        heads,
                        kv_len);
                });
            });
        } else {
            scores.zero_();
        }
        if (return_scores) {
            return scores.view({bsz, 1, std::max(kv_len, 1)});
        }
        auto topk = torch::empty({bsz, std::max(k, 1)}, q.options().dtype(torch::kInt64));
        if (k > 0) {
            auto scores_flat = scores.view({bsz, kv_len});
            launch_c4_topk_kernel_or_tile_merge<float>(
                scores_flat.data_ptr<float>(),
                topk.data_ptr<int64_t>(),
                bsz,
                kv_len,
                k,
                static_cast<int>(offset),
                q.options());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return topk.view({bsz, 1, std::max(k, 1)});
    }

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
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores_flat.scalar_type(), "c4_topk_from_fused_scores", [&] {
            launch_c4_topk_kernel_or_tile_merge<scalar_t>(
                scores_flat.data_ptr<scalar_t>(),
                topk.data_ptr<int64_t>(),
                bsz,
                kv_len,
                k,
                static_cast<int>(offset),
                q.options());
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

__device__ __forceinline__ float q8_0_block_scale(const uint8_t* block) {
    const uint16_t bits = static_cast<uint16_t>(block[0]) | (static_cast<uint16_t>(block[1]) << 8);
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ float gguf_block_scale_f16(const uint8_t* ptr) {
    const uint16_t bits = static_cast<uint16_t>(ptr[0]) | (static_cast<uint16_t>(ptr[1]) << 8);
    return __half2float(__ushort_as_half(bits));
}

// Decode the shared f16 super-block scale of an IQ1_M block from the high
// nibbles of its four uint16 scale words.  Mirrors gguf-py IQ1_M.dequantize.
__device__ __forceinline__ float iq1m_super_scale(const uint16_t* sc) {
    const uint16_t d_bits = static_cast<uint16_t>(
        ((sc[0] & 0xF000u) >> 12) | ((sc[1] & 0xF000u) >> 8) |
        ((sc[2] & 0xF000u) >> 4) | (sc[3] & 0xF000u));
    return __half2float(__ushort_as_half(d_bits));
}

// Full IQ1_M block dot for 256 elements.  Must be entered by all 32 lanes of a
// warp (the per-lane reduction uses a full-warp shuffle); the returned value is
// valid on lane 0.  iq1_grid is the (2048, 8) int8 codebook flattened to 16384.
// Block layout (56 bytes): qs[32] + qh[16] + scales[8] (4 uint16 words).
__device__ __forceinline__ float iq1m_block_dot_256(
    const float* __restrict__ x_shared,
    const uint8_t* __restrict__ block,
    const int8_t* __restrict__ iq1_grid,
    int lane) {
    const uint8_t* qs = block;
    const uint8_t* qh = block + 32;
    const uint16_t* sc = reinterpret_cast<const uint16_t*>(block + 48);
    const float d = iq1m_super_scale(sc);
    float local = 0.0f;
    const int j = lane;  // grid index 0..31, one per lane
    if (j < 32) {
        const int sub = j >> 2;
        const int half = (j >> 1) & 1;
        const int pair = j & 1;
        const int s = sub * 2 + half;
        const int local_scale = (sc[s >> 2] >> ((s & 3) * 3)) & 0x07;
        const float dl = d * static_cast<float>(2 * local_scale + 1);
        const int qhv = (qh[j >> 1] >> ((j & 1) * 4)) & 0x0F;
        const int qidx = static_cast<int>(qs[j]) | ((qhv & 0x07) << 8);
        const float delta = (qhv & 0x08) ? -0.125f : 0.125f;
        const int8_t* gvals = iq1_grid + qidx * 8;
        const int k_out = sub * 32 + half * 16 + pair * 8;
        float partial = 0.0f;
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            partial += x_shared[k_out + g] * (static_cast<float>(gvals[g]) + delta);
        }
        local = dl * partial;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_down_sync(0xffffffff, local, offset);
    }
    return local;  // valid on lane 0
}

// IQ1_M block dot for up to 4 rows sharing the same weight block.  Row r reads
// activations from x_shared + r*256.  Accumulates into acc[r] on lane 0.
__device__ __forceinline__ void iq1m_block_dot_256_rows4(
    const float* __restrict__ x_shared,
    const uint8_t* __restrict__ block,
    const int8_t* __restrict__ iq1_grid,
    int lane,
    int valid_rows,
    float* __restrict__ acc) {
    const uint8_t* qs = block;
    const uint8_t* qh = block + 32;
    const uint16_t* sc = reinterpret_cast<const uint16_t*>(block + 48);
    const float d = iq1m_super_scale(sc);
    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
    const int j = lane;
    if (j < 32) {
        const int sub = j >> 2;
        const int half = (j >> 1) & 1;
        const int pair = j & 1;
        const int s = sub * 2 + half;
        const int local_scale = (sc[s >> 2] >> ((s & 3) * 3)) & 0x07;
        const float dl = d * static_cast<float>(2 * local_scale + 1);
        const int qhv = (qh[j >> 1] >> ((j & 1) * 4)) & 0x0F;
        const int qidx = static_cast<int>(qs[j]) | ((qhv & 0x07) << 8);
        const float delta = (qhv & 0x08) ? -0.125f : 0.125f;
        const int8_t* gvals = iq1_grid + qidx * 8;
        const int k_out = sub * 32 + half * 16 + pair * 8;
        float gv[8];
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            gv[g] = static_cast<float>(gvals[g]) + delta;
        }
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            const float w = gv[g];
            const int kk = k_out + g;
            l0 += x_shared[kk] * w;
            if (valid_rows > 1) l1 += x_shared[256 + kk] * w;
            if (valid_rows > 2) l2 += x_shared[512 + kk] * w;
            if (valid_rows > 3) l3 += x_shared[768 + kk] * w;
        }
        l0 *= dl; l1 *= dl; l2 *= dl; l3 *= dl;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        l0 += __shfl_down_sync(0xffffffff, l0, offset);
        l1 += __shfl_down_sync(0xffffffff, l1, offset);
        l2 += __shfl_down_sync(0xffffffff, l2, offset);
        l3 += __shfl_down_sync(0xffffffff, l3, offset);
    }
    if (lane == 0) {
        acc[0] += l0;
        if (valid_rows > 1) acc[1] += l1;
        if (valid_rows > 2) acc[2] += l2;
        if (valid_rows > 3) acc[3] += l3;
    }
}


constexpr int kQ8_0TileN = 8;
constexpr int kGGUFQuantTileN = 8;
constexpr int kGGUFQuantPrefillRows = 4;

__device__ __forceinline__ int pack_i8x4(int a, int b, int c, int d) {
    return (a & 0xff) | ((b & 0xff) << 8) | ((c & 0xff) << 16) | ((d & 0xff) << 24);
}

template <typename scalar_t>
__global__ void gguf_q8_1_quantize_16_kernel(
    const scalar_t* __restrict__ x,
    int8_t* __restrict__ x_q,
    float* __restrict__ x_scale,
    int rows,
    int row_elems) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= rows || group * 16 >= row_elems || lane >= 16) return;
    const int k = group * 16 + lane;
    const float xv = k < row_elems ? static_cast<float>(x[static_cast<int64_t>(row) * row_elems + k]) : 0.0f;
    float maxv = fabsf(xv);
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        maxv = fmaxf(maxv, __shfl_down_sync(0xffff, maxv, offset));
    }
    const float scale = __shfl_sync(0xffff, maxv > 0.0f ? maxv / 127.0f : 1.0e-8f, 0);
    if (lane == 0) {
        x_scale[row * ((row_elems + 15) / 16) + group] = scale;
    }
    int q = __float2int_rn(xv / scale);
    q = max(-127, min(127, q));
    x_q[static_cast<int64_t>(row) * row_elems + k] = static_cast<int8_t>(q);
}

__global__ void gguf_q2k_gemm_dp4a_prequant_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const uint8_t* __restrict__ blocks,
    c10::BFloat16* __restrict__ out,
    int rows,
    int n,
    int row_elems,
    int blocks_per_row,
    int x_groups) {
    const int row = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kGGUFQuantTileN + warp;
    if (row >= rows || warp >= kGGUFQuantTileN || out_col >= n) return;

    const uint8_t* weight_row = blocks + static_cast<int64_t>(out_col) * blocks_per_row * 84;
    const int8_t* x_row = x_q + static_cast<int64_t>(row) * row_elems;
    const float* xs_row = x_scale + static_cast<int64_t>(row) * x_groups;
    float acc = 0.0f;

    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const uint8_t* block = weight_row + static_cast<int64_t>(block_idx) * 84;
        const uint8_t* scales = block;
        const uint8_t* qs = block + 16;
        const float d = gguf_block_scale_f16(block + 80);
        const float dmin = gguf_block_scale_f16(block + 82);
        const int k_base = block_idx * 256;
        for (int group = 0; group < 16; ++group) {
            float local = 0.0f;
            if (lane < 4) {
                const int half_block = group >> 3;
                const int group_in_half = group & 7;
                const int shift = (group_in_half >> 1) * 2;
                const int byte_start = half_block * 32 + (group_in_half & 1) * 16;
                const int idx = lane * 4;
                const int q0 = static_cast<int>((qs[byte_start + idx + 0] >> shift) & 0x03);
                const int q1 = static_cast<int>((qs[byte_start + idx + 1] >> shift) & 0x03);
                const int q2 = static_cast<int>((qs[byte_start + idx + 2] >> shift) & 0x03);
                const int q3 = static_cast<int>((qs[byte_start + idx + 3] >> shift) & 0x03);
                const int w_pack = pack_i8x4(q0, q1, q2, q3);
                const int x_idx = k_base + group * 16 + idx;
                const int x_pack = pack_i8x4(
                    static_cast<int>(x_row[x_idx + 0]),
                    static_cast<int>(x_row[x_idx + 1]),
                    static_cast<int>(x_row[x_idx + 2]),
                    static_cast<int>(x_row[x_idx + 3]));
                const int dot_q = __dp4a(w_pack, x_pack, 0);
                const int sum_x = __dp4a(0x01010101, x_pack, 0);
                const float qscale = d * static_cast<float>(scales[group] & 0x0f);
                const float base = dmin * static_cast<float>(scales[group] >> 4);
                local = xs_row[block_idx * 16 + group] * (qscale * static_cast<float>(dot_q) - base * static_cast<float>(sum_x));
            }
            #pragma unroll
            for (int offset = 2; offset > 0; offset >>= 1) {
                local += __shfl_down_sync(0x0f, local, offset);
            }
            if (lane == 0) acc += local;
        }
    }
    if (lane == 0) {
        out[static_cast<int64_t>(row) * n + out_col] = c10::BFloat16(acc);
    }
}

template <typename scalar_t>
__global__ void gguf_q2k_gemm_dp4a_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ blocks,
    c10::BFloat16* __restrict__ out,
    int rows,
    int n,
    int row_elems,
    int blocks_per_row) {
    const int row = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kGGUFQuantTileN + warp;
    if (row >= rows || warp >= kGGUFQuantTileN) return;

    __shared__ int8_t x_q8[256];
    __shared__ float x_scale[16];
    const scalar_t* x_row = x + static_cast<int64_t>(row) * row_elems;
    const uint8_t* weight_row = blocks + static_cast<int64_t>(out_col) * blocks_per_row * 84;
    float acc = 0.0f;

    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        if (warp == 0) {
            for (int group = 0; group < 16; ++group) {
                const int k = group * 16 + lane;
                float xv = 0.0f;
                if (lane < 16 && k_base + k < row_elems) {
                    xv = static_cast<float>(x_row[k_base + k]);
                }
                float maxv = fabsf(xv);
                #pragma unroll
                for (int offset = 8; offset > 0; offset >>= 1) {
                    maxv = fmaxf(maxv, __shfl_down_sync(0xffff, maxv, offset));
                }
                float scale = 0.0f;
                float inv_scale = 0.0f;
                if (lane == 0) {
                    scale = maxv > 0.0f ? maxv / 127.0f : 0.0f;
                    x_scale[group] = scale;
                }
                scale = __shfl_sync(0xffff, scale, 0);
                inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
                if (lane < 16) {
                    int q = __float2int_rn(xv * inv_scale);
                    q = max(-127, min(127, q));
                    x_q8[k] = static_cast<int8_t>(q);
                }
            }
        }
        __syncthreads();

        if (out_col < n) {
            const uint8_t* block = weight_row + static_cast<int64_t>(block_idx) * 84;
            const uint8_t* scales = block;
            const uint8_t* qs = block + 16;
            const float d = gguf_block_scale_f16(block + 80);
            const float dmin = gguf_block_scale_f16(block + 82);
            for (int group = 0; group < 16; ++group) {
                float local = 0.0f;
                if (lane < 4) {
                    const int half_block = group >> 3;
                    const int group_in_half = group & 7;
                    const int shift = (group_in_half >> 1) * 2;
                    const int byte_start = half_block * 32 + (group_in_half & 1) * 16;
                    const int idx = lane * 4;
                    const int q0 = static_cast<int>((qs[byte_start + idx + 0] >> shift) & 0x03);
                    const int q1 = static_cast<int>((qs[byte_start + idx + 1] >> shift) & 0x03);
                    const int q2 = static_cast<int>((qs[byte_start + idx + 2] >> shift) & 0x03);
                    const int q3 = static_cast<int>((qs[byte_start + idx + 3] >> shift) & 0x03);
                    const int w_pack = pack_i8x4(q0, q1, q2, q3);
                    const int x_pack = pack_i8x4(
                        static_cast<int>(x_q8[group * 16 + idx + 0]),
                        static_cast<int>(x_q8[group * 16 + idx + 1]),
                        static_cast<int>(x_q8[group * 16 + idx + 2]),
                        static_cast<int>(x_q8[group * 16 + idx + 3]));
                    const int dot_q = __dp4a(w_pack, x_pack, 0);
                    const int sum_x = __dp4a(0x01010101, x_pack, 0);
                    const float qscale = d * static_cast<float>(scales[group] & 0x0f);
                    const float base = dmin * static_cast<float>(scales[group] >> 4);
                    local = x_scale[group] * (qscale * static_cast<float>(dot_q) - base * static_cast<float>(sum_x));
                }
                #pragma unroll
                for (int offset = 2; offset > 0; offset >>= 1) {
                    local += __shfl_down_sync(0x0f, local, offset);
                }
                if (lane == 0) acc += local;
            }
        }
        __syncthreads();
    }

    if (lane == 0 && out_col < n) {
        out[static_cast<int64_t>(row) * n + out_col] = c10::BFloat16(acc);
    }
}

__device__ __forceinline__ float gguf_quant_block_dot_256(
    const float* __restrict__ x_shared,
    const uint8_t* __restrict__ block,
    const int8_t* __restrict__ signed_grid,
    int type_id,
    int lane);

__device__ __forceinline__ void gguf_quant_block_dot_256_rows4(
    const float* __restrict__ x_shared,
    const uint8_t* __restrict__ block,
    const int8_t* __restrict__ signed_grid,
    int type_id,
    int lane,
    int valid_rows,
    float* __restrict__ acc);

template <typename scalar_t>
__global__ void gguf_quant_gemm_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ blocks,
    const int8_t* __restrict__ signed_grid,
    c10::BFloat16* __restrict__ out,
    int rows,
    int n,
    int row_elems,
    int blocks_per_row,
    int block_bytes,
    int type_id) {
    const int row = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kGGUFQuantTileN + warp;
    if (row >= rows || warp >= kGGUFQuantTileN) return;

    __shared__ float x_shared[256];
    float acc = 0.0f;
    const scalar_t* x_row = x + row * row_elems;
    const uint8_t* weight_row = blocks + (static_cast<size_t>(out_col) * blocks_per_row * block_bytes);
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int k = lane; k < 256; k += 32) {
            const int x_idx = k_base + k;
            x_shared[k] = x_idx < row_elems ? static_cast<float>(x_row[x_idx]) : 0.0f;
        }
        __syncthreads();
        if (out_col < n) {
            const uint8_t* block = weight_row + static_cast<size_t>(block_idx) * block_bytes;
            if (type_id == 0) {
                const float d = gguf_block_scale_f16(block);
                const uint8_t* qs = block + 2;
                for (int sub = 0; sub < 8; ++sub) {
                    const uint8_t* chunk = qs + sub * 8;
                    const uint32_t aux = static_cast<uint32_t>(chunk[4]) |
                        (static_cast<uint32_t>(chunk[5]) << 8) |
                        (static_cast<uint32_t>(chunk[6]) << 16) |
                        (static_cast<uint32_t>(chunk[7]) << 24);
                    const float scale = 0.125f * d * static_cast<float>(2 * (aux >> 28) + 1);
                    for (int part = 0; part < 4; ++part) {
                        const int grid_id = static_cast<int>(chunk[part]);
                        const int sign_idx = static_cast<int>((aux >> (7 * part)) & 127);
                        const int8_t* vals = signed_grid + ((grid_id * 128 + sign_idx) * 8);
                        const int k_start = sub * 32 + part * 8;
                        float local = 0.0f;
                        if (lane < 8) {
                            local = x_shared[k_start + lane] * static_cast<float>(vals[lane]) * scale;
                            #pragma unroll
                            for (int offset = 4; offset > 0; offset >>= 1) {
                                local += __shfl_down_sync(0xff, local, offset);
                            }
                            if (lane == 0) acc += local;
                        }
                    }
                }
            } else if (type_id == 2) {
                const float blk = iq1m_block_dot_256(x_shared, block, signed_grid, lane);
                if (lane == 0) acc += blk;
            } else {
                const uint8_t* scales = block;
                const uint8_t* qs = block + 16;
                const float d = gguf_block_scale_f16(block + 80);
                const float dmin = gguf_block_scale_f16(block + 82);
                for (int group = 0; group < 16; ++group) {
                    const int half_block = group / 8;
                    const int group_in_half = group % 8;
                    const int shift = (group_in_half / 2) * 2;
                    const int byte_start = half_block * 32 + (group_in_half % 2) * 16;
                    const float qscale = d * static_cast<float>(scales[group] & 0x0f);
                    const float base = dmin * static_cast<float>(scales[group] >> 4);
                    const int k_start = group * 16;
                    float local = 0.0f;
                    if (lane < 16) {
                        const uint8_t q = static_cast<uint8_t>((qs[byte_start + lane] >> shift) & 0x03);
                        const float xv = x_shared[k_start + lane];
                        local = qscale * xv * static_cast<float>(q) - base * xv;
                        #pragma unroll
                        for (int offset = 8; offset > 0; offset >>= 1) {
                            local += __shfl_down_sync(0xffff, local, offset);
                        }
                        if (lane == 0) acc += local;
                    }
                }
            }
        }
        __syncthreads();
    }
    if (lane == 0 && out_col < n) {
        out[row * n + out_col] = c10::BFloat16(acc);
    }
}

template <typename scalar_t>
__global__ void gguf_quant_gemm_prefill_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ blocks,
    const int8_t* __restrict__ signed_grid,
    c10::BFloat16* __restrict__ out,
    int rows,
    int n,
    int row_elems,
    int blocks_per_row,
    int block_bytes,
    int type_id) {
    const int row_base = blockIdx.y * kGGUFQuantPrefillRows;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kGGUFQuantTileN + warp;
    if (row_base >= rows || warp >= kGGUFQuantTileN) return;

    __shared__ float x_shared[kGGUFQuantTileN][kGGUFQuantPrefillRows][256];
    float acc[kGGUFQuantPrefillRows] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int valid_rows = min(kGGUFQuantPrefillRows, rows - row_base);
    const uint8_t* weight_row = blocks + (static_cast<size_t>(out_col) * blocks_per_row * block_bytes);
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int r = 0; r < kGGUFQuantPrefillRows; ++r) {
            const int row = row_base + r;
            const scalar_t* x_row = x + static_cast<int64_t>(row) * row_elems;
            for (int k = lane; k < 256; k += 32) {
                const int x_idx = k_base + k;
                x_shared[warp][r][k] = (r < valid_rows && x_idx < row_elems) ? static_cast<float>(x_row[x_idx]) : 0.0f;
            }
        }
        __syncthreads();
        if (out_col < n) {
            gguf_quant_block_dot_256_rows4(&x_shared[warp][0][0], weight_row + static_cast<int64_t>(block_idx) * block_bytes, signed_grid, type_id, lane, valid_rows, acc);
        }
        __syncthreads();
    }
    if (lane == 0 && out_col < n) {
        for (int r = 0; r < valid_rows; ++r) {
            out[static_cast<int64_t>(row_base + r) * n + out_col] = c10::BFloat16(acc[r]);
        }
    }
}


template <typename scalar_t>
__global__ void gguf_quant_gemm_pair_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ blocks0,
    const uint8_t* __restrict__ blocks1,
    const int8_t* __restrict__ signed_grid,
    c10::BFloat16* __restrict__ out0,
    c10::BFloat16* __restrict__ out1,
    int rows,
    int n,
    int row_elems,
    int blocks0_per_row,
    int block0_bytes,
    int type0_id,
    int blocks1_per_row,
    int block1_bytes,
    int type1_id) {
    const int row = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kGGUFQuantTileN + warp;
    if (row >= rows || warp >= kGGUFQuantTileN) return;

    __shared__ float x_shared[256];
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const scalar_t* x_row = x + row * row_elems;
    const uint8_t* weight0_row = blocks0 + (static_cast<size_t>(out_col) * blocks0_per_row * block0_bytes);
    const uint8_t* weight1_row = blocks1 + (static_cast<size_t>(out_col) * blocks1_per_row * block1_bytes);
    for (int block_idx = 0; block_idx < blocks0_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int k = lane; k < 256; k += 32) {
            const int x_idx = k_base + k;
            x_shared[k] = x_idx < row_elems ? static_cast<float>(x_row[x_idx]) : 0.0f;
        }
        __syncthreads();
        if (out_col < n) {
            acc0 += gguf_quant_block_dot_256(x_shared, weight0_row + static_cast<int64_t>(block_idx) * block0_bytes, signed_grid, type0_id, lane);
            acc1 += gguf_quant_block_dot_256(x_shared, weight1_row + static_cast<int64_t>(block_idx) * block1_bytes, signed_grid, type1_id, lane);
        }
        __syncthreads();
    }
    if (lane == 0 && out_col < n) {
        out0[row * n + out_col] = c10::BFloat16(acc0);
        out1[row * n + out_col] = c10::BFloat16(acc1);
    }
}

__device__ __forceinline__ float gguf_quant_block_dot_256(
    const float* __restrict__ x_shared,
    const uint8_t* __restrict__ block,
    const int8_t* __restrict__ signed_grid,
    int type_id,
    int lane) {
    float acc = 0.0f;
    if (type_id == 0) {
        const float d = gguf_block_scale_f16(block);
        const uint8_t* qs = block + 2;
        for (int sub = 0; sub < 8; ++sub) {
            const uint8_t* chunk = qs + sub * 8;
            const uint32_t aux = static_cast<uint32_t>(chunk[4]) |
                (static_cast<uint32_t>(chunk[5]) << 8) |
                (static_cast<uint32_t>(chunk[6]) << 16) |
                (static_cast<uint32_t>(chunk[7]) << 24);
            const float scale = 0.125f * d * static_cast<float>(2 * (aux >> 28) + 1);
            for (int part = 0; part < 4; ++part) {
                const int grid_id = static_cast<int>(chunk[part]);
                const int sign_idx = static_cast<int>((aux >> (7 * part)) & 127);
                const int8_t* vals = signed_grid + ((grid_id * 128 + sign_idx) * 8);
                const int k_start = sub * 32 + part * 8;
                float local = 0.0f;
                if (lane < 8) {
                    local = x_shared[k_start + lane] * static_cast<float>(vals[lane]) * scale;
                    #pragma unroll
                    for (int offset = 4; offset > 0; offset >>= 1) {
                        local += __shfl_down_sync(0xff, local, offset);
                    }
                    if (lane == 0) acc += local;
                }
            }
        }
    } else if (type_id == 2) {
        const float blk = iq1m_block_dot_256(x_shared, block, signed_grid, lane);
        if (lane == 0) acc += blk;
    } else {
        const uint8_t* scales = block;
        const uint8_t* qs = block + 16;
        const float d = gguf_block_scale_f16(block + 80);
        const float dmin = gguf_block_scale_f16(block + 82);
        for (int group = 0; group < 16; ++group) {
            const int half_block = group / 8;
            const int group_in_half = group % 8;
            const int shift = (group_in_half / 2) * 2;
            const int byte_start = half_block * 32 + (group_in_half % 2) * 16;
            const float qscale = d * static_cast<float>(scales[group] & 0x0f);
            const float base = dmin * static_cast<float>(scales[group] >> 4);
            const int k_start = group * 16;
            float local = 0.0f;
            if (lane < 16) {
                const uint8_t q = static_cast<uint8_t>((qs[byte_start + lane] >> shift) & 0x03);
                const float xv = x_shared[k_start + lane];
                local = qscale * xv * static_cast<float>(q) - base * xv;
                #pragma unroll
                for (int offset = 8; offset > 0; offset >>= 1) {
                    local += __shfl_down_sync(0xffff, local, offset);
                }
                if (lane == 0) acc += local;
            }
        }
    }
    return acc;
}

__device__ __forceinline__ void gguf_quant_block_dot_256_rows4(
    const float* __restrict__ x_shared,
    const uint8_t* __restrict__ block,
    const int8_t* __restrict__ signed_grid,
    int type_id,
    int lane,
    int valid_rows,
    float* __restrict__ acc) {
    if (type_id == 0) {
        const float d = gguf_block_scale_f16(block);
        const uint8_t* qs = block + 2;
        for (int sub = 0; sub < 8; ++sub) {
            const uint8_t* chunk = qs + sub * 8;
            const uint32_t aux = static_cast<uint32_t>(chunk[4]) |
                (static_cast<uint32_t>(chunk[5]) << 8) |
                (static_cast<uint32_t>(chunk[6]) << 16) |
                (static_cast<uint32_t>(chunk[7]) << 24);
            const float scale = 0.125f * d * static_cast<float>(2 * (aux >> 28) + 1);
            for (int part = 0; part < 4; ++part) {
                const int grid_id = static_cast<int>(chunk[part]);
                const int sign_idx = static_cast<int>((aux >> (7 * part)) & 127);
                const int8_t* vals = signed_grid + ((grid_id * 128 + sign_idx) * 8);
                const int k_start = sub * 32 + part * 8;
                if (lane < 8) {
                    float local0 = valid_rows > 0 ? x_shared[k_start + lane] * static_cast<float>(vals[lane]) * scale : 0.0f;
                    float local1 = valid_rows > 1 ? x_shared[256 + k_start + lane] * static_cast<float>(vals[lane]) * scale : 0.0f;
                    float local2 = valid_rows > 2 ? x_shared[512 + k_start + lane] * static_cast<float>(vals[lane]) * scale : 0.0f;
                    float local3 = valid_rows > 3 ? x_shared[768 + k_start + lane] * static_cast<float>(vals[lane]) * scale : 0.0f;
                    #pragma unroll
                    for (int offset = 4; offset > 0; offset >>= 1) {
                        local0 += __shfl_down_sync(0xff, local0, offset);
                        local1 += __shfl_down_sync(0xff, local1, offset);
                        local2 += __shfl_down_sync(0xff, local2, offset);
                        local3 += __shfl_down_sync(0xff, local3, offset);
                    }
                    if (lane == 0) {
                        acc[0] += local0;
                        if (valid_rows > 1) acc[1] += local1;
                        if (valid_rows > 2) acc[2] += local2;
                        if (valid_rows > 3) acc[3] += local3;
                    }
                }
            }
        }
    } else if (type_id == 2) {
        iq1m_block_dot_256_rows4(x_shared, block, signed_grid, lane, valid_rows, acc);
    } else {
        const uint8_t* scales = block;
        const uint8_t* qs = block + 16;
        const float d = gguf_block_scale_f16(block + 80);
        const float dmin = gguf_block_scale_f16(block + 82);
        for (int group = 0; group < 16; ++group) {
            const int half_block = group / 8;
            const int group_in_half = group % 8;
            const int shift = (group_in_half / 2) * 2;
            const int byte_start = half_block * 32 + (group_in_half % 2) * 16;
            const float qscale = d * static_cast<float>(scales[group] & 0x0f);
            const float base = dmin * static_cast<float>(scales[group] >> 4);
            const int k_start = group * 16;
            if (lane < 16) {
                const uint8_t q = static_cast<uint8_t>((qs[byte_start + lane] >> shift) & 0x03);
                const float qv = qscale * static_cast<float>(q) - base;
                float local0 = valid_rows > 0 ? x_shared[k_start + lane] * qv : 0.0f;
                float local1 = valid_rows > 1 ? x_shared[256 + k_start + lane] * qv : 0.0f;
                float local2 = valid_rows > 2 ? x_shared[512 + k_start + lane] * qv : 0.0f;
                float local3 = valid_rows > 3 ? x_shared[768 + k_start + lane] * qv : 0.0f;
                #pragma unroll
                for (int offset = 8; offset > 0; offset >>= 1) {
                    local0 += __shfl_down_sync(0xffff, local0, offset);
                    local1 += __shfl_down_sync(0xffff, local1, offset);
                    local2 += __shfl_down_sync(0xffff, local2, offset);
                    local3 += __shfl_down_sync(0xffff, local3, offset);
                }
                if (lane == 0) {
                    acc[0] += local0;
                    if (valid_rows > 1) acc[1] += local1;
                    if (valid_rows > 2) acc[2] += local2;
                    if (valid_rows > 3) acc[3] += local3;
                }
            }
        }
    }
}

__global__ void gguf_route_experts_from_segments_kernel(
    const int32_t* __restrict__ seg_starts,
    int32_t* __restrict__ route_experts,
    int n_experts) {
    const int expert = blockIdx.x;
    if (expert >= n_experts) return;
    const int start = seg_starts[expert];
    const int end = seg_starts[expert + 1];
    for (int route = start + threadIdx.x; route < end; route += blockDim.x) {
        route_experts[route] = expert;
    }
}

template <typename scalar_t>
__global__ void gguf_moe_w13_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ route_experts,
    const uint8_t* __restrict__ w1_blocks,
    const uint8_t* __restrict__ w3_blocks,
    const int8_t* __restrict__ signed_grid,
    float* __restrict__ gate,
    float* __restrict__ up,
    int routes,
    int dim,
    int inter_dim,
    int w1_blocks_per_row,
    int w1_block_bytes,
    int w1_type_id,
    int w3_blocks_per_row,
    int w3_block_bytes,
    int w3_type_id) {
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kGGUFQuantTileN + warp;
    if (route >= routes || warp >= kGGUFQuantTileN) return;

    __shared__ float x_shared[256];
    const int64_t token = route_tokens[route];
    const int expert = route_experts[route];
    const scalar_t* x_row = x + token * dim;
    const uint8_t* w1_row = w1_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * w1_blocks_per_row * w1_block_bytes;
    const uint8_t* w3_row = w3_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * w3_blocks_per_row * w3_block_bytes;
    float acc1 = 0.0f;
    float acc3 = 0.0f;
    for (int block_idx = 0; block_idx < w1_blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int k = lane; k < 256; k += 32) {
            const int x_idx = k_base + k;
            x_shared[k] = x_idx < dim ? static_cast<float>(x_row[x_idx]) : 0.0f;
        }
        __syncthreads();
        if (out_col < inter_dim) {
            acc1 += gguf_quant_block_dot_256(x_shared, w1_row + static_cast<int64_t>(block_idx) * w1_block_bytes, signed_grid, w1_type_id, lane);
            acc3 += gguf_quant_block_dot_256(x_shared, w3_row + static_cast<int64_t>(block_idx) * w3_block_bytes, signed_grid, w3_type_id, lane);
        }
        __syncthreads();
    }
    if (lane == 0 && out_col < inter_dim) {
        gate[static_cast<int64_t>(route) * inter_dim + out_col] = acc1;
        up[static_cast<int64_t>(route) * inter_dim + out_col] = acc3;
    }
}

template <typename scalar_t>
__global__ void gguf_q8_1_quantize_x_32_kernel(
    const scalar_t* __restrict__ x,
    int8_t* __restrict__ x_q,
    float* __restrict__ x_scale,
    int rows,
    int row_elems) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= rows || group * 32 >= row_elems || lane >= 32) return;
    const int k = group * 32 + lane;
    const float xv = k < row_elems ? static_cast<float>(x[static_cast<int64_t>(row) * row_elems + k]) : 0.0f;
    float maxv = fabsf(xv);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maxv = fmaxf(maxv, __shfl_down_sync(0xffffffff, maxv, offset));
    }
    const float scale = __shfl_sync(0xffffffff, maxv > 0.0f ? maxv / 127.0f : 1.0e-8f, 0);
    if (lane == 0) {
        x_scale[row * ((row_elems + 31) / 32) + group] = scale;
    }
    int q = __float2int_rn(xv / scale);
    q = max(-127, min(127, q));
    if (k < row_elems) {
        x_q[static_cast<int64_t>(row) * row_elems + k] = static_cast<int8_t>(q);
    }
}

__global__ void gguf_moe_w13_iq2_xxs_dp4a_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ route_experts,
    const uint8_t* __restrict__ w1_blocks,
    const uint8_t* __restrict__ w3_blocks,
    const int8_t* __restrict__ signed_grid,
    float* __restrict__ gate,
    float* __restrict__ up,
    int routes,
    int dim,
    int inter_dim,
    int blocks_per_row,
    int x_groups) {
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kGGUFQuantTileN + warp;
    if (route >= routes || warp >= kGGUFQuantTileN) return;

    const int64_t token = route_tokens[route];
    const int expert = route_experts[route];
    const int8_t* x_q_row = x_q + token * dim;
    const float* x_scale_row = x_scale + token * x_groups;
    const uint8_t* w1_row = w1_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 66;
    const uint8_t* w3_row = w3_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 66;

    __shared__ int x_shared_q[64];
    __shared__ float x_shared_scale[8];

    float acc1 = 0.0f;
    float acc3 = 0.0f;

    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        const int tid = threadIdx.x;
        if (tid < 64) {
            const int byte_off = tid * 4;
            int v = 0;
            if (k_base + byte_off + 4 <= dim) {
                v = *reinterpret_cast<const int*>(x_q_row + k_base + byte_off);
            }
            x_shared_q[tid] = v;
        }
        if (tid < 8) {
            const int sg_idx = block_idx * 8 + tid;
            x_shared_scale[tid] = sg_idx < x_groups ? x_scale_row[sg_idx] : 0.0f;
        }
        __syncthreads();

        if (out_col < inter_dim) {
            const int sub = lane >> 2;
            const int part = lane & 3;
            const uint8_t* w1_block = w1_row + static_cast<int64_t>(block_idx) * 66;
            const uint8_t* w3_block = w3_row + static_cast<int64_t>(block_idx) * 66;
            const uint8_t* chunk1 = w1_block + 2 + sub * 8;
            const uint8_t* chunk3 = w3_block + 2 + sub * 8;

            const float d1 = gguf_block_scale_f16(w1_block);
            const float d3 = gguf_block_scale_f16(w3_block);

            const uint32_t aux1 = static_cast<uint32_t>(chunk1[4]) |
                (static_cast<uint32_t>(chunk1[5]) << 8) |
                (static_cast<uint32_t>(chunk1[6]) << 16) |
                (static_cast<uint32_t>(chunk1[7]) << 24);
            const uint32_t aux3 = static_cast<uint32_t>(chunk3[4]) |
                (static_cast<uint32_t>(chunk3[5]) << 8) |
                (static_cast<uint32_t>(chunk3[6]) << 16) |
                (static_cast<uint32_t>(chunk3[7]) << 24);

            const int grid_id1 = chunk1[part];
            const int grid_id3 = chunk3[part];
            const int sign_idx1 = static_cast<int>((aux1 >> (7 * part)) & 127);
            const int sign_idx3 = static_cast<int>((aux3 >> (7 * part)) & 127);

            const float s1 = 0.125f * d1 * static_cast<float>(2 * (aux1 >> 28) + 1);
            const float s3 = 0.125f * d3 * static_cast<float>(2 * (aux3 >> 28) + 1);

            const int8_t* vals1 = signed_grid + (grid_id1 * 128 + sign_idx1) * 8;
            const int8_t* vals3 = signed_grid + (grid_id3 * 128 + sign_idx3) * 8;
            const int v1_p0 = *reinterpret_cast<const int*>(vals1);
            const int v1_p1 = *reinterpret_cast<const int*>(vals1 + 4);
            const int v3_p0 = *reinterpret_cast<const int*>(vals3);
            const int v3_p1 = *reinterpret_cast<const int*>(vals3 + 4);

            const int xq_base = sub * 8 + part * 2;
            const int x_p0 = x_shared_q[xq_base];
            const int x_p1 = x_shared_q[xq_base + 1];

            int sumi1 = __dp4a(v1_p0, x_p0, 0);
            sumi1 = __dp4a(v1_p1, x_p1, sumi1);
            int sumi3 = __dp4a(v3_p0, x_p0, 0);
            sumi3 = __dp4a(v3_p1, x_p1, sumi3);

            const float xs = x_shared_scale[sub];
            float local1 = s1 * xs * static_cast<float>(sumi1);
            float local3 = s3 * xs * static_cast<float>(sumi3);

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                local1 += __shfl_down_sync(0xffffffff, local1, offset);
                local3 += __shfl_down_sync(0xffffffff, local3, offset);
            }
            if (lane == 0) {
                acc1 += local1;
                acc3 += local3;
            }
        }
        __syncthreads();
    }

    if (lane == 0 && out_col < inter_dim) {
        gate[static_cast<int64_t>(route) * inter_dim + out_col] = acc1;
        up[static_cast<int64_t>(route) * inter_dim + out_col] = acc3;
    }
}

__global__ void gguf_moe_w13_iq2_xxs_dp4a_expert_tile_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w1_blocks,
    const uint8_t* __restrict__ w3_blocks,
    const int8_t* __restrict__ signed_grid,
    float* __restrict__ gate,
    float* __restrict__ up,
    int n_experts,
    int dim,
    int inter_dim,
    int blocks_per_row,
    int x_groups) {
    constexpr int kRouteTile = 8;
    const int expert = blockIdx.y;
    const int route_tile = blockIdx.z;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kGGUFQuantTileN + warp;
    if (expert >= n_experts || warp >= kGGUFQuantTileN) return;

    const int route_start = seg_starts[expert] + route_tile * kRouteTile;
    const int route_end = seg_starts[expert + 1];
    const int valid_rows = min(kRouteTile, route_end - route_start);
    if (valid_rows <= 0) return;

    __shared__ int x_shared_q[kRouteTile][64];
    __shared__ float x_shared_scale[kRouteTile][8];

    float acc1[kRouteTile];
    float acc3[kRouteTile];
    #pragma unroll
    for (int r = 0; r < kRouteTile; ++r) {
        acc1[r] = 0.0f;
        acc3[r] = 0.0f;
    }

    const uint8_t* w1_row = w1_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 66;
    const uint8_t* w3_row = w3_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 66;

    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        const int tid = threadIdx.x;
        #pragma unroll
        for (int r = 0; r < kRouteTile; ++r) {
            if (r < valid_rows) {
                const int64_t token = route_tokens[route_start + r];
                const int8_t* x_q_row = x_q + token * dim;
                const float* x_scale_row = x_scale + token * x_groups;
                if (tid < 64) {
                    const int byte_off = tid * 4;
                    int v = 0;
                    if (k_base + byte_off + 4 <= dim) {
                        v = *reinterpret_cast<const int*>(x_q_row + k_base + byte_off);
                    }
                    x_shared_q[r][tid] = v;
                }
                if (tid < 8) {
                    const int sg_idx = block_idx * 8 + tid;
                    x_shared_scale[r][tid] = sg_idx < x_groups ? x_scale_row[sg_idx] : 0.0f;
                }
            }
        }
        __syncthreads();

        if (out_col < inter_dim) {
            const int sub = lane >> 2;
            const int part = lane & 3;
            const uint8_t* w1_block = w1_row + static_cast<int64_t>(block_idx) * 66;
            const uint8_t* w3_block = w3_row + static_cast<int64_t>(block_idx) * 66;
            const uint8_t* chunk1 = w1_block + 2 + sub * 8;
            const uint8_t* chunk3 = w3_block + 2 + sub * 8;

            const float d1 = gguf_block_scale_f16(w1_block);
            const float d3 = gguf_block_scale_f16(w3_block);

            const uint32_t aux1 = static_cast<uint32_t>(chunk1[4]) |
                (static_cast<uint32_t>(chunk1[5]) << 8) |
                (static_cast<uint32_t>(chunk1[6]) << 16) |
                (static_cast<uint32_t>(chunk1[7]) << 24);
            const uint32_t aux3 = static_cast<uint32_t>(chunk3[4]) |
                (static_cast<uint32_t>(chunk3[5]) << 8) |
                (static_cast<uint32_t>(chunk3[6]) << 16) |
                (static_cast<uint32_t>(chunk3[7]) << 24);

            const int grid_id1 = chunk1[part];
            const int grid_id3 = chunk3[part];
            const int sign_idx1 = static_cast<int>((aux1 >> (7 * part)) & 127);
            const int sign_idx3 = static_cast<int>((aux3 >> (7 * part)) & 127);

            const float s1 = 0.125f * d1 * static_cast<float>(2 * (aux1 >> 28) + 1);
            const float s3 = 0.125f * d3 * static_cast<float>(2 * (aux3 >> 28) + 1);

            const int8_t* vals1 = signed_grid + (grid_id1 * 128 + sign_idx1) * 8;
            const int8_t* vals3 = signed_grid + (grid_id3 * 128 + sign_idx3) * 8;
            const int v1_p0 = *reinterpret_cast<const int*>(vals1);
            const int v1_p1 = *reinterpret_cast<const int*>(vals1 + 4);
            const int v3_p0 = *reinterpret_cast<const int*>(vals3);
            const int v3_p1 = *reinterpret_cast<const int*>(vals3 + 4);

            const int xq_base = sub * 8 + part * 2;
            #pragma unroll
            for (int r = 0; r < kRouteTile; ++r) {
                float local1 = 0.0f;
                float local3 = 0.0f;
                if (r < valid_rows) {
                    const int x_p0 = x_shared_q[r][xq_base];
                    const int x_p1 = x_shared_q[r][xq_base + 1];

                    int sumi1 = __dp4a(v1_p0, x_p0, 0);
                    sumi1 = __dp4a(v1_p1, x_p1, sumi1);
                    int sumi3 = __dp4a(v3_p0, x_p0, 0);
                    sumi3 = __dp4a(v3_p1, x_p1, sumi3);

                    const float xs = x_shared_scale[r][sub];
                    local1 = s1 * xs * static_cast<float>(sumi1);
                    local3 = s3 * xs * static_cast<float>(sumi3);
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    local1 += __shfl_down_sync(0xffffffff, local1, offset);
                    local3 += __shfl_down_sync(0xffffffff, local3, offset);
                }
                if (lane == 0) {
                    acc1[r] += local1;
                    acc3[r] += local3;
                }
            }
        }
        __syncthreads();
    }

    if (lane == 0 && out_col < inter_dim) {
        #pragma unroll
        for (int r = 0; r < kRouteTile; ++r) {
            if (r < valid_rows) {
                const int route = route_start + r;
                gate[static_cast<int64_t>(route) * inter_dim + out_col] = acc1[r];
                up[static_cast<int64_t>(route) * inter_dim + out_col] = acc3[r];
            }
        }
    }
}

__global__ void gguf_moe_w13_iq2_xxs_dp4a_subwarp_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ route_experts,
    const uint8_t* __restrict__ w1_blocks,
    const uint8_t* __restrict__ w3_blocks,
    const int8_t* __restrict__ signed_grid,
    float* __restrict__ gate,
    float* __restrict__ up,
    int routes,
    int dim,
    int inter_dim,
    int blocks_per_row,
    int x_groups) {
    constexpr int kSubwarpCols = 8;
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int sub_col = lane >> 2;
    const int part = lane & 3;
    const int out_col = blockIdx.x * (kGGUFQuantTileN * kSubwarpCols) + warp * kSubwarpCols + sub_col;
    if (route >= routes || warp >= kGGUFQuantTileN) return;

    const int64_t token = route_tokens[route];
    const int expert = route_experts[route];
    const int8_t* x_q_row = x_q + token * dim;
    const float* x_scale_row = x_scale + token * x_groups;

    __shared__ int x_shared_q[64];
    __shared__ float x_shared_scale[8];

    const unsigned mask = 0x0fu << (sub_col * 4);
    float acc1 = 0.0f;
    float acc3 = 0.0f;

    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        const int tid = threadIdx.x;
        if (tid < 64) {
            const int byte_off = tid * 4;
            int v = 0;
            if (k_base + byte_off + 4 <= dim) {
                v = *reinterpret_cast<const int*>(x_q_row + k_base + byte_off);
            }
            x_shared_q[tid] = v;
        }
        if (tid < 8) {
            const int sg_idx = block_idx * 8 + tid;
            x_shared_scale[tid] = sg_idx < x_groups ? x_scale_row[sg_idx] : 0.0f;
        }
        __syncthreads();

        if (out_col < inter_dim) {
            const uint8_t* w1_row = w1_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 66;
            const uint8_t* w3_row = w3_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 66;
            const uint8_t* w1_block = w1_row + static_cast<int64_t>(block_idx) * 66;
            const uint8_t* w3_block = w3_row + static_cast<int64_t>(block_idx) * 66;
            const float d1 = gguf_block_scale_f16(w1_block);
            const float d3 = gguf_block_scale_f16(w3_block);
            const float s1_base = 0.125f * d1;
            const float s3_base = 0.125f * d3;

            #pragma unroll
            for (int sub = 0; sub < 8; ++sub) {
                const uint8_t* chunk1 = w1_block + 2 + sub * 8;
                const uint8_t* chunk3 = w3_block + 2 + sub * 8;
                const uint32_t aux1 = static_cast<uint32_t>(chunk1[4]) |
                    (static_cast<uint32_t>(chunk1[5]) << 8) |
                    (static_cast<uint32_t>(chunk1[6]) << 16) |
                    (static_cast<uint32_t>(chunk1[7]) << 24);
                const uint32_t aux3 = static_cast<uint32_t>(chunk3[4]) |
                    (static_cast<uint32_t>(chunk3[5]) << 8) |
                    (static_cast<uint32_t>(chunk3[6]) << 16) |
                    (static_cast<uint32_t>(chunk3[7]) << 24);
                const int grid_id1 = chunk1[part];
                const int grid_id3 = chunk3[part];
                const int sign_idx1 = static_cast<int>((aux1 >> (7 * part)) & 127);
                const int sign_idx3 = static_cast<int>((aux3 >> (7 * part)) & 127);
                const float s1 = s1_base * static_cast<float>(2 * (aux1 >> 28) + 1);
                const float s3 = s3_base * static_cast<float>(2 * (aux3 >> 28) + 1);
                const int8_t* vals1 = signed_grid + (grid_id1 * 128 + sign_idx1) * 8;
                const int8_t* vals3 = signed_grid + (grid_id3 * 128 + sign_idx3) * 8;
                const int v1_p0 = *reinterpret_cast<const int*>(vals1);
                const int v1_p1 = *reinterpret_cast<const int*>(vals1 + 4);
                const int v3_p0 = *reinterpret_cast<const int*>(vals3);
                const int v3_p1 = *reinterpret_cast<const int*>(vals3 + 4);
                const int xq_base = sub * 8 + part * 2;
                const int x_p0 = x_shared_q[xq_base];
                const int x_p1 = x_shared_q[xq_base + 1];
                int sumi1 = __dp4a(v1_p0, x_p0, 0);
                sumi1 = __dp4a(v1_p1, x_p1, sumi1);
                int sumi3 = __dp4a(v3_p0, x_p0, 0);
                sumi3 = __dp4a(v3_p1, x_p1, sumi3);
                const float xs = x_shared_scale[sub];
                float local1 = s1 * xs * static_cast<float>(sumi1);
                float local3 = s3 * xs * static_cast<float>(sumi3);
                #pragma unroll
                for (int offset = 2; offset > 0; offset >>= 1) {
                    local1 += __shfl_down_sync(mask, local1, offset);
                    local3 += __shfl_down_sync(mask, local3, offset);
                }
                if (part == 0) {
                    acc1 += local1;
                    acc3 += local3;
                }
            }
        }
        __syncthreads();
    }

    if (part == 0 && out_col < inter_dim) {
        gate[static_cast<int64_t>(route) * inter_dim + out_col] = acc1;
        up[static_cast<int64_t>(route) * inter_dim + out_col] = acc3;
    }
}

__global__ void gguf_moe_w2_scatter_kernel(
    const float* __restrict__ hidden,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ route_experts,
    const uint8_t* __restrict__ w2_blocks,
    const int8_t* __restrict__ signed_grid,
    float* __restrict__ y,
    int routes,
    int dim,
    int inter_dim,
    int w2_blocks_per_row,
    int w2_block_bytes,
    int w2_type_id) {
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kGGUFQuantTileN + warp;
    if (route >= routes || warp >= kGGUFQuantTileN) return;

    __shared__ float hidden_shared[256];
    const int64_t token = route_tokens[route];
    const int expert = route_experts[route];
    const float* hidden_row = hidden + static_cast<int64_t>(route) * inter_dim;
    const uint8_t* w2_row = w2_blocks + (static_cast<int64_t>(expert) * dim + out_col) * w2_blocks_per_row * w2_block_bytes;
    float acc = 0.0f;
    for (int block_idx = 0; block_idx < w2_blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int k = lane; k < 256; k += 32) {
            const int h_idx = k_base + k;
            hidden_shared[k] = h_idx < inter_dim ? hidden_row[h_idx] : 0.0f;
        }
        __syncthreads();
        if (out_col < dim) {
            acc += gguf_quant_block_dot_256(hidden_shared, w2_row + static_cast<int64_t>(block_idx) * w2_block_bytes, signed_grid, w2_type_id, lane);
        }
        __syncthreads();
    }
    if (lane == 0 && out_col < dim) {
        atomicAdd(y + token * dim + out_col, acc);
    }
}

__global__ void gguf_route_swiglu_quantize_hidden_16_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ route_weights,
    int8_t* __restrict__ hidden_q,
    float* __restrict__ hidden_scale,
    int routes,
    int inter_dim,
    float swiglu_limit) {
    const int route = blockIdx.x;
    const int group = blockIdx.y;
    const int lane = threadIdx.x;
    if (route >= routes || group * 16 >= inter_dim || lane >= 16) return;
    const int k = group * 16 + lane;
    float g = k < inter_dim ? gate[static_cast<int64_t>(route) * inter_dim + k] : 0.0f;
    float u = k < inter_dim ? up[static_cast<int64_t>(route) * inter_dim + k] : 0.0f;
    if (swiglu_limit > 0.0f) {
        u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
        g = fminf(g, swiglu_limit);
    }
    const float v = (g / (1.0f + expf(-g))) * u * route_weights[route];
    float maxv = fabsf(v);
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        maxv = fmaxf(maxv, __shfl_down_sync(0xffff, maxv, offset));
    }
    const float scale = __shfl_sync(0xffff, maxv > 0.0f ? maxv / 127.0f : 1.0e-8f, 0);
    if (lane == 0) {
        hidden_scale[route * ((inter_dim + 15) / 16) + group] = scale;
    }
    int q = __float2int_rn(v / scale);
    q = max(-127, min(127, q));
    if (k < inter_dim) {
        hidden_q[static_cast<int64_t>(route) * inter_dim + k] = static_cast<int8_t>(q);
    }
}
__global__ void gguf_q8_1_quantize_hidden_16_kernel(
    const float* __restrict__ hidden,
    int8_t* __restrict__ hidden_q,
    float* __restrict__ hidden_scale,
    int routes,
    int inter_dim) {
    const int route = blockIdx.x;
    const int group = blockIdx.y;
    const int lane = threadIdx.x;
    if (route >= routes || group * 16 >= inter_dim || lane >= 16) return;
    const int k = group * 16 + lane;
    const float xv = k < inter_dim ? hidden[static_cast<int64_t>(route) * inter_dim + k] : 0.0f;
    float maxv = fabsf(xv);
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        maxv = fmaxf(maxv, __shfl_down_sync(0xffff, maxv, offset));
    }
    const float scale = __shfl_sync(0xffff, maxv > 0.0f ? maxv / 127.0f : 1.0e-8f, 0);
    if (lane == 0) {
        hidden_scale[route * ((inter_dim + 15) / 16) + group] = scale;
    }
    int q = __float2int_rn(xv / scale);
    q = max(-127, min(127, q));
    hidden_q[static_cast<int64_t>(route) * inter_dim + k] = static_cast<int8_t>(q);
}

__global__ void gguf_moe_w2_scatter_q2k_dp4a_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ route_experts,
    const uint8_t* __restrict__ w2_blocks,
    float* __restrict__ y,
    int routes,
    int dim,
    int inter_dim,
    int w2_blocks_per_row,
    int hidden_groups) {
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kGGUFQuantTileN + warp;
    if (route >= routes || warp >= kGGUFQuantTileN || out_col >= dim) return;

    const int64_t token = route_tokens[route];
    const int expert = route_experts[route];
    const int8_t* hidden_row = hidden_q + static_cast<int64_t>(route) * inter_dim;
    const float* hs_row = hidden_scale + static_cast<int64_t>(route) * hidden_groups;
    const uint8_t* w2_row = w2_blocks + (static_cast<int64_t>(expert) * dim + out_col) * w2_blocks_per_row * 84;
    float acc = 0.0f;

    for (int block_idx = 0; block_idx < w2_blocks_per_row; ++block_idx) {
        const uint8_t* block = w2_row + static_cast<int64_t>(block_idx) * 84;
        const uint8_t* scales = block;
        const uint8_t* qs = block + 16;
        const float d = gguf_block_scale_f16(block + 80);
        const float dmin = gguf_block_scale_f16(block + 82);
        const int k_base = block_idx * 256;
        for (int group = 0; group < 16; ++group) {
            float local = 0.0f;
            if (lane < 4) {
                const int half_block = group >> 3;
                const int group_in_half = group & 7;
                const int shift = (group_in_half >> 1) * 2;
                const int byte_start = half_block * 32 + (group_in_half & 1) * 16;
                const int idx = lane * 4;
                const int q0 = static_cast<int>((qs[byte_start + idx + 0] >> shift) & 0x03);
                const int q1 = static_cast<int>((qs[byte_start + idx + 1] >> shift) & 0x03);
                const int q2 = static_cast<int>((qs[byte_start + idx + 2] >> shift) & 0x03);
                const int q3 = static_cast<int>((qs[byte_start + idx + 3] >> shift) & 0x03);
                const int w_pack = pack_i8x4(q0, q1, q2, q3);
                const int h_idx = k_base + group * 16 + idx;
                const int h_pack = pack_i8x4(
                    static_cast<int>(hidden_row[h_idx + 0]),
                    static_cast<int>(hidden_row[h_idx + 1]),
                    static_cast<int>(hidden_row[h_idx + 2]),
                    static_cast<int>(hidden_row[h_idx + 3]));
                const int dot_q = __dp4a(w_pack, h_pack, 0);
                const int sum_h = __dp4a(0x01010101, h_pack, 0);
                const float qscale = d * static_cast<float>(scales[group] & 0x0f);
                const float base = dmin * static_cast<float>(scales[group] >> 4);
                local = hs_row[block_idx * 16 + group] * (qscale * static_cast<float>(dot_q) - base * static_cast<float>(sum_h));
            }
            #pragma unroll
            for (int offset = 2; offset > 0; offset >>= 1) {
                local += __shfl_down_sync(0x0f, local, offset);
            }
            if (lane == 0) acc += local;
        }
    }

    if (lane == 0) {
        atomicAdd(y + token * dim + out_col, acc);
    }
}

__global__ void gguf_moe_w2_scatter_q2k_dp4a_subwarp_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ route_experts,
    const uint8_t* __restrict__ w2_blocks,
    float* __restrict__ y,
    int routes,
    int dim,
    int inter_dim,
    int w2_blocks_per_row,
    int hidden_groups) {
    constexpr int kSubwarpCols = 8;
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int sub = lane >> 2;
    const int part = lane & 3;
    const int out_col = blockIdx.x * (kGGUFQuantTileN * kSubwarpCols) + warp * kSubwarpCols + sub;
    if (route >= routes || warp >= kGGUFQuantTileN || out_col >= dim) return;

    const int64_t token = route_tokens[route];
    const int expert = route_experts[route];
    const int8_t* hidden_row = hidden_q + static_cast<int64_t>(route) * inter_dim;
    const float* hs_row = hidden_scale + static_cast<int64_t>(route) * hidden_groups;
    const uint8_t* w2_row = w2_blocks + (static_cast<int64_t>(expert) * dim + out_col) * w2_blocks_per_row * 84;
    const unsigned mask = 0x0fu << (sub * 4);
    float acc = 0.0f;

    for (int block_idx = 0; block_idx < w2_blocks_per_row; ++block_idx) {
        const uint8_t* block = w2_row + static_cast<int64_t>(block_idx) * 84;
        const uint8_t* scales = block;
        const uint8_t* qs = block + 16;
        const float d = gguf_block_scale_f16(block + 80);
        const float dmin = gguf_block_scale_f16(block + 82);
        const int k_base = block_idx * 256;
        for (int group = 0; group < 16; ++group) {
            const int half_block = group >> 3;
            const int group_in_half = group & 7;
            const int shift = (group_in_half >> 1) * 2;
            const int byte_start = half_block * 32 + (group_in_half & 1) * 16;
            const int idx = part * 4;
            const int q0 = static_cast<int>((qs[byte_start + idx + 0] >> shift) & 0x03);
            const int q1 = static_cast<int>((qs[byte_start + idx + 1] >> shift) & 0x03);
            const int q2 = static_cast<int>((qs[byte_start + idx + 2] >> shift) & 0x03);
            const int q3 = static_cast<int>((qs[byte_start + idx + 3] >> shift) & 0x03);
            const int w_pack = pack_i8x4(q0, q1, q2, q3);
            const int h_idx = k_base + group * 16 + idx;
            const int h_pack = pack_i8x4(
                static_cast<int>(hidden_row[h_idx + 0]),
                static_cast<int>(hidden_row[h_idx + 1]),
                static_cast<int>(hidden_row[h_idx + 2]),
                static_cast<int>(hidden_row[h_idx + 3]));
            const int dot_q = __dp4a(w_pack, h_pack, 0);
            const int sum_h = __dp4a(0x01010101, h_pack, 0);
            const float qscale = d * static_cast<float>(scales[group] & 0x0f);
            const float base = dmin * static_cast<float>(scales[group] >> 4);
            float local = hs_row[block_idx * 16 + group] * (qscale * static_cast<float>(dot_q) - base * static_cast<float>(sum_h));
            #pragma unroll
            for (int offset = 2; offset > 0; offset >>= 1) {
                local += __shfl_down_sync(mask, local, offset);
            }
            if (part == 0) acc += local;
        }
    }

    if (part == 0) {
        atomicAdd(y + token * dim + out_col, acc);
    }
}

__global__ void gguf_moe_w2_scatter_q2k_dp4a_expert_tile_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w2_blocks,
    float* __restrict__ y,
    int n_experts,
    int dim,
    int inter_dim,
    int w2_blocks_per_row,
    int hidden_groups) {
    constexpr int kRouteTile = 8;
    constexpr int kSubwarpCols = 8;
    const int expert = blockIdx.y;
    const int route_tile = blockIdx.z;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int sub = lane >> 2;
    const int part = lane & 3;
    const int out_col = blockIdx.x * (kGGUFQuantTileN * kSubwarpCols) + warp * kSubwarpCols + sub;
    if (expert >= n_experts || warp >= kGGUFQuantTileN || out_col >= dim) return;

    const int route_start = seg_starts[expert] + route_tile * kRouteTile;
    const int route_end = seg_starts[expert + 1];
    const int valid_rows = min(kRouteTile, route_end - route_start);
    if (valid_rows <= 0) return;

    const uint8_t* w2_row = w2_blocks + (static_cast<int64_t>(expert) * dim + out_col) * w2_blocks_per_row * 84;
    const unsigned mask = 0x0fu << (sub * 4);
    float acc[kRouteTile];
    #pragma unroll
    for (int r = 0; r < kRouteTile; ++r) {
        acc[r] = 0.0f;
    }

    for (int block_idx = 0; block_idx < w2_blocks_per_row; ++block_idx) {
        const uint8_t* block = w2_row + static_cast<int64_t>(block_idx) * 84;
        const uint8_t* scales = block;
        const uint8_t* qs = block + 16;
        const float d = gguf_block_scale_f16(block + 80);
        const float dmin = gguf_block_scale_f16(block + 82);
        const int k_base = block_idx * 256;
        for (int group = 0; group < 16; ++group) {
            const int half_block = group >> 3;
            const int group_in_half = group & 7;
            const int shift = (group_in_half >> 1) * 2;
            const int byte_start = half_block * 32 + (group_in_half & 1) * 16;
            const int idx = part * 4;
            const int q0 = static_cast<int>((qs[byte_start + idx + 0] >> shift) & 0x03);
            const int q1 = static_cast<int>((qs[byte_start + idx + 1] >> shift) & 0x03);
            const int q2 = static_cast<int>((qs[byte_start + idx + 2] >> shift) & 0x03);
            const int q3 = static_cast<int>((qs[byte_start + idx + 3] >> shift) & 0x03);
            const int w_pack = pack_i8x4(q0, q1, q2, q3);
            const float qscale = d * static_cast<float>(scales[group] & 0x0f);
            const float base = dmin * static_cast<float>(scales[group] >> 4);
            const int h_idx = k_base + group * 16 + idx;
            #pragma unroll
            for (int r = 0; r < kRouteTile; ++r) {
                float local = 0.0f;
                if (r < valid_rows) {
                    const int route = route_start + r;
                    const int8_t* hidden_row = hidden_q + static_cast<int64_t>(route) * inter_dim;
                    const float* hs_row = hidden_scale + static_cast<int64_t>(route) * hidden_groups;
                    const int h_pack = pack_i8x4(
                        static_cast<int>(hidden_row[h_idx + 0]),
                        static_cast<int>(hidden_row[h_idx + 1]),
                        static_cast<int>(hidden_row[h_idx + 2]),
                        static_cast<int>(hidden_row[h_idx + 3]));
                    const int dot_q = __dp4a(w_pack, h_pack, 0);
                    const int sum_h = __dp4a(0x01010101, h_pack, 0);
                    local = hs_row[block_idx * 16 + group] * (qscale * static_cast<float>(dot_q) - base * static_cast<float>(sum_h));
                }
                #pragma unroll
                for (int offset = 2; offset > 0; offset >>= 1) {
                    local += __shfl_down_sync(mask, local, offset);
                }
                if (part == 0) acc[r] += local;
            }
        }
    }

    if (part == 0) {
        #pragma unroll
        for (int r = 0; r < kRouteTile; ++r) {
            if (r < valid_rows) {
                const int route = route_start + r;
                const int64_t token = route_tokens[route];
                atomicAdd(y + token * dim + out_col, acc[r]);
            }
        }
    }
}

__global__ void gguf_moe_single_w13_iq2_xxs_dp4a_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const int64_t* __restrict__ route_slots,
    const uint8_t* __restrict__ w1_blocks,
    const uint8_t* __restrict__ w3_blocks,
    const int8_t* __restrict__ signed_grid,
    float* __restrict__ gate,
    float* __restrict__ up,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    int blocks_per_row,
    int x_groups) {
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kGGUFQuantTileN + warp;
    if (route >= routes || warp >= kGGUFQuantTileN) return;
    const int expert = static_cast<int>(route_slots[route]);
    if (expert < 0 || expert >= n_experts) return;

    const uint8_t* w1_row = w1_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 66;
    const uint8_t* w3_row = w3_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 66;

    __shared__ int x_shared_q[64];
    __shared__ float x_shared_scale[8];

    float acc1 = 0.0f;
    float acc3 = 0.0f;

    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        const int tid = threadIdx.x;
        if (tid < 64) {
            const int byte_off = tid * 4;
            int v = 0;
            if (k_base + byte_off + 4 <= dim) {
                v = *reinterpret_cast<const int*>(x_q + k_base + byte_off);
            }
            x_shared_q[tid] = v;
        }
        if (tid < 8) {
            const int sg_idx = block_idx * 8 + tid;
            x_shared_scale[tid] = sg_idx < x_groups ? x_scale[sg_idx] : 0.0f;
        }
        __syncthreads();

        if (out_col < inter_dim) {
            const int sub = lane >> 2;
            const int part = lane & 3;
            const uint8_t* w1_block = w1_row + static_cast<int64_t>(block_idx) * 66;
            const uint8_t* w3_block = w3_row + static_cast<int64_t>(block_idx) * 66;
            const uint8_t* chunk1 = w1_block + 2 + sub * 8;
            const uint8_t* chunk3 = w3_block + 2 + sub * 8;

            const float d1 = gguf_block_scale_f16(w1_block);
            const float d3 = gguf_block_scale_f16(w3_block);

            const uint32_t aux1 = static_cast<uint32_t>(chunk1[4]) |
                (static_cast<uint32_t>(chunk1[5]) << 8) |
                (static_cast<uint32_t>(chunk1[6]) << 16) |
                (static_cast<uint32_t>(chunk1[7]) << 24);
            const uint32_t aux3 = static_cast<uint32_t>(chunk3[4]) |
                (static_cast<uint32_t>(chunk3[5]) << 8) |
                (static_cast<uint32_t>(chunk3[6]) << 16) |
                (static_cast<uint32_t>(chunk3[7]) << 24);

            const int grid_id1 = chunk1[part];
            const int grid_id3 = chunk3[part];
            const int sign_idx1 = static_cast<int>((aux1 >> (7 * part)) & 127);
            const int sign_idx3 = static_cast<int>((aux3 >> (7 * part)) & 127);

            const float s1 = 0.125f * d1 * static_cast<float>(2 * (aux1 >> 28) + 1);
            const float s3 = 0.125f * d3 * static_cast<float>(2 * (aux3 >> 28) + 1);

            const int8_t* vals1 = signed_grid + (grid_id1 * 128 + sign_idx1) * 8;
            const int8_t* vals3 = signed_grid + (grid_id3 * 128 + sign_idx3) * 8;
            const int v1_p0 = *reinterpret_cast<const int*>(vals1);
            const int v1_p1 = *reinterpret_cast<const int*>(vals1 + 4);
            const int v3_p0 = *reinterpret_cast<const int*>(vals3);
            const int v3_p1 = *reinterpret_cast<const int*>(vals3 + 4);

            const int xq_base = sub * 8 + part * 2;
            const int x_p0 = x_shared_q[xq_base];
            const int x_p1 = x_shared_q[xq_base + 1];

            int sumi1 = __dp4a(v1_p0, x_p0, 0);
            sumi1 = __dp4a(v1_p1, x_p1, sumi1);
            int sumi3 = __dp4a(v3_p0, x_p0, 0);
            sumi3 = __dp4a(v3_p1, x_p1, sumi3);

            const float xs = x_shared_scale[sub];
            float local1 = s1 * xs * static_cast<float>(sumi1);
            float local3 = s3 * xs * static_cast<float>(sumi3);

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                local1 += __shfl_down_sync(0xffffffff, local1, offset);
                local3 += __shfl_down_sync(0xffffffff, local3, offset);
            }
            if (lane == 0) {
                acc1 += local1;
                acc3 += local3;
            }
        }
        __syncthreads();
    }

    if (lane == 0 && out_col < inter_dim) {
        gate[static_cast<int64_t>(route) * inter_dim + out_col] = acc1;
        up[static_cast<int64_t>(route) * inter_dim + out_col] = acc3;
    }
}

__global__ void gguf_moe_single_w2_q2k_dp4a_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_slots,
    const uint8_t* __restrict__ w2_blocks,
    float* __restrict__ y,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    int w2_blocks_per_row,
    int hidden_groups) {
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kGGUFQuantTileN + warp;
    if (route >= routes || warp >= kGGUFQuantTileN || out_col >= dim) return;
    const int expert = static_cast<int>(route_slots[route]);
    if (expert < 0 || expert >= n_experts) return;

    const int8_t* hidden_row = hidden_q + static_cast<int64_t>(route) * inter_dim;
    const float* hs_row = hidden_scale + static_cast<int64_t>(route) * hidden_groups;
    const uint8_t* w2_row = w2_blocks + (static_cast<int64_t>(expert) * dim + out_col) * w2_blocks_per_row * 84;
    float acc = 0.0f;

    for (int block_idx = 0; block_idx < w2_blocks_per_row; ++block_idx) {
        const uint8_t* block = w2_row + static_cast<int64_t>(block_idx) * 84;
        const uint8_t* scales = block;
        const uint8_t* qs = block + 16;
        const float d = gguf_block_scale_f16(block + 80);
        const float dmin = gguf_block_scale_f16(block + 82);
        const int k_base = block_idx * 256;
        for (int group = 0; group < 16; ++group) {
            float local = 0.0f;
            if (lane < 4) {
                const int half_block = group >> 3;
                const int group_in_half = group & 7;
                const int shift = (group_in_half >> 1) * 2;
                const int byte_start = half_block * 32 + (group_in_half & 1) * 16;
                const int idx = lane * 4;
                const int q0 = static_cast<int>((qs[byte_start + idx + 0] >> shift) & 0x03);
                const int q1 = static_cast<int>((qs[byte_start + idx + 1] >> shift) & 0x03);
                const int q2 = static_cast<int>((qs[byte_start + idx + 2] >> shift) & 0x03);
                const int q3 = static_cast<int>((qs[byte_start + idx + 3] >> shift) & 0x03);
                const int w_pack = pack_i8x4(q0, q1, q2, q3);
                const int h_idx = k_base + group * 16 + idx;
                const int h_pack = pack_i8x4(
                    static_cast<int>(hidden_row[h_idx + 0]),
                    static_cast<int>(hidden_row[h_idx + 1]),
                    static_cast<int>(hidden_row[h_idx + 2]),
                    static_cast<int>(hidden_row[h_idx + 3]));
                const int dot_q = __dp4a(w_pack, h_pack, 0);
                const int sum_h = __dp4a(0x01010101, h_pack, 0);
                const float qscale = d * static_cast<float>(scales[group] & 0x0f);
                const float base = dmin * static_cast<float>(scales[group] >> 4);
                local = hs_row[block_idx * 16 + group] * (qscale * static_cast<float>(dot_q) - base * static_cast<float>(sum_h));
            }
            #pragma unroll
            for (int offset = 2; offset > 0; offset >>= 1) {
                local += __shfl_down_sync(0x0f, local, offset);
            }
            if (lane == 0) acc += local;
        }
    }

    if (lane == 0) {
        atomicAdd(y + out_col, acc);
    }
}

template <typename scalar_t>
__global__ void q8_0_gemm_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ blocks,
    c10::BFloat16* __restrict__ out,
    int rows,
    int n,
    int row_elems,
    int blocks_per_row) {
    const int row = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kQ8_0TileN + warp;
    if (row >= rows || warp >= kQ8_0TileN) return;

    __shared__ float x_shared[32];
    float acc = 0.0f;
    const scalar_t* x_row = x + row * row_elems;
    const uint8_t* weight_row = blocks + (out_col * blocks_per_row * 34);
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k = block_idx * 32 + lane;
        if (warp == 0) {
            x_shared[lane] = k < row_elems ? static_cast<float>(x_row[k]) : 0.0f;
        }
        __syncthreads();
        if (out_col < n && k < row_elems) {
            const uint8_t* block = weight_row + block_idx * 34;
            const float scale = q8_0_block_scale(block);
            const int8_t q = reinterpret_cast<const int8_t*>(block + 2)[lane];
            acc += x_shared[lane] * (static_cast<float>(q) * scale);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0 && out_col < n) {
        out[row * n + out_col] = c10::BFloat16(acc);
    }
}


}  // namespace

torch::Tensor gguf_q2k_gemm_dp4a_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& blocks,
    int64_t row_elems) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto x_contig = x.contiguous();
    auto blocks_contig = blocks.contiguous();
    const int k = static_cast<int>(row_elems);
    const int n = static_cast<int>(blocks_contig.size(0));
    const int blocks_per_row = static_cast<int>(blocks_contig.size(1));
    const int rows = static_cast<int>(x_contig.numel() / k);
    auto out = torch::empty({rows, n}, x.options().dtype(torch::kBFloat16));
    auto x_q = torch::empty({rows, k}, x.options().dtype(torch::kInt8));
    const int x_groups = (k + 15) / 16;
    auto x_scale = torch::empty({rows, x_groups}, x.options().dtype(torch::kFloat32));
    const dim3 quant_grid(rows, x_groups);
    const dim3 quant_block(16);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "gguf_q2k_q8_1_quantize", [&] {
        gguf_q8_1_quantize_16_kernel<scalar_t><<<quant_grid, quant_block, 0, at::cuda::getCurrentCUDAStream()>>>(
            x_contig.data_ptr<scalar_t>(),
            x_q.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(),
            rows,
            k);
    });
    const dim3 grid(ceil_div(n, kGGUFQuantTileN), rows);
    const dim3 block(256);
    gguf_q2k_gemm_dp4a_prequant_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x_q.data_ptr<int8_t>(),
        x_scale.data_ptr<float>(),
        blocks_contig.data_ptr<uint8_t>(),
        out.data_ptr<c10::BFloat16>(),
        rows,
        n,
        k,
        blocks_per_row,
        x_groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto out_shape = x.sizes().vec();
    out_shape.back() = n;
    return out.view(out_shape);
}


torch::Tensor q8_0_gemm_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& blocks,
    int64_t row_elems) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto x_contig = x.contiguous();
    auto blocks_contig = blocks.contiguous();
    const int k = static_cast<int>(row_elems);
    const int n = static_cast<int>(blocks_contig.size(0));
    const int blocks_per_row = static_cast<int>(blocks_contig.size(1));
    const int rows = static_cast<int>(x_contig.numel() / k);
    auto out = torch::empty({rows, n}, x.options().dtype(torch::kBFloat16));
    const dim3 grid(ceil_div(n, kQ8_0TileN), rows);
    const dim3 block(256);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "q8_0_gemm_forward", [&] {
        q8_0_gemm_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            x_contig.data_ptr<scalar_t>(),
            blocks_contig.data_ptr<uint8_t>(),
            out.data_ptr<c10::BFloat16>(),
            rows,
            n,
            k,
            blocks_per_row);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto out_shape = x.sizes().vec();
    out_shape.back() = n;
    return out.view(out_shape);
}

torch::Tensor gguf_quant_gemm_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& blocks,
    int64_t row_elems,
    int64_t type_id,
    const torch::Tensor& signed_grid) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto x_contig = x.contiguous();
    auto blocks_contig = blocks.contiguous();
    const int k = static_cast<int>(row_elems);
    const int n = static_cast<int>(blocks_contig.size(0));
    const int blocks_per_row = static_cast<int>(blocks_contig.size(1));
    const int block_bytes = static_cast<int>(blocks_contig.size(2));
    const int rows = static_cast<int>(x_contig.numel() / k);
    auto out = torch::empty({rows, n}, x.options().dtype(torch::kBFloat16));
    const int8_t* grid_ptr = nullptr;
    torch::Tensor grid_contig;
    if (type_id == 0 || type_id == 2) {
        grid_contig = signed_grid.contiguous();
        grid_ptr = grid_contig.data_ptr<int8_t>();
    }
    const dim3 grid(ceil_div(n, kGGUFQuantTileN), rows);
    const dim3 block(256);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "gguf_quant_gemm_forward", [&] {
        gguf_quant_gemm_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            x_contig.data_ptr<scalar_t>(),
            blocks_contig.data_ptr<uint8_t>(),
            grid_ptr,
            out.data_ptr<c10::BFloat16>(),
            rows,
            n,
            k,
            blocks_per_row,
            block_bytes,
            static_cast<int>(type_id));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto out_shape = x.sizes().vec();
    out_shape.back() = n;
    return out.view(out_shape);
}

torch::Tensor gguf_quant_gemm_prefill_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& blocks,
    int64_t row_elems,
    int64_t type_id,
    const torch::Tensor& signed_grid) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto x_contig = x.contiguous();
    auto blocks_contig = blocks.contiguous();
    const int k = static_cast<int>(row_elems);
    const int n = static_cast<int>(blocks_contig.size(0));
    const int blocks_per_row = static_cast<int>(blocks_contig.size(1));
    const int block_bytes = static_cast<int>(blocks_contig.size(2));
    const int rows = static_cast<int>(x_contig.numel() / k);
    auto out = torch::empty({rows, n}, x.options().dtype(torch::kBFloat16));
    const int8_t* grid_ptr = nullptr;
    torch::Tensor grid_contig;
    if (type_id == 0 || type_id == 2) {
        grid_contig = signed_grid.contiguous();
        grid_ptr = grid_contig.data_ptr<int8_t>();
    }
    const dim3 grid(ceil_div(n, kGGUFQuantTileN), ceil_div(rows, kGGUFQuantPrefillRows));
    const dim3 block(256);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "gguf_quant_gemm_prefill_forward", [&] {
        gguf_quant_gemm_prefill_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            x_contig.data_ptr<scalar_t>(),
            blocks_contig.data_ptr<uint8_t>(),
            grid_ptr,
            out.data_ptr<c10::BFloat16>(),
            rows,
            n,
            k,
            blocks_per_row,
            block_bytes,
            static_cast<int>(type_id));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto out_shape = x.sizes().vec();
    out_shape.back() = n;
    return out.view(out_shape);
}

torch::Tensor gguf_quant_gemm_pair_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& blocks0,
    int64_t row_elems0,
    int64_t type_id0,
    const torch::Tensor& blocks1,
    int64_t row_elems1,
    int64_t type_id1,
    const torch::Tensor& signed_grid) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto x_contig = x.contiguous();
    auto blocks0_contig = blocks0.contiguous();
    auto blocks1_contig = blocks1.contiguous();
    const int k = static_cast<int>(row_elems0);
    const int n = static_cast<int>(blocks0_contig.size(0));
    const int blocks0_per_row = static_cast<int>(blocks0_contig.size(1));
    const int blocks1_per_row = static_cast<int>(blocks1_contig.size(1));
    const int block0_bytes = static_cast<int>(blocks0_contig.size(2));
    const int block1_bytes = static_cast<int>(blocks1_contig.size(2));
    const int rows = static_cast<int>(x_contig.numel() / k);
    auto out0 = torch::empty({rows, n}, x.options().dtype(torch::kBFloat16));
    auto out1 = torch::empty({rows, n}, x.options().dtype(torch::kBFloat16));
    const int8_t* grid_ptr = nullptr;
    torch::Tensor grid_contig;
    if (type_id0 == 0 || type_id1 == 0 || type_id0 == 2 || type_id1 == 2) {
        grid_contig = signed_grid.contiguous();
        grid_ptr = grid_contig.data_ptr<int8_t>();
    }
    const dim3 grid(ceil_div(n, kGGUFQuantTileN), rows);
    const dim3 block(256);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "gguf_quant_gemm_pair_forward", [&] {
        gguf_quant_gemm_pair_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            x_contig.data_ptr<scalar_t>(),
            blocks0_contig.data_ptr<uint8_t>(),
            blocks1_contig.data_ptr<uint8_t>(),
            grid_ptr,
            out0.data_ptr<c10::BFloat16>(),
            out1.data_ptr<c10::BFloat16>(),
            rows,
            n,
            k,
            blocks0_per_row,
            block0_bytes,
            static_cast<int>(type_id0),
            blocks1_per_row,
            block1_bytes,
            static_cast<int>(type_id1));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto out_shape = x.sizes().vec();
    out_shape.back() = n;
    return torch::stack({out0.view(out_shape), out1.view(out_shape)}, 0);
}

torch::Tensor gguf_moe_prefill_grouped_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& route_tokens,
    const torch::Tensor& route_weights_sorted,
    const torch::Tensor& seg_starts,
    const torch::Tensor& w1_blocks,
    const torch::Tensor& w3_blocks,
    const torch::Tensor& w2_blocks,
    int64_t w1_row_elems,
    int64_t w1_type_id,
    int64_t w3_row_elems,
    int64_t w3_type_id,
    int64_t w2_row_elems,
    int64_t w2_type_id,
    const torch::Tensor& signed_grid,
    double swiglu_limit) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto x_contig = x.contiguous();
    auto route_tokens_contig = route_tokens.contiguous();
    auto route_weights_contig = route_weights_sorted.contiguous();
    auto seg_i32 = seg_starts.scalar_type() == torch::kInt32 ? seg_starts.contiguous() : seg_starts.to(torch::kInt32);
    auto w1_contig = w1_blocks.contiguous();
    auto w3_contig = w3_blocks.contiguous();
    auto w2_contig = w2_blocks.contiguous();
    const int routes = static_cast<int>(route_tokens_contig.size(0));
    const int tokens = static_cast<int>(x_contig.size(0));
    const int dim = static_cast<int>(x_contig.size(1));
    const int n_experts = static_cast<int>(w1_contig.size(0));
    const int inter_dim = static_cast<int>(w1_contig.size(1));
    auto y = torch::zeros({tokens, dim}, x.options().dtype(torch::kFloat32));
    if (routes == 0 || n_experts <= 0) {
        return y;
    }

    const int w1_blocks_per_row = static_cast<int>(w1_contig.size(2));
    const int w1_block_bytes = static_cast<int>(w1_contig.size(3));
    const int w3_blocks_per_row = static_cast<int>(w3_contig.size(2));
    const int w3_block_bytes = static_cast<int>(w3_contig.size(3));
    const int w2_blocks_per_row = static_cast<int>(w2_contig.size(2));
    const int w2_block_bytes = static_cast<int>(w2_contig.size(3));
    auto route_experts = torch::empty({routes}, seg_i32.options());
    gguf_route_experts_from_segments_kernel<<<n_experts, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        seg_i32.data_ptr<int32_t>(),
        route_experts.data_ptr<int32_t>(),
        n_experts);

    const int8_t* grid_ptr = nullptr;
    torch::Tensor grid_contig;
    if (w1_type_id == 0 || w3_type_id == 0 || w2_type_id == 0 || w1_type_id == 2 || w3_type_id == 2 || w2_type_id == 2) {
        grid_contig = signed_grid.contiguous();
        grid_ptr = grid_contig.data_ptr<int8_t>();
    }

    const bool profile = env_enabled_explicit("DEEPSEEK_GGUF_PREFILL_KERNEL_PROFILE");
    const int profile_limit = std::max(0, env_int_default("DEEPSEEK_GGUF_PREFILL_KERNEL_PROFILE_LIMIT", 256));
    const int profile_index = profile ? g_gguf_prefill_profile_count.fetch_add(1) : 0;
    const bool emit_profile = profile && profile_index < profile_limit;
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_w13 = nullptr;
    cudaEvent_t ev_hidden = nullptr;
    cudaEvent_t ev_w2 = nullptr;
    if (emit_profile) {
        check_cuda(cudaEventCreate(&ev_start), "create gguf prefill profile start event");
        check_cuda(cudaEventCreate(&ev_w13), "create gguf prefill profile w13 event");
        check_cuda(cudaEventCreate(&ev_hidden), "create gguf prefill profile hidden event");
        check_cuda(cudaEventCreate(&ev_w2), "create gguf prefill profile w2 event");
        check_cuda(cudaEventRecord(ev_start, at::cuda::getCurrentCUDAStream()), "record gguf prefill profile start event");
    }

    auto gate = torch::empty({routes, inter_dim}, x.options().dtype(torch::kFloat32));
    auto up = torch::empty({routes, inter_dim}, x.options().dtype(torch::kFloat32));
    const dim3 w13_grid(ceil_div(inter_dim, kGGUFQuantTileN), routes);
    const dim3 block(256);
    const bool use_iq2_xxs_w13_dp4a =
        w1_type_id == 0 && w3_type_id == 0 &&
        w1_block_bytes == 66 && w3_block_bytes == 66 &&
        w1_blocks_per_row == w3_blocks_per_row &&
        env_enabled_explicit("DEEPSEEK_GGUF_IQ2_XXS_W13_DP4A");
    if (use_iq2_xxs_w13_dp4a) {
        const int x_groups = ceil_div(dim, 32);
        auto x_q = torch::empty({tokens, dim}, x.options().dtype(torch::kInt8));
        auto x_scale = torch::empty({tokens, x_groups}, x.options().dtype(torch::kFloat32));
        const dim3 quant_grid(tokens, x_groups);
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "gguf_q8_1_quantize_x_32", [&] {
            gguf_q8_1_quantize_x_32_kernel<scalar_t><<<quant_grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
                x_contig.data_ptr<scalar_t>(),
                x_q.data_ptr<int8_t>(),
                x_scale.data_ptr<float>(),
                tokens,
                dim);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (env_enabled_explicit("DEEPSEEK_GGUF_IQ2_XXS_W13_EXPERT_TILE")) {
            auto counts_i32 = seg_i32.slice(0, 1, n_experts + 1) - seg_i32.slice(0, 0, n_experts);
            const int max_count = counts_i32.max().item<int>();
            const dim3 expert_tile_grid(ceil_div(inter_dim, kGGUFQuantTileN), n_experts, ceil_div(max_count, 8));
            gguf_moe_w13_iq2_xxs_dp4a_expert_tile_kernel<<<expert_tile_grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                x_q.data_ptr<int8_t>(),
                x_scale.data_ptr<float>(),
                route_tokens_contig.data_ptr<int64_t>(),
                seg_i32.data_ptr<int32_t>(),
                w1_contig.data_ptr<uint8_t>(),
                w3_contig.data_ptr<uint8_t>(),
                grid_ptr,
                gate.data_ptr<float>(),
                up.data_ptr<float>(),
                n_experts,
                dim,
                inter_dim,
                w1_blocks_per_row,
                x_groups);
        } else if (env_enabled_explicit("DEEPSEEK_GGUF_IQ2_XXS_W13_SUBWARP")) {
            const dim3 subwarp_grid(ceil_div(inter_dim, kGGUFQuantTileN * 8), routes);
            gguf_moe_w13_iq2_xxs_dp4a_subwarp_kernel<<<subwarp_grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                x_q.data_ptr<int8_t>(),
                x_scale.data_ptr<float>(),
                route_tokens_contig.data_ptr<int64_t>(),
                route_experts.data_ptr<int32_t>(),
                w1_contig.data_ptr<uint8_t>(),
                w3_contig.data_ptr<uint8_t>(),
                grid_ptr,
                gate.data_ptr<float>(),
                up.data_ptr<float>(),
                routes,
                dim,
                inter_dim,
                w1_blocks_per_row,
                x_groups);
        } else {
            gguf_moe_w13_iq2_xxs_dp4a_kernel<<<w13_grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                x_q.data_ptr<int8_t>(),
                x_scale.data_ptr<float>(),
                route_tokens_contig.data_ptr<int64_t>(),
                route_experts.data_ptr<int32_t>(),
                w1_contig.data_ptr<uint8_t>(),
                w3_contig.data_ptr<uint8_t>(),
                grid_ptr,
                gate.data_ptr<float>(),
                up.data_ptr<float>(),
                routes,
                dim,
                inter_dim,
                w1_blocks_per_row,
                x_groups);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "gguf_moe_w13", [&] {
            gguf_moe_w13_kernel<scalar_t><<<w13_grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                x_contig.data_ptr<scalar_t>(),
                route_tokens_contig.data_ptr<int64_t>(),
                route_experts.data_ptr<int32_t>(),
                w1_contig.data_ptr<uint8_t>(),
                w3_contig.data_ptr<uint8_t>(),
                grid_ptr,
                gate.data_ptr<float>(),
                up.data_ptr<float>(),
                routes,
                dim,
                inter_dim,
                w1_blocks_per_row,
                w1_block_bytes,
                static_cast<int>(w1_type_id),
                w3_blocks_per_row,
                w3_block_bytes,
                static_cast<int>(w3_type_id));
        });
    }
    if (emit_profile) {
        check_cuda(cudaEventRecord(ev_w13, at::cuda::getCurrentCUDAStream()), "record gguf prefill profile w13 event");
    }

    const dim3 w2_grid(ceil_div(dim, kGGUFQuantTileN), routes);
    const bool use_q2k_w2_dp4a =
        w2_type_id == 1 &&
        w2_block_bytes == 84 &&
        env_enabled_explicit("DEEPSEEK_GGUF_Q2K_W2_DP4A");
    if (use_q2k_w2_dp4a) {
        const int hidden_groups = ceil_div(inter_dim, 16);
        auto hidden_q = torch::empty({routes, inter_dim}, x.options().dtype(torch::kInt8));
        auto hidden_scale = torch::empty({routes, hidden_groups}, x.options().dtype(torch::kFloat32));
        const dim3 quant_grid(routes, hidden_groups);
        if (env_enabled_explicit("DEEPSEEK_GGUF_Q2K_FUSED_SWIGLU_QUANT")) {
            gguf_route_swiglu_quantize_hidden_16_kernel<<<quant_grid, 16, 0, at::cuda::getCurrentCUDAStream()>>>(
                gate.data_ptr<float>(),
                up.data_ptr<float>(),
                route_weights_contig.data_ptr<float>(),
                hidden_q.data_ptr<int8_t>(),
                hidden_scale.data_ptr<float>(),
                routes,
                inter_dim,
                static_cast<float>(swiglu_limit));
        } else {
            auto hidden = torch::empty({routes, inter_dim}, x.options().dtype(torch::kFloat32));
            const int hidden_threads = 256;
            const int hidden_blocks = static_cast<int>((static_cast<int64_t>(routes) * inter_dim + hidden_threads - 1) / hidden_threads);
            route_swiglu_cast_kernel<float><<<hidden_blocks, hidden_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                gate.data_ptr<float>(),
                up.data_ptr<float>(),
                route_weights_contig.data_ptr<float>(),
                hidden.data_ptr<float>(),
                routes,
                inter_dim,
                static_cast<float>(swiglu_limit));
            gguf_q8_1_quantize_hidden_16_kernel<<<quant_grid, 16, 0, at::cuda::getCurrentCUDAStream()>>>(
                hidden.data_ptr<float>(),
                hidden_q.data_ptr<int8_t>(),
                hidden_scale.data_ptr<float>(),
                routes,
                inter_dim);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (emit_profile) {
            check_cuda(cudaEventRecord(ev_hidden, at::cuda::getCurrentCUDAStream()), "record gguf prefill profile hidden event");
        }
        if (env_enabled_explicit("DEEPSEEK_GGUF_Q2K_W2_EXPERT_TILE")) {
            auto counts_i32 = seg_i32.slice(0, 1, n_experts + 1) - seg_i32.slice(0, 0, n_experts);
            const int max_count = counts_i32.max().item<int>();
            const dim3 expert_tile_grid(ceil_div(dim, kGGUFQuantTileN * 8), n_experts, ceil_div(max_count, 8));
            gguf_moe_w2_scatter_q2k_dp4a_expert_tile_kernel<<<expert_tile_grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                hidden_q.data_ptr<int8_t>(),
                hidden_scale.data_ptr<float>(),
                route_tokens_contig.data_ptr<int64_t>(),
                seg_i32.data_ptr<int32_t>(),
                w2_contig.data_ptr<uint8_t>(),
                y.data_ptr<float>(),
                n_experts,
                dim,
                inter_dim,
                w2_blocks_per_row,
                hidden_groups);
        } else if (env_enabled_explicit("DEEPSEEK_GGUF_Q2K_W2_SUBWARP")) {
            const dim3 subwarp_grid(ceil_div(dim, kGGUFQuantTileN * 8), routes);
            gguf_moe_w2_scatter_q2k_dp4a_subwarp_kernel<<<subwarp_grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                hidden_q.data_ptr<int8_t>(),
                hidden_scale.data_ptr<float>(),
                route_tokens_contig.data_ptr<int64_t>(),
                route_experts.data_ptr<int32_t>(),
                w2_contig.data_ptr<uint8_t>(),
                y.data_ptr<float>(),
                routes,
                dim,
                inter_dim,
                w2_blocks_per_row,
                hidden_groups);
        } else {
            gguf_moe_w2_scatter_q2k_dp4a_kernel<<<w2_grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                hidden_q.data_ptr<int8_t>(),
                hidden_scale.data_ptr<float>(),
                route_tokens_contig.data_ptr<int64_t>(),
                route_experts.data_ptr<int32_t>(),
                w2_contig.data_ptr<uint8_t>(),
                y.data_ptr<float>(),
                routes,
                dim,
                inter_dim,
                w2_blocks_per_row,
                hidden_groups);
        }
    } else {
        auto hidden = torch::empty({routes, inter_dim}, x.options().dtype(torch::kFloat32));
        const int hidden_threads = 256;
        const int hidden_blocks = static_cast<int>((static_cast<int64_t>(routes) * inter_dim + hidden_threads - 1) / hidden_threads);
        route_swiglu_cast_kernel<float><<<hidden_blocks, hidden_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            gate.data_ptr<float>(),
            up.data_ptr<float>(),
            route_weights_contig.data_ptr<float>(),
            hidden.data_ptr<float>(),
            routes,
            inter_dim,
            static_cast<float>(swiglu_limit));
        if (emit_profile) {
            check_cuda(cudaEventRecord(ev_hidden, at::cuda::getCurrentCUDAStream()), "record gguf prefill profile hidden event");
        }
        gguf_moe_w2_scatter_kernel<<<w2_grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            hidden.data_ptr<float>(),
            route_tokens_contig.data_ptr<int64_t>(),
            route_experts.data_ptr<int32_t>(),
            w2_contig.data_ptr<uint8_t>(),
            grid_ptr,
            y.data_ptr<float>(),
            routes,
            dim,
            inter_dim,
            w2_blocks_per_row,
            w2_block_bytes,
            static_cast<int>(w2_type_id));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (emit_profile) {
        check_cuda(cudaEventRecord(ev_w2, at::cuda::getCurrentCUDAStream()), "record gguf prefill profile w2 event");
        check_cuda(cudaEventSynchronize(ev_w2), "synchronize gguf prefill profile w2 event");
        float w13_ms = 0.0f;
        float hidden_ms = 0.0f;
        float w2_ms = 0.0f;
        float total_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&w13_ms, ev_start, ev_w13), "measure gguf prefill profile w13 time");
        check_cuda(cudaEventElapsedTime(&hidden_ms, ev_w13, ev_hidden), "measure gguf prefill profile hidden time");
        check_cuda(cudaEventElapsedTime(&w2_ms, ev_hidden, ev_w2), "measure gguf prefill profile w2 time");
        check_cuda(cudaEventElapsedTime(&total_ms, ev_start, ev_w2), "measure gguf prefill profile total time");
        std::printf(
            "gguf_prefill_kernel_profile call=%d tokens=%d routes=%d experts=%d dim=%d inter=%d w13_dp4a=%d w2_dp4a=%d w13=%.4fms hidden=%.4fms w2=%.4fms total=%.4fms\n",
            profile_index,
            tokens,
            routes,
            n_experts,
            dim,
            inter_dim,
            static_cast<int>(use_iq2_xxs_w13_dp4a),
            static_cast<int>(use_q2k_w2_dp4a),
            w13_ms,
            hidden_ms,
            w2_ms,
            total_ms);
        std::fflush(stdout);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_w13);
        cudaEventDestroy(ev_hidden);
        cudaEventDestroy(ev_w2);
    }
    return y;
}

torch::Tensor gguf_moe_single_token_iq2_q2k_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& route_slots,
    const torch::Tensor& route_weights,
    const torch::Tensor& w1_blocks,
    const torch::Tensor& w3_blocks,
    const torch::Tensor& w2_blocks,
    const torch::Tensor& signed_grid,
    double swiglu_limit) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto x_contig = x.contiguous();
    auto route_slots_contig = route_slots.contiguous();
    auto route_weights_contig = route_weights.contiguous();
    auto w1_contig = w1_blocks.contiguous();
    auto w3_contig = w3_blocks.contiguous();
    auto w2_contig = w2_blocks.contiguous();
    auto grid_contig = signed_grid.contiguous();

    const int routes = static_cast<int>(route_slots_contig.size(0));
    const int n_experts = static_cast<int>(w1_contig.size(0));
    const int dim = static_cast<int>(x_contig.size(1));
    const int inter_dim = static_cast<int>(w1_contig.size(1));
    auto y = torch::zeros({1, dim}, x.options().dtype(torch::kFloat32));
    if (routes == 0 || n_experts <= 0) {
        return y;
    }

    const int w1_blocks_per_row = static_cast<int>(w1_contig.size(2));
    const int w2_blocks_per_row = static_cast<int>(w2_contig.size(2));
    const int x_groups = ceil_div(dim, 32);
    const int hidden_groups = ceil_div(inter_dim, 16);

    auto x_q = torch::empty({1, dim}, x.options().dtype(torch::kInt8));
    auto x_scale = torch::empty({1, x_groups}, x.options().dtype(torch::kFloat32));
    const dim3 quant_x_grid(1, x_groups);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "gguf_single_q8_1_quantize_x_32", [&] {
        gguf_q8_1_quantize_x_32_kernel<scalar_t><<<quant_x_grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
            x_contig.data_ptr<scalar_t>(),
            x_q.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(),
            1,
            dim);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto gate = torch::empty({routes, inter_dim}, x.options().dtype(torch::kFloat32));
    auto up = torch::empty({routes, inter_dim}, x.options().dtype(torch::kFloat32));
    const dim3 w13_grid(ceil_div(inter_dim, kGGUFQuantTileN), routes);
    const dim3 block(256);
    gguf_moe_single_w13_iq2_xxs_dp4a_kernel<<<w13_grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x_q.data_ptr<int8_t>(),
        x_scale.data_ptr<float>(),
        route_slots_contig.data_ptr<int64_t>(),
        w1_contig.data_ptr<uint8_t>(),
        w3_contig.data_ptr<uint8_t>(),
        grid_contig.data_ptr<int8_t>(),
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        routes,
        n_experts,
        dim,
        inter_dim,
        w1_blocks_per_row,
        x_groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto hidden_q = torch::empty({routes, inter_dim}, x.options().dtype(torch::kInt8));
    auto hidden_scale = torch::empty({routes, hidden_groups}, x.options().dtype(torch::kFloat32));
    const dim3 quant_h_grid(routes, hidden_groups);
    if (env_enabled_explicit("DEEPSEEK_GGUF_Q2K_FUSED_SWIGLU_QUANT")) {
        gguf_route_swiglu_quantize_hidden_16_kernel<<<quant_h_grid, 16, 0, at::cuda::getCurrentCUDAStream()>>>(
            gate.data_ptr<float>(),
            up.data_ptr<float>(),
            route_weights_contig.data_ptr<float>(),
            hidden_q.data_ptr<int8_t>(),
            hidden_scale.data_ptr<float>(),
            routes,
            inter_dim,
            static_cast<float>(swiglu_limit));
    } else {
        auto hidden = torch::empty({routes, inter_dim}, x.options().dtype(torch::kFloat32));
        const int hidden_threads = 256;
        const int hidden_blocks = static_cast<int>((static_cast<int64_t>(routes) * inter_dim + hidden_threads - 1) / hidden_threads);
        route_swiglu_cast_kernel<float><<<hidden_blocks, hidden_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            gate.data_ptr<float>(),
            up.data_ptr<float>(),
            route_weights_contig.data_ptr<float>(),
            hidden.data_ptr<float>(),
            routes,
            inter_dim,
            static_cast<float>(swiglu_limit));
        gguf_q8_1_quantize_hidden_16_kernel<<<quant_h_grid, 16, 0, at::cuda::getCurrentCUDAStream()>>>(
            hidden.data_ptr<float>(),
            hidden_q.data_ptr<int8_t>(),
            hidden_scale.data_ptr<float>(),
            routes,
            inter_dim);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const dim3 w2_grid(ceil_div(dim, kGGUFQuantTileN), routes);
    gguf_moe_single_w2_q2k_dp4a_kernel<<<w2_grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        hidden_q.data_ptr<int8_t>(),
        hidden_scale.data_ptr<float>(),
        route_slots_contig.data_ptr<int64_t>(),
        w2_contig.data_ptr<uint8_t>(),
        y.data_ptr<float>(),
        routes,
        n_experts,
        dim,
        inter_dim,
        w2_blocks_per_row,
        hidden_groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor gguf_moe_single_token_iq1m_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& route_slots,
    const torch::Tensor& route_weights,
    const torch::Tensor& w1_blocks,
    const torch::Tensor& w3_blocks,
    const torch::Tensor& w2_blocks,
    const torch::Tensor& iq1_grid,
    double swiglu_limit) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto x_contig = x.contiguous();
    auto route_slots_contig = route_slots.contiguous();
    auto route_weights_contig = route_weights.contiguous();
    auto w1_contig = w1_blocks.contiguous();
    auto w3_contig = w3_blocks.contiguous();
    auto w2_contig = w2_blocks.contiguous();
    auto grid_contig = iq1_grid.contiguous();

    const int routes = static_cast<int>(route_slots_contig.size(0));
    const int n_experts = static_cast<int>(w1_contig.size(0));
    const int dim = static_cast<int>(x_contig.size(1));
    const int inter_dim = static_cast<int>(w1_contig.size(1));
    auto y = torch::zeros({1, dim}, x.options().dtype(torch::kFloat32));
    if (routes == 0 || n_experts <= 0) {
        return y;
    }

    const int w1_blocks_per_row = static_cast<int>(w1_contig.size(2));
    const int w3_blocks_per_row = static_cast<int>(w3_contig.size(2));
    const int w2_blocks_per_row = static_cast<int>(w2_contig.size(2));
    const int w_block_bytes = 56;

    auto route_tokens = torch::zeros({routes}, route_slots_contig.options());
    auto route_experts = route_slots_contig.to(torch::kInt32);
    auto gate = torch::empty({routes, inter_dim}, x.options().dtype(torch::kFloat32));
    auto up = torch::empty({routes, inter_dim}, x.options().dtype(torch::kFloat32));
    const dim3 block(256);
    const dim3 w13_grid(ceil_div(inter_dim, kGGUFQuantTileN), routes);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "gguf_moe_single_token_iq1m_w13", [&] {
        gguf_moe_w13_kernel<scalar_t><<<w13_grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            x_contig.data_ptr<scalar_t>(),
            route_tokens.data_ptr<int64_t>(),
            route_experts.data_ptr<int32_t>(),
            w1_contig.data_ptr<uint8_t>(),
            w3_contig.data_ptr<uint8_t>(),
            grid_contig.data_ptr<int8_t>(),
            gate.data_ptr<float>(),
            up.data_ptr<float>(),
            routes,
            dim,
            inter_dim,
            w1_blocks_per_row,
            w_block_bytes,
            2,
            w3_blocks_per_row,
            w_block_bytes,
            2);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto hidden = torch::empty({routes, inter_dim}, x.options().dtype(torch::kFloat32));
    const int hidden_threads = 256;
    const int hidden_blocks = static_cast<int>((static_cast<int64_t>(routes) * inter_dim + hidden_threads - 1) / hidden_threads);
    route_swiglu_cast_kernel<float><<<hidden_blocks, hidden_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        route_weights_contig.data_ptr<float>(),
        hidden.data_ptr<float>(),
        routes,
        inter_dim,
        static_cast<float>(swiglu_limit));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const dim3 w2_grid(ceil_div(dim, kGGUFQuantTileN), routes);
    gguf_moe_w2_scatter_kernel<<<w2_grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        hidden.data_ptr<float>(),
        route_tokens.data_ptr<int64_t>(),
        route_experts.data_ptr<int32_t>(),
        w2_contig.data_ptr<uint8_t>(),
        grid_contig.data_ptr<int8_t>(),
        y.data_ptr<float>(),
        routes,
        dim,
        inter_dim,
        w2_blocks_per_row,
        w_block_bytes,
        2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

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
