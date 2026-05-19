#include "cuda_ops.hpp"

#include <cuda_runtime.h>

#include <cmath>

namespace dsv4 {
namespace {

__global__ void silu_mul_kernel(const float* gate, const float* up, float* y, int cols) {
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float g = gate[c];
        const float s = g / (1.0f + expf(-g));
        y[c] = s * up[c];
    }
}

__global__ void silu_mul_clamped_kernel(const float* gate, const float* up, float* y, int cols, float limit) {
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float g = fminf(gate[c], limit);
        const float u = fminf(fmaxf(up[c], -limit), limit);
        const float s = g / (1.0f + expf(-g));
        y[c] = s * u;
    }
}

__global__ void vector_add_kernel(const float* a, const float* b, float* y, int cols) {
    for (int c = threadIdx.x; c < cols; c += blockDim.x) y[c] = a[c] + b[c];
}

__global__ void vector_accum_kernel(const float* x, float* y, int cols, float scale) {
    for (int c = threadIdx.x; c < cols; c += blockDim.x) y[c] += x[c] * scale;
}

__global__ void repeat_vector_kernel(const float* x, float* y, int cols, int repeats) {
    const int total = cols * repeats;
    for (int i = threadIdx.x; i < total; i += blockDim.x) y[i] = x[i % cols];
}

__global__ void head_rmsnorm_rope_kernel(
    float* x,
    int head_dim,
    int rope_dim,
    int position,
    float theta,
    bool inverse,
    float eps) {
    const int head = blockIdx.x;
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = x[head * head_dim + i];
        sum_sq += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float norm = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) x[head * head_dim + i] *= norm;
    __syncthreads();
    const int rope_start = head_dim - rope_dim;
    for (int pair = threadIdx.x * 2; pair < rope_dim; pair += blockDim.x * 2) {
        const int offset = rope_start + pair;
        const float angle = static_cast<float>(position) / powf(theta, static_cast<float>(pair) / static_cast<float>(rope_dim));
        const float c = cosf(angle);
        const float s = inverse ? -sinf(angle) : sinf(angle);
        const size_t base = static_cast<size_t>(head) * head_dim + offset;
        const float a = x[base];
        const float b = x[base + 1];
        x[base] = a * c - b * s;
        x[base + 1] = a * s + b * c;
    }
}

__global__ void rope_kernel(
    float* x,
    int head_dim,
    int rope_dim,
    int position,
    float theta,
    bool inverse) {
    const int head = blockIdx.x;
    const int rope_start = head_dim - rope_dim;
    for (int pair = threadIdx.x * 2; pair < rope_dim; pair += blockDim.x * 2) {
        const int offset = rope_start + pair;
        const float angle = static_cast<float>(position) / powf(theta, static_cast<float>(pair) / static_cast<float>(rope_dim));
        const float c = cosf(angle);
        const float s = inverse ? -sinf(angle) : sinf(angle);
        const size_t base = static_cast<size_t>(head) * head_dim + offset;
        const float a = x[base];
        const float b = x[base + 1];
        x[base] = a * c - b * s;
        x[base + 1] = a * s + b * c;
    }
}

__global__ void single_token_sparse_attention_kernel(
    const float* q,
    const float* kv,
    const float* attn_sink,
    float* y,
    int head_dim,
    float scale) {
    const int head = blockIdx.x;
    float dot = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) dot += q[head * head_dim + i] * kv[i];
    __shared__ float partial[256];
    partial[threadIdx.x] = dot;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float token_logit = partial[0] * scale;
    const float sink_logit = attn_sink == nullptr ? -INFINITY : attn_sink[head];
    const float m = fmaxf(token_logit, sink_logit);
    const float token_weight = expf(token_logit - m) / (expf(token_logit - m) + expf(sink_logit - m));
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) y[head * head_dim + i] = kv[i] * token_weight;
}

__global__ void cached_single_token_attention_kernel(
    const float* q,
    const float* kv_cache,
    const float* attn_sink,
    float* y,
    int head_dim,
    int cache_len,
    float scale) {
    const int head = blockIdx.x;
    __shared__ float logits[256];
    __shared__ float denom;
    __shared__ float max_logit;
    float local_max = attn_sink == nullptr ? -INFINITY : attn_sink[head];
    for (int t = threadIdx.x; t < cache_len; t += blockDim.x) {
        const float* kv = kv_cache + static_cast<size_t>(t) * head_dim;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i) dot += q[head * head_dim + i] * kv[i];
        const float logit = dot * scale;
        logits[t] = logit;
        local_max = fmaxf(local_max, logit);
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_logit = partial[0];
    __syncthreads();

    float local_denom = 0.0f;
    for (int t = threadIdx.x; t < cache_len; t += blockDim.x) local_denom += expf(logits[t] - max_logit);
    partial[threadIdx.x] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + (attn_sink == nullptr ? 0.0f : expf(attn_sink[head] - max_logit));
    __syncthreads();

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float out = 0.0f;
        for (int t = 0; t < cache_len; ++t) {
            const float w = expf(logits[t] - max_logit) / denom;
            out += w * kv_cache[static_cast<size_t>(t) * head_dim + i];
        }
        y[head * head_dim + i] = out;
    }
}

__device__ float bf16_to_float_device(uint16_t bits) {
    uint32_t value = static_cast<uint32_t>(bits) << 16;
    return __uint_as_float(value);
}

__global__ void indexed_cached_single_token_attention_kernel(
    const float* q,
    const float* kv_cache,
    const int* indices,
    const float* attn_sink,
    float* y,
    int head_dim,
    int index_count,
    float scale) {
    const int head = blockIdx.x;
    __shared__ float logits[640];
    __shared__ float denom;
    __shared__ float max_logit;
    float local_max = attn_sink == nullptr ? -INFINITY : attn_sink[head];
    for (int t = threadIdx.x; t < index_count; t += blockDim.x) {
        const int idx = indices[t];
        float logit = -INFINITY;
        if (idx >= 0) {
            const float* kv = kv_cache + static_cast<size_t>(idx) * head_dim;
            float dot = 0.0f;
            for (int i = 0; i < head_dim; ++i) dot += q[head * head_dim + i] * kv[i];
            logit = dot * scale;
            local_max = fmaxf(local_max, logit);
        }
        logits[t] = logit;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_logit = partial[0];
    __syncthreads();

    float local_denom = 0.0f;
    for (int t = threadIdx.x; t < index_count; t += blockDim.x) {
        if (indices[t] >= 0) local_denom += expf(logits[t] - max_logit);
    }
    partial[threadIdx.x] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + (attn_sink == nullptr ? 0.0f : expf(attn_sink[head] - max_logit));
    __syncthreads();

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float out = 0.0f;
        for (int t = 0; t < index_count; ++t) {
            const int idx = indices[t];
            if (idx >= 0) {
                const float w = expf(logits[t] - max_logit) / denom;
                out += w * kv_cache[static_cast<size_t>(idx) * head_dim + i];
            }
        }
        y[head * head_dim + i] = out;
    }
}

__global__ void indexer_select_topk_kernel(
    const float* index_q,
    const float* index_kv,
    const uint16_t* weight_proj,
    const float* x,
    int* out_indices,
    int compressed_count,
    int keep,
    int heads,
    int head_dim,
    int dim,
    int offset) {
    __shared__ float scores[640];
    if (threadIdx.x < compressed_count) {
        const int c = threadIdx.x;
        float score = 0.0f;
        const float scale = rsqrtf(static_cast<float>(heads * head_dim));
        for (int h = 0; h < heads; ++h) {
            float weight = 0.0f;
            for (int d = 0; d < dim; ++d) weight += bf16_to_float_device(weight_proj[static_cast<size_t>(h) * dim + d]) * x[d];
            weight *= scale;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += index_q[static_cast<size_t>(h) * head_dim + d] * index_kv[static_cast<size_t>(c) * head_dim + d];
            score += fmaxf(0.0f, dot) * weight;
        }
        scores[c] = score;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        bool selected[640];
        for (int i = 0; i < compressed_count; ++i) selected[i] = false;
        for (int k = 0; k < keep; ++k) {
            int best = 0;
            float best_score = -INFINITY;
            for (int i = 0; i < compressed_count; ++i) {
                if (!selected[i] && scores[i] > best_score) {
                    best_score = scores[i];
                    best = i;
                }
            }
            selected[best] = true;
            out_indices[k] = offset + best;
        }
    }
}

}  // namespace

bool cuda_runtime_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

bool silu_mul_cuda(const float* d_gate, const float* d_up, float* d_y, int cols, void* stream) {
    if (d_gate == nullptr || d_up == nullptr || d_y == nullptr || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    silu_mul_kernel<<<1, 256, 0, cuda_stream>>>(d_gate, d_up, d_y, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool silu_mul_clamped_cuda(const float* d_gate, const float* d_up, float* d_y, int cols, float limit, void* stream) {
    if (d_gate == nullptr || d_up == nullptr || d_y == nullptr || cols <= 0 || limit <= 0.0f) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    silu_mul_clamped_kernel<<<1, 256, 0, cuda_stream>>>(d_gate, d_up, d_y, cols, limit);
    return cudaGetLastError() == cudaSuccess;
}

bool vector_add_cuda(const float* d_a, const float* d_b, float* d_y, int cols, void* stream) {
    if (d_a == nullptr || d_b == nullptr || d_y == nullptr || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    vector_add_kernel<<<1, 256, 0, cuda_stream>>>(d_a, d_b, d_y, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool vector_accum_cuda(const float* d_x, float* d_y, int cols, float scale, void* stream) {
    if (d_x == nullptr || d_y == nullptr || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    vector_accum_kernel<<<1, 256, 0, cuda_stream>>>(d_x, d_y, cols, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool repeat_vector_cuda(const float* d_x, float* d_y, int cols, int repeats, void* stream) {
    if (d_x == nullptr || d_y == nullptr || cols <= 0 || repeats <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    repeat_vector_kernel<<<1, 256, 0, cuda_stream>>>(d_x, d_y, cols, repeats);
    return cudaGetLastError() == cudaSuccess;
}

bool single_token_sparse_attention_cuda(
    const float* d_q,
    const float* d_kv,
    const float* d_attn_sink,
    float* d_y,
    int heads,
    int head_dim,
    float scale,
    void* stream) {
    if (d_q == nullptr || d_kv == nullptr || d_y == nullptr || heads <= 0 || head_dim <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    single_token_sparse_attention_kernel<<<heads, 256, 0, cuda_stream>>>(d_q, d_kv, d_attn_sink, d_y, head_dim, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool cached_single_token_attention_cuda(
    const float* d_q,
    const float* d_kv_cache,
    const float* d_attn_sink,
    float* d_y,
    int heads,
    int head_dim,
    int cache_len,
    float scale,
    void* stream) {
    if (d_q == nullptr || d_kv_cache == nullptr || d_y == nullptr || heads <= 0 || head_dim <= 0 || cache_len <= 0 || cache_len > 256) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cached_single_token_attention_kernel<<<heads, 256, 0, cuda_stream>>>(d_q, d_kv_cache, d_attn_sink, d_y, head_dim, cache_len, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool indexed_cached_single_token_attention_cuda(
    const float* d_q,
    const float* d_kv_cache,
    const int* d_indices,
    const float* d_attn_sink,
    float* d_y,
    int heads,
    int head_dim,
    int index_count,
    float scale,
    void* stream) {
    if (d_q == nullptr || d_kv_cache == nullptr || d_indices == nullptr || d_y == nullptr || heads <= 0 || head_dim <= 0 || index_count <= 0 || index_count > 640) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    indexed_cached_single_token_attention_kernel<<<heads, 256, 0, cuda_stream>>>(d_q, d_kv_cache, d_indices, d_attn_sink, d_y, head_dim, index_count, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool indexer_select_topk_cuda(
    const float* d_index_q,
    const float* d_index_kv,
    const uint16_t* d_weight_proj_bf16,
    const float* d_x,
    int* d_out_indices,
    int compressed_count,
    int keep,
    int heads,
    int head_dim,
    int dim,
    int offset,
    void* stream) {
    if (d_index_q == nullptr || d_index_kv == nullptr || d_weight_proj_bf16 == nullptr || d_x == nullptr || d_out_indices == nullptr) return false;
    if (compressed_count <= 0 || compressed_count > 640 || keep <= 0 || keep > compressed_count || heads <= 0 || head_dim <= 0 || dim <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    indexer_select_topk_kernel<<<1, 640, 0, cuda_stream>>>(d_index_q, d_index_kv, d_weight_proj_bf16, d_x, d_out_indices, compressed_count, keep, heads, head_dim, dim, offset);
    return cudaGetLastError() == cudaSuccess;
}

bool head_rmsnorm_rope_cuda(
    float* d_x,
    int heads,
    int head_dim,
    int rope_dim,
    int position,
    float theta,
    bool inverse,
    float eps,
    void* stream) {
    if (d_x == nullptr || heads <= 0 || head_dim <= 0 || rope_dim < 0 || rope_dim > head_dim || (rope_dim % 2) != 0 || theta <= 0.0f) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (eps > 0.0f) {
        head_rmsnorm_rope_kernel<<<heads, 256, 0, cuda_stream>>>(d_x, head_dim, rope_dim, position, theta, inverse, eps);
    } else {
        rope_kernel<<<heads, 256, 0, cuda_stream>>>(d_x, head_dim, rope_dim, position, theta, inverse);
    }
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace dsv4
