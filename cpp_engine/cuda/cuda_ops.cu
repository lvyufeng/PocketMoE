#include "cuda_ops.hpp"

#include <cuda_runtime.h>

#include <cmath>

namespace dsv4 {
namespace {

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void hc_pre_float_kernel(
    const float* h4,
    const float* fn,
    const float* scale,
    const float* base,
    float* x,
    float* post_out,
    float* comb_out,
    int dim) {
    constexpr float eps = 1e-6f;
    const int tid = threadIdx.x;
    __shared__ float mixes[24];
    __shared__ float pre[4];
    __shared__ float post[4];
    __shared__ float comb[16];
    __shared__ float partial[256];
    float sum_sq = 0.0f;
    for (int i = tid; i < 4 * dim; i += blockDim.x) {
        const float v = h4[i];
        sum_sq += v * v;
    }
    partial[tid] = sum_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float rsqrt = rsqrtf(partial[0] / static_cast<float>(4 * dim) + eps);
    for (int r = tid; r < 24; r += blockDim.x) {
        float dot = 0.0f;
        const float* row = fn + static_cast<size_t>(r) * 4 * dim;
        for (int i = 0; i < 4 * dim; ++i) dot += row[i] * h4[i];
        mixes[r] = dot * rsqrt;
    }
    __syncthreads();
    if (tid < 4) {
        pre[tid] = sigmoidf_fast(mixes[tid] * scale[0] + base[tid]) + eps;
        post[tid] = 2.0f * sigmoidf_fast(mixes[4 + tid] * scale[1] + base[4 + tid]);
    }
    if (tid < 16) {
        float vals[4];
        const int row = tid / 4;
        const int col = tid - row * 4;
        const float* cm = mixes + 8 + row * 4;
        float max_v = cm[0] * scale[2] + base[8 + row * 4];
        for (int c = 1; c < 4; ++c) max_v = fmaxf(max_v, cm[c] * scale[2] + base[8 + row * 4 + c]);
        float sum = 0.0f;
        for (int c = 0; c < 4; ++c) {
            vals[c] = expf(cm[c] * scale[2] + base[8 + row * 4 + c] - max_v);
            sum += vals[c];
        }
        comb[tid] = vals[col] / sum + eps;
    }
    __syncthreads();
    if (tid < 4) {
        float col_sum = 0.0f;
        for (int r = 0; r < 4; ++r) col_sum += comb[r * 4 + tid];
        for (int r = 0; r < 4; ++r) comb[r * 4 + tid] /= col_sum + eps;
    }
    __syncthreads();
    for (int iter = 0; iter < 19; ++iter) {
        if (tid < 4) {
            float row_sum = 0.0f;
            for (int c = 0; c < 4; ++c) row_sum += comb[tid * 4 + c];
            for (int c = 0; c < 4; ++c) comb[tid * 4 + c] /= row_sum + eps;
        }
        __syncthreads();
        if (tid < 4) {
            float col_sum = 0.0f;
            for (int r = 0; r < 4; ++r) col_sum += comb[r * 4 + tid];
            for (int r = 0; r < 4; ++r) comb[r * 4 + tid] /= col_sum + eps;
        }
        __syncthreads();
    }
    if (tid < 4) post_out[tid] = post[tid];
    if (tid < 16) comb_out[tid] = comb[tid];
    for (int d = tid; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int h = 0; h < 4; ++h) acc += pre[h] * h4[static_cast<size_t>(h) * dim + d];
        x[d] = acc;
    }
}

__global__ void hc_post_float_kernel(const float* x, const float* residual, const float* post, const float* comb, float* y, int dim) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    for (int d = tid; d < dim; d += blockDim.x) {
        float acc = post[h] * x[d];
        for (int i = 0; i < 4; ++i) acc += comb[i * 4 + h] * residual[static_cast<size_t>(i) * dim + d];
        y[static_cast<size_t>(h) * dim + d] = acc;
    }
}

__global__ void silu_mul_kernel(const float* gate, const float* up, float* y, int cols) {
    const int row = blockIdx.x;
    const float* row_gate = gate + static_cast<size_t>(row) * cols;
    const float* row_up = up + static_cast<size_t>(row) * cols;
    float* row_y = y + static_cast<size_t>(row) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float g = row_gate[c];
        const float s = g / (1.0f + expf(-g));
        row_y[c] = s * row_up[c];
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
    const int row = blockIdx.x;
    const float* row_x = x + static_cast<size_t>(row) * cols;
    float* row_y = y + static_cast<size_t>(row) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) row_y[c] += row_x[c] * scale;
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

__global__ void head_rmsnorm_rope_freqs_kernel(
    float* x,
    const float* inv_freqs,
    int head_dim,
    int rope_dim,
    int position,
    bool inverse,
    float eps) {
    const int head = blockIdx.x;
    if (eps > 0.0f) {
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
    }
    const int rope_start = head_dim - rope_dim;
    for (int pair = threadIdx.x * 2; pair < rope_dim; pair += blockDim.x * 2) {
        const int offset = rope_start + pair;
        const float freq = inv_freqs[pair >> 1];
        const float angle = static_cast<float>(position) * freq;
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
    __shared__ float denom;
    __shared__ float max_logit;
    __shared__ float partial[256];
    float local_max = attn_sink == nullptr ? -INFINITY : attn_sink[head];
    for (int t = threadIdx.x; t < cache_len; t += blockDim.x) {
        const float* kv = kv_cache + static_cast<size_t>(t) * head_dim;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i) dot += q[head * head_dim + i] * kv[i];
        local_max = fmaxf(local_max, dot * scale);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_logit = partial[0];
    __syncthreads();

    float local_denom = 0.0f;
    for (int t = threadIdx.x; t < cache_len; t += blockDim.x) {
        const float* kv = kv_cache + static_cast<size_t>(t) * head_dim;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i) dot += q[head * head_dim + i] * kv[i];
        local_denom += expf(dot * scale - max_logit);
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
        for (int t = 0; t < cache_len; ++t) {
            const float* kv = kv_cache + static_cast<size_t>(t) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += q[head * head_dim + d] * kv[d];
            const float w = expf(dot * scale - max_logit) / denom;
            out += w * kv[i];
        }
        y[head * head_dim + i] = out;
    }
}

__device__ float bf16_to_float_device(uint16_t bits) {
    uint32_t value = static_cast<uint32_t>(bits) << 16;
    return __uint_as_float(value);
}

__device__ float round_pow2_device(float x) {
    return exp2f(roundf(log2f(x)));
}

__device__ float fp8_e4m3_quant_dequant(float v, float scale) {
    float q = fminf(448.0f, fmaxf(-448.0f, nearbyintf(v / scale)));
    return q * scale;
}

__global__ void fp8_act_quant_dequant_kernel(float* x, int cols, int block_size) {
    const int block = blockIdx.x;
    const int start = block * block_size;
    float max_abs = 0.0f;
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) max_abs = fmaxf(max_abs, fabsf(x[start + i]));
    __shared__ float partial[256];
    partial[threadIdx.x] = max_abs;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    const float scale = round_pow2_device(fmaxf(partial[0], 1e-4f) / 448.0f);
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) x[start + i] = fp8_e4m3_quant_dequant(x[start + i], scale);
}

__global__ void compressor_update_state_kernel(
    const float* kv,
    const float* score,
    const float* ape,
    float* kv_state,
    float* score_state,
    int write_slot,
    int state_cols) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= state_cols) return;
    kv_state[static_cast<size_t>(write_slot) * state_cols + c] = kv[c];
    score_state[static_cast<size_t>(write_slot) * state_cols + c] = score[c] + ape[c];
}

__global__ void compressor_pool_kernel(
    const float* kv_state,
    const float* score_state,
    float* out,
    int ratio,
    int head_dim,
    int state_cols,
    bool overlap) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    const int pool_slots = overlap ? ratio * 2 : ratio;
    float max_score = -INFINITY;
    for (int t = 0; t < pool_slots; ++t) {
        const int col = (overlap && t >= ratio) ? head_dim + d : d;
        max_score = fmaxf(max_score, score_state[static_cast<size_t>(t) * state_cols + col]);
    }
    float denom = 0.0f;
    for (int t = 0; t < pool_slots; ++t) {
        const int col = (overlap && t >= ratio) ? head_dim + d : d;
        denom += expf(score_state[static_cast<size_t>(t) * state_cols + col] - max_score);
    }
    float sum = 0.0f;
    for (int t = 0; t < pool_slots; ++t) {
        const int col = (overlap && t >= ratio) ? head_dim + d : d;
        const float w = expf(score_state[static_cast<size_t>(t) * state_cols + col] - max_score) / denom;
        sum += w * kv_state[static_cast<size_t>(t) * state_cols + col];
    }
    out[d] = sum;
}

__global__ void compressor_shift_overlap_state_kernel(float* kv_state, float* score_state, int ratio, int state_cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ratio * state_cols;
    if (idx >= total) return;
    kv_state[idx] = kv_state[total + idx];
    score_state[idx] = score_state[total + idx];
    kv_state[total + idx] = 0.0f;
    score_state[total + idx] = -INFINITY;
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
    __shared__ float denom;
    __shared__ float max_logit;
    __shared__ float partial[256];
    float local_max = attn_sink == nullptr ? -INFINITY : attn_sink[head];
    for (int t = threadIdx.x; t < index_count; t += blockDim.x) {
        const int idx = indices[t];
        if (idx >= 0) {
            const float* kv = kv_cache + static_cast<size_t>(idx) * head_dim;
            float dot = 0.0f;
            for (int i = 0; i < head_dim; ++i) dot += q[head * head_dim + i] * kv[i];
            local_max = fmaxf(local_max, dot * scale);
        }
    }
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
        const int idx = indices[t];
        if (idx >= 0) {
            const float* kv = kv_cache + static_cast<size_t>(idx) * head_dim;
            float dot = 0.0f;
            for (int i = 0; i < head_dim; ++i) dot += q[head * head_dim + i] * kv[i];
            local_denom += expf(dot * scale - max_logit);
        }
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
                const float* kv = kv_cache + static_cast<size_t>(idx) * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += q[head * head_dim + d] * kv[d];
                const float w = expf(dot * scale - max_logit) / denom;
                out += w * kv[i];
            }
        }
        y[head * head_dim + i] = out;
    }
}


__device__ __forceinline__ float fp4_fake_quant_value_device(float x) {
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

__device__ __forceinline__ float fp4_pow2_scale_device(float amax) {
    const float min_scale = 1.1754943508222875e-38f;
    const float raw = fmaxf(amax / 6.0f, min_scale);
    return exp2f(ceilf(log2f(raw) - 1.0e-6f));
}

__device__ void hadamard128_inplace_device(float* x, int tid) {
    for (int stride = 1; stride < 128; stride <<= 1) {
        if (tid < 128 && (tid & stride) == 0) {
            const float a = x[tid];
            const float b = x[tid + stride];
            x[tid] = a + b;
            x[tid + stride] = a - b;
        }
        __syncthreads();
    }
    if (tid < 128) x[tid] *= 0.08838834764831845f;
    __syncthreads();
}

__global__ void hadamard128_rows_kernel(const float* x, float* y, int rows) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float vec[128];
    if (row >= rows) return;
    if (tid < 128) vec[tid] = x[static_cast<size_t>(row) * 128 + tid];
    __syncthreads();
    hadamard128_inplace_device(vec, tid);
    if (tid < 128) y[static_cast<size_t>(row) * 128 + tid] = vec[tid];
}

__global__ void fp4_fake_quant128_rows_kernel(float* x, int rows) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float vec[128];
    __shared__ float scales[4];
    if (row >= rows) return;
    if (tid < 128) vec[tid] = x[static_cast<size_t>(row) * 128 + tid];
    __syncthreads();
    if (tid < 4) {
        float amax = 0.0f;
        const int base = tid * 32;
        #pragma unroll
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(vec[base + i]));
        scales[tid] = fp4_pow2_scale_device(amax);
    }
    __syncthreads();
    if (tid < 128) {
        const float scale = scales[tid >> 5];
        const float yv = fminf(fmaxf(vec[tid] / scale, -6.0f), 6.0f);
        vec[tid] = fp4_fake_quant_value_device(yv) * scale;
    }
    __syncthreads();
    if (tid < 128) x[static_cast<size_t>(row) * 128 + tid] = vec[tid];
}

__global__ void indexer_head_weights_kernel(
    const uint16_t* weight_proj,
    const float* x,
    float* head_weights,
    int heads,
    int head_dim,
    int dim) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    if (head >= heads) return;
    __shared__ float reduce[256];
    float local = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) local += bf16_to_float_device(weight_proj[static_cast<size_t>(head) * dim + d]) * x[d];
    reduce[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        const float scale = rsqrtf(static_cast<float>(heads * head_dim));
        head_weights[head] = reduce[0] * scale;
    }
}

__global__ void indexer_score_kernel(
    const float* index_q,
    const float* index_kv,
    const float* head_weights,
    float* scores,
    int compressed_count,
    int heads,
    int head_dim) {
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    if (c >= compressed_count) return;
    __shared__ float reduce[256];
    float total = 0.0f;
    for (int h = 0; h < heads; ++h) {
        float partial = 0.0f;
        if (tid < head_dim) {
            const size_t idx = static_cast<size_t>(h) * head_dim + tid;
            partial = index_q[idx] * index_kv[static_cast<size_t>(c) * head_dim + tid];
        }
        reduce[tid] = partial;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            const float dot = reduce[0];
            if (dot > 0.0f) total += dot * head_weights[h];
        }
        __syncthreads();
    }
    if (tid == 0) scores[c] = total;
}

__global__ void indexer_topk_kernel(
    float* scores,
    int* out_indices,
    int compressed_count,
    int keep,
    int offset) {
    const int tid = threadIdx.x;
    extern __shared__ char smem_raw[];
    float* best_vals = reinterpret_cast<float*>(smem_raw);
    int* best_idxs = reinterpret_cast<int*>(best_vals + blockDim.x);
    for (int k = 0; k < keep; ++k) {
        float local_best = -INFINITY;
        int local_idx = compressed_count;
        for (int i = tid; i < compressed_count; i += blockDim.x) {
            const float v = scores[i];
            if (v > local_best || (v == local_best && i < local_idx)) {
                local_best = v;
                local_idx = i;
            }
        }
        best_vals[tid] = local_best;
        best_idxs[tid] = local_idx;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
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
            const int idx = best_idxs[0] < 0 ? 0 : best_idxs[0];
            out_indices[k] = offset + idx;
            if (idx >= 0 && idx < compressed_count) scores[idx] = -INFINITY;
        }
        __syncthreads();
    }
}

__global__ void fp32_to_bf16_kernel(const float* x, uint16_t* y, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t bits = __float_as_uint(x[i]);
    const uint32_t lsb = (bits >> 16) & 1u;
    const uint32_t rounded = bits + 0x7FFFu + lsb;
    y[i] = static_cast<uint16_t>(rounded >> 16);
}

__global__ void bf16_to_fp32_kernel(const uint16_t* x, float* y, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    y[i] = __uint_as_float(static_cast<uint32_t>(x[i]) << 16);
}

}  // namespace

bool cuda_runtime_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

bool hc_pre_float_cuda(const float* d_h4, const float* d_fn, const float* d_scale, const float* d_base, float* d_x, float* d_post, float* d_comb, int dim, void* stream) {
    if (d_h4 == nullptr || d_fn == nullptr || d_scale == nullptr || d_base == nullptr || d_x == nullptr || d_post == nullptr || d_comb == nullptr || dim <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    hc_pre_float_kernel<<<1, 256, 0, cuda_stream>>>(d_h4, d_fn, d_scale, d_base, d_x, d_post, d_comb, dim);
    return cudaGetLastError() == cudaSuccess;
}

bool hc_post_float_cuda(const float* d_x, const float* d_residual_h4, const float* d_post, const float* d_comb, float* d_y_h4, int dim, void* stream) {
    if (d_x == nullptr || d_residual_h4 == nullptr || d_post == nullptr || d_comb == nullptr || d_y_h4 == nullptr || dim <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    hc_post_float_kernel<<<4, 256, 0, cuda_stream>>>(d_x, d_residual_h4, d_post, d_comb, d_y_h4, dim);
    return cudaGetLastError() == cudaSuccess;
}

bool silu_mul_cuda(const float* d_gate, const float* d_up, float* d_y, int cols, void* stream) {
    if (d_gate == nullptr || d_up == nullptr || d_y == nullptr || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    silu_mul_kernel<<<1, 256, 0, cuda_stream>>>(d_gate, d_up, d_y, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool silu_mul_rows_cuda(const float* d_gate, const float* d_up, float* d_y, int rows, int cols, void* stream) {
    if (d_gate == nullptr || d_up == nullptr || d_y == nullptr || rows <= 0 || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    silu_mul_kernel<<<rows, 256, 0, cuda_stream>>>(d_gate, d_up, d_y, cols);
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

bool vector_accum_rows_cuda(const float* d_x, float* d_y, int rows, int cols, float scale, void* stream) {
    if (d_x == nullptr || d_y == nullptr || rows <= 0 || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    vector_accum_kernel<<<rows, 256, 0, cuda_stream>>>(d_x, d_y, cols, scale);
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
    if (d_q == nullptr || d_kv_cache == nullptr || d_y == nullptr || heads <= 0 || head_dim <= 0 || cache_len <= 0) return false;
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
    if (d_q == nullptr || d_kv_cache == nullptr || d_indices == nullptr || d_y == nullptr || heads <= 0 || head_dim <= 0 || index_count <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    indexed_cached_single_token_attention_kernel<<<heads, 256, 0, cuda_stream>>>(d_q, d_kv_cache, d_indices, d_attn_sink, d_y, head_dim, index_count, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool indexer_select_topk_cuda(
    const float* d_index_q,
    const float* d_index_kv,
    const uint16_t* d_weight_proj_bf16,
    const float* d_x,
    float* d_scores_scratch,
    int* d_out_indices,
    int compressed_count,
    int keep,
    int heads,
    int head_dim,
    int dim,
    int offset,
    void* stream) {
    if (d_index_q == nullptr || d_index_kv == nullptr || d_weight_proj_bf16 == nullptr || d_x == nullptr || d_scores_scratch == nullptr || d_out_indices == nullptr) return false;
    if (compressed_count <= 0 || keep <= 0 || keep > compressed_count || heads <= 0 || head_dim <= 0 || dim <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    float* d_head_weights = d_scores_scratch;
    float* d_scores = d_scores_scratch + heads;
    indexer_head_weights_kernel<<<heads, 256, 0, cuda_stream>>>(d_weight_proj_bf16, d_x, d_head_weights, heads, head_dim, dim);
    indexer_score_kernel<<<compressed_count, 256, 0, cuda_stream>>>(d_index_q, d_index_kv, d_head_weights, d_scores, compressed_count, heads, head_dim);
    indexer_topk_kernel<<<1, 256, 256 * (sizeof(float) + sizeof(int)), cuda_stream>>>(d_scores, d_out_indices, compressed_count, keep, offset);
    return cudaGetLastError() == cudaSuccess;
}

bool hadamard128_rows_cuda(const float* d_x, float* d_y, int rows, void* stream) {
    if (d_x == nullptr || d_y == nullptr || rows <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    hadamard128_rows_kernel<<<rows, 128, 0, cuda_stream>>>(d_x, d_y, rows);
    return cudaGetLastError() == cudaSuccess;
}

bool fp4_fake_quant128_rows_cuda(float* d_x, int rows, void* stream) {
    if (d_x == nullptr || rows <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fp4_fake_quant128_rows_kernel<<<rows, 128, 0, cuda_stream>>>(d_x, rows);
    return cudaGetLastError() == cudaSuccess;
}

bool fp8_act_quant_dequant_cuda(float* d_x, int cols, int block_size, void* stream) {
    if (d_x == nullptr || cols <= 0 || block_size <= 0 || (cols % block_size) != 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fp8_act_quant_dequant_kernel<<<cols / block_size, 256, 0, cuda_stream>>>(d_x, cols, block_size);
    return cudaGetLastError() == cudaSuccess;
}

bool compressor_update_state_cuda(const float* d_kv, const float* d_score, const float* d_ape, float* d_kv_state, float* d_score_state, int offset, int write_slot, int state_cols, void* stream) {
    (void)offset;
    if (d_kv == nullptr || d_score == nullptr || d_ape == nullptr || d_kv_state == nullptr || d_score_state == nullptr || write_slot < 0 || state_cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int threads = 256;
    const int blocks = (state_cols + threads - 1) / threads;
    compressor_update_state_kernel<<<blocks, threads, 0, cuda_stream>>>(d_kv, d_score, d_ape, d_kv_state, d_score_state, write_slot, state_cols);
    return cudaGetLastError() == cudaSuccess;
}

bool compressor_pool_cuda(const float* d_kv_state, const float* d_score_state, float* d_out, int ratio, int head_dim, int state_cols, bool overlap, void* stream) {
    if (d_kv_state == nullptr || d_score_state == nullptr || d_out == nullptr || ratio <= 0 || head_dim <= 0 || state_cols < head_dim) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int threads = 256;
    const int blocks = (head_dim + threads - 1) / threads;
    compressor_pool_kernel<<<blocks, threads, 0, cuda_stream>>>(d_kv_state, d_score_state, d_out, ratio, head_dim, state_cols, overlap);
    return cudaGetLastError() == cudaSuccess;
}

bool compressor_shift_overlap_state_cuda(float* d_kv_state, float* d_score_state, int ratio, int state_cols, void* stream) {
    if (d_kv_state == nullptr || d_score_state == nullptr || ratio <= 0 || state_cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int total = ratio * state_cols;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    compressor_shift_overlap_state_kernel<<<blocks, threads, 0, cuda_stream>>>(d_kv_state, d_score_state, ratio, state_cols);
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

bool head_rmsnorm_rope_freqs_cuda(
    float* d_x,
    const float* d_inv_freqs,
    int heads,
    int head_dim,
    int rope_dim,
    int position,
    bool inverse,
    float eps,
    void* stream) {
    if (d_x == nullptr || d_inv_freqs == nullptr || heads <= 0 || head_dim <= 0 || rope_dim <= 0 || rope_dim > head_dim || (rope_dim % 2) != 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    head_rmsnorm_rope_freqs_kernel<<<heads, 256, 0, cuda_stream>>>(d_x, d_inv_freqs, head_dim, rope_dim, position, inverse, eps);
    return cudaGetLastError() == cudaSuccess;
}

bool fp32_to_bf16_cuda(const float* d_x, uint16_t* d_y, int count, void* stream) {
    if (d_x == nullptr || d_y == nullptr || count <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    fp32_to_bf16_kernel<<<blocks, threads, 0, cuda_stream>>>(d_x, d_y, count);
    return cudaGetLastError() == cudaSuccess;
}

bool bf16_to_fp32_cuda(const uint16_t* d_x, float* d_y, int count, void* stream) {
    if (d_x == nullptr || d_y == nullptr || count <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    bf16_to_fp32_kernel<<<blocks, threads, 0, cuda_stream>>>(d_x, d_y, count);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace dsv4
