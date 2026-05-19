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

}  // namespace dsv4
