#include "cuda_ops.hpp"

#include <cuda_runtime.h>

namespace dsv4 {
namespace {

__device__ __forceinline__ float bf16_to_float(uint16_t bits) {
    uint32_t value = static_cast<uint32_t>(bits) << 16;
    float out;
    __builtin_memcpy(&out, &value, sizeof(out));
    return out;
}

__global__ void bf16_row_to_float_kernel(const uint16_t* matrix, float* y, int row, int cols) {
    const uint16_t* src = matrix + static_cast<size_t>(row) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) y[c] = bf16_to_float(src[c]);
}

__global__ void bf16_matvec_kernel(const float* x, const uint16_t* w, float* y, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        sum += bf16_to_float(w[static_cast<size_t>(row) * cols + c]) * x[c];
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

__device__ __forceinline__ float sqrt_softplus(float x) {
    return sqrtf(log1pf(expf(x)));
}

__global__ void gate_scores_kernel(const float* x, const uint16_t* w, const float* bias, float* original, float* scored, int experts, int cols) {
    const int row = blockIdx.x;
    if (row >= experts) return;
    float sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) sum += bf16_to_float(w[static_cast<size_t>(row) * cols + c]) * x[c];
    extern __shared__ float scratch[];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float orig = sqrt_softplus(scratch[0]);
        original[row] = orig;
        scored[row] = orig + (bias == nullptr ? 0.0f : bias[row]);
    }
}

__global__ void gate_topk_finalize_kernel(const float* original, const float* scored, int64_t* indices, float* weights, int experts, int topk, float route_scale) {
    if (threadIdx.x != 0) return;
    float denom = 0.0f;
    for (int k = 0; k < topk; ++k) {
        int best = 0;
        float best_score = -INFINITY;
        for (int e = 0; e < experts; ++e) {
            bool used = false;
            for (int j = 0; j < k; ++j) used = used || (indices[j] == e);
            const float s = used ? -INFINITY : scored[e];
            if (s > best_score) {
                best_score = s;
                best = e;
            }
        }
        indices[k] = best;
        denom += original[best];
    }
    denom = denom == 0.0f ? 1.0f : denom;
    for (int k = 0; k < topk; ++k) weights[k] = original[indices[k]] / denom * route_scale;
}

}  // namespace

bool bf16_row_to_float_cuda(const uint16_t* d_matrix_bf16, float* d_y, int row, int cols, void* stream) {
    if (d_matrix_bf16 == nullptr || d_y == nullptr || row < 0 || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    bf16_row_to_float_kernel<<<1, 256, 0, cuda_stream>>>(d_matrix_bf16, d_y, row, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool bf16_matvec_cuda(const float* d_x, const uint16_t* d_w_bf16, float* d_y, int rows, int cols, void* stream) {
    if (d_x == nullptr || d_w_bf16 == nullptr || d_y == nullptr || rows <= 0 || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    bf16_matvec_kernel<<<rows, 256, 256 * sizeof(float), cuda_stream>>>(d_x, d_w_bf16, d_y, rows, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool gate_topk_bf16_cuda_with_buffers(
    const float* d_x,
    const uint16_t* d_w_bf16,
    const float* d_bias,
    float* d_original,
    float* d_scored,
    int64_t* d_indices,
    float* d_weights,
    int experts,
    int cols,
    int topk,
    float route_scale,
    void* stream) {
    if (d_x == nullptr || d_w_bf16 == nullptr || d_original == nullptr || d_scored == nullptr || d_indices == nullptr || d_weights == nullptr || experts <= 0 || cols <= 0 || topk <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    gate_scores_kernel<<<experts, 256, 256 * sizeof(float), cuda_stream>>>(d_x, d_w_bf16, d_bias, d_original, d_scored, experts, cols);
    gate_topk_finalize_kernel<<<1, 1, 0, cuda_stream>>>(d_original, d_scored, d_indices, d_weights, experts, topk, route_scale);
    return cudaGetLastError() == cudaSuccess;
}

bool gate_topk_bf16_cuda(
    const float* d_x,
    const uint16_t* d_w_bf16,
    const float* d_bias,
    int64_t* d_indices,
    float* d_weights,
    int experts,
    int cols,
    int topk,
    float route_scale,
    void* stream) {
    if (d_x == nullptr || d_w_bf16 == nullptr || d_indices == nullptr || d_weights == nullptr || experts <= 0 || cols <= 0 || topk <= 0) return false;
    float* d_original = nullptr;
    float* d_scored = nullptr;
    if (cudaMalloc(&d_original, static_cast<size_t>(experts) * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&d_scored, static_cast<size_t>(experts) * sizeof(float)) != cudaSuccess) {
        cudaFree(d_original);
        return false;
    }
    const bool ok = gate_topk_bf16_cuda_with_buffers(d_x, d_w_bf16, d_bias, d_original, d_scored, d_indices, d_weights, experts, cols, topk, route_scale, stream);
    cudaFree(d_original);
    cudaFree(d_scored);
    return ok;
}

}  // namespace dsv4
