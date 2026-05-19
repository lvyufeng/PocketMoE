#include "cuda_ops.hpp"

#include <cuda_runtime.h>

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

}  // namespace dsv4
