#include "cuda_ops.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace dsv4 {
namespace {

__device__ __forceinline__ float load_f16_scale(const uint8_t* p) {
    const uint16_t h = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

__global__ void q8_0_matvec_kernel(const float* __restrict__ x, const uint8_t* __restrict__ w, float* __restrict__ y, int rows, int cols, int blocks_per_row) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    extern __shared__ float shared[];
    float sum = 0.0f;
    const int tid = threadIdx.x;
    const uint8_t* row_ptr = w + static_cast<size_t>(row) * blocks_per_row * 34;
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        const uint8_t* block = row_ptr + b * 34;
        const float d = load_f16_scale(block);
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
        const int base = b * 32;
        float block_sum = 0.0f;
        for (int j = 0; j < 32; ++j) {
            const int col = base + j;
            if (col < cols) {
                block_sum += x[col] * static_cast<float>(qs[j]);
            }
        }
        sum += d * block_sum;
    }
    shared[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        y[row] = shared[0];
    }
}

}  // namespace

bool q8_0_matvec_cuda(const float* d_x, const uint8_t* d_w, float* d_y, int rows, int cols, void* stream) {
    if (d_x == nullptr || d_w == nullptr || d_y == nullptr || rows <= 0 || cols <= 0) {
        return false;
    }
    const int blocks_per_row = (cols + 31) / 32;
    const int threads = 128;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    q8_0_matvec_kernel<<<rows, threads, threads * sizeof(float), cuda_stream>>>(d_x, d_w, d_y, rows, cols, blocks_per_row);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace dsv4
