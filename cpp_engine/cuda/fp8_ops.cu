#include "cuda_ops.hpp"

#include <cuda_runtime.h>

namespace dsv4 {
namespace {

constexpr int kFp8BlockSize = 128;

__device__ __forceinline__ float fp8_e4m3_value(uint8_t code) {
    const int sign = (code >> 7) & 0x1;
    const int exp = (code >> 3) & 0xf;
    const int mant = code & 0x7;
    float value;
    if (exp == 0) {
        value = ldexpf(static_cast<float>(mant) * (1.0f / 8.0f), -6);
    } else {
        value = ldexpf(1.0f + static_cast<float>(mant) * (1.0f / 8.0f), exp - 7);
    }
    return sign ? -value : value;
}

__device__ __forceinline__ float fp8_e8m0_value(uint8_t code) {
    return exp2f(static_cast<float>(static_cast<int>(code) - 127));
}

__global__ void fp8_e4m3_e8m0_matvec_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale,
    float* __restrict__ y,
    int rows,
    int cols,
    int scale_cols) {
    const int row = blockIdx.x;
    const int batch = blockIdx.y;
    if (row >= rows) return;
    const int row_block = row / kFp8BlockSize;
    const float* batch_x = x + static_cast<size_t>(batch) * cols;
    float* batch_y = y + static_cast<size_t>(batch) * rows;

    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const int col_block = col / kFp8BlockSize;
        const uint8_t code = weight[static_cast<size_t>(row) * cols + col];
        const float s = fp8_e8m0_value(scale[static_cast<size_t>(row_block) * scale_cols + col_block]);
        sum += fp8_e4m3_value(code) * s * batch_x[col];
    }

    extern __shared__ float scratch[];
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) batch_y[row] = scratch[0];
}

}  // namespace

bool fp8_e4m3_e8m0_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int rows,
    int cols,
    void* stream) {
    if (d_x == nullptr || d_weight == nullptr || d_scale == nullptr || d_y == nullptr) return false;
    if (rows <= 0 || cols <= 0) return false;
    if ((rows % kFp8BlockSize) != 0 || (cols % kFp8BlockSize) != 0) return false;
    const int threads = 256;
    const int scale_cols = cols / kFp8BlockSize;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fp8_e4m3_e8m0_matvec_kernel<<<dim3(rows, 1), threads, threads * sizeof(float), cuda_stream>>>(
        d_x, d_weight, d_scale, d_y, rows, cols, scale_cols);
    return cudaGetLastError() == cudaSuccess;
}

bool fp8_e4m3_e8m0_matmul_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int batch,
    int rows,
    int cols,
    void* stream) {
    if (d_x == nullptr || d_weight == nullptr || d_scale == nullptr || d_y == nullptr) return false;
    if (batch <= 0 || rows <= 0 || cols <= 0) return false;
    if ((rows % kFp8BlockSize) != 0 || (cols % kFp8BlockSize) != 0) return false;
    const int threads = 256;
    const int scale_cols = cols / kFp8BlockSize;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fp8_e4m3_e8m0_matvec_kernel<<<dim3(rows, batch), threads, threads * sizeof(float), cuda_stream>>>(
        d_x, d_weight, d_scale, d_y, rows, cols, scale_cols);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace dsv4
