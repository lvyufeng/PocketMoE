#pragma once

#include <cstddef>
#include <cstdint>

namespace dsv4 {

bool cuda_runtime_available();

bool q8_0_matvec_cuda(
    const float* d_x,
    const uint8_t* d_w,
    float* d_y,
    int rows,
    int cols,
    void* stream = nullptr);

bool fp4_e2m1_e8m0_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int rows,
    int cols,
    void* stream = nullptr);

bool fp8_e4m3_e8m0_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int rows,
    int cols,
    void* stream = nullptr);

bool rmsnorm_bf16_gamma_cuda(
    const float* d_x,
    const uint16_t* d_gamma_bf16,
    float* d_y,
    int cols,
    float eps,
    void* stream = nullptr);

bool silu_mul_cuda(
    const float* d_gate,
    const float* d_up,
    float* d_y,
    int cols,
    void* stream = nullptr);

bool bf16_row_to_float_cuda(
    const uint16_t* d_matrix_bf16,
    float* d_y,
    int row,
    int cols,
    void* stream = nullptr);

bool bf16_matvec_cuda(
    const float* d_x,
    const uint16_t* d_w_bf16,
    float* d_y,
    int rows,
    int cols,
    void* stream = nullptr);

bool vector_add_cuda(
    const float* d_a,
    const float* d_b,
    float* d_y,
    int cols,
    void* stream = nullptr);

}  // namespace dsv4
