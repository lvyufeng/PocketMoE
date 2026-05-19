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

bool silu_mul_clamped_cuda(
    const float* d_gate,
    const float* d_up,
    float* d_y,
    int cols,
    float limit,
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

bool vector_accum_cuda(
    const float* d_x,
    float* d_y,
    int cols,
    float scale,
    void* stream = nullptr);

bool repeat_vector_cuda(
    const float* d_x,
    float* d_y,
    int cols,
    int repeats,
    void* stream = nullptr);

bool single_token_sparse_attention_cuda(
    const float* d_q,
    const float* d_kv,
    const float* d_attn_sink,
    float* d_y,
    int heads,
    int head_dim,
    float scale,
    void* stream = nullptr);

bool cached_single_token_attention_cuda(
    const float* d_q,
    const float* d_kv_cache,
    const float* d_attn_sink,
    float* d_y,
    int heads,
    int head_dim,
    int cache_len,
    float scale,
    void* stream = nullptr);

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
    void* stream = nullptr);

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
    void* stream = nullptr);

bool head_rmsnorm_rope_cuda(
    float* d_x,
    int heads,
    int head_dim,
    int rope_dim,
    int position,
    float theta,
    bool inverse,
    float eps,
    void* stream = nullptr);

}  // namespace dsv4
