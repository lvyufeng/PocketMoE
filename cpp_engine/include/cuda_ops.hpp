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

bool moe_single_token_fp4_cuda(
    const float* d_x,
    const int64_t* d_indices,
    const float* d_weights,
    const uint8_t* d_w1q,
    const uint8_t* d_w1s,
    const uint8_t* d_w2q,
    const uint8_t* d_w2s,
    const uint8_t* d_w3q,
    const uint8_t* d_w3s,
    float* d_y,
    int topk,
    int experts_start_idx,
    int n_local_experts,
    int dim,
    int inter_dim,
    float swiglu_limit,
    void* stream = nullptr);

bool moe_group_routes_cuda(
    const int64_t* d_indices,
    const float* d_weights,
    int64_t* d_route_tokens,
    float* d_route_weights,
    int32_t* d_seg_starts,
    int32_t* d_counts,
    int32_t* d_offsets,
    int32_t* d_total_routes,
    int tokens,
    int topk,
    int experts_start_idx,
    int n_local_experts,
    void* stream = nullptr);

bool moe_prefill_fp4_grouped_cuda(
    const float* d_x,
    const int64_t* d_route_tokens,
    const float* d_route_weights,
    const int32_t* d_seg_starts,
    const uint8_t* d_w1q,
    const uint8_t* d_w1s,
    const uint8_t* d_w2q,
    const uint8_t* d_w2s,
    const uint8_t* d_w3q,
    const uint8_t* d_w3s,
    float* d_y,
    int tokens,
    int routes,
    int n_local_experts,
    int max_count,
    int dim,
    int inter_dim,
    float swiglu_limit,
    void* stream = nullptr);

bool fp8_e4m3_e8m0_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int rows,
    int cols,
    void* stream = nullptr);

bool fp8_e4m3_e8m0_matmul_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int batch,
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

bool rmsnorm_bf16_gamma_rows_cuda(
    const float* d_x,
    const uint16_t* d_gamma_bf16,
    float* d_y,
    int rows,
    int cols,
    float eps,
    void* stream = nullptr);

bool silu_mul_cuda(
    const float* d_gate,
    const float* d_up,
    float* d_y,
    int cols,
    void* stream = nullptr);

bool silu_mul_rows_cuda(
    const float* d_gate,
    const float* d_up,
    float* d_y,
    int rows,
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

bool bf16_rows_to_float_cuda(
    const uint16_t* d_matrix_bf16,
    const int* d_rows,
    float* d_y,
    int rows,
    int cols,
    void* stream = nullptr);

bool bf16_matvec_cuda(
    const float* d_x,
    const uint16_t* d_w_bf16,
    float* d_y,
    int rows,
    int cols,
    void* stream = nullptr);

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
    void* stream = nullptr);

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
    void* stream = nullptr);

bool gate_topk_bf16_rows_cuda(
    const float* d_x,
    const uint16_t* d_w_bf16,
    const float* d_bias,
    int64_t* d_indices,
    float* d_weights,
    int tokens,
    int experts,
    int cols,
    int topk,
    float route_scale,
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

bool vector_accum_rows_cuda(
    const float* d_x,
    float* d_y,
    int rows,
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

bool fp8_act_quant_dequant_cuda(
    float* d_x,
    int cols,
    int block_size,
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
