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

// --- IQ2_XXS / Q2_K single-token MoE decode kernels (cuda/q2_ops.cu). ---

// Returns a device pointer to the lazy-initialized signed_grid lookup table
// (shape [256, 128, 8] int8, ~256 KiB). Returns nullptr on allocation failure.
const int8_t* q2_signed_grid_device();

// Quantize fp32 activations to Q8_1 with 32-element groups (one row per launch).
bool q2_quantize_x_q8_1_cuda(
    const float* d_x,
    int8_t* d_x_q,
    float* d_x_scale,
    int rows,
    int row_elems,
    void* stream = nullptr);

// Quantize fp32 hidden activations to Q8_1 with 16-element groups (post-SwiGLU).
bool q2_quantize_hidden_q8_1_cuda(
    const float* d_hidden,
    int8_t* d_hidden_q,
    float* d_hidden_scale,
    int routes,
    int inter_dim,
    void* stream = nullptr);

// Fused: SwiGLU(gate, up) * route_weight -> Q8_1 with 16-element groups.
bool q2_route_swiglu_quantize_hidden_q8_1_cuda(
    const float* d_gate,
    const float* d_up,
    const float* d_route_weights,
    int8_t* d_hidden_q,
    float* d_hidden_scale,
    int routes,
    int inter_dim,
    float swiglu_limit,
    void* stream = nullptr);

// IQ2_XXS W1+W3 single-token matvec (per-route gate/up outputs).
// d_w1_blocks/d_w3_blocks layout: per-expert [inter_dim, blocks_per_row*66] flat bytes
// where blocks_per_row = dim / 256. Output: d_gate/d_up [routes, inter_dim].
bool q2_moe_single_w13_iq2_xxs_cuda(
    const int8_t* d_x_q,
    const float* d_x_scale,
    const int64_t* d_route_slots,
    const uint8_t* d_w1_blocks,
    const uint8_t* d_w3_blocks,
    float* d_gate,
    float* d_up,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    void* stream = nullptr);

// Q2_K W2 single-token matvec. Accumulates onto d_y (caller must zero beforehand).
// d_w2_blocks layout: per-expert [dim, blocks_per_row*84] flat bytes
// where blocks_per_row = inter_dim / 256.
bool q2_moe_single_w2_q2k_cuda(
    const int8_t* d_hidden_q,
    const float* d_hidden_scale,
    const int64_t* d_route_slots,
    const uint8_t* d_w2_blocks,
    float* d_y,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    void* stream = nullptr);

bool fp4_e2m1_e8m0_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int rows,
    int cols,
    void* stream = nullptr);

struct MoeSingleTokenFp4Workspace {
    int8_t* d_x_q = nullptr;
    float* d_x_scale = nullptr;
    float* d_gate = nullptr;
    float* d_up = nullptr;
    int8_t* d_hidden_q = nullptr;
    float* d_hidden_scale = nullptr;
    float* d_route_y = nullptr;
    int topk = 0;
    int dim = 0;
    int inter_dim = 0;
};

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

bool moe_single_token_fp4_cuda_with_workspace(
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
    MoeSingleTokenFp4Workspace workspace,
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
    int topk,
    int routes,
    int n_local_experts,
    int max_count,
    int dim,
    int inter_dim,
    float swiglu_limit,
    void* stream = nullptr);

struct MoePrefillFp4GroupedWorkspace {
    float* d_x_sorted = nullptr;
    int8_t* d_x_q = nullptr;
    float* d_x_scale = nullptr;
    int8_t* d_x_pad = nullptr;
    float* d_x_scale_pad = nullptr;
    float* d_gate = nullptr;
    float* d_up = nullptr;
    int8_t* d_hidden_q = nullptr;
    float* d_hidden_scale = nullptr;
    int32_t* d_tile_experts = nullptr;
    int32_t* d_tile_rows = nullptr;
    int routes_cap = 0;
    int padded_rows_cap = 0;
    int tile_cap = 0;
    int tile_count = 0;
    int dim = 0;
    int inter_dim = 0;
};

bool moe_prefill_fp4_grouped_cuda_with_workspace(
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
    int topk,
    int routes,
    int n_local_experts,
    int max_count,
    int dim,
    int inter_dim,
    float swiglu_limit,
    MoePrefillFp4GroupedWorkspace workspace,
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

bool fp8_e4m3_e8m0_matmul_strided_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    void* stream = nullptr);

bool wo_a_int8_decode_cuda(
    const float* d_x,
    const int8_t* d_weight_q,
    const float* d_weight_scale,
    float* d_y,
    int groups,
    int group_rank,
    int group_dim,
    int8_t* d_x_q,
    float* d_x_scale,
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

// GGUF embed/head are stored F16 (IEEE half), not BF16. This is the F16
// counterpart of bf16_row_to_float_cuda for the GGUF dense path.
bool f16_row_to_float_cuda(
    const uint16_t* d_matrix_f16,
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

bool bf16_dual_matvec_cuda(
    const float* d_x,
    const uint16_t* d_w_a_bf16,
    const uint16_t* d_w_b_bf16,
    float* d_y_a,
    float* d_y_b,
    int rows,
    int cols,
    void* stream = nullptr);

bool bf16_matvec_cpu_order_cuda(
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

bool gate_hash_bf16_rows_cuda(
    const float* d_x,
    const uint16_t* d_w_bf16,
    const int64_t* d_tid2eid,
    const int* d_token_ids,
    int64_t* d_indices,
    float* d_weights,
    int tokens,
    int cols,
    int table_topk,
    int topk,
    float route_scale,
    void* stream = nullptr);

bool gate_hash_bf16_cuda(
    const float* d_x,
    const uint16_t* d_w_bf16,
    const int64_t* d_tid2eid,
    float* d_original_scratch,
    int64_t* d_indices,
    float* d_weights,
    int token,
    int cols,
    int table_topk,
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

bool prefill_causal_attention_cuda(
    const float* d_q,
    const float* d_kv,
    const float* d_attn_sink,
    float* d_y,
    int tokens,
    int heads,
    int head_dim,
    int window_size,
    float scale,
    void* stream = nullptr);

bool build_prefill_window_indices_cuda(
    int32_t* d_indices,
    int tokens,
    int window_size,
    int topk,
    void* stream = nullptr);

bool build_decode_kv_indices_cuda(
    int* d_indices,
    int window_start,
    int window_len,
    int window_size,
    int compressed_count,
    int compressed_offset,
    void* stream = nullptr);

bool prefill_sparse_attention_headpair_cuda(
    const float* d_q,
    const float* d_kv,
    const float* d_attn_sink,
    const int32_t* d_topk_indices,
    float* d_y,
    int tokens,
    int heads,
    int kv_len,
    int topk,
    int head_dim,
    float scale,
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
    float* d_scores_scratch,
    int* d_out_indices,
    int compressed_count,
    int keep,
    int heads,
    int head_dim,
    int dim,
    int offset,
    void* stream = nullptr);

bool hadamard128_rows_cuda(
    const float* d_x,
    float* d_y,
    int rows,
    void* stream = nullptr);

bool fp4_fake_quant128_rows_cuda(
    float* d_x,
    int rows,
    void* stream = nullptr);

bool fp8_act_quant_dequant_cuda(
    float* d_x,
    int cols,
    int block_size,
    void* stream = nullptr);

bool compressor_update_state_cuda(
    const float* d_kv,
    const float* d_score,
    const float* d_ape,
    float* d_kv_state,
    float* d_score_state,
    int offset,
    int write_slot,
    int state_cols,
    void* stream = nullptr);

bool compressor_pool_cuda(
    const float* d_kv_state,
    const float* d_score_state,
    float* d_out,
    int ratio,
    int head_dim,
    int state_cols,
    bool overlap,
    void* stream = nullptr);

bool compressor_shift_overlap_state_cuda(
    float* d_kv_state,
    float* d_score_state,
    int ratio,
    int state_cols,
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

bool head_rmsnorm_rope_freqs_cuda(
    float* d_x,
    const float* d_inv_freqs,
    int heads,
    int head_dim,
    int rope_dim,
    int position,
    bool inverse,
    float eps,
    void* stream = nullptr);

bool head_rmsnorm_rope_freqs_rows_cuda(
    float* d_x,
    const float* d_inv_freqs,
    int tokens,
    int heads,
    int head_dim,
    int rope_dim,
    int start_position,
    bool inverse,
    float eps,
    void* stream = nullptr);

bool fp8_act_quant_dequant_rows_cuda(
    float* d_x,
    int rows,
    int cols,
    int block_size,
    void* stream = nullptr);

bool fp8_act_quant_dequant_rows_strided_cuda(
    float* d_x,
    int rows,
    int cols,
    int row_stride,
    int block_size,
    void* stream = nullptr);

bool copy_rows_to_kv_cache_cuda(
    const float* d_rows,
    float* d_cache,
    int rows,
    int cols,
    int window_size,
    int start_position = 0,
    void* stream = nullptr);

bool fp32_to_bf16_cuda(
    const float* d_x,
    uint16_t* d_y,
    int count,
    void* stream = nullptr);

bool bf16_to_fp32_cuda(
    const uint16_t* d_x,
    float* d_y,
    int count,
    void* stream = nullptr);

bool hc_repeat_rows_cuda(
    const float* d_x_rows,
    float* d_h4_rows,
    int rows,
    int dim,
    void* stream = nullptr);

bool hc_pre_float_rows_cuda(
    const float* d_h4_rows,
    const float* d_fn,
    const float* d_scale,
    const float* d_base,
    float* d_x_rows,
    float* d_post_rows,
    float* d_comb_rows,
    int rows,
    int dim,
    void* stream = nullptr);

bool hc_post_float_rows_cuda(
    const float* d_x_rows,
    const float* d_residual_h4_rows,
    const float* d_post_rows,
    const float* d_comb_rows,
    float* d_y_h4_rows,
    int rows,
    int dim,
    void* stream = nullptr);


bool hc_pre_float_cuda(
    const float* d_h4,
    const float* d_fn,
    const float* d_scale,
    const float* d_base,
    float* d_x,
    float* d_post,
    float* d_comb,
    int dim,
    void* stream = nullptr);

bool hc_post_float_cuda(
    const float* d_x,
    const float* d_residual_h4,
    const float* d_post,
    const float* d_comb,
    float* d_y_h4,
    int dim,
    void* stream = nullptr);

}  // namespace dsv4
