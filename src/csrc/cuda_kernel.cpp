#include <torch/extension.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>


torch::Tensor int8_gemm_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_s);

torch::Tensor int8_gemm_pair_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_q0,
    const torch::Tensor& weight_s0,
    const torch::Tensor& weight_q1,
    const torch::Tensor& weight_s1);

torch::Tensor wo_a_int8_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_s);

torch::Tensor fused_decode_sparse_attn_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& attn_sink,
    const torch::Tensor& topk_idxs,
    double softmax_scale);

torch::Tensor fused_decode_sparse_attn_wmma_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& attn_sink,
    const torch::Tensor& topk_idxs,
    double softmax_scale);

torch::Tensor flashinfer_style_sparse_attn_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& attn_sink,
    const torch::Tensor& topk_idxs,
    double softmax_scale);

torch::Tensor flashinfer_style_sparse_attn_headpair_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& attn_sink,
    const torch::Tensor& topk_idxs,
    double softmax_scale);

torch::Tensor prefill_sparse_attn_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& attn_sink,
    const torch::Tensor& topk_idxs,
    double softmax_scale);

torch::Tensor prefill_sparse_attn_headpair_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& attn_sink,
    const torch::Tensor& topk_idxs,
    double softmax_scale);

torch::Tensor hadamard128_forward_cuda(const torch::Tensor& x);

torch::Tensor fused_c4_indexer_decode_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& score,
    const torch::Tensor& weights,
    const torch::Tensor& ape,
    const torch::Tensor& norm_weight,
    const torch::Tensor& freqs,
    const torch::Tensor& kv_state,
    const torch::Tensor& score_state,
    const torch::Tensor& kv_cache,
    int64_t start_pos,
    int64_t offset,
    int64_t index_topk,
    double norm_eps,
    bool return_scores);

torch::Tensor c4_topk_from_scores_cuda(
    const torch::Tensor& scores,
    int64_t offset,
    int64_t index_topk);

std::vector<torch::Tensor> hc_split_pre_forward_cuda(
    const torch::Tensor& mixes,
    const torch::Tensor& x,
    const torch::Tensor& hc_scale,
    const torch::Tensor& hc_base,
    int64_t hc_mult,
    int64_t sinkhorn_iters,
    double eps);

torch::Tensor hc_post_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb);

std::vector<torch::Tensor> moe_group_routes_cuda(
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    int64_t experts_start_idx,
    int64_t n_local_experts);

torch::Tensor moe_single_token_int8_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    int64_t experts_start_idx,
    double swiglu_limit);

torch::Tensor moe_single_token_fp4_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    int64_t experts_start_idx,
    double swiglu_limit);

torch::Tensor moe_single_token_int8_forward_v2_cuda(
    const torch::Tensor& x,
    const torch::Tensor& route_to_slot,
    const torch::Tensor& weights,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    double swiglu_limit);
torch::Tensor moe_prefill_int8_grouped_forward_cuda(
    const torch::Tensor& x_sorted,
    const torch::Tensor& route_weights_sorted,
    const torch::Tensor& seg_starts,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    double swiglu_limit);

torch::Tensor moe_prefill_int8_grouped_gemm_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& route_tokens,
    const torch::Tensor& route_weights_sorted,
    const torch::Tensor& seg_starts,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    double swiglu_limit);

torch::Tensor moe_prefill_fp4_grouped_gemm_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& route_tokens,
    const torch::Tensor& route_weights_sorted,
    const torch::Tensor& seg_starts,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    double swiglu_limit);

std::vector<torch::Tensor> fp4_weight_to_int8_forward_cuda(
    const torch::Tensor& wq,
    const torch::Tensor& ws);

torch::Tensor moe_prefill_int8_grouped_gemm_bucketed_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& route_tokens,
    const torch::Tensor& route_weights_sorted,
    const torch::Tensor& seg_starts,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    double swiglu_limit);

torch::Tensor moe_prefill_int8_fused_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& route_tokens,
    const torch::Tensor& route_weights_sorted,
    const torch::Tensor& seg_starts,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    double swiglu_limit);

void fused_q_rmsnorm_rope_inplace_cuda(
    torch::Tensor& q,
    const torch::Tensor& freqs_real,
    const torch::Tensor& freqs_imag,
    double eps);

void fused_kv_rope_actquant_inplace_cuda(
    torch::Tensor& kv,
    const torch::Tensor& norm_weight,
    const torch::Tensor& freqs_real,
    const torch::Tensor& freqs_imag,
    int64_t block_size,
    double norm_eps,
    const c10::optional<torch::Tensor>& kv_cache_out,
    int64_t cache_slot);

torch::Tensor int8_gemm_imma_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_s);

torch::Tensor int8_gemm_pair_imma_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_q0,
    const torch::Tensor& weight_s0,
    const torch::Tensor& weight_q1,
    const torch::Tensor& weight_s1);

torch::Tensor q8_0_gemm_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& blocks,
    int64_t row_elems);

torch::Tensor gguf_q2k_gemm_dp4a_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& blocks,
    int64_t row_elems);

torch::Tensor gguf_quant_gemm_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& blocks,
    int64_t row_elems,
    int64_t type_id,
    const torch::Tensor& signed_grid);

torch::Tensor gguf_quant_gemm_prefill_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& blocks,
    int64_t row_elems,
    int64_t type_id,
    const torch::Tensor& signed_grid);

torch::Tensor gguf_quant_gemm_pair_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& blocks0,
    int64_t row_elems0,
    int64_t type_id0,
    const torch::Tensor& blocks1,
    int64_t row_elems1,
    int64_t type_id1,
    const torch::Tensor& signed_grid);

torch::Tensor gguf_moe_prefill_grouped_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& route_tokens,
    const torch::Tensor& route_weights_sorted,
    const torch::Tensor& seg_starts,
    const torch::Tensor& w1_blocks,
    const torch::Tensor& w3_blocks,
    const torch::Tensor& w2_blocks,
    int64_t w1_row_elems,
    int64_t w1_type_id,
    int64_t w3_row_elems,
    int64_t w3_type_id,
    int64_t w2_row_elems,
    int64_t w2_type_id,
    const torch::Tensor& signed_grid,
    double swiglu_limit);

torch::Tensor gguf_moe_single_token_iq2_q2k_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& route_slots,
    const torch::Tensor& route_weights,
    const torch::Tensor& w1_blocks,
    const torch::Tensor& w3_blocks,
    const torch::Tensor& w2_blocks,
    const torch::Tensor& signed_grid,
    double swiglu_limit);

void fused_o_inverse_rope_inplace_cuda(
    torch::Tensor& o,
    const torch::Tensor& freqs_real,
    const torch::Tensor& freqs_imag);

std::vector<std::string> custom_allreduce_ipc_handle_cuda(
    const torch::Tensor& buffer,
    const torch::Tensor& flags);

int64_t custom_allreduce_open_cuda(
    const torch::Tensor& local_buffer,
    const torch::Tensor& local_flags,
    const std::vector<std::string>& buffer_handles,
    const std::vector<std::string>& flag_handles,
    int64_t rank,
    int64_t world_size,
    int64_t dim,
    int64_t dtype_code);

void custom_allreduce_close_cuda(int64_t handle);

bool custom_allreduce_inplace_cuda(
    int64_t handle,
    torch::Tensor& tensor,
    int64_t seq,
    int64_t timeout_us);

torch::Tensor moe_finalize_reduce_forward_cuda(
    const torch::Tensor& y_reduce,
    const torch::Tensor& shared,
    int64_t out_dtype_code);


namespace {

void check_tensor(const torch::Tensor& tensor, const char* name, c10::ScalarType dtype) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has unexpected dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

bool int8_gemm_imma_enabled() {
    static const bool enabled = []() {
        const char* v = std::getenv("DEEPSEEK_INT8_GEMM_IMMA");
        if (!v || !*v) return false;
        if (std::strcmp(v, "0") == 0) return false;
        if (std::strcmp(v, "false") == 0) return false;
        if (std::strcmp(v, "False") == 0) return false;
        return true;
    }();
    return enabled;
}

}  // namespace

pybind11::tuple custom_allreduce_ipc_handle(const torch::Tensor& buffer, const torch::Tensor& flags) {
    TORCH_CHECK(buffer.is_cuda() && flags.is_cuda(), "custom allreduce buffers must be CUDA tensors");
    TORCH_CHECK(buffer.is_contiguous() && flags.is_contiguous(), "custom allreduce buffers must be contiguous");
    TORCH_CHECK(flags.scalar_type() == torch::kInt32, "custom allreduce flags must be int32");
    TORCH_CHECK(buffer.scalar_type() == torch::kFloat16 || buffer.scalar_type() == torch::kBFloat16 || buffer.scalar_type() == torch::kFloat32,
                "custom allreduce buffer must be float16, bfloat16, or float32");
    auto handles = custom_allreduce_ipc_handle_cuda(buffer, flags);
    return pybind11::make_tuple(pybind11::bytes(handles[0]), pybind11::bytes(handles[1]));
}

int64_t custom_allreduce_open(
    const torch::Tensor& local_buffer,
    const torch::Tensor& local_flags,
    const std::vector<std::string>& buffer_handles,
    const std::vector<std::string>& flag_handles,
    int64_t rank,
    int64_t world_size,
    int64_t dim,
    int64_t dtype_code) {
    TORCH_CHECK(local_buffer.is_cuda() && local_flags.is_cuda(), "custom allreduce local tensors must be CUDA");
    TORCH_CHECK(local_buffer.is_contiguous() && local_flags.is_contiguous(), "custom allreduce local tensors must be contiguous");
    TORCH_CHECK(local_flags.scalar_type() == torch::kInt32, "custom allreduce local flags must be int32");
    TORCH_CHECK(local_buffer.numel() == dim, "custom allreduce local buffer dim mismatch");
    TORCH_CHECK(world_size == 4, "custom allreduce currently supports world_size == 4");
    TORCH_CHECK(rank >= 0 && rank < world_size, "custom allreduce rank out of range");
    TORCH_CHECK(buffer_handles.size() == static_cast<size_t>(world_size), "buffer handle count mismatch");
    TORCH_CHECK(flag_handles.size() == static_cast<size_t>(world_size), "flag handle count mismatch");
    TORCH_CHECK(dim > 0, "custom allreduce dim must be positive");
    return custom_allreduce_open_cuda(local_buffer, local_flags, buffer_handles, flag_handles, rank, world_size, dim, dtype_code);
}

void custom_allreduce_close(int64_t handle) {
    custom_allreduce_close_cuda(handle);
}

bool custom_allreduce_inplace(int64_t handle, torch::Tensor& tensor, int64_t seq, int64_t timeout_us) {
    TORCH_CHECK(tensor.is_cuda(), "custom allreduce tensor must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), "custom allreduce tensor must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat16 || tensor.scalar_type() == torch::kBFloat16 || tensor.scalar_type() == torch::kFloat32,
                "custom allreduce tensor must be float16, bfloat16, or float32");
    TORCH_CHECK(seq > 0, "custom allreduce seq must be positive");
    return custom_allreduce_inplace_cuda(handle, tensor, seq, timeout_us);
}

torch::Tensor moe_finalize_reduce_forward(const torch::Tensor& y_reduce, const torch::Tensor& shared, int64_t out_dtype_code) {
    TORCH_CHECK(y_reduce.is_cuda() && shared.is_cuda(), "MoE finalize tensors must be CUDA");
    TORCH_CHECK(y_reduce.is_contiguous() && shared.is_contiguous(), "MoE finalize tensors must be contiguous");
    TORCH_CHECK(y_reduce.sizes() == shared.sizes(), "MoE finalize shape mismatch");
    TORCH_CHECK(y_reduce.scalar_type() == torch::kFloat16 || y_reduce.scalar_type() == torch::kBFloat16 || y_reduce.scalar_type() == torch::kFloat32,
                "y_reduce must be float16, bfloat16, or float32");
    TORCH_CHECK(shared.scalar_type() == torch::kFloat16 || shared.scalar_type() == torch::kBFloat16 || shared.scalar_type() == torch::kFloat32,
                "shared must be float16, bfloat16, or float32");
    TORCH_CHECK(out_dtype_code == 1 || out_dtype_code == 2 || out_dtype_code == 3, "unsupported output dtype code");
    return moe_finalize_reduce_forward_cuda(y_reduce, shared, out_dtype_code);
}

torch::Tensor int8_gemm_forward(
    const torch::Tensor& x,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_s) {
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must have shape [M, K] or [B, S, K]");
    TORCH_CHECK(weight_q.dim() == 2, "weight_q must have shape [N, K]");
    TORCH_CHECK(weight_s.dim() == 1, "weight_s must have shape [N]");
    TORCH_CHECK(weight_q.size(0) == weight_s.size(0), "weight rows mismatch");
    TORCH_CHECK(x.size(-1) == weight_q.size(1), "inner dimension mismatch");
    TORCH_CHECK(x.size(-1) % 4 == 0, "K must be divisible by 4");

    TORCH_CHECK(x.is_cuda() && weight_q.is_cuda() && weight_s.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    check_tensor(weight_q, "weight_q", torch::kInt8);
    check_tensor(weight_s, "weight_s", torch::kFloat32);
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");

    if (int8_gemm_imma_enabled() && (x.size(-1) % 16 == 0) && (weight_q.size(0) % 16 == 0)) {
        return int8_gemm_imma_cuda(x, weight_q, weight_s);
    }
    return int8_gemm_forward_cuda(x, weight_q, weight_s);
}

torch::Tensor q8_0_gemm_forward(
    const torch::Tensor& x,
    const torch::Tensor& blocks,
    int64_t row_elems) {
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must have shape [M, K] or [B, S, K]");
    TORCH_CHECK(blocks.dim() == 3, "blocks must have shape [N, K_blocks, 34]");
    TORCH_CHECK(blocks.size(2) == 34, "q8_0 block size must be 34 bytes");
    TORCH_CHECK(row_elems > 0, "row_elems must be positive");
    TORCH_CHECK(x.size(-1) == row_elems, "inner dimension mismatch");
    TORCH_CHECK(blocks.size(1) * 32 >= row_elems, "blocks do not cover row_elems");
    TORCH_CHECK(x.is_cuda() && blocks.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    check_tensor(blocks, "blocks", torch::kUInt8);
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");
    return q8_0_gemm_forward_cuda(x, blocks, row_elems);
}

torch::Tensor gguf_q2k_gemm_dp4a_forward(
    const torch::Tensor& x,
    const torch::Tensor& blocks,
    int64_t row_elems) {
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must have shape [M, K] or [B, S, K]");
    TORCH_CHECK(blocks.dim() == 3, "blocks must have shape [N, K_blocks, 84]");
    TORCH_CHECK(blocks.size(2) == 84, "q2_k block size must be 84 bytes");
    TORCH_CHECK(row_elems > 0, "row_elems must be positive");
    TORCH_CHECK(x.size(-1) == row_elems, "inner dimension mismatch");
    TORCH_CHECK(blocks.size(1) * 256 >= row_elems, "blocks do not cover row_elems");
    TORCH_CHECK(x.is_cuda() && blocks.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    check_tensor(blocks, "blocks", torch::kUInt8);
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");
    return gguf_q2k_gemm_dp4a_forward_cuda(x, blocks, row_elems);
}

torch::Tensor gguf_quant_gemm_forward(
    const torch::Tensor& x,
    const torch::Tensor& blocks,
    int64_t row_elems,
    int64_t type_id,
    const torch::Tensor& signed_grid) {
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must have shape [M, K] or [B, S, K]");
    TORCH_CHECK(blocks.dim() == 3, "blocks must have shape [N, K_blocks, block_bytes]");
    TORCH_CHECK(type_id == 0 || type_id == 1, "type_id must be 0 (iq2_xxs) or 1 (q2_k)");
    TORCH_CHECK(row_elems > 0, "row_elems must be positive");
    TORCH_CHECK(x.size(-1) == row_elems, "inner dimension mismatch");
    TORCH_CHECK(blocks.size(1) * 256 >= row_elems, "blocks do not cover row_elems");
    TORCH_CHECK(blocks.size(2) == (type_id == 0 ? 66 : 84), "unexpected GGUF block size for type_id");
    TORCH_CHECK(x.is_cuda() && blocks.is_cuda(), "x and blocks must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && blocks.is_contiguous(), "x and blocks must be contiguous");
    check_tensor(blocks, "blocks", torch::kUInt8);
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");
    if (type_id == 0) {
        TORCH_CHECK(signed_grid.is_cuda(), "signed_grid must be CUDA for iq2_xxs");
        TORCH_CHECK(signed_grid.scalar_type() == torch::kInt8, "signed_grid must be int8");
        TORCH_CHECK(signed_grid.is_contiguous(), "signed_grid must be contiguous");
        TORCH_CHECK(signed_grid.numel() == 256 * 128 * 8, "signed_grid must contain 256*128*8 entries");
    }
    return gguf_quant_gemm_forward_cuda(x, blocks, row_elems, type_id, signed_grid);
}

torch::Tensor gguf_quant_gemm_prefill_forward(
    const torch::Tensor& x,
    const torch::Tensor& blocks,
    int64_t row_elems,
    int64_t type_id,
    const torch::Tensor& signed_grid) {
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must have shape [M, K] or [B, S, K]");
    TORCH_CHECK(blocks.dim() == 3, "blocks must have shape [N, K_blocks, block_bytes]");
    TORCH_CHECK(type_id == 0 || type_id == 1, "type_id must be 0 (iq2_xxs) or 1 (q2_k)");
    TORCH_CHECK(row_elems > 0, "row_elems must be positive");
    TORCH_CHECK(x.size(-1) == row_elems, "inner dimension mismatch");
    TORCH_CHECK(blocks.size(1) * 256 >= row_elems, "blocks do not cover row_elems");
    TORCH_CHECK(blocks.size(2) == (type_id == 0 ? 66 : 84), "unexpected GGUF block size for type_id");
    TORCH_CHECK(x.is_cuda() && blocks.is_cuda(), "x and blocks must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && blocks.is_contiguous(), "x and blocks must be contiguous");
    check_tensor(blocks, "blocks", torch::kUInt8);
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");
    if (type_id == 0) {
        TORCH_CHECK(signed_grid.is_cuda(), "signed_grid must be CUDA for iq2_xxs");
        TORCH_CHECK(signed_grid.scalar_type() == torch::kInt8, "signed_grid must be int8");
        TORCH_CHECK(signed_grid.is_contiguous(), "signed_grid must be contiguous");
        TORCH_CHECK(signed_grid.numel() == 256 * 128 * 8, "signed_grid must contain 256*128*8 entries");
    }
    return gguf_quant_gemm_prefill_forward_cuda(x, blocks, row_elems, type_id, signed_grid);
}
torch::Tensor gguf_quant_gemm_pair_forward(
    const torch::Tensor& x,
    const torch::Tensor& blocks0,
    int64_t row_elems0,
    int64_t type_id0,
    const torch::Tensor& blocks1,
    int64_t row_elems1,
    int64_t type_id1,
    const torch::Tensor& signed_grid) {
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must have shape [M, K] or [B, S, K]");
    TORCH_CHECK(blocks0.dim() == 3 && blocks1.dim() == 3, "blocks must have shape [N, K_blocks, block_bytes]");
    TORCH_CHECK(type_id0 == 0 || type_id0 == 1, "type_id0 must be 0 or 1");
    TORCH_CHECK(type_id1 == 0 || type_id1 == 1, "type_id1 must be 0 or 1");
    TORCH_CHECK(row_elems0 > 0 && row_elems1 > 0, "row_elems must be positive");
    TORCH_CHECK(row_elems0 == row_elems1, "paired GGUF GEMM requires matching input dimensions");
    TORCH_CHECK(x.size(-1) == row_elems0, "inner dimension mismatch");
    TORCH_CHECK(blocks0.size(0) == blocks1.size(0), "paired blocks must have same output rows");
    TORCH_CHECK(blocks0.size(1) * 256 >= row_elems0 && blocks1.size(1) * 256 >= row_elems1, "blocks do not cover row_elems");
    TORCH_CHECK(blocks0.size(2) == (type_id0 == 0 ? 66 : 84), "unexpected blocks0 GGUF block size");
    TORCH_CHECK(blocks1.size(2) == (type_id1 == 0 ? 66 : 84), "unexpected blocks1 GGUF block size");
    TORCH_CHECK(x.is_cuda() && blocks0.is_cuda() && blocks1.is_cuda(), "x and blocks must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && blocks0.is_contiguous() && blocks1.is_contiguous(), "x and blocks must be contiguous");
    check_tensor(blocks0, "blocks0", torch::kUInt8);
    check_tensor(blocks1, "blocks1", torch::kUInt8);
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");
    if (type_id0 == 0 || type_id1 == 0) {
        TORCH_CHECK(signed_grid.is_cuda(), "signed_grid must be CUDA for iq2_xxs");
        TORCH_CHECK(signed_grid.scalar_type() == torch::kInt8, "signed_grid must be int8");
        TORCH_CHECK(signed_grid.is_contiguous(), "signed_grid must be contiguous");
        TORCH_CHECK(signed_grid.numel() == 256 * 128 * 8, "signed_grid must contain 256*128*8 entries");
    }
    return gguf_quant_gemm_pair_forward_cuda(
        x,
        blocks0,
        row_elems0,
        type_id0,
        blocks1,
        row_elems1,
        type_id1,
        signed_grid);
}

torch::Tensor gguf_moe_prefill_grouped_forward(
    const torch::Tensor& x,
    const torch::Tensor& route_tokens,
    const torch::Tensor& route_weights_sorted,
    const torch::Tensor& seg_starts,
    const torch::Tensor& w1_blocks,
    const torch::Tensor& w3_blocks,
    const torch::Tensor& w2_blocks,
    int64_t w1_row_elems,
    int64_t w1_type_id,
    int64_t w3_row_elems,
    int64_t w3_type_id,
    int64_t w2_row_elems,
    int64_t w2_type_id,
    const torch::Tensor& signed_grid,
    double swiglu_limit) {
    TORCH_CHECK(x.dim() == 2, "x must have shape [T, D]");
    TORCH_CHECK(route_tokens.dim() == 1, "route_tokens must have shape [R]");
    TORCH_CHECK(route_weights_sorted.dim() == 1, "route_weights_sorted must have shape [R]");
    TORCH_CHECK(seg_starts.dim() == 1, "seg_starts must have shape [E + 1]");
    TORCH_CHECK(w1_blocks.dim() == 4 && w3_blocks.dim() == 4 && w2_blocks.dim() == 4, "GGUF blocks must have shape [E, N, K_blocks, block_bytes]");
    TORCH_CHECK(route_tokens.size(0) == route_weights_sorted.size(0), "route count mismatch");
    TORCH_CHECK(seg_starts.size(0) == w1_blocks.size(0) + 1, "seg_starts/expert count mismatch");
    TORCH_CHECK(w1_blocks.size(0) == w3_blocks.size(0) && w1_blocks.size(0) == w2_blocks.size(0), "expert count mismatch");
    TORCH_CHECK(w1_blocks.size(2) * 256 >= w1_row_elems && w3_blocks.size(2) * 256 >= w3_row_elems && w2_blocks.size(2) * 256 >= w2_row_elems, "blocks do not cover row_elems");
    TORCH_CHECK(w1_type_id == 0 || w1_type_id == 1, "w1_type_id must be 0 or 1");
    TORCH_CHECK(w3_type_id == 0 || w3_type_id == 1, "w3_type_id must be 0 or 1");
    TORCH_CHECK(w2_type_id == 0 || w2_type_id == 1, "w2_type_id must be 0 or 1");
    TORCH_CHECK(w1_blocks.size(3) == (w1_type_id == 0 ? 66 : 84), "unexpected w1 GGUF block size");
    TORCH_CHECK(w3_blocks.size(3) == (w3_type_id == 0 ? 66 : 84), "unexpected w3 GGUF block size");
    TORCH_CHECK(w2_blocks.size(3) == (w2_type_id == 0 ? 66 : 84), "unexpected w2 GGUF block size");
    TORCH_CHECK(w1_row_elems == x.size(1) && w3_row_elems == x.size(1), "w1/w3 input dimension mismatch");
    TORCH_CHECK(w2_row_elems == w1_blocks.size(1) && w1_blocks.size(1) == w3_blocks.size(1), "w2 input dimension mismatch");
    TORCH_CHECK(w2_blocks.size(1) == x.size(1), "w2 output dimension mismatch");
    TORCH_CHECK(x.is_cuda() && route_tokens.is_cuda() && route_weights_sorted.is_cuda() && seg_starts.is_cuda(), "route tensors must be CUDA tensors");
    TORCH_CHECK(w1_blocks.is_cuda() && w3_blocks.is_cuda() && w2_blocks.is_cuda(), "GGUF blocks must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && route_tokens.is_contiguous() && route_weights_sorted.is_contiguous() && seg_starts.is_contiguous(), "route tensors must be contiguous");
    TORCH_CHECK(w1_blocks.is_contiguous() && w3_blocks.is_contiguous() && w2_blocks.is_contiguous(), "GGUF blocks must be contiguous");
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");
    TORCH_CHECK(route_tokens.scalar_type() == torch::kInt64, "route_tokens must be int64");
    TORCH_CHECK(route_weights_sorted.scalar_type() == torch::kFloat32, "route_weights_sorted must be float32");
    TORCH_CHECK(seg_starts.scalar_type() == torch::kInt32 || seg_starts.scalar_type() == torch::kInt64, "seg_starts must be int32 or int64");
    check_tensor(w1_blocks, "w1_blocks", torch::kUInt8);
    check_tensor(w3_blocks, "w3_blocks", torch::kUInt8);
    check_tensor(w2_blocks, "w2_blocks", torch::kUInt8);
    if (w1_type_id == 0 || w3_type_id == 0 || w2_type_id == 0) {
        TORCH_CHECK(signed_grid.is_cuda(), "signed_grid must be CUDA when iq2_xxs blocks are used");
        TORCH_CHECK(signed_grid.scalar_type() == torch::kInt8, "signed_grid must be int8");
        TORCH_CHECK(signed_grid.is_contiguous(), "signed_grid must be contiguous");
        TORCH_CHECK(signed_grid.numel() == 256 * 128 * 8, "signed_grid must contain 256*128*8 entries");
    }
    return gguf_moe_prefill_grouped_forward_cuda(
        x,
        route_tokens,
        route_weights_sorted,
        seg_starts,
        w1_blocks,
        w3_blocks,
        w2_blocks,
        w1_row_elems,
        w1_type_id,
        w3_row_elems,
        w3_type_id,
        w2_row_elems,
        w2_type_id,
        signed_grid,
        swiglu_limit);
}

torch::Tensor gguf_moe_single_token_iq2_q2k_forward(
    const torch::Tensor& x,
    const torch::Tensor& route_slots,
    const torch::Tensor& route_weights,
    const torch::Tensor& w1_blocks,
    const torch::Tensor& w3_blocks,
    const torch::Tensor& w2_blocks,
    const torch::Tensor& signed_grid,
    double swiglu_limit) {
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "x must have shape [1, D]");
    TORCH_CHECK(route_slots.dim() == 1, "route_slots must have shape [K]");
    TORCH_CHECK(route_weights.dim() == 1, "route_weights must have shape [K]");
    TORCH_CHECK(route_slots.size(0) == route_weights.size(0), "route_slots/route_weights length mismatch");
    TORCH_CHECK(w1_blocks.dim() == 4 && w3_blocks.dim() == 4 && w2_blocks.dim() == 4, "GGUF blocks must have shape [E, N, K_blocks, block_bytes]");
    TORCH_CHECK(w1_blocks.size(0) == w3_blocks.size(0) && w1_blocks.size(0) == w2_blocks.size(0), "expert count mismatch");
    TORCH_CHECK(w1_blocks.size(1) == w3_blocks.size(1), "w1/w3 inter_dim mismatch");
    TORCH_CHECK(w1_blocks.size(2) == w3_blocks.size(2), "w1/w3 block count mismatch");
    TORCH_CHECK(w1_blocks.size(3) == 66 && w3_blocks.size(3) == 66, "w1/w3 must be IQ2_XXS blocks");
    TORCH_CHECK(w2_blocks.size(3) == 84, "w2 must be Q2_K blocks");
    TORCH_CHECK(w1_blocks.size(2) * 256 >= x.size(1), "w1/w3 blocks do not cover x dim");
    TORCH_CHECK(w2_blocks.size(1) == x.size(1), "w2 output dimension mismatch");
    TORCH_CHECK(w2_blocks.size(2) * 256 >= w1_blocks.size(1), "w2 blocks do not cover inter dim");
    TORCH_CHECK(x.is_cuda() && route_slots.is_cuda() && route_weights.is_cuda(), "x/route tensors must be CUDA tensors");
    TORCH_CHECK(w1_blocks.is_cuda() && w3_blocks.is_cuda() && w2_blocks.is_cuda(), "GGUF blocks must be CUDA tensors");
    TORCH_CHECK(signed_grid.is_cuda(), "signed_grid must be CUDA for IQ2_XXS");
    TORCH_CHECK(x.is_contiguous() && route_slots.is_contiguous() && route_weights.is_contiguous(), "x/route tensors must be contiguous");
    TORCH_CHECK(w1_blocks.is_contiguous() && w3_blocks.is_contiguous() && w2_blocks.is_contiguous(), "GGUF blocks must be contiguous");
    TORCH_CHECK(signed_grid.is_contiguous(), "signed_grid must be contiguous");
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");
    TORCH_CHECK(route_slots.scalar_type() == torch::kInt64, "route_slots must be int64");
    TORCH_CHECK(route_weights.scalar_type() == torch::kFloat32, "route_weights must be float32");
    check_tensor(w1_blocks, "w1_blocks", torch::kUInt8);
    check_tensor(w3_blocks, "w3_blocks", torch::kUInt8);
    check_tensor(w2_blocks, "w2_blocks", torch::kUInt8);
    check_tensor(signed_grid, "signed_grid", torch::kInt8);
    TORCH_CHECK(signed_grid.numel() == 256 * 128 * 8, "signed_grid must contain 256*128*8 entries");
    return gguf_moe_single_token_iq2_q2k_forward_cuda(
        x,
        route_slots,
        route_weights,
        w1_blocks,
        w3_blocks,
        w2_blocks,
        signed_grid,
        swiglu_limit);
}

torch::Tensor int8_gemm_pair_forward(
    const torch::Tensor& x,
    const torch::Tensor& weight_q0,
    const torch::Tensor& weight_s0,
    const torch::Tensor& weight_q1,
    const torch::Tensor& weight_s1) {
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must have shape [M, K] or [B, S, K]");
    TORCH_CHECK(weight_q0.dim() == 2 && weight_q1.dim() == 2, "weights must have shape [N, K]");
    TORCH_CHECK(weight_s0.dim() == 1 && weight_s1.dim() == 1, "scales must have shape [N]");
    TORCH_CHECK(weight_q0.size(1) == x.size(-1) && weight_q1.size(1) == x.size(-1), "inner dimension mismatch");
    TORCH_CHECK(weight_q0.size(0) == weight_s0.size(0) && weight_q1.size(0) == weight_s1.size(0), "weight rows mismatch");
    TORCH_CHECK(weight_q0.size(0) == weight_q1.size(0), "paired weights must have same output size");
    TORCH_CHECK(x.is_cuda() && weight_q0.is_cuda() && weight_s0.is_cuda() && weight_q1.is_cuda() && weight_s1.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    check_tensor(weight_q0, "weight_q0", torch::kInt8);
    check_tensor(weight_q1, "weight_q1", torch::kInt8);
    check_tensor(weight_s0, "weight_s0", torch::kFloat32);
    check_tensor(weight_s1, "weight_s1", torch::kFloat32);
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");
    if (int8_gemm_imma_enabled() && (x.size(-1) % 16 == 0) && (weight_q0.size(0) % 16 == 0)) {
        return int8_gemm_pair_imma_cuda(x, weight_q0, weight_s0, weight_q1, weight_s1);
    }
    return int8_gemm_pair_forward_cuda(x, weight_q0, weight_s0, weight_q1, weight_s1);
}


torch::Tensor wo_a_int8_forward(
    const torch::Tensor& x,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_s) {
    TORCH_CHECK(x.dim() == 4, "x must have shape [B, S, G, D]");
    TORCH_CHECK(weight_q.dim() == 3, "weight_q must have shape [G, R, D]");
    TORCH_CHECK(weight_s.dim() == 2, "weight_s must have shape [G, R]");
    TORCH_CHECK(x.size(2) == weight_q.size(0), "group dimension mismatch");
    TORCH_CHECK(weight_q.size(0) == weight_s.size(0), "weight group mismatch");
    TORCH_CHECK(weight_q.size(1) == weight_s.size(1), "weight rank mismatch");
    TORCH_CHECK(x.size(3) == weight_q.size(2), "inner dimension mismatch");
    TORCH_CHECK(x.size(3) % 4 == 0, "D must be divisible by 4");

    TORCH_CHECK(x.is_cuda() && weight_q.is_cuda() && weight_s.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    check_tensor(weight_q, "weight_q", torch::kInt8);
    check_tensor(weight_s, "weight_s", torch::kFloat32);
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");

    return wo_a_int8_forward_cuda(x, weight_q, weight_s);
}


torch::Tensor fused_decode_sparse_attn_forward(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& attn_sink,
    const torch::Tensor& topk_idxs,
    double softmax_scale) {
    TORCH_CHECK(q.dim() == 4, "q must have shape [B, 1, H, D]");
    TORCH_CHECK(q.size(1) == 1, "only decode seqlen=1 is supported");
    TORCH_CHECK(kv.dim() == 3, "kv must have shape [B, T, D]");
    TORCH_CHECK(attn_sink.dim() == 1, "attn_sink must have shape [H]");
    TORCH_CHECK(topk_idxs.dim() == 3, "topk_idxs must have shape [B, 1, K]");
    TORCH_CHECK(topk_idxs.size(1) == 1, "only decode seqlen=1 topk is supported");
    TORCH_CHECK(q.size(0) == kv.size(0), "batch mismatch");
    TORCH_CHECK(q.size(0) == topk_idxs.size(0), "topk batch mismatch");
    TORCH_CHECK(q.size(2) == attn_sink.size(0), "head mismatch");
    TORCH_CHECK(q.size(3) == kv.size(2), "head dim mismatch");
    TORCH_CHECK(q.is_cuda() && kv.is_cuda() && attn_sink.is_cuda() && topk_idxs.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(kv.is_contiguous(), "kv must be contiguous");
    TORCH_CHECK(attn_sink.is_contiguous(), "attn_sink must be contiguous");
    TORCH_CHECK(topk_idxs.is_contiguous(), "topk_idxs must be contiguous");
    TORCH_CHECK(topk_idxs.scalar_type() == torch::kInt32, "topk_idxs must be int32");
    TORCH_CHECK(attn_sink.scalar_type() == torch::kFloat32, "attn_sink must be float32");
    TORCH_CHECK(
        q.scalar_type() == torch::kFloat16 ||
        q.scalar_type() == torch::kBFloat16 ||
        q.scalar_type() == torch::kFloat32,
        "q must be float16, bfloat16, or float32");
    TORCH_CHECK(kv.scalar_type() == q.scalar_type(), "kv dtype must match q dtype");
    return fused_decode_sparse_attn_forward_cuda(q, kv, attn_sink, topk_idxs, softmax_scale);
}

torch::Tensor fused_decode_sparse_attn_wmma_forward(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& attn_sink,
    const torch::Tensor& topk_idxs,
    double softmax_scale) {
    TORCH_CHECK(q.dim() == 4, "q must have shape [B, 1, H, D]");
    TORCH_CHECK(q.size(1) == 1, "only decode seqlen=1 is supported");
    TORCH_CHECK(kv.dim() == 3, "kv must have shape [B, T, D]");
    TORCH_CHECK(attn_sink.dim() == 1, "attn_sink must have shape [H]");
    TORCH_CHECK(topk_idxs.dim() == 3, "topk_idxs must have shape [B, 1, K]");
    TORCH_CHECK(q.is_cuda() && kv.is_cuda() && attn_sink.is_cuda() && topk_idxs.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16, "q must be float16");
    TORCH_CHECK(kv.scalar_type() == torch::kFloat16, "kv must be float16");
    TORCH_CHECK(topk_idxs.scalar_type() == torch::kInt32, "topk_idxs must be int32");
    TORCH_CHECK(attn_sink.scalar_type() == torch::kFloat32, "attn_sink must be float32");
    return fused_decode_sparse_attn_wmma_forward_cuda(q, kv, attn_sink, topk_idxs, softmax_scale);
}

torch::Tensor flashinfer_style_sparse_attn_forward(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& attn_sink,
    const torch::Tensor& topk_idxs,
    double softmax_scale) {
    TORCH_CHECK(q.dim() == 4, "q must have shape [B, 1, H, D]");
    TORCH_CHECK(q.size(1) == 1, "only decode seqlen=1 is supported");
    TORCH_CHECK(kv.dim() == 3, "kv must have shape [B, T, D]");
    TORCH_CHECK(attn_sink.dim() == 1, "attn_sink must have shape [H]");
    TORCH_CHECK(topk_idxs.dim() == 3, "topk_idxs must have shape [B, 1, K]");
    TORCH_CHECK(q.is_cuda() && kv.is_cuda() && attn_sink.is_cuda() && topk_idxs.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(kv.is_contiguous(), "kv must be contiguous");
    TORCH_CHECK(attn_sink.is_contiguous(), "attn_sink must be contiguous");
    TORCH_CHECK(topk_idxs.is_contiguous(), "topk_idxs must be contiguous");
    TORCH_CHECK(topk_idxs.scalar_type() == torch::kInt32, "topk_idxs must be int32");
    TORCH_CHECK(attn_sink.scalar_type() == torch::kFloat32, "attn_sink must be float32");
    TORCH_CHECK(
        q.scalar_type() == torch::kFloat16 ||
        q.scalar_type() == torch::kBFloat16 ||
        q.scalar_type() == torch::kFloat32,
        "q must be float16, bfloat16, or float32");
    TORCH_CHECK(kv.scalar_type() == q.scalar_type(), "kv dtype must match q dtype");
    return flashinfer_style_sparse_attn_forward_cuda(q, kv, attn_sink, topk_idxs, softmax_scale);
}


torch::Tensor flashinfer_style_sparse_attn_headpair_forward(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& attn_sink,
    const torch::Tensor& topk_idxs,
    double softmax_scale) {
    TORCH_CHECK(q.dim() == 4, "q must have shape [B, 1, H, D]");
    TORCH_CHECK(q.size(1) == 1, "only decode seqlen=1 is supported");
    TORCH_CHECK(q.size(3) == 512, "headpair decode sparse attention requires D=512");
    TORCH_CHECK(kv.dim() == 3, "kv must have shape [B, T, D]");
    TORCH_CHECK(attn_sink.dim() == 1, "attn_sink must have shape [H]");
    TORCH_CHECK(topk_idxs.dim() == 3, "topk_idxs must have shape [B, 1, K]");
    TORCH_CHECK(topk_idxs.size(1) == 1, "only decode seqlen=1 topk is supported");
    TORCH_CHECK(topk_idxs.size(2) <= 256, "headpair decode sparse attention supports topk <= 256");
    TORCH_CHECK(q.size(0) == kv.size(0), "batch mismatch");
    TORCH_CHECK(q.size(0) == topk_idxs.size(0), "topk batch mismatch");
    TORCH_CHECK(q.size(2) == attn_sink.size(0), "head mismatch");
    TORCH_CHECK(q.size(3) == kv.size(2), "head dim mismatch");
    TORCH_CHECK(q.is_cuda() && kv.is_cuda() && attn_sink.is_cuda() && topk_idxs.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(kv.is_contiguous(), "kv must be contiguous");
    TORCH_CHECK(attn_sink.is_contiguous(), "attn_sink must be contiguous");
    TORCH_CHECK(topk_idxs.is_contiguous(), "topk_idxs must be contiguous");
    TORCH_CHECK(topk_idxs.scalar_type() == torch::kInt32, "topk_idxs must be int32");
    TORCH_CHECK(attn_sink.scalar_type() == torch::kFloat32, "attn_sink must be float32");
    TORCH_CHECK(
        q.scalar_type() == torch::kFloat16 ||
        q.scalar_type() == torch::kBFloat16 ||
        q.scalar_type() == torch::kFloat32,
        "q must be float16, bfloat16, or float32");
    TORCH_CHECK(kv.scalar_type() == q.scalar_type(), "kv dtype must match q dtype");
    return flashinfer_style_sparse_attn_headpair_forward_cuda(q, kv, attn_sink, topk_idxs, softmax_scale);
}


torch::Tensor prefill_sparse_attn_forward(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& attn_sink,
    const torch::Tensor& topk_idxs,
    double softmax_scale) {
    TORCH_CHECK(q.dim() == 4, "q must have shape [B, S, H, D]");
    TORCH_CHECK(kv.dim() == 3, "kv must have shape [B, T, D]");
    TORCH_CHECK(attn_sink.dim() == 1, "attn_sink must have shape [H]");
    TORCH_CHECK(topk_idxs.dim() == 3, "topk_idxs must have shape [B, S, K]");
    TORCH_CHECK(q.size(0) == kv.size(0), "batch mismatch");
    TORCH_CHECK(q.size(0) == topk_idxs.size(0), "topk batch mismatch");
    TORCH_CHECK(q.size(1) == topk_idxs.size(1), "sequence mismatch");
    TORCH_CHECK(q.size(2) == attn_sink.size(0), "head mismatch");
    TORCH_CHECK(q.size(3) == kv.size(2), "head dim mismatch");
    TORCH_CHECK(q.is_cuda() && kv.is_cuda() && attn_sink.is_cuda() && topk_idxs.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(kv.is_contiguous(), "kv must be contiguous");
    TORCH_CHECK(attn_sink.is_contiguous(), "attn_sink must be contiguous");
    TORCH_CHECK(topk_idxs.is_contiguous(), "topk_idxs must be contiguous");
    TORCH_CHECK(topk_idxs.scalar_type() == torch::kInt32, "topk_idxs must be int32");
    TORCH_CHECK(attn_sink.scalar_type() == torch::kFloat32, "attn_sink must be float32");
    TORCH_CHECK(
        q.scalar_type() == torch::kFloat16 ||
        q.scalar_type() == torch::kBFloat16 ||
        q.scalar_type() == torch::kFloat32,
        "q must be float16, bfloat16, or float32");
    TORCH_CHECK(kv.scalar_type() == q.scalar_type(), "kv dtype must match q dtype");
    return prefill_sparse_attn_forward_cuda(q, kv, attn_sink, topk_idxs, softmax_scale);
}


torch::Tensor prefill_sparse_attn_headpair_forward(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& attn_sink,
    const torch::Tensor& topk_idxs,
    double softmax_scale) {
    TORCH_CHECK(q.dim() == 4, "q must have shape [B, S, H, D]");
    TORCH_CHECK(kv.dim() == 3, "kv must have shape [B, T, D]");
    TORCH_CHECK(attn_sink.dim() == 1, "attn_sink must have shape [H]");
    TORCH_CHECK(topk_idxs.dim() == 3, "topk_idxs must have shape [B, S, K]");
    TORCH_CHECK(q.size(0) == kv.size(0), "batch mismatch");
    TORCH_CHECK(q.size(0) == topk_idxs.size(0), "topk batch mismatch");
    TORCH_CHECK(q.size(1) == topk_idxs.size(1), "sequence mismatch");
    TORCH_CHECK(q.size(2) == attn_sink.size(0), "head mismatch");
    TORCH_CHECK(q.size(3) == kv.size(2), "head dim mismatch");
    TORCH_CHECK(q.is_cuda() && kv.is_cuda() && attn_sink.is_cuda() && topk_idxs.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(kv.is_contiguous(), "kv must be contiguous");
    TORCH_CHECK(attn_sink.is_contiguous(), "attn_sink must be contiguous");
    TORCH_CHECK(topk_idxs.is_contiguous(), "topk_idxs must be contiguous");
    TORCH_CHECK(topk_idxs.scalar_type() == torch::kInt32, "topk_idxs must be int32");
    TORCH_CHECK(attn_sink.scalar_type() == torch::kFloat32, "attn_sink must be float32");
    TORCH_CHECK(
        q.scalar_type() == torch::kFloat16 ||
        q.scalar_type() == torch::kBFloat16 ||
        q.scalar_type() == torch::kFloat32,
        "q must be float16, bfloat16, or float32");
    TORCH_CHECK(kv.scalar_type() == q.scalar_type(), "kv dtype must match q dtype");
    return prefill_sparse_attn_headpair_forward_cuda(q, kv, attn_sink, topk_idxs, softmax_scale);
}


torch::Tensor fused_c4_indexer_decode_forward(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& score,
    const torch::Tensor& weights,
    const torch::Tensor& ape,
    const torch::Tensor& norm_weight,
    const torch::Tensor& freqs,
    const torch::Tensor& kv_state,
    const torch::Tensor& score_state,
    const torch::Tensor& kv_cache,
    int64_t start_pos,
    int64_t offset,
    int64_t index_topk,
    double norm_eps,
    bool return_scores) {
    TORCH_CHECK(q.dim() == 4, "q must have shape [B, 1, H, 128]");
    TORCH_CHECK(q.size(1) == 1, "only decode seqlen=1 is supported");
    TORCH_CHECK(q.size(3) == 128, "q head_dim must be 128");
    TORCH_CHECK(kv.dim() == 3 && kv.size(1) == 1 && kv.size(2) == 256, "kv must have shape [B, 1, 256]");
    TORCH_CHECK(score.dim() == 3 && score.size(1) == 1 && score.size(2) == 256, "score must have shape [B, 1, 256]");
    TORCH_CHECK(weights.dim() == 3 && weights.size(1) == 1 && weights.size(2) == q.size(2), "weights must have shape [B, 1, H]");
    TORCH_CHECK(ape.dim() == 2 && ape.size(0) == 4 && ape.size(1) == 256, "ape must have shape [4, 256]");
    TORCH_CHECK(norm_weight.dim() == 1 && norm_weight.size(0) == 128, "norm_weight must have shape [128]");
    TORCH_CHECK(freqs.numel() == 64, "freqs must contain 64 complex-real values");
    TORCH_CHECK(kv_state.dim() == 3 && kv_state.size(1) == 8 && kv_state.size(2) == 256, "kv_state must have shape [Bmax, 8, 256]");
    TORCH_CHECK(score_state.sizes() == kv_state.sizes(), "score_state shape must match kv_state");
    TORCH_CHECK(kv_cache.dim() == 3 && kv_cache.size(2) == 128, "kv_cache must have shape [Bmax, T, 128]");
    TORCH_CHECK(q.size(0) == kv.size(0) && q.size(0) == score.size(0) && q.size(0) == weights.size(0), "batch mismatch");
    TORCH_CHECK(q.size(0) <= kv_state.size(0) && q.size(0) <= kv_cache.size(0), "state/cache batch too small");
    TORCH_CHECK(q.is_cuda() && kv.is_cuda() && score.is_cuda() && weights.is_cuda() && ape.is_cuda() && norm_weight.is_cuda() && freqs.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(kv_state.is_cuda() && score_state.is_cuda() && kv_cache.is_cuda(), "state/cache must be CUDA tensors");
    TORCH_CHECK(kv.scalar_type() == torch::kFloat32 && score.scalar_type() == torch::kFloat32, "kv and score must be float32");
    TORCH_CHECK(ape.scalar_type() == torch::kFloat32 && norm_weight.scalar_type() == torch::kFloat32, "ape and norm_weight must be float32");
    TORCH_CHECK(kv_state.scalar_type() == torch::kFloat32 && score_state.scalar_type() == torch::kFloat32, "states must be float32");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(kv.is_contiguous() && score.is_contiguous() && weights.is_contiguous(), "kv/score/weights must be contiguous");
    TORCH_CHECK(ape.is_contiguous() && norm_weight.is_contiguous() && freqs.is_contiguous(), "ape/norm/freqs must be contiguous");
    TORCH_CHECK(kv_state.is_contiguous() && score_state.is_contiguous() && kv_cache.is_contiguous(), "state/cache must be contiguous");
    return fused_c4_indexer_decode_forward_cuda(
        q,
        kv,
        score,
        weights,
        ape,
        norm_weight,
        freqs,
        kv_state,
        score_state,
        kv_cache,
        start_pos,
        offset,
        index_topk,
        norm_eps,
        return_scores);
}

torch::Tensor c4_topk_from_scores(
    const torch::Tensor& scores,
    int64_t offset,
    int64_t index_topk) {
    TORCH_CHECK(scores.dim() == 3 && scores.size(1) == 1, "scores must have shape [B, 1, T]");
    TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
    TORCH_CHECK(
        scores.scalar_type() == torch::kFloat32 || scores.scalar_type() == torch::kBFloat16 || scores.scalar_type() == torch::kFloat16,
        "scores must be float32, bfloat16, or float16");
    TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
    TORCH_CHECK(index_topk >= 0, "index_topk must be non-negative");
    return c4_topk_from_scores_cuda(scores, offset, index_topk);
}

std::vector<torch::Tensor> hc_split_pre_forward(
    const torch::Tensor& mixes,
    const torch::Tensor& x,
    const torch::Tensor& hc_scale,
    const torch::Tensor& hc_base,
    int64_t hc_mult,
    int64_t sinkhorn_iters,
    double eps) {
    TORCH_CHECK(mixes.dim() == 3, "mixes must have shape [B, S, mix_hc]");
    TORCH_CHECK(x.dim() == 4, "x must have shape [B, S, HC, D]");
    TORCH_CHECK(hc_scale.dim() == 1 && hc_scale.size(0) == 3, "hc_scale must have shape [3]");
    TORCH_CHECK(hc_base.dim() == 1, "hc_base must be 1D");
    TORCH_CHECK(hc_mult == 4, "hc_split_pre_forward currently supports hc_mult=4");
    TORCH_CHECK(mixes.size(0) == x.size(0) && mixes.size(1) == x.size(1), "batch/sequence mismatch");
    TORCH_CHECK(x.size(2) == hc_mult, "x HC dimension mismatch");
    TORCH_CHECK(mixes.size(2) == (2 + hc_mult) * hc_mult, "mix_hc mismatch");
    TORCH_CHECK(hc_base.size(0) == (2 + hc_mult) * hc_mult, "hc_base size mismatch");
    TORCH_CHECK(mixes.is_cuda() && x.is_cuda() && hc_scale.is_cuda() && hc_base.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(mixes.is_contiguous() && x.is_contiguous() && hc_scale.is_contiguous() && hc_base.is_contiguous(), "all tensors must be contiguous");
    TORCH_CHECK(mixes.scalar_type() == torch::kFloat32, "mixes must be float32");
    TORCH_CHECK(hc_scale.scalar_type() == torch::kFloat32 && hc_base.scalar_type() == torch::kFloat32, "HC parameters must be float32");
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");
    return hc_split_pre_forward_cuda(mixes, x, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps);
}

torch::Tensor hc_post_forward(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb) {
    TORCH_CHECK(x.dim() == 3, "x must have shape [B, S, D]");
    TORCH_CHECK(residual.dim() == 4, "residual must have shape [B, S, HC, D]");
    TORCH_CHECK(post.dim() == 3, "post must have shape [B, S, HC]");
    TORCH_CHECK(comb.dim() == 4, "comb must have shape [B, S, HC, HC]");
    TORCH_CHECK(residual.size(2) == 4, "hc_post_forward currently supports HC=4");
    TORCH_CHECK(x.size(0) == residual.size(0) && x.size(1) == residual.size(1) && x.size(2) == residual.size(3), "x/residual shape mismatch");
    TORCH_CHECK(post.size(0) == x.size(0) && post.size(1) == x.size(1) && post.size(2) == residual.size(2), "post shape mismatch");
    TORCH_CHECK(comb.size(0) == x.size(0) && comb.size(1) == x.size(1) && comb.size(2) == residual.size(2) && comb.size(3) == residual.size(2), "comb shape mismatch");
    TORCH_CHECK(x.is_cuda() && residual.is_cuda() && post.is_cuda() && comb.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && residual.is_contiguous() && post.is_contiguous() && comb.is_contiguous(), "all tensors must be contiguous");
    TORCH_CHECK(x.scalar_type() == residual.scalar_type(), "x and residual dtype must match");
    TORCH_CHECK(post.scalar_type() == torch::kFloat32 && comb.scalar_type() == torch::kFloat32, "post and comb must be float32");
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");
    return hc_post_forward_cuda(x, residual, post, comb);
}


std::vector<torch::Tensor> moe_group_routes(
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    int64_t experts_start_idx,
    int64_t n_local_experts) {
    TORCH_CHECK(indices.dim() == 2, "indices must have shape [T, K]");
    TORCH_CHECK(weights.dim() == 2, "weights must have shape [T, K]");
    TORCH_CHECK(indices.size(0) == weights.size(0) && indices.size(1) == weights.size(1), "indices/weights shape mismatch");
    TORCH_CHECK(indices.is_cuda() && weights.is_cuda(), "indices and weights must be CUDA tensors");
    TORCH_CHECK(indices.is_contiguous() && weights.is_contiguous(), "indices and weights must be contiguous");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(indices.scalar_type() == torch::kInt64 || indices.scalar_type() == torch::kInt32, "indices must be int64 or int32");
    TORCH_CHECK(n_local_experts > 0, "n_local_experts must be positive");
    return moe_group_routes_cuda(indices, weights, experts_start_idx, n_local_experts);
}

std::vector<torch::Tensor> gguf_single_token_route_slots(
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    int64_t experts_start_idx,
    int64_t n_local_experts) {
    TORCH_CHECK(indices.dim() == 2 && indices.size(0) == 1, "indices must have shape [1, K]");
    TORCH_CHECK(weights.dim() == 2 && weights.size(0) == 1, "weights must have shape [1, K]");
    TORCH_CHECK(indices.size(1) == weights.size(1), "indices/weights shape mismatch");
    TORCH_CHECK(indices.is_cuda() && weights.is_cuda(), "indices and weights must be CUDA tensors");
    TORCH_CHECK(indices.is_contiguous() && weights.is_contiguous(), "indices and weights must be contiguous");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(indices.scalar_type() == torch::kInt64 || indices.scalar_type() == torch::kInt32, "indices must be int64 or int32");
    TORCH_CHECK(n_local_experts > 0, "n_local_experts must be positive");

    const int64_t topk = indices.size(1);
    const int64_t experts = n_local_experts;
    auto idx_cpu = indices.to(torch::kCPU);
    std::vector<std::vector<int64_t>> routes_by_local(static_cast<size_t>(experts));
    if (indices.scalar_type() == torch::kInt64) {
        const auto* idx = idx_cpu.data_ptr<int64_t>();
        for (int64_t route_pos = 0; route_pos < topk; ++route_pos) {
            const int64_t local = idx[route_pos] - experts_start_idx;
            if (0 <= local && local < experts) {
                routes_by_local[static_cast<size_t>(local)].push_back(route_pos);
            }
        }
    } else {
        const auto* idx = idx_cpu.data_ptr<int32_t>();
        for (int64_t route_pos = 0; route_pos < topk; ++route_pos) {
            const int64_t local = static_cast<int64_t>(idx[route_pos]) - experts_start_idx;
            if (0 <= local && local < experts) {
                routes_by_local[static_cast<size_t>(local)].push_back(route_pos);
            }
        }
    }

    int64_t active = 0;
    int64_t routes = 0;
    for (const auto& positions : routes_by_local) {
        if (!positions.empty()) {
            ++active;
            routes += static_cast<int64_t>(positions.size());
        }
    }

    auto cpu_long_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto cpu_i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto local_ids_cpu = torch::empty({active}, cpu_long_opts);
    auto route_slots_cpu = torch::empty({routes}, cpu_long_opts);
    auto route_positions_cpu = torch::empty({routes}, cpu_long_opts);
    auto compact_starts_cpu = torch::empty({active + 1}, cpu_i32_opts);

    auto* local_ids_ptr = local_ids_cpu.data_ptr<int64_t>();
    auto* route_slots_ptr = route_slots_cpu.data_ptr<int64_t>();
    auto* route_positions_ptr = route_positions_cpu.data_ptr<int64_t>();
    auto* compact_ptr = compact_starts_cpu.data_ptr<int32_t>();
    int64_t active_pos = 0;
    int64_t route_out = 0;
    compact_ptr[0] = 0;
    for (int64_t local = 0; local < experts; ++local) {
        const auto& positions = routes_by_local[static_cast<size_t>(local)];
        if (positions.empty()) {
            continue;
        }
        local_ids_ptr[active_pos] = local;
        for (int64_t route_pos : positions) {
            route_slots_ptr[route_out] = active_pos;
            route_positions_ptr[route_out] = route_pos;
            ++route_out;
        }
        ++active_pos;
        compact_ptr[active_pos] = static_cast<int32_t>(route_out);
    }

    auto route_slots = torch::empty({routes}, indices.options().dtype(torch::kInt64));
    auto route_positions = torch::empty({routes}, indices.options().dtype(torch::kInt64));
    auto route_tokens = torch::zeros({routes}, indices.options().dtype(torch::kInt64));
    auto compact_starts = torch::empty({active + 1}, indices.options().dtype(torch::kInt32));
    route_slots.copy_(route_slots_cpu);
    route_positions.copy_(route_positions_cpu);
    compact_starts.copy_(compact_starts_cpu);
    auto route_weights = weights.view({-1}).index_select(0, route_positions);
    return {local_ids_cpu, route_tokens, route_weights, route_slots, compact_starts};
}

torch::Tensor moe_single_token_int8_forward(
    const torch::Tensor& x,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    int64_t experts_start_idx,
    double swiglu_limit) {
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "x must have shape [1, D]");
    TORCH_CHECK(indices.dim() == 1, "indices must have shape [K]");
    TORCH_CHECK(weights.dim() == 1, "weights must have shape [K]");
    TORCH_CHECK(indices.size(0) == weights.size(0), "indices/weights length mismatch");
    TORCH_CHECK(w1q.dim() == 3 && w2q.dim() == 3 && w3q.dim() == 3, "weights must have shape [E, N, K]");
    TORCH_CHECK(w1s.dim() == 2 && w2s.dim() == 2 && w3s.dim() == 2, "scales must have shape [E, N]");
    TORCH_CHECK(w1q.size(0) == w2q.size(0) && w1q.size(0) == w3q.size(0), "expert count mismatch");
    TORCH_CHECK(w1q.size(1) == w3q.size(1), "w1/w3 inter_dim mismatch");
    TORCH_CHECK(w1q.size(2) == x.size(1) && w3q.size(2) == x.size(1), "w1/w3 input dim mismatch");
    TORCH_CHECK(w2q.size(2) == w1q.size(1), "w2 input dim mismatch");
    TORCH_CHECK(w2q.size(1) == x.size(1), "w2 output dim mismatch");
    TORCH_CHECK(x.is_cuda() && indices.is_cuda() && weights.is_cuda(), "x/indices/weights must be CUDA tensors");
    TORCH_CHECK(w1q.is_cuda() && w1s.is_cuda() && w2q.is_cuda() && w2s.is_cuda() && w3q.is_cuda() && w3s.is_cuda(), "weight tensors must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && indices.is_contiguous() && weights.is_contiguous(), "x/indices/weights must be contiguous");
    TORCH_CHECK(w1q.is_contiguous() && w1s.is_contiguous() && w2q.is_contiguous() && w2s.is_contiguous() && w3q.is_contiguous() && w3s.is_contiguous(), "weight tensors must be contiguous");
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");
    TORCH_CHECK(indices.scalar_type() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
    check_tensor(w1q, "w1q", torch::kInt8);
    check_tensor(w2q, "w2q", torch::kInt8);
    check_tensor(w3q, "w3q", torch::kInt8);
    check_tensor(w1s, "w1s", torch::kFloat32);
    check_tensor(w2s, "w2s", torch::kFloat32);
    check_tensor(w3s, "w3s", torch::kFloat32);
    return moe_single_token_int8_forward_cuda(
        x,
        indices,
        weights,
        w1q,
        w1s,
        w2q,
        w2s,
        w3q,
        w3s,
        experts_start_idx,
        swiglu_limit);
}

torch::Tensor moe_single_token_fp4_forward(
    const torch::Tensor& x,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    int64_t experts_start_idx,
    double swiglu_limit) {
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "x must have shape [1, D]");
    TORCH_CHECK(indices.dim() == 1, "indices must have shape [K]");
    TORCH_CHECK(weights.dim() == 1, "weights must have shape [K]");
    TORCH_CHECK(indices.size(0) == weights.size(0), "indices/weights length mismatch");
    TORCH_CHECK(w1q.dim() == 3 && w2q.dim() == 3 && w3q.dim() == 3, "fp4 weights must have shape [E, N, K/2]");
    TORCH_CHECK(w1s.dim() == 3 && w2s.dim() == 3 && w3s.dim() == 3, "fp4 scales must have shape [E, N, K/32]");
    TORCH_CHECK(w1q.size(0) == w2q.size(0) && w1q.size(0) == w3q.size(0), "expert count mismatch");
    TORCH_CHECK(w1q.size(1) == w3q.size(1), "w1/w3 inter_dim mismatch");
    const int64_t dim = x.size(1);
    const int64_t inter_dim = w1q.size(1);
    TORCH_CHECK(dim % 32 == 0, "dim must be divisible by 32 for fp4 path");
    TORCH_CHECK(inter_dim % 32 == 0, "inter_dim must be divisible by 32 for fp4 path");
    TORCH_CHECK(w1q.size(2) == dim / 2 && w3q.size(2) == dim / 2, "w1/w3 packed K must equal dim/2");
    TORCH_CHECK(w1s.size(2) == dim / 32 && w3s.size(2) == dim / 32, "w1/w3 scale K must equal dim/32");
    TORCH_CHECK(w2q.size(1) == dim, "w2 output dim must equal x.dim");
    TORCH_CHECK(w2q.size(2) == inter_dim / 2, "w2 packed K must equal inter_dim/2");
    TORCH_CHECK(w2s.size(2) == inter_dim / 32, "w2 scale K must equal inter_dim/32");
    TORCH_CHECK(w1s.size(1) == w1q.size(1) && w3s.size(1) == w3q.size(1) && w2s.size(1) == w2q.size(1), "scale row count mismatch");
    TORCH_CHECK(w1s.size(0) == w1q.size(0) && w3s.size(0) == w3q.size(0) && w2s.size(0) == w2q.size(0), "scale expert dim mismatch");
    TORCH_CHECK(x.is_cuda() && indices.is_cuda() && weights.is_cuda(), "x/indices/weights must be CUDA tensors");
    TORCH_CHECK(w1q.is_cuda() && w1s.is_cuda() && w2q.is_cuda() && w2s.is_cuda() && w3q.is_cuda() && w3s.is_cuda(), "weight tensors must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && indices.is_contiguous() && weights.is_contiguous(), "x/indices/weights must be contiguous");
    TORCH_CHECK(w1q.is_contiguous() && w1s.is_contiguous() && w2q.is_contiguous() && w2s.is_contiguous() && w3q.is_contiguous() && w3s.is_contiguous(), "weight tensors must be contiguous");
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");
    TORCH_CHECK(indices.scalar_type() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
    check_tensor(w1q, "w1q", torch::kUInt8);
    check_tensor(w2q, "w2q", torch::kUInt8);
    check_tensor(w3q, "w3q", torch::kUInt8);
    check_tensor(w1s, "w1s", torch::kUInt8);
    check_tensor(w2s, "w2s", torch::kUInt8);
    check_tensor(w3s, "w3s", torch::kUInt8);
    return moe_single_token_fp4_forward_cuda(
        x,
        indices,
        weights,
        w1q,
        w1s,
        w2q,
        w2s,
        w3q,
        w3s,
        experts_start_idx,
        swiglu_limit);
}

torch::Tensor moe_single_token_int8_forward_v2(
    const torch::Tensor& x,
    const torch::Tensor& route_to_slot,
    const torch::Tensor& weights,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    double swiglu_limit) {
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "x must have shape [1, D]");
    TORCH_CHECK(route_to_slot.dim() == 1, "route_to_slot must have shape [K]");
    TORCH_CHECK(weights.dim() == 1, "weights must have shape [K]");
    TORCH_CHECK(route_to_slot.size(0) == weights.size(0), "route_to_slot/weights length mismatch");
    TORCH_CHECK(w1q.dim() == 3 && w2q.dim() == 3 && w3q.dim() == 3, "packed weights must have shape [S, N, K]");
    TORCH_CHECK(w1s.dim() == 2 && w2s.dim() == 2 && w3s.dim() == 2, "packed scales must have shape [S, N]");
    TORCH_CHECK(w1q.size(0) == w2q.size(0) && w1q.size(0) == w3q.size(0), "packed slot count mismatch");
    TORCH_CHECK(w1q.size(1) == w3q.size(1), "w1/w3 inter_dim mismatch");
    TORCH_CHECK(w1q.size(2) == x.size(1) && w3q.size(2) == x.size(1), "w1/w3 input dim mismatch");
    TORCH_CHECK(w2q.size(2) == w1q.size(1), "w2 input dim mismatch");
    TORCH_CHECK(w2q.size(1) == x.size(1), "w2 output dim mismatch");
    TORCH_CHECK(x.is_cuda() && route_to_slot.is_cuda() && weights.is_cuda(), "x/route_to_slot/weights must be CUDA tensors");
    TORCH_CHECK(w1q.is_cuda() && w1s.is_cuda() && w2q.is_cuda() && w2s.is_cuda() && w3q.is_cuda() && w3s.is_cuda(), "weight tensors must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && route_to_slot.is_contiguous() && weights.is_contiguous(), "x/route_to_slot/weights must be contiguous");
    TORCH_CHECK(w1q.is_contiguous() && w1s.is_contiguous() && w2q.is_contiguous() && w2s.is_contiguous() && w3q.is_contiguous() && w3s.is_contiguous(), "weight tensors must be contiguous");
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");
    TORCH_CHECK(route_to_slot.scalar_type() == torch::kInt64, "route_to_slot must be int64");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
    check_tensor(w1q, "w1q", torch::kInt8);
    check_tensor(w2q, "w2q", torch::kInt8);
    check_tensor(w3q, "w3q", torch::kInt8);
    check_tensor(w1s, "w1s", torch::kFloat32);
    check_tensor(w2s, "w2s", torch::kFloat32);
    check_tensor(w3s, "w3s", torch::kFloat32);
    return moe_single_token_int8_forward_v2_cuda(
        x,
        route_to_slot,
        weights,
        w1q,
        w1s,
        w2q,
        w2s,
        w3q,
        w3s,
        swiglu_limit);
}

torch::Tensor moe_prefill_int8_grouped_forward(
    const torch::Tensor& x_sorted,
    const torch::Tensor& route_weights_sorted,
    const torch::Tensor& seg_starts,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    double swiglu_limit) {
    TORCH_CHECK(x_sorted.dim() == 2, "x_sorted must have shape [R, D]");
    TORCH_CHECK(route_weights_sorted.dim() == 1, "route_weights_sorted must have shape [R]");
    TORCH_CHECK(seg_starts.dim() == 1, "seg_starts must have shape [E + 1]");
    TORCH_CHECK(w1q.dim() == 3 && w2q.dim() == 3 && w3q.dim() == 3, "weights must have shape [E, N, K]");
    TORCH_CHECK(w1s.dim() == 2 && w2s.dim() == 2 && w3s.dim() == 2, "scales must have shape [E, N]");
    TORCH_CHECK(x_sorted.size(0) == route_weights_sorted.size(0), "route count mismatch");
    TORCH_CHECK(seg_starts.size(0) == w1q.size(0) + 1, "seg_starts/expert count mismatch");
    TORCH_CHECK(w1q.size(0) == w1s.size(0) && w2q.size(0) == w2s.size(0) && w3q.size(0) == w3s.size(0), "weight/scale expert count mismatch");
    TORCH_CHECK(w1q.size(0) == w2q.size(0) && w1q.size(0) == w3q.size(0), "expert count mismatch");
    TORCH_CHECK(w1q.size(1) == w1s.size(1) && w2q.size(1) == w2s.size(1) && w3q.size(1) == w3s.size(1), "weight/scale row mismatch");
    TORCH_CHECK(w1q.size(1) == w3q.size(1), "w1/w3 inter_dim mismatch");
    TORCH_CHECK(w1q.size(2) == x_sorted.size(1) && w3q.size(2) == x_sorted.size(1), "w1/w3 input dim mismatch");
    TORCH_CHECK(w2q.size(2) == w1q.size(1), "w2 input dim mismatch");
    TORCH_CHECK(w2q.size(1) == x_sorted.size(1), "w2 output dim mismatch");
    TORCH_CHECK(x_sorted.is_cuda() && route_weights_sorted.is_cuda() && seg_starts.is_cuda(), "route tensors must be CUDA tensors");
    TORCH_CHECK(w1q.is_cuda() && w1s.is_cuda() && w2q.is_cuda() && w2s.is_cuda() && w3q.is_cuda() && w3s.is_cuda(), "weight tensors must be CUDA tensors");
    TORCH_CHECK(x_sorted.is_contiguous() && route_weights_sorted.is_contiguous() && seg_starts.is_contiguous(), "route tensors must be contiguous");
    TORCH_CHECK(w1q.is_contiguous() && w1s.is_contiguous() && w2q.is_contiguous() && w2s.is_contiguous() && w3q.is_contiguous() && w3s.is_contiguous(), "weight tensors must be contiguous");
    TORCH_CHECK(
        x_sorted.scalar_type() == torch::kFloat16 ||
        x_sorted.scalar_type() == torch::kBFloat16 ||
        x_sorted.scalar_type() == torch::kFloat32,
        "x_sorted must be float16, bfloat16, or float32");
    TORCH_CHECK(route_weights_sorted.scalar_type() == torch::kFloat32, "route_weights_sorted must be float32");
    TORCH_CHECK(seg_starts.scalar_type() == torch::kInt32 || seg_starts.scalar_type() == torch::kInt64, "seg_starts must be int32 or int64");
    check_tensor(w1q, "w1q", torch::kInt8);
    check_tensor(w2q, "w2q", torch::kInt8);
    check_tensor(w3q, "w3q", torch::kInt8);
    check_tensor(w1s, "w1s", torch::kFloat32);
    check_tensor(w2s, "w2s", torch::kFloat32);
    check_tensor(w3s, "w3s", torch::kFloat32);
    return moe_prefill_int8_grouped_forward_cuda(
        x_sorted,
        route_weights_sorted,
        seg_starts,
        w1q,
        w1s,
        w2q,
        w2s,
        w3q,
        w3s,
        swiglu_limit);
}

torch::Tensor moe_prefill_int8_grouped_gemm_forward(
    const torch::Tensor& x,
    const torch::Tensor& route_tokens,
    const torch::Tensor& route_weights_sorted,
    const torch::Tensor& seg_starts,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    double swiglu_limit) {
    TORCH_CHECK(x.dim() == 2, "x must have shape [T, D]");
    TORCH_CHECK(route_tokens.dim() == 1, "route_tokens must have shape [R]");
    TORCH_CHECK(route_weights_sorted.dim() == 1, "route_weights_sorted must have shape [R]");
    TORCH_CHECK(seg_starts.dim() == 1, "seg_starts must have shape [E + 1]");
    TORCH_CHECK(w1q.dim() == 3 && w2q.dim() == 3 && w3q.dim() == 3, "weights must have shape [E, N, K]");
    TORCH_CHECK(w1s.dim() == 2 && w2s.dim() == 2 && w3s.dim() == 2, "scales must have shape [E, N]");
    TORCH_CHECK(route_tokens.size(0) == route_weights_sorted.size(0), "route count mismatch");
    TORCH_CHECK(seg_starts.size(0) == w1q.size(0) + 1, "seg_starts/expert count mismatch");
    TORCH_CHECK(w1q.size(0) == w1s.size(0) && w2q.size(0) == w2s.size(0) && w3q.size(0) == w3s.size(0), "weight/scale expert count mismatch");
    TORCH_CHECK(w1q.size(0) == w2q.size(0) && w1q.size(0) == w3q.size(0), "expert count mismatch");
    TORCH_CHECK(w1q.size(1) == w1s.size(1) && w2q.size(1) == w2s.size(1) && w3q.size(1) == w3s.size(1), "weight/scale row mismatch");
    TORCH_CHECK(w1q.size(1) == w3q.size(1), "w1/w3 inter_dim mismatch");
    TORCH_CHECK(w1q.size(2) == x.size(1) && w3q.size(2) == x.size(1), "w1/w3 input dim mismatch");
    TORCH_CHECK(w2q.size(2) == w1q.size(1), "w2 input dim mismatch");
    TORCH_CHECK(w2q.size(1) == x.size(1), "w2 output dim mismatch");
    TORCH_CHECK(x.is_cuda() && route_tokens.is_cuda() && route_weights_sorted.is_cuda() && seg_starts.is_cuda(), "route tensors must be CUDA tensors");
    TORCH_CHECK(w1q.is_cuda() && w1s.is_cuda() && w2q.is_cuda() && w2s.is_cuda() && w3q.is_cuda() && w3s.is_cuda(), "weight tensors must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && route_tokens.is_contiguous() && route_weights_sorted.is_contiguous() && seg_starts.is_contiguous(), "route tensors must be contiguous");
    TORCH_CHECK(w1q.is_contiguous() && w1s.is_contiguous() && w2q.is_contiguous() && w2s.is_contiguous() && w3q.is_contiguous() && w3s.is_contiguous(), "weight tensors must be contiguous");
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");
    TORCH_CHECK(route_tokens.scalar_type() == torch::kInt64, "route_tokens must be int64");
    TORCH_CHECK(route_weights_sorted.scalar_type() == torch::kFloat32, "route_weights_sorted must be float32");
    TORCH_CHECK(seg_starts.scalar_type() == torch::kInt32 || seg_starts.scalar_type() == torch::kInt64, "seg_starts must be int32 or int64");
    check_tensor(w1q, "w1q", torch::kInt8);
    check_tensor(w2q, "w2q", torch::kInt8);
    check_tensor(w3q, "w3q", torch::kInt8);
    check_tensor(w1s, "w1s", torch::kFloat32);
    check_tensor(w2s, "w2s", torch::kFloat32);
    check_tensor(w3s, "w3s", torch::kFloat32);
    return moe_prefill_int8_grouped_gemm_forward_cuda(
        x,
        route_tokens,
        route_weights_sorted,
        seg_starts,
        w1q,
        w1s,
        w2q,
        w2s,
        w3q,
        w3s,
        swiglu_limit);
}

torch::Tensor moe_prefill_fp4_grouped_gemm_forward(
    const torch::Tensor& x,
    const torch::Tensor& route_tokens,
    const torch::Tensor& route_weights_sorted,
    const torch::Tensor& seg_starts,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    double swiglu_limit) {
    TORCH_CHECK(x.dim() == 2, "x must have shape [T, D]");
    TORCH_CHECK(route_tokens.dim() == 1, "route_tokens must have shape [R]");
    TORCH_CHECK(route_weights_sorted.dim() == 1, "route_weights_sorted must have shape [R]");
    TORCH_CHECK(seg_starts.dim() == 1, "seg_starts must have shape [E + 1]");
    TORCH_CHECK(w1q.dim() == 3 && w2q.dim() == 3 && w3q.dim() == 3, "fp4 weights must have shape [E, N, K/2]");
    TORCH_CHECK(w1s.dim() == 3 && w2s.dim() == 3 && w3s.dim() == 3, "fp4 scales must have shape [E, N, K/32]");
    TORCH_CHECK(route_tokens.size(0) == route_weights_sorted.size(0), "route count mismatch");
    TORCH_CHECK(seg_starts.size(0) == w1q.size(0) + 1, "seg_starts/expert count mismatch");
    TORCH_CHECK(w1q.size(0) == w2q.size(0) && w1q.size(0) == w3q.size(0), "expert count mismatch");
    TORCH_CHECK(w1q.size(1) == w3q.size(1), "w1/w3 inter_dim mismatch");
    const int64_t dim = x.size(1);
    const int64_t inter_dim = w1q.size(1);
    TORCH_CHECK(dim % 32 == 0, "dim must be divisible by 32 for fp4 path");
    TORCH_CHECK(inter_dim % 32 == 0, "inter_dim must be divisible by 32 for fp4 path");
    TORCH_CHECK(w1q.size(2) == dim / 2 && w3q.size(2) == dim / 2, "w1/w3 packed K must equal dim/2");
    TORCH_CHECK(w1s.size(2) == dim / 32 && w3s.size(2) == dim / 32, "w1/w3 scale K must equal dim/32");
    TORCH_CHECK(w2q.size(1) == dim, "w2 output dim must equal x.dim");
    TORCH_CHECK(w2q.size(2) == inter_dim / 2, "w2 packed K must equal inter_dim/2");
    TORCH_CHECK(w2s.size(2) == inter_dim / 32, "w2 scale K must equal inter_dim/32");
    TORCH_CHECK(w1s.size(0) == w1q.size(0) && w3s.size(0) == w3q.size(0) && w2s.size(0) == w2q.size(0), "scale expert dim mismatch");
    TORCH_CHECK(w1s.size(1) == w1q.size(1) && w3s.size(1) == w3q.size(1) && w2s.size(1) == w2q.size(1), "scale row count mismatch");
    TORCH_CHECK(x.is_cuda() && route_tokens.is_cuda() && route_weights_sorted.is_cuda() && seg_starts.is_cuda(), "route tensors must be CUDA tensors");
    TORCH_CHECK(w1q.is_cuda() && w1s.is_cuda() && w2q.is_cuda() && w2s.is_cuda() && w3q.is_cuda() && w3s.is_cuda(), "weight tensors must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && route_tokens.is_contiguous() && route_weights_sorted.is_contiguous() && seg_starts.is_contiguous(), "route tensors must be contiguous");
    TORCH_CHECK(w1q.is_contiguous() && w1s.is_contiguous() && w2q.is_contiguous() && w2s.is_contiguous() && w3q.is_contiguous() && w3s.is_contiguous(), "weight tensors must be contiguous");
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");
    TORCH_CHECK(route_tokens.scalar_type() == torch::kInt64, "route_tokens must be int64");
    TORCH_CHECK(route_weights_sorted.scalar_type() == torch::kFloat32, "route_weights_sorted must be float32");
    TORCH_CHECK(seg_starts.scalar_type() == torch::kInt32 || seg_starts.scalar_type() == torch::kInt64, "seg_starts must be int32 or int64");
    check_tensor(w1q, "w1q", torch::kUInt8);
    check_tensor(w2q, "w2q", torch::kUInt8);
    check_tensor(w3q, "w3q", torch::kUInt8);
    check_tensor(w1s, "w1s", torch::kUInt8);
    check_tensor(w2s, "w2s", torch::kUInt8);
    check_tensor(w3s, "w3s", torch::kUInt8);
    return moe_prefill_fp4_grouped_gemm_forward_cuda(
        x,
        route_tokens,
        route_weights_sorted,
        seg_starts,
        w1q,
        w1s,
        w2q,
        w2s,
        w3q,
        w3s,
        swiglu_limit);
}

std::vector<torch::Tensor> fp4_weight_to_int8_forward(
    const torch::Tensor& wq,
    const torch::Tensor& ws) {
    TORCH_CHECK(wq.dim() == 3, "wq must have shape [E, N, K/2]");
    TORCH_CHECK(ws.dim() == 3, "ws must have shape [E, N, K/32]");
    TORCH_CHECK(wq.is_cuda() && ws.is_cuda(), "FP4 conversion tensors must be CUDA tensors");
    TORCH_CHECK(wq.is_contiguous() && ws.is_contiguous(), "FP4 conversion tensors must be contiguous");
    check_tensor(wq, "wq", torch::kUInt8);
    check_tensor(ws, "ws", torch::kUInt8);
    TORCH_CHECK(wq.size(0) == ws.size(0) && wq.size(1) == ws.size(1), "FP4 conversion expert/row shape mismatch");
    TORCH_CHECK(wq.size(2) % 16 == 0, "packed FP4 K/2 must be divisible by 16");
    TORCH_CHECK(ws.size(2) == wq.size(2) / 16, "FP4 scale K/32 shape mismatch");
    return fp4_weight_to_int8_forward_cuda(wq, ws);
}

torch::Tensor moe_prefill_int8_grouped_gemm_bucketed_forward(
    const torch::Tensor& x,
    const torch::Tensor& route_tokens,
    const torch::Tensor& route_weights_sorted,
    const torch::Tensor& seg_starts,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    double swiglu_limit) {
    TORCH_CHECK(x.dim() == 2, "x must have shape [T, D]");
    TORCH_CHECK(route_tokens.dim() == 1, "route_tokens must have shape [R]");
    TORCH_CHECK(route_weights_sorted.dim() == 1, "route_weights_sorted must have shape [R]");
    TORCH_CHECK(seg_starts.dim() == 1, "seg_starts must have shape [E + 1]");
    TORCH_CHECK(w1q.dim() == 3 && w2q.dim() == 3 && w3q.dim() == 3, "weights must have shape [E, N, K]");
    TORCH_CHECK(w1s.dim() == 2 && w2s.dim() == 2 && w3s.dim() == 2, "scales must have shape [E, N]");
    TORCH_CHECK(route_tokens.size(0) == route_weights_sorted.size(0), "route count mismatch");
    TORCH_CHECK(seg_starts.size(0) == w1q.size(0) + 1, "seg_starts/expert count mismatch");
    TORCH_CHECK(w1q.size(0) == w1s.size(0) && w2q.size(0) == w2s.size(0) && w3q.size(0) == w3s.size(0), "weight/scale expert count mismatch");
    TORCH_CHECK(w1q.size(0) == w2q.size(0) && w1q.size(0) == w3q.size(0), "expert count mismatch");
    TORCH_CHECK(w1q.size(1) == w1s.size(1) && w2q.size(1) == w2s.size(1) && w3q.size(1) == w3s.size(1), "weight/scale row mismatch");
    TORCH_CHECK(w1q.size(1) == w3q.size(1), "w1/w3 inter_dim mismatch");
    TORCH_CHECK(w1q.size(2) == x.size(1) && w3q.size(2) == x.size(1), "w1/w3 input dim mismatch");
    TORCH_CHECK(w2q.size(2) == w1q.size(1), "w2 input dim mismatch");
    TORCH_CHECK(w2q.size(1) == x.size(1), "w2 output dim mismatch");
    TORCH_CHECK(x.is_cuda() && route_tokens.is_cuda() && route_weights_sorted.is_cuda() && seg_starts.is_cuda(), "route tensors must be CUDA tensors");
    TORCH_CHECK(w1q.is_cuda() && w1s.is_cuda() && w2q.is_cuda() && w2s.is_cuda() && w3q.is_cuda() && w3s.is_cuda(), "weight tensors must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && route_tokens.is_contiguous() && route_weights_sorted.is_contiguous() && seg_starts.is_contiguous(), "route tensors must be contiguous");
    TORCH_CHECK(w1q.is_contiguous() && w1s.is_contiguous() && w2q.is_contiguous() && w2s.is_contiguous() && w3q.is_contiguous() && w3s.is_contiguous(), "weight tensors must be contiguous");
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");
    TORCH_CHECK(route_tokens.scalar_type() == torch::kInt64, "route_tokens must be int64");
    TORCH_CHECK(route_weights_sorted.scalar_type() == torch::kFloat32, "route_weights_sorted must be float32");
    TORCH_CHECK(seg_starts.scalar_type() == torch::kInt32 || seg_starts.scalar_type() == torch::kInt64, "seg_starts must be int32 or int64");
    check_tensor(w1q, "w1q", torch::kInt8);
    check_tensor(w2q, "w2q", torch::kInt8);
    check_tensor(w3q, "w3q", torch::kInt8);
    check_tensor(w1s, "w1s", torch::kFloat32);
    check_tensor(w2s, "w2s", torch::kFloat32);
    check_tensor(w3s, "w3s", torch::kFloat32);
    return moe_prefill_int8_grouped_gemm_bucketed_forward_cuda(
        x,
        route_tokens,
        route_weights_sorted,
        seg_starts,
        w1q,
        w1s,
        w2q,
        w2s,
        w3q,
        w3s,
        swiglu_limit);
}



torch::Tensor moe_prefill_int8_fused_forward(
    const torch::Tensor& x,
    const torch::Tensor& route_tokens,
    const torch::Tensor& route_weights_sorted,
    const torch::Tensor& seg_starts,
    const torch::Tensor& w1q,
    const torch::Tensor& w1s,
    const torch::Tensor& w2q,
    const torch::Tensor& w2s,
    const torch::Tensor& w3q,
    const torch::Tensor& w3s,
    double swiglu_limit) {
    TORCH_CHECK(x.dim() == 2, "x must have shape [T, D]");
    TORCH_CHECK(route_tokens.dim() == 1, "route_tokens must have shape [R]");
    TORCH_CHECK(route_weights_sorted.dim() == 1, "route_weights_sorted must have shape [R]");
    TORCH_CHECK(seg_starts.dim() == 1, "seg_starts must have shape [E + 1]");
    TORCH_CHECK(w1q.dim() == 3 && w2q.dim() == 3 && w3q.dim() == 3, "weights must have shape [E, N, K]");
    TORCH_CHECK(w1s.dim() == 2 && w2s.dim() == 2 && w3s.dim() == 2, "scales must have shape [E, N]");
    TORCH_CHECK(route_tokens.size(0) == route_weights_sorted.size(0), "route count mismatch");
    TORCH_CHECK(seg_starts.size(0) == w1q.size(0) + 1, "seg_starts/expert count mismatch");
    TORCH_CHECK(w1q.size(0) == w1s.size(0) && w2q.size(0) == w2s.size(0) && w3q.size(0) == w3s.size(0), "weight/scale expert count mismatch");
    TORCH_CHECK(w1q.size(0) == w2q.size(0) && w1q.size(0) == w3q.size(0), "expert count mismatch");
    TORCH_CHECK(w1q.size(1) == w1s.size(1) && w2q.size(1) == w2s.size(1) && w3q.size(1) == w3s.size(1), "weight/scale row mismatch");
    TORCH_CHECK(w1q.size(1) == w3q.size(1), "w1/w3 inter_dim mismatch");
    TORCH_CHECK(w1q.size(2) == x.size(1) && w3q.size(2) == x.size(1), "w1/w3 input dim mismatch");
    TORCH_CHECK(w2q.size(2) == w1q.size(1), "w2 input dim mismatch");
    TORCH_CHECK(w2q.size(1) == x.size(1), "w2 output dim mismatch");
    TORCH_CHECK(x.is_cuda() && route_tokens.is_cuda() && route_weights_sorted.is_cuda() && seg_starts.is_cuda(), "route tensors must be CUDA tensors");
    TORCH_CHECK(w1q.is_cuda() && w1s.is_cuda() && w2q.is_cuda() && w2s.is_cuda() && w3q.is_cuda() && w3s.is_cuda(), "weight tensors must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && route_tokens.is_contiguous() && route_weights_sorted.is_contiguous() && seg_starts.is_contiguous(), "route tensors must be contiguous");
    TORCH_CHECK(w1q.is_contiguous() && w1s.is_contiguous() && w2q.is_contiguous() && w2s.is_contiguous() && w3q.is_contiguous() && w3s.is_contiguous(), "weight tensors must be contiguous");
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");
    TORCH_CHECK(route_tokens.scalar_type() == torch::kInt64, "route_tokens must be int64");
    TORCH_CHECK(route_weights_sorted.scalar_type() == torch::kFloat32, "route_weights_sorted must be float32");
    TORCH_CHECK(seg_starts.scalar_type() == torch::kInt32 || seg_starts.scalar_type() == torch::kInt64, "seg_starts must be int32 or int64");
    check_tensor(w1q, "w1q", torch::kInt8);
    check_tensor(w2q, "w2q", torch::kInt8);
    check_tensor(w3q, "w3q", torch::kInt8);
    check_tensor(w1s, "w1s", torch::kFloat32);
    check_tensor(w2s, "w2s", torch::kFloat32);
    check_tensor(w3s, "w3s", torch::kFloat32);
    return moe_prefill_int8_fused_forward_cuda(
        x,
        route_tokens,
        route_weights_sorted,
        seg_starts,
        w1q,
        w1s,
        w2q,
        w2s,
        w3q,
        w3s,
        swiglu_limit);
}

void fused_q_rmsnorm_rope_inplace(
    torch::Tensor& q,
    const torch::Tensor& freqs_real,
    const torch::Tensor& freqs_imag,
    double eps) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(q.dim() == 4, "q must be [B, S, H, D]");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(freqs_real.is_cuda() && freqs_imag.is_cuda(), "freqs must be CUDA");
    TORCH_CHECK(freqs_real.scalar_type() == at::kFloat, "freqs_real must be float32");
    TORCH_CHECK(freqs_imag.scalar_type() == at::kFloat, "freqs_imag must be float32");
    TORCH_CHECK(freqs_real.is_contiguous() && freqs_imag.is_contiguous(), "freqs must be contiguous");
    TORCH_CHECK(freqs_real.dim() == 2 && freqs_imag.dim() == 2, "freqs must be [S, rd/2]");
    const int S = static_cast<int>(q.size(1));
    TORCH_CHECK(freqs_real.size(0) == S && freqs_imag.size(0) == S, "freqs S mismatch");
    const int rd = static_cast<int>(freqs_real.size(1)) * 2;
    const int D = static_cast<int>(q.size(3));
    TORCH_CHECK(rd > 0 && rd <= D && (rd % 2) == 0, "rd must be even and <= D");
    fused_q_rmsnorm_rope_inplace_cuda(q, freqs_real, freqs_imag, eps);
}

void fused_kv_rope_actquant_inplace(
    torch::Tensor& kv,
    const torch::Tensor& norm_weight,
    const torch::Tensor& freqs_real,
    const torch::Tensor& freqs_imag,
    int64_t block_size,
    double norm_eps,
    const c10::optional<torch::Tensor>& kv_cache_out,
    int64_t cache_slot) {
    TORCH_CHECK(kv.is_cuda() && kv.scalar_type() == at::kBFloat16 && kv.is_contiguous(), "kv must be CUDA bf16 contiguous");
    TORCH_CHECK(kv.dim() == 3, "kv must be [B, S, kv_dim]");
    TORCH_CHECK(norm_weight.is_cuda() && norm_weight.scalar_type() == at::kFloat, "norm_weight must be CUDA float32");
    TORCH_CHECK(norm_weight.is_contiguous(), "norm_weight must be contiguous");
    TORCH_CHECK(norm_weight.dim() == 1 && norm_weight.size(0) == kv.size(2), "norm_weight shape mismatch");
    TORCH_CHECK(freqs_real.is_cuda() && freqs_imag.is_cuda(), "freqs must be CUDA");
    TORCH_CHECK(freqs_real.scalar_type() == at::kFloat && freqs_imag.scalar_type() == at::kFloat, "freqs must be float32");
    TORCH_CHECK(freqs_real.is_contiguous() && freqs_imag.is_contiguous(), "freqs must be contiguous");
    TORCH_CHECK(freqs_real.dim() == 2 && freqs_imag.dim() == 2, "freqs must be [S, rd/2]");
    const int S = static_cast<int>(kv.size(1));
    TORCH_CHECK(freqs_real.size(0) == S && freqs_imag.size(0) == S, "freqs S mismatch");
    const int rd = static_cast<int>(freqs_real.size(1)) * 2;
    const int kv_dim = static_cast<int>(kv.size(2));
    TORCH_CHECK(rd > 0 && rd < kv_dim && (rd % 2) == 0, "rd must be even and < kv_dim");
    const int q_dim = kv_dim - rd;
    TORCH_CHECK(block_size > 0 && (q_dim % block_size) == 0, "block_size must divide kv_dim - rd");
    if (kv_cache_out.has_value() && kv_cache_out->defined()) {
        const auto& cache = *kv_cache_out;
        TORCH_CHECK(cache.is_cuda() && cache.scalar_type() == at::kBFloat16, "kv_cache_out must be CUDA bf16");
        TORCH_CHECK(cache.dim() == 3, "kv_cache_out must be [B, cache_len, kv_dim]");
        TORCH_CHECK(cache.size(0) >= kv.size(0), "kv_cache_out batch dim too small");
        TORCH_CHECK(cache.size(2) == kv_dim, "kv_cache_out last dim must equal kv_dim");
        TORCH_CHECK(cache.stride(2) == 1 && cache.stride(1) == kv_dim,
                    "kv_cache_out must be contiguous on last two dims");
        TORCH_CHECK(cache_slot >= 0 && cache_slot < cache.size(1),
                    "cache_slot out of range");
        // Decode-only fast path: writing one slot per batch row.
        TORCH_CHECK(S == 1, "kv_cache_out path requires S == 1 (decode)");
    }
    fused_kv_rope_actquant_inplace_cuda(kv, norm_weight, freqs_real, freqs_imag,
                                         block_size, norm_eps,
                                         kv_cache_out, cache_slot);
}

void fused_o_inverse_rope_inplace(
    torch::Tensor& o,
    const torch::Tensor& freqs_real,
    const torch::Tensor& freqs_imag) {
    TORCH_CHECK(o.is_cuda(), "o must be CUDA");
    TORCH_CHECK(o.scalar_type() == at::kBFloat16, "o must be bfloat16");
    TORCH_CHECK(o.dim() == 4, "o must be [B, S, H, D]");
    TORCH_CHECK(o.is_contiguous(), "o must be contiguous");
    TORCH_CHECK(freqs_real.is_cuda() && freqs_imag.is_cuda(), "freqs must be CUDA");
    TORCH_CHECK(freqs_real.scalar_type() == at::kFloat && freqs_imag.scalar_type() == at::kFloat, "freqs must be float32");
    TORCH_CHECK(freqs_real.is_contiguous() && freqs_imag.is_contiguous(), "freqs must be contiguous");
    TORCH_CHECK(freqs_real.dim() == 2 && freqs_imag.dim() == 2, "freqs must be [S, rd/2]");
    const int S = static_cast<int>(o.size(1));
    TORCH_CHECK(freqs_real.size(0) == S && freqs_imag.size(0) == S, "freqs S mismatch");
    const int rd = static_cast<int>(freqs_real.size(1)) * 2;
    const int D = static_cast<int>(o.size(3));
    TORCH_CHECK(rd > 0 && rd <= D && (rd % 2) == 0, "rd must be even and <= D");
    fused_o_inverse_rope_inplace_cuda(o, freqs_real, freqs_imag);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_allreduce_ipc_handle", &custom_allreduce_ipc_handle, "export custom allreduce CUDA IPC handles");
    m.def("custom_allreduce_open", &custom_allreduce_open, "open custom allreduce CUDA IPC handles");
    m.def("custom_allreduce_close", &custom_allreduce_close, "close custom allreduce handle");
    m.def("custom_allreduce_inplace", &custom_allreduce_inplace, "custom allreduce in-place over CUDA IPC buffers");
    m.def("moe_finalize_reduce_forward", &moe_finalize_reduce_forward, "finalize reduced MoE output with shared expert add and dtype cast (CUDA)");
    m.def("q8_0_gemm_forward", &q8_0_gemm_forward, "raw GGUF q8_0 GEMM forward (CUDA)");
    m.def("gguf_q2k_gemm_dp4a_forward", &gguf_q2k_gemm_dp4a_forward, "raw GGUF q2_k GEMM forward using Q8 activation + DP4A (CUDA)");
    m.def("gguf_quant_gemm_forward", &gguf_quant_gemm_forward, "raw GGUF iq2_xxs/q2_k GEMM forward (CUDA)");
    m.def("gguf_quant_gemm_prefill_forward", &gguf_quant_gemm_prefill_forward, "prefill-only raw GGUF iq2_xxs/q2_k GEMM forward (CUDA)");
    m.def("gguf_quant_gemm_pair_forward", &gguf_quant_gemm_pair_forward, "paired raw GGUF iq2_xxs/q2_k GEMM forward (CUDA)");
    m.def("gguf_moe_prefill_grouped_forward", &gguf_moe_prefill_grouped_forward, "raw GGUF routed grouped MoE prefill forward (CUDA)");
    m.def("gguf_moe_single_token_iq2_q2k_forward", &gguf_moe_single_token_iq2_q2k_forward, "single-token GGUF IQ2_XXS/Q2_K routed MoE forward (CUDA)");
    m.def("int8_gemm_forward", &int8_gemm_forward, "generic int8 GEMM forward (CUDA)");
    m.def("int8_gemm_pair_forward", &int8_gemm_pair_forward, "paired int8 GEMM forward with shared activation quantization (CUDA)");
    m.def("wo_a_int8_forward", &wo_a_int8_forward, "wo_a grouped int8 forward (CUDA)");
    m.def("fused_decode_sparse_attn_forward", &fused_decode_sparse_attn_forward, "fused decode sparse attention forward (CUDA)");
    m.def("fused_decode_sparse_attn_wmma_forward", &fused_decode_sparse_attn_wmma_forward, "fused decode sparse attention WMMA forward (CUDA)");
    m.def("flashinfer_style_sparse_attn_forward", &flashinfer_style_sparse_attn_forward, "FlashInfer-style online decode sparse attention forward (CUDA)");
    m.def("flashinfer_style_sparse_attn_headpair_forward", &flashinfer_style_sparse_attn_headpair_forward, "FlashInfer-style decode sparse attention computing two heads per block (CUDA)");
    m.def("prefill_sparse_attn_forward", &prefill_sparse_attn_forward, "prefill sparse attention forward (CUDA)");
    m.def("prefill_sparse_attn_headpair_forward", &prefill_sparse_attn_headpair_forward, "prefill sparse attention computing two heads per block (CUDA)");
    m.def("hadamard128_forward", &hadamard128_forward_cuda, "Hadamard128 forward (CUDA)");
    m.def("fused_c4_indexer_decode_forward", &fused_c4_indexer_decode_forward, "fused C4 indexer decode forward (CUDA)");
    m.def("c4_topk_from_scores", &c4_topk_from_scores, "C4 indexer top-k from scores (CUDA)");
    m.def("hc_split_pre_forward", &hc_split_pre_forward, "HC split/sinkhorn/pre-sum forward (CUDA)");
    m.def("hc_post_forward", &hc_post_forward, "HC post forward (CUDA)");
    m.def("moe_group_routes", &moe_group_routes, "group MoE routes by local expert (CUDA)");
    m.def("gguf_single_token_route_slots", &gguf_single_token_route_slots, "single-token GGUF route compaction for slot-cache decode");
    m.def("moe_single_token_int8_forward", &moe_single_token_int8_forward, "single-token top-k MoE int8 forward (CUDA)");
    m.def("moe_single_token_int8_forward_v2", &moe_single_token_int8_forward_v2, "single-token top-k MoE int8 forward with compact buffer + slot map (CUDA)");
    m.def("moe_single_token_fp4_forward", &moe_single_token_fp4_forward, "single-token top-k MoE FP4 (e2m1fn_x2 + e8m0 block) forward (CUDA)");
    m.def("moe_prefill_int8_grouped_forward", &moe_prefill_int8_grouped_forward, "prefill MoE grouped int8 forward (CUDA)");
    m.def("moe_prefill_int8_grouped_gemm_forward", &moe_prefill_int8_grouped_gemm_forward, "prefill MoE grouped-GEMM int8 forward (CUDA)");
    m.def("moe_prefill_fp4_grouped_gemm_forward", &moe_prefill_fp4_grouped_gemm_forward, "prefill MoE grouped-GEMM FP4 forward (CUDA)");
    m.def("fp4_weight_to_int8_forward", &fp4_weight_to_int8_forward, "convert FP4 block-scaled weights to int8 row-scaled weights (CUDA)");
    m.def("moe_prefill_int8_grouped_gemm_bucketed_forward", &moe_prefill_int8_grouped_gemm_bucketed_forward, "prefill MoE bucketed grouped-GEMM int8 forward (CUDA)");
    m.def("fused_q_rmsnorm_rope_inplace", &fused_q_rmsnorm_rope_inplace, "fused per-head rmsnorm + rope on q (CUDA, bf16, in-place)");
    m.def("fused_kv_rope_actquant_inplace", &fused_kv_rope_actquant_inplace,
          "fused kv_norm + rope + block-FP8 act_quant (CUDA, bf16, in-place); "
          "optionally writes the result row into kv_cache_out[b, cache_slot, :] (decode S==1).",
          pybind11::arg("kv"), pybind11::arg("norm_weight"),
          pybind11::arg("freqs_real"), pybind11::arg("freqs_imag"),
          pybind11::arg("block_size"), pybind11::arg("norm_eps"),
          pybind11::arg("kv_cache_out") = c10::optional<torch::Tensor>(),
          pybind11::arg("cache_slot") = static_cast<int64_t>(0));
    m.def("fused_o_inverse_rope_inplace", &fused_o_inverse_rope_inplace, "inverse rope on attention output o (CUDA, bf16, in-place)");
}
