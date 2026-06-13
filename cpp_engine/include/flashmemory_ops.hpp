#pragma once

#include <cstdint>
#include <vector>

namespace dsv4 {

// Score one FlashMemory CSA layer against all compressed-K chunks for a single
// decode token. Reproduces retriever.py forward_and_score (fp32 weights).
//
// Inputs (device):
//   d_hidden        [hidden_dim]                      decode-token hidden state
//   d_wq_a          [q_lora_rank, hidden_dim]         row-major weight
//   d_wq_b          [n_heads*head_dim, q_lora_rank]   row-major weight
//   d_q_norm        [q_lora_rank]                     RMSNorm gamma
//   d_weights_proj  [n_heads, hidden_dim]             row-major weight
//   d_inv_freqs     [rope_dim/2]                      YaRN inverse frequencies
//   d_compressed_k  [n_chunks, head_dim + 4] uint8    fp8 K + f32 scale per chunk
// Scratch (device, caller-owned):
//   d_q_scratch     [n_heads*head_dim]
//   d_qlora_scratch [q_lora_rank]
//   d_fused_w       [n_heads]
// Output (device):
//   d_scores        [n_chunks]  sigmoid in [0,1] if apply_sigmoid else raw logits
bool flashmemory_score_layer_cuda(
    const float* d_hidden,
    const float* d_wq_a,
    const float* d_wq_b,
    const float* d_q_norm,
    const float* d_weights_proj,
    const float* d_inv_freqs,
    const uint8_t* d_compressed_k,
    float* d_q_scratch,
    float* d_qlora_scratch,
    float* d_fused_w,
    float* d_scores,
    int n_chunks,
    int n_heads,
    int head_dim,
    int q_lora_rank,
    int hidden_dim,
    int rope_dim,
    float rms_norm_eps,
    long long position,
    bool apply_sigmoid,
    void* stream = nullptr);

// Compute YaRN RoPE inverse frequencies on host (matches retriever.py
// precompute_freqs_cis mixed-freqs construction). Returns [rope_dim/2] floats:
// the per-pair angular frequency (mixed = freq*(1-ramp) + (freq/factor)*ramp).
std::vector<float> flashmemory_yarn_inv_freqs(
    int rope_dim,
    double base,
    double factor,
    int original_seq_len,
    double beta_fast,
    double beta_slow);

}  // namespace dsv4
