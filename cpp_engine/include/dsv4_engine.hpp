#pragma once

#include "gguf_reader.hpp"
#include "model_config.hpp"

#include <string>
#include <vector>

namespace dsv4 {

class Dsv4Engine {
public:
    explicit Dsv4Engine(const std::string& model_path);

    const GGUFFile& gguf() const { return gguf_; }
    const ModelConfig& config() const { return config_; }

private:
    GGUFFile gguf_;
    ModelConfig config_;
};

struct ForwardSmokeResult {
    int token = 0;
    int dim = 0;
    int inter = 0;
    int logits = 0;
    int layers = 0;
    int top_token = 0;
    float top_logit = 0.0f;
    float checksum = 0.0f;
};

struct GenerateSmokeResult {
    std::vector<ForwardSmokeResult> tokens;
    double wall_seconds = 0.0;
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    int prompt_tokens = 0;
    int decode_tokens = 0;
};

struct ForwardSmokeOptions {
    int tp_world = 1;
    int tp_rank = 0;
    int device = 0;
    bool skip_fp4_host_prepare = false;
    std::string nccl_id_path;
};

ForwardSmokeResult run_safetensors_min_layer_smoke(const std::string& ckpt_dir);
ForwardSmokeResult run_safetensors_layer_loop_smoke(const std::string& ckpt_dir, int layer_count);
ForwardSmokeResult run_safetensors_token_forward(const std::string& ckpt_dir, int token, int layer_count);
ForwardSmokeResult run_safetensors_token_forward_at_position(const std::string& ckpt_dir, int token, int layer_count, int position);
ForwardSmokeResult run_safetensors_token_forward_with_options(const std::string& ckpt_dir, int token, int layer_count, int position, const ForwardSmokeOptions& options);
ForwardSmokeResult run_safetensors_prompt_forward(const std::string& ckpt_dir, const std::vector<int>& tokens, int layer_count);
ForwardSmokeResult run_safetensors_prompt_forward_with_options(const std::string& ckpt_dir, const std::vector<int>& tokens, int layer_count, const ForwardSmokeOptions& options);
std::vector<ForwardSmokeResult> run_safetensors_generate_tokens(const std::string& ckpt_dir, const std::vector<int>& seed_tokens, int layer_count, int max_new_tokens);
std::vector<ForwardSmokeResult> run_safetensors_generate_tokens_with_options(const std::string& ckpt_dir, const std::vector<int>& seed_tokens, int layer_count, int max_new_tokens, const ForwardSmokeOptions& options);
GenerateSmokeResult run_safetensors_generate_tokens_timed_with_options(const std::string& ckpt_dir, const std::vector<int>& seed_tokens, int layer_count, int max_new_tokens, const ForwardSmokeOptions& options);

// --- GGUF Q2 forward entry points ------------------------------------------
//
// Sibling to the safetensors path. The full forward chain isn't wired in yet
// (Phase 3); this initial entry point only verifies that we can construct a
// GgufForwardContext and that the GGUF mapping table resolves the dense
// metadata we'll need. Implementation is in dsv4_engine.cpp alongside the
// safetensors path so future GGUF forward operators can share helpers.

struct GgufSmokeResult {
    int n_layers = 0;
    int n_hash_layers = 0;
    int dim = 0;
    int moe_inter_dim = 0;
    int n_routed_experts = 0;
    int n_activated_experts = 0;
    int vocab = 0;
};

GgufSmokeResult run_gguf_min_layer_smoke(const std::string& ckpt_path);

// Phase 3 step: exercise the dense input chain for layer 0 attention's
// q_a branch. Embed F16 lookup -> attn_norm F32->BF16 RMSNorm -> wq_a Q8_0
// matvec. Validates the F32 norm gamma conversion path and the first
// Q8_0 attention projection against real GGUF data.
struct GgufAttnNormWqaResult {
    int dim = 0;
    int q_a_dim = 0;
    float embed_rms = 0.0f;
    float normed_rms = 0.0f;
    float q_a_rms = 0.0f;
    float q_a_first[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

GgufAttnNormWqaResult run_gguf_attn_norm_wq_a_smoke(const std::string& ckpt_path, int token);

// Phase 3 step: full attention Q path for layer 0. Extends the q_a smoke
// through q_norm RMSNorm, wq_b Q8_0 projection, and head_rmsnorm_rope.
// Output is the final per-head Q tensor ready to feed attention compute.
struct GgufAttnQPathResult {
    int dim = 0;
    int q_a_dim = 0;
    int heads = 0;
    int head_dim = 0;
    int rope_dim = 0;
    float q_normed_rms = 0.0f;     // RMS of q_a after q_norm
    float q_pre_rope_rms = 0.0f;   // RMS of q (heads*head_dim) after wq_b, before head norm/rope
    float q_post_rope_rms = 0.0f;  // RMS of q after head_rmsnorm_rope
    float q_first[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

GgufAttnQPathResult run_gguf_attn_q_path_smoke(const std::string& ckpt_path, int token, int position);

// Phase 3 step: attention KV path for layer 0. Embed -> attn_norm -> wkv Q8_0
// projection (dim -> kv_dim) -> kv_norm RMSNorm -> head_rmsnorm_rope with
// heads=1, head_dim=kv_dim, rope_dim. RoPE is a rotation so the final L2 norm
// equals the post-rmsnorm L2 norm (sqrt(kv_dim) for unit-RMS output).
struct GgufAttnKvPathResult {
    int dim = 0;
    int kv_dim = 0;
    int rope_dim = 0;
    float kv_a_rms = 0.0f;       // RMS of kv_a after wkv (before rmsnorm)
    float kv_norm_rms = 0.0f;    // RMS of kv after rmsnorm with kv_norm gamma
    float kv_post_rope_rms = 0.0f; // RMS of kv after RoPE; rotation preserves L2
    float kv_first[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

GgufAttnKvPathResult run_gguf_attn_kv_path_smoke(const std::string& ckpt_path, int token, int position);

// Phase 3 step: full single-token dense attention for layer 0.
// Embed -> attn_norm -> Q-path (wq_a, q_norm, wq_b, head_rmsnorm_rope) ||
// KV-path (wkv, kv_norm, head_rmsnorm_rope) -> single_token_sparse_attention
// (with attn_sink) -> inverse RoPE on rope tail of each head -> grouped wo_a
// Q8_0 (o_groups separate matvecs) -> wo_b Q8_0 (-> dim).
// Output `attn_out` is the per-layer attention residual (length dim).
struct GgufAttnFullResult {
    int dim = 0;
    int q_a_dim = 0;
    int heads = 0;
    int head_dim = 0;
    int kv_dim = 0;
    int rope_dim = 0;
    int o_groups = 0;
    int o_lora_rank = 0;
    int attn_mid = 0;
    float q_rms = 0.0f;
    float kv_rms = 0.0f;
    float attn_value_rms = 0.0f;       // RMS of sparse-attention output
    float attn_value_post_inv_rms = 0.0f; // after inverse RoPE
    float attn_mid_rms = 0.0f;
    float attn_out_rms = 0.0f;
    float attn_out_first[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

GgufAttnFullResult run_gguf_attn_full_smoke(const std::string& ckpt_path, int token, int position);

// Phase 3 step: shared-expert FFN smoke for layer 0.
// Embed -> ffn_norm RMSNorm -> shared w1 / w3 Q8_0 matvec -> silu_mul ->
// shared w2 Q8_0 matvec. All three shared-expert projections are stored as
// Q8_0 in the GGUF model. Output is the shared-expert contribution to the
// FFN residual (length dim).
struct GgufSharedExpertResult {
    int dim = 0;
    int moe_inter_dim = 0;
    float ffn_normed_rms = 0.0f;
    float gate_rms = 0.0f;
    float up_rms = 0.0f;
    float hidden_rms = 0.0f;   // after silu_mul
    float shared_out_rms = 0.0f;
    float shared_out_first[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

GgufSharedExpertResult run_gguf_shared_expert_smoke(const std::string& ckpt_path, int token);

}  // namespace dsv4
