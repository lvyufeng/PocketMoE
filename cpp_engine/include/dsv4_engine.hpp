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

}  // namespace dsv4
