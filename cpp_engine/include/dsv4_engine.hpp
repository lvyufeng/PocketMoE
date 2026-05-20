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

struct ForwardSmokeOptions {
    int tp_world = 1;
    int tp_rank = 0;
    int device = 0;
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

}  // namespace dsv4
