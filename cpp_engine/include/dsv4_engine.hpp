#pragma once

#include "gguf_reader.hpp"
#include "model_config.hpp"

#include <string>

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
    float checksum = 0.0f;
};

ForwardSmokeResult run_safetensors_min_layer_smoke(const std::string& ckpt_dir);
ForwardSmokeResult run_safetensors_layer_loop_smoke(const std::string& ckpt_dir, int layer_count);

}  // namespace dsv4
