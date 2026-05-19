#include "dsv4_engine.hpp"

#include "cuda_ops.hpp"
#include "safetensors_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace dsv4 {
namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

const SafeTensorInfo* require_tensor(const SafeTensorsShard& shard, const std::string& name) {
    const auto* info = shard.find_tensor(name);
    if (info == nullptr) throw std::runtime_error("missing tensor: " + name);
    return info;
}

struct Fp4Handle {
    SafeTensorsShard shard;
    const SafeTensorInfo* w = nullptr;
    const SafeTensorInfo* s = nullptr;
    SafeFp4TensorPair pair;
};

float bf16_to_float(uint16_t bits) {
    uint32_t value = static_cast<uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

struct RoutedExpert {
    int id = 0;
    float weight = 0.0f;
};

std::vector<RoutedExpert> select_smoke_experts(
    const SafeTensorsIndex& index,
    const std::string& prefix,
    int layer_id,
    int token,
    const std::vector<float>& ffn_norm,
    uint64_t n_hash_layers,
    uint64_t n_experts,
    uint64_t topk,
    float route_scale) {
    const int k = static_cast<int>(std::max<uint64_t>(1, topk));
    const std::string tid_name = prefix + "ffn.gate.tid2eid";
    if (static_cast<uint64_t>(layer_id) < n_hash_layers && index.shard_for_tensor(tid_name) != nullptr) {
        SafeTensorsShard shard(index.shard_path(*index.shard_for_tensor(tid_name)));
        const auto* info = require_tensor(shard, tid_name);
        const auto* ids = reinterpret_cast<const int64_t*>(shard.tensor_data(*info));
        const int count = static_cast<int>(std::min<uint64_t>(info->shape[1], k));
        std::vector<RoutedExpert> out(count);
        const float weight = route_scale / static_cast<float>(count);
        for (int i = 0; i < count; ++i) out[i] = RoutedExpert{static_cast<int>(ids[static_cast<size_t>(token) * info->shape[1] + i]), weight};
        return out;
    }

    const std::string weight_name = prefix + "ffn.gate.weight";
    SafeTensorsShard weight_shard(index.shard_path(*index.shard_for_tensor(weight_name)));
    const auto* weight = require_tensor(weight_shard, weight_name);
    const auto* w = reinterpret_cast<const uint16_t*>(weight_shard.tensor_data(*weight));
    const float* b = nullptr;
    const std::string bias_name = prefix + "ffn.gate.bias";
    const std::string* bias_shard_name = index.shard_for_tensor(bias_name);
    SafeTensorsShard bias_shard(index.shard_path(bias_shard_name == nullptr ? *index.shard_for_tensor(weight_name) : *bias_shard_name));
    if (bias_shard_name != nullptr) {
        const auto* bias = bias_shard.find_tensor(bias_name);
        b = bias == nullptr ? nullptr : reinterpret_cast<const float*>(bias_shard.tensor_data(*bias));
    }

    const int experts = static_cast<int>(std::min<uint64_t>(n_experts, weight->shape[0]));
    const int dim = static_cast<int>(weight->shape[1]);
    std::vector<float> original(experts);
    std::vector<float> scored(experts);
    for (int e = 0; e < experts; ++e) {
        float dot = 0.0f;
        for (int i = 0; i < dim; ++i) dot += ffn_norm[i] * bf16_to_float(w[static_cast<size_t>(e) * dim + i]);
        original[e] = std::sqrt(std::log1pf(std::exp(dot)));
        scored[e] = original[e] + (b == nullptr ? 0.0f : b[e]);
    }
    std::vector<int> order(experts);
    std::iota(order.begin(), order.end(), 0);
    const int count = std::min(k, experts);
    std::partial_sort(order.begin(), order.begin() + count, order.end(), [&](int a, int bidx) { return scored[a] > scored[bidx]; });
    float denom = 0.0f;
    for (int i = 0; i < count; ++i) denom += original[order[i]];
    if (denom == 0.0f) denom = 1.0f;
    std::vector<RoutedExpert> out(count);
    for (int i = 0; i < count; ++i) out[i] = RoutedExpert{order[i], original[order[i]] / denom * route_scale};
    return out;
}

Fp4Handle open_fp4(const SafeTensorsIndex& index, const std::string& name) {
    const std::string* shard_name = index.shard_for_tensor(name);
    if (shard_name == nullptr) throw std::runtime_error("missing FP4 tensor: " + name);
    Fp4Handle h{SafeTensorsShard(index.shard_path(*shard_name)), nullptr, nullptr, {}};
    h.pair = resolve_fp4_tensor_pair(index, h.shard, name);
    h.w = h.shard.find_tensor(h.pair.weight_name);
    h.s = h.shard.find_tensor(h.pair.scale_name);
    return h;
}

}  // namespace

Dsv4Engine::Dsv4Engine(const std::string& model_path) : gguf_(model_path), config_(ModelConfig::from_gguf(gguf_)) {}

ForwardSmokeResult run_safetensors_min_layer_smoke(const std::string& ckpt_dir) {
    return run_safetensors_layer_loop_smoke(ckpt_dir, 1);
}

ForwardSmokeResult run_safetensors_layer_loop_smoke(const std::string& ckpt_dir, int layer_count) {
    if (!cuda_runtime_available()) throw std::runtime_error("CUDA runtime is not available");
    SafeTensorsIndex index(ckpt_dir);
    ModelConfig config = ModelConfig::from_hf_config(ckpt_dir);
    if (layer_count <= 0) layer_count = 1;
    if (config.n_layers > 0) layer_count = std::min(layer_count, static_cast<int>(config.n_layers));

    SafeTensorsShard embed_shard(index.shard_path(*index.shard_for_tensor("embed.weight")));
    SafeTensorsShard head_shard(index.shard_path(*index.shard_for_tensor("head.weight")));
    const auto* embed = require_tensor(embed_shard, "embed.weight");
    const auto* head = require_tensor(head_shard, "head.weight");
    Fp4Handle first_w1 = open_fp4(index, "layers.0.ffn.experts.0.w1.weight");
    Fp4Handle first_w2 = open_fp4(index, "layers.0.ffn.experts.0.w2.weight");
    Fp4Handle first_w3 = open_fp4(index, "layers.0.ffn.experts.0.w3.weight");

    const int token = 1234;
    const int dim = static_cast<int>(embed->shape[1]);
    const int q_a_dim = 1024;
    const int q_dim = 32768;
    const int kv_dim = 512;
    const int attn_mid = 8192;
    const int inter = static_cast<int>(first_w1.pair.rows);
    const int head_rows = static_cast<int>(head->shape[0]);

    uint16_t* d_embed = nullptr;
    uint16_t* d_attn_gamma = nullptr;
    uint8_t* d_wq_a = nullptr;
    uint8_t* d_wq_a_scale = nullptr;
    uint16_t* d_q_gamma = nullptr;
    uint8_t* d_wq_b = nullptr;
    uint8_t* d_wq_b_scale = nullptr;
    uint8_t* d_wkv = nullptr;
    uint8_t* d_wkv_scale = nullptr;
    uint16_t* d_kv_gamma = nullptr;
    uint8_t* d_wo_a = nullptr;
    uint8_t* d_wo_a_scale = nullptr;
    uint8_t* d_wo_b = nullptr;
    uint8_t* d_wo_b_scale = nullptr;
    uint16_t* d_ffn_gamma = nullptr;
    uint8_t* d_w1 = nullptr;
    uint8_t* d_s1 = nullptr;
    uint8_t* d_w2 = nullptr;
    uint8_t* d_s2 = nullptr;
    uint8_t* d_w3 = nullptr;
    uint8_t* d_s3 = nullptr;
    uint16_t* d_head = nullptr;
    float* d_x = nullptr;
    float* d_attn_norm = nullptr;
    float* d_q_a = nullptr;
    float* d_q_norm = nullptr;
    float* d_q = nullptr;
    float* d_kv_a = nullptr;
    float* d_kv_norm = nullptr;
    float* d_attn_value = nullptr;
    float* d_attn_mid = nullptr;
    float* d_attn_out = nullptr;
    float* d_resid1 = nullptr;
    float* d_ffn_norm = nullptr;
    float* d_gate = nullptr;
    float* d_up = nullptr;
    float* d_hidden = nullptr;
    float* d_moe = nullptr;
    float* d_resid2 = nullptr;
    float* d_logits = nullptr;

    const auto* embed_data = reinterpret_cast<const uint16_t*>(embed_shard.tensor_data(*embed)) + static_cast<size_t>(token) * dim;
    check_cuda(cudaMalloc(&d_embed, static_cast<size_t>(dim) * sizeof(uint16_t)), "cudaMalloc embed");
    check_cuda(cudaMalloc(&d_attn_gamma, static_cast<size_t>(dim) * sizeof(uint16_t)), "cudaMalloc attn gamma");
    check_cuda(cudaMalloc(&d_wq_a, static_cast<size_t>(q_a_dim) * dim), "cudaMalloc wq_a");
    check_cuda(cudaMalloc(&d_wq_a_scale, static_cast<size_t>(q_a_dim / 128) * (dim / 128)), "cudaMalloc wq_a scale");
    check_cuda(cudaMalloc(&d_q_gamma, static_cast<size_t>(q_a_dim) * sizeof(uint16_t)), "cudaMalloc q gamma");
    check_cuda(cudaMalloc(&d_wq_b, static_cast<size_t>(q_dim) * q_a_dim), "cudaMalloc wq_b");
    check_cuda(cudaMalloc(&d_wq_b_scale, static_cast<size_t>(q_dim / 128) * (q_a_dim / 128)), "cudaMalloc wq_b scale");
    check_cuda(cudaMalloc(&d_wkv, static_cast<size_t>(kv_dim) * dim), "cudaMalloc wkv");
    check_cuda(cudaMalloc(&d_wkv_scale, static_cast<size_t>(kv_dim / 128) * (dim / 128)), "cudaMalloc wkv scale");
    check_cuda(cudaMalloc(&d_kv_gamma, static_cast<size_t>(kv_dim) * sizeof(uint16_t)), "cudaMalloc kv gamma");
    check_cuda(cudaMalloc(&d_wo_a, static_cast<size_t>(attn_mid) * dim), "cudaMalloc wo_a");
    check_cuda(cudaMalloc(&d_wo_a_scale, static_cast<size_t>(attn_mid / 128) * (dim / 128)), "cudaMalloc wo_a scale");
    check_cuda(cudaMalloc(&d_wo_b, static_cast<size_t>(dim) * attn_mid), "cudaMalloc wo_b");
    check_cuda(cudaMalloc(&d_wo_b_scale, static_cast<size_t>(dim / 128) * (attn_mid / 128)), "cudaMalloc wo_b scale");
    check_cuda(cudaMalloc(&d_ffn_gamma, static_cast<size_t>(dim) * sizeof(uint16_t)), "cudaMalloc ffn gamma");
    check_cuda(cudaMalloc(&d_w1, first_w1.w->nbytes), "cudaMalloc w1");
    check_cuda(cudaMalloc(&d_s1, first_w1.s->nbytes), "cudaMalloc s1");
    check_cuda(cudaMalloc(&d_w2, first_w2.w->nbytes), "cudaMalloc w2");
    check_cuda(cudaMalloc(&d_s2, first_w2.s->nbytes), "cudaMalloc s2");
    check_cuda(cudaMalloc(&d_w3, first_w3.w->nbytes), "cudaMalloc w3");
    check_cuda(cudaMalloc(&d_s3, first_w3.s->nbytes), "cudaMalloc s3");
    check_cuda(cudaMalloc(&d_head, static_cast<size_t>(head_rows) * dim * sizeof(uint16_t)), "cudaMalloc head");
    check_cuda(cudaMalloc(&d_x, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc x");
    check_cuda(cudaMalloc(&d_attn_norm, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc attn norm");
    check_cuda(cudaMalloc(&d_q_a, static_cast<size_t>(q_a_dim) * sizeof(float)), "cudaMalloc q_a");
    check_cuda(cudaMalloc(&d_q_norm, static_cast<size_t>(q_a_dim) * sizeof(float)), "cudaMalloc q_norm");
    check_cuda(cudaMalloc(&d_q, static_cast<size_t>(q_dim) * sizeof(float)), "cudaMalloc q");
    check_cuda(cudaMalloc(&d_kv_a, static_cast<size_t>(kv_dim) * sizeof(float)), "cudaMalloc kv_a");
    check_cuda(cudaMalloc(&d_kv_norm, static_cast<size_t>(kv_dim) * sizeof(float)), "cudaMalloc kv_norm");
    check_cuda(cudaMalloc(&d_attn_value, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc attn value");
    check_cuda(cudaMalloc(&d_attn_mid, static_cast<size_t>(attn_mid) * sizeof(float)), "cudaMalloc attn mid");
    check_cuda(cudaMalloc(&d_attn_out, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc attn out");
    check_cuda(cudaMalloc(&d_resid1, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc resid1");
    check_cuda(cudaMalloc(&d_ffn_norm, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc ffn norm");
    check_cuda(cudaMalloc(&d_gate, static_cast<size_t>(inter) * sizeof(float)), "cudaMalloc gate");
    check_cuda(cudaMalloc(&d_up, static_cast<size_t>(inter) * sizeof(float)), "cudaMalloc up");
    check_cuda(cudaMalloc(&d_hidden, static_cast<size_t>(inter) * sizeof(float)), "cudaMalloc hidden");
    check_cuda(cudaMalloc(&d_moe, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc moe");
    check_cuda(cudaMalloc(&d_resid2, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc resid2");
    check_cuda(cudaMalloc(&d_logits, static_cast<size_t>(head_rows) * sizeof(float)), "cudaMalloc logits");

    check_cuda(cudaMemcpy(d_embed, embed_data, static_cast<size_t>(dim) * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy embed");
    check_cuda(cudaMemcpy(d_head, head_shard.tensor_data(*head), static_cast<size_t>(head_rows) * dim * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy head");
    if (!bf16_row_to_float_cuda(d_embed, d_x, 0, dim)) throw std::runtime_error("embed launch failed");

    for (int li = 0; li < layer_count; ++li) {
        const std::string prefix = "layers." + std::to_string(li) + ".";
        SafeTensorsShard attn_norm_shard(index.shard_path(*index.shard_for_tensor(prefix + "attn_norm.weight")));
        SafeTensorsShard qkv_shard(index.shard_path(*index.shard_for_tensor(prefix + "attn.wq_a.weight")));
        SafeTensorsShard wo_a_shard(index.shard_path(*index.shard_for_tensor(prefix + "attn.wo_a.weight")));
        SafeTensorsShard wo_b_shard(index.shard_path(*index.shard_for_tensor(prefix + "attn.wo_b.weight")));
        SafeTensorsShard ffn_norm_shard(index.shard_path(*index.shard_for_tensor(prefix + "ffn_norm.weight")));
        const auto* attn_norm = require_tensor(attn_norm_shard, prefix + "attn_norm.weight");
        const auto* wq_a = require_tensor(qkv_shard, prefix + "attn.wq_a.weight");
        const auto* wq_a_scale = require_tensor(qkv_shard, prefix + "attn.wq_a.scale");
        const auto* q_norm = require_tensor(qkv_shard, prefix + "attn.q_norm.weight");
        const auto* wq_b = require_tensor(qkv_shard, prefix + "attn.wq_b.weight");
        const auto* wq_b_scale = require_tensor(qkv_shard, prefix + "attn.wq_b.scale");
        const auto* wkv = require_tensor(qkv_shard, prefix + "attn.wkv.weight");
        const auto* wkv_scale = require_tensor(qkv_shard, prefix + "attn.wkv.scale");
        const auto* kv_norm = require_tensor(qkv_shard, prefix + "attn.kv_norm.weight");
        const auto* wo_a = require_tensor(wo_a_shard, prefix + "attn.wo_a.weight");
        const auto* wo_a_scale = require_tensor(wo_a_shard, prefix + "attn.wo_a.scale");
        const auto* wo_b = require_tensor(wo_b_shard, prefix + "attn.wo_b.weight");
        const auto* wo_b_scale = require_tensor(wo_b_shard, prefix + "attn.wo_b.scale");
        const auto* ffn_norm = require_tensor(ffn_norm_shard, prefix + "ffn_norm.weight");

        check_cuda(cudaMemcpy(d_attn_gamma, attn_norm_shard.tensor_data(*attn_norm), attn_norm->nbytes, cudaMemcpyHostToDevice), "copy attn gamma");
        check_cuda(cudaMemcpy(d_wq_a, qkv_shard.tensor_data(*wq_a), wq_a->nbytes, cudaMemcpyHostToDevice), "copy wq_a");
        check_cuda(cudaMemcpy(d_wq_a_scale, qkv_shard.tensor_data(*wq_a_scale), wq_a_scale->nbytes, cudaMemcpyHostToDevice), "copy wq_a scale");
        check_cuda(cudaMemcpy(d_q_gamma, qkv_shard.tensor_data(*q_norm), q_norm->nbytes, cudaMemcpyHostToDevice), "copy q norm");
        check_cuda(cudaMemcpy(d_wq_b, qkv_shard.tensor_data(*wq_b), wq_b->nbytes, cudaMemcpyHostToDevice), "copy wq_b");
        check_cuda(cudaMemcpy(d_wq_b_scale, qkv_shard.tensor_data(*wq_b_scale), wq_b_scale->nbytes, cudaMemcpyHostToDevice), "copy wq_b scale");
        check_cuda(cudaMemcpy(d_wkv, qkv_shard.tensor_data(*wkv), wkv->nbytes, cudaMemcpyHostToDevice), "copy wkv");
        check_cuda(cudaMemcpy(d_wkv_scale, qkv_shard.tensor_data(*wkv_scale), wkv_scale->nbytes, cudaMemcpyHostToDevice), "copy wkv scale");
        check_cuda(cudaMemcpy(d_kv_gamma, qkv_shard.tensor_data(*kv_norm), kv_norm->nbytes, cudaMemcpyHostToDevice), "copy kv norm");
        check_cuda(cudaMemcpy(d_wo_a, wo_a_shard.tensor_data(*wo_a), static_cast<size_t>(attn_mid) * dim, cudaMemcpyHostToDevice), "copy wo_a");
        check_cuda(cudaMemcpy(d_wo_a_scale, wo_a_shard.tensor_data(*wo_a_scale), static_cast<size_t>(attn_mid / 128) * (dim / 128), cudaMemcpyHostToDevice), "copy wo_a scale");
        check_cuda(cudaMemcpy(d_wo_b, wo_b_shard.tensor_data(*wo_b), static_cast<size_t>(dim) * attn_mid, cudaMemcpyHostToDevice), "copy wo_b");
        check_cuda(cudaMemcpy(d_wo_b_scale, wo_b_shard.tensor_data(*wo_b_scale), static_cast<size_t>(dim / 128) * (attn_mid / 128), cudaMemcpyHostToDevice), "copy wo_b scale");
        check_cuda(cudaMemcpy(d_ffn_gamma, ffn_norm_shard.tensor_data(*ffn_norm), ffn_norm->nbytes, cudaMemcpyHostToDevice), "copy ffn gamma");

        if (!rmsnorm_bf16_gamma_cuda(d_x, d_attn_gamma, d_attn_norm, dim, 1e-6f)) throw std::runtime_error("attn norm launch failed");
        if (!fp8_e4m3_e8m0_matvec_cuda(d_attn_norm, d_wq_a, d_wq_a_scale, d_q_a, q_a_dim, dim)) throw std::runtime_error("wq_a launch failed");
        if (!rmsnorm_bf16_gamma_cuda(d_q_a, d_q_gamma, d_q_norm, q_a_dim, 1e-6f)) throw std::runtime_error("q norm launch failed");
        if (!fp8_e4m3_e8m0_matvec_cuda(d_q_norm, d_wq_b, d_wq_b_scale, d_q, q_dim, q_a_dim)) throw std::runtime_error("wq_b launch failed");
        if (!fp8_e4m3_e8m0_matvec_cuda(d_attn_norm, d_wkv, d_wkv_scale, d_kv_a, kv_dim, dim)) throw std::runtime_error("wkv launch failed");
        if (!rmsnorm_bf16_gamma_cuda(d_kv_a, d_kv_gamma, d_kv_norm, kv_dim, 1e-6f)) throw std::runtime_error("kv norm launch failed");
        if (!repeat_vector_cuda(d_kv_norm, d_attn_value, kv_dim, dim / kv_dim)) throw std::runtime_error("single-token attention value expand failed");
        if (!fp8_e4m3_e8m0_matvec_cuda(d_attn_value, d_wo_a, d_wo_a_scale, d_attn_mid, attn_mid, dim)) throw std::runtime_error("wo_a launch failed");
        if (!fp8_e4m3_e8m0_matvec_cuda(d_attn_mid, d_wo_b, d_wo_b_scale, d_attn_out, dim, attn_mid)) throw std::runtime_error("wo_b launch failed");
        if (!vector_add_cuda(d_x, d_attn_out, d_resid1, dim)) throw std::runtime_error("resid1 launch failed");
        if (!rmsnorm_bf16_gamma_cuda(d_resid1, d_ffn_gamma, d_ffn_norm, dim, 1e-6f)) throw std::runtime_error("ffn norm launch failed");
        std::vector<float> ffn_norm_host(dim);
        check_cuda(cudaMemcpy(ffn_norm_host.data(), d_ffn_norm, static_cast<size_t>(dim) * sizeof(float), cudaMemcpyDeviceToHost), "copy ffn norm for gate");
        const auto routed = select_smoke_experts(
            index, prefix, li, token, ffn_norm_host, config.n_hash_layers, config.n_routed_experts, config.n_activated_experts, static_cast<float>(config.route_scale));
        check_cuda(cudaMemset(d_moe, 0, static_cast<size_t>(dim) * sizeof(float)), "zero moe");
        for (const RoutedExpert& route : routed) {
            Fp4Handle w1 = open_fp4(index, prefix + "ffn.experts." + std::to_string(route.id) + ".w1.weight");
            Fp4Handle w2 = open_fp4(index, prefix + "ffn.experts." + std::to_string(route.id) + ".w2.weight");
            Fp4Handle w3 = open_fp4(index, prefix + "ffn.experts." + std::to_string(route.id) + ".w3.weight");
            check_cuda(cudaMemcpy(d_w1, w1.shard.tensor_data(*w1.w), w1.w->nbytes, cudaMemcpyHostToDevice), "copy selected w1");
            check_cuda(cudaMemcpy(d_s1, w1.shard.tensor_data(*w1.s), w1.s->nbytes, cudaMemcpyHostToDevice), "copy selected s1");
            check_cuda(cudaMemcpy(d_w2, w2.shard.tensor_data(*w2.w), w2.w->nbytes, cudaMemcpyHostToDevice), "copy selected w2");
            check_cuda(cudaMemcpy(d_s2, w2.shard.tensor_data(*w2.s), w2.s->nbytes, cudaMemcpyHostToDevice), "copy selected s2");
            check_cuda(cudaMemcpy(d_w3, w3.shard.tensor_data(*w3.w), w3.w->nbytes, cudaMemcpyHostToDevice), "copy selected w3");
            check_cuda(cudaMemcpy(d_s3, w3.shard.tensor_data(*w3.s), w3.s->nbytes, cudaMemcpyHostToDevice), "copy selected s3");
            if (!fp4_e2m1_e8m0_matvec_cuda(d_ffn_norm, d_w1, d_s1, d_gate, inter, dim)) throw std::runtime_error("w1 launch failed");
            if (!fp4_e2m1_e8m0_matvec_cuda(d_ffn_norm, d_w3, d_s3, d_up, inter, dim)) throw std::runtime_error("w3 launch failed");
            if (!silu_mul_cuda(d_gate, d_up, d_hidden, inter)) throw std::runtime_error("silu launch failed");
            if (!fp4_e2m1_e8m0_matvec_cuda(d_hidden, d_w2, d_s2, d_resid2, dim, inter)) throw std::runtime_error("w2 launch failed");
            if (!vector_accum_cuda(d_resid2, d_moe, dim, route.weight)) throw std::runtime_error("moe accum failed");
        }
        if (!vector_add_cuda(d_resid1, d_moe, d_resid2, dim)) throw std::runtime_error("resid2 launch failed");
        check_cuda(cudaMemcpy(d_x, d_resid2, static_cast<size_t>(dim) * sizeof(float), cudaMemcpyDeviceToDevice), "advance residual");
    }

    if (!bf16_matvec_cuda(d_x, d_head, d_logits, head_rows, dim)) throw std::runtime_error("head launch failed");
    check_cuda(cudaDeviceSynchronize(), "sync kernels");

    std::vector<float> logits(head_rows);
    check_cuda(cudaMemcpy(logits.data(), d_logits, logits.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy logits");
    float checksum = 0.0f;
    int top_token = 0;
    float top_logit = -INFINITY;
    for (int i = 0; i < head_rows; ++i) {
        const float v = logits[i];
        checksum += v;
        if (v > top_logit) {
            top_logit = v;
            top_token = i;
        }
    }
    if (!std::isfinite(checksum) || !std::isfinite(top_logit)) throw std::runtime_error("non-finite smoke logits");

    cudaFree(d_embed);
    cudaFree(d_attn_gamma);
    cudaFree(d_wq_a);
    cudaFree(d_wq_a_scale);
    cudaFree(d_q_gamma);
    cudaFree(d_wq_b);
    cudaFree(d_wq_b_scale);
    cudaFree(d_wkv);
    cudaFree(d_wkv_scale);
    cudaFree(d_kv_gamma);
    cudaFree(d_wo_a);
    cudaFree(d_wo_a_scale);
    cudaFree(d_wo_b);
    cudaFree(d_wo_b_scale);
    cudaFree(d_ffn_gamma);
    cudaFree(d_w1);
    cudaFree(d_s1);
    cudaFree(d_w2);
    cudaFree(d_s2);
    cudaFree(d_w3);
    cudaFree(d_s3);
    cudaFree(d_head);
    cudaFree(d_x);
    cudaFree(d_attn_norm);
    cudaFree(d_q_a);
    cudaFree(d_q_norm);
    cudaFree(d_q);
    cudaFree(d_kv_a);
    cudaFree(d_kv_norm);
    cudaFree(d_attn_value);
    cudaFree(d_attn_mid);
    cudaFree(d_attn_out);
    cudaFree(d_resid1);
    cudaFree(d_ffn_norm);
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_hidden);
    cudaFree(d_moe);
    cudaFree(d_resid2);
    cudaFree(d_logits);

    return ForwardSmokeResult{token, dim, inter, head_rows, layer_count, top_token, top_logit, checksum};
}

}  // namespace dsv4
