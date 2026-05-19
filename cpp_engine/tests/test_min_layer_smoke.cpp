#include "cuda_ops.hpp"
#include "safetensors_reader.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

const dsv4::SafeTensorInfo* tensor(const dsv4::SafeTensorsShard& shard, const std::string& name) {
    const auto* info = shard.find_tensor(name);
    if (info == nullptr) throw std::runtime_error("missing tensor: " + name);
    return info;
}

struct Fp4Handle {
    dsv4::SafeTensorsShard shard;
    const dsv4::SafeTensorInfo* w;
    const dsv4::SafeTensorInfo* s;
    dsv4::SafeFp4TensorPair pair;
};

Fp4Handle open_fp4(const dsv4::SafeTensorsIndex& index, const std::string& name) {
    const std::string* shard_name = index.shard_for_tensor(name);
    if (shard_name == nullptr) throw std::runtime_error("missing FP4 tensor: " + name);
    Fp4Handle h{dsv4::SafeTensorsShard(index.shard_path(*shard_name)), nullptr, nullptr, {}};
    h.pair = dsv4::resolve_fp4_tensor_pair(index, h.shard, name);
    h.w = h.shard.find_tensor(h.pair.weight_name);
    h.s = h.shard.find_tensor(h.pair.scale_name);
    return h;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: test_min_layer_smoke <ckpt_dir>\n";
            return 2;
        }
        if (!dsv4::cuda_runtime_available()) {
            std::cout << "[SKIP] CUDA runtime is not available\n";
            return 0;
        }
        const std::string ckpt = argv[1];
        dsv4::SafeTensorsIndex index(ckpt);

        dsv4::SafeTensorsShard embed_shard(index.shard_path(*index.shard_for_tensor("embed.weight")));
        dsv4::SafeTensorsShard attn_norm_shard(index.shard_path(*index.shard_for_tensor("layers.0.attn_norm.weight")));
        dsv4::SafeTensorsShard wkv_shard(index.shard_path(*index.shard_for_tensor("layers.0.attn.wkv.weight")));
        dsv4::SafeTensorsShard ffn_norm_shard(index.shard_path(*index.shard_for_tensor("layers.0.ffn_norm.weight")));
        dsv4::SafeTensorsShard head_shard(index.shard_path(*index.shard_for_tensor("head.weight")));

        const auto* embed = tensor(embed_shard, "embed.weight");
        const auto* attn_norm = tensor(attn_norm_shard, "layers.0.attn_norm.weight");
        const auto* wkv = tensor(wkv_shard, "layers.0.attn.wkv.weight");
        const auto* wkv_scale = tensor(wkv_shard, "layers.0.attn.wkv.scale");
        const auto* ffn_norm = tensor(ffn_norm_shard, "layers.0.ffn_norm.weight");
        const auto* head = tensor(head_shard, "head.weight");
        Fp4Handle w1 = open_fp4(index, "layers.0.ffn.experts.0.w1.weight");
        Fp4Handle w2 = open_fp4(index, "layers.0.ffn.experts.0.w2.weight");
        Fp4Handle w3 = open_fp4(index, "layers.0.ffn.experts.0.w3.weight");

        const int token = 1234;
        const int dim = static_cast<int>(embed->shape[1]);
        const int attn_rows = 128;
        const int inter = static_cast<int>(w1.pair.rows);
        const int head_rows = 128;

        uint16_t* d_embed = nullptr;
        uint16_t* d_attn_gamma = nullptr;
        uint8_t* d_wkv = nullptr;
        uint8_t* d_wkv_scale = nullptr;
        uint16_t* d_ffn_gamma = nullptr;
        uint8_t* d_w1 = nullptr; uint8_t* d_s1 = nullptr;
        uint8_t* d_w2 = nullptr; uint8_t* d_s2 = nullptr;
        uint8_t* d_w3 = nullptr; uint8_t* d_s3 = nullptr;
        uint16_t* d_head = nullptr;
        float* d_x = nullptr;
        float* d_attn_norm = nullptr;
        float* d_attn_proj = nullptr;
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
        check_cuda(cudaMalloc(&d_attn_gamma, attn_norm->nbytes), "cudaMalloc attn gamma");
        check_cuda(cudaMalloc(&d_wkv, static_cast<size_t>(attn_rows) * dim), "cudaMalloc wkv");
        check_cuda(cudaMalloc(&d_wkv_scale, static_cast<size_t>(attn_rows / 128) * (dim / 128)), "cudaMalloc wkv scale");
        check_cuda(cudaMalloc(&d_ffn_gamma, ffn_norm->nbytes), "cudaMalloc ffn gamma");
        check_cuda(cudaMalloc(&d_w1, w1.w->nbytes), "cudaMalloc w1"); check_cuda(cudaMalloc(&d_s1, w1.s->nbytes), "cudaMalloc s1");
        check_cuda(cudaMalloc(&d_w2, w2.w->nbytes), "cudaMalloc w2"); check_cuda(cudaMalloc(&d_s2, w2.s->nbytes), "cudaMalloc s2");
        check_cuda(cudaMalloc(&d_w3, w3.w->nbytes), "cudaMalloc w3"); check_cuda(cudaMalloc(&d_s3, w3.s->nbytes), "cudaMalloc s3");
        check_cuda(cudaMalloc(&d_head, static_cast<size_t>(head_rows) * dim * sizeof(uint16_t)), "cudaMalloc head");
        check_cuda(cudaMalloc(&d_x, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc x");
        check_cuda(cudaMalloc(&d_attn_norm, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc attn norm");
        check_cuda(cudaMalloc(&d_attn_proj, static_cast<size_t>(attn_rows) * sizeof(float)), "cudaMalloc attn proj");
        check_cuda(cudaMalloc(&d_resid1, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc resid1");
        check_cuda(cudaMalloc(&d_ffn_norm, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc ffn norm");
        check_cuda(cudaMalloc(&d_gate, static_cast<size_t>(inter) * sizeof(float)), "cudaMalloc gate");
        check_cuda(cudaMalloc(&d_up, static_cast<size_t>(inter) * sizeof(float)), "cudaMalloc up");
        check_cuda(cudaMalloc(&d_hidden, static_cast<size_t>(inter) * sizeof(float)), "cudaMalloc hidden");
        check_cuda(cudaMalloc(&d_moe, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc moe");
        check_cuda(cudaMalloc(&d_resid2, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc resid2");
        check_cuda(cudaMalloc(&d_logits, static_cast<size_t>(head_rows) * sizeof(float)), "cudaMalloc logits");

        check_cuda(cudaMemcpy(d_embed, embed_data, static_cast<size_t>(dim) * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy embed");
        check_cuda(cudaMemcpy(d_attn_gamma, attn_norm_shard.tensor_data(*attn_norm), attn_norm->nbytes, cudaMemcpyHostToDevice), "copy attn gamma");
        check_cuda(cudaMemcpy(d_wkv, wkv_shard.tensor_data(*wkv), static_cast<size_t>(attn_rows) * dim, cudaMemcpyHostToDevice), "copy wkv");
        check_cuda(cudaMemcpy(d_wkv_scale, wkv_shard.tensor_data(*wkv_scale), static_cast<size_t>(attn_rows / 128) * (dim / 128), cudaMemcpyHostToDevice), "copy wkv scale");
        check_cuda(cudaMemcpy(d_ffn_gamma, ffn_norm_shard.tensor_data(*ffn_norm), ffn_norm->nbytes, cudaMemcpyHostToDevice), "copy ffn gamma");
        check_cuda(cudaMemcpy(d_w1, w1.shard.tensor_data(*w1.w), w1.w->nbytes, cudaMemcpyHostToDevice), "copy w1"); check_cuda(cudaMemcpy(d_s1, w1.shard.tensor_data(*w1.s), w1.s->nbytes, cudaMemcpyHostToDevice), "copy s1");
        check_cuda(cudaMemcpy(d_w2, w2.shard.tensor_data(*w2.w), w2.w->nbytes, cudaMemcpyHostToDevice), "copy w2"); check_cuda(cudaMemcpy(d_s2, w2.shard.tensor_data(*w2.s), w2.s->nbytes, cudaMemcpyHostToDevice), "copy s2");
        check_cuda(cudaMemcpy(d_w3, w3.shard.tensor_data(*w3.w), w3.w->nbytes, cudaMemcpyHostToDevice), "copy w3"); check_cuda(cudaMemcpy(d_s3, w3.shard.tensor_data(*w3.s), w3.s->nbytes, cudaMemcpyHostToDevice), "copy s3");
        check_cuda(cudaMemcpy(d_head, head_shard.tensor_data(*head), static_cast<size_t>(head_rows) * dim * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy head");

        if (!dsv4::bf16_row_to_float_cuda(d_embed, d_x, 0, dim)) throw std::runtime_error("embed launch failed");
        if (!dsv4::rmsnorm_bf16_gamma_cuda(d_x, d_attn_gamma, d_attn_norm, dim, 1e-6f)) throw std::runtime_error("attn norm launch failed");
        if (!dsv4::fp8_e4m3_e8m0_matvec_cuda(d_attn_norm, d_wkv, d_wkv_scale, d_attn_proj, attn_rows, dim)) throw std::runtime_error("wkv launch failed");
        if (!dsv4::vector_add_cuda(d_x, d_x, d_resid1, dim)) throw std::runtime_error("resid1 launch failed");
        if (!dsv4::rmsnorm_bf16_gamma_cuda(d_resid1, d_ffn_gamma, d_ffn_norm, dim, 1e-6f)) throw std::runtime_error("ffn norm launch failed");
        if (!dsv4::fp4_e2m1_e8m0_matvec_cuda(d_ffn_norm, d_w1, d_s1, d_gate, inter, dim)) throw std::runtime_error("w1 launch failed");
        if (!dsv4::fp4_e2m1_e8m0_matvec_cuda(d_ffn_norm, d_w3, d_s3, d_up, inter, dim)) throw std::runtime_error("w3 launch failed");
        if (!dsv4::silu_mul_cuda(d_gate, d_up, d_hidden, inter)) throw std::runtime_error("silu launch failed");
        if (!dsv4::fp4_e2m1_e8m0_matvec_cuda(d_hidden, d_w2, d_s2, d_moe, dim, inter)) throw std::runtime_error("w2 launch failed");
        if (!dsv4::vector_add_cuda(d_resid1, d_moe, d_resid2, dim)) throw std::runtime_error("resid2 launch failed");
        if (!dsv4::bf16_matvec_cuda(d_resid2, d_head, d_logits, head_rows, dim)) throw std::runtime_error("head launch failed");
        check_cuda(cudaDeviceSynchronize(), "sync kernels");

        std::vector<float> logits(head_rows);
        check_cuda(cudaMemcpy(logits.data(), d_logits, logits.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy logits");
        float checksum = 0.0f;
        for (float v : logits) checksum += v;
        if (!std::isfinite(checksum)) throw std::runtime_error("non-finite smoke checksum");
        std::cout << "[PASS] min_layer_smoke token=" << token << " dim=" << dim << " inter=" << inter << " logits=" << head_rows << " checksum=" << checksum << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
