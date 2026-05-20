#include "dsv4_engine.hpp"

#include "cuda_ops.hpp"
#include "safetensors_reader.hpp"
#include "tp_comm.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <chrono>
#include <string>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <memory>
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

struct Fp4View {
    SafeTensorsShard* shard = nullptr;
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

float round_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1);
    bits &= 0xffff0000u;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

void round_vector_to_bf16(std::vector<float>& values) {
    for (float& v : values) v = round_to_bf16(v);
}

std::vector<uint8_t> slice_rows_u8(const uint8_t* src, int row_start, int rows, int cols) {
    std::vector<uint8_t> out(static_cast<size_t>(rows) * cols);
    std::memcpy(out.data(), src + static_cast<size_t>(row_start) * cols, out.size());
    return out;
}

std::vector<float> slice_rows_f32(const float* src, int row_start, int rows) {
    std::vector<float> out(rows);
    std::memcpy(out.data(), src + row_start, static_cast<size_t>(rows) * sizeof(float));
    return out;
}

std::vector<uint8_t> slice_cols_u8(const uint8_t* src, int rows, int cols, int col_start, int col_count) {
    std::vector<uint8_t> out(static_cast<size_t>(rows) * col_count);
    for (int r = 0; r < rows; ++r) {
        std::memcpy(out.data() + static_cast<size_t>(r) * col_count, src + static_cast<size_t>(r) * cols + col_start, static_cast<size_t>(col_count));
    }
    return out;
}

struct RoutedExpert {
    int id = 0;
    float weight = 0.0f;
};

struct HcPreResult {
    std::vector<float> x;
    float post[4] = {};
    float comb[16] = {};
};

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

HcPreResult hc_pre_cpu(const std::vector<float>& h4, const float* fn, const float* scale, const float* base, int dim) {
    constexpr int hc = 4;
    constexpr int mix = 24;
    constexpr float eps = 1e-6f;
    HcPreResult out;
    out.x.assign(dim, 0.0f);
    double sum_sq = 0.0;
    for (float v : h4) sum_sq += static_cast<double>(v) * v;
    const float rsqrt = 1.0f / std::sqrt(static_cast<float>(sum_sq / h4.size()) + eps);
    float mixes[mix];
    for (int r = 0; r < mix; ++r) {
        double dot = 0.0;
        const float* row = fn + static_cast<size_t>(r) * h4.size();
        for (size_t i = 0; i < h4.size(); ++i) dot += static_cast<double>(row[i]) * h4[i];
        mixes[r] = static_cast<float>(dot) * rsqrt;
    }
    float pre[hc];
    for (int i = 0; i < hc; ++i) {
        pre[i] = sigmoid(mixes[i] * scale[0] + base[i]) + eps;
        out.post[i] = 2.0f * sigmoid(mixes[hc + i] * scale[1] + base[hc + i]);
    }
    for (int r = 0; r < hc; ++r) {
        float row_max = -INFINITY;
        for (int c = 0; c < hc; ++c) row_max = std::max(row_max, mixes[2 * hc + r * hc + c] * scale[2] + base[2 * hc + r * hc + c]);
        float denom = 0.0f;
        for (int c = 0; c < hc; ++c) {
            float v = std::exp(mixes[2 * hc + r * hc + c] * scale[2] + base[2 * hc + r * hc + c] - row_max) + eps;
            out.comb[r * hc + c] = v;
            denom += v;
        }
        for (int c = 0; c < hc; ++c) out.comb[r * hc + c] /= denom;
    }
    for (int c = 0; c < hc; ++c) {
        float denom = eps;
        for (int r = 0; r < hc; ++r) denom += out.comb[r * hc + c];
        for (int r = 0; r < hc; ++r) out.comb[r * hc + c] /= denom;
    }
    for (int iter = 1; iter < 20; ++iter) {
        for (int r = 0; r < hc; ++r) {
            float denom = eps;
            for (int c = 0; c < hc; ++c) denom += out.comb[r * hc + c];
            for (int c = 0; c < hc; ++c) out.comb[r * hc + c] /= denom;
        }
        for (int c = 0; c < hc; ++c) {
            float denom = eps;
            for (int r = 0; r < hc; ++r) denom += out.comb[r * hc + c];
            for (int r = 0; r < hc; ++r) out.comb[r * hc + c] /= denom;
        }
    }
    for (int m = 0; m < hc; ++m) {
        for (int d = 0; d < dim; ++d) out.x[d] += pre[m] * h4[static_cast<size_t>(m) * dim + d];
    }
    return out;
}

std::vector<float> hc_post_cpu(const std::vector<float>& x, const std::vector<float>& residual, const HcPreResult& pre, int dim) {
    constexpr int hc = 4;
    std::vector<float> out(static_cast<size_t>(hc) * dim, 0.0f);
    for (int m = 0; m < hc; ++m) {
        for (int d = 0; d < dim; ++d) {
            float v = pre.post[m] * x[d];
            for (int j = 0; j < hc; ++j) v += pre.comb[j * hc + m] * residual[static_cast<size_t>(j) * dim + d];
            out[static_cast<size_t>(m) * dim + d] = v;
        }
    }
    return out;
}

std::vector<float> hc_head_cpu(const std::vector<float>& h4, const float* fn, const float* scale, const float* base, int dim) {
    constexpr int hc = 4;
    constexpr float eps = 1e-6f;
    double sum_sq = 0.0;
    for (float v : h4) sum_sq += static_cast<double>(v) * v;
    const float rsqrt = 1.0f / std::sqrt(static_cast<float>(sum_sq / h4.size()) + eps);
    float pre[hc];
    for (int m = 0; m < hc; ++m) {
        double dot = 0.0;
        const float* row = fn + static_cast<size_t>(m) * h4.size();
        for (size_t i = 0; i < h4.size(); ++i) dot += static_cast<double>(row[i]) * h4[i];
        pre[m] = sigmoid(static_cast<float>(dot) * rsqrt * scale[0] + base[m]) + eps;
    }
    std::vector<float> out(dim, 0.0f);
    for (int m = 0; m < hc; ++m) {
        for (int d = 0; d < dim; ++d) out[d] += pre[m] * h4[static_cast<size_t>(m) * dim + d];
    }
    return out;
}

std::vector<float> bf16_matvec_cpu(const std::vector<float>& x, const uint16_t* w, int rows, int cols) {
    std::vector<float> y(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (int c = 0; c < cols; ++c) sum += static_cast<double>(bf16_to_float(w[static_cast<size_t>(r) * cols + c])) * x[c];
        y[r] = static_cast<float>(sum);
    }
    return y;
}

std::vector<float> rmsnorm_cpu(const std::vector<float>& x, const uint16_t* gamma, float eps) {
    double sum_sq = 0.0;
    for (float v : x) sum_sq += static_cast<double>(v) * v;
    const float inv = 1.0f / std::sqrt(static_cast<float>(sum_sq / x.size()) + eps);
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * inv * bf16_to_float(gamma[i]);
    return y;
}

bool debug_forward_enabled() {
    const char* env = std::getenv("DSV4_CPP_DEBUG_FORWARD");
    return env != nullptr && std::string(env) != "0";
}

bool profile_forward_enabled() {
    const char* env = std::getenv("DSV4_CPP_PROFILE_FORWARD");
    return env != nullptr && std::string(env) != "0";
}

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_summary(const std::string& name, const std::vector<float>& x) {
    double sum = 0.0;
    double sum_sq = 0.0;
    for (float v : x) {
        sum += v;
        sum_sq += static_cast<double>(v) * v;
    }
    const double mean = x.empty() ? 0.0 : sum / static_cast<double>(x.size());
    const double rms = x.empty() ? 0.0 : std::sqrt(sum_sq / static_cast<double>(x.size()));
    const float first = x.empty() ? 0.0f : x.front();
    const float last = x.empty() ? 0.0f : x.back();
    std::cout << "CPP " << name << " sum=" << static_cast<float>(sum)
              << " mean=" << static_cast<float>(mean)
              << " rms=" << static_cast<float>(rms)
              << " first=" << first
              << " last=" << last << "\n";
}

struct AttentionSmokeDims {
    int dim = 0;
    int q_a_dim = 0;
    int q_dim = 0;
    int kv_dim = 0;
    int heads = 0;
    int head_dim = 0;
    int groups = 0;
    int group_dim = 0;
    int group_rank = 0;
    int attn_mid = 0;
    int rope_dim = 0;
    int position = 0;
    int layer_id = 0;
    int window_size = 128;
    int cache_write_slot = 0;
    float rope_theta = 0.0f;
};

AttentionSmokeDims make_attention_dims(const ModelConfig& config, int dim, int tp_world, int position) {
    AttentionSmokeDims dims;
    dims.dim = dim;
    dims.q_a_dim = static_cast<int>(config.q_lora_rank);
    const int global_heads = static_cast<int>(config.n_heads);
    const int global_groups = static_cast<int>(config.o_groups);
    if ((global_heads % tp_world) != 0 || (global_groups % tp_world) != 0) throw std::runtime_error("attention heads/groups must divide TP world");
    dims.heads = global_heads / tp_world;
    dims.head_dim = static_cast<int>(config.head_dim);
    dims.q_dim = dims.heads * dims.head_dim;
    dims.kv_dim = static_cast<int>(config.kv_heads * config.head_dim);
    dims.groups = global_groups / tp_world;
    dims.group_rank = static_cast<int>(config.o_lora_rank);
    dims.attn_mid = dims.group_rank * dims.groups;
    dims.rope_dim = static_cast<int>(config.rope_dim);
    dims.position = position;
    dims.window_size = static_cast<int>(config.window_size == 0 ? 128 : config.window_size);
    if (dims.q_a_dim <= 0 || dims.heads <= 0 || dims.head_dim <= 0 || dims.kv_dim <= 0 || dims.groups <= 0 || dims.group_rank <= 0 || dims.rope_dim <= 0) throw std::runtime_error("invalid attention dimensions in config");
    if (dims.q_dim % dims.groups != 0) throw std::runtime_error("attention q dim must be divisible by output groups");
    if (dims.kv_dim != dims.head_dim) throw std::runtime_error("single-token attention expects one KV head");
    dims.group_dim = dims.q_dim / dims.groups;
    return dims;
}

bool run_single_token_attention_smoke(
    const AttentionSmokeDims& dims,
    const float* d_x,
    const uint16_t* d_attn_gamma,
    const uint8_t* d_wq_a,
    const uint8_t* d_wq_a_scale,
    const uint16_t* d_q_gamma,
    const uint8_t* d_wq_b,
    const uint8_t* d_wq_b_scale,
    const uint8_t* d_wkv,
    const uint8_t* d_wkv_scale,
    const uint16_t* d_kv_gamma,
    const uint8_t* d_wo_a,
    const uint8_t* d_wo_a_scale,
    const uint8_t* d_wo_b,
    const uint8_t* d_wo_b_scale,
    const float* d_attn_sink,
    float* d_kv_cache,
    const int* d_kv_indices,
    int index_count,
    int cache_len,
    float* d_attn_norm,
    float* d_q_a,
    float* d_q_norm,
    float* d_q,
    float* d_kv_a,
    float* d_kv_norm,
    float* d_attn_value,
    float* d_attn_mid,
    float* d_attn_out) {
    if (!rmsnorm_bf16_gamma_cuda(d_x, d_attn_gamma, d_attn_norm, dims.dim, 1e-6f)) return false;
    if (!fp8_e4m3_e8m0_matvec_cuda(d_attn_norm, d_wq_a, d_wq_a_scale, d_q_a, dims.q_a_dim, dims.dim)) return false;
    if (!rmsnorm_bf16_gamma_cuda(d_q_a, d_q_gamma, d_q_norm, dims.q_a_dim, 1e-6f)) return false;
    if (!fp8_e4m3_e8m0_matvec_cuda(d_q_norm, d_wq_b, d_wq_b_scale, d_q, dims.q_dim, dims.q_a_dim)) return false;
    if (!head_rmsnorm_rope_cuda(d_q, dims.heads, dims.head_dim, dims.rope_dim, dims.position, dims.rope_theta, false, 1e-6f)) return false;
    if (!fp8_e4m3_e8m0_matvec_cuda(d_attn_norm, d_wkv, d_wkv_scale, d_kv_a, dims.kv_dim, dims.dim)) return false;
    if (!rmsnorm_bf16_gamma_cuda(d_kv_a, d_kv_gamma, d_kv_norm, dims.kv_dim, 1e-6f)) return false;
    if (!head_rmsnorm_rope_cuda(d_kv_norm, 1, dims.head_dim, dims.rope_dim, dims.position, dims.rope_theta, false, 0.0f)) return false;
    if (!fp8_act_quant_dequant_cuda(d_kv_norm, dims.head_dim - dims.rope_dim, 64)) return false;
    if (d_kv_cache != nullptr) {
        if (cudaMemcpy(d_kv_cache + static_cast<size_t>(dims.cache_write_slot) * dims.head_dim, d_kv_norm, static_cast<size_t>(dims.head_dim) * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) return false;
        if (d_kv_indices != nullptr && index_count > 0) {
            if (!indexed_cached_single_token_attention_cuda(d_q, d_kv_cache, d_kv_indices, d_attn_sink, d_attn_value, dims.heads, dims.head_dim, index_count, 1.0f / std::sqrt(static_cast<float>(dims.head_dim)))) return false;
        } else if (!cached_single_token_attention_cuda(d_q, d_kv_cache, d_attn_sink, d_attn_value, dims.heads, dims.head_dim, cache_len, 1.0f / std::sqrt(static_cast<float>(dims.head_dim)))) return false;
    } else if (!single_token_sparse_attention_cuda(d_q, d_kv_norm, d_attn_sink, d_attn_value, dims.heads, dims.head_dim, 1.0f / std::sqrt(static_cast<float>(dims.head_dim)))) return false;
    if (!head_rmsnorm_rope_cuda(d_attn_value, dims.heads, dims.head_dim, dims.rope_dim, dims.position, dims.rope_theta, true, 0.0f)) return false;
    for (int g = 0; g < dims.groups; ++g) {
        const float* group_x = d_attn_value + static_cast<size_t>(g) * dims.group_dim;
        const uint8_t* group_w = d_wo_a + static_cast<size_t>(g) * dims.group_rank * dims.group_dim;
        const uint8_t* group_s = d_wo_a_scale + static_cast<size_t>(g) * (dims.group_rank / 128) * (dims.group_dim / 128);
        float* group_y = d_attn_mid + static_cast<size_t>(g) * dims.group_rank;
        if (!fp8_e4m3_e8m0_matvec_cuda(group_x, group_w, group_s, group_y, dims.group_rank, dims.group_dim)) return false;
    }
    if (!fp8_e4m3_e8m0_matvec_cuda(d_attn_mid, d_wo_b, d_wo_b_scale, d_attn_out, dims.dim, dims.attn_mid)) return false;
    return true;
}

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
    const std::string weight_name = prefix + "ffn.gate.weight";
    const std::string tid_name = prefix + "ffn.gate.tid2eid";
    if (static_cast<uint64_t>(layer_id) < n_hash_layers && index.shard_for_tensor(tid_name) != nullptr) {
        SafeTensorsShard tid_shard(index.shard_path(*index.shard_for_tensor(tid_name)));
        const auto* info = require_tensor(tid_shard, tid_name);
        const auto* ids = reinterpret_cast<const int64_t*>(tid_shard.tensor_data(*info));
        SafeTensorsShard weight_shard(index.shard_path(*index.shard_for_tensor(weight_name)));
        const auto* weight = require_tensor(weight_shard, weight_name);
        const auto* w = reinterpret_cast<const uint16_t*>(weight_shard.tensor_data(*weight));
        const int count = static_cast<int>(std::min<uint64_t>(info->shape[1], k));
        const int dim = static_cast<int>(weight->shape[1]);
        std::vector<float> original(count);
        float denom = 0.0f;
        for (int i = 0; i < count; ++i) {
            const int e = static_cast<int>(ids[static_cast<size_t>(token) * info->shape[1] + i]);
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) dot += ffn_norm[d] * bf16_to_float(w[static_cast<size_t>(e) * dim + d]);
            original[i] = std::sqrt(std::log1pf(std::exp(dot)));
            denom += original[i];
        }
        if (denom == 0.0f) denom = 1.0f;
        std::vector<RoutedExpert> out(count);
        for (int i = 0; i < count; ++i) out[i] = RoutedExpert{static_cast<int>(ids[static_cast<size_t>(token) * info->shape[1] + i]), original[i] / denom * route_scale};
        return out;
    }

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

const std::string& require_shard_name(const SafeTensorsIndex& index, const std::string& tensor) {
    const std::string* shard = index.shard_for_tensor(tensor);
    if (shard == nullptr) throw std::runtime_error("missing tensor in checkpoint index: " + tensor);
    return *shard;
}

struct DeviceAttentionCache {
    uint8_t* wq_a = nullptr;
    uint8_t* wq_a_scale = nullptr;
    uint8_t* wq_b = nullptr;
    uint8_t* wq_b_scale = nullptr;
    uint8_t* wkv = nullptr;
    uint8_t* wkv_scale = nullptr;
    uint8_t* wo_a = nullptr;
    uint8_t* wo_a_scale = nullptr;
    uint8_t* wo_b = nullptr;
    uint8_t* wo_b_scale = nullptr;
    float* attn_sink = nullptr;
};

struct DeviceSharedCache {
    uint8_t* w1 = nullptr;
    uint8_t* s1 = nullptr;
    uint8_t* w2 = nullptr;
    uint8_t* s2 = nullptr;
    uint8_t* w3 = nullptr;
    uint8_t* s3 = nullptr;
};

struct DeviceFp4ExpertCache {
    uint8_t* w1 = nullptr;
    uint8_t* s1 = nullptr;
    uint8_t* w2 = nullptr;
    uint8_t* s2 = nullptr;
    uint8_t* w3 = nullptr;
    uint8_t* s3 = nullptr;
    size_t w1_bytes = 0;
    size_t s1_bytes = 0;
    size_t w2_bytes = 0;
    size_t s2_bytes = 0;
    size_t w3_bytes = 0;
    size_t s3_bytes = 0;
};

int env_int_or_default(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    try {
        return std::stoi(value);
    } catch (...) {
        return fallback;
    }
}

struct ActiveArenaDeviceBuffers {
    uint8_t* w1 = nullptr;
    uint8_t* s1 = nullptr;
    uint8_t* w2 = nullptr;
    uint8_t* s2 = nullptr;
    uint8_t* w3 = nullptr;
    uint8_t* s3 = nullptr;
    int capacity = 0;
    size_t w1_bytes = 0;
    size_t s1_bytes = 0;
    size_t w2_bytes = 0;
    size_t s2_bytes = 0;
    size_t w3_bytes = 0;
    size_t s3_bytes = 0;
};

struct DeviceFp4ActiveArena {
    uint8_t* w1 = nullptr;
    uint8_t* s1 = nullptr;
    uint8_t* w2 = nullptr;
    uint8_t* s2 = nullptr;
    uint8_t* w3 = nullptr;
    uint8_t* s3 = nullptr;
    int capacity = 0;
    size_t w1_bytes = 0;
    size_t s1_bytes = 0;
    size_t w2_bytes = 0;
    size_t s2_bytes = 0;
    size_t w3_bytes = 0;
    size_t s3_bytes = 0;
    std::unordered_set<int> staged_local;
};

struct HostFp4ExpertSlot {
    uint8_t* h_w1q = nullptr;
    uint8_t* h_w1s = nullptr;
    uint8_t* h_w2q = nullptr;
    uint8_t* h_w2s = nullptr;
    uint8_t* h_w3q = nullptr;
    uint8_t* h_w3s = nullptr;
    size_t w1q_bytes = 0;
    size_t w1s_bytes = 0;
    size_t w2q_bytes = 0;
    size_t w2s_bytes = 0;
    size_t w3q_bytes = 0;
    size_t w3s_bytes = 0;
};

struct DeviceGateCache {
    uint16_t* weight = nullptr;
    float* bias = nullptr;
    float* original = nullptr;
    float* scored = nullptr;
    int experts = 0;
    int dim = 0;
};

struct SafeForwardContext {
    explicit SafeForwardContext(const std::string& dir)
        : ckpt_dir(dir),
          index(dir),
          config(ModelConfig::from_hf_config(dir)),
          embed_shard(index.shard_path(require_shard_name(index, "embed.weight"))),
          head_shard(index.shard_path(require_shard_name(index, "head.weight"))),
          final_norm_shard(index.shard_path(require_shard_name(index, "norm.weight"))),
          hc_head_shard(index.shard_path(require_shard_name(index, "hc_head_fn"))) {
        embed = require_tensor(embed_shard, "embed.weight");
        head = require_tensor(head_shard, "head.weight");
        final_norm = require_tensor(final_norm_shard, "norm.weight");
        hc_head_fn = require_tensor(hc_head_shard, "hc_head_fn");
        hc_head_scale = require_tensor(hc_head_shard, "hc_head_scale");
        hc_head_base = require_tensor(hc_head_shard, "hc_head_base");
    }

    ~SafeForwardContext() {
        for (auto& [_, ptr] : kv_cache) cudaFree(ptr);
        for (auto& [_, ptr] : indexer_kv_cache) cudaFree(ptr);
        for (auto& [_, c] : attention_cache) {
            cudaFree(c.wq_a);
            cudaFree(c.wq_a_scale);
            cudaFree(c.wq_b);
            cudaFree(c.wq_b_scale);
            cudaFree(c.wkv);
            cudaFree(c.wkv_scale);
            cudaFree(c.wo_a);
            cudaFree(c.wo_a_scale);
            cudaFree(c.wo_b);
            cudaFree(c.wo_b_scale);
            cudaFree(c.attn_sink);
        }
        for (auto& [_, c] : shared_cache) {
            cudaFree(c.w1);
            cudaFree(c.s1);
            cudaFree(c.w2);
            cudaFree(c.s2);
            cudaFree(c.w3);
            cudaFree(c.s3);
        }
        for (auto& [_, c] : expert_cache) {
            cudaFree(c.w1);
            cudaFree(c.s1);
            cudaFree(c.w2);
            cudaFree(c.s2);
            cudaFree(c.w3);
            cudaFree(c.s3);
        }
        for (auto& [_, c] : active_arena_cache) {
            if (c.w1 != nullptr) cudaFree(c.w1);
            if (c.s1 != nullptr) cudaFree(c.s1);
            if (c.w2 != nullptr) cudaFree(c.w2);
            if (c.s2 != nullptr) cudaFree(c.s2);
            if (c.w3 != nullptr) cudaFree(c.w3);
            if (c.s3 != nullptr) cudaFree(c.s3);
        }
        for (auto& blk : active_arena_device_freelist) {
            if (blk.w1 != nullptr) cudaFree(blk.w1);
            if (blk.s1 != nullptr) cudaFree(blk.s1);
            if (blk.w2 != nullptr) cudaFree(blk.w2);
            if (blk.s2 != nullptr) cudaFree(blk.s2);
            if (blk.w3 != nullptr) cudaFree(blk.w3);
            if (blk.s3 != nullptr) cudaFree(blk.s3);
        }
        active_arena_device_freelist.clear();
        for (auto& [_, c] : host_fp4_slot_cache) {
            if (c.h_w1q != nullptr) cudaFreeHost(c.h_w1q);
            if (c.h_w1s != nullptr) cudaFreeHost(c.h_w1s);
            if (c.h_w2q != nullptr) cudaFreeHost(c.h_w2q);
            if (c.h_w2s != nullptr) cudaFreeHost(c.h_w2s);
            if (c.h_w3q != nullptr) cudaFreeHost(c.h_w3q);
            if (c.h_w3s != nullptr) cudaFreeHost(c.h_w3s);
        }
        for (auto& [_, c] : gate_cache) {
            cudaFree(c.weight);
            cudaFree(c.bias);
            cudaFree(c.original);
            cudaFree(c.scored);
        }
    }

    SafeTensorsShard& shard_for_tensor(const std::string& tensor) {
        const std::string& shard_name = require_shard_name(index, tensor);
        auto it = shard_cache.find(shard_name);
        if (it != shard_cache.end()) return *it->second;
        auto shard = std::make_unique<SafeTensorsShard>(index.shard_path(shard_name));
        SafeTensorsShard& ref = *shard;
        shard_cache.emplace(shard_name, std::move(shard));
        return ref;
    }

    Fp4View fp4_view(const std::string& name) {
        SafeTensorsShard& shard = shard_for_tensor(name);
        SafeFp4TensorPair pair = resolve_fp4_tensor_pair(index, shard, name);
        Fp4View view;
        view.shard = &shard;
        view.pair = std::move(pair);
        view.w = shard.find_tensor(view.pair.weight_name);
        view.s = shard.find_tensor(view.pair.scale_name);
        if (view.w == nullptr || view.s == nullptr) throw std::runtime_error("missing FP4 tensor pair: " + name);
        return view;
    }

    DeviceGateCache& gate_device_cache(int layer_id) {
        const std::string key = std::to_string(layer_id);
        auto it = gate_cache.find(key);
        if (it != gate_cache.end()) return it->second;
        const std::string prefix = "layers." + std::to_string(layer_id) + ".";
        SafeTensorsShard& weight_shard = shard_for_tensor(prefix + "ffn.gate.weight");
        const auto* weight = require_tensor(weight_shard, prefix + "ffn.gate.weight");
        DeviceGateCache c;
        c.experts = static_cast<int>(weight->shape[0]);
        c.dim = static_cast<int>(weight->shape[1]);
        check_cuda(cudaMalloc(&c.weight, weight->nbytes), "cudaMalloc gate weight");
        check_cuda(cudaMemcpy(c.weight, weight_shard.tensor_data(*weight), weight->nbytes, cudaMemcpyHostToDevice), "copy gate weight");
        check_cuda(cudaMalloc(&c.original, static_cast<size_t>(c.experts) * sizeof(float)), "cudaMalloc gate original");
        check_cuda(cudaMalloc(&c.scored, static_cast<size_t>(c.experts) * sizeof(float)), "cudaMalloc gate scored");
        const std::string bias_name = prefix + "ffn.gate.bias";
        const std::string* bias_shard_name = index.shard_for_tensor(bias_name);
        if (bias_shard_name != nullptr) {
            SafeTensorsShard& bias_shard = shard_for_tensor(bias_name);
            const auto* bias = bias_shard.find_tensor(bias_name);
            if (bias != nullptr) {
                check_cuda(cudaMalloc(&c.bias, bias->nbytes), "cudaMalloc gate bias");
                check_cuda(cudaMemcpy(c.bias, bias_shard.tensor_data(*bias), bias->nbytes, cudaMemcpyHostToDevice), "copy gate bias");
            }
        }
        auto inserted = gate_cache.emplace(key, c);
        return inserted.first->second;
    }

    DeviceAttentionCache& attention_device_cache(int layer_id, int tp_world, int tp_rank, const AttentionSmokeDims& dims) {
        const std::string key = std::to_string(layer_id) + ":" + std::to_string(tp_world) + ":" + std::to_string(tp_rank);
        auto it = attention_cache.find(key);
        if (it != attention_cache.end()) return it->second;
        const std::string prefix = "layers." + std::to_string(layer_id) + ".";
        SafeTensorsShard& qkv_shard = shard_for_tensor(prefix + "attn.wq_a.weight");
        SafeTensorsShard& wo_a_shard = shard_for_tensor(prefix + "attn.wo_a.weight");
        SafeTensorsShard& wo_b_shard = shard_for_tensor(prefix + "attn.wo_b.weight");
        const auto* wq_a = require_tensor(qkv_shard, prefix + "attn.wq_a.weight");
        const auto* wq_a_scale = require_tensor(qkv_shard, prefix + "attn.wq_a.scale");
        const auto* wq_b = require_tensor(qkv_shard, prefix + "attn.wq_b.weight");
        const auto* wq_b_scale = require_tensor(qkv_shard, prefix + "attn.wq_b.scale");
        const auto* wkv = require_tensor(qkv_shard, prefix + "attn.wkv.weight");
        const auto* wkv_scale = require_tensor(qkv_shard, prefix + "attn.wkv.scale");
        const auto* attn_sink = require_tensor(qkv_shard, prefix + "attn.attn_sink");
        const auto* wo_a = require_tensor(wo_a_shard, prefix + "attn.wo_a.weight");
        const auto* wo_a_scale = require_tensor(wo_a_shard, prefix + "attn.wo_a.scale");
        const auto* wo_b = require_tensor(wo_b_shard, prefix + "attn.wo_b.weight");
        const auto* wo_b_scale = require_tensor(wo_b_shard, prefix + "attn.wo_b.scale");
        DeviceAttentionCache c;
        check_cuda(cudaMalloc(&c.wq_a, wq_a->nbytes), "cudaMalloc cached wq_a");
        check_cuda(cudaMalloc(&c.wq_a_scale, wq_a_scale->nbytes), "cudaMalloc cached wq_a scale");
        check_cuda(cudaMemcpy(c.wq_a, qkv_shard.tensor_data(*wq_a), wq_a->nbytes, cudaMemcpyHostToDevice), "copy cached wq_a");
        check_cuda(cudaMemcpy(c.wq_a_scale, qkv_shard.tensor_data(*wq_a_scale), wq_a_scale->nbytes, cudaMemcpyHostToDevice), "copy cached wq_a scale");
        const int local_head_start = tp_rank * dims.heads;
        const int q_row_start = local_head_start * dims.head_dim;
        auto wq_b_local = slice_rows_u8(reinterpret_cast<const uint8_t*>(qkv_shard.tensor_data(*wq_b)), q_row_start, dims.q_dim, dims.q_a_dim);
        auto wq_b_scale_local = slice_rows_u8(reinterpret_cast<const uint8_t*>(qkv_shard.tensor_data(*wq_b_scale)), q_row_start / 128, dims.q_dim / 128, dims.q_a_dim / 128);
        check_cuda(cudaMalloc(&c.wq_b, wq_b_local.size()), "cudaMalloc cached wq_b");
        check_cuda(cudaMalloc(&c.wq_b_scale, wq_b_scale_local.size()), "cudaMalloc cached wq_b scale");
        check_cuda(cudaMemcpy(c.wq_b, wq_b_local.data(), wq_b_local.size(), cudaMemcpyHostToDevice), "copy cached wq_b");
        check_cuda(cudaMemcpy(c.wq_b_scale, wq_b_scale_local.data(), wq_b_scale_local.size(), cudaMemcpyHostToDevice), "copy cached wq_b scale");
        check_cuda(cudaMalloc(&c.wkv, wkv->nbytes), "cudaMalloc cached wkv");
        check_cuda(cudaMalloc(&c.wkv_scale, wkv_scale->nbytes), "cudaMalloc cached wkv scale");
        check_cuda(cudaMemcpy(c.wkv, qkv_shard.tensor_data(*wkv), wkv->nbytes, cudaMemcpyHostToDevice), "copy cached wkv");
        check_cuda(cudaMemcpy(c.wkv_scale, qkv_shard.tensor_data(*wkv_scale), wkv_scale->nbytes, cudaMemcpyHostToDevice), "copy cached wkv scale");
        const int local_group_start = tp_rank * dims.groups;
        const int wo_a_row_start = local_group_start * dims.group_rank;
        auto wo_a_local = slice_rows_u8(reinterpret_cast<const uint8_t*>(wo_a_shard.tensor_data(*wo_a)), wo_a_row_start, dims.attn_mid, dims.dim);
        auto wo_a_scale_local = slice_rows_u8(reinterpret_cast<const uint8_t*>(wo_a_shard.tensor_data(*wo_a_scale)), wo_a_row_start / 128, dims.attn_mid / 128, dims.dim / 128);
        auto wo_b_local = slice_cols_u8(reinterpret_cast<const uint8_t*>(wo_b_shard.tensor_data(*wo_b)), dims.dim, static_cast<int>(config.o_lora_rank * config.o_groups), wo_a_row_start, dims.attn_mid);
        auto wo_b_scale_local = slice_cols_u8(reinterpret_cast<const uint8_t*>(wo_b_shard.tensor_data(*wo_b_scale)), dims.dim / 128, static_cast<int>((config.o_lora_rank * config.o_groups) / 128), wo_a_row_start / 128, dims.attn_mid / 128);
        auto attn_sink_local = slice_rows_f32(reinterpret_cast<const float*>(qkv_shard.tensor_data(*attn_sink)), local_head_start, dims.heads);
        check_cuda(cudaMalloc(&c.wo_a, wo_a_local.size()), "cudaMalloc cached wo_a");
        check_cuda(cudaMalloc(&c.wo_a_scale, wo_a_scale_local.size()), "cudaMalloc cached wo_a scale");
        check_cuda(cudaMalloc(&c.wo_b, wo_b_local.size()), "cudaMalloc cached wo_b");
        check_cuda(cudaMalloc(&c.wo_b_scale, wo_b_scale_local.size()), "cudaMalloc cached wo_b scale");
        check_cuda(cudaMalloc(&c.attn_sink, attn_sink_local.size() * sizeof(float)), "cudaMalloc cached attn sink");
        check_cuda(cudaMemcpy(c.wo_a, wo_a_local.data(), wo_a_local.size(), cudaMemcpyHostToDevice), "copy cached wo_a");
        check_cuda(cudaMemcpy(c.wo_a_scale, wo_a_scale_local.data(), wo_a_scale_local.size(), cudaMemcpyHostToDevice), "copy cached wo_a scale");
        check_cuda(cudaMemcpy(c.wo_b, wo_b_local.data(), wo_b_local.size(), cudaMemcpyHostToDevice), "copy cached wo_b");
        check_cuda(cudaMemcpy(c.wo_b_scale, wo_b_scale_local.data(), wo_b_scale_local.size(), cudaMemcpyHostToDevice), "copy cached wo_b scale");
        check_cuda(cudaMemcpy(c.attn_sink, attn_sink_local.data(), attn_sink_local.size() * sizeof(float), cudaMemcpyHostToDevice), "copy cached attn sink");
        auto inserted = attention_cache.emplace(key, c);
        return inserted.first->second;
    }

    DeviceFp4ExpertCache& fp4_expert_device_cache(int layer_id, int expert_id) {
        const std::string prefix = "layers." + std::to_string(layer_id) + ".ffn.experts." + std::to_string(expert_id) + ".";
        const std::string key = std::to_string(layer_id) + ":" + std::to_string(expert_id);
        auto it = expert_cache.find(key);
        if (it != expert_cache.end()) return it->second;
        Fp4View w1 = fp4_view(prefix + "w1.weight");
        Fp4View w2 = fp4_view(prefix + "w2.weight");
        Fp4View w3 = fp4_view(prefix + "w3.weight");
        DeviceFp4ExpertCache c;
        c.w1_bytes = w1.w->nbytes;
        c.s1_bytes = w1.s->nbytes;
        c.w2_bytes = w2.w->nbytes;
        c.s2_bytes = w2.s->nbytes;
        c.w3_bytes = w3.w->nbytes;
        c.s3_bytes = w3.s->nbytes;
        check_cuda(cudaMalloc(&c.w1, c.w1_bytes), "cudaMalloc cached expert w1");
        check_cuda(cudaMalloc(&c.s1, c.s1_bytes), "cudaMalloc cached expert s1");
        check_cuda(cudaMalloc(&c.w2, c.w2_bytes), "cudaMalloc cached expert w2");
        check_cuda(cudaMalloc(&c.s2, c.s2_bytes), "cudaMalloc cached expert s2");
        check_cuda(cudaMalloc(&c.w3, c.w3_bytes), "cudaMalloc cached expert w3");
        check_cuda(cudaMalloc(&c.s3, c.s3_bytes), "cudaMalloc cached expert s3");
        check_cuda(cudaMemcpy(c.w1, w1.shard->tensor_data(*w1.w), c.w1_bytes, cudaMemcpyHostToDevice), "copy cached expert w1");
        check_cuda(cudaMemcpy(c.s1, w1.shard->tensor_data(*w1.s), c.s1_bytes, cudaMemcpyHostToDevice), "copy cached expert s1");
        check_cuda(cudaMemcpy(c.w2, w2.shard->tensor_data(*w2.w), c.w2_bytes, cudaMemcpyHostToDevice), "copy cached expert w2");
        check_cuda(cudaMemcpy(c.s2, w2.shard->tensor_data(*w2.s), c.s2_bytes, cudaMemcpyHostToDevice), "copy cached expert s2");
        check_cuda(cudaMemcpy(c.w3, w3.shard->tensor_data(*w3.w), c.w3_bytes, cudaMemcpyHostToDevice), "copy cached expert w3");
        check_cuda(cudaMemcpy(c.s3, w3.shard->tensor_data(*w3.s), c.s3_bytes, cudaMemcpyHostToDevice), "copy cached expert s3");
        auto inserted = expert_cache.emplace(key, c);
        return inserted.first->second;
    }

    void release_active_arena_device(DeviceFp4ActiveArena& arena) {
        if (arena.w1 == nullptr) return;
        ActiveArenaDeviceBuffers blk;
        blk.w1 = arena.w1;
        blk.s1 = arena.s1;
        blk.w2 = arena.w2;
        blk.s2 = arena.s2;
        blk.w3 = arena.w3;
        blk.s3 = arena.s3;
        blk.capacity = arena.capacity;
        blk.w1_bytes = arena.w1_bytes;
        blk.s1_bytes = arena.s1_bytes;
        blk.w2_bytes = arena.w2_bytes;
        blk.s2_bytes = arena.s2_bytes;
        blk.w3_bytes = arena.w3_bytes;
        blk.s3_bytes = arena.s3_bytes;
        active_arena_device_freelist.push_back(blk);
        arena.w1 = arena.s1 = arena.w2 = arena.s2 = arena.w3 = arena.s3 = nullptr;
        arena.staged_local.clear();
    }

    bool try_pop_active_arena_freelist(DeviceFp4ActiveArena& arena) {
        for (auto it = active_arena_device_freelist.begin(); it != active_arena_device_freelist.end(); ++it) {
            if (it->capacity == arena.capacity && it->w1_bytes == arena.w1_bytes && it->s1_bytes == arena.s1_bytes && it->w2_bytes == arena.w2_bytes && it->s2_bytes == arena.s2_bytes && it->w3_bytes == arena.w3_bytes && it->s3_bytes == arena.s3_bytes) {
                arena.w1 = it->w1;
                arena.s1 = it->s1;
                arena.w2 = it->w2;
                arena.s2 = it->s2;
                arena.w3 = it->w3;
                arena.s3 = it->s3;
                arena.staged_local.clear();
                active_arena_device_freelist.erase(it);
                return true;
            }
        }
        return false;
    }

    void touch_active_arena(const std::string& key) {
        auto it = active_arena_lru_pos.find(key);
        if (it != active_arena_lru_pos.end()) active_arena_lru.erase(it->second);
        active_arena_lru.push_back(key);
        active_arena_lru_pos[key] = std::prev(active_arena_lru.end());
        const int cap = active_arena_max_layers;
        while (cap > 0 && static_cast<int>(active_arena_lru.size()) > cap) {
            const std::string evict = active_arena_lru.front();
            active_arena_lru.pop_front();
            active_arena_lru_pos.erase(evict);
            auto victim = active_arena_cache.find(evict);
            if (victim != active_arena_cache.end()) release_active_arena_device(victim->second);
        }
    }

    HostFp4ExpertSlot& host_fp4_slot(int layer_id, int expert_id, const Fp4View& w1, const Fp4View& w2, const Fp4View& w3) {
        const std::string key = std::to_string(layer_id) + ":" + std::to_string(expert_id);
        auto it = host_fp4_slot_cache.find(key);
        if (it != host_fp4_slot_cache.end()) return it->second;
        HostFp4ExpertSlot slot;
        slot.w1q_bytes = w1.w->nbytes;
        slot.w1s_bytes = w1.s->nbytes;
        slot.w2q_bytes = w2.w->nbytes;
        slot.w2s_bytes = w2.s->nbytes;
        slot.w3q_bytes = w3.w->nbytes;
        slot.w3s_bytes = w3.s->nbytes;
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&slot.h_w1q), slot.w1q_bytes), "cudaMallocHost fp4 expert w1");
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&slot.h_w1s), slot.w1s_bytes), "cudaMallocHost fp4 expert s1");
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&slot.h_w2q), slot.w2q_bytes), "cudaMallocHost fp4 expert w2");
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&slot.h_w2s), slot.w2s_bytes), "cudaMallocHost fp4 expert s2");
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&slot.h_w3q), slot.w3q_bytes), "cudaMallocHost fp4 expert w3");
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&slot.h_w3s), slot.w3s_bytes), "cudaMallocHost fp4 expert s3");
        std::memcpy(slot.h_w1q, w1.shard->tensor_data(*w1.w), slot.w1q_bytes);
        std::memcpy(slot.h_w1s, w1.shard->tensor_data(*w1.s), slot.w1s_bytes);
        std::memcpy(slot.h_w2q, w2.shard->tensor_data(*w2.w), slot.w2q_bytes);
        std::memcpy(slot.h_w2s, w2.shard->tensor_data(*w2.s), slot.w2s_bytes);
        std::memcpy(slot.h_w3q, w3.shard->tensor_data(*w3.w), slot.w3q_bytes);
        std::memcpy(slot.h_w3s, w3.shard->tensor_data(*w3.s), slot.w3s_bytes);
        auto inserted = host_fp4_slot_cache.emplace(key, slot);
        return inserted.first->second;
    }

    int active_arena_cache_limit(int tp_world) const {
        if (active_arena_max_layers > 0) return active_arena_max_layers;
        return tp_world > 1 ? 3 : 1;
    }

    void evict_active_arena_if_needed(const std::string& incoming_key, int tp_world) {
        const int cap = active_arena_cache_limit(tp_world);
        while (cap > 0 && static_cast<int>(active_arena_lru.size()) >= cap) {
            const std::string evict = active_arena_lru.front();
            if (evict == incoming_key) break;
            active_arena_lru.pop_front();
            active_arena_lru_pos.erase(evict);
            auto victim = active_arena_cache.find(evict);
            if (victim != active_arena_cache.end()) release_active_arena_device(victim->second);
        }
    }

    void prepare_fp4_host_weights(int layer_count, int tp_world, int tp_rank) {
        if (layer_count <= 0) layer_count = static_cast<int>(config.n_layers);
        layer_count = std::min(layer_count, static_cast<int>(config.n_layers));
        const int world = std::max(1, tp_world);
        const int rank = std::max(0, tp_rank);
        const int experts_per_rank = static_cast<int>(config.n_routed_experts / world);
        const int expert_start = rank * experts_per_rank;
        const int expert_end = world > 1 ? expert_start + experts_per_rank : static_cast<int>(config.n_routed_experts);
        for (int li = 0; li < layer_count; ++li) {
            if (env_int_or_default("DSV4_CPP_PREPARE_PROGRESS", 1) != 0 && tp_rank == 0) {
                std::cerr << "CPP_PREPARE_FP4_HOST layer=" << li << "/" << layer_count << " experts=" << expert_start << "-" << expert_end << "\n";
            }
            const std::string prefix = "layers." + std::to_string(li) + ".ffn.experts.";
            for (int expert_id = expert_start; expert_id < expert_end; ++expert_id) {
                Fp4View w1 = fp4_view(prefix + std::to_string(expert_id) + ".w1.weight");
                Fp4View w2 = fp4_view(prefix + std::to_string(expert_id) + ".w2.weight");
                Fp4View w3 = fp4_view(prefix + std::to_string(expert_id) + ".w3.weight");
                (void)host_fp4_slot(li, expert_id, w1, w2, w3);
            }
        }
    }

    void prepare_resident_device_caches(int layer_count, int tp_world, int tp_rank, int dim) {
        if (layer_count <= 0) layer_count = static_cast<int>(config.n_layers);
        layer_count = std::min(layer_count, static_cast<int>(config.n_layers));
        const int world = std::max(1, tp_world);
        const int rank = std::max(0, tp_rank);
        AttentionSmokeDims dims = make_attention_dims(config, dim, world, 0);
        for (int li = 0; li < layer_count; ++li) {
            (void)attention_device_cache(li, world, rank, dims);
            (void)shared_device_cache(li, world, rank, dim);
            (void)gate_device_cache(li);
        }
    }

    DeviceFp4ActiveArena& active_fp4_arena(int layer_id, int tp_world, int tp_rank, int capacity, const DeviceFp4ExpertCache& sample) {
        const std::string key = std::to_string(layer_id) + ":" + std::to_string(tp_world) + ":" + std::to_string(tp_rank) + ":" + std::to_string(capacity);
        auto it = active_arena_cache.find(key);
        if (it != active_arena_cache.end()) {
            DeviceFp4ActiveArena& arena = it->second;
            if (arena.w1 == nullptr) {
                if (!try_pop_active_arena_freelist(arena)) {
                    evict_active_arena_if_needed(key, tp_world);
                    if (!try_pop_active_arena_freelist(arena)) {
                        check_cuda(cudaMalloc(&arena.w1, static_cast<size_t>(arena.capacity) * arena.w1_bytes), "cudaMalloc active arena w1");
                        check_cuda(cudaMalloc(&arena.s1, static_cast<size_t>(arena.capacity) * arena.s1_bytes), "cudaMalloc active arena s1");
                        check_cuda(cudaMalloc(&arena.w2, static_cast<size_t>(arena.capacity) * arena.w2_bytes), "cudaMalloc active arena w2");
                        check_cuda(cudaMalloc(&arena.s2, static_cast<size_t>(arena.capacity) * arena.s2_bytes), "cudaMalloc active arena s2");
                        check_cuda(cudaMalloc(&arena.w3, static_cast<size_t>(arena.capacity) * arena.w3_bytes), "cudaMalloc active arena w3");
                        check_cuda(cudaMalloc(&arena.s3, static_cast<size_t>(arena.capacity) * arena.s3_bytes), "cudaMalloc active arena s3");
                    }
                    arena.staged_local.clear();
                }
            }
            touch_active_arena(key);
            return arena;
        }
        DeviceFp4ActiveArena arena;
        arena.capacity = capacity;
        arena.w1_bytes = sample.w1_bytes;
        arena.s1_bytes = sample.s1_bytes;
        arena.w2_bytes = sample.w2_bytes;
        arena.s2_bytes = sample.s2_bytes;
        arena.w3_bytes = sample.w3_bytes;
        arena.s3_bytes = sample.s3_bytes;
        if (!try_pop_active_arena_freelist(arena)) {
            evict_active_arena_if_needed(key, tp_world);
            if (!try_pop_active_arena_freelist(arena)) {
                check_cuda(cudaMalloc(&arena.w1, static_cast<size_t>(capacity) * arena.w1_bytes), "cudaMalloc active arena w1");
                check_cuda(cudaMalloc(&arena.s1, static_cast<size_t>(capacity) * arena.s1_bytes), "cudaMalloc active arena s1");
                check_cuda(cudaMalloc(&arena.w2, static_cast<size_t>(capacity) * arena.w2_bytes), "cudaMalloc active arena w2");
                check_cuda(cudaMalloc(&arena.s2, static_cast<size_t>(capacity) * arena.s2_bytes), "cudaMalloc active arena s2");
                check_cuda(cudaMalloc(&arena.w3, static_cast<size_t>(capacity) * arena.w3_bytes), "cudaMalloc active arena w3");
                check_cuda(cudaMalloc(&arena.s3, static_cast<size_t>(capacity) * arena.s3_bytes), "cudaMalloc active arena s3");
            }
        }
        auto inserted = active_arena_cache.emplace(key, arena);
        touch_active_arena(key);
        return inserted.first->second;
    }

    DeviceSharedCache& shared_device_cache(int layer_id, int tp_world, int tp_rank, int dim) {
        const std::string key = std::to_string(layer_id) + ":" + std::to_string(tp_world) + ":" + std::to_string(tp_rank);
        auto it = shared_cache.find(key);
        if (it != shared_cache.end()) return it->second;
        const std::string prefix = "layers." + std::to_string(layer_id) + ".";
        SafeTensorsShard& shared_shard = shard_for_tensor(prefix + "ffn.shared_experts.w1.weight");
        const auto* w1 = require_tensor(shared_shard, prefix + "ffn.shared_experts.w1.weight");
        const auto* s1 = require_tensor(shared_shard, prefix + "ffn.shared_experts.w1.scale");
        const auto* w2 = require_tensor(shared_shard, prefix + "ffn.shared_experts.w2.weight");
        const auto* s2 = require_tensor(shared_shard, prefix + "ffn.shared_experts.w2.scale");
        const auto* w3 = require_tensor(shared_shard, prefix + "ffn.shared_experts.w3.weight");
        const auto* s3 = require_tensor(shared_shard, prefix + "ffn.shared_experts.w3.scale");
        const int shared_inter = static_cast<int>(w1->shape[0]);
        if ((shared_inter % tp_world) != 0) throw std::runtime_error("shared expert inter dim must divide TP world");
        const int local_shared_inter = shared_inter / tp_world;
        const int shared_row_start = tp_rank * local_shared_inter;
        auto shared_w1 = slice_rows_u8(reinterpret_cast<const uint8_t*>(shared_shard.tensor_data(*w1)), shared_row_start, local_shared_inter, dim);
        auto shared_s1 = slice_rows_u8(reinterpret_cast<const uint8_t*>(shared_shard.tensor_data(*s1)), shared_row_start / 128, local_shared_inter / 128, dim / 128);
        auto shared_w3 = slice_rows_u8(reinterpret_cast<const uint8_t*>(shared_shard.tensor_data(*w3)), shared_row_start, local_shared_inter, dim);
        auto shared_s3 = slice_rows_u8(reinterpret_cast<const uint8_t*>(shared_shard.tensor_data(*s3)), shared_row_start / 128, local_shared_inter / 128, dim / 128);
        auto shared_w2 = slice_cols_u8(reinterpret_cast<const uint8_t*>(shared_shard.tensor_data(*w2)), dim, shared_inter, shared_row_start, local_shared_inter);
        auto shared_s2 = slice_cols_u8(reinterpret_cast<const uint8_t*>(shared_shard.tensor_data(*s2)), dim / 128, shared_inter / 128, shared_row_start / 128, local_shared_inter / 128);
        DeviceSharedCache c;
        check_cuda(cudaMalloc(&c.w1, shared_w1.size()), "cudaMalloc cached shared w1");
        check_cuda(cudaMalloc(&c.s1, shared_s1.size()), "cudaMalloc cached shared s1");
        check_cuda(cudaMalloc(&c.w2, shared_w2.size()), "cudaMalloc cached shared w2");
        check_cuda(cudaMalloc(&c.s2, shared_s2.size()), "cudaMalloc cached shared s2");
        check_cuda(cudaMalloc(&c.w3, shared_w3.size()), "cudaMalloc cached shared w3");
        check_cuda(cudaMalloc(&c.s3, shared_s3.size()), "cudaMalloc cached shared s3");
        check_cuda(cudaMemcpy(c.w1, shared_w1.data(), shared_w1.size(), cudaMemcpyHostToDevice), "copy cached shared w1");
        check_cuda(cudaMemcpy(c.s1, shared_s1.data(), shared_s1.size(), cudaMemcpyHostToDevice), "copy cached shared s1");
        check_cuda(cudaMemcpy(c.w2, shared_w2.data(), shared_w2.size(), cudaMemcpyHostToDevice), "copy cached shared w2");
        check_cuda(cudaMemcpy(c.s2, shared_s2.data(), shared_s2.size(), cudaMemcpyHostToDevice), "copy cached shared s2");
        check_cuda(cudaMemcpy(c.w3, shared_w3.data(), shared_w3.size(), cudaMemcpyHostToDevice), "copy cached shared w3");
        check_cuda(cudaMemcpy(c.s3, shared_s3.data(), shared_s3.size(), cudaMemcpyHostToDevice), "copy cached shared s3");
        auto inserted = shared_cache.emplace(key, c);
        return inserted.first->second;
    }

    std::vector<float>& compressor_kv_state_for_layer(int layer_id, int slots, int cols) {
        auto it = compressor_kv_state.find(layer_id);
        if (it != compressor_kv_state.end()) return it->second;
        auto& state = compressor_kv_state[layer_id];
        state.assign(static_cast<size_t>(slots) * cols, 0.0f);
        return state;
    }

    std::vector<float>& compressor_score_state_for_layer(int layer_id, int slots, int cols) {
        auto it = compressor_score_state.find(layer_id);
        if (it != compressor_score_state.end()) return it->second;
        auto& state = compressor_score_state[layer_id];
        state.assign(static_cast<size_t>(slots) * cols, -INFINITY);
        return state;
    }

    int kv_cache_capacity_for_layer(int layer_id) const {
        const int window = static_cast<int>(config.window_size == 0 ? 128 : config.window_size);
        uint64_t ratio = 0;
        if (layer_id >= 0 && static_cast<size_t>(layer_id) < config.compress_ratios.size()) ratio = config.compress_ratios[static_cast<size_t>(layer_id)];
        const int compressed = ratio == 0 ? 0 : (kv_cache_tokens + static_cast<int>(ratio) - 1) / static_cast<int>(ratio);
        return std::min(256, window + compressed);
    }

    float* kv_cache_for_layer(int layer_id, int head_dim) {
        auto it = kv_cache.find(layer_id);
        if (it != kv_cache.end()) return it->second;
        float* ptr = nullptr;
        const int capacity = kv_cache_capacity_for_layer(layer_id);
        check_cuda(cudaMalloc(&ptr, static_cast<size_t>(capacity) * head_dim * sizeof(float)), "cudaMalloc kv cache");
        kv_cache[layer_id] = ptr;
        return ptr;
    }

    float* indexer_kv_cache_for_layer(int layer_id, int head_dim) {
        auto it = indexer_kv_cache.find(layer_id);
        if (it != indexer_kv_cache.end()) return it->second;
        const int capacity = std::max(1, (kv_cache_tokens + 3) / 4);
        float* ptr = nullptr;
        check_cuda(cudaMalloc(&ptr, static_cast<size_t>(capacity) * head_dim * sizeof(float)), "cudaMalloc indexer kv cache");
        indexer_kv_cache[layer_id] = ptr;
        return ptr;
    }

    std::vector<float>& indexer_compressor_kv_state_for_layer(int layer_id, int slots, int cols) {
        auto it = indexer_compressor_kv_state.find(layer_id);
        if (it != indexer_compressor_kv_state.end()) return it->second;
        auto& state = indexer_compressor_kv_state[layer_id];
        state.assign(static_cast<size_t>(slots) * cols, 0.0f);
        return state;
    }

    std::vector<float>& indexer_compressor_score_state_for_layer(int layer_id, int slots, int cols) {
        auto it = indexer_compressor_score_state.find(layer_id);
        if (it != indexer_compressor_score_state.end()) return it->second;
        auto& state = indexer_compressor_score_state[layer_id];
        state.assign(static_cast<size_t>(slots) * cols, -INFINITY);
        return state;
    }

    std::string ckpt_dir;
    SafeTensorsIndex index;
    ModelConfig config;
    SafeTensorsShard embed_shard;
    SafeTensorsShard head_shard;
    SafeTensorsShard final_norm_shard;
    SafeTensorsShard hc_head_shard;
    const SafeTensorInfo* embed = nullptr;
    const SafeTensorInfo* head = nullptr;
    const SafeTensorInfo* final_norm = nullptr;
    const SafeTensorInfo* hc_head_fn = nullptr;
    const SafeTensorInfo* hc_head_scale = nullptr;
    const SafeTensorInfo* hc_head_base = nullptr;
    int kv_cache_tokens = 0;
    ForwardSmokeOptions options;
    std::unordered_map<std::string, std::unique_ptr<SafeTensorsShard>> shard_cache;
    std::unordered_map<std::string, DeviceAttentionCache> attention_cache;
    std::unordered_map<std::string, DeviceSharedCache> shared_cache;
    std::unordered_map<std::string, DeviceFp4ExpertCache> expert_cache;
    std::unordered_map<std::string, DeviceFp4ActiveArena> active_arena_cache;
    std::unordered_map<std::string, HostFp4ExpertSlot> host_fp4_slot_cache;
    std::list<std::string> active_arena_lru;
    std::vector<ActiveArenaDeviceBuffers> active_arena_device_freelist;
    std::unordered_map<std::string, std::list<std::string>::iterator> active_arena_lru_pos;
    int active_arena_max_layers = env_int_or_default("DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS", 0);
    std::unordered_map<std::string, DeviceGateCache> gate_cache;
    std::unordered_map<int, float*> kv_cache;
    std::unordered_map<int, float*> indexer_kv_cache;
    std::unordered_map<int, std::vector<float>> compressor_kv_state;
    std::unordered_map<int, std::vector<float>> compressor_score_state;
    std::unordered_map<int, std::vector<float>> indexer_compressor_kv_state;
    std::unordered_map<int, std::vector<float>> indexer_compressor_score_state;
};

}  // namespace

ForwardSmokeResult run_safetensors_token_forward_impl(SafeForwardContext& ctx, int token, int layer_count, int position);

Dsv4Engine::Dsv4Engine(const std::string& model_path) : gguf_(model_path), config_(ModelConfig::from_gguf(gguf_)) {}

ForwardSmokeResult run_safetensors_min_layer_smoke(const std::string& ckpt_dir) {
    return run_safetensors_layer_loop_smoke(ckpt_dir, 1);
}

ForwardSmokeResult run_safetensors_layer_loop_smoke(const std::string& ckpt_dir, int layer_count) {
    return run_safetensors_token_forward(ckpt_dir, 1234, layer_count);
}

ForwardSmokeResult run_safetensors_token_forward(const std::string& ckpt_dir, int token, int layer_count) {
    return run_safetensors_token_forward_at_position(ckpt_dir, token, layer_count, 0);
}

ForwardSmokeResult run_safetensors_prompt_forward(const std::string& ckpt_dir, const std::vector<int>& tokens, int layer_count) {
    return run_safetensors_prompt_forward_with_options(ckpt_dir, tokens, layer_count, ForwardSmokeOptions{});
}

ForwardSmokeResult run_safetensors_prompt_forward_with_options(const std::string& ckpt_dir, const std::vector<int>& tokens, int layer_count, const ForwardSmokeOptions& options) {
    if (tokens.empty()) throw std::runtime_error("prompt has no tokens");
    SafeForwardContext ctx(ckpt_dir);
    ctx.options = options;
    ctx.kv_cache_tokens = static_cast<int>(std::min<size_t>(tokens.size(), 256));
    ForwardSmokeResult result;
    for (size_t i = 0; i < tokens.size(); ++i) {
        result = run_safetensors_token_forward_impl(ctx, tokens[i], layer_count, static_cast<int>(i));
    }
    return result;
}

std::vector<ForwardSmokeResult> run_safetensors_generate_tokens(const std::string& ckpt_dir, const std::vector<int>& seed_tokens, int layer_count, int max_new_tokens) {
    return run_safetensors_generate_tokens_with_options(ckpt_dir, seed_tokens, layer_count, max_new_tokens, ForwardSmokeOptions{});
}

GenerateSmokeResult run_safetensors_generate_tokens_timed_with_options(const std::string& ckpt_dir, const std::vector<int>& seed_tokens, int layer_count, int max_new_tokens, const ForwardSmokeOptions& options) {
    if (seed_tokens.empty()) throw std::runtime_error("generation seed has no tokens");
    if (max_new_tokens <= 0) return {};
    SafeForwardContext ctx(ckpt_dir);
    ctx.options = options;
    ctx.kv_cache_tokens = static_cast<int>(std::min<size_t>(seed_tokens.size() + static_cast<size_t>(max_new_tokens), 256));
    const int dim = static_cast<int>(ctx.embed->shape[1]);
    ctx.prepare_resident_device_caches(layer_count, options.tp_world, options.tp_rank, dim);
    const bool prepare_fp4_host = !options.skip_fp4_host_prepare && env_int_or_default("DSV4_CPP_PREPARE_FP4_HOST", 0) != 0;
    if (prepare_fp4_host) ctx.prepare_fp4_host_weights(layer_count, options.tp_world, options.tp_rank);
    const auto timed_t0 = Clock::now();
    ForwardSmokeResult result;
    for (size_t i = 0; i < seed_tokens.size(); ++i) {
        result = run_safetensors_token_forward_impl(ctx, seed_tokens[i], layer_count, static_cast<int>(i));
    }
    std::vector<ForwardSmokeResult> out;
    out.reserve(static_cast<size_t>(max_new_tokens));
    int token = result.top_token;
#ifdef DSV4_HAVE_NCCL
    if (options.tp_world > 1 && !options.nccl_id_path.empty()) {
        TpTopResult global = nccl_global_top1(options.tp_world, options.tp_rank, options.device, options.nccl_id_path.c_str(), result.top_token, result.top_logit);
        token = global.token;
        result.top_token = global.token;
        result.top_logit = global.logit;
    }
#endif
    ForwardSmokeResult generated = result;
    generated.token = token;
    out.push_back(generated);
    int position = static_cast<int>(seed_tokens.size());
    for (int step = 1; step < max_new_tokens; ++step) {
        result = run_safetensors_token_forward_impl(ctx, token, layer_count, position + step - 1);
#ifdef DSV4_HAVE_NCCL
        if (options.tp_world > 1 && !options.nccl_id_path.empty()) {
            TpTopResult global = nccl_global_top1(options.tp_world, options.tp_rank, options.device, options.nccl_id_path.c_str(), result.top_token, result.top_logit);
            result.top_token = global.token;
            result.top_logit = global.logit;
        }
#endif
        token = result.top_token;
        generated = result;
        generated.token = token;
        out.push_back(generated);
    }
    return GenerateSmokeResult{std::move(out), elapsed_ms(timed_t0, Clock::now()) / 1000.0};
}

std::vector<ForwardSmokeResult> run_safetensors_generate_tokens_with_options(const std::string& ckpt_dir, const std::vector<int>& seed_tokens, int layer_count, int max_new_tokens, const ForwardSmokeOptions& options) {
    return run_safetensors_generate_tokens_timed_with_options(ckpt_dir, seed_tokens, layer_count, max_new_tokens, options).tokens;
}

ForwardSmokeResult run_safetensors_token_forward_impl(SafeForwardContext& ctx, int token, int layer_count, int position) {
    if (!cuda_runtime_available()) throw std::runtime_error("CUDA runtime is not available");
    SafeTensorsIndex& index = ctx.index;
    ModelConfig& config = ctx.config;
    if (layer_count <= 0) layer_count = 1;
    if (config.n_layers > 0) layer_count = std::min(layer_count, static_cast<int>(config.n_layers));

    const auto* embed = ctx.embed;
    const auto* head = ctx.head;
    Fp4View first_w1 = ctx.fp4_view("layers.0.ffn.experts.0.w1.weight");
    Fp4View first_w2 = ctx.fp4_view("layers.0.ffn.experts.0.w2.weight");
    Fp4View first_w3 = ctx.fp4_view("layers.0.ffn.experts.0.w3.weight");

    if (token < 0 || token >= static_cast<int>(embed->shape[0])) throw std::runtime_error("token id out of range");
    const int tp_world = std::max(1, ctx.options.tp_world);
    const int tp_rank = std::max(0, ctx.options.tp_rank);
    if (tp_rank >= tp_world) throw std::runtime_error("invalid TP rank in forward options");
    const int dim = static_cast<int>(embed->shape[1]);
    AttentionSmokeDims attn_dims = make_attention_dims(config, dim, tp_world, position);
    const int inter = static_cast<int>(first_w1.pair.rows);
    const int head_rows = static_cast<int>(head->shape[0]);
    if (head_rows % tp_world != 0) throw std::runtime_error("head vocab rows must divide TP world");
    const int local_head_rows = head_rows / tp_world;
    const int local_head_start = tp_rank * local_head_rows;

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
    float* d_attn_sink = nullptr;
    uint16_t* d_ffn_gamma = nullptr;
    uint8_t* d_w1 = nullptr;
    uint8_t* d_s1 = nullptr;
    uint8_t* d_w2 = nullptr;
    uint8_t* d_s2 = nullptr;
    uint8_t* d_w3 = nullptr;
    uint8_t* d_s3 = nullptr;
    uint16_t* d_head = nullptr;
    uint16_t* d_final_norm_gamma = nullptr;
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
    float* d_index_q = nullptr;
    float* d_indexer_kv = nullptr;
    uint16_t* d_index_weight_proj = nullptr;
    int* d_index_selected = nullptr;
    int* d_kv_indices = nullptr;
    float* d_resid1 = nullptr;
    float* d_ffn_norm = nullptr;
    float* d_gate = nullptr;
    float* d_up = nullptr;
    float* d_hidden = nullptr;
    float* d_moe = nullptr;
    float* d_resid2 = nullptr;
    int64_t* d_route_indices = nullptr;
    float* d_route_weights = nullptr;
    float* d_logits = nullptr;

    const auto* embed_data = reinterpret_cast<const uint16_t*>(ctx.embed_shard.tensor_data(*embed)) + static_cast<size_t>(token) * dim;
    check_cuda(cudaMalloc(&d_embed, static_cast<size_t>(dim) * sizeof(uint16_t)), "cudaMalloc embed");
    check_cuda(cudaMalloc(&d_attn_gamma, static_cast<size_t>(dim) * sizeof(uint16_t)), "cudaMalloc attn gamma");
    check_cuda(cudaMalloc(&d_wq_a, static_cast<size_t>(attn_dims.q_a_dim) * dim), "cudaMalloc wq_a");
    check_cuda(cudaMalloc(&d_wq_a_scale, static_cast<size_t>(attn_dims.q_a_dim / 128) * (dim / 128)), "cudaMalloc wq_a scale");
    check_cuda(cudaMalloc(&d_q_gamma, static_cast<size_t>(attn_dims.q_a_dim) * sizeof(uint16_t)), "cudaMalloc q gamma");
    check_cuda(cudaMalloc(&d_wq_b, static_cast<size_t>(attn_dims.q_dim) * attn_dims.q_a_dim), "cudaMalloc wq_b");
    check_cuda(cudaMalloc(&d_wq_b_scale, static_cast<size_t>(attn_dims.q_dim / 128) * (attn_dims.q_a_dim / 128)), "cudaMalloc wq_b scale");
    check_cuda(cudaMalloc(&d_wkv, static_cast<size_t>(attn_dims.kv_dim) * dim), "cudaMalloc wkv");
    check_cuda(cudaMalloc(&d_wkv_scale, static_cast<size_t>(attn_dims.kv_dim / 128) * (dim / 128)), "cudaMalloc wkv scale");
    check_cuda(cudaMalloc(&d_kv_gamma, static_cast<size_t>(attn_dims.kv_dim) * sizeof(uint16_t)), "cudaMalloc kv gamma");
    check_cuda(cudaMalloc(&d_wo_a, static_cast<size_t>(attn_dims.attn_mid) * dim), "cudaMalloc wo_a");
    check_cuda(cudaMalloc(&d_wo_a_scale, static_cast<size_t>(attn_dims.attn_mid / 128) * (dim / 128)), "cudaMalloc wo_a scale");
    check_cuda(cudaMalloc(&d_wo_b, static_cast<size_t>(dim) * attn_dims.attn_mid), "cudaMalloc wo_b");
    check_cuda(cudaMalloc(&d_wo_b_scale, static_cast<size_t>(dim / 128) * (attn_dims.attn_mid / 128)), "cudaMalloc wo_b scale");
    check_cuda(cudaMalloc(&d_attn_sink, static_cast<size_t>(attn_dims.heads) * sizeof(float)), "cudaMalloc attn sink");
    check_cuda(cudaMalloc(&d_ffn_gamma, static_cast<size_t>(dim) * sizeof(uint16_t)), "cudaMalloc ffn gamma");
    check_cuda(cudaMalloc(&d_w1, static_cast<size_t>(inter) * dim), "cudaMalloc w1");
    check_cuda(cudaMalloc(&d_s1, first_w1.s->nbytes), "cudaMalloc s1");
    check_cuda(cudaMalloc(&d_w2, static_cast<size_t>(dim) * inter), "cudaMalloc w2");
    check_cuda(cudaMalloc(&d_s2, first_w2.s->nbytes), "cudaMalloc s2");
    check_cuda(cudaMalloc(&d_w3, static_cast<size_t>(inter) * dim), "cudaMalloc w3");
    check_cuda(cudaMalloc(&d_s3, first_w3.s->nbytes), "cudaMalloc s3");
    check_cuda(cudaMalloc(&d_head, static_cast<size_t>(local_head_rows) * dim * sizeof(uint16_t)), "cudaMalloc head");
    check_cuda(cudaMalloc(&d_final_norm_gamma, static_cast<size_t>(dim) * sizeof(uint16_t)), "cudaMalloc final norm gamma");
    check_cuda(cudaMalloc(&d_x, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc x");
    check_cuda(cudaMalloc(&d_attn_norm, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc attn norm");
    check_cuda(cudaMalloc(&d_q_a, static_cast<size_t>(attn_dims.q_a_dim) * sizeof(float)), "cudaMalloc q_a");
    check_cuda(cudaMalloc(&d_q_norm, static_cast<size_t>(attn_dims.q_a_dim) * sizeof(float)), "cudaMalloc q_norm");
    check_cuda(cudaMalloc(&d_q, static_cast<size_t>(attn_dims.q_dim) * sizeof(float)), "cudaMalloc q");
    check_cuda(cudaMalloc(&d_kv_a, static_cast<size_t>(attn_dims.kv_dim) * sizeof(float)), "cudaMalloc kv_a");
    check_cuda(cudaMalloc(&d_kv_norm, static_cast<size_t>(attn_dims.kv_dim) * sizeof(float)), "cudaMalloc kv_norm");
    check_cuda(cudaMalloc(&d_attn_value, static_cast<size_t>(attn_dims.q_dim) * sizeof(float)), "cudaMalloc attn value");
    check_cuda(cudaMalloc(&d_attn_mid, static_cast<size_t>(attn_dims.attn_mid) * sizeof(float)), "cudaMalloc attn mid");
    check_cuda(cudaMalloc(&d_attn_out, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc attn out");
    check_cuda(cudaMalloc(&d_index_q, static_cast<size_t>(std::max<uint64_t>(1, config.index_n_heads * config.index_head_dim)) * sizeof(float)), "cudaMalloc index q");
    check_cuda(cudaMalloc(&d_indexer_kv, static_cast<size_t>(std::max<uint64_t>(1, config.index_head_dim)) * sizeof(float)), "cudaMalloc indexer kv tmp");
    check_cuda(cudaMalloc(&d_index_weight_proj, static_cast<size_t>(std::max<uint64_t>(1, config.index_n_heads * config.dim)) * sizeof(uint16_t)), "cudaMalloc index weight proj");
    check_cuda(cudaMalloc(&d_index_selected, 640 * sizeof(int)), "cudaMalloc index selected");
    check_cuda(cudaMalloc(&d_kv_indices, 640 * sizeof(int)), "cudaMalloc kv indices");
    check_cuda(cudaMalloc(&d_resid1, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc resid1");
    check_cuda(cudaMalloc(&d_ffn_norm, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc ffn norm");
    check_cuda(cudaMalloc(&d_gate, static_cast<size_t>(inter) * sizeof(float)), "cudaMalloc gate");
    check_cuda(cudaMalloc(&d_up, static_cast<size_t>(inter) * sizeof(float)), "cudaMalloc up");
    check_cuda(cudaMalloc(&d_hidden, static_cast<size_t>(inter) * sizeof(float)), "cudaMalloc hidden");
    check_cuda(cudaMalloc(&d_moe, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc moe");
    check_cuda(cudaMalloc(&d_resid2, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc resid2");
    check_cuda(cudaMalloc(&d_route_indices, static_cast<size_t>(config.n_activated_experts) * sizeof(int64_t)), "cudaMalloc route indices");
    check_cuda(cudaMalloc(&d_route_weights, static_cast<size_t>(config.n_activated_experts) * sizeof(float)), "cudaMalloc route weights");
    check_cuda(cudaMalloc(&d_logits, static_cast<size_t>(local_head_rows) * sizeof(float)), "cudaMalloc logits");

    check_cuda(cudaMemcpy(d_embed, embed_data, static_cast<size_t>(dim) * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy embed");
    const auto* head_data = reinterpret_cast<const uint16_t*>(ctx.head_shard.tensor_data(*head)) + static_cast<size_t>(local_head_start) * dim;
    check_cuda(cudaMemcpy(d_head, head_data, static_cast<size_t>(local_head_rows) * dim * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy head");
    check_cuda(cudaMemcpy(d_final_norm_gamma, ctx.final_norm_shard.tensor_data(*ctx.final_norm), ctx.final_norm->nbytes, cudaMemcpyHostToDevice), "copy final norm gamma");
    if (!bf16_row_to_float_cuda(d_embed, d_x, 0, dim)) throw std::runtime_error("embed launch failed");
    const bool debug_forward = debug_forward_enabled();
    const bool profile_forward = profile_forward_enabled();
    double total_load_ms = 0.0;
    double total_hc_ms = 0.0;
    double total_attn_ms = 0.0;
    double total_route_ms = 0.0;
    double total_moe_ms = 0.0;
    double total_shared_ms = 0.0;
    double total_post_ms = 0.0;
    std::vector<float> h4(static_cast<size_t>(4) * dim);
    std::vector<float> host_x(dim);
    check_cuda(cudaMemcpy(host_x.data(), d_x, static_cast<size_t>(dim) * sizeof(float), cudaMemcpyDeviceToHost), "copy embed host");
    for (int m = 0; m < 4; ++m) std::copy(host_x.begin(), host_x.end(), h4.begin() + static_cast<size_t>(m) * dim);

    for (int li = 0; li < layer_count; ++li) {
        const auto layer_t0 = Clock::now();
        auto stage_t = layer_t0;
        double load_ms = 0.0;
        double hc_ms = 0.0;
        double attn_ms = 0.0;
        double route_ms = 0.0;
        double moe_ms = 0.0;
        double shared_ms = 0.0;
        double post_ms = 0.0;
        const std::string prefix = "layers." + std::to_string(li) + ".";
        attn_dims.layer_id = li;
        attn_dims.cache_write_slot = position % attn_dims.window_size;
        float* d_layer_kv_cache = ctx.kv_cache_tokens > 0 ? ctx.kv_cache_for_layer(li, attn_dims.head_dim) : nullptr;
        uint64_t layer_compress_ratio = static_cast<size_t>(li) < ctx.config.compress_ratios.size() ? ctx.config.compress_ratios[static_cast<size_t>(li)] : 0;
        attn_dims.rope_theta = static_cast<float>(layer_compress_ratio == 0 ? config.rope_theta : config.compress_rope_theta);
        if (attn_dims.rope_theta <= 0.0f) throw std::runtime_error("invalid layer rope_theta");
        const int compressed_ready = layer_compress_ratio == 0 ? 0 : (position + 1) / static_cast<int>(layer_compress_ratio);
        const int window_len = std::min(position + 1, attn_dims.window_size);
        const int layer_cache_len = d_layer_kv_cache == nullptr ? 0 : std::min(ctx.kv_cache_capacity_for_layer(li), window_len + compressed_ready);
        std::vector<int> kv_indices;
        SafeTensorsShard& attn_norm_shard = ctx.shard_for_tensor(prefix + "attn_norm.weight");
        SafeTensorsShard& qkv_shard = ctx.shard_for_tensor(prefix + "attn.wq_a.weight");
        SafeTensorsShard& wo_a_shard = ctx.shard_for_tensor(prefix + "attn.wo_a.weight");
        SafeTensorsShard& wo_b_shard = ctx.shard_for_tensor(prefix + "attn.wo_b.weight");
        SafeTensorsShard& ffn_norm_shard = ctx.shard_for_tensor(prefix + "ffn_norm.weight");
        const auto* hc_attn_fn = require_tensor(qkv_shard, prefix + "hc_attn_fn");
        const auto* hc_attn_scale = require_tensor(qkv_shard, prefix + "hc_attn_scale");
        const auto* hc_attn_base = require_tensor(qkv_shard, prefix + "hc_attn_base");
        const auto* hc_ffn_fn = require_tensor(qkv_shard, prefix + "hc_ffn_fn");
        const auto* hc_ffn_scale = require_tensor(qkv_shard, prefix + "hc_ffn_scale");
        const auto* hc_ffn_base = require_tensor(qkv_shard, prefix + "hc_ffn_base");
        const auto* attn_norm = require_tensor(attn_norm_shard, prefix + "attn_norm.weight");
        const auto* q_norm = require_tensor(qkv_shard, prefix + "attn.q_norm.weight");
        const auto* kv_norm = require_tensor(qkv_shard, prefix + "attn.kv_norm.weight");
        const auto* ffn_norm = require_tensor(ffn_norm_shard, prefix + "ffn_norm.weight");
        DeviceAttentionCache& attn_cache = ctx.attention_device_cache(li, tp_world, tp_rank, attn_dims);

        check_cuda(cudaMemcpy(d_attn_gamma, attn_norm_shard.tensor_data(*attn_norm), attn_norm->nbytes, cudaMemcpyHostToDevice), "copy attn gamma");
        check_cuda(cudaMemcpy(d_q_gamma, qkv_shard.tensor_data(*q_norm), q_norm->nbytes, cudaMemcpyHostToDevice), "copy q norm");
        check_cuda(cudaMemcpy(d_kv_gamma, qkv_shard.tensor_data(*kv_norm), kv_norm->nbytes, cudaMemcpyHostToDevice), "copy kv norm");
        check_cuda(cudaMemcpy(d_ffn_gamma, ffn_norm_shard.tensor_data(*ffn_norm), ffn_norm->nbytes, cudaMemcpyHostToDevice), "copy ffn gamma");
        load_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();

        const HcPreResult attn_hc = hc_pre_cpu(
            h4,
            reinterpret_cast<const float*>(qkv_shard.tensor_data(*hc_attn_fn)),
            reinterpret_cast<const float*>(qkv_shard.tensor_data(*hc_attn_scale)),
            reinterpret_cast<const float*>(qkv_shard.tensor_data(*hc_attn_base)),
            dim);
        check_cuda(cudaMemcpy(d_x, attn_hc.x.data(), static_cast<size_t>(dim) * sizeof(float), cudaMemcpyHostToDevice), "copy hc attn pre");
        hc_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();

        uint64_t compress_ratio = layer_compress_ratio;
        if (d_layer_kv_cache != nullptr && compress_ratio > 0 && compress_ratio <= 256 && index.shard_for_tensor(prefix + "attn.compressor.wkv.weight") != nullptr) {
            SafeTensorsShard& comp_shard = ctx.shard_for_tensor(prefix + "attn.compressor.wkv.weight");
            const auto* comp_wkv = require_tensor(comp_shard, prefix + "attn.compressor.wkv.weight");
            const auto* comp_wgate = require_tensor(comp_shard, prefix + "attn.compressor.wgate.weight");
            const auto* comp_ape = require_tensor(comp_shard, prefix + "attn.compressor.ape");
            const auto* comp_norm = require_tensor(comp_shard, prefix + "attn.compressor.norm.weight");
            const int comp_cols = static_cast<int>(comp_wkv->shape[0]);
            const int ratio = static_cast<int>(compress_ratio);
            const bool overlap = comp_cols == attn_dims.head_dim * 2;
            const int slots = ratio * (overlap ? 2 : 1);
            std::vector<float> comp_kv = bf16_matvec_cpu(attn_hc.x, reinterpret_cast<const uint16_t*>(comp_shard.tensor_data(*comp_wkv)), comp_cols, dim);
            std::vector<float> comp_score = bf16_matvec_cpu(attn_hc.x, reinterpret_cast<const uint16_t*>(comp_shard.tensor_data(*comp_wgate)), comp_cols, dim);
            const int offset = position % ratio;
            const float* ape = reinterpret_cast<const float*>(comp_shard.tensor_data(*comp_ape)) + static_cast<size_t>(offset) * comp_cols;
            std::vector<float>& kv_state = ctx.compressor_kv_state_for_layer(li, slots, attn_dims.head_dim);
            std::vector<float>& score_state = ctx.compressor_score_state_for_layer(li, slots, attn_dims.head_dim);
            if (overlap) {
                for (int d = 0; d < attn_dims.head_dim; ++d) {
                    kv_state[static_cast<size_t>(ratio + offset) * attn_dims.head_dim + d] = comp_kv[attn_dims.head_dim + d];
                    score_state[static_cast<size_t>(ratio + offset) * attn_dims.head_dim + d] = comp_score[attn_dims.head_dim + d] + ape[attn_dims.head_dim + d];
                }
            } else {
                for (int d = 0; d < attn_dims.head_dim; ++d) {
                    kv_state[static_cast<size_t>(offset) * attn_dims.head_dim + d] = comp_kv[d];
                    score_state[static_cast<size_t>(offset) * attn_dims.head_dim + d] = comp_score[d] + ape[d];
                }
            }
            if ((position + 1) % ratio == 0) {
                std::vector<float> pooled(attn_dims.head_dim, 0.0f);
                const int pool_slots = overlap ? ratio * 2 : ratio;
                for (int d = 0; d < attn_dims.head_dim; ++d) {
                    float max_score = -INFINITY;
                    for (int t = 0; t < pool_slots; ++t) max_score = std::max(max_score, score_state[static_cast<size_t>(t) * attn_dims.head_dim + d]);
                    float denom = 0.0f;
                    for (int t = 0; t < pool_slots; ++t) denom += std::exp(score_state[static_cast<size_t>(t) * attn_dims.head_dim + d] - max_score);
                    for (int t = 0; t < pool_slots; ++t) {
                        const float w = std::exp(score_state[static_cast<size_t>(t) * attn_dims.head_dim + d] - max_score) / denom;
                        pooled[d] += w * kv_state[static_cast<size_t>(t) * attn_dims.head_dim + d];
                    }
                }
                pooled = rmsnorm_cpu(pooled, reinterpret_cast<const uint16_t*>(comp_shard.tensor_data(*comp_norm)), 1e-6f);
                const int compressed_slot = attn_dims.window_size + position / ratio;
                if (compressed_slot < ctx.kv_cache_capacity_for_layer(li)) {
                    check_cuda(cudaMemcpy(d_layer_kv_cache + static_cast<size_t>(compressed_slot) * attn_dims.head_dim, pooled.data(), static_cast<size_t>(attn_dims.head_dim) * sizeof(float), cudaMemcpyHostToDevice), "copy compressed kv");
                }
                if (overlap) {
                    std::copy(kv_state.begin() + static_cast<size_t>(ratio) * attn_dims.head_dim, kv_state.end(), kv_state.begin());
                    std::copy(score_state.begin() + static_cast<size_t>(ratio) * attn_dims.head_dim, score_state.end(), score_state.begin());
                }
            }
        }
        if (d_layer_kv_cache != nullptr) {
            kv_indices.reserve(static_cast<size_t>(window_len + compressed_ready));
            const int window_start = std::max(0, position - window_len + 1);
            for (int p = window_start; p <= position; ++p) kv_indices.push_back(p % attn_dims.window_size);
            if (compress_ratio == 4 && compressed_ready > 0 && index.shard_for_tensor(prefix + "attn.indexer.wq_b.weight") != nullptr) {
                SafeTensorsShard& idx_shard = ctx.shard_for_tensor(prefix + "attn.indexer.wq_b.weight");
                const auto* idx_wq_b = require_tensor(idx_shard, prefix + "attn.indexer.wq_b.weight");
                const auto* idx_wq_b_scale = require_tensor(idx_shard, prefix + "attn.indexer.wq_b.scale");
                const auto* idx_weights = require_tensor(idx_shard, prefix + "attn.indexer.weights_proj.weight");
                const auto* idx_comp_wkv = require_tensor(idx_shard, prefix + "attn.indexer.compressor.wkv.weight");
                const auto* idx_comp_wgate = require_tensor(idx_shard, prefix + "attn.indexer.compressor.wgate.weight");
                const auto* idx_comp_ape = require_tensor(idx_shard, prefix + "attn.indexer.compressor.ape");
                const auto* idx_comp_norm = require_tensor(idx_shard, prefix + "attn.indexer.compressor.norm.weight");
                const int idx_heads = static_cast<int>(config.index_n_heads);
                const int idx_head_dim = static_cast<int>(config.index_head_dim);
                const int idx_cols = static_cast<int>(idx_comp_wkv->shape[0]);
                const bool idx_overlap = idx_cols == idx_head_dim * 2;
                const int idx_slots = 4 * (idx_overlap ? 2 : 1);
                std::vector<float> idx_comp_kv = bf16_matvec_cpu(attn_hc.x, reinterpret_cast<const uint16_t*>(idx_shard.tensor_data(*idx_comp_wkv)), idx_cols, dim);
                std::vector<float> idx_comp_score = bf16_matvec_cpu(attn_hc.x, reinterpret_cast<const uint16_t*>(idx_shard.tensor_data(*idx_comp_wgate)), idx_cols, dim);
                const int offset = position % 4;
                const float* ape = reinterpret_cast<const float*>(idx_shard.tensor_data(*idx_comp_ape)) + static_cast<size_t>(offset) * idx_cols;
                std::vector<float>& idx_kv_state = ctx.indexer_compressor_kv_state_for_layer(li, idx_slots, idx_head_dim);
                std::vector<float>& idx_score_state = ctx.indexer_compressor_score_state_for_layer(li, idx_slots, idx_head_dim);
                if (idx_overlap) {
                    for (int d = 0; d < idx_head_dim; ++d) {
                        idx_kv_state[static_cast<size_t>(4 + offset) * idx_head_dim + d] = idx_comp_kv[idx_head_dim + d];
                        idx_score_state[static_cast<size_t>(4 + offset) * idx_head_dim + d] = idx_comp_score[idx_head_dim + d] + ape[idx_head_dim + d];
                    }
                } else {
                    for (int d = 0; d < idx_head_dim; ++d) {
                        idx_kv_state[static_cast<size_t>(offset) * idx_head_dim + d] = idx_comp_kv[d];
                        idx_score_state[static_cast<size_t>(offset) * idx_head_dim + d] = idx_comp_score[d] + ape[d];
                    }
                }
                if ((position + 1) % 4 == 0) {
                    std::vector<float> pooled(idx_head_dim, 0.0f);
                    const int pool_slots = idx_overlap ? 8 : 4;
                    for (int d = 0; d < idx_head_dim; ++d) {
                        float max_score = -INFINITY;
                        for (int t = 0; t < pool_slots; ++t) max_score = std::max(max_score, idx_score_state[static_cast<size_t>(t) * idx_head_dim + d]);
                        float denom = 0.0f;
                        for (int t = 0; t < pool_slots; ++t) denom += std::exp(idx_score_state[static_cast<size_t>(t) * idx_head_dim + d] - max_score);
                        for (int t = 0; t < pool_slots; ++t) {
                            const float w = std::exp(idx_score_state[static_cast<size_t>(t) * idx_head_dim + d] - max_score) / denom;
                            pooled[d] += w * idx_kv_state[static_cast<size_t>(t) * idx_head_dim + d];
                        }
                    }
                    pooled = rmsnorm_cpu(pooled, reinterpret_cast<const uint16_t*>(idx_shard.tensor_data(*idx_comp_norm)), 1e-6f);
                    float* d_idx_cache = ctx.indexer_kv_cache_for_layer(li, idx_head_dim);
                    check_cuda(cudaMemcpy(d_idx_cache + static_cast<size_t>(position / 4) * idx_head_dim, pooled.data(), static_cast<size_t>(idx_head_dim) * sizeof(float), cudaMemcpyHostToDevice), "copy indexer compressed kv");
                    if (idx_overlap) {
                        std::copy(idx_kv_state.begin() + static_cast<size_t>(4) * idx_head_dim, idx_kv_state.end(), idx_kv_state.begin());
                        std::copy(idx_score_state.begin() + static_cast<size_t>(4) * idx_head_dim, idx_score_state.end(), idx_score_state.begin());
                    }
                }
                if (!rmsnorm_bf16_gamma_cuda(d_x, d_attn_gamma, d_attn_norm, dim, 1e-6f)) throw std::runtime_error("indexer attn norm launch failed");
                if (!fp8_e4m3_e8m0_matvec_cuda(d_attn_norm, d_wq_a, d_wq_a_scale, d_q_a, attn_dims.q_a_dim, dim)) throw std::runtime_error("indexer wq_a launch failed");
                if (!rmsnorm_bf16_gamma_cuda(d_q_a, d_q_gamma, d_q_norm, attn_dims.q_a_dim, 1e-6f)) throw std::runtime_error("indexer q norm launch failed");
                check_cuda(cudaMemcpy(d_w2, idx_shard.tensor_data(*idx_wq_b), idx_wq_b->nbytes, cudaMemcpyHostToDevice), "copy indexer wq_b");
                check_cuda(cudaMemcpy(d_s2, idx_shard.tensor_data(*idx_wq_b_scale), idx_wq_b_scale->nbytes, cudaMemcpyHostToDevice), "copy indexer wq_b scale");
                if (!fp8_e4m3_e8m0_matvec_cuda(d_q_norm, d_w2, d_s2, d_index_q, idx_heads * idx_head_dim, attn_dims.q_a_dim)) throw std::runtime_error("indexer wq_b launch failed");
                if (!head_rmsnorm_rope_cuda(d_index_q, idx_heads, idx_head_dim, attn_dims.rope_dim, position, attn_dims.rope_theta, false, 1e-6f)) throw std::runtime_error("indexer q rope launch failed");
                const int keep = std::min<int>(compressed_ready, std::max<uint64_t>(1, config.index_topk));
                check_cuda(cudaMemcpy(d_index_weight_proj, idx_shard.tensor_data(*idx_weights), idx_weights->nbytes, cudaMemcpyHostToDevice), "copy indexer weights proj");
                if (!indexer_select_topk_cuda(
                        d_index_q,
                        ctx.indexer_kv_cache_for_layer(li, idx_head_dim),
                        d_index_weight_proj,
                        d_x,
                        d_index_selected,
                        compressed_ready,
                        keep,
                        idx_heads,
                        idx_head_dim,
                        dim,
                        attn_dims.window_size)) {
                    throw std::runtime_error("indexer topk launch failed");
                }
                std::vector<int> selected(keep);
                check_cuda(cudaMemcpy(selected.data(), d_index_selected, selected.size() * sizeof(int), cudaMemcpyDeviceToHost), "copy indexer selected");
                kv_indices.insert(kv_indices.end(), selected.begin(), selected.end());
            } else {
                for (int c = 0; c < compressed_ready; ++c) {
                    const int slot = attn_dims.window_size + c;
                    if (slot < ctx.kv_cache_capacity_for_layer(li)) kv_indices.push_back(slot);
                }
            }
            if (!kv_indices.empty()) check_cuda(cudaMemcpy(d_kv_indices, kv_indices.data(), kv_indices.size() * sizeof(int), cudaMemcpyHostToDevice), "copy kv indices");
        }

        route_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();
        if (!run_single_token_attention_smoke(
                attn_dims,
                d_x,
                d_attn_gamma,
                attn_cache.wq_a,
                attn_cache.wq_a_scale,
                d_q_gamma,
                attn_cache.wq_b,
                attn_cache.wq_b_scale,
                attn_cache.wkv,
                attn_cache.wkv_scale,
                d_kv_gamma,
                attn_cache.wo_a,
                attn_cache.wo_a_scale,
                attn_cache.wo_b,
                attn_cache.wo_b_scale,
                attn_cache.attn_sink,
                d_layer_kv_cache,
                kv_indices.empty() ? nullptr : d_kv_indices,
                static_cast<int>(kv_indices.size()),
                layer_cache_len,
                d_attn_norm,
                d_q_a,
                d_q_norm,
                d_q,
                d_kv_a,
                d_kv_norm,
                d_attn_value,
                d_attn_mid,
                d_attn_out)) {
            throw std::runtime_error("single-token attention smoke launch failed");
        }
#ifdef DSV4_HAVE_NCCL
        if (ctx.options.tp_world > 1) {
            if (ctx.options.nccl_id_path.empty()) throw std::runtime_error("TP attention all-reduce requires --nccl-id-path");
            nccl_all_reduce_sum_float_inplace(ctx.options.tp_world, ctx.options.tp_rank, ctx.options.device, ctx.options.nccl_id_path.c_str(), d_attn_out, dim);
        }
#endif
        check_cuda(cudaMemcpy(host_x.data(), d_attn_out, static_cast<size_t>(dim) * sizeof(float), cudaMemcpyDeviceToHost), "copy attn out host");
        attn_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();
        if (debug_forward) print_summary("layer=" + std::to_string(li) + ".attn_out", host_x);
        h4 = hc_post_cpu(host_x, h4, attn_hc, dim);
        round_vector_to_bf16(h4);
        post_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();
        if (debug_forward) print_summary("layer=" + std::to_string(li) + ".attn_post", h4);
        const HcPreResult ffn_hc = hc_pre_cpu(
            h4,
            reinterpret_cast<const float*>(qkv_shard.tensor_data(*hc_ffn_fn)),
            reinterpret_cast<const float*>(qkv_shard.tensor_data(*hc_ffn_scale)),
            reinterpret_cast<const float*>(qkv_shard.tensor_data(*hc_ffn_base)),
            dim);
        if (debug_forward) print_summary("layer=" + std::to_string(li) + ".ffn_hc_pre", ffn_hc.x);
        check_cuda(cudaMemcpy(d_resid1, ffn_hc.x.data(), static_cast<size_t>(dim) * sizeof(float), cudaMemcpyHostToDevice), "copy hc ffn pre");
        if (!rmsnorm_bf16_gamma_cuda(d_resid1, d_ffn_gamma, d_ffn_norm, dim, 1e-6f)) throw std::runtime_error("ffn norm launch failed");
        const int route_count = static_cast<int>(std::min<uint64_t>(config.n_activated_experts, config.n_routed_experts));
        std::vector<RoutedExpert> routed;
        std::vector<int64_t> selected_route_ids;
        std::vector<float> selected_route_weights;
        if (static_cast<uint64_t>(li) < config.n_hash_layers && index.shard_for_tensor(prefix + "ffn.gate.tid2eid") != nullptr) {
            std::vector<float> ffn_norm_host(dim);
            check_cuda(cudaMemcpy(ffn_norm_host.data(), d_ffn_norm, static_cast<size_t>(dim) * sizeof(float), cudaMemcpyDeviceToHost), "copy ffn norm for hash gate");
            const auto selected = select_smoke_experts(index, prefix, li, token, ffn_norm_host, config.n_hash_layers, config.n_routed_experts, config.n_activated_experts, static_cast<float>(config.route_scale));
            selected_route_ids.reserve(selected.size());
            selected_route_weights.reserve(selected.size());
            for (const RoutedExpert& route : selected) {
                selected_route_ids.push_back(route.id);
                selected_route_weights.push_back(route.weight);
            }
        } else {
            DeviceGateCache& gate = ctx.gate_device_cache(li);
            if (!gate_topk_bf16_cuda_with_buffers(d_ffn_norm, gate.weight, gate.bias, gate.original, gate.scored, d_route_indices, d_route_weights, gate.experts, gate.dim, route_count, static_cast<float>(config.route_scale))) throw std::runtime_error("gate topk launch failed");
            selected_route_ids.resize(route_count);
            selected_route_weights.resize(route_count);
            check_cuda(cudaMemcpy(selected_route_ids.data(), d_route_indices, selected_route_ids.size() * sizeof(int64_t), cudaMemcpyDeviceToHost), "copy gate route ids");
            check_cuda(cudaMemcpy(selected_route_weights.data(), d_route_weights, selected_route_weights.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy gate route weights");
        }
        routed.reserve(selected_route_ids.size());
        for (size_t ri = 0; ri < selected_route_ids.size(); ++ri) routed.push_back(RoutedExpert{static_cast<int>(selected_route_ids[ri]), selected_route_weights[ri]});
        route_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();
        check_cuda(cudaMemset(d_moe, 0, static_cast<size_t>(dim) * sizeof(float)), "zero moe");
        const int experts_per_rank = ctx.options.tp_world > 1 ? static_cast<int>(config.n_routed_experts / ctx.options.tp_world) : static_cast<int>(config.n_routed_experts);
        const int expert_start = ctx.options.tp_rank * experts_per_rank;
        const int expert_end = ctx.options.tp_world > 1 ? expert_start + experts_per_rank : static_cast<int>(config.n_routed_experts);
        std::vector<int64_t> route_indices;
        std::vector<float> route_weights;
        std::vector<Fp4View> active_w1;
        std::vector<Fp4View> active_w2;
        std::vector<Fp4View> active_w3;
        route_indices.reserve(routed.size());
        route_weights.reserve(routed.size());
        active_w1.reserve(routed.size());
        active_w2.reserve(routed.size());
        active_w3.reserve(routed.size());
        std::vector<int> active_local_ids;
        for (const RoutedExpert& route : routed) {
            if (ctx.options.tp_world > 1 && (route.id < expert_start || route.id >= expert_end)) continue;
            route_indices.push_back(route.id);
            route_weights.push_back(route.weight);
            active_local_ids.push_back(route.id - expert_start);
            active_w1.push_back(ctx.fp4_view(prefix + "ffn.experts." + std::to_string(route.id) + ".w1.weight"));
            active_w2.push_back(ctx.fp4_view(prefix + "ffn.experts." + std::to_string(route.id) + ".w2.weight"));
            active_w3.push_back(ctx.fp4_view(prefix + "ffn.experts." + std::to_string(route.id) + ".w3.weight"));
        }
        if (!active_w1.empty()) {
            DeviceFp4ExpertCache sample;
            sample.w1_bytes = active_w1.front().w->nbytes;
            sample.s1_bytes = active_w1.front().s->nbytes;
            sample.w2_bytes = active_w2.front().w->nbytes;
            sample.s2_bytes = active_w2.front().s->nbytes;
            sample.w3_bytes = active_w3.front().w->nbytes;
            sample.s3_bytes = active_w3.front().s->nbytes;
            DeviceFp4ActiveArena& arena = ctx.active_fp4_arena(li, tp_world, tp_rank, experts_per_rank, sample);
            for (size_t ri = 0; ri < active_w1.size(); ++ri) {
                const int local = active_local_ids[ri];
                if (arena.staged_local.insert(local).second) {
                    HostFp4ExpertSlot& slot = ctx.host_fp4_slot(li, expert_start + local, active_w1[ri], active_w2[ri], active_w3[ri]);
                    check_cuda(cudaMemcpyAsync(arena.w1 + static_cast<size_t>(local) * arena.w1_bytes, slot.h_w1q, arena.w1_bytes, cudaMemcpyHostToDevice), "stage active w1");
                    check_cuda(cudaMemcpyAsync(arena.s1 + static_cast<size_t>(local) * arena.s1_bytes, slot.h_w1s, arena.s1_bytes, cudaMemcpyHostToDevice), "stage active s1");
                    check_cuda(cudaMemcpyAsync(arena.w2 + static_cast<size_t>(local) * arena.w2_bytes, slot.h_w2q, arena.w2_bytes, cudaMemcpyHostToDevice), "stage active w2");
                    check_cuda(cudaMemcpyAsync(arena.s2 + static_cast<size_t>(local) * arena.s2_bytes, slot.h_w2s, arena.s2_bytes, cudaMemcpyHostToDevice), "stage active s2");
                    check_cuda(cudaMemcpyAsync(arena.w3 + static_cast<size_t>(local) * arena.w3_bytes, slot.h_w3q, arena.w3_bytes, cudaMemcpyHostToDevice), "stage active w3");
                    check_cuda(cudaMemcpyAsync(arena.s3 + static_cast<size_t>(local) * arena.s3_bytes, slot.h_w3s, arena.s3_bytes, cudaMemcpyHostToDevice), "stage active s3");
                }
            }
            check_cuda(cudaMemcpy(d_route_indices, route_indices.data(), route_indices.size() * sizeof(int64_t), cudaMemcpyHostToDevice), "copy active route indices");
            check_cuda(cudaMemcpy(d_route_weights, route_weights.data(), route_weights.size() * sizeof(float), cudaMemcpyHostToDevice), "copy active route weights");
            if (!moe_single_token_fp4_cuda(d_ffn_norm, d_route_indices, d_route_weights, arena.w1, arena.s1, arena.w2, arena.s2, arena.w3, arena.s3, d_resid2, static_cast<int>(route_indices.size()), expert_start, experts_per_rank, dim, inter, static_cast<float>(config.swiglu_limit))) {
                throw std::runtime_error("active fp4 moe launch failed");
            }
            if (!vector_accum_cuda(d_resid2, d_moe, dim, 1.0f)) throw std::runtime_error("moe accum failed");
        }
#ifdef DSV4_HAVE_NCCL
        if (ctx.options.tp_world > 1) {
            if (ctx.options.nccl_id_path.empty()) throw std::runtime_error("TP MoE all-reduce requires --nccl-id-path");
            nccl_all_reduce_sum_float_inplace(ctx.options.tp_world, ctx.options.tp_rank, ctx.options.device, ctx.options.nccl_id_path.c_str(), d_moe, dim);
        }
#endif
        moe_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();
        {
            DeviceSharedCache& shared = ctx.shared_device_cache(li, tp_world, tp_rank, dim);
            const int shared_inter = inter;
            const int local_shared_inter = shared_inter / tp_world;
            if (!fp8_e4m3_e8m0_matvec_cuda(d_ffn_norm, shared.w1, shared.s1, d_gate, local_shared_inter, dim)) throw std::runtime_error("shared w1 launch failed");
            if (!fp8_e4m3_e8m0_matvec_cuda(d_ffn_norm, shared.w3, shared.s3, d_up, local_shared_inter, dim)) throw std::runtime_error("shared w3 launch failed");
            if (!silu_mul_cuda(d_gate, d_up, d_hidden, local_shared_inter)) throw std::runtime_error("shared silu launch failed");
            if (!fp8_e4m3_e8m0_matvec_cuda(d_hidden, shared.w2, shared.s2, d_resid2, dim, local_shared_inter)) throw std::runtime_error("shared w2 launch failed");
#ifdef DSV4_HAVE_NCCL
            if (ctx.options.tp_world > 1) {
                if (ctx.options.nccl_id_path.empty()) throw std::runtime_error("TP shared expert all-reduce requires --nccl-id-path");
                nccl_all_reduce_sum_float_inplace(ctx.options.tp_world, ctx.options.tp_rank, ctx.options.device, ctx.options.nccl_id_path.c_str(), d_resid2, dim);
            }
#endif
            if (!vector_accum_cuda(d_resid2, d_moe, dim, 1.0f)) throw std::runtime_error("shared accum failed");
        }
        shared_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();
        check_cuda(cudaMemcpy(host_x.data(), d_moe, static_cast<size_t>(dim) * sizeof(float), cudaMemcpyDeviceToHost), "copy moe host");
        if (debug_forward) print_summary("layer=" + std::to_string(li) + ".moe_out", host_x);
        h4 = hc_post_cpu(host_x, h4, ffn_hc, dim);
        round_vector_to_bf16(h4);
        post_ms += elapsed_ms(stage_t, Clock::now());
        if (profile_forward) {
            const double layer_ms = elapsed_ms(layer_t0, Clock::now());
            std::cout << "CPP_PROFILE layer=" << li
                      << " total_ms=" << layer_ms
                      << " load_ms=" << load_ms
                      << " hc_ms=" << hc_ms
                      << " attn_ms=" << attn_ms
                      << " route_ms=" << route_ms
                      << " moe_ms=" << moe_ms
                      << " shared_ms=" << shared_ms
                      << " post_ms=" << post_ms << "\n";
        }
        total_load_ms += load_ms;
        total_hc_ms += hc_ms;
        total_attn_ms += attn_ms;
        total_route_ms += route_ms;
        total_moe_ms += moe_ms;
        total_shared_ms += shared_ms;
        total_post_ms += post_ms;
        if (debug_forward) print_summary("layer=" + std::to_string(li), h4);
    }

    if (profile_forward) {
        std::cout << "CPP_PROFILE_TOTAL load_ms=" << total_load_ms
                  << " hc_ms=" << total_hc_ms
                  << " attn_ms=" << total_attn_ms
                  << " route_ms=" << total_route_ms
                  << " moe_ms=" << total_moe_ms
                  << " shared_ms=" << total_shared_ms
                  << " post_ms=" << total_post_ms << "\n";
    }
    if (debug_forward) print_summary("final_h", h4);
    host_x = hc_head_cpu(
        h4,
        reinterpret_cast<const float*>(ctx.hc_head_shard.tensor_data(*ctx.hc_head_fn)),
        reinterpret_cast<const float*>(ctx.hc_head_shard.tensor_data(*ctx.hc_head_scale)),
        reinterpret_cast<const float*>(ctx.hc_head_shard.tensor_data(*ctx.hc_head_base)),
        dim);
    check_cuda(cudaMemcpy(d_x, host_x.data(), static_cast<size_t>(dim) * sizeof(float), cudaMemcpyHostToDevice), "copy hc head");
    if (!rmsnorm_bf16_gamma_cuda(d_x, d_final_norm_gamma, d_resid1, dim, 1e-6f)) throw std::runtime_error("final norm launch failed");
    if (!bf16_matvec_cuda(d_resid1, d_head, d_logits, local_head_rows, dim)) throw std::runtime_error("head launch failed");
    check_cuda(cudaDeviceSynchronize(), "sync kernels");

    std::vector<float> logits(local_head_rows);
    check_cuda(cudaMemcpy(logits.data(), d_logits, logits.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy logits");
    float checksum = 0.0f;
    int top_token = local_head_start;
    float top_logit = -INFINITY;
    for (int i = 0; i < local_head_rows; ++i) {
        const float v = logits[i];
        checksum += v;
        if (v > top_logit) {
            top_logit = v;
            top_token = local_head_start + i;
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
    cudaFree(d_attn_sink);
    cudaFree(d_ffn_gamma);
    cudaFree(d_w1);
    cudaFree(d_s1);
    cudaFree(d_w2);
    cudaFree(d_s2);
    cudaFree(d_w3);
    cudaFree(d_s3);
    cudaFree(d_head);
    cudaFree(d_final_norm_gamma);
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
    cudaFree(d_index_q);
    cudaFree(d_indexer_kv);
    cudaFree(d_index_weight_proj);
    cudaFree(d_index_selected);
    cudaFree(d_kv_indices);
    cudaFree(d_resid1);
    cudaFree(d_ffn_norm);
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_hidden);
    cudaFree(d_moe);
    cudaFree(d_resid2);
    cudaFree(d_route_indices);
    cudaFree(d_route_weights);
    cudaFree(d_logits);

    return ForwardSmokeResult{token, dim, inter, head_rows, layer_count, top_token, top_logit, checksum};
}

ForwardSmokeResult run_safetensors_token_forward_at_position(const std::string& ckpt_dir, int token, int layer_count, int position) {
    return run_safetensors_token_forward_with_options(ckpt_dir, token, layer_count, position, ForwardSmokeOptions{});
}

ForwardSmokeResult run_safetensors_token_forward_with_options(const std::string& ckpt_dir, int token, int layer_count, int position, const ForwardSmokeOptions& options) {
    SafeForwardContext ctx(ckpt_dir);
    ctx.options = options;
    return run_safetensors_token_forward_impl(ctx, token, layer_count, position);
}

}  // namespace dsv4
