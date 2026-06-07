#include "dsv4_engine.hpp"

#include "cuda_ops.hpp"
#include "cmd_channel.hpp"
#include "persistent_engine.hpp"
#include "safetensors_reader.hpp"
#include "sampler.hpp"
#include "tokenizer.hpp"
#include "tp_comm.hpp"
#include "weight_source.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <chrono>
#include <random>
#include <sstream>
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

// Guards the safetensors-only forward path against GGUF checkpoints. Returns
// `dir` unchanged for the common safetensors case so it can be chained into
// SafeForwardContext::ckpt_dir's initializer. GGUF dense forward is not yet
// wired into SafeForwardContext; we surface that here instead of crashing on
// model.safetensors.index.json open.
const std::string& check_safetensors_path(const std::string& dir) {
    if (is_gguf_path(dir)) {
        throw std::runtime_error(
            "cpp_engine: safetensors entry point received a GGUF path '" + dir +
            "'. GGUF Q2 dense forward is not yet wired up — use the FP4 "
            "safetensors directory for now.");
    }
    return dir;
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

float fp8_e4m3_to_float(uint8_t code) {
    const int sign = (code >> 7) & 0x1;
    const int exp = (code >> 3) & 0xf;
    const int mant = code & 0x7;
    const float value = exp == 0
        ? std::ldexp(static_cast<float>(mant) * 0.125f, -6)
        : std::ldexp(1.0f + static_cast<float>(mant) * 0.125f, exp - 7);
    return sign ? -value : value;
}

float e8m0_to_float(uint8_t code) {
    return std::exp2(static_cast<float>(static_cast<int>(code) - 127));
}

#ifdef DSV4_HAVE_NCCL
struct BF16AllReduceScratch {
    uint16_t* d_bf16 = nullptr;
    int capacity = 0;
    ~BF16AllReduceScratch() {
        if (d_bf16 != nullptr) cudaFree(d_bf16);
    }
    void ensure(int count) {
        if (count <= capacity) return;
        if (d_bf16 != nullptr) cudaFree(d_bf16);
        d_bf16 = nullptr;
        cudaError_t err = cudaMalloc(&d_bf16, static_cast<size_t>(count) * sizeof(uint16_t));
        if (err != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc bf16 all-reduce scratch: ") + cudaGetErrorString(err));
        capacity = count;
    }
};

struct ReduceBreakdown {
    bool enabled = false;
    double pre_sync_ms = 0.0;
    double pack_ms = 0.0;
    double nccl_ms = 0.0;
    double unpack_ms = 0.0;
};

void all_reduce_sum_fp32_via_bf16_inplace(
    int world,
    int rank,
    int device,
    const char* id_path,
    float* d_values,
    int count,
    BF16AllReduceScratch& scratch,
    ReduceBreakdown* detail = nullptr) {
    scratch.ensure(count);
    using Clock = std::chrono::steady_clock;
    auto elapsed_ms_local = [](Clock::time_point t0, Clock::time_point t1) {
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };
    if (detail && detail->enabled) {
        auto pre_t = Clock::now();
        check_cuda(cudaDeviceSynchronize(), "sync reduce pre");
        detail->pre_sync_ms += elapsed_ms_local(pre_t, Clock::now());
        auto pack_t = Clock::now();
        if (!fp32_to_bf16_cuda(d_values, scratch.d_bf16, count)) throw std::runtime_error("fp32_to_bf16 launch failed");
        check_cuda(cudaDeviceSynchronize(), "sync reduce pack");
        detail->pack_ms += elapsed_ms_local(pack_t, Clock::now());
        auto nccl_t = Clock::now();
        nccl_all_reduce_sum_bf16_inplace(world, rank, device, id_path, scratch.d_bf16, count);
        check_cuda(cudaDeviceSynchronize(), "sync reduce nccl");
        detail->nccl_ms += elapsed_ms_local(nccl_t, Clock::now());
        auto unpack_t = Clock::now();
        if (!bf16_to_fp32_cuda(scratch.d_bf16, d_values, count)) throw std::runtime_error("bf16_to_fp32 launch failed");
        check_cuda(cudaDeviceSynchronize(), "sync reduce unpack");
        detail->unpack_ms += elapsed_ms_local(unpack_t, Clock::now());
        return;
    }
    if (!fp32_to_bf16_cuda(d_values, scratch.d_bf16, count)) throw std::runtime_error("fp32_to_bf16 launch failed");
    nccl_all_reduce_sum_bf16_inplace(world, rank, device, id_path, scratch.d_bf16, count);
    if (!bf16_to_fp32_cuda(scratch.d_bf16, d_values, count)) throw std::runtime_error("bf16_to_fp32 launch failed");
}
#endif

struct GgufReduceContext {
    int world = 1;
    int rank = 0;
    int device = 0;
    const char* id_path = nullptr;
#ifdef DSV4_HAVE_NCCL
    BF16AllReduceScratch* scratch = nullptr;
#endif
};

void gguf_all_reduce_sum_fp32_inplace(float* d_values, int count, const GgufReduceContext* ctx, const char* what) {
    if (ctx == nullptr || ctx->world <= 1) return;
#ifdef DSV4_HAVE_NCCL
    if (ctx->id_path == nullptr || ctx->id_path[0] == '\0') throw std::runtime_error(std::string(what) + " requires NCCL id path");
    static const bool use_fp32_reduce = [] {
        const char* v = std::getenv("DSV4_GGUF_REDUCE_FP32");
        if (v == nullptr) return true;  // default fp32 for Q2 (bf16 round-trip compounds noise)
        try { return std::stoi(v) != 0; } catch (...) { return true; }
    }();
    if (use_fp32_reduce) {
        nccl_all_reduce_sum_float_inplace(ctx->world, ctx->rank, ctx->device, ctx->id_path, d_values, count);
        return;
    }
    if (ctx->scratch == nullptr) throw std::runtime_error(std::string(what) + " requires BF16 reduce scratch");
    all_reduce_sum_fp32_via_bf16_inplace(ctx->world, ctx->rank, ctx->device, ctx->id_path, d_values, count, *ctx->scratch);
#else
    (void)d_values;
    (void)count;
    throw std::runtime_error(std::string(what) + " requires DSV4_HAVE_NCCL");
#endif
}

struct WoAInt8Host {
    std::vector<int8_t> weight;
    std::vector<float> scale;
};

WoAInt8Host make_wo_a_int8_from_fp8(
    const uint8_t* weight,
    const uint8_t* scale,
    int rows,
    int cols,
    int scale_cols) {
    WoAInt8Host out;
    out.weight.resize(static_cast<size_t>(rows) * cols);
    out.scale.resize(rows);
    std::vector<float> row(cols);
    for (int r = 0; r < rows; ++r) {
        float amax = 0.0f;
        const int rb = r / 128;
        for (int c = 0; c < cols; ++c) {
            const float v = fp8_e4m3_to_float(weight[static_cast<size_t>(r) * cols + c]) * e8m0_to_float(scale[static_cast<size_t>(rb) * scale_cols + c / 128]);
            row[c] = v;
            amax = std::max(amax, std::fabs(v));
        }
        const float row_scale = std::max(amax, 1.0e-6f) / 127.0f;
        out.scale[r] = row_scale;
        const float inv_scale = 1.0f / row_scale;
        for (int c = 0; c < cols; ++c) {
            int q = static_cast<int>(std::nearbyint(row[c] * inv_scale));
            q = std::max(-127, std::min(127, q));
            out.weight[static_cast<size_t>(r) * cols + c] = static_cast<int8_t>(q);
        }
    }
    return out;
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

size_t q8_0_row_bytes(int cols) {
    return static_cast<size_t>((cols + 31) / 32) * 34;
}

std::vector<uint8_t> slice_q8_0_rows(const uint8_t* src, int row_start, int rows, int cols) {
    const size_t row_bytes = q8_0_row_bytes(cols);
    std::vector<uint8_t> out(static_cast<size_t>(rows) * row_bytes);
    std::memcpy(out.data(), src + static_cast<size_t>(row_start) * row_bytes, out.size());
    return out;
}

std::vector<uint8_t> slice_q8_0_cols(const uint8_t* src, int rows, int cols, int col_start, int col_count) {
    if ((col_start % 32) != 0 || (col_count % 32) != 0) {
        throw std::runtime_error("Q8_0 column slices must align to 32 columns");
    }
    const int src_blocks = (cols + 31) / 32;
    const int dst_blocks = (col_count + 31) / 32;
    std::vector<uint8_t> out(static_cast<size_t>(rows) * static_cast<size_t>(dst_blocks) * 34);
    for (int r = 0; r < rows; ++r) {
        const uint8_t* src_row = src + static_cast<size_t>(r) * static_cast<size_t>(src_blocks) * 34 + static_cast<size_t>(col_start / 32) * 34;
        uint8_t* dst_row = out.data() + static_cast<size_t>(r) * static_cast<size_t>(dst_blocks) * 34;
        std::memcpy(dst_row, src_row, static_cast<size_t>(dst_blocks) * 34);
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

std::vector<HcPreResult> hc_pre_rows_cpu(const std::vector<float>& h4_rows, const float* fn, const float* scale, const float* base, int tokens, int dim) {
    std::vector<HcPreResult> out;
    out.reserve(tokens);
    const size_t row_stride = static_cast<size_t>(4) * dim;
    for (int t = 0; t < tokens; ++t) {
        std::vector<float> row(h4_rows.begin() + static_cast<size_t>(t) * row_stride, h4_rows.begin() + static_cast<size_t>(t + 1) * row_stride);
        out.push_back(hc_pre_cpu(row, fn, scale, base, dim));
    }
    return out;
}

std::vector<float> hc_pre_x_rows_cpu(const std::vector<HcPreResult>& pre_rows, int dim) {
    std::vector<float> out(static_cast<size_t>(pre_rows.size()) * dim);
    for (size_t t = 0; t < pre_rows.size(); ++t) {
        std::copy(pre_rows[t].x.begin(), pre_rows[t].x.end(), out.begin() + t * static_cast<size_t>(dim));
    }
    return out;
}

std::vector<float> hc_post_rows_cpu(const std::vector<float>& x_rows, const std::vector<float>& residual_rows, const std::vector<HcPreResult>& pre_rows, int dim) {
    const int tokens = static_cast<int>(pre_rows.size());
    const size_t h4_stride = static_cast<size_t>(4) * dim;
    std::vector<float> out(static_cast<size_t>(tokens) * h4_stride);
    for (int t = 0; t < tokens; ++t) {
        std::vector<float> x(x_rows.begin() + static_cast<size_t>(t) * dim, x_rows.begin() + static_cast<size_t>(t + 1) * dim);
        std::vector<float> residual(residual_rows.begin() + static_cast<size_t>(t) * h4_stride, residual_rows.begin() + static_cast<size_t>(t + 1) * h4_stride);
        std::vector<float> row = hc_post_cpu(x, residual, pre_rows[static_cast<size_t>(t)], dim);
        std::copy(row.begin(), row.end(), out.begin() + static_cast<size_t>(t) * h4_stride);
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
    const float* d_inv_freqs = nullptr;
};

struct AttentionProfileBreakdown {
    bool enabled = false;
    double q_ms = 0.0;
    double kv_ms = 0.0;
    double core_ms = 0.0;
    double wo_a_ms = 0.0;
    double wo_b_ms = 0.0;
    double reduce_ms = 0.0;
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
    const int8_t* d_wo_a_int8,
    const float* d_wo_a_int8_scale,
    int8_t* d_wo_a_x_q,
    float* d_wo_a_x_scale,
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
    float* d_attn_out,
    AttentionProfileBreakdown* profile = nullptr) {
    auto profile_stage_sync = [&](const char* what) {
        if (profile != nullptr && profile->enabled) check_cuda(cudaDeviceSynchronize(), what);
    };
    auto profile_t = Clock::now();
    if (!rmsnorm_bf16_gamma_cuda(d_x, d_attn_gamma, d_attn_norm, dims.dim, 1e-6f)) return false;
    if (!fp8_e4m3_e8m0_matvec_cuda(d_attn_norm, d_wq_a, d_wq_a_scale, d_q_a, dims.q_a_dim, dims.dim)) return false;
    if (!rmsnorm_bf16_gamma_cuda(d_q_a, d_q_gamma, d_q_norm, dims.q_a_dim, 1e-6f)) return false;
    if (!fp8_e4m3_e8m0_matvec_cuda(d_q_norm, d_wq_b, d_wq_b_scale, d_q, dims.q_dim, dims.q_a_dim)) return false;
    if (dims.d_inv_freqs != nullptr) {
        if (!head_rmsnorm_rope_freqs_cuda(d_q, dims.d_inv_freqs, dims.heads, dims.head_dim, dims.rope_dim, dims.position, false, 1e-6f)) return false;
    } else {
        if (!head_rmsnorm_rope_cuda(d_q, dims.heads, dims.head_dim, dims.rope_dim, dims.position, dims.rope_theta, false, 1e-6f)) return false;
    }
    profile_stage_sync("decode attn q");
    if (profile != nullptr && profile->enabled) profile->q_ms += elapsed_ms(profile_t, Clock::now());
    profile_t = Clock::now();
    if (!fp8_e4m3_e8m0_matvec_cuda(d_attn_norm, d_wkv, d_wkv_scale, d_kv_a, dims.kv_dim, dims.dim)) return false;
    if (!rmsnorm_bf16_gamma_cuda(d_kv_a, d_kv_gamma, d_kv_norm, dims.kv_dim, 1e-6f)) return false;
    if (dims.d_inv_freqs != nullptr) {
        if (!head_rmsnorm_rope_freqs_cuda(d_kv_norm, dims.d_inv_freqs, 1, dims.head_dim, dims.rope_dim, dims.position, false, 0.0f)) return false;
    } else {
        if (!head_rmsnorm_rope_cuda(d_kv_norm, 1, dims.head_dim, dims.rope_dim, dims.position, dims.rope_theta, false, 0.0f)) return false;
    }
    if (!fp8_act_quant_dequant_cuda(d_kv_norm, dims.head_dim - dims.rope_dim, 64)) return false;
    profile_stage_sync("decode attn kv");
    if (profile != nullptr && profile->enabled) profile->kv_ms += elapsed_ms(profile_t, Clock::now());
    profile_t = Clock::now();
    if (d_kv_cache != nullptr) {
        if (cudaMemcpy(d_kv_cache + static_cast<size_t>(dims.cache_write_slot) * dims.head_dim, d_kv_norm, static_cast<size_t>(dims.head_dim) * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) return false;
        if (d_kv_indices != nullptr && index_count > 0) {
            if (!indexed_cached_single_token_attention_cuda(d_q, d_kv_cache, d_kv_indices, d_attn_sink, d_attn_value, dims.heads, dims.head_dim, index_count, 1.0f / std::sqrt(static_cast<float>(dims.head_dim)))) return false;
        } else if (!cached_single_token_attention_cuda(d_q, d_kv_cache, d_attn_sink, d_attn_value, dims.heads, dims.head_dim, cache_len, 1.0f / std::sqrt(static_cast<float>(dims.head_dim)))) return false;
    } else if (!single_token_sparse_attention_cuda(d_q, d_kv_norm, d_attn_sink, d_attn_value, dims.heads, dims.head_dim, 1.0f / std::sqrt(static_cast<float>(dims.head_dim)))) return false;
    if (dims.d_inv_freqs != nullptr) {
        if (!head_rmsnorm_rope_freqs_cuda(d_attn_value, dims.d_inv_freqs, dims.heads, dims.head_dim, dims.rope_dim, dims.position, true, 0.0f)) return false;
    } else {
        if (!head_rmsnorm_rope_cuda(d_attn_value, dims.heads, dims.head_dim, dims.rope_dim, dims.position, dims.rope_theta, true, 0.0f)) return false;
    }
    profile_stage_sync("decode attn core");
    if (profile != nullptr && profile->enabled) profile->core_ms += elapsed_ms(profile_t, Clock::now());
    profile_t = Clock::now();
    if (d_wo_a_int8 != nullptr && d_wo_a_int8_scale != nullptr && d_wo_a_x_q != nullptr && d_wo_a_x_scale != nullptr) {
        if (!wo_a_int8_decode_cuda(d_attn_value, d_wo_a_int8, d_wo_a_int8_scale, d_attn_mid, dims.groups, dims.group_rank, dims.group_dim, d_wo_a_x_q, d_wo_a_x_scale)) return false;
    } else {
        for (int g = 0; g < dims.groups; ++g) {
            const float* group_x = d_attn_value + static_cast<size_t>(g) * dims.group_dim;
            const uint8_t* group_w = d_wo_a + static_cast<size_t>(g) * dims.group_rank * dims.group_dim;
            const uint8_t* group_s = d_wo_a_scale + static_cast<size_t>(g) * (dims.group_rank / 128) * (dims.group_dim / 128);
            float* group_y = d_attn_mid + static_cast<size_t>(g) * dims.group_rank;
            if (!fp8_e4m3_e8m0_matvec_cuda(group_x, group_w, group_s, group_y, dims.group_rank, dims.group_dim)) return false;
        }
    }
    profile_stage_sync("decode attn wo_a");
    if (profile != nullptr && profile->enabled) profile->wo_a_ms += elapsed_ms(profile_t, Clock::now());
    profile_t = Clock::now();
    if (!fp8_e4m3_e8m0_matvec_cuda(d_attn_mid, d_wo_b, d_wo_b_scale, d_attn_out, dims.dim, dims.attn_mid)) return false;
    profile_stage_sync("decode attn wo_b");
    if (profile != nullptr && profile->enabled) profile->wo_b_ms += elapsed_ms(profile_t, Clock::now());
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

struct DeviceCompressorCache {
    uint16_t* wkv = nullptr;
    uint16_t* wgate = nullptr;
    float* ape = nullptr;
    uint16_t* norm = nullptr;
    int cols = 0;
    int dim = 0;
};

struct DeviceCompressorState {
    float* kv = nullptr;
    float* score = nullptr;
    int slots = 0;
    int cols = 0;
};

struct DeviceMoeDecodeWorkspace {
    MoeSingleTokenFp4Workspace fp4;
};

struct DeviceMoePrefillWorkspace {
    MoePrefillFp4GroupedWorkspace fp4;
    ~DeviceMoePrefillWorkspace() { release(); }

    void release() {
        cudaFree(fp4.d_x_sorted);
        cudaFree(fp4.d_x_q);
        cudaFree(fp4.d_x_scale);
        cudaFree(fp4.d_x_pad);
        cudaFree(fp4.d_x_scale_pad);
        cudaFree(fp4.d_gate);
        cudaFree(fp4.d_up);
        cudaFree(fp4.d_hidden_q);
        cudaFree(fp4.d_hidden_scale);
        cudaFree(fp4.d_tile_experts);
        cudaFree(fp4.d_tile_rows);
        fp4 = MoePrefillFp4GroupedWorkspace{};
    }

    void ensure(int routes_cap, int tile_cap, int dim, int inter_dim, int padded_rows_cap = 0) {
        if (padded_rows_cap <= 0) padded_rows_cap = routes_cap;
        if (fp4.d_x_sorted != nullptr && fp4.routes_cap >= routes_cap && fp4.padded_rows_cap >= padded_rows_cap && fp4.tile_cap >= tile_cap && fp4.dim == dim && fp4.inter_dim == inter_dim) return;
        release();
        fp4.dim = dim;
        fp4.inter_dim = inter_dim;
        fp4.routes_cap = routes_cap;
        fp4.padded_rows_cap = padded_rows_cap;
        fp4.tile_cap = tile_cap;
        const size_t routes_dim = static_cast<size_t>(routes_cap) * dim;
        const size_t routes_inter = static_cast<size_t>(routes_cap) * inter_dim;
        const size_t padded_dim = static_cast<size_t>(padded_rows_cap) * dim;
        const size_t padded_inter = static_cast<size_t>(padded_rows_cap) * inter_dim;
        check_cuda(cudaMalloc(&fp4.d_x_sorted, routes_dim * sizeof(float)), "cudaMalloc prefill moe x sorted");
        check_cuda(cudaMalloc(&fp4.d_x_q, routes_dim), "cudaMalloc prefill moe x q");
        check_cuda(cudaMalloc(&fp4.d_x_scale, static_cast<size_t>(routes_cap) * sizeof(float)), "cudaMalloc prefill moe x scale");
        check_cuda(cudaMalloc(&fp4.d_x_pad, padded_dim), "cudaMalloc prefill moe x pad");
        check_cuda(cudaMalloc(&fp4.d_x_scale_pad, static_cast<size_t>(padded_rows_cap) * sizeof(float)), "cudaMalloc prefill moe x scale pad");
        check_cuda(cudaMalloc(&fp4.d_gate, padded_inter * sizeof(float)), "cudaMalloc prefill moe gate");
        check_cuda(cudaMalloc(&fp4.d_up, padded_inter * sizeof(float)), "cudaMalloc prefill moe up");
        check_cuda(cudaMalloc(&fp4.d_hidden_q, padded_inter), "cudaMalloc prefill moe hidden q");
        check_cuda(cudaMalloc(&fp4.d_hidden_scale, static_cast<size_t>(padded_rows_cap) * sizeof(float)), "cudaMalloc prefill moe hidden scale");
        if (tile_cap > 0) {
            check_cuda(cudaMalloc(&fp4.d_tile_experts, static_cast<size_t>(tile_cap) * sizeof(int32_t)), "cudaMalloc prefill moe tile experts");
            check_cuda(cudaMalloc(&fp4.d_tile_rows, static_cast<size_t>(tile_cap) * sizeof(int32_t)), "cudaMalloc prefill moe tile rows");
        }
    }
};

struct DeviceHcCache {
    float* attn_fn = nullptr;
    float* attn_scale = nullptr;
    float* attn_base = nullptr;
    float* ffn_fn = nullptr;
    float* ffn_scale = nullptr;
    float* ffn_base = nullptr;
};

struct DeviceAttentionCache {
    uint16_t* attn_norm = nullptr;
    uint16_t* q_norm = nullptr;
    uint16_t* kv_norm = nullptr;
    uint16_t* ffn_norm = nullptr;
    uint8_t* wq_a = nullptr;
    uint8_t* wq_a_scale = nullptr;
    uint8_t* wq_b = nullptr;
    uint8_t* wq_b_scale = nullptr;
    uint8_t* wkv = nullptr;
    uint8_t* wkv_scale = nullptr;
    uint8_t* wo_a = nullptr;
    uint8_t* wo_a_scale = nullptr;
    int8_t* wo_a_int8 = nullptr;
    float* wo_a_int8_scale = nullptr;
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

int gguf_indexed_attn_mode_from_env() {
    const char* value = std::getenv("DSV4_GGUF_INDEXED_ATTN");
    if (value == nullptr || *value == '\0') return 0;
    const std::string s(value);
    if (s == "auto" || s == "AUTO") return 2;
    try {
        return std::stoi(s);
    } catch (...) {
        return 0;
    }
}

void gguf_log_mem(const char* tag, int tp_rank) {
    size_t free_bytes = 0, total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), tag);
    std::cout << "gguf_mem tag=" << tag
              << " tp_rank=" << tp_rank
              << " free_gib=" << (static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0))
              << " total_gib=" << (static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0))
              << " used_gib=" << (static_cast<double>(total_bytes - free_bytes) / (1024.0 * 1024.0 * 1024.0))
              << "\n";
}

void gguf_check_min_free(const char* tag, int tp_rank, int min_free_mib) {
    if (min_free_mib <= 0) return;
    size_t free_bytes = 0, total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), tag);
    const size_t min_bytes = static_cast<size_t>(min_free_mib) * 1024ULL * 1024ULL;
    if (free_bytes < min_bytes) {
        std::ostringstream oss;
        oss << "GGUF free memory below guard after " << tag
            << ": tp_rank=" << tp_rank
            << " free_mib=" << (static_cast<double>(free_bytes) / (1024.0 * 1024.0))
            << " min_free_mib=" << min_free_mib;
        throw std::runtime_error(oss.str());
    }
}

struct GgufPinnedStageSlot {
    uint8_t* ptr = nullptr;
    size_t capacity = 0;
    cudaEvent_t done = nullptr;
    bool in_flight = false;
};

struct GgufPinnedStagingRing {
    std::vector<GgufPinnedStageSlot> slots;
    size_t next = 0;

    ~GgufPinnedStagingRing() { release(); }

    bool enabled() const { return !slots.empty(); }

    void init(size_t slot_bytes, int requested_slots, int cap_mib, int tp_rank) {
        release();
        if (slot_bytes == 0 || requested_slots <= 0 || cap_mib <= 0) return;
        const size_t cap_bytes = static_cast<size_t>(cap_mib) * 1024ULL * 1024ULL;
        if (slot_bytes > cap_bytes) {
            if (tp_rank == 0) {
                std::cout << "gguf_pinned_staging=0 reason=slot_exceeds_cap"
                          << " slot_bytes=" << slot_bytes
                          << " cap_mib=" << cap_mib << "\n";
            }
            return;
        }
        const size_t max_slots_by_cap = std::max<size_t>(1, cap_bytes / slot_bytes);
        const int slots_to_alloc = static_cast<int>(std::min<size_t>(
            static_cast<size_t>(requested_slots), max_slots_by_cap));
        slots.resize(static_cast<size_t>(slots_to_alloc));
        for (auto& slot : slots) {
            slot.capacity = slot_bytes;
            check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&slot.ptr), slot.capacity, cudaHostAllocDefault),
                       "cudaHostAlloc GGUF pinned staging slot");
            check_cuda(cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming),
                       "create GGUF pinned staging event");
        }
        next = 0;
        if (tp_rank == 0) {
            const double total_mib = static_cast<double>(slot_bytes) * static_cast<double>(slots_to_alloc) /
                (1024.0 * 1024.0);
            std::cout << "gguf_pinned_staging=1"
                      << " slot_bytes=" << slot_bytes
                      << " slots=" << slots_to_alloc
                      << " total_mib=" << total_mib
                      << " cap_mib=" << cap_mib << "\n";
        }
    }

    void copy_async(uint8_t* dst, const uint8_t* src, size_t bytes, cudaStream_t stream, const char* label) {
        if (!enabled()) {
            check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream), label);
            return;
        }
        GgufPinnedStageSlot& slot = slots[next];
        if (bytes > slot.capacity) {
            throw std::runtime_error("GGUF pinned staging slot too small");
        }
        if (slot.in_flight) {
            check_cuda(cudaEventSynchronize(slot.done), "wait GGUF pinned staging slot");
            slot.in_flight = false;
        }
        std::memcpy(slot.ptr, src, bytes);
        check_cuda(cudaMemcpyAsync(dst, slot.ptr, bytes, cudaMemcpyHostToDevice, stream), label);
        check_cuda(cudaEventRecord(slot.done, stream), "record GGUF pinned staging slot");
        slot.in_flight = true;
        next = (next + 1) % slots.size();
    }

    void release() {
        for (auto& slot : slots) {
            if (slot.in_flight && slot.done != nullptr) cudaEventSynchronize(slot.done);
            if (slot.ptr != nullptr) cudaFreeHost(slot.ptr);
            if (slot.done != nullptr) cudaEventDestroy(slot.done);
            slot = GgufPinnedStageSlot{};
        }
        slots.clear();
        next = 0;
    }
};

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
    // Sparse slot mapping (decode path): maps a routed expert's local_id to
    // a small slot index inside this arena (range [0, capacity)). Enables
    // per-layer arenas to stay resident across decode steps with a small
    // total GPU footprint via LRU eviction within the arena.
    bool sparse_slots = false;
    std::unordered_map<int, int> slot_by_local;
    std::vector<int> slot_local;
    std::list<int> slot_lru;
    std::vector<std::list<int>::iterator> slot_lru_pos;
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

struct DeviceIndexerCache {
    uint8_t* wq_b = nullptr;
    uint8_t* wq_b_scale = nullptr;
    uint16_t* weights_proj = nullptr;
};

struct DeviceGateCache {
    uint16_t* weight = nullptr;
    float* bias = nullptr;
    int64_t* tid2eid = nullptr;
    float* original = nullptr;
    float* scored = nullptr;
    int experts = 0;
    int dim = 0;
    int hash_topk = 0;
};

struct SafeForwardContext {
    explicit SafeForwardContext(const std::string& dir)
        : ckpt_dir(check_safetensors_path(dir)),
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
        for (auto& [_, ptr] : rope_inv_freqs_compress) cudaFree(ptr);
        for (auto& [_, ptr] : rope_inv_freqs_plain) cudaFree(ptr);
        for (auto& [_, c] : attention_cache) {
            cudaFree(c.attn_norm);
            cudaFree(c.q_norm);
            cudaFree(c.kv_norm);
            cudaFree(c.ffn_norm);
            cudaFree(c.wq_a);
            cudaFree(c.wq_a_scale);
            cudaFree(c.wq_b);
            cudaFree(c.wq_b_scale);
            cudaFree(c.wkv);
            cudaFree(c.wkv_scale);
            cudaFree(c.wo_a);
            cudaFree(c.wo_a_scale);
            cudaFree(c.wo_a_int8);
            cudaFree(c.wo_a_int8_scale);
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
            cudaFree(c.tid2eid);
            cudaFree(c.original);
            cudaFree(c.scored);
        }
        for (auto& [_, c] : compressor_cache) {
            cudaFree(c.wkv);
            cudaFree(c.wgate);
            cudaFree(c.ape);
            cudaFree(c.norm);
        }
        for (auto& [_, c] : hc_cache) {
            cudaFree(c.attn_fn);
            cudaFree(c.attn_scale);
            cudaFree(c.attn_base);
            cudaFree(c.ffn_fn);
            cudaFree(c.ffn_scale);
            cudaFree(c.ffn_base);
        }
        for (auto& [_, w] : moe_decode_workspace_cache) {
            cudaFree(w.fp4.d_x_q);
            cudaFree(w.fp4.d_x_scale);
            cudaFree(w.fp4.d_gate);
            cudaFree(w.fp4.d_up);
            cudaFree(w.fp4.d_hidden_q);
            cudaFree(w.fp4.d_hidden_scale);
            cudaFree(w.fp4.d_route_y);
        }
        for (auto& [_, c] : indexer_compressor_cache) {
            cudaFree(c.wkv);
            cudaFree(c.wgate);
            cudaFree(c.ape);
            cudaFree(c.norm);
        }
        for (auto& [_, s] : compressor_device_state) {
            cudaFree(s.kv);
            cudaFree(s.score);
        }
        for (auto& [_, s] : indexer_compressor_device_state) {
            cudaFree(s.kv);
            cudaFree(s.score);
        }
        for (auto& [_, c] : indexer_cache) {
            cudaFree(c.wq_b);
            cudaFree(c.wq_b_scale);
            cudaFree(c.weights_proj);
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
        const std::string tid_name = prefix + "ffn.gate.tid2eid";
        if (index.shard_for_tensor(tid_name) != nullptr) {
            SafeTensorsShard& tid_shard = shard_for_tensor(tid_name);
            const auto* tid = require_tensor(tid_shard, tid_name);
            c.hash_topk = static_cast<int>(tid->shape[1]);
            check_cuda(cudaMalloc(&c.tid2eid, tid->nbytes), "cudaMalloc gate tid2eid");
            check_cuda(cudaMemcpy(c.tid2eid, tid_shard.tensor_data(*tid), tid->nbytes, cudaMemcpyHostToDevice), "copy gate tid2eid");
        }
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
        SafeTensorsShard& attn_norm_shard = shard_for_tensor(prefix + "attn_norm.weight");
        SafeTensorsShard& ffn_norm_shard = shard_for_tensor(prefix + "ffn_norm.weight");
        SafeTensorsShard& qkv_shard = shard_for_tensor(prefix + "attn.wq_a.weight");
        SafeTensorsShard& wo_a_shard = shard_for_tensor(prefix + "attn.wo_a.weight");
        SafeTensorsShard& wo_b_shard = shard_for_tensor(prefix + "attn.wo_b.weight");
        const auto* attn_norm = require_tensor(attn_norm_shard, prefix + "attn_norm.weight");
        const auto* q_norm = require_tensor(qkv_shard, prefix + "attn.q_norm.weight");
        const auto* kv_norm = require_tensor(qkv_shard, prefix + "attn.kv_norm.weight");
        const auto* ffn_norm = require_tensor(ffn_norm_shard, prefix + "ffn_norm.weight");
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
        check_cuda(cudaMalloc(&c.attn_norm, attn_norm->nbytes), "cudaMalloc cached attn norm");
        check_cuda(cudaMalloc(&c.q_norm, q_norm->nbytes), "cudaMalloc cached q norm");
        check_cuda(cudaMalloc(&c.kv_norm, kv_norm->nbytes), "cudaMalloc cached kv norm");
        check_cuda(cudaMalloc(&c.ffn_norm, ffn_norm->nbytes), "cudaMalloc cached ffn norm");
        check_cuda(cudaMemcpy(c.attn_norm, attn_norm_shard.tensor_data(*attn_norm), attn_norm->nbytes, cudaMemcpyHostToDevice), "copy cached attn norm");
        check_cuda(cudaMemcpy(c.q_norm, qkv_shard.tensor_data(*q_norm), q_norm->nbytes, cudaMemcpyHostToDevice), "copy cached q norm");
        check_cuda(cudaMemcpy(c.kv_norm, qkv_shard.tensor_data(*kv_norm), kv_norm->nbytes, cudaMemcpyHostToDevice), "copy cached kv norm");
        check_cuda(cudaMemcpy(c.ffn_norm, ffn_norm_shard.tensor_data(*ffn_norm), ffn_norm->nbytes, cudaMemcpyHostToDevice), "copy cached ffn norm");
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
        if (env_int_or_default("DSV4_CPP_DECODE_WO_A_INT8", 0) != 0) {
            WoAInt8Host wo_a_int8 = make_wo_a_int8_from_fp8(wo_a_local.data(), wo_a_scale_local.data(), dims.attn_mid, dims.dim, dims.dim / 128);
            check_cuda(cudaMalloc(&c.wo_a_int8, wo_a_int8.weight.size() * sizeof(int8_t)), "cudaMalloc cached wo_a int8");
            check_cuda(cudaMalloc(&c.wo_a_int8_scale, wo_a_int8.scale.size() * sizeof(float)), "cudaMalloc cached wo_a int8 scale");
            check_cuda(cudaMemcpy(c.wo_a_int8, wo_a_int8.weight.data(), wo_a_int8.weight.size() * sizeof(int8_t), cudaMemcpyHostToDevice), "copy cached wo_a int8");
            check_cuda(cudaMemcpy(c.wo_a_int8_scale, wo_a_int8.scale.data(), wo_a_int8.scale.size() * sizeof(float), cudaMemcpyHostToDevice), "copy cached wo_a int8 scale");
        }
        auto inserted = attention_cache.emplace(key, c);
        return inserted.first->second;
    }
    DeviceCompressorCache& compressor_device_cache(int layer_id) {
        const std::string key = std::to_string(layer_id);
        auto it = compressor_cache.find(key);
        if (it != compressor_cache.end()) return it->second;
        const std::string prefix = "layers." + std::to_string(layer_id) + ".";
        SafeTensorsShard& comp_shard = shard_for_tensor(prefix + "attn.compressor.wkv.weight");
        const auto* wkv = require_tensor(comp_shard, prefix + "attn.compressor.wkv.weight");
        const auto* wgate = require_tensor(comp_shard, prefix + "attn.compressor.wgate.weight");
        const auto* ape = require_tensor(comp_shard, prefix + "attn.compressor.ape");
        const auto* norm = require_tensor(comp_shard, prefix + "attn.compressor.norm.weight");
        DeviceCompressorCache c;
        c.cols = static_cast<int>(wkv->shape[0]);
        c.dim = static_cast<int>(wkv->shape[1]);
        check_cuda(cudaMalloc(&c.wkv, wkv->nbytes), "cudaMalloc cached compressor wkv");
        check_cuda(cudaMalloc(&c.wgate, wgate->nbytes), "cudaMalloc cached compressor wgate");
        check_cuda(cudaMalloc(&c.ape, ape->nbytes), "cudaMalloc cached compressor ape");
        check_cuda(cudaMalloc(&c.norm, norm->nbytes), "cudaMalloc cached compressor norm");
        check_cuda(cudaMemcpy(c.wkv, comp_shard.tensor_data(*wkv), wkv->nbytes, cudaMemcpyHostToDevice), "copy cached compressor wkv");
        check_cuda(cudaMemcpy(c.wgate, comp_shard.tensor_data(*wgate), wgate->nbytes, cudaMemcpyHostToDevice), "copy cached compressor wgate");
        check_cuda(cudaMemcpy(c.ape, comp_shard.tensor_data(*ape), ape->nbytes, cudaMemcpyHostToDevice), "copy cached compressor ape");
        check_cuda(cudaMemcpy(c.norm, comp_shard.tensor_data(*norm), norm->nbytes, cudaMemcpyHostToDevice), "copy cached compressor norm");
        auto inserted = compressor_cache.emplace(key, c);
        return inserted.first->second;
    }

    DeviceCompressorCache& indexer_compressor_device_cache(int layer_id) {
        const std::string key = std::to_string(layer_id);
        auto it = indexer_compressor_cache.find(key);
        if (it != indexer_compressor_cache.end()) return it->second;
        const std::string prefix = "layers." + std::to_string(layer_id) + ".";
        SafeTensorsShard& idx_shard = shard_for_tensor(prefix + "attn.indexer.wq_b.weight");
        const auto* wkv = require_tensor(idx_shard, prefix + "attn.indexer.compressor.wkv.weight");
        const auto* wgate = require_tensor(idx_shard, prefix + "attn.indexer.compressor.wgate.weight");
        const auto* ape = require_tensor(idx_shard, prefix + "attn.indexer.compressor.ape");
        const auto* norm = require_tensor(idx_shard, prefix + "attn.indexer.compressor.norm.weight");
        DeviceCompressorCache c;
        c.cols = static_cast<int>(wkv->shape[0]);
        c.dim = static_cast<int>(wkv->shape[1]);
        check_cuda(cudaMalloc(&c.wkv, wkv->nbytes), "cudaMalloc cached indexer compressor wkv");
        check_cuda(cudaMalloc(&c.wgate, wgate->nbytes), "cudaMalloc cached indexer compressor wgate");
        check_cuda(cudaMalloc(&c.ape, ape->nbytes), "cudaMalloc cached indexer compressor ape");
        check_cuda(cudaMalloc(&c.norm, norm->nbytes), "cudaMalloc cached indexer compressor norm");
        check_cuda(cudaMemcpy(c.wkv, idx_shard.tensor_data(*wkv), wkv->nbytes, cudaMemcpyHostToDevice), "copy cached indexer compressor wkv");
        check_cuda(cudaMemcpy(c.wgate, idx_shard.tensor_data(*wgate), wgate->nbytes, cudaMemcpyHostToDevice), "copy cached indexer compressor wgate");
        check_cuda(cudaMemcpy(c.ape, idx_shard.tensor_data(*ape), ape->nbytes, cudaMemcpyHostToDevice), "copy cached indexer compressor ape");
        check_cuda(cudaMemcpy(c.norm, idx_shard.tensor_data(*norm), norm->nbytes, cudaMemcpyHostToDevice), "copy cached indexer compressor norm");
        auto inserted = indexer_compressor_cache.emplace(key, c);
        return inserted.first->second;
    }


    DeviceIndexerCache& indexer_device_cache(int layer_id) {
        const std::string key = std::to_string(layer_id);
        auto it = indexer_cache.find(key);
        if (it != indexer_cache.end()) return it->second;
        const std::string prefix = "layers." + std::to_string(layer_id) + ".";
        SafeTensorsShard& idx_shard = shard_for_tensor(prefix + "attn.indexer.wq_b.weight");
        const auto* wq_b = require_tensor(idx_shard, prefix + "attn.indexer.wq_b.weight");
        const auto* wq_b_scale = require_tensor(idx_shard, prefix + "attn.indexer.wq_b.scale");
        const auto* weights = require_tensor(idx_shard, prefix + "attn.indexer.weights_proj.weight");
        DeviceIndexerCache c;
        check_cuda(cudaMalloc(&c.wq_b, wq_b->nbytes), "cudaMalloc cached indexer wq_b");
        check_cuda(cudaMalloc(&c.wq_b_scale, wq_b_scale->nbytes), "cudaMalloc cached indexer wq_b scale");
        check_cuda(cudaMalloc(&c.weights_proj, weights->nbytes), "cudaMalloc cached indexer weights proj");
        check_cuda(cudaMemcpy(c.wq_b, idx_shard.tensor_data(*wq_b), wq_b->nbytes, cudaMemcpyHostToDevice), "copy cached indexer wq_b");
        check_cuda(cudaMemcpy(c.wq_b_scale, idx_shard.tensor_data(*wq_b_scale), wq_b_scale->nbytes, cudaMemcpyHostToDevice), "copy cached indexer wq_b scale");
        check_cuda(cudaMemcpy(c.weights_proj, idx_shard.tensor_data(*weights), weights->nbytes, cudaMemcpyHostToDevice), "copy cached indexer weights proj");
        auto inserted = indexer_cache.emplace(key, c);
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
        arena.slot_by_local.clear();
        arena.slot_lru.clear();
        arena.slot_lru_pos.clear();
        arena.slot_local.clear();
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

    static bool key_is_sparse(const std::string& key) {
        return key.size() >= 2 && key[key.size() - 2] == ':' && key[key.size() - 1] == 's';
    }

    void touch_active_arena(const std::string& key) {
        auto it = active_arena_lru_pos.find(key);
        if (it != active_arena_lru_pos.end()) active_arena_lru.erase(it->second);
        active_arena_lru.push_back(key);
        active_arena_lru_pos[key] = std::prev(active_arena_lru.end());
        // Sparse arenas (per-layer persistent cache) get a separate cap.
        // Default sparse cap = active_arena_max_layers; dense (prefill)
        // arenas keep the historic small cap (3 for tp>1) to avoid blowing
        // the GPU memory budget on full-capacity prefill arenas.
        const bool incoming_sparse = key_is_sparse(key);
        const int dense_cap = active_arena_cache_limit_dense();
        const int sparse_cap = active_arena_max_layers;
        // Count sparse and dense separately.
        int sparse_n = 0, dense_n = 0;
        for (const std::string& k : active_arena_lru) (key_is_sparse(k) ? sparse_n : dense_n)++;
        // Evict from the front according to the matching cap.
        auto evict_front_if = [&](bool want_sparse, int& count, int cap) {
            while (cap > 0 && count > cap) {
                bool evicted = false;
                for (auto lr = active_arena_lru.begin(); lr != active_arena_lru.end(); ++lr) {
                    if (key_is_sparse(*lr) == want_sparse) {
                        const std::string evict = *lr;
                        if (evict == key) continue;
                        active_arena_lru.erase(lr);
                        active_arena_lru_pos.erase(evict);
                        auto victim = active_arena_cache.find(evict);
                        if (victim != active_arena_cache.end()) release_active_arena_device(victim->second);
                        --count;
                        evicted = true;
                        break;
                    }
                }
                if (!evicted) break;
            }
        };
        evict_front_if(true, sparse_n, sparse_cap);
        evict_front_if(false, dense_n, dense_cap);
        (void)incoming_sparse;
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

    int active_arena_cache_limit_dense() const {
        const int dense_max_layers = env_int_or_default("DSV4_CPP_DENSE_ARENA_MAX_LAYERS", 0);
        if (dense_max_layers > 0) return dense_max_layers;
        return options.tp_world > 1 ? 3 : 1;
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

    // Release all active arenas whose key has a given suffix (":d" = dense,
    // ":s" = sparse). Used to free prefill's dense arenas before decode
    // starts when decode uses a sparse arena, since the two have different
    // capacities and cannot share a buffer via the freelist.
    void release_active_arenas_with_suffix(const std::string& suffix) {
        std::vector<std::string> victims;
        for (auto& kv : active_arena_cache) {
            const std::string& key = kv.first;
            if (key.size() >= suffix.size() && key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
                victims.push_back(key);
            }
        }
        for (const std::string& key : victims) {
            auto it = active_arena_lru_pos.find(key);
            if (it != active_arena_lru_pos.end()) {
                active_arena_lru.erase(it->second);
                active_arena_lru_pos.erase(it);
            }
            auto cache_it = active_arena_cache.find(key);
            if (cache_it != active_arena_cache.end()) {
                release_active_arena_device(cache_it->second);
            }
        }
        // Free the now-stranded buffers so cudaMalloc can succeed below.
        for (auto& blk : active_arena_device_freelist) {
            if (blk.w1) cudaFree(blk.w1);
            if (blk.s1) cudaFree(blk.s1);
            if (blk.w2) cudaFree(blk.w2);
            if (blk.s2) cudaFree(blk.s2);
            if (blk.w3) cudaFree(blk.w3);
            if (blk.s3) cudaFree(blk.s3);
        }
        active_arena_device_freelist.clear();
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
            (void)hc_device_cache(li);
            if (use_gpu_compressor != 0 && static_cast<size_t>(li) < config.compress_ratios.size() && config.compress_ratios[static_cast<size_t>(li)] > 0) {
                const std::string prefix = "layers." + std::to_string(li) + ".";
                if (index.shard_for_tensor(prefix + "attn.compressor.wkv.weight") != nullptr) (void)compressor_device_cache(li);
            }
            if (static_cast<size_t>(li) < config.compress_ratios.size() && config.compress_ratios[static_cast<size_t>(li)] == 4) {
                const std::string prefix = "layers." + std::to_string(li) + ".";
                if (index.shard_for_tensor(prefix + "attn.indexer.wq_b.weight") != nullptr) {
                    (void)indexer_device_cache(li);
                    if (use_gpu_compressor != 0) (void)indexer_compressor_device_cache(li);
                }
            }
        }
    }

    DeviceFp4ActiveArena& active_fp4_arena(int layer_id, int tp_world, int tp_rank, int capacity, const DeviceFp4ExpertCache& sample, bool sparse = false) {
        const std::string key = std::to_string(layer_id) + ":" + std::to_string(tp_world) + ":" + std::to_string(tp_rank) + ":" + std::to_string(capacity) + (sparse ? ":s" : ":d");
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
                    if (arena.sparse_slots) {
                        arena.slot_by_local.clear();
                        arena.slot_lru.clear();
                        arena.slot_lru_pos.assign(static_cast<size_t>(arena.capacity), arena.slot_lru.end());
                        arena.slot_local.assign(static_cast<size_t>(arena.capacity), -1);
                    }
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
        arena.sparse_slots = sparse;
        if (sparse) {
            arena.slot_lru_pos.assign(static_cast<size_t>(capacity), arena.slot_lru.end());
            arena.slot_local.assign(static_cast<size_t>(capacity), -1);
        }
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

    // Acquire a slot index inside a sparse arena for an expert with local_id.
    // Returns the slot index in [0, arena.capacity). If the expert is already
    // staged, returns its slot. Otherwise: if there's an empty slot use it,
    // else evict the LRU slot. Caller must follow up with H2D writes when
    // already_staged is false.
    int acquire_sparse_slot(DeviceFp4ActiveArena& arena, int local_id, bool& already_staged) {
        auto hit = arena.slot_by_local.find(local_id);
        if (hit != arena.slot_by_local.end()) {
            int slot = hit->second;
            // Touch LRU.
            arena.slot_lru.erase(arena.slot_lru_pos[static_cast<size_t>(slot)]);
            arena.slot_lru.push_back(slot);
            arena.slot_lru_pos[static_cast<size_t>(slot)] = std::prev(arena.slot_lru.end());
            already_staged = true;
            return slot;
        }
        int slot;
        if (static_cast<int>(arena.slot_lru.size()) < arena.capacity) {
            slot = static_cast<int>(arena.slot_lru.size());
            for (int s = 0; s < arena.capacity; ++s) {
                if (arena.slot_local[static_cast<size_t>(s)] == -1) {
                    slot = s;
                    break;
                }
            }
        } else {
            slot = arena.slot_lru.front();
            arena.slot_lru.pop_front();
            int victim_local = arena.slot_local[static_cast<size_t>(slot)];
            if (victim_local >= 0) arena.slot_by_local.erase(victim_local);
        }
        arena.slot_local[static_cast<size_t>(slot)] = local_id;
        arena.slot_by_local[local_id] = slot;
        arena.slot_lru.push_back(slot);
        arena.slot_lru_pos[static_cast<size_t>(slot)] = std::prev(arena.slot_lru.end());
        already_staged = false;
        return slot;
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
        DeviceSharedCache c;
        check_cuda(cudaMalloc(&c.w1, w1->nbytes), "cudaMalloc cached shared w1");
        check_cuda(cudaMalloc(&c.s1, s1->nbytes), "cudaMalloc cached shared s1");
        check_cuda(cudaMalloc(&c.w2, w2->nbytes), "cudaMalloc cached shared w2");
        check_cuda(cudaMalloc(&c.s2, s2->nbytes), "cudaMalloc cached shared s2");
        check_cuda(cudaMalloc(&c.w3, w3->nbytes), "cudaMalloc cached shared w3");
        check_cuda(cudaMalloc(&c.s3, s3->nbytes), "cudaMalloc cached shared s3");
        check_cuda(cudaMemcpy(c.w1, shared_shard.tensor_data(*w1), w1->nbytes, cudaMemcpyHostToDevice), "copy cached shared w1");
        check_cuda(cudaMemcpy(c.s1, shared_shard.tensor_data(*s1), s1->nbytes, cudaMemcpyHostToDevice), "copy cached shared s1");
        check_cuda(cudaMemcpy(c.w2, shared_shard.tensor_data(*w2), w2->nbytes, cudaMemcpyHostToDevice), "copy cached shared w2");
        check_cuda(cudaMemcpy(c.s2, shared_shard.tensor_data(*s2), s2->nbytes, cudaMemcpyHostToDevice), "copy cached shared s2");
        check_cuda(cudaMemcpy(c.w3, shared_shard.tensor_data(*w3), w3->nbytes, cudaMemcpyHostToDevice), "copy cached shared w3");
        check_cuda(cudaMemcpy(c.s3, shared_shard.tensor_data(*s3), s3->nbytes, cudaMemcpyHostToDevice), "copy cached shared s3");
        auto inserted = shared_cache.emplace(key, c);
        return inserted.first->second;
    }

    DeviceMoeDecodeWorkspace& moe_decode_workspace(int topk, int dim, int inter_dim) {
        const std::string key = std::to_string(topk) + ":" + std::to_string(dim) + ":" + std::to_string(inter_dim);
        auto it = moe_decode_workspace_cache.find(key);
        if (it != moe_decode_workspace_cache.end()) return it->second;
        DeviceMoeDecodeWorkspace w;
        w.fp4.topk = topk;
        w.fp4.dim = dim;
        w.fp4.inter_dim = inter_dim;
        check_cuda(cudaMalloc(&w.fp4.d_x_q, static_cast<size_t>(dim)), "cudaMalloc moe decode x q");
        check_cuda(cudaMalloc(&w.fp4.d_x_scale, sizeof(float)), "cudaMalloc moe decode x scale");
        check_cuda(cudaMalloc(&w.fp4.d_gate, static_cast<size_t>(topk) * inter_dim * sizeof(float)), "cudaMalloc moe decode gate");
        check_cuda(cudaMalloc(&w.fp4.d_up, static_cast<size_t>(topk) * inter_dim * sizeof(float)), "cudaMalloc moe decode up");
        check_cuda(cudaMalloc(&w.fp4.d_hidden_q, static_cast<size_t>(topk) * inter_dim), "cudaMalloc moe decode hidden q");
        check_cuda(cudaMalloc(&w.fp4.d_hidden_scale, static_cast<size_t>(topk) * sizeof(float)), "cudaMalloc moe decode hidden scale");
        check_cuda(cudaMalloc(&w.fp4.d_route_y, static_cast<size_t>(topk) * dim * sizeof(float)), "cudaMalloc moe decode route y");
        auto inserted = moe_decode_workspace_cache.emplace(key, w);
        return inserted.first->second;
    }

    DeviceHcCache& hc_device_cache(int layer_id) {
        auto it = hc_cache.find(layer_id);
        if (it != hc_cache.end()) return it->second;
        const std::string prefix = "layers." + std::to_string(layer_id) + ".";
        SafeTensorsShard& shard = shard_for_tensor(prefix + "hc_attn_fn");
        const auto* attn_fn = require_tensor(shard, prefix + "hc_attn_fn");
        const auto* attn_scale = require_tensor(shard, prefix + "hc_attn_scale");
        const auto* attn_base = require_tensor(shard, prefix + "hc_attn_base");
        const auto* ffn_fn = require_tensor(shard, prefix + "hc_ffn_fn");
        const auto* ffn_scale = require_tensor(shard, prefix + "hc_ffn_scale");
        const auto* ffn_base = require_tensor(shard, prefix + "hc_ffn_base");
        DeviceHcCache c;
        check_cuda(cudaMalloc(&c.attn_fn, attn_fn->nbytes), "cudaMalloc hc attn fn");
        check_cuda(cudaMalloc(&c.attn_scale, attn_scale->nbytes), "cudaMalloc hc attn scale");
        check_cuda(cudaMalloc(&c.attn_base, attn_base->nbytes), "cudaMalloc hc attn base");
        check_cuda(cudaMalloc(&c.ffn_fn, ffn_fn->nbytes), "cudaMalloc hc ffn fn");
        check_cuda(cudaMalloc(&c.ffn_scale, ffn_scale->nbytes), "cudaMalloc hc ffn scale");
        check_cuda(cudaMalloc(&c.ffn_base, ffn_base->nbytes), "cudaMalloc hc ffn base");
        check_cuda(cudaMemcpy(c.attn_fn, shard.tensor_data(*attn_fn), attn_fn->nbytes, cudaMemcpyHostToDevice), "copy hc attn fn");
        check_cuda(cudaMemcpy(c.attn_scale, shard.tensor_data(*attn_scale), attn_scale->nbytes, cudaMemcpyHostToDevice), "copy hc attn scale");
        check_cuda(cudaMemcpy(c.attn_base, shard.tensor_data(*attn_base), attn_base->nbytes, cudaMemcpyHostToDevice), "copy hc attn base");
        check_cuda(cudaMemcpy(c.ffn_fn, shard.tensor_data(*ffn_fn), ffn_fn->nbytes, cudaMemcpyHostToDevice), "copy hc ffn fn");
        check_cuda(cudaMemcpy(c.ffn_scale, shard.tensor_data(*ffn_scale), ffn_scale->nbytes, cudaMemcpyHostToDevice), "copy hc ffn scale");
        check_cuda(cudaMemcpy(c.ffn_base, shard.tensor_data(*ffn_base), ffn_base->nbytes, cudaMemcpyHostToDevice), "copy hc ffn base");
        auto inserted = hc_cache.emplace(layer_id, c);
        return inserted.first->second;
    }

    DeviceCompressorState& compressor_state_for_layer(int layer_id, int slots, int cols) {
        auto it = compressor_device_state.find(layer_id);
        if (it != compressor_device_state.end()) return it->second;
        DeviceCompressorState state;
        state.slots = slots;
        state.cols = cols;
        check_cuda(cudaMalloc(&state.kv, static_cast<size_t>(slots) * cols * sizeof(float)), "cudaMalloc compressor kv state");
        check_cuda(cudaMalloc(&state.score, static_cast<size_t>(slots) * cols * sizeof(float)), "cudaMalloc compressor score state");
        check_cuda(cudaMemset(state.kv, 0, static_cast<size_t>(slots) * cols * sizeof(float)), "zero compressor kv state");
        std::vector<float> init(static_cast<size_t>(slots) * cols, -INFINITY);
        check_cuda(cudaMemcpy(state.score, init.data(), init.size() * sizeof(float), cudaMemcpyHostToDevice), "init compressor score state");
        auto inserted = compressor_device_state.emplace(layer_id, state);
        return inserted.first->second;
    }

    DeviceCompressorState& indexer_compressor_state_for_layer(int layer_id, int slots, int cols) {
        auto it = indexer_compressor_device_state.find(layer_id);
        if (it != indexer_compressor_device_state.end()) return it->second;
        DeviceCompressorState state;
        state.slots = slots;
        state.cols = cols;
        check_cuda(cudaMalloc(&state.kv, static_cast<size_t>(slots) * cols * sizeof(float)), "cudaMalloc indexer compressor kv state");
        check_cuda(cudaMalloc(&state.score, static_cast<size_t>(slots) * cols * sizeof(float)), "cudaMalloc indexer compressor score state");
        check_cuda(cudaMemset(state.kv, 0, static_cast<size_t>(slots) * cols * sizeof(float)), "zero indexer compressor kv state");
        std::vector<float> init(static_cast<size_t>(slots) * cols, -INFINITY);
        check_cuda(cudaMemcpy(state.score, init.data(), init.size() * sizeof(float), cudaMemcpyHostToDevice), "init indexer compressor score state");
        auto inserted = indexer_compressor_device_state.emplace(layer_id, state);
        return inserted.first->second;
    }

    int kv_cache_capacity_for_layer(int layer_id) const {
        const int window = static_cast<int>(config.window_size == 0 ? 128 : config.window_size);
        uint64_t ratio = 0;
        if (layer_id >= 0 && static_cast<size_t>(layer_id) < config.compress_ratios.size()) ratio = config.compress_ratios[static_cast<size_t>(layer_id)];
        const int compressed = ratio == 0 ? 0 : (kv_cache_tokens + static_cast<int>(ratio) - 1) / static_cast<int>(ratio);
        return window + compressed;
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

    const float* rope_inv_freqs_for(int layer_id, bool use_compress, int rope_dim, float theta) {
        auto& store = use_compress ? rope_inv_freqs_compress : rope_inv_freqs_plain;
        auto it = store.find(layer_id);
        if (it != store.end()) return it->second;
        const int n = rope_dim / 2;
        std::vector<float> host(n);
        for (int i = 0; i < n; ++i) {
            host[i] = std::pow(theta, -2.0 * i / static_cast<double>(rope_dim));
        }
        if (use_compress) {
            // YaRN: when original_seq_len > 0, blend freqs with smooth ramp.
            const double original_seq_len = static_cast<double>(config.original_context_length == 0 ? 65536 : config.original_context_length);
            const double factor = config.rope_factor > 0.0 ? config.rope_factor : 16.0;
            const double beta_fast = config.beta_fast > 0.0 ? config.beta_fast : 32.0;
            const double beta_slow = config.beta_slow > 0.0 ? config.beta_slow : 1.0;
            if (original_seq_len > 0.0 && factor > 0.0) {
                auto correction_dim = [&](double num_rotations) {
                    return rope_dim * std::log(original_seq_len / (num_rotations * 2.0 * M_PI)) / (2.0 * std::log(theta));
                };
                double low_d = std::floor(correction_dim(beta_fast));
                double high_d = std::ceil(correction_dim(beta_slow));
                double low = std::max(low_d, 0.0);
                double high = std::min(high_d, static_cast<double>(rope_dim - 1));
                if (low == high) high += 0.001;
                for (int i = 0; i < n; ++i) {
                    double pair_dim = static_cast<double>(2 * i);
                    double t = std::clamp((pair_dim - low) / (high - low), 0.0, 1.0);
                    double smooth = 1.0 - t;
                    double base = host[i];
                    host[i] = static_cast<float>(base / factor * (1.0 - smooth) + base * smooth);
                }
            }
        }
        float* ptr = nullptr;
        check_cuda(cudaMalloc(&ptr, static_cast<size_t>(n) * sizeof(float)), "cudaMalloc rope inv freqs");
        check_cuda(cudaMemcpy(ptr, host.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice), "copy rope inv freqs");
        store[layer_id] = ptr;
        return ptr;
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
    // Rank-local logits captured by the most recent forward pass. Used by
    // PersistentEngine to drive stochastic sampling without re-running the
    // head matmul.
    std::vector<float> last_local_logits;
    int last_head_rows = 0;
    int last_local_head_start = 0;
    std::unordered_map<std::string, std::unique_ptr<SafeTensorsShard>> shard_cache;
    std::unordered_map<std::string, DeviceAttentionCache> attention_cache;
    std::unordered_map<std::string, DeviceSharedCache> shared_cache;
    std::unordered_map<std::string, DeviceFp4ExpertCache> expert_cache;
    std::unordered_map<std::string, DeviceCompressorCache> compressor_cache;
    std::unordered_map<int, DeviceHcCache> hc_cache;
    std::unordered_map<std::string, DeviceMoeDecodeWorkspace> moe_decode_workspace_cache;
    std::unordered_map<std::string, DeviceCompressorCache> indexer_compressor_cache;
    std::unordered_map<int, DeviceCompressorState> compressor_device_state;
    std::unordered_map<int, DeviceCompressorState> indexer_compressor_device_state;
    std::unordered_map<std::string, DeviceIndexerCache> indexer_cache;
    std::unordered_map<std::string, DeviceFp4ActiveArena> active_arena_cache;
    std::unordered_map<std::string, HostFp4ExpertSlot> host_fp4_slot_cache;
    std::list<std::string> active_arena_lru;
    std::vector<ActiveArenaDeviceBuffers> active_arena_device_freelist;
    std::unordered_map<std::string, std::list<std::string>::iterator> active_arena_lru_pos;
    int active_arena_max_layers = env_int_or_default("DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS", 0);
    int use_gpu_compressor = env_int_or_default("DSV4_CPP_GPU_COMPRESSOR", 1);
    std::unordered_map<std::string, DeviceGateCache> gate_cache;
    std::unordered_map<int, float*> kv_cache;
    std::unordered_map<int, float*> indexer_kv_cache;
    std::unordered_map<int, float*> rope_inv_freqs_compress;
    std::unordered_map<int, float*> rope_inv_freqs_plain;
};

}  // namespace

ForwardSmokeResult run_safetensors_token_forward_impl(SafeForwardContext& ctx, int token, int layer_count, int position);
ForwardSmokeResult run_safetensors_prompt_prefill_impl(SafeForwardContext& ctx, const std::vector<int>& tokens, int layer_count);

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
    ctx.kv_cache_tokens = static_cast<int>(tokens.size());
    return run_safetensors_prompt_prefill_impl(ctx, tokens, layer_count);
}

std::vector<ForwardSmokeResult> run_safetensors_generate_tokens(const std::string& ckpt_dir, const std::vector<int>& seed_tokens, int layer_count, int max_new_tokens) {
    return run_safetensors_generate_tokens_with_options(ckpt_dir, seed_tokens, layer_count, max_new_tokens, ForwardSmokeOptions{});
}

GenerateSmokeResult run_safetensors_generate_tokens_timed_with_options(const std::string& ckpt_dir, const std::vector<int>& seed_tokens, int layer_count, int max_new_tokens, const ForwardSmokeOptions& options) {
    if (seed_tokens.empty()) throw std::runtime_error("generation seed has no tokens");
    if (max_new_tokens <= 0) return {};
    SafeForwardContext ctx(ckpt_dir);
    ctx.options = options;
    ctx.kv_cache_tokens = static_cast<int>(seed_tokens.size() + static_cast<size_t>(max_new_tokens));
    const int dim = static_cast<int>(ctx.embed->shape[1]);
    ctx.prepare_resident_device_caches(layer_count, options.tp_world, options.tp_rank, dim);
    const bool prepare_fp4_host = !options.skip_fp4_host_prepare && env_int_or_default("DSV4_CPP_PREPARE_FP4_HOST", 1) != 0;
    if (prepare_fp4_host) ctx.prepare_fp4_host_weights(layer_count, options.tp_world, options.tp_rank);
    const int warmup_prefill_passes = env_int_or_default("DSV4_CPP_PREFILL_WARMUP_PASSES", 0);
    for (int pass = 0; pass < warmup_prefill_passes; ++pass) {
        (void)run_safetensors_prompt_prefill_impl(ctx, seed_tokens, layer_count);
    }
    const auto timed_t0 = Clock::now();
    const auto prefill_t0 = timed_t0;
    ForwardSmokeResult result = run_safetensors_prompt_prefill_impl(ctx, seed_tokens, layer_count);
    int token = result.top_token;
#ifdef DSV4_HAVE_NCCL
    if (options.tp_world > 1 && !options.nccl_id_path.empty()) {
        TpTopResult global = nccl_global_top1(options.tp_world, options.tp_rank, options.device, options.nccl_id_path.c_str(), result.top_token, result.top_logit);
        token = global.token;
        result.top_token = global.token;
        result.top_logit = global.logit;
    }
#endif
    const auto prefill_t1 = Clock::now();
    // When decode uses a sparse per-layer arena (matching PyTorch's lazy
    // per-layer cache), prefill's dense (full n_local_experts) arenas would
    // double the per-layer footprint and OOM at MAX_CACHED_LAYERS=43. Drop
    // them here so decode can keep all 43 sparse arenas resident.
    if (env_int_or_default("DSV4_CPP_DECODE_SPARSE_ARENA", 0) > 0) {
        ctx.release_active_arenas_with_suffix(":d");
    }
    std::vector<ForwardSmokeResult> out;
    out.reserve(static_cast<size_t>(max_new_tokens));
    ForwardSmokeResult generated = result;
    generated.token = token;
    out.push_back(generated);
    int position = static_cast<int>(seed_tokens.size());
    const auto decode_t0 = Clock::now();
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
    const auto decode_t1 = Clock::now();
    GenerateSmokeResult timed;
    timed.tokens = std::move(out);
    timed.wall_seconds = elapsed_ms(timed_t0, decode_t1) / 1000.0;
    timed.prefill_seconds = elapsed_ms(prefill_t0, prefill_t1) / 1000.0;
    timed.decode_seconds = max_new_tokens > 1 ? elapsed_ms(decode_t0, decode_t1) / 1000.0 : 0.0;
    timed.prompt_tokens = static_cast<int>(seed_tokens.size());
    timed.decode_tokens = max_new_tokens > 1 ? max_new_tokens - 1 : 0;
    return timed;
}

std::vector<ForwardSmokeResult> run_safetensors_generate_tokens_with_options(const std::string& ckpt_dir, const std::vector<int>& seed_tokens, int layer_count, int max_new_tokens, const ForwardSmokeOptions& options) {
    return run_safetensors_generate_tokens_timed_with_options(ckpt_dir, seed_tokens, layer_count, max_new_tokens, options).tokens;
}

ForwardSmokeResult run_safetensors_prompt_prefill_impl(SafeForwardContext& ctx, const std::vector<int>& tokens, int layer_count) {
    if (!cuda_runtime_available()) throw std::runtime_error("CUDA runtime is not available");
    if (tokens.empty()) throw std::runtime_error("prompt has no tokens");
    SafeTensorsIndex& index = ctx.index;
    ModelConfig& config = ctx.config;
    if (layer_count <= 0) layer_count = 1;
    if (config.n_layers > 0) layer_count = std::min(layer_count, static_cast<int>(config.n_layers));
    const int token_count = static_cast<int>(tokens.size());
    const int last_token = tokens.back();
    const int tp_world = std::max(1, ctx.options.tp_world);
    const int tp_rank = std::max(0, ctx.options.tp_rank);
    if (tp_rank >= tp_world) throw std::runtime_error("invalid TP rank in prefill options");

    const auto* embed = ctx.embed;
    const auto* head = ctx.head;
    Fp4View first_w1 = ctx.fp4_view("layers.0.ffn.experts.0.w1.weight");
    const int dim = static_cast<int>(embed->shape[1]);
    const int inter = static_cast<int>(first_w1.pair.rows);
    const int route_count = static_cast<int>(std::min<uint64_t>(config.n_activated_experts, config.n_routed_experts));
    const int experts_per_rank = tp_world > 1 ? static_cast<int>(config.n_routed_experts / tp_world) : static_cast<int>(config.n_routed_experts);
    const int expert_start = tp_rank * experts_per_rank;

    // Optional: pre-stage all layers' experts to GPU before the timed prefill,
    // so the layer loop hits the staged_local cache. WARNING: the regular
    // layer-by-layer staging is already overlapped with compute via the stage
    // stream; pre-staging serializes H2D before compute and *slows* prefill on
    // 2080 Ti PCIe. Keep this gated and OFF by default. Useful only when GPU
    // can hold every layer's active arena and PCIe is plentiful.
    if (env_int_or_default("DSV4_CPP_PREFILL_PRESTAGE_EXPERTS", 0) != 0) {
        for (int li = 0; li < layer_count; ++li) {
            const std::string prefix = "layers." + std::to_string(li) + ".";
            DeviceFp4ExpertCache sample;
            Fp4View sample_w1 = ctx.fp4_view(prefix + "ffn.experts." + std::to_string(expert_start) + ".w1.weight");
            Fp4View sample_w2 = ctx.fp4_view(prefix + "ffn.experts." + std::to_string(expert_start) + ".w2.weight");
            Fp4View sample_w3 = ctx.fp4_view(prefix + "ffn.experts." + std::to_string(expert_start) + ".w3.weight");
            sample.w1_bytes = sample_w1.w->nbytes;
            sample.s1_bytes = sample_w1.s->nbytes;
            sample.w2_bytes = sample_w2.w->nbytes;
            sample.s2_bytes = sample_w2.s->nbytes;
            sample.w3_bytes = sample_w3.w->nbytes;
            sample.s3_bytes = sample_w3.s->nbytes;
            DeviceFp4ActiveArena& arena = ctx.active_fp4_arena(li, tp_world, tp_rank, experts_per_rank, sample);
            for (int local = 0; local < experts_per_rank; ++local) {
                if (!arena.staged_local.insert(local).second) continue;
                Fp4View w1 = ctx.fp4_view(prefix + "ffn.experts." + std::to_string(expert_start + local) + ".w1.weight");
                Fp4View w2 = ctx.fp4_view(prefix + "ffn.experts." + std::to_string(expert_start + local) + ".w2.weight");
                Fp4View w3 = ctx.fp4_view(prefix + "ffn.experts." + std::to_string(expert_start + local) + ".w3.weight");
                HostFp4ExpertSlot& slot = ctx.host_fp4_slot(li, expert_start + local, w1, w2, w3);
                check_cuda(cudaMemcpyAsync(arena.w1 + static_cast<size_t>(local) * arena.w1_bytes, slot.h_w1q, arena.w1_bytes, cudaMemcpyHostToDevice), "prestage w1");
                check_cuda(cudaMemcpyAsync(arena.s1 + static_cast<size_t>(local) * arena.s1_bytes, slot.h_w1s, arena.s1_bytes, cudaMemcpyHostToDevice), "prestage s1");
                check_cuda(cudaMemcpyAsync(arena.w2 + static_cast<size_t>(local) * arena.w2_bytes, slot.h_w2q, arena.w2_bytes, cudaMemcpyHostToDevice), "prestage w2");
                check_cuda(cudaMemcpyAsync(arena.s2 + static_cast<size_t>(local) * arena.s2_bytes, slot.h_w2s, arena.s2_bytes, cudaMemcpyHostToDevice), "prestage s2");
                check_cuda(cudaMemcpyAsync(arena.w3 + static_cast<size_t>(local) * arena.w3_bytes, slot.h_w3q, arena.w3_bytes, cudaMemcpyHostToDevice), "prestage w3");
                check_cuda(cudaMemcpyAsync(arena.s3 + static_cast<size_t>(local) * arena.s3_bytes, slot.h_w3s, arena.s3_bytes, cudaMemcpyHostToDevice), "prestage s3");
            }
        }
        check_cuda(cudaDeviceSynchronize(), "prestage sync");
    }
    const int head_rows = static_cast<int>(head->shape[0]);
    if (head_rows % tp_world != 0) throw std::runtime_error("head vocab rows must divide TP world");
    const int local_head_rows = head_rows / tp_world;
    const int local_head_start = tp_rank * local_head_rows;

#ifdef DSV4_HAVE_NCCL
    BF16AllReduceScratch bf16_reduce_scratch;
#endif
    DeviceMoePrefillWorkspace prefill_moe_workspace;

    const bool prefill_moe_prefetch_enabled = env_int_or_default("DSV4_CPP_PREFILL_MOE_PREFETCH", 0) != 0;
    const bool prefill_moe_copy_stream_enabled = prefill_moe_prefetch_enabled || env_int_or_default("DSV4_CPP_PREFILL_MOE_COPY_STREAM", 0) != 0;
    cudaStream_t prefill_moe_copy_stream = nullptr;
    cudaEvent_t prefill_moe_stage_event = nullptr;
    if (prefill_moe_copy_stream_enabled) {
        check_cuda(cudaStreamCreateWithFlags(&prefill_moe_copy_stream, cudaStreamNonBlocking), "create prefill moe copy stream");
        check_cuda(cudaEventCreateWithFlags(&prefill_moe_stage_event, cudaEventDisableTiming), "create prefill moe stage event");
    }
    const bool prefill_shared_overlap_enabled = env_int_or_default("DSV4_CPP_PREFILL_SHARED_OVERLAP", 1) != 0;
    cudaStream_t prefill_shared_stream = nullptr;
    cudaEvent_t prefill_shared_ready_event = nullptr;
    cudaEvent_t prefill_shared_done_event = nullptr;
    if (prefill_shared_overlap_enabled) {
        check_cuda(cudaStreamCreateWithFlags(&prefill_shared_stream, cudaStreamNonBlocking), "create prefill shared stream");
        check_cuda(cudaEventCreateWithFlags(&prefill_shared_ready_event, cudaEventDisableTiming), "create prefill shared ready event");
        check_cuda(cudaEventCreateWithFlags(&prefill_shared_done_event, cudaEventDisableTiming), "create prefill shared done event");
    }
    std::vector<int> prev_layer_active_locals;

    auto stage_experts_for_layer = [&](int layer_idx, const std::vector<int>& locals, cudaStream_t stream) -> int {
        if (locals.empty()) return 0;
        const std::string layer_prefix = "layers." + std::to_string(layer_idx) + ".";
        DeviceFp4ExpertCache sample;
        Fp4View sample_w1 = ctx.fp4_view(layer_prefix + "ffn.experts." + std::to_string(expert_start) + ".w1.weight");
        Fp4View sample_w2 = ctx.fp4_view(layer_prefix + "ffn.experts." + std::to_string(expert_start) + ".w2.weight");
        Fp4View sample_w3 = ctx.fp4_view(layer_prefix + "ffn.experts." + std::to_string(expert_start) + ".w3.weight");
        sample.w1_bytes = sample_w1.w->nbytes;
        sample.s1_bytes = sample_w1.s->nbytes;
        sample.w2_bytes = sample_w2.w->nbytes;
        sample.s2_bytes = sample_w2.s->nbytes;
        sample.w3_bytes = sample_w3.w->nbytes;
        sample.s3_bytes = sample_w3.s->nbytes;
        DeviceFp4ActiveArena& arena_l = ctx.active_fp4_arena(layer_idx, tp_world, tp_rank, experts_per_rank, sample);
        int staged = 0;
        for (int local : locals) {
            if (local < 0 || local >= experts_per_rank) continue;
            if (!arena_l.staged_local.insert(local).second) continue;
            Fp4View w1 = ctx.fp4_view(layer_prefix + "ffn.experts." + std::to_string(expert_start + local) + ".w1.weight");
            Fp4View w2 = ctx.fp4_view(layer_prefix + "ffn.experts." + std::to_string(expert_start + local) + ".w2.weight");
            Fp4View w3 = ctx.fp4_view(layer_prefix + "ffn.experts." + std::to_string(expert_start + local) + ".w3.weight");
            HostFp4ExpertSlot& slot = ctx.host_fp4_slot(layer_idx, expert_start + local, w1, w2, w3);
            check_cuda(cudaMemcpyAsync(arena_l.w1 + static_cast<size_t>(local) * arena_l.w1_bytes, slot.h_w1q, arena_l.w1_bytes, cudaMemcpyHostToDevice, stream), "prefetch w1");
            check_cuda(cudaMemcpyAsync(arena_l.s1 + static_cast<size_t>(local) * arena_l.s1_bytes, slot.h_w1s, arena_l.s1_bytes, cudaMemcpyHostToDevice, stream), "prefetch s1");
            check_cuda(cudaMemcpyAsync(arena_l.w2 + static_cast<size_t>(local) * arena_l.w2_bytes, slot.h_w2q, arena_l.w2_bytes, cudaMemcpyHostToDevice, stream), "prefetch w2");
            check_cuda(cudaMemcpyAsync(arena_l.s2 + static_cast<size_t>(local) * arena_l.s2_bytes, slot.h_w2s, arena_l.s2_bytes, cudaMemcpyHostToDevice, stream), "prefetch s2");
            check_cuda(cudaMemcpyAsync(arena_l.w3 + static_cast<size_t>(local) * arena_l.w3_bytes, slot.h_w3q, arena_l.w3_bytes, cudaMemcpyHostToDevice, stream), "prefetch w3");
            check_cuda(cudaMemcpyAsync(arena_l.s3 + static_cast<size_t>(local) * arena_l.s3_bytes, slot.h_w3s, arena_l.s3_bytes, cudaMemcpyHostToDevice, stream), "prefetch s3");
            ++staged;
        }
        return staged;
    };

    std::vector<int> token_ids(tokens.begin(), tokens.end());
    for (int token : token_ids) {
        if (token < 0 || token >= static_cast<int>(embed->shape[0])) throw std::runtime_error("token id out of range");
    }

    int* d_token_ids = nullptr;
    uint16_t* d_embed_matrix = nullptr;
    float* d_x_rows = nullptr;
    float* d_h4_rows = nullptr;
    float* d_h4_next_rows = nullptr;
    uint16_t* d_h4_bf16_rows = nullptr;
    float* d_hc_post_rows = nullptr;
    float* d_hc_comb_rows = nullptr;
    float* d_attn_out_rows = nullptr;
    uint16_t* d_ffn_gamma = nullptr;
    float* d_ffn_norm_rows = nullptr;
    int64_t* d_route_indices = nullptr;
    float* d_route_weights = nullptr;
    int64_t* d_group_route_tokens = nullptr;
    float* d_group_route_weights = nullptr;
    int32_t* d_seg_starts = nullptr;
    int32_t* d_counts = nullptr;
    int32_t* d_offsets = nullptr;
    int32_t* d_total_routes = nullptr;
    float* d_attn_x = nullptr;
    float* d_attn_norm = nullptr;
    float* d_attn_norm_rows = nullptr;
    float* d_q_a = nullptr;
    float* d_q_norm = nullptr;
    float* d_q = nullptr;
    float* d_q_a_rows = nullptr;
    float* d_q_norm_rows = nullptr;
    float* d_q_rows = nullptr;
    float* d_kv_a = nullptr;
    float* d_kv_norm = nullptr;
    float* d_kv_a_rows = nullptr;
    float* d_kv_norm_rows = nullptr;
    float* d_attn_value = nullptr;
    float* d_attn_value_rows = nullptr;
    float* d_attn_mid = nullptr;
    float* d_attn_mid_rows = nullptr;
    float* d_attn_out = nullptr;
    int32_t* d_prefill_window_indices = nullptr;
    float* d_moe_rows = nullptr;
    float* d_shared_gate = nullptr;
    float* d_shared_up = nullptr;
    float* d_shared_hidden = nullptr;
    float* d_shared_out = nullptr;
    uint16_t* d_head = nullptr;
    uint16_t* d_final_norm_gamma = nullptr;
    float* d_last_x = nullptr;
    float* d_final_norm = nullptr;
    float* d_logits = nullptr;

    // Chunked prefill: scratch buffers (everything except the residual stream
    // d_h4_rows, the per-layer KV memory d_kv_norm_rows, and the window-indices
    // table) are sized to one chunk's worth of rows. The layer loop iterates
    // over chunks inside each layer so the KV memory accumulates correctly.
    int prefill_chunk_size = env_int_or_default("DSV4_CPP_PREFILL_CHUNK_TOKENS", 4096);
    if (prefill_chunk_size <= 0 || prefill_chunk_size > token_count) prefill_chunk_size = token_count;
    const int chunk_alloc = prefill_chunk_size;
    const size_t token_dim = static_cast<size_t>(token_count) * dim;
    const size_t chunk_dim = static_cast<size_t>(chunk_alloc) * dim;
    const size_t routes_cap_chunk = static_cast<size_t>(chunk_alloc) * route_count;
    check_cuda(cudaMalloc(&d_token_ids, static_cast<size_t>(token_count) * sizeof(int)), "cudaMalloc token ids");
    check_cuda(cudaMalloc(&d_embed_matrix, ctx.embed->nbytes), "cudaMalloc embed matrix");
    check_cuda(cudaMalloc(&d_x_rows, chunk_dim * sizeof(float)), "cudaMalloc x rows");
    check_cuda(cudaMalloc(&d_h4_rows, token_dim * 4 * sizeof(float)), "cudaMalloc prefill hc h4 rows");
    check_cuda(cudaMalloc(&d_h4_next_rows, chunk_dim * 4 * sizeof(float)), "cudaMalloc prefill hc h4 next rows");
    check_cuda(cudaMalloc(&d_h4_bf16_rows, chunk_dim * 4 * sizeof(uint16_t)), "cudaMalloc prefill hc h4 bf16 rows");
    check_cuda(cudaMalloc(&d_hc_post_rows, static_cast<size_t>(chunk_alloc) * 4 * sizeof(float)), "cudaMalloc prefill hc post rows");
    check_cuda(cudaMalloc(&d_hc_comb_rows, static_cast<size_t>(chunk_alloc) * 16 * sizeof(float)), "cudaMalloc prefill hc comb rows");
    check_cuda(cudaMalloc(&d_attn_out_rows, chunk_dim * sizeof(float)), "cudaMalloc prefill attn out rows");
    check_cuda(cudaMalloc(&d_ffn_gamma, static_cast<size_t>(dim) * sizeof(uint16_t)), "cudaMalloc ffn gamma");
    check_cuda(cudaMalloc(&d_ffn_norm_rows, chunk_dim * sizeof(float)), "cudaMalloc ffn norm rows");
    check_cuda(cudaMalloc(&d_route_indices, routes_cap_chunk * sizeof(int64_t)), "cudaMalloc prefill route indices");
    check_cuda(cudaMalloc(&d_route_weights, routes_cap_chunk * sizeof(float)), "cudaMalloc prefill route weights");
    check_cuda(cudaMalloc(&d_group_route_tokens, routes_cap_chunk * sizeof(int64_t)), "cudaMalloc grouped route tokens");
    check_cuda(cudaMalloc(&d_group_route_weights, routes_cap_chunk * sizeof(float)), "cudaMalloc grouped route weights");
    check_cuda(cudaMalloc(&d_seg_starts, static_cast<size_t>(experts_per_rank + 1) * sizeof(int32_t)), "cudaMalloc seg starts");
    check_cuda(cudaMalloc(&d_counts, static_cast<size_t>(experts_per_rank) * sizeof(int32_t)), "cudaMalloc route counts");
    check_cuda(cudaMalloc(&d_offsets, static_cast<size_t>(experts_per_rank) * sizeof(int32_t)), "cudaMalloc route offsets");
    check_cuda(cudaMalloc(&d_total_routes, sizeof(int32_t)), "cudaMalloc total routes");
    AttentionSmokeDims attn_dims = make_attention_dims(config, dim, tp_world, 0);
    check_cuda(cudaMalloc(&d_attn_x, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc prefill attn x");
    check_cuda(cudaMalloc(&d_attn_norm, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc prefill attn norm");
    check_cuda(cudaMalloc(&d_attn_norm_rows, chunk_dim * sizeof(float)), "cudaMalloc prefill attn norm rows");
    check_cuda(cudaMalloc(&d_q_a, static_cast<size_t>(attn_dims.q_a_dim) * sizeof(float)), "cudaMalloc prefill q_a");
    check_cuda(cudaMalloc(&d_q_norm, static_cast<size_t>(attn_dims.q_a_dim) * sizeof(float)), "cudaMalloc prefill q_norm");
    check_cuda(cudaMalloc(&d_q, static_cast<size_t>(attn_dims.q_dim) * sizeof(float)), "cudaMalloc prefill q");
    check_cuda(cudaMalloc(&d_q_a_rows, static_cast<size_t>(chunk_alloc) * attn_dims.q_a_dim * sizeof(float)), "cudaMalloc prefill q_a rows");
    check_cuda(cudaMalloc(&d_q_norm_rows, static_cast<size_t>(chunk_alloc) * attn_dims.q_a_dim * sizeof(float)), "cudaMalloc prefill q_norm rows");
    check_cuda(cudaMalloc(&d_q_rows, static_cast<size_t>(chunk_alloc) * attn_dims.q_dim * sizeof(float)), "cudaMalloc prefill q rows");
    check_cuda(cudaMalloc(&d_kv_a, static_cast<size_t>(attn_dims.kv_dim) * sizeof(float)), "cudaMalloc prefill kv_a");
    check_cuda(cudaMalloc(&d_kv_norm, static_cast<size_t>(attn_dims.kv_dim) * sizeof(float)), "cudaMalloc prefill kv_norm");
    check_cuda(cudaMalloc(&d_kv_a_rows, static_cast<size_t>(chunk_alloc) * attn_dims.kv_dim * sizeof(float)), "cudaMalloc prefill kv_a rows");
    // d_kv_norm_rows stays full-N: chunked sparse attention reads KV for absolute
    // positions 0..ce of the current layer, so each chunk's KV slice accumulates
    // into the same per-layer buffer (reused across layers).
    check_cuda(cudaMalloc(&d_kv_norm_rows, static_cast<size_t>(token_count) * attn_dims.kv_dim * sizeof(float)), "cudaMalloc prefill kv_norm rows");
    check_cuda(cudaMalloc(&d_attn_value, static_cast<size_t>(attn_dims.q_dim) * sizeof(float)), "cudaMalloc prefill attn value");
    check_cuda(cudaMalloc(&d_attn_value_rows, static_cast<size_t>(chunk_alloc) * attn_dims.q_dim * sizeof(float)), "cudaMalloc prefill attn value rows");
    check_cuda(cudaMalloc(&d_attn_mid, static_cast<size_t>(attn_dims.attn_mid) * sizeof(float)), "cudaMalloc prefill attn mid");
    check_cuda(cudaMalloc(&d_attn_mid_rows, static_cast<size_t>(chunk_alloc) * attn_dims.attn_mid * sizeof(float)), "cudaMalloc prefill attn mid rows");
    check_cuda(cudaMalloc(&d_attn_out, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc prefill attn out");
    int window_topk_global = 0;
    {
        const int window_topk = static_cast<int>(std::min<uint64_t>(static_cast<uint64_t>(token_count), std::max<uint64_t>(1, config.window_size == 0 ? 128 : config.window_size)));
        window_topk_global = window_topk;
        // Window indices depend only on (token_count, window_size) — precompute once
        // and reuse across all layers; for chunked queries we slice by cs * topk.
        check_cuda(cudaMalloc(&d_prefill_window_indices, static_cast<size_t>(token_count) * window_topk * sizeof(int32_t)), "cudaMalloc prefill window indices");
    }
    check_cuda(cudaMalloc(&d_moe_rows, chunk_dim * sizeof(float)), "cudaMalloc moe rows");
    check_cuda(cudaMalloc(&d_shared_gate, static_cast<size_t>(chunk_alloc) * inter * sizeof(float)), "cudaMalloc shared gate rows");
    check_cuda(cudaMalloc(&d_shared_up, static_cast<size_t>(chunk_alloc) * inter * sizeof(float)), "cudaMalloc shared up rows");
    check_cuda(cudaMalloc(&d_shared_hidden, static_cast<size_t>(chunk_alloc) * inter * sizeof(float)), "cudaMalloc shared hidden rows");
    check_cuda(cudaMalloc(&d_shared_out, chunk_dim * sizeof(float)), "cudaMalloc shared out rows");
    check_cuda(cudaMalloc(&d_head, static_cast<size_t>(local_head_rows) * dim * sizeof(uint16_t)), "cudaMalloc head");
    check_cuda(cudaMalloc(&d_final_norm_gamma, static_cast<size_t>(dim) * sizeof(uint16_t)), "cudaMalloc final norm gamma");
    check_cuda(cudaMalloc(&d_last_x, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc last x");
    check_cuda(cudaMalloc(&d_final_norm, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc final norm");
    check_cuda(cudaMalloc(&d_logits, static_cast<size_t>(local_head_rows) * sizeof(float)), "cudaMalloc logits");

    check_cuda(cudaMemcpy(d_token_ids, token_ids.data(), static_cast<size_t>(token_count) * sizeof(int), cudaMemcpyHostToDevice), "copy token ids");
    check_cuda(cudaMemcpy(d_embed_matrix, ctx.embed_shard.tensor_data(*embed), ctx.embed->nbytes, cudaMemcpyHostToDevice), "copy embed matrix");
    check_cuda(cudaMemcpy(d_head, reinterpret_cast<const uint16_t*>(ctx.head_shard.tensor_data(*head)) + static_cast<size_t>(local_head_start) * dim, static_cast<size_t>(local_head_rows) * dim * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy head");
    check_cuda(cudaMemcpy(d_final_norm_gamma, ctx.final_norm_shard.tensor_data(*ctx.final_norm), ctx.final_norm->nbytes, cudaMemcpyHostToDevice), "copy final norm gamma");
    // Initial embed + hc_repeat: process the prompt chunk-by-chunk so d_x_rows
    // can stay chunk-sized while d_h4_rows accumulates full-N.
    for (int cs = 0; cs < token_count; cs += prefill_chunk_size) {
        const int ce = std::min(token_count, cs + prefill_chunk_size);
        const int cn = ce - cs;
        if (!bf16_rows_to_float_cuda(d_embed_matrix, d_token_ids + cs, d_x_rows, cn, dim)) throw std::runtime_error("embed rows launch failed");
        if (!hc_repeat_rows_cuda(d_x_rows, d_h4_rows + static_cast<size_t>(cs) * 4 * dim, cn, dim)) throw std::runtime_error("prefill hc repeat rows launch failed");
    }
    // Window indices are independent of layer; precompute once and slice per chunk.
    if (!build_prefill_window_indices_cuda(d_prefill_window_indices, token_count, attn_dims.window_size, window_topk_global)) throw std::runtime_error("prefill window indices launch failed");
    const bool profile_forward = profile_forward_enabled();
    double total_prefill_hc_pre_ms = 0.0;
    double total_prefill_attn_ms = 0.0;
    double total_prefill_attn_post_ms = 0.0;
    double total_prefill_ffn_pre_ms = 0.0;
    double total_prefill_gate_ms = 0.0;
    double total_prefill_group_ms = 0.0;
    double total_prefill_moe_ms = 0.0;
    double total_prefill_reduce_ms = 0.0;
    double total_prefill_shared_ms = 0.0;
    double total_prefill_ffn_post_ms = 0.0;
    auto sync_prefill_profile = [&](const char* what) {
        if (profile_forward) check_cuda(cudaDeviceSynchronize(), what);
    };

    for (int li = 0; li < layer_count; ++li) {
        const std::string prefix = "layers." + std::to_string(li) + ".";
        SafeTensorsShard& attn_norm_shard = ctx.shard_for_tensor(prefix + "attn_norm.weight");
        SafeTensorsShard& qkv_shard = ctx.shard_for_tensor(prefix + "attn.wq_a.weight");
        SafeTensorsShard& ffn_norm_shard = ctx.shard_for_tensor(prefix + "ffn_norm.weight");
        const auto* attn_norm = require_tensor(attn_norm_shard, prefix + "attn_norm.weight");
        const auto* q_norm = require_tensor(qkv_shard, prefix + "attn.q_norm.weight");
        const auto* kv_norm = require_tensor(qkv_shard, prefix + "attn.kv_norm.weight");
        const auto* ffn_norm = require_tensor(ffn_norm_shard, prefix + "ffn_norm.weight");
        DeviceAttentionCache& attn_cache = ctx.attention_device_cache(li, tp_world, tp_rank, attn_dims);
        DeviceHcCache& hc_cache = ctx.hc_device_cache(li);
        const uint16_t* d_attn_gamma_ptr = attn_cache.attn_norm;
        const uint16_t* d_q_gamma_ptr = attn_cache.q_norm;
        const uint16_t* d_kv_gamma_ptr = attn_cache.kv_norm;
        uint64_t layer_compress_ratio = static_cast<size_t>(li) < ctx.config.compress_ratios.size() ? ctx.config.compress_ratios[static_cast<size_t>(li)] : 0;
        attn_dims.rope_theta = static_cast<float>(layer_compress_ratio == 0 ? config.rope_theta : config.compress_rope_theta);
        if (attn_dims.rope_theta <= 0.0f) throw std::runtime_error("invalid layer rope_theta");
        attn_dims.d_inv_freqs = ctx.rope_inv_freqs_for(li, layer_compress_ratio != 0, attn_dims.rope_dim, attn_dims.rope_theta);
        float* d_layer_kv_cache = ctx.kv_cache_tokens > 0 ? ctx.kv_cache_for_layer(li, attn_dims.head_dim) : nullptr;

        if (prefill_moe_prefetch_enabled && prefill_moe_copy_stream_enabled && li > 0 && !prev_layer_active_locals.empty()) {
            const int prefetched = stage_experts_for_layer(li, prev_layer_active_locals, prefill_moe_copy_stream);
            if (prefetched > 0) {
                check_cuda(cudaEventRecord(prefill_moe_stage_event, prefill_moe_copy_stream), "record prefill moe prefetch event");
                if (profile_forward && tp_rank == 0) {
                    std::cerr << "CPP_PREFILL_MOE_PREFETCH layer=" << li << " hinted=" << static_cast<int>(prev_layer_active_locals.size()) << " staged=" << prefetched << "\n";
                }
            }
        }

        for (int cs = 0; cs < token_count; cs += prefill_chunk_size) {
            const int ce = std::min(token_count, cs + prefill_chunk_size);
            const int cn = ce - cs;
            const size_t cn_dim_sz = static_cast<size_t>(cn) * dim;
            const size_t cn_routes_cap = static_cast<size_t>(cn) * route_count;
            float* h4_chunk = d_h4_rows + static_cast<size_t>(cs) * 4 * dim;
            float* kv_norm_chunk = d_kv_norm_rows + static_cast<size_t>(cs) * attn_dims.kv_dim;

            auto stage_t = Clock::now();
            if (!hc_pre_float_rows_cuda(h4_chunk, hc_cache.attn_fn, hc_cache.attn_scale, hc_cache.attn_base, d_x_rows, d_hc_post_rows, d_hc_comb_rows, cn, dim)) throw std::runtime_error("prefill hc attn pre rows launch failed");
            sync_prefill_profile("profile sync prefill hc attn pre");
            total_prefill_hc_pre_ms += elapsed_ms(stage_t, Clock::now());
            stage_t = Clock::now();
            if (env_int_or_default("DSV4_CPP_PREFILL_BATCHED_ATTN", 1) != 0) {
                const bool profile_attn = profile_forward && env_int_or_default("DSV4_CPP_PROFILE_ATTN", 0) != 0;
                auto attn_stage_sync = [&](const char* what) {
                    if (profile_attn) check_cuda(cudaDeviceSynchronize(), what);
                };
                auto attn_t = Clock::now();
                if (!rmsnorm_bf16_gamma_rows_cuda(d_x_rows, d_attn_gamma_ptr, d_attn_norm_rows, cn, dim, 1e-6f)) throw std::runtime_error("prefill attn norm rows launch failed");
                if (!fp8_e4m3_e8m0_matmul_cuda(d_attn_norm_rows, attn_cache.wq_a, attn_cache.wq_a_scale, d_q_a_rows, cn, attn_dims.q_a_dim, dim)) throw std::runtime_error("prefill wq_a rows launch failed");
                if (!rmsnorm_bf16_gamma_rows_cuda(d_q_a_rows, d_q_gamma_ptr, d_q_norm_rows, cn, attn_dims.q_a_dim, 1e-6f)) throw std::runtime_error("prefill q norm rows launch failed");
                if (!fp8_e4m3_e8m0_matmul_cuda(d_q_norm_rows, attn_cache.wq_b, attn_cache.wq_b_scale, d_q_rows, cn, attn_dims.q_dim, attn_dims.q_a_dim)) throw std::runtime_error("prefill wq_b rows launch failed");
                if (!head_rmsnorm_rope_freqs_rows_cuda(d_q_rows, attn_dims.d_inv_freqs, cn, attn_dims.heads, attn_dims.head_dim, attn_dims.rope_dim, cs, false, 1e-6f)) throw std::runtime_error("prefill q rope rows launch failed");
                attn_stage_sync("attn q");
                double attn_q_ms = elapsed_ms(attn_t, Clock::now());
                attn_t = Clock::now();
                if (!fp8_e4m3_e8m0_matmul_cuda(d_attn_norm_rows, attn_cache.wkv, attn_cache.wkv_scale, d_kv_a_rows, cn, attn_dims.kv_dim, dim)) throw std::runtime_error("prefill wkv rows launch failed");
                if (!rmsnorm_bf16_gamma_rows_cuda(d_kv_a_rows, d_kv_gamma_ptr, kv_norm_chunk, cn, attn_dims.kv_dim, 1e-6f)) throw std::runtime_error("prefill kv norm rows launch failed");
                if (!head_rmsnorm_rope_freqs_rows_cuda(kv_norm_chunk, attn_dims.d_inv_freqs, cn, 1, attn_dims.head_dim, attn_dims.rope_dim, cs, false, 0.0f)) throw std::runtime_error("prefill kv rope rows launch failed");
                if (!fp8_act_quant_dequant_rows_strided_cuda(kv_norm_chunk, cn, attn_dims.head_dim - attn_dims.rope_dim, attn_dims.head_dim, 64)) throw std::runtime_error("prefill kv act quant rows failed");
                if (d_layer_kv_cache != nullptr && !copy_rows_to_kv_cache_cuda(kv_norm_chunk, d_layer_kv_cache, cn, attn_dims.head_dim, attn_dims.window_size, cs)) throw std::runtime_error("prefill kv cache rows copy failed");
                attn_stage_sync("attn kv");
                double attn_kv_ms = elapsed_ms(attn_t, Clock::now());
                attn_t = Clock::now();
                const int window_topk = window_topk_global;
                double attn_build_window_ms = 0.0;
                auto attn_sparse_t = Clock::now();
                if (!prefill_sparse_attention_headpair_cuda(d_q_rows, d_kv_norm_rows, attn_cache.attn_sink, d_prefill_window_indices + static_cast<size_t>(cs) * window_topk, d_attn_value_rows, cn, attn_dims.heads, ce, window_topk, attn_dims.head_dim, 1.0f / std::sqrt(static_cast<float>(attn_dims.head_dim)))) throw std::runtime_error("prefill sparse attention headpair launch failed");
                attn_stage_sync("attn sparse kernel");
                double attn_sparse_kernel_ms = profile_attn ? elapsed_ms(attn_sparse_t, Clock::now()) : 0.0;
                auto attn_inv_rope_t = Clock::now();
                if (!head_rmsnorm_rope_freqs_rows_cuda(d_attn_value_rows, attn_dims.d_inv_freqs, cn, attn_dims.heads, attn_dims.head_dim, attn_dims.rope_dim, cs, true, 0.0f)) throw std::runtime_error("prefill attn value inverse rope rows launch failed");
                attn_stage_sync("attn sparse");
                double attn_inv_rope_ms = profile_attn ? elapsed_ms(attn_inv_rope_t, Clock::now()) : 0.0;
                double attn_sparse_ms = elapsed_ms(attn_t, Clock::now());
                attn_t = Clock::now();
                for (int g = 0; g < attn_dims.groups; ++g) {
                    const float* group_x = d_attn_value_rows + static_cast<size_t>(g) * attn_dims.group_dim;
                    const uint8_t* group_w = attn_cache.wo_a + static_cast<size_t>(g) * attn_dims.group_rank * attn_dims.group_dim;
                    const uint8_t* group_s = attn_cache.wo_a_scale + static_cast<size_t>(g) * (attn_dims.group_rank / 128) * (attn_dims.group_dim / 128);
                    float* group_y = d_attn_mid_rows + static_cast<size_t>(g) * attn_dims.group_rank;
                    if (!fp8_e4m3_e8m0_matmul_strided_cuda(group_x, group_w, group_s, group_y, cn, attn_dims.group_rank, attn_dims.group_dim, attn_dims.q_dim, attn_dims.attn_mid)) throw std::runtime_error("prefill wo_a rows launch failed");
                }
                if (!fp8_e4m3_e8m0_matmul_cuda(d_attn_mid_rows, attn_cache.wo_b, attn_cache.wo_b_scale, d_attn_out_rows, cn, dim, attn_dims.attn_mid)) throw std::runtime_error("prefill wo_b rows launch failed");
                attn_stage_sync("attn wo");
                double attn_wo_ms = elapsed_ms(attn_t, Clock::now());
#ifdef DSV4_HAVE_NCCL
                attn_t = Clock::now();
                if (ctx.options.tp_world > 1) {
                    if (ctx.options.nccl_id_path.empty()) throw std::runtime_error("TP prefill attention all-reduce requires --nccl-id-path");
                    all_reduce_sum_fp32_via_bf16_inplace(ctx.options.tp_world, ctx.options.tp_rank, ctx.options.device, ctx.options.nccl_id_path.c_str(), d_attn_out_rows, static_cast<int>(cn_dim_sz), bf16_reduce_scratch);
                }
                attn_stage_sync("attn reduce");
                double attn_reduce_ms = elapsed_ms(attn_t, Clock::now());
#else
                double attn_reduce_ms = 0.0;
#endif
                if (profile_attn && tp_rank == 0) {
                    std::cerr << "CPP_PREFILL_ATTN_STAGE layer=" << li
                              << " chunk=[" << cs << "," << ce << ")"
                              << " q_ms=" << attn_q_ms
                              << " kv_ms=" << attn_kv_ms
                              << " sparse_ms=" << attn_sparse_ms
                              << " (window_ms=" << attn_build_window_ms
                              << " sparse_kernel_ms=" << attn_sparse_kernel_ms
                              << " inv_rope_ms=" << attn_inv_rope_ms
                              << ")"
                              << " wo_ms=" << attn_wo_ms
                              << " reduce_ms=" << attn_reduce_ms << "\n";
                }
            } else {
                for (int t = cs; t < ce; ++t) {
                    attn_dims.position = t;
                    attn_dims.cache_write_slot = t % attn_dims.window_size;
                    const int window_len = std::min(t + 1, attn_dims.window_size);
                    const int layer_cache_len = d_layer_kv_cache == nullptr ? 0 : std::min(ctx.kv_cache_capacity_for_layer(li), window_len);
                    check_cuda(cudaMemcpy(d_attn_x, d_x_rows + static_cast<size_t>(t - cs) * dim, static_cast<size_t>(dim) * sizeof(float), cudaMemcpyDeviceToDevice), "copy prefill attn token x");
                    if (!run_single_token_attention_smoke(
                            attn_dims,
                            d_attn_x,
                            d_attn_gamma_ptr,
                            attn_cache.wq_a,
                            attn_cache.wq_a_scale,
                            d_q_gamma_ptr,
                            attn_cache.wq_b,
                            attn_cache.wq_b_scale,
                            attn_cache.wkv,
                            attn_cache.wkv_scale,
                            d_kv_gamma_ptr,
                            attn_cache.wo_a,
                            attn_cache.wo_a_scale,
                            nullptr,
                            nullptr,
                            nullptr,
                            nullptr,
                            attn_cache.wo_b,
                            attn_cache.wo_b_scale,
                            attn_cache.attn_sink,
                            d_layer_kv_cache,
                            nullptr,
                            0,
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
                        throw std::runtime_error("prefill attention launch failed");
                    }
#ifdef DSV4_HAVE_NCCL
                    if (ctx.options.tp_world > 1) {
                        if (ctx.options.nccl_id_path.empty()) throw std::runtime_error("TP prefill attention all-reduce requires --nccl-id-path");
                        all_reduce_sum_fp32_via_bf16_inplace(ctx.options.tp_world, ctx.options.tp_rank, ctx.options.device, ctx.options.nccl_id_path.c_str(), d_attn_out, dim, bf16_reduce_scratch);
                    }
#endif
                    check_cuda(cudaMemcpy(d_attn_out_rows + static_cast<size_t>(t - cs) * dim, d_attn_out, static_cast<size_t>(dim) * sizeof(float), cudaMemcpyDeviceToDevice), "copy prefill attn out token");
                }
            }
            sync_prefill_profile("profile sync prefill attention");
            total_prefill_attn_ms += elapsed_ms(stage_t, Clock::now());
            stage_t = Clock::now();
            if (!hc_post_float_rows_cuda(d_attn_out_rows, h4_chunk, d_hc_post_rows, d_hc_comb_rows, d_h4_next_rows, cn, dim)) throw std::runtime_error("prefill hc attn post rows launch failed");
            if (!fp32_to_bf16_cuda(d_h4_next_rows, d_h4_bf16_rows, static_cast<int>(cn_dim_sz * 4))) throw std::runtime_error("prefill hc attn post bf16 round failed");
            if (!bf16_to_fp32_cuda(d_h4_bf16_rows, h4_chunk, static_cast<int>(cn_dim_sz * 4))) throw std::runtime_error("prefill hc attn post bf16 restore failed");
            sync_prefill_profile("profile sync prefill hc attn post");
            total_prefill_attn_post_ms += elapsed_ms(stage_t, Clock::now());
            stage_t = Clock::now();

            if (!hc_pre_float_rows_cuda(h4_chunk, hc_cache.ffn_fn, hc_cache.ffn_scale, hc_cache.ffn_base, d_x_rows, d_hc_post_rows, d_hc_comb_rows, cn, dim)) throw std::runtime_error("prefill hc ffn pre rows launch failed");
            if (!rmsnorm_bf16_gamma_rows_cuda(d_x_rows, attn_cache.ffn_norm, d_ffn_norm_rows, cn, dim, 1e-6f)) throw std::runtime_error("prefill ffn norm rows launch failed");
            sync_prefill_profile("profile sync prefill hc ffn pre");
            total_prefill_ffn_pre_ms += elapsed_ms(stage_t, Clock::now());
            stage_t = Clock::now();

            if (prefill_shared_overlap_enabled) {
                DeviceSharedCache& shared = ctx.shared_device_cache(li, tp_world, tp_rank, dim);
                const int shared_inter = inter;
                check_cuda(cudaEventRecord(prefill_shared_ready_event, nullptr), "record prefill shared ready event");
                check_cuda(cudaStreamWaitEvent(prefill_shared_stream, prefill_shared_ready_event, 0), "prefill shared stream wait ready");
                if (!fp8_e4m3_e8m0_matmul_cuda(d_ffn_norm_rows, shared.w1, shared.s1, d_shared_gate, cn, shared_inter, dim, prefill_shared_stream)) throw std::runtime_error("prefill shared w1 overlap launch failed");
                if (!fp8_e4m3_e8m0_matmul_cuda(d_ffn_norm_rows, shared.w3, shared.s3, d_shared_up, cn, shared_inter, dim, prefill_shared_stream)) throw std::runtime_error("prefill shared w3 overlap launch failed");
                if (!silu_mul_rows_cuda(d_shared_gate, d_shared_up, d_shared_hidden, cn, shared_inter, prefill_shared_stream)) throw std::runtime_error("prefill shared silu overlap launch failed");
                if (!fp8_e4m3_e8m0_matmul_cuda(d_shared_hidden, shared.w2, shared.s2, d_shared_out, cn, dim, shared_inter, prefill_shared_stream)) throw std::runtime_error("prefill shared w2 overlap launch failed");
                check_cuda(cudaEventRecord(prefill_shared_done_event, prefill_shared_stream), "record prefill shared done event");
            }

            DeviceGateCache& gate = ctx.gate_device_cache(li);
            const bool use_gpu_hash_gate = env_int_or_default("DSV4_CPP_PREFILL_GPU_HASH_GATE", 0) != 0;
            if (static_cast<uint64_t>(li) < config.n_hash_layers && gate.tid2eid != nullptr && use_gpu_hash_gate) {
                if (!gate_hash_bf16_rows_cuda(d_ffn_norm_rows, gate.weight, gate.tid2eid, d_token_ids + cs, d_route_indices, d_route_weights, cn, gate.dim, gate.hash_topk, route_count, static_cast<float>(config.route_scale))) throw std::runtime_error("prefill hash gate rows launch failed");
            } else if (static_cast<uint64_t>(li) < config.n_hash_layers && gate.tid2eid != nullptr) {
                std::vector<float> ffn_norm_host(cn_dim_sz);
                check_cuda(cudaMemcpy(ffn_norm_host.data(), d_ffn_norm_rows, cn_dim_sz * sizeof(float), cudaMemcpyDeviceToHost), "copy prefill ffn norm for host hash gate");
                SafeTensorsShard& gate_shard = ctx.shard_for_tensor(prefix + "ffn.gate.tid2eid");
                const auto* tid2eid = require_tensor(gate_shard, prefix + "ffn.gate.tid2eid");
                const auto* ids = reinterpret_cast<const int64_t*>(gate_shard.tensor_data(*tid2eid));
                SafeTensorsShard& gate_weight_shard = ctx.shard_for_tensor(prefix + "ffn.gate.weight");
                const auto* gate_weight = require_tensor(gate_weight_shard, prefix + "ffn.gate.weight");
                const auto* gate_w = reinterpret_cast<const uint16_t*>(gate_weight_shard.tensor_data(*gate_weight));
                const int gate_dim = static_cast<int>(gate_weight->shape[1]);
                std::vector<int64_t> h_indices(cn_routes_cap);
                std::vector<float> h_weights(cn_routes_cap);
                for (int t = 0; t < cn; ++t) {
                    float denom = 0.0f;
                    for (int k = 0; k < route_count; ++k) {
                        const int64_t e = ids[static_cast<size_t>(tokens[static_cast<size_t>(cs + t)]) * config.n_activated_experts + k];
                        h_indices[static_cast<size_t>(t) * route_count + k] = e;
                        float dot = 0.0f;
                        const float* norm_row = ffn_norm_host.data() + static_cast<size_t>(t) * dim;
                        for (int d = 0; d < gate_dim; ++d) dot += norm_row[d] * bf16_to_float(gate_w[static_cast<size_t>(e) * gate_dim + d]);
                        const float original = std::sqrt(std::log1pf(std::exp(dot)));
                        h_weights[static_cast<size_t>(t) * route_count + k] = original;
                        denom += original;
                    }
                    if (denom == 0.0f) denom = 1.0f;
                    for (int k = 0; k < route_count; ++k) h_weights[static_cast<size_t>(t) * route_count + k] = h_weights[static_cast<size_t>(t) * route_count + k] / denom * static_cast<float>(config.route_scale);
                }
                check_cuda(cudaMemcpy(d_route_indices, h_indices.data(), cn_routes_cap * sizeof(int64_t), cudaMemcpyHostToDevice), "copy hash route indices rows");
                check_cuda(cudaMemcpy(d_route_weights, h_weights.data(), cn_routes_cap * sizeof(float), cudaMemcpyHostToDevice), "copy hash route weights rows");
            } else {
                if (!gate_topk_bf16_rows_cuda(d_ffn_norm_rows, gate.weight, gate.bias, d_route_indices, d_route_weights, cn, gate.experts, gate.dim, route_count, static_cast<float>(config.route_scale))) throw std::runtime_error("prefill gate topk rows launch failed");
            }
            sync_prefill_profile("profile sync prefill gate");
            total_prefill_gate_ms += elapsed_ms(stage_t, Clock::now());
            stage_t = Clock::now();

            if (!moe_group_routes_cuda(d_route_indices, d_route_weights, d_group_route_tokens, d_group_route_weights, d_seg_starts, d_counts, d_offsets, d_total_routes, cn, route_count, expert_start, experts_per_rank)) throw std::runtime_error("prefill group routes launch failed");
            const bool profile_moe_host = profile_forward && env_int_or_default("DSV4_CPP_PROFILE_MOE_HOST", 0) != 0;
            auto moe_host_t0 = Clock::now();
            int32_t total_routes = 0;
            std::vector<int32_t> h_counts(experts_per_rank);
            check_cuda(cudaMemcpy(&total_routes, d_total_routes, sizeof(int32_t), cudaMemcpyDeviceToHost), "copy total routes");
            check_cuda(cudaMemcpy(h_counts.data(), d_counts, h_counts.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "copy route counts");
            const double moe_d2h_ms = profile_moe_host ? elapsed_ms(moe_host_t0, Clock::now()) : 0.0;
            int max_count = 0;
            int active_experts = 0;
            int64_t route_sum = 0;
            for (int32_t c : h_counts) {
                max_count = std::max(max_count, static_cast<int>(c));
                if (c > 0) ++active_experts;
                route_sum += c;
            }
            if (env_int_or_default("DSV4_CPP_PREFILL_ROUTE_STATS", 0) != 0 && tp_rank == 0) {
                std::cerr << "CPP_PREFILL_ROUTE_STATS layer=" << li
                          << " chunk=[" << cs << "," << ce << ")"
                          << " total_routes=" << total_routes
                          << " route_sum=" << route_sum
                          << " active_experts=" << active_experts
                          << " max_count=" << max_count
                          << " padded_rows=" << static_cast<int64_t>(experts_per_rank) * max_count << "\n";
            }
            sync_prefill_profile("profile sync prefill group");
            total_prefill_group_ms += elapsed_ms(stage_t, Clock::now());
            stage_t = Clock::now();
            if (total_routes > 0 && max_count > 0) {
                DeviceFp4ExpertCache sample;
                Fp4View sample_w1 = ctx.fp4_view(prefix + "ffn.experts." + std::to_string(expert_start) + ".w1.weight");
                Fp4View sample_w2 = ctx.fp4_view(prefix + "ffn.experts." + std::to_string(expert_start) + ".w2.weight");
                Fp4View sample_w3 = ctx.fp4_view(prefix + "ffn.experts." + std::to_string(expert_start) + ".w3.weight");
                sample.w1_bytes = sample_w1.w->nbytes;
                sample.s1_bytes = sample_w1.s->nbytes;
                sample.w2_bytes = sample_w2.w->nbytes;
                sample.s2_bytes = sample_w2.s->nbytes;
                sample.w3_bytes = sample_w3.w->nbytes;
                sample.s3_bytes = sample_w3.s->nbytes;
                DeviceFp4ActiveArena& arena = ctx.active_fp4_arena(li, tp_world, tp_rank, experts_per_rank, sample);
                int staged_this_layer = 0;
                const bool profile_stage = profile_forward && env_int_or_default("DSV4_CPP_PROFILE_MOE_STAGE", 0) != 0;
                cudaEvent_t stage_evt_begin = nullptr, stage_evt_end = nullptr;
                if (profile_stage) {
                    cudaEventCreate(&stage_evt_begin);
                    cudaEventCreate(&stage_evt_end);
                    cudaEventRecord(stage_evt_begin);
                }
                for (int local = 0; local < experts_per_rank; ++local) {
                    if (h_counts[static_cast<size_t>(local)] == 0) continue;
                    if (arena.staged_local.insert(local).second) {
                        Fp4View w1 = ctx.fp4_view(prefix + "ffn.experts." + std::to_string(expert_start + local) + ".w1.weight");
                        Fp4View w2 = ctx.fp4_view(prefix + "ffn.experts." + std::to_string(expert_start + local) + ".w2.weight");
                        Fp4View w3 = ctx.fp4_view(prefix + "ffn.experts." + std::to_string(expert_start + local) + ".w3.weight");
                        HostFp4ExpertSlot& slot = ctx.host_fp4_slot(li, expert_start + local, w1, w2, w3);
                        cudaStream_t stage_stream = prefill_moe_copy_stream_enabled ? prefill_moe_copy_stream : nullptr;
                        check_cuda(cudaMemcpyAsync(arena.w1 + static_cast<size_t>(local) * arena.w1_bytes, slot.h_w1q, arena.w1_bytes, cudaMemcpyHostToDevice, stage_stream), "stage prefill w1");
                        check_cuda(cudaMemcpyAsync(arena.s1 + static_cast<size_t>(local) * arena.s1_bytes, slot.h_w1s, arena.s1_bytes, cudaMemcpyHostToDevice, stage_stream), "stage prefill s1");
                        check_cuda(cudaMemcpyAsync(arena.w2 + static_cast<size_t>(local) * arena.w2_bytes, slot.h_w2q, arena.w2_bytes, cudaMemcpyHostToDevice, stage_stream), "stage prefill w2");
                        check_cuda(cudaMemcpyAsync(arena.s2 + static_cast<size_t>(local) * arena.s2_bytes, slot.h_w2s, arena.s2_bytes, cudaMemcpyHostToDevice, stage_stream), "stage prefill s2");
                        check_cuda(cudaMemcpyAsync(arena.w3 + static_cast<size_t>(local) * arena.w3_bytes, slot.h_w3q, arena.w3_bytes, cudaMemcpyHostToDevice, stage_stream), "stage prefill w3");
                        check_cuda(cudaMemcpyAsync(arena.s3 + static_cast<size_t>(local) * arena.s3_bytes, slot.h_w3s, arena.s3_bytes, cudaMemcpyHostToDevice, stage_stream), "stage prefill s3");
                        ++staged_this_layer;
                    }
                }
                if (profile_stage) {
                    cudaEventRecord(stage_evt_end);
                    cudaEventSynchronize(stage_evt_end);
                    float ms = 0.0f;
                    cudaEventElapsedTime(&ms, stage_evt_begin, stage_evt_end);
                    cudaEventDestroy(stage_evt_begin);
                    cudaEventDestroy(stage_evt_end);
                    if (tp_rank == 0 && staged_this_layer > 0) {
                        std::cerr << "CPP_PREFILL_MOE_STAGE_TIME layer=" << li
                                  << " experts=" << staged_this_layer
                                  << " stage_ms=" << ms << "\n";
                    }
                } else if (profile_forward && tp_rank == 0 && staged_this_layer > 0) {
                    std::cerr << "CPP_PREFILL_MOE_STAGED layer=" << li << " experts_staged=" << staged_this_layer << "\n";
                }
                // For prev-layer-active hint to the next layer's prefetch, only
                // populate from the LAST chunk of this layer. Earlier chunks may
                // touch experts not seen later; the hint is an optimization, not
                // correctness — keeping it scoped to the last chunk is fine.
                if (ce == token_count) {
                    prev_layer_active_locals.clear();
                    prev_layer_active_locals.reserve(active_experts);
                    for (int local = 0; local < experts_per_rank; ++local) {
                        if (h_counts[static_cast<size_t>(local)] > 0) prev_layer_active_locals.push_back(local);
                    }
                }
                if (prefill_moe_copy_stream_enabled && staged_this_layer > 0) {
                    check_cuda(cudaEventRecord(prefill_moe_stage_event, prefill_moe_copy_stream), "record prefill moe stage event");
                }
                const bool force_padded_moe = env_int_or_default("DSV4_CPP_MOE_FORCE_PADDED", 0) != 0;
                auto moe_build_t0 = profile_moe_host ? Clock::now() : moe_host_t0;
                std::vector<int32_t> h_tile_experts;
                std::vector<int32_t> h_tile_rows;
                if (!force_padded_moe) {
                    h_tile_experts.reserve(static_cast<size_t>((total_routes + 15) / 16 + experts_per_rank));
                    h_tile_rows.reserve(h_tile_experts.capacity());
                    for (int local = 0; local < experts_per_rank; ++local) {
                        const int count = h_counts[static_cast<size_t>(local)];
                        for (int row = 0; row < count; row += 16) {
                            h_tile_experts.push_back(local);
                            h_tile_rows.push_back(row);
                        }
                    }
                }
                const double moe_build_ms = profile_moe_host ? elapsed_ms(moe_build_t0, Clock::now()) : 0.0;
                const int padded_rows_cap = force_padded_moe ? experts_per_rank * max_count : total_routes;
                auto moe_h2d_t0 = profile_moe_host ? Clock::now() : moe_host_t0;
                prefill_moe_workspace.ensure(total_routes, static_cast<int>(h_tile_experts.size()), dim, inter, padded_rows_cap);
                prefill_moe_workspace.fp4.tile_count = force_padded_moe ? 0 : static_cast<int>(h_tile_experts.size());
                if (!force_padded_moe) {
                    check_cuda(cudaMemcpy(prefill_moe_workspace.fp4.d_tile_experts, h_tile_experts.data(), h_tile_experts.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy prefill moe tile experts");
                    check_cuda(cudaMemcpy(prefill_moe_workspace.fp4.d_tile_rows, h_tile_rows.data(), h_tile_rows.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy prefill moe tile rows");
                }
                const double moe_h2d_ms = profile_moe_host ? elapsed_ms(moe_h2d_t0, Clock::now()) : 0.0;
                if (prefill_moe_copy_stream_enabled && staged_this_layer > 0) {
                    check_cuda(cudaStreamWaitEvent(nullptr, prefill_moe_stage_event, 0), "wait prefill moe stage");
                }
                auto moe_kernel_t0 = profile_moe_host ? Clock::now() : moe_host_t0;
                if (!moe_prefill_fp4_grouped_cuda_with_workspace(d_ffn_norm_rows, d_group_route_tokens, d_group_route_weights, d_seg_starts, arena.w1, arena.s1, arena.w2, arena.s2, arena.w3, arena.s3, d_moe_rows, cn, route_count, total_routes, experts_per_rank, max_count, dim, inter, static_cast<float>(config.swiglu_limit), prefill_moe_workspace.fp4)) throw std::runtime_error("prefill grouped fp4 moe launch failed");
                if (profile_moe_host) {
                    check_cuda(cudaDeviceSynchronize(), "sync prefill moe kernel host profile");
                    const double moe_kernel_ms = elapsed_ms(moe_kernel_t0, Clock::now());
                    if (tp_rank == 0) {
                        std::cerr << "CPP_PREFILL_MOE_HOST layer=" << li
                                  << " d2h_ms=" << moe_d2h_ms
                                  << " build_ms=" << moe_build_ms
                                  << " h2d_ms=" << moe_h2d_ms
                                  << " kernel_ms=" << moe_kernel_ms
                                  << " tiles=" << prefill_moe_workspace.fp4.tile_count
                                  << " staged=" << staged_this_layer << "\n";
                    }
                }
            } else {
                check_cuda(cudaMemset(d_moe_rows, 0, cn_dim_sz * sizeof(float)), "zero empty prefill moe rows");
            }
            sync_prefill_profile("profile sync prefill moe");
            total_prefill_moe_ms += elapsed_ms(stage_t, Clock::now());
            stage_t = Clock::now();
#ifdef DSV4_HAVE_NCCL
            if (ctx.options.tp_world > 1) {
                if (ctx.options.nccl_id_path.empty()) throw std::runtime_error("TP prefill MoE all-reduce requires --nccl-id-path");
                all_reduce_sum_fp32_via_bf16_inplace(ctx.options.tp_world, ctx.options.tp_rank, ctx.options.device, ctx.options.nccl_id_path.c_str(), d_moe_rows, static_cast<int>(cn_dim_sz), bf16_reduce_scratch);
            }
#endif
            sync_prefill_profile("profile sync prefill reduce");
            total_prefill_reduce_ms += elapsed_ms(stage_t, Clock::now());
            stage_t = Clock::now();
            if (prefill_shared_overlap_enabled) {
                check_cuda(cudaStreamWaitEvent(nullptr, prefill_shared_done_event, 0), "default stream wait shared done");
                if (!vector_accum_rows_cuda(d_shared_out, d_moe_rows, cn, dim, 1.0f)) throw std::runtime_error("prefill shared accum overlap failed");
            } else {
                DeviceSharedCache& shared = ctx.shared_device_cache(li, tp_world, tp_rank, dim);
                const int shared_inter = inter;
                if (!fp8_e4m3_e8m0_matmul_cuda(d_ffn_norm_rows, shared.w1, shared.s1, d_shared_gate, cn, shared_inter, dim)) throw std::runtime_error("prefill shared w1 launch failed");
                if (!fp8_e4m3_e8m0_matmul_cuda(d_ffn_norm_rows, shared.w3, shared.s3, d_shared_up, cn, shared_inter, dim)) throw std::runtime_error("prefill shared w3 launch failed");
                if (!silu_mul_rows_cuda(d_shared_gate, d_shared_up, d_shared_hidden, cn, shared_inter)) throw std::runtime_error("prefill shared silu launch failed");
                if (!fp8_e4m3_e8m0_matmul_cuda(d_shared_hidden, shared.w2, shared.s2, d_shared_out, cn, dim, shared_inter)) throw std::runtime_error("prefill shared w2 launch failed");
                if (!vector_accum_rows_cuda(d_shared_out, d_moe_rows, cn, dim, 1.0f)) throw std::runtime_error("prefill shared accum failed");
            }
            sync_prefill_profile("profile sync prefill shared");
            total_prefill_shared_ms += elapsed_ms(stage_t, Clock::now());
            stage_t = Clock::now();
            if (!hc_post_float_rows_cuda(d_moe_rows, h4_chunk, d_hc_post_rows, d_hc_comb_rows, d_h4_next_rows, cn, dim)) throw std::runtime_error("prefill hc ffn post rows launch failed");
            if (!fp32_to_bf16_cuda(d_h4_next_rows, d_h4_bf16_rows, static_cast<int>(cn_dim_sz * 4))) throw std::runtime_error("prefill hc ffn post bf16 round failed");
            if (!bf16_to_fp32_cuda(d_h4_bf16_rows, h4_chunk, static_cast<int>(cn_dim_sz * 4))) throw std::runtime_error("prefill hc ffn post bf16 restore failed");
            sync_prefill_profile("profile sync prefill hc ffn post");
            total_prefill_ffn_post_ms += elapsed_ms(stage_t, Clock::now());
        }
    }

    if (profile_forward) {
        std::cout << "CPP_PREFILL_PROFILE_TOTAL hc_pre_ms=" << total_prefill_hc_pre_ms
                  << " attn_ms=" << total_prefill_attn_ms
                  << " attn_post_ms=" << total_prefill_attn_post_ms
                  << " ffn_pre_ms=" << total_prefill_ffn_pre_ms
                  << " gate_ms=" << total_prefill_gate_ms
                  << " group_ms=" << total_prefill_group_ms
                  << " moe_ms=" << total_prefill_moe_ms
                  << " reduce_ms=" << total_prefill_reduce_ms
                  << " shared_ms=" << total_prefill_shared_ms
                  << " ffn_post_ms=" << total_prefill_ffn_post_ms << "\n";
    }

    std::vector<float> last_h4(4 * static_cast<size_t>(dim));
    check_cuda(cudaMemcpy(last_h4.data(), d_h4_rows + static_cast<size_t>(token_count - 1) * 4 * dim, 4 * static_cast<size_t>(dim) * sizeof(float), cudaMemcpyDeviceToHost), "copy prefill last h4");
    std::vector<float> host_x = hc_head_cpu(last_h4, reinterpret_cast<const float*>(ctx.hc_head_shard.tensor_data(*ctx.hc_head_fn)), reinterpret_cast<const float*>(ctx.hc_head_shard.tensor_data(*ctx.hc_head_scale)), reinterpret_cast<const float*>(ctx.hc_head_shard.tensor_data(*ctx.hc_head_base)), dim);
    check_cuda(cudaMemcpy(d_last_x, host_x.data(), static_cast<size_t>(dim) * sizeof(float), cudaMemcpyHostToDevice), "copy prefill hc head");
    if (!rmsnorm_bf16_gamma_cuda(d_last_x, d_final_norm_gamma, d_final_norm, dim, 1e-6f)) throw std::runtime_error("prefill final norm launch failed");
    if (!bf16_matvec_cuda(d_final_norm, d_head, d_logits, local_head_rows, dim)) throw std::runtime_error("prefill head launch failed");
    check_cuda(cudaDeviceSynchronize(), "sync prefill kernels");
    std::vector<float> logits(local_head_rows);
    check_cuda(cudaMemcpy(logits.data(), d_logits, logits.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy prefill logits");
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
    ctx.last_local_logits = std::move(logits);
    ctx.last_head_rows = head_rows;
    ctx.last_local_head_start = local_head_start;

    cudaFree(d_token_ids); cudaFree(d_embed_matrix); cudaFree(d_x_rows); cudaFree(d_h4_rows); cudaFree(d_h4_next_rows); cudaFree(d_h4_bf16_rows); cudaFree(d_hc_post_rows); cudaFree(d_hc_comb_rows); cudaFree(d_attn_out_rows); cudaFree(d_ffn_gamma); cudaFree(d_ffn_norm_rows);
    cudaFree(d_route_indices); cudaFree(d_route_weights); cudaFree(d_group_route_tokens); cudaFree(d_group_route_weights); cudaFree(d_seg_starts); cudaFree(d_counts); cudaFree(d_offsets); cudaFree(d_total_routes);
    cudaFree(d_attn_x); cudaFree(d_attn_norm); cudaFree(d_attn_norm_rows); cudaFree(d_q_a); cudaFree(d_q_norm); cudaFree(d_q); cudaFree(d_q_a_rows); cudaFree(d_q_norm_rows); cudaFree(d_q_rows); cudaFree(d_kv_a); cudaFree(d_kv_norm); cudaFree(d_kv_a_rows); cudaFree(d_kv_norm_rows); cudaFree(d_attn_value); cudaFree(d_attn_value_rows); cudaFree(d_attn_mid); cudaFree(d_attn_mid_rows); cudaFree(d_attn_out); cudaFree(d_prefill_window_indices);
    cudaFree(d_moe_rows); cudaFree(d_shared_gate); cudaFree(d_shared_up); cudaFree(d_shared_hidden); cudaFree(d_shared_out);
    cudaFree(d_head); cudaFree(d_final_norm_gamma); cudaFree(d_last_x); cudaFree(d_final_norm); cudaFree(d_logits);
    if (prefill_moe_copy_stream_enabled) {
        cudaEventDestroy(prefill_moe_stage_event);
        cudaStreamDestroy(prefill_moe_copy_stream);
    }
    if (prefill_shared_overlap_enabled) {
        cudaEventDestroy(prefill_shared_ready_event);
        cudaEventDestroy(prefill_shared_done_event);
        cudaStreamDestroy(prefill_shared_stream);
    }
    return ForwardSmokeResult{last_token, dim, inter, head_rows, layer_count, top_token, top_logit, checksum};
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

#ifdef DSV4_HAVE_NCCL
    BF16AllReduceScratch bf16_reduce_scratch;
#endif

    uint16_t* d_embed = nullptr;
    uint8_t* d_w1 = nullptr;
    uint8_t* d_s1 = nullptr;
    uint8_t* d_w2 = nullptr;
    uint8_t* d_s2 = nullptr;
    uint8_t* d_w3 = nullptr;
    uint8_t* d_s3 = nullptr;
    uint16_t* d_head = nullptr;
    uint16_t* d_final_norm_gamma = nullptr;
    float* d_x = nullptr;
    float* d_h4 = nullptr;
    float* d_h4_next = nullptr;
    uint16_t* d_h4_bf16 = nullptr;
    float* d_hc_post = nullptr;
    float* d_hc_comb = nullptr;
    float* d_attn_norm = nullptr;
    float* d_q_a = nullptr;
    float* d_q_norm = nullptr;
    float* d_q = nullptr;
    float* d_kv_a = nullptr;
    float* d_kv_norm = nullptr;
    float* d_attn_value = nullptr;
    float* d_attn_mid = nullptr;
    int8_t* d_wo_a_x_q = nullptr;
    float* d_wo_a_x_scale = nullptr;
    float* d_attn_out = nullptr;
    uint16_t* d_compressor_input_bf16 = nullptr;
    float* d_compressor_input_rounded = nullptr;
    float* d_compressor_kv = nullptr;
    float* d_compressor_score = nullptr;
    float* d_indexer_comp_kv = nullptr;
    float* d_indexer_comp_score = nullptr;
    float* d_index_q = nullptr;
    float* d_indexer_kv = nullptr;
    uint16_t* d_index_weight_proj = nullptr;
    float* d_index_scores = nullptr;
    int* d_kv_indices = nullptr;
    float* d_resid1 = nullptr;
    float* d_ffn_norm = nullptr;
    float* d_gate = nullptr;
    float* d_up = nullptr;
    float* d_hidden = nullptr;
    float* d_moe = nullptr;
    float* d_resid2 = nullptr;
    float* d_shared_out = nullptr;
    int64_t* d_route_indices = nullptr;
    float* d_route_weights = nullptr;
    float* d_logits = nullptr;

    const auto* embed_data = reinterpret_cast<const uint16_t*>(ctx.embed_shard.tensor_data(*embed)) + static_cast<size_t>(token) * dim;
    check_cuda(cudaMalloc(&d_embed, static_cast<size_t>(dim) * sizeof(uint16_t)), "cudaMalloc embed");
    check_cuda(cudaMalloc(&d_w1, static_cast<size_t>(inter) * dim), "cudaMalloc w1");
    check_cuda(cudaMalloc(&d_s1, first_w1.s->nbytes), "cudaMalloc s1");
    check_cuda(cudaMalloc(&d_w2, static_cast<size_t>(dim) * inter), "cudaMalloc w2");
    check_cuda(cudaMalloc(&d_s2, first_w2.s->nbytes), "cudaMalloc s2");
    check_cuda(cudaMalloc(&d_w3, static_cast<size_t>(inter) * dim), "cudaMalloc w3");
    check_cuda(cudaMalloc(&d_s3, first_w3.s->nbytes), "cudaMalloc s3");
    check_cuda(cudaMalloc(&d_head, static_cast<size_t>(local_head_rows) * dim * sizeof(uint16_t)), "cudaMalloc head");
    check_cuda(cudaMalloc(&d_final_norm_gamma, static_cast<size_t>(dim) * sizeof(uint16_t)), "cudaMalloc final norm gamma");
    check_cuda(cudaMalloc(&d_x, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc x");
    check_cuda(cudaMalloc(&d_h4, static_cast<size_t>(4) * dim * sizeof(float)), "cudaMalloc hc h4");
    check_cuda(cudaMalloc(&d_h4_next, static_cast<size_t>(4) * dim * sizeof(float)), "cudaMalloc hc h4 next");
    check_cuda(cudaMalloc(&d_h4_bf16, static_cast<size_t>(4) * dim * sizeof(uint16_t)), "cudaMalloc hc h4 bf16");
    check_cuda(cudaMalloc(&d_hc_post, static_cast<size_t>(4) * sizeof(float)), "cudaMalloc hc post");
    check_cuda(cudaMalloc(&d_hc_comb, static_cast<size_t>(16) * sizeof(float)), "cudaMalloc hc comb");
    check_cuda(cudaMalloc(&d_attn_norm, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc attn norm");
    check_cuda(cudaMalloc(&d_q_a, static_cast<size_t>(attn_dims.q_a_dim) * sizeof(float)), "cudaMalloc q_a");
    check_cuda(cudaMalloc(&d_q_norm, static_cast<size_t>(attn_dims.q_a_dim) * sizeof(float)), "cudaMalloc q_norm");
    check_cuda(cudaMalloc(&d_q, static_cast<size_t>(attn_dims.q_dim) * sizeof(float)), "cudaMalloc q");
    check_cuda(cudaMalloc(&d_kv_a, static_cast<size_t>(attn_dims.kv_dim) * sizeof(float)), "cudaMalloc kv_a");
    check_cuda(cudaMalloc(&d_kv_norm, static_cast<size_t>(attn_dims.kv_dim) * sizeof(float)), "cudaMalloc kv_norm");
    check_cuda(cudaMalloc(&d_attn_value, static_cast<size_t>(attn_dims.q_dim) * sizeof(float)), "cudaMalloc attn value");
    check_cuda(cudaMalloc(&d_attn_mid, static_cast<size_t>(attn_dims.attn_mid) * sizeof(float)), "cudaMalloc attn mid");
    check_cuda(cudaMalloc(&d_wo_a_x_q, static_cast<size_t>(attn_dims.q_dim) * sizeof(int8_t)), "cudaMalloc wo_a x q");
    check_cuda(cudaMalloc(&d_wo_a_x_scale, static_cast<size_t>(attn_dims.groups) * sizeof(float)), "cudaMalloc wo_a x scale");
    check_cuda(cudaMalloc(&d_attn_out, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc attn out");
    if (ctx.use_gpu_compressor != 0) {
        check_cuda(cudaMalloc(&d_compressor_input_bf16, static_cast<size_t>(dim) * sizeof(uint16_t)), "cudaMalloc compressor input bf16");
        check_cuda(cudaMalloc(&d_compressor_input_rounded, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc compressor input rounded");
        check_cuda(cudaMalloc(&d_compressor_kv, static_cast<size_t>(1024) * sizeof(float)), "cudaMalloc compressor kv");
        check_cuda(cudaMalloc(&d_compressor_score, static_cast<size_t>(1024) * sizeof(float)), "cudaMalloc compressor score");
        check_cuda(cudaMalloc(&d_indexer_comp_kv, static_cast<size_t>(std::max<uint64_t>(1, config.index_head_dim * 2)) * sizeof(float)), "cudaMalloc indexer compressor kv");
        check_cuda(cudaMalloc(&d_indexer_comp_score, static_cast<size_t>(std::max<uint64_t>(1, config.index_head_dim * 2)) * sizeof(float)), "cudaMalloc indexer compressor score");
    }
    check_cuda(cudaMalloc(&d_index_q, static_cast<size_t>(std::max<uint64_t>(1, config.index_n_heads * config.index_head_dim)) * sizeof(float)), "cudaMalloc index q");
    check_cuda(cudaMalloc(&d_indexer_kv, static_cast<size_t>(std::max<uint64_t>(1, config.index_head_dim)) * sizeof(float)), "cudaMalloc indexer kv tmp");
    check_cuda(cudaMalloc(&d_index_weight_proj, static_cast<size_t>(std::max<uint64_t>(1, config.index_n_heads * config.dim)) * sizeof(uint16_t)), "cudaMalloc index weight proj");
    {
        const int max_compressed = std::max(1, (ctx.kv_cache_tokens + 3) / 4);
        const int max_keep = static_cast<int>(std::max<uint64_t>(1, config.index_topk));
        const int max_kv_indices = static_cast<int>(std::max<uint64_t>(1, config.window_size == 0 ? 128 : config.window_size)) + std::max(max_keep, max_compressed);
        const int max_index_heads = static_cast<int>(std::max<uint64_t>(1, config.index_n_heads));
        check_cuda(cudaMalloc(&d_index_scores, static_cast<size_t>(max_compressed + max_index_heads) * sizeof(float)), "cudaMalloc index scores");
        check_cuda(cudaMalloc(&d_kv_indices, static_cast<size_t>(max_kv_indices) * sizeof(int)), "cudaMalloc kv indices");
    }
    check_cuda(cudaMalloc(&d_resid1, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc resid1");
    check_cuda(cudaMalloc(&d_ffn_norm, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc ffn norm");
    check_cuda(cudaMalloc(&d_gate, static_cast<size_t>(inter) * sizeof(float)), "cudaMalloc gate");
    check_cuda(cudaMalloc(&d_up, static_cast<size_t>(inter) * sizeof(float)), "cudaMalloc up");
    check_cuda(cudaMalloc(&d_hidden, static_cast<size_t>(inter) * sizeof(float)), "cudaMalloc hidden");
    check_cuda(cudaMalloc(&d_moe, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc moe");
    check_cuda(cudaMalloc(&d_resid2, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc resid2");
    check_cuda(cudaMalloc(&d_shared_out, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc shared out");
    check_cuda(cudaMalloc(&d_route_indices, static_cast<size_t>(config.n_activated_experts) * sizeof(int64_t)), "cudaMalloc route indices");
    check_cuda(cudaMalloc(&d_route_weights, static_cast<size_t>(config.n_activated_experts) * sizeof(float)), "cudaMalloc route weights");
    check_cuda(cudaMalloc(&d_logits, static_cast<size_t>(local_head_rows) * sizeof(float)), "cudaMalloc logits");

    cudaStream_t moe_copy_stream = nullptr;
    cudaEvent_t moe_stage_event = nullptr;
    bool moe_stage_event_recorded = false;
    check_cuda(cudaStreamCreateWithFlags(&moe_copy_stream, cudaStreamNonBlocking), "create moe copy stream");
    check_cuda(cudaEventCreateWithFlags(&moe_stage_event, cudaEventDisableTiming), "create moe stage event");

    check_cuda(cudaMemcpy(d_embed, embed_data, static_cast<size_t>(dim) * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy embed");
    const auto* head_data = reinterpret_cast<const uint16_t*>(ctx.head_shard.tensor_data(*head)) + static_cast<size_t>(local_head_start) * dim;
    check_cuda(cudaMemcpy(d_head, head_data, static_cast<size_t>(local_head_rows) * dim * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy head");
    check_cuda(cudaMemcpy(d_final_norm_gamma, ctx.final_norm_shard.tensor_data(*ctx.final_norm), ctx.final_norm->nbytes, cudaMemcpyHostToDevice), "copy final norm gamma");
    if (!bf16_row_to_float_cuda(d_embed, d_x, 0, dim)) throw std::runtime_error("embed launch failed");
    const bool debug_forward = debug_forward_enabled();
    const bool profile_forward = profile_forward_enabled();
    const bool profile_decode_sync = profile_forward && env_int_or_default("DSV4_CPP_DECODE_PROFILE", 0) != 0;
    const bool profile_attn = profile_forward && env_int_or_default("DSV4_CPP_PROFILE_ATTN", 0) != 0;
    const bool profile_reduce_detail = profile_forward && env_int_or_default("DSV4_CPP_PROFILE_REDUCE_DETAIL", 0) != 0;
    const int sparse_slots_per_layer = env_int_or_default("DSV4_CPP_DECODE_SPARSE_ARENA", 0);
    const bool use_sparse_arena = sparse_slots_per_layer > 0;
    double total_route_gate_kernel_ms = 0.0;
    double total_route_d2h_ms = 0.0;
    double total_load_ms = 0.0;
    double total_hc_ms = 0.0;
    double total_attn_ms = 0.0;
    AttentionProfileBreakdown total_attn_profile;
    total_attn_profile.enabled = profile_attn;
    ReduceBreakdown total_attn_reduce_detail;
    total_attn_reduce_detail.enabled = profile_reduce_detail;
    ReduceBreakdown total_moe_reduce_detail;
    total_moe_reduce_detail.enabled = profile_reduce_detail;
    double total_route_ms = 0.0;
    double total_route_comp_ms = 0.0;
    double total_route_indexer_comp_ms = 0.0;
    double total_route_indexer_q_ms = 0.0;
    double total_route_indexer_topk_ms = 0.0;
    double total_route_gate_ms = 0.0;
    double total_moe_ms = 0.0;
    double total_moe_stage_ms = 0.0;
    double total_moe_kernel_ms = 0.0;
    double total_moe_reduce_ms = 0.0;
    double total_shared_ms = 0.0;
    double total_post_ms = 0.0;
    std::vector<float> h4(static_cast<size_t>(4) * dim);
    std::vector<float> host_x(dim);
    check_cuda(cudaMemcpy(host_x.data(), d_x, static_cast<size_t>(dim) * sizeof(float), cudaMemcpyDeviceToHost), "copy embed host");
    for (int m = 0; m < 4; ++m) std::copy(host_x.begin(), host_x.end(), h4.begin() + static_cast<size_t>(m) * dim);
    check_cuda(cudaMemcpy(d_h4, h4.data(), static_cast<size_t>(4) * dim * sizeof(float), cudaMemcpyHostToDevice), "copy initial hc h4");
    const int decode_window_len = std::min(position + 1, attn_dims.window_size);
    const int decode_window_start = std::max(0, position - decode_window_len + 1);
    if (decode_window_len > 0) {
        if (!build_decode_kv_indices_cuda(d_kv_indices, decode_window_start, decode_window_len, attn_dims.window_size, 0, attn_dims.window_size)) {
            throw std::runtime_error("build decode kv window indices launch failed");
        }
    }

    for (int li = 0; li < layer_count; ++li) {
        const auto layer_t0 = Clock::now();
        auto stage_t = layer_t0;
        double load_ms = 0.0;
        double hc_ms = 0.0;
        double attn_ms = 0.0;
        AttentionProfileBreakdown attn_profile;
        attn_profile.enabled = profile_attn;
        ReduceBreakdown attn_reduce_detail;
        attn_reduce_detail.enabled = profile_reduce_detail;
        ReduceBreakdown moe_reduce_detail;
        moe_reduce_detail.enabled = profile_reduce_detail;
        double route_ms = 0.0;
        double route_comp_ms = 0.0;
        double route_indexer_comp_ms = 0.0;
        double route_indexer_q_ms = 0.0;
        double route_indexer_topk_ms = 0.0;
        double route_gate_ms = 0.0;
        double moe_ms = 0.0;
        double moe_stage_ms = 0.0;
        double moe_kernel_ms = 0.0;
        double moe_reduce_ms = 0.0;
        double shared_ms = 0.0;
        double post_ms = 0.0;
        const std::string prefix = "layers." + std::to_string(li) + ".";
        attn_dims.layer_id = li;
        attn_dims.cache_write_slot = position % attn_dims.window_size;
        float* d_layer_kv_cache = ctx.kv_cache_tokens > 0 ? ctx.kv_cache_for_layer(li, attn_dims.head_dim) : nullptr;
        uint64_t layer_compress_ratio = static_cast<size_t>(li) < ctx.config.compress_ratios.size() ? ctx.config.compress_ratios[static_cast<size_t>(li)] : 0;
        attn_dims.rope_theta = static_cast<float>(layer_compress_ratio == 0 ? config.rope_theta : config.compress_rope_theta);
        if (attn_dims.rope_theta <= 0.0f) throw std::runtime_error("invalid layer rope_theta");
        attn_dims.d_inv_freqs = ctx.rope_inv_freqs_for(li, layer_compress_ratio != 0, attn_dims.rope_dim, attn_dims.rope_theta);
        const int compressed_ready = layer_compress_ratio == 0 ? 0 : (position + 1) / static_cast<int>(layer_compress_ratio);
        const int window_len = decode_window_len;
        const int layer_cache_len = d_layer_kv_cache == nullptr ? 0 : std::min(ctx.kv_cache_capacity_for_layer(li), window_len + compressed_ready);
        int kv_index_count = 0;
        SafeTensorsShard& qkv_shard = ctx.shard_for_tensor(prefix + "attn.wq_a.weight");
        SafeTensorsShard& wo_a_shard = ctx.shard_for_tensor(prefix + "attn.wo_a.weight");
        SafeTensorsShard& wo_b_shard = ctx.shard_for_tensor(prefix + "attn.wo_b.weight");
        DeviceAttentionCache& attn_cache = ctx.attention_device_cache(li, tp_world, tp_rank, attn_dims);

        load_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();

        DeviceHcCache& hc_cache = ctx.hc_device_cache(li);
        if (!hc_pre_float_cuda(d_h4, hc_cache.attn_fn, hc_cache.attn_scale, hc_cache.attn_base, d_x, d_hc_post, d_hc_comb, dim)) throw std::runtime_error("hc attn pre launch failed");
        if (profile_decode_sync) check_cuda(cudaDeviceSynchronize(), "sync hc pre");
        hc_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();
        auto route_stage_t = stage_t;

        uint64_t compress_ratio = layer_compress_ratio;
        std::vector<float> compressor_input_host;
        if (d_layer_kv_cache != nullptr && compress_ratio > 0 && compress_ratio <= 256 && index.shard_for_tensor(prefix + "attn.compressor.wkv.weight") != nullptr) {
            route_stage_t = Clock::now();
            if (!rmsnorm_bf16_gamma_cuda(d_x, attn_cache.attn_norm, d_attn_norm, dim, 1e-6f)) throw std::runtime_error("compressor pre-norm launch failed");
            SafeTensorsShard& comp_shard = ctx.shard_for_tensor(prefix + "attn.compressor.wkv.weight");
            const auto* comp_wkv = require_tensor(comp_shard, prefix + "attn.compressor.wkv.weight");
            const auto* comp_wgate = require_tensor(comp_shard, prefix + "attn.compressor.wgate.weight");
            const auto* comp_ape = require_tensor(comp_shard, prefix + "attn.compressor.ape");
            const auto* comp_norm = require_tensor(comp_shard, prefix + "attn.compressor.norm.weight");
            const int comp_cols = static_cast<int>(comp_wkv->shape[0]);
            const int ratio = static_cast<int>(compress_ratio);
            const bool overlap = comp_cols == attn_dims.head_dim * 2;
            const int state_cols = overlap ? attn_dims.head_dim * 2 : attn_dims.head_dim;
            const int slots = ratio * (overlap ? 2 : 1);
            if (ctx.use_gpu_compressor == 0) throw std::runtime_error("host compressor path is disabled for PyTorch resident parity");
            DeviceCompressorCache& comp_cache = ctx.compressor_device_cache(li);
            if (!fp32_to_bf16_cuda(d_attn_norm, d_compressor_input_bf16, dim)) throw std::runtime_error("compressor input bf16 round failed");
            if (!bf16_to_fp32_cuda(d_compressor_input_bf16, d_compressor_input_rounded, dim)) throw std::runtime_error("compressor input bf16 restore failed");
            if (ctx.use_gpu_compressor == 2) {
                if (!bf16_matvec_cpu_order_cuda(d_compressor_input_rounded, comp_cache.wkv, d_compressor_kv, comp_cols, dim)) throw std::runtime_error("compressor wkv launch failed");
                if (!bf16_matvec_cpu_order_cuda(d_compressor_input_rounded, comp_cache.wgate, d_compressor_score, comp_cols, dim)) throw std::runtime_error("compressor wgate launch failed");
            } else {
                if (!bf16_dual_matvec_cuda(d_compressor_input_rounded, comp_cache.wkv, comp_cache.wgate, d_compressor_kv, d_compressor_score, comp_cols, dim)) throw std::runtime_error("compressor matvec launch failed");
            }
            if (debug_forward) {
                compressor_input_host.resize(dim);
                check_cuda(cudaMemcpy(compressor_input_host.data(), d_compressor_input_rounded, static_cast<size_t>(dim) * sizeof(float), cudaMemcpyDeviceToHost), "copy compressor input debug");
                print_summary("layer=" + std::to_string(li) + ".compressor_input", compressor_input_host);
            }
            const int offset = position % ratio;
            const float* ape = comp_cache.ape + static_cast<size_t>(offset) * comp_cols;
            DeviceCompressorState& comp_state = ctx.compressor_state_for_layer(li, slots, state_cols);
            const int write_slot = overlap ? ratio + offset : offset;
            if (!compressor_update_state_cuda(d_compressor_kv, d_compressor_score, ape, comp_state.kv, comp_state.score, offset, write_slot, state_cols)) throw std::runtime_error("compressor state update launch failed");
            if ((position + 1) % ratio == 0) {
                const int compressed_slot = attn_dims.window_size + position / ratio;
                if (compressed_slot < ctx.kv_cache_capacity_for_layer(li)) {
                    float* d_pooled_slot = d_layer_kv_cache + static_cast<size_t>(compressed_slot) * attn_dims.head_dim;
                    if (!compressor_pool_cuda(comp_state.kv, comp_state.score, d_pooled_slot, ratio, attn_dims.head_dim, state_cols, overlap)) throw std::runtime_error("compressor pool launch failed");
                    if (!rmsnorm_bf16_gamma_cuda(d_pooled_slot, comp_cache.norm, d_pooled_slot, attn_dims.head_dim, 1e-6f)) throw std::runtime_error("compressed kv norm launch failed");
                    const float comp_rope_theta = static_cast<float>(config.compress_rope_theta == 0 ? 160000 : config.compress_rope_theta);
                    const float* comp_freqs = ctx.rope_inv_freqs_for(li, true, attn_dims.rope_dim, comp_rope_theta);
                    if (!head_rmsnorm_rope_freqs_cuda(d_pooled_slot, comp_freqs, 1, attn_dims.head_dim, attn_dims.rope_dim, position + 1 - ratio, false, 0.0f)) throw std::runtime_error("compressed kv rope failed");
                    if (!fp8_act_quant_dequant_cuda(d_pooled_slot, attn_dims.head_dim - attn_dims.rope_dim, 64)) throw std::runtime_error("compressed kv act quant failed");
                    if (debug_forward) {
                        std::vector<float> pooled_slot(attn_dims.head_dim);
                        check_cuda(cudaMemcpy(pooled_slot.data(), d_pooled_slot, static_cast<size_t>(attn_dims.head_dim) * sizeof(float), cudaMemcpyDeviceToHost), "copy compressed kv debug");
                        print_summary("layer=" + std::to_string(li) + ".compressed_kv", pooled_slot);
                    }
                }
                if (overlap && !compressor_shift_overlap_state_cuda(comp_state.kv, comp_state.score, ratio, state_cols)) throw std::runtime_error("compressor state shift launch failed");
            }
            route_comp_ms += elapsed_ms(route_stage_t, Clock::now());
        }
        if (d_layer_kv_cache != nullptr) {
            if (compress_ratio == 4 && compressed_ready > 0 && index.shard_for_tensor(prefix + "attn.indexer.wq_b.weight") != nullptr) {
                route_stage_t = Clock::now();
                SafeTensorsShard& idx_shard = ctx.shard_for_tensor(prefix + "attn.indexer.wq_b.weight");
                const auto* idx_comp_wkv = require_tensor(idx_shard, prefix + "attn.indexer.compressor.wkv.weight");
                const auto* idx_comp_wgate = require_tensor(idx_shard, prefix + "attn.indexer.compressor.wgate.weight");
                const auto* idx_comp_ape = require_tensor(idx_shard, prefix + "attn.indexer.compressor.ape");
                const auto* idx_comp_norm = require_tensor(idx_shard, prefix + "attn.indexer.compressor.norm.weight");
                const int idx_heads = static_cast<int>(config.index_n_heads);
                const int idx_head_dim = static_cast<int>(config.index_head_dim);
                const int idx_cols = static_cast<int>(idx_comp_wkv->shape[0]);
                const bool idx_overlap = idx_cols == idx_head_dim * 2;
                const int idx_state_cols = idx_overlap ? idx_head_dim * 2 : idx_head_dim;
                const int idx_slots = 4 * (idx_overlap ? 2 : 1);
                if (ctx.use_gpu_compressor == 0) throw std::runtime_error("host indexer compressor path is disabled for PyTorch resident parity");
                DeviceCompressorCache& idx_comp_cache = ctx.indexer_compressor_device_cache(li);
                if (ctx.use_gpu_compressor == 2) {
                    if (!bf16_matvec_cpu_order_cuda(d_compressor_input_rounded, idx_comp_cache.wkv, d_indexer_comp_kv, idx_cols, dim)) throw std::runtime_error("indexer compressor wkv launch failed");
                    if (!bf16_matvec_cpu_order_cuda(d_compressor_input_rounded, idx_comp_cache.wgate, d_indexer_comp_score, idx_cols, dim)) throw std::runtime_error("indexer compressor wgate launch failed");
                } else {
                    if (!bf16_dual_matvec_cuda(d_compressor_input_rounded, idx_comp_cache.wkv, idx_comp_cache.wgate, d_indexer_comp_kv, d_indexer_comp_score, idx_cols, dim)) throw std::runtime_error("indexer compressor matvec launch failed");
                }
                const int offset = position % 4;
                const float* ape = idx_comp_cache.ape + static_cast<size_t>(offset) * idx_cols;
                DeviceCompressorState& idx_state = ctx.indexer_compressor_state_for_layer(li, idx_slots, idx_state_cols);
                const int idx_write_slot = idx_overlap ? 4 + offset : offset;
                if (!compressor_update_state_cuda(d_indexer_comp_kv, d_indexer_comp_score, ape, idx_state.kv, idx_state.score, offset, idx_write_slot, idx_state_cols)) throw std::runtime_error("indexer compressor state update launch failed");
                if ((position + 1) % 4 == 0) {
                    float* d_idx_cache = ctx.indexer_kv_cache_for_layer(li, idx_head_dim);
                    float* d_idx_slot = d_idx_cache + static_cast<size_t>(position / 4) * idx_head_dim;
                    if (!compressor_pool_cuda(idx_state.kv, idx_state.score, d_idx_slot, 4, idx_head_dim, idx_state_cols, idx_overlap)) throw std::runtime_error("indexer compressor pool launch failed");
                    if (!rmsnorm_bf16_gamma_cuda(d_idx_slot, idx_comp_cache.norm, d_idx_slot, idx_head_dim, 1e-6f)) throw std::runtime_error("indexer compressed kv norm launch failed");
                    const float comp_rope_theta = static_cast<float>(config.compress_rope_theta == 0 ? 160000 : config.compress_rope_theta);
                    const float* comp_freqs = ctx.rope_inv_freqs_for(li, true, attn_dims.rope_dim, comp_rope_theta);
                    if (!head_rmsnorm_rope_freqs_cuda(d_idx_slot, comp_freqs, 1, idx_head_dim, attn_dims.rope_dim, position + 1 - 4, false, 0.0f)) throw std::runtime_error("indexer compressed kv rope failed");
                    if (!hadamard128_rows_cuda(d_idx_slot, d_idx_slot, 1)) throw std::runtime_error("indexer compressed kv hadamard failed");
                    if (!fp4_fake_quant128_rows_cuda(d_idx_slot, 1)) throw std::runtime_error("indexer compressed kv fp4 quant failed");
                    if (debug_forward) {
                        std::vector<float> idx_slot(idx_head_dim);
                        check_cuda(cudaMemcpy(idx_slot.data(), d_idx_slot, static_cast<size_t>(idx_head_dim) * sizeof(float), cudaMemcpyDeviceToHost), "copy indexer compressed kv debug");
                        print_summary("layer=" + std::to_string(li) + ".indexer_compressed_kv", idx_slot);
                    }
                    if (idx_overlap && !compressor_shift_overlap_state_cuda(idx_state.kv, idx_state.score, 4, idx_state_cols)) throw std::runtime_error("indexer compressor state shift launch failed");
                }
                route_indexer_comp_ms += elapsed_ms(route_stage_t, Clock::now());
                route_stage_t = Clock::now();
                if (!rmsnorm_bf16_gamma_cuda(d_x, attn_cache.attn_norm, d_attn_norm, dim, 1e-6f)) throw std::runtime_error("indexer attn norm launch failed");
                if (!fp8_e4m3_e8m0_matvec_cuda(d_attn_norm, attn_cache.wq_a, attn_cache.wq_a_scale, d_q_a, attn_dims.q_a_dim, dim)) throw std::runtime_error("indexer wq_a launch failed");
                if (!rmsnorm_bf16_gamma_cuda(d_q_a, attn_cache.q_norm, d_q_norm, attn_dims.q_a_dim, 1e-6f)) throw std::runtime_error("indexer q norm launch failed");
                DeviceIndexerCache& idx_cache = ctx.indexer_device_cache(li);
                if (!fp8_e4m3_e8m0_matvec_cuda(d_q_norm, idx_cache.wq_b, idx_cache.wq_b_scale, d_index_q, idx_heads * idx_head_dim, attn_dims.q_a_dim)) throw std::runtime_error("indexer wq_b launch failed");
                {
                    const float* idx_freqs = attn_dims.d_inv_freqs;
                    if (idx_freqs == nullptr) throw std::runtime_error("missing indexer rope freqs");
                    if (!head_rmsnorm_rope_freqs_cuda(d_index_q, idx_freqs, idx_heads, idx_head_dim, attn_dims.rope_dim, position, false, 0.0f)) throw std::runtime_error("indexer q rope launch failed");
                }
                if (!hadamard128_rows_cuda(d_index_q, d_index_q, idx_heads)) throw std::runtime_error("indexer q hadamard launch failed");
                if (!fp4_fake_quant128_rows_cuda(d_index_q, idx_heads)) throw std::runtime_error("indexer q fp4 quant launch failed");
                route_indexer_q_ms += elapsed_ms(route_stage_t, Clock::now());
                route_stage_t = Clock::now();
                const int keep = std::min<int>(compressed_ready, std::max<uint64_t>(1, config.index_topk));
                if (!indexer_select_topk_cuda(
                        d_index_q,
                        ctx.indexer_kv_cache_for_layer(li, idx_head_dim),
                        idx_cache.weights_proj,
                        d_x,
                        d_index_scores,
                        d_kv_indices + window_len,
                        compressed_ready,
                        keep,
                        idx_heads,
                        idx_head_dim,
                        dim,
                        attn_dims.window_size)) {
                    throw std::runtime_error("indexer topk launch failed");
                }
                route_indexer_topk_ms += elapsed_ms(route_stage_t, Clock::now());
                kv_index_count = window_len + keep;
            } else {
                const int compressed_count = std::min(compressed_ready, std::max(0, ctx.kv_cache_capacity_for_layer(li) - attn_dims.window_size));
                if (compressed_count > 0 && !build_decode_kv_indices_cuda(d_kv_indices + window_len, 0, 0, attn_dims.window_size, compressed_count, attn_dims.window_size)) {
                    throw std::runtime_error("build decode compressed kv indices launch failed");
                }
                kv_index_count = window_len + compressed_count;
            }
            if (debug_forward) {
                std::vector<int> kv_indices(static_cast<size_t>(kv_index_count));
                if (kv_index_count > 0) check_cuda(cudaMemcpy(kv_indices.data(), d_kv_indices, kv_indices.size() * sizeof(int), cudaMemcpyDeviceToHost), "copy kv indices debug");
                std::cout << "CPP layer=" << li << ".kv_indices count=" << kv_indices.size();
                for (int idx : kv_indices) std::cout << ' ' << idx;
                std::cout << "\n";
            }
        }

        if (profile_decode_sync) check_cuda(cudaDeviceSynchronize(), "sync route block");
        route_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();
        if (!run_single_token_attention_smoke(
                attn_dims,
                d_x,
                attn_cache.attn_norm,
                attn_cache.wq_a,
                attn_cache.wq_a_scale,
                attn_cache.q_norm,
                attn_cache.wq_b,
                attn_cache.wq_b_scale,
                attn_cache.wkv,
                attn_cache.wkv_scale,
                attn_cache.kv_norm,
                attn_cache.wo_a,
                attn_cache.wo_a_scale,
                attn_cache.wo_a_int8,
                attn_cache.wo_a_int8_scale,
                d_wo_a_x_q,
                d_wo_a_x_scale,
                attn_cache.wo_b,
                attn_cache.wo_b_scale,
                attn_cache.attn_sink,
                d_layer_kv_cache,
                kv_index_count > 0 ? d_kv_indices : nullptr,
                kv_index_count,
                layer_cache_len,
                d_attn_norm,
                d_q_a,
                d_q_norm,
                d_q,
                d_kv_a,
                d_kv_norm,
                d_attn_value,
                d_attn_mid,
                d_attn_out,
                profile_attn ? &attn_profile : nullptr)) {
            throw std::runtime_error("single-token attention smoke launch failed");
        }
#ifdef DSV4_HAVE_NCCL
        if (ctx.options.tp_world > 1) {
            if (ctx.options.nccl_id_path.empty()) throw std::runtime_error("TP attention all-reduce requires --nccl-id-path");
            auto reduce_t = Clock::now();
            all_reduce_sum_fp32_via_bf16_inplace(ctx.options.tp_world, ctx.options.tp_rank, ctx.options.device, ctx.options.nccl_id_path.c_str(), d_attn_out, dim, bf16_reduce_scratch, profile_reduce_detail ? &attn_reduce_detail : nullptr);
            if (profile_attn) {
                check_cuda(cudaDeviceSynchronize(), "sync attn reduce");
                attn_profile.reduce_ms += elapsed_ms(reduce_t, Clock::now());
            }
        }
#endif
        if (profile_decode_sync) check_cuda(cudaDeviceSynchronize(), "sync attn");
        attn_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();
        if (debug_forward) {
            check_cuda(cudaMemcpy(host_x.data(), d_attn_out, static_cast<size_t>(dim) * sizeof(float), cudaMemcpyDeviceToHost), "copy attn out debug");
            print_summary("layer=" + std::to_string(li) + ".attn_out", host_x);
        }
        if (!hc_post_float_cuda(d_attn_out, d_h4, d_hc_post, d_hc_comb, d_h4_next, dim)) throw std::runtime_error("hc attn post launch failed");
        if (!fp32_to_bf16_cuda(d_h4_next, d_h4_bf16, 4 * dim)) throw std::runtime_error("hc attn post bf16 round failed");
        if (!bf16_to_fp32_cuda(d_h4_bf16, d_h4, 4 * dim)) throw std::runtime_error("hc attn post bf16 restore failed");
        post_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();
        if (debug_forward) {
            check_cuda(cudaMemcpy(h4.data(), d_h4, static_cast<size_t>(4) * dim * sizeof(float), cudaMemcpyDeviceToHost), "copy attn post debug");
            print_summary("layer=" + std::to_string(li) + ".attn_post", h4);
        }
        if (!hc_pre_float_cuda(d_h4, hc_cache.ffn_fn, hc_cache.ffn_scale, hc_cache.ffn_base, d_resid1, d_hc_post, d_hc_comb, dim)) throw std::runtime_error("hc ffn pre launch failed");
        if (debug_forward) {
            check_cuda(cudaMemcpy(host_x.data(), d_resid1, static_cast<size_t>(dim) * sizeof(float), cudaMemcpyDeviceToHost), "copy ffn hc pre debug");
            print_summary("layer=" + std::to_string(li) + ".ffn_hc_pre", host_x);
        }
        if (!rmsnorm_bf16_gamma_cuda(d_resid1, attn_cache.ffn_norm, d_ffn_norm, dim, 1e-6f)) throw std::runtime_error("ffn norm launch failed");
        const int route_count = static_cast<int>(std::min<uint64_t>(config.n_activated_experts, config.n_routed_experts));
        std::vector<RoutedExpert> routed;
        std::vector<int64_t> selected_route_ids;
        if (profile_decode_sync) check_cuda(cudaDeviceSynchronize(), "sync before gate");
        route_stage_t = Clock::now();
        DeviceGateCache& gate = ctx.gate_device_cache(li);
        if (static_cast<uint64_t>(li) < config.n_hash_layers && gate.tid2eid != nullptr) {
            if (!gate_hash_bf16_cuda(d_ffn_norm, gate.weight, gate.tid2eid, gate.original, d_route_indices, d_route_weights, token, gate.dim, gate.hash_topk, route_count, static_cast<float>(config.route_scale))) throw std::runtime_error("hash gate launch failed");
        } else {
            if (!gate_topk_bf16_cuda_with_buffers(d_ffn_norm, gate.weight, gate.bias, gate.original, gate.scored, d_route_indices, d_route_weights, gate.experts, gate.dim, route_count, static_cast<float>(config.route_scale))) throw std::runtime_error("gate topk launch failed");
        }
        double route_gate_kernel_ms = 0.0;
        double route_d2h_ms = 0.0;
        if (profile_decode_sync) {
            check_cuda(cudaDeviceSynchronize(), "sync route gate kernel");
            route_gate_kernel_ms = elapsed_ms(route_stage_t, Clock::now());
        }
        auto d2h_t0 = Clock::now();
        selected_route_ids.resize(route_count);
        check_cuda(cudaMemcpy(selected_route_ids.data(), d_route_indices, selected_route_ids.size() * sizeof(int64_t), cudaMemcpyDeviceToHost), "copy gate route ids");
        if (profile_decode_sync) {
            route_d2h_ms = elapsed_ms(d2h_t0, Clock::now());
        }
        route_gate_ms += elapsed_ms(route_stage_t, Clock::now());
        routed.reserve(selected_route_ids.size());
        for (int64_t route_id : selected_route_ids) routed.push_back(RoutedExpert{static_cast<int>(route_id), 0.0f});
        route_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();
        check_cuda(cudaMemset(d_moe, 0, static_cast<size_t>(dim) * sizeof(float)), "zero moe");
        moe_stage_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();
        const int experts_per_rank = ctx.options.tp_world > 1 ? static_cast<int>(config.n_routed_experts / ctx.options.tp_world) : static_cast<int>(config.n_routed_experts);
        const int expert_start = ctx.options.tp_rank * experts_per_rank;
        const int expert_end = ctx.options.tp_world > 1 ? expert_start + experts_per_rank : static_cast<int>(config.n_routed_experts);
        std::vector<int64_t> route_indices;
        std::vector<Fp4View> active_w1;
        std::vector<Fp4View> active_w2;
        std::vector<Fp4View> active_w3;
        route_indices.reserve(routed.size());
        active_w1.reserve(routed.size());
        active_w2.reserve(routed.size());
        active_w3.reserve(routed.size());
        std::vector<int> active_local_ids;
        for (const RoutedExpert& route : routed) {
            if (ctx.options.tp_world > 1 && (route.id < expert_start || route.id >= expert_end)) continue;
            route_indices.push_back(route.id);
            active_local_ids.push_back(route.id - expert_start);
            active_w1.push_back(ctx.fp4_view(prefix + "ffn.experts." + std::to_string(route.id) + ".w1.weight"));
            active_w2.push_back(ctx.fp4_view(prefix + "ffn.experts." + std::to_string(route.id) + ".w2.weight"));
            active_w3.push_back(ctx.fp4_view(prefix + "ffn.experts." + std::to_string(route.id) + ".w3.weight"));
        }
        bool has_active_moe = !active_w1.empty();
        DeviceFp4ActiveArena* active_arena = nullptr;
        if (has_active_moe) {
            DeviceFp4ExpertCache sample;
            sample.w1_bytes = active_w1.front().w->nbytes;
            sample.s1_bytes = active_w1.front().s->nbytes;
            sample.w2_bytes = active_w2.front().w->nbytes;
            sample.s2_bytes = active_w2.front().s->nbytes;
            sample.w3_bytes = active_w3.front().w->nbytes;
            sample.s3_bytes = active_w3.front().s->nbytes;
            active_arena = &ctx.active_fp4_arena(li, tp_world, tp_rank, use_sparse_arena ? sparse_slots_per_layer : experts_per_rank, sample, use_sparse_arena);
            if (moe_stage_event_recorded) check_cuda(cudaStreamWaitEvent(moe_copy_stream, moe_stage_event, 0), "wait prior moe stage event");
            std::vector<int64_t> route_indices_kernel = route_indices;
            for (size_t ri = 0; ri < active_w1.size(); ++ri) {
                const int local = active_local_ids[ri];
                int slot = local;
                bool need_stage = false;
                if (use_sparse_arena) {
                    bool already_staged = false;
                    slot = ctx.acquire_sparse_slot(*active_arena, local, already_staged);
                    need_stage = !already_staged;
                    route_indices_kernel[ri] = static_cast<int64_t>(expert_start + slot);
                } else {
                    need_stage = active_arena->staged_local.insert(local).second;
                }
                if (need_stage) {
                    HostFp4ExpertSlot& slot_h = ctx.host_fp4_slot(li, expert_start + local, active_w1[ri], active_w2[ri], active_w3[ri]);
                    check_cuda(cudaMemcpyAsync(active_arena->w1 + static_cast<size_t>(slot) * active_arena->w1_bytes, slot_h.h_w1q, active_arena->w1_bytes, cudaMemcpyHostToDevice, moe_copy_stream), "stage active w1");
                    check_cuda(cudaMemcpyAsync(active_arena->s1 + static_cast<size_t>(slot) * active_arena->s1_bytes, slot_h.h_w1s, active_arena->s1_bytes, cudaMemcpyHostToDevice, moe_copy_stream), "stage active s1");
                    check_cuda(cudaMemcpyAsync(active_arena->w2 + static_cast<size_t>(slot) * active_arena->w2_bytes, slot_h.h_w2q, active_arena->w2_bytes, cudaMemcpyHostToDevice, moe_copy_stream), "stage active w2");
                    check_cuda(cudaMemcpyAsync(active_arena->s2 + static_cast<size_t>(slot) * active_arena->s2_bytes, slot_h.h_w2s, active_arena->s2_bytes, cudaMemcpyHostToDevice, moe_copy_stream), "stage active s2");
                    check_cuda(cudaMemcpyAsync(active_arena->w3 + static_cast<size_t>(slot) * active_arena->w3_bytes, slot_h.h_w3q, active_arena->w3_bytes, cudaMemcpyHostToDevice, moe_copy_stream), "stage active w3");
                    check_cuda(cudaMemcpyAsync(active_arena->s3 + static_cast<size_t>(slot) * active_arena->s3_bytes, slot_h.h_w3s, active_arena->s3_bytes, cudaMemcpyHostToDevice, moe_copy_stream), "stage active s3");
                }
            }
            check_cuda(cudaMemcpyAsync(d_route_indices, route_indices_kernel.data(), route_indices_kernel.size() * sizeof(int64_t), cudaMemcpyHostToDevice, moe_copy_stream), "copy active route indices");
            check_cuda(cudaEventRecord(moe_stage_event, moe_copy_stream), "record moe stage event");
            moe_stage_event_recorded = true;
        }
        if (profile_decode_sync) check_cuda(cudaDeviceSynchronize(), "sync moe stage");
        moe_stage_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();
        {
            DeviceSharedCache& shared = ctx.shared_device_cache(li, tp_world, tp_rank, dim);
            const int shared_inter = inter;
            if (!fp8_e4m3_e8m0_matvec_cuda(d_ffn_norm, shared.w1, shared.s1, d_gate, shared_inter, dim)) throw std::runtime_error("shared w1 launch failed");
            if (!fp8_e4m3_e8m0_matvec_cuda(d_ffn_norm, shared.w3, shared.s3, d_up, shared_inter, dim)) throw std::runtime_error("shared w3 launch failed");
            if (!silu_mul_cuda(d_gate, d_up, d_hidden, shared_inter)) throw std::runtime_error("shared silu launch failed");
            if (!fp8_e4m3_e8m0_matvec_cuda(d_hidden, shared.w2, shared.s2, d_shared_out, dim, shared_inter)) throw std::runtime_error("shared w2 launch failed");
        }
        if (profile_decode_sync) check_cuda(cudaDeviceSynchronize(), "sync shared moe");
        shared_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();
        if (has_active_moe) {
            check_cuda(cudaStreamWaitEvent(nullptr, moe_stage_event, 0), "wait active moe stage");
            DeviceMoeDecodeWorkspace& moe_workspace = ctx.moe_decode_workspace(route_count, dim, inter);
            const int moe_n_local = use_sparse_arena ? active_arena->capacity : experts_per_rank;
            if (!moe_single_token_fp4_cuda_with_workspace(d_ffn_norm, d_route_indices, d_route_weights, active_arena->w1, active_arena->s1, active_arena->w2, active_arena->s2, active_arena->w3, active_arena->s3, d_resid2, route_count, expert_start, moe_n_local, dim, inter, static_cast<float>(config.swiglu_limit), moe_workspace.fp4)) {
                throw std::runtime_error("active fp4 moe launch failed");
            }
            if (!vector_accum_cuda(d_resid2, d_moe, dim, 1.0f)) throw std::runtime_error("moe accum failed");
        }
        if (profile_decode_sync) check_cuda(cudaDeviceSynchronize(), "sync moe kernel");
        moe_kernel_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();
#ifdef DSV4_HAVE_NCCL
        if (ctx.options.tp_world > 1) {
            if (ctx.options.nccl_id_path.empty()) throw std::runtime_error("TP MoE all-reduce requires --nccl-id-path");
            all_reduce_sum_fp32_via_bf16_inplace(ctx.options.tp_world, ctx.options.tp_rank, ctx.options.device, ctx.options.nccl_id_path.c_str(), d_moe, dim, bf16_reduce_scratch, profile_reduce_detail ? &moe_reduce_detail : nullptr);
        }
#endif
        if (profile_decode_sync) check_cuda(cudaDeviceSynchronize(), "sync moe reduce");
        moe_reduce_ms += elapsed_ms(stage_t, Clock::now());
        moe_ms += moe_stage_ms + moe_kernel_ms + moe_reduce_ms;
        stage_t = Clock::now();
        if (!vector_accum_cuda(d_shared_out, d_moe, dim, 1.0f)) throw std::runtime_error("shared accum failed");
        shared_ms += elapsed_ms(stage_t, Clock::now());
        stage_t = Clock::now();
        if (debug_forward) {
            check_cuda(cudaMemcpy(host_x.data(), d_moe, static_cast<size_t>(dim) * sizeof(float), cudaMemcpyDeviceToHost), "copy moe debug");
            print_summary("layer=" + std::to_string(li) + ".moe_out", host_x);
        }
        if (!hc_post_float_cuda(d_moe, d_h4, d_hc_post, d_hc_comb, d_h4_next, dim)) throw std::runtime_error("hc ffn post launch failed");
        if (!fp32_to_bf16_cuda(d_h4_next, d_h4_bf16, 4 * dim)) throw std::runtime_error("hc ffn post bf16 round failed");
        if (!bf16_to_fp32_cuda(d_h4_bf16, d_h4, 4 * dim)) throw std::runtime_error("hc ffn post bf16 restore failed");
        post_ms += elapsed_ms(stage_t, Clock::now());
        if (profile_forward) {
            const double layer_ms = elapsed_ms(layer_t0, Clock::now());
            std::cout << "CPP_PROFILE layer=" << li
                      << " total_ms=" << layer_ms
                      << " load_ms=" << load_ms
                      << " hc_ms=" << hc_ms
                      << " attn_ms=" << attn_ms;
            if (profile_attn) {
                std::cout << " attn_q_ms=" << attn_profile.q_ms
                          << " attn_kv_ms=" << attn_profile.kv_ms
                          << " attn_core_ms=" << attn_profile.core_ms
                          << " attn_wo_a_ms=" << attn_profile.wo_a_ms
                          << " attn_wo_b_ms=" << attn_profile.wo_b_ms
                          << " attn_reduce_ms=" << attn_profile.reduce_ms;
            }
            std::cout << " route_ms=" << route_ms
                      << " route_comp_ms=" << route_comp_ms
                      << " route_indexer_comp_ms=" << route_indexer_comp_ms
                      << " route_indexer_q_ms=" << route_indexer_q_ms
                      << " route_indexer_topk_ms=" << route_indexer_topk_ms
                      << " route_gate_ms=" << route_gate_ms;
            if (profile_decode_sync) {
                std::cout << " route_gate_kernel_ms=" << route_gate_kernel_ms
                          << " route_d2h_ms=" << route_d2h_ms;
            }
            std::cout << " moe_ms=" << moe_ms
                      << " moe_stage_ms=" << moe_stage_ms
                      << " moe_kernel_ms=" << moe_kernel_ms
                      << " moe_reduce_ms=" << moe_reduce_ms;
            if (profile_reduce_detail) {
                std::cout << " attn_reduce_pre_ms=" << attn_reduce_detail.pre_sync_ms
                          << " attn_reduce_pack_ms=" << attn_reduce_detail.pack_ms
                          << " attn_reduce_nccl_ms=" << attn_reduce_detail.nccl_ms
                          << " attn_reduce_unpack_ms=" << attn_reduce_detail.unpack_ms
                          << " moe_reduce_pre_ms=" << moe_reduce_detail.pre_sync_ms
                          << " moe_reduce_pack_ms=" << moe_reduce_detail.pack_ms
                          << " moe_reduce_nccl_ms=" << moe_reduce_detail.nccl_ms
                          << " moe_reduce_unpack_ms=" << moe_reduce_detail.unpack_ms;
            }
            std::cout << " shared_ms=" << shared_ms
                      << " post_ms=" << post_ms << "\n";
        }
        total_load_ms += load_ms;
        total_hc_ms += hc_ms;
        total_attn_ms += attn_ms;
        total_attn_profile.q_ms += attn_profile.q_ms;
        total_attn_profile.kv_ms += attn_profile.kv_ms;
        total_attn_profile.core_ms += attn_profile.core_ms;
        total_attn_profile.wo_a_ms += attn_profile.wo_a_ms;
        total_attn_profile.wo_b_ms += attn_profile.wo_b_ms;
        total_attn_profile.reduce_ms += attn_profile.reduce_ms;
        total_route_ms += route_ms;
        total_route_comp_ms += route_comp_ms;
        total_route_indexer_comp_ms += route_indexer_comp_ms;
        total_route_indexer_q_ms += route_indexer_q_ms;
        total_route_indexer_topk_ms += route_indexer_topk_ms;
        total_route_gate_ms += route_gate_ms;
        total_route_gate_kernel_ms += route_gate_kernel_ms;
        total_route_d2h_ms += route_d2h_ms;
        total_moe_ms += moe_ms;
        total_moe_stage_ms += moe_stage_ms;
        total_moe_kernel_ms += moe_kernel_ms;
        total_moe_reduce_ms += moe_reduce_ms;
        if (profile_reduce_detail) {
            total_attn_reduce_detail.pre_sync_ms += attn_reduce_detail.pre_sync_ms;
            total_attn_reduce_detail.pack_ms += attn_reduce_detail.pack_ms;
            total_attn_reduce_detail.nccl_ms += attn_reduce_detail.nccl_ms;
            total_attn_reduce_detail.unpack_ms += attn_reduce_detail.unpack_ms;
            total_moe_reduce_detail.pre_sync_ms += moe_reduce_detail.pre_sync_ms;
            total_moe_reduce_detail.pack_ms += moe_reduce_detail.pack_ms;
            total_moe_reduce_detail.nccl_ms += moe_reduce_detail.nccl_ms;
            total_moe_reduce_detail.unpack_ms += moe_reduce_detail.unpack_ms;
        }
        total_shared_ms += shared_ms;
        total_post_ms += post_ms;
        if (debug_forward) {
            check_cuda(cudaMemcpy(h4.data(), d_h4, static_cast<size_t>(4) * dim * sizeof(float), cudaMemcpyDeviceToHost), "copy layer h4 debug");
            print_summary("layer=" + std::to_string(li), h4);
        }
    }

    if (profile_forward) {
        std::cout << "CPP_PROFILE_TOTAL load_ms=" << total_load_ms
                  << " hc_ms=" << total_hc_ms
                  << " attn_ms=" << total_attn_ms;
        if (profile_attn) {
            std::cout << " attn_q_ms=" << total_attn_profile.q_ms
                      << " attn_kv_ms=" << total_attn_profile.kv_ms
                      << " attn_core_ms=" << total_attn_profile.core_ms
                      << " attn_wo_a_ms=" << total_attn_profile.wo_a_ms
                      << " attn_wo_b_ms=" << total_attn_profile.wo_b_ms
                      << " attn_reduce_ms=" << total_attn_profile.reduce_ms;
        }
        std::cout << " route_ms=" << total_route_ms
                  << " route_comp_ms=" << total_route_comp_ms
                  << " route_indexer_comp_ms=" << total_route_indexer_comp_ms
                  << " route_indexer_q_ms=" << total_route_indexer_q_ms
                  << " route_indexer_topk_ms=" << total_route_indexer_topk_ms
                  << " route_gate_ms=" << total_route_gate_ms;
        if (profile_decode_sync) {
            std::cout << " route_gate_kernel_ms=" << total_route_gate_kernel_ms
                      << " route_d2h_ms=" << total_route_d2h_ms;
        }
        std::cout << " moe_ms=" << total_moe_ms
                  << " moe_stage_ms=" << total_moe_stage_ms
                  << " moe_kernel_ms=" << total_moe_kernel_ms
                  << " moe_reduce_ms=" << total_moe_reduce_ms;
        if (profile_reduce_detail) {
            std::cout << " attn_reduce_pre_ms=" << total_attn_reduce_detail.pre_sync_ms
                      << " attn_reduce_pack_ms=" << total_attn_reduce_detail.pack_ms
                      << " attn_reduce_nccl_ms=" << total_attn_reduce_detail.nccl_ms
                      << " attn_reduce_unpack_ms=" << total_attn_reduce_detail.unpack_ms
                      << " moe_reduce_pre_ms=" << total_moe_reduce_detail.pre_sync_ms
                      << " moe_reduce_pack_ms=" << total_moe_reduce_detail.pack_ms
                      << " moe_reduce_nccl_ms=" << total_moe_reduce_detail.nccl_ms
                      << " moe_reduce_unpack_ms=" << total_moe_reduce_detail.unpack_ms;
        }
        std::cout << " shared_ms=" << total_shared_ms
                  << " post_ms=" << total_post_ms << "\n";
    }
    check_cuda(cudaMemcpy(h4.data(), d_h4, static_cast<size_t>(4) * dim * sizeof(float), cudaMemcpyDeviceToHost), "copy final hc h4");
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
    ctx.last_local_logits = std::move(logits);
    ctx.last_head_rows = head_rows;
    ctx.last_local_head_start = local_head_start;
    if (!std::isfinite(checksum) || !std::isfinite(top_logit)) throw std::runtime_error("non-finite smoke logits");

    cudaFree(d_embed);
    cudaFree(d_w1);
    cudaFree(d_s1);
    cudaFree(d_w2);
    cudaFree(d_s2);
    cudaFree(d_w3);
    cudaFree(d_s3);
    cudaFree(d_head);
    cudaFree(d_final_norm_gamma);
    cudaFree(d_x);
    cudaFree(d_h4);
    cudaFree(d_h4_next);
    cudaFree(d_h4_bf16);
    cudaFree(d_hc_post);
    cudaFree(d_hc_comb);
    cudaFree(d_attn_norm);
    cudaFree(d_q_a);
    cudaFree(d_q_norm);
    cudaFree(d_q);
    cudaFree(d_kv_a);
    cudaFree(d_kv_norm);
    cudaFree(d_attn_value);
    cudaFree(d_attn_mid);
    cudaFree(d_wo_a_x_q);
    cudaFree(d_wo_a_x_scale);
    cudaFree(d_attn_out);
    cudaFree(d_compressor_input_bf16);
    cudaFree(d_compressor_input_rounded);
    cudaFree(d_compressor_kv);
    cudaFree(d_compressor_score);
    cudaFree(d_indexer_comp_kv);
    cudaFree(d_indexer_comp_score);
    cudaFree(d_index_q);
    cudaFree(d_indexer_kv);
    cudaFree(d_index_weight_proj);
    cudaFree(d_index_scores);
    cudaFree(d_kv_indices);
    cudaFree(d_resid1);
    cudaFree(d_ffn_norm);
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_hidden);
    cudaFree(d_moe);
    cudaFree(d_resid2);
    cudaFree(d_shared_out);
    cudaFree(d_route_indices);
    cudaFree(d_route_weights);
    cudaFree(d_logits);

    cudaEventDestroy(moe_stage_event);
    cudaStreamDestroy(moe_copy_stream);

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

// --- GGUF Q2 forward path (Phase 3, work in progress) ---------------------

namespace {

// Parallels SafeForwardContext for the GGUF Q2 path. Currently a minimal
// shell that owns the GGUF mmap + WeightSource + ModelConfig; per-layer
// caches and forward state will be added as dense/MoE operators are wired
// in. Lives in the anonymous namespace because external callers go through
// the run_gguf_* entry points.
struct GgufForwardContext {
    std::string ckpt_path;
    std::unique_ptr<GGUFWeightSource> weight_source;
    ModelConfig config;

    explicit GgufForwardContext(const std::string& path)
        : ckpt_path(path),
          weight_source(std::make_unique<GGUFWeightSource>(path)),
          config(ModelConfig::from_gguf(weight_source->file())) {}
};

}  // namespace

GgufSmokeResult run_gguf_min_layer_smoke(const std::string& ckpt_path) {
    if (!is_gguf_path(ckpt_path)) {
        throw std::runtime_error("run_gguf_min_layer_smoke: not a GGUF path: " + ckpt_path);
    }
    GgufForwardContext ctx(ckpt_path);
    WeightView embed = ctx.weight_source->require("embed.weight");
    if (embed.shape.size() != 2) {
        throw std::runtime_error("gguf embed.weight rank != 2");
    }
    GgufSmokeResult r;
    r.n_layers = static_cast<int>(ctx.config.n_layers);
    r.n_hash_layers = static_cast<int>(ctx.config.n_hash_layers);
    r.dim = static_cast<int>(ctx.config.dim);
    r.moe_inter_dim = static_cast<int>(ctx.config.moe_inter_dim);
    r.n_routed_experts = static_cast<int>(ctx.config.n_routed_experts);
    r.n_activated_experts = static_cast<int>(ctx.config.n_activated_experts);
    r.vocab = static_cast<int>(embed.shape[1]);
    return r;
}

namespace {

// Convert host F32 array to BF16 by truncating the low 16 bits of the IEEE
// 754 binary32 representation. Matches the round-down-to-bf16 convention
// used by the safetensors loader's BF16 norm gammas, so the GGUF F32-gamma
// path produces RMSNorm output bit-equivalent to a BF16-gamma checkpoint.
std::vector<uint16_t> f32_to_bf16_host(const float* src, size_t n) {
    std::vector<uint16_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &src[i], sizeof(bits));
        out[i] = static_cast<uint16_t>(bits >> 16);
    }
    return out;
}

// F16 (IEEE binary16) -> BF16 via F32 intermediate. F16 has 10-bit mantissa,
// BF16 has 7-bit mantissa, so this drops 3 bits of fraction. Used for staging
// GGUF F16 weights into the existing BF16 matvec kernels (e.g., gate scoring).
float f16_bits_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp16 = (h >> 10) & 0x1Fu;
    const uint32_t man16 = h & 0x3FFu;
    uint32_t f32_bits;
    if (exp16 == 0) {
        if (man16 == 0) {
            f32_bits = sign;
        } else {
            uint32_t m = man16;
            int e = -1;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3FFu;
            const uint32_t exp32 = static_cast<uint32_t>(127 - 15 + e);
            f32_bits = sign | (exp32 << 23) | (m << 13);
        }
    } else if (exp16 == 0x1Fu) {
        f32_bits = sign | (0xFFu << 23) | (man16 << 13);
    } else {
        const uint32_t exp32 = exp16 + (127 - 15);
        f32_bits = sign | (exp32 << 23) | (man16 << 13);
    }
    float out;
    std::memcpy(&out, &f32_bits, sizeof(out));
    return out;
}

std::vector<uint16_t> f16_to_bf16_host(const uint16_t* src, size_t n) {
    std::vector<uint16_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        const float f = f16_bits_to_float(src[i]);
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        out[i] = static_cast<uint16_t>(bits >> 16);
    }
    return out;
}

std::vector<float> f16_to_f32_host(const uint16_t* src, size_t n) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = f16_bits_to_float(src[i]);
    return out;
}

float device_vector_rms(const float* d_x, int n) {
    std::vector<float> h(n);
    check_cuda(cudaMemcpy(h.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost),
               "device_vector_rms memcpy");
    double s = 0.0;
    for (float v : h) s += static_cast<double>(v) * static_cast<double>(v);
    return n > 0 ? static_cast<float>(std::sqrt(s / n)) : 0.0f;
}

// ---- Per-layer GGUF forward helper -----------------------------------------
//
// Inputs:
//   - GgufLayerDeviceWeights: all dense weights already staged on device
//     (attention Q8_0 + norms BF16 + shared experts Q8_0 + gate W BF16 + EITHER
//     tid2eid_i64 [topk] for hash layers OR exp_probs_b bias F32 [n_experts]
//     for non-hash) + top-k routed Q2 experts packed into 3 device buffers.
//   - GgufLayerScratch: pre-allocated activation + staging buffers.
//   - GgufLayerDims: dim/heads/etc.
//   - token: used by hash gate's tid2eid lookup; ignored otherwise.
//   - position: for RoPE.
// In-place: reads s.d_x [dim], writes back s.d_x [dim] after both residuals.
// For non-hash layers, the chosen expert ids land in s.d_gate_indices.

struct GgufCompressorDeviceWeights {
    const uint16_t* wkv = nullptr;
    const uint16_t* wgate = nullptr;
    const float* ape = nullptr;
    const uint16_t* norm = nullptr;
    int cols = 0;
    int state_cols = 0;
    int slots = 0;
    int ratio = 0;
    bool overlap = false;
    bool present = false;
};

struct GgufIndexerDeviceWeights {
    const uint16_t* wq_b = nullptr;
    const uint16_t* weights_proj = nullptr;
    GgufCompressorDeviceWeights comp;
    int heads = 0;
    int head_dim = 0;
    bool present = false;
};

struct GgufSparseLayerState {
    float* compressor_kv = nullptr;
    float* compressor_score = nullptr;
    float* indexer_kv_cache = nullptr;
    float* indexer_comp_kv = nullptr;
    float* indexer_comp_score = nullptr;
};

struct GgufLayerDeviceWeights {
    const uint16_t* d_attn_gamma = nullptr;
    const uint16_t* d_q_gamma = nullptr;
    const uint16_t* d_kv_gamma = nullptr;
    const uint8_t* d_wq_a = nullptr;
    const uint8_t* d_wq_b = nullptr;
    const uint8_t* d_wkv = nullptr;
    const uint8_t* d_wo_a = nullptr;
    const uint8_t* d_wo_b = nullptr;
    const float* d_attn_sink = nullptr;

    const uint16_t* d_ffn_gamma = nullptr;
    const uint8_t* d_shared_w1 = nullptr;
    const uint8_t* d_shared_w2 = nullptr;
    const uint8_t* d_shared_w3 = nullptr;

    const uint16_t* d_gate_w_bf16 = nullptr;
    bool is_hash = false;
    const int64_t* d_tid2eid_i64 = nullptr;  // hash only: [topk] for this token
    const float* d_gate_bias_f32 = nullptr;  // non-hash only: [n_experts]

    const uint8_t* d_routed_w1 = nullptr;
    const uint8_t* d_routed_w2 = nullptr;
    const uint8_t* d_routed_w3 = nullptr;
    // Hash layers can either pass a one-token tid2eid slice (hash_token=0,
    // hash_table_topk=topk) or a full resident [vocab, topk] table
    // (hash_token=real token id).
    int hash_token = 0;
    int hash_table_topk = 0;
    // Number of expert slots in d_routed_w*. Legacy staging uses top-k slots;
    // resident TP mode uses experts_per_rank local-expert slots.
    int routed_n_experts = 0;
    DType routed_w1_dtype = DType::Unknown;
    DType routed_w2_dtype = DType::Unknown;
    DType routed_w3_dtype = DType::Unknown;
    const GgufCompressorDeviceWeights* compressor = nullptr;
    const GgufIndexerDeviceWeights* indexer = nullptr;
    GgufSparseLayerState* sparse_state = nullptr;
    int compress_ratio = 0;
    int sparse_window_size = 0;
    int sparse_attn_threshold = 0;
    int index_topk = 0;
};

struct GgufLayerDims {
    int dim = 0;
    int q_a_dim = 0;
    int heads = 0;
    int head_dim = 0;
    int q_full = 0;          // heads * head_dim
    int kv_dim = 0;
    int rope_dim = 0;
    int o_groups = 0;
    int o_lora_rank = 0;
    int group_in_dim = 0;
    int attn_mid = 0;
    int moe_inter = 0;
    int n_experts = 0;
    int topk = 0;
    float rope_theta = 0.0f;
    float compress_rope_theta = 0.0f;
    float swiglu_limit = 0.0f;
    float route_scale = 0.0f;
};

struct GgufLayerScratch {
    // hidden state (read at entry, written at exit)
    float* d_x = nullptr;
    // attention scratch
    float* d_x_pre_attn = nullptr;
    float* d_x_normed = nullptr;
    float* d_q_a = nullptr;
    float* d_q_normed = nullptr;
    float* d_q = nullptr;
    float* d_kv_a = nullptr;
    float* d_kv = nullptr;
    float* d_attn_value = nullptr;
    float* d_attn_mid = nullptr;
    float* d_attn_out = nullptr;
    // FFN scratch
    float* d_x_pre_ffn = nullptr;
    float* d_x_normed_ffn = nullptr;
    float* d_shared_gate = nullptr;
    float* d_shared_up = nullptr;
    float* d_shared_hidden = nullptr;
    float* d_shared_out = nullptr;
    float* d_moe_out = nullptr;
    float* d_ffn_combined = nullptr;
    // routed MoE staging
    int8_t* d_x_q = nullptr;
    float* d_x_scale = nullptr;
    int64_t* d_route_slots = nullptr;     // [topk] = {0,1,..k-1}
    float* d_route_weights = nullptr;     // [topk]
    float* d_route_gate = nullptr;        // [topk, moe_inter]
    float* d_route_up = nullptr;          // [topk, moe_inter]
    int8_t* d_route_hidden_q = nullptr;   // [topk, moe_inter]
    float* d_route_hidden_scale = nullptr; // [topk, hidden_groups]
    float* d_route_hidden = nullptr;      // [topk, moe_inter] for IQ1_M path
    // gate scratch
    float* d_gate_scores_scratch = nullptr; // hash:[topk]; non-hash:[n_experts]
    float* d_gate_scored_scratch = nullptr; // non-hash only: [n_experts]
    int64_t* d_gate_indices = nullptr;    // [topk]
    // Per-layer KV cache [cache_capacity, head_dim] float. If non-null, the
    // attention helper writes the current step's kv_norm into slot `position`
    // and runs cached attention over the [0..cache_len) prefix. If
    // d_attn_weight_scratch is non-null, cached attention uses it as global
    // [heads, cache_capacity] softmax scratch so long contexts are not limited
    // by dynamic shared-memory size.
    int* d_kv_indices = nullptr;
    int kv_index_count = 0;
    int sparse_kv_index_count = 0;
    bool sparse_attn_used = false;
    float* d_attn_weight_scratch = nullptr;
    float* d_kv_cache = nullptr;
    int cache_capacity = 0;
    uint16_t* d_compressor_input_bf16 = nullptr;
    float* d_compressor_input_rounded = nullptr;
    float* d_compressor_kv = nullptr;
    float* d_compressor_score = nullptr;
    float* d_indexer_comp_kv = nullptr;
    float* d_indexer_comp_score = nullptr;
    float* d_index_q = nullptr;
    float* d_index_scores = nullptr;
};

// Phase 1: attention residual + ffn_norm + shared expert + gate scoring.
// On exit, d_gate_indices contains the top-k expert ids for this token (both
// hash and non-hash layers populate this). The caller stages those experts'
// w1/w2/w3 into w.d_routed_* before calling gguf_layer_forward_moe.
void gguf_layer_forward_attn_to_gate(const GgufLayerDeviceWeights& w,
                                     GgufLayerScratch& s,
                                     const GgufLayerDims& d,
                                     int position,
                                     const GgufReduceContext* reduce_ctx) {
    // ===== Attention =====
    check_cuda(cudaMemcpy(s.d_x_pre_attn, s.d_x, d.dim * sizeof(float),
                          cudaMemcpyDeviceToDevice), "save x_pre_attn");
    if (!rmsnorm_bf16_gamma_cuda(s.d_x, w.d_attn_gamma, s.d_x_normed, d.dim, 1e-6f))
        throw std::runtime_error("attn_norm failed");
    const bool use_sparse_compressor =
        w.compressor != nullptr && w.compressor->present && w.sparse_state != nullptr &&
        w.compress_ratio > 0 && w.sparse_window_size > 0 && s.d_kv_cache != nullptr &&
        s.d_kv_indices != nullptr && s.d_compressor_input_bf16 != nullptr &&
        s.d_compressor_kv != nullptr && s.d_compressor_score != nullptr;
    const bool use_sparse_attention = use_sparse_compressor &&
        (w.sparse_attn_threshold <= 0 || (position + 1) > w.sparse_attn_threshold);
    if (use_sparse_compressor) {
        const auto& comp = *w.compressor;
        if (!fp32_to_bf16_cuda(s.d_x_normed, s.d_compressor_input_bf16, d.dim))
            throw std::runtime_error("GGUF sparse compressor input bf16 failed");
        if (!bf16_dual_matvec_bf16_x_cuda(s.d_compressor_input_bf16, comp.wkv, comp.wgate,
                                          s.d_compressor_kv, s.d_compressor_score,
                                          comp.cols, d.dim))
            throw std::runtime_error("GGUF sparse compressor matvec failed");
        const int offset = position % comp.ratio;
        const float* ape = comp.ape + static_cast<size_t>(offset) * comp.cols;
        const int write_slot = comp.overlap ? comp.ratio + offset : offset;
        if (!compressor_update_state_cuda(s.d_compressor_kv, s.d_compressor_score, ape,
                                          w.sparse_state->compressor_kv,
                                          w.sparse_state->compressor_score,
                                          offset, write_slot, comp.state_cols))
            throw std::runtime_error("GGUF sparse compressor state update failed");
        if ((position + 1) % comp.ratio == 0) {
            const int compressed_slot = w.sparse_window_size + position / comp.ratio;
            if (compressed_slot < s.cache_capacity) {
                float* d_pooled_slot = s.d_kv_cache + static_cast<size_t>(compressed_slot) * d.kv_dim;
                if (!compressor_pool_cuda(w.sparse_state->compressor_kv,
                                          w.sparse_state->compressor_score,
                                          d_pooled_slot, comp.ratio, d.head_dim,
                                          comp.state_cols, comp.overlap))
                    throw std::runtime_error("GGUF sparse compressor pool failed");
                if (!rmsnorm_bf16_gamma_cuda(d_pooled_slot, comp.norm, d_pooled_slot,
                                             d.head_dim, 1e-6f))
                    throw std::runtime_error("GGUF sparse compressed kv norm failed");
                const float comp_theta = d.compress_rope_theta > 0.0f ? d.compress_rope_theta : d.rope_theta;
                if (!head_rmsnorm_rope_cuda(d_pooled_slot, /*heads=*/1, d.head_dim,
                                            d.rope_dim, position + 1 - comp.ratio,
                                            comp_theta, false, 0.0f))
                    throw std::runtime_error("GGUF sparse compressed kv rope failed");
                if (d.head_dim > d.rope_dim &&
                    !fp8_act_quant_dequant_cuda(d_pooled_slot, d.head_dim - d.rope_dim, 64))
                    throw std::runtime_error("GGUF sparse compressed kv act quant failed");
            }
            if (comp.overlap && !compressor_shift_overlap_state_cuda(
                    w.sparse_state->compressor_kv, w.sparse_state->compressor_score,
                    comp.ratio, comp.state_cols))
                throw std::runtime_error("GGUF sparse compressor overlap shift failed");
        }
        if (use_sparse_compressor && w.indexer != nullptr && w.indexer->present && w.indexer->comp.present &&
            w.sparse_state->indexer_comp_kv != nullptr &&
            w.sparse_state->indexer_comp_score != nullptr &&
            w.sparse_state->indexer_kv_cache != nullptr &&
            s.d_indexer_comp_kv != nullptr && s.d_indexer_comp_score != nullptr) {
            const auto& ic = w.indexer->comp;
            if (!bf16_dual_matvec_bf16_x_cuda(s.d_compressor_input_bf16, ic.wkv, ic.wgate,
                                              s.d_indexer_comp_kv, s.d_indexer_comp_score,
                                              ic.cols, d.dim))
                throw std::runtime_error("GGUF sparse indexer compressor matvec failed");
            const int idx_offset = position % ic.ratio;
            const float* idx_ape = ic.ape + static_cast<size_t>(idx_offset) * ic.cols;
            const int idx_write_slot = ic.overlap ? ic.ratio + idx_offset : idx_offset;
            if (!compressor_update_state_cuda(s.d_indexer_comp_kv, s.d_indexer_comp_score,
                                              idx_ape,
                                              w.sparse_state->indexer_comp_kv,
                                              w.sparse_state->indexer_comp_score,
                                              idx_offset, idx_write_slot, ic.state_cols))
                throw std::runtime_error("GGUF sparse indexer compressor state update failed");
            if ((position + 1) % ic.ratio == 0) {
                float* d_idx_slot = w.sparse_state->indexer_kv_cache +
                    static_cast<size_t>(position / ic.ratio) * w.indexer->head_dim;
                if (!compressor_pool_cuda(w.sparse_state->indexer_comp_kv,
                                          w.sparse_state->indexer_comp_score,
                                          d_idx_slot, ic.ratio, w.indexer->head_dim,
                                          ic.state_cols, ic.overlap))
                    throw std::runtime_error("GGUF sparse indexer compressor pool failed");
                if (!rmsnorm_bf16_gamma_cuda(d_idx_slot, ic.norm, d_idx_slot,
                                             w.indexer->head_dim, 1e-6f))
                    throw std::runtime_error("GGUF sparse indexer compressed kv norm failed");
                const float comp_theta = d.compress_rope_theta > 0.0f ? d.compress_rope_theta : d.rope_theta;
                if (!head_rmsnorm_rope_cuda(d_idx_slot, /*heads=*/1, w.indexer->head_dim,
                                            d.rope_dim, position + 1 - ic.ratio,
                                            comp_theta, false, 0.0f))
                    throw std::runtime_error("GGUF sparse indexer compressed kv rope failed");
                if (!hadamard128_rows_cuda(d_idx_slot, d_idx_slot, 1))
                    throw std::runtime_error("GGUF sparse indexer compressed kv hadamard failed");
                if (!fp4_fake_quant128_rows_cuda(d_idx_slot, 1))
                    throw std::runtime_error("GGUF sparse indexer compressed kv fp4 failed");
                if (ic.overlap && !compressor_shift_overlap_state_cuda(
                        w.sparse_state->indexer_comp_kv,
                        w.sparse_state->indexer_comp_score,
                        ic.ratio, ic.state_cols))
                    throw std::runtime_error("GGUF sparse indexer compressor overlap shift failed");
            }
        }
    }
    if (!q8_0_matvec_cuda(s.d_x_normed, w.d_wq_a, s.d_q_a, d.q_a_dim, d.dim))
        throw std::runtime_error("wq_a failed");
    if (!rmsnorm_bf16_gamma_cuda(s.d_q_a, w.d_q_gamma, s.d_q_normed, d.q_a_dim, 1e-6f))
        throw std::runtime_error("q_norm failed");
    if (use_sparse_attention && w.indexer != nullptr && w.indexer->present &&
        s.d_index_q != nullptr) {
        const int index_q_dim = w.indexer->heads * w.indexer->head_dim;
        if (!bf16_matvec_cuda(s.d_q_normed, w.indexer->wq_b, s.d_index_q,
                              index_q_dim, d.q_a_dim))
            throw std::runtime_error("GGUF sparse indexer q matvec failed");
        if (!head_rmsnorm_rope_cuda(s.d_index_q, w.indexer->heads,
                                    w.indexer->head_dim, d.rope_dim, position,
                                    d.rope_theta, false, 0.0f))
            throw std::runtime_error("GGUF sparse indexer q rope failed");
        if (!hadamard128_rows_cuda(s.d_index_q, s.d_index_q, w.indexer->heads))
            throw std::runtime_error("GGUF sparse indexer q hadamard failed");
        if (!fp4_fake_quant128_rows_cuda(s.d_index_q, w.indexer->heads))
            throw std::runtime_error("GGUF sparse indexer q fp4 failed");
    }
    if (!q8_0_matvec_cuda(s.d_q_normed, w.d_wq_b, s.d_q, d.q_full, d.q_a_dim))
        throw std::runtime_error("wq_b failed");
    if (!head_rmsnorm_rope_cuda(s.d_q, d.heads, d.head_dim, d.rope_dim, position,
                                d.rope_theta, false, 1e-6f))
        throw std::runtime_error("q head rope failed");
    if (!q8_0_matvec_cuda(s.d_x_normed, w.d_wkv, s.d_kv_a, d.kv_dim, d.dim))
        throw std::runtime_error("wkv failed");
    if (!rmsnorm_bf16_gamma_cuda(s.d_kv_a, w.d_kv_gamma, s.d_kv, d.kv_dim, 1e-6f))
        throw std::runtime_error("kv_norm failed");
    if (!head_rmsnorm_rope_cuda(s.d_kv, /*heads=*/1, d.kv_dim, d.rope_dim, position,
                                d.rope_theta, false, 0.0f))
        throw std::runtime_error("kv head rope failed");
    const float scale = 1.0f / std::sqrt(static_cast<float>(d.head_dim));
    s.sparse_kv_index_count = 0;
    s.sparse_attn_used = false;
    if (s.d_kv_cache != nullptr) {
        if (use_sparse_attention) {
            const int window = w.sparse_window_size;
            if (position < 0 || s.cache_capacity <= window)
                throw std::runtime_error("invalid GGUF sparse KV cache capacity");
            const int write_pos = position % window;
            check_cuda(cudaMemcpy(s.d_kv_cache + static_cast<size_t>(write_pos) * d.kv_dim,
                                  s.d_kv, static_cast<size_t>(d.kv_dim) * sizeof(float),
                                  cudaMemcpyDeviceToDevice), "write sparse kv to cache");
            const int window_len = std::min(position + 1, window);
            const int window_start = std::max(0, position - window_len + 1);
            if (!build_decode_kv_indices_cuda(s.d_kv_indices, window_start, window_len,
                                              window, /*compressed_count=*/0,
                                              /*compressed_offset=*/window))
                throw std::runtime_error("build GGUF sparse window indices failed");
            const int compressed_ready = (position + 1) / w.compress_ratio;
            const int compressed_cap = std::max(0, s.cache_capacity - window);
            int extra_count = 0;
            if (w.indexer != nullptr && w.indexer->present && compressed_ready > 0 &&
                w.sparse_state->indexer_kv_cache != nullptr && s.d_index_scores != nullptr &&
                s.d_index_q != nullptr && w.index_topk > 0) {
                const int available = std::min(compressed_ready, compressed_cap);
                const int keep = std::min(available, w.index_topk);
                if (!indexer_select_topk_cuda(s.d_index_q,
                                              w.sparse_state->indexer_kv_cache,
                                              w.indexer->weights_proj,
                                              s.d_x,
                                              s.d_index_scores,
                                              s.d_kv_indices + window_len,
                                              available,
                                              keep,
                                              w.indexer->heads,
                                              w.indexer->head_dim,
                                              d.dim,
                                              window))
                    throw std::runtime_error("GGUF sparse indexer topk failed");
                extra_count = keep;
            } else {
                extra_count = std::min(compressed_ready, compressed_cap);
                if (extra_count > 0 && !build_decode_kv_indices_cuda(
                        s.d_kv_indices + window_len, /*window_start=*/0, /*window_len=*/0,
                        window, extra_count, window))
                    throw std::runtime_error("build GGUF sparse compressed indices failed");
            }
            const int index_count = window_len + extra_count;
            s.sparse_kv_index_count = index_count;
            s.sparse_attn_used = true;
            if (!indexed_cached_single_token_attention_cuda(s.d_q, s.d_kv_cache, s.d_kv_indices,
                                                            w.d_attn_sink, s.d_attn_value,
                                                            d.heads, d.head_dim,
                                                            index_count, scale))
                throw std::runtime_error("GGUF sparse indexed attention failed");
        } else {
            if (position < 0 || position >= s.cache_capacity)
                throw std::runtime_error("position out of KV cache capacity");
            // Write current step's MLA latent K/V into the per-layer cache at slot
            // `position`, then run cached attention over [0..position] + attn_sink.
            check_cuda(cudaMemcpy(s.d_kv_cache + static_cast<size_t>(position) * d.kv_dim,
                                  s.d_kv, static_cast<size_t>(d.kv_dim) * sizeof(float),
                                  cudaMemcpyDeviceToDevice), "write kv to cache");
            if (s.d_kv_indices != nullptr && s.kv_index_count > 0) {
                if (!indexed_cached_single_token_attention_cuda(s.d_q, s.d_kv_cache, s.d_kv_indices,
                                                                w.d_attn_sink, s.d_attn_value,
                                                                d.heads, d.head_dim,
                                                                s.kv_index_count, scale))
                    throw std::runtime_error("indexed cached attention failed");
            } else if (s.d_attn_weight_scratch != nullptr) {
                if (!cached_single_token_attention_workspace_cuda(s.d_q, s.d_kv_cache, w.d_attn_sink,
                                                                  s.d_attn_weight_scratch,
                                                                  s.d_attn_value, d.heads, d.head_dim,
                                                                  position + 1, scale))
                    throw std::runtime_error("cached attention workspace failed");
            } else if (!cached_single_token_attention_cuda(s.d_q, s.d_kv_cache, w.d_attn_sink,
                                                           s.d_attn_value, d.heads, d.head_dim,
                                                           position + 1, scale))
                throw std::runtime_error("cached attention failed");
        }
    } else {
        if (!single_token_sparse_attention_cuda(s.d_q, s.d_kv, w.d_attn_sink,
                                                s.d_attn_value, d.heads, d.head_dim, scale))
            throw std::runtime_error("sparse_attention failed");
    }
    if (!head_rmsnorm_rope_cuda(s.d_attn_value, d.heads, d.head_dim, d.rope_dim,
                                position, d.rope_theta, /*inverse=*/true, 0.0f))
        throw std::runtime_error("inverse rope failed");
    const size_t group_w_bytes =
        static_cast<size_t>(d.o_lora_rank) *
        static_cast<size_t>((d.group_in_dim + 31) / 32) * 34;
    for (int g = 0; g < d.o_groups; ++g) {
        const float* group_x = s.d_attn_value + static_cast<size_t>(g) * d.group_in_dim;
        const uint8_t* group_w = w.d_wo_a + static_cast<size_t>(g) * group_w_bytes;
        float* group_y = s.d_attn_mid + static_cast<size_t>(g) * d.o_lora_rank;
        if (!q8_0_matvec_cuda(group_x, group_w, group_y, d.o_lora_rank, d.group_in_dim))
            throw std::runtime_error("wo_a group matvec failed");
    }
    if (!q8_0_matvec_cuda(s.d_attn_mid, w.d_wo_b, s.d_attn_out, d.dim, d.attn_mid))
        throw std::runtime_error("wo_b failed");
    gguf_all_reduce_sum_fp32_inplace(s.d_attn_out, d.dim, reduce_ctx, "GGUF attention all-reduce");
    // Attention residual.
    if (!vector_add_cuda(s.d_x_pre_attn, s.d_attn_out, s.d_x, d.dim))
        throw std::runtime_error("attn residual add failed");

    // ===== FFN: norm + shared expert + gate =====
    check_cuda(cudaMemcpy(s.d_x_pre_ffn, s.d_x, d.dim * sizeof(float),
                          cudaMemcpyDeviceToDevice), "save x_pre_ffn");
    if (!rmsnorm_bf16_gamma_cuda(s.d_x, w.d_ffn_gamma, s.d_x_normed_ffn, d.dim, 1e-6f))
        throw std::runtime_error("ffn_norm failed");

    // Shared expert (Q8_0 w1/w3 + silu_mul + Q8_0 w2).
    if (!q8_0_matvec_cuda(s.d_x_normed_ffn, w.d_shared_w1, s.d_shared_gate,
                          d.moe_inter, d.dim))
        throw std::runtime_error("shared w1 failed");
    if (!q8_0_matvec_cuda(s.d_x_normed_ffn, w.d_shared_w3, s.d_shared_up,
                          d.moe_inter, d.dim))
        throw std::runtime_error("shared w3 failed");
    if (!silu_mul_cuda(s.d_shared_gate, s.d_shared_up, s.d_shared_hidden, d.moe_inter))
        throw std::runtime_error("shared silu_mul failed");
    if (!q8_0_matvec_cuda(s.d_shared_hidden, w.d_shared_w2, s.d_shared_out,
                          d.dim, d.moe_inter))
        throw std::runtime_error("shared w2 failed");

    // Gate scoring → route weights + indices.
    if (w.is_hash) {
        const int table_topk = w.hash_table_topk > 0 ? w.hash_table_topk : d.topk;
        if (!gate_hash_bf16_cuda(s.d_x_normed_ffn, w.d_gate_w_bf16, w.d_tid2eid_i64,
                                 s.d_gate_scores_scratch, s.d_gate_indices,
                                 s.d_route_weights,
                                 w.hash_token, /*cols=*/d.dim,
                                 table_topk, /*topk=*/d.topk,
                                 d.route_scale))
            throw std::runtime_error("gate_hash_bf16_cuda failed");
    } else {
        if (!gate_topk_bf16_cuda_with_buffers(
                s.d_x_normed_ffn, w.d_gate_w_bf16, w.d_gate_bias_f32,
                s.d_gate_scores_scratch, s.d_gate_scored_scratch,
                s.d_gate_indices, s.d_route_weights,
                d.n_experts, d.dim, d.topk, d.route_scale))
            throw std::runtime_error("gate_topk_bf16_cuda failed");
    }
}

// Phase 2: routed Q2 MoE compute + FFN residual.
// Caller must have staged the top-k routed experts (matching s.d_gate_indices)
// into w.d_routed_w1/w2/w3 before this call.
void gguf_layer_forward_moe(const GgufLayerDeviceWeights& w,
                            GgufLayerScratch& s,
                            const GgufLayerDims& d,
                            const GgufReduceContext* reduce_ctx) {
    const int routes = d.topk;

    check_cuda(cudaMemset(s.d_moe_out, 0, d.dim * sizeof(float)), "zero moe_out");
    const bool q2_recipe =
        w.routed_w1_dtype == DType::IQ2_XXS &&
        w.routed_w3_dtype == DType::IQ2_XXS &&
        w.routed_w2_dtype == DType::Q2_K;
    const bool iq1_all_recipe =
        w.routed_w1_dtype == DType::IQ1_M &&
        w.routed_w3_dtype == DType::IQ1_M &&
        w.routed_w2_dtype == DType::IQ1_M;
    const bool iq1_w13_q2_w2_recipe =
        w.routed_w1_dtype == DType::IQ1_M &&
        w.routed_w3_dtype == DType::IQ1_M &&
        w.routed_w2_dtype == DType::Q2_K;

    const int routed_n_experts = w.routed_n_experts > 0 ? w.routed_n_experts : routes;
    const bool iq1_fused_w13_swiglu = env_int_or_default("DSV4_GGUF_IQ1_FUSED_W13_SWIGLU", 1) != 0;

    if (q2_recipe) {
        if (!q2_quantize_x_q8_1_cuda(s.d_x_normed_ffn, s.d_x_q, s.d_x_scale,
                                      /*routes=*/1, d.dim))
            throw std::runtime_error("q2_quantize_x_q8_1 failed");
        if (!q2_moe_single_w13_iq2_xxs_cuda(s.d_x_q, s.d_x_scale, s.d_route_slots,
                                             w.d_routed_w1, w.d_routed_w3,
                                             s.d_route_gate, s.d_route_up,
                                             routes, routed_n_experts,
                                             d.dim, d.moe_inter))
            throw std::runtime_error("q2 w13 iq2_xxs failed");
        if (!q2_route_swiglu_quantize_hidden_q8_1_cuda(
                s.d_route_gate, s.d_route_up, s.d_route_weights,
                s.d_route_hidden_q, s.d_route_hidden_scale,
                routes, d.moe_inter, d.swiglu_limit))
            throw std::runtime_error("q2 swiglu + quantize failed");
        if (!q2_moe_single_w2_q2k_cuda(s.d_route_hidden_q, s.d_route_hidden_scale,
                                        s.d_route_slots, w.d_routed_w2, s.d_moe_out,
                                        routes, routed_n_experts,
                                        d.dim, d.moe_inter))
            throw std::runtime_error("q2 w2 q2k failed");
    } else if (iq1_all_recipe) {
        if (iq1_fused_w13_swiglu) {
            if (!iq1_moe_single_w13_swiglu_cuda(
                    s.d_x_normed_ffn, s.d_route_slots, s.d_route_weights,
                    w.d_routed_w1, w.d_routed_w3, s.d_route_hidden,
                    routes, routed_n_experts, d.dim, d.moe_inter, d.swiglu_limit))
                throw std::runtime_error("iq1 fused w13+swiglu failed");
        } else {
            if (!iq1_moe_single_w13_cuda(s.d_x_normed_ffn, s.d_route_slots,
                                          w.d_routed_w1, w.d_routed_w3,
                                          s.d_route_gate, s.d_route_up,
                                          routes, routed_n_experts,
                                          d.dim, d.moe_inter))
                throw std::runtime_error("iq1 w13 failed");
            if (!iq1_route_swiglu_cuda(s.d_route_gate, s.d_route_up, s.d_route_weights,
                                        s.d_route_hidden, routes, d.moe_inter,
                                        d.swiglu_limit))
                throw std::runtime_error("iq1 swiglu failed");
        }
        const bool iq1_w2_reduce = env_int_or_default("DSV4_GGUF_IQ1_W2_REDUCE", 0) != 0;
        const bool w2_ok = iq1_w2_reduce
            ? iq1_moe_single_w2_reduce_cuda(s.d_route_hidden, s.d_route_slots,
                                            w.d_routed_w2, s.d_moe_out,
                                            routes, routed_n_experts,
                                            d.dim, d.moe_inter)
            : iq1_moe_single_w2_cuda(s.d_route_hidden, s.d_route_slots,
                                      w.d_routed_w2, s.d_moe_out,
                                      routes, routed_n_experts,
                                      d.dim, d.moe_inter);
        if (!w2_ok)
            throw std::runtime_error("iq1 w2 failed");
    } else if (iq1_w13_q2_w2_recipe) {
        // Dynamic IQ1_M recipe early layers: W1/W3 are IQ1_M, W2 remains Q2_K.
        if (!iq1_moe_single_w13_cuda(s.d_x_normed_ffn, s.d_route_slots,
                                      w.d_routed_w1, w.d_routed_w3,
                                      s.d_route_gate, s.d_route_up,
                                      routes, routed_n_experts,
                                      d.dim, d.moe_inter))
            throw std::runtime_error("iq1 w13 failed");
        if (!q2_route_swiglu_quantize_hidden_q8_1_cuda(
                s.d_route_gate, s.d_route_up, s.d_route_weights,
                s.d_route_hidden_q, s.d_route_hidden_scale,
                routes, d.moe_inter, d.swiglu_limit))
            throw std::runtime_error("iq1/q2 swiglu + quantize failed");
        if (!q2_moe_single_w2_q2k_cuda(s.d_route_hidden_q, s.d_route_hidden_scale,
                                        s.d_route_slots, w.d_routed_w2, s.d_moe_out,
                                        routes, routed_n_experts,
                                        d.dim, d.moe_inter))
            throw std::runtime_error("iq1/q2 w2 q2k failed");
    } else {
        throw std::runtime_error("unsupported routed GGUF MoE dtype recipe");
    }

    gguf_all_reduce_sum_fp32_inplace(s.d_moe_out, d.dim, reduce_ctx, "GGUF MoE all-reduce");

    if (!vector_add_cuda(s.d_shared_out, s.d_moe_out, s.d_ffn_combined, d.dim))
        throw std::runtime_error("ffn combined add failed");
    if (!vector_add_cuda(s.d_x_pre_ffn, s.d_ffn_combined, s.d_x, d.dim))
        throw std::runtime_error("ffn residual add failed");
}

// Convenience wrapper for the legacy single-layer smoke that pre-stages
// experts (the layer-0 smoke uses this since experts are derived from
// tid2eid before any compute runs).
void gguf_layer_forward(const GgufLayerDeviceWeights& w,
                        GgufLayerScratch& s,
                        const GgufLayerDims& d,
                        int /*token*/,
                        int position) {
    gguf_layer_forward_attn_to_gate(w, s, d, position, nullptr);
    gguf_layer_forward_moe(w, s, d, nullptr);
}

}  // namespace

GgufAttnNormWqaResult run_gguf_attn_norm_wq_a_smoke(const std::string& ckpt_path, int token) {
    if (!is_gguf_path(ckpt_path)) {
        throw std::runtime_error("run_gguf_attn_norm_wq_a_smoke: not a GGUF path: " + ckpt_path);
    }
    if (token < 0) {
        throw std::runtime_error("run_gguf_attn_norm_wq_a_smoke: token must be >= 0");
    }
    GgufForwardContext ctx(ckpt_path);
    WeightSource& ws = *ctx.weight_source;

    WeightView embed = ws.require("embed.weight");
    WeightView attn_norm = ws.require("layers.0.attn_norm.weight");
    WeightView wq_a = ws.require("layers.0.attn.wq_a.weight");

    if (embed.dtype != DType::F16 || embed.shape.size() != 2) {
        throw std::runtime_error("embed.weight is not 2D F16");
    }
    if (attn_norm.dtype != DType::F32 || attn_norm.shape.size() != 1) {
        throw std::runtime_error("layers.0.attn_norm.weight is not 1D F32");
    }
    if (wq_a.dtype != DType::Q8_0 || wq_a.shape.size() != 2) {
        throw std::runtime_error("layers.0.attn.wq_a.weight is not 2D Q8_0");
    }

    const int dim = static_cast<int>(embed.shape[0]);
    const int vocab = static_cast<int>(embed.shape[1]);
    if (token >= vocab) throw std::runtime_error("token id out of vocab range");
    if (static_cast<int>(attn_norm.shape[0]) != dim) {
        throw std::runtime_error("attn_norm shape mismatch with dim");
    }
    // GGUF wq_a shape [cols=dim, rows=q_a_dim] with cols fastest varying.
    if (static_cast<int>(wq_a.shape[0]) != dim) {
        throw std::runtime_error("wq_a inner dim != dim");
    }
    const int q_a_dim = static_cast<int>(wq_a.shape[1]);

    // 1. Upload one token's F16 embedding row and dequantize to fp32.
    const uint16_t* host_embed_row =
        reinterpret_cast<const uint16_t*>(embed.data) +
        static_cast<size_t>(token) * dim;
    uint16_t* d_row_f16 = nullptr;
    float* d_x = nullptr;
    check_cuda(cudaMalloc(&d_row_f16, dim * sizeof(uint16_t)), "alloc d_row_f16");
    check_cuda(cudaMalloc(&d_x, dim * sizeof(float)), "alloc d_x");
    check_cuda(cudaMemcpy(d_row_f16, host_embed_row, dim * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy embed row");
    if (!f16_row_to_float_cuda(d_row_f16, d_x, /*row=*/0, dim)) {
        throw std::runtime_error("f16_row_to_float_cuda failed");
    }

    // 2. Convert attn_norm F32 gamma to BF16 (host-side) and upload.
    auto bf16 = f32_to_bf16_host(reinterpret_cast<const float*>(attn_norm.data), dim);
    uint16_t* d_attn_gamma = nullptr;
    check_cuda(cudaMalloc(&d_attn_gamma, dim * sizeof(uint16_t)), "alloc d_attn_gamma");
    check_cuda(cudaMemcpy(d_attn_gamma, bf16.data(), dim * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy attn_norm gamma");

    // 3. RMSNorm.
    float* d_x_normed = nullptr;
    check_cuda(cudaMalloc(&d_x_normed, dim * sizeof(float)), "alloc d_x_normed");
    if (!rmsnorm_bf16_gamma_cuda(d_x, d_attn_gamma, d_x_normed, dim, 1e-6f)) {
        throw std::runtime_error("rmsnorm_bf16_gamma_cuda failed");
    }

    // 4. Upload wq_a Q8_0 blocks and run matvec.
    uint8_t* d_wq_a = nullptr;
    check_cuda(cudaMalloc(&d_wq_a, wq_a.nbytes), "alloc d_wq_a");
    check_cuda(cudaMemcpy(d_wq_a, wq_a.data, wq_a.nbytes, cudaMemcpyHostToDevice),
               "copy wq_a");
    float* d_q_a = nullptr;
    check_cuda(cudaMalloc(&d_q_a, q_a_dim * sizeof(float)), "alloc d_q_a");
    if (!q8_0_matvec_cuda(d_x_normed, d_wq_a, d_q_a, q_a_dim, dim)) {
        throw std::runtime_error("q8_0_matvec_cuda failed");
    }
    check_cuda(cudaDeviceSynchronize(), "sync after wq_a");

    // 5. Collect diagnostics.
    GgufAttnNormWqaResult r;
    r.dim = dim;
    r.q_a_dim = q_a_dim;
    r.embed_rms = device_vector_rms(d_x, dim);
    r.normed_rms = device_vector_rms(d_x_normed, dim);
    r.q_a_rms = device_vector_rms(d_q_a, q_a_dim);
    std::vector<float> first(4);
    check_cuda(cudaMemcpy(first.data(), d_q_a, 4 * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy q_a first");
    for (int i = 0; i < 4; ++i) r.q_a_first[i] = first[i];

    cudaFree(d_row_f16);
    cudaFree(d_x);
    cudaFree(d_attn_gamma);
    cudaFree(d_x_normed);
    cudaFree(d_wq_a);
    cudaFree(d_q_a);
    return r;
}

GgufAttnQPathResult run_gguf_attn_q_path_smoke(const std::string& ckpt_path,
                                                int token,
                                                int position) {
    if (!is_gguf_path(ckpt_path)) {
        throw std::runtime_error("run_gguf_attn_q_path_smoke: not a GGUF path: " + ckpt_path);
    }
    if (token < 0) throw std::runtime_error("token must be >= 0");
    if (position < 0) throw std::runtime_error("position must be >= 0");

    GgufForwardContext ctx(ckpt_path);
    WeightSource& ws = *ctx.weight_source;

    WeightView embed = ws.require("embed.weight");
    WeightView attn_norm = ws.require("layers.0.attn_norm.weight");
    WeightView wq_a = ws.require("layers.0.attn.wq_a.weight");
    WeightView q_norm = ws.require("layers.0.attn.q_norm.weight");
    WeightView wq_b = ws.require("layers.0.attn.wq_b.weight");

    if (embed.dtype != DType::F16) throw std::runtime_error("embed dtype");
    if (attn_norm.dtype != DType::F32) throw std::runtime_error("attn_norm dtype");
    if (q_norm.dtype != DType::F32) throw std::runtime_error("q_norm dtype");
    if (wq_a.dtype != DType::Q8_0) throw std::runtime_error("wq_a dtype");
    if (wq_b.dtype != DType::Q8_0) throw std::runtime_error("wq_b dtype");

    const int dim = static_cast<int>(embed.shape[0]);
    const int q_a_dim = static_cast<int>(wq_a.shape[1]);
    // wq_b is stored [q_a_dim, heads*head_dim] in GGUF (cols fastest).
    if (static_cast<int>(wq_b.shape[0]) != q_a_dim) {
        throw std::runtime_error("wq_b inner dim != q_a_dim");
    }
    const int q_full = static_cast<int>(wq_b.shape[1]);
    const int heads = static_cast<int>(ctx.config.n_heads);
    if (heads <= 0) throw std::runtime_error("n_heads not set");
    if (q_full % heads != 0) {
        throw std::runtime_error("wq_b output cols not divisible by n_heads");
    }
    const int head_dim = q_full / heads;
    const int rope_dim = static_cast<int>(ctx.config.rope_dim);
    if (rope_dim <= 0 || rope_dim > head_dim) {
        throw std::runtime_error("rope_dim out of range");
    }

    // ----- Embed -----
    const uint16_t* host_row =
        reinterpret_cast<const uint16_t*>(embed.data) +
        static_cast<size_t>(token) * dim;
    uint16_t* d_row_f16 = nullptr;
    float* d_x = nullptr;
    check_cuda(cudaMalloc(&d_row_f16, dim * sizeof(uint16_t)), "alloc d_row_f16");
    check_cuda(cudaMalloc(&d_x, dim * sizeof(float)), "alloc d_x");
    check_cuda(cudaMemcpy(d_row_f16, host_row, dim * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy embed row");
    if (!f16_row_to_float_cuda(d_row_f16, d_x, /*row=*/0, dim)) {
        throw std::runtime_error("f16_row_to_float_cuda failed");
    }

    // ----- attn_norm gamma -----
    auto attn_bf16 = f32_to_bf16_host(reinterpret_cast<const float*>(attn_norm.data), dim);
    uint16_t* d_attn_gamma = nullptr;
    check_cuda(cudaMalloc(&d_attn_gamma, dim * sizeof(uint16_t)), "alloc d_attn_gamma");
    check_cuda(cudaMemcpy(d_attn_gamma, attn_bf16.data(), dim * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy attn_gamma");
    float* d_x_normed = nullptr;
    check_cuda(cudaMalloc(&d_x_normed, dim * sizeof(float)), "alloc d_x_normed");
    if (!rmsnorm_bf16_gamma_cuda(d_x, d_attn_gamma, d_x_normed, dim, 1e-6f)) {
        throw std::runtime_error("rmsnorm attn failed");
    }

    // ----- wq_a -----
    uint8_t* d_wq_a = nullptr;
    check_cuda(cudaMalloc(&d_wq_a, wq_a.nbytes), "alloc d_wq_a");
    check_cuda(cudaMemcpy(d_wq_a, wq_a.data, wq_a.nbytes, cudaMemcpyHostToDevice), "copy wq_a");
    float* d_q_a = nullptr;
    check_cuda(cudaMalloc(&d_q_a, q_a_dim * sizeof(float)), "alloc d_q_a");
    if (!q8_0_matvec_cuda(d_x_normed, d_wq_a, d_q_a, q_a_dim, dim)) {
        throw std::runtime_error("q8_0 wq_a failed");
    }

    // ----- q_norm gamma -----
    auto q_bf16 = f32_to_bf16_host(reinterpret_cast<const float*>(q_norm.data), q_a_dim);
    uint16_t* d_q_gamma = nullptr;
    check_cuda(cudaMalloc(&d_q_gamma, q_a_dim * sizeof(uint16_t)), "alloc d_q_gamma");
    check_cuda(cudaMemcpy(d_q_gamma, q_bf16.data(), q_a_dim * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy q_gamma");
    float* d_q_normed = nullptr;
    check_cuda(cudaMalloc(&d_q_normed, q_a_dim * sizeof(float)), "alloc d_q_normed");
    if (!rmsnorm_bf16_gamma_cuda(d_q_a, d_q_gamma, d_q_normed, q_a_dim, 1e-6f)) {
        throw std::runtime_error("rmsnorm q failed");
    }

    // ----- wq_b -> q[heads*head_dim] -----
    uint8_t* d_wq_b = nullptr;
    check_cuda(cudaMalloc(&d_wq_b, wq_b.nbytes), "alloc d_wq_b");
    check_cuda(cudaMemcpy(d_wq_b, wq_b.data, wq_b.nbytes, cudaMemcpyHostToDevice), "copy wq_b");
    float* d_q = nullptr;
    check_cuda(cudaMalloc(&d_q, q_full * sizeof(float)), "alloc d_q");
    if (!q8_0_matvec_cuda(d_q_normed, d_wq_b, d_q, q_full, q_a_dim)) {
        throw std::runtime_error("q8_0 wq_b failed");
    }
    check_cuda(cudaDeviceSynchronize(), "sync after wq_b");

    GgufAttnQPathResult r;
    r.dim = dim;
    r.q_a_dim = q_a_dim;
    r.heads = heads;
    r.head_dim = head_dim;
    r.rope_dim = rope_dim;
    r.q_normed_rms = device_vector_rms(d_q_normed, q_a_dim);
    r.q_pre_rope_rms = device_vector_rms(d_q, q_full);

    // ----- head-wise RMSNorm + RoPE -----
    if (!head_rmsnorm_rope_cuda(d_q, heads, head_dim, rope_dim,
                                position,
                                static_cast<float>(ctx.config.rope_theta),
                                false, 1e-6f)) {
        throw std::runtime_error("head_rmsnorm_rope failed");
    }
    check_cuda(cudaDeviceSynchronize(), "sync after head rope");

    r.q_post_rope_rms = device_vector_rms(d_q, q_full);
    std::vector<float> first(4);
    check_cuda(cudaMemcpy(first.data(), d_q, 4 * sizeof(float), cudaMemcpyDeviceToHost),
               "copy q first");
    for (int i = 0; i < 4; ++i) r.q_first[i] = first[i];

    cudaFree(d_row_f16);
    cudaFree(d_x);
    cudaFree(d_attn_gamma);
    cudaFree(d_x_normed);
    cudaFree(d_wq_a);
    cudaFree(d_q_a);
    cudaFree(d_q_gamma);
    cudaFree(d_q_normed);
    cudaFree(d_wq_b);
    cudaFree(d_q);
    return r;
}

GgufAttnKvPathResult run_gguf_attn_kv_path_smoke(const std::string& ckpt_path,
                                                  int token,
                                                  int position) {
    if (!is_gguf_path(ckpt_path)) {
        throw std::runtime_error("run_gguf_attn_kv_path_smoke: not a GGUF path: " + ckpt_path);
    }
    if (token < 0) throw std::runtime_error("token must be >= 0");
    if (position < 0) throw std::runtime_error("position must be >= 0");

    GgufForwardContext ctx(ckpt_path);
    WeightSource& ws = *ctx.weight_source;

    WeightView embed = ws.require("embed.weight");
    WeightView attn_norm = ws.require("layers.0.attn_norm.weight");
    WeightView wkv = ws.require("layers.0.attn.wkv.weight");
    WeightView kv_norm = ws.require("layers.0.attn.kv_norm.weight");

    if (embed.dtype != DType::F16) throw std::runtime_error("embed dtype");
    if (attn_norm.dtype != DType::F32) throw std::runtime_error("attn_norm dtype");
    if (kv_norm.dtype != DType::F32) throw std::runtime_error("kv_norm dtype");
    if (wkv.dtype != DType::Q8_0) throw std::runtime_error("wkv dtype");

    const int dim = static_cast<int>(embed.shape[0]);
    // wkv stored [dim, kv_dim] with cols fastest varying.
    if (static_cast<int>(wkv.shape[0]) != dim) {
        throw std::runtime_error("wkv inner dim != dim");
    }
    const int kv_dim = static_cast<int>(wkv.shape[1]);
    if (static_cast<int>(kv_norm.shape[0]) != kv_dim) {
        throw std::runtime_error("kv_norm gamma length != kv_dim");
    }
    const int rope_dim = static_cast<int>(ctx.config.rope_dim);
    if (rope_dim <= 0 || rope_dim > kv_dim) {
        throw std::runtime_error("rope_dim out of range");
    }

    // ----- Embed -----
    const uint16_t* host_row =
        reinterpret_cast<const uint16_t*>(embed.data) +
        static_cast<size_t>(token) * dim;
    uint16_t* d_row_f16 = nullptr;
    float* d_x = nullptr;
    check_cuda(cudaMalloc(&d_row_f16, dim * sizeof(uint16_t)), "alloc d_row_f16");
    check_cuda(cudaMalloc(&d_x, dim * sizeof(float)), "alloc d_x");
    check_cuda(cudaMemcpy(d_row_f16, host_row, dim * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy embed row");
    if (!f16_row_to_float_cuda(d_row_f16, d_x, /*row=*/0, dim)) {
        throw std::runtime_error("f16_row_to_float_cuda failed");
    }

    // ----- attn_norm gamma -----
    auto attn_bf16 = f32_to_bf16_host(reinterpret_cast<const float*>(attn_norm.data), dim);
    uint16_t* d_attn_gamma = nullptr;
    check_cuda(cudaMalloc(&d_attn_gamma, dim * sizeof(uint16_t)), "alloc d_attn_gamma");
    check_cuda(cudaMemcpy(d_attn_gamma, attn_bf16.data(), dim * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy attn_gamma");
    float* d_x_normed = nullptr;
    check_cuda(cudaMalloc(&d_x_normed, dim * sizeof(float)), "alloc d_x_normed");
    if (!rmsnorm_bf16_gamma_cuda(d_x, d_attn_gamma, d_x_normed, dim, 1e-6f)) {
        throw std::runtime_error("rmsnorm attn failed");
    }

    // ----- wkv: dim -> kv_dim -----
    uint8_t* d_wkv = nullptr;
    check_cuda(cudaMalloc(&d_wkv, wkv.nbytes), "alloc d_wkv");
    check_cuda(cudaMemcpy(d_wkv, wkv.data, wkv.nbytes, cudaMemcpyHostToDevice), "copy wkv");
    float* d_kv_a = nullptr;
    check_cuda(cudaMalloc(&d_kv_a, kv_dim * sizeof(float)), "alloc d_kv_a");
    if (!q8_0_matvec_cuda(d_x_normed, d_wkv, d_kv_a, kv_dim, dim)) {
        throw std::runtime_error("q8_0 wkv failed");
    }

    // ----- kv_norm RMSNorm (full kv_dim gamma) -----
    auto kv_bf16 = f32_to_bf16_host(reinterpret_cast<const float*>(kv_norm.data), kv_dim);
    uint16_t* d_kv_gamma = nullptr;
    check_cuda(cudaMalloc(&d_kv_gamma, kv_dim * sizeof(uint16_t)), "alloc d_kv_gamma");
    check_cuda(cudaMemcpy(d_kv_gamma, kv_bf16.data(), kv_dim * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy kv_gamma");
    float* d_kv = nullptr;
    check_cuda(cudaMalloc(&d_kv, kv_dim * sizeof(float)), "alloc d_kv");
    if (!rmsnorm_bf16_gamma_cuda(d_kv_a, d_kv_gamma, d_kv, kv_dim, 1e-6f)) {
        throw std::runtime_error("rmsnorm kv failed");
    }
    check_cuda(cudaDeviceSynchronize(), "sync after kv rmsnorm");

    GgufAttnKvPathResult r;
    r.dim = dim;
    r.kv_dim = kv_dim;
    r.rope_dim = rope_dim;
    r.kv_a_rms = device_vector_rms(d_kv_a, kv_dim);
    r.kv_norm_rms = device_vector_rms(d_kv, kv_dim);

    // ----- RoPE on the last rope_dim elements (heads=1, head_dim=kv_dim) -----
    // Pass do_rmsnorm=false because rmsnorm was already done with gamma above.
    if (!head_rmsnorm_rope_cuda(d_kv, /*heads=*/1, kv_dim, rope_dim,
                                position,
                                static_cast<float>(ctx.config.rope_theta),
                                false, 0.0f)) {
        throw std::runtime_error("head_rmsnorm_rope kv failed");
    }
    check_cuda(cudaDeviceSynchronize(), "sync after kv rope");

    r.kv_post_rope_rms = device_vector_rms(d_kv, kv_dim);
    std::vector<float> first(4);
    check_cuda(cudaMemcpy(first.data(), d_kv, 4 * sizeof(float), cudaMemcpyDeviceToHost),
               "copy kv first");
    for (int i = 0; i < 4; ++i) r.kv_first[i] = first[i];

    cudaFree(d_row_f16);
    cudaFree(d_x);
    cudaFree(d_attn_gamma);
    cudaFree(d_x_normed);
    cudaFree(d_wkv);
    cudaFree(d_kv_a);
    cudaFree(d_kv_gamma);
    cudaFree(d_kv);
    return r;
}

GgufAttnFullResult run_gguf_attn_full_smoke(const std::string& ckpt_path,
                                             int token,
                                             int position) {
    if (!is_gguf_path(ckpt_path)) {
        throw std::runtime_error("run_gguf_attn_full_smoke: not a GGUF path: " + ckpt_path);
    }
    if (token < 0) throw std::runtime_error("token must be >= 0");
    if (position < 0) throw std::runtime_error("position must be >= 0");

    GgufForwardContext ctx(ckpt_path);
    WeightSource& ws = *ctx.weight_source;

    // ----- weights -----
    WeightView embed = ws.require("embed.weight");
    WeightView attn_norm = ws.require("layers.0.attn_norm.weight");
    WeightView wq_a = ws.require("layers.0.attn.wq_a.weight");
    WeightView q_norm = ws.require("layers.0.attn.q_norm.weight");
    WeightView wq_b = ws.require("layers.0.attn.wq_b.weight");
    WeightView wkv = ws.require("layers.0.attn.wkv.weight");
    WeightView kv_norm = ws.require("layers.0.attn.kv_norm.weight");
    WeightView attn_sink = ws.require("layers.0.attn.attn_sink");
    WeightView wo_a = ws.require("layers.0.attn.wo_a.weight");
    WeightView wo_b = ws.require("layers.0.attn.wo_b.weight");

    if (embed.dtype != DType::F16) throw std::runtime_error("embed dtype");
    if (attn_norm.dtype != DType::F32) throw std::runtime_error("attn_norm dtype");
    if (q_norm.dtype != DType::F32) throw std::runtime_error("q_norm dtype");
    if (kv_norm.dtype != DType::F32) throw std::runtime_error("kv_norm dtype");
    if (attn_sink.dtype != DType::F32) throw std::runtime_error("attn_sink dtype");
    if (wq_a.dtype != DType::Q8_0 || wq_b.dtype != DType::Q8_0 ||
        wkv.dtype != DType::Q8_0 || wo_a.dtype != DType::Q8_0 ||
        wo_b.dtype != DType::Q8_0) {
        throw std::runtime_error("attention projections must be Q8_0");
    }

    const int dim = static_cast<int>(embed.shape[0]);
    const int q_a_dim = static_cast<int>(wq_a.shape[1]);
    const int heads = static_cast<int>(ctx.config.n_heads);
    const int q_full = static_cast<int>(wq_b.shape[1]);
    if (q_full % heads != 0) throw std::runtime_error("wq_b cols % heads != 0");
    const int head_dim = q_full / heads;
    const int kv_dim = static_cast<int>(wkv.shape[1]);
    const int rope_dim = static_cast<int>(ctx.config.rope_dim);
    if (kv_dim != head_dim) {
        throw std::runtime_error("expected kv_dim == head_dim for single KV head");
    }
    const int o_groups = static_cast<int>(ctx.config.o_groups);
    const int o_lora_rank = static_cast<int>(ctx.config.o_lora_rank);
    if (o_groups <= 0 || o_lora_rank <= 0) {
        throw std::runtime_error("o_groups / o_lora_rank not set");
    }
    if (q_full % o_groups != 0) {
        throw std::runtime_error("q_full not divisible by o_groups");
    }
    const int group_in_dim = q_full / o_groups;            // 32768 / 8 = 4096
    const int attn_mid = o_groups * o_lora_rank;            // 8 * 1024 = 8192
    if (static_cast<int>(wo_a.shape[0]) != group_in_dim ||
        static_cast<int>(wo_a.shape[1]) != attn_mid) {
        throw std::runtime_error("wo_a shape mismatch with o_groups/o_lora_rank");
    }
    if (static_cast<int>(wo_b.shape[0]) != attn_mid ||
        static_cast<int>(wo_b.shape[1]) != dim) {
        throw std::runtime_error("wo_b shape mismatch");
    }
    if (static_cast<int>(attn_sink.shape[0]) != heads) {
        throw std::runtime_error("attn_sink length != heads");
    }

    // ----- upload all weights / norms -----
    auto upload_u8 = [](const void* src, size_t bytes) {
        uint8_t* d = nullptr;
        check_cuda(cudaMalloc(&d, bytes), "alloc");
        check_cuda(cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice), "copy");
        return d;
    };
    auto upload_f32 = [](const void* src, size_t n) {
        float* d = nullptr;
        check_cuda(cudaMalloc(&d, n * sizeof(float)), "alloc f32");
        check_cuda(cudaMemcpy(d, src, n * sizeof(float), cudaMemcpyHostToDevice), "copy f32");
        return d;
    };
    auto upload_bf16_from_f32 = [&](const void* src, size_t n) {
        auto bf = f32_to_bf16_host(reinterpret_cast<const float*>(src), n);
        uint16_t* d = nullptr;
        check_cuda(cudaMalloc(&d, n * sizeof(uint16_t)), "alloc bf16");
        check_cuda(cudaMemcpy(d, bf.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice),
                   "copy bf16");
        return d;
    };

    const uint16_t* host_row =
        reinterpret_cast<const uint16_t*>(embed.data) +
        static_cast<size_t>(token) * dim;
    uint16_t* d_row_f16 = nullptr;
    check_cuda(cudaMalloc(&d_row_f16, dim * sizeof(uint16_t)), "alloc d_row_f16");
    check_cuda(cudaMemcpy(d_row_f16, host_row, dim * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy embed row");

    uint16_t* d_attn_gamma = upload_bf16_from_f32(attn_norm.data, dim);
    uint16_t* d_q_gamma = upload_bf16_from_f32(q_norm.data, q_a_dim);
    uint16_t* d_kv_gamma = upload_bf16_from_f32(kv_norm.data, kv_dim);

    uint8_t* d_wq_a = upload_u8(wq_a.data, wq_a.nbytes);
    uint8_t* d_wq_b = upload_u8(wq_b.data, wq_b.nbytes);
    uint8_t* d_wkv = upload_u8(wkv.data, wkv.nbytes);
    uint8_t* d_wo_a = upload_u8(wo_a.data, wo_a.nbytes);
    uint8_t* d_wo_b = upload_u8(wo_b.data, wo_b.nbytes);
    float* d_attn_sink = upload_f32(attn_sink.data, heads);

    // ----- activation buffers -----
    float* d_x = nullptr;
    float* d_x_normed = nullptr;
    float* d_q_a = nullptr;
    float* d_q_normed = nullptr;
    float* d_q = nullptr;
    float* d_kv_a = nullptr;
    float* d_kv = nullptr;
    float* d_attn_value = nullptr;
    float* d_attn_mid = nullptr;
    float* d_attn_out = nullptr;
    check_cuda(cudaMalloc(&d_x, dim * sizeof(float)), "alloc d_x");
    check_cuda(cudaMalloc(&d_x_normed, dim * sizeof(float)), "alloc d_x_normed");
    check_cuda(cudaMalloc(&d_q_a, q_a_dim * sizeof(float)), "alloc d_q_a");
    check_cuda(cudaMalloc(&d_q_normed, q_a_dim * sizeof(float)), "alloc d_q_normed");
    check_cuda(cudaMalloc(&d_q, q_full * sizeof(float)), "alloc d_q");
    check_cuda(cudaMalloc(&d_kv_a, kv_dim * sizeof(float)), "alloc d_kv_a");
    check_cuda(cudaMalloc(&d_kv, kv_dim * sizeof(float)), "alloc d_kv");
    check_cuda(cudaMalloc(&d_attn_value, q_full * sizeof(float)), "alloc d_attn_value");
    check_cuda(cudaMalloc(&d_attn_mid, attn_mid * sizeof(float)), "alloc d_attn_mid");
    check_cuda(cudaMalloc(&d_attn_out, dim * sizeof(float)), "alloc d_attn_out");

    // ----- compute -----
    if (!f16_row_to_float_cuda(d_row_f16, d_x, /*row=*/0, dim)) {
        throw std::runtime_error("f16 embed failed");
    }
    if (!rmsnorm_bf16_gamma_cuda(d_x, d_attn_gamma, d_x_normed, dim, 1e-6f)) {
        throw std::runtime_error("attn_norm failed");
    }

    // Q path
    if (!q8_0_matvec_cuda(d_x_normed, d_wq_a, d_q_a, q_a_dim, dim)) {
        throw std::runtime_error("wq_a failed");
    }
    if (!rmsnorm_bf16_gamma_cuda(d_q_a, d_q_gamma, d_q_normed, q_a_dim, 1e-6f)) {
        throw std::runtime_error("q_norm failed");
    }
    if (!q8_0_matvec_cuda(d_q_normed, d_wq_b, d_q, q_full, q_a_dim)) {
        throw std::runtime_error("wq_b failed");
    }
    if (!head_rmsnorm_rope_cuda(d_q, heads, head_dim, rope_dim,
                                position,
                                static_cast<float>(ctx.config.rope_theta),
                                false, 1e-6f)) {
        throw std::runtime_error("q head rope failed");
    }

    // KV path
    if (!q8_0_matvec_cuda(d_x_normed, d_wkv, d_kv_a, kv_dim, dim)) {
        throw std::runtime_error("wkv failed");
    }
    if (!rmsnorm_bf16_gamma_cuda(d_kv_a, d_kv_gamma, d_kv, kv_dim, 1e-6f)) {
        throw std::runtime_error("kv_norm failed");
    }
    if (!head_rmsnorm_rope_cuda(d_kv, /*heads=*/1, kv_dim, rope_dim,
                                position,
                                static_cast<float>(ctx.config.rope_theta),
                                false, 0.0f)) {
        throw std::runtime_error("kv head rope failed");
    }
    check_cuda(cudaDeviceSynchronize(), "sync after q/kv");

    GgufAttnFullResult r;
    r.dim = dim;
    r.q_a_dim = q_a_dim;
    r.heads = heads;
    r.head_dim = head_dim;
    r.kv_dim = kv_dim;
    r.rope_dim = rope_dim;
    r.o_groups = o_groups;
    r.o_lora_rank = o_lora_rank;
    r.attn_mid = attn_mid;
    r.q_rms = device_vector_rms(d_q, q_full);
    r.kv_rms = device_vector_rms(d_kv, kv_dim);

    // Sparse single-token attention against the (single) current KV vector.
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if (!single_token_sparse_attention_cuda(d_q, d_kv, d_attn_sink, d_attn_value,
                                            heads, head_dim, scale)) {
        throw std::runtime_error("single_token_sparse_attention failed");
    }
    check_cuda(cudaDeviceSynchronize(), "sync after attn");
    r.attn_value_rms = device_vector_rms(d_attn_value, q_full);

    // Inverse RoPE on the rope-tail of each head (V was rope-rotated as part of
    // KV; undo it before output projection so subsequent layers see consistent
    // representations). do_rmsnorm=false (just inverse rope).
    if (!head_rmsnorm_rope_cuda(d_attn_value, heads, head_dim, rope_dim,
                                position,
                                static_cast<float>(ctx.config.rope_theta),
                                /*inverse=*/true, 0.0f)) {
        throw std::runtime_error("inverse rope failed");
    }
    check_cuda(cudaDeviceSynchronize(), "sync after inverse rope");
    r.attn_value_post_inv_rms = device_vector_rms(d_attn_value, q_full);

    // Grouped wo_a: per-group matvec [group_in_dim] -> [o_lora_rank].
    // wo_a Q8_0 is laid out [output=attn_mid, input=group_in_dim] row-major,
    // so group g occupies rows [g*o_lora_rank, (g+1)*o_lora_rank) which is a
    // contiguous byte range of o_lora_rank * (group_in_dim/32) * 34 bytes.
    const size_t group_w_bytes =
        static_cast<size_t>(o_lora_rank) *
        static_cast<size_t>((group_in_dim + 31) / 32) * 34;
    for (int g = 0; g < o_groups; ++g) {
        const float* group_x = d_attn_value + static_cast<size_t>(g) * group_in_dim;
        const uint8_t* group_w = d_wo_a + static_cast<size_t>(g) * group_w_bytes;
        float* group_y = d_attn_mid + static_cast<size_t>(g) * o_lora_rank;
        if (!q8_0_matvec_cuda(group_x, group_w, group_y, o_lora_rank, group_in_dim)) {
            throw std::runtime_error("wo_a group matvec failed");
        }
    }

    // wo_b: [attn_mid] -> [dim]
    if (!q8_0_matvec_cuda(d_attn_mid, d_wo_b, d_attn_out, dim, attn_mid)) {
        throw std::runtime_error("wo_b failed");
    }
    check_cuda(cudaDeviceSynchronize(), "sync after wo_b");

    r.attn_mid_rms = device_vector_rms(d_attn_mid, attn_mid);
    r.attn_out_rms = device_vector_rms(d_attn_out, dim);
    std::vector<float> first(4);
    check_cuda(cudaMemcpy(first.data(), d_attn_out, 4 * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy attn_out first");
    for (int i = 0; i < 4; ++i) r.attn_out_first[i] = first[i];

    cudaFree(d_row_f16);
    cudaFree(d_attn_gamma);
    cudaFree(d_q_gamma);
    cudaFree(d_kv_gamma);
    cudaFree(d_wq_a);
    cudaFree(d_wq_b);
    cudaFree(d_wkv);
    cudaFree(d_wo_a);
    cudaFree(d_wo_b);
    cudaFree(d_attn_sink);
    cudaFree(d_x);
    cudaFree(d_x_normed);
    cudaFree(d_q_a);
    cudaFree(d_q_normed);
    cudaFree(d_q);
    cudaFree(d_kv_a);
    cudaFree(d_kv);
    cudaFree(d_attn_value);
    cudaFree(d_attn_mid);
    cudaFree(d_attn_out);
    return r;
}

GgufSharedExpertResult run_gguf_shared_expert_smoke(const std::string& ckpt_path,
                                                     int token) {
    if (!is_gguf_path(ckpt_path)) {
        throw std::runtime_error("run_gguf_shared_expert_smoke: not a GGUF path: " + ckpt_path);
    }
    if (token < 0) throw std::runtime_error("token must be >= 0");

    GgufForwardContext ctx(ckpt_path);
    WeightSource& ws = *ctx.weight_source;

    WeightView embed = ws.require("embed.weight");
    WeightView ffn_norm = ws.require("layers.0.ffn_norm.weight");
    WeightView w1 = ws.require("layers.0.ffn.shared_experts.w1.weight");
    WeightView w2 = ws.require("layers.0.ffn.shared_experts.w2.weight");
    WeightView w3 = ws.require("layers.0.ffn.shared_experts.w3.weight");

    if (embed.dtype != DType::F16) throw std::runtime_error("embed dtype");
    if (ffn_norm.dtype != DType::F32) throw std::runtime_error("ffn_norm dtype");
    if (w1.dtype != DType::Q8_0 || w2.dtype != DType::Q8_0 || w3.dtype != DType::Q8_0) {
        throw std::runtime_error("shared expert projections must be Q8_0");
    }

    const int dim = static_cast<int>(embed.shape[0]);
    const int moe_inter = static_cast<int>(ctx.config.moe_inter_dim);
    if (static_cast<int>(w1.shape[0]) != dim ||
        static_cast<int>(w1.shape[1]) != moe_inter) {
        throw std::runtime_error("shared w1 shape mismatch");
    }
    if (static_cast<int>(w3.shape[0]) != dim ||
        static_cast<int>(w3.shape[1]) != moe_inter) {
        throw std::runtime_error("shared w3 shape mismatch");
    }
    if (static_cast<int>(w2.shape[0]) != moe_inter ||
        static_cast<int>(w2.shape[1]) != dim) {
        throw std::runtime_error("shared w2 shape mismatch");
    }

    // ----- Embed -----
    const uint16_t* host_row =
        reinterpret_cast<const uint16_t*>(embed.data) +
        static_cast<size_t>(token) * dim;
    uint16_t* d_row_f16 = nullptr;
    float* d_x = nullptr;
    check_cuda(cudaMalloc(&d_row_f16, dim * sizeof(uint16_t)), "alloc d_row_f16");
    check_cuda(cudaMalloc(&d_x, dim * sizeof(float)), "alloc d_x");
    check_cuda(cudaMemcpy(d_row_f16, host_row, dim * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy embed row");
    if (!f16_row_to_float_cuda(d_row_f16, d_x, /*row=*/0, dim)) {
        throw std::runtime_error("f16_row_to_float_cuda failed");
    }

    // ----- ffn_norm -----
    auto ffn_bf16 = f32_to_bf16_host(reinterpret_cast<const float*>(ffn_norm.data), dim);
    uint16_t* d_ffn_gamma = nullptr;
    check_cuda(cudaMalloc(&d_ffn_gamma, dim * sizeof(uint16_t)), "alloc d_ffn_gamma");
    check_cuda(cudaMemcpy(d_ffn_gamma, ffn_bf16.data(), dim * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy ffn_gamma");
    float* d_x_normed = nullptr;
    check_cuda(cudaMalloc(&d_x_normed, dim * sizeof(float)), "alloc d_x_normed");
    if (!rmsnorm_bf16_gamma_cuda(d_x, d_ffn_gamma, d_x_normed, dim, 1e-6f)) {
        throw std::runtime_error("ffn_norm failed");
    }

    // ----- Shared expert chain: silu(w1(x)) * w3(x), then w2 -----
    uint8_t* d_w1 = nullptr;
    uint8_t* d_w2 = nullptr;
    uint8_t* d_w3 = nullptr;
    check_cuda(cudaMalloc(&d_w1, w1.nbytes), "alloc d_w1");
    check_cuda(cudaMalloc(&d_w2, w2.nbytes), "alloc d_w2");
    check_cuda(cudaMalloc(&d_w3, w3.nbytes), "alloc d_w3");
    check_cuda(cudaMemcpy(d_w1, w1.data, w1.nbytes, cudaMemcpyHostToDevice), "copy w1");
    check_cuda(cudaMemcpy(d_w2, w2.data, w2.nbytes, cudaMemcpyHostToDevice), "copy w2");
    check_cuda(cudaMemcpy(d_w3, w3.data, w3.nbytes, cudaMemcpyHostToDevice), "copy w3");

    float* d_gate = nullptr;
    float* d_up = nullptr;
    float* d_hidden = nullptr;
    float* d_shared_out = nullptr;
    check_cuda(cudaMalloc(&d_gate, moe_inter * sizeof(float)), "alloc d_gate");
    check_cuda(cudaMalloc(&d_up, moe_inter * sizeof(float)), "alloc d_up");
    check_cuda(cudaMalloc(&d_hidden, moe_inter * sizeof(float)), "alloc d_hidden");
    check_cuda(cudaMalloc(&d_shared_out, dim * sizeof(float)), "alloc d_shared_out");

    if (!q8_0_matvec_cuda(d_x_normed, d_w1, d_gate, moe_inter, dim)) {
        throw std::runtime_error("shared w1 failed");
    }
    if (!q8_0_matvec_cuda(d_x_normed, d_w3, d_up, moe_inter, dim)) {
        throw std::runtime_error("shared w3 failed");
    }
    if (!silu_mul_cuda(d_gate, d_up, d_hidden, moe_inter)) {
        throw std::runtime_error("silu_mul failed");
    }
    if (!q8_0_matvec_cuda(d_hidden, d_w2, d_shared_out, dim, moe_inter)) {
        throw std::runtime_error("shared w2 failed");
    }
    check_cuda(cudaDeviceSynchronize(), "sync after shared expert");

    GgufSharedExpertResult r;
    r.dim = dim;
    r.moe_inter_dim = moe_inter;
    r.ffn_normed_rms = device_vector_rms(d_x_normed, dim);
    r.gate_rms = device_vector_rms(d_gate, moe_inter);
    r.up_rms = device_vector_rms(d_up, moe_inter);
    r.hidden_rms = device_vector_rms(d_hidden, moe_inter);
    r.shared_out_rms = device_vector_rms(d_shared_out, dim);
    std::vector<float> first(4);
    check_cuda(cudaMemcpy(first.data(), d_shared_out, 4 * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy shared_out first");
    for (int i = 0; i < 4; ++i) r.shared_out_first[i] = first[i];

    cudaFree(d_row_f16);
    cudaFree(d_x);
    cudaFree(d_ffn_gamma);
    cudaFree(d_x_normed);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_w3);
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_hidden);
    cudaFree(d_shared_out);
    return r;
}

GgufRoutedExpertResult run_gguf_routed_expert_smoke(const std::string& ckpt_path,
                                                     int token,
                                                     int expert_id) {
    if (!is_gguf_path(ckpt_path)) {
        throw std::runtime_error("run_gguf_routed_expert_smoke: not a GGUF path: " + ckpt_path);
    }
    if (token < 0) throw std::runtime_error("token must be >= 0");
    if (expert_id < 0) throw std::runtime_error("expert_id must be >= 0");

    GgufForwardContext ctx(ckpt_path);
    GGUFWeightSource& gws = *ctx.weight_source;

    WeightView embed = gws.require("embed.weight");
    WeightView ffn_norm = gws.require("layers.0.ffn_norm.weight");
    // Per-expert slices from the routed 3D tensors. expert_id must be
    // < n_routed_experts; the GGUFWeightSource computes the byte offset by
    // dividing the total tensor bytes by n_experts.
    WeightView w1 = gws.get_expert("layers.0.ffn.experts.routed.w1",
                                    "layers.0.ffn.experts.routed.w1", expert_id);
    WeightView w2 = gws.get_expert("layers.0.ffn.experts.routed.w2",
                                    "layers.0.ffn.experts.routed.w2", expert_id);
    WeightView w3 = gws.get_expert("layers.0.ffn.experts.routed.w3",
                                    "layers.0.ffn.experts.routed.w3", expert_id);

    if (!w1.found || !w2.found || !w3.found) {
        throw std::runtime_error("routed expert slice not found (check expert_id)");
    }
    if (embed.dtype != DType::F16) throw std::runtime_error("embed dtype");
    if (ffn_norm.dtype != DType::F32) throw std::runtime_error("ffn_norm dtype");
    if (w1.dtype != DType::IQ2_XXS || w3.dtype != DType::IQ2_XXS) {
        throw std::runtime_error("routed w1/w3 must be IQ2_XXS");
    }
    if (w2.dtype != DType::Q2_K) {
        throw std::runtime_error("routed w2 must be Q2_K");
    }

    const int dim = static_cast<int>(embed.shape[0]);
    const int moe_inter = static_cast<int>(ctx.config.moe_inter_dim);
    if (dim <= 0 || moe_inter <= 0) {
        throw std::runtime_error("invalid dim / moe_inter");
    }

    // ----- Embed → ffn_norm -----
    const uint16_t* host_row =
        reinterpret_cast<const uint16_t*>(embed.data) +
        static_cast<size_t>(token) * dim;
    uint16_t* d_row_f16 = nullptr;
    float* d_x = nullptr;
    check_cuda(cudaMalloc(&d_row_f16, dim * sizeof(uint16_t)), "alloc d_row_f16");
    check_cuda(cudaMalloc(&d_x, dim * sizeof(float)), "alloc d_x");
    check_cuda(cudaMemcpy(d_row_f16, host_row, dim * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy embed row");
    if (!f16_row_to_float_cuda(d_row_f16, d_x, /*row=*/0, dim)) {
        throw std::runtime_error("f16 embed failed");
    }

    auto ffn_bf16 = f32_to_bf16_host(reinterpret_cast<const float*>(ffn_norm.data), dim);
    uint16_t* d_ffn_gamma = nullptr;
    check_cuda(cudaMalloc(&d_ffn_gamma, dim * sizeof(uint16_t)), "alloc d_ffn_gamma");
    check_cuda(cudaMemcpy(d_ffn_gamma, ffn_bf16.data(), dim * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy ffn_gamma");
    float* d_x_normed = nullptr;
    check_cuda(cudaMalloc(&d_x_normed, dim * sizeof(float)), "alloc d_x_normed");
    if (!rmsnorm_bf16_gamma_cuda(d_x, d_ffn_gamma, d_x_normed, dim, 1e-6f)) {
        throw std::runtime_error("ffn_norm failed");
    }

    // ----- Routed Q2 expert chain -----
    // Single route, single-expert buffer view (n_experts=1 + slot=0).
    const int routes = 1;
    const int x_groups = (dim + 31) / 32;
    const int hidden_groups = (moe_inter + 15) / 16;
    const int64_t h_route_slot = 0;
    const float h_route_weight = 1.0f;

    int8_t* d_x_q = nullptr;
    float* d_x_scale = nullptr;
    uint8_t* d_w1 = nullptr;
    uint8_t* d_w2 = nullptr;
    uint8_t* d_w3 = nullptr;
    int64_t* d_route_slots = nullptr;
    float* d_route_weights = nullptr;
    float* d_gate = nullptr;
    float* d_up = nullptr;
    int8_t* d_hidden_q = nullptr;
    float* d_hidden_scale = nullptr;
    float* d_y = nullptr;

    check_cuda(cudaMalloc(&d_x_q, dim), "alloc d_x_q");
    check_cuda(cudaMalloc(&d_x_scale, x_groups * sizeof(float)), "alloc d_x_scale");
    check_cuda(cudaMalloc(&d_w1, w1.nbytes), "alloc d_w1");
    check_cuda(cudaMalloc(&d_w2, w2.nbytes), "alloc d_w2");
    check_cuda(cudaMalloc(&d_w3, w3.nbytes), "alloc d_w3");
    check_cuda(cudaMalloc(&d_route_slots, sizeof(int64_t)), "alloc d_route_slots");
    check_cuda(cudaMalloc(&d_route_weights, sizeof(float)), "alloc d_route_weights");
    check_cuda(cudaMalloc(&d_gate, moe_inter * sizeof(float)), "alloc d_gate");
    check_cuda(cudaMalloc(&d_up, moe_inter * sizeof(float)), "alloc d_up");
    check_cuda(cudaMalloc(&d_hidden_q, moe_inter), "alloc d_hidden_q");
    check_cuda(cudaMalloc(&d_hidden_scale, hidden_groups * sizeof(float)),
               "alloc d_hidden_scale");
    check_cuda(cudaMalloc(&d_y, dim * sizeof(float)), "alloc d_y");

    check_cuda(cudaMemcpy(d_w1, w1.data, w1.nbytes, cudaMemcpyHostToDevice), "copy w1");
    check_cuda(cudaMemcpy(d_w2, w2.data, w2.nbytes, cudaMemcpyHostToDevice), "copy w2");
    check_cuda(cudaMemcpy(d_w3, w3.data, w3.nbytes, cudaMemcpyHostToDevice), "copy w3");
    check_cuda(cudaMemcpy(d_route_slots, &h_route_slot, sizeof(int64_t),
                          cudaMemcpyHostToDevice), "copy route_slots");
    check_cuda(cudaMemcpy(d_route_weights, &h_route_weight, sizeof(float),
                          cudaMemcpyHostToDevice), "copy route_weights");
    check_cuda(cudaMemset(d_y, 0, dim * sizeof(float)), "zero d_y");

    if (!q2_quantize_x_q8_1_cuda(d_x_normed, d_x_q, d_x_scale, routes, dim)) {
        throw std::runtime_error("q2_quantize_x_q8_1 failed");
    }
    const int kernel_n_experts = 1;
    if (!q2_moe_single_w13_iq2_xxs_cuda(d_x_q, d_x_scale, d_route_slots,
                                         d_w1, d_w3, d_gate, d_up,
                                         routes, kernel_n_experts, dim, moe_inter)) {
        throw std::runtime_error("q2 w13 iq2_xxs failed");
    }
    const float swiglu_limit = static_cast<float>(ctx.config.swiglu_limit);
    if (!q2_route_swiglu_quantize_hidden_q8_1_cuda(d_gate, d_up, d_route_weights,
                                                   d_hidden_q, d_hidden_scale,
                                                   routes, moe_inter, swiglu_limit)) {
        throw std::runtime_error("q2 swiglu + quantize failed");
    }
    if (!q2_moe_single_w2_q2k_cuda(d_hidden_q, d_hidden_scale, d_route_slots,
                                    d_w2, d_y, routes, kernel_n_experts,
                                    dim, moe_inter)) {
        throw std::runtime_error("q2 w2 q2k failed");
    }
    check_cuda(cudaDeviceSynchronize(), "sync after routed expert");

    GgufRoutedExpertResult r;
    r.dim = dim;
    r.moe_inter_dim = moe_inter;
    r.expert_id = expert_id;
    r.ffn_normed_rms = device_vector_rms(d_x_normed, dim);
    r.gate_rms = device_vector_rms(d_gate, moe_inter);
    r.up_rms = device_vector_rms(d_up, moe_inter);
    // Reconstruct hidden RMS from the Q8_1 representation: hidden_q is int8
    // grouped by 16 with per-group fp32 scale (`hidden_scale`). RMS of the
    // dequantized hidden = sqrt(mean(sum_g hidden_q[g] * scale[g])^2 / N).
    {
        std::vector<int8_t> hq(moe_inter);
        std::vector<float> hs(hidden_groups);
        check_cuda(cudaMemcpy(hq.data(), d_hidden_q, moe_inter, cudaMemcpyDeviceToHost),
                   "copy hidden_q");
        check_cuda(cudaMemcpy(hs.data(), d_hidden_scale, hidden_groups * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy hidden_scale");
        double s = 0.0;
        for (int i = 0; i < moe_inter; ++i) {
            const float v = static_cast<float>(hq[i]) * hs[i / 16];
            s += static_cast<double>(v) * static_cast<double>(v);
        }
        r.hidden_rms = static_cast<float>(std::sqrt(s / moe_inter));
    }
    r.route_out_rms = device_vector_rms(d_y, dim);
    std::vector<float> first(4);
    check_cuda(cudaMemcpy(first.data(), d_y, 4 * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy route_out first");
    for (int i = 0; i < 4; ++i) r.route_out_first[i] = first[i];

    cudaFree(d_row_f16);
    cudaFree(d_x);
    cudaFree(d_ffn_gamma);
    cudaFree(d_x_normed);
    cudaFree(d_x_q);
    cudaFree(d_x_scale);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_w3);
    cudaFree(d_route_slots);
    cudaFree(d_route_weights);
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_hidden_q);
    cudaFree(d_hidden_scale);
    cudaFree(d_y);
    return r;
}

GgufRoutedMoeResult run_gguf_routed_moe_smoke(const std::string& ckpt_path, int token) {
    if (!is_gguf_path(ckpt_path)) {
        throw std::runtime_error("run_gguf_routed_moe_smoke: not a GGUF path: " + ckpt_path);
    }
    if (token < 0) throw std::runtime_error("token must be >= 0");

    GgufForwardContext ctx(ckpt_path);
    GGUFWeightSource& gws = *ctx.weight_source;

    WeightView embed = gws.require("embed.weight");
    WeightView ffn_norm = gws.require("layers.0.ffn_norm.weight");
    if (embed.dtype != DType::F16) throw std::runtime_error("embed dtype");
    if (ffn_norm.dtype != DType::F32) throw std::runtime_error("ffn_norm dtype");

    // Layer 0 is a hash gate layer: read precomputed expert ids from tid2eid.
    // GGUF native shape [topk, vocab] = [6, 129280] stored as int32 row-major
    // where the outer dim (vocab) varies slowest. Per-token row offset:
    // token * topk * sizeof(int32). Internal-name lookup goes through the
    // mapping table which Transpose2Ds the shape view to [vocab, topk].
    WeightView tid2eid = gws.require("layers.0.ffn.gate.tid2eid");
    // tid2eid dtype is i32 which is not in DType; just verify byte count.
    const int topk = static_cast<int>(ctx.config.n_activated_experts);
    const int vocab = static_cast<int>(embed.shape[1]);
    if (topk <= 0 || topk > 8) {
        throw std::runtime_error("topk out of supported range (1..8)");
    }
    if (token >= vocab) throw std::runtime_error("token id out of vocab range");
    if (tid2eid.nbytes < static_cast<uint64_t>(token + 1) * topk * sizeof(int32_t)) {
        throw std::runtime_error("tid2eid too small for requested token");
    }
    const int32_t* token_eids =
        reinterpret_cast<const int32_t*>(tid2eid.data) +
        static_cast<size_t>(token) * topk;
    int h_expert_ids[8] = {0};
    for (int k = 0; k < topk; ++k) h_expert_ids[k] = token_eids[k];

    const int dim = static_cast<int>(embed.shape[0]);
    const int moe_inter = static_cast<int>(ctx.config.moe_inter_dim);

    // ----- Embed → ffn_norm -----
    const uint16_t* host_row =
        reinterpret_cast<const uint16_t*>(embed.data) +
        static_cast<size_t>(token) * dim;
    uint16_t* d_row_f16 = nullptr;
    float* d_x = nullptr;
    check_cuda(cudaMalloc(&d_row_f16, dim * sizeof(uint16_t)), "alloc d_row_f16");
    check_cuda(cudaMalloc(&d_x, dim * sizeof(float)), "alloc d_x");
    check_cuda(cudaMemcpy(d_row_f16, host_row, dim * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy embed row");
    if (!f16_row_to_float_cuda(d_row_f16, d_x, /*row=*/0, dim)) {
        throw std::runtime_error("f16 embed failed");
    }
    auto ffn_bf16 = f32_to_bf16_host(reinterpret_cast<const float*>(ffn_norm.data), dim);
    uint16_t* d_ffn_gamma = nullptr;
    check_cuda(cudaMalloc(&d_ffn_gamma, dim * sizeof(uint16_t)), "alloc d_ffn_gamma");
    check_cuda(cudaMemcpy(d_ffn_gamma, ffn_bf16.data(), dim * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy ffn_gamma");
    float* d_x_normed = nullptr;
    check_cuda(cudaMalloc(&d_x_normed, dim * sizeof(float)), "alloc d_x_normed");
    if (!rmsnorm_bf16_gamma_cuda(d_x, d_ffn_gamma, d_x_normed, dim, 1e-6f)) {
        throw std::runtime_error("ffn_norm failed");
    }

    // ----- Active expert staging: pack topk experts' w1/w2/w3 bytes -----
    // Use the first expert's view to learn per-expert sizes. Each routed
    // tensor has shape [inner=4096, mid=2048, n_experts=256] (or w2 variant
    // with mid=4096, inner=2048); per-expert byte size is total_nbytes / 256.
    auto first_w1 = gws.get_expert("layers.0.ffn.experts.routed.w1",
                                    "layers.0.ffn.experts.routed.w1", 0);
    auto first_w2 = gws.get_expert("layers.0.ffn.experts.routed.w2",
                                    "layers.0.ffn.experts.routed.w2", 0);
    auto first_w3 = gws.get_expert("layers.0.ffn.experts.routed.w3",
                                    "layers.0.ffn.experts.routed.w3", 0);
    if (!first_w1.found || !first_w2.found || !first_w3.found) {
        throw std::runtime_error("get_expert(0) failed");
    }
    const uint64_t per_w1_bytes = first_w1.nbytes;
    const uint64_t per_w2_bytes = first_w2.nbytes;
    const uint64_t per_w3_bytes = first_w3.nbytes;
    const uint64_t staged_w1_bytes = per_w1_bytes * static_cast<uint64_t>(topk);
    const uint64_t staged_w2_bytes = per_w2_bytes * static_cast<uint64_t>(topk);
    const uint64_t staged_w3_bytes = per_w3_bytes * static_cast<uint64_t>(topk);
    uint8_t* d_w1 = nullptr;
    uint8_t* d_w2 = nullptr;
    uint8_t* d_w3 = nullptr;
    check_cuda(cudaMalloc(&d_w1, staged_w1_bytes), "alloc staged d_w1");
    check_cuda(cudaMalloc(&d_w2, staged_w2_bytes), "alloc staged d_w2");
    check_cuda(cudaMalloc(&d_w3, staged_w3_bytes), "alloc staged d_w3");
    for (int k = 0; k < topk; ++k) {
        const int eid = h_expert_ids[k];
        auto wv1 = gws.get_expert("layers.0.ffn.experts.routed.w1",
                                   "layers.0.ffn.experts.routed.w1", eid);
        auto wv2 = gws.get_expert("layers.0.ffn.experts.routed.w2",
                                   "layers.0.ffn.experts.routed.w2", eid);
        auto wv3 = gws.get_expert("layers.0.ffn.experts.routed.w3",
                                   "layers.0.ffn.experts.routed.w3", eid);
        if (!wv1.found || !wv2.found || !wv3.found) {
            throw std::runtime_error("get_expert failed for active expert");
        }
        check_cuda(cudaMemcpy(d_w1 + per_w1_bytes * k, wv1.data, per_w1_bytes,
                              cudaMemcpyHostToDevice), "stage w1");
        check_cuda(cudaMemcpy(d_w2 + per_w2_bytes * k, wv2.data, per_w2_bytes,
                              cudaMemcpyHostToDevice), "stage w2");
        check_cuda(cudaMemcpy(d_w3 + per_w3_bytes * k, wv3.data, per_w3_bytes,
                              cudaMemcpyHostToDevice), "stage w3");
    }

    // ----- Routes / quant buffers -----
    const int routes = topk;
    const int x_groups = (dim + 31) / 32;
    const int hidden_groups = (moe_inter + 15) / 16;
    int8_t* d_x_q = nullptr;
    float* d_x_scale = nullptr;
    int64_t* d_route_slots = nullptr;
    float* d_route_weights = nullptr;
    float* d_gate = nullptr;
    float* d_up = nullptr;
    int8_t* d_hidden_q = nullptr;
    float* d_hidden_scale = nullptr;
    float* d_y = nullptr;
    check_cuda(cudaMalloc(&d_x_q, dim), "alloc d_x_q");
    check_cuda(cudaMalloc(&d_x_scale, x_groups * sizeof(float)), "alloc d_x_scale");
    check_cuda(cudaMalloc(&d_route_slots, routes * sizeof(int64_t)), "alloc slots");
    check_cuda(cudaMalloc(&d_route_weights, routes * sizeof(float)), "alloc weights");
    // Per-route gate / up / hidden buffers (kernel addresses by route).
    check_cuda(cudaMalloc(&d_gate, static_cast<size_t>(routes) * moe_inter * sizeof(float)),
               "alloc d_gate");
    check_cuda(cudaMalloc(&d_up, static_cast<size_t>(routes) * moe_inter * sizeof(float)),
               "alloc d_up");
    check_cuda(cudaMalloc(&d_hidden_q, static_cast<size_t>(routes) * moe_inter),
               "alloc d_hidden_q");
    check_cuda(cudaMalloc(&d_hidden_scale,
                          static_cast<size_t>(routes) * hidden_groups * sizeof(float)),
               "alloc d_hidden_scale");
    check_cuda(cudaMalloc(&d_y, dim * sizeof(float)), "alloc d_y");
    check_cuda(cudaMemset(d_y, 0, dim * sizeof(float)), "zero d_y");

    // Staged expert buffer holds K experts back-to-back; slots index 0..K-1.
    std::vector<int64_t> h_slots(routes);
    for (int k = 0; k < routes; ++k) h_slots[k] = k;
    check_cuda(cudaMemcpy(d_route_slots, h_slots.data(), routes * sizeof(int64_t),
                          cudaMemcpyHostToDevice), "copy slots");

    // ----- Hash-gate route weights: gate W matvec → sqrt_softplus → gather →
    // normalize → ×route_scale. PyTorch reference at runtime/transformer.py:1682
    // computes the same gather+normalize for hash layers (where indices come
    // from tid2eid). Gate W is stored F16 in this GGUF; convert to BF16 once
    // and reuse the existing gate_hash_bf16_cuda kernel pair. -----
    WeightView gate_w = gws.require("layers.0.ffn.gate.weight");
    if (gate_w.dtype != DType::F16) {
        throw std::runtime_error("gate.weight dtype expected F16");
    }
    // Native GGUF shape [dim, n_experts]; bf16_matvec_cuda treats the bytes
    // as [n_experts, dim] row-major which matches the logical layout because
    // dim is the fast (innermost) GGUF dimension.
    const int n_experts = static_cast<int>(ctx.config.n_routed_experts);
    if (static_cast<int>(gate_w.shape[0]) != dim ||
        static_cast<int>(gate_w.shape[1]) != n_experts) {
        throw std::runtime_error("gate.weight shape mismatch");
    }
    auto gate_w_bf16 = f16_to_bf16_host(
        reinterpret_cast<const uint16_t*>(gate_w.data),
        static_cast<size_t>(dim) * static_cast<size_t>(n_experts));
    uint16_t* d_gate_w_bf16 = nullptr;
    int64_t* d_tid2eid_i64 = nullptr;
    float* d_gate_scores_scratch = nullptr;
    int64_t* d_gate_indices = nullptr;
    check_cuda(cudaMalloc(&d_gate_w_bf16, gate_w_bf16.size() * sizeof(uint16_t)),
               "alloc d_gate_w_bf16");
    check_cuda(cudaMemcpy(d_gate_w_bf16, gate_w_bf16.data(),
                          gate_w_bf16.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy gate_w_bf16");
    // Stage just the per-token tid2eid slice as int64[topk]; kernel will
    // index it with token=0, table_topk=topk so it reads positions [0..topk).
    std::vector<int64_t> h_tid2eid_slice(topk);
    for (int k = 0; k < topk; ++k) h_tid2eid_slice[k] = h_expert_ids[k];
    check_cuda(cudaMalloc(&d_tid2eid_i64, topk * sizeof(int64_t)),
               "alloc d_tid2eid_i64");
    check_cuda(cudaMemcpy(d_tid2eid_i64, h_tid2eid_slice.data(),
                          topk * sizeof(int64_t), cudaMemcpyHostToDevice),
               "copy d_tid2eid_i64");
    check_cuda(cudaMalloc(&d_gate_scores_scratch, topk * sizeof(float)),
               "alloc d_gate_scores_scratch");
    check_cuda(cudaMalloc(&d_gate_indices, topk * sizeof(int64_t)),
               "alloc d_gate_indices");
    const float route_scale = static_cast<float>(ctx.config.route_scale);
    if (!gate_hash_bf16_cuda(d_x_normed, d_gate_w_bf16, d_tid2eid_i64,
                             d_gate_scores_scratch, d_gate_indices,
                             d_route_weights,
                             /*token=*/0, /*cols=*/dim,
                             /*table_topk=*/topk, /*topk=*/topk,
                             route_scale)) {
        throw std::runtime_error("gate_hash_bf16_cuda failed");
    }

    // ----- Quantize x once (the w13 kernel reads x_q without per-route stride) -----
    if (!q2_quantize_x_q8_1_cuda(d_x_normed, d_x_q, d_x_scale, /*routes=*/1, dim)) {
        throw std::runtime_error("q2_quantize_x_q8_1 failed");
    }

    // ----- Batched IQ2_XXS w1/w3 across all active experts -----
    if (!q2_moe_single_w13_iq2_xxs_cuda(d_x_q, d_x_scale, d_route_slots,
                                         d_w1, d_w3, d_gate, d_up,
                                         routes, /*n_experts=*/routes,
                                         dim, moe_inter)) {
        throw std::runtime_error("q2 w13 iq2_xxs failed");
    }

    // ----- SwiGLU + route weight + Q8_1 quantize hidden per-route -----
    const float swiglu_limit = static_cast<float>(ctx.config.swiglu_limit);
    if (!q2_route_swiglu_quantize_hidden_q8_1_cuda(d_gate, d_up, d_route_weights,
                                                   d_hidden_q, d_hidden_scale,
                                                   routes, moe_inter, swiglu_limit)) {
        throw std::runtime_error("q2 swiglu + quantize failed");
    }

    // ----- Batched Q2_K w2 (accumulates into d_y via atomicAdd) -----
    if (!q2_moe_single_w2_q2k_cuda(d_hidden_q, d_hidden_scale, d_route_slots,
                                    d_w2, d_y,
                                    routes, /*n_experts=*/routes,
                                    dim, moe_inter)) {
        throw std::runtime_error("q2 w2 q2k failed");
    }
    check_cuda(cudaDeviceSynchronize(), "sync after routed moe");

    GgufRoutedMoeResult r;
    r.dim = dim;
    r.moe_inter_dim = moe_inter;
    r.n_active = topk;
    for (int k = 0; k < topk; ++k) r.expert_ids[k] = h_expert_ids[k];
    r.ffn_normed_rms = device_vector_rms(d_x_normed, dim);
    r.moe_out_rms = device_vector_rms(d_y, dim);
    std::vector<float> first(4);
    check_cuda(cudaMemcpy(first.data(), d_y, 4 * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy moe_out first");
    for (int i = 0; i < 4; ++i) r.moe_out_first[i] = first[i];
    std::vector<float> h_rw(topk);
    check_cuda(cudaMemcpy(h_rw.data(), d_route_weights, topk * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy route_weights");
    float sum = 0.0f;
    for (int k = 0; k < topk; ++k) {
        r.route_weights[k] = h_rw[k];
        sum += h_rw[k];
    }
    r.route_weights_sum = sum;

    cudaFree(d_row_f16);
    cudaFree(d_x);
    cudaFree(d_ffn_gamma);
    cudaFree(d_x_normed);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_w3);
    cudaFree(d_x_q);
    cudaFree(d_x_scale);
    cudaFree(d_route_slots);
    cudaFree(d_route_weights);
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_hidden_q);
    cudaFree(d_hidden_scale);
    cudaFree(d_y);
    cudaFree(d_gate_w_bf16);
    cudaFree(d_tid2eid_i64);
    cudaFree(d_gate_scores_scratch);
    cudaFree(d_gate_indices);
    return r;
}

GgufLayer0FullResult run_gguf_layer0_full_smoke(const std::string& ckpt_path,
                                                 int token,
                                                 int position) {
    if (!is_gguf_path(ckpt_path)) {
        throw std::runtime_error("run_gguf_layer0_full_smoke: not a GGUF path: " + ckpt_path);
    }
    if (token < 0) throw std::runtime_error("token must be >= 0");
    if (position < 0) throw std::runtime_error("position must be >= 0");

    GgufForwardContext ctx(ckpt_path);
    GGUFWeightSource& gws = *ctx.weight_source;

    // ----- attention weights -----
    WeightView embed = gws.require("embed.weight");
    WeightView attn_norm = gws.require("layers.0.attn_norm.weight");
    WeightView wq_a = gws.require("layers.0.attn.wq_a.weight");
    WeightView q_norm = gws.require("layers.0.attn.q_norm.weight");
    WeightView wq_b = gws.require("layers.0.attn.wq_b.weight");
    WeightView wkv = gws.require("layers.0.attn.wkv.weight");
    WeightView kv_norm = gws.require("layers.0.attn.kv_norm.weight");
    WeightView attn_sink = gws.require("layers.0.attn.attn_sink");
    WeightView wo_a = gws.require("layers.0.attn.wo_a.weight");
    WeightView wo_b = gws.require("layers.0.attn.wo_b.weight");
    // ----- FFN weights -----
    WeightView ffn_norm = gws.require("layers.0.ffn_norm.weight");
    WeightView shared_w1 = gws.require("layers.0.ffn.shared_experts.w1.weight");
    WeightView shared_w2 = gws.require("layers.0.ffn.shared_experts.w2.weight");
    WeightView shared_w3 = gws.require("layers.0.ffn.shared_experts.w3.weight");
    WeightView gate_w = gws.require("layers.0.ffn.gate.weight");
    WeightView tid2eid = gws.require("layers.0.ffn.gate.tid2eid");

    if (embed.dtype != DType::F16) throw std::runtime_error("embed dtype");
    if (attn_norm.dtype != DType::F32 || q_norm.dtype != DType::F32 ||
        kv_norm.dtype != DType::F32 || ffn_norm.dtype != DType::F32 ||
        attn_sink.dtype != DType::F32) {
        throw std::runtime_error("norm/sink dtypes must be F32");
    }
    if (wq_a.dtype != DType::Q8_0 || wq_b.dtype != DType::Q8_0 ||
        wkv.dtype != DType::Q8_0 || wo_a.dtype != DType::Q8_0 ||
        wo_b.dtype != DType::Q8_0) {
        throw std::runtime_error("attention projections must be Q8_0");
    }
    if (shared_w1.dtype != DType::Q8_0 || shared_w2.dtype != DType::Q8_0 ||
        shared_w3.dtype != DType::Q8_0) {
        throw std::runtime_error("shared expert weights must be Q8_0");
    }
    if (gate_w.dtype != DType::F16) {
        throw std::runtime_error("gate.weight dtype expected F16");
    }

    // ----- dimensions -----
    const int dim = static_cast<int>(embed.shape[0]);
    const int q_a_dim = static_cast<int>(wq_a.shape[1]);
    const int heads = static_cast<int>(ctx.config.n_heads);
    const int q_full = static_cast<int>(wq_b.shape[1]);
    if (q_full % heads != 0) throw std::runtime_error("wq_b cols % heads != 0");
    const int head_dim = q_full / heads;
    const int kv_dim = static_cast<int>(wkv.shape[1]);
    const int rope_dim = static_cast<int>(ctx.config.rope_dim);
    if (kv_dim != head_dim) throw std::runtime_error("expected kv_dim == head_dim");
    const int o_groups = static_cast<int>(ctx.config.o_groups);
    const int o_lora_rank = static_cast<int>(ctx.config.o_lora_rank);
    if (q_full % o_groups != 0) throw std::runtime_error("q_full % o_groups");
    const int group_in_dim = q_full / o_groups;
    const int attn_mid = o_groups * o_lora_rank;
    const int moe_inter = static_cast<int>(ctx.config.moe_inter_dim);
    const int n_experts = static_cast<int>(ctx.config.n_routed_experts);
    const int topk = static_cast<int>(ctx.config.n_activated_experts);
    const int vocab = static_cast<int>(embed.shape[1]);
    if (token >= vocab) throw std::runtime_error("token id out of vocab range");
    if (topk <= 0 || topk > 8) throw std::runtime_error("topk out of supported range");
    if (tid2eid.nbytes < static_cast<uint64_t>(token + 1) * topk * sizeof(int32_t)) {
        throw std::runtime_error("tid2eid too small for requested token");
    }

    // tid2eid hash gate row for this token.
    const int32_t* token_eids =
        reinterpret_cast<const int32_t*>(tid2eid.data) +
        static_cast<size_t>(token) * topk;
    int h_expert_ids[8] = {0};
    for (int k = 0; k < topk; ++k) h_expert_ids[k] = token_eids[k];

    // ----- upload helpers -----
    auto upload_u8 = [](const void* src, size_t bytes) {
        uint8_t* d = nullptr;
        check_cuda(cudaMalloc(&d, bytes), "alloc");
        check_cuda(cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice), "copy");
        return d;
    };
    auto upload_f32 = [](const void* src, size_t n) {
        float* d = nullptr;
        check_cuda(cudaMalloc(&d, n * sizeof(float)), "alloc f32");
        check_cuda(cudaMemcpy(d, src, n * sizeof(float), cudaMemcpyHostToDevice), "copy f32");
        return d;
    };
    auto upload_bf16_from_f32 = [&](const void* src, size_t n) {
        auto bf = f32_to_bf16_host(reinterpret_cast<const float*>(src), n);
        uint16_t* d = nullptr;
        check_cuda(cudaMalloc(&d, n * sizeof(uint16_t)), "alloc bf16");
        check_cuda(cudaMemcpy(d, bf.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice),
                   "copy bf16");
        return d;
    };

    // Embed row.
    const uint16_t* host_row =
        reinterpret_cast<const uint16_t*>(embed.data) +
        static_cast<size_t>(token) * dim;
    uint16_t* d_row_f16 = nullptr;
    check_cuda(cudaMalloc(&d_row_f16, dim * sizeof(uint16_t)), "alloc d_row_f16");
    check_cuda(cudaMemcpy(d_row_f16, host_row, dim * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy embed row");

    // Attention norm gammas.
    uint16_t* d_attn_gamma = upload_bf16_from_f32(attn_norm.data, dim);
    uint16_t* d_q_gamma = upload_bf16_from_f32(q_norm.data, q_a_dim);
    uint16_t* d_kv_gamma = upload_bf16_from_f32(kv_norm.data, kv_dim);
    uint16_t* d_ffn_gamma = upload_bf16_from_f32(ffn_norm.data, dim);
    // Attention Q8_0 weights.
    uint8_t* d_wq_a = upload_u8(wq_a.data, wq_a.nbytes);
    uint8_t* d_wq_b = upload_u8(wq_b.data, wq_b.nbytes);
    uint8_t* d_wkv = upload_u8(wkv.data, wkv.nbytes);
    uint8_t* d_wo_a = upload_u8(wo_a.data, wo_a.nbytes);
    uint8_t* d_wo_b = upload_u8(wo_b.data, wo_b.nbytes);
    float* d_attn_sink = upload_f32(attn_sink.data, heads);
    // Shared expert Q8_0 weights.
    uint8_t* d_shared_w1 = upload_u8(shared_w1.data, shared_w1.nbytes);
    uint8_t* d_shared_w2 = upload_u8(shared_w2.data, shared_w2.nbytes);
    uint8_t* d_shared_w3 = upload_u8(shared_w3.data, shared_w3.nbytes);
    // Gate W (F16 -> BF16).
    auto gate_w_bf16 = f16_to_bf16_host(
        reinterpret_cast<const uint16_t*>(gate_w.data),
        static_cast<size_t>(dim) * static_cast<size_t>(n_experts));
    uint16_t* d_gate_w_bf16 = nullptr;
    check_cuda(cudaMalloc(&d_gate_w_bf16, gate_w_bf16.size() * sizeof(uint16_t)),
               "alloc d_gate_w_bf16");
    check_cuda(cudaMemcpy(d_gate_w_bf16, gate_w_bf16.data(),
                          gate_w_bf16.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy gate_w_bf16");

    // Stage top-k routed experts.
    auto first_w1 = gws.get_expert("layers.0.ffn.experts.routed.w1",
                                    "layers.0.ffn.experts.routed.w1", 0);
    auto first_w2 = gws.get_expert("layers.0.ffn.experts.routed.w2",
                                    "layers.0.ffn.experts.routed.w2", 0);
    auto first_w3 = gws.get_expert("layers.0.ffn.experts.routed.w3",
                                    "layers.0.ffn.experts.routed.w3", 0);
    if (!first_w1.found || !first_w2.found || !first_w3.found) {
        throw std::runtime_error("get_expert(0) failed");
    }
    const uint64_t per_w1_bytes = first_w1.nbytes;
    const uint64_t per_w2_bytes = first_w2.nbytes;
    const uint64_t per_w3_bytes = first_w3.nbytes;
    uint8_t* d_routed_w1 = nullptr;
    uint8_t* d_routed_w2 = nullptr;
    uint8_t* d_routed_w3 = nullptr;
    check_cuda(cudaMalloc(&d_routed_w1, per_w1_bytes * topk), "alloc routed_w1");
    check_cuda(cudaMalloc(&d_routed_w2, per_w2_bytes * topk), "alloc routed_w2");
    check_cuda(cudaMalloc(&d_routed_w3, per_w3_bytes * topk), "alloc routed_w3");
    for (int k = 0; k < topk; ++k) {
        const int eid = h_expert_ids[k];
        auto wv1 = gws.get_expert("layers.0.ffn.experts.routed.w1",
                                   "layers.0.ffn.experts.routed.w1", eid);
        auto wv2 = gws.get_expert("layers.0.ffn.experts.routed.w2",
                                   "layers.0.ffn.experts.routed.w2", eid);
        auto wv3 = gws.get_expert("layers.0.ffn.experts.routed.w3",
                                   "layers.0.ffn.experts.routed.w3", eid);
        if (!wv1.found || !wv2.found || !wv3.found) {
            throw std::runtime_error("get_expert failed for active expert");
        }
        check_cuda(cudaMemcpy(d_routed_w1 + per_w1_bytes * k, wv1.data, per_w1_bytes,
                              cudaMemcpyHostToDevice), "stage routed_w1");
        check_cuda(cudaMemcpy(d_routed_w2 + per_w2_bytes * k, wv2.data, per_w2_bytes,
                              cudaMemcpyHostToDevice), "stage routed_w2");
        check_cuda(cudaMemcpy(d_routed_w3 + per_w3_bytes * k, wv3.data, per_w3_bytes,
                              cudaMemcpyHostToDevice), "stage routed_w3");
    }

    // tid2eid slice + scratch for gate scoring.
    std::vector<int64_t> h_tid2eid_slice(topk);
    for (int k = 0; k < topk; ++k) h_tid2eid_slice[k] = h_expert_ids[k];
    int64_t* d_tid2eid_i64 = nullptr;
    float* d_gate_scores_scratch = nullptr;
    int64_t* d_gate_indices = nullptr;
    check_cuda(cudaMalloc(&d_tid2eid_i64, topk * sizeof(int64_t)), "alloc d_tid2eid_i64");
    check_cuda(cudaMemcpy(d_tid2eid_i64, h_tid2eid_slice.data(),
                          topk * sizeof(int64_t), cudaMemcpyHostToDevice),
               "copy d_tid2eid_i64");
    check_cuda(cudaMalloc(&d_gate_scores_scratch, topk * sizeof(float)),
               "alloc d_gate_scores_scratch");
    check_cuda(cudaMalloc(&d_gate_indices, topk * sizeof(int64_t)),
               "alloc d_gate_indices");

    // ----- activation buffers -----
    float* d_x = nullptr;            // working hidden state (residual-updated)
    float* d_x_pre_attn = nullptr;   // saved before attn_norm for residual
    float* d_x_normed = nullptr;
    float* d_q_a = nullptr;
    float* d_q_normed = nullptr;
    float* d_q = nullptr;
    float* d_kv_a = nullptr;
    float* d_kv = nullptr;
    float* d_attn_value = nullptr;
    float* d_attn_mid = nullptr;
    float* d_attn_out = nullptr;
    float* d_x_pre_ffn = nullptr;    // saved after attn residual for FFN residual
    float* d_x_normed_ffn = nullptr;
    float* d_shared_gate = nullptr;
    float* d_shared_up = nullptr;
    float* d_shared_hidden = nullptr;
    float* d_shared_out = nullptr;
    float* d_moe_out = nullptr;
    float* d_ffn_combined = nullptr;
    check_cuda(cudaMalloc(&d_x, dim * sizeof(float)), "alloc d_x");
    check_cuda(cudaMalloc(&d_x_pre_attn, dim * sizeof(float)), "alloc d_x_pre_attn");
    check_cuda(cudaMalloc(&d_x_normed, dim * sizeof(float)), "alloc d_x_normed");
    check_cuda(cudaMalloc(&d_q_a, q_a_dim * sizeof(float)), "alloc d_q_a");
    check_cuda(cudaMalloc(&d_q_normed, q_a_dim * sizeof(float)), "alloc d_q_normed");
    check_cuda(cudaMalloc(&d_q, q_full * sizeof(float)), "alloc d_q");
    check_cuda(cudaMalloc(&d_kv_a, kv_dim * sizeof(float)), "alloc d_kv_a");
    check_cuda(cudaMalloc(&d_kv, kv_dim * sizeof(float)), "alloc d_kv");
    check_cuda(cudaMalloc(&d_attn_value, q_full * sizeof(float)), "alloc d_attn_value");
    check_cuda(cudaMalloc(&d_attn_mid, attn_mid * sizeof(float)), "alloc d_attn_mid");
    check_cuda(cudaMalloc(&d_attn_out, dim * sizeof(float)), "alloc d_attn_out");
    check_cuda(cudaMalloc(&d_x_pre_ffn, dim * sizeof(float)), "alloc d_x_pre_ffn");
    check_cuda(cudaMalloc(&d_x_normed_ffn, dim * sizeof(float)), "alloc d_x_normed_ffn");
    check_cuda(cudaMalloc(&d_shared_gate, moe_inter * sizeof(float)), "alloc d_shared_gate");
    check_cuda(cudaMalloc(&d_shared_up, moe_inter * sizeof(float)), "alloc d_shared_up");
    check_cuda(cudaMalloc(&d_shared_hidden, moe_inter * sizeof(float)), "alloc d_shared_hidden");
    check_cuda(cudaMalloc(&d_shared_out, dim * sizeof(float)), "alloc d_shared_out");
    check_cuda(cudaMalloc(&d_moe_out, dim * sizeof(float)), "alloc d_moe_out");
    check_cuda(cudaMalloc(&d_ffn_combined, dim * sizeof(float)), "alloc d_ffn_combined");
    check_cuda(cudaMemset(d_moe_out, 0, dim * sizeof(float)), "zero d_moe_out");

    // Routed MoE staging buffers.
    const int routes = topk;
    const int x_groups = (dim + 31) / 32;
    const int hidden_groups = (moe_inter + 15) / 16;
    int8_t* d_x_q = nullptr;
    float* d_x_scale = nullptr;
    int64_t* d_route_slots = nullptr;
    float* d_route_weights = nullptr;
    float* d_route_gate = nullptr;
    float* d_route_up = nullptr;
    int8_t* d_route_hidden_q = nullptr;
    float* d_route_hidden_scale = nullptr;
    float* d_route_hidden = nullptr;
    check_cuda(cudaMalloc(&d_x_q, dim), "alloc d_x_q");
    check_cuda(cudaMalloc(&d_x_scale, x_groups * sizeof(float)), "alloc d_x_scale");
    check_cuda(cudaMalloc(&d_route_slots, routes * sizeof(int64_t)), "alloc d_route_slots");
    check_cuda(cudaMalloc(&d_route_weights, routes * sizeof(float)), "alloc d_route_weights");
    check_cuda(cudaMalloc(&d_route_gate, static_cast<size_t>(routes) * moe_inter * sizeof(float)),
               "alloc d_route_gate");
    check_cuda(cudaMalloc(&d_route_up, static_cast<size_t>(routes) * moe_inter * sizeof(float)),
               "alloc d_route_up");
    check_cuda(cudaMalloc(&d_route_hidden_q, static_cast<size_t>(routes) * moe_inter),
               "alloc d_route_hidden_q");
    check_cuda(cudaMalloc(&d_route_hidden_scale,
                          static_cast<size_t>(routes) * hidden_groups * sizeof(float)),
               "alloc d_route_hidden_scale");
    check_cuda(cudaMalloc(&d_route_hidden, static_cast<size_t>(routes) * moe_inter * sizeof(float)),
               "alloc d_route_hidden");
    std::vector<int64_t> h_route_slots(routes);
    for (int k = 0; k < routes; ++k) h_route_slots[k] = k;
    check_cuda(cudaMemcpy(d_route_slots, h_route_slots.data(),
                          routes * sizeof(int64_t), cudaMemcpyHostToDevice),
               "copy d_route_slots");

    // ===== compute =====

    // 1. Embed (runs once before the first layer's forward).
    if (!f16_row_to_float_cuda(d_row_f16, d_x, /*row=*/0, dim)) {
        throw std::runtime_error("f16 embed failed");
    }

    // 2-9. Per-layer forward via shared helper (attention + FFN + residuals).
    GgufLayerDeviceWeights lw;
    lw.d_attn_gamma = d_attn_gamma;
    lw.d_q_gamma = d_q_gamma;
    lw.d_kv_gamma = d_kv_gamma;
    lw.d_wq_a = d_wq_a;
    lw.d_wq_b = d_wq_b;
    lw.d_wkv = d_wkv;
    lw.d_wo_a = d_wo_a;
    lw.d_wo_b = d_wo_b;
    lw.d_attn_sink = d_attn_sink;
    lw.d_ffn_gamma = d_ffn_gamma;
    lw.d_shared_w1 = d_shared_w1;
    lw.d_shared_w2 = d_shared_w2;
    lw.d_shared_w3 = d_shared_w3;
    lw.d_gate_w_bf16 = d_gate_w_bf16;
    lw.is_hash = true;
    lw.d_tid2eid_i64 = d_tid2eid_i64;
    lw.d_gate_bias_f32 = nullptr;
    lw.d_routed_w1 = d_routed_w1;
    lw.d_routed_w2 = d_routed_w2;
    lw.d_routed_w3 = d_routed_w3;
    lw.routed_n_experts = routes;
    lw.routed_w1_dtype = first_w1.dtype;
    lw.routed_w2_dtype = first_w2.dtype;
    lw.routed_w3_dtype = first_w3.dtype;

    GgufLayerDims ld;
    ld.dim = dim;
    ld.q_a_dim = q_a_dim;
    ld.heads = heads;
    ld.head_dim = head_dim;
    ld.q_full = q_full;
    ld.kv_dim = kv_dim;
    ld.rope_dim = rope_dim;
    ld.o_groups = o_groups;
    ld.o_lora_rank = o_lora_rank;
    ld.group_in_dim = group_in_dim;
    ld.attn_mid = attn_mid;
    ld.moe_inter = moe_inter;
    ld.n_experts = n_experts;
    ld.topk = topk;
    ld.rope_theta = static_cast<float>(ctx.config.rope_theta);
    ld.compress_rope_theta = static_cast<float>(ctx.config.compress_rope_theta == 0 ? ctx.config.rope_theta : ctx.config.compress_rope_theta);
    ld.swiglu_limit = static_cast<float>(ctx.config.swiglu_limit);
    ld.route_scale = static_cast<float>(ctx.config.route_scale);

    GgufLayerScratch ls;
    ls.d_x = d_x;
    ls.d_x_pre_attn = d_x_pre_attn;
    ls.d_x_normed = d_x_normed;
    ls.d_q_a = d_q_a;
    ls.d_q_normed = d_q_normed;
    ls.d_q = d_q;
    ls.d_kv_a = d_kv_a;
    ls.d_kv = d_kv;
    ls.d_attn_value = d_attn_value;
    ls.d_attn_mid = d_attn_mid;
    ls.d_attn_out = d_attn_out;
    ls.d_x_pre_ffn = d_x_pre_ffn;
    ls.d_x_normed_ffn = d_x_normed_ffn;
    ls.d_shared_gate = d_shared_gate;
    ls.d_shared_up = d_shared_up;
    ls.d_shared_hidden = d_shared_hidden;
    ls.d_shared_out = d_shared_out;
    ls.d_moe_out = d_moe_out;
    ls.d_ffn_combined = d_ffn_combined;
    ls.d_x_q = d_x_q;
    ls.d_x_scale = d_x_scale;
    ls.d_route_slots = d_route_slots;
    ls.d_route_weights = d_route_weights;
    ls.d_route_gate = d_route_gate;
    ls.d_route_up = d_route_up;
    ls.d_route_hidden_q = d_route_hidden_q;
    ls.d_route_hidden_scale = d_route_hidden_scale;
    ls.d_route_hidden = d_route_hidden;
    ls.d_gate_scores_scratch = d_gate_scores_scratch;
    ls.d_gate_scored_scratch = nullptr;  // hash gate path doesn't need this
    ls.d_gate_indices = d_gate_indices;

    gguf_layer_forward(lw, ls, ld, token, position);
    check_cuda(cudaDeviceSynchronize(), "sync after layer-0 forward");

    GgufLayer0FullResult r;
    r.dim = dim;
    r.moe_inter_dim = moe_inter;
    r.heads = heads;
    r.head_dim = head_dim;
    r.n_active = topk;
    for (int k = 0; k < topk; ++k) r.expert_ids[k] = h_expert_ids[k];
    r.embed_rms = device_vector_rms(d_x_pre_attn, dim);
    r.attn_out_rms = device_vector_rms(d_attn_out, dim);
    // Reuse d_x_normed as scratch — overwrite ok, we're done with it.
    r.x_post_attn_rms = device_vector_rms(d_x_pre_ffn, dim);
    r.shared_out_rms = device_vector_rms(d_shared_out, dim);
    r.moe_out_rms = device_vector_rms(d_moe_out, dim);
    r.ffn_combined_rms = device_vector_rms(d_ffn_combined, dim);
    r.x_post_ffn_rms = device_vector_rms(d_x, dim);
    std::vector<float> first(4);
    check_cuda(cudaMemcpy(first.data(), d_x, 4 * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy x_post_ffn first");
    for (int i = 0; i < 4; ++i) r.x_post_ffn_first[i] = first[i];
    std::vector<float> h_rw(topk);
    check_cuda(cudaMemcpy(h_rw.data(), d_route_weights, topk * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy route_weights");
    float sum = 0.0f;
    for (int k = 0; k < topk; ++k) sum += h_rw[k];
    r.route_weights_sum = sum;

    cudaFree(d_row_f16);
    cudaFree(d_attn_gamma);
    cudaFree(d_q_gamma);
    cudaFree(d_kv_gamma);
    cudaFree(d_ffn_gamma);
    cudaFree(d_wq_a);
    cudaFree(d_wq_b);
    cudaFree(d_wkv);
    cudaFree(d_wo_a);
    cudaFree(d_wo_b);
    cudaFree(d_attn_sink);
    cudaFree(d_shared_w1);
    cudaFree(d_shared_w2);
    cudaFree(d_shared_w3);
    cudaFree(d_gate_w_bf16);
    cudaFree(d_routed_w1);
    cudaFree(d_routed_w2);
    cudaFree(d_routed_w3);
    cudaFree(d_tid2eid_i64);
    cudaFree(d_gate_scores_scratch);
    cudaFree(d_gate_indices);
    cudaFree(d_x);
    cudaFree(d_x_pre_attn);
    cudaFree(d_x_normed);
    cudaFree(d_q_a);
    cudaFree(d_q_normed);
    cudaFree(d_q);
    cudaFree(d_kv_a);
    cudaFree(d_kv);
    cudaFree(d_attn_value);
    cudaFree(d_attn_mid);
    cudaFree(d_attn_out);
    cudaFree(d_x_pre_ffn);
    cudaFree(d_x_normed_ffn);
    cudaFree(d_shared_gate);
    cudaFree(d_shared_up);
    cudaFree(d_shared_hidden);
    cudaFree(d_shared_out);
    cudaFree(d_moe_out);
    cudaFree(d_ffn_combined);
    cudaFree(d_x_q);
    cudaFree(d_x_scale);
    cudaFree(d_route_slots);
    cudaFree(d_route_weights);
    cudaFree(d_route_gate);
    cudaFree(d_route_up);
    cudaFree(d_route_hidden_q);
    cudaFree(d_route_hidden_scale);
    cudaFree(d_route_hidden);
    return r;
}

GgufFullForwardResult run_gguf_full_forward_smoke(const std::string& ckpt_path,
                                                  int token,
                                                  int position) {
    if (!is_gguf_path(ckpt_path)) {
        throw std::runtime_error("run_gguf_full_forward_smoke: not a GGUF path: " + ckpt_path);
    }
    if (token < 0) throw std::runtime_error("token must be >= 0");
    if (position < 0) throw std::runtime_error("position must be >= 0");

    GgufForwardContext ctx(ckpt_path);
    GGUFWeightSource& gws = *ctx.weight_source;

    // ===== model dims =====
    const int n_layers = static_cast<int>(ctx.config.n_layers);
    const int n_hash = static_cast<int>(ctx.config.n_hash_layers);
    const int dim = static_cast<int>(ctx.config.dim);
    const int heads = static_cast<int>(ctx.config.n_heads);
    const int n_experts = static_cast<int>(ctx.config.n_routed_experts);
    const int topk = static_cast<int>(ctx.config.n_activated_experts);
    const int rope_dim = static_cast<int>(ctx.config.rope_dim);
    const int o_groups = static_cast<int>(ctx.config.o_groups);
    const int o_lora_rank = static_cast<int>(ctx.config.o_lora_rank);
    const int moe_inter = static_cast<int>(ctx.config.moe_inter_dim);
    if (topk <= 0 || topk > 8) throw std::runtime_error("topk out of supported range");

    // Derive q_a_dim / q_full / kv_dim from layer-0 weights.
    WeightView wq_a0 = gws.require("layers.0.attn.wq_a.weight");
    WeightView wq_b0 = gws.require("layers.0.attn.wq_b.weight");
    WeightView wkv0 = gws.require("layers.0.attn.wkv.weight");
    const int q_a_dim = static_cast<int>(wq_a0.shape[1]);
    const int q_full = static_cast<int>(wq_b0.shape[1]);
    if (q_full % heads != 0) throw std::runtime_error("wq_b cols % heads");
    const int head_dim = q_full / heads;
    const int kv_dim = static_cast<int>(wkv0.shape[1]);
    if (q_full % o_groups != 0) throw std::runtime_error("q_full % o_groups");
    const int group_in_dim = q_full / o_groups;
    const int attn_mid = o_groups * o_lora_rank;

    WeightView embed = gws.require("embed.weight");
    if (embed.dtype != DType::F16 || embed.shape.size() != 2 || static_cast<int>(embed.shape[0]) != dim)
        throw std::runtime_error("embed shape/dtype mismatch");
    const int vocab = static_cast<int>(embed.shape[1]);
    if (token >= vocab) throw std::runtime_error("token id out of vocab range");

    // ===== upload helpers =====
    auto upload_u8 = [](const void* src, size_t bytes) {
        uint8_t* d = nullptr;
        check_cuda(cudaMalloc(&d, bytes), "alloc u8");
        check_cuda(cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice), "copy u8");
        return d;
    };
    auto upload_f32 = [](const void* src, size_t n) {
        float* d = nullptr;
        check_cuda(cudaMalloc(&d, n * sizeof(float)), "alloc f32");
        check_cuda(cudaMemcpy(d, src, n * sizeof(float), cudaMemcpyHostToDevice), "copy f32");
        return d;
    };
    auto upload_bf16_from_f32 = [&](const void* src, size_t n) {
        auto bf = f32_to_bf16_host(reinterpret_cast<const float*>(src), n);
        uint16_t* d = nullptr;
        check_cuda(cudaMalloc(&d, n * sizeof(uint16_t)), "alloc bf16");
        check_cuda(cudaMemcpy(d, bf.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy bf16");
        return d;
    };
    auto upload_bf16_from_f16 = [&](const void* src, size_t n) {
        auto bf = f16_to_bf16_host(reinterpret_cast<const uint16_t*>(src), n);
        uint16_t* d = nullptr;
        check_cuda(cudaMalloc(&d, n * sizeof(uint16_t)), "alloc bf16-from-f16");
        check_cuda(cudaMemcpy(d, bf.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy bf16-from-f16");
        return d;
    };
    auto upload_f32_from_f16 = [&](const void* src, size_t n) {
        auto f32 = f16_to_f32_host(reinterpret_cast<const uint16_t*>(src), n);
        float* d = nullptr;
        check_cuda(cudaMalloc(&d, n * sizeof(float)), "alloc f32-from-f16");
        check_cuda(cudaMemcpy(d, f32.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy f32-from-f16");
        return d;
    };

    // ===== per-layer dense weights resident on device =====
    std::vector<uint16_t*> d_attn_gamma(n_layers, nullptr);
    std::vector<uint16_t*> d_q_gamma(n_layers, nullptr);
    std::vector<uint16_t*> d_kv_gamma(n_layers, nullptr);
    std::vector<uint16_t*> d_ffn_gamma(n_layers, nullptr);
    std::vector<float*> d_attn_sink(n_layers, nullptr);
    std::vector<uint8_t*> d_wq_a(n_layers, nullptr);
    std::vector<uint8_t*> d_wq_b(n_layers, nullptr);
    std::vector<uint8_t*> d_wkv(n_layers, nullptr);
    std::vector<uint8_t*> d_wo_a(n_layers, nullptr);
    std::vector<uint8_t*> d_wo_b(n_layers, nullptr);
    std::vector<uint8_t*> d_shared_w1(n_layers, nullptr);
    std::vector<uint8_t*> d_shared_w2(n_layers, nullptr);
    std::vector<uint8_t*> d_shared_w3(n_layers, nullptr);
    std::vector<uint16_t*> d_gate_w(n_layers, nullptr);
    std::vector<float*> d_gate_bias(n_layers, nullptr);
    // Per-layer host pointer to tid2eid raw bytes for hash layers (i32 [topk, vocab]).
    std::vector<const int32_t*> h_tid2eid_table(n_layers, nullptr);

    for (int L = 0; L < n_layers; ++L) {
        const std::string lp = "layers." + std::to_string(L);
        WeightView attn_norm = gws.require(lp + ".attn_norm.weight");
        WeightView q_norm = gws.require(lp + ".attn.q_norm.weight");
        WeightView kv_norm = gws.require(lp + ".attn.kv_norm.weight");
        WeightView ffn_norm = gws.require(lp + ".ffn_norm.weight");
        WeightView attn_sink = gws.require(lp + ".attn.attn_sink");
        WeightView wq_a = gws.require(lp + ".attn.wq_a.weight");
        WeightView wq_b = gws.require(lp + ".attn.wq_b.weight");
        WeightView wkv = gws.require(lp + ".attn.wkv.weight");
        WeightView wo_a = gws.require(lp + ".attn.wo_a.weight");
        WeightView wo_b = gws.require(lp + ".attn.wo_b.weight");
        WeightView shared_w1 = gws.require(lp + ".ffn.shared_experts.w1.weight");
        WeightView shared_w2 = gws.require(lp + ".ffn.shared_experts.w2.weight");
        WeightView shared_w3 = gws.require(lp + ".ffn.shared_experts.w3.weight");
        WeightView gate_w = gws.require(lp + ".ffn.gate.weight");

        d_attn_gamma[L] = upload_bf16_from_f32(attn_norm.data, dim);
        d_q_gamma[L] = upload_bf16_from_f32(q_norm.data, q_a_dim);
        d_kv_gamma[L] = upload_bf16_from_f32(kv_norm.data, kv_dim);
        d_ffn_gamma[L] = upload_bf16_from_f32(ffn_norm.data, dim);
        d_attn_sink[L] = upload_f32(attn_sink.data, heads);
        d_wq_a[L] = upload_u8(wq_a.data, wq_a.nbytes);
        d_wq_b[L] = upload_u8(wq_b.data, wq_b.nbytes);
        d_wkv[L] = upload_u8(wkv.data, wkv.nbytes);
        d_wo_a[L] = upload_u8(wo_a.data, wo_a.nbytes);
        d_wo_b[L] = upload_u8(wo_b.data, wo_b.nbytes);
        d_shared_w1[L] = upload_u8(shared_w1.data, shared_w1.nbytes);
        d_shared_w2[L] = upload_u8(shared_w2.data, shared_w2.nbytes);
        d_shared_w3[L] = upload_u8(shared_w3.data, shared_w3.nbytes);
        d_gate_w[L] = upload_bf16_from_f16(gate_w.data,
                                            static_cast<size_t>(dim) * static_cast<size_t>(n_experts));

        if (L < n_hash) {
            WeightView tid2eid = gws.require(lp + ".ffn.gate.tid2eid");
            if (tid2eid.nbytes < static_cast<uint64_t>(vocab) * topk * sizeof(int32_t))
                throw std::runtime_error("tid2eid table truncated");
            h_tid2eid_table[L] = reinterpret_cast<const int32_t*>(tid2eid.data);
            d_gate_bias[L] = nullptr;
        } else {
            WeightView bias = gws.require(lp + ".ffn.gate.bias");
            if (bias.dtype != DType::F32 || bias.shape.size() != 1 ||
                static_cast<int>(bias.shape[0]) != n_experts)
                throw std::runtime_error("exp_probs_b.bias shape/dtype");
            d_gate_bias[L] = upload_f32(bias.data, n_experts);
            h_tid2eid_table[L] = nullptr;
        }
    }

    // ===== final norm + head =====
    WeightView final_norm = gws.require("norm.weight");
    WeightView head = gws.require("head.weight");
    if (final_norm.dtype != DType::F32 || final_norm.shape.size() != 1 ||
        static_cast<int>(final_norm.shape[0]) != dim)
        throw std::runtime_error("norm.weight shape/dtype");
    if (head.dtype != DType::Q8_0 || head.shape.size() != 2 ||
        static_cast<int>(head.shape[0]) != dim ||
        static_cast<int>(head.shape[1]) != vocab)
        throw std::runtime_error("head.weight shape/dtype (expected Q8_0 [dim, vocab])");
    uint16_t* d_final_norm_gamma = upload_bf16_from_f32(final_norm.data, dim);
    uint8_t* d_head = upload_u8(head.data, head.nbytes);

    // ===== expert staging buffers (reused per layer) =====
    std::vector<uint64_t> layer_w1_bytes(n_layers, 0);
    std::vector<uint64_t> layer_w2_bytes(n_layers, 0);
    std::vector<uint64_t> layer_w3_bytes(n_layers, 0);
    std::vector<DType> layer_w1_dtype(n_layers, DType::Unknown);
    std::vector<DType> layer_w2_dtype(n_layers, DType::Unknown);
    std::vector<DType> layer_w3_dtype(n_layers, DType::Unknown);
    uint64_t max_per_w1_bytes = 0;
    uint64_t max_per_w2_bytes = 0;
    uint64_t max_per_w3_bytes = 0;
    for (int L = 0; L < n_layers; ++L) {
        const std::string lp = "layers." + std::to_string(L);
        const std::string w1_name = lp + ".ffn.experts.routed.w1";
        const std::string w2_name = lp + ".ffn.experts.routed.w2";
        const std::string w3_name = lp + ".ffn.experts.routed.w3";
        auto f1 = gws.get_expert(w1_name, w1_name, 0);
        auto f2 = gws.get_expert(w2_name, w2_name, 0);
        auto f3 = gws.get_expert(w3_name, w3_name, 0);
        if (!f1.found || !f2.found || !f3.found)
            throw std::runtime_error("get_expert(0) failed while sizing routed buffers");
        layer_w1_bytes[L] = f1.nbytes;
        layer_w2_bytes[L] = f2.nbytes;
        layer_w3_bytes[L] = f3.nbytes;
        layer_w1_dtype[L] = f1.dtype;
        layer_w2_dtype[L] = f2.dtype;
        layer_w3_dtype[L] = f3.dtype;
        max_per_w1_bytes = std::max(max_per_w1_bytes, f1.nbytes);
        max_per_w2_bytes = std::max(max_per_w2_bytes, f2.nbytes);
        max_per_w3_bytes = std::max(max_per_w3_bytes, f3.nbytes);
    }

    uint8_t* d_routed_w1 = nullptr;
    uint8_t* d_routed_w2 = nullptr;
    uint8_t* d_routed_w3 = nullptr;
    check_cuda(cudaMalloc(&d_routed_w1, max_per_w1_bytes * topk), "alloc routed_w1");
    check_cuda(cudaMalloc(&d_routed_w2, max_per_w2_bytes * topk), "alloc routed_w2");
    check_cuda(cudaMalloc(&d_routed_w3, max_per_w3_bytes * topk), "alloc routed_w3");

    // ===== per-layer KV cache buffers (sized for this smoke's single step) =====
    // Cache capacity = position + 1: enough to write the current step's KV into
    // slot `position` and run cached attention over [0..position]. At
    // position=0 this is equivalent to the legacy single_token_sparse_attention
    // path and exercises the cached_single_token_attention helper end-to-end.
    const int cache_capacity = position + 1;
    std::vector<float*> d_kv_cache(n_layers, nullptr);
    for (int L = 0; L < n_layers; ++L) {
        check_cuda(cudaMalloc(&d_kv_cache[L],
                              static_cast<size_t>(cache_capacity) *
                                  static_cast<size_t>(kv_dim) * sizeof(float)),
                   "alloc d_kv_cache");
        check_cuda(cudaMemset(d_kv_cache[L], 0,
                              static_cast<size_t>(cache_capacity) *
                                  static_cast<size_t>(kv_dim) * sizeof(float)),
                   "memset d_kv_cache");
    }

    // ===== activation + scratch buffers (allocated once, reused per layer) =====
    const int x_groups = (dim + 31) / 32;
    const int hidden_groups = (moe_inter + 15) / 16;
    const int gate_scratch_n = std::max(topk, n_experts);
    float* d_x = nullptr;
    float* d_x_pre_attn = nullptr;
    float* d_x_normed = nullptr;
    float* d_q_a = nullptr;
    float* d_q_normed = nullptr;
    float* d_q = nullptr;
    float* d_kv_a = nullptr;
    float* d_kv = nullptr;
    float* d_attn_value = nullptr;
    float* d_attn_mid = nullptr;
    float* d_attn_out = nullptr;
    float* d_x_pre_ffn = nullptr;
    float* d_x_normed_ffn = nullptr;
    float* d_shared_gate = nullptr;
    float* d_shared_up = nullptr;
    float* d_shared_hidden = nullptr;
    float* d_shared_out = nullptr;
    float* d_moe_out = nullptr;
    float* d_ffn_combined = nullptr;
    int8_t* d_x_q = nullptr;
    float* d_x_scale = nullptr;
    int64_t* d_route_slots = nullptr;
    float* d_route_weights = nullptr;
    float* d_route_gate = nullptr;
    float* d_route_up = nullptr;
    int8_t* d_route_hidden_q = nullptr;
    float* d_route_hidden_scale = nullptr;
    float* d_route_hidden = nullptr;
    float* d_gate_scores_scratch = nullptr;
    float* d_gate_scored_scratch = nullptr;
    int64_t* d_gate_indices = nullptr;
    int64_t* d_tid2eid_i64 = nullptr;
    uint16_t* d_embed_row_f16 = nullptr;
    uint16_t* d_compressor_input_bf16 = nullptr;
    float* d_compressor_input_rounded = nullptr;
    float* d_compressor_kv = nullptr;
    float* d_compressor_score = nullptr;
    float* d_indexer_comp_kv = nullptr;
    float* d_indexer_comp_score = nullptr;
    float* d_index_q = nullptr;
    float* d_index_scores = nullptr;
    float* d_logits = nullptr;
    check_cuda(cudaMalloc(&d_x, dim * sizeof(float)), "alloc d_x");
    check_cuda(cudaMalloc(&d_x_pre_attn, dim * sizeof(float)), "alloc d_x_pre_attn");
    check_cuda(cudaMalloc(&d_x_normed, dim * sizeof(float)), "alloc d_x_normed");
    check_cuda(cudaMalloc(&d_q_a, q_a_dim * sizeof(float)), "alloc d_q_a");
    check_cuda(cudaMalloc(&d_q_normed, q_a_dim * sizeof(float)), "alloc d_q_normed");
    check_cuda(cudaMalloc(&d_q, q_full * sizeof(float)), "alloc d_q");
    check_cuda(cudaMalloc(&d_kv_a, kv_dim * sizeof(float)), "alloc d_kv_a");
    check_cuda(cudaMalloc(&d_kv, kv_dim * sizeof(float)), "alloc d_kv");
    check_cuda(cudaMalloc(&d_attn_value, q_full * sizeof(float)), "alloc d_attn_value");
    check_cuda(cudaMalloc(&d_attn_mid, attn_mid * sizeof(float)), "alloc d_attn_mid");
    check_cuda(cudaMalloc(&d_attn_out, dim * sizeof(float)), "alloc d_attn_out");
    check_cuda(cudaMalloc(&d_x_pre_ffn, dim * sizeof(float)), "alloc d_x_pre_ffn");
    check_cuda(cudaMalloc(&d_x_normed_ffn, dim * sizeof(float)), "alloc d_x_normed_ffn");
    check_cuda(cudaMalloc(&d_shared_gate, moe_inter * sizeof(float)), "alloc d_shared_gate");
    check_cuda(cudaMalloc(&d_shared_up, moe_inter * sizeof(float)), "alloc d_shared_up");
    check_cuda(cudaMalloc(&d_shared_hidden, moe_inter * sizeof(float)), "alloc d_shared_hidden");
    check_cuda(cudaMalloc(&d_shared_out, dim * sizeof(float)), "alloc d_shared_out");
    check_cuda(cudaMalloc(&d_moe_out, dim * sizeof(float)), "alloc d_moe_out");
    check_cuda(cudaMalloc(&d_ffn_combined, dim * sizeof(float)), "alloc d_ffn_combined");
    check_cuda(cudaMalloc(&d_x_q, dim), "alloc d_x_q");
    check_cuda(cudaMalloc(&d_x_scale, x_groups * sizeof(float)), "alloc d_x_scale");
    check_cuda(cudaMalloc(&d_route_slots, topk * sizeof(int64_t)), "alloc d_route_slots");
    check_cuda(cudaMalloc(&d_route_weights, topk * sizeof(float)), "alloc d_route_weights");
    check_cuda(cudaMalloc(&d_route_gate, static_cast<size_t>(topk) * moe_inter * sizeof(float)),
               "alloc d_route_gate");
    check_cuda(cudaMalloc(&d_route_up, static_cast<size_t>(topk) * moe_inter * sizeof(float)),
               "alloc d_route_up");
    check_cuda(cudaMalloc(&d_route_hidden_q, static_cast<size_t>(topk) * moe_inter),
               "alloc d_route_hidden_q");
    check_cuda(cudaMalloc(&d_route_hidden_scale,
                          static_cast<size_t>(topk) * hidden_groups * sizeof(float)),
               "alloc d_route_hidden_scale");
    check_cuda(cudaMalloc(&d_route_hidden, static_cast<size_t>(topk) * moe_inter * sizeof(float)),
               "alloc d_route_hidden");
    check_cuda(cudaMalloc(&d_gate_scores_scratch, gate_scratch_n * sizeof(float)),
               "alloc d_gate_scores_scratch");
    check_cuda(cudaMalloc(&d_gate_scored_scratch, n_experts * sizeof(float)),
               "alloc d_gate_scored_scratch");
    check_cuda(cudaMalloc(&d_gate_indices, topk * sizeof(int64_t)), "alloc d_gate_indices");
    check_cuda(cudaMalloc(&d_tid2eid_i64, topk * sizeof(int64_t)), "alloc d_tid2eid_i64");
    check_cuda(cudaMalloc(&d_embed_row_f16, dim * sizeof(uint16_t)), "alloc d_embed_row_f16");
    check_cuda(cudaMalloc(&d_logits, vocab * sizeof(float)), "alloc d_logits");

    std::vector<int64_t> h_route_slots(topk);
    for (int k = 0; k < topk; ++k) h_route_slots[k] = k;
    check_cuda(cudaMemcpy(d_route_slots, h_route_slots.data(),
                          topk * sizeof(int64_t), cudaMemcpyHostToDevice),
               "copy d_route_slots");

    // ===== embed =====
    const uint16_t* host_embed_row =
        reinterpret_cast<const uint16_t*>(embed.data) +
        static_cast<size_t>(token) * dim;
    check_cuda(cudaMemcpy(d_embed_row_f16, host_embed_row, dim * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy embed row");
    if (!f16_row_to_float_cuda(d_embed_row_f16, d_x, /*row=*/0, dim))
        throw std::runtime_error("f16 embed failed");

    // ===== shared dims for helper =====
    GgufLayerDims ld;
    ld.dim = dim; ld.q_a_dim = q_a_dim; ld.heads = heads; ld.head_dim = head_dim;
    ld.q_full = q_full; ld.kv_dim = kv_dim; ld.rope_dim = rope_dim;
    ld.o_groups = o_groups; ld.o_lora_rank = o_lora_rank; ld.group_in_dim = group_in_dim;
    ld.attn_mid = attn_mid; ld.moe_inter = moe_inter;
    ld.n_experts = n_experts; ld.topk = topk;
    ld.rope_theta = static_cast<float>(ctx.config.rope_theta);
    ld.swiglu_limit = static_cast<float>(ctx.config.swiglu_limit);
    ld.route_scale = static_cast<float>(ctx.config.route_scale);

    GgufLayerScratch ls;
    ls.d_x = d_x;
    ls.d_x_pre_attn = d_x_pre_attn; ls.d_x_normed = d_x_normed;
    ls.d_q_a = d_q_a; ls.d_q_normed = d_q_normed; ls.d_q = d_q;
    ls.d_kv_a = d_kv_a; ls.d_kv = d_kv;
    ls.d_attn_value = d_attn_value; ls.d_attn_mid = d_attn_mid; ls.d_attn_out = d_attn_out;
    ls.d_x_pre_ffn = d_x_pre_ffn; ls.d_x_normed_ffn = d_x_normed_ffn;
    ls.d_shared_gate = d_shared_gate; ls.d_shared_up = d_shared_up;
    ls.d_shared_hidden = d_shared_hidden; ls.d_shared_out = d_shared_out;
    ls.d_moe_out = d_moe_out; ls.d_ffn_combined = d_ffn_combined;
    ls.d_x_q = d_x_q; ls.d_x_scale = d_x_scale;
    ls.d_route_slots = d_route_slots; ls.d_route_weights = d_route_weights;
    ls.d_route_gate = d_route_gate; ls.d_route_up = d_route_up;
    ls.d_route_hidden_q = d_route_hidden_q; ls.d_route_hidden_scale = d_route_hidden_scale;
    ls.d_route_hidden = d_route_hidden;
    ls.d_gate_scores_scratch = d_gate_scores_scratch;
    ls.d_gate_scored_scratch = d_gate_scored_scratch;
    ls.d_gate_indices = d_gate_indices;

    // ===== per-layer loop =====
    std::vector<int64_t> h_gate_indices(topk);
    for (int L = 0; L < n_layers; ++L) {
        const bool is_hash = (L < n_hash);

        GgufLayerDeviceWeights lw;
        lw.d_attn_gamma = d_attn_gamma[L]; lw.d_q_gamma = d_q_gamma[L]; lw.d_kv_gamma = d_kv_gamma[L];
        lw.d_wq_a = d_wq_a[L]; lw.d_wq_b = d_wq_b[L]; lw.d_wkv = d_wkv[L];
        lw.d_wo_a = d_wo_a[L]; lw.d_wo_b = d_wo_b[L];
        lw.d_attn_sink = d_attn_sink[L];
        lw.d_ffn_gamma = d_ffn_gamma[L];
        lw.d_shared_w1 = d_shared_w1[L]; lw.d_shared_w2 = d_shared_w2[L]; lw.d_shared_w3 = d_shared_w3[L];
        lw.d_gate_w_bf16 = d_gate_w[L];
        lw.is_hash = is_hash;
        lw.d_routed_w1 = d_routed_w1; lw.d_routed_w2 = d_routed_w2; lw.d_routed_w3 = d_routed_w3;
        lw.routed_n_experts = topk;
        lw.routed_w1_dtype = layer_w1_dtype[L];
        lw.routed_w2_dtype = layer_w2_dtype[L];
        lw.routed_w3_dtype = layer_w3_dtype[L];

        ls.d_kv_cache = d_kv_cache[L];
        ls.cache_capacity = cache_capacity;

        if (is_hash) {
            // Upload per-token tid2eid row, converted to int64 (kernel expects i64).
            const int32_t* row = h_tid2eid_table[L] + static_cast<size_t>(token) * topk;
            int64_t h_slice[8];
            for (int k = 0; k < topk; ++k) h_slice[k] = row[k];
            check_cuda(cudaMemcpy(d_tid2eid_i64, h_slice, topk * sizeof(int64_t),
                                  cudaMemcpyHostToDevice), "copy tid2eid slice");
            lw.d_tid2eid_i64 = d_tid2eid_i64;
            lw.d_gate_bias_f32 = nullptr;
        } else {
            lw.d_tid2eid_i64 = nullptr;
            lw.d_gate_bias_f32 = d_gate_bias[L];
        }

        // Phase A: attention + ffn_norm + shared + gate.
        gguf_layer_forward_attn_to_gate(lw, ls, ld, position, nullptr);

        // Read gate's chosen expert ids, then stage those experts H2D.
        check_cuda(cudaMemcpy(h_gate_indices.data(), d_gate_indices, topk * sizeof(int64_t),
                              cudaMemcpyDeviceToHost), "copy gate indices");
        const std::string lp = "layers." + std::to_string(L);
        const std::string w1_name = lp + ".ffn.experts.routed.w1";
        const std::string w2_name = lp + ".ffn.experts.routed.w2";
        const std::string w3_name = lp + ".ffn.experts.routed.w3";
        for (int k = 0; k < topk; ++k) {
            const int eid = static_cast<int>(h_gate_indices[k]);
            if (eid < 0 || eid >= n_experts)
                throw std::runtime_error("gate produced out-of-range expert id");
            auto wv1 = gws.get_expert(w1_name, w1_name, eid);
            auto wv2 = gws.get_expert(w2_name, w2_name, eid);
            auto wv3 = gws.get_expert(w3_name, w3_name, eid);
            if (!wv1.found || !wv2.found || !wv3.found)
                throw std::runtime_error("get_expert failed during full forward staging");
            if (wv1.dtype != layer_w1_dtype[L] || wv2.dtype != layer_w2_dtype[L] || wv3.dtype != layer_w3_dtype[L])
                throw std::runtime_error("routed expert dtype changed within full forward layer");
            check_cuda(cudaMemcpy(d_routed_w1 + layer_w1_bytes[L] * k, wv1.data, layer_w1_bytes[L],
                                  cudaMemcpyHostToDevice), "stage routed_w1");
            check_cuda(cudaMemcpy(d_routed_w2 + layer_w2_bytes[L] * k, wv2.data, layer_w2_bytes[L],
                                  cudaMemcpyHostToDevice), "stage routed_w2");
            check_cuda(cudaMemcpy(d_routed_w3 + layer_w3_bytes[L] * k, wv3.data, layer_w3_bytes[L],
                                  cudaMemcpyHostToDevice), "stage routed_w3");
        }

        // Phase B: routed MoE + FFN residual.
        gguf_layer_forward_moe(lw, ls, ld, nullptr);
    }

    // ===== final norm + head =====
    if (!rmsnorm_bf16_gamma_cuda(d_x, d_final_norm_gamma, d_x_normed, dim, 1e-6f))
        throw std::runtime_error("final norm failed");
    if (!q8_0_matvec_cuda(d_x_normed, d_head, d_logits, vocab, dim))
        throw std::runtime_error("head matvec failed");
    check_cuda(cudaDeviceSynchronize(), "sync after head");

    std::vector<float> h_logits(vocab);
    check_cuda(cudaMemcpy(h_logits.data(), d_logits, vocab * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy logits");

    int top_token = 0;
    float top_logit = -INFINITY;
    double sum = 0.0;
    double sq = 0.0;
    for (int i = 0; i < vocab; ++i) {
        const float v = h_logits[i];
        sum += v;
        sq += static_cast<double>(v) * static_cast<double>(v);
        if (v > top_logit) { top_logit = v; top_token = i; }
    }

    GgufFullForwardResult r;
    r.n_layers = n_layers;
    r.dim = dim;
    r.vocab = vocab;
    r.top_token = top_token;
    r.top_logit = top_logit;
    r.checksum = static_cast<float>(sum);
    r.final_x_rms = device_vector_rms(d_x, dim);
    r.final_normed_rms = device_vector_rms(d_x_normed, dim);
    r.logits_rms = static_cast<float>(std::sqrt(sq / vocab));
    for (int i = 0; i < 4; ++i) r.logits_first[i] = h_logits[i];

    // ===== cleanup =====
    auto free_vec_u8 = [](std::vector<uint8_t*>& v) { for (auto* p : v) cudaFree(p); };
    auto free_vec_u16 = [](std::vector<uint16_t*>& v) { for (auto* p : v) cudaFree(p); };
    auto free_vec_f32 = [](std::vector<float*>& v) { for (auto* p : v) cudaFree(p); };
    free_vec_u16(d_attn_gamma); free_vec_u16(d_q_gamma); free_vec_u16(d_kv_gamma);
    free_vec_u16(d_ffn_gamma); free_vec_f32(d_attn_sink);
    free_vec_u8(d_wq_a); free_vec_u8(d_wq_b); free_vec_u8(d_wkv);
    free_vec_u8(d_wo_a); free_vec_u8(d_wo_b);
    free_vec_u8(d_shared_w1); free_vec_u8(d_shared_w2); free_vec_u8(d_shared_w3);
    free_vec_u16(d_gate_w); free_vec_f32(d_gate_bias);
    cudaFree(d_final_norm_gamma); cudaFree(d_head);
    cudaFree(d_routed_w1); cudaFree(d_routed_w2); cudaFree(d_routed_w3);
    free_vec_f32(d_kv_cache);
    cudaFree(d_x); cudaFree(d_x_pre_attn); cudaFree(d_x_normed);
    cudaFree(d_q_a); cudaFree(d_q_normed); cudaFree(d_q);
    cudaFree(d_kv_a); cudaFree(d_kv); cudaFree(d_attn_value);
    cudaFree(d_attn_mid); cudaFree(d_attn_out);
    cudaFree(d_x_pre_ffn); cudaFree(d_x_normed_ffn);
    cudaFree(d_shared_gate); cudaFree(d_shared_up); cudaFree(d_shared_hidden);
    cudaFree(d_shared_out); cudaFree(d_moe_out); cudaFree(d_ffn_combined);
    cudaFree(d_x_q); cudaFree(d_x_scale);
    cudaFree(d_route_slots); cudaFree(d_route_weights);
    cudaFree(d_route_gate); cudaFree(d_route_up);
    cudaFree(d_route_hidden_q); cudaFree(d_route_hidden_scale); cudaFree(d_route_hidden);
    cudaFree(d_gate_scores_scratch); cudaFree(d_gate_scored_scratch); cudaFree(d_gate_indices);
    cudaFree(d_tid2eid_i64); cudaFree(d_embed_row_f16); cudaFree(d_logits);
    return r;
}

GgufDecodeResult run_gguf_generate_smoke(const std::string& ckpt_path,
                                          const std::vector<int>& seed_tokens,
                                          int max_new_tokens,
                                          const ForwardSmokeOptions& options) {
    if (!is_gguf_path(ckpt_path))
        throw std::runtime_error("run_gguf_generate_smoke: not a GGUF path: " + ckpt_path);
    if (seed_tokens.empty()) throw std::runtime_error("seed_tokens must be non-empty");
    if (max_new_tokens < 0) throw std::runtime_error("max_new_tokens must be >= 0");

    auto t_load_start = std::chrono::steady_clock::now();
    const int tp_world = std::max(1, options.tp_world);
    const int tp_rank = std::max(0, options.tp_rank);
    const int tp_device = options.device >= 0 ? options.device : tp_rank;
    if (tp_rank >= tp_world) throw std::runtime_error("run_gguf_generate_smoke: invalid TP rank");
    if (tp_world > 1 && options.nccl_id_path.empty())
        throw std::runtime_error("run_gguf_generate_smoke: TP requires --nccl-id-path");
    GgufForwardContext ctx(ckpt_path);
    GGUFWeightSource& gws = *ctx.weight_source;

    // ===== optional GGUF mmap pinning for async H2D out of expert bytes =====
    // Do NOT register the whole GGUF mmap by default. Large file-backed
    // cudaHostRegister() calls can make the kernel account the entire mapping as
    // dirty/pinned; with TP4 an ~86 GiB GGUF becomes ~344 GiB per run and can
    // quickly throttle unrelated git/build/source writes via balance_dirty_pages.
    // Resident-routed decode no longer needs this for its hot path, so keep it as
    // an explicit debug/legacy staging knob with a conservative size guard.
    void* gguf_base = const_cast<uint8_t*>(gws.file().bytes());
    const size_t gguf_bytes = gws.file().file_size();
    bool gguf_registered = false;
    const bool gguf_register_mmap = env_int_or_default("DSV4_GGUF_REGISTER_MMAP", 0) != 0;
    const int gguf_register_mmap_max_gib = env_int_or_default("DSV4_GGUF_REGISTER_MMAP_MAX_GIB", 4);
    const size_t gguf_register_mmap_max_bytes =
        gguf_register_mmap_max_gib <= 0 ? 0 : static_cast<size_t>(gguf_register_mmap_max_gib) * 1024ULL * 1024ULL * 1024ULL;
    if (gguf_register_mmap && gguf_register_mmap_max_bytes > 0 && gguf_bytes <= gguf_register_mmap_max_bytes) {
        if (cudaHostRegister(gguf_base, gguf_bytes, cudaHostRegisterReadOnly) == cudaSuccess) {
            gguf_registered = true;
        } else if (cudaHostRegister(gguf_base, gguf_bytes, cudaHostRegisterDefault) == cudaSuccess) {
            gguf_registered = true;
        } else {
            // Clear the error and continue with the pageable path; correctness
            // is unaffected, only bandwidth.
            cudaGetLastError();
        }
    }

    // ===== model dims =====
    const int n_layers = static_cast<int>(ctx.config.n_layers);
    const int n_hash = static_cast<int>(ctx.config.n_hash_layers);
    const int dim = static_cast<int>(ctx.config.dim);
    const int global_heads = static_cast<int>(ctx.config.n_heads);
    const int n_experts = static_cast<int>(ctx.config.n_routed_experts);
    const int topk = static_cast<int>(ctx.config.n_activated_experts);
    const int rope_dim = static_cast<int>(ctx.config.rope_dim);
    const int global_o_groups = static_cast<int>(ctx.config.o_groups);
    const int o_lora_rank = static_cast<int>(ctx.config.o_lora_rank);
    const int moe_inter = static_cast<int>(ctx.config.moe_inter_dim);
    if (topk <= 0 || topk > 8) throw std::runtime_error("topk out of supported range");
    if ((global_heads % tp_world) != 0 || (global_o_groups % tp_world) != 0)
        throw std::runtime_error("GGUF TP requires heads and o_groups divisible by tp_world");

    WeightView wq_a0 = gws.require("layers.0.attn.wq_a.weight");
    WeightView wq_b0 = gws.require("layers.0.attn.wq_b.weight");
    WeightView wkv0 = gws.require("layers.0.attn.wkv.weight");
    const int q_a_dim = static_cast<int>(wq_a0.shape[1]);
    const int global_q_full = static_cast<int>(wq_b0.shape[1]);
    if (global_q_full % global_heads != 0) throw std::runtime_error("wq_b cols % heads");
    const int head_dim = global_q_full / global_heads;
    const int heads = global_heads / tp_world;
    const int q_full = heads * head_dim;
    const int kv_dim = static_cast<int>(wkv0.shape[1]);
    const int o_groups = global_o_groups / tp_world;
    if (global_q_full % global_o_groups != 0) throw std::runtime_error("q_full % o_groups");
    const int group_in_dim = q_full / o_groups;
    const int attn_mid = o_groups * o_lora_rank;
    const int local_head_start = tp_rank * heads;
    const int local_q_row_start = local_head_start * head_dim;
    const int local_group_start = tp_rank * o_groups;
    const int local_wo_a_row_start = local_group_start * o_lora_rank;

    WeightView embed = gws.require("embed.weight");
    if (embed.dtype != DType::F16 || embed.shape.size() != 2 || static_cast<int>(embed.shape[0]) != dim)
        throw std::runtime_error("embed shape/dtype mismatch");
    const int vocab = static_cast<int>(embed.shape[1]);
    if (vocab % tp_world != 0) throw std::runtime_error("GGUF TP requires vocab divisible by tp_world");
    const int local_vocab = vocab / tp_world;
    const int local_vocab_start = tp_rank * local_vocab;
    const int experts_per_rank = tp_world > 1 ? n_experts / tp_world : n_experts;
    if ((n_experts % tp_world) != 0) throw std::runtime_error("GGUF TP requires experts divisible by tp_world");
    const int expert_start = tp_rank * experts_per_rank;
    const int expert_end = tp_world > 1 ? expert_start + experts_per_rank : n_experts;
    for (int t : seed_tokens) if (t < 0 || t >= vocab)
        throw std::runtime_error("seed token id out of vocab range");

    const int gguf_min_free_mib = env_int_or_default("DSV4_GGUF_MIN_FREE_MIB", 512);
    const bool gguf_mem_profile = env_int_or_default("DSV4_GGUF_MEM_PROFILE", 1) != 0;
    const int gguf_indexed_attn = gguf_indexed_attn_mode_from_env();
    const int gguf_indexed_attn_window = env_int_or_default(
        "DSV4_GGUF_INDEXED_ATTN_WINDOW",
        static_cast<int>(ctx.config.window_size == 0 ? 128 : ctx.config.window_size));
    const int gguf_indexed_attn_auto_threshold = env_int_or_default(
        "DSV4_GGUF_INDEXED_ATTN_AUTO_THRESHOLD", gguf_indexed_attn_window);
    const bool gguf_sparse_compressor = env_int_or_default("DSV4_GGUF_SPARSE_COMPRESSOR", 0) != 0;
    // Optional indexer top-k for compressed slots. Default off for now: the main
    // compressor + all compressed slots already cuts 32K attention scans by ~4x,
    // while the GGUF indexer matvec path is not optimized yet.
    const bool gguf_sparse_indexer = env_int_or_default("DSV4_GGUF_SPARSE_INDEXER", 0) != 0;
    const int gguf_sparse_window = env_int_or_default(
        "DSV4_GGUF_SPARSE_WINDOW",
        static_cast<int>(ctx.config.window_size == 0 ? 128 : ctx.config.window_size));
    const int gguf_sparse_attn_threshold = env_int_or_default(
        "DSV4_GGUF_SPARSE_ATTN_THRESHOLD", gguf_sparse_window);
    if (gguf_indexed_attn < 0 || gguf_indexed_attn > 2)
        throw std::runtime_error("DSV4_GGUF_INDEXED_ATTN must be 0, 1, 2, or auto");
    if (gguf_indexed_attn != 0 && gguf_indexed_attn_window <= 0)
        throw std::runtime_error("DSV4_GGUF_INDEXED_ATTN_WINDOW must be > 0");
    if (gguf_indexed_attn == 2 && gguf_indexed_attn_auto_threshold <= 0)
        throw std::runtime_error("DSV4_GGUF_INDEXED_ATTN_AUTO_THRESHOLD must be > 0");
    if (gguf_sparse_compressor && gguf_sparse_window <= 0)
        throw std::runtime_error("DSV4_GGUF_SPARSE_WINDOW must be > 0");
    if (gguf_sparse_compressor && (gguf_sparse_attn_threshold < 0 || gguf_sparse_attn_threshold > gguf_sparse_window))
        throw std::runtime_error("DSV4_GGUF_SPARSE_ATTN_THRESHOLD must be in [0, DSV4_GGUF_SPARSE_WINDOW]");
    if (gguf_mem_profile) gguf_log_mem("after_context_init", tp_rank);

    auto upload_u8 = [](const void* src, size_t bytes) {
        uint8_t* d = nullptr;
        check_cuda(cudaMalloc(&d, bytes), "alloc u8");
        check_cuda(cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice), "copy u8");
        return d;
    };
    auto upload_f32 = [](const void* src, size_t n) {
        float* d = nullptr;
        check_cuda(cudaMalloc(&d, n * sizeof(float)), "alloc f32");
        check_cuda(cudaMemcpy(d, src, n * sizeof(float), cudaMemcpyHostToDevice), "copy f32");
        return d;
    };
    auto upload_bf16_from_f32 = [&](const void* src, size_t n) {
        auto bf = f32_to_bf16_host(reinterpret_cast<const float*>(src), n);
        uint16_t* d = nullptr;
        check_cuda(cudaMalloc(&d, n * sizeof(uint16_t)), "alloc bf16");
        check_cuda(cudaMemcpy(d, bf.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy bf16");
        return d;
    };
    auto upload_bf16_from_f16 = [&](const void* src, size_t n) {
        auto bf = f16_to_bf16_host(reinterpret_cast<const uint16_t*>(src), n);
        uint16_t* d = nullptr;
        check_cuda(cudaMalloc(&d, n * sizeof(uint16_t)), "alloc bf16-from-f16");
        check_cuda(cudaMemcpy(d, bf.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy bf16-from-f16");
        return d;
    };
    auto upload_f32_from_f16 = [&](const void* src, size_t n) {
        auto f32 = f16_to_f32_host(reinterpret_cast<const uint16_t*>(src), n);
        float* d = nullptr;
        check_cuda(cudaMalloc(&d, n * sizeof(float)), "alloc f32-from-f16");
        check_cuda(cudaMemcpy(d, f32.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy f32-from-f16");
        return d;
    };

    // ===== per-layer dense weights resident on device =====
    std::vector<uint16_t*> d_attn_gamma(n_layers, nullptr);
    std::vector<uint16_t*> d_q_gamma(n_layers, nullptr);
    std::vector<uint16_t*> d_kv_gamma(n_layers, nullptr);
    std::vector<uint16_t*> d_ffn_gamma(n_layers, nullptr);
    std::vector<float*> d_attn_sink(n_layers, nullptr);
    std::vector<uint8_t*> d_wq_a(n_layers, nullptr);
    std::vector<uint8_t*> d_wq_b(n_layers, nullptr);
    std::vector<uint8_t*> d_wkv(n_layers, nullptr);
    std::vector<uint8_t*> d_wo_a(n_layers, nullptr);
    std::vector<uint8_t*> d_wo_b(n_layers, nullptr);
    std::vector<uint8_t*> d_shared_w1(n_layers, nullptr);
    std::vector<uint8_t*> d_shared_w2(n_layers, nullptr);
    std::vector<uint8_t*> d_shared_w3(n_layers, nullptr);
    std::vector<uint16_t*> d_gate_w(n_layers, nullptr);
    std::vector<float*> d_gate_bias(n_layers, nullptr);
    std::vector<GgufCompressorDeviceWeights> d_compressor_w(n_layers);
    std::vector<GgufIndexerDeviceWeights> d_indexer_w(n_layers);
    std::vector<GgufSparseLayerState> d_sparse_state(n_layers);
    std::vector<const int32_t*> h_tid2eid_table(n_layers, nullptr);
    std::vector<int64_t*> d_tid2eid_table(n_layers, nullptr);
    const bool hash_table_resident = env_int_or_default("DSV4_GGUF_HASH_TABLE_RESIDENT", 1) != 0;

    for (int L = 0; L < n_layers; ++L) {
        const std::string lp = "layers." + std::to_string(L);
        d_attn_gamma[L] = upload_bf16_from_f32(gws.require(lp + ".attn_norm.weight").data, dim);
        d_q_gamma[L] = upload_bf16_from_f32(gws.require(lp + ".attn.q_norm.weight").data, q_a_dim);
        d_kv_gamma[L] = upload_bf16_from_f32(gws.require(lp + ".attn.kv_norm.weight").data, kv_dim);
        d_ffn_gamma[L] = upload_bf16_from_f32(gws.require(lp + ".ffn_norm.weight").data, dim);
        d_attn_sink[L] = upload_f32(reinterpret_cast<const float*>(gws.require(lp + ".attn.attn_sink").data) + local_head_start, heads);
        WeightView wq_a = gws.require(lp + ".attn.wq_a.weight");
        WeightView wq_b = gws.require(lp + ".attn.wq_b.weight");
        WeightView wkv = gws.require(lp + ".attn.wkv.weight");
        WeightView wo_a = gws.require(lp + ".attn.wo_a.weight");
        WeightView wo_b = gws.require(lp + ".attn.wo_b.weight");
        WeightView shared_w1 = gws.require(lp + ".ffn.shared_experts.w1.weight");
        WeightView shared_w2 = gws.require(lp + ".ffn.shared_experts.w2.weight");
        WeightView shared_w3 = gws.require(lp + ".ffn.shared_experts.w3.weight");
        WeightView gate_w = gws.require(lp + ".ffn.gate.weight");
        auto wq_b_local = slice_q8_0_rows(wq_b.data, local_q_row_start, q_full, q_a_dim);
        auto wo_a_local = slice_q8_0_rows(wo_a.data, local_wo_a_row_start, attn_mid, group_in_dim);
        auto wo_b_local = slice_q8_0_cols(wo_b.data, dim, global_o_groups * o_lora_rank,
                                          local_wo_a_row_start, attn_mid);
        d_wq_a[L] = upload_u8(wq_a.data, wq_a.nbytes);
        d_wq_b[L] = upload_u8(wq_b_local.data(), wq_b_local.size());
        d_wkv[L] = upload_u8(wkv.data, wkv.nbytes);
        d_wo_a[L] = upload_u8(wo_a_local.data(), wo_a_local.size());
        d_wo_b[L] = upload_u8(wo_b_local.data(), wo_b_local.size());
        d_shared_w1[L] = upload_u8(shared_w1.data, shared_w1.nbytes);
        d_shared_w2[L] = upload_u8(shared_w2.data, shared_w2.nbytes);
        d_shared_w3[L] = upload_u8(shared_w3.data, shared_w3.nbytes);
        d_gate_w[L] = upload_bf16_from_f16(gate_w.data,
                                            static_cast<size_t>(dim) * static_cast<size_t>(n_experts));
        if (L < n_hash) {
            WeightView tid2eid = gws.require(lp + ".ffn.gate.tid2eid");
            if (tid2eid.nbytes < static_cast<uint64_t>(vocab) * topk * sizeof(int32_t))
                throw std::runtime_error("tid2eid table truncated");
            h_tid2eid_table[L] = reinterpret_cast<const int32_t*>(tid2eid.data);
            if (hash_table_resident) {
                const size_t entries = static_cast<size_t>(vocab) * static_cast<size_t>(topk);
                std::vector<int64_t> tid2eid_i64(entries);
                const int32_t* src = h_tid2eid_table[L];
                for (size_t i = 0; i < entries; ++i) tid2eid_i64[i] = src[i];
                check_cuda(cudaMalloc(&d_tid2eid_table[L], entries * sizeof(int64_t)),
                           "alloc resident tid2eid table");
                check_cuda(cudaMemcpy(d_tid2eid_table[L], tid2eid_i64.data(),
                                      entries * sizeof(int64_t), cudaMemcpyHostToDevice),
                           "copy resident tid2eid table");
            }
        } else {
            WeightView bias = gws.require(lp + ".ffn.gate.bias");
            if (bias.dtype != DType::F32 || bias.shape.size() != 1 ||
                static_cast<int>(bias.shape[0]) != n_experts)
                throw std::runtime_error("exp_probs_b.bias shape/dtype");
            d_gate_bias[L] = upload_f32(bias.data, n_experts);
        }
        if (gguf_sparse_compressor && L < static_cast<int>(ctx.config.compress_ratios.size()) && ctx.config.compress_ratios[L] > 0) {
            auto load_comp = [&](const std::string& prefix, int ratio, int state_head_dim) {
                GgufCompressorDeviceWeights c;
                WeightView wkv_c = gws.get(prefix + ".wkv.weight");
                WeightView wgate_c = gws.get(prefix + ".wgate.weight");
                WeightView ape_c = gws.get(prefix + ".ape");
                WeightView norm_c = gws.get(prefix + ".norm.weight");
                if (!wkv_c.found || !wgate_c.found || !ape_c.found || !norm_c.found) return c;
                if (wkv_c.dtype != DType::F16 || wgate_c.dtype != DType::F16 || ape_c.dtype != DType::F16 || norm_c.dtype != DType::F32)
                    throw std::runtime_error("GGUF compressor dtype mismatch");
                if (wkv_c.shape.size() != 2 || static_cast<int>(wkv_c.shape[0]) != dim || wgate_c.shape != wkv_c.shape)
                    throw std::runtime_error("GGUF compressor shape mismatch");
                c.cols = static_cast<int>(wkv_c.shape[1]); // GGUF raw [dim, cols], logical direct data [cols, dim].
                c.ratio = ratio;
                c.overlap = (c.cols == state_head_dim * 2);
                c.state_cols = c.overlap ? state_head_dim * 2 : state_head_dim;
                c.slots = ratio * (c.overlap ? 2 : 1);
                c.wkv = upload_bf16_from_f16(wkv_c.data, static_cast<size_t>(c.cols) * static_cast<size_t>(dim));
                c.wgate = upload_bf16_from_f16(wgate_c.data, static_cast<size_t>(c.cols) * static_cast<size_t>(dim));
                c.ape = upload_f32_from_f16(ape_c.data, static_cast<size_t>(ape_c.shape[0]) * static_cast<size_t>(ape_c.shape[1]));
                c.norm = upload_bf16_from_f32(norm_c.data, static_cast<size_t>(state_head_dim));
                c.present = true;
                return c;
            };
            const int ratio = static_cast<int>(ctx.config.compress_ratios[L]);
            d_compressor_w[L] = load_comp(lp + ".attn.compressor", ratio, head_dim);
            if (gguf_sparse_indexer && ratio == 4 && ctx.config.index_n_heads > 0 && ctx.config.index_head_dim > 0) {
                WeightView idx_wq_b = gws.get(lp + ".attn.indexer.wq_b.weight");
                WeightView idx_proj = gws.get(lp + ".attn.indexer.weights_proj.weight");
                if (idx_wq_b.found && idx_proj.found) {
                    if (idx_wq_b.dtype != DType::F16 || idx_proj.dtype != DType::F16)
                        throw std::runtime_error("GGUF indexer dtype mismatch");
                    const int idx_heads = static_cast<int>(ctx.config.index_n_heads);
                    const int idx_head_dim = static_cast<int>(ctx.config.index_head_dim);
                    const int idx_q_dim = idx_heads * idx_head_dim;
                    if (idx_wq_b.shape.size() != 2 || static_cast<int>(idx_wq_b.shape[0]) != q_a_dim || static_cast<int>(idx_wq_b.shape[1]) != idx_q_dim)
                        throw std::runtime_error("GGUF indexer wq_b shape mismatch");
                    if (idx_proj.shape.size() != 2 || static_cast<int>(idx_proj.shape[0]) != dim || static_cast<int>(idx_proj.shape[1]) != idx_heads)
                        throw std::runtime_error("GGUF indexer proj shape mismatch");
                    d_indexer_w[L].heads = idx_heads;
                    d_indexer_w[L].head_dim = idx_head_dim;
                    d_indexer_w[L].wq_b = upload_bf16_from_f16(idx_wq_b.data, static_cast<size_t>(idx_q_dim) * static_cast<size_t>(q_a_dim));
                    d_indexer_w[L].weights_proj = upload_bf16_from_f16(idx_proj.data, static_cast<size_t>(idx_heads) * static_cast<size_t>(dim));
                    d_indexer_w[L].comp = load_comp(lp + ".attn.indexer.compressor", 4, idx_head_dim);
                    d_indexer_w[L].present = d_indexer_w[L].comp.present;
                }
            }
        }
    }
    if (gguf_mem_profile) gguf_log_mem("after_dense_weights", tp_rank);
    gguf_check_min_free("after_dense_weights", tp_rank, gguf_min_free_mib);
    if (hash_table_resident && gguf_mem_profile) {
        const double hash_table_gib = static_cast<double>(n_hash) *
            static_cast<double>(vocab) * static_cast<double>(topk) *
            static_cast<double>(sizeof(int64_t)) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "gguf_hash_table_resident=1 tp_rank=" << tp_rank
                  << " n_hash=" << n_hash
                  << " table_topk=" << topk
                  << " hash_table_gib=" << hash_table_gib
                  << "\n";
    }

    WeightView final_norm = gws.require("norm.weight");
    WeightView head = gws.require("head.weight");
    if (final_norm.dtype != DType::F32 || static_cast<int>(final_norm.shape[0]) != dim)
        throw std::runtime_error("norm.weight shape/dtype");
    if (head.dtype != DType::Q8_0 || static_cast<int>(head.shape[0]) != dim ||
        static_cast<int>(head.shape[1]) != vocab)
        throw std::runtime_error("head.weight shape/dtype (expected Q8_0 [dim, vocab])");
    uint16_t* d_final_norm_gamma = upload_bf16_from_f32(final_norm.data, dim);
    auto head_local = slice_q8_0_rows(head.data, local_vocab_start, local_vocab, dim);
    uint8_t* d_head = upload_u8(head_local.data(), head_local.size());

    // ===== routed expert staging / resident buffers =====
    // Dynamic IQ1_M GGUF changes W2 dtype/byte-size after early layers, so size
    // the reusable staging arena by the maximum per-layer expert byte count and
    // dispatch kernels from each layer's exact dtype recipe.
    std::vector<uint64_t> layer_w1_bytes(n_layers, 0);
    std::vector<uint64_t> layer_w2_bytes(n_layers, 0);
    std::vector<uint64_t> layer_w3_bytes(n_layers, 0);
    std::vector<DType> layer_w1_dtype(n_layers, DType::Unknown);
    std::vector<DType> layer_w2_dtype(n_layers, DType::Unknown);
    std::vector<DType> layer_w3_dtype(n_layers, DType::Unknown);
    uint64_t max_per_w1_bytes = 0;
    uint64_t max_per_w2_bytes = 0;
    uint64_t max_per_w3_bytes = 0;
    for (int L = 0; L < n_layers; ++L) {
        const std::string lp = "layers." + std::to_string(L);
        const std::string w1_name = lp + ".ffn.experts.routed.w1";
        const std::string w2_name = lp + ".ffn.experts.routed.w2";
        const std::string w3_name = lp + ".ffn.experts.routed.w3";
        auto f1 = gws.get_expert(w1_name, w1_name, 0);
        auto f2 = gws.get_expert(w2_name, w2_name, 0);
        auto f3 = gws.get_expert(w3_name, w3_name, 0);
        if (!f1.found || !f2.found || !f3.found)
            throw std::runtime_error("get_expert(0) failed while sizing routed buffers");
        const bool q2_recipe = f1.dtype == DType::IQ2_XXS && f3.dtype == DType::IQ2_XXS && f2.dtype == DType::Q2_K;
        const bool iq1_all_recipe = f1.dtype == DType::IQ1_M && f3.dtype == DType::IQ1_M && f2.dtype == DType::IQ1_M;
        const bool iq1_w13_q2_w2_recipe = f1.dtype == DType::IQ1_M && f3.dtype == DType::IQ1_M && f2.dtype == DType::Q2_K;
        if (!q2_recipe && !iq1_all_recipe && !iq1_w13_q2_w2_recipe) {
            throw std::runtime_error("unsupported routed dtype recipe at layer " + std::to_string(L) +
                                     ": w1=" + dtype_name(f1.dtype) + " w2=" + dtype_name(f2.dtype) +
                                     " w3=" + dtype_name(f3.dtype));
        }
        layer_w1_bytes[L] = f1.nbytes;
        layer_w2_bytes[L] = f2.nbytes;
        layer_w3_bytes[L] = f3.nbytes;
        layer_w1_dtype[L] = f1.dtype;
        layer_w2_dtype[L] = f2.dtype;
        layer_w3_dtype[L] = f3.dtype;
        max_per_w1_bytes = std::max(max_per_w1_bytes, f1.nbytes);
        max_per_w2_bytes = std::max(max_per_w2_bytes, f2.nbytes);
        max_per_w3_bytes = std::max(max_per_w3_bytes, f3.nbytes);
    }
    uint8_t* d_routed_w1 = nullptr;
    uint8_t* d_routed_w2 = nullptr;
    uint8_t* d_routed_w3 = nullptr;
    const int routed_stage_slots = std::max(topk, experts_per_rank);
    check_cuda(cudaMalloc(&d_routed_w1, max_per_w1_bytes * routed_stage_slots), "alloc routed_w1");
    check_cuda(cudaMalloc(&d_routed_w2, max_per_w2_bytes * routed_stage_slots), "alloc routed_w2");
    check_cuda(cudaMalloc(&d_routed_w3, max_per_w3_bytes * routed_stage_slots), "alloc routed_w3");

    // Optional resident local routed experts. The legacy path stages active top-k
    // experts every token/layer from the GGUF mmap; with TP=4 that H2D staging
    // dominates decode latency. Resident mode uploads this rank's local expert
    // slice once and maps route_slots to local expert ordinals, while still
    // consuming the original raw quantized blocks.
    const bool has_iq1_routed = std::any_of(
        layer_w1_dtype.begin(), layer_w1_dtype.end(),
        [](DType t) { return t == DType::IQ1_M; });
    const int resident_layers_env = env_int_or_default("DSV4_GGUF_RESIDENT_ROUTED_LAYERS", -1);
    const bool resident_routed_requested = tp_world > 1 && (
        env_int_or_default("DSV4_GGUF_RESIDENT_ROUTED_EXPERTS", has_iq1_routed ? 1 : 0) != 0 ||
        resident_layers_env > 0);
    const int resident_routed_layers = resident_routed_requested
        ? std::max(0, std::min(n_layers, resident_layers_env >= 0 ? resident_layers_env : n_layers))
        : 0;
    std::vector<uint8_t> is_resident_routed_layer(static_cast<size_t>(n_layers), 0);
    for (int L = 0; L < resident_routed_layers; ++L) is_resident_routed_layer[L] = 1;
    const bool any_resident_routed = resident_routed_layers > 0;
    const bool all_resident_routed = resident_routed_layers == n_layers;
    const bool device_route_slots = any_resident_routed &&
        env_int_or_default("DSV4_GGUF_DEVICE_ROUTE_SLOTS", 1) != 0;
    GgufPinnedStagingRing gguf_pinned_stage;
    const bool gguf_pinned_stage_enabled = env_int_or_default("DSV4_GGUF_PINNED_STAGE", 0) != 0;
    if (gguf_pinned_stage_enabled && !all_resident_routed) {
        const size_t max_stage_copy_bytes = std::max<size_t>(
            static_cast<size_t>(std::max(max_per_w1_bytes, std::max(max_per_w2_bytes, max_per_w3_bytes))),
            static_cast<size_t>(topk) * sizeof(int64_t));
        gguf_pinned_stage.init(max_stage_copy_bytes,
                               env_int_or_default("DSV4_GGUF_PINNED_STAGE_SLOTS", 32),
                               env_int_or_default("DSV4_GGUF_PINNED_STAGE_CAP_MIB", 256),
                               tp_rank);
    }
    std::vector<uint8_t*> d_resident_w1(n_layers, nullptr);
    std::vector<uint8_t*> d_resident_w2(n_layers, nullptr);
    std::vector<uint8_t*> d_resident_w3(n_layers, nullptr);
    uint64_t resident_bytes = 0;
    if (any_resident_routed) {
        for (int L = 0; L < n_layers; ++L) {
            if (!is_resident_routed_layer[L]) continue;
            const std::string lp = "layers." + std::to_string(L);
            const std::string w1_name = lp + ".ffn.experts.routed.w1";
            const std::string w2_name = lp + ".ffn.experts.routed.w2";
            const std::string w3_name = lp + ".ffn.experts.routed.w3";
            const size_t w1_layer_bytes = static_cast<size_t>(layer_w1_bytes[L]) * experts_per_rank;
            const size_t w2_layer_bytes = static_cast<size_t>(layer_w2_bytes[L]) * experts_per_rank;
            const size_t w3_layer_bytes = static_cast<size_t>(layer_w3_bytes[L]) * experts_per_rank;
            check_cuda(cudaMalloc(&d_resident_w1[L], w1_layer_bytes), "alloc resident routed_w1");
            check_cuda(cudaMalloc(&d_resident_w2[L], w2_layer_bytes), "alloc resident routed_w2");
            check_cuda(cudaMalloc(&d_resident_w3[L], w3_layer_bytes), "alloc resident routed_w3");
            resident_bytes += static_cast<uint64_t>(w1_layer_bytes + w2_layer_bytes + w3_layer_bytes);
            for (int local_e = 0; local_e < experts_per_rank; ++local_e) {
                const int eid = expert_start + local_e;
                auto wv1 = gws.get_expert(w1_name, w1_name, eid);
                auto wv2 = gws.get_expert(w2_name, w2_name, eid);
                auto wv3 = gws.get_expert(w3_name, w3_name, eid);
                if (!wv1.found || !wv2.found || !wv3.found)
                    throw std::runtime_error("get_expert failed during resident routed load");
                if (wv1.dtype != layer_w1_dtype[L] || wv2.dtype != layer_w2_dtype[L] || wv3.dtype != layer_w3_dtype[L])
                    throw std::runtime_error("routed expert dtype changed during resident load");
                check_cuda(cudaMemcpy(d_resident_w1[L] + layer_w1_bytes[L] * local_e,
                                      wv1.data, layer_w1_bytes[L], cudaMemcpyHostToDevice),
                           "copy resident routed_w1");
                check_cuda(cudaMemcpy(d_resident_w2[L] + layer_w2_bytes[L] * local_e,
                                      wv2.data, layer_w2_bytes[L], cudaMemcpyHostToDevice),
                           "copy resident routed_w2");
                check_cuda(cudaMemcpy(d_resident_w3[L] + layer_w3_bytes[L] * local_e,
                                      wv3.data, layer_w3_bytes[L], cudaMemcpyHostToDevice),
                           "copy resident routed_w3");
            }
        }
        size_t free_bytes = 0, total_bytes = 0;
        check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo resident routed");
        std::cout << "gguf_resident_routed_experts=1 tp_rank=" << tp_rank
                  << " layers=" << resident_routed_layers << "/" << n_layers
                  << " local_experts=" << experts_per_rank
                  << " resident_gib=" << (static_cast<double>(resident_bytes) / (1024.0 * 1024.0 * 1024.0))
                  << " free_gib=" << (static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0))
                  << " total_gib=" << (static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0))
                  << "\n";
    }
    if (gguf_mem_profile) gguf_log_mem("after_resident_experts", tp_rank);
    gguf_check_min_free("after_resident_experts", tp_rank, gguf_min_free_mib);

    // ===== per-layer KV cache sized for full sequence =====
    // To generate K tokens from an N-token seed, positions 0..N-1 are the
    // prefill/TTFT pass and positions N..N+K-2 are continuation decode passes.
    const int decode_only_context = env_int_or_default("DSV4_GGUF_DECODE_ONLY_CONTEXT", 0);
    if (decode_only_context < 0) throw std::runtime_error("DSV4_GGUF_DECODE_ONLY_CONTEXT must be >= 0");
    const int generate_total_positions = static_cast<int>(seed_tokens.size()) + std::max(0, max_new_tokens - 1);
    const int total_positions = std::max(generate_total_positions, decode_only_context > 0 ? decode_only_context : 0);
    int max_sparse_compressed_cap = 0;
    if (gguf_sparse_compressor) {
        for (int L = 0; L < n_layers; ++L) {
            if (!d_compressor_w[L].present || d_compressor_w[L].ratio <= 0) continue;
            max_sparse_compressed_cap = std::max(
                max_sparse_compressed_cap,
                (std::max(1, total_positions) + d_compressor_w[L].ratio - 1) / d_compressor_w[L].ratio);
        }
    }
    std::vector<int> layer_cache_capacity(n_layers, std::max(1, total_positions));
    int max_layer_cache_capacity = std::max(1, total_positions);
    if (gguf_sparse_compressor) {
        max_layer_cache_capacity = 1;
        for (int L = 0; L < n_layers; ++L) {
            if (d_compressor_w[L].present && d_compressor_w[L].ratio > 0) {
                const int compressed_cap =
                    (std::max(1, total_positions) + d_compressor_w[L].ratio - 1) /
                    d_compressor_w[L].ratio;
                layer_cache_capacity[L] = std::max(1, gguf_sparse_window + std::max(1, compressed_cap));
            } else {
                // Layers without compressor still run dense cached attention and
                // need direct position-addressable KV storage.
                layer_cache_capacity[L] = std::max(1, total_positions);
            }
            max_layer_cache_capacity = std::max(max_layer_cache_capacity, layer_cache_capacity[L]);
        }
    }
    const int cache_capacity = max_layer_cache_capacity;
    std::vector<float*> d_kv_cache(n_layers, nullptr);
    for (int L = 0; L < n_layers; ++L) {
        check_cuda(cudaMalloc(&d_kv_cache[L],
                              static_cast<size_t>(layer_cache_capacity[L]) *
                                  static_cast<size_t>(kv_dim) * sizeof(float)),
                   "alloc d_kv_cache");
        check_cuda(cudaMemset(d_kv_cache[L], 0,
                              static_cast<size_t>(layer_cache_capacity[L]) *
                                  static_cast<size_t>(kv_dim) * sizeof(float)),
                   "memset d_kv_cache");
    }
    if (gguf_sparse_compressor) {
        for (int L = 0; L < n_layers; ++L) {
            const auto& c = d_compressor_w[L];
            if (!c.present) continue;
            auto init_score = [](float* ptr, size_t n, const char* label) {
                std::vector<float> neg_inf(n, -INFINITY);
                check_cuda(cudaMemcpy(ptr, neg_inf.data(), n * sizeof(float), cudaMemcpyHostToDevice), label);
            };
            check_cuda(cudaMalloc(&d_sparse_state[L].compressor_kv,
                                  static_cast<size_t>(c.slots) * c.state_cols * sizeof(float)),
                       "alloc GGUF sparse compressor kv state");
            check_cuda(cudaMalloc(&d_sparse_state[L].compressor_score,
                                  static_cast<size_t>(c.slots) * c.state_cols * sizeof(float)),
                       "alloc GGUF sparse compressor score state");
            check_cuda(cudaMemset(d_sparse_state[L].compressor_kv, 0,
                                  static_cast<size_t>(c.slots) * c.state_cols * sizeof(float)),
                       "zero GGUF sparse compressor kv state");
            init_score(d_sparse_state[L].compressor_score,
                       static_cast<size_t>(c.slots) * c.state_cols,
                       "init GGUF sparse compressor score state");
            if (d_indexer_w[L].present && d_indexer_w[L].comp.present) {
                const auto& ic = d_indexer_w[L].comp;
                check_cuda(cudaMalloc(&d_sparse_state[L].indexer_comp_kv,
                                      static_cast<size_t>(ic.slots) * ic.state_cols * sizeof(float)),
                           "alloc GGUF sparse indexer compressor kv state");
                check_cuda(cudaMalloc(&d_sparse_state[L].indexer_comp_score,
                                      static_cast<size_t>(ic.slots) * ic.state_cols * sizeof(float)),
                           "alloc GGUF sparse indexer compressor score state");
                check_cuda(cudaMemset(d_sparse_state[L].indexer_comp_kv, 0,
                                      static_cast<size_t>(ic.slots) * ic.state_cols * sizeof(float)),
                           "zero GGUF sparse indexer compressor kv state");
                init_score(d_sparse_state[L].indexer_comp_score,
                           static_cast<size_t>(ic.slots) * ic.state_cols,
                           "init GGUF sparse indexer compressor score state");
                const int indexer_cache_cap = std::max(1, max_sparse_compressed_cap);
                check_cuda(cudaMalloc(&d_sparse_state[L].indexer_kv_cache,
                                      static_cast<size_t>(indexer_cache_cap) *
                                          static_cast<size_t>(d_indexer_w[L].head_dim) * sizeof(float)),
                           "alloc GGUF sparse indexer kv cache");
                check_cuda(cudaMemset(d_sparse_state[L].indexer_kv_cache, 0,
                                      static_cast<size_t>(indexer_cache_cap) *
                                          static_cast<size_t>(d_indexer_w[L].head_dim) * sizeof(float)),
                           "zero GGUF sparse indexer kv cache");
            }
        }
    }
    uint64_t kv_cache_elems = 0;
    for (int cap : layer_cache_capacity) {
        kv_cache_elems += static_cast<uint64_t>(cap) * static_cast<uint64_t>(kv_dim);
    }
    const double kv_cache_gib = static_cast<double>(kv_cache_elems) *
        static_cast<double>(sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    if (gguf_mem_profile) {
        int sparse_layers = 0;
        for (int L = 0; L < n_layers; ++L) if (d_compressor_w[L].present) ++sparse_layers;
        std::cout << "gguf_kv_cache tp_rank=" << tp_rank
                  << " max_capacity=" << cache_capacity
                  << " sparse_layers=" << sparse_layers
                  << " n_layers=" << n_layers
                  << " kv_dim=" << kv_dim
                  << " kv_cache_gib=" << kv_cache_gib
                  << "\n";
        gguf_log_mem("after_kv_cache", tp_rank);
    }
    gguf_check_min_free("after_kv_cache", tp_rank, gguf_min_free_mib);

    // Cached attention scratch. The legacy cached attention kernel stores
    // per-token softmax logits in dynamic shared memory, which cannot scale to
    // 32K context. The workspace variant keeps the same dense attention math but
    // uses one global [heads, cache_capacity] scratch buffer reused by all layers.
    float* d_attn_weight_scratch = nullptr;
    if (gguf_indexed_attn != 1) {
        check_cuda(cudaMalloc(&d_attn_weight_scratch,
                              static_cast<size_t>(heads) * static_cast<size_t>(cache_capacity) * sizeof(float)),
                   "alloc d_attn_weight_scratch");
    }
    int* d_kv_indices = nullptr;
    if (gguf_indexed_attn != 0 || gguf_sparse_compressor) {
        const int max_kv_index_count = gguf_sparse_compressor
            ? std::min(cache_capacity, gguf_sparse_window + std::max(1, max_sparse_compressed_cap))
            : std::min(cache_capacity, gguf_indexed_attn_window);
        check_cuda(cudaMalloc(&d_kv_indices, static_cast<size_t>(max_kv_index_count) * sizeof(int)),
                   "alloc d_kv_indices");
    }
    if (gguf_mem_profile && tp_rank == 0) {
        std::cout << "gguf_indexed_attn mode=" << gguf_indexed_attn
                  << " window=" << gguf_indexed_attn_window
                  << " auto_threshold=" << gguf_indexed_attn_auto_threshold
                  << " sparse_compressor=" << (gguf_sparse_compressor ? 1 : 0)
                  << " sparse_indexer=" << (gguf_sparse_indexer ? 1 : 0)
                  << " sparse_window=" << gguf_sparse_window
                  << " sparse_attn_threshold=" << gguf_sparse_attn_threshold
                  << " decode_only_context=" << decode_only_context
                  << " cache_capacity=" << cache_capacity
                  << "\n";
    }

    // ===== activation + scratch buffers (one set, reused per step + per layer) =====
    const int x_groups = (dim + 31) / 32;
    const int hidden_groups = (moe_inter + 15) / 16;
    const int gate_scratch_n = std::max(topk, n_experts);
    float* d_x = nullptr;
    float* d_x_pre_attn = nullptr;
    float* d_x_normed = nullptr;
    float* d_q_a = nullptr;
    float* d_q_normed = nullptr;
    float* d_q = nullptr;
    float* d_kv_a = nullptr;
    float* d_kv = nullptr;
    float* d_attn_value = nullptr;
    float* d_attn_mid = nullptr;
    float* d_attn_out = nullptr;
    float* d_x_pre_ffn = nullptr;
    float* d_x_normed_ffn = nullptr;
    float* d_shared_gate = nullptr;
    float* d_shared_up = nullptr;
    float* d_shared_hidden = nullptr;
    float* d_shared_out = nullptr;
    float* d_moe_out = nullptr;
    float* d_ffn_combined = nullptr;
    int8_t* d_x_q = nullptr;
    float* d_x_scale = nullptr;
    int64_t* d_route_slots = nullptr;
    float* d_route_weights = nullptr;
    float* d_route_gate = nullptr;
    float* d_route_up = nullptr;
    int8_t* d_route_hidden_q = nullptr;
    float* d_route_hidden_scale = nullptr;
    float* d_route_hidden = nullptr;
    float* d_gate_scores_scratch = nullptr;
    float* d_gate_scored_scratch = nullptr;
    int64_t* d_gate_indices = nullptr;
    int64_t* d_tid2eid_i64 = nullptr;
    uint16_t* d_embed_row_f16 = nullptr;
    uint16_t* d_compressor_input_bf16 = nullptr;
    float* d_compressor_input_rounded = nullptr;
    float* d_compressor_kv = nullptr;
    float* d_compressor_score = nullptr;
    float* d_indexer_comp_kv = nullptr;
    float* d_indexer_comp_score = nullptr;
    float* d_index_q = nullptr;
    float* d_index_scores = nullptr;
    float* d_logits = nullptr;
    int* d_argmax_token = nullptr;
    float* d_argmax_logit = nullptr;
    check_cuda(cudaMalloc(&d_x, dim * sizeof(float)), "alloc d_x");
    check_cuda(cudaMalloc(&d_x_pre_attn, dim * sizeof(float)), "alloc d_x_pre_attn");
    check_cuda(cudaMalloc(&d_x_normed, dim * sizeof(float)), "alloc d_x_normed");
    check_cuda(cudaMalloc(&d_q_a, q_a_dim * sizeof(float)), "alloc d_q_a");
    check_cuda(cudaMalloc(&d_q_normed, q_a_dim * sizeof(float)), "alloc d_q_normed");
    check_cuda(cudaMalloc(&d_q, q_full * sizeof(float)), "alloc d_q");
    check_cuda(cudaMalloc(&d_kv_a, kv_dim * sizeof(float)), "alloc d_kv_a");
    check_cuda(cudaMalloc(&d_kv, kv_dim * sizeof(float)), "alloc d_kv");
    check_cuda(cudaMalloc(&d_attn_value, q_full * sizeof(float)), "alloc d_attn_value");
    check_cuda(cudaMalloc(&d_attn_mid, attn_mid * sizeof(float)), "alloc d_attn_mid");
    check_cuda(cudaMalloc(&d_attn_out, dim * sizeof(float)), "alloc d_attn_out");
    check_cuda(cudaMalloc(&d_x_pre_ffn, dim * sizeof(float)), "alloc d_x_pre_ffn");
    check_cuda(cudaMalloc(&d_x_normed_ffn, dim * sizeof(float)), "alloc d_x_normed_ffn");
    check_cuda(cudaMalloc(&d_shared_gate, moe_inter * sizeof(float)), "alloc d_shared_gate");
    check_cuda(cudaMalloc(&d_shared_up, moe_inter * sizeof(float)), "alloc d_shared_up");
    check_cuda(cudaMalloc(&d_shared_hidden, moe_inter * sizeof(float)), "alloc d_shared_hidden");
    check_cuda(cudaMalloc(&d_shared_out, dim * sizeof(float)), "alloc d_shared_out");
    check_cuda(cudaMalloc(&d_moe_out, dim * sizeof(float)), "alloc d_moe_out");
    check_cuda(cudaMalloc(&d_ffn_combined, dim * sizeof(float)), "alloc d_ffn_combined");
    check_cuda(cudaMalloc(&d_x_q, dim), "alloc d_x_q");
    check_cuda(cudaMalloc(&d_x_scale, x_groups * sizeof(float)), "alloc d_x_scale");
    check_cuda(cudaMalloc(&d_route_slots, topk * sizeof(int64_t)), "alloc d_route_slots");
    check_cuda(cudaMalloc(&d_route_weights, topk * sizeof(float)), "alloc d_route_weights");
    check_cuda(cudaMalloc(&d_route_gate, static_cast<size_t>(topk) * moe_inter * sizeof(float)),
               "alloc d_route_gate");
    check_cuda(cudaMalloc(&d_route_up, static_cast<size_t>(topk) * moe_inter * sizeof(float)),
               "alloc d_route_up");
    check_cuda(cudaMalloc(&d_route_hidden_q, static_cast<size_t>(topk) * moe_inter),
               "alloc d_route_hidden_q");
    check_cuda(cudaMalloc(&d_route_hidden_scale,
                          static_cast<size_t>(topk) * hidden_groups * sizeof(float)),
               "alloc d_route_hidden_scale");
    check_cuda(cudaMalloc(&d_route_hidden, static_cast<size_t>(topk) * moe_inter * sizeof(float)),
               "alloc d_route_hidden");
    check_cuda(cudaMalloc(&d_gate_scores_scratch, gate_scratch_n * sizeof(float)),
               "alloc d_gate_scores_scratch");
    check_cuda(cudaMalloc(&d_gate_scored_scratch, n_experts * sizeof(float)),
               "alloc d_gate_scored_scratch");
    check_cuda(cudaMalloc(&d_gate_indices, topk * sizeof(int64_t)), "alloc d_gate_indices");
    check_cuda(cudaMalloc(&d_tid2eid_i64, topk * sizeof(int64_t)), "alloc d_tid2eid_i64");
    check_cuda(cudaMalloc(&d_embed_row_f16, dim * sizeof(uint16_t)), "alloc d_embed_row_f16");
    if (gguf_sparse_compressor) {
        int max_comp_cols = 1;
        int max_index_q_dim = 1;
        int max_index_heads = 1;
        for (int L = 0; L < n_layers; ++L) {
            if (d_compressor_w[L].present) max_comp_cols = std::max(max_comp_cols, d_compressor_w[L].cols);
            if (d_indexer_w[L].present) {
                max_comp_cols = std::max(max_comp_cols, d_indexer_w[L].comp.cols);
                max_index_q_dim = std::max(max_index_q_dim, d_indexer_w[L].heads * d_indexer_w[L].head_dim);
                max_index_heads = std::max(max_index_heads, d_indexer_w[L].heads);
            }
        }
        check_cuda(cudaMalloc(&d_compressor_input_bf16, dim * sizeof(uint16_t)),
                   "alloc GGUF sparse compressor input bf16");
        check_cuda(cudaMalloc(&d_compressor_input_rounded, dim * sizeof(float)),
                   "alloc GGUF sparse compressor input rounded");
        check_cuda(cudaMalloc(&d_compressor_kv, static_cast<size_t>(max_comp_cols) * sizeof(float)),
                   "alloc GGUF sparse compressor kv scratch");
        check_cuda(cudaMalloc(&d_compressor_score, static_cast<size_t>(max_comp_cols) * sizeof(float)),
                   "alloc GGUF sparse compressor score scratch");
        check_cuda(cudaMalloc(&d_indexer_comp_kv, static_cast<size_t>(max_comp_cols) * sizeof(float)),
                   "alloc GGUF sparse indexer compressor kv scratch");
        check_cuda(cudaMalloc(&d_indexer_comp_score, static_cast<size_t>(max_comp_cols) * sizeof(float)),
                   "alloc GGUF sparse indexer compressor score scratch");
        check_cuda(cudaMalloc(&d_index_q, static_cast<size_t>(max_index_q_dim) * sizeof(float)),
                   "alloc GGUF sparse index q scratch");
        check_cuda(cudaMalloc(&d_index_scores,
                              static_cast<size_t>(max_index_heads + std::max(1, max_sparse_compressed_cap)) * sizeof(float)),
                   "alloc GGUF sparse index score scratch");
    }
    check_cuda(cudaMalloc(&d_logits, local_vocab * sizeof(float)), "alloc d_logits");
    check_cuda(cudaMalloc(&d_argmax_token, sizeof(int)), "alloc d_argmax_token");
    check_cuda(cudaMalloc(&d_argmax_logit, sizeof(float)), "alloc d_argmax_logit");
    if (gguf_mem_profile) gguf_log_mem("after_scratch", tp_rank);
    gguf_check_min_free("after_scratch", tp_rank, gguf_min_free_mib);

    // Dedicated copy stream for staging (H2D for routed Q2 experts). Currently
    // neutral in wall-time (staging is on the critical path between gate D2H
    // and MoE compute; nothing on default stream can overlap with it within a
    // layer) but kept so future cross-step prefetch can issue early stages on
    // copy_stream while the prior step's MoE drains on default.
    const bool moe_copy_stream_enabled =
        env_int_or_default("DSV4_GGUF_MOE_COPY_STREAM", 1) != 0;
    cudaStream_t gguf_moe_copy_stream = nullptr;
    cudaEvent_t gguf_moe_stage_event = nullptr;
    cudaEvent_t gguf_moe_consume_event = nullptr;
    if (moe_copy_stream_enabled) {
        check_cuda(cudaStreamCreateWithFlags(&gguf_moe_copy_stream, cudaStreamNonBlocking),
                   "create gguf moe copy stream");
        check_cuda(cudaEventCreateWithFlags(&gguf_moe_stage_event, cudaEventDisableTiming),
                   "create gguf moe stage event");
        check_cuda(cudaEventCreateWithFlags(&gguf_moe_consume_event, cudaEventDisableTiming),
                   "create gguf moe consume event");
        check_cuda(cudaEventRecord(gguf_moe_consume_event, nullptr),
                   "record initial gguf moe consume event");
    }

#ifdef DSV4_HAVE_NCCL
    BF16AllReduceScratch bf16_reduce_scratch;
#endif
    GgufReduceContext reduce_ctx;
    reduce_ctx.world = tp_world;
    reduce_ctx.rank = tp_rank;
    reduce_ctx.device = tp_device;
    reduce_ctx.id_path = options.nccl_id_path.empty() ? nullptr : options.nccl_id_path.c_str();
#ifdef DSV4_HAVE_NCCL
    reduce_ctx.scratch = &bf16_reduce_scratch;
#endif
    const GgufReduceContext* reduce_ctx_ptr = (tp_world > 1) ? &reduce_ctx : nullptr;

    GgufLayerDims ld;
    ld.dim = dim; ld.q_a_dim = q_a_dim; ld.heads = heads; ld.head_dim = head_dim;
    ld.q_full = q_full; ld.kv_dim = kv_dim; ld.rope_dim = rope_dim;
    ld.o_groups = o_groups; ld.o_lora_rank = o_lora_rank; ld.group_in_dim = group_in_dim;
    ld.attn_mid = attn_mid; ld.moe_inter = moe_inter;
    ld.n_experts = n_experts; ld.topk = topk;
    ld.rope_theta = static_cast<float>(ctx.config.rope_theta);
    ld.swiglu_limit = static_cast<float>(ctx.config.swiglu_limit);
    ld.route_scale = static_cast<float>(ctx.config.route_scale);

    GgufLayerScratch ls;
    ls.d_x = d_x;
    ls.d_x_pre_attn = d_x_pre_attn; ls.d_x_normed = d_x_normed;
    ls.d_q_a = d_q_a; ls.d_q_normed = d_q_normed; ls.d_q = d_q;
    ls.d_kv_a = d_kv_a; ls.d_kv = d_kv;
    ls.d_attn_value = d_attn_value; ls.d_attn_mid = d_attn_mid; ls.d_attn_out = d_attn_out;
    ls.d_x_pre_ffn = d_x_pre_ffn; ls.d_x_normed_ffn = d_x_normed_ffn;
    ls.d_shared_gate = d_shared_gate; ls.d_shared_up = d_shared_up;
    ls.d_shared_hidden = d_shared_hidden; ls.d_shared_out = d_shared_out;
    ls.d_moe_out = d_moe_out; ls.d_ffn_combined = d_ffn_combined;
    ls.d_x_q = d_x_q; ls.d_x_scale = d_x_scale;
    ls.d_route_slots = d_route_slots; ls.d_route_weights = d_route_weights;
    ls.d_route_gate = d_route_gate; ls.d_route_up = d_route_up;
    ls.d_route_hidden_q = d_route_hidden_q; ls.d_route_hidden_scale = d_route_hidden_scale;
    ls.d_route_hidden = d_route_hidden;
    ls.d_gate_scores_scratch = d_gate_scores_scratch;
    ls.d_gate_scored_scratch = d_gate_scored_scratch;
    ls.d_gate_indices = d_gate_indices;
    ls.cache_capacity = cache_capacity;
    ls.d_compressor_input_bf16 = d_compressor_input_bf16;
    ls.d_compressor_input_rounded = d_compressor_input_rounded;
    ls.d_compressor_kv = d_compressor_kv;
    ls.d_compressor_score = d_compressor_score;
    ls.d_indexer_comp_kv = d_indexer_comp_kv;
    ls.d_indexer_comp_score = d_indexer_comp_score;
    ls.d_index_q = d_index_q;
    ls.d_index_scores = d_index_scores;

    auto t_load_end = std::chrono::steady_clock::now();
    const bool gguf_load_only = env_int_or_default("DSV4_GGUF_LOAD_ONLY", 0) != 0;

    // ===== per-step forward (embed + 43 layers + final norm + head + argmax) =====
    const bool profile_gguf = env_int_or_default("DSV4_GGUF_PROFILE", 0) != 0;
    const bool gguf_host_logits = env_int_or_default("DSV4_GGUF_HOST_LOGITS", 0) != 0;
    const bool decode_only_attention_only =
        decode_only_context > 0 && env_int_or_default("DSV4_GGUF_DECODE_ONLY_ATTENTION_ONLY", 0) != 0;
    if (gguf_mem_profile && tp_rank == 0 && decode_only_context > 0) {
        std::cout << "gguf_decode_only context=" << decode_only_context
                  << " attention_only=" << (decode_only_attention_only ? 1 : 0)
                  << " warmup=" << env_int_or_default("DSV4_GGUF_DECODE_ONLY_WARMUP", all_resident_routed ? 0 : 1)
                  << "\n";
    }
    double prof_attn_ms = 0.0, prof_stage_ms = 0.0, prof_moe_ms = 0.0;
    double prof_embed_ms = 0.0, prof_head_ms = 0.0;
    long long prof_indexed_attn_steps = 0;
    long long prof_indexed_attn_indices = 0;
    long long prof_sparse_attn_layer_steps = 0;
    long long prof_sparse_attn_indices = 0;
    int prof_steps = 0;
    auto sync_now = [&]() {
        if (profile_gguf) check_cuda(cudaDeviceSynchronize(), "profile sync");
        return std::chrono::steady_clock::now();
    };
    std::vector<int64_t> h_gate_indices(topk);
    std::vector<int64_t> h_route_slots(topk);
    std::vector<float> h_logits(local_vocab);
    std::vector<int> debug_logit_tokens;
    if (const char* dbg = std::getenv("DSV4_GGUF_DEBUG_LOGIT_TOKENS")) {
        std::stringstream ss(dbg);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) debug_logit_tokens.push_back(std::stoi(item));
        }
    }
    auto debug_print_logits = [&](const char* tag, int position, int top_t, float top_l) {
        if (debug_logit_tokens.empty()) return;
        const int final_prompt_pos = static_cast<int>(seed_tokens.size()) - 1;
        if (position != final_prompt_pos) return;
        check_cuda(cudaDeviceSynchronize(), "sync GGUF debug logits");
        std::ostringstream oss;
        oss << "gguf_debug_logits tag=" << tag << " rank=" << tp_rank
            << " position=" << position << " top=" << top_t << ":" << top_l;
        for (int tok : debug_logit_tokens) {
            oss << " token" << tok << "=";
            if (tok >= local_vocab_start && tok < local_vocab_start + local_vocab) {
                float v = -INFINITY;
                check_cuda(cudaMemcpy(&v, d_logits + (tok - local_vocab_start), sizeof(float), cudaMemcpyDeviceToHost), "copy GGUF debug logit");
                oss << v;
            } else {
                oss << "NA";
            }
        }
        std::cout << oss.str() << "\n";
    };
    auto run_step = [&](int token, int position) -> std::pair<int, float> {
        auto t_embed_start = sync_now();
        const uint16_t* host_embed_row =
            reinterpret_cast<const uint16_t*>(embed.data) +
            static_cast<size_t>(token) * dim;
        check_cuda(cudaMemcpy(d_embed_row_f16, host_embed_row, dim * sizeof(uint16_t),
                              cudaMemcpyHostToDevice), "copy embed row");
        if (!f16_row_to_float_cuda(d_embed_row_f16, d_x, /*row=*/0, dim))
            throw std::runtime_error("f16 embed failed");
        auto t_embed_end = sync_now();
        if (profile_gguf) prof_embed_ms += elapsed_ms(t_embed_start, t_embed_end);
        int gguf_kv_index_count = 0;
        const bool use_indexed_attn_for_step =
            (gguf_indexed_attn == 1) ||
            (gguf_indexed_attn == 2 && (position + 1) > gguf_indexed_attn_auto_threshold);
        if (d_kv_indices != nullptr && use_indexed_attn_for_step) {
            gguf_kv_index_count = std::min(position + 1, gguf_indexed_attn_window);
            const int window_start = std::max(0, position - gguf_kv_index_count + 1);
            if (!build_decode_kv_indices_cuda(d_kv_indices, window_start, gguf_kv_index_count,
                                              cache_capacity, /*compressed_count=*/0,
                                              /*compressed_offset=*/cache_capacity))
                throw std::runtime_error("build GGUF decode kv indices failed");
            if (profile_gguf) {
                ++prof_indexed_attn_steps;
                prof_indexed_attn_indices += gguf_kv_index_count;
            }
        }
        for (int L = 0; L < n_layers; ++L) {
            auto t_layer_start = sync_now();
            const bool is_hash = (L < n_hash);
            GgufLayerDeviceWeights lw;
            lw.d_attn_gamma = d_attn_gamma[L]; lw.d_q_gamma = d_q_gamma[L]; lw.d_kv_gamma = d_kv_gamma[L];
            lw.d_wq_a = d_wq_a[L]; lw.d_wq_b = d_wq_b[L]; lw.d_wkv = d_wkv[L];
            lw.d_wo_a = d_wo_a[L]; lw.d_wo_b = d_wo_b[L];
            lw.d_attn_sink = d_attn_sink[L];
            lw.d_ffn_gamma = d_ffn_gamma[L];
            lw.d_shared_w1 = d_shared_w1[L]; lw.d_shared_w2 = d_shared_w2[L]; lw.d_shared_w3 = d_shared_w3[L];
            lw.d_gate_w_bf16 = d_gate_w[L];
            lw.is_hash = is_hash;
            const bool layer_resident_routed = is_resident_routed_layer[L] != 0;
            if (layer_resident_routed) {
                lw.d_routed_w1 = d_resident_w1[L];
                lw.d_routed_w2 = d_resident_w2[L];
                lw.d_routed_w3 = d_resident_w3[L];
                lw.routed_n_experts = experts_per_rank;
            } else {
                lw.d_routed_w1 = d_routed_w1;
                lw.d_routed_w2 = d_routed_w2;
                lw.d_routed_w3 = d_routed_w3;
                lw.routed_n_experts = topk;
            }
            lw.routed_w1_dtype = layer_w1_dtype[L];
            lw.routed_w2_dtype = layer_w2_dtype[L];
            lw.routed_w3_dtype = layer_w3_dtype[L];
            const bool layer_sparse_enabled = gguf_sparse_compressor && d_compressor_w[L].present;
            lw.compressor = layer_sparse_enabled ? &d_compressor_w[L] : nullptr;
            lw.indexer = (layer_sparse_enabled && d_indexer_w[L].present) ? &d_indexer_w[L] : nullptr;
            lw.sparse_state = (layer_sparse_enabled && d_sparse_state[L].compressor_kv != nullptr) ? &d_sparse_state[L] : nullptr;
            lw.compress_ratio = layer_sparse_enabled ? d_compressor_w[L].ratio : 0;
            lw.sparse_window_size = layer_sparse_enabled ? gguf_sparse_window : 0;
            lw.sparse_attn_threshold = layer_sparse_enabled ? gguf_sparse_attn_threshold : 0;
            lw.index_topk = static_cast<int>(ctx.config.index_topk == 0 ? 0 : ctx.config.index_topk);
            ld.rope_theta = static_cast<float>(ctx.config.rope_theta);
            if (gguf_sparse_compressor) {
                ls.d_kv_indices = d_kv_indices;
                ls.kv_index_count = 0;
            } else if (d_kv_indices != nullptr && use_indexed_attn_for_step) {
                ls.d_kv_indices = d_kv_indices;
                ls.kv_index_count = gguf_kv_index_count;
            } else {
                ls.d_kv_indices = nullptr;
                ls.kv_index_count = 0;
            }
            ls.d_attn_weight_scratch = d_attn_weight_scratch;
            ls.d_kv_cache = d_kv_cache[L];
            ls.cache_capacity = layer_cache_capacity[L];
            const std::string lp = "layers." + std::to_string(L);
            const std::string w1_name = lp + ".ffn.experts.routed.w1";
            const std::string w2_name = lp + ".ffn.experts.routed.w2";
            const std::string w3_name = lp + ".ffn.experts.routed.w3";
            bool hash_prestaged = false;
            if (is_hash) {
                const int32_t* row = h_tid2eid_table[L] + static_cast<size_t>(token) * topk;
                int64_t h_slice[8];
                for (int k = 0; k < topk; ++k) h_slice[k] = row[k];
                if (d_tid2eid_table[L] != nullptr) {
                    lw.d_tid2eid_i64 = d_tid2eid_table[L];
                    lw.hash_token = token;
                    lw.hash_table_topk = topk;
                } else {
                    check_cuda(cudaMemcpy(d_tid2eid_i64, h_slice, topk * sizeof(int64_t),
                                          cudaMemcpyHostToDevice), "copy tid2eid slice");
                    lw.d_tid2eid_i64 = d_tid2eid_i64;
                    lw.hash_token = 0;
                    lw.hash_table_topk = topk;
                }
                lw.d_gate_bias_f32 = nullptr;
                // Hash-layer expert ids come straight from the deterministic
                // tid2eid table -- no dependency on the gate matvec. In resident
                // mode only tiny route metadata is needed; legacy mode pre-stages
                // the routed experts on copy_stream so H2D overlaps with attention.
                if (layer_resident_routed) {
                    for (int k = 0; k < topk; ++k) {
                        const int eid = static_cast<int>(h_slice[k]);
                        if (eid < 0 || eid >= n_experts)
                            throw std::runtime_error("hash table produced out-of-range expert id");
                        const bool is_local = (eid >= expert_start && eid < expert_end);
                        h_route_slots[k] = is_local ? static_cast<int64_t>(eid - expert_start) : -1;
                    }
                } else if (!decode_only_attention_only && moe_copy_stream_enabled) {
                    check_cuda(cudaStreamWaitEvent(gguf_moe_copy_stream, gguf_moe_consume_event, 0),
                               "copy stream wait prior GGUF MoE consume event");
                    for (int k = 0; k < topk; ++k) {
                        const int eid = static_cast<int>(h_slice[k]);
                        if (eid < 0 || eid >= n_experts)
                            throw std::runtime_error("hash table produced out-of-range expert id");
                        const bool is_local = (eid >= expert_start && eid < expert_end);
                        h_route_slots[k] = is_local ? static_cast<int64_t>(k) : -1;
                        if (!is_local) continue;
                        auto wv1 = gws.get_expert(w1_name, w1_name, eid);
                        auto wv2 = gws.get_expert(w2_name, w2_name, eid);
                        auto wv3 = gws.get_expert(w3_name, w3_name, eid);
                        if (!wv1.found || !wv2.found || !wv3.found)
                            throw std::runtime_error("get_expert failed during hash prestaging");
                        if (wv1.dtype != layer_w1_dtype[L] || wv2.dtype != layer_w2_dtype[L] || wv3.dtype != layer_w3_dtype[L])
                            throw std::runtime_error("routed expert dtype changed within hash layer");
                        gguf_pinned_stage.copy_async(d_routed_w1 + layer_w1_bytes[L] * k,
                                                     wv1.data, layer_w1_bytes[L],
                                                     gguf_moe_copy_stream, "prestage routed_w1");
                        gguf_pinned_stage.copy_async(d_routed_w2 + layer_w2_bytes[L] * k,
                                                     wv2.data, layer_w2_bytes[L],
                                                     gguf_moe_copy_stream, "prestage routed_w2");
                        gguf_pinned_stage.copy_async(d_routed_w3 + layer_w3_bytes[L] * k,
                                                     wv3.data, layer_w3_bytes[L],
                                                     gguf_moe_copy_stream, "prestage routed_w3");
                    }
                    gguf_pinned_stage.copy_async(reinterpret_cast<uint8_t*>(d_route_slots),
                                                 reinterpret_cast<const uint8_t*>(h_route_slots.data()),
                                                 static_cast<size_t>(topk) * sizeof(int64_t),
                                                 gguf_moe_copy_stream,
                                                 "prestage d_route_slots");
                    check_cuda(cudaEventRecord(gguf_moe_stage_event, gguf_moe_copy_stream),
                               "record gguf moe stage event (hash prestage)");
                    hash_prestaged = true;
                }
            } else {
                lw.d_tid2eid_i64 = nullptr;
                lw.d_gate_bias_f32 = d_gate_bias[L];
            }
            gguf_layer_forward_attn_to_gate(lw, ls, ld, position, reduce_ctx_ptr);
            if (profile_gguf && ls.sparse_attn_used) {
                ++prof_sparse_attn_layer_steps;
                prof_sparse_attn_indices += ls.sparse_kv_index_count;
            }
            auto t_attn_end = sync_now();
            if (profile_gguf) prof_attn_ms += elapsed_ms(t_layer_start, t_attn_end);
            if (!decode_only_attention_only) {
                if (!hash_prestaged) {
                    if (layer_resident_routed && device_route_slots) {
                        if (!gguf_route_slots_from_indices_cuda(d_gate_indices, d_route_slots,
                                                                topk, expert_start, experts_per_rank))
                            throw std::runtime_error("gguf_route_slots_from_indices_cuda failed");
                    } else {
                        check_cuda(cudaMemcpy(h_gate_indices.data(), d_gate_indices, topk * sizeof(int64_t),
                                              cudaMemcpyDeviceToHost), "copy gate indices");
                        for (int k = 0; k < topk; ++k) {
                            const int eid = static_cast<int>(h_gate_indices[k]);
                            if (eid < 0 || eid >= n_experts)
                                throw std::runtime_error("gate produced out-of-range expert id");
                            const bool is_local = (eid >= expert_start && eid < expert_end);
                            h_route_slots[k] = is_local
                                ? static_cast<int64_t>(layer_resident_routed ? (eid - expert_start) : k)
                                : -1;
                            if (layer_resident_routed || !is_local) continue;
                            if (moe_copy_stream_enabled) {
                                check_cuda(cudaStreamWaitEvent(gguf_moe_copy_stream, gguf_moe_consume_event, 0),
                                           "copy stream wait prior GGUF MoE consume event");
                            }
                            auto wv1 = gws.get_expert(w1_name, w1_name, eid);
                            auto wv2 = gws.get_expert(w2_name, w2_name, eid);
                            auto wv3 = gws.get_expert(w3_name, w3_name, eid);
                            if (!wv1.found || !wv2.found || !wv3.found)
                                throw std::runtime_error("get_expert failed during decode staging");
                            if (wv1.dtype != layer_w1_dtype[L] || wv2.dtype != layer_w2_dtype[L] || wv3.dtype != layer_w3_dtype[L])
                                throw std::runtime_error("routed expert dtype changed within layer");
                            gguf_pinned_stage.copy_async(d_routed_w1 + layer_w1_bytes[L] * k,
                                                         wv1.data, layer_w1_bytes[L],
                                                         gguf_moe_copy_stream, "stage routed_w1");
                            gguf_pinned_stage.copy_async(d_routed_w2 + layer_w2_bytes[L] * k,
                                                         wv2.data, layer_w2_bytes[L],
                                                         gguf_moe_copy_stream, "stage routed_w2");
                            gguf_pinned_stage.copy_async(d_routed_w3 + layer_w3_bytes[L] * k,
                                                         wv3.data, layer_w3_bytes[L],
                                                         gguf_moe_copy_stream, "stage routed_w3");
                        }
                        if (layer_resident_routed) {
                            check_cuda(cudaMemcpy(d_route_slots, h_route_slots.data(),
                                                  topk * sizeof(int64_t),
                                                  cudaMemcpyHostToDevice),
                                       "copy resident d_route_slots");
                        } else {
                            if (moe_copy_stream_enabled) {
                                check_cuda(cudaStreamWaitEvent(gguf_moe_copy_stream, gguf_moe_consume_event, 0),
                                           "copy stream wait prior GGUF route-slot consume event");
                            }
                            gguf_pinned_stage.copy_async(reinterpret_cast<uint8_t*>(d_route_slots),
                                                         reinterpret_cast<const uint8_t*>(h_route_slots.data()),
                                                         static_cast<size_t>(topk) * sizeof(int64_t),
                                                         gguf_moe_copy_stream,
                                                         "copy d_route_slots");
                            if (moe_copy_stream_enabled) {
                                check_cuda(cudaEventRecord(gguf_moe_stage_event, gguf_moe_copy_stream),
                                           "record gguf moe stage event");
                            }
                        }
                    }
                }
                if (moe_copy_stream_enabled && !layer_resident_routed) {
                    check_cuda(cudaStreamWaitEvent(nullptr, gguf_moe_stage_event, 0),
                               "default stream wait stage event");
                }
            }
            auto t_stage_end = sync_now();
            if (profile_gguf) prof_stage_ms += elapsed_ms(t_attn_end, t_stage_end);
            if (!decode_only_attention_only) {
                gguf_layer_forward_moe(lw, ls, ld, reduce_ctx_ptr);
                if (moe_copy_stream_enabled) {
                    check_cuda(cudaEventRecord(gguf_moe_consume_event, nullptr),
                               "record GGUF MoE consume event");
                }
            }
            auto t_moe_end = sync_now();
            if (profile_gguf && !decode_only_attention_only) prof_moe_ms += elapsed_ms(t_stage_end, t_moe_end);
        }
        auto t_head_start = sync_now();
        if (!rmsnorm_bf16_gamma_cuda(d_x, d_final_norm_gamma, d_x_normed, dim, 1e-6f))
            throw std::runtime_error("final norm failed");
        if (!q8_0_matvec_cuda(d_x_normed, d_head, d_logits, local_vocab, dim))
            throw std::runtime_error("head matvec failed");
        int top_t = local_vocab_start;
        float top_l = -INFINITY;
        if (gguf_host_logits) {
            check_cuda(cudaDeviceSynchronize(), "sync after head");
            check_cuda(cudaMemcpy(h_logits.data(), d_logits, local_vocab * sizeof(float),
                                  cudaMemcpyDeviceToHost), "copy logits");
            int local_t = 0;
            float local_l = -INFINITY;
            for (int i = 0; i < local_vocab; ++i) {
                if (h_logits[i] > local_l) { local_l = h_logits[i]; local_t = i; }
            }
            top_t = local_t + local_vocab_start;
            top_l = local_l;
        } else {
            if (!argmax_fp32_cuda(d_logits, d_argmax_token, d_argmax_logit,
                                  local_vocab, local_vocab_start))
                throw std::runtime_error("argmax_fp32_cuda failed");
            check_cuda(cudaMemcpy(&top_t, d_argmax_token, sizeof(int),
                                  cudaMemcpyDeviceToHost), "copy argmax token");
            check_cuda(cudaMemcpy(&top_l, d_argmax_logit, sizeof(float),
                                  cudaMemcpyDeviceToHost), "copy argmax logit");
        }
#ifdef DSV4_HAVE_NCCL
        if (tp_world > 1 && !options.nccl_id_path.empty()) {
            TpTopResult global = nccl_global_top1(tp_world, tp_rank, tp_device,
                                                   options.nccl_id_path.c_str(),
                                                   top_t, top_l);
            top_t = global.token;
            top_l = global.logit;
        }
#endif
        if (profile_gguf) {
            auto t_head_end = std::chrono::steady_clock::now();
            prof_head_ms += elapsed_ms(t_head_start, t_head_end);
            ++prof_steps;
        }
        return {top_t, top_l};
    };

    const bool gguf_chunked_prefill_requested = env_int_or_default("DSV4_GGUF_CHUNKED_PREFILL", 0) != 0;
    const bool gguf_chunked_prefill_available =
        gguf_chunked_prefill_requested &&
        decode_only_context == 0 &&
        max_new_tokens > 0 &&
        !gguf_load_only &&
        !gguf_sparse_compressor &&
        gguf_indexed_attn == 0 &&
        hash_table_resident;
    if (gguf_chunked_prefill_requested && !gguf_chunked_prefill_available && tp_rank == 0) {
        const char* reason = "unknown";
        if (decode_only_context != 0) reason = "decode_only";
        else if (max_new_tokens <= 0) reason = "no_generation";
        else if (gguf_load_only) reason = "load_only";
        else if (gguf_sparse_compressor) reason = "sparse_compressor_not_supported";
        else if (gguf_indexed_attn != 0) reason = "indexed_attention_not_supported";
        else if (!hash_table_resident) reason = "requires_resident_hash_table";
        std::cout << "gguf_chunked_prefill=0 reason=" << reason << "\n";
    }

    auto run_chunked_prefill = [&]() -> std::pair<int, float> {
        const int prompt_tokens = static_cast<int>(seed_tokens.size());
        int chunk_tokens = env_int_or_default("DSV4_GGUF_PREFILL_CHUNK_TOKENS", 512);
        if (chunk_tokens <= 0 || chunk_tokens > prompt_tokens) chunk_tokens = prompt_tokens;
        const int chunk_alloc = chunk_tokens;
        const int routes_cap = chunk_alloc * topk;
        const size_t chunk_dim = static_cast<size_t>(chunk_alloc) * dim;
        const size_t chunk_q_a = static_cast<size_t>(chunk_alloc) * q_a_dim;
        const size_t chunk_q = static_cast<size_t>(chunk_alloc) * q_full;
        const size_t chunk_kv = static_cast<size_t>(chunk_alloc) * kv_dim;
        const size_t chunk_attn_mid = static_cast<size_t>(chunk_alloc) * attn_mid;
        const size_t chunk_inter = static_cast<size_t>(chunk_alloc) * moe_inter;
        const size_t routes_dim = static_cast<size_t>(routes_cap) * dim;
        const size_t routes_inter = static_cast<size_t>(routes_cap) * moe_inter;
        int prefill_attn_window = env_int_or_default("DSV4_GGUF_PREFILL_ATTN_WINDOW", prompt_tokens);
        if (prefill_attn_window <= 0 || prefill_attn_window > prompt_tokens) prefill_attn_window = prompt_tokens;
        const int prefill_attn_topk = std::min(prompt_tokens, prefill_attn_window);
        const int64_t prefill_attn_index_elems = static_cast<int64_t>(prompt_tokens) * prefill_attn_topk;
        const int64_t prefill_attn_max_index_elems = static_cast<int64_t>(env_int_or_default("DSV4_GGUF_PREFILL_ATTN_MAX_INDEX_ELEMS", 64 * 1024 * 1024));
        const int prefill_attn_max_index_topk = env_int_or_default("DSV4_GGUF_PREFILL_INDEXED_ATTN_MAX_TOPK", 8192);
        const bool use_prefill_indexed_attn = env_int_or_default("DSV4_GGUF_PREFILL_INDEXED_ATTN", 0) != 0 &&
            prefill_attn_topk > 0 && prefill_attn_index_elems > 0 &&
            (prefill_attn_max_index_topk <= 0 || prefill_attn_topk <= prefill_attn_max_index_topk) &&
            (prefill_attn_max_index_elems <= 0 || prefill_attn_index_elems <= prefill_attn_max_index_elems);
        const bool iq1_grouped_w2_q8_enabled = env_int_or_default("DSV4_IQ1_GROUPED_W2_Q8", 1) != 0;
        const bool iq1_grouped_gemm_enabled = env_int_or_default("DSV4_IQ1_GROUPED_GEMM", 1) != 0;
        const bool iq1_grouped_route_tile4_enabled = env_int_or_default("DSV4_IQ1_GROUPED_ROUTE_TILE4", 0) != 0;
        const bool iq1_grouped_route_major_enabled = !iq1_grouped_route_tile4_enabled && env_int_or_default("DSV4_IQ1_GROUPED_ROUTE_MAJOR", 1) != 0;
        const int iq1_x_groups16 = (dim + 15) / 16;

        std::vector<uint16_t> h_embed_chunk(static_cast<size_t>(chunk_alloc) * dim);
        uint16_t* d_embed_chunk_f16 = nullptr;
        int* d_prefill_token_ids = nullptr;
        float* d_state_rows = nullptr;
        float* d_attn_norm_rows = nullptr;
        float* d_q_a_rows = nullptr;
        float* d_q_norm_rows = nullptr;
        float* d_q_rows = nullptr;
        float* d_kv_a_rows = nullptr;
        float* d_kv_rows = nullptr;
        float* d_attn_value_rows = nullptr;
        float* d_attn_mid_rows = nullptr;
        float* d_attn_out_rows = nullptr;
        float* d_ffn_norm_rows = nullptr;
        float* d_shared_gate_rows = nullptr;
        float* d_shared_up_rows = nullptr;
        float* d_shared_hidden_rows = nullptr;
        float* d_shared_out_rows = nullptr;
        float* d_moe_rows = nullptr;
        int64_t* d_prefill_route_indices = nullptr;
        float* d_prefill_route_weights = nullptr;
        int64_t* d_group_route_tokens = nullptr;
        float* d_group_route_weights = nullptr;
        int32_t* d_seg_starts = nullptr;
        int32_t* d_counts = nullptr;
        int32_t* d_offsets = nullptr;
        int32_t* d_total_routes = nullptr;
        int32_t* d_prefill_attn_indices = nullptr;
        float* d_final_norm_rows = nullptr;

        MoePrefillQ2GroupedWorkspace q2_ws;
        MoePrefillIq1GroupedWorkspace iq1_ws;
        bool need_q2_ws = false;
        bool need_iq1_ws = false;
        bool need_iq1_q2_ws = false;
        bool need_iq1_gemm_ws = false;
        for (int L = 0; L < n_layers; ++L) {
            const bool q2_recipe = layer_w1_dtype[L] == DType::IQ2_XXS && layer_w3_dtype[L] == DType::IQ2_XXS && layer_w2_dtype[L] == DType::Q2_K;
            const bool iq1_all_recipe = layer_w1_dtype[L] == DType::IQ1_M && layer_w3_dtype[L] == DType::IQ1_M && layer_w2_dtype[L] == DType::IQ1_M;
            const bool iq1_w13_q2_w2_recipe = layer_w1_dtype[L] == DType::IQ1_M && layer_w3_dtype[L] == DType::IQ1_M && layer_w2_dtype[L] == DType::Q2_K;
            need_q2_ws = need_q2_ws || q2_recipe;
            need_iq1_ws = need_iq1_ws || iq1_all_recipe || iq1_w13_q2_w2_recipe;
            need_iq1_q2_ws = need_iq1_q2_ws || iq1_w13_q2_w2_recipe;
            need_iq1_gemm_ws = need_iq1_gemm_ws || iq1_all_recipe;
        }

        check_cuda(cudaMalloc(&d_embed_chunk_f16, chunk_dim * sizeof(uint16_t)), "alloc GGUF prefill embed chunk");
        check_cuda(cudaMalloc(&d_prefill_token_ids, static_cast<size_t>(prompt_tokens) * sizeof(int)), "alloc GGUF prefill token ids");
        check_cuda(cudaMalloc(&d_state_rows, static_cast<size_t>(prompt_tokens) * dim * sizeof(float)), "alloc GGUF prefill state rows");
        check_cuda(cudaMalloc(&d_attn_norm_rows, chunk_dim * sizeof(float)), "alloc GGUF prefill attn norm rows");
        check_cuda(cudaMalloc(&d_q_a_rows, chunk_q_a * sizeof(float)), "alloc GGUF prefill q_a rows");
        check_cuda(cudaMalloc(&d_q_norm_rows, chunk_q_a * sizeof(float)), "alloc GGUF prefill q_norm rows");
        check_cuda(cudaMalloc(&d_q_rows, chunk_q * sizeof(float)), "alloc GGUF prefill q rows");
        check_cuda(cudaMalloc(&d_kv_a_rows, chunk_kv * sizeof(float)), "alloc GGUF prefill kv_a rows");
        check_cuda(cudaMalloc(&d_kv_rows, chunk_kv * sizeof(float)), "alloc GGUF prefill kv rows");
        check_cuda(cudaMalloc(&d_attn_value_rows, chunk_q * sizeof(float)), "alloc GGUF prefill attn value rows");
        check_cuda(cudaMalloc(&d_attn_mid_rows, chunk_attn_mid * sizeof(float)), "alloc GGUF prefill attn mid rows");
        check_cuda(cudaMalloc(&d_attn_out_rows, chunk_dim * sizeof(float)), "alloc GGUF prefill attn out rows");
        check_cuda(cudaMalloc(&d_ffn_norm_rows, chunk_dim * sizeof(float)), "alloc GGUF prefill ffn norm rows");
        check_cuda(cudaMalloc(&d_shared_gate_rows, chunk_inter * sizeof(float)), "alloc GGUF prefill shared gate rows");
        check_cuda(cudaMalloc(&d_shared_up_rows, chunk_inter * sizeof(float)), "alloc GGUF prefill shared up rows");
        check_cuda(cudaMalloc(&d_shared_hidden_rows, chunk_inter * sizeof(float)), "alloc GGUF prefill shared hidden rows");
        check_cuda(cudaMalloc(&d_shared_out_rows, chunk_dim * sizeof(float)), "alloc GGUF prefill shared out rows");
        check_cuda(cudaMalloc(&d_moe_rows, chunk_dim * sizeof(float)), "alloc GGUF prefill moe rows");
        check_cuda(cudaMalloc(&d_prefill_route_indices, static_cast<size_t>(routes_cap) * sizeof(int64_t)), "alloc GGUF prefill route indices");
        check_cuda(cudaMalloc(&d_prefill_route_weights, static_cast<size_t>(routes_cap) * sizeof(float)), "alloc GGUF prefill route weights");
        check_cuda(cudaMalloc(&d_group_route_tokens, static_cast<size_t>(routes_cap) * sizeof(int64_t)), "alloc GGUF prefill grouped route tokens");
        check_cuda(cudaMalloc(&d_group_route_weights, static_cast<size_t>(routes_cap) * sizeof(float)), "alloc GGUF prefill grouped route weights");
        check_cuda(cudaMalloc(&d_seg_starts, static_cast<size_t>(experts_per_rank + 1) * sizeof(int32_t)), "alloc GGUF prefill seg starts");
        check_cuda(cudaMalloc(&d_counts, static_cast<size_t>(experts_per_rank) * sizeof(int32_t)), "alloc GGUF prefill route counts");
        check_cuda(cudaMalloc(&d_offsets, static_cast<size_t>(experts_per_rank) * sizeof(int32_t)), "alloc GGUF prefill route offsets");
        check_cuda(cudaMalloc(&d_total_routes, sizeof(int32_t)), "alloc GGUF prefill total routes");
        if (use_prefill_indexed_attn) {
            check_cuda(cudaMalloc(&d_prefill_attn_indices, static_cast<size_t>(prefill_attn_index_elems) * sizeof(int32_t)), "alloc GGUF prefill attention indices");
            if (!build_prefill_window_indices_cuda(d_prefill_attn_indices, prompt_tokens, prefill_attn_window, prefill_attn_topk))
                throw std::runtime_error("GGUF prefill attention window indices failed");
        } else if (tp_rank == 0 && env_int_or_default("DSV4_GGUF_VERBOSE", 0) != 0) {
            std::cout << "gguf_prefill_indexed_attn=0 topk=" << prefill_attn_topk
                      << " max_topk=" << prefill_attn_max_index_topk
                      << " index_elems=" << prefill_attn_index_elems
                      << " max_index_elems=" << prefill_attn_max_index_elems << "\n";
        }
        check_cuda(cudaMalloc(&d_final_norm_rows, static_cast<size_t>(dim) * sizeof(float)), "alloc GGUF prefill final norm");
        if (need_q2_ws) {
            q2_ws.routes_cap = routes_cap; q2_ws.dim = dim; q2_ws.inter_dim = moe_inter;
            check_cuda(cudaMalloc(&q2_ws.d_x_route, routes_dim * sizeof(float)), "alloc GGUF prefill q2 x route");
            check_cuda(cudaMalloc(&q2_ws.d_x_q, routes_dim), "alloc GGUF prefill q2 x q");
            check_cuda(cudaMalloc(&q2_ws.d_x_scale, static_cast<size_t>(routes_cap) * x_groups * sizeof(float)), "alloc GGUF prefill q2 x scale");
            check_cuda(cudaMalloc(&q2_ws.d_route_slots, static_cast<size_t>(routes_cap) * sizeof(int64_t)), "alloc GGUF prefill q2 route slots");
            check_cuda(cudaMalloc(&q2_ws.d_gate, routes_inter * sizeof(float)), "alloc GGUF prefill q2 gate");
            check_cuda(cudaMalloc(&q2_ws.d_up, routes_inter * sizeof(float)), "alloc GGUF prefill q2 up");
            check_cuda(cudaMalloc(&q2_ws.d_hidden_q, routes_inter), "alloc GGUF prefill q2 hidden q");
            check_cuda(cudaMalloc(&q2_ws.d_hidden_scale, static_cast<size_t>(routes_cap) * hidden_groups * sizeof(float)), "alloc GGUF prefill q2 hidden scale");
        }
        if (need_iq1_ws) {
            iq1_ws.routes_cap = routes_cap; iq1_ws.dim = dim; iq1_ws.inter_dim = moe_inter;
            check_cuda(cudaMalloc(&iq1_ws.d_hidden, routes_inter * sizeof(float)), "alloc GGUF prefill iq1 hidden");
            if (need_iq1_q2_ws || iq1_grouped_w2_q8_enabled) {
                check_cuda(cudaMalloc(&iq1_ws.d_hidden_q, routes_inter), "alloc GGUF prefill iq1 hidden q");
                check_cuda(cudaMalloc(&iq1_ws.d_hidden_scale, static_cast<size_t>(routes_cap) * hidden_groups * sizeof(float)), "alloc GGUF prefill iq1 hidden scale");
            }
            if (iq1_grouped_gemm_enabled && need_iq1_gemm_ws) {
                const int tile_cap = (routes_cap + 15) / 16 + experts_per_rank;
                iq1_ws.tile_cap = tile_cap;
                check_cuda(cudaMalloc(&iq1_ws.d_x_q, routes_dim), "alloc GGUF prefill iq1 x q");
                check_cuda(cudaMalloc(&iq1_ws.d_x_scale, static_cast<size_t>(routes_cap) * iq1_x_groups16 * sizeof(float)), "alloc GGUF prefill iq1 x scale");
                check_cuda(cudaMalloc(&iq1_ws.d_tile_experts, static_cast<size_t>(tile_cap) * sizeof(int32_t)), "alloc GGUF prefill iq1 tile experts");
                check_cuda(cudaMalloc(&iq1_ws.d_tile_rows, static_cast<size_t>(tile_cap) * sizeof(int32_t)), "alloc GGUF prefill iq1 tile rows");
            }
        }

        check_cuda(cudaMemcpy(d_prefill_token_ids, seed_tokens.data(), static_cast<size_t>(prompt_tokens) * sizeof(int), cudaMemcpyHostToDevice), "copy GGUF prefill token ids");
        const uint16_t* embed_f16 = reinterpret_cast<const uint16_t*>(embed.data);
        for (int cs = 0; cs < prompt_tokens; cs += chunk_tokens) {
            const int cn = std::min(chunk_tokens, prompt_tokens - cs);
            for (int t = 0; t < cn; ++t) {
                const int tok = seed_tokens[static_cast<size_t>(cs + t)];
                std::memcpy(h_embed_chunk.data() + static_cast<size_t>(t) * dim,
                            embed_f16 + static_cast<size_t>(tok) * dim,
                            static_cast<size_t>(dim) * sizeof(uint16_t));
            }
            check_cuda(cudaMemcpy(d_embed_chunk_f16, h_embed_chunk.data(), static_cast<size_t>(cn) * dim * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy GGUF prefill embed chunk");
            if (!f16_contiguous_rows_to_float_cuda(d_embed_chunk_f16, d_state_rows + static_cast<size_t>(cs) * dim, cn, dim))
                throw std::runtime_error("GGUF prefill embed rows failed");
        }

        std::vector<int32_t> h_counts(static_cast<size_t>(experts_per_rank));
        double prof_prefill_attn_ms = 0.0;
        double prof_prefill_gate_ms = 0.0;
        double prof_prefill_moe_ms = 0.0;
        double prof_prefill_shared_ms = 0.0;
        for (int L = 0; L < n_layers; ++L) {
            const bool is_hash = (L < n_hash);
            const bool q2_recipe = layer_w1_dtype[L] == DType::IQ2_XXS && layer_w3_dtype[L] == DType::IQ2_XXS && layer_w2_dtype[L] == DType::Q2_K;
            const bool iq1_all_recipe = layer_w1_dtype[L] == DType::IQ1_M && layer_w3_dtype[L] == DType::IQ1_M && layer_w2_dtype[L] == DType::IQ1_M;
            const bool iq1_w13_q2_w2_recipe = layer_w1_dtype[L] == DType::IQ1_M && layer_w3_dtype[L] == DType::IQ1_M && layer_w2_dtype[L] == DType::Q2_K;
            // For host-routed GGUF prefill, keep the staged local expert arena valid
            // across all chunks of this layer. Long prompts otherwise re-copy the same
            // local expert weights once per chunk; the layer loop overwrites the arena
            // before the next layer, so a per-layer bitmap is sufficient.
            std::vector<uint8_t> h_prefill_staged_local(static_cast<size_t>(experts_per_rank), 0);
            for (int cs = 0; cs < prompt_tokens; cs += chunk_tokens) {
                const int cn = std::min(chunk_tokens, prompt_tokens - cs);
                const int ce = cs + cn;
                const size_t cn_dim = static_cast<size_t>(cn) * dim;
                float* state_chunk = d_state_rows + static_cast<size_t>(cs) * dim;
                auto stage_t = sync_now();
                if (!rmsnorm_bf16_gamma_rows_cuda(state_chunk, d_attn_gamma[L], d_attn_norm_rows, cn, dim, 1e-6f)) throw std::runtime_error("GGUF prefill attn norm rows failed");
                if (!q8_0_matmul_rows_cuda(d_attn_norm_rows, d_wq_a[L], d_q_a_rows, cn, q_a_dim, dim)) throw std::runtime_error("GGUF prefill wq_a rows failed");
                if (!rmsnorm_bf16_gamma_rows_cuda(d_q_a_rows, d_q_gamma[L], d_q_norm_rows, cn, q_a_dim, 1e-6f)) throw std::runtime_error("GGUF prefill q norm rows failed");
                if (!q8_0_matmul_rows_cuda(d_q_norm_rows, d_wq_b[L], d_q_rows, cn, q_full, q_a_dim)) throw std::runtime_error("GGUF prefill wq_b rows failed");
                if (!head_rmsnorm_rope_rows_cuda(d_q_rows, cn, heads, head_dim, rope_dim, cs, ld.rope_theta, false, 1e-6f)) throw std::runtime_error("GGUF prefill q rope rows failed");
                if (!q8_0_matmul_rows_cuda(d_attn_norm_rows, d_wkv[L], d_kv_a_rows, cn, kv_dim, dim)) throw std::runtime_error("GGUF prefill wkv rows failed");
                if (!rmsnorm_bf16_gamma_rows_cuda(d_kv_a_rows, d_kv_gamma[L], d_kv_rows, cn, kv_dim, 1e-6f)) throw std::runtime_error("GGUF prefill kv norm rows failed");
                if (!head_rmsnorm_rope_rows_cuda(d_kv_rows, cn, 1, head_dim, rope_dim, cs, ld.rope_theta, false, 0.0f)) throw std::runtime_error("GGUF prefill kv rope rows failed");
                if (!copy_rows_to_kv_cache_cuda(d_kv_rows, d_kv_cache[L], cn, head_dim, layer_cache_capacity[L], cs)) throw std::runtime_error("GGUF prefill kv cache rows copy failed");
                if (use_prefill_indexed_attn) {
                    if (!prefill_sparse_attention_indexed_cuda(d_q_rows, d_kv_cache[L], d_attn_sink[L], d_prefill_attn_indices + static_cast<size_t>(cs) * prefill_attn_topk, d_attn_value_rows, cn, heads, ce, prefill_attn_topk, head_dim, 1.0f / std::sqrt(static_cast<float>(head_dim)))) throw std::runtime_error("GGUF prefill indexed attention rows failed");
                } else {
                    if (!prefill_causal_attention_chunk_cuda(d_q_rows, d_kv_cache[L], d_attn_sink[L], d_attn_value_rows, cn, heads, ce, head_dim, layer_cache_capacity[L], cs, 1.0f / std::sqrt(static_cast<float>(head_dim)))) throw std::runtime_error("GGUF prefill attention rows failed");
                }
                if (!head_rmsnorm_rope_rows_cuda(d_attn_value_rows, cn, heads, head_dim, rope_dim, cs, ld.rope_theta, true, 0.0f)) throw std::runtime_error("GGUF prefill inverse rope rows failed");
                const size_t wo_a_row_bytes = static_cast<size_t>((group_in_dim + 31) / 32) * 34;
                for (int g = 0; g < o_groups; ++g) {
                    const float* group_x = d_attn_value_rows + static_cast<size_t>(g) * group_in_dim;
                    const uint8_t* group_w = d_wo_a[L] + static_cast<size_t>(g) * o_lora_rank * wo_a_row_bytes;
                    float* group_y = d_attn_mid_rows + static_cast<size_t>(g) * o_lora_rank;
                    if (!q8_0_matmul_rows_strided_cuda(group_x, group_w, group_y, cn, o_lora_rank, group_in_dim, q_full, attn_mid)) throw std::runtime_error("GGUF prefill wo_a rows failed");
                }
                if (!q8_0_matmul_rows_cuda(d_attn_mid_rows, d_wo_b[L], d_attn_out_rows, cn, dim, attn_mid)) throw std::runtime_error("GGUF prefill wo_b rows failed");
                gguf_all_reduce_sum_fp32_inplace(d_attn_out_rows, static_cast<int>(cn_dim), reduce_ctx_ptr, "GGUF prefill attention all-reduce");
                if (!vector_accum_rows_cuda(d_attn_out_rows, state_chunk, cn, dim, 1.0f)) throw std::runtime_error("GGUF prefill attention residual failed");
                if (profile_gguf) prof_prefill_attn_ms += elapsed_ms(stage_t, sync_now());

                stage_t = sync_now();
                if (!rmsnorm_bf16_gamma_rows_cuda(state_chunk, d_ffn_gamma[L], d_ffn_norm_rows, cn, dim, 1e-6f)) throw std::runtime_error("GGUF prefill ffn norm rows failed");
                if (is_hash) {
                    if (d_tid2eid_table[L] == nullptr) throw std::runtime_error("GGUF chunked prefill requires resident hash table");
                    if (!gate_hash_bf16_rows_cuda(d_ffn_norm_rows, d_gate_w[L], d_tid2eid_table[L], d_prefill_token_ids + cs, d_prefill_route_indices, d_prefill_route_weights, cn, dim, topk, topk, ld.route_scale)) throw std::runtime_error("GGUF prefill hash gate rows failed");
                } else {
                    if (!gate_topk_bf16_rows_cuda(d_ffn_norm_rows, d_gate_w[L], d_gate_bias[L], d_prefill_route_indices, d_prefill_route_weights, cn, n_experts, dim, topk, ld.route_scale)) throw std::runtime_error("GGUF prefill topk gate rows failed");
                }
                if (profile_gguf) prof_prefill_gate_ms += elapsed_ms(stage_t, sync_now());

                if (!moe_group_routes_cuda(d_prefill_route_indices, d_prefill_route_weights, d_group_route_tokens, d_group_route_weights, d_seg_starts, d_counts, d_offsets, d_total_routes, cn, topk, expert_start, experts_per_rank)) throw std::runtime_error("GGUF prefill group routes failed");
                int32_t total_routes = 0;
                check_cuda(cudaMemcpy(&total_routes, d_total_routes, sizeof(int32_t), cudaMemcpyDeviceToHost), "copy GGUF prefill total routes");
                check_cuda(cudaMemcpy(h_counts.data(), d_counts, h_counts.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "copy GGUF prefill route counts");
                int max_count = 0;
                for (int32_t c : h_counts) max_count = std::max(max_count, static_cast<int>(c));
                if (iq1_grouped_gemm_enabled && iq1_all_recipe && iq1_ws.d_tile_experts != nullptr && iq1_ws.d_tile_rows != nullptr) {
                    std::vector<int32_t> h_iq1_tile_experts;
                    std::vector<int32_t> h_iq1_tile_rows;
                    h_iq1_tile_experts.reserve(static_cast<size_t>((total_routes + 15) / 16 + experts_per_rank));
                    h_iq1_tile_rows.reserve(h_iq1_tile_experts.capacity());
                    for (int local = 0; local < experts_per_rank; ++local) {
                        const int count = h_counts[static_cast<size_t>(local)];
                        for (int row = 0; row < count; row += 16) {
                            h_iq1_tile_experts.push_back(local);
                            h_iq1_tile_rows.push_back(row);
                        }
                    }
                    iq1_ws.tile_count = static_cast<int>(h_iq1_tile_experts.size());
                    if (iq1_ws.tile_count > iq1_ws.tile_cap) throw std::runtime_error("GGUF prefill IQ1 tile capacity exceeded");
                    if (iq1_ws.tile_count > 0) {
                        check_cuda(cudaMemcpy(iq1_ws.d_tile_experts, h_iq1_tile_experts.data(), h_iq1_tile_experts.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy GGUF prefill iq1 tile experts");
                        check_cuda(cudaMemcpy(iq1_ws.d_tile_rows, h_iq1_tile_rows.data(), h_iq1_tile_rows.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy GGUF prefill iq1 tile rows");
                    }
                } else if (need_iq1_ws) {
                    iq1_ws.tile_count = 0;
                }

                stage_t = sync_now();
                if (total_routes > 0 && max_count > 0) {
                    const uint8_t* routed_w1 = is_resident_routed_layer[L] ? d_resident_w1[L] : d_routed_w1;
                    const uint8_t* routed_w2 = is_resident_routed_layer[L] ? d_resident_w2[L] : d_routed_w2;
                    const uint8_t* routed_w3 = is_resident_routed_layer[L] ? d_resident_w3[L] : d_routed_w3;
                    int routed_n = is_resident_routed_layer[L] ? experts_per_rank : experts_per_rank;
                    if (!is_resident_routed_layer[L]) {
                        const size_t w1_layer_bytes = static_cast<size_t>(layer_w1_bytes[L]);
                        const size_t w2_layer_bytes = static_cast<size_t>(layer_w2_bytes[L]);
                        const size_t w3_layer_bytes = static_cast<size_t>(layer_w3_bytes[L]);
                        for (int local = 0; local < experts_per_rank; ++local) {
                            if (h_counts[static_cast<size_t>(local)] == 0 || h_prefill_staged_local[static_cast<size_t>(local)] != 0) continue;
                            const int eid = expert_start + local;
                            const std::string lp = "layers." + std::to_string(L);
                            auto wv1 = gws.get_expert(lp + ".ffn.experts.routed.w1", lp + ".ffn.experts.routed.w1", eid);
                            auto wv2 = gws.get_expert(lp + ".ffn.experts.routed.w2", lp + ".ffn.experts.routed.w2", eid);
                            auto wv3 = gws.get_expert(lp + ".ffn.experts.routed.w3", lp + ".ffn.experts.routed.w3", eid);
                            if (!wv1.found || !wv2.found || !wv3.found) throw std::runtime_error("GGUF prefill get_expert failed");
                            check_cuda(cudaMemcpy(d_routed_w1 + w1_layer_bytes * local, wv1.data, w1_layer_bytes, cudaMemcpyHostToDevice), "stage GGUF prefill routed w1");
                            check_cuda(cudaMemcpy(d_routed_w2 + w2_layer_bytes * local, wv2.data, w2_layer_bytes, cudaMemcpyHostToDevice), "stage GGUF prefill routed w2");
                            check_cuda(cudaMemcpy(d_routed_w3 + w3_layer_bytes * local, wv3.data, w3_layer_bytes, cudaMemcpyHostToDevice), "stage GGUF prefill routed w3");
                            h_prefill_staged_local[static_cast<size_t>(local)] = 1;
                        }
                    }
                    if (q2_recipe) {
                        if (!moe_prefill_q2_grouped_cuda_with_workspace(d_ffn_norm_rows, d_group_route_tokens, d_group_route_weights, d_seg_starts, routed_w1, routed_w2, routed_w3, d_moe_rows, cn, total_routes, routed_n, max_count, dim, moe_inter, ld.swiglu_limit, q2_ws)) throw std::runtime_error("GGUF prefill grouped Q2 MoE failed");
                    } else if (iq1_all_recipe) {
                        if (!moe_prefill_iq1_grouped_cuda_with_workspace(d_ffn_norm_rows, d_group_route_tokens, d_group_route_weights, d_seg_starts, routed_w1, routed_w2, routed_w3, d_moe_rows, cn, total_routes, routed_n, max_count, dim, moe_inter, ld.swiglu_limit, iq1_ws)) throw std::runtime_error("GGUF prefill grouped IQ1 MoE failed");
                    } else if (iq1_w13_q2_w2_recipe) {
                        if (!iq1_moe_grouped_w13_swiglu_cuda(d_ffn_norm_rows, d_group_route_tokens, d_group_route_weights, d_seg_starts, routed_w1, routed_w3, iq1_ws.d_hidden, total_routes, routed_n, max_count, dim, moe_inter, ld.swiglu_limit)) throw std::runtime_error("GGUF prefill IQ1 W13 grouped failed");
                        if (!q2_quantize_hidden_q8_1_cuda(iq1_ws.d_hidden, iq1_ws.d_hidden_q, iq1_ws.d_hidden_scale, total_routes, moe_inter)) throw std::runtime_error("GGUF prefill IQ1/Q2 hidden quant failed");
                        if (!q2_moe_grouped_w2_q2k_cuda(iq1_ws.d_hidden_q, iq1_ws.d_hidden_scale, d_group_route_tokens, d_seg_starts, routed_w2, d_moe_rows, cn, total_routes, routed_n, max_count, dim, moe_inter)) throw std::runtime_error("GGUF prefill IQ1/Q2 grouped W2 failed");
                    } else {
                        throw std::runtime_error("GGUF prefill unsupported routed dtype recipe");
                    }
                } else {
                    check_cuda(cudaMemset(d_moe_rows, 0, cn_dim * sizeof(float)), "zero GGUF prefill empty moe rows");
                }
                gguf_all_reduce_sum_fp32_inplace(d_moe_rows, static_cast<int>(cn_dim), reduce_ctx_ptr, "GGUF prefill MoE all-reduce");
                if (profile_gguf) prof_prefill_moe_ms += elapsed_ms(stage_t, sync_now());

                stage_t = sync_now();
                if (!q8_0_matmul_rows_cuda(d_ffn_norm_rows, d_shared_w1[L], d_shared_gate_rows, cn, moe_inter, dim)) throw std::runtime_error("GGUF prefill shared w1 rows failed");
                if (!q8_0_matmul_rows_cuda(d_ffn_norm_rows, d_shared_w3[L], d_shared_up_rows, cn, moe_inter, dim)) throw std::runtime_error("GGUF prefill shared w3 rows failed");
                if (!silu_mul_rows_cuda(d_shared_gate_rows, d_shared_up_rows, d_shared_hidden_rows, cn, moe_inter)) throw std::runtime_error("GGUF prefill shared silu rows failed");
                if (!q8_0_matmul_rows_cuda(d_shared_hidden_rows, d_shared_w2[L], d_shared_out_rows, cn, dim, moe_inter)) throw std::runtime_error("GGUF prefill shared w2 rows failed");
                if (!vector_accum_rows_cuda(d_shared_out_rows, d_moe_rows, cn, dim, 1.0f)) throw std::runtime_error("GGUF prefill shared accum failed");
                if (!vector_accum_rows_cuda(d_moe_rows, state_chunk, cn, dim, 1.0f)) throw std::runtime_error("GGUF prefill ffn residual failed");
                if (profile_gguf) prof_prefill_shared_ms += elapsed_ms(stage_t, sync_now());
            }
        }

        float* last_row = d_state_rows + static_cast<size_t>(prompt_tokens - 1) * dim;
        if (!rmsnorm_bf16_gamma_cuda(last_row, d_final_norm_gamma, d_final_norm_rows, dim, 1e-6f)) throw std::runtime_error("GGUF prefill final norm failed");
        if (!q8_0_matvec_cuda(d_final_norm_rows, d_head, d_logits, local_vocab, dim)) throw std::runtime_error("GGUF prefill head failed");
        int top_t = local_vocab_start;
        float top_l = -INFINITY;
        if (gguf_host_logits) {
            check_cuda(cudaDeviceSynchronize(), "sync GGUF prefill head");
            check_cuda(cudaMemcpy(h_logits.data(), d_logits, local_vocab * sizeof(float), cudaMemcpyDeviceToHost), "copy GGUF prefill logits");
            int local_t = 0;
            float local_l = -INFINITY;
            for (int i = 0; i < local_vocab; ++i) {
                if (h_logits[i] > local_l) { local_l = h_logits[i]; local_t = i; }
            }
            top_t = local_t + local_vocab_start;
            top_l = local_l;
        } else {
            if (!argmax_fp32_cuda(d_logits, d_argmax_token, d_argmax_logit, local_vocab, local_vocab_start)) throw std::runtime_error("GGUF prefill argmax failed");
            check_cuda(cudaMemcpy(&top_t, d_argmax_token, sizeof(int), cudaMemcpyDeviceToHost), "copy GGUF prefill argmax token");
            check_cuda(cudaMemcpy(&top_l, d_argmax_logit, sizeof(float), cudaMemcpyDeviceToHost), "copy GGUF prefill argmax logit");
        }
#ifdef DSV4_HAVE_NCCL
        if (tp_world > 1 && !options.nccl_id_path.empty()) {
            TpTopResult global = nccl_global_top1(tp_world, tp_rank, tp_device,
                                                   options.nccl_id_path.c_str(),
                                                   top_t, top_l);
            top_t = global.token;
            top_l = global.logit;
        }
#endif
        double prof_prefill_refine_ms = 0.0;
        int refine_last_tokens = env_int_or_default("DSV4_GGUF_PREFILL_REFINE_LAST_TOKENS", env_int_or_default("DSV4_GGUF_PREFILL_REFINE_LAST", 1) != 0 ? 1 : 0);
        if (refine_last_tokens < 0) refine_last_tokens = 0;
        if (refine_last_tokens > prompt_tokens) refine_last_tokens = prompt_tokens;
        if (refine_last_tokens > 0 && prompt_tokens > 0) {
            auto refine_t = sync_now();
            const int refine_start = prompt_tokens - refine_last_tokens;
            for (int pos = refine_start; pos < prompt_tokens; ++pos) {
                auto refined = run_step(seed_tokens[static_cast<size_t>(pos)], pos);
                top_t = refined.first;
                top_l = refined.second;
            }
            if (profile_gguf) prof_prefill_refine_ms = elapsed_ms(refine_t, sync_now());
        }
        check_cuda(cudaDeviceSynchronize(), "sync GGUF chunked prefill");
        if (profile_gguf && tp_rank == 0) {
            std::cout << "gguf_chunked_prefill_profile chunk_tokens=" << chunk_tokens
                      << " iq1_w2_q8=" << (iq1_grouped_w2_q8_enabled ? 1 : 0)
                      << " iq1_gemm=" << (iq1_grouped_gemm_enabled ? 1 : 0)
                      << " iq1_tile4=" << (iq1_grouped_route_tile4_enabled ? 1 : 0)
                      << " iq1_route_major=" << (iq1_grouped_route_major_enabled ? 1 : 0)
                      << " indexed_attn=" << (use_prefill_indexed_attn ? 1 : 0)
                      << " attn_topk=" << prefill_attn_topk
                      << " attn_ms=" << prof_prefill_attn_ms
                      << " gate_ms=" << prof_prefill_gate_ms
                      << " moe_ms=" << prof_prefill_moe_ms
                      << " shared_ms=" << prof_prefill_shared_ms
                      << " refine_last_tokens=" << refine_last_tokens
                      << " refine_last_ms=" << prof_prefill_refine_ms << "\n";
        }

        cudaFree(d_embed_chunk_f16); cudaFree(d_prefill_token_ids); cudaFree(d_state_rows);
        cudaFree(d_attn_norm_rows); cudaFree(d_q_a_rows); cudaFree(d_q_norm_rows); cudaFree(d_q_rows);
        cudaFree(d_kv_a_rows); cudaFree(d_kv_rows); cudaFree(d_attn_value_rows); cudaFree(d_attn_mid_rows); cudaFree(d_attn_out_rows);
        cudaFree(d_ffn_norm_rows); cudaFree(d_shared_gate_rows); cudaFree(d_shared_up_rows); cudaFree(d_shared_hidden_rows); cudaFree(d_shared_out_rows); cudaFree(d_moe_rows);
        cudaFree(d_prefill_route_indices); cudaFree(d_prefill_route_weights); cudaFree(d_group_route_tokens); cudaFree(d_group_route_weights);
        cudaFree(d_seg_starts); cudaFree(d_counts); cudaFree(d_offsets); cudaFree(d_total_routes); cudaFree(d_prefill_attn_indices); cudaFree(d_final_norm_rows);
        cudaFree(q2_ws.d_x_route); cudaFree(q2_ws.d_x_q); cudaFree(q2_ws.d_x_scale); cudaFree(q2_ws.d_route_slots); cudaFree(q2_ws.d_gate); cudaFree(q2_ws.d_up); cudaFree(q2_ws.d_hidden_q); cudaFree(q2_ws.d_hidden_scale);
        cudaFree(iq1_ws.d_hidden); cudaFree(iq1_ws.d_hidden_q); cudaFree(iq1_ws.d_hidden_scale); cudaFree(iq1_ws.d_x_q); cudaFree(iq1_ws.d_x_scale); cudaFree(iq1_ws.d_tile_experts); cudaFree(iq1_ws.d_tile_rows);
        return {top_t, top_l};
    };

    // ===== generate loop =====
    GgufDecodeResult r;
    r.n_layers = n_layers;
    r.dim = dim;
    r.vocab = vocab;
    r.prompt_tokens = gguf_load_only && decode_only_context > 0
        ? decode_only_context
        : static_cast<int>(seed_tokens.size());
    // Continuation decode passes after TTFT. The first generated token is the
    // argmax from the final prefill position; remaining generated tokens each
    // require one decode step.
    r.decode_tokens = std::max(0, max_new_tokens - 1);
    r.top_logits.reserve(total_positions);
    r.generated_tokens.reserve(max_new_tokens);

    std::chrono::steady_clock::time_point t_forward_start = std::chrono::steady_clock::now();
    auto t_prefill_start = t_forward_start;
    auto t_prefill_end = t_forward_start;
    auto t_decode_start = t_forward_start;
    auto t_decode_end = t_forward_start;
    int last_argmax = 0;
    if (!gguf_load_only && decode_only_context > 0) {
        if (max_new_tokens <= 0) throw std::runtime_error("DSV4_GGUF_DECODE_ONLY_CONTEXT requires max_new_tokens > 0");
        // Decode-only profiler: allocate caches as if context tokens already exist,
        // then run a single real forward step at position context-1. KV contents are
        // zero/dummy, so this is for timing/scaling only, not logits parity.
        //
        // The first step touches the cold GGUF mmap for non-resident Q2 expert
        // staging (random page faults across an ~86 GiB file). To measure steady
        // state, run a configurable number of warmup steps (excluded from the
        // profile counters and timing) before the measured step.
        const int warmup = env_int_or_default("DSV4_GGUF_DECODE_ONLY_WARMUP", all_resident_routed ? 0 : 1);
        const int feed_token = seed_tokens.empty() ? 1234 : seed_tokens[0];
        for (int w = 0; w < warmup; ++w) {
            (void)run_step(feed_token, decode_only_context - 1);
        }
        // Reset profile accumulators so only the measured step is reported.
        prof_embed_ms = prof_attn_ms = prof_stage_ms = prof_moe_ms = prof_head_ms = 0.0;
        prof_indexed_attn_steps = prof_indexed_attn_indices = 0;
        prof_sparse_attn_layer_steps = prof_sparse_attn_indices = 0;
        prof_steps = 0;
        r.prompt_tokens = decode_only_context;
        r.decode_tokens = 1;
        t_forward_start = std::chrono::steady_clock::now();
        t_prefill_start = t_forward_start;
        t_prefill_end = t_forward_start;
        t_decode_start = t_forward_start;
        auto [argmax, logit] = run_step(feed_token, decode_only_context - 1);
        r.top_logits.push_back(logit);
        r.generated_tokens.push_back(argmax);
        last_argmax = argmax;
    } else {
        const int forward_positions = (!gguf_load_only && max_new_tokens > 0) ? total_positions : 0;
        if (gguf_chunked_prefill_available && forward_positions > 0) {
            if (tp_rank == 0) {
                std::cout << "gguf_chunked_prefill=1 prompt_tokens=" << r.prompt_tokens
                          << " resident_layers=" << resident_routed_layers << "/" << n_layers << "\n";
            }
            auto [argmax, logit] = run_chunked_prefill();
            r.top_logits.push_back(logit);
            r.generated_tokens.push_back(argmax);
            last_argmax = argmax;
            t_prefill_end = std::chrono::steady_clock::now();
            t_decode_start = t_prefill_end;
            for (int pos = r.prompt_tokens; pos < forward_positions; ++pos) {
                auto [step_argmax, step_logit] = run_step(last_argmax, pos);
                r.top_logits.push_back(step_logit);
                if (static_cast<int>(r.generated_tokens.size()) < max_new_tokens) {
                    r.generated_tokens.push_back(step_argmax);
                }
                last_argmax = step_argmax;
            }
        } else {
            for (int pos = 0; pos < forward_positions; ++pos) {
                if (pos == r.prompt_tokens) t_decode_start = std::chrono::steady_clock::now();
                const int feed_token = (pos < r.prompt_tokens) ? seed_tokens[pos] : last_argmax;
                auto [argmax, logit] = run_step(feed_token, pos);
                r.top_logits.push_back(logit);
                // argmax produced at position p predicts the token for position p+1.
                // Record argmax once we reach the last seed (its prediction is the first
                // generated token) and every step thereafter.
                if (pos >= r.prompt_tokens - 1 &&
                    static_cast<int>(r.generated_tokens.size()) < max_new_tokens) {
                    r.generated_tokens.push_back(argmax);
                }
                last_argmax = argmax;
                if (pos == r.prompt_tokens - 1) {
                    t_prefill_end = std::chrono::steady_clock::now();
                    t_decode_start = t_prefill_end;
                }
            }
        }
    }
    auto t_forward_end = std::chrono::steady_clock::now();
    t_decode_end = t_forward_end;
    r.load_seconds = std::chrono::duration<double>(t_load_end - t_load_start).count();
    r.forward_seconds = std::chrono::duration<double>(t_forward_end - t_forward_start).count();
    r.prefill_seconds = std::chrono::duration<double>(t_prefill_end - t_prefill_start).count();
    r.decode_seconds = std::chrono::duration<double>(t_decode_end - t_decode_start).count();

    if (profile_gguf && prof_steps > 0 && tp_rank == 0) {
        const double inv = 1.0 / static_cast<double>(prof_steps);
        const double per_layer_attn = prof_attn_ms * inv / static_cast<double>(n_layers);
        const double per_layer_stage = prof_stage_ms * inv / static_cast<double>(n_layers);
        const double per_layer_moe = prof_moe_ms * inv / static_cast<double>(n_layers);
        const double per_step_total = (prof_embed_ms + prof_attn_ms + prof_stage_ms + prof_moe_ms + prof_head_ms) * inv;
        const double avg_index_count = prof_indexed_attn_steps > 0
            ? static_cast<double>(prof_indexed_attn_indices) / static_cast<double>(prof_indexed_attn_steps)
            : 0.0;
        const double avg_sparse_index_count = prof_sparse_attn_layer_steps > 0
            ? static_cast<double>(prof_sparse_attn_indices) / static_cast<double>(prof_sparse_attn_layer_steps)
            : 0.0;
        std::printf("[gguf_profile steps=%d n_layers=%d] per_step: embed=%.2f attn+gate=%.2f stage=%.2f moe=%.2f head=%.2f total=%.2f ms; per_layer: attn+gate=%.3f stage=%.3f moe=%.3f ms; indexed_attn_steps=%lld avg_index_count=%.1f sparse_attn_layer_steps=%lld avg_sparse_index_count=%.1f\n",
                    prof_steps, n_layers,
                    prof_embed_ms * inv, prof_attn_ms * inv, prof_stage_ms * inv, prof_moe_ms * inv, prof_head_ms * inv,
                    per_step_total, per_layer_attn, per_layer_stage, per_layer_moe,
                    prof_indexed_attn_steps, avg_index_count,
                    prof_sparse_attn_layer_steps, avg_sparse_index_count);
    }

    // ===== cleanup =====
    auto free_vec_u8 = [](std::vector<uint8_t*>& v) { for (auto* p : v) cudaFree(p); };
    auto free_vec_u16 = [](std::vector<uint16_t*>& v) { for (auto* p : v) cudaFree(p); };
    auto free_vec_f32 = [](std::vector<float*>& v) { for (auto* p : v) cudaFree(p); };
    free_vec_u16(d_attn_gamma); free_vec_u16(d_q_gamma); free_vec_u16(d_kv_gamma);
    free_vec_u16(d_ffn_gamma); free_vec_f32(d_attn_sink);
    free_vec_u8(d_wq_a); free_vec_u8(d_wq_b); free_vec_u8(d_wkv);
    free_vec_u8(d_wo_a); free_vec_u8(d_wo_b);
    free_vec_u8(d_shared_w1); free_vec_u8(d_shared_w2); free_vec_u8(d_shared_w3);
    free_vec_u16(d_gate_w); free_vec_f32(d_gate_bias);
    for (auto& c : d_compressor_w) {
        cudaFree(const_cast<uint16_t*>(c.wkv));
        cudaFree(const_cast<uint16_t*>(c.wgate));
        cudaFree(const_cast<float*>(c.ape));
        cudaFree(const_cast<uint16_t*>(c.norm));
    }
    for (auto& idx : d_indexer_w) {
        cudaFree(const_cast<uint16_t*>(idx.wq_b));
        cudaFree(const_cast<uint16_t*>(idx.weights_proj));
        cudaFree(const_cast<uint16_t*>(idx.comp.wkv));
        cudaFree(const_cast<uint16_t*>(idx.comp.wgate));
        cudaFree(const_cast<float*>(idx.comp.ape));
        cudaFree(const_cast<uint16_t*>(idx.comp.norm));
    }
    for (auto& st : d_sparse_state) {
        cudaFree(st.compressor_kv);
        cudaFree(st.compressor_score);
        cudaFree(st.indexer_kv_cache);
        cudaFree(st.indexer_comp_kv);
        cudaFree(st.indexer_comp_score);
    }
    for (auto* p : d_tid2eid_table) cudaFree(p);
    cudaFree(d_final_norm_gamma); cudaFree(d_head);
    cudaFree(d_routed_w1); cudaFree(d_routed_w2); cudaFree(d_routed_w3);
    cudaFree(d_attn_weight_scratch); cudaFree(d_kv_indices);
    free_vec_f32(d_kv_cache);
    cudaFree(d_x); cudaFree(d_x_pre_attn); cudaFree(d_x_normed);
    cudaFree(d_q_a); cudaFree(d_q_normed); cudaFree(d_q);
    cudaFree(d_kv_a); cudaFree(d_kv); cudaFree(d_attn_value);
    cudaFree(d_attn_mid); cudaFree(d_attn_out);
    cudaFree(d_x_pre_ffn); cudaFree(d_x_normed_ffn);
    cudaFree(d_shared_gate); cudaFree(d_shared_up); cudaFree(d_shared_hidden);
    cudaFree(d_shared_out); cudaFree(d_moe_out); cudaFree(d_ffn_combined);
    cudaFree(d_x_q); cudaFree(d_x_scale);
    cudaFree(d_route_slots); cudaFree(d_route_weights);
    cudaFree(d_route_gate); cudaFree(d_route_up);
    cudaFree(d_route_hidden_q); cudaFree(d_route_hidden_scale); cudaFree(d_route_hidden);
    cudaFree(d_gate_scores_scratch); cudaFree(d_gate_scored_scratch); cudaFree(d_gate_indices);
    cudaFree(d_tid2eid_i64); cudaFree(d_embed_row_f16);
    cudaFree(d_compressor_input_bf16); cudaFree(d_compressor_input_rounded);
    cudaFree(d_compressor_kv); cudaFree(d_compressor_score);
    cudaFree(d_indexer_comp_kv); cudaFree(d_indexer_comp_score);
    cudaFree(d_index_q); cudaFree(d_index_scores);
    cudaFree(d_logits);
    cudaFree(d_argmax_token); cudaFree(d_argmax_logit);
    free_vec_u8(d_resident_w1); free_vec_u8(d_resident_w2); free_vec_u8(d_resident_w3);
    if (moe_copy_stream_enabled) {
        cudaEventDestroy(gguf_moe_stage_event);
        cudaEventDestroy(gguf_moe_consume_event);
        cudaStreamDestroy(gguf_moe_copy_stream);
    }
    if (gguf_registered) cudaHostUnregister(gguf_base);
    return r;
}

GgufDecodeResult run_gguf_generate_smoke(const std::string& ckpt_path,
                                          const std::vector<int>& seed_tokens,
                                          int max_new_tokens) {
    return run_gguf_generate_smoke(ckpt_path, seed_tokens, max_new_tokens, ForwardSmokeOptions{});
}

struct PersistentEngine::State {
    std::unique_ptr<SafeForwardContext> ctx;
    Tokenizer tokenizer;
    ForwardSmokeOptions opts;
    int layer_count = 0;
    int max_context = 0;
    int eos_token_id = 1;
    std::mt19937 rng{0xDEEDBEEFu};
    uint64_t rng_seed = 0;
    std::unique_ptr<CmdChannel> cmd;

    State(const std::string& ckpt_dir, const ForwardSmokeOptions& options, int lc, int mc)
        : ctx(std::make_unique<SafeForwardContext>(ckpt_dir)),
          tokenizer(ckpt_dir),
          opts(options),
          layer_count(lc),
          max_context(mc) {
        if (const char* env = std::getenv("DSV4_CPP_EOS_TOKEN_ID")) {
            try { eos_token_id = std::stoi(env); } catch (...) {}
        }
    }
};

PersistentEngine::PersistentEngine(const std::string& ckpt_dir,
                                   const ForwardSmokeOptions& opts,
                                   int layer_count,
                                   int max_context)
    : state_(std::make_unique<State>(ckpt_dir, opts, layer_count, max_context)) {
    if (max_context <= 0) throw std::runtime_error("PersistentEngine: max_context must be > 0");
    if (layer_count <= 0) throw std::runtime_error("PersistentEngine: layer_count must be > 0");
    auto& ctx = *state_->ctx;
    ctx.options = opts;
    ctx.kv_cache_tokens = max_context;
    const int dim = static_cast<int>(ctx.embed->shape[1]);
    ctx.prepare_resident_device_caches(layer_count, opts.tp_world, opts.tp_rank, dim);
    const bool prepare_fp4_host = !opts.skip_fp4_host_prepare && env_int_or_default("DSV4_CPP_PREPARE_FP4_HOST", 1) != 0;
    if (prepare_fp4_host) ctx.prepare_fp4_host_weights(layer_count, opts.tp_world, opts.tp_rank);
}

PersistentEngine::~PersistentEngine() = default;

void PersistentEngine::reset_session() {
    // KV / indexer caches are pre-allocated to max_context capacity. Prefill
    // overwrites positions 0..N-1, and decode writes positions >= N, so cross
    // request contamination is impossible by construction. We still sync to
    // drain any in-flight stream work from the previous request.
    cudaDeviceSynchronize();
}

namespace {

// Unified token selection for PersistentEngine: every rank participates in the
// same NCCL pattern (allgather logits + broadcast 1 int32) regardless of
// greedy vs stochastic. Rank 0 picks argmax or samples on the full logits.
//
// Keeping the protocol uniform lets workers run a fixed sequence that doesn't
// depend on per-request SamplingParams flags they never see.
int select_token(SafeForwardContext& ctx, const ForwardSmokeOptions& opts,
                 const SamplingParams& sp, std::mt19937& rng) {
    const int local = static_cast<int>(ctx.last_local_logits.size());
    if (local <= 0) throw std::runtime_error("select_token: empty local logits");
    const bool greedy = sp.greedy || sp.temperature <= 1.0e-5f;

#ifdef DSV4_HAVE_NCCL
    if (opts.tp_world > 1 && !opts.nccl_id_path.empty()) {
        std::vector<float> all;
        if (opts.tp_rank == 0) all.resize(static_cast<size_t>(opts.tp_world) * static_cast<size_t>(local));
        nccl_gather_floats_to_root(opts.tp_world, opts.tp_rank, opts.device,
                                   opts.nccl_id_path.c_str(),
                                   ctx.last_local_logits.data(), local,
                                   opts.tp_rank == 0 ? all.data() : nullptr, 0);
        int32_t token_buf[1] = {0};
        if (opts.tp_rank == 0) {
            const int vocab = static_cast<int>(all.size());
            if (greedy) {
                int best = 0;
                float best_v = all[0];
                for (int i = 1; i < vocab; ++i) {
                    if (all[i] > best_v) { best_v = all[i]; best = i; }
                }
                token_buf[0] = best;
            } else {
                token_buf[0] = sample_token_top_p(all.data(), vocab, sp.temperature, sp.top_p, rng);
            }
        }
        nccl_broadcast_int32(opts.tp_world, opts.tp_rank, opts.device,
                             opts.nccl_id_path.c_str(), token_buf, 1, 0);
        return token_buf[0];
    }
#endif
    if (greedy) {
        int best = ctx.last_local_head_start;
        float best_v = -INFINITY;
        for (int i = 0; i < local; ++i) {
            if (ctx.last_local_logits[i] > best_v) {
                best_v = ctx.last_local_logits[i];
                best = ctx.last_local_head_start + i;
            }
        }
        return best;
    }
    return sample_token_top_p(ctx.last_local_logits.data(), local, sp.temperature, sp.top_p, rng);
}

void maybe_reseed(uint64_t seed, uint64_t& current_seed, std::mt19937& rng) {
    if (seed != 0 && seed != current_seed) {
        current_seed = seed;
        rng.seed(static_cast<std::mt19937::result_type>(seed));
    }
}

}  // namespace

int PersistentEngine::prefill(const std::vector<int>& token_ids, const SamplingParams& sp) {
    if (token_ids.empty()) throw std::runtime_error("PersistentEngine::prefill empty");
    auto& s = *state_;
    auto& ctx = *s.ctx;
    ctx.options = s.opts;
    maybe_reseed(sp.seed, s.rng_seed, s.rng);
    (void)run_safetensors_prompt_prefill_impl(ctx, token_ids, s.layer_count);
    const int token = select_token(ctx, s.opts, sp, s.rng);
    if (env_int_or_default("DSV4_CPP_DECODE_SPARSE_ARENA", 0) > 0) {
        ctx.release_active_arenas_with_suffix(":d");
    }
    return token;
}

int PersistentEngine::decode_step(int last_token, int position, const SamplingParams& sp) {
    auto& s = *state_;
    auto& ctx = *s.ctx;
    ctx.options = s.opts;
    maybe_reseed(sp.seed, s.rng_seed, s.rng);
    (void)run_safetensors_token_forward_impl(ctx, last_token, s.layer_count, position);
    return select_token(ctx, s.opts, sp, s.rng);
}

int PersistentEngine::eos_id() const { return state_->eos_token_id; }
int PersistentEngine::max_context() const { return state_->max_context; }
int PersistentEngine::layer_count() const { return state_->layer_count; }
const Tokenizer& PersistentEngine::tokenizer() const { return state_->tokenizer; }
const ForwardSmokeOptions& PersistentEngine::options() const { return state_->opts; }

// Worker loop / command channel: send 4 int32s [cmd, arg0, arg1, payload_count]
// followed by an optional int32 payload buffer of payload_count entries. Uses
// a CPU-only unix socket so workers truly sleep in read() while idle (0% GPU)
// rather than spinning inside an NCCL collective.
namespace {

constexpr int kCmdHeaderInts = 4;

}  // namespace

void PersistentEngine::worker_command_prefill(const std::vector<int>& token_ids) {
    auto& s = *state_;
    if (s.opts.tp_world <= 1 || !s.cmd) return;
    int32_t header[kCmdHeaderInts] = {
        static_cast<int32_t>(WorkerCommand::Prefill),
        0,
        0,
        static_cast<int32_t>(token_ids.size())
    };
    s.cmd->send_to_workers(header, kCmdHeaderInts);
    if (!token_ids.empty()) {
        std::vector<int32_t> payload(token_ids.begin(), token_ids.end());
        s.cmd->send_to_workers(payload.data(), payload.size());
    }
}

void PersistentEngine::worker_command_decode(int32_t last_token, int32_t position) {
    auto& s = *state_;
    if (s.opts.tp_world <= 1 || !s.cmd) return;
    int32_t header[kCmdHeaderInts] = {
        static_cast<int32_t>(WorkerCommand::DecodeStep),
        last_token,
        position,
        0
    };
    s.cmd->send_to_workers(header, kCmdHeaderInts);
}

void PersistentEngine::worker_command_reset() {
    auto& s = *state_;
    if (s.opts.tp_world <= 1 || !s.cmd) return;
    int32_t header[kCmdHeaderInts] = {
        static_cast<int32_t>(WorkerCommand::Reset), 0, 0, 0
    };
    s.cmd->send_to_workers(header, kCmdHeaderInts);
}

void PersistentEngine::worker_command_shutdown() {
    auto& s = *state_;
    if (s.opts.tp_world <= 1 || !s.cmd) return;
    int32_t header[kCmdHeaderInts] = {
        static_cast<int32_t>(WorkerCommand::Shutdown), 0, 0, 0
    };
    s.cmd->send_to_workers(header, kCmdHeaderInts);
}

void PersistentEngine::run_worker_loop() {
    auto& s = *state_;
    if (s.opts.tp_world <= 1) return;
    if (!s.cmd) throw std::runtime_error("run_worker_loop: command channel not initialized (call warmup_tp first)");
    SamplingParams sp;
    while (true) {
        int32_t header[kCmdHeaderInts] = {0, 0, 0, 0};
        s.cmd->recv_from_root(header, kCmdHeaderInts);
        const auto cmd = static_cast<WorkerCommand>(header[0]);
        if (cmd == WorkerCommand::Shutdown) return;

        // Drain any payload first (outside the try). Socket I/O failures are
        // unrecoverable — let them kill the worker.
        std::vector<int> prefill_tokens;
        if (cmd == WorkerCommand::Prefill && header[3] > 0) {
            std::vector<int32_t> payload(static_cast<size_t>(header[3]));
            s.cmd->recv_from_root(payload.data(), payload.size());
            prefill_tokens.assign(payload.begin(), payload.end());
        }

        // Catch per-command compute exceptions (OOM, CUDA errors) so a single
        // failed request doesn't kill the worker and deadlock the server.
        // Rank 0 catches its own exceptions in the HTTP handler and returns
        // 500; with workers staying alive, subsequent requests can succeed.
        try {
            if (cmd == WorkerCommand::Reset) {
                reset_session();
            } else if (cmd == WorkerCommand::Prefill) {
                (void)prefill(prefill_tokens, sp);
            } else if (cmd == WorkerCommand::DecodeStep) {
                (void)decode_step(header[1], header[2], sp);
            } else {
                throw std::runtime_error("PersistentEngine worker received unknown command");
            }
        } catch (const std::exception& ex) {
            std::cerr << "[worker rank " << s.opts.tp_rank << "] caught: " << ex.what() << "\n";
            cudaGetLastError();  // clear sticky CUDA error so next request starts clean
        }
    }
}

void PersistentEngine::warmup_tp() {
    auto& s = *state_;
    if (s.opts.tp_world <= 1) return;
    // Bring up the CPU command channel first so workers block on read() while
    // idle (0% GPU). Path is derived from nccl_id_path so both rendezvous use
    // the same launch-script-provided location.
    if (!s.cmd) {
        s.cmd = CmdChannel::create(s.opts.tp_world, s.opts.tp_rank, s.opts.nccl_id_path);
    }
    // Force NCCL communicator init by issuing one tiny broadcast so the id
    // file is written and the comm is ready before the first request. This
    // is the only NCCL bcast we keep — actual workload bcasts (allgather
    // logits, token bcast) happen inside prefill/decode.
#ifdef DSV4_HAVE_NCCL
    if (!s.opts.nccl_id_path.empty()) {
        int32_t warm[kCmdHeaderInts] = {0, 0, 0, 0};
        nccl_broadcast_int32(s.opts.tp_world, s.opts.tp_rank, s.opts.device,
                             s.opts.nccl_id_path.c_str(), warm, kCmdHeaderInts, 0);
    }
#endif
}

}  // namespace dsv4
