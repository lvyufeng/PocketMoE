#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline float fp4_code_to_float(uint8_t code) {
    static const float levels[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return levels[code & 0x0F];
}

static inline float silu_scalar(float x) {
    return x / (1.0f + std::exp(-x));
}


static inline int32_t dot_i8_avx2(const int8_t* a, const int8_t* b, long long n) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc32 = _mm256_setzero_si256();
    long long i = 0;
    for (; i + 31 < n; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m128i va_lo_128 = _mm256_castsi256_si128(va);
        __m128i va_hi_128 = _mm256_extracti128_si256(va, 1);
        __m128i vb_lo_128 = _mm256_castsi256_si128(vb);
        __m128i vb_hi_128 = _mm256_extracti128_si256(vb, 1);
        __m256i va_lo = _mm256_cvtepi8_epi16(va_lo_128);
        __m256i va_hi = _mm256_cvtepi8_epi16(va_hi_128);
        __m256i vb_lo = _mm256_cvtepi8_epi16(vb_lo_128);
        __m256i vb_hi = _mm256_cvtepi8_epi16(vb_hi_128);
        __m256i prod_lo = _mm256_mullo_epi16(va_lo, vb_lo);
        __m256i prod_hi = _mm256_mullo_epi16(va_hi, vb_hi);
        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(prod_lo, ones));
        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(prod_hi, ones));
    }
    alignas(32) int32_t tmp[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), acc32);
    int32_t acc = 0;
    for (int j = 0; j < 8; ++j) acc += tmp[j];
    for (; i < n; ++i) acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    return acc;
}

static void quantize_i8_row(const float* x, int8_t* xq, long long n, float* scale_out) {
    float max_abs = 0.0f;
    for (long long i = 0; i < n; ++i) max_abs = std::max(max_abs, std::fabs(x[i]));
    const float scale = std::max(max_abs, 1.0e-6f) / 127.0f;
    const float inv = 1.0f / scale;
    for (long long i = 0; i < n; ++i) {
        int q = static_cast<int>(std::nearbyint(x[i] * inv));
        q = std::max(-127, std::min(127, q));
        xq[i] = static_cast<int8_t>(q);
    }
    *scale_out = scale;
}

static void int8_expert_forward_impl(
    const float* x,
    const int8_t* w1,
    const float* s1,
    const int8_t* w2,
    const float* s2,
    const int8_t* w3,
    const float* s3,
    float* y,
    long long hidden_dim,
    long long inter_dim,
    float swiglu_limit,
    float route_w)
{
    std::vector<int8_t> xq(hidden_dim);
    float x_scale = 1.0f;
    quantize_i8_row(x, xq.data(), hidden_dim, &x_scale);
    std::vector<float> hidden(inter_dim);
    for (long long o = 0; o < inter_dim; ++o) {
        const int32_t acc1 = dot_i8_avx2(xq.data(), w1 + o * hidden_dim, hidden_dim);
        const int32_t acc3 = dot_i8_avx2(xq.data(), w3 + o * hidden_dim, hidden_dim);
        float gate = static_cast<float>(acc1) * x_scale * s1[o];
        float up = static_cast<float>(acc3) * x_scale * s3[o];
        if (swiglu_limit > 0.0f) {
            up = std::max(-swiglu_limit, std::min(swiglu_limit, up));
            gate = std::min(swiglu_limit, gate);
        }
        hidden[o] = route_w * silu_scalar(gate) * up;
    }
    std::fill(y, y + hidden_dim, 0.0f);
    std::vector<int8_t> hq(inter_dim);
    float h_scale = 1.0f;
    quantize_i8_row(hidden.data(), hq.data(), inter_dim, &h_scale);
    #pragma omp parallel for schedule(static)
    for (long long o = 0; o < hidden_dim; ++o) {
        const int32_t acc2 = dot_i8_avx2(hq.data(), w2 + o * inter_dim, inter_dim);
        y[o] = static_cast<float>(acc2) * h_scale * s2[o];
    }
}

static constexpr long long kFp4BlockSize = 32;
static constexpr long long kFp4BytesPerBlock = kFp4BlockSize / 2;
static constexpr long long kW2TileRows = 64;

alignas(16) static const int8_t kFp4LutI8x2[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12,
};

alignas(32) static const int32_t kEvenIdx8[8] = {0, 2, 4, 6, 8, 10, 12, 14};
alignas(32) static const int32_t kOddIdx8[8] = {1, 3, 5, 7, 9, 11, 13, 15};

static inline __m256 decode_fp4_nibbles8_to_ps(__m128i nibbles) {
    const __m128i lut = _mm_load_si128(reinterpret_cast<const __m128i*>(kFp4LutI8x2));
    const __m128i decoded_i8 = _mm_shuffle_epi8(lut, nibbles);
    const __m256i decoded_i32 = _mm256_cvtepi8_epi32(decoded_i8);
    const __m256 decoded = _mm256_cvtepi32_ps(decoded_i32);
    return _mm256_mul_ps(decoded, _mm256_set1_ps(0.5f));
}

static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static void fused_fp4_w13_hidden_range(
    const float* x_row,
    const uint8_t* w1,
    const uint8_t* w3,
    const float* s1,
    const float* s3,
    long long hidden_dim,
    long long in_blocks,
    float swiglu_limit,
    float route_w,
    float* hidden,
    long long o_start,
    long long o_end)
{
    const long long row_stride = hidden_dim / 2;
    const __m256i even_idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(kEvenIdx8));
    const __m256i odd_idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(kOddIdx8));
    const __m128i nibble_mask = _mm_set1_epi8(0x0F);

    for (long long o = o_start; o < o_end; ++o) {
        const uint8_t* w1_row = w1 + o * row_stride;
        const uint8_t* w3_row = w3 + o * row_stride;
        const float* s1_row = s1 + o * in_blocks;
        const float* s3_row = s3 + o * in_blocks;
        __m256 gate_acc = _mm256_setzero_ps();
        __m256 up_acc = _mm256_setzero_ps();

        for (long long b = 0; b < in_blocks; ++b) {
            const long long byte_base = b * kFp4BytesPerBlock;
            const long long feature_base = b * kFp4BlockSize;
            const __m256 scale1 = _mm256_set1_ps(s1_row[b]);
            const __m256 scale3 = _mm256_set1_ps(s3_row[b]);
            const uint8_t* w1_block = w1_row + byte_base;
            const uint8_t* w3_block = w3_row + byte_base;
            const float* x_block = x_row + feature_base;

            for (int chunk = 0; chunk < 2; ++chunk) {
                const __m128i raw1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w1_block + chunk * 8));
                const __m128i raw3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w3_block + chunk * 8));
                const __m128i lo1 = _mm_and_si128(raw1, nibble_mask);
                const __m128i hi1 = _mm_and_si128(_mm_srli_epi16(raw1, 4), nibble_mask);
                const __m128i lo3 = _mm_and_si128(raw3, nibble_mask);
                const __m128i hi3 = _mm_and_si128(_mm_srli_epi16(raw3, 4), nibble_mask);

                const __m256 w1_lo = _mm256_mul_ps(decode_fp4_nibbles8_to_ps(lo1), scale1);
                const __m256 w1_hi = _mm256_mul_ps(decode_fp4_nibbles8_to_ps(hi1), scale1);
                const __m256 w3_lo = _mm256_mul_ps(decode_fp4_nibbles8_to_ps(lo3), scale3);
                const __m256 w3_hi = _mm256_mul_ps(decode_fp4_nibbles8_to_ps(hi3), scale3);

                const float* x_chunk = x_block + chunk * 16;
                const __m256 x_even = _mm256_i32gather_ps(x_chunk, even_idx, 4);
                const __m256 x_odd = _mm256_i32gather_ps(x_chunk, odd_idx, 4);

                gate_acc = _mm256_fmadd_ps(w1_lo, x_even, gate_acc);
                gate_acc = _mm256_fmadd_ps(w1_hi, x_odd, gate_acc);
                up_acc = _mm256_fmadd_ps(w3_lo, x_even, up_acc);
                up_acc = _mm256_fmadd_ps(w3_hi, x_odd, up_acc);
            }
        }

        float gate = hsum256_ps(gate_acc);
        float up = hsum256_ps(up_acc);
        if (swiglu_limit > 0.0f) {
            up = std::max(-swiglu_limit, std::min(swiglu_limit, up));
            gate = std::min(swiglu_limit, gate);
        }
        hidden[o] = route_w * silu_scalar(gate) * up;
    }
}

static inline void fused_fp4_w13_hidden(
    const float* x_row,
    const uint8_t* w1,
    const uint8_t* w3,
    const float* s1,
    const float* s3,
    long long hidden_dim,
    long long inter_dim,
    long long in_blocks,
    float swiglu_limit,
    float route_w,
    float* hidden)
{
    fused_fp4_w13_hidden_range(
        x_row, w1, w3, s1, s3, hidden_dim, in_blocks, swiglu_limit, route_w, hidden, 0, inter_dim);
}

static void decode_fp4_w2_down_project_tile(
    const float* hidden,
    const uint8_t* w2_tiled,
    const float* s2_tiled,
    long long inter_dim,
    long long out_blocks,
    float* y_row,
    long long t_start,
    long long t_end)
{
    const long long tile_idx = t_start / kW2TileRows;
    const long long tile_row_start = tile_idx * kW2TileRows;
    const long long rows = t_end - t_start;
    float accs[kW2TileRows] = {0.0f};
    const uint8_t* tile_w = w2_tiled + tile_idx * out_blocks * kFp4BytesPerBlock * kW2TileRows;
    const float* tile_s = s2_tiled + tile_idx * out_blocks * kW2TileRows;
    for (long long b = 0; b < out_blocks; ++b) {
        const long long feature_base = b * kFp4BlockSize;
        const uint8_t* block_w = tile_w + b * kFp4BytesPerBlock * kW2TileRows;
        const float* block_s = tile_s + b * kW2TileRows;
        for (long long bb = 0; bb < kFp4BytesPerBlock; ++bb) {
            const float h0 = hidden[feature_base + bb * 2];
            const float h1 = hidden[feature_base + bb * 2 + 1];
            const uint8_t* packed_col = block_w + bb * kW2TileRows;
            for (long long r = 0; r < rows; ++r) {
                const long long tile_r = (t_start - tile_row_start) + r;
                const uint8_t packed = packed_col[tile_r];
                const float scale = block_s[tile_r];
                accs[r] += fp4_code_to_float(packed & 0x0F) * scale * h0;
                accs[r] += fp4_code_to_float((packed >> 4) & 0x0F) * scale * h1;
            }
        }
    }
    for (long long r = 0; r < rows; ++r) y_row[t_start + r] += accs[r];
}

static PyObject* set_omp_num_threads(PyObject*, PyObject* args) {
    int n;
    if (!PyArg_ParseTuple(args, "i", &n)) return nullptr;
#ifdef _OPENMP
    omp_set_num_threads(n);
#endif
    Py_RETURN_NONE;
}

static PyObject* routed_fp4_moe_forward(PyObject*, PyObject* args) {
    unsigned long long input_ptr_ull, expert_ids_ptr_ull, weights_ptr_ull, output_ptr_ull;
    unsigned long long w1_ptrs_ull, w2_ptrs_ull, w3_ptrs_ull, s1_ptrs_ull, s2_ptrs_ull, s3_ptrs_ull;
    long long tokens, hidden_dim, topk, inter_dim, num_experts, experts_start_idx, experts_end_idx;
    float swiglu_limit;
    if (!PyArg_ParseTuple(
            args,
            "KKKKLLLLLLLKKKKKKf",
            &input_ptr_ull,
            &expert_ids_ptr_ull,
            &weights_ptr_ull,
            &output_ptr_ull,
            &tokens,
            &hidden_dim,
            &topk,
            &inter_dim,
            &num_experts,
            &experts_start_idx,
            &experts_end_idx,
            &w1_ptrs_ull,
            &w2_ptrs_ull,
            &w3_ptrs_ull,
            &s1_ptrs_ull,
            &s2_ptrs_ull,
            &s3_ptrs_ull,
            &swiglu_limit)) {
        return nullptr;
    }

    auto* input = reinterpret_cast<float*>(input_ptr_ull);
    auto* expert_ids = reinterpret_cast<int64_t*>(expert_ids_ptr_ull);
    auto* weights = reinterpret_cast<float*>(weights_ptr_ull);
    auto* output = reinterpret_cast<float*>(output_ptr_ull);
    auto* w1_ptrs = reinterpret_cast<int64_t*>(w1_ptrs_ull);
    auto* w2_ptrs = reinterpret_cast<int64_t*>(w2_ptrs_ull);
    auto* w3_ptrs = reinterpret_cast<int64_t*>(w3_ptrs_ull);
    auto* s1_ptrs = reinterpret_cast<int64_t*>(s1_ptrs_ull);
    auto* s2_ptrs = reinterpret_cast<int64_t*>(s2_ptrs_ull);
    auto* s3_ptrs = reinterpret_cast<int64_t*>(s3_ptrs_ull);

    const long long in_blocks = hidden_dim / kFp4BlockSize;
    const long long out_blocks = inter_dim / kFp4BlockSize;

    Py_BEGIN_ALLOW_THREADS
    if (tokens > 1) {
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < tokens * hidden_dim; ++i) output[i] = 0.0f;

        #pragma omp parallel
        {
            std::vector<float> hidden(inter_dim);
            #pragma omp for schedule(static)
            for (long long t = 0; t < tokens; ++t) {
                const float* x_row = input + t * hidden_dim;
                float* y_row = output + t * hidden_dim;
                for (long long k = 0; k < topk; ++k) {
                    const long long expert_id = expert_ids[t * topk + k];
                    if (expert_id < experts_start_idx || expert_id >= experts_end_idx) continue;
                    auto* w1 = reinterpret_cast<uint8_t*>(w1_ptrs[expert_id]);
                    auto* w2 = reinterpret_cast<uint8_t*>(w2_ptrs[expert_id]);
                    auto* w3 = reinterpret_cast<uint8_t*>(w3_ptrs[expert_id]);
                    auto* s1 = reinterpret_cast<float*>(s1_ptrs[expert_id]);
                    auto* s2 = reinterpret_cast<float*>(s2_ptrs[expert_id]);
                    auto* s3 = reinterpret_cast<float*>(s3_ptrs[expert_id]);
                    if (!w1 || !w2 || !w3 || !s1 || !s2 || !s3) continue;

                    fused_fp4_w13_hidden(
                        x_row, w1, w3, s1, s3, hidden_dim, inter_dim, in_blocks,
                        swiglu_limit, weights[t * topk + k], hidden.data());
                    for (long long t_start = 0; t_start < hidden_dim; t_start += kW2TileRows) {
                        long long t_end = t_start + kW2TileRows;
                        if (t_end > hidden_dim) t_end = hidden_dim;
                        decode_fp4_w2_down_project_tile(hidden.data(), w2, s2, inter_dim, out_blocks, y_row, t_start, t_end);
                    }
                }
            }
        }
    } else {
        for (long long i = 0; i < hidden_dim; ++i) output[i] = 0.0f;

        const float* x_row = input;
        float* y_row = output;
        std::vector<float> hidden(inter_dim);
        for (long long k = 0; k < topk; ++k) {
            const long long expert_id = expert_ids[k];
            if (expert_id < experts_start_idx || expert_id >= experts_end_idx) continue;
            auto* w1 = reinterpret_cast<uint8_t*>(w1_ptrs[expert_id]);
            auto* w2 = reinterpret_cast<uint8_t*>(w2_ptrs[expert_id]);
            auto* w3 = reinterpret_cast<uint8_t*>(w3_ptrs[expert_id]);
            auto* s1 = reinterpret_cast<float*>(s1_ptrs[expert_id]);
            auto* s2 = reinterpret_cast<float*>(s2_ptrs[expert_id]);
            auto* s3 = reinterpret_cast<float*>(s3_ptrs[expert_id]);
            if (!w1 || !w2 || !w3 || !s1 || !s2 || !s3) continue;

            fused_fp4_w13_hidden(
                x_row, w1, w3, s1, s3, hidden_dim, inter_dim, in_blocks,
                swiglu_limit, weights[k], hidden.data());
            #pragma omp parallel for schedule(static)
            for (long long t_start = 0; t_start < hidden_dim; t_start += kW2TileRows) {
                long long t_end = t_start + kW2TileRows;
                if (t_end > hidden_dim) t_end = hidden_dim;
                decode_fp4_w2_down_project_tile(hidden.data(), w2, s2, inter_dim, out_blocks, y_row, t_start, t_end);
            }
        }
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}


static PyObject* routed_int8_moe_forward(PyObject*, PyObject* args) {
    unsigned long long input_ptr_ull, expert_ids_ptr_ull, weights_ptr_ull, output_ptr_ull;
    unsigned long long w1_ptrs_ull, w2_ptrs_ull, w3_ptrs_ull, s1_ptrs_ull, s2_ptrs_ull, s3_ptrs_ull;
    long long tokens, hidden_dim, topk, inter_dim, num_experts, experts_start_idx, experts_end_idx;
    float swiglu_limit;
    if (!PyArg_ParseTuple(
            args,
            "KKKKLLLLLLLKKKKKKf",
            &input_ptr_ull,
            &expert_ids_ptr_ull,
            &weights_ptr_ull,
            &output_ptr_ull,
            &tokens,
            &hidden_dim,
            &topk,
            &inter_dim,
            &num_experts,
            &experts_start_idx,
            &experts_end_idx,
            &w1_ptrs_ull,
            &w2_ptrs_ull,
            &w3_ptrs_ull,
            &s1_ptrs_ull,
            &s2_ptrs_ull,
            &s3_ptrs_ull,
            &swiglu_limit)) {
        return nullptr;
    }
    auto* input = reinterpret_cast<float*>(input_ptr_ull);
    auto* expert_ids = reinterpret_cast<int64_t*>(expert_ids_ptr_ull);
    auto* weights = reinterpret_cast<float*>(weights_ptr_ull);
    auto* output = reinterpret_cast<float*>(output_ptr_ull);
    auto* w1_ptrs = reinterpret_cast<int64_t*>(w1_ptrs_ull);
    auto* w2_ptrs = reinterpret_cast<int64_t*>(w2_ptrs_ull);
    auto* w3_ptrs = reinterpret_cast<int64_t*>(w3_ptrs_ull);
    auto* s1_ptrs = reinterpret_cast<int64_t*>(s1_ptrs_ull);
    auto* s2_ptrs = reinterpret_cast<int64_t*>(s2_ptrs_ull);
    auto* s3_ptrs = reinterpret_cast<int64_t*>(s3_ptrs_ull);

    Py_BEGIN_ALLOW_THREADS
    std::fill(output, output + tokens * hidden_dim, 0.0f);
    std::vector<float> scratch(hidden_dim);
    for (long long t = 0; t < tokens; ++t) {
        const float* x_row = input + t * hidden_dim;
        float* y_row = output + t * hidden_dim;
        for (long long k = 0; k < topk; ++k) {
            const long long expert_id = expert_ids[t * topk + k];
            if (expert_id < experts_start_idx || expert_id >= experts_end_idx) continue;
            auto* w1 = reinterpret_cast<int8_t*>(w1_ptrs[expert_id]);
            auto* w2 = reinterpret_cast<int8_t*>(w2_ptrs[expert_id]);
            auto* w3 = reinterpret_cast<int8_t*>(w3_ptrs[expert_id]);
            auto* s1 = reinterpret_cast<float*>(s1_ptrs[expert_id]);
            auto* s2 = reinterpret_cast<float*>(s2_ptrs[expert_id]);
            auto* s3 = reinterpret_cast<float*>(s3_ptrs[expert_id]);
            if (!w1 || !w2 || !w3 || !s1 || !s2 || !s3) continue;
            int8_expert_forward_impl(
                x_row, w1, s1, w2, s2, w3, s3, scratch.data(),
                hidden_dim, inter_dim, swiglu_limit, weights[t * topk + k]);
            for (long long o = 0; o < hidden_dim; ++o) y_row[o] += scratch[o];
        }
    }
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}


static PyObject* int8_expert_forward(PyObject*, PyObject* args) {
    unsigned long long x_ptr_ull, w1_ptr_ull, s1_ptr_ull, w2_ptr_ull, s2_ptr_ull, w3_ptr_ull, s3_ptr_ull, y_ptr_ull;
    long long hidden_dim, inter_dim;
    float swiglu_limit, route_w;
    if (!PyArg_ParseTuple(
            args,
            "KKKKKKKKLLff",
            &x_ptr_ull,
            &w1_ptr_ull,
            &s1_ptr_ull,
            &w2_ptr_ull,
            &s2_ptr_ull,
            &w3_ptr_ull,
            &s3_ptr_ull,
            &y_ptr_ull,
            &hidden_dim,
            &inter_dim,
            &swiglu_limit,
            &route_w)) {
        return nullptr;
    }
    auto* x = reinterpret_cast<float*>(x_ptr_ull);
    auto* w1 = reinterpret_cast<int8_t*>(w1_ptr_ull);
    auto* s1 = reinterpret_cast<float*>(s1_ptr_ull);
    auto* w2 = reinterpret_cast<int8_t*>(w2_ptr_ull);
    auto* s2 = reinterpret_cast<float*>(s2_ptr_ull);
    auto* w3 = reinterpret_cast<int8_t*>(w3_ptr_ull);
    auto* s3 = reinterpret_cast<float*>(s3_ptr_ull);
    auto* y = reinterpret_cast<float*>(y_ptr_ull);
    Py_BEGIN_ALLOW_THREADS
    int8_expert_forward_impl(x, w1, s1, w2, s2, w3, s3, y, hidden_dim, inter_dim, swiglu_limit, route_w);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyMethodDef Methods[] = {
    {"routed_fp4_moe_forward", routed_fp4_moe_forward, METH_VARARGS, nullptr},
    {"routed_int8_moe_forward", routed_int8_moe_forward, METH_VARARGS, nullptr},
    {"int8_expert_forward", int8_expert_forward, METH_VARARGS, nullptr},
    {"set_omp_num_threads", set_omp_num_threads, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "deepseek_cpu_moe_ext",
    nullptr,
    -1,
    Methods,
};

PyMODINIT_FUNC PyInit_deepseek_cpu_moe_ext(void) {
    return PyModule_Create(&ModuleDef);
}
