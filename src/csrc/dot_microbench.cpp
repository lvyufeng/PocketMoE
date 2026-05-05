// dot_microbench.cpp — Microbench for INT8 dot kernels used in CPU MoE.
//
// Build:
//   g++ -O3 -mavx2 -mfma -std=c++17 dot_microbench.cpp -o /tmp/dot_microbench
// Run:
//   /tmp/dot_microbench
//
// Measures wall time per dot product across (a) baseline cvt+mullo+madd,
// (b) candidate pmaddubsw+pmaddwd with sign trick, for both single and pair
// variants and at the two relevant inner dims (4096 for compute_hidden, 2048
// for compute_output).

#include <immintrin.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ============================================================================
// BASELINE: cvtepi8_epi16 -> mullo -> madd path (matches deepseek_cpu_moe_ext.cpp).
// ============================================================================
static inline int32_t dot_baseline(const int8_t* a, const int8_t* b, long long n) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    long long i = 0;
    for (; i + 31 < n; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
        __m256i p_lo = _mm256_mullo_epi16(va_lo, vb_lo);
        __m256i p_hi = _mm256_mullo_epi16(va_hi, vb_hi);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p_lo, ones));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p_hi, ones));
    }
    alignas(32) int32_t tmp[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), acc);
    int32_t s = 0;
    for (int j = 0; j < 8; ++j) s += tmp[j];
    for (; i < n; ++i) s += int32_t(a[i]) * int32_t(b[i]);
    return s;
}

static inline void dot_baseline_pair(const int8_t* a, const int8_t* b0, const int8_t* b1,
                                     long long n, int32_t* out) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    long long i = 0;
    for (; i + 31 < n; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b0 + i));
        __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b1 + i));
        __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        __m256i vb0_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb0));
        __m256i vb0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb0, 1));
        __m256i vb1_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb1));
        __m256i vb1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb1, 1));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_mullo_epi16(va_lo, vb0_lo), ones));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_mullo_epi16(va_hi, vb0_hi), ones));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_mullo_epi16(va_lo, vb1_lo), ones));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_mullo_epi16(va_hi, vb1_hi), ones));
    }
    alignas(32) int32_t t0[8], t1[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(t0), acc0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(t1), acc1);
    int32_t s0 = 0, s1 = 0;
    for (int j = 0; j < 8; ++j) { s0 += t0[j]; s1 += t1[j]; }
    for (; i < n; ++i) {
        int32_t av = int32_t(a[i]);
        s0 += av * int32_t(b0[i]);
        s1 += av * int32_t(b1[i]);
    }
    out[0] = s0; out[1] = s1;
}

// ============================================================================
// CANDIDATE: pabsb(a) + psignb(b, a) + pmaddubsw -> madd_epi16 -> add_epi32.
// Uses the sign trick: signed(a) * signed(b) = unsigned(|a|) * signed(b * sign(a)).
// Each 32-byte stride emits 5 vec ops (vs 10 in baseline).
// pmaddubsw produces s16 (with saturation guard: |a_abs| <= 127, |b_signed| <= 127,
// so each pair-product is in [-127*127, 127*127] = [-16129, 16129] -> safe within s16).
// ============================================================================
static inline int32_t dot_pmaddubsw(const int8_t* a, const int8_t* b, long long n) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    long long i = 0;
    for (; i + 31 < n; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256i va_abs = _mm256_abs_epi8(va);             // unsigned in [0, 127]
        __m256i vb_signed = _mm256_sign_epi8(vb, va);     // b * sign(a)
        __m256i prod_s16 = _mm256_maddubs_epi16(va_abs, vb_signed); // pair sum, s16
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod_s16, ones));
    }
    alignas(32) int32_t tmp[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), acc);
    int32_t s = 0;
    for (int j = 0; j < 8; ++j) s += tmp[j];
    for (; i < n; ++i) s += int32_t(a[i]) * int32_t(b[i]);
    return s;
}

static inline void dot_pmaddubsw_pair(const int8_t* a, const int8_t* b0, const int8_t* b1,
                                      long long n, int32_t* out) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    long long i = 0;
    for (; i + 31 < n; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b0 + i));
        __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b1 + i));
        __m256i va_abs = _mm256_abs_epi8(va);
        __m256i vb0_s = _mm256_sign_epi8(vb0, va);
        __m256i vb1_s = _mm256_sign_epi8(vb1, va);
        __m256i p0 = _mm256_maddubs_epi16(va_abs, vb0_s);
        __m256i p1 = _mm256_maddubs_epi16(va_abs, vb1_s);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p0, ones));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(p1, ones));
    }
    alignas(32) int32_t t0[8], t1[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(t0), acc0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(t1), acc1);
    int32_t s0 = 0, s1 = 0;
    for (int j = 0; j < 8; ++j) { s0 += t0[j]; s1 += t1[j]; }
    for (; i < n; ++i) {
        int32_t av = int32_t(a[i]);
        s0 += av * int32_t(b0[i]);
        s1 += av * int32_t(b1[i]);
    }
    out[0] = s0; out[1] = s1;
}

// ============================================================================
// Bench harness.
// ============================================================================

static double bench_single(int (*fn)(const int8_t*, const int8_t*, long long),
                           const int8_t* a, const int8_t* b, long long n,
                           long long iters, int32_t* sink) {
    auto t0 = std::chrono::steady_clock::now();
    int32_t acc = 0;
    for (long long it = 0; it < iters; ++it) {
        acc += static_cast<int32_t>(fn(a, b, n));
    }
    auto t1 = std::chrono::steady_clock::now();
    *sink ^= acc;
    return std::chrono::duration<double>(t1 - t0).count();
}

static double bench_pair(void (*fn)(const int8_t*, const int8_t*, const int8_t*, long long, int32_t*),
                         const int8_t* a, const int8_t* b0, const int8_t* b1, long long n,
                         long long iters, int32_t* sink) {
    auto t0 = std::chrono::steady_clock::now();
    int32_t acc = 0;
    int32_t out[2];
    for (long long it = 0; it < iters; ++it) {
        fn(a, b0, b1, n, out);
        acc += out[0] ^ out[1];
    }
    auto t1 = std::chrono::steady_clock::now();
    *sink ^= acc;
    return std::chrono::duration<double>(t1 - t0).count();
}

int main() {
    const long long dims[] = {2048, 4096};
    const long long iters_per_dim = 200000;
    int32_t sink = 0;

    // Use a single large pool; align to 64B to mimic real workload.
    const long long max_dim = 4096;
    std::vector<int8_t> a(max_dim), b(max_dim), b0(max_dim), b1(max_dim);
    std::srand(42);
    // Production data is always in [-127, 127] (quantize_i8_row clips, checkpoint weights too).
    // Generate test data in the same range so the sign-trick path's domain assumption holds.
    auto rand_i8 = []() {
        int v = (std::rand() % 255) - 127;  // [-127, 127]
        return static_cast<int8_t>(v);
    };
    for (long long i = 0; i < max_dim; ++i) {
        a[i] = rand_i8();
        b[i] = rand_i8();
        b0[i] = rand_i8();
        b1[i] = rand_i8();
    }

    // Sanity: check that baseline and candidate produce the same answer.
    for (long long n : dims) {
        int32_t s_base = dot_baseline(a.data(), b.data(), n);
        int32_t s_cand = dot_pmaddubsw(a.data(), b.data(), n);
        if (s_base != s_cand) {
            std::fprintf(stderr, "MISMATCH single dim=%lld baseline=%d candidate=%d\n",
                         n, s_base, s_cand);
            return 1;
        }
        int32_t out_base[2], out_cand[2];
        dot_baseline_pair(a.data(), b0.data(), b1.data(), n, out_base);
        dot_pmaddubsw_pair(a.data(), b0.data(), b1.data(), n, out_cand);
        if (out_base[0] != out_cand[0] || out_base[1] != out_cand[1]) {
            std::fprintf(stderr, "MISMATCH pair dim=%lld baseline=(%d,%d) candidate=(%d,%d)\n",
                         n, out_base[0], out_base[1], out_cand[0], out_cand[1]);
            return 1;
        }
    }
    std::fprintf(stderr, "correctness OK; both dims, single + pair\n\n");

    std::printf("%-15s %-8s %15s %15s %15s\n", "kernel", "dim", "iters", "ns/dot", "GiB/s");
    std::printf("%-15s %-8s %15s %15s %15s\n", "------", "---", "-----", "------", "-----");
    for (long long n : dims) {
        // Warmup.
        bench_single(dot_baseline, a.data(), b.data(), n, 1000, &sink);
        bench_single(dot_pmaddubsw, a.data(), b.data(), n, 1000, &sink);
        bench_pair(dot_baseline_pair, a.data(), b0.data(), b1.data(), n, 1000, &sink);
        bench_pair(dot_pmaddubsw_pair, a.data(), b0.data(), b1.data(), n, 1000, &sink);

        double t_base_single = bench_single(dot_baseline, a.data(), b.data(), n, iters_per_dim, &sink);
        double t_cand_single = bench_single(dot_pmaddubsw, a.data(), b.data(), n, iters_per_dim, &sink);
        double t_base_pair = bench_pair(dot_baseline_pair, a.data(), b0.data(), b1.data(), n, iters_per_dim, &sink);
        double t_cand_pair = bench_pair(dot_pmaddubsw_pair, a.data(), b0.data(), b1.data(), n, iters_per_dim, &sink);

        const double ns = 1e9;
        const double base_single_ns = (t_base_single / iters_per_dim) * ns;
        const double cand_single_ns = (t_cand_single / iters_per_dim) * ns;
        const double base_pair_ns = (t_base_pair / iters_per_dim) * ns;
        const double cand_pair_ns = (t_cand_pair / iters_per_dim) * ns;

        // GiB/s for single = 2 * n bytes per iter / time per iter.
        auto gibs = [](double per_iter_s, long long bytes_per_iter) {
            return double(bytes_per_iter) / per_iter_s / (1024.0 * 1024.0 * 1024.0);
        };
        std::printf("%-15s %-8lld %15lld %15.2f %15.2f\n", "baseline_single", n, iters_per_dim,
                    base_single_ns, gibs(t_base_single / iters_per_dim, 2 * n));
        std::printf("%-15s %-8lld %15lld %15.2f %15.2f\n", "pmaddubsw_single", n, iters_per_dim,
                    cand_single_ns, gibs(t_cand_single / iters_per_dim, 2 * n));
        std::printf("%-15s %-8lld %15lld %15.2f %15.2f\n", "baseline_pair", n, iters_per_dim,
                    base_pair_ns, gibs(t_base_pair / iters_per_dim, 3 * n));
        std::printf("%-15s %-8lld %15lld %15.2f %15.2f\n", "pmaddubsw_pair", n, iters_per_dim,
                    cand_pair_ns, gibs(t_cand_pair / iters_per_dim, 3 * n));
        std::printf("  speedup single: %.2fx   pair: %.2fx\n",
                    base_single_ns / cand_single_ns, base_pair_ns / cand_pair_ns);
    }

    std::fprintf(stderr, "sink=%d\n", sink);
    return 0;
}
