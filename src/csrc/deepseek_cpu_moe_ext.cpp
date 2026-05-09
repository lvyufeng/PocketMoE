#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <condition_variable>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
static int g_omp_num_threads = 0;
static inline void apply_omp_runtime_config() {
    if (g_omp_num_threads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(g_omp_num_threads);
    }
}
#else
static inline void apply_omp_runtime_config() {}
#endif

static class PersistentParallelTeam {
public:
    PersistentParallelTeam() = default;
    ~PersistentParallelTeam() { stop(); }

    void ensure(int n_threads) {
        if (n_threads <= 0) n_threads = 1;
        std::unique_lock<std::mutex> lock(mu_);
        if (started_ && n_threads_ == n_threads) return;
        lock.unlock();
        stop();
        lock.lock();
        n_threads_ = n_threads;
        stop_ = false;
        generation_ = 0;
        finished_ = 0;
        has_work_ = false;
        workers_.reserve(n_threads_);
        for (int i = 0; i < n_threads_; ++i) {
            workers_.emplace_back([this, i]() { worker_loop(i); });
        }
        started_ = true;
    }

    void parallel_for(long long begin, long long end, const std::function<void(long long, long long, int)>& fn) {
        if (end <= begin) return;
        ensure(g_omp_num_threads > 0 ? g_omp_num_threads : 1);
        {
            std::unique_lock<std::mutex> lock(mu_);
            begin_ = begin;
            end_ = end;
            fn_ = &fn;
            finished_ = 0;
            has_work_ = true;
            ++generation_;
            cv_.notify_all();
        }
        std::unique_lock<std::mutex> lock(mu_);
        done_cv_.wait(lock, [this]() { return !has_work_; });
    }

private:
    void stop() {
        std::unique_lock<std::mutex> lock(mu_);
        if (!started_) return;
        stop_ = true;
        ++generation_;
        cv_.notify_all();
        lock.unlock();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
        workers_.clear();
        lock.lock();
        started_ = false;
        stop_ = false;
    }

    void worker_loop(int tid) {
        long long seen = 0;
        while (true) {
            const std::function<void(long long, long long, int)>* fn = nullptr;
            long long begin = 0;
            long long end = 0;
            int n_threads = 1;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this, &seen]() { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
                fn = fn_;
                begin = begin_;
                end = end_;
                n_threads = n_threads_;
            }
            const long long total = end - begin;
            const long long chunk = (total + n_threads - 1) / n_threads;
            const long long s = begin + tid * chunk;
            const long long e = std::min(end, s + chunk);
            if (s < e && fn) (*fn)(s, e, tid);
            {
                std::unique_lock<std::mutex> lock(mu_);
                ++finished_;
                if (finished_ == n_threads_) {
                    has_work_ = false;
                    done_cv_.notify_one();
                }
            }
        }
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::vector<std::thread> workers_;
    const std::function<void(long long, long long, int)>* fn_ = nullptr;
    long long begin_ = 0;
    long long end_ = 0;
    long long generation_ = 0;
    int n_threads_ = 1;
    int finished_ = 0;
    bool started_ = false;
    bool stop_ = false;
    bool has_work_ = false;
} g_persistent_team;


static inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#else
    cpu_relax();
#endif
}

static inline void idle_wait_backoff(int spin_count) {
    if (spin_count < 1024) {
        cpu_relax();
        return;
    }
    if (spin_count < 8192) {
        sched_yield();
        return;
    }
    const struct timespec ts{0, 50000};
    nanosleep(&ts, nullptr);
}

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


static inline void dot_i8_avx2_pair(const int8_t* a, const int8_t* b0, const int8_t* b1, long long n, int32_t* out) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    long long i = 0;
    for (; i + 31 < n; i += 32) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b0 + i));
        const __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b1 + i));
        const __m128i va_lo_128 = _mm256_castsi256_si128(va);
        const __m128i va_hi_128 = _mm256_extracti128_si256(va, 1);
        const __m256i va_lo = _mm256_cvtepi8_epi16(va_lo_128);
        const __m256i va_hi = _mm256_cvtepi8_epi16(va_hi_128);

        const __m128i vb0_lo_128 = _mm256_castsi256_si128(vb0);
        const __m128i vb0_hi_128 = _mm256_extracti128_si256(vb0, 1);
        const __m256i vb0_lo = _mm256_cvtepi8_epi16(vb0_lo_128);
        const __m256i vb0_hi = _mm256_cvtepi8_epi16(vb0_hi_128);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_mullo_epi16(va_lo, vb0_lo), ones));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_mullo_epi16(va_hi, vb0_hi), ones));

        const __m128i vb1_lo_128 = _mm256_castsi256_si128(vb1);
        const __m128i vb1_hi_128 = _mm256_extracti128_si256(vb1, 1);
        const __m256i vb1_lo = _mm256_cvtepi8_epi16(vb1_lo_128);
        const __m256i vb1_hi = _mm256_cvtepi8_epi16(vb1_hi_128);
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_mullo_epi16(va_lo, vb1_lo), ones));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_mullo_epi16(va_hi, vb1_hi), ones));
    }
    alignas(32) int32_t tmp0[8], tmp1[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp0), acc0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp1), acc1);
    int32_t s0 = 0;
    int32_t s1 = 0;
    for (int j = 0; j < 8; ++j) {
        s0 += tmp0[j];
        s1 += tmp1[j];
    }
    for (; i < n; ++i) {
        const int32_t av = static_cast<int32_t>(a[i]);
        s0 += av * static_cast<int32_t>(b0[i]);
        s1 += av * static_cast<int32_t>(b1[i]);
    }
    out[0] = s0;
    out[1] = s1;
}

static inline void dot_i8_avx2_4(
    const int8_t* a0,
    const int8_t* a1,
    const int8_t* a2,
    const int8_t* a3,
    const int8_t* b,
    long long n,
    int32_t* out)
{
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    long long i = 0;
    for (; i + 31 < n; i += 32) {
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        const __m128i vb_lo_128 = _mm256_castsi256_si128(vb);
        const __m128i vb_hi_128 = _mm256_extracti128_si256(vb, 1);
        const __m256i vb_lo = _mm256_cvtepi8_epi16(vb_lo_128);
        const __m256i vb_hi = _mm256_cvtepi8_epi16(vb_hi_128);

        const __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a0 + i));
        const __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a1 + i));
        const __m256i va2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a2 + i));
        const __m256i va3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a3 + i));

        const __m256i va0_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va0));
        const __m256i va0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va0, 1));
        const __m256i va1_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va1));
        const __m256i va1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va1, 1));
        const __m256i va2_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va2));
        const __m256i va2_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va2, 1));
        const __m256i va3_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va3));
        const __m256i va3_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va3, 1));

        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_mullo_epi16(va0_lo, vb_lo), ones));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_mullo_epi16(va0_hi, vb_hi), ones));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_mullo_epi16(va1_lo, vb_lo), ones));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_mullo_epi16(va1_hi, vb_hi), ones));
        acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_mullo_epi16(va2_lo, vb_lo), ones));
        acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_mullo_epi16(va2_hi, vb_hi), ones));
        acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_mullo_epi16(va3_lo, vb_lo), ones));
        acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_mullo_epi16(va3_hi, vb_hi), ones));
    }
    alignas(32) int32_t tmp0[8], tmp1[8], tmp2[8], tmp3[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp0), acc0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp1), acc1);
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp2), acc2);
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp3), acc3);
    int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int j = 0; j < 8; ++j) {
        s0 += tmp0[j];
        s1 += tmp1[j];
        s2 += tmp2[j];
        s3 += tmp3[j];
    }
    for (; i < n; ++i) {
        const int32_t bv = static_cast<int32_t>(b[i]);
        s0 += static_cast<int32_t>(a0[i]) * bv;
        s1 += static_cast<int32_t>(a1[i]) * bv;
        s2 += static_cast<int32_t>(a2[i]) * bv;
        s3 += static_cast<int32_t>(a3[i]) * bv;
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

static inline float hmax256_abs_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    return _mm_cvtss_f32(m);
}

static void quantize_i8_row(const float* x, int8_t* xq, long long n, float* scale_out) {
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 max_v = _mm256_setzero_ps();
    long long i = 0;
    for (; i + 7 < n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        max_v = _mm256_max_ps(max_v, _mm256_and_ps(v, sign_mask));
    }
    float max_abs = hmax256_abs_ps(max_v);
    for (; i < n; ++i) max_abs = std::max(max_abs, std::fabs(x[i]));
    const float scale = std::max(max_abs, 1.0e-6f) / 127.0f;
    const float inv = 1.0f / scale;
    i = 0;
    const __m256 inv_v = _mm256_set1_ps(inv);
    const __m256 min_v = _mm256_set1_ps(-127.0f);
    const __m256 max_q_v = _mm256_set1_ps(127.0f);
    for (; i + 7 < n; i += 8) {
        __m256 qf = _mm256_mul_ps(_mm256_loadu_ps(x + i), inv_v);
        qf = _mm256_min_ps(max_q_v, _mm256_max_ps(min_v, _mm256_round_ps(qf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
        const __m256i qi32 = _mm256_cvtps_epi32(qf);
        alignas(32) int32_t tmp[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), qi32);
        for (int j = 0; j < 8; ++j) xq[i + j] = static_cast<int8_t>(tmp[j]);
    }
    for (; i < n; ++i) {
        int q = static_cast<int>(std::nearbyint(x[i] * inv));
        q = std::max(-127, std::min(127, q));
        xq[i] = static_cast<int8_t>(q);
    }
    *scale_out = scale;
}

static void int8_expert_accumulate_prequant_impl(
    const int8_t* xq,
    float x_scale,
    const int8_t* w1,
    const float* s1,
    const int8_t* w2,
    const float* s2,
    const int8_t* w3,
    const float* s3,
    float* hidden,
    int8_t* hq,
    float* y,
    long long hidden_dim,
    long long inter_dim,
    float swiglu_limit,
    float route_w)
{
    #pragma omp parallel for schedule(static)
    for (long long o = 0; o < inter_dim; ++o) {
        const int32_t acc1 = dot_i8_avx2(xq, w1 + o * hidden_dim, hidden_dim);
        const int32_t acc3 = dot_i8_avx2(xq, w3 + o * hidden_dim, hidden_dim);
        float gate = static_cast<float>(acc1) * x_scale * s1[o];
        float up = static_cast<float>(acc3) * x_scale * s3[o];
        if (swiglu_limit > 0.0f) {
            up = std::max(-swiglu_limit, std::min(swiglu_limit, up));
            gate = std::min(swiglu_limit, gate);
        }
        hidden[o] = route_w * silu_scalar(gate) * up;
    }
    float h_scale = 1.0f;
    quantize_i8_row(hidden, hq, inter_dim, &h_scale);
    #pragma omp parallel for schedule(static)
    for (long long o = 0; o < hidden_dim; ++o) {
        const int32_t acc2 = dot_i8_avx2(hq, w2 + o * inter_dim, inter_dim);
        y[o] += static_cast<float>(acc2) * h_scale * s2[o];
    }
}

static void int8_expert_accumulate_prequant_serial_impl(
    const int8_t* xq,
    float x_scale,
    const int8_t* w1,
    const float* s1,
    const int8_t* w2,
    const float* s2,
    const int8_t* w3,
    const float* s3,
    float* hidden,
    int8_t* hq,
    float* y,
    long long hidden_dim,
    long long inter_dim,
    float swiglu_limit,
    float route_w)
{
    for (long long o = 0; o < inter_dim; ++o) {
        int32_t accs[2];
        dot_i8_avx2_pair(xq, w1 + o * hidden_dim, w3 + o * hidden_dim, hidden_dim, accs);
        float gate = static_cast<float>(accs[0]) * x_scale * s1[o];
        float up = static_cast<float>(accs[1]) * x_scale * s3[o];
        if (swiglu_limit > 0.0f) {
            up = std::max(-swiglu_limit, std::min(swiglu_limit, up));
            gate = std::min(swiglu_limit, gate);
        }
        hidden[o] = route_w * silu_scalar(gate) * up;
    }
    float h_scale = 1.0f;
    quantize_i8_row(hidden, hq, inter_dim, &h_scale);
    for (long long o = 0; o < hidden_dim; ++o) {
        const int32_t acc2 = dot_i8_avx2(hq, w2 + o * inter_dim, inter_dim);
        y[o] = static_cast<float>(acc2) * h_scale * s2[o];
    }
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
    std::vector<int8_t> hq(inter_dim);
    std::fill(y, y + hidden_dim, 0.0f);
    int8_expert_accumulate_prequant_impl(
        xq.data(), x_scale, w1, s1, w2, s2, w3, s3, hidden.data(), hq.data(), y,
        hidden_dim, inter_dim, swiglu_limit, route_w);
}

struct Int8DecodeRoute {
    long long token;
    long long expert_id;
    float weight;
};

static void int8_single_token_accumulate_routes_prealloc_impl(
    const float* input,
    const int64_t* expert_ids,
    const float* weights,
    float* output,
    long long hidden_dim,
    long long topk,
    long long inter_dim,
    long long experts_start_idx,
    long long experts_end_idx,
    const int64_t* w1_ptrs,
    const int64_t* w2_ptrs,
    const int64_t* w3_ptrs,
    const int64_t* s1_ptrs,
    const int64_t* s2_ptrs,
    const int64_t* s3_ptrs,
    float swiglu_limit,
    int8_t* xq,
    float* hidden,
    int8_t* hq,
    float* h_scales,
    bool use_persistent_team)
{
    Int8DecodeRoute routes[16];
    long long route_count = 0;
    for (long long k = 0; k < topk && route_count < 16; ++k) {
        const long long expert_id = expert_ids[k];
        if (expert_id < experts_start_idx || expert_id >= experts_end_idx) continue;
        if (!w1_ptrs[expert_id] || !w2_ptrs[expert_id] || !w3_ptrs[expert_id] || !s1_ptrs[expert_id] || !s2_ptrs[expert_id] || !s3_ptrs[expert_id]) continue;
        routes[route_count++] = {0, expert_id, weights[k]};
    }
    if (route_count == 0) return;

    float x_scale = 1.0f;
    quantize_i8_row(input, xq, hidden_dim, &x_scale);

    auto compute_hidden = [&](long long start, long long end, int) {
        for (long long ro = start; ro < end; ++ro) {
            const long long r = ro / inter_dim;
            const long long o = ro - r * inter_dim;
            const Int8DecodeRoute& route = routes[r];
            const auto* w1 = reinterpret_cast<const int8_t*>(w1_ptrs[route.expert_id]);
            const auto* w3 = reinterpret_cast<const int8_t*>(w3_ptrs[route.expert_id]);
            const auto* s1 = reinterpret_cast<const float*>(s1_ptrs[route.expert_id]);
            const auto* s3 = reinterpret_cast<const float*>(s3_ptrs[route.expert_id]);
            int32_t accs[2];
            dot_i8_avx2_pair(xq, w1 + o * hidden_dim, w3 + o * hidden_dim, hidden_dim, accs);
            float gate = static_cast<float>(accs[0]) * x_scale * s1[o];
            float up = static_cast<float>(accs[1]) * x_scale * s3[o];
            if (swiglu_limit > 0.0f) {
                up = std::max(-swiglu_limit, std::min(swiglu_limit, up));
                gate = std::min(swiglu_limit, gate);
            }
            hidden[ro] = route.weight * silu_scalar(gate) * up;
        }
    };
    auto compute_scales = [&](long long start, long long end, int) {
        for (long long r = start; r < end; ++r) {
            quantize_i8_row(hidden + r * inter_dim, hq + r * inter_dim, inter_dim, &h_scales[r]);
        }
    };
    auto compute_output = [&](long long start, long long end, int) {
        for (long long o = start; o < end; ++o) {
            float acc_out = 0.0f;
            for (long long r = 0; r < route_count; ++r) {
                const Int8DecodeRoute& route = routes[r];
                const auto* w2 = reinterpret_cast<const int8_t*>(w2_ptrs[route.expert_id]);
                const auto* s2 = reinterpret_cast<const float*>(s2_ptrs[route.expert_id]);
                const int32_t acc2 = dot_i8_avx2(hq + r * inter_dim, w2 + o * inter_dim, inter_dim);
                acc_out += static_cast<float>(acc2) * h_scales[r] * s2[o];
            }
            output[o] += acc_out;
        }
    };

    if (use_persistent_team) {
        g_persistent_team.parallel_for(0, route_count * inter_dim, compute_hidden);
        g_persistent_team.parallel_for(0, route_count, compute_scales);
        g_persistent_team.parallel_for(0, hidden_dim, compute_output);
    } else {
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (long long ro = 0; ro < route_count * inter_dim; ++ro) compute_hidden(ro, ro + 1, 0);
            #pragma omp for schedule(static)
            for (long long r = 0; r < route_count; ++r) compute_scales(r, r + 1, 0);
            #pragma omp for schedule(static)
            for (long long o = 0; o < hidden_dim; ++o) compute_output(o, o + 1, 0);
        }
    }
}

static void int8_single_token_accumulate_routes_impl(
    const float* input,
    const int64_t* expert_ids,
    const float* weights,
    float* output,
    long long hidden_dim,
    long long topk,
    long long inter_dim,
    long long experts_start_idx,
    long long experts_end_idx,
    const int64_t* w1_ptrs,
    const int64_t* w2_ptrs,
    const int64_t* w3_ptrs,
    const int64_t* s1_ptrs,
    const int64_t* s2_ptrs,
    const int64_t* s3_ptrs,
    float swiglu_limit,
    bool use_persistent_team)
{
    Int8DecodeRoute routes[16];
    long long route_count = 0;
    for (long long k = 0; k < topk && route_count < 16; ++k) {
        const long long expert_id = expert_ids[k];
        if (expert_id < experts_start_idx || expert_id >= experts_end_idx) continue;
        if (!w1_ptrs[expert_id] || !w2_ptrs[expert_id] || !w3_ptrs[expert_id] || !s1_ptrs[expert_id] || !s2_ptrs[expert_id] || !s3_ptrs[expert_id]) continue;
        routes[route_count++] = {0, expert_id, weights[k]};
    }
    if (route_count == 0) return;

    std::vector<int8_t> xq(hidden_dim);
    std::vector<float> hidden(route_count * inter_dim);
    std::vector<int8_t> hq(route_count * inter_dim);
    std::vector<float> h_scales(route_count);
    float x_scale = 1.0f;
    quantize_i8_row(input, xq.data(), hidden_dim, &x_scale);

    auto compute_hidden = [&](long long start, long long end, int) {
        for (long long ro = start; ro < end; ++ro) {
            const long long r = ro / inter_dim;
            const long long o = ro - r * inter_dim;
            const Int8DecodeRoute& route = routes[r];
            const auto* w1 = reinterpret_cast<const int8_t*>(w1_ptrs[route.expert_id]);
            const auto* w3 = reinterpret_cast<const int8_t*>(w3_ptrs[route.expert_id]);
            const auto* s1 = reinterpret_cast<const float*>(s1_ptrs[route.expert_id]);
            const auto* s3 = reinterpret_cast<const float*>(s3_ptrs[route.expert_id]);
            int32_t accs[2];
            dot_i8_avx2_pair(xq.data(), w1 + o * hidden_dim, w3 + o * hidden_dim, hidden_dim, accs);
            float gate = static_cast<float>(accs[0]) * x_scale * s1[o];
            float up = static_cast<float>(accs[1]) * x_scale * s3[o];
            if (swiglu_limit > 0.0f) {
                up = std::max(-swiglu_limit, std::min(swiglu_limit, up));
                gate = std::min(swiglu_limit, gate);
            }
            hidden[ro] = route.weight * silu_scalar(gate) * up;
        }
    };
    auto compute_scales = [&](long long start, long long end, int) {
        for (long long r = start; r < end; ++r) {
            quantize_i8_row(hidden.data() + r * inter_dim, hq.data() + r * inter_dim, inter_dim, &h_scales[r]);
        }
    };
    auto compute_output = [&](long long start, long long end, int) {
        for (long long o = start; o < end; ++o) {
            float acc_out = 0.0f;
            for (long long r = 0; r < route_count; ++r) {
                const Int8DecodeRoute& route = routes[r];
                const auto* w2 = reinterpret_cast<const int8_t*>(w2_ptrs[route.expert_id]);
                const auto* s2 = reinterpret_cast<const float*>(s2_ptrs[route.expert_id]);
                const int32_t acc2 = dot_i8_avx2(hq.data() + r * inter_dim, w2 + o * inter_dim, inter_dim);
                acc_out += static_cast<float>(acc2) * h_scales[r] * s2[o];
            }
            output[o] += acc_out;
        }
    };

    if (use_persistent_team) {
        g_persistent_team.parallel_for(0, route_count * inter_dim, compute_hidden);
        g_persistent_team.parallel_for(0, route_count, compute_scales);
        g_persistent_team.parallel_for(0, hidden_dim, compute_output);
    } else {
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (long long ro = 0; ro < route_count * inter_dim; ++ro) compute_hidden(ro, ro + 1, 0);
            #pragma omp for schedule(static)
            for (long long r = 0; r < route_count; ++r) compute_scales(r, r + 1, 0);
            #pragma omp for schedule(static)
            for (long long o = 0; o < hidden_dim; ++o) compute_output(o, o + 1, 0);
        }
    }
}

static void int8_single_token_topk_parallel_impl(
    const float* input,
    const int64_t* expert_ids,
    const float* weights,
    float* output,
    long long hidden_dim,
    long long topk,
    long long inter_dim,
    long long experts_start_idx,
    long long experts_end_idx,
    const int64_t* w1_ptrs,
    const int64_t* w2_ptrs,
    const int64_t* w3_ptrs,
    const int64_t* s1_ptrs,
    const int64_t* s2_ptrs,
    const int64_t* s3_ptrs,
    float swiglu_limit)
{
    Int8DecodeRoute routes[16];
    long long route_count = 0;
    for (long long k = 0; k < topk && route_count < 16; ++k) {
        const long long expert_id = expert_ids[k];
        if (expert_id < experts_start_idx || expert_id >= experts_end_idx) continue;
        if (!w1_ptrs[expert_id] || !w2_ptrs[expert_id] || !w3_ptrs[expert_id] || !s1_ptrs[expert_id] || !s2_ptrs[expert_id] || !s3_ptrs[expert_id]) continue;
        routes[route_count++] = {0, expert_id, weights[k]};
    }
    std::fill(output, output + hidden_dim, 0.0f);
    if (route_count == 0) return;

    std::vector<int8_t> xq(hidden_dim);
    float x_scale = 1.0f;
    quantize_i8_row(input, xq.data(), hidden_dim, &x_scale);
    std::vector<float> hidden(route_count * inter_dim);
    std::vector<int8_t> hq(route_count * inter_dim);
    std::vector<float> h_scales(route_count, 1.0f);
    std::vector<float> partial(route_count * hidden_dim);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int n_threads = omp_get_num_threads();
        const int route_count_i = static_cast<int>(route_count);
        const int expert_lane = tid % route_count_i;
        const int lane_id = tid / route_count_i;
        const int lanes_per_expert = (n_threads - 1 - expert_lane) / route_count_i + 1;

        if (expert_lane < route_count) {
            const Int8DecodeRoute& route = routes[expert_lane];
            const auto* w1 = reinterpret_cast<const int8_t*>(w1_ptrs[route.expert_id]);
            const auto* w2 = reinterpret_cast<const int8_t*>(w2_ptrs[route.expert_id]);
            const auto* w3 = reinterpret_cast<const int8_t*>(w3_ptrs[route.expert_id]);
            const auto* s1 = reinterpret_cast<const float*>(s1_ptrs[route.expert_id]);
            const auto* s2 = reinterpret_cast<const float*>(s2_ptrs[route.expert_id]);
            const auto* s3 = reinterpret_cast<const float*>(s3_ptrs[route.expert_id]);
            float* h = hidden.data() + expert_lane * inter_dim;
            int8_t* hq_row = hq.data() + expert_lane * inter_dim;
            float* y = partial.data() + expert_lane * hidden_dim;

            for (long long o = lane_id; o < inter_dim; o += lanes_per_expert) {
                int32_t accs[2];
                dot_i8_avx2_pair(xq.data(), w1 + o * hidden_dim, w3 + o * hidden_dim, hidden_dim, accs);
                float gate = static_cast<float>(accs[0]) * x_scale * s1[o];
                float up = static_cast<float>(accs[1]) * x_scale * s3[o];
                if (swiglu_limit > 0.0f) {
                    up = std::max(-swiglu_limit, std::min(swiglu_limit, up));
                    gate = std::min(swiglu_limit, gate);
                }
                h[o] = route.weight * silu_scalar(gate) * up;
            }

            #pragma omp barrier

            #pragma omp single
            {
                for (long long r = 0; r < route_count; ++r) {
                    quantize_i8_row(hidden.data() + r * inter_dim, hq.data() + r * inter_dim, inter_dim, &h_scales[r]);
                }
            }

            #pragma omp barrier

            for (long long o = lane_id; o < hidden_dim; o += lanes_per_expert) {
                const int32_t acc2 = dot_i8_avx2(hq_row, w2 + o * inter_dim, inter_dim);
                y[o] = static_cast<float>(acc2) * h_scales[expert_lane] * s2[o];
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (long long o = 0; o < hidden_dim; ++o) {
        float acc = 0.0f;
        for (long long r = 0; r < route_count; ++r) {
            acc += partial[r * hidden_dim + o];
        }
        output[o] = acc;
    }
}

static void int8_single_token_topk_persistent_impl(
    const float* input,
    const int64_t* expert_ids,
    const float* weights,
    float* output,
    long long hidden_dim,
    long long topk,
    long long inter_dim,
    long long experts_start_idx,
    long long experts_end_idx,
    const int64_t* w1_ptrs,
    const int64_t* w2_ptrs,
    const int64_t* w3_ptrs,
    const int64_t* s1_ptrs,
    const int64_t* s2_ptrs,
    const int64_t* s3_ptrs,
    float swiglu_limit)
{
    Int8DecodeRoute routes[16];
    long long route_count = 0;
    for (long long k = 0; k < topk && route_count < 16; ++k) {
        const long long expert_id = expert_ids[k];
        if (expert_id < experts_start_idx || expert_id >= experts_end_idx) continue;
        if (!w1_ptrs[expert_id] || !w2_ptrs[expert_id] || !w3_ptrs[expert_id] || !s1_ptrs[expert_id] || !s2_ptrs[expert_id] || !s3_ptrs[expert_id]) continue;
        routes[route_count++] = {0, expert_id, weights[k]};
    }
    std::fill(output, output + hidden_dim, 0.0f);
    if (route_count == 0) return;

    // Optional internal profile (inline path; the server path has its own profiler).
    // DEEPSEEK_CPU_MOE_INLINE_PROFILE=1 turns it on; the same _PROFILE_EVERY env (default 256)
    // controls how many requests are accumulated before printing a row.
    static const bool kProfile = []() {
        const char* env = std::getenv("DEEPSEEK_CPU_MOE_INLINE_PROFILE");
        return env && (env[0] == '1' || env[0] == 't' || env[0] == 'T');
    }();
    static const long long kProfileEvery = []() -> long long {
        const char* env = std::getenv("DEEPSEEK_CPU_MOE_INLINE_PROFILE_EVERY");
        long long n = env ? std::atoll(env) : 256;
        return n > 0 ? n : 256;
    }();
    static thread_local long long s_calls = 0;
    static thread_local double s_t_quant_x = 0.0;
    static thread_local double s_t_hidden = 0.0;
    static thread_local double s_t_qhid = 0.0;
    static thread_local double s_t_output = 0.0;
    static thread_local double s_t_reduce = 0.0;
    auto profile_now = []() -> double {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
    };

    // Persistent per-call scratch buffers. The previous std::vector path malloc'd ~80KB total
    // every call (xq, hidden, hq, h_scales, partial). Hoisting to thread_local resizable vectors
    // keeps the same memory addresses warm across hundreds of layers and amortizes malloc cost.
    static thread_local std::vector<int8_t> tl_xq;
    static thread_local std::vector<float> tl_hidden;
    static thread_local std::vector<int8_t> tl_hq;
    static thread_local std::vector<float> tl_h_scales;
    static thread_local std::vector<float> tl_partial;
    if (static_cast<long long>(tl_xq.size()) < hidden_dim) tl_xq.resize(hidden_dim);
    const size_t hidden_total = static_cast<size_t>(route_count) * static_cast<size_t>(inter_dim);
    const size_t partial_total = static_cast<size_t>(route_count) * static_cast<size_t>(hidden_dim);
    if (tl_hidden.size() < hidden_total) tl_hidden.resize(hidden_total);
    if (tl_hq.size() < hidden_total) tl_hq.resize(hidden_total);
    if (static_cast<long long>(tl_h_scales.size()) < route_count) tl_h_scales.resize(route_count, 1.0f);
    if (tl_partial.size() < partial_total) tl_partial.resize(partial_total);
    int8_t* xq = tl_xq.data();
    float* hidden = tl_hidden.data();
    int8_t* hq = tl_hq.data();
    float* h_scales = tl_h_scales.data();
    float* partial = tl_partial.data();

    float x_scale = 1.0f;
    const double t0 = kProfile ? profile_now() : 0.0;
    quantize_i8_row(input, xq, hidden_dim, &x_scale);
    const double t1 = kProfile ? profile_now() : 0.0;

    auto compute_hidden = [&](long long start, long long end, int) {
        for (long long ro = start; ro < end; ++ro) {
            const long long r = ro / inter_dim;
            const long long o = ro - r * inter_dim;
            const Int8DecodeRoute& route = routes[r];
            const auto* w1 = reinterpret_cast<const int8_t*>(w1_ptrs[route.expert_id]);
            const auto* w3 = reinterpret_cast<const int8_t*>(w3_ptrs[route.expert_id]);
            const auto* s1 = reinterpret_cast<const float*>(s1_ptrs[route.expert_id]);
            const auto* s3 = reinterpret_cast<const float*>(s3_ptrs[route.expert_id]);
            int32_t accs[2];
            dot_i8_avx2_pair(xq, w1 + o * hidden_dim, w3 + o * hidden_dim, hidden_dim, accs);
            float gate = static_cast<float>(accs[0]) * x_scale * s1[o];
            float up = static_cast<float>(accs[1]) * x_scale * s3[o];
            if (swiglu_limit > 0.0f) {
                up = std::max(-swiglu_limit, std::min(swiglu_limit, up));
                gate = std::min(swiglu_limit, gate);
            }
            hidden[ro] = route.weight * silu_scalar(gate) * up;
        }
    };
    g_persistent_team.parallel_for(0, route_count * inter_dim, compute_hidden);
    const double t2 = kProfile ? profile_now() : 0.0;

    auto compute_scales = [&](long long start, long long end, int) {
        for (long long r = start; r < end; ++r) {
            quantize_i8_row(hidden + r * inter_dim, hq + r * inter_dim, inter_dim, &h_scales[r]);
        }
    };
    g_persistent_team.parallel_for(0, route_count, compute_scales);
    const double t3 = kProfile ? profile_now() : 0.0;

    auto compute_output = [&](long long start, long long end, int) {
        for (long long ro = start; ro < end; ++ro) {
            const long long r = ro / hidden_dim;
            const long long o = ro - r * hidden_dim;
            const Int8DecodeRoute& route = routes[r];
            const auto* w2 = reinterpret_cast<const int8_t*>(w2_ptrs[route.expert_id]);
            const auto* s2 = reinterpret_cast<const float*>(s2_ptrs[route.expert_id]);
            const int32_t acc2 = dot_i8_avx2(hq + r * inter_dim, w2 + o * inter_dim, inter_dim);
            partial[ro] = static_cast<float>(acc2) * h_scales[r] * s2[o];
        }
    };
    g_persistent_team.parallel_for(0, route_count * hidden_dim, compute_output);
    const double t4 = kProfile ? profile_now() : 0.0;

    auto reduce_output = [&](long long start, long long end, int) {
        for (long long o = start; o < end; ++o) {
            float acc = 0.0f;
            for (long long r = 0; r < route_count; ++r) {
                acc += partial[r * hidden_dim + o];
            }
            output[o] = acc;
        }
    };
    g_persistent_team.parallel_for(0, hidden_dim, reduce_output);
    const double t5 = kProfile ? profile_now() : 0.0;

    if (kProfile) {
        s_calls++;
        s_t_quant_x += t1 - t0;
        s_t_hidden  += t2 - t1;
        s_t_qhid    += t3 - t2;
        s_t_output  += t4 - t3;
        s_t_reduce  += t5 - t4;
        if (s_calls >= kProfileEvery) {
            const double n = double(s_calls);
            fprintf(stderr,
                    "cpu_moe_inline_profile_v1 calls=%lld avg_quant_x=%.6fs avg_hidden=%.6fs "
                    "avg_qhid=%.6fs avg_output=%.6fs avg_reduce=%.6fs avg_total=%.6fs route_count=%lld\n",
                    s_calls,
                    s_t_quant_x / n, s_t_hidden / n, s_t_qhid / n,
                    s_t_output / n, s_t_reduce / n,
                    (s_t_quant_x + s_t_hidden + s_t_qhid + s_t_output + s_t_reduce) / n,
                    route_count);
            fflush(stderr);
            s_calls = 0;
            s_t_quant_x = s_t_hidden = s_t_qhid = s_t_output = s_t_reduce = 0.0;
        }
    }
}

static void int8_decode_accumulate_routes_impl(
    const float* input,
    const int64_t* expert_ids,
    const float* weights,
    float* output,
    long long tokens,
    long long hidden_dim,
    long long topk,
    long long inter_dim,
    long long experts_start_idx,
    long long experts_end_idx,
    const int64_t* w1_ptrs,
    const int64_t* w2_ptrs,
    const int64_t* w3_ptrs,
    const int64_t* s1_ptrs,
    const int64_t* s2_ptrs,
    const int64_t* s3_ptrs,
    float swiglu_limit)
{
    std::vector<Int8DecodeRoute> routes;
    routes.reserve(tokens * topk);
    std::vector<uint8_t> token_has_route(tokens, 0);
    for (long long t = 0; t < tokens; ++t) {
        for (long long k = 0; k < topk; ++k) {
            const long long expert_id = expert_ids[t * topk + k];
            if (expert_id < experts_start_idx || expert_id >= experts_end_idx) continue;
            if (!w1_ptrs[expert_id] || !w2_ptrs[expert_id] || !w3_ptrs[expert_id] || !s1_ptrs[expert_id] || !s2_ptrs[expert_id] || !s3_ptrs[expert_id]) continue;
            routes.push_back({t, expert_id, weights[t * topk + k]});
            token_has_route[t] = 1;
        }
    }
    const long long route_count = static_cast<long long>(routes.size());
    if (route_count == 0) return;

    std::vector<int8_t> xq_all(tokens * hidden_dim);
    std::vector<float> x_scales(tokens);
    #pragma omp parallel for schedule(static)
    for (long long t = 0; t < tokens; ++t) {
        if (token_has_route[t]) {
            quantize_i8_row(input + t * hidden_dim, xq_all.data() + t * hidden_dim, hidden_dim, &x_scales[t]);
        }
    }

    std::vector<float> hidden(route_count * inter_dim);
    std::vector<int8_t> hq(route_count * inter_dim);
    std::vector<float> h_scales(route_count);

    #pragma omp parallel for collapse(2) schedule(static)
    for (long long r = 0; r < route_count; ++r) {
        for (long long o = 0; o < inter_dim; ++o) {
            const Int8DecodeRoute& route = routes[r];
            const auto* w1 = reinterpret_cast<const int8_t*>(w1_ptrs[route.expert_id]);
            const auto* w3 = reinterpret_cast<const int8_t*>(w3_ptrs[route.expert_id]);
            const auto* s1 = reinterpret_cast<const float*>(s1_ptrs[route.expert_id]);
            const auto* s3 = reinterpret_cast<const float*>(s3_ptrs[route.expert_id]);
            const int8_t* xq = xq_all.data() + route.token * hidden_dim;
            const float x_scale = x_scales[route.token];
            const int32_t acc1 = dot_i8_avx2(xq, w1 + o * hidden_dim, hidden_dim);
            const int32_t acc3 = dot_i8_avx2(xq, w3 + o * hidden_dim, hidden_dim);
            float gate = static_cast<float>(acc1) * x_scale * s1[o];
            float up = static_cast<float>(acc3) * x_scale * s3[o];
            if (swiglu_limit > 0.0f) {
                up = std::max(-swiglu_limit, std::min(swiglu_limit, up));
                gate = std::min(swiglu_limit, gate);
            }
            hidden[r * inter_dim + o] = route.weight * silu_scalar(gate) * up;
        }
    }

    #pragma omp parallel for schedule(static)
    for (long long r = 0; r < route_count; ++r) {
        quantize_i8_row(hidden.data() + r * inter_dim, hq.data() + r * inter_dim, inter_dim, &h_scales[r]);
    }

    #pragma omp parallel for schedule(static)
    for (long long o = 0; o < hidden_dim; ++o) {
        for (long long r = 0; r < route_count; ++r) {
            const Int8DecodeRoute& route = routes[r];
            const auto* w2 = reinterpret_cast<const int8_t*>(w2_ptrs[route.expert_id]);
            const auto* s2 = reinterpret_cast<const float*>(s2_ptrs[route.expert_id]);
            const int32_t acc2 = dot_i8_avx2(hq.data() + r * inter_dim, w2 + o * inter_dim, inter_dim);
            output[route.token * hidden_dim + o] += static_cast<float>(acc2) * h_scales[r] * s2[o];
        }
    }
}

static void int8_expert_accumulate_group_impl(
    const int8_t* xq_all,
    const float* x_scales,
    const long long* token_ids,
    const float* route_weights,
    long long group_size,
    const int8_t* w1,
    const float* s1,
    const int8_t* w2,
    const float* s2,
    const int8_t* w3,
    const float* s3,
    float* output,
    long long hidden_dim,
    long long inter_dim,
    float swiglu_limit)
{
    if (group_size <= 0) return;
    std::vector<float> hidden(group_size * inter_dim);
    std::vector<int8_t> hq(group_size * inter_dim);
    std::vector<float> h_scales(group_size);

    #pragma omp parallel for schedule(static)
    for (long long o = 0; o < inter_dim; ++o) {
        const int8_t* w1_row = w1 + o * hidden_dim;
        const int8_t* w3_row = w3 + o * hidden_dim;
        long long j = 0;
        for (; j + 3 < group_size; j += 4) {
            const long long t0 = token_ids[j];
            const long long t1 = token_ids[j + 1];
            const long long t2 = token_ids[j + 2];
            const long long t3 = token_ids[j + 3];
            const int8_t* xq0 = xq_all + t0 * hidden_dim;
            const int8_t* xq1 = xq_all + t1 * hidden_dim;
            const int8_t* xq2 = xq_all + t2 * hidden_dim;
            const int8_t* xq3 = xq_all + t3 * hidden_dim;
            int32_t acc1[4];
            int32_t acc3[4];
            dot_i8_avx2_4(xq0, xq1, xq2, xq3, w1_row, hidden_dim, acc1);
            dot_i8_avx2_4(xq0, xq1, xq2, xq3, w3_row, hidden_dim, acc3);
            const float x_scale0 = x_scales[t0];
            const float x_scale1 = x_scales[t1];
            const float x_scale2 = x_scales[t2];
            const float x_scale3 = x_scales[t3];
            float gate0 = static_cast<float>(acc1[0]) * x_scale0 * s1[o];
            float gate1 = static_cast<float>(acc1[1]) * x_scale1 * s1[o];
            float gate2 = static_cast<float>(acc1[2]) * x_scale2 * s1[o];
            float gate3 = static_cast<float>(acc1[3]) * x_scale3 * s1[o];
            float up0 = static_cast<float>(acc3[0]) * x_scale0 * s3[o];
            float up1 = static_cast<float>(acc3[1]) * x_scale1 * s3[o];
            float up2 = static_cast<float>(acc3[2]) * x_scale2 * s3[o];
            float up3 = static_cast<float>(acc3[3]) * x_scale3 * s3[o];
            if (swiglu_limit > 0.0f) {
                up0 = std::max(-swiglu_limit, std::min(swiglu_limit, up0));
                up1 = std::max(-swiglu_limit, std::min(swiglu_limit, up1));
                up2 = std::max(-swiglu_limit, std::min(swiglu_limit, up2));
                up3 = std::max(-swiglu_limit, std::min(swiglu_limit, up3));
                gate0 = std::min(swiglu_limit, gate0);
                gate1 = std::min(swiglu_limit, gate1);
                gate2 = std::min(swiglu_limit, gate2);
                gate3 = std::min(swiglu_limit, gate3);
            }
            hidden[j * inter_dim + o] = route_weights[j] * silu_scalar(gate0) * up0;
            hidden[(j + 1) * inter_dim + o] = route_weights[j + 1] * silu_scalar(gate1) * up1;
            hidden[(j + 2) * inter_dim + o] = route_weights[j + 2] * silu_scalar(gate2) * up2;
            hidden[(j + 3) * inter_dim + o] = route_weights[j + 3] * silu_scalar(gate3) * up3;
        }
        for (; j < group_size; ++j) {
            const long long t = token_ids[j];
            const int8_t* xq = xq_all + t * hidden_dim;
            const float x_scale = x_scales[t];
            const int32_t acc1 = dot_i8_avx2(xq, w1_row, hidden_dim);
            const int32_t acc3 = dot_i8_avx2(xq, w3_row, hidden_dim);
            float gate = static_cast<float>(acc1) * x_scale * s1[o];
            float up = static_cast<float>(acc3) * x_scale * s3[o];
            if (swiglu_limit > 0.0f) {
                up = std::max(-swiglu_limit, std::min(swiglu_limit, up));
                gate = std::min(swiglu_limit, gate);
            }
            hidden[j * inter_dim + o] = route_weights[j] * silu_scalar(gate) * up;
        }
    }

    for (long long j = 0; j < group_size; ++j) {
        quantize_i8_row(hidden.data() + j * inter_dim, hq.data() + j * inter_dim, inter_dim, &h_scales[j]);
    }

    #pragma omp parallel for schedule(static)
    for (long long o = 0; o < hidden_dim; ++o) {
        const int8_t* w2_row = w2 + o * inter_dim;
        const float row_scale = s2[o];
        long long j = 0;
        for (; j + 3 < group_size; j += 4) {
            const long long t0 = token_ids[j];
            const long long t1 = token_ids[j + 1];
            const long long t2 = token_ids[j + 2];
            const long long t3 = token_ids[j + 3];
            int32_t acc2[4];
            dot_i8_avx2_4(
                hq.data() + j * inter_dim,
                hq.data() + (j + 1) * inter_dim,
                hq.data() + (j + 2) * inter_dim,
                hq.data() + (j + 3) * inter_dim,
                w2_row,
                inter_dim,
                acc2);
            output[t0 * hidden_dim + o] += static_cast<float>(acc2[0]) * h_scales[j] * row_scale;
            output[t1 * hidden_dim + o] += static_cast<float>(acc2[1]) * h_scales[j + 1] * row_scale;
            output[t2 * hidden_dim + o] += static_cast<float>(acc2[2]) * h_scales[j + 2] * row_scale;
            output[t3 * hidden_dim + o] += static_cast<float>(acc2[3]) * h_scales[j + 3] * row_scale;
        }
        for (; j < group_size; ++j) {
            const long long t = token_ids[j];
            const int32_t acc2 = dot_i8_avx2(hq.data() + j * inter_dim, w2_row, inter_dim);
            output[t * hidden_dim + o] += static_cast<float>(acc2) * h_scales[j] * row_scale;
        }
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

struct CPUMoEServerHeader {
    long long request_seq;
    long long response_seq;
    long long layer_id;
    long long stop;
};

static inline long long atomic_load_ll(const long long* ptr) {
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

static inline void atomic_store_ll(long long* ptr, long long value) {
    __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
}
static PyObject* set_omp_num_threads(PyObject*, PyObject* args) {
    int n;
    if (!PyArg_ParseTuple(args, "i", &n)) return nullptr;
#ifdef _OPENMP
    g_omp_num_threads = n;
    omp_set_dynamic(0);
    omp_set_num_threads(n);
#endif
    Py_RETURN_NONE;
}

static inline float fp4_e8m0_scale(uint8_t scale_byte) {
    return std::exp2(static_cast<float>(static_cast<int>(scale_byte) - 127));
}

static void fused_fp4_w13_hidden_range_raw(
    const float* x_row,
    const uint8_t* w1,
    const uint8_t* w3,
    const uint8_t* s1,
    const uint8_t* s3,
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
        const uint8_t* s1_row = s1 + o * in_blocks;
        const uint8_t* s3_row = s3 + o * in_blocks;
        __m256 gate_acc = _mm256_setzero_ps();
        __m256 up_acc = _mm256_setzero_ps();

        for (long long b = 0; b < in_blocks; ++b) {
            const long long byte_base = b * kFp4BytesPerBlock;
            const long long feature_base = b * kFp4BlockSize;
            const __m256 scale1 = _mm256_set1_ps(fp4_e8m0_scale(s1_row[b]));
            const __m256 scale3 = _mm256_set1_ps(fp4_e8m0_scale(s3_row[b]));
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

static inline void fused_fp4_w13_hidden_raw(
    const float* x_row,
    const uint8_t* w1,
    const uint8_t* w3,
    const uint8_t* s1,
    const uint8_t* s3,
    long long hidden_dim,
    long long inter_dim,
    long long in_blocks,
    float swiglu_limit,
    float route_w,
    float* hidden)
{
    fused_fp4_w13_hidden_range_raw(
        x_row, w1, w3, s1, s3, hidden_dim, in_blocks, swiglu_limit, route_w, hidden, 0, inter_dim);
}

static void decode_fp4_w2_down_project_raw(
    const float* hidden,
    const uint8_t* w2,
    const uint8_t* s2,
    long long inter_dim,
    long long hidden_dim,
    long long out_blocks,
    float* y_row,
    long long o_start,
    long long o_end)
{
    const long long row_stride = inter_dim / 2;
    const __m256i even_idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(kEvenIdx8));
    const __m256i odd_idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(kOddIdx8));
    const __m128i nibble_mask = _mm_set1_epi8(0x0F);

    for (long long o = o_start; o < o_end; ++o) {
        const uint8_t* w2_row = w2 + o * row_stride;
        const uint8_t* s2_row = s2 + o * out_blocks;
        __m256 acc = _mm256_setzero_ps();
        for (long long b = 0; b < out_blocks; ++b) {
            const __m256 scale = _mm256_set1_ps(fp4_e8m0_scale(s2_row[b]));
            const long long feature_base = b * kFp4BlockSize;
            const uint8_t* w2_block = w2_row + b * kFp4BytesPerBlock;
            const float* h_block = hidden + feature_base;
            for (int chunk = 0; chunk < 2; ++chunk) {
                const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w2_block + chunk * 8));
                const __m128i lo = _mm_and_si128(raw, nibble_mask);
                const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), nibble_mask);
                const __m256 w_lo = _mm256_mul_ps(decode_fp4_nibbles8_to_ps(lo), scale);
                const __m256 w_hi = _mm256_mul_ps(decode_fp4_nibbles8_to_ps(hi), scale);
                const float* h_chunk = h_block + chunk * 16;
                const __m256 h_even = _mm256_i32gather_ps(h_chunk, even_idx, 4);
                const __m256 h_odd = _mm256_i32gather_ps(h_chunk, odd_idx, 4);
                acc = _mm256_fmadd_ps(w_lo, h_even, acc);
                acc = _mm256_fmadd_ps(w_hi, h_odd, acc);
            }
        }
        y_row[o] += hsum256_ps(acc);
    }
}

static PyObject* routed_fp4_moe_forward_raw(PyObject*, PyObject* args) {
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
    apply_omp_runtime_config();
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
                    auto* s1 = reinterpret_cast<uint8_t*>(s1_ptrs[expert_id]);
                    auto* s2 = reinterpret_cast<uint8_t*>(s2_ptrs[expert_id]);
                    auto* s3 = reinterpret_cast<uint8_t*>(s3_ptrs[expert_id]);
                    if (!w1 || !w2 || !w3 || !s1 || !s2 || !s3) continue;

                    fused_fp4_w13_hidden_raw(
                        x_row, w1, w3, s1, s3, hidden_dim, inter_dim, in_blocks,
                        swiglu_limit, weights[t * topk + k], hidden.data());
                    for (long long o_start = 0; o_start < hidden_dim; o_start += kW2TileRows) {
                        long long o_end = o_start + kW2TileRows;
                        if (o_end > hidden_dim) o_end = hidden_dim;
                        decode_fp4_w2_down_project_raw(hidden.data(), w2, s2, inter_dim, hidden_dim, out_blocks, y_row, o_start, o_end);
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
            auto* s1 = reinterpret_cast<uint8_t*>(s1_ptrs[expert_id]);
            auto* s2 = reinterpret_cast<uint8_t*>(s2_ptrs[expert_id]);
            auto* s3 = reinterpret_cast<uint8_t*>(s3_ptrs[expert_id]);
            if (!w1 || !w2 || !w3 || !s1 || !s2 || !s3) continue;

            fused_fp4_w13_hidden_raw(
                x_row, w1, w3, s1, s3, hidden_dim, inter_dim, in_blocks,
                swiglu_limit, weights[k], hidden.data());
            #pragma omp parallel for schedule(static)
            for (long long o_start = 0; o_start < hidden_dim; o_start += kW2TileRows) {
                long long o_end = o_start + kW2TileRows;
                if (o_end > hidden_dim) o_end = hidden_dim;
                decode_fp4_w2_down_project_raw(hidden.data(), w2, s2, inter_dim, hidden_dim, out_blocks, y_row, o_start, o_end);
            }
        }
    }
    Py_END_ALLOW_THREADS

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
    apply_omp_runtime_config();
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
    apply_omp_runtime_config();
    std::fill(output, output + tokens * hidden_dim, 0.0f);
    if (tokens == 1) {
        int8_single_token_accumulate_routes_impl(
            input, expert_ids, weights, output, hidden_dim, topk, inter_dim,
            experts_start_idx, experts_end_idx, w1_ptrs, w2_ptrs, w3_ptrs, s1_ptrs, s2_ptrs, s3_ptrs,
            swiglu_limit,
            false);
    } else if (tokens <= 16) {
        int8_decode_accumulate_routes_impl(
            input, expert_ids, weights, output, tokens, hidden_dim, topk, inter_dim,
            experts_start_idx, experts_end_idx, w1_ptrs, w2_ptrs, w3_ptrs, s1_ptrs, s2_ptrs, s3_ptrs,
            swiglu_limit);
    } else {
        std::vector<int8_t> xq_all(tokens * hidden_dim);
        std::vector<float> x_scales(tokens);
        #pragma omp parallel for schedule(static)
        for (long long t = 0; t < tokens; ++t) {
            quantize_i8_row(input + t * hidden_dim, xq_all.data() + t * hidden_dim, hidden_dim, &x_scales[t]);
        }

        const long long local_experts = experts_end_idx - experts_start_idx;
        const long long total_routes = tokens * topk;
        std::vector<long long> counts(local_experts);
        std::vector<long long> offsets(local_experts + 1);
        std::vector<long long> cursor(local_experts);
        std::vector<long long> grouped_tokens(total_routes);
        std::vector<float> grouped_weights(total_routes);

        std::fill(counts.begin(), counts.end(), 0);
        for (long long t = 0; t < tokens; ++t) {
            for (long long k = 0; k < topk; ++k) {
                const long long expert_id = expert_ids[t * topk + k];
                if (expert_id < experts_start_idx || expert_id >= experts_end_idx) continue;
                ++counts[expert_id - experts_start_idx];
            }
        }
        offsets[0] = 0;
        for (long long e = 0; e < local_experts; ++e) offsets[e + 1] = offsets[e] + counts[e];
        std::copy(offsets.begin(), offsets.end() - 1, cursor.begin());
        for (long long t = 0; t < tokens; ++t) {
            for (long long k = 0; k < topk; ++k) {
                const long long expert_id = expert_ids[t * topk + k];
                if (expert_id < experts_start_idx || expert_id >= experts_end_idx) continue;
                const long long pos = cursor[expert_id - experts_start_idx]++;
                grouped_tokens[pos] = t;
                grouped_weights[pos] = weights[t * topk + k];
            }
        }

        for (long long e = 0; e < local_experts; ++e) {
            const long long group_start = offsets[e];
            const long long group_end = offsets[e + 1];
            const long long group_size = group_end - group_start;
            if (group_size <= 0) continue;
            const long long expert_id = experts_start_idx + e;
            auto* w1 = reinterpret_cast<int8_t*>(w1_ptrs[expert_id]);
            auto* w2 = reinterpret_cast<int8_t*>(w2_ptrs[expert_id]);
            auto* w3 = reinterpret_cast<int8_t*>(w3_ptrs[expert_id]);
            auto* s1 = reinterpret_cast<float*>(s1_ptrs[expert_id]);
            auto* s2 = reinterpret_cast<float*>(s2_ptrs[expert_id]);
            auto* s3 = reinterpret_cast<float*>(s3_ptrs[expert_id]);
            if (!w1 || !w2 || !w3 || !s1 || !s2 || !s3) continue;
            int8_expert_accumulate_group_impl(
                xq_all.data(), x_scales.data(), grouped_tokens.data() + group_start,
                grouped_weights.data() + group_start, group_size,
                w1, s1, w2, s2, w3, s3, output,
                hidden_dim, inter_dim, swiglu_limit);
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
    apply_omp_runtime_config();
    int8_expert_forward_impl(x, w1, s1, w2, s2, w3, s3, y, hidden_dim, inter_dim, swiglu_limit, route_w);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyObject* routed_int8_moe_forward_topk_parallel(PyObject*, PyObject* args) {
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
    apply_omp_runtime_config();
    if (tokens == 1) {
        int8_single_token_topk_parallel_impl(
            input, expert_ids, weights, output, hidden_dim, topk, inter_dim,
            experts_start_idx, experts_end_idx, w1_ptrs, w2_ptrs, w3_ptrs, s1_ptrs, s2_ptrs, s3_ptrs,
            swiglu_limit);
    } else {
        std::fill(output, output + tokens * hidden_dim, 0.0f);
    }
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyObject* routed_int8_moe_forward_persistent(PyObject*, PyObject* args) {
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
    apply_omp_runtime_config();
    std::fill(output, output + tokens * hidden_dim, 0.0f);
    if (tokens == 1) {
        int8_single_token_accumulate_routes_impl(
            input, expert_ids, weights, output, hidden_dim, topk, inter_dim,
            experts_start_idx, experts_end_idx, w1_ptrs, w2_ptrs, w3_ptrs, s1_ptrs, s2_ptrs, s3_ptrs,
            swiglu_limit,
            true);
    }
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyObject* routed_int8_moe_forward_prealloc_persistent(PyObject*, PyObject* args) {
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
    apply_omp_runtime_config();
    std::fill(output, output + tokens * hidden_dim, 0.0f);
    if (tokens == 1) {
        thread_local std::vector<int8_t> xq;
        thread_local std::vector<float> hidden;
        thread_local std::vector<int8_t> hq;
        thread_local std::vector<float> h_scales;
        const size_t route_cap = static_cast<size_t>(std::min<long long>(topk, 16));
        xq.resize(static_cast<size_t>(hidden_dim));
        hidden.resize(route_cap * static_cast<size_t>(inter_dim));
        hq.resize(route_cap * static_cast<size_t>(inter_dim));
        h_scales.resize(route_cap);
        int8_single_token_accumulate_routes_prealloc_impl(
            input, expert_ids, weights, output, hidden_dim, topk, inter_dim,
            experts_start_idx, experts_end_idx, w1_ptrs, w2_ptrs, w3_ptrs, s1_ptrs, s2_ptrs, s3_ptrs,
            swiglu_limit, xq.data(), hidden.data(), hq.data(), h_scales.data(), false);
    }
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyObject* routed_int8_moe_forward_topk_persistent(PyObject*, PyObject* args) {
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
    apply_omp_runtime_config();
    if (tokens == 1) {
        int8_single_token_topk_persistent_impl(
            input, expert_ids, weights, output, hidden_dim, topk, inter_dim,
            experts_start_idx, experts_end_idx, w1_ptrs, w2_ptrs, w3_ptrs, s1_ptrs, s2_ptrs, s3_ptrs,
            swiglu_limit);
    } else {
        std::fill(output, output + tokens * hidden_dim, 0.0f);
    }
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyObject* host_float_allreduce(PyObject*, PyObject* args) {
    unsigned long long data_ptr_ull;
    long long count, rank, world_size, slot;
    const char* name_c = nullptr;
    if (!PyArg_ParseTuple(args, "KLLLLs", &data_ptr_ull, &count, &rank, &world_size, &slot, &name_c)) {
        return nullptr;
    }
    if (count <= 0 || world_size <= 1 || rank < 0 || rank >= world_size || name_c == nullptr) {
        Py_RETURN_NONE;
    }

    auto* data = reinterpret_cast<float*>(data_ptr_ull);
    const std::string shm_name(name_c);
    const size_t header_longs = 2;
    const size_t data_floats = static_cast<size_t>(world_size) * static_cast<size_t>(count);
    const size_t bytes = header_longs * sizeof(uint64_t) + data_floats * sizeof(float);

    int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0600);
    if (fd < 0) {
        return PyErr_Format(PyExc_RuntimeError, "shm_open(%s) failed: %s", shm_name.c_str(), std::strerror(errno));
    }
    if (rank == 0) {
        if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
            const int err = errno;
            close(fd);
            return PyErr_Format(PyExc_RuntimeError, "ftruncate(%s) failed: %s", shm_name.c_str(), std::strerror(err));
        }
    } else {
        struct timespec ts{0, 1000000};
        for (int tries = 0; tries < 1000; ++tries) {
            off_t size = lseek(fd, 0, SEEK_END);
            if (size >= static_cast<off_t>(bytes)) break;
            nanosleep(&ts, nullptr);
        }
    }

    void* mapped = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        return PyErr_Format(PyExc_RuntimeError, "mmap(%s) failed: %s", shm_name.c_str(), std::strerror(errno));
    }

    auto* counters = reinterpret_cast<volatile uint64_t*>(mapped);
    auto* base = reinterpret_cast<float*>(reinterpret_cast<char*>(mapped) + header_longs * sizeof(uint64_t));
    auto* rank_buf = base + static_cast<size_t>(rank) * static_cast<size_t>(count);
    const uint64_t gen = static_cast<uint64_t>(slot) + 1;

    Py_BEGIN_ALLOW_THREADS
    std::memcpy(rank_buf, data, static_cast<size_t>(count) * sizeof(float));
    __sync_synchronize();
    __sync_fetch_and_add(&counters[0], 1);
    while (counters[0] < gen * static_cast<uint64_t>(world_size)) {
        cpu_relax();
    }
    for (long long i = 0; i < count; ++i) {
        float sum = 0.0f;
        for (long long r = 0; r < world_size; ++r) {
            sum += base[static_cast<size_t>(r) * static_cast<size_t>(count) + static_cast<size_t>(i)];
        }
        data[i] = sum;
    }
    __sync_synchronize();
    __sync_fetch_and_add(&counters[1], 1);
    while (counters[1] < gen * static_cast<uint64_t>(world_size)) {
        cpu_relax();
    }
    Py_END_ALLOW_THREADS

    munmap(mapped, bytes);
    Py_RETURN_NONE;
}

static PyObject* host_float_allreduce_unlink(PyObject*, PyObject* args) {
    const char* name_c = nullptr;
    if (!PyArg_ParseTuple(args, "s", &name_c)) {
        return nullptr;
    }
    if (name_c != nullptr) {
        shm_unlink(name_c);
    }
    Py_RETURN_NONE;
}

static PyObject* cpu_moe_server_loop_int8(PyObject*, PyObject* args) {
    const char* shm_name_c = nullptr;
    unsigned long long w1_layers_ull, w2_layers_ull, w3_layers_ull, s1_layers_ull, s2_layers_ull, s3_layers_ull;
    long long n_layers, hidden_dim, topk, inter_dim, num_experts, output_slots_ll;
    float swiglu_limit;
    if (!PyArg_ParseTuple(
            args,
            "sKKKKKKLLLLLLf",
            &shm_name_c,
            &w1_layers_ull,
            &w2_layers_ull,
            &w3_layers_ull,
            &s1_layers_ull,
            &s2_layers_ull,
            &s3_layers_ull,
            &n_layers,
            &hidden_dim,
            &topk,
            &inter_dim,
            &num_experts,
            &output_slots_ll,
            &swiglu_limit)) {
        return nullptr;
    }
    if (shm_name_c == nullptr) {
        return PyErr_Format(PyExc_ValueError, "shm_name must not be null");
    }

    auto* w1_layers = reinterpret_cast<int64_t*>(w1_layers_ull);
    auto* w2_layers = reinterpret_cast<int64_t*>(w2_layers_ull);
    auto* w3_layers = reinterpret_cast<int64_t*>(w3_layers_ull);
    auto* s1_layers = reinterpret_cast<int64_t*>(s1_layers_ull);
    auto* s2_layers = reinterpret_cast<int64_t*>(s2_layers_ull);
    auto* s3_layers = reinterpret_cast<int64_t*>(s3_layers_ull);

    const size_t header_bytes = sizeof(CPUMoEServerHeader);
    const size_t input_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
    const size_t ids_bytes = static_cast<size_t>(topk) * sizeof(int64_t);
    const size_t weights_bytes = static_cast<size_t>(topk) * sizeof(float);
    const size_t output_slots = static_cast<size_t>(std::max<long long>(1, output_slots_ll));
    const size_t output_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
    const size_t ack_bytes = output_slots * 8 * sizeof(long long);
    const size_t total_bytes = header_bytes + input_bytes + ids_bytes + weights_bytes + output_bytes * output_slots + ack_bytes;

    int fd = shm_open(shm_name_c, O_RDWR, 0600);
    if (fd < 0) {
        return PyErr_Format(PyExc_RuntimeError, "shm_open(%s) failed: %s", shm_name_c, std::strerror(errno));
    }
    void* mapped = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        return PyErr_Format(PyExc_RuntimeError, "mmap(%s) failed: %s", shm_name_c, std::strerror(errno));
    }

    auto* header = reinterpret_cast<CPUMoEServerHeader*>(mapped);
    auto* input = reinterpret_cast<float*>(reinterpret_cast<char*>(mapped) + header_bytes);
    auto* ids = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(input) + input_bytes);
    auto* weights = reinterpret_cast<float*>(reinterpret_cast<char*>(ids) + ids_bytes);
    auto* outputs = reinterpret_cast<float*>(reinterpret_cast<char*>(weights) + weights_bytes);

    const char* profile_env = std::getenv("DEEPSEEK_CPU_MOE_NATIVE_PROFILE");
    const bool profile = profile_env && (profile_env[0] == '1' || profile_env[0] == 't' || profile_env[0] == 'T');
    const char* profile_every_env = std::getenv("DEEPSEEK_CPU_MOE_NATIVE_PROFILE_EVERY");
    long long profile_every = profile_every_env ? std::atoll(profile_every_env) : 256;
    if (profile_every <= 0) profile_every = 256;

    Py_BEGIN_ALLOW_THREADS
    apply_omp_runtime_config();
    long long last_seq = 0;
    int idle_spins = 0;
    long long profile_count = 0;
    double profile_wait_total = 0.0;
    double profile_compute_total = 0.0;
    long long profile_first_layer = -1;
    auto profile_now = [&]() -> double {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
    };
    while (true) {
        const double wait_start = profile ? profile_now() : 0.0;
        long long req = atomic_load_ll(&header->request_seq);
        while (req <= last_seq && !atomic_load_ll(&header->stop)) {
            idle_wait_backoff(idle_spins++);
            req = atomic_load_ll(&header->request_seq);
        }
        idle_spins = 0;
        if (atomic_load_ll(&header->stop)) break;
        const long long layer_id = atomic_load_ll(&header->layer_id);
        const size_t output_slot = static_cast<size_t>(req % static_cast<long long>(output_slots));
        float* output = outputs + output_slot * hidden_dim;
        const double compute_start = profile ? profile_now() : 0.0;
        if (layer_id >= 0 && layer_id < n_layers) {
            std::fill(output, output + hidden_dim, 0.0f);
            const int64_t* w1_ptrs = reinterpret_cast<const int64_t*>(w1_layers[layer_id]);
            const int64_t* w2_ptrs = reinterpret_cast<const int64_t*>(w2_layers[layer_id]);
            const int64_t* w3_ptrs = reinterpret_cast<const int64_t*>(w3_layers[layer_id]);
            const int64_t* s1_ptrs = reinterpret_cast<const int64_t*>(s1_layers[layer_id]);
            const int64_t* s2_ptrs = reinterpret_cast<const int64_t*>(s2_layers[layer_id]);
            const int64_t* s3_ptrs = reinterpret_cast<const int64_t*>(s3_layers[layer_id]);
            int8_single_token_accumulate_routes_impl(
                input,
                ids,
                weights,
                output,
                hidden_dim,
                topk,
                inter_dim,
                0,
                num_experts,
                w1_ptrs,
                w2_ptrs,
                w3_ptrs,
                s1_ptrs,
                s2_ptrs,
                s3_ptrs,
                swiglu_limit,
                false);
        }
        atomic_store_ll(&header->response_seq, req);
        last_seq = req;
        if (profile) {
            const double compute_end = profile_now();
            profile_wait_total += compute_start - wait_start;
            profile_compute_total += compute_end - compute_start;
            if (profile_count == 0) profile_first_layer = layer_id;
            profile_count++;
            if (profile_count >= profile_every) {
                fprintf(stderr,
                        "cpu_moe_native_profile_v1 requests=%lld first_layer=%lld last_layer=%lld "
                        "avg_wait=%.6fs avg_compute=%.6fs\n",
                        profile_count, profile_first_layer, layer_id,
                        profile_wait_total / double(profile_count),
                        profile_compute_total / double(profile_count));
                fflush(stderr);
                profile_count = 0;
                profile_wait_total = 0.0;
                profile_compute_total = 0.0;
                profile_first_layer = -1;
            }
        }
    }
    Py_END_ALLOW_THREADS

    munmap(mapped, total_bytes);
    Py_RETURN_NONE;
}

static PyObject* cpu_moe_server_loop_int8_v2(PyObject*, PyObject* args) {
    const char* shm_name_c = nullptr;
    unsigned long long w1_layers_ull, w2_layers_ull, w3_layers_ull, s1_layers_ull, s2_layers_ull, s3_layers_ull;
    long long n_layers, hidden_dim, topk, inter_dim, num_experts, output_slots_ll;
    float swiglu_limit;
    if (!PyArg_ParseTuple(
            args,
            "sKKKKKKLLLLLLf",
            &shm_name_c,
            &w1_layers_ull,
            &w2_layers_ull,
            &w3_layers_ull,
            &s1_layers_ull,
            &s2_layers_ull,
            &s3_layers_ull,
            &n_layers,
            &hidden_dim,
            &topk,
            &inter_dim,
            &num_experts,
            &output_slots_ll,
            &swiglu_limit)) {
        return nullptr;
    }
    if (shm_name_c == nullptr) {
        return PyErr_Format(PyExc_ValueError, "shm_name must not be null");
    }

    auto* w1_layers = reinterpret_cast<int64_t*>(w1_layers_ull);
    auto* w2_layers = reinterpret_cast<int64_t*>(w2_layers_ull);
    auto* w3_layers = reinterpret_cast<int64_t*>(w3_layers_ull);
    auto* s1_layers = reinterpret_cast<int64_t*>(s1_layers_ull);
    auto* s2_layers = reinterpret_cast<int64_t*>(s2_layers_ull);
    auto* s3_layers = reinterpret_cast<int64_t*>(s3_layers_ull);

    const size_t header_bytes = sizeof(CPUMoEServerHeader);
    const size_t input_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
    const size_t ids_bytes = static_cast<size_t>(topk) * sizeof(int64_t);
    const size_t weights_bytes = static_cast<size_t>(topk) * sizeof(float);
    const size_t output_slots = static_cast<size_t>(std::max<long long>(1, output_slots_ll));
    const size_t output_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
    const size_t ack_bytes = output_slots * 8 * sizeof(long long);
    const size_t total_bytes = header_bytes + input_bytes + ids_bytes + weights_bytes + output_bytes * output_slots + ack_bytes;

    int fd = shm_open(shm_name_c, O_RDWR, 0600);
    if (fd < 0) {
        return PyErr_Format(PyExc_RuntimeError, "shm_open(%s) failed: %s", shm_name_c, std::strerror(errno));
    }
    void* mapped = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        return PyErr_Format(PyExc_RuntimeError, "mmap(%s) failed: %s", shm_name_c, std::strerror(errno));
    }

    auto* header = reinterpret_cast<CPUMoEServerHeader*>(mapped);
    auto* input = reinterpret_cast<float*>(reinterpret_cast<char*>(mapped) + header_bytes);
    auto* ids = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(input) + input_bytes);
    auto* weights = reinterpret_cast<float*>(reinterpret_cast<char*>(ids) + ids_bytes);
    auto* outputs = reinterpret_cast<float*>(reinterpret_cast<char*>(weights) + weights_bytes);

    const char* profile_env = std::getenv("DEEPSEEK_CPU_MOE_NATIVE_PROFILE");
    const bool profile = profile_env && (profile_env[0] == '1' || profile_env[0] == 't' || profile_env[0] == 'T');
    const char* profile_every_env = std::getenv("DEEPSEEK_CPU_MOE_NATIVE_PROFILE_EVERY");
    long long profile_every = profile_every_env ? std::atoll(profile_every_env) : 256;
    if (profile_every <= 0) profile_every = 256;

    Py_BEGIN_ALLOW_THREADS
    apply_omp_runtime_config();
    std::vector<int8_t> xq(static_cast<size_t>(hidden_dim));
    std::vector<float> hidden(static_cast<size_t>(std::min<long long>(topk, 16)) * static_cast<size_t>(inter_dim));
    std::vector<int8_t> hq(static_cast<size_t>(std::min<long long>(topk, 16)) * static_cast<size_t>(inter_dim));
    std::vector<float> h_scales(static_cast<size_t>(std::min<long long>(topk, 16)));
    long long last_seq = 0;
    int idle_spins = 0;
    long long profile_count = 0;
    double profile_wait_total = 0.0;
    double profile_compute_total = 0.0;
    long long profile_first_layer = -1;
    auto profile_now = [&]() -> double {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
    };
    while (true) {
        const double wait_start = profile ? profile_now() : 0.0;
        long long req = atomic_load_ll(&header->request_seq);
        while (req <= last_seq && !atomic_load_ll(&header->stop)) {
            idle_wait_backoff(idle_spins++);
            req = atomic_load_ll(&header->request_seq);
        }
        idle_spins = 0;
        if (atomic_load_ll(&header->stop)) break;
        const long long layer_id = atomic_load_ll(&header->layer_id);
        const size_t output_slot = static_cast<size_t>(req % static_cast<long long>(output_slots));
        float* output = outputs + output_slot * hidden_dim;
        const double compute_start = profile ? profile_now() : 0.0;
        if (layer_id >= 0 && layer_id < n_layers) {
            std::fill(output, output + hidden_dim, 0.0f);
            const int64_t* w1_ptrs = reinterpret_cast<const int64_t*>(w1_layers[layer_id]);
            const int64_t* w2_ptrs = reinterpret_cast<const int64_t*>(w2_layers[layer_id]);
            const int64_t* w3_ptrs = reinterpret_cast<const int64_t*>(w3_layers[layer_id]);
            const int64_t* s1_ptrs = reinterpret_cast<const int64_t*>(s1_layers[layer_id]);
            const int64_t* s2_ptrs = reinterpret_cast<const int64_t*>(s2_layers[layer_id]);
            const int64_t* s3_ptrs = reinterpret_cast<const int64_t*>(s3_layers[layer_id]);
            int8_single_token_accumulate_routes_prealloc_impl(
                input,
                ids,
                weights,
                output,
                hidden_dim,
                topk,
                inter_dim,
                0,
                num_experts,
                w1_ptrs,
                w2_ptrs,
                w3_ptrs,
                s1_ptrs,
                s2_ptrs,
                s3_ptrs,
                swiglu_limit,
                xq.data(),
                hidden.data(),
                hq.data(),
                h_scales.data(),
                false);
        }
        atomic_store_ll(&header->response_seq, req);
        last_seq = req;
        if (profile) {
            const double compute_end = profile_now();
            profile_wait_total += compute_start - wait_start;
            profile_compute_total += compute_end - compute_start;
            if (profile_count == 0) profile_first_layer = layer_id;
            profile_count++;
            if (profile_count >= profile_every) {
                fprintf(stderr,
                        "cpu_moe_native_profile_v2 requests=%lld first_layer=%lld last_layer=%lld "
                        "avg_wait=%.6fs avg_compute=%.6fs\n",
                        profile_count, profile_first_layer, layer_id,
                        profile_wait_total / double(profile_count),
                        profile_compute_total / double(profile_count));
                fflush(stderr);
                profile_count = 0;
                profile_wait_total = 0.0;
                profile_compute_total = 0.0;
                profile_first_layer = -1;
            }
        }
    }
    Py_END_ALLOW_THREADS

    munmap(mapped, total_bytes);
    Py_RETURN_NONE;
}


static PyMethodDef Methods[] = {
    {"routed_fp4_moe_forward", routed_fp4_moe_forward, METH_VARARGS, nullptr},
    {"routed_fp4_moe_forward_raw", routed_fp4_moe_forward_raw, METH_VARARGS, nullptr},
    {"routed_int8_moe_forward", routed_int8_moe_forward, METH_VARARGS, nullptr},
    {"routed_int8_moe_forward_topk_parallel", routed_int8_moe_forward_topk_parallel, METH_VARARGS, nullptr},
    {"routed_int8_moe_forward_topk_persistent", routed_int8_moe_forward_topk_persistent, METH_VARARGS, nullptr},
    {"routed_int8_moe_forward_persistent", routed_int8_moe_forward_persistent, METH_VARARGS, nullptr},
    {"routed_int8_moe_forward_prealloc_persistent", routed_int8_moe_forward_prealloc_persistent, METH_VARARGS, nullptr},
    {"cpu_moe_server_loop_int8", cpu_moe_server_loop_int8, METH_VARARGS, nullptr},
    {"cpu_moe_server_loop_int8_v2", cpu_moe_server_loop_int8_v2, METH_VARARGS, nullptr},
    {"int8_expert_forward", int8_expert_forward, METH_VARARGS, nullptr},
    {"host_float_allreduce", host_float_allreduce, METH_VARARGS, nullptr},
    {"host_float_allreduce_unlink", host_float_allreduce_unlink, METH_VARARGS, nullptr},
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
