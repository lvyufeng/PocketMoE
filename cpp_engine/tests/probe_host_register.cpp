// Measure cudaHostRegister overhead on the full GGUF mmap region and verify
// that subsequent H2D from that region is async + high-bandwidth (pinned).
// Goal: confirm the "register-once at load, async H2D forever after" pattern
// before wiring it into the decode path.

#include "gguf_reader.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cuda_runtime.h>

static void check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: probe_host_register <model.gguf>\n";
        return 2;
    }

    dsv4::GGUFFile gguf(argv[1]);
    const void* base = gguf.bytes();
    const size_t bytes = gguf.file_size();
    std::printf("gguf size = %.2f GB\n", bytes / (1024.0 * 1024.0 * 1024.0));

    auto t0 = std::chrono::steady_clock::now();
    cudaError_t err = cudaHostRegister(const_cast<void*>(base), bytes,
                                       cudaHostRegisterReadOnly);
    auto t1 = std::chrono::steady_clock::now();
    if (err != cudaSuccess) {
        std::printf("cudaHostRegister(ReadOnly) failed: %s; trying default flags\n",
                    cudaGetErrorString(err));
        t0 = std::chrono::steady_clock::now();
        err = cudaHostRegister(const_cast<void*>(base), bytes, cudaHostRegisterDefault);
        t1 = std::chrono::steady_clock::now();
        if (err != cudaSuccess) {
            std::printf("cudaHostRegister(Default) failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        std::printf("register(Default) ok, took %.3f s\n",
                    std::chrono::duration<double>(t1 - t0).count());
    } else {
        std::printf("register(ReadOnly) ok, took %.3f s\n",
                    std::chrono::duration<double>(t1 - t0).count());
    }

    // Bandwidth test: copy a 100 MB region (offset 1 GB in to skip header/metadata).
    constexpr size_t test_bytes = 100ull * 1024 * 1024;
    constexpr size_t test_offset = 1ull * 1024 * 1024 * 1024;
    if (test_offset + test_bytes > bytes) {
        std::printf("file too small for bandwidth test\n");
        cudaHostUnregister(const_cast<void*>(base));
        return 0;
    }
    void* d_buf = nullptr;
    check(cudaMalloc(&d_buf, test_bytes), "cudaMalloc");

    // Warmup + 3 trials
    const char* src = reinterpret_cast<const char*>(base) + test_offset;
    for (int trial = -1; trial < 3; ++trial) {
        check(cudaDeviceSynchronize(), "sync before");
        auto t_h = std::chrono::steady_clock::now();
        check(cudaMemcpy(d_buf, src, test_bytes, cudaMemcpyHostToDevice), "memcpy");
        check(cudaDeviceSynchronize(), "sync after");
        auto t_e = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t_e - t_h).count();
        double gbs = (test_bytes / s) / (1024.0 * 1024.0 * 1024.0);
        if (trial >= 0) {
            std::printf("trial %d: 100 MB H2D in %.3f ms = %.2f GB/s\n",
                        trial, s * 1000.0, gbs);
        }
    }

    check(cudaFree(d_buf), "free");
    auto u0 = std::chrono::steady_clock::now();
    check(cudaHostUnregister(const_cast<void*>(base)), "cudaHostUnregister");
    auto u1 = std::chrono::steady_clock::now();
    std::printf("unregister took %.3f s\n",
                std::chrono::duration<double>(u1 - u0).count());
    return 0;
}
