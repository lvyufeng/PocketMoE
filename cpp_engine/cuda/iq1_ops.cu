// IQ1_M single-token MoE decode kernels for cpp_engine.
// Correctness-first path: consumes raw IQ1_M blocks directly on device; does not
// route IQ1_M through the existing IQ2_XXS/Q2_K DP4A kernels.

#include "cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace dsv4 {
namespace {

#include "iq1_grid.inc"

constexpr int kIQ1TileN = 8;
static int8_t* g_iq1_grid_device = nullptr;
static std::once_flag g_iq1_grid_init_flag;

const int8_t* iq1_grid_device() {
    std::call_once(g_iq1_grid_init_flag, [] {
        constexpr size_t bytes = 2048ULL * 8;
        if (cudaMalloc(&g_iq1_grid_device, bytes) != cudaSuccess) {
            g_iq1_grid_device = nullptr;
            return;
        }
        if (cudaMemcpy(g_iq1_grid_device, kIQ1MGrid, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(g_iq1_grid_device);
            g_iq1_grid_device = nullptr;
        }
    });
    return g_iq1_grid_device;
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ float iq1m_super_scale(const uint16_t* sc) {
    const uint16_t d_bits = static_cast<uint16_t>(
        ((sc[0] & 0xF000u) >> 12) | ((sc[1] & 0xF000u) >> 8) |
        ((sc[2] & 0xF000u) >> 4) | (sc[3] & 0xF000u));
    return __half2float(__ushort_as_half(d_bits));
}

__device__ __forceinline__ float iq1m_block_dot_256(
    const float* __restrict__ x_shared,
    const uint8_t* __restrict__ block,
    const int8_t* __restrict__ iq1_grid,
    int lane) {
    const uint8_t* qs = block;
    const uint8_t* qh = block + 32;
    const uint16_t* sc = reinterpret_cast<const uint16_t*>(block + 48);
    const float d = iq1m_super_scale(sc);
    float local = 0.0f;
    const int j = lane;
    const int sub = j >> 2;
    const int half = (j >> 1) & 1;
    const int pair = j & 1;
    const int s = sub * 2 + half;
    const int local_scale = (sc[s >> 2] >> ((s & 3) * 3)) & 0x07;
    const float dl = d * static_cast<float>(2 * local_scale + 1);
    const int qhv = (qh[j >> 1] >> ((j & 1) * 4)) & 0x0F;
    const int qidx = static_cast<int>(qs[j]) | ((qhv & 0x07) << 8);
    const float delta = (qhv & 0x08) ? -0.125f : 0.125f;
    const int8_t* gvals = iq1_grid + qidx * 8;
    const int k_out = sub * 32 + half * 16 + pair * 8;
    float partial = 0.0f;
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        partial += x_shared[k_out + g] * (static_cast<float>(gvals[g]) + delta);
    }
    local = dl * partial;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_down_sync(0xffffffff, local, offset);
    }
    return local;
}

__global__ void iq1_moe_single_w13_kernel(
    const float* __restrict__ x,
    const int64_t* __restrict__ route_slots,
    const uint8_t* __restrict__ w1_blocks,
    const uint8_t* __restrict__ w3_blocks,
    const int8_t* __restrict__ iq1_grid,
    float* __restrict__ gate,
    float* __restrict__ up,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    int blocks_per_row) {
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kIQ1TileN + warp;
    if (route >= routes || warp >= kIQ1TileN || out_col >= inter_dim) return;
    const int expert = static_cast<int>(route_slots[route]);
    if (expert < 0 || expert >= n_experts) return;

    __shared__ float x_shared[256];
    const uint8_t* w1_row = w1_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 56;
    const uint8_t* w3_row = w3_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 56;
    float acc1 = 0.0f;
    float acc3 = 0.0f;
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int k = threadIdx.x; k < 256; k += blockDim.x) {
            const int idx = k_base + k;
            x_shared[k] = idx < dim ? x[idx] : 0.0f;
        }
        __syncthreads();
        const float b1 = iq1m_block_dot_256(x_shared, w1_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
        const float b3 = iq1m_block_dot_256(x_shared, w3_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
        if (lane == 0) {
            acc1 += b1;
            acc3 += b3;
        }
        __syncthreads();
    }
    if (lane == 0) {
        gate[static_cast<int64_t>(route) * inter_dim + out_col] = acc1;
        up[static_cast<int64_t>(route) * inter_dim + out_col] = acc3;
    }
}

__global__ void iq1_moe_single_w13_swiglu_kernel(
    const float* __restrict__ x,
    const int64_t* __restrict__ route_slots,
    const float* __restrict__ route_weights,
    const uint8_t* __restrict__ w1_blocks,
    const uint8_t* __restrict__ w3_blocks,
    const int8_t* __restrict__ iq1_grid,
    float* __restrict__ hidden,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    int blocks_per_row,
    float swiglu_limit) {
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kIQ1TileN + warp;
    if (route >= routes || warp >= kIQ1TileN || out_col >= inter_dim) return;
    const int expert = static_cast<int>(route_slots[route]);
    if (expert < 0 || expert >= n_experts) return;

    __shared__ float x_shared[256];
    const uint8_t* w1_row = w1_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 56;
    const uint8_t* w3_row = w3_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 56;
    float acc1 = 0.0f;
    float acc3 = 0.0f;
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int k = threadIdx.x; k < 256; k += blockDim.x) {
            const int idx = k_base + k;
            x_shared[k] = idx < dim ? x[idx] : 0.0f;
        }
        __syncthreads();
        const float b1 = iq1m_block_dot_256(x_shared, w1_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
        const float b3 = iq1m_block_dot_256(x_shared, w3_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
        if (lane == 0) {
            acc1 += b1;
            acc3 += b3;
        }
        __syncthreads();
    }
    if (lane == 0) {
        float g = acc1;
        float u = acc3;
        if (swiglu_limit > 0.0f) {
            u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
            g = fminf(g, swiglu_limit);
        }
        hidden[static_cast<int64_t>(route) * inter_dim + out_col] =
            (g / (1.0f + expf(-g))) * u * route_weights[route];
    }
}

__global__ void iq1_route_swiglu_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ route_weights,
    float* __restrict__ hidden,
    int routes,
    int inter_dim,
    float swiglu_limit) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = routes * inter_dim;
    if (idx >= total) return;
    const int route = idx / inter_dim;
    float g = gate[idx];
    float u = up[idx];
    if (swiglu_limit > 0.0f) {
        u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
        g = fminf(g, swiglu_limit);
    }
    hidden[idx] = (g / (1.0f + expf(-g))) * u * route_weights[route];
}

__global__ void iq1_moe_single_w2_kernel(
    const float* __restrict__ hidden,
    const int64_t* __restrict__ route_slots,
    const uint8_t* __restrict__ w2_blocks,
    const int8_t* __restrict__ iq1_grid,
    float* __restrict__ y,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    int blocks_per_row) {
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kIQ1TileN + warp;
    if (route >= routes || warp >= kIQ1TileN || out_col >= dim) return;
    const int expert = static_cast<int>(route_slots[route]);
    if (expert < 0 || expert >= n_experts) return;

    __shared__ float h_shared[256];
    const float* hidden_row = hidden + static_cast<int64_t>(route) * inter_dim;
    const uint8_t* w2_row = w2_blocks + (static_cast<int64_t>(expert) * dim + out_col) * blocks_per_row * 56;
    float acc = 0.0f;
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int k = threadIdx.x; k < 256; k += blockDim.x) {
            const int idx = k_base + k;
            h_shared[k] = idx < inter_dim ? hidden_row[idx] : 0.0f;
        }
        __syncthreads();
        const float b = iq1m_block_dot_256(h_shared, w2_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
        if (lane == 0) acc += b;
        __syncthreads();
    }
    if (lane == 0) atomicAdd(y + out_col, acc);
}

__global__ void iq1_moe_single_w2_reduce_kernel(
    const float* __restrict__ hidden,
    const int64_t* __restrict__ route_slots,
    const uint8_t* __restrict__ w2_blocks,
    const int8_t* __restrict__ iq1_grid,
    float* __restrict__ y,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    int blocks_per_row) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kIQ1TileN + warp;
    if (warp >= kIQ1TileN || out_col >= dim) return;

    __shared__ float h_shared[256];
    float total = 0.0f;
    for (int route = 0; route < routes; ++route) {
        const int expert = static_cast<int>(route_slots[route]);
        if (expert < 0 || expert >= n_experts) continue;
        const float* hidden_row = hidden + static_cast<int64_t>(route) * inter_dim;
        const uint8_t* w2_row = w2_blocks + (static_cast<int64_t>(expert) * dim + out_col) * blocks_per_row * 56;
        float acc = 0.0f;
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            const int k_base = block_idx * 256;
            for (int k = threadIdx.x; k < 256; k += blockDim.x) {
                const int idx = k_base + k;
                h_shared[k] = idx < inter_dim ? hidden_row[idx] : 0.0f;
            }
            __syncthreads();
            const float b = iq1m_block_dot_256(h_shared, w2_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
            if (lane == 0) acc += b;
            __syncthreads();
        }
        if (lane == 0) total += acc;
    }
    if (lane == 0) y[out_col] = total;
}

}  // namespace

bool iq1_moe_single_w13_cuda(
    const float* d_x,
    const int64_t* d_route_slots,
    const uint8_t* d_w1_blocks,
    const uint8_t* d_w3_blocks,
    float* d_gate,
    float* d_up,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    void* stream) {
    if (routes <= 0 || n_experts <= 0 || dim <= 0 || inter_dim <= 0) return true;
    const int8_t* grid = iq1_grid_device();
    if (grid == nullptr) return false;
    const int blocks_per_row = (dim + 255) / 256;
    dim3 grid_dim(ceil_div(inter_dim, kIQ1TileN), routes);
    dim3 block(256);
    iq1_moe_single_w13_kernel<<<grid_dim, block, 0, static_cast<cudaStream_t>(stream)>>>(
        d_x, d_route_slots, d_w1_blocks, d_w3_blocks, grid, d_gate, d_up,
        routes, n_experts, dim, inter_dim, blocks_per_row);
    return cudaGetLastError() == cudaSuccess;
}

bool iq1_moe_single_w13_swiglu_cuda(
    const float* d_x,
    const int64_t* d_route_slots,
    const float* d_route_weights,
    const uint8_t* d_w1_blocks,
    const uint8_t* d_w3_blocks,
    float* d_hidden,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    float swiglu_limit,
    void* stream) {
    if (routes <= 0 || n_experts <= 0 || dim <= 0 || inter_dim <= 0) return true;
    const int8_t* grid = iq1_grid_device();
    if (grid == nullptr) return false;
    const int blocks_per_row = (dim + 255) / 256;
    dim3 grid_dim(ceil_div(inter_dim, kIQ1TileN), routes);
    dim3 block(256);
    iq1_moe_single_w13_swiglu_kernel<<<grid_dim, block, 0, static_cast<cudaStream_t>(stream)>>>(
        d_x, d_route_slots, d_route_weights, d_w1_blocks, d_w3_blocks, grid, d_hidden,
        routes, n_experts, dim, inter_dim, blocks_per_row, swiglu_limit);
    return cudaGetLastError() == cudaSuccess;
}

bool iq1_route_swiglu_cuda(
    const float* d_gate,
    const float* d_up,
    const float* d_route_weights,
    float* d_hidden,
    int routes,
    int inter_dim,
    float swiglu_limit,
    void* stream) {
    const int total = routes * inter_dim;
    if (total <= 0) return true;
    const int threads = 256;
    const int blocks = ceil_div(total, threads);
    iq1_route_swiglu_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        d_gate, d_up, d_route_weights, d_hidden, routes, inter_dim, swiglu_limit);
    return cudaGetLastError() == cudaSuccess;
}

bool iq1_moe_single_w2_cuda(
    const float* d_hidden,
    const int64_t* d_route_slots,
    const uint8_t* d_w2_blocks,
    float* d_y,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    void* stream) {
    if (routes <= 0 || n_experts <= 0 || dim <= 0 || inter_dim <= 0) return true;
    const int8_t* grid = iq1_grid_device();
    if (grid == nullptr) return false;
    const int blocks_per_row = (inter_dim + 255) / 256;
    dim3 grid_dim(ceil_div(dim, kIQ1TileN), routes);
    dim3 block(256);
    iq1_moe_single_w2_kernel<<<grid_dim, block, 0, static_cast<cudaStream_t>(stream)>>>(
        d_hidden, d_route_slots, d_w2_blocks, grid, d_y,
        routes, n_experts, dim, inter_dim, blocks_per_row);
    return cudaGetLastError() == cudaSuccess;
}

bool iq1_moe_single_w2_reduce_cuda(
    const float* d_hidden,
    const int64_t* d_route_slots,
    const uint8_t* d_w2_blocks,
    float* d_y,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    void* stream) {
    if (routes <= 0 || n_experts <= 0 || dim <= 0 || inter_dim <= 0) return true;
    const int8_t* grid = iq1_grid_device();
    if (grid == nullptr) return false;
    const int blocks_per_row = (inter_dim + 255) / 256;
    dim3 grid_dim(ceil_div(dim, kIQ1TileN));
    dim3 block(256);
    iq1_moe_single_w2_reduce_kernel<<<grid_dim, block, 0, static_cast<cudaStream_t>(stream)>>>(
        d_hidden, d_route_slots, d_w2_blocks, grid, d_y,
        routes, n_experts, dim, inter_dim, blocks_per_row);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace dsv4
