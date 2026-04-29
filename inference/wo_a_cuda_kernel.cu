#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kQuantThreads = 256;
constexpr int kGemmThreads = 128;

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

template <typename scalar_t>
__global__ void quantize_rows_kernel(
    const scalar_t* __restrict__ x,
    int8_t* __restrict__ x_q,
    float* __restrict__ x_scale,
    int rows,
    int groups,
    int k) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    const int tid = threadIdx.x;
    const int base = (row * groups + group) * k;

    __shared__ float sdata[kQuantThreads];
    float local_max = 0.0f;
    for (int idx = tid; idx < k; idx += blockDim.x) {
        const float v = static_cast<float>(x[base + idx]);
        local_max = fmaxf(local_max, fabsf(v));
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    const float scale = fmaxf(sdata[0], 1.0e-6f) / 127.0f;
    if (tid == 0) {
        x_scale[row * groups + group] = scale;
    }
    __syncthreads();

    const float inv_scale = 1.0f / scale;
    for (int idx = tid; idx < k; idx += blockDim.x) {
        const float v = static_cast<float>(x[base + idx]);
        int q = __float2int_rn(v * inv_scale);
        q = max(-127, min(127, q));
        x_q[base + idx] = static_cast<int8_t>(q);
    }
}

__global__ void int8_gemm_rows_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ weight_q,
    const float* __restrict__ weight_s,
    float* __restrict__ out,
    int rows,
    int groups,
    int n,
    int k) {
    const int row = blockIdx.y;
    const int group = blockIdx.z;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int packs = k / 4;

    extern __shared__ int x_shared[];
    const int8_t* x_row = x_q + (row * groups + group) * k;
    const int* x_row_i32 = reinterpret_cast<const int*>(x_row);
    for (int idx = threadIdx.x; idx < packs; idx += blockDim.x) {
        x_shared[idx] = x_row_i32[idx];
    }
    __syncthreads();

    if (col >= n) {
        return;
    }

    const int8_t* w_row = weight_q + (group * n + col) * k;
    const int* w_row_i32 = reinterpret_cast<const int*>(w_row);
    int acc = 0;
    for (int idx = 0; idx < packs; ++idx) {
        acc = __dp4a(x_shared[idx], w_row_i32[idx], acc);
    }

    const float scale = x_scale[row * groups + group] * weight_s[group * n + col];
    out[(row * groups + group) * n + col] = static_cast<float>(acc) * scale;
}

}  // namespace

torch::Tensor wo_a_int8_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_s) {
    c10::cuda::CUDAGuard device_guard(x.device());

    const auto bsz = static_cast<int>(x.size(0));
    const auto seqlen = static_cast<int>(x.size(1));
    const auto groups = static_cast<int>(x.size(2));
    const auto k = static_cast<int>(x.size(3));
    const auto n = static_cast<int>(weight_q.size(1));
    const auto rows = bsz * seqlen;

    auto x_contig = x.contiguous();
    auto wq_contig = weight_q.contiguous();
    auto ws_contig = weight_s.contiguous();

    auto x_q = torch::empty({rows, groups, k}, x.options().dtype(torch::kInt8));
    auto x_scale = torch::empty({rows, groups}, x.options().dtype(torch::kFloat32));
    auto out = torch::empty({rows, groups, n}, x.options().dtype(torch::kFloat32));

    const dim3 quant_grid(rows, groups);
    const dim3 quant_block(kQuantThreads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "wo_a_int8_quantize", [&] {
        quantize_rows_kernel<scalar_t><<<quant_grid, quant_block, 0, at::cuda::getDefaultCUDAStream()>>>(
            x_contig.data_ptr<scalar_t>(),
            x_q.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(),
            rows,
            groups,
            k);
    });

    const dim3 gemm_grid(ceil_div(n, kGemmThreads), rows, groups);
    const dim3 gemm_block(kGemmThreads);
    const size_t shared_bytes = static_cast<size_t>(k / 4) * sizeof(int);
    int8_gemm_rows_kernel<<<gemm_grid, gemm_block, shared_bytes, at::cuda::getDefaultCUDAStream()>>>(
        x_q.data_ptr<int8_t>(),
        x_scale.data_ptr<float>(),
        wq_contig.data_ptr<int8_t>(),
        ws_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        groups,
        n,
        k);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.view({bsz, seqlen, groups, n});
}
