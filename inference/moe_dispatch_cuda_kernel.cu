#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 128;

__global__ void count_items_kernel(
    const int32_t* __restrict__ owner,
    int n,
    int local_rank,
    int world_size,
    int32_t* __restrict__ local_count,
    int32_t* __restrict__ remote_counts) {
    extern __shared__ int32_t shared[];
    int32_t* shared_remote = shared;
    int32_t* shared_local = shared + world_size;
    const int tid = threadIdx.x;

    if (tid < world_size) {
        shared_remote[tid] = 0;
    }
    if (tid == 0) {
        shared_local[0] = 0;
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        const int o = owner[i];
        if (o == local_rank) {
            atomicAdd(shared_local, 1);
        } else if (o >= 0 && o < world_size) {
            atomicAdd(&shared_remote[o], 1);
        }
    }
    __syncthreads();

    if (tid < world_size) {
        remote_counts[tid] = shared_remote[tid];
    }
    if (tid == 0) {
        local_count[0] = shared_local[0];
    }
}

template <typename scalar_t>
__global__ void pack_items_kernel(
    const scalar_t* __restrict__ flat_x,
    const int32_t* __restrict__ flat_expert,
    const float* __restrict__ flat_weight,
    const int32_t* __restrict__ flat_owner,
    int n,
    int dim,
    int local_rank,
    const int32_t* __restrict__ remote_base_offsets,
    int32_t* __restrict__ remote_write_offsets,
    int32_t* __restrict__ local_counter,
    int32_t* __restrict__ local_positions,
    scalar_t* __restrict__ local_x,
    int32_t* __restrict__ local_expert,
    float* __restrict__ local_weight,
    int32_t* __restrict__ remote_positions,
    scalar_t* __restrict__ send_payload) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const int owner = flat_owner[i];
    if (owner == local_rank) {
        const int local_idx = atomicAdd(local_counter, 1);
        local_positions[local_idx] = i;
        local_expert[local_idx] = flat_expert[i];
        local_weight[local_idx] = flat_weight[i];
        const scalar_t* src = flat_x + static_cast<long long>(i) * dim;
        scalar_t* dst = local_x + static_cast<long long>(local_idx) * dim;
        for (int j = 0; j < dim; ++j) {
            dst[j] = src[j];
        }
    } else {
        const int offset = atomicAdd(&remote_write_offsets[owner], 1);
        const int remote_idx = remote_base_offsets[owner] + offset;
        remote_positions[remote_idx] = i;
        const scalar_t* src = flat_x + static_cast<long long>(i) * dim;
        scalar_t* dst = send_payload + static_cast<long long>(remote_idx) * (dim + 2);
        for (int j = 0; j < dim; ++j) {
            dst[j] = src[j];
        }
        dst[dim] = static_cast<scalar_t>(flat_expert[i]);
        dst[dim + 1] = static_cast<scalar_t>(flat_weight[i]);
    }
}

template <typename scalar_t>
__global__ void pack_items_padded_kernel(
    const scalar_t* __restrict__ flat_x,
    const int32_t* __restrict__ flat_expert,
    const float* __restrict__ flat_weight,
    const int32_t* __restrict__ flat_owner,
    int n,
    int dim,
    int local_rank,
    int32_t* __restrict__ local_counter,
    int32_t* __restrict__ local_positions,
    scalar_t* __restrict__ local_x,
    int32_t* __restrict__ local_expert,
    float* __restrict__ local_weight,
    int32_t* __restrict__ remote_write_offsets,
    int32_t* __restrict__ remote_positions,
    scalar_t* __restrict__ send_payload) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const int owner = flat_owner[i];
    if (owner == local_rank) {
        const int local_idx = atomicAdd(local_counter, 1);
        local_positions[local_idx] = i;
        local_expert[local_idx] = flat_expert[i];
        local_weight[local_idx] = flat_weight[i];
        const scalar_t* src = flat_x + static_cast<long long>(i) * dim;
        scalar_t* dst = local_x + static_cast<long long>(local_idx) * dim;
        for (int j = 0; j < dim; ++j) {
            dst[j] = src[j];
        }
    } else if (owner >= 0) {
        const int offset = atomicAdd(&remote_write_offsets[owner], 1);
        const int remote_idx = owner * n + offset;
        remote_positions[remote_idx] = i;
        const scalar_t* src = flat_x + static_cast<long long>(i) * dim;
        scalar_t* dst = send_payload + static_cast<long long>(remote_idx) * (dim + 2);
        for (int j = 0; j < dim; ++j) {
            dst[j] = src[j];
        }
        dst[dim] = static_cast<scalar_t>(flat_expert[i]);
        dst[dim + 1] = static_cast<scalar_t>(flat_weight[i]);
    }
}

template <typename scalar_t>
__global__ void count_valid_payload_kernel(
    const scalar_t* __restrict__ payload,
    int rows,
    int width,
    int32_t* __restrict__ count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) {
        return;
    }
    const scalar_t expert = payload[static_cast<long long>(i) * width + (width - 2)];
    if (static_cast<float>(expert) >= 0.0f) {
        atomicAdd(count, 1);
    }
}

template <typename scalar_t>
__global__ void compact_payload_kernel(
    const scalar_t* __restrict__ payload,
    int rows,
    int width,
    int32_t* __restrict__ write_index,
    scalar_t* __restrict__ compacted,
    int32_t* __restrict__ positions) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) {
        return;
    }
    const scalar_t expert = payload[static_cast<long long>(i) * width + (width - 2)];
    if (static_cast<float>(expert) < 0.0f) {
        return;
    }
    const int dst_row = atomicAdd(write_index, 1);
    positions[dst_row] = i;
    const scalar_t* src = payload + static_cast<long long>(i) * width;
    scalar_t* dst = compacted + static_cast<long long>(dst_row) * width;
    for (int j = 0; j < width; ++j) {
        dst[j] = src[j];
    }
}

template <typename scalar_t>
__global__ void scatter_rows_kernel(
    float* __restrict__ out,
    const int32_t* __restrict__ positions,
    const scalar_t* __restrict__ values,
    int rows,
    int dim) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int dst_row = positions[row];
    if (dst_row < 0) {
        return;
    }
    const int tid = threadIdx.x;
    const scalar_t* src = values + static_cast<long long>(row) * dim;
    float* dst = out + static_cast<long long>(dst_row) * dim;
    for (int j = tid; j < dim; j += blockDim.x) {
        dst[j] = static_cast<float>(src[j]);
    }
}

}  // namespace

std::vector<torch::Tensor> partition_items_cuda(
    const torch::Tensor& flat_x,
    const torch::Tensor& flat_expert,
    const torch::Tensor& flat_weight,
    const torch::Tensor& flat_owner,
    int64_t local_rank,
    int64_t world_size) {
    c10::cuda::CUDAGuard device_guard(flat_x.device());

    const int n = static_cast<int>(flat_x.size(0));
    const int dim = static_cast<int>(flat_x.size(1));
    auto i32_opts = flat_x.options().dtype(torch::kInt32);
    auto f32_opts = flat_x.options().dtype(torch::kFloat32);

    auto local_count = torch::zeros({1}, i32_opts);
    auto send_counts = torch::zeros({world_size}, i32_opts);

    if (n > 0) {
        const size_t shared_bytes = static_cast<size_t>(world_size + 1) * sizeof(int32_t);
        count_items_kernel<<<1, kThreads, shared_bytes, at::cuda::getDefaultCUDAStream()>>>(
            flat_owner.data_ptr<int32_t>(),
            n,
            static_cast<int>(local_rank),
            static_cast<int>(world_size),
            local_count.data_ptr<int32_t>(),
            send_counts.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto local_count_cpu = local_count.to(torch::kCPU);
    auto send_counts_cpu = send_counts.to(torch::kCPU);
    const int local_total = local_count_cpu.data_ptr<int32_t>()[0];

    std::vector<int32_t> remote_bases(static_cast<size_t>(world_size), 0);
    int remote_total = 0;
    auto* send_counts_ptr = send_counts_cpu.data_ptr<int32_t>();
    for (int i = 0; i < world_size; ++i) {
        remote_bases[static_cast<size_t>(i)] = remote_total;
        remote_total += send_counts_ptr[i];
    }

    auto local_positions = torch::empty({local_total}, i32_opts);
    auto local_x = torch::empty({local_total, dim}, flat_x.options());
    auto local_expert = torch::empty({local_total}, i32_opts);
    auto local_weight = torch::empty({local_total}, f32_opts);
    auto remote_positions = torch::empty({remote_total}, i32_opts);
    auto send_payload = torch::empty({remote_total, dim + 2}, flat_x.options());

    if (n > 0) {
        auto remote_base_offsets = torch::tensor(remote_bases, i32_opts);
        auto remote_write_offsets = torch::zeros({world_size}, i32_opts);
        auto local_counter = torch::zeros({1}, i32_opts);
        const int blocks = (n + kThreads - 1) / kThreads;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, flat_x.scalar_type(), "moe_dispatch_pack_items", [&] {
            pack_items_kernel<scalar_t><<<blocks, kThreads, 0, at::cuda::getDefaultCUDAStream()>>>(
                flat_x.data_ptr<scalar_t>(),
                flat_expert.data_ptr<int32_t>(),
                flat_weight.data_ptr<float>(),
                flat_owner.data_ptr<int32_t>(),
                n,
                dim,
                static_cast<int>(local_rank),
                remote_base_offsets.data_ptr<int32_t>(),
                remote_write_offsets.data_ptr<int32_t>(),
                local_counter.data_ptr<int32_t>(),
                local_positions.data_ptr<int32_t>(),
                local_x.data_ptr<scalar_t>(),
                local_expert.data_ptr<int32_t>(),
                local_weight.data_ptr<float>(),
                remote_positions.data_ptr<int32_t>(),
                send_payload.data_ptr<scalar_t>());
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {
        local_positions,
        local_x,
        local_expert,
        local_weight,
        send_counts,
        remote_positions,
        send_payload,
    };
}

std::vector<torch::Tensor> partition_items_padded_cuda(
    const torch::Tensor& flat_x,
    const torch::Tensor& flat_expert,
    const torch::Tensor& flat_weight,
    const torch::Tensor& flat_owner,
    int64_t local_rank,
    int64_t world_size) {
    c10::cuda::CUDAGuard device_guard(flat_x.device());

    const int n = static_cast<int>(flat_x.size(0));
    const int dim = static_cast<int>(flat_x.size(1));
    auto i32_opts = flat_x.options().dtype(torch::kInt32);
    auto f32_opts = flat_x.options().dtype(torch::kFloat32);

    auto local_positions = torch::full({n}, -1, i32_opts);
    auto local_x = torch::zeros({n, dim}, flat_x.options());
    auto local_expert = torch::full({n}, -1, i32_opts);
    auto local_weight = torch::zeros({n}, f32_opts);
    auto remote_positions = torch::full({world_size * n}, -1, i32_opts);
    auto send_payload = torch::zeros({world_size * n, dim + 2}, flat_x.options());
    if (n > 0) {
        send_payload.select(1, dim).fill_(-1);
    }

    if (n > 0) {
        auto local_counter = torch::zeros({1}, i32_opts);
        auto remote_write_offsets = torch::zeros({world_size}, i32_opts);
        const int blocks = (n + kThreads - 1) / kThreads;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, flat_x.scalar_type(), "moe_dispatch_pack_items_padded", [&] {
            pack_items_padded_kernel<scalar_t><<<blocks, kThreads, 0, at::cuda::getDefaultCUDAStream()>>>(
                flat_x.data_ptr<scalar_t>(),
                flat_expert.data_ptr<int32_t>(),
                flat_weight.data_ptr<float>(),
                flat_owner.data_ptr<int32_t>(),
                n,
                dim,
                static_cast<int>(local_rank),
                local_counter.data_ptr<int32_t>(),
                local_positions.data_ptr<int32_t>(),
                local_x.data_ptr<scalar_t>(),
                local_expert.data_ptr<int32_t>(),
                local_weight.data_ptr<float>(),
                remote_write_offsets.data_ptr<int32_t>(),
                remote_positions.data_ptr<int32_t>(),
                send_payload.data_ptr<scalar_t>());
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {
        local_positions,
        local_x,
        local_expert,
        local_weight,
        remote_positions,
        send_payload,
    };
}

std::vector<torch::Tensor> compact_payload_cuda(const torch::Tensor& payload) {
    c10::cuda::CUDAGuard device_guard(payload.device());

    const int rows = static_cast<int>(payload.size(0));
    const int width = static_cast<int>(payload.size(1));
    auto i32_opts = payload.options().dtype(torch::kInt32);

    auto valid_count = torch::zeros({1}, i32_opts);
    if (rows > 0) {
        const int blocks = (rows + kThreads - 1) / kThreads;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, payload.scalar_type(), "moe_dispatch_count_valid_payload", [&] {
            count_valid_payload_kernel<scalar_t><<<blocks, kThreads, 0, at::cuda::getDefaultCUDAStream()>>>(
                payload.data_ptr<scalar_t>(),
                rows,
                width,
                valid_count.data_ptr<int32_t>());
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    const int valid_total = valid_count.to(torch::kCPU).data_ptr<int32_t>()[0];
    auto compacted = torch::empty({valid_total, width}, payload.options());
    auto positions = torch::empty({valid_total}, i32_opts);
    if (valid_total > 0) {
        auto write_index = torch::zeros({1}, i32_opts);
        const int blocks = (rows + kThreads - 1) / kThreads;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, payload.scalar_type(), "moe_dispatch_compact_payload", [&] {
            compact_payload_kernel<scalar_t><<<blocks, kThreads, 0, at::cuda::getDefaultCUDAStream()>>>(
                payload.data_ptr<scalar_t>(),
                rows,
                width,
                write_index.data_ptr<int32_t>(),
                compacted.data_ptr<scalar_t>(),
                positions.data_ptr<int32_t>());
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {compacted, positions};
}

void scatter_rows_cuda(
    const torch::Tensor& out,
    const torch::Tensor& positions,
    const torch::Tensor& values) {
    c10::cuda::CUDAGuard device_guard(out.device());

    const int rows = static_cast<int>(positions.size(0));
    const int dim = static_cast<int>(out.size(1));
    if (rows == 0) {
        return;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, values.scalar_type(), "moe_dispatch_scatter_rows", [&] {
        scatter_rows_kernel<scalar_t><<<rows, kThreads, 0, at::cuda::getDefaultCUDAStream()>>>(
            out.data_ptr<float>(),
            positions.data_ptr<int32_t>(),
            values.data_ptr<scalar_t>(),
            rows,
            dim);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
