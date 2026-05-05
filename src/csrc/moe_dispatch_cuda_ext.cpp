#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> partition_items_cuda(
    const torch::Tensor& flat_x,
    const torch::Tensor& flat_expert,
    const torch::Tensor& flat_weight,
    const torch::Tensor& flat_owner,
    int64_t local_rank,
    int64_t world_size);

std::vector<torch::Tensor> partition_items_padded_cuda(
    const torch::Tensor& flat_x,
    const torch::Tensor& flat_expert,
    const torch::Tensor& flat_weight,
    const torch::Tensor& flat_owner,
    int64_t local_rank,
    int64_t world_size);

std::vector<torch::Tensor> compact_payload_cuda(
    const torch::Tensor& payload);

void scatter_rows_cuda(
    const torch::Tensor& out,
    const torch::Tensor& positions,
    const torch::Tensor& values);

namespace {

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> partition_items(
    const torch::Tensor& flat_x,
    const torch::Tensor& flat_expert,
    const torch::Tensor& flat_weight,
    const torch::Tensor& flat_owner,
    int64_t local_rank,
    int64_t world_size) {
    TORCH_CHECK(flat_x.dim() == 2, "flat_x must have shape [N, D]");
    TORCH_CHECK(flat_expert.dim() == 1, "flat_expert must have shape [N]");
    TORCH_CHECK(flat_weight.dim() == 1, "flat_weight must have shape [N]");
    TORCH_CHECK(flat_owner.dim() == 1, "flat_owner must have shape [N]");
    TORCH_CHECK(flat_x.size(0) == flat_expert.size(0), "flat_expert row count mismatch");
    TORCH_CHECK(flat_x.size(0) == flat_weight.size(0), "flat_weight row count mismatch");
    TORCH_CHECK(flat_x.size(0) == flat_owner.size(0), "flat_owner row count mismatch");
    TORCH_CHECK(local_rank >= 0 && local_rank < world_size, "local_rank out of range");
    TORCH_CHECK(world_size > 0, "world_size must be positive");

    check_cuda_contiguous(flat_x, "flat_x");
    check_cuda_contiguous(flat_expert, "flat_expert");
    check_cuda_contiguous(flat_weight, "flat_weight");
    check_cuda_contiguous(flat_owner, "flat_owner");

    TORCH_CHECK(
        flat_x.scalar_type() == torch::kFloat16 ||
        flat_x.scalar_type() == torch::kBFloat16 ||
        flat_x.scalar_type() == torch::kFloat32,
        "flat_x must be float16, bfloat16, or float32");
    TORCH_CHECK(flat_expert.scalar_type() == torch::kInt32, "flat_expert must be int32");
    TORCH_CHECK(flat_weight.scalar_type() == torch::kFloat32, "flat_weight must be float32");
    TORCH_CHECK(flat_owner.scalar_type() == torch::kInt32, "flat_owner must be int32");

    return partition_items_cuda(flat_x, flat_expert, flat_weight, flat_owner, local_rank, world_size);
}

std::vector<torch::Tensor> partition_items_padded(
    const torch::Tensor& flat_x,
    const torch::Tensor& flat_expert,
    const torch::Tensor& flat_weight,
    const torch::Tensor& flat_owner,
    int64_t local_rank,
    int64_t world_size) {
    TORCH_CHECK(flat_x.dim() == 2, "flat_x must have shape [N, D]");
    TORCH_CHECK(flat_expert.dim() == 1, "flat_expert must have shape [N]");
    TORCH_CHECK(flat_weight.dim() == 1, "flat_weight must have shape [N]");
    TORCH_CHECK(flat_owner.dim() == 1, "flat_owner must have shape [N]");
    TORCH_CHECK(flat_x.size(0) == flat_expert.size(0), "flat_expert row count mismatch");
    TORCH_CHECK(flat_x.size(0) == flat_weight.size(0), "flat_weight row count mismatch");
    TORCH_CHECK(flat_x.size(0) == flat_owner.size(0), "flat_owner row count mismatch");
    TORCH_CHECK(local_rank >= 0 && local_rank < world_size, "local_rank out of range");
    TORCH_CHECK(world_size > 0, "world_size must be positive");

    check_cuda_contiguous(flat_x, "flat_x");
    check_cuda_contiguous(flat_expert, "flat_expert");
    check_cuda_contiguous(flat_weight, "flat_weight");
    check_cuda_contiguous(flat_owner, "flat_owner");

    TORCH_CHECK(
        flat_x.scalar_type() == torch::kFloat16 ||
        flat_x.scalar_type() == torch::kBFloat16 ||
        flat_x.scalar_type() == torch::kFloat32,
        "flat_x must be float16, bfloat16, or float32");
    TORCH_CHECK(flat_expert.scalar_type() == torch::kInt32, "flat_expert must be int32");
    TORCH_CHECK(flat_weight.scalar_type() == torch::kFloat32, "flat_weight must be float32");
    TORCH_CHECK(flat_owner.scalar_type() == torch::kInt32, "flat_owner must be int32");

    return partition_items_padded_cuda(flat_x, flat_expert, flat_weight, flat_owner, local_rank, world_size);
}

std::vector<torch::Tensor> compact_payload(
    const torch::Tensor& payload) {
    TORCH_CHECK(payload.dim() == 2, "payload must have shape [N, D+2]");
    check_cuda_contiguous(payload, "payload");
    TORCH_CHECK(
        payload.scalar_type() == torch::kFloat16 ||
        payload.scalar_type() == torch::kBFloat16 ||
        payload.scalar_type() == torch::kFloat32,
        "payload must be float16, bfloat16, or float32");
    TORCH_CHECK(payload.size(1) >= 3, "payload width must be at least 3");
    return compact_payload_cuda(payload);
}

void scatter_rows(
    const torch::Tensor& out,
    const torch::Tensor& positions,
    const torch::Tensor& values) {
    TORCH_CHECK(out.dim() == 2, "out must have shape [N, D]");
    TORCH_CHECK(positions.dim() == 1, "positions must have shape [R]");
    TORCH_CHECK(values.dim() == 2, "values must have shape [R, D]");
    TORCH_CHECK(values.size(0) == positions.size(0), "values row count mismatch");
    TORCH_CHECK(values.size(1) == out.size(1), "values dim mismatch");

    check_cuda_contiguous(out, "out");
    check_cuda_contiguous(positions, "positions");
    check_cuda_contiguous(values, "values");

    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");
    TORCH_CHECK(positions.scalar_type() == torch::kInt32, "positions must be int32");
    TORCH_CHECK(
        values.scalar_type() == torch::kFloat16 ||
        values.scalar_type() == torch::kBFloat16 ||
        values.scalar_type() == torch::kFloat32,
        "values must be float16, bfloat16, or float32");

    scatter_rows_cuda(out, positions, values);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("partition_items", &partition_items, "Partition decode dispatch items (CUDA)");
    m.def("partition_items_padded", &partition_items_padded, "Partition decode dispatch items with padded owner buckets (CUDA)");
    m.def("compact_payload", &compact_payload, "Compact padded decode dispatch payload (CUDA)");
    m.def("scatter_rows", &scatter_rows, "Scatter rows back to flat output (CUDA)");
}
