#include <torch/extension.h>

#include <vector>

torch::Tensor wo_a_int8_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_s);

namespace {

void check_tensor(const torch::Tensor& tensor, const char* name, c10::ScalarType dtype) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has unexpected dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor wo_a_int8_forward(
    const torch::Tensor& x,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_s) {
    TORCH_CHECK(x.dim() == 4, "x must have shape [B, S, G, D]");
    TORCH_CHECK(weight_q.dim() == 3, "weight_q must have shape [G, R, D]");
    TORCH_CHECK(weight_s.dim() == 2, "weight_s must have shape [G, R]");
    TORCH_CHECK(x.size(2) == weight_q.size(0), "group dimension mismatch");
    TORCH_CHECK(weight_q.size(0) == weight_s.size(0), "weight group mismatch");
    TORCH_CHECK(weight_q.size(1) == weight_s.size(1), "weight rank mismatch");
    TORCH_CHECK(x.size(3) == weight_q.size(2), "inner dimension mismatch");
    TORCH_CHECK(x.size(3) % 4 == 0, "D must be divisible by 4");

    TORCH_CHECK(x.is_cuda() && weight_q.is_cuda() && weight_s.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    check_tensor(weight_q, "weight_q", torch::kInt8);
    check_tensor(weight_s, "weight_s", torch::kFloat32);
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16 ||
        x.scalar_type() == torch::kFloat32,
        "x must be float16, bfloat16, or float32");

    return wo_a_int8_forward_cuda(x, weight_q, weight_s);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wo_a_int8_forward", &wo_a_int8_forward, "wo_a grouped int8 forward (CUDA)");
}
