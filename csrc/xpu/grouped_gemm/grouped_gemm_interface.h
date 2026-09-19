#include <torch/all.h>

torch::Tensor cutlass_grouped_gemm_interface(
    torch::Tensor ptr_A,
    const std::optional<at::Tensor>& ptr_A_scale,
    torch::Tensor ptr_B,
    const std::optional<at::Tensor>& ptr_B_scale,
    const std::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts);
