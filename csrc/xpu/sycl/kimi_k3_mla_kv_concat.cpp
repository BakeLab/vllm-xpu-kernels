// SPDX-License-Identifier: Apache-2.0
#include <sycl/sycl.hpp>

#include "utils.h"
#include "xpu/ops.h"

namespace {
using bf16_t = sycl::ext::oneapi::bfloat16;

class kimi_k3_mla_kv_concat_kernel {
 public:
  kimi_k3_mla_kv_concat_kernel(
      const bf16_t* k_nope,
      const bf16_t* k_pe,
      bf16_t* k_out,
      int64_t tokens,
      int64_t heads,
      int64_t nope_dim,
      int64_t rope_dim,
      int64_t nope_token_stride,
      int64_t nope_head_stride,
      int64_t nope_dim_stride,
      int64_t pe_token_stride,
      int64_t pe_dim_stride)
      : k_nope_(k_nope),
        k_pe_(k_pe),
        k_out_(k_out),
        tokens_(tokens),
        heads_(heads),
        nope_dim_(nope_dim),
        rope_dim_(rope_dim),
        nope_token_stride_(nope_token_stride),
        nope_head_stride_(nope_head_stride),
        nope_dim_stride_(nope_dim_stride),
        pe_token_stride_(pe_token_stride),
        pe_dim_stride_(pe_dim_stride) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t index = item.get_global_linear_id();
    const int64_t out_dim = nope_dim_ + rope_dim_;
    const int64_t row = index / out_dim;
    const int64_t dim = index % out_dim;
    const int64_t token = row / heads_;
    const int64_t head = row % heads_;
    if (token >= tokens_) return;

    if (dim < nope_dim_) {
      k_out_[index] = k_nope_
          [token * nope_token_stride_ + head * nope_head_stride_ +
           dim * nope_dim_stride_];
    } else {
      const int64_t rope_dim = dim - nope_dim_;
      k_out_[index] =
          k_pe_[token * pe_token_stride_ + rope_dim * pe_dim_stride_];
    }
  }

 private:
  const bf16_t* k_nope_;
  const bf16_t* k_pe_;
  bf16_t* k_out_;
  int64_t tokens_, heads_, nope_dim_, rope_dim_;
  int64_t nope_token_stride_, nope_head_stride_, nope_dim_stride_;
  int64_t pe_token_stride_, pe_dim_stride_;
};
}  // namespace

void fused_kimi_k3_mla_kv_concat(
    const torch::Tensor& k_nope,
    const torch::Tensor& k_pe,
    torch::Tensor& k_out) {
  TORCH_CHECK(
      k_nope.scalar_type() == at::kBFloat16 &&
          k_pe.scalar_type() == at::kBFloat16 &&
          k_out.scalar_type() == at::kBFloat16,
      "Kimi-K3 XPU KV concat currently supports BF16 only");
  TORCH_CHECK(
      k_nope.dim() == 3 && k_nope.size(2) == 128, "expected k_nope [T,H,128]");
  const int64_t tokens = k_nope.size(0);
  const int64_t heads = k_nope.size(1);
  TORCH_CHECK(
      k_pe.dim() == 2 && k_pe.size(0) == tokens && k_pe.size(1) == 64,
      "expected k_pe [T,64]");
  TORCH_CHECK(
      k_out.dim() == 3 && k_out.size(0) == tokens && k_out.size(1) == heads &&
          k_out.size(2) == 192 && k_out.is_contiguous(),
      "expected contiguous k_out [T,H,192]");
  TORCH_CHECK(
      k_nope.device() == k_pe.device() && k_nope.device() == k_out.device(),
      "Kimi-K3 KV concat tensors must be on the same XPU");
  TORCH_CHECK(
      k_nope.device().is_xpu(), "Kimi-K3 KV concat tensors must be on XPU");
  if (tokens == 0) return;

  const int64_t total = tokens * heads * 192;
  constexpr int64_t local = 256;
  at::DeviceGuard guard(k_nope.device());
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>((total + local - 1) / local * local, local),
        kimi_k3_mla_kv_concat_kernel(
            reinterpret_cast<const bf16_t*>(k_nope.data_ptr()),
            reinterpret_cast<const bf16_t*>(k_pe.data_ptr()),
            reinterpret_cast<bf16_t*>(k_out.data_ptr()),
            tokens,
            heads,
            k_nope.size(2),
            k_pe.size(1),
            k_nope.stride(0),
            k_nope.stride(1),
            k_nope.stride(2),
            k_pe.stride(0),
            k_pe.stride(1)));
  });
}
