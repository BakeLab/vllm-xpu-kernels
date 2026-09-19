// SPDX-License-Identifier: Apache-2.0
#include <sycl/sycl.hpp>

#include "xpu/ops.h"
#include "utils.h"

namespace {
using bf16_t = sycl::ext::oneapi::bfloat16;

class kimi_k3_decode_kernel {
 public:
  kimi_k3_decode_kernel(
      const bf16_t* ql_nope,
      const bf16_t* q_pe,
      const bf16_t* kv_c_normed,
      const bf16_t* k_pe,
      bf16_t* mqa_q,
      bf16_t* k_cache,
      const int64_t* slot_mapping,
      const int64_t* position_ids,
      const float* cos_sin_cache,
      int64_t tokens,
      int64_t heads,
      int64_t ql_token_stride,
      int64_t ql_head_stride,
      int64_t qpe_token_stride,
      int64_t qpe_head_stride,
      int64_t kv_token_stride,
      int64_t kpe_token_stride,
      int64_t mqa_token_stride,
      int64_t mqa_head_stride,
      int64_t cache_block_stride,
      int64_t cache_token_stride,
      int64_t cos_sin_stride,
      int64_t block_size,
      int64_t rows_per_token,
      bool rope)
      : ql_nope_(ql_nope),
        q_pe_(q_pe),
        kv_c_normed_(kv_c_normed),
        k_pe_(k_pe),
        mqa_q_(mqa_q),
        k_cache_(k_cache),
        slot_mapping_(slot_mapping),
        position_ids_(position_ids),
        cos_sin_cache_(cos_sin_cache),
        tokens_(tokens),
        heads_(heads),
        ql_token_stride_(ql_token_stride),
        ql_head_stride_(ql_head_stride),
        qpe_token_stride_(qpe_token_stride),
        qpe_head_stride_(qpe_head_stride),
        kv_token_stride_(kv_token_stride),
        kpe_token_stride_(kpe_token_stride),
        mqa_token_stride_(mqa_token_stride),
        mqa_head_stride_(mqa_head_stride),
        cache_block_stride_(cache_block_stride),
        cache_token_stride_(cache_token_stride),
        cos_sin_stride_(cos_sin_stride),
        block_size_(block_size),
        rows_per_token_(rows_per_token),
        rope_(rope) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t linear = item.get_global_linear_id();
    const int64_t row = linear / 576;
    const int64_t dim = linear % 576;
    const int64_t token = row / rows_per_token_;
    const int64_t head = row % rows_per_token_;
    if (token >= tokens_) return;

    if (head < heads_) {
      const int64_t src = token * ql_token_stride_ + head * ql_head_stride_;
      const int64_t pe_src =
          token * qpe_token_stride_ + head * qpe_head_stride_;
      float value;
      if (dim < 512) {
        value = static_cast<float>(ql_nope_[src + dim]);
      } else {
        const int64_t rope_dim = dim - 512;
        value = static_cast<float>(q_pe_[pe_src + rope_dim]);
        if (rope_) value = rotate_rope(value, q_pe_ + pe_src, token, rope_dim);
      }
      mqa_q_[token * mqa_token_stride_ + head * mqa_head_stride_ + dim] =
          static_cast<bf16_t>(value);
      return;
    }

    const int64_t slot = slot_mapping_[token];
    if (slot < 0) return;
    const int64_t cache_offset = (slot / block_size_) * cache_block_stride_ +
                                 (slot % block_size_) * cache_token_stride_;
    float value;
    if (dim < 512) {
      value = static_cast<float>(kv_c_normed_[token * kv_token_stride_ + dim]);
    } else {
      const int64_t rope_dim = dim - 512;
      const int64_t pe_src = token * kpe_token_stride_;
      value = static_cast<float>(k_pe_[pe_src + rope_dim]);
      if (rope_) value = rotate_rope(value, k_pe_ + pe_src, token, rope_dim);
    }
    k_cache_[cache_offset + dim] = static_cast<bf16_t>(value);
  }

 private:
  float rotate_rope(
      float value, const bf16_t* src, int64_t token, int64_t rope_dim) const {
    const int64_t pair = rope_dim / 2;
    const int64_t position = position_ids_[token];
    const int64_t cache_row = position * cos_sin_stride_;
    const float cos = cos_sin_cache_[cache_row + pair];
    const float sin = cos_sin_cache_[cache_row + 32 + pair];
    const float x = static_cast<float>(src[pair * 2]);
    const float y = static_cast<float>(src[pair * 2 + 1]);
    return (rope_dim & 1) ? x * sin + y * cos : x * cos - y * sin;
  }

  const bf16_t* ql_nope_;
  const bf16_t* q_pe_;
  const bf16_t* kv_c_normed_;
  const bf16_t* k_pe_;
  bf16_t* mqa_q_;
  bf16_t* k_cache_;
  const int64_t* slot_mapping_;
  const int64_t* position_ids_;
  const float* cos_sin_cache_;
  int64_t tokens_, heads_;
  int64_t ql_token_stride_, ql_head_stride_;
  int64_t qpe_token_stride_, qpe_head_stride_;
  int64_t kv_token_stride_, kpe_token_stride_;
  int64_t mqa_token_stride_, mqa_head_stride_;
  int64_t cache_block_stride_, cache_token_stride_, cos_sin_stride_;
  int64_t block_size_, rows_per_token_;
  bool rope_;
};
}  // namespace

void fused_kimi_k3_mla_decode_q_concat_kv_cache_insert(
    const torch::Tensor& ql_nope,
    const torch::Tensor& q_pe,
    const torch::Tensor& kv_c_normed,
    const torch::Tensor& k_pe,
    torch::Tensor& mqa_q,
    torch::Tensor& k_cache,
    const torch::Tensor& slot_mapping,
    int64_t cache_block_size,
    std::optional<torch::Tensor> position_ids,
    std::optional<torch::Tensor> cos_sin_cache) {
  TORCH_CHECK(
      ql_nope.scalar_type() == at::kBFloat16 &&
          q_pe.scalar_type() == at::kBFloat16 &&
          kv_c_normed.scalar_type() == at::kBFloat16 &&
          k_pe.scalar_type() == at::kBFloat16 &&
          mqa_q.scalar_type() == at::kBFloat16 &&
          k_cache.scalar_type() == at::kBFloat16,
      "Kimi-K3 XPU decode currently supports BF16 only");
  TORCH_CHECK(
      ql_nope.dim() == 3 && ql_nope.size(2) == 512,
      "expected ql_nope [B,H,512]");
  TORCH_CHECK(
      ql_nope.stride(2) == 1,
      "ql_nope must be contiguous on its last dimension");
  const int64_t tokens = ql_nope.size(0);
  const int64_t heads = ql_nope.size(1);
  TORCH_CHECK(
      q_pe.dim() == 3 && q_pe.size(0) == tokens && q_pe.size(1) == heads &&
          q_pe.size(2) == 64,
      "expected q_pe [B,H,64]");
  TORCH_CHECK(
      q_pe.stride(2) == 1, "q_pe must be contiguous on its last dimension");
  TORCH_CHECK(
      kv_c_normed.dim() == 2 && kv_c_normed.size(0) == tokens &&
          kv_c_normed.size(1) == 512,
      "expected kv_c_normed [B,512]");
  TORCH_CHECK(
      kv_c_normed.stride(1) == 1,
      "kv_c_normed must be contiguous on its last dimension");
  TORCH_CHECK(
      k_pe.dim() == 2 && k_pe.size(0) == tokens && k_pe.size(1) == 64,
      "expected k_pe [B,64]");
  TORCH_CHECK(
      k_pe.stride(1) == 1, "k_pe must be contiguous on its last dimension");
  TORCH_CHECK(
      mqa_q.dim() == 3 && mqa_q.size(0) == tokens && mqa_q.size(1) == heads &&
          mqa_q.size(2) == 576,
      "expected mqa_q [B,H,576]");
  TORCH_CHECK(
      mqa_q.stride(2) == 1, "mqa_q must be contiguous on its last dimension");
  TORCH_CHECK(
      cache_block_size > 0 && k_cache.dim() == 3 &&
          k_cache.size(1) == cache_block_size && k_cache.size(2) == 576 &&
          k_cache.stride(2) == 1,
      "expected k_cache [blocks,block_size,576]");
  TORCH_CHECK(
      slot_mapping.scalar_type() == at::kLong && slot_mapping.dim() == 1 &&
          slot_mapping.numel() >= tokens,
      "slot_mapping must be int64 [B]");
  TORCH_CHECK(
      ql_nope.device() == q_pe.device() &&
          ql_nope.device() == kv_c_normed.device() &&
          ql_nope.device() == k_pe.device() &&
          ql_nope.device() == mqa_q.device() &&
          ql_nope.device() == k_cache.device() &&
          ql_nope.device() == slot_mapping.device(),
      "Kimi-K3 decode tensors must be on the same XPU");

  const bool rope = position_ids.has_value() && cos_sin_cache.has_value();
  TORCH_CHECK(
      position_ids.has_value() == cos_sin_cache.has_value(),
      "position_ids and cos_sin_cache must be provided together");
  if (rope) {
    TORCH_CHECK(
        position_ids->scalar_type() == at::kLong && position_ids->dim() == 1 &&
            position_ids->numel() >= tokens &&
            cos_sin_cache->scalar_type() == at::kFloat &&
            cos_sin_cache->dim() == 2 && cos_sin_cache->size(1) == 64 &&
            cos_sin_cache->stride(1) == 1 &&
            position_ids->device() == ql_nope.device() &&
            cos_sin_cache->device() == ql_nope.device(),
        "invalid Kimi-K3 decode RoPE tensors");
  }
  if (tokens == 0) return;

  const int64_t rows_per_token = heads + 1;
  const int64_t total = tokens * rows_per_token * 576;
  constexpr int64_t local = 256;
  at::DeviceGuard guard(ql_nope.device());
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>((total + local - 1) / local * local, local),
        kimi_k3_decode_kernel(
            reinterpret_cast<const bf16_t*>(ql_nope.data_ptr()),
            reinterpret_cast<const bf16_t*>(q_pe.data_ptr()),
            reinterpret_cast<const bf16_t*>(kv_c_normed.data_ptr()),
            reinterpret_cast<const bf16_t*>(k_pe.data_ptr()),
            reinterpret_cast<bf16_t*>(mqa_q.data_ptr()),
            reinterpret_cast<bf16_t*>(k_cache.data_ptr()),
            slot_mapping.data_ptr<int64_t>(),
            rope ? position_ids->data_ptr<int64_t>() : nullptr,
            rope ? cos_sin_cache->data_ptr<float>() : nullptr,
            tokens,
            heads,
            ql_nope.stride(0),
            ql_nope.stride(1),
            q_pe.stride(0),
            q_pe.stride(1),
            kv_c_normed.stride(0),
            k_pe.stride(0),
            mqa_q.stride(0),
            mqa_q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            rope ? cos_sin_cache->stride(0) : 0,
            cache_block_size,
            rows_per_token,
            rope));
  });
}
