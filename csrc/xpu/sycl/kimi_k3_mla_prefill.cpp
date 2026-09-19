// SPDX-License-Identifier: Apache-2.0
#include <sycl/sycl.hpp>

#include "xpu/ops.h"
#include "utils.h"

namespace {
using bf16_t = sycl::ext::oneapi::bfloat16;

class kimi_k3_prefill_kernel {
 public:
  kimi_k3_prefill_kernel(
      bf16_t* q, const bf16_t* nope, const bf16_t* pe,
      const bf16_t* latent, bf16_t* out, bf16_t* cache,
      const int64_t* slots, const int64_t* positions, const float* cs,
      int64_t tokens, int64_t heads, int64_t qts, int64_t qhs,
      int64_t nts, int64_t nhs, int64_t pes, int64_t cbs,
      int64_t block_stride, int64_t block_size, int64_t max_work,
      bool rope)
      : q(q), nope(nope), pe(pe), latent(latent), out(out), cache(cache),
        slots(slots), positions(positions), cs(cs), tokens(tokens),
        heads(heads), qts(qts), qhs(qhs), nts(nts), nhs(nhs), pes(pes),
        cbs(cbs), block_stride(block_stride), block_size(block_size),
        max_work(max_work), rope(rope) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t i = item.get_global_linear_id();
    const int64_t token = i / max_work;
    const int64_t x = i % max_work;
    if (token >= tokens) return;

    if (x < heads * 192) {
      const int64_t h = x / 192;
      const int d = x % 192;
      if (d < 128) {
        out[token * heads * 192 + x] = nope[token * nts + h * nhs + d];
      } else {
        const int r = d - 128;
        out[token * heads * 192 + x] = pe[token * pes + r];
      }
      if (rope && d >= 128 && (d - 128) % 2 == 0) {
        const int r = d - 128;
        const int pair = r / 2;
        const int64_t qbase = token * qts + h * qhs + 128;
        const int64_t pos = positions[token];
        const float c = cs[pos * cbs + pair];
        const float s = cs[pos * cbs + 32 + pair];
        const float a = static_cast<float>(q[qbase + pair * 2]);
        const float b = static_cast<float>(q[qbase + pair * 2 + 1]);
        q[qbase + r] = static_cast<bf16_t>(a*c-b*s);
        q[qbase + r + 1] = static_cast<bf16_t>(a*s+b*c);
      }
    }
    if (x < 576) {
      const int64_t slot = slots[token];
      if (slot >= 0) {
        const int64_t dst = (slot / block_size) * block_stride +
                            (slot % block_size) * 576 + x;
        cache[dst] = x < 512 ? latent[token * 512 + x]
                             : pe[token * pes + x - 512];
      }
    }
  }

 private:
  bf16_t* q; const bf16_t* nope; const bf16_t* pe; const bf16_t* latent;
  bf16_t* out; bf16_t* cache; const int64_t* slots;
  const int64_t* positions; const float* cs;
  int64_t tokens, heads, qts, qhs, nts, nhs, pes, cbs;
  int64_t block_stride, block_size, max_work;
  bool rope;
};
}  // namespace

void fused_kimi_k3_mla_key_concat_kv_cache_insert(
    torch::Tensor& q, const torch::Tensor& k_nope,
    const torch::Tensor& k_pe, const torch::Tensor& kv_c_normed,
    torch::Tensor& k_out, torch::Tensor& k_cache,
    const torch::Tensor& slot_mapping, int64_t cache_block_size,
    std::optional<torch::Tensor> position_ids,
    std::optional<torch::Tensor> cos_sin_cache) {
  TORCH_CHECK(k_nope.scalar_type() == at::kBFloat16,
              "Kimi-K3 XPU prefill currently supports BF16 only");
  TORCH_CHECK(q.dim() == 3 && k_nope.dim() == 3 && k_nope.size(2) == 128 &&
              q.size(0) == k_nope.size(0) && q.size(1) == k_nope.size(1) &&
              q.size(2) == 192, "expected q [T,H,192], k_nope [T,H,128]");
  TORCH_CHECK(k_pe.dim() == 2 && k_pe.size(0) == q.size(0) &&
              k_pe.size(1) == 64 && k_pe.stride(1) == 1,
              "expected k_pe [T,64] with unit inner stride");
  TORCH_CHECK(kv_c_normed.dim() == 2 && kv_c_normed.size(1) == 512 &&
              kv_c_normed.is_contiguous(), "expected contiguous kv_c_normed [T,512]");
  TORCH_CHECK(k_out.is_contiguous() && k_out.size(0) == q.size(0) &&
              k_out.size(1) == q.size(1) && k_out.size(2) == 192,
              "expected contiguous k_out [T,H,192]");
  TORCH_CHECK(k_cache.scalar_type() == at::kBFloat16 && k_cache.dim() == 3 &&
              k_cache.size(1) == cache_block_size && k_cache.size(2) == 576,
              "expected BF16 cache [blocks,block_size,576]");
  TORCH_CHECK(slot_mapping.scalar_type() == at::kLong &&
              slot_mapping.numel() >= q.size(0), "invalid slot_mapping");
  const bool rope = position_ids.has_value() && cos_sin_cache.has_value();
  TORCH_CHECK(position_ids.has_value() == cos_sin_cache.has_value(),
              "position_ids and cos_sin_cache must be provided together");
  if (rope) TORCH_CHECK(cos_sin_cache->scalar_type() == at::kFloat &&
                        cos_sin_cache->size(1) == 64,
                        "expected fp32 cos_sin_cache [positions,64]");

  const int64_t tokens = q.size(0), heads = q.size(1);
  if (!tokens) return;
  const int64_t work = std::max<int64_t>(heads * 192, 576);
  const int64_t total = tokens * work;
  const int64_t local = 256;
  at::DeviceGuard guard(q.device());
  auto& queue = vllm::xpu::vllmGetQueue();
  auto* pos = rope ? position_ids->data_ptr<int64_t>() : nullptr;
  auto* cs = rope ? cos_sin_cache->data_ptr<float>() : nullptr;
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>((total + local - 1) / local * local, local),
      kimi_k3_prefill_kernel(
        reinterpret_cast<bf16_t*>(q.data_ptr()),
        reinterpret_cast<const bf16_t*>(k_nope.data_ptr()),
        reinterpret_cast<const bf16_t*>(k_pe.data_ptr()),
        reinterpret_cast<const bf16_t*>(kv_c_normed.data_ptr()),
        reinterpret_cast<bf16_t*>(k_out.data_ptr()),
        reinterpret_cast<bf16_t*>(k_cache.data_ptr()),
        slot_mapping.data_ptr<int64_t>(), pos, cs, tokens, heads,
        q.stride(0), q.stride(1), k_nope.stride(0), k_nope.stride(1),
        k_pe.stride(0), rope ? cos_sin_cache->stride(0) : 0,
        k_cache.stride(0), cache_block_size, work, rope));
  });
}
