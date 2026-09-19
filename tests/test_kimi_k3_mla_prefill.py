# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401


@pytest.mark.parametrize(
    "tokens,heads,block_size",
    [(1, 1, 16), (3, 2, 4), (5, 4, 8)],
)
@pytest.mark.parametrize("use_rope", [False, True])
def test_kimi_k3_mla_bf16_prefill(tokens, heads, block_size, use_rope):
    torch.manual_seed(17)
    device = "xpu"
    q = torch.randn(tokens, heads, 192, device=device, dtype=torch.bfloat16)
    q_ref = q.clone()
    nope = torch.randn(tokens, heads, 128, device=device, dtype=torch.bfloat16)
    pe_storage = torch.randn(tokens, 2, 64, device=device, dtype=torch.bfloat16)
    pe = pe_storage[:, 0, :]  # contiguous inner dimension, padded row stride
    latent = torch.randn(tokens, 512, device=device, dtype=torch.bfloat16)
    cache = torch.zeros(4, block_size, 576, device=device, dtype=torch.bfloat16)
    ref_cache = cache.clone()
    slots = torch.tensor(
        [(-1 if t % 3 == 1 else (t * 3) % (cache.size(0) * block_size))
         for t in range(tokens)],
        device=device,
        dtype=torch.long,
    )
    positions = torch.arange(tokens, device=device, dtype=torch.long) + 1
    cs = torch.randn(8, 64, device=device, dtype=torch.float32)
    out = torch.empty(tokens, heads, 192, device=device, dtype=torch.bfloat16)
    kwargs = {"position_ids": positions, "cos_sin_cache": cs} if use_rope else {
        "position_ids": None, "cos_sin_cache": None
    }

    torch.ops._xpu_C.fused_kimi_k3_mla_key_concat_kv_cache_insert(
        q, nope, pe, latent, out, cache, slots, block_size, **kwargs
    )

    expected = torch.cat((nope, pe[:, None, :].expand(-1, heads, -1)), dim=-1)
    if use_rope:
        for t in range(tokens):
            c, s = cs[positions[t], :32], cs[positions[t], 32:]
            even, odd = q_ref[t, :, 128::2].float(), q_ref[t, :, 129::2].float()
            q_ref[t, :, 128::2] = (even*c - odd*s).to(torch.bfloat16)
            q_ref[t, :, 129::2] = (even*s + odd*c).to(torch.bfloat16)
    for t, slot in enumerate(slots.tolist()):
        if slot >= 0:
            ref_cache[slot // block_size, slot % block_size] = torch.cat(
                (latent[t], pe[t])
            )
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(q, q_ref)
    torch.testing.assert_close(cache, ref_cache)
