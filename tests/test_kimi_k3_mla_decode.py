# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401


@pytest.mark.parametrize("tokens,heads,block_size", [(1, 2, 16), (5, 3, 8)])
@pytest.mark.parametrize("use_rope", [False, True])
def test_kimi_k3_mla_bf16_decode(tokens, heads, block_size, use_rope):
    torch.manual_seed(31)
    device = "xpu"
    ql_nope = torch.randn(tokens, heads, 512, device=device, dtype=torch.bfloat16)
    q_pe_storage = torch.randn(
        tokens, heads, 2, 64, device=device, dtype=torch.bfloat16
    )
    q_pe = q_pe_storage[:, :, 0]
    kv_c_normed = torch.randn(tokens, 512, device=device, dtype=torch.bfloat16)
    k_pe_storage = torch.randn(tokens, 2, 64, device=device, dtype=torch.bfloat16)
    k_pe = k_pe_storage[:, 0]
    cache = torch.randn(4, block_size, 576, device=device, dtype=torch.bfloat16)
    expected_cache = cache.clone()
    slots = torch.tensor(
        [-1 if token % 3 == 1 else (token * 3) % (cache.size(0) * block_size)
         for token in range(tokens)],
        device=device,
        dtype=torch.long,
    )
    positions = torch.arange(tokens, device=device, dtype=torch.long) + 1
    cos_sin_cache = torch.randn(8, 64, device=device, dtype=torch.float32)
    mqa_q = torch.empty(
        tokens, heads, 576, device=device, dtype=torch.bfloat16
    )
    expected_q = torch.cat(
        (ql_nope, q_pe.expand(-1, heads, -1)), dim=-1
    )
    kwargs = (
        {"position_ids": positions, "cos_sin_cache": cos_sin_cache}
        if use_rope
        else {"position_ids": None, "cos_sin_cache": None}
    )

    if use_rope:
        for token in range(tokens):
            cos = cos_sin_cache[positions[token], :32]
            sin = cos_sin_cache[positions[token], 32:]
            for pe in (expected_q[token, :, 512:],):
                even = pe[:, 0::2].float().clone()
                odd = pe[:, 1::2].float().clone()
                pe[:, 0::2] = (even * cos - odd * sin).to(torch.bfloat16)
                pe[:, 1::2] = (even * sin + odd * cos).to(torch.bfloat16)

    for token, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        entry = torch.cat((kv_c_normed[token], k_pe[token]))
        if use_rope:
            cos = cos_sin_cache[positions[token], :32]
            sin = cos_sin_cache[positions[token], 32:]
            even = entry[512::2].float().clone()
            odd = entry[513::2].float().clone()
            entry[512::2] = (even * cos - odd * sin).to(torch.bfloat16)
            entry[513::2] = (even * sin + odd * cos).to(torch.bfloat16)
        expected_cache[slot // block_size, slot % block_size] = entry

    torch.ops._xpu_C.fused_kimi_k3_mla_decode_q_concat_kv_cache_insert(
        ql_nope,
        q_pe,
        kv_c_normed,
        k_pe,
        mqa_q,
        cache,
        slots,
        block_size,
        **kwargs,
    )

    torch.testing.assert_close(mqa_q, expected_q)
    torch.testing.assert_close(cache, expected_cache)
