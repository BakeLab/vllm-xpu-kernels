# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401


@pytest.mark.parametrize("tokens,heads", [(1, 1), (3, 2), (5, 4)])
@pytest.mark.parametrize("strided", [False, True])
def test_kimi_k3_mla_kv_concat(tokens, heads, strided):
    torch.manual_seed(19)
    device = "xpu"
    if strided:
        nope_storage = torch.randn(
            tokens, heads, 256, device=device, dtype=torch.bfloat16
        )
        k_nope = nope_storage[..., ::2]
        pe_storage = torch.randn(
            tokens, 2, 64, device=device, dtype=torch.bfloat16
        )
        k_pe = pe_storage[:, 0, :]
    else:
        k_nope = torch.randn(tokens, heads, 128, device=device, dtype=torch.bfloat16)
        k_pe = torch.randn(tokens, 64, device=device, dtype=torch.bfloat16)
    k_out = torch.empty(tokens, heads, 192, device=device, dtype=torch.bfloat16)

    torch.ops._xpu_C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out)

    expected = torch.cat((k_nope, k_pe[:, None, :].expand(-1, heads, -1)), dim=-1)
    torch.testing.assert_close(k_out, expected)
