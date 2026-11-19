# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""specialized_prefill.draft_attention == FlashAttention 2 (the DSpark drafter's
backend) on the prefill draft step: causal and non-causal sliding-window GQA over
a paged BF16 cache in FlashAttention's [blocks, kv_heads, page, 2 x head_dim]
layout, a few query rows per request, windows shorter and longer than the
context."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="Kimi K3 prefill kernels target SM100+",
)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("window", [2048, -1])
@pytest.mark.parametrize("page", [16, 64])
def test_draft_attention_matches_flash_attn_2(causal, window, page):
    from vllm.models.kimi_k3.nvidia.specialized_prefill import draft_attention
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    torch.manual_seed(page)
    heads, kv_heads, hd, q_len = 96, 16, 64, 4
    seq_lens = torch.tensor([2600, 6157, 300], dtype=torch.int32, device="cuda")
    reqs = seq_lens.numel()
    max_pages = (int(seq_lens.max()) + page - 1) // page
    blocks = reqs * max_pages + 4
    cache = torch.randn(blocks, kv_heads, page, 2 * hd, device="cuda") * 2
    cache = cache.to(torch.bfloat16)
    key_cache, value_cache = cache.transpose(1, 2).split(hd, dim=-1)
    block_table = torch.randperm(blocks, device="cuda")[: reqs * max_pages]
    block_table = block_table.view(reqs, max_pages).to(torch.int32)
    cu_q = torch.arange(0, (reqs + 1) * q_len, q_len, device="cuda", dtype=torch.int32)
    q = torch.randn(reqs * q_len, heads, hd, device="cuda").to(torch.bfloat16)
    scale = hd**-0.5
    win = (window - 1, window - 1) if window > 0 else (-1, -1)
    if causal:
        win = (win[0], 0)
    ref = flash_attn_varlen_func(
        q=q, k=key_cache, v=value_cache, cu_seqlens_q=cu_q, max_seqlen_q=q_len,
        seqused_k=seq_lens, max_seqlen_k=int(seq_lens.max()), softmax_scale=scale,
        causal=causal, window_size=list(win), block_table=block_table, fa_version=2,
    )  # fmt: skip
    assert draft_attention.covers(q_len, heads, kv_heads, hd, key_cache, value_cache)
    out = torch.empty_like(q)
    draft_attention.attention(
        q, key_cache, value_cache, block_table, seq_lens, cu_q, q_len, scale, win, out
    )
    diff = (out.float() - ref.float()).abs()
    # Same math, another summation order: within a BF16 rounding of the output.
    assert diff.max().item() <= 2 * 2**-8 * ref.float().abs().max().item()
    assert diff.mean().item() < 1e-3
