# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TokenSpeed MLA decode with length-proportional split-KV (max_split_kv)
matches the uniform-split decode and an FP32 reference on skewed batches."""

import pytest
import torch

tokenspeed_mla = pytest.importorskip("tokenspeed_mla")
pytestmark = pytest.mark.skipif(
    not hasattr(tokenspeed_mla, "tokenspeed_mla_var_split_counts"),
    reason="tokenspeed_mla without length-proportional split-KV",
)

KV_LORA, ROPE, PAGE = 512, 64, 64


def _ref(q, kv, block_table, seq_lens, scale):
    # FP32 non-causal MLA over each request's whole context (DSpark blocks).
    out = []
    for b, n in enumerate(seq_lens.tolist()):
        pages = block_table[b, : -(-n // PAGE)]
        c = kv[pages].reshape(-1, KV_LORA + ROPE)[:n].float()
        s = torch.einsum("qhd,td->qht", q[b].float(), c) * scale
        p = s.softmax(-1)
        out.append(torch.einsum("qht,td->qhd", p, c[:, :KV_LORA]))
    return torch.stack(out)


@pytest.mark.parametrize("total_splits", [0, 96])
@pytest.mark.parametrize("max_split", [16, 64])
@pytest.mark.parametrize("q_len", [4, 8])
def test_var_split_matches_reference(max_split, q_len, total_splits):
    torch.manual_seed(0)
    heads = 64
    lens = [4096, 37888, 131072, 3000, 659456, 129]
    pages_per = [-(-n // PAGE) for n in lens]
    total = sum(pages_per)
    kv = (torch.randn(total, PAGE, KV_LORA + ROPE, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    perm = torch.randperm(total, device="cuda").to(torch.int32)
    width = max(pages_per)
    block_table = torch.zeros(len(lens), width, dtype=torch.int32, device="cuda")
    off = 0
    for b, p in enumerate(pages_per):
        block_table[b, :p] = perm[off : off + p]
        off += p
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    q = (torch.randn(len(lens), q_len, heads, KV_LORA + ROPE, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    scale = (KV_LORA + ROPE) ** -0.5
    ws = torch.empty(1 << 30, dtype=torch.int8, device="cuda")
    common = dict(
        query=q,
        kv_cache=kv,
        workspace_buffer=ws,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE,
        block_tables=block_table,
        seq_lens=seq_lens,
        max_seq_len=max(lens),
        softmax_scale=scale,
        causal_mask=False,
    )
    base = tokenspeed_mla.tokenspeed_mla_decode(**common, min_split_kv=16)
    counts = tokenspeed_mla.tokenspeed_mla_var_split_counts(
        seq_lens, max_split, heads, q_len, total_splits=total_splits
    )
    assert counts.dtype == torch.int32 and int(counts.max()) <= max_split
    assert int(counts.min()) >= 1
    # No empty split: ceil(t / ceil(t / s)) == s for t KV tiles of 128.
    t = (seq_lens + 127) // 128
    assert torch.equal(-(-t // -(-t // counts)), counts.long())
    if total_splits:
        assert int(counts.sum()) <= total_splits + len(lens)
    var = tokenspeed_mla.tokenspeed_mla_decode(
        **common, max_split_kv=max_split, block_split_kvs=counts
    )
    ref = _ref(q, kv, block_table, seq_lens, scale)
    torch.testing.assert_close(var.float(), ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(var.float(), base.float(), atol=2e-2, rtol=2e-2)
