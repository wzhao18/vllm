# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill draft-step pieces at the F16K slice shape (3 requests at L78K + 5461,
4 query rows each, release DSpark: 96 / 16 heads, head_dim 64, window 2048):
FlashAttention 2 vs specialized_prefill.draft_attention; context RoPE + K/V insert
(stock rotary_embedding + reshape_and_cache_flash vs prefill_glue.rope_store_kv);
DSA gate_sigmoid_mul at M=16384. CUDA-event times (us, median of 50)."""

import torch


def timed(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return sorted(ts)[len(ts) // 2]


def main():
    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.ops import prefill_glue
    from vllm.models.kimi_k3.nvidia.specialized_prefill import draft_attention
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    heads, kv_heads, hd, q_len, page = 96, 16, 64, 4, 16
    seq = 78360 + 5461 + q_len
    seq_lens = torch.full((3,), seq, dtype=torch.int32, device="cuda")
    max_pages = (seq + page - 1) // page
    blocks = 3 * max_pages + 8
    cache = torch.randn(blocks, kv_heads, page, 2 * hd, device="cuda").to(torch.bfloat16)
    kc, vc = cache.transpose(1, 2).split(hd, dim=-1)
    bt = torch.randperm(blocks, device="cuda")[: 3 * max_pages].view(3, -1).int()
    cu = torch.arange(0, 4 * q_len, q_len, device="cuda", dtype=torch.int32)
    q = torch.randn(3 * q_len, heads, hd, device="cuda").to(torch.bfloat16)
    win = (2047, 2047)
    out = torch.empty_like(q)
    fa = timed(lambda: flash_attn_varlen_func(
        q=q, k=kc, v=vc, cu_seqlens_q=cu, max_seqlen_q=q_len, seqused_k=seq_lens,
        max_seqlen_k=seq, softmax_scale=hd**-0.5, causal=False, window_size=list(win),
        block_table=bt, fa_version=2))  # fmt: skip
    da = timed(lambda: draft_attention.attention(
        q, kc, vc, bt, seq_lens, cu, q_len, hd**-0.5, win, out))  # fmt: skip
    print(f"draft attention: FA2 {fa:.1f} us, split-KV {da:.1f} us")

    rows = 6156
    k = torch.randn(rows, kv_heads, hd, device="cuda").to(torch.bfloat16)
    kv = torch.randn(rows, 5, 2, kv_heads, hd, device="cuda").to(torch.bfloat16)
    v = kv[:, 0, 1]
    pos = torch.randint(0, 80000, (rows,), device="cuda")
    cs = torch.randn(1 << 17, hd, device="cuda").to(torch.bfloat16)
    slots = torch.randperm(blocks * page, device="cuda")[:rows]
    scale = torch.ones((), device="cuda")

    def stock():
        kk = k.clone()
        ops.rotary_embedding(pos, kk.view(rows, -1), None, hd, cs, True)
        ops.reshape_and_cache_flash(kk, v, kc, vc, slots, "auto", scale, scale)

    st = timed(stock)
    fu = timed(lambda: prefill_glue.rope_store_kv(k, v, pos, cs, slots, kc, vc))
    print(f"context rope+insert per layer: stock (incl clone) {st:.1f} us, fused {fu:.1f} us")

    attn = torch.randn(16384, 12288, device="cuda").to(torch.bfloat16)
    gate = torch.randn(16384, 12288, device="cuda").to(torch.bfloat16)
    g = timed(lambda: prefill_glue.gate_sigmoid_mul(attn, gate))
    floor = 3 * attn.numel() * 2 / 13.5e12 * 1e6
    print(f"gate_sigmoid_mul 16384x12288: {g:.1f} us (13.5 TB/s floor {floor:.1f})")


if __name__ == "__main__":
    main()
