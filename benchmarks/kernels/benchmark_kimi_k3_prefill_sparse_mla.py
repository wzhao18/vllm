# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 prefill sparse MLA block at DEP16 (TP1, 96 heads per GPU), one GPU.

Times the stock DCP-1 chain of one DSA layer at a prefill chunk of F fresh tokens
(3 requests at L78K context) kernel by kernel, as forward_impl runs it:
  w_uk   bmm (N,B,P)x(N,P,L) into the token-major [B,N,576] query + RoPE copy
  quant  scaled_fp8_quant (static per-tensor) of the [B,N*576] query
  pad    _pad_query_heads 96 -> 128 (trtllm-gen runs 128 heads)
  attn   trtllm-gen sparse MLA (FP8 Q, FP8 KV 576 B rows, top-2048)
  w_uv   bmm (N,B,L)x(N,L,V) from the head-padded output view
  gate   attn * sigmoid(gate) (prefill_glue.eager_gate_sigmoid_mul)
  insert concat_and_cache_mla (FP8 per-tensor) of the F new KV rows
and, where the installed FlashInfer has it, the prims_ts sparse MLA at 96 and 128
heads. Each layer instance uses its own KV pool / indices (rotated over --layers
copies) so the attention does not run from a warm L2.

usage: python benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 [--json out]
"""

import argparse
import json
import math
import statistics

import torch

H, P, L, R, V = 96, 128, 512, 64, 128
TOPK = 2048


def make_layer(F, ctx, nreq, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    per = F // nreq
    fresh = [per] * (nreq - 1) + [F - per * (nreq - 1)]
    page = 64
    seq_total = [ctx + f for f in fresh]
    pages = [math.ceil(s / page) for s in seq_total]
    perm = torch.randperm(sum(pages), device=dev, generator=g)  # paged, shuffled
    kv = (torch.randn(sum(pages), page, 576, device=dev, generator=g) * 0.5).to(
        torch.float8_e4m3fn
    )
    idx = torch.empty(F, TOPK, dtype=torch.int32, device=dev)
    row = 0
    base = 0
    for r in range(nreq):
        bt = perm[base : base + pages[r]].to(torch.int64)
        base += pages[r]
        n = fresh[r]
        pos = ctx + torch.arange(n, device=dev)  # causal: key <= pos
        for c0 in range(0, n, 512):
            c1 = min(n, c0 + 512)
            lim = pos[c0:c1] + 1
            keys = torch.rand(c1 - c0, ctx + n, device=dev, generator=g)
            keys = keys.masked_fill(
                torch.arange(ctx + n, device=dev)[None] >= lim[:, None], -1.0
            )
            logical = keys.topk(TOPK, dim=1).indices.sort(dim=1).values
            phys = bt[logical // page] * page + logical % page
            idx[row + c0 : row + c1] = phys.to(torch.int32)
        row += n
    return kv, idx


def time_stage(fn, iters):
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        s[i].record()
        fn(i)
        e[i].record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) * 1e3 for a, b in zip(s, e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--ctx", type=int, default=72900)
    ap.add_argument("--reqs", type=int, default=3)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--loaders", default="3,5")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    F = a.tokens
    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.ops import prefill_glue
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        _pad_query_heads,
    )

    layers = [make_layer(F, a.ctx, a.reqs, dev, 17 + i) for i in range(a.layers)]
    q_in = [torch.randn(F, H, P + R, device=dev, dtype=torch.bfloat16) for _ in range(2)]
    W_UK_T = torch.randn(H, P, L, device=dev, dtype=torch.bfloat16) * 0.05
    W_UV = torch.randn(H, L, V, device=dev, dtype=torch.bfloat16) * 0.05
    gate = torch.randn(F, H * V, device=dev, dtype=torch.bfloat16)
    q_scale = torch.tensor([0.05], device=dev, dtype=torch.float32)
    k_scale = torch.tensor([1.0], device=dev, dtype=torch.float32)
    ws = torch.zeros(1 << 30, dtype=torch.int8, device=dev)
    seq = torch.full((F,), TOPK, dtype=torch.int32, device=dev)
    from flashinfer.utils import (
        get_device_sm_count,
        get_trtllm_gen_multi_ctas_kv_counter_bytes,
    )

    counters = torch.zeros(
        max(16, get_trtllm_gen_multi_ctas_kv_counter_bytes(F, 128, get_device_sm_count(dev))),
        dtype=torch.uint8,
        device=dev,
    )
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    res = {"tokens": F, "ctx": a.ctx, "reqs": a.reqs, "dev": torch.cuda.get_device_name()}
    st = {}

    qcat = torch.empty(F, H, L + R, device=dev, dtype=torch.bfloat16)

    def w_uk(i):
        q = q_in[i % 2]
        qn = q[..., :P].transpose(0, 1)
        torch.bmm(qn, W_UK_T, out=qcat[..., :L].transpose(0, 1))
        qcat[..., L:].copy_(q[..., P:])

    st["w_uk"] = time_stage(w_uk, a.iters)
    qf = torch.empty(F, H * (L + R), device=dev, dtype=torch.float8_e4m3fn)

    def quant(i):
        ops.scaled_fp8_quant(qcat.view(F, -1), q_scale, output=qf)

    st["quant"] = time_stage(quant, a.iters)
    qf3 = qf.view(F, H, L + R)
    qpad = _pad_query_heads(qf3, 128)
    st["pad"] = time_stage(lambda i: _pad_query_heads(qf3, 128), a.iters)
    # Fused path (this branch): bmm into a [B,N,L] buffer, then one kernel for
    # concat + static FP8 quant + head pad (decode_query_quant_pad).
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        decode_query_quant_pad,
    )

    qln = torch.empty(F, H, L, device=dev, dtype=torch.bfloat16)

    def w_uk_only(i):
        q = q_in[i % 2]
        torch.bmm(q[..., :P].transpose(0, 1), W_UK_T, out=qln.transpose(0, 1))

    st["fused_w_uk"] = time_stage(w_uk_only, a.iters)
    st["fused_quant_pad"] = time_stage(
        lambda i: decode_query_quant_pad(qln, q_in[i % 2][..., P:], q_scale, 128),
        a.iters,
    )
    st["fused_quant_nopad"] = time_stage(
        lambda i: decode_query_quant_pad(qln, q_in[i % 2][..., P:], q_scale, H),
        a.iters,
    )
    out = torch.empty(F, 128, L, device=dev, dtype=torch.bfloat16)

    def attn(i):
        kv, idx = layers[i % len(layers)]
        for s0 in range(0, F, 65535):
            s1 = min(F, s0 + 65535)
            trtllm_batch_decode_with_kv_cache_mla(
                query=qpad[s0:s1].unsqueeze(1),
                kv_cache=kv.unsqueeze(1),
                workspace_buffer=ws,
                qk_nope_head_dim=128,
                kv_lora_rank=L,
                qk_rope_head_dim=R,
                block_tables=idx[s0:s1].unsqueeze(1),
                seq_lens=seq[s0:s1],
                max_seq_len=TOPK,
                out=out[s0:s1].unsqueeze(1),
                bmm1_scale=0.0722,
                bmm2_scale=1.0,
                sparse_mla_top_k=TOPK,
                multi_ctas_kv_counter_buffer=counters,
            )

    attn(0)
    torch.cuda.synchronize()
    res["attn_finite"] = bool(torch.isfinite(out[:, :H].float()).all())
    st["attn_trtllm"] = time_stage(attn, a.iters)
    # specialized_prefill sparse MLA (sm_107, native 96 heads, no padded query)
    try:
        from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla import (
            sparse_mla_prefill,
        )

        vr_out = torch.empty(F, H, L, device=dev, dtype=torch.bfloat16)
        for var in a.loaders.split(","):
            nl, pp = (int(v) for v in (var.split(":") + ["0"])[:2])
            tag = f"l{nl}p{pp}"

            def vr(i, nl=nl, pp=pp):
                kv, idx = layers[i % len(layers)]
                sparse_mla_prefill(qf3, kv.view(-1, 576), 64, idx, seq, 0.0722, 1.0,
                                   out=vr_out, loaders=nl, pipelined=bool(pp))

            vr(0)
            torch.cuda.synchronize()
            d = (vr_out.float() - out[:, :H].float()).norm(dim=-1)
            rel = (d / out[:, :H].float().norm(dim=-1).clamp(min=1e-6)).mean().item()
            res[f"vr_{tag}_rel_vs_trtllm"] = rel
            st[f"attn_vr_{tag}"] = time_stage(vr, a.iters)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        res["vr_error"] = repr(e)[:400]
    o = torch.empty(F, H * V, device=dev, dtype=torch.bfloat16)

    def w_uv(i):
        x = out[:, :H].transpose(0, 1)
        torch.bmm(x, W_UV, out=o.view(F, H, V).transpose(0, 1))

    st["w_uv"] = time_stage(w_uv, a.iters)
    st["gate"] = time_stage(lambda i: prefill_glue.eager_gate_sigmoid_mul(o, gate), a.iters)
    kvc = torch.randn(F, L, device=dev, dtype=torch.bfloat16)
    kpe = torch.randn(F, R, device=dev, dtype=torch.bfloat16)
    cache = torch.zeros(4096, 64, 576, device=dev, dtype=torch.uint8)
    slots = torch.randperm(4096 * 64, device=dev)[:F].to(torch.int64)
    st["insert"] = time_stage(
        lambda i: ops.concat_and_cache_mla(kvc, kpe, cache, slots, "fp8", k_scale),
        a.iters,
    )

    # prims_ts (FlashInfer task-scheduled sparse MLA), if present
    try:
        from flashinfer.attention.prims_ts import (
            BatchSparseMLADecodePagedTSWrapper,
            get_prims_ts_sparse_mla_decode_workspace_size,
        )

        for heads in (96, 128):
            rows = 4096
            kw = dict(
                max_topk=TOPK,
                max_q_len=1,
                packed_query=True,
                q_dtype=torch.float8_e4m3fn,
                qk_rope_head_dim=64,
                kv_cache_format="native",
                return_lse=False,
            )
            size = get_prims_ts_sparse_mla_decode_workspace_size(dev, rows, heads, **kw)
            pws = torch.empty(size, dtype=torch.uint8, device=dev)
            plan = BatchSparseMLADecodePagedTSWrapper(pws)
            plan.plan(dev, rows, heads, **kw)
            qo = torch.arange(rows + 1, dtype=torch.int32, device=dev)
            qh = qpad[:, :heads].contiguous()
            po = torch.empty(F, heads, L, device=dev, dtype=torch.bfloat16)

            def pts(i, qh=qh, po=po, plan=plan, qo=qo):
                kv, idx = layers[i % len(layers)]
                rowsv = kv.view(-1, 576)
                for s0 in range(0, F, rows):
                    s1 = min(F, s0 + rows)
                    plan.run(
                        qh[s0:s1],
                        rowsv,
                        idx[s0:s1],
                        seq[s0:s1],
                        qo_indptr=qo,
                        softmax_scale=0.0722,
                        output_scale=1.0,
                        out=po[s0:s1],
                        lse=None,
                        validate=False,
                    )

            pts(0)
            torch.cuda.synchronize()
            st[f"attn_prims_ts_h{heads}"] = time_stage(pts, a.iters)
            d = (po[:, :H].float() - out[:, :H].float()).abs().max().item()
            res[f"prims_ts_h{heads}_maxdiff_vs_trtllm"] = d
    except Exception as e:  # noqa: BLE001
        res["prims_ts_error"] = repr(e)[:300]

    res["stages_us"] = {k: round(v, 1) for k, v in st.items()}
    fl = 2.0 * F * H * TOPK * (576 + 512)
    res["attn_tflops_real_heads"] = {
        k: round(fl / v / 1e6, 0) for k, v in st.items() if k.startswith("attn")
    }
    stock = ["w_uk", "quant", "pad", "attn_trtllm", "w_uv", "gate", "insert"]
    res["stock_chain_us"] = round(sum(st[k] for k in stock), 1)
    fused = ["fused_w_uk", "fused_quant_pad", "attn_trtllm", "w_uv", "gate", "insert"]
    res["fused_query_chain_us"] = round(sum(st[k] for k in fused), 1)
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
