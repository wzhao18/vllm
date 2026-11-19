# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P3-16 prefill DSA indexer microbench (one GPU, one DSA layer).

Per-GPU shape of DEP16 F16K at L78K: 3 requests x ~5461 fresh rows at KV length
78360, 32 index heads x 128 dim, MXFP4 Q and K, top-2048. Emulates the stock
chunk loop of ``sparse_attn_indexer`` (VLLM_SPARSE_INDEXER_MAX_LOGITS_MB chunks
per request): DeepGEMM ``fp8_fp4_mqa_logits`` (BF16 logits) + the configured
prefill top-k (trtllm_gvr2 by default). Prints per-layer time and a kernel
breakdown from the torch profiler.
"""

import argparse
import json

import torch

from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import mxfp4_q_quant
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    get_mqa_logits_metadata,
    mqa_logits_mx_inputs,
)


def make_inputs(nreq, rows, kv_len, heads=32, dim=128, seed=0, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    t = nreq * rows
    index_q = torch.randn(t, heads, dim, device=dev, dtype=torch.bfloat16, generator=g)
    index_w = torch.randn(t, heads, device=dev, dtype=torch.float32, generator=g)
    q, q_scale, w = mxfp4_q_quant(index_q, index_w, (128**-0.5) * (heads**-0.5), heads)
    # MXFP4 keys: random e2m1 nibbles, ue8m0 scales near 2^0.
    keys = []
    for _ in range(nreq):
        kq = torch.randint(0, 256, (kv_len, dim // 2), device=dev, dtype=torch.uint8,
                           generator=g)
        ks = torch.randint(122, 130, (kv_len, dim // 32), device=dev, dtype=torch.uint8,
                           generator=g)
        keys.append((kq, ks))
    return q, q_scale, w, keys


def chunk_plan(rows, kv_len, max_logits_mb):
    max_q = max(1, (max_logits_mb * 1024 * 1024 // 4) // kv_len)
    return [(o, min(max_q, rows - o)) for o in range(0, rows, max_q)]


def stock_layer(state, topk_tokens=2048):
    """One DSA layer of the stock prefill indexer on this GPU's rows."""
    from vllm.model_executor.layers.indexer_topk import get_indexer_topk

    topk = get_indexer_topk(state["backend"])
    out = state["topk_out"]
    for (r, off, n, ks, ke, meta, rlen) in state["chunks"]:
        q, q_scale, w = state["q"], state["q_scale"], state["w"]
        kq, kscale = state["keys"][r]
        base = r * state["rows"] + off
        q_sl = q[base:base + n].view(torch.int8)
        qs_sl = q_scale[base:base + n]
        qs_sl, w_sl = mqa_logits_mx_inputs(q_sl, qs_sl, w[base:base + n])
        logits = fp8_fp4_mqa_logits(
            (q_sl, qs_sl),
            (kq.view(torch.int8), kscale.view(torch.int32).squeeze(-1)),
            w_sl, ks, ke, state["kv_len"], schedule_meta=meta,
        )
        topk.forward_prefill(logits, None, rlen, out[base:base + n], topk_tokens)
        state["last_logits"] = logits
    return out


def build_state(args):
    dev = "cuda"
    q, q_scale, w, keys = make_inputs(args.nreq, args.rows, args.kv_len)
    prefix = args.kv_len - args.rows
    chunks = []
    for r in range(args.nreq):
        for off, n in chunk_plan(args.rows, args.kv_len, args.max_logits_mb):
            ks = torch.zeros(n, dtype=torch.int32, device=dev)
            ke = (prefix + off + 1 + torch.arange(n, device=dev, dtype=torch.int32))
            meta = get_mqa_logits_metadata(ks, ke, args.kv_len, 32) if args.schedule else None
            chunks.append((r, off, n, ks, ke, meta, ke - ks))
    out = torch.empty(args.nreq * args.rows, 2048, dtype=torch.int32, device=dev)
    return dict(q=q, q_scale=q_scale, w=w, keys=keys, chunks=chunks, rows=args.rows,
                kv_len=args.kv_len, topk_out=out, backend=args.backend)


def timeit(fn, iters, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def profile(fn, iters):
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as tprof

    with tprof(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    agg = {}
    for ev in p.events():
        if ev.device_type.name != "CUDA":
            continue
        k = ev.name[:110]
        d = agg.setdefault(k, [0, 0.0])
        d[0] += 1
        d[1] += ev.device_time_total if hasattr(ev, "device_time_total") else ev.cuda_time_total
    rows = sorted(agg.items(), key=lambda kv: -kv[1][1])
    return [(k, c / iters, us / iters) for k, (c, us) in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nreq", type=int, default=3)
    ap.add_argument("--rows", type=int, default=5461)
    ap.add_argument("--kv-len", type=int, default=78360)
    ap.add_argument("--max-logits-mb", type=int, default=512)
    ap.add_argument("--backend", default="trtllm_gvr2")
    ap.add_argument("--schedule", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    torch.cuda.set_device(0)
    st = build_state(args)
    res = {"args": vars(args), "nchunks": len(st["chunks"])}
    res["stock_ms"] = timeit(lambda: stock_layer(st), args.iters)
    print(f"stock per-layer {res['stock_ms']:.3f} ms ({len(st['chunks'])} chunks)", flush=True)
    prof = profile(lambda: stock_layer(st), 2)
    res["kernels"] = prof
    for k, c, us in prof[:15]:
        print(f"  {us/1000:8.3f} ms  x{c:5.1f}  {k}")
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
