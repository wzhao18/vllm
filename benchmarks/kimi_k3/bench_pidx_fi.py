# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer FP8MQATopKWrapper (selective prefill top-k, MXFP4, native_bf16) at the
P3-16 per-GPU shape (3 x 5461 rows at 78360 keys) vs stock DeepGEMM logits.

One wrapper per request (the selective path is batch-1); per DSA layer = the three
run() graphs. --check compares the selected keys with the exact top-2048 of the
stock (DeepGEMM) logits per row: identical index sets, identical value multisets.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_pidx_fused import make, order16, stock_logits  # noqa: E402

import flashinfer.experimental.attn_scores._topk_runtime as _fi_rt  # noqa: E402
from flashinfer.attn_scores.mqa_topk import FP8MQATopKWrapper  # noqa: E402

# FlashInfer gates the selective path to SM100/SM103; its CuTe DSL kernels compile
# for the device arch (sm_107a here) unchanged.
_fi_rt._validate_topk_device = lambda device: None


def force_blackwell_tiles():
    """FlashInfer's Rubin (N=256) kernel path is not wired on the host side
    (weights staged for block_q=4); run its SM100 N=128 tiles instead."""
    import importlib
    import pkgutil

    import flashinfer.experimental.attn_scores.kernels.selective_topk.prefill as pre
    for m in pkgutil.iter_modules(pre.__path__):
        try:
            mod = importlib.import_module(f"{pre.__name__}.{m.name}")
        except Exception:
            continue
        if hasattr(mod, "_target_is_rubin"):
            mod._target_is_rubin = lambda: False


K = 2048


def wrappers(d, math_mode, wdt=torch.bfloat16):
    t0, out = 0, []
    for r, (rows, kv) in enumerate(d["reqs"]):
        b = d["bases"][r]
        nk = (kv + 255) // 256 * 256
        keys = torch.zeros(nk, 64, dtype=torch.uint8, device="cuda")
        sf = torch.full((nk, 4), 127, dtype=torch.uint8, device="cuda")
        keys[:kv].copy_(d["k8"][b:b + kv])
        sf[:kv].copy_(d["k_sf"][b:b + kv].view(torch.uint8).view(kv, 4))
        cap = (rows + 7) // 8 * 8
        w = FP8MQATopKWrapper(
            backend="cutedsl", num_heads=32, head_dim=128, top_k=K,
            math_mode=math_mode, weights_dtype=wdt,
            compression_ratio=1, max_context_length=kv, max_chunk_size=cap,
            max_batch_size=1, device=torch.device("cuda"), num_keys=nk,
            qkv_dtype="mxfp4", logits_dtype=torch.bfloat16,
        )
        st = w.input_buffers()
        q = d["q"][t0:t0 + rows].view(torch.uint8).reshape(rows, 32, 64)
        st["q"][:rows].copy_(q.view(st["q"].dtype))
        qs = d["q_scale"][t0:t0 + rows].contiguous().view(torch.uint8).reshape(-1)
        qs_dst = st["q_scale"]
        qs_dst.view(torch.uint8).reshape(cap, -1)[:rows].copy_(qs.view(rows, -1))
        st["weights"][:rows].copy_(d["w"][t0:t0 + rows, :32])
        st["qo_indptr"].copy_(torch.tensor([0, rows], dtype=torch.int32))
        st["kv_indptr"].copy_(torch.tensor([0, kv], dtype=torch.int32))
        w.prepare_pool(keys, k_scale=sf)
        w.plan(st["qo_indptr"], st["kv_indptr"])
        call = (lambda w=w, st=st, keys=keys, sf=sf: w.run(
            st["q"], st["weights"], keys, k_scale=sf, q_scale=st["q_scale"]))
        out.append(dict(w=w, st=st, call=call, rows=rows, kv=kv, t0=t0))
        t0 += rows
    return out


def check(d, ws):
    res = {"rows": 0, "set_equal": 0, "value_equal": 0, "short": 0, "max_value_miss": 0}
    for r, x in enumerate(ws):
        got = x["call"]()
        torch.cuda.synchronize()
        e, re_ = int(x["w"].error.item()), int(x["w"].repair_error.item())
        res[f"err{r}"] = [e, re_]
        rows, kv = x["rows"], x["kv"]
        sl = order16(stock_logits(d, r).view(torch.int16)).to(torch.int32)
        ar = torch.arange(kv, device="cuda")
        for c in range(0, rows, 512):
            n = min(512, rows - c)
            ke = kv - rows + 1 + c + torch.arange(n, device="cuda")
            key = torch.where(ar[None, :] < ke[:, None], sl[c:c + n], -1)
            kk = torch.minimum(ke, torch.tensor(K, device="cuda"))
            ref_v = torch.topk(key, K, dim=1).values  # desc
            ref_i = torch.topk(key, K, dim=1).indices
            g = got[c:c + n, :K].to(torch.int64)
            valid = g >= 0
            res["short"] += int((valid.sum(1) != kk).sum())
            gv = torch.where(valid, torch.gather(key, 1, g.clamp(min=0)), -2)
            gv = torch.sort(gv, dim=1, descending=True).values
            slot = torch.arange(K, device="cuda")[None, :] < kk[:, None]
            veq = ((gv == ref_v) | ~slot).all(1)
            res["value_equal"] += int(veq.sum())
            miss = ((gv != ref_v) & slot).sum(1)
            res["max_value_miss"] = max(res["max_value_miss"], int(miss.max()))
            gs = torch.sort(torch.where(valid, g, 1 << 40), dim=1).values
            rs = torch.sort(torch.where(slot, ref_i, 1 << 40), dim=1).values
            res["set_equal"] += int((gs == rs).all(1).sum())
        res["rows"] += rows
        del sl
    print(json.dumps(res), flush=True)
    return res


def perf(ws, iters):
    gs = []
    for x in ws:
        x["call"]()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            x["call"]()
        pg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(pg):
            x["w"].plan(x["st"]["qo_indptr"], x["st"]["kv_indptr"])
        gs.append((g, pg))
    def layer():
        for g, _ in gs:
            g.replay()
    def step():
        for _, pg in gs:
            pg.replay()
    res = {}
    for name, fn in (("layer", layer), ("plan", step)):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        res[name] = s.elapsed_time(e) / iters
        print(f"FI selective {name}: {res[name]:.3f} ms", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--perf", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--math-mode", default="native_bf16")
    ap.add_argument("--weights-dtype", default="float32")
    ap.add_argument("--rubin", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    d = make([(5461, 78360), (5461, 78360), (5462, 78360)], seed=0)
    print({k: (tuple(v.shape), str(v.dtype)) for k, v in d.items()
           if isinstance(v, torch.Tensor) and k in ("q", "q_scale", "w", "k8", "k_sf")},
          flush=True)
    if not a.rubin:
        force_blackwell_tiles()
    ws = wrappers(d, a.math_mode, getattr(torch, a.weights_dtype))
    print("strategy", ws[0]["w"].strategy, "math", ws[0]["w"].math_mode,
          {k: (tuple(v.shape), str(v.dtype)) for k, v in ws[0]["st"].items()
           if isinstance(v, torch.Tensor)}, flush=True)
    out = {}
    if a.perf:
        out["perf"] = perf(ws, a.iters)
    if a.check:
        out["check"] = check(d, ws)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
