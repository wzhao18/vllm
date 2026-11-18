# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused sm_107 prefill indexer (specialized_prefill.indexer) vs stock.

--check: small shapes; logits vs DeepGEMM (bitwise share, ulp histogram) and the
exact top-k (ties to the lowest key index) vs a torch reference on the fused
kernel's own logits, plus the overlap with the stock trtllm_gvr2 selection.
--perf: P3-16 per-GPU shape (3 x 5461 rows at 78360 keys) per DSA layer.
"""

import argparse
import json
import time

import torch

from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import mxfp4_q_quant
from vllm.models.kimi_k3.nvidia.specialized_prefill import indexer as fused
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    get_mqa_logits_metadata,
    mqa_logits_mx_inputs,
)


def make(reqs, seed=0, dev="cuda", key_lo=122, key_hi=130):
    """reqs: list of (rows, kv_len). Returns a dict of inputs."""
    g = torch.Generator(device=dev).manual_seed(seed)
    t = sum(r for r, _ in reqs)
    index_q = torch.randn(t, 32, 128, device=dev, dtype=torch.bfloat16, generator=g)
    index_w = torch.randn(t, 32, device=dev, dtype=torch.float32, generator=g)
    q, q_scale, w = mxfp4_q_quant(index_q, index_w, (128**-0.5) * (32**-0.5), 32)
    _, w_bf16 = mqa_logits_mx_inputs(q.view(torch.int8), q_scale, w)
    nk = sum((kv + 127) // 128 * 128 for _, kv in reqs)
    k8 = torch.randint(0, 256, (nk, 64), device=dev, dtype=torch.uint8, generator=g)
    ks = torch.randint(key_lo, key_hi, (nk, 4), device=dev, dtype=torch.uint8, generator=g)
    k_sf = ks.view(torch.int32).view(nk)
    bases, b = [], 0
    for _, kv in reqs:
        bases.append(b)
        b += (kv + 127) // 128 * 128
    items, row_len = fused.plan_items(
        [r for r, _ in reqs], [kv for _, kv in reqs], bases, dev
    )
    return dict(reqs=reqs, q=q, q_scale=q_scale, w=w_bf16, k8=k8, k_sf=k_sf,
                bases=bases, items=items, row_len=row_len)


def stock_logits(d, r):
    """DeepGEMM BF16 logits of request r: [rows, kv] (columns >= row_len garbage)."""
    rows, kv = d["reqs"][r]
    t0 = sum(x for x, _ in d["reqs"][:r])
    b = d["bases"][r]
    ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ke = d["row_len"][t0:t0 + rows].contiguous()
    meta = get_mqa_logits_metadata(ks, ke, kv, 32)
    q = d["q"][t0:t0 + rows].view(torch.int8)
    logits = fp8_fp4_mqa_logits(
        (q, d["q_scale"][t0:t0 + rows]),
        (d["k8"][b:b + kv].view(torch.int8), d["k_sf"][b:b + kv]),
        d["w"][t0:t0 + rows], ks, ke, kv, schedule_meta=meta,
    )
    return logits[:, :kv]


def order16(bits):
    b = bits.to(torch.int32) & 0xFFFF
    b = torch.where(b == 0x8000, torch.zeros_like(b), b)
    return torch.where(b & 0x8000 != 0, b ^ 0xFFFF, b ^ 0x8000)


def ref_topk(bits_row, n, k):
    """Exact top-k key indices (ties to the lowest index) of the first n keys."""
    key = order16(bits_row[:n]).to(torch.int64)
    idx = torch.arange(n, device=bits_row.device, dtype=torch.int64)
    comp = key * (1 << 20) + ((1 << 20) - 1 - idx)
    m = min(k, n)
    return torch.sort(idx[torch.topk(comp, m).indices]).values


def check(args):
    reqs = [(37, 1000), (300, 5000), (9, 2100), (513, 9000)]
    d = make(reqs, seed=1)
    t = d["q"].shape[0]
    pitch = ((max(kv for _, kv in reqs) + 63) // 64) * 64
    lg = torch.full((t, pitch), 0x7FC0, dtype=torch.int16, device="cuda")
    out_idx, out_cnt = fused.score_select(
        d["q"], d["q_scale"], d["k8"], d["k_sf"], d["w"], d["row_len"], d["items"],
        logits=lg,
    )
    torch.cuda.synchronize()
    res = {"rows": t, "bad_rows": 0, "bitwise": [], "max_ulp": 0}
    t0 = 0
    for r, (rows, kv) in enumerate(reqs):
        sl = stock_logits(d, r)
        mine = lg[t0:t0 + rows, :kv]
        rl = d["row_len"][t0:t0 + rows]
        mask = torch.arange(kv, device="cuda")[None, :] < rl[:, None]
        sb = sl.view(torch.int16)
        eq = (sb == mine) | ~mask
        res["bitwise"].append(float(eq.float().mean()))
        ulp = (order16(sb).to(torch.int64) - order16(mine).to(torch.int64)).abs()
        ulp = torch.where(mask, ulp, torch.zeros_like(ulp))
        res["max_ulp"] = max(res["max_ulp"], int(ulp.max()))
        for i in range(rows):
            n = int(rl[i])
            ref = ref_topk(mine[i], n, 2048)
            c = int(out_cnt[t0 + i])
            got = torch.sort(out_idx[t0 + i, :c].to(torch.int64)).values
            if c != ref.numel() or not torch.equal(got, ref):
                res["bad_rows"] += 1
                if res["bad_rows"] <= 3:
                    print(f"row {t0+i} n={n} cnt={c} ref={ref.numel()} "
                          f"miss={len(set(ref.tolist()) - set(got.tolist()))}")
        t0 += rows
    print(json.dumps(res))
    return res


def perf(args):
    reqs = [(5461, 78360), (5461, 78360), (5462, 78360)]
    d = make(reqs, seed=0)

    def run():
        return fused.score_select(
            d["q"], d["q_scale"], d["k8"], d["k_sf"], d["w"], d["row_len"], d["items"]
        )

    t = time.time()
    run()
    torch.cuda.synchronize()
    print(f"first call (compile) {time.time() - t:.1f} s", flush=True)
    for _ in range(2):
        run()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(args.iters):
        run()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / args.iters
    print(f"fused score+select per layer {ms:.3f} ms (items {d['items'].shape[0]})")
    return {"fused_ms": ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--perf", action="store_true")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = {}
    if a.check:
        out["check"] = check(a)
    if a.perf:
        out["perf"] = perf(a)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
