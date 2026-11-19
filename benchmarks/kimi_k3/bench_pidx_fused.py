# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_107 prefill indexer (specialized_prefill.indexer: IndexerScore + PrefillSelect)
vs stock.

--check: small shapes; logits vs DeepGEMM (bitwise share, ulp) and the exact top-k
(ties to the lowest key index) vs a torch reference on the kernel's own logits.
--perf: P3-16 per-GPU shape (3 x 5461 rows at 78360 keys) per DSA layer.
"""

import argparse
import json

import torch

from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import mxfp4_q_quant
from vllm.models.kimi_k3.nvidia.specialized_prefill import indexer as pidx
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    get_mqa_logits_metadata,
    mqa_logits_mx_inputs,
)


def make(reqs, seed=0, dev="cuda", zero_rows=None):
    """reqs: list of (rows, kv_len); zero_rows: token rows whose head weights are 0
    (all logits tie)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    t = sum(r for r, _ in reqs)
    index_q = torch.randn(t, 32, 128, device=dev, dtype=torch.bfloat16, generator=g)
    index_w = torch.randn(t, 32, device=dev, dtype=torch.float32, generator=g)
    q, q_scale, w = mxfp4_q_quant(index_q, index_w, (128**-0.5) * (32**-0.5), 32)
    if zero_rows is not None:
        w[zero_rows] = 0
    _, w_bf16 = mqa_logits_mx_inputs(q.view(torch.int8), q_scale, w)
    pads = [(kv + 127) // 128 * 128 for _, kv in reqs]
    nk = sum(pads)
    k8 = torch.randint(0, 256, (nk, 64), device=dev, dtype=torch.uint8, generator=g)
    ks = torch.randint(122, 130, (nk, 4), device=dev, dtype=torch.uint8, generator=g)
    k_sf = ks.view(torch.int32).view(nk)
    bases = [sum(pads[:i]) for i in range(len(reqs))]
    items, row_len, row_base, scratch = pidx.plan(
        [r for r, _ in reqs], [kv for _, kv in reqs], bases, dev
    )
    logits = torch.empty(scratch, dtype=torch.bfloat16, device=dev)
    row_bin = torch.zeros(2 * t, dtype=torch.int32, device=dev)
    out = torch.empty(t, 2048, dtype=torch.int32, device=dev)
    return dict(reqs=reqs, q=q, q_scale=q_scale, w=w_bf16, k8=k8, k_sf=k_sf,
                bases=bases, items=items, row_len=row_len, row_base=row_base,
                logits=logits, row_bin=row_bin, out=out)


def run_score(d, mode=0):
    pidx.score(d["q"], d["q_scale"], d["k8"], d["k_sf"], d["w"], d["row_len"],
               d["items"], d["row_base"], d["logits"], d["row_bin"], mode=mode)


def run_select(d):
    pidx.select(d["logits"], d["row_base"], d["row_len"], d["row_bin"], d["out"])


def stock_logits(d, r):
    rows, kv = d["reqs"][r]
    t0 = sum(x for x, _ in d["reqs"][:r])
    b = d["bases"][r]
    ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ke = d["row_len"][t0:t0 + rows].contiguous()
    meta = get_mqa_logits_metadata(ks, ke, kv, 32)
    logits = fp8_fp4_mqa_logits(
        (d["q"][t0:t0 + rows].view(torch.int8), d["q_scale"][t0:t0 + rows]),
        (d["k8"][b:b + kv].view(torch.int8), d["k_sf"][b:b + kv]),
        d["w"][t0:t0 + rows], ks, ke, kv, schedule_meta=meta,
    )
    return logits[:, :kv]


def order16(bits):
    b = bits.to(torch.int32) & 0xFFFF
    b = torch.where(b == 0x8000, torch.zeros_like(b), b)
    return torch.where(b & 0x8000 != 0, b ^ 0xFFFF, b ^ 0x8000)


def position(idx):
    """Logits position of key idx (tile rows in SF-atom order)."""
    return (idx & ~127) + 32 * (idx & 3) + ((idx & 127) >> 2)


def ref_topk(bits_row, n, k):
    """Exact top-k; ties to the lowest logits position."""
    key = order16(bits_row[:n]).to(torch.int64)
    idx = torch.arange(n, device=bits_row.device, dtype=torch.int64)
    comp = key * (1 << 20) + ((1 << 20) - 1 - position(idx))
    return torch.sort(idx[torch.topk(comp, min(k, n)).indices]).values


def check(args):
    # request 2 ties completely (candidate path with ties), the first 64 rows of
    # request 3 overflow the candidate buffer (exact path), request 5 is longer than
    # the vector-maxima buffer (exact path)
    reqs = [(37, 1000), (300, 5000), (9, 2100), (513, 9000), (40, 40000), (3, 140000)]
    zero = list(range(337, 346)) + list(range(346, 346 + 64))
    d = make(reqs, seed=1, zero_rows=zero)
    run_score(d)
    run_select(d)
    torch.cuda.synchronize()
    lg = d["logits"].view(torch.int16)
    res = {"rows": d["q"].shape[0], "bad_rows": 0, "bitwise": [], "max_ulp": 0}
    t0 = 0
    for r, (rows, kv) in enumerate(reqs):
        sl = stock_logits(d, r).view(torch.int16)
        pitch = (kv + 127) // 128 * 128
        base = int(d["row_base"][t0])
        pos = position(torch.arange(kv, device="cuda"))
        mine = lg[base:base + rows * pitch].view(rows, pitch)[:, pos]
        rl = d["row_len"][t0:t0 + rows]
        mask = torch.arange(kv, device="cuda")[None, :] < rl[:, None]
        res["bitwise"].append(float(((sl == mine) | ~mask).float().mean()))
        ulp = (order16(sl).to(torch.int64) - order16(mine).to(torch.int64)).abs()
        res["max_ulp"] = max(res["max_ulp"], int(torch.where(mask, ulp, 0).max()))
        for i in range(rows):
            n = int(rl[i])
            ref = ref_topk(mine[i], n, 2048)
            got = d["out"][t0 + i]
            got = torch.sort(got[got >= 0].to(torch.int64)).values
            if got.numel() != ref.numel() or not torch.equal(got, ref):
                res["bad_rows"] += 1
                if res["bad_rows"] <= 3:
                    print(f"row {t0 + i} n={n} got={got.numel()} ref={ref.numel()} "
                          f"miss={len(set(ref.tolist()) - set(got.tolist()))}")
        t0 += rows
    print(json.dumps(res), flush=True)
    return res


def timed(fn, iters):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def perf(args):
    d = make([(5461, 78360), (5461, 78360), (5462, 78360)], seed=0)
    res = {}
    for rep in range(args.reps):
        for mode in args.modes:
            res[f"score_mode{mode}"] = timed(lambda: run_score(d, mode), args.iters)
            print(f"score mode {mode}: {res[f'score_mode{mode}']:.3f} ms", flush=True)
    res["fill"] = timed(lambda: d["logits"].fill_(0), args.iters)
    print(f"torch fill of the scratch: {res['fill']:.3f} ms", flush=True)
    run_score(d)
    res["select"] = timed(lambda: run_select(d), args.iters)
    for groups in args.groups:
        for unroll in args.unrolls:
            for smode in args.smodes:
                kw = dict(groups=groups, unroll=unroll, mode=smode)
                tag = f"select_g{groups}_u{unroll}_m{smode}"
                res[tag] = timed(lambda: pidx.select(
                    d["logits"], d["row_base"], d["row_len"], d["row_bin"], d["out"], **kw),
                    args.iters)
                dbg = torch.zeros(8, dtype=torch.int64, device="cuda")
                pidx.select(d["logits"], d["row_base"], d["row_len"], d["row_bin"], d["out"],
                            dbg=dbg, **kw)
                torch.cuda.synchronize()
                v = dbg.tolist()
                n = max(v[0], 1)
                print(f"{tag}: {res[tag]:.3f} ms; clk per row: zero {v[1] / n:.0f} stream "
                      f"{v[2] / n:.0f} digit {v[3] / n:.0f} resolve {v[4] / n:.0f} emit "
                      f"{v[5] / n:.0f} (stream load-wait {v[6] / n:.0f}, dense {v[7] / n:.0f})", flush=True)
    res["score+select"] = timed(lambda: (run_score(d), run_select(d)), args.iters)
    print(f"select {res['select']:.3f} ms; score+select {res['score+select']:.3f} ms "
          f"per layer (items {d['items'].shape[0]}, scratch "
          f"{d['logits'].numel() * 2 / 2**30:.2f} GiB)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--perf", action="store_true")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--modes", type=int, nargs="+", default=[0])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--groups", type=int, nargs="+", default=[4])
    ap.add_argument("--unrolls", type=int, nargs="+", default=[8])
    ap.add_argument("--smodes", type=int, nargs="+", default=[0])
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
