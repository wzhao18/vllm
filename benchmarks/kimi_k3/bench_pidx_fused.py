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
    row_bin = torch.zeros(3 * t, dtype=torch.int32, device=dev)
    words = max(pads) // 32
    bitmap = torch.zeros(t * words, dtype=torch.int32, device=dev)
    cand = torch.zeros(t * pidx.CAND, dtype=torch.int32, device=dev)
    out = torch.empty(t, 2048, dtype=torch.int32, device=dev)
    return dict(reqs=reqs, q=q, q_scale=q_scale, w=w_bf16, k8=k8, k_sf=k_sf,
                bases=bases, items=items, row_len=row_len, row_base=row_base,
                logits=logits, row_bin=row_bin, bitmap=bitmap, cand=cand, out=out)


def run_score(d, mode=0):
    pidx.score(d["q"], d["q_scale"], d["k8"], d["k_sf"], d["w"], d["row_len"],
               d["items"], d["row_base"], d["logits"], d["row_bin"], mode=mode)


def run_select(d):
    pidx.select(d["logits"], d["row_base"], d["row_len"], d["row_bin"], d["bitmap"],
                d["cand"], d["out"])


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
    """Exact top-k; ties to the lowest key index (the stock rule)."""
    key = order16(bits_row[:n]).to(torch.int64)
    idx = torch.arange(n, device=bits_row.device, dtype=torch.int64)
    comp = key * (1 << 20) + ((1 << 20) - 1 - idx)
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
    def sel_only():
        d["row_bin"][2::3].zero_()  # the scorer zeroes the candidate counts
        run_select(d)
    res["select"] = timed(sel_only, args.iters)
    words = d["bitmap"].numel() // d["row_len"].shape[0]
    ctas = torch.cuda.get_device_properties(0).multi_processor_count
    mark, fin = pidx._select_kernels(ctas * 4, 2048, words)
    st = torch.cuda.current_stream().cuda_stream
    a = (d["logits"].view(torch.int16), d["row_base"], d["row_len"], d["row_bin"],
         d["bitmap"], d["cand"])
    def mark_only():
        d["row_bin"][2::3].zero_()
        mark(*a, st)
    res["mark"] = timed(mark_only, args.iters)
    res["finish"] = timed(lambda: fin(*a, d["out"], st), args.iters)
    print(f"select (mark + finish): {res['select']:.3f} ms; mark {res['mark']:.3f}, "
          f"finish {res['finish']:.3f} (incl. a count reset)", flush=True)
    res["score+select"] = timed(lambda: (run_score(d), run_select(d)), args.iters)
    print(f"select {res['select']:.3f} ms; score+select {res['score+select']:.3f} ms "
          f"per layer (items {d['items'].shape[0]}, scratch "
          f"{d['logits'].numel() * 2 / 2**30:.2f} GiB)", flush=True)
    return res


def lane_counts(cand, t):
    return cand.view(-1, pidx.REGION // pidx.TILE, pidx.TILE)[:t, 0, :].sum(1)


def sel_plan(d):
    return pidx.SelectivePlan([r for r, _ in d["reqs"]], [kv for _, kv in d["reqs"]],
                              d["bases"], "cuda")


def run_sel(d, sp):
    return pidx.selective_topk(sp, d["q"], d["q_scale"], d["k8"], d["k_sf"], d["w"],
                               d["out"])


def check_sel(reqs, zero=None, seed=1):
    """Selective path vs the exact top-k of the stock (DeepGEMM) logits, ties to
    the lowest key index; flagged rows (recomputed by the full path) counted."""
    d = make(reqs, seed=seed, zero_rows=zero)
    sp = sel_plan(d)
    row_bin, cand = run_sel(d, sp)
    torch.cuda.synchronize()
    flag = row_bin[1:3 * sp.t:3]
    cnt = lane_counts(cand, sp.t)
    res = {"rows": sp.t, "flagged": int((flag != 0).sum()), "bad_rows": 0,
           "cand_mean": float(cnt.float().mean()), "cand_max": int(cnt.max())}
    t0 = 0
    for r, (rows, kv) in enumerate(reqs):
        sl = order16(stock_logits(d, r).view(torch.int16)).to(torch.int64)
        for c in range(0, rows, 256):
            n = min(256, rows - c)
            ke = kv - rows + 1 + c + torch.arange(n, device="cuda")
            idx = torch.arange(kv, device="cuda")
            comp = torch.where(idx[None, :] < ke[:, None],
                               sl[c:c + n] * (1 << 20) + ((1 << 20) - 1 - idx), -1)
            kk = min(2048, kv)
            tv = torch.topk(comp, kk, dim=1)
            ref = torch.sort(torch.where(tv.values >= 0, tv.indices, 1 << 40),
                             dim=1).values
            got = d["out"][t0 + c:t0 + c + n].to(torch.int64)
            got = torch.sort(torch.where(got >= 0, got, 1 << 40), dim=1).values[:, :kk]
            ok = (got == ref).all(1)  # flagged rows too (device recovery)
            res["bad_rows"] += int((~ok).sum())
        t0 += rows
    print("check_sel", json.dumps(res), flush=True)
    return res


def perf_sel(args):
    d = make([(5461, 78360), (5461, 78360), (5462, 78360)], seed=0)
    sp = sel_plan(d)
    res = {"total": timed(lambda: run_sel(d, sp), args.iters)}
    row_bin, cand = run_sel(d, sp)
    torch.cuda.synchronize()
    res["flagged"] = int((row_bin[1:3 * sp.t:3] != 0).sum())
    res["cand_mean"] = float(lane_counts(cand, sp.t).float().mean())
    rb0 = row_bin.clone()
    res["publish"] = timed(lambda: pidx.score(
        d["q"], d["q_scale"], d["k8"], d["k_sf"], d["w"], sp.row_len, sp.items,
        sp.cand_base, cand.view(torch.bfloat16), rb0, mode=pidx.PUBLISH), args.iters)
    for epi, name in ((1, "publish_skeleton"), (2, "publish_half_math"), (3, "publish_mma_only"), (4, "publish_ld_only"), (6, "publish_math_no_mma")):
        rb2, cand2 = row_bin.clone(), cand.clone()
        res[name] = timed(lambda: pidx.score(
            d["q"], d["q_scale"], d["k8"], d["k_sf"], d["w"], sp.row_len, sp.items,
            sp.cand_base, cand2.view(torch.bfloat16), rb2, mode=pidx.PUBLISH, epi=epi),
            args.iters)
    res["score_store"] = timed(lambda: run_score(d), args.iters)
    args5 = (sp, d["q"], d["q_scale"], d["k8"], d["k_sf"], d["w"])
    res["thresholds"] = timed(lambda: pidx.thresholds(*args5), args.iters)
    res["recover_idle"] = timed(lambda: pidx.recover(*args5, d["out"]), args.iters)
    res["total_recover_all"] = timed(lambda: pidx.selective_topk(
        sp, d["q"], d["q_scale"], d["k8"], d["k_sf"], d["w"], d["out"],
        recover_all=True), args.iters)
    st = torch.cuda.current_stream().cuda_stream
    ctas = pidx.CS2_CTAS_PER_SM * torch.cuda.get_device_properties(0).multi_processor_count
    for name, sk in (("cand_select", False), ("cand_select_skeleton", True)):
        k4 = pidx._cand_select2_kernel(2048, ctas, sk)
        rb1 = rb0.clone()
        res[name] = timed(lambda: k4(cand, sp.row_len, rb1, d["out"].view(-1), st),
                          args.iters)
    print("perf_sel", json.dumps(res), flush=True)
    return res


def perf_cs(args):
    """Same-process ABBA of the candidate select of --refmod vs this tree on one
    P3-16 layer's PUBLISH candidates; the two outputs must hold the same index set
    per row."""
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "pidx_ref", os.path.join(os.path.dirname(__file__), args.refmod))
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    d = make([(5461, 78360), (5461, 78360), (5462, 78360)], seed=0)
    sp = sel_plan(d)
    row_bin, cand = run_sel(d, sp)
    torch.cuda.synchronize()
    rb0 = row_bin.clone()
    st = torch.cuda.current_stream().cuda_stream
    ctas = pidx.CS2_CTAS_PER_SM * torch.cuda.get_device_properties(0).multi_processor_count
    outs, res = {}, {}
    for name, mod in (("ref", ref), ("new", pidx), ("ref2", ref), ("new2", pidx)):
        k4 = mod._cand_select2_kernel(2048, ctas)
        rb = rb0.clone()
        o = torch.full_like(d["out"], -7)
        res[name] = timed(lambda: k4(cand, sp.row_len, rb, o.view(-1), st), args.iters)
        outs[name[:3]] = (torch.sort(o, dim=1).values, rb[1::3].clone())
    res["same_rows"] = int((outs["ref"][0] == outs["new"][0]).all(1).sum())
    res["rows"] = int(d["out"].shape[0])
    res["same_flags"] = bool(torch.equal(outs["ref"][1], outs["new"][1]))
    res["flagged"] = int((outs["new"][1] != 0).sum())
    print("perf_cs", json.dumps(res), flush=True)
    return res


def perf_ref(args):
    """Same-process A/B: the selective path of benchmarks/kimi_k3/pidx_ref_9eca.py
    (the landed kernels) vs this tree, PUBLISH and total, interleaved."""
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "pidx_ref", os.path.join(os.path.dirname(__file__), args.refmod))
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    d = make([(5461, 78360), (5461, 78360), (5462, 78360)], seed=0)
    res = {}
    for name, mod in (("ref", ref), ("new", pidx), ("ref2", ref), ("new2", pidx)):
        sp = mod.SelectivePlan([r for r, _ in d["reqs"]], [kv for _, kv in d["reqs"]],
                               d["bases"], "cuda")
        a = (sp, d["q"], d["q_scale"], d["k8"], d["k_sf"], d["w"])
        res[name + "_total"] = timed(lambda: mod.selective_topk(*a, d["out"]), args.iters)
        mod.thresholds(*a)
        res[name + "_publish"] = timed(lambda: mod.publish(*a), args.iters)
    print("perf_ref", json.dumps(res), flush=True)
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
    ap.add_argument("--sel", action="store_true")
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--cs", action="store_true")
    ap.add_argument("--ncu", action="store_true")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--refmod", default="pidx_ref_9eca.py")
    a = ap.parse_args()
    if a.samples:
        pidx.SAMPLES = a.samples
    if a.margin:
        pidx.MARGIN = a.margin
    out = {}
    if a.check:
        out["check"] = check(a)
    if a.ncu:  # two selective layers: compile + warm, then the profiled one
        d = make([(5461, 78360), (5461, 78360), (5462, 78360)], seed=0)
        sp = sel_plan(d)
        for _ in range(2):
            run_sel(d, sp)
        torch.cuda.synchronize()
        return
    if a.sel:
        out["check_sel"] = [check_sel([(37, 1000), (300, 5000), (9, 2100), (513, 9000), (40, 40000), (3, 140000)], zero=list(range(337, 346))), check_sel([(700, 78360)], seed=3)]
        out["perf_sel"] = perf_sel(a)
    if a.cs:
        out["check_sel"] = [check_sel([(37, 1000), (300, 5000), (9, 2100), (513, 9000), (40, 40000), (3, 140000)], zero=list(range(337, 346))), check_sel([(700, 78360)], seed=3)]
        out["perf_cs"] = perf_cs(a)
    if a.ref:
        out["perf_ref"] = perf_ref(a)
    if a.perf:
        out["perf"] = perf(a)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
