# SOL M1 probes for designs/dsa-layer.md (NOT production code). One GPU unless noted.
#   python3 m1dsa.py sel|topn|gqk|pre [--quick]
# Prints one "REC {json}" line per measured configuration.
import argparse
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def ext():
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    arch = "100f" if major == 10 else f"{major}{minor}a"  # VR200: build 10.0f (10.7a extension builds trap)
    build = os.environ.get("M1_BUILD", str(HERE / "build"))
    Path(build).mkdir(parents=True, exist_ok=True)
    return load(
        name=f"m1dsa_sm{arch}",
        sources=[str(HERE / "m1dsa.cu")],
        extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-std=c++20", "-lineinfo"],
        build_directory=build,
        verbose=False,
    )


def clk_ghz(c, g):
    c, g = c.double(), g.double()
    m = g > 2000
    return float((c[m]).sum() / g[m].sum()) if m.any() else 2.39  # (short CTAs: the probes' measured ~2.39 GHz)


def stats(x):
    x = sorted(x)
    n = len(x)
    return dict(med=x[n // 2], p10=x[n // 10], p90=x[(9 * n) // 10], mean=sum(x) / n, n=n)


def rec(**k):
    print("REC " + json.dumps(k), flush=True)


def ord16_t(b):
    b = b.to(torch.int32) & 0xFFFF
    return torch.where((b & 0x8000) != 0, (~b) & 0xFFFF, b | 0x8000)


# ------------------------------------------------------------------ M1-a select
def probe_sel(E, args):
    W, N, K = 16, 192, 2048
    for rows in args.rows:
        g = torch.Generator(device="cuda").manual_seed(rows)
        # each rank's top-192 of 8250 randn logits (BF16), sorted descending (the #63 "tail" case)
        lg = torch.randn((rows, W, 8250), generator=g, device="cuda").to(torch.bfloat16)
        sc = lg.float().topk(N, dim=2).values.to(torch.bfloat16).contiguous()
        bits = sc.view(torch.int16).reshape(rows, W * N).contiguous()
        m = torch.zeros((rows, W), dtype=torch.int32, device="cuda")
        thr = torch.zeros((rows,), dtype=torch.int32, device="cuda")
        st = torch.zeros((rows, 4), dtype=torch.int64, device="cuda")
        # reference: the K-th largest ord value; m_r = count > thr + tie quota by rank
        o = ord16_t(bits.view(rows, W, N))
        kth = o.reshape(rows, -1).sort(dim=1, descending=True).values[:, K - 1]
        for T, agg in [(t, a) for a in args.agg for t in args.threads]:
            durs, spans, cks, gs = [], [], [], []
            for it in range(args.reps + 3):
                E.sel_run(bits, W, N, K, m, thr, st, T, agg)
                torch.cuda.synchronize()
                if it < 3:
                    continue
                s = st.clone()
                cks.append(s[:, 1] - s[:, 0]); gs.append(s[:, 3] - s[:, 2])
                spans.append(float((s[:, 3].max() - s[:, 2].min())) / 1e3)
            ok = bool((thr.to(torch.int64) == kth.to(torch.int64)).all()) and bool((m.sum(1) == K).all())
            gt = (o > kth[:, None, None]).sum(2)
            ge = (o >= kth[:, None, None]).sum(2)
            ok = ok and bool(((m >= gt) & (m <= ge)).all())
            c = torch.cat(cks); gg = torch.cat(gs)
            ghz = clk_ghz(c, gg)
            per = (c.double() / ghz / 1e3).tolist()
            rec(probe="sel", rows=rows, threads=T, agg=agg, W=W, N=N, K=K, ok=ok, ghz=round(ghz, 3),
                cta_us=stats(per), grid_span_us=stats(spans))


# ------------------------------------------------------------------ M1-b top-N
def probe_topn(E, args):
    NK, TOPN = args.nk, 192
    for Q in args.requests:
        g = torch.Generator(device="cuda").manual_seed(Q)
        lg = torch.randn((Q, 8, NK), generator=g, device="cuda").to(torch.bfloat16).contiguous()
        bits = lg.view(torch.int16)
        o = ord16_t(bits).to(torch.int64)
        key = torch.arange(NK, device="cuda")
        comp = o * (1 << 20) + ((1 << 20) - 1 - key)  # score desc, key asc
        ref = comp.sort(dim=2, descending=True).indices[:, :, :TOPN].to(torch.int32)
        # row-per-CTA variant (C = 0 in the records)
        ok_k = torch.zeros((Q, 8, TOPN), dtype=torch.int32, device="cuda")
        ok_s = torch.zeros((Q, 8, TOPN), dtype=torch.int16, device="cuda")
        st = torch.zeros((Q * 8, 6), dtype=torch.int64, device="cuda")
        cks, gs, spans, phases = [], [], [], []
        for it in range(args.reps + 3):
            E.topn_row_run(bits, TOPN, ok_k, ok_s, st)
            torch.cuda.synchronize()
            if it < 3:
                continue
            s = st.clone()
            cks.append(s[:, 4] - s[:, 0]); gs.append(s[:, 5])
            phases.append((s[:, 1:5] - s[:, 0:1]).double().mean(0))
        ok = bool((ok_k == ref).all())
        c = torch.cat(cks); gg = torch.cat(gs)
        ghz = clk_ghz(c, gg)
        ph = (torch.stack(phases).mean(0) / ghz / 1e3).tolist()
        rec(probe="topn_row", requests=Q, rows=8 * Q, NK=NK, ok=ok, ghz=round(ghz, 3),
            cta_us=stats((c.double() / ghz / 1e3).tolist()),
            phase_us=dict(pass1=ph[0], pass2=ph[1], emit=ph[2], end=ph[3]))
        for C in args.clusters:
            if Q * C > 4 * 212 and C > 1:
                continue
            ok_k = torch.zeros((Q, 8, TOPN), dtype=torch.int32, device="cuda")
            ok_s = torch.zeros((Q, 8, TOPN), dtype=torch.int16, device="cuda")
            st = torch.zeros((Q * C, 8), dtype=torch.int64, device="cuda")
            cks, gs, spans, phases = [], [], [], []
            try:
                for it in range(args.reps + 3):
                    E.topn_run(bits, C, TOPN, ok_k, ok_s, st)
                    torch.cuda.synchronize()
                    if it < 3:
                        continue
                    s = st.clone()
                    cks.append(s[:, 4] - s[:, 0]); gs.append(s[:, 6] - s[:, 5])
                    spans.append(float(s[:, 6].max() - s[:, 5].min()) / 1e3)
                    phases.append((s[:, 1:5] - s[:, 0:1]).double().mean(0))
            except Exception as e:  # noqa: BLE001
                rec(probe="topn", requests=Q, C=C, error=str(e)[:200])
                continue
            ok = bool((ok_k == ref).all())
            c = torch.cat(cks); gg = torch.cat(gs)
            ghz = clk_ghz(c, gg)
            ph = (torch.stack(phases).mean(0) / ghz / 1e3).tolist()
            rec(probe="topn", requests=Q, rows=8 * Q, C=C, NK=NK, ok=ok, ghz=round(ghz, 3),
                cta_us=stats((c.double() / ghz / 1e3).tolist()), grid_span_us=stats(spans),
                phase_us=dict(hist1=ph[0], hist2=ph[1], counts=ph[2], end=ph[3]))


# ------------------------------------------------------------------ F1 gather + QK
def probe_gqk(E, args):
    NC, G = 192, 6
    nkv = args.kv_rows
    kv = torch.randint(0, 255, (nkv, 576), dtype=torch.uint8, device="cuda")
    kv = (kv & 0x37) | 0x08  # finite E4M3 values (no NaN 0x7F/0xFF)
    for R in args.rows:
        g = torch.Generator(device="cuda").manual_seed(R)
        q = ((torch.randint(0, 255, (R, G * 16, 576), generator=g, dtype=torch.uint8, device="cuda") & 0x37) | 0x08)
        slots = torch.randint(0, nkv, (R, NC), generator=g, dtype=torch.int32, device="cuda")
        out = torch.zeros((R, G * 16, NC), dtype=torch.float32, device="cuda")
        # reference QK on a small subset
        for S in args.splits:
            st = torch.zeros((S * G * R, 6), dtype=torch.int64, device="cuda")
            cg, cq, gq, spans = [], [], [], []
            for it in range(args.reps + 3):
                # new slots each rep (cold rows, as a step's fresh selection)
                slots = torch.randint(0, nkv, (R, NC), dtype=torch.int32, device="cuda")
                torch.cuda.synchronize()
                E.gqk_run(q, slots, kv, S, out, st)
                torch.cuda.synchronize()
                if it < 3:
                    continue
                s = st.clone()
                cg.append(s[:, 1] - s[:, 0]); cq.append(s[:, 2] - s[:, 0]); gq.append(s[:, 4] - s[:, 3])
                spans.append(float(s[:, 4].max() - s[:, 3].min()) / 1e3)
            ghz = clk_ghz(torch.cat(cq), torch.cat(gq))
            # correctness: row 0, head 0..15, group 0 vs float reference
            qf = q[0, :16].view(torch.float8_e4m3fn).float()
            kf = kv[slots[0].long()].view(torch.float8_e4m3fn).float()
            refs = qf @ kf.T
            ok = bool(torch.allclose(out[0, :16], refs, rtol=1e-3, atol=1e-2))
            g_us = (torch.cat(cg).double() / ghz / 1e3).tolist()
            q_us = (torch.cat(cq).double() / ghz / 1e3).tolist()
            rec(probe="gqk", rows=R, splits=S, cand=NC, per_cta=-(-NC // S), ctas=S * G * R, ok=ok, ghz=round(ghz, 3),
                gather_us=stats(g_us), gather_qk_us=stats(q_us), grid_span_us=stats(spans))


# ------------------------------------------------------------------ (f) prestaging
def probe_pre(E, args):
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    MB = 1e6
    main = int(43.6 * MB)
    exb_all = {
        "B1": [int(12.6 * MB / sms) & ~15, int(4.3 * MB / sms) & ~15, int(0.57 * MB / sms) & ~15, int(12.6 * MB / sms) & ~15],
    }
    sets = max(4, int(3 * l2 / ((main + 30 * MB))) + 1)
    mains = [torch.empty(main + 1024, dtype=torch.uint8, device="cuda").fill_(1) for _ in range(sets)]
    for tag, exb in exb_all.items():
        exs = [torch.empty(sum(exb) * sms + 1024, dtype=torch.uint8, device="cuda").fill_(2) for _ in range(sets)]
        for mode in (0, 1, 2, 3):
            st = torch.zeros((sms, 8), dtype=torch.int64, device="cuda")
            res = []
            for it in range(args.reps + 3):
                k = it % sets
                E.pre_run(mains[k], main, exs[k], exb, mode, sms, st)
                torch.cuda.synchronize()
                if it < 3:
                    continue
                s = st.clone()
                t0 = s[:, 0].min()
                res.append([float((s[:, j].max() - t0)) / 1e3 for j in (1, 2, 3, 4, 5)])
            cols = list(zip(*res))
            rec(probe="pre", tag=tag, mode=mode, sms=sms, l2_MB=l2 / 1e6, sets=sets, main_MB=main / 1e6,
                ex_MB=[round(x * sms / 1e6, 2) for x in exb],
                main_done_us=stats(cols[0]), wq_idx_us=stats(cols[1]), qb_wuk_us=stats(cols[2]),
                keys_us=stats(cols[3]), w_uv_us=stats(cols[4]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--threads", default="256,512,1024")
    ap.add_argument("--agg", default="2,0")
    ap.add_argument("--requests", default="1,2,4,8,16,32,64")
    ap.add_argument("--clusters", default="1,2,4,8,16")
    ap.add_argument("--splits", default="1,2,4,8")
    ap.add_argument("--nk", type=int, default=8250)
    ap.add_argument("--kv-rows", type=int, default=6_000_000)
    args = ap.parse_args()
    lst = lambda s: [int(x) for x in s.split(",")]  # noqa: E731
    args.threads, args.requests, args.clusters, args.splits, args.agg = map(lst, (args.threads, args.requests, args.clusters, args.splits, args.agg))
    torch.cuda.set_device(0)
    p = torch.cuda.get_device_properties(0)
    print(f"DEV {p.name} sms {p.multi_processor_count} cap {torch.cuda.get_device_capability()} l2 {p.L2_cache_size}", flush=True)
    E = ext()
    if args.probe == "sel":
        args.rows = lst(args.rows or "8,16,32,64,128,256,512")
        probe_sel(E, args)
    elif args.probe == "topn":
        probe_topn(E, args)
    elif args.probe == "gqk":
        args.rows = lst(args.rows or "8,16,32,64,128,256,512")
        probe_gqk(E, args)
    elif args.probe == "pre":
        probe_pre(E, args)
    else:
        sys.exit(f"unknown probe {args.probe}")


if __name__ == "__main__":
    main()
