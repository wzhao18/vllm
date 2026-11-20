# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-tile clock stamps of the KDA conv GEMM (tactic flag "stamp").

Slots per (CTA, tile): 0 MMA warp before the TMEM-empty wait, 1 after it,
2 after the accumulator commit (last MMA issued); 3 epilogue entry, 4 after
the TMEM-full wait, 5 after the TMEM release (drain end), 6 column pass end,
7 globaltimer at slot 4. Leader CTAs carry the MMA slots.

Prints per arm and state (burst = one call after an idle gap, cap = the last
call of a sustained back-to-back run): time, clock, and per-tile means of the
MMA TMEM wait, MMA span, hand-off (commit -> epilogue wake), drain, column
pass, plus the per-CTA sum of the MMA TMEM wait.
"""

import argparse
import json
import statistics
import time

import torch
from pkda_m9_bench import RASTER, sustained
from pgemm_conv_bench import kernel_times, operands


def analyze(st, ncta):
    s = st.view(1024, 256, 16)[:ncta].cpu()
    out = {"mma_wait": [], "mma_span": [], "handoff": [], "epi_wait": [], "drain": [],
           "colpass": [], "resume": [], "sum_wait_us": [], "sum_epi_us": [], "tiles": []}
    mhz = []
    for c in range(ncta):
        rows = s[c]
        epi = any(rows[i, 7] != 0 for i in range(256))
        cs, gs = (4, 7) if epi else (1, 8)
        valid = [i for i in range(256) if rows[i, cs] != 0 and rows[i, gs] != 0]
        if len(valid) < 2:
            continue
        i0, i1 = valid[0], valid[-1]
        dclk = int(rows[i1, cs] - rows[i0, cs])
        dns = int(rows[i1, gs] - rows[i0, gs])
        if dns <= 0 or dclk <= 0:
            continue
        f = dclk / dns  # clk per ns
        mhz.append(f * 1000)
        lead = all(rows[i, 1] != 0 for i in valid)
        out["tiles"].append(len(valid))
        sw, se = 0.0, 0.0
        for n, i in enumerate(valid):
            r = rows[i]
            if lead:
                w = int(r[1] - r[0]) / f / 1000
                out["mma_wait"].append(w)
                sw += w
                out["mma_span"].append(int(r[2] - r[1]) / f / 1000)
                if epi:
                    out["handoff"].append(int(r[4] - r[2]) / f / 1000)
                if epi and n + 1 < len(valid):
                    out["resume"].append(int(rows[valid[n + 1], 1] - r[5]) / f / 1000)
            if not epi:
                continue
            ew = int(r[4] - r[3]) / f / 1000
            out["epi_wait"].append(ew)
            se += ew
            out["drain"].append(int(r[5] - r[4]) / f / 1000)
            out["colpass"].append(int(r[6] - r[5]) / f / 1000)
        if lead:
            out["sum_wait_us"].append(sw)
        if epi:
            out["sum_epi_us"].append(se)
    res = {"mhz": round(statistics.median(mhz)) if mhz else 0}
    for k, v in out.items():
        if v:
            res[k] = round(statistics.mean(v), 3)
            if k in ("sum_wait_us", "sum_epi_us"):
                res[k + "_med"] = round(statistics.median(v), 1)
                res[k + "_max"] = round(max(v), 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="5461,2731")
    ap.add_argument("--p", type=int, default=12288)
    ap.add_argument("--k", type=int, default=7168)
    ap.add_argument("--cluster", default="2x4")
    ap.add_argument("--arms", default="a:cskel4+stamp,e:,es:stamp")
    ap.add_argument("--cool", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    from vllm.models.kimi_k3.nvidia.ops import kda_conv  # noqa: F401
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        KdaConvGemm,
    )

    p, k = args.p, args.k
    n = 3 * p
    cl = tuple(int(v) for v in args.cluster.split("x"))
    arms = [tuple(a.split(":")) for a in args.arms.split(",")]
    for sl in [int(v) for v in args.seqs.split(",")]:
        lens = [sl] * 3
        m = sum(lens)
        xq, xs, wq, ws = operands(m, n, k)
        cw = (torch.randn(n, 4, device="cuda") * 0.5).contiguous()
        qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
        lines = torch.arange(1, 4, dtype=torch.int32, device="cuda")
        has_init = torch.tensor([False, True, False], device="cuda")
        state0 = torch.randn(5, n, 4, device="cuda").to(torch.bfloat16).float()[:, :, :3]
        gs = {}
        for spec in arms:
            arm, flag = spec[0], spec[1]
            rz = (int(spec[2]), spec[3]) if len(spec) > 3 else RASTER[cl]
            tac = ((512, 256), cl, 256, *rz, 0, 0, flag)
            g = KdaConvGemm(wq, ws, cw, p, tactic=tac)
            st = state0.clone()
            g(xq, xs, st, qsl, lines, has_init)
            torch.cuda.synchronize()
            gs[arm] = (g, st)
        ncta = torch.cuda.get_device_properties(0).multi_processor_count
        for rnd in range(args.rounds):
            for spec in arms:
                arm, flag = spec[0], spec[1]
                g, st = gs[arm]

                def f_arm():
                    g(xq, xs, st, qsl, lines, has_init)

                row = {"m": m, "arm": arm, "flag": flag, "round": rnd}
                time.sleep(args.cool)
                row["burst"] = kernel_times(f_arm, iters=args.iters)[0]
                if g.stamps is not None:
                    time.sleep(args.cool)
                    g.stamps.zero_()
                    f_arm()
                    torch.cuda.synchronize()
                    row["st_burst"] = analyze(g.stamps, ncta)
                row["sus"] = sustained(f_arm)
                if g.stamps is not None:
                    row["st_cap"] = analyze(g.stamps, ncta)
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
