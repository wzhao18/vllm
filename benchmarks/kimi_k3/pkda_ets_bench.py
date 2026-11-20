# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P1 producer store contract: KDA conv GEMM arms a (mainloop skeleton),
es (production STG column pass), ens (stores predicated off) and ets (column
pass -> smem + one 3D TMA store per 128x32 subtile), all stamped. Per arm:
burst (one call after an idle gap), cap (sustained, the last call), J/call at
the cap, THRASH (512 MB L2 flush + straight-line code on all SMs before each
timed call) and ets vs es bitwise (q|k|v and conv state)."""
import argparse
import json
import statistics
import time

import torch
from pgemm_conv_bench import kernel_times, operands
from pkda_m9_bench import RASTER, sustained
from pkda_stamp_bench import analyze

from vllm.triton_utils import tl, triton


@triton.jit
def _icode(x_ptr, N: tl.constexpr):
    x = tl.load(x_ptr + tl.program_id(0) * 128 + tl.arange(0, 128))
    for i in tl.static_range(N):
        x = x * (1.0 + i * 1e-7) + (i * 1e-6)
    tl.store(x_ptr + tl.program_id(0) * 128 + tl.arange(0, 128), x)


def thrash_times(fn, flush, code, iters=8):
    ts = []
    for _ in range(iters):
        flush.add_(1)
        _icode[(2048,)](code, N=4096)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    return round(statistics.median(ts), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="5461,2731")
    ap.add_argument("--arms", default="a:cskel4+stamp,es:stamp,ens:stamp+nost,ets:stamp+ets")
    ap.add_argument("--thrash", default="es,ets")
    ap.add_argument("--cool", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    from vllm.models.kimi_k3.nvidia.ops import kda_conv  # noqa: F401
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        KdaConvGemm,
    )

    p, k, cl = 12288, 7168, (2, 4)
    n = 3 * p
    arms = [tuple(a.split(":")) for a in args.arms.split(",")]
    thr = set(args.thrash.split(","))
    flush = torch.zeros(128 * 1024 * 1024, dtype=torch.float32, device="cuda")
    code = torch.zeros(2048 * 128, dtype=torch.float32, device="cuda")
    for sl in [int(v) for v in args.seqs.split(",")]:
        lens = [sl] * 3
        m = sum(lens)
        xq, xs, wq, ws = operands(m, n, k)
        cw = (torch.randn(n, 4, device="cuda") * 0.5).contiguous()
        qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
        lines = torch.arange(1, 4, dtype=torch.int32, device="cuda")
        has_init = torch.tensor([False, True, False], device="cuda")
        state0 = torch.randn(5, n, 4, device="cuda").to(torch.bfloat16).float()[:, :, :3]
        gs, outs = {}, {}
        for arm, flag in arms:
            tac = ((512, 256), cl, 256, *RASTER[cl], 0, 0, flag)
            g = KdaConvGemm(wq, ws, cw, p, tactic=tac)
            st = state0.clone()
            out, _ = g(xq, xs, st, qsl, lines, has_init)
            torch.cuda.synchronize()
            gs[arm] = (g, st)
            outs[arm] = (out.clone(), st.clone())
        if "es" in outs and "ets" in outs:
            eo, es_ = outs["es"]
            to, ts_ = outs["ets"]
            print(json.dumps({"m": m, "check": "ets_vs_es", "out_bitwise": bool(torch.equal(eo, to)),
                              "out_ndiff": int((eo != to).sum()), "state_bitwise": bool(torch.equal(es_, ts_))}), flush=True)
        for rnd in range(args.rounds):
            for arm, flag in arms:
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
                    row["st_burst"] = analyze(g.stamps, torch.cuda.get_device_properties(0).multi_processor_count)
                row["sus"] = sustained(f_arm)
                row["sus"]["mj"] = round(row["sus"]["w"] * row["sus"]["us"] / 1000, 1)
                if g.stamps is not None:
                    row["st_cap"] = analyze(g.stamps, torch.cuda.get_device_properties(0).multi_processor_count)
                if arm in thr:
                    time.sleep(args.cool)
                    row["thrash"] = thrash_times(f_arm, flush, code)
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
