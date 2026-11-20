# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill KDA core with the gated o_norm + MXFP8 fused into its epilogue vs
the core followed by the standalone o_norm_mx pass, at the DEP16 prefill
shapes: bitwise check of the MXFP8 data and scales (and of O / final state
when the fused arm keeps O), then interleaved CUDA-graph timing."""

import argparse
import json

import torch

from vllm.models.kimi_k3.nvidia.ops.mxfp8_act import Mxfp8Buffer
from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda
from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent import (
    forward as kp,
)

D = 128


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", nargs="+", default=["5461,5461,5461", "2731,2731,2730"])
    ap.add_argument("--heads", type=int, default=96)
    ap.add_argument("--row-stride", type=int, default=49376)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--arms", default="fip,a,c,d")
    a = ap.parse_args()
    H = a.heads
    P = H * D
    bf = torch.bfloat16
    for lens_s in a.lens:
        lens = [int(x) for x in lens_s.split(",")]
        torch.manual_seed(0)
        T, N = sum(lens), len(lens)
        offs = [0, *torch.tensor(lens).cumsum(0).tolist()]
        cu_cpu = torch.tensor(offs, dtype=torch.int32)
        cu = cu_cpu.cuda()
        q, k, g = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
        v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
        beta = torch.randn(1, T, H, device="cuda").to(bf)
        A_log = torch.randn(H, device="cuda") * 0.5
        dt_bias = torch.randn(H * D, device="cuda") * 0.1
        st0 = torch.randn(N, H, D, D, device="cuda") * 0.1
        proj = (torch.randn(T, a.row_stride, device="cuda") * 2).to(bf)
        gate = proj[:, 3 * P : 4 * P]
        w = torch.rand(D, device="cuda") + 0.5
        eps = 1e-6
        plan = kp.make_plan(offs, H, q.device)
        bufs = {n: Mxfp8Buffer(T, P, "cuda", bf) for n in ("a", "c", "d")}
        outs = {n: torch.empty(1, T, H, D, dtype=bf, device="cuda") for n in "acd"}
        fins = {n: torch.empty_like(st0) for n in "acd"}

        def run(arm):
            if arm == "fip":
                kp.fwd(q, k, v, g, beta, D**-0.5, outs["a"], A_log, dt_bias.view(H, D),
                       -5.0, initial_state=st0, final_state=fins["a"], cu_seqlens=cu,
                       plan=plan)
            elif arm == "onorm":
                sp_kda.o_norm_mxfp8(outs["a"], gate, w, eps, bufs["a"])
            elif arm == "a":
                run("fip")
                run("onorm")
            else:
                b = bufs[arm]
                kp.fwd(q, k, v, g, beta, D**-0.5, outs[arm], A_log, dt_bias.view(H, D),
                       -5.0, initial_state=st0, final_state=fins[arm], cu_seqlens=cu,
                       plan=plan, o_norm=(gate, w, eps, b.data, b.scale),
                       onorm_keep_out=(arm == "c"))

        arms = a.arms.split(",")
        for n in ("a", "c", "d"):
            bufs[n].data.view(torch.uint8).zero_()
            bufs[n].scale.zero_()
        for arm in ("a", "c", "d"):
            if arm in arms or arm == "a":
                run(arm)
        torch.cuda.synchronize()
        res = {"lens": lens_s}
        for arm in ("c", "d"):
            if arm not in arms:
                continue
            dq = torch.equal(bufs[arm].data.view(torch.uint8), bufs["a"].data.view(torch.uint8))
            ds = torch.equal(bufs[arm].scale, bufs["a"].scale)
            nd = int((bufs[arm].data.view(torch.uint8) != bufs["a"].data.view(torch.uint8)).sum())
            ns = int((bufs[arm].scale != bufs["a"].scale).sum())
            res[f"{arm}_bitwise"] = bool(dq and ds)
            res[f"{arm}_ndiff"] = [nd, ns]
            res[f"{arm}_state_eq"] = bool(torch.equal(fins[arm], fins["a"]))
            if arm == "c":
                res["c_out_eq"] = bool(torch.equal(outs["c"], outs["a"]))
        print(json.dumps(res), flush=True)

        graphs = {}
        tarms = [x for x in arms] + (["onorm"] if "a" in arms else [])
        for arm in tarms:
            for _ in range(2):
                run(arm)
            torch.cuda.synchronize()
            gph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gph):
                run(arm)
            graphs[arm] = gph
        tm = {arm: [] for arm in tarms}
        for r in range(a.rounds):
            order = tarms if r % 2 == 0 else tarms[::-1]
            for arm in order:
                ts = []
                for _ in range(a.iters):
                    e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    e0.record()
                    graphs[arm].replay()
                    e1.record()
                    e1.synchronize()
                    ts.append(e0.elapsed_time(e1) * 1e3)
                ts.sort()
                tm[arm].append(ts[len(ts) // 2])
        out = {"lens": lens_s, "us": {k2: [round(x, 1) for x in v2] for k2, v2 in tm.items()}}
        med = {k2: sorted(v2)[len(v2) // 2] for k2, v2 in tm.items()}
        out["med"] = {k2: round(v2, 1) for k2, v2 in med.items()}
        if "a" in med:
            for arm in ("c", "d"):
                if arm in med:
                    out[f"{arm}-a"] = round(med[arm] - med["a"], 1)
                    if "fip" in med:
                        out[f"{arm}-fip"] = round(med[arm] - med["fip"], 1)
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
