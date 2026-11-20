# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MegaMoE latent up-projection behind a predecessor that releases it early.

In-step the latent up-projection follows the MegaMoE top-k reduce, which polls its
combine cells for several microseconds. This bench stands a 1-CTA spin kernel (PDL
trigger at entry, then ``--window`` ns of spinning) in for the reduce and times
``spin -> latent_up`` vs ``spin`` alone in CUDA graphs, with the weights rotated over
``--copies`` layers (cold in L2 as in-step):

  exposed = (graph(spin + op) - graph(spin)) / iterations

for stock's cuBLAS ``residual.addmm_`` and ``gemm.latent_up`` (PDL, weights staged
before its wait). ``--window 0`` = back to back without a predecessor window."""

import argparse
import json

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass.cute.runtime import from_dlpack, make_fake_stream, make_fake_tensor

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    architecture,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    stream as current_stream,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import asm
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.latent_up import (
    PreparedLatentUp,
    tactic_for,
)


@asm(Int64)
def _now():
    return "mov.u64 $0, %globaltimer;", "=l,~{memory}", ()


@asm()
def _red_release_add(address, value):
    code = "red.release.gpu.global.add.s32 [$0], $1;"
    return code, "l,r,~{memory}", (Int64(address), Int32(value))


class Spin:
    """The reduce's stand-in: releases its dependents at entry, spins, then posts
    add arrivals on the sync pair (as the reduce's CTAs do)."""

    @cute.experimental.jit
    def __call__(self, ns: Int64, sync: cute.Tensor, add: Int32, stream):
        self.kernel(ns, sync, add).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.experimental.kernel
    def kernel(self, ns: Int64, sync: cute.Tensor, add: Int32):
        cute.arch.griddepcontrol_launch_dependents()
        start = _now()
        while _now() - start < ns:
            pass
        if cute.arch.lane_idx() == 0:
            _red_release_add(sync.iterator.toint(), add)


def graph_time(fn, reps):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()  # warm
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    times.sort()
    return times[len(times) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="32,64,96,128")
    p.add_argument("--window", type=int, default=6000, help="spin ns before each op")
    p.add_argument("--copies", type=int, default=6)
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--json", default=None)
    p.add_argument(
        "--variants", default="",
        help="extra tactics n:slots:tmem_tiles (comma list), timed beside the default",
    )
    a = p.parse_args()
    dev = torch.device("cuda")
    n, k = 7168, 3584
    torch.manual_seed(0)
    ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(a.copies)]
    arch = architecture(dev)
    fake_pair = make_fake_tensor(cutlass.Int32, (2,), (1,), assumed_align=8)
    spin = cute.experimental.compile(
        Spin(), Int64(0), fake_pair, Int32(0), make_fake_stream(),
        options=f"--gpu-arch={arch}",
    )
    idle = torch.zeros(2, device=dev, dtype=torch.int32)
    pair = torch.zeros(2, device=dev, dtype=torch.int32)

    def spin_once(sync, add):
        view = from_dlpack(sync, assumed_align=8)
        spin(Int64(a.window), view, Int32(add), current_stream(dev))

    results = []
    for t in (int(v) for v in a.tokens.split(",")):
        x = torch.randn(t, k, device=dev, dtype=torch.bfloat16)
        res = [torch.randn(t, n, device=dev, dtype=torch.bfloat16) for _ in range(a.copies)]
        base_tactic = tactic_for(t)
        variants = {"tmem": base_tactic}
        for v in filter(None, a.variants.split(",")):
            vn, vs, vt = (int(f) for f in v.split(":"))
            if vn == base_tactic[1]:
                variants[f"tmem_{vn}_{vs}_{vt}"] = (128, vn, base_tactic[2], vs, vt)
        ups_by = {
            key: [
                PreparedLatentUp(w, (t,), reduce_split=4, tactics={t: tac}).prepare()
                for w in ws
            ]
            for key, tac in variants.items()
        }
        ups = ups_by["tmem"]
        # correctness vs stock on copy 0
        ref = res[0].clone().addmm_(x, ws[0].t())
        out = res[0].clone()
        pair.zero_()
        pair[0] = 4 * t
        ups[0].run(x, out, pair)
        assert pair.tolist() == [0, 0]
        err = (out.float() - ref.float()).abs().max().item()

        def body(op, with_spin):
            def fn():
                for i in range(a.iters):
                    if op.startswith("tmem"):
                        spin_once(pair, 4 * t)  # the reduce's arrivals
                        ups_by[op][i % a.copies].run(x, res[i % a.copies], pair)
                        continue
                    if with_spin:
                        spin_once(idle, 0)
                    if op == "cublas":
                        res[i % a.copies].addmm_(x, ws[i % a.copies].t())
            return fn

        base = graph_time(body("none", True), a.reps) if a.window else 0.0
        row = {"tokens": t, "window_ns": a.window, "max_abs_err_vs_addmm": err}
        for op in ("cublas", *variants):
            total = graph_time(body(op, a.window > 0), a.reps)
            row[op + "_us"] = round((total - base) / a.iters, 3)
        row["spin_us"] = round(base / a.iters, 3) if a.window else 0.0
        print(json.dumps(row), flush=True)
        results.append(row)
    if a.json:
        with open(a.json, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
