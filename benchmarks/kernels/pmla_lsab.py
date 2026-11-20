# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA E1: is the S chain (single S buffer, L_S) the bound once the gather is fast?
h8 timing arms at F16K L78K: full / full with gskip=4 (gather-fast proxy: 1 of 5 gather4
ops per quad, stale stage data) / MMA-only (no gather), each with qkh 0 and 1 (QK in
two 32-key halves with their own S-full / S-free barriers), plus a bitwise check of qkh.
usage: pmla_lsab.py"""
import torch
from pmla_h8g_ab import case, timeit

from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import (
    sparse_mla_prefill_h8 as k8)


def kn(no_gather=0, no_math=0, gskip=0, qkh=0):
    return (0, no_gather, no_math, 0, 0, 0, 0, 0, 0, gskip, 0, qkh)


def arm(knobs):
    def f(*a, **kw):
        return k8(*a, knobs=knobs, **kw)
    return f


def main():
    args = case(300, 9000, 0.15)
    ra = k8(*args, 0.07, 1.0, unit_rows=64)
    rb = arm(kn(qkh=1))(*args, 0.07, 1.0, unit_rows=64)
    torch.cuda.synchronize()
    nd = (ra.view(torch.int16) != rb.view(torch.int16)).sum().item()
    print(f"qkh bitwise={nd == 0} ndiff={nd}", flush=True)
    args = case(16384, 78000, 0.15)
    out = torch.empty_like(k8(*args, 0.07, 1.0, unit_rows=64))
    modes = (("full", {}), ("gskip4", {"gskip": 4}), ("mma_only", {"no_gather": 1}))
    fs = {}
    for name, kw in modes:
        for qkh in (0, 1):
            f = arm(kn(qkh=qkh, **kw))
            for _ in range(3):
                f(*args, 0.07, 1.0, unit_rows=64, out=out)
            fs[(name, qkh)] = f
    torch.cuda.synchronize()
    res = {k: [] for k in fs}
    for _ in range(4):  # interleaved passes: every arm once per pass (ABBA-like drift control)
        for k in list(fs) + list(fs)[::-1]:
            res[k].append(timeit(fs[k], args, out))
    for name, _ in modes:
        a = sorted(res[(name, 0)])
        b = sorted(res[(name, 1)])
        ma, mb = a[len(a) // 2], b[len(b) // 2]
        print(f"{name:9s} qkh0 {ma:8.1f} us  qkh1 {mb:8.1f} us  delta {mb - ma:+7.1f} "
              f"({(mb / ma - 1) * 100:+.1f}%)  min {a[0]:.1f}/{b[0]:.1f}", flush=True)


if __name__ == "__main__":
    main()
