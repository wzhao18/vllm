# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA E4: h8 pipeline bound with real data: full vs gather-once (no_gather=2: the
first NS stages per CTA gathered, later stages reuse them) vs gather-once no-math.
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
    modes = (("full", {}), ("gonce", {"no_gather": 2}), ("gonce_nomath", {"no_gather": 2, "no_math": 1}))
    fs = {}
    for name, kw in modes:
        for qkh in (0,):
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
        print(f"{name:13s} {a[len(a) // 2]:8.1f} us  min {a[0]:.1f}", flush=True)


if __name__ == "__main__":
    main()
