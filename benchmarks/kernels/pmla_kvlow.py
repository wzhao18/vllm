# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA E2: does the h8 KV ring's position in the oversized shared memory matter?
Today the ring sits at 90-306 KB (stages 3-5 above the 228 KB non-oversized limit).
Arm B moves it to 0-216 KB and every other buffer above it (same total footprint), by
patching the module's offsets before its compile (pad=1 only makes a distinct cache key).
Interleaved timing at F16K L78K: full and gskip=4 (gather-fast proxy), plus bitwise check.
usage: pmla_kvlow.py"""
import torch
from pmla_h8g_ab import case, timeit

import vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 as m8


def kn(gskip=0, pad=0):
    return (0, 0, 0, 0, 0, pad, 0, 0, 0, gskip)


def arm(knobs):
    def f(*a, **kw):
        return m8.sparse_mla_prefill_h8(*a, knobs=knobs, **kw)
    return f


def compile_all(pad, args, out):
    fs = {}
    for name, gs in (("full", 0), ("gskip4", 4)):
        f = arm(kn(gs, pad))
        f(*args, 0.07, 1.0, unit_rows=64, out=out)
        fs[name] = f
    return fs


def kv_low():
    ring = m8.NS * m8.STAGE64
    for k in [k for k in dir(m8) if k.startswith("OFF_") and k != "OFF_KV"]:
        setattr(m8, k, getattr(m8, k) + ring)
    m8.OFF_KV = 0
    assert m8.OFF_QN + (m8.SMEM_BYTES - ring) == m8.SMEM_BYTES


def main():
    small = case(300, 9000, 0.15)
    args = case(16384, 78000, 0.15)
    ra = m8.sparse_mla_prefill_h8(*small, 0.07, 1.0, unit_rows=64)
    out = torch.empty_like(m8.sparse_mla_prefill_h8(*args, 0.07, 1.0, unit_rows=64))
    fa = compile_all(0, args, out)
    print(f"A layout OFF_KV={m8.OFF_KV} OFF_QN={m8.OFF_QN} SMEM={m8.SMEM_BYTES}", flush=True)
    kv_low()
    print(f"B layout OFF_KV={m8.OFF_KV} OFF_QN={m8.OFF_QN} OFF_P={m8.OFF_P} "
          f"OFF_BAR={m8.OFF_BAR}", flush=True)
    rb = arm(kn(0, 1))(*small, 0.07, 1.0, unit_rows=64)
    torch.cuda.synchronize()
    nd = (ra.view(torch.int16) != rb.view(torch.int16)).sum().item()
    print(f"kvlow bitwise={nd == 0} ndiff={nd}", flush=True)
    fb = compile_all(1, args, out)
    torch.cuda.synchronize()
    res = {(n, ab): [] for n in fa for ab in "AB"}
    for _ in range(4):
        for n in fa:
            for ab, fs in (("A", fa), ("B", fb), ("B", fb), ("A", fa)):
                res[(n, ab)].append(timeit(fs[n], args, out))
    for n in fa:
        a, b = sorted(res[(n, "A")]), sorted(res[(n, "B")])
        ma, mb = a[len(a) // 2], b[len(b) // 2]
        print(f"{n:7s} A(ring 90-306K) {ma:8.1f} us  B(ring 0-216K) {mb:8.1f} us  "
              f"delta {mb - ma:+7.1f} ({(mb / ma - 1) * 100:+.1f}%)", flush=True)


if __name__ == "__main__":
    main()
