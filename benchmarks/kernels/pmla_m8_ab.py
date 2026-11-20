# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA M8 (LDG row-producer warps): bitwise + ABBA timing of h8 variants in this tree
(B, knobs nldg / lq) vs a saved copy of the previous h8 (A, pmla_patches/sparse_mla_h8_prev.py).
usage: pmla_m8_ab.py <prev_kernel.py> nldg:lq [nldg:lq ...]"""
import importlib.util
import sys

import torch

from pmla_m7_ab import case, timeit
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import (
    sparse_mla_prefill_h8 as kb)


def load_prev(path):
    spec = importlib.util.spec_from_file_location("sparse_mla_h8_prev", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.sparse_mla_prefill_h8


def main():
    ka = load_prev(sys.argv[1])
    variants = []
    for v in sys.argv[2:]:
        nldg, lq = (int(x) for x in v.split(":"))
        variants.append((v, (0,) * 11 + (nldg + 1, lq + 1)))
    for t, ctx in ((300, 9000), (16384, 78000)):
        args = case(t, ctx, 0.15)
        ra = ka(*args, 0.07, 1.0, unit_rows=64)
        oa = torch.empty_like(ra)
        for name, knobs in variants:
            def fb(*a, **k):
                return kb(*a, knobs=knobs, **k)
            try:
                rb = fb(*args, 0.07, 1.0, unit_rows=64)
                torch.cuda.synchronize()
            except Exception as e:  # report and go on with the next variant
                print(f"{name} T={t} ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            nd = (ra.view(torch.int16) != rb.view(torch.int16)).sum().item()
            line = f"{name} T={t} bitwise={nd == 0} ndiff={nd}"
            if t >= 4096:
                ob = torch.empty_like(rb)
                for _ in range(3):
                    ka(*args, 0.07, 1.0, unit_rows=64, out=oa)
                    fb(*args, 0.07, 1.0, unit_rows=64, out=ob)
                ta, tb = [], []
                for _ in range(3):  # ABBA x 3
                    ta.append(timeit(ka, args, oa))
                    tb.append(timeit(fb, args, ob))
                    tb.append(timeit(fb, args, ob))
                    ta.append(timeit(ka, args, oa))
                ta.sort()
                tb.sort()
                ma, mb = ta[len(ta) // 2], tb[len(tb) // 2]
                line += f" A_us={ma:.1f} B_us={mb:.1f} delta={mb - ma:+.1f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
