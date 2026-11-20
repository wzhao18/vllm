# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA M10: h8 (TMA gather4 producer, A) vs h8g (LDGSTS row producer, B): bitwise
equality on masked/short rows and ABBA timing at F16K L78K, full kernel and the
gather-only diagnostic arm (no_store + no_math + no_mma).
usage: pmla_h8g_ab.py [nlw:lag ...]"""
import sys

import torch

from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import (
    sparse_mla_prefill_h8 as ka)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8g import (
    sparse_mla_prefill_h8g)

H, ROW = 96, 576
GATHER_ONLY = (1, 0, 1, 1)


def case(t, ctx, qmag, topk=2048, seed=0, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    q = (torch.randn((t, H, ROW), device=dev, generator=g) * qmag).to(torch.float8_e4m3fn)
    kv = (torch.randn((ctx, ROW), device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
    slots = torch.stack([torch.randperm(ctx, device=dev, generator=g)[:topk]
                         for _ in range(t)]).int()
    slots[:, 7::97] = -1
    lengths = torch.full((t,), topk, dtype=torch.int32, device=dev)
    lengths[1::5] = torch.randint(1, topk, (lengths[1::5].numel(),), device=dev,
                                  generator=g).int()
    return q, kv, slots.contiguous(), lengths


def timeit(f, args, out, n=10):
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(n):
        f(*args, 0.07, 1.0, unit_rows=64, out=out)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / n * 1e3


def abba(fa, fb, args, oa, ob, rounds=4):
    for _ in range(3):
        fa(*args, 0.07, 1.0, unit_rows=64, out=oa)
        fb(*args, 0.07, 1.0, unit_rows=64, out=ob)
    ta, tb = [], []
    for _ in range(rounds):
        ta.append(timeit(fa, args, oa))
        tb.append(timeit(fb, args, ob))
        tb.append(timeit(fb, args, ob))
        ta.append(timeit(fa, args, oa))
    ta.sort()
    tb.sort()
    return ta[len(ta) // 2], tb[len(tb) // 2], ta[0], tb[0]


def main():
    arms = [tuple(int(v) for v in x.split(":")) for x in sys.argv[1:]] or [(4, 3)]
    for nlw, lag in arms:
        def kb(*a, knobs=(), **kw):
            return sparse_mla_prefill_h8g(*a, knobs=knobs, nlw=nlw, lag=lag, **kw)

        for t, ctx in ((300, 9000), (16384, 78000)):
            args = case(t, ctx, 0.15)
            ra = ka(*args, 0.07, 1.0, unit_rows=64)
            rb = kb(*args, 0.07, 1.0, unit_rows=64)
            torch.cuda.synchronize()
            nd = (ra.view(torch.int16) != rb.view(torch.int16)).sum().item()
            md = (ra.float() - rb.float()).abs().max().item()
            line = f"nlw={nlw} lag={lag} T={t} bitwise={nd == 0} ndiff={nd} maxdiff={md:.3g}"
            if t >= 4096:
                oa, ob = torch.empty_like(ra), torch.empty_like(rb)
                a, b, amin, bmin = abba(ka, kb, args, oa, ob)
                line += f" full A_us={a:.1f} B_us={b:.1f} delta={b - a:+.1f} (min {amin:.1f}/{bmin:.1f})"

                def ga(*x, **kw):
                    return ka(*x, knobs=GATHER_ONLY, **kw)

                def gb(*x, **kw):
                    return kb(*x, knobs=GATHER_ONLY, **kw)

                a, b, _, _ = abba(ga, gb, args, oa, ob, rounds=2)
                line += f" | gather_only A_us={a:.1f} B_us={b:.1f} delta={b - a:+.1f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
