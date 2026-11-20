# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA M7 (RoPE LDG warps): bitwise + timing A/B of the h8 kernel in this tree (B) vs a saved copy of the
previous h8 (A, pmla_patches/sparse_mla_h8_prev.py). ABBA-interleaved timing at F16K.
usage: pmla_m5_ab.py <prev_kernel.py> [qmag ...]"""
import importlib.util
import sys

import torch

from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import (
    sparse_mla_prefill_h8 as kb)

H, ROW = 96, 576


def load_prev(path):
    spec = importlib.util.spec_from_file_location("sparse_mla_h8_prev", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.sparse_mla_prefill_h8


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


def main():
    ka = load_prev(sys.argv[1])
    qmags = [float(x) for x in sys.argv[2:]] or [0.15]
    for qmag in qmags:
        for t, ctx in ((300, 9000), (16384, 78000)):
            args = case(t, ctx, qmag)
            ra = ka(*args, 0.07, 1.0, unit_rows=64)
            rb = kb(*args, 0.07, 1.0, unit_rows=64)
            torch.cuda.synchronize()
            nd = (ra.view(torch.int16) != rb.view(torch.int16)).sum().item()
            md = (ra.float() - rb.float()).abs().max().item()
            line = f"qmag={qmag} T={t} bitwise={nd == 0} ndiff={nd} maxdiff={md:.3g}"
            if t >= 4096:
                oa, ob = torch.empty_like(ra), torch.empty_like(rb)
                for _ in range(3):
                    ka(*args, 0.07, 1.0, unit_rows=64, out=oa)
                    kb(*args, 0.07, 1.0, unit_rows=64, out=ob)
                ta, tb = [], []
                for _ in range(4):  # ABBA x 4
                    ta.append(timeit(ka, args, oa))
                    tb.append(timeit(kb, args, ob))
                    tb.append(timeit(kb, args, ob))
                    ta.append(timeit(ka, args, oa))
                ta.sort()
                tb.sort()
                line += (f" A_us={ta[len(ta) // 2]:.1f} (min {ta[0]:.1f}) B_us={tb[len(tb) // 2]:.1f}"
                         f" (min {tb[0]:.1f}) delta={tb[len(tb) // 2] - ta[len(ta) // 2]:+.1f}")
            print(line, flush=True)


if __name__ == "__main__":
    main()
