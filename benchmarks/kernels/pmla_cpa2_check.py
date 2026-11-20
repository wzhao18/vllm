# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA-2CTA: h8 gather by cp.async (knob cpa=2) vs TMA gather4 (base): bitwise check on mixed
token lists (short / masked / full rows), then F16K-like timing ABBA (synthetic slots over a 78K
context, real top-k shape). Usage: python3 pmla_cpa2_check.py [knob lists, e.g. 0:0:0:0:0:0:0:0:0:0:2]"""

import sys

import torch

from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import sparse_mla_prefill_h8

H, ROW, TOPK = 96, 576, 2048


def inputs(t, ctx, unit=64, seed=0, short=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = (ctx + unit - 1) // unit * unit
    q8 = (torch.randn(t, H, ROW, device="cuda", generator=g) * 0.15).to(torch.float8_e4m3fn)
    kv = (torch.randn(rows, ROW, device="cuda", generator=g)).to(torch.float8_e4m3fn)
    pages = torch.randperm(rows // unit, device="cuda", generator=g).int()
    logical = torch.randint(0, ctx, (t, TOPK), device="cuda", generator=g).int()
    slots = pages[logical // unit] * unit + logical % unit
    lens = torch.full((t,), TOPK, dtype=torch.int32, device="cuda")
    if short:
        lens = torch.randint(1, TOPK + 1, (t,), device="cuda", generator=g).int()
        lens[::7] = 1
        lens[1::7] = 64
        lens[2::7] = 65
        slots[3::5, 100:300] = -1
    return q8, kv, slots.contiguous(), lens


def run(q8, kv, slots, lens, knobs):
    return sparse_mla_prefill_h8(q8, kv, slots, lens, 0.0722, 1.0, knobs=knobs)


def main():
    kl = [tuple(int(v) for v in k.split(":")) for k in ([a for a in sys.argv[1:] if not a.startswith("--")] or ["0:0:0:0:0:0:0:0:0:0:2"])]
    for t, short, seed in (() if '--nocheck' in sys.argv else ((300, True, 1), (149, True, 2), (2048, False, 3), (640, True, 4))):
        inp = inputs(t, 9000, seed=seed, short=short)
        ref = run(*inp, ())
        for k in kl:
            o = run(*inp, k)
            torch.cuda.synchronize()
            print(f"check t={t} short={short} knobs={k}: equal={torch.equal(o, ref)} "
                  f"maxdiff={(o.float() - ref.float()).abs().max().item():.3g}", flush=True)
    inp = inputs(16384, 78 * 1024, seed=5)
    arms = ([] if '--nobase' in sys.argv else [()]) + kl
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    res = {a: [] for a in arms}
    for _ in range(2):
        run(*inp, ())
    order = arms + arms[::-1] + arms + arms[::-1]
    for a in order:
        ev[0].record()
        for _ in range(3):
            run(*inp, a)
        ev[1].record()
        torch.cuda.synchronize()
        res[a].append(ev[0].elapsed_time(ev[1]) / 3 * 1000)
    for a in arms:
        v = sorted(res[a])
        print(f"F16K L78K knobs={a or 'base'}: median {v[len(v) // 2]:.1f} us  all {[round(x) for x in res[a]]}",
              flush=True)


if __name__ == "__main__":
    main()
