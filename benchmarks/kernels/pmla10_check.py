# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA-2CTA round 2: h8 gather by lean per-lane cp.async (knob cpa=3) vs TMA gather4.
1) bitwise: physical mixed lists + logical (fused index remap, short requests) vs base;
2) F16K L78K ABBA, physical and logical (one 78K request, block size 64).
Usage: python3 pmla10_check.py [knob lists, default 0:0:0:0:0:0:0:0:0:0:3] [--nocheck]"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmla_cpa2_check import inputs, run  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import (  # noqa: E402
    sparse_mla_prefill_h8,
    sparse_mla_prefill_h8_logical,
)

H, ROW = 96, 576
E4M3 = torch.float8_e4m3fn


def logical_case(t_list, ctx_blocks, block_size=64, stride=64, width=2048, seed=1, blocks=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    reqs = len(ctx_blocks)
    blocks = blocks or sum(ctx_blocks) + 8
    kv = (torch.randn(blocks * stride, ROW, device="cuda", generator=g) * 0.8).to(E4M3)
    bt = torch.zeros((reqs, max(ctx_blocks) + 1), dtype=torch.int32, device="cuda")
    perm = torch.randperm(blocks, device="cuda", generator=g).to(torch.int32)
    start = 0
    for r, nb in enumerate(ctx_blocks):
        bt[r, :nb] = perm[start:start + nb]
        start += nb
    t = len(t_list)
    req_ids = torch.arange(t, device="cuda", dtype=torch.int32) % reqs
    ctxs = torch.tensor([ctx_blocks[i % reqs] * block_size for i in range(t)], device="cuda")
    lens = torch.minimum(torch.tensor(t_list, device="cuda"), ctxs).to(torch.int32)
    r = torch.rand((t, width), device="cuda", generator=g)
    topk = (r * ctxs[:, None]).to(torch.int32)
    topk = torch.where(torch.arange(width, device="cuda")[None, :] < lens[:, None], topk, -1)
    q8 = (torch.randn(t, H, ROW, device="cuda", generator=g) * 0.8).to(E4M3)
    valid = topk >= 0
    blk = bt[req_ids.long()[:, None], (topk.clamp(min=0) // block_size).long()]
    slots = torch.where(valid, blk * stride + topk.clamp(min=0) % block_size, -1).to(torch.int32)
    return q8, kv, topk.contiguous(), req_ids, bt, block_size, stride, slots, lens


def run_l(c, knobs):
    q8, kv, topk, req_ids, bt, bs, stride, slots, lens = c
    # lengths / phys_dbg are read only by the lglob / xphys timing diagnostics
    return sparse_mla_prefill_h8_logical(q8, kv, topk, req_ids, bt, bs, stride, 0.0722, 1.3,
                                         knobs=knobs, lengths=lens, phys_dbg=slots)


def abba(fn, arms, label):
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    res = {a: [] for a in arms}
    for a in arms:
        fn(a)
    for a in arms + arms[::-1] + arms + arms[::-1]:
        ev[0].record()
        for _ in range(3):
            fn(a)
        ev[1].record()
        torch.cuda.synchronize()
        res[a].append(ev[0].elapsed_time(ev[1]) / 3 * 1000)
    for a in arms:
        v = sorted(res[a])
        print(f"{label} knobs={a or 'base'}: median {v[len(v) // 2]:.1f} us  all {[round(x) for x in res[a]]}",
              flush=True)


def main():
    import vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 as h8

    for opt, name in (("--ns", "NS"), ("--cap", "CP_CAP")):  # diagnostics: ring depth, cap
        if opt in sys.argv:
            setattr(h8, name, int(sys.argv[sys.argv.index(opt) + 1]))
            sys.argv = [a for i, a in enumerate(sys.argv) if a != opt and sys.argv[i - 1] != opt]
    print(f"NS = {h8.NS} CP_CAP = {h8.CP_CAP}", flush=True)
    kl = [tuple(int(v) for v in k.split(":")) for k in
          ([a for a in sys.argv[1:] if not a.startswith("--")] or ["0:0:0:0:0:0:0:0:0:0:3"])]
    if "--nocheck" not in sys.argv:
        for t, short, seed in ((300, True, 1), (149, True, 2), (2048, False, 3), (640, True, 4)):
            inp = inputs(t, 9000, seed=seed, short=short)
            ref = run(*inp, ())
            for k in kl:
                o = run(*inp, k)
                torch.cuda.synchronize()
                print(f"phys t={t} short={short} knobs={k}: equal={torch.equal(o, ref)}", flush=True)
        for bs in (64, 32):
            c = logical_case([2048, 1, 2047, 129, 0, 2048, 64, 1000] * 40, [400, 120, 300], block_size=bs,
                             blocks=900)
            q8, kv, _, _, _, _, stride, slots, lens = c
            ref = sparse_mla_prefill_h8(q8, kv, slots, lens, 0.0722, 1.3)
            base = run_l(c, ())
            for k in kl:
                o = run_l(c, k)
                torch.cuda.synchronize()
                print(f"logical bs={bs} knobs={k}: equal_phys={torch.equal(o, ref)} "
                      f"equal_base={torch.equal(o, base)} base_eq_phys={torch.equal(base, ref)}", flush=True)
    arms = [()] + kl
    if "--logical-only" not in sys.argv:
        inp = inputs(16384, 78 * 1024, seed=5)
        abba(lambda a: run(*inp, a), arms, "phys F16K L78K")
        del inp
    c = logical_case([2048] * 16384, [78 * 1024 // 64], seed=6)
    abba(lambda a: run_l(c, a), arms, "logical F16K L78K")


if __name__ == "__main__":
    main()
