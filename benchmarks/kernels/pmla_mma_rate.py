# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tcgen05 FP8 (kind::f8f6f4, 2x-K K64) MMA issue-rate probe on sm_107, one CTA per SM.

Back-to-back M=128 MMAs (K=512 per pass, 8 x K64) from one thread, A and B both in
shared memory (ss) or A in TMEM (ts), B K-major or MN-major, N in {64, 96, 128, 192,
256}. Reports FLOP per SM-clock. The ss/ts gap at a given N measures whether operand
fetch from shared memory limits the tensor core (the sparse-MLA design question:
heads on M with Q/P in TMEM, or keys on M with both operands in SMEM).
usage: python pmla_mma_rate.py [--json out]
"""

import argparse
import json

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32, Uint64
from cutlass.cute.runtime import from_dlpack

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    alloc,
    asm,
    commit_asm,
    fence_tmem,
    free_tmem,
    shared_address,
    wait,
)

A_ATOM = 128 * 128  # 128 rows x 128 B


@asm()
def mma_ss(d, a, b, instruction):
    code = "{ .reg .pred p; setp.eq.b32 p,0,0; "
    code += "tcgen05.mma.cta_group::1.kind::f8f6f4 [$0],$1,$2,$3,p; }"
    return code, "r,l,l,r", (Uint32(d), Uint64(a), Uint64(b), Uint32(instruction))


@asm()
def mma_ts(d, a, b, instruction):
    code = "{ .reg .pred p; setp.eq.b32 p,0,0; "
    code += "tcgen05.mma.cta_group::1.kind::f8f6f4 [$0],[$1],$2,$3,p; }"
    return code, "r,r,l,r", (Uint32(d), Uint32(a), Uint64(b), Uint32(instruction))


@asm(Int64)
def clock64():
    return "mov.u64 $0, %clock64;", "=l,~{memory}", ()


@cute.jit
def desc(pointer, lbo: cutlass.Constexpr[int]):
    address = shared_address(pointer)
    value = Uint64((address & 262143) >> 4) | Uint64(1) << 46
    value |= Uint64(lbo >> 4) << 16 | Uint64(64) << 32 | Uint64(2) << 61
    return value


class Probe:
    def __init__(self, mode, n, b_mn, iters):
        self.mode, self.n, self.b_mn, self.iters = mode, n, b_mn, iters
        self.b_atom = n * 128

    @cute.jit
    def __call__(self, out: cute.Tensor, sms: Int32, stream):
        self.kernel(out).launch(grid=(sms, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, out: cute.Tensor):
        smem = cutlass.utils.SmemAllocator()
        abuf = alloc(smem, cutlass.Uint8, 4 * A_ATOM, 1024)
        bbuf = alloc(smem, cutlass.Uint8, 4 * 256 * 128, 1024)
        bars = alloc(smem, Uint64, 2, 8)
        tslot = alloc(smem, Uint32, 1, 4)
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        if tid == 0:
            cute.arch.mbarrier_init(bars.iterator, 1)
            cute.arch.mbarrier_init_fence()
        if tid < 32:
            cute.arch.alloc_tmem(512, tslot.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        tmem = tslot[0]
        instr = Uint32(1 << 4 | (self.n >> 3) << 17 | 8 << 24 | 1 << 29)
        if cutlass.const_expr(self.b_mn):
            instr = instr | Uint32(1 << 16)
        if cutlass.const_expr(self.mode == "ss_amn"):
            instr = instr | Uint32(1 << 15)
        t0 = Int64(0)
        t1 = Int64(0)
        if tid == 0:
            fence_tmem()
            t0 = clock64()
            for it in range(self.iters):
                for k in cutlass.range_constexpr(8):
                    bd = desc(bbuf.iterator + (k // 2) * self.b_atom, 8192) + Uint64(
                        4 * (k % 2)
                    )
                    if cutlass.const_expr(self.mode != "ts"):
                        ad = desc(abuf.iterator + (k // 2) * A_ATOM, 0) + Uint64(
                            4 * (k % 2)
                        )
                        mma_ss(tmem, ad, bd, instr)
                    else:
                        mma_ts(tmem, tmem + Uint32(256 + 16 * k), bd, instr)
            commit_asm(bars.iterator)
            wait(bars.iterator, 0, scope="cta")
            t1 = clock64()
            out[cta] = t1 - t0
        fence_tmem(before=True)
        cute.arch.sync_threads()
        if tid < 32:
            fence_tmem()
            free_tmem(tmem, 512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--iters", type=int, default=2000)
    a = ap.parse_args()
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    out = torch.zeros(sms, dtype=torch.int64, device=dev)
    stream = torch.cuda.current_stream()
    from cuda.bindings.driver import CUstream

    cs = CUstream(stream.cuda_stream)
    res = {"sms": sms, "rows": []}
    for mode in ("ts", "ss", "ss_amn"):
        for b_mn in (False, True):
            for n in (64, 96, 128, 192, 256):
                try:
                    p = Probe(mode, n, b_mn, a.iters)
                    f = cute.compile(p, from_dlpack(out), Int32(sms), cs)
                    for _ in range(2):
                        f(from_dlpack(out), Int32(sms), cs)
                    torch.cuda.synchronize()
                    cyc = out.float()
                    flops = 2 * 128 * n * 64 * 8 * a.iters
                    row = dict(
                        mode=mode, b_mn=b_mn, n=n,
                        flop_per_clk_med=round(flops / cyc.median().item(), 1),
                        flop_per_clk_min=round(flops / cyc.max().item(), 1),
                        smem_B_per_clk_med=round(
                            (n * 64 + (128 * 64 if mode != "ts" else 0)) * 8 * a.iters
                            / cyc.median().item(), 1),
                    )
                except Exception as e:  # noqa: BLE001
                    row = dict(mode=mode, b_mn=b_mn, n=n, error=repr(e)[:300])
                print(row, flush=True)
                res["rows"].append(row)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
