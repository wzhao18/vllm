# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Do TMA writes into shared memory steal tcgen05 operand-fetch bandwidth on sm_107?

One CTA per SM. Thread 0 issues back-to-back FP8 M=128 N=128 K64 MMAs (A from TMEM:
"ts", only B read from SMEM, 128 B/clk at peak; or "ss"). Thread 32 issues 16 KB
cp.async.bulk copies from an L2-resident buffer into a 64 KB SMEM ring (batches of 4,
one mbarrier). Modes: mma alone, copy alone, both. Reports MMA FLOP/clk and copy
B/clk (per SM) in each mode, plus tcgen05.cp (SMEM -> TMEM) throughput alone.
usage: python pmla_smem_contention.py [--json out]
"""

import argparse
import json

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32, Uint64

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    alloc,
    asm,
    commit_asm,
    copy_g2s,
    fence_tmem,
    free_tmem,
    shared_address,
    wait,
)

ATOM = 128 * 128


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


@asm()
def tmem_cp(column, descriptor):
    code = "tcgen05.cp.cta_group::1.128x256b [$0],$1;"
    return code, "r,l", (Uint32(column), Uint64(descriptor))


@asm()
def arrive_expect(bar, nbytes):
    code = "mbarrier.arrive.expect_tx.shared::cta.b64 _,[$0],$1;"
    return code, "r,r", (bar, Int32(nbytes))


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
    def __init__(self, mode, mma_iters, copies):
        self.mode, self.mma_iters, self.copies = mode, mma_iters, copies

    @cute.jit
    def __call__(self, out: Int64, src: Int64, sms: Int32, stream: cuda.CUstream):
        self.kernel(out, src, sms).launch(grid=(sms, 1, 1), block=(64, 1, 1),
                                          stream=stream)

    @cute.kernel
    def kernel(self, out: Int64, src: Int64, sms: Int32):
        smem = cutlass.utils.SmemAllocator()
        abuf = alloc(smem, cutlass.Uint8, 4 * ATOM, 1024)  # ss A operand
        bbuf = alloc(smem, cutlass.Uint8, 4 * ATOM, 1024)  # B operand
        ring = alloc(smem, cutlass.Uint8, 4 * ATOM, 1024)  # copy destination
        bars = alloc(smem, Uint64, 2, 8)
        tslot = alloc(smem, Uint32, 1, 4)
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        optr = cute.make_ptr(Int64, out, cute.AddressSpace.gmem, assumed_align=8)
        o = cute.make_tensor(optr, cute.make_layout(1 << 20))
        if tid == 0:
            cute.arch.mbarrier_init(bars.iterator, 1)
            cute.arch.mbarrier_init(bars.iterator + 1, 1)
            cute.arch.mbarrier_init_fence()
        if tid < 32:
            cute.arch.alloc_tmem(512, tslot.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        tmem = tslot[0]
        instr = Uint32(1 << 4 | (128 >> 3) << 17 | 8 << 24 | 1 << 29)
        if tid == 0:
            fence_tmem()
            t0 = clock64()
            if cutlass.const_expr(self.mode in ("mma", "both", "both_ss", "mma_ss")):
                for it in range(self.mma_iters):
                    for k in cutlass.range_constexpr(8):
                        bd = desc(bbuf.iterator + (k // 2) * ATOM, 8192) + Uint64(
                            4 * (k % 2))
                        if cutlass.const_expr(self.mode in ("both_ss", "mma_ss")):
                            ad = desc(abuf.iterator + (k // 2) * ATOM, 0) + Uint64(
                                4 * (k % 2))
                            mma_ss(tmem, ad, bd, instr)
                        else:
                            mma_ts(tmem, tmem + Uint32(256 + 16 * k), bd, instr)
            if cutlass.const_expr(self.mode == "tmemcp"):
                for it in range(self.mma_iters):
                    for k in cutlass.range_constexpr(8):
                        sd = desc(bbuf.iterator + (k // 2) * ATOM, 8192) + Uint64(
                            4 * (k % 2))
                        tmem_cp(tmem + Uint32(256 + 16 * k), sd)
            commit_asm(bars.iterator)
            wait(bars.iterator, 0, scope="cta")
            o[cta] = clock64() - t0
        if tid == 32:
            t0 = clock64()
            if cutlass.const_expr(self.mode in ("copy", "both", "both_ss")):
                sptr = cute.make_ptr(cutlass.Uint8, src, cute.AddressSpace.gmem,
                                     assumed_align=128)
                for c in range(self.copies):
                    arrive_expect(bars.iterator + 1, 4 * ATOM)
                    for q in cutlass.range_constexpr(4):
                        off = ((c * 4 + q) % 256) * ATOM
                        copy_g2s(ring.iterator + q * ATOM, sptr + off, ATOM,
                                 bars.iterator + 1)
                    wait(bars.iterator + 1, c % 2, scope="cta")
            o[sms + cta] = clock64() - t0
        fence_tmem(before=True)
        cute.arch.sync_threads()
        if tid < 32:
            fence_tmem()
            free_tmem(tmem, 512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    out = torch.zeros(2 * sms, dtype=torch.int64, device=dev)
    src = torch.randint(0, 255, (256 * ATOM,), dtype=torch.uint8, device=dev)  # 4 MB
    cs = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    res = {"sms": sms, "rows": []}
    I, C = 2000, 1000
    for mode in ("mma", "mma_ss", "copy", "both", "both_ss", "tmemcp"):
        try:
            p = Probe(mode, I, C)
            args = (Int64(out.data_ptr()), Int64(src.data_ptr()), Int32(sms), cs)
            f = cute.compile(p, *args)
            for _ in range(2):
                f(*args)
            torch.cuda.synchronize()
            m = out[:sms].float().median().item()
            c = out[sms:].float().median().item()
            row = dict(mode=mode, mma_cyc=m, copy_cyc=c)
            if mode in ("mma", "both", "mma_ss", "both_ss"):
                row["mma_flop_per_clk"] = round(2 * 128 * 128 * 64 * 8 * I / m, 1)
            if mode in ("copy", "both", "both_ss"):
                row["copy_B_per_clk"] = round(C * 4 * ATOM / c, 1)
            if mode == "tmemcp":
                row["tmemcp_B_per_clk"] = round(I * 8 * 128 * 32 / m, 1)
        except Exception as e:  # noqa: BLE001
            row = dict(mode=mode, error=repr(e)[:400])
        print(row, flush=True)
        res["rows"].append(row)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
