# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tcgen05 latency vs throughput of the sparse-MLA h2 MMA groups (sm_107).

One CTA per SM, one issuing thread. Per case and repetition: clock64 before issue,
the case's MMAs, tcgen05.commit, wait, clock64 after. Latency cases issue one group
per commit; throughput cases (suffix x8) issue 8 groups per commit. Groups (h2 shapes):
QK = 8 TS M128 N64 K64 (Q latent in TMEM) + 1 SS RoPE MMA into one S accumulator;
PV = 4 SS M128 N96 K64 (V^T MN-major, P^T in the B collector) into 4 O^T blocks.
--pollers N:M adds N warps that wait on an mbarrier that never completes while the
MMAs run (M 0: try_wait without a suspend hint, 1: with a 2 us hint, 2: ld.shared spin)
to measure what polling warps cost the tensor pipe.
usage: python pmla_mma_lat.py [--json out] [--pollers 0:0,8:0,...]
"""

import argparse
import json
import statistics

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint8, Uint32, Uint64

from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla import (
    SDESC_SW64,
    SDESC_SW128,
    ld_shared_u32,
    mbar_init,
    mbar_wait,
    st_shared_u32,
    stamp_clock,
    tc_commit,
    tc_fence_after,
    tc_fence_before,
    tc_mma,
    tmem_alloc_exclusive,
    tmem_free_exclusive,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h2 import (
    IDESC_PV,
    IDESC_QK,
    tc_mma_ts,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_sol import tc_mma_cb

OFF_ST = 0  # one 64-key KV stage: 4 latent SW128 atoms + RoPE SW64
OFF_QR = 36864
OFF_P = OFF_QR + 128 * 64
OFF_BAR = OFF_P + 2 * 96 * 64
OFF_TMEM = OFF_BAR + 64
OFF_FLAG = OFF_TMEM + 16
SMEM = OFF_TMEM + 64
C_Q, C_S, C_O = 0, 128, 192
C_S2 = C_O + 320  # last 64 O^T columns (probe only)
REPS = 16

CASES = {
    "qk": 0, "qk_ts_only": 1, "pv_cb": 2, "pv_plain": 3, "qk_then_pv": 4,
    "qk_pv_interleaved": 5, "qk2_interleaved": 6, "ts_single": 7, "qkpv_x8": 8,
    "qk_x8": 9, "pv_x8": 10, "qk_split_acc": 11, "qk_n128_ts_only": 12,
}


@cute.jit
def _qk_mma(base, taddr, j: cutlass.Constexpr, cs: cutlass.Constexpr, first: cutlass.Constexpr):
    """QK step j (0-7 latent K64 TS, 8 RoPE SS) into S at column cs."""
    st = base + Uint32(OFF_ST)
    if cutlass.const_expr(j < 8):
        bd = Uint64(SDESC_SW128) | Uint64((st + Uint32((j // 2) * 8192)) >> Uint32(4))
        tc_mma_ts(taddr + Uint32(cs), taddr + Uint32(C_Q + 16 * j),
                  bd + Uint64((j % 2) * 4), Uint32(IDESC_QK), Uint32(0 if first else 1))
    else:
        ad = Uint64(SDESC_SW64) | Uint64((base + Uint32(OFF_QR)) >> Uint32(4))
        bd = Uint64(SDESC_SW64) | Uint64((st + Uint32(4 * 8192)) >> Uint32(4))
        tc_mma(taddr + Uint32(cs), ad, bd, Uint32(IDESC_QK), Uint32(1))


@cute.jit
def _pv_mma(base, taddr, b: cutlass.Constexpr, cb: cutlass.Constexpr):
    st = base + Uint32(OFF_ST)
    ad = Uint64(SDESC_SW128) | Uint64((st + Uint32(b * 8192)) >> Uint32(4))
    bd = Uint64(SDESC_SW64) | Uint64((base + Uint32(OFF_P)) >> Uint32(4))
    if cutlass.const_expr(cb):
        op = "fill" if b == 0 else ("lastuse" if b == 3 else "use")
        tc_mma_cb(taddr + Uint32(C_O + 96 * b), ad, bd, Uint32(IDESC_PV), Uint32(1), op)
    else:
        tc_mma(taddr + Uint32(C_O + 96 * b), ad, bd, Uint32(IDESC_PV), Uint32(1))


from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (  # noqa: E402
    asm,
)


@asm(Uint32, pure=False)
def try_wait_once(bar, parity, mode: cutlass.Constexpr):
    hint = " , 2000" if mode == 1 else ""
    return ("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2"
            + hint + "; selp.u32 $0, 1, 0, p; }"), "=r,r,r,~{memory}", (Uint32(bar),
                                                                       Uint32(parity))


class Probe:
    def __init__(self, case: int, pollers: int = 0, mode: int = 0):
        self.case, self.pollers, self.mode = case, pollers, mode

    @cute.jit
    def __call__(self, out: Int64, grid: Int32, stream: cuda.CUstream):
        self.kernel(out).launch(grid=(grid, 1, 1), block=(128 + 32 * self.pollers, 1, 1),
                                smem=SMEM, stream=stream)

    @cute.jit
    def group(self, base, taddr):
        c = self.case
        if cutlass.const_expr(c == 0 or c == 9):
            for j in cutlass.range_constexpr(9):
                _qk_mma(base, taddr, j, C_S, j == 0)
        if cutlass.const_expr(c == 1):
            for j in cutlass.range_constexpr(8):
                _qk_mma(base, taddr, j, C_S, j == 0)
        if cutlass.const_expr(c == 2 or c == 10):
            for b in cutlass.range_constexpr(4):
                _pv_mma(base, taddr, b, True)
        if cutlass.const_expr(c == 3):
            for b in cutlass.range_constexpr(4):
                _pv_mma(base, taddr, b, False)
        if cutlass.const_expr(c == 4 or c == 8):
            for j in cutlass.range_constexpr(9):
                _qk_mma(base, taddr, j, C_S, j == 0)
            for b in cutlass.range_constexpr(4):
                _pv_mma(base, taddr, b, True)
        if cutlass.const_expr(c == 5):
            for j in cutlass.range_constexpr(9):
                _qk_mma(base, taddr, j, C_S, j == 0)
                if cutlass.const_expr(j % 2 == 1):
                    _pv_mma(base, taddr, j // 2, False)
        if cutlass.const_expr(c == 6):
            for j in cutlass.range_constexpr(9):
                _qk_mma(base, taddr, j, C_S, j == 0)
                _qk_mma(base, taddr, j, C_S2, j == 0)
        if cutlass.const_expr(c == 7):
            _qk_mma(base, taddr, 0, C_S, True)
        if cutlass.const_expr(c == 11):
            for j in cutlass.range_constexpr(4):
                _qk_mma(base, taddr, j, C_S, j == 0)
                _qk_mma(base, taddr, j + 4, C_S2, j == 0)
            _qk_mma(base, taddr, 8, C_S2, False)
        if cutlass.const_expr(c == 12):
            # N128 (two 64-key subtiles per MMA): S 128 columns at C_O (probe only)
            st = base + Uint32(OFF_ST)
            for j in cutlass.range_constexpr(8):
                bd = Uint64(SDESC_SW128) | Uint64((st + Uint32((j // 2) * 8192)) >> Uint32(4))
                tc_mma_ts(taddr + Uint32(C_O), taddr + Uint32(C_Q + 16 * j),
                          bd + Uint64((j % 2) * 4),
                          Uint32(IDESC_QK + (8 << 17)), Uint32(0 if j == 0 else 1))

    @cute.kernel
    def kernel(self, out: Int64):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        smem = cutlass.utils.SmemAllocator()
        raw = smem.allocate_array(Uint8, SMEM, byte_alignment=1024)
        base = Uint32(raw.toint())
        bar = base + Uint32(OFF_BAR)
        if tid == 0:
            mbar_init(bar, Uint32(1))
            mbar_init(bar + Uint32(8), Uint32(1))
            st_shared_u32(base + Uint32(OFF_FLAG), Uint32(0))
            cute.arch.mbarrier_init_fence()
        if warp == 0:
            tmem_alloc_exclusive(base + Uint32(OFF_TMEM))
        tc_fence_before()
        cute.arch.sync_threads()
        tc_fence_after()
        taddr = ld_shared_u32(base + Uint32(OFF_TMEM))
        if tid == 0:
            o = out + Int64(cta * REPS * 2 * 8)
            for r in range(REPS):
                stamp_clock(o + Int64(r * 16))
                if cutlass.const_expr(self.case >= 8 and self.case <= 10):
                    for _ in cutlass.range_constexpr(8):
                        self.group(base, taddr)
                else:
                    self.group(base, taddr)
                tc_commit(bar)
                mbar_wait(bar, Uint32(r & 1))
                tc_fence_after()
                stamp_clock(o + Int64(r * 16 + 8))
            st_shared_u32(base + Uint32(OFF_FLAG), Uint32(1))
        if warp >= 4:
            while ld_shared_u32(base + Uint32(OFF_FLAG)) == Uint32(0):
                if cutlass.const_expr(self.mode < 2):
                    try_wait_once(bar + Uint32(8), Uint32(0), self.mode)
        tc_fence_before()
        cute.arch.sync_threads()
        if warp == 0:
            tc_fence_after()
            tmem_free_exclusive(taddr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--pollers", default="0:0")
    a = ap.parse_args()
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    cs = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    res = {"sms": sms, "rows": []}
    for name, pol in [(n, p) for n in a.cases.split(",") for p in a.pollers.split(",")]:
        npol, mode = (int(v) for v in pol.split(":"))
        for grid in (1, sms):
            try:
                buf = torch.zeros(grid * REPS * 2, dtype=torch.int64, device=dev)
                f = cute.compile(Probe(CASES[name], npol, mode), Int64(buf.data_ptr()),
                                 Int32(grid), cs)
                f(Int64(buf.data_ptr()), Int32(grid), cs)
                torch.cuda.synchronize()
                f(Int64(buf.data_ptr()), Int32(grid), cs)
                torch.cuda.synchronize()
                x = buf.view(grid, REPS, 2).cpu()
                d = (x[:, 2:, 1] - x[:, 2:, 0]).flatten().tolist()
                per = 8 if CASES[name] in (8, 9, 10) else 1
                row = dict(case=name, pollers=pol, grid=grid, clk=round(statistics.median(d) / per, 1),
                           clk_p90=round(sorted(d)[int(0.9 * len(d))] / per, 1))
            except Exception as e:  # noqa: BLE001
                row = dict(case=name, pollers=pol, grid=grid, error=repr(e)[:400])
            print(row, flush=True)
            res["rows"].append(row)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
