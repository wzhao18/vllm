# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather floor under concurrent tensor-core shared-memory operand reads (sm_107).

Like pmla_gather_tma.py (persistent, one CTA per SM, 16 tiles of 128 random
576 B rows per token, 16 tile::gather4 warps, 3 stages), plus an MMA warp that,
for every landed tile, issues ``mmas`` SS tcgen05 FP8 MMAs (M128 N96 K64: 14 KB of
A + B operand reads each) on that stage before releasing it, and optionally a
TS variant (A from TMEM, B 6 KB per MMA). Reports the gather rate per mode, i.e.
how much the operand reads of a sparse-MLA tile slow the row gather.
usage: python pmla_gather_mma.py --tokens 16384 [--json out]
"""

import argparse
import json
import statistics
import sys

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint8, Uint32, Uint64

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from benchmark_kimi_k3_prefill_sparse_mla import make_layer  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import (  # noqa: E402
    tensormap,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla import (  # noqa: E402
    IDESC_QK,
    SDESC_SW128,
    gather4,
    ld_global_v4_s32,
    ld_shared_u32,
    mbar_arrive,
    mbar_arrive_tx,
    mbar_init,
    mbar_wait,
    tc_commit,
    tc_fence_after,
    tc_fence_before,
    tc_mma,
    tmem_alloc_exclusive,
    tmem_free_exclusive,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h2 import (  # noqa: E402
    tc_mma_ts,
)

KATOM = 128 * 128
STAGE = 4 * KATOM + 128 * 64
NS = 3
WARPS = 16
OFF_B = 0  # 4 x [96 x 128 B] SW128 "Q^T" (B operand)
OFF_BAR = 4 * 96 * 128
OFF_TMEM = OFF_BAR + 16 * 8
OFF_KV = (OFF_TMEM + 16 + 1023) // 1024 * 1024
SMEM = OFF_KV + NS * STAGE
IDESC_N96 = (1 << 4) | (96 >> 3 << 17) | (128 >> 4 << 24) | (1 << 29)


class Probe:
    def __init__(self, mmas, ts):
        self.mmas, self.ts = mmas, ts

    @cute.jit
    def __call__(self, kv: Int64, rows: Int32, idx: Int64, tokens: Int32, grid: Int32,
                 stream: cuda.CUstream):
        u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
        l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        sw = cuda.CUtensorMapSwizzle
        main = tensormap.encode_map(kv, u8, (576, rows), (576,), (128, 1),
                                    sw.CU_TENSOR_MAP_SWIZZLE_128B, l2)
        rope = tensormap.encode_map(kv, u8, (576, rows), (576,), (64, 1),
                                    sw.CU_TENSOR_MAP_SWIZZLE_64B, l2)
        self.kernel(main, rope, idx, tokens).launch(
            grid=(grid, 1, 1), block=(32 * (WARPS + 1), 1, 1), smem=SMEM, stream=stream)

    @cute.kernel
    def kernel(self, main: tensormap.GridMap, rope: tensormap.GridMap, idx: Int64,
               tokens: Int32):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        smem = cutlass.utils.SmemAllocator()
        raw = smem.allocate_array(Uint8, SMEM, byte_alignment=1024)
        base = Uint32(raw.toint())
        if tid == 0:
            for s in cutlass.range_constexpr(NS):
                mbar_init(base + Uint32(OFF_BAR + 8 * s), Uint32(1))
                mbar_init(base + Uint32(OFF_BAR + 8 * (8 + s)), Uint32(1))
            cute.arch.mbarrier_init_fence()
        if warp == 0:
            tmem_alloc_exclusive(base + Uint32(OFF_TMEM))
        tc_fence_before()
        cute.arch.sync_threads()
        tc_fence_after()
        taddr = ld_shared_u32(base + Uint32(OFF_TMEM))
        dmain = tensormap.map_address(main)
        drope = tensormap.map_address(rope)
        g = Int32(0)
        t = cta
        while t < tokens:
            for j in range(16):
                s = g % NS
                ph = (g // NS) % 2
                st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE)
                full = base + Uint32(OFF_BAR + 8 * s)
                empty = base + Uint32(OFF_BAR + 8 * (8 + s))
                if warp < WARPS:
                    q = warp * 2 + (lane & 1)
                    r = ld_global_v4_s32(idx + Int64((t * 2048 + j * 128 + 4 * q) * 4))
                    if lane == 0:
                        mbar_wait(empty, Uint32(ph ^ 1))
                        if warp == 0:
                            mbar_arrive_tx(full, Uint32(STAGE))
                    cute.arch.sync_warp()
                    if lane < 2:
                        for a in cutlass.range_constexpr(4):
                            gather4(st + Uint32(a * KATOM) + Uint32(q) * Uint32(512), dmain,
                                    full, 128 * a, r[0], r[1], r[2], r[3])
                        gather4(st + Uint32(4 * KATOM) + Uint32(q) * Uint32(256), drope,
                                full, 512, r[0], r[1], r[2], r[3])
                else:
                    if lane == 0:
                        mbar_wait(full, Uint32(ph))
                        tc_fence_after()
                        for m in cutlass.range_constexpr(self.mmas):
                            a = m % 8
                            ad = Uint64(SDESC_SW128) | Uint64(
                                (st + Uint32((a // 2) * KATOM)) >> Uint32(4))
                            bd = Uint64(SDESC_SW128) | Uint64(
                                (base + Uint32(OFF_B + (a // 2) * 96 * 128)) >> Uint32(4))
                            if cutlass.const_expr(self.ts):
                                tc_mma_ts(taddr + Uint32(128), taddr + Uint32(16 * a),
                                          bd + Uint64((a % 2) * 4), Uint32(IDESC_N96),
                                          Uint32(1))
                            else:
                                tc_mma(taddr, ad + Uint64((a % 2) * 4),
                                       bd + Uint64((a % 2) * 4), Uint32(IDESC_QK), Uint32(1))
                        tc_commit(empty)
                    cute.arch.sync_warp()
                g += 1
            t += grid
        tc_fence_before()
        cute.arch.sync_threads()
        if warp == 0:
            tc_fence_after()
            tmem_free_exclusive(taddr)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--cases", default="0:0,8:0,17:0,34:0,9:1,18:1")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    F = a.tokens
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    layers = [make_layer(F, 72900, 3, dev, 17 + i) for i in range(3)]
    cs = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    res = {"tokens": F, "sms": sms, "rows": []}
    nbytes = F * 2048 * 576
    for case in a.cases.split(","):
        mmas, ts = (int(x) for x in case.split(":"))
        try:
            p = Probe(mmas, bool(ts))

            def args(i):
                kv, idx = layers[i % 3]
                return (Int64(kv.data_ptr()), Int32(kv.numel() // 576), Int64(idx.data_ptr()),
                        Int32(F), Int32(sms), cs)

            f = cute.compile(p, *args(0))
            times = []
            for i in range(9):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                f(*args(i))
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e) * 1e3)
            t = statistics.median(times[1:])
            opb = (14 if not ts else 6) * 1024 * mmas
            row = dict(mmas=mmas, ts=ts, us=round(t, 1), TBps=round(nbytes / t / 1e6, 2),
                       operand_KB_per_tile=opb // 1024)
        except Exception as e:  # noqa: BLE001
            row = dict(case=case, error=repr(e)[:600])
        print(row, flush=True)
        res["rows"].append(row)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
