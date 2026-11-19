# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed TMA gather4 + LDG row gather of prefill sparse MLA on sm_107 (no math).

Persistent grid (one CTA per SM), the h6 stage shape: per token 32 stages of 64
top-k rows (576 B FP8: four 128 B SW128 boxes + one 64 B SW64 rope box). Rows
[0, RG) of a stage come from G gather4 warps (one quad per lane), rows [RG, 64)
from L LDG warps (16 B ld.global.nc per lane, all loads of a stage issued before
the swizzled st.shared, then fence.proxy.async + one arrive per warp). A consumer
warp releases each stage when it lands. Measures delivered L2 -> SMEM TB/s, to
test whether gather4 in-flight capacity per SM bounds the gather floor.
usage: python pmla_gather_mix.py --tokens 16384 [--cfg G:L:RG:NS,...] [--json out]
"""

import argparse
import json
import statistics
import sys

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from benchmark_kimi_k3_prefill_sparse_mla import make_layer  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import (  # noqa: E402
    tensormap,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (  # noqa: E402
    asm,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla import (  # noqa: E402
    fence_async_smem,
    gather4,
    ld_global_v4_s32,
    mbar_arrive,
    mbar_arrive_tx,
    mbar_init,
    mbar_wait,
    st_shared_v4,
)

STAGE = 4 * 64 * 128 + 64 * 64  # 36864


@asm(Int32)
def ld_global_s32(ptr):
    return "ld.global.nc.b32 $0, [$1];", "=r,l", (Int64(ptr),)


@asm()
def ldg_v4(ptr):
    return ("ld.global.nc.L1::no_allocate.v4.b32 {$0,$1,$2,$3}, [$4];",
            "=r,=r,=r,=r,l", (Int64(ptr),), (Uint32,) * 4)


class Mix:
    def __init__(self, g, nl, rg, ns):
        self.g, self.nl, self.rg, self.ns = g, nl, rg, ns
        self.rl = (64 - rg) // nl if nl else 0
        assert nl == 0 or self.rl * nl == 64 - rg
        assert self.rl <= 32
        self.npl = (self.rl * 36 + 31) // 32
        self.threads = 32 * (g + nl + 1)
        self.smem = ns * STAGE + 1024 + 2 * ns * 8

    @cute.jit
    def __call__(self, kv: Int64, rows: Int32, idx: Int64, tokens: Int32, out: Int64,
                 grid: Int32, stream: cuda.CUstream):
        u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
        l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        sw = cuda.CUtensorMapSwizzle
        main = tensormap.encode_map(kv, u8, (576, rows), (576,), (128, 1),
                                    sw.CU_TENSOR_MAP_SWIZZLE_128B, l2)
        rope = tensormap.encode_map(kv, u8, (576, rows), (576,), (64, 1),
                                    sw.CU_TENSOR_MAP_SWIZZLE_64B, l2)
        self.kernel(main, rope, kv, idx, tokens, out).launch(
            grid=(grid, 1, 1), block=(self.threads, 1, 1), smem=self.smem, stream=stream)

    @cute.kernel
    def kernel(self, main: tensormap.GridMap, rope: tensormap.GridMap, kv: Int64,
               idx: Int64, tokens: Int32, out: Int64):
        smem = cutlass.utils.SmemAllocator()
        raw = smem.allocate_array(cutlass.Uint8, self.smem, byte_alignment=1024)
        base = Uint32(raw.toint())
        bars = base + Uint32(self.ns * STAGE)
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        nfull = (1 if self.g > 0 else 0) + self.nl
        if tid == 0:
            for s in cutlass.range_constexpr(self.ns):
                mbar_init(bars + Uint32(8 * s), Uint32(nfull))
                mbar_init(bars + Uint32(8 * (self.ns + s)), Uint32(1))
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        dmain = tensormap.map_address(main)
        drope = tensormap.map_address(rope)
        n_mine = (tokens - cta + grid - 1) // grid
        total = n_mine * 32
        if warp < self.g:
            quads = (self.rg // 4 - warp + self.g - 1) // self.g
            q = warp + self.g * lane
            if lane >= quads:
                q = warp
            for gi in cutlass.range(total, unroll=1):
                t = cta + (gi // 32) * grid
                j = gi % 32
                s = gi % self.ns
                st = base + Uint32(s) * Uint32(STAGE)
                full = bars + Uint32(8) * Uint32(s)
                c0, c1, c2, c3 = ld_global_v4_s32(idx + Int64(4) * (Int64(t) * 2048 + j * 64 + 4 * q))
                mbar_wait(bars + Uint32(8) * Uint32(self.ns + s), Uint32(((gi // self.ns) & 1) ^ 1))
                if warp == 0 and lane == 0:
                    mbar_arrive_tx(full, Uint32(self.rg * 576))
                if lane < quads:
                    for a in cutlass.range_constexpr(4):
                        gather4(st + Uint32(a * 8192) + Uint32(q) * Uint32(512), dmain, full,
                                128 * a, c0, c1, c2, c3)
                    gather4(st + Uint32(4 * 8192) + Uint32(q) * Uint32(256), drope, full,
                            512, c0, c1, c2, c3)
        elif warp < self.g + self.nl:
            lw = warp - self.g
            r0 = self.rg + lw * self.rl
            vals = cute.make_rmem_tensor((4 * self.npl,), Uint32)
            for gi in cutlass.range(total, unroll=1):
                t = cta + (gi // 32) * grid
                j = gi % 32
                s = gi % self.ns
                st = base + Uint32(s) * Uint32(STAGE)
                myidx = Int32(0)
                if lane < self.rl:
                    myidx = ld_global_s32(idx + Int64(4) * (Int64(t) * 2048 + j * 64 + r0 + lane))
                mbar_wait(bars + Uint32(8) * Uint32(self.ns + s), Uint32(((gi // self.ns) & 1) ^ 1))
                for i in cutlass.range_constexpr(self.npl):
                    c = lane + 32 * i
                    rr = c // 36
                    if rr >= self.rl:
                        rr = Int32(self.rl - 1)
                    k = c % 36
                    ri = cute.arch.shuffle_sync(myidx, rr)
                    v = ldg_v4(kv + Int64(ri) * Int64(576) + Int64(k * 16))
                    for e in cutlass.range_constexpr(4):
                        vals[4 * i + e] = v[e]
                for i in cutlass.range_constexpr(self.npl):
                    c = lane + 32 * i
                    if c < self.rl * 36:
                        rr = c // 36
                        k = c % 36
                        r = r0 + rr
                        dst = st + Uint32(4 * 8192) + Uint32(r * 64) + (
                            (Uint32(k - 32) ^ ((Uint32(r) >> Uint32(1)) & Uint32(3))) << Uint32(4))
                        if k < 32:
                            dst = st + Uint32((k // 8) * 8192) + Uint32(r * 128) + (
                                (Uint32(k % 8) ^ (Uint32(r) & Uint32(7))) << Uint32(4))
                        st_shared_v4(dst, vals[4 * i], vals[4 * i + 1], vals[4 * i + 2],
                                     vals[4 * i + 3])
                fence_async_smem()
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(bars + Uint32(8) * Uint32(s))
        else:
            if lane == 0:
                for gi in cutlass.range(total, unroll=1):
                    s = gi % self.ns
                    mbar_wait(bars + Uint32(8) * Uint32(s), Uint32((gi // self.ns) & 1))
                    mbar_arrive(bars + Uint32(8) * Uint32(self.ns + s))
        cute.arch.sync_threads()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--cfg", default="16:0:64:4,0:16:0:4,0:24:0:4,8:8:32:4,12:8:48:4,"
                                     "8:16:32:4,16:8:48:4,16:12:40:4,9:9:28:6")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    F = a.tokens
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    layers = [make_layer(F, 72900, 3, dev, 17 + i) for i in range(3)]
    out = torch.zeros(2 * sms, dtype=torch.int64, device=dev)
    cs = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    res = {"tokens": F, "sms": sms, "rows": []}
    nbytes = F * 2048 * 576
    for cfg in a.cfg.split(","):
        g, nl, rg, ns = (int(v) for v in cfg.split(":"))
        try:
            m = Mix(g, nl, rg, ns)
            kv0, idx0 = layers[0]
            args = (Int64(kv0.data_ptr()), Int32(kv0.numel() // 576), Int64(idx0.data_ptr()),
                    Int32(F), Int64(out.data_ptr()), Int32(sms), cs)
            f = cute.compile(m, *args)
            times = []
            for i in range(9):
                kv, idx = layers[i % 3]
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                f(Int64(kv.data_ptr()), Int32(kv.numel() // 576), Int64(idx.data_ptr()),
                  Int32(F), Int64(out.data_ptr()), Int32(sms), cs)
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e) * 1e3)
            t = statistics.median(times[1:])
            row = dict(cfg=cfg, us=round(t, 1), TBps=round(nbytes / t / 1e6, 2))
        except Exception as ex:  # noqa: BLE001
            row = dict(cfg=cfg, error=repr(ex)[:400])
        print(row, flush=True)
        res["rows"].append(row)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
