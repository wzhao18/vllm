# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather floor of prefill sparse MLA on sm_107: multicast pairs and row alignment.

Like pmla_gather_tma.py (persistent, 16 tiles of 128 top-k rows per token, TMA
tile::gather4 into a ring, no math), plus:
  mc=1: CTA pairs (cluster 2) work on the same token; each CTA gathers half of
        the tile's quads with .multicast::cluster to both CTAs (L2 read once,
        both SMs receive the full tile; the layout a 2-CTA V-split kernel needs).
  pitch=640: rows stored at a 128 B aligned pitch (576 B used) instead of 576.
Reported: unique-bytes rate (tokens x 2048 x 576 / time) and delivered rate.
usage: python pmla_gather_mc.py --tokens 16384 [--json out]
"""

import argparse
import json
import statistics
import sys

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32, Uint64

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from benchmark_kimi_k3_prefill_sparse_mla import make_layer  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import (  # noqa: E402
    tensormap,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (  # noqa: E402
    alloc,
    asm,
    load_words,
    wait,
)

KATOM = 128 * 128
KROPE = 4 * KATOM
KVBYTES = KROPE + 128 * 64


@asm()
def gather4(dst, desc, bar, col, r0, r1, r2, r3):
    code = (
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes [$0],[$1,{$3,$4,$5,$6,$7}],[$2];"
    )
    ops = (dst, desc, bar, Int32(col), Int32(r0), Int32(r1), Int32(r2), Int32(r3))
    return code, "r,l,r,r,r,r,r,r", ops


@asm()
def gather4_mc(dst, desc, bar, col, r0, r1, r2, r3):
    code = (
        "{ .reg .b16 m; mov.b16 m, 3; "
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.multicast::cluster "
        "[$0],[$1,{$3,$4,$5,$6,$7}],[$2], m; }"
    )
    ops = (dst, desc, bar, Int32(col), Int32(r0), Int32(r1), Int32(r2), Int32(r3))
    return code, "r,l,r,r,r,r,r,r", ops


@asm()
def arrive_expect(bar, nbytes):
    code = "mbarrier.arrive.expect_tx.shared::cta.b64 _,[$0],$1;"
    return code, "r,r", (bar, Int32(nbytes))


@asm()
def arrive(bar):
    return "mbarrier.arrive.shared::cta.b64 _,[$0];", "r", (bar,)


@asm()
def arrive_peer(bar, peer):
    code = ("{ .reg .b32 a; mapa.shared::cluster.u32 a, $0, $1; "
            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [a]; }")
    return code, "r,r,~{memory}", (bar, Uint32(peer))


@asm(Uint32)
def cluster_rank():
    return "mov.u32 $0, %cluster_ctarank;", "=r", ()


@asm()
def cluster_sync():
    return ("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;",
            "~{memory}", ())


@asm(Int64)
def clock64():
    return "mov.u64 $0, %globaltimer;", "=l,~{memory}", ()


class Gather:
    def __init__(self, warps, stages, mc):
        self.warps, self.stages, self.mc = warps, stages, mc

    @cute.jit
    def __call__(self, kv: Int64, rows: Int32, pitch: Int32, idx: Int64, tokens: Int32,
                 out: Int64, grid: Int32, stream: cuda.CUstream):
        u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
        l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        sw = cuda.CUtensorMapSwizzle
        main = tensormap.encode_map(kv, u8, (576, rows), (pitch,), (128, 1),
                                    sw.CU_TENSOR_MAP_SWIZZLE_128B, l2)
        rope = tensormap.encode_map(kv, u8, (576, rows), (pitch,), (64, 1),
                                    sw.CU_TENSOR_MAP_SWIZZLE_64B, l2)
        if cutlass.const_expr(self.mc):
            self.kernel(main, rope, idx, tokens, out).launch(
                grid=(grid, 1, 1), block=(32 * (self.warps + 1), 1, 1),
                cluster=(2, 1, 1), stream=stream)
        else:
            self.kernel(main, rope, idx, tokens, out).launch(
                grid=(grid, 1, 1), block=(32 * (self.warps + 1), 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, main: tensormap.GridMap, rope: tensormap.GridMap, idx: Int64,
               tokens: Int32, out: Int64):
        smem = cutlass.utils.SmemAllocator()
        ring = alloc(smem, cutlass.Uint8, self.stages * KVBYTES, 1024)
        bars = alloc(smem, Uint64, 2 * self.stages, 8)
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        warp, lane = tid // 32, tid % 32
        full, empty = bars.iterator, bars.iterator + self.stages
        nc: cutlass.Constexpr = 2 if self.mc else 1
        if tid == 0:
            for s in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(full + s, 1)
                cute.arch.mbarrier_init(empty + s, nc)
            cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.mc):
            cluster_sync()
        else:
            cute.arch.sync_threads()
        rank = Uint32(0)
        if cutlass.const_expr(self.mc):
            rank = cluster_rank()
        dmain = tensormap.map_address(main)
        drope = tensormap.map_address(rope)
        iptr = cute.make_ptr(Int32, idx, cute.AddressSpace.gmem, assumed_align=16)
        t0 = clock64()
        g = Int32(0)
        t = cta // nc
        step = grid // nc
        quads: cutlass.Constexpr = 32 // nc // self.warps  # quads per warp
        while t < tokens:
            for j in range(16):
                s = g % self.stages
                ph = (g // self.stages) % 2
                if warp < self.warps:
                    if lane == 0:
                        wait(empty + s, ph ^ 1, scope="cta")
                        if warp == 0:
                            arrive_expect(full + s, KVBYTES)
                    cute.arch.sync_warp()
                    if lane < quads:
                        q = Int32(rank) * (32 // nc) + warp * quads + lane
                        r = load_words(iptr + (t * 2048 + j * 128 + 4 * q), count=4)
                        st = ring.iterator + s * KVBYTES
                        for a in cutlass.range_constexpr(4):
                            if cutlass.const_expr(self.mc):
                                gather4_mc(st + a * KATOM + q * 512, dmain, full + s,
                                           128 * a, r[0], r[1], r[2], r[3])
                            else:
                                gather4(st + a * KATOM + q * 512, dmain, full + s,
                                        128 * a, r[0], r[1], r[2], r[3])
                        if cutlass.const_expr(self.mc):
                            gather4_mc(st + KROPE + q * 256, drope, full + s, 512,
                                       r[0], r[1], r[2], r[3])
                        else:
                            gather4(st + KROPE + q * 256, drope, full + s, 512,
                                    r[0], r[1], r[2], r[3])
                else:
                    if lane == 0:
                        wait(full + s, ph, scope="cta")
                        arrive(empty + s)
                        if cutlass.const_expr(self.mc):
                            arrive_peer(empty + s, rank ^ Uint32(1))
                g += 1
            t += step
        if cutlass.const_expr(self.mc):
            cluster_sync()
        else:
            cute.arch.sync_threads()
        if tid == 0:
            optr = cute.make_ptr(Int64, out, cute.AddressSpace.gmem, assumed_align=8)
            o = cute.make_tensor(optr, cute.make_layout(1 << 20))
            o[cta] = clock64() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--cases", default="16:0:576,16:1:576,8:1:576,16:0:640,16:1:640")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    F = a.tokens
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    grid = sms - sms % 2
    layers = [make_layer(F, 72900, 3, dev, 17 + i) for i in range(3)]
    padded = {}
    out = torch.zeros(2 * sms, dtype=torch.int64, device=dev)
    cs = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    res = {"tokens": F, "sms": sms, "grid": grid, "rows": []}
    nbytes = F * 2048 * 576
    for case in a.cases.split(","):
        warps, mc, pitch = (int(x) for x in case.split(":"))
        stages = 3 if mc else 2
        try:
            if pitch != 576 and pitch not in padded:
                padded[pitch] = []
                for kv, idx in layers:
                    k2 = torch.zeros(kv.numel() // 576, pitch, dtype=torch.uint8,
                                     device=dev)
                    k2[:, :576] = kv.view(torch.uint8).view(-1, 576)
                    padded[pitch].append((k2, idx))
            lay = layers if pitch == 576 else padded[pitch]
            g = Gather(warps, stages, bool(mc))

            def args(i):
                kv, idx = lay[i % 3]
                return (Int64(kv.data_ptr()), Int32(kv.numel() // pitch), Int32(pitch),
                        Int64(idx.data_ptr()), Int32(F), Int64(out.data_ptr()),
                        Int32(grid), cs)

            f = cute.compile(g, *args(0))
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
            row = dict(warps=warps, mc=mc, pitch=pitch, stages=stages, us=round(t, 1),
                       unique_TBps=round(nbytes / t / 1e6, 2),
                       delivered_TBps=round(nbytes * (2 if mc else 1) / t / 1e6, 2),
                       cta_ns_med=int(out[:grid].float().median().item()))
        except Exception as e:  # noqa: BLE001
            row = dict(case=case, error=repr(e)[:600])
        print(row, flush=True)
        res["rows"].append(row)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
