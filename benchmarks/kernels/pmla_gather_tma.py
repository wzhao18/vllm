# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TMA tile::gather4 gather floor of prefill sparse MLA on sm_107.

Persistent grid (one CTA per SM). Each CTA walks tokens t = cta, cta + grid, ...;
per token 16 tiles of 128 top-k rows (576 B FP8 each: four 128 B boxes under a
128B-swizzled map + one 64 B rope box under a 64B-swizzled map) are gathered into
a ring of NS shared-memory stages; a consumer warp releases each stage as soon as
it lands. No math: this is the L2 -> SMEM bandwidth the attention kernel can reach.
Variants: issuing warps (1, 2, 4), stages (2, 3, 4), and multicast to a CTA pair
(the pair receives the same rows; L2 read once).
usage: python pmla_gather_tma.py --tokens 16384 [--json out]
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
def arrive_expect(bar, nbytes):
    code = "mbarrier.arrive.expect_tx.shared::cta.b64 _,[$0],$1;"
    return code, "r,r", (bar, Int32(nbytes))


@asm()
def arrive(bar):
    return "mbarrier.arrive.shared::cta.b64 _,[$0];", "r", (bar,)


@asm(Uint32)
def smid_reg():
    return "mov.u32 $0, %smid;", "=r", ()


@asm(Int64)
def clock64():
    return "mov.u64 $0, %globaltimer;", "=l,~{memory}", ()


class Gather:
    def __init__(self, warps, stages):
        self.warps, self.stages = warps, stages

    @cute.jit
    def __call__(self, kv: Int64, rows: Int32, idx: Int64, tokens: Int32,
                 out: Int64, grid: Int32, stream: cuda.CUstream):
        u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
        l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        sw = cuda.CUtensorMapSwizzle
        main = tensormap.encode_map(kv, u8, (576, rows), (576,), (128, 1),
                                    sw.CU_TENSOR_MAP_SWIZZLE_128B, l2)
        rope = tensormap.encode_map(kv, u8, (576, rows), (576,), (64, 1),
                                    sw.CU_TENSOR_MAP_SWIZZLE_64B, l2)
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
        if tid == 0:
            for s in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(full + s, 1)
                cute.arch.mbarrier_init(empty + s, self.warps)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        dmain = tensormap.map_address(main)
        drope = tensormap.map_address(rope)
        iptr = cute.make_ptr(Int32, idx, cute.AddressSpace.gmem, assumed_align=16)
        t0 = clock64()
        g = Int32(0)
        t = cta
        quads: cutlass.Constexpr = 32 // self.warps  # quads (4 rows) per lane group
        while t < tokens:
            for j in range(16):
                s = g % self.stages
                ph = (g // self.stages) % 2
                if warp < self.warps:
                    # warp w issues quads w*per .. of the tile, one quad per lane
                    if lane == 0:
                        wait(empty + s, ph ^ 1, scope="cta")
                        if warp == 0:
                            arrive_expect(full + s, KVBYTES)
                    cute.arch.sync_warp()
                    if lane < quads:
                        q = warp * quads + lane
                        r = load_words(iptr + (t * 2048 + j * 128 + 4 * q), count=4)
                        st = ring.iterator + s * KVBYTES
                        for a in cutlass.range_constexpr(4):
                            gather4(st + a * KATOM + q * 512, dmain, full + s, 128 * a,
                                    r[0], r[1], r[2], r[3])
                        gather4(st + KROPE + q * 256, drope, full + s, 512,
                                r[0], r[1], r[2], r[3])
                else:
                    if lane == 0:
                        wait(full + s, ph, scope="cta")
                        for w in cutlass.range_constexpr(self.warps):
                            arrive(empty + s)
                g += 1
            t += grid
        cute.arch.sync_threads()
        if tid == 0:
            optr = cute.make_ptr(Int64, out, cute.AddressSpace.gmem, assumed_align=8)
            o = cute.make_tensor(optr, cute.make_layout(1 << 20))
            o[cta] = clock64() - t0
            o[grid + cta] = Int64(smid_reg())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--warps", default="4,8,16")
    ap.add_argument("--die-local", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    F = a.tokens
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    layers = [make_layer(F, 72900, 3, dev, 17 + i) for i in range(3)]
    out = torch.zeros(2 * sms, dtype=torch.int64, device=dev)
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    doms = dies.sm_domains(dev.index or 0)
    if a.die_local:
        layers = [(dies.localized_like(kv, 0), idx) for kv, idx in layers]
    cs = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    res = {"tokens": F, "sms": sms, "rows": []}
    nbytes = F * 2048 * 576
    for warps in [int(w) for w in a.warps.split(",")]:
        for stages in (2, 3):
            if stages * KVBYTES > 227 * 1024:
                continue
            try:
                g = Gather(warps, stages)
                kv0, idx0 = layers[0]
                args = (Int64(kv0.data_ptr()), Int32(kv0.numel() // 576),
                        Int64(idx0.data_ptr()), Int32(F), Int64(out.data_ptr()),
                        Int32(sms), cs)
                f = cute.compile(g, *args)
                times = []
                for i in range(7):
                    kv, idx = layers[i % 3]
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    f(Int64(kv.data_ptr()), Int32(kv.numel() // 576),
                      Int64(idx.data_ptr()), Int32(F), Int64(out.data_ptr()),
                      Int32(sms), cs)
                    e.record()
                    torch.cuda.synchronize()
                    times.append(s.elapsed_time(e) * 1e3)
                t = statistics.median(times[1:])
                if True:
                    per = out[:sms].float().cpu()
                    sm = out[sms:].cpu().tolist()
                    d = torch.tensor([doms[s] for s in sm])
                    print("die0 CTA ns med", per[d == 0].median().item(), "die1", per[d == 1].median().item(), flush=True)
                row = dict(warps=warps, stages=stages, us=round(t, 1),
                           TBps=round(nbytes / t / 1e6, 2),
                           cta_ns_med=int(out[:sms].float().median().item()))
            except Exception as e:  # noqa: BLE001
                row = dict(warps=warps, stages=stages, error=repr(e)[:400])
            print(row, flush=True)
            res["rows"].append(row)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
