# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-SM issue rate of the sparse-MLA softmax instructions on sm_107.

One CTA of W warps per SM; every thread runs ITERS iterations of 16 independent
instances of one operation (dependent across iterations), clock64 around the loop.
Reported: instructions per clock per SM (warp-level instructions x 32 lanes / clk).
Ops: ex2 (ex2.approx.ftz.f32), cvt_e4m3 (cvt.rn.satfinite.e4m3x2.f32), cvt_bf16x2,
cvt_f16x2, ex2_f16x2, cvt_e4m3_f16x2 (cvt.rn.satfinite.e4m3x2.f16x2), ffma, ffma2.
usage: python pmla_cvt_rate.py [--json out]
"""

import argparse
import json
import statistics

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import asm
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla import stamp_clock
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h5 import st_global_u32

ITERS = 64
OPS = ["ex2", "cvt_e4m3", "cvt_bf16x2", "cvt_f16x2", "ex2_f16x2", "cvt_e4m3_f16x2", "ffma",
       "ffma2"]


@asm(Uint32, pure=False)
def op_u32(a, b, kind: cutlass.Constexpr):
    """One instance of the op on (a, b) (bit patterns); returns a new bit pattern that
    depends on the result so the loop chain is kept."""
    body = {
        "ex2": "{ .reg .f32 x, y; mov.b32 x, $1; ex2.approx.ftz.f32 y, x; mov.b32 $0, y; }",
        "cvt_e4m3": ("{ .reg .f32 x, y; .reg .b16 t; mov.b32 x, $1; mov.b32 y, $2; "
                     "cvt.rn.satfinite.e4m3x2.f32 t, x, y; cvt.u32.u16 $0, t; }"),
        "cvt_bf16x2": ("{ .reg .f32 x, y; mov.b32 x, $1; mov.b32 y, $2; "
                       "cvt.rn.bf16x2.f32 $0, x, y; }"),
        "cvt_f16x2": ("{ .reg .f32 x, y; mov.b32 x, $1; mov.b32 y, $2; "
                      "cvt.rn.f16x2.f32 $0, x, y; }"),
        "ex2_f16x2": "{ .reg .b32 x; mov.b32 x, $1; ex2.approx.f16x2 $0, x; }",
        "cvt_e4m3_f16x2": ("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, $1; "
                           "cvt.u32.u16 $0, t; }"),
        "ffma": ("{ .reg .f32 x, y; mov.b32 x, $1; mov.b32 y, $2; fma.rn.f32 x, x, y, y; "
                 "mov.b32 $0, x; }"),
        "ffma2": ("{ .reg .b64 a, d; mov.b64 a, {$1, $2}; fma.rn.f32x2 d, a, a, a; "
                  "mov.b64 {$0, _}, d; }"),
    }[kind]
    return body, "=r,r,r", (Uint32(a), Uint32(b))


class Probe:
    def __init__(self, kind: str, warps: int):
        self.kind, self.warps = kind, warps

    @cute.jit
    def __call__(self, out: Int64, stream: cuda.CUstream):
        self.kernel(out).launch(grid=(1, 1, 1), block=(32 * self.warps, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, out: Int64):
        tid, _, _ = cute.arch.thread_idx()
        v = [Uint32(0x3F000000 + 97 * k) + Uint32(tid) for k in range(16)]
        cute.arch.sync_threads()
        if tid == 0:
            stamp_clock(out)
        for _ in cutlass.range_constexpr(ITERS):
            for k in cutlass.range_constexpr(16):
                v[k] = op_u32(v[k], v[(k + 1) % 16], self.kind) & Uint32(0x3F7FFFFF)
        cute.arch.sync_threads()
        if tid == 0:
            stamp_clock(out + Int64(8))
        acc = v[0]
        for k in cutlass.range_constexpr(1, 16):
            acc = acc ^ v[k]
        st_global_u32(out + Int64(24 + 4 * tid), acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    cs = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    rows = []
    for kind in OPS:
        for warps in (4, 8, 16):
            buf = torch.zeros(3 + 16 * warps, dtype=torch.int64, device=dev)
            try:
                f = cute.compile(Probe(kind, warps), Int64(buf.data_ptr()), cs)
                ts = []
                for _ in range(3):
                    f(Int64(buf.data_ptr()), cs)
                    torch.cuda.synchronize()
                    ts.append(int(buf[1] - buf[0]))
                clk = statistics.median(ts)
                n = warps * ITERS * 16  # warp-level instances (each also has an AND)
                row = dict(op=kind, warps=warps, clk=clk,
                           warp_instr_per_clk=round(n / clk, 3),
                           lanes_per_clk=round(32 * n / clk, 1))
            except Exception as e:  # noqa: BLE001
                row = dict(op=kind, warps=warps, error=repr(e)[:300])
            print(row, flush=True)
            rows.append(row)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
