# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA: per-warp softmax stamps of h8. Patches SparseMlaPrefillH8.ss so that each of
the 4 warps of the owning softmax warpgroup writes its own 8 slots (row = subtile u - 16,
col = 8 * (warp % 4) + stamp: 0 entry, 1 S full passed, 2 S loaded, 3 m decided (after
bar_red_or), 4 P buffer free, 5 P written, 6 fenced before OREADY), then runs the stamp
harness (args passed through; use --h4-knobs 0:0:0:0:1:0:1) and prints the raw rows."""
import json
import runpy
import sys

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32

import vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 as m8


@cute.jit
def ss(self, base, cta, lane, u, slot, role: cutlass.Constexpr = 15):
    if cutlass.const_expr(self.sstamp and (self.ssmask & role)):
        if cta == 0 and lane == 0 and u >= 16 and u < 24:
            tid, _, _ = cute.arch.thread_idx()
            w = cute.arch.make_warp_uniform(tid // 32) % Int32(4)
            m8.sstamp_clock(base + Uint32(m8.SMEM_BYTES) + Uint32(
                ((u - 16) * 32 + Int32(slot) % Int32(8) + 8 * w) * 4))


m8.SparseMlaPrefillH8.ss = ss
out = sys.argv[sys.argv.index("--json") + 1]
sys.argv = ["benchmark_kimi_k3_prefill_sparse_mla.py"] + sys.argv[1:]
try:
    runpy.run_path("benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py",
                   run_name="__main__")
except SystemExit:
    pass
d = json.load(open(out))
for k in [k for k in d if k.startswith("sstamp")]:
    raw = d[k]["raw"]
    t0 = raw[0][0]
    print(k)
    for r, row in enumerate(raw):
        rel = [((v - t0) & 0xFFFFFFFF) if v else -1 for v in row]
        print(f"u{16 + r} WG{r % 2}", " | ".join(
            " ".join(f"{x:6d}" for x in rel[8 * w:8 * w + 8]) for w in range(4)))
