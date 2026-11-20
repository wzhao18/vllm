# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA-2CTA round 4: decompose the h8 single-S round trip from the shared-memory clock
stamps of CTA 0 (subtiles 16-23, physical F16K L78K). Per subtile u -> u + 1:
  a = softmax S-full observed - QK MMAs issued (commit -> mbarrier -> wake; MMA time in full mode)
  b = S loaded + S-free arrived - S-full observed
  c = QK warp S-free observed - S-free arrived (arrive -> wake)
  d = QK MMAs issued - S-free observed (stage wait, proxy fence, MMA issue)
plus the softmax tail (bar_red_or, P-buffer wait, P math, O-ready) and the PV issue.
Usage: python3 pmla12_stamps.py [knob lists without the sstamp field, e.g. 0:1:1:1 0:0:0:0]"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmla_cpa2_check import inputs  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import (  # noqa: E402
    sparse_mla_prefill_h8,
)


def med(v):
    v = sorted(v)
    return v[len(v) // 2]


def main():
    arms = [a for a in sys.argv[1:] if not a.startswith("--")] or ["0:1:1:1", "0:0:0:0"]
    inp = inputs(16384, 78 * 1024, seed=5)
    q8, kv, slots, lens = inp
    for arm in arms:
        k = [int(v) for v in arm.split(":")]
        k = (k + [0] * 5)[:4] + [1] + k[5:]  # knob 4 = sstamp
        knobs = tuple(k)
        st = torch.zeros(256, dtype=torch.int32, device="cuda")
        for _ in range(2):
            sparse_mla_prefill_h8(q8, kv, slots, lens, 0.0722, 1.0, knobs=knobs, stamps=st)
        torch.cuda.synchronize()
        s = st.view(8, 32).cpu().long()

        def d(x, y):  # clock deltas (32-bit wrap)
            return [int((y[i] - x[i]) & 0xFFFFFFFF) for i in range(len(x))]

        rows = range(7)  # u = 16..22 -> u + 1
        sm = lambda u, k: s[u, 8 * ((u + 16) % 2) + k]  # noqa: E731
        qk = lambda u, k: s[u, k]  # noqa: E731
        a = d([qk(u, 17) for u in rows], [sm(u, 1) for u in rows])
        b = d([sm(u, 1) for u in rows], [sm(u, 2) for u in rows])
        c = d([sm(u, 2) for u in rows], [qk(u + 1, 24) for u in rows])
        dd = d([qk(u + 1, 24) for u in rows], [qk(u + 1, 17) for u in rows])
        rt = d([qk(u, 17) for u in rows], [qk(u + 1, 17) for u in rows])
        land = d([qk(u + 1, 16) for u in rows], [qk(u + 1, 20) for u in rows])
        smt = [d([sm(u, j) for u in rows], [sm(u, j + 1) for u in rows]) for j in range(2, 6)]
        pvi = d([s[u, 18] for u in rows], [s[u, 19] for u in rows])
        print(f"{arm}: round trip {med(rt)} clk = a {med(a)} (commit->S-full wake) + b {med(b)} "
              f"(S load+arrive) + c {med(c)} (S-free wake) + d {med(dd)} (QK issue; stage wait "
              f"{med(land)}) | softmax after S: bar_red_or {med(smt[0])}, P-buf wait {med(smt[1])}, "
              f"P math {med(smt[2])}, fence+O-ready {med(smt[3])} | PV issue {med(pvi)}", flush=True)
        print(f"  rt all {rt}", flush=True)
        # round 5: PV warp waits for subtile u (slots 7 -> 15 QKI(u + 2), 15 -> 18 O-ready(u))
        # and the softmax latency S-full observed -> O-ready arrived (slot 1 -> 6)
        pq = d([s[u, 7] for u in rows], [s[u, 15] for u in rows])
        po = d([s[u, 15] for u in rows], [s[u, 18] for u in rows])
        sl = d([sm(u, 1) for u in rows], [sm(u, 6) for u in rows])
        print(f"  PV(u) waits: QKI(u+2) {med(pq)} {pq} | O-ready(u) {med(po)} {po} | softmax "
              f"S-full -> O-ready {med(sl)} {sl}", flush=True)
        if len(k) > 16 and k[16] == 28:  # window 28-35: token 0 ends at u = 31 (WG1)
            t0 = s[3, 9]
            r = lambda row, slot: int((s[row, slot] - t0) & 0xFFFFFFFF)  # noqa: E731
            names = [("O-ready31", 3, 14), ("PV31 issued", 3, 19), ("tok-end", 3, 21),
                     ("l/m xbar", 3, 22), ("QLFREE", 3, 23), ("OFULL", 3, 26),
                     ("inv bar", 3, 27), ("blk0", 3, 28), ("blk1", 3, 29), ("blk2", 3, 30),
                     ("blk3", 3, 31), ("QK32 iss", 4, 17), ("WG0 S32", 4, 1),
                     ("QK33 iss", 5, 17), ("WG1 S33", 5, 1), ("S33 freed", 5, 2),
                     ("QK34 iss", 6, 17)]
            print("  boundary (clk from WG1 S-full(31)): " + ", ".join(
                f"{n} {r(a, b)}" for n, a, b in names), flush=True)


if __name__ == "__main__":
    main()
