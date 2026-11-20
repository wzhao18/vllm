# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase stamps (%globaltimer) of dsa.query_uk per CTA (bench-only copy _quk_stamp)."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import _quk_stamp as qs  # noqa: E402

G, P, QK, NOPE, LAT, ROW = 96, 128, 192, 128, 512, 576
NAMES = {0: "start", 1: "b_full", 2: "a_full0", 3: "a_full1", 4: "a_full2",
         6: "acc0g0", 7: "acc0g1", 8: "acc1g0", 9: "acc1g1", 10: "acc2g0", 11: "acc2g1",
         14: "epi0g0", 15: "epi0g1", 16: "epi1g0", 17: "epi1g1", 18: "epi2g0",
         19: "epi2g1", 30: "end"}


def main():
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 384
    dev = torch.device("cuda")
    q = (torch.randn(rows, G * QK, device=dev) * 2).to(torch.bfloat16)
    uk_t = (torch.randn(G, LAT, NOPE, device=dev) * 0.1).to(torch.bfloat16)
    buf = torch.zeros(64 + 2 * G * 32 * 2, dtype=torch.float32, device=dev)
    buf[0] = 0.02
    out = torch.zeros((rows, P, ROW), dtype=torch.float8_e4m3fn, device=dev)
    flush = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
    for mode in ("warm", "cold"):
        for _ in range(3):
            if mode == "cold":
                flush.zero_()
            else:
                qs.query_uk(q, uk_t, buf, out)
            torch.cuda.synchronize()
            buf[64:].zero_()
            qs.query_uk(q, uk_t, buf, out)
            torch.cuda.synchronize()
        st = buf[64:].view(torch.int64).view(2 * G, 32).cpu().double()
        t0 = st[:, 0].min()
        rel = (st - t0) / 1000.0
        print(f"== {mode} rows {rows}: us from the first CTA start (median / max over CTAs)")
        for k, name in NAMES.items():
            col = rel[:, k]
            col = col[st[:, k] > 0]
            if len(col):
                print(f"  {name:8s} med {col.median().item():7.2f} max {col.max().item():7.2f}")


if __name__ == "__main__":
    main()
