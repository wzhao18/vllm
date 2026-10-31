# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize MegaMoE per-SM phase stamps (bench_megamoe.py --trace): us from the
earliest SM entry on that rank; min / median / max over SMs (0 = not stamped)."""
import sys

import torch

NAMES = ["entry", "pdl_wait", "disp_count", "disp_gsync", "disp_nvlbar", "disp_pull_end",
         "recv_counts", "w_first", "w_done", "mma_done", "epi_done", "epi_gsync",
         "combine_end", "cleanup_end", "tok_first", "epi_first"]
ORDER = [0, 1, 2, 3, 4, 6, 7, 14, 15, 5, 8, 9, 10, 11, 12, 13]

for path in sys.argv[1:]:
    for r in torch.load(path):
        t = r["trace"].double()
        valid = t > 0
        t0 = t[:, 0][valid[:, 0]].min()
        rel = (t - t0) / 1e3
        print(f"\n{path.split('/')[-1]} E{r['experts_per_rank']} T{r['tokens']} {r['routing']} "
              f"bm{r['block_m']}: {r['us']:.1f} us/launch")
        for i in ORDER:
            v = rel[:, i][valid[:, i]]
            if v.numel() == 0:
                continue
            q = torch.quantile(v, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64))
            print(f"  {i:2d} {NAMES[i]:14s} n={v.numel():3d}  min {q[0]:7.2f}  med {q[1]:7.2f}  max {q[2]:7.2f}")
