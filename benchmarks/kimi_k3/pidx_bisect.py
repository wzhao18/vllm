# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IndexerScore IMA bisect: one (shape, mode, epi) case per process.
Usage: python3 pidx_bisect.py <small|big> <mode> <epi>"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import bench_pidx_fused as b  # noqa: E402

pidx = b.pidx
shape, mode, epi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
reqs = [(64, 1024)] if shape == "small" else (
    [(300, 4096), (17, 9000)] if shape == "mid" else
    [(5461, 78360), (5461, 78360), (5462, 78360)])
d = b.make(reqs, seed=0)
print("GROUPS", pidx.GROUPS, "THREADS", pidx.THREADS, flush=True)
if mode == pidx.PUBLISH:
    sp = b.sel_plan(d)
    row_bin, cand = b.run_sel(d, sp)
    torch.cuda.synchronize()
    pidx.score(d["q"], d["q_scale"], d["k8"], d["k_sf"], d["w"], sp.row_len, sp.items,
               sp.cand_base, cand.view(torch.bfloat16), row_bin.clone(), mode=mode, epi=epi)
else:
    pidx.score(d["q"], d["q_scale"], d["k8"], d["k_sf"], d["w"], d["row_len"], d["items"],
               d["row_base"], d["logits"], d["row_bin"], mode=mode, epi=epi)
torch.cuda.synchronize()
print("OK", shape, mode, epi, flush=True)
