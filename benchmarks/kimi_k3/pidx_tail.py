# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IndexerScore PUBLISH time per 128-key tile at the P3-16 shape (2049 items: 141 CTAs
take 10, 71 take 9) vs a shape with exactly 9 items per CTA (1908 items): the work
imbalance of the persistent item schedule."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import bench_pidx_fused as B  # noqa: E402

res = {}
for name, reqs in (("p3_16", [(5461, 78360), (5461, 78360), (5462, 78360)]),
                   ("balanced9", [(5088, 78360), (5088, 78360), (5088, 78360)])):
    d = B.make(reqs, seed=0)
    sp = B.sel_plan(d)
    row_bin, cand = B.run_sel(d, sp)
    it = sp.items.cpu()
    ntp = (it[:, 3] + (it[:, 3] & 1)).tolist()
    ctas = torch.cuda.get_device_properties(0).multi_processor_count
    load = [0] * ctas
    for k, t in enumerate(ntp):
        load[k % ctas] += t
    us = min(B.timed(lambda: B.pidx.score(
        d["q"], d["q_scale"], d["k8"], d["k_sf"], d["w"], sp.row_len, sp.items,
        sp.cand_base, cand.view(torch.bfloat16), row_bin, mode=B.pidx.PUBLISH), 10)
        for _ in range(3)) * 1e3
    res[name] = {"items": len(ntp), "tiles": sum(ntp), "max_cta_tiles": max(load),
                 "us": round(us, 1), "ns_per_tile_avg": round(us * 1e3 * ctas / sum(ntp), 2),
                 "ns_per_tile_max": round(us * 1e3 / max(load), 2)}
print("TAIL", json.dumps(res), flush=True)
