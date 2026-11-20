# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CandSelect2 at the P3-16 per-GPU shape (one DSA layer, 1 GPU) with 1..4 CTAs per SM:
time per launch (ABBA over the CTA counts) and same index sets as the production count."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import bench_pidx_fused as B  # noqa: E402

d = B.make([(5461, 78360), (5461, 78360), (5462, 78360)], seed=0)
sp = B.sel_plan(d)
row_bin, cand = B.run_sel(d, sp)
torch.cuda.synchronize()
rb0 = row_bin.clone()
st = torch.cuda.current_stream().cuda_stream
sms = torch.cuda.get_device_properties(0).multi_processor_count
per = [int(x) for x in os.environ.get("CS_PER", "1 2 3 4").split()]
ks = {p: B.pidx._cand_select2_kernel(2048, p * sms) for p in per}
outs, us = {}, {}
for p in per + per[::-1]:
    rb = rb0.clone()
    o = torch.full_like(d["out"], -7)
    t = B.timed(lambda: ks[p](cand, sp.row_len, rb, o.view(-1), st), 10)
    us.setdefault(p, []).append(round(t * 1e3, 1))
    outs[p] = torch.sort(o, dim=1).values
base = B.pidx.CS2_CTAS_PER_SM
res = {"us": us, "min_us": {p: min(v) for p, v in us.items()},
       "same_rows_vs_%d" % base: {p: int((outs[p] == outs[base]).all(1).sum()) for p in per if base in outs},
       "rows": int(d["out"].shape[0])}
print("CSCTAS", json.dumps(res), flush=True)
