# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CandSelect2 variants at the P3-16 per-GPU shape (one DSA layer, 1 GPU): REFS modules
(files in this directory, each at its own CS2_CTAS_PER_SM, or 'file.py:N' for N CTAs per
SM) and this tree; ABBA time per launch and same index sets / flags as this tree."""
import importlib.util
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import bench_pidx_fused as B  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(
        name[:-3], os.path.join(os.path.dirname(__file__), name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sms = torch.cuda.get_device_properties(0).multi_processor_count
arms = []
for r in [x for x in os.environ.get("REFS", "").split() if x]:
    f, _, n = r.partition(":")
    m = load(f)
    arms.append((r, m, int(n) if n else m.CS2_CTAS_PER_SM))
arms.append(("new", B.pidx, B.pidx.CS2_CTAS_PER_SM))
d = B.make([(5461, 78360), (5461, 78360), (5462, 78360)], seed=0)
sp = B.sel_plan(d)
row_bin, cand = B.run_sel(d, sp)
torch.cuda.synchronize()
rb0 = row_bin.clone()
st = torch.cuda.current_stream().cuda_stream
ks = {n: m._cand_select2_kernel(2048, per * sms) for n, m, per in arms}
outs, us = {}, {}
for n, m, per in arms + arms[::-1]:
    rb = rb0.clone()
    o = torch.full_like(d["out"], -7)
    t = B.timed(lambda: ks[n](cand, sp.row_len, rb, o.view(-1), st), 10)
    us.setdefault(n, []).append(round(t * 1e3, 1))
    outs[n] = (torch.sort(o, dim=1).values, rb[1::3].clone())
res = {"us": us, "min_us": {n: min(v) for n, v in us.items()},
       "same_rows": {n: int((outs[n][0] == outs["new"][0]).all(1).sum()) for n in outs},
       "same_flags": {n: bool(torch.equal(outs[n][1], outs["new"][1])) for n in outs},
       "rows": int(d["out"].shape[0])}
print("CSVAR", json.dumps(res), flush=True)
