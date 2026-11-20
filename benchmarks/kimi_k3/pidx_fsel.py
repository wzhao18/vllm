# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused item-end selection vs PUBLISH + CandSelect2 at the P3-16 per-GPU shape (one DSA
layer, 1 GPU), same process, ABBA: REF (module file in this directory, separate CandSelect2)
vs this tree (IndexerScore PUBLISH selects each row's top-k). Prints per-arm times (score
alone / select alone / both), whether every row holds the same index set and the same
flags, and the flagged-row count."""
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


ref = load(os.environ.get("REF", "pidx_ref_x32.py"))
new = B.pidx
d = B.make([(5461, 78360), (5461, 78360), (5462, 78360)], seed=0)
sp = B.sel_plan(d)
args = (sp, d["q"], d["q_scale"], d["k8"], d["k_sf"], d["w"])
new.thresholds(*args)
torch.cuda.synchronize()
rb_thr = sp.row_bin.clone()
o_ref = torch.full_like(d["out"], -7)
o_new = torch.full_like(d["out"], -7)


def ref_score():
    return ref.publish(*args)


cand_ref = ref_score()
ctas = ref.CS2_CTAS_PER_SM * torch.cuda.get_device_properties(0).multi_processor_count
ksel = ref._cand_select2_kernel(2048, ctas)
st = torch.cuda.current_stream().cuda_stream


def ref_select():
    ksel(cand_ref, sp.row_len, sp.row_bin, o_ref.view(-1), st)


def ref_both():
    c = ref.publish(*args)
    ksel(c, sp.row_len, sp.row_bin, o_ref.view(-1), st)


def new_both():
    new.publish(*args, out=o_new)


us = {}
for name, fn in (("ref_score", ref_score), ("ref_select", ref_select),
                 ("ref_both", ref_both), ("new_both", new_both),
                 ("new_both", new_both), ("ref_both", ref_both)):
    sp.row_bin.copy_(rb_thr)
    us.setdefault(name, []).append(round(B.timed(fn, 10) * 1e3, 1))
sp.row_bin.copy_(rb_thr)
ref_both()
torch.cuda.synchronize()
f_ref = sp.row_bin[1:3 * sp.t:3].clone()
sp.row_bin.copy_(rb_thr)
new_both()
torch.cuda.synchronize()
f_new = sp.row_bin[1:3 * sp.t:3].clone()
a = torch.sort(o_ref, dim=1).values
b = torch.sort(o_new, dim=1).values
ok = (f_ref == 0) & (f_new == 0)
res = {"us": us, "min_us": {k: min(v) for k, v in us.items()},
       "rows": int(a.shape[0]), "flag_ref": int((f_ref != 0).sum()),
       "flag_new": int((f_new != 0).sum()),
       "same_rows_unflagged": int(((a == b).all(1) & ok).sum()), "unflagged": int(ok.sum())}
print("FSEL", json.dumps(res), flush=True)
