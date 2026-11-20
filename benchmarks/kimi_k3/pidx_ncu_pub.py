# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One IndexerScore PUBLISH launch per arm (REFS modules in this directory, then this
tree) at the P3-16 per-GPU shape, inside cudaProfilerStart/Stop, for ncu
--profile-from-start off. Prints the (token, key) pair count of the launch."""
import importlib.util
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


refs = [r for r in os.environ.get("REFS", "").split() if r]
mods = [(r[:-3], load(r)) for r in refs] + [("new", B.pidx)]
d = B.make([(5461, 78360), (5461, 78360), (5462, 78360)], seed=0)
sp = B.sel_plan(d)
row_bin, cand = B.run_sel(d, sp)
torch.cuda.synchronize()
outs = []
for n, m in mods:  # compile + warm every arm before profiling
    c = cand.clone()
    m.score(d["q"], d["q_scale"], d["k8"], d["k_sf"], d["w"], sp.row_len, sp.items,
            sp.cand_base, c.view(torch.bfloat16), row_bin.clone(), mode=m.PUBLISH)
    outs.append(c)
torch.cuda.synchronize()
torch.cuda.profiler.start()
for (n, m), c in zip(mods, outs):
    m.score(d["q"], d["q_scale"], d["k8"], d["k_sf"], d["w"], sp.row_len, sp.items,
            sp.cand_base, c.view(torch.bfloat16), row_bin.clone(), mode=m.PUBLISH)
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print("ARMS", [n for n, _ in mods], "pairs", int(sum(
    int(r) for r in sp.row_len.tolist() if r > 0)), flush=True)
