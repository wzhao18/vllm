# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E4/E5 (coordinator 09:42): IndexerScore PUBLISH arms at the P3-16 per-GPU shape (one DSA
layer, 1 GPU), each a ~4 s back-to-back loop with NVML total energy and SM MHz, plus ABBA
CUDA-event time: this tree at epi 0 (production), 1 (skeleton: TMEM loads + release +
publish, no math), 3 (MMA path only, no TMEM loads), and REFS modules (e.g. the packed
publish) at epi 0. Prints us, J/launch, W, MHz and cycles = us x MHz."""
import importlib.util
import json
import os
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import bench_pidx_fused as B  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(
        name[:-3], os.path.join(os.path.dirname(__file__), name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


arms = [("prod", B.pidx, 0), ("epi1", B.pidx, 1), ("epi3", B.pidx, 3)]
arms += [(r[:-3], load(r), 0) for r in os.environ.get("REFS", "").split() if r]
d = B.make([(5461, 78360), (5461, 78360), (5462, 78360)], seed=0)
sp = B.sel_plan(d)
row_bin, cand = B.run_sel(d, sp)
torch.cuda.synchronize()
rb = row_bin.clone()
cands = {n: cand.clone() for n, _, _ in arms}


def launch(n, m, epi):
    m.score(d["q"], d["q_scale"], d["k8"], d["k_sf"], d["w"], sp.row_len, sp.items,
            sp.cand_base, cands[n].view(torch.bfloat16), rb, mode=m.PUBLISH, epi=epi)


us = {}
for n, m, e in arms + arms[::-1]:
    us.setdefault(n, []).append(round(B.timed(lambda: launch(n, m, e), 10) * 1e3, 1))
import pynvml as N  # noqa: E402

N.nvmlInit()
h = N.nvmlDeviceGetHandleByUUID("GPU-" + str(torch.cuda.get_device_properties(0).uuid))
res = {}
for n, m, e in arms + arms[::-1]:
    k = max(int(4.0e6 / min(us[n])), 10)
    clk, stop = [], [False]

    def poll():
        while not stop[0]:
            clk.append(N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM))
            time.sleep(0.02)

    launch(n, m, e)
    torch.cuda.synchronize()
    th = threading.Thread(target=poll)
    th.start()
    e0 = N.nvmlDeviceGetTotalEnergyConsumption(h)
    t0 = time.perf_counter()
    for _ in range(k):
        launch(n, m, e)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    e1 = N.nvmlDeviceGetTotalEnergyConsumption(h)
    stop[0] = True
    th.join()
    mid = clk[len(clk) // 4: 3 * len(clk) // 4] or clk
    mhz = sum(mid) / max(len(mid), 1)
    t_us = (t1 - t0) / k * 1e6
    res.setdefault(n, []).append({
        "us_loop": round(t_us, 1), "J": round((e1 - e0) / 1e3 / k, 4),
        "J_dyn": round(((e1 - e0) / 1e3 - 494.0 * (t1 - t0)) / k, 4),
        "W": round((e1 - e0) / 1e3 / (t1 - t0), 1), "MHz": round(mhz),
        "Kclk": round(t_us * mhz / 1e3, 1)})
print("E45", json.dumps({"us_abba": {n: min(v) for n, v in us.items()}, "loop": res}), flush=True)
