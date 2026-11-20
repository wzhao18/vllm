# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-process A/B of the IndexerScore PUBLISH launch at the P3-16 per-GPU shape (one
DSA layer, 1 GPU): reference module(s) (REFS, files in this directory) vs this tree.
Per arm: ABBA CUDA-event time, NVML energy per layer over a ~3 s loop (J, W, SM MHz)
and bitwise equality of the candidate regions with the first arm."""
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


def main():
    refs = [r for r in os.environ.get("REFS", "pidx_ref_el.py").split() if r]
    mods = [(r[:-3], load(r)) for r in refs] + [("new", B.pidx)]
    d = B.make([(5461, 78360), (5461, 78360), (5462, 78360)], seed=0)
    sp = B.sel_plan(d)
    row_bin, cand = B.run_sel(d, sp)
    torch.cuda.synchronize()
    rb = row_bin.clone()
    cands = {n: cand.clone() for n, _ in mods}

    def launch(n, m):
        m.score(d["q"], d["q_scale"], d["k8"], d["k_sf"], d["w"], sp.row_len, sp.items,
                sp.cand_base, cands[n].view(torch.bfloat16), rb, mode=m.PUBLISH)

    order = mods + mods[::-1] + mods
    us = {}
    for n, m in order:
        us.setdefault(n, []).append(round(B.timed(lambda: launch(n, m), 10) * 1e3, 1))
    torch.cuda.synchronize()
    first = mods[0][0]
    res = {"us": us, "min_us": {n: min(v) for n, v in us.items()},
           "equal_to_" + first: {n: bool(torch.equal(cands[first], cands[n])) for n, _ in mods}}
    try:
        import pynvml as N
        N.nvmlInit()
        h = N.nvmlDeviceGetHandleByUUID("GPU-" + str(torch.cuda.get_device_properties(0).uuid))
        en = {}
        for n, m in order[: 2 * len(mods)]:
            k = max(int(3.0e6 / min(us[n])), 10)
            clk, stop = [], [False]

            def poll():
                while not stop[0]:
                    clk.append(N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM))
                    time.sleep(0.02)
            launch(n, m)
            torch.cuda.synchronize()
            th = threading.Thread(target=poll)
            th.start()
            e0 = N.nvmlDeviceGetTotalEnergyConsumption(h)
            t0 = time.perf_counter()
            for _ in range(k):
                launch(n, m)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            e1 = N.nvmlDeviceGetTotalEnergyConsumption(h)
            stop[0] = True
            th.join()
            mid = clk[len(clk) // 4: 3 * len(clk) // 4] or clk
            en.setdefault(n, []).append({"J": round((e1 - e0) / 1e3 / k, 4),
                                         "W": round((e1 - e0) / 1e3 / (t1 - t0), 1),
                                         "MHz": round(sum(mid) / max(len(mid), 1))})
        res["energy"] = en
    except Exception as ex:  # noqa: BLE001
        res["energy"] = repr(ex)[:200]
    print("ABE", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
