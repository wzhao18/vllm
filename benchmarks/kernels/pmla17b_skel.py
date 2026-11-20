# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA round 7: h8 knob arms at F16K L78K (1 GPU), ABBA-ordered; the SM clock of every
physical-mode call comes from the kernel's own clock64/globaltimer stamps (knob clk),
so cycles per subtile and the gather's B/clk/SM are on the kernel's real clock.

Usage: python3 pmla17_skel.py <out.json> <arm> [<arm> ...]
  arm = name=mode/knob:list  (mode p = physical h8, l = logical h8L; knob list as the
  h8 tuple: no_store:no_gather:no_math:no_mma:...; empty list = production)."""

import json
import os
import statistics
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_kimi_k3_prefill_sparse_mla import H, L, R, TOPK, make_layer  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import (  # noqa: E402
    sparse_mla_prefill_h8, sparse_mla_prefill_h8_logical)

F, CTX, REQS, NL = 16384, 72900, 3, 3
SUBS_PER_SM = F * (TOPK // 64) / 212


class Nvml:
    def __init__(self, dev):
        self.ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(dev)
            self.ok = True
        except Exception as e:  # noqa: BLE001
            print("nvml unavailable:", e, flush=True)
        self.samples = []
        self.run = False

    def _loop(self):
        while self.run:
            try:
                if self.ok:
                    nv = self.nv
                    self.samples.append((nv.nvmlDeviceGetClockInfo(self.h, nv.NVML_CLOCK_SM),
                                         nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0))
                else:
                    import subprocess
                    gid = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                    o = subprocess.check_output(
                        ["nvidia-smi", "-i", gid, "--query-gpu=clocks.sm,power.draw",
                         "--format=csv,noheader,nounits"], text=True)
                    a, b = o.strip().split(",")[:2]
                    self.samples.append((float(a), float(b)))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.01)

    def start(self):
        self.samples = []
        self.run = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def stop(self):
        self.run = False
        self.t.join()
        if not self.samples:
            return 0.0, 0.0
        return (statistics.median(s[0] for s in self.samples),
                statistics.median(s[1] for s in self.samples))


def main():
    out_path, arms = sys.argv[1], sys.argv[2:]
    dev = torch.device("cuda")
    layers = [make_layer(F, CTX, REQS, dev, 17 + i) for i in range(NL)]
    q = (torch.randn(F, H, L + R, device=dev) * 0.15).to(torch.float8_e4m3fn)
    seq = torch.full((F,), TOPK, dtype=torch.int32, device=dev)
    o = torch.empty(F, H, L, device=dev, dtype=torch.bfloat16)
    nvml = Nvml(int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]))  # physical index

    stamp = torch.zeros(4, dtype=torch.int64, device=dev)

    def call(mode, knobs, i):
        kv, idx = layers[i % NL]
        if mode == "p":
            kn = tuple(knobs) + (0,) * (28 - len(knobs)) + (1,)  # clk stamps
            sparse_mla_prefill_h8(q, kv.view(-1, 576), idx, seq, 0.0722, 1.0, out=o,
                                  knobs=kn, stamps=stamp)
        else:
            lidx, rq, btab = idx.logical
            sparse_mla_prefill_h8_logical(q, kv.view(-1, 576), lidx, rq, btab, 64, 64,
                                          0.0722, 1.0, out=o, knobs=knobs, lengths=seq,
                                          phys_dbg=idx)

    parsed = []
    for a in arms:
        name, spec = a.split("=")
        mode, _, kl = spec.partition("/")
        parsed.append((name, mode, tuple(int(v) for v in kl.split(":") if v != "")))
    ref, bitwise = {}, {}
    for name, mode, knobs in parsed:  # compile + warm every arm first
        t0 = time.time()
        call(mode, knobs, 0)
        torch.cuda.synchronize()
        msg = ""
        if not any(knobs[:4]):  # full kernel: output must equal the first full arm's
            if mode in ref:
                bitwise[name] = bool(torch.equal(o, ref[mode]))
                msg = f" bitwise-equal-to-first-full-{mode}: {bitwise[name]}"
            else:
                ref[mode] = o.clone()
        print(f"compiled {name} {mode} {knobs} in {time.time() - t0:.1f} s{msg}", flush=True)
    res = {n: [] for n, _, _ in parsed}
    order = parsed + parsed[::-1] + parsed + parsed[::-1]
    n_it = 200
    for name, mode, knobs in order:
        for i in range(3):
            call(mode, knobs, i)
        torch.cuda.synchronize()
        s = [torch.cuda.Event(enable_timing=True) for _ in range(n_it)]
        e = [torch.cuda.Event(enable_timing=True) for _ in range(n_it)]
        nvml.start()
        stamp.zero_()
        for i in range(n_it):
            s[i].record()
            call(mode, knobs, i)
            e[i].record()
        torch.cuda.synchronize()
        mhz, w = nvml.stop()
        if mode == "p":
            cyc, ns, nc = stamp.tolist()[:3]
            mhz = cyc / max(ns, 1) * 1e3  # kernel-body SM clock (MHz)
        us = statistics.median(a.elapsed_time(b) * 1e3 for a, b in zip(s, e))
        res[name].append((us, mhz, w))
        print(f"{name:10s} {us:8.1f} us  {mhz:6.0f} MHz  {w:6.0f} W  "
              f"{us * mhz / SUBS_PER_SM:6.0f} clk/subtile "
              f"{36864 * SUBS_PER_SM / (us * mhz):5.1f} B/clk/SM", flush=True)
    summ = {}
    for name, mode, knobs in parsed:
        r = res[name]
        us = statistics.median(x[0] for x in r)
        mhz = statistics.median(x[1] for x in r)
        w = statistics.median(x[2] for x in r)
        summ[name] = {"mode": mode, "knobs": knobs, "us": us, "us_all": [x[0] for x in r],
                      "mhz": mhz, "w": w, "clk_per_subtile": us * mhz / SUBS_PER_SM,
                      "bitwise": bitwise.get(name)}
        print(f"SUMMARY {name:10s} {mode} {':'.join(map(str, knobs)):24s} {us:8.1f} us "
              f"(min {min(x[0] for x in r):.1f} max {max(x[0] for x in r):.1f}) "
              f"{mhz:6.0f} MHz {w:6.0f} W {us * mhz / SUBS_PER_SM:6.0f} clk/subtile",
              flush=True)
    json.dump(summ, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
