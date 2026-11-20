# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA-2CTA round 6 (copied from P-MLA pmla14_energy.py): h8 knob arms with time, SM clock, NVML power
and J/layer per window (--nit N iterations per window; 2000 = ~3 s sustained at the power cap).
Original: is the h8 sparse MLA prefill kernel power-bound (operand-energy sensitive)?
Production logical path, F16K L78K (one 78K request, block size 64), 1 GPU. Arms change only the DATA:
  base: Q randn * 0.8 (E4M3), KV randn * 0.8 | q0: Q = 0 (every QK MMA multiplies zero A rows)
  kv0: KV = 0 (QK B and PV A operands zero, gather bytes unchanged) | qpad: Q heads 72-95 = 0 (a quarter of the real
  heads: the energy share of h8's 32 padding lanes 96-127, which today hold the neighbouring Q atom's data)
ABBA over the arms; per arm the median kernel time and the mean SM clock / power sampled by NVML while it runs.
Usage: python3 pmla14_energy.py [knob list]"""

import os
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmla10_check import logical_case, run_l  # noqa: E402


try:
    import pynvml

    pynvml.nvmlInit()
    _NVH = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
except Exception:  # noqa: BLE001
    _NVH = None


def energy_mj():
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(_NVH) if _NVH is not None else 0


class Sampler:
    def __init__(self):
        self.on = False
        self.samples = []
        self.t = threading.Thread(target=self.loop, daemon=True)
        self.t.start()

    def loop(self):
        while True:
            if self.on:
                try:
                    self.samples.append((torch.cuda.clock_rate(), torch.cuda.power_draw() / 1000.0))
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(0.002)

    def window(self, fn):
        self.samples = []
        self.on = True
        r = fn()
        self.on = False
        s = self.samples
        mhz = sum(x for x, _ in s) / max(len(s), 1)
        w = sum(y for _, y in s) / max(len(s), 1)
        return r, mhz, w, len(s)


NIT = int(sys.argv[sys.argv.index("--nit") + 1]) if "--nit" in sys.argv else 2000
JL = []


def main():
    kls = [tuple(int(v) for v in a.split(":")) for a in sys.argv[1:] if ":" in a]
    c = logical_case([2048] * 16384, [78 * 1024 // 64], seed=6)
    q8, kv = c[0], c[1]
    arms = {"base": (q8, kv, ())}
    for k in kls:
        arms[":".join(map(str, k))] = (q8, kv, k)
    if "--data" in sys.argv:
        qp = q8.clone()
        qp[:, 72:96] = 0
        arms.update({"q0": (torch.zeros_like(q8), kv, ()), "kv0": (q8, torch.zeros_like(kv), ()),
                     "qpad": (qp, kv, ())})
    ref = run_l(c, ())
    for name, (qa, ka, kk) in arms.items():
        if qa is q8 and ka is kv and kk:
            o = run_l(c, kk)
            torch.cuda.synchronize()
            print(f"{name}: bitwise equal to base: {torch.equal(o, ref)}", flush=True)
    smp = Sampler()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def timed(a):
        qa, ka, kn = arms[a]
        cc = (qa, ka) + tuple(c[2:])
        run_l(cc, kn)
        torch.cuda.synchronize()

        def body():
            e0 = energy_mj()
            ev[0].record()
            for _ in range(NIT):
                run_l(cc, kn)
            ev[1].record()
            torch.cuda.synchronize()
            e1 = energy_mj()
            JL.append((e1 - e0) / 1000.0 / NIT)
            return ev[0].elapsed_time(ev[1]) / NIT * 1000

        return smp.window(body)

    res = {a: [] for a in arms}
    order = list(arms)
    for a in order:
        timed(a)
    jl = {a: [] for a in arms}
    for a in order + order[::-1]:
        res[a].append(timed(a))
        jl[a].append(JL[-1])
    for a in order:
        v = sorted(x[0] for x in res[a])
        mhz = sum(x[1] for x in res[a]) / len(res[a])
        w = sum(x[2] for x in res[a]) / len(res[a])
        print(f"{a:12s}: median {v[len(v) // 2]:7.1f} us  all {[round(x[0]) for x in res[a]]}  "
              f"SM clock {mhz:6.0f} MHz  NVML power {w:6.0f} W  energy {sum(jl[a]) / len(jl[a]):6.3f} J/layer "
              f"(= {sum(jl[a]) / len(jl[a]) / (v[len(v) // 2] * 1e-6):5.0f} W avg)", flush=True)
    e0 = energy_mj()
    time.sleep(1.0)
    print(f"idle {energy_mj() - e0:.0f} mJ/s", flush=True)


if __name__ == "__main__":
    main()
