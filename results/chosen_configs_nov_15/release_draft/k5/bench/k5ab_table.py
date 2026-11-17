#!/usr/bin/env python3
"""k5ab_table.py <results dir> [AL_k5 / AL_k3]: the k5 vs k3 step A/B of 08-nq4-dsrel-k5ab-c160-m60.sh.

Reads <dir>/c160k{5,3}-{relfa,tgt}.log (nq_run1.sh logs: '[forward] q<q>_B<B>_M<M>_<ctx>: ... median <us> us') and
prints, per request count B, the whole-step medians at q6 and q4 for the release drafter arm (relfa) and target only
(tgt), the drafter time (relfa - tgt), the step ratios T(q6) / T(q4) and the projected interactivity change at that B,
x(k5) / x(k3) - 1 = (AL5 / AL3) / (T(q6) / T(q4)) - 1 (interactivity = 1000 / p90 ITL; ITL ~ step / AL at a fixed B).
Standard library."""
import os
import re
import sys

FWD = re.compile(r"^\[forward\] q(\d+)_B(\d+)_M(\d+)_(\S+): .*median ([0-9.]+) us")
ROOT = sys.argv[1]
AL_RATIO = float(sys.argv[2]) if len(sys.argv) > 2 else 3.81 / 3.08


def load(name):
    path = os.path.join(ROOT, f"{name}.log")
    out, end = {}, None
    if not os.path.exists(path):
        return out, "missing"
    for line in open(path, errors="replace"):
        m = FWD.match(line)
        if m:
            out[int(m.group(2))] = float(m.group(5))
        elif line.startswith("NQEND"):
            end = line.split()[1]
    return out, end


arms = {f"k{k}-{a}": load(f"c160k{k}-{a}") for k in (5, 3) for a in ("relfa", "tgt")}
print(f"results {ROOT}; AL5/AL3 = {AL_RATIO:.4f}")
for name, (_, end) in arms.items():
    print(f"  {name}: {end}")
bs = sorted(set().union(*[set(v[0]) for v in arms.values()]))
hdr = ("  B   k5 relfa   k3 relfa  k5 tgt    k3 tgt    d(k5)  d(k3)   relfa q6/q4  tgt q6/q4  "
       "dx (relfa)  dx (tgt)")
print(hdr)


def get(name, b):
    return arms[name][0].get(b)


def fmt(v, w=9, p=1):
    return f"{v:>{w}.{p}f}" if v is not None else f"{'-':>{w}}"


for b in bs:
    r5, r3, t5, t3 = (get(n, b) for n in ("k5-relfa", "k3-relfa", "k5-tgt", "k3-tgt"))
    d5 = r5 - t5 if r5 is not None and t5 is not None else None
    d3 = r3 - t3 if r3 is not None and t3 is not None else None
    rr = r5 / r3 if r5 and r3 else None
    rt = t5 / t3 if t5 and t3 else None
    dxr = AL_RATIO / rr - 1 if rr else None
    dxt = AL_RATIO / rt - 1 if rt else None
    pct = lambda v: f"{v * 100:>+9.1f}%" if v is not None else f"{'-':>10}"  # noqa: E731
    print(f"{b:>3} {fmt(r5, 10)} {fmt(r3, 10)} {fmt(t5)} {fmt(t3)} {fmt(d5, 6, 0)} {fmt(d3, 6, 0)}   "
          f"{fmt(rr, 9, 3)}   {fmt(rt, 9, 3)}  {pct(dxr)} {pct(dxt)}")
