"""Per (layer kind, module path, kernel) exclusive GPU time per forward from a pf_bench trace.
usage: python3 opmap.py <trace.json.gz> [--min-ms 0.2] [--json out.json]"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1_attr as A  # noqa: E402


def short(k):
    for pre in ("kernel_cutlass_kernel_", "kernel_cutlass__kernel_", "kernel_cutlass__", "kernel_cutlass_"):
        if k.startswith(pre):
            k = k[len(pre):]
    k = k.replace("void ", "").replace("(anonymous namespace)::", "")
    return k.split("(")[0].split("<")[0][:90]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--min-ms", type=float, default=0.2)
    ap.add_argument("--json")
    a = ap.parse_args()
    ev = A.load(a.trace)
    fwd = sorted((e for e in ev if e.get("name") == "p1/forward" and e.get("cat") == "user_annotation"), key=lambda e: e["ts"])
    gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    launch = {e["args"]["correlation"]: e for e in ev
              if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    ann = defaultdict(list)
    for e in ev:
        if e.get("cat") == "user_annotation" and "dur" in e and e.get("name") != "p1/forward":
            ann[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"]))
    for v in ann.values():
        v.sort()
    import bisect
    starts = {t: [x[0] for x in v] for t, v in ann.items()}

    def stack(tid, ts):
        v = ann.get(tid, [])
        i = bisect.bisect_right(starts.get(tid, []), ts)
        out = [x for x in v[max(0, i - 4000):i] if x[0] <= ts <= x[1]]
        out.sort(key=lambda x: (x[0], -x[1]))
        return [x[2] for x in out]

    agg = defaultdict(float)
    busy = defaultdict(float)
    cnt = defaultdict(int)
    nf = len(fwd)
    for r in fwd:
        lo, hi = r["ts"], r["ts"] + r["dur"]
        acts = sorted((e for e in gpu if lo <= e["ts"] <= hi), key=lambda e: e["ts"] + e["dur"])
        maxend = None
        for e in acts:
            le = launch.get(e.get("args", {}).get("correlation"))
            st = stack(le["tid"], le["ts"]) if le else []
            kind, lid, rest = "top", None, []
            for s in st:
                m = A.LAYER.match(s)
                if m:
                    kind = "draft" if m.group(1) == "draft" else m.group(3)
                    lid = int(m.group(2))
                else:
                    rest.append(s.split(":")[0])
            if kind == "KimiK3DeltaAttention" and lid == 0:
                kind = "L0"
            path = rest[-1] if rest else "-"
            s0, s1 = e["ts"], e["ts"] + e["dur"]
            if maxend is None:
                ex = e["dur"]; maxend = s1
            else:
                ex = max(0.0, s1 - max(s0, maxend)); maxend = max(maxend, s1)
            key = (kind, path, short(e["name"]))
            agg[key] += ex / nf / 1e3
            busy[key] += e["dur"] / nf / 1e3
            cnt[key] += 1
    rows = sorted(agg.items(), key=lambda x: -x[1])
    tot = sum(agg.values())
    print(f"total excl {tot:.2f} ms/forward over {nf} forwards")
    out = []
    for (k, p, n), v in rows:
        out.append(dict(kind=k, path=p, kernel=n, excl_ms=round(v, 3), busy_ms=round(busy[(k, p, n)], 3),
                        launches=cnt[(k, p, n)] // max(nf, 1)))
        if v >= a.min_ms:
            print(f"{v:8.2f} {busy[(k, p, n)]:8.2f} {cnt[(k, p, n)] // max(nf, 1):5d}  {k:22s} {p:40s} {n}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=0)


if __name__ == "__main__":
    main()
