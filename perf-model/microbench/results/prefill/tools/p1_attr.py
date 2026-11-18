"""Attribute a P1 prefill-step CUPTI trace to model components.

Input: ``<case>.rank<r>.trace.json.gz`` written by ``bench_prefill_step.py`` (torch.profiler,
CPU + CUDA, a ``p1/forward`` range per profiled forward, and a ``record_function`` range per
decoder layer / sub-module). Each GPU activity is mapped to the module ranges that enclosed
its launch (runtime event with the same correlation id, same thread), then to a component.

Two times per component, per forward (median over the profiled forwards):
- ``busy``: sum of activity durations;
- ``excl``: critical ("exclusive") time: how far each activity advances the running max end
  time of everything before it (sorted by end); idle time with nothing resident is booked as
  ``host_gap`` of the component of the activity that ends the gap.
The excl column (plus host gaps) sums to the forward span.

usage: python p1_attr.py <trace.json.gz> [more traces ...] [--json out.json] [--tokens N]
"""

import argparse
import bisect
import gzip
import json
import statistics
from collections import defaultdict

import re

COMM = re.compile(
    r"nccl|all_?reduce|allreduce|all_?gather|allgather|reduce_?scatter|reducescatter|multimem|"
    r"lamport|oneshot|twoshot|one_shot|two_shot|symm|barrier|cross_device|custom_ar|a2a|alltoall|"
    r"all_to_all|dcp_|mnnvl",
    re.I,
)
GEMM = re.compile(r"nvjet|gemm|cutlass|cublas|sm\d+_xmma|matmul|mma|wgmma|fp8_gemm|mxfp8|nvfp4_gemm", re.I)
LAYER = re.compile(r"^(L|draft)(\d+)\.(\w+)$")


def comp_of(stack, kname, layer_kind):
    """component, sub-component for one activity.

    stack: enclosing module range names, outermost first (layer range excluded).
    """
    is_comm = bool(COMM.search(kname))
    path = " / ".join(stack).lower()
    if layer_kind is None:
        if any(s.startswith("drafter") or s.startswith("draft") for s in stack):
            return "drafter", "comm" if is_comm else "compute"
        if "embed" in path:
            return "embed", "comm" if is_comm else "compute"
        if "lm_head" in path or "logits" in path:
            return "lm_head", "comm" if is_comm else "compute"
        return "top_other", "comm" if is_comm else "compute"
    if layer_kind == "draft":
        return "drafter", "comm" if is_comm else "compute"
    # MoE / MLP block
    if re.search(r"block_sparse_moe|(^|[ /])mlp[.:]|moe", path):
        if is_comm:
            return "moe", "comm"
        if re.search(r"shared_expert", path):
            return "moe", "shared_experts"
        if re.search(r"(^|[ /.])gate[:]|router", path):
            return "moe", "router"
        if re.search(r"mega|deep_gemm|fp8_fp4|m_grouped|grouped", kname, re.I) or re.search(r"experts|routed", path):
            return "moe", "routed_experts"
        if GEMM.search(kname):
            return "moe", "gemm_other"
        return "moe", "elementwise"
    if re.search(r"self_attn", path):
        kind = "dsa" if "dsa" in layer_kind.lower() else "kda"
        if kind == "dsa":
            if "indexer" in path:
                return "dsa_indexer", "comm" if is_comm else ("gemm" if GEMM.search(kname) else "kernel")
            if re.search(r"(^|[ /.])(attn|mla_attn|attention|sparse)[\w]*:", path) and not re.search(
                r"(o_proj|q_b_proj|fused_qkv_a_proj|q_a_proj|kv_a_proj|g_proj)[\w]*:", stack[-1].lower() if stack else ""
            ):
                return "dsa_attn_op", "comm" if is_comm else ("gemm" if GEMM.search(kname) else "kernel")
            if is_comm:
                return "dsa_proj", "comm"
            return "dsa_proj", "gemm" if GEMM.search(kname) else "elementwise"
        # KDA
        if is_comm:
            return "kda_proj", "comm"
        if re.search(r"(in_proj|o_proj|qkv|f_a_proj|f_b_proj|g_proj|b_proj|a_proj)[\w]*:", path) and GEMM.search(kname):
            return "kda_proj", "gemm"
        if re.search(r"kda|chunk|recur|delta|segscan|fla|ssm|conv", kname, re.I):
            return "kda_core", "kernel"
        if GEMM.search(kname):
            return "kda_proj", "gemm"
        return "kda_core", "elementwise"
    if is_comm:
        return "layer_other", "comm"
    return "layer_other", "norm_attnres"


def load(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)["traceEvents"]


def analyze(events):
    fwd = sorted(
        (e for e in events if e.get("name") == "p1/forward" and e.get("cat") == "user_annotation"),
        key=lambda e: e["ts"],
    )
    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    launch = {}
    for e in events:
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "args" in e and "correlation" in e["args"]:
            launch[e["args"]["correlation"]] = e
    ann = defaultdict(list)  # tid -> [(ts, end, name)]
    for e in events:
        if e.get("cat") == "user_annotation" and "dur" in e and e.get("name") != "p1/forward":
            ann[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"]))
    for v in ann.values():
        v.sort()
    starts = {t: [a[0] for a in v] for t, v in ann.items()}

    def stack_at(tid, ts):
        v = ann.get(tid, [])
        i = bisect.bisect_right(starts.get(tid, []), ts)
        out = [a for a in v[max(0, i - 4000) : i] if a[0] <= ts <= a[1]]
        out.sort(key=lambda a: (a[0], -a[1]))
        return [a[2] for a in out]

    per_fwd = []
    for r in fwd:
        lo, hi = r["ts"], r["ts"] + r["dur"]
        acts = sorted((e for e in gpu if lo <= e["ts"] <= hi), key=lambda e: (e["ts"] + e["dur"]))
        busy = defaultdict(float)
        excl = defaultdict(float)
        gaps = defaultdict(float)
        kern = defaultdict(lambda: defaultdict(float))
        layers = defaultdict(float)
        maxend = None
        first = min((e["ts"] for e in acts), default=lo)
        for e in acts:
            corr = e.get("args", {}).get("correlation")
            le = launch.get(corr)
            st = stack_at(le["tid"], le["ts"]) if le else []
            lk, lid = None, None
            rest = []
            for s in st:
                m = LAYER.match(s)
                if m:
                    lk = "draft" if m.group(1) == "draft" else m.group(3)
                    lid = f"{m.group(1)}{m.group(2)}"
                else:
                    rest.append(s)
            name = e["name"]
            c, sub = comp_of(rest, name, lk)
            key = (c, sub)
            s0, s1 = e["ts"], e["ts"] + e["dur"]
            busy[key] += e["dur"]
            if maxend is None:
                ex, gap = e["dur"], s0 - first if s0 > first else 0.0
                maxend = s1
            else:
                gap = max(0.0, s0 - maxend)
                ex = max(0.0, s1 - max(s0, maxend))
                maxend = max(maxend, s1)
            excl[key] += ex
            gaps[c] += gap
            kern[c][name[:120]] += ex
            if lid:
                layers[(lid, lk)] += ex + gap
        span = (max((e["ts"] + e["dur"] for e in acts), default=lo) - first) if acts else 0.0
        per_fwd.append(dict(busy=busy, excl=excl, gaps=gaps, kern=kern, span=span, layers=layers, n=len(acts)))
    return per_fwd


def merge(per_fwd):
    keys = set()
    for f in per_fwd:
        keys |= set(f["excl"]) | set(f["busy"])
    comps = defaultdict(lambda: dict(busy=0.0, excl=0.0, gap=0.0, sub={}))
    for k in keys:
        c, sub = k
        b = statistics.median(f["busy"].get(k, 0.0) for f in per_fwd)
        x = statistics.median(f["excl"].get(k, 0.0) for f in per_fwd)
        comps[c]["busy"] += b
        comps[c]["excl"] += x
        comps[c]["sub"][sub] = dict(busy=round(b, 1), excl=round(x, 1))
    for c in comps:
        comps[c]["gap"] = statistics.median(f["gaps"].get(c, 0.0) for f in per_fwd)
    top = {}
    for c in comps:
        agg = defaultdict(float)
        for f in per_fwd:
            for n, v in f["kern"].get(c, {}).items():
                agg[n] += v / len(per_fwd)
        top[c] = sorted(((round(v, 1), n) for n, v in agg.items()), reverse=True)[:8]
    lay = defaultdict(float)
    for f in per_fwd:
        for k, v in f["layers"].items():
            lay[k] += v / len(per_fwd)
    kinds = defaultdict(list)
    for (lid, lk), v in lay.items():
        kinds[lk].append(v)
    return dict(
        span=statistics.median(f["span"] for f in per_fwd),
        activities=statistics.median(f["n"] for f in per_fwd),
        comps={c: dict(busy=round(v["busy"], 1), excl=round(v["excl"], 1), gap=round(v["gap"], 1), sub=v["sub"])
               for c, v in comps.items()},
        top=top,
        layers={f"{lid}.{lk}": round(v, 1) for (lid, lk), v in sorted(lay.items(), key=lambda x: int(x[0][0].lstrip("Ldraft") or 0))},
        layer_kinds={k: dict(n=len(v), mean=round(statistics.mean(v), 1), min=round(min(v), 1), max=round(max(v), 1))
                     for k, v in kinds.items()},
    )


ORDER = ["moe", "kda_core", "kda_proj", "dsa_proj", "dsa_indexer", "dsa_attn_op", "layer_other", "drafter",
         "embed", "lm_head", "top_other"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--tokens", type=int, default=16384)
    a = ap.parse_args()
    out = {}
    for t in a.traces:
        r = merge(analyze(load(t)))
        out[t] = r
        span = r["span"]
        print(f"== {t}: span {span / 1e3:.2f} ms, {r['activities']:.0f} activities, "
              f"{span / a.tokens:.3f} us/token")
        print(f"{'component':14s} {'excl ms':>9s} {'gap ms':>8s} {'busy ms':>9s} {'us/tok':>7s} {'%':>6s}  sub (excl ms)")
        for c in sorted(r["comps"], key=lambda c: ORDER.index(c) if c in ORDER else 99):
            v = r["comps"][c]
            tot = v["excl"] + v["gap"]
            subs = ", ".join(f"{s} {d['excl'] / 1e3:.2f}" for s, d in sorted(v["sub"].items(), key=lambda x: -x[1]["excl"]))
            print(f"{c:14s} {v['excl'] / 1e3:9.2f} {v['gap'] / 1e3:8.2f} {v['busy'] / 1e3:9.2f} "
                  f"{tot / a.tokens:7.3f} {100 * tot / span if span else 0:6.1f}  {subs}")
        print("layer kinds (ms per layer incl. gaps):",
              {k: (v["n"], round(v["mean"] / 1e3, 3)) for k, v in r["layer_kinds"].items()})
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
