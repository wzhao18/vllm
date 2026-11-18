"""kb_parse.py: per-kernel / per-component latency breakdown of profiled decode steps (kb_run1.sh --profile-dir).

    python3 kb_parse.py <results dir> [--rules rules.json] [--top 12] [--json out.json]

Reads, under <results dir>:
  *.log                                   '[forward] <case>: ... median X us' = CUDA-graph replay step time
                                          (median of 20 x 5 replays, max over ranks; target verify + DSpark drafter;
                                          target LM head / sampling / rejection and CPU overhead are NOT in the graph)
  prof-*/<case>.rank<r>.trace.json.gz     torch-profiler trace of 3 replays per case and rank (vllm bench forward)
  prof-*/<case>.rank<r>.kernels.json      bench summary: replays + activities per forward (used to drop the
                                          per-replay KV/state restore + rendezvous activities exactly)
Prints per case (shape, B): step time, per-rank span / kernel-busy (union over streams) / idle gaps, a component
table (rank 0 us/step + min/median/max over ranks), the top kernels per component (name, grid, calls, us), the
target-vs-drafter split and the unclassified kernels; writes everything to <results dir>/kb_breakdown.json (or --json).

Components (first matching rule wins; regex on the lower-cased full kernel name; edit with --rules, a JSON list of
[component, regex]). The drafter phase = every activity after the last MoE kernel of the replay (the DSpark drafter
has no MoE; the target's last layer ends with one): those kernels are reported under 'drafter/<component>'.
GEMMs are one bucket ('gemm: dense'); q/kv/o proj vs shared expert vs KDA proj split by grid signature in the
per-kernel table (same name, different grid).
"""
import argparse
import glob
import gzip
import json
import os
import re
import statistics
from collections import defaultdict

RULES = [
    ("comm: nccl", r"nccl"),
    ("comm: allreduce (custom/flashinfer)", r"allreduce|all_reduce|lamport|oneshot|twoshot|mnnvl|cross_device|ar_fusion|"
                                            r"allgather|all_gather|reduce_scatter|reducescatter|symm"),
    ("attn: dcp exchange/combine", r"dcp|a2a|all2all|alltoall|merge_attn|merge_state|correct_attn|attn_combine|cp_combine|"
                                   r"_lse|lse_|q_gather|kv_gather|mailbox|bprime"),
    ("attn: indexer", r"indexer|mqa_logits|mqa|index_k|weights_proj|paged_logits|fp8_index|mxfp4_index"),
    ("attn: top-k", r"topk|top_k|deep_select|deepselect|radix|select"),
    ("moe: dispatch", r"dispatch"),
    ("moe: combine", r"combine|finalize|unpermute"),
    ("moe: routing", r"routing|noaux|grouped_topk|gating|expert_map|permute|moe_align|sort_tokens"),
    ("moe: experts", r"mega_?moe|megamoe|moe|expert|grouped_gemm|groupgemm"),
    ("attn: sparse mla core", r"fmha|flashmla|flash_mla|sparse_mla|mla|dsac|dsa_core|dsa|sparse_attn|attention|"
                              r"tokenspeed|trtllm|xqa"),
    ("attn: kv write / rope", r"cache|rope|rotary"),
    ("kda", r"kda|delta|gdn|recurrent|ssm|conv1d|causal_conv|chunk_|mamba|replay"),
    ("sampling", r"sample|sampling|gumbel|argmax|multinomial|exponential|rejection|markov"),
    ("gemm: dense", r"nvjet|gemm|xmma|cutlass|cublas|matmul|bmm|gemv|splitk|sm1\d\d_"),
    ("elementwise: norm/act/quant/copy", r"norm|rms|silu|swiglu|gelu|act|quant|fp8|fp4|mxfp|scale|elementwise|"
                                        r"vectorized|reduce|copy|cat|fill|index|unrolled|memcpy|memset|add|mul"),
]
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")


def short(name):
    try:
        from vllm.benchmarks.forward.profile import short_name  # noqa: PLC0415 (only when vllm is importable)
        return short_name(name)
    except Exception:
        s = name.strip()
        if s.startswith("kernel_cutlass_"):
            return re.split(r"_object_at_|_tensorptr", s, maxsplit=1)[0][:120]
        s = re.sub(r"\(.*\)\s*$", "", s)
        s = s.split("<", 1)[0] if not re.search(r"(device_kernel|Kernel|kernel|KernelWrapper)<", s) else s[:120]
        return s.split()[-1][:120] if s.split() else name[:120]


def classify(name, rules):
    low = name.lower()
    for comp, rx in rules:
        if rx.search(low):
            return comp
    return "other (unclassified)"


def union_us(iv):
    iv = sorted(iv)
    tot, cs, ce = 0.0, None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None:
        tot += ce - cs
    return tot


def split_replays(acts, kj):
    """[(pre activities, forward activities)] per replay."""
    replays = int(kj.get("replays") or 3)
    if kj.get("activities") and len(acts) % replays == 0:
        per = len(acts) // replays
        skip = per - int(kj["activities"])
        if skip >= 0:
            return [(acts[r * per: r * per + skip], acts[r * per + skip: (r + 1) * per]) for r in range(replays)]
    # fallback: cut at the (replays - 1) largest idle gaps (host sync between replays), then drop the leading
    # memcpy/memset run (state restore) and one leading nccl kernel (rendezvous)
    gaps = sorted(range(1, len(acts)), key=lambda i: acts[i]["ts"] - (acts[i - 1]["ts"] + acts[i - 1]["dur"]))
    cuts = sorted(gaps[-(replays - 1):]) if replays > 1 else []
    out, lo = [], 0
    for c in cuts + [len(acts)]:
        chunk = acts[lo:c]
        lo = c
        k = 0
        while k < len(chunk) and chunk[k]["cat"] != "kernel":
            k += 1
        if k < len(chunk) and "nccl" in chunk[k]["name"].lower():
            k += 1
        out.append((chunk[:k], chunk[k:]))
    return out


def analyze_trace(path, kjpath, rules):
    with gzip.open(path, "rt") as f:
        events = json.load(f)["traceEvents"]
    acts = sorted((e for e in events if e.get("cat") in GPU_CATS and "dur" in e), key=lambda e: e["ts"])
    kj = {}
    if os.path.exists(kjpath):
        with open(kjpath) as f:
            kj = json.load(f)
    reps = split_replays(acts, kj)
    per = []
    for pre, fwd in reps:
        if not fwd:
            continue
        t0 = min(e["ts"] for e in fwd)
        t1 = max(e["ts"] + e["dur"] for e in fwd)
        last_moe = max((i for i, e in enumerate(fwd) if classify(e["name"], rules).startswith("moe:")), default=None)
        comp, kern = defaultdict(float), defaultdict(lambda: [0.0, 0])
        for i, e in enumerate(fwd):
            c = classify(e["name"], rules)
            if last_moe is not None and i > last_moe:
                c = "drafter/" + c
            comp[c] += e["dur"]
            args = e.get("args", {})
            grid = args.get("grid")
            key = (c, short(e["name"]) if e["cat"] == "kernel" else e["name"], str(grid) if grid else "")
            kern[key][0] += e["dur"]
            kern[key][1] += 1
        drafter_t0 = fwd[last_moe + 1]["ts"] if last_moe is not None and last_moe + 1 < len(fwd) else t1
        un = union_us([(e["ts"], e["ts"] + e["dur"]) for e in fwd])
        per.append({
            "span_us": t1 - t0, "busy_sum_us": sum(e["dur"] for e in fwd), "busy_union_us": un,
            "idle_us": (t1 - t0) - un, "pre_us": sum(e["dur"] for e in pre), "n": len(fwd),
            "target_span_us": drafter_t0 - t0, "drafter_span_us": t1 - drafter_t0,
            "comp": dict(comp), "kern": {"\t".join(k): v for k, v in kern.items()},
        })
    if not per:
        return None
    med = lambda xs: statistics.median(xs)  # noqa: E731
    out = {k: med([p[k] for p in per]) for k in ("span_us", "busy_sum_us", "busy_union_us", "idle_us", "pre_us", "n",
                                                 "target_span_us", "drafter_span_us")}
    comps = set().union(*(p["comp"] for p in per))
    out["comp"] = {c: med([p["comp"].get(c, 0.0) for p in per]) for c in comps}
    keys = set().union(*(p["kern"] for p in per))
    out["kern"] = {k: [med([p["kern"].get(k, [0, 0])[0] for p in per]), med([p["kern"].get(k, [0, 0])[1] for p in per])]
                   for k in keys}
    out["replays"] = len(per)
    out["summary_error"] = kj.get("error")
    return out


def step_medians(root):
    med = {}
    for log in glob.glob(os.path.join(root, "*.log")):
        with open(log, errors="replace") as f:
            for line in f:
                m = re.match(r"\[forward\] (\S+): .*median ([0-9.]+) us", line)
                if m:
                    med[m.group(1)] = (float(m.group(2)), os.path.basename(log))
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--rules")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json")
    a = ap.parse_args()
    rules = RULES
    if a.rules:
        with open(a.rules) as f:
            rules = [tuple(x) for x in json.load(f)]
    rules = [(c, re.compile(rx)) for c, rx in rules]
    meds = step_medians(a.root)
    traces = defaultdict(dict)
    for p in glob.glob(os.path.join(a.root, "prof-*", "*.trace.json.gz")):
        m = re.match(r"(.+)\.rank(\d+)\.trace\.json\.gz$", os.path.basename(p))
        if m:
            traces[(os.path.basename(os.path.dirname(p)), m.group(1))][int(m.group(2))] = p
    result = {}
    for (pdir, case), ranks in sorted(traces.items(), key=lambda kv: (kv[0][0], int(re.search(r"_B(\d+)_", kv[0][1]).group(1)) if re.search(r"_B(\d+)_", kv[0][1]) else 0)):
        res = {}
        for r, p in sorted(ranks.items()):
            try:
                res[r] = analyze_trace(p, p.replace(".trace.json.gz", ".kernels.json"), rules)
            except Exception as e:  # keep going: one bad trace must not hide the others
                print(f"WARN {p}: {e}")
        res = {r: v for r, v in res.items() if v}
        if not res:
            continue
        r0 = res[min(res)]
        step = meds.get(case, (None, ""))[0]
        print("=" * 118)
        print(f"{pdir}/{case}: timed step median {step if step is not None else '?'} us (graph replay, rank-max) | "
              f"{len(res)} ranks traced, {r0['replays']} replays")
        spans = [v["span_us"] for v in res.values()]
        print(f"  traced span us: rank0 {r0['span_us']:.0f}, min {min(spans):.0f} med {statistics.median(spans):.0f} "
              f"max {max(spans):.0f} | rank0 kernel-busy(union) {r0['busy_union_us']:.0f}, sum {r0['busy_sum_us']:.0f}, "
              f"idle gaps {r0['idle_us']:.0f} ({100 * r0['idle_us'] / max(r0['span_us'], 1):.1f}%), {r0['n']:.0f} activities | "
              f"target {r0['target_span_us']:.0f} us, drafter {r0['drafter_span_us']:.0f} us")
        if r0.get("summary_error"):
            print(f"  note: bench summary error ({r0['summary_error']}); replays split by gaps")
        comps = sorted(set().union(*(v["comp"] for v in res.values())), key=lambda c: -r0["comp"].get(c, 0.0))
        print(f"  {'component':44s} {'r0 us':>9s} {'%span':>6s} {'min':>9s} {'med':>9s} {'max':>9s}  (us/step over ranks)")
        rows = []
        for c in comps:
            xs = [v["comp"].get(c, 0.0) for v in res.values()]
            print(f"  {c:44s} {r0['comp'].get(c, 0.0):9.1f} {100 * r0['comp'].get(c, 0.0) / max(r0['span_us'], 1):6.1f} "
                  f"{min(xs):9.1f} {statistics.median(xs):9.1f} {max(xs):9.1f}")
            rows.append({"component": c, "rank0_us": r0["comp"].get(c, 0.0), "min_us": min(xs),
                         "median_us": statistics.median(xs), "max_us": max(xs)})
        print(f"  {'idle gaps (span - union of kernels)':44s} {r0['idle_us']:9.1f} "
              f"{100 * r0['idle_us'] / max(r0['span_us'], 1):6.1f} {min(v['idle_us'] for v in res.values()):9.1f} "
              f"{statistics.median([v['idle_us'] for v in res.values()]):9.1f} {max(v['idle_us'] for v in res.values()):9.1f}")
        bycomp = defaultdict(list)
        for k, (us, n) in r0["kern"].items():
            c, name, grid = k.split("\t")
            bycomp[c].append((us, n, name, grid))
        print("  top kernels per component (rank 0, us/step, calls/step, grid):")
        for c in comps:
            ks = sorted(bycomp.get(c, []), reverse=True)[: a.top]
            if not ks:
                continue
            print(f"   [{c}]")
            for us, n, name, grid in ks:
                print(f"     {us:9.1f} {n:5.0f}  {name[:90]}  {grid}")
        result[f"{pdir}/{case}"] = {
            "step_median_us": step, "ranks": len(res),
            "per_rank": {r: {k: v[k] for k in ("span_us", "busy_union_us", "busy_sum_us", "idle_us", "target_span_us",
                                              "drafter_span_us", "n")} for r, v in res.items()},
            "components": rows,
            "rank0_kernels": [{"component": c, "name": nm, "grid": g, "us": us, "calls": n}
                              for c, lst in bycomp.items() for us, n, nm, g in sorted(lst, reverse=True)],
        }
    out = a.json or os.path.join(a.root, "kb_breakdown.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {out} ({len(result)} cases)")


if __name__ == "__main__":
    main()
