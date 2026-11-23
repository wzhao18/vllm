#!/usr/bin/env python3
"""win_metrics.py <run outdir> [label] [from_min] [to_min]: the minutes 10-60 metrics used in the campaign ledger, next to the
full-profiling-phase numbers of frz_metrics.py. Prints one JSON line with two windowings of the same profiling records:

- "end" (eval/ATTRIBUTION.md section 0): ok records only (no error, not cancelled) whose request END falls in
  [t0 + from, t0 + to); t0 = first profiling request start; y = sum(ISL + OSL) / (to - from) / GPUs.
- "start" (E2E12 winwin.py, the ledger's posted numbers): profiling records (not cancelled) whose request START falls in
  [t0 + from, t0 + to) and that completed; y = sum(ISL + output_token_count) / (to - from) / GPUs.

Both: x = 1000 / p90 per-request ITL, p90/p50 TTFT, p90 E2E latency, p90 E2E-normalized interactivity, client prefix hit
= sum(usage_prompt_cache_read_tokens) / sum(ISL). GPUs from the node configs, as in frz_metrics.py."""
import glob, json, sys
out = sys.argv[1].rstrip("/"); label = sys.argv[2] if len(sys.argv) > 2 else out.split("/")[-1]
a = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0; b = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
nodes = [f for f in glob.glob(f"{out}/logs/*_config.json") if "mooncake" not in f]
gpus = sum(json.load(open(f)).get("gpu_info", {}).get("count", 0) for f in nodes)
recs = []
for f in glob.glob(f"{out}/logs/agentic/conc_*/aiperf_artifacts/profile_export.jsonl"):
    for line in open(f):
        try: d = json.loads(line)
        except Exception: continue
        md = d.get("metadata", {})
        if md.get("benchmark_phase") != "profiling" or md.get("was_cancelled"): continue
        recs.append(d)
def m(d, k):
    v = d["metrics"].get(k); return v.get("value") if isinstance(v, dict) else v
def pct(xs, p):
    xs = sorted(x for x in xs if x is not None); return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")
t0 = min(d["metadata"]["request_start_ns"] for d in recs); lo, hi = t0 + a * 60e9, t0 + b * 60e9; span = (b - a) * 60
def summ(W, out_key):
    isl = sum(m(d, "input_sequence_length") or 0 for d in W); osl = sum(m(d, out_key) or 0 for d in W)
    cache = sum(m(d, "usage_prompt_cache_read_tokens") or 0 for d in W)
    itl = pct([m(d, "inter_token_latency") for d in W], 90)
    norm = [(m(d, "request_latency") / 1000.0) / m(d, "output_sequence_length") for d in W
            if (m(d, "output_sequence_length") or 0) > 0 and m(d, "request_latency")]
    return dict(n=len(W), tput_per_gpu=round((isl + osl) / span / gpus, 1), p90_itl_ms=round(itl, 3), p90_intvty=round(1000 / itl, 2),
                p90_e2e_norm_intvty=round(1 / pct(norm, 90), 2), p90_e2el_s=round(pct([m(d, "request_latency") for d in W], 90) / 1000, 3),
                p90_ttft_s=round(pct([m(d, "time_to_first_token") for d in W], 90) / 1000, 3),
                p50_ttft_s=round(pct([m(d, "time_to_first_token") for d in W], 50) / 1000, 3),
                hit_pct=round(100 * cache / max(isl, 1), 2), mean_isl=round(isl / max(len(W), 1)), mean_osl=round(osl / max(len(W), 1)))
end = [d for d in recs if not d.get("error") and "request_end_ns" in d["metadata"] and lo <= d["metadata"]["request_end_ns"] < hi]
start = [d for d in recs if "request_end_ns" in d["metadata"] and lo <= d["metadata"]["request_start_ns"] < hi]
print(json.dumps(dict(label=label, out=out, gpus=gpus, window_min=[a, b], end=summ(end, "output_sequence_length"),
                      start=summ(start, "output_token_count"))))
