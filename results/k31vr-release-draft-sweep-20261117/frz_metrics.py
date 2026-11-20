#!/usr/bin/env python3
"""frz_metrics.py <run outdir> [label]: the 4-way pareto metrics of one AgentX run, profiling-phase records only, same method as
live_metrics.py (official numbers): GPUs from the node configs; TPS/GPU = sum(ISL+OSL) / profiling wall / GPUs;
p90 interactivity = 1000 / p90 per-request ITL; p90 E2E-normalized interactivity = 1 / p90(request_latency_s / OSL);
p90 E2E latency and p90 TTFT in s. Also: client prefix hit not computed here; ok/err counts and profiling span. Prints one JSON line."""
import glob, json, sys
out = sys.argv[1].rstrip("/"); label = sys.argv[2] if len(sys.argv) > 2 else out.split("/")[-1]
nodes = [f for f in glob.glob(f"{out}/logs/*_config.json") if "mooncake" not in f]
gpus = sum(json.load(open(f)).get("gpu_info", {}).get("count", 0) for f in nodes)
recs, errs = [], 0
for f in glob.glob(f"{out}/logs/agentic/conc_*/aiperf_artifacts/profile_export.jsonl"):
    for line in open(f):
        try: d = json.loads(line)
        except Exception: continue
        if d.get("metadata", {}).get("benchmark_phase") != "profiling": continue
        if d.get("error"): errs += 1; continue
        recs.append(d)
def m(d, k):
    v = d["metrics"].get(k); return v.get("value") if isinstance(v, dict) else v
def pct(xs, p):
    xs = sorted(x for x in xs if x is not None); return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")
if not recs or not gpus: print(json.dumps(dict(label=label, out=out, error=f"records {len(recs)} gpus {gpus}"))); sys.exit(1)
toks = sum((m(d, "input_sequence_length") or 0) + (m(d, "output_sequence_length") or 0) for d in recs)
t0 = min(d["metadata"]["request_start_ns"] for d in recs); t1 = max(d["metadata"]["request_end_ns"] for d in recs); wall = (t1 - t0) / 1e9
norm = [(m(d, "request_latency") / 1000.0) / m(d, "output_sequence_length") for d in recs if (m(d, "output_sequence_length") or 0) > 0 and m(d, "request_latency")]
print(json.dumps(dict(label=label, out=out, gpus=gpus, ok=len(recs), err=errs, span_min=round(wall / 60, 1),
    tput_per_gpu=round(toks / wall / gpus, 1), p90_itl_ms=round(pct([m(d, "inter_token_latency") for d in recs], 90), 3),
    p90_intvty=round(1000 / pct([m(d, "inter_token_latency") for d in recs], 90), 2),
    p90_e2e_norm_intvty=round(1 / pct(norm, 90), 2), p90_e2el_s=round(pct([m(d, "request_latency") for d in recs], 90) / 1000, 3),
    p90_ttft_s=round(pct([m(d, "time_to_first_token") for d in recs], 90) / 1000, 3),
    p50_ttft_s=round(pct([m(d, "time_to_first_token") for d in recs], 50) / 1000, 3))))
