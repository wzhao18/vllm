#!/usr/bin/env python3
"""Workload statistics from AgentX aiperf profile_export.jsonl(.gz) records.

Usage: python3 -I analyze_agentx.py <runs_dir> <out_json>
Only benchmark_phase == 'profiling', not cancelled, with OSL >= 1 records are used
(same population as frz_metrics.py).  Warmup records are reported separately.
"""
import gzip, json, os, sys, glob, statistics, collections

def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def wpct(vals, wts, p):
    """weighted percentile"""
    pairs = sorted(zip(vals, wts))
    tot = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= tot * p / 100.0:
            return v
    return pairs[-1][0]

def summ(xs, ps=(10, 25, 50, 75, 90, 95, 99)):
    if not xs:
        return {}
    d = {"n": len(xs), "mean": sum(xs) / len(xs), "min": min(xs), "max": max(xs)}
    for p in ps:
        d[f"p{p}"] = pct(xs, p)
    return d

def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def mv(r, k):
    m = r["metrics"].get(k)
    return None if m is None else m["value"]

def analyze(path):
    recs = []
    warm = []
    for r in load(path):
        md = r["metadata"]
        if md.get("was_cancelled"):
            continue
        isl = mv(r, "input_sequence_length") or mv(r, "usage_prompt_tokens")
        osl = mv(r, "output_sequence_length") or mv(r, "usage_completion_tokens")
        if isl is None or osl is None:
            continue
        cr = mv(r, "usage_prompt_cache_read_tokens") or 0
        row = dict(conv=md.get("conversation_id"), turn=md.get("turn_index", 0),
                   kind=md.get("source_kind"), depth=md.get("agent_depth", 0),
                   start=md["request_start_ns"] / 1e9, end=md["request_end_ns"] / 1e9,
                   credit=md.get("credit_issued_ns", md["request_start_ns"]) / 1e9,
                   isl=isl, osl=osl, cr=cr, ttft=mv(r, "time_to_first_token"),
                   lat=mv(r, "request_latency"))
        if md.get("benchmark_phase") == "profiling":
            recs.append(row)
        else:
            warm.append(row)
    return recs, warm

def stats(recs, warm):
    out = {}
    isl = [r["isl"] for r in recs]
    osl = [r["osl"] for r in recs]
    cr = [r["cr"] for r in recs]
    fresh = [r["isl"] - r["cr"] for r in recs]
    out["n_profiling"] = len(recs)
    out["n_warmup"] = len(warm)
    out["isl"] = summ(isl)
    out["osl"] = summ(osl)
    out["cache_read"] = summ(cr)
    out["fresh"] = summ(fresh)
    out["fresh_isl_weighted_mean"] = sum(fresh) / len(fresh)
    out["hit_rate_token_weighted"] = sum(cr) / sum(isl)
    out["fresh_frac_requests_zero_hit"] = sum(1 for r in recs if r["cr"] == 0) / len(recs)
    # cache-read granularity
    g = 0
    import math
    for c in cr:
        if c:
            g = math.gcd(g, int(c))
    out["cache_read_gcd_tokens"] = g
    # source kind mix
    kinds = collections.Counter(r["kind"] for r in recs)
    out["source_kind_mix"] = {k: v / len(recs) for k, v in kinds.items()}
    out["agent_depth_mix"] = {str(k): v / len(recs) for k, v in collections.Counter(r["depth"] for r in recs).items()}
    for k in kinds:
        sub = [r for r in recs if r["kind"] == k]
        out[f"by_kind_{k}"] = {"isl_mean": sum(r["isl"] for r in sub) / len(sub),
                               "osl_mean": sum(r["osl"] for r in sub) / len(sub),
                               "fresh_mean": sum(r["isl"] - r["cr"] for r in sub) / len(sub),
                               "hit_rate": sum(r["cr"] for r in sub) / max(1, sum(r["isl"] for r in sub))}
    # fresh-token mass by fresh-size bucket (what prefill chunks see)
    buckets = [0, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 10**9]
    tot = sum(fresh)
    mass = {}
    cnt = {}
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        sel = [f for f in fresh if lo <= f < hi]
        mass[f"[{lo},{hi})"] = sum(sel) / tot
        cnt[f"[{lo},{hi})"] = len(sel) / len(fresh)
    out["fresh_size_bucket_request_frac"] = cnt
    out["fresh_size_bucket_token_mass"] = mass
    # prefix context the fresh tokens attend over: mean cached prefix weighted by fresh tokens
    out["fresh_weighted_mean_prefix"] = sum(r["cr"] * (r["isl"] - r["cr"]) for r in recs) / max(1, tot)
    out["fresh_weighted_mean_end_ctx"] = sum(r["isl"] * (r["isl"] - r["cr"]) for r in recs) / max(1, tot)
    # decode context: token-weighted (each output token at ctx isl + j)
    dctx_vals = [r["isl"] + r["osl"] / 2 for r in recs]
    w = [r["osl"] for r in recs]
    out["decode_ctx_token_weighted"] = {"mean": sum(v * x for v, x in zip(dctx_vals, w)) / sum(w),
                                        **{f"p{p}": wpct(dctx_vals, w, p) for p in (10, 25, 50, 75, 90, 95, 99)}}
    # time-weighted decode residency context (KV held during decode): weight = decode duration
    dw = []
    for r in recs:
        dd = max(0.0, (r["lat"] - (r["ttft"] or 0)) / 1000.0)
        dw.append(dd)
    if sum(dw) > 0:
        out["decode_ctx_time_weighted"] = {"mean": sum(v * x for v, x in zip(dctx_vals, dw)) / sum(dw),
                                           **{f"p{p}": wpct(dctx_vals, dw, p) for p in (50, 90, 99)}}
    out["isl_plus_osl_max_ctx"] = summ([r["isl"] + r["osl"] for r in recs])
    # structural prefix: same conversation, previous turn
    byconv = collections.defaultdict(list)
    for r in recs + warm:
        byconv[r["conv"]].append(r)
    gaps = []
    gaps_credit = []
    delta_prev_full = []   # isl - (prev isl + prev osl)
    delta_prev_in = []     # isl - prev isl
    turns_per_conv = []
    profiling_ids = set(id(r) for r in recs)
    for conv, rs in byconv.items():
        rs.sort(key=lambda r: (r["turn"], r["start"]))
        turns_per_conv.append(len(rs))
        for a, b in zip(rs[:-1], rs[1:]):
            if id(b) not in profiling_ids:
                continue
            gaps.append(b["start"] - a["end"])
            gaps_credit.append(b["credit"] - a["end"])
            delta_prev_full.append(b["isl"] - (a["isl"] + a["osl"]))
            delta_prev_in.append(b["isl"] - a["isl"])
    out["think_time_s_start_minus_prev_end"] = summ(gaps)
    out["think_time_s_credit_minus_prev_end"] = summ(gaps_credit)
    out["n_followup_turns"] = len(gaps)
    out["structural_new_tokens_vs_prev_turn_in_plus_out"] = summ(delta_prev_full)
    out["structural_new_tokens_vs_prev_turn_in"] = summ(delta_prev_in)
    out["turns_per_conversation_observed"] = summ(turns_per_conv)
    # first-turn (no same-conversation predecessor) share and their stats
    first = []
    for conv, rs in byconv.items():
        rs.sort(key=lambda r: (r["turn"], r["start"]))
        if id(rs[0]) in profiling_ids:
            first.append(rs[0])
    out["first_turn_in_window_frac"] = len(first) / len(recs)
    if first:
        out["first_turn_isl_mean"] = sum(r["isl"] for r in first) / len(first)
        out["first_turn_fresh_mean"] = sum(r["isl"] - r["cr"] for r in first) / len(first)
    # throughput-ish
    t0 = min(r["start"] for r in recs); t1 = max(r["end"] for r in recs)
    out["profiling_span_s"] = t1 - t0
    out["req_per_s"] = len(recs) / (t1 - t0)
    out["fresh_tok_per_s"] = sum(fresh) / (t1 - t0)
    out["out_tok_per_s"] = sum(osl) / (t1 - t0)
    out["mean_inflight_requests_littles_law"] = sum(r["end"] - r["start"] for r in recs) / (t1 - t0)
    out["ttft_s"] = summ([(r["ttft"] or 0) / 1000.0 for r in recs], ps=(50, 90, 99))
    return out

def main():
    runs_dir, out_json = sys.argv[1], sys.argv[2]
    res = {}
    pooled, pooled_w = [], []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "agentx", "logs", "agentic", "*", "aiperf_artifacts", "profile_export.jsonl*"))):
        run = p.split(os.sep)[-7]
        recs, warm = analyze(p)
        if not recs:
            continue
        st = stats(recs, warm)
        st["source"] = os.path.relpath(p, runs_dir)
        res[run] = st
        # tag conv ids per run to avoid cross-run merging
        for r in recs:
            r["conv"] = run + "|" + str(r["conv"])
        for r in warm:
            r["conv"] = run + "|" + str(r["conv"])
        pooled += recs; pooled_w += warm
        print(run, len(recs), round(st["isl"]["mean"]), round(st["fresh"]["mean"]), round(st["osl"]["mean"]),
              round(st["hit_rate_token_weighted"], 4), file=sys.stderr)
    res["POOLED"] = stats(pooled, pooled_w)
    with open(out_json, "w") as f:
        json.dump(res, f, indent=1, default=float)

if __name__ == "__main__":
    main()
