"""Join commbench rows with the in-scope optables comm cases -> rows.json + fits."""
import glob
import json
import sys

S = sys.argv[1]          # dir with rows_N*.jsonl
CASES = json.load(open(sys.argv[2]))
OUT = sys.argv[3]

rows = []
meta = []
for f in sorted(glob.glob(S + "/rows_N*.jsonl")):
    for line in open(f):
        d = json.loads(line)
        if "kind" in d and d["kind"] == "a2a" and d["bytes"] % 64 and not d["grid"]:
            pass
        if "kind" in d:
            rows.append(d)
        else:
            meta.append(d)

idx = {}
for r in rows:
    idx[(r["kind"], r["N"], r["bytes"], r["hops"], r["grid"])] = r

# variants that implement each model primitive (production-implementable symmetric-memory kernels)
IMPL = {
    "ar": ["nvls_tri", "torch_multimem", "torch_twoshot", "torch_oneshot", "lamport_oneshot", "lamport_twoshot"],
    "rs": ["nvls_tri", "push_sum", "lamport_sum"],
    "ag": ["push", "push2d", "mc", "lamport"],
    "a2a": ["push", "push2d", "pull", "lamport"],
    "a2a_ce": ["ce"],
    "bar": ["barrier"],
}
out = []
for c in CASES:
    r = idx.get((c["kind"], c["N"], c["bytes"], c["hops"], False))
    rec = dict(case=c["id"], primitive=c["prim"], kind=c["kind"], N=c["N"], topology=c["topos"], ckpts=c["ckpts"],
               op_ids=c["ops"], priority=c["priority"], count_per_step=c["count"], bytes_param=c["bytes"],
               hops=c["hops"], t_sol_us=round(c["t_sol_us"], 2), t_ach_us=round(c["t_ach_us"], 2))
    if r is None:
        rec.update(measured_us=None, derate=None, notes="not measured")
        out.append(rec)
        continue
    cand = {v: r["us"][v] for v in IMPL[c["kind"]] if v in r["us"]}
    if not cand:
        rec.update(measured_us=None, derate=None, notes="all implementable variants failed: %s" % r.get("err"))
        out.append(rec)
        continue
    best = min(cand, key=cand.get)
    m = cand[best]
    rec.update(measured_us=m, variant=best, derate=round(m / c["t_ach_us"], 3), sol_frac=round(c["t_sol_us"] / m, 3),
               all_variants_us=r["us"], bytes_actual=r["bytes_actual"])
    notes = []
    if rec["derate"] > 2.0:
        notes.append("KERNEL-DEBUG derate > 2")
    if "nccl" in r["us"]:
        notes.append("nccl %.1f us" % r["us"]["nccl"])
    if r.get("err"):
        notes.append("errors: " + ",".join(r["err"]))
    rec["notes"] = "; ".join(notes)
    out.append(rec)
json.dump(out, open(OUT, "w"), indent=1)

# grid fits: t = alpha + bytes_on_wire / bw over points >= 4 MiB traffic, per (kind, N, variant)
fits = {}
for r in rows:
    if not r["grid"]:
        continue
    for v, t in r["us"].items():
        b = r["bytes_actual"] * (max(r["N"] - 1, 1) if r["kind"] in ("ag", "a2a") else 1)
        fits.setdefault((r["kind"], r["N"], v), []).append((b, t))
fit_out = {}
for k, pts in sorted(fits.items()):
    pts.sort()
    small = [t for b, t in pts if b <= 64 << 10]
    big = [(b, t) for b, t in pts if b >= 16 << 20]
    rec = {"pts": [(b, round(t, 2)) for b, t in pts]}
    if small:
        rec["alpha_us"] = round(min(small), 2)
    if len(big) >= 2:
        n = len(big); sx = sum(b for b, _ in big); sy = sum(t for _, t in big)
        sxx = sum(b * b for b, _ in big); sxy = sum(b * t for b, t in big)
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        icpt = (sy - slope * sx) / n
        rec["bw_TBps"] = round(1.0 / slope / 1e6, 3) if slope > 0 else None
        rec["icpt_us"] = round(icpt, 2)
        b, t = big[-1]
        rec["bw_at_max_TBps"] = round(b / t / 1e6, 3)
    fit_out["%s|N%d|%s" % k] = rec
json.dump(dict(meta=meta, fits=fit_out), open(OUT.replace("rows.json", "fits.json"), "w"), indent=1)
for k, v in fit_out.items():
    print(k, {x: v.get(x) for x in ("alpha_us", "bw_TBps", "icpt_us", "bw_at_max_TBps")})
