"""mk_measured.py <gap.json> <out.json> <kind decode|prefill> <slice text> <overlay text> <run path>: per-family table for
k31ws/measured/ from a parsed step-bench run (gap_parse.py decode / pf_fam.py output), INFRA 2026-11-18.

Achievable = perf-model nvfp4_modelopt rows of the parse (MegaMoE = the fused model layer since 2026-11-18 17:50).
Decode rows carry exposed time (critical path) as the family number; prefill rows carry busy time (serial stream)."""
import json
import sys

gap, outp, kind, slice_txt, ov, run = sys.argv[1:7]
d = json.load(open(gap))
out = dict(slice=slice_txt, ckpt="nvfp4", overlay=ov, run=run,
           model="tools/stepbench/*_L93* rows with the fused MegaMoE layer (t_ach_ms_fused = model step)",
           units="ms per step (decode) / per prefill forward (prefill) per GPU", cases={})
for case, c in d.items():
    if not isinstance(c, dict) or ("families" not in c and "rows" not in c):
        continue
    step = c.get("step_median_us") or c.get("step_us") or c.get("median_us")
    ach = c.get("model_t_ach_us")
    row = dict(step_ms=round(step / 1e3, 3) if step else None, model_ach_ms=round(ach / 1e3, 3) if ach else None,
               model_sol_ms=round(c["model_t_sol_us"] / 1e3, 3) if c.get("model_t_sol_us") else None)
    if step and ach:
        row["ratio"] = round(step / ach, 3)
    if kind == "prefill" and c.get("F") and step:
        row["tok_s_gpu"] = round(c["F"] / (step / 1e6), 1)
    fams = {}
    src = c.get("families")
    if src is None:  # decode gap_parse output: aggregate groups with fam.py's family map
        import fam as FM
        src = {r["family"]: r for r in FM.fam_table(c)}
        ach = c["model_t_ach_us"] - c.get("model_not_in_bench_ach_us", 0.0)
        row["model_ach_ms"] = round(ach / 1e3, 3)
        row["ratio"] = round(step / ach, 3)
    for name, f in src.items():
        busy = f.get("busy_us", 0.0)
        expo = f.get("exposed_us", busy)
        a = f.get("t_ach_us", f.get("ach_us"))
        val = expo if kind == "decode" else busy
        fams[name] = dict(busy_ms=round(busy / 1e3, 3), exposed_ms=round(expo / 1e3, 3),
                          ach_ms=round(a / 1e3, 3) if a is not None else None,
                          ratio=round(val / a, 2) if a else None)
    row["families"] = dict(sorted(fams.items(), key=lambda kv: -((kv[1]["exposed_ms"] if kind == "decode"
                                                                   else kv[1]["busy_ms"]) - (kv[1]["ach_ms"] or 0))))
    out["cases"][case] = row
json.dump(out, open(outp, "w"), indent=1)
for case, r in out["cases"].items():
    print(case, r["step_ms"], "vs ach", r["model_ach_ms"], "x%s" % r.get("ratio"))
    for n, f in r["families"].items():
        print("   %-62s busy %8.2f expo %8.2f ach %8s x%s" % (n[:62], f["busy_ms"], f["exposed_ms"], f["ach_ms"], f["ratio"]))
