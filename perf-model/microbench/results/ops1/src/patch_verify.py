"""Fold the verify task (raw/verify-*/rows.g*.jsonl) into rows.json: repeat medians for the big GEMMs, padded-N
timings for the unaligned-N BF16 GEMMs (as notes + measured_padded_us)."""
import glob
import json
import statistics as st
import sys

base = sys.argv[1]
rows = json.load(open(base + "/rows.json"))
ver = [json.loads(l) for f in glob.glob(base + "/raw/verify-*/rows.g*.jsonl") for l in open(f)]
ver = [v for v in ver if v.get("kind") == "verify"]
align = {(v["M"], v["N"]): v["us"] for v in ver if v["test"] == "align"}
rep = {}
for v in ver:
    if v["test"] == "repeat":
        rep.setdefault((v["M"], v["N"], v["K"], v["dtype"]), []).append(v["us"])
for r in rows:
    s = r.get("shape") or {}
    if not r["case"].startswith("gemm") or r.get("measured_us") is None:
        continue
    key = (s["M"], s["N"], s["K"], r["dtype"])
    if key in rep:
        med = st.median(rep[key])
        r["notes"] = (f"repeat check (6x, 5 s idle gaps, other GPU): median {med:.1f} us (range {min(rep[key]):.1f}-"
                      f"{max(rep[key]):.1f}); sweep value {r['measured_us']:.1f}; large compute-bound GEMMs vary "
                      f"+-15-25% across GPUs/runs; measured_us := repeat median; " + r["notes"])
        r["measured_us"] = round(med, 3)
        r["derate"] = round(med / r["t_ach_us"], 3)
    if r["dtype"] == "bf16" and s.get("batch", 1) == 1 and s["N"] % 8 and s["K"] == 7168:
        Np = -(-s["N"] // 8) * 8
        pad = align.get((s["M"], Np))
        if pad:
            r["measured_padded_us"] = pad
            r["notes"] = (f"N={s['N']} rows are not 16 B aligned: cuBLAS falls off its fast path; same GEMM with N "
                          f"padded to {Np}: {pad:.1f} us (derate {pad / r['t_ach_us']:.2f}); " + r["notes"])
            r["hypothesis"] = (f"unaligned N={s['N']} (BF16 row stride {s['N'] * 2} B, not /16): cuBLAS picks a "
                               f"non-TMA kernel at ~0.5 PF; padding the weight to N={Np} restores "
                               f"{r['measured_us'] / pad:.1f}x (verified)")
json.dump(rows, open(base + "/rows.json", "w"), indent=1)
print("patched", len(rep), "repeat shapes,", len(align), "align points")
