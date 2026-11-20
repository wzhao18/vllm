"""rows.json for the prefill lane = op-mapped rows of the profiled runs (build_rows.py output) + step-only rows."""
import json
import sys

S = sys.argv[1]
mt = json.load(open(f"{S}/model_tables.json"))["tables"]
rows = json.load(open(f"{S}/rows_all.json"))
H = "/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/mb/prefill/results"
# (model table, topology, case, measured step us, tokens per group, GPUs per group, MoE quant, raw log, note)
STEP = [
    ("P1 F16384", "P1", "pf16k_L78k release MegaMoE policy (NVFP4 W4A4 MoE)", 203393.7, 16384, 8, "nvfp4", "p1/p1_rel.log",
     "deployed release policy: NVFP4 activations above 512 tok/rank (model O4, not in the mxfp4_release headline)"),
    ("P1 F16384", "P1", "pf16k_fresh (1 x 16384 fresh, L 16K) MXFP8", 222396.4, 16384, 8, "mxfp8", "p1/p1_mx.log",
     "model is at L 78K (F16K L32K grid point 215.8 ms)"),
    ("P2 F32768", "P2", "pf32k_L78k MXFP8 (production MNBT 32768)", 653064.9, 32768, 4, "mxfp8", "p2b/p2_mx32.log",
     "bench GMU 0.85 (0.92 OOMs with the bench state snapshot at 32K)"),
    ("P2 F32768", "P2", "pf32k_L78k release MegaMoE policy (NVFP4)", 579497.3, 32768, 4, "nvfp4", "p2b/p2_rel32.log",
     "deployed release policy (model O4 not in headline)"),
    ("P2 F16384", "P2", "pf16k_L78k release MegaMoE policy (NVFP4)", 307703.5, 16384, 4, "nvfp4", "p2/p2_rel.log",
     "deployed release policy (model O4 not in headline)"),
    ("P2 F16384", "P2", "pf16k_fresh MXFP8", 347730.7, 16384, 4, "mxfp8", "p2/p2_mx.log", "fresh 16K chunk"),
    ("P3-16 F16384", "P3-16", "pf16k_L78k MXFP8 (no profile)", 1179833.5, 16384, 1, "mxfp8", "p3h/p3_mx16.log",
     "KV pool pinned 14 GiB, max-num-seqs 4, expandable segments (16K tokens/rank at TP1 does not fit the 0.92 pool)"),
    ("P3-16 F16384", "P3-16", "pf16k_L78k release MegaMoE policy (NVFP4)", 1063283.2, 16384, 1, "nvfp4", "p3e/p3_rel.log",
     "KV 16 GiB, max-num-seqs 4"),
    ("P3-16 F8192", "P3-16", "pf8k_L78k release MegaMoE policy (NVFP4)", 531925.7, 8192, 1, "nvfp4", "p3g/p3_rel.log", ""),
    ("P4-4 F16384 f4096", "P4-4", "pf16k_fresh (4 x 4096 fresh) MXFP8", 319763.0, 16384, 4, "mxfp8", "p4/p4_4_mx.log", ""),
    ("P4-2 F16384 f4096", "P4-2", "pf16k_L78k (4 x 4096 @ 78K) MXFP8", 622976.4, 16384, 2, "mxfp8", "p4/p4_2_mx.log",
     "PR #20 tree, VLLM_K3_TP_DSA_DP=1, replicated DSA KV, owner = batch index mod TP; no profile"),
    ("P4-2 F16384 f4096", "P4-2", "pf16k_fresh MXFP8", 607843.3, 16384, 2, "mxfp8", "p4/p4_2_mx.log", ""),
]
for key, topo, case, us, F, n, q, log, note in STEP:
    t = mt[key]
    tok = F / (n * us * 1e-6)
    rows.append(dict(case=f"{topo} {case} [model {key}]", topology=topo, op_ids=["step"], row="full step",
                     measured_us=us, t_sol_us=round(t["t_sol_ms"] * 1e3, 1), t_ach_us=round(t["t_ach_ms"] * 1e3, 1),
                     derate=round(us / (t["t_ach_ms"] * 1e3), 3),
                     notes=f"{tok:.0f} tok/s/GPU measured vs model achievable {t['tok_s_gpu_ach']:.0f} (MXFP8 MoE); "
                           f"MegaMoE quant {q}; raw {H}/{log}. {note}".strip()))
for r in rows:
    if r["row"] == "full step" and "tok/s/GPU" not in r["notes"]:
        key = r["case"].split("[model ")[1].rstrip("]")
        t = mt[key]
        F, n = t["F"], {"P1": 8, "P2": 4, "P3-16": 1, "P4-4": 4, "P4-2": 2}[t["topo"]]
        r["notes"] = (f"{F / (n * r['measured_us'] * 1e-6):.0f} tok/s/GPU measured vs model achievable "
                      f"{t['tok_s_gpu_ach']:.0f}; MegaMoE quant mxfp8 (pinned); " + r["notes"])
json.dump(rows, open(sys.argv[2], "w"), indent=1)
print(len(rows), "rows")
for r in rows:
    if r["row"] == "full step":
        print(f"{r['case'][:90]:90s} {r['measured_us'] / 1e3:8.1f} {r['t_ach_us'] / 1e3:8.1f} {r['derate']}  {r['notes'][:60]}")
