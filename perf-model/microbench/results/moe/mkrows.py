"""Map MegaMoE B10 bench rows (FlashInfer 6e438852b) onto perf-model MOE.* rows -> rows.json for lane moe.
usage: python3 mkrows.py <b10_rows.jsonl> [<lane_rows.jsonl> ...] <out rows.json>"""
import json, sys

PM = "/Users/weizha/k31vr-dev/perf-model"
MOE_OPS = ("MOE.dispatch", "MOE.fc1", "MOE.act", "MOE.fc2", "MOE.combine")
QV = {"mxfp8": "mxfp4w_mxfp8a", "nvfp4": "nvfp4w_nvfp4a"}
CK = {"mxfp4_release": "mxfp4", "nvfp4_modelopt": "nvfp4"}

*srcs, out = sys.argv[1:]
meas = {}
for s in srcs:
    for l in open(s):
        r = json.loads(l)
        if r.get("baseline") != "B10" or r.get("status") != "ok" or r.get("routing") != "random":
            continue
        k = (r["variant"], r["ep"], r["tokens_per_rank"])
        if k not in meas or r["ts"] > meas[k]["ts"]:
            meas[k] = r

mx = json.load(open(PM + "/microbench/matrix.json"))
ot = json.load(open(PM + "/out/optables.json"))["tables"]
oc = json.load(open(PM + "/out/optables.json"))["cols"]
cc = json.load(open(PM + "/out/optables.json"))["comm_cols"]

rows = []
for c in mx["cases"]:
    if c.get("family") != "megamoe" or c["shape"].get("EP") not in (8, 16):
        continue
    s = c["shape"]
    v = QV[s["act"]]
    T, EP = s["tokens_per_rank"], s["EP"]
    m, mk, mv = meas.get((v, EP, T)), meas.get((v + "+kernel", EP, T)), meas.get((v + "+vllm", EP, T))
    for u in c["uses"]:
        key = "%s|%s|%s" % (u["ckpt"], u["topo"], u["phase"] if u["phase"] == "prefill" else "decode r%d" % u["r"])
        tb = ot[key]
        ops = {r[0]: dict(zip(oc, r)) for r in tb["ops"] if r[0] in MOE_OPS}
        com = [dict(zip(cc, r)) for r in tb["comm"] if r[0] == "moe" and str(r[1]).startswith(("P11", "P12"))]
        sol_ops = sum(o["t_sol_s"] for o in ops.values()) * 1e6
        ach_ops = sum(o["t_ach_s"] for o in ops.values()) * 1e6
        sol_c = sum(x["t_sol_s"] for x in com) * 1e6
        ach_c = sum(x["t_ach_s"] for x in com) * 1e6
        t_sol, t_ach = sol_ops + sol_c, ach_ops + ach_c
        row = dict(case=c["id"], topology=key, ep=EP, tokens_per_rank=T, quant_kind=v,
                   model_act=s["act"], op_ids=list(MOE_OPS) + [x["prim"] for x in com],
                   measured_us=round(m["median_us"], 2) if m else None,
                   measured_kernel_us=round(mk["median_us"], 2) if mk else None,
                   measured_vllm_us=round(mv["median_us"], 2) if mv else None,
                   t_sol_us=round(t_sol, 2), t_ach_us=round(t_ach, 2),
                   t_sol_ops_us=round(sol_ops, 2), t_ach_ops_us=round(ach_ops, 2),
                   t_sol_comm_P11P12_us=round(sol_c, 2), t_ach_comm_P11P12_us=round(ach_c, 2),
                   per_op_t_ach_us={k: round(o["t_ach_s"] * 1e6, 2) for k, o in ops.items()},
                   per_op_bound={k: o["bound_ach"] for k, o in ops.items()},
                   comm_t_ach_us={x["prim"]: round(x["t_ach_s"] * 1e6, 2) for x in com},
                   count_per_step=u["count"], step_share=u["share"])
        if m:
            row["derate"] = round(m["median_us"] / t_ach, 3)
            row["derate_vs_ops_only"] = round(m["median_us"] / ach_ops, 3)
            row["pct_sol"] = round(100 * t_sol / m["median_us"], 1)
            row["kernel_debug"] = row["derate"] > 2.0
            row["version"] = dict(flashinfer=m["code"]["flashinfer"], vllm_frozen=m["code"]["vllm"],
                                  harness="megamoe-bench run_b10.sh / fi_bench2.py (bench_kimi_k3_moe_ep provider current)",
                                  job=m["job"], nodes=m["nodes"], log=m["log"], ts=m["ts"])
        notes = ["measured = whole fused MegaMoE layer (BF16 in, input quant + routing + NVLink dispatch + fc1 + act + fc2 "
                 "+ combine; persistent phase-interleaved kernel, so dispatch/GEMM/combine are not separable by kernel "
                 "timing); t_ach/t_sol = MOE.dispatch..combine ops + P11+P12 comm (model adds them serially per MoE layer)",
                 "measured_kernel_us = staged compute only (input quant/routing staging excluded)"]
        if not m:
            notes.append("NOT MEASURED")
        if u["ckpt"] == "mxfp4_release" and u["phase"] == "prefill":
            notes.append("quant_kind asserted mxfp4_mxfp8 in bench log (pinned MXFP8 activations at all T)")
        if s.get("dedup"):
            notes.append("model assumes rank-level dedup dispatch (P11/P12 scaled by U/k); shipped MegaMoE 6e438852b has no dedup "
                         "(C7 dedup measured +1.4..3.8% slower in the MegaMoE SOL work), so measured = non-dedup kernel")
        row["dedup_model"] = bool(s.get("dedup"))
        row["notes"] = notes
        rows.append(row)
rows.sort(key=lambda r: (r["quant_kind"], r["ep"], r["tokens_per_rank"], r["topology"]))
json.dump(rows, open(out, "w"), indent=1)
for r in rows:
    print("%-34s EP%-2d T%-5d %-14s meas %8s kern %8s  sol %8.1f ach %8.1f (ops %7.1f comm %7.1f)  derate %s" % (
        r["topology"], r["ep"], r["tokens_per_rank"], r["quant_kind"], r["measured_us"], r["measured_kernel_us"],
        r["t_sol_us"], r["t_ach_us"], r["t_ach_ops_us"], r["t_ach_comm_P11P12_us"], r.get("derate")))
