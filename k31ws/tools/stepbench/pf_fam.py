"""pf_fam.py <prof dir (prof-pf/)> <pf_model_rows_L24.json> <ckpt mxfp4|nvfp4> [out.json]: k31ws P3-16 DEP4 slice
prefill traces -> per op group and per family, measured vs model achievable (24-layer view). Adapted from gap512
pf_parse.py (C512 P2 prefill): TP1 (no ShardQ / GemmRsAr: o_proj GEMMs by ordinal), vendored Rubin MegaMoE names.
Original doc:

Same method as gap_parse.py (AttnRes-delimited segments, PDL clip, exposed share); prefill kernel map below.
GEMM ordinals per segment:
  KDA:  g0 in_proj, g1 f_b_proj; CuTe GemmRsAr = o_proj fused with the SP reduce-scatter
  DSA:  g0 qkv_a_packed, g1..g3 q_b / indexer wq / weights (qb_wq_packed), then (after top-k) NN GEMM w_uk,
        NN GEMM w_uv, g_proj, GemmRsAr o_proj
  MOE:  g0 router, g1 latent_down, MegaMoE, g2 shared_gate_up, g3 shared_down, g4 latent_up
"""
import json
import os
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gap_parse as G  # noqa: E402

GROUPS = {
    "KDA.in_proj": ["KDA.in_proj", "KDA.f_b_proj"],
    "KDA.core": ["KDA.conv", "KDA.gate_beta_norm", "KDA.core_prefill", "KDA.o_norm"],
    "KDA.o_proj": ["KDA.o_proj"],
    "DSA.qkv_a": ["DSA.qkv_a_packed"],
    "DSA.q_proj": ["DSA.qb_wq_packed"],
    "DSA.idx_quant+kv_insert": ["DSA.idx_q_quant", "DSA.prologue"],
    "DSA.idx_score": ["DSA.idx_score"],
    "DSA.topk": ["DSA.topk"],
    "DSA.shardq_comm": ["COMM.P9.dsa", "COMM.P5.dsa"],
    "DSA.w_uk": ["DSA.w_uk"],
    "DSA.sparse_mla": ["DSA.sparse_mla"],
    "DSA.w_uv": ["DSA.w_uv"],
    "DSA.g_proj+gate": ["DSA.g_proj", "DSA.gate_mul"],
    "DSA.o_proj": ["DSA.o_proj"],
    "COMM.SP_AG": ["COMM.P2.kda", "COMM.P2.dsa"],
    "COMM.embed_RS": ["COMM.P3b.embed", "G.embed"],
    "ATTNRES": ["L.attnres_pre", "L.attnres_post", "F.attnres_final"],
    "MOE.router": ["MOE.router"],
    "MOE.topk": ["MOE.topk"],
    "MOE.latent_down": ["MOE.latent_down"],
    "MOE.megamoe": ["MOE.dispatch", "MOE.fc1", "MOE.act", "MOE.fc2", "MOE.combine", "COMM.P11.moe", "COMM.P12.moe"],
    "MOE.prepare": [],
    "MOE.shared_gate_up": ["MOE.shared_gate_up"],
    "MOE.shared_act": ["MOE.shared_act"],
    "MOE.shared_down": ["MOE.shared_down"],
    "MOE.latent_norm": ["MOE.latent_norm"],
    "MOE.latent_up": ["MOE.latent_up"],
    "DENSE": ["DENSE.gate_up", "DENSE.act", "DENSE.down"],
    "TAIL.drafter+final": ["DS.aux_capture", "DS.fc", "DS.hidden_norm", "DS.ctx_kv_proj", "DS.ctx_kv_insert", "F.norm",
                           "F.lm_head", "COMM.P2.drafter", "COMM.P2.final", "COMM.P16.final"],
    "OTHER": [],
}
GEMM = re.compile(r"nvjet|gemm|xmma")


def seg_type(names):
    j = " ".join(names)
    if "mega_moe" in j or "megamoe" in j:
        return "MOE"
    if "flash_kda" in j or "qkv_conv" in j or "chain_scan" in j:
        return "KDA"
    if "mqa_logits" in j or "fmha" in j:
        return "DSA"
    if "situ_and_mul" in j or "act_and_mul" in j:
        return "DENSE"
    return "MISC"


def assign(seg, stype):
    out = []
    gi = 0
    after_topk = False
    moe_seen = False
    for e in seg:
        n = e["name"].lower()
        g = None
        if "attn_res" in n:
            g = "ATTNRES"
        elif stype == "HEAD":
            g = "COMM.embed_RS"
        elif stype == "TAIL":
            g = "TAIL.drafter+final"
        elif "push_shard" in n or "multimem_barrier" in n:
            g = "COMM.SP_AG" if gi == 0 else ("DSA.shardq_comm" if stype == "DSA" else "COMM.SP_AG")
        elif "gemmrsar" in n or "gemm_rs_ar" in n:
            g = "KDA.o_proj+RS" if stype == "KDA" else "DSA.o_proj+RS"
        elif stype == "KDA":
            if GEMM.search(n) and "nvjet" in n:
                g = "KDA.in_proj" if gi < 2 else "KDA.o_proj"
                gi += 1
            else:
                g = "KDA.core"
        elif stype == "DSA":
            if "mqa_logits" in n:
                g = "DSA.idx_score"
            elif "topk" in n:
                g = "DSA.topk"
                after_topk = True
            elif "fmha" in n or "index_remap" in n or "computelse" in n:
                g = "DSA.sparse_mla"
            elif "mxfp4" in n or "concat_and_cache" in n or "kv_insert" in n:
                g = "DSA.idx_quant+kv_insert"
            elif "gate_sigmoid" in n:
                g = "DSA.g_proj+gate"
            elif "nvjet" in n:
                if not after_topk:
                    g = "DSA.qkv_a" if gi == 0 else "DSA.q_proj"
                else:
                    k = sum(1 for x in out if x[0] in ("DSA.w_uk", "DSA.w_uv", "DSA.g_proj+gate") and "nvjet" in
                            x[1]["name"].lower())
                    k = sum(1 for x in out if x[0] in ("DSA.w_uk", "DSA.w_uv", "DSA.g_proj+gate", "DSA.o_proj")
                            and "nvjet" in x[1]["name"].lower())
                    g = ["DSA.w_uk", "DSA.w_uv", "DSA.g_proj+gate", "DSA.o_proj"][min(k, 3)]
                gi += 1
            elif gi <= 1 and not after_topk and ("void" in n or "rms" in n or "norm" in n):
                g = "DSA.qkv_a"
            else:
                g = "DSA.shardq_comm"
        elif stype == "MOE":
            if "mega_moe" in n or "megamoe" in n or n.startswith("_stage_kernel"):
                g = "MOE.megamoe"
                moe_seen = True
            elif "prepare_megamoe" in n:
                g = "MOE.prepare"
            elif "topk" in n:
                g = "MOE.topk"
            elif "situ_and_mul" in n or "act_and_mul" in n:
                g = "MOE.shared_act"
            elif "nvjet" in n:
                if not moe_seen:
                    g = ["MOE.router", "MOE.latent_down"][min(gi, 1)]
                else:
                    k = sum(1 for x in out if x[0] in ("MOE.shared_gate_up", "MOE.shared_down", "MOE.latent_up"))
                    g = ["MOE.shared_gate_up", "MOE.shared_down", "MOE.latent_up"][min(k, 2)]
                gi += 1
            elif moe_seen:
                g = "MOE.latent_norm"
            else:
                g = "OTHER"
        elif stype == "DENSE":
            g = "DENSE"
        else:
            g = "OTHER"
        out.append((g, e))
    return out


def main():
    prof_dir, mpath, ck = sys.argv[1:4]
    root = os.path.dirname(os.path.abspath(prof_dir))
    outp = sys.argv[4] if len(sys.argv) > 4 else os.path.join(root, "gap.json")
    ckpt = {"mxfp4": "mxfp4_release", "nvfp4": "nvfp4_modelopt"}[ck]
    allm = json.load(open(mpath))["prefill"]
    model = {}
    for case in ("pf16k_L78k", "pf8k_L78k", "pf16k_fresh"):
        F = 8192 if case.startswith("pf8k") else 16384
        m = dict(allm["%s|F%d" % (ckpt, F)])
        m.setdefault("f_req", 5456)
        model[case] = m
    G.seg_type = seg_type
    G.GROUPS = GROUPS

    def assign_wrap(seg, t):
        return assign(seg, t)

    G.assign = assign_wrap
    meds = {}
    for line in open(os.path.join(root, "pf.log"), errors="replace"):
        m = re.match(r"\[forward\] (\S+): .*median ([0-9.]+) us", line)
        if m:
            meds[m.group(1)] = float(m.group(2))
    prof = os.path.abspath(prof_dir)
    cases = sorted({f.split(".rank")[0] for f in os.listdir(prof) if f.endswith(".trace.json.gz")})
    out = {}
    for case in cases:
        mrow = model[case]
        ranks = sorted(f for f in os.listdir(prof) if f.startswith(case + ".rank") and f.endswith(".trace.json.gz"))
        per = [G.analyze(os.path.join(prof, f)) for f in ranks]
        groups = set().union(*(p["tot"] for p in per))
        rows = []
        for g in sorted(groups):
            vals = [p["tot"].get(g, 0.0) for p in per]
            ids = GROUPS.get(g, [])
            s, a = G.model_sum(mrow, ids)
            ex = statistics.median([p["exp"].get(g, 0.0) for p in per])
            med = statistics.median(vals)
            rows.append(dict(group=g, op_ids=ids, measured_us=med, measured_rank0_us=vals[0], measured_max_us=max(vals),
                             exposed_us=ex, exposed_gap_us=ex - a, t_sol_us=s, t_ach_us=a,
                             derate=med / a if a > 0 else None, gap_us=med - a, calls=per[0]["cnt"].get(g, 0),
                             kernels=per[0]["names"].get(g, {})))
        rows.sort(key=lambda r: -r["exposed_gap_us"])
        nib = G.model_sum(mrow, ["COMM.P19.step"])
        out[case] = dict(step_median_us=meds.get(case), span_med_us=statistics.median(p["span"] for p in per),
                         union_med_us=statistics.median(p["union"] for p in per), seg_types=per[0]["seg_types"],
                         model_key=case, F=mrow["F"], L=mrow["L"], f_req=mrow["f_req"],
                         model_t_sol_us=mrow["t_sol_ms"] * 1e3, model_t_ach_us=mrow["t_ach_ms"] * 1e3,
                         model_not_in_bench_ach_us=nib[1], ranks=len(per), rows=rows)
        print("== %s step %.0f us (span %.0f) | model ach %.0f sol %.0f | x%.2f ach" % (
            case, meds.get(case, 0), out[case]["span_med_us"], mrow["t_ach_ms"] * 1e3, mrow["t_sol_ms"] * 1e3,
            meds.get(case, 0) / (mrow["t_ach_ms"] * 1e3)))
        print("   %-24s %9s %9s %9s %9s %7s %9s  %s" % ("group", "busy_us", "expo_us", "ach_us", "sol_us", "derate",
                                                     "expo-ach", "calls"))
        for r in rows:
            print("   %-24s %9.0f %9.0f %9.0f %9.0f %7s %9.0f  %d" % (
                r["group"], r["measured_us"], r["exposed_us"], r["t_ach_us"], r["t_sol_us"],
                "%.2f" % r["derate"] if r["derate"] else "-", r["exposed_gap_us"], r["calls"]))
        for f, gs in FAM.items():
            rs = [r for r in rows if r["group"] in gs]
            e = sum(r["exposed_us"] for r in rs); b = sum(r["measured_us"] for r in rs); a = sum(r["t_ach_us"] for r in rs)
            out[case].setdefault("families", {})[f] = dict(busy_us=b, exposed_us=e, t_ach_us=a, ratio=(b / a if a else None))
        print("   -- families (busy / exposed / ach ms, busy/ach):")
        for f, v in sorted(out[case]["families"].items(), key=lambda kv: -(kv[1]["exposed_us"] - kv[1]["t_ach_us"])):
            print("   %-46s %8.2f %8.2f %8.2f  x%s" % (f, v["busy_us"] / 1e3, v["exposed_us"] / 1e3, v["t_ach_us"] / 1e3,
                                                       "%.2f" % v["ratio"] if v["ratio"] else "-"))
    json.dump(out, open(outp, "w"), indent=1)


FAM = {
    "MegaMoE (router-comm, dispatch, fc1/act/fc2, combine)": ["MOE.megamoe", "MOE.prepare"],
    "KDA core (conv, prepare, recurrence, gated norm)": ["KDA.core"],
    "KDA in/f_b/o projections": ["KDA.in_proj", "KDA.o_proj"],
    "Shared experts (gate_up, act, down)": ["MOE.shared_gate_up", "MOE.shared_act", "MOE.shared_down"],
    "Router + latent down/up + latent norm + top-k": ["MOE.router", "MOE.topk", "MOE.latent_down", "MOE.latent_up",
                                                      "MOE.latent_norm"],
    "DSA projections (qkv_a, q_b/wq, w_uk, w_uv, g_proj, o_proj)": ["DSA.qkv_a", "DSA.q_proj", "DSA.w_uk", "DSA.w_uv",
                                                                     "DSA.g_proj+gate", "DSA.o_proj"],
    "DSA sparse MLA": ["DSA.sparse_mla"],
    "DSA indexer logits + exact top-k + quant/insert": ["DSA.idx_score", "DSA.topk", "DSA.idx_quant+kv_insert"],
    "AttnRes": ["ATTNRES"],
    "Dense layer + embed": ["DENSE", "COMM.embed_RS"],
    "Drafter + tail": ["TAIL.drafter+final"],
    "comm / other": ["COMM.SP_AG", "DSA.shardq_comm", "OTHER"],
}


if __name__ == "__main__":
    main()
