"""gap_parse.py: C512 decode step torch-profiler traces -> per model-op measured time (us/step), vs the perf model.

    python3 gap_parse.py decode <results dir (has prof-agentx/, agentx.log)> <model_rows.json> <out.json> [--L 190]

Segmentation: one replay's forward kernels are cut at the AttnRes mix kernels (attn_res_fwd: one before each attention
and each MLP sublayer, one final). A segment is KDA (KdaWide / kda), DSA (MlaQuery / fmha / mqa_logits), MOE (mega_moe),
DENSE (first MLP, no MoE) or TAIL (after the last MoE segment: final AttnRes, DSpark drafter, sampling glue). Inside a
segment, GEMM kernels (nvjet / cutlass GEMM / CuTe split-K GEMM / O-proj) are assigned by ordinal, a cuBLAS split-K
reduce follows its GEMM. Named kernels are assigned by regex. Measured = sum of kernel durations per step (same-stream PDL overlap clipped: a kernel's time starts when its
predecessor on the stream ends) (median of the
3 profiled replays, then median over ranks; rank-max also kept). Kernels on a side stream overlap, so the per-op sums
add up to more than the step (busy_sum > span); spin-wait kernels (QueryWait) are reported as WAIT, not as work.
"""
import gzip
import json
import os
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kb_parse as K  # noqa: E402

GEMM = re.compile(r"nvjet|splitkgemm|spoprojwarp|ll_bf16_splitk|cutlass.*gemm|gemm.*cutlass|xmma|sm\d+_xmma|cublaslt.*gemm|"
                  r"inproj64|gatherprojection|_gemm")
REDUCE = re.compile(r"splitkreduce")

# op groups: name -> model op ids
GROUPS = {
    "KDA.in_proj": ["KDA.in_proj", "KDA.f_b_proj"],
    "KDA.core": ["KDA.conv", "KDA.gate_beta_norm", "KDA.core_decode", "KDA.o_norm"],
    "KDA.o_proj": ["KDA.o_proj"],
    "DSA.qkv_a": ["DSA.qkv_a_packed"],
    "DSA.q_path": ["DSA.qb_wq_packed", "DSA.w_uk", "DSA.idx_q_quant", "COMM.P6.dsa"],
    "DSA.prologue": ["DSA.prologue"],
    "DSA.g_proj": ["DSA.g_proj"],
    "DSA.idx_score": ["DSA.idx_score"],
    "DSA.topk": ["DSA.topk"],
    "DSA.topk_merge": ["COMM.P8.dsa"],
    "DSA.sparse_mla": ["DSA.sparse_mla"],
    "DSA.w_uv": ["DSA.w_uv", "DSA.gate_mul"],
    "DSA.dcp_value": ["COMM.P7b.dsa"],
    "DSA.o_proj": ["DSA.o_proj"],
    "COMM.SP_AG": ["COMM.P2.kda", "COMM.P2.dsa"],
    "COMM.SP_RS": ["COMM.P3b.kda", "COMM.P3b.dsa", "COMM.P3b.embed"],
    "ATTNRES": ["L.attnres_pre", "L.attnres_post", "F.attnres_final", "G.attnres_init"],
    "MOE.router+latent_down": ["MOE.router", "MOE.latent_down"],
    "MOE.topk": ["MOE.topk"],
    "MOE.latent_norm": ["MOE.latent_norm"],
    "MOE.megamoe": ["MOE.dispatch", "MOE.fc1", "MOE.act", "MOE.fc2", "MOE.combine", "COMM.P11.moe", "COMM.P12.moe"],
    "MOE.prepare": [],
    "MOE.shared_gate_up": ["MOE.shared_gate_up"],
    "MOE.shared_act": ["MOE.shared_act"],
    "MOE.shared_down": ["MOE.shared_down"],
    "MOE.latent_up": ["MOE.latent_up"],
    "DENSE": ["DENSE.gate_up", "DENSE.act", "DENSE.down"],
    "EMBED": ["G.embed"],
    "TAIL.drafter+final": ["DS.aux_capture", "DS.fc", "DS.hidden_norm", "DS.ctx_kv_proj", "DS.ctx_kv_insert", "DS.embed",
                           "DS.norm_ar", "DS.qkv", "DS.attn", "DS.o_proj", "DS.gate_up", "DS.silu", "DS.down",
                           "DS.lm_head", "DS.markov", "DS.confidence", "F.norm", "COMM.P2.drafter", "COMM.P18.drafter",
                           "COMM.P1c.drafter", "COMM.P16.drafter", "COMM.P2.final"],
    "WAIT": [],
    "OTHER": [],
}
NOT_IN_BENCH = ["F.lm_head", "F.rejection", "COMM.P16.final", "COMM.P19.step"]  # target LM head/rejection: not in graph


def shortn(name):
    s = K.short(name)
    s = re.sub(r"^kernel_cutlass_(kernel_|tiled_)?(projected_)?vllmmodelskimi_k3nvidiaspecialized_decodecutedsl", "sd:", s)
    s = re.sub(r"^kernel_cutlass_(kernel_|tiled_)?vllmmodel_executorkernels", "vk:", s)
    return s[:60]


def exposed(labeled):
    """Additive split of the busy union: each instant is shared equally by the kernels running then (WAIT kernels only
    get instants where nothing else runs). Returns {group: us}."""
    pts = []
    for g, e in labeled:
        if e["cdur"] <= 0:
            continue
        pts.append((e["cts"], 1, g))
        pts.append((e["cts"] + e["cdur"], -1, g))
    pts.sort(key=lambda x: (x[0], x[1]))
    act = defaultdict(int)
    out = defaultdict(float)
    prev = None
    for t, d, g in pts:
        if prev is not None and t > prev:
            live = [k for k, v in act.items() if v > 0]
            work = [k for k in live if k != "WAIT"] or live
            for k in work:
                out[k] += (t - prev) * act[k] / sum(act[x] for x in work)
        act[g] += d
        prev = t
    return out


def seg_type(names):
    j = " ".join(names)
    if "mega_moe" in j or "megamoe" in j:  # deep_gemm mega_moe or the vendored megamoe_rubin (kimi_k3_mega_moe)
        return "MOE"
    if "kdawide" in j or re.search(r"kda", j) and "draft" not in j:
        return "KDA"
    if "mlaquery" in j or "fmha" in j or "mqa_logits" in j:
        return "DSA"
    if "situ_and_mul" in j or "act_and_mul" in j:
        return "DENSE"
    return "MISC"


def assign(seg, stype):
    """[(group, event)] for one segment of kernels (events in ts order)."""
    out = []
    gemms = []
    last = None
    moe_seen = False
    for e in seg:
        n = e["name"].lower()
        g = None
        if "attn_res" in n:
            g = "ATTNRES"
        elif "spallgather" in n or "sp_all_gather" in n:
            g = "COMM.SP_AG"
        elif "spreducescatter" in n or "reduce_scatter" in n:
            g = "COMM.SP_RS"
        elif "allreduce" in n and stype != "TAIL":
            g = "COMM.SP_RS"
        elif "querywait" in n:
            g = "WAIT"
        elif stype == "TAIL":
            g = "TAIL.drafter+final"
        elif "mega_moe" in n or "megamoe" in n or (stype == "MOE" and n.startswith("_stage_kernel")):
            g = "MOE.megamoe"; moe_seen = True
        elif "prepare_megamoe" in n:
            g = "MOE.prepare"
        elif "topk_merge" in n:
            g = "DSA.topk_merge"
        elif "topk_pp" in n or "deep_select" in n or ("topk" in n and stype == "DSA"):
            g = "DSA.topk"
        elif "single_group_topk" in n or "grouped_topk" in n or ("topk" in n and stype == "MOE"):
            g = "MOE.topk"
        elif "mqa_logits" in n:
            g = "DSA.idx_score"
        elif "dsaindex" in n:
            g = "DSA.prologue"
        elif "mlaquery" in n or "querynorm" in n:
            g = "DSA.q_path"
        elif "fmha" in n or "computelse" in n or "sparse_mla" in n:
            g = "DSA.sparse_mla"
        elif "dcp_value" in n or "dcpvalue" in n:
            g = "DSA.dcp_value"
        elif "inproj64" in n:
            g = "KDA.in_proj"
        elif "kdawide" in n or "kda" in n:
            g = "KDA.core"
        elif "situ_and_mul" in n or "act_and_mul" in n:
            g = "MOE.shared_act" if stype == "MOE" else "DENSE"
        elif "rms_norm" in n and stype == "MOE":
            g = "MOE.latent_norm"
        elif "embedding" in n:
            g = "EMBED"
        elif REDUCE.search(n) and last is not None:
            g = last
        elif GEMM.search(n):
            k = len(gemms)
            gemms.append(e)
            if stype == "KDA":
                g = ["KDA.in_proj", "KDA.o_proj"][min(k, 1)]
            elif stype == "DSA":
                g = ["DSA.qkv_a", "DSA.g_proj", "DSA.w_uv", "DSA.o_proj"][min(k, 3)]
            elif stype == "MOE":
                if not moe_seen:
                    g = "MOE.router+latent_down"
                else:
                    post = sum(1 for x in gemms if x.get("_post"))
                    e["_post"] = True
                    g = ["MOE.shared_gate_up", "MOE.shared_down", "MOE.latent_up"][min(post, 2)]
            elif stype == "DENSE":
                g = "DENSE"
            else:
                g = "OTHER"
        else:
            g = "OTHER"
        if GEMM.search(n) and not REDUCE.search(n):
            last = g
        out.append((g, e))
    return out


def analyze(trace):
    kj = json.load(open(trace.replace(".trace.json.gz", ".kernels.json")))
    ev = json.load(gzip.open(trace, "rt"))["traceEvents"]
    acts = sorted((e for e in ev if e.get("cat") in K.GPU_CATS and "dur" in e), key=lambda e: e["ts"])
    reps = []
    for pre, fwd in K.split_replays(acts, kj):
        if not fwd:
            continue
        # PDL: a kernel launched before its same-stream predecessor ends waits inside (griddepcontrol.wait); its work
        # starts at the predecessor's end. Clip same-stream overlap to the earlier kernel (cdur, cts).
        lastend = {}
        for e in fwd:
            sid = e.get("args", {}).get("stream", e.get("tid"))
            s0 = max(e["ts"], lastend.get(sid, e["ts"]))
            e["cts"] = s0
            e["cdur"] = max(0.0, e["ts"] + e["dur"] - s0)
            lastend[sid] = max(lastend.get(sid, 0.0), e["ts"] + e["dur"])
        # segments at AttnRes kernels
        segs, cur = [], []
        for e in fwd:
            if "attn_res" in e["name"].lower() and cur:
                segs.append(cur)
                cur = []
            cur.append(e)
        segs.append(cur)
        types = [seg_type([x["name"].lower() for x in s]) for s in segs]
        last_moe = max(i for i, t in enumerate(types) if t == "MOE")
        tot = defaultdict(float)
        names = defaultdict(lambda: defaultdict(float))
        cnt = defaultdict(int)
        labeled = []
        for i, s in enumerate(segs):
            t = "TAIL" if i > last_moe else types[i]
            if t == "MISC" and i == 0:
                t = "HEAD"
            for g, e in assign(s, t):
                tot[g] += e["cdur"]
                labeled.append((g, e))
                names[g][shortn(e["name"])] += e["cdur"]
                cnt[g] += 1
        t0 = fwd[0]["ts"]
        t1 = max(e["ts"] + e["dur"] for e in fwd)
        reps.append(dict(tot=tot, names=names, cnt=cnt, span=t1 - t0, exp=exposed(labeled),
                         union=K.union_us([(e["ts"], e["ts"] + e["dur"]) for e in fwd]),
                         seg_types="".join({"KDA": "k", "DSA": "d", "MOE": "m", "DENSE": "D", "MISC": "x"}[x] for x in types)))
    med = statistics.median
    groups = set().union(*(r["tot"] for r in reps))
    res = dict(span=med([r["span"] for r in reps]), union=med([r["union"] for r in reps]),
               seg_types=reps[0]["seg_types"],
               tot={g: med([r["tot"].get(g, 0.0) for r in reps]) for g in groups},
               exp={g: med([r["exp"].get(g, 0.0) for r in reps]) for g in groups},
               cnt={g: reps[0]["cnt"].get(g, 0) for g in groups})
    nm = defaultdict(dict)
    for g in groups:
        for n, v in reps[0]["names"][g].items():
            nm[g][n] = round(v, 1)
    res["names"] = nm
    return res


def model_sum(mrow, ids):
    s = a = 0.0
    for i in ids:
        if i.startswith("COMM."):
            c = mrow["comm"].get(i)
            if c:
                s += c["t_sol_us"]; a += c["t_ach_us"]
        else:
            o = mrow["ops"].get(i)
            if o:
                s += o["t_sol_us"]; a += o["t_ach_us"]
    return s, a


def main():
    mode, root, mpath, outp = sys.argv[1:5]
    Lk = 190
    if "--L" in sys.argv:
        Lk = int(sys.argv[sys.argv.index("--L") + 1])
    model = json.load(open(mpath))
    meds = {}
    for log in [os.path.join(root, "agentx.log")]:
        for line in open(log, errors="replace"):
            m = re.match(r"\[forward\] (\S+): .*median ([0-9.]+) us", line)
            if m:
                meds[m.group(1)] = float(m.group(2))
    prof = os.path.join(root, "prof-agentx")
    cases = sorted({f.split(".rank")[0] for f in os.listdir(prof) if f.endswith(".trace.json.gz")})
    out = {}
    for case in cases:
        B = int(re.search(r"_B(\d+)_", case).group(1))
        mrow = model["decode"].get("B%d_L%dK" % (B, Lk))
        ranks = sorted(f for f in os.listdir(prof) if f.startswith(case + ".rank") and f.endswith(".trace.json.gz"))
        per = [analyze(os.path.join(prof, f)) for f in ranks]
        groups = set().union(*(p["tot"] for p in per))
        rows = []
        for g in sorted(groups):
            vals = [p["tot"].get(g, 0.0) for p in per]
            ids = GROUPS.get(g, [])
            s, a = model_sum(mrow, ids) if mrow else (0, 0)
            ex = statistics.median([p["exp"].get(g, 0.0) for p in per])
            rows.append(dict(group=g, op_ids=ids, measured_us=statistics.median(vals), measured_rank0_us=vals[0],
                             exposed_us=ex, exposed_gap_us=ex - a,
                             measured_max_us=max(vals), t_sol_us=s, t_ach_us=a,
                             derate=(statistics.median(vals) / a) if a > 0 else None,
                             gap_us=statistics.median(vals) - a, calls=per[0]["cnt"].get(g, 0),
                             kernels=per[0]["names"].get(g, {})))
        rows.sort(key=lambda r: -r["exposed_gap_us"])
        nib = model_sum(mrow, NOT_IN_BENCH) if mrow else (0, 0)
        out[case] = dict(B=B, step_median_us=meds.get(case), span_med_us=statistics.median(p["span"] for p in per),
                         union_med_us=statistics.median(p["union"] for p in per), seg_types=per[0]["seg_types"],
                         model_key="B%d_L%dK" % (B, Lk),
                         model_t_sol_us=mrow["t_sol_ms"] * 1e3 if mrow else None,
                         model_t_ach_us=mrow["t_ach_ms"] * 1e3 if mrow else None,
                         model_not_in_bench_ach_us=nib[1], model_launch_ach_us=(mrow["fam_ach"].get("launch", 0) * 1e3
                                                                               if mrow else None),
                         ranks=len(per), rows=rows)
        print("== %s B%d step %.0f us (span %.0f, union %.0f) | model ach %.0f us (minus LM head/rejection %.0f) sol %.0f"
              % (case, B, meds.get(case, 0), out[case]["span_med_us"], out[case]["union_med_us"],
                 out[case]["model_t_ach_us"], out[case]["model_t_ach_us"] - nib[1], out[case]["model_t_sol_us"]))
        print("   %-24s %9s %9s %9s %9s %7s %9s  %s" % ("group", "busy_us", "expo_us", "ach_us", "sol_us", "derate",
                                                     "expo-ach", "calls"))
        for r in rows:
            print("   %-24s %9.1f %9.1f %9.1f %9.1f %7s %9.1f  %d" % (
                r["group"], r["measured_us"], r["exposed_us"], r["t_ach_us"], r["t_sol_us"],
                "%.2f" % r["derate"] if r["derate"] else "-", r["exposed_gap_us"], r["calls"]))
        print("   sum exposed %.0f us, sum ach %.0f us" % (sum(r["exposed_us"] for r in rows), sum(r["t_ach_us"] for r in rows)))
    json.dump(out, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
