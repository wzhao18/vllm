"""evid.py <dir> <case> <model_rows.json> <modelkey>: evidence stats for the top gap contributors."""
import gzip
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gap_parse as G  # noqa: E402
import kb_parse as K  # noqa: E402

root, case, mpath, mkey = sys.argv[1:5]
model = json.load(open(mpath))["decode"][mkey]
prof = os.path.join(root, "prof-agentx")
ranks = sorted((f for f in os.listdir(prof) if f.startswith(case + ".rank") and f.endswith(".trace.json.gz")),
               key=lambda f: int(f.split(".rank")[1].split(".")[0]))
per_rank = {}
calls = {}
for f in ranks:
    tr = os.path.join(prof, f)
    kj = json.load(open(tr.replace(".trace.json.gz", ".kernels.json")))
    ev = json.load(gzip.open(tr, "rt"))["traceEvents"]
    acts = sorted((e for e in ev if e.get("cat") in K.GPU_CATS and "dur" in e), key=lambda e: e["ts"])
    pre, fwd = K.split_replays(acts, kj)[1]
    segs, cur = [], []
    for e in fwd:
        if "attn_res" in e["name"].lower() and cur:
            segs.append(cur)
            cur = []
        cur.append(e)
    segs.append(cur)
    types = [G.seg_type([x["name"].lower() for x in s]) for s in segs]
    last_moe = max(i for i, t in enumerate(types) if t == "MOE")
    r = int(f.split(".rank")[1].split(".")[0])
    d = {}
    for i, s in enumerate(segs):
        t = "TAIL" if i > last_moe else types[i]
        for g, e in G.assign(s, t):
            key = (g, G.shortn(e["name"]), str(e.get("args", {}).get("grid")))
            d.setdefault(key, []).append(e["dur"])
    per_rank[r] = d
keys = sorted(set().union(*per_rank.values()), key=lambda k: -sum(per_rank[0].get(k, [0])))
print("kernel (group | name | grid): per-call median us rank0, per-rank step-sum min/med/max, calls")
for k in keys[:40]:
    sums = [sum(per_rank[r].get(k, [])) for r in per_rank]
    c0 = per_rank[0].get(k, [])
    if not c0:
        continue
    print("  %-22s %-46s %-14s call %6.1f (p10 %6.1f p90 %6.1f) | sum %7.0f %7.0f %7.0f | n %d" % (
        k[0], k[1][:46], k[2], st.median(c0), sorted(c0)[len(c0) // 10], sorted(c0)[len(c0) * 9 // 10],
        min(sums), st.median(sums), max(sums), len(c0)))
# achieved HBM bandwidth of GEMM groups (model bytes per call / measured per call incl. split-K reduce)
print("GEMM groups: model bytes/call, measured us/call (incl reduce), achieved TB/s, model ach us/call")
ops = model["ops"]
import re  # noqa: E402
for g, ids in G.GROUPS.items():
    if not any(i in ops for i in ids):
        continue
    if g not in ("KDA.in_proj", "KDA.o_proj", "DSA.qkv_a", "DSA.g_proj", "DSA.o_proj", "DSA.w_uv", "MOE.router+latent_down",
                 "MOE.shared_gate_up", "MOE.shared_down", "MOE.latent_up", "KDA.core", "MOE.megamoe", "DSA.sparse_mla",
                 "DSA.idx_score", "ATTNRES"):
        continue
    meas = st.median([sum(sum(v) for k, v in per_rank[r].items() if k[0] == g) for r in per_rank])
    n = ops[ids[0]]["count"]
    ach = sum(ops[i]["t_ach_us"] for i in ids if i in ops)
    print("  %-24s measured %7.1f us/call  model ach %6.2f us/call  derate %.2f" % (g, meas / n, ach / n, meas / ach))
