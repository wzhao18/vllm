"""Prefill: time per chunk of F fresh tokens per engine over context L -> fresh tok/s/GPU, req/s/GPU, TTFT.

Engine = one TP group (N GPUs) for TP/SP/DSA-DP layouts, one GPU for DEP / DSA-CP; all engines of the EP domain step in
lock-step with the same chunk.  Chunks are packed from AgentX requests with f_req fresh tokens over a prefix L - f_req.
    tok/s/GPU = F / (N * t_step)            (PP: F / (S * max stage time))
Variants: base (no overlap), O1 (two-microbatch overlap), O1+O2 (+ dispatch dedup and pre-combine).
"""
import math

import cost
import model as Mdl
from hw import GiB, HWS
from kv import RESERVE_GIB

F_SCAN = [4096, 8192, 16384, 32768, 65536]
L_SCAN = [32768, 65536, 98304, 131072, 150000]
L_REF = Mdl.AGENTX["prefix_fresh_weighted"] + Mdl.AGENTX["fresh_per_req"]     # ~78.4K
F_REF = 16384
F_REQ = Mdl.AGENTX["fresh_per_req"]
ACT_B_PER_ROW = 0.5e6          # live activation workspace per row per GPU (AttnRes blocks + MoE routes + KDA proj)
VARIANTS = {"base": dict(o1=False, o2=False), "O1": dict(o1=True, o2=False), "O1+O2": dict(o1=True, o2=True)}


def _partition(times, S):
    """Contiguous split of a list of times into S stages minimising the max stage sum (DP)."""
    n = len(times)
    pre = [0.0]
    for x in times:
        pre.append(pre[-1] + x)
    INF = float("inf")
    best = [[INF] * (n + 1) for _ in range(S + 1)]
    cut = [[0] * (n + 1) for _ in range(S + 1)]
    best[0][0] = 0.0
    for s in range(1, S + 1):
        for j in range(1, n + 1):
            for i in range(s - 1, j):
                v = max(best[s - 1][i], pre[j] - pre[i])
                if v < best[s][j]:
                    best[s][j] = v
                    cut[s][j] = i
    bounds, j = [], n
    for s in range(S, 0, -1):
        i = cut[s][j]
        bounds.append((i, j))
        j = i
    return best[S][n], bounds[::-1]


def run_point(t, ck, col, F, L, f_req=F_REQ, variant="base", detail=False, moe_act=None):
    """Variants with O1 take the better of overlap on / off (overlap only where it helps)."""
    v = VARIANTS[variant]
    if v["o1"]:
        a = _run(t, ck, col, F, L, f_req, True, v["o2"], detail, moe_act)
        b = _run(t, ck, col, F, L, f_req, False, v["o2"], detail, moe_act)
        return a if a["t_ms"] <= b["t_ms"] else b
    return _run(t, ck, col, F, L, f_req, False, v["o2"], detail, moe_act)


def _run(t, ck, col, F, L, f_req, o1, o2, detail, moe_act=None):
    Fe = F / 2.0 if o1 else F
    env = Mdl.env_prefill(Fe, L, f_req=min(f_req, Fe))
    res = cost.evaluate(t, "prefill", ck, col, env, o2=o2, ops_detail=detail, moe_act=moe_act)
    st = cost.step(res, o1=o1, reps=2 if o1 else 1)
    hw = HWS[col]
    if t.S > 1:
        # pipeline: per-layer times (+globals at the ends) split into S balanced stages; boundary push per token
        lay = list(st["per_layer"])
        lay[0] += st["glob"]["embed"]
        lay[-1] += st["glob"]["final"] + st["glob"]["drafter"] + st["glob"]["step"]
        tmax, bounds = _partition(lay, t.S)
        bnd = 0.0
        for (i, j) in bounds[:-1]:
            nb = 1 + math.ceil(j / 12.0)
            bnd = max(bnd, F * nb * cost.ROW_B / hw.nvl + hw.hop_s)
        t_eff = tmax + bnd
        tok_gpu = F / (t.S * t_eff)
        stages = [b for b in bounds]
    else:
        t_eff = st["t"]
        tok_gpu = F / (t.N * t_eff)
        stages = None
    w, wby = cost.resident_weights(t, "prefill", ck)
    rows_gpu = F / t.N
    n_req = F / min(f_req, F)
    kv_rep = t.N if t.dsa == "rep" else 1
    chunk_state = n_req * (L * Mdl.DSA_KV_PER_TOKEN * kv_rep + 0.47e9) / t.N
    need = w + RESERVE_GIB * GiB + rows_gpu * ACT_B_PER_ROW + chunk_state
    out = dict(F=F, L=L, o1=o1, t_ms=t_eff * 1e3, tok_s_gpu=tok_gpu, req_s_gpu=tok_gpu / F_REQ, weights_GB=w / 1e9,
               free_GB=(hw.hbm_budget - w) / 1e9, feasible=need <= hw.hbm_budget, need_GB=need / 1e9,
               groups={k: x * 1e3 for k, x in st["groups"].items()},
               fam={k: x * 1e3 for k, x in st["fam"].items()}, comm_by={k: x * 1e3 for k, x in st["comm_by"].items()},
               comm_raw={k: x * 1e3 for k, x in st["comm_raw"].items()},
               comp_ms=st["comp"] * 1e3, comm_ms=st["comm"] * 1e3, kern_ms=st["kern"] * 1e3,
               hops=st["hops"], n_comm=st["n_comm"], topk_select=res["topk_select"],
               experts_ms=(res["megamoe"]["layer"] * 92 * (2 if o1 else 1) * 1e3) if res.get("megamoe") else 0.0,
               rows_per_rank=env["T"] / t.N,
               flops_gpu=res["flops"] * (2 if o1 else 1), bytes_gpu=res["bytes"] * (2 if o1 else 1))
    if stages:
        out["stages"] = stages
    if detail:
        out["units"] = {k: dict(comp=x["comp"] * 1e3, comm=x["comm"] * 1e3) for k, x in cost.unit_summary(res).items()}
        out["per_layer_ms"] = [x * 1e3 for x in st["per_layer"]]
        out["glob_ms"] = {k: x * 1e3 for k, x in st["glob"].items()}
        out["ops"] = res["rows"]
        out["comm_rows"] = res["comm_rows"]
    return out


def ttft(t, ck, col, f, prefix, F):
    """Lower-bound TTFT of one request alone on its engine: chunks of min(F, remaining)."""
    done, tt = 0.0, 0.0
    while done < f - 1e-9:
        c = min(F, f - done)
        env_L = prefix + done + c
        r = run_point(t, ck, col, c, env_L, f_req=c)
        tt += r["t_ms"]
        done += c
    return tt


def nixl_bytes(L, k=3):
    s = Mdl.per_request_state_bytes(k, L)
    return dict(dsa_GB=s["dsa"] / 1e9, kda_GB=(s["kda"] + s["conv"]) / 1e9, swa_GB=s["swa"] / 1e9,
                total_GB=(s["dsa"] + s["kda"] + s["conv"] + s["swa"]) / 1e9)


def fan_in(t):
    if t.dsa == "dcp":
        return "DSA KV from %d ranks (t mod %d), KDA state from %d head shards" % (t.N, t.N, t.N)
    if t.dsa in ("dsadp", "dsadp_split"):
        return "DSA KV from the owner card, KDA state from %d head shards" % t.N
    if t.dsa == "rep":
        return "DSA KV from any one rank (replicated), KDA from %d head shards" % t.N
    if t.dsa == "dsacp":
        return "DSA KV from %d ranks, KDA state from 1 GPU" % t.G
    if t.S > 1:
        return "KV/state from %d pipeline stages (one GPU each)" % t.S
    return "everything from 1 GPU"
