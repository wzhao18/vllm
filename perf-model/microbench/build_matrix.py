"""Build the microbenchmark matrix: distinct per-GPU op shapes for the priority topologies.

    python3 k31vr/solmodel/microbench/build_matrix.py      # -> microbench/matrix.json (stdlib only, ~2 s)

Times come from out/optables.json (run.py output). Shapes are derived from inputs/ops.json op definitions, the step
environment (model.env_prefill / env_decode at the optables regimes) and the per-GPU divisors of topologies.op_map;
every GEMM shape is cross-checked against the optables per-GPU FLOPs (assert below). Comm cases are rebuilt from
topologies.comm_events (the optables comm rows carry no N) and their times checked against the optables rows.

Priority = sum over uses of (share of step, achievable column) x (gap prior - 1), where the gap prior is the measured
step / model-achievable ratio of that topology (or the nearest measured proxy, marked).

Scope (2026-11-18): the model scope is topologies.SCOPE (EP <= 16, <= 16 GPUs per engine / EP domain). Three tiers of
topologies: PRIORITY (ranked), COVER (rest of SCOPE: listed with weight 0, never P0/P1 on their own) and OOS (the
pre-rescope EP64/EP32 ids, kept only so their case ids stay valid; priority 'OOS').

Case ids are stable: ids_v1.json maps (family, shape, dtype) -> id; new keys get ids after the highest existing number.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import cost  # noqa: E402
import decode as Dec  # noqa: E402
import model as Mdl  # noqa: E402
import prefill as Pre  # noqa: E402
from hw import HWS  # noqa: E402
from topologies import ALL, SCOPE, comm_events, op_map  # noqa: E402

C = Mdl.C
OPT = json.load(open(os.path.join(ROOT, "out", "optables.json")))
COLS = OPT["cols"]
GAP_S = HWS["achievable"].kernel_s

# PRIORITY: deployed layouts + the EP<=16 frontier (results/model/README.md); COVER: rest of topologies.SCOPE (weight 0);
# OOS: EP64 / EP32 layouts dropped by the 2026-11-18 scope rule (kept for id continuity only).
PREFILL_T = ["P1", "P2", "P3-16", "P4-2", "P6-16", "P9"]
DECODE_T = ["D1", "D3", "D4-16", "D5", "D7-2-16", "D7-4-16", "D7-8-16", "D8-16"]
COVER_T = [t for t in SCOPE if t not in PREFILL_T + DECODE_T]
OOS_T = ["P6", "D4-32", "D7-4", "D7-2", "D7-8", "D8"]
CKS = ["mxfp4_release", "nvfp4_modelopt"]
assert all(t in SCOPE for t in PREFILL_T + DECODE_T) and not set(OOS_T) & set(SCOPE)
WEIGHT = {t: 1.0 for t in PREFILL_T + DECODE_T}
WEIGHT.update({t: 0.0 for t in COVER_T + OOS_T})
TIER_OF = {t: "priority" for t in PREFILL_T + DECODE_T}
TIER_OF.update({t: "cover" for t in COVER_T})
TIER_OF.update({t: "oos" for t in OOS_T})

# measured step / model-achievable (out/meta.json anchors). 'proxy' = nearest measured layout.
GAP = {
    ("P1", None): (1.18, "PR#20 TP8+DCP8 24-layer prefill"),
    ("P2", None): (1.74, "C512 P2 prefill (MXFP8 experts headline)"),
    ("P3-16", None): (1.33, "PR#20 DEP12 (proxy)"),
    ("P4-2", None): (1.32, "PR#20 TP4xDP3 DSA-DP (proxy)"),
    ("P6", None): (1.32, "PR#20 DSA-DP (proxy; EP64 dedup unmeasured)"),
    ("P6-16", None): (1.32, "PR#20 DSA-DP (proxy; EP16 dedup unmeasured)"),
    ("P6-4-16", None): (1.32, "PR#20 TP4xDP3 DSA-DP (proxy)"),
    ("P9", None): (1.33, "PR#20 DEP12 (proxy; DSA-CP gather unmeasured)"),
    ("D1", 1): (1.12, "D1 e2e B~16 (C16-C72)"),
    ("D1", 8): (1.60, "D1 e2e C224-C256 (B70-89, nearest)"),
    ("D1", 32): (1.67, "D1 e2e C256 (B89, extrapolated)"),
    ("D3", 1): (1.93, "C512 D3 B4"),
    ("D3", 8): (1.71, "C512 D3 B32"),
    ("D3", 32): (1.92, "C512 D3 B64 (r=16, nearest)"),
    ("D8", None): (1.40, "D1 attention + unmeasured EP64 (proxy)"),
    ("D8-16", None): (1.40, "D1 attention + EP16 MegaMoE (proxy)"),
}
DEFAULT_DEC_GAP = (1.80, "proxy: C512 D3 hybrid decode 1.7-1.9x (no anchor for this layout)")
DEFAULT_PRE_GAP = (1.33, "proxy: PR#20 DEP12 prefill (no anchor for this layout)")


def gap_prior(tid, r):
    if (tid, r) in GAP:
        return GAP[(tid, r)]
    if (tid, None) in GAP:
        return GAP[(tid, None)]
    return DEFAULT_PRE_GAP if tid.startswith("P") else DEFAULT_DEC_GAP


# ---------------------------------------------------------------------------------------------- GEMM dims
# op -> (M symbol, N, K, shard dim of a head/column-shard map: 'n' column-parallel, 'k' row-parallel, 'h' bmm heads)
GEMM = {
    "KDA.in_proj": ("T", 4 * C["P"] + C["dk"] + C["Hk"], C["H"], "n"),
    "KDA.f_b_proj": ("T", C["P"], C["dk"], "n"),
    "KDA.o_proj": ("T", C["H"], C["P"], "k"),
    "DSA.qkv_a_packed": ("T", C["QLR"] + C["KVL"] + C["DR"] + C["Di"] + C["Ni"], C["H"], "n"),
    "DSA.qb_wq_packed": ("T", C["Nh"] * C["DQK"] + C["Ni"] * C["Di"], C["QLR"], "n"),
    "DSA.g_proj": ("T", C["Nh"] * C["DV"], C["H"], "n"),
    "DSA.w_uk": ("T", C["KVL"], C["DN"], "h"),
    "DSA.w_uv": ("T", C["DV"], C["KVL"], "h"),
    "DSA.o_proj": ("T", C["H"], C["Nh"] * C["DV"], "k"),
    "MOE.router": ("T", C["E"], C["H"], "n"),
    "MOE.latent_down": ("T", C["LAT"], C["H"], "n"),
    "MOE.latent_up": ("T", C["H"], C["LAT"], "k"),
    "MOE.shared_gate_up": ("T", 2 * C["I_SH"], C["H"], "n"),
    "MOE.shared_down": ("T", C["H"], C["I_SH"], "k"),
    "DENSE.gate_up": ("T", 2 * C["I_D"], C["H"], "n"),
    "DENSE.down": ("T", C["H"], C["I_D"], "k"),
    "F.lm_head": ("R", C["V"], C["H"], "n"),
    "DS.fc": ("Tc", C["H"], C["D_AUX"] * C["H"], "n"),
    "DS.ctx_kv_proj": ("Tc", C["D_L"] * 2 * C["D_HKV"] * C["D_HD"], C["H"], "n"),
    "DS.qkv": ("Tq", C["D_HQ"] * C["D_HD"] + 2 * C["D_HKV"] * C["D_HD"], C["H"], "n"),
    "DS.o_proj": ("Tq", C["H"], C["D_HQ"] * C["D_HD"], "k"),
    "DS.gate_up": ("Tq", 2 * C["D_I"], C["H"], "n"),
    "DS.down": ("Tq", C["H"], C["D_I"], "k"),
    "DS.lm_head": ("Tq", C["V"], C["H"], "n"),
}
GEMM_ROLE = {  # human label of the weight being multiplied
    "KDA.in_proj": "KDA in_proj (q,k,v,g1 + f_a + beta)", "KDA.f_b_proj": "KDA f_b", "KDA.o_proj": "KDA o_proj",
    "DSA.qkv_a_packed": "MLA q_a|kv_a|k_pe|idx_k|idx_w packed", "DSA.qb_wq_packed": "MLA q_b | indexer wq packed",
    "DSA.g_proj": "MLA output gate", "DSA.w_uk": "MLA W_UK bmm", "DSA.w_uv": "MLA W_UV bmm", "DSA.o_proj": "MLA o_proj",
    "MOE.router": "MoE router", "MOE.latent_down": "MoE latent down", "MOE.latent_up": "MoE latent up",
    "MOE.shared_gate_up": "shared experts gate_up", "MOE.shared_down": "shared experts down",
    "DENSE.gate_up": "L0 dense gate_up", "DENSE.down": "L0 dense down", "F.lm_head": "target lm_head",
    "DS.fc": "DSpark fc (5x7168)", "DS.ctx_kv_proj": "DSpark ctx K/V proj", "DS.qkv": "DSpark qkv",
    "DS.o_proj": "DSpark o_proj", "DS.gate_up": "DSpark gate_up", "DS.down": "DSpark down", "DS.lm_head": "DSpark lm_head",
}

KDA_OPS = ("KDA.conv", "KDA.gate_beta_norm", "KDA.core_prefill", "KDA.core_decode", "KDA.o_norm")
IDX_OPS = ("DSA.idx_q_quant", "DSA.idx_score", "DSA.topk")
MLA_OPS = ("DSA.prologue", "DSA.sparse_mla", "DSA.gate_mul")
MOE_OPS = ("MOE.dispatch", "MOE.fc1", "MOE.act", "MOE.fc2", "MOE.combine")
ATTNRES_OPS = ("L.attnres_pre", "L.attnres_post", "F.attnres_final", "DS.aux_capture", "G.embed", "G.attnres_init")
DRAFT_OPS = ("DS.attn", "DS.markov", "DS.embed", "DS.norm_ar", "DS.confidence", "DS.hidden_norm", "DS.ctx_kv_insert",
             "DS.silu", "F.rejection", "F.norm")
EPI_OPS = ("MOE.topk", "MOE.latent_norm", "MOE.shared_act", "DENSE.act")


def isint(x):
    return abs(x - round(x)) < 1e-6


def ri(x):
    return int(round(x)) if isint(x) else round(x, 2)


def gemm_cases(oid, t, phase, env, row):
    """Per-GPU sub-GEMMs [(batch, M, N, K, note)] of one GEMM op on topology t."""
    msym, Nn, K, sd = GEMM[oid]
    Mf = env[msym]
    m = op_map(oid, t, phase)
    nd = t.ND if oid.startswith("DS.") else t.N
    batch = C["Nh"] if sd == "h" else 1
    w, r = m["w"], m["row"]
    if abs(w - 1) < 1e-9 and abs(r - 1) < 1e-9:
        return [(batch, Mf, Nn, K, "replicated")]
    if abs(w - 1) < 1e-9:                                 # token shard (weights replicated)
        if oid == "DSA.qb_wq_packed" and t.dsa == "dcp" and phase == "prefill":   # ShardQ prefill: q_b on the slice, wq on all rows
            qn = C["Nh"] * C["DQK"]
            return [(1, Mf / nd, qn, K, "q_b on ShardQ slice"), (1, Mf, C["Ni"] * C["Di"], K, "indexer wq all rows")]
        return [(batch, Mf / r, Nn, K, "token-shard /%s" % ri(r))]
    if isint(w) and abs(w - r) < 1e-9:                     # head / column / row-parallel shard
        s = w
        if sd == "n":
            return [(batch, Mf, Nn / s, K, "col-shard /%d" % s)]
        if sd == "k":
            return [(batch, Mf, Nn, K / s, "row-shard K/%d" % s)]
        return [(batch / s, Mf, Nn, K, "heads /%d" % s)]
    if oid == "KDA.in_proj":                               # (4P+Hk) head-sharded, f_a replicated
        return [(1, Mf, (4 * C["P"] + C["Hk"]) / nd + C["dk"], K, "head-TP%d + f_a rep" % nd)]
    if oid == "DSA.qb_wq_packed":
        return [(1, Mf, C["Nh"] * C["DQK"] / nd + C["Ni"] * C["Di"], K, "q_b heads/%d + wq rep" % nd)]
    raise ValueError("unhandled map %s %s %r" % (oid, t.id, m))


# ---------------------------------------------------------------------------------------------- envs
def env_for(t, phase, r):
    if phase == "prefill":
        return Mdl.env_prefill(Pre.F_REF, Pre.L_REF, f_req=min(Pre.F_REQ, Pre.F_REF))
    return Mdl.env_decode(r * t.N, 3, Dec.L_DEC)


def table_rows(key):
    tb = OPT["tables"][key]
    return {r[0]: dict(zip(COLS, r)) for r in tb["ops"]}, tb


# ---------------------------------------------------------------------------------------------- cases
cases = {}
NREG = {}


def add(fam, shape, dtype, use, ops, t_sol, t_ach, t_ach_k, share, flops=None, notes=None):
    key = json.dumps([fam, shape, dtype], sort_keys=True)
    c = cases.get(key)
    if c is None:
        c = cases[key] = dict(family=fam, shape=shape, dtype=dtype, ops=sorted(set(ops)), uses=[],
                              t_sol_us=t_sol * 1e6, t_ach_us=t_ach * 1e6, t_ach_kernel_us=t_ach_k * 1e6,
                              notes=sorted(set(notes or [])))
    c["ops"] = sorted(set(c["ops"]) | set(ops))
    c["t_ach_us_range"] = [min(c.get("t_ach_us_range", [t_ach * 1e6, t_ach * 1e6])[0], t_ach * 1e6),
                           max(c.get("t_ach_us_range", [t_ach * 1e6, t_ach * 1e6])[1], t_ach * 1e6)]
    use["ops"] = list(ops)
    c["notes"] = sorted(set(c["notes"]) | set(notes or []))
    c["uses"].append(use)


def op_t(row):
    return row["t_sol_s"], row["t_ach_s"]


def kern(t, fused):
    return t if fused else max(t, GAP_S)


def run():
    checks = []
    for ck in CKS:
        regs = [(tid, "prefill", None) for tid in PREFILL_T + [x for x in COVER_T + OOS_T if x.startswith("P")]]
        for tid in DECODE_T + [x for x in COVER_T + OOS_T if x.startswith("D")]:
            for r in (1, 8, 32):
                if "%s|%s|decode r%d" % (ck, tid, r) in OPT["tables"]:
                    regs.append((tid, "decode", r))
        for tid, phase, r in regs:
            NREG[(ck, tid)] = sum(1 for x in regs if x[0] == tid)
        for tid, phase, r in regs:
            t = ALL[tid]
            key = "%s|%s|%s" % (ck, tid, "prefill" if phase == "prefill" else "decode r%d" % r)
            rows, tb = table_rows(key)
            step_ach = tb["t_ach_ms"] * 1e-3
            env = env_for(t, phase, r)
            N, T = t.N, env["T"]
            g, gsrc = gap_prior(tid, r)
            reg = "prefill F%d L%.1fK" % (Pre.F_REF, Pre.L_REF / 1e3) if phase == "prefill" else \
                "decode r%d k3 L135K (B_group=%d, T_group=%d)" % (r, r * N, T)

            def use(cnt, t_ach, fused=False, extra=None):
                sh = cnt * kern(t_ach, fused) / step_ach
                u = dict(ckpt=ck, topo=tid, phase=phase, r=r, regime=reg, count=cnt, share=sh, gap_prior=g,
                         gap_src=gsrc, scope=TIER_OF[tid], score=WEIGHT[tid] * sh * (g - 1.0) / NREG[(ck, tid)])
                if extra:
                    u.update(extra)
                return u

            def rowsum(ids):
                ts = ta = tk = 0.0
                cnt = None
                for i in ids:
                    if i in rows:
                        x = rows[i]
                        ts += x["t_sol_s"]
                        ta += x["t_ach_s"]
                        tk += kern(x["t_ach_s"], Mdl.OPS_BY_ID[i].fused)
                        cnt = x["count"]
                return ts, ta, tk, cnt

            # ---- GEMMs
            for oid in GEMM:
                if oid not in rows or rows[oid]["t_ach_s"] <= 0 or rows[oid]["flops"] <= 0:
                    continue
                x = rows[oid]
                subs = gemm_cases(oid, t, phase, env, x)
                fl = sum(2 * b * M * Nn * K for b, M, Nn, K, _ in subs)
                if oid == "DSA.qb_wq_packed" and len(subs) == 2:
                    pass
                checks.append((key, oid, fl, x["flops"]))
                for b, M, Nn, K, note in subs:
                    shape = dict(batch=ri(b), M=ri(M), N=ri(Nn), K=ri(K))
                    frac = 2 * b * M * Nn * K / fl
                    ts, ta = x["t_sol_s"] * frac, x["t_ach_s"] * frac
                    add("gemm", shape, x["dtype"], use(x["count"], ta), [oid], ts, ta, kern(ta, False), 0,
                        notes=[GEMM_ROLE[oid], note])
            # ---- KDA core (conv + gate/beta norm + recurrence + o_norm, one bench per shape)
            ts, ta, tk, cnt = rowsum(KDA_OPS)
            if cnt:
                heads = C["Hk"] / (N if N > 1 else 1)
                if phase == "prefill":
                    shape = dict(phase="prefill", tokens=ri(T), nseq=ri(env["nseq"]), fresh_per_seq=ri(T / env["nseq"]),
                                 heads=ri(heads), chunk=C["C_KDA"], state="fp32", ckpt_export=True)
                else:
                    shape = dict(phase="decode", B=ri(env["B"]), q=int(env["q"]), heads=ri(heads), AL=env["AL"],
                                 state="fp32", conv="fp32")
                core = "KDA.core_prefill" if phase == "prefill" else "KDA.core_decode"
                add("kda", shape, rows[core]["dtype"], use(cnt, ta, extra=dict(t_ach_kernel_us=tk * 1e6)),
                    [i for i in KDA_OPS if i in rows], ts, ta, tk, 0,
                    notes=["FlashKDA chunked prefill" if phase == "prefill" else "sdec KDA recurrence + RecoverSSM records"])
            # ---- indexer
            ts, ta, tk, cnt = rowsum(IDX_OPS)
            if cnt:
                m = op_map("DSA.idx_score", t, phase)
                dcp_dec = t.dsa == "dcp" and phase == "decode"
                rows_g = T if dcp_dec else T / m["row"]
                keys_row = env["Lsum"] / T / (N if dcp_dec else 1)
                mode = {"dcp": "DCP%d key shard + exact merge" % N if phase == "decode" else "ShardQ%d own rows vs gathered ctx" % N,
                        "dsadp": "DSA-DP owner (all rows of owned requests)", "local": "local (DEP)",
                        "dsacp": "DSA-CP: local rows vs ctx gathered from %d GPUs" % (t.G or 1),
                        "rep": "replicated DSA (TP%d, rows / %s per GPU)" % (N, ri(m["row"]))}[t.dsa]
                shape = dict(rows=ri(rows_g), mean_keys_per_row=ri(keys_row), topk=C["TOPK"], heads=C["Ni"],
                             head_dim=C["Di"], key_B=C["IDX_ROW"], mode=mode)
                if phase == "prefill":
                    shape.update(requests_in_chunk=ri(env["nseq"]), prefix=ri(env["L"] - Pre.F_REQ), fresh=ri(Pre.F_REQ))
                add("dsa_indexer_topk", shape, "fp4 MMA + fp32 epilogue", use(cnt, ta, extra=dict(t_ach_kernel_us=tk * 1e6)),
                    [i for i in IDX_OPS if i in rows], ts, ta, tk, 0,
                    notes=["DeepGEMM fp4_mqa_logits + exact top-2048 (BF16 scratch select)"])
            # ---- sparse MLA (+ KV insert prologue, gate_mul fused)
            ts, ta, tk, cnt = rowsum(MLA_OPS)
            if cnt:
                m = op_map("DSA.sparse_mla", t, phase)
                dcp_dec = t.dsa == "dcp" and phase == "decode"
                if t.dsa == "rep":
                    rows_g, heads = T, C["Nh"] / N
                elif dcp_dec:
                    rows_g, heads = T, C["Nh"]
                else:
                    rows_g, heads = T / m["row"], C["Nh"]
                keys = env["Ltk"] / T / (N if dcp_dec else 1)
                shape = dict(rows=ri(rows_g), heads=ri(heads), keys_per_row=ri(keys), kv_row_B=C["MLA_ROW"],
                             qk_dim=C["KVL"] + C["DR"], v_dim=C["KVL"],
                             mode="DCP%d local-key partial (all heads, LSE combine)" % N if dcp_dec else
                             ("ShardQ%d rows" % N if (t.dsa == "dcp") else t.dsa))
                add("sparse_mla", shape, "fp8 Q x fp8 KV", use(cnt, ta, extra=dict(t_ach_kernel_us=tk * 1e6)),
                    [i for i in MLA_OPS if i in rows], ts, ta, tk, 0,
                    notes=["FlashMLA sparse / CuTe sparse MLA FP8 query path; prologue = KV+indexer-key insert"])
            # ---- MegaMoE
            ts, ta, tk, cnt = rowsum(MOE_OPS)
            if cnt:
                cfg = Mdl.ckpt_cfg(ck)
                act = "mxfp8" if cfg["exp_aB"] > 1 else "nvfp4"
                if t.experts == "tp":
                    shape = dict(mode="TP%d experts (1/%d of every expert)" % (N, N), tokens_per_rank=ri(T),
                                 routes_per_rank=ri(T * C["TOPK_E"]), inter_per_rank=ri(C["I_E"] / N),
                                 experts_touched_per_gpu=ri(cost.expert_env(env, t, phase) * N), act=act)
                else:
                    tok = T / N
                    shape = dict(mode="EP%d MegaMoE" % t.E, EP=t.E, tokens_per_rank=ri(tok),
                                 routes_per_rank=ri(tok * C["TOPK_E"]), expert_slots_per_gpu=cost.experts_per_gpu(t.E),
                                 experts_touched_per_gpu=round(cost.expert_env(env, t, phase), 2), act=act,
                                 dedup=bool(t.dedup))
                notes = ["time dispatch / fc1 / fc2 / combine separately; P11/P12 NVLink cost is in the comm family"]
                # The fused MegaMoE kernel moves tokens over NVLink inside the kernel: its whole-kernel target is
                # t_ach(compute ops) + t_ach(P11) + t_ach(P12) of the same step (t_ach_fused_us). TP experts have none.
                p11 = p12 = 0.0
                for e in comm_events(t, phase, env, Mdl.ckpt_cfg(ck)).get("moe", []):
                    if e.get("prim") == "P11 EP-dispatch":
                        p11 = cost.comm_time(e, HWS["achievable"])[0]
                    elif e.get("prim") == "P12 EP-combine":
                        p12 = cost.comm_time(e, HWS["achievable"])[0]
                fused = dict(t_ach_kernel_us=tk * 1e6, p11_t_ach_us=round(p11 * 1e6, 3), p12_t_ach_us=round(p12 * 1e6, 3),
                             t_ach_fused_us=round((tk + p11 + p12) * 1e6, 3))
                if ck == "mxfp4_release" and phase == "prefill" and T / N > 512:
                    notes.append("PIN MXFP8 activations (Rubin MegaMoE switches to NVFP4 >512 tok/rank) or compare O4")
                add("megamoe", shape, rows["MOE.fc1"]["dtype"], use(cnt, ta, extra=fused),
                    [i for i in MOE_OPS if i in rows], ts, ta, tk, 0, notes=notes)
            # ---- AttnRes / embed
            for oid in ATTNRES_OPS:
                if oid not in rows or rows[oid]["t_ach_s"] <= 0:
                    continue
                x = rows[oid]
                m = op_map(oid, t, phase)
                if oid == "F.attnres_final":
                    Mf = env["R"] if phase == "prefill" else T
                elif oid == "DS.aux_capture":
                    Mf = env["Tc"] if phase == "prefill" else T
                else:
                    Mf = T
                shape = dict(op=oid, rows=ri(Mf / m["row"]), hidden=C["H"],
                             sources="1..8 BF16 blocks (cdiv(l,12)), mean over 93 layers" if oid.startswith("L.") else
                             ("8 blocks" if oid == "F.attnres_final" else "-"))
                add("attnres", shape, x["dtype"], use(x["count"], x["t_ach_s"], fused=Mdl.OPS_BY_ID[oid].fused), [oid],
                    x["t_sol_s"], x["t_ach_s"], kern(x["t_ach_s"], Mdl.OPS_BY_ID[oid].fused), 0)
            # ---- drafter non-GEMM + verify
            for oid in DRAFT_OPS:
                if oid not in rows or rows[oid]["t_ach_s"] <= 0:
                    continue
                x = rows[oid]
                ND = t.ND
                if oid == "DS.attn":
                    shape = dict(op=oid, rows=ri(env["Tq"]), B=ri(env["B"]), k=int(env["k"]), q_heads=ri(C["D_HQ"] / ND),
                                 kv_heads=ri(C["D_HKV"] / ND), head_dim=C["D_HD"], window=C["D_W"])
                elif oid == "DS.markov":
                    shape = dict(op=oid, B=ri(env["B"]), positions=int(env["k"]), rank=C["D_R"], vocab_shard=ri(C["V"] / ND))
                elif oid == "F.rejection":
                    shape = dict(op=oid, B=ri(env["B"]), q=int(env["q"]), vocab_shard=ri(C["V"] / (N if N > 1 else 1)))
                else:
                    m = op_map(oid, t, phase)
                    sym = {"DS.embed": "Tq", "DS.norm_ar": "Tq", "DS.confidence": "Tq", "DS.hidden_norm": "Tc",
                           "DS.ctx_kv_insert": "Tc", "DS.silu": "Tq", "F.norm": "R"}[oid]
                    shape = dict(op=oid, rows=ri(env[sym] / m["row"]))
                add("drafter_verify", shape, x["dtype"], use(x["count"], x["t_ach_s"], fused=Mdl.OPS_BY_ID[oid].fused),
                    [oid], x["t_sol_s"], x["t_ach_s"], kern(x["t_ach_s"], Mdl.OPS_BY_ID[oid].fused), 0,
                    notes=["DSpark draft step bench (sdec)"])
            # ---- remaining fused epilogues (bench only inside their GEMM; listed for completeness)
            for oid in EPI_OPS:
                if oid not in rows or rows[oid]["t_ach_s"] <= 0:
                    continue
                x = rows[oid]
                m = op_map(oid, t, phase)
                shape = dict(op=oid, rows=ri(T / m["row"]), width={"MOE.topk": C["E"], "MOE.latent_norm": C["LAT"],
                             "MOE.shared_act": ri(2 * C["I_SH"] / (m["w"] if m["w"] > 1 else 1)),
                             "DENSE.act": ri(2 * C["I_D"] / (m["w"] if m["w"] > 1 else 1))}[oid])
                add("fused_epilogue", shape, x["dtype"], use(x["count"], x["t_ach_s"], fused=True), [oid],
                    x["t_sol_s"], x["t_ach_s"], x["t_ach_s"], 0, notes=["fused in SOL: verify inside the producer GEMM bench"])
            # ---- comm (rebuilt with N; times checked against the optables comm rows)
            cfg = Mdl.ckpt_cfg(ck)
            evs = comm_events(t, phase, env, cfg)
            n_of = {"moe": 92, "kda": 69, "dsa": 24}
            ref = [dict(zip(OPT["comm_cols"], c)) for c in tb["comm"]]
            ri_ = 0
            for lt, lst in evs.items():
                for e in lst:
                    tsol, asol = cost.comm_time(e, HWS["sol"])
                    tach, aach = cost.comm_time(e, HWS["achievable"])
                    rr = ref[ri_]
                    ri_ += 1
                    assert abs(rr["t_ach_s"] - tach) <= 1e-3 * tach + 1e-12, (key, rr, tach)
                    cntl = n_of.get(lt, 1)
                    if "alts" in e:
                        nv = e["alts"][0]
                        prim = "AR" if nv["kind"] == "nvls_ar" else "RS"
                        shape = dict(primitive=prim, N=nv["N"], msg_B=ri(nv["M"]),
                                     candidates=[a["prim"] for a in e["alts"]], chosen_sol=asol["prim"],
                                     chosen_ach=aach["prim"])
                    else:
                        shape = dict(primitive=e["prim"], N=e["N"], in_B=ri(e["in_B"]), out_B=ri(e["out_B"]),
                                     hops=e["hops"], async_ce=bool(e.get("async_")))
                    add("comm", shape, "-", use(cntl, tach, fused=True, extra=dict(layer=lt, event=e.get("name") or nv["name"])),
                        [lt + ":" + (e.get("name") or nv["name"])], tsol, tach, tach, 0,
                        notes=["symmbench/ncclbench" + (" (async, hidden under attention)" if e.get("async_") else "")])
            assert ri_ == len(ref), (key, ri_, len(ref))
    # FLOP cross-check of the GEMM shapes
    bad = [(k, o, a, b) for k, o, a, b in checks if abs(a - b) > 2e-3 * b + 1]
    assert not bad, bad[:5]
    return len(checks)


def finalize():
    out = []
    for c in cases.values():
        # merge uses of the same step (one regime may hit a shape from several ops / layer types)
        g = {}
        for u in c["uses"]:
            k = (u["ckpt"], u["topo"], u["phase"], u["r"])
            if k not in g:
                g[k] = dict(u, ops=list(u["ops"]))
            else:
                v = g[k]
                v["count"] += u["count"]
                v["share"] += u["share"]
                v["score"] += u["score"]
                v["ops"] = v["ops"] + [o for o in u["ops"] if o not in v["ops"]]
        us = list(g.values())
        if c["family"] == "megamoe":
            fz = sorted(set((u["p11_t_ach_us"], u["p12_t_ach_us"], u["t_ach_fused_us"]) for u in us))
            c["p11_t_ach_us"], c["p12_t_ach_us"], c["t_ach_fused_us"] = fz[0]
            c["t_ach_fused_us_range"] = [min(x[2] for x in fz), max(x[2] for x in fz)]
        for u in us:
            u.pop("event", None)
            u.pop("layer", None)
            u.pop("t_ach_kernel_us", None)
        c["uses"] = us
        c["score"] = sum(u["score"] for u in us)
        c["share_max"] = max(u["share"] for u in us)
        c["share_sum"] = sum(u["share"] for u in us)
        c["topologies"] = sorted(set("%s|%s%s" % (u["ckpt"][:5], u["topo"], "" if u["r"] is None else " r%d" % u["r"]) for u in us))
        c["topo_ids"] = sorted(set(u["topo"] for u in us))
        c["topo_ids_in_scope"] = sorted(set(u["topo"] for u in us if u["scope"] != "oos"))
        c["scope"] = "priority" if any(u["scope"] == "priority" for u in us) else (
            "cover" if any(u["scope"] == "cover" for u in us) else "oos")
        c["ckpts"] = sorted(set(u["ckpt"] for u in us))
        c["count_per_step"] = max(u["count"] for u in us)
        c["gap_prior_max"] = max(u["gap_prior"] for u in us)
        c["t_ach_over_sol"] = round(c["t_ach_us"] / c["t_sol_us"], 3) if c["t_sol_us"] > 0 else None
        # absolute achievable budget of this case inside one step (count x max(t, kernel gap)), us
        c["step_us_ach"] = c["count_per_step"] * c["t_ach_kernel_us"]
        for k in ("t_sol_us", "t_ach_us", "t_ach_kernel_us", "step_us_ach"):
            c[k] = round(c[k], 3)
        c["t_ach_us_range"] = [round(x, 3) for x in c["t_ach_us_range"]]
        out.append(c)
    # tiers per phase (prefill and decode ranked separately; a case takes its better tier)
    for c in out:
        c["score_prefill"] = sum(u["score"] for u in c["uses"] if u["phase"] == "prefill")
        c["score_decode"] = sum(u["score"] for u in c["uses"] if u["phase"] == "decode")
    tier = {}
    for ph in ("prefill", "decode"):
        key = "score_" + ph
        lst = sorted([c for c in out if c[key] > 0], key=lambda c: -c[key])
        tot = sum(c[key] for c in lst)
        acc = 0.0
        for c in lst:
            p = "P0" if acc < 0.6 * tot else ("P1" if acc < 0.9 * tot else "P2")
            acc += c[key]
            c[key + "_frac"] = c[key] / tot
            tier[id(c)] = min(tier.get(id(c), "P2"), p)
    out.sort(key=lambda c: -max(c.get("score_prefill_frac", 0.0), c.get("score_decode_frac", 0.0)))
    for i, c in enumerate(out):
        c["rank"] = i + 1
        c["priority"] = "OOS" if c["scope"] == "oos" else tier.get(id(c), "P2")
        for k in ("score_prefill", "score_decode", "score_prefill_frac", "score_decode_frac"):
            if k in c:
                c[k] = round(c[k], 5)
        c["score"] = round(c["score"], 5)
        c["share_max"] = round(c["share_max"], 5)
        c["share_sum"] = round(c["share_sum"], 5)
        c["uses"] = [dict(u, share=round(u["share"], 5), score=round(u["score"], 6)) for u in c["uses"]]
    # stable ids (ids_v1.json); new keys numbered after the highest existing number, in rank order
    reg = json.load(open(os.path.join(HERE, "ids_v1.json")))["ids"]
    nxt = max(int(v.rsplit("-", 1)[1]) for v in reg.values()) + 1
    for c in out:
        k = json.dumps([c["family"], c["shape"], c["dtype"]], sort_keys=True)
        if k in reg:
            c["id"] = reg[k]
            c["id_new"] = False
        else:
            c["id"] = "%s-%03d" % (c["family"], nxt)
            c["id_new"] = True
            nxt += 1
    missing = set(reg.values()) - set(c["id"] for c in out)
    assert not missing, ("v1 ids lost", sorted(missing)[:10])
    return out


def _brief(c):
    sh = {k: v for k, v in c["shape"].items() if k not in ("candidates", "chosen_sol", "qk_dim", "v_dim", "kv_row_B",
                                                          "head_dim", "key_B", "topk", "sources", "hidden")}
    return dict(id=c["id"], rank=c["rank"], shape=sh, dtype=c["dtype"], t_sol_us=c["t_sol_us"], t_ach_us=c["t_ach_us"],
                count=c["count_per_step"], share_max=c["share_max"], topologies=c["topo_ids"], ops=c["ops"][:4])


def summarize(rows):
    out = {}
    for f in ("megamoe", "gemm", "comm", "kda", "dsa_indexer_topk", "sparse_mla", "attnres", "drafter_verify",
              "fused_epilogue"):
        fc = [c for c in rows if c["family"] == f]
        out[f] = dict(cases=len(fc), P0=[_brief(c) for c in fc if c["priority"] == "P0"],
                      n_P1=sum(1 for c in fc if c["priority"] == "P1"), n_P2=sum(1 for c in fc if c["priority"] == "P2"),
                      n_OOS=sum(1 for c in fc if c["priority"] == "OOS"))
    return out


def coverage(rows):
    """Per regime: sum of case shares (compute ops at max(t, gap) + all comm incl. async) vs the step; ~1 means the
    matrix accounts for the whole step (comm overlap/async makes it exceed 1 slightly)."""
    cov = {}
    for c in rows:
        for u in c["uses"]:
            k = "%s|%s%s" % (u["ckpt"], u["topo"], "" if u["r"] is None else " r%d" % u["r"])
            v = cov.setdefault(k, dict(share=0.0, cases=0, p0_share=0.0))
            v["share"] += u["share"]
            v["cases"] += 1
            if c["priority"] == "P0":
                v["p0_share"] += u["share"]
    return {k: dict(share=round(v["share"], 3), cases=v["cases"], p0_share=round(v["p0_share"], 3))
            for k, v in sorted(cov.items())}


if __name__ == "__main__":
    n = run()
    rows = finalize()
    fams = {}
    for c in rows:
        f = fams.setdefault(c["family"], dict(cases=0, P0=0, P1=0, P2=0, OOS=0, score=0.0))
        f["cases"] += 1
        f[c["priority"]] += 1
        f["score"] += c["score"]
    doc = dict(
        generated_from="out/optables.json + inputs/ops.json + topologies.op_map/comm_events (build_matrix.py)",
        scope=dict(prefill=PREFILL_T, decode=DECODE_T, cover=COVER_T, out_of_scope=OOS_T, checkpoints=CKS,
                   rule="topologies.SCOPE (2026-11-18): EP <= 16 and <= 16 GPUs per engine / EP domain; priority = "
                        "deployed + EP<=16 frontier; cover = rest of SCOPE (weight 0); out_of_scope = EP64/EP32 rows "
                        "kept for id continuity (priority OOS, never benched)",
                   regimes="prefill F=16384 L=78.4K (5,456 fresh/request); decode r in {1,8,32} k=3 L=135K "
                           "(r = requests per GPU, B_group = r*N, T_group = B*(1+k)); r above r_max dropped"),
        megamoe_target="megamoe cases: t_ach_kernel_us = on-GPU compute ops only (dispatch/fc1/act/fc2/combine "
                       "SIMT+MMA); t_ach_fused_us = that + P11 + P12 achievable NVLink time of the same step. A "
                       "whole-kernel MegaMoE measurement is divided by t_ach_fused_us; the P11/P12 comm cases of an "
                       "EP topology are then NOT benched separately for the derating (their d = the MegaMoE d).",
        id_rule="stable ids from ids_v1.json; id_new marks cases added after v1",
        priority_rule="use score = share_of_step_ach x (gap_prior - 1) / (regimes of that topology+ckpt), so every "
                      "(topology, ckpt) weighs the same; share = count x max(t_ach, 0.78us kernel gap) / step_ach "
                      "(comm: t_ach x layers, async included). Tiers per phase: P0 = cases covering the first 60% of "
                      "that phase's total score, P1 to 90%, P2 the rest; a case takes its better tier. rank = by the "
                      "larger of its phase score fractions. gap_prior = measured / model-achievable step of that "
                      "topology (proxies marked).",
        gap_priors={"%s%s" % (k[0], "" if k[1] is None else " r%d" % k[1]): dict(gap=v[0], src=v[1]) for k, v in GAP.items()},
        gap_prior_default_decode=dict(gap=DEFAULT_DEC_GAP[0], src=DEFAULT_DEC_GAP[1]),
        gemm_flop_checks=n, families=fams, summary=summarize(rows), n_cases=len(rows), coverage=coverage(rows), cases=rows)
    with open(os.path.join(HERE, "matrix.json"), "w") as f:
        json.dump(doc, f, indent=1)
    print("cases", len(rows), "new", sum(1 for c in rows if c["id_new"]), "gemm checks", n)
    for k, v in sorted(fams.items(), key=lambda x: -x[1]["score"]):
        print("%-18s cases %4d  P0 %3d P1 %3d P2 %3d OOS %3d  score %.3f" % (k, v["cases"], v["P0"], v["P1"], v["P2"],
                                                                         v["OOS"], v["score"]))
