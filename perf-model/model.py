"""K3.1-Pro + DSpark op graph (from inputs/ops.json) and the two serving checkpoints.

Every op in ops.json carries python formulas for FLOPs and bytes of the FULL (unsharded) op as a function of the
step symbols (T, B, k, q, L, Lsum, Ltk, Lu, R, AL, Tc, Tq, nseq, switches PREFILL/CKPT/SCRATCH, E_ACT).
This module:
  * compiles those formulas once;
  * builds the symbol environment for a prefill chunk or a decode step (env_prefill / env_decode);
  * applies the checkpoint (weight bytes per element and GEMM dtype per op family);
  * evaluates one op -> dict(flops, w, a_in, a_out, kv_r, kv_w, st_r, st_w) for one instance (one layer).
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "inputs", "ops.json")) as _f:
    OPSJ = json.load(_f)
with open(os.path.join(HERE, "inputs", "workload.json")) as _f:
    WORK = json.load(_f)
with open(os.path.join(HERE, "inputs", "checkpoints.json")) as _f:
    CKJ = json.load(_f)

C = dict(OPSJ["constants"])
# FP8 MLA KV in the serving rules is the standard per-tensor FP8 row (512 latent + 64 rope = 576 B/token/layer),
# not the fp8_ds_mla 656-B row that ops.json uses as its default (workload.json kv_state_bytes, release NIXL logs).
C["MLA_ROW"] = 576
C["IDX_ROW"] = 68
DSA_KV_PER_TOKEN = C["N_DSA"] * (C["MLA_ROW"] + C["IDX_ROW"])   # 15,456 B

FIELDS = ("flops", "weight_bytes", "act_in_bytes", "act_out_bytes", "kv_read_bytes", "kv_write_bytes",
          "state_read_bytes", "state_write_bytes", "simt_flops", "sfu_ops")
SHORT = dict(flops="flops", weight_bytes="w", act_in_bytes="a_in", act_out_bytes="a_out", kv_read_bytes="kv_r",
             kv_write_bytes="kv_w", state_read_bytes="st_r", state_write_bytes="st_w", simt_flops="simt",
             sfu_ops="sfu")

_SAFE = {"__builtins__": {}, "min": min, "max": max}


class Op:
    def __init__(self, j):
        self.j = j
        self.id = j["id"]
        self.name = j["name"]
        self.layer_type = j["layer_type"]
        self.kind = j["kind"]
        self.phase = j.get("phase", "both")
        self.fused = bool(j.get("fused_in_sol"))
        self.compute = (j.get("dtype") or {}).get("compute", "none")
        self.shapes = j.get("shapes") or ""
        self.cite = j.get("cite") or []
        self.count_expr = compile(str(j.get("count_per_forward") or "1"), self.id + ".count", "eval")
        self.expr = {}
        for k in FIELDS:
            s = j.get(k)
            self.expr[k] = compile(str(s if s not in (None, "") else "0"), self.id + "." + k, "eval")

    def in_phase(self, phase):
        return self.phase == "both" or self.phase == phase

    def base_dtype(self):
        c = self.compute
        if "mxf4" in c:
            return "fp4"
        if "mxf8" in c or c.startswith("fp8"):
            return "fp8"
        if "bf16" in c:
            return "bf16"
        return "simt"


OPS = [Op(j) for j in OPSJ["ops"]]
OPS_BY_ID = {o.id: o for o in OPS}

# ---------------------------------------------------------------------------------------------- families
FAMILY = {}
for _o in OPS:
    i = _o.id
    if i in ("G.embed", "G.attnres_init"):
        f = "embed"
    elif i.startswith("L.attnres") or i == "F.attnres_final":
        f = "attnres"
    elif i in ("L.sp_gather", "F.sp_gather_out"):
        f = "comm_op"
    elif i.startswith("DENSE."):
        f = "dense_mlp"
    elif i in ("KDA.in_proj", "KDA.f_b_proj", "KDA.o_proj"):
        f = "kda_proj"
    elif i.startswith("KDA."):
        f = "kda_core"
    elif i in ("DSA.idx_q_quant", "DSA.idx_score", "DSA.topk"):
        f = "indexer"
    elif i in ("DSA.prologue", "DSA.sparse_mla"):
        f = "dsa_attn"
    elif i.startswith("DSA."):
        f = "mla_proj"
    elif i in ("MOE.router", "MOE.topk"):
        f = "router"
    elif i in ("MOE.latent_down", "MOE.latent_norm", "MOE.latent_up"):
        f = "latent"
    elif i.startswith("MOE.shared"):
        f = "shared"
    elif i.startswith("MOE."):
        f = "experts"
    elif i.startswith("F."):
        f = "lm_head"
    elif i.startswith("DS."):
        f = "drafter"
    else:
        f = "other"
    FAMILY[i] = f

FAMILY_ORDER = ["embed", "attnres", "dense_mlp", "kda_proj", "kda_core", "mla_proj", "indexer", "dsa_attn", "router",
                "latent", "shared", "experts", "lm_head", "drafter", "launch"]
FAMILY_LABEL = {
    "embed": "embedding", "attnres": "AttnRes mixes", "dense_mlp": "dense MLP (L0)", "kda_proj": "KDA projections",
    "kda_core": "KDA core (conv+recurrence)", "mla_proj": "MLA projections", "indexer": "indexer (score+top-k)",
    "dsa_attn": "DSA attention (KV insert+sparse MLA)", "router": "MoE router", "latent": "MoE latent down/up",
    "shared": "shared experts", "experts": "routed experts", "lm_head": "lm_head / verify", "drafter": "DSpark drafter",
    "launch": "kernel launch gaps", "comm": "communication",
}
# Coarse stack groups for charts (<= 8 so the categorical palette stays legible).
GROUP_OF = {
    "kda_proj": "KDA", "kda_core": "KDA", "mla_proj": "MLA/DSA", "dsa_attn": "MLA/DSA", "indexer": "indexer",
    "router": "MoE dense", "latent": "MoE dense", "shared": "MoE dense", "dense_mlp": "MoE dense",
    "experts": "routed experts", "embed": "other", "attnres": "other", "lm_head": "other", "launch": "other",
    "drafter": "drafter", "comm": "comm",
}
GROUP_ORDER = ["KDA", "MLA/DSA", "indexer", "MoE dense", "routed experts", "drafter", "other", "comm"]

# ---------------------------------------------------------------------------------------------- checkpoints
MXFP4_B = 0.5 + 1.0 / 32     # E2M1 + UE8M0 scale per 32
NVFP4_B = 0.5 + 1.0 / 16     # E2M1 + E4M3 scale per 16
MXFP8_B = 1.0 + 1.0 / 32     # E4M3 + UE8M0 per 32

CHECKPOINTS = {
    "mxfp4_release": dict(
        label="mxfp4_release (moonshotai d9e99b07)",
        exp_wB=MXFP4_B, exp_aB=MXFP8_B, exp_dt="fp8",         # W4A8: mxf8f6f4 MMA at the FP8 rate
        kda_dt="bf16", kda_wB=2.0, sh_dt="bf16", sh_wB=2.0,
        dispatch_row=round(C["LAT"] * MXFP8_B),                 # 3696 B per route
    ),
    "nvfp4_modelopt": dict(
        label="nvfp4_modelopt (nvidia NVFP4)",
        exp_wB=NVFP4_B, exp_aB=NVFP4_B, exp_dt="fp4",          # W4A4 NVFP4
        kda_dt="fp8", kda_wB=MXFP8_B, sh_dt="fp8", sh_wB=MXFP8_B,  # KDA projections + shared experts MXFP8 W8A8
        dispatch_row=round(C["LAT"] * NVFP4_B),                 # 2016 B per route
    ),
}
CKPT_ORDER = ["mxfp4_release", "nvfp4_modelopt"]
KDA_Q_OPS = ("KDA.in_proj", "KDA.o_proj")          # f_b_proj stays BF16 in the NVFP4 checkpoint
SHARED_OPS = ("MOE.shared_gate_up", "MOE.shared_down")
EXPERT_GEMMS = ("MOE.fc1", "MOE.fc2")


def ckpt_cfg(name, moe_act=None):
    c = dict(CHECKPOINTS[name])
    if moe_act == "nvfp4" and name == "mxfp4_release":   # overlay O4 (open question): NVFP4 activations
        c.update(exp_aB=NVFP4_B, exp_dt="fp4", dispatch_row=round(C["LAT"] * NVFP4_B))
    return c


def sparse_mla_dtype(mla_row):
    """Sparse-MLA MMA dtype implied by the FP8 KV format: the 576-B per-tensor FP8 row (SM10x default) runs FP8 Q x
    FP8 KV (flashinfer_mla_sparse.py: FP8 query required with an FP8 cache); the 656-B fp8_ds_mla row is
    dequantized to BF16 on load."""
    if mla_row == 576:
        return "fp8"
    if mla_row == 656:
        return "bf16"
    raise ValueError("unknown MLA KV row %r" % mla_row)


def op_dtype(op, ck):
    if op.id == "DSA.sparse_mla":
        return sparse_mla_dtype(C["MLA_ROW"])
    if op.id in KDA_Q_OPS:
        return ck["kda_dt"]
    if op.id in SHARED_OPS:
        return ck["sh_dt"]
    if op.id in EXPERT_GEMMS:
        return ck["exp_dt"]
    return op.base_dtype()


# ---------------------------------------------------------------------------------------------- workload
POOLED = WORK["pooled"]
AGENTX = dict(
    fresh_per_req=POOLED["fresh_prefill_tokens"]["mean"],            # 5,456
    prefix_fresh_weighted=72900.0,                                  # workload.md: cached prefix weighted by fresh
    isl_mean=POOLED["isl"]["mean"],
    osl_mean=POOLED["osl"]["mean"],
    decode_ctx=135000.0,                                            # mission reference decode context
    decode_ctx_osl_weighted=WORK["attention_work_moments"]["decode_ctx_per_output_token_mean"],
    think_s=WORK["think_time"]["closed_loop_idle_per_request_s"]["mean_request_weighted_C>=16"],
)
AL_GOLDEN = {3: 3.08, 5: 3.81, 7: 4.16}


def env_prefill(F, L, f_req=None, o7=True, ckpt_export=1):
    """Prefill chunk of F fresh tokens on one engine (TP group), packed from requests with f_req fresh tokens each
    over a prefix P = L - f_req (L = context of each request after the chunk)."""
    f = float(min(F, f_req or F))
    n = F / f                       # requests (sequence pieces) in the chunk
    P = max(L - f, 0.0)
    e = dict(C)
    e.update(T=float(F), B=n, nseq=n, k=0, q=1, L=float(L), AL=1.0,
             Lsum=n * (f * P + f * (f + 1) / 2.0),
             Ltk=n * sum_min_topk(P, f),
             Lu=n * (P + f), R=n,
             Tc=n * min(f, C["D_W"]) if o7 else float(F), Tq=0.0,
             PREFILL=True, CKPT=ckpt_export, SCRATCH=0 if o7 else 1, O7=1 if o7 else 0, MOE_SPILL=0,
             E_ACT=C["E"])
    return e


def sum_min_topk(P, f):
    """sum_{i=1..f} min(P + i, TOPK) (keys read by sparse MLA rows of one request piece)."""
    K = C["TOPK"]
    if P >= K:
        return f * K
    m = min(f, K - P)             # rows still below TOPK
    return m * P + m * (m + 1) / 2.0 + (f - m) * K


def env_decode(B, k, L, AL=None, o7=True):
    q = 1 + k
    al = AL if AL is not None else AL_GOLDEN.get(k, 1 + 0.6 * k)
    e = dict(C)
    e.update(T=float(B * q), B=float(B), nseq=float(B), k=k, q=q, L=float(L), AL=al,
             Lsum=B * q * L, Ltk=B * q * min(L, C["TOPK"]), Lu=B * L, R=float(B * q),
             Tc=B * al if o7 else float(B * q), Tq=float(B * k),
             PREFILL=False, CKPT=0, SCRATCH=0 if o7 else 1, O7=1 if o7 else 0, MOE_SPILL=0, E_ACT=C["E"])
    return e


def evaluate(op, env, ck):
    """Full (unsharded) per-instance numbers of one op, with the checkpoint applied."""
    r = {}
    for k in FIELDS:
        r[SHORT[k]] = float(eval(op.expr[k], _SAFE, env))
    if op.id in KDA_Q_OPS:
        r["w"] *= ck["kda_wB"] / 2.0
    elif op.id in SHARED_OPS:
        r["w"] *= ck["sh_wB"] / 2.0
    elif op.id in EXPERT_GEMMS:
        r["w"] *= ck["exp_wB"] / MXFP4_B
        if op.id == "MOE.fc1":
            r["a_in"] *= ck["exp_aB"] / MXFP8_B
            r["a_out"] *= ck["exp_aB"] / MXFP8_B      # spilled intermediate (MOE_SPILL), act dtype
        else:
            r["a_in"] *= ck["exp_aB"] / MXFP8_B
    elif op.id == "MOE.dispatch":
        r["a_out"] *= ck["exp_aB"] / MXFP8_B
    if op.fused:                      # SOL view: fused into the neighbour, activations never hit HBM
        r["a_in"] = 0.0
        r["a_out"] = 0.0
    return r


def count(op, env):
    return float(eval(op.count_expr, _SAFE, env))


def expert_bytes(ck):
    """Bytes of one routed expert (fc1 + fc2) in the checkpoint."""
    return C["LAT"] * 3 * C["I_E"] * ck["exp_wB"]


def touched_frac(rows):
    """Probability one expert is touched when `rows` tokens each pick TOPK_E of E experts (uniform routing)."""
    if rows <= 0:
        return 0.0
    return 1.0 - (1.0 - C["TOPK_E"] / C["E"]) ** rows


def per_request_state_bytes(k, L, conv_fp32=True):
    """SOL per-request KV/state, whole model, before sharding (workload.json kv_state_bytes, minimal state)."""
    q = 1 + k
    dsa = DSA_KV_PER_TOKEN * L
    kda = C["N_KDA"] * C["Hk"] * C["dk"] * C["dv"] * 4                       # 434.1 MB FP32 recurrent
    conv = C["N_KDA"] * 3 * C["P"] * (C["CONV"] - 1 + k) * (4 if conv_fp32 else 2)
    recover = C["N_KDA"] * C["Hk"] * C["REC_B"] * q                          # RecoverSSM records (6.78 MB x q)
    swa = 20480 * min(L, C["D_W"])                                            # drafter SWA BF16
    return dict(dsa=dsa, kda=kda, conv=conv, recover=recover, swa=swa)


def model_constants_table():
    """Rows for the HTML model table."""
    c = C
    rows = [
        ("layers", "93 = 69 KDA + 24 DSA (0-based DSA idx 3,7,...,91,92); layer 0 dense MLP, 1..92 MoE", "ops.json layer_counts"),
        ("hidden / vocab", "%d / %d" % (c["H"], c["V"]), "config text_config"),
        ("KDA", "96 heads x 128, FP32 state 96x128x128/layer (434 MB/request), conv 4, chunk 16", "kda.py, kda_segscan.py"),
        ("MLA (DSA layers)", "96 heads, q_lora 1536, kv_lora 512, nope 128 + rope 64, v 128, output gate", "dsa.py, mla.py"),
        ("indexer", "32 heads x 128, MXFP4 keys 68 B/token/layer, exact top-2048", "dsa.py indexer"),
        ("DSA KV per token", "%d B (24 x (576 FP8 MLA + 68 MXFP4 indexer))" % DSA_KV_PER_TOKEN, "workload.json kv_state_bytes"),
        ("MoE", "896 experts top-16, latent 3584, expert inter 3072, 2 shared (6144), sigmoid router", "latent_moe_runner.py"),
        ("AttnRes", "block 12, up to 8 block sources per mix, 2 mixes per layer + final", "model.py attn_res"),
        ("DSpark draft", "5 GQA layers 96q/16kv x 64, SWA 2048, MLP 14336, fc 5x7168->7168, Markov rank 512", "dspark_gqa.py, draft config e4e6e626"),
        ("golden AL", "k3 3.08 / k5 3.81 / k7 4.16", "mission"),
        ("AgentX fresh tokens / request", "%.0f mean (prefix 96.2%% hit); prefix weighted by fresh 72.9K" % AGENTX["fresh_per_req"], "workload.json pooled"),
        ("AgentX ISL / OSL / think", "%.0f / %.0f / %.1f s" % (AGENTX["isl_mean"], AGENTX["osl_mean"], AGENTX["think_s"]), "workload.json"),
    ]
    return [dict(name=a, value=b, source=s) for a, b, s in rows]


def checkpoint_table():
    W = WORK["weights"]["by_category_bytes"]
    kda_proj_params = (C["H"] * (4 * C["P"] + C["dk"] + C["Hk"]) + C["P"] * C["H"]) * C["N_KDA"]
    kda_nv = W["kda_attn_bf16_f32"] - kda_proj_params * 2 + kda_proj_params * MXFP8_B
    sh_nv = W["moe_shared_experts_bf16"] / 2 * MXFP8_B
    exp_nv = W["moe_routed_experts_mxfp4"] / MXFP4_B * NVFP4_B
    rows = [
        ("routed experts", "MXFP4 W / MXFP8 act (FP8 MMA rate)", W["moe_routed_experts_mxfp4"], "NVFP4 W4A4 (FP4 rate)", exp_nv),
        ("KDA projections (+conv/f_b)", "BF16", W["kda_attn_bf16_f32"], "MXFP8 W8A8 (f_b, conv BF16/FP32)", kda_nv),
        ("shared experts", "BF16", W["moe_shared_experts_bf16"], "MXFP8 W8A8", sh_nv),
        ("MLA + indexer projections", "BF16", W["dsa_mla_attn_bf16"] + W["dsa_indexer_proj_bf16"], "BF16", W["dsa_mla_attn_bf16"] + W["dsa_indexer_proj_bf16"]),
        ("router + latent up/down", "BF16", W["moe_router_bf16"] + W["moe_latent_up_down_norm_bf16"], "BF16", W["moe_router_bf16"] + W["moe_latent_up_down_norm_bf16"]),
        ("dense MLP (layer 0)", "BF16", W["layer0_dense_mlp_bf16"], "BF16", W["layer0_dense_mlp_bf16"]),
        ("embedding + lm_head", "BF16", W["embed_tokens_bf16"] + W["lm_head_bf16"], "BF16", W["embed_tokens_bf16"] + W["lm_head_bf16"]),
        ("DSpark draft", "BF16", 4.96e9, "BF16", 4.96e9),
    ]
    return [dict(family=a, mx_dtype=b, mx_GB=c / 1e9, nv_dtype=d, nv_GB=e / 1e9) for a, b, c, d, e in rows]
