#!/usr/bin/env python3
"""Generate the K3.1-Pro (+ DSpark draft) op inventory: ops.json + ops.md.

All formulas are python expressions over the symbols in SYMBOLS and the model
constants in CONST. Byte/flop values are for the FULL (unsharded) op; the topology
model divides them (sharding axis given per op). Reference evaluations at the end.
"""
import json, math, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
R = "vllm/models/kimi_k3/nvidia/"   # code root prefix for cites (repo-relative)

CONST = dict(
    H=7168, V=163840, NL=93, N_KDA=69, N_DSA=24, N_MOE=92, N_DENSE=1,
    # KDA
    Hk=96, dk=128, dv=128, P=12288, CONV=4, C_KDA=16,
    # MLA / DSA
    Nh=96, QLR=1536, KVL=512, DN=128, DR=64, DQK=192, DV=128, TOPK=2048,
    MLA_ROW=656,  # fp8_ds_mla bytes/token: 512 fp8 + 4x fp32 scales + 64 bf16 rope
    Ni=32, Di=128, IDX_ROW=68,  # MXFP4 indexer key: 64 B E2M1 + 4 B UE8M0
    # MoE
    E=896, TOPK_E=16, LAT=3584, I_E=3072, I_SH=6144, I_D=33792,
    MXFP4_B=0.5 + 1 / 32,   # bytes per MXFP4 weight element incl. UE8M0 scale
    MXFP8_B=1 + 1 / 32,     # bytes per MXFP8 activation element incl. scale
    # AttnRes
    ARB=12, NB=8,
    # DSpark draft
    D_L=5, D_HQ=96, D_HKV=16, D_HD=64, D_I=14336, D_W=2048, D_R=512, D_AUX=5,
)
C = CONST

SYMBOLS = {
    "T": "token rows of the forward (prefill: chunk tokens; decode/verify: B*q)",
    "B": "requests in the step",
    "k": "draft tokens per request (DSpark num_speculative_tokens)",
    "q": "verify rows per request in decode = 1 + k",
    "L": "context length (KV tokens incl. this step's) of a request",
    "Lsum": "sum over query rows of visible keys (prefill chunk of C tokens after "
            "prefix P: C*P + C*(C+1)/2; decode: B*q*L)",
    "Ltk": "sum over query rows of min(visible keys, TOPK) (keys a sparse-MLA row reads)",
    "Lu": "sum over requests of unique context keys touched this step (HBM SOL: each "
          "request's cache read once; prefill: P+C, decode: L)",
    "R": "rows through lm_head (prefill: B last-token rows; decode: B*q)",
    "AL": "accepted tokens per request per step (incl. bonus)",
    "Tc": "DSpark context rows (prefill: T, SOL min(T, D_W) per request; decode stock: "
          "B*q, SOL: B*AL)",
    "Tq": "DSpark query rows = B*k (sample_from_anchor: k rows per request)",
    "nseq": "sequences in a prefill chunk (state read/write count)",
}

ops = []


def op(**kw):
    kw.setdefault("phase", "both")
    kw.setdefault("count_per_forward", "1")
    kw.setdefault("weight_bytes", "0")
    kw.setdefault("act_in_bytes", "0")
    kw.setdefault("act_out_bytes", "0")
    kw.setdefault("kv_read_bytes", "0")
    kw.setdefault("kv_write_bytes", "0")
    kw.setdefault("state_read_bytes", "0")
    kw.setdefault("state_write_bytes", "0")
    kw.setdefault("comm", None)
    kw.setdefault("notes", "")
    ops.append(kw)


def gemm(id, name, layer, count, Kdim, Ndim, cite, wdt="bf16", adt="bf16",
         odt="bf16", rows="T", shard="", notes="", phase="both", comm=None):
    wb = {"bf16": "2", "fp32": "4", "mxfp4": "MXFP4_B", "mxfp8": "MXFP8_B"}[wdt]
    ab = {"bf16": "2", "mxfp8": "MXFP8_B", "fp32": "4"}[adt]
    ob = {"bf16": "2", "fp32": "4"}[odt]
    op(id=id, name=name, layer_type=layer, kind="gemm", phase=phase,
       count_per_forward=count,
       shapes=f"[{rows}, {Kdim}] x [{Kdim}, {Ndim}] -> [{rows}, {Ndim}]",
       dtype=dict(act_in=adt, weight=wdt, out=odt, compute=_compute(wdt, adt)),
       flops=f"2*({rows})*({Kdim})*({Ndim})",
       weight_bytes=f"({Kdim})*({Ndim})*{wb}",
       act_in_bytes=f"({rows})*({Kdim})*{ab}",
       act_out_bytes=f"({rows})*({Ndim})*{ob}",
       shard_axis=shard, cite=cite, notes=notes, comm=comm)


def _compute(w, a):
    if w == "bf16":
        return "bf16_mma"
    if w == "fp32":
        return "fp32_simt"
    if w == "mxfp4" and a == "mxfp8":
        return "mxf8f6f4_mma (fp8 rate)"
    return w


# ---------------------------------------------------------------- global (once)
op(id="G.embed", name="token embedding lookup", layer_type="global", kind="gather",
   shapes="ids[T] -> [T, H] from table [V, H]",
   dtype=dict(act_in="int32", weight="bf16", out="bf16", compute="none"),
   flops="0", weight_bytes="T*H*2", act_in_bytes="T*4", act_out_bytes="T*H*2",
   shard_axis="vocab (VocabParallelEmbedding) -> partial rows need AR / SP reduce-scatter",
   comm="TP: all-reduce [T,H] bf16 (or SP: reduce-scatter, sp_embed)",
   cite=[R + "model.py:1548", R + "model.py:233 (sp_embed)", R + "model.py:1730"],
   notes="weight_bytes = rows touched only (2.35 GB table never streamed).")
op(id="G.attnres_init", name="AttnRes block buffer alloc/seed", layer_type="global",
   kind="elementwise", shapes="block_residual [T, NB, H] bf16; prefix_sum = embed",
   dtype=dict(act_in="bf16", weight="-", out="bf16", compute="none"), flops="0",
   act_out_bytes="0", cite=[R + "model.py:1791-1801"],
   notes="new_empty only (no traffic); blocks written by block-write layers.")

# ---------------------------------------------------------------- per-layer AttnRes
# pre-attn call (layer l): sources = cdiv(l,12) blocks + updated prefix (+delta)
# post-attn call: sources = cdiv(l,12)+is_write blocks + prefix (+delta)
pre_sources = sum(math.ceil(l / 12) + 1 for l in range(93))
post_sources = sum(math.ceil(l / 12) + (l % 12 == 0) + 1 for l in range(93))
op(id="L.attnres_pre", name="AttnRes pre-attention mix + input RMSNorm (attn_res kernel)",
   layer_type="all", kind="memory", count_per_forward="NL",
   shapes="prefix[T,H] (+delta[T,H]) , blocks[T,nb,H] -> [T,H]; nb=cdiv(l,12)",
   dtype=dict(act_in="bf16", weight="bf16 (norm w, proj w [H])", out="bf16",
              compute="fp32_simt"),
   flops=f"T*H*8*{pre_sources}/NL", weight_bytes="2*H*2*2",
   act_in_bytes=f"T*H*2*({pre_sources}/NL + 1)",
   act_out_bytes="T*H*2*(1 + 8/93)",
   shard_axis="token-parallel (replicated weights); runs on the rank's token shard under SP",
   cite=[R + "model.py:1383-1412", R + "ops/attn_res.py:35-168", R + "ops/attn_res.py:170"],
   notes=(f"Per token: rmsnorm of each source (dot with norm_w*qk_w) -> softmax over "
          f"sources -> weighted sum -> output RMSNorm(input_layernorm). Sum over 93 layers "
          f"of sources = {pre_sources} (avg {pre_sources/93:.2f}); +1 read for delta, "
          f"block-write layers (l%12==0, 8 of them) store prefix into blocks. "
          f"Memory-bound in every regime (AI ~4 FLOP/B)."))
op(id="L.attnres_post", name="AttnRes post-attention mix + post_attention RMSNorm",
   layer_type="all", kind="memory", count_per_forward="NL",
   shapes="prefix[T,H], delta=attn_out[T,H], blocks[T,nb',H] -> [T,H]",
   dtype=dict(act_in="bf16", weight="bf16", out="bf16", compute="fp32_simt"),
   flops=f"T*H*8*{post_sources}/NL", weight_bytes="2*H*2*2",
   act_in_bytes=f"T*H*2*({post_sources}/NL + 1)", act_out_bytes="T*H*2*2",
   shard_axis="token-parallel",
   cite=[R + "model.py:1414-1449"],
   notes=f"sum of sources over layers = {post_sources}; delta add writes prefix back (BF16).")
op(id="L.sp_gather", name="SP all-gather of attention input (topology-dependent)",
   layer_type="all", kind="comm", count_per_forward="NL",
   shapes="[T/tp, H] -> [T, H] bf16", dtype=dict(act_in="bf16", weight="-", out="bf16",
                                                    compute="none"),
   flops="0", act_in_bytes="T*H*2", act_out_bytes="T*H*2",
   comm="SP/TP: all-gather T*H*2 bytes per layer (only when attention is head-sharded "
        "and the stream is token-sharded)",
   cite=[R + "model.py:1464-1468", R + "ops/sp_all_gather.py"],
   notes="Present only in SP topologies (C512 hybrid). Listed so the topology model can switch it on.")

# ---------------------------------------------------------------- KDA layer (69)
KDA = "kda"
gemm("KDA.in_proj", "in_proj_qkvgfab (q|k|v|g full-rank, f_a low-rank gate, beta)", KDA,
     "N_KDA", "H", "4*P + dk + Hk",
     [R + "kda.py:666-693", R + "kda.py:1001-1015"],
     shard="N: q/k/v/g/beta head-sharded over TP; f_a (128) replicated",
     notes="N = 4*12288 + 128 + 96 = 49376 (+pad to 16). 353.9M params, 708 MB BF16 per layer "
           "(the largest attention weight; 48.8 GB over 69 layers).")
gemm("KDA.f_b_proj", "f_b_proj (gate up-projection 128 -> 12288)", KDA, "N_KDA", "dk", "P",
     [R + "kda.py:695-701", R + "kda.py:1016"], shard="N (heads)")
op(id="KDA.conv", name="short causal depthwise conv (q,k,v; width 4) + SiLU",
   layer_type=KDA, kind="elementwise", count_per_forward="N_KDA",
   shapes="[T, 3P] with FP32 weight [3P, 4], conv state [3P, CONV-1+k] fp32 per request",
   dtype=dict(act_in="bf16", weight="fp32", out="bf16", compute="fp32_simt"),
   flops="T*3*P*(2*CONV + 4)", weight_bytes="3*P*CONV*4",
   act_in_bytes="T*3*P*2", act_out_bytes="T*3*P*2",
   state_read_bytes="(nseq if PREFILL else B)*3*P*(CONV-1)*4",
   state_write_bytes="(nseq if PREFILL else B)*3*P*(CONV-1+(0 if PREFILL else k))*4",
   shard_axis="channels (heads) over TP",
   cite=[R + "kda.py:707-716", R + "kda.py:1405-1488", R + "ops/kda_conv.py",
         "vllm/model_executor/layers/mamba/mamba_utils.py:296-319"],
   notes="Fused into the in_proj epilogue / decode kernel in the specialized decode "
         "(cutedsl/kda/inproj64.py). Conv state dtype = mamba_cache_dtype (FP32 per mission). "
         "Decode cache holds CONV-1+k columns (num_spec).")
op(id="KDA.gate_beta_norm", name="KDA gate g=lb*sigmoid(A*(g1+dt_bias)), beta=sigmoid, q/k L2 norm",
   layer_type=KDA, kind="elementwise", count_per_forward="N_KDA",
   shapes="g1[T,P], beta[T,Hk], q,k [T,Hk,dk]",
   dtype=dict(act_in="bf16", weight="fp32 (A_log[Hk], dt_bias[P])", out="fp32 regs",
              compute="fp32_simt"),
   flops="T*(P*6 + Hk*4 + 2*Hk*dk*3)", weight_bytes="(Hk + P)*4",
   act_in_bytes="0", act_out_bytes="0", shard_axis="heads",
   cite=[R + "ops/recoverssm.py:18-60", R + "kda.py:702-704", R + "kda.py:845-848"],
   notes="Always fused into the KDA core kernel (no separate traffic). gate_lower_bound=-5.")
op(id="KDA.core_prefill", name="chunked KDA (FlashKDA K1+K2, segscan) with FP32 state",
   layer_type=KDA, kind="attention", phase="prefill", count_per_forward="N_KDA",
   shapes="q,k,v [T,Hk,dk], g [T,Hk,dk], beta [T,Hk]; state [nseq,Hk,dk,dv] fp32",
   dtype=dict(act_in="bf16", weight="-", out="bf16",
              compute="bf16_mma (state staged bf16 into k.S / q.S MMAs, fp32 accum/state)"),
   flops="T*Hk*(2*C_KDA*(3*dk + 2*dv) + 6*dk*dv + (2/3)*C_KDA**2)",
   act_in_bytes="T*Hk*(3*dk*2 + dk*2 + 2)", act_out_bytes="T*Hk*dv*2",
   state_read_bytes="nseq*Hk*dk*dv*4",
   state_write_bytes="nseq*Hk*dk*dv*4*(1 + CKPT)",
   shard_axis="heads over TP (96 heads); segscan splits long sequences into segments",
   cite=[R + "kda.py:1490-1620", R + "ops/kda_segscan.py:1-60", R + "kda.py:421-476"],
   notes="FLOPs per token per head with chunk C: 2C(3dk+2dv) [KK^T, QK^T, W=A(Kg b), "
         "U=A(Vb), intra O] + 6 dk dv [W.S, q.S, K^T.Vnew] + (2/3)C^2 [triangular inverse]; "
         "C=16 (FlashKDA) -> 119K, C=64 -> 183K FLOP/token/head. Segment-parallel scan "
         "(segscan) adds ~2x K2 state flops on segmented sequences (M_p, C_p) -- "
         "derate term, not SOL. CKPT=1 when a prefix-cache checkpoint state is exported "
         "for the chunk (mamba align mode), else 0. Serial K2 over chunks: parallelism "
         "limited to Hk/tp x nseq CTAs pairs (latency term).")
op(id="KDA.core_decode", name="KDA verify recurrence (q rows/request) + deferred RecoverSSM commit",
   layer_type=KDA, kind="attention", phase="decode", count_per_forward="N_KDA",
   shapes="q,k,v,g [B*q,Hk,dk]; state [B,Hk,dk,dv] fp32; records u,kg [B,q,Hk,dk] bf16",
   dtype=dict(act_in="bf16", weight="-", out="bf16", compute="fp32_simt (FP32 state)"),
   flops="B*q*Hk*7*dk*dv + B*AL*Hk*3*dk*dv",
   act_in_bytes="B*q*Hk*(4*dk*2 + 2)", act_out_bytes="B*q*Hk*dv*2",
   state_read_bytes="B*Hk*dk*dv*4 + B*q*Hk*(dk+dv)*2",
   state_write_bytes="B*Hk*dk*dv*4 + B*q*Hk*(dk+dv)*2",
   shard_axis="heads over TP",
   cite=[R + "kda.py:1060-1146", R + "specialized_decode/recoverssm/__init__.py:1-30",
         R + "specialized_decode/cutedsl/kda/commit.py:1-20",
         R + "specialized_decode/cutedsl/kda/verify.py:1-10",
         "vllm/model_executor/layers/mamba/mamba_utils.py:133-149"],
   notes="Per verify row per head: decay (dk dv), S^T k (2 dk dv), rank-1 update (2 dk dv), "
         "q.S (2 dk dv) = 7 dk dv; deferred commit folds the previous step's AL accepted "
         "tokens (decay + outer product ~3 dk dv each) in the same pass. SOL traffic: FP32 "
         "state read once + written once per request per layer = 2*96*128*128*4 = 12.6 MB "
         "(869 MB per request per step over 69 layers) -- THE dominant decode byte term. "
         "Records (verify corrections + key-gates, bf16) are written and read back next step. "
         "Non-deferred alternatives: stock RecoverSSM commit after sampling (read S twice, "
         "write once) or q-slot state writes (write q states).")
op(id="KDA.o_norm", name="gated RMSNorm (o_norm, sigmoid(g2) gate, per-head 128)",
   layer_type=KDA, kind="elementwise", count_per_forward="N_KDA",
   shapes="[T,Hk,dv] x gate [T,Hk,dv] -> [T,P]",
   dtype=dict(act_in="bf16", weight="bf16/fp32 [dv]", out="bf16", compute="fp32_simt"),
   flops="T*P*8", weight_bytes="dv*4",
   act_in_bytes="T*P*2*2", act_out_bytes="T*P*2", shard_axis="heads",
   cite=[R + "kda.py:913", R + "kda.py:1720-1736", R + "ops/kda_norm.py"],
   notes="Fused into the decode kernel (output_gate / norm_weight args); separate in prefill.")
gemm("KDA.o_proj", "o_proj 12288 -> 7168", KDA, "N_KDA", "P", "H",
     [R + "kda.py:932-951", R + "kda.py:1047-1058"],
     shard="K (row-parallel): partial sums -> all-reduce / reduce-scatter",
     comm="TP: all-reduce (or RS under SP) of [T,H] bf16 partial sums; fused GEMM-AR "
          "(gemm_rs_ar) available")

# ---------------------------------------------------------------- DSA layer (24)
D = "dsa"
gemm("DSA.qkv_a_packed", "packed [q_a | kv_a(+k_pe) | indexer wk | indexer weights_proj]",
     D, "N_DSA", "H", "QLR + KVL + DR + Di + Ni",
     [R + "dsa.py:533-540", R + "dsa.py:189-196", R + "dsa.py:773-820", R + "dsa.py:897-913"],
     shard="replicated (disable_tp) in TP; under DSA-DP owned by one card",
     notes="N = 1536+512+64+128+32 = 2272. k_pe exists although the model is NoPE "
           "(mla_use_nope): the 64 dims are stored, not rotated.")
op(id="DSA.prologue", name="fused prologue: q_a RMSNorm, kv_a RMSNorm, FP8 MLA KV insert, "
                           "indexer key LayerNorm + MXFP4 quant + insert",
   layer_type=D, kind="elementwise", count_per_forward="N_DSA",
   shapes="q_c[T,1536], kv_c[T,512], k_pe[T,64], idx_k[T,128] -> caches",
   dtype=dict(act_in="bf16", weight="bf16/fp32 norm weights", out="bf16 q_c; fp8_ds_mla; mxfp4",
              compute="fp32_simt"),
   flops="T*(QLR*4 + KVL*6 + Di*10)", weight_bytes="(QLR + KVL)*2 + Di*8",
   act_in_bytes="T*(QLR + KVL + DR + Di)*2", act_out_bytes="T*QLR*2 + T*KVL*2",
   kv_write_bytes="T*(MLA_ROW + IDX_ROW)",
   shard_axis="token rows (replicated in TP: every rank writes its own copy unless "
              "DCP/DSA-DP stores KV once)",
   cite=[R + "dsa.py:922-948", R + "ops/dsa_indexer.py", R + "ops/dsa_indexer_mxfp4.py:1-20",
         "vllm/models/kimi_k3/common/dsa_cache.py:58-63",
         "vllm/v1/attention/backends/mla/flashmla_sparse.py:157-166"],
   notes="fp8_ds_mla row = 656 B (512 FP8 latent + 4 fp32 scales + 64 bf16 rope); MXFP4 "
         "indexer key = 68 B. kv_c_normed (bf16) also kept for the prefill path.")
gemm("DSA.qb_wq_packed", "packed [q_b (96 x 192) | indexer wq (32 x 128)]", D, "N_DSA",
     "QLR", "Nh*DQK + Ni*Di",
     [R + "dsa.py:544-551", R + "dsa.py:179-185", R + "dsa.py:950-963"],
     shard="N: q_b head-sharded over TP; wq_index replicated (indexer scored per rank)",
     notes="N = 18432 + 4096 = 22528.")
gemm("DSA.g_proj", "output-gate projection 7168 -> 96*128", D, "N_DSA", "H", "Nh*DV",
     [R + "dsa.py:626-636", R + "dsa.py:855-866"], shard="N (heads)",
     notes="Runs on an aux stream overlapping the attention front-end for small batches.")
op(id="DSA.idx_q_quant", name="indexer query MXFP4 quant + head-weight fold",
   layer_type=D, kind="elementwise", count_per_forward="N_DSA",
   shapes="index_q [T,32,128] bf16 -> E2M1 [T,32,64] + scales; weights [T,32]",
   dtype=dict(act_in="bf16", weight="-", out="mxfp4", compute="fp32_simt"),
   flops="T*Ni*Di*4", act_in_bytes="T*(Ni*Di + Ni)*2", act_out_bytes="T*Ni*(Di/2 + 4 + 4)",
   shard_axis="rows", cite=[R + "dsa.py:420-440", R + "ops/dsa_indexer_mxfp4.py"])
op(id="DSA.idx_score", name="indexer scoring logits[t,s]=sum_h w[t,h]*relu(q[t,h].k[s]) "
                            "over the full context",
   layer_type=D, kind="attention", count_per_forward="N_DSA",
   shapes="q [T,32,128] mxfp4, keys [L,128] mxfp4 (68 B) -> logits [T, L]",
   dtype=dict(act_in="mxfp4", weight="-", out="bf16 logits (scratch) / fp32",
              compute="mxf4 mma (block-scaled FP4 x FP4)"),
   flops="2*Ni*Di*Lsum + 3*Ni*Lsum",
   kv_read_bytes="Lu*IDX_ROW",
   act_in_bytes="T*Ni*(Di/2+8)", act_out_bytes="Lsum*2*SCRATCH",
   shard_axis="TP: every rank scores the whole context (replicated) unless DCP splits keys / "
              "DSA-DP gives the request to one card / ShardQ splits query rows",
   cite=[R + "dsa.py:420-472", R + "specialized_decode/cutedsl/dsa/indexer.py:1-40",
         "vllm/model_executor/layers/sparse_attn_indexer.py"],
   notes="Prefill: causal, Lsum = C*P + C(C+1)/2 per request. Skipped for rows with "
         "L <= TOPK (short-context bypass). Decode: one [128 keys x q*32] MMA per tile reads "
         "every key once per request per step (kv_read = L*68 B), FLOPs 2*32*128*q*L. "
         "SCRATCH=1 when logits are materialized (stock + current specialized: BF16 "
         "[T,L] write + read by select); SCRATCH=0 for a fused score+select (new method). "
         "At L=135K: 26.5 GFLOP per query row over 24 layers.")
op(id="DSA.topk", name="exact top-2048 over L logits per row (radix select)",
   layer_type=D, kind="memory", count_per_forward="N_DSA",
   shapes="logits [T, L] -> indices [T, 2048] int32",
   dtype=dict(act_in="bf16", weight="-", out="int32", compute="int/simt"),
   flops="Lsum*4", act_in_bytes="Lsum*2*SCRATCH", act_out_bytes="T*TOPK*4",
   shard_axis="rows; under DCP each rank selects a local top-k then a merge exchange",
   comm="DCP: local top-k candidates exchanged (decode packed top-N merge, prefill "
        "histogram merge) -- size T*TOPK*(4..8) B per rank",
   cite=[R + "specialized_decode/cutedsl/dsa/indexer.py:20-30", R + "dsa.py:270-296",
         R + "dsa_kernels/dcp_prefill_topk_merge/hist_merge.py"],
   notes="Exact (mission). Also converts logical -> physical KV slots.")
op(id="DSA.w_uk", name="W_UK absorption: q_nope [T,96,128] x W_UK^T [96,128,512] -> q_lat",
   layer_type=D, kind="gemm", count_per_forward="N_DSA",
   shapes="bmm over 96 heads: [T,128]x[128,512]",
   dtype=dict(act_in="bf16", weight="bf16 (kv_b_proj first half)", out="bf16",
              compute="bf16_mma"),
   flops="2*T*Nh*DN*KVL", weight_bytes="Nh*DN*KVL*2",
   act_in_bytes="T*Nh*DN*2", act_out_bytes="T*Nh*(KVL+DR)*2",
   shard_axis="heads", cite=["vllm/model_executor/layers/attention/mla_attention.py:55-120",
                             "vllm/model_executor/layers/attention/mla_attention.py:419",
                             "vllm/model_executor/layers/attention/sparse_mla_attention.py:1006"],
   notes="kv_b_proj [512 -> 96*(128+128)] (12.6M params) is split into W_UK / W_UV; with "
         "contexts > TOPK both prefill and decode run the absorbed MQA path "
         "(dense MHA prefill only when prefill_max_seq_len <= 2048, "
         "sparse_mla_attention.py:172). Query row = 512 latent + 64 rope = 576.")
op(id="DSA.sparse_mla", name="sparse MLA (FlashMLA sparse, absorbed MQA) over top-2048",
   layer_type=D, kind="attention", count_per_forward="N_DSA",
   shapes="q [T,96,576] bf16; KV gathered [T,2048,576] fp8_ds_mla -> o_lat [T,96,512]",
   dtype=dict(act_in="bf16", weight="-", out="bf16",
              compute="bf16_mma (FP8 KV dequantized to BF16 on load)"),
   flops="2*Nh*(KVL+DR+KVL)*Ltk + 4*Nh*Ltk",
   kv_read_bytes="(min(Lu, Ltk) if PREFILL else Ltk)*MLA_ROW",
   l2_gather_bytes="Ltk*MLA_ROW",
   act_in_bytes="T*Nh*(KVL+DR)*2", act_out_bytes="T*Nh*KVL*2",
   shard_axis="heads (TP: 96/tp heads, decode pads to 64/rank); DCP: keys; DSA-DP: request owner",
   cite=["vllm/model_executor/layers/attention/sparse_mla_attention.py:1006-1168",
         R + "specialized_decode/cutedsl/dsa/attn_ds.py:1-40", R + "dsa.py:965-976"],
   notes="l2_gather_bytes = gathered volume (every row reads its 2048 rows: 1.34 MB/row/layer). "
         "kv_read_bytes = HBM SOL: prefill unique rows min(Lu, Ltk)*656 (each request's cache "
         "once); decode Ltk*656 (no cross-row dedupe; dedupe lower bound B*2048*656) -- in prefill the "
         "request's whole cache (L*656 = 88.6 MB/layer at 135K) fits L2 (132 MB) so gather "
         "hits L2; in decode q verify rows of a request overlap heavily (union <= q*2048). "
         "Intensity on gathered bytes = 2*96*1088/656 = 318 FLOP/B (BF16 ridge 259) -> "
         "compute-bound-ish if L2-fed; decode is gather/latency bound. "
         "Rows with L<=2048 attend all L keys (Ltk counts min(L, TOPK)).")
op(id="DSA.w_uv", name="W_UV: o_lat [T,96,512] x W_UV [96,512,128] -> [T,96*128]",
   layer_type=D, kind="gemm", count_per_forward="N_DSA",
   shapes="bmm over 96 heads: [T,512]x[512,128]",
   dtype=dict(act_in="bf16", weight="bf16 (kv_b_proj second half)", out="bf16",
              compute="bf16_mma"),
   flops="2*T*Nh*KVL*DV", weight_bytes="Nh*KVL*DV*2",
   act_in_bytes="T*Nh*KVL*2", act_out_bytes="T*Nh*DV*2", shard_axis="heads",
   cite=["vllm/model_executor/layers/attention/mla_attention.py:116",
         "vllm/model_executor/layers/attention/mla_attention.py:494-500"])
op(id="DSA.gate_mul", name="output gate sigmoid(g) * attn_out",
   layer_type=D, kind="elementwise", count_per_forward="N_DSA",
   shapes="[T, 12288] x [T, 12288]", dtype=dict(act_in="bf16", weight="-", out="bf16",
                                               compute="fp32_simt"),
   flops="T*Nh*DV*5", act_in_bytes="T*Nh*DV*2*2", act_out_bytes="T*Nh*DV*2",
   shard_axis="heads", cite=[R + "dsa.py:868-875", R + "mla.py:121-124"],
   notes="Fusable into o_proj prologue (mxfp8 variant: gate_sigmoid_mul_quantize).")
gemm("DSA.o_proj", "o_proj 12288 -> 7168", D, "N_DSA", "Nh*DV", "H",
     [R + "dsa.py:576-582", R + "dsa.py:877-887"],
     shard="K (row-parallel) -> AR/RS",
     comm="TP: all-reduce / reduce-scatter [T,H] bf16 partial sums; DSA-DP: push_out of "
          "o (or o_lat) rows back to the TP group before o_proj")

# ---------------------------------------------------------------- MoE layer (92)
M = "moe"
gemm("MOE.router", "router gate 7168 -> 896 (fp32 logits)", M, "N_MOE", "H", "E",
     [R + "model.py:684-691", "vllm/model_executor/layers/fused_moe/router/gate_linear.py:25-110"],
     odt="fp32", shard="replicated (token-parallel)",
     notes="bf16 x bf16 -> fp32 cuBLAS; overlaps routed down-proj on an aux stream "
           "(<=256 tokens).")
op(id="MOE.topk", name="sigmoid + e_score_correction_bias, top-16 of 896, renormalize",
   layer_type=M, kind="elementwise", count_per_forward="N_MOE",
   shapes="[T,896] fp32 -> weights [T,16] fp32, ids [T,16] int32",
   dtype=dict(act_in="fp32", weight="fp32 bias [896]", out="fp32/int32", compute="fp32_simt"),
   flops="T*E*6", weight_bytes="E*4", act_in_bytes="T*E*4", act_out_bytes="T*TOPK_E*8",
   shard_axis="token-parallel", cite=[R + "model.py:937-976", R + "ops/prefill_glue.py"],
   notes="noaux_tc, one expert group; routed_scaling_factor 1.0.")
gemm("MOE.latent_down", "routed_expert_down_proj 7168 -> 3584 (latent)", M, "N_MOE",
     "H", "LAT", [R + "model.py:745-752", R + "model.py:1005-1049"],
     shard="replicated (ReplicatedLinear) -> token-parallel under SP/DEP")
op(id="MOE.dispatch", name="MegaMoE dispatch of latent rows to expert owners",
   layer_type=M, kind="comm", count_per_forward="N_MOE",
   shapes="[T,3584] -> T*16 (token,expert) rows; dedup per destination rank possible",
   dtype=dict(act_in="bf16", weight="-", out="mxfp8 (3584 + 112 B scale)", compute="quant"),
   flops="T*LAT*4", act_in_bytes="T*LAT*2", act_out_bytes="T*TOPK_E*LAT*MXFP8_B",
   comm="EP: T*min(TOPK_E, EP)*LAT*MXFP8_B bytes leave the rank in the worst case "
        "(per (token, destination rank) dedup; fraction (EP-1)/EP remote). In-kernel "
        "(fused dispatch, symmetric memory).",
   shard_axis="experts over EP",
   cite=[R + "megamoe_rubin/experts.py:1-30", R + "megamoe_rubin/runtime.py:1-10",
         R + "model.py:796-819"],
   notes="Activation dtype: MXFP8 up to 512 tokens/rank (RUBIN_MX_MAX_TOKENS_PER_RANK, "
         "experts.py:87), NVFP4 (w4a4, exact MXFP4->NVFP4 weight conversion) above -- "
         "mission allows MXFP8 activations; NVFP4 activation is an open rule question.")
op(id="MOE.fc1", name="expert FC1 (w13 gate|up) 3584 -> 6144, MXFP4 weights x MXFP8 act",
   layer_type=M, kind="grouped_gemm", count_per_forward="N_MOE",
   shapes="sum_e [m_e, 3584] x [3584, 6144]; sum m_e = T*16",
   dtype=dict(act_in="mxfp8", weight="mxfp4", out="fp32 acc -> act", compute="mxf8f6f4_mma (fp8 rate)"),
   flops="2*T*TOPK_E*LAT*2*I_E", weight_bytes="E_ACT*LAT*2*I_E*MXFP4_B",
   act_in_bytes="T*TOPK_E*LAT*MXFP8_B", act_out_bytes="0",
   shard_axis="experts over EP",
   cite=[R + "model.py:796-819", R + "megamoe_rubin/experts.py:1-30"],
   notes="E_ACT = experts touched this step = E*(1-(1-16/896)^T) (all 896 once T >~ 300); "
         "weights 11.7 MB/expert (FC1) -> 10.5 GB/layer. Rows per expert = 16T/896 = T/56: "
         "compute-bound only when T/56 >~ 200 (T >~ 11K tokens per EP group).")
op(id="MOE.act", name="SiTU(gate)*up (+clamp) and MXFP8 requant (fused in MegaMoE)",
   layer_type=M, kind="elementwise", count_per_forward="N_MOE",
   shapes="[T*16, 2*3072] -> [T*16, 3072]", dtype=dict(act_in="fp32", weight="-", out="mxfp8",
                                                    compute="fp32_simt"),
   flops="T*TOPK_E*I_E*12", shard_axis="experts",
   cite=[R + "ops/situ_and_mul.py:36-45", R + "model.py:6-38 (KimiVecSituAndMul)"],
   notes="beta=4, linear_beta=25 (tanh-based SiTU). Fused (no HBM traffic).")
op(id="MOE.fc2", name="expert FC2 (w2) 3072 -> 3584, MXFP4 weights x MXFP8 act",
   layer_type=M, kind="grouped_gemm", count_per_forward="N_MOE",
   shapes="sum_e [m_e, 3072] x [3072, 3584]",
   dtype=dict(act_in="mxfp8", weight="mxfp4", out="bf16", compute="mxf8f6f4_mma (fp8 rate)"),
   flops="2*T*TOPK_E*I_E*LAT", weight_bytes="E_ACT*I_E*LAT*MXFP4_B",
   act_out_bytes="T*TOPK_E*LAT*2", shard_axis="experts over EP",
   cite=[R + "model.py:796-819"])
op(id="MOE.combine", name="MegaMoE combine (weighted sum of 16 expert rows, BF16)",
   layer_type=M, kind="comm", count_per_forward="N_MOE",
   shapes="T*16 rows [3584] bf16 -> [T,3584]",
   dtype=dict(act_in="bf16", weight="-", out="bf16", compute="fp32 acc"),
   flops="T*TOPK_E*LAT*2", act_in_bytes="T*TOPK_E*LAT*2", act_out_bytes="T*LAT*2",
   comm="EP: return T*16*LAT*2 B (per (token,expert)) or T*min(16,EP)*LAT*2 B if each "
        "rank pre-reduces its experts' rows per token",
   shard_axis="experts", cite=[R + "megamoe_rubin/runtime.py:1-10", R + "model.py:1156"],
   notes="BF16 combine per mission.")
op(id="MOE.latent_norm", name="routed latent RMSNorm (3584)", layer_type=M, kind="elementwise",
   count_per_forward="N_MOE", shapes="[T,3584]",
   dtype=dict(act_in="bf16", weight="bf16", out="bf16", compute="fp32_simt"),
   flops="T*LAT*4", weight_bytes="LAT*2", act_in_bytes="T*LAT*2", act_out_bytes="T*LAT*2",
   shard_axis="token-parallel", cite=[R + "model.py:753-759", R + "model.py:213-243 (KimiRoutedOutputTransform)"],
   notes="Fusable into the up-proj prologue.")
gemm("MOE.latent_up", "routed_expert_up_proj 3584 -> 7168 (+ shared output beta-add)", M,
     "N_MOE", "LAT", "H", [R + "model.py:761-772", R + "model.py:213-243"],
     shard="replicated (token-parallel); shared partial added in epilogue")
gemm("MOE.shared_gate_up", "shared experts gate_up 7168 -> 2*6144", M, "N_MOE", "H", "2*I_SH",
     [R + "model.py:697-716", R + "model.py:419-426"], shard="N (TP) or replicated (SP)")
op(id="MOE.shared_act", name="shared SiTU-and-mul", layer_type=M, kind="elementwise",
   count_per_forward="N_MOE", shapes="[T, 12288] -> [T, 6144]",
   dtype=dict(act_in="bf16", weight="-", out="bf16", compute="fp32_simt"),
   flops="T*I_SH*12", act_in_bytes="T*2*I_SH*2", act_out_bytes="T*I_SH*2",
   shard_axis="N", cite=[R + "model.py:6-38", R + "ops/situ_and_mul.py"],
   notes="Fusable into gate_up epilogue.")
gemm("MOE.shared_down", "shared experts down 6144 -> 7168", M, "N_MOE", "I_SH", "H",
     [R + "model.py:427-438", R + "model.py:1108-1119"],
     shard="K (row-parallel) -> AR (one fused AR per MoE layer: shared partial + routed rows)",
     comm="TP: all-reduce [T,H] bf16 (token-shard path: one AR carries shared partials + "
          "routed rows, model.py:1051-1119)")

# ---------------------------------------------------------------- layer-0 dense MLP
DN = "dense_mlp"
gemm("DENSE.gate_up", "layer-0 dense MLP gate_up 7168 -> 2*33792", DN, "N_DENSE", "H", "2*I_D",
     [R + "model.py:1327-1340", R + "model.py:419-426"], shard="N (TP) / replicated under SP")
op(id="DENSE.act", name="layer-0 SiTU-and-mul", layer_type=DN, kind="elementwise",
   count_per_forward="N_DENSE", shapes="[T, 67584] -> [T, 33792]",
   dtype=dict(act_in="bf16", weight="-", out="bf16", compute="fp32_simt"),
   flops="T*I_D*12", act_in_bytes="T*2*I_D*2", act_out_bytes="T*I_D*2", shard_axis="N",
   cite=[R + "model.py:166-186"])
gemm("DENSE.down", "layer-0 dense MLP down 33792 -> 7168", DN, "N_DENSE", "I_D", "H",
     [R + "model.py:427-438"], shard="K -> AR/RS",
     comm="TP: all-reduce [T,H] bf16")

# ---------------------------------------------------------------- final / logits / sampling
F = "final"
op(id="F.attnres_final", name="final AttnRes mix (8 blocks + prefix, no output norm)",
   layer_type=F, kind="memory", shapes="[T,H] + blocks[T,8,H] -> [T,H]",
   dtype=dict(act_in="bf16", weight="bf16", out="bf16", compute="fp32_simt"),
   flops="T*H*8*(NB+1)", weight_bytes="2*H*2", act_in_bytes="T*H*2*(NB+2)",
   act_out_bytes="T*H*2", shard_axis="token-parallel",
   cite=[R + "model.py:1842-1857"],
   notes="Prefill could restrict to the R logit rows + Tc draft rows (new method): only "
         "those rows are consumed after the last layer.")
op(id="F.sp_gather_out", name="SP all-gather of the final hidden (+aux hidden) rows",
   layer_type=F, kind="comm", shapes="[T/tp, H*(1+D_AUX)] -> [T, ...]",
   dtype=dict(act_in="bf16", weight="-", out="bf16", compute="none"),
   flops="0", act_in_bytes="T*H*2*(1+D_AUX)", act_out_bytes="T*H*2*(1+D_AUX)",
   comm="SP only: all-gather T*H*2*(1+D_AUX) bytes", shard_axis="-",
   cite=[R + "model.py:1861-1873"])
op(id="F.norm", name="final RMSNorm (applied in compute_logits on logit rows)",
   layer_type=F, kind="elementwise", shapes="[R,H]",
   dtype=dict(act_in="bf16", weight="bf16", out="bf16", compute="fp32_simt"),
   flops="R*H*4", weight_bytes="H*2", act_in_bytes="R*H*2", act_out_bytes="R*H*2",
   shard_axis="rows", cite=[R + "model.py:2246-2254"])
gemm("F.lm_head", "lm_head 7168 -> 163840 (target verify / last-token logits)", F, "1",
     "H", "V", [R + "model.py:2246-2300", R + "specialized_decode/cutedsl/verify/lmhead.py"],
     rows="R", odt="fp32", shard="N = vocab over TP (each rank its vocab shard)",
     comm="TP: argmax/top token exchange or sampled-verify partial reductions "
          "(no full-vocab gather on the fast path: verify/exchange.py)",
     notes="2.35 GB BF16 weights; decode R = B*q rows.")
op(id="F.rejection", name="probabilistic rejection sampling (target vs draft probs) + bonus",
   layer_type=F, kind="memory", phase="decode",
   shapes="target logits [B,q,V] (vocab-sharded) , draft logits cache [B,k,V]",
   dtype=dict(act_in="fp32 target logits / head_dtype draft logits", weight="-",
              out="int32 tokens", compute="fp32_simt"),
   flops="B*q*V*8 + B*k*V*6", act_in_bytes="B*q*V*4 + B*k*V*2",
   act_out_bytes="B*q*4", shard_axis="vocab over TP (sampled_verify.run on shard logits)",
   comm="TP: per-row LSE / max / sampled-candidate exchange, O(B*q*tp*16 B)",
   cite=[R + "model.py:2278-2300", R + "specialized_decode/runtime/sampled_verify.py:524-640",
         "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:1110",
         "vllm/v1/worker/gpu/spec_decode/speculator.py:499-505"],
   notes="softmax (LSE) of target and draft per row, accept test p/q, residual "
         "max(0,p-q) resample at first rejection, bonus sample. Memory-bound on the "
         "logit bytes. Prefill: plain sampling of R rows (R*V*4 read).")

# ---------------------------------------------------------------- DSpark draft
DS = "dspark"
op(id="DS.aux_capture", name="aux hidden capture: AttnRes stream after target layers "
                             "75,79,83,87,91 (aux ids 76..92)",
   layer_type=DS, kind="memory", count_per_forward="D_AUX",
   shapes="[T,H] per aux layer (attn_res mix with the consumer layer's weights)",
   dtype=dict(act_in="bf16", weight="bf16", out="bf16", compute="fp32_simt"),
   flops="T*H*8*(NB+1)", act_in_bytes="T*H*2*(NB+2)", act_out_bytes="T*H*2",
   shard_axis="token-parallel", cite=[R + "model.py:1693-1728", R + "model.py:1803-1820"],
   notes="Extra attn_res pass per aux layer on every target row (SOL: only Tc rows).")
gemm("DS.fc", "context fc: cat(5 aux) 35840 -> 7168", DS, "1", "D_AUX*H", "H",
     [R + "dspark_gqa.py:158-177", R + "dspark_gqa.py:327-355",
      "vllm/model_executor/models/qwen3_dflash.py:495-510"], rows="Tc",
     shard="replicated; row-sharded over TP for >=256 rows/rank (combine_aux_rows)",
     notes="256.9M params (514 MB). Decode stock graph runs all B*q verify rows; SOL B*AL. "
           "Prefill (P/D prefill instance) runs it on the 16K chunk; SOL only the last "
           "D_W=2048 tokens per request matter for the sliding window (new method).")
op(id="DS.hidden_norm", name="hidden_norm RMSNorm of context rows", layer_type=DS,
   kind="elementwise", shapes="[Tc,H]",
   dtype=dict(act_in="bf16", weight="bf16", out="bf16", compute="fp32_simt"),
   flops="Tc*H*4", weight_bytes="H*2", act_in_bytes="Tc*H*2", act_out_bytes="Tc*H*2",
   shard_axis="rows", cite=[R + "dspark_gqa.py:170-177"])
gemm("DS.ctx_kv_proj", "context K/V projection of all 5 layers (fused K/V weight) "
     "7168 -> 5*2*16*64", DS, "1", "H", "D_L*2*D_HKV*D_HD",
     [R + "dspark_gqa.py:179-201", "vllm/model_executor/models/qwen3_dflash.py:592-720"],
     rows="Tc", shard="N (kv heads) or replicated")
op(id="DS.ctx_kv_insert", name="context k_norm + RoPE + paged KV insert (5 layers)",
   layer_type=DS, kind="elementwise", shapes="[Tc, 5, 2, 16, 64] -> 5 BF16 paged caches",
   dtype=dict(act_in="bf16", weight="bf16", out="bf16", compute="fp32_simt"),
   flops="Tc*D_L*D_HKV*D_HD*10", act_in_bytes="Tc*D_L*2*D_HKV*D_HD*2",
   kv_write_bytes="Tc*D_L*2*D_HKV*D_HD*2", shard_axis="kv heads",
   cite=[R + "specialized_decode/cutedsl/draft/gqa_context_kv.py:1-30"],
   notes="20,480 B per context token (BF16, 5 layers). Prefill SOL: only the last 2048 "
         "tokens of the prompt need draft KV.")
op(id="DS.embed", name="draft query embedding (anchor + k-1 mask tokens, target table)",
   layer_type=DS, kind="gather", phase="decode", shapes="ids[Tq] -> [Tq,H]",
   dtype=dict(act_in="int32", weight="bf16", out="bf16", compute="none"),
   flops="0", weight_bytes="Tq*H*2", act_out_bytes="Tq*H*2",
   shard_axis="vocab -> AR", comm="TP: all-reduce [Tq,H] bf16 (or replicated table)",
   cite=[R + "dspark_gqa.py:246-262", "vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:1-20"],
   notes="sample_from_anchor: k query rows per request (anchor + k-1 noise).")
op(id="DS.norm_ar", name="draft input/post RMSNorm with fused all-reduce (x2 per layer) + final",
   layer_type=DS, kind="elementwise", phase="decode", count_per_forward="2*D_L",
   shapes="[Tq,H]", dtype=dict(act_in="bf16", weight="bf16", out="bf16", compute="fp32_simt"),
   flops="Tq*H*6", weight_bytes="H*2", act_in_bytes="Tq*H*2*2", act_out_bytes="Tq*H*2*2",
   comm="TP: all-reduce [Tq,H] bf16 per site (fused_allreduce_rms_norm); 2*D_L sites",
   shard_axis="rows", cite=[R + "dspark_gqa.py:83-125"])
gemm("DS.qkv", "draft qkv_proj 7168 -> 96*64 + 2*16*64", DS, "D_L", "H",
     "D_HQ*D_HD + 2*D_HKV*D_HD", ["vllm/model_executor/models/qwen3_dflash.py:217-225",
                                  R + "dspark_gqa.py:95-105"], rows="Tq",
     shard="N (heads)", phase="decode")
op(id="DS.attn", name="draft sliding-window GQA attention (window 2048, causal in block)",
   layer_type=DS, kind="attention", phase="decode", count_per_forward="D_L",
   shapes="q [Tq,96,64], KV [min(L,W)+k, 16, 64] bf16 per request",
   dtype=dict(act_in="bf16", weight="-", out="bf16", compute="bf16_mma (mma.sync)"),
   flops="2*Tq*D_HQ*D_HD*2*D_W + Tq*D_HQ*(D_HD*8)",
   kv_read_bytes="B*D_W*2*D_HKV*D_HD*2", kv_write_bytes="Tq*2*D_HKV*D_HD*2",
   act_in_bytes="Tq*(D_HQ + 2*D_HKV)*D_HD*2", act_out_bytes="Tq*D_HQ*D_HD*2",
   shard_axis="kv heads over TP (16 kv heads)",
   cite=[R + "specialized_decode/cutedsl/draft/gqa_window.py:1-40", R + "dspark_gqa.py:62-78"],
   notes="Includes q/k RMSNorm + neox RoPE + query K/V insert. KV read 8.39 MB per "
         "request per layer (41.9 MB per draft step). AI = 2*96*64*2/(16*64*2*2) * k "
         "~ 6k FLOP/B -> memory bound.")
gemm("DS.o_proj", "draft o_proj 6144 -> 7168", DS, "D_L", "D_HQ*D_HD", "H",
     ["vllm/model_executor/models/qwen3_dflash.py:226-232"], rows="Tq", shard="K -> AR",
     phase="decode", comm="TP: AR fused into the next RMSNorm")
gemm("DS.gate_up", "draft MLP gate_up 7168 -> 2*14336", DS, "D_L", "H", "2*D_I",
     ["vllm/model_executor/models/qwen3_dflash.py:365-372"], rows="Tq", shard="N",
     phase="decode")
op(id="DS.silu", name="draft SiLU-and-mul", layer_type=DS, kind="elementwise", phase="decode",
   count_per_forward="D_L", shapes="[Tq, 28672] -> [Tq, 14336]",
   dtype=dict(act_in="bf16", weight="-", out="bf16", compute="fp32_simt"),
   flops="Tq*D_I*5", act_in_bytes="Tq*2*D_I*2", act_out_bytes="Tq*D_I*2", shard_axis="N",
   cite=["vllm/model_executor/models/qwen2.py (Qwen2MLP)"])
gemm("DS.down", "draft MLP down 14336 -> 7168", DS, "D_L", "D_I", "H",
     ["vllm/model_executor/models/qwen3_dflash.py:365-372"], rows="Tq", shard="K -> AR",
     phase="decode", comm="TP: AR fused into next RMSNorm")
gemm("DS.lm_head", "draft base logits on k rows (target lm_head weights)", DS, "1", "H", "V",
     [R + "dspark_gqa.py:363-368", "vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:285-292"],
     rows="Tq", odt="fp32", shard="N = vocab over TP", phase="decode",
     notes="Same 2.35 GB weight as F.lm_head (re-streamed unless fused with verify).")
op(id="DS.markov", name="sequential Markov head x k: w1[prev] (512) -> w2 512 -> 163840 bias "
                       "+ sample",
   layer_type=DS, kind="gemm", phase="decode", count_per_forward="k",
   shapes="per position i: [B,512] x [512, V] -> [B,V]; add base logits; gumbel/argmax",
   dtype=dict(act_in="bf16", weight="bf16 (w1 [V,512] gather, w2 [512,V])", out="fp32 logits",
              compute="bf16_mma / gemv"),
   flops="2*B*D_R*V + B*V*6", weight_bytes="D_R*V*2 + B*D_R*2",
   act_in_bytes="B*V*4", act_out_bytes="B*V*2 + B*4",
   shard_axis="vocab columns over TP (fused sampled_markov_draft), argmax/gumbel exchange per position",
   comm="TP: per-position (max, idx) exchange O(B*tp*8 B) -- k sequential latency-bound rounds",
   cite=["vllm/model_executor/models/qwen3_dspark.py:48-93",
         "vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:270-325",
         R + "dspark_gqa.py:383-425", R + "specialized_decode/cutedsl/draft/markov.py"],
   notes="w2 = 84M params (168 MB BF16) re-read per position unless the vocab shard stays "
         "L2-resident (168/tp MB; fits L2 at tp>=2). Writes the draft-logits cache "
         "(head_dtype) for probabilistic rejection. Strictly sequential over k positions.")
op(id="DS.confidence", name="confidence head (optional; only with adaptive verification)",
   layer_type=DS, kind="gemm", phase="decode", shapes="[Tq, 7168+512] fp32 -> [Tq]",
   dtype=dict(act_in="fp32", weight="fp32", out="fp32", compute="fp32"),
   flops="2*Tq*(H+D_R)", weight_bytes="(H+D_R)*4", shard_axis="rows",
   cite=["vllm/model_executor/models/qwen3_dspark.py:164-190",
         "vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:106-114"],
   notes="Off by default (enable_adaptive_verification); excluded from SOL totals.")

# ---------------------------------------------------------------- SOL fusion flags
FUSED = {"KDA.conv", "KDA.gate_beta_norm", "KDA.o_norm", "DSA.idx_q_quant", "DSA.gate_mul",
         "MOE.topk", "MOE.act", "MOE.latent_norm", "MOE.shared_act", "DENSE.act", "DS.silu",
         "DS.hidden_norm", "F.norm", "DS.ctx_kv_insert"}
for o in ops:
    o["fused_in_sol"] = o["id"] in FUSED
    if o["fused_in_sol"]:
        o["notes"] = (o["notes"] + " " if o["notes"] else "") + (
            "SOL: fused into the neighbouring GEMM/kernel prologue/epilogue -> its "
            "activation bytes are not HBM traffic (kv/state bytes still count).")

# ---------------------------------------------------------------- evaluation
RIDGE = {  # FLOP/B at 13.9 TB/s
    "bf16_mma": 3.6e15 / 13.9e12, "mxf8f6f4_mma (fp8 rate)": 10.5e15 / 13.9e12,
    "mxf4 mma (block-scaled FP4 x FP4)": 17.5e15 / 13.9e12,
    "fp32_simt": 131.5e12 / 13.9e12,
}


def ridge(op_):
    c = op_["dtype"]["compute"]
    for k_, v_ in RIDGE.items():
        if c.startswith(k_.split(" ")[0]):
            return v_
    return RIDGE["fp32_simt"]


def env_prefill(Cc=16384, Pp=120000, nseq=1):
    T = Cc
    L = Pp + Cc
    Lsum = Cc * Pp + Cc * (Cc + 1) / 2
    Ltk = Cc * C["TOPK"] if Pp >= C["TOPK"] else sum(min(Pp + i + 1, C["TOPK"]) for i in range(Cc))
    e = dict(C, T=T, B=nseq, nseq=nseq, k=0, q=1, L=L, Lsum=Lsum, Ltk=Ltk, Lu=L, R=nseq,
             AL=1, Tc=T, Tq=0, PREFILL=True, CKPT=1, SCRATCH=1)
    e["E_ACT"] = C["E"] * (1 - (1 - 16 / C["E"]) ** T)
    return e


def env_decode(Bb=64, kk=3, Ll=135000, AL=3.08):
    q = 1 + kk
    T = Bb * q
    e = dict(C, T=T, B=Bb, nseq=Bb, k=kk, q=q, L=Ll, Lsum=T * Ll, Ltk=T * min(Ll, C["TOPK"]),
             Lu=Bb * Ll, R=T, AL=AL, Tc=T, Tq=Bb * kk, PREFILL=False, CKPT=0, SCRATCH=1)
    e["E_ACT"] = C["E"] * (1 - (1 - 16 / C["E"]) ** T)
    return e


def ev(expr, env):
    return float(eval(str(expr), {"min": min, "max": max, "math": math}, env))


def evaluate(env, phase):
    rows = []
    for o in ops:
        if o["phase"] not in ("both", phase):
            continue
        n = ev(o["count_per_forward"], env)
        f = ev(o["flops"], env) * n
        w = ev(o["weight_bytes"], env) * n
        a = 0.0 if o["fused_in_sol"] else (ev(o["act_in_bytes"], env) + ev(o["act_out_bytes"], env)) * n
        kv = (ev(o["kv_read_bytes"], env) + ev(o["kv_write_bytes"], env)) * n
        st = (ev(o["state_read_bytes"], env) + ev(o["state_write_bytes"], env)) * n
        byt = w + a + kv + st
        ai = f / byt if byt else float("inf")
        if o["kind"] == "comm":
            bound = "comm"
        else:
            bound = "compute" if ai > ridge(o) else "memory"
        rows.append(dict(id=o["id"], count=n, flops=f, weight_bytes=w, act_bytes=a,
                         kv_bytes=kv, state_bytes=st, ai=ai, ridge=ridge(o), bound=bound))
    return rows


REF = {
    "prefill_C16384_P120000": ("prefill", env_prefill()),
    "decode_B64_k3_L135K": ("decode", env_decode(64, 3)),
    "decode_B64_k7_L135K": ("decode", env_decode(64, 7, AL=4.16)),
    "decode_B8_k3_L135K": ("decode", env_decode(8, 3)),
}
evals = {}
for name, (ph, env) in REF.items():
    rr = evaluate(env, ph)
    for o in ops:
        pass
    evals[name] = rr
# per-request marginal (non-weight) decode bytes/flops: (B=65) - (B=64), all ops
marginal = {}
for kk, al in ((3, 3.08), (5, 3.81), (7, 4.16)):
    a = {r["id"]: r for r in evaluate(env_decode(64, kk, AL=al), "decode")}
    b = {r["id"]: r for r in evaluate(env_decode(65, kk, AL=al), "decode")}
    rows_ = {}
    for i in a:
        if a[i]["bound"] == "comm":
            continue
        d_bytes = (b[i]["act_bytes"] + b[i]["kv_bytes"] + b[i]["state_bytes"]) - \
                  (a[i]["act_bytes"] + a[i]["kv_bytes"] + a[i]["state_bytes"])
        d_fl = b[i]["flops"] - a[i]["flops"]
        if d_bytes > 1e5 or d_fl > 1e8:
            rows_[i] = dict(bytes=d_bytes, flops=d_fl)
    marginal[f"k{kk}_L135K"] = rows_
# bound classification per op (prefill ref / decode ref B64 k3)
bmap_p = {r["id"]: r["bound"] for r in evals["prefill_C16384_P120000"]}
bmap_d = {r["id"]: r["bound"] for r in evals["decode_B64_k3_L135K"]}
for o in ops:
    o["bound_prefill_ref"] = bmap_p.get(o["id"], "n/a")
    o["bound_decode_ref"] = bmap_d.get(o["id"], "n/a")

# parameter inventory
PARAMS = {
    "embed": C["V"] * C["H"],
    "lm_head": C["V"] * C["H"],
    "kda_per_layer": C["H"] * (4 * C["P"] + C["dk"] + C["Hk"]) + C["dk"] * C["P"]
                     + C["P"] * C["H"] + 3 * C["P"] * C["CONV"],
    "dsa_per_layer": C["H"] * 2272 + C["QLR"] * 22528 + C["KVL"] * C["Nh"] * 256
                     + C["H"] * C["Nh"] * C["DV"] + C["Nh"] * C["DV"] * C["H"],
    "moe_dense_per_layer": C["H"] * C["E"] + 2 * C["H"] * C["LAT"] + 3 * C["H"] * C["I_SH"],
    "experts_per_layer": C["E"] * 3 * C["LAT"] * C["I_E"],
    "dense_mlp_layer0": 3 * C["H"] * C["I_D"],
    "draft_per_layer": C["H"] * (C["D_HQ"] + 2 * C["D_HKV"]) * C["D_HD"]
                       + C["D_HQ"] * C["D_HD"] * C["H"] + 3 * C["H"] * C["D_I"],
    "draft_fc": C["D_AUX"] * C["H"] * C["H"],
    "draft_markov": 2 * C["V"] * C["D_R"],
}
BYTES = {
    "bf16_nonexpert_target_GB": (PARAMS["embed"] + PARAMS["lm_head"]
                                 + 69 * PARAMS["kda_per_layer"] + 24 * PARAMS["dsa_per_layer"]
                                 + 92 * PARAMS["moe_dense_per_layer"] + PARAMS["dense_mlp_layer0"]) * 2 / 1e9,
    "mxfp4_experts_GB": 92 * PARAMS["experts_per_layer"] * C["MXFP4_B"] / 1e9,
    "draft_bf16_GB": (5 * PARAMS["draft_per_layer"] + PARAMS["draft_fc"] + PARAMS["draft_markov"]) * 2 / 1e9,
    "kv_per_token_B": dict(mla_fp8_ds=24 * C["MLA_ROW"], indexer_mxfp4=24 * C["IDX_ROW"],
                           total_dsa=24 * (C["MLA_ROW"] + C["IDX_ROW"]),
                           draft_bf16_window_only=5 * 2 * 16 * 64 * 2),
    "kda_state_per_request_B": dict(recurrent_fp32=69 * 96 * 128 * 128 * 4,
                                    conv_fp32_k0=69 * 3 * 12288 * 3 * 4),
}

doc = dict(
    model="Kimi K3.1-Pro (d9e99b07) + DSpark draft (e4e6e626)",
    generated_by="solmodel_work/gen_ops.py",
    conventions=dict(
        bytes_and_flops="FULL op (unsharded); topology model divides by its shard_axis",
        flops="multiply-add = 2 FLOPs; elementwise ops counted approximately",
        formulas="python expressions over symbols + constants; PREFILL bool, CKPT (0/1 "
                 "prefix-cache checkpoint export), SCRATCH (0/1 materialized indexer "
                 "logits), E_ACT (experts touched) are switches",
        layer_index="0-based: config lists are 1-based (is_kda_layer(l) = l+1 in kda_layers, "
                    "kimi_linear.py:225-235). KDA idx {0,1,2,4,5,6,...,90}; DSA idx "
                    "{3,7,...,91,92}; layer 0 = KDA + dense MLP; layers 1..92 MoE",
        ridge_flop_per_byte={k_: round(v_, 1) for k_, v_ in RIDGE.items()},
    ),
    symbols=SYMBOLS, constants=CONST, layer_counts=dict(kda=69, dsa=24, moe=92, dense=1,
                                                         attnres_blocks=8, draft_layers=5),
    params=PARAMS, bytes_summary=BYTES, ops=ops, reference_evaluations=evals,
    decode_per_request_marginal=marginal,
)
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "ops.json"), "w") as fh:
    json.dump(doc, fh, indent=1, default=str)
print("ops", len(ops))
for name, rr in evals.items():
    tf = sum(r["flops"] for r in rr); tb = sum(r["weight_bytes"] + r["act_bytes"] + r["kv_bytes"] + r["state_bytes"] for r in rr)
    print(name, f"{tf/1e12:.1f} TFLOP  {tb/1e9:.1f} GB")
    top = sorted(rr, key=lambda r: -(r["weight_bytes"] + r["act_bytes"] + r["kv_bytes"] + r["state_bytes"]))[:6]
    for r in top:
        print("   B", r["id"], f"{(r['weight_bytes']+r['act_bytes']+r['kv_bytes']+r['state_bytes'])/1e9:.2f} GB", r["bound"])
    top = sorted(rr, key=lambda r: -r["flops"])[:6]
    for r in top:
        print("   F", r["id"], f"{r['flops']/1e12:.2f} TF", r["bound"], f"ai={r['ai']:.0f}")
print(json.dumps(BYTES, indent=1))
print({k: v / 1e6 for k, v in PARAMS.items()})


# ---------------------------------------------------------------- markdown
def fmt(x, unit=""):
    for s, d in (("P", 1e15), ("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= d:
            return f"{x/d:.2f}{s}{unit}"
    return f"{x:.0f}{unit}"


def by_type(rr):
    lt = {o["id"]: o["layer_type"] for o in ops}
    agg = {}
    for r in rr:
        t = lt[r["id"]]
        a = agg.setdefault(t, dict(flops=0, w=0, a=0, kv=0, st=0, comm=0))
        hb = r["weight_bytes"] + r["act_bytes"] + r["kv_bytes"] + r["state_bytes"]
        if r["bound"] == "comm":
            a["comm"] += hb
        else:
            a["flops"] += r["flops"]; a["w"] += r["weight_bytes"]; a["a"] += r["act_bytes"]
            a["kv"] += r["kv_bytes"]; a["st"] += r["state_bytes"]
    return agg


md = []
md.append("# K3.1-Pro + DSpark op inventory (SOL model input)\n")
md.append("Byte columns below are the SOL view: activation bytes of ops flagged "
          "`fused_in_sol` are dropped (they live in a neighbour's prologue/epilogue); "
          "GEMM activations, KV, state and weights always count; indexer logits scratch "
          "counted (SCRATCH=1, stock/current kernels).\n")
md.append("Generated by `solmodel_work/gen_ops.py` -> `ops.json` (machine-readable, formulas as "
          "python expressions). Every op cites the serving code (branch wzhao/k31-agentx-dev, "
          "paths relative to the vLLM repo root). Bytes/FLOPs are for the FULL op "
          "(unsharded); the topology model divides by `shard_axis` and adds the `comm` "
          "entries. 2 FLOPs per MAC.\n")
md.append("## Structure (from code, not only config)\n")
md.append("""- 93 decoder layers, 0-based. Config layer lists are **1-based** (`is_kda_layer(l)` tests `l+1`,
  `vllm/transformers_utils/configs/kimi_linear.py:225-235`): KDA = idx {0,1,2,4,5,6,...,88,89,90} (69),
  DSA = idx {3,7,11,...,87,91,92} (24; the last two layers are both DSA). Layer 0 = **KDA attention + dense
  MLP** (first_k_dense_replace=1); layers 1..92 = MoE (`model.py:1232-1237`).
- Every layer: AttnRes pre-mix (+input RMSNorm) -> attention -> AttnRes post-mix (+post RMSNorm) -> MLP/MoE
  (`model.py:1451-1490`). AttnRes keeps `[T, 8, 7168]` BF16 block residuals; block-write layers l%12==0.
- KDA: `in_proj_qkvgfab` 7168 -> 4x12288 + 128 + 96, `f_b_proj` 128 -> 12288, FP32 depthwise conv (4) on q,k,v,
  KDA core (FP32 state 96x128x128 per request), gated RMSNorm, `o_proj` 12288 -> 7168 (`kda.py:666-951`).
- DSA (MLA, NoPE, output gate): packed `[q_a 1536 | kv_a 512+64 | idx wk 128 | idx w 32]`, fused prologue
  (2 RMSNorms + indexer LayerNorm + FP8 MLA insert 656 B + MXFP4 indexer insert 68 B), packed
  `[q_b 96x192 | wq_index 32x128]`, `g_proj`, indexer score + exact top-2048, absorbed MQA
  (W_UK, FlashMLA sparse over 2048 FP8 rows of 576, W_UV), sigmoid gate, `o_proj` (`dsa.py:475-976`).
  Contexts > 2048 always take the absorbed path, also in prefill (`sparse_mla_attention.py:172`).
- MoE: router (bf16 -> fp32, 896), sigmoid+bias top-16, latent down 7168->3584 (BF16, replicated), MegaMoE
  (dispatch MXFP8 latent, FC1 3584->6144 MXFP4, SiTU, FC2 3072->3584, BF16 combine), latent RMSNorm,
  latent up 3584->7168 (BF16, shared output added in epilogue), shared experts 7168->12288->6144->7168 BF16
  (`model.py:605-839`).
- Final: AttnRes final mix, RMSNorm in compute_logits, lm_head 7168->163840 BF16, probabilistic rejection.
- DSpark (release e4e6e626, `dspark_gqa.py`, `qwen3_dspark.py`, `dspark/speculator.py`): aux hidden from
  target layers 76/80/84/88/92 (AttnRes-mixed captures), context `fc` 35840->7168 + hidden_norm, fused K/V
  projection of all 5 layers (7168 -> 10240) + k_norm/RoPE/insert (BF16, 20 KB/token); query block = k rows per
  request (sample_from_anchor), 5 Qwen3 layers (qkv 7168->8192, SWA-2048 GQA 96q/16kv x 64, o 6144->7168,
  MLP 7168->2x14336->7168, AR fused into RMSNorms), lm_head on k rows, then k **sequential** Markov steps
  (w1 gather 512, w2 512->163840 bias, sample). Confidence head only with adaptive verification (off).
""")
md.append("## Model totals\n")
md.append(f"- BF16 non-expert target weights: **{BYTES['bf16_nonexpert_target_GB']:.1f} GB** "
          f"(KDA {69*PARAMS['kda_per_layer']*2/1e9:.1f} GB, DSA {24*PARAMS['dsa_per_layer']*2/1e9:.1f} GB, "
          f"MoE dense parts {92*PARAMS['moe_dense_per_layer']*2/1e9:.1f} GB, layer-0 MLP "
          f"{PARAMS['dense_mlp_layer0']*2/1e9:.2f} GB, embed+lm_head {2*PARAMS['embed']*2/1e9:.2f} GB)")
md.append(f"- MXFP4 routed experts: **{BYTES['mxfp4_experts_GB']:.0f} GB** "
          f"({PARAMS['experts_per_layer']*C['MXFP4_B']/1e9:.2f} GB/layer, 17.6 MB/expert incl. UE8M0 scales)")
md.append(f"- DSpark draft BF16: {BYTES['draft_bf16_GB']:.2f} GB (5 layers x {PARAMS['draft_per_layer']/1e6:.0f}M, "
          f"fc {PARAMS['draft_fc']/1e6:.0f}M, Markov {PARAMS['draft_markov']/1e6:.0f}M; embed/lm_head shared)")
kv = BYTES["kv_per_token_B"]
md.append(f"- KV per context token: MLA fp8_ds_mla {kv['mla_fp8_ds']} B + MXFP4 indexer {kv['indexer_mxfp4']} B = "
          f"**{kv['total_dsa']} B** (24 layers; 2.35 GB per 135K-token request); draft BF16 {kv['draft_bf16_window_only']} B "
          f"(window 2048 only -> 41.9 MB/request)")
st = BYTES["kda_state_per_request_B"]
md.append(f"- KDA state per request: recurrent FP32 **{st['recurrent_fp32']/1e6:.0f} MB** + conv FP32 "
          f"{st['conv_fp32_k0']/1e6:.1f} MB (k=0; +k columns with spec) -- constant in L\n")

md.append("## Op table\n")
md.append("Symbols: " + "; ".join(f"`{k}` {v}" for k, v in SYMBOLS.items()) + "\n")
md.append("| id | op | count | shapes | dtype in/w/out (compute) | FLOPs | weight B | KV/state B | bound P / D | cite |")
md.append("|---|---|---|---|---|---|---|---|---|---|")
for o in ops:
    d = o["dtype"]
    kvst = " ; ".join(x for x in (
        f"kvR {o['kv_read_bytes']}" if o["kv_read_bytes"] != "0" else "",
        f"kvW {o['kv_write_bytes']}" if o["kv_write_bytes"] != "0" else "",
        f"stR {o['state_read_bytes']}" if o["state_read_bytes"] != "0" else "",
        f"stW {o['state_write_bytes']}" if o["state_write_bytes"] != "0" else "") if x) or "-"
    cite = "<br>".join(c.replace(R, "nv/") for c in o["cite"])
    md.append(f"| {o['id']} | {o['name']} | {o['count_per_forward']} | {o['shapes']} | "
              f"{d['act_in']}/{d['weight']}/{d['out']} ({d['compute']}) | `{o['flops']}` | "
              f"`{o['weight_bytes']}` | `{kvst}` | {o['bound_prefill_ref']} / {o['bound_decode_ref']} | {cite} |")
md.append("\n`nv/` = `vllm/models/kimi_k3/nvidia/`. Bound column = classification at the reference "
          "points below (prefill C16384/P120000; decode B64 k3 L135K), unsharded, AI vs ridge "
          "(BF16 259, FP8-rate 755, FP4 1259, FP32 SIMT 9.5 FLOP/B at 13.9 TB/s). Per-op notes "
          "(derivations, alternatives, new methods) are in ops.json.\n")

md.append("## Reference evaluations (whole forward, unsharded, one engine)\n")
for name, rr in evals.items():
    agg = by_type(rr)
    md.append(f"### {name}\n")
    md.append("| layer type | TFLOP | weight GB | act GB | KV GB | state GB | comm-op bytes GB |")
    md.append("|---|---|---|---|---|---|---|")
    tot = dict(flops=0, w=0, a=0, kv=0, st=0, comm=0)
    for t, a in agg.items():
        md.append(f"| {t} | {a['flops']/1e12:.2f} | {a['w']/1e9:.2f} | {a['a']/1e9:.2f} | "
                  f"{a['kv']/1e9:.2f} | {a['st']/1e9:.2f} | {a['comm']/1e9:.2f} |")
        for kk_ in tot:
            tot[kk_] += a[kk_]
    md.append(f"| **total** | {tot['flops']/1e12:.1f} | {tot['w']/1e9:.1f} | {tot['a']/1e9:.1f} | "
              f"{tot['kv']/1e9:.1f} | {tot['st']/1e9:.1f} | {tot['comm']/1e9:.1f} |\n")
    def rt(r):
        hb = r["weight_bytes"] + r["act_bytes"] + r["kv_bytes"] + r["state_bytes"]
        return 0.0 if r["bound"] == "comm" else max(r["flops"] / (r["ridge"] * 13.9e12), hb / 13.9e12)
    top = sorted(rr, key=lambda r: -rt(r))[:10]
    tsum = sum(rt(r) for r in rr)
    md.append(f"Top ops by unsharded single-GPU roofline time max(FLOPs/peak(dtype), HBM bytes/13.9 TB/s); "
              f"sum over all non-comm ops = {tsum*1e3:.1f} ms (one GPU, no overlap, SOL):\n")
    md.append("| op | TFLOP | HBM GB | AI | ridge | bound | roofline ms | share |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in top:
        hb = r["weight_bytes"] + r["act_bytes"] + r["kv_bytes"] + r["state_bytes"]
        md.append(f"| {r['id']} | {r['flops']/1e12:.2f} | {hb/1e9:.2f} | {r['ai']:.0f} | {r['ridge']:.0f} | "
                  f"{r['bound']} | {rt(r)*1e3:.2f} | {rt(r)/tsum*100:.1f}% |")
    md.append("")

md.append("## Decode: marginal bytes/FLOPs per extra request (non-weight, unsharded, L=135K)\n")
md.append("| op | " + " | ".join(f"{k_} MB" for k_ in marginal) + " | " + " | ".join(f"{k_} GFLOP" for k_ in marginal) + " |")
md.append("|---|" + "---|" * (2 * len(marginal)))
ids_ = sorted({i for m_ in marginal.values() for i in m_}, key=lambda i: -marginal["k3_L135K"].get(i, {"bytes": 0})["bytes"])
for i in ids_:
    md.append(f"| {i} | " + " | ".join(f"{marginal[k_].get(i, {'bytes':0})['bytes']/1e6:.1f}" for k_ in marginal) + " | "
              + " | ".join(f"{marginal[k_].get(i, {'flops':0})['flops']/1e9:.2f}" for k_ in marginal) + " |")
md.append("| **total** | " + " | ".join(f"{sum(v['bytes'] for v in marginal[k_].values())/1e6:.0f}" for k_ in marginal) + " | "
          + " | ".join(f"{sum(v['flops'] for v in marginal[k_].values())/1e9:.1f}" for k_ in marginal) + " |\n")
md.append(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops_findings.md")).read())
with open(os.path.join(OUT, "ops.md"), "w") as fh:
    fh.write("\n".join(md))
print("md written")
