"""Sharding strategies (proposals.md P1-P9, D1-D11) as data + the generic per-op sharding rules and per-layer
communication lists.

A topology is a set of modes:
  N         attention TP group (KDA head-TP N; MLA heads / DCP / DSA-DP inside the same N group); N=1 = DEP
  residual  'tp'  replicated residual, AR after row-parallel GEMMs
            'sp'  sequence-parallel residual (AG before attention, RS after o_proj)
            'dep' no TP (N = 1)
  dsa       'rep' replicated DSA KV + indexer + qkv_a, sparse-MLA heads padded to 64
            'dcp' DCP W=N (decode: q publish + exact top-k merge + proj combine; prefill: ShardQ over gathered ctx)
            'dsadp' DSA-DP: request owned by one card, all 96 heads there, KV once per group (PR #20/#23)
            'local' (DEP)        'dsacp' DEP everywhere, DCP over G GPUs only on the 24 DSA layers
  moe_dense 'rep' SP-shard rows with replicated weights | 'tp' TP-N sharded (AG/RS or AR) |
            'tokenshard' (P/D TP8: router/latent on T/N rows, shared TP + one AR) | 'tpfront' (sdec TP16 MoE-TP)
  experts   'ep' MegaMoE EP over E GPUs | 'tp' each GPU holds 1/N of every expert (D1)
  drafter   TP over min(N, 16) (kv heads 16) or local
  overlays  O1 two-microbatch overlap, O2 dispatch dedup + pre-combine, O3 die-local streaming, O4 NVFP4 MoE act,
            O7 op-level SOL fusions (always on: fused indexer select, rows-consumed-only final mix / drafter ctx)
"""
from model import C

ROW = C["H"] * 2          # BF16 hidden row, 14,336 B
X_TOPK = 1536             # DcpExchange bytes per row per peer (inferred, comm.json open question)
VERIFY_B = 100            # sharded verify bytes per row per peer (P16)


class Topo:
    def __init__(self, id, name, phase, N, residual, dsa, moe_dense, E, experts="ep", G=None, S=1, dedup=False,
                 existing=False, proposal="", mapping=None, comm="", why="", risks="", afd=None, gpus=None):
        self.id, self.name, self.phase = id, name, phase
        self.N, self.residual, self.dsa, self.moe_dense = N, residual, dsa, moe_dense
        self.E, self.experts, self.G, self.S = E, experts, G, S
        self.dedup = dedup
        self.existing, self.proposal = existing, proposal
        self.mapping = mapping or {}
        self.comm, self.why, self.risks = comm, why, risks
        self.afd = afd
        self.ND = min(N, 16)
        self.gpus = gpus or max(E, N) * S

    def as_dict(self):
        return dict(id=self.id, name=self.name, phase=self.phase, N=self.N, residual=self.residual, dsa=self.dsa,
                    moe_dense=self.moe_dense, E=self.E, experts=self.experts, G=self.G, S=self.S, dedup=self.dedup,
                    existing=self.existing, proposal=self.proposal, mapping=self.mapping, comm=self.comm,
                    why=self.why, risks=self.risks, gpus=self.gpus, afd=self.afd)


def M(**kw):
    return kw


# ---------------------------------------------------------------------------------------------- prefill set
PREFILL = [
    Topo("P1", "P1 TP8+DCP8 ShardQ, EP8 token-shard", "prefill", 8, "tp", "dcp", "tokenshard", 8, existing=True,
         proposal="P1",
         mapping=M(kda="head-TP8 (f_a rep)", mla="qkv_a rep-compute; q_b/g/W_UK/W_UV on ShardQ slice (all heads); o_proj row-TP8",
                   indexer="wq rep; own ShardQ rows vs gathered ctx", dsa_attn="DCP8 KV (t mod 8), ShardQ",
                   moe_dense="router/latent on T/8 rows (rep W); shared TP8 + one AR", experts="EP8", lm_head="vocab/8", drafter="TP8"),
         comm="AR [T,h] after every attention and MoE layer; ShardQ q scatter / out gather + CE context push; EP8"),
    Topo("P2", "P2 DP4x(TP4=DCP4) SP ShardQ, EP16", "prefill", 4, "sp", "dcp", "rep", 16, existing=True, proposal="P2",
         mapping=M(kda="SP AG -> head-TP4 -> RS", mla="ShardQ as P1, W=4", indexer="own ShardQ rows vs gathered ctx",
                   dsa_attn="DCP4 KV", moe_dense="rep W on SP shard (no TP comm)", experts="EP16", lm_head="vocab/4", drafter="TP4, SP-local aux"),
         comm="SP AG+RS per attention layer; ShardQ + context push; EP16"),
    Topo("P2b", "P2 DP2x(TP8=DCP8) SP ShardQ, EP16", "prefill", 8, "sp", "dcp", "rep", 16, existing=True, proposal="P2",
         mapping=M(kda="SP AG -> head-TP8 -> RS", mla="ShardQ W=8", indexer="ShardQ", dsa_attn="DCP8 KV",
                   moe_dense="rep W on SP shard", experts="EP16", lm_head="vocab/8", drafter="TP8"),
         comm="SP AG+RS (N=8); ShardQ; EP16"),
    Topo("P3-16", "P3 DEP16", "prefill", 1, "dep", "local", "rep", 16, existing=True, proposal="P3",
         mapping=M(kda="local", mla="local, all heads", indexer="local", dsa_attn="local", moe_dense="local (rep W)",
                   experts="EP16", lm_head="full vocab", drafter="local"), comm="EP only"),
    Topo("P3-32", "P3 DEP32", "prefill", 1, "dep", "local", "rep", 32, existing=True, proposal="P3",
         mapping=M(kda="local", mla="local", indexer="local", dsa_attn="local", moe_dense="local", experts="EP32",
                   lm_head="full", drafter="local"), comm="EP only"),
    Topo("P3-72", "P3 DEP72", "prefill", 1, "dep", "local", "rep", 72, proposal="P3",
         mapping=M(kda="local", mla="local", indexer="local", dsa_attn="local", moe_dense="local", experts="EP72 (rack): ceil(896/72) = 13 expert slots per GPU, 40 redundant replicas",
                   lm_head="full", drafter="local"), comm="EP only (rack-wide)"),
    Topo("P4-4", "P4 TP4xDP4 DSA-DP SP, EP16", "prefill", 4, "sp", "dsadp", "rep", 16, existing=True, proposal="P4",
         mapping=M(kda="SP AG -> head-TP4 -> RS", mla="DSA-DP owner, all heads, W rep", indexer="owner",
                   dsa_attn="owner region (KV once per group)", moe_dense="rep W on SP shard", experts="EP16",
                   lm_head="vocab/4", drafter="TP4 SP-local aux"), comm="KDA AG+RS; DSA push_in/push_out; EP16"),
    Topo("P4-2", "P4 TP2xDP8 DSA-DP SP, EP16", "prefill", 2, "sp", "dsadp", "rep", 16, existing=True, proposal="P4",
         mapping=M(kda="SP AG -> head-TP2 -> RS", mla="DSA-DP owner", indexer="owner", dsa_attn="owner region",
                   moe_dense="rep W on SP shard", experts="EP16", lm_head="vocab/2", drafter="TP2"), comm="KDA AG+RS (N=2); push; EP16"),
    Topo("P5-4", "P5 TP4 rep-DSA SP, EP16", "prefill", 4, "sp", "rep", "rep", 16, existing=True, proposal="P5",
         mapping=M(kda="head-TP4", mla="qkv_a rep, q_b/W_UK/W_UV head-TP4, heads padded 24->64", indexer="rep (every rank all rows)",
                   dsa_attn="KV replicated x4", moe_dense="rep W on SP shard", experts="EP16", lm_head="vocab/4", drafter="TP4"),
         comm="SP AG+RS; EP16"),
    Topo("P5-8", "P5 TP8 rep-DSA token-shard, EP8", "prefill", 8, "tp", "rep", "tokenshard", 8, existing=True, proposal="P5",
         mapping=M(kda="head-TP8", mla="rep qkv_a, heads padded 12->64", indexer="rep", dsa_attn="KV replicated x8",
                   moe_dense="token shard + shared TP8", experts="EP8", lm_head="vocab/8", drafter="TP8"),
         comm="AR per layer; EP8"),
    Topo("P6", "P6 LTP: TP2 KDA + DSA-DP, EP64 + dedup", "prefill", 2, "sp", "dsadp", "rep", 64, dedup=True, proposal="P6",
         mapping=M(kda="head-TP2 with SP", mla="DSA-DP over the pair", indexer="owner (split indexer optional)",
                   dsa_attn="owner, KV once", moe_dense="SP-local rep W", experts="EP64 rack-wide, O2 dedup + pre-combine",
                   lm_head="vocab/2", drafter="TP2, ctx K/V last 2048 rows only"),
         comm="69 x AG+RS (N=2) + 24 x push + 92 x deduped EP"),
    Topo("P6-16", "P6 LTP: TP2 KDA + DSA-DP, EP16 + dedup", "prefill", 2, "sp", "dsadp", "rep", 16, dedup=True,
         proposal="P6",
         mapping=M(kda="head-TP2 with SP", mla="DSA-DP over the pair", indexer="owner", dsa_attn="owner, KV once",
                   moe_dense="SP-local rep W", experts="EP16 (8 pairs), O2 dedup + pre-combine", lm_head="vocab/2",
                   drafter="TP2, ctx K/V last 2048 rows only"),
         comm="69 x AG+RS (N=2) + 24 x push + 92 x deduped EP16"),
    Topo("P6-4-16", "P6 LTP: TP4 KDA + DSA-DP, EP16 + dedup", "prefill", 4, "sp", "dsadp", "rep", 16, dedup=True,
         proposal="P6",
         mapping=M(kda="head-TP4 with SP", mla="DSA-DP over 4", indexer="owner", dsa_attn="owner, KV once",
                   moe_dense="SP-local rep W", experts="EP16 (4 groups), O2 dedup + pre-combine", lm_head="vocab/4",
                   drafter="TP4"),
         comm="69 x AG+RS (N=4) + 24 x push + 92 x deduped EP16"),
    Topo("P6-32", "P6 LTP: TP2 KDA + DSA-DP, EP32 + dedup", "prefill", 2, "sp", "dsadp", "rep", 32, dedup=True,
         proposal="P6",
         mapping=M(kda="head-TP2 with SP", mla="DSA-DP over the pair", indexer="owner", dsa_attn="owner, KV once",
                   moe_dense="SP-local rep W", experts="EP32 (16 pairs), O2 dedup + pre-combine", lm_head="vocab/2",
                   drafter="TP2, ctx K/V last 2048 rows only"),
         comm="69 x AG+RS (N=2) + 24 x push + 92 x deduped EP32"),
    Topo("P9-32", "P9 DSA-CP: DEP32 + ShardQ DCP16 on DSA layers", "prefill", 1, "dep", "dsacp", "rep", 32, G=16,
         proposal="P9",
         mapping=M(kda="local", mla="local rows, KV t mod 16 over 16 GPUs", indexer="local rows vs gathered ctx",
                   dsa_attn="DCP16 KV, context gathered per DSA layer", moe_dense="local", experts="EP32",
                   lm_head="full", drafter="local"),
         comm="EP32 + per-DSA-layer context gather from 16 GPUs"),
    Topo("P9-16", "P9 DSA-CP: DEP16 + ShardQ DCP16 on DSA layers", "prefill", 1, "dep", "dsacp", "rep", 16, G=16,
         proposal="P9",
         mapping=M(kda="local", mla="local rows, KV t mod 16 over 16 GPUs", indexer="local rows vs gathered ctx",
                   dsa_attn="DCP16 KV, context gathered per DSA layer", moe_dense="local", experts="EP16",
                   lm_head="full", drafter="local"),
         comm="EP16 + per-DSA-layer context gather from 16 GPUs"),
    Topo("P7", "P7 PP4 x DEP16 (stage = ~23 layers)", "prefill", 1, "dep", "local", "rep", 16, S=4, proposal="P7",
         mapping=M(kda="local (stage layers)", mla="local", indexer="local", dsa_attn="local, stage layers only",
                   moe_dense="local", experts="EP16 per stage (each GPU 1/64 of experts)", lm_head="last stage", drafter="last stage"),
         comm="EP16 inside stage + residual/AttnRes boundary push (<1% of EP)", gpus=64),
    Topo("P9", "P9 DSA-CP: DEP16 + ShardQ DCP8 on DSA layers", "prefill", 1, "dep", "dsacp", "rep", 16, G=8, proposal="P9",
         mapping=M(kda="local", mla="local rows, KV t mod 8 over 8 GPUs", indexer="local rows vs gathered ctx",
                   dsa_attn="DCP8 KV, context gathered per DSA layer", moe_dense="local", experts="EP16", lm_head="full", drafter="local"),
         comm="EP + per-DSA-layer context gather from 8 GPUs"),
]

# ---------------------------------------------------------------------------------------------- decode set
DECODE = [
    Topo("D1", "D1 TP16+DCP16 sdec, MoE-TP16", "decode", 16, "tp", "dcp", "tpfront", 1, experts="tp", existing=True,
         proposal="D1", gpus=16,
         mapping=M(kda="head-TP16 + AR", mla="qkv_a + wq rep; q_b/g col; W_UV all heads rep", indexer="own C/16 keys + exact DcpExchange merge",
                   dsa_attn="DCP16 KV, FP8 q publish, proj value combine", moe_dense="TP16 front (P14) / tail (P15)",
                   experts="TP16 (1/16 of every expert)", lm_head="vocab/16 sharded verify", drafter="TP16 (11 AR sites)"),
         comm="AR per attention layer + DCP16 + TP-MoE front/tail"),
    Topo("D2", "D2 TP8+DCP8, EP8 token-shard", "decode", 8, "tp", "dcp", "tokenshard", 8, existing=True, proposal="D2",
         mapping=M(kda="head-TP8", mla="as D1 (N=8)", indexer="DCP8 + merge", dsa_attn="DCP8", moe_dense="token shard, shared TP8 + AR",
                   experts="EP8", lm_head="vocab/8", drafter="TP8"), comm="AR + DCP8 + EP8"),
    Topo("D3", "D3 DP4x(TP4=DCP4) SP, EP16", "decode", 4, "sp", "dcp", "rep", 16, existing=True, proposal="D3",
         mapping=M(kda="SP multicast AG -> head-TP4 -> push RS", mla="DCP4", indexer="DCP4 + merge", dsa_attn="DCP4",
                   moe_dense="rep W on SP shard", experts="EP16", lm_head="vocab/4", drafter="TP4"), comm="SP AG+RS + DCP4 + EP16"),
    Topo("D4-16", "D4 DEP16", "decode", 1, "dep", "local", "rep", 16, existing=True, proposal="D4",
         mapping=M(kda="local", mla="local", indexer="local", dsa_attn="local", moe_dense="local", experts="EP16", lm_head="full", drafter="local"),
         comm="EP only"),
    Topo("D4-32", "D4 DEP32", "decode", 1, "dep", "local", "rep", 32, existing=True, proposal="D4",
         mapping=M(kda="local", mla="local", indexer="local", dsa_attn="local", moe_dense="local", experts="EP32", lm_head="full", drafter="local"),
         comm="EP only"),
    Topo("D4-72", "D4 DEP72", "decode", 1, "dep", "local", "rep", 72, proposal="D4",
         mapping=M(kda="local", mla="local", indexer="local", dsa_attn="local", moe_dense="local", experts="EP72: 13 expert slots per GPU (40 redundant replicas)", lm_head="full", drafter="local"),
         comm="EP only (rack)"),
    Topo("D5", "D5 TP4 DSA-DP SP, EP16", "decode", 4, "sp", "dsadp", "rep", 16, existing=True, proposal="D5",
         mapping=M(kda="SP + head-TP4", mla="DSA-DP owner, all heads", indexer="owner", dsa_attn="owner region",
                   moe_dense="rep W on SP shard", experts="EP16", lm_head="vocab/4", drafter="TP4"), comm="KDA AG+RS + DSA push + EP16"),
    Topo("D6", "D6 TP4 rep-DSA SP, EP16", "decode", 4, "sp", "rep", "rep", 16, existing=True, proposal="D6",
         mapping=M(kda="head-TP4", mla="rep qkv_a, heads padded 24->64", indexer="rep (full ctx on every rank)",
                   dsa_attn="KV replicated x4", moe_dense="rep W on SP shard", experts="EP16", lm_head="vocab/4", drafter="TP4"),
         comm="SP AG+RS + EP16"),
    Topo("D7-4", "D7 TP4 DSA-DP, TP-dense, EP64", "decode", 4, "sp", "dsadp", "tp", 64, proposal="D7",
         mapping=M(kda="head-TP4 + SP", mla="DSA-DP over 4", indexer="owner", dsa_attn="owner, KV once",
                   moe_dense="router/latent/shared TP4 (AG+RS)", experts="EP64 rack-wide", lm_head="vocab/4", drafter="TP4"),
         comm="KDA AG+RS + DSA push + MoE-dense AG/RS + EP64"),
    Topo("D7-2", "D7 TP2 DSA-DP, TP-dense, EP64", "decode", 2, "sp", "dsadp", "tp", 64, proposal="D7",
         mapping=M(kda="head-TP2", mla="DSA-DP over 2", indexer="owner", dsa_attn="owner", moe_dense="TP2", experts="EP64",
                   lm_head="vocab/2", drafter="TP2"), comm="as D7-4 with N=2"),
    Topo("D7-8", "D7 TP8 DSA-DP, TP-dense, EP64", "decode", 8, "sp", "dsadp", "tp", 64, proposal="D7",
         mapping=M(kda="head-TP8", mla="DSA-DP over 8", indexer="owner", dsa_attn="owner", moe_dense="TP8", experts="EP64",
                   lm_head="vocab/8", drafter="TP8"), comm="as D7-4 with N=8"),
    Topo("D8", "D8 TP16+DCP16, TP-dense, EP64", "decode", 16, "tp", "dcp", "tp", 64, proposal="D8",
         mapping=M(kda="head-TP16 + AR", mla="as D1", indexer="DCP16 + merge", dsa_attn="DCP16", moe_dense="TP16 (front AG + AR)",
                   experts="EP64 (4 attention groups)", lm_head="vocab/16", drafter="TP16"), comm="AR + DCP16 + front AG + EP64"),
    # ---- EP <= 16 re-scope (2026-11-18): same layer-typed layouts with the EP domain capped at 16 GPUs (4 nodes)
    Topo("D7-4-16", "D7 TP4 DSA-DP, TP-dense, EP16", "decode", 4, "sp", "dsadp", "tp", 16, proposal="D7",
         mapping=M(kda="head-TP4 + SP", mla="DSA-DP over 4", indexer="owner", dsa_attn="owner, KV once",
                   moe_dense="router/latent/shared TP4 (AG+RS)", experts="EP16 (4 attention groups, 56 experts/GPU)",
                   lm_head="vocab/4", drafter="TP4"),
         comm="KDA AG+RS + DSA push + MoE-dense AG/RS + EP16"),
    Topo("D7-2-16", "D7 TP2 DSA-DP, TP-dense, EP16", "decode", 2, "sp", "dsadp", "tp", 16, proposal="D7",
         mapping=M(kda="head-TP2", mla="DSA-DP over 2", indexer="owner", dsa_attn="owner", moe_dense="TP2",
                   experts="EP16 (8 attention groups)", lm_head="vocab/2", drafter="TP2"), comm="as D7-4-16 with N=2"),
    Topo("D7-8-16", "D7 TP8 DSA-DP, TP-dense, EP16", "decode", 8, "sp", "dsadp", "tp", 16, proposal="D7",
         mapping=M(kda="head-TP8", mla="DSA-DP over 8", indexer="owner", dsa_attn="owner", moe_dense="TP8",
                   experts="EP16 (2 attention groups)", lm_head="vocab/8", drafter="TP8"), comm="as D7-4-16 with N=8"),
    Topo("D8-16", "D8 TP16+DCP16, TP-dense, EP16", "decode", 16, "tp", "dcp", "tp", 16, proposal="D8",
         mapping=M(kda="head-TP16 + AR", mla="as D1", indexer="DCP16 + merge", dsa_attn="DCP16",
                   moe_dense="TP16 (router/latent transposes + AR)", experts="EP16 (same 16 GPUs, 56 experts/GPU)",
                   lm_head="vocab/16", drafter="TP16"), comm="AR + DCP16 + MoE-dense transposes + EP16"),
    Topo("D9-16", "D9 DSA-CP: DEP16 + DCP16 on DSA", "decode", 1, "dep", "dsacp", "rep", 16, G=16, proposal="D9",
         mapping=M(kda="local", mla="local rows", indexer="own key shard of 16 GPUs' requests + merge",
                   dsa_attn="DCP16 KV", moe_dense="local", experts="EP16", lm_head="full", drafter="local"),
         comm="EP16 + DCP16 on 24 DSA layers"),
    # ---- <= 32-GPU re-scope (2026-11-18): EP32 variants (8 VR200 nodes per EP domain)
    Topo("D3-32", "D3 DP8x(TP4=DCP4) SP, EP32", "decode", 4, "sp", "dcp", "rep", 32, proposal="D3",
         mapping=M(kda="SP multicast AG -> head-TP4 -> push RS", mla="DCP4", indexer="DCP4 + merge", dsa_attn="DCP4",
                   moe_dense="rep W on SP shard", experts="EP32 (8 DP groups, 28 experts/GPU)", lm_head="vocab/4",
                   drafter="TP4"), comm="SP AG+RS + DCP4 + EP32"),
    Topo("D7-4-32", "D7 TP4 DSA-DP, TP-dense, EP32", "decode", 4, "sp", "dsadp", "tp", 32, proposal="D7",
         mapping=M(kda="head-TP4 + SP", mla="DSA-DP over 4", indexer="owner", dsa_attn="owner, KV once",
                   moe_dense="router/latent/shared TP4 (AG+RS)", experts="EP32 (8 attention groups, 28 experts/GPU)",
                   lm_head="vocab/4", drafter="TP4"), comm="KDA AG+RS + DSA push + MoE-dense AG/RS + EP32"),
    Topo("D7-2-32", "D7 TP2 DSA-DP, TP-dense, EP32", "decode", 2, "sp", "dsadp", "tp", 32, proposal="D7",
         mapping=M(kda="head-TP2", mla="DSA-DP over 2", indexer="owner", dsa_attn="owner", moe_dense="TP2",
                   experts="EP32 (16 attention groups)", lm_head="vocab/2", drafter="TP2"), comm="as D7-4-32 with N=2"),
    Topo("D7-8-32", "D7 TP8 DSA-DP, TP-dense, EP32", "decode", 8, "sp", "dsadp", "tp", 32, proposal="D7",
         mapping=M(kda="head-TP8", mla="DSA-DP over 8", indexer="owner", dsa_attn="owner", moe_dense="TP8",
                   experts="EP32 (4 attention groups)", lm_head="vocab/8", drafter="TP8"), comm="as D7-4-32 with N=8"),
    Topo("D8-32", "D8 TP16+DCP16, TP-dense, EP32", "decode", 16, "tp", "dcp", "tp", 32, proposal="D8",
         mapping=M(kda="head-TP16 + AR", mla="as D1", indexer="DCP16 + merge", dsa_attn="DCP16",
                   moe_dense="TP16 (router/latent transposes + AR)", experts="EP32 (2 attention groups, 28 experts/GPU)",
                   lm_head="vocab/16", drafter="TP16"), comm="AR + DCP16 + MoE-dense transposes + EP32"),
    Topo("D9-32", "D9 DSA-CP: DEP32 + DCP16 on DSA", "decode", 1, "dep", "dsacp", "rep", 32, G=16, proposal="D9",
         mapping=M(kda="local", mla="local rows", indexer="own key shard of 16 GPUs' requests + merge",
                   dsa_attn="DCP16 KV", moe_dense="local", experts="EP32", lm_head="full", drafter="local"),
         comm="EP32 + DCP16 on 24 DSA layers"),
    Topo("D9", "D9 DSA-CP: DEP64 + DCP16 on DSA", "decode", 1, "dep", "dsacp", "rep", 64, G=16, proposal="D9",
         mapping=M(kda="local", mla="local rows", indexer="own key shard of 16 GPUs' requests + merge", dsa_attn="DCP16 KV",
                   moe_dense="local", experts="EP64", lm_head="full", drafter="local"), comm="EP + DCP16 on 24 DSA layers"),
    Topo("D10", "D10 D7-4 + split indexer", "decode", 4, "sp", "dsadp_split", "tp", 64, proposal="D10",
         mapping=M(kda="head-TP4", mla="DSA-DP owner", indexer="owner keys split over 4 + exact merge", dsa_attn="owner",
                   moe_dense="TP4", experts="EP64", lm_head="vocab/4", drafter="TP4"), comm="D7-4 + indexer q broadcast + merge"),
    Topo("D11", "D11 AFD: A-pool TP4 (as D7) + F-pool EP", "decode", 4, "sp", "dsadp", "tp", 24, proposal="D11",
         afd=dict(choices=[(56, 16), (48, 24), (40, 32)]), gpus=72,
         mapping=M(kda="A-pool head-TP4", mla="A-pool DSA-DP", indexer="A-pool", dsa_attn="A-pool", moe_dense="A-pool TP4",
                   experts="F-pool EP_F (no KV)", lm_head="A-pool", drafter="A-pool TP4"),
         comm="A->F latent rows, F->A BF16 rows per MoE layer, 2 microbatches ping-pong"),
]

ALL = {t.id: t for t in PREFILL + DECODE}

EP_MAX = 32      # 2026-11-18 scope (user rule, replaces EP <= 16): at most 32 GPUs (8 VR200 nodes) per instance


def in_scope(t, max_gpus=EP_MAX):
    """True when one instance (engine / EP domain) spans <= 32 GPUs: EP <= 32 and gpus <= 32. MoE-TP D1 has no EP
    domain; AFD (72 GPUs), PP4 (64), EP64/72 exceed it."""
    if t.afd or t.gpus > max_gpus:
        return False
    return t.experts == "tp" or t.E <= max_gpus


SCOPE = [t.id for t in PREFILL + DECODE if in_scope(t)]


# ---------------------------------------------------------------------------------------------- per-op sharding
def _mix(N, frac_sharded):
    """Divisor when a fraction of the op is split N ways and the rest replicated."""
    return 1.0 / (frac_sharded / N + (1.0 - frac_sharded))


QB_FRAC = (C["Nh"] * C["DQK"]) / (C["Nh"] * C["DQK"] + C["Ni"] * C["Di"])     # q_b share of the packed GEMM
INPROJ_FRAC = (4 * C["P"] + C["Hk"]) / (4 * C["P"] + C["dk"] + C["Hk"])          # f_a (128 cols) replicated


def op_map(op_id, t, phase):
    """Per-GPU divisors of the FULL op for topology t: dict(row, w, kv, st, fl).

    row: divisor for FLOPs and activation bytes, w: weight bytes, kv: KV bytes, st: state bytes,
    fl: extra FLOP multiplier (head padding).  Experts are handled in cost.py (touched experts per GPU)."""
    N = t.N
    rres = N if t.residual == "sp" else 1           # token-parallel ops: replicated under TP residual
    d = dict(row=1.0, w=1.0, kv=1.0, st=1.0, fl=1.0)
    head = dict(row=N, w=N, kv=N, st=N, fl=1.0)
    rows = dict(row=N, w=1.0, kv=N, st=N, fl=1.0)
    rep = d
    i = op_id
    if N == 1 and t.dsa != "dsacp":
        return d
    if i == "G.embed":
        return head
    if i.startswith("L.attnres") and phase == "prefill" and t.residual == "tp":
        return dict(d, row=N)          # TP prefill (P1): AttnRes on the 1/N RS-owned token shard, then AG (comm_events)
    if i.startswith("L.attnres") or i == "F.attnres_final" or i == "DS.aux_capture":
        return dict(d, row=rres)
    if i.startswith("KDA."):
        r = dict(head)
        if i == "KDA.in_proj":
            r["row"] = r["w"] = _mix(N, INPROJ_FRAC)
        return r
    if i.startswith("DSA."):
        return _dsa_map(i, t, phase)
    if i.startswith("MOE.") or i.startswith("DENSE."):
        md = t.moe_dense
        if i in ("MOE.dispatch", "MOE.fc1", "MOE.act", "MOE.fc2", "MOE.combine"):
            return dict(d, row=N)
        if md == "rep":
            return dict(d, row=N)
        if md in ("tp", "tpfront"):
            return head
        if md == "tokenshard":
            if i.startswith("MOE.shared") or i.startswith("DENSE."):
                return head
            return dict(d, row=N)
    if i in ("F.lm_head", "F.rejection"):
        return head
    if i == "F.norm":
        return dict(d, row=rres)
    if i.startswith("DS."):
        ND = t.ND
        if i in ("DS.fc", "DS.hidden_norm") and phase == "prefill":
            return dict(d, row=ND)
        if i in ("DS.norm_ar", "DS.confidence"):
            return rep
        return dict(row=ND, w=ND, kv=ND, st=ND, fl=1.0)
    return d


def _dsa_map(i, t, phase):
    N, mode = t.N, t.dsa
    d = dict(row=1.0, w=1.0, kv=1.0, st=1.0, fl=1.0)
    head = dict(row=N, w=N, kv=1.0, st=1.0, fl=1.0)
    owner = dict(row=N, w=1.0, kv=N, st=1.0, fl=1.0)
    if mode == "local":
        return d
    if mode == "dsacp":
        # rows are local (DEP); keys of G GPUs' requests are spread t mod G -> per-GPU KV traffic unchanged on
        # average; the prologue writes 1/G of the new rows locally.
        if i == "DSA.prologue":
            return dict(d)
        return d
    if mode in ("dsadp", "dsadp_split"):
        return owner
    shardq = (mode == "dcp" and phase == "prefill")
    if i in ("DSA.qkv_a_packed", "DSA.idx_q_quant"):
        return d                                                  # replicated compute, all rows
    if i == "DSA.prologue":
        return dict(d, kv=(N if mode == "dcp" else 1.0))          # KV written once under DCP
    if i == "DSA.qb_wq_packed":
        if shardq:
            return dict(d, row=_mix(N, QB_FRAC))                  # q_b on the ShardQ slice, all-head weights
        return dict(d, row=_mix(N, QB_FRAC), w=_mix(N, QB_FRAC))
    if i in ("DSA.g_proj", "DSA.w_uk"):
        return dict(d, row=N) if shardq else head
    if i in ("DSA.idx_score", "DSA.topk"):
        if mode == "rep":
            return d
        if shardq:
            return dict(d, row=N)                                 # own rows vs the gathered context (KV read local)
        return dict(d, row=N, kv=N)                               # DCP decode: own C/N keys for all rows
    if i == "DSA.sparse_mla":
        if mode == "rep":
            pad = max(C["Nh"] / N, 64.0) / C["Nh"]                 # local heads padded to the 64-head tile
            return dict(d, row=N, fl=pad * N)
        if shardq:
            return dict(d, row=N)
        return dict(d, row=N, kv=N)
    if i == "DSA.w_uv":
        if mode == "dcp" and phase == "decode":
            return d                                              # sdec 'proj': W_UV on all 96 heads on every rank
        return dict(d, row=N) if shardq else head
    if i in ("DSA.gate_mul", "DSA.o_proj"):
        return head
    return d


# ---------------------------------------------------------------------------------------------- communication
def ev(prim, name, in_B, out_B, hops, async_=False, kind="push", N=2, M=0.0):
    """One comm event. kind: 'push' (symmetric-memory push / Lamport mailbox: hops x push hop + bytes / NVLink),
    'nvls_ar' / 'nvls_rs' (multimem: priced by hw.nvls_ar from the message size M), 'ce' (copy engine)."""
    return dict(prim=prim, name=name, in_B=float(in_B), out_B=float(out_B), hops=hops, async_=async_, kind=kind,
                N=N, M=float(M))


def AR(name, Mb, N):
    """Best available all-reduce of Mb bytes over N: NVLS multimem (P1), one-shot Lamport push (P1c, 1 hop) or
    two-shot push (P1b: RS-push + AG-push, 2 hops)."""
    return dict(alts=[ev("P1 AR-NVLS", name, Mb, Mb * (1 + 1.0 / N), 2, kind="nvls_ar", N=N, M=Mb),
                      ev("P1c AR-oneshot", name, (N - 1) * Mb, Mb, 1, N=N),
                      ev("P1b AR-twoshot", name, 2 * (N - 1) / N * Mb, 2 * (N - 1) / N * Mb, 2, N=N)])


def AG(name, n_rows, N, row=ROW):
    return ev("P2 SP-AG", name, (N - 1) * n_rows * row, (N - 1) * n_rows * row, 1, N=N)


def RS(name, T, N, row=ROW):
    return dict(alts=[ev("P3 RS-NVLS", name, T * row / N, T * row * (N - 1) / N, 2, kind="nvls_rs", N=N, M=T * row),
                      ev("P3b RS-push", name, (N - 1) * T / N * row, (N - 1) * T / N * row, 1, N=N)])


def XPOSE(name, T, N, row_B):
    """Column-shard -> row-shard transpose of a [T, row_B] tensor over N ranks (each keeps T/N full rows)."""
    b = T * (N - 1) / N * row_B / N
    return ev("P14 TP-MoE transpose", name, b, b, 1, N=N)


def ep_events(m, E, ck, dedup, imb=1.0):
    k16 = C["TOPK_E"]
    rem = (E - 1.0) / E
    f = 1.0
    if dedup:
        U = E * (1 - (1 - 1.0 / E) ** k16)          # destinations per token (rank-level dedup + pre-combine)
        f = U / k16
    a = ck["dispatch_row"]
    return [ev("P11 EP-dispatch", "MegaMoE dispatch", imb * m * k16 * a * rem * f, m * k16 * a * rem * f, 2, N=E),
            ev("P12 EP-combine", "MegaMoE combine", m * k16 * 7168 * rem * f, imb * m * k16 * 7168 * rem * f, 1, N=E)]


def tp_reduce(name, T, N, phase):
    """Row-parallel output reduction under a replicated (TP) residual. Decode: one all-reduce. Prefill (P1 code
    path, prefill lane): the o_proj GEMM reduce-scatters (GemmRsAr), AttnRes runs on the 1/N RS-owned token shard and
    a push all-gather rebuilds the rows (_push_shard_kernel) -> RS + AG instead of AR."""
    if phase == "prefill":
        return [RS(name + " RS", T, N), AG("AttnRes shard AG", T / N, N)]
    return [AR(name + " AR", T * ROW, N)]


def comm_events(t, phase, env, ck, imb=1.0, o2=False):
    """Per layer type -> list of comm events (bytes per GPU). Layer types: embed, kda, dsa, moe, dense, final,
    drafter, step."""
    N = t.N
    T = env["T"]
    n = T / N
    out = {k: [] for k in ("embed", "kda", "dsa", "moe", "dense", "final", "drafter", "step")}
    if N > 1:
        if t.residual == "tp":
            out["embed"].append(AR("embed vocab reduce", T * ROW, N))
            out["kda"] += tp_reduce("o_proj", T, N, phase)
            if t.dsa not in ("dsadp", "dsadp_split"):
                out["dsa"] += tp_reduce("o_proj", T, N, phase)
        else:
            out["embed"].append(RS("embed RS", T, N))
            out["kda"] += [AG("SP AG", n, N), RS("o_proj RS", T, N)]
            if t.dsa in ("dsadp", "dsadp_split"):
                # prefill: ~5.4K-token request pieces cannot align with SP shards -> (N-1)/N of the rows move.
                # decode: the scheduler gives each SP shard whole requests, so the SP holder is the owner (0 bytes,
                # only the hand-off remains).
                push = n * ROW * (N - 1) / N if phase == "prefill" else 0.0
                out["dsa"] += [ev("P13 DSA-DP push_in", "push_in rows to owner", push, push, 1, N=N),
                               ev("P13 DSA-DP push_out", "push_out owner rows", push, push, 1, N=N)]
            else:
                out["dsa"] += [AG("SP AG", n, N), RS("o_proj RS", T, N)]
    # DSA attention comm
    W = t.G if t.dsa == "dsacp" else N
    Tw = T * W if t.dsa == "dsacp" else T        # rows sharing the DCP group
    if t.dsa in ("dcp", "dsacp") and W > 1:
        if phase == "decode":
            hq = C["Nh"] / W
            out["dsa"] += [ev("P6 DCP q publish", "FP8 MQA q publish", Tw * (W - 1) * hq * 576, Tw * hq * 576, 1, N=W),
                           ev("P8 DCP top-k merge", "exact top-k merge", Tw * (W - 1) * X_TOPK, Tw * (W - 1) * X_TOPK, 2, N=W),
                           ev("P7b DCP value push", "proj value combine", Tw * (W - 1) * hq * 260, Tw * (W - 1) * hq * 260, 1, N=W)]
        else:
            ctx = env["Lu"] * (C["MLA_ROW"] + C["IDX_ROW"]) * (W - 1) / W
            out["dsa"].append(ev("P9 ShardQ ctx push (CE)", "context gather", ctx, ctx, 1, async_=True, N=W))
            if t.dsa == "dcp":
                # ShardQ variant used here (SOL, consistent with op_map): q_b / W_UK / g run slice-local with
                # all-head weights, so no P4 q scatter; o_proj stays head-TP, so the P5 out gather remains.
                o = T * (N - 1) / N * (C["Nh"] / N) * 128 * 2
                out["dsa"].append(ev("P5 ShardQ out gather", "out gather", o, o, 1, N=N))
    if t.dsa == "dsadp_split" and N > 1:
        rows_o = T / N
        b = rows_o * (N - 1) * (2200 + X_TOPK) + rows_o * 8192 * (N - 1) / N
        out["dsa"].append(ev("S1 split-indexer", "indexer q bcast + merge + ids", b, b, 2, N=N))
    # MoE layers
    md = t.moe_dense
    if md == "tpfront":
        out["moe"].append(ev("P14 TP-MoE front AG", "router logits + latent AG", T * (N - 1) / N * (896 * 4 + 3584 * 2),
                             T / N * (896 * 4 + 3584 * 2) * (N - 1), 1, N=N))
        out["moe"] += [AR("latent AR", T * 7168, N), AR("shared/out AR", T * ROW, N)]
    else:
        if N > 1:
            if md == "tp":
                # router / latent_down column-sharded -> transpose to row ownership for top-k + dispatch (T/N rows
                # per rank); after combine the rank owns T/N full latent rows -> latent_norm there, then transpose
                # back to column shards for the K-split latent_up, whose partial sums ride the shared-expert
                # RS / AR (no extra reduction, no replicated latent_up weights).
                front = XPOSE("router logits + latent col->row", T, N, 896 * 4 + 3584 * 2)
                back = XPOSE("combined latent row->col (K-split latent_up)", T, N, 3584 * 2)
                if t.residual == "sp":
                    out["moe"] += [AG("MoE-dense AG", n, N), front, back, RS("MoE-dense RS", T, N)]
                else:
                    out["moe"] += [front, back, AR("shared+latent_up AR", T * ROW, N)]
            elif md == "tokenshard":
                out["moe"] += tp_reduce("shared partial + routed", T, N, phase)
        if t.experts == "ep" and t.E > 1:
            out["moe"] += ep_events(T / N, t.E, ck, t.dedup or o2, imb)
    # dense MLP layer 0
    if N > 1 and md != "rep":
        if t.residual == "sp":
            out["dense"] += [AG("dense AG", n, N), RS("dense RS", T, N)]
        else:
            out["dense"] += tp_reduce("dense down", T, N, phase)
    # final + lm_head / verify
    R = env["R"]
    if N > 1:
        if t.residual == "sp":
            out["final"].append(AG("final AG (rows consumed)", R / N, N))
        out["final"].append(ev("P16 sharded verify", "vocab-sharded verify", R * (N - 1) * VERIFY_B, R * VERIFY_B, 2 if phase == "decode" else 1, N=N))
    # drafter
    ND = t.ND
    if ND > 1:
        Tc = env["Tc"]
        out["drafter"].append(ev("P2 drafter ctx AG", "fc/ctx rows AG", Tc * (ND - 1) / ND * ROW, Tc * (ND - 1) / ND * ROW, 1, N=ND))
        if phase == "decode" and t.residual == "sp":
            # SP residual: the 5 aux captures are row-sharded; the column-sharded fc needs all Tc rows
            aux = Tc * (ND - 1) / ND * C["D_AUX"] * ROW
            out["drafter"].append(ev("P18 drafter aux AG", "aux hidden rows AG (5 x 7168)", aux, aux, 1, N=ND))
        if phase == "decode":
            Tq = env["Tq"]
            for s in range(2 * C["D_L"] + 1):
                out["drafter"].append(AR("ArNorm site %d" % s, Tq * ROW, ND))
            for s in range(int(env["k"])):
                out["drafter"].append(ev("P16 Markov exchange", "Markov argmax pos %d" % s, env["B"] * (ND - 1) * 16, env["B"] * 16, 1, N=ND))
    if t.E > N and t.experts == "ep":
        out["step"].append(ev("P19 DP sync", "EP lock-step sync", 0, 0, 1, N=t.E))
    return out


def comm_row_event(row):
    """Model comm event for one comm-lane microbench row (microbench/results/comm/rows.json; kind -> size mapping per
    the comm lane README: AR/RS bytes = full tensor, AG bytes = shard, A2A bytes = per-peer chunk)."""
    k, N, b, h = row["kind"], row["N"], float(row["bytes_param"]), row["hops"]
    if k == "ar":
        return AR(row["primitive"], b, N)
    if k == "rs":
        return RS(row["primitive"], b / ROW, N)
    if k == "bar":
        return ev(row["primitive"], row["primitive"], 0, 0, max(h, 1), N=N)
    return ev(row["primitive"], row["primitive"], (N - 1) * b, (N - 1) * b, max(h, 1), N=N)
