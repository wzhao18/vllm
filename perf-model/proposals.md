# K3.1-Pro on VR200: sharding strategies to model (prefill and decode)

Status: proposal, 2026-11-17. Input to the SOL model (`k31vr/solmodel/`). This lists every strategy the model should
evaluate, for prefill and decode separately, under both serving checkpoints (`inputs/checkpoints.json`).
Each strategy gives:
- the mapping of every op family;
- the communication per layer, with sizes;
- what is replicated (weights, KV, state);
- expected strengths and weaknesses;
- the knobs the model sweeps.

The inputs are `inputs/ops.json` (op inventory), `inputs/comm.json` + `inputs/topologies.md` (primitives P1-P21 and
strategies S1-S13), `inputs/hw.json` and `inputs/workload.json`. Primitive ids (P#) and new-method ids (S#) refer to
`topologies.md`.

The numbers in the "screening" tables (§1.3, §4) are **first-cut estimates to rank candidates, not the SOL model**:
- They come from `screening/screen_decode.py` and `screening/screen_prefill.py`, which read `inputs/ops.json`.
- They use our own Hecate measurements: 13.9 TB/s HBM, 1.25 TB/s NVLink, 2.86 us per dependent hop, 0.78 us per
  graph node, and 3.65 / 10.9 / 21.1 PF for BF16 / FP8 / FP4. Prefill also gets a "spec" column (public
  4 / 17.5 / 35 PF, 22 TB/s).
- Simplifications: no overlap except where stated, imbalance factor 1, k = 3 (AL 3.08), context 135K.

The SOL model must replace every screening number with the op-level sum.

---

## 0. Vocabulary

**Op families.** Each strategy table has one row per family:

| family | ops (`ops.json` ids) | weights mxfp4_release / nvfp4_modelopt |
|---|---|---|
| embedding | `G.embed` | BF16 2.35 GB (rows touched only) |
| AttnRes / norms | `L.attnres_pre/post`, `F.attnres_final` | none; ~4 MB per row per forward, memory-bound |
| dense MLP (layer 0) | `DENSE.*` | BF16 1.45 GB |
| KDA projections (69) | `KDA.in_proj`, `f_b_proj`, `o_proj` | BF16 61.3 GB / **MXFP8 31.6 GB** |
| KDA core (69) | `KDA.conv`, `gate_beta_norm`, `core_*`, `o_norm` | FP32 conv weights; FP32 state 6.29 MB/head-set/layer |
| MLA projections (24) | `DSA.qkv_a_packed` (q_a, kv_a), `qb_wq_packed` (q_b part), `g_proj`, `w_uk`, `w_uv`, `o_proj` | BF16 11.15 GB (both) |
| indexer (24) | `wq_index`, `wk`/weights (inside packed GEMMs), `idx_q_quant`, `idx_score`, `topk` | BF16 0.36 GB; MXFP4 keys 68 B/token/layer |
| DSA attention (24) | `DSA.prologue` (KV insert), `sparse_mla`, `gate_mul` | FP8 MLA KV 576 B/token/layer |
| router (92) | `MOE.router`, `MOE.topk` | BF16 1.18 GB |
| latent projections (92) | `MOE.latent_down`, `latent_norm`, `latent_up` | BF16 9.46 GB |
| shared experts (92) | `MOE.shared_*` | BF16 24.3 GB / **MXFP8 12.5 GB** |
| routed experts (92) | `MOE.dispatch`, `fc1`, `act`, `fc2`, `combine` | MXFP4 W + MXFP8 act 1446 GB / **NVFP4 W4A4 1532 GB** |
| lm_head / verify | `F.norm`, `F.lm_head`, `F.rejection` | BF16 2.35 GB |
| drafter (DSpark) | `DS.*` (aux capture, fc, ctx K/V, 5 SWA GQA layers, lm_head on k rows, k Markov steps) | BF16 4.96 GB; SWA KV 20,480 B/token x 2048 |

**Sharding modes used in the tables:**

| mode | meaning |
|---|---|
| `TP-N col/row` | Megatron column / row split over N; the row output needs AR or RS. |
| `SP-N` | Residual row-sharded n = T/N. AG before attention and RS after o_proj (P2/P3). |
| `head-TP-N` | KDA heads / N (12288/N channels). State, conv and recurrence local; no intra-layer comm. |
| `DCP-W` | DSA KV sharded by token (t mod W). Decode: q publish (P6) + exact top-k merge (P8) + proj combine (P7b). Prefill: ShardQ (P9 context push, P4 q scatter, P5 out gather). |
| `DSA-DP-N` | Request owned by one card of the N group, all 96 heads there; DSA weights replicated. push_in/push_out (P13/P13b). KV once per group. |
| `DEP` | Whole non-expert layer on one GPU per request; all dense weights replicated. |
| `EP-E` | MegaMoE over E ranks, 896/E experts each, per-route dispatch/combine (P11/P12). |
| `rep` | Replicated weight. "rep-compute" means the same rows are also recomputed on every rank. |

**Per-layer comm unit sizes.** Bytes **received per GPU per own row** (own row = a token/row whose residual lives on
this GPU; under plain TP every rank's own rows = T/N):

| primitive | formula | N or W or E = 2 / 4 / 8 / 16 |
|---|---|---|
| SP AG + RS (or NVLS AR equivalent), one attention layer | 2(N-1) x 14,336 | 28.7 / 86 / 201 / 430 KB |
| DSA-DP push_in + push_out, one DSA layer | 2 x 14,336 x (N-1)/N | 14.3 / 21.5 / 25.1 / 26.9 KB |
| DCP decode, one DSA layer (q publish FP8 + proj value + top-k merge) | (W-1) x (96 x 576 + 96 x 260 + ~1.5K) | 82 / 245 / 573 / 1227 KB |
| ShardQ prefill q scatter + out gather, one DSA layer | (N-1)/N x 96 x (192 + 128) x 2 | 31 / 46 / 54 / 58 KB |
| ShardQ context push, one DSA layer | ΣC x 644 x (W-1)/W / (T/W) | 36 KB at C = 130K, T = 16K, W = 8 (copy engine) |
| MegaMoE dispatch + combine, one MoE layer (MXFP8 / NVFP4 act) | 16 x (3696 or 2016 + 7168) x (E-1)/E | 174 / 147 KB at large E |
| TP-MoE front + tail (no EP), one MoE layer | ~(N-1) x 53.8 KB | 54 / 161 / 377 / 807 KB |

Per step, multiply by the layer counts: 93 attention layers (69 KDA + 24 DSA) and 92 MoE layers.

---

## 1. What drives the choice (from the op inventory)

### 1.1 Prefill (compute-bound)
- **Prefill is GEMM-bound.** At the 16K-chunk / 120K-prefix reference, unsharded:
  - mxfp4_release: BF16 GEMMs are ~60% of roofline time. KDA `in_proj` alone is 28%.
  - nvfp4_modelopt: the KDA projections and the shared expert become MXFP8 and the experts run at FP4 rate. That
    roughly **doubles the compute SOL**: ~46K vs ~26K fresh tok/s/GPU at spec rates, and ~33K vs ~21K at measured
    rates.
- **Topologies differ in three ways:**
  - **replicated compute.** Plain TP replicates `qkv_a`, the indexer and the KV insert, and pads heads. That costs
    15-45%.
  - **communication.** The EP payload is ~16 MB per token per step (92 x 174 KB). At nvfp4 rates that is 37-53% of
    achievable NVLink even for DEP. TP collectives add 2(N-1) x 14 KB per layer on top.
  - **memory capacity.** Weights per GPU limit how much prefix KV, Mooncake staging and workspace fit.
- **Pure compute differs little** between the non-replicating layouts (DEP, DSA-DP, ShardQ-DCP).

### 1.2 Decode (memory-bound, then compute-bound)
- **Step time per GPU ≈ [weights streamed + r x per-request bytes] / HBM + comm + latency floor.**
  - weights streamed = dense / N_dense + experts touched / E;
  - per-request bytes = 1.79 / 2.10 / 2.40 GB at k = 3 / 5 / 7, L = 135K. About 0.93 GB of that is the FP32 KDA
    state.
- **Dense weights:**
  - 113.8 GB BF16 (mxfp4) or 72 GB (nvfp4) for the whole non-expert model;
  - **replicated per GPU under DEP**, which costs 5-8 ms per step whatever the batch;
  - /N under TP. The MoE dense parts (35 GB) stay replicated under SP unless they are TP-sharded explicitly.
- **Experts:** 1446 / 1532 GB in total; each GPU streams 1/E of the touched experts. All 896 are touched once the EP
  domain runs more than ~250 rows.
- **High r turns compute-bound.** At about 50+ requests per GPU (k3), the BF16 GEMMs (~120 GFLOP per row) and the
  FP32 KDA recurrence (~1 GFLOP per row on SIMT) take over. The nvfp4 checkpoint moves ~86 GFLOP per row to the FP8
  rate.
- **KV capacity.** Per request at 135K: 2.09 GB DSA KV + 0.50 GB KDA + 0.05 GB SWA. Capacity per GPU is
  `(272.5 GiB - weights - reserve) / (per-request bytes x replication / group)`. DEP layouts cap at 25-65 requests
  per GPU. Layouts with TP-sharded dense parts and wide EP reach 300+.

### 1.3 Screening summary

Full tables are in §4. The best candidate per regime, k = 3, ctx 135K, tokens/s/GPU:

| regime (requests per GPU) | mxfp4_release best | nvfp4_modelopt best | existing best today |
|---|---|---|---|
| r ≤ 1 (C1-C16 class) | D8 TP16+DCP16 + EP64: 556 tok/s/GPU at r = 1, step 5.5 ms | D8: 567 | D1 TP16 MoE-TP16: 390 (step 7.9 ms) |
| r = 4-16 | D7 TP4 DSA-DP + TP-dense + EP64: 1.5K / 4.6K | D7: 1.7K / 5.0K | D3 hybrid: 0.9K / 3.0K |
| r = 32-64 | D7 TP2-TP4: 6.9K-9.1K | D7 TP2 / DEP72: 7.2K-10.0K | D3 / D5: 4.9K-7.6K |
| prefill | DEP / DSA-DP / P6 tie at ~21K measured-rate compute; DEP has the least comm | same set at ~37K; comm overlap and dedup decide | P2 hybrid |

---

## 2. Prefill strategies

Metric: fresh tokens/s per GPU at AgentX contexts. Fresh tokens average 5.46K per request; the cached prefix is
72.9K weighted by fresh tokens; chunks are 16K-32K. The model must also report:
- TTFT for the long tail (12% zero-hit requests carry 47% of fresh tokens; p99 74K fresh);
- weights per GPU and the free HBM for prefix KV and Mooncake staging;
- per-request NIXL bytes to D.

### P1. TP8 + DCP8 (DCP ShardQ), EP8 MoE token shard — *existing (today's P/D prefill)*

| family | mapping |
|---|---|
| embedding | vocab / 8, AR |
| dense MLP | TP8 col/row, AR |
| KDA proj / core | TP8 col (`f_a` rep) / head-TP8 |
| MLA proj | `qkv_a` rep-compute (all T rows on all ranks); q_b, g, W_UK/W_UV on the ShardQ slice with all heads; o_proj row |
| indexer | `wq_index` rep; scores own ShardQ rows (fold split 2W) vs the gathered context |
| DSA attention | DCP8 KV (t mod 8). Context push P9 by copy engine, q scatter P4, out gather P5. No top-k merge. |
| router / latent | on T/8 token-shard rows |
| shared experts | TP8 on all T rows; routed rows added in the GEMM-AR epilogue |
| routed experts | EP8 on T/8 rows |
| lm_head | vocab / 8, last token only |
| drafter | TP8 (ctx fc row-shard + AG; ctx K/V by kv-head) |

- **Comm per layer (per GPU, per own token):**
  - every attention layer and every MoE layer has an AR [T, h]. With NVLS that is ~8 x 14,336 B, plus 24 DSA layers
    x (54 KB q/out + ~36 KB context push).
  - MoE: 174 KB x 7/8.
  - Total ≈ 21 MB per token per step from TP ARs plus 14 MB from EP. **This is 62-100% of NVLink at the compute
    rate.**
- **Replicated:**
  - weights: `qkv_a` + indexer projections (45 MB per DSA layer), `f_a`;
  - KV: none (DCP);
  - state: none.
  - Weights are 205 GB per GPU, which leaves ~65 GB for KV and workspaces.
- **Strengths:** low TTFT per request (8 GPUs on one chunk); prefix KV spread over 8.
- **Weaknesses:**
  - the AR volume grows with N, so this is the worst comm of all;
  - the 205 GB of weights;
  - screening: 21K compute (mx) but only 13K with comm exposed. The nvfp4 compute SOL (35-47K) cannot be reached
    without overlap.
- **Model knobs:** N ∈ {4, 8}; GEMM-AR fusion on/off (overlap fraction).

### P2. Hybrid DP x (TP_N = DCP_N = SP_N) ShardQ, EP = D x N — *existing (C512 prefill, DP4 x TP4, EP16)*

| family | mapping |
|---|---|
| embedding | vocab / N, RS (`sp_embed`) |
| dense MLP | rep weights, on the SP shard |
| KDA | AG -> TP-N col / head-TP-N / row -> RS |
| MLA proj, indexer, DSA attn | as P1 with W = N (ShardQ, DCP-N KV) |
| router, latent, shared | **rep weights** on the SP shard (no TP comm); shared overlaps the MegaMoE launch |
| routed experts | EP over all D x N ranks |
| lm_head | vocab / N |
| drafter | TP-N, SP-local aux (PR #20) |

- **Comm per layer per own token:**
  - 93 x SP AG+RS (86 KB at N = 4);
  - 24 x ShardQ (46 KB + context push);
  - 92 x EP (174 / 147 KB);
  - plus the cross-DP MegaMoE lock-step and the DP token sync.
- **Replicated:** `qkv_a` rep-compute; MoE dense weights (380 MB per layer, 35 GB per GPU); layer-0 MLP. Weights are
  147 GB per GPU at EP16.
- **Strengths:**
  - TP comm is a quarter of P1's;
  - the KV is DCP-sharded;
  - measured C512 anchor.
- **Weaknesses:**
  - SP AG still exposed (~12 ms per step in PR #20);
  - MoE dense weight replication eats HBM;
  - DP lock-step padding.
- **Screening:** mx 21K compute / 15K exposed; nv 36K / 22K.
- **Knobs:** N ∈ {2, 4, 8}, D, E = D x N, chunk 16-32K.

### P3. DEP-E — *existing (DEP8/12/16/32)*

| family | mapping |
|---|---|
| everything except routed experts | one GPU per request (all 96 MLA heads, all 96 KDA heads, full vocab), weights rep |
| routed experts | EP-E |
| drafter | local |

- **Comm per own token:** EP only, 92 x 174 KB (mx) or 147 KB (nv). No TP collectives. DP sync + MegaMoE bucket
  padding.
- **Replicated:**
  - all dense weights (113.8 GB mx / 72 GB nv);
  - KV and state are per request on one GPU. That is not a copy, but the prefix load (~1.9 GB per request) lands on
    one GPU.
- **Strengths:**
  - zero replicated compute and the lowest comm. Screening: mx 21K compute / 17K exposed; nv 37K / 27K. Best
    measured on the 24-layer model (50.0K, PR #20).
  - P->D export is simple.
- **Weaknesses:**
  - weights 204-234 GB at E = 12-16, which leaves little room for prefix KV, Mooncake staging or a second in-flight
    request;
  - one GPU per request, so long zero-hit prompts give high TTFT;
  - DP load imbalance under lock-step MegaMoE;
  - the indexer at 1M context sits on one GPU.
- **Knobs:** E ∈ {8, 12, 16, 32, 72}, chunk size, imbalance factor.

### P4. TP_N x DP with DSA-DP + SP — *existing PR #20 (TP4 x DP3, TP2 x DP6)*

| family | mapping |
|---|---|
| embedding / dense MLP | as P2 (SP) |
| KDA | SP AG -> head-TP-N -> RS |
| MLA proj + indexer + DSA attention | **DSA-DP**: owner card runs all heads for its requests (weights rep, 480 MB per DSA layer); push_in/push_out (P13) replace AG/RS |
| DSA KV | owner region only: one copy per TP group, static per-card regions |
| router / latent / shared | rep weights on the SP shard |
| routed experts | EP over D x N |
| drafter | TP-N, SP-local aux |

- **Comm per own token:** 69 x SP AG+RS (86 KB at N = 4, 28.7 KB at N = 2) + 24 x push (21.5 KB) + 92 x EP.
- **Replicated:** DSA weights (11.5 GB), MoE dense weights. 185 GB per GPU at TP4 x DP3 EP12.
- **Strengths:**
  - no replicated DSA compute and no head padding;
  - DSA comm about 1/4 of the AG;
  - measured 46.1K vs DEP 50.0K on the 24-layer model.
- **Weaknesses:**
  - owner hotspot: one chunk's DSA lands on one card (long context -7..12%);
  - static KV regions waste capacity under skew;
  - KDA AG is still exposed.
- **Knobs:** N ∈ {2, 4}, owner-balance factor, chunk 4096 (`VLLM_K3_DSADP_CHUNK`).

### P5. TP-only (replicated DSA) — *existing baseline*

TP-N for everything. Variants:
- (a) plain: the indexer and KV are replicated and heads padded (24 -> 64 at TP4, 12 -> 64 at TP8);
- (b) indexer split: score T/N rows, all-gather `[T, 2048]` int32;
- (c) TP ShardQ (`VLLM_K3_TP_SHARDQ`): KV replicated, no context push.

The MoE is either an EP token shard (TP8) or SP + EP.

- **Comm:** as P1/P2 without DCP, plus (b) T x 8 KB x (N-1)/N per DSA layer.
- **Replicated:** KV x N (the full DSA KV on every rank), indexer, `qkv_a`, padded heads.
- **Screening:** mx TP4 17K, TP8 14K. The worst; it is the reference that every other strategy must beat.
- **Strength:** simplest P/D (any D layout reads from any rank).
- **Weakness:** prefix KV is replicated N x, so prefill capacity is lowest.

### P6. Layer-typed prefill (LTP) — *new*

Pick the parallelism per layer type instead of one TP group:

| family | mapping |
|---|---|
| KDA (69 layers, 64% of attention FLOPs incl. projections) | head-TP_k with SP, k ∈ {1, 2} (k = 1: DEP-like local) |
| MLA proj + indexer + DSA attention | DSA-DP over the k group (k = 1: local). Optional **S1/S13 split indexer**: indexer keys of the owner's requests sharded over a group G_i; exact histogram merge; ids to owner |
| DSA KV | one copy (owner), indexer keys optionally /G_i |
| router / latent / shared | SP-local on own rows, rep weights; or TP_k-sharded (variant) |
| routed experts | **rack-wide EP** (E = 32-72) with per-node dispatch dedup + node multicast (O2) |
| lm_head / drafter | local or TP_k; the drafter `fc` / ctx K/V only on the last 2048 rows per request (ops finding 5) |

- **Comm per own token:** 69 x 28.7 KB (k = 2) + 24 x 14.3 KB + 92 x EP. EP with O2 dedup is about 110 KB at
  E = 72 (U ≈ 10.4 destination nodes x (3696 + 7168) B).
- **Replicated:** dense weights at 1/k. E = 72 brings experts to 20 GB per GPU, so weights are ~80 GB per GPU at k = 2
  (mx). That frees ~170 GB per GPU for prefix KV and Mooncake staging.
- **Strengths:**
  - DEP-level compute efficiency with a much lower weight footprint;
  - comm close to DEP;
  - the split indexer removes the long-context owner hotspot.
- **Weaknesses:**
  - EP over 72 GPUs means rack-wide lock-step and larger imbalance tails;
  - needs a dispatch-dedup MegaMoE variant (new kernel);
  - DSA-DP owner balancing.
- **Screening:** k = 2, EP64: mx 21K / 16K exposed; nv 37K / 25K. The comm term is what the overlay O1 must hide.
- **Knobs:** k, E, G_i, O1/O2 on/off.

### P7. Pipeline-split prefill (PP_S x DEP or x LTP per stage) — *new (S4)*

S stages of contiguous layers (e.g. S = 4: 23 / 23 / 23 / 24 layers, DSA layers 3, 7, ..., 91, 92 split ~6 per
stage). Inside a stage, DEP or LTP with stage-local EP.

| family | mapping |
|---|---|
| per-stage layers | as P3 / P6 inside the stage; each GPU holds only 1/S of the dense weights and 1/S of the experts at the same E |
| embedding / lm_head / drafter | first / last stage (the drafter reads aux layers 76-92: all in the last stage at S ≤ 4) |
| KV and KDA state | only for the stage's layers (6 DSA + ~17 KDA layers per stage) |

- **Comm:** EP inside the stage, plus boundary transfers per token of the residual + AttnRes block state:
  ≤ (1 + ceil(layer/12)) x 14,336 B ≤ 129 KB, once per boundary. That is under 1% of EP bytes.
- **Replicated:** nothing new. Weights per GPU / S, so DEP-style layouts fit with ~200 GB free per GPU.
- **Strengths:**
  - DEP compute efficiency at a fraction of the HBM;
  - prefix KV per GPU / S: the Mooncake load is spread over S GPUs;
  - no KDA TP comm.
- **Weaknesses:**
  - pipeline fill/drain bubbles (needs ≥ S chunks in flight; AgentX prefill has many requests, so OK);
  - S x stage latency in TTFT;
  - stage imbalance (DSA layers cluster at 91/92);
  - P->D: D must collect KV from S stages (NIXL handles it, more descriptors).
- **Knobs:** S ∈ {2, 4}, intra-stage layout, microbatch count.

### P8. Long-request context parallel (CP over fresh tokens), spilled from a DEP/LTP pool — *new (S8 + ShardQ)*

Applies only to requests whose fresh part exceeds a threshold (e.g. > 32K fresh tokens: the 12% zero-hit /
long-tail requests).

| family | mapping |
|---|---|
| GEMMs (all projections, MoE dense) | token-sharded over the CP group G (each rank owns a contiguous slice of the fresh tokens) |
| KDA core | **segment-parallel scan**: each rank runs its slice from a zero state, exchanges per-head transitions (96 x 128 x 128 FP32 per layer) in log2(G) rounds, then a fix-up pass |
| DSA attention | ShardQ-like: each rank's rows attend the gathered context (P9) of the earlier slices |
| routed experts | EP as the pool |

- **Comm:**
  - KDA scan: log2(G) x 6.3 MB per layer per request (not per token);
  - DSA context push ≈ C x 644 x (G-1)/G per DSA layer.
- **Replicated:** none.
- **Strengths:** cuts TTFT of 74K-744K fresh prompts by ~G without changing the throughput of the 88% short requests.
- **Weaknesses:**
  - the KDA scan changes FP32 reduction order (allowed: "order may differ") and adds a fix-up pass (~+1x KDA core
    FLOPs);
  - throughput-neutral at best.
- **Model:** report it as a TTFT term only.

### P9. DSA-CP prefill (DEP everywhere, ShardQ-DCP on the 24 DSA layers only) — *new (S2 for prefill)*

| family | mapping |
|---|---|
| KDA, MoE dense, router, shared, dense MLP, lm_head, drafter | DEP (local per request) |
| MLA proj + indexer + DSA attention | the request's DSA-layer rows are ShardQ-split over a group G: q scatter, context gathered from the G ranks that hold the prefix KV (t mod G) |
| DSA KV | /G |
| routed experts | EP |

- **Comm:** EP + per DSA layer (q 192 + out 128) x 96 x 2 x (G-1)/G ≈ 54 KB per token at G = 8, plus the context
  push. No KDA comm.
- **Replicated:** dense weights (as DEP).
- **Strengths:**
  - the long-context prefix KV and the indexer scan are spread over G, which fixes DEP's 1M-context hotspot and the
    Mooncake load concentration;
  - KDA stays comm-free.
- **Weaknesses:** DEP weight footprint; G-wide barriers on DSA layers couple DP ranks.
- **Pairs well with:** P7 (PP) to recover HBM.

### Prefill overlays (apply to P2-P9; modeled as switches)

| id | overlay | model effect |
|---|---|---|
| O1 | two-microbatch overlap (S9) of AG/RS, push and EP | `t_layer = max(compute, comm) + switch cost` instead of the sum. Today GEMM-AR/RS fusion and MegaMoE dispatch-in-FC1 hide part. |
| O2 | dispatch dedup + hierarchical EP (S5): one copy per (token, destination node); node-local NVLS multicast; FP32 pre-combine per node | EP bytes x U(E_nodes) / 16. Needs sign-off: pre-combine changes BF16 rounding order. |
| O3 | die-local placement (S11): experts and per-request KV/state on the die whose SMs use them | HBM rate for those bytes 17.1 vs 13.9 TB/s (mostly a decode term) |
| O4 | MoE activation format for mxfp4_release: MXFP8 (rule) vs NVFP4 (current Rubin backend above 512 tokens/rank) | FP8 vs FP4 rate for fc1/fc2; dispatch 3696 vs 2016 B per route. **Open question to the user.** |
| O5 | EPLB redundancy | imbalance factor on EP compute + comm per E |

---

## 3. Decode strategies

Metric: step time and tokens/s/GPU vs requests per GPU r. Each request contributes q = 1 + k rows per step. The
limits are concurrency (KV capacity, `rmax`) and the per-step latency (interactivity = AL / step).

### D1. TP16 + DCP16 (specialized decode), MoE TP16 — *existing (low-latency P/D decode)*

| family | mapping |
|---|---|
| embedding | vocab / 16, owner publish |
| dense MLP | TP16, AR |
| KDA | TP16 col / head-TP16 / row + two-shot or one-shot AR (P1b/P1c) |
| MLA proj | `qkv_a` + `wq_index` rep-compute (all rows); q_b, g col; **W_UV over all 96 heads rep** (proj variant) |
| indexer | scores own C/16 keys; **exact DcpExchange top-k merge** (P8) |
| DSA attention | DCP16 KV; q publish FP8 (P6); proj value combine (P7b) |
| router / latent | sharded front + multicast AG (P14) |
| shared experts | TP16 |
| routed experts | **TP16** (every rank holds 1/16 of all 896 experts; intermediate 192) |
| lm_head / verify | vocab / 16, sharded verify (P16) |
| drafter | TP16, 11 fused AR+RMSNorm sites, k Markov exchanges |

- **Comm per own row per step:**
  - attention ARs 93 x 430 KB;
  - DCP 24 x 1.23 MB;
  - TP-MoE 92 x 807 KB.
  - Total ≈ 145 MB per own row. At r = 16 that is ~7 ms, the largest comm of the decode set.
- **Replicated:**
  - weights: `qkv_a` + indexer (45 MB per DSA layer), `f_a`;
  - KV: none;
  - state: none.
  - 98 GB per GPU leaves the largest KV room (rmax ≈ 1050 at 135K, SOL accounting).
- **Strengths:**
  - lowest dense weight per GPU among existing layouts;
  - no EP lock-step;
  - all hops are intra-group.
- **Weaknesses:**
  - experts / 16 = 90 GB streamed per step once more than ~16 requests run;
  - comm grows with N x rows;
  - screening: 390 tok/s/GPU at r = 1, 2.6K at r = 16.
- **Knobs:** N ∈ {8, 16}.

### D2. TP8 + DCP8, MoE EP8 or TP8 — *existing (P/D low)*

As D1 with N = 8 (DCP decode comm 573 KB per DSA layer per own row).

- **Weakness:** experts / 8 = 181 GB per GPU (EP8), so weights are 196 GB per GPU and rmax ≈ 230. The expert stream
  dominates every r.
- **Kept as a reference.** Screening: 309 tok/s/GPU at r = 1, 2.2K at r = 16.

### D3. Hybrid DP_D x (TP_N = DCP_N = SP_N), EP = D x N — *existing (C512: DP4 x TP4, EP16)*

| family | mapping |
|---|---|
| KDA | SP multicast AG (P2b) -> head-TP4 -> `SpOproj` push to row owner (P3b) |
| MLA / indexer / DSA | DCP4 (q publish, top-k merge, proj combine), W_UV all heads rep |
| router / latent / shared / dense MLP | rep weights on the SP shard |
| routed experts | EP16 MegaMoE (decode bucket ≤ 4096 routes/rank) |
| lm_head / drafter | TP4, sharded verify |

- **Comm per own row:** 93 x 86 KB + 24 x 245 KB + 92 x 174 KB ≈ 30 MB. DP lock-step.
- **Replicated:** MoE dense 35 GB per GPU (2.5 ms per step of streaming at every r), `qkv_a`, indexer.
  147 GB per GPU, rmax ≈ 190.
- **Strengths:** measured anchor (B16 25.1 ms per step); balanced.
- **Weaknesses:**
  - replicated MoE dense weights;
  - EP16 still streams 90 GB of experts per step.
- **Screening:** 891 / 3.0K / 7.3K tok/s/GPU at r = 4 / 16 / 64 (mx).

### D4. DEP-E — *existing (DEP16/32; DEP72 extension)*

| family | mapping |
|---|---|
| all non-expert | local per request (all heads, full vocab, full drafter) |
| routed experts | EP-E |

- **Comm per own row:** EP only, 92 x 174 KB ≈ 16 MB.
- **Replicated:** all dense weights, 113.8 GB (mx) / 72 GB (nv) streamed every step: **8.2 / 5.2 ms per step floor**.
  KV and state are full per request on one GPU (2.66 GB at 135K), so rmax ≈ 24-65.
- **Strengths:** least comm; no TP hops; simple P/D. Benefits most from the nvfp4 checkpoint (its dense weights shrink
  37%).
- **Weaknesses:**
  - poor at low r because of the dense floor;
  - **capacity-capped before the compute crossover** (DEP16 caps at r ≈ 24 in mx).
- **Screening:** DEP72 nv reaches 9.9K tok/s/GPU at r = 64, competitive with D7 at high r.

### D5. TP_N DSA-DP decode + SP, EP — *existing PR #23 (TP4)*

| family | mapping |
|---|---|
| KDA | SP + head-TP-N |
| MLA / indexer / DSA | DSA-DP owner, all 96 heads, weights rep; Lamport exchange P13b (no barrier); batch in DEP order padded to the busiest owner (graph sizes multiples of N x q) |
| DSA KV | one copy per group (owner region) |
| MoE dense / experts | SP-rep / EP |
| drafter | TP-N (KV in the shared pool) |

- **Comm per own row:** 69 x 86 KB + 24 x 21.5 KB + 92 x 174 KB. The DSA comm is ~1/11 of DCP4's.
- **Replicated:** DSA weights 11.5 GB per GPU (0.8 ms per step).
- **Strengths:**
  - no DCP top-k merge or q publish;
  - no head padding;
  - one KV copy.
- **Weaknesses:**
  - owner padding overhead N x max_o(B_o) / B;
  - indexer O(C) scan and KV gather on the owner only (per-owner hotspot at small B);
  - static KV regions.
  - Measured -5% step time at 32 requests per GPU vs replicated DSA, +1.1% at 2.
- **Screening ≈ D3:** 871 / 3.0K / 7.6K.

### D6. TP-only, replicated DSA KV — *existing baseline*

TP_N with SP + EP and no DCP. Every rank stores and scans the full DSA KV.

- **Replicated:** KV x N (2.09 GB per request per rank), indexer scan x N, sparse MLA gather x N, padded heads.
- **Screening:** rmax 56 (mx, TP4) and 2.8K tok/s/GPU at r = 16. Capacity-capped.
- **Reference only.**

### D7. Layer-typed wide-EP decode (TP_a attention, TP-sharded MoE dense, rack-wide EP) — *new, main mid/high candidate*

| family | mapping |
|---|---|
| embedding | vocab / a |
| dense MLP (layer 0) | TP_a |
| KDA | head-TP_a with SP (multicast AG + push-RS) |
| MLA / indexer / DSA attention | **DSA-DP over the a group** (variant: DCP_a; variant: S1 split indexer) |
| router / latent / shared | **TP_a-sharded** (not SP-replicated): AG of the SP rows, column-sharded router + latent down + shared gate_up, row-sharded latent up + shared down, RS. That is the existing `VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT` path extended to router and latent |
| routed experts | **EP over the rack** (E = 32-72), with O2 node-level dedup as an option |
| lm_head / verify | vocab / a, sharded verify |
| drafter | TP_a (or O6) |

- **Comm per own row:**
  - 69 x 2(a-1) x 14 KB (KDA);
  - 24 x DSA-DP push;
  - 92 x [2(a-1) x 14 KB (MoE dense AG/RS) + 174 KB (EP)].
  - At a = 4: ≈ 22 MB per own row.
- **Replicated:** DSA weights only (11.5 GB). Weights per GPU: 61 GB (mx, a = 4, E = 64), 48 GB (a = 8), 88 GB (a = 2).
  rmax 318 / 677 / 139 at 135K.
- **Strengths:**
  - streams the least weight per GPU at every r: 113.8 / a dense + 1446 / 64 experts ≈ 51 GB (mx, a = 4) vs D3's
    ~147 GB per step;
  - largest capacity among non-DCP layouts;
  - **screening winner from r ≈ 1 to 64**: mx 1.5K / 4.6K / 6.9K / 8.7K tok/s/GPU at r = 4 / 16 / 32 / 64.
- **Weaknesses:**
  - rack-wide EP lock-step: all 64-72 GPUs step together, with stragglers and imbalance tails (needs the O5 factor);
  - MoE dense AG/RS adds 92 collectives (latency floor 2.6 ms at a = 4);
  - EP16 → EP64 dispatch is cross-node, which is flat on NVL72 (`hw.md`).
- **Knobs:** a ∈ {2, 4, 8}, E ∈ {32, 64, 72}, DSA mode {DSA-DP, DCP_a, split indexer}, MoE dense {rep, TP_a}.

### D8. Low-latency wide: TP16 + DCP16 attention, TP16 MoE dense, EP64-72 — *new (low r)*

D1 with the routed experts moved from TP16 to rack-wide EP, and the router, latent and shared experts TP16-sharded.

| family | mapping |
|---|---|
| KDA / MLA / indexer / DSA | as D1 (head-TP16, DCP16 sdec) |
| router / latent / shared | TP16 |
| routed experts | EP64 (4 attention groups x 16) |
| lm_head / drafter | TP16 |

- **Comm per own row:** 93 x 430 KB + 24 x 1.23 MB + 92 x (430 KB + 174 KB).
- **Replicated:** `qkv_a` / indexer weights only. 30 GB per GPU (mx), the largest KV room of all.
- **Strengths:**
  - at r ≤ 1, step ≈ the latency floor plus ~7 GB of streaming: **5.5 ms vs D1's 7.9 ms at r = 1**;
  - the per-step expert stream is touched experts / 64.
- **Weaknesses:**
  - comm grows fast with r (r ≥ 16 loses to D7);
  - the latency floor is set by 93 x ~4 dependent hops (≈ 1.1 ms at 2.86 us) + graph nodes;
  - the four attention groups must all be busy, or the EP lock-step idles three.
- **Knobs:** a ∈ {8, 16}, E.

### D9. DSA-CP decode (DEP / small TP everywhere, DCP_G only on the DSA layers) — *new (S2)*

| family | mapping |
|---|---|
| KDA, MoE dense, dense MLP, lm_head, drafter | DEP (local) or TP_a, a ≤ 2 |
| MLA / indexer / DSA attention | DCP over a wider group G (8-16). Each request's DSA KV is spread over G GPUs; q publish of all 96 heads per row to G-1 peers; exact merge; proj combine |
| routed experts | EP rack-wide |

- **Comm per own row:** 24 x (G-1) x 82 KB (1.23 MB at G = 16) + 92 x 174 KB.
- **Replicated:** dense weights (DEP). The DSA KV is / G, but the KDA state stays full per request.
- **Strengths:** DEP's zero-TP-comm KDA and MoE, with DSA KV capacity per GPU / G. Per GPU per request: 0.13 GB DSA
  + 0.57 GB KDA/SWA at 135K, vs 2.66 GB for DEP.
- **Weaknesses:**
  - the dense weight floor (8.2 ms mx);
  - the KDA state dominates capacity anyway (0.5 GB per request);
  - screening rmax 49 (mx). It loses to D7.
- **Kept to close the option:** it helps only if the DSA KV, not the KDA state, binds (contexts ≫ 135K).

### D10. DSA-DP with split indexer (S1 / S13) — *new (variant of D5 / D7)*

- **Mapping:** the owner keeps the MLA KV and sparse MLA. The owner's indexer keys (68 B per token) are sharded over
  the a group; the indexer q (2.2 KB per row) is broadcast; each card scores its key shard; an exact histogram top-k
  merge follows; 2048 ids go to the owner.
- **Comm per own row per DSA layer:** (a-1) x (2.2 + ~1.5) KB + 8 KB.
- **Replicated:** none.
- **Strengths:** removes the owner's O(C) indexer hotspot. At 135K the indexer scan is 0.25 GB per request per step,
  and 107 GFLOP per request at k3. This makes owner skew cheap.
- **Weakness:** one more dependent exchange per DSA layer (+24 x 2 hops).
- **Model:** as a D5 / D7 DSA-mode option.

### D11. Attention-FFN disaggregation (AFD) — *new*

Two pools inside the rack, with ping-pong of 2-3 microbatches:
- **A-pool:** a GPUs per attention group, TP_a. It holds embedding, KDA, MLA, indexer, DSA, router, latent down/up,
  shared experts, lm_head and the drafter, plus all KV and state.
- **F-pool:** routed experts only, EP_F.

| family | mapping |
|---|---|
| everything except routed experts | A-pool, as D7 (head-TP_a, DSA-DP, TP_a MoE dense) |
| routed experts | F-pool EP_F (896 / F experts per GPU) |

- **Comm:** latent rows (MXFP8 3696 B per route) A -> F, BF16 7168 B per route F -> A, per MoE layer: the same bytes
  as EP, but cross-pool and one-directional per phase.
- **Replicated:** none.
- **Strengths:**
  - A-pool GPUs hold no experts, so HBM goes almost entirely to KV (rmax per A-GPU ≈ (292 - 30) / 0.66 ≈ 400 at a = 4);
  - F-pool GPUs see the whole rack's rows, so experts reach the compute-bound regime (~200 rows per expert) at high
    load;
  - the pool ratio is a free knob.
- **Weaknesses:**
  - needs ≥ 2 microbatches in flight to hide the A <-> F round trip (93-92 serial hops per step otherwise);
  - per-request latency rises by the ping-pong;
  - lowest TRL (new runtime).
- **Model:** step = max(A-pool time, F-pool time) x microbatches + exposed transfer; tokens/s per total GPU.

### Decode overlays (switches on D1-D11)

| id | overlay | model effect |
|---|---|---|
| O1 | two-batch overlap: run batch A's AG / RS / EP under batch B's GEMMs | comm hidden up to the GEMM time; two half batches raise the latency floor |
| O2 | dispatch dedup + node multicast (S5) | EP bytes x U / 16 (-36% EP16, -23% EP32, -9% EP72 at rank level; more at node level) |
| O3 | **die-local placement of all streamed bytes**: each die streams its own half of every weight matrix (N-split by die), its own experts, and the KV / KDA state of the requests whose heads run on its SMs | HBM 17.1 vs 13.9 TB/s on the streaming term (-19% of memory time). Cross-die traffic only for the small activations. |
| O5 | EPLB redundancy / imbalance factor | multiplies EP compute and comm |
| O6 | drafter placement: TP_a (default), TP4-in-node under TP16, or drafter-DP (5 GB local per GPU, 0 ARs) | 11 AR sites x hops vs 4.96 GB / BW = 0.36 ms per step (S12) |
| O7 | fused indexer score + select, deferred RecoverSSM commit, Markov w2 in L2 (ops finding 5) | op-level SOL terms, independent of topology |

### Rejected (recorded to close the option)

- **ADP-GTP (S7):** KDA core on the request owner with TP projections. It needs ~123 KB per row per KDA layer of
  exchange, which is worse than head-TP, whose state shards with no comm.
- **Pipeline-parallel decode:** with S microbatches each stage streams its weights per microbatch, so weight bytes
  per request equal TP-less at batch B/S, plus S x latency. Never better than TP for memory-bound decode.
- **Legacy DCP prefill (non-ShardQ):** repeats attention W times plus a 1.9 GB per rank candidate merge.

---

## 4. Screening tables (both checkpoints)

Decode: step ms / tokens per s per GPU, k = 3 (AL 3.08), ctx 135K. "cap" means r exceeds rmax. W/GPU is the weights
per GPU in GB. rmax uses SOL state accounting (one KDA state per request, FP32 conv); the deployed allocator reserves
far more (`workload.md`).

**mxfp4_release**

| topology | W/GPU | rmax 135K / 189K | r = 0.25 | r = 1 | r = 4 | r = 16 | r = 32 | r = 64 |
|---|---|---|---|---|---|---|---|---|
| D1 TP16+DCP16, MoE-TP16 | 98 | 1051 / 800 | 4.6 / 166 | 7.9 / 390 | 11.7 / 1056 | 18.8 / 2622 | 28.2 / 3494 | 47.0 / 4190 |
| D2 TP8+DCP8, EP8 | 196 | 231 / 176 | 5.8 / 133 | 10.0 / 309 | 17.0 / 725 | 22.3 / 2206 | 27.7 / 3555 | 38.5 / 5122 |
| D3 DP4x(TP4=DCP4) SP, EP16 | 147 | 188 / 143 | 8.2 / 94 | 11.2 / 276 | 13.8 / 891 | 16.5 / 2978 | 20.1 / 4907 | 27.2 / 7257 |
| D4 DEP16 | 209 | 24 / 18 | 12.1 / 64 | 15.1 / 205 | 17.6 / 700 | 19.8 / 2492 | cap | cap |
| D4 DEP32 | 164 | 41 / 31 | 11.9 / 65 | 13.5 / 227 | 14.4 / 855 | 16.5 / 2978 | 19.4 / 5080 | cap |
| D4 DEP72 | 139 | 50 / 38 | 11.5 / 67 | 12.1 / 255 | 12.6 / 977 | 14.8 / 3340 | 17.6 / 5593 | cap |
| D5 TP4 DSA-DP SP, EP16 | 156 | 175 / 133 | 8.6 / 90 | 11.6 / 267 | 14.1 / 871 | 16.6 / 2972 | 19.7 / 4993 | 26.1 / 7565 |
| D6 TP4 rep-DSA SP, EP16 | 147 | 56 / 41 | 8.0 / 96 | 11.0 / 279 | 13.9 / 885 | 17.5 / 2815 | 22.2 / 4437 | cap |
| **D7 TP4 DSA-DP, TP-dense, EP64** | 61 | 318 / 242 | 6.6 / 116 | 7.3 / 421 | **8.0 / 1538** | **10.7 / 4601** | **14.3 / 6887** | 22.8 / 8663 |
| D7 TP2, EP64 | 88 | 139 / 106 | 8.7 / 89 | 9.3 / 331 | 9.9 / 1243 | 12.2 / 4023 | 15.4 / 6413 | **21.6 / 9123** |
| D7 TP8, EP64 | 48 | 677 / 515 | 5.7 / 136 | 6.4 / 484 | 7.2 / 1702 | 10.7 / 4627 | 15.2 / 6483 | 26.5 / 7436 |
| **D8 TP16+DCP16, TP-dense, EP64** | 30 | 1459 / 1110 | **4.6 / 166** | **5.5 / 556** | 7.1 / 1726 | 13.5 / 3658 | 22.6 / 4367 | 42.3 / 4663 |
| D9 DSA-CP (DEP + DCP16 on DSA), EP64 | 141 | 49 / 38 | 11.8 / 65 | 12.5 / 246 | 13.4 / 922 | 16.6 / 2960 | 21.0 / 4688 | cap |

**nvfp4_modelopt**

| topology | W/GPU | rmax 135K / 189K | r = 0.25 | r = 1 | r = 4 | r = 16 | r = 32 | r = 64 |
|---|---|---|---|---|---|---|---|---|
| D1 TP16+DCP16, MoE-TP16 | 101 | 1035 / 787 | 4.5 / 169 | 8.0 / 387 | 11.9 / 1039 | 19.0 / 2595 | 28.4 / 3470 | 47.2 / 4173 |
| D2 TP8+DCP8, EP8 | 201 | 215 / 164 | 5.5 / 140 | 9.9 / 311 | 17.3 / 713 | 22.6 / 2178 | 27.9 / 3533 | 38.4 / 5129 |
| D3 DP4x(TP4=DCP4) SP, EP16 | 134 | 209 / 159 | 6.9 / 112 | 10.0 / 307 | 12.8 / 963 | 15.4 / 3194 | 18.9 / 5228 | 25.7 / 7673 |
| D4 DEP16 | 173 | 37 / 28 | 9.2 / 84 | 12.3 / 250 | 15.0 / 824 | 17.1 / 2889 | 19.8 / 4987 | cap |
| D4 DEP32 | 125 | 55 / 42 | 9.0 / 86 | 10.7 / 287 | 11.6 / 1063 | 13.6 / 3614 | 16.4 / 6023 | cap |
| D4 DEP72 | 99 | 65 / 50 | 8.6 / 89 | 9.2 / 336 | 9.7 / 1273 | 11.7 / 4200 | 14.5 / 6808 | 20.0 / 9875 |
| D5 TP4 DSA-DP SP, EP16 | 142 | 196 / 149 | 7.3 / 105 | 10.4 / 296 | 13.1 / 939 | 15.5 / 3186 | 18.5 / 5326 | 24.6 / 8019 |
| D6 TP4 rep-DSA SP, EP16 | 134 | 62 / 45 | 6.7 / 115 | 9.9 / 311 | 12.9 / 956 | 16.4 / 3007 | 21.0 / 4699 | cap |
| **D7 TP4 DSA-DP, TP-dense, EP64** | 52 | 332 / 253 | 6.0 / 129 | 6.7 / 463 | **7.3 / 1681** | **9.9 / 4960** | **13.4 / 7348** | 20.4 / 9679 |
| D7 TP2, EP64 | 68 | 154 / 117 | 7.2 / 106 | 7.9 / 390 | 8.5 / 1452 | 10.7 / 4593 | 13.7 / 7182 | **19.7 / 9999** |
| D7 TP8, EP64 | 44 | 689 / 524 | 5.3 / 144 | 6.1 / 506 | 6.9 / 1778 | 10.2 / 4808 | 14.7 / 6716 | 23.5 / 8378 |
| **D8 TP16+DCP16, TP-dense, EP64** | 29 | 1467 / 1116 | **4.5 / 171** | **5.4 / 567** | 7.0 / 1756 | 13.3 / 3717 | 21.6 / 4567 | 38.2 / 5157 |
| D9 DSA-CP, EP64 | 101 | 64 / 49 | 8.9 / 86 | 9.6 / 320 | 10.5 / 1179 | 13.6 / 3613 | 17.9 / 5510 | 26.4 / 7471 |

Breakdown at r = 16 (mx; memory / compute / exposed comm / latency floor, ms):

| topology | memory | compute | comm | latency |
|---|---|---|---|---|
| D3 | 12.7 | 3.5 | 1.5 | 2.25 |
| DEP16 | 17.1 | 3.5 | 0.8 | 1.5 |
| D5 | 13.3 | 3.5 | 1.1 | 2.05 |
| D7 a = 4 | 6.4 | 3.5 | 1.5 | 2.6 |
| D1 | 9.1 | 3.5 | 7.4 | 2.25 |

The measured C512 B16 step is 25.1 ms vs 16.5 for D3 here. The SOL model must explain the gap op by op.

**Prefill** (fresh tok/s/GPU at the 16K / 120K reference; compute only / comm fully exposed / comm overlapped;
EP comm at 1.25 TB/s):

| topology | mx compute (spec / measured rates) | mx exposed | nv compute (spec / measured) | nv exposed | nv comm share of NVLink when overlapped |
|---|---|---|---|---|---|
| P1 TP8+DCP8 ShardQ, EP8 token shard | 25K / 21K | 13K | 47K / 35K | 18K | ~100% (comm-bound) |
| P2 DP4x(TP4=DCP4) ShardQ, EP16 | 25K / 21K | 15K | 49K / 36K | 22K | 64-87% |
| P3 DEP16 / DEP32 | 26K / 21K | 17K | 51K / 37K | 27K | 37-53% |
| P4 TP4xDP DSA-DP, EP16 | 26K / 21K | 16K | 51K / 37K | 23K | 58-79% |
| P4 TP2xDP DSA-DP, EP16 | 26K / 21K | 16K | 51K / 37K | 25K | 45-62% |
| P5 TP4 rep-DSA (padded) | 21K / 17K | 13K | 36K / 27K | 18K | 47-64% |
| P5 TP8 rep-DSA, token shard | 17K / 14K | 10K | 25K / 19K | 12K | 54-71% |
| P6 LTP k = 2, EP64 | 26K / 21K | 16K | 51K / 37K | 25K | 47-65% (before O2 dedup) |
| P7 PP4 x DEP16 | 26K / 21K | 17K | 51K / 37K | 27K | 37-53% (plus bubbles, not modeled here) |

Reading:
- Among layouts without replicated compute, prefill is decided by **comm volume (EP + TP) and HBM headroom**, not
  compute.
- The nvfp4 checkpoint makes prefill comm-sensitive. EP comm alone is ~40-50% of NVLink at its compute rate, so O1
  (overlap) and O2 (dedup) become first-order.
- P1's AR volume makes it comm-bound under nvfp4.

---

## 5. What the SOL model must implement per strategy

1. **Op table × sharding vector.** For every op in `ops.json`, a shard divisor (N, W, E, 1) and a replication flag
   per strategy, taken from the tables above.
2. **Comm list per layer type.** Primitive id, bytes in/out per GPU and dependent hops (`comm.json`), priced with
   `hops x hop + bytes / BW`:
   - SOL column: 1.8 TB/s per direction, no hop cost beyond one graph node;
   - achievable column: 1.25 TB/s and 2.86 us.
3. **Overlap policy per strategy:** fused (GEMM-AR / RS, MegaMoE dispatch-in-FC1, copy-engine context push) vs
   exposed, and O1 as a switch.
4. **Capacity:** weights per GPU per checkpoint, the reserve (MegaMoE arena, SP buffer, ShardQ workspace, graphs) and
   per-request bytes with the strategy's replication. That gives rmax at 135K / 189K / p90.
5. **Outputs for both checkpoints:**
   - prefill: fresh tok/s/GPU, TTFT for 6K / 74K fresh, weights and free HBM;
   - decode: step-time curve and tok/s/GPU vs r for k = 3 / 5 / 7, rmax;
   - per op family time + comm per layer, for later matching against layer microbenchmarks.
6. **P/D pairing:** any P x any D through NIXL. Per-request KV moved is 2.24 GB at mean ISL plus the KDA state. The
   model reports NIXL bytes per D GPU per request and the descriptor fan-in (P7 stages, DSA-DP owner, DCP W) but
   does not restrict pairings.

## 6. Open questions for the user

1. **MoE activations in mxfp4_release prefill:** MXFP8 (mission wording) or NVFP4 (the Rubin backend above 512
   tokens/rank, exact weight conversion, lossy activation)? This changes P1-P9 by ~1.3x on the expert term (O4).
2. **Pre-combine in dispatch dedup (O2):** FP32 per-node partial, then one BF16 rounding. Is that acceptable as
   "order may differ"?
3. **KDA segment scan (P8):** it reorders the FP32 recurrence across ranks. Acceptable for TTFT-only modeling?
4. **Rack-wide EP lock-step (D7, D8, P6):** which imbalance factor should the SOL column use? The proposal is 1.0 for
   SOL and the measured EPLB max/mean for the achievable column.
