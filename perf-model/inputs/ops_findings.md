## Findings that matter for the SOL model

1. **Prefill is GEMM-dominated, and the largest single op is the KDA input projection, not attention.**
   At a 16K chunk on a 120K prefix (unsharded, one GPU), `KDA.in_proj` (7168 -> 49376, BF16, 69 layers)
   is ~28% of roofline time; with `KDA.o_proj` and the BF16 shared expert / latent projections, BF16 GEMMs
   are ~60%. The routed experts (MXFP4 x MXFP8, at FP8 rate) are ~19%. Sparse MLA is ~6% and the
   indexer ~3% (both compute-bound when the request's cache is L2-resident). The chunked KDA core is
   ~1.6% of FLOPs, but its K2 scan runs serially over chunks: that is a latency/parallelism term (segscan),
   not a FLOP term.
2. **Decode is weight-bound by the experts, and the per-request bytes are dominated by the FP32 KDA state.**
   Each extra request costs ~1.8 / 2.1 / 2.4 GB of HBM traffic per step at k=3/5/7 (L=135K, SOL fusion
   view). Of that, the KDA recurrent state read+write is ~0.87 GB (69 x 12.6 MB) whatever k is; the indexer key
   scan is ~0.22 GB at L=135K (+ BF16 logits scratch); the sparse-MLA gather is q x 32 MB. AttnRes is ~4 MB
   per row per forward (2 mixes x 93 layers over <=9 BF16 sources). Weights: 1.45 TB of experts + 114 GB
   BF16 per engine step. The topology model shards these over EP/TP and sets requests per GPU.
3. **The weight GEMMs only reach the ridge at the batch where they turn compute-bound.** BF16 ridge = 259 rows,
   so KDA in_proj / shared expert / o_proj become compute-bound at about 260 token rows per TP-sharded GEMM
   (B*q). Experts (ridge ~755 FLOP/B, ~3.8 FLOP/B per row) need ~200 rows per expert, which means ~11K tokens
   per EP group. Decode never gets there.
4. **Replication factors are a topology choice, and they are big.** Under plain TP every rank re-runs
   `DSA.qkv_a_packed`, the indexer (`wq_index` + full-context scoring + top-k) and the KV inserts (replicated
   MLA/indexer cache). DCP splits keys, ShardQ splits query rows, and DSA-DP (PRs #20/#23) gives each request
   to one card and stores KV once per TP group. These are the ops whose `shard_axis` mentions
   "replicated".
5. **Places where SOL beats the current code (new methods, flagged in the op notes):**
   - fused indexer score+select (`SCRATCH=0`). This drops the BF16 [rows x L] logits write+read: ~4 B/key/row,
     about 27 MB per request per step at k=3, L=135K.
   - deferred RecoverSSM commit folded into the next verify. The KDA state is then read once and written once
     per step; this is already the specialized decode's "deferred" mode. The stock commit costs 2 reads + 1 write.
   - in prefill, run the final AttnRes mix, the aux captures and the DSpark `fc` / context K/V only on the rows
     that get consumed. For logits that is the last token; for the draft window it is the last 2048 tokens
     per request.
   - keep the vocab shard of the Markov `w2` in L2 (168 MB / tp) across the k sequential positions.
   - store the MLA/indexer KV once per TP group (DSA-DP) instead of once per rank.
6. **Sanity anchors (not fitted).** Unsharded SOL for a 16K chunk at a 120K prefix is ~0.79 s on one GPU,
   i.e. ~21K fresh tokens/s/GPU with perfect parallelism and no communication. Measured C512 hybrid prefill
   (13K-token step >= 165 ms + 11 ms/1K on a DP4 engine) and PR #20 (46K tok/s/GPU, but on a 24-layer model)
   sit in the expected range relative to that. Decode at B64 k3: ~120 ms of single-GPU roofline per engine
   step, which is ~7.5 ms/GPU over 16 GPUs if it parallelised perfectly. The release C512 decode step bench
   gives 31.7 ms at B64 per rank. The topology/comm model has to explain that gap.

## Open questions / assumptions to confirm

- **MegaMoE activation precision above 512 tokens/rank.** The code switches to NVFP4 (w4a4, exact MXFP4->NVFP4
  weight conversion) above `RUBIN_MX_MAX_TOKENS_PER_RANK=512` (`megamoe_rubin/experts.py:87`). The mission allows
  MXFP8 activations. This inventory uses MXFP8 (FP8 rate). Should the prefill SOL also offer NVFP4 activations,
  which would double MMA throughput?
- **KDA core MMA precision.** FlashKDA stages the FP32 state through BF16 into the k.S / q.S MMAs
  (`ops/kda_segscan.py:28-36`). The SOL model assumes stock precision (BF16 MMA operands, FP32 state storage
  and accumulation). If "FP32 KDA state" means FP32/TF32 math, the prefill KDA core slows ~2x (TF32 1.95 PF).
- The conv state dtype comes from `mamba_cache_dtype`; it is assumed FP32 per the mission. The DSpark KV cache
  is BF16 (`draft/gqa_window.py` docstring). Draft logits use `head_dtype` (assumed BF16).
- The code has online-MXFP8 paths for the shared expert, the latent projections and the DSA projections
  (`pack_projections`, `mxfp8_act`). Per the mission rules, this inventory uses BF16 for every linear except
  the routed experts.
- MoE dispatch/combine bytes are given per (token, expert) and per (token, destination rank). Which one MegaMoE
  actually sends (per-rank dedup or pre-reduction) has to be checked in `megamoe_rubin/src`.
- The KDA prefill checkpoint export (`CKPT`) only applies under prefix caching (mamba align mode). It is set to
  1 per chunk at the prefill reference point.
