# ops1: single-GPU kernel microbenchmarks vs the SOL model (VR200)

Lane ops1 covers GEMMs and the attention-side kernels for the EP <= 16 topologies (P1, P2, P3-16, P4-2, P9, D1, D3, D4-16,
D5, D7-2/4/8). P6, D4-32 and D8 appear only where they share a case with an in-scope topology. Every in-scope case of
these families in `microbench/matrix.json` was measured once, each at its exact per-GPU shard shape:

| family | cases | what was timed |
|---|---|---|
| gemm | 440 | BF16: cuBLAS `F.linear`, plus the Kimi K3 low-latency CuTe GEMM where its shape table applies (it applied to none of these shapes in this tree). BF16 batched (`DSA.w_uk/w_uv`): `torch.bmm`. MXFP8 (nvfp4_modelopt KDA/shared): `torch._scaled_mm` (cuBLASLt) and FlashInfer CuTe DSL `mm_mxfp8` (at rows <= 256), i.e. both arms of `KimiK3Mxfp8Kernel`, on pre-quantized activations. Activation quant is timed separately (`quant` in the notes). |
| dsa_indexer_topk | 17 | Prefill: DeepGEMM `fp8_fp4_mqa_logits` (MXFP4 Q and keys, cost-balanced schedule) plus DeepSelect exact top-2048 on the BF16 logits, using the `sparse_attn_indexer` call sequence. The chunk is 3 requests x (prefix 72900, rows spread over 5456 fresh tokens), which gives a mean of 75.6K keys per row. Decode: stock DeepGEMM `fp8_fp4_paged_mqa_logits` (page 64, distinct pages per request, q = 4 rows per request) plus DeepSelect, at the case's keys per row (the DCP key shard in DCP modes). |
| sparse_mla | 17 | FP8 Q x FP8 KV (576 B rows) via FlashInfer trtllm-gen `trtllm_batch_decode_with_kv_cache_mla`, the FLASHINFER_MLA_SPARSE non-DCP path. 96 heads are padded to 128 (as production does), with the multi-CTA KV counters preallocated. |
| kda | 19 | Prefill: `kda._flashkda_prefill` (the in-tree v-split build when it fits, otherwise stock FlashKDA) and `kda_segscan.segmented_prefill` when its planner accepts (it declined every case). 3 x 5456 tokens, FP32 state. Decode: sdec `PreparedKda` verify core (conv + recurrence + norm) at (B, q = 4), FP32 state. |
| attnres | 27 of 43 | `ops.attn_res.attn_res` (`torch.ops._C.kimi_k3_attn_res`) at rows x 1..8 sources, giving the layer mean over l = 1..93 with ceil(l/12) sources (the model's convention). `pre` has no delta; `post` has a delta and a block write; `final` uses 8 sources. Not benched: `G.embed` and `DS.aux_capture` (P2). |
| drafter_verify | 30 of 94 | `DS.markov` = the BF16 logit GEMM [B*3, 512] x [512, vocab_shard] (cuBLAS). `DS.attn` = SWA GQA d64 over 2048 tokens, q3 per request; the faster of FlashInfer paged prefill and `vllm_flash_attn` varlen with paged KV. Not benched (P2, tiny): `DS.confidence`, `F.rejection`, `DS.norm_ar`, `F.norm`, `DS.ctx_kv_insert`, `DS.hidden_norm`, `DS.embed`, `DS.silu`. |

**Timing.** FlashInfer `bench_gpu_time_with_cupti`: the CUPTI per-call span from the first kernel's start to the last
kernel's end, with cold L2 (flushed between calls) and the median taken. Cold L2 is the right assumption for streamed
weights and KV. For the decode attention cases a warm-L2 time is also recorded in `parts` (`*_warm`). Cold and warm
agree within about 5% everywhere, so the small-shape floors below are not caused by the L2 flush.

**Versions and environment.** GPU NVIDIA VR NVL72 (cc 10.7), torch 2.16.0.dev20261009+cu134, CUDA 13.4.
- Image: `k3vr/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh`.
- vLLM: the release tree `$W/dsrel/trees/rel-1767cf7` (SOURCE_REV 6a14e2fada wzhao/k31-dspark-release + 230 files over
  242fb49cd4).
- FlashInfer 0.7.3.dev20261031 (`$W/env/fi-0.7.3.dev20261031`, the tree's `K31VR_FI`), with DeepGEMM / DeepSelect /
  FlashKDA as built in that tree.
- No MegaMoE in this lane.

**Precision.** BF16 linears for mxfp4_release, MXFP8 for the nvfp4_modelopt KDA/shared GEMMs, MXFP4 indexer Q and
keys, FP8 MLA KV and FP32 KDA state. Exact top-k (DeepSelect) is used throughout. All weights are random
(timing only).

## Runs (Hecate, runner2 queue, pool nq4)

The runner2 holds serve only `POOLS=nq4`, so the nq1 tasks I staged first never started. I withdrew them to
`$W/mb-ops1/withdrawn/` and ran each nq4 task as 4 independent single-node / single-GPU benches.

| task | hold job | hosts | what |
|---|---|---|---|
| `19-nq4-mbops1-all4-m80.sh` | 1137218 | hecate0370-0373 | first full pass. 46 cases failed: fractional M/rows from AL-scaled shapes; MXFP8 `_scaled_mm` with N not /16; one 16K-row sparse MLA overflowed its workspace. |
| `19-nq4-mbops1-all4r2-m25.sh` | 1137218 | hecate0370-0373 | full rerun with fixes: ceil(M); MXFP8 N padded to /16 for cuBLASLt plus CuTe DSL at any M for those shapes; 1 GiB MLA workspace; preallocated KV counters; warm-L2 parts. 550/550 cases. |
| `19-nq4-mbops1-verify-m15.sh` | 1137240 | hecate0217 | follow-up checks: (a) BF16 cuBLAS at unaligned N vs N padded to /8; (b) the large prefill GEMMs repeated 6x with idle gaps and clocks logged. |

The second run reproduces the first to within a median ratio of 1.000 (p5-p95 0.98-1.01), except for the big
compute-bound GEMMs. Those vary by +-15-25% across GPUs and runs (gemm-003: 3.71 / 5.00 / 3.85-3.93 ms), so their
measured_us is the 6x repeat median.

Raw data on Hecate is under `$W/mb-ops1/results/{gemm,attn,draft}-1118030{1,9}/` and `verify-11180314/`, with per-GPU
`rows.g*.jsonl` and `g*.log` plus `inner-<host>.log`. A copy is in `raw/`.
- Scripts are in `src/`: `mbc.py`, `gemm_bench.py`, `attn_bench.py`, `draft_bench.py`, `verify_bench.py`, `inner.sh`,
  `task_*.sh`.
- `collect.py`, `patch_verify.py` and `finalize.py` rebuild `rows.json` and `summary.md`.

`rows.json` holds one row per case with these fields:
- case, topology, ckpts, op_ids, shape, dtype;
- measured_us, t_sol_us, t_ach_us, derate = measured / t_ach;
- method, priority, count_per_step, excess_us_per_step;
- kernel_debug (derate > 2), hypothesis, notes (per-backend times, parts, correctness).

## Results vs model (derate = measured / t_ach)

| family | n | median derate | time-weighted (sum meas / sum t_ach) | max | > 2.0 |
|---|---|---|---|---|---|
| gemm bf16, M > 1024 (prefill) | 75 | 1.21 | 1.19 | 18.5 (unaligned N) | 17 |
| gemm fp8 (MXFP8), M > 1024 | 19 | 1.26 | 1.25 | 7.2 (unaligned N) | 5 |
| gemm bf16, M <= 64 (decode) | 138 | 2.23 | 1.37 | 110 (tiny N) | 82 |
| gemm bf16, 64 < M <= 1024 | 129 | 2.84 | 1.65 | 51 | 88 |
| gemm fp8, M <= 1024 | 56 | 2.6-4.1 | 1.95-2.48 | 18.5 | 44 |
| gemm bf16 bmm (w_uk/w_uv) | 23 | 1.2 (prefill) / 3-4 (decode) | 1.22 / 2.7-4.3 | 26 | 14 |
| dsa idx+top-k prefill | 5 | 1.85 | 1.85 | 1.90 | 0 |
| dsa idx+top-k decode (stock kernels) | 12 | 7.6 | 6.3 | 54 | 12 |
| sparse_mla prefill rows (>= 2048) | 5 | 3.60 | 3.70 | 3.86 | 5 |
| sparse_mla decode rows | 12 | 6.9 | 5.1 | 37 | 12 |
| kda prefill (FlashKDA) | 4 | 1.71 | 1.98 | 2.77 | 1 |
| kda decode (sdec PreparedKda) | 15 | 4.3 | 3.7 | 10.0 | 15 |
| attnres | 27 | 5.2 (1.2-1.6 at >= 2048 rows) | 1.46 | 483 (rows=1) | 18 |
| drafter DS.markov | 15 | 4.2 | 2.25 | 11.4 | 10 |
| drafter DS.attn | 15 | 8.3 | 6.0 | 38 | 15 |

The P0 case table and the largest excess-per-step list are in `summary.md`. The P0 GEMMs at prefill shapes land at
0.98-1.23x (MXFP8 1.06-1.40x). The P0 decode GEMMs land at 1.1-1.75x, except `MOE.latent_down/up` at M = 4
(2.5-2.7x, fixed floor).

### KERNEL-DEBUG (derate > 2.0) by root cause

1. **Unaligned-N BF16 GEMMs (verified fix): KDA.in_proj at N = 6284 (P1, D7-8) and N = 3206 (D1).**
   - Cause: the BF16 row stride (12568 B / 6412 B) is not 16 B aligned, so cuBLAS takes a non-TMA kernel at about 0.55 PF.
   - Evidence: padding the weight to N = 6288 / 3208 gives 6.0x at M = 16384 (2516 -> 417 us) and 2-4.5x at M <= 2048.
   - Effect: gemm-109 (P1, 69 layers) would otherwise be the largest excess in the lane (+149 ms/step).
   - Action: pad the packed in_proj shard to a multiple of 8 columns, or check that production already does (the sdec
     decode packs to PITCH = 3216).
   - N = 12440 is aligned and runs at 3.7 PF.
2. **Sparse MLA prefill rows (trtllm-gen), 3.3-3.9x. Biggest real excess: P3-16 / P9 +44-45 ms per step, P4-2 +21 ms.**
   - The kernel sustains 3.7-4.1 PFLOPS on 128 padded heads (about 2.9 PF useful), and cold == warm, so it is not
     HBM-bound.
   - Hypothesis: the head padding 96 -> 128 costs 1.33x. The rest is the per-token decode formulation: one request
     per token, every row re-gathers its 2048 x 576 B KV with no reuse across rows, plus the softmax/PV pipeline.
   - The model's FP8 MMA rate assumes a prefill-style sparse kernel.
   - Next: profile `trtllm-gen` vs a prefill sparse-MLA (FlashMLA sparse / CuTe) at 16K rows, and check MUFU vs MMA
     issue in ncu.
3. **Fixed per-kernel floor at decode shapes (most derate > 2 rows with t_ach < 3 us).**
   - Measured floors:
     - cuBLAS GEMM: 4-10 us.
     - AttnRes: about 4-6 us.
     - trtllm-gen MLA: about 10-12 us.
     - DeepSelect top-k on a few long rows: 10-37 us.
     - DS.attn (FlashInfer / FA paged): about 22 us.
     - KDA verify: about 8-11 us.
   - The model's t_ach has no per-kernel floor; only the 0.78 us gap appears in the scoring.
   - Examples: dsa_indexer-434/557 (4 rows: 44 us vs 0.8), sparse_mla-442/564 (12.5 vs 0.4), MOE.router and
     KDA.f_b_proj GEMMs (5-6 us vs < 0.2).
   - This is a model gap more than a kernel bug: either add a floor term (about 4 us cold for a single kernel) or
     credit the fusion that removes these launches (sdec fuses most of them).
4. **KDA decode via `PreparedKda`, 3.1-4.3x at B >= 32 and 7.5-10x at B <= 8.**
   - The bench path writes every token's FP32 checkpoint state: 1 read + 4 writes of 64 KiB per head per request,
     about 5x the model's state bytes. At B128 h24 that is 1.0 GB in 108 us, about 9.4 TB/s effective.
   - Production RecoverSSM writes records instead of states and is not covered here.
   - Cross-check from the release c512 step profile (D3, B50 q4, `$W/dsrel/results/prof-6a14e2f/c512-relfa/breakdown.txt`):
     kda is 33.2 us per layer (may include in-proj) against about 13.7 us interpolated t_ach, i.e. about 2.4x.
5. **DSA indexer decode (stock paged DeepGEMM + DeepSelect), 5-9x.**
   - Hypothesis: the paged kernel runs next_n = 1 per row, so each of a request's q = 4 rows streams the keys
     (4x the model's 68 B/key scan).
   - DeepSelect adds 10-37 us per call.
   - Production sdec uses its own fused indexer. Cross-check from the c512 profile: indexer 29 + top-k 16.7 us per
     layer against about 10.3 us interpolated t_ach, i.e. about 4.4x. This agrees in direction with the stock-kernel
     numbers.
6. **KDA prefill at 96 heads (P3-16 / P9), 2.8x.**
   - The v-split FlashKDA build does not fit 96 heads x 3 sequences, so the stock path runs, and the segscan planner
     declines.
   - The 24 / 48 / 12-head cases run 1.4-1.9x.
7. **Indexer prefill (1.84-1.90x, just under the flag).**
   - The logits run at about 6.9 PF FP4, close to the model.
   - DeepSelect top-2048 over 78K BF16 columns adds about 75% (1.09 ms at 16K rows, about 2.4 TB/s over the logits).
     The model treats this as nearly free.
8. **Small-M MXFP8 GEMMs (2.6-4x) and BF16 64 < M <= 1024 (2.8x median).**
   - Weight-streaming kernels run below the HBM roofline: cuBLASLt split-K tiles and CuTe DSL.
   - The low-latency CuTe table did not apply to any of these sharded shapes in this tree.
9. **Batched w_uk/w_uv `torch.bmm` at decode rows (3-4x, up to 26x at 4 rows).**
   - These are per-head tiny GEMMs (K = 128 or N = 128). Production fuses them into the MLA query/output kernels, so
     treat this as a model-vs-stock gap, not a production one.
10. **DS.attn (FlashInfer paged prefill), 4.4-8.3x.**
    - About 88 us for 268 MB of window KV (3 TB/s) and a 22 us floor.
    - `vllm_flash_attn` varlen is 2.5x slower at large B and equal at small B.
    - A decode-style GQA kernel (or the release drafter's tuned path) is needed. These are P2.

### Unrun or partial cases, and coverage caveats

- These cases are not benched (P2, tiny, or no standalone kernel):
  - attnres `G.embed` and `DS.aux_capture` (16 cases);
  - drafter `DS.confidence`, `F.rejection`, `DS.norm_ar`, `F.norm`, `DS.ctx_kv_insert`, `DS.hidden_norm`, `DS.embed`
    and `DS.silu` (64 cases);
  - `fused_epilogue` (52, P2; not in this lane's list).
- Several measurements cover only the core kernel and leave out ops the model row includes:
  - sparse_mla: `DSA.prologue` (the KV/indexer-key insert) and `gate_mul` are not timed.
  - indexer: `idx_q_quant` is not timed.
  - KDA prefill: conv, gate_beta_norm, o_norm and the checkpoint export are not timed (core recurrence only).
  - DS.markov: the argmax/epilogue is not timed.
  - These rows' derates are therefore lower bounds.
- Some cases ran stock kernels or a different path than production:
  - Decode attention used stock kernels (DeepGEMM paged, trtllm-gen) and the `PreparedKda` verify path, not the
    production sdec fused indexer / MlaQuery / RecoverSSM kernels.
  - For the DCP partial MLA cases, production uses prims_ts; these were benched on trtllm-gen with LSE.
  - The DCP top-k merge and exchange are not timed (comm lane).
- The MXFP8 shapes with N not /16 (12440, 6284, 3206) cannot run cuBLASLt `_scaled_mm` as is. They were timed with N
  padded to /16 and with CuTe DSL; the notes give both.
- Fractional M/rows (AL-scaled drafter, lm_head and `F.attnres_final` cases) were rounded up to whole rows.


## Review fixes (adversarial review, 2026-11-18)

Applied to `rows.json` by a one-off script; every changed row carries a `review_fix` object with the old values
(`derate_old`, `t_ach_us_bundle` / `measured_unpadded_us`) and the reason. The tables above and `summary.md` predate
these fixes.

1. **Derate was computed against model ops the bench did not time (38 rows).** For `sparse_mla` (17),
   `dsa_indexer_topk` (17) and KDA prefill (4) the matrix case bundles several model ops and its `t_ach_us` is their
   sum, but the bench times only the core kernel. `t_sol_us` / `t_ach_us` are now re-based to the timed ops only,
   from `out/optables.json`:
   - sparse_mla: `DSA.sparse_mla` only (prologue and gate_mul removed). P1 3.26 -> 3.71, P2 3.31 -> 3.56,
     P3-16 3.75 -> 3.86, D1/D8 5.5 -> 6.1.
   - indexer: `DSA.idx_score + DSA.topk` (idx_q_quant removed); changes are under 1% except D1/D8 (+1%).
   - KDA prefill: `KDA.core_prefill` only (conv, gate_beta_norm, o_norm removed). P1 1.37 -> 1.47, P2 1.55 -> 1.75,
     P4-2/P6 1.86 -> **2.34 (now KERNEL-DEBUG)**, P3-16/P9 2.77 -> **4.19**.
   - KDA decode rows are unchanged: `PreparedKda` times conv + recurrence + norm, which is the bundle.
2. **Unaligned-N KDA.in_proj is a bench artifact (7 BF16 rows).** The release tree pads the packed in_proj shard
   (`vllm/models/kimi_k3/nvidia/kda.py`, `in_proj_padding = -local_output_size % 16`, 128 for FP8_PB_WO), so production
   never runs N = 6284 / 3206. The prefill lane confirms it in-step: P1 KDA.in_proj is 29.2 ms / 69 layers = 423 us
   per layer, derate 1.05, not 2566 us. `measured_us` is now the padded-N run (bench padded 3206 to 3208, production
   to 3216; the difference is negligible). gemm-109 P1 6.35 -> 1.03, gemm-212 D7-8 6.23 -> 1.36, gemm-252 D1/D8
   6.12 -> 1.35; the small-M rows (gemm-270/298/326/414) stay at 2.1-3.0 from the per-kernel floor. **The "+149 ms
   per step on P1" claim and KERNEL-DEBUG cause #1 do not apply to production; drop the "pad the in_proj shard" fix.**

KERNEL-DEBUG count after fixes: 336 of 550 (was 338).
