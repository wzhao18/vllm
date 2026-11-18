# Microbench lane: prefill (P1, P2, P3-16, P4-4, P4-2)

Prefill step timing for one prefill instance under `vllm bench forward`. Each run covers the whole forward: 93 target layers plus the DSpark drafter. A torch-profiler trace gives the per-layer and per-op split. Results are compared with the SOL model (`out/optables.json` regime `prefill F16384 L78K`). The model is re-evaluated at the exact measured shapes in `analysis/model_tables.json` (`tools/gen_tables.py`, which calls `prefill.run_point`).

## What ran

- **Harness.** `tools/pf_bench.py` wraps the r3pf P1 harness (`bench_prefill_step.py` / `bench_dep_step.py`) and adds:
  - eager timing of the whole forward, with the max over ranks;
  - the nq_step MX-aware random indexer-key fill;
  - a pin of the vendored Rubin MegaMoE to MXFP8 activations (`PF_MOE_QUANT=mxfp8`).

  The launch path is `tools/pf_run.sh` to `tools/inner.sh`, one srun step on the task's own hosts. Bench args are in `tools/pf_common.sh`. They come from the release prefill roles (`e2e/renders/dsrel-c48-1p1d-xdr`, `dsrel-c512-xdr`) with serving-only flags dropped.
- **Code.** Release draft sweep tree `dsrel/trees/rel-1767cf7`, which is 6a14e2fada `wzhao/k31-dspark-release` over vllm-jh-242fb49cd4.
  - P4 used PR #20 (`juntianl/k31vr-tp-dsadp` 83cd6d8e12). It was staged over vllm-jh-242fb49cd4 as `mb/prefill/ov/pr20-83cd6d8` (`tools/stage_pr20.sh`). Settings: `VLLM_K3_TP_DSA_DP=1` and `VLLM_K3_TP_DSA_DP_KV=0` (replicated DSA KV, because the bench has no per-owner region manager). The bench has no scheduler, so the owner is the batch index mod TP: 4 requests, one per owner at TP4 and two per owner at TP2. The log confirms `DSA-DP: DSA outputs pushed over symmetric memory`.
- **Image and versions.** Image `k31vr-kmc_kmr-sweep-freeze_p242fb49cd4-...-fi0.7.3.dev20261031_mc0.3.14_base3f1bcfd82b_r1`. FlashInfer 0.7.3.dev20261031 (`/opt/k3`). Caches: the release CuTe pretrace `cache/cute-pretrace/k31vr-kmc-kmr-sweep-freeze-r1`.
- **MegaMoE.** Every number here uses the vLLM-vendored `vllm/models/kimi_k3/nvidia/megamoe_rubin`, not the newer FlashInfer 6e438852b build.
  - Knob records: P1 EP8 i2048 `moe4/results/fi107/knobs_xo_ep8_mxfp4_mxfp8_i2048.json`, P4-4 EP16 i4096.
  - P2, P4-2 and P3 reuse a record tuned at 4096 tokens/rank for 8192 and 16384 (logged: "knob record tuned at max_tokens 4096 used for ...").
- **Precision.** All bench-only settings are logged in each run:
  - FP8 MLA KV (`kv-cache-dtype fp8`), MXFP4 indexer KV (`indexer_kv_dtype mxfp4`), FP32 KDA state (`mamba-ssm-cache-dtype float32`);
  - BF16 linears; exact top-k (`trtllm_gvr2`);
  - MoE `activation mxfp8`, with the logged quant kind asserted (`Rubin MegaMoE: mxfp4_mxfp8 ... buckets`);
  - `--load-format dummy` with the real K3.1-Pro config view `e2e/model-view/kimi-k3.1-pro-k3type`, `--routing balanced`, `--kv-len-jitter 0`;
  - drafter: release DSpark e4e6e626 (`jh/noquant/cfg2/deps/kimi-k3.1-pro-dspark-e4e6e626`), FLASH_ATTN, k=7 (P1) or k=3.
- **Release-policy arms.** `PF_MOE_QUANT=release` is the deployed MegaMoE, which switches to NVFP4 W4A4 above 512 tok/rank. That is the model's O4 setting, not the mxfp4_release headline.
- **Cases.** One case per DP rank / attention group:
  - `pf16k_L78k`: 3 requests (2x5461 + 5462 fresh) at KV length 78,360 (prefix-cache hit), F = 16,384.
  - P4 uses 4x4096 at 78,360, so every owner gets work. The model is re-run with f_req = 4096.
  - Plus a fresh 16K chunk, and `pf32k_L78k` (production MNBT) for P2.
  - P3-16 at F16K needed a pinned 14-16 GiB KV pool and max-num-seqs 4. F8K was also measured, with a profile.

| task (queue/done) | job | hosts | runs (raw dirs under `$W/mb/prefill/results/`) |
|---|---|---|---|
| 19-nq4-mbprefill-p1-m65 | 1137222 | hecate0077-0080 | p1/p1_mx (+profile, 0077-78), p1/p1_rel (0079-80) |
| 20-nq4-mbprefill-p2-m75 | 1118808 | hecate0344,0345,0350,0351 | p2/p2_mx (+profile; F16K ok, F32K OOM), p2/p2_rel (F16K ok) |
| 18-nq4-mbprefill-p2b-m50 | 1137218 | hecate0370-0373 | p2b/p2_mx32, p2b/p2_rel32 (bench GMU 0.85) |
| 21-nq4-mbprefill-p4-m80 | 1137221 | hecate0073-0076 | p4/p4_4_mx (+profile), p4/p4_2_mx |
| 15-nq4-mbprefill-p3e-m60 | 1137218 | hecate0370-0373 | p3e/p3_rel (F16K NVFP4 ok; MXFP8 arm OOM) |
| 14-nq4-mbprefill-p3g-m50 | 1137118 | hecate0218,0230-0232 | p3g/p3_mx (F8K +profile), p3g/p3_rel |
| 14-nq4-mbprefill-p3h-m40 | 1137223 | hecate0221,0222,0224,0225 | p3h/p3_mx16 (F16K MXFP8) |

The failed P3 attempts (p3, p3b, p3c, p3d, p3f) are in `$W/mb/prefill/results/status.txt`:

- p3 failed because a 1M max-model-len does not fit the TP1 KV pool.
- p3b and p3e-mx hit OOM in the sparse-MLA output, because 225 GiB of weights + the KV pool + the bench snapshot left no room.
- p3c set GMU 0.86, which gives a negative KV pool.
- p3d and p3f failed the serving warmup with too few KV blocks.

Profile traces are `<run>.prof/<case>.rank<r>.trace.json.gz`. Per-component attribution is in `analysis/*.attr.txt` and per-kernel module mapping in `analysis/*.opmap.json`.

## Step results (fresh tok/s/GPU vs model achievable)

| topology | case | measured step | model ach step | derate | tok/s/GPU meas | model ach | release NVFP4-MoE arm |
|---|---|---|---|---|---|---|---|
| P1 TP8+DCP8 EP8 | F16K L78K | 228.2 ms | 218.2 | 1.05 | 8.97K | 9.38K (best F64K: 9.78K) | 203.4 ms, 10.07K |
| P2 DP4xTP4=DCP4 EP16 | F16K L78K | 350.7 ms | 282.2 | 1.24 | 11.68K | 14.51K | 307.7 ms, 13.31K |
| P2 | F32K L78K (prod MNBT) | 653.1 ms | 551.3 | 1.18 | 12.54K | 14.86K (best F64K: 15.04K) | 579.5 ms, 14.14K |
| P3-16 DEP16 | F16K L78K | 1179.8 ms | 960.4 | 1.23 | 13.89K | 17.06K | 1063.3 ms, 15.41K |
| P3-16 | F8K L78K | 610.9 ms | 486.5 | 1.26 | 13.41K | 16.84K | 531.9 ms, 15.40K |
| P4-4 TP4xDP4 DSA-DP EP16 (PR #20) | F16K (4x4096) L78K | 330.5 ms | 269.1 | 1.23 | 12.39K | 15.22K | - |
| P4-2 TP2xDP8 DSA-DP EP16 (PR #20) | F16K (4x4096) L78K | 623.0 ms | 500.1 | 1.25 | 13.15K | 16.38K | - |

**Ranking.** Model (achievable, F16K) is P3-16 > P4-2 > P4-4 > P2 > P1. Measured (MXFP8 MoE, F16K) is P3-16 13.9K > P4-2 13.2K > P4-4 12.4K > P2 11.7K > P1 9.0K, the same order. P1 is near the model (1.05). Every layout without DCP8 lands at 1.18-1.26.

## Per-layer split (ms per layer incl. gaps, measured / model achievable)

| run | layer 0 (KDA + dense) | KDA + MoE (mean of 68) | DSA + MoE (mean of 24) | drafter |
|---|---|---|---|---|
| P1 F16K | 2.299 / 2.415 (0.95) | 2.210 / 2.307 (0.96) | 2.583 / 2.428 (1.06) | 0.45 |
| P2 F16K | 3.444 / 3.191 (1.08) | 3.437 / 2.992 (1.15) | 4.213 / 3.125 (1.35) | 0.49 |
| P4-4 F16K | 3.551 / 3.150 (1.13) | 3.362 / 2.951 (1.14) | 3.911 / 2.692 (1.45) | 0.54 |
| P3-16 F8K | 6.500 / 5.629 (1.15) | 6.160 / 5.220 (1.18) | 7.556 / 5.213 (1.45) | 0.79 |

## Derating per op family (critical-path exclusive time; concurrent streams counted once)

| family | P1 | P2 | P4-4 | P3-16 (F8K) |
|---|---|---|---|---|
| KDA attention + its AR / SP comm | 1.19 | 1.10 | 1.10 | 1.21 |
| DSA attention + its comm | 1.34 | 1.39 | 1.73 | 1.81 |
| MoE block (router, latent, shared, MegaMoE) | 0.88 | 1.18 | 1.17 | 1.15 |
| AttnRes | 0.17 (*) | 1.28 | 1.32 | 1.27 |
| BF16 big GEMMs (KDA in_proj, g_proj, dense, o_proj on TP1) | 0.96-1.05 | 0.96-1.08 | 0.98-1.09 | 0.98-1.07 |
| MegaMoE kernel (busy, overlapped with shared experts) | 1.64 | 1.38 | 1.36 | 1.28 |

(*) P1 AttnRes runs on the 1/8 token shard and then all-gathers (`_push_shard_kernel`, 22.9 ms/step, counted in the attention blocks). The model prices AttnRes on all 16,384 rows under TP8. Model term to fix: rows/N plus an AG.

The GEMM-RS fused o_proj (`GemmRsAr`) costs 0.55-0.71 of the model's o_proj + all-reduce. The model prices a two-shot AR, while the implementation does RS in the GEMM and AG as a separate push.

## KERNEL-DEBUG (derate > 2.0), first hypotheses from the profiles

| op | derate | measured (P1 / P2 / P4-4 / P3) | hypothesis |
|---|---|---|---|
| DSA.sparse_mla (`fmhaSm107a...QkvE4m3...HQk576HV512HVPerCta256...SparseP1...Q64Kv128...2Cta`) | 3.2-3.3 everywhere | 6.5 / 12.4 / 12.7 / 25.4 ms per step (259-530 us/layer) | ~3.4 PF of the FP8 MMA rate, not exp-bound (the model's MUFU term is ~50 us/layer at P1). HVPerCta256 splits V512 in two, so each CTA may redo QK^T (576-wide) per V half. A Q64 tile per CTA re-gathers its own 2048 KV rows. Check with ncu: tensor pipe %, dram bytes vs 576 B x 2048 x rows/64. |
| DSA.topk (`trtllm_gvr ... topk_decode_self_sampling GvrMainKernel`) | 3.6-5.6 | 3.2 / 4.6 / 4.0 / 8.7 ms | Decode-oriented per-row top-2048 over ~78K FP32 logits (no BF16 scratch). Its multi-pass sampling is priced in the model as one scratch write and read. |
| DSA.prologue+aux | 21-53 | 5.2 / 8.5 / 3.5 / 6.6 ms | A CuTe kernel named just `Kernel` in mla_attn (2.4-5.8 ms), plus `shardq_query_quant`, `shardq_kv_insert`, DtoD copies and `_pad_query_heads` (P4). The model prices only RoPE/quant. The unnamed kernel is likely top-k index to paged-slot translation or the ShardQ context gather; it needs a name. |
| DSA.gate_mul, idx_q_quant+insert | 19-55 | 0.5-1.7 ms | Separate elementwise / quant-insert kernels. The model treats them as fused (no HBM traffic). |
| KDA.conv (`k3conv::qkv_conv_kernel`) | 4.3-5.0 | 2.3 / 4.5 / 4.7 / 8.2 ms | The model has no activation bytes for conv (fused_in_sol). The kernel reads and writes q/k/v, about 300 MB/layer/GPU at P1, so it runs at about the HBM roofline. Fuse conv into the in_proj epilogue or the flash_kda prepare. |
| KDA.o_norm (`_kda_rmsnorm_gated`) | 12 | 1.4 / 2.5 / 2.5 / 5.2 ms | Same cause: a separate gated-RMSNorm pass the model assumes is fused. |
| KDA.core_prefill (`flash_kda_fwd_recurrence` + `_flash_kda_fwd_prepare`) | 1.4 / 1.8 / 2.0 / 2.4 | 13.8 / 20.2 / 17.2 / 27.7 ms | The prepare kernel is a separate pass (43 us/layer at P1). The recurrence is worst at TP1 (96 heads per GPU, 2.7K-token requests: chunk-serial walk). |
| KDA.f_b_proj | 2.1-3.5 | 0.9-2.2 ms | Small-N GEMM with a poor tile (nvjet 128x192). |
| MOE.shared_down (+ shared_act 60-75x) | 3.8-4.0 | 18.1 / 34.8 / 34.9 ms | Shared experts run on a side stream concurrently with the persistent MegaMoE kernel and get few SMs; `situ_and_mul` is unfused (model: fused). The MoE block as a whole is still 0.88-1.18 on the critical path. |
| MOE.router / MOE.topk / MOE.latent_norm | 2.2-2.8 / 117-146 / 22-40 | 1.9-3.1 / 1.1-1.8 / 0.8-0.9 ms | The router is a small-N (896) GEMM. topk and norm are priced as compute only (no logits / activation reads) in the model. |
| DS.drafter | 5.8-8.8 | 2.1-3.5 ms | Drafter ctx AllGather (NCCL ring, 0.5 ms) plus fc and the window attention in prefill. The model prices 0.24-0.6 ms. |
| DENSE.act | 13.5 | 0.1-0.3 ms | Unfused activation (model: fused). |

Rows whose derate comes from a model term priced as fused or with no bytes (conv, o_norm, gate_mul, act, topk, latent_norm) are tagged KERNEL-DEBUG in rows.json as instructed. The likely fix is a fusion, or a model term that charges the activation pass.

## Not run / caveats

- **P9 (DSA-CP: DEP16 + ShardQ DCP8 on DSA layers).** Needs implementation; no DSA-CP code path exists in the release tree or in PR #20/#23.
- **P6 (TP2 KDA + DSA-DP, EP64 + O2 dedup).** Out of scope (EP > 16) and needs implementation (O2 dedup / pre-combine).
- **P4 DSA-DP is benched without the scheduler.** The owner is the batch index mod TP and the DSA KV is replicated (`VLLM_K3_TP_DSA_DP_KV=0`). Per-owner compute is the same as production, but owner imbalance and region KV are not measured.
- **P3-16 F16K is not bench-feasible at the release memory settings.** 225 GiB of weights at TP1 (the model's memory model says 205 GB), plus a 16K-token TP1 activation peak (~30 GiB), does not fit next to a 0.92 KV pool. Measured with a pinned 14 GiB KV pool and max-num-seqs 4 instead. P3 took 7 submissions, over the 3-attempt guideline; each failure was a different bench sizing issue (listed above).
- **P2 F32K at the release GMU 0.92 OOMs in the bench.** The bench snapshots state on top of the KV pool. The p2/p2_rel arm will also OOM at 32K and hold its block until its 30-min timeout. Ranks that OOM leave the others hanging in collectives instead of exiting, so the first P2 arm sat until its 50-min timeout.
- **Model best chunk F64K not measured.** Production MNBT is 16K (P1) and 32K (P2). The grid gain from F16K to F64K is +4% (P1) and +4% (P2).
- **Dummy weights with `--routing balanced`.** Real-weight routing skew is not included.


## Review fixes (adversarial review, 2026-11-18)

1. **Fresh-chunk step rows were compared with the prefix-hit model step (4 rows fixed).** `pf16k_fresh` (P1, P2: 1 x
   16384 fresh; P4-4, P4-2: 4 x 4096 fresh) used the L78K model step as `t_ach`. The model was re-evaluated at the
   fresh shapes with `prefill.run_point(..., F=16384, L=f_req, f_req)` (mxfp4_release); the row keeps the old values
   under `review_fix`. Derates: P1 1.019 -> 1.045, P2 1.232 -> 1.246, P4-4 1.188 -> 1.230, P4-2 1.215 -> 1.261.
   The headline L78K rows were already correct (the model reproduces 218,245 us for P1 F16384 f5456 L78356).
2. **MegaMoE derate is mostly the vendored kernel version, not side-stream SM contention (annotation, no value
   change).** The vendored `megamoe_rubin` knob records logged in these runs give a standalone p50 of 742 us at EP8 /
   2048 tok/rank (P1) and 1096 us at EP16 / 4096 (P2, P4, P3 buckets). The newest FlashInfer 6e438852b (moe lane, B10)
   runs the same shapes in 511 us and 901 us. In-step P1 MegaMoE busy is 69.73 ms / 92 = 758 us per layer, i.e. the
   vendored standalone time plus ~2%. So the "MegaMoE kernel 1.64 / 1.38 / 1.36 / 1.28" family derates are about
   1.45x (EP8) and 1.2x (EP16) from the kernel build; with B10 wired in, P1 would drop by roughly
   (742 - 511) x 92 = ~21 ms per step (projection, not measured). Read the MoE rows as "vendored MegaMoE", not as the
   moe lane's numbers.
3. **Release-policy (NVFP4 W4A4 MoE) step rows** are compared with the mxfp4_release (MXFP8-activation) model step.
   Their derates (0.93-1.11) are not like-for-like and should not be used as model derates; they are kept only as
   the deployed-policy reference.
