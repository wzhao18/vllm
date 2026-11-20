# Lane gap512: decomposing the C512 hybrid gap (D3 decode, P2 prefill)

Measured 2026-11-18 on Hecate VR200 through the queue/runner2 system (4-node `nq4` tasks, srun --overlap in runner2 holds).
Each C512 decode step and prefill forward was profiled with torch-profiler (`vllm bench forward --profile-dir`, 3 replays,
all 16 ranks) on the release tree. Every kernel is mapped to perf-model op ids (`tools/gap_parse.py`, `tools/pf_parse.py`),
and op groups are ranked by **exposed - achievable**. Model numbers come from `perf-model` at the bench shapes
(`tools/model_rows.py` -> `parsed/model_rows.json`, mxfp4_release, k=3, L = the bench's mean context).

## What ran

| task (queue name) | job / hosts | what | result |
|---|---|---|---|
| `20-nq4-mbgap512-dprof-m80.sh` (`tools/dprof.sh`) | 1137218 hecate0370-0373 | D3 decode, rows 128/200/256/368 (B 32/50/64/92 per DP rank), agentx contexts (mean 190K) | B32/B50/B64 ok. B92 OOM in the bench's per-case state snapshot (30.9 GiB, 9 GiB free): a bench artifact |
| `20-nq4-mbgap512-dprof2-m45.sh` (`tools/dprof2.sh` -> `dprof_v.sh`) | 1137216 hecate0046-0049 | B80 alone (agentx contexts); B92 alone with contexts capped to 12.0M tokens per DP rank (scale 0.686, mean 130K) | both ok |
| `21-nq4-mbgap512-pprof-m80.sh` (`tools/pprof.sh` -> `pf_run1.sh` + `pf_cfg.py`) | 1137222 hecate0077-0080 | P2 prefill role of nq-c512-xdr: per DP rank, 12K chunk at 78K context, 2x6K at 78K, 8K, 16K, and a fresh 12K | all ok |

**Versions.** Code is the release-tree overlay `$W/dsrel/ov/rel-1767cf7` (staged by `mkov.sh`; SOURCE_REV base 6a14e2fad;
vLLM 0.32.0rc2.dev0+g242fb49cd4.k31vr). Image `vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh`, FlashInfer
0.7.3.dev20261031, torch 2.16.0.dev20261009+cu134.
- **MegaMoE in the step**: the vendored `deep_gemm::sm100_fp8_fp4_mega_moe_situ_impl` (image deep_gemm with the vLLM
  sm_107 situ patch, `moe-backend deep_gemm_mega_moe`). It is **not** FlashInfer 6e438852b.
- **Draft**: the release DSpark draft `kimi-k3.1-pro-dspark-e4e6e626` on FLASH_ATTN.
- **Weights**: dummy (`--load-format dummy`), with routing `balanced`.

**Precision.** FP8 MLA KV (576 B), MXFP4 indexer keys, FP32 KDA state, and BF16 linears. MoE activations are MXFP8. The
`quantization-config moe.activation=mxfp8` is in the args, and every MegaMoE kernel in decode and prefill is the
`fp8_fp4` variant (FP8 act x FP4 weight). That includes prefill at 2048-4096 tokens/rank, so the NVFP4 switch is not taken.
Top-k is exact (DCP top-k + merge, and deep_select / topk_select_bf16 in prefill).

**Harness.**
- Decode uses the dsrel nq step bench (`dsrel/tools/nq_run1.sh` -> `nq_step.py`), with the production decode role of
  `jh/noquant/cfg/nq-c512-xdr.yaml` and `decode_backend latency_specialized`.
- Prefill uses the same `nq_inner.sh` / `nq_step.py` with the **prefill** role (`pf_cfg.py`). That role has kv_both, so
  ShardQ is on, along with `VLLM_K3_KDA_SEGSCAN=1` and `VLLM_KIMI_K3_VEC_SITU=1`. Serving, prefix-cache and connector args
  are dropped. The bench timing is a CUDA-graph replay median, the max over ranks, of 20 x 5.
- Neither number includes the target LM head, rejection or CPU, and the model rows are compared without them.

**Raw outputs on Hecate** (`W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr`):
- `$W/gap512/dec-rel1767/c512-relfa/`
- `$W/gap512/dec-rel1767-b80/c512-relfa/`
- `$W/gap512/dec-rel1767-b92cap/c512-relfa/`
- `$W/gap512/pre-rel1767/c512p/`

Each holds `agentx.log` / `pf.log`, `prof-agentx/` / `prof/` (16 ranks x trace.json.gz + kernels.json), and
`breakdown.txt` (kb_parse).

## Method

1. One replay's forward kernels are cut at the AttnRes mix kernels (one before each attention and MLP sublayer).
2. Each segment is typed as KDA, DSA, MoE, dense, or tail (DSpark propose and final).
3. GEMMs are assigned to ops by their position in the segment. A cuBLAS split-K reduce kernel goes with its GEMM.
4. Same-stream PDL overlap is clipped: a kernel's time starts when its predecessor on the stream ends.
5. **measured** = the summed kernel time per step, as the median of 3 replays and then the median of 16 ranks.
6. **exposed** = an additive share of the busy timeline. Kernels overlapping on other streams split the time evenly, and
   spin-wait kernels only get idle instants. The sum of the exposed values equals the step.
7. **derate** = measured / t_ach, as in the lane spec.
8. The rank order uses exposed - t_ach, which is the op's contribution to the step gap.

`rows.json` holds the following rows:
- per case, one STEP row
- per case, the family rows (decode)
- per case, the op-group rows: op_ids, measured, exposed, t_sol, t_ach, derate, kernels and calls
- MegaMoE arm rows: in-step against the standalone vendored (B7) and newest (B10) rows

## Results vs model

### Decode step (D3, us per step)

| case (B per DP rank) | measured | model ach | model SOL | measured/ach |
|---|---|---|---|---|
| B32 (r 8) | 24,539 | 14,706 | 9,733 | 1.67 |
| B50 (r 12.5) | 28,936 | 15,759 | 10,526 | 1.84 |
| B64 (r 16) | 30,756 | 16,576 | 11,365 | 1.86 |
| B80 (r 20) | 33,017 | 17,700 | 12,341 | 1.87 |
| B92 (r 23), contexts x0.686 | 32,741 | 18,437 | 12,930 | 1.78 |

### Decode gap by op family at B50 (B32 / B64 / B80 gap in brackets)

The gap is exposed - ach, in ms per step.

| family | measured busy | ach | derate | gap ms |
|---|---|---|---|---|
| BF16 dense GEMMs (cuBLAS nvjet + CuTe split-K + split-K reduce) | 9.33 | 4.11 | 2.27 | **4.15** (2.87 / 4.83 / 4.75) |
| MegaMoE (vendored deep_gemm, incl. P11/P12) | 10.77 | 8.17 | 1.32 | **2.60** (2.35 / 2.40 / 2.84) |
| DSA DCP exact top-k (pp hist/window/win + merge, P8) | 2.60 | 0.19 | 13.6 | **1.85** (1.37 / 1.97 / 2.34) |
| DSA q path + sparse MLA + DCP value (P6/P7b) | 3.41 | 0.57 | 6.0 | 1.61 (1.05 / 1.26 / 1.50) |
| small elementwise (AttnRes mix, SiLU, latent RMSNorm, router top-k, MegaMoE prep) | 1.57 | 0.08 | 19.6 | 1.27 (1.28 / 1.26 / 1.17) |
| KDA core (KdaWide; KdaWS at B80+) | 2.08 | 0.94 | 2.20 | 1.14 (0.87 / 1.45 / 1.61) |
| drafter + final (DSpark k3) | 0.68 | 0.30 | 2.27 | 0.38 |
| SP comm (AG / RS push) | 1.18 | 0.87 | 1.36 | 0.31 |
| indexer logits (paged_mqa_logits) | 0.69 | 0.31 | 2.22 | 0.04 (half hidden under the MlaQuery stream) |

### Prefill forward (P2, per DP rank, us)

| case | measured | model ach | model SOL | measured/ach |
|---|---|---|---|---|
| 12K chunk at 78K context | 266,968 | 215,194 | 171,861 | 1.24 |
| 2 x 6K at 78K | 262,335 | 215,640 | 172,250 | 1.22 |
| 8K at 78K | 188,296 | 144,188 | 115,217 | 1.31 |
| 16K at 78K | 350,006 | 286,088 | 228,398 | 1.22 |
| fresh 12K | 260,105 | 209,515 | 166,497 | 1.24 |

The forward slope is 19.8 ms per 1K chunk tokens (8K -> 16K), against 17.7 in the model. The forward itself is only
1.22-1.31x achievable. The rest of the quoted 1.74x C512 prefill gap is outside the forward: scheduling, Mooncake
recompute, P/D and chunk mix.

### Prefill gap at 12K / 78K (exposed - ach, ms per forward)

| op group | gap | derate |
|---|---|---|
| KDA core: qkv_conv 47 + FlashKDA prepare 64 + 3 recurrence passes 69/69/93 + chain_scan 20 + gated RMSNorm 28 us/layer | **14.3** | 2.07 |
| DSA sparse MLA (fmha gen-phase kernel, 387 us/layer) | **6.6** | 3.3 |
| DSA ShardQ glue/comm: rank-major pack, gathers, query quant, PtoP, multimem barriers | **5.9** | 4.4 |
| DSA top-k (topk_select_bf16, 2 x ~120 us/layer) | 4.8 | 7.0 |
| drafter tail (incl. NCCL ring AllGather 2.4 ms against 20 us modelled) | 3.8 | 23 |
| SP AllGather push (`_push_shard_kernel` 130 us/layer) | 3.0 | 1.30 |
| MoE small ops: SiLU 2.3, router top-k 1.4, latent norm 0.9, prepare 0.6 | 5.3 | 16-125 |

These groups are at or under the model:
- MegaMoE at 1.00 (in-step 762 us/layer at 3072 tok/rank)
- KDA in_proj at 0.98
- shared gate_up at 1.04
- the fused o_proj + RS GemmRsAr at 0.76-0.79, because the model charges the RS separately

## Top 3 decode contributors (KERNEL-DEBUG) with first hypotheses

1. **BF16 dense GEMMs: 4.1-4.8 ms of gap at B50-B80, derate 2.3-3.0.** These are weight-streaming GEMMs with M = 50
   (SP rows) or M = 200-368 (TP4 group rows). Achieved HBM bandwidth at B50, measured per call:
   - KDA in_proj (178 MB, 27 us): 6.5 TB/s
   - shared gate_up (176 MB, 20 us): 8.7 TB/s
   - shared_down (88 MB, 11.3 us + 4.7 us split-K reduce): 5.5 TB/s
   - KDA o_proj (44 MB, 12.3 us): 3.6 TB/s
   - latent_up (51 MB, 12.5 us): 4.1 TB/s
   - router + latent_down (64 MB, 25 us incl. 2 reduces): 2.6 TB/s
   - DSA qkv_a (33 MB, 18 us): 1.8 TB/s
   - DSA o_proj (44 MB, 19 us): 2.3 TB/s

   This compares with 13.9 TB/s achievable. The CuTe `SplitKGemm` labelled DSA.g_proj by position runs 16.8 us at B50
   and **44-62 us at B64-B92** (grid [64,4..5]), which is 0.7-1.0 TB/s.

   Hypothesis: the cuBLAS heuristics pick 2-CTA 64xN or 160x64 tiles with 156-448 CTAs and no persistent split. That
   leaves too few bytes in flight per SM for a 30-180 MB stream. Split-K adds separate `cublasLt::splitKreduce` launches,
   about 1.2 ms per step. The DSA projections on the side stream also contend with the indexer and top-k stream.

   The CuTe split-K tactic for the DSA SplitKGemm collapses above 200 rows. Check its tile policy. Also check whether the
   kernel is really g_proj or the indexer q projection; it was identified by position only.
2. **MegaMoE: 2.4-2.8 ms of gap, derate 1.27-1.33.** This one is below the 2.0 flag but is the second-largest gap.
   - The in-step per-layer time matches the standalone vendored kernel almost exactly. The vendored kernel is B7
     `deepgemm_vllm_sm107_situ_patch` from `megamoe-bench/results/rows.jsonl`. In-step vs B7 per layer: 109 vs 109 us at
     32 tok/rank, 115 vs 115 at 64, 122 vs 116 at 80.
   - So the gap is the kernel version, not in-step sync or skew. Per-rank step sums are within ±2%, and per-call p10/p90 is
     113/125 us.
   - The newest build, FI 6e438852b (B10), runs 91-95 us standalone at the same token counts, a 1.02-1.06 derate.
     Swapping it in projects to **-1.6 to -2.5 ms per decode step** and about -7 ms per 12K prefill forward (683 vs 762
     us/layer at 3072 tok/rank).
   - Not measured in-step: rel-1767cf7 has no wiring for the FI build, so this arm comes from the existing standalone
     rows (`kind: megamoe-arm` in rows.json).
3. **DSA DCP exact top-k: 1.4-2.3 ms of gap, derate 10-14.**
   - `k31dcpx::topk_merge_kernel<16>` runs 52-77 us per DSA layer against 6-7 us for P8 in the model.
   - The `topk_pp_hist/window/win` passes run 25-45 us per layer against 1.4 us.
   - Per-rank sums are flat (merge 1683-2007 us at B50), so the time is in the kernel and not in skew waiting.
   - Hypothesis: the merge grid is [2, rows]. Two CTAs per verify row walk 4 ranks x 2048 candidates serially, with remote
     NVLink reads and flag polls (`K31_DCPX_PDL=0`, ONE_ROUND_ROWS 256).
   - The pp histogram/window passes re-read the full logits three times.
   - The merge sits on the critical stream, and only MlaQuery overlaps it.

Other KERNEL-DEBUG items (derate > 2 and > 100 us per step; all flagged in rows.json):
- **MlaQuery q path**: 28-50 us per layer against 5-10 us, about half hidden on the side stream.
- **sparse MLA fmha**: 18.7 us against 5 us at B50.
- **KDA KdaWide**: 30-49 us per layer against 14-25 us, which is 35% of HBM on 166 MB of FP32 state per layer. The likely
  cause is a per-(head-group, request) CTA grid that is latency-bound on the q=4 serial update.
- **AttnRes mix**: 5.8 us per call x 187, against 0.4-0.8 us.
- **Unfused SiLU, latent RMSNorm, router top-k and MegaMoE prep**: about 1.2 ms per step in total. The model fuses them.
- **Prefill**:
  - FlashKDA segscan runs 3 recurrence passes plus a separate conv; the model is one pass with fused conv.
  - The sparse MLA uses the generation-phase `...P1VarSeqQ64Kv128Persistent2CtaKeepsAbForGen` kernel on 3072 ShardQ rows,
    reaching about 30% of FP8 MMA.
  - DSpark context AllGather runs on the NCCL RING kernel at 2.4 ms per forward.

## Failed or unrun cases

- **B92 at full agentx contexts**: OOM in the bench's state snapshot (30.9 GiB clone, 9 GiB free). This is a harness
  limit. It was measured with contexts capped to 12.0M per DP rank (mean 130K, model at L = 130K) instead.
- **MegaMoE FI 6e438852b in-step arm**: not run. The release tree has no vLLM integration of the FI build. It is replaced
  by the existing standalone B7 and B10 rows.
- **nsys**: not used, because the torch-profiler traces were sufficient.
- **Real weights / real routing**: not run. Timing uses dummy weights with balanced routing.

## Files

- `rows.json`: lane rows (370)
- `parsed/`:
  - per-case group tables (`*.txt`, `*.json`)
  - `decode_families.txt`
  - `evid_B50.txt`: per-kernel per-call and per-rank stats
  - `model_rows.json`
- `tools/`:
  - parsers: `gap_parse.py`, `pf_parse.py`, `fam.py`, `evid.py`, `dumpseq.py`, `assemble.py`, `model_rows.py`
  - task scripts: `dprof*.sh`, `pprof.sh`, `pf_run1.sh`, `pf_cfg.py`, `pf_shapes.json`
  - The parsers import `kb_parse.py` (`$W/hti/kb/tools`).


## Review fixes (adversarial review, 2026-11-18)

No row values changed. Spot checks passed: derate = measured / t_ach on every row, and `t_ach_us` / `t_sol_us` match
the current `out/optables.json` (the model lane's re-scope did not change any per-op value used here).
- The 14 rows where measured/t_ach differs from `derate` by more than 1% are rounding only: t_ach_us is stored at 0.1 us (0.1-1.9 us values for MOE.topk, latent_norm, shared_act, G.embed, DSA.prologue) while derate was computed unrounded. Those derates (20-3400x) are dominated by a near-zero model term and should be read as 'unfused op, not priced', not as a kernel ratio.
- B-dependence checked: B32 model 14.76 ms at L190K before removing LM head/rejection/DP sync, 14.706 after; the decode lane's L135K D3 r8 point is 14.70 ms and measured 24.16 vs 24.54 ms here (1.6%), consistent.
