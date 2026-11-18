# K3.1 VR200 optimization plan: DEP16 prefill + DCP4xDP8 EP32 decode, nvfp4_modelopt (2026-11-18)

This plan comes from three per-component gap ledgers, each checked by a skeptical reviewer.
- Raw data: `nvfp4_ledgers_raw.json`
- Condensed: `nvfp4_ledgers_summary.txt`

Projections apply measured MXFP4 derates to the NVFP4 model wherever the kernel is unchanged. MegaMoE NVFP4 and the MXFP8 shared/KDA GEMMs are new kernels, so their numbers are bounded ranges until measured.

## Where we are and where the plan lands

| | measured today | near-term projection after enablement | model achievable (nvfp4) |
|---|---|---|---|
| DEP16 prefill, F16K L78K | 13.9K tok/s/GPU (mxfp4); 15.4K (release NVFP4-act arm) | 19.4-20.5K (798-844 ms/step), mostly from MXFP8 KDA/shared GEMMs plus FI MegaMoE | 30.7K (532.9 ms) |
| D3-32 decode r8 | 1,178 (20.9 ms) | 1,225-1,450 (17.0-20.1 ms) | 2,855 (8.6 ms) |
| D3-32 decode r24 | 2,425 (30.5 ms) | about 2.6-2.9K | 6,280 |

On decode, NVFP4 itself adds little:
- the NVFP4 kernel saves about 0.6 ms/step over MXFP8 (FlashInfer vs FlashInfer at T32);
- MXFP8 linears save 0.2-1.3 ms;
- the kernel items below are worth several times more.

## Phase 0: enablement (blocks every NVFP4 number)

| # | item | status | evidence | effort |
|---|---|---|---|---|
| B1 | MegaMoE format chosen from the checkpoint; the ModelOpt expert names and scales mapped | missing | model.py:816 hard-codes MEGA_MOE_MXFP4. The mapping at model.py:585-602 has only weight_packed/weight_scale. | S |
| B1b | Native NVFP4 parameter set in the vendored megamoe_rubin (E4M3 block-16 scales, weight_scale_2, input_scale) | missing | experts.py:289 and :442 refuse it. The NVFP4 kind exists only as an MXFP4->NVFP4 conversion. | M |
| B2 | NVFP4 model view plus a dummy-safe bench, asserting the logged quant kind | missing | | S |
| B3 | NVFP4 MegaMoE at EP32 for decode | missing in every tree | megamoe_rubin knobs cover WS 4/8/16 only; deep_gemm_mega_moe is fp8_fp4 only. Fix: re-vendor FI a368c5f66 (SM107_MAX_EP_WORLD_SIZE=32). | M-L |
| B4 | MXFP8 projections on the specialized-decode SP path (DecodeProjection + SpOproj variants, per-shape BF16/MXFP8 dispatch) | missing | sdec bf16_weight raises on quantized linears | M |
| B5 | Real checkpoint on Lustre. Verify the safetensors key names (w1/w2/w3 vs gate/up/down), input_scale values (pow2?), KV k/v scales; always pass --kv-cache-dtype fp8 explicitly | unverified | no local config/index | XS-S (download needs the user's OK) |
| B6 | Memory: P3-16 F16K at TP1 OOMed on mxfp4 (225 GiB of weights measured vs 205 GB modelled). NVFP4 weights are about 155 GiB per GPU at P3-16 and 81 GiB at D3-32, so it likely fits; verify. | check | | XS |

Gate: real-weight load, then real-rejection gsm8k on both topologies (informational), then profiled step benches.

## Phase 1: baseline measurements (first Hecate jobs)

1. Profiled D3-32 MXFP4 step at r8/r16/r24 (dprof tool with d3x32k3.json, 8 nodes). This replaces the EP16-transferred breakdown.
2. Profiled NVFP4 P3-16 step at F8K and F16K. Under the NVFP4 MoE arm, F8K = F16K in tok/s (15,401 vs 15,409), so F8K may be the operating point.
3. Profiled NVFP4 D3-32 step at r1/4/8/16/24 once B3+B4 land.
4. Recalibrate `calibrated.json`; pre-register projections before each lane.

## Phase 2: kernel lanes, ranked by projected excess over model achievable

### Prefill DEP16 (projected about 825 ms/step vs 533 model)

| rank | item | excess (ms/step, % of step) | basis | fix | gain | effort |
|---|---|---|---|---|---|---|
| 1 | MegaMoE vendored -> FI 6e438852b, with mandatory 2x8K sub-calls | 83-160 (10-19%) | vendored in-step inferred 281-358; FI standalone 230.7 measured | Wire into specialized_prefill. Net out the loss of shared-expert overlap under the vendored router kernel (about 60 ms at F16K): land it together with the shared-expert SM-budget overlap (item 6). | 37-127 ms | M |
| 2 | KDA chain: core + conv + gated o_norm | 62 (7.6%) | measured | v-split FlashKDA build for 96 heads; fuse prepare into the recurrence; conv into the load; o_norm + MXFP8 quant into the epilogue | 40-50 ms | M-L |
| 3 | Sparse MLA + prologue + gate_mul | 52 (6.2%) | measured | ncu first (V-split recompute is a hypothesis). Prefill-form CuTe sparse MLA with native 96 heads; fuse FP8 q-quant + RoPE into the w_uk epilogue, gate_mul into the w_uv epilogue; identify the anonymous CuTe kernel (1.05 ms) | 30-40 ms | L |
| 4 | Indexer logits + exact top-k | 28 (3.4%) | measured | exact histogram select with an exact graph-safe overflow path | 12-18 ms | M |
| 5 | Router / top-k / latent BF16 GEMMs / latent norm | 20 (2.5%) | measured | fused epilogues, per-shape tactics | 10-14 ms | S-M |
| 6 | Shared experts MXFP8 + SiLU-mul + side stream | 14-24 (2.3%) | new kernel | fused SiLU-mul + quant at stock rounding; SM budget beside MegaMoE | 15-20 ms plus overlap | M |
| 7 | Small BF16 GEMMs + drafter + dense | 14 (1.7%) | measured | tactics; check whether the drafter AllGather exists at P3-16 | 6-9 ms | S |
| 8 | KDA in/o_proj MXFP8 | 3-12 (0.4-1.5%) | new kernel | re-measure at padded N; fuse the input quant | | S-M |
| 9 | AttnRes pre+post | 7 (0.8%) | measured | fuse | 3-5 ms | S |
| S | Two-microbatch overlap (O1) | model -90 ms | model only | reassess after 1, 2 and 6 (KDA CTA halving, memory) | | L |

### Decode D3-32 (r8 projected about 18.6 ms vs 8.4 model bench-comparable; r24 about 26-28 ms)

| rank | item | r8 excess | r24 excess | basis | fix | effort |
|---|---|---|---|---|---|---|
| 1 | CuTe low-latency GEMM family: BF16 SP (DSA/MLA/router/latent) + MXFP8 shared (M=32/96) + MXFP8 KDA proj, with producer-fused quant at stock rounding | 2.4-3.5 ms (13-19%) | 3.7-4.6 | BF16 measured; MXFP8 projected | Port D1's specialized CuTe GEMMs; per-shape BF16/MXFP8 dispatch; drop the split-K reduce launches; identify the SplitKGemm collapse above 200 rows (kernel identity unverified) | M |
| 2 | MegaMoE NVFP4 EP32 | 2.3-2.9 (12-16%) | 2.7-3.9 | FI standalone measured; in-step unknown | B3 first: FI vs vendored gives 0.7-1.3 ms, NVFP4 0.6 ms; overlap shared GEMMs with MegaMoE (shared runs serial today) | M-L |
| 3 | Small unfused ops (AttnRes 0.49, SiTU 0.33, router top-k 0.26, latent norm 0.25, prepare 0.20) | 1.47 (8%) | 1.37 | measured | one fusion site each (router top-k into the router-GEMM epilogue) | M |
| 4 | DSA DCP exact top-k + merge | 1.31 (7%) | 2.0-2.15 | measured | merge grid beyond 2 CTAs/row, PDL, single-pass hist/window | M |
| 5 | DSA attention path (MlaQuery q path, sparse MLA, DCP value, indexer 4-row joint scoring) | 1.16 (6%) | about 2.2 | measured | fused q path; streamed dedup gather; joint 4-row indexer | L |
| 6 | KDA core (KdaWide at r8, KdaWS at B80+) | 0.87 (5%) | 1.6-1.7 | measured | per-shape kernel covering B32-B128; raise HBM utilization above 35% | M |
| 7 | Drafter + tail | 0.28 | 0.96 | measured | | M |
| 8 | SP comm RS/AG | 0.16 | 0.61 | measured | fuse exchange + reduce (with the GEMM epilogue) | M |
| S | Lift the SP 512-verify-row cap (r up to about 48) | | model +17% tok/s/GPU at r48 | model | sdec runtime + graph-capture memory (the r32 bench OOM is a harness artifact) | M |

Unknowns to measure first:
- real-weight routing skew at EP16/EP32 with DP lock-step;
- AL under NVFP4;
- DP-rank imbalance in DEP16;
- in-step placement of the MXFP8 activation quant;
- KV r_max under NVFP4 (projected about 49.8 on D3-32).

## Lanes (at most 2 concurrent)

- Lane A (MoE + GEMM): B1/B1b/B2 -> B3 (FI a368c5f66 re-vendor) -> B4 + decode CuTe GEMM family -> prefill FI wiring + shared-expert overlap.
- Lane B (DSA/KDA, checkpoint-independent, can start now): decode DCP top-k/merge + small-op fusion -> prefill KDA chain -> sparse MLA (ncu first).

Each landing gets a step microbench A/B, a recalibration, and a model update. e2e checkpoints run at milestones.
