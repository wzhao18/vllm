# K3.1 VR200 microbench vs SOL model: synthesis (2026-11-18, scope EP <= 16)

Sources: lane results in `microbench/results/{moe,comm,ops1,prefill,decode,gap512,model}/` (rows.json and README for each,
including review fixes). Machine-readable factors are in `inputs/derating.json`, regenerated with
`python3 perf-model/microbench/mk_derating.py`.

- Definition: derate = measured / model `t_ach` (the achievable column of `out/optables.json`). A derate above 2.0 is
  flagged KERNEL-DEBUG.
- Checkpoint: `mxfp4_release`, i.e. MXFP8 MoE activations (asserted in logs), BF16 linears, FP8 MLA KV, MXFP4 indexer
  KV, FP32 KDA state and exact top-k. Decode uses k=3 at L=135K; prefill uses F=16K at L=78K.
- Code: release tree `rel-1767cf7` (6a14e2fada), FlashInfer 0.7.3.dev20261031, dummy weights.
- **In-step MegaMoE is the older vLLM-vendored kernel** (`megamoe_rubin` / deep_gemm `sm100_fp8_fp4_mega_moe_situ`).
  The newest MegaMoE (FlashInfer 6e438852b) was measured standalone only.

## 1. Measured vs model per topology

### Prefill (tok/s/GPU, MXFP8 MoE, F16K at L78K unless noted)

| topology | measured step | model t_ach | derate | tok/s/GPU measured | model | release NVFP4 MoE (not like-for-like) |
|---|---|---|---|---|---|---|
| P1 TP8+DCP8 EP8 | 228.2 ms | 218.2 | 1.05 | 8,974 | 9,384 | 10,069 |
| P2 DP4x(TP4=DCP4) EP16 | 350.7 ms | 282.2 | 1.24 | 11,680 | 14,514 | 13,312 |
| P2 at F32K (production MNBT) | 653.1 ms | 551.3 | 1.18 | 12,544 | 14,860 | 14,136 |
| P2, C512 harness (8K, 12K, 2x6K, 16K, fresh 12K) | 188-350 ms | 144-286 | 1.22-1.31 | | | |
| P4-4 TP4xDP4 DSA-DP (PR #20) | 330.5 ms | 269.1 | 1.23 | 12,393 | 15,222 | |
| P4-2 TP2xDP8 DSA-DP (PR #20) | 623.0 ms | 500.1 | 1.25 | 13,150 | 16,381 | |
| P3-16 DEP16 | 1,179.8 ms | 960.4 | 1.23 | 13,887 | 17,060 | 15,409 |
| P3-16 at F8K | 610.9 ms | 486.5 | 1.26 | 13,409 | 16,838 | 15,401 |

- Every layout without DCP8 runs at **1.18-1.26x**. P1 runs at 1.05, but only because the model overprices P1
  AttnRes by about 20 ms (it charges all rows; the code runs AttnRes on 1/8 of the rows and then all-gathers). That
  error hides a real excess of about 1.15x.
- The excess is concentrated in the DSA block (1.34-1.81) and the MoE block with shared experts (1.15-1.18). The KDA
  block runs at 1.10-1.21, and the large GEMMs at 0.96-1.08.

### Decode (step ms by r = requests per GPU, specialized decode, k=3)

| topology | r <= 0.25 | r 1 | r 4 | r 8 | r 12-24 | median derate by band (<=0.25 / 0.5-2 / 3-8 / 12-24) |
|---|---|---|---|---|---|---|
| D1 TP16=DCP16, MoE-TP16 | 5.2-6.8 vs 3.6-5.3 | 10.8-11.1 vs 8.95 | 18.3 vs 13.1 | 25.1 vs 16.5 | over measured KV capacity | 1.24 / 1.31 / 1.46 / - |
| D2 TP8=DCP8 (ran MoE-TP8; model is EP8) | 7.3-10.8 vs 5.1-6.0 | 15.5 vs 10.45 | 33.3 vs 17.8 | 38.9 vs 20.5 | r9 42.7 vs 21.1 | 1.50 / 1.48 / 1.90 / 2.03 |
| D3 DP4x(TP4=DCP4) SP EP16 | 15.6 vs 8.2 | 21.2 vs 11.1 | 24.9 vs 13.7 | 24.2-24.5 vs 14.7 | 27.2-33.3 vs 15.6-18.8 | 1.91 / 1.87 / 1.74 / 1.81 |

- D1 at k=7 runs at 1.15-1.53 up to r=4. At r=8 it hits a cliff: 51.2 ms against 22.9 modelled (derate 2.24),
  because the MoE switches from the specialized-decode kernels to stock trtllm-gen. This point is left out of the
  band derates.
- D3 C512 (gap512, B32-B92 = r8-23) runs at 1.67-1.87. The decode gap is real kernel time inside the forward.
- **KV capacity is 3-7x below the model for D1 and D2.** Each request needs a fixed floor of blocks regardless of
  context.

  | topology | measured r_max | model r_max |
  |---|---|---|
  | D1 k3 | ~15.6 | 66 |
  | D1 k7 | ~9.7 | 65 |
  | D2 k3 | ~9.9 | 26 |
  | D3 k3 | ~33 | 48 |

  D1 and D2 therefore drop out of the high bands sooner than the model says.

## 2. Does the topology ranking hold after derating?

Topologies that were not measured are projected by analog transfer (`derating.json: projection_unmeasured`). Each
one takes the measured derate of the closest implemented path at the same r, given as a range between additive excess
and ratio:

- SP-path layouts D5, D6, D7-x-16, D4-16 and D9-16 use D3.
- D8-16 uses D1.
- P6-x uses P4-x, plus a no-dedup MoE penalty because dedup is not implemented.
- P9 and P9-16 use P3-16.
- P2b and P5-4 use P2; P5-8 uses P1.

| band | model order (EP <= 16) | measured / derated order | holds? | evidence |
|---|---|---|---|---|
| prefill, F16K | P6-16 > P3-16 = P9 = P9-16 > P4-2 > P6-4-16 > P4-4 > P2 > P5-4 > P2b > P1 > P5-8 | P3-16 13.9K ≈ P9/P9-16 13.7K (proj) > P6-16 13.5K (proj) > P4-2 13.1K > P6-4-16 12.7K (proj) > P4-4 12.3K > P2 11.7K > P5-4 10.6K > P2b 9.9K > P1 9.0K > P5-8 7.7K | **Yes among the measured five**: P3-16 > P4-2 > P4-4 > P2 > P1, the same as the model, with similar margins (P3/P4-2 is +5.6% measured vs +4.1% model; P4-4/P2 is +6.1% vs +4.9%). **No for top-1**: P6-16 drops below P3-16 once the model's dedup is removed, because the shipped MegaMoE has no dedup and dedup measured slower. | prefill rows, moe P6 rows (1.09-1.14 vs dedup model) |
| decode r <= 0.25 | D1 > D8-16 / D2 | D1 > D2 > D3 measured; D8-16 not projected at this r (about 5% behind D1 at r1) | **Yes** | D1 5.2 / D2 7.3 / D3 15.6 ms (r 0.06-0.25) |
| decode r 0.5-2 | D1, with D8-16 / D7-8-16 within 4-10% | D1 10.8 < D8-16 11.3-11.4 (proj) < D2 15.5 < D7-8-16 18.7-19.9 (proj) < D3 21.2 at r1 | **Yes for D1 as best.** The SP-path runners-up fall to 1.7-1.9x of D1 rather than within 10%. | D3 derate 1.87-1.91 vs D1 1.20-1.31 |
| decode r 3-4 | D7-8-16 > D7-4-16 / D1 (within 0-4%) | at r4, D1 18.3 ms vs D3 24.9 (model 13.1 vs 13.7, a tie). The SP-path excess (about 11 ms) projects D7-x-16 above D1. | **No**: D1 is best here, not D7-8-16. The SP path carries a constant excess of about 10 ms that the model doesn't charge. | D1 r4 1.40; D3 r4 1.81 |
| decode r 8 | D7-8-16 13.8 > D7-4-16 14.1 > D3 ≈ D5 14.7 > D1 16.5 > D8-16 | D7-8-16 23.0-23.6 (proj) ≈ D7-4-16 23.5-23.8 (proj) < D3 24.4 ≈ D5 24.4 (proj) < D1 25.1 < D8-16 25.3 (proj) < D2 38.9 | **Weakly yes.** The order is preserved, but the margin over D1 shrinks from 16% to 3% (D3 measured vs D1) or 6-8% (D7-8-16 projected), which is inside the analog-transfer error. | D3 24.4 vs D1 25.1 measured |
| decode r 12-32 | D5 / D7-2-16 / D7-4-16 (tie within 1%) | only D3 measured (1.75-1.87). The SP-path analogs share one derate, so the tie is unresolved by construction. D1 and D2 fall out: over measured KV capacity above r ~15.6 and ~9.9. | **Unverified.** The winner needs a D5 or D7-x-16 implementation and measurement. | D3 B48-B96 rows; capacity table above |

Summary:

- **Prefill trend holds.**
- **Decode trend holds at r <= 2** and holds weakly at r = 8.
- **Decode trend breaks at r 3-4**, where D1 should be preferred.
- **Decode trend is unverified at r >= 12.** The model's mid and high-band near-ties (under 2-4%) are far smaller
  than the measured path-dependent excess (D3 at 1.7-1.9 vs D1 at 1.2-1.5), so the per-path kernels, not the model,
  decide those bands.

## 3. KERNEL-DEBUG list, ranked by end-to-end worth

Excess = measured minus model achievable, per step, on the topology where it matters.

| # | item | derate | excess per step (% of step) | first hypothesis | source rows |
|---|---|---|---|---|---|
| 1 | **BF16 dense GEMMs on the decode SP path** (cuBLAS nvjet + CuTe split-K + separate split-K reduce) | 1.95-2.65 | 4.1-4.8 ms on D3 C512 (14-16%); applies to every SP layout, including the model's mid/high winners | Tile/CTA heuristics at M=50-368 leave too few bytes in flight (1.8-8.7 vs 13.9 TB/s achievable). The split-K reduce adds about 1.2 ms. A CuTe split-K tactic breaks down above 200 rows (17 → 44-62 µs). Port D1's specialized CuTe GEMMs to the SP path. | gap512 FAMILY BF16; decode prof3 |
| 2 | **MegaMoE vendored → FlashInfer 6e438852b** (integration; not flagged) | 1.27-1.33 decode in-step; 1.64 P1 | -1.6 to -2.5 ms D3 (6-9%); about -21 ms P1 (9%); about -7 ms P2 F16K (2%). These are projections. | The newest kernel runs 91-95 µs vs 109-115 µs at 32-64 tokens/rank, and 511/901 vs 742/1096 µs in prefill. It is not wired into the release tree. | moe rows; gap512 MOE per layer; review notes |
| 3 | **D1 k7 MoE path cliff** at 1024 verify rows | 2.24 (step) | +26 ms at D1 k7 r8 (about 2x step) | The specialized MoE path caps wide rows and falls to stock trtllm-gen bmm (about 18 ms). It only matters if D1 k7 at r ≥ 8 is an operating point; the measured KV max is r ~9.7. | decode d1k7hi2, d1k7p |
| 4 | **Sparse MLA (trtllm-gen, decode form)**, prefill and decode | prefill 3.2-4.0; decode 5-10 | +45 ms P3-16/P9 (3.8%), +21 ms P4-2 (3.4%), +9.9 ms P2 (2.8%), +5 ms P1 (2.3%); decode +1.6 ms D3 (5.6%, q path + DCP value) | Pads 96 to 128 heads. The per-token decode formulation re-gathers 2048 KV rows for every row. The 512-wide V may be split into two halves that redo the QK. It reaches about 1/3 of the FP8 MMA rate and is not HBM-bound (cold equals warm). Try a prefill-style sparse kernel, then confirm with ncu. | ops1 sparse_mla-161/162/094/089/310; prefill DSA.sparse_mla; gap512 |
| 5 | **Shared experts on the prefill side stream** | shared_down 3.8-4.0; shared_act 15-75 | MoE block +22-24 ms on P2/P4-4 (7%) and +40 ms on P3-16 at F8K (part of it is the vendored MegaMoE) | Few SMs left beside the persistent MegaMoE, plus a separate SiLU kernel. Fuse the activation, or reserve SMs / schedule after MegaMoE. | prefill MOE.shared_down/act, block MoE |
| 6 | **DSA exact top-k + DCP merge**, decode | 9.6-13.6 (D3), 4.6-18.9 (D1) | 1.4-2.6 ms on D3 C512 (6-8%) | Two CTAs per row walk 4 x 2048 candidates serially with remote reads and flag polls (PDL off). The pre-passes re-read the logits, and the merge sits on the critical stream. | gap512 FAMILY top-k; decode prof1b/prof3 |
| 7 | **KDA core** (FP32 state) | decode 2.0-3.4 (9.7-11.4 at r ≤ 1); prefill 1.4-2.4; 96-head 4.19 | +36.7 ms P3-16/P9 (3.1%, stock path); +8.7 ms P2 (2.5%); +1.1 ms D3 (4%); +0.8 ms D1 r1 (7%) | Prefill: three recurrence passes plus a separate conv, and the 96-head shape falls off the v-split FlashKDA build. Decode: a per-layer floor of about 10 µs plus FP32 state traffic at about 35% of HBM. | ops1 kda-077/051/099; prefill KDA.core_prefill; gap512 KDA core |
| 8 | **DSA prologue in prefill** (unnamed CuTe `Kernel`, ShardQ quant/insert, head pad) | 21-53 | 3.4-8.5 ms (1-2.5%) | Separate kernels that the model treats as fused. The anonymous CuTe kernel alone is 2.4-5.8 ms; identify it first. | prefill DSA.prologue+aux |
| 9 | **Small unfused ops**: decode AttnRes mix, SiLU, latent RMSNorm, router top-k, MoE prep; prefill KDA conv/o_norm, MOE.topk, latent norm, gate_mul | 10-146 | decode flat 1.55 ms (5% D3 C512, more at low r); prefill 5-7 ms (2%) | Fuse them into neighbouring kernels, or have the model charge the extra memory pass. | gap512 FAMILY small; prefill rows |
| 10 | **Comm**: comm-112 RS N=2 (P4-2/P6) | 2.70 | +11.5 ms P4-2 (1.8%) | Unfused `torch.sum` over [2, 58.7M] runs at about 5x the HBM roofline; a fused owner-sum gives about 1.55. | comm rows |
| 11 | **Comm**: small RS/AR at the Lamport floor (D3/D5/D7-x) | 2.0-2.45 | 0.7-1.0 ms per primitive per step (3-4% decode) | Lamport floor of about 5 µs plus a separate 2-3 µs local-sum kernel. Fuse the exchange and the reduce. | comm-123/119/137/164/197/203/210 |
| 12 | **Drafter** (prefill AllGather; decode tail) | prefill 5.8-8.8; decode 2.0-3.4 | 2-3.8 ms prefill (1%); 0.4 ms decode | NCCL AllGather takes 2.4 ms against 20 µs modelled. Use the symmetric-memory push. | prefill DS.drafter; gap512 TAIL |
| 13 | **Decode indexer** (stock paged DeepGEMM, one row at a time; DeepSelect floor) | 5-9 standalone; 2.2-2.8 in-step logits | 0.04-0.7 ms | Each of a request's 4 rows re-streams the keys. The production sdec indexer measures about 4.4x in the c512 profile. | ops1 dsa_indexer_topk-*; gap512 |
| - | Not kernel bugs | | | ops1 unaligned-N KDA.in_proj is a bench artifact (the release pads in_proj). D2 at r ≥ 5 at 2.0-2.2 is a model/what-ran mismatch (MoE-TP8 vs EP8). comm-230 P9 copy-engine push (2.5x) is async, and P9 is not implemented. Decode BMM 3-26x standalone is fused in production. | review notes |

**Model fixes, independent of kernels.** These also feed `derating.json`:

- **Per-kernel floor:** the model has none. Measured floors are 4-50 µs per kernel per layer, which drive most decode
  derates at r ≤ 1.
- **MegaMoE cost:** charge it as `max(ops, P11+P12)` plus the exposed part, not the sum.
- **Large-message NVLS all-reduce:** the model's achievable time is too slow. Measured multimem algbw is
  0.64/0.77/0.84/0.88 TB/s at N = 2/4/8/16 (the model uses 0.29-0.46).
- **Reduce-scatter:** costs about 0.75x an all-reduce, not 0.5x.
- **Push bandwidth:** use 1.0 TB/s, not 1.25.
- **Standalone exchange floor:** about 5 µs, or about 8 µs with a separate barrier.
- **P1 AttnRes:** price it on 1/8 of the rows plus the all-gather.
- **KV capacity:** add the fixed per-request block floor (3-7x on D1/D2).

## 4. Still unmeasured

- **Decode topologies:** D4-16, D5, D6, D7-4/2/8-16, D8-16 and D9-16 were not measured. The specialized decode
  refuses EP outside the TP4 sequence-parallel layout; D7-x-16 and D8-16 are unimplemented; and running the stock
  backend on VR is not allowed.
  - These include the model's winners at r ≥ 3, so the mid/high-band ranking is projected, not measured.
  - D2 was not measured with EP8. D3 was not measured at k=7 (SP supports only 3 draft tokens).
  - Missing r points: D1 above r=8, D3 at r=32, and the bench state-snapshot OOM points (a bench artifact).
- **Prefill topologies:** P2b, P5-4 and P5-8 were not run. P6-16 and P6-4-16 need dispatch dedup, which is not
  implemented. P9 and P9-16 need DSA-CP, which has no code path. Also not measured: the model's best 64K chunk (about
  +4%), and P4 with the real scheduler (owner imbalance, per-owner KV).
- **In-step MegaMoE with FlashInfer 6e438852b:** not wired into the release tree. All in-step prefill and decode
  numbers use the vendored kernel.
- **nvfp4_modelopt checkpoint:** only the release NVFP4-MoE arms ran (P1, P2, P3-16). MXFP8 shared/KDA GEMMs were
  measured standalone only.
- **Overlap variants O1-O3** (two-microbatch overlap, dedup, die-local streaming) are model-only.
- **ops1 cases not benched:**
  - `G.embed` and `DS.aux_capture` (16 cases), the tiny drafter ops (64) and `fused_epilogue` (52).
  - The production sdec fused indexer, MlaQuery and RecoverSSM as standalone kernels.
  - Separate dispatch / GEMM / combine splits inside MegaMoE (that would need in-kernel instrumentation).
- **Comm cases not benched:** production comm kernels standalone (`gemm_rs_ar`, `SpReduceScatter`,
  `mnnvl_multimem_reduce_scatter`), and EP dispatch with real routing imbalance.
- **Real weights, routing and correctness:** no real-weight routing skew anywhere. The decode step benches have no
  numeric correctness check.
- **KV capacity:** not measured for D5 or D7-x-16. If they share D1/D2's per-request floor, their high-r end moves too.
