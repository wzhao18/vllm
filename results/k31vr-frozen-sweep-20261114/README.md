# Kimi K3.1 AgentX on VR200: frozen-code pareto sweep (started 2026-11-14)

Goal: the best 4-way pareto curve for VR200, picked by config only.
- The main metric is throughput per GPU vs P90 interactivity.
- A point is not chosen if it has a clear drawback on P90 E2E-normalized interactivity, P90 E2E latency or P90 TTFT.
  Example: the C160 1P1D run of 2026-11-14 queued on its single prefill instance, which pushed TTFT p90 to 4.2 s.

<!-- SUMMARY -->
## Summary (2026-11-14, all runs landed; gsm8k for C256 pending at time of writing)
- **Final chosen set (6 points, frozen code b68d67e7e, K3.1 golden AL, watermark 0.05; user selection 2026-11-15 + xPyD search):** C1, C16 1P2D, C96 1P2D, C112 1P1D, C224 2P1D, C512. All other measured runs are valid but unselected (grey); reasons in the points table.
- **Reproductions:** every previously chosen config reproduced within its pre-registered lines. x is 2-5% below the old runs, which is the expected effect of moving from the Kimi K3 AL (3.84 / 3.0) to the InferenceX Kimi K3.1 golden AL (3.69 / 2.88). y is within 0-4%.
- **New points:**
  - **C96 1P2D:** one prefill shared by two C48-shape decodes. It gives +10% y over C48 at -2% x. Its TTFT p90 is 1.5 s vs 0.73 s.
  - **C112 1P1D:** about +6% y over the C64-C224 chord at x 186. Its TTFT p90 is 2.75 s, from a moderate single-prefill queue.
  - **C256 2P1D:** the only point between x 80 and 113 with TTFT p90 under 3 s.
- **Not used:**
  - **C160 1P1D:** 38.6K at 170, but TTFT p90 6.9 s from single-prefill saturation. Replaced by C160 2P1D at 31.0K, 161, TTFT 1.6 s.
  - **3P2D C384:** not run. Its gate (TTFT) failed on the measured C112 and C224 runs.
  - **About 15 other configs:** rejected in planning (`PHASE2_PLAN.md` on Hecate) because they were within drift or saturated the prefill.
- **VR200 vs the GB300 front at the same P90 interactivity** (GB300 = SemiAnalysis InferenceX Kimi-K3.1-Pro dynamo-vllm, 2026-11-10; linear interpolation along the GB300 front):

| point | TPS/GPU | P90 intvty | TTFT p90 (s) | GB300 front at same intvty | VR200 / GB300 per GPU | per MW (x 2.12/3.3) |
|---|---|---|---|---|---|---|
| C1 | 1,342 | 779 | 0.36 | beyond GB300 range (max x 450) | - | - |
| C16 1P2D | 3,104 | 601 | 0.39 | beyond GB300 range (max x 450) | - | - |
| C96 1P2D | 17,132 | 290 | 1.51 | 4,176 | 4.10x | 2.64x |
| C112 1P1D | 29,863 | 186 | 2.75 | 9,775 | 3.05x | 1.96x |
| C224 2P1D | 41,591 | 113 | 1.88 | 18,112 | 2.30x | 1.48x |
| C512 | 73,736 | 80 | 6.43 | 26,741 | 2.76x | 1.77x |

VR200 also covers x 450-780 (C8, C1), which the GB300 series does not reach.
<!-- /SUMMARY -->

## Fixed for every run
- **Code:** frozen at `wzhao/agentx-vr-nov-13` **b68d67e7e**. That is d0d09dc plus F1 (`dsa/wide.py`) and F3a (`kda/prepared.py`), and it is byte-identical to the deployed Hecate tree `hti/lowlat/trees/ll-integ-dea4ebd06f`. GitLab mirror: `weizha/vllm` `wzhao/k31-vr-freeze-ll`.
- **Speculative decoding:** DSpark, probabilistic drafting, synthetic rejection at the **InferenceX Kimi K3.1 golden AL**. The source is `golden_al_distribution/kimi-k3.1-pro__dspark__kimi-k3.1-pro-dspark-preview__probabilistic-block.yaml`, thinking_on: k3 = 2.88, k7 = 3.69. This is the same table the GB300 baseline uses.
  - All VR results published before this folder used the Kimi K3 values (k3 = 3.0, k7 = 3.84). They are about 4% optimistic on interactivity and are not mixed into this sweep.
- **Decode:** `watermark: 0.05` on every config.
- **Precision:** unchanged official rules (BF16 linears, MXFP4 MoE, FP8 MLA KV, FP32 KDA).
- **Per allocation:** one 60-min AgentX run (DURATION 3600, profiling phase), then a full gsm8k with real rejection on the same allocation, same deployment and same nodes.

## Metrics (`frz_metrics.py`, same method as `live_metrics.py`)
- **TPS/GPU:** sum(ISL + OSL) over profiling records ÷ profiling wall time ÷ GPUs. GPUs are counted from the node configs.
- **P90 interactivity:** 1000 / p90 ITL.
- **P90 E2E-normalized interactivity:** 1 / p90(request latency / OSL).
- **P90 E2E latency** and **P90 TTFT**.
- **Per MW:** VR200 3.3 kW and GB300 2.12 kW all-in per GPU.

## Selection rule (which measured points are chosen)
A measured point is **chosen** unless one of these holds (then it is greyed):
1. It is dominated on throughput vs P90 interactivity by another valid run (higher or equal on both).
2. Clear TTFT drawback: TTFT p90 > 3 s while a neighbouring chosen point at similar interactivity (within ~15%) has TTFT p90 under half of it (e.g. C160 1P1D: 6.9 s vs 1.6 s for 2P1D).
3. It is dominated on at least two of the three secondary panels (E2E-norm interactivity, E2E latency, TTFT) by a chosen point within 10% of its P90 interactivity.
4. It fails validity (profiling span, errors, keepalive overlap, P/D transfers) or gsm8k strict < 0.95.
Trade-offs that do not meet 2 or 3 are kept and noted (e.g. C96 1P2D: +10% y vs C48 at -2% x, TTFT p90 1.5 s vs 0.73 s).

## Files
- `points.json`: every landed run. The `chosen` flag marks the pareto set.
- `runs/<point>-<job>/`: AgentX summary, metrics, render config and gsm8k result.
- `pareto4_per_gpu.png` and `pareto4_per_mw.png`: chosen points in blue, measured-but-unused points in grey, GB300 in orange.
- `gb300_sa_k31.json`: the GB300 baseline, SemiAnalysis InferenceX Kimi-K3.1-Pro, dynamo-vllm, 7 runs from 2026-11-10.
- Regenerate the plots with `python3 plot_pareto4.py`.

## Points
Phase 1 reproduces the previously chosen configs on the frozen code. Phase 2 adds configs chosen from the phase-1 results to push the front.

Still queued or running: see the sweep status below the table.

<!-- POINTS -->
| point | topology (GPUs) | AgentX run | TPS/GPU | P90 intvty | P90 E2E-norm | P90 E2E (s) | P90 TTFT (s) | gsm8k strict | chosen | note |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 1P1D: P TP8+DCP8, D TP16+DCP16 (24) | C1-a1093683 | 1,342 | 778.6 | 328.8 | 6.5 | 0.36 | 0.9659 | **yes** | accept: old K3-AL run 1,347 @ 805 -> -0.4% y, -3.2% x (AL correction); gsm8k on the same hold |
| C16 1P2D | 1P2D: P TP8+DCP8, 2 x D TP16+DCP16 (k7) (40) | C16-1P2D-a1096999 | 3,104 | 600.6 | 293.5 | 3.9 | 0.39 | 0.9704 | **yes** | xPyD search (wf_82555e99): dominates C8 (2.89K @ 530); replaces it on the chosen curve; gsm8k on the same hold |
| C8 | 1P1D: P TP8+DCP8, D TP16+DCP16 (24) | C8-a1093682 | 2,890 | 529.7 | 326.7 | 4.7 | 0.35 | 0.9659 | no | accept: old K3-AL run 2,960 @ 553 -> -2.3% y, -4.2% x (= AL 3.84 -> 3.69); gsm8k on the same hold; superseded by C16 1P2D (higher y and x) |
| C48 1P1D | 1P1D: P TP8+DCP8, D TP16+DCP16 (24) | C48-1P1D-a1093681 | 15,544 | 296.1 | 158.1 | 7.6 | 0.73 | 0.9674 | no | accept (proj ~293 @ 15.0-15.4K; old K3-AL 15,593 @ 305 -> -0.3% y, -3% x); gsm8k on the same hold; unselected by user 2026-11-15 (final set: C1, C8, C96 1P2D, C112 1P1D, C224 2P1D, C512) |
| C96 1P2D | 1P2D: P TP8+DCP8, 2 x D TP16+DCP16 (40) | C96-1P2D-a1093473 | 17,132 | 290.3 | 106.3 | 8.2 | 1.51 | 0.9697 | **yes** | new config (shared prefill): +10% y vs frozen C48 1P1D at -2% x; trade-off TTFT p90 1.5 s vs 0.73 s (shared P), E2E-norm 106; meets pre-registered accept; gsm8k 0.9697 |
| C64 | 1P1D: P TP8+DCP8, D TP16+DCP16 (24) | C64-a1093680 | 20,778 | 232.8 | 128.7 | 9.1 | 0.85 | 0.9689 | no | borderline vs pre-registered (x 232.8 between fail 230 and accept 235; y at top of 20.2-20.8K); ~-3% x vs expectation while C48 is ~+3% on the same code -> run-to-run drift; chosen (only point at this x); gsm8k on the same hold; unselected by user 2026-11-15 (final set: C1, C8, C96 1P2D, C112 1P1D, C224 2P1D, C512) |
| C112 1P1D | 1P1D: P TP8+DCP8, D TP16+DCP16 (k3) (24) | C112-1P1D-a1094730 | 29,863 | 186.3 | 60.2 | 12.7 | 2.75 | 0.9682 | **yes** | phase-2 new config: +~6% y vs the C64-C224 chord at x 186 (proj +10%); TTFT p90 between accept 2.5 s and fail 3.0 s (single-P queue, Waiting mean ~4.5); kept per README rule (no chosen point within 10% x dominates it on the secondary panels); gsm8k on the same hold |
| C128 1P1D | 1P1D: P TP8+DCP8, D TP16+DCP16 (k3, lookup flag off) (24) | C128-1P1D-a1096270 | 32,194 | 184.6 | 53.0 | 13.9 | 3.20 | 0.9719 | no | mid-curve run (user 2026-11-15): meets pre-registered accept (>=32K @ >=175, TTFT <= 4.5 s); +7.5% y vs C112 at ~same x, TTFT p90 3.2 s vs 2.75 s; selection pending its lookup-flag twin frz-c128-1p1d-llh (a1096428) |
| C128 1P1D llh | 1P1D: P TP8+DCP8, D TP16+DCP16 (k3, lookup flag on) (24) | C128-1P1D-llh-a1096428 | 31,660 | 184.5 | 38.7 | 16.7 | 6.80 | 0.9689 | no | FAIL vs pre-registered (TTFT not down >=35%: 6.8 s vs 3.2 s flag-off). pfcap: flag acted (lookup 97->52 ms) but Mooncake load ready-wait 26->63 ms and prefill steps/s 5.0->4.4 on this xdr allocation (flag-off twin ran on spx): fabric/node confound, not the flag |
| C160 1P1D | 1P1D: P TP8+DCP8, D TP16+DCP16 (24) | C160-1P1D-a1093679 | 38,569 | 170.4 | 27.2 | 18.7 | 6.89 | 0.9689 | no | not used: TTFT p90 6.9 s (4x 2P1D) from prefill queueing on one P; replaced by C160 2P1D (phase 2) |
| C160 1P1D llh | 1P1D: P TP8+DCP8, D TP16+DCP16 (k3, lookup flag on) (24) | C160-1P1D-llh-a1096427 | 39,734 | 161.6 | 42.3 | 17.3 | 4.67 | 0.9674 | no | lookup_from_local_hit A/B vs C160 1P1D (a1093679): same y, TTFT p90 cut ~40% but still > 3 s (fails the TTFT rule vs C160 2P1D 1.6 s); x ~5% lower; between accept and fail lines |
| C160 2P1D | 2P1D: 2 x P TP8+DCP8, D TP16+DCP16 (32) | C160-2P1D-a1094590 | 31,049 | 161.3 | 73.5 | 14.5 | 1.64 | 0.9712 | no | phase-2 replacement for greyed C160 1P1D; on projection (~158 @ 30-31K, TTFT <= 2 s); old K3-AL run 31.2K @ 164; gsm8k on the same hold; unselected by user 2026-11-15 (final set: C1, C8, C96 1P2D, C112 1P1D, C224 2P1D, C512) |
| C224 2P1D | 2P1D: 2 x P TP8+DCP8, D TP16+DCP16 (32) | C224-2P1D-a1093685 | 41,591 | 113.1 | 54.6 | 19.5 | 1.88 | 0.9697 | **yes** | accept: old K3-AL run 42.9K @ 119 -> AL-adjusted ~114; gsm8k on the same hold |
| C256 2P1D | 2P1D: 2 x P TP8+DCP8, D TP16+DCP16 (32) | C256-2P1D-a1095076 | 44,940 | 99.9 | 44.1 | 22.4 | 2.43 | 0.9674 | no | phase-2 gap filler: only TTFT-reasonable point between x 80 (C512, TTFT 6.4 s) and 113; below the C224-C512 chord on y, non-dominated; gsm8k on the same hold; unselected by user 2026-11-15 (final set: C1, C8, C96 1P2D, C112 1P1D, C224 2P1D, C512) |
| C512 | 1P1D hybrid: P and D each DP4 x (TP4+DCP4), EP16 (32) | C512-a1093684 | 73,736 | 80.3 | 20.2 | 31.8 | 6.43 | 0.9659 | **yes** | accept vs pre-registered >=80 @ >=71K (old K3-AL run 76.5K @ 86); gsm8k running on the same hold |
<!-- /POINTS -->

## Sweep status
Topologies: C1-C160 1P1D = P TP8+DCP8 (8 GPUs) + D TP16+DCP16 (16 GPUs); C96 1P2D = one such P + two such D (40 GPUs); C224 2P1D = 2 x P TP8+DCP8 + D TP16+DCP16 (32 GPUs); C512 = hybrid P and D each DP4 x (TP4+DCP4) EP16 (32 GPUs).

Phase 1 (all launched 2026-11-14 17:46-19:14 PST): C1, C8, C48 1P1D, C64, C96 1P2D, C160 1P1D, C224, C512.
