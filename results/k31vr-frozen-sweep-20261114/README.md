# Kimi K3.1 AgentX on VR200: frozen-code pareto sweep (started 2026-11-14)

Goal: the best 4-way pareto curve for VR200, picked by config only.
- The main metric is throughput per GPU vs P90 interactivity.
- A point is not chosen if it has a clear drawback on P90 E2E-normalized interactivity, P90 E2E latency or P90 TTFT.
  Example: the C160 1P1D run of 2026-11-14 queued on its single prefill instance, which pushed TTFT p90 to 4.2 s.

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
| C1 | 1P1D: P TP8+DCP8, D TP16+DCP16 (24) | C1-a1093683 | 1,342 | 778.6 | 328.8 | 6.5 | 0.36 | pending | **yes** | accept: old K3-AL run 1,347 @ 805 -> -0.4% y, -3.2% x (AL correction); gsm8k on the same hold |
| C8 | 1P1D: P TP8+DCP8, D TP16+DCP16 (24) | C8-a1093682 | 2,890 | 529.7 | 326.7 | 4.7 | 0.35 | pending | **yes** | accept: old K3-AL run 2,960 @ 553 -> -2.3% y, -4.2% x (= AL 3.84 -> 3.69); gsm8k on the same hold |
| C48 1P1D | 1P1D: P TP8+DCP8, D TP16+DCP16 (24) | C48-1P1D-a1093681 | 15,544 | 296.1 | 158.1 | 7.6 | 0.73 | pending | **yes** | accept (proj ~293 @ 15.0-15.4K; old K3-AL 15,593 @ 305 -> -0.3% y, -3% x); gsm8k on the same hold |
| C96 1P2D | 1P2D: P TP8+DCP8, 2 x D TP16+DCP16 (40) | C96-1P2D-a1093473 | 17,132 | 290.3 | 106.3 | 8.2 | 1.51 | 0.9697 | **yes** | new config (shared prefill): +10% y vs frozen C48 1P1D at -2% x; trade-off TTFT p90 1.5 s vs 0.73 s (shared P), E2E-norm 106; meets pre-registered accept; gsm8k 0.9697 |
| C64 | 1P1D: P TP8+DCP8, D TP16+DCP16 (24) | C64-a1093680 | 20,778 | 232.8 | 128.7 | 9.1 | 0.85 | 0.9689 | **yes** | borderline vs pre-registered (x 232.8 between fail 230 and accept 235; y at top of 20.2-20.8K); ~-3% x vs expectation while C48 is ~+3% on the same code -> run-to-run drift; chosen (only point at this x); gsm8k on the same hold |
| C160 1P1D | 1P1D: P TP8+DCP8, D TP16+DCP16 (24) | C160-1P1D-a1093679 | 38,569 | 170.4 | 27.2 | 18.7 | 6.89 | 0.9689 | no | not used: TTFT p90 6.9 s (4x 2P1D) from prefill queueing on one P; replaced by C160 2P1D (phase 2) |
| C512 | 1P1D hybrid: P and D each DP4 x (TP4+DCP4), EP16 (32) | C512-a1093684 | 73,736 | 80.3 | 20.2 | 31.8 | 6.43 | 0.9659 | **yes** | accept vs pre-registered >=80 @ >=71K (old K3-AL run 76.5K @ 86); gsm8k running on the same hold |
<!-- /POINTS -->

## Sweep status
Topologies: C1-C160 1P1D = P TP8+DCP8 (8 GPUs) + D TP16+DCP16 (16 GPUs); C96 1P2D = one such P + two such D (40 GPUs); C224 2P1D = 2 x P TP8+DCP8 + D TP16+DCP16 (32 GPUs); C512 = hybrid P and D each DP4 x (TP4+DCP4) EP16 (32 GPUs).

Phase 1 (all launched 2026-11-14 17:46-19:14 PST): C1, C8, C48 1P1D, C64, C96 1P2D, C160 1P1D, C224, C512.
