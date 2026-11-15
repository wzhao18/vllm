# Kimi K3.1 AgentX on VR200: chosen configs (2026-11-15)

This is the final selected set of 8 VR200 configurations. It was picked from the 17 measured runs of the frozen-code sweep in
`../k31vr-frozen-sweep-20261114/`, which keeps every run and the selection rationale.

## Selection
The selection favours throughput vs P90 interactivity. Each point also stays close to the front on P90 E2E-normalized
interactivity and P90 E2E latency, and keeps an acceptable P90 TTFT. The user chose the set on 2026-11-15.

| point | topology (GPUs) | TPS/GPU | P90 intvty | P90 E2E-norm | P90 E2E (s) | P90 TTFT (s) | gsm8k strict | vs GB300 per GPU / per MW | config | run |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 1P1D: P TP8+DCP8, D TP16+DCP16 (24) | 1,342 | 779 | 329 | 6.5 | 0.36 | 0.9659 | beyond GB300 range | `configs/C1/` | `runs/C1-a1093683/` |
| C16 1P2D | 1P2D: P TP8+DCP8, 2 x D TP16+DCP16 (k7) (40) | 3,104 | 601 | 294 | 3.9 | 0.39 | 0.9704 | beyond GB300 range | `configs/C16-1P2D/` | `runs/C16-1P2D-a1096999/` |
| C72 1P3D | 1P3D: P TP8+DCP8, 3 x D TP16+DCP16 (k7) (56) | 10,221 | 397 | 140 | 6.0 | 1.08 | 0.9666 | 4.26x / 2.74x | `configs/C72-1P3D/` | `runs/C72-1P3D-a1098025/` |
| C48 1P1D | 1P1D: P TP8+DCP8, D TP16+DCP16 (24) | 15,544 | 296 | 158 | 7.6 | 0.73 | 0.9674 | 3.81x / 2.45x | `configs/C48-1P1D/` | `runs/C48-1P1D-a1093681/` |
| C64 | 1P1D: P TP8+DCP8, D TP16+DCP16 (24) | 20,778 | 233 | 129 | 9.1 | 0.85 | 0.9689 | 3.29x / 2.11x | `configs/C64/` | `runs/C64-a1093680/` |
| C128 1P1D | 1P1D: P TP8+DCP8, D TP16+DCP16 (k3, lookup flag off) (24) | 32,194 | 185 | 53 | 13.9 | 3.20 | 0.9719 | 3.24x / 2.08x | `configs/C128-1P1D/` | `runs/C128-1P1D-a1096270/` |
| C160 1P1D llh | 1P1D: P TP8+DCP8, D TP16+DCP16 (k3, lookup flag on) (24) | 39,734 | 162 | 42 | 17.3 | 4.67 | 0.9674 | 3.30x / 2.12x | `configs/C160-1P1D-llh/` | `runs/C160-1P1D-llh-a1096427/` |
| C512 | 1P1D hybrid: P and D each DP4 x (TP4+DCP4), EP16 (32) | 73,736 | 80 | 20 | 31.8 | 6.43 | 0.9659 | 2.76x / 1.77x | `configs/C512/` | `runs/C512-a1093684/` |

GB300 baseline: SemiAnalysis InferenceX Kimi-K3.1-Pro GB300 dynamo-vllm (7 runs, 2026-11-10), interpolated along its front at the same
P90 interactivity. Per MW assumes VR200 3.3 kW and GB300 2.12 kW all-in per GPU.

## Fixed for every config
- **Code:** frozen at `wzhao/agentx-vr-nov-13` **b68d67e7e** (Hecate tree `hti/lowlat/trees/ll-integ-dea4ebd06f`).
- **Speculative decoding:** DSpark with probabilistic drafting and synthetic rejection at the **InferenceX Kimi K3.1 golden AL**
  (`golden_al_distribution/kimi-k3.1-pro__dspark__kimi-k3.1-pro-dspark-preview__probabilistic-block.yaml`): k7 = 3.69, k3 = 2.88.
- **Decode:** `watermark: 0.05`.
- **Precision:** BF16 linears, MXFP4 MoE, FP8 MLA KV, FP32 KDA.
- **Measurement:** each point is one 60-min AgentX run (profiling phase). It is followed by a full gsm8k on the same allocation and
  deployment, with real rejection.
- **Prefill MooncakeStore `lookup_from_local_hit`:** true in C16 1P2D, C72 1P3D, C48 1P1D, C64 and C160 1P1D llh; false in
  C128 1P1D and C512, as they were measured. C1 uses NIXL only, with no Mooncake store, so the flag does not apply.

## Layout
- `configs/<point>/agentx/`: the exact srtctl render that produced the AgentX run. It includes `render/config.yaml` (full serving
  config), `render/sbatch_script.sh`, `render/render.json`, `workflow.env.resolved` (CONC, DURATION, aiperf settings),
  `golden-acceptance.json`, `point.env`, `hybrid.yaml` and `FRZ.txt` (derivation).
- `configs/<point>/gsm8k/`: the gsm8k twin. It is the same deployment with real rejection plus the eval settings.
- `runs/<point>-a<job>/`: the AgentX artifacts under `agentx/` (aiperf summary, server logs, gzipped per-request records where
  they fit), `metrics.json` (computed by `frz_metrics.py`), `gsm8k/` (eval artifacts and `score.json`) and `meta.json`.
- `points.json`: all 17 sweep runs. `chosen` marks this set.
- `pareto4_per_gpu.png` and `pareto4_per_mw.png`: the chosen set (blue), the other measured runs (grey) and GB300 (orange). Regenerate
  them with `python3 plot_pareto4.py`.

## Reproduce a point
On Hecate, run the render as an srtctl job (`run_in_alloc.sh <render dir>` inside an allocation of `render.json`'s
`total_nodes`), then run the gsm8k twin on the same allocation. Compute metrics with `python3 frz_metrics.py <output dir>`.
