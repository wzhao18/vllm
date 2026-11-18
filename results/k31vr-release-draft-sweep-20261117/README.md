# K3.1 VR200 AgentX sweep: release DSpark draft on the real K3.1-Pro weights (2026-11-17)

This sweep reruns the 8 chosen points of the frozen VR200 sweep (`../k31vr-frozen-sweep-20261114`) with two changes, so the
comparison isolates them:

- **Draft model.** The released draft `Inferact/Kimi-K3.1-Pro-DSpark @ e4e6e626` (GQA, sliding window 2048) replaces the
  preview draft. It runs on the optimized specialized-decode path in branch `wzhao/k31-dspark-release`.
- **Acceptance length.** AgentX uses the release golden table: k7 4.16, k5 3.81, k3 3.08 (the preview table was k7 3.69).
- **Target weights.** The real `moonshotai/Kimi-K3.1-Pro @ d9e99b07` replaces the rainhollow test weights.

Everything else is the frozen sweep's render: topology, concurrency, precision and AgentX client. The configs are in
`../chosen_configs_nov_15/release_draft/` (`configs/` + `k5/configs/`).

- **Code:** frozen tree `rel-1767cf7`, i.e. commit 1767cf7 of `wzhao/k31-dspark-release`. Prefill and decode both run it
  (PYTHONPATH of both roles).
- **AgentX:** 60-minute profiling phase and synthetic acceptance at the golden AL, as in the frozen sweep.
- **gsm8k:** the twin of each AgentX run uses real rejection sampling, on the same hold. Its decode logs give the real AL.

## Release vs preview draft

<!-- CMP -->
| point | TPS/GPU (preview -> release) | P90 intvty (preview -> release) | P90 TTFT s (preview -> release) | gsm8k strict | real AL (k) |
|---|---|---|---|---|---|
| C1 | 1,342 -> 1,359 (+1.2%) | 778.6 -> 901.4 (+15.8%) | 0.36 -> 0.40 | 0.9659 -> 0.9795 | 4.763 (7) |
| C16 1P2D | 3,104 -> 3,162 (+1.9%) | 600.6 -> 701.6 (+16.8%) | 0.39 -> 0.44 | 0.9704 -> 0.978 | 4.752 (7) |
| C72 1P3D | 10,221 -> 10,460 (+2.3%) | 397.5 -> 498.1 (+25.3%) | 1.08 -> 0.95 | 0.9666 -> 0.9788 | 4.764 (7) |
| C48 1P1D | 15,544 -> 15,934 (+2.5%) | 296.1 -> 367.2 (+24.0%) | 0.73 -> 0.71 | 0.9674 -> 0.9788 | 4.793 (7) |
| C64 | 20,778 -> 21,527 (+3.6%) | 232.8 -> 300.0 (+28.9%) | 0.85 -> 0.96 | 0.9689 -> 0.9803 | 4.807 (7) |
| C128 1P1D | 32,194 -> 33,119 (+2.9%) | 184.6 -> 212.2 (+14.9%) | 3.20 -> 3.00 | 0.9719 -> 0.9788 | 3.293 (3) |
| C512 | 73,736 -> 64,045 (-13.1%) | 80.3 -> 85.5 (+6.5%) | 6.43 -> 16.21 | 0.9659 -> 0.9788 | 3.248 (3) |
<!-- /CMP -->

## Points

<!-- POINTS -->
| point | topology (GPUs) | AgentX run | TPS/GPU | P90 intvty | P90 E2E-norm | P90 E2E (s) | P90 TTFT (s) | gsm8k strict | chosen | note |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 1P1D: P TP8+DCP8, D TP16+DCP16 (24) | C1-a1118776 | 1,359 | 901.4 | 332.2 | 5.6 | 0.40 | 0.9795 | **yes** |  |
| C16 1P2D | 1P2D: P TP8+DCP8, 2 x D TP16+DCP16 (k7) (40) | C16-1P2D-a1118773 | 3,162 | 701.6 | 297.4 | 3.5 | 0.44 | 0.978 | **yes** |  |
| C72 1P3D | 1P3D: P TP8+DCP8, 3 x D TP16+DCP16 (k7) (56) | C72-1P3D-a1118772 | 10,460 | 498.1 | 164.3 | 4.9 | 0.95 | 0.9788 | **yes** |  |
| C48 1P1D | 1P1D: P TP8+DCP8, D TP16+DCP16 (24) | C48-1P1D-a1118778 | 15,934 | 367.2 | 177.8 | 6.1 | 0.71 | 0.9788 | **yes** |  |
| C64 | 1P1D: P TP8+DCP8, D TP16+DCP16 (24) | C64-a1118777 | 21,527 | 300.0 | 136.3 | 7.2 | 0.96 | 0.9803 | **yes** |  |
| C128 1P1D | 1P1D: P TP8+DCP8, D TP16+DCP16 (k3, lookup flag off) (24) | C128-1P1D-a1118779 | 33,119 | 212.2 | 61.5 | 11.6 | 3.00 | 0.9788 | **yes** |  |
| C512 | 1P1D hybrid: P and D each DP4 x (TP4+DCP4), EP16 (32) | C512-a1118813 | 64,045 | 85.5 | 8.5 | 35.5 | 16.21 | 0.9788 | **yes** | REGRESSION under root-cause: -13% tput, p90 TTFT 16.2s vs frozen 6.4s (xdr IB vs frozen spx RoCE; real vs test weights); not accepted yet |
<!-- /POINTS -->

![per GPU](pareto4_per_gpu.png)
![per MW](pareto4_per_mw.png)

## Incidents

- **C512, first attempt (tree `rel-fa43a52`).** A decode engine died about 37 minutes in, during AgentX warmup:
  `AssertionError` in `KVCacheManager.truncate_computed_blocks`.
  - Cause: NIXL's divergent local-hit lookup let only Mamba groups lag the full-attention hit. The release draft's
    slot-packed sliding-window group caches window tails only at state boundaries, so it lagged too. The scheduler's
    partial-tail truncation then overran its block list.
  - Fix: commit 1767cf7. A non-Mamba group that lags the full-attention hit now falls back to the reconciled lookup.
    Two unit tests were added; the core tests show the same 24 failures as the previous tree.
  - Sweep reset: every point reruns on `rel-1767cf7`. That includes C1, whose first launch (07:34) used the old tree
    and was stopped after 37 minutes.
  - The C512 gsm8k twin on the old tree scored strict 0.9803 with real AL 3.25 at k3. It is not counted.

## Files

- `runs/<label>-a<job>/`: one point. Contents:
  - `agentx/`: logs and config.
  - `gsm8k/`: logs and config, plus `score.json` (strict, flexible, real AL).
  - `metrics.json` and `meta.json`.
- `collect.sh`: copies one point from Hecate and computes `metrics.json` (with `frz_metrics.py`, unchanged from the
  frozen sweep) and `gsm8k/score.json`.
- `update_points.py`: rebuilds `points.json`, the two tables above and the plots.
- `plot_pareto4.py`: the frozen sweep's plot, adding the frozen preview-draft points as a grey dashed series.
