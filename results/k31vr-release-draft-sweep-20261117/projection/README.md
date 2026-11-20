# Next-round configs: measured-data projections (2026-11-17)

This round only runs configs whose projection, built from measured capacity, shows a gain over the expected
release-draft front. Each projection was reviewed adversarially before it was queued.

How the projections were made (workflow `k31vr-pareto-projection`):
- Four independent analyses of the frozen sweep runs (`../../k31vr-frozen-sweep-20261114/runs`) and the measured
  release points:
  - prefill capacity;
  - decode capacity;
  - bottleneck class per point;
  - the SGLang VR recipes in NVIDIA/InferenceMAX PR #1910.
- A closed-loop model, `proj.py`, combines the measurements. It reproduces the frozen runs:

  | Run | Model | Measured |
  |---|---|---|
  | C512 | 73.5K @ 79.8 | 73.7K @ 80.3 |
  | C256 2P1D | 44.9K @ 98.7 | 44.9K @ 99.9 |
  | C160 1P1D | 38.2K @ 169.6 | 38.6K @ 170.4 |

- Two adversarial reviewers re-checked every candidate: one on capacity and arithmetic, one on feasibility and rules.
  The ranges below are the reviewers' corrected ones.

## Measured capacity (the inputs)

**Prefill load.**
- Only about 5.5K of the ~146K input tokens per request are fresh, since the prefix hit rate is about 95-98%.
- Throughput satisfies y = R × (ISL+OSL) / GPUs exactly, where R is completed requests per second.

**TP8+DCP8 prefill instance.**
- It saturates at about 6.4-6.5 req/s (36.5K fresh tokens/s).
- p90 TTFT ≈ 0.32 + 0.871·ρ/(1−ρ), with ρ = r/7.2 and r the request rate per instance. The RMSE over 13 runs is 0.25 s.
- Steps are only 5-8K tokens, so the fixed per-step cost dominates: about 120-170 ms.

**C512 hybrid prefill (DP4 × (TP4+DCP4), EP16).**
- It is the most GPU-efficient prefill measured: 1.09 req/s per GPU, against 0.80 for TP8 at its ceiling.
- Its TTFT is dominated by Mooncake loads (load_get 270 ms) plus a 1.8 s residual outside prefill.

**Decode.**
- The hybrid step is about 32.4 + 0.30·(B_rank − 52) ms, where B_rank is running requests per DP rank.
- The release draft cuts the step by about 5% and lifts the golden AL to 3.08 at k3, against 2.88 for the preview.
- Decode KV never binds at these loads: the hybrid uses about 57% of KV at C512.

**Release front so far.**

| Point | Status | y (tok/s/GPU) | x (p90 interactivity) |
|---|---|---|---|
| C1 | measured | 1,359 | 901 |
| C64 | measured | 21,527 | 300 |
| C128 | projected | about 34.2K | 232 |
| C160-llh | projected | about 39.0K | 212 |
| C512 | projected | about 77.1K | 96 |

Between x 96 and x 212 the release plan has no point. That band is where SGLang's C400-C720 rows sit.

## Status (2026-11-17 13:30 PST)

Paused by the user: no sweep beyond the 8 baseline points. The experiment dispatcher is stopped, its holds are cancelled and the
k5 variants are dropped from the baseline queue. The candidates below stay on record, and so does the max-throughput plan
(k31vr-max-throughput workflow: P24+D16 at 86-89K projected, gated on decode and prefill microbenchmarks).

## Queued experiments (paused)

All four run on 8 nodes (32 GPUs) and are derived from existing renders without code changes. The renders are in
`../../chosen_configs_nov_15/release_draft/next/configs`. Each runs AgentX plus its gsm8k twin on the same hold.

| # | Config | Render | Projected y | Projected x | TTFT p90 | Gate |
|---|---|---|---|---|---|---|
| 1 | TP16 2P1D C224, release draft | `dsrel-c224-*` (frozen C224 2P1D render) | 42.0-44.1K | 133-145 | 2.1-3.0 s | none |
| 2 | Hybrid 1P1D C384 | `dsrelc384-c512-*` | 61-65K | 104-111 | 3.5-5.0 s | `gates/c512ok` |
| 3 | Hybrid 1P1D C320 | `dsrelc320-c512-*` | 52-56K | 113-124 | 2.9-4.4 s | `gates/c512ok` |

For the two hybrid rows, the render is the release C512 render with only the concurrency changed (`conc_variant.py`).
Decode max-num-seqs stays 128 per rank (512 in total), above the new concurrency.

**Gate `c512ok`.** The hybrid inputs are unvalidated on the fixed tree. The release C512's first attempt crashed on
rel-fa43a52, and the hybrid step and TTFT residual are calibrated on one frozen point. So the C384 and C320 rows launch
only after the release C512 AgentX run on rel-1767cf7 is healthy:
- NIXL transfers succeed;
- there are no engine errors;
- y is roughly within the model, about 70K or more.

**Run order and conditions.**
- C384 comes before C320.
- HYB-C448 (71-73K @ 101-103) is not queued. It lies on the chord between C384 and C512, so it waits for both
  measurements.

## Rejected (projected no gain, or breaks a rule)

- **SGLang-style 1.5:1 prefill-to-decode, 40 GPUs (C576-C720):** 70-81K @ 71-85. Dominated per GPU by the hybrid
  C512.
- **2 hybrid prefill instances + 1 hybrid decode, 48 GPUs:** prefill would sit about 40% idle. Dominated.
- **Hybrid 1P1D at C576-C640:** already prefill-bound at C512, so more users add TTFT (8-10 s), not throughput.
- **TP16 3P1D C288-C320, 40 GPUs:** 42-45K @ 96-107. Dominated by the hybrid points.
- **TP16 2P1D C256, release draft:** 47.6K @ 125. Dominated by hybrid C320/C384 on throughput.
  - Kept in reserve if the hybrid points underdeliver.
- **Stream interval 20-30, KV-aware router flags, hybrid at k5:** no measured data to project from. These need a
  microbenchmark or A/B first.
- **Rule violations:**
  - a binding decode cap;
  - prefill and decode with different DCP layouts;
  - more than 18 nodes on one rack.
