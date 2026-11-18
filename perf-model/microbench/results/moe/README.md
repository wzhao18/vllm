# Lane moe: MegaMoE at EP8 / EP16 vs perf-model MOE.* rows

Scope (2026-11-18): every EP8/EP16 `megamoe` case in `microbench/matrix.json` (priority + cover topologies,
EP <= 16), both checkpoints (mxfp4_release = MXFP4 weights x MXFP8 activations, nvfp4_modelopt = NVFP4 W4A4),
random routing. 78 (topology, regime) rows over 20 distinct (quant, EP, tokens/rank) shapes. All are measured;
none are missing.

## What ran

- **Reused:** the final A/B rows of the MegaMoE SOL work, baseline `B10`. They are in Hecate
  `$W/megamoe-bench/results/rows.jsonl`, summarised in `opt/final_ab/rows_final.txt`, and copied here as
  `b10_rows_megamoe_bench.jsonl`. They cover T = 1..8192 tokens/rank at EP8 (2 nodes) and EP16 (4 nodes), in three
  modes: bench, `+kernel` and `+vllm`. For each key the latest row is used.
- **New (this lane):** T = 16384 at EP16, needed by P3-16, P9 and P9-16. The task was
  `queue/done/18-nq4-mbmoe-ep16t16k-m30.sh` (log beside it), run in hold job 1137217 on
  hecate0361,0366,0367,0369 at 2026-11-18 03:01-03:08 PST. It ran both quants in bench and `+kernel` mode, with a
  T = 8192 same-node anchor. Rows: `lane_rows.jsonl`; the T = 16384 subset used here is `lane_rows_t16384.jsonl`.
  Lane runner: `$W/mb/moe/scripts/` holds a copy of `megamoe-bench/scripts/run_b10.sh` and a `common.sh` override
  that takes the hosts from `NQ_HOSTS` and writes rows to `$W/mb/moe/results/rows.jsonl`. Status:
  `$W/mb/moe/status.txt`. Raw logs: `$W/megamoe-bench/logs/B10-*-ep16-random-111803*.log`.
- **Mapping:** `python3 mkrows.py b10_rows_megamoe_bench.jsonl lane_rows_t16384.jsonl rows.json` reads the matrix
  and `out/optables.json`.

## Versions

- MegaMoE: FlashInfer **6e438852b**, tag `wzhao/kimi-k3-megamoe-sm107-final-20261117` (the newest build), from the
  read-only overlay `frozen/fi-6e438852b`. Identity was checked in every log (MMIDENT/MMTREE).
- Frozen vLLM 036d832e7. These numbers do **not** use the vLLM-vendored `megamoe_rubin`.
- Image: `k31vr-kmc_kmr-sweep-freeze_p242fb49cd4-..._fi0.7.3.dev20261031_mc0.3.14_...sqsh`; torch
  2.16.0.dev20261009+cu134, cutlass-dsl 4.8.0.
- Harness: `fi_bench2.py` (`bench_kimi_k3_moe_ep.py`, `--provider current`, `--wire bf16`). Settings: capacity =
  T, CUDA graph with 10 calls, 10 replays, 50 samples, the slowest rank per sample, median. A correctness check
  (rel-L2 vs BF16) passed on every point.
- The logged `quant_kind` was asserted: `mxfp4_mxfp8` on every MX row (MXFP8 activations are pinned at all T,
  including > 512) and `nvfp4` on the NV rows.
- Weights are the deterministic packed layout-test weights, not the real checkpoint (timing only).

## Comparison basis

The measured number is the whole fused MegaMoE layer: BF16 in, input quantization, routing, NVLink dispatch, fc1,
act, fc2 and combine. The kernel is persistent and phase-interleaved (`schedule_policy phase_interleave`), so
**dispatch, the GEMMs and combine cannot be timed separately with kernel timers**. `+kernel` leaves out the
staging (input quant and routing), which is 1-2 us in decode and 0.5% in prefill.

The model charges a MoE layer as the sum of `MOE.dispatch + fc1 + act + fc2 + combine` (`t_ach_ops`) **plus** the
P11/P12 EP comm, added serially (cost.layer_time). So `t_ach_us` and `t_sol_us` in rows.json are ops + P11 + P12, and
`derate = measured / t_ach`. `derate_vs_ops_only` is kept for reference.

## Results (median us per MoE layer call)

| quant | EP | T/rank | topologies | measured | +kernel | t_sol | t_ach (ops+P11/P12) | derate | vs ops-only | %SOL |
|---|---|---|---|---|---|---|---|---|---|---|
| mxfp4w_mxfp8a | EP8 | 4 | D2 r1 | 89.4 | 87.5 | 48.1 | 71.8 (62.1+9.7) | 1.25 | 1.44 | 54% |
| mxfp4w_mxfp8a | EP8 | 32 | D2 r8 | 145.1 | 143.1 | 100.4 | 154.1 (141.0+13.1) | 0.94 | 1.03 | 69% |
| mxfp4w_mxfp8a | EP8 | 2048 | P1, P5-8 | 511.4 | 538.2 | 335.9 | 499.2 (240.8+258.4) | 1.02 | 2.12 | 66% |
| mxfp4w_mxfp8a | EP16 | 4 | D3/D4-16/D5/D6/D7-x-16/D8-16/D9-16 r1 | 77.8 | 75.7 | 39.6 | 59.1 (48.5+10.6) | 1.32 | 1.60 | 51% |
| mxfp4w_mxfp8a | EP16 | 32 | same, r8 | 91.4 | 89.5 | 56.8 | 85.9 (71.7+14.2) | 1.06 | 1.27 | 62% |
| mxfp4w_mxfp8a | EP16 | 128 | D3/D5/D7-x-16/D8-16 r32 | 97.1 | 95.2 | 67.5 | 101.3 (74.6+26.8) | 0.96 | 1.30 | 70% |
| mxfp4w_mxfp8a | EP16 | 2048 | P2b | 465.0 | 463.6 | 344.4 | 512.5 (235.5+277.0) | 0.91 | 1.97 | 74% |
| mxfp4w_mxfp8a | EP16 | 4096 | P2, P4-4, P5-4 | 900.7 | 896.9 | 680.1 | 1015.1 (471.0+544.1) | 0.89 | 1.91 | 76% |
| mxfp4w_mxfp8a | EP16 | 4096 | P6-4-16 (dedup model) | 900.7 | 896.9 | 548.1 | 824.9 (471.0+353.9) | 1.09 | 1.91 | 61% |
| mxfp4w_mxfp8a | EP16 | 8192 | P4-2 | 1789.0 | 1781.0 | 1351.8 | 2020.0 (942.0+1078.0) | 0.89 | 1.90 | 76% |
| mxfp4w_mxfp8a | EP16 | 8192 | P6-16 (dedup model) | 1789.0 | 1781.0 | 1087.8 | 1639.8 (942.0+697.8) | 1.09 | 1.90 | 61% |
| mxfp4w_mxfp8a | EP16 | 16384 | P3-16, P9, P9-16 (new) | 3427.1 | 3413.5 | 2695.0 | 4030.2 (1883.8+2146.4) | 0.85 | 1.82 | 79% |
| nvfp4w_nvfp4a | EP8 | 4 | D2 r1 | 89.8 | 88.3 | 50.4 | 75.3 (65.7+9.6) | 1.19 | 1.37 | 56% |
| nvfp4w_nvfp4a | EP8 | 32 | D2 r8 | 144.1 | 142.3 | 105.1 | 161.6 (149.1+12.5) | 0.89 | 0.97 | 73% |
| nvfp4w_nvfp4a | EP8 | 2048 | P1, P5-8 | 379.4 | 367.9 | 287.3 | 424.2 (204.3+219.9) | 0.89 | 1.86 | 76% |
| nvfp4w_nvfp4a | EP16 | 4 | D3/D4-16/D5/D6/D7-x-16/D8-16/D9-16 r1 | 77.0 | 75.2 | 41.3 | 61.8 (51.3+10.5) | 1.25 | 1.50 | 54% |
| nvfp4w_nvfp4a | EP16 | 32 | same, r8 | 85.7 | 84.0 | 58.9 | 89.3 (75.7+13.6) | 0.96 | 1.13 | 69% |
| nvfp4w_nvfp4a | EP16 | 128 | D3/D4-16/D5/D7-x-16/D8-16/D9-16 r32 | 88.1 | 86.5 | 68.0 | 102.4 (78.3+24.2) | 0.86 | 1.13 | 77% |
| nvfp4w_nvfp4a | EP16 | 2048 | P2b | 354.7 | 351.7 | 257.2 | 378.9 (143.2+235.8) | 0.94 | 2.48 | 72% |
| nvfp4w_nvfp4a | EP16 | 4096 | P2, P4-4, P5-4 | 653.1 | 648.9 | 495.1 | 732.6 (271.1+461.5) | 0.89 | 2.41 | 76% |
| nvfp4w_nvfp4a | EP16 | 4096 | P6-4-16 (dedup model) | 653.1 | 648.9 | 383.5 | 571.8 (271.1+300.7) | 1.14 | 2.41 | 59% |
| nvfp4w_nvfp4a | EP16 | 8192 | P4-2 | 1254.2 | 1247.8 | 979.6 | 1455.1 (542.2+912.9) | 0.86 | 2.31 | 78% |
| nvfp4w_nvfp4a | EP16 | 8192 | P6-16 (dedup model) | 1254.2 | 1247.8 | 756.5 | 1133.6 (542.2+591.4) | 1.11 | 2.31 | 60% |
| nvfp4w_nvfp4a | EP16 | 16384 | P3-16, P9, P9-16 (new) | 2649.3 | 2632.7 | 1950.9 | 2900.4 (1084.3+1816.1) | 0.91 | 2.44 | 74% |

T = 8192 same-node anchor from the new task:

- MX: 1690.3 us here vs 1789.0 us in the final rows (hecate0314-0317), -5.5%.
- NV: 1248.8 us vs 1254.2 us, -0.4%.

MX prefill therefore carries roughly ±5% variance from node to node (MMA-bound at a power-limited clock). The final
rows are kept for T <= 8192.

## Derating by op family

- **Decode, T = 4 tokens/rank (r1): derate 1.19-1.32, 51-56% of SOL.** The fixed chain (router, the sizes round,
  dispatch to the first fc1, and the tail) costs a constant 26-30 us. The model does not charge it beyond the
  0.78 us per-kernel gap. The FINAL_REPORT of the MegaMoE SOL work also places the gap there.
- **Decode, T = 32/128 (r8/r32): derate 0.86-1.06.** The weight stream of all 56 (EP16) or 112 (EP8) touched
  experts dominates, and the kernel matches or beats the model's achievable.
- **Prefill without dedup (P1/P2/P2b/P3-16/P4/P5/P9): derate 0.85-1.02, 66-79% of SOL.** The model adds the
  P11/P12 NVLink time serially after the GEMMs. The fused kernel overlaps them, so the measured time falls below
  `t_ach`.
  - Measured vs ops-only is 1.8-2.1x (MX) and 1.9-2.5x (NV). The layer is NVLink-bound for NV and MMA/clock-bound
    for MX.
  - **Model fix suggestion:** MegaMoE should be `max(ops, P11 + P12) + exposed fraction`, not a sum.
- **Prefill with the dedup model (P6-4-16, P6-16): derate 1.09-1.14, 59-61% of SOL.** The model assumes rank-level
  dedup dispatch and pre-combine (P11/P12 scaled by U/k). The shipped kernel has no dedup: C7 dedup measured
  1.4-3.8% *slower*, because the link is already near its practical byte rate. These rows are therefore
  optimistic about link bytes, but even after that correction the derate stays under 2.

## KERNEL-DEBUG flags

**None.** No row has a derate above 2.0 against the model's full MoE-layer `t_ach`.

Against ops-only, the prefill rows read 1.8-2.5. That gap is the NVLink dispatch/combine (P11/P12), not a kernel
defect. The MegaMoE SOL work already found link efficiency near its practical limit: 1.144 vs 1.18 TB/s on 256 B
stores.

## Caveats and unrun items

- Separate dispatch/GEMM/combine times are **not available**: one persistent fused kernel. A per-phase split would
  need in-kernel clock64 instrumentation, which this harness does not have.
- T = 16384 is beyond the vLLM-mode workspace cap of 8192 tokens/rank. With that cap, vLLM would run a 16K chunk as
  2 x 8192. That costs 2 x 1690-1789 = 3.4-3.6 ms (MX) or about 2.5 ms (NV), roughly the same as a single 16K call.
  vLLM mode (`+vllm`, cap 8192) is within about 2% of bench mode at T <= 8192.
- In production, the vLLM-vendored `megamoe_rubin` (older) switches mxfp4 prefill to NVFP4 activations above
  512 tokens/rank. That deployment would run the NV rows (about 653 vs 901 us at T = 4096), not the pinned MXFP8
  rows here.
- Out of scope (EP > 16): P6 (EP64), D4-32, D7-4/D7-2/D7-8, D8, and the TP-experts cases.


## Review fixes (adversarial review, 2026-11-18)

No row values changed. Spot checks passed: derate = measured / t_ach on every row, and `t_ach_us` / `t_sol_us` match
the current `out/optables.json` (the model lane's re-scope did not change any per-op value used here).
- moe: the T=16384 rows match `$W/mb/moe/status.txt` (MX 3427.1 us, NV 2649.3 us, check=pass); the MMIDENT line shows frozen fi-6e438852b and `--quant-kind mxfp4_mxfp8`. Tokens/rank per topology checked against the shard shapes (D2 r1 = 4, D3 r1/r8/r32 = 4/32/128, P1 2048, P2 4096, P4-2 8192, P3-16 16384).
- Cross-lane caution: the prefill, decode (D3) and gap512 in-step numbers use the vendored `megamoe_rubin`, which is 1.45x (EP8, 2048 tok/rank: 742 vs 511 us) and 1.2x (EP16, 4096: 1096 vs 901 us) slower than these B10 rows. Do not mix the two when attributing a step gap.
