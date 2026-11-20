# MegaMoE at EP16 / EP28 / EP32: FlashInfer SM107 EP32 build vs the perf model (2026-11-18)

Measured on Hecate by the MegaMoE lane: the build is tag `wzhao/kimi-k3-megamoe-sm107-ep32-20261118`, merge commit
a368c5f66 (lane `wzhao/sm107-r3-ep` 89fafd77e, i.e. 6e438852b plus EP28/EP32 support). Source:
`megamoe-bench/opt/ep32/{SUMMARY.md,tables.md}` and the `ep32-*` rows of `opt/results.jsonl`, copied to `raw/`.

Run setup:
- hold 1137209, 8 nodes in one NVL72 segment
- bench chain `fi_l1f5.py` with 50 samples x 10 replays, taking the slowest rank per sample
- random routing, seed 0
- the bf16-reference check passed in every run (NVFP4 rel_l2 0.165-0.169, MX 0.047-0.049)
- the cross-EP outputs are bitwise equal to EP16 at 16/16 points
- the EP16 rows are bitwise equal to 6e438852b and their timing is within 0.3%

## Rows

`rows.json` is built by `python3 mkrows.py`:
- **Kept:** decode-shaped bench mode (`ep32-bn`, capacity = T, T = 1-512 tokens/rank) and prefill bench mode
  (`ep32-pf`, `pf16k`, `pf16g`, capacity = T, T = 4096-16384), at EP16, EP28 and EP32, for both quants.
- **Statistic:** the median over repeats. EP32 has 2 repeats, except T16384 with 1.
- **Not kept:** vLLM mode (within 1-3% of bench mode), the C (EP16 regression) rows and the X (cross-EP) rows.
- **Model columns:** `t_ach_us` / `t_sol_us` are the current model's MegaMoE layer at the same EP and tokens per rank
  (`calibrate.megamoe_model_layer`). That is decode environment below 4096 rows and prefill environment above;
  `derate = measured / t_ach`.

## Results vs the current model (achievable)

| quant | EP | T 1 | T 4-128 | T 256 | T 512 | T 4096 | T 8192 | T 16384 |
|---|---|---|---|---|---|---|---|---|
| MX | 16 | 3.04 | 1.48-1.82 | 1.55 | 1.85 | 1.34 | 1.34 | 1.35 |
| MX | 28 | 3.14 | 1.82-1.95 | - | 1.75 | - | 1.35 | - |
| MX | 32 | 3.23 | 1.97-2.11 | 2.20 | 1.80 | 1.28 | 1.33 | **1.57** |
| NVFP4 | 16 | 2.76 | 1.30-1.69 | 1.41 | 1.74 | 1.21 | 1.17 | 1.24 |
| NVFP4 | 32 | 2.91 | 1.71-1.89 | 2.29 | 1.93 | 1.21 | 1.22 | **1.54** |

**Upper bound.** The model's MegaMoE layer is below every measured row, and `test_megamoe_rows_upper_bound` checks
this.

**Decode at EP32 (T 4-128).** The measured kernel is 0.70-0.85x of EP16: it streams half the expert weights. The model
predicts about 0.5-0.55x, so the EP32 decode derate (about 2.0) is higher than the EP16 one (about 1.5). A fixed
chain of about 50 us at T1, the same at every EP, dominates; the model has 17 us.

**Prefill cliff at T16384, EP32 vs EP16.** NVFP4 is +30% and MX is +21%; EP28 at T8192 is also slower than EP32.
- This is a kernel issue: push-router or landing/combine scaling with routes x fan-out, not root-caused.
- It is **not modelled** in the achievable column.
- The calibrated column includes it, because it is what the kernel measures.

## Use in calibrate.py

- `megamoe_table` merges these rows with the wave-1 moe-lane rows (EP8 / EP16) and takes the median derate per
  (EP, T).
- Every EP32 layout takes the measured EP32 family; nothing is transferred from EP16:
  - decode (D3-32, D4-32, D7-x-32, D8-32, D9-32): x 1.2 for the vendored in-step kernel of the release tree;
  - prefill (P3-32, P6-32, P9-32): the MegaMoE part is replaced at the target's EP and tokens per rank, and the rest
    of the analog's measured step is kept (`calibrate.prefill_calibrated`).
- The release tree's `megamoe_rubin` refuses EP32 (`KnobLookupError`: knob records exist only for world sizes 4/8/16),
  so the EP32 prefill numbers assume this FI build.
