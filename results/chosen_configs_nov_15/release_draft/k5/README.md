# k5 renders of C128 1P1D and C160 1P1D llh (release draft)

The user approved moving the two k3 points with TP16 = DCP16 decode, C128 1P1D and C160 1P1D llh (max-num-seqs 160), to
draft length k5. The release golden AL ratio AL5 / AL3 = 3.81 / 3.08 = 1.237, so k5 wins at a batch where
step(q6) / step(q4) < 1.237 (`../README.md`, k-choice note).

The renders are `dsrelk5-c128-1p1d[-ev]-{xdr,spx}` and `dsrelk5-c160-1p1d-llh[-ev]-{xdr,spx}`. Each is the point's
`dsrel-*` render with draft length 5. `gen_dsrel.py --k-override` writes them (change 7 in its docstring) and
`check_dsrel.py --k-override` checks them. `configs/<point>/{agentx,gsm8k}` is the repo mirror of the chosen fabric
(C128 spx, C160 llh xdr). It is byte-identical to the Hecate-mode renders that passed `check_dsrel.py`, stamp
`2026-11-16T21:30:00-08:00`.

## What changes vs the dsrel render (and nothing else)

1. **`num_speculative_tokens` 3 -> 5** in the prefill and decode speculative-configs. The prefill role keeps one draft
   length with the decode role.
2. **AL.** AgentX renders use `synthetic_acceptance_length` 3.81, the release golden AL of k5, with per-position rates
   [1, 1, 0.81, 0, 0]. `golden-acceptance.json` and `AL_FIX.txt` record it. The gsm8k twins keep real block rejection
   and no AL.
3. **Decode capture keys.** `max-cudagraph-capture-size` goes 640 -> 960 (160 x q6), and `cudagraph_capture_sizes`
   goes from 104 to 123 sizes (`derive_capture_sizes`):
   - Every frozen size is a verify row 4B, a drafter row 3B (B <= max-num-seqs), or both. That gives 79 verify batches
     and 56 drafter batches.
   - The k5 list holds 6B for the verify batches and 5B for the drafter batches.
   - Re-deriving at k = 3 returns the frozen list unchanged (checked).
   - Model Runner V2 captures one FULL graph per size, rounded up to whole requests. The verify graphs go from 82 to
     96 batches and the drafter graphs from 80 to 90. Every frozen verify or drafter batch keeps a graph.
   - Batches with a frozen graph that lose their own at k5: verify B41, and 17 drafter by-products of the k3 rounding
     (B11, B14, B19, B43, B59, B67, B78, B86, B91, B102, B107, B118, B123, B134, B139, B150, B155). Each pads to its
     next graph.
   - At the operating batches (B20-B50) every batch pads to the same graph as at k3 or a closer one, except verify B41
     (-> B42) and drafter B43 (-> B44).
   - The largest verify and drafter batches (the capacity anchors) stay B160 / B160.
4. **Names.** `dsrelk5-...` everywhere the frozen render name appears. `DSREL.txt` records the derivation.

Nothing else changes with k, and `k_requirements` checks why:
- max-num-batched-tokens (prefill 16384, decode 8192) holds the DSpark draft pass of max-num-seqs x 5 query rows, and on
  decode also 160 x 6 = 960 verify rows.
- The prefill capture list does not depend on k: the frozen k7 and k3 points share it.

Exact config diff vs the frozen render, C160 xdr AgentX (`diff ../../configs/C160-1P1D-llh/agentx/render/config.yaml
configs/C160-1P1D-llh/agentx/render/config.yaml`):
```
1c1      name: ...frz-c160-1p1d-llh-xdr -> ...dsrelk5-c160-1p1d-llh-xdr
209c209 (prefill), 254c254 (decode)
<       speculative-config: '{"method": "dspark", "model": "/raid/common-models/kimi-k3.1-pro-dspark-preview", "num_speculative_tokens": 3, "attention_backend": "TOKENSPEED_MLA", ..., "synthetic_acceptance_length": 2.88}'
>       speculative-config: '{"method": "dspark", "model": "/raid/common-models/kimi-k3.1-pro-dspark", "num_speculative_tokens": 5, "attention_backend": "FLASH_ATTN", ..., "synthetic_acceptance_length": 3.81}'
259,260c259,260 (decode)
<       max-cudagraph-capture-size: 640
<       compilation-config: '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6,8,15,16,...,624,640]}'
>       max-cudagraph-capture-size: 960
>       compilation-config: '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[10,12,24,25,...,936,960]}'
265c265  the draft extra_mount (preview -> e4e6e626), as in every dsrel render
```

## Generate and check (Hecate login node)

Use the same tree and weights as the dsrel sweep renders (`/Users/weizha/k31vr-dsrel/DSREL_NOTES.md`, sweep plan step 2):
```
cd $W/dsrel/cfg   # with this gen_dsrel.py / check_dsrel.py in tools/
python3 tools/gen_dsrel.py --k-override C128-1P1D=5,C160-1P1D-llh=5 \
    --tree $W/dsrel/trees/<tag> \
    --target-weights /lustre/fsw/coreai_comparch_inferencex/weizha/models/kimi-k3.1-pro-release-d9e99b07,$W/e2e/model-view/kimi-k3.1-pro-k3type \
    [--replace]
/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr/srtctl-venv/bin/python tools/check_dsrel.py \
    --k-override C128-1P1D=5,C160-1P1D-llh=5 --tree $W/dsrel/trees/<tag> --target-weights <same>
```
- `--k-override` defaults `--points` to the overridden points and the prefix to `dsrelk5`.
- It refuses a point without an override, the prefix `dsrel`, K outside 1..7, and any decode that is not TP16 = DCP16
  (C512's sequence-parallel decode accepts k3 only).
- The renders join the sweep as their own items, `agx|6|dsrelk5-c128-1p1d` / `agx|6|dsrelk5-c160-1p1d-llh` and the
  `ev|6|...` twins. The `dsrel-*` k3 renders stay as they are.
- Mirror: `python3 gen_dsrel.py --mirror .. --k-override C128-1P1D=5,C160-1P1D-llh=5 --stamp <iso>` writes
  `release_draft/k5/configs`.

## Validation (checks/)

- `gen-k5-20261116T2130.out`: Hecate mode, both fabrics and both kinds (8 renders), from a copy of the 8 frozen renders.
  Each render's parsed diff vs its source is 5 keys:
  - the 2 speculative-configs (JSON: model, attention_backend, synthetic AL, num_speculative_tokens);
  - the draft mount;
  - decode `compilation-config` (JSON: `cudagraph_capture_sizes` only);
  - decode `max-cudagraph-capture-size`.
- `check-k5-20261116T2130.out`: `check_dsrel.py` on those 8 renders gives `CHECK_DSREL PASS: 0 failure(s), 2 warning(s)`
  over 475 checks. It includes srtctl `load_config` and RuntimeContext mounts, and `srtctl render` with preflight.
  The 2 warnings say the frozen tree lacks the release-draft code.
- `vllm_check-k5.out`: `vllm_check.py` on the 8 renders gives `VLLM_CHECK PASS`.
  - vLLM's SpeculativeConfig parses k=5 on both roles and resolves AL 3.81 to the rates [1, 1, 0.81, 0, 0].
  - FLASH_ATTN accepts the draft layers.
  - The release checkpoint loads as the GQA DSpark draft.
- K = k identity: `--k-override ...=3` gives byte-for-byte the `dsrel-*` renders except names and the DSREL.txt k lines.
  The base `dsrel-*` output of the new generator equals the previous generator's except its md5 line.
  `--k-override C64=5` (k7 -> k5) also derives cleanly.
- `--k-override` combines with `--target-weights` and `--tree`, giving 8 parsed keys.

## Coverage at k5 (q6), C128 / C160 decode role (TP16 = DCP16, VR200; strict)

- `coverage_k5.py` writes `checks/coverage_k5.out`: the target step plan and the release drafter's GQA tables per
  captured shape, k3 vs k5.
- `tests/models/kimi_k3/test_specialized_decode_coverage_preflight.py` gains the `c160k5-prod` (production list) and
  `c160k5-any` (every q6 row up to 1024) deployments and the q6 DSA wide grids. All 53 pass.
- `tests/models/kimi_k3/test_dspark_gqa_fastpath.py` gains `C128-1P1D-k5` / `C160-1P1D-llh-k5` in POINTS. All 29 pass
  (`checks/tests-k5.out`: 82 passed).
- Both suites ran on CPU through `gpurun.sh` on overlay `k5audit` (this branch).

Target, 96 captured verify rows M12-M960:
- **Nothing refuses.** No row runs the whole-step stock forward. Every wide-row family (KDA, DSA, o-projection, MoE)
  admits M66-M960. The DSA MlaQuery and DcpValue grids are resident at every q6 row up to 1024.
- **Every row is GENERIC.** TUNED exists only at q1 / q8 prepared rows, which is the same at q4.
- KDA verify (FP32 state) is untuned at q6:
  - FP32_VERIFY_WIDE, FP32_FEED, FP32_SPLIT_Q and DEVICE_SPEC_SCHEDULE_Q have q4 / q8 rows only.
  - So q6 takes VERIFY_WIDE's BF16 crossovers: KdaSplitStock at B2-B15, KdaVerifyLoopStock from B16.
  - q4 has tuned picks: LoopStock B17-35, KdaSplitStock 1.8.3.4 at B36-42, KdaVerifyFeed from B44.
  - KdaVerifyFeed caps at q4 (`MAX_TOKENS`).
- The KDA heads do not serve q6. KdaHeadMulti asserts `tokens in (4, 8)` (`head_multi.py:131`), and HEAD_MULTI /
  HEAD_MULTI_FP32 have no q6 row, so q6 keeps the in-projection -> KdaSplit pair.
  - `head_constructs` turns the assert into "not supported", so this tree does not refuse at q6.
  - A BF16-state q6 head would need a q6 MULTI_TILE row and the even-token loop extended to 6.
- AttnRes runs the stock op from 384 rows and the embedding is stock above 256 rows. Both are as at q4: measured
  choices, not refusals.

Release drafter, 90 drafter batches B2-B160 (query rows B*5 = 10..800, context rows B*6 = 12..960):
- **No strict-table gap.** GQA_FUSED_MARKOV covers every batch: gated at B2, gated+staged at B3-B8, MarkovWide from B10,
  with a plan at every batch. GQA_FAST_CONTEXT covers every context row. GQA_FAST_AR gives two-shot sites at B2-B12
  and ArNormRows from B13.
- The preflight's kernel-level holds pass at every k5 row: PreparedContext gathers (persistent above 212 rows),
  PreparedDraftAr rows and ArNormRows residency, MarkovWide plans.
- LM head: evict-first at B5-B6, the stock cuBLAS shard GEMM elsewhere. These are GQA_WIDE_LM_HEAD's measured choices.
- Split-K projections run up to 64 rows (B <= 12), cuBLAS above.
- **The attention refuses at k5: a real gap, patch `gqa_window_k5.patch`.** The drafter's attention is track K's fused
  GQA window kernel (`cutedsl/draft/gqa_attention.py`, not in this tree; read from overlay `dsrel/ov/attn3`, md5
  a87bdcfe).
  - Its launch-shape table `GQA_WINDOW` is keyed by (KV heads per rank, draft length) and has k3 / k7 rows only.
  - `install_gqa_attention` therefore finds no shape for any k5 batch (`checks/gqa_window_k5.out`) and refuses on the
    strict TP16 = DCP16 decode: "no GQA_WINDOW launch shape for rows [...]".
  - The patch adds `(1, 5)` with `(1, 3)`'s shapes. k5's 30 query rows per (request, KV head) fill the same two 16-row
    M tiles as k3's 18, so every shape builds alike and keeps k3's CTA count per batch.
  - With the patch, all 90 k5 batches have a shape and every shape builds at 1024- and 4096-token pages
    (`check_gqa_window_k5.py`: GQA_WINDOW_K5 PASS).
  - On 1 VR200 GPU, track K's `test_window_matches_stock` at TP16 q5 passes for B1-B128 in both draft-cache layouts
    (1024-token rows, 128-token slots): cache bytes equal stock's, rows within BF16 noise (`checks/gqa_attn_k5_gpu.out`).
  - NQ-TUNE: a q5 `--shape` sweep (`bench_gqa_attention_int.py`) at B16-B48.
  - The context K/V insert (`gqa_context_kv`) does not depend on the draft length.
  - No `runtime/draft.py` change is needed for k5.

GPU (16 ranks): see Measurement below. JIT warmup: the ksweep's k6 (q7) release-drafter run failed with
`JIT_WARMUP_GAP Triton _causal_conv1d_update_kernel (seqlen=7, ...)` (`dsrel/results/ksweep-6a14e2f/c160k6-relfa.log`).
The q6 runs below show whether q6 has one.

## Measurement

### Status (2026-11-17 00:35 PST)

The release-drafter k5 vs k3 A/B is **not measured yet**.

- **Attempt 1 was lost.** `08-nq4-dsrel-k5ab-c160-m60/m90.sh` ran at 23:38 on backfill hold 1119830 (int1 overlay).
  - The slow startup, from "Loading model from scratch" to the first "MoE backend" line, is seconds on the ksweep's
    warm nodes. On these cold nodes it took 11-22 min, plus CuTe cache loads.
  - The k5 relfa run had spent 38 min in that startup without timing any row when its step got SIGTERM at 00:16:40.
    The job was PREEMPTED exactly 10 min later at 00:26:40, which fits a preemption grace period, and it took the k3
    relfa run with it (`checks/bench-attempt1.out`).
  - The task's entry stays in `queue/running/` (`...m90.sh.1119830`), so the runner will not retry it. Its logs are in
    `dsrel/results/k5ab-int1`.
- **Attempt 2 is queued:** `08-nq4-dsrel-k5ab2-c160-m90.sh`, with the same runs, writing to `dsrel/results/k5ab2-int1`.
  - The next hold that can take it is r2hd8 1118813 (begin time 01:59), after the 06 / 07 tasks ahead of it.
  - To read it: `python3 bench/k5ab_table.py $W/dsrel/results/k5ab2-int1` gives the table and the projection per B.
- **Also queued:** `08-nq4-dsrel-k5strict-c160-m120.sh` (results `dsrel/results/k5strict`). It runs the k5 production
  list with full capture on overlay `k5attn`: the strict integrated drafter, track K's attention and
  `gqa_window_k5.patch`. Target-only runs first.

### Projection from the data in hand (same contexts, same c160 config)

Inputs:
- **Target only** at k5 vs k3: `dsrel/results/ksweep-6a14e2f`, one task on the same 4 nodes (job 1117399), base overlay.
- **Release drafter at k3**: int1's fast path on c160 (`dsrel/results/int1`). Its drafter time, int1 relfa minus the
  ksweep target, is 477 / 566 / 564 us at B16 / B28 / B38. These come from different allocations.
- **Release drafter at k5**: not yet measured. The table brackets it:
  - d5 = d3: int1's drafter is latency-bound, flat from k3 at c160 (490-565 us) to k7 at c64 (447-548 us).
  - d5 = 5/3 d3: every drafter cost scales with its 5/3 more query rows.

Projected change of interactivity, dx = (AL5 / AL3 = 1.237) / (T(q6) / T(q4)) - 1, at a fixed batch:

| B (point) | target q6/q4 | dx target only | release q6/q4, d5 = d3 | dx | release q6/q4, d5 = 5/3 d3 | dx |
|---|---|---|---|---|---|---|
| 16 | 1.249 | -1.0% | 1.239 | -0.1% | 1.267 | -2.4% |
| 28 (C128) | 1.120 | +10.4% | 1.116 | **+10.9%** | 1.142 | +8.4% |
| 38 (C160) | 1.163 | +6.3% | 1.157 | **+6.9%** | 1.181 | +4.7% |
| 48 | 1.174 | +5.4% | - | - | - | - |

So k5 projects to **+8..11% interactivity at C128 and +5..7% at C160**, both above break-even. At a fixed concurrency
tok/s/GPU moves ~5x less than interactivity (`../README.md`). Around B16 and below, q6 breaks even or loses.
`08-nq4-dsrel-k5ab2` replaces this projection with the same-node release-drafter ratio.

## Files

- `configs/<point>/{agentx,gsm8k}`: the k5 renders, repo mirror of the chosen fabric.
- `coverage_k5.py`: the CPU coverage audit, k3 vs k5 (target step plan and drafter GQA tables).
- `gqa_window_k5.patch`: the `(1, 5)` GQA_WINDOW row for track K's `cutedsl/draft/gqa_attention.py`, against overlay
  attn3's file (md5 a87bdcfe -> c3494526).
- `check_gqa_window_k5.py`: the patch's CPU check (launch shapes and GqaWindow builds at k5 / k3 / k7).
- `bench/`:
  - the two hold-queue tasks: the k5 / k3 A/B, and the k5 production-list full capture;
  - `mk_bench_cfg.py`, which writes the latter's bench-only config;
  - `k5ab_table.py`, the A/B table and projection.
- `checks/`: generation, render checks, vllm_check, tests, coverage audit, GQA_WINDOW checks, bench results.
