> **Final (2026-11-18): ACCEPTED and merged.** Tag wzhao/kimi-k3-megamoe-sm107-ep32-20261118 = merge commit a368c5f66 on wzhao18 wzhao/kimi-k3-megamoe-sm107-rebase (lane wzhao/sm107-r3-ep 89fafd77e pushed). Supported EP: 1, 4, 8, 16, 28, 32.

# EP28 / EP32 GPU validation of the SM107 MegaMoE: summary (2026-11-18)

- Lane: wzhao/sm107-r3-ep at head 89fafd77e (one commit on 6e438852b). It is not pushed or merged, and this phase made no code change.
- Tree: opt/trees/r3-ep = trees/final (6e438852b) plus the 4 files of git diff 6e438852b..89fafd77e.
  - The md5 of all 4 files matches the local worktree.
  - diff -rq final r3-ep lists exactly those 4 files.
- Hold: 1137209 (8 nodes hecate[0091-0093,0095-0098,0103], batch-xdr, one --segment=8 allocation). Runs took place 03:21-04:54 PST.
- Container and environment are the same as the final A/B: inner_env_opt.sh and the image from frozen/MANIFEST.txt.

## Method

### Harness
ep32/ep_run.sh is a copy of final_ab/ab_fin.sh. It uses the same bench chain (fi_l1f5.py -> fi_l1f.py -> fi_opt.py), the same statistic (50 samples x 10 replays x CPG 10, slowest rank per sample) and writes rows with opt_rows.py. It adds:
- HOST_SKIP and a lock per first host, so two EP16 halves can run on one hold;
- an NVS override of NVSHMEM_SYMMETRIC_SIZE;
- the driver ep32/fi_bench2x.py, which is fi_bench2.py plus a global-batch hook that is active only with MMX_CROSS=1.

### Queue
- ep32/qrun2.sh runs these lists: p1.list, halfA.list, halfB.list, p3.list and p4.list.
- qrun.sh (v1) had a bug: srun read the list from stdin, so only 1 item per list ran.
  - The items that did run are still valid.
  - qrun2 redirects stdin from /dev/null.

### Modes
- bench: capacity = T, knob cache.
- vLLM: capacity 8192, knobs=None, hint = T, prepare([T]), CUDA graph.
- prefill: bench mode with NVSHMEM_SYMMETRIC_SIZE 64G (128G for T 16384).
- Routing: random, seed 0.

### Repeats
- EP32: 2 repeats at every point, except T16384 (1 repeat).
- EP16 and EP28: 1 repeat.
- All EP16 runs used hosts 1-4 or 5-8 of the same hold.

### Stage split
Not available. kernels_us and stamps are null without a stamps build, so only the total is reported.

## Correctness

### Container tests
- tests/moe_ep/test_sm107_ep_sizes.py: 75/75 passed, including the 4 router-geometry and pool-bound tests that were skipped on CPU (ep32/t_epsizes.out).
- cutlass-dsl is 4.8.0, so the >=4.7 peer-table path is the one in use.

### (C) EP16 regression
- Setup: r3-ep vs trees/final, bench mode, T 1/64/512, NVFP4 and MX.
- Result: bitwise equal at 6/6 points on all 16 ranks, max_abs 0.
- Timing: within 0.3%.

### (D) Cross-EP
- Setup:
  - one global batch of G tokens;
  - hidden = randn with seed 7001+G;
  - ids = top-16 of randn(G,896) and weights = rand, both with seed 9001+G;
  - rank r takes rows [rT,(r+1)T), and the rank outputs are concatenated in rank order (ep32/crosscmp.py, output in ep32/xcmp.out).
- Result: BITWISE EQUAL at 16/16 points, diff_elems 0, both precisions:
  - EP32 T 1/4/64/256/1024 vs EP16 T 2/8/128/512/2048;
  - EP28 T 4/64/256 vs EP16 T 7/112/448.
- Variant classes covered:
  - fast decode;
  - the N256-tile switch at EP32 T256;
  - the cap-1024 class.
- Paths exercised that had never run on GPU:
  - the 2nd broadcast round;
  - all 32 token_in ring lanes;
  - barriers and peer pointers to PEs 16-31 across 7-8 nodes;
  - combine to 32 source ranks.
- The bf16-reference check passed in every run:
  - NVFP4 rel_l2 0.165-0.169 (threshold 0.21);
  - MX rel_l2 0.047-0.049 (threshold 0.06).
- Not covered:
  - an idle-rank case (the bench uses a uniform T on every rank);
  - the Kimi isolated backend (separate kernel tree) at EP4/28/32.

## Memory

### NVSHMEM heap
- The default 16G NVSHMEM_SYMMETRIC_SIZE was enough. These runs passed the check, with timing equal to the 64G/128G runs:
  - EP32 NVFP4 T8192: 1373.2 us;
  - EP32 MX T8192: 1775.2 us;
  - EP32 MX T16384: 4183.8 us.
- The code-lane estimate of about 27 GiB at EP32 MX T8192 does not bind on the NVSHMEM heap.
- No capacity or workspace failure occurred up to T16384 per rank at EP32.

### HBM
Peak HBM per rank was not measured. The MMMEM nvidia-smi samples are taken after the processes exit.

## Competitor
- B4 (vLLM-vendored megamoe_rubin) was not run at EP32.
- Its knobs/k31_vr.json has world sizes 4/8/16 only, and config.py raises KnobLookupError for any other size.
- Running it at EP32 needs new knob records, which is a tuning job.

## Per-EP median tables
Each cell is median us [p05-p95 over repeats] (repeat medians). The source is ep32/ep32sum.py on results.jsonl.

### bench (capacity = T, knob cache), nvfp4: median us [p05-p95 over repeats] (repeat medians)

| T | EP16 | EP28 | EP32 | EP32/EP16 |
|---|---|---|---|---|
| 1 | 51.4 [51.2-51.7] (51.4) | 50.5 [50.4-50.7] (50.5) | 50.4 [50.1-50.6] (50.3/50.4) | 0.979 |
| 4 | 76.1 [75.9-76.2] (76.1) | - | 59.3 [59.1-64.9] (59.2/59.3) | 0.779 |
| 16 | 85.4 [85.3-85.6] (85.4) | 62.5 [62.5-62.8] (62.5) | 59.4 [59.2-59.7] (59.5/59.3) | 0.696 |
| 32 | 85.8 [85.6-86.0] (85.8) | - | 60.2 [60.1-60.5] (60.3/60.1) | 0.702 |
| 64 | 85.8 [85.7-85.9] (85.8) | - | 62.8 [62.7-63.0] (62.8/62.8) | 0.732 |
| 128 | 88.1 [88.0-88.3] (88.1) | 71.2 [71.1-71.3] (71.2) | 69.1 [68.9-69.6] (68.9/69.3) | 0.784 |
| 256 | 99.6 [99.5-99.8] (99.6) | - | 94.5 [94.3-95.1] (94.6/94.5) | 0.948 |
| 512 | 133.4 [133.3-133.6] (133.4) | 140.8 [140.6-142.5] (140.8) | 143.3 [143.2-143.5] (143.3/143.3) | 1.074 |

### bench (capacity = T, knob cache), mxfp4_mxfp8: median us [p05-p95 over repeats] (repeat medians)

| T | EP16 | EP28 | EP32 | EP32/EP16 |
|---|---|---|---|---|
| 1 | 54.0 [53.7-54.3] (54.0) | 52.8 [52.7-53.0] (52.8) | 53.4 [53.3-53.6] (53.4/53.5) | 0.989 |
| 4 | 77.9 [77.7-78.0] (77.9) | - | 62.9 [62.8-63.2] (63.0/62.9) | 0.808 |
| 16 | 90.3 [90.1-90.4] (90.3) | 67.4 [67.3-67.5] (67.4) | 65.2 [65.1-65.4] (65.2/65.2) | 0.722 |
| 32 | 91.4 [91.2-91.6] (91.4) | - | 66.6 [66.5-66.9] (66.6/66.7) | 0.729 |
| 64 | 93.8 [93.7-94.0] (93.8) | - | 70.3 [70.2-70.7] (70.3/70.3) | 0.749 |
| 128 | 96.8 [96.6-96.9] (96.8) | 79.2 [79.0-79.3] (79.2) | 77.5 [77.1-78.1] (77.1/77.9) | 0.801 |
| 256 | 105.5 [105.3-105.6] (105.5) | - | 101.1 [100.9-101.2] (101.1/101.1) | 0.958 |
| 512 | 154.2 [146.6-156.8] (154.2) | 151.2 [150.3-153.1] (151.2) | 156.8 [156.5-157.2] (156.9/156.7) | 1.017 |

### vLLM (capacity 8192, knobs=None, prepare([T]), hint T), nvfp4: median us [p05-p95 over repeats] (repeat medians)

| T | EP16 | EP32 | EP32/EP16 |
|---|---|---|---|
| 1 | 51.5 [51.2-51.7] (51.5) | 50.4 [50.2-50.5] (50.4/50.3) | 0.978 |
| 4 | 76.0 [75.9-76.2] (76.0) | 59.1 [59.0-59.3] (59.1/59.1) | 0.778 |
| 16 | 85.1 [84.9-85.3] (85.1) | 59.2 [59.1-65.8] (59.2/59.2) | 0.696 |
| 32 | 84.8 [84.7-84.9] (84.8) | 61.1 [61.0-61.3] (61.1/61.1) | 0.720 |
| 64 | 87.1 [87.0-87.3] (87.1) | 61.8 [61.7-62.0] (61.8/61.8) | 0.710 |
| 128 | 89.9 [89.8-90.0] (89.9) | 71.5 [71.4-71.6] (71.5/71.5) | 0.795 |
| 256 | 101.2 [101.1-101.3] (101.2) | 98.2 [98.1-98.9] (98.2/98.2) | 0.970 |
| 512 | 134.3 [134.2-134.5] (134.3) | 143.7 [143.5-144.2] (143.7/143.6) | 1.070 |

### vLLM (capacity 8192, knobs=None, prepare([T]), hint T), mxfp4_mxfp8: median us [p05-p95 over repeats] (repeat medians)

| T | EP16 | EP32 | EP32/EP16 |
|---|---|---|---|
| 1 | 55.2 [54.8-55.4] (55.2) | 53.0 [52.8-53.2] (53.0/53.0) | 0.961 |
| 4 | 77.2 [77.0-77.3] (77.2) | 62.7 [62.6-62.9] (62.7/62.7) | 0.812 |
| 16 | 88.3 [88.3-88.4] (88.3) | 64.4 [64.3-64.6] (64.4/64.4) | 0.728 |
| 32 | 89.7 [89.6-89.8] (89.7) | 66.9 [66.8-67.2] (67.0/66.9) | 0.747 |
| 64 | 92.5 [92.4-92.7] (92.5) | 68.7 [68.6-68.9] (68.7/68.7) | 0.742 |
| 128 | 94.3 [94.2-94.4] (94.3) | 79.7 [79.6-80.0] (79.7/79.7) | 0.845 |
| 256 | 107.4 [107.3-112.9] (107.4) | 105.0 [104.9-105.2] (105.0/105.1) | 0.978 |
| 512 | 155.2 [148.0-156.8] (155.2) | 157.6 [157.4-158.2] (157.7/157.6) | 1.016 |

### prefill bench (capacity = T), nvfp4: median us [p05-p95 over repeats] (repeat medians)

| T | EP16 | EP28 | EP32 | EP32/EP16 |
|---|---|---|---|---|
| 4096 | 651.9 [651.4-652.6] (651.9) | - | 682.1 [681.4-692.0] (682.1/682.1) | 1.046 |
| 8192 | 1255.0 [1253.2-1257.6] (1255.0) | 1423.9 [1422.8-1425.1] (1423.9) | 1373.1 [1371.6-1374.6] (1373.1/1373.1) | 1.094 |
| 16384 | 2660.5 [2657.6-2662.9] (2660.5) | - | 3464.1 [3457.0-3472.0] (3464.1) | 1.302 |

### prefill bench (capacity = T), mxfp4_mxfp8: median us [p05-p95 over repeats] (repeat medians)

| T | EP16 | EP28 | EP32 | EP32/EP16 |
|---|---|---|---|---|
| 4096 | 860.3 [837.0-874.4] (860.3) | - | 859.3 [839.1-867.6] (857.6/861.0) | 0.999 |
| 8192 | 1717.8 [1681.8-1735.3] (1717.8) | 1790.3 [1783.2-1795.6] (1790.3) | 1775.6 [1757.9-1783.5] (1774.5/1776.6) | 1.034 |
| 16384 | 3450.8 [3436.4-3459.9] (3450.8) | - | 4185.0 [4170.8-4197.8] (4185.0) | 1.213 |

### other ep32-* rows (C regression, X cross-EP)


## Same global batch (strong scaling)
Cross-EP runs, 5 samples, bench mode, us.

| G global | NVFP4 EP16 | NVFP4 EP28 | NVFP4 EP32 | MX EP16 | MX EP28 | MX EP32 |
|---|---|---|---|---|---|---|
| 32 | 60.9 | - | 50.2 | 63.7 | - | 53.9 |
| 112 | 82.5 | 63.2 | - | 86.5 | 66.3 | - |
| 128 | 82.7 | - | 59.8 | 87.1 | - | 63.7 |
| 1792 | 87.9 | 65.6 | - | 95.4 | 73.0 | - |
| 2048 | 88.4 | - | 63.0 | 96.7 | - | 70.4 |
| 7168 | 125.3 | 93.8 | - | 135.8 | 101.9 | - |
| 8192 | 132.8 | - | 94.7 | 144.9 | - | 101.0 |
| 32768 | 355.0 | - | 219.8 | 455.5 | - | 250.9 |

## Perf cliffs
Per-rank time at the same T per rank, EP32 vs EP16.

1. Decode T 4-128 is faster at EP32 because each rank holds 28 experts instead of 56 and streams half the weights:
   - NVFP4: 0.70-0.78x;
   - MX: 0.72-0.85x;
   - T1: 0.96-0.99x.
2. The crossover is at T 256-512:
   - T256: 0.95-0.98x;
   - T512: 1.07x for NVFP4 (bench and vLLM) and 1.02x for MX.
   Per-rank GEMM work is equal and the weight bytes are halved, so the extra time is most likely the 32-peer dispatch/combine fan-out (2 broadcast rounds, 8 nodes). This is not measured.
3. Prefill cliff, EP32 vs EP16:
   - T4096: NVFP4 +4.6%, MX -0.1%;
   - T8192: NVFP4 +9.4%, MX +3.4%;
   - T16384: NVFP4 +30.2%, MX +21.3%.
   EP28 at T8192 is slower than EP32 (NVFP4 1423.9 vs 1373.1 us, MX 1790.3 vs 1775.6 us).
   Suspect: the push router (above 1024 tokens; EP x 896 SMEM table of 112 KiB at EP32; 32 x T x 16 routes) or landing/combine that scales with routes times fan-out. Not root-caused; it needs a stamps build. This is the main open perf item for EP32 prefill.
4. The die_split and tile policy scale as designed. N256 turns on at T256 at EP32 (vs T512 at EP16) because rows per expert double, and there is no cliff at that switch.

## Paths
- Directory opt/ep32/ holds:
  - scripts: ep_run.sh, qrun2.sh, fi_bench2x.py, crosshook.py, crosscmp.py, ep32sum.py;
  - run lists: *.list;
  - outputs: xcmp.out, t_epsizes.out, tables.md, holds.txt.
- Rows: opt/results.jsonl, 169 rows labelled ep32-*:
  - bn = bench;
  - vl = vLLM;
  - pf and pf16k = prefill;
  - pf16g = prefill at the 16G heap;
  - C = regression;
  - X = cross-EP.
- Logs: opt/logs/ep32-*. Dumps: opt/dumps/ep32-*.

## Holds
- 1137209: used, then cancelled at the end.
- 1137369 (submitted by this lane) and 1137210: cancelled without being used.

## Remaining gaps
- Stage split at EP32 (needs a stamps build). This is required to root-cause the T>=512 and prefill cliffs.
- Idle-rank case.
- Peak HBM.
- Kimi isolated backend smoke at EP4/28/32.
- EP4 on GPU.
- B4 at EP32.
- EP28: 1 repeat, subset of T, no vLLM mode.
