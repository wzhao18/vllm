# nq: stock vs specialized decode step A/B (TP16 = DCP16, Rubin / VR200)

The whole Kimi K3.1 decode step, stock (`decode_backend` auto) against `latency_specialized`. It uses the
no-quant serving precision: fp8 MLA KV, FP32 KDA recurrent state, mxfp4 indexer keys, BF16 linears and mxfp8 MoE
activations. It runs with the old drafter (kimi-k3-dspark: 64 heads, Markov rank 256) or the new one
(kimi-k3.1-pro-dspark-preview: 96 heads, Markov rank 512).

The harness is new and kept separate from `wtp1/`, which stays untouched. It fixes the bench blockers from the audit:
- **ReplaySSM refusal:** the flag is not passed.
- **FULL-graph refusal under `vllm bench forward`:** handled in the `nq_step.py` wrapper.
- **Empty drafter batches:** the wrapper runs the real `draft_batches` on the production capture list.
- **Hard-coded bf16 state and old drafter:** everything comes from the production config.
- **Unrepresentative layout, attention and kernel config:** replaced by the `cfg2bench.py` output.

## What is timed

`[forward] ... median` is one CUDA graph per case, containing:
- the target forward over all 93 layers, with dummy weights;
- the drafter's K-step propose, with every draft accepted.

Each replay is the maximum over the 16 ranks, and the reported value is the median of iters x samples.

Not included in either arm:
- target logits and LM head;
- rejection and sampled verify, including the specialized `runtime/sampled_verify.py`;
- CPU step overhead.

`--exclude-drafter` runs give the target/drafter split. The ls arm's "graph vs eager rel rms" is around 1.4 with dummy
weights. It is not a correctness signal; correctness needs real-weight runs.

## Files

| file | role |
|---|---|
| `cfg2bench.py` | Reads `jh/noquant/cfg/nq-<point>-xdr.yaml`, decode role plus global env. Writes the exact bench engine args and env as sourceable sh: TP16 DCP16, fp8 KV, float32 SSM state, mxfp4 indexer, attention and kernel flags, block size, max-num-seqs, capture sizes, speculative config with the drafter override, and `decode_backend` per arm. Each production arg is kept, dropped with a reason, or rewritten; an unknown arg is an error. `--snapshot` writes JSON copies so batch hosts need no PyYAML; a stale snapshot is refused. |
| `nq_step.py` | Wrapper around `vllm bench forward`. Handles the rendezvous and checks the code-under-test overlay (`--overlay`: vllm and kimi_k3 must resolve inside it; logs its revision and shipped files). Runs the real `specialized_decode_config` / `draft_batches` under the production compilation fields (FULL_AND_PIECEWISE, capture sizes, max capture). Fills the indexer with MX-aware random keys. |
| `nq_shapes.py` | Cases for a list of verify rows M = B*q, in ascending rows. Context options: AgentX vectors (`q8_C1_C8` for c1/c8, `q8_C64`, `q4_C48_C160_C256`), `uNk`, `mix`. |
| `nq_inner.sh` | One rank inside the e2e image. Sets PYTHONPATH to the overlay tree (or `env/vllm-jh-242fb49cd4` without one), FlashInfer 0.7.3.dev20261031 and the e2e pydeps, like `mk_nq.py`'s decode role but without dynamo and pinhook. Also sets the cache layout and the per-overlay extension build dir (`$NQ_BUILD`). |
| `nq_run1.sh` | One arm process: `srun --overlap --jobid J -N4 -w <4 hosts> --ntasks-per-node=4` in image `vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh`. That is the image the nq e2e configs' container key resolves to (`jh/srtslurm-jh.yaml`, `e2e/k31r/srtslurm.yaml`); the DEP configs use `internal-vr-mum-92ccd9647f` instead, which `NQ_IMG` can select. |
| `nq_ab.sh` | One A/B point: arms in a given order on the same 4 nodes, then the summary. Rows at or below the point's production limit (`min(max-num-seqs * q, capture cap)`) and rows above it run as separate processes (`<tag>` and `<tag>-hi`), so the rows inside the limit always run under the production `max-num-seqs` and capture cap. After an arm dies at startup, its later runs with the same process tag are skipped (same point, drafter, ctx, extra args and part); other lines still run. Also skips any run that cannot finish before the block deadline (listed in `SKIPPED`). |
| `nq_summary.py` | Per row: stock and ls medians, ls/stock, the ls coverage line (`Kimi K3 decode coverage M..`), and the drafter fast paths installed for that B (markov, dcp, ctx, lm). Flags WORSE, MISSING(arm) (a requested row an arm did not measure), UNCOVERED (no ls coverage line), XCAP (above the production limit), whole-step stock and family stock seams. Also writes CSV. |
| `nq_block.sh` | 12-node block driver, in this order: prechecks (overlay, 12 nodes, every plan line resolves); the lock against a requeued copy (a failed precheck takes no lock); idle GPUs (if processes are still on the GPUs after `NQ_BUSY_WAIT`, 600 s, it writes `BUSY` to `SUMMARY.txt`, releases the lock and exits without cancelling the block); container creation; 3 concurrent 4-node slots; summary; `scancel`. Default end margin is 10 min. |
| `nq_gputest.sh`, `nq_gputest_inner.sh` | The GPU correctness stage (plan line `<slot> gputest`). Runs on the slot's first node (4 GPUs) and second node (2 GPUs) in the overlay env, inside the e2e image. It uses the merge-gate test harness: `env/k31vr_env.sh`, `--noconftest -p k31m_fixtures`, DS conv layout. On 4 GPUs it runs pytest per file on `tests/models/kimi_k3/test_specialized_decode_{*gpu*,draft_nq,wide_nq,plan,dsa*}.py`, then the files that are torchrun programs under torchrun. On 2 GPUs it runs the specialized-decode files in `tests/distributed`, matched by name or by being changed vs 242fb49cd4. Writes junit and a log per file. |
| `nq_junit.py` | `GPUTEST.txt`: per file passed/failed/errors/skipped, failing tests, skip reasons; flags FAIL, TIMEOUT, NO-TESTS, ALL-SKIPPED, NO-JUNIT, NOT RUN, UNCOVERED (no test files). |
| `tasks/4{0,1,2}-mx12-nq-blk{1,2,3}.sh` | Queue task templates for `holds/runner.sbatch POOL=mx12`, one per block (`__OVTAG__` is filled in by `stage_overlay.sh`). |
| `plans/block{1,2,3}.plan` | The measurement matrix, in slot order. |
| `push_tools.sh`, `stage_overlay.sh` | Copy the tools to `$NB/tools` and refresh `$NB/cfg`. Stage a checkout as `$NB/ov/<tag>` and render the tasks into `$NB/tasks` (see below). |
| `test_nq_harness.py` | Pure-Python checks (no torch) of cfg2bench, the wrapper patches, overlay resolution, shapes and summary. |

Here `$W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr` and `$NB=$W/nqbench`.

## Use

```
bash push_tools.sh                                   # tools -> $NB/tools, nq configs -> $NB/cfg/*.json
bash stage_overlay.sh <checkout> <ovtag>             # code under test -> $NB/ov/<ovtag>; tasks -> $NB/tasks/*-<ovtag>.sh
# staging is the user's call (nothing here touches queue/):
cp $NB/tasks/40-mx12-nq-blk1-<ovtag>.sh $W/queue/pending/
# results: $NB/results/blk1-<ovtag>/{SUMMARY.txt, gputest/ (GPUTEST.txt, junit, logs), A.csv, B.csv, C.csv,
#          block.log, A/ B/ C/ (per-run logs)}
```

To run one point by hand on a held 4-node allocation, after `push_tools.sh`:

```
NQ_OV=$NB/ov/<ovtag> bash $NB/tools/nq_ab.sh <jobid> h1,h2,h3,h4 $NB/results/manual c256 new 8,64,256,1024 ls,stock agentx
python3 $NB/tools/cfg2bench.py c256 --arm ls --drafter new --rows 8,1024   # inspect exactly what a run gets
```

Rows not in the production capture list are added, together with their drafter rows B*K. Each measured shape is
then a captured FULL graph: unpadded verify, B in `draft_batches()`. The production limit is
`min(max-num-seqs * q, capture cap)`. Rows above it run in their own process (`<tag>-hi`), which raises
max-cudagraph-capture-size and max-num-seqs to the largest measured row (a `NOT PRODUCTION` note; the summary marks
those rows XCAP). All of this is printed as comments in `<run>.cfg.sh` and at the top of the log; every dropped
production arg and env key is listed with its reason. The removed env switches `VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS` and
`VLLM_K31_DSA_PARTIALS` (now an automatic per-shape policy) are dropped if a config still carries them.

**Overlay** (`stage_overlay.sh <checkout> <tag>`): `$NB/ov/<tag>` is a whole vllm tree. It starts as
`env/vllm-jh-242fb49cd4`, hard-linked, so the base is never written. Every file that differs from 242fb49cd4 under
`vllm/` is shipped over it: modified, added and untracked files are copied, and deleted files are removed. The tree
also gets the checkout's `tests/` and `pyproject.toml`. `SOURCE_REV` records the revision and the list:
- `M|A|D <path>` for files under `vllm/`;
- `T <path>` for changed tests;
- `NOT-SHIPPED <path>` for changes outside `vllm/` and `tests/` that need a build (also printed as a warning).

The shipped files are checked by md5 against the checkout. An existing tag is refused; the tree is staged under
`.<tag>.partial` and renamed when complete.

## Measurement matrix (3 blocks: 12 nodes x 3:30 h, 3 slots of 4 nodes)

| ID | Point (decode role) | q | Rows (M = B*q) |
|---|---|---|---|
| q4-hi | `c256` (c160 has the identical decode role): wide DSA 512, mns 128, cap 512, DCPX 1 | 4 | 8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024 |
| q8-hi | `c64` (c48-1p1d is identical): wide DSA 256, mns 32, cap 256, DCPX 1 | 8 | 8, 16, 32, 64, 128, 256, 512, 1024 |
| q8-lo | `c1`: DSA core rows 1-16, DCPX 0 | 8 | 8, 16, 32, 64, 128, 256 |
| q8-c8 | `c8`: DSA core and DCPX 1 | 8 | 8, 16, 32, 64, 128, 256 |
| q4-mid | `c48-2p1d`: wide DSA 256, mns 64, cap 256 | 4 | 8, 16, 32, 64, 128, 192, 256 |

Rows above a point's production cap (q4-hi above 512, q8-hi above 256) measure the arbitrary-batch goal.

Budget per slot: about 25 min a process and a 10 min end margin, so at most 7 processes. Slots A and B use 6, so a slow
first build (extension and CuTe compile under one lock) does not cut an A/B pair. A line that straddles the production
limit costs two processes per arm.

**Block 1** (`plans/block1.plan`), A/B order ls then stock:
- **Slot A, q4-hi:** new drafter at 8-1024 (4 processes: the 512-and-below part and the 768/1024 part); old drafter at
  8-512 (2).
- **Slot B, q8-hi:** new drafter at 8-1024 (4); old drafter at 8-256 (2).
- **Slot C:** first the GPU correctness stage (at most 75 min, on two of its nodes), then q8-lo, q4-mid and q8-c8 with
  the new drafter (about 15 min a process; q8-c8 is the first to be skipped if the tests run long).

**Block 2** (`plans/block2.plan`):
- **Slots A and B:** the q4-hi and q8-hi new-drafter lines again in the order stock then ls. With block 1 this gives
  ABBA over two node sets. Then `--profile-dir` runs at 3 rows within the production limit (edit them to block 1's
  worst ls/stock rows before staging).
- **Slot C:** q8-lo, q4-mid and q8-c8 new in the order stock then ls, then q8-lo old.

**Block 3** (`plans/block3.plan`):
- **Slots A and B:** `--exclude-drafter` (the target/drafter split) at rows up to 1024, then u192k.
- **Slot C:** `mix` contexts at q4-hi and q8-hi within the limit, then q8-c8 old and q4-mid old.

The old drafter is no longer measured above the production limit, since the previous-performance comparison only
needs production rows. The priority order inside each slot is the line order. When time runs short, the later lines
are skipped and listed under SKIPPED in `SUMMARY.txt`.

## Previous-audit failures that block 1 re-checks

The first audit expected the following failures. The current tree addresses each of them, but none of those fixes
has run on a GPU yet. Block 1 shows whether they hold: the summary prints the error of each run that dies, and
flags MISSING and UNCOVERED rows.
- **FP32 KDA state at q4 rows 144-1024** (c256, c48-2p1d): `KdaVerifyStream3` asserted a bf16 state at prepare, so
  the ls process died at startup.
- **New drafter (Markov rank 512):** `PreparedMarkov` raised `unsupported Markov head shapes`. Uncovered heads must
  now keep the stock Markov loop, with a warning in the log.
- **Causal drafter attention:** the drafter's Lamport DCP attention now logs its mask (`(causal)` or
  `(non-causal)`); `nq_summary.py` matches either.
- **Rows above 512:** `WIDE_ROWS` used to stop at 512. A row that still runs whole-step stock is listed under
  "whole-step stock in ls".
- **WideDsa row cap:** `VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS` is gone (now an automatic policy). Remaining
  `dsa=stock(...)` seams show up per row.

## Previous performance (old precision, for "keep the previous performance")

Every number below is from dummy weights, the old drafter and **bf16 KDA state**. None of them uses fp32 state, fp8 KV
with mxfp4 index keys, block 1024 with IL1 at q8, or the production env. Searching `$W` for step-bench logs newer than
Nov 2 12:00 finds nothing, and no TP16 step log with `float32` state exists anywhere.

The numbers are the median step in us, AgentX contexts unless noted.

| Source | Precision and layout | Drafter | Rows: us |
|---|---|---|---|
| `$W/merge/results/ab9/ab_B1.log` (bl-a60e0e0f6, train 9 base, Nov 1) | fp8 KV, trtllm sparse, fp8 index, blk/IL 896 | fast path | q8 M8 5020, M40 9746, M48 10436, M56 10957, M64 11320 |
| `$W/r9x/results/s16/ab2_B1_q8.log` (r9x-c1, Nov 1) | same | fast path | q8 M8 4988, M40 9610, M48 10244, M56 10750, M64 11071 |
| `$W/oracle/results/o_q8s.log`, `o_q8w.log` (bl-7b490dfa6, Nov 1) | same | fast path | q8 M8 5042, M56 11395, M64 11320, M512 28121, M768 38617 |
| `$W/merge/results/n4-bl-7c601cbde/c256_B.log` (train 10 head, Nov 2) / `c256_A.log` (base bl-a60e0e0f6) | nvfp4_ds_mla KV, mxfp4 index, prims_ts, blk/IL 1024, packed groups 2 | fast path | q4 M256 18975 / 19001, M344 22038 / 22017 |
| `$W/r9x/results/s16/ab_B1_q4.log` | same | fast path | q4 M344 21716 |
| `$W/oracle/results/o_q4c256.log` | same | fast path | q4 M256 19467, M344 22713 |
| `$W/wide/results/B1_q4.log` (Oct 30, wide arm) | fp8, trtllm, IL896 | **stock** | q4 M8 8335, M32 12555, M64 15845, M72 18325, M128 22335, M256 33075, M384 38617, M512 44471, M768 57810 |
| `$W/w3/results/AB_A1_q4.log` (Oct 31) | fp8, trtllm, IL896 | **stock** | q4 M8 8197, M64 15462, M128 21968, M256 32433, M512 43731, M768 57314 |
| `$W/d4x4/results/n4/tp16ls_bf16_fp8.log` (Oct 31) | fp8, IL1 | **stock** | q4 M64 13402, M256 22744, M512 33186, M768 44048 |
| `$W/w4/results/tp16_full_q4/bf16.log`: pure stock process, `decode_backend` auto, ReplaySSM, IL1, uniform 200K | fp8, trtllm, bf16 state | stock | q4 M64 26135, M256 44260, M512 63237 |

Two kinds of stock number appear in these logs, and neither is a fair stock arm:
- `$W/wide/results/A1_q*.log` and the w3 M96/M192 cases are the in-process stock fallback of an ls process. It degraded
  to 49.6-102 ms in later builds.
- w4 is a separate stock process, but it uses ReplaySSM.

The only fair comparison is a separate `decode_backend` auto process with identical flags, which is what this harness
runs.

Old-to-new caveats:
- q8 moves from IL896 and blk 896 to IL1 and blk 1024, the production layout.
- q4 C256 moves from nvfp4 KV, prims_ts and packed groups 2 to fp8 KV, the default sparse kernel (`auto`) and packed
  groups 3.
- KDA moves from a bf16 to an fp32 state. That alone loses the KdaHead fused paths at q8 small B
  (`runtime/providers.py:434-437`).
