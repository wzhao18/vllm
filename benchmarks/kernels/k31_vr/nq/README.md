# nq: stock vs specialized decode step A/B (TP16 = DCP16 and C512 TP4 x DCP4 x DP4, Rubin / VR200)

The whole Kimi K3.1 decode step, stock (`decode_backend` auto) against `latency_specialized`. It uses the
no-quant serving precision: fp8 MLA KV, FP32 KDA recurrent state, mxfp4 indexer keys, BF16 linears and mxfp8 MoE
activations. It runs with the old drafter (kimi-k3-dspark: 64 heads, Markov rank 256) or the new one
(kimi-k3.1-pro-dspark-preview: 96 heads, Markov rank 512).

The harness is new and kept separate from `wtp1/`, which stays untouched. It fixes the bench blockers from the audit:
- **ReplaySSM refusal:** the flag is not passed.
- **FULL-graph refusal under `vllm bench forward`:** handled in the `nq_step.py` wrapper.
- **Empty drafter batches:** the wrapper runs the real `draft_batches` on the run's capture list.
- **20-30+ min startup per arm:** each process captures only the rows it measures (see "Capture list" below),
  not the production list (96 sizes at c256, 60 at c64), which the specialized decode prepares row by row.
- **Hard-coded bf16 state and old drafter:** everything comes from the production config.
- **Unrepresentative layout, attention and kernel config:** replaced by the `cfg2bench.py` output.

## What is timed

`[forward] ... median` is one CUDA graph per case, containing:
- the target forward over all 93 layers, with dummy weights;
- the drafter's K-step propose, with every draft accepted.

Each replay is the maximum over the 16 ranks (all DP ranks at C512), and the reported value is the median of iters x
samples.

Not included in either arm:
- target logits and LM head;
- rejection and sampled verify, including the specialized `runtime/sampled_verify.py`;
- CPU step overhead.

`--exclude-drafter` runs give the target/drafter split. The ls arm's "graph vs eager rel rms" is around 1.4 with dummy
weights. It is not a correctness signal; correctness needs real-weight runs.

## Files

| file | role |
|---|---|
| `cfg2bench.py` | Reads `jh/noquant/cfg/nq-<point>-xdr.yaml`, decode role plus global env. Writes the exact bench engine args and env as sourceable sh: TP16 DCP16, fp8 KV, float32 SSM state, mxfp4 indexer, attention and kernel flags, block size, max-num-seqs, capture sizes, speculative config with the drafter override, and `decode_backend` per arm. The capture list holds only the measured rows, their drafter batches and the capacity anchors (`--full-capture`: the production list). Each production arg is kept, dropped with a reason, or rewritten; an unknown arg is an error. `--snapshot` writes JSON copies so batch hosts need no PyYAML; a stale snapshot is refused. |
| `nq_step.py` | Wrapper around `vllm bench forward`. Handles the rendezvous and checks the code-under-test overlay (`--overlay`: vllm and kimi_k3 must resolve inside it; logs its revision and shipped files). Runs the real `specialized_decode_config` / `draft_batches` under the production compilation fields (FULL_AND_PIECEWISE, capture sizes, max capture). On the ls arm it also runs them on the full list and refuses to start unless every measured row is captured, every B is a drafter batch and the largest verify row / drafter batch equal the full list's (`NQ: capture:`). Fills the indexer with MX-aware random keys. Sizes the bench KV pool to the largest case (same pool in both arms) instead of gpu-memory-utilization, so the bench-only per-case state snapshot (~158 MiB per request at q4) and graph/eager transients fit; refuses a case that needs more KV blocks than the server's capacity at that utilization; logs per case the context set, KV blocks vs pool and capacity, snapshot GiB and peak memory (`NQ: KV:`, `NQ: case`). |
| `nq_shapes.py` | Cases for a list of verify rows M = B*q, in ascending rows. Prints and stores (`"ctx"`) each case's context summary; `--max-kv-tokens N` (`NQ_MAX_KV_TOKENS`) scales a case whose contexts sum above N. Context options: AgentX vectors (`q8_C1_C8` for c1/c8, `q8_C64`, `q4_C48_C160_C256`), `uNk`, `mix`. |
| (DP) | C512's decode role is TP4 x DCP4 x DP4 with EP16 MegaMoE; see "C512 point" below. `cfg2bench.py` derives the layout (`NQ_TP`, `NQ_DP`, `NQ_WORLD`; a world other than 16 is refused), `nq_step.py` logs per case where the rows go, `nq_summary.py` reads the SP coverage line and prints requests per GPU. |
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
| `test_nq_harness.py` | Pure-Python checks (no torch) of cfg2bench (including the c64 / c256 measured capture lists and their rounding against the checkout's `captured_rows` / `draft_batches`), the wrapper patches, overlay resolution, shapes and summary. |

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

### Capture list

The specialized decode prepares (and compiles kernels for) every captured verify row and every drafter batch at
startup, so the production list (96 sizes at c256, 60 at c64) cost 20-30+ min per process for ~8-10 measured rows.
`cfg2bench.py` now passes only what the process measures, identically in both arms, with max-num-seqs and
max-cudagraph-capture-size unchanged:
- every measured row M = B*q as a captured verify row (the runner rounds a size s up to whole requests,
  `shapes.captured_rows`) and B as a drafter batch (round_up(s, K) / K, `runtime/draft.draft_batches`). For B < q one
  size, B*K, gives both; otherwise two sizes, and each also gives its other rounding (a by-product the full list
  captures too);
- the two capacity anchors: the largest verify row and the largest drafter batch the full list captures under the
  run's limits. The decode buffers, mailboxes and several compiled kernels are sized by those maxima (below), so with
  them a measured row gets the kernels the production capture builds;
- the fewest sizes that do this (a greedy cover; the test checks it against the matching bound).

| point, rows | sizes | captured verify rows | drafter batches |
|---|---|---|---|
| c64 (q8, mns 32, cap 256): 8,16,32,64,128,192,256 | 7,14,28,56,64,112,128,168,192,224,256 (60 in production) | 8,16,32,64,128,192,256 + 56,112,168,224 | 1,2,4,8,16,24,32 + 10,19,28 |
| c256 (q4, mns 128, cap 512): 8,16,32,64,128,256,384,512 | 6,12,16,24,32,48,64,96,128,192,256,288,384,512 (96) | 8,16,32,64,128,256,384,512 + 12,24,48,96,192,288 | 2,4,8,16,32,64,96,128 + 6,11,22,43,86 |

`<run>.cfg.sh` lists the sizes, rows, batches, by-products and anchors as `# capture:` comments; rank 0 logs the
`NQ: capture:` check. `--full-capture` (`NQ_FULL_CAPTURE=1` for `nq_run1.sh` / `nq_ab.sh`) restores the old list:
the production sizes plus every measured row and its drafter row B*K (the NQRUN line says `capture=full`).

What depends on the other captured sizes (runtime/decode.py, providers and the cutedsl families, read 2026-11-06):
- not the per-shape choices: every family's `choose` and the coverage plan are functions of the shape and the
  `Context` (the one capture-derived field, `max_rows`, is read by no `choose`); the drafter tables are closed at the
  largest drafter batch (`closed`), which leaves every batch at or below it unchanged; `wide_rows`, the tuned set
  (`BATCHES` plus captured rows <= 64) and the drafter's Markov / context / all-reduce sets are per-row filters;
- but the capacity does: `SpecializedDecode.capacity` (the largest captured row) sizes the hidden / MLA query /
  projection / indexer buffers, the ShardBox and the o-projection's wide two-shot box, and `PreparedOproj` compiles the
  largest wide row's x tile as `min(round_up(b, chunk), cap)` (a partial last tile when cap = b); `MoEScratch` sizes the
  wide tail's `TailRowsBox` by the largest MoE batch and `tail_rows` compiles its kernels with `box.cap`; the dense MLP
  buffers take `min(capacity, 128)`; on the drafter side `PreparedDraftAr` compiles every wide all-reduce site with
  `wide_cap` = the largest wide query row, the draft DCP attention widens its box to the largest wide row, and
  `PreparedMarkov` gets its wide mailbox when the largest batch exceeds 8. The anchors make all of these equal to the
  full capture's (on the target topology every wide row is admitted by the MoE / o-projection tables, `check_wide`,
  so their maxima are the capacity); nq_step.py refuses a list whose maxima differ. Not anchored: maxima over rows a
  table admits only in part, e.g. the head ring mailbox (`RING_HEAD`, <= 256 rows; its row stride does not depend on
  it) and the draft attention's widened box when residency drops the largest drafter row (`uncovered_rows`).

Either way each measured shape is a captured FULL graph: unpadded verify, B in `draft_batches()`. The production
limit is
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

## C512 point (TP4 x DCP4 x DP4, EP16 MegaMoE)

`nq-c512-xdr`'s decode role has 16 GPUs: 4 DP groups of TP4 = DCP4, with MoE expert-parallel over all 16 ranks
(`deep_gemm_mega_moe`, mxfp8 activations), `--use-replayssm`, q4 (3 drafts), max-num-seqs 128 per DP rank and capture
sizes 4..512. The specialized arm is the sequence-parallel decode from `wzhao/k31-c512-hybrid` (phase 1: the drafter
and the MoE are declared stock modules, coverage lines `Kimi K3 SP decode coverage M<R>(q4)`). Stage that checkout as
the overlay: both arms run it, so the stock arm is the same tree's stock code.

Sparse MLA kernel: the config leaves `flashinfer_sparse_mla_kernel` at `auto`, which resolves to prims_ts under DCP4
with the per-tensor fp8 cache on VR200, and the SP decode refuses that (`runtime.sp_unsupported`; it runs trtllm-gen
only, `c512_contract.md`: no prims_ts). `cfg2bench.py` therefore sets `--kernel-config
{"flashinfer_sparse_mla_kernel": "trtllm"}` on **both** arms for this layout (a `C512:` note in the log) and refuses an
explicit prims_ts. The production C512 config needs the same setting before the specialized decode can serve it.

GPU tests: the 4-GPU stage of `nq_gputest_inner.sh` also runs `tests/distributed/test_kimi_k3_sdec_sp_*.py` (TP4 =
DCP4 needs 4 GPUs; the 2-GPU stage skips them) plus the torch-importing CPU tests `test_specialized_decode_sp_*.py`
and `test_specialized_decode_coverage_preflight.py`.

How DP runs in the bench (nothing to patch): `vllm bench forward` uses the external launcher, so the world is
tp x dp = 16 ranks and each rank's DP rank is `RANK // 4` (`config/parallel.py`). Ranks 0-3 (one node) form DP rank 0,
and so on. Every rank builds the same case: cases are per DP rank, and the context draw is seeded by `--seed` only.
The DP sync inside `execute_model` therefore sees equal token counts on every DP rank: no DP rank idles, there is no
dummy batch and no DP padding (a padded case would be logged as `WARNING: ... padded`). Synthetic routing builds the
whole fixture for rows x dp and gives each DP rank its slice (`routing.py`). The SP path still calls the stock MoE
gate, so both arms get the same routing. Because of all this:
- `--rows`, B, max-num-seqs and the capture list are **per DP rank**, as in serving. Rank 0 logs, per case,
  `NQ:   per DP rank: B <B>, <rows> rows (TP4 group), requests per GPU <B/4>; DP4: ... <rows x 4> rows into the EP16
  MoE per step`.
- The summary prints `req/GPU` = B / tp (B / 16 at TP16), and the CSV also has `tp`, `dp` and `req_per_gpu`.
- Not measured: unequal DP batches (the SP padding-mask path) and DP ranks with no work.

Rows (per DP rank; production limit min(128 x 4, 512) = 512, so every row runs in one process per arm):

| rows M | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 384 | 512 |
|---|---|---|---|---|---|---|---|---|---|
| B per DP rank | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 96 | 128 |
| requests per GPU | 0.25 | 0.5 | 1 | 2 | 4 | 8 | 16 | 24 | 32 |

M512 (B = 128 per DP rank, 512 in flight in all) is the C512 operating point. M4 is the n = 1 point the hybrid plan
lists as a stock baseline gap. The measured capture list is 15 sizes (3,6,12,16,24,32,48,64,96,128,192,256,288,384,512)
instead of the production 128. Captured verify rows: 4,8,12,16,24,32,48,64,96,128,192,256,288,384,512. Drafter
batches: 1,2,4,6,8,11,16,22,32,43,64,86,96,128. The capacity anchors are 512 / 128, as in production.

```
bash stage_overlay.sh <k31c512 checkout> <c512tag>
NQ_OV=$NB/ov/<c512tag> bash $NB/tools/nq_ab.sh <jobid> h1,h2,h3,h4 $NB/results/c512-<c512tag> c512 new \
  4,8,16,32,64,128,256,384,512 ls,stock,stock,ls agentx
```

Before the first C512 run, check:
- **KV capacity.** The AgentX q4 vectors (from c256 in-flight contexts) at B = 128 sum to about 24M tokens. A DCP4
  group holds a quarter of TP16's capacity, so M512 may exceed the server's capacity. `nq_step.py` then refuses at
  startup and prints `~N fits every case`. Rerun with `NQ_MAX_KV_TOKENS=N`; both arms get the same scaled contexts.
- **Bench-only snapshot memory.** This is the per-case state clone. At TP16 q4 it was ~158 MiB per request. At TP4 it
  is roughly 4x that per rank (4x the KDA heads per rank), so tens of GiB at B = 128. A case that OOMs only costs
  the larger rows, because cases run in ascending order. Check `NQ:   state snapshot` on the first run.

## Measurement matrix (3 blocks: 12 nodes x 3:30 h, 3 slots of 4 nodes)

| ID | Point (decode role) | q | Rows (M = B*q) |
|---|---|---|---|
| q4-hi | `c256` (c160 has the identical decode role): wide DSA 512, mns 128, cap 512, DCPX 1 | 4 | 8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024 |
| q8-hi | `c64` (c48-1p1d is identical): wide DSA 256, mns 32, cap 256, DCPX 1 | 8 | 8, 16, 32, 64, 128, 256, 512, 1024 |
| q8-lo | `c1`: DSA core rows 1-16, DCPX 0 | 8 | 8, 16, 32, 64, 128, 256 |
| q8-c8 | `c8`: DSA core and DCPX 1 | 8 | 8, 16, 32, 64, 128, 256 |
| q4-mid | `c48-2p1d`: wide DSA 256, mns 64, cap 256 | 4 | 8, 16, 32, 64, 128, 192, 256 |

Rows above a point's production cap (q4-hi above 512, q8-hi above 256) measure the arbitrary-batch goal.

Budget per slot: about 25 min a process and a 10 min end margin, so at most 7 processes (set with the full capture
list; remeasure with the measured-rows list and lower `NQ_RUN_EST`). Slots A and B use 6, so a slow
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
