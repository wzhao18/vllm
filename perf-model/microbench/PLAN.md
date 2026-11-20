# Layer-wise microbenchmark plan (next phase)

Status: **plan only**. Nothing below is launched until the user says go. When it runs, every step is an
`srun --overlap` step on an existing hold (Hecate, holdd21 pool). No new `sbatch`, no re-queue between stages, and
keepalives stop before each timed step (check that the ka log stops). The `$W/...` paths here are read-only
references; new harness code is written as copies under `perf-model/microbench/harness/` and synced to
`$W/pm-mb/` (a new directory, not created yet). Existing scripts are never edited in place.

- `W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr`
- `PM=/Users/weizha/k31vr-dev/perf-model`
- `MB=$W/pm-mb`, the run directory on Lustre, which owns copies of every dependency (sweep-owns-dependencies)
- `MMB=$W/megamoe-bench` (MegaMoE harness, read-only reference)

**Scope (2026-11-18).** The model scope is `topologies.SCOPE`: EP <= 16 and at most 16 GPUs (4 VR200 nodes) per
engine or EP domain (`topologies.EP_MAX`, `microbench/results/model/README.md`). The earlier version of this plan
targeted EP64 layouts (D7-a, D8, P6) and EP32 (D4-32) with 16-node holds. Those are out of scope and are not benched.
Their matrix rows are kept with priority `OOS` only so that their ids stay valid.

Inputs:
- `microbench/matrix.json`: 1006 cases (56 P0, 175 P1, 737 P2, 38 OOS), built by `microbench/build_matrix.py`
  (`python3 perf-model/microbench/build_matrix.py`, stdlib only).
  - Every case id below (`gemm-004`, `megamoe-006`, ...) refers to `matrix.json:cases[].id`.
  - **Ids are stable.** `microbench/ids_v1.json` freezes the 913 ids of the first build: the same (family, shape,
    dtype) always gets the same id, and new cases get ids from 914 up (`id_new: true`). The ops1 lane
    (`microbench/results/ops1/src/cases_ops1.json`) uses v1 ids, so never renumber.
  - Topology tiers (`cases[].scope`, `uses[].scope`):
    - priority: P1, P2, P3-16, P4-2, P6-16, P9, D1, D3, D4-16, D5, D7-2-16, D7-4-16, D7-8-16 and D8-16 (deployed
      layouts plus the EP <= 16 frontier).
    - cover: the rest of SCOPE (P2b, P4-4, P5-4, P5-8, P6-4-16, P9-16, D2, D6, D9-16). These are listed with score
      weight 0, so on their own they are never P0 or P1.
    - oos: P6, D4-32, D7-2/4/8 and D8 (EP64/EP32).
  - `topo_ids_in_scope` lists the in-scope topologies of a case. Stage selection uses it, not `topo_ids`.
- `out/optables.json`: the per-op t_sol and t_ach the measurements are divided by.
  `microbench/results/model/targets.json` holds the same numbers per in-scope topology.

Goal: per-op derating `d = t_meas / t_ach` at the exact per-GPU shard shape, then per-layer and full-step agreement
with `vllm bench forward`. Explain the known gaps with named op rows (`out/meta.json` anchors):

| anchor | gap (measured / achievable step) |
|---|---|
| C512 hybrid decode D3 (B4..B64 per DP4 rank = r 1..16) | 1.71-1.94x |
| C512 prefill P2 (MXFP8-expert headline) | 1.74x |
| D1 e2e decode | 1.06-1.21x for the release runs (B <= 23, r <= 1.5); frozen runs 1.08-1.31x up to B 39, then 1.55-1.67x at B 70-89 (C224/C256) |

---

## 0. Ground rules for every bench row

1. **Shape.** Use the `shape` of the matrix case exactly: per-GPU M/N/K, tokens/rank, heads/GPU and keys/row. Drafter
   rows with fractional M are rounded up (`ceil`).
2. **Timing.** Use a CUDA graph with rotating weights or inputs larger than 2x L2, so weights stream from HBM. Report
   the median of at least 20 samples, with p05 and p95. Multi-rank rows use the rank-max per sample.
   - Prefill-sized ops above the capture cap (16K-row KDA chunk, EP16 T16384 MegaMoE) are timed eagerly with events,
     a barrier before, and the rank-max.
   - Decode ops are graph-timed. `t_ach_kernel_us` already includes the 0.78 us launch gap. Compare against that
     value, not `t_ach_us`.
   - **MegaMoE whole-kernel rows compare against `t_ach_fused_us`**, not `t_ach_kernel_us`. The fused kernel moves
     tokens over NVLink inside the kernel. `t_ach_kernel_us` covers only the on-GPU ops (dispatch/fc1/act/fc2/combine);
     `t_ach_fused_us` adds the P11 + P12 achievable time of the same step (`p11_t_ach_us`, `p12_t_ach_us`). Dividing a
     whole-kernel time by `t_ach_kernel_us` would inflate d by up to 2.7x at prefill sizes (megamoe-017: 271 vs
     733 us; megamoe-010: 1884 vs 4030 us). TP-expert cases (D1) have no P11/P12, so there the two values are equal.
3. **Identity line on every row.** Record the tree, commit, image, MegaMoE version and quant kind as logged by the
   kernel (not as requested), plus the job id and hosts.
4. **Same fabric.** A multi-node row runs inside one allocation and one NVL72 segment (`--segment`). xdr and spx
   are never mixed.
5. **Gate per kernel.** Before any timing, run the kernel once and compare it with the reference
   (`--dump-output` / `_check` / MMCHECK). A row whose correctness check fails is `error`, not a measurement.
6. **Precision pins (rules; a row that violates one is invalid).**
   - `mxfp4_release`: every linear (KDA/MLA projections, router, latent, shared experts, dense, lm_head, drafter) is
     BF16. In `wpg2/step/run.sh` that means `wq=bf16`, never `mxfp8`; `bench_proj_gemm` MXFP8 arms only feed
     `nvfp4_modelopt` rows (gemm cases with `dtype=fp8` are all `nvfp4_modelopt`, by construction).
   - MLA KV is FP8 (576 B per-tensor row). `kv=fp8` or `f8pts` only; never `fp4` / `nvfp4_ds_mla` (the default
     `--kv fp8,nvfp4_ds_mla` of `bench_sparse_core.py` must be narrowed to `--kv fp8`).
   - The indexer is MXFP4 Q x MXFP4 keys (68 B/key) with an FP32 epilogue. Top-k is the exact top-2048. No capped or
     approximate top-k arm counts as a measurement.
   - KDA recurrent state and conv state are FP32. Use the DS conv layout (production), not SD.
   - Routed experts: MXFP4 weights; activations MXFP8 for `mxfp4_release`, NVFP4 for `nvfp4_modelopt`
     (`shape.act`). Section 2 covers the vendored NVFP4 switch.
   - k = 3 (q = 4 rows per request). `bench_layer.py` and `bench_sparse_core.py` default to `--q 8` (k = 7) and
     need `--q 4`.

---

## 1. Bench families: harness, gaps, launch template

`J` is the hold job id, `H1..Hn` its hosts (`scontrol show hostnames`), `G` a GPU index. All paths below were checked
to exist on Hecate on 2026-11-18, unless they are marked new.

**Already running: the ops1 lane** (`microbench/results/ops1/src/`, run dir `$W/mb-ops1`). It runs `gemm_bench.py`
(BF16 cuBLAS / low-latency CuTe, BF16 BMM, MXFP8 cuBLASLt + CuTe DSL) and `attn_bench.py` (idx, mla, kda, ares) over
`cases_ops1.json` (682 v1 ids) with CUPTI cold-L2 timing. Its rows key on v1 ids and stay valid. Stages A-C reuse
its rows and do not re-run its cases. The new ids (`id_new`) are the delta still to run.

### 1.1 GEMM (464 cases, 21 P0)

- **Reuse:** `$W/wpg2/src-d11/benchmarks/kernels/k31_vr/wep/bench_proj_gemm.py`. It is cuBLAS BF16 vs MXFP8 with
  rotating weights and the activation quant split out. Its `SETS` dict (line 38: `tp1`, `tp4`, ...) is hard-wired.
  NVFP4 rows use `wep/bench_nvfp4_gemm.py`. MXFP8 backend choice uses `wep/bench_mxfp8_backends.py`. Single GPU,
  run through `$W/wpg2/run1.sh <tag> <gpus> <log> <cmd...>`.
- **ops1 `gemm_bench.py`** already reads matrix cases. Prefer it over writing a new `gemm_cases.py`.
  - Batched rows (`batch>1`, MLA w_uk/w_uv) use `torch.bmm` / grouped.
  - The `fused_epilogue` family (54 cases) is not timed on its own. It is checked inside its producer GEMM through
    the `ops` list.
- **Launch (if a copy is needed):**
  ```
  WPG2_JOB=$J bash $W/wpg2/run1.sh gemmA $G $MB/logs/gemm-A-$G.log \
    python3 $MB/harness/gemm_cases.py --cases $MB/matrix.json --ids-file $MB/stages/A.gemm.$G.ids \
      --out $MB/rows/gemm.jsonl --rot-bytes 4G
  ```
  Shard the id list over 4 GPUs; all 4 run in parallel on one node.

### 1.2 MegaMoE (44 cases, 14 P0; 14 OOS)

Harness: copies of `$MMB/scripts/run_b10.sh` + `inner_env_b10.sh` + `b10_ident.sh` + `emit_rows3.py` and the
drivers `$MMB/scripts/py/{fi_bench2.py, fi_vllm2.py, mmb_env.py, mmb_weights.py, mmb_ident.py}`. Copy them; do not
edit them. (An earlier version of this plan cited `py/fi_opt.py`, which does not exist.) Two code paths per id, and
both are recorded:

- **(a) FI final (B10).** `run_b10.sh B10` runs the frozen FlashInfer `benchmarks/bench_kimi_k3_moe_ep.py` through
  `fi_bench2.py` with **`--provider current`** (not `kimi`):
  ```
  --provider current --quant-kind {nvfp4 | mxfp4_mxfp8} --wire bf16 --routing random
  --execution-mode cuda_graph --calls-per-graph 10 --tokens <tokens_per_rank> --max-tokens <tokens_per_rank>
  --profile        # per-kernel device time of one graph replay (split)
  ```
  - `--routing random` is required. The default pattern routing touches only min(T, 56) experts.
  - `+kernel` (`fi_bench2 --mm-kernel-only`) times the staged compute only.
  - `+vllm` (`fi_vllm2.py`: workspace capacity 8192, `runtime_max_tokens_per_rank` = T, knobs=None, `prepare([T])`)
    is still **the FI final tree** called the way vLLM's FlashInfer moe_ep path calls it. It is not the vendored
    `megamoe_rubin` code.
- **(b) Vendored vLLM path.** This is the code the deployed step runs:
  `vllm/models/kimi_k3/nvidia/megamoe_rubin`. Per `VENDOR.md` that is FlashInfer 0.7.3.dev20261031 `cutedsl_megamoe`
  1667b47a plus S0/S1; vLLM-side S2 and die-local streams (a9958ca1c4) sit on top.
  - **New: `harness/moe_vendored.py`.** It builds a `megamoe_rubin` session (`runtime.py`) at EP and T and reuses
    the `mmb_weights` seed-0 experts and the `reference.bf16_partial` check that `fi_bench2` already uses.
  - It is needed because the derating that feeds the stage D step comparison must come from the same code the step
    runs.
- **Launch:**
  ```
  J=$J bash $MB/harness/run_b10.sh B10 <EP> <tokens csv> random <nvfp4w_nvfp4a|mxfp4w_mxfp8a>[+kernel|+vllm]
  ```
  One srun step on the first EP/4 hosts. EP16 needs 4 nodes, EP8 needs 2.
- **EP coverage.** In scope the matrix needs only EP16 and EP8 (P1); D1 uses TP experts.
  - The `kimi` provider rejects world sizes outside {1, 8, 16} (`sm107/kimi_k3/backend.py:59`).
  - `fi_bench2` lifts the test helper's world-size guard. B6 measured EP28/EP32 this way (`$MMB/results/rows.jsonl`).
    EP > 16 is out of scope anyway.
  - The vendored runtime only requires `896 % EP == 0`. Its shipped knob records (`knobs/k31_vr.json`) cover EP4/8/16.
- **O2 dedup (P6-16: `megamoe-915` mxfp8, `megamoe-916` nvfp4, `comm-914`).** FI final has no dispatch dedup (the
  L4 dedup lane was dropped: "dedup slower", `$MMB/opt/final_ab/ledger_wrapup.md`). Measure the non-dedup twin
  (`megamoe-041` / `megamoe-049`, same EP16 T8192) and mark the dedup rows `decomposed`: compute d from the twin,
  and P11/P12 bytes from the model. Section 6 has the open question.
- **D1 TP16 expert slice** (`megamoe-083/086` P0; `-152/163/229/233` P1; 1/16 of every expert, inter 192): this is
  not MegaMoE EP. Use `$W/dsrel/tools/kern/moe/bench_moe_kern.py experts --rows 64,512,2048` (wide routed experts at
  TP16 rank geometry), run through a copy of its `task.sh`.

### 1.3 Comm (225 cases, 18 P0; 23 OOS)

- **Reuse:** `$W/hw/symmbench.py` (flag ping-pong, symm-mem one-shot AR) and `$W/hw/ncclbench.py` (NCCL over MNNVL).
  ncclbench has a fixed `sizes` list (line 16) that stops at 256 MiB, with groups of 4/8/16/32.
- **New: `harness/comm_cases.py`.** A copy that takes `(primitive, N, msg_B / in_B)` from the matrix comm cases and
  times every `candidates` entry, so both `chosen_sol` and `chosen_ach` are measured. It covers:
  - AR: P1 NVLS, P1b two-shot, P1c one-shot.
  - RS: P3 NVLS, P3b push.
  - P2 SP-AG.
  - **Large-message multimem/NVLS AR and RS at N = 4/8/16 from 64 MiB to 235 MiB** (`comm-011` AR N8 235 MB,
    `comm-018` RS N4 235 MB, `comm-019` AG N4 176 MB). These are unmeasured today.
- **P11 EP-dispatch / P12 EP-combine on EP topologies are not benched for the derating.** They run inside the fused
  MegaMoE kernel, so their d is the MegaMoE whole-kernel d (section 0.2). That covers `comm-001/002/015/914`,
  `comm-075/117/135` (P12) and `comm-147/148/157/166/170/184` (P11), all of which are P0 only because of their share.
  - Optional diagnostic: the `--profile` dispatch/combine split of the MegaMoE row, plus a token_comm-only timing on
    the same symm buffers and SM count. Never NCCL all-to-all, which is kept as a reference column at most.
- **Launch:**
  ```
  srun --overlap --jobid $J -N<n> -w <hosts> --ntasks-per-node=4 --container-image=$IMG \
    --no-container-mount-home --container-mounts=$W:$W \
    python3 $MB/harness/comm_cases.py $MB/rdv/$(date +%s%N) --cases $MB/matrix.json --ids-file <ids> \
      --out $MB/rows/comm.jsonl
  ```
  - N=4 runs on 1 node, N=8 on 2, N=16 on 4.
  - `$IMG` is the 3f1bcfd82b image from `$W/bench/run.sh`.

### 1.4 KDA (19 cases, 2 P0)

- **Decode** (`kda-025` B128 x 24 heads P0; P1: `kda-090` B32 x 24, `kda-064/149` x 96 heads (DEP),
  `kda-082/185` x 48, `kda-099/175` x 12, `kda-146/196` x 6).
  - `$W/dsrel/tools/kern/kda/bench_kda.py` is **hard-wired to TP16** (`HEADS = 6`, line 51). Its `--shard i/n`
    splits the case list across processes; it does **not** set heads. It only fits the D1/D8-16 6-head rows.
  - The other head counts need a copy with a heads override: **new `harness/kda_decode.py`**, built from
    bench_kda section b (verify core, FP32 state, q4, iso mode, 69 cold layers). Alternatively, ops1 `attn_bench.py`
    `kda` (sdec PreparedKda core at (B, q=4)) if its heads follow the case.
  - `benchmarks/kernels/benchmark_kimi_k3_kda_decode.py` in k31vr-dev has `--heads`. It needs the
    `_C.fused_kda_decode` op and is written for an Infinity-Cache (non-NVIDIA) target. Do not use it on VR.
- **Prefill** (`kda-020`: 16384 tokens, 3 seqs, 24 heads, chunk 16, FP32 state, checkpoint export; P1 `kda-051` 48
  heads, `kda-077` 96 heads, `kda-132` 12 heads): ops1 `attn_bench.py kda` (FlashKDA, plus the segment-parallel
  prefill when its planner accepts) or **new `harness/kda_prefill.py`**. The latter times the FlashKDA
  chunked-prefill chain (conv, gate_beta_norm, core_prefill, o_norm) eagerly, using `$W/b1/kda/tools/kda_chain.py`
  as reference.
- **Launch:** `WPG2_JOB=$J bash $W/wpg2/run1.sh kda $G <log> python3 $MB/harness/kda_prefill.py --cases ... --ids ...`

### 1.5 DSA indexer + top-k (28 cases, 1 P0), sparse MLA (27), AttnRes (49), drafter (96)

- **Indexer/top-k:** reuse `$W/bench/k31_vr/bench_layer.py --model k31 --arms indexer` for the DCP key-shard modes
  (`--tp/--dcp/--interleave/--indexer-kv-dtype`, `--kv-cache-dtype fp8`, `--q 4`; release defaults: fp8 MLA KV,
  mxfp4 indexer).
  - **DSA-DP owner mode** (`dsa_indexer_topk-042` P0: prefill 8192 rows x 75.6K keys, P4-2/P6-16;
    `-087`/`-181` P1: decode 128/32 rows x 135K keys): **new** single-GPU `harness/idx_owner.py`. It calls
    `fp4_mqa_logits` + exact top-2048 over all rows of the owned requests and measures the MMA+epilogue and top-k
    columns separately. ops1 `attn_bench.py idx` already does the paged decode variant.
  - Build for 10.0f, not 10.7a (DeepSelect traps on 10.7a). `bench/run.sh` env `k31` is tree 98c712e55 +
    FlashInfer 0.7.3.dev20261030; point it at the release overlay instead (`ov-*` envs).
- **Sparse MLA:** `$W/bench/k31_vr/bench_sparse_core.py --kv fp8 --q 4 --kernels trtllm,prims_ts`. Existing
  `--layouts` are dcp16, dcp8, tp8, tp16, shardq8 and shardq16.
  - Matrix layouts it lacks need a copy with new layout rows:
    - `dcp4` (D3: `sparse_mla-336/527/626`);
    - `shardq4` (P2: `sparse_mla-089`);
    - full 96 heads x 2048 keys per row with no DCP (DEP / DSA-DP: `sparse_mla-094/167/272`, P1; `-290` P2).
  - Replicated-DSA padded heads (D6/P5, cover only) come last.
  - Record whether the kernel runs FP8 Q x FP8 KV (the model's assumption) or BF16 Q. ops1 `mla` uses trtllm-gen
    with 96 heads padded to 128, which is a real kernel cost and stays in d.
- **AttnRes:** `$PM/../benchmarks/kernels/benchmark_kimi_k3_attn_res.py` (flags `--num-tokens`, `--num-blocks`).
  Map matrix `rows` -> `--num-tokens` and sources -> `--num-blocks`; for `L.*` take the layer mean over
  ceil(l/12) blocks. ops1 `attn_bench.py ares` already does this.
- **Drafter:** `$W/dsrel/tools/kern/draft/{bench_draft_gemm,bench_draft_markov,bench_draft_dcp}.py`. These are all
  P2, so they are deferred to stage D.
- **Launch:** `bash $W/bench/run.sh $J <nodes> <tasks-per-node> <env> <log> bench_layer.py --phase {prefill|decode} ...`
  for the existing modes; single-GPU harnesses go through `run1.sh`.

### 1.6 Step and per-layer (stage D)

- **Decode step (D3, deployed):** `$W/hti/kb/tools/kb_run1.sh <jobid> <4 hosts> <out> {c512|c256} <rows> <ctx> --
  --profile-dir <out>.prof`. It wraps nq_step on the deployed decode tree, and `kb_parse.py` gives the component
  split. kb_parse's rules are an inline default list (override with `--rules <json>`); there is no rules.json file.
  The cfg JSONs exist (`hti/kb/cfg/c512-{spx,xdr}.json`, c256 too), so mk_cfg is not needed unless a new point is
  added. Rows = B x 4 per DP rank.
- **Prefill step:** `$W/hti/kb/pf/pf_run1.sh <jobid> <hosts> <out> $W/hti/kb/cfg/c512.cfg.sh <cases>`, then
  `pf_breakdown.py` for the critical-path split.
- **Per layer:** `$W/wpg2/step/run.sh <jobid> <nnodes> <name> <layout> <wq> <kv> <set>` with `K31W0_LAYERS=N`
  (`bench_dep_step.py`: it truncates the target to its **first** N decoder layers).
  - Layouts are d4x4, d4x1, dep16, dep4, tp16 and tp16ls. There is no D5, D7-a-16 or D8-16 layout.
  - Always `wq=bf16` and `kv=fp8` or `f8pts` (section 0.6).
  - Layer 0 has a dense MLP. DSA layers are 3, 7, 11, ... (0-based). So:
    - `step(3) - step(2)` is one KDA + MoE layer (layer 2);
    - `step(4) - step(3)` is one DSA + MoE layer (layer 3);
    - `(step(8) - step(4)) / 4` is the 3 KDA : 1 DSA mean;
    - `step(2) - KDA+MoE layer - (KDA + dense layer 0)` is the fixed cost (embedding, drafter, F.*).
  - The earlier "4-layer step minus 4 x mean" mixed layer 0's dense MLP into the fixed cost.
- **New: `harness/case2kernels.py`.** The adapter from kernel name to matrix op id. It merges kb_parse's rule list
  and the `wpg2/step/components.py` regexes into one map whose keys are optables `id`/`fam`, so trace time lands on
  the same rows as the op benches. It also lists unclassified kernels. If more than 3% of step time is
  unclassified, the comparison for that row fails.

---

## 2. MegaMoE build and precision pinning

**Version.** Every MegaMoE row records one of these two strings. Evidence is in `$MMB/audit/B10-identity.md`,
`$MMB/opt/LEDGER.md` (PHASE INTEGRATE) and `$MMB/opt/final_ab/ledger_wrapup.md`.

- **FI final.** FlashInfer commit `6e438852bf5a993b522ceec31e62c0bdb8c65d63`, tag
  `wzhao/kimi-k3-megamoe-sm107-final-20261117` (branch `wzhao/kimi-k3-megamoe-sm107-rebase`).
  - Contents: round 1 a783e5470 (C3 N-tile policy, C4 decode landing up to T1024, C5 one-barrier tail, C6 die-local
    expert weights, die_split policy) plus L1 per-batch-size kernels (14f06a4ca) plus L3n NVFP4 decode weight
    prefetch / EVICT_FIRST (6e438852b). C1, C2, L2, L4 and L5 were rejected or dropped.
  - **The frozen tree is an overlay, not a checkout.** `frozen/fi-6e438852b` = `frozen/fi-e5cc18d95` plus the 19
    files of `git diff e5cc18d95 6e438852b` (`version.txt` says 0.7.5).
  - Version string:
    `flashinfer 6e438852b (frozen overlay e5cc18d95 + 19 files), next_cutedsl_megamoe manifest md5 fda3fe0862b3088018416550cac8023f (58 files)`.
  - `frozen/MANIFEST.txt` does not list fi-6e438852b; B10-identity.md is its record. No wheel: CuTe DSL JITs at
    first use.
  - **Pin:** use the read-only overlay `/lustre/fsw/coreai_comparch_inferencex/weizha/frozen/fi-6e438852b`, placed
    first on PYTHONPATH by a copy of `inner_env_b10.sh`, together with `frozen/vllm-036d832e7`.
  - Use the image named in `frozen/MANIFEST.txt`, mounted read-only.
  - Use our own `FLASHINFER_WORKSPACE_BASE`, `FLASHINFER_JIT_DIR`, `FLASHINFER_CUBIN_DIR` and `CUTE_DSL_CACHE_DIR`
    under `$MB/cache/cutedsl-fi6e43`, never shared with another tree. `inner_env_b10.sh` points at
    `$MMB/cache/*-b10`, and the copy must change that. Also set `CUTE_DSL_ARCH=sm_107a`. HOME goes under Lustre.
  - **Proof:** every step runs `b10_ident.sh` and keeps its MMTREE (must be `fda3fe08...`; every MMTREE line in
    the existing B10 EP16 logs (24) shows it), MMOUTSIDE and MMPKG lines. A step with a mismatching MMTREE is dropped.
- **Vendored (what vLLM runs).** `vllm/models/kimi_k3/nvidia/megamoe_rubin` at the tree under test.
  - The last base change on `wzhao/k31-agentx-dev` is `dfaac3bc10` (2026-11-11). Record whether `fe2e1985df`
    (redundant experts, 2026-11-15) and `8c544f2843` (2026-11-17) are present; neither is in the local HEAD.
  - Version string: `vendored megamoe_rubin <vllm commit> (FI 0.7.3.dev20261031 cutedsl 1667b47a + S0/S1 + S2/DL)`.
  - It predates round 1, so it lacks C3-C6/L1/L3n as FlashInfer commits. This has not been checked file by file; the
    die-local streams are a separate vLLM implementation.
  - The ratio `FI final / vendored` per case is reported as the "MegaMoE upgrade headroom" column. It is not a
    derating.

**MXFP8 activations for `mxfp4_release`** (the model assumes MXFP4 weights with MXFP8 activations at every token
count):

- **FI final:** `--quant-kind mxfp4_mxfp8`. Matrix cases with `act=mxfp8` (`megamoe-008/009/010/012/033/041/086/915`
  ...) use it. Cases with `act=nvfp4` use `--quant-kind nvfp4`.
- **Vendored:** there is no flag, by design. `rubin_quant_kind(max_tokens_per_rank)` (`experts.py`, constant
  `RUBIN_MX_MAX_TOKENS_PER_RANK = 512`; line 87 in k31vr-dev, line 81 in frozen vllm-036d832e7) picks MX only when
  the layer's **workspace capacity** (the server's max tokens per rank, not the live batch) is <= 512.
  - So **the deployed C512 decode (MNBT 8192 / TP4 = 2048 per rank) runs NVFP4 activations at every decode batch**,
    including B4..B64. The D3 anchor is therefore an NVFP4-activation measurement. Its stage A attribution uses the
    `act=nvfp4` twins (`megamoe-006/007/013`) for the as-deployed derating and the `mxfp8` rows for the model's
    precision.
  - **Prefill rows** (4096/8192/16384 tokens/rank) also run NVFP4 in vLLM. For these, bench the vendored path both
    ways. One run is as shipped (NVFP4), which is what the step measures. The other uses a **scratch copy** of the
    tree with `RUBIN_MX_MAX_TOKENS_PER_RANK` raised. That copy lives in `$MB/trees/vllm-mxpin`, is never committed,
    and its md5 is logged.
  - Every row asserts the quant kind the kernel logs (`quant_kind=` line) and fails if it does not match the
    requested one.
  - In stage D, the `mxfp4_release` step comparisons (D3 C512, P2) use the NVFP4 vendored derating, because that is
    what runs. The MX gap is reported separately as "precision mismatch vs model". NVFP4 W4A4 experts are within the
    official rules; this is a model/deployment mismatch, not a rule violation.

---

## 3. Stages

Ids are matrix case ids. Select a stage with
`python3 -c "import json;m=json.load(open('matrix.json'));print('\n'.join(c['id'] for c in m['cases'] if set(c['topo_ids_in_scope'])&{'P2','D3'} and c['priority'] in ('P0','P1')))"`.
P0 and P1 go first, and P2 only after the re-rank (section 5). OOS cases are never run. Skip any id the ops1 lane has
already measured.

### Stage A: C512 hybrid gap decomposition (P2 prefill, D3 decode)

Goal: explain the 1.74x (P2) and 1.71-1.94x (D3) gaps op by op. There are 242 in-scope cases: 31 P0 and 60 P1.

| order | what | ids (P0) | harness | GPUs |
|---|---|---|---|---|
| A0 | Re-measure both anchors with profile: D3 at rows 16/128/512 (B4/B32/B128 per DP group = r 1/8/32), ctx = AgentX L=135K, both ckpts; P2 c512 cases | step | kb_run1 c512, pf_run1 c512 | 16 |
| A1 | MegaMoE EP16 decode T4/32/128, FI final and vendored, nvfp4 (as deployed) and mxfp8 (model); compare with `t_ach_fused_us` | megamoe-006/007/008/009/013/033 | §1.2 | 16 |
| A2 | MegaMoE EP16 prefill T4096: FI mxfp8, FI nvfp4, vendored NVFP4 as shipped, vendored MX-pinned | megamoe-012/017 | §1.2 | 16 |
| A3 | Comm: RS N4 235 MB and SP-AG N4 176 MB prefill (P1 tier: decode RS/AG N4 `comm-092/093/119/120/123/124`). P11/P12 rows (`comm-002/075/117/135/147/148/157/166/170/184`) take the MegaMoE d (§1.3) | comm-018/019 | §1.3 | 4 |
| A4 | GEMMs (all bf16 except `gemm-030` fp8, nvfp4_modelopt only). Prefill: KDA in_proj 16384x12440x7168 bf16 + fp8, q/g projection 4096x12288x7168, o_proj 16384x7168x3072. Decode: M 4/32/128 x 12288x7168, KDA in_proj M 16/512 x 12440. P1 tier has the 3584-wide shared-expert GEMMs (`gemm-103/104`, fp8 `gemm-113`) | gemm-004/023/026/030/032/035/066/096/097 | §1.1 | 1 node |
| A5 | KDA decode B128 x 24 heads (P1: B32, B4); KDA prefill 16384 tok x 24 heads | kda-020/025 | §1.4 | 1 node |
| A6 | P1 tier of the same topologies (indexer DCP4 key shard `dsa_indexer_topk-241`, ShardQ4 `-043`, sparse MLA `sparse_mla-089/336`, attnres) | P1 ids | §1.5 | 1 node |

Decomposition output: for each (P2 or D3, regime), list the op rows ranked by `(d - 1) x t_ach x count`. Then compute
the sum of derated ops against the A0 step and kb_parse/pf_breakdown component times.
- A0 r32 (B128 at 135K) may not fit KV per DP group. Set `NQ_MAX_KV_TOKENS`. If it still does not fit, measure B64
  (the existing anchor, r 16) and add r16 to the build so the model has a matching row.
- Cross-check against the prior C512 tables (`$W/hti/SOL_TABLE.md`, `GAP_LEDGER.md`, `hti/cpm/c512`) by reading
  only. Nothing from them is copied into the repo.

### Stage B: prefill candidates DEP16 (P3-16), P9, P6-16, P4-2, and P1

138 in-scope cases: 17 P0 and 32 P1.

| order | what | ids (P0) | notes |
|---|---|---|---|
| B1 | MegaMoE EP16 T16384 mxfp8 and nvfp4 (P3-16/P9); EP16 T8192 mxfp8 (P4-2; P1: nvfp4 `megamoe-049`). P6-16 dedup rows `megamoe-915` (P0) / `-916` (P1) are `decomposed` from the T8192 twins (§1.2) | megamoe-010/014/041 | eager timing (above the capture cap); FI and vendored; target `t_ach_fused_us` |
| B2 | Big GEMMs: KDA in_proj N49376 (DEP, replicated) and N24752 (TP2) bf16 + fp8; 16384x12288x7168; 8192x12288x7168; o_proj K12288/6144 | gemm-003/005/021/022/024/027/029/037 | 1 node; each 0.4-3.2 ms, so few reps |
| B3 | Comm: AR N8 235 MB (P1, TP8 per-layer AR, the P1 bottleneck). P12 N16 1.76 GB / 0.88 GB / 0.57 GB (`comm-001/015/914`) take the MegaMoE d | comm-011 | 2 nodes |
| B4 | Indexer DSA-DP owner, 8192 rows x 75.6K keys (P4-2, P6-16) | dsa_indexer_topk-042 | new `idx_owner.py` |
| B5 | P1 tier: KDA prefill 48/96/12 heads (`kda-051/077/132`), indexer DEP / DSA-CP (`-080/081`), sparse MLA dsadp (`sparse_mla-094`), EP8 MegaMoE T2048 (`megamoe-133`, P1) | P1 ids | 1-2 nodes |
| B6 | Prefill step at DEP16 (P3-16) and TP8/DCP8 (P1 sanity) | step | new pf cfg `$MB/cfg/dep16.cfg.sh` from `pf_cfg.py` on a rendered DEP16 prefill yaml; P1 uses `$W/p1/tools/bench_prefill_step.py` |

P9 (DSA-CP: DEP everywhere, ShardQ-DCP8 on the 24 DSA layers) and P6-16 (TP2 KDA + DSA-DP, EP16 + dedup) have no vLLM
implementation. Their steps are compositions of the B rows and the stage A DCP rows only. They are labelled
model+microbench and get no step check.

### Stage C: EP16 decode candidates D4-16, D5, D7-a-16, D8-16 and D1 (4 nodes, replaces the EP64 stage)

674 in-scope cases: 29 P0 and 131 P1. Most P0 rows are shared with stage A: MegaMoE EP16 T4/32/128 (`megamoe-006..033`)
covers D4-16/D5/D7-a-16/D8-16 too, and the P11/P12 rows ride on them. New P0 rows:

- AR N16 for D1/D8-16: comm-031/045 (P1: comm-102).
- GEMM: gemm-038/048 (D4-16 replicated KDA in_proj 49376 at M 4/32), gemm-046/055 (o_proj K12288 at M 4/32).
- D1 TP16 expert slice: megamoe-083/086 (P1: -152/163/229/233).

P1 rows to run in the same pass:
- KDA decode at 96/48/12/6 heads: kda-064/149/082/185/099/175/146/196 (§1.4 heads override);
- indexer DSA-DP owner decode: dsa_indexer_topk-087/181;
- DCP16 key shard: dsa_indexer_topk-297;
- sparse MLA dsadp: sparse_mla-167/272;
- RS/AG N8 for D7-8-16: comm-110/111.

**New multi-GPU harness piece: heterogeneous group layer (C1).** D7-a-16 uses TP_a attention groups (a = 2/4/8)
under one EP16 MoE group over the same 16 GPUs, with TP-sharded MoE-dense (router/latent col->row and row->col
transposes). D8-16 uses TP16=DCP16 attention plus TP16 MoE-dense plus EP16. vLLM has neither layout: one process group
cannot carry TP_a attention and EP16 MoE together.
- New `harness/d7_layer.py` builds one KDA layer and one DSA layer: the attention part at TP_a (or TP16+DCP16) with
  real collectives (RS/AG N=a, AR N16, transposes), followed by the EP16 MegaMoE (vendored and FI final), all in one
  16-rank job on 4 nodes.
- It times attention, MoE and the boundary resharding separately and together, to check overlap and boundary cost
  against the model's per-layer timeline (`per_layer_ach` in optables).
- The attention modules are reused from `bench_layer.py` (`--tp a` real multi-rank, not single-GPU emulation).

D5 (TP4 DSA-DP + SP, EP16; PR #23) has no wpg2/kb layout either. Its layer check uses `d7_layer.py` with a = 4 and
replicated MoE-dense (D5's MoE-dense is token-sharded, not TP-sharded).

### Stage D: per-layer and full-step trend match against `vllm bench forward`

For every topology vLLM can run with an existing step harness, at r in {1, 8, 32} and both ckpts:
- decode: D1 (`tp16ls`), D3 (`kb_run1` c512 / `d4x4`), D4-16 (`dep16`);
- prefill: P1, P2 and P3-16.

1. Full step: kb_run1 / nq_step for decode, pf_run1 for prefill. `--exclude-drafter` gives the target-only split.
2. Per layer type: `wpg2/step` with `K31W0_LAYERS` = 2, 3, 4 and 8 (the §1.6 differences).
3. A profile pass mapped through `case2kernels.py` onto op rows.

**Topologies without a vLLM step path** (D5, D7-a-16, D8-16, P6-16, P9) are compared at layer level only, against
`d7_layer.py` (C1). Their step is a model composition and is flagged as such.

`vllm bench forward` caveats to carry into the comparison:
- Dummy weights distort routing. Use `--routing random` to match the MegaMoE benches.
- The large prefill runs eager.
- DP is uniform only.
- The LM head, rejection and CPU are excluded from nq_step. The model's `F.*` rows are left out of the comparison
  to match.

---

## 4. Node-hour estimate (4-GPU VR200 nodes)

Basis:
- 10-15 min startup per step bench (dummy weights, graph capture) and about 5 min per op-bench process (JIT is warm
  after the first).
- 30% retry and identity-check margin.
- These estimates assume the hold already exists. Hold wait is not counted. Rows the ops1 lane already measured are
  not counted.

| stage | content | nodes x hours | node-h |
|---|---|---|---|
| A | A0: 3 decode points x 2 ckpts with profile (4 n x 3 h) and P2 prefill 2 ckpts (4 n x 2 h); A1/A2 MegaMoE FI and vendored, 2 quant kinds (4 n x 2.5 h); A3 comm (1 n x 0.5 h); A4-A6 single-GPU (1 n x 3 h) | | **~34** (x1.3 = 44) |
| B | B1 (4 n x 2 h), B2/B4/B5 (1 n x 3 h), B3 (2 n x 0.5 h), B6 DEP16 step + P1 (4 n x 2.5 h + 2 n x 1 h) | | **~24** (x1.3 = 31) |
| C | C1 heterogeneous layer bring-up and correctness (4 n x 3 h), C1 timing D7-2/4/8-16 + D8-16 + D5, 3 r x 2 ckpts (4 n x 3 h), D1 slice + KDA/indexer/MLA heads rows (1 n x 3 h), AR N16 (4 n x 0.5 h) | | **~29** (x1.3 = 38) |
| D | 6 topologies x 2 ckpts, step + 4 truncated steps + profile; on average 4 n x 0.6 h per (topology, ckpt) triple of r | | **~29** (x1.3 = 38) |
| **total** | | | **~150 node-h** |

- Every stage now fits one 4-node same-segment hold (the holdd21 2x8n pattern provides it). No 16-node hold is
  needed.
- Wall clock with one 4-node hold: about 11 h for A, 8 h for B, 10 h for C and 10 h for D.

---

## 5. Derating bookkeeping

**Raw rows.** `$MB/rows/<family>.jsonl`, mirrored locally to `microbench/rows/`. Each line:

```json
{"case_id": "megamoe-006", "family": "megamoe", "shape": {...matrix shape...}, "dtype": "fp4",
 "ckpt": "nvfp4_modelopt", "impl": "fi-6e438852b|vendored-<commit>|cublas|...",
 "impl_ident": "MMTREE=fda3fe08... quant_kind=nvfp4 ...", "harness": "harness/run_b10.sh@<md5>",
 "timing": "graph|eager", "rot_bytes": 4294967296, "ranks": 16, "agg": "rank-max",
 "t_meas_us": {"p05": 0, "med": 0, "p95": 0}, "split_us": {"dispatch": 0, "fc1": 0, "act": 0, "fc2": 0, "combine": 0},
 "t_sol_us": 47.878, "t_ach_us": 75.695, "t_ach_kernel_us": 76.884, "t_ach_fused_us": 90.474,
 "derate": 0.0, "status": "ok|error|unsupported|decomposed", "job": 0, "hosts": "...", "date": "2026-11-xx"}
```

`derate = t_meas.med / t_ach_fused_us` for a whole-kernel MegaMoE row, and `t_meas.med / t_ach_kernel_us` for
everything else.

**Derating table.** This is the model input, written only after the user OKs the results, to
`perf-model/inputs/derating.json`. It is produced by a new `microbench/fold_derating.py`.

- One entry per optables op row key `(ckpt, topology, regime, op id)`, obtained by expanding each case through
  `cases[].uses`:
  `{"d": 1.37, "src": "measured|interp|family_median|prior", "case_id": "gemm-026", "impl": "...", "n_rows": 1}`.
- **interp:** linear in log(M) or log(tokens) between two measured cases of the same (family, N, K / EP, dtype). No
  extrapolation beyond 2x the measured range.
- **family_median:** the median d of measured cases in the same family and phase. Used only for P2-tier rows.
- **prior:** the step-level prior from `matrix.json:gap_priors`. This is the last resort and is reported as
  coverage debt.
- Comm rows are keyed by `(primitive, N, bytes)` and take the derating of the `chosen_ach` candidate. The
  best-candidate derating is stored as `d_best`. P11/P12 rows of EP topologies take the MegaMoE whole-kernel d of
  the same (EP, tokens/rank, act), applied to MOE.dispatch..combine and to the P11/P12 events together.
- MegaMoE rows store both `d_vendored` (used for the stage D comparison) and `d_fi_final` (used for projections that
  assume the FI final build ships).
- The model reads it as `t_eff = t_ach * d`. `render.py` and `out/*.json` gain a `t_eff` column and a `d_src`
  column, and nothing else changes.
- A coverage line per regime gives the share of step t_ach whose d comes from `measured`. **Stage acceptance
  requires at least 80%.**

**Acceptance criteria.**

1. **Per op.**
   - `0.9 <= d <= 2.0` is accepted.
   - `d > 2.0` goes on the **kernel-debug list** (`microbench/DEBUG_LIST.md`, generated) with the case id, split_us
     and the bound the model assumed. Examples: a MegaMoE row whose split shows combine above 2x, or a GEMM at M=4
     running far from its weight-stream time.
   - `d < 0.9` means the model's achievable figure is too pessimistic. That is a model bug: fix `hw.json`/`cost.py`
     and do not clamp. Achievable is meant to be an optimistic upper bound.
2. **Per layer.** The sum of derated op times for one KDA+MoE layer and one DSA+MoE layer is each within ±10% of
   the matching truncated-step difference (`step(3) - step(2)`, `step(4) - step(3)`, §1.6). The 3:1 mean
   `(step(8) - step(4)) / 4` is the cross-check. The residual is split into host gaps (from pf_breakdown/kb_parse
   idle) and overlap, and both are reported.
3. **Full step.** The derated model step is within ±15% of the `vllm bench forward` step for D1, D3, D4-16, P1, P2
   and P3-16. **The known gaps must be explained.** For each of D3 (1.71-1.94x), P2 (1.74x) and D1 (1.06-1.67x),
   named op rows must account for at least 80% of `(measured step - achievable step)`, with the top 5 contributors
   listed.
4. **Trend alignment (primary).**
   - The topology ranking by derated model step (and tok/s/GPU at the proposals.md screening r points) must equal
     the ranking by measured step for every regime that has two or more measured topologies.
   - Pairs whose measured steps are within 5% count as ties and do not break alignment. The EP <= 16 mid/high bands
     are model near-ties (often < 2%, `results/model/README.md`), so the derated ranking decides there.
   - Any rank flip is a finding. Attribute it to the op rows that cause it before stage C results are used to pick
     D7-a-16/D8-16 configs.
   - For D5, D7-a-16, D8-16, P6-16 and P9 (no step), report how the projected ranking against D3/D4-16/P2/P3-16 moves
     from the underated to the derated model.
5. **Re-rank.** After stage A, rerun `build_matrix.py` with measured per-op d in place of the step-level
   `gap_priors`. Ids do not change (ids_v1.json). Promote any P2 case whose new score enters the top 60% before
   starting stage B.

---

## 6. Open items before launch (need user OK or input)

- **Scope confirmation.** This plan follows the 2026-11-18 EP <= 16 rescope (`topologies.SCOPE`). If EP64 (D7-a,
  D8, P6) or EP32 (D4-32) come back, the OOS rows are already in the matrix. They would need a 16-node same-segment
  hold and an FI world-size relaxation, which is scratch kernel bring-up work.
- **P6-16 dedup.** FI final has no O2 dispatch dedup (L4 lane dropped). Accept `decomposed` dedup rows, or descope
  O2 for P6-16?
- **D5 / D7-a-16 / D8-16 have no vLLM step path.** Is a layer-level check through `d7_layer.py` enough, or should a
  wpg2 layout be added (out of this lane: it touches vLLM code)?
- **KDA decode heads.** bench_kda is TP16-only. Approve the `kda_decode.py` copy with a heads override, or use the
  ops1 `attn_bench.py kda` rows?
- JHB opbreak/sol2 harnesses were not inspected (JHB unreachable). Check whether they add shard-parameterized benches
  before writing `comm_cases.py`.
- The `nvfp4_modelopt` ckpt rows use `--quant-kind nvfp4` throughout. Confirm that the shared experts and KDA FP8
  GEMMs (`gemm-030/022/027/113`) use MXFP8 with the backend the ckpt actually ships.
- The ops1 lane's cases_ops1.json (682 v1 ids) includes now-OOS rows (P6 EP64, D4-32, D7-a, D8). Their rows are
  harmless but are not counted.
