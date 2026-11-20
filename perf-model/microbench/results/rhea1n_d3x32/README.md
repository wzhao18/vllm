# rhea1n_d3x32: kernel-level gap of D3-32 decode, reproduced on one Rhea node (2026-11-18)

**Goal.** Reproduce, kernel by kernel, the gap between the perf model's achievable step and the measured step for D3-32
decode. D3-32 is hybrid DCP4 x DP8, EP32, with the sequence-parallel (SP) dense path, ShardQ DSA and k=3. The code is
the latest branch `wzhao/k31-sp-gemm` 4fbaa15189 (release 1767cf7 plus the SP GEMM tactic table), with the release
tree as an A/B arm. Hardware is one Rhea VR200 node (4 GPUs).

**Result.** One DP rank of D3-32, measured on one node and scaled to 93 layers, reproduces the 8-node D3-32 step within
2-4% at every r. The gap to the model is about 2.1-2.3x achievable, and its composition matches the earlier
EP16-transferred estimate. The difference is that DCP top-k is larger and the dense-GEMM gap is smaller on the branch.

## What ran

| item | value |
|---|---|
| hold | Rhea job 72377: 1 node rhea0145, account coreai_cserve_dev (best FairShare, 0.42), partition backfill. Ran 20 min, cancelled when done. |
| image | `images/kimi-k31-wzhao-nov-11-aarch64.sqsh`. vLLM base 242fb49cd4, FlashInfer 0.7.3.dev20261031 (`/opt/k3/fi-*`), deep_gemm from the image tree. |
| code under test | `k31gap/ov/spgemm-4fbaa15`: every git-tracked `vllm/` file of 4fbaa15189 shipped over the image's `/opt/k3/ov-d`. The build artifacts are kept (`*.so`, `third_party/deep_gemm`, flashmla interface, `_version.py`). The 3 files deleted vs 242fb49cd4 are removed. There are no csrc changes vs 242fb49cd4. The overlay log line `NQ: overlay: vllm from .../spgemm-4fbaa15 rev 4fbaa15189` confirms the tree. |
| A/B arm | `k31gap/ov/rel-1767cf7`: the release tree, built the same way |
| harness | nq step bench, Rhea 1-node variant in `perf-model/microbench/rhea1n/`: `cfg2bench.py` with a 4-rank world and Rhea paths, plus `nq_run1.sh`, `nq_inner.sh`, `nq_shapes.py` and `nq_step.py` (unchanged). `vllm bench forward` takes the CUDA-graph replay median of the max over ranks, then a torch profile of 3 replays on all 4 ranks. |
| decode role | the release C512 decode role (`results/k31vr-release-draft-sweep-20261117` C512-a1118813 decode command), changed as follows: data-parallel-size 4 -> 1; moe-backend kimi_k3_mega_moe -> deep_gemm_mega_moe (the measured D3-32 / gap512 MoE). Kept: latency_specialized, TP4 = DCP4 a2a, interleave 1, fp8 KV, mxfp4 indexer, deep_select, FP32 SSM, BF16 linears, MXFP8 MoE activations, trtllm sparse MLA, use-replayssm, max-num-seqs 128, capture to 512. Config: `rhea1n/rhea_d3x32_dp1.json`. |
| shapes | B = 32 / 64 / 96 per DP rank (verify rows 128 / 256 / 384), so r = 8 / 16 / 24 requests per GPU. Uniform context u132k = 135,168 tokens (the model's L is 135K). Dummy weights, balanced routing. |
| analysis | `rhea1n/gap_parse.py` (gap512 method: segment at AttnRes, same-stream PDL clip, exposed = additive share of the busy timeline), `rescale_rows.py`, `project93.py`, `summarize.py`. |

## Deviations from real D3-32, and how each is handled

1. **24-layer view instead of 93.** At EP4, 896 experts do not fit in 93 layers (224 experts per GPU), and the
   specialized decode hard-codes the 896-expert MoE front, so 112 experts cannot emulate EP32's 28 per GPU.
   - The view `k31gap/models/k31-view-L24` has 18 KDA + 6 DSA layers, 23 MoE + 1 dense, and AttnRes block 12. That
     keeps the same 3:1 KDA:DSA pattern.
   - Each op group's measured time is scaled to 93 layers by its layer kind: KDA x 69/18, DSA x 4, MoE sublayer x 4,
     AttnRes x 187/49, SP comm x 93/24.
   - Model rows are compared on the same layer counts (`model_rows_L24.json`) and on the full 93-layer model
     (`model_rows.json`, D3-32 at B 32/64/96, L 135K).
2. **MegaMoE is substituted.** The 1-node MegaMoE is EP4 with 224 experts per GPU: 335 µs per layer measured, about 8x
   the EP32 expert weight stream. It is replaced by the FlashInfer SM107 EP32 MX standalone row (`moe_ep32`, T = B
   tokens per rank) x 92 x 1.1. The range is x1.0 (FlashInfer) to x1.2 (the vendored in-step kernel). Routes per GPU
   (4T) match EP32, but expert count per GPU does not.
3. **Drafter tail is substituted.** The release GQA drafter is not on Rhea. The bench ran the DSpark preview (5 MLA
   layers, run as the stock drafter), which measured about 4.1 ms. It is replaced by gap512's measured
   release-drafter tail at the nearest B.
4. **DP lock-step and cross-node EP traffic are not present.** They are one GPU group of the 8, all NVLink in one
   node. The multi-node agreement below says these cost little at these r.
5. **The 8-node reference used agentx contexts and the drafter.** Its step includes the drafter; this run uses a
   uniform 135K context.

## Step: measured vs model (ms per decode step; bench scope = no LM head, rejection or CPU)

| r | 24-layer bench step (branch / release) | projected 93-layer step (branch / release) | 8-node D3-32 measured (wave2) | model achievable | measured / achievable | model SOL |
|---|---|---|---|---|---|---|
| 8 | 12.25 / 12.41 | **20.42** / 21.10 | 20.92 | 9.81 | 2.08 | 6.85 |
| 16 | 15.21 / 15.49 | **25.90** / 27.00 | 26.98 | 11.34 | 2.28 | 8.18 |
| 24 | 17.34 / 17.43 | **29.78** / 30.17 | 30.48 | 13.38 | 2.23 | 9.83 |

## Per-family gap at r=24, branch, 93-layer projection (ms per step)

| family | busy | exposed | achievable | derate (busy / achievable) | gap (exposed - achievable) |
|---|---|---|---|---|---|
| BF16 dense GEMMs (cuBLAS nvjet + CuTe split-K + split-K reduce) | 9.41 | 8.25 | 4.74 | 1.99 | **3.51** |
| DSA DCP exact top-k (pp hist/window/win + DCP merge) | 2.59 | 2.59 | 0.23 | 11.4 | **2.36** |
| MegaMoE (substituted: FlashInfer EP32 x1.1; dispatch + experts + combine) | 7.70 | 7.69 | 5.35 | 1.44 | **2.33** (1.6-3.1) |
| DSA q path + sparse MLA + DCP value | 3.95 | 2.68 | 1.00 | 3.97 | **1.68** |
| KDA core (KdaWS at B96) | 3.40 | 3.40 | 1.81 | 1.88 | **1.59** |
| Small elementwise (AttnRes mix, SiTU, latent RMSNorm, router top-k, MoE prepare) | 1.53 | 1.28 | 0.15 | 9.95 | **1.13** |
| SP comm (all-gather / reduce-scatter push) | 2.06 | 2.06 | 1.28 | 1.61 | 0.78 |
| Drafter + final (substituted: gap512 release) | 0.89 | 0.89 | 0.28 | 3.19 | 0.61 |
| DSA indexer logits (paged_mqa_logits) | 1.63 | 0.94 | 0.43 | 3.83 | 0.52 |
| **step** | | **29.78** | **13.38** | | **16.4** |

The tables for r8 and r16 are in `parsed/spgemm_93.txt`. The release arm is in `parsed/rel_93.txt`. The 24-layer
per-group tables, with the model on the same layer counts, are in `parsed/*_groups.txt`.

## Top kernels at r=24 (24-layer view; per-call time = total / calls)

| group | kernel | per call | calls | derate |
|---|---|---|---|---|
| KDA core | `kdaspecialized KdaWS` | 49 µs / layer | 18 | 1.88 |
| DSA top-k merge | `k31dcpx::topk_merge_kernel<16>` | 60 µs / DSA layer | 6 | 8.1 |
| DSA top-k pre-passes | `topk_pp_win` 25, `topk_pp_hist` 17, `topk_pp_window` 4.5 µs / DSA layer | | 18 | 24 |
| DSA q path | `mlaquery MlaQuery` | 48 µs | 12 | 4.4 |
| sparse MLA | `fmhaSm107a...QkvE4m3...HQk576HV512` | 16 µs | 12 | 3.5 |
| indexer logits | `deep_gemm::sm100_paged_mqa_logits` | 69 µs / DSA layer | 6 | 3.8 |
| shared gate_up | `gemmsplitk SplitKGemm` (CuTe, SP table) | 20 µs | 23 | 1.5 |
| KDA in_proj | `nvjet_sm107_tst_128x192 ... 2cta` | 32 µs | 18 | 1.42 |
| router + latent_down | `nvjet 64x64 ... 2cta` + splitK nvjet + reduce | 4.8 µs x 3 / layer | 69 | 4.05 |
| SP reduce-scatter | `sp_comm SpReduceScatter` | 9 µs | 25 | 2.1 |
| shared SiTU | `vllm::situ_and_mul_kernel` | 3.7 µs | 23 | 69 |
| MegaMoE (EP4, not used) | `deep_gemm::sm100_fp8_fp4_mega_moe_situ_impl` | 334 µs | 23 | n/a |

## SP GEMM tactic table (branch) vs release tree

| r | 93-layer step, release -> branch | BF16 GEMM family exposed | notes |
|---|---|---|---|
| 8 | 21.10 -> 20.42 ms (-3.2%) | 6.96 -> 6.14 | shared gate_up / down and DSA g_proj faster; DSA q path exposure +0.13 (overlap shifts to the main stream) |
| 16 | 27.00 -> 25.90 ms (-4.1%) | 9.07 -> 7.33 | g_proj 534 -> 95 µs, latent_up -0.44 ms; q path exposure +0.52 |
| 24 | 30.17 -> 29.78 ms (-1.3%) | 8.99 -> 8.25 | top-k + merge exposure +0.27 and q path +0.10 eat part of the GEMM gain |

On the 24-layer bench the step drops by 1.3%, 1.8% and 0.5%. Part of the GEMM gain moves into exposed DSA side-stream
time: the side-stream work is no longer hidden.

## Comparison with the earlier (EP16-transferred) r24 estimate

| family | earlier estimate (gap512 D3 EP16 profile, MoE row moved to EP32) | this run (D3-32 slice, branch) |
|---|---|---|
| BF16 dense GEMMs | 4.2 | 3.51 (the SP table removes about 0.7; release arm 4.25) |
| MegaMoE | about 3.4 | 2.33 (substituted, range 1.6-3.1) |
| DSA exact top-k + merge | 2.0 | **2.36** |
| DSA q path / sparse MLA / DCP value | 1.7 | 1.68 |
| KDA core | 1.6 | 1.59 |
| small ops | 1.2 | 1.13 |
| SP comm | 0.5 | 0.78 |
| indexer logits | 0.3 | 0.52 |

## Files

- `rows.json`:
  - `kind` step / family / group on the 24-layer view, tags spgemm-24L and rel-24L;
  - `kind` step93 / family93 for the 93-layer projection, tags spgemm-93L and rel-93L.
- `parsed/`: gap_parse JSON and text per arm (24 layers), and the 93-layer projections.
- `raw/`:
  - `spgemm-prof/`, `rel-prof/`: bench logs, cfg.sh, shapes, and the torch traces of 4 ranks x 3 cases;
  - `smoke2/`: the first successful run.
- Harness and analysis: `perf-model/microbench/rhea1n/`. On Rhea: `/lustre/fsw/coreai_comparch_inferencex/weizha/k31gap/`
  (tools, ov, models, runs).
