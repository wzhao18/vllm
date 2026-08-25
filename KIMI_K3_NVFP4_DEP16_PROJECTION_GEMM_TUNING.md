# Kimi-K3 NVFP4 DEP16 projection GEMM tuning

Last updated: 2026-08-25

## Objective

Improve projection GEMM performance for the TP1/DP16/EP16 decode shape using
standalone microbenchmarks, followed by end-to-end performance and accuracy
validation. The microbenchmarks do not reduce the number of SMs available to
a kernel.

The authoritative shape inventory comes from the 100-replay GPU-0 trace in:

```text
srt-slurm/outputs/
  kimi-k3-nvfp4-dep16-dspark4-megamoe-shared-fc12-dep-gemm-ac60-fp8-nsys/
  logs/profiles/agg/
  nvl72d148-T01_agg_w0_profile_gpu0-1-2-3.sqlite
```

The original target-verification shape was approximately `M=20`: four local
sequences and five target-verification positions for DSpark K=4. The final
search covers every integer `M` from 1 through 64; representative points use
real checkpoint weights and the full range uses cache-faithful weight rings.

## Projection inventory

### FP8_PB_WO attention projections

These projections dynamically quantize BF16 activations per token/per 128
elements to E4M3. Checkpoint weights are E4M3 with static 128x128 block scales,
and the GEMM returns BF16. Physical `N` includes ModelOpt's output padding to a
multiple of 128.

| Projection | Calls/step | M | Physical N | K | Current trace activity/call |
| --- | ---: | ---: | ---: | ---: | ---: |
| KDA `in_proj_qkvgfab` | 69 | 20 | 49408 | 7168 | 56.25 us |
| KDA `f_b_proj` | 69 | 20 | 12288 | 128 | 4.14 us |
| Shared KDA/MLA `o_proj` | 93 | 20 | 7168 | 12288 | 20-23 us |
| MLA fused Q/KV-A | 24 | 20 | 2176 | 7168 | 14.37 us |
| MLA Q-B | 24 | 20 | 18432 | 1536 | 8.58 us |
| MLA output gate | 24 | 20 | 12288 | 7168 | 21.63 us |

All six shapes use the same `sm100_fp8_fp4_gemm_1d1d_impl` symbol in Nsight,
but their operands are FP8, not NVFP4. Aggregate GEMM activity is 7.327 ms per
target step across 303 calls. Dynamic activation quantization is additional
work and will be measured both separately and together with each GEMM.

The starting branch constrained DeepGEMM's block-M candidate to multiples of
256 for `(N,K)=(12288,128)` and `(7168,12288)` at M 17 through 256. The final
implementation replaces the narrow KDA `f_b` path with a fused kernel and
extends shape-specific DeepGEMM plans to the useful Q-B and QKV-A ranges. KDA
input and MLA output-gate projections retain unconstrained DeepGEMM because no
stable improvement survived cache-faithful replay.

### BF16 projections

| Projection | Calls/step | M | N | K | Current trace activity/call |
| --- | ---: | ---: | ---: | ---: | ---: |
| MoE router | 92 | 20 | 896 | 7168 | 9.54 us |
| Routed latent down | 92 | 20 | 3584 | 7168 | 16.05 us |
| Routed latent up + shared add | 92 | 20 | 7168 | 3584 | 11.09 us |
| Layer-0 dense gate/up | 1 | 20 | 67584 | 7168 | 165.41 us |
| Layer-0 dense down | 1 | 20 | 7168 | 33792 | 76.11 us plus 3.11 us reduction |
| Target LM head | 1 | 20 | 163840 | 7168 | 368.50 us |

The latent-down and router GEMMs run concurrently on separate streams in the
model. The microbenchmarks measured their isolated kernel latency without
artificially restricting SMs. The routed-up benchmark includes the BF16
beta/add epilogue used in the model.

### MLA absorbed BMMs

The 24 MLA layers also contain absorbed W-UK and W-UV BF16 batched GEMMs:

| Operation | Calls/step | Aggregate activity |
| --- | ---: | ---: |
| W-UK absorbed BMM | 24 | 0.140 ms |
| W-UV absorbed BMM | 24 | 0.116 ms |

They are lower priority than the dense projections but remain in scope after
their exact batched layouts are reproduced.

The exact TP1 layouts are 96-head BF16 batches: W-UK is
`[96,M,128] x [96,128,512]`, and W-UV is
`[96,M,512] x [96,512,128]`. A cache-faithful microbenchmark rotates through
24 unique layer weights per replay instead of repeatedly benchmarking one
6-MiB weight resident in L2. At M=20, PyTorch's nvJet BMM path takes 111.9 us
for all 24 W-UK projections and 100.8 us for all 24 W-UV projections. The
trace records 140.2 and 117.4 us respectively, with the difference plausibly
coming from model-wide cache pressure absent from the isolated replay.

A parameter sweep of a conventional Triton tensor-core BMM did not improve
either operation. Its best tested W-UK configuration takes 132.9 us (18.8%
slower), while W-UV takes at least 227.2 us (125% slower). The current nvJet
path is retained. Raw data is in `mla-absorbed-bmm-cold-weight-paired.json`.

### DSpark draft and sampling projections

The target graph is not the complete decode cadence. The separate DSpark graph
takes 1.893 ms and contains another set of replicated TP1 BF16 projections.
The checkpoint headers and implementation give the following complete draft
inventory:

| Projection | Calls/cadence | N | K |
| --- | ---: | ---: | ---: |
| Target-context combine | 1 | 7168 | 35840 |
| Fused cross-layer context KV | 1 | 2880 | 7168 |
| MLA fused Q/KV-A | 5 | 2112 | 7168 |
| MLA Q-B | 5 | 12288 | 1536 |
| MLA output | 5 | 7168 | 8192 |
| Dense gate/up | 5 | 28672 | 7168 |
| Dense down | 5 | 7168 | 14336 |
| Target LM head | 1 | 163840 | 7168 |
| Markov W2 | 4 | 163840 | 256 |

The source trace contains two distinct full-vocabulary LM-head calls per decode
cadence: 368.50 us in target verification at local M=20 and 366.32 us in the
DSpark graph at local M=4. The draft graph contains about 0.75 ms of additional
repeated nvJet projection/MLP work. These shapes are screened separately over
M=1-64 so target-only improvements are not mistaken for an exhaustive
full-cadence projection audit. The scalar confidence head is inactive in this
workload and is not a material GEMM.

Five-weight cache-faithful replay confirms that the existing fused-A selector
for `(N,K)=(2112,7168)` is highly effective once it is installed on the DSpark
model: it takes 5.51-6.40 us at every M=1-16 versus 8.20-8.94 us for nvJet, a
32.3-48.8% reduction. At M=17-64 the selector falls back to nvJet and differs
by less than 0.1%. The production gap was therefore model installation, not a
new QKV kernel or a missing per-M policy.

The final DSpark policies use the same five-weight replay for projections
repeated in each of the five draft layers. One-weight replay is retained only
for the unique context projections and the shared LM head.

| Projection | Selected policy over M=1-64 | Improvement over prior selector |
| --- | --- | ---: |
| Context combine `(7168,35840)` | static-K CuTe M=1-5; torch M=6-64 | 3.1-6.0% at M=1-5 |
| Context KV `(2880,7168)` | static-K CuTe M=1-5; torch M=6-64 | 7.8-15.3% at M=1-5 with cold weights |
| QKV-A `(2112,7168)` | static-K CuTe M=1-2; fused-A M=3-16; torch M=17-64 | 25.9% at M=1; 10.0% at M=2 |
| Q-B `(12288,1536)` | static-K CuTe M=1-3; torch M=4-64 | 6.1-19.8% at M=1-3 |
| MLA output `(7168,8192)` | static-K CuTe M=1-4; torch M=5-64 | 4.7-10.2% at M=1-4 |
| Dense gate/up `(28672,7168)` | CuTe M=1-3; DeepGEMM M=4-64 | 5.3-8.2% over DeepGEMM at M=1-3 |
| Dense down `(7168,14336)` | static-K CuTe M=1-4; torch M=5-64 | 1.5-3.3% at M=1-4 |
| LM head `(163840,7168)` | CuTe M=1-4; DeepGEMM default/BM64/default at M=5-9/10-48/49-64 | 7.2-12.9% versus torch at M=1-64 |
| Markov W2 `(163840,256)` | DeepGEMM M=43-48; torch otherwise | 2.3-3.4% at M=43-48 |

The original shape-dynamic QKV crossover was a useful negative result. With
one repeatedly cached weight it appeared faster at M=1-3, but five distinct
layer weights showed fused-A winning. Compile-time K changes that result: the
fully unrolled CuTe kernel wins by 25.9% and 10.0% at M=1 and M=2 even with a
five-weight ring. Fused-A remains faster from M=3 through M=16. Raw data is in
`draft-projection-static-k-draft_mla_fused_qkv_a-ring5-563717-group0.json`.

The remaining raw DSpark measurements use the
`draft-projection-skinny-*`, `draft-projection-selected-all-m1-64-*`, and
`draft-projection-selected-*-ring*` files under
`/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/weizha/tmp/dep16-gemm-microbench`.

## Measurement protocol

1. Run on one exclusively allocated GB300 node.
2. Use CUDA-graph replay timings to match the model's captured decode path.
3. Report median, p10, and p90 over repeated batches of graph replays.
4. Keep fixed input/output allocations and preloaded weights.
5. For FP8, report prequantized GEMM latency, activation quantization latency,
   and the combined projection latency separately.
6. Compare numerical output against the current backend before accepting a
   candidate.
7. Sweep full-kernel tile/layout choices and alternate backends. Do not tune by
   withholding SMs from a kernel.

## Investigation log

### 2026-08-24: trace and implementation inventory

The trace counts exactly match the model architecture: 69 KDA layers, 24 MLA
layers, and 92 MoE layers. The six unique attention projection shapes account
for all 303 FP8 GEMMs in the target CUDA graph. Five BF16 projection shapes
account for the material non-expert dense GEMM work.

The branch already contains one targeted DeepGEMM layout optimization, but it
only covers two FP8 shapes. Its use of block-M candidate constraints should be
re-swept across all six TP1 shapes at M=20 and neighboring M values rather than
assumed globally optimal.

Completed measurement matrix:

1. DeepGEMM default versus block-M/block-N candidate constraints.
2. CUTLASS and Triton block-FP8 baselines.
3. cuBLASLt, the existing fused-A kernel, and CuTe skinny GEMM for all BF16
   shapes at M=20.
4. Full configuration sweeps and cold-weight follow-ups for every shape where
   the initial winner was not clearly separated from the alternatives.

## Current production recommendations

| Projection | Production change | Target saving |
| --- | --- | ---: |
| KDA `f_b`, 69 calls | Fused K128 activation quantization + FP8 GEMM | 1.4-2.0x vs tuned Triton |
| KDA output, 69 calls | Fuse gated RMSNorm with packed FP8 quantization | 0.167-0.194 ms/step |
| Shared KDA/MLA `o_proj`, 93 calls | DeepGEMM BM256 for M=11-256 | 1.4-20.5% per call over default in M=11-64 |
| MLA Q-B, 24 calls | DeepGEMM BM256 at M=17-64 | 5 us/step |
| MLA QKV-A + gate, 24 pairs | Reuse input quantization and constrain QKV to BM256/BN64 | 16.4-23.9% per pair |
| MoE router, 92 calls | Shape-specific CuTe split-K plans | 2-21% at selected M>16; 4-5% over prior CuTe at M=5-16 |
| Dense gate/up, 1 call | Static-K CuTe M=1-4; DeepGEMM M=5 and M=17-64 | 3.6-10.4% at M=1-4; 10.3-11.1% at high M |
| Latent projections | Static-K CuTe at routed-down M=1-5 and routed-up M=1-4 | 5.6-21.4% over the prior selector |
| Dense down | Static-K CuTe at M=1-4; torch M=5-64 | 6-17% over the prior selector at M=1-4 |
| MLA absorbed BMMs | Retain PyTorch/nvJet | custom Triton loses |

The table reports isolated projection savings. End-to-end cadence is validated
separately because several projection families overlap work on other streams.

### 2026-08-25: exhaustive M=1-64 audit

The final audit benchmarks all integer M values for the selected candidates,
with broader backend/configuration screens at M=1,2,4,8,16,32,48,64 and exact
crossover checks where a backend changes. Weight rings exceed L2 for the 69,
92, and 93-layer projection families. This avoids selecting kernels from a
single repeatedly cached weight.

#### FP8 projection results

The KDA `f_b` projection is the largest new opportunity. Its K dimension is
exactly one 128-element quantization group, so a new Triton kernel loads the
strided BF16 slice directly, computes its UE8M0 row scale, quantizes it, and
performs the FP8 dot product in one launch. It uses BM4/BN64/W4 for M=1-2,
BM16/BN64/W4 for M=3-63, and W2 at M=64. With all 69 real checkpoint weights,
it measures 2.36-3.58 us across M=1-64 versus 4.26-5.07 us for the previously
tuned two-launch Triton path and 6.00-9.94 us for DeepGEMM. CUDA graph capture
and arbitrary checkpoint-scale correctness both pass. Raw results:
`fb-ring69-w2-w4-all-m1-64-trials3.json` and
`fb-checkpoint-ring69-selected-m1-64.json`.

The other FP8 conclusions are:

- KDA output normalization and the packed UE8M0 quantization consumed by its
  `o_proj` are fused into one Triton launch. The production policy uses
  BR4/W2 at M=1-42 and M=51-63, and retains BR16/W8 at M=43-50 and M=64.
  It measures 2.83-3.99 us versus 5.55-6.44 us for the two-launch path across
  every integer M=1-64, a 37.8-49.7% reduction or 0.167-0.194 ms across 69
  KDA layers. BR4/W2 exactly matched the prior BF16-boundary output in 640
  checks covering all M values and ten random seeds; CUDA graph replay also
  passes. Raw data: `kda-output-fusion-config-screen-pdl-m1-64.json`,
  `kda-output-fusion-selected-pdl-all-m1-64.json`, and
  `kda-output-fusion-br4w2-correctness-seeds0-9-all-m1-64.json`.
- KDA input `(49408,7168)`: exhaustive DeepGEMM constraints at representative
  M values differ from default by at most 0.03%; retain default.
- Shared KDA/MLA output `(7168,12288)`: cache-faithful crossover is M=11.
  BM256 is 1.4% faster at M=11 and reaches 17.8-20.5% at M=32-64. The selected
  default/BM256 candidates were replayed at every integer M from 1 through 64.
- MLA Q-B `(18432,1536)`: BM256 wins throughout M=17-64, with 1.5-5.2%
  improvement and the largest gain at M=49-64.
- MLA QKV-A and output gate consume identical activation quantization. Sharing
  it and constraining QKV to BM256/BN64 improves the concurrently launched pair
  by 16.4-23.9% over separate quantization across every integer M from 1
  through 64. The constraint intentionally trades a negligible serial QKV
  difference for less contention with the gate GEMM.

#### BF16 projection results

The replicated `(896,7168)` router uses a cache-faithful ten-weight ring. The
SM103 CuTe split-K choices are `(5,4)` at M=5-11, `(5,5)` at M=12-16,
`(2,6)` at M=17-32, and `(3,3)` at M=41-47. Gaps M=33-40 and M=48-64 retain
torch because the specialized kernel loses there. Incremental improvement over
the previous CuTe default is about 4-5% at M=5-16; improvement over torch is
2-3% at M=17-24, about 20% at M=25-31 and M=41-47, and 2.5% at M=32. Raw data:
`router-ring10-selected-all-m1-64.json`.

The layer-0 dense gate/up projection has two sharp cuBLAS algorithm boundaries.
Static-K CuTe improves on DeepGEMM by 3.6-10.4% at M=1-4, unconstrained
DeepGEMM wins at M=5, and torch wins at M=6-16;
BM64/BN64 DeepGEMM wins by 10.3-11.1% at M=17-48; and unconstrained DeepGEMM
wins by 10.3-10.6% at M=49-64. Static-K CuTe improves dense down by about
6-17% at M=1-4; torch remains best at M=5-64. Raw data:
`dense-selected-all-m1-64.json`.

A final exhaustive DeepGEMM constraint search at M=1,5,17,32,48,49,64
confirms those coarse ranges. Default selection is tied with BM16 at M=1 and
M=5; BM64 wins at M=17 and M=32; BM32/BM64/default converge within 0.25% at
M=48-64. The production rule deliberately avoids per-M choices inside that
noise band. Raw files are `dense-gate-full-dg-m{M}.json`.

Ten-weight cache-faithful static-K replay further improves routed latent down
by 5.6-21.4% at M=1-5; fused-A remains selected at M=6-8. The residual-fused
latent-up projection improves by 6.0-9.0% at M=1-4. At all tested points
outside those ranges the installed selector falls back to torch. Raw data is
in the `draft-projection-static-k-routed_latent_*` files.

#### Compile-time K follow-up

The CuTe skinny kernel normally receives K as a symbolic dimension and loops
over split-K tiles at runtime. All Kimi decode projection K dimensions are
fixed by the model, so the follow-up supplied K at compilation and fully
unrolled that loop. The search covered block sizes 32-256, output groups
1/2/3/4/6/7/8, vector widths 4/8, and register caps 48-255. Every selected
window was measured against the already tuned production selector; M values
outside those windows retain the prior backend.

The register-cap sweep found only one selected exception to the existing
64-register default: draft dense-down M=1 uses 48 for a small repeatable gain.
The cold-weight context-KV winners use the default 64-register cap; caps from
48 through 255 either tied or regressed. The production table therefore avoids
shape-specific resource tuning where it is not material.

Context-KV required a separate cache audit. Its 39 MB BF16 weight fits in the
GPU L2, so repeating one allocation made the old dynamic kernel look tied or
faster. Rotating five weights exceeds L2 and better represents the intervening
model work between decode cadences. Under that pressure, static-K configurations
win by 9.1%, 15.3%, 7.8%, and 8.4% at M=1,2,3,4 respectively. The selected
tiles are BS256/OPB2/VW4 at M=1-2 and BS128/OPB8/VW8 at M=3-4. Raw data is in
`draft-projection-static-k-draft_context_kv_proj-ring5-564070-group0.json` and
`draft-projection-static-register-draft_context_kv_proj-ring5-564100-group0.json`.
The exhaustive crossover follow-up adds BS64/OPB4/VW8 at M=5 for an 8.3%
improvement; M=6-7 return to torch. A final cold five-weight search over every
static-K tile at M=8-16 found the best candidate 11.8% slower at M=8, with the
gap widening thereafter. The corresponding raw results are
`draft-projection-static-k-draft_context_kv_proj-ring5-564218-group0.json`.
The context-combine gap check likewise rejects static K at M=6-7 by 5.0% and
23.1% (`draft-projection-static-k-draft_context_proj-ring1-564202-group0.json`).

The final static-K dispatch table is:

| Projection `(N,K)` | M | BS / OPB / VW | Register cap |
| --- | --- | --- | ---: |
| DSpark QKV-A `(2112,7168)` | 1 / 2 | 224/3/8; 224/2/8 | 64 |
| Routed latent down `(3584,7168)` | 1 / 2-5 | 128/1/4; 64/4/8 | 64 |
| Routed latent up+add `(7168,3584)` | 1 / 2 / 3-4 | 64/2/4; 64/8/4; 32/4/8 | 64 |
| Dense gate/up `(67584,7168)` | 1 / 2-4 | 32/1/8; 64/4/4 | 64 |
| Dense down `(7168,33792)` | 1 / 2 / 3-4 | 192/1/8; 128/4/8; 128/4/4 | 64 |
| DSpark context `(7168,35840)` | 1 / 2-3 / 4-5 | 160/1/8; 128/4/8; 128/4/4 | 64 |
| DSpark context-KV `(2880,7168)` | 1-2 / 3-4 / 5 | 256/2/4; 128/8/8; 64/4/8 | 64 |
| DSpark Q-B `(12288,1536)` | 1 / 2-3 | 32/4/8; 32/8/8 | 64 |
| DSpark MLA output `(7168,8192)` | 1 / 2-4 | 128/1/4; 64/8/8 | 64 |
| DSpark dense down `(7168,14336)` | 1 / 2 / 3 / 4 | 64/1/4; 64/4/8; 64/8/8; 32/2/8 | 48 / 64 / 64 / 64 |

`BS`, `OPB`, and `VW` denote split-K block size, outputs per block, and vector
width. Static K fully unrolls the K-tile loop, so the dynamic kernel's
`k_unroll` knob is inactive and fixed to one in these configurations.

#### Absorbed BMM results

Both 24-layer absorbed BMMs were screened over 96 Triton tile/warp/stage
combinations at M=1,2,4,8,16,32,48,64. PyTorch/nvJet wins every point. W-UK
takes 95.7-123.2 us per 24 calls and W-UV takes 90.3-129.9 us; no custom BMM is
installed. The strongest configurations from those screens were then checked
at every integer M=1-64 and remained slower throughout. Raw data:
`bmm-full-screen-sampled-m1-64.json` and `bmm-{wuk,wuv}-selected-m*.json`.

### 2026-08-24: first complete M=20 backend and layout sweep

This section preserves the chronological M=20 investigation. Intermediate
recommendations here—especially the two-launch all-Triton `f_b` path—are
superseded by the exhaustive M=1-64 results and production table above.

All results below are isolated CUDA-graph replay medians on GB300. Inputs,
weights, outputs, and packed scale tensors remain allocated across replays.
The FP8 numbers time the prequantized GEMM only; combined dynamic
quantization-plus-GEMM and weight-ring results appear below.

#### FP8 x FP8 projections

| Shape `(M,N,K)` | Current/default | Best candidate | Best | Improvement |
| --- | ---: | --- | ---: | ---: |
| `(20,49408,7168)` KDA input | 54.65 us | DeepGEMM default | 54.65 us | none |
| `(20,12288,128)` KDA `f_b` | 2.85 us default; 2.78 us current BM256 | BM256/BN128 | 2.72 us | 2.3% versus current |
| `(20,7168,12288)` shared `o_proj` | 13.46 us current BM256 | BM32 isolated only | 13.00 us | rejected by cold replay |
| `(20,2176,7168)` MLA Q/KV-A | 7.55 us | several tied layouts | 7.53 us | noise-level |
| `(20,18432,1536)` MLA Q-B | 5.00 us | DeepGEMM default | 5.00 us | none |
| `(20,12288,7168)` MLA output gate | 9.55 us | DeepGEMM default | 9.55 us | none |

CUTLASS and Triton were slower for every prequantized FP8 shape. Examples:
the KDA input projection measured 92.1 us with CUTLASS and 126.8 us with
Triton versus 54.6 us with DeepGEMM; `o_proj` measured 24.6 us and 83.4 us
versus 13.0 us. DeepGEMM remains the correct backend.

At M=20 these are predominantly skinny, weight-streaming GEMMs rather than
tensor-core-throughput problems. Using only the FP8 weight bytes, the KDA
input projection sustains approximately 6.48 TB/s (259 TFLOP/s) and the shared
`o_proj` approximately 6.83 TB/s (273 TFLOP/s). MLA Q-B reaches 5.66 TB/s;
the output gate's 9.22-TB/s weight-only estimate indicates substantial cache
reuse. The very narrow KDA `f_b` shape reaches only 0.38 TB/s and is dominated
by quantization and launch/layout overhead, explaining why backend dispatch
has much larger leverage there. This roofline check makes a custom replacement
for the 338-MiB KDA input projection unlikely to beat DeepGEMM materially
without changing weight representation; parameter/layout tuning is the
appropriate scope.

A 15-trial full-pipeline follow-up for the 338-MiB KDA input weight found
default selection and BM32 indistinguishable (both 57.4-58.2 us as clocks
shifted), while BM256/BN128 was slightly slower. The apparent 0.9% BM32 win in
the shorter run was ordering noise, so this high-call-count shape deliberately
keeps default DeepGEMM selection.

The repeated single-weight `o_proj` sweep initially appeared to find a sharp
layout crossover. BM32 measured 12.901 us versus 13.472 us for BM256 at M=20,
while BM256 won starting at M=33. This result was an L2-cache artifact: one
`o_proj` weight is approximately 84 MiB, whereas production rotates through
93 layer weights. With a three-weight ring that exceeds L2, BM256 measured
20.445 us per projection versus 21.961 us for BM32 and 24.181 us for
unconstrained selection. The 20-23 us range also matches the source Nsight
trace. The production plan therefore retains BM256 over M=11-256.

The earlier broad sweep also exposed that BM32 is not merely slow above the
crossover: at M=40 it triggers a device-side gather assertion in DeepGEMM's
configuration-selection path. BM256 is valid at M=40 and measured 12.767 us.
This reinforces keeping the constraint narrowly bounded by the tested M range
instead of extrapolating an isolated-cache result to production.

#### BF16 projections

| Shape `(M,N,K)` | Current PyTorch | Best candidate | Best | Improvement |
| --- | ---: | --- | ---: | ---: |
| `(20,896,7168)` router, FP32 output | 5.887 us | CuTe split-K=5, stages=2 | 5.855 us | 0.5%; reject |
| `(20,3584,7168)` latent down | 9.65 us | cuBLASLt | 9.62 us | noise-level |
| `(20,7168,3584)` latent up + add | 7.778 us | cuBLASLt beta epilogue | 7.770 us | 0.1%; reject |
| `(20,67584,7168)` dense gate/up | 155.94 us | DeepGEMM BM64 | 141.85 us | 9.0% |
| `(20,7168,33792)` dense down | 75.89 us | cuBLASLt | 75.75 us | noise-level |

The initial router result did not survive a longer, clock-preconditioned paired
run: CuTe measured 5.855 us versus 5.887 us for PyTorch, only 0.5%. Its output
also differs by `2.2e-6` relative mean absolute error, so this candidate is not
worth its integration cost. The dense gate/up switch remains well separated
and saves approximately 14.1 us once per step. The initial latent-up
baseline used `mm + add`, while production uses an `addmm_` beta epilogue.
After correcting the benchmark, PyTorch `addmm` measured 7.778 us and the
direct cuBLASLt wrapper measured 7.770 us over 11 paired trials. That 0.1%
difference is not actionable; the current path should remain unchanged.

DeepGEMM BF16 is not a general replacement: it loses on both latent
projections and dense down. Its large-N dense gate/up shape is the sole clear
BF16 winner. The longer paired run is recorded in
`bf16-selected-m20-paired.json`; the corrected routed-up beta-epilogue run is
`latent-up-exact-addmm-paired.json`.

The first dense gate/up integration covered only the measured M=20 shape,
using DeepGEMM's BM64/BN64-constrained BF16 GEMM behind a registered vLLM
custom op. In a seven-trial paired validation, the production wrapper
measured 141.30 us versus 156.77 us for PyTorch, a 9.9% reduction, and exactly
matched the direct DeepGEMM result. A `torch.compile(fullgraph=True)` test of
the complete allocation/dispatch/custom-op path succeeded and matched eager
output exactly. Backend availability is resolved while installing the static
shape plan, not inside the compiled forward graph. The raw validation is
`dense-gate-up-custom-op-m20.json`.

A follow-up crossover sweep initially kept the production rule narrow.
Unconstrained DeepGEMM wins over PyTorch by 3.3-4.4% at M=2-5, but PyTorch
switches to a faster cuBLASLt algorithm and wins at M=6-7. At M=17-19,
BM64/BN64 DeepGEMM wins by 9.8-9.9%, consistent with the M=20 result. These
disjoint ranges make a broad shape-level backend replacement inappropriate.
Only M=20 was used by the source DSpark K=4 replay, so that first selector did
not speculate about unmeasured M=8-16 or M>20 behavior. The later exhaustive
audit above supersedes that restriction. The initial crossover data is in
`dense-gate-up-crossover.json`.

#### Full `f_b` projection pipeline with the real strided input

The `f_a` input to `f_b_proj` is not a standalone contiguous tensor. It is a
128-column view at offset 49152 in the packed `[M,49408]`
`in_proj_qkvgfab` output. The benchmark now reproduces that parent allocation
and row stride. These measurements include activation quantization and GEMM:

| Backend | M=20 median | Relative to tuned DeepGEMM |
| --- | ---: | ---: |
| Packed UE8M0 quantization + DeepGEMM BM256/BN64 | 7.336 us | baseline |
| Stride-aware quantization + CUTLASS | 5.106 us | 30.4% faster |
| Stride-aware quantization + default Triton GEMM | 5.096 us | 30.5% faster |
| Stride-aware quantization + tuned Triton GEMM | **4.122 us** | **43.8% faster** |
| FlashInfer BF16-input block-scale GEMM | unsupported | SM90-only kernel |

Both non-DeepGEMM candidates reproduce the reference output for the test
tensor. This is a stronger result than further DeepGEMM layout tuning: the
packed DeepGEMM quantizer first materializes the row-strided 128-column view,
whereas the existing general quantizer consumes its leading stride directly.
No new GPU kernel is required. The best configuration across the batch sweep
uses the existing Triton kernel with `BLOCK_SIZE_M=16`,
`BLOCK_SIZE_N=128`, four warps, and two pipeline stages. It is 19.1% faster
than the default Triton configuration at M=20. A separate 15-trial run found
4.144 us for BM16 and 4.134 us for BM32, a noise-level reversal; BM16 is the
better general choice because it wins consistently at the other measured M
values. The sustained DeepGEMM baseline varied from 7.336 to 7.658 us between
the two paired sessions, so the measured improvement is 43.8-46.0%. All four
supported paths produced exactly matching output in these tests.

The backend crossover is batch-dependent. In the repeated full-pipeline
sweep, DeepGEMM wins at M=1 (4.761 us versus 4.952 us for tuned BM16 Triton),
while tuned Triton wins for every measured M from 2 through 64. Its medians
are 3.915 us at M=2, 3.748 us at M=4, 3.749 us at M=8, 3.882 us at M=16,
4.122 us at M=20, 4.109 us at M=24, 4.062 us at M=32, and 4.610 us at M=64.
These results support replacing this exact `(N,K)=(12288,128)` layer with the
stride-aware Triton path. This is preferable to adding a KDA-specific copy or
fusion kernel.

The synthetic sweep originally used power-of-two E8M0 weight scales. A second
run loaded layer 12 directly from the NVFP4 checkpoint. Its `f_b_proj` has FP32
block scales with only 1 of 96 values being an exact power of two. DeepGEMM's
GB300 path therefore requantizes the checkpoint weight to E8M0 during loading,
whereas Triton can consume the original FP8 weight and FP32 scale directly.
Against a BF16 matmul using the checkpoint-dequantized weight, the full Triton
pipeline had 2.722% relative mean absolute error, compared with 3.784% for
DeepGEMM. Thus the faster backend is also closer to the original checkpoint
semantics in this test. The checkpoint-backed M=20 medians were 4.117 us for
tuned Triton and 7.309 us for DeepGEMM after warmup.

A hybrid M=1/other-M implementation would need to retain both the original
weight/scale and DeepGEMM's requantized weight/packed scale. That costs about
1.5 MiB per `f_b_proj`, or about 104 MiB for all 69 KDA layers. This is only
about 0.04% of a 288-GB GB300 GPU. A simpler all-Triton implementation avoids
that duplication and gives up only 0.19 us per `f_b` call at M=1; the final
recommendation will compare those two integration choices explicitly.

The simpler all-Triton option was selected. It avoids checkpoint-weight
duplication, improves fidelity, and remains faster than the existing
DeepGEMM pipeline through M=256 except for the 0.19-us M=1 difference. The
production configuration follows tile-count boundaries rather than a broad
nearest-M guess: BM16 for M=1-64, BM32 for M=65-128, BM64/BN256 with eight
warps for M=129-192, then BM32 again for M=193-256. Direct validation through
the normal configuration loader measured 4.128 us at M=20, 4.754 us at M=65,
6.062 us at M=129, 7.073 us at M=193, and 7.775 us at M=256. The backend swap
is model-, shape-, architecture-, and batch-invariant-mode gated by the
existing Kimi-K3 installation hook; it does not change a GPU kernel.

CUTLASS becomes 2-11% faster than the best Triton configuration near
M=224-256, but introducing a second dynamic quantization/backend path for
those non-target batches is not justified yet. Even there, configured Triton
remains faster than the prior DeepGEMM path.

#### Cache-faithful weight-ring replay

Repeatedly executing one projection can leave a meaningful fraction of its
weight resident in L2, especially for the 1.5-MiB `f_b` matrix. The final
backend decisions therefore use CUDA graphs that rotate distinct weights:
69 for `f_b`, three for the 84-MiB `o_proj`, nine for Q/KV-A, five for Q-B,
and two for the 84-MiB MLA output gate. Reported values are normalized per
projection and include activation quantization.

At M=20, the 69-weight `f_b` replay measured 4.792 us for configured Triton
versus 8.992 us for the prior BM256 DeepGEMM path, a 46.7% reduction. Across
69 KDA layers this is approximately 290 us per target step. The same replay
also reverses the isolated M=1 result: Triton measured 5.805 us versus
6.056 us for BM256 DeepGEMM. Consequently, the simple all-Triton backend swap
does not need a dual-format M=1 exception or an additional 104 MiB of duplicate
weights.

The cold-weight results reject the apparent BM32 `o_proj` optimization and
confirm DeepGEMM for Q/KV-A and the MLA output gate. Q-B shows a reproducible
BM256 advantage from M=17 through every measured point up to M=64. At M=20 it
measured 10.022 us versus 10.241 us for default selection; across 24 layers
that saves approximately 5.3 us. At M=16, default selection remains 0.3%
faster, so the production constraint starts at M=17 and stops at the highest
validated M rather than extrapolating. Raw files use the
`{projection}-weight-ring{count}-*.json` naming convention.

The BF16 weight rings do not change any backend decision. PyTorch/nvJet and
the direct cuBLASLt wrapper remain tied for latent down and latent up; PyTorch
remains faster for the FP32 router output. DeepGEMM loses on all three under
sustained cold-weight replay. The large dense matrices already exceed L2 by a
wide margin individually, so their single-weight measurements are naturally
cache-cold.

#### Reusing MLA input quantization

`fused_qkv_a_proj` and the output-gate `g_proj` consume the same BF16
`hidden_states` tensor and use identical ModelOpt FP8 block-linear consumers.
Previously, each linear independently produced the same per-token/per-128
E8M0 activation and packed scale tensor. vLLM already has a
`QuantizedActivation` contract that lets a linear consume prequantized input,
so this duplication can be removed without changing a GPU kernel.

A two-weight-pair replay first measured the sequential projection pipelines at
31.136 us with separate quantization and 28.676 us with shared quantization.
More importantly, a multi-stream replay reproducing the model's gate overlap
measured 27.268 us versus 26.571 us. The 0.697-us reduction per MLA layer is
stable across nine rotated trials and corresponds to approximately 16.7 us
over 24 layers. The Python integration only shares when both ModelOpt methods
select the exact same kernel class, quantization key, and layout-affecting
quantizer attributes. Other backend/layout combinations retain independent
quantization.

Raw data is in `mla-shared-quant-parallel-ring2-m20-paired.json`. This is a
microbenchmark result; no end-to-end model run was used.

#### Custom CuTe skinny GEMM result

The existing shape-dynamic CuTe skinny kernel was exhaustively swept over
valid block sizes, outputs per block, and K-unroll choices for both latent
projections. It is not competitive at M=20:

| Projection | Existing library path | Best CuTe skinny | Result |
| --- | ---: | ---: | --- |
| Latent down | 9.62 us | approximately 26 us | reject |
| Latent up + add | 7.77 us | approximately 31 us | reject |

The fused-residual CuTe output also differed from the reference by roughly
`1.4e-3` relative mean absolute error. This implementation direction is
closed; further work should use the already efficient nvJet/cuBLASLt path or
a materially different UMMA design, not incremental tuning of this kernel.

#### Integration validation

The final production selector table passes the complete focused GPU suite:
572 tests pass and one is skipped on GB300. This includes CuTe compilation and
numerics for every installed static-K shape, FP8 fused-kernel installation and
CUDA-graph capture, shared-quantization behavior, and fallback dispatch.
An artifact audit also confirms complete integer-M coverage from 1 through 64
for every selected target and draft projection path and both absorbed BMMs;
the broader backend/tile searches use representative points plus exact
crossover neighborhoods.

* Four focused selector tests cover the dense BF16 DeepGEMM plan, unavailable
  backend fallback, the measured FP8 ranges, and shape/quantization-specific
  installation.
* Three ModelOpt tests cover compatible shared quantization, FP8-PB-WO kernel
  finalization, and mixed-precision dispatch.
* The Kimi MLA helper test verifies that compatible QKV-A/gate projections
  receive one shared quantized activation and incompatible methods fall back
  to the original BF16 input.
* Ruff check, Ruff format, and `git diff --check` all pass for the modified
  implementation, tests, and benchmark.

The CUDA-dependent tests ran on the same allocated GB300 node used for the
microbenchmarks. No test limits kernel occupancy or reserves SMs.

#### End-to-end accuracy validation

The final selector was evaluated with the NVFP4 checkpoint, TP1/DP16/EP16,
FP8 KV cache, the FlashInfer MegaMoE shared-FC1+FC2 backend, and DSpark K=4
using standard rejection sampling. The complete 1,319-question, five-shot
GSM8K evaluation produced 93.6% accuracy with 0.1% invalid responses. It
generated 140,656 output tokens in 343.659 seconds. This is a real speculative
decode accuracy gate rather than the synthetic-acceptance workload used for
profiling.

The run artifact is:

```text
srt-slurm/outputs/
  kimi-k3-nvfp4-dep16-dspark4-projection-gemm-m1-64-final-gsm8k/
  564353/
```

#### End-to-end decode profile

The final Nsys replay uses the same TP1/DP16/EP16, FP8-KV, MegaMoE, and
DSpark-K4 stack with synthetic acceptance length 3.36 so that 100 decode
iterations can be compared with the original projection-tuned trace. Kernel
activity is normalized by device-step across four GPUs. The baseline router's
`nvjet_sm103_tss_64x16_64x16_2x2_2cta_h_bz_splitK_TNT` symbol and the new
`LLBf16SplitK` symbol are both assigned to MoE routing/preparation; without
that correction, moving the router between backends falsely appears as a
dense-projection saving.

| Kernel group | Prior DEP-GEMM trace | Final M=1-64 selector |
| --- | ---: | ---: |
| Dense/projection GEMMs | 13.70 ms (30.2%) | 12.69 ms (28.7%) |
| Fused EP MoE: dispatch + experts + combine + shared FC1/FC2 | 17.72 ms (39.1%) | 17.52 ms (39.6%) |
| MoE routing/preparation | 1.82 ms (4.0%) | 2.04 ms (4.6%) |
| MLA attention | 6.43 ms (14.2%) | 6.41 ms (14.5%) |
| KDA linear attention | 2.62 ms (5.8%) | 2.61 ms (5.9%) |
| AttnRes mixing | 1.09 ms (2.4%) | 1.09 ms (2.5%) |
| Speculative sampling | 0.03 ms (0.1%) | 0.03 ms (0.1%) |
| Other kernels | 1.88 ms (4.2%) | 1.82 ms (4.1%) |
| **Summed GPU activity** | **45.29 ms** | **44.20 ms** |
| **Measured decode cadence p50** | **43.61 ms** | **42.49 ms** |
| **Measured decode cadence p90** | **43.84 ms** | **42.81 ms** |

Dense/projection activity falls by 7.4%, total GPU activity by 2.4%, median
decode cadence by 2.6%, and p90 cadence by 2.3%. A preceding tuned replay
measured 12.83 ms projection activity, 44.76 ms total activity, and
42.63/43.07 ms p50/p90 cadence; the fresh result therefore confirms rather
than reverses the earlier gain.

This workload exercises the observed target-verification decode region. It
does not execute every low-M static-K window. Those M=1-64 paths are instead
covered by the exhaustive cache-faithful CUDA-graph microbenchmarks and the
572-pass GPU regression suite described above; the profile is the
end-to-end validation of the active workload, not a substitute for the
integer-M sweep.

Final profile artifact:

```text
srt-slurm/outputs/
  kimi-k3-nvfp4-dep16-dspark4-projection-gemm-m1-64-final-nsys/
  564644/logs/profiles/agg/
  nvl72d117-T13_agg_w0_profile_gpu0-1-2-3.sqlite
```

### 2026-08-24: exhaustive M=1-64 follow-up

The initial investigation emphasized the observed DSpark verification shape
at M=20. A follow-up now treats every integer M from 1 through 64 as a target.
It expands cache-faithful weight-ring mode to expose all tested DeepGEMM
layout constraints, the full Triton FP8 tile search, and CUTLASS. BF16 ring
results are normalized per projection, matching FP8 ring reporting.

All authoritative measurements run serially while reserving the complete
four-GPU allocation. A preliminary attempt to run four independent sweeps on
the four GPUs inflated latency by 2-4x due to node-level interference; those
measurements were stopped and discarded. Candidate discovery uses shorter
trials, followed by longer interleaved validation at every configuration
boundary. No sweep changes the number of SMs available to a kernel.

For the narrow KDA `f_b` projection, the follow-up also includes a
benchmark-only fused Triton candidate that quantizes the BF16 row and performs
the K=128 FP8 block-scaled GEMM in one launch. This tests whether eliminating
the quantization launch and intermediate FP8 write offsets the duplicated row
reduction across N tiles; it is not integrated unless cache-faithful data show
a robust win across an M range.

#### Reproduction

The benchmark harness is:

```text
benchmarks/kernels/benchmark_kimi_k3_nvfp4_dep16_projections.py
```

Representative commands, run inside the allocated GB300 environment, are:

```bash
DEP16_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/weizha/tmp/dep16-gemm-microbench
export XDG_CACHE_HOME="$DEP16_CACHE/xdg"
export TRITON_CACHE_DIR="$DEP16_CACHE/triton"

.venv/bin/python benchmarks/kernels/benchmark_kimi_k3_nvfp4_dep16_projections.py \
  --precision fp8 --shape kda_f_b_proj --m 20 \
  --include-fp8-pipeline --weight-ring 69 \
  --trials 7 --precondition-rep 500 --rep 1000 \
  --output "$DEP16_CACHE/fb-weight-ring69-m20.json"

.venv/bin/python benchmarks/kernels/benchmark_kimi_k3_nvfp4_dep16_projections.py \
  --precision fp8 --shape mla_q_b_proj --m 16 20 24 32 64 \
  --include-fp8-pipeline --weight-ring 5 \
  --candidate projection_deepgemm_default \
  --candidate projection_deepgemm_bm256_bn1 \
  --trials 7 --precondition-rep 500 --rep 800 \
  --output "$DEP16_CACHE/q-b-weight-ring5-crossover.json"

.venv/bin/python benchmarks/kernels/benchmark_kimi_k3_nvfp4_dep16_projections.py \
  --mla-shared-quant --m 20 --weight-ring 2 \
  --trials 9 --precondition-rep 1500 --rep 1500 \
  --output "$DEP16_CACHE/mla-shared-quant-parallel-ring2-m20-paired.json"

.venv/bin/python benchmarks/kernels/benchmark_kimi_k3_nvfp4_dep16_projections.py \
  --precision bf16 --shape dense_gate_up --m 20 \
  --bf16-deepgemm-sweep --trials 7 --precondition-rep 1000 --rep 1000 \
  --output "$DEP16_CACHE/dense-gate-up-custom-op-m20.json"
```

Raw JSON is stored outside the home filesystem under:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/weizha/
  tmp/dep16-gemm-microbench/
```

Relevant files are `fp8-m20-layout-exhaustive.json`,
`bf16-m20-all-v2.json`, `router-m20-cute-sweep.json`,
`bf16-dense-m20-layout-sweep.json`, and
`bf16-latent-m20-cute-skinny-sweep.json`. The exact strided-input runs are
`fb-strided-pipeline-m-sweep.json`,
`fb-strided-pipeline-backends-paired.json`,
`fb-triton-config-sweep-m20.json`, and
`fb-triton-winners-paired-m20.json`. The repeated batch sweep is
`fb-tuned-backends-m-sweep.json`; higher-M selection and boundary checks are
in `fb-tuned-backends-high-m.json`, `fb-triton-config-sweep-high-m.json`,
`fb-tuned-backends-crossover-high.json`,
`fb-triton-crossover-{65,129,193}.json`, and
`fb-production-triton-config-validation.json`. The real-checkpoint validation
is `fb-checkpoint-layer12-m20.json`.

The `o_proj` crossover data is in `o-proj-crossover-paired.json`,
`o-proj-crossover-33-35.json`, and `o-proj-m40-bm256.json`. The first file
ends at M=36 because the deliberately tested BM32 candidate asserts at M=40;
the narrower follow-up files isolate the safe production ranges.
Dense gate/up results are in `dense-gate-up-m-sweep.json`,
`dense-gate-up-crossover.json`, and `dense-gate-up-custom-op-m20.json`.
The cache-faithful MLA BMM comparison is
`mla-absorbed-bmm-cold-weight-paired.json`.
Full projection-pipeline and cold-weight results are in
`fp8-full-pipeline-m20-paired.json`,
`kda-input-layout-paired-long-m20.json`, `fb-weight-ring69-{m1,m20}.json`,
`o-proj-weight-ring3-m20.json`, `qkv-a-weight-ring9-m20.json`,
`q-b-weight-ring5-{crossover,boundary17-19}.json`,
`gate-weight-ring2-m20.json`, and the corresponding BF16
`{router,latent-down,latent-up}-weight-ring*.json` files.
