# Kimi K3 FlashInfer KDA Optimization Plan

Last updated: 2026-08-15

## Execution status (2026-08-15)

The first gather/scatter recurrent-prefill adapter and its reusable vLLM
microbenchmark are implemented on `integrate-flashinfer-kda-prefill`. The
change is not yet ready to submit upstream because full-model evaluation and
serving validation remain outstanding.

- Installed and import-checked `flashinfer-python==0.6.18.dev20260811`,
  `flashinfer-cubin==0.6.18.dev20260811`, and
  `flashinfer-jit-cache==0.6.18.dev20260811+cu130` from the official
  `nightly-v0.6.18-20260811` release.
- The nightly exposes `flashinfer.kda.recurrent_kda` and its frozen recurrent
  prefill supports SM100 and SM103, BF16 state, D=128, and H=12 beta padding.
- Added opt-in BF16 KDA state through `--mamba-ssm-cache-dtype bfloat16` while
  preserving FP32 for `auto` and `float32`.
- Added `flashinfer` backend resolution with SM100/SM103, dtype, head-dimension,
  gate, and API capability guards. Automatic selection prefers it only for
  eligible BF16-state configurations.
- Added the public-API adapter, caller-owned output path, existing state
  gather/scatter integration, and FP32-only native decode guard. The hot path
  reuses int64 sequence offsets and sorted sequence order across KDA layers and
  keeps a dedicated FlashInfer prefill workspace per layer.
- Added focused dtype, selection, SM103/CUDA-version eligibility,
  call-contract, decode-safety, and BF16-recurrence correctness tests.
- The focused suite passes on an NVIDIA B300: 24 passed. The real launch used
  the nightly's `flash_kda_bf16_fused_m128_sm100f` specialization for H=12.
- Added `benchmarks/kernels/benchmark_kimi_k3_kda_prefill.py`, reusing
  FlashInfer's CUPTI, cold-L2, packed-sequence, sorted-sequence, and rotating
  state methodology while invoking the actual vLLM adapters.
- A 50-sample cold-L2 B300 TP8 sweep shows FlashInfer speedups over the current
  Triton fallback from 1.32x to 13.20x. Detailed results and the exact command
  are recorded under Performance benchmark plan.
- Still required: complete-layer and full-model evaluation on B300/GB300;
  serving-level mixed-batch and graph-boundary validation; and official
  duplicate checks with `gh` before proposing a PR.

The repository's stable FlashInfer requirement remains unchanged. Stable
0.6.17 lacks the KDA modules, so this backend is capability-gated and requires
the installed nightly (or a later release containing the same public API).

## Executive decision

For NVIDIA B300/GB300 (SM103), the highest-priority performance target is the
FlashInfer CAKE recurrent-prefill kernel when KDA recurrent state is explicitly
configured as BF16.

The first production milestone must contain both:

1. BF16 KDA state plus FlashInfer CAKE prefill.
2. BF16-safe decode selection.

The second item is required because vLLM's native fused KDA decoder currently
requires FP32 recurrent state. Enabling BF16 state without updating decode
selection can route to an incompatible kernel.

The existing FP32 behavior remains the default. BF16 is an explicit user
choice and must be validated as a model-semantics change, not treated as a
transparent storage optimization.

## Goals

- Improve Kimi K3 prefill latency, TTFT, and serving throughput on GB300.
- Allow users to opt into BF16 KDA recurrent state.
- Preserve current FP32 behavior under the default configuration.
- Retain safe fallbacks for unsupported architectures, state dtypes, shapes,
  FlashInfer versions, and speculative-decode modes.
- Integrate through public FlashInfer APIs rather than copying frozen CAKE
  kernels into vLLM.
- Support CUDA Graph execution without per-replay allocation or JIT work.
- Establish full-model accuracy and performance evidence before recommending
  BF16 state for production.

## Non-goals for the first milestone

- Changing the default KDA recurrent-state dtype from FP32 to BF16.
- Replacing AMD KDA kernels.
- Replacing speculative decode before a compatible FlashInfer kernel is
  demonstrated.
- Depending on an unmerged FlashInfer API in a production vLLM dependency.
- Adding one-off performance benchmarks under `tests/`.

## Current vLLM implementation

The relevant implementation is in
[`vllm/models/kimi_k3/nvidia/kda.py`](vllm/models/kimi_k3/nvidia/kda.py).

The normal Kimi K3 TP=8 configuration uses:

- 12 local KDA heads from 96 total heads.
- Head dimension 128.
- Convolution width 4.
- BF16 activations and convolution state.
- FP32 recurrent state.
- Bounded decay gate with lower bound `-5.0`.

Current execution paths:

| Workload | Current path |
| --- | --- |
| Non-speculative T=1 decode | vLLM fused CUDA kernel |
| Speculative decode | causal-conv update plus vendored Triton recurrence |
| Prefill | three causal convolutions plus standalone FlashKDA or Triton |
| Plain decode fallback | causal-conv update plus Triton packed recurrence |

KDA currently ignores `mamba_ssm_cache_dtype` and always assigns FP32 to its
recurrent state in
[`vllm/model_executor/layers/mamba/mamba_utils.py`](vllm/model_executor/layers/mamba/mamba_utils.py).

The native fused decoder explicitly requires FP32 state in
[`csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu`](csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu).
Its support predicate does not currently receive the recurrent-state dtype.

## FlashInfer kernel assessment for GB300

| Kernel | SM103 | State dtype | K3 TP8 fit | Priority |
| --- | --- | --- | --- | --- |
| CAKE recurrent prefill | Yes | BF16 | H=12, D=128 supported | Highest |
| Indexed/checkpoint CAKE prefill from PR #4445 | Yes | BF16 | Excellent serving contract | Highest after merge |
| `flashinfer.fused_kda_decode` | Expected to run on SM103; GB300 performance must be measured | BF16 or FP32 | Excellent | Same first milestone or next |
| CAKE recurrent/spec decode | Yes | BF16 | Current variants are a partial contract match | Later |
| Merged packed CuTe T=1 decode from PR #4417 | No; exact SM100/B200 route | BF16 | H=12 supported | Do not use on GB300 |

Published FlashInfer evidence motivating prefill priority includes:

- PR #4351: 1.8238x six-shape geometric-mean KDA prefill kernel speedup on
  GB300 for H=12 relative to official FlashKDA.
- PR #4445: 1.846388x geometric-mean B300 prefill kernel speedup relative to
  official FlashKDA.
- PR #4445: approximately 1.03x-1.086x TTFT improvement and 1.4284x throughput
  improvement in its reported two-GB300 tiny-K3 experiment with decode held
  fixed.

These results are evidence of potential, not acceptance evidence for vLLM.
They use BF16 recurrent state and must be reproduced with vLLM's scheduler,
cache layout, CUDA Graphs, TP configuration, and full Kimi K3 model.

The BF16 correctness oracle must round recurrent state after every token.
vLLM's current chunked Triton prefill keeps FP32 accumulation inside the
prompt and casts only when writing the final cache state, so it is not a
semantically matching oracle for the opt-in BF16 FlashInfer path.

## User-facing configuration

Reuse the existing option:

```text
--mamba-ssm-cache-dtype {auto,float32,float16,bfloat16}
```

KDA-specific behavior:

| Value | KDA behavior |
| --- | --- |
| `auto` | FP32 recurrent state, preserving the current default |
| `float32` | FP32 recurrent state |
| `bfloat16` | BF16 recurrent state; eligible for FlashInfer CAKE |
| `float16` | Reject for Kimi K3 initially with a clear error |

Do not use `mamba_cache_dtype` to opt into BF16 recurrent state. That option
also controls convolution state, while `mamba_ssm_cache_dtype` expresses the
intended state-only override.

Extend the prefill selector to:

```text
--kda-prefill-backend {auto,triton,flashkda,flashinfer}
```

Proposed selection policy:

| Configuration | Selected path |
| --- | --- |
| `auto`, SM100/SM103, BF16 state, compatible FlashInfer | FlashInfer CAKE |
| `auto`, FP32 state | Existing FlashKDA when supported |
| Explicit `flashinfer` with an incompatible configuration | Fail closed with the precise reason |
| Unsupported platform or missing API under `auto` | Existing backend fallback |

`flashinfer` is the public backend name. CAKE is an internal implementation
detail and should not be exposed as a separate user-facing backend unless
FlashInfer itself introduces multiple selectable KDA implementations.

## Phase 0: dependency and duplicate-work checks

- [ ] Re-run the required vLLM issue and open-PR searches immediately before
  starting a PR.
- [ ] Confirm the first released `flashinfer-python` version containing the
  required recurrent-prefill API.
- [ ] Confirm the live merge status and final API of FlashInfer PR #4445.
- [ ] Update the FlashInfer dependency only after the required API is in an
  official release or an accepted vLLM dependency channel.
- [ ] Record the exact FlashInfer commit, wheel version, CUDA version, and
  generated-kernel identities used for benchmark evidence.

The current vLLM dependency does not yet expose all of the new main-branch KDA
APIs. The integration must use capability/version detection and retain a
fallback rather than assuming their presence.

## Phase 1: BF16 KDA state plumbing

### Implementation

- [ ] Extend `MambaStateDtypeCalculator.kda_state_dtype` to accept
  `mamba_ssm_cache_dtype`.
- [ ] Preserve FP32 for `auto`; use BF16 only for explicit `bfloat16`.
- [ ] Pass the SSM dtype through every Kimi K3 state-shape/dtype query,
  including both layer construction and model-level cache sizing.
- [ ] Audit other callers of `kda_state_dtype`, such as Bailing/Kimi linear
  attention implementations, and preserve their existing default behavior.
- [ ] Reject unsupported FP16 KDA state rather than silently selecting a kernel.
- [ ] Ensure cache allocation, offloading, prefix-cache bookkeeping, and graph
  buffers preserve the selected recurrent dtype.

### Decode safety

- [ ] Add recurrent-state dtype to `is_fused_kda_decode_supported`.
- [ ] Select the native vLLM fused decoder only for FP32 recurrent state.
- [ ] Route BF16 state to the existing dtype-generic Triton packed decoder until
  the FlashInfer fused decoder is enabled.
- [ ] Add a regression test proving explicit BF16 never calls the FP32-only
  native operation.

### Memory expectation

For TP=8, one recurrent-state slot per layer is:

```text
12 * 128 * 128 * sizeof(state dtype)
```

This is 0.75 MiB in FP32 and 0.375 MiB in BF16. For a configuration with 69
KDA layers, BF16 saves approximately 25.9 MiB per cached sequence/state slot
per rank. Measure actual cache capacity after allocator and block-alignment
overheads rather than treating this calculation as an end-to-end result.

## Phase 2: merged CAKE recurrent-prefill adapter

Target the public `flashinfer.kda.recurrent_kda` facade, not private generated
bindings.

### Tensor mapping

Map the existing vLLM tensors as follows:

| vLLM value | FlashInfer argument |
| --- | --- |
| Convolved Q/K/V `[1,T,H,128]` | `q`, `k`, `v` |
| Raw KDA gate `g1` | `g` with `use_gate_in_kernel=True` |
| Raw beta logits | `beta` with `beta_is_logit=True` |
| `A_log` / `dt_bias` | Same FP32 parameters |
| `-5.0` bound | `lower_bound` |
| Packed query starts | `cu_seqlens` |
| Gathered BF16 state | `initial_state` |
| `core_attn_out` slice | Caller-owned `output` |

### First adapter version

The merged FlashInfer prefill route does not accept indexed state pools. The
first version can retain vLLM's existing sequence:

```text
gather initial recurrent states
    -> FlashInfer CAKE recurrent prefill
    -> scatter final recurrent states
```

- [ ] Preserve `has_initial_state` behavior, including zero initial state for a
  sequence without a committed state.
- [ ] Validate Q/K/V/G contiguity and explicitly account for any materialization
  in benchmarks.
- [ ] Provide contiguous beta if required by the released API.
- [ ] Request a caller-owned FlashInfer workspace suitable for capture.
- [ ] Warm every specialization and descriptor before CUDA Graph capture.
- [ ] Do not allocate or compile during graph replay.
- [ ] Use one safe workspace per captured invocation if required by the final
  FlashInfer workspace contract.

### Backend resolution

- [ ] Add `flashinfer` to `resolve_kda_prefill_backend`.
- [ ] Include architecture, head dimension, activation dtype, recurrent-state
  dtype, gate mode, API availability, and CUDA version in eligibility.
- [ ] Make explicit selection fail closed.
- [ ] Log the resolved backend once without adding hot-path synchronization.

## Phase 3: serving-native indexed prefill

After PR #4445 merges and is released, replace gather/scatter with its direct
state-pool interface.

- [ ] Pass `non_spec_state_indices_tensor` as `ssm_state_indices`.
- [ ] Pass the recurrent state pool directly, retaining padded physical slot
  stride.
- [ ] Consume row-strided beta without `.contiguous()`.
- [ ] Update final state in place at the correct cache slot.
- [ ] Preserve null/padded graph slots and fresh-sequence zero-state semantics.
- [ ] Prove aliasing and state-slot behavior under mixed prefill/decode batches.

Do not enable intermediate state checkpoints in the initial indexed-state
change. Treat checkpoints as an additional feature because:

- FlashInfer checkpoints require 32-token alignment.
- vLLM permits Mamba block sizes that are only multiples of eight.
- Recurrent and convolution state must remain synchronized at every reusable
  checkpoint.

Checkpoint follow-up:

- [ ] Enable only when `mamba_block_size` and scheduler boundaries satisfy the
  FlashInfer alignment contract.
- [ ] Fall back to final-state-only behavior otherwise.
- [ ] Validate the pre-block checkpoint convention against vLLM cache metadata.
- [ ] Verify both recurrent-state and convolution-state recovery.

## Phase 4: BF16-capable fused decode

Integrate `flashinfer.fused_kda_decode` after or alongside Phase 2.

- [ ] Add a lazy import through `vllm/utils/flashinfer.py`.
- [ ] Add an internal decode backend resolver; expose a CLI option only if
  operational debugging or stable A/B selection requires it.
- [ ] Enable for non-speculative T=1, D=128, width=4, H in the supported set,
  BF16 activations, and BF16 or FP32 recurrent state.
- [ ] Preserve the current native vLLM decoder as the FP32 fallback.
- [ ] Preserve Triton as the BF16 fallback.
- [ ] Warm CuTe/JIT work before graph capture.
- [ ] Verify output, convolution state, and recurrent state after graph replay.

GB300 latency must be measured directly. The existing FlashInfer fused-decode
performance report is from B200 and is not sufficient evidence for SM103.

## Phase 5: speculative decode

Keep the current Triton speculative path initially.

Only consider CAKE/CuTe speculative decode after the candidate supports:

- BF16 indexed recurrent state.
- Kimi K3 TP8 H=12.
- Raw bounded gate and beta logits.
- The exact token counts scheduled by vLLM.
- Accepted-token state selection.
- vLLM checkpoint/null-slot semantics.
- A demonstrated end-to-end improvement after any gate/beta preprocessing.

## Correctness test plan

Extend existing Kimi K3 tests where possible, particularly
[`tests/models/kimi_k3/test_kda.py`](tests/models/kimi_k3/test_kda.py).

### State dtype and selection

- [ ] `auto` produces FP32 recurrent state.
- [ ] Explicit `float32` produces FP32 recurrent state.
- [ ] Explicit `bfloat16` produces BF16 recurrent state.
- [ ] Explicit `float16` fails as documented.
- [ ] FP32 selects only FP32-compatible decode kernels.
- [ ] BF16 never selects the vLLM FP32-only native decoder.
- [ ] Explicit unavailable `flashinfer` fails; `auto` falls back.

### Kernel correctness

Compare all observable results:

- BF16 output.
- Complete recurrent state, including untouched slots.
- Convolution state.
- Final state indices.
- CUDA Graph changed-input replay.

Cover:

- H=12 as the primary TP8 case; H=24/48/96 as secondary cases.
- Empty initial state and committed initial state.
- Packed prompts with varied lengths and chunk tails.
- H=12 beta padding.
- Row-strided beta from the fused projection.
- Padded physical state-slot stride.
- Null graph slots.
- Mixed prefill/decode batches.
- Prefix caching on and off.
- Chunked prefill.
- Capture and replay on the current stream.

For BF16-state correctness, compare CAKE to a BF16-state reference with the
same per-update rounding semantics. Also report the difference from the FP32
default, but do not require BF16 and FP32 state trajectories to be identical.

## Model evaluation plan

Because BF16 state can change model output, kernel tests are insufficient.

- [ ] Fixed-prompt greedy output and logit differential tests.
- [ ] Short and long-context tests, including repeated state updates across
  scheduler steps.
- [ ] Prefix-cache hit and recovered-state tests.
- [ ] Chunked-prefill differential tests.
- [ ] Full CUDA Graph tests.
- [ ] TP=8 Kimi K3 evaluation on GB300.
- [ ] DCP configurations used by the target deployment.
- [ ] Speculative decoding with the BF16-safe fallback path.
- [ ] Relevant suites under `tests/evals/` plus task-level evaluation agreed
  for Kimi K3.
- [ ] Document any accuracy difference between FP32 and opt-in BF16.

The FP32 default must remain numerically and behaviorally unchanged.

## Performance benchmark plan

Place reusable microbenchmarks under `benchmarks/kernels/`, not `tests/`.

### Four-way prefill comparison

On the same GB300 system compare:

1. FP32 state plus current standalone FlashKDA.
2. BF16 state plus the existing Triton or compatible FlashKDA baseline.
3. BF16 state plus merged CAKE with gather/scatter.
4. BF16 state plus serving-native indexed CAKE after PR #4445.

This separates:

- The effect of state dtype.
- CAKE recurrence compute.
- Gather/scatter and beta materialization.
- The indexed serving interface.

### Shapes

Prioritize Kimi K3 TP8 H=12 with:

- Single sequences from short prompts through long context.
- Packed mixed lengths.
- Scheduler-sized chunks.
- Prompt concurrency representative of production.
- Full-chunk and tail-heavy cases.

Record H=24/48/96 for broader TP coverage without allowing those shapes to
hide an H=12 regression.

### Measurements

- CAKE recurrence kernel latency.
- Complete KDA layer span, including convolution, gather/scatter, copies, and
  normalization.
- End-to-end TTFT.
- Decode TPOT.
- Total input/output throughput.
- Peak and steady-state HBM use.
- Maximum cache capacity/concurrency.
- Kernel-launch count and inter-kernel gaps.
- CUDA Graph replay latency.

Use cold-L2 and steady-state graph measurements where appropriate, report the
method, and retain per-shape results rather than only a geometric mean.

### First B300 kernel results (2026-08-15)

Command:

```bash
FLASHINFER_WORKSPACE_BASE=/tmp .venv/bin/python \
  benchmarks/kernels/benchmark_kimi_k3_kda_prefill.py \
  --warmup-iters 10 --benchmark-iters 50 \
  --json /tmp/vllm_k3_kda_prefill_b300.json
```

Environment: NVIDIA B300 SXM6 AC, CC 10.3, Torch 2.13.0+cu130,
FlashInfer 0.6.18.dev20260811, CUDA 13.0, and cupti-python 13.3.1. Results are
median GPU spans with cold L2, H=12, D=128, and BF16 recurrent state.

| Case | Sequence lengths | FlashInfer | Triton | Speedup |
| --- | --- | ---: | ---: | ---: |
| `fixed128` | 128 | 21.728 us | 286.897 us | 13.20x |
| `fixed512` | 512 | 44.352 us | 288.545 us | 6.51x |
| `fixed2048` | 2048 | 135.105 us | 326.161 us | 2.41x |
| `fixed8192` | 8192 | 498.066 us | 656.867 us | 1.32x |
| `mixed` | 128, 256, 512, 1024, 2048, 4096 | 257.425 us | 540.082 us | 2.10x |
| `uniform` | 8 x 1024 | 76.945 us | 490.402 us | 6.37x |

These are core KDA backend spans, not complete-layer or end-to-end results.
The benchmark excludes convolution, state gather/scatter, output norm, and
projection. Its cross-backend diagnostics report approximately 0.44%-0.45%
output relative L2 and 0.31%-0.34% state relative L2; this is expected because
FlashInfer rounds BF16 state after every token while Triton accumulates within
a chunk.

## Acceptance criteria

### Required for merge

- [ ] Default FP32 behavior is unchanged.
- [ ] BF16 is explicit and documented.
- [ ] No incompatible native decode selection under BF16.
- [ ] Correct output and complete state against the matching BF16 reference.
- [ ] CUDA Graph capture/replay passes without replay allocation or JIT.
- [ ] Unsupported configurations fall back or fail according to explicit vs
  automatic selection.
- [ ] Relevant unit, integration, lint, and model tests pass.
- [ ] Full-model evaluation results are included in the PR.
- [ ] GB300 benchmark commands and raw per-shape results are included.
- [ ] The PR description records duplicate-work checks and AI assistance as
  required by `AGENTS.md`.

### Required before making CAKE the automatic BF16 path

- [ ] No end-to-end regression in the target prompt/decode workload mix.
- [ ] A reproducible GB300 improvement beyond benchmark noise.
- [ ] No material correctness regression relative to the agreed BF16-state
  baseline.
- [ ] Stable memory use and graph replay across production batch shapes.

Do not set a universal percentage threshold before collecting the baseline.
Report TTFT, TPOT, throughput, and memory separately because the best backend
may depend on workload mix.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| BF16 recurrence changes outputs | Opt-in only, matching BF16 reference, full-model evaluations |
| Native fused decode crashes on BF16 | Include recurrent dtype in support checks; use Triton/FlashInfer fallback |
| CAKE kernel gain is hidden by gather/scatter | Measure complete layer span; adopt indexed interface after merge |
| Projection views force materialization | Validate strides and use PR #4445 strided beta support |
| CUDA Graph workspace aliasing | Warm explicitly and follow one-workspace-per-captured-invocation contract |
| Checkpoint alignment differs from vLLM block size | Enable only for compatible multiples of 32; otherwise fall back |
| FlashInfer API changes before release | Depend on released public API and capability detection |
| B200 results fail to transfer to B300 | Require direct SM103 microbenchmark and full-model evidence |
| BF16 cache saves memory but reduces accuracy at long context | Test long trajectories and document the production tradeoff |

## Recommended delivery sequence

1. BF16 dtype plumbing, selection guards, and tests.
2. Merged FlashInfer CAKE prefill adapter using existing gather/scatter.
3. GB300 microbenchmarks and full Kimi K3 evaluation.
4. BF16-capable FlashInfer fused decode, if the Triton decode fallback offsets
   too much of the prefill gain or direct GB300 results justify it.
5. Upgrade to PR #4445 indexed/strided prefill after merge and release.
6. Add aligned state checkpoints only after recurrent and convolution cache
   semantics are proven together.
7. Evaluate speculative-decode kernels last.

## References

- FlashInfer CAKE tracker:
  <https://github.com/flashinfer-ai/flashinfer/issues/4254>
- Fused Kimi K3 decode:
  <https://github.com/flashinfer-ai/flashinfer/pull/4243>
- B200 CAKE recurrent prefill:
  <https://github.com/flashinfer-ai/flashinfer/pull/4262>
- SM103 recurrent prefill:
  <https://github.com/flashinfer-ai/flashinfer/pull/4313>
- H=12 recurrent-prefill support and GB300 measurements:
  <https://github.com/flashinfer-ai/flashinfer/pull/4351>
- B200 CAKE recurrent decode:
  <https://github.com/flashinfer-ai/flashinfer/pull/4279>
- SM103 recurrent decode:
  <https://github.com/flashinfer-ai/flashinfer/pull/4314>
- Packed CuTe decode:
  <https://github.com/flashinfer-ai/flashinfer/pull/4417>
- Indexed/checkpoint prefill and packed decode:
  <https://github.com/flashinfer-ai/flashinfer/pull/4445>
