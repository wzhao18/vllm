# Kimi K3 native NVFP4 MLA kernel bring-up

This directory records the initial standalone bring-up of the native FP4 MLA
decode kernels from TensorRT-LLM MR 10576 before integrating them with vLLM.

## Environment

- GPU: NVIDIA GB300, compute capability 10.3
- Container: `nvcr.io/nvidia/pytorch:25.12-py3`
- PyTorch: `2.10.0a0+b4e4ee81d3.nv25.12`
- CUDA: 13.1
- Triton: 3.5.1
- Source revision: TensorRT-LLM MR 10576 commit
  `ca5ed123f9310a12ce45b4953aac11c0d1000ad7`
- Shape: batch 1, 128 query heads, MLA rank 512, RoPE dimension 64,
  128-token cache pages

The harness uses a synthetic but structurally valid native cache containing
packed E2M1 values, swizzled E4M3 scales, the auxiliary PV scale view, and a
fixed-stride page table. Timings use CUDA events after five warmup iterations
and average 20 decode iterations.

## Triton native FP4 QK/PV

The native kernel compiles and executes correctly on SM103. Validation at
context length 128 against a dequantized reference produced:

- output relative L2 error: `0.00134254`
- output cosine similarity: `0.99999905`
- output maximum absolute error: `0.00012090`
- probability maximum absolute error: `0.00125946`

The final SM103 path reads V directly from the canonical 392-byte/token page.
It uses `BLOCK_V=128`, because the 32-wide inline transpose issues a
misaligned access on SM103. It does not allocate a persistent transposed V
copy or a context-sized dequantization buffer.

| Context tokens | TMA disabled (ms) | TMA enabled (ms) |
|---:|---:|---:|
| 128 | 0.2808 | 0.2744 |
| 1,024 | 0.3184 | 0.3156 |
| 4,096 | 0.3555 | 0.3608 |
| 16,384 | 0.3409 | 0.3593 |
| 65,536 | 0.3589 | 0.3611 |

TMA was not clearly beneficial for this batch-1 shape and defaults off on
SM103. More batch sizes and an FP8 backend comparison are still required.

The vLLM adapter was also validated against the TRT-LLM wrapper using the same
opaque page allocation: maximum absolute and relative L2 differences were both
zero. This includes the DSpark target shape (batch 8, query length 5, context
1,024), which ran in `0.3299 ms` per attention layer on GB300. Native cache
insertion produced the expected quantization error:

- one-token decode update relative L2: about `0.09` to `0.10`
- full 16-token prefill tile relative L2: about `0.10`
- DCP speculative-offset case relative L2: `0.1039`
- mid-tile chunked-prefill continuation relative L2: about `0.10`
- Q prefix relative L2: about `0.095`
- Q RoPE tail with residual relative L2: about `0.0088`

## CuTe DSL status

The MR's CuTe DSL kernel is not directly buildable with the public CUTLASS DSL
packages currently declared by vLLM:

- CUTLASS DSL 4.7.1 lacks the `is_exclusive` argument used by
  `tcgen05_alloc`.
- CUTLASS DSL 4.8.0.dev0 accepts that call, but the MR also relies on internal
  CTM helpers `cute.nvgpu.cfence`, `cute.nvgpu.warp_switch`, a
  `k_segment_offset` descriptor argument, and `PipelineProducer.index()`.

Diagnostic no-op shims for the two scheduling helpers and removal of the
zero-valued reserved descriptor argument let compilation advance further, but
it then stops at the missing `PipelineProducer.index()` API. These shims are
not production changes. The next step is to identify the exact CTM build used
by the MR or adapt the pipeline state calls to public CUTLASS DSL 4.8 before
evaluating SM103 code generation and performance.

## Reproduction

The benchmark can compare the integrated vLLM adapter with an extracted copy
of the MR source:

```bash
python benchmarks/kernels/benchmark_kimi_k3_nvfp4_mla.py \
  --source-dir /path/to/trtllm-mr10576/fp4_mla \
  --batch 8 --heads 128 --context 1024 --query-len 5 \
  --warmup 5 --iters 20 --opaque-vllm-cache --validate-vllm-adapter
```

The integrated SM103 path selects `BLOCK_V=128` automatically. It can be
overridden with `VLLM_KIMI_K3_NVFP4_BLOCK_V` for tuning.

## vLLM end-to-end status

The integrated backend was exercised with real Kimi-K3 weights on two GB300
nodes (TP8, DCP8), including the RoPE-enabled K3 DSpark draft and standard
block rejection. The run completed with native FP4 cache reads for both target
and draft. Two DCP-specific issues found during bring-up were fixed:

- vLLM keeps query tokens in global order, so an interleave-1 DCP rank must
  gather cache-update source rows `r, r + dcp_size, ...`; the original TRT-LLM
  writer assumed its generation rows were contiguous.
- Each speculative target row's global causal bound must be localized to the
  current DCP rank independently. Subtracting query offsets from one already
  localized final length is incorrect under round-robin sharding (for example,
  rank 1 at world size 2 maps global bounds `[8, 9, 10]` to `[4, 4, 5]`, not
  `[3, 4, 5]`). The corrected bounds and expanded block table are prepared once
  per batch in stable metadata buffers and reused by every layer.

The implementation currently rejects DCP interleave sizes other than one.

The real-weight arithmetic smoke recovered the expected reasoning after that
fix (`17 + 29 = 46`, `9 + 11 = 20`) under TP8/DCP8/DSpark. TP8/DCP1 and the
existing FP8 path were also used as isolation controls.

A deterministic 32-request probe used the first five GSM8K training examples
as few-shot demonstrations and evaluated the first 32 test examples. Both runs
used identical completion prompts, temperature 0, a 256-token output cap,
batch size 8, TP8, DCP8, the 4-token DSpark draft, and block rejection. Timings
exclude an explicit batch-8 warmup. This is a controlled subset check, not a
full GSM8K benchmark.

| Cache/backend | Exact match | Output tokens | Elapsed (s) | Output tok/s |
|---|---:|---:|---:|---:|
| `nvfp4_kimi_k3` / native | 32 / 32 | 3,349 | 204.42 | 16.38 |
| `fp8` / TOKENSPEED_MLA | 32 / 32 | 3,393 | 48.46 | 70.01 |
| `fp8` / FLASHINFER_MLA | 32 / 32 | 3,389 | 52.17 | 64.96 |

Before per-query DCP causal localization, NVFP4 scored 25/32 and reached only
6.93 output tok/s, so the correction addresses both wrong masking and a large
part of the observed slowdown. Native Triton is nevertheless 4.27x slower than
FP8/Tokenspeed and 3.97x slower than FP8/FlashInfer on this batched DSpark
workload, so it is not yet performance competitive.

With the real-weight batch-8 configuration, vLLM reported 617,378 cache tokens
for NVFP4 versus 600,250 for FP8 (+2.85% end-to-end capacity). A simpler dummy
configuration reported 722,199 versus 679,191 (+6.3%). Hybrid recurrent-state
groups dilute the attention-cache saving; the attention-state payload itself
is 392 versus 576 bytes per token per layer. Kernel profiling/tuning and a
production-compatible CuTe DSL port remain required before enablement.
