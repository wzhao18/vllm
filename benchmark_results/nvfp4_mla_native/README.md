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
zero. Native cache insertion produced the expected quantization error:

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
  --batch 1 --heads 128 --context 128 --warmup 5 --iters 20 --validate
```

The integrated SM103 path selects `BLOCK_V=128` automatically. It can be
overridden with `VLLM_KIMI_K3_NVFP4_BLOCK_V` for tuning.
