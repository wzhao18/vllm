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
and average 20 decode iterations. Cache construction and persistent V repack
are outside the timed region.

## Triton native FP4 QK/PV

The prepacked-V kernel compiles and executes correctly on SM103. Validation at
context length 128 against a dequantized reference produced:

- output relative L2 error: `0.00134254`
- output cosine similarity: `0.99999905`
- output maximum absolute error: `0.00012090`
- probability maximum absolute error: `0.00125946`

| Context tokens | TMA disabled (ms) | TMA enabled (ms) |
|---:|---:|---:|
| 128 | 0.2808 | 0.3228 |
| 1,024 | 0.3348 | 0.3453 |
| 4,096 | 0.3518 | 0.4035 |
| 16,384 | 0.3534 | 0.3855 |
| 65,536 | 0.3745 | 0.3945 |

TMA was not beneficial for this batch-1 shape. More batch sizes and an FP8
backend comparison are still required.

The on-the-fly V transpose variant consistently fails in the PV kernel with a
CUDA `misaligned address`, with both TMA enabled and disabled. The prepacked-V
variant avoids that path, but its persistent V view costs another 256 bytes per
token per layer. We should fix the SM103 inline V path or use a fused-V kernel
before treating this as the final memory-efficient implementation.

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

The benchmark script intentionally loads the MR source from an external source
directory so no TensorRT-LLM implementation is silently vendored into vLLM:

```bash
python benchmarks/kernels/benchmark_kimi_k3_nvfp4_mla.py \
  --source-dir /path/to/trtllm-mr10576/fp4_mla \
  --batch 1 --heads 128 --context 128 --warmup 5 --iters 20 --validate
```

Set `TRTLLM_FP4_MLA_TRITON_PREPACK_V=1` for the currently working SM103 path.
