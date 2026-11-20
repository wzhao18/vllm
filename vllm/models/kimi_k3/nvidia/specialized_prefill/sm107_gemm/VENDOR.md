# Vendoring record: specialized_prefill/sm107_gemm

## Upstream

- **Source**: FlashInfer main `855eece` (Rubin checkout), `flashinfer/gemm/kernels/`:
  `dense_blockscaled_gemm_sm107.py` (the CuTe DSL persistent dense block-scaled GEMM for
  sm_107, the same kernel as TensorRT-LLM `tensorrt_llm/_torch/kernels/rubin/
  dense_blockscaled_gemm_persistent.py` from PR 21996), its parents
  `dense_blockscaled_gemm_sm100.py` and `dense_blockscaled_gemm_sm100_common.py`, and
  `epilogue_utils.py`. Licenses as in each file header.
- **Entry point**: `packed_situ_gemm.PackedSituGemm` (vLLM).

## Local differences

- **R1** (import rewrite): `dense_blockscaled_gemm_sm107.py` imports `.epilogue_utils` instead of
  `flashinfer.gemm.kernels.epilogue_utils`.
- **S1** (packed SiTU epilogue, vLLM): `Sm107BlockScaledPersistentDenseGemmKernel.__call__` takes
  `situ_q_ptr`, `situ_sf_ptr`, `situ_q_ld`, `situ_sf_pitch` and `situ_k`; with them the GEMM's N
  tiles `[0, situ_n_lat_tiles)` store BF16 through the TMA epilogue into C (row pitch `c_ld`),
  and the remaining tiles (interleaved shared gate/up rows) run
  `situ_epilogue.epilogue_situ_mxfp8`. The tile scheduler covers the full packed N. Without the
  new arguments the kernel is unchanged.

Everything else is byte-identical to upstream.
