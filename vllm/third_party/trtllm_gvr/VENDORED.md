# Vendored TensorRT-LLM Blackwell top-k kernels

- Source: <https://github.com/NVIDIA/TensorRT-LLM>, commit
  `fac0f146862adfc4626302ed1cddce39374c6110` (merge of PR #19285, "Support FP32,
  BF16, and FP16 in unified self-sampling GVR templates").
- License: Apache-2.0 (headers kept in every file).
- Files: `tensorrt_llm/_torch/kernels/blackwell/utils.py` and
  `tensorrt_llm/_torch/kernels/blackwell/top_k/*.py`, copied to
  `_torch/kernels/blackwell/` here. The tree keeps TensorRT-LLM's layout so the
  kernels' relative imports resolve unchanged.
- Local changes (only these two import lines):
  - `top_k/gvr_topk_decode_dispatch.py`: `from tensorrt_llm.logger import logger`
    → `from .....logger import logger` (`logger.py` here forwards to vLLM's logger).
  - `top_k/filtered_top_k_prefill_varlen.py`:
    `from tensorrt_llm._torch.kernels.blackwell.utils import (` → `from ..utils import (`.
- Requires `nvidia-cutlass-dsl` 4.8 (TensorRT-LLM's pin at that commit).
- Used by `sparse_indexer_topk_backend="trtllm_gvr2"`
  (`vllm/model_executor/layers/indexer_topk.py`): `run_prefill` / `run_varlen` /
  `warmup_*` of `top_k/gvr_topk_decode_self_sampling_host.py`.
- Drop this copy once FlashInfer's `topk_varlen` gvr_2 port reads 16-bit logits.
