# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 DEP decode backend (DP attention, EP MegaMoE, online MXFP8 projections),
enabled by ``additional_config={"dep_decode_backend": "fused"}``.

Decode-sized batches (up to ``gemm.MAX_TILE_N`` tokens per rank) run the MoE block's
projections as fused stream-K MXFP8 GEMMs (``moe.py``); everything else, and every
larger batch, stays on the stock path.
"""


def dep_decode_enabled(vllm_config) -> bool:
    """Whether the deployment selects the DEP decode backend.

    Raises when it is selected for a deployment it does not support."""
    extra = vllm_config.additional_config
    backend = (
        extra.get("dep_decode_backend", "auto") if isinstance(extra, dict) else "auto"
    )
    if backend == "auto":
        return False
    if backend != "fused":
        raise ValueError("Supported dep_decode_backend values are auto/fused")
    from vllm.platforms import current_platform

    parallel = vllm_config.parallel_config
    reasons = []
    if not current_platform.is_device_capability_family(100):
        reasons.append("a GPU other than SM100/SM103")
    if parallel.tensor_parallel_size != 1 or not parallel.enable_expert_parallel:
        reasons.append("anything but TP1 attention with expert parallelism")
    if vllm_config.kernel_config.moe_backend != "deep_gemm_mega_moe":
        reasons.append("an MoE backend other than deep_gemm_mega_moe")
    if reasons:
        raise ValueError("Kimi K3 DEP decode does not support " + ", ".join(reasons))
    return True
