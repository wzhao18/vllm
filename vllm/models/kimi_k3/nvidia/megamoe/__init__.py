# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 MegaMoE (DeepGEMM FP8 x FP4 SiTU) wide-EP decode tuning.

``prepare_megamoe`` runs once per worker after the weights are loaded, before
the DeepGEMM warmup compiles the kernels: it JIT-compiles MegaMoE from the header
that ``situ_patch`` derives from the installed stock header (bitwise-identical
output, -5..-12% per launch at EP16/EP32 per-rank shapes). Capability fallback: the
stock kernel, with a warning, when the installed DeepGEMM's header does not have the
expected text.

Not selected: expert localization (MLOPart; each die streams its half of N from its
own HBM, bitwise identical). Full K3.1 DEP16 on VR: -0.57..-0.65 ms per step on one
4-node block, but the KV pool is 0.1-3% smaller per GPU and varies from one start to
the next, EPLB cannot move the localized weights, and on a shared DEP4 node 4 of 8
localized processes had cases 17-25% slower; no serving recipe enabled it. Restore it
with a measured per-deployment policy (DeepGEMM ``localize`` on ``w13_weight`` /
``w2_weight``, ``destroy_localizer`` after).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)



def install_patched_header(deep_gemm, cache_root: Path | None = None) -> Path | None:
    """Re-initialize ``deep_gemm`` on an include tree with the Kimi K3.1 MegaMoE
    header (derived from the installed stock one). Returns the overlay root, or
    ``None`` when the stock header cannot be patched (the stock kernel stays)."""
    from .overlay import install
    from .situ_patch import SCHEDULER_HEADER, STOCK_HEADER, PatchError, patch_situ_header

    include = Path(deep_gemm.__file__).resolve().parent / "include"
    stock = include / STOCK_HEADER
    try:
        sched = include / SCHEDULER_HEADER
        text = patch_situ_header(
            stock.read_text(), scheduler_text=sched.read_text() if sched.is_file() else None
        )
    except (OSError, PatchError) as e:
        logger.warning("Kimi K3 MegaMoE: stock kernel kept, cannot patch %s: %s", stock, e)
        return None
    if cache_root is None:
        from vllm import envs

        cache_root = Path(envs.VLLM_CACHE_ROOT) / "kimi_k3_megamoe"
    cache_root.mkdir(parents=True, exist_ok=True)
    # Content-addressed, so concurrent ranks write identical bytes
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    header = cache_root / f"sm100_fp8_fp4_mega_moe_situ.k31.{digest}.cuh"
    if not header.is_file():
        tmp = header.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(text)
        tmp.replace(header)
    return install(deep_gemm, {STOCK_HEADER: header}, cache_root)


def prepare_megamoe(model: torch.nn.Module, vllm_config) -> None:
    """Install the Kimi K3.1 MegaMoE kernel header for every ``DeepGemmMegaMoEExperts``
    (FP8 x FP4 SiTU) of ``model``."""
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
        DeepGemmMegaMoEExperts,
    )

    experts = [m for m in model.modules() if isinstance(m, DeepGemmMegaMoEExperts)]
    if not experts:
        return
    from vllm.utils.deep_gemm import _import_deep_gemm

    deep_gemm = _import_deep_gemm()
    if deep_gemm is None:
        return
    fmt = experts[0].weight_format
    if fmt.kernel == "fp8_fp4_mega_moe" and experts[0].activation.name == "situ":
        root = install_patched_header(deep_gemm)
        if root is not None:
            logger.info_once("Kimi K3 MegaMoE: K3.1 wide-EP kernel header (%s)", root)
