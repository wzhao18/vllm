# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 MegaMoE (DeepGEMM FP8 x FP4 SiTU) wide-EP decode tuning.

``prepare_megamoe`` runs once per worker after the weights are loaded, before
the DeepGEMM warmup compiles the kernels:

- ``VLLM_K3_MEGAMOE_PATCH`` (default 1): JIT-compile MegaMoE from the header that
  ``situ_patch`` derives from the installed stock header (bitwise-identical output,
  -5..-12% per launch at EP16/EP32 per-rank shapes). Falls back to the stock kernel, with a warning, when the
  installed DeepGEMM's header does not have the expected text.
- ``VLLM_K3_MEGAMOE_LOCALIZE`` (default 1): place each rank's routed expert weights
  in the GPU's two locality domains (MLOPart), so every die streams its half of N
  from its own HBM (bitwise-identical output). Full K3.1 DEP16 on VR: -0.58..-0.66 ms
  per step (MegaMoE 119.7 -> 113.7 us/launch at 128 rows), 5 of 5 processes clean,
  KV pool 0.1-0.4% smaller (the MLOPart weights live outside the PyTorch
  allocator; vLLM's profile sees them as non-torch memory). Falls back to the plain
  placement without MLOPart or under EPLB (the weights change shape to
  ``(2, E, N / 2, K)``, which EPLB cannot move). Caveat: on a shared DEP4 node
  (224 experts/rank) 4 of 8 localized processes had cases 17-25% slower (one rank's
  drafter MLA decode ~1.4x slower) while other jobs used the same GPUs.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCH = os.environ.get("VLLM_K3_MEGAMOE_PATCH", "1") == "1"
_LOCALIZE = os.environ.get("VLLM_K3_MEGAMOE_LOCALIZE", "1") == "1"


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


def _localize_experts(experts, deep_gemm) -> None:
    """Routed weights ``(E, N, K)`` -> ``(2, E, N / 2, K)``, one N half per locality
    domain; the block scales stay as they are."""
    for name in ("w13_weight", "w2_weight"):
        w = getattr(experts, name)
        if not deep_gemm.is_localized(w.data):
            w.data = deep_gemm.localize(w.data)


def prepare_megamoe(model: torch.nn.Module, vllm_config) -> None:
    """Apply the env-selected MegaMoE tuning to every ``DeepGemmMegaMoEExperts``."""
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
        DeepGemmMegaMoEExperts,
    )

    experts = [m for m in model.modules() if isinstance(m, DeepGemmMegaMoEExperts)]
    if not experts or not (_PATCH or _LOCALIZE):
        return
    from vllm.utils.deep_gemm import _import_deep_gemm

    deep_gemm = _import_deep_gemm()
    if deep_gemm is None:
        return
    fmt = experts[0].weight_format
    if _PATCH and fmt.kernel == "fp8_fp4_mega_moe" and experts[0].activation.name == "situ":
        root = install_patched_header(deep_gemm)
        if root is not None:
            logger.info_once("Kimi K3 MegaMoE: K3.1 wide-EP kernel header (%s)", root)
    if _LOCALIZE:
        if vllm_config.parallel_config.enable_eplb:
            logger.info_once("Kimi K3 MegaMoE: routed expert weights not localized (EPLB)")
            return
        if not deep_gemm.is_localization_available():
            logger.info_once("Kimi K3 MegaMoE: routed expert weights not localized (no MLOPart)")
            return
        for e in experts:
            _localize_experts(e, deep_gemm)
            torch.cuda.empty_cache()
        deep_gemm.destroy_localizer()
        logger.info_once(
            "Kimi K3 MegaMoE: localized the routed expert weights of %d layers", len(experts)
        )
