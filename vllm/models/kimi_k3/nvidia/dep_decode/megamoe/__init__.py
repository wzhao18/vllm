# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DEP decode's MegaMoE kernel: DeepGEMM's FP8 x FP4 SiTU MegaMoE with a faster
combine, decode-shaped scheduling and programmatic dependent launch (see the header in
this directory).

DeepGEMM JIT-compiles its kernels from the headers under its library root. ``install``
points that root at an overlay where every header is the stock one except this
kernel's, so only MegaMoE changes; the JIT cache is keyed by header contents, so stock
and DEP builds never collide. ``mega_moe`` runs a stock ``DeepGemmMegaMoEExperts``
layer (symmetric buffers, input preparation) with PDL on for the kernel launch.
"""

import hashlib
import os
from pathlib import Path

import torch

HEADER = "sm100_fp8_fp4_mega_moe_situ.cuh"
_installed: set[str] = set()


def _overlay_root(stock_root: Path, cache_root: Path) -> Path:
    """``<cache>/dep_megamoe_<hash>``, whose ``include/`` mirrors the stock tree
    (symlinks) except HEADER."""
    header = Path(__file__).with_name(HEADER)
    digest = hashlib.sha256(header.read_bytes()).hexdigest()[:16]
    root = cache_root / f"dep_megamoe_{digest}"
    if (root / "include").is_dir():
        return root
    tmp = cache_root / f".dep_megamoe_{digest}.{os.getpid()}"
    stock_include = stock_root / "include"
    for entry in stock_include.iterdir():
        if entry.name != "deep_gemm":
            (tmp / "include").mkdir(parents=True, exist_ok=True)
            (tmp / "include" / entry.name).symlink_to(entry.resolve())
    for sub in (stock_include / "deep_gemm").iterdir():
        dst = tmp / "include" / "deep_gemm" / sub.name
        if sub.name != "impls":
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(sub.resolve())
            continue
        dst.mkdir(parents=True)
        for impl in sub.iterdir():
            (dst / impl.name).symlink_to(
                header.resolve() if impl.name == HEADER else impl.resolve()
            )
    try:
        tmp.rename(root)
    except OSError:  # another process installed it first
        pass
    return root


def install(deep_gemm) -> None:
    """Point DeepGEMM's JIT at the overlay (once per process, before any MegaMoE
    launch is compiled)."""
    from vllm import envs

    stock_root = Path(deep_gemm.__file__).resolve().parent
    if str(stock_root) in _installed:
        return
    cache_root = Path(envs.VLLM_CACHE_ROOT) / "dep_decode"
    cache_root.mkdir(parents=True, exist_ok=True)
    deep_gemm._C.init(str(_overlay_root(stock_root, cache_root)))
    _installed.add(str(stock_root))


def mega_moe(experts, deep_gemm, hidden_states: torch.Tensor, topk_weights, topk_ids):
    """``experts(hidden_states, topk_weights, topk_ids)`` with the MegaMoE kernel
    launched under PDL."""
    previous = deep_gemm.get_pdl()
    deep_gemm.set_pdl(True)
    try:
        return experts(hidden_states, topk_weights, topk_ids)
    finally:
        deep_gemm.set_pdl(previous)
