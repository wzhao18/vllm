# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Point DeepGEMM's JIT at an include tree with some headers replaced.

DeepGEMM JIT-compiles its kernels from the headers under its library root.
``install`` builds ``<cache>/dg_overlay_<hash>/include``: a mirror of the stock
include tree (symlinks) in which the given ``deep_gemm/...`` headers are
replaced, and re-initializes DeepGEMM with it. The JIT cache key covers header
contents, so stock and overlay kernels never collide. Install once per process,
before the first launch of an affected kernel is compiled.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

_installed: dict[str, Path] = {}


def _digest(stock_root: Path, headers: dict[str, Path]) -> str:
    # The mirror's other entries link into ``stock_root``: two DeepGEMM installs
    # sharing one cache root must not share a mirror (the second would compile its
    # stock kernels against the first install's headers).
    h = hashlib.sha256(str(stock_root.resolve()).encode())
    for rel in sorted(headers):
        h.update(rel.encode())
        h.update(Path(headers[rel]).read_bytes())
    return h.hexdigest()[:16]


def _mirror(src: Path, dst: Path, replace: dict[str, Path], rel: str) -> None:
    """Mirror ``src`` into ``dst`` with symlinks; recurse only into directories
    that contain a replaced header."""
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        sub = f"{rel}/{entry.name}" if rel else entry.name
        target = dst / entry.name
        if sub in replace:
            target.symlink_to(Path(replace[sub]).resolve())
        elif entry.is_dir() and any(r.startswith(sub + "/") for r in replace):
            _mirror(entry, target, replace, sub)
        else:
            target.symlink_to(entry.resolve())


def overlay_root(stock_root: Path, cache_root: Path, headers: dict[str, Path]) -> Path:
    """``<cache_root>/dg_overlay_<hash>`` whose ``include/`` is the stock include
    tree with ``headers`` (paths relative to ``include/``) replaced."""
    root = cache_root / f"dg_overlay_{_digest(stock_root, headers)}"
    if (root / "include").is_dir():
        return root
    stock_include = stock_root / "include"
    for rel in headers:
        if not (stock_include / rel).is_file():
            raise FileNotFoundError(f"DeepGEMM has no header {rel!r} to replace")
    tmp = cache_root / f".{root.name}.{os.getpid()}"
    _mirror(stock_include, tmp / "include", dict(headers), "")
    try:
        tmp.rename(root)
    except OSError:  # another process installed it first
        pass
    return root


def install(deep_gemm, headers: dict[str, Path], cache_root: Path | None = None) -> Path:
    """Re-initialize ``deep_gemm`` on an overlay with ``headers`` replaced."""
    stock_root = Path(deep_gemm.__file__).resolve().parent
    if cache_root is None:
        from vllm import envs

        cache_root = Path(envs.VLLM_CACHE_ROOT) / "kimi_k3_megamoe"
    cache_root.mkdir(parents=True, exist_ok=True)
    root = overlay_root(stock_root, cache_root, headers)
    key = str(stock_root)
    if _installed.get(key) != root:
        deep_gemm._C.init(str(root))
        _installed[key] = root
    return root
