# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-architecture policy: Rubin (compute capability 10.7) with a CuTe DSL that
targets ``sm_107a``. Automatic and strict: no switch, no fallback; any mismatch raises
at construction with the fix.

Re-implements FlashInfer ``shim/dependencies.py::require_sm107_dsl`` and the
``flashinfer.cute_dsl.availability._dsl_captured_arch`` probe without importing
FlashInfer (hashes in VENDOR.md).
"""

from __future__ import annotations

import os
import sys

import torch

_RUBIN_CC = (10, 7)
_ACCEPTED_TARGETS = ("sm_107", "sm_107a")


def captured_dsl_arch() -> str | None:
    """The target the CuTe DSL captured when ``cutlass`` was first imported, or None if
    the (internal) attribute is unavailable in the installed DSL."""
    try:
        from cutlass.cutlass_dsl import CuTeDSL

        arch = str(CuTeDSL._get_dsl().envar.arch)
    except Exception:
        return None
    return arch.split(".")[-1]  # tolerate an "Arch.sm_107a" spelling


def require_sm107() -> None:
    """Raise unless this process can build and run the sm_107a Rubin MegaMoE kernels."""
    if not torch.cuda.is_available():
        raise RuntimeError("the Rubin MegaMoE backend needs a CUDA device.")
    cc = torch.cuda.get_device_capability()
    if cc != _RUBIN_CC:
        raise RuntimeError(
            f"the Rubin MegaMoE backend requires compute capability {_RUBIN_CC} "
            f"(VR200), this device reports {cc}."
        )
    if "cutlass" not in sys.modules:
        # The DSL captures its target at first import; default it to the device's.
        # A launcher-provided value is kept and checked through the captured target.
        os.environ.setdefault("CUTE_DSL_ARCH", "sm_107a")
    try:
        from cutlass.base_dsl.arch import Arch
        from cutlass.utils import rubin_helpers

        Arch["sm_107a"]
        if not callable(
            getattr(rubin_helpers, "make_blockscaled_trivial_tiled_mma", None)
        ):
            raise AttributeError("missing Rubin block-scaled MMA helper")
    except (ImportError, KeyError, AttributeError) as exc:
        raise RuntimeError(
            "the Rubin MegaMoE backend requires a CuTe DSL with native sm_107a and "
            "cutlass.utils.rubin_helpers (nvidia-cutlass-dsl 4.8.0 or a compatible "
            "build)."
        ) from exc
    captured = captured_dsl_arch()
    if captured is None:
        raise RuntimeError(
            "cannot read the CuTe DSL's captured target "
            "(CuTeDSL._get_dsl().envar.arch) from the installed cutlass; the Rubin "
            "MegaMoE backend refuses to compile "
            "without knowing it targets sm_107a."
        )
    if captured not in _ACCEPTED_TARGETS:
        raise RuntimeError(
            f"the CuTe DSL already captured target {captured!r}; the Rubin MegaMoE "
            "kernels need sm_107a. Export CUTE_DSL_ARCH=sm_107a in the launcher before "
            "Python starts (vLLM imports the DSL early on VR)."
        )
