# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone Kimi K3.1 routed-MoE backend for VR200 (Rubin, sm_107a).

Vendors FlashInfer 0.7.3.dev20261031's sm107 CuTe DSL MegaMoE kernels (``src/sources``,
byte-identical, see VENDOR.md) and runs them on torch symmetric memory instead of
NVSHMEM; no runtime dependency on flashinfer or nvshmem4py.

Exports are lazy: importing this package does not import cutlass, cuda-python or triton.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    # config.py (CPU-safe)
    "RubinMoeProblem": "config",
    "RubinKnobs": "config",
    "RubinBucket": "config",
    "RubinMoeConfig": "config",
    "KnobLookupError": "config",
    "RouteIdError": "config",
    "resolve_knobs": "config",
    "make_buckets": "config",
    "default_capacity": "config",
    "tuned_capacity": "config",
    "bench_capacity": "config",
    "nvfp4_norm_const_from_amax": "config",
    "merge_knob_files": "config",
    "SHIPPED_KNOBS": "config",
    # weights.py (CPU-safe)
    "RubinMoeWeights": "weights",
    "NonRepresentableError": "weights",
    "ConversionReport": "weights",
    "scan_mxfp4_to_nvfp4": "weights",
    "convert_mxfp4_to_nvfp4": "weights",
    "prepare_weights": "weights",
    "parse_expert_key": "weights",
    # model integration (moe_backend="kimi_k3_mega_moe")
    "KimiK3MegaMoEExperts": "experts",
    "prepare_rubin_megamoe": "experts",
    # GPU runtime
    "RubinMegaMoE": "runtime",
    "require_sm107": "arch",
    "SymmetricArena": "symm_arena",
    "stage_inputs": "staging",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{module}"), name)
