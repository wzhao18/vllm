# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-local BF16 decode GEMM for two-die parts (VR200): weights placed on the dies
by halves (``placement``), each die's SMs streaming only their own die's half, K split
over a cluster with a DSMEM reduce (``cluster``); the model-facing engine
(``engine``)."""

from .engine import (
    MAX_ROWS,
    DieLocalEngine,
    DieLocalLinearMethod,
    attach_die_local,
    attach_die_local_method,
    attach_mlp_die_local,
    die_local_problem,
    place_layer_on_dies,
    run_die_local,
    two_dies,
)
from .placement import relocate_halves, relocate_parameter_
from .primitives import OUT_BF16, OUT_BF16_ADD, OUT_F32

__all__ = [
    "MAX_ROWS",
    "OUT_BF16",
    "OUT_BF16_ADD",
    "OUT_F32",
    "DieLocalEngine",
    "DieLocalLinearMethod",
    "attach_die_local",
    "attach_die_local_method",
    "attach_mlp_die_local",
    "die_local_problem",
    "place_layer_on_dies",
    "relocate_halves",
    "relocate_parameter_",
    "run_die_local",
    "two_dies",
]
