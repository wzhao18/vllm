# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``vllm.envs`` plus the Kimi K3 decode variables this package reads that the
base ``vllm.envs`` does not define.

Each local variable is parsed exactly as the source branch's ``vllm/envs.py``
entry parses it, and read on every access (as ``vllm.envs`` does before its
cache is enabled). Every other attribute is ``vllm.envs``'s.
"""

import os
from collections.abc import Callable
from typing import Any

import vllm.envs as _envs

_LOCAL: dict[str, Callable[[], Any]] = {
    # Kimi K3 on Rubin: DS-T2P (mla.core_p2) runs the q 8 verify work list at
    # batch 128-256 instead of core_q's polled core. Off by default.
    "VLLM_KIMI_K3_DS_T2P": lambda: bool(int(os.getenv("VLLM_KIMI_K3_DS_T2P", "0"))),
    # Kimi K3 on Rubin, with VLLM_KIMI_K3_DS_T2P: DS-T2P runs lockstep page-range
    # chunks (LPC, mla.dcp.lpc_table) at the q 8 verify rows LPC_ROWS. Off by
    # default.
    "VLLM_KIMI_K3_DS_LPC": lambda: bool(int(os.getenv("VLLM_KIMI_K3_DS_LPC", "0"))),
    # The routed MoE on token shards over a replicated stream (prefill side; the
    # decode only reads it to refuse sequence-parallel decode under it).
    "VLLM_KIMI_K3_MOE_TOKEN_SHARD": lambda: bool(
        int(os.getenv("VLLM_KIMI_K3_MOE_TOKEN_SHARD", "0"))
    ),
}


def __getattr__(name: str) -> Any:
    if name in _LOCAL:
        return _LOCAL[name]()
    return getattr(_envs, name)


def __dir__() -> list[str]:
    return sorted(set(dir(_envs)) | set(_LOCAL))
