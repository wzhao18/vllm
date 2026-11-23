# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spin-guard timeouts, applied at RUNTIME (int64.3 item 1).

The CuTe DSL spin guards read their bound from a module global (guard_global.py) that the host patches after every
kernel load, so a retune here changes no kernel source and no compile-cache key: this module is imported by the
patcher only, never while a kernel is traced. Every non-MegaMoE guard must time out before MegaMoE's (a stuck rank
then reports its own wait before healthy peers trap in MegaMoE; test_k31_spin_guard_order.py)."""

GUARD_NS = 600 * 1000 * 1000 * 1000  # k31_spin_guard (PIDX, KDA, latent MoE tail, decode fabric)
MEGAMOE_GUARD_NS = 900 * 1000 * 1000 * 1000  # MegaMoE cross-rank waits

VALUES = {"k31_guard_ns": GUARD_NS, "k31_megamoe_guard_ns": MEGAMOE_GUARD_NS}
