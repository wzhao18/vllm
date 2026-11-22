# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Served EP-balance ring slot layout (int64.3 item 3), shared by the dedup router (global per-expert counts) and the
genphase token comm (this rank's local per-expert counts); host side megamoe_rubin/ep_balance.py uses the same values.

Slot (int32 words): [0] call id (1-based, written last), [1..2] t_ready %globaltimer (lo, hi), [3] entries, [4] world
size, [5] experts per rank, [6] 0 = global record, local_rank + 1 = this rank's local experts only, [7] 0,
[8 ..] per-expert token counts. One call in EPBAL_PERIOD writes slot (call >> EPBAL_SHIFT) % EPBAL_SLOTS."""

EPBAL_SHIFT = 6
EPBAL_PERIOD = 1 << EPBAL_SHIFT
EPBAL_SLOTS = 256
EPBAL_HEADER_WORDS = 8
EPBAL_MAX_EXPERTS = 512
EPBAL_SLOT_WORDS = EPBAL_HEADER_WORDS + EPBAL_MAX_EXPERTS
