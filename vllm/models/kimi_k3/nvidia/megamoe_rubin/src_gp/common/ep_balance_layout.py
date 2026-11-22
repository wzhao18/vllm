# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Served EP-balance ring slot layout (int64.3 item 3), shared by the dedup router (global per-expert counts) and the
genphase token comm (this rank's local per-expert counts); host side megamoe_rubin/ep_balance.py uses the same values.

Slot (int32 words): [0] call id (1-based, written last), [1..2] t_ready %globaltimer (lo, hi), [3] entries, [4] world
size, [5] experts per rank, [6] 0 = global record, local_rank + 1 = this rank's local experts only, [7] global record:
epbal_checksum(call, sum(counts)) (written without a system fence, so the host checks it); local record: 0 (fenced),
[8 ..] per-expert token counts. One call in EPBAL_PERIOD writes slot (call >> EPBAL_SHIFT) % EPBAL_SLOTS."""

EPBAL_SHIFT = 6
EPBAL_PERIOD = 1 << EPBAL_SHIFT
EPBAL_SLOTS = 256
EPBAL_HEADER_WORDS = 8
EPBAL_MAX_EXPERTS = 512
EPBAL_SLOT_WORDS = EPBAL_HEADER_WORDS + EPBAL_MAX_EXPERTS
EPBAL_CHECKSUM_SALT = 0x5BD1E995
EPBAL_CHECKSUM_MUL = 65599


def epbal_checksum(call: int, total: int) -> int:
    """Word 7 of a global record, int32 arithmetic as on the device: (sum ^ SALT) + call * MUL, wrapped."""
    v = (((total & 0xFFFFFFFF) ^ EPBAL_CHECKSUM_SALT) + (call & 0xFFFFFFFF) * EPBAL_CHECKSUM_MUL) & 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v
