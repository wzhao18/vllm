# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Served EP-balance ring slot layout (int64.3 item 3), shared by the dedup router (global per-expert counts) and the
genphase token comm (this rank's local per-expert counts); host side megamoe_rubin/ep_balance.py uses the same values.

Slot (int32 words): [0] call id (1-based, written last), [1..2] t_ready %globaltimer (lo, hi), [3] entries, [4] world
size, [5] experts per rank, [6] 0 = global record, local_rank + 1 = this rank's local experts only,
[7] epbal_checksum(call, counts) (every record: the ring is in device memory, written with gpu-scope stores and no
fence, and the host copies slots out while kernels run, so it checks the checksum), [8 ..] per-expert token counts. One call in EPBAL_PERIOD writes slot (call >> EPBAL_SHIFT) % EPBAL_SLOTS."""

EPBAL_SHIFT = 6
EPBAL_PERIOD = 1 << EPBAL_SHIFT
EPBAL_SLOTS = 256
EPBAL_HEADER_WORDS = 8
EPBAL_MAX_EXPERTS = 512
EPBAL_SLOT_WORDS = EPBAL_HEADER_WORDS + EPBAL_MAX_EXPERTS
EPBAL_CHECKSUM_SALT = 0x5BD1E995
EPBAL_CHECKSUM_MUL = 65599
EPBAL_POS_MUL = 0x9E3779B1
EPBAL_POS_ADD = 0x7F4A7C15
EPBAL_POS_MUL_S32 = EPBAL_POS_MUL - (1 << 32)  # the same bits as a signed int32 constant for the device


def epbal_position_weight(i: int) -> int:
    """Odd pseudo-random uint32 weight of expert i: (i * POS_MUL + POS_ADD) | 1, wrapped."""
    return ((i * EPBAL_POS_MUL + EPBAL_POS_ADD) & 0xFFFFFFFF) | 1


def epbal_checksum(call: int, counts) -> int:
    """Word 7 of a global record, in wrapping int32 as on the device: S = sum_i count_i * weight(i) (position
    sensitive: two calls with the same tokens x top-k but different routing differ), then (S ^ SALT) + call * MUL
    (bound to the call id)."""
    acc = 0
    for i, c in enumerate(counts):
        acc = (acc + (c & 0xFFFFFFFF) * epbal_position_weight(i)) & 0xFFFFFFFF
    v = ((acc ^ EPBAL_CHECKSUM_SALT) + (call & 0xFFFFFFFF) * EPBAL_CHECKSUM_MUL) & 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v
