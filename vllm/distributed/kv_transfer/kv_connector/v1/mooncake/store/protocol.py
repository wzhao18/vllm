# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire-format constants for the LookupKey ZMQ admin channel.

This is the single source of truth shared by ``LookupKeyClient`` and
``LookupKeyServer`` on the scheduler<->worker rank-0 admin channel.

Wire format (REQ/REP over IPC):

    Request: [msg_type: bytes] [payload_frames...]

      msg_type == LOOKUP_MSG:
          frame 1: num_tokens (u32 big-endian, 4 bytes); the worker derives
                   the aligned lookup length
          frame 2: hash_len (u16 big-endian, 2 bytes) — byte length of each
                   fixed-size block hash (0 when there are no hashes)
          frame 3: raw block hashes concatenated back-to-back (each hash_len
                   bytes); the server splits on hash_len
        Response:
          frame 0: hit_count (u32 big-endian, 4 bytes)
          frame 1: zero or more 12-byte load boundaries. Each entry is
                   group_id (u32), chunk_id (u32), boundary_tokens (u32).

      msg_type == RESET_MSG:
          (no payload frames)
        Response: [RESP_OK] or [RESP_ERR]

The first frame of every request is a named bytes tag (not a numeric
sentinel that aliases the data field) so the protocol stays
self-describing and extensible: adding new admin commands requires
only a new tag and a new dispatch branch.

Mirrors the named-tag convention used by the NIXL connector (see
``vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py``).
"""

from collections.abc import Sequence

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    LoadBoundary,
    MooncakeLookupResult,
)

# Request message-type tags. Frame 0 of every request.
LOOKUP_MSG: bytes = b"lookup"
RESET_MSG: bytes = b"reset"

# Single-byte response status codes for admin commands.
RESP_OK: bytes = b"\x01"
RESP_ERR: bytes = b"\x00"

# group_id (u32), chunk_id (u32), boundary_tokens (u32).
LOAD_BOUNDARY_ENTRY_SIZE: int = 12


def encode_lookup_response(result: MooncakeLookupResult) -> tuple[bytes, bytes]:
    payload = bytearray()
    for boundary in result.load_boundaries:
        payload.extend(boundary.group_id.to_bytes(4, "big"))
        payload.extend(boundary.chunk_id.to_bytes(4, "big"))
        payload.extend(boundary.num_tokens.to_bytes(4, "big"))
    return result.hit_length.to_bytes(4, "big"), bytes(payload)


def decode_lookup_response(frames: Sequence[bytes]) -> MooncakeLookupResult:
    hit_length = int.from_bytes(frames[0], "big")
    if len(frames) == 1 or not frames[1]:
        return MooncakeLookupResult(hit_length)

    payload = frames[1]
    if len(payload) % LOAD_BOUNDARY_ENTRY_SIZE != 0:
        raise ValueError("Invalid Mooncake lookup response")
    boundaries = []
    for offset in range(0, len(payload), LOAD_BOUNDARY_ENTRY_SIZE):
        boundaries.append(
            LoadBoundary(
                group_id=int.from_bytes(payload[offset : offset + 4], "big"),
                chunk_id=int.from_bytes(payload[offset + 4 : offset + 8], "big"),
                num_tokens=int.from_bytes(payload[offset + 8 : offset + 12], "big"),
            )
        )
    return MooncakeLookupResult(hit_length, tuple(boundaries))
