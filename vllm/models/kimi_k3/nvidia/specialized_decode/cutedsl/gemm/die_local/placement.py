# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weights placed on the two dies of a VR200 by halves: the first ceil(chunks / 2)
2 MiB chunks of a BF16 [N, K] weight homed on die 0, the rest on die 1
(``dies.localized_like``); the die-local GEMM hands each die's tiles to that die's
SMs. Values and the tensor's shape / strides do not change, so every other reader
keeps working (cuBLAS reading the halves is ~4% slower at the decode shapes)."""

import torch

from ...common import dies

CHUNK = dies.CHUNK


def halves_domains(n: int, k: int) -> list[int]:
    """Die of each 2 MiB chunk of a BF16 [n, k] weight."""
    chunks = -(-(n * k * 2) // CHUNK)
    return [int(2 * c >= chunks) for c in range(chunks)]


def tile_split(n: int, k: int, tile_rows: int) -> int:
    """Tiles of ``tile_rows`` weight rows on die 0: those whose middle byte lies in
    die 0's chunks (each tile's majority is on its die)."""
    tiles = -(-n // tile_rows)
    tile_bytes = tile_rows * k * 2
    boundary = (-(-(n * k * 2) // CHUNK) + 1) // 2 * CHUNK
    return min(tiles, max(0, -(-(boundary - tile_bytes // 2) // tile_bytes)))


def relocate_halves(tensor: torch.Tensor) -> torch.Tensor:
    """A copy of a contiguous BF16 [N, K] ``tensor`` placed on the dies by halves."""
    assert tensor.is_contiguous() and tensor.dim() == 2 and tensor.element_size() == 2
    return dies.localized_like(tensor, halves_domains(*tensor.shape))


def relocate_parameter_(param: torch.nn.Parameter) -> torch.Tensor:
    """Move ``param``'s data to ``relocate_halves`` memory in place (same values; the
    old storage is released to the allocator); returns the new data. Idempotent."""
    if not getattr(param, "on_dies", False):
        param.data = relocate_halves(param.data)
        param.on_dies = True
    return param.data
