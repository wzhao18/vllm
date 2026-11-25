# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drafter aux hidden states kept on their sequence shards.

Under sequence parallelism the target model ends with one all-gather of
``[hidden | aux_1 .. aux_n]`` so every TP rank holds every row's aux states, and
the drafter then projects all rows on every rank. When the drafter supports it
(``aux_head_rows`` / ``combine_aux_rows``), a large eager step instead gathers
only the final hidden states and hands the drafter the local aux shards as
``SpLocalAux``: each rank projects its own shard's rows and all-gathers the
projected rows. Small steps keep the packed gather.

CUDA graphs: the forward is captured with ``SpLocalAux`` outputs too (the
packed gather moved 6x the bytes of the hidden-only gather: 1.23 GB vs 205 MB
per step at 2,048 rows per rank, TP8). A graph wrapper hands back weak
references of its outputs (``weak_ref_tensors``), which keeps the type through
``map_tensors``; the shard size comes from the tensors, so a replay for a
smaller batch (same capture size) reads the same rows as the capture.
"""

import torch

MIN_TOKENS = 4096


class SpLocalAux(list):
    """The aux hidden states of a step as this rank's sequence shards
    ([shard, hidden] each, rows ``[rank * shard, (rank + 1) * shard)`` of the
    step's ``num_rows`` SP-padded rows)."""

    def __init__(self, shards: list[torch.Tensor], num_rows: int) -> None:
        super().__init__(shards)
        assert shards
        self.shard = shards[0].shape[0]
        self.num_rows = num_rows

    def map_tensors(self, fn) -> "SpLocalAux":
        """The same aux states with ``fn`` applied to each shard tensor (graph
        output weak references keep the sequence-shard type)."""
        return SpLocalAux([fn(t) for t in self], self.num_rows)


def use_local_aux(num_rows: int) -> bool:
    return num_rows >= MIN_TOKENS


def local_span(start: int, end: int, shard: int, rank: int) -> tuple[int, int]:
    """Rows of ``[start, end)`` on ``rank``'s shard, as local indices [lo, hi)."""
    lo = max(start, rank * shard)
    hi = min(end, (rank + 1) * shard)
    if hi <= lo:
        return 0, 0
    return lo - rank * shard, hi - rank * shard


def head_counts(n: int, shard: int, world: int) -> list[int]:
    """Rows of the step's leading ``n`` rows on each rank's shard."""
    return [max(0, min(shard, n - r * shard)) for r in range(world)]
