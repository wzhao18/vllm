# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The drafter's o_proj / down_proj GEMMs of 10-64 rows as two-shot producers.

``TwoShotProducer`` runs the target's tcgen05 o-projection producer
(``oproj.producer.OprojUmma``, compiled by ``oproj.prepared.compile_oproj``: 112 CTAs
of 64 output rows, weights streamed with the L2 evict_first policy and loaded before
the PDL wait, dependents released right after it) on a drafter weight [7168, K] and
pushes its BF16 rows of x @ W.T, 8 columns at a time, straight into the owners'
receive slots of the drafter's two-shot all-reduce (``common.twoshot.push``; the
polled ``draft.allreduce.ArNormTwoShot`` sites reduce them). The rows are bitwise
those of the stock GEMM (1 GPU, 56 rows, K 512 and 896: 100% equal to torch.mm), so
the site's stored-partial read and its own push are gone.
"""

import cutlass
import torch

from ..common.host import stream, to_cute

HIDDEN = 7168  # the producer's output rows (112 CTAs x 64)
ROWS = range(9, 129)  # batch rows a producer serves
# Rows of one o-projection producer launch (``OprojUmma``); above, the rows run as two
# launches of balanced halves (each pushing its tokens at their own token rows).
LAUNCH_ROWS = 64


def halves(rows):
    """(first token, rows) of each launch serving ``rows``."""
    if rows <= LAUNCH_ROWS:
        return ((0, rows),)
    first = -(-rows // 2)
    return ((0, first), (first, rows - first))


class TwoShotProducer:
    """``weight`` [7168, K] BF16 (row-major); ``rows``: the batch rows to compile
    (each above 8 and at most 64). Compile outside graph captures."""

    def __init__(self, weight, rows, tp, wide_cap=0):
        from ..oproj.prepared import compile_oproj

        if weight.dim() != 2 or weight.shape[0] != HIDDEN or not weight.is_contiguous():
            raise ValueError(f"two-shot producer weight {tuple(weight.shape)}")
        self.k, self.device = weight.shape[1], weight.device
        assert self.k % 64 == 0 and all(r in ROWS for r in rows)
        self.weight = to_cute(weight)
        with torch.accelerator.device_index(self.device.index):
            # rows above LAUNCH_ROWS push into the wide box (``wide_cap`` rows)
            self.codes = {}
            for r in rows:
                cap = 64 if r <= LAUNCH_ROWS else wide_cap
                assert cap >= r
                self.codes[r] = tuple(
                    compile_oproj("producer_umma", self.k, n, tp, 1, True, False,
                                  cap=cap, twoshot=True)
                    for _, n in halves(r)
                )  # fmt: skip

    def run_two_shot(self, x, recv, peers, rank, shifted=None):
        """Push this rank's rows of ``x`` [rows, K] @ W.T into slot ``rank`` of their
        owners' ``recv`` [TP, cap, 7168 / TP]; ``peers`` [TP] int64: the owners'
        receive bases; ``shifted`` (rows above LAUNCH_ROWS): the bases moved to the
        second launch's first token row."""
        rows = x.shape[0]
        if rows not in self.codes or x.shape[1] != self.k or not x.is_contiguous():
            raise ValueError(f"two-shot producer input {tuple(x.shape)}")
        recv_view, s = to_cute(recv), stream(self.device)
        for code, (first, n) in zip(self.codes[rows], halves(rows)):
            bases = peers if first == 0 else shifted
            args = (to_cute(x[first : first + n]), self.weight, recv_view)
            code(*args, to_cute(bases), cutlass.Int32(rank), s)
