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
ROWS = range(9, 65)  # batch rows a producer compiles (the o-projection producer's)


class TwoShotProducer:
    """``weight`` [7168, K] BF16 (row-major); ``rows``: the batch rows to compile
    (each above 8 and at most 64). Compile outside graph captures."""

    def __init__(self, weight, rows, tp):
        from ..oproj.prepared import compile_oproj

        if weight.dim() != 2 or weight.shape[0] != HIDDEN or not weight.is_contiguous():
            raise ValueError(f"two-shot producer weight {tuple(weight.shape)}")
        self.k, self.device = weight.shape[1], weight.device
        assert self.k % 64 == 0 and all(r in ROWS for r in rows)
        self.weight = to_cute(weight)
        with torch.accelerator.device_index(self.device.index):
            self.codes = {
                r: compile_oproj("producer_umma", self.k, r, tp, 1, True, False,
                                 twoshot=True)
                for r in rows
            }  # fmt: skip

    def run_two_shot(self, x, recv, peers, rank):
        """Push this rank's rows of ``x`` [rows, K] @ W.T into slot ``rank`` of their
        owners' ``recv`` [TP, cap, 7168 / TP]; ``peers`` [TP] int64: the owners'
        receive bases."""
        rows = x.shape[0]
        if rows not in self.codes or x.shape[1] != self.k or not x.is_contiguous():
            raise ValueError(f"two-shot producer input {tuple(x.shape)}")
        args = (to_cute(x), self.weight, to_cute(recv), to_cute(peers))
        self.codes[rows](*args, cutlass.Int32(rank), stream(self.device))
