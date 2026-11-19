# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""window_context_rows: with sliding-window draft layers, the context rows kept
for the fc / K/V projection are exactly those whose draft K/V can be read:
within ``span`` of the chunk end, or of a prefix-cache hit point (a multiple of
the hit alignment) inside the chunk."""

import numpy as np
import pytest

from vllm.v1.worker.gpu.spec_decode.dflash.utils import window_context_rows

SPAN = 2048 + 3 + 1


def _qsl(qlens):
    return np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)


def _needed(qlens, seq_ends, align):
    rows, base = [], 0
    for q, end in zip(qlens, seq_ends):
        start = end - q
        hits = [end] + (
            [h for h in range(start + 1, end) if h % align == 0] if align else []
        )
        for j in range(q):
            pos = start + j
            if q <= SPAN or any(h - SPAN <= pos < h for h in hits):
                rows.append(base + j)
        base += q
    return np.array(rows, dtype=np.int64)


CASES = [
    # (qlens, seq_ends): the step-bench F16K case, a fresh 16K prompt, a mix
    # with decode-sized requests, a chunk barely above the span.
    ([5461, 5461, 5462], [83821, 83821, 83822]),
    ([16384], [16384]),
    ([4, 9000, 1, 3000], [500, 40000, 77, 5400]),
    ([2053], [9000]),
]


@pytest.mark.parametrize("qlens,seq_ends", CASES)
@pytest.mark.parametrize("align", [None, 4096, 16384])
def test_window_context_rows_exact(qlens, seq_ends, align):
    got = window_context_rows(_qsl(qlens), np.array(seq_ends), SPAN, align)
    want = _needed(qlens, seq_ends, align)
    if want.size == sum(qlens):
        assert got is None
    else:
        assert got is not None and np.array_equal(got, want)


def test_window_context_rows_keeps_all():
    qsl = _qsl([16384])
    # Hits finer than the span: every row may be read later.
    assert window_context_rows(qsl, np.array([16384]), SPAN, 1024) is None
    # Decode-sized steps never trim.
    assert window_context_rows(_qsl([4, 4]), np.array([900, 30000]), SPAN, None) is None


def test_window_context_rows_f16k():
    rows = window_context_rows(
        _qsl([5461, 5461, 5462]), np.array([83821, 83821, 83822]), SPAN, None
    )
    assert rows.size == 3 * SPAN
