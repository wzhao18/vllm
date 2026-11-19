# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlashProposer._window_rows: with sliding-window draft layers, the context
rows kept for the fc / K/V projection are exactly those whose draft K/V can be
read: within the window (+ draft span) of the chunk end, or of a prefix-cache
hit point (a multiple of the hit alignment) inside the chunk."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.spec_decode.dflash import DFlashProposer

NUM_SPEC = 3


def _proposer(window, *, caching=False, align=None, connector=False):
    p = object.__new__(DFlashProposer)
    p._context_window_cache = window
    p.num_speculative_tokens = NUM_SPEC
    cache = SimpleNamespace(
        enable_prefix_caching=caching,
        cache_hit_alignment_tokens=align,
        hash_block_size=align,
        prefix_match_unit=None,
        block_size=align or 16,
    )
    p.vllm_config = SimpleNamespace(
        cache_config=cache, kv_transfer_config=object() if connector else None
    )
    return p


def _cad(qlens, seq_lens):
    qsl = torch.zeros(len(qlens) + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(np.cumsum(qlens), dtype=torch.int32)
    return SimpleNamespace(
        num_reqs=len(qlens),
        query_start_loc_cpu=qsl,
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
    )


def _needed(qlens, seq_lens, window, align):
    span = window + NUM_SPEC + 1
    rows, base = [], 0
    for q, end in zip(qlens, seq_lens):
        start = end - q
        hits = [end] + (
            [h for h in range(start + 1, end) if h % align == 0] if align else []
        )
        for j in range(q):
            pos = start + j
            if q <= span or any(h - span <= pos < h for h in hits):
                rows.append(base + j)
        base += q
    return np.array(rows, dtype=np.int64)


CASES = [
    # (qlens, seq_lens): the step-bench F16K case, a fresh 16K prompt, a mix
    # with decode-sized requests, a chunk shorter than the window.
    ([5461, 5461, 5462], [83821, 83821, 83822]),
    ([16384], [16384]),
    ([4, 9000, 1, 3000], [500, 40000, 77, 5400]),
    ([2051], [9000]),
]


@pytest.mark.parametrize("qlens,seq_lens", CASES)
@pytest.mark.parametrize("align", [None, 4096, 16384])
def test_window_rows_exact(qlens, seq_lens, align):
    window = 2048
    p = _proposer(window, caching=align is not None, align=align)
    got = p._window_rows(_cad(qlens, seq_lens))
    want = _needed(qlens, seq_lens, window, align)
    if want.size == sum(qlens):
        assert got is None
    else:
        assert got is not None and np.array_equal(got, want)


def test_window_rows_off_cases():
    cad = _cad([16384], [16384])
    # Full-attention drafter.
    assert _proposer(None)._window_rows(cad) is None
    # Hits finer than the window: every row may be read later.
    assert _proposer(2048, caching=True, align=1024)._window_rows(cad) is None
    # A KV connector without local prefix caching counts as caching.
    assert _proposer(2048, connector=True, align=512)._window_rows(cad) is None
    # Decode-sized steps never trim.
    assert _proposer(2048)._window_rows(_cad([4, 4], [900, 30000])) is None


def test_window_rows_f16k_saving():
    p = _proposer(2048)
    rows = p._window_rows(_cad([5461, 5461, 5462], [83821, 83821, 83822]))
    assert rows.size == 3 * (2048 + NUM_SPEC + 1)


def test_trim_context_metadata():
    from vllm.v1.attention.backend import CommonAttentionMetadata

    qlens, seq_lens = [3000, 4], [3000, 900]
    qsl = torch.tensor([0, 3000, 3004], dtype=torch.int32)
    cad = CommonAttentionMetadata(
        query_start_loc=qsl.clone(),
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=3004,
        max_query_len=3000,
        max_seq_len=3000,
        block_table_tensor=torch.zeros(2, 4, dtype=torch.int32),
        slot_mapping=torch.arange(3004),
    )
    p = _proposer(2048)
    p.device = torch.device("cuda")
    p._context_stage = None
    cad.slot_mapping = cad.slot_mapping.cuda()
    rows = p._window_rows(_cad(qlens, seq_lens))
    index, out = p.trim_context(cad, rows)
    torch.cuda.synchronize()
    assert torch.equal(index.cpu(), torch.from_numpy(rows))
    assert out.query_start_loc.cpu().tolist() == [0, 2052, 2056]
    assert out.query_start_loc_cpu.tolist() == [0, 2052, 2056]
    assert out.num_actual_tokens == 2056 and out.max_query_len == 2052
    assert torch.equal(out.slot_mapping.cpu(), torch.from_numpy(rows))
    assert torch.equal(out.seq_lens, cad.seq_lens)
