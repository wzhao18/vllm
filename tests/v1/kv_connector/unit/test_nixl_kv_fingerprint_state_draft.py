# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV fingerprints of the pages NIXL moves besides the DSA rows: KDA state pages
(conv sub-projections + temporal state, sliced by TP head shard) and head-sharded
GQA drafter pages. P (TP1, large blocks) and the D TP ranks (small blocks) must
sum to the same F; a misplaced or corrupted slice must change it. CPU only."""

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.kv_fingerprint import (
    DraftView,
    StateView,
    byte_pages,
    draft_fingerprint,
    state_fingerprint,
)

M = 1 << 64
TP = 4
# Per D rank: conv q, k, v sub-projections, then the temporal state (bytes).
D_SIZES = (48, 48, 64, 256)
LAYER = "model.layers.3.self_attn"


def _segments(scale: int) -> tuple[tuple[int, int], ...]:
    out, offset = [], 0
    for size in D_SIZES:
        out.append((offset, size * scale))
        offset += size * scale
    return tuple(out)


def _d_pages(p_page: torch.Tensor) -> list[torch.Tensor]:
    """D rank r holds slice r of every TP1 segment (hetero-TP SSM reads)."""
    return [
        torch.cat(
            [p_page[o + r * s // TP : o + (r + 1) * s // TP] for o, s in _segments(TP)]
        )
        for r in range(TP)
    ]


def _state_view(page: torch.Tensor, scale: int, tp: int, rank: int) -> StateView:
    """``page`` stored as block 1 of a 3-block region with padded stride."""
    stride = page.numel() + 16
    buf = torch.zeros(3 * stride, dtype=torch.uint8)
    buf[stride : stride + page.numel()] = page
    pages = byte_pages(buf, buf.data_ptr(), page.numel(), stride, 3)
    return StateView(LAYER, pages, _segments(scale), tp, rank)


def _state_f(p_page: torch.Tensor, d_pages: list[torch.Tensor]) -> tuple[int, int]:
    fp = state_fingerprint(_state_view(p_page, TP, 1, 0), 1)
    fd = sum(
        state_fingerprint(_state_view(d, 1, TP, r), 1) for r, d in enumerate(d_pages)
    )
    return fp, fd % M


def _p_page(seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (TP * sum(D_SIZES),), dtype=torch.uint8, generator=gen)


def test_state_tp_slices_sum_to_the_tp1_page():
    p = _p_page()
    fp, fd = _state_f(p, _d_pages(p))
    assert fp == fd


def test_state_misplaced_or_corrupted_slices_change_f():
    p = _p_page(1)
    d = _d_pages(p)
    swapped = [d[0], d[2], d[1], d[3]]  # two ranks' head slices exchanged
    assert len(set(_state_f(p, swapped))) == 2
    corrupt = [x.clone() for x in d]
    corrupt[3][-1] ^= 1  # one bit of the last temporal-state word
    assert len(set(_state_f(p, corrupt))) == 2
    sub = [x.clone() for x in d]
    sub[0][:48], sub[0][48:96] = d[0][48:96].clone(), d[0][:48].clone()  # conv q <-> k
    assert len(set(_state_f(p, sub))) == 2


def test_byte_pages_stay_inside_the_storage():
    buf = torch.zeros(100, dtype=torch.uint8)
    with pytest.raises(ValueError):
        byte_pages(buf, buf.data_ptr(), 40, 50, 3)
    assert byte_pages(buf, buf.data_ptr() + 10, 40, 50, 2).shape == (2, 40)


H, B_P, B_D, C = 8, 16, 4, 8


def _draft_case(num_tokens: int, window: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    rows = torch.randint(0, 256, (num_tokens, H, C), dtype=torch.uint8, generator=gen)
    n_p = -(-num_tokens // B_P)
    table_p = torch.randperm(n_p + 2, generator=gen)[:n_p].tolist()
    p = torch.zeros(n_p + 2, H, B_P, C, dtype=torch.uint8)
    n_d = -(-num_tokens // B_D)
    table_d = torch.randperm(n_d + 3, generator=gen)[:n_d].tolist()
    heads = H // TP
    d = [torch.zeros(n_d + 3, heads, B_D, C, dtype=torch.uint8) for _ in range(TP)]
    for t in range(num_tokens):
        p[table_p[t // B_P], :, t % B_P] = rows[t]
        for r in range(TP):
            d[r][table_d[t // B_D], :, t % B_D] = rows[t, r * heads : (r + 1) * heads]
    pv = DraftView(LAYER, p, 0, H, window)
    dv = [DraftView(LAYER, d[r], r * heads, H, window) for r in range(TP)]
    return pv, table_p, dv, table_d


def _draft_f(pv, table_p, dv, table_d, num_tokens):
    fp = draft_fingerprint(pv, table_p, num_tokens)
    fd = sum(draft_fingerprint(v, table_d, num_tokens) for v in dv) % M
    return fp, fd


@pytest.mark.parametrize("num_tokens", [3, 37, 64])
def test_draft_head_shards_sum_to_the_tp1_pages(num_tokens: int):
    pv, tp_, dv, td = _draft_case(num_tokens, window=10)
    assert len(set(_draft_f(pv, tp_, dv, td, num_tokens))) == 1


def test_draft_window_and_corruption():
    num_tokens, window = 37, 10
    pv, tp_, dv, td = _draft_case(num_tokens, window, seed=2)
    old = num_tokens - window - 1  # outside the window: not attended, not checked
    dv[1].pages[td[old // B_D], 0, old % B_D, 0] ^= 0xFF
    assert len(set(_draft_f(pv, tp_, dv, td, num_tokens))) == 1
    last = num_tokens - 1
    dv[1].pages[td[last // B_D], 1, last % B_D, 3] ^= 1
    assert len(set(_draft_f(pv, tp_, dv, td, num_tokens))) == 2


def test_draft_heads_swapped_between_ranks_change_f():
    num_tokens = 37
    pv, tp_, dv, td = _draft_case(num_tokens, window=None, seed=3)
    swapped = [
        dv[0],
        DraftView(LAYER, dv[2].pages, 2, H, None),
        DraftView(LAYER, dv[1].pages, 4, H, None),
        dv[3],
    ]
    assert len(set(_draft_f(pv, tp_, swapped, td, num_tokens))) == 2


def _worker_draft_views(
    region: torch.Tensor,
    spec_block: int,
    lcm_block: int,
    ratio: int = 1,
    tp: int = 1,
    rank: int = 0,
):
    """``NixlBaseConnectorWorker._fingerprint_state_draft_views`` on one
    head-sharded drafter region ``region`` ([blocks, page bytes]) whose layer
    spec has ``spec_block`` tokens per block, next to transfer groups whose LCM
    (the worker's block size) is ``lcm_block``."""
    from types import SimpleNamespace

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        NixlBaseConnectorWorker,
    )

    name = "model.layers.24.self_attn.attn"
    blocks, page = region.shape
    worker = SimpleNamespace(
        kv_caches_base_addr={"e": {rank: [region.data_ptr()]}},
        engine_id="e",
        tp_rank=rank,
        world_size=tp,
        _has_mamba=False,
        _conv_decomp=None,
        _head_sharded_draft_kv_heads=H,
        _head_sharded_draft_regions={0},
        region_names=[name],
        block_len_per_layer=[page],
        block_stride_per_layer=[page],
        region_num_blocks=[blocks],
        block_size=lcm_block // ratio,
        _physical_blocks_per_logical_kv_block=ratio,
        _layer_specs={name: SimpleNamespace(block_size=spec_block, sliding_window=10)},
    )
    _, drafts = NixlBaseConnectorWorker._fingerprint_state_draft_views(
        worker, {name: region}, {name: 2}
    )
    return drafts


@pytest.mark.parametrize("ratio", [1, 2])
def test_worker_draft_view_uses_the_drafter_block_size(ratio: int):
    """BUG-INTEGRATE-2: a sliding-window drafter keeps its own block size (128
    next to a 16384-token hybrid attention block under MC=1 prefix caching), so
    its page is ``heads x spec block x row bytes``, not the transfer groups'
    LCM. The worker's draft view must page by the drafter's block and log the
    same F as the TP1 reference pages of that block size."""
    num_tokens = 37
    spec_block = B_P * ratio  # logical; the registered pages hold B_P tokens
    pv, table_p, _, _ = _draft_case(num_tokens, window=10, seed=4)
    region = pv.pages.reshape(pv.pages.shape[0], -1).clone()
    drafts = _worker_draft_views(region, spec_block, 1024 * ratio, ratio)
    (view,) = drafts[2]
    assert view.pages.shape == pv.pages.shape
    assert view.window == 10 and view.head0 == 0 and view.total_heads == H
    assert draft_fingerprint(view, table_p, num_tokens) == draft_fingerprint(
        pv, table_p, num_tokens
    )


def test_worker_draft_view_refuses_a_page_that_is_not_heads_x_block():
    pv, _, _, _ = _draft_case(5, window=None)
    region = pv.pages.reshape(pv.pages.shape[0], -1)[:, :-4].contiguous()
    with pytest.raises(ValueError, match="draft page"):
        _worker_draft_views(region, B_P, 1024)
    with pytest.raises(ValueError, match="physical pages"):
        _worker_draft_views(pv.pages.reshape(pv.pages.shape[0], -1), 3, 1024, 2)
