# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Kimi K3 SP MXFP8 all-gather (``ops/sp_all_gather.py``), CPU part:

* Layout: MXFP8 scales are per row (1x32 blocks along K) and the F8_128x4
  layout keeps the 128-row tile outermost, so with shards of a multiple of 128
  rows the ranks' local quantizations concatenated are bitwise the
  quantization of the gathered rows; other shard sizes are not (which is why
  they keep the BF16 gather).
* Dispatch (``mxfp8_gather_rows_ok``): only symmetric-memory pushes of a
  multiple of 128 rows; a shard the custom (Lamport) all-gather takes stays
  BF16, as does every shard without the buffer.
* ``kimi_sp_all_gather_mxfp8`` drops the SP padding rows (data and the scale
  tiles past the last real row's).

The GPU bitwise test over the symmetric-memory push is
``tests/distributed/test_kimi_k3_sp_all_gather_mxfp8.py``."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.models.kimi_k3.nvidia.ops import sp_all_gather as sp_ag

HIDDEN = 7168
K_SMALL = 256  # layout checks: any multiple of 128 columns


def _quant(x):
    return _mxfp8_e4m3_quantize_torch(x, is_sf_swizzled_layout=True)


def _bytes(t):
    return t.contiguous().view(torch.uint8).reshape(-1)


@pytest.mark.parametrize("rows", [128, 256, 384])
@pytest.mark.parametrize("world", [2, 8])
def test_local_quant_concat_is_gathered_quant(rows, world):
    gen = torch.Generator().manual_seed(rows * world)
    shards = [
        (torch.randn(rows, K_SMALL, generator=gen) * 4 ** (r - 2)).bfloat16()
        for r in range(world)
    ]
    ref_q, ref_s = _quant(torch.cat(shards))
    local = [_quant(s) for s in shards]
    assert torch.equal(_bytes(torch.cat([q for q, _ in local])), _bytes(ref_q))
    assert torch.equal(torch.cat([_bytes(s) for _, s in local]), _bytes(ref_s))


@pytest.mark.parametrize("rows", [96, 200])
def test_local_swizzle_differs_off_tile(rows):
    """Shards that split a 128-row tile: the concatenated local scales are
    not the gathered layout (the dispatch keeps BF16 for them)."""
    gen = torch.Generator().manual_seed(rows)
    shards = [torch.randn(rows, K_SMALL, generator=gen).bfloat16() for _ in range(2)]
    _, ref_s = _quant(torch.cat(shards))
    cat = torch.cat([_bytes(_quant(s)[1]) for s in shards])
    assert cat.numel() != ref_s.numel() or not torch.equal(cat, _bytes(ref_s))


@pytest.fixture
def buffer(monkeypatch):
    """A push buffer of ``max_rows`` rows per rank and a custom all-gather of
    up to ``ca_bytes`` (None: no custom all-gather)."""

    def setup(max_rows=2048, ca_bytes=2 * 1024 * 1024, present=True):
        gather = (
            SimpleNamespace(dtype=torch.bfloat16, hidden_size=HIDDEN, max_rows=max_rows)
            if present
            else None
        )
        monkeypatch.setattr(sp_ag, "_sp_all_gather", gather)
        ca_comm = (
            None
            if ca_bytes is None
            else SimpleNamespace(
                should_custom_all_gather=lambda t: (
                    t.dtype == torch.bfloat16 and t.nbytes <= ca_bytes
                )
            )
        )
        group = SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=ca_comm))
        monkeypatch.setattr(sp_ag, "get_tp_group", lambda: group)

    return setup


def test_dispatch(buffer):
    buffer()
    ok = sp_ag.mxfp8_gather_rows_ok
    # 128 rows x 7168 BF16 = 1.75 MiB: the custom (Lamport) all-gather takes
    # it, and it stays BF16 (FP8 bytes can alias the Lamport sentinel).
    assert not ok(128, HIDDEN)
    for rows in (256, 512, 1024, 2048):
        assert ok(rows, HIDDEN), rows
    for rows in (0, 200, 257, 1000, 2047):
        assert not ok(rows, HIDDEN), rows
    assert not ok(2048 + 128, HIDDEN)  # above the buffer (raises on BF16 too)
    assert not ok(512, HIDDEN // 2)  # another width
    buffer(ca_bytes=None)  # no custom all-gather: 128 rows push as MXFP8
    assert ok(128, HIDDEN)
    buffer(present=False)  # no buffer: the stock all-gather, BF16
    assert not ok(512, HIDDEN)


def test_gather_drops_padding_rows(monkeypatch):
    world, rows = 4, 256
    total = world * rows
    data = torch.arange(total * HIDDEN, dtype=torch.int32).to(torch.uint8)
    data = data.view(torch.float8_e4m3fn).view(total, HIDDEN)
    sf = torch.arange(total * HIDDEN // 32, dtype=torch.int32).to(torch.uint8)
    calls = []

    def gather_mxfp8(d, s):
        calls.append((d.shape, s.numel()))
        return data, sf

    monkeypatch.setattr(
        sp_ag, "_sp_all_gather", SimpleNamespace(gather_mxfp8=gather_mxfp8)
    )
    qa = QuantizedActivation(
        data=torch.empty(rows, HIDDEN, dtype=torch.float8_e4m3fn),
        scale=torch.empty(rows * HIDDEN // 32, dtype=torch.uint8),
        orig_dtype=torch.bfloat16,
        orig_shape=torch.Size((rows, HIDDEN)),
        quant_key=kMxfp8Dynamic,
    )
    for num_tokens, sf_rows in ((total, total), (total - 5, total), (897, 1024)):
        out = sp_ag.kimi_sp_all_gather_mxfp8(qa, num_tokens)
        assert out.data.shape == (num_tokens, HIDDEN)
        assert out.data.data_ptr() == data.data_ptr()
        assert out.scale.numel() == sf_rows * HIDDEN // 32
        assert out.orig_shape == (num_tokens, HIDDEN)
        assert out.orig_dtype == torch.bfloat16
        assert out.quant_key == kMxfp8Dynamic
    assert calls == [((rows, HIDDEN), rows * HIDDEN // 32)] * 3
