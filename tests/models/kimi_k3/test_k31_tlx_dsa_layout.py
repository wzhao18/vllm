# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA TLX-S (``ops/tlx_dsa.py``): zero-allocation staging layout and
byte views, on CPU.

``TlxDsa.__init__`` runs against stand-ins of the SP all-gather buffer and the
GEMM-RS workspace (CPU tensors, fake symmetric handles, a no-op push
extension), so the real constructor code places the layout:

* first fit at TP8 for MNBT 16384 / 24576 / 65536: every part inside a dead
  region, no two parts of one buffer overlapping, the W_g / W_q_b column parts
  covering their matrices;
* the capability path below the fit point: ``_NoFit`` from the constructor and
  ``get()`` returning None (TLX-S off, never a mid-run fallback);
* the byte views of the 2D GEMM-RS partial are flat: a view at a byte offset
  inside a row addresses exactly those bytes (a 2D uint8 view would slice rows).

The 8-GPU push / release path runs in the TP8+DCP8 prefill engine smoke (steps
of >= MIN_ROWS rows per rank) and its 1-GPU selftest; it needs the SM100
extension and symmetric memory, so it is not exercised here.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia.ops import tlx_dsa

W = 8
HIDDEN = 7168


class _Handle:
    """Fake symmetric-memory handle: every peer's buffer is the local tensor."""

    def __init__(self, t: torch.Tensor, pad_size: int = 4096) -> None:
        self.t = t
        self.signal_pad_size = pad_size
        self.signal_pad_ptrs = [0x10000 * (p + 1) for p in range(W)]

    def get_buffer(self, peer, shape, dtype, offset):
        return self.t


def _stand_ins(mnbt: int, monkeypatch):
    max_rows = mnbt // W
    # SP all-gather: two slots of world x max_rows x hidden BF16 rows (sp_all_gather_rows: the
    # shard of an MNBT step); the slot tails past the MXFP8 gather bytes are the dead regions.
    slot_numel = W * max_rows * HIDDEN
    storage = torch.empty(2 * slot_numel, dtype=torch.bfloat16)
    spag = SimpleNamespace(
        world=W, rank=0, storage=storage, slot_numel=slot_numel, max_rows=max_rows,
        hidden_size=HIDDEN, handle=_Handle(storage), tma=True,
    )  # fmt: skip
    partial = torch.empty(mnbt, HIDDEN, dtype=torch.bfloat16)  # GEMM-RS workspace [max_M, N]
    rs = SimpleNamespace(partial=partial, partial_handle=_Handle(partial), N=HIDDEN)
    ext = SimpleNamespace(tlx_push=lambda *a, **k: None, tlx_release=lambda *a, **k: None)
    monkeypatch.setattr(tlx_dsa, "_module", lambda: ext)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda dev: SimpleNamespace(multi_processor_count=148)
    )
    return spag, rs


def _parts(layout):
    """(name, buf, byte offset, nbytes) of every placed part."""
    out = []
    buf, off, w = layout["o"]
    out.append(("o", buf, off, w.numel() * 2))
    for which in ("g", "qb"):
        for n0, n1, buf, off, v in layout[which]:
            out.append((f"{which}[{n0}:{n1}]", buf, off, v.numel() * 2))
    return out


@pytest.mark.parametrize("mnbt", [16384, 24576, 65536])
def test_layout_first_fit(mnbt, monkeypatch):
    spag, rs = _stand_ins(mnbt, monkeypatch)
    t = tlx_dsa.TlxDsa(spag, rs)
    lay = t.layout
    slot = spag.slot_numel * 2
    tail = (W * spag.max_rows * HIDDEN * 33 // 32 + 255) // 256 * 256
    regions = {
        1: [(0, rs.partial.numel() * 2)],
        0: [(tail, slot), (slot + tail, 2 * slot)],
    }
    spans = _parts(lay)
    for i, (buf, off, w) in enumerate(lay["planes"]):
        spans.append((f"plane{i}", buf, off, (W + 1) * spag.max_rows * w * 2))
    # every part inside one dead region of its buffer
    for name, buf, off, nb in spans:
        assert any(a <= off and off + nb <= b for a, b in regions[buf]), (name, buf, off, nb)
    # no overlap within a buffer
    for buf in (0, 1):
        s = sorted((off, off + nb, name) for name, b, off, nb in spans if b == buf)
        for (a0, a1, an), (b0, b1, bn) in zip(s, s[1:]):
            assert a1 <= b0, (an, bn)
    # column parts cover W_g / W_q_b exactly, split at shard boundaries
    for which, n, shard in (("g", tlx_dsa.G_N, tlx_dsa.SHARD_G), ("qb", tlx_dsa.QB_N, tlx_dsa.SHARD_QB)):
        ps = lay[which]
        assert ps[0][0] == 0 and ps[-1][1] == n
        assert all(p[1] == q[0] for p, q in zip(ps, ps[1:]))
        assert all((p[1] - p[0]) % shard == 0 for p in ps)
    assert tuple(t.w_o.shape) == (tlx_dsa.O_N, tlx_dsa.O_K)
    # row planes: send rows and every rank's receive rows have the step's shape
    R = spag.max_rows
    kv, pe, wkv = t.send_rows(R)
    assert kv.shape == (R, tlx_dsa.KV_C) and pe.shape == (R, tlx_dsa.K_PE) and wkv.shape == (R, tlx_dsa.WKV)
    kv, pe, wkv = t.recv_rows(R)
    assert kv.shape == (W * R, tlx_dsa.KV_C) and wkv.shape == (W * R, tlx_dsa.WKV)
    assert "no new memory" in t.layout_desc


def test_layout_16384_fewest_parts(monkeypatch):
    """The documented MNBT 16384 packing: W_o whole, W_g in its two N halves (slot tails)."""
    spag, rs = _stand_ins(16384, monkeypatch)
    lay = tlx_dsa.TlxDsa(spag, rs).layout
    assert len(lay["g"]) == 2 and {p[2] for p in lay["g"]} == {0}
    assert lay["o"][0] == 1


@pytest.mark.parametrize("mnbt", [8192, 12288, 16128])
def test_no_fit_below_threshold(mnbt, monkeypatch):
    spag, rs = _stand_ins(mnbt, monkeypatch)
    with pytest.raises(tlx_dsa._NoFit, match="max_num_batched_tokens >= 16384"):
        tlx_dsa.TlxDsa(spag, rs)


def test_get_off_when_it_does_not_fit(monkeypatch):
    """Capability path: get() returns None (one INFO line), never a mid-run fallback."""
    from vllm.model_executor.kernels.linear.cute_dsl import gemm_rs_ar
    from vllm.models.kimi_k3.nvidia.ops import sp_all_gather as sp_ag

    spag, rs = _stand_ins(12288, monkeypatch)
    monkeypatch.setattr(sp_ag, "_sp_all_gather", spag, raising=False)
    monkeypatch.setattr(gemm_rs_ar, "_gemm_rs_ar", rs, raising=False)
    monkeypatch.setattr(tlx_dsa, "_tlx", None)
    monkeypatch.setattr(tlx_dsa, "_tlx_init_done", False)
    assert tlx_dsa.get() is None
    monkeypatch.setattr(tlx_dsa, "_tlx_init_done", False)
    spag2, rs2 = _stand_ins(16384, monkeypatch)
    monkeypatch.setattr(sp_ag, "_sp_all_gather", spag2, raising=False)
    monkeypatch.setattr(gemm_rs_ar, "_gemm_rs_ar", rs2, raising=False)
    assert isinstance(tlx_dsa.get(), tlx_dsa.TlxDsa)


def test_partial_byte_views_are_flat(monkeypatch):
    """a7e29a7393: a view at byte offset ``off`` of the [max_M, 7168] partial is those bytes, not rows."""
    spag, rs = _stand_ins(16384, monkeypatch)
    rs.partial.zero_()
    t = tlx_dsa.TlxDsa(spag, rs)
    assert all(r.dim() == 1 for r in t.raws)
    off, n = 3 * HIDDEN * 2 + 1000, 4096  # inside row 3, crossing into row 4
    v = t._view(1, off, n)
    assert v.numel() == n
    v.fill_(0xAB)
    flat = rs.partial.view(torch.uint8).view(-1)
    assert torch.all(flat[off : off + n] == 0xAB)
    assert torch.all(flat[:off] == 0) and torch.all(flat[off + n :] == 0)
    # W_o lands at its recorded byte offset of its buffer
    buf, o, w = t.layout["o"]
    assert w.data_ptr() == t.raws[buf].data_ptr() + o
