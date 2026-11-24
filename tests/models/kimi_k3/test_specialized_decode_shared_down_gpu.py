# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode, Rubin TP16 verify rows <= 8 (``front.SITU_SPLIT_WIDE``):
the shared expert's down rows run ``shared.SharedDown`` on a few SMs, peer-storing
into the tail's RS mailboxes, after the gate/up GEMM's SiTU epilogue
(``PreparedSitu``). Checks the mailbox rows against the split-K down GEMM those rows
replaced (``PreparedProjection``, the stock tactic) and an FP32 reference of
h @ down^T, at the SM counts the table may pick."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)), reason="sm_107 only"
)

TP, CAP = 16, 64


def _mailbox_rows(boxes, parity, rank, rows):
    """The [rows, 7168] down rows the peers' RS slots received."""
    shard = 7168 // TP
    first = (parity * TP + rank) * CAP
    parts = [box.view(-1, shard)[first : first + rows] for box in boxes]
    return torch.cat(parts, dim=1)


@pytest.mark.parametrize("sms", [8, 12, 16])
@pytest.mark.parametrize("rows", [8])
def test_shared_down_matches_splitk_down(sms, rows):
    from cutlass import Int32

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (
        PreparedProjection,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import front
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.shared import (
        compile_down,
        weight_maps,
    )

    dev = torch.device("cuda")
    hidden = 6144 // TP
    g = torch.Generator(device=dev).manual_seed(rows * 100 + sms)
    gate_up = (torch.randn(2 * hidden, 7168, device=dev, generator=g) * 0.02).bfloat16()
    down = (torch.randn(7168, hidden, device=dev, generator=g) * 0.02).bfloat16()
    h = torch.randn(rows, hidden, device=dev, generator=g).bfloat16()

    tactic = front.batch_table(front.SHARED_TACTICS[hidden][1])[rows]
    stock = PreparedProjection(down, {rows: tactic}).prepare()
    expect = torch.empty(rows, 7168, dtype=torch.bfloat16, device=dev)
    stock.run(h, expect)

    boxes = [
        torch.full((2 * TP * CAP * (7168 // TP),), float("nan"), device=dev).bfloat16()
        for _ in range(TP)
    ]
    peers = torch.tensor([b.data_ptr() for b in boxes], dtype=torch.int64, device=dev)
    maps = weight_maps(gate_up, down)
    code = compile_down(TP, rows, dev.index, sms)
    for parity, rank in ((0, 3), (1, 11)):  # both ring halves, two source ranks
        code(
            to_cute(h), to_cute(maps, 64), to_cute(peers), Int32(parity), Int32(rank),
            Int32(CAP), stream(dev),
        )  # fmt: skip
        torch.cuda.synchronize()
        got = _mailbox_rows(boxes, parity, rank, rows)
        assert not got.isnan().any(), "mailbox rows left unwritten"
        ref = h.float() @ down.float().t()
        torch.testing.assert_close(got.float(), ref, rtol=1.6e-2, atol=1e-3)
        torch.testing.assert_close(got, expect, rtol=1.6e-2, atol=1e-3)
        # -0 made +0 (the RS sums must not see a negative zero from this rank)
        assert not torch.signbit(got[got == 0]).any()


def test_situ_split_table_rows():
    """Rubin TP16 verify: 8 rows take SharedDown behind the SiTU epilogue; q=1 and
    other TPs keep the split-K down."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import front

    table = front.batch_table(front.SITU_SPLIT_WIDE[16][1])
    assert set(table) == {8} and table[8] > 0
    assert not front.SITU_SPLIT_WIDE[16][0]
    assert front.batch_table(front.SITU_FUSED_WIDE[384]).get(8)
