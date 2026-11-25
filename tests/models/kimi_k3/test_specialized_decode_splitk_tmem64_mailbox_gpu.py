# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode, ``SplitKTmemGemm64`` with separate reduce mailboxes
(``separate``: the owners' FP32 mailboxes past the A / x slots, no cluster barrier
between the MMAs and the DSMEM pushes) is bitwise the same kernel with the mailboxes in
the A / x slots behind that barrier, at the decode tactics' 16 / 32 / 64 rows (N64,
one N tile, partial last row tile) and over repeated calls; both within FP32
accumulation noise of an FP64 reference. Also: which tactics take separate mailboxes
(they must fit below the lone CTA's shared-memory request). One GPU, seconds."""

import pytest
import torch

pytest.importorskip("cutlass")

K = 7168
TACTIC = (128, 64, 8, 14, 14)  # the TMEM split-K decode tactic at N64 (front, in-proj)


def _rubin():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _rubin(), reason="Rubin / Blackwell GPU required")


def _run(weight, x, separate):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import (
        projection,
        splitk_tmem64,
    )

    cls = splitk_tmem64.SplitKTmemGemm64
    init = cls.__init__

    def barrier_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.separate = False  # the mailboxes in the A / x slots, behind the barrier

    projection.compile_projection_tmem.cache_clear()
    if not separate:
        cls.__init__ = barrier_init
    try:
        prepared = projection.PreparedProjection(weight, {x.shape[0]: TACTIC}).prepare()
    finally:
        cls.__init__ = init
        projection.compile_projection_tmem.cache_clear()
    outs = []
    for _ in range(3):
        out = torch.full((x.shape[0], weight.shape[0]), float("nan"), device=x.device,
                         dtype=torch.bfloat16)  # fmt: skip
        prepared.run(x, out)
        torch.cuda.synchronize()
        outs.append(out)
    return outs


@pytest.mark.parametrize("rows", [16, 32, 64])
@pytest.mark.parametrize("n", [1024, 3040])
def test_separate_mailboxes_bitwise(rows, n):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.splitk_tmem64 import (  # noqa: E501
        SplitKTmemGemm64,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.splitk import (
        SplitKTactic,
    )

    assert SplitKTmemGemm64(tactic=SplitKTactic(*TACTIC[:4]), k=K).separate
    gen = torch.Generator(device="cuda").manual_seed(1000 * rows + n)
    weight = (torch.randn(n, K, device="cuda", generator=gen) * 0.02).bfloat16()
    x = torch.randn(rows, K, device="cuda", generator=gen).bfloat16()
    got = _run(weight, x, separate=True)
    ref = _run(weight, x, separate=False)
    for a, b in zip(got, ref):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    assert torch.equal(got[0].view(torch.int16), got[2].view(torch.int16))
    exact = x.double() @ weight.double().T
    err = (got[0].double() - exact).abs().max() / exact.abs().max()
    assert err < 1e-2, err  # BF16 output rounding


def test_separate_only_below_the_lone_request():
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.splitk import (
        SplitKTactic,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.splitk_tmem64 import (  # noqa: E501
        ALONE_SMEM,
        SplitKTmemGemm64,
    )

    for n, slots in ((64, 14), (128, 8)):
        gemm = SplitKTmemGemm64(tactic=SplitKTactic(128, n, 8, slots), k=K)
        assert gemm.smem_bytes == max(gemm.smem_bytes, ALONE_SMEM)
        if gemm.separate:
            assert gemm.smem_bytes == ALONE_SMEM  # the request is unchanged
    lone = SplitKTmemGemm64(tactic=SplitKTactic(128, 64, 8, 14), k=K, alone=False)
    assert not lone.separate
