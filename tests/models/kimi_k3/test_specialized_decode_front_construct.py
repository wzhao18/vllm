# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode: ``moe.front.SharedFront`` construction (the MXFP4 /
BF16-front MoE layers' front: shared-expert tactics, SharedDown splits, merged front,
routing options) at TP8 and TP16, q 1 / 4 / 8, on GB300-class (152 SMs) and
Rubin-class (212 SMs: the *_WIDE tables) devices, every policy branch executed with
the kernel compiles and weight maps stubbed out (no GPU). The NVFP4 checkpoint's MoE
layers build no SharedFront (``moe.nvfp4.Nvfp4MoE``: the stock router and latent
down): its decode is checked to take that path, so a front table edit cannot reach
it. Guards table refactors such as 16e76e1592's, which left a name the constructor
still read undefined (NameError at every MXFP4 MoE layer's build)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass")

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (  # noqa: E402
    front,
)

BATCHES = (1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64)


class _Stub:
    """Any compiled kernel, projection, map or view: callable, preparable, indexable."""

    def __init__(self, *args, **kwargs):
        self.kernels = {}

    def __call__(self, *args, **kwargs):
        return self

    def prepare(self):
        return self

    def data_ptr(self):
        return 0


def _stub(*args, **kwargs):
    return _Stub()


def _layer(tp):
    shared, shard, valid = 6144 // tp, 3584 // tp, 896 // tp
    rows = -(-valid // 16) * 16 + shard  # pack_front: router rows (16-padded), down
    empty = lambda *shape: torch.empty(shape, dtype=torch.bfloat16)  # noqa: E731
    experts = SimpleNamespace(
        gate_up_proj=SimpleNamespace(weight=empty(2 * shared, 7168)),
        down_proj=SimpleNamespace(weight=empty(7168, shared)),
    )
    moe = SimpleNamespace(
        shared_experts=experts, routed_scaling_factor=1.0, moe_renormalize=True,
        gate=SimpleNamespace(e_score_correction_bias=torch.zeros(896)),
    )  # fmt: skip
    box = SimpleNamespace(buffer=empty(1, 64, front.WIDTH), mc=0, rank=0)
    storage = tuple(empty(64, 8) for _ in range(7))
    return moe, empty(rows, 7168), box, storage


@pytest.mark.parametrize("wide", [False, True], ids=["gb300", "rubin"])
@pytest.mark.parametrize("q", [1, 4, 8])
@pytest.mark.parametrize("tp", [8, 16])
def test_shared_front_constructs(monkeypatch, tp, q, wide):
    sms = 212 if wide else 152
    props = SimpleNamespace(multi_processor_count=sms, major=10, minor=7 if wide else 3)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: props)
    for name in (
        "to_cute", "PreparedProjection", "PreparedSitu", "weight_maps", "weight_map",
        "matrix_view", "compile_front_tmem", "compile_publish", "compile_route",
        "compile_shared", "compile_down", "prepare_shared_carveout",
    ):  # fmt: skip
        if hasattr(front, name):
            monkeypatch.setattr(front, name, _stub)
    monkeypatch.setattr(front, "rubin", lambda *a: False, raising=False)
    monkeypatch.setattr(front, "polled", lambda *a: False, raising=False)
    moe, packed, box, storage = _layer(tp)
    counters = torch.zeros(4, dtype=torch.int32)
    built = front.SharedFront(
        moe, packed, box, storage, BATCHES, 4.0, 25.0, plan=torch.zeros(8),
        counters=counters, q=q,
    )  # fmt: skip
    assert set(built.plans) == set(BATCHES)
    split = front.split_table(tp, q > 1, wide)
    expect = {b: split[b] for b in BATCHES if b > 8 and split.get(b)}
    if wide and tp == 16:  # SiTU-epilogue rows with SharedDown beside the experts
        situ = front.batch_table(front.SITU_SPLIT_WIDE[tp][q > 1])
        expect.update({b: n for b, n in situ.items() if b in BATCHES})
    assert built.split == expect
    assert set(built.situ_down) == {b for b in expect if b <= 8}
    if wide and q > 1:
        assert all(built.split[b] == 8 for b in (16, 32, 48)), built.split
    assert built.merged <= set(built.split)


def test_nvfp4_moe_builds_no_shared_front():
    """The NVFP4 MoE layers (``Nvfp4MoE``) take no packed front and no SharedFront:
    the decode's build reads ``nvfp4`` before any front is made."""
    import inspect

    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import decode

    source = inspect.getsource(decode.SpecializedDecode.build_providers)
    nvfp4_branch = source.index("if index in nvfp4:")
    assert nvfp4_branch < source.index("front = SharedFront(")
    assert "continue" in source[nvfp4_branch : source.index("front = SharedFront(")]
