# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 DWDP load path (CPU): each rank allocates and reads only the experts it
holds.

Under expert DWDP (two DEP engines, EP x DWDP2) every MoE layer after the first keeps
one half of the EP-local experts (``ExpertDwdp.claim``); the NVFP4 parameters are built
at that size, the loader skips the other half before reading it
(``LayerExpertIds``), and the home half converts to the same kernel-layout bytes as
the same experts of a whole-layer conversion (the layout is per expert).
"""

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.model_loader.ep_weight_filter import (
    LayerExpertIds,
    should_skip_weight,
)


# -- loader filter ------------------------------------------------------------------


def test_layer_expert_ids_skip():
    ids = LayerExpertIds({0, 1, 2, 3}, {5: {0, 1}, 6: {2, 3}})
    pre = "language_model.model.layers"
    # layer 5 keeps experts 0-1 only; layer 6 keeps 2-3; other layers the default
    assert not should_skip_weight(f"{pre}.5.block_sparse_moe.experts.1.w1.weight", ids)
    assert should_skip_weight(f"{pre}.5.block_sparse_moe.experts.2.w1.weight", ids)
    assert should_skip_weight(f"{pre}.6.block_sparse_moe.experts.1.w2.weight", ids)
    assert not should_skip_weight(f"{pre}.6.block_sparse_moe.experts.3.w3.weight", ids)
    assert not should_skip_weight(f"{pre}.1.block_sparse_moe.experts.3.w3.weight", ids)
    assert should_skip_weight(f"{pre}.1.block_sparse_moe.experts.4.w3.weight", ids)
    # scales and non-expert weights are never skipped
    assert not should_skip_weight(
        f"{pre}.5.block_sparse_moe.experts.2.w1.weight_scale", ids
    )
    assert not should_skip_weight(f"{pre}.5.self_attn.q_proj.weight", ids)
    assert len(ids) == 4


# -- expert DWDP claim --------------------------------------------------------------


def _dwdp():
    pytest.importorskip("cuda.bindings.driver")
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import dwdp

    return dwdp


def _engine(dwdp, index):
    x = dwdp.ExpertDwdp.__new__(dwdp.ExpertDwdp)
    x.index, x.order = index, []
    return x


@pytest.mark.parametrize("experts", [224, 112, 8])
def test_claim_home_halves(experts):
    """The first layer is whole; later layers alternate the home half, the two
    engines' halves are disjoint and cover every expert, and each claimed half is
    the one ``_assemble`` exports (owner = half ^ (ordinal & 1))."""
    dwdp = _dwdp()
    e0, e1 = _engine(dwdp, 0), _engine(dwdp, 1)
    half = experts // 2
    for ordinal in range(7):
        (l0, a0, b0), (l1, a1, b1) = e0.claim(experts), e1.claim(experts)
        assert l0.ordinal == l1.ordinal == ordinal
        if ordinal == 0:
            assert (a0, b0) == (a1, b1) == (0, experts)
            assert l0.home is None and l1.home is None
            continue
        assert b0 - a0 == b1 - a1 == half
        assert {a0, a1} == {0, half}
        for engine, layer, lo in ((0, l0, a0), (1, l1, a1)):
            assert lo == layer.home * half
            assert layer.home ^ (ordinal & 1) == engine
            assert layer.experts == experts


def test_claim_rejects_odd():
    dwdp = _dwdp()
    e = _engine(dwdp, 0)
    e.claim(6)
    with pytest.raises(ValueError, match="even local expert count"):
        e.claim(7)


# -- the home half converts to the same bytes -----------------------------------------


def _phys_u8(t: torch.Tensor) -> torch.Tensor:
    """Expert-major physical bytes of a kernel tensor (fc codes are K-major views)."""
    if t.dim() == 3 and t.stride(1) == 1:
        t = t.permute(0, 2, 1)
    return t.contiguous().view(torch.uint8)


@pytest.mark.parametrize("home", [0, 1])
def test_half_conversion_bitwise(home):
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.weights import (
        NV_BLOCK,
        prepare_nvfp4_checkpoint_weights,
    )

    E, H, I = 8, 512, 256
    g = torch.Generator().manual_seed(3)
    u8 = torch.uint8

    def rnd(*shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, dtype=u8, generator=g)

    w13, w2 = rnd(E, 2 * I, H // 2), rnd(E, H, I // 2)
    s13 = rnd(E, 2 * I, H // NV_BLOCK, lo=56, hi=120).view(torch.float8_e4m3fn)
    s2 = rnd(E, H, I // NV_BLOCK, lo=56, hi=120).view(torch.float8_e4m3fn)
    g13 = torch.rand(E, generator=g) + 0.5
    g2 = torch.rand(E, generator=g) + 0.5
    full = prepare_nvfp4_checkpoint_weights(
        w13, s13, w2, s2, g13, g2, hidden=H, intermediate=I
    )
    lo, hi = home * E // 2, (home + 1) * E // 2
    part = prepare_nvfp4_checkpoint_weights(
        w13[lo:hi].clone(),
        s13[lo:hi].clone(),
        w2[lo:hi].clone(),
        s2[lo:hi].clone(),
        g13[lo:hi].clone(),
        g2[lo:hi].clone(),
        hidden=H,
        intermediate=I,
    )
    for a, b in zip((*full.fc1, *full.fc2), (*part.fc1, *part.fc2)):
        assert b.shape[0] == hi - lo and a.shape[1:] == b.shape[1:]
        assert a.stride() == b.stride()
        assert torch.equal(_phys_u8(a)[lo:hi], _phys_u8(b))


# -- sharded parameters and their loader ----------------------------------------------


def _experts_cls():
    pytest.importorskip("cuda.bindings.driver")
    try:
        from vllm.models.kimi_k3.nvidia.megamoe_rubin import experts
    except ImportError as e:  # the Rubin MegaMoE stack is not importable here
        pytest.skip(f"megamoe_rubin.experts: {e}")
    return experts


def _bare_experts(mod, *, layer, start, local, resident, E=16, H=512, I=256):
    """A KimiK3MegaMoEExperts with only the state the loader path reads."""
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
        MEGA_MOE_NVFP4,
    )

    m = mod.KimiK3MegaMoEExperts.__new__(mod.KimiK3MegaMoEExperts)
    nn.Module.__init__(m)
    m.prefix = f"language_model.model.layers.{layer}.block_sparse_moe.experts"
    m.kernel_weights = None
    m.weight_format = MEGA_MOE_NVFP4
    m.num_experts, m.num_logical_experts = E, E
    m.num_local_experts, m.experts_start_idx = local, start
    m.experts_end_idx = start + local
    m.hidden_size, m.intermediate_size = H, I
    m.resident_experts = resident
    m.sharded_experts = resident != (0, local)
    m._create_nvfp4_weights()
    return m


def test_sharded_params_and_loader():
    mod = _experts_cls()
    H, I = 512, 256
    # EP rank 1 of EP2 (experts 8-15), home half = local [4, 8) = global 12-15
    m = _bare_experts(mod, layer=3, start=8, local=8, resident=(4, 8), H=H, I=I)
    assert m.w13_weight.shape == (4, 2 * I, H // 2)
    assert m.w2_weight_scale.shape[0] == 4
    assert m.w13_weight_scale_2.shape == (8, 2)  # global scales stay whole
    assert m.w2_input_scale.shape == (8,)

    w1 = torch.full((I, H // 2), 7, dtype=torch.uint8)
    w3 = torch.full((I, H // 2), 9, dtype=torch.uint8)
    # the loader passes the parameter-mapped name (expert id and shard as args)
    w13 = "layers.3.block_sparse_moe.experts.w13_weight"
    assert m.weight_loader(m.w13_weight, w1, w13, "w1", 13, return_success=True)
    m.weight_loader(m.w13_weight, w3, w13, "w3", 13)
    assert bool((m.w13_weight.data[1, :I] == 7).all())  # global 13 = resident row 1
    assert bool((m.w13_weight.data[1, I:] == 9).all())
    # a non-resident expert (global 9 = local 1) is skipped, nothing written
    before = m.w13_weight.data.clone()
    skipped = m.weight_loader(m.w13_weight, w1, w13, "w1", 9, return_success=True)
    assert skipped is False
    assert torch.equal(before, m.w13_weight.data)
    # global scales land at the local index of every expert, resident or not
    one = torch.tensor([0.25])
    m.weight_loader(m.w13_weight_scale_2, one, w13 + "_scale_2", "w1", 9)
    assert float(m.w13_weight_scale_2.data[1, 0]) == 0.25


def test_restrict_local_expert_ids():
    mod = _experts_cls()
    layers = nn.ModuleList(
        [
            _bare_experts(mod, layer=1, start=8, local=8, resident=(0, 8)),
            _bare_experts(mod, layer=2, start=8, local=8, resident=(4, 8)),
            _bare_experts(mod, layer=3, start=8, local=8, resident=(0, 4)),
        ]
    )
    default = set(range(8, 16))
    ids = mod.restrict_local_expert_ids(layers, default)
    assert isinstance(ids, LayerExpertIds)
    assert ids.layers == {2: set(range(12, 16)), 3: set(range(8, 12))}
    assert ids.default == default
    # no sharded layer: the filter is unchanged
    whole = nn.ModuleList([layers[0]])
    assert mod.restrict_local_expert_ids(whole, default) is default
