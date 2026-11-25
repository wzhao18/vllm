# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact NVFP4 scale planes of the Rubin MegaMoE (megamoe_rubin/compact_scales.py).

Guards against: a compact plane whose doubled bytes are not the kernel's NVFP4 scale
bytes (bit for bit, at the K3.1 layer shape), compaction where it would change the math
(an unequal pair on any EP rank keeps every layer's full planes), compaction for a
bucket whose kernel does not read compact planes (GenPhase, the a368c5f66 inference
kernel, a fused shared expert), and EPLB views that are not the planes the kernel
reads. CPU only; the kernel itself: test_megamoe_rubin_compact_scales_gpu.py.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia.megamoe_rubin import compact_scales as S
from vllm.models.kimi_k3.nvidia.megamoe_rubin import weights as W

K31 = dict(hidden=3584, intermediate=3072)
M256 = SimpleNamespace(mma_tiler_mnk=(256, 128, 256))
PREFILL = SimpleNamespace(
    variant="inference_dedup_wg2", max_tokens=8192, shared=None, knobs=M256
)


def _mx_layer(gen, experts, hidden, intermediate):
    """Random canonical MXFP4 experts (exponent spread 12: exact as NVFP4)."""

    def leg(rows, cols):
        codes = torch.randint(0, 256, (experts, rows, cols // 2), generator=gen,
                              dtype=torch.uint8)  # fmt: skip
        k = torch.randint(-14, -2, (experts, rows, cols // 32), generator=gen)
        return codes, (k + 127).to(torch.uint8)

    w13, s13 = leg(2 * intermediate, hidden)
    w2, s2 = leg(hidden, intermediate)
    return w13, s13, w2, s2


def _nvfp4_layers(n, experts, hidden, intermediate, seed=0):
    """Kernel-layout NVFP4 layers from MXFP4 by the backend's exact converter."""
    gen = torch.Generator().manual_seed(seed)
    return [
        W.prepare_nvfp4_weights(
            *_mx_layer(gen, experts, hidden, intermediate),
            hidden=hidden,
            intermediate=intermediate,
        )  # fmt: skip
        for _ in range(n)
    ]


def _modules(layers, buckets=(PREFILL,)):
    return [
        SimpleNamespace(compact_scales=True, kernel_weights=w, buckets=tuple(buckets))
        for w in layers
    ]


@pytest.fixture
def one_rank(monkeypatch):
    """EP agreement as seen by one rank; ``other[0]`` is another rank's flag."""
    other = [0]

    def all_reduce(t, op, group):
        t.copy_(torch.maximum(t, torch.tensor([other[0]])))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    return other


def _planes(w):
    return w.fc1[1].view(torch.uint8).clone(), w.fc2[1].view(torch.uint8).clone()


def test_compact_planes_double_back_to_the_k31_kernel_planes(one_rank):
    """At the K3.1 layer shape the kernel's per-16 E4M3 planes (from the backend's
    exact MXFP4 -> NVFP4 conversion) are kept as half the bytes, the kernel weights
    flagged compact, and doubling every byte restores them bit for bit."""
    layers = _nvfp4_layers(3, 3, **K31)
    want = [_planes(w) for w in layers]
    mods = _modules(layers)
    assert S.install(mods, None)
    n1, n2 = want[0][0].shape[1], want[0][1].shape[1]
    assert (n1, n2) == (6144 * 224, 3584 * 192)
    for o, m in enumerate(mods):
        w = m.kernel_weights
        assert w.compact_sf and not layers[o].compact_sf
        assert w.fc1[1].shape == (3, n1 // 2) and w.fc2[1].shape == (3, n2 // 2)
        assert w.fc1[1].dtype == w.fc2[1].dtype == torch.float8_e4m3fn
        # one buffer per layer [FC1 | FC2]
        assert w.fc2[1].data_ptr() == w.fc1[1].data_ptr() + 3 * n1 // 2
        W.validate_kernel_weights(w, quant_kind="nvfp4", experts=3, **K31)
        assert w.fc1[0] is layers[o].fc1[0] and w.fc1_alpha is layers[o].fc1_alpha
        assert torch.equal(S.expand(w.fc1[1]), want[o][0])
        assert torch.equal(S.expand(w.fc2[1]), want[o][1])
    with pytest.raises(ValueError, match="scale shape"):
        W.validate_kernel_weights(mods[0].kernel_weights._replace(compact_sf=False),
                                  quant_kind="nvfp4", experts=3, **K31)  # fmt: skip


def test_eplb_views_are_the_compact_planes(one_rank):
    """EPLB moves rows of exactly the tensors the kernel reads."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.experts import KimiK3MegaMoEExperts

    layers = _nvfp4_layers(1, 4, hidden=256, intermediate=256)
    mods = _modules(layers)
    assert S.install(mods, None)
    w = mods[0].kernel_weights
    fake = SimpleNamespace(kernel_weights=w, num_local_experts=4)
    views = KimiK3MegaMoEExperts.get_expert_weights(fake)
    assert views[1].data_ptr() == w.fc1[1].data_ptr()
    assert views[1].shape == (4, w.fc1[1].shape[1])
    assert views[3].data_ptr() == w.fc2[1].data_ptr()


def test_unequal_pair_on_any_rank_keeps_full_planes(one_rank):
    """One differing per-16 pair (here, or on another EP rank) keeps every layer's
    full planes: same tensors."""
    layers = _nvfp4_layers(2, 2, hidden=256, intermediate=256)
    plane = layers[1].fc2[1].view(torch.uint8)
    plane[1, 7] ^= 0x01  # odd byte of a pair
    mods = _modules(layers)
    assert not S.install(mods, None)
    assert all(m.kernel_weights is w for m, w in zip(mods, layers))
    assert S.unequal_pairs(layers[1].fc2[1]) == 1

    clean = _nvfp4_layers(2, 2, hidden=256, intermediate=256, seed=1)
    one_rank[0] = 1  # another rank found a pair
    mods = _modules(clean)
    assert not S.install(mods, None)
    assert all(m.kernel_weights is w for m, w in zip(mods, clean))


@pytest.mark.parametrize(
    "bucket",
    [
        SimpleNamespace(variant="genphase", max_tokens=128, shared=None, knobs=M256),
        SimpleNamespace(variant="inference", max_tokens=2048, shared=None, knobs=M256),
        SimpleNamespace(variant="inference_dedup_wg2", max_tokens=8192,
                        shared=(3584, 1536), knobs=M256),
        SimpleNamespace(variant="inference_dedup_wg2", max_tokens=8192, shared=None,
                        knobs=SimpleNamespace(mma_tiler_mnk=(128, 128, 256))),
    ],
)  # fmt: skip
def test_bucket_without_a_compact_reader_keeps_full_planes(one_rank, bucket):
    """Any bucket whose kernel does not read compact planes keeps every layer's full
    planes (no expansion fallback)."""
    layers = _nvfp4_layers(1, 2, hidden=256, intermediate=256)
    mods = _modules(layers, buckets=(bucket, PREFILL))
    assert not S.install(mods, None) and mods[0].kernel_weights is layers[0]


def test_native_nvfp4_checkpoint_with_generic_scales_keeps_full_planes(one_rank):
    """A ModelOpt NVFP4 export quantized natively (independent per-16 scales, not a
    cast of MXFP4) runs as stored."""
    gen = torch.Generator().manual_seed(3)
    e, h, i = 2, 256, 256
    w13 = torch.randint(0, 256, (e, 2 * i, h // 2), generator=gen, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (e, h, i // 2), generator=gen, dtype=torch.uint8)
    u8 = dict(generator=gen, dtype=torch.uint8)
    s13 = torch.randint(0x30, 0x40, (e, 2 * i, h // 16), **u8)
    s2 = torch.randint(0x30, 0x40, (e, h, i // 16), **u8)
    g = torch.full((e,), 2.0**-13)
    w = W.prepare_nvfp4_checkpoint_weights(w13, s13, w2, s2, g, g, hidden=h,
                                           intermediate=i)  # fmt: skip
    mods = _modules([w])
    assert not S.install(mods, None) and mods[0].kernel_weights is w


@pytest.mark.parametrize(
    ("role", "extra", "on"),
    [
        (None, {}, False),  # unified serving
        ("kv_producer", {}, True),
        ("kv_both", {}, True),  # the K3.1 P/D prefill recipes
        ("kv_consumer", {}, False),
        # a decode server launched kv_both on the specialized decode (pd_prefill_worker
        # alone calls it a prefill worker)
        ("kv_both", {"decode_backend": "latency_specialized"}, False),
        ("kv_consumer", {"megamoe_compact_scales": True}, True),  # forced
        ("kv_producer", {"megamoe_compact_scales": False}, False),
    ],
)
def test_auto_policy_is_prefill_role_only(role, extra, on):
    """Auto compacts on P/D prefill workers and never on a decode instance."""
    from vllm.models.kimi_k3.nvidia.model import pd_prefill_worker

    kv = None
    if role is not None:
        kv = SimpleNamespace(
            kv_role=role,
            is_kv_producer=role in ("kv_producer", "kv_both"),
            is_kv_consumer=role in ("kv_consumer", "kv_both"),
        )
    cfg = SimpleNamespace(additional_config=extra, kv_transfer_config=kv)
    assert S.wanted(cfg, pd_prefill_worker(cfg)) is on


def test_policy_and_mx_kind(one_rank):
    """Bad knob values raise; only NVFP4 kernel layouts are touched."""
    with pytest.raises(ValueError, match="megamoe_compact_scales"):
        S.wanted(SimpleNamespace(additional_config={"megamoe_compact_scales": "on"}),
                 prefill_worker=True)  # fmt: skip

    gen = torch.Generator().manual_seed(2)
    mx = W.prepare_mxfp4_mxfp8_weights(*_mx_layer(gen, 2, 256, 256), hidden=256,
                                       intermediate=256)  # fmt: skip
    mods = _modules([mx])
    assert not S.install(mods, None) and mods[0].kernel_weights is mx
    mods = _modules(_nvfp4_layers(1, 2, hidden=256, intermediate=256))
    mods[0].compact_scales = False
    assert not S.install(mods, None)
