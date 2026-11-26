# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact NVFP4 scale planes of the Rubin MegaMoE (megamoe_rubin/scale_staging.py).

Guards against: an expanded staging plane that is not the kernel's NVFP4 scale bytes
(bit for bit, at the K3.1 layer shape), a layer reading a slot before its own planes
landed (one-ahead schedule, prefill sub-calls, two forwards, EPLB row moves between
forwards), and compaction where it would change the math (an unequal pair on any EP
rank keeps every layer's full planes). CPU, except the Triton expansion and its CUDA
graph replay at the EP8 / EP16 per-GPU expert counts.
"""

import pytest
import torch

from vllm.models.kimi_k3.nvidia.megamoe_rubin import scale_staging as S
from vllm.models.kimi_k3.nvidia.megamoe_rubin import weights as W

K31 = dict(hidden=3584, intermediate=3072)


def _mx_layer(gen, experts, hidden, intermediate, device="cpu"):
    """Random canonical MXFP4 experts (exponent spread 12: exact as NVFP4)."""

    def leg(rows, cols):
        codes = torch.randint(0, 256, (experts, rows, cols // 2), generator=gen,
                              dtype=torch.uint8)  # fmt: skip
        k = torch.randint(-14, -2, (experts, rows, cols // 32), generator=gen)
        return codes.to(device), (k + 127).to(torch.uint8).to(device)

    w13, s13 = leg(2 * intermediate, hidden)
    w2, s2 = leg(hidden, intermediate)
    return w13, s13, w2, s2


def _nvfp4_layers(n, experts, hidden, intermediate, seed=0, device="cpu"):
    """Kernel-layout NVFP4 layers from MXFP4 by the backend's exact converter."""
    gen = torch.Generator().manual_seed(seed)
    return [
        W.prepare_nvfp4_weights(
            *_mx_layer(gen, experts, hidden, intermediate, device),
            hidden=hidden,
            intermediate=intermediate,
        )  # fmt: skip
        for _ in range(n)
    ]


def _modules(layers):
    from types import SimpleNamespace

    return [
        SimpleNamespace(
            compact_scales=True, kernel_weights=w, scale_staging=None, moe_ordinal=-1
        )  # fmt: skip
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


def test_expand_restores_the_k31_kernel_planes_bitwise(one_rank):
    """At the K3.1 layer shape the kernel's per-16 E4M3 planes (from the backend's
    exact MXFP4 -> NVFP4 conversion) are kept as half the bytes and expand back to
    the same bytes; each layer's kernel tensors view slot ``ordinal % 2``."""
    layers = _nvfp4_layers(3, 3, **K31)
    want = [_planes(w) for w in layers]
    mods = _modules(layers)
    staging = S.install(mods, None)
    assert staging is not None
    n1, n2 = want[0][0].shape[1], want[0][1].shape[1]
    assert (n1, n2) == (6144 * 224, 3584 * 192)
    for o, m in enumerate(mods):
        c1, c2 = staging.legs[o]
        assert c1.shape == (3, n1 // 2) and c2.shape == (3, n2 // 2)
        assert staging.compact[o].numel() == 3 * (n1 + n2) // 2
        w = m.kernel_weights
        W.validate_kernel_weights(w, quant_kind="nvfp4", experts=3, **K31)
        assert w.fc1[1].data_ptr() == staging.slots[o % 2].data_ptr()
        assert w.fc2[1].data_ptr() == staging.slots[o % 2].data_ptr() + 3 * n1
        assert w.fc1[0] is layers[o].fc1[0] and w.fc1_alpha is layers[o].fc1_alpha
    for o, m in enumerate(mods):
        staging.before_kernel(o)
        assert torch.equal(m.kernel_weights.fc1[1].view(torch.uint8), want[o][0])
        assert torch.equal(m.kernel_weights.fc2[1].view(torch.uint8), want[o][1])
        staging.after_kernel(o)


def test_one_ahead_schedule_survives_subcalls_forwards_and_eplb(one_rank):
    """Each layer's kernel sees its own planes: prefill sub-calls (a layer launched
    twice), a second forward, and EPLB moving compact rows between forwards."""
    layers = _nvfp4_layers(5, 4, hidden=256, intermediate=256)
    want = [_planes(w) for w in layers]
    mods = _modules(layers)
    staging = S.install(mods, None)
    expanded = []
    real = staging._expand
    staging._expand = lambda o: (expanded.append(o), real(o))

    def forward(calls_per_layer):
        for o, m in enumerate(mods):
            for _ in range(calls_per_layer[o]):
                staging.before_kernel(o)
                got = _planes(m.kernel_weights)
                assert torch.equal(got[0], want[o][0])
                assert torch.equal(got[1], want[o][1])
                staging.after_kernel(o)

    forward([1, 2, 1, 1, 3])
    # every layer once; the last layer's later sub-calls re-expand it after the
    # forward-end reset
    assert expanded == [0, 1, 2, 3, 4, 4, 4]
    expanded.clear()
    # EPLB swaps local experts 0 and 2 of layer 3 (compact rows, as it moves views)
    c1, c2 = staging.legs[3]
    for leg, full in ((c1, want[3][0]), (c2, want[3][1])):
        leg[[0, 2]] = leg[[2, 0]]
        full[[0, 2]] = full[[2, 0]]
    forward([1] * 5)
    assert expanded == [0, 1, 2, 3, 4]


def test_unequal_pair_on_any_rank_keeps_full_planes(one_rank):
    """One differing per-16 pair (here, or on another EP rank) keeps every layer's
    full planes: same tensors, nothing staged."""
    layers = _nvfp4_layers(2, 2, hidden=256, intermediate=256)
    plane = layers[1].fc2[1].view(torch.uint8)
    plane[1, 7] ^= 0x01  # odd byte of a pair
    mods = _modules(layers)
    assert S.install(mods, None) is None
    assert all(m.kernel_weights is w and m.scale_staging is None
               for m, w in zip(mods, layers))  # fmt: skip
    assert S.unequal_pairs(layers[1].fc2[1]) == 1

    clean = _nvfp4_layers(2, 2, hidden=256, intermediate=256, seed=1)
    one_rank[0] = 1  # another rank found a pair
    mods = _modules(clean)
    assert S.install(mods, None) is None
    assert all(m.kernel_weights is w for m, w in zip(mods, clean))


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
    assert S.install(mods, None) is None and mods[0].kernel_weights is w


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
def test_auto_policy_is_prefill_role_only(role, extra, on, monkeypatch):
    """The previous auto policy (K31_COMPACT_SCALES_AUTO_PREFILL=1) compacts on
    P/D prefill workers and never on a decode instance; the default auto keeps
    full planes everywhere (PF-P3: the expansion is not hidden on DEP prefill)."""
    from types import SimpleNamespace

    from vllm.models.kimi_k3.nvidia.model import pd_prefill_worker

    kv = None
    if role is not None:
        kv = SimpleNamespace(
            kv_role=role,
            is_kv_producer=role in ("kv_producer", "kv_both"),
            is_kv_consumer=role in ("kv_consumer", "kv_both"),
        )
    cfg = SimpleNamespace(additional_config=extra, kv_transfer_config=kv)
    forced = "megamoe_compact_scales" in extra
    monkeypatch.delenv("K31_COMPACT_SCALES_AUTO_PREFILL", raising=False)
    assert S.wanted(cfg, pd_prefill_worker(cfg)) is (on if forced else False)
    monkeypatch.setenv("K31_COMPACT_SCALES_AUTO_PREFILL", "1")
    assert S.wanted(cfg, pd_prefill_worker(cfg)) is on


def test_policy_and_mx_kind(one_rank):
    """Bad knob values raise; only NVFP4 kernel layouts are touched."""
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="megamoe_compact_scales"):
        S.wanted(SimpleNamespace(additional_config={"megamoe_compact_scales": "on"}),
                 prefill_worker=True)  # fmt: skip

    gen = torch.Generator().manual_seed(2)
    mx = W.prepare_mxfp4_mxfp8_weights(*_mx_layer(gen, 2, 256, 256), hidden=256,
                                       intermediate=256)  # fmt: skip
    mods = _modules([mx])
    assert S.install(mods, None) is None and mods[0].kernel_weights is mx
    mods = _modules(_nvfp4_layers(1, 2, hidden=256, intermediate=256))
    mods[0].compact_scales = False
    assert S.install(mods, None) is None


# -- CUDA: the Triton expansion ------------------------------------------------------

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("experts", [113, 57])  # K3.1 EP8 / EP16 local experts
def test_triton_expand_matches_the_plane_bitwise(experts):
    gen = torch.Generator(device="cuda").manual_seed(experts)
    n = experts * (6144 * 224 + 3584 * 192)
    plane = torch.randint(0, 0x7F, (n // 2,), generator=gen, device="cuda",
                          dtype=torch.uint8).repeat_interleave(2)  # fmt: skip
    compact = plane[0::2].contiguous()
    out = torch.full_like(plane, 0xFF)
    S.expand(compact, out)
    assert torch.equal(out, plane)


@cuda
def test_side_stream_schedule_under_cuda_graph_replay(one_rank):
    """A captured forward holds every expansion: replays after the slots were
    overwritten (another forward) still give each layer its own planes."""
    layers = _nvfp4_layers(4, 8, hidden=512, intermediate=512, device="cuda")
    want = [_planes(w) for w in layers]
    mods = _modules(layers)
    staging = S.install(mods, None)
    seen = [torch.empty_like(torch.cat(p, dim=1)) for p in want]

    def forward():
        for o, m in enumerate(mods):
            staging.before_kernel(o)
            # stands in for the kernel: reads its slot on the current stream
            seen[o].copy_(torch.cat(_planes_view(m.kernel_weights), dim=1))
            staging.after_kernel(o)

    forward()  # eager: compiles, warms
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    for s in seen:
        s.zero_()
    staging.slots[0].fill_(0xEE)
    staging.slots[1].fill_(0xEE)
    graph.replay()
    torch.cuda.synchronize()
    for o in range(4):
        assert torch.equal(seen[o], torch.cat(want[o], dim=1)), o


def _planes_view(w):
    return w.fc1[1].view(torch.uint8), w.fc2[1].view(torch.uint8)


@cuda
def test_breakable_capture_with_eager_breaks_between_layers(one_rank):
    """Production prefill graphs are breakable (piecewise) captures: an eager break
    (attention) ends the segment between two MoE layers. The one-ahead expansion must
    not fork a side stream there (cudaErrorStreamCaptureUnjoined at the break); the
    replay still gives each layer its own planes."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    layers = _nvfp4_layers(4, 8, hidden=512, intermediate=512, device="cuda")
    want = [_planes(w) for w in layers]
    mods = _modules(layers)
    staging = S.install(mods, None)
    seen = [torch.empty_like(torch.cat(p, dim=1)) for p in want]
    eager_runs = [0]

    def attention():  # an eager break, as kda.py / the MLA attention ops
        eager_runs[0] += 1

    def forward(cap):
        for o, m in enumerate(mods):
            if cap is not None and o:
                cap.add_eager(attention)
            staging.before_kernel(o)
            seen[o].copy_(torch.cat(_planes_view(m.kernel_weights), dim=1))
            staging.after_kernel(o)
            if cap is not None:
                assert staging.pending is None  # nothing forked to the side stream

    forward(None)  # eager: compiles, warms
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cap = BreakableCUDAGraphCapture()
        with cap:
            forward(cap)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert cap.num_eager_breaks == 3 and cap.num_graphs == 4
    for s in seen:
        s.zero_()
    staging.slots[0].fill_(0xEE)
    staging.slots[1].fill_(0xEE)
    cap.replay()
    torch.cuda.synchronize()
    for o in range(4):
        assert torch.equal(seen[o], torch.cat(want[o], dim=1)), o
    # an eager forward after the capture still uses the side stream
    forward(None)
    torch.cuda.synchronize()
    for o in range(4):
        assert torch.equal(seen[o], torch.cat(want[o], dim=1)), o
