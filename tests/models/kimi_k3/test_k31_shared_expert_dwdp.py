# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-expert DWDP on the TP > 1 P/D prefill worker (dense_dwdp's shared-expert
selector): when it is on (capability, no setting), what it selects, and that its
staging slots are never refilled while a reader may still use them, under forced
skew between the main stream, the shared expert's side stream and the copy engines.

The selection and load-window equivalence tests (CPU, fake fabric) live next to the
whole-layer ones in test_k31_dense_dwdp_construction.py."""

import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.kimi_k3.nvidia import dense_dwdp


def _config(
    kv_role="kv_both",
    moe_backend="kimi_k3_mega_moe",
    tp=8,
    dp=1,
    dwdp="none",
    decode_backend=None,
):
    kv_transfer = None
    if kv_role is not None:
        kv_transfer = SimpleNamespace(
            is_kv_producer=kv_role in ("kv_producer", "kv_both"),
            is_kv_consumer=kv_role in ("kv_consumer", "kv_both"),
        )
    return SimpleNamespace(
        kv_transfer_config=kv_transfer,
        kernel_config=SimpleNamespace(moe_backend=moe_backend),
        additional_config=(
            {} if decode_backend is None else {"decode_backend": decode_backend}
        ),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            pipeline_parallel_size=1,
            data_parallel_size=dp,
            tensor_parallel_size=tp,
            dwdp=dwdp,
        ),
    )


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, True),  # TP8+DCP8 kv_both prefill worker (production P/D prefill)
        ({"kv_role": "kv_producer"}, True),
        ({"dp": 2, "tp": 4}, True),  # any SP layout with replicated shared experts
        ({"kv_role": "kv_consumer"}, False),  # decode instance
        ({"kv_role": None}, False),  # unified serving
        ({"kv_role": "kv_both", "decode_backend": "latency_specialized"}, False),
        ({"kv_role": "kv_producer", "decode_backend": "latency_specialized"}, False),
        ({"tp": 1, "dp": 8}, False),  # DEP: no SP (whole-layer dwdp="dense" instead)
        ({"moe_backend": "deep_gemm_mega_moe"}, False),  # not the Rubin MegaMoE
        ({"dwdp": "dense", "tp": 1, "dp": 8}, False),
    ],
)
def test_capability_gate(kwargs, expected):
    """On for a P/D prefill worker (kv_producer / kv_both, not the specialized
    decode) under SP on the Rubin MegaMoE; off for every other role or layout. No
    environment variable or CLI switch turns it on or off."""
    from vllm.models.kimi_k3.nvidia.model import shared_expert_dwdp

    assert shared_expert_dwdp(_config(**kwargs)) is expected


def test_capability_gate_off_when_shared_expert_tp_sharded(monkeypatch):
    """The upstream opt-in that TP-shards the SP shared expert makes it
    non-identical across ranks: nothing to share."""
    from vllm.models.kimi_k3.nvidia.model import shared_expert_dwdp

    monkeypatch.setenv("VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT", "1")
    assert not shared_expert_dwdp(_config())


def test_root_is_the_overlapped_replicated_shared_expert():
    """The selector returns a decoder layer's shared expert only for the SP
    replicated layout (``_shared_overlap``), never the layer, its attention, router
    or latent projections."""
    from vllm.models.kimi_k3.nvidia.model import KimiMoE, shared_expert_dwdp_root

    def layer(overlap):
        moe = object.__new__(KimiMoE)
        nn.Module.__init__(moe)
        moe.shared_experts = nn.Linear(4, 4)
        moe.gate = nn.Linear(4, 4)
        moe._shared_overlap = overlap
        out = nn.Module()
        out.self_attn = nn.Linear(4, 4)
        out.mlp = moe
        return out

    on = layer(True)
    assert shared_expert_dwdp_root(on) is on.mlp.shared_experts
    assert shared_expert_dwdp_root(layer(False)) is None
    dense = nn.Module()
    dense.mlp = nn.Linear(4, 4)  # the dense first layer's MLP
    assert shared_expert_dwdp_root(dense) is None


# ---- GPU: forced skew, slot reuse ----

LAYERS = 12
SLOT_BYTES = 1 << 20


def _sleep(stream, cycles):
    if cycles:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(cycles)


class _Forward:
    """The decoder-layer loop's DWDP schedule with the real copy-engine ``Copies``
    over plain device memory: layer i's weights land in staging slot i % 3; at each
    layer the shared expert (side stream, forked before the MegaMoE launch, joined
    after it) reads its slot; ``skew`` delays the attention, the MegaMoE, the side
    stream before its read and the copy engines before their copies."""

    def __init__(self):
        from vllm.models.kimi_k3.nvidia.megamoe_rubin import dwdp as edwdp

        g = torch.Generator(device="cuda").manual_seed(0)
        self.home = torch.randint(
            0, 256, (LAYERS, SLOT_BYTES), generator=g, device="cuda", dtype=torch.uint8
        )
        self.depth = dense_dwdp.DEPTH  # staging slots (fixed at load)
        self.slots = torch.empty(
            self.depth, SLOT_BYTES, device="cuda", dtype=torch.uint8
        )
        self.out = torch.empty_like(self.home)
        class SkewCopies(edwdp.Copies):
            """Copies whose copy engines first sleep ``delay`` cycles (captured
            with them: the sleeps hang off the main stream like the copies)."""

            delay = 0

            def fetch(self, streams, handles):
                if self.delay and self.jobs and not self.in_flight:
                    self.pre.record()
                    for s in streams:
                        s.wait_event(self.pre)
                        _sleep(s, self.delay)
                super().fetch(streams, handles)

        d = object.__new__(dense_dwdp.DenseDwdp)
        d.start = 0
        d.streams, d.handles = edwdp.copy_streams()
        d.copies = [SkewCopies([], len(d.streams))] + [
            SkewCopies([(self.weights(i), self.home[i])], len(d.streams))
            for i in range(1, LAYERS)
        ]
        for c in d.copies:
            c.pre = torch.cuda.Event()
        d.issue_at_moe, d._ordinal = True, None
        self.d = d
        self.side = torch.cuda.Stream()
        self.events = [(torch.cuda.Event(), torch.cuda.Event()) for _ in range(LAYERS)]

    def weights(self, i):
        return self.home[0] if i == 0 else self.slots[i % self.depth]

    def __call__(self, skew):
        main = torch.cuda.current_stream()
        d = self.d
        for i in range(LAYERS):
            d.copies[i].delay = skew[i][3]
        for i in range(LAYERS):
            attn, moe, read, _ = skew[i]  # [3]: the copy engines' delay
            d.layer_start(i)
            _sleep(main, attn)  # attention (its SP all-gather)
            fork, join = self.events[i]
            fork.record(main)
            with torch.cuda.stream(self.side):  # the shared expert
                self.side.wait_event(fork)
                torch.cuda._sleep(read) if read else None
                self.out[i].copy_(self.weights(i))
                join.record(self.side)
            if i:  # the dense first layer launches no MegaMoE
                d.moe_start()
            _sleep(main, moe)  # the MegaMoE kernel
            main.wait_event(join)

    def check(self):
        torch.cuda.synchronize()
        bad = (self.out != self.home).any(dim=1)
        return [i for i in range(LAYERS) if bad[i]]


def _skew(rng, big=60_000):
    return [
        tuple(rng.choice((0, rng.randrange(big), big)) for _ in range(4))
        for _ in range(LAYERS)
    ]


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@needs_cuda
def test_forced_skew_slot_reuse_eager():
    """1000 eager forwards, each with a fresh random skew of every stream (and the
    slots overwritten with garbage in between): every layer's shared expert reads
    exactly its own weights."""
    f = _Forward()
    rng = random.Random(1234)
    for replay in range(1000):
        f.slots.fill_(0xEE)
        f.out.zero_()
        f(_skew(rng))
        assert f.check() == [], replay


@needs_cuda
def test_forced_skew_slot_reuse_graph():
    """The same schedule captured in CUDA graphs (as at r128/r256): 4 skew patterns,
    250 replays each."""
    f = _Forward()
    rng = random.Random(99)
    graphs = []
    for _ in range(4):
        skew = _skew(rng)
        f(skew)  # warm (and leave no copy in flight)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            f(skew)
        graphs.append(graph)
    for replay in range(1000):
        f.slots.fill_(0xEE)
        f.out.zero_()
        graphs[replay % 4].replay()
        assert f.check() == [], replay


@needs_cuda
def test_forced_skew_detects_a_too_early_issue(monkeypatch):
    """Control: issuing a layer's copies one layer earlier (into the slot the
    current layer's slow shared expert still reads) is caught by the skew harness."""
    f = _Forward()
    monkeypatch.setattr(dense_dwdp, "DEPTH", dense_dwdp.DEPTH + 1)
    # layer i + 3 into slot i % 3 at layer i's MegaMoE, while its reader sleeps
    skew = [(0, 0, 200_000, 0)] * LAYERS
    f.slots.fill_(0xEE)
    f(skew)
    assert f.check() != []
