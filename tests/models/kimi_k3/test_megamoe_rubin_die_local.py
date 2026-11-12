# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-local routed-expert streaming of the Rubin MegaMoE (``die_local``).

CPU: the die mask from the SM -> die map (capability fallbacks), the chunk homes of
an expert-split tensor, the decode-server policy in ``reserve_sms_for_shared_expert``
and the kernel admission of the die-split bucket.
GPU (VR200, 2 dies): placed tensors are byte-identical copies homed as requested.
The bitwise / timing gate of the kernel is the EP4 lane gate (gate_die.py).
"""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia.megamoe_rubin import config as C
from vllm.models.kimi_k3.nvidia.megamoe_rubin import die_local as D
from vllm.models.kimi_k3.nvidia.megamoe_rubin import experts as X


class _FakeDies:
    CHUNK = 2 << 20

    def __init__(self, count=2, table=(0, 1, 1, 0) * 53, wrong=(), raises=False):
        self.count, self.table, self.wrong, self.raises = count, table, wrong, raises

    def domain_count(self, device):
        return self.count

    def sm_domains(self, device):
        if self.raises:
            raise RuntimeError("no clear gap")
        return self.table

    def stream_check(self, device):
        return self.wrong


@pytest.fixture
def fake_dies(monkeypatch):
    def install(**kw):
        fake = _FakeDies(**kw)
        monkeypatch.setattr(D, "_dies", lambda: fake)
        D.die_sm_mask32.cache_clear()
        return fake

    yield install
    D.die_sm_mask32.cache_clear()


def test_die_mask_bits(fake_dies):
    table = tuple((s // 3) % 2 for s in range(212))
    fake_dies(table=table)
    mask = D.die_sm_mask32(0)
    assert len(mask) == 7  # ceil(212 / 32)
    for s, die in enumerate(table):
        assert (mask[s // 32] >> (s % 32)) & 1 == die
    assert all(0 <= w < (1 << 32) for w in mask)


@pytest.mark.parametrize(
    "kw", [dict(count=1), dict(raises=True), dict(wrong=(5, 7))]
)
def test_die_mask_capability_fallback(fake_dies, kw):
    fake_dies(**kw)
    assert D.die_sm_mask32(0) is None


def test_chunk_domains_split():
    mib = 1 << 20
    # w13 NVFP4 codes of 56 experts: 294 MiB per half, chunk-aligned split
    nbytes = 56 * 6144 * 1792
    doms = D.chunk_domains(nbytes, nbytes // 56 * 28)
    assert len(doms) == 294 and doms == [0] * 147 + [1] * 147
    # w2 scales: 18.375 MiB per half, the straddling chunk goes to its majority
    nbytes = 56 * 3584 * 192
    split = nbytes // 56 * 28
    doms = D.chunk_domains(nbytes, split)
    assert doms.count(0) == 9 and doms[9] == 1 and len(doms) == -(-nbytes // (2 * mib))
    assert D.expert_split(56) == 28 and D.expert_split(7) == 3


def _server_buckets(tokens=2048, world=16):
    X.select_rubin_bucket.cache_clear()
    problem = C.RubinMoeProblem(quant_kind=X.rubin_quant_kind(tokens))
    buckets, _ = X.select_rubin_buckets(problem, world, tokens, prefill_worker=False)
    return problem, buckets


@pytest.mark.parametrize("prefill_worker", [False, True])
def test_reserve_sets_die_mask_on_decode_servers(monkeypatch, prefill_worker):
    mask = (0x0F0F0F0F,) * 7
    monkeypatch.setattr(D, "current_die_sm_mask32", lambda: mask)
    problem, buckets = _server_buckets()
    module = SimpleNamespace(
        bucket=buckets[-1], buckets=buckets, kernel_weights=None, prefix="layer",
        prefill_worker=prefill_worker,
    )
    X.KimiK3MegaMoEExperts.reserve_sms_for_shared_expert(module)
    want = None if prefill_worker else mask
    assert all(b.die_sm_mask32 == want for b in module.buckets)
    # die-local streams take the full grid; die-oblivious ones keep the SM cap
    cap = X.RUBIN_SHARED_OVERLAP_MAX_SMS if want is None else None
    assert all(b.max_sm_count == cap for b in module.buckets)
    assert all(b.knobs.fallback_cluster_shape_mn is None for b in module.buckets)
    for b in module.buckets:
        cfg = C.RubinMoeConfig.make(problem, b, rank=0, world_size=16)
        assert cfg.die_sm_mask32 == want


def test_reserve_without_die_map_stays_die_oblivious(monkeypatch):
    monkeypatch.setattr(D, "current_die_sm_mask32", lambda: None)
    _, buckets = _server_buckets()
    module = SimpleNamespace(
        bucket=buckets[-1], buckets=buckets, kernel_weights=None, prefix="layer",
        prefill_worker=False,
    )
    X.KimiK3MegaMoEExperts.reserve_sms_for_shared_expert(module)
    assert all(b.die_sm_mask32 is None for b in module.buckets)


def test_die_mask_admission_rules():
    problem, buckets = _server_buckets()
    capped = dataclasses.replace(
        buckets[-1],
        knobs=dataclasses.replace(buckets[-1].knobs, fallback_cluster_shape_mn=None),
        max_sm_count=148,
        die_sm_mask32=(1, 2),
    )
    C.RubinMoeConfig.make(problem, capped, rank=0, world_size=16)
    # mixed-CGA fallback clusters: refused
    with pytest.raises(ValueError):
        C.RubinMoeConfig.make(
            problem, dataclasses.replace(buckets[-1], die_sm_mask32=(1,)), rank=0,
            world_size=16,
        )
    with pytest.raises(ValueError):
        C.RubinMoeConfig.make(
            problem, dataclasses.replace(capped, die_sm_mask32=(1 << 32,)), rank=0,
            world_size=16,
        )


def test_kernel_name_and_cache_key_of_the_die_mask():
    """The kernel name (also the EP-uniform arena layout tag) only says whether the
    die split is on: ranks' die maps differ (floorsweeping). The per-GPU mask keys
    the compiled-kernel cache."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.session import (
        build_kernel,
        compile_cache_key,
    )

    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")  # a current context for the kernel frontend

    problem, buckets = _server_buckets(2048, 16)
    capped = dataclasses.replace(
        buckets[-1],
        knobs=dataclasses.replace(buckets[-1].knobs, fallback_cluster_shape_mn=None),
        max_sm_count=148,
    )
    names, keys = set(), set()
    for mask in (None, (1, 2), (2, 1)):
        cfg = C.RubinMoeConfig.make(
            problem, dataclasses.replace(capped, die_sm_mask32=mask), rank=0,
            world_size=16,
        )
        try:
            kernel = build_kernel(cfg)
        except Exception as exc:  # needs the CuTe DSL toolchain
            pytest.skip(f"kernel construction unavailable: {exc!r}")
        names.add(kernel.name())
        keys.add(compile_cache_key(cfg, kernel.name(), "v", 0))
        is_die = type(kernel.scheduler).__name__ == "DieSplitPhaseInterleavedFc12Scheduler"
        assert is_die == (mask is not None)
        assert ("_diesplit_" in kernel.name()) == (mask is not None)
    assert len(names) == 2 and len(keys) == 3


def _two_die_gpu():
    if not torch.cuda.is_available():
        return False
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    return dies.domain_count(torch.accelerator.current_device_index()) == 2


@pytest.mark.skipif(not _two_die_gpu(), reason="needs a 2-die (VR200) GPU")
def test_place_experts_homes_halves_bytewise():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.weights import RubinMoeWeights
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    dev = torch.device("cuda", torch.accelerator.current_device_index())
    g = torch.Generator(device=dev).manual_seed(0)
    e = 8

    def rnd(*shape):
        return torch.randint(0, 255, shape, generator=g, device=dev, dtype=torch.uint8)

    codes1 = rnd(e, 1024, 1024).view(W_FP4()).permute(0, 2, 1)
    codes2 = rnd(e, 512, 1024).view(W_FP4()).permute(0, 2, 1)
    s1 = rnd(e, 1 << 17).view(torch.float8_e4m3fn)
    s2 = rnd(e, 1 << 16).view(torch.float8_e4m3fn)
    w = RubinMoeWeights("nvfp4", (codes1, s1), (codes2, s2))
    placed = D.place_experts(w)
    for a, b in zip((*w.fc1, *w.fc2), (*placed.fc1, *placed.fc2)):
        assert a.shape == b.shape and a.stride() == b.stride() and a.dtype == b.dtype
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    t = placed.fc1[0].permute(0, 2, 1)
    nbytes = t.numel()
    want = D.chunk_domains(nbytes, nbytes // e * D.expert_split(e))
    assert t.data_ptr() % dies.CHUNK == 0
    homes = dies.chunk_homes(t, list(range(len(want))), 8)
    assert sum(h != d for h, d in zip(homes, want)) <= D.PLACE_UNCLEAR


def W_FP4():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.weights import fp4_storage_dtype

    return fp4_storage_dtype()
