# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Rubin MegaMoE prefill kernel reading compact NVFP4 scale planes, on one VR200.

Guards against: a kernel that reads the compact planes (``RubinMoeWeights.compact_sf``:
the even bytes of the swizzled per-16 planes, ``compact_scales``) and computes anything
but what it computes from the full planes. The prefill buckets (refreshed inference
kernel, dedup combine, two epilogue warpgroups, 2-CTA M256 tiles) of the EP8 per-GPU
K3.1 shape (112 local experts, top-16, 3584 x 3072) on the shipped EP8 NVFP4 knob
records (N128 and N256 tiles; the SM-capped small bucket of a sequence-parallel
prefill worker) must give BITWISE-equal outputs on both planes, the full planes twice
(the kernel is deterministic, so the comparison is meaningful), and a different output
when one compact scale byte changes (the kernel reads the compact tensor).
"""

import dataclasses

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin MegaMoE kernel needs a VR200 (sm_107a)",
)

EXPERTS = 112  # K3.1 EP8 per-GPU experts
TOP_K = 16
# (record tile N, capacity, SM cap): the TP8+DCP8 prefill worker's buckets on its
# records (N128 and N256 i2048), and the sequence-parallel small bucket (148 SMs).
BUCKETS = {
    "n128_c2048": (128, 2048, None),
    "n256_c2048": (256, 2048, None),
    "n128_c1024_cap148": (128, 1024, 148),
}


def _mods():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import compact_scales as S
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import config as C
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import experts as X
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import weights as W
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.runtime import RubinMegaMoE

    return C, S, W, X, RubinMegaMoE


def _mx_layer(gen, experts, hidden, intermediate, device):
    """Random canonical MXFP4 experts (exponent spread 12: exact as NVFP4)."""

    def leg(rows, cols):
        codes = torch.randint(0, 256, (experts, rows, cols // 2), generator=gen,
                              dtype=torch.uint8, device=device)  # fmt: skip
        k = torch.randint(-14, -2, (experts, rows, cols // 32), generator=gen,
                          device=device)  # fmt: skip
        return codes, (k + 127).to(torch.uint8)

    w13, s13 = leg(2 * intermediate, hidden)
    w2, s2 = leg(hidden, intermediate)
    return w13, s13, w2, s2


def _bucket(C, X, entries, tile_n, cap, sms):
    records = [
        r for r in entries
        if r["dtype"] == "nvfp4" and r["world_size"] == 8 and r["max_tokens"] == 2048
        and r["kernel_variant"] == "inference"
        and r["knobs"]["mma_tiler_mnk"][1] == tile_n
    ]  # fmt: skip
    assert records, f"no shipped EP8 NVFP4 i2048 record with tile N {tile_n}"
    rec = min(records, key=lambda r: r.get("p50_us") or float("inf"))
    knobs = C.RubinKnobs.from_dict(rec["knobs"])
    bucket = X._prefill_dedup(C.RubinBucket("inference", 4096, knobs))
    assert bucket.variant in C.DEDUP_VARIANTS
    bucket = dataclasses.replace(bucket, max_tokens=cap)
    if sms is not None:
        bucket = dataclasses.replace(
            bucket,
            knobs=dataclasses.replace(bucket.knobs, fallback_cluster_shape_mn=None),
            max_sm_count=sms,
        )
    return bucket


@pytest.fixture(scope="module")
def build():
    C, S, W, X, RubinMegaMoE = _mods()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.zeros(1, device=device)  # current primary context before the DSL queries it
    entries = C.load_knob_source(C.SHIPPED_KNOBS)["entries"]
    problem = C.RubinMoeProblem(num_experts=EXPERTS, quant_kind="nvfp4")
    gen = torch.Generator(device=device).manual_seed(3)
    full = W.prepare_nvfp4_weights(
        *_mx_layer(gen, EXPERTS, problem.hidden, problem.intermediate, device),
        hidden=problem.hidden,
        intermediate=problem.intermediate,
    )
    assert S.unequal_pairs(full.fc1[1]) == 0 and S.unequal_pairs(full.fc2[1]) == 0
    e4m3 = torch.float8_e4m3fn
    legs = []
    for plane in (full.fc1[1], full.fc2[1]):
        c = torch.empty(plane.shape[0], plane.shape[1] // 2, dtype=e4m3, device=device)
        S.compact(plane, c)
        assert torch.equal(S.expand(c), plane.view(torch.uint8))
        legs.append(c)
    compact = full._replace(
        fc1=(full.fc1[0], legs[0]), fc2=(full.fc2[0], legs[1]), compact_sf=True
    )
    buckets = [_bucket(C, X, entries, *spec) for spec in BUCKETS.values()]
    moe = RubinMegaMoE(problem, None, buckets)
    moe.warmup(full, ids_dtypes=(torch.int32,))
    moe.warmup(compact, ids_dtypes=(torch.int32,))
    return problem, moe, dict(zip(BUCKETS, buckets)), full, compact


def _routes(n, experts, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    ids = torch.argsort(torch.rand(n, experts, generator=gen, device=device), dim=1)
    ids = ids[:, :TOP_K].to(torch.int32)
    ids[ids == experts - 1] = -1  # one expert never routed; masked routes
    wts = torch.rand(n, TOP_K, generator=gen, device=device)
    return ids, wts / wts.sum(dim=1, keepdim=True)


def _run(moe, bucket, weights, hidden, ids, wts):
    out = torch.full_like(hidden, float("nan"))
    moe.forward(hidden, ids, wts, weights, bucket=bucket, out=out)
    torch.cuda.synchronize()
    moe.check_route_errors()
    return out


@pytest.mark.parametrize(
    "name,n",
    [("n128_c2048", 1), ("n128_c2048", 333), ("n128_c2048", 2048),
     ("n256_c2048", 1500), ("n128_c1024_cap148", 128), ("n128_c1024_cap148", 512),
     ("n128_c1024_cap148", 1024)],
)  # fmt: skip
def test_compact_scale_planes_give_bitwise_equal_outputs(build, name, n):
    problem, moe, buckets, full, compact = build
    bucket = buckets[name]
    device = torch.device("cuda", 0)
    gen = torch.Generator(device=device).manual_seed(100 + n)
    hidden = torch.randn(n, problem.hidden, generator=gen, device=device)
    hidden = hidden.to(torch.bfloat16)
    ids, wts = _routes(n, problem.num_experts, device, seed=200 + n)
    want = _run(moe, bucket, full, hidden, ids, wts)
    again = _run(moe, bucket, full, hidden, ids, wts)
    got = _run(moe, bucket, compact, hidden, ids, wts)
    assert bool(torch.isfinite(want).all())
    assert torch.equal(again.view(torch.int16), want.view(torch.int16)), "nondeterministic"
    diff = int((got.view(torch.int16) != want.view(torch.int16)).sum())
    assert diff == 0, f"{name} n={n}: {diff} of {want.numel()} outputs differ"


def test_compact_kernel_reads_the_compact_planes(build):
    """One compact FC2 scale byte of a routed expert doubled: the output moves."""
    problem, moe, buckets, full, compact = build
    bucket = buckets["n128_c2048"]
    device = torch.device("cuda", 0)
    n = 256
    gen = torch.Generator(device=device).manual_seed(9)
    hidden = torch.randn(n, problem.hidden, generator=gen, device=device)
    hidden = hidden.to(torch.bfloat16)
    ids, wts = _routes(n, problem.num_experts, device, seed=10)
    expert = int(ids[0, 0])
    base = _run(moe, bucket, compact, hidden, ids, wts)
    fc2_sf = compact.fc2[1].clone()
    row = fc2_sf[expert].view(torch.uint8)
    row[: row.numel() // 2] += 8  # E4M3 exponent + 1 on the first rows' scales
    changed = compact._replace(fc2=(compact.fc2[0], fc2_sf))
    moved = _run(moe, bucket, changed, hidden, ids, wts)
    assert not torch.equal(moved.view(torch.int16), base.view(torch.int16))
