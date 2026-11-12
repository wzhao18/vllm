# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rubin MegaMoE backend (vllm/models/kimi_k3/nvidia/megamoe_rubin), CPU only:

- exact MXFP4 -> NVFP4 weight conversion: bit-identical dequantized weights for every
  representable expert (scale exponent spread <= 17, incl. the subnormal-scale 15..17
  window, all-zero and signed-zero blocks, tiny exponents), unchanged codes, the
  closed-form E4M3 scale bytes, exact per-expert alphas;
- detection of non-representable weights (spread 18, NaN scale byte, byte-0 scale on
  live codes) with the offending (expert, leg) named;
- the MXFP4 (``mxfp4_mxfp8``) kernel layout equals FlashInfer's
  ``preprocess_prequantized_block_scaled_weights`` byte for byte;
- the knob reader on the shipped VR knob file, the knob merge (record count per
  source file) and the bench capacity selection;
- checkpoint expert-key parsing (``weight_packed`` / ``weight_scale``);
- compiled-kernel / output-binding cache keys and the arena layout tag;
- the S2 epilogue dispatch policy (capped grids only, per-launch crossover, shared
  workspace layout) and the EP-mean live token count;
- rejection of out-of-range expert ids (only -1 masks) in the torch staging twin;
- power-of-two NVFP4 norm constants from activation amax;
- the vendored kernel tree still matches the VENDOR.md sha256 manifest.

The backend package is loaded under an alias straight from its directory, so the test
imports only torch and the package's CPU-safe modules (not vllm.models.kimi_k3, not
cutlass / triton).
"""

import hashlib
import importlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest
import torch

PKG_DIR = (
    Path(__file__).resolve().parents[3]
    / "vllm"
    / "models"
    / "kimi_k3"
    / "nvidia"
    / "megamoe_rubin"
)
_ALIAS = "_megamoe_rubin_cpu_test"


def _load(module: str):
    if _ALIAS not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            _ALIAS, PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)]
        )
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[_ALIAS] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module(f"{_ALIAS}.{module}")


W = _load("weights")

E, H = 4, 128


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def _random_leg(gen, experts, rows, cols, *, kmax, spread, zero_frac=0.2):
    """Random MXFP4 leg (codes (E, rows, cols/2), UE8M0 (E, rows, cols/32)) whose live
    blocks span exactly [kmax - spread, kmax] per expert; zero blocks mix signed zeros
    and carry arbitrary (ignored) scale bytes."""
    nb = cols // 32
    codes = torch.randint(
        0, 256, (experts, rows, nb, 16), generator=gen, dtype=torch.uint8
    )
    # make every live block live for sure: force one non-zero nibble
    codes[..., 0] = (codes[..., 0] & 0xF0) | 0x05
    zero = torch.rand((experts, rows, nb), generator=gen) < zero_frac
    signed_zeros = torch.tensor([0x00, 0x08, 0x80, 0x88], dtype=torch.uint8)
    zfill = signed_zeros[torch.randint(0, 4, (experts, rows, nb, 16), generator=gen)]
    codes = torch.where(zero.unsqueeze(-1), zfill, codes)
    k = torch.randint(kmax - spread, kmax + 1, (experts, rows, nb), generator=gen)
    # pin both ends on a live block of every expert
    zero[:, 0, 0] = False
    zero[:, 0, 1] = False
    codes[:, 0, 0, 0] = 0x05
    codes[:, 0, 1, 0] = 0x05
    k[:, 0, 0] = kmax
    k[:, 0, 1] = kmax - spread
    scales = (k + 127).to(torch.uint8)
    junk = torch.randint(0, 255, (experts, rows, nb), generator=gen, dtype=torch.uint8)
    scales = torch.where(zero, junk, scales)
    return codes.reshape(experts, rows, cols // 2), scales


def _random_layer(seed, intermediate, *, kmax=-4, spread=13, kmax2=None, spread2=None):
    gen = torch.Generator().manual_seed(seed)
    w13, s13 = _random_leg(gen, E, 2 * intermediate, H, kmax=kmax, spread=spread)
    w2, s2 = _random_leg(
        gen,
        E,
        H,
        intermediate,
        kmax=kmax if kmax2 is None else kmax2,
        spread=spread if spread2 is None else spread2,
    )
    return w13, s13, w2, s2


def _expected_bytes(m: int) -> int:
    return (m + 7) << 3 if m >= -6 else 1 << (m + 9)


def test_e4m3_power_of_two_table():
    for m in range(-9, 9):
        byte = int(W._e4m3_pow2_bytes(torch.tensor([m], dtype=torch.int32))[0])
        assert byte == _expected_bytes(m)
        value = (
            torch.tensor([byte], dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        )
        assert float(value[0]) == 2.0**m


@pytest.mark.parametrize(
    "kmax,spread,intermediate",
    [
        (-4, 13, 64),  # the K3.1 checkpoint's worst spread (scale scan, Nov 2026)
        (-4, 3, 128),
        (-5, 14, 64),  # last spread without subnormal scales
        (-3, 15, 64),  # subnormal 2^-7
        (0, 17, 128),  # widest exact spread: 2^-9 .. 2^8
        (20, 17, 64),
        (-100, 16, 64),
        (-120, 7, 64),  # c clamps to -126 (fp32-normal global scale); byte 0 is live
    ],
)
def test_nvfp4_conversion_is_bit_exact(kmax, spread, intermediate):
    w13, s13, w2, s2 = _random_layer(
        1234 + spread, intermediate, kmax=kmax, spread=spread
    )
    conv = W.convert_mxfp4_to_nvfp4(
        w13, s13, w2, s2, hidden=H, intermediate=intermediate, layer_id=7
    )
    assert not conv.report.offending
    assert conv.report.max_spread == spread
    # codes unchanged
    assert torch.equal(conv.w13_codes, w13) and torch.equal(conv.w2_codes, w2)
    for codes, s32, s16, c in (
        (w13, s13, conv.w13_scale16, conv.c1),
        (w2, s2, conv.w2_scale16, conv.c2),
    ):
        ref = W.dequant_mxfp4(codes, s32)
        got = W.dequant_nvfp4(codes, s16, c)
        assert torch.equal(_bits(ref), _bits(got))
        # closed-form scale bytes: 2^(k - c) on live blocks, 0 on zero blocks
        live = ((codes.view(E, codes.shape[1], -1, 16) & 0x77) != 0).any(-1)
        m = s32.to(torch.int32) - 127 - c.view(-1, 1, 1)
        expect = torch.where(live, W._e4m3_pow2_bytes(m), 0).to(torch.uint8)
        assert torch.equal(s16, expect.repeat_interleave(2, dim=-1))
        assert torch.equal(c, torch.clamp(torch.full_like(c, kmax - 8), min=-126))
    # subnormal E4M3 scales appear exactly when the smallest m = kmin - c is below -6
    smallest_m = (kmax - spread) - max(kmax - 8, -126)
    sub = sum(s.subnormal for s in conv.report.stats)
    assert (sub > 0) == (smallest_m < -6)


def test_nvfp4_kernel_layout_round_trips():
    intermediate = 64
    w13, s13, w2, s2 = _random_layer(7, intermediate, kmax=-2, spread=16)
    w = W.prepare_nvfp4_weights(
        w13,
        s13,
        w2,
        s2,
        hidden=H,
        intermediate=intermediate,
        input_norm_const=4.0,
        fc1_norm_const=0.5,
    )
    conv = W.convert_mxfp4_to_nvfp4(
        w13, s13, w2, s2, hidden=H, intermediate=intermediate
    )
    shapes = W.kernel_weight_shapes("nvfp4", E, H, intermediate)
    W.validate_kernel_weights(
        w, quant_kind="nvfp4", experts=E, hidden=H, intermediate=intermediate
    )
    assert tuple(w.fc1[0].shape) == shapes[0] and tuple(w.fc2[0].shape) == shapes[1]
    # data: K-major physical == gate/up interleaved codes
    phys13 = w.fc1[0].permute(0, 2, 1).contiguous().view(torch.uint8)
    assert torch.equal(
        W.deinterleave_gate_up_16(phys13, intermediate_size=intermediate), w13
    )
    assert torch.equal(w.fc2[0].permute(0, 2, 1).contiguous().view(torch.uint8), w2)
    # scales: swizzle round trip, interleave applied identically to data and scales
    for e in range(E):
        plane = W.from_blocked(w.fc1[1][e], 2 * intermediate, H // 16)
        assert torch.equal(
            W.deinterleave_gate_up_16(plane, intermediate_size=intermediate),
            conv.w13_scale16[e],
        )
        assert torch.equal(
            W.from_blocked(w.fc2[1][e], H, intermediate // 16), conv.w2_scale16[e]
        )
    # alphas: 2^c / norm, exact
    assert torch.equal(w.fc1_alpha.double(), torch.pow(2.0, conv.c1.double()) / 4.0)
    assert torch.equal(w.fc2_alpha.double(), torch.pow(2.0, conv.c2.double()) / 0.5)
    assert torch.equal(w.fc1_norm_const, torch.full((E,), 0.5))
    # alpha folding reproduces the MXFP4 weight: alpha * dequant(kernel layout) * norm
    deq = W.dequant_nvfp4(w2, conv.w2_scale16, torch.zeros_like(conv.c2))
    assert torch.equal(
        _bits(deq * w.fc2_alpha.view(-1, 1, 1) * 0.5), _bits(W.dequant_mxfp4(w2, s2))
    )


def test_to_blocked_batched_matches_flashinfer():
    gen = torch.Generator().manual_seed(3)
    for rows, cols in ((256, 8), (128, 4), (200, 6), (384, 112)):
        s = torch.randint(0, 256, (3, rows, cols), generator=gen, dtype=torch.uint8)
        ref = torch.stack([W.to_blocked(x) for x in s])
        got = W.to_blocked_batched(s)
        assert torch.equal(ref, got)
        assert got.shape[1] == W.swizzled_flat_sf_size(rows, cols)
        for e in range(3):
            assert torch.equal(W.from_blocked(got[e], rows, cols), s[e])


def _fi_preprocess_prequantized_mx(w13, w2, w13_scale, w2_scale, intermediate):
    """FlashInfer 0.7.3.dev20261031 ``preprocess_prequantized_block_scaled_weights``
    (quant_kind="mxfp4_mxfp8") transform, verbatim modulo imports."""

    def transform(weight, scale, *, gate_up):
        data = weight.view(torch.uint8)
        sf = scale.view(torch.uint8)
        if gate_up:
            data = W.interleave_gate_up_16(data, intermediate_size=intermediate)
            sf = W.interleave_gate_up_16(sf, intermediate_size=intermediate)
        swizzled = torch.stack([W.to_blocked(s) for s in sf])
        data = data.contiguous()
        if data.data_ptr() % 16:
            data = data.clone()
        return (
            data.view(W.fp4_storage_dtype()).permute(0, 2, 1),
            swizzled.view(torch.float8_e8m0fnu),
        )

    return transform(w13, w13_scale, gate_up=True), transform(
        w2, w2_scale, gate_up=False
    )


def test_mxfp4_mxfp8_layout_matches_flashinfer():
    intermediate = 128
    w13, s13, w2, s2 = _random_layer(11, intermediate, kmax=-4, spread=40)
    ours = W.prepare_mxfp4_mxfp8_weights(
        w13, s13.view(torch.float8_e8m0fnu), w2, s2, hidden=H, intermediate=intermediate
    )
    ref1, ref2 = _fi_preprocess_prequantized_mx(w13, w2, s13, s2, intermediate)
    for (a, b), (c, d) in ((ours.fc1, ref1), (ours.fc2, ref2)):
        assert a.dtype == c.dtype and b.dtype == d.dtype
        assert a.shape == c.shape and b.shape == d.shape
        assert torch.equal(a.view(torch.uint8), c.view(torch.uint8))
        assert torch.equal(b.view(torch.uint8), d.view(torch.uint8))
    W.validate_kernel_weights(
        ours, quant_kind="mxfp4_mxfp8", experts=E, hidden=H, intermediate=intermediate
    )
    assert ours.fc1_alpha is None and ours.report is None


def _offenders(exc):
    return {(s.leg, s.expert) for s in exc.report.offending}


def test_spread_18_is_rejected():
    intermediate = 64
    w13, s13, w2, s2 = _random_layer(5, intermediate, kmax=-4, spread=13)
    # widen expert 2's fc2 spread to 18 on one live block
    s2[2, 0, 1] = (-4 - 18) + 127
    with pytest.raises(W.NonRepresentableError) as info:
        W.convert_mxfp4_to_nvfp4(
            w13,
            s13,
            w2,
            s2,
            hidden=H,
            intermediate=intermediate,
            layer_id=3,
            expert_offset=100,
        )
    assert _offenders(info.value) == {("fc2", 102)}
    (bad,) = info.value.report.offending
    assert (bad.kmin, bad.kmax, bad.spread, bad.nonrep, bad.nan) == (-22, -4, 18, 1, 0)
    assert "layer 3 leg fc2 expert 102" in str(info.value)
    # the non-raising scan reports the same and leaves other entries clean
    conv = W.scan_mxfp4_to_nvfp4(w13, s13, w2, s2, hidden=H, intermediate=intermediate)
    assert {(s.leg, s.expert) for s in conv.report.offending} == {("fc2", 2)}


def test_nan_scale_and_byte0_on_live_codes_are_rejected():
    intermediate = 64
    w13, s13, w2, s2 = _random_layer(9, intermediate, kmax=-4, spread=10)
    s13[1, 5, 2] = 255  # NaN scale (even on a zero block it is rejected)
    w13.view(E, 2 * intermediate, -1, 16)[3, 7, 1] = 0x01  # live codes ...
    s13[3, 7, 1] = 0  # ... under a byte-0 (2^-127) scale: spread 123
    with pytest.raises(W.NonRepresentableError) as info:
        W.convert_mxfp4_to_nvfp4(w13, s13, w2, s2, hidden=H, intermediate=intermediate)
    assert _offenders(info.value) == {("fc1", 1), ("fc1", 3)}
    stats = {(s.leg, s.expert): s for s in info.value.report.offending}
    assert stats[("fc1", 1)].nan == 1
    assert stats[("fc1", 3)].kmin == -127 and stats[("fc1", 3)].nonrep == 1


def test_byte0_on_zero_block_is_fine():
    intermediate = 64
    w13, s13, w2, s2 = _random_layer(13, intermediate, kmax=-4, spread=13)
    w13.view(E, 2 * intermediate, -1, 16)[0, 3, 2] = 0x88  # all signed zeros
    s13[0, 3, 2] = 0
    conv = W.convert_mxfp4_to_nvfp4(
        w13, s13, w2, s2, hidden=H, intermediate=intermediate
    )
    assert conv.w13_scale16[0, 3, 4] == 0 and conv.w13_scale16[0, 3, 5] == 0


def test_alpha_norm_consts():
    c = torch.tensor([-10, 0, 5], dtype=torch.int32)
    a1, a2, n1 = W.nvfp4_alphas(c, c, input_norm_const=2.0**-3, fc1_norm_const=8.0)
    assert torch.equal(a1.double(), torch.pow(2.0, c.double()) * 8.0)
    assert torch.equal(a2.double(), torch.pow(2.0, c.double()) / 8.0)
    assert torch.equal(n1, torch.full((3,), 8.0))
    for bad in (3.0, 0.0, -2.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            W.nvfp4_alphas(c, c, input_norm_const=bad, fc1_norm_const=1.0)
    with pytest.raises(ValueError):  # 2^-126 / 2^30 underflows FP32
        W.nvfp4_alphas(
            torch.tensor([-126], dtype=torch.int32),
            c[:1],
            input_norm_const=2.0**30,
            fc1_norm_const=1.0,
        )


def test_shipped_knobs_resolve():
    C = _load("config")
    problem = C.RubinMoeProblem(quant_kind="nvfp4")
    knobs, record, _ = C.resolve_knobs(
        C.SHIPPED_KNOBS,
        problem=problem,
        variant="inference",
        world_size=16,
        max_tokens=6144,
        live_tokens=5925,
    )
    assert record["live_tokens"] == 5925
    assert record["knobs"]["mma_tiler_mnk"] == list(knobs.mma_tiler_mnk)
    # equal-capacity ties (several tuner runs of one shape) resolve to the lowest p50
    import json

    with open(C.SHIPPED_KNOBS) as f:
        same = [
            e
            for e in json.load(f)["entries"]
            if e["kernel_variant"] == "inference"
            and e["dtype"] == "nvfp4"
            and e["world_size"] == 16
            and e["max_tokens"] == 6144
            and e["live_tokens"] == 5925
        ]
    assert record["p50_us"] == min(e["p50_us"] for e in same)
    mx = C.RubinMoeProblem(quant_kind="mxfp4_mxfp8")
    knobs, _, _ = C.resolve_knobs(
        C.SHIPPED_KNOBS,
        problem=mx,
        variant="inference",
        world_size=8,
        max_tokens=5760,
        live_tokens=5760,
    )
    assert knobs.fallback_cluster_shape_mn == (2, 1)
    assert knobs.schedule_policy == ("phase_interleave", None)
    # every shipped record passes the ported admission rules
    import json

    with open(C.SHIPPED_KNOBS) as f:
        shipped = json.load(f)["entries"]
    for entry in shipped:
        p = C.RubinMoeProblem(quant_kind=entry["dtype"])
        bucket = C.RubinBucket(
            entry["kernel_variant"],
            entry["max_tokens"],
            C.RubinKnobs.from_dict(entry["knobs"]),
        )
        C.RubinMoeConfig.make(p, bucket, rank=0, world_size=entry["world_size"])
    with pytest.raises(C.KnobLookupError):
        C.resolve_knobs(
            C.SHIPPED_KNOBS,
            problem=problem,
            variant="inference",
            world_size=32,
            max_tokens=6144,
        )
    with pytest.raises(ValueError):  # genphase N64 + multirank phase_interleave
        C.RubinMoeConfig.make(
            mx,
            C.RubinBucket(
                "genphase",
                128,
                C.RubinKnobs.from_dict(
                    dict(
                        mma_tiler_mnk=(256, 64, 128),
                        cluster_shape_mn=(4, 1),
                        fallback_cluster_shape_mn=None,
                        schedule_policy=("phase_interleave", None),
                        work_id_mode="atomic_counter",
                        fc2_use_bulk=True,
                        fc2_tma_stages=1,
                        epi_flag_batches=(1, 4),
                        token_in_flag_batch=1,
                        token_back_mode="epi_warps",
                        reduce_topk_in_kernel=False,
                    )
                ),
            ),
            rank=0,
            world_size=16,
        )


@pytest.mark.parametrize("quant_kind,bound", [("mxfp4_mxfp8", 0.08), ("nvfp4", 0.25)])
def test_oracle_reads_kernel_layout(quant_kind, bound):
    """The quantization-faithful oracle (kernel-layout weights, staged bytes) agrees
    with the full-precision reference (canonical weights, BF16 tokens) up to activation
    quantization, independent of the power-of-two norm constants; a layout mistake
    (double gate/up interleave) is caught."""
    R = _load("reference")
    inter, topk, n = 128, 2, 32
    gen = torch.Generator().manual_seed(0)
    q13, s13 = W.quantize_mxfp4_block32(
        torch.randn(E, 2 * inter, H, generator=gen) * H**-0.5
    )
    q2, s2 = W.quantize_mxfp4_block32(
        torch.randn(E, H, inter, generator=gen) * inter**-0.5
    )
    hid = torch.randn(n, H, generator=gen).bfloat16()
    ids = torch.stack([torch.randperm(E, generator=gen)[:topk] for _ in range(n)])
    ids = ids.to(torch.int32)
    ids[0, 1] = -1  # a masked route
    wts = torch.rand(n, topk, generator=gen)
    situ = dict(situ_beta=4.0, situ_linear_beta=25.0)
    ref = R.bf16_partial(
        hid,
        ids,
        wts,
        (q13, s13, q2, s2),
        local_expert_offset=0,
        intermediate=inter,
        **situ,
    )
    norm_in = 2.0 if quant_kind == "nvfp4" else 1.0
    norm_1 = 4.0 if quant_kind == "nvfp4" else 1.0

    def rel(w):
        vec = 16 if quant_kind == "nvfp4" else 32
        x, xsf, idx, sw = R.stage_inputs_torch(
            hid,
            ids,
            wts,
            quant_kind=quant_kind,
            capacity=40,
            sf_cols=H // vec,
            num_experts=E,
            input_norm_const=norm_in,
        )
        assert bool((idx[n:] == -1).all()) and int(idx[0, 1]) == -1 and sw[0, 1] == 0
        out = R.megamoe_oracle(
            x[:n],
            xsf[:n],
            idx[:n],
            sw[:n],
            w,
            quant_kind=quant_kind,
            local_expert_offset=0,
            hidden=H,
            intermediate=inter,
            **situ,
        )
        return float((out - ref).norm() / ref.norm())

    good = W.prepare_weights(
        quant_kind,
        q13,
        s13,
        q2,
        s2,
        hidden=H,
        intermediate=inter,
        input_norm_const=norm_in,
        fc1_norm_const=norm_1,
    )
    assert rel(good) < bound
    broken = W.prepare_weights(
        quant_kind,
        W.interleave_gate_up_16(q13, intermediate_size=inter),
        W.interleave_gate_up_16(s13, intermediate_size=inter),
        q2,
        s2,
        hidden=H,
        intermediate=inter,
        input_norm_const=norm_in,
        fc1_norm_const=norm_1,
    )
    assert rel(broken) > 1.0


def _knob_file(*records) -> bytes:
    import json

    return json.dumps({"version": 2, "entries": list(records)}).encode()


def test_knob_merge_counts_every_source_file():
    import json

    C = _load("config")
    with open(C.SHIPPED_KNOBS) as f:
        shipped = json.load(f)
    entries = shipped["entries"]
    files = {e["provenance"]["file"] for e in entries}
    # every $W/moe4/results/fi107/knobs_*.json (one record each), incl. the EP4 smoke
    # files, the EP16 inference 4096 crossover files and the wide EP16 search (w16c)
    assert shipped["source_files"] == len(files) == len(entries) >= 44
    names = {f.rsplit("/", 1)[-1] for f in files}
    for must in (
        "knobs_smoke_nvfp4.json",
        "knobs_smoke_mxfp4_mxfp8.json",
        "knobs_xo_ep16_nvfp4_i4096.json",
        "knobs_xo_ep16_mxfp4_mxfp8_i4096.json",
    ):
        assert must in names, must
    for e in entries:
        assert re.fullmatch(r"[0-9a-f]{64}", e["provenance"]["sha256"])
        assert e["profiling_policy"]["timer"] == "native_events"
    # EP4 (1 node) resolves from the smoke records
    knobs, record, _ = C.resolve_knobs(
        C.SHIPPED_KNOBS,
        problem=C.RubinMoeProblem(quant_kind="nvfp4"),
        variant="inference",
        world_size=4,
        max_tokens=2048,
        live_tokens=2048,
    )
    assert record["world_size"] == 4 and record["max_tokens"] == 2048

    # merge: provenance + sha256 per record, count check, no silent drop
    rec = dict(entries[0])
    rec.pop("provenance")
    a = _knob_file(rec, rec)
    b = _knob_file(rec)
    merged = C.merge_knob_files([("a.json", a), ("b.json", b)], comment="t")
    assert merged["source_files"] == 2 and len(merged["entries"]) == 3
    assert [e["provenance"]["file"] for e in merged["entries"]] == [
        "a.json",
        "a.json",
        "b.json",
    ]
    assert merged["entries"][2]["provenance"]["sha256"] == hashlib.sha256(b).hexdigest()
    with pytest.raises(ValueError):  # a source file without records
        C.merge_knob_files([("a.json", a), ("e.json", _knob_file())], comment="t")
    with pytest.raises(ValueError):  # duplicate names
        C.merge_knob_files([("a.json", a), ("a.json", b)], comment="t")
    with pytest.raises(ValueError):  # wrong version
        C.merge_knob_files(
            [("v1.json", json.dumps({"version": 1, "entries": [rec]}).encode())],
            comment="t",
        )


def test_ep16_prefill_records_tile_n256():
    """The EP16 NVFP4 prefill buckets (C512 prefill: 8192 per rank) resolve to the
    tile N=256 records tuned for the S2 kernel at real EP16; the decode-sized EP16
    bucket keeps its tuner record."""
    C = _load("config")
    problem = C.RubinMoeProblem(quant_kind="nvfp4")

    def resolve(max_tokens):
        return C.resolve_knobs(
            C.SHIPPED_KNOBS,
            problem=problem,
            variant="inference",
            world_size=16,
            max_tokens=max_tokens,
        )

    for max_tokens, cap in ((4350, 8192), (8192, 8192), (2880, 4096), (4096, 4096)):
        knobs, record, _ = resolve(max_tokens)
        assert record["max_tokens"] == cap, max_tokens
        assert record["source"] == "bench-eager-native-events", max_tokens
        assert tuple(knobs.mma_tiler_mnk) == (256, 256, 256), max_tokens
    knobs, record, _ = resolve(2048)
    assert record["max_tokens"] == 2048
    assert tuple(knobs.mma_tiler_mnk) == (256, 128, 256)


def test_bench_capacity_prefers_tuned_record():
    C = _load("config")

    def cap(quant, world, n, variant="inference"):
        return C.bench_capacity(
            C.SHIPPED_KNOBS,
            problem=C.RubinMoeProblem(quant_kind=quant),
            variant=variant,
            world_size=world,
            live_tokens=n,
        )

    assert cap("nvfp4", 8, 5760) == 5760  # rounding would give 5888
    assert cap("mxfp4_mxfp8", 8, 5760) == 5760
    assert cap("nvfp4", 16, 5925) == 6144
    assert cap("nvfp4", 4, 2048) == 2048
    assert cap("nvfp4", 8, 720, "genphase") == 768
    assert cap("nvfp4", 16, 5000) == 5120  # no record at 5000: rounding
    assert C.default_capacity(5760) == 5888 and C.default_capacity(100) == 128
    assert (
        C.tuned_capacity(
            '{"mma_tiler_mnk": [256, 128, 256]}',
            problem=C.RubinMoeProblem(),
            variant="inference",
            world_size=8,
            live_tokens=5760,
        )
        is None
    )


def test_parse_expert_key():
    p = W.parse_expert_key
    pre = "language_model.model.layers.5.block_sparse_moe.experts.7"
    assert p(f"{pre}.w1.weight_packed") == (5, 7, "w1", "weight")
    assert p(f"{pre}.w3.weight_scale") == (5, 7, "w3", "weight_scale")
    assert p(f"{pre}.w2.weight") == (5, 7, "w2", "weight")
    assert p("model.layers.12.mlp.experts.895.w2.weight_packed") == (
        12,
        895,
        "w2",
        "weight",
    )
    for key in (
        "language_model.model.layers.5.block_sparse_moe.shared_experts.w1.weight",
        f"{pre}.w1.weight_global_scale",
        f"{pre}.w4.weight_packed",
        "language_model.model.layers.5.block_sparse_moe.gate.weight",
        f"{pre}.w1.weight_packed.extra",
    ):
        assert p(key) is None, key


def test_cache_keys_and_layout_tag():
    from types import SimpleNamespace

    S = _load("session")
    cfg = SimpleNamespace(
        quant_kind="nvfp4",
        situ_beta=4.0,
        situ_linear_beta=25.0,
        kernel_variant="genphase",
        max_tokens_per_rank=128,
    )
    base = S.compile_cache_key(cfg, "kname", "4.8.0", 0)
    assert base == ("kname", "4.8.0", "nvfp4", 4.0, 25.0, 0, None)
    for change in (
        dict(die_sm_mask32=(1, 2)),
        dict(situ_beta=2.0),
        dict(situ_linear_beta=10.0),
        dict(quant_kind="mxfp4_mxfp8"),
    ):
        other = SimpleNamespace(**{**vars(cfg), **change})
        assert S.compile_cache_key(other, "kname", "4.8.0", 0) != base
    assert S.compile_cache_key(cfg, "kname", "4.8.0", 1) != base
    tag = S.layout_tag(0, cfg, "RubinGenphase_t256x128_c4x1")
    assert tag.endswith(":RubinGenphase_t256x128_c4x1") and "4.0/25.0" in tag

    out = torch.empty(8, 16, dtype=torch.bfloat16)
    key = S.output_binding_key(out)
    assert S.output_binding_key(out.view(torch.float16)) != key  # dtype
    strided = torch.empty(16, 8, dtype=torch.bfloat16).t()  # same shape, other stride
    assert S.output_binding_key(strided)[1] == key[1]
    assert S.output_binding_key(strided)[2] != key[2]


def test_s2_epilogue_dispatch_policy(monkeypatch):
    """S2 only on capped (shared-overlap) grids, per launch up to the crossover; both
    variants of one session must share the workspace layout; the EP-mean token count
    follows sequence_parallel_chunk's ceil(n / sp) shards."""
    from types import SimpleNamespace

    S = _load("session")
    x = S.S2_MAX_TOKENS_PER_RANK

    class _Ws:
        def __init__(self, extra=0):
            self.extra = extra

        def total_bytes(self, space):
            return 1024 + self.extra

        def regions(self, space):
            return (SimpleNamespace(name="r", reset="data", dtype="i32"),)

        def offset(self, name):
            return 0

        def nbytes(self, name):
            return 64 + self.extra

    built = []

    def fake_build(cfg, epilogue_warpgroups=None, *, eligible=True, ws_extra=0):
        wg = epilogue_warpgroups or (2 if eligible else 1)
        built.append(epilogue_warpgroups)
        return SimpleNamespace(
            epilogue_warpgroups=wg,
            _device_workspace=_Ws(ws_extra if wg == 1 else 0),
        )

    def cfg(cap, sm=148, variant="inference"):
        return SimpleNamespace(
            kernel_variant=variant, max_sm_count=sm, max_tokens_per_rank=cap
        )

    monkeypatch.setattr(S, "build_kernel", fake_build)
    # uncapped: stock epilogue only, requested explicitly
    built.clear()
    ks = S.build_kernels(cfg(8192, sm=None))
    assert set(ks) == {False} and built == [1]
    # genphase: the kernel's own single epilogue
    assert set(S.build_kernels(cfg(1024, variant="genphase"))) == {False}
    # capped, capacity above the crossover: both variants
    built.clear()
    ks = S.build_kernels(cfg(8192))
    assert set(ks) == {True, False} and built == [None, 1]
    assert ks[True].epilogue_warpgroups == 2 and ks[False].epilogue_warpgroups == 1
    # capped, capacity within the S2 range: S2 only
    assert set(S.build_kernels(cfg(x))) == {True}
    # capped but not S2-eligible (e.g. a 2048 decode bucket): stock only
    monkeypatch.setattr(
        S, "build_kernel", lambda c, w=None: fake_build(c, w, eligible=False)
    )
    assert set(S.build_kernels(cfg(2048))) == {False}
    # variants whose workspace layouts differ must not share a session
    monkeypatch.setattr(
        S, "build_kernel", lambda c, w=None: fake_build(c, w, ws_extra=128)
    )
    with pytest.raises(RuntimeError, match="workspace layout"):
        S.build_kernels(cfg(8192))

    def use_s2(kernels, cap, tokens):
        sess = SimpleNamespace(kernels=kernels, cfg=cfg(cap))
        return S.RubinMegaMoeSession.use_s2(sess, tokens)

    both = {True: object(), False: object()}
    assert use_s2(both, 8192, 0) and use_s2(both, 8192, x)
    assert not use_s2(both, 8192, x + 1) and not use_s2(both, 8192, 8192)
    assert not use_s2(both, 8192, None)  # None = the capacity
    assert use_s2({True: object()}, x, None)
    assert not use_s2({False: object()}, 8192, 0)

    t = torch.tensor
    assert S.ep_mean_tokens(t([17400, 17400, 17400, 17400]), 4) == 4350
    assert S.ep_mean_tokens(t([32766] * 4), 4) == 8192  # ceil(32766 / 4) per shard
    assert S.ep_mean_tokens(t([1, 0, 0, 0]), 4) == 1  # rounded up
    assert S.ep_mean_tokens(t([100, 300]), 1) == 200
    with pytest.raises(ValueError):
        S.ep_mean_tokens(t([1]), 0)


def test_out_of_range_expert_ids_rejected():
    C = _load("config")
    R = _load("reference")
    n, topk, experts = 4, 2, 8
    hid = torch.randn(n, H).bfloat16()
    wts = torch.rand(n, topk)

    def stage(ids):
        return R.stage_inputs_torch(
            hid,
            ids,
            wts,
            quant_kind="mxfp4_mxfp8",
            capacity=n,
            sf_cols=H // 32,
            num_experts=experts,
        )

    ok = torch.tensor([[0, 7], [-1, 3], [-1, -1], [5, 6]], dtype=torch.int32)
    _, _, idx, sw = stage(ok)
    assert torch.equal(idx, ok) and float(sw[2].abs().sum()) == 0.0
    for bad_id in (experts, experts + 100, -2):
        for dtype in (torch.int32, torch.int64):
            bad = ok.clone().to(dtype)
            bad[3, 1] = bad_id
            with pytest.raises(C.RouteIdError) as info:
                stage(bad)
            assert info.value.expert_id == bad_id
            assert info.value.num_experts == experts


def test_nvfp4_norm_const_from_amax():
    C = _load("config")
    f = C.nvfp4_norm_const_from_amax
    assert f(6.0 * 448.0) == 1.0
    assert f(1.0) == 2.0**11  # 2048 <= 2688 < 4096
    assert f(1.0, headroom=2.0) == 2.0**10
    assert f(6.0 * 448.0 * 8) == 2.0**-3
    for amax in (1e-3, 0.0625, 0.37, 5.1, 300.0, 2.0**-20):
        n = f(amax)
        assert W.is_power_of_two(n)
        assert amax * n <= 6.0 * 448.0 < amax * n * 2
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            f(bad)
    with pytest.raises(ValueError):
        f(1.0, headroom=0.5)
    # derived constants are accepted by the problem contract
    C.RubinMoeProblem(input_norm_const=f(4.0), fc1_norm_const=f(0.02))


def test_vendored_sources_match_manifest():
    text = (PKG_DIR / "VENDOR.md").read_text()
    manifest = dict(
        (path, digest)
        for digest, path in re.findall(r"^([0-9a-f]{64})  (sources/\S+)$", text, re.M)
    )
    src = PKG_DIR / "src"
    files = {
        str(p.relative_to(src)): p
        for p in src.joinpath("sources").rglob("*.py")
        if "__pycache__" not in p.parts
    }
    assert len(manifest) == 42 and set(files) == set(manifest)
    for rel, path in files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[rel], rel
    # no FlashInfer / NVSHMEM import anywhere in the backend
    pattern = re.compile(r"^\s*(from|import)\s+(flashinfer|nvshmem)", re.M)
    for path in PKG_DIR.rglob("*.py"):
        assert not pattern.search(path.read_text()), path


@pytest.mark.parametrize(
    ("world", "max_tokens", "tiler", "cluster", "fallback"),
    [
        (8, 2048, (256, 256, 256), (2, 1), None),  # C256 prefill token shard, EP8
        (16, 8192, (256, 256, 256), (4, 1), (2, 1)),  # C512 prefill SP shard, EP16 (S2)
        (4, 8192, (256, 256, 256), (2, 1), None),  # 1-node TP4 prefill bench, EP4
    ],
)
def test_prefill_records_phase_interleaved(world, max_tokens, tiler, cluster, fallback):
    """The NVFP4 prefill buckets resolve the phase-interleaved N256 records: the grouped
    grid-stride schedule put each group's FC2 tiles right behind its FC1 tiles, so
    their TMA-B warp spun on FC1-done counters (590 -> 375 us at the C256 per-rank
    shape, bitwise-equal outputs; pf-c256-moe)."""
    C = _load("config")
    knobs, record, _ = C.resolve_knobs(
        C.SHIPPED_KNOBS,
        problem=C.RubinMoeProblem(quant_kind="nvfp4"),
        variant="inference",
        world_size=world,
        max_tokens=max_tokens,
    )
    assert record["max_tokens"] == max_tokens
    assert knobs.schedule_policy == ("phase_interleave", None)
    assert knobs.work_id_mode == "atomic_counter"
    assert tuple(knobs.mma_tiler_mnk) == tiler
    assert tuple(knobs.cluster_shape_mn) == cluster
    fb = knobs.fallback_cluster_shape_mn
    assert (None if fb is None else tuple(fb)) == fallback
