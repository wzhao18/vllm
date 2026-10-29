# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests: CuTe sparse-decode scheduler <-> pinned FlashMLA.

``vllm/v1/attention/ops/flashmla_sched_cutedsl.py`` writes FlashMLA's private
split-KV schedule (``DecodingSchedMeta`` rows + ``num_splits``) and hands it
to ``flash_mla_with_kvcache`` pre-initialized, so FlashMLA skips its own
scheduler. That couples us to FlashMLA internals (vllm-project/FlashMLA @
5dceadbd, cmake/external_projects/flashmla.cmake):

  * flash_mla_interface.py: ``FlashMLASchedMeta`` / ``Config`` fields and the
    "already initialized -> use the tensors as given" behaviour;
  * csrc/params.h: ``DecodingSchedMeta`` = 8 int32 {begin_req, end_req,
    begin_block, end_block (excl), begin_split, is_first_req_splitted,
    is_last_req_splitted, pad};
  * csrc/api/sparse_decode.cpp: ``num_sm_parts`` per impl (``get_meta``), the
    shape checks, and the split-KV threshold (``topk > 640``);
  * the legacy SM100 head64 kernel / combine: how a part's rows are read
    (64-token blocks, ``max(1, ceil(len / 64))`` blocks per request, split
    flags, ``num_splits[r] + begin_split`` accumulator slots).

Every test here fails if one of those changes. When one does, re-read the
FlashMLA sources at the new pin and update the CuTe scheduler (or drop it and
fall back to ``get_mla_metadata()``) before bumping the pin.
"""

import dataclasses
import math

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(100):
    pytest.skip("FlashMLA sparse fp8 decode: SM100 only", allow_module_level=True)

from vllm.v1.attention.ops.flashmla import (  # noqa: E402
    FlashMLASchedMeta,
    flash_mla_with_kvcache,
    get_mla_metadata,
    is_flashmla_sparse_supported,
)

if not is_flashmla_sparse_supported()[0]:
    pytest.skip(is_flashmla_sparse_supported()[1], allow_module_level=True)

from vllm.v1.attention.ops.flashmla_sched_cutedsl import (  # noqa: E402
    FLASHMLA_SPARSE_SCHED_KERNEL,
    SCHED_INTS,
    build_sparse_sched_meta,
    flashmla_num_sm_parts,
)

PAGE = 64  # fp8_ds_mla page block size
BLOCK_TOPK = 64  # FlashMLA block_size_topk (get_meta)
OVERHEAD = 5  # FlashMLA fixed_overhead_num_blocks (get_meta)
SPLIT_KV_MAX_UNSPLIT_TOPK = 640  # sparse_decode.cpp: split-KV iff topk > 640
D_QK, D_V, BYTES_PER_TOKEN = 576, 512, 656
NUM_PAGES = 1024  # 65536 tokens; the last page is poison (see _make_cache)
SM_SCALE = 576**-0.5

DEV = torch.device("cuda")


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def _make_cache(seed: int) -> torch.Tensor:
    """fp8_ds_mla cache [pages, 64, 1, 656]: 512 e4m3 NoPE, 4 fp32 pow2 scales,
    64 bf16 RoPE. The last page is "poison" (huge scale): any token read past
    topk_length lands there and blows up the LSE."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    tok = NUM_PAGES * PAGE
    nope = (torch.randn(tok, 512, device=DEV, generator=g) * 2).to(torch.float8_e4m3fn)
    exps = torch.randint(-6, -1, (tok, 4), device=DEV, generator=g)
    scales = torch.pow(2.0, exps.float())
    scales[-PAGE:] = 2.0**6
    rope = (torch.randn(tok, 64, device=DEV, generator=g) * 0.5).bfloat16()
    cache = torch.empty(tok, BYTES_PER_TOKEN, dtype=torch.uint8, device=DEV)
    cache[:, :512] = nope.view(torch.uint8)
    cache[:, 512:528] = scales.contiguous().view(torch.uint8)
    cache[:, 528:] = rope.view(torch.uint8)
    return cache.view(NUM_PAGES, PAGE, 1, BYTES_PER_TOKEN)


def _make_nvfp4_cache(seed: int) -> torch.Tensor:
    """nvfp4_ds_mla cache [pages, 64, 1, 352]: 512 e2m1 NoPE (any byte is two
    valid values), 64 e4m3 RoPE, 32 e4m3 NoPE scales (uniform, so their byte
    permutation does not matter). FlashMLA infers the format from the bytes
    per token. The last page is poison, as above."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    tok = NUM_PAGES * PAGE
    cache = torch.randint(
        0, 256, (tok, 352), dtype=torch.uint8, device=DEV, generator=g
    )
    rope = torch.randn(tok, 64, device=DEV, generator=g) * 0.5
    cache[:, 256:320] = rope.to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.full((tok, 32), 2.0**-3, device=DEV)
    scales[-PAGE:] = 2.0**8
    cache[:, 320:] = scales.to(torch.float8_e4m3fn).view(torch.uint8)
    return cache.view(NUM_PAGES, PAGE, 1, 352)


def _dequant(cache: torch.Tensor) -> torch.Tensor:
    """[tokens, 576] fp32 of a cache from _make_cache or _make_nvfp4_cache."""
    flat = cache.reshape(-1, cache.shape[-1])
    if flat.shape[1] == BYTES_PER_TOKEN:
        nope = flat[:, :512].view(torch.float8_e4m3fn).float().view(-1, 4, 128)
        scale = flat[:, 512:528].contiguous().view(torch.float32)
        rope = flat[:, 528:].contiguous().view(torch.bfloat16).float()
        return torch.cat([(nope * scale[..., None]).view(-1, 512), rope], dim=-1)
    e2m1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=DEV)
    codes = torch.stack([flat[:, :256] & 15, flat[:, :256] >> 4], -1).flatten(1)
    nope = e2m1[(codes & 7).long()] * torch.where(codes >= 8, -1.0, 1.0)
    scale = flat[:, 320:321].view(torch.float8_e4m3fn).float()
    rope = flat[:, 256:320].view(torch.float8_e4m3fn).float()
    return torch.cat([nope * scale, rope], dim=-1)


def _lengths(b: int, width: int, dist: str, g: torch.Generator) -> torch.Tensor:
    if dist == "full":
        lens = torch.full((b,), width, device=DEV)
    elif dist == "rand":
        lens = torch.randint(0, width + 1, (b,), device=DEV, generator=g)
    elif dist == "zeros":  # many rows with no local candidate (DCP)
        lens = torch.randint(0, 3, (b,), device=DEV, generator=g) * (width // 3)
    elif dist == "all_zero":
        lens = torch.zeros((b,), device=DEV)
    elif dist == "skew":  # DCP8 compacted: ~width/8, a few full rows
        lens = torch.randint(width // 10, width // 6 + 1, (b,), device=DEV, generator=g)
        lens[:: max(1, b // 7)] = width
    elif dist == "small":  # all inside one or two 64-token blocks
        lens = torch.randint(0, 2 * BLOCK_TOPK + 2, (b,), device=DEV, generator=g)
    elif dist == "edges":  # block-boundary lengths
        edges = torch.tensor(
            [0, 1, 63, 64, 65, 127, 128, 129, width - 1, width], device=DEV
        ).clamp(max=width)
        lens = edges[torch.randint(0, len(edges), (b,), device=DEV, generator=g)]
    else:
        raise ValueError(dist)
    return lens.to(torch.int32).clamp(max=width)


def _make_inputs(b, s_q, h, width, dist, seed):
    """Build q [b, s_q, h, 576] bf16, indices [b, s_q, width], lengths [b].

    Positions < len hold real tokens (with a few -1, as group-padded rows in
    the DCP call site have); positions >= len hold poison tokens, which
    FlashMLA must ignore (it masks by topk_length)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    lens = _lengths(b, width, dist, g)
    num_real = (NUM_PAGES - 1) * PAGE
    idx = torch.randint(0, num_real, (b, s_q, width), device=DEV, generator=g)
    if s_q > 1:
        holes = torch.rand((b, s_q, width), device=DEV, generator=g) < 0.03
        idx = torch.where(holes, -1, idx)
    poison = num_real + torch.randint(0, PAGE, (b, s_q, width), device=DEV, generator=g)
    col = torch.arange(width, device=DEV)
    idx = torch.where(col < lens[:, None, None], idx, poison).to(torch.int32)
    q = (torch.randn((b, s_q, h, D_QK), device=DEV, generator=g) * 0.5).bfloat16()
    return q, idx, lens


def _run(q, cache, idx, lens, sched):
    return flash_mla_with_kvcache(
        q=q,
        k_cache=cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=D_V,
        tile_scheduler_metadata=sched,
        is_fp8_kvcache=True,
        indices=idx,
        softmax_scale=SM_SCALE,
        topk_length=lens,
    )


# --------------------------------------------------------------------------
# FlashMLA semantics, written out
# --------------------------------------------------------------------------


def _nblocks(length: int) -> int:
    return (max(length, 1) - 1) // BLOCK_TOPK + 1


def _flashmla_serial_schedule(lens: list[int], parts: int):
    """Python transcription of get_decoding_sched_meta.cu @5dceadbd (columns
    0..6; the pad column is left uninitialized by FlashMLA)."""
    b = len(lens)
    nb = [_nblocks(x) for x in lens]
    total = sum(n + OVERHEAD for n in nb)
    payload = -(-total // parts) + OVERHEAD
    meta, num_splits = [], [0] * (b + 1)
    req = blk = nsplit = cum = 0
    for _ in range(parts):
        m = [req, 0, blk, 0, nsplit, int(blk != 0), 0]
        remain = payload
        while req < b:
            rem_blocks = nb[req] - blk
            if remain >= rem_blocks + OVERHEAD:
                cum += nsplit + 1
                num_splits[req + 1] = cum
                remain -= rem_blocks + OVERHEAD
                req, blk, nsplit = req + 1, 0, 0
            else:
                if remain - OVERHEAD > 0:
                    blk += remain - OVERHEAD
                    nsplit += 1
                break
        m[1] = req if blk > 0 else req - 1
        m[3] = blk if blk > 0 else nb[req - 1]
        m[6] = int(m[3] != nb[m[1]])
        if m[0] == m[1]:
            m[5] = m[6] = int(m[5] or m[6])
        meta.append(m)
    assert req == b and blk == 0 and nsplit == 0
    return meta, num_splits


def _check_schedule(meta: torch.Tensor, num_splits: torch.Tensor, lens, parts):
    """Assert a schedule is valid for FlashMLA's sparse decode kernel + combine
    (legacy/sm100/decode/sparse/head64/kernel.cuh, smxx/decode/combine)."""
    lens = [int(x) for x in lens.tolist()]
    b = len(lens)
    nb = [_nblocks(x) for x in lens]
    assert meta.dtype == torch.int32 and num_splits.dtype == torch.int32
    assert meta.is_contiguous() and num_splits.is_contiguous()
    assert tuple(meta.shape) == (parts, SCHED_INTS)
    assert tuple(num_splits.shape) == (b + 1,)
    M = meta.tolist()
    S = num_splits.tolist()
    assert S[0] == 0
    ns = [S[r + 1] - S[r] for r in range(b)]
    assert all(n >= 1 for n in ns), "every request owns >= 1 split slot"
    assert max(ns) <= parts, "combine's MAX_SPLITS bucket is >= num_sm_parts"
    assert S[b] <= b + parts, "o_accum/lse_accum hold b + num_sm_parts splits"
    pieces: list[list[tuple[int, int, int, int]]] = [[] for _ in range(b)]
    for p, (br, er, bb, eb, bs, fs, ls, _pad) in enumerate(M):
        if br >= b or br > er:  # empty part (kernel returns / loop is empty)
            continue
        assert 0 <= br <= er < b, (p, M[p])
        for r in range(br, er + 1):
            s = bb if r == br else 0
            e = eb if r == er else nb[r]
            split = fs if r == br else (ls if r == er else 0)
            slot = bs if r == br else 0
            assert 0 <= s < e <= nb[r], f"part {p} req {r}: blocks [{s},{e})"
            pieces[r].append((s, e, split, slot))
    for r in range(b):
        ps = sorted(pieces[r])
        assert ps, f"req {r} not scheduled"
        cover = 0
        for s, e, _, _ in ps:
            assert s == cover, f"req {r}: gap/overlap at block {s} ({ps})"
            cover = e
        assert cover == nb[r], f"req {r}: covers {cover}/{nb[r]} blocks"
        assert len(ps) == ns[r], f"req {r}: {len(ps)} pieces, num_splits {ns[r]}"
        want_split = int(ns[r] > 1)
        assert all(sp == want_split for _, _, sp, _ in ps), (r, ps)
        if want_split:
            assert sorted(sl for *_, sl in ps) == list(range(ns[r])), (r, ps)


def _flashmla_own_meta(q, cache, idx, lens):
    sched = get_mla_metadata()[0]
    out = _run(q, cache, idx, lens, sched)
    return sched, out


# --------------------------------------------------------------------------
# (1) Metadata format / FlashMLA's own scheduler unchanged
# --------------------------------------------------------------------------


def test_sched_meta_config_fields():
    """build_sparse_sched_meta builds Config positionally."""
    names = [f.name for f in dataclasses.fields(FlashMLASchedMeta.Config)]
    assert names == [
        "b",
        "s_q",
        "h_q",
        "page_block_size",
        "h_k",
        "causal",
        "is_fp8_kvcache",
        "topk",
        "extra_page_block_size",
        "extra_topk",
        "enable_batch_invariant",
    ]
    fields = {f.name for f in dataclasses.fields(FlashMLASchedMeta)}
    assert {"have_initialized", "config", "tile_scheduler_metadata"} <= fields
    assert "num_splits" in fields


@pytest.mark.parametrize("h", [64, 128])
@pytest.mark.parametrize("s_q", [1, 5])
@pytest.mark.parametrize(
    "b,width,dist",
    [(1, 2048, "full"), (37, 2048, "rand"), (300, 1024, "edges"), (4096, 2048, "skew")],
)
def test_metadata_format_matches_flashmla(b, s_q, h, width, dist):
    """Shapes/dtypes/num_sm_parts/Config of our metadata equal FlashMLA's own
    for the same call; FlashMLA's own schedule is bit-exact with the
    transcribed serial algorithm and passes the same validity checker."""
    if b * s_q > 4096:
        b = 4096 // s_q
    cache = _make_cache(0)
    q, idx, lens = _make_inputs(b, s_q, h, width, dist, seed=b + 31 * s_q + h)
    ref, _ = _flashmla_own_meta(q, cache, idx, lens)
    ours = build_sparse_sched_meta(lens, (b, s_q, h), width)

    assert dataclasses.asdict(ours.config) == dataclasses.asdict(ref.config)
    parts = flashmla_num_sm_parts(s_q)
    for sm in (ref, ours):
        assert sm.have_initialized
        assert sm.tile_scheduler_metadata.dtype == torch.int32
        assert sm.num_splits.dtype == torch.int32
        assert tuple(sm.tile_scheduler_metadata.shape) == (parts, SCHED_INTS)
        assert tuple(sm.num_splits.shape) == (b + 1,)
        assert sm.tile_scheduler_metadata.device == sm.num_splits.device == lens.device

    meta_ref, splits_ref = _flashmla_serial_schedule(lens.tolist(), parts)
    got = ref.tile_scheduler_metadata[:, :7].tolist()
    assert got == meta_ref, "FlashMLA's scheduler changed since 5dceadbd"
    assert ref.num_splits.tolist() == splits_ref
    _check_schedule(ref.tile_scheduler_metadata, ref.num_splits, lens, parts)
    _check_schedule(ours.tile_scheduler_metadata, ours.num_splits, lens, parts)


@pytest.mark.parametrize("s_q", [1, 5])
@pytest.mark.parametrize(
    "dist", ["full", "rand", "zeros", "all_zero", "skew", "small", "edges"]
)
@pytest.mark.parametrize("b", [1, 2, 7, 64, 147, 148, 149, 1023, 1025, 4096, 9000])
def test_cute_schedule_is_valid(b, s_q, dist):
    """Pure metadata check over many batch sizes (incl. > one CTA stride and
    around num_sms), no attention call."""
    g = torch.Generator(device=DEV).manual_seed(7 * b + s_q)
    lens = _lengths(b, 2048, dist, g)
    parts = flashmla_num_sm_parts(s_q)
    sm = build_sparse_sched_meta(lens, (b, s_q, 64), 2048)
    _check_schedule(sm.tile_scheduler_metadata, sm.num_splits, lens, parts)


def test_warmup_compiles_the_dcp_key():
    """JIT warmup compiles the executor the DCP calls (s_q = 1) use."""
    lens = torch.zeros(4, dtype=torch.int32, device=DEV)
    sm = build_sparse_sched_meta(lens, (4, 1, 128), 2048)
    (key,) = FLASHMLA_SPARSE_SCHED_KERNEL.get_warmup_keys()
    assert key.parts == sm.tile_scheduler_metadata.shape[0]


def test_split_kv_threshold():
    """Split-KV (and with it the metadata) is only used for index width > 640;
    narrower calls ignore the metadata (FlashMLA returns none)."""
    cache = _make_cache(0)
    for width, used in [
        (SPLIT_KV_MAX_UNSPLIT_TOPK, False),
        (SPLIT_KV_MAX_UNSPLIT_TOPK + 64, True),
    ]:
        q, idx, lens = _make_inputs(16, 1, 64, width, "rand", seed=width)
        sched, _ = _flashmla_own_meta(q, cache, idx, lens)
        assert (sched.tile_scheduler_metadata is not None) == used, width


# --------------------------------------------------------------------------
# (2) End-to-end: FlashMLA with our metadata == FlashMLA with its own
# --------------------------------------------------------------------------


def _live(idx, lens):
    """[b, 1, s_q]: query rows with >= 1 attended token (lse layout)."""
    col = torch.arange(idx.shape[-1], device=DEV)
    ok = (idx >= 0) & (col < lens[:, None, None])
    return ok.any(dim=-1)[:, None, :]


def _assert_same_attention(o_new, l_new, o_ref, l_ref, live):
    assert not torch.isnan(o_new).any() and not torch.isnan(l_new).any()
    assert not torch.isnan(o_ref).any() and not torch.isnan(l_ref).any()
    assert torch.isfinite(o_new).all()
    # Only fp32 accumulation order / bf16 rounding of split partials differ
    # (GB300: max |dout| ~1e-3 = half a bf16 ulp, max |dlse| ~1.5e-6). Rows
    # with topk_length 0 are never split: identical (0, +inf) in both.
    torch.testing.assert_close(o_new.float(), o_ref.float(), rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(l_new, l_ref, rtol=1e-5, atol=2e-5)
    assert torch.isfinite(torch.where(live, l_new, 0)).all()
    # Nothing past topk_length was read (a poison token pushes the LSE >> 30).
    assert (torch.where(live, l_new, 0) < 30).all()


EQUIV_CASES = [
    # (b, s_q, h, width, dist)
    (1, 1, 64, 2048, "full"),
    (7, 1, 128, 2048, "rand"),
    (64, 5, 128, 2048, "skew"),
    (148, 1, 128, 2048, "rand"),
    (149, 1, 64, 2048, "small"),
    (320, 1, 128, 2048, "zeros"),  # DCP8 decode, B64 x (1 + 4 MTP)
    (1000, 1, 64, 2048, "edges"),
    (2048, 1, 128, 704, "edges"),
    (4096, 1, 128, 2048, "skew"),  # DCP8 16K prefill chunk
    (4096, 1, 64, 1024, "full"),
    (4096, 1, 128, 2048, "all_zero"),
    (256, 1, 64, 640, "rand"),  # no split-KV: metadata unused
]


@pytest.mark.parametrize(
    "b,s_q,h,width,dist",
    [pytest.param(*c, id="b{}-q{}-h{}-w{}-{}".format(*c)) for c in EQUIV_CASES],
)
def test_attention_matches_flashmla_scheduler(b, s_q, h, width, dist):
    cache = _make_cache(1)
    q, idx, lens = _make_inputs(b, s_q, h, width, dist, seed=b * 131 + s_q * 7 + h)
    o_ref, l_ref = _run(q, cache, idx, lens, get_mla_metadata()[0])
    ours = build_sparse_sched_meta(lens, (b, s_q, h), width)
    meta, splits = ours.tile_scheduler_metadata, ours.num_splits
    meta_before, splits_before = meta.clone(), splits.clone()
    o_new, l_new = _run(q, cache, idx, lens, ours)
    # FlashMLA consumed our tensors as given (did not rebuild or replace them).
    assert ours.tile_scheduler_metadata is meta and ours.num_splits is splits
    assert torch.equal(meta, meta_before) and torch.equal(splits, splits_before)
    _assert_same_attention(o_new, l_new, o_ref, l_ref, _live(idx, lens))


@pytest.mark.parametrize("seed", range(8))
def test_attention_matches_flashmla_scheduler_fuzz(seed):
    rng = torch.Generator().manual_seed(1000 + seed)

    def pick(xs):
        return xs[int(torch.randint(0, len(xs), (1,), generator=rng))]

    s_q = pick([1, 1, 5])
    b = int(torch.randint(1, 4096 // s_q + 1, (1,), generator=rng))
    h = pick([64, 128])
    width = pick([2048, 2048, 1536, 1024, 768])
    dist = pick(["rand", "zeros", "skew", "small", "edges", "full"])
    test_attention_matches_flashmla_scheduler(b, s_q, h, width, dist)


def _reference_attention(q, cache, idx, lens):
    """fp32 sparse MLA over the dequantized cache."""
    kv = _dequant(cache)
    b, s_q, h, _ = q.shape
    out = torch.zeros(b, s_q, h, D_V, device=DEV)
    lse = torch.full((b, h, s_q), float("inf"), device=DEV)
    for i in range(b):
        for j in range(s_q):
            sel = idx[i, j, : int(lens[i])]
            sel = sel[sel >= 0]
            if sel.numel() == 0:
                continue
            k = kv[sel.long()]
            s = q[i, j].float() @ k.T * SM_SCALE
            lse[i, :, j] = torch.logsumexp(s, dim=-1)
            out[i, j] = torch.softmax(s, dim=-1) @ k[:, :D_V]
    return out, lse


@pytest.mark.parametrize("make_cache", [_make_cache, _make_nvfp4_cache])
def test_dcp_compact_call_site_matches_flashmla_scheduler(monkeypatch, make_cache):
    """The production helper (chunking, head padding, empty rows) gives the
    same result with our scheduler as with FlashMLA's."""
    from vllm.v1.attention.backends.mla import flashmla_sparse
    from vllm.v1.attention.ops import flashmla_sched_cutedsl as sched_mod

    cache = make_cache(3)
    pages = cache.squeeze(2)
    heads = 96  # DCP8 at TP8: 96 gathered heads, padded to 128
    num_tokens = flashmla_sparse.fp8_mixed_batch_chunk_tokens(heads) + 517
    q4, idx, lens = _make_inputs(num_tokens, 1, heads, 2048, "skew", seed=5)
    lens[::11] = 0
    q, slots = q4[:, 0], idx[:, 0]
    slots = torch.where(torch.arange(2048, device=DEV) < lens[:, None], slots, -1)

    def call():
        return flashmla_sparse._dcp_compact_fp8_flash_mla(
            q, pages, slots, lens, SM_SCALE
        )

    o_new, l_new = call()
    monkeypatch.setattr(
        sched_mod, "build_sparse_sched_meta", lambda *a, **k: get_mla_metadata()[0]
    )
    o_ref, l_ref = call()
    assert torch.equal(l_new.isneginf(), lens[:, None].expand_as(l_new) == 0)
    assert (o_new[lens == 0] == 0).all()
    torch.testing.assert_close(o_new.float(), o_ref.float(), rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(l_new, l_ref, rtol=1e-5, atol=2e-5)


@pytest.mark.parametrize("make_cache", [_make_cache, _make_nvfp4_cache])
@pytest.mark.parametrize("width", [2048, SPLIT_KV_MAX_UNSPLIT_TOPK])
@pytest.mark.parametrize("h", [64, 128])
@pytest.mark.parametrize("b", [320, 4096])
def test_empty_rows_write_zero_out_and_inf_lse(b, h, width, make_cache):
    """The DCP call site does not mask out: it relies on FlashMLA writing
    out = 0 (and lse = +inf) for a row with topk_length 0, also inside a busy
    batch with split rows, in the split-KV and unsplit kernels and for both
    quantized caches. A NaN-filled out buffer shows the rows are written,
    not left as they were. Rows with topk_length > 0 never give +inf."""
    cache = make_cache(5)
    q, idx, lens = _make_inputs(b, 1, h, width, "skew", seed=b + h)
    lens[::3] = 0
    lens[0] = lens[-1] = 0
    out = torch.full((b, 1, h, D_V), float("nan"), dtype=q.dtype, device=DEV)
    o, lse = flash_mla_with_kvcache(
        q=q,
        k_cache=cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=D_V,
        tile_scheduler_metadata=build_sparse_sched_meta(lens, (b, 1, h), width),
        is_fp8_kvcache=True,
        indices=idx,
        softmax_scale=SM_SCALE,
        topk_length=lens,
        out=out,
    )
    assert o.data_ptr() == out.data_ptr()
    empty = lens == 0
    assert (o[empty] == 0).all()
    assert torch.isposinf(lse[empty]).all()
    assert torch.isfinite(o[~empty]).all()
    assert torch.isfinite(lse[~empty]).all()


def test_dcp_compact_strided_out_and_empty_rows():
    """A single-chunk DCP call returns out as a view of FlashMLA's head-padded
    output (no slice copy) with the same values as the packed copy and as an
    fp32 reference, and count-0 rows as (0, -inf) without a pass over out.
    A call with no rows returns empty out and lse."""
    from vllm.v1.attention.backends.mla import flashmla_sparse

    cache = _make_cache(6)
    pages = cache.view(NUM_PAGES, PAGE, BYTES_PER_TOKEN)
    heads, num_tokens = 96, 320  # DCP8 decode, B64 x (1 + 4 MTP)
    q4, idx, lens = _make_inputs(num_tokens, 1, heads, 2048, "skew", seed=7)
    lens[::4] = 0
    q, slots = q4[:, 0], idx[:, 0]
    slots = torch.where(torch.arange(2048, device=DEV) < lens[:, None], slots, -1)
    # Leave NaN in recently freed memory, where FlashMLA's output may land.
    junk = torch.full((num_tokens, 128, D_V * 4), float("nan"), device=DEV)
    del junk
    o_view, l_view = flashmla_sparse._dcp_compact_fp8_flash_mla(
        q, pages, slots, lens, SM_SCALE
    )
    o_packed, l_packed = flashmla_sparse._dcp_compact_fp8_flash_mla(
        q, pages, slots, lens, SM_SCALE, packed_out=True
    )
    assert o_view.shape == o_packed.shape == (num_tokens, heads, D_V)
    assert o_view.stride() == (128 * D_V, D_V, 1)
    assert l_view.stride() == (128, 1)
    assert o_packed.is_contiguous()
    assert torch.equal(o_view, o_packed)
    assert torch.equal(l_view, l_packed)
    empty = lens == 0
    assert (o_view[empty] == 0).all()
    assert torch.isneginf(l_view[empty]).all()
    assert torch.isfinite(l_view[~empty]).all()
    assert torch.isfinite(o_view).all()
    o_ref, l_ref = _reference_attention(q4, cache, slots[:, None], lens)
    torch.testing.assert_close(o_view.float(), o_ref[:, 0], rtol=2e-2, atol=4e-3)
    torch.testing.assert_close(
        l_view[~empty], l_ref[~empty, :, 0], rtol=1e-3, atol=1e-3
    )
    o_none, l_none = flashmla_sparse._dcp_compact_fp8_flash_mla(
        q[:0], pages, slots[:0], lens[:0], SM_SCALE
    )
    assert o_none.shape == (0, heads, D_V) and l_none.shape == (0, heads)


@pytest.mark.parametrize("make_cache", [_make_cache, _make_nvfp4_cache])
@pytest.mark.parametrize(
    "dcp,heads,interleave", [(2, 24, 1), (4, 64, 64), (8, 96, 1), (8, 128, 16)]
)
def test_dcp_ranks_merge_to_reference(dcp, heads, interleave, make_cache):
    """The DCP path end to end on the merged DSA cache (MLA rows at the head
    of each padded 64-token unit): per rank, the real index converter
    (compacted, with counts) feeds _dcp_compact_fp8_flash_mla, and the LSE
    merge of the ranks equals fp32 attention over the full top-k rows."""
    from vllm.v1.attention.backends.mla import flashmla_sparse
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_filter_and_convert_dcp_index,
    )
    from vllm.v1.kv_cache_interface import MergedMLAIndexerSpec

    g = torch.Generator(device=DEV).manual_seed(dcp * heads + interleave)
    dense = make_cache(8)
    row = dense.shape[-1]
    cache_dtype = "fp8_ds_mla" if row == BYTES_PER_TOKEN else "nvfp4_ds_mla"
    unit = MergedMLAIndexerSpec.merged_padded_unit_bytes(
        cache_dtype, D_QK, torch.uint8, 132, kernel_block_size=PAGE
    )
    assert unit == {"fp8_ds_mla": 50512, "nvfp4_ds_mla": 36960}[cache_dtype]
    merged = torch.randint(
        0, 256, (NUM_PAGES, unit), dtype=torch.uint8, device=DEV, generator=g
    )
    pages = merged.as_strided((NUM_PAGES, PAGE, row), (unit, row, 1))
    pages.copy_(dense.view(NUM_PAGES, PAGE, row))

    # Each rank's local KV of request r sits in its own pages (never the
    # poison page); global token t is on rank (t // interleave) % dcp.
    seq_lens = [2048, 5000, 8191]
    num_blocks = -(-max(seq_lens) // (dcp * PAGE)) + 1
    block_tables = (
        torch.randperm(NUM_PAGES - 1, device=DEV, generator=g)[
            : dcp * len(seq_lens) * num_blocks
        ]
        .view(dcp, len(seq_lens), num_blocks)
        .int()
    )
    num_tokens, topk = 40, 2048
    req_id = torch.arange(num_tokens, device=DEV, dtype=torch.int32) % len(seq_lens)
    topk_ids = torch.full((num_tokens, topk), -1, dtype=torch.int32, device=DEV)
    for i, r in enumerate(req_id.tolist()):
        ids = torch.randperm(seq_lens[r], device=DEV, generator=g)
        if i == 0:  # owned by rank 0 only: an empty row on the other ranks
            ids = ids[(ids // interleave) % dcp == 0]
        n = min(len(ids), topk - 7 * (i % 3))  # some rows end in -1s
        topk_ids[i, :n] = ids[:n]
    q = (
        torch.randn((num_tokens, heads, D_QK), device=DEV, generator=g) * 0.5
    ).bfloat16()

    outs, lses = [], []
    for rank in range(dcp):
        slots, counts = triton_filter_and_convert_dcp_index(
            req_id,
            block_tables[rank],
            topk_ids,
            dcp_size=dcp,
            dcp_rank=rank,
            cp_kv_cache_interleave_size=interleave,
            BLOCK_SIZE=PAGE,
            NUM_TOPK_TOKENS=topk,
            return_valid_counts=True,
            compact_valid_to_front=True,
        )
        out, lse = flashmla_sparse._dcp_compact_fp8_flash_mla(
            q, pages, slots, counts, SM_SCALE
        )
        if rank > 0:
            assert counts[0] == 0 and torch.isneginf(lse[0]).all()
            assert (out[0] == 0).all()
        outs.append(out.float())
        lses.append(lse)
    lse = torch.logsumexp(torch.stack(lses), dim=0)
    out = sum(
        o * (rank_lse - lse).exp()[..., None]
        for o, rank_lse in zip(outs, lses, strict=True)
    )

    # Reference: global token t -> its rank's local slot.
    t = topk_ids.clamp(min=0).long()
    owner = (t // interleave) % dcp
    local = t // (interleave * dcp) * interleave + t % interleave
    block = block_tables[owner, req_id[:, None].long(), local // PAGE]
    ref_slots = torch.where(topk_ids >= 0, block * PAGE + local % PAGE, -1)
    lens = (topk_ids >= 0).sum(dim=1).int()
    o_ref, l_ref = _reference_attention(q[:, None], dense, ref_slots[:, None], lens)
    torch.testing.assert_close(out, o_ref[:, 0], rtol=2e-2, atol=4e-3)
    torch.testing.assert_close(lse, l_ref[..., 0], rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------
# (3) Stale metadata is NOT detected by FlashMLA
# --------------------------------------------------------------------------


def test_stale_metadata_with_larger_lengths_is_silently_wrong():
    """Documents why the call site rebuilds metadata on every call: FlashMLA
    only checks the Config (shapes), not the lengths baked into the split
    boundaries. Reusing metadata built for shorter rows runs without error and
    drops the extra blocks of the first/last request of each part. If FlashMLA
    starts rejecting this, the per-call rebuild may become optional."""
    b, s_q, h, width = 512, 1, 128, 2048
    cache = _make_cache(4)
    q, idx, lens = _make_inputs(b, s_q, h, width, "full", seed=11)
    short = torch.randint(
        1, 512, (b,), device=DEV, generator=torch.Generator(device=DEV).manual_seed(3)
    ).to(torch.int32)
    stale = build_sparse_sched_meta(short, (b, s_q, h), width)
    o_stale, l_stale = _run(q, cache, idx, lens, stale)
    o_ref, l_ref = _run(q, cache, idx, lens, get_mla_metadata()[0])
    torch.cuda.synchronize()
    bad = ((l_stale - l_ref).abs() > 1e-2).any(dim=(1, 2))
    assert bad.any(), "stale metadata no longer changes the result"
    assert not bad.all()  # requests strictly inside a part use runtime lengths
    assert math.isfinite(float(o_stale.float().abs().max()))
