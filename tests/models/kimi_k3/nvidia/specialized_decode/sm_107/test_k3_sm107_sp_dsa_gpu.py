# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from the source branch's tests/models/kimi_k3/test_specialized_decode_sp_dsa_gpu.py for the SM107
# specialized decode (vllm/models/kimi_k3/nvidia/specialized_decode/sm_107).
"""Kimi K3.1 specialized decode under sequence parallelism (C512: TP4 = DCP4, 24
DSA heads per rank, q4, rows 4..512), one GPU: the SP DSA policy covers every
captured row count with a publish-only query (dsa.sp), the 24-head MlaQuery geometry
(mla.query TILED_FROM[24]) and its refusal of a gathered poll, its query rows, q_c
and KV row against the stock arithmetic, QueryNorm's q_c bit for bit, and the
DCP4 kernels (query, QueryWait, index, "proj" DcpValue) compiling co-resident at
rows up to 512. The 4-rank mailbox and value exchange:
tests/distributed/test_kimi_k3_sm107_sdec_sp_dsa.py."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels need SM100/SM103/SM107",
)

# The SM107 specialized decode (specialized_decode/sm_107) serves VR200 only.
_SM107_ONLY = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)),
    reason="the SM107 specialized decode runs on compute capability 10.7 only",
)
pytestmark = [
    *(pytestmark if isinstance(pytestmark, list) else [pytestmark]),
    _SM107_ONLY,
]

HEADS, DCP, Q_HEADS, CAP, Q = 24, 4, 128, 512, 4
V, INDEX_DIM, INDEX_HEADS = 128, 128, 32
GATE = HEADS * V
PITCH = 2112 + GATE + INDEX_DIM + INDEX_HEADS  # 5344
UNIT, FP8_ROW, INDEX_ROW = 64, 576, 132
EPS = 1e-6
ROWS = tuple(range(4, CAP + 1, 4))  # every captured R (n = 1..128 requests)
SOME_ROWS = (4, 8, 16, 20, 64, 68, 128, 256, 260, 320, 324, 512)


def rms_norm(x, weight):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return (x * weight.float()).to(torch.bfloat16)


def layer_weights(seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    bf, dev = torch.bfloat16, "cuda"

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=dev) * scale).to(bf)

    return (
        (torch.rand(1536, generator=gen, device=dev) + 0.5).to(bf),
        (torch.rand(512, generator=gen, device=dev) + 0.5).to(bf),
        randn(HEADS * 192, 1536, scale=1 / 40),
        randn(HEADS, 128, 512, scale=1 / 12),
        randn(HEADS, 128, 512, scale=1 / 16).transpose(1, 2),
    )


def test_sp_policy_covers_every_row():
    """Every captured R (4..512, q4) gets a publish-only query overlap mode, the
    24-head MlaQuery tiles it, and the DcpBox holds it: no row is left to a stock
    seam or to MlaQuery's gathered poll."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa import sp
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.overlap import (
        overlap_for,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.mla.query import (
        TILED_FROM,
    )

    assert sp.SP_HEADS == HEADS and sp.SP_ROWS == CAP
    assert TILED_FROM[HEADS] <= ROWS[0]
    modes = {r: overlap_for(r, sp.SP_QUERY_OVERLAP, True, True) for r in ROWS}
    assert set(modes.values()) <= set(sp.SP_MODES), modes
    assert "serial" not in sp.SP_MODES
    assert sp.SP_PARTIALS == "proj" and sp.SP_KERNELS == ("trtllm",)


def test_query_geometry_24_heads():
    """mla.query at 24 heads: tiled at every SP row (8-CTA clusters x 24 heads x
    16-64-request tiles, a tile's 192 CTAs inserting its KV rows), and a DCP query
    that would poll the gathered mailbox in-kernel (24 clusters per tile: more than
    Rubin holds at once) is refused at construction."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.mla.query import (
        MlaQuery,
    )

    for r in ROWS:
        query = MlaQuery(
            HEADS, r, dsa=True, fp8=True, dcp=DCP, q_heads=Q_HEADS, gather=False,
            store_qc=False,
        )  # fmt: skip
        assert query.wide and query.cluster == 8, r
        assert query.tokens in (16, 32, 64) and query.tokens <= 8 * HEADS
        assert query.grid == (8, HEADS, -(-r // query.tokens)), (r, query.grid)
        assert not query.gather and not query.tile_poll
    with pytest.raises(AssertionError, match="publishes only"):
        MlaQuery(HEADS, 64, dsa=True, fp8=True, dcp=DCP, q_heads=Q_HEADS)
    # Without DCP (no mailbox) the 24-head query keeps its own rows only.
    assert MlaQuery(HEADS, 64, dsa=True, fp8=True).gather


@pytest.mark.parametrize("batch", [4, 16, 64, 128, 324, 512])
def test_fp8_query_24_heads(batch):
    """The 24-head per-tensor FP8 query rows [B, 24, 576], q_c and the FP8 KV row
    as the stock path computes and writes them; the forked query (SP's publish-only
    modes) writes the same query and KV bytes and no q_c, and QueryNorm's q_c is
    the serial query's bit for bit."""
    import cutlass

    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.common.host import (
        fake,
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.query_stage import (  # noqa: E501
        compile_norm,
        compile_query,
    )

    bf, dev, fp8 = torch.bfloat16, "cuda", torch.float8_e4m3fn
    gen = torch.Generator(device=dev).manual_seed(batch)
    projected = (torch.randn(batch, PITCH, generator=gen, device=dev) * 3).to(bf)
    weights = layer_weights(batch)[:4]
    q_norm, kv_norm, q_b, w_uk = weights
    blocks = -(-batch // UNIT) + 2
    slots = torch.randperm(blocks * UNIT, generator=gen, device=dev)[:batch]
    slots[1::7] = -1  # other ranks' tokens write nothing
    q_scale = torch.full((1,), 0.5, device=dev)
    k_scale = torch.full((1,), 0.875, device=dev)
    dev_index = torch.accelerator.current_device_index()

    def run(mode):
        mla = torch.zeros(blocks, UNIT, FP8_ROW, dtype=torch.uint8, device=dev)
        specs = tuple(layout(t) for t in (*weights, mla))
        fakes = tuple(
            fake(shape, stride, dtype, 4 if dtype == torch.float32 else 16)
            for shape, stride, dtype in specs
        )
        kernel = compile_query(
            batch, HEADS, PITCH, fakes, dev_index, fp8=True, overlap=mode
        )
        query = torch.zeros(batch, HEADS, 576, device=dev, dtype=torch.uint8)
        q_c = torch.full((batch, 1536), -7.0, device=dev, dtype=bf)
        args = (to_cute(projected), *(to_cute(w) for w in weights), to_cute(mla))
        args += (to_cute(slots, 8), to_cute(1 / q_scale, 4), to_cute(k_scale, 4))
        args += (cutlass.Float32(EPS), to_cute(query), to_cute(q_c), stream(dev))
        kernel(*args)
        return query, q_c, mla

    serial, forked = run("serial"), run("fork")
    assert torch.equal(serial[0], forked[0]) and torch.equal(serial[2], forked[2])
    assert (forked[1] == -7.0).all()

    query, q_c, mla = serial
    ref_q_c = rms_norm(projected[:, :1536], q_norm)
    torch.testing.assert_close(q_c, ref_q_c, atol=1e-2, rtol=1e-2)
    q = (ref_q_c @ q_b.T).view(batch, HEADS, 192)
    latent = torch.bmm(q[..., :128].transpose(0, 1), w_uk).transpose(0, 1)
    ref_query = torch.cat([latent, q[..., 128:]], -1).float() / q_scale
    torch.testing.assert_close(query.view(fp8).float(), ref_query, atol=3e-2, rtol=7e-2)

    stock = torch.zeros_like(mla)
    kv = rms_norm(projected[:, 1536:2048], kv_norm)
    ops.concat_and_cache_mla(kv, projected[:, 2048:2112], stock, slots, "fp8", k_scale)
    live = slots[slots >= 0]
    ours, ref = (c[live // UNIT, live % UNIT] for c in (mla, stock))
    assert torch.equal(ours[:, 512:], ref[:, 512:])  # k_pe
    assert (ours[:, :512] != ref[:, :512]).float().mean() < 0.01
    dead = torch.ones(blocks * UNIT, dtype=torch.bool, device=dev)
    dead[live] = False
    assert (mla.view(-1, FP8_ROW)[dead] == 0).all()  # unowned slots untouched

    norm = compile_norm(batch, HEADS, PITCH)
    normed = torch.zeros(batch, 1536, device=dev, dtype=bf)
    args = (to_cute(projected), to_cute(q_norm), to_cute(normed))
    norm(*args, cutlass.Float32(EPS), stream(dev))
    assert torch.equal(normed.view(torch.int16), q_c.view(torch.int16))


def test_sp_dsa_kernels_compile_resident():
    """PreparedDsa's kernels at the SP geometry (24 heads, DCP4, a 128-head FP8 q
    mailbox of 512 rows, base-2 trtllm-gen LSE, "proj" partials) compile at rows up to
    512 in each row's overlap mode, QueryWait polls the 96 gathered heads, and the
    "proj" DcpValue's whole grid (4 x 24 projecting CTAs, 24 x min(tiles, (SMs - 96)
    // 24) merging ones) is co-resident at one 512-thread CTA per SM."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.common.host import (
        layout,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa import sp, wide
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.overlap import (
        overlap_for,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.prepared import (
        compile_core,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.query_stage import (  # noqa: E501
        compile_norm,
        compile_wait,
    )

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if sms <= 152:
        pytest.skip("Rubin-class SP decode")
    dev = "cuda"
    weights = layer_weights(0)
    caches = (
        torch.zeros(8, UNIT, FP8_ROW, dtype=torch.uint8, device=dev),
        torch.zeros(8, UNIT, INDEX_ROW, dtype=torch.uint8, device=dev),
    )
    specs = tuple(layout(t) for t in (*weights, *caches))
    device = torch.accelerator.current_device_index()
    for rows in SOME_ROWS:
        mode = overlap_for(rows, sp.SP_QUERY_OVERLAP, True, True)
        args = rows, HEADS, PITCH, specs, device, Q, DCP, Q_HEADS, sms
        *_, value = compile_core(
            *args, fp8=True, lse_base_e=False, overlap=mode, partials="proj", cap=CAP
        )
        grid = wide.check_proj_residency(value, HEADS, DCP, rows, sms, device)
        merge = min(-(-rows // 16), (sms - DCP * HEADS) // HEADS)
        assert grid == (HEADS, DCP + merge, 1) and HEADS * grid[1] <= sms, grid
        assert compile_wait(rows, HEADS, DCP, Q_HEADS, True, CAP) is not None
        if mode == "fork":
            assert compile_norm(rows, HEADS, PITCH) is not None
