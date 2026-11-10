# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DEP decode kernels on Rubin (compute capability 10.7; ``runtime.dep``:
TP1 = DCP1, all 96 heads per rank), one GPU:

* the decode projections at the DEP shapes (dsa_in 14560 x 7168, kda_in 49376 x
  7168, o_proj 7168 x 12288) with ``sp_policy``'s DEP tactics (warp split-K, the
  narrow split-K for the o_proj's K = 12288 at 1-16 rows, split-K at 17-64, cuBLAS
  above), R = 1..80 per rank x q4 and wider, within one BF16 rounding of an FP32
  reference; the policy covers every DEP row and keys (1, 1);
* the 96-head DSA query into trtllm-gen's 128-head padded FP8 buffer (``fi_heads``:
  a host-side stride): the first 96 heads bit for bit the unpadded query's, the pad
  heads zero, the KV row the same; MlaValue over the padded [B, 128, 512] latent
  bit for bit the contiguous [B, 96, 512] one's;
* ``SpKdaCore`` in "none" mode at 96 heads (KdaSplitStock / KdaVerifyLoopStock, gate
  mode 0) under the DEP context vs the stock Triton verify (``assert_stock``).

Not here: the whole DSA layer (``group_ep.TpDsaCore`` over the fused indexer and
trtllm-gen) against the stock ``KimiK3DSAAttention`` forward at 96 heads, which needs
the real FlashInfer impl and metadata: tests/distributed/
test_kimi_k3_sm107_sdec_dep_step.py compares the whole step.

Each test is at most ~2 minutes (CuTe compiles on a cold cache)."""

import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("needs a Rubin GPU (compute capability 10.7)", allow_module_level=True)

os.environ.setdefault("VLLM_SSM_CONV_STATE_LAYOUT", "DS")  # the decode's conv layout

from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.gemm.decode import (  # noqa: E402
    DecodeProjection,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.gemm.sp_policy import (  # noqa: E402
    DEP_GROUP,
    sp_projection_tactics,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.shapes import (  # noqa: E402
    Context,
    Path,
    Unsupported,
)
from vllm.platforms import current_platform  # noqa: E402

_SM107_ONLY = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)),
    reason="the SM107 specialized decode runs on compute capability 10.7 only",
)
pytestmark = _SM107_ONLY

DEVICE = torch.device("cuda", 0)
WIDTH, HEADS = 7168, 96
DSA_PITCH = 2112 + HEADS * 128 + 128 + 32  # 14560
KDA_PITCH = -(-(4 * HEADS * 128 + 128 + HEADS) // 16) * 16  # 49376
SHAPES = {
    "dsa_in": (DSA_PITCH, WIDTH),
    "kda_in": (KDA_PITCH, WIDTH),
    "o_proj": (WIDTH, HEADS * 128),
}


def context(q=4, **overrides):
    torch.cuda.set_device(DEVICE)
    sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    fields = dict(
        arch="rubin", tp=1, rank=0, dcp=1, sms=sms, q=q, max_rows=512,
        max_context=1 << 20, bf16_state=False, dp=16, ep=16,
        heads_per_rank=HEADS, dep=True,
    )  # fmt: skip
    return Context(**(fields | overrides))


def operands(n, k, rows, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(rows, k, device=DEVICE, generator=g).bfloat16()
    w = (torch.randn(n, k, device=DEVICE, generator=g) * 0.02).bfloat16()
    return x, w


def check_close(out, x, w):
    ref = (x.float() @ w.float().t()).bfloat16().float()
    err = (out.float() - ref).abs()
    bound = ref.abs() * 2**-7 + 2**-12
    assert bool((err <= bound).all()), f"max err {err.max().item()} over the BF16 bound"


# -- projections ---------------------------------------------------------------------
@pytest.mark.parametrize("q", [2, 4, 8])
def test_dep_policy_covers_rows(q):
    """Every DEP row (n = 1..512 requests x q) gets a structurally valid tactic: the
    in-projections warp split-K up to 16 rows, the o_proj (K / 128 = 96 > 64) the
    narrow split-K there; split-K at 17-64; cuBLAS above."""
    rows = tuple(range(q, 512 * q + 1, q))
    for kind, (n, k) in SHAPES.items():
        tactics = sp_projection_tactics(kind, n, k, rows, context(q))
        assert set(tactics) == set(rows), kind
        small = {tactics[r][0] for r in rows if r <= 16}
        assert small == ({"split_k"} if kind == "o_proj" else {"warp"}), (kind, small)
        assert {tactics[r][0] for r in rows if 16 < r <= 64} <= {"split_k"}, kind
        assert {tactics[r][0] for r in rows if r > 64} == {"cublas"}, kind
    narrow = sp_projection_tactics("o_proj", WIDTH, HEADS * 128, (2, 8, 10, 16),
                                   context(2))  # fmt: skip
    assert [t[1][:3] for t in narrow.values()] == [(64, 8, 2), (64, 8, 2),
                                                   (64, 16, 2), (64, 16, 2)]


def test_dep_policy_refuses_other_contexts():
    with pytest.raises(Unsupported, match="TP1 = DCP1"):
        sp_projection_tactics("o_proj", WIDTH, HEADS * 128, (4,), context(tp=8))
    with pytest.raises(Unsupported, match="TP1 = DCP1"):
        sp_projection_tactics("o_proj", WIDTH, HEADS * 128, (4,), context(dep=False))


@pytest.mark.parametrize("kind", list(SHAPES))
def test_dep_projection_matches_reference(kind):
    """Each DEP family at R = 1..80 per rank (q4 rows, odd and boundary rows) and a
    few wide rows, within one BF16 rounding of the FP32 reference; GENERIC (no
    table) keyed (1, 1)."""
    rows = tuple(sorted({*range(4, 81, 4), 1, 2, 3, 9, 15, 16, 17, 33, 64, 65,
                         128, 320}))  # fmt: skip
    n, k = SHAPES[kind]
    x, w = operands(n, k, max(rows), seed=len(kind))
    tactics = sp_projection_tactics(kind, n, k, rows, context(4))
    projection = DecodeProjection(w, rows, tactics, family=kind, group=DEP_GROUP)
    out = torch.empty(max(rows), n, device=DEVICE, dtype=torch.bfloat16)
    projection.check_operands(x, out)
    for r in rows:
        out.fill_(float("nan"))
        projection.run(x[:r], out[:r])
        check_close(out[:r], x[:r], w)
        assert projection.choice(r).path is Path.GENERIC


def test_dep_oproj_graph_replay():
    """The o_proj (the DEP block's output) replays in a CUDA graph at the narrow
    split-K rows and wider ones with fresh inputs."""
    n, k = SHAPES["o_proj"]
    rows = (4, 8, 12, 16, 32, 64, 80)
    x, w = operands(n, k, max(rows), seed=7)
    tactics = sp_projection_tactics("o_proj", n, k, rows, context(4))
    projection = DecodeProjection(w, rows, tactics, family="oproj", group=DEP_GROUP)
    out = torch.empty(max(rows), n, device=DEVICE, dtype=torch.bfloat16)
    for r in rows:
        projection.run(x[:r], out[:r])
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for r in rows:
            projection.run(x[:r], out[:r])
    for seed in (1, 2):
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        x.copy_(torch.randn(x.shape, device=DEVICE, generator=gen))
        graph.replay()
        check_close(out[: max(rows)], x, w)


# -- the 96-head DSA query and value -------------------------------------------------
UNIT, FP8_ROW, INDEX_ROW = 64, 576, 132
EPS = 1e-6
FI_HEADS = 128


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


def test_trtllm_gen_pads_96_heads():
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        trtllm_gen_query_heads,
    )

    assert trtllm_gen_query_heads(HEADS) == FI_HEADS
    for heads in (6, 12, 24):  # TP_EXPERTS / EP_GROUP head counts: unpadded
        assert trtllm_gen_query_heads(heads) == heads


@pytest.mark.parametrize("batch", [2, 4, 16, 64, 80, 320])
def test_padded_fp8_query_96_heads(batch):
    """MlaQuery at 96 heads (lean below TILED_FROM[96] = 4, tiled from it) into the
    128-head padded buffer: the first 96 heads are the unpadded query's bytes, the
    pad heads stay zero, the KV row is the same (the kernel stores by layout)."""
    import cutlass

    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.common.host import (
        fake,
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.query_stage import (  # noqa: E501
        compile_query,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.mla.query import (
        TILED_FROM,
        MlaQuery,
    )

    assert MlaQuery(HEADS, batch, dsa=True, fp8=True).wide == (batch >= TILED_FROM[96])
    bf, dev = torch.bfloat16, "cuda"
    pitch = DSA_PITCH
    gen = torch.Generator(device=dev).manual_seed(batch)
    projected = (torch.randn(batch, pitch, generator=gen, device=dev) * 3).to(bf)
    weights = layer_weights(batch)[:4]
    blocks = -(-batch // UNIT) + 2
    slots = torch.randperm(blocks * UNIT, generator=gen, device=dev)[:batch]
    slots[1::7] = -1  # padding rows (DP dummy steps) write nothing
    q_scale = torch.full((1,), 0.5, device=dev)
    k_scale = torch.full((1,), 0.875, device=dev)
    dev_index = torch.accelerator.current_device_index()

    def run(fi_heads):
        mla = torch.zeros(blocks, UNIT, FP8_ROW, dtype=torch.uint8, device=dev)
        specs = tuple(layout(t) for t in (*weights, mla))
        fakes = tuple(
            fake(shape, stride, dtype, 4 if dtype == torch.float32 else 16)
            for shape, stride, dtype in specs
        )
        kernel = compile_query(
            batch, HEADS, pitch, fakes, dev_index, fp8=True, overlap="serial",
            fi_heads=fi_heads,
        )  # fmt: skip
        rows = torch.zeros(batch, fi_heads or HEADS, 576, device=dev, dtype=torch.uint8)
        q_c = torch.zeros(batch, 1536, device=dev, dtype=bf)
        args = (to_cute(projected), *(to_cute(w) for w in weights), to_cute(mla))
        args += (to_cute(slots, 8), to_cute(1 / q_scale, 4), to_cute(k_scale, 4))
        args += (cutlass.Float32(EPS), to_cute(rows[:, :HEADS]), to_cute(q_c))
        kernel(*args, stream(dev))
        return rows, q_c, mla

    padded, plain = run(FI_HEADS), run(0)
    torch.cuda.synchronize()
    assert torch.equal(padded[0][:, :HEADS], plain[0])
    assert (padded[0][:, HEADS:] == 0).all()
    assert torch.equal(padded[1], plain[1]) and torch.equal(padded[2], plain[2])
    assert (plain[0] != 0).any()


@pytest.mark.parametrize("batch", [4, 64, 320])
def test_value_reads_the_padded_latent(batch):
    """MlaValue compiled for the 128-head latent stride (``compile_core(fi_heads=
    128)``) over ``latent[:, :96]`` of a [B, 128, 512] buffer equals the unpadded
    compile over a contiguous [B, 96, 512] copy, bit for bit."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.common.host import (
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.prepared import (
        compile_core,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.query_stage import (  # noqa: E501
        gate_columns,
    )

    dev, bf, q = "cuda", torch.bfloat16, 4
    weights = layer_weights(batch)
    caches = (
        torch.zeros(8, UNIT, FP8_ROW, dtype=torch.uint8, device=dev),
        torch.zeros(8, UNIT, INDEX_ROW, dtype=torch.uint8, device=dev),
    )
    specs = tuple(layout(t) for t in (*weights, *caches))
    device = torch.accelerator.current_device_index()
    args = batch, HEADS, DSA_PITCH, specs, device, q
    *_, padded_value = compile_core(*args, fp8=True, fi_heads=FI_HEADS)
    *_, plain_value = compile_core(*args, fp8=True)
    gen = torch.Generator(device=dev).manual_seed(batch)
    latent = torch.randn(batch, FI_HEADS, 512, generator=gen, device=dev).to(bf)
    projected = torch.randn(batch, DSA_PITCH, generator=gen, device=dev).to(bf)
    lengths = torch.full((batch // q,), 300, dtype=torch.int32, device=dev)
    lengths[1::3] = 0  # empty (padding) requests
    gate = to_cute(projected[:, slice(*gate_columns(HEADS))])
    uv = to_cute(weights[4])
    outs = []
    for value, rows in ((padded_value, latent[:, :HEADS]),
                        (plain_value, latent[:, :HEADS].contiguous())):  # fmt: skip
        out = torch.full((batch, HEADS * 128), float("nan"), device=dev, dtype=bf)
        value(to_cute(rows), uv, gate, to_cute(lengths, 4), to_cute(out), stream(dev))
        outs.append(out)
    torch.cuda.synchronize()
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))


def test_padded_query_heads_refused_outside_tp_flashinfer():
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.common.host import (
        layout,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.prepared import (
        compile_core,
    )

    weights = layer_weights(0)
    caches = (
        torch.zeros(8, UNIT, FP8_ROW, dtype=torch.uint8, device="cuda"),
        torch.zeros(8, UNIT, INDEX_ROW, dtype=torch.uint8, device="cuda"),
    )
    specs = tuple(layout(t) for t in (*weights, *caches))
    device = torch.accelerator.current_device_index()
    with pytest.raises(ValueError, match="padded query heads"):
        compile_core(4, HEADS, DSA_PITCH, specs, device, 4, fp8=False, fi_heads=128)
    with pytest.raises(ValueError, match="padded query heads"):
        compile_core(4, HEADS, DSA_PITCH, specs, device, 4, fp8=True, fi_heads=64)


# -- the 96-head KDA plain verify ----------------------------------------------------
def kda_layer(q, seed=0):
    from tests.models.kimi_k3.nvidia.specialized_decode.sm_107.test_k3_sm107_kda_loop_gpu import (  # noqa: E501
        kda_attention,
        pitch_of,
    )

    attention = kda_attention(seed=seed, heads=HEADS)
    attention.head_dim, attention.conv_size, attention.spec_query_len = 128, 4, q
    attention.get_state_dtype = lambda: (torch.bfloat16, torch.float32)
    attention.in_proj_qkvgfab = SimpleNamespace(
        weight=torch.empty(pitch_of(HEADS), 8, device="meta")
    )
    attention.recoverssm_records = None
    # stock's resolved verify backend: the plain cores (and stock_run) are native
    attention.kda_spec_decode_backend = "native"
    return attention


# (q, requests): KdaSplitStock and the stock-math wide verify (FP32_VERIFY_WIDE)
# rows at 96 heads (the tables are keyed by requests: B4 retunes them).
KDA_SHAPES = [
    *(pytest.param(4, n, id=f"q4-n{n}") for n in (1, 4, 12, 16, 36, 64)),
    *(pytest.param(8, n, id=f"q8-n{n}") for n in (1, 8, 16)),
    pytest.param(2, 20, id="q2-n20"),
]


@pytest.mark.parametrize("q,requests", KDA_SHAPES)
def test_dep_kda_core_matches_stock_verify(monkeypatch, q, requests):
    from tests.models.kimi_k3.nvidia.specialized_decode.sm_107.test_k3_sm107_kda_loop_gpu import (  # noqa: E501
        caches,
        pitch_of,
        projected_rows,
        spec_inputs,
    )
    from tests.models.kimi_k3.nvidia.specialized_decode.sm_107.test_k3_sm107_kda_stock_gpu import (  # noqa: E501
        assert_stock,
        clone,
        stock_run,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.kda import sp

    rows = requests * q
    attention = kda_layer(q)
    core = sp.SpKdaCore(attention, (rows,), records="none", ctx=context(q))
    assert sp.plain_gate_mode(HEADS) == 0
    assert set(core.plain.policies.values()) == {(pitch_of(HEADS), 0)}
    assert core.choice(rows).path.value == "generic"
    slots = rows + 1
    base = caches(slots, q, heads=HEADS)
    table, accepted = spec_inputs(requests, q, slots)
    projected = projected_rows(rows, pitch=pitch_of(HEADS))
    core.prepare_caches(list(base))
    got_caches = clone(base)
    got = torch.empty(rows, 128 * HEADS, dtype=torch.bfloat16, device=DEVICE)
    metadata = SimpleNamespace(
        spec_state_indices_tensor=table, num_accepted_tokens=accepted
    )
    core.run(projected, got, caches=list(got_caches), metadata=metadata)
    torch.cuda.synchronize()
    want, want_caches = stock_run(
        attention, projected, base, table, accepted, q, monkeypatch
    )
    assert_stock(got, got_caches, want, want_caches, base, table.view(-1).long())


def test_dep_kda_core_refusals():
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.kda import sp

    with pytest.raises(Unsupported, match="expert-parallel decode modes only"):
        sp.SpKdaCore(kda_layer(4), (16,), records="none", ctx=context(4, dep=False))
    with pytest.raises(Unsupported, match="at DEP"):
        sp.SpKdaCore(kda_layer(4), (16,), records="triton", ctx=context(4))
    with pytest.raises(Unsupported, match="verify length q1"):
        sp.SpKdaCore(kda_layer(1), (4,), records="none", ctx=context(1))
    # stock resolving FlashInfer's packed fused KDA decode at 96 heads (auto):
    # another rounding form than the plain cores' native verify
    attention = kda_layer(4)
    attention.kda_spec_decode_backend = "flashinfer"
    with pytest.raises(Unsupported, match="stock KDA verify backend 'flashinfer'"):
        sp.SpKdaCore(attention, (16,), records="none", ctx=context(4))
