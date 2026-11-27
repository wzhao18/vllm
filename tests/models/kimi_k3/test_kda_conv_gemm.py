# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA prefill in_proj Q|K|V GEMM with the conv + SiLU epilogue (sm_107,
specialized_prefill.sm107_gemm.kda_conv_gemm) is bitwise equal to the
cuBLASLt in_proj followed by ops/kda_qkv_conv: Q|K|V planes and the FP32 conv
state, for sequence starts inside and at 128-row blocks, sequences shorter
than the conv window, padding rows past the last sequence and accumulators
deep enough below zero to take the exact ``div.full`` SiLU path.

Exact-input operands (small-integer FP8, power-of-two scales) make every
summation order give the same FP32 accumulator, so the comparison checks the
epilogue.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)), reason="sm_107 only"
)

PART, K = 512, 1024


def _operands(m, n, k, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randint(-2, 3, (m, k), device=dev, generator=g).float()
    w = torch.randint(-2, 3, (n, k), device=dev, generator=g).float()
    sx = torch.randint(118, 122, (((m + 127) // 128) * 128 * k // 32,), device=dev,
                       generator=g).to(torch.uint8)  # fmt: skip
    sw = torch.randint(117, 121, (n * k // 32,), device=dev, generator=g).to(torch.uint8)
    return x.to(torch.float8_e4m3fn), sx, w.to(torch.float8_e4m3fn), sw


def _run(seqlens, m, deep=False, tactic=None, seed=0, state_dtype=torch.float32,
         part=PART, k=K):  # fmt: skip
    from vllm.models.kimi_k3.nvidia.ops import kda_conv
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        KdaConvGemm,
    )

    dev = torch.device("cuda")
    n = 3 * part
    xq, xs, wq, ws = _operands(m, n, k, dev, seed)
    proj = torch._scaled_mm(
        xq, wq.t(),
        scale_a=xs.view(torch.float8_e8m0fnu), scale_b=ws.view(torch.float8_e8m0fnu),
        out_dtype=torch.bfloat16,
    )  # fmt: skip
    g = torch.Generator(device=dev).manual_seed(seed + 1)
    cw = torch.randn(n, 4, device=dev, generator=g) * 0.5
    if deep:
        # Columns whose conv reaches acc < -87 (d = 1 + exp(-acc) > 2^126).
        cw[::7] *= 60.0 / float(proj.float().abs().mean())
    cw = cw.contiguous()
    qsl = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32,
                       device=dev)  # fmt: skip
    nseq = len(seqlens)
    lines = torch.randperm(2 * nseq + 2, device=dev, generator=g)[:nseq].to(torch.int32)
    has_init = torch.arange(nseq, device=dev) % 2 == 1
    state0 = torch.randn(2 * nseq + 2, n, 4, device=dev, generator=g)
    state0 = state0.to(torch.bfloat16).to(state_dtype)[:, :, :3]

    st_ref = state0.clone()
    ref = kda_conv.qkv_conv(proj, cw, st_ref, qsl, lines, has_init, part)
    gemm = KdaConvGemm(wq, ws, cw, part, tactic=tactic)
    st = state0.clone()
    out, _ = gemm(xq, xs, st, qsl, lines, has_init, null_block_id=NULL_BLOCK_ID)
    torch.cuda.synchronize()
    tot = int(qsl[-1])
    # qkv_conv leaves the rows of null-block sequences (padding) unwritten.
    keep = torch.ones(tot, dtype=torch.bool, device=dev)
    for i in (lines == NULL_BLOCK_ID).nonzero().flatten().tolist():
        keep[int(qsl[i]) : int(qsl[i + 1])] = False
    assert torch.equal(out[:, :tot][:, keep], ref[:, :tot][:, keep])
    assert torch.equal(st, st_ref)
    if deep:
        # Tiny nonzero outputs only come from the exact path's band.
        y = ref[:, :tot][:, keep].float()
        assert bool(((y != 0) & (y.abs() < 1e-30)).any())


@pytest.mark.parametrize(
    "seqlens,m",
    [
        ([1365] * 3, 4096),
        ([1, 2, 3, 125, 128, 3, 256, 600, 5], 1300),
        ([127, 1, 128, 129, 2], 512),
        ([700, 300], 1100),
    ],
)
def test_kda_conv_gemm_bitwise(seqlens, m):
    _run(seqlens, m)


def test_kda_conv_gemm_bf16_state():
    # The deployed KDA layer keeps a BF16 conv state with FP32 conv weights.
    _run([1, 2, 3, 125, 128, 3, 256, 600, 5], 1300, state_dtype=torch.bfloat16)


def test_kda_conv_gemm_exact_silu_path():
    _run([900, 513, 7], 1420, deep=True, seed=3)


def test_kda_conv_gemm_old_column_pass():
    # The previous per-row column pass (kept for the inverse-norm variant).
    _run([1, 2, 3, 125, 128, 3, 256, 600, 5], 1300,
         tactic=((512, 256), (4, 1), 256, 8, "n", 0, 0, "ct1"))  # fmt: skip


@pytest.mark.parametrize(
    "part,k,m",
    [
        (12288, 7168, 16384),  # deployed K3.1 KDA layer (DEP prefill)
        (1024, 1024, 16384),  # 3 cluster columns: swizzle 4 -> 3 (was an IMA)
        (3072, 1024, 16384),  # 9 cluster columns: swizzle 4 -> 3 (was an IMA)
        (6144, 7168, 8192),  # 18 cluster columns: swizzle 4 -> 3 (TP2 plane)
        (3072, 7168, 16384),  # TP4 plane (DSA-DP PE): 9 columns, swizzle 3
        (1536, 7168, 8192),  # TP8 plane (TP8+DCP8 PA): SHAPE_TACTICS[4608]
    ],
)
def test_kda_conv_gemm_deployed_tactic_shapes(part, k, m):
    """The deployed 2x4-cluster "m"-raster TACTIC at plane widths whose cluster
    column count the swizzle (4) does not divide: the persistent scheduler
    pads the cluster grid to a multiple of the swizzle, and the conv
    epilogue's unbounded stores of the padded tiles wrote past the planes
    (illegal address at P=1024 K=1024 M=16384). shape_tactic fits the swizzle
    to the shape; output and conv state stay bitwise equal to stock."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_conv_in_proj import (
        tactic_for,
    )

    tactic = tactic_for(3 * part)
    _run([m // 3, m // 3, m - 2 * (m // 3)], m, tactic=tactic, part=part, k=k)


@pytest.mark.parametrize(
    "seqlens,offsets,m",
    [
        ([2000, 3, 4100, 1], [0, 2, 2048, 1], 8192),
    ],
)
def test_kda_conv_gemm_tp8_checkpoint_fixups(seqlens, offsets, m):
    """TP8 plane (P 1536) with the per-shape tactic: block / sequence fix-ups
    (short sequences, initial state on odd sequences) and checkpoint rows,
    bitwise vs in_proj + qkv_conv."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_conv_in_proj import (
        tactic_for,
    )

    _run_checkpoint(seqlens, offsets, m, tactic=tactic_for(3 * 1536), seed=5,
              part=1536, k=7168)  # fmt: skip


def test_tactic_for_shapes():
    """Every deployed q|k|v width has a tactic whose clusters tile it: TP1
    36864, TP2 18432, TP4 9216 (TACTIC) and TP8 4608 (SHAPE_TACTICS)."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_conv_in_proj import (
        TACTIC,
        tactic_for,
    )
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        shape_tactic,
    )

    for n in (36864, 18432, 9216):
        assert tactic_for(n) is TACTIC
        shape_tactic(n, tactic_for(n))
    from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_conv_in_proj import (
        SHAPE_TACTICS,
    )

    assert shape_tactic(4608, tactic_for(4608)) == SHAPE_TACTICS[4608]
    with pytest.raises(ValueError):
        shape_tactic(4608, TACTIC)


def test_shape_tactic():
    """Swizzle fitted to the cluster column count ("m" raster only); N that the
    clusters cannot tile is rejected (kda_conv_in_proj.unsupported_reason)."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_conv_in_proj import TACTIC
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        DEFAULT_TACTIC,
        shape_tactic,
    )

    assert shape_tactic(3 * 12288, TACTIC) == TACTIC  # 36 cluster columns
    assert shape_tactic(3 * 1024, TACTIC)[3] == 3
    assert shape_tactic(3 * 2048, TACTIC)[3] == 3  # 6 columns
    assert shape_tactic(3 * 4096, TACTIC)[3] == 4
    assert shape_tactic(3 * 1024 * 7, TACTIC)[3] == 3  # 21 columns
    assert shape_tactic(3 * 1024 * 5, TACTIC)[3] == 3  # 15 columns
    fitted = shape_tactic(3 * 1024, TACTIC)
    assert fitted[:3] == TACTIC[:3] and fitted[4:] == TACTIC[4:]
    for p in (256, 512, 768, 1280):
        with pytest.raises(ValueError):
            shape_tactic(3 * p, TACTIC)
    assert shape_tactic(3 * 512, DEFAULT_TACTIC) == DEFAULT_TACTIC  # "n" raster


def test_contiguous_rows_bytes():
    """The 16-byte-element copy of a row-strided slice is byte-identical to
    .contiguous() (KDA beta columns of the g|f_a|beta projection)."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_conv_in_proj import (
        contiguous_rows,
    )

    rest = torch.randn(1031, 12512, device="cuda").to(torch.bfloat16)
    for lo, hi in ((12416, 12512), (12288, 12416), (3, 99)):
        x = rest[:, lo:hi]
        y = contiguous_rows(x)
        assert y.is_contiguous() and y.dtype == x.dtype and y.shape == x.shape
        assert torch.equal(y.view(torch.int16), x.contiguous().view(torch.int16))



def test_fused_in_proj_never_captured(monkeypatch):
    """The fused in_proj + conv reads this step's KDA metadata (sequence count,
    has_initial_state, conv state rows), so it must not be baked into a CUDA
    graph: for a prefill-only batch of MIN_ROWS rows (the largest piecewise
    capture size) under stream capture, fused() leaves the batch to the stock
    in_proj (whose conv runs in the eager KDA break) and launches nothing;
    outside capture the same batch takes the fused GEMM, unless the step
    exports a prefill checkpoint (its conv state = raw q|k|v rows)."""
    from types import SimpleNamespace

    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda_conv_in_proj

    calls = []

    def _gemm(*args, **kwargs):
        calls.append(torch.cuda.is_current_stream_capturing())
        raise StopIteration

    dev = "cuda"
    md = SimpleNamespace(
        num_prefills=1,
        num_decodes=0,
        num_spec_decodes=0,
        non_spec_query_start_loc=torch.tensor([0, 2048], dtype=torch.int32, device=dev),
        non_spec_state_indices_tensor=torch.ones(1, dtype=torch.int32, device=dev),
        has_initial_state=torch.ones(1, dtype=torch.bool, device=dev),
        checkpoint=None,
    )
    monkeypatch.setattr(
        kda_conv_in_proj,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"kda": md}),
    )
    fused = kda_conv_in_proj.PrefillKdaConvInProj.__new__(
        kda_conv_in_proj.PrefillKdaConvInProj
    )
    fused._gemm = _gemm
    fused.layer = SimpleNamespace(
        prefix="kda", kv_cache=[torch.zeros(2, 3, 768, device=dev)]
    )
    rows = kda_conv_in_proj.MIN_ROWS
    qa = SimpleNamespace(
        data=torch.zeros(rows, 128, device=dev).to(torch.float8_e4m3fn),
        scale=torch.zeros(rows * 4, dtype=torch.uint8, device=dev),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert fused.fused(qa) is None
    assert calls == []
    with pytest.raises(StopIteration):
        fused.fused(qa)
    assert calls == [False]
    # A step that exports a prefill checkpoint: the fused GEMM also saves the
    # raw q|k|v rows the exporter stores (its epilogue's checkpoint halo) ...
    fused.p = 256
    md.checkpoint = SimpleNamespace(
        checkpoint_offsets=torch.tensor([16], dtype=torch.int32, device=dev),
        offsets=[16],
    )
    _gemm.supports_checkpoint_halo = True
    with pytest.raises(StopIteration):
        fused.fused(qa)
    assert calls == [False, False]
    # ... unless the GEMM variant cannot (stock in_proj + conv).
    _gemm.supports_checkpoint_halo = False
    assert fused.fused(qa) is None
    assert calls == [False, False]


@pytest.mark.parametrize("dim_first", [True, False])
def test_conv_fixup_warm_covers_serving_steps(monkeypatch, dim_first):
    """``conv_fixup.warm`` compiles every fix-up variant a prefill step of up
    to max_num_seqs sequences launches: row / block / sequence counts and the
    state / index strides of the serving KV cache add no compile (either conv
    state layout)."""
    from vllm.model_executor.layers.mamba import mamba_utils
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import conv_fixup

    monkeypatch.setattr(mamba_utils, "is_conv_state_dim_first", lambda: dim_first)
    kern = conv_fixup._conv_fixup_all_kernel

    def variants() -> set:
        return {k for caches in kern.device_caches.values() for k in caches[0]}

    dev = torch.device("cuda")
    nq = 3 * PART
    cw = torch.zeros(nq, 4, device=dev)
    for dtype in (torch.bfloat16, torch.float32):
        conv_fixup.warm(PART, cw, dtype, 32, NULL_BLOCK_ID)
        warmed = variants()
        assert warmed
        # A paged conv state view (row stride of the KV cache page), as the
        # fused path passes it, and 2-D state index rows.
        if dim_first:
            st = torch.empty(5, nq, 7, dtype=dtype, device=dev)[:, :, :3]
        else:
            st = torch.empty(5, 7, nq, dtype=dtype, device=dev)[:, :3].transpose(-1, -2)
        for nseq in (1, 2, 3, 5, 16, 17, 31, 32):
            qsl = torch.zeros(nseq + 1, dtype=torch.int32, device=dev)
            lines = torch.zeros(nseq, 9, dtype=torch.int32, device=dev)[:, 0]
            init = torch.zeros(nseq, dtype=torch.bool, device=dev)
            for rows in (2048, 2049, 2050, 6656, 12287, 12288, 16384):
                out = torch.empty(3, 1, PART, dtype=torch.bfloat16, device=dev)
                side = torch.empty(1, nq, dtype=torch.bfloat16, device=dev)
                conv_fixup.conv_fixup(out.expand(3, rows, PART), side, None, cw, st,
                                      qsl, lines, init, PART, NULL_BLOCK_ID,
                                      compile_only=True)  # fmt: skip
        assert variants() == warmed


def _run_checkpoint(
    seqlens, offsets, m, tactic=None, seed=0, host_offsets=True, part=PART, k=K
):
    """Prefix-hit-shaped prefill steps exporting checkpoints: the fused GEMM's
    halo rows + the stock exporter store bit for bit the conv state the stock
    path (in_proj, qkv_conv, exporter on the in_proj output) stores."""
    from vllm.model_executor.layers.mamba.checkpoint import (
        MambaPrefillCheckpointExporter,
        MambaPrefillCheckpointMetadata,
    )
    from vllm.models.kimi_k3.nvidia.ops import kda_conv
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda_conv_in_proj
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        KdaConvGemm,
    )

    dev = torch.device("cuda")
    n = 3 * part
    xq, xs, wq, ws = _operands(m, n, k, dev, seed)
    proj = torch._scaled_mm(
        xq, wq.t(),
        scale_a=xs.view(torch.float8_e8m0fnu), scale_b=ws.view(torch.float8_e8m0fnu),
        out_dtype=torch.bfloat16,
    )  # fmt: skip
    g = torch.Generator(device=dev).manual_seed(seed + 1)
    cw = (torch.randn(n, 4, device=dev, generator=g) * 0.5).contiguous()
    qsl = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32,
                       device=dev)  # fmt: skip
    nseq = len(seqlens)
    # Cache rows other than the null block (sequence lines must be real rows).
    slots = torch.randperm(3 * nseq + 2, device=dev, generator=g).to(torch.int32)
    slots = slots[slots != NULL_BLOCK_ID]
    lines, ck_slots = slots[:nseq], slots[nseq : 2 * nseq].clone()
    ck_slots[nseq // 2] = NULL_BLOCK_ID  # a checkpoint without a cache block
    # Prefix hits (resumed state) mixed with fresh sequences.
    has_init = torch.arange(nseq, device=dev) % 3 != 0
    state0 = torch.randn(3 * nseq + 2, n, 4, device=dev, generator=g)
    state0 = state0.to(torch.bfloat16)[:, :, :3]
    ckpt = MambaPrefillCheckpointMetadata(
        torch.tensor(offsets, dtype=torch.int32, device=dev),
        ck_slots,
        offsets=list(offsets) if host_offsets else None,
    )
    exporter = MambaPrefillCheckpointExporter(state_len=3)

    st_ref = state0.clone()
    ref = kda_conv.qkv_conv(proj, cw, st_ref, qsl, lines, has_init, part)
    exporter.export(ckpt, raw_qkv=proj[:m], conv_state=st_ref, cu_seqlens=qsl)

    gemm = KdaConvGemm(wq, ws, cw, part, tactic=tactic)
    assert gemm.supports_checkpoint_halo
    st = state0.clone()
    halo = torch.full((nseq, 3, n), float("nan"), dtype=torch.bfloat16, device=dev)
    out, _ = gemm(xq, xs, st, qsl, lines, has_init, null_block_id=NULL_BLOCK_ID,
                  checkpoint_offsets=ckpt.checkpoint_offsets, halo=halo)  # fmt: skip
    kda_conv_in_proj._HALO_QSL[:] = [None, None]
    rows = kda_conv_in_proj.CheckpointRows(
        halo.view(3 * nseq, n), kda_conv_in_proj._halo_qsl(ckpt)
    )
    exporter.export(ckpt, raw_qkv=rows.raw, conv_state=st, cu_seqlens=rows.qsl)
    torch.cuda.synchronize()
    tot = int(qsl[-1])
    assert torch.equal(out[:, :tot], ref[:, :tot])
    # Every conv state row: sequence lines and checkpoint slots, bitwise.
    assert torch.equal(st.view(torch.int16), st_ref.view(torch.int16))
    for s, off in enumerate(offsets):
        if off >= 3:
            r = int(qsl[s]) + off
            assert torch.equal(halo[s].view(torch.int16), proj[r - 3 : r].view(torch.int16))


# Checkpoint offsets per sequence (0 = none, < 3 = no conv window): mid-chunk
# multiples of 16, at the chunk end, windows across 128-row block edges and
# sequence-start fix-up rows, offset 3 (window = the first rows).
@pytest.mark.parametrize(
    "seqlens,offsets,m",
    [
        ([1365] * 3, [1024, 1365, 0], 4096),
        ([700, 300, 64, 900, 37], [640, 16, 64, 128, 32], 2048),
        ([1, 2, 3, 125, 128, 3, 256, 600, 5], [1, 2, 3, 112, 128, 0, 16, 512, 5], 1300),
        ([127, 1, 128, 129, 2], [3, 0, 48, 129, 2], 512),
    ],
)
def test_kda_conv_gemm_checkpoint_bitwise(seqlens, offsets, m):
    _run_checkpoint(seqlens, offsets, m)


def test_kda_conv_gemm_checkpoint_device_offsets():
    # Checkpoint metadata without host offsets: halo starts built on device.
    _run_checkpoint([700, 300, 64, 900, 37], [640, 16, 64, 128, 32], 2048,
                    host_offsets=False, seed=5)  # fmt: skip


def test_kda_conv_gemm_checkpoint_prod_tactic():
    # The deployed fused in_proj tactic (512x256 tile, 2x4 cluster) at the
    # deployed K3.1 plane and K sizes, an F16K prefix-hit step.
    from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_conv_in_proj import TACTIC

    _run_checkpoint([2000, 1000, 5000, 192, 8000], [1984, 1000, 4096, 176, 7680],
                    16384, tactic=TACTIC, seed=7, part=12288, k=7168)  # fmt: skip


def _fake_layer(part):
    from types import SimpleNamespace

    lin = SimpleNamespace(
        weight=SimpleNamespace(data=torch.empty(4 * part, 7168)),
        input_size_per_partition=7168,
    )
    conv = SimpleNamespace(weight=torch.empty(3 * part, 1, 4, dtype=torch.float32))
    return SimpleNamespace(
        in_proj_qkvgfab=lin, conv1d=conv, local_projection_size=part
    )


def test_missing_tactic_raises(monkeypatch):
    """A supported layer whose q|k|v width no tactic tiles raises at load
    (missing SHAPE_TACTICS entry), never silently keeps the stock conv; the
    deployed widths pass."""
    from vllm.models.kimi_k3.nvidia import mxfp8
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda_conv_in_proj
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_device_capability", lambda c: True)
    monkeypatch.setattr(mxfp8, "is_mxfp8_linear", lambda lin: True)
    monkeypatch.setattr(mxfp8, "_kernel_layout", lambda lin: "nk")
    for part in (12288, 6144, 3072, 1536):
        assert kda_conv_in_proj.unsupported_reason(_fake_layer(part)) is None
    with pytest.raises(ValueError, match="SHAPE_TACTICS"):
        kda_conv_in_proj.unsupported_reason(_fake_layer(768))  # TP16 width 2304


@pytest.mark.parametrize(
    "decodes,spec,prefills", [(1, 0, 3), (0, 2, 3), (0, 0, 0)]
)
def test_fused_in_proj_declines_mixed_batches(monkeypatch, decodes, spec, prefills):
    """kv_both prefill workers also serve mixed steps (aggregated + store): a
    step with decode or spec-verify rows (or no prefill) keeps the stock
    in_proj + conv, whatever its row count."""
    from types import SimpleNamespace

    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda_conv_in_proj

    md = SimpleNamespace(
        num_prefills=prefills, num_decodes=decodes, num_spec_decodes=spec
    )
    monkeypatch.setattr(
        kda_conv_in_proj,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer.kda": md}),
    )
    conv = kda_conv_in_proj.PrefillKdaConvInProj.__new__(
        kda_conv_in_proj.PrefillKdaConvInProj
    )
    conv.layer = SimpleNamespace(prefix="layer.kda")
    conv._gemm = object()  # never reached
    qa = SimpleNamespace(data=torch.empty(kda_conv_in_proj.MIN_ROWS * 4, 8))
    assert conv.fused(qa) is None


def test_fused_in_proj_drops_tp_padding_rows(monkeypatch):
    """TP8 KDA in_proj rows end in in_proj_padding (g|f_a|beta 1536+128+12 =
    1676, padded to 1680): fused() returns only the g|f_a|beta columns, so the
    caller's split [P, head_dim, heads] holds (TP8+DCP8 prefill crashed on the
    padded block before e1d3b62e0d)."""
    from types import SimpleNamespace

    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda_conv_in_proj

    part, hd, heads, pad, rows = 1536, 128, 12, 4, kda_conv_in_proj.MIN_ROWS
    md = SimpleNamespace(
        num_prefills=1, num_decodes=0, num_spec_decodes=0, checkpoint=None,
        non_spec_query_start_loc=torch.tensor([0, rows], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.zeros(1, dtype=torch.int32),
        has_initial_state=torch.zeros(1, dtype=torch.bool),
    )  # fmt: skip
    monkeypatch.setattr(
        kda_conv_in_proj,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"kda": md}),
    )
    rest_rows = part + hd + heads + pad
    monkeypatch.setattr(
        torch, "_scaled_mm",
        lambda a, b, **kw: torch.arange(a.shape[0] * b.shape[1], dtype=torch.float32)
        .view(a.shape[0], b.shape[1]),
    )  # fmt: skip
    conv = kda_conv_in_proj.PrefillKdaConvInProj.__new__(
        kda_conv_in_proj.PrefillKdaConvInProj
    )
    conv.layer = SimpleNamespace(
        prefix="kda", local_projection_size=part, head_dim=hd, local_num_heads=heads,
        kv_cache=[torch.zeros(1, 3 * part, 3, dtype=torch.float32)],
    )  # fmt: skip
    conv.p = part
    conv.rest_cols = part + hd + heads
    conv.w_rest = torch.zeros(rest_rows, 64)
    conv.s_rest = torch.zeros(rest_rows * 2, dtype=torch.uint8)
    conv._gemm = lambda *a, **kw: (torch.zeros(3, rows, part), None)
    conv._gemm.supports_checkpoint_halo = True
    qa = SimpleNamespace(
        data=torch.zeros(rows, 64), scale=torch.zeros(rows * 2, dtype=torch.uint8)
    )
    _, rest, _ = conv.fused(qa)
    assert rest.shape == (rows, part + hd + heads)
    g, f_a, beta = rest.split([part, hd, heads], dim=-1)
    assert beta.shape == (rows, heads)
    # the kept columns are the GEMM's first columns (padding dropped at the end)
    assert torch.equal(rest[1], torch.arange(rest_rows, 2 * rest_rows)[: rest.shape[1]].float())


def test_fused_in_proj_defers_to_eager_break(monkeypatch):
    """PF-P3: under a breakable (piecewise) capture of a MIN_ROWS batch the
    q|k|v rows go to the eager KDA break (defer_to_break); a plain capture,
    a FULL-mode breakable capture, smaller batches and eager steps do not
    defer (eager steps take fused() directly)."""
    from types import SimpleNamespace

    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.config import CUDAGraphMode
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda_conv_in_proj

    mode = {"m": CUDAGraphMode.PIECEWISE}
    monkeypatch.setattr(
        kda_conv_in_proj,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=mode["m"]),
    )
    monkeypatch.setattr(
        "vllm.forward_context.is_forward_context_available", lambda: True
    )
    fused = kda_conv_in_proj.PrefillKdaConvInProj.__new__(
        kda_conv_in_proj.PrefillKdaConvInProj
    )
    dev = "cuda"
    rows = kda_conv_in_proj.MIN_ROWS
    qa = SimpleNamespace(data=torch.zeros(rows, 128, device=dev))
    small = SimpleNamespace(data=torch.zeros(rows - 1, 128, device=dev))
    assert not fused.defer_to_break(qa)  # eager
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert not fused.defer_to_break(qa)  # plain (one-shot) capture
    seen = []
    side = torch.cuda.Stream()
    with torch.cuda.stream(side), BreakableCUDAGraphCapture():
        seen.append(fused.defer_to_break(qa))
        seen.append(fused.defer_to_break(small))
        mode["m"] = CUDAGraphMode.FULL
        seen.append(fused.defer_to_break(qa))
    assert seen == [True, False, False]
    monkeypatch.setattr(kda_conv_in_proj, "DEFER_TO_BREAK", False)
    mode["m"] = CUDAGraphMode.PIECEWISE
    with torch.cuda.stream(side), BreakableCUDAGraphCapture():
        seen.append(fused.defer_to_break(qa))
    assert seen[-1] is False
