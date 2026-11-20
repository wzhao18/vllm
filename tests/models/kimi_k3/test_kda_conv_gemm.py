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


def _run(seqlens, m, deep=False, tactic=None, seed=0, state_dtype=torch.float32):
    from vllm.models.kimi_k3.nvidia.ops import kda_conv
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        KdaConvGemm,
    )

    dev = torch.device("cuda")
    n = 3 * PART
    xq, xs, wq, ws = _operands(m, n, K, dev, seed)
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
    ref = kda_conv.qkv_conv(proj, cw, st_ref, qsl, lines, has_init, PART)
    gemm = KdaConvGemm(wq, ws, cw, PART, tactic=tactic)
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


def _run_checkpoint(
    seqlens, offsets, m, tactic=None, seed=0, host_offsets=True, part=PART
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
    xq, xs, wq, ws = _operands(m, n, K, dev, seed)
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
    slots = torch.randperm(3 * nseq + 2, device=dev, generator=g).to(torch.int32)
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
    # The deployed fused in_proj tactic (2x4 cluster, MIN_ROWS and up).
    from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_conv_in_proj import TACTIC

    _run_checkpoint([2000, 1000, 5000, 192, 8000], [1984, 1000, 4096, 176, 7680],
                    16384, tactic=TACTIC, seed=7, part=1024)  # fmt: skip
