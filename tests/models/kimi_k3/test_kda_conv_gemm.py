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


def _run(
    seqlens, m, deep=False, tactic=None, seed=0, state_dtype=torch.float32,
    normalize_qk=False,
):  # fmt: skip
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
    out, _ = gemm(
        xq, xs, st, qsl, lines, has_init, null_block_id=NULL_BLOCK_ID,
        normalize_qk=normalize_qk,
    )  # fmt: skip
    torch.cuda.synchronize()
    tot = int(qsl[-1])
    # qkv_conv leaves the rows of null-block sequences (padding) unwritten.
    keep = torch.ones(tot, dtype=torch.bool, device=dev)
    for i in (lines == NULL_BLOCK_ID).nonzero().flatten().tolist():
        keep[int(qsl[i]) : int(qsl[i + 1])] = False
    if normalize_qk:
        # Stock l2norm_fwd of the reference Q / K (FP32 sum of squares, then
        # y * rsqrt(s + eps) in BF16): equal up to the reduction order, so at
        # most one BF16 ulp apart; V and the conv state stay bitwise.
        rh = ref[:2].float().unflatten(-1, (PART // 128, 128))
        inv = torch.rsqrt(rh.square().sum(-1, keepdim=True) + 1e-6)
        rn = (rh * inv).flatten(-2).to(torch.bfloat16)[:, :tot][:, keep]
        got = out[:2, :tot][:, keep]
        ulp = (got.view(torch.int16).int() - rn.view(torch.int16).int()).abs()
        assert int(ulp.max()) <= 1
        assert float((ulp > 0).float().mean()) < 0.01
        assert torch.equal(out[2, :tot][keep], ref[2, :tot][keep])
        assert torch.equal(st, st_ref)
        return
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


@pytest.mark.parametrize(
    "seqlens,m,deep",
    [([1, 2, 3, 125, 128, 3, 256, 600, 5], 1300, False), ([900, 513, 7], 1420, True)],
)
def test_kda_conv_gemm_normalize_qk(seqlens, m, deep):
    # Q / K planes as the persistent prefill core consumes them (QKN route).
    _run(seqlens, m, deep=deep, seed=3 if deep else 0, normalize_qk=True)


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
