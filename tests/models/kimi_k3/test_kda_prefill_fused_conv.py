# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KDA prefill with the short conv fused into FlashKDA's prepare kernel
(specialized_prefill/kda.py) against the stock chain it replaces: the fused
Q|K|V conv (ops/kda_qkv_conv) followed by FlashKDA. Outputs, final recurrent
state and updated conv state must match bit for bit."""

import pytest
import torch

from vllm.models.kimi_k3.nvidia.ops import kda_segscan
from vllm.models.kimi_k3.nvidia.ops.kda_conv import qkv_conv
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(100),
    reason="FlashKDA split build targets SM10x",
)

D = 128


def _inputs(seqlens, heads, conv_state_dtype, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    T, N, P = sum(seqlens), len(seqlens), heads * D
    proj = (torch.randn(T, 4 * P + 136, device=dev) * 0.7).to(torch.bfloat16)
    mixed = proj[:, : 3 * P]
    beta = proj[:, 3 * P : 3 * P + heads].unsqueeze(0)
    g = (torch.randn(1, T, heads, D, device=dev) * 0.5).to(torch.bfloat16)
    w = torch.randn(3 * P, 4, device=dev) * 0.4
    lines = 2 * N + 2
    conv_state = (
        (torch.randn(lines, 3 * P, 3, device=dev) * 0.7)
        .to(torch.bfloat16)
        .to(conv_state_dtype)
    )
    cache_indices = (torch.randperm(lines - 1, device=dev)[:N] + 1).to(torch.int32)
    has_init = torch.rand(N, device=dev) < 0.6
    has_init[0] = True
    if N > 1:
        has_init[1] = False
    cu = torch.zeros(N + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.tensor(seqlens, device=dev).cumsum(0)
    A_log = torch.randn(heads, device=dev) * 0.5
    dt_bias = torch.randn(heads * D, device=dev) * 0.1
    init = (torch.randn(N, heads, D, D, device=dev) * 0.05) * has_init[
        :, None, None, None
    ]
    return (
        mixed,
        beta,
        g,
        w,
        conv_state,
        cache_indices,
        has_init,
        cu,
        A_log,
        dt_bias,
        init,
    )


@pytest.mark.parametrize(
    "seqlens,heads",
    [
        ([37, 1, 2, 200, 16], 4),
        ([5461, 5461, 5462], 8),
        ([3, 64, 1000], 96),
    ],
)
@pytest.mark.parametrize("conv_state_dtype", [torch.bfloat16, torch.float32])
def test_fused_conv_prefill_bitwise(seqlens, heads, conv_state_dtype):
    from vllm.models.kimi_k3.nvidia.specialized_prefill.kda import (
        fused_conv_prefill,
    )

    kda_segscan.load_extension()
    (mixed, beta, g, w, cs, ci, hi, cu, A_log, dt_bias, init) = _inputs(
        seqlens, heads, conv_state_dtype
    )
    T, N, P = sum(seqlens), len(seqlens), heads * D
    lb = -5.0
    ws = torch.empty(
        kda_segscan.workspace_size(T, heads, N), dtype=torch.uint8, device="cuda"
    )

    # stock chain: fused Q|K|V conv, then FlashKDA (same build, unfused K1)
    cs_ref = cs.clone()
    q, k, v = (
        x.view(1, T, heads, D) for x in qkv_conv(mixed, w, cs_ref, cu, ci, hi, P)
    )
    out_ref = torch.empty(1, T, heads, D, dtype=torch.bfloat16, device="cuda")
    fs_ref = torch.empty_like(init)
    torch.ops.k3_flashkda_split.fwd(
        q,
        k,
        v,
        g,
        beta,
        D**-0.5,
        out_ref,
        ws,
        A_log,
        dt_bias.view(heads, D),
        lb,
        init,
        fs_ref,
        cu,
        None,
        None,
        3,
    )

    cs_new = cs.clone()
    out = torch.empty_like(out_ref)
    fs = torch.empty_like(init)
    v_buf = torch.empty_like(out_ref)
    fused_conv_prefill(
        mixed_qkv=mixed,
        conv_weight=w,
        conv_state=cs_new,
        cache_indices=ci,
        has_initial_state=hi,
        null_block_id=-1,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lb,
        initial_state=init,
        cu_seqlens=cu,
        out=out,
        final_state=fs,
        workspace=ws,
        v_buf=v_buf,
    )
    torch.accelerator.synchronize()
    assert torch.equal(v_buf, v)
    assert torch.equal(cs_new, cs_ref)
    assert torch.equal(fs, fs_ref)
    assert torch.equal(out, out_ref)


@pytest.mark.parametrize("tokens,heads", [(1, 4), (300, 8), (2051, 96)])
def test_o_norm_mxfp8(tokens, heads):
    """specialized_prefill.kda.o_norm_mxfp8 against kda_norm.rmsnorm_gated_mxfp8:
    same FP8 codes and scales up to rare one-step flips (the sum of squares
    runs in another order)."""
    from vllm.models.kimi_k3.nvidia.mxfp8 import unswizzle_mxfp8_scale
    from vllm.models.kimi_k3.nvidia.ops import kda_norm
    from vllm.models.kimi_k3.nvidia.ops.mxfp8_act import Mxfp8Buffer
    from vllm.models.kimi_k3.nvidia.specialized_prefill.kda import o_norm_mxfp8

    torch.manual_seed(0)
    T, P = tokens, heads * D
    x = (torch.randn(1, T, heads, D, device="cuda") * 3).to(torch.bfloat16)
    x[0, 0, 0] = 0  # an all-zero row
    gate = (torch.randn(T, P + 64, device="cuda") * 2).to(torch.bfloat16)[:, :P]
    w = torch.rand(D, device="cuda") + 0.5
    ref = Mxfp8Buffer(T, P, "cuda", torch.bfloat16)
    kda_norm.rmsnorm_gated_mxfp8(x, gate.view(T, heads, D), w, 1e-6, out=ref)
    out = Mxfp8Buffer(T, P, "cuda", torch.bfloat16)
    o_norm_mxfp8(x, gate, w, 1e-6, out)
    torch.accelerator.synchronize()
    rows = ((T + 127) // 128) * 128
    su = unswizzle_mxfp8_scale(out.scale, rows, P)[:T]
    su_ref = unswizzle_mxfp8_scale(ref.scale, rows, P)[:T]
    assert (su != su_ref).float().mean().item() <= 1e-3
    a = out.data.view(torch.uint8).to(torch.int16)
    b = ref.data.view(torch.uint8).to(torch.int16)
    assert ((a - b).abs() > 1).float().mean().item() <= 1e-4
    assert (a != b).float().mean().item() <= 2e-3
