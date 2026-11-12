# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused Q|K|V KDA prefill conv (CUDA, ops/kda_qkv_conv) against three stock
``causal_conv1d_fn`` calls and the deployed per-plane Triton kernel: outputs and
updated conv state bit for bit, for FP32 conv weights (the KDA layer's) with a bf16
conv state (deployed) or an FP32 one, and bf16 weights with a bf16 state."""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.models.kimi_k3.nvidia.ops.kda_conv import qkv_conv, qkv_conv_supported
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(100),
    reason="the fused KDA conv targets SM10x",
)


def _stock(x, weight, state, qsl, cache_indices, has_init, part):
    outs = []
    for i in range(3):
        sl = slice(i * part, (i + 1) * part)
        outs.append(
            causal_conv1d_fn(
                x[:, sl].transpose(0, 1),
                weight[sl],
                None,
                activation="silu",
                conv_states=state[:, sl],
                has_initial_state=has_init,
                cache_indices=cache_indices,
                query_start_loc=qsl,
            ).transpose(0, 1)
        )
    return outs


def _case(
    seqlens,
    part,
    state_dtype,
    dim_first,
    num_spec,
    scale=1.0,
    seed=0,
    w_dtype=torch.float32,
):
    torch.manual_seed(seed)
    device = "cuda"
    width, dim = 4, 3 * part
    state_len = width - 1 + num_spec
    num_tokens, num_seqs = sum(seqlens), len(seqlens)
    num_lines = 2 * num_seqs + 2
    # Row-strided Q|K|V inside a wider in_proj output (row stride a multiple of 8).
    proj = torch.randn(num_tokens, dim + 136, device=device) * scale
    proj = proj.to(torch.bfloat16)
    x = proj[:, :dim]
    weight = (torch.randn(dim, width, device=device) * 0.5).to(w_dtype)
    if dim_first:  # DS: (lines, dim, state) contiguous
        state = torch.randn(num_lines, dim, state_len, device=device)
    else:  # SD: (lines, state, dim) storage, viewed as (lines, dim, state)
        state = torch.randn(num_lines, state_len, dim, device=device).transpose(-1, -2)
    state = state.to(torch.bfloat16).to(state_dtype)
    qsl = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    qsl[1:] = torch.tensor(seqlens, device=device).cumsum(0)
    lines = torch.randperm(num_lines - 1, device=device)[:num_seqs] + 1
    if num_seqs > 2:
        lines[-1] = NULL_BLOCK_ID  # padding: neither output nor state is written
    cache_indices = lines.to(torch.int32)
    has_init = torch.rand(num_seqs, device=device) < 0.6
    if num_seqs > 1:
        has_init[0] = True
        has_init[1] = False
    return x, weight, state, qsl, cache_indices, has_init


def _check(x, weight, state, qsl, cache_indices, has_init, part):
    assert qkv_conv_supported(x, weight, state, part)
    ref_state = state.clone()
    ref = _stock(x, weight, ref_state, qsl, cache_indices, has_init, part)
    new_state = state.clone()
    out = qkv_conv(x, weight, new_state, qsl, cache_indices, has_init, part)
    assert out.shape == (3, x.size(0), part) and out.dtype == torch.bfloat16
    valid = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
    for i in range(cache_indices.numel()):
        if int(cache_indices[i]) != NULL_BLOCK_ID:
            valid[int(qsl[i]) : int(qsl[i + 1])] = True
    for got, want in zip(out.unbind(0), ref):
        torch.testing.assert_close(got[valid], want[valid], atol=0, rtol=0)
    torch.testing.assert_close(new_state, ref_state, atol=0, rtol=0)


SEQLENS = [
    [1],
    [2, 1, 3],
    [37],
    [5, 64, 1, 2, 33, 100],
    [4096, 3, 2047],
    [3480] * 5,
    [16384],
    [1] * 300 + [70] + [2] * 40 + [129],  # many sequences: the row split search
]


DTYPES = [  # (conv state, conv weight)
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.float32),
    (torch.bfloat16, torch.bfloat16),
]


@pytest.mark.parametrize("seqlens", SEQLENS)
@pytest.mark.parametrize("part", [256, 512, 1024])
@pytest.mark.parametrize("dtypes", DTYPES)
@pytest.mark.parametrize("dim_first", [False, True])
@torch.inference_mode()
def test_qkv_conv_matches_stock(seqlens, part, dtypes, dim_first):
    # The KDA conv state holds width - 1 + num_spec columns; prefill uses the first 3.
    state_dtype, w_dtype = dtypes
    case = _case(seqlens, part, state_dtype, dim_first, num_spec=3, w_dtype=w_dtype)
    _check(*case, part)


@pytest.mark.parametrize("dim_first", [False, True])
@torch.inference_mode()
def test_qkv_conv_matches_deployed_triton(dim_first):
    """The deployed path: per-plane _kda_conv_fwd_kernel, bf16 state, FP32 weights."""
    from vllm.models.kimi_k3.nvidia.ops import kda_conv
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    part = 512
    x, weight, state, qsl, ci, hi = _case(
        [3480] * 5, part, torch.bfloat16, dim_first, 3
    )
    nums_dict, batch_ptr, offs = compute_causal_conv1d_metadata(
        qsl.cpu(), device=x.device
    )

    class _M:
        pass

    m = _M()
    m.nums_dict, m.batch_ptr, m.token_chunk_offset_ptr = nums_dict, batch_ptr, offs
    ref_state = state.clone()
    ref = []
    for i in range(3):
        sl = slice(i * part, (i + 1) * part)
        xt = x[:, sl].transpose(0, 1)
        assert kda_conv.supported(xt, weight[sl], ref_state[:, sl], m)
        ref.append(
            kda_conv.kda_prefill_conv(
                xt, weight[sl], ref_state[:, sl], qsl, ci, hi, m
            ).transpose(0, 1)
        )
    new_state = state.clone()
    out = qkv_conv(x, weight, new_state, qsl, ci, hi, part)
    valid = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
    for i in range(ci.numel()):
        if int(ci[i]) != NULL_BLOCK_ID:
            valid[int(qsl[i]) : int(qsl[i + 1])] = True
    for got, want in zip(out.unbind(0), ref):
        torch.testing.assert_close(got[valid], want[valid], atol=0, rtol=0)
    torch.testing.assert_close(new_state, ref_state, atol=0, rtol=0)


@pytest.mark.parametrize("dtypes", DTYPES)
@torch.inference_mode()
def test_qkv_conv_silu_range(dtypes):
    """Large inputs: accumulators below -87 take div.full's range scaling, above
    +88 saturate; exact zeros keep the stock signed-zero result."""
    x, weight, state, qsl, ci, hi = _case(
        [500, 7, 1300],
        256,
        dtypes[0],
        True,
        num_spec=0,
        scale=60.0,
        seed=3,
        w_dtype=dtypes[1],
    )
    x[100:140] = 0
    _check(x, weight, state, qsl, ci, hi, 256)


@torch.inference_mode()
def test_qkv_conv_empty_sequences():
    """Zero-length sequences (padded query_start_loc entries) produce nothing and
    leave their conv state line untouched; the others match stock."""
    seqlens = [70, 0, 1, 0, 200, 0, 0]
    x, weight, state, _, _, _ = _case([sum(seqlens)], 256, torch.bfloat16, True, 0)
    device = x.device
    qsl = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    qsl[1:] = torch.tensor(seqlens, device=device).cumsum(0)
    state = torch.randn(len(seqlens) + 1, 768, 3, device=device).to(torch.bfloat16)
    cache_indices = torch.arange(1, len(seqlens) + 1, dtype=torch.int32, device=device)
    has_init = torch.ones(len(seqlens), dtype=torch.bool, device=device)
    new_state = state.clone()
    out = qkv_conv(x, weight, new_state, qsl, cache_indices, has_init, 256)
    keep = [i for i, n in enumerate(seqlens) if n > 0]
    ref_qsl = torch.zeros(len(keep) + 1, dtype=torch.int32, device=device)
    ref_qsl[1:] = torch.tensor([seqlens[i] for i in keep], device=device).cumsum(0)
    ref_state = state.clone()
    ref = _stock(
        x, weight, ref_state, ref_qsl, cache_indices[keep], has_init[keep], 256
    )
    for got, want in zip(out.unbind(0), ref):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    torch.testing.assert_close(new_state, ref_state, atol=0, rtol=0)


@torch.inference_mode()
def test_qkv_conv_cuda_graph_replay():
    """Captured once, replayed with other sequence lengths, initial states and
    inputs: the row split and sequence map are read on the device, and the input
    written by the kernel just before is seen."""
    part, num_tokens = 512, 3000
    x, weight, state, _, _, _ = _case([num_tokens], part, torch.bfloat16, True, 0)
    device = x.device
    seed = state.clone()
    src = x.clone()
    qsl = torch.tensor([0, 1000, 2999, 3000], dtype=torch.int32, device=device)
    cache_indices = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    has_init = torch.tensor([True, False, True], device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        x.copy_(src * 2)  # the producer kernel right before the conv
        out = qkv_conv(x, weight, state, qsl, cache_indices, has_init, part)
    for lens, init in (
        ([1000, 1999, 1], [True, False, True]),
        ([2, 2900, 98], [False, True, True]),
        ([1, 1, 2998], [True, True, False]),
    ):
        src.normal_()
        qsl[1:] = torch.tensor(lens, device=device).cumsum(0)
        has_init.copy_(torch.tensor(init, device=device))
        state.copy_(seed)
        graph.replay()
        ref_state = seed.clone()
        ref = _stock(x, weight, ref_state, qsl, cache_indices, has_init, part)
        for got, want in zip(out.unbind(0), ref):
            torch.testing.assert_close(got, want, atol=0, rtol=0)
        torch.testing.assert_close(state, ref_state, atol=0, rtol=0)


def test_qkv_conv_supported_shapes():
    x = torch.zeros(4, 3 * 256, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(3 * 256, 4, device="cuda", dtype=torch.bfloat16)
    state = torch.zeros(2, 3 * 256, 3, device="cuda", dtype=torch.bfloat16)
    assert qkv_conv_supported(x, w, state, 256)
    assert qkv_conv_supported(x, w.float(), state, 256)
    assert qkv_conv_supported(x, w.float(), state.float(), 256)
    # Capability (coverage) misses: logged once, the caller keeps Triton.
    # bf16 weights with an FP32 state are not covered.
    assert not qkv_conv_supported(x, w, state.float(), 256)
    # Channels per plane must be a multiple of 256.
    x384 = torch.zeros(4, 3 * 384, device="cuda", dtype=torch.bfloat16)
    w384 = torch.zeros(3 * 384, 4, device="cuda", dtype=torch.bfloat16)
    s384 = torch.zeros(2, 3 * 384, 3, device="cuda", dtype=torch.bfloat16)
    assert not qkv_conv_supported(x384, w384, s384, 384)
    # CPU tensors never take the CUDA kernel.
    assert not qkv_conv_supported(x.cpu(), w.cpu(), state.cpu(), 256)
    # Layout violations inside the covered dtypes are caller bugs: they raise
    # instead of silently dropping to the slow path.
    wide = torch.zeros(4, 3 * 256 + 4, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="16 B aligned"):
        qkv_conv_supported(wide[:, : 3 * 256], w, state, 256)
    with pytest.raises(ValueError, match="conv state shape"):
        qkv_conv_supported(x, w, state[..., :2], 256)
    with pytest.raises(ValueError, match="weight not contiguous"):
        qkv_conv_supported(x, w.t().contiguous().t(), state, 256)
