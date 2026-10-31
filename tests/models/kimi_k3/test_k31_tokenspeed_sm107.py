# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TokenSpeed SM107 M256 FP8 MLA decode with a BF16 output (the TOKENSPEED_MLA
SM107 path, used by the K3.1 DSpark drafter) vs an FP32 reference and vs the
SM100 FP8 kernel the backend used before."""

import math

import pytest
import torch

D_LAT, D_ROPE, PAGE = 512, 64, 64

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="SM107 only",
)


def _rand_fp8(shape, g):
    b = torch.randint(0x18, 0x38, shape, dtype=torch.uint8, device="cuda", generator=g)
    b |= torch.randint(0, 2, shape, dtype=torch.uint8, device="cuda", generator=g) << 7
    return b.view(torch.float8_e4m3fn)


def _inputs(ctxs, qlen, heads, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = [math.ceil(c / PAGE) for c in ctxs]
    kv = _rand_fp8((sum(pages) + 3, PAGE, D_LAT + D_ROPE), g)
    bt = torch.zeros(len(ctxs), max(pages), dtype=torch.int32, device="cuda")
    perm = torch.randperm(sum(pages) + 3, device="cuda", generator=g).to(torch.int32)
    off = 0
    for i, p in enumerate(pages):
        bt[i, :p] = perm[off : off + p]
        off += p
    q = _rand_fp8((len(ctxs), qlen, heads, D_LAT + D_ROPE), g)
    return q, kv, bt, torch.tensor(ctxs, dtype=torch.int32, device="cuda")


def _reference(q, kv, bt, seq, scale, causal, out_scale=1.0):
    outs = []
    for i in range(q.shape[0]):
        n = int(seq[i])
        k = kv[bt[i, : -(-n // PAGE)].long()].reshape(-1, D_LAT + D_ROPE)[:n].float()
        qi = q[i].float()  # [qlen, H, 576]
        s = torch.einsum("thd,nd->thn", qi, k) * scale
        if causal:
            qlen = q.shape[1]
            pos = torch.arange(n, device="cuda")
            limit = n - qlen + torch.arange(qlen, device="cuda")
            s = s.masked_fill(pos[None, None, :] > limit[:, None, None], float("-inf"))
        outs.append(
            torch.einsum("thn,nd->thd", torch.softmax(s, -1), k[:, :D_LAT]) * out_scale
        )
        del s
    return torch.stack(outs)


def _forward_sm107(q, kv, bt, seq, scale, out_scale, causal, min_split, query_len=8):
    """``TokenspeedMLAImpl._forward_sm107`` itself (the backend's call, including
    the installed TokenSpeed's extra keyword arguments) on a stand-in impl."""
    from types import SimpleNamespace

    from vllm.v1.attention.backends.mla.tokenspeed_mla import (
        TokenspeedMLAImpl,
        _sm107_decode_extra_kwargs,
    )

    assert _sm107_decode_extra_kwargs() is not None
    impl = SimpleNamespace(
        kv_lora_rank=D_LAT, _sm107_workspace=1 << 30, _sm107_split=min_split,
        softmax_scale=scale, output_scale=out_scale, _query_len=query_len,
    )  # fmt: skip
    meta = SimpleNamespace(
        decode=SimpleNamespace(block_table=bt, seq_lens=seq),
        max_seq_len=int(seq.max()), causal=causal,
    )  # fmt: skip
    out = TokenspeedMLAImpl._forward_sm107(impl, q, kv, meta)
    return out.view(q.shape[0], q.shape[1], q.shape[2], D_LAT)


def _check(heads, qlen, causal, min_split, scale, out_scale, ctxs, seed):
    from tokenspeed_mla import tokenspeed_mla_decode

    q, kv, bt, seq = _inputs(ctxs, qlen, heads, seed=seed)
    out = _forward_sm107(q, kv, bt, seq, scale, out_scale, causal, min_split)
    ws = torch.empty(1 << 30, dtype=torch.int8, device="cuda")
    stock = tokenspeed_mla_decode(
        query=q, kv_cache=kv, workspace_buffer=ws, kv_lora_rank=D_LAT,
        qk_rope_head_dim=D_ROPE, block_tables=bt, seq_lens=seq, max_seq_len=max(ctxs),
        softmax_scale=scale, output_scale=out_scale, causal_mask=causal,
    )  # fmt: skip
    for i in range(len(ctxs)):
        r = _reference(
            q[i : i + 1], kv, bt[i : i + 1], seq[i : i + 1], scale, causal, out_scale
        )[0]
        assert torch.isfinite(out[i].float()).all(), i
        err_new = (
            (out[i].float() - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()
        ).item()
        err_old = (
            (stock[i].float() - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()
        ).item()
        # FP8 P (both kernels): a few % rms on these random inputs; the SM107
        # kernel must be no worse than the SM100 one it replaces.
        assert err_new < 0.05, (i, err_new)
        assert err_new <= err_old * 1.1 + 1e-3, (i, err_new, err_old)


# Contexts up to 200K (the r28 owner shapes reach ~660K tokens per rank in total).
CTXS = [37, 129, 4096 + 5, 70001, 200003]


@pytest.mark.parametrize("heads,qlen", [(64, 8), (64, 7), (64, 4), (96, 8)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("min_split", [1, 32])
def test_sm107_bf16_out_matches_reference(
    workspace_init, heads, qlen, causal, min_split
):
    """q_len 7 = the DSpark drafter's query_len - 1 blocks (query_len 8)."""
    _check(heads, qlen, causal, min_split, 0.05, 1.0, CTXS, seed=heads + qlen)


@pytest.mark.parametrize("out_scale", [0.37, 2.5])
@pytest.mark.parametrize("scale", [0.05, 0.8 / math.sqrt(192)])
@pytest.mark.parametrize("qlen", [7, 8])
def test_sm107_output_and_softmax_scale(workspace_init, out_scale, scale, qlen):
    """Production passes output_scale = k_scale (both kernels apply it in the
    split-KV partial epilogue)."""
    _check(64, qlen, True, 32, scale, out_scale, CTXS, seed=7 + qlen)


def test_signature_check_disables_unknown(monkeypatch):
    """An installed TokenSpeed whose private SM107 helper needs an argument this
    path does not pass disables the path instead of failing at the first decode;
    2.0.2's required ``enable_packed_q`` is passed as True."""
    from tokenspeed_mla import mla_decode as md

    from vllm.v1.attention.backends.mla import tokenspeed_mla as ts

    def needs_more(*, q_latent, new_required_arg, **kw):  # noqa: ARG001
        raise AssertionError

    def packed(*, q_latent, q_rope, c_latent, c_rope, block_tables, seq_lens,
               workspace_buffer, out, page_size, max_seq_len, softmax_scale,
               output_scale, is_var_seq, causal_mask, window_left, enable_pdl,
               return_lse, causal_seqs, cp_world, cp_rank, local_visible_lens,
               min_split_kv, enable_packed_q, compute_capability):  # fmt: skip
        raise AssertionError

    ts._sm107_decode_extra_kwargs.cache_clear()
    try:
        monkeypatch.setattr(md, "_sm107_fp8_out_decode", needs_more)
        assert ts._sm107_decode_extra_kwargs() is None
        ts._sm107_decode_extra_kwargs.cache_clear()
        monkeypatch.setattr(md, "_sm107_fp8_out_decode", packed)
        assert ts._sm107_decode_extra_kwargs() == {"enable_packed_q": True}
    finally:
        monkeypatch.undo()
        ts._sm107_decode_extra_kwargs.cache_clear()
    assert ts._sm107_decode_extra_kwargs() is not None


def test_workspace_bound_covers_batches():
    from tokenspeed_mla import mla_decode as md

    from vllm.v1.attention.backends.mla.tokenspeed_mla import _sm107_workspace_bound

    bound = _sm107_workspace_bound(32, 2, D_LAT, 1 << 20, 32)
    assert bound > 0
    clusters = md.get_max_active_clusters(4) * 4
    for b in (1, 7, 32):
        for n in (100, 70000, 1 << 20):
            _, size = md._get_sm107_split_kv_and_workspace_size(
                b, 2, 256, D_LAT, clusters, n, 32
            )
            assert size <= bound


def test_warmup_covers_reachable_keys(workspace_init):
    """After sm107_warmup, decodes within its batch / length bounds compile nothing."""
    from types import SimpleNamespace

    from tokenspeed_mla import mla_decode as md

    from vllm.v1.attention.backends.mla.tokenspeed_mla import sm107_warmup

    impl = SimpleNamespace(
        _sm107_split=32, num_heads=64, kv_lora_rank=D_LAT, _query_len=8
    )
    n = sm107_warmup(impl, PAGE, {7, 8}, (False,), 4, 9000)
    assert n > 0
    misses = md._get_compiled_sm107_mla_kernel.cache_info().misses
    for ctxs in ([5], [300, 9000], [8191, 1, 4096, 129]):
        for qlen in (7, 8):
            q, kv, bt, seq = _inputs(ctxs, qlen, 64, seed=len(ctxs))
            _forward_sm107(q, kv, bt, seq, 0.05, 1.0, False, 32)
    torch.cuda.synchronize()
    assert md._get_compiled_sm107_mla_kernel.cache_info().misses == misses


@pytest.mark.parametrize("heads", [4, 16])
def test_partial_tile_shapes_not_taken(heads):
    """num_heads * q_len < 256 (a mostly padded M256 tile, slower than the SM100
    path) is not an SM107 shape: nothing to warm up for it."""
    from types import SimpleNamespace

    from vllm.v1.attention.backends.mla.tokenspeed_mla import sm107_warmup

    impl = SimpleNamespace(
        _sm107_split=32, num_heads=heads, kv_lora_rank=D_LAT, _query_len=8
    )
    assert sm107_warmup(impl, PAGE, {7, 8}, (False,), 4, 9000) == 0
