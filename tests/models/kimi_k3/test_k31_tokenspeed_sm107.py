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


def _reference(q, kv, bt, seq, scale, causal):
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
        outs.append(torch.einsum("thn,nd->thd", torch.softmax(s, -1), k[:, :D_LAT]))
    return torch.stack(outs)


@pytest.mark.parametrize("heads,qlen", [(64, 8), (64, 4), (96, 8)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("min_split", [1, 32])
def test_sm107_bf16_out_matches_reference(heads, qlen, causal, min_split):
    from tokenspeed_mla import mla_decode as md
    from tokenspeed_mla import tokenspeed_mla_decode

    ctxs = [37, 129, 4096 + 5, 20000, 70001]
    q, kv, bt, seq = _inputs(ctxs, qlen, heads, seed=heads + qlen)
    scale = 0.05
    ws = torch.empty(1 << 30, dtype=torch.int8, device="cuda")
    out = torch.empty(len(ctxs), qlen, heads, D_LAT, dtype=torch.bfloat16, device="cuda")
    md._sm107_fp8_out_decode(
        q_latent=q[..., :D_LAT], q_rope=q[..., D_LAT:], c_latent=kv[..., :D_LAT],
        c_rope=kv[..., D_LAT:], block_tables=bt, seq_lens=seq, workspace_buffer=ws,
        out=out, page_size=PAGE, max_seq_len=max(ctxs), softmax_scale=scale,
        output_scale=1.0, is_var_seq=True, causal_mask=causal, window_left=-1,
        enable_pdl=False, return_lse=False, causal_seqs=None, cp_world=1, cp_rank=0,
        local_visible_lens=None, min_split_kv=min_split, compute_capability=(10, 7),
    )
    stock = tokenspeed_mla_decode(
        query=q, kv_cache=kv, workspace_buffer=ws, kv_lora_rank=D_LAT,
        qk_rope_head_dim=D_ROPE, block_tables=bt, seq_lens=seq, max_seq_len=max(ctxs),
        softmax_scale=scale, output_scale=1.0, causal_mask=causal,
    )
    ref = _reference(q, kv, bt, seq, scale, causal)
    for i in range(len(ctxs)):
        r = ref[i]
        err_new = ((out[i].float() - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item()
        err_old = ((stock[i].float() - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item()
        # FP8 P (both kernels): a few % rms on these random inputs; the SM107
        # kernel must be no worse than the SM100 one it replaces.
        assert err_new < 0.05, (i, err_new)
        assert err_new <= err_old * 1.1 + 1e-3, (i, err_new, err_old)


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


def test_warmup_covers_reachable_keys():
    """After sm107_warmup, decodes within its batch / length bounds compile nothing."""
    from types import SimpleNamespace

    from tokenspeed_mla import mla_decode as md

    from vllm.v1.attention.backends.mla.tokenspeed_mla import sm107_warmup

    impl = SimpleNamespace(_sm107_split=32, num_heads=64, kv_lora_rank=D_LAT, _query_len=8)
    n = sm107_warmup(impl, PAGE, {7, 8}, (False,), 4, 9000)
    assert n > 0
    misses = md._get_compiled_sm107_mla_kernel.cache_info().misses
    ws = torch.empty(1 << 30, dtype=torch.int8, device="cuda")
    for ctxs in ([5], [300, 9000], [8191, 1, 4096, 129]):
        for qlen in (7, 8):
            q, kv, bt, seq = _inputs(ctxs, qlen, 64, seed=len(ctxs))
            out = torch.empty(len(ctxs), qlen, 64, D_LAT, dtype=torch.bfloat16, device="cuda")
            md._sm107_fp8_out_decode(
                q_latent=q[..., :D_LAT], q_rope=q[..., D_LAT:], c_latent=kv[..., :D_LAT],
                c_rope=kv[..., D_LAT:], block_tables=bt, seq_lens=seq, workspace_buffer=ws,
                out=out, page_size=PAGE, max_seq_len=max(ctxs), softmax_scale=0.05,
                output_scale=1.0, is_var_seq=True, causal_mask=False, window_left=-1,
                enable_pdl=False, return_lse=False, causal_seqs=None, cp_world=1, cp_rank=0,
                local_visible_lens=None, min_split_kv=32, compute_capability=(10, 7),
            )
    torch.cuda.synchronize()
    assert md._get_compiled_sm107_mla_kernel.cache_info().misses == misses
