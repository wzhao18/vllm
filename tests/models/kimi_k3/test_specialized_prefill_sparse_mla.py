# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 prefill sparse MLA (specialized_prefill, sm_107) against an FP32
reference over the same FP8 query / cache (the kernel's only rounding beyond the
reference is the stock E4M3 P), and against the stock trtllm-gen kernel."""

import math

import pytest
import torch

from vllm.platforms import current_platform

if not (current_platform.is_cuda() and current_platform.has_device_capability(107)):
    pytest.skip("needs sm_107 (Rubin)", allow_module_level=True)

from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla import (  # noqa: E402
    sparse_mla_prefill,
    sparse_mla_prefill_v1,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h2 import (  # noqa: E402
    sparse_mla_prefill_h2,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h4 import (  # noqa: E402
    sparse_mla_prefill_h4,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h5 import (  # noqa: E402
    sparse_mla_prefill_h5,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h6 import (  # noqa: E402
    sparse_mla_prefill_h6,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h7 import (  # noqa: E402
    sparse_mla_prefill_h7,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import (  # noqa: E402
    sparse_mla_prefill_h8,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_sol import (  # noqa: E402
    sparse_mla_prefill_sol,
)

H, ROW, L = 96, 576, 512
E4M3 = torch.float8_e4m3fn


def _problem(lengths, pool_pages, width, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    t = len(lengths)
    kv = (torch.randn(pool_pages * 64, ROW, device="cuda", generator=g) * 0.8).to(E4M3)
    q8 = (torch.randn(t, H, ROW, device="cuda", generator=g) * 0.8).to(E4M3)
    slots = torch.full((t, width), -1, dtype=torch.int32, device="cuda")
    for i, n in enumerate(lengths):
        if n:
            sel = torch.randperm(pool_pages * 64, device="cuda", generator=g)[:n]
            slots[i, :n] = sel.to(torch.int32)
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    return q8, kv, slots, lens


def _reference(q8, kv, slots, lens, scale, out_scale):
    t = q8.shape[0]
    out = torch.zeros(t, H, L, device="cuda")
    kvf = kv.float()
    for i in range(t):
        n = int(lens[i])
        if n == 0:
            continue
        k = kvf[slots[i, :n].long()]  # [n, 576]
        s = q8[i].float() @ k.T * scale  # [96, n]
        p = torch.softmax(s, dim=-1)
        out[i] = p @ k[:, :L] * out_scale
    return out


@pytest.mark.parametrize(
    "loaders,pipelined,tma,regs,stages",
    [(3, False, False, (), 2), (3, True, False, (), 2), (7, False, True, (), 2),
     (7, True, True, (184, 72), 2), (7, True, True, (184, 72), 3)],
)
def test_matches_fp32_reference(loaders, pipelined, tma, regs, stages):
    lengths = [1, 2, 127, 128, 129, 300, 1000, 2047, 2048, 2048, 64, 255, 256, 2000]
    lengths = lengths * 23  # more rows than SMs: persistent CTAs take several
    q8, kv, slots, lens = _problem(lengths, pool_pages=600, width=2048, seed=0)
    scale, out_scale = 192**-0.5 * 0.7, 1.3
    out = sparse_mla_prefill_v1(q8, kv, 64, slots, lens, scale, out_scale,
                             loaders=loaders, pipelined=pipelined, tma=tma,
                             regs=regs, stages=stages)
    torch.cuda.synchronize()
    ref = _reference(q8, kv, slots, lens, scale, out_scale)
    err = (out.float() - ref).abs()
    rel = err.norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-6)
    assert torch.isfinite(out.float()).all()
    # E4M3 P (as the stock kernel) bounds the error; measured ~0.5-1% mean
    assert rel.mean().item() < 0.02, rel.mean().item()
    assert rel.max().item() < 0.08, rel.max().item()


def test_matches_stock_trtllm_kernel():
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    lengths = [2048] * 200 + [517] * 40
    q8, kv, slots, lens = _problem(lengths, pool_pages=1500, width=2048, seed=1)
    scale, out_scale = 192**-0.5, 1.0
    out = sparse_mla_prefill(q8, kv, 64, slots, lens, scale, out_scale)
    sol = sparse_mla_prefill_sol(q8, kv, slots, lens, scale, out_scale)
    h2 = sparse_mla_prefill_h2(q8, kv, slots, lens, scale, out_scale)
    t = q8.shape[0]
    qp = torch.zeros(t, 128, ROW, dtype=E4M3, device="cuda")
    qp[:, :H] = q8
    ref = torch.empty(t, 128, L, dtype=torch.bfloat16, device="cuda")
    ws = torch.zeros(1 << 28, dtype=torch.int8, device="cuda")
    trtllm_batch_decode_with_kv_cache_mla(
        query=qp.unsqueeze(1), kv_cache=kv.view(-1, 64, ROW).unsqueeze(1),
        workspace_buffer=ws, qk_nope_head_dim=128, kv_lora_rank=L, qk_rope_head_dim=64,
        block_tables=slots.unsqueeze(1), seq_lens=lens, max_seq_len=2048,
        out=ref.unsqueeze(1), bmm1_scale=scale, bmm2_scale=out_scale,
        sparse_mla_top_k=2048,
    )
    torch.cuda.synchronize()
    # both round P to E4M3 (differently scaled): ours must be no less precise
    exact = _reference(q8, kv, slots, lens, scale, out_scale)

    def err(x):
        d = (x.float() - exact).norm(dim=-1) / exact.norm(dim=-1).clamp(min=1e-6)
        return d.mean().item()

    ours, stock, ours_sol = err(out), err(ref[:, :H]), err(sol)
    assert not math.isnan(ours) and not math.isnan(ours_sol)
    assert ours <= 1.25 * stock + 1e-3, (ours, stock)
    assert ours_sol <= 1.25 * stock + 1e-3, (ours_sol, stock)
    ours_h2 = err(h2)
    assert ours_h2 <= 1.25 * stock + 1e-3, (ours_h2, stock)
    ours_h4 = err(sparse_mla_prefill_h4(q8, kv, slots, lens, scale, out_scale))
    assert ours_h4 <= 1.25 * stock + 1e-3, (ours_h4, stock)
    ours_h5 = err(sparse_mla_prefill_h5(q8, kv, slots, lens, scale, out_scale))
    assert ours_h5 <= 1.25 * stock + 1e-3, (ours_h5, stock)
    ours_h6 = err(sparse_mla_prefill_h6(q8, kv, slots, lens, scale, out_scale))
    assert ours_h6 <= 1.25 * stock + 1e-3, (ours_h6, stock)


@pytest.mark.parametrize("kernel", ["sol", "h2", "h4", "h4_tma", "h5", "h5_tma", "h5_split", "h5_stm", "h6", "h7", "h8", "h8_qkh"])
@pytest.mark.parametrize("scale_mul", [0.7, 3.0])
def test_sol_matches_fp32_reference(scale_mul, kernel):
    """Gather-floor kernel; scale_mul 3 drives the lazy-max slow path often."""
    lengths = [1, 2, 127, 128, 129, 300, 1000, 2047, 2048, 2048, 64, 255, 256, 2000, 0]
    lengths = lengths * 23  # more rows than SMs: persistent CTAs take several
    q8, kv, slots, lens = _problem(lengths, pool_pages=600, width=2048, seed=0)
    scale, out_scale = 192**-0.5 * scale_mul, 1.3
    fn = {"sol": sparse_mla_prefill_sol, "h2": sparse_mla_prefill_h2,
          "h4": sparse_mla_prefill_h4, "h5": sparse_mla_prefill_h5,
          "h6": sparse_mla_prefill_h6, "h7": sparse_mla_prefill_h7,
          "h8": sparse_mla_prefill_h8}[kernel.split("_")[0]]
    knobs = {"tma": (0, 0, 1, 1), "split": (0, 0, 0, 0, 0, 0, 1),
             "stm": (0, 0, 2, 0, 0, 0, 1), "pp": (0, 0, 2, 0, 0, 0, 1, 1),
             "pp2": (0, 0, 2, 0, 0, 0, 1, 1, 0, 1),
             "qkh": (0,) * 11 + (1,)}.get(
        kernel.split("_")[-1], ())
    out = (fn(q8, kv, slots, lens, scale, out_scale, knobs=knobs) if knobs
           else fn(q8, kv, slots, lens, scale, out_scale))
    torch.cuda.synchronize()
    ref = _reference(q8, kv, slots, lens, scale, out_scale)
    err = (out.float() - ref).abs()
    rel = err.norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-6)
    assert torch.isfinite(out.float()).all()
    zero = lens == 0
    assert (out[zero].float() == 0).all()
    rel = rel[~zero]
    assert rel.mean().item() < 0.02, rel.mean().item()
    assert rel.max().item() < 0.08, rel.max().item()


@pytest.mark.parametrize("block_size", [64, 32])
def test_h8_logical_topk_matches_physical(block_size):
    """The fused index remap (request-relative top-k through the block table) gathers
    the rows the stock remap would have handed the physical-slot kernel: same output
    bit for bit, including tokens with fewer valid entries than the top-k width."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import (
        sparse_mla_prefill_h8_logical,
    )

    g = torch.Generator(device="cuda").manual_seed(1)
    stride, blocks, reqs, width = 64, 900, 3, 2048
    ctx_blocks = [400, 120, 300]  # 120 x 32 < 2048: short request rows end in -1
    kv = (torch.randn(blocks * stride, ROW, device="cuda", generator=g) * 0.8).to(E4M3)
    bt = torch.full((reqs, 512), 0, dtype=torch.int32, device="cuda")
    perm = torch.randperm(blocks, device="cuda", generator=g).to(torch.int32)
    start = 0
    for r, nb in enumerate(ctx_blocks):
        bt[r, :nb] = perm[start:start + nb]
        start += nb
    lengths = [2048, 1, 2047, 129, 0, 2048, 64, 1000] * 40
    t = len(lengths)
    req_ids = torch.arange(t, device="cuda", dtype=torch.int32) % reqs
    topk = torch.full((t, width), -1, dtype=torch.int32, device="cuda")
    for i, n in enumerate(lengths):
        ctx = ctx_blocks[i % reqs] * block_size
        n = min(n, ctx)
        lengths[i] = n
        if n:
            topk[i, :n] = torch.randperm(ctx, device="cuda", generator=g)[:n].to(torch.int32)
    q8 = (torch.randn(t, H, ROW, device="cuda", generator=g) * 0.8).to(E4M3)
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    valid = topk >= 0
    blk = bt[req_ids.long()[:, None], (topk.clamp(min=0) // block_size).long()]
    slots = torch.where(valid, blk * stride + topk.clamp(min=0) % block_size, -1)
    scale, out_scale = 192**-0.5 * 3.0, 1.3
    ref = sparse_mla_prefill_h8(q8, kv, slots.to(torch.int32), lens, scale, out_scale)
    out = sparse_mla_prefill_h8_logical(q8, kv, topk, req_ids, bt, block_size, stride,
                                        scale, out_scale)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


@pytest.mark.parametrize("kernel", ["h6", "h8"])
def test_two_short_tokens_per_cta(kernel):
    """Two 1-subtile tokens per persistent CTA: the token-end l/m exchange barriers are
    token-parity double-buffered (one set let the second token's owner complete a
    barrier generation with the first token's arrive and hang the kernel)."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    lengths = [1, 0, 64, 2048] * (sms // 2)  # CTA c: tokens c, c + sms (same kind)
    q8, kv, slots, lens = _problem(lengths, pool_pages=64, width=2048, seed=3)
    fn = {"h6": sparse_mla_prefill_h6, "h8": sparse_mla_prefill_h8}[kernel]
    out = fn(q8, kv, slots, lens, 192**-0.5, 1.3)
    torch.cuda.synchronize()
    ref = _reference(q8, kv, slots, lens, 192**-0.5, 1.3)
    nz = lens > 0
    rel = (out.float() - ref).norm(dim=-1)[nz] / ref.norm(dim=-1)[nz].clamp(min=1e-6)
    assert rel.mean().item() < 0.02, rel.mean().item()
    assert (out[~nz].float() == 0).all()


def test_backend_prefill_takes_logical_topk():
    """FLASHINFER_MLA_SPARSE hands prefill-sized calls the indexer's request-relative
    top-k (the kernel's gather does the block-table remap): bit for bit the stock remap
    followed by the physical-slot kernel, with other layers' pages between blocks."""
    from types import SimpleNamespace

    from vllm.models.kimi_k3.nvidia.specialized_prefill import sparse_mla as psm
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseImpl,
    )
    from vllm.v1.attention.backends.mla.sparse_utils import (
        flat_kv_row_view,
        triton_convert_req_index_to_global_index,
    )

    g = torch.Generator(device="cuda").manual_seed(4)
    block_size, blocks, width, t = 64, 700, 2048, psm.MIN_ROWS
    ctx_blocks = [300, 100, 280]
    paged = torch.randn(blocks, 2 * block_size, ROW, device="cuda", generator=g)
    kv_cache = (paged * 0.8).to(E4M3)[:, :block_size]
    bt = torch.zeros((len(ctx_blocks), 320), dtype=torch.int32, device="cuda")
    perm = torch.randperm(blocks, device="cuda", generator=g).to(torch.int32)
    start = 0
    for r, nb in enumerate(ctx_blocks):
        bt[r, :nb] = perm[start:start + nb]
        start += nb
    req_ids = (torch.arange(t, device="cuda") % len(ctx_blocks)).to(torch.int32)
    ctx = torch.tensor(ctx_blocks, device="cuda")[req_ids.long()] * block_size
    topk = (torch.rand((t, width), device="cuda", generator=g) * ctx[:, None]).int()
    short = torch.arange(t, device="cuda") % 7 == 0
    tail = torch.arange(width, device="cuda")[None, :] >= 1000
    topk = torch.where(short[:, None] & tail, -1, topk).contiguous()
    q8 = (torch.randn(t, H, ROW, device="cuda", generator=g) * 0.8).to(E4M3)
    scale, out_scale = 192**-0.5 * 3.0, 1.3

    impl = object.__new__(FlashInferMLASparseImpl)
    impl._prefill_sparse_mla = psm
    impl.need_to_return_lse_for_decode = False
    impl.index_group = None
    impl.topk_indices_buffer = topk
    impl.bmm1_scale, impl.bmm2_scale = scale, out_scale
    impl._fused_query = None
    impl._prepare_mqa_kernel = lambda layer, device: None
    meta = SimpleNamespace(block_size=block_size, req_id_per_token=req_ids,
                           block_table=bt)
    out, lse = impl.forward_mqa(q8, kv_cache, meta, layer=None)

    kv_rows, stride = flat_kv_row_view(kv_cache, block_size)
    slots, counts = triton_convert_req_index_to_global_index(
        req_ids, bt, topk, BLOCK_SIZE=block_size, BLOCK_STRIDE_ROWS=stride,
        NUM_TOPK_TOKENS=width, return_valid_counts=True)
    ref = psm.sparse_mla_prefill(q8, kv_rows, 64, slots, counts, scale, out_scale)
    torch.cuda.synchronize()
    assert lse is None
    assert torch.equal(out, ref)


_EPW, _EWSLOW = 18, 27  # SparseMlaPrefillH8 knob positions: epw (9 = off), ewslow (ns)


@pytest.mark.parametrize("last_n", [1, 32, 64])
def test_h8_epw_store_warp_lag_bitwise(last_n):
    """epw (production): the softmax owner writes each epilogue block into a staging half
    the store warp then stores. With the store warp held back (ewslow) the owner must wait
    for every half's previous use, b < 2 included: otherwise it overwrote a half before its
    stores read it (silent output corruption) and, two completions ahead, hung epi_store on
    an aliased B_EPIFULL parity (the K3.1 prefill startup stall). Output must match the
    non-epw path bit for bit, with three tokens per CTA and a short last token."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    lengths = [2048] * sms + [2048] * sms + [last_n] * sms  # CTA c: tokens c, c+sms, c+2sms
    q8, kv, slots, lens = _problem(lengths, pool_pages=64, width=2048, seed=7)
    scale, out_scale = 192**-0.5, 1.3
    ref = sparse_mla_prefill_h8(q8, kv, slots, lens, scale, out_scale,
                                knobs=(0,) * _EPW + (9,))
    slow = (0,) * _EWSLOW + (20000,)
    for _ in range(3):
        out = sparse_mla_prefill_h8(q8, kv, slots, lens, scale, out_scale, knobs=slow)
        torch.cuda.synchronize()
        assert torch.equal(out, ref)
