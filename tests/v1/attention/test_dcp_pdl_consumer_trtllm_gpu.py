# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU: FlashInfer's trtllm-gen sparse MLA decode (launched with PDL) reads its
top-k rows and per-row counts only after its griddepcontrol.wait. The DCP top-k
rescue and the DCP index remap release their dependents at entry, so the decode
becomes resident while they still write those rows: a producer that releases at
entry, stalls, then writes the rows and counts must still give the decode's
result over the final rows, eagerly and in a CUDA graph."""

import functools
import math
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="trtllm-gen sparse MLA decode (SM100 family)",
)

HEADS = 128  # DCP16 x 6 heads per rank (96), padded to 128 by trtllm-gen
TOPK = 2048
BLOCK = 64
NUM_BLOCKS = 1024


@functools.cache
def _late_writer():
    from vllm.triton_utils import tl, triton

    @triton.jit
    def late_writer(src, dst, numel, spin, SPIN: tl.constexpr, BLOCK_N: tl.constexpr):
        # Release the dependents first (as the rescue / remap do), then stall on a
        # chain of dependent loads before writing: a consumer that read before its
        # griddepcontrol.wait would see the stale rows.
        tl.extra.cuda.gdc_launch_dependents()
        idx = tl.program_id(0) * 0
        for _ in range(SPIN):
            idx = tl.load(spin + idx, volatile=True)
        offs = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N) + idx
        mask = offs < numel
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)

    return late_writer


def _late_copy(src, dst, spin):
    numel = src.numel()
    _late_writer()[(-(-numel // 4096),)](src, dst, numel, spin, SPIN=256, BLOCK_N=4096)


def _impl(rows):
    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fims

    nbytes = fims._required_workspace_bytes(16, HEADS // 16, max(rows, 1024))
    impl = object.__new__(fims.FlashInferMLASparseImpl)
    impl.__dict__.update(
        use_prims_ts=False,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        is_nope_mla=False,
        need_to_return_lse_for_decode=True,
        index_group=None,
        _fused_query=None,
        _workspace_buffer=torch.zeros(nbytes, dtype=torch.int8, device="cuda"),
        bmm1_scale=1.0 / math.sqrt(192),
        bmm2_scale=1.0,
    )
    fims._reserve_kv_counters(torch.device("cuda"), rows, HEADS)
    return impl


@pytest.mark.parametrize("last", ["topk", "counts"])
@pytest.mark.parametrize("rows", [8, 16, 32, 64])
def test_trtllm_gen_sparse_decode_waits_for_early_releasing_producer(rows, last):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    q = (0.5 * torch.randn(rows, HEADS, 576, device="cuda", generator=gen)).to(
        torch.float8_e4m3fn
    )
    kv = (0.5 * torch.randn(NUM_BLOCKS, BLOCK, 576, device="cuda", generator=gen)).to(
        torch.float8_e4m3fn
    )
    topk_src = torch.randint(
        0, NUM_BLOCKS * BLOCK, (rows, TOPK), device="cuda", generator=gen
    ).int()
    counts_src = torch.randint(1, TOPK + 1, (rows,), device="cuda", generator=gen).int()
    counts_src[1::3] = TOPK
    cols = torch.arange(TOPK, device="cuda")
    topk_src[cols[None, :] >= counts_src[:, None]] = -1
    spin = torch.zeros(1, device="cuda", dtype=torch.int32)
    impl = _impl(rows)

    def decode(topk, counts):
        out, lse = impl._forward_mqa_kernel(
            q, kv, topk, counts, layer=SimpleNamespace(), block_size=BLOCK,
            is_decode=True,
        )
        return out, lse

    ref_out, ref_lse = (t.clone() for t in decode(topk_src, counts_src))
    torch.accelerator.synchronize()

    topk = torch.empty_like(topk_src)
    counts = torch.empty_like(counts_src)

    def stale():
        # stale rows: another request's ids and no local entries
        topk.copy_(topk_src.roll(1, dims=0))
        counts.zero_()

    def step():
        # the decode's immediate predecessor writes ``last``
        pairs = [(topk_src, topk), (counts_src, counts)]
        for src, dst in pairs if last == "counts" else pairs[::-1]:
            _late_copy(src, dst, spin)
        return decode(topk, counts)

    stale()
    stale_out, _ = decode(topk, counts)
    torch.accelerator.synchronize()
    assert not torch.equal(stale_out.view(torch.int16), ref_out.view(torch.int16))

    for i in range(4):
        stale()
        torch.accelerator.synchronize()
        out, lse = step()
        torch.accelerator.synchronize()
        assert torch.equal(out.view(torch.int16), ref_out.view(torch.int16)), i
        assert torch.equal(lse.view(torch.int32), ref_lse.view(torch.int32)), i

    graph = torch.cuda.CUDAGraph()
    stale()
    with torch.cuda.graph(graph):
        g_out, g_lse = step()
    for i in range(8):
        stale()
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(g_out.view(torch.int16), ref_out.view(torch.int16)), i
        assert torch.equal(g_lse.view(torch.int32), ref_lse.view(torch.int32)), i
