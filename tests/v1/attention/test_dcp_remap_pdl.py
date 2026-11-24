# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DCP decode index remap (``triton_filter_and_convert_dcp_index``) releases
its PDL dependents at entry: a consumer launched with programmatic stream
serialization that waits before its reads (as the K3.1 sparse decode's QueryWait
and attention core) still sees the final rows, eagerly and in a CUDA graph, and
the rows match a plain PyTorch reference of the remap."""

import functools

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(90),
    reason="griddepcontrol (PDL) needs CUDA sm_90+",
)

TOPK = 2048
BLOCK = 64


@functools.cache
def _pdl_copy_kernel():
    from vllm.triton_utils import tl, triton

    @triton.jit
    def pdl_copy_kernel(src, dst, numel, BLOCK_N: tl.constexpr):
        tl.extra.cuda.gdc_wait()
        offs = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < numel
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)

    return pdl_copy_kernel


def _pdl_copy(src: torch.Tensor, dst: torch.Tensor) -> None:
    numel = src.numel()
    _pdl_copy_kernel()[(-(-numel // 1024),)](
        src, dst, numel, BLOCK_N=1024, launch_pdl=True
    )


def _reference(req_id, block_table, ids, dcp, rank):
    """Per row: the sorted flat KV rows of the ids this rank owns (interleave 1)."""
    out = []
    for t in range(ids.shape[0]):
        tok = ids[t]
        tok = tok[tok >= 0]
        local = tok // dcp
        mine = (tok % dcp == rank) & (local // BLOCK < block_table.shape[1])
        local = local[mine]
        rows = block_table[req_id[t], local // BLOCK] * BLOCK + local % BLOCK
        out.append(sorted(rows.tolist()))
    return out


@pytest.mark.parametrize("rows", [8, 128, 512])
def test_dcp_remap_pdl_consumer(rows):
    from vllm.v1.attention.backends.mla.sparse_utils import (
        _remap_launch_dependents,
        triton_filter_and_convert_dcp_index,
    )

    dcp, rank = 16, 3
    assert _remap_launch_dependents(dcp) and not _remap_launch_dependents(1)
    gen = torch.Generator(device="cuda").manual_seed(rows)
    reqs = max(1, rows // 8)
    max_local = 135_176 // dcp + 1
    blocks = -(-max_local // BLOCK)
    req_id = torch.arange(rows, device="cuda", dtype=torch.int32) // (rows // reqs)
    block_table = torch.randperm(reqs * blocks, generator=gen, device="cuda")
    block_table = block_table.to(torch.int32).view(reqs, blocks)
    ids = torch.randint(
        0, 135_176, (rows, TOPK), generator=gen, device="cuda", dtype=torch.int32
    )
    ids[:, -100:] = -1  # padding ids attend nothing

    def step():
        slots, valid = triton_filter_and_convert_dcp_index(
            req_id,
            block_table,
            ids,
            dcp_size=dcp,
            dcp_rank=rank,
            BLOCK_SIZE=BLOCK,
            BLOCK_STRIDE_ROWS=BLOCK,
            NUM_TOPK_TOKENS=TOPK,
            return_valid_counts=True,
        )
        _pdl_copy(slots, slots_seen)
        _pdl_copy(valid, valid_seen)
        return slots, valid

    slots_seen = torch.empty_like(ids)
    valid_seen = torch.empty(rows, device="cuda", dtype=torch.int32)
    slots, valid = step()
    torch.accelerator.synchronize()
    assert torch.equal(slots_seen, slots) and torch.equal(valid_seen, valid)
    want = _reference(req_id, block_table, ids, dcp, rank)
    got = [
        sorted(slots[t, : valid[t]].tolist()) for t in range(rows)
    ]
    assert got == want
    assert bool((slots[:, valid.max() :] == -1).all())

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_slots, g_valid = step()
    for i in range(8):
        ids.copy_(ids.roll(1, dims=0))
        slots_seen.fill_(-7)
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(slots_seen, g_slots), f"replay {i}"
        assert torch.equal(valid_seen, g_valid), f"replay {i}"
