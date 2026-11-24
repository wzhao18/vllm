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


def _compiled_ptx(jit_fn) -> list[str]:
    """PTX of every compiled specialization of a triton JIT function."""
    caches = getattr(jit_fn, "device_caches", None)
    if caches is not None:
        kernels = [k for cache in caches.values() for k in cache[0].values()]
    else:  # older triton
        kernels = [k for cache in jit_fn.cache.values() for k in cache.values()]
    return [k.asm["ptx"] for k in kernels if "ptx" in getattr(k, "asm", {})]


def test_dcp_remap_emits_launch_dependents():
    """The DCP remap's compiled kernel carries the early PDL release (the
    consumer tests above pass with or without it), and the DCP-1 remap does not."""
    from vllm.v1.attention.backends.mla import sparse_utils

    jit_fn = sparse_utils.ConvertReqIndexToGlobalIndexKernel.sparse_mla_index_remap_kernel
    rows, dcp, rank = 8, 16, 0
    req_id = torch.zeros(rows, device="cuda", dtype=torch.int32)
    block_table = torch.arange(64, device="cuda", dtype=torch.int32).view(1, 64)
    ids = torch.randint(0, 64 * BLOCK * dcp, (rows, TOPK), device="cuda",
                        dtype=torch.int32)

    def remap(dcp_size):
        before = set(_compiled_ptx(jit_fn))
        sparse_utils.triton_filter_and_convert_dcp_index(
            req_id, block_table, ids, dcp_size=dcp_size, dcp_rank=0,
            BLOCK_SIZE=BLOCK, BLOCK_STRIDE_ROWS=BLOCK, NUM_TOPK_TOKENS=TOPK,
            return_valid_counts=True,
        ) if dcp_size > 1 else sparse_utils.triton_convert_req_index_to_global_index(
            req_id, block_table, ids, BLOCK_SIZE=BLOCK, BLOCK_STRIDE_ROWS=BLOCK,
            NUM_TOPK_TOKENS=TOPK, return_valid_counts=True,
        )
        torch.accelerator.synchronize()
        return [p for p in _compiled_ptx(jit_fn) if p not in before]

    new = remap(dcp)
    ptx = new or _compiled_ptx(jit_fn)
    assert ptx, "no compiled remap kernel found"
    assert any("griddepcontrol.launch_dependents" in p for p in ptx)
    plain = remap(1)
    assert all("griddepcontrol.launch_dependents" not in p for p in plain)
