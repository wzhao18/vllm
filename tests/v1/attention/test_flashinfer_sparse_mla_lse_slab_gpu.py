# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU: the trtllm-gen sparse MLA decode with LSE (the DCP path) over more rows
than the workspace's softmax-stats slab holds runs in balanced row launches whose
output and LSE are bit for bit the single launch's over a slab that holds every
row (a decode-only instance's 1024-row slab, trtllm_gen_lse_slab_rows, against
its rare larger steps; the old slab held max_num_batched_tokens rows)."""

import math
from types import SimpleNamespace
from unittest.mock import patch

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


def _impl(workspace_rows: int):
    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fims

    nbytes = fims._required_workspace_bytes(16, HEADS // 16, workspace_rows)
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
    assert fims.trtllm_gen_lse_rows_capacity(nbytes, HEADS) == workspace_rows
    return impl


def _case(rows: int):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    q = (0.5 * torch.randn(rows, HEADS, 576, device="cuda", generator=gen)).to(
        torch.float8_e4m3fn
    )
    kv = (0.5 * torch.randn(NUM_BLOCKS, BLOCK, 576, device="cuda", generator=gen)).to(
        torch.float8_e4m3fn
    )
    topk = torch.randint(
        0, NUM_BLOCKS * BLOCK, (rows, TOPK), device="cuda", generator=gen
    ).int()
    counts = torch.randint(0, TOPK + 1, (rows,), device="cuda", generator=gen).int()
    counts[::97] = 0  # rows this rank owns none of
    counts[1::89] = TOPK
    cols = torch.arange(TOPK, device="cuda")
    topk[cols[None, :] >= counts[:, None]] = -1
    return q, kv, topk, counts


def _run(impl, case):
    import flashinfer.decode

    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fims

    q, kv, topk, counts = case
    launches = []
    kernel = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla

    def counted(**kwargs):
        launches.append(kwargs["query"].shape[0])
        return kernel(**kwargs)

    fims._reserve_kv_counters(q.device, q.shape[0], HEADS)
    with patch.object(
        flashinfer.decode, "trtllm_batch_decode_with_kv_cache_mla", counted
    ):
        out, lse = impl._forward_mqa_kernel(
            q,
            kv,
            topk,
            counts,
            layer=SimpleNamespace(),
            block_size=BLOCK,
            is_decode=True,
        )
    torch.accelerator.synchronize()
    return out.clone(), lse.clone(), launches


@pytest.mark.parametrize(
    "rows, slab_rows",
    [(1024, 1024), (1025, 1024), (2049, 1024), (8192, 1024), (4097, 2048)],
)
def test_lse_launches_beyond_slab_match_single_launch(rows, slab_rows):
    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fims

    case = _case(rows)
    single_out, single_lse, single = _run(_impl(rows), case)
    assert single == [rows]
    out, lse, launches = _run(_impl(slab_rows), case)
    step = fims.trtllm_gen_lse_launch_rows(rows, slab_rows)
    assert launches == [min(step, rows - s) for s in range(0, rows, step)]
    assert len(launches) == -(-rows // slab_rows)
    assert torch.isfinite(single_out).all()
    assert torch.equal(out.view(torch.int16), single_out.view(torch.int16))
    assert torch.equal(lse.view(torch.int32), single_lse.view(torch.int32))
