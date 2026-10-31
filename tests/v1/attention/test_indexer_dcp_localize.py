# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl
from vllm.v1.attention.backends.mla.indexer import build_prefill_chunk_metadata
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.ops.dcp import CPTritonContext, correct_attn_out


def _local_count(length: int, rank: int, world: int, interleave: int) -> int:
    return sum(1 for pos in range(length) if (pos // interleave) % world == rank)


def _global_to_local_indices(
    global_indices: torch.Tensor,
    rank: int,
    world: int,
    interleave: int,
) -> torch.Tensor:
    valid = global_indices >= 0
    global_i64 = global_indices.to(torch.int64).clamp_min(0)
    owner = (global_i64 // interleave) % world
    local = (global_i64 // (world * interleave)) * interleave + global_i64 % interleave
    return torch.where(valid & (owner == rank), local, -1).to(torch.int64)


def _local_to_global_indices(
    local_indices: torch.Tensor,
    rank: int,
    world: int,
    interleave: int,
) -> torch.Tensor:
    valid = local_indices >= 0
    local = local_indices.to(torch.int64).clamp_min(0)
    global_indices = (
        (local // interleave) * (world * interleave)
        + rank * interleave
        + local % interleave
    )
    return torch.where(valid, global_indices, -1).to(torch.int64)


def _ref_stable_topk_from_candidates_fp64(
    candidate_scores: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Pure-PyTorch reference for the CuteDSL stable-topk selector order
    (score desc, then lowest global token id). Selects the same SET as the
    kernel; only the set is compared in tests."""
    num_rows, num_candidates = candidate_scores.shape
    device = candidate_scores.device
    select_k = min(k, num_candidates)
    valid = candidate_token_ids >= 0
    bits = (
        candidate_scores.to(torch.float32).view(torch.int32).to(torch.int64)
        & 0xFFFFFFFF
    )
    sign = (bits >> 31) & 1
    score_key = (
        torch.where(sign.bool(), bits ^ 0xFFFFFFFF, bits ^ 0x80000000) & 0xFFFFFFFF
    )
    id_key = (~candidate_token_ids.to(torch.int64)) & 0xFFFFFFFF
    key = (score_key << 32) | id_key
    key = torch.where(valid, key, torch.zeros_like(key))
    topk_key = key ^ torch.iinfo(torch.int64).min
    _, topk_pos = topk_key.topk(select_k, dim=-1)

    selected = candidate_token_ids.gather(1, topk_pos).to(torch.int32)
    selected_valid = valid.gather(1, topk_pos)
    selected = torch.where(selected_valid, selected, selected.new_full((), -1))
    if select_k == k:
        return selected
    pad = torch.full((num_rows, k - select_k), -1, dtype=torch.int32, device=device)
    return torch.cat((selected, pad), dim=1)


def _attention_from_indices(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = indices >= 0
    safe_indices = indices.clamp_min(0)
    selected_k = k[safe_indices]
    selected_v = v[safe_indices]
    scores = torch.einsum("td,tkd->tk", q, selected_k)
    scores = scores.masked_fill(~valid, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1).masked_fill(~valid, 0.0)
    out = torch.einsum("tk,tkd->td", probs, selected_v)
    empty_rows = ~valid.any(dim=-1)
    out[empty_rows] = 0
    lse[empty_rows] = float("-inf")
    return out, lse


def _dcp_lse_merge(
    local_outs: list[torch.Tensor],
    local_lses: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    outs = torch.stack(local_outs, dim=0)
    lses = torch.stack(local_lses, dim=0)
    merged_lse = torch.logsumexp(lses, dim=0)
    weights = torch.exp(lses - merged_lse.unsqueeze(0))
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    merged_out = (outs * weights.unsqueeze(-1)).sum(dim=0)
    return merged_out, merged_lse


class _FakeDCPGroup(GroupCoordinator):
    """Single-process stand-in: ``all_gather`` returns the pre-built
    concatenation of every rank's packed ``(score, global_id)`` candidates,
    mirroring the one packed all-gather the merge issues.

    ``GroupCoordinator.__init__`` is never run, so no process group exists."""

    def __init__(self, gathered_packed: torch.Tensor) -> None:
        self.gathered_packed = gathered_packed

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == 1
        return self.gathered_packed.clone()


def _run_decode_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    next_n: int,
    topk: int,
) -> torch.Tensor:
    indices = torch.empty(
        (logits.shape[0], topk), dtype=torch.int32, device=logits.device
    )
    torch.ops._C.top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        topk,
    )
    return indices


def _run_persistent_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int,
    max_seq_len: int,
) -> torch.Tensor:
    indices = torch.empty(
        (logits.shape[0], topk), dtype=torch.int32, device=logits.device
    )
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=logits.device)
    torch.ops._C.persistent_topk(
        logits,
        seq_lens,
        indices,
        workspace,
        topk,
        max_seq_len,
    )
    return indices


def _dcp_attention_from_local_topks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    local_topks: list[torch.Tensor],
    world: int,
    interleave: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_outs = []
    local_lses = []
    for rank, local_topk in enumerate(local_topks):
        owned = [
            pos for pos in range(k.shape[0]) if (pos // interleave) % world == rank
        ]
        local_out, local_lse = _attention_from_indices(
            q, k[owned], v[owned], local_topk.to(torch.int64)
        )
        local_outs.append(local_out)
        local_lses.append(local_lse)
    return _dcp_lse_merge(local_outs, local_lses)


def _merge_local_topks_global_with_fake_dcp(
    local_logits: list[torch.Tensor],
    local_topks: list[torch.Tensor],
    topk: int,
    world: int,
    interleave: int,
    row_starts: list[torch.Tensor | None] | None = None,
) -> list[torch.Tensor]:
    """Run ``_merge_dcp_topk_global`` for every rank against a faked
    all-gather and return each rank's global-index result (all ranks should
    agree). The fake pre-builds the packed candidate concatenation the merge's
    single ``all_gather`` would return."""
    # The merge is now CuteDSL-only (no PyTorch fallback), so it runs the real
    # Triton pack + CuteDSL selector kernels even behind the faked all-gather.
    if not current_platform.is_cuda() or not has_cutedsl():
        pytest.skip("DCP merge requires CUDA and CuteDSL")
    packed_per_rank = []
    for rank, (logits, indices) in enumerate(zip(local_logits, local_topks)):
        score_indices = indices.clamp_min(0).to(torch.long)
        if row_starts is not None:
            rs = row_starts[rank]
            assert rs is not None
            score_indices = score_indices + rs.to(torch.long).view(-1, 1)
        if logits.shape[1] == 0:
            scores = torch.full_like(indices, float("-inf"), dtype=torch.float32)
        else:
            score_indices = score_indices.clamp_max(logits.shape[1] - 1)
            scores = logits.gather(1, score_indices).masked_fill(
                indices < 0, float("-inf")
            )
        global_ids = _local_to_global_indices(indices, rank, world, interleave)
        packed_per_rank.append(
            torch.stack((scores.float(), global_ids.to(torch.float32)), dim=-1)
        )

    fake_group = _FakeDCPGroup(torch.cat(packed_per_rank, dim=1).contiguous())
    original_get_dcp_group = sparse_indexer.get_dcp_group
    sparse_indexer.get_dcp_group = lambda: fake_group
    try:
        merged = []
        for rank, (logits, indices) in enumerate(zip(local_logits, local_topks)):
            rank_indices = indices.clone()
            # in-place merge; the None return is asserted below
            result = sparse_indexer._merge_dcp_topk_global(  # type: ignore[func-returns-value]
                logits,
                rank_indices,
                topk,
                rank,
                world,
                interleave,
                row_starts=None if row_starts is None else row_starts[rank],
            )
            assert result is None
            merged.append(rank_indices)
        return merged
    finally:
        sparse_indexer.get_dcp_group = original_get_dcp_group


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("interleave", [1, 2, 4])
def test_get_dcp_local_seq_lens_matches_naive(world: int, interleave: int):
    seq_lens = torch.arange(0, 33, dtype=torch.int32)

    for rank in range(world):
        actual = get_dcp_local_seq_lens(seq_lens, world, rank, interleave)
        expected = torch.tensor(
            [
                _local_count(int(seq_len), rank, world, interleave)
                for seq_len in seq_lens
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual, expected)


_NIXL_P = {
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "connectors": [{"kv_connector": "NixlConnector", "kv_role": "kv_producer"}]
    },
}
_NIXL_D = {"kv_connector": "NixlConnector", "kv_role": "kv_consumer"}
_NCCL_PD = {"kv_connector": "P2pNcclConnector", "kv_role": "kv_both"}


@pytest.mark.parametrize(
    ("interleave", "dcp", "kv_transfer", "explicit", "dcp_interleave", "expected"),
    [
        (1, 16, None, False, 1, True),
        (1024, 16, None, True, 1, False),
        (1, 1, _NIXL_D, False, 1, True),  # no DCP: nothing resolves it
        (1, 16, _NIXL_D, True, 1, True),
        (1024, 16, _NIXL_D, True, 1, False),
        # NIXL raises an interleave left unset to the block size.
        (1, 16, _NIXL_D, False, 1, False),
        (1, 8, _NIXL_P, False, 1, False),
        (1, 8, _NIXL_P, True, 1, True),
        # The deprecated DCP interleave overrides it on a P/D instance.
        (1, 8, _NCCL_PD, False, 1, True),
        (1, 8, _NCCL_PD, True, 64, False),
    ],
)
def test_cp_kv_cache_interleave_one_is_final(
    interleave, dcp, kv_transfer, explicit, dcp_interleave, expected
):
    """Model construction gates interleave-1 paths on the interleave that
    VllmConfig.adjust_dcp_kv_cache_interleave_size will leave, not on the
    current value."""
    from types import SimpleNamespace

    from vllm.config import KVTransferConfig
    from vllm.v1.attention.ops.dcp import cp_kv_cache_interleave_one

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            cp_kv_cache_interleave_size=interleave,
            decode_context_parallel_size=dcp,
            dcp_kv_cache_interleave_size=dcp_interleave,
            _allow_auto_resolve_cp_interleave_size=not explicit,
        ),
        kv_transfer_config=(
            KVTransferConfig(**kv_transfer) if kv_transfer is not None else None
        ),
    )
    assert cp_kv_cache_interleave_one(config) is expected


_MOONCAKE = {"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both"}


@pytest.mark.parametrize("kv_transfer", [None, _NIXL_P, _NIXL_D, _NCCL_PD, _MOONCAKE])
@pytest.mark.parametrize("block_size", [64, 1024])
def test_cp_kv_cache_interleave_one_matches_adjust(kv_transfer, block_size):
    """The construction-time prediction agrees with the interleave that
    VllmConfig.adjust_dcp_kv_cache_interleave_size actually leaves, so a
    change to the adjust cannot silently leave the interleave-1 gates
    (ShardQ, the one-round top-k merge, #61) on the wrong side."""
    import copy
    import itertools
    from types import SimpleNamespace

    from vllm.config import KVTransferConfig
    from vllm.config.vllm import VllmConfig
    from vllm.v1.attention.ops.dcp import cp_kv_cache_interleave_one

    groups = [SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=block_size))]
    for interleave, dcp, explicit, dcp_interleave in itertools.product(
        (1, 64, 1024), (2, 8, 16), (False, True), (1, 64)
    ):
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                cp_kv_cache_interleave_size=interleave,
                decode_context_parallel_size=dcp,
                dcp_kv_cache_interleave_size=dcp_interleave,
                _allow_auto_resolve_cp_interleave_size=not explicit,
            ),
            kv_transfer_config=(
                KVTransferConfig(**copy.deepcopy(kv_transfer))
                if kv_transfer is not None
                else None
            ),
        )
        predicted = cp_kv_cache_interleave_one(config)
        VllmConfig.adjust_dcp_kv_cache_interleave_size(config, groups)
        final = config.parallel_config.cp_kv_cache_interleave_size
        assert predicted is (final == 1), (interleave, dcp, explicit, dcp_interleave)


def test_get_dcp_local_seq_lens_can_localize_per_token_bounds():
    seq_lens = torch.tensor([0, 1, 2, 3, 4, 7, 8, 17], dtype=torch.int32)
    world = 4
    interleave = 2

    for rank in range(world):
        actual = get_dcp_local_seq_lens(seq_lens, world, rank, interleave)
        expected = torch.tensor(
            [
                _local_count(int(seq_len), rank, world, interleave)
                for seq_len in seq_lens
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual, expected)


def test_get_dcp_local_seq_lens_preserves_mtp_bounds_shape():
    seq_lens = torch.tensor([[8, 9, 10], [11, 12, 13]], dtype=torch.int32)
    world = 2
    rank = 1
    interleave = 1

    actual = get_dcp_local_seq_lens(seq_lens, world, rank, interleave)
    expected = torch.tensor(
        [
            [_local_count(int(seq_len), rank, world, interleave) for seq_len in row]
            for row in seq_lens
        ],
        dtype=torch.int32,
    )

    assert actual.shape == seq_lens.shape
    torch.testing.assert_close(actual, expected)


def test_get_dcp_local_seq_lens_reuses_device_rank_tensor(monkeypatch):
    seq_lens = torch.tensor([0, 1, 7, 8, 17], dtype=torch.int32)
    world = 4
    rank = 2
    rank_tensor = torch.tensor(rank, dtype=torch.int32)
    expected = get_dcp_local_seq_lens(seq_lens, world, rank, 2)

    def fail_tensor_construction(*args, **kwargs):
        raise AssertionError("rank tensor must be reused")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "tensor", fail_tensor_construction)
        actual = get_dcp_local_seq_lens(seq_lens, world, rank_tensor, 2)

    torch.testing.assert_close(actual, expected)


def test_get_dcp_local_seq_lens_must_run_after_decode_expansion():
    world = 2
    rank = 1
    interleave = 1
    expanded_bounds = torch.tensor([8, 9, 10], dtype=torch.int32)

    localized_after_expansion = get_dcp_local_seq_lens(
        expanded_bounds, world, rank, interleave
    )
    localized_request_len_minus_offsets = get_dcp_local_seq_lens(
        torch.tensor([10], dtype=torch.int32), world, rank
    ) - torch.tensor([2, 1, 0], dtype=torch.int32)

    assert not torch.equal(
        localized_after_expansion, localized_request_len_minus_offsets
    )
    torch.testing.assert_close(
        localized_after_expansion, torch.tensor([4, 4, 5], dtype=torch.int32)
    )


def _run_decode_metadata_kernel(
    outputs, qsl, seq_lens, block_table, num_tokens, dcp=(1, 0, 1), write_global=0
):
    from vllm.v1.attention.ops.metadata import _indexer_decode_metadata_kernel

    out_seq, out_bt, out_lens, indices, per_req, global_seq = outputs
    num_reqs = block_table.shape[0]
    capacity = out_seq.numel()
    _indexer_decode_metadata_kernel[
        (max(num_reqs, num_tokens + (capacity - num_tokens + 255) // 256),)
    ](
        qsl,
        seq_lens,
        block_table,
        out_seq,
        out_bt,
        out_lens,
        indices,
        per_req,
        global_seq,
        num_reqs,
        num_tokens,
        capacity,
        block_table.stride(0),
        out_bt.stride(0),
        *dcp,
        write_global,
        BLOCK_COLS=block_table.shape[1],
        num_warps=4,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("world", [8, 16])
@pytest.mark.parametrize("interleave", [1, 1024])
def test_indexer_decode_metadata_kernel_localizes_like_torch(world, interleave):
    """The decode metadata kernel's DCP localization equals the unlocalized
    kernel followed by get_dcp_local_seq_lens, bit for bit and on every rank:
    graph padding and negative bounds (a sequence shorter than its query)
    included; the global bounds land where the old copy put them."""
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(world + interleave)
    query_lens = [8, 8, 3, 8, 1, 8, 8, 0, 0]
    seq_lens = [1_100_000, 1, 7, 1024 * world + 5, 1, 9, 65536, 0, 0]
    qsl = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(query_lens).cumsum(0)
    num_tokens = int(qsl[-1]) + 5
    qsl = qsl.to(device)
    seq = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    block_table = torch.randint(
        0, 1 << 20, (len(query_lens), 72), generator=gen, dtype=torch.int32
    ).to(device)[:, :65]
    capacity = 600

    def outputs():
        torch.manual_seed(0)
        return [
            torch.randint(-(1 << 30), 1 << 30, shape, dtype=torch.int32, device=device)
            for shape in [
                (capacity,),
                (num_tokens, 69),
                (capacity,),
                (capacity,),
                (capacity,),
                (capacity,),
            ]
        ]

    for rank in range(world):
        for write_global in (0, 1):
            fused, ref = outputs(), outputs()
            _run_decode_metadata_kernel(
                fused,
                qsl,
                seq,
                block_table,
                num_tokens,
                (world, rank, interleave),
                write_global,
            )
            _run_decode_metadata_kernel(ref, qsl, seq, block_table, num_tokens)
            ref_seq = ref[0][:num_tokens]
            assert (ref_seq < 0).any()
            if write_global:
                ref[5][:num_tokens].copy_(ref_seq)
            ref_seq.copy_(
                get_dcp_local_seq_lens(
                    ref_seq,
                    world,
                    torch.tensor(rank, dtype=torch.int32, device=device),
                    interleave,
                )
            )
            for got, want in zip(fused, ref):
                assert torch.equal(got, want)


def _varlen_dcp_indexer_builder(monkeypatch, world: int, rank: int, interleave: int):
    from types import SimpleNamespace

    from vllm.v1.attention.backends.mla import indexer as indexer_module
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    monkeypatch.setattr(
        indexer_module, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=rank)
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1 << 20),
        scheduler_config=SimpleNamespace(max_num_seqs=32, max_num_batched_tokens=8192),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=world,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=interleave,
        ),
        attention_config=SimpleNamespace(
            sparse_mla_prefill_workspace_factor=5,
            resolve_indexer_kv_dtype=lambda default: default,
        ),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=7, enable_adaptive_verification=False
        ),
        num_speculative_tokens=7,
    )
    spec = MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=132, dtype=torch.uint8
    )
    builder = indexer_module.DeepseekV32IndexerMetadataBuilder(
        spec, ["indexer"], config, torch.device("cuda"), block_table_width=1040
    )
    assert builder.supports_varlen and builder.dcp_rank == rank
    builder.scheduler_metadata_buffer.fill_(-7)
    return builder


def _unfused_varlen_dcp_decode(builder, common) -> int:
    """The varlen DCP decode metadata as built before the kernel localized it:
    the global bounds, their copy, get_dcp_local_seq_lens over the expanded
    bounds and the schedule copied into its buffer. Returns its rows."""
    from vllm.v1.attention.backends.mla.indexer import get_paged_mqa_logits_metadata

    num_decodes, num_tokens = common.num_reqs, common.num_actual_tokens
    outputs = [
        builder.decode_seq_lens_buffer,
        builder.expanded_block_table_buffer,
        builder.decode_lens_buffer,
        builder.decode_indices_buffer,
        builder.per_req_decode_lens_buffer,
        builder.global_decode_seq_lens_buffer,
    ]
    _run_decode_metadata_kernel(
        outputs,
        common.query_start_loc,
        common.seq_lens[:num_decodes],
        common.block_table_tensor[:num_decodes],
        num_tokens,
    )
    seq_lens = builder.decode_seq_lens_buffer[:num_tokens]
    if int(torch.diff(common.query_start_loc_cpu).max()) > 1:
        builder.global_decode_seq_lens_buffer[:num_tokens].copy_(seq_lens)
    seq_lens = builder._dcp_localize_decode_seq_lens(seq_lens, num_decodes, True)
    metadata = get_paged_mqa_logits_metadata(
        seq_lens.unsqueeze(-1),
        builder.kv_cache_spec.num_states,
        builder.num_sms,
        indices=builder.decode_indices_buffer[:num_tokens],
    )
    builder.scheduler_metadata_buffer[: metadata.shape[0]] = metadata
    return metadata.shape[0]


def _decode_steps(world: int, gen: torch.Generator):
    """(query lens, seq lens, padded requests, padded tokens) of decode steps:
    q8 batches of 1-8 requests (M8-M64), graph padding, mixed and q1 rows."""
    long = [1_100_000, 1 << 20, 65536 * world + 1, 1024 * world, 1024 * world - 1]
    short = [8, 9, 15, 16, 17, 1023, 1024, 1025]
    for num_reqs in range(1, 9):
        seqs = torch.randint(8, 1_100_001, (num_reqs,), generator=gen).tolist()
        yield [8] * num_reqs, seqs, 0, 0
    yield [8] * 5, long, 0, 0
    yield [8] * 8, short, 0, 0
    yield [8] * 3, long[:3], 1, 0
    yield [8] * 6, short[:6], 2, 8
    yield [8, 3, 8, 1, 8], [40, 3, 1 << 20, 1, 65537], 3, 5
    yield [1] * 4, [1, 2, 1024, 1_100_000], 0, 4


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("world", [8, 16])
@pytest.mark.parametrize("interleave", [1, 1024])
def test_varlen_dcp_decode_build_matches_unfused_metadata(
    monkeypatch, world, interleave
):
    """Folding the DCP localization (and the global bounds' copy) into the
    varlen decode metadata kernel leaves every decode buffer of the build bit
    for bit as the unfused ops left it, over the buffers' whole capacity and
    across steps, on every rank, and the metadata views the same addresses."""
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.mla import indexer as indexer_module

    if not indexer_module._supports_varlen_paged_mqa_logits():
        pytest.skip("requires DeepGEMM's varlen paged MQA logits (SM100 family)")

    device = torch.device("cuda")
    names = [
        "decode_seq_lens_buffer",
        "global_decode_seq_lens_buffer",
        "expanded_block_table_buffer",
        "decode_lens_buffer",
        "decode_indices_buffer",
        "per_req_decode_lens_buffer",
        "scheduler_metadata_buffer",
    ]
    for rank in range(world):
        fused = _varlen_dcp_indexer_builder(monkeypatch, world, rank, interleave)
        ref = _varlen_dcp_indexer_builder(monkeypatch, world, rank, interleave)
        gen = torch.Generator().manual_seed(rank)
        for query_lens, seq_lens, pad_reqs, pad_tokens in _decode_steps(world, gen):
            query_lens = query_lens + [0] * pad_reqs
            qsl = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
            qsl[1:] = torch.tensor(query_lens).cumsum(0)
            seq = torch.tensor(seq_lens + [0] * pad_reqs, dtype=torch.int32)
            num_reqs, num_tokens = len(query_lens), int(qsl[-1]) + pad_tokens
            common = CommonAttentionMetadata(
                query_start_loc=qsl.to(device),
                query_start_loc_cpu=qsl,
                seq_lens=seq.to(device),
                seq_lens_cpu_upper_bound=seq,
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                max_query_len=max(query_lens),
                max_seq_len=max(seq_lens),
                block_table_tensor=torch.randint(
                    0, 1 << 24, (num_reqs, 1040), generator=gen, dtype=torch.int32
                ).to(device),
                slot_mapping=torch.arange(num_tokens, device=device),
                dcp_local_seq_lens=get_dcp_local_seq_lens(
                    seq.to(device), world, rank, interleave
                ),
            )
            decode = fused.build(0, common).decode
            rows = _unfused_varlen_dcp_decode(ref, common)
            for name in names:
                assert torch.equal(getattr(fused, name), getattr(ref, name)), name
            assert decode is not None
            assert decode.seq_lens.shape == (num_tokens, 1)
            assert decode.seq_lens.data_ptr() == fused.decode_seq_lens_buffer.data_ptr()
            assert decode.schedule_metadata.shape == (rows, 2)
            assert (
                decode.schedule_metadata.data_ptr()
                == fused.scheduler_metadata_buffer.data_ptr()
            )
            assert decode.global_seq_lens is not None
            if max(query_lens) > 1:
                assert decode.global_seq_lens.shape == (num_tokens,)
                global_ptr = fused.global_decode_seq_lens_buffer.data_ptr()
            else:
                assert decode.global_seq_lens.shape == (num_reqs,)
                global_ptr = common.seq_lens.data_ptr()
            assert decode.global_seq_lens.data_ptr() == global_ptr
            for view, buffer in [
                (decode.block_table, fused.expanded_block_table_buffer),
                (decode.decode_lens, fused.decode_lens_buffer),
                (decode.indices, fused.decode_indices_buffer),
                (decode.per_req_decode_lens, fused.per_req_decode_lens_buffer),
            ]:
                assert view is not None and view.data_ptr() == buffer.data_ptr()


@pytest.mark.parametrize("interleave", [1, 2])
def test_sparse_dcp_attention_matches_global_topk_attention(interleave: int):
    torch.manual_seed(0)
    world = 2
    topk = 3
    num_queries = 4
    max_seq_len = 13
    head_dim = 8

    q = torch.randn(num_queries, head_dim)
    k = torch.randn(max_seq_len, head_dim)
    v = torch.randn(max_seq_len, head_dim)
    seq_lens = torch.tensor([6, 8, 11, 13], dtype=torch.int64)

    scores = q @ k.T
    global_topk = torch.full((num_queries, topk), -1, dtype=torch.int64)
    for row, seq_len in enumerate(seq_lens.tolist()):
        global_topk[row, :topk] = scores[row, :seq_len].topk(topk).indices

    ref_out, ref_lse = _attention_from_indices(q, k, v, global_topk)

    local_outs = []
    local_lses = []
    local_topks = []
    for rank in range(world):
        owned = [
            pos for pos in range(max_seq_len) if (pos // interleave) % world == rank
        ]
        k_local = k[owned]
        v_local = v[owned]
        local_topk = _global_to_local_indices(global_topk, rank, world, interleave)
        local_topks.append(local_topk)
        local_out, local_lse = _attention_from_indices(q, k_local, v_local, local_topk)
        local_outs.append(local_out)
        local_lses.append(local_lse)

    dcp_out, dcp_lse = _dcp_lse_merge(local_outs, local_lses)

    torch.testing.assert_close(dcp_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dcp_lse, ref_lse, atol=1e-5, rtol=1e-5)

    gathered_global = torch.cat(
        [
            _local_to_global_indices(local_topks[rank], rank, world, interleave)
            for rank in range(world)
        ],
        dim=1,
    )
    assert set(gathered_global[gathered_global >= 0].tolist()) == set(
        global_topk.flatten().tolist()
    )


def test_local_topk_union_is_not_equivalent_to_global_topk_attention():
    world = 2
    interleave = 1
    topk = 2
    q = torch.tensor([[1.0]])
    k = torch.tensor(
        [
            [1.00],
            [0.90],
            [0.95],
            [0.85],
        ]
    )
    v = torch.tensor([[0.0], [1000.0], [0.0], [1000.0]])

    scores = q @ k.T
    global_topk = scores.topk(topk, dim=-1).indices
    ref_out, ref_lse = _attention_from_indices(q, k, v, global_topk)

    local_topks = []
    for rank in range(world):
        owned = [
            pos for pos in range(k.shape[0]) if (pos // interleave) % world == rank
        ]
        local_topks.append(scores[:, owned].topk(topk, dim=-1).indices)

    local_union_out, local_union_lse = _dcp_attention_from_local_topks(
        q, k, v, local_topks, world, interleave
    )

    assert not torch.allclose(local_union_out, ref_out)
    assert not torch.allclose(local_union_lse, ref_lse)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_sparse_decode_dcp_persistent_topk_matches_non_dcp():
    torch.manual_seed(3)
    device = torch.device("cuda")
    world = 2
    interleave = 1
    topk = 512
    num_rows = 2
    max_seq_len = 1025
    head_dim = 16

    q = torch.randn(num_rows, head_dim, device=device)
    k = torch.randn(max_seq_len, head_dim, device=device)
    v = torch.randn(max_seq_len, head_dim, device=device)
    logits = q @ k.T
    seq_lens = torch.tensor([[1024], [1025]], dtype=torch.int32, device=device)

    non_dcp_topk = torch.empty((num_rows, topk), dtype=torch.int64, device=device)
    for row, seq_len in enumerate(seq_lens.flatten().tolist()):
        non_dcp_topk[row] = logits[row, :seq_len].topk(topk).indices
    ref_out, ref_lse = _attention_from_indices(q, k, v, non_dcp_topk)

    local_logits = []
    local_topks = []
    for rank in range(world):
        owned = [
            pos for pos in range(max_seq_len) if (pos // interleave) % world == rank
        ]
        rank_logits = logits[:, owned].contiguous()
        rank_seq_lens = get_dcp_local_seq_lens(
            seq_lens, world, rank, interleave
        ).contiguous()
        local_logits.append(rank_logits)
        local_topks.append(
            _run_persistent_topk(
                rank_logits,
                rank_seq_lens,
                topk,
                max_seq_len=rank_logits.shape[1],
            )
        )

    merged_global_topks = _merge_local_topks_global_with_fake_dcp(
        local_logits, local_topks, topk, world, interleave
    )
    # The radix top-K kernel selects a deterministic SET but writes it in
    # nondeterministic (atomicAdd) order; the production path is permutation-
    # invariant (compaction + softmax), so all ranks must agree on the set, not
    # the array order. (The fp64 fallback happens to return sorted order.)
    ref_topk = merged_global_topks[0]
    for rank_topk in merged_global_topks[1:]:
        for row in range(rank_topk.shape[0]):
            assert set(rank_topk[row].tolist()) == set(ref_topk[row].tolist())

    local_outs = []
    local_lses = []
    for rank, global_topk in enumerate(merged_global_topks):
        owned = [
            pos for pos in range(max_seq_len) if (pos // interleave) % world == rank
        ]
        local_topk = _global_to_local_indices(
            global_topk.to(torch.int64),
            rank,
            world,
            interleave,
        )
        local_out, local_lse = _attention_from_indices(
            q, k[owned], v[owned], local_topk
        )
        local_outs.append(local_out)
        local_lses.append(local_lse)

    dcp_out, dcp_lse = _dcp_lse_merge(local_outs, local_lses)
    torch.testing.assert_close(dcp_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dcp_lse, ref_lse, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not has_cutedsl(),
    reason="This test requires CUDA and CuteDSL",
)
@pytest.mark.parametrize("use_row_starts", [False, True])
def test_cutedsl_dcp_candidate_pack_and_select_matches_reference(
    use_row_starts: bool,
):
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )

    torch.manual_seed(13)
    device = torch.device("cuda")
    rows = 4
    valid_width = 1024
    width = valid_width + (8 if use_row_starts else 0)
    topk = 512
    world = 2
    row_starts = (
        torch.tensor([0, 2, 4, 1], device=device, dtype=torch.int32)
        if use_row_starts
        else None
    )
    row_offsets = (
        row_starts
        if row_starts is not None
        else torch.zeros(rows, device=device, dtype=torch.int32)
    )

    packed_by_rank = []
    for rank in range(world):
        logits = torch.randn((rows, width), device=device, dtype=torch.float32)
        local_topks = []
        for row in range(rows):
            start = int(row_offsets[row].item())
            local_topks.append(
                logits[row, start : start + valid_width].topk(topk).indices
            )
        topk_indices = torch.stack(local_topks).to(torch.int32)

        packed = torch.empty((rows, topk, 2), device=device, dtype=torch.float32)
        pack_dcp_topk_candidates_cutedsl(
            logits,
            topk_indices,
            packed,
            rank,
            world,
            1,
            row_starts,
        )

        expected_scores = logits.gather(
            1, topk_indices.to(torch.long) + row_offsets.to(torch.long).view(-1, 1)
        )
        expected_ids = (topk_indices * world + rank).to(torch.float32)
        torch.testing.assert_close(packed[..., 0], expected_scores)
        torch.testing.assert_close(packed[..., 1], expected_ids)
        packed_by_rank.append(packed)

    gathered = torch.cat(packed_by_rank, dim=1).contiguous()
    actual = torch.empty((rows, topk), device=device, dtype=torch.int32)
    returned = stable_topk_from_gathered_candidates_cutedsl(gathered, topk, out=actual)
    assert returned is actual
    expected = _ref_stable_topk_from_candidates_fp64(
        gathered[..., 0],
        gathered[..., 1].to(torch.int32),
        topk,
    )

    for row in range(rows):
        assert set(actual[row].cpu().tolist()) == set(expected[row].cpu().tolist())


@pytest.mark.skipif(
    not current_platform.is_cuda() or not has_cutedsl(),
    reason="This test requires CUDA and CuteDSL",
)
@pytest.mark.parametrize("num_positions", [16384, 131072, 1 << 20])
def test_cutedsl_gathered_select_compact_keys_match_reference(num_positions: int):
    """BF16-valued scores (upcast SM100 MQA logits) tie at the top-k boundary.
    Keying only their 16 high bits and the id bits below ``num_positions``
    must select the same set as the full 64-bit key."""
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        dcp_topk_id_bits,
        stable_topk_from_gathered_candidates_cutedsl,
    )

    torch.manual_seed(7)
    device = torch.device("cuda")
    rows, world, topk = 5, 8, 2048
    num_candidates = world * topk
    scores = torch.randn((rows, num_candidates), device=device)
    # Row 1: 64 distinct values, so the boundary tie spans hundreds of ids.
    scores[1] = (scores[1] * 8).round().clamp(-32, 31)
    # Row 2: all negative. Row 3: padding mixed in. Row 4: all tied.
    scores[2] = -scores[2].abs()
    scores[4] = 0.5
    scores = scores.bfloat16().float()
    ids = torch.stack(
        [
            torch.randperm(num_positions, device=device)[:num_candidates]
            for _ in range(rows)
        ]
    ).to(torch.int32)
    ids[:, 0] = num_positions - 1
    ids[3, ::3] = -1
    gathered = torch.stack((scores, ids.float()), dim=-1).contiguous()

    expected = _ref_stable_topk_from_candidates_fp64(scores, ids, topk)
    full = torch.empty((rows, topk), device=device, dtype=torch.int32)
    stable_topk_from_gathered_candidates_cutedsl(gathered, topk, out=full)
    compact = torch.empty_like(full)
    stable_topk_from_gathered_candidates_cutedsl(
        gathered,
        topk,
        out=compact,
        score_bits=16,
        id_bits=dcp_topk_id_bits(num_positions),
    )
    for row in range(rows):
        want = sorted(expected[row].tolist())
        assert sorted(full[row].tolist()) == want
        assert sorted(compact[row].tolist()) == want


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("interleave", [1, 64])
def test_sparse_prefill_dcp_metadata_host_extents_without_sync(
    world: int, interleave: int
):
    """The DCP chunk extents come from the CPU lengths, without a device sync,
    and equal the device-side per-rank KV lengths, on random chunked,
    prefix-hit and mixed prefill batches of 1-32 requests."""
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(0)
    for trial in range(24):
        num_reqs = int(torch.randint(1, 33, (1,), generator=gen))
        query_lens = torch.randint(1, 600, (num_reqs,), generator=gen)
        # Prefix hits (context before the chunk), some long, some none.
        context = torch.randint(0, 200_000, (num_reqs,), generator=gen)
        context[torch.rand(num_reqs, generator=gen) < 0.3] = 0
        seq_lens_cpu = (context + query_lens).to(torch.int32)
        query_start_loc_cpu = torch.zeros(num_reqs + 1, dtype=torch.int32)
        query_start_loc_cpu[1:] = torch.cumsum(query_lens, 0)
        start = int(torch.randint(0, num_reqs, (1,), generator=gen))
        end = int(torch.randint(start + 1, num_reqs + 1, (1,), generator=gen))
        query_start_loc = query_start_loc_cpu.to(device)
        seq_lens = seq_lens_cpu.to(device)
        block_table = torch.zeros((num_reqs, 1), dtype=torch.int32, device=device)
        kwargs = dict(
            start_idx=start,
            end_idx=end,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            uncompressed_seq_lens=seq_lens,
            compressed_seq_lens=seq_lens,
            compressed_seq_lens_cpu=seq_lens_cpu,
            block_table=block_table,
            compress_ratio=1,
            dcp_world_size=world,
            cp_kv_cache_interleave_size=interleave,
        )

        if trial == 0:
            # JIT-compiles the metadata kernel outside the sync check.
            build_prefill_chunk_metadata(dcp_rank=0, **kwargs)
        for rank in range(world):
            torch.accelerator.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                chunk = build_prefill_chunk_metadata(dcp_rank=rank, **kwargs)
            finally:
                torch.cuda.set_sync_debug_mode(0)
            assert chunk is not None
            local = get_dcp_local_seq_lens(seq_lens[start:end], world, None, interleave)
            assert chunk.local_total_seq_lens == int(local[:, rank].sum()), trial
            assert chunk.max_local_total_seq_lens == int(local.sum(0).max()), trial
            assert chunk.local_cu_seq_lens is not None
            assert int(chunk.local_cu_seq_lens[-1]) == chunk.local_total_seq_lens


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_sparse_prefill_dcp_metadata_localizes_causal_bounds():
    device = torch.device("cuda")
    seq_len = 8

    query_start_loc = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    query_start_loc_cpu = torch.tensor([0, seq_len], dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int32)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)

    def build(dcp_world_size, dcp_rank, interleave=1):
        chunk = build_prefill_chunk_metadata(
            start_idx=0,
            end_idx=1,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            uncompressed_seq_lens=seq_lens,
            compressed_seq_lens=seq_lens,
            compressed_seq_lens_cpu=seq_lens_cpu,
            block_table=block_table,
            compress_ratio=1,
            dcp_rank=dcp_rank,
            dcp_world_size=dcp_world_size,
            cp_kv_cache_interleave_size=interleave,
        )
        assert chunk is not None
        torch.accelerator.synchronize()
        return chunk

    # Non-DCP: local_cu_seq_lens aliases the global cu_seq_lens, and
    # cu_seqlen_ks/ke carry the global causal bounds.
    chunk = build(dcp_world_size=1, dcp_rank=0)
    assert chunk.local_cu_seq_lens is chunk.cu_seq_lens
    assert chunk.max_seqlen_k == seq_len
    torch.testing.assert_close(
        chunk.cu_seqlen_ks.cpu(),
        torch.zeros(seq_len, dtype=torch.int32),
    )
    torch.testing.assert_close(
        chunk.cu_seqlen_ke.cpu(),
        torch.arange(1, seq_len + 1, dtype=torch.int32),
    )

    # DCP: cu_seqlen_ks/ke are localized in place to this rank's shard.
    chunk = build(dcp_world_size=4, dcp_rank=0)
    assert chunk.local_cu_seq_lens is not None
    torch.testing.assert_close(
        chunk.local_cu_seq_lens.cpu(),
        torch.tensor([0, 2], dtype=torch.int32),
    )
    # DeepGEMM writes logits only as wide as this rank's shard.
    assert chunk.max_seqlen_k == 2
    torch.testing.assert_close(
        chunk.cu_seqlen_ks.cpu(),
        torch.zeros(seq_len, dtype=torch.int32),
    )
    torch.testing.assert_close(
        chunk.cu_seqlen_ke.cpu(),
        torch.tensor([1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.int32),
    )

    # DCP with interleave=2: per-token causal bounds localize differently from
    # interleave=1 (groups of 2 consecutive tokens are owned together). For
    # world=4, rank=0, K=2, per-token global len L=1..8 -> local len
    # [1,2,2,2,2,2,2,2] (matches get_dcp_local_seq_lens).
    chunk = build(dcp_world_size=4, dcp_rank=0, interleave=2)
    assert chunk.local_cu_seq_lens is not None
    torch.testing.assert_close(
        chunk.cu_seqlen_ks.cpu(),
        torch.zeros(seq_len, dtype=torch.int32),
    )
    torch.testing.assert_close(
        chunk.cu_seqlen_ke.cpu(),
        torch.tensor([1, 2, 2, 2, 2, 2, 2, 2], dtype=torch.int32),
    )


@pytest.mark.parametrize("interleave", [1, 64])
def test_dcp_prefill_chunks_budget_logits_on_the_local_shard(interleave: int):
    """Under DCP each rank's logits span only its KV shard, so the prefill
    chunker budgets the largest shard (rank 0's): a 16K chunk at 128K context
    takes 3 sub-chunks at DCP8 instead of 19, the same on every rank."""
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )

    def plan(dcp_world_size: int) -> list[tuple[slice, slice]]:
        builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
        builder.dcp_world_size = dcp_world_size
        builder.compressed_cp_interleave_size = interleave
        return builder._split_indexer_prefill_chunks(
            builder._prefill_split_seq_lens(torch.tensor([147_456])),
            torch.tensor([16_384]),
            1 << 30,
            512 << 20,
        )

    assert len(plan(1)) == 19
    local = get_dcp_local_seq_lens(torch.tensor([147_456]), 8, 0, interleave)
    max_rows = (512 << 20) // 4 // int(local[0])
    assert [q.stop - q.start for _, q in plan(8)] == [max_rows] * 2 + [
        16_384 - 2 * max_rows
    ]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("block_stride_rows", [None, 12])
def test_dcp_filter_compacts_valid_slots_for_sparse_kernel(
    block_stride_rows: int | None,
):
    block_size = 4
    num_topk = 128
    dcp_size = 2
    req_id = torch.zeros(1, dtype=torch.int32, device="cuda")
    token_indices = torch.full((1, num_topk), -1, dtype=torch.int32, device="cuda")
    token_indices[0, :8] = torch.arange(8, dtype=torch.int32, device="cuda")
    block_table = torch.tensor([[10]], dtype=torch.int32, device="cuda")

    out, valid_counts = triton_filter_and_convert_dcp_index(
        req_id,
        block_table,
        token_indices,
        dcp_size=dcp_size,
        dcp_rank=0,
        BLOCK_SIZE=block_size,
        BLOCK_STRIDE_ROWS=block_stride_rows,
        NUM_TOPK_TOKENS=num_topk,
        return_valid_counts=True,
    )

    valid = int(valid_counts.item())
    assert valid == 4
    assert (out[0, :valid] >= 0).all()
    assert (out[0, valid:] == -1).all()
    # In-kernel compaction packs valid slots to the front; prefix order is
    # unspecified, so compare as a set.
    block_stride_rows = block_stride_rows or block_size
    expected_base = 10 * block_stride_rows
    assert set(out[0, :valid].cpu().tolist()) == set(
        range(expected_base, expected_base + block_size)
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("interleave", [1, 2])
@pytest.mark.parametrize("dcp_rank", [0, 1])
# Include padded single-tile compaction and the multi-tile atomic fallback.
@pytest.mark.parametrize("num_topk", [1024, 2176, 4224])
def test_dcp_filter_compaction_matches_reference(
    interleave: int, dcp_rank: int, num_topk: int
):
    """In-kernel compaction must, for every row, produce exactly the rank-owned
    physical slots packed into [0, valid_count) with -1 in the tail -- the same
    SET a reference filter + sort/gather produces. Uses wide rows (> BLOCK_N
    valid slots), with interior -1 gaps."""
    device = torch.device("cuda")
    torch.manual_seed(7)
    dcp_size = 2
    block_size = 8
    num_rows = 5
    max_blocks = 64
    seq = max_blocks * block_size

    req_id = torch.randint(0, 3, (num_rows,), dtype=torch.int32, device=device)
    block_table = torch.randint(
        0, 1000, (3, max_blocks), dtype=torch.int32, device=device
    )
    # Each row: a dense valid prefix of distinct global token ids, then -1 pad.
    token_indices = torch.full(
        (num_rows, num_topk), -1, dtype=torch.int32, device=device
    )
    for r in range(num_rows):
        # Ids are distinct, so a row holds at most `seq` valid entries.
        hi = min(num_topk * 3 // 4, seq)
        n_valid = int(torch.randint(hi // 3, hi, (1,)).item())
        perm = torch.randperm(seq, device=device)[:n_valid].to(torch.int32)
        token_indices[r, :n_valid] = perm

    out, valid_counts = triton_filter_and_convert_dcp_index(
        req_id,
        block_table,
        token_indices,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=interleave,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk,
        return_valid_counts=True,
    )

    for r in range(num_rows):
        toks = token_indices[r]
        toks = toks[toks >= 0]
        owner = (toks // interleave) % dcp_size
        owned = toks[owner == dcp_rank]
        local = (owned // (dcp_size * interleave)) * interleave + owned % interleave
        blk = local // block_size
        off = local % block_size
        expected = (
            block_table[int(req_id[r].item()), blk].to(torch.int64) * block_size + off
        )
        n = int(valid_counts[r].item())
        assert n == owned.numel()
        assert (out[r, :n] >= 0).all()
        assert (out[r, n:] == -1).all()
        assert set(out[r, :n].cpu().tolist()) == set(expected.cpu().tolist())


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("interleave", [1, 2, 4])
def test_dcp_global_topk_physical_attention_matches_non_dcp(interleave: int):
    torch.manual_seed(2)
    device = torch.device("cuda")
    dcp_size = 2
    block_size = 4
    num_topk = 128
    selected_k = 8
    seq_len = 16
    head_dim = 16
    num_queries = 3

    q = torch.randn(num_queries, head_dim, device=device)
    k_global = torch.randn(seq_len, head_dim, device=device)
    v_global = torch.randn(seq_len, head_dim, device=device)
    scores = q @ k_global.T
    global_topk = torch.full(
        (num_queries, num_topk), -1, dtype=torch.int32, device=device
    )
    global_topk[:, :selected_k] = scores.topk(selected_k, dim=-1).indices.to(
        torch.int32
    )

    ref_out, ref_lse = _attention_from_indices(
        q, k_global, v_global, global_topk[:, :selected_k].to(torch.int64)
    )

    local_outs = []
    local_lses = []
    for rank in range(dcp_size):
        block_ids = torch.tensor(
            [[rank * 10 + 1, rank * 10 + 2]], dtype=torch.int32, device=device
        )
        num_slots = int((block_ids.max().item() + 1) * block_size)
        k_cache = torch.zeros(num_slots, head_dim, device=device)
        v_cache = torch.zeros(num_slots, head_dim, device=device)
        for global_idx in range(seq_len):
            if (global_idx // interleave) % dcp_size != rank:
                continue
            local_idx = (
                global_idx // (dcp_size * interleave)
            ) * interleave + global_idx % interleave
            block = local_idx // block_size
            offset = local_idx % block_size
            slot = int(block_ids[0, block].item()) * block_size + offset
            k_cache[slot] = k_global[global_idx]
            v_cache[slot] = v_global[global_idx]

        slots, valid_counts = triton_filter_and_convert_dcp_index(
            torch.zeros(num_queries, dtype=torch.int32, device=device),
            block_ids,
            global_topk,
            dcp_size=dcp_size,
            dcp_rank=rank,
            cp_kv_cache_interleave_size=interleave,
            BLOCK_SIZE=block_size,
            NUM_TOPK_TOKENS=num_topk,
            return_valid_counts=True,
        )
        row_ids = torch.arange(num_queries, device=device)
        assert (slots[row_ids, valid_counts] == -1).all()
        local_out, local_lse = _attention_from_indices(
            q, k_cache, v_cache, slots.to(torch.int64)
        )
        local_outs.append(local_out)
        local_lses.append(local_lse)

    dcp_out, dcp_lse = _dcp_lse_merge(local_outs, local_lses)
    torch.testing.assert_close(dcp_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dcp_lse, ref_lse, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("is_lse_base_on_e", [True, False])
def test_correct_attn_out_zeroes_empty_nan_partial(is_lse_base_on_e: bool):
    out = torch.full((1, 1, 4), float("nan"), device="cuda")
    lses = torch.tensor(
        [[[0.0]], [[float("-inf")]]],
        dtype=torch.float32,
        device="cuda",
    )

    corrected, final_lse = correct_attn_out(
        out,
        lses,
        cp_rank=1,
        ctx=CPTritonContext(),
        is_lse_base_on_e=is_lse_base_on_e,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(corrected, torch.zeros_like(corrected))
    torch.testing.assert_close(final_lse, torch.zeros_like(final_lse))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_decode_topk_pads_surplus_with_negative_one():
    """When a row's valid length < topk the top-k kernel must pad the surplus
    slots with -1: the DCP merge (`topk_indices >= 0`) and
    `triton_filter_and_convert_dcp_index` (`tok < 0`) both treat <0 as invalid,
    so a non-(-1) pad would be silently attended. This is the common case under
    DCP, where each rank's local seq_len = global / world is usually << topk.
    Checked on the *set* of valid indices (the kernel may order them by score,
    not position)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    topk = 8
    next_n = 1
    seq_lens_list = [0, 3, 5, 8]
    num_rows = len(seq_lens_list)
    logits = torch.randn(num_rows, 16, device=device)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device).view(
        num_rows, next_n
    )
    idx = _run_decode_topk(logits, seq_lens, next_n, topk)
    for r, sl in enumerate(seq_lens_list):
        valid = idx[r][idx[r] >= 0]
        # seq_len <= topk, so top-k selects exactly the whole valid range.
        assert valid.numel() == min(sl, topk)
        assert set(valid.tolist()) == set(range(sl))
        assert (idx[r] == -1).sum().item() == topk - min(sl, topk)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_persistent_topk_pads_surplus_with_negative_one():
    """Same surplus=-1 invariant for the persistent_topk kernel (k>=512)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    topk = 512  # persistent_topk requires k in {512, 1024, 2048}
    seq_lens_list = [100, 300, 512]
    num_rows = len(seq_lens_list)
    max_seq_len = 600
    logits = torch.randn(num_rows, max_seq_len, device=device)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device).view(
        num_rows, 1
    )
    idx = _run_persistent_topk(logits, seq_lens, topk, max_seq_len)
    for r, sl in enumerate(seq_lens_list):
        valid = idx[r][idx[r] >= 0]
        assert valid.numel() == min(sl, topk)
        assert set(valid.tolist()) == set(range(sl))
        assert (idx[r] == -1).sum().item() == topk - min(sl, topk)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_sparse_decode_dcp_short_context_matches_non_dcp():
    """End-to-end DCP decode where the global seq_len < topk (so every rank's
    local top-k is surplus-padded). Exercises the kernel surplus -> merge mask
    -> global top-k -> physical localize -> LSE merge chain for the common
    short-context decode case, vs the non-DCP reference."""
    torch.manual_seed(4)
    device = torch.device("cuda")
    world = 2
    interleave = 1
    topk = 512
    num_rows = 2
    max_seq_len = 300  # < topk -> surplus everywhere
    head_dim = 16

    q = torch.randn(num_rows, head_dim, device=device)
    k = torch.randn(max_seq_len, head_dim, device=device)
    v = torch.randn(max_seq_len, head_dim, device=device)
    logits = q @ k.T
    seq_lens = torch.tensor([[250], [300]], dtype=torch.int32, device=device)

    non_dcp_topk = torch.empty((num_rows, topk), dtype=torch.int64, device=device)
    for row, seq_len in enumerate(seq_lens.flatten().tolist()):
        sel = logits[row, :seq_len].topk(min(topk, seq_len)).indices
        non_dcp_topk[row, : sel.numel()] = sel
        non_dcp_topk[row, sel.numel() :] = -1
    ref_out, ref_lse = _attention_from_indices(q, k, v, non_dcp_topk)

    local_logits = []
    local_topks = []
    for rank in range(world):
        owned = [p for p in range(max_seq_len) if (p // interleave) % world == rank]
        rank_logits = logits[:, owned].contiguous()
        rank_seq_lens = get_dcp_local_seq_lens(seq_lens, world, rank, interleave)
        local_logits.append(rank_logits)
        local_topks.append(
            _run_persistent_topk(
                rank_logits,
                rank_seq_lens.contiguous(),
                topk,
                max_seq_len=rank_logits.shape[1],
            )
        )

    merged_global_topks = _merge_local_topks_global_with_fake_dcp(
        local_logits, local_topks, topk, world, interleave
    )
    # The radix top-K kernel selects a deterministic SET but writes it in
    # nondeterministic (atomicAdd) order; the production path is permutation-
    # invariant (compaction + softmax), so all ranks must agree on the set, not
    # the array order. (The fp64 fallback happens to return sorted order.)
    ref_topk = merged_global_topks[0]
    for rank_topk in merged_global_topks[1:]:
        for row in range(rank_topk.shape[0]):
            assert set(rank_topk[row].tolist()) == set(ref_topk[row].tolist())

    local_outs = []
    local_lses = []
    for rank, global_topk in enumerate(merged_global_topks):
        owned = [p for p in range(max_seq_len) if (p // interleave) % world == rank]
        local_topk = _global_to_local_indices(
            global_topk.to(torch.int64), rank, world, interleave
        )
        local_out, local_lse = _attention_from_indices(
            q, k[owned], v[owned], local_topk
        )
        local_outs.append(local_out)
        local_lses.append(local_lse)

    dcp_out, dcp_lse = _dcp_lse_merge(local_outs, local_lses)
    torch.testing.assert_close(dcp_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dcp_lse, ref_lse, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dcp_world_size", [2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 64])
@pytest.mark.parametrize("rows_per_req", [1, 2])
@pytest.mark.parametrize(
    "req_lens", [[7], [8, 8], [1, 9], [63, 64, 65], [255, 256, 257], [256, 1, 2730]]
)
def test_pcp_plan_deinterleave_restores_global_order(
    monkeypatch, dcp_world_size, interleave, rows_per_req, req_lens
):
    """Gathering block-interleaved shards must restore each request's token order."""
    from vllm.v1.attention.backends.mla.indexer import build_pcp_global_chunk_plan

    monkeypatch.setattr("vllm.v1.attention.backends.mla.indexer.PIN_MEMORY", False)
    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.indexer.async_tensor_h2d",
        lambda x, device: x.to(device),
    )
    scheduled = np.array(req_lens, dtype=np.int64)
    shard_rows = np.array(
        [
            max(
                _local_count(g, r, dcp_world_size, interleave)
                for r in range(dcp_world_size)
            )
            for g in req_lens
        ]
    )
    rows = np.repeat(np.arange(len(scheduled)), rows_per_req)
    plan = build_pcp_global_chunk_plan(
        rows, shard_rows[rows], dcp_world_size, torch.device("cpu"), interleave
    )

    starts = np.concatenate([[0], np.cumsum(scheduled)])

    # Build each rank's padded shard exactly as the cache gather would.
    padded_cu = plan.padded_local_cu[::rows_per_req].tolist()
    shards = torch.zeros(dcp_world_size, plan.padded_local_total, 1)
    for r in range(dcp_world_size):
        for i, g in enumerate(req_lens):
            owned = [t for t in range(g) if (t // interleave) % dcp_world_size == r]
            for local, t in enumerate(owned):
                shards[r, padded_cu[i] + local, 0] = starts[i] + t

    gathered = shards.reshape(dcp_world_size * plan.padded_local_total, 1)
    restored = gathered[plan.deinterleave_idx]
    # The layout is padded per request; only the real positions are read.
    row_start = plan.row_start_cu.tolist()
    for row, i in enumerate(rows):
        g = req_lens[i]
        torch.testing.assert_close(
            restored[row_start[row] : row_start[row] + g, 0],
            torch.arange(starts[i], starts[i] + g, dtype=torch.float32),
        )


@pytest.mark.skipif(
    not current_platform.is_cuda() or not has_cutedsl(),
    reason="This test requires CUDA and CuteDSL",
)
@pytest.mark.parametrize("use_row_starts", [False, True])
def test_cutedsl_dcp_candidate_pack_reads_bf16_logits(use_row_starts: bool):
    """SM100 MQA logits are BF16: packing them directly must give the same
    candidates, bit for bit, as packing their FP32 upcast."""
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
    )

    torch.manual_seed(17)
    device = torch.device("cuda")
    rows, width, topk, world, rank = 5, 1536, 512, 4, 3
    logits = torch.randn((rows, width), device=device).to(torch.bfloat16)
    row_starts = (
        torch.tensor([0, 7, 300, 1, 512], device=device, dtype=torch.int32)
        if use_row_starts
        else None
    )
    topk_indices = torch.stack(
        [torch.randperm(1024, device=device)[:topk] for _ in range(rows)]
    ).to(torch.int32)
    topk_indices[1, 100:] = -1

    packed = {}
    for dtype in (torch.bfloat16, torch.float32):
        packed[dtype] = torch.empty((rows, topk, 2), device=device)
        pack_dcp_topk_candidates_cutedsl(
            logits.to(dtype), topk_indices, packed[dtype], rank, world, 1, row_starts
        )
    assert torch.equal(
        packed[torch.bfloat16].view(torch.int32),
        packed[torch.float32].view(torch.int32),
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("interleave", [1, 2])
def test_dcp_merge_of_bf16_logits_matches_fp32(interleave: int):
    """The DCP merge scores BF16 logits as their exact FP32 upcast."""
    torch.manual_seed(19)
    device = torch.device("cuda")
    world, topk, rows, width = 4, 512, 3, 1200
    local_logits = [
        torch.randn((rows, width), device=device).to(torch.bfloat16)
        for _ in range(world)
    ]
    local_topks = [
        logits.float().topk(topk, dim=-1).indices.to(torch.int32)
        for logits in local_logits
    ]
    merged = {
        dtype: _merge_local_topks_global_with_fake_dcp(
            [logits.to(dtype) for logits in local_logits],
            local_topks,
            topk,
            world,
            interleave,
        )
        for dtype in (torch.bfloat16, torch.float32)
    }
    for bf16_topk, fp32_topk in zip(merged[torch.bfloat16], merged[torch.float32]):
        for row in range(rows):
            assert set(bf16_topk[row].tolist()) == set(fp32_topk[row].tolist())


@pytest.mark.parametrize("sm100", [True, False])
def test_dcp_candidate_pack_warms_bf16_logits_on_sm100(monkeypatch, sm100: bool):
    from types import SimpleNamespace

    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        PackDCPTopkCandidatesKernel,
    )

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=2048)),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=8, cp_kv_cache_interleave_size=1
        ),
    )
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: sm100
    )
    keys = PackDCPTopkCandidatesKernel().get_warmup_keys(vllm_config)
    dtypes = (torch.float32, torch.bfloat16) if sm100 else (torch.float32,)
    assert {(key.logits_dtype, key.has_row_starts) for key in keys} == {
        (dtype, has_row_starts) for dtype in dtypes for has_row_starts in (False, True)
    }


def _distinct_bf16_logits(rows: int, width: int, stride: int) -> torch.Tensor:
    """[rows, width] distinct positive BF16 values (consecutive bit patterns
    from 1.0, so every top-k set is unique) with the given row stride."""
    perms = torch.stack(
        [torch.randperm(stride, device="cuda") for _ in range(rows)]
    ).to(torch.int32)
    return (perms + 0x3F80).to(torch.int16).view(torch.bfloat16)[:, :width]


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="DeepSelect requires SM100",
)
@pytest.mark.parametrize(
    "width,stride_pad,topk_backend,uses_deep_select",
    [
        # DeepGEMM pads the row stride to 1 KiB: DeepSelect on BF16.
        (3072, 0, "auto", True),
        (300, 0, "auto", True),
        (3072, 0, "deep_select", True),
        # Another top-k backend, or a stride DeepSelect rejects: FP32 upcast
        # and the per-row kernel.
        (3072, 0, "persistent", False),
        (3072, 8, "auto", False),
    ],
)
def test_prefill_topk_on_bf16_logits(
    monkeypatch, width: int, stride_pad: int, topk_backend: str, uses_deep_select
):
    """The SM100 prefill top-k runs DeepSelect on the BF16 logits (no FP32
    copy) under the auto/deep_select backends and falls back to the upcast +
    per-row kernel otherwise; both select each row's top-k within its end and
    -1 pad rows shorter than k."""
    torch.manual_seed(23)
    device = torch.device("cuda")
    topk = 512
    row_ends_list = [e for e in (0, 1, 300, 512, 513, 1024, 3000) if e <= width]
    rows = len(row_ends_list)
    stride = -(-width // 512) * 512 + stride_pad
    logits = _distinct_bf16_logits(rows, width, stride)
    row_ends = torch.tensor(row_ends_list, dtype=torch.int32, device=device)
    topk_indices = torch.full((rows, topk), -7, dtype=torch.int32, device=device)

    calls: list[str] = []
    for name in ("deep_select_topk", "as_fp32_logits"):
        fn = getattr(sparse_indexer, name)
        monkeypatch.setattr(
            sparse_indexer,
            name,
            lambda *args, _fn=fn, _name=name, **kwargs: (
                calls.append(_name) or _fn(*args, **kwargs)
            ),
        )
    sparse_indexer._prefill_topk(
        logits, None, row_ends, topk_indices, topk, topk_backend
    )
    assert calls == (["deep_select_topk"] if uses_deep_select else ["as_fp32_logits"])

    for row, end in enumerate(row_ends_list):
        chosen = topk_indices[row]
        expected = logits[row, :end].float().topk(min(topk, end)).indices
        assert set(chosen[chosen >= 0].tolist()) == set(expected.tolist())
        assert (chosen == -1).sum().item() == topk - min(topk, end)


def _merge_gate_config(dcp: int, interleave: int, kv_transfer=None, explicit=True):
    from types import SimpleNamespace

    from vllm.config import KVTransferConfig

    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=interleave,
            dcp_kv_cache_interleave_size=1,
            _allow_auto_resolve_cp_interleave_size=not explicit,
            use_ubatching=False,
        ),
        kv_transfer_config=(
            KVTransferConfig(**kv_transfer) if kv_transfer is not None else None
        ),
        kernel_config=SimpleNamespace(
            sparse_indexer_topk_backend="deep_select", enable_jit_warmup=False
        ),
        model_config=SimpleNamespace(max_model_len=1 << 20),
    )


@pytest.mark.parametrize(
    ("dcp", "interleave", "kv_transfer", "explicit", "expected"),
    [
        (16, 1, None, False, 192),
        (8, 1, None, False, 384),
        (16, 1024, None, True, 0),
        (8, 1024, _NIXL_P, True, 0),
        (16, 1, _NIXL_D, True, 192),
        (16, 1, _NIXL_D, False, 0),  # NIXL raises it to the block size later
    ],
)
def test_dcp_topk_merge_construction_gate_follows_interleave(
    monkeypatch, dcp, interleave, kv_transfer, explicit, expected
):
    """The one-round decode merge (Inferact/mum#63) is set up only for an
    interleave that stays 1. Otherwise nothing is set up: no collective, no
    rescue workspace, no JIT registration."""
    import sys
    import types

    import vllm.v1.attention.ops.dcp_topk_merge as gate

    monkeypatch.setattr(
        gate,
        "current_platform",
        types.SimpleNamespace(
            is_cuda=lambda: True, is_device_capability_family=lambda cap: cap == 100
        ),
    )
    monkeypatch.setattr(gate, "has_cutedsl", lambda: True)
    monkeypatch.setitem(
        sys.modules, "vllm._deepselect_C", types.ModuleType("vllm._deepselect_C")
    )
    monkeypatch.setattr(gate, "_MERGE_INIT", {})
    monkeypatch.setattr(gate, "_MERGE_MAX_ROWS", {})
    config = _merge_gate_config(dcp, interleave, kv_transfer, explicit)
    assert gate.dcp_topk_merge_local_topk(config, 2048) == expected
    if not expected:

        def collective(*args, **kwargs):
            raise AssertionError("set up a merge that no step takes")

        monkeypatch.setattr(gate, "_init_rescue", collective)
        assert gate.init_dcp_topk_merge(config, 2048) == 0
        assert gate.dcp_topk_merge_max_rows(2048, 192) == 0


def test_dcp16_topk_merge_sizes():
    """DCP16 (the D instance): N = 192 candidates per rank in 198-word rows
    (12.7 KB per row from 16 ranks against 256 KB for the generic merge's
    (score, id) pairs), at most 10 saturated ranks per row, 1M-token contexts
    within the 17-bit local index."""
    from vllm.v1.attention.ops.dcp_topk_merge import (
        LOCAL_INDEX_BITS,
        MAX_WORLD_SIZE,
        dcp_merge_local_topk,
        packed_row_words,
        rescue_slots,
    )

    topk, world = 2048, 16
    local_topk = dcp_merge_local_topk(topk, world)
    assert local_topk == 192
    assert packed_row_words(local_topk) == 198
    assert world * packed_row_words(local_topk) * 4 == 12_672
    assert world * topk * 8 == 262_144
    assert rescue_slots(topk, local_topk) == 10
    assert world <= MAX_WORLD_SIZE
    assert -(-(1 << 20) // world) <= 1 << LOCAL_INDEX_BITS
    assert dcp_merge_local_topk(topk, 8) == 384
    assert rescue_slots(topk, 384) == 5


@pytest.mark.parametrize("interleave", [1, 1024])
@pytest.mark.parametrize("num_rows", [8, 256, 257])
def test_dcp_decode_step_takes_merge_at_interleave_one(
    monkeypatch, interleave, num_rows
):
    """A decode step takes the one-round merge only at interleave 1 and when
    its rows fit the rescue workspace (256 rows here); the others keep the
    generic merge."""
    import vllm.v1.attention.ops.dcp_topk_merge as gate

    monkeypatch.setattr(gate, "_MERGE_MAX_ROWS", {(2048, 192): 256})
    got = sparse_indexer._use_dcp_decode_topk_merge(
        192, True, interleave, False, num_rows, 2048
    )
    assert got is (interleave == 1 and num_rows <= 256)


@pytest.mark.parametrize("fp32_decode_logits", [False, True])
@pytest.mark.parametrize("logits_dtype", [torch.bfloat16, torch.float32])
def test_decode_logits_profile_bytes_without_merge(fp32_decode_logits, logits_dtype):
    """Without the one-round merge (interleave 1024) the memory profile
    reserves what it did before: the logits plus their FP32 copy when the
    top-k reads one."""
    rows, max_model_len, row_align = 128 * 8, 1 << 20, 384
    stride = -(-max_model_len // row_align) * row_align
    copy = 4 if fp32_decode_logits and logits_dtype != torch.float32 else 0
    old = rows * stride * (logits_dtype.itemsize + copy)
    got = sparse_indexer._decode_logits_profile_bytes(
        rows, max_model_len, logits_dtype, row_align, fp32_decode_logits, 0, 16
    )
    assert got == old


class _ThreadedDCP:
    """Simulated DCP ranks as threads; all_gather stacks every rank's input."""

    def __init__(self, world: int) -> None:
        import threading

        self.world = world
        self.barrier = threading.Barrier(world)
        self.inputs: list[torch.Tensor | None] = [None] * world

    def all_gather(self, rank: int):
        def gather(tensor: torch.Tensor) -> torch.Tensor:
            self.inputs[rank] = tensor.clone()
            self.barrier.wait()
            out = torch.stack(self.inputs)  # type: ignore[arg-type]
            self.barrier.wait()
            return out

        return gather

    def run(self, fn) -> list:
        import threading

        results: list = [None] * self.world
        errors: list[BaseException] = []

        def target(rank: int) -> None:
            try:
                results[rank] = fn(rank)
            except BaseException as error:
                errors.append(error)
                self.barrier.abort()

        threads = [
            threading.Thread(target=target, args=(r,)) for r in range(self.world)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        return results


def _check_case(world: int = 4, topk: int = 64, width: int = 256):
    """Global bf16 scores with many ties, per-rank logits at interleave 1 and
    the exact top-K ids (score desc, id asc) of each row; rows:
    0 exact, 1 swaps a tie at tau, 2 loses a token above tau, 3 drops an id,
    4 repeats an id, 5 holds a NaN, 6 is empty (a padded row)."""
    gen = torch.Generator().manual_seed(0)
    lengths = [world * width, 1000, 900, 777, 1024, 500, 0]
    scores = (
        torch.randint(-40, 40, (len(lengths), world * width), generator=gen) / 8
    ).to(torch.bfloat16)
    reference = torch.full((len(lengths), topk), -1, dtype=torch.int32)
    for row, length in enumerate(lengths):
        values = scores[row, :length].float().tolist()
        order = sorted(range(length), key=lambda g: (-values[g], g))[:topk]
        reference[row, : len(order)] = torch.tensor(order, dtype=torch.int32)
    merged = reference.flip(1).clone()  # any order
    value = scores.float()

    def tau(row: int) -> float:
        return value[row, reference[row][reference[row] >= 0].long()].min().item()

    chosen = set(reference[1].tolist())
    tie_in = next(
        g for g in range(lengths[1]) if value[1, g] == tau(1) and g not in chosen
    )
    tie_out = int(reference[1][value[1, reference[1].long()] == tau(1)][0])
    merged[1][merged[1] == tie_out] = tie_in
    chosen = set(reference[2].tolist())
    low = next(g for g in range(lengths[2]) if value[2, g] < tau(2))
    high = int(reference[2][value[2, reference[2].long()] > tau(2)][0])
    merged[2][merged[2] == high] = low
    merged[3][0] = -1
    merged[4][0] = merged[4][1]
    scores[5, 3] = float("nan")
    logits = [scores[:, rank::world].contiguous() for rank in range(world)]
    row_ends = [
        torch.tensor(
            [max(length - rank + world - 1, 0) // world for length in lengths],
            dtype=torch.int32,
        )
        for rank in range(world)
    ]
    return logits, row_ends, merged, reference


def test_dcp_topk_merge_check_classifies_rows():
    """VLLM_DSA_BPRIME_CHECK passes rows whose ids match the reference merge's
    or differ only among tokens scoring exactly tau, and flags every other
    difference on all ranks; NaN and empty rows are counted apart."""
    from vllm.v1.attention.ops.dcp_topk_merge.check import compare_dcp_topk_merge

    world = 4
    logits, row_ends, merged, reference = _check_case(world)
    dcp = _ThreadedDCP(world)
    results = dcp.run(
        lambda rank: compare_dcp_topk_merge(
            logits[rank],
            row_ends[rank],
            merged,
            reference,
            rank,
            world,
            dcp.all_gather(rank),
        )
    )
    for got in results:
        assert got["miss"].tolist() == [False, False, True, True, True, False, False]
        assert got["tie"].tolist() == [False, True, False, False, False, False, False]
        assert got["nan"].tolist() == [False] * 5 + [True, False]
        assert got["valid"].tolist() == [True] * 6 + [False]
        assert got["missing"].tolist()[:4] == [0, 1, 1, 1]


def test_dcp_topk_merge_check_ties_signed_zeros():
    """bf16 -0.0 and +0.0 tie at tau: DeepSelect orders them as equal, so a
    merge that keeps -0.0 tokens where the reference keeps +0.0 ones differs
    only at ties (row 0), while losing a token above tau is still a miss
    (row 1)."""
    from vllm.v1.attention.ops.dcp_topk_merge.check import compare_dcp_topk_merge

    world, width = 2, 32
    scores = torch.full((2, world * width), -0.0, dtype=torch.bfloat16)
    scores[:, :4] = 1.0
    scores[:, 4:12] = 0.0
    reference = torch.arange(8, dtype=torch.int32).repeat(2, 1)
    merged = torch.tensor(
        [[0, 1, 2, 3, 20, 21, 22, 23], [0, 1, 2, 20, 4, 5, 6, 7]], dtype=torch.int32
    )
    logits = [scores[:, rank::world].contiguous() for rank in range(world)]
    row_ends = torch.full((2,), width, dtype=torch.int32)
    dcp = _ThreadedDCP(world)
    results = dcp.run(
        lambda rank: compare_dcp_topk_merge(
            logits[rank], row_ends, merged, reference, rank, world, dcp.all_gather(rank)
        )
    )
    for got in results:
        assert got["miss"].tolist() == [False, True]
        assert got["tie"].tolist() == [True, False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned host counters")
def test_dcp_topk_merge_check_host_counters_under_cuda_default_device():
    """Model construction runs under a CUDA default-device context; the
    check's counters still live in pinned host memory."""
    from vllm.v1.attention.ops.dcp_topk_merge import check

    with torch.device("cuda"):
        state = check.DcpTopkMergeCheck(torch.device("cuda"))
    assert state.host.device.type == "cpu" and state.host.is_pinned()
    assert state.stats.device.type == "cuda"


def test_dcp_topk_merge_check_records_misses(monkeypatch):
    """The check's counters and its record of misses (layer, step, row), as
    DCP rank 0 logs them; only the first layer counts the passes."""
    from vllm.v1.attention.ops.dcp_topk_merge import check

    state = check.DcpTopkMergeCheck(torch.device("cpu"))
    monkeypatch.setattr(check, "_STATE", state)
    assert check.dcp_topk_merge_check_enabled()
    world = 4
    logits, row_ends, merged, reference = _check_case(world)
    sat_word = torch.zeros(merged.shape[0], dtype=torch.int32)
    sat_word[2] = (0b1001 << 16) | 0x3F80
    sat_word[0] = -(1 << 31) | (1 << 16)  # ranks 15 and 0
    dcp = _ThreadedDCP(world)
    for layer in ("layers.1.indexer", "layers.5.indexer", "layers.1.indexer"):
        lock = __import__("threading").Lock()

        def one_rank(rank: int, layer=layer, lock=lock) -> None:
            gather = dcp.all_gather(rank)
            rows = check.compare_dcp_topk_merge(
                logits[rank], row_ends[rank], merged, reference, rank, world, gather
            )
            if rank == 0:  # one recorder, as on each real rank
                with lock:
                    state.record(
                        state.layer_id(layer), rows, check.saturated_ranks(sat_word)
                    )

        dcp.run(one_rank)
    snap = state.snapshot()
    totals = snap["totals"]
    assert snap["steps"] == 2 and snap["layers"] == 2
    assert totals[check.CALLS] == 3 and totals[check.ROWS] == 18
    assert totals[check.MISS_ROWS] == 9 and totals[check.TIE_ROWS] == 3
    assert totals[check.NAN_ROWS] == 3 and totals[check.SAT_ROWS] == 6
    assert totals[check.MAX_SAT_RANKS] == 2 and totals[check.MAX_MISSING] == 1
    assert snap["misses"][:3] == [
        ("layers.1.indexer", 1, 2, 2, 1),
        ("layers.1.indexer", 1, 3, 0, 1),
        ("layers.1.indexer", 1, 4, 0, 1),
    ]
    assert snap["misses"][-1] == ("layers.1.indexer", 2, 4, 0, 1)
    state.log(final=True)
