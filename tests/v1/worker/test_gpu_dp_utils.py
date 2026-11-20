# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DP batch-descriptor sync, in particular the AND-reduced
allow_skip_drafts flag that lets speculators skip their draft phase
uniformly across DP ranks."""

from unittest import mock

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


def _run_sync(local_allow_skip: bool, other_allow_skip: bool) -> bool:
    """Run sync_cudagraph_and_dp_padding as dp rank 0 of 2, with rank 1's
    all_reduce contribution simulated."""

    def fake_all_reduce(tensor):
        tensor[0][1] = 8  # rank 1 num_tokens
        tensor[1][1] = CUDAGraphMode.NONE.value
        tensor[2][1] = 0  # uniform_token_count (0 means None)
        tensor[3][1] = -1  # max_query_len (-1 means None)
        tensor[4][1] = 0  # allow_ubatching
        tensor[5][1] = 2  # rank 1 num_reqs
        tensor[6][1] = int(other_allow_skip)

    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE, num_tokens=4, num_reqs=2
    )
    with mock.patch.object(dp_utils, "get_dp_group") as mock_group:
        mock_group.return_value.cpu_all_reduce_ = fake_all_reduce
        _, sync = dp_utils.sync_cudagraph_and_dp_padding(
            cudagraph_manager=None,
            desired_batch_desc=desc,
            num_tokens=4,
            num_reqs=2,
            uniform_token_count=None,
            dp_size=2,
            dp_rank=0,
            allow_skip_drafts=local_allow_skip,
        )
    assert sync is not None
    return sync.skip_drafts


def test_skip_drafts_requires_all_ranks():
    assert _run_sync(True, True)
    assert not _run_sync(True, False)
    assert not _run_sync(False, True)
    assert not _run_sync(False, False)


def test_dispatch_dp1_returns_no_sync():
    # With a single rank there is no agreement to make; callers fall back to
    # their local allow_skip_drafts value.
    _, sync = dp_utils.dispatch_cg_and_sync_dp(
        cudagraph_manager=None,
        num_reqs=2,
        num_tokens=4,
        uniform_token_count=None,
        dp_size=1,
        dp_rank=0,
        need_eager=True,
        allow_skip_drafts=True,
    )
    assert sync is None


def _run_piecewise_sync(counts, pad_policy):
    """sync_cudagraph_and_dp_padding as DP rank 0 of len(counts) ranks whose
    batches all dispatch to a PIECEWISE graph of the largest count rounded up
    to 256, with the given ModelState.pad_dp_for_piecewise hook."""
    dp = len(counts)
    padded = -(-max(counts) // 256) * 256

    def fake_all_reduce(tensor):
        for r in range(1, dp):
            tensor[0][r] = counts[r]
            tensor[1][r] = CUDAGraphMode.PIECEWISE.value
            tensor[2][r] = 0
            tensor[3][r] = -1
            tensor[4][r] = 0
            tensor[5][r] = 1
            tensor[6][r] = 0

    manager = mock.Mock()
    manager.dispatch.return_value = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=padded, num_reqs=None
    )
    manager.pad_dp_for_piecewise = pad_policy
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=padded, num_reqs=None
    )
    with mock.patch.object(dp_utils, "get_dp_group") as mock_group:
        mock_group.return_value.cpu_all_reduce_ = fake_all_reduce
        return dp_utils.sync_cudagraph_and_dp_padding(
            cudagraph_manager=manager,
            desired_batch_desc=desc,
            num_tokens=counts[0],
            num_reqs=1,
            uniform_token_count=None,
            dp_size=dp,
            dp_rank=0,
        )


def test_piecewise_pads_every_rank_without_policy():
    desc, sync = _run_piecewise_sync([2000, 4, 4, 4], None)
    assert desc.cg_mode == CUDAGraphMode.PIECEWISE and desc.num_tokens == 2048
    assert sync is not None and not sync.eager
    assert sync.num_tokens_across_dp.tolist() == [2048] * 4


def test_piecewise_policy_declines_padding_runs_eager_unpadded():
    seen = []

    def policy(counts, padded):
        seen.append((counts.tolist(), padded))
        return False

    desc, sync = _run_piecewise_sync([2000, 4, 4, 4], policy)
    assert seen == [([2000, 4, 4, 4], 2048)]
    assert desc.cg_mode == CUDAGraphMode.NONE and desc.num_tokens == 2000
    assert sync is not None and sync.eager
    # Every rank keeps its own count (no graph padding).
    assert sync.num_tokens_across_dp.tolist() == [2000, 4, 4, 4]


def test_piecewise_policy_accepts_padding():
    desc, sync = _run_piecewise_sync([2000, 1900, 2048, 1800], lambda c, p: True)
    assert desc.cg_mode == CUDAGraphMode.PIECEWISE and desc.num_tokens == 2048
    assert sync is not None and not sync.eager
    assert sync.num_tokens_across_dp.tolist() == [2048] * 4
