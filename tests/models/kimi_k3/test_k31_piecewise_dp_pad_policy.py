# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KimiK3ModelState.pad_dp_for_piecewise: which PIECEWISE (prefill) steps pad
every DP rank to the capture size, and which run eager on each rank's own
tokens (CPU only)."""

import torch

from vllm.models.kimi_k3.nvidia import model_state
from vllm.models.kimi_k3.nvidia.model_state import KimiK3ModelState


def _pad(counts, padded):
    return KimiK3ModelState.pad_dp_for_piecewise(
        None, torch.tensor(counts, dtype=torch.int32), padded
    )


def test_one_rank_prefilling_runs_eager_above_the_small_step_cap():
    cap = model_state._PIECEWISE_PAD_ALWAYS_MAX_TOKENS
    for padded in (1024, 1536, 2048):
        if padded > cap:
            # Idle ranks (4-token dummies) would replay the whole graph.
            assert not _pad([padded - 7, 4, 4, 4], padded)


def test_small_steps_always_pad():
    cap = model_state._PIECEWISE_PAD_ALWAYS_MAX_TOKENS
    for padded in (7, 8, 64, 256, cap):
        if 0 < padded <= cap:
            assert _pad([padded, 4, 4, 4], padded)


def test_balanced_ranks_pad():
    assert _pad([2048, 1900, 2000, 1024], 2048)
    assert _pad([1536, 1536, 1536, 1536], 1536)


def test_rank_under_half_of_the_capture_size_runs_eager():
    cap = model_state._PIECEWISE_PAD_ALWAYS_MAX_TOKENS
    if 2048 > cap:
        assert not _pad([2048, 2048, 2048, 1023], 2048)
