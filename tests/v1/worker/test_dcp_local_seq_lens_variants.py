# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""prepare_dcp_local_seq_lens compiles one variant for every DCP rank and
request count (2026-11-12: the serving-shape verification step of a TP16=DCP16
decode server compiled _dcp_local_seq_lens_kernel on every rank, for a request
count and, on rank 1, a rank value the warmup steps never used)."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils import jit_monitor
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens


def _reference(seq_lens, dcp_size, dcp_rank, interleave):
    rounds = seq_lens // (dcp_size * interleave)
    rem = (seq_lens % (dcp_size * interleave) - dcp_rank * interleave).clamp(min=0)
    return rounds * interleave + rem.clamp(max=interleave)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("interleave", [1, 64])
def test_one_variant_for_every_rank_and_request_count(interleave):
    dcp_size, max_num_reqs = 16, 256
    out = torch.empty(max_num_reqs, dtype=torch.int32, device="cuda")
    seq_lens = torch.randint(
        0, 100_000, (max_num_reqs,), dtype=torch.int32, device="cuda"
    )
    prepare_dcp_local_seq_lens(out, seq_lens, 7, dcp_size, 0, interleave)
    with jit_monitor.record_jit_events() as events:
        for dcp_rank in range(dcp_size):
            for num_reqs in (1, 16, 17, 32, 255):
                prepare_dcp_local_seq_lens(
                    out, seq_lens, num_reqs, dcp_size, dcp_rank, interleave
                )
                want = _reference(seq_lens[:num_reqs], dcp_size, dcp_rank, interleave)
                assert torch.equal(out[:num_reqs], want)
                assert not out[num_reqs:].any()
    assert events == [], f"compiled after the first launch: {events}"
