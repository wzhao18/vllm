# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank MegaMoE symmetric-buffer token count (KimiMoE construction)."""

import pytest

from vllm.models.kimi_k3.nvidia.model import mega_moe_max_num_tokens


@pytest.mark.parametrize(
    ("mnbt", "tp_size", "token_sharded", "expected"),
    [
        # DEP (TP1): every rank's MoE sees its full batch.
        (8192, 1, False, 8192),
        # Plain TP without a token shard: replicated stream, full batch.
        (8192, 8, False, 8192),
        # DCP8xDEP2 under sequence parallelism: one 1/TP shard, divided once
        # (the double division gave 128 slots for a 1024-token shard).
        (8192, 8, True, 1024),
        # TP16/DCP16 decode under sequence parallelism.
        (16384, 16, True, 1024),
        # sp_shard pads to a multiple of TP: round the shard up.
        (8193, 8, True, 1025),
        (1, 16, True, 1),
    ],
)
def test_mega_moe_max_num_tokens(mnbt, tp_size, token_sharded, expected):
    assert mega_moe_max_num_tokens(mnbt, tp_size, token_sharded) == expected


@pytest.mark.parametrize("tp_size", [2, 4, 8, 16])
@pytest.mark.parametrize("mnbt", [256, 1000, 8192, 16384])
def test_mega_moe_shard_holds_largest_sp_shard(mnbt, tp_size):
    # The largest per-rank shard of an MNBT-token batch padded to a multiple of
    # TP must fit, and the buffer must not exceed it.
    padded = -(-mnbt // tp_size) * tp_size
    assert mega_moe_max_num_tokens(mnbt, tp_size, True) == padded // tp_size
