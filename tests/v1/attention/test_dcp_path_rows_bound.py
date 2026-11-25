# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests: on a decode-only instance (P/D kv_consumer) the DCP path's
MQA passes are bounded to the DCP workspace's tokens, so a larger step runs in
row chunks on the direct symmetric-memory gather and combine and never takes
(or warms) their NCCL fallback over the DCP group."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fi
from vllm.v1.attention.ops import dcp

# Kimi K3.1 C256 decode: TP16 = DCP16, max_num_batched_tokens 8192, max_num_seqs
# 256, 3 speculative tokens, max_cudagraph_capture_size 1024.
_MNBT = 8192
_WORKSPACE = 1024


def _kv(role: str | None):
    if role is None:
        return None
    return SimpleNamespace(
        is_kv_consumer=role in ("kv_consumer", "kv_both"),
        is_kv_producer=role in ("kv_producer", "kv_both"),
    )


def _cfg(role: str | None, mnbt: int = _MNBT):
    return SimpleNamespace(
        kv_transfer_config=_kv(role),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=mnbt, max_num_seqs=256
        ),
        num_speculative_tokens=3,
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=1024),
    )


def _patch_workspace():
    return patch.object(
        dcp, "get_dcp_workspace_max_num_tokens", return_value=_WORKSPACE
    )


@pytest.mark.parametrize(
    "role, bound",
    [
        ("kv_consumer", _WORKSPACE),
        ("kv_both", None),
        ("kv_producer", None),
        (None, None),
    ],
)
def test_rows_bound_only_on_decode_only_instances(role, bound):
    with _patch_workspace():
        assert dcp.dcp_path_rows_bound(_cfg(role)) == bound


def test_rows_bound_is_the_dcp_workspace_tokens():
    """Unpatched: min(MNBT, max(mns x (1 + spec), max capture)) = 1024 at C256."""
    with patch.object(dcp, "max_decode_query_len", return_value=1):
        assert dcp.dcp_path_rows_bound(_cfg("kv_consumer")) == 1024


@pytest.mark.parametrize(
    "role, shardq, mnbt, rows",
    [
        # Decode-only C256: 1024-row chunks instead of whole 8192-token passes.
        ("kv_consumer", False, _MNBT, _WORKSPACE),
        # Never above max_num_batched_tokens.
        ("kv_consumer", False, 512, 512),
        # Prefill, colocated and kv_both instances keep their rows.
        ("kv_both", False, _MNBT, _MNBT),
        (None, False, _MNBT, _MNBT),
    ],
)
def test_dcp_path_max_rows(role, shardq, mnbt, rows):
    with _patch_workspace():
        assert fi.dcp_path_max_rows(_cfg(role, mnbt), shardq) == rows


def test_dcp_path_max_rows_unchanged_under_shardq():
    with (
        _patch_workspace(),
        patch.object(fi, "trtllm_gen_lse_max_rows", return_value=2048),
    ):
        assert fi.dcp_path_max_rows(_cfg("kv_consumer"), True) == 2048


def test_chunks_never_exceed_the_direct_workspace():
    """Every DCP-path pass of a decode-only instance fits the direct workspace,
    so warm_up_dcp_fallback (called with dcp_path_max_rows) runs no NCCL
    collective: its fallback branch needs num_tokens > max_num_tokens."""
    with _patch_workspace():
        rows = fi.dcp_path_max_rows(_cfg("kv_consumer"), False)
    assert rows <= _WORKSPACE
