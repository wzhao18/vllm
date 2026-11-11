# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which instances allocate the Kimi K3 SP all-gather's symmetric-memory buffer,
and for how many rows per rank (``ops/sp_all_gather.sp_all_gather_rows``):
prefill and unified instances for max_num_batched_tokens, nothing when that
step fits the custom all-gather; a P/D decode instance never (stock
all-gather for every step)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia.ops import sp_all_gather as sp_ag

HIDDEN = 7168
CAPTURE = [7, 8, 14, 16, 28, 32, 56, 64, 112, 128, 256, 384, 512, 1024, 2048]


def _config(
    *,
    tp=4,
    mnbt=32768,
    max_num_seqs=128,
    num_spec=3,
    kv_role=None,
    capture=CAPTURE,
):
    kv_transfer = None
    if kv_role is not None:
        kv_transfer = SimpleNamespace(
            is_kv_producer=kv_role in ("kv_producer", "kv_both"),
            is_kv_consumer=kv_role in ("kv_consumer", "kv_both"),
        )
    return SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=mnbt, max_num_seqs=max_num_seqs
        ),
        speculative_config=(
            None
            if num_spec is None
            else SimpleNamespace(num_speculative_tokens=num_spec)
        ),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=capture),
        kv_transfer_config=kv_transfer,
    )


@pytest.fixture
def custom_all_gather(monkeypatch):
    """The TP group's custom all-gather: up to ``max_bytes`` (2 MiB, the stock
    default), or none."""

    def set_ca(max_bytes: int | None = 2 * 1024 * 1024):
        ca_comm = (
            None
            if max_bytes is None
            else SimpleNamespace(
                should_custom_all_gather=lambda t: t.nbytes <= max_bytes
            )
        )
        group = SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=ca_comm))
        monkeypatch.setattr(sp_ag, "get_tp_group", lambda: group)

    set_ca()
    return set_ca


def _rows(config):
    return sp_ag.sp_all_gather_rows(config, HIDDEN, torch.bfloat16)


@pytest.mark.parametrize("kv_role", [None, "kv_producer", "kv_both"])
def test_prefill_and_unified_size_for_max_num_batched_tokens(
    custom_all_gather, kv_role
):
    # C512 prefill: TP4, MNBT 32K -> 8192 rows per rank (896 MiB for two slots).
    assert _rows(_config(kv_role=kv_role)) == 8192
    assert _rows(_config(kv_role=kv_role, mnbt=17401)) == 4351


@pytest.mark.parametrize("max_num_seqs", [128, 256, 1024])
def test_decode_allocates_nothing(custom_all_gather, max_num_seqs):
    # C512's decode role (128 x q4 = 512 tokens, inside the custom gather) and
    # larger decode batches alike: a decode instance's local prefills reach
    # max_num_batched_tokens, so no smaller buffer covers its steps.
    config = _config(kv_role="kv_consumer", max_num_seqs=max_num_seqs)
    assert sp_ag.is_decode_instance(config)
    assert _rows(config) == 0
    custom_all_gather(None)
    assert _rows(config) == 0


def test_small_max_num_batched_tokens_allocates_nothing(custom_all_gather):
    # 512 tokens / TP4 = 128 rows x 7168 BF16 = 1.75 MiB: custom all-gather.
    assert _rows(_config(mnbt=512)) == 0
    # 147 rows (2.01 MiB): above it.
    assert _rows(_config(mnbt=588)) == 147


def test_without_custom_all_gather_every_size_allocates(custom_all_gather):
    custom_all_gather(None)
    assert _rows(_config(mnbt=512)) == 128
    assert _rows(_config()) == 8192
