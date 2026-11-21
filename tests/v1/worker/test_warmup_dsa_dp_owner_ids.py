# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA-DP warmup block ids: hand-built warmup steps allocate this card's pages
``[0, S + D)``; the DSA groups of a request owned by card ``c`` must reach the
worker in that card's region form (``[S + c * D, S + (c + 1) * D)``), which
``dsa_dp.localize_tables`` maps back to the local pages. Before the fix a warmup
id in ``[S, S + D)`` owned by card 2 of TP4 was read as a negative page (illegal
address in the first DSA-DP prefill warmup step)."""

from types import SimpleNamespace

from vllm.models.kimi_k3.nvidia.ops import dsa_dp
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.worker.gpu.warmup import _owner_region_block_ids

S, D, TP = 10, 20, 4
OWNERS = {"a": 0, "b": 2, "c": 3}


def _runner(region_blocks: int = D):
    cfg = SimpleNamespace(
        dsa_dp_group_ids=(1,),
        dsa_dp_shared_blocks=S,
        dsa_dp_region_blocks=region_blocks,
    )
    return SimpleNamespace(kv_cache_config=cfg)


def _new(req_id: str, block_ids):
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=[0],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=block_ids,
        num_computed_tokens=0,
        lora_request=None,
        prefill_token_ids=[0],
    )


def test_warmup_ids_take_the_owner_region(monkeypatch):
    monkeypatch.setattr(
        dsa_dp, "owners_of", lambda ids: [OWNERS[r] for r in ids]
    )
    seen = []
    execute = _owner_region_block_ids(_runner(), seen.append)
    out = SchedulerOutput.make_empty()
    # Group 0: shared region (untouched); group 1: DSA (ids >= S move by c * D).
    out.scheduled_new_reqs = [_new("a", ([1, 12], [3, 15])), _new("b", ([2], [4, 16]))]
    cached = CachedRequestData.make_empty()
    cached.req_ids = ["c", "a"]
    cached.new_block_ids = [([5], [11, 29]), None]
    out.scheduled_cached_reqs = cached
    execute(out)
    assert seen == [out]
    assert out.scheduled_new_reqs[0].block_ids == ([1, 12], [3, 15])
    assert out.scheduled_new_reqs[1].block_ids == ([2], [4, 16 + 2 * D])
    assert cached.new_block_ids == [([5], [11 + 3 * D, 29 + 3 * D]), None]
    # Every translated DSA id is in its owner's region and maps back to the
    # same local page on that card.
    for owner, ids in ((2, [4, 16 + 2 * D]), (3, [11 + 3 * D, 29 + 3 * D])):
        local = [dsa_dp.to_local_block_id(b, S, D, owner) for b in ids]
        assert min(local) >= 0
        assert all(b < S or S + owner * D <= b < S + (owner + 1) * D for b in ids)
    # Wrapping twice translates once.
    assert _owner_region_block_ids(_runner(), execute) is execute


def test_warmup_ids_unchanged_without_regions():
    execute = print
    assert _owner_region_block_ids(_runner(region_blocks=0), execute) is execute
