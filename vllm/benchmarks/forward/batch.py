# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a `SchedulerOutput` for an exact batch, without the scheduler."""

import os
from dataclasses import dataclass

import numpy as np

from vllm.benchmarks.forward.shapes import BenchCase
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

# Windows of real token ids (a 1-D .npy, e.g. AgentX prompts) instead of uniform
# random ids, so MoE routing follows a real expert-load distribution.
_TOKEN_FILE = os.environ.get("VLLM_BENCH_FWD_TOKEN_FILE")
_REAL_TOKENS = np.load(_TOKEN_FILE) if _TOKEN_FILE else None


@dataclass
class BenchBatch:
    scheduler_output: SchedulerOutput
    # req_id -> token IDs of the query tokens, in order.
    query_token_ids: dict[str, np.ndarray]
    # Per KV cache group: allocated (begin, end) block ranges.
    group_block_ranges: list[list[tuple[int, int]]]
    num_blocks: int
    # req_id -> the runner's expected (scheduled, computed, is_decode).
    expected: dict[str, tuple[int, int, bool]]

    @property
    def num_tokens(self) -> int:
        return self.scheduler_output.total_num_scheduled_tokens

    @property
    def req_ids(self) -> list[str]:
        return list(self.scheduler_output.num_scheduled_tokens)

    def finish(self) -> SchedulerOutput:
        """A step that only removes this batch's requests."""
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = set(self.req_ids)
        return output


def kv_block_size(spec: KVCacheSpec, dcp_size: int) -> int:
    """Tokens per block as the scheduler counts them: DCP spreads each
    block's tokens over the DCP ranks."""
    return spec.block_size * (dcp_size if spec.dcp_sharded else 1)


def _num_blocks(
    spec: KVCacheSpec, computed: int, kv_len: int, dcp_size: int
) -> tuple[int, int]:
    """Return (leading null blocks, live blocks) for one request."""
    block_size = kv_block_size(spec, dcp_size)
    if isinstance(spec, UniformTypeKVCacheSpecs):
        spec = spec.first_spec
    if isinstance(spec, MambaSpec):
        return 0, 1 + spec.num_speculative_blocks
    if isinstance(spec, CircularBufferSpec):
        return 0, 1
    total = -(-kv_len // block_size)
    skipped = 0
    if isinstance(spec, SlidingWindowSpec):
        # Blocks wholly before the window are freed to the null block.
        skipped = max(0, computed - spec.sliding_window + 1) // block_size
    return skipped, total - skipped


def _dsa_dp_owners(case: BenchCase) -> list[int]:
    """DSA-DP owner card of each request (models/kimi_k3/nvidia/ops/dsa_dp.py), or
    -1 without DSA-DP. Like the scheduler's load-based owner choice: longest
    DSA cost first onto the least-loaded card of the TP group (same on every
    rank: config and case only), so a bench step spreads its DSA rows over the
    cards instead of hashing request ids."""
    from vllm.models.kimi_k3.nvidia.ops import dsa_dp

    if not dsa_dp.ENABLED:
        return [-1] * len(case.requests)
    from vllm.distributed import get_tensor_model_parallel_world_size

    tp = get_tensor_model_parallel_world_size()
    costs = [dsa_dp.chunk_cost(r.query_len, r.prefix_len) for r in case.requests]
    load = [0.0] * tp
    owners = [-1] * len(case.requests)
    for i in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        card = min(range(tp), key=lambda c: (load[c], c))
        owners[i] = card
        load[card] += costs[i]
    return owners


def _dsa_dp_regions(kv_config) -> tuple[frozenset[int], int, int, int] | None:
    """(DSA group ids, shared pages S, region pages D, this card's TP rank) of a
    worker KV config split into DSA-DP regions, else None."""
    if kv_config is None or getattr(kv_config, "dsa_dp_region_blocks", 0) <= 0:
        return None
    from vllm.distributed import get_tensor_model_parallel_rank

    return (
        frozenset(kv_config.dsa_dp_group_ids),
        kv_config.dsa_dp_shared_blocks,
        kv_config.dsa_dp_region_blocks,
        get_tensor_model_parallel_rank(),
    )


def build_batch(
    case: BenchCase,
    kv_cache_groups,
    num_kv_blocks: int,
    vocab_size: int,
    rng: np.random.Generator,
    dcp_size: int = 1,
    lookahead: int = 0,
    kv_config=None,
) -> BenchBatch:
    """Allocate disjoint blocks and build `NewRequestData` for every request.

    Groups share one block pool, so block IDs are unique across groups. Block
    0 is the null block. Lookahead reserves space for the drafter's KV writes.

    With prefix P = K - Q (KV length K, query length Q):

    | Kind               | prompt tokens | num_computed_tokens | spec tokens |
    | ------------------ | ------------- | ------------------- | ----------- |
    | Prefill            | K             | P                   | -           |
    | Decode             | P             | P                   | -           |
    | Verify (D drafts)  | P             | P                   | D           |

    Unless chunked, a prefill's prefix is a prefix hit, which replays its last
    R = min(window, P) tokens on models with SWA bounded replay (window 0
    otherwise), like the scheduler: it schedules R + Q tokens from
    replay_start = num_computed_tokens = P - R.
    """
    replay_window = max(
        (group.kv_cache_spec.prefix_replay_tokens for group in kv_cache_groups),
        default=0,
    )
    expected: dict[str, tuple[int, int, bool]] = {}
    cursor = 1
    new_reqs = []
    query_token_ids = {}
    group_block_ranges: list[list[tuple[int, int]]] = [[] for _ in kv_cache_groups]
    output = SchedulerOutput.make_empty()
    owners = _dsa_dp_owners(case)
    # DSA-DP per-card DSA KV regions (v1/core/dsa_dp_regions.py): a request's DSA
    # group blocks come from its owner card's region, scheduler ids
    # [S + c * D, S + (c + 1) * D); the other groups from the shared [1, S).
    regions = _dsa_dp_regions(kv_config) if owners and owners[0] >= 0 else None
    if regions is not None:
        dsa_ids, shared, region, rank = regions
        region_cursor = {}
    for i, req in enumerate(case.requests):
        req_id = f"bench-{i}"
        replay = min(replay_window, req.prefix_len) if req.prefix_hit else 0
        computed = req.prefix_len - replay
        block_ids = []
        for gi, (group, ranges) in enumerate(zip(kv_cache_groups, group_block_ranges)):
            skipped, live = _num_blocks(
                group.kv_cache_spec, computed, req.kv_len + lookahead, dcp_size
            )
            if regions is not None and gi in dsa_ids:
                owner = owners[i]
                first = region_cursor.get(owner, shared + owner * region)
                region_cursor[owner] = first + live
                if first + live > shared + (owner + 1) * region:
                    raise ValueError(
                        f"Case {case.name!r} needs more than the {region} DSA "
                        f"region pages of card {owner}. Increase the KV cache size."
                    )
                block_ids.append([0] * skipped + list(range(first, first + live)))
                if owner == rank:  # this card's pages (kv_init zeroes them)
                    local = first - owner * region
                    ranges.append((local, local + live))
                continue
            # A slot-packed group (``blocks_per_pool_block`` S > 1) addresses slot
            # j in pool row j // S: reserve whole rows from the shared cursor and
            # hand out the slot ids inside them. ``ranges`` stays in row units,
            # which kv_init's per-tensor factor (S) expands to those rows' slots.
            slots = getattr(group, "blocks_per_pool_block", 1)
            rows = -(-live // slots)
            first = cursor * slots
            block_ids.append([0] * skipped + list(range(first, first + live)))
            ranges.append((cursor, cursor + rows))
            cursor += rows
        if _REAL_TOKENS is not None and len(_REAL_TOKENS) > req.kv_len:
            start = int(rng.integers(0, len(_REAL_TOKENS) - req.kv_len))
            tokens = _REAL_TOKENS[start : start + req.kv_len].astype(np.int64)
        else:
            tokens = rng.integers(0, vocab_size, req.kv_len, dtype=np.int64)
        prompt = tokens[: req.prefix_len] if req.is_decode else tokens
        query_token_ids[req_id] = tokens[computed:]
        expected[req_id] = (req.query_len + replay, computed, req.is_decode)
        new_reqs.append(
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=prompt.tolist(),
                prefill_token_ids=prompt.tolist(),
                mm_features=[],
                sampling_params=SamplingParams(max_tokens=16, temperature=0.0),
                pooling_params=None,
                block_ids=tuple(block_ids),
                num_computed_tokens=computed,
                lora_request=None,
                replay_start=computed if replay else 0,
                dsa_owner=owners[i],
            )
        )
        output.num_scheduled_tokens[req_id] = req.query_len + replay
        if req.num_draft_tokens:
            output.scheduled_spec_decode_tokens[req_id] = tokens[
                req.prefix_len + 1 :
            ].tolist()
    if regions is not None and cursor > regions[1]:
        raise ValueError(
            f"Case {case.name!r} needs {cursor} shared-region KV blocks, but the "
            f"DSA-DP shared region has {regions[1]}. Increase the KV cache size."
        )
    if cursor > num_kv_blocks:
        raise ValueError(
            f"Case {case.name!r} needs {cursor} KV blocks, but only "
            f"{num_kv_blocks} exist. Increase the KV cache size."
        )
    output.scheduled_new_reqs = new_reqs
    output.total_num_scheduled_tokens = sum(output.num_scheduled_tokens.values())
    return BenchBatch(output, query_token_ids, group_block_ranges, cursor, expected)
