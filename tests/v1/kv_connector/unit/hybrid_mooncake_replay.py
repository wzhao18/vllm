# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bindings for metadata-only hybrid cache replay; no GPU kernels or RDMA.

The adapter uses real vLLM allocation, scheduling and Mooncake key publication.
Only the store transport is fake. Sequence assertions live outside this module.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker as mooncake_store_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    PoolKey,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    KVCacheStoreSendingThread,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    _annotate_eagle_groups,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.request import Request, RequestStatus


@dataclass(frozen=True)
class ReplaySetup:
    pmu: int
    mamba_block: int
    attention_gpu_block: int
    attention_mooncake_block: int
    dcp: int


REPLAY_SETUP = ReplaySetup(
    pmu=128,
    mamba_block=896,
    attention_gpu_block=896,
    attention_mooncake_block=7_168,
    dcp=8,
)
PREFILL_CHUNK_TOKENS = 8_192
# FlashKDA prefill checkpoints are aligned to 16 tokens.
FLASHKDA_CHECKPOINT_ALIGNMENT = 16
PREFILL_CHECKPOINT_BLOCKS = 1

PMU = REPLAY_SETUP.pmu
MAMBA_BLOCK = REPLAY_SETUP.mamba_block
DCP = REPLAY_SETUP.dcp
SCHEDULER_BLOCK = REPLAY_SETUP.attention_mooncake_block


@dataclass(frozen=True)
class Sequence:
    name: str
    steps: tuple[int | str, ...]
    branch_suffix_tokens: int = 0


NUM_BLOCKS = 16_384


class MemoryStore:
    """The Mooncake calls used by the production sender and lookup worker."""

    def __init__(self) -> None:
        self.keys: set[str] = set()

    def batch_is_exist(self, keys):
        return [int(key in self.keys) for key in keys]

    def batch_is_exist_no_lease(self, keys):
        return self.batch_is_exist(keys)

    def batch_put_from_multi_buffers(self, keys, _addrs, _sizes, *_args, **_kwargs):
        self.keys.update(keys)
        return [0] * len(keys)


def annotate_draft_groups(groups: list[KVCacheGroupSpec], layer_specs: dict) -> None:
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark",
            use_eagle=lambda: True,
            use_eagle_block_drop=lambda: True,
        )
    )
    _annotate_eagle_groups(
        config,
        layer_specs,
        groups,
    )


def groups(*, external: bool, setup: ReplaySetup) -> list[KVCacheGroupSpec]:
    mamba = MambaSpec(
        block_size=setup.mamba_block,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_prefill_checkpoint_blocks=PREFILL_CHECKPOINT_BLOCKS,
        prefill_checkpoint_alignment=FLASHKDA_CHECKPOINT_ALIGNMENT,
    )
    attention_spec = MLAAttentionSpec(
        block_size=(
            setup.attention_mooncake_block if external else setup.attention_gpu_block
        ),
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        non_causal_multi_token_decode=True,
    )
    result = [
        *(KVCacheGroupSpec([f"mamba{i}"], mamba) for i in range(3)),
        KVCacheGroupSpec(["full"], attention_spec),
    ]
    # vLLM identifies the draft group from the DSpark MLA spec marker.
    layer_specs = {f"mamba{i}": mamba for i in range(3)}
    layer_specs["full"] = attention_spec
    annotate_draft_groups(result, layer_specs)
    return result


def new_request(
    name: str, prompt: list[int], salt: str, hash_block_size: int = PMU
) -> Request:
    sampling = SamplingParams(max_tokens=17)
    sampling.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=name,
        prompt_token_ids=prompt,
        sampling_params=sampling,
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_block_size, sha256),
        cache_salt=salt,
    )


class Replay:
    def __init__(self, num_blocks: int, *, use_mooncake: bool = True) -> None:
        init_none_hash(sha256)
        self.num_blocks = num_blocks
        setup = REPLAY_SETUP
        self.use_mooncake = use_mooncake
        self.setup = setup
        self.groups = groups(external=False, setup=setup)
        self.external_groups = groups(external=True, setup=setup)
        self.manager = self._new_gpu_cache()
        self.coord = MooncakeStoreCoordinator(
            self.external_groups,
            scheduler_block_size=SCHEDULER_BLOCK,
            hash_block_size=PMU,
            use_eagle=True,
            retention_interval=0,
            dcp_world_size=DCP,
        )
        self.store = MemoryStore()
        self.databases = []
        for group_id, block_size in enumerate(
            (MAMBA_BLOCK, MAMBA_BLOCK, MAMBA_BLOCK, SCHEDULER_BLOCK)
        ):
            database = ChunkedTokenDatabase(
                KeyMetadata("synthetic-kimi", 0, 0, 0, 0, group_id=group_id),
                block_size=block_size,
                hash_block_size=PMU,
            )
            # The fake store never dereferences these addresses.
            database.set_kv_caches_base_addr([group_id * 10_000_000])
            database.set_block_len([1])
            self.databases.append(database)
        self.sender = KVCacheStoreSendingThread(
            store=self.store,
            coord=self.coord,
            token_databases=self.databases,
            block_size=SCHEDULER_BLOCK,
            tp_rank=0,
            group_put_steps=[1] * len(self.groups),
            kv_role="kv_both",
            ready_event=threading.Event(),
            replicate_config=MagicMock(),
        )
        self.worker = object.__new__(mooncake_store_worker.MooncakeStoreWorker)
        self.worker._capacity_only = False
        self.worker.coord = self.coord
        self.worker.token_dbs = self.databases
        self.worker._kv_cache_groups = self.external_groups
        self.worker.hash_block_size = PMU
        self.worker.store = self.store
        self.worker._lookup_key_prefixes = tuple(
            (PoolKey.build_prefix(database.metadata),) for database in self.databases
        )
        self.worker._record_kv_connector_operation = lambda *_args, **_kwargs: None
        self.next_job_id = 0
        self.scheduler = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=MAMBA_BLOCK),
            max_num_scheduled_tokens=PREFILL_CHUNK_TOKENS,
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
            use_eagle=True,
            use_eagle_block_drop=True,
            hash_block_size=PMU,
            mamba_partial_cache_hit=True,
            mamba_shared_prefix_checkpoint=getattr(
                self.manager, "mamba_shared_prefix_checkpoint", False
            ),
            mamba_fine_grained_prefix_cache=True,
            mamba_has_prefill_checkpoint_blocks=True,
            mamba_prefill_checkpoint_alignment=FLASHKDA_CHECKPOINT_ALIGNMENT,
        )

    def _new_gpu_cache(self) -> KVCacheManager:
        return KVCacheManager(
            KVCacheConfig(
                num_blocks=self.num_blocks,
                kv_cache_tensors=[],
                kv_cache_groups=self.groups,
                prefix_cache_retention_interval=0,
            ),
            max_model_len=1_048_576,
            enable_caching=True,
            dcp_world_size=DCP,
            scheduler_block_size=SCHEDULER_BLOCK,
            hash_block_size=PMU,
            use_eagle=True,
        )

    def publish(self, request: Request, offloads: list[tuple[int, int, int]]) -> dict:
        before = set(self.store.keys)
        metadata_kwargs = dict(
            req_id=request.request_id,
            token_len_chunk=request.num_computed_tokens,
            block_ids=self.manager.get_block_ids(request.request_id),
            block_hashes=request.block_hashes,
            can_save=True,
            num_prompt_tokens=request.num_prompt_tokens,
            store_job_id=self.next_job_id,
            boundary_state_offloads=offloads or None,
        )
        metadata_kwargs["completed_token_len"] = request.num_computed_tokens
        metadata = ReqMeta(**metadata_kwargs)
        self.next_job_id += 1
        self.sender.add_request(metadata)
        assert self.sender.request_queue.get_nowait() is metadata
        self.sender._handle_request(metadata)
        return {
            "computed_end": request.num_computed_tokens,
            "mamba_boundary_offloads": [
                {"group": group_id, "position": position}
                for group_id, _block_id, position in offloads
            ],
            "new_key_count": len(self.store.keys - before),
        }

    def complete_step(self) -> None:
        # Metadata-only execution completes copies synchronously.
        _, retained = self.manager.take_kv_cache_block_copies()
        self.manager.block_pool.free_blocks(retained)

    def run_request(self, request: Request) -> dict:
        computed_blocks, local_hit, junction = self.manager.get_computed_blocks(request)
        connector_hit = self.manager.get_computed_blocks_for_connector(request)[1]
        mooncake_hit = self.worker.lookup(
            request.num_tokens, request.block_hashes
        ).hit_length
        external_hit = max(0, mooncake_hit - local_hit) if self.use_mooncake else 0
        request.shared_prefix_boundary = junction

        first = True
        while request.num_computed_tokens < request.num_tokens:
            admitted_hit = local_hit + external_hit if first else 0
            remaining = request.num_tokens - request.num_computed_tokens - admitted_hit
            scheduled = Scheduler._mamba_block_aligned_split(
                self.scheduler,
                request,
                min(PREFILL_CHUNK_TOKENS, remaining),
                num_new_local_computed_tokens=admitted_hit,
            )
            assert scheduled > 0
            self.manager.new_step_starts()
            was_first = first
            if first:
                allocated = self.manager.allocate_slots(
                    request,
                    scheduled,
                    local_hit,
                    computed_blocks,
                    num_external_computed_tokens=external_hit,
                )
                first = False
            else:
                allocated = self.manager.allocate_slots(request, scheduled)
            if allocated is None:
                raise RuntimeError("metadata block pool exhausted")
            request.num_computed_tokens += scheduled + (
                local_hit + external_hit if was_first else 0
            )
            offloads = self.manager.take_boundary_state_offloads().get(
                request.request_id, []
            )
            self.publish(request, offloads)
            self.complete_step()

        final_offloads = self.manager.finalize_partial_tail_offloads(request)
        if final_offloads:
            self.publish(request, final_offloads)
        self.manager.free(request)
        self.manager.new_step_starts()
        result = {
            "request_id": request.request_id,
            "prompt_tokens": request.num_tokens,
            "initial_gpu_local_hit": local_hit,
            "initial_connector_hit": connector_hit,
            "initial_mooncake_hit": mooncake_hit,
            "external_hit_admitted": external_hit,
        }
        self.sender.delete_finished_stored_request(request.request_id)
        return result

    def run_sequence(self, sequence: Sequence) -> list[dict]:
        results = []
        turn = 0
        first_reset = next(
            (i for i, step in enumerate(sequence.steps) if step == "reset"),
            len(sequence.steps),
        )
        target_turn = sum(isinstance(s, int) for s in sequence.steps[:first_reset]) - 1
        for step in sequence.steps:
            if step == "reset":
                assert self.manager.reset_prefix_cache(), "GPU reset has active blocks"
                results.append({"operation": "reset"})
                continue
            assert isinstance(step, int)
            prompt = [1000 + i for i in range(step)]
            suffix = sequence.branch_suffix_tokens if turn < target_turn else 0
            if suffix:
                prompt[-suffix:] = [2_000_000 + 1000 * turn + i for i in range(suffix)]
            request = new_request(f"{sequence.name}-turn-{turn}", prompt, sequence.name)
            result = self.run_request(request)
            results.append(
                {
                    "operation": "prompt",
                    "length": step,
                    "branch_suffix_tokens": suffix,
                    **result,
                }
            )
            turn += 1
        return results


class PreemptionReplay:
    """Preempt while a FlashKDA Mamba boundary state awaits handoff."""

    def __init__(self, sequence: Sequence) -> None:
        self.sequence = sequence
        self.replay = Replay(64)

    def run_sequence(self) -> dict:
        seed_len, prompt_len, preempt, resume, reset, retry_len = self.sequence.steps
        assert isinstance(prompt_len, int) and retry_len == prompt_len
        assert (preempt, resume, reset) == ("preempt", "resume", "reset")
        seed = new_request(
            f"{self.sequence.name}-seed",
            [1_000 + i for i in range(seed_len)],
            self.sequence.name,
        )
        self.replay.run_request(seed)
        owner = new_request(
            f"{self.sequence.name}-owner",
            [1_000 + i for i in range(prompt_len)],
            self.sequence.name,
        )
        # Stop before the final continuation overwrites its pending tail state.
        target = prompt_len // PMU * PMU - PMU
        blocks, local_hit, junction = self.replay.manager.get_computed_blocks(owner)
        assert 0 < local_hit < target
        owner.shared_prefix_boundary = junction
        chunk_ends = []
        while owner.num_computed_tokens < target:
            admitted_hit = local_hit if not chunk_ends else 0
            remaining = prompt_len - owner.num_computed_tokens - admitted_hit
            scheduled = Scheduler._mamba_block_aligned_split(
                self.replay.scheduler,
                owner,
                min(PREFILL_CHUNK_TOKENS, remaining),
                num_new_local_computed_tokens=admitted_hit,
            )
            assert 0 < scheduled <= target - owner.num_computed_tokens - admitted_hit, (
                owner.num_computed_tokens,
                scheduled,
                target,
            )
            self.replay.manager.new_step_starts()
            allocated = self.replay.manager.allocate_slots(
                owner,
                scheduled,
                num_new_computed_tokens=local_hit if not chunk_ends else 0,
                new_computed_blocks=blocks if not chunk_ends else None,
                has_scheduled_reqs=False,
            )
            assert allocated is not None
            owner.num_computed_tokens += scheduled + admitted_hit
            chunk_ends.append(owner.num_computed_tokens)
            offloads = self.replay.manager.take_boundary_state_offloads().get(
                owner.request_id, []
            )
            assert not offloads, "tail was handed off before preemption"
            if owner.num_computed_tokens < target:
                self.replay.publish(owner, [])
            self.replay.complete_step()

        handoffs = []
        replay = self.replay

        class Connector:
            def register_finished_partial_tail(self, request, block_ids, offloads):
                handoffs.append(
                    (block_ids, offloads, replay.publish(request, offloads))
                )
                return True

        scheduler = object.__new__(Scheduler)
        scheduler.connector = Connector()
        scheduler.aux_output_connector = None
        scheduler.vllm_config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(is_kv_producer=True)
        )
        scheduler.kv_cache_manager = self.replay.manager
        scheduler._free_request_blocks = self.replay.manager.free
        scheduler.encoder_cache_manager = MagicMock()
        scheduler._inflight_prefills = set()
        scheduler.log_stats = False
        scheduler.waiting = MagicMock()
        scheduler.reset_preempted_req_ids = set()
        owner.status = RequestStatus.RUNNING
        Scheduler._preempt_request(scheduler, owner, timestamp=0.0)
        checkpoint_hash = owner.block_hashes[target // PMU - 1]
        preemption_state_present = all(
            db.key_for(checkpoint_hash) in self.replay.store.keys
            for db in self.replay.sender.token_databases[:3]
        )
        self.replay.run_request(owner)

        retry = new_request(
            f"{self.sequence.name}-retry",
            [1_000 + i for i in range(retry_len)],
            self.sequence.name,
        )
        local_hit = self.replay.manager.get_computed_blocks(retry)[1]
        assert self.replay.manager.reset_prefix_cache(), "GPU reset has active blocks"
        mooncake_hit = self.replay.worker.lookup(
            retry.num_tokens, retry.block_hashes
        ).hit_length
        return {
            "target": target,
            "chunk_ends": chunk_ends,
            "handoffs": handoffs,
            "preemption_state_present": preemption_state_present,
            "local_hit": local_hit,
            "mooncake_hit": mooncake_hit,
            "draft_groups": [
                group.layer_names
                for group in self.replay.groups
                if group.is_eagle_group
            ],
        }
