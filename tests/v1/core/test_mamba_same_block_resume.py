# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" prefills whose last chunk ends in the block it starts in.

With DEP-sized Mamba blocks (16384 tokens) and 128-token prefix hits, most
conversation turns resume in the middle of a block and end in that same block.
A first chunk exports the prompt checkpoint into a snapshot block below its
initial state (CacheConfig.mamba_internal_snapshot). A running request's last
chunk (the first chunk was cut by the token budget, the long-prefill threshold
or the block end) used to stop at the checkpoint and run the 1-127 token tail as
another forward pass: its worker block table is append-only, so the column below
the initial state could hold a stale block id. The scheduler now rewrites that
column (SchedulerOutput.block_table_writes), and the chunk runs in one pass.

These tests drive the real scheduler split, KV cache manager and worker
checkpoint planner, and model the worker's block table (whole table for a new
request, appends plus column writes for a running one) and which state every
Mamba block holds: CoW copies, the align pre-copy into the running column, the
checkpoint export and the running state. Every hashed block, every boundary
offered to a KV connector (Mooncake) and the block a P/D connector (NIXL) sends
must hold the state at exactly the token count claimed.
"""

from types import SimpleNamespace

import pytest

from tests.v1.core.prefix_cache.test_partial_prefix_cache_hits import (
    make_full_mamba_manager,
)
from tests.v1.core.test_prefix_caching import make_request
from vllm.distributed.kv_transfer.kv_connector.utils import clip_ssm_state_blocks
from vllm.model_executor.layers.mamba.checkpoint import (
    compute_mamba_prefill_checkpoints,
)
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import get_group_id, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import get_mamba_prefill_checkpoint_position
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test

HASH = 128  # --prefix-match-unit
BLOCK = 16384  # DEP16 prefill Mamba block
BUDGET = 16384  # max_num_batched_tokens
MAMBA_GID = 1


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _checkpoint(num_tokens: int) -> int:
    return get_mamba_prefill_checkpoint_position(num_tokens, HASH, False)


class _Engine:
    """Scheduler split + KV cache manager + a CPU model of the worker."""

    def __init__(
        self, *, snapshot: bool, num_spec: int = 0, retention_interval=0
    ) -> None:
        self.snapshot = snapshot
        self.retention_interval = retention_interval
        self.num_spec = num_spec
        self.manager = make_full_mamba_manager(
            dcp_world_size=1,
            hash_block_size=HASH,
            full_block_size=HASH,
            mamba_block_size=BLOCK,
            num_blocks=4096,
            num_speculative_blocks=num_spec,
            num_prefill_checkpoint_blocks=1,
            retention_interval=retention_interval,
            prefill_snapshot=snapshot,
            max_model_len=262144,
        )
        self.mamba = self.manager.coordinator.single_type_managers[MAMBA_GID]
        self.null_id = self.manager.block_pool.null_block.block_id
        # Mamba block id -> token count of the state it holds.
        self.state: dict[int, int] = {}
        # The worker's block table and running column per request.
        self.tables: dict[str, list[int]] = {}
        self.running_col: dict[str, int] = {}
        # The chunk start whose snapshot column the worker holds fresh.
        self.snapshot_start: dict[str, int] = {}
        # Boundaries offered to the KV connector: request -> [(block, tokens)].
        self.offloads: dict[str, list[tuple[int, int]]] = {}

    def split(self, request, num_new: int, local: int) -> int:
        stub = SimpleNamespace(
            block_size=BLOCK,
            mamba_block_size=BLOCK,
            cache_config=SimpleNamespace(block_size=BLOCK),
            use_eagle_block_drop=False,
            num_hash_lookahead_tokens=0,
            max_num_scheduled_tokens=BUDGET,
            # The DEP16 prefill's threshold: sub-block chunks are allowed.
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=8192),
            mamba_partial_cache_hit=True,
            mamba_shared_prefix_checkpoint=False,
            hash_block_size=HASH,
            mamba_has_prefill_checkpoint_blocks=True,
            mamba_prefill_checkpoint_alignment=16,
            mamba_internal_snapshot=self.snapshot,
            # Scheduler.mamba_crossing_chunks: snapshot support + sparse retention.
            mamba_crossing_chunks=(
                self.snapshot and self.retention_interval in (0, BLOCK)
            ),
            kv_cache_manager=self.manager,
        )
        return Scheduler._mamba_block_aligned_split(stub, request, num_new, local)

    def allocate(self, request, budget: int = BUDGET):
        """Schedule one chunk; returns (start, end, new blocks, local hits)."""
        rid = request.request_id
        blocks, local = None, 0
        if rid not in self.tables:
            blocks, local, _ = self.manager.get_computed_blocks(request)
        start = request.num_computed_tokens + local
        num_new = self.split(request, min(request.num_tokens - start, budget), local)
        assert num_new > 0
        new = self.manager.allocate_slots(request, num_new, local, blocks)
        assert new is not None
        return start, start + num_new, new

    def forward(self, request, start: int, end: int, new) -> None:
        """The worker's step for one allocated chunk."""
        rid = request.request_id
        first = rid not in self.tables
        if first:
            # A new request ships its whole block table.
            self.tables[rid] = [b.block_id for b in self.mamba.req_to_blocks[rid]]
            prev = (start - 1) // BLOCK
            self.snapshot_start[rid] = start
        else:
            # A running request's table is append-only...
            self.tables[rid] += new.get_block_ids()[MAMBA_GID]
            prev = self.running_col[rid]
        # ...except for the scheduler's column writes.
        writes = self.manager.take_block_table_writes()
        assert set(writes) <= (set() if first else {rid})
        for group_id, col, block_id in writes.get(rid, []):
            assert group_id == MAMBA_GID
            self.tables[rid][col] = block_id
            self.snapshot_start[rid] = start
        table = self.tables[rid]
        sched_table = [b.block_id for b in self.mamba.req_to_blocks[rid]]
        assert len(table) == len(sched_table)
        offloads = self.manager.take_boundary_state_offloads().get(rid, [])
        # CoW copies, then the align pre-copy into the running column.
        copies, retained = self.manager.take_kv_cache_block_copies()
        for copy in copies:
            if copy.group_id == MAMBA_GID and copy.src_block_id in self.state:
                self.state[copy.dst_block_id] = self.state[copy.src_block_id]
        self.manager.block_pool.free_blocks(retained)
        cur = cdiv(end, BLOCK) - 1
        if 0 <= prev != cur:
            self.state[table[cur]] = self.state.get(table[prev], -1)
        # Forward: checkpoint export, then the running state.
        offsets, cols = compute_mamba_prefill_checkpoints(
            [end],
            [end - start],
            hash_block_size=HASH,
            mamba_block_size=BLOCK,
            checkpoint_alignment=16,
            drop_eagle_block=False,
            cache_hit_alignment_tokens=HASH,
            first_chunk=(
                [self.snapshot_start.get(rid) == start] if self.snapshot else None
            ),
            # The V2 model state ships each row's prefill end with the snapshot
            # support (crossing-chunk boundary export).
            prefill_ends=(
                [max(request.num_prompt_tokens, request.num_tokens - 1)]
                if self.snapshot and self.retention_interval in (0, BLOCK)
                else None
            ),
        )
        if offsets[0] and table[cols[0]] != self.null_id:
            # Never into a block this request does not hold at that column.
            assert table[cols[0]] == sched_table[cols[0]]
            assert cols[0] != cur
            self.state[table[cols[0]]] = start + offsets[0]
        self.state[table[cur]] = end
        self.running_col[rid] = cur
        request.num_computed_tokens = end
        request.status = RequestStatus.RUNNING
        self.manager.new_step_starts()
        for _, block_id, tokens in offloads:
            assert self.state.get(block_id) == tokens, (block_id, tokens)
            self.offloads.setdefault(rid, []).append((block_id, tokens))
        assert self.stale_blocks() == []

    def prefill(self, request, budgets=()) -> list[tuple[int, int]]:
        """Prefill the rest of the prompt; returns each forward pass's span."""
        spans: list[tuple[int, int]] = []
        while request.num_computed_tokens < request.num_tokens:
            budget = budgets[len(spans)] if len(spans) < len(budgets) else BUDGET
            start, end, new = self.allocate(request, budget)
            self.forward(request, start, end, new)
            spans.append((start, end))
        return spans

    def finish(self, request) -> None:
        self.manager.free(request)
        self.manager.new_step_starts()
        self.tables.pop(request.request_id, None)

    def stale_blocks(self) -> list[tuple[int, int | None, int | None]]:
        """Hashed Mamba blocks whose state is not at the position they claim."""
        return [
            (b.block_id, b.block_hash_num_tokens, self.state.get(b.block_id))
            for b in self.manager.block_pool.blocks
            if not b.is_null
            and b.block_hash is not None
            and get_group_id(b.block_hash) == MAMBA_GID
            and self.state.get(b.block_id) != b.block_hash_num_tokens
        ]

    def hashed_states(self) -> set[tuple[int, bytes]]:
        """(token count, hash) of every cached Mamba state."""
        return {
            (b.block_hash_num_tokens, bytes(b.block_hash))
            for b in self.manager.block_pool.blocks
            if not b.is_null
            and b.block_hash is not None
            and get_group_id(b.block_hash) == MAMBA_GID
        }

    def sent_state(self, request) -> int | None:
        """State of the block a P/D connector sends for ``request``."""
        ids = self.manager.get_block_ids(request.request_id)[MAMBA_GID]
        (block_id,) = clip_ssm_state_blocks(list(ids), self.num_spec)
        return self.state.get(block_id)


def _turn(request_id: str, tokens: list[int]):
    return make_request(request_id, list(tokens), HASH, sha256)


def _prefix(engine: _Engine, num_tokens: int) -> list[int]:
    """Prefill and finish a cold first turn of ``num_tokens`` tokens."""
    tokens = list(range(1, num_tokens + 1))
    request = _turn("t0", tokens)
    engine.prefill(request)
    engine.finish(request)
    return tokens


# Turn 1 ends at 120000: its checkpoint 119936 is in block 7 ([114688, 131072)).
PREFIX = 120000
RESUME = _checkpoint(PREFIX)
# A 2000-token budget cut of the next turn's first chunk, kept on the hash grid.
CUT = RESUME + 2000 // HASH * HASH


def _next_turn(engine: _Engine, new_tokens: int, request_id: str = "t1"):
    tokens = _prefix(engine, PREFIX) + list(range(10**6, 10**6 + new_tokens))
    return _turn(request_id, tokens)


@pytest.mark.parametrize(
    "retention_interval", [None, 0, BLOCK], ids=["dense", "sparse", "grid"]
)
@pytest.mark.parametrize("num_spec", [0, 3])
@pytest.mark.parametrize(
    ("new_tokens", "budgets", "spans"),
    [
        # Resumes mid-block, the budget cuts the first chunk inside the block
        # (on the hash grid), the last chunk ends in the same block: was 3
        # passes (stop at the checkpoint 124928, then a 72-token tail).
        (5000, [2000], [(RESUME, CUT), (CUT, PREFIX + 5000)]),
        # Ends exactly at the block end, 131072 (checkpoint 130944).
        (11072, [2000], [(RESUME, CUT), (CUT, 131072)]),
        # The first chunk stops at the block end; the last chunk starts at the
        # boundary and ends in that block: was 3 passes (checkpoint 134912).
        # (Crossing chunks come only from the DP pull-up, not this split.)
        (15000, [12000], [(RESUME, 131072), (131072, PREFIX + 15000)]),
        # A first chunk ending in its block: the first-chunk snapshot, as before.
        (5000, [], [(RESUME, PREFIX + 5000)]),
        # Runs past its block: the resumed block's column (unchanged).
        (15000, [], [(RESUME, PREFIX + 15000)]),
    ],
    ids=["mid-block", "block-end", "block-aligned", "first-chunk", "past-block"],
)
def test_last_chunk_ending_in_its_block_runs_in_one_pass(
    new_tokens, budgets, spans, num_spec, retention_interval
):
    engine = _Engine(
        snapshot=True, num_spec=num_spec, retention_interval=retention_interval
    )
    request = _next_turn(engine, new_tokens)
    _, hit, _ = engine.manager.get_computed_blocks(request)
    assert hit == RESUME and RESUME % BLOCK != 0
    assert engine.prefill(request, budgets) == spans
    prompt_len = PREFIX + new_tokens
    checkpoint = _checkpoint(prompt_len)
    # P/D: the connector sends the state at the prompt end.
    assert engine.sent_state(request) == prompt_len
    # The checkpoint is offered to the store under its own boundary.
    assert checkpoint in [t for _, t in engine.offloads["t1"]]
    engine.finish(request)
    # The next turn resumes at the checkpoint (or the full block at the end).
    nxt = _turn("t2", list(request.all_token_ids) + list(range(7, 107)))
    _, num_hit, _ = engine.manager.get_computed_blocks(nxt)
    assert num_hit == (
        prompt_len
        # A full block at the prompt end is kept by dense retention and by a
        # one-block grid (it is a grid point).
        if prompt_len % BLOCK == 0 and retention_interval in (None, BLOCK)
        else checkpoint
    )
    assert engine.stale_blocks() == []


@pytest.mark.parametrize(
    "retention_interval", [None, 0, BLOCK], ids=["dense", "sparse", "grid"]
)
@pytest.mark.parametrize(
    ("new_tokens", "budgets"),
    [(5000, [2000]), (11072, [2000]), (15000, [12000]), (5000, []), (15000, [])],
)
def test_one_pass_caches_what_the_split_cached(new_tokens, budgets, retention_interval):
    """Same prefix-cache contents and hits as the split path (snapshot off):
    the same states under the same hashes, the same boundaries offered to the
    store, the same hit for the next turn."""
    results = []
    for snapshot in (False, True):
        engine = _Engine(snapshot=snapshot, retention_interval=retention_interval)
        request = _next_turn(engine, new_tokens)
        spans = engine.prefill(request, budgets)
        engine.finish(request)
        nxt = _turn("t2", list(request.all_token_ids) + list(range(7, 107)))
        _, num_hit, _ = engine.manager.get_computed_blocks(nxt)
        offered = sorted({t for _, t in engine.offloads.get("t1", [])})
        results.append((engine.hashed_states(), offered, num_hit, len(spans)))
        assert engine.stale_blocks() == []
    (split_states, split_offered, split_hit, split_passes) = results[0]
    (states, offered, hit, passes) = results[1]
    assert states == split_states
    assert offered == split_offered
    assert hit == split_hit
    assert passes <= split_passes


@pytest.mark.parametrize(
    ("prompt_len", "spans"),
    [
        # Last chunk [32768, 40000) starts at a boundary, ends in its block: the
        # snapshot goes to column 0.
        (40000, [(0, 16384), (16384, 32768), (32768, 40000)]),
        # [16384, 20000): no column below the initial state, so it still stops
        # at the checkpoint.
        (20000, [(0, 16384), (16384, 19968), (19968, 20000)]),
        # Fits one chunk: no initial state, stops at the checkpoint.
        (5000, [(0, 4992), (4992, 5000)]),
    ],
)
def test_cold_start(prompt_len, spans):
    engine = _Engine(snapshot=True)
    request = _turn("cold", list(range(1, prompt_len + 1)))
    assert engine.prefill(request) == spans
    assert engine.sent_state(request) == prompt_len
    engine.finish(request)
    nxt = _turn("t2", list(request.all_token_ids) + [5] * 100)
    _, num_hit, _ = engine.manager.get_computed_blocks(nxt)
    assert num_hit == _checkpoint(prompt_len)


def test_preempted_after_the_first_chunk():
    """Preempted after its budget-cut first chunk, the request is re-added with
    its whole table and resumes from the prefix cache: the remainder is a first
    chunk again, one pass, and the cache stays exact."""
    engine = _Engine(snapshot=True)
    request = _next_turn(engine, 5000)
    start, end, new = engine.allocate(request, 2000)
    engine.forward(request, start, end, new)
    engine.finish(request)
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0
    _, hit, _ = engine.manager.get_computed_blocks(request)
    assert hit == RESUME
    assert engine.prefill(request) == [(RESUME, PREFIX + 5000)]
    assert engine.sent_state(request) == PREFIX + 5000


def test_preempted_in_the_step_that_placed_the_snapshot():
    """A request preempted after its last chunk was allocated (later in the same
    step) has its column write dropped with its blocks."""
    engine = _Engine(snapshot=True)
    request = _next_turn(engine, 5000)
    start, end, new = engine.allocate(request, 2000)
    engine.forward(request, start, end, new)
    start, end, _ = engine.allocate(request)
    assert (start, end) == (CUT, PREFIX + 5000)
    engine.finish(request)
    assert engine.manager.take_block_table_writes() == {}


def test_budget_cut_chunk_ends_on_the_hash_grid():
    """A budget cut inside the block ends on the hash grid, so the last chunk
    starts there and can export its checkpoint (the export needs a hash-aligned
    query start); a cut outside it would have split the last chunk again."""
    engine = _Engine(snapshot=True)
    request = _next_turn(engine, 5000)
    for budget in (1, 127, 128, 2000, 2047):
        num_new = engine.split(request, budget, RESUME)
        assert num_new == (budget // HASH * HASH or budget)


def test_snapshot_block_retires_after_the_step():
    """The snapshot block of a running request sits below its running state: a
    decode step releases it to the prefix cache like a first-chunk snapshot."""
    engine = _Engine(snapshot=True)
    request = _next_turn(engine, 5000)
    engine.prefill(request, [2000])
    col = RESUME // BLOCK - 1
    block = engine.mamba.req_to_blocks["t1"][col]
    assert block.block_hash_num_tokens == _checkpoint(PREFIX + 5000)
    assert block.ref_cnt == 1
    request.append_output_token_ids(11)
    assert engine.manager.allocate_slots(request, 1) is not None
    assert engine.mamba.req_to_blocks["t1"][col].is_null
    assert block.ref_cnt == 0 and block.block_hash_num_tokens == _checkpoint(
        PREFIX + 5000
    )
    assert engine.manager.take_block_table_writes() == {}


def test_stale_worker_column_never_receives_the_export():
    """Cold 40000: the worker's column 0 still holds the block of the first
    chunk's state after the scheduler retired it. Another request takes that
    block before the last chunk. The last chunk's checkpoint goes to the block
    the scheduler wrote into the column, never to the other request's block."""
    engine = _Engine(snapshot=True)
    request = _turn("cold", list(range(1, 40001)))
    for _ in range(2):
        start, end, new = engine.allocate(request)
        engine.forward(request, start, end, new)
    stale = engine.tables["cold"][0]
    # The retirement the next allocation runs first (synchronous scheduling).
    engine.manager.remove_skipped_blocks("cold", 32768)
    assert engine.mamba.req_to_blocks["cold"][0].is_null
    pool = engine.manager.block_pool
    taken = []
    while not taken or taken[-1].block_id != stale:
        taken.extend(pool.get_new_blocks(1))
    engine.state[stale] = -999  # the other request's state
    start, end, new = engine.allocate(request)
    assert (start, end) == (32768, 40000)
    engine.forward(request, start, end, new)
    assert engine.state[stale] == -999
    snapshot = engine.mamba.req_to_blocks["cold"][0]
    assert snapshot.block_id != stale
    assert engine.tables["cold"][0] == snapshot.block_id
    assert engine.state[snapshot.block_id] == _checkpoint(40000)
    pool.free_blocks(taken)
    engine.finish(request)
    nxt = _turn("t2", list(request.all_token_ids) + [5] * 100)
    _, num_hit, _ = engine.manager.get_computed_blocks(nxt)
    assert num_hit == _checkpoint(40000)
    assert engine.stale_blocks() == []


def test_block_table_writes_cover_every_snapshot_group():
    """The worker exports per row into every Mamba group: a write in only some
    of the snapshot groups would leave the others' stale ids exported into, so
    the KV cache manager refuses it."""
    import copy

    engine = _Engine(snapshot=True)
    mgr = engine.manager
    second = copy.copy(engine.mamba)
    second.kv_cache_group_id = MAMBA_GID + 1
    second._pending_block_table_writes = []
    coordinator = mgr.coordinator
    coordinator.single_type_managers = (*coordinator.single_type_managers, second)
    engine.mamba._pending_block_table_writes = [("r", 6, 11)]
    second._pending_block_table_writes = [("r", 6, 12)]
    assert mgr.take_block_table_writes() == {
        "r": [(MAMBA_GID, 6, 11), (MAMBA_GID + 1, 6, 12)]
    }
    engine.mamba._pending_block_table_writes = [("r", 6, 11)]
    with pytest.raises(RuntimeError, match="expected each of"):
        mgr.take_block_table_writes()
    assert mgr.take_block_table_writes() == {}


@pytest.mark.parametrize(
    "prompt_len",
    [40000, 60000, PREFIX + 5000],
    ids=["cold-40000", "cold-60000", "resume"],
)
@pytest.mark.parametrize("snapshot", [False, True], ids=["split", "snapshot"])
def test_async_scheduling_keeps_the_checkpoint(prompt_len, snapshot):
    """Async scheduling allocates a chunk while the previous one is in flight:
    the manager retires blocks on the processed-token basis only, so the column
    below the initial state can still hold the previous chunk's initial state.
    The prompt checkpoint must still be cached (one pass or not)."""
    engine = _Engine(snapshot=snapshot)
    if prompt_len == PREFIX + 5000:
        request = _next_turn(engine, 5000)
        budgets = [2000]
    else:
        request = _turn("cold", list(range(1, prompt_len + 1)))
        budgets = []
    rid = request.request_id
    spans = []
    prev = 0
    while request.num_computed_tokens < request.num_tokens:
        budget = budgets[len(spans)] if len(spans) < len(budgets) else BUDGET
        request.num_in_flight_tokens = prev
        start, end, new = engine.allocate(request, budget)
        request.num_in_flight_tokens = 0
        engine.forward(request, start, end, new)
        spans.append((start, end))
        prev = end - start
    checkpoint = _checkpoint(prompt_len)
    assert checkpoint in [t for _, t in engine.offloads.get(rid, [])], spans
    engine.finish(request)
    nxt = _turn("t2", list(request.all_token_ids) + [5] * 100)
    _, num_hit, _ = engine.manager.get_computed_blocks(nxt)
    assert num_hit == checkpoint, spans
    assert engine.stale_blocks() == []
