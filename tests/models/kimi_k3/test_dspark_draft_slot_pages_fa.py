# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The P instances' release DSpark draft attention over its packed KV group on one
GPU: stock FLASH_ATTN (FlashAttentionImpl.do_kv_cache_update, then .forward with the
sliding window 2048) over the draft views a P worker builds, at the P deployments'
layouts: P TP8/DCP8 (2 KV heads, 48 draft slots per BLHNC row) and C512 TP4/DCP4
(4 KV heads, 87 slots per row), 128-token draft blocks.

Everything between the plan and the kernels is the production code: the groups
(GQADraftKVCacheGrouping over test_gqa_draft_kv_plan's K3.1 specs), the views
(get_kv_cache_config_from_groups + prepare_kernel_block_sizes + allocate_kv_cache),
the block IDs (the scheduler's KVCacheManager: slot IDs, the sliding-window manager
freeing blocks below the window, a prefix hit restoring the window) and the per-step
block tables and slot mappings (the GPU runner's BlockTables), with the deployment's
DCP size (the runner's: it turns the below-window nulling of the gathered block table
off) and without DCP (nulling on).

Every byte of the backing allocation starts as NaN (slots are never zeroed), and every
draft block the manager frees is poisoned again (another request may take it at once).
A writer prompt runs in a 16K and a 32K chunk and a tail, then an 8-row spec-decode
step; the next turn hits the writer's prefix and runs its new tokens. Each step writes
the five layers' K / V through the step's slot mapping, then runs the step's query
rows (the whole chunk, and the drafter's 8 rows at its end) through FlashAttention.
Checks: every key of the step's windows reads back bit-exact through the block table,
the only non-NaN bytes of the allocation are the live written tokens', and the rows
match a dense FP32 reference (SDPA over the window) within BF16 noise.
"""

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.models.kimi_k3.test_gqa_draft_kv_plan import (
    HIT_UNIT,
    LAYOUT,
    NUM_DRAFT,
    POINTS,
    WINDOW,
    _config,
    _plan,
)
from tests.v1.core.test_prefix_caching import make_request
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_planning import (
    _get_kv_cache_bytes_per_block,
    get_kv_cache_config_from_groups,
)
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    init_none_hash,
    resolve_dcp_kv_block_size,
)
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HEAD, GROUP = 64, 6  # head_dim; query heads per KV head (96 / 16)
POISON = 0x7FC0  # a BF16 quiet NaN
NUM_ROWS = 160  # BLHNC rows of the test's pool
MAX_LEN = 1 << 17
SPEC_ROWS = 8  # the drafter's query rows at the end of a step (k7 + 1)
# writer: a 16K chunk, a 32K chunk, a tail; then an 8-token spec-decode step
CHUNKS = (16384, 32768, 1234)
TURN_NEW = 1800  # the next turn's new tokens past the writer's prompt
P_POINTS = ("P_TP8_DCP8", "C512_TP4_DCP4")


@pytest.fixture(autouse=True)
def _hash_fn():
    init_none_hash(sha256)


def _device():
    return torch.device("cuda", 0)


def _plan_groups(point: str):
    """The planned groups with KDA prompt checkpoints inside the chunk (as K3.1
    exports them), the draft group's index."""
    groups = _plan(point, "release")
    for group in groups:
        if isinstance(group.kv_cache_spec.first_spec, MambaSpec):
            group.kv_cache_spec = UniformTypeKVCacheSpecs.from_specs(
                {
                    name: replace(
                        spec,
                        num_prefill_checkpoint_blocks=1,
                        prefill_checkpoint_alignment=HIT_UNIT,
                    )
                    for name, spec in group.kv_cache_spec.kv_cache_specs.items()
                }
            )
    (draft_id,) = [
        i for i, g in enumerate(groups) if g.layer_names[0].startswith("draft.")
    ]
    return groups, draft_id


class _Layer:
    """The scales the Attention layer passes FlashAttentionImpl."""

    def __init__(self, device):
        one = torch.ones((), dtype=torch.float32, device=device)
        self._q_scale = self._k_scale = self._v_scale = one


class PWorker:
    """One P rank's draft KV: the planner's views over one NaN-poisoned allocation,
    the scheduler's manager and the runner's BlockTables for the draft group."""

    def __init__(self, point: str, dcp: int):
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionBackend,
            FlashAttentionImpl,
        )
        from vllm.v1.worker.gpu.block_table import BlockTables
        from vllm.v1.worker.utils import (
            AttentionGroup,
            allocate_kv_cache,
            prepare_kernel_block_sizes,
            select_common_block_size,
        )

        device = _device()
        tp, block_size, _, _, _ = POINTS[point]
        self.device, self.tp = device, tp
        groups, self.draft_id = _plan_groups(point)
        draft = groups[self.draft_id]
        self.slots = draft.blocks_per_pool_block
        spec = draft.kv_cache_spec
        spec = getattr(spec, "first_spec", spec)
        assert isinstance(spec, SlidingWindowSpec)
        # (the manager's window keeps the block below the attention window that
        # FlashAttention's 256-key tiles read; the layers attend over 2048)
        assert (spec.block_size, spec.sliding_window) == (HIT_UNIT, WINDOW + 128)
        self.spec, self.H = spec, spec.num_kv_heads
        self.heads = GROUP * self.H
        self.row_bytes = _get_kv_cache_bytes_per_block(groups, LAYOUT)

        def kv_cache_config():
            return get_kv_cache_config_from_groups(
                _config(block_size), groups, NUM_ROWS * self.row_bytes
            )

        # ---- the worker's views and the kernel block FlashAttention takes
        worker_config = kv_cache_config()
        self.names = list(draft.layer_names)
        assert len(self.names) == NUM_DRAFT
        attn_groups = [[] for _ in groups]
        attn_groups[self.draft_id] = [
            AttentionGroup(FlashAttentionBackend, self.names, spec, self.draft_id)
        ]
        kernel_block_sizes = prepare_kernel_block_sizes(worker_config, attn_groups)
        self.kernel_block_size = kernel_block_sizes[self.draft_id]
        # (AttentionGroup.create_metadata_builders keeps no kernel block when the
        # backend takes the manager's block whole)
        self.backend_block_size = select_common_block_size(
            spec.block_size, [FlashAttentionBackend], [spec]
        )
        views = allocate_kv_cache(worker_config, device, LAYOUT, kernel_block_sizes)
        self.views = [views[name] for name in self.names]
        storage = self.views[0].untyped_storage()
        self.raw = torch.empty(0, dtype=torch.int16, device=device)
        self.raw.set_(storage, 0, (storage.nbytes() // 2,), (1,))
        self.raw.fill_(POISON)

        # ---- the scheduler's manager over the same plan
        scheduler_config = generate_scheduler_kv_cache_config([kv_cache_config()])
        self.scheduler_block = math.lcm(
            *(resolve_dcp_kv_block_size(g.kv_cache_spec, tp) for g in groups)
        )
        self.manager = KVCacheManager(
            scheduler_config,
            max_model_len=MAX_LEN,
            enable_caching=True,
            hash_block_size=HIT_UNIT,
            scheduler_block_size=self.scheduler_block,
            dcp_world_size=tp,
        )

        # ---- the runner's block tables (the draft group: replicated over DCP)
        self.block_tables = BlockTables(
            block_sizes=[spec.block_size],
            max_num_reqs=4,
            max_num_batched_tokens=1 << 16,
            max_num_blocks_per_group=[cdiv(MAX_LEN, spec.block_size)],
            device=device,
            cp_size=dcp,
            cp_rank=dcp - 1,
            slot_mapping_enabled=[True],
            dcp_sharded=[False],
            sliding_window_reach=[spec.num_retained_tokens],
        )
        self.impl = FlashAttentionImpl(
            self.heads, HEAD, HEAD**-0.5, self.H, None, WINDOW, "auto"
        )
        self.layer = _Layer(device)
        # dense K / V by layer and position: what was written
        self.K = [
            torch.zeros((MAX_LEN, self.H, HEAD), dtype=torch.bfloat16, device=device)
            for _ in self.names
        ]
        self.V = [torch.zeros_like(k) for k in self.K]
        self.held: dict[str, list[int]] = {}  # manager block IDs per request
        self.live: set[int] = set()  # slots holding written tokens
        self.poisoned: set[int] = set()  # freed blocks poisoned since
        self.gen = torch.Generator(device=device).manual_seed(1000 + tp + 10 * dcp)

    # ------------------------------------------------------------- bookkeeping
    def poison(self, block: int):
        for view in self.views:
            view[block].view(torch.int16).fill_(POISON)
        self.live -= set(range(block * HIT_UNIT, (block + 1) * HIT_UNIT))
        self.poisoned.add(block)

    def schedule(self, req_index, request, num_new, overwrite=False, **kwargs):
        """The manager's allocation for the step, then the worker's block table
        update with the request's new block IDs (all of them for a new request);
        blocks the manager freed are poisoned."""
        rid = request.request_id
        before = [b for b in self.held.get(rid, []) if b]
        self.manager.new_step_starts()  # (the scheduler's, every step)
        assert self.manager.allocate_slots(request, num_new, **kwargs) is not None
        ids = list(self.manager.get_block_ids(rid)[self.draft_id])
        for block in set(before) - {b for b in ids if b}:
            self.poison(block)
        new = ids if overwrite else ids[len(self.held.get(rid, [])) :]
        self.held[rid] = ids
        self.block_tables.append_block_ids(req_index, (new,), overwrite=overwrite)
        self.block_tables.apply_staged_writes()

    def step_tables(self, req_index, start, num):
        """(block table [1, width], slot mapping [num]) as the runner builds them
        for a step of ``num`` tokens from position ``start``."""
        device = self.device
        idx = torch.tensor([req_index], dtype=torch.int64, device=device)
        computed = torch.zeros(4, dtype=torch.int32, device=device)
        computed[req_index] = start
        (table,) = self.block_tables.gather_block_tables(
            idx, num_reqs_padded=1, num_computed_tokens=computed
        )
        query_start = torch.tensor([0, num], dtype=torch.int32, device=device)
        positions = torch.arange(start, start + num, device=device)
        slots = self.block_tables.compute_slot_mappings(
            idx, query_start, positions, num_tokens_padded=num
        )[0]
        return table.clone(), slots.clone()

    # ------------------------------------------------------------- one step
    def run_step(self, req_index, start, num, label):
        """Write ``num`` tokens' K / V from ``start`` in every layer, then attend
        the step's rows and the drafter's last rows; check the reads and rows."""
        end = start + num
        table, slots = self.step_tables(req_index, start, num)
        assert int((slots < 0).sum()) == 0, f"{label}: a written token without a slot"
        expected_slots = torch.tensor(
            self.held_for(req_index)[start // HIT_UNIT : cdiv(end, HIT_UNIT)],
            device=self.device,
        ).repeat_interleave(HIT_UNIT)[start % HIT_UNIT :][:num] * HIT_UNIT + (
            torch.arange(start, end, device=self.device) % HIT_UNIT
        )
        assert torch.equal(slots, expected_slots), f"{label}: slot mapping"
        for i, view in enumerate(self.views):
            key = torch.randn((num, self.H, HEAD), generator=self.gen,
                              device=self.device).bfloat16()  # fmt: skip
            value = torch.randn((num, self.H, HEAD), generator=self.gen,
                                device=self.device).bfloat16()  # fmt: skip
            self.impl.do_kv_cache_update(self.layer, key, value, view, slots)
            self.K[i][start:end] = key
            self.V[i][start:end] = value
        self.live |= set(slots.tolist())
        torch.accelerator.synchronize(self.device)
        self.check_reads(table, max(0, start - (WINDOW - 1)), end, label)
        self.check_untouched(label)
        for i, view in enumerate(self.views):
            q = torch.randn((num, self.heads, HEAD), generator=self.gen,
                            device=self.device).bfloat16()  # fmt: skip
            out = self.attend(view, q, table, slots, end)
            ref = reference(q, self.K[i], self.V[i], start, end)
            check(out, ref, f"{label} layer {i} chunk rows")
            spec = min(SPEC_ROWS, num)
            out = self.attend(view, q[-spec:], table, slots[-spec:], end)
            check(out, ref[-spec:], f"{label} layer {i} drafter rows")

    def held_for(self, req_index):
        return self.held[self.rids[req_index]]

    def attend(self, view, q, table, slots, seq_len):
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

        rows = q.shape[0]
        md = FlashAttentionMetadata(
            num_actual_tokens=rows,
            max_query_len=rows,
            query_start_loc=torch.tensor([0, rows], dtype=torch.int32,
                                         device=self.device),  # fmt: skip
            max_seq_len=seq_len,
            seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=self.device),
            block_table=table,
            slot_mapping=slots,
            causal=True,
        )
        out = torch.empty((rows, self.heads, HEAD), dtype=q.dtype, device=self.device)
        empty = q.new_empty((0, self.H, HEAD))
        self.impl.forward(self.layer, q, empty, empty, view, md, output=out)
        torch.accelerator.synchronize(self.device)
        return out

    def check_reads(self, table, lo, end, label):
        """Every key in [lo, end) reads back bit-exact through the block table."""
        pos = torch.arange(lo, end, device=self.device)
        blocks = table[0, pos // HIT_UNIT].long()
        assert bool((blocks > 0).all()), f"{label}: a window key in the null block"
        for i, view in enumerate(self.views):
            kv = view[blocks, :, pos % HIT_UNIT, :]  # [n, H, 128]
            assert torch.equal(kv[..., :HEAD], self.K[i][lo:end]), f"{label} K {i}"
            assert torch.equal(kv[..., HEAD:], self.V[i][lo:end]), f"{label} V {i}"

    def check_untouched(self, label):
        """The allocation's only non-NaN bytes are the live tokens' K / V."""
        written = int((self.raw != POISON).sum())
        assert written == len(self.live) * NUM_DRAFT * self.H * 2 * HEAD, (
            f"{label}: {written} written elements, live tokens "
            f"{len(self.live)}"
        )

    def restore(self, request, num_hit):
        """A prefix hit restores cached blocks this test poisoned when the manager
        freed them (the real cache keeps a freed block's bytes until it is reused):
        put the writer's bytes back."""
        hit = self.held[request.request_id][: num_hit // HIT_UNIT]
        for j, block in enumerate(hit):
            if block in self.poisoned:
                pos = torch.arange(j * HIT_UNIT, (j + 1) * HIT_UNIT, device=self.device)
                for i, view in enumerate(self.views):
                    view[block] = torch.cat(
                        (self.K[i][pos], self.V[i][pos]), -1
                    ).transpose(0, 1)
                self.poisoned.discard(block)
                self.live |= set((block * HIT_UNIT + pos % HIT_UNIT).tolist())


def reference(q, K, V, start, end, tile=1024):
    """FP32 sliding-window rows of query positions [start, end) over the dense keys
    (key j visible to position p iff p - 2047 <= j <= p): SDPA per query tile."""
    rows, heads = q.shape[0], q.shape[1]
    group = heads // K.shape[1]
    out = torch.empty((rows, heads, HEAD), dtype=torch.float32, device=q.device)
    for t0 in range(0, rows, tile):
        t1 = min(rows, t0 + tile)
        p = torch.arange(start + t0, start + t1, device=q.device)
        k_lo = max(0, start + t0 - (WINDOW - 1))
        j = torch.arange(k_lo, start + t1, device=q.device)
        mask = (j[None, :] <= p[:, None]) & (j[None, :] >= p[:, None] - (WINDOW - 1))
        keys = K[k_lo : start + t1].float().repeat_interleave(group, 1)
        values = V[k_lo : start + t1].float().repeat_interleave(group, 1)
        o = F.scaled_dot_product_attention(
            q[t0:t1].float().transpose(0, 1),
            keys.transpose(0, 1),
            values.transpose(0, 1),
            attn_mask=mask,
            scale=HEAD**-0.5,
        )
        out[t0:t1] = o.transpose(0, 1)
    return out


def check(out, ref, label):
    out = out.float()
    assert bool(torch.isfinite(out).all()), f"{label}: non-finite rows"
    err = (out - ref).abs().max().item()
    scale = ref.abs().max().item()
    # BF16 P and output: a few BF16 ulps of the largest output
    assert err <= max(scale * 2.0**-6, 1e-3), f"{label}: err {err} scale {scale}"


@pytest.mark.parametrize("dcp_nulling", [False, True], ids=["dcp", "no_dcp"])
@pytest.mark.parametrize("point", P_POINTS)
def test_stock_flash_attn_over_packed_draft_slots(point, dcp_nulling):
    """Stock FLASH_ATTN writes and reads the P draft's packed 128-token slots
    exactly: the windows read back through the block tables, no write outside the
    live tokens' slots, rows equal to the dense reference."""
    tp = POINTS[point][0]
    worker = PWorker(point, 1 if dcp_nulling else tp)
    # the view the backend is bound to: manager block = kernel block = 128, one
    # slot per block (row / S apart), every layer's page inside the slot
    assert worker.kernel_block_size == worker.backend_block_size == HIT_UNIT
    assert worker.slots == {"P_TP8_DCP8": 48, "C512_TP4_DCP4": 87}[point]
    for i, view in enumerate(worker.views):
        assert tuple(view.shape) == (
            NUM_ROWS * worker.slots, worker.H, HIT_UNIT, 2 * HEAD
        )  # fmt: skip
        assert view.stride(0) * 2 == worker.row_bytes // worker.slots
        assert view.stride()[1:] == (HIT_UNIT * 2 * HEAD, 2 * HEAD, 1)
        assert view.data_ptr() - worker.views[0].data_ptr() == (
            i * worker.H * HIT_UNIT * 2 * HEAD * 2
        )
    print(f"\n{point} cp{1 if dcp_nulling else tp}: FA{worker.impl.vllm_flash_attn_version}, "
          f"{worker.slots} slots of {worker.row_bytes // worker.slots} B per "
          f"{worker.row_bytes} B row, scheduler block {worker.scheduler_block}")  # fmt: skip

    # ---- the writer: chunked prompt, then a spec-decode step
    prompt = sum(CHUNKS)
    tokens = list(range(prompt))
    writer = make_request("writer", tokens, HIT_UNIT, sha256)
    worker.rids = {0: "writer", 1: "turn"}
    start = 0
    for n in CHUNKS:
        worker.schedule(0, writer, n)
        worker.run_step(0, start, n, f"writer [{start}, {start + n})")
        start += n
        writer.num_computed_tokens = start
    writer.append_output_token_ids(list(range(7, 7 + SPEC_ROWS)))
    worker.schedule(0, writer, SPEC_ROWS)
    worker.run_step(0, start, SPEC_ROWS, "writer spec-decode step")
    freed = len(worker.poisoned)
    assert freed > 0, "the sliding window freed no draft block"
    worker.manager.free(writer)

    # ---- the next turn: a prefix hit restores the writer's window
    turn = make_request("turn", tokens + list(range(9, 9 + TURN_NEW)), HIT_UNIT,
                        sha256)  # fmt: skip
    blocks, num_hit, _ = worker.manager.get_computed_blocks(turn)
    assert num_hit >= WINDOW and num_hit % HIT_UNIT == 0, num_hit
    worker.schedule(
        1, turn, turn.num_tokens - num_hit, overwrite=True,
        num_new_computed_tokens=num_hit, new_computed_blocks=blocks,
    )  # fmt: skip
    restored = worker.held["turn"][: num_hit // HIT_UNIT]
    window = restored[(num_hit - (WINDOW - 1)) // HIT_UNIT :]
    assert all(window), "the hit did not restore the draft window"
    turn.num_computed_tokens = num_hit  # (the scheduler's, once scheduled)
    worker.restore(turn, num_hit)
    worker.run_step(1, num_hit, turn.num_tokens - num_hit, f"turn hit {num_hit}")
    print(f"  {freed} draft blocks freed and poisoned in the writer's steps; turn hit "
          f"{num_hit} of {turn.num_tokens}")  # fmt: skip


@pytest.mark.parametrize("kv_heads", [2, 4])
def test_flash_attn_reads_only_blocks_the_window_keeps(kv_heads):
    """FlashAttention-2's paged kernel reads 256-key tiles aligned to 256 (head 64):
    a window starting in a tile's upper half also reads the 128-token block below
    its first block, masked. With every block the packed draft group's manager
    frees (below start - num_retained_tokens) NaN, the rows stay finite at every
    window-start offset and query length; freeing by the bare 2048 window would
    let that block's NaN through (masked keys still multiply their V by 0)."""
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionMetadata,
    )

    device = _device()
    groups, draft_id = _plan_groups("P_TP8_DCP8")
    spec = groups[draft_id].kv_cache_spec.first_spec
    heads = GROUP * kv_heads
    impl = FlashAttentionImpl(heads, HEAD, HEAD**-0.5, kv_heads, None, WINDOW, "auto")
    layer = _Layer(device)
    gen = torch.Generator(device=device).manual_seed(kv_heads)

    def finite(q, seq_len, reach):
        start = seq_len - q  # the step's first query position
        pages = cdiv(seq_len, HIT_UNIT)
        cache = torch.randn((pages + 1, kv_heads, HIT_UNIT, 2 * HEAD), generator=gen,
                            device=device).bfloat16()  # fmt: skip
        freed = max(0, start - reach) // HIT_UNIT  # blocks the manager frees
        cache[1 : freed + 1].view(torch.int16).fill_(POISON)
        md = FlashAttentionMetadata(
            num_actual_tokens=q, max_query_len=q,
            query_start_loc=torch.tensor([0, q], dtype=torch.int32, device=device),
            max_seq_len=seq_len,
            seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
            block_table=torch.arange(1, pages + 1, dtype=torch.int32,
                                     device=device)[None],
            slot_mapping=torch.full((q,), -1, device=device), causal=True,
        )  # fmt: skip
        query = torch.randn((q, heads, HEAD), generator=gen, device=device).bfloat16()
        out = torch.empty_like(query)
        empty = query.new_empty((0, kv_heads, HEAD))
        impl.forward(layer, query, empty, empty, cache, md, output=out)
        return bool(torch.isfinite(out).all())

    leaks = 0
    for q in (1, 8, 64, 1024):
        for r in range(0, 512, 16):
            seq_len = 50000 + r
            assert finite(q, seq_len, spec.num_retained_tokens), (q, seq_len)
            leaks += not finite(q, seq_len, WINDOW - 1)
    assert leaks > 0, "the bare window never exposed the block below (tile changed?)"
