"""Die-local phase-interleaved FC12 scheduler (vLLM Kimi K3 local addition, VENDOR.md L2).

Rubin is two dies, each with its own HBM. The host homes the local experts
``[0, split)`` on die 0 and ``[split, E)`` on die 1 (2 MiB-granular localized memory);
this scheduler lets a cluster stream only its own die's experts:

- Four atomic work-ID streams: (FC1, FC2) x die, stream index ``2 * die + phase``.
- Die ``d``'s FC1/FC2 streams are the stock phase-interleaved streams restricted to its
  expert range, in the stock linear order. Die 0 maps with ``expert_count = split``;
  die 1 primes its cursors to expert ``split - 1`` (cumulative row / block offsets of
  the stock walk) and restarts tile IDs at zero there, so the stock mapping is reused.
- A cluster reads ``%smid`` once and takes its die from a compile-time SM mask
  (``die_sm_mask32``: 32-bit words, bit ``s`` set = SM ``s`` on die 1). Clusters
  never span dies, so every CTA of a cluster computes the same die.
- FC2 tiles depend only on FC1 tiles of the same expert, so each die's stream pair is a
  self-contained copy of the stock schedule (FC2 claimed only past the die's FC1 watermark,
  checked cluster-uniformly before the claim; FC1 claimed instead until then). When
  both streams of its die are exhausted a cluster steals from the other die's streams
  (same claim protocol), which balances an uneven die split of the persistent grid.

Tiles, their accumulation order and every epilogue are unchanged: the output is
bitwise identical to ``PhaseInterleavedFc12Scheduler``; only which cluster runs which
tile (and when) differs.
"""

from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Boolean, Int32, T, Uint32, extract_mlir_values, new_from_mlir_values

from .fc12_mapping import (
    BlockPhase,
    _advance_phase_cursor,
    create_phase_interleaved_fc12_mapping_state,
    make_fc12_done_tile,
    map_phase_interleaved_fc12_work_id,
    resolve_phase_interleaved_fc1_claim_target,
)
from .fc12_scheduler import PhaseInterleavedFc12Scheduler, _to_fc12_mapping_cta_coord
from .non_clc_mixed_cga import NonClcMixedCgaSchedulerWorker
from .work_id_claim import broadcast_atomic_counter_ready, claim_work_id


@cute.jit
def _read_smid() -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [], "mov.u32 $0, %smid;", "=r", has_side_effects=True, asm_dialect=0))


@cute.jit
def _die_of_sm(sm: Int32, mask32: tuple) -> Int32:
    die = Int32(0)
    word_idx = sm >> Int32(5)
    for w in cutlass.range_constexpr(len(mask32)):
        if cutlass.const_expr(mask32[w] != 0):
            if word_idx == Int32(w):
                die = Int32((Uint32(mask32[w]) >> Uint32(sm & Int32(31))) & Uint32(1))
    return die


@cute.jit
def _prime_cursor(cursor, mapping_state, split: int):
    """Advance a fresh cursor to expert ``split - 1`` and restart its tile IDs there
    (a primed cursor is returned unchanged)."""
    fresh = cursor.expert_idx < Int32(split - 1)
    while cursor.expert_idx < Int32(split - 1):
        cursor = _advance_phase_cursor(cursor, mapping_state)
    tile_start = cursor.expert_tile_start
    tile_end = cursor.expert_tile_end
    if fresh:
        tile_start = Int32(0)
        tile_end = Int32(0)
    cursor.expert_tile_start = tile_start
    cursor.expert_tile_end = tile_end
    return cursor


@cute.jit
def _prime_die1_state(mapping_state, split: int):
    """Prime both cursors of the die-1 mapping state (expert token sizes are ready)."""
    fc1_cursor = _prime_cursor(mapping_state.fc1_cursor, mapping_state, split)
    fc2_cursor = _prime_cursor(mapping_state.fc2_cursor, mapping_state, split)
    mapping_state.fc1_cursor = fc1_cursor
    mapping_state.fc2_cursor = fc2_cursor
    return mapping_state


class _DieSplitControlState:
    """Per-cluster cadence, per-stream exhaustion / watermark bits and the current die."""

    def __init__(
        self,
        prologue_remaining: Int32,
        cycle_position: Int32,
        exhausted: Int32,
        synchronized: Int32,
        die: Int32,
    ) -> None:
        self.prologue_remaining = prologue_remaining
        self.cycle_position = cycle_position
        self.exhausted = exhausted  # bit 2 * die + (0: FC1, 1: FC2)
        self.synchronized = synchronized  # bit die: FC1 watermark observed
        self.die = die

    def _fields(self) -> Tuple:
        return (self.prologue_remaining, self.cycle_position, self.exhausted, self.synchronized, self.die)

    def __extract_mlir_values__(self) -> list:
        values = []
        for field in self._fields():
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(self, values: list) -> "_DieSplitControlState":
        value_index = 0
        rebuilt = []
        for field in self._fields():
            count = len(extract_mlir_values(field))
            rebuilt.append(new_from_mlir_values(field, values[value_index : value_index + count]))
            value_index += count
        if value_index != len(values):
            raise ValueError(
                f"_DieSplitControlState MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )
        return type(self)(*rebuilt)


class DieSplitPhaseInterleavedFc12Scheduler(PhaseInterleavedFc12Scheduler):
    """``PhaseInterleavedFc12Scheduler`` with die-local expert streams (module docstring)."""

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {**super().impl_desc_require(), "die_sm_mask32": tuple}

    def __init__(self, problem_desc, impl_desc) -> None:
        super().__init__(problem_desc, impl_desc)
        mask = impl_desc["die_sm_mask32"]
        if not mask or not all(type(w) is int and 0 <= w < (1 << 32) for w in mask):
            raise ValueError("die_sm_mask32 must be a non-empty tuple of 32-bit ints.")
        if self.non_clc_mixed_cga_config.is_mixed:
            raise NotImplementedError("die-split FC12 scheduling needs one cluster shape (no fallback clusters).")
        if self.defer_consumer_until_full_ready or self.fuse_ready_probe_and_linear1_claim:
            raise NotImplementedError("die-split FC12 scheduling runs the ordinary claim cadence only.")
        if self.expert_count < 2:
            raise ValueError("die-split FC12 scheduling needs at least two local experts.")
        self.die_sm_mask32 = tuple(mask)
        self.die_expert_split = self.expert_count // 2
        self._work_id_worker = NonClcMixedCgaSchedulerWorker(
            config=self.non_clc_mixed_cga_config, work_id_mode=self.work_id_mode, stream_count=4
        )

    @cute.jit
    def assign_device_members(
        self,
        *,
        expert_token_sizes,
        expert_token_prefix_sum,
        actual_expert_shape,
        block_idx,
        smem_workspace,
        smem_base,
        device_workspace,
        is_fallback_cluster=None,
    ) -> None:
        if cutlass.const_expr((expert_token_sizes is None) == (expert_token_prefix_sum is None)):
            raise ValueError("Exactly one of expert_token_sizes and expert_token_prefix_sum must be provided.")
        self.create_scheduler_pipelines(smem_workspace, smem_base)
        self._work_id_worker.assign_device_members(
            is_fallback_cluster=is_fallback_cluster,
            block_idx=block_idx,
            smem_workspace=smem_workspace,
            smem_base=smem_base,
            device_workspace=device_workspace,
        )
        common = dict(
            intermediate_gateup_size=self.intermediate_gateup_size,
            hidden_size=self.hidden_size,
            mapping_cta_tile_shape_mnk=self.mapping_cta_tile_shape_mnk,
            mapping_cluster_shape_mn=self.mapping_cluster_shape_mn,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            is_swap_ab=self.is_swap_ab,
            expert_token_sizes=expert_token_sizes,
            expert_token_prefix_sum=expert_token_prefix_sum,
            dfc1_m_group=self.dfc1_m_group,
        )
        # Die 0: experts [0, split). Die 1: experts [split, E), cursors primed lazily.
        self._task_mapping_state = create_phase_interleaved_fc12_mapping_state(
            expert_count=self.die_expert_split, **common
        )
        self._task_mapping_state_die1 = create_phase_interleaved_fc12_mapping_state(
            expert_count=self.expert_count, **common
        )
        self._control_state = _DieSplitControlState(
            prologue_remaining=Int32(self.fc1_prologue_tiles),
            cycle_position=(Int32(block_idx[2]) % Int32(self.interleave_cycle_length)),
            exhausted=Int32(0),
            synchronized=Int32(0),
            die=_die_of_sm(_read_smid(), self.die_sm_mask32),
        )

    @cute.jit
    def _die_fc1_watermark_ready(self, die: Int32, work_id_state, mapping_state):
        """Whether die ``die``'s FC1 stream has been claimed up to the stock watermark (or
        to its end), observed by the cluster leader and broadcast (cluster-uniform).

        The stock scheduler claims FC2 and then waits for the watermark; that relies on
        the FC1 prologue claims of ALL clusters (clusters x hint >= watermark). A die
        stream only has its own die's clusters, whose prologue can fall short (e.g. the
        shared expert holding SMs of one die), and waiting clusters claim nothing: so a
        cluster here takes an FC1 tile instead of an FC2 one until the watermark is
        reached, and never holds an FC2 tile with unclaimed FC1 dependencies."""
        fc1_counter_pointer = work_id_state.counter_pointer + Int32(2) * die
        claim_target, _ = resolve_phase_interleaved_fc1_claim_target(
            Int32(self.minimum_global_fc1_claims), mapping_state
        )
        # Pure state threading (returns the advanced claim-channel state): the
        # caller rebinds a plain local across the staged control flow.
        return broadcast_atomic_counter_ready(work_id_state, fc1_counter_pointer, claim_target)

    @cute.jit
    def gen_next_work(self):
        """Claim from this cluster's die (then the other die) until a stream yields work."""
        work_tile = make_fc12_done_tile(self.is_swap_ab)
        work_id_worker = self._work_id_worker
        # One cluster shape (no fallback groups): the worker's claim channel is a plain
        # AtomicCounterWorkIdState, threaded as a local through the loop below.
        work_id_state = work_id_worker._work_id_state
        state0 = self._task_mapping_state
        state1 = self._task_mapping_state_die1
        control_state = self._control_state
        prologue_remaining = control_state.prologue_remaining
        cycle_position = control_state.cycle_position
        exhausted = control_state.exhausted
        synchronized = control_state.synchronized
        die = control_state.die
        stream_has_work = Boolean(False)
        resolved = Boolean(False)
        split = self.die_expert_split

        while not resolved:
            if exhausted == Int32(15):
                work_tile = make_fc12_done_tile(self.is_swap_ab)
                resolved = Boolean(True)

            if not resolved:
                if ((exhausted >> (Int32(2) * die)) & Int32(3)) == Int32(3):
                    die = Int32(1) - die  # steal: the other die still has work
                fc1_bit = Int32(1) << (Int32(2) * die)
                fc2_bit = fc1_bit << Int32(1)
                want_fc1 = Boolean(True)
                if prologue_remaining <= Int32(0):
                    is_fc2_slot = (cycle_position * Int32(self.interleave_fc2_slots)) % Int32(
                        self.interleave_cycle_length
                    ) < Int32(self.interleave_fc2_slots)
                    want_fc1 = not is_fc2_slot
                if want_fc1 and (exhausted & fc1_bit) != Int32(0):
                    want_fc1 = Boolean(False)
                if (not want_fc1) and (exhausted & fc2_bit) != Int32(0):
                    want_fc1 = Boolean(True)
                die_bit = Int32(1) << die
                if (not want_fc1) and (synchronized & die_bit) == Int32(0):
                    # FC1 left (else FC2 is safe): FC2 only past the die's FC1 watermark.
                    if (exhausted & fc1_bit) == Int32(0):
                        ready = Boolean(False)
                        if die == Int32(0):
                            ready, work_id_state = self._die_fc1_watermark_ready(die, work_id_state, state0)
                        else:
                            state1 = _prime_die1_state(state1, split)
                            ready, work_id_state = self._die_fc1_watermark_ready(die, work_id_state, state1)
                        if ready:
                            synchronized = synchronized | die_bit
                        else:
                            want_fc1 = Boolean(True)

                stream_index = Int32(2) * die + Int32(1)
                phase = Int32(BlockPhase.Linear2)
                if want_fc1:
                    stream_index = Int32(2) * die
                    phase = Int32(BlockPhase.Linear1)
                linear_work_id, work_id_state = claim_work_id(work_id_state, atomic_counter_index=stream_index)
                cta_id_in_mapping_cluster = _to_fc12_mapping_cta_coord(
                    work_id_worker.cta_coord_in_preferred_cluster, self.is_swap_ab
                )
                if die == Int32(0):
                    work_tile, stream_has_work, state0 = map_phase_interleaved_fc12_work_id(
                        linear_work_id, phase, cta_id_in_mapping_cluster, state0
                    )
                else:
                    state1 = _prime_die1_state(state1, split)
                    work_tile, stream_has_work, state1 = map_phase_interleaved_fc12_work_id(
                        linear_work_id, phase, cta_id_in_mapping_cluster, state1
                    )
                if stream_has_work:
                    if prologue_remaining > Int32(0):
                        prologue_remaining = prologue_remaining - Int32(1)
                    else:
                        cycle_position = (cycle_position + Int32(1)) % Int32(self.interleave_cycle_length)
                    resolved = Boolean(True)
                else:
                    if want_fc1:
                        exhausted = exhausted | fc1_bit
                    else:
                        exhausted = exhausted | fc2_bit

        control_state.prologue_remaining = prologue_remaining
        control_state.cycle_position = cycle_position
        control_state.exhausted = exhausted
        control_state.synchronized = synchronized
        control_state.die = die
        work_id_worker._work_id_state = work_id_state
        work_id_worker.claimed_stream_index = Int32(0)  # unused by this scheduler
        self._work_id_worker = work_id_worker
        self._task_mapping_state = state0
        self._task_mapping_state_die1 = state1
        self._control_state = control_state
        return work_tile

    def __extract_mlir_values__(self) -> list:
        values = super(PhaseInterleavedFc12Scheduler, self).__extract_mlir_values__()
        for state in (
            self._work_id_worker,
            self._task_mapping_state,
            self._task_mapping_state_die1,
            self._control_state,
        ):
            values.extend(extract_mlir_values(state))
        return values

    def __new_from_mlir_values__(self, values: list) -> "DieSplitPhaseInterleavedFc12Scheduler":
        base = super(PhaseInterleavedFc12Scheduler, self)
        base_value_count = len(base.__extract_mlir_values__())
        result = base.__new_from_mlir_values__(values[:base_value_count])
        value_index = base_value_count

        def rebuild(state):
            nonlocal value_index
            count = len(extract_mlir_values(state))
            rebuilt = new_from_mlir_values(state, values[value_index : value_index + count])
            value_index += count
            return rebuilt

        result._work_id_worker = rebuild(self._work_id_worker)
        result._task_mapping_state = rebuild(self._task_mapping_state)
        result._task_mapping_state_die1 = rebuild(self._task_mapping_state_die1)
        result._control_state = rebuild(self._control_state)
        if value_index != len(values):
            raise ValueError(
                "DieSplitPhaseInterleavedFc12Scheduler MLIR value count mismatch: "
                f"consumed {value_index}, got {len(values)}."
            )
        for name, value in vars(self).items():
            if name not in ("_work_id_worker", "_task_mapping_state", "_task_mapping_state_die1", "_control_state"):
                if name not in vars(result):
                    setattr(result, name, value)
        return result


__all__ = ["DieSplitPhaseInterleavedFc12Scheduler"]
