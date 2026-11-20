# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-split dynamic tile claim for the prefill shared gate_up GEMM (sm_107).

The GEMM's weight rows (N) are homed half on each die (``localize_rows``).
Every cluster reads its die from ``%smid`` and claims cluster tiles of its
die's N half from one global list of ``M tiles x N cluster tiles`` entries:
die 0 takes entries from the front, die 1 from the back, so a die that runs
out of its own half continues into the other one. Within a half the entries
walk the half's N cluster tiles of one M tile before the next M tile, so a die
keeps its weight half and the A rows in flight in its own L2.

The leader CTA's scheduler warp claims (global atomics) and writes every claim
into a ring of ``STAGES`` slots in the shared memory of each CTA of the cluster
(``st.async`` completing the slot's full-barrier transaction); the TMA, MMA and
epilogue warps of every CTA read each claim and release it to the leader's
empty barrier. A claim past the end is the stop marker. The last cluster to
finish zeroes the four global counters for the next launch.
"""

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

try:
    _WorkTileInfo = utils.WorkTileInfo
except AttributeError:  # older DSL package layout
    from cutlass.utils.static_persistent_tile_scheduler import (
        WorkTileInfo as _WorkTileInfo,
    )

STAGES = 4
# Consumer warps per CTA: 8 epilogue warps, the MMA warp and the TMA warp.
CONSUMER_WARPS = 10


@dsl_user_op
def _smid(*, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(), [], "mov.u32 $0, %smid;", "=r",
            has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _ld_global_u8(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.u8 $0, [$1];", "=r,l",
            has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _atom_add(addr: Int64, value: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
            "atom.global.add.u32 $0, [$1], $2;", "=r,l,r",
            has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _atom_exch(addr: Int64, value: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
            "atom.global.exch.b32 $0, [$1], $2;", "=r,l,r",
            has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _st_peer_i32(smem_pointer, value: Int32, mbar_pointer, rank, *, loc=None, ip=None):
    """``value`` into CTA ``rank``'s copy of ``smem_pointer``, completing 4 bytes
    of the transaction on its copy of ``mbar_pointer``."""
    smem = llvm.ptrtoint(T.i32(), smem_pointer.llvm_ptr, loc=loc, ip=ip)
    mbar = llvm.ptrtoint(T.i32(), mbar_pointer.llvm_ptr, loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [smem, Int32(value).ir_value(loc=loc, ip=ip), mbar, Int32(rank).ir_value(loc=loc, ip=ip)],
        "{ .reg .u32 ra, rb; mapa.shared::cluster.u32 ra, $0, $3; "
        "mapa.shared::cluster.u32 rb, $2, $3; "
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 [ra], $1, [rb]; }",
        "r,r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _arrive_expect_tx_peer(mbar_pointer, nbytes: Int32, rank, *, loc=None, ip=None):
    mbar = llvm.ptrtoint(T.i32(), mbar_pointer.llvm_ptr, loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [mbar, Int32(rank).ir_value(loc=loc, ip=ip), Int32(nbytes).ir_value(loc=loc, ip=ip)],
        "{ .reg .u32 rb; mapa.shared::cluster.u32 rb, $0, $1; "
        "mbarrier.arrive.expect_tx.shared::cluster.b64 _, [rb], $2; }",
        "r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


def make_pipeline(barrier_storage, cluster_size: int):
    """The claim ring's barriers (call before the kernel's cluster sync)."""
    return pipeline.PipelineAsync.create(
        num_stages=STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, CONSUMER_WARPS * cluster_size
        ),
        barrier_storage=barrier_storage,
        defer_sync=True,
    )


@cute.jit
def produce(
    sched_pipeline,
    slot_ptr: cute.Pointer,
    ctr: Int64,
    die_map: Int64,
    n_tiles: Int32,
    num_clusters: Int32,
    cluster_size: cutlass.Constexpr,
):
    """Scheduler warp of the cluster's leader CTA: claim until past the end."""
    lane = cute.arch.thread_idx()[0] % Int32(32)
    die = Int32(0)
    if lane == 0:
        die = _ld_global_u8(die_map + Int64(_smid()))
    die = cute.arch.shuffle_sync(die, offset=0, mask=0xFFFFFFFF, mask_and_clamp=31)
    ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, STAGES)
    half = n_tiles // 2
    own = ctr + Int64(4) * Int64(die)
    other = ctr + Int64(4) * Int64(1 - die)
    entry = Int32(0)
    while entry >= 0:
        sched_pipeline.producer_acquire(ps)
        entry = Int32(-1)
        if lane == 0:
            # Both dies walk their halves in the same M order (the A rows in
            # flight are shared through L2); a die that runs out continues in
            # the other half.
            j = _atom_add(own, Int32(1))
            if j < half:
                entry = die * half + j
            else:
                j2 = _atom_add(other, Int32(1))
                if j2 < half:
                    entry = (1 - die) * half + j2
                else:
                    done = _atom_add(ctr + Int64(8), Int32(1))
                    if done == num_clusters - 1:
                        for c in cutlass.range_constexpr(3):
                            _atom_exch(ctr + Int64(4 * c), Int32(0))
        entry = cute.arch.shuffle_sync(entry, offset=0, mask=0xFFFFFFFF, mask_and_clamp=31)
        full = sched_pipeline.sync_object_full.get_barrier(ps.index)
        if lane < cluster_size:
            _st_peer_i32(slot_ptr + ps.index * 4, entry, full, lane)
            _arrive_expect_tx_peer(full, Int32(4), lane)
        ps.advance()


@cute.jit
def consume(
    sched_pipeline,
    slot_ptr: cute.Pointer,
    cs: pipeline.PipelineState,
    n_tiles: Int32,
    n_half: Int32,
    cluster_m: cutlass.Constexpr,
    cluster_n: cutlass.Constexpr,
    cta_m: Int32,
    cta_n: Int32,
    group: cutlass.Constexpr = 0,
):
    """Next claim of this warp: (WorkTileInfo in CTA-tile units, state).

    ``group`` N cluster tiles of a die's half are walked per M sweep (0: the
    whole half): a smaller group keeps less of the die's weight half in its L2
    at the cost of reading the A rows once per group."""
    sched_pipeline.consumer_wait(cs)
    slots = cute.make_tensor(slot_ptr, cute.make_layout((STAGES * 4,)))
    entry = slots[cs.index * 4]
    cute.arch.fence_acq_rel_cta()
    with cute.arch.elect_one():
        sched_pipeline.sync_object_empty.arrive(cs.index, Int32(0))
    cs.advance()
    valid = entry >= 0
    e = Int32(0)
    if entry >= 0:
        e = entry
    half = n_tiles // 2
    j = e
    n_off = Int32(0)
    if e >= half:
        j = e - half
        n_off = n_half
    if cutlass.const_expr(group == 0):
        mt = j // n_half
        nt = n_off + j % n_half
    else:
        per_group = (n_tiles // (2 * n_half)) * group
        grp = j // per_group
        r = j - grp * per_group
        mt = r // group
        nt = n_off + grp * group + r % group
    coord = (mt * cluster_m + cta_m, nt * cluster_n + cta_n, Int32(0))
    return _WorkTileInfo(coord, valid), cs


def localize_rows(t: torch.Tensor) -> torch.Tensor:
    """``t`` ([N, ...] contiguous) with rows [0, N/2) homed on die 0 and
    [N/2, N) on die 1; each half must be a whole number of 2 MiB chunks."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    nbytes = t.numel() * t.element_size()
    chunk = 2 << 20
    if nbytes % (2 * chunk):
        raise ValueError(f"die-split rows need 2 x k x 2 MiB, got {nbytes} B")
    half = nbytes // (2 * chunk)
    return dies.localized_like(t, [0] * half + [1] * half)


def die_map(device) -> torch.Tensor:
    """uint8 [1024]: die of each %smid (the scheduler warp's lookup)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    idx = torch.device(device).index or 0
    doms = dies.sm_domains(idx)
    m = torch.zeros(1024, dtype=torch.uint8)
    m[: len(doms)] = torch.tensor(doms, dtype=torch.uint8)
    return m.to(device)
