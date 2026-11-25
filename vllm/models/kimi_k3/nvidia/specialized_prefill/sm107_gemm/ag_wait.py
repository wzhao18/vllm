# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GEMM side of the overlapped SP all-gather (PR-3, ``ops/sp_all_gather``).

The KDA in_proj Q|K|V GEMM starts while peers still push their MXFP8 shards
into this rank's gathered operand. Two pieces:

* ``RotatedTileScheduler``: the persistent tile scheduler with its cluster M
  coordinate rotated by ``rot`` (this rank's shard first, then rank + 1, ...),
  the order in which the pushers deliver the peers' shards. Only the order of
  the output tiles changes; every tile is computed exactly as before.
* ``wait_rows``: before the TMA producer loads the A rows of a cluster M tile
  from a peer's shard, it waits until that tile's landed-chunk counter (raised
  by the pushers with counted, release-ordered adds) reaches the gather's
  target. Wrap-safe compare; a bounded spin that traps (never a silent hang).
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, extract_mlir_values, new_from_mlir_values
from cutlass.utils.static_persistent_tile_scheduler import (
    StaticPersistentTileScheduler,
    WorkTileInfo,
)

# ~2^30 polls x >= 64 ns: >= 60 s before the trap (a hang, not a late peer).
SPIN_LIMIT = 1 << 30


@dsl_user_op
def _ld_acquire_sys_u32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            Int32.mlir_type,
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.acquire.sys.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _nanosleep(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], "nanosleep.u32 64;", "", has_side_effects=True, loc=loc, ip=ip)


@dsl_user_op
def _trap(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], "trap;", "", has_side_effects=True, loc=loc, ip=ip)


@dsl_user_op
def _fence_proxy_async_global(*, loc=None, ip=None) -> None:
    # The acquire above is a generic-proxy load; the A tiles are read by TMA
    # (async proxy) and were written by the peers' bulk stores.
    llvm.inline_asm(
        None, [], "fence.proxy.async.global;", "", has_side_effects=True, loc=loc, ip=ip
    )


@cute.jit
def wait_rows(ctr: Int64, cl_idx: Int32, target: Int32, lo: Int32, hi: Int32):
    """Wait until counter ``cl_idx`` (u32 at ``ctr + 4 * cl_idx``) reaches
    ``target`` (wrap-safe), unless ``lo <= cl_idx < hi`` (this rank's own rows,
    in place before the GEMM starts). Executed by every lane of the producer
    warp (one coalesced poll)."""
    if cl_idx < lo or cl_idx >= hi:
        addr = ctr + Int64(cl_idx) * Int64(4)
        v = _ld_acquire_sys_u32(addr)
        n = Int32(0)
        while (v - target) < Int32(0):
            _nanosleep()
            n = n + Int32(1)
            if n >= Int32(SPIN_LIMIT):
                _trap()
            v = _ld_acquire_sys_u32(addr)
        _fence_proxy_async_global()


class RotatedTileScheduler(StaticPersistentTileScheduler):
    """``StaticPersistentTileScheduler`` whose cluster M coordinate is
    ``(m + rot) mod ncl_m``."""

    def __init__(self, params, num_persistent_clusters, current_work_linear_idx,
                 cta_id_in_cluster, num_tiles_executed, rot, ncl_m):  # fmt: skip
        super().__init__(
            params,
            num_persistent_clusters,
            current_work_linear_idx,
            cta_id_in_cluster,
            num_tiles_executed,
        )
        self.rot = rot
        self.ncl_m = ncl_m

    @staticmethod
    def wrap(base, rot, ncl_m) -> "RotatedTileScheduler":
        return RotatedTileScheduler(
            base.params,
            base.num_persistent_clusters,
            base._current_work_linear_idx,
            base.cta_id_in_cluster,
            base._num_tiles_executed,
            Int32(rot),
            Int32(ncl_m),
        )

    def __extract_mlir_values__(self):
        values = super().__extract_mlir_values__()
        values.extend(extract_mlir_values(self.rot))
        values.extend(extract_mlir_values(self.ncl_m))
        return values

    def __new_from_mlir_values__(self, values):
        base = super().__new_from_mlir_values__(values[:-2])
        return RotatedTileScheduler(
            base.params,
            base.num_persistent_clusters,
            base._current_work_linear_idx,
            base.cta_id_in_cluster,
            base._num_tiles_executed,
            new_from_mlir_values(self.rot, [values[-2]]),
            new_from_mlir_values(self.ncl_m, [values[-1]]),
        )

    def _get_current_work_for_linear_idx(self, current_work_linear_idx, *, loc=None, ip=None):
        work = super()._get_current_work_for_linear_idx(current_work_linear_idx, loc=loc, ip=ip)
        coord = work.tile_idx
        cm = Int32(self.params.cluster_shape_mn[0])
        cl = coord[0] // cm
        m = ((cl + self.rot) % self.ncl_m) * cm + (coord[0] - cl * cm)
        return WorkTileInfo(
            (cute.arch.make_warp_uniform(Int32(m)), coord[1], coord[2]),
            work.is_valid_tile,
        )
