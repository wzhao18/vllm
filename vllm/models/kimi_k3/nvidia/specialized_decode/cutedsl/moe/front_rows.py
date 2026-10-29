# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE front and shared expert above the 64-row mailboxes (65 rows up to the front
box's capacity): generic kernels for any verify length.

Main stream: ``MoeFrontRows`` (``MoeFrontTmem64``'s three packed front tiles and, at
``MERGED_ROWS``, its six shared gate/up tiles; weights in tensor memory before the PDL
wait, one 64- or 128-token tile per cluster column) multicasts this rank's router
logits and latent-down shard of every token into the token's row of the front box
(merged: and stores h, SiTU of the BF16-rounded gate / up); the routing kernel
(``RouteSitu``, route + copy roles, polled) ranks each token's top 16 and copies its
gathered latent, with the experts' plan in 64-row blocks (token t at row t % 64 of
block t // 64). Per token, all of it is bitwise the 64-row kernels' outputs.

Side stream (the caller's): the shared expert into the plain partial [M, 7168]: at
merged rows the down rows of h (after ``route``), else cuBLAS gate/up with FP32
output, the routing kernel's SiTU role and the down rows; the down rows by cuBLAS,
or ``PreparedProjection`` at rows with a split-K tactic (``PreparedSitu`` likewise for
gate/up + SiTU). Every value is rounded where the 64-row path rounds it (gate and up
to BF16 before SiTU, BF16 h and partial); the GEMMs sum K in their own orders (FP32
accumulation reorders of the stock layer's GEMMs)."""

import functools

import cutlass as c
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from ...shapes import Shape, rows_policy
from ..common.host import (
    dynamic_view,
    fake,
    prepare_shared_carveout,
    stream,
    to_cute,
)
from ..gemm.projection import PreparedProjection, matrix_view, weight_map
from ..gemm.splitk_situ import PreparedSitu
from .experts import PLAN_BYTES
from .front import WIDTH, compile_route
from .shared import compile_down, weight_maps

# Split-K tactics (M tile, N tile, split, stages) of the shared expert's GEMMs by rows
# ({q | "*": rows spec}, ``shapes.rows_policy``); rows without one run cuBLAS, which
# wins at every row count here (TP16 Rubin, 1 GPU, 8 weight copies rotated, us per
# call at M 72 / 128 / 256 / 512): gate/up + SiTU 6.3 / 6.7 / 7.3 / 10.2 against the
# best split-K tactic 6.2 / 8.7 / 10.9 / 12.1; down 2.0 / 2.2 / 2.8 / 3.5 against
# 3.0 / 3.1 / 5.2 / 8.1.
SHARED_GATE_ROWS = {"*": ""}
SHARED_DOWN_ROWS = {"*": ""}
# Merged rows whose shared down rows run ``shared.SharedDown`` (the 64-row tail's
# persistent down kernel, MMA N up to 128 tokens) on this many SMs beside the wide
# experts, peer-storing straight into the tail's RS slots (no cuBLAS down, no
# ``PushShared``); the experts' persistent grid leaves those SMs (``moe.prepared``).
# The RS rows are bitwise the cuBLAS down + PushShared rows (one K order; checked at
# M65 / 72 / 96 / 112 / 128). The experts' FC2 partition follows their grid: a stated
# FP32 reorder (at grid 208 vs 212 their partial rows are bitwise at M65 / 96 / 104 /
# 120 / 128 on real routes, one row of M112 differs by one BF16 ulp). The cuBLAS
# down's CTAs held the SMs the experts start on
# (224 CTAs at 81-96 rows): 1 GPU, one wide MoE layer on real routes, us per layer at
# M96 / M128: cuBLAS down + push 70.11 / 78.27, SharedDown on 4 / 8 / 16 SMs 68.39 /
# 69.00 / 69.63 and 78.25 / 78.23 / 78.41; the experts start 0.10 us before routing
# ends instead of 1.89 / 0.54 after, run 0.4-1.6 us longer on fewer SMs, and SharedDown
# (4 SMs: 30.6 / 37.0 us) ends 10.6 / 12.8 us before them. 16 ranks (32 layers, DL3):
# B24 (96 rows) -10.2 us/step, B32 (128 rows) +20.2: rows above 96 keep the cuBLAS down.
SHARED_DOWN_SMS = {"*": "65-96:4"}
# Token columns per front tile (``MoeFrontRows``: 64 or 128) by rows. Its (3 front +
# 6 gate/up for MERGED_ROWS) x ceil(M / n) clusters of 8 take one wave on Rubin up to
# 22 clusters: merged 64-token tiles to 128 rows, 128 to 256; the front alone 64 to
# 448 rows, 128 above (1 GPU, real weights, us per call, front alone / front +
# routing: M384 5.1 / 9.0 at 64 against 5.5 / 10.1 at 128; M512 8.7 / 13.3 against
# 6.4 / 10.7). Both tiles publish bitwise-equal rows.
FRONT_N_ROWS = {"*": "65-128:64,129-192:96,193-256:128,257-448:64,449-768:128"}
# Rows whose front kernel also runs the shared expert's gate/up + SiTU (``MoeFrontRows``
# with the gate/up tiles: h for the down rows, as the 64-row merged front); other rows
# run the whole shared expert beside routing and the routed experts. Their persistent
# grid loses every SM a shared-expert CTA holds, so the shared expert costs the layer
# most of its time either way; merged, only the down rows remain beside the experts.
# 1 GPU chain (o-projection stand-in, front, routing, the wide routed experts on real
# routes), us per layer over the experts alone, at M128 / 256 / 512: no shared expert
# +7.8 / +8.8 / +9.6; cuBLAS shared expert +12.3 / +17.1 / +19.2; merged (n 64 / 128 /
# 128) + cuBLAS down +10.4 / +16.3 / +23.7. Fewer experts CTAs (196-200) to leave SMs
# to the shared expert lose 1.3-5.8 us more.
MERGED_ROWS = {"*": "65-256:1"}
# Merged rows (cuBLAS down, not ``SHARED_DOWN_SMS``) whose shared down rows start as
# soon as the front kernel completes (the side stream forks between the front and
# routing kernels: the down overlaps routing instead of the persistent routed experts'
# launch), with the shared rows' RS push after the routed experts (``PreparedMoE``); 2:
# the experts also wait for the down rows. Same kernels and inputs: bitwise. With the
# experts pushing the latent (EXPERTS_PUSH), the cuBLAS down + push ended ~3 us after
# LatentRows at 192-256 rows (it gets SMs only as the experts' grid drains) and gated
# NormRows. 16 ranks (axhb-sw3, mirrored, 93 layers, 32k), us/step: B17 (136 rows)
# -249, B24 -228, B28 -171, B32 -182.
SHARED_EARLY_ROWS = {"*": "129-256:1"}
# Unmerged rows (not MERGED_ROWS) whose whole shared expert forks to the side stream at
# a later point than before the front kernel (``PreparedMoE.run_front_rows``): 1 once
# the front kernel has completed (beside routing), 2 once routing has completed (beside
# the routed experts). Same kernels and inputs: bitwise. 1 GPU chain (o-projection
# stand-in, FrontRows, routing, the wide experts on real routes, layer 3), us per layer
# over the experts alone, before / 1 / 2: M384 20.0 / 22.6 / 23.2, M512 21.5 / 21.5 /
# 21.8, M768 21.9 / 20.3 / 20.7. Empty until a 16-rank screen pays (the routing kernel
# shares SMs with the cuBLAS gate/up there).
SHARED_FORK_ROWS = {"*": ""}
# Copy CTAs whose warps all hold their chunks before any warp stores (``RouteSitu``'s
# ``sync_copy``): the plan's shuffles do not queue behind sibling warps' polls.
COPY_SYNC_ROWS = {"*": "65-768:1"}
# Rows whose routing runs the 64-row path's lean sigmoids (``front.ROUTE_RCP_WIDE``:
# each thread's exact sigmoids share one div.rn fallback branch) and its copy role on
# two CTAs per token (``front.COPY_HALVES_WIDE``: one 8-column chunk per thread, the
# plan's 64-row blocks as ``copy`` writes them). Same ids, weights, latent and plan.
ROUTE_RCP_ROWS = {"*": ""}
COPY_HALVES_ROWS = {"*": ""}
# Rows whose routing releases the wide experts once this rank's front kernel has
# completed (``RouteSitu`` early="front", the 64-row merged front's
# RELEASE_AFTER_FRONT_WIDE): their launch and prologue overlap routing; they still
# wait for routing's plan. Only rows whose shared down runs SharedDown on the SMs the
# experts' grid leaves (SHARED_DOWN_SMS). axl wrel (ds33w, before the experts' push):
# B9 +45, B10 +20 us/step; empty until it pays.
RELEASE_AFTER_FRONT_ROWS = {"*": ""}
# Rows whose routing kernel runs with the whole-SM shared-memory carve-out (it uses
# ~14 KB), as the 64-row front's ``front.ROUTE_CARVEOUT_WIDE``: an SM keeps the
# carve-out of the kernel it runs, so at the default the wide experts' big CTAs,
# launched by PDL, cannot become resident on routing's SMs until it exits. A launch
# attribute: bitwise.
ROUTE_CARVEOUT_ROWS = {"*": ""}


@functools.cache
def compile_front_rows(valid, split, shard, hidden, rows, device, n=64):
    """``MoeFrontRows`` over the packed front rows (router ``valid`` padded to
    ``split``, then the latent shard) and ``hidden`` gate/up pairs (0: none) into a
    front box of ``rows`` rows, ``n``-token tiles; any tokens."""
    from .front_tmem import MoeFrontRows

    kernel = MoeFrontRows(
        valid=valid, split_rows=split, shard=shard, hidden=hidden, n=n
    )

    def matrix(rank=3):
        shape = tuple(cute.sym_int() for _ in range(rank))
        stride = (cute.sym_int(64), 1, 0)[:rank]
        return make_fake_tensor(c.BFloat16, shape, stride, assumed_align=16)

    args: tuple = (matrix(), matrix(), matrix(2), fake((1, rows, WIDTH)))
    args += (c.Int64(0), Int32(0), Int32(0), c.Float32(1), c.Float32(1))
    args += (c.Int64(0), c.Int64(0), make_fake_stream())
    with torch.accelerator.device_index(device):
        arch = "sm_{}{}a".format(*torch.cuda.get_device_capability(device))
        return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")


class FrontRows:
    def __init__(
        self, moe, packed, box, storage, batches, beta, linear_beta, *, plan, q=1,
    ):  # fmt: skip
        """``packed``: this rank's front rows [router shard; pad; latent-down shard];
        ``box``: the front mailbox (``buffer`` [1, capacity, WIDTH], ``mc``, ``rank``);
        ``storage``: MoEScratch.front at the capacity (logits, gate/up, ids, weights,
        h, shared partial, latent); ``plan``: the experts' plan blocks (at least
        ceil(capacity / 64) x PLAN_BYTES bytes); ``batches``: rows above 64."""
        experts = moe.shared_experts
        gate_up, down = experts.gate_up_proj.weight, experts.down_proj.weight
        self.gate_up_weight, self.down_weight = gate_up, down
        self.shared = shared = gate_up.shape[0] // 2
        tp = 6144 // shared
        shard, valid = 3584 // tp, 896 // tp
        rows = box.buffer.shape[1]
        if min(batches) <= 64 or max(batches) > rows:
            raise ValueError(f"FrontRows serves 65-{rows} rows, not {sorted(batches)}")
        self.device, dev = packed.device, packed.device.index
        split = packed.shape[0] - shard
        # Kept alive: the kernels read the tensor maps.
        self.maps = (weight_map(packed, 128), weight_map(gate_up, 64))
        self.front_args = (matrix_view(packed),)
        box_args = (to_cute(box.buffer), c.Int64(box.mc))
        box_args += (Int32(box.rank * valid), Int32(box.rank * shard))
        box_args += tuple(map(c.Float32, (beta, linear_beta)))
        self.box_args = box_args + tuple(c.Int64(m.data_ptr()) for m in self.maps)
        bias = to_cute(moe.gate.e_score_correction_bias)
        scale, renorm = moe.routed_scaling_factor, moe.moe_renormalize
        scalars = tuple(map(c.Float32, (scale, beta, linear_beta)))
        plan = plan.view(torch.uint8).reshape(-1)
        shapes = {b: Shape(b, q) for b in batches}
        self.merged = {b for b, s in shapes.items() if rows_policy(MERGED_ROWS, s, 0)}
        self.early = {
            b: n
            for b, s in shapes.items()
            if b in self.merged and (n := rows_policy(SHARED_EARLY_ROWS, s, 0))
        }
        self.fork = {
            b: n
            for b, s in shapes.items()
            if b not in self.merged and (n := rows_policy(SHARED_FORK_ROWS, s, 0))
        }
        gate_rows = {
            b: rows_policy(SHARED_GATE_ROWS, s, None) for b, s in shapes.items()
        }
        down_rows = {
            b: rows_policy(SHARED_DOWN_ROWS, s, None) for b, s in shapes.items()
        }
        gate_rows = {
            b: t for b, t in gate_rows.items() if t is not None and b not in self.merged
        }
        down_rows = {b: t for b, t in down_rows.items() if t is not None}
        self.gate_situ = PreparedSitu(gate_up, gate_rows, beta, linear_beta)
        self.down = PreparedProjection(down, down_rows).prepare()
        # SHARED_DOWN_SMS batches: SharedDown's SM count and code
        self.down_sms = {
            b: n
            for b, s in shapes.items()
            if b in self.merged and (n := rows_policy(SHARED_DOWN_SMS, s, 0))
        }
        self.shared_down = {
            b: compile_down(tp, b, dev, n) for b, n in self.down_sms.items()
        }
        if self.shared_down:  # kept alive: the kernels read the tensor maps
            self.down_maps = weight_maps(gate_up, down)
        self.plans: dict[int, tuple] = {}
        for b in batches:
            views = tuple(t[:b] for t in storage)
            logits, gate, ids, weights, h, _, latent = map(to_cute, views)
            sync = bool(rows_policy(COPY_SYNC_ROWS, shapes[b], 0))
            rcp = bool(rows_policy(ROUTE_RCP_ROWS, shapes[b], 0))
            halves = bool(rows_policy(COPY_HALVES_ROWS, shapes[b], 0))
            release = rows_policy(RELEASE_AFTER_FRONT_ROWS, shapes[b], 0)
            early = "front" if release and self.down_sms.get(b) else None
            route = compile_route(
                b, shared, renorm, (True, True, False), dev, True, True, early=early,
                sync_copy=sync and not halves, box_rows=rows, rcp=rcp, halves=halves,
            )  # fmt: skip
            if rows_policy(ROUTE_CARVEOUT_ROWS, shapes[b], 0):
                route = prepare_shared_carveout(route, dev, 100)
            situ = None
            if b not in self.merged and b not in self.gate_situ.kernels:
                # cuBLAS gate/up, then the SiTU role
                situ = compile_route(b, shared, renorm, (False, False, True), dev)
            args = (logits, bias, ids, weights, box_args[0], latent, gate, h, *scalars)
            blocks = to_cute(plan[: -(-b // 64) * PLAN_BYTES])
            n = rows_policy(FRONT_N_ROWS, shapes[b], 64)
            hidden = shared if b in self.merged else 0
            front = compile_front_rows(valid, split, shard, hidden, rows, dev, n)
            self.plans[b] = views, route, situ, args, blocks, front

    def merged_batch(self, batch):
        """Rows whose front kernel also leaves h: ``shared_expert`` then runs the down
        rows alone and must follow ``route`` (as ``SharedFront``'s merged batches)."""
        return batch in self.merged

    def route(self, x):
        """The front GEMM publishes this rank's logits and latent columns of every
        token (merged rows: and stores h); routing then writes ids [M, 16], weights
        [M, 16], the latent copy and the plan. Returns (ids, weights, latent) views."""
        self.front(x)
        return self.route_only(x)

    def front(self, x):
        """``route``'s front GEMM alone (``SHARED_EARLY_ROWS``: the caller forks the
        shared down rows between it and ``route_only``)."""
        views, *_, front = self.plans[x.shape[0]]
        h = dynamic_view(views[4])
        front(*self.front_args, matrix_view(x), h, *self.box_args, stream(self.device))

    def route_only(self, x):
        """``route``'s routing kernel, after ``front``."""
        views, route, _, args, blocks, _ = self.plans[x.shape[0]]
        route(*args, blocks, stream(self.device))
        return views[2], views[3], views[6]

    def push_down(self, x, push):
        """``SHARED_DOWN_SMS`` rows: the down rows of the h ``route`` stored, pushed
        into the tail's RS slots (``push``: ``PreparedTailRows.push_args``) by
        SharedDown on ``down_sms[M]`` SMs, on the current stream."""
        b = x.shape[0]
        h = self.plans[b][0][4]
        code = self.shared_down[b]
        code(to_cute(h), to_cute(self.down_maps, 64), *push, stream(self.device))

    def shared_expert(self, x, out=None):
        """The shared expert's partial [M, 7168] BF16 into ``out`` (default: the
        storage's shared rows), on the current stream; returns it. Merged rows: the
        down rows of the h ``route`` stored."""
        b = x.shape[0]
        views, _, situ, args, *_ = self.plans[b]
        h = views[4]
        out = views[5] if out is None else out
        if b not in self.merged and situ is None:
            self.gate_situ.run(x, h)
        elif b not in self.merged:
            torch.mm(x, self.gate_up_weight.t(), out_dtype=torch.float32, out=views[1])
            situ(*args, stream(self.device))
        if b in self.down.kernels:
            self.down.run(h, out)
        else:
            torch.mm(h, self.down_weight.t(), out=out)
        return out
