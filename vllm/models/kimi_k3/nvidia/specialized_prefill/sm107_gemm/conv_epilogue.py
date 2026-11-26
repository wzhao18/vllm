# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Epilogue of the KDA in_proj Q|K|V GEMM: BF16 round, width-4 causal conv
(FP32, stock FMA order), Triton-lowered SiLU, BF16 Q|K|V planes and the
optional inverse L2 norms of the Q / K head rows.

Each epilogue thread owns one output row (TMEM lane). The three previous rows
come from warp shuffles, and for the first three lanes of a warp from the
previous warp through shared memory. Rows 0-2 of every 128-row block and rows
0-2 of every sequence are left to ``conv_fixup``; the epilogue saves the
pre-conv BF16 rows that the fix-up and the conv state update read.

Side buffer rows (BF16, n columns): block b rows 0-2 at ``6b + r`` and rows
125-127 at ``6b + 3 + r - 125``; sequence s rows 0-2 at ``6 Mb + 6s + pos`` and
its last three rows at ``6 Mb + 6s + 3 + pos - (len - 3)``.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Constexpr, T, dsl_user_op

from . import dgs0
from cutlass.utils.gemm.sm100 import (
    epilogue_tmem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

_ASM = dict(
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
)


def _v(x, cls, loc, ip):
    return cls(x).ir_value(loc=loc, ip=ip)


@dsl_user_op
def pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [_v(lo, Float32, loc, ip), _v(hi, Float32, loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
            **_ASM,
        )
    )


# $0 bf16x2 out, $1 sum-of-squares out; $2..$5 rows t-3..t (bf16x2);
# $6..$13 FP32 weights (tap0 lo, tap0 hi, ..., tap3 hi); $14 sum-of-squares in.
_CONV_PAIR = """{
.reg .b32 l<4>, h<4>, yp, yl, yh;
.reg .b64 X<4>, W<4>, A;
.reg .f32 a<2>, t<2>, e<2>, d<2>, r<2>, y<2>, s;
shl.b32 l0, $2, 16;
and.b32 h0, $2, -65536;
shl.b32 l1, $3, 16;
and.b32 h1, $3, -65536;
shl.b32 l2, $4, 16;
and.b32 h2, $4, -65536;
shl.b32 l3, $5, 16;
and.b32 h3, $5, -65536;
mov.b64 X0, {l0, h0};
mov.b64 X1, {l1, h1};
mov.b64 X2, {l2, h2};
mov.b64 X3, {l3, h3};
mov.b64 W0, {$6, $7};
mov.b64 W1, {$8, $9};
mov.b64 W2, {$10, $11};
mov.b64 W3, {$12, $13};
mov.b64 A, 0;
fma.rn.f32x2 A, X0, W0, A;
fma.rn.f32x2 A, X1, W1, A;
fma.rn.f32x2 A, X2, W2, A;
fma.rn.f32x2 A, X3, W3, A;
mov.b64 {a0, a1}, A;
mul.f32 t0, a0, 0fBFB8AA3B;
mul.f32 t1, a1, 0fBFB8AA3B;
ex2.approx.ftz.f32 e0, t0;
ex2.approx.ftz.f32 e1, t1;
add.f32 d0, e0, 0f3F800000;
add.f32 d1, e1, 0f3F800000;
div.full.f32 r0, a0, d0;
div.full.f32 r1, a1, d1;
cvt.rn.bf16x2.f32 yp, r1, r0;
shl.b32 yl, yp, 16;
and.b32 yh, yp, -65536;
mov.b32 y0, yl;
mov.b32 y1, yh;
fma.rn.f32 s, y0, y0, $14;
fma.rn.f32 $1, y1, y1, s;
mov.b32 $0, yp;
}"""


@dsl_user_op
def conv_pair(x3, x2, x1, x0, w, s_in, *, loc=None, ip=None):
    """Conv + SiLU of one column pair: rows (t-3, t-2, t-1, t) as bf16x2,
    weights ``w`` = 8 FP32 (tap-major lo/hi). Returns (bf16x2, sum of squares)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [_v(x, Uint32, loc, ip) for x in (x3, x2, x1, x0)]
        + [_v(v, Float32, loc, ip) for v in w]
        + [_v(s_in, Float32, loc, ip)],
        _CONV_PAIR,
        "=r,=f," + ",".join(["r"] * 4 + ["f"] * 9),
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM,
    )
    y = Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip))
    s = Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip))
    return y, s


@dsl_user_op
def shfl_up(v: Uint32, d: int, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [_v(v, Uint32, loc, ip)],
            f"shfl.sync.up.b32 $0, $1, {d}, 0, -1;",
            "=r,r",
            has_side_effects=False,
            loc=loc,
            ip=ip,
            **_ASM,
        )
    )


@dsl_user_op
def sel(c: Int32, a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """``c != 0 ? a : b``."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [_v(c, Int32, loc, ip), _v(a, Uint32, loc, ip), _v(b, Uint32, loc, ip)],
            "{ .reg .pred p; setp.ne.s32 p, $1, 0; selp.b32 $0, $2, $3, p; }",
            "=r,r,r,r",
            has_side_effects=False,
            loc=loc,
            ip=ip,
            **_ASM,
        )
    )


@dsl_user_op
def self32(c: Int32, a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_v(c, Int32, loc, ip), _v(a, Float32, loc, ip), _v(b, Float32, loc, ip)],
            "{ .reg .pred p; setp.ne.s32 p, $1, 0; selp.f32 $0, $2, $3, p; }",
            "=f,r,f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
            **_ASM,
        )
    )


@dsl_user_op
def ldg_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [_v(addr, Int64, loc, ip)],
            "ld.global.nc.s32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            loc=loc,
            ip=ip,
            **_ASM,
        )
    )


@dsl_user_op
def ldg_f32x4(addr: Int64, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_v(addr, Int64, loc, ip)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM,
    )
    return [
        Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def stg_v4(addr: Int64, pred: Int32, a, b, c, d, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [_v(addr, Int64, loc, ip), _v(pred, Int32, loc, ip)]
        + [_v(x, Uint32, loc, ip) for x in (a, b, c, d)],
        "{ .reg .pred p; setp.ne.s32 p, $1, 0; "
        "@p st.global.v4.b32 [$0], {$2, $3, $4, $5}; }",
        "l,r,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM,
    )


@dsl_user_op
def stg_f32(addr: Int64, pred: Int32, v: Float32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [_v(addr, Int64, loc, ip), _v(pred, Int32, loc, ip), _v(v, Float32, loc, ip)],
        "{ .reg .pred p; setp.ne.s32 p, $1, 0; @p st.global.f32 [$0], $2; }",
        "l,r,f",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM,
    )


@dsl_user_op
def sts_v4(addr: Int32, pred: Int32, a, b, c, d, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [_v(addr, Int32, loc, ip), _v(pred, Int32, loc, ip)]
        + [_v(x, Uint32, loc, ip) for x in (a, b, c, d)],
        "{ .reg .pred p; setp.ne.s32 p, $1, 0; "
        "@p st.shared.v4.b32 [$0], {$2, $3, $4, $5}; }",
        "r,r,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM,
    )


@dsl_user_op
def lds_v4(addr: Int32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_v(addr, Int32, loc, ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def rsqrt_eps(s: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_v(s, Float32, loc, ip)],
            "{ .reg .f32 t; add.f32 t, $1, 0f358637BD; rsqrt.approx.ftz.f32 $0, t; }",
            "=f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
            **_ASM,
        )
    )


def b2i(c) -> Int32:
    return Int32(c)


@cute.jit
def find_seq(qsl_addr: Int64, nseq: Int32, top: Int32, nsteps: Int32, row: Int32):
    """Largest s in [0, nseq) with qsl[s] <= row (qsl[0] <= row)."""
    s = Int32(0)
    step = top
    for _ in cutlass.range(nsteps, unroll=1):
        cand = s + step
        if cand < nseq:
            if ldg_s32(qsl_addr + Int64(cand) * 4) <= row:
                s = cand
        step = step // 2
    return s


@cute.jit
def epilogue_kda_conv(
    gemm_kernel,
    epi_tidx: Int32,
    tCtAcc_base: cute.Tensor,
    tCgC_base: cute.Tensor,
    tCcC_base: cute.Tensor,
    epi_tile: cute.Tile,
    mma_tile_coord_mnl: tuple,
    m_rows: Int32,
    n_cols: Int32,
    conv: tuple,
    smem_x: Int32,
    norms: Constexpr,
    acc_consumer_state: pipeline.PipelineState,
    acc_pipeline: pipeline.PipelineAsync,
) -> pipeline.PipelineState:
    out_addr, w_addr, side_addr, qsl_addr, norm_addr, nseq, top, nsteps = conv
    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
    )
    thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
    tTR_cC = thr_copy_t2r.partition_D(
        cute.flat_divide(transform_partitioned_tensor_layout(tCcC_base), epi_tile)
    )[(None, None, None, None, None, *mma_tile_coord_mnl)]
    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

    epi_m = cute.size(tTR_tAcc.shape, mode=[3])
    epi_n = cute.size(tTR_tAcc.shape, mode=[4])
    ncol = cute.size(tTR_rAcc)
    npair = ncol // 2
    nq = npair // 4
    p_dim = n_cols // 3
    n_blk = (m_rows + 127) // 128
    bar = gemm_kernel.epilog_sync_bar_id

    # Sequence of each row block, before the accumulator wait (overlaps the MMA).
    rows = []
    for mi in cutlass.range_constexpr(epi_m):
        row = tTR_cC[(None, None, None, mi, 0)][0][0]
        s = find_seq(qsl_addr, nseq, top, nsteps, row)
        start = ldg_s32(qsl_addr + Int64(s) * 4)
        end = ldg_s32(qsl_addr + Int64(s + 1) * 4)
        rows.append((row, s, start, end))

    acc_pipeline.consumer_wait(acc_consumer_state)
    for mi in cutlass.range_constexpr(epi_m):
        row, s, start, end = rows[mi]
        rb = row % 128
        lane = row % 32
        wq = rb // 32
        pos = row - start
        ln = end - start
        valid = b2i(row < m_rows) * b2i(row < end)
        fix = b2i(rb < 3) | b2i(pos < 3)
        store = valid * (1 - fix)
        side_tile = Int64(row // 128) * 6
        side_seq = Int64(n_blk) * 6 + Int64(s) * 6
        w_head = valid * b2i(rb < 3)
        r_head = side_tile + Int64(rb)
        w_tail = valid * b2i(rb >= 125)
        r_tail = side_tile + Int64(rb - 122)
        w_shead = valid * b2i(pos < 3)
        r_shead = side_seq + Int64(pos)
        w_stail = valid * b2i(pos >= ln - 3)
        r_stail = side_seq + Int64(pos - ln + 6)
        w_xchg = b2i(lane >= 29)
        l0 = b2i(lane == 0)
        l1 = b2i(lane == 1)
        l2 = b2i(lane == 2)
        ssq = Float32(0.0)
        for ni in cutlass.range_constexpr(epi_n):
            cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, mi, ni)], tTR_rAcc)
            if cutlass.const_expr(mi == epi_m - 1 and ni == epi_n - 1):
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
            col = tTR_cC[(None, None, None, mi, ni)][0][1]
            xs = [
                pack_bf16x2(tTR_rAcc[2 * p], tTR_rAcc[2 * p + 1])
                for p in range(npair)
            ]
            # Pre-conv rows for the fix-up and the conv state.
            sid = mi * epi_n + ni
            for wr, rr in (
                (w_head, r_head),
                (w_tail, r_tail),
                (w_shead, r_shead),
                (w_stail, r_stail),
            ):
                base = side_addr + (rr * Int64(n_cols) + Int64(col)) * 2
                for q in cutlass.range_constexpr(nq):
                    stg_v4(base + q * 16, wr, *xs[4 * q : 4 * q + 4])
            # Warp-boundary rows through shared memory.
            xbase = smem_x + ((sid * 4 + wq) * 3 + (lane - 29)) * (npair * 4)
            for q in cutlass.range_constexpr(nq):
                sts_v4(xbase + q * 16, w_xchg, *xs[4 * q : 4 * q + 4])
            cute.arch.barrier(barrier_id=bar, number_of_threads=128)
            pbase = smem_x + ((sid * 4 + (wq + 3) % 4) * 3) * (npair * 4)
            prev = [[], [], []]
            for j in cutlass.range_constexpr(3):
                for q in cutlass.range_constexpr(nq):
                    prev[j] += lds_v4(pbase + j * (npair * 4) + q * 16)
            ys = []
            for p in cutlass.range_constexpr(npair):
                p1 = sel(l0, prev[2][p], shfl_up(xs[p], 1))
                p2 = sel(l0, prev[1][p], sel(l1, prev[2][p], shfl_up(xs[p], 2)))
                p3 = sel(
                    l0,
                    prev[0][p],
                    sel(l1, prev[1][p], sel(l2, prev[2][p], shfl_up(xs[p], 3))),
                )
                c = col + 2 * p
                wa = ldg_f32x4(w_addr + Int64(c) * 16)
                wb = ldg_f32x4(w_addr + Int64(c + 1) * 16)
                wv = [wa[0], wb[0], wa[1], wb[1], wa[2], wb[2], wa[3], wb[3]]
                if cutlass.const_expr(norms and p == 0):
                    ssq = self32(b2i(col % 128 == 0), Float32(0.0), ssq)
                y, ssq = conv_pair(p3, p2, p1, xs[p], wv, ssq)
                ys.append(y)
            part = col // p_dim
            cc = col - part * p_dim
            obase = out_addr + ((Int64(part) * Int64(m_rows) + Int64(row)) * Int64(p_dim) + Int64(cc)) * 2
            for q in cutlass.range_constexpr(nq):
                stg_v4(obase + q * 16, store, *ys[4 * q : 4 * q + 4])
            if cutlass.const_expr(norms):
                head_end = b2i((col + ncol) % 128 == 0) * b2i(part < 2)
                h = p_dim // 128
                naddr = norm_addr + ((Int64(part) * Int64(m_rows) + Int64(row)) * Int64(h) + Int64(cc // 128)) * 4
                stg_f32(naddr, store * head_end, rsqrt_eps(ssq))
    return acc_consumer_state


@cute.jit
def epilogue_kda_conv_drain(
    gemm_kernel,
    tidx: Int32,
    tCtAcc_base: cute.Tensor,
    tCgC_base: cute.Tensor,
    tCcC_base: cute.Tensor,
    epi_tile: cute.Tile,
    mma_tile_coord_mnl: tuple,
    m_rows: Int32,
    n_cols: Int32,
    conv: tuple,
    smem_x: Int32,
    norms: Constexpr,
    acc_consumer_state: pipeline.PipelineState,
    acc_pipeline: pipeline.PipelineAsync,
) -> pipeline.PipelineState:
    """Eight epilogue warps for the B-reuse tile (one TMEM accumulator stage):
    warp group g owns M block g. Each thread drains its row (BF16-rounded) into
    registers and releases the accumulator before the conv, so the next tile's
    MMA overlaps the conv math."""
    out_addr, w_addr, side_addr, qsl_addr, norm_addr, nseq, top, nsteps = conv
    g = tidx // 128
    epi_tidx = tidx % 128
    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
    )
    thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
    tTR_cC = thr_copy_t2r.partition_D(
        cute.flat_divide(transform_partitioned_tensor_layout(tCcC_base), epi_tile)
    )[(None, None, None, None, None, *mma_tile_coord_mnl)]
    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

    epi_m = cute.size(tTR_tAcc.shape, mode=[3])
    epi_n = cute.size(tTR_tAcc.shape, mode=[4])
    if cutlass.const_expr(epi_m != 2):
        raise ValueError(f"drain conv epilogue needs 2 M blocks per CTA, got {epi_m}")
    ncol = cute.size(tTR_rAcc)
    npair = ncol // 2
    nq = npair // 4
    p_dim = n_cols // 3
    n_blk = (m_rows + 127) // 128
    bar = gemm_kernel.epilog_sync_bar_id

    row = tTR_cC[(None, None, None, g, 0)][0][0]
    s = find_seq(qsl_addr, nseq, top, nsteps, row)
    start = ldg_s32(qsl_addr + Int64(s) * 4)
    end = ldg_s32(qsl_addr + Int64(s + 1) * 4)

    acc_pipeline.consumer_wait(acc_consumer_state)
    xs_all = []
    for ni in cutlass.range_constexpr(epi_n):
        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, g, ni)], tTR_rAcc)
        xs_all.append(
            [pack_bf16x2(tTR_rAcc[2 * p], tTR_rAcc[2 * p + 1]) for p in range(npair)]
        )
    cute.arch.fence_view_async_tmem_load()
    with cute.arch.elect_one():
        acc_pipeline.consumer_release(acc_consumer_state)
    acc_consumer_state.advance()

    rb = row % 128
    lane = row % 32
    wq = rb // 32
    pos = row - start
    ln = end - start
    valid = b2i(row < m_rows) * b2i(row < end)
    fix = b2i(rb < 3) | b2i(pos < 3)
    store = valid * (1 - fix)
    side_tile = Int64(row // 128) * 6
    side_seq = Int64(n_blk) * 6 + Int64(s) * 6
    side_w = (
        (valid * b2i(rb < 3), side_tile + Int64(rb)),
        (valid * b2i(rb >= 125), side_tile + Int64(rb - 122)),
        (valid * b2i(pos < 3), side_seq + Int64(pos)),
        (valid * b2i(pos >= ln - 3), side_seq + Int64(pos - ln + 6)),
    )
    w_xchg = b2i(lane >= 29)
    l0 = b2i(lane == 0)
    l1 = b2i(lane == 1)
    l2 = b2i(lane == 2)
    ssq = Float32(0.0)
    for ni in cutlass.range_constexpr(epi_n):
        col = tTR_cC[(None, None, None, g, ni)][0][1]
        xs = xs_all[ni]
        for wr, rr in side_w:
            base = side_addr + (rr * Int64(n_cols) + Int64(col)) * 2
            for q in cutlass.range_constexpr(nq):
                stg_v4(base + q * 16, wr, *xs[4 * q : 4 * q + 4])
        sid = g * epi_n + ni
        xbase = smem_x + ((sid * 4 + wq) * 3 + (lane - 29)) * (npair * 4)
        for q in cutlass.range_constexpr(nq):
            sts_v4(xbase + q * 16, w_xchg, *xs[4 * q : 4 * q + 4])
        cute.arch.barrier(barrier_id=bar, number_of_threads=256)
        pbase = smem_x + ((sid * 4 + (wq + 3) % 4) * 3) * (npair * 4)
        ys = []
        for q in cutlass.range_constexpr(nq):
            pv = [lds_v4(pbase + j * (npair * 4) + q * 16) for j in range(3)]
            for i in cutlass.range_constexpr(4):
                p = 4 * q + i
                p1 = sel(l0, pv[2][i], shfl_up(xs[p], 1))
                p2 = sel(l0, pv[1][i], sel(l1, pv[2][i], shfl_up(xs[p], 2)))
                p3 = sel(
                    l0,
                    pv[0][i],
                    sel(l1, pv[1][i], sel(l2, pv[2][i], shfl_up(xs[p], 3))),
                )
                c = col + 2 * p
                wa = ldg_f32x4(w_addr + Int64(c) * 16)
                wb = ldg_f32x4(w_addr + Int64(c + 1) * 16)
                wv = [wa[0], wb[0], wa[1], wb[1], wa[2], wb[2], wa[3], wb[3]]
                if cutlass.const_expr(norms and p == 0):
                    ssq = self32(b2i(col % 128 == 0), Float32(0.0), ssq)
                y, ssq = conv_pair(p3, p2, p1, xs[p], wv, ssq)
                ys.append(y)
        part = col // p_dim
        cc = col - part * p_dim
        obase = out_addr + (
            (Int64(part) * Int64(m_rows) + Int64(row)) * Int64(p_dim) + Int64(cc)
        ) * 2
        for q in cutlass.range_constexpr(nq):
            stg_v4(obase + q * 16, store, *ys[4 * q : 4 * q + 4])
        if cutlass.const_expr(norms):
            head_end = b2i((col + ncol) % 128 == 0) * b2i(part < 2)
            h = p_dim // 128
            naddr = norm_addr + (
                (Int64(part) * Int64(m_rows) + Int64(row)) * Int64(h) + Int64(cc // 128)
            ) * 4
            stg_f32(naddr, store * head_end, rsqrt_eps(ssq))
    return acc_consumer_state


# $0 bf16x2 out, $1 sum of squares out; $2..$7 taps t-3..t-1 as FP32 (lo, hi);
# $8 row t (bf16x2); $9..$16 weights (tap-major lo/hi); $17 sum of squares in.
# SiLU: a * rcp.approx(1 + e), which equals div.full.f32 while 1 + e <= 2^126;
# larger denominators take div.full itself.
_CONV_ROW = """{
.reg .b32 l3, h3, yp, yl, yh;
.reg .b64 X<4>, W<4>, A, R, D, E, T;
.reg .f32 a<2>, d<2>, r<2>, e<2>, y<2>, s;
.reg .pred p0, p1;
shl.b32 l3, $8, 16;
and.b32 h3, $8, -65536;
mov.b64 X0, {$2, $3};
mov.b64 X1, {$4, $5};
mov.b64 X2, {$6, $7};
mov.b64 X3, {l3, h3};
mov.b64 W0, {$9, $10};
mov.b64 W1, {$11, $12};
mov.b64 W2, {$13, $14};
mov.b64 W3, {$15, $16};
mov.b64 A, 0;
fma.rn.f32x2 A, X0, W0, A;
fma.rn.f32x2 A, X1, W1, A;
fma.rn.f32x2 A, X2, W2, A;
fma.rn.f32x2 A, X3, W3, A;
mov.b64 T, 0xBFB8AA3BBFB8AA3B;
mul.rn.f32x2 E, A, T;
mov.b64 {e0, e1}, E;
ex2.approx.ftz.f32 e0, e0;
ex2.approx.ftz.f32 e1, e1;
add.f32 d0, e0, 0f3F800000;
add.f32 d1, e1, 0f3F800000;
rcp.approx.ftz.f32 r0, d0;
rcp.approx.ftz.f32 r1, d1;
mov.b64 R, {r0, r1};
mul.rn.f32x2 R, A, R;
mov.b64 {r0, r1}, R;
mov.b64 {a0, a1}, A;
setp.gt.f32 p0, d0, 0f7E800000;
setp.gt.f32 p1, d1, 0f7E800000;
@p0 div.full.f32 r0, a0, d0;
@p1 div.full.f32 r1, a1, d1;
cvt.rn.bf16x2.f32 yp, r1, r0;
shl.b32 yl, yp, 16;
and.b32 yh, yp, -65536;
mov.b32 y0, yl;
mov.b32 y1, yh;
fma.rn.f32 s, y0, y0, $17;
fma.rn.f32 $1, y1, y1, s;
mov.b32 $0, yp;
}"""


@dsl_user_op
def conv_row(h, x, w, s_in, *, loc=None, ip=None):
    """h: 6 FP32 taps (rows t-3..t-1, lo/hi); x: row t bf16x2; w: 8 FP32."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [_v(v, Float32, loc, ip) for v in h]
        + [_v(x, Uint32, loc, ip)]
        + [_v(v, Float32, loc, ip) for v in w]
        + [_v(s_in, Float32, loc, ip)],
        _CONV_ROW,
        "=r,=f," + ",".join(["f"] * 6 + ["r"] + ["f"] * 9),
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM,
    )
    y = Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip))
    s = Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip))
    return y, s


@dsl_user_op
def unpack_bf16x2(x: Uint32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [_v(x, Uint32, loc, ip)],
        "{ .reg .b32 l, h; shl.b32 l, $2, 16; and.b32 h, $2, -65536; "
        "mov.b32 $0, l; mov.b32 $1, h; }",
        "=f,=f,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def lds_u32(addr: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [_v(addr, Int32, loc, ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            loc=loc,
            ip=ip,
            **_ASM,
        )
    )


@dsl_user_op
def stg_u32(addr: Int64, pred: Int32, v: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [_v(addr, Int64, loc, ip), _v(pred, Int32, loc, ip), _v(v, Uint32, loc, ip)],
        "{ .reg .pred p; setp.ne.s32 p, $1, 0; @p st.global.b32 [$0], $2; }",
        "l,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM,
    )


@dsl_user_op
def shfl_xor_f32(v: Float32, d: int, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_v(v, Float32, loc, ip)],
            f"shfl.sync.bfly.b32 $0, $1, {d}, 31, -1;",
            "=f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
            **_ASM,
        )
    )


#: Shared-memory row pitch of a drained 32-column subtile, in 32-bit words;
#: 16-byte chunks are XOR-swizzled by row % 4 against bank conflicts.
_PITCH = 16
#: Shared bytes the column-wise conv epilogue needs (two warp groups).
CONV_SMEM_BYTES = 2 * 128 * _PITCH * 4


@cute.jit
def _skel3_math(
    gbase: Int32,
    cq: Int32,
    cr: Int32,
    rc: Int32,
    out_addr: Int64,
    w_addr: Int64,
    col: Int32,
    row0: Int64,
    p_dim: Int32,
    st_on: Int32,
):
    """Microbench skeleton 3: the column pass of row chunks rc and rc - 4 on
    one thread (the SMSP 2/3 warps), stores predicated off."""
    wa = ldg_f32x4(w_addr + Int64(col) * 16)
    wb = ldg_f32x4(w_addr + Int64(col + 1) * 16)
    wv = [wa[0], wb[0], wa[1], wb[1], wa[2], wb[2], wa[3], wb[3]]
    for half in cutlass.range_constexpr(2):
        krc = rc - 4 * half
        kw = []
        for j in cutlass.range_constexpr(3):
            kr = krc * 16 - 3 + j + 3 * b2i(krc == 0)
            kw.append(unpack_bf16x2(lds_u32(gbase + kr * (_PITCH * 4) + ((cq ^ (kr % 4)) * 16) + cr)))
        kbase = out_addr + (row0 + Int64(krc * 16)) * Int64(p_dim) * 2 + Int64(col) * 2
        ks = Float32(0.0)
        for i in cutlass.range_constexpr(16):
            kr = krc * 16 + i
            kx = lds_u32(gbase + kr * (_PITCH * 4) + ((cq ^ (kr % 4)) * 16) + cr)
            ky, ks = conv_row(
                (kw[0][0], kw[0][1], kw[1][0], kw[1][1], kw[2][0], kw[2][1]),
                kx, wv, ks,
            )  # fmt: skip
            stg_u32(kbase + Int64(i) * Int64(p_dim) * 2, st_on, ky)
            kw = [kw[1], kw[2], unpack_bf16x2(kx)]


@cute.jit
def epilogue_kda_conv_t(
    gemm_kernel,
    tidx: Int32,
    tCtAcc_base: cute.Tensor,
    tCgC_base: cute.Tensor,
    tCcC_base: cute.Tensor,
    epi_tile: cute.Tile,
    mma_tile_coord_mnl: tuple,
    m_rows: Int32,
    n_cols: Int32,
    conv: tuple,
    smem_x: Int32,
    norms: Constexpr,
    acc_consumer_state: pipeline.PipelineState,
    acc_pipeline: pipeline.PipelineAsync,
) -> pipeline.PipelineState:
    """Drain-and-release (8 warps, warp group g = M block g of the B-reuse
    tile), then the conv runs column-wise: per 32-column subtile the group's
    128 BF16 rows go through shared memory and thread (pair cp, chunk rc)
    sweeps rows 16 rc .. 16 rc + 15 of columns 2 cp, 2 cp + 1 with a sliding
    window, weights in registers and coalesced 4-byte stores."""
    out_addr, w_addr, side_addr, qsl_addr, norm_addr, nseq, top, nsteps = conv
    # Microbench skeletons (M9a ladder; production = 0): 1 = drain + release +
    # BF16 shared staging; 2 = + conv/SiLU math, no global stores; 3 = 2 with
    # the math only on the warps of SMSP 2/3 (twice the rows each); 4 =
    # release only (mainloop skeleton in the same CTA structure).
    skel = getattr(gemm_kernel, "conv_skel", 0)
    g = tidx // 128
    gt = tidx % 128
    if cutlass.const_expr(skel == 4):
        acc_pipeline.consumer_wait(acc_consumer_state)
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)
        acc_consumer_state.advance()
        return acc_consumer_state
    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
        gemm_kernel, gt, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
    )
    thr_copy_t2r = tiled_copy_t2r.get_slice(gt)
    tTR_cC = thr_copy_t2r.partition_D(
        cute.flat_divide(transform_partitioned_tensor_layout(tCcC_base), epi_tile)
    )[(None, None, None, None, None, *mma_tile_coord_mnl)]
    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

    epi_m = cute.size(tTR_tAcc.shape, mode=[3])
    epi_n = cute.size(tTR_tAcc.shape, mode=[4])
    ncol = cute.size(tTR_rAcc)
    if cutlass.const_expr(epi_m != 2 or ncol != 32):
        raise ValueError(f"conv epilogue: need 2 M blocks x 32 columns, got {epi_m}/{ncol}")
    p_dim = n_cols // 3
    n_blk = (m_rows + 127) // 128
    bar = 3 + g

    # This thread's TMEM row (drain) and its column-pass coordinates.
    trow = tTR_cC[(None, None, None, g, 0)][0][0]
    blk0 = trow - trow % 128
    cp = gt % 16
    rc = gt // 16
    r0 = blk0 + rc * 16
    s = find_seq(qsl_addr, nseq, top, nsteps, r0)
    start = ldg_s32(qsl_addr + Int64(s) * 4)
    end = ldg_s32(qsl_addr + Int64(s + 1) * 4)

    acc_pipeline.consumer_wait(acc_consumer_state)
    xs_all = []
    for ni in cutlass.range_constexpr(epi_n):
        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, g, ni)], tTR_rAcc)
        xs_all.append([pack_bf16x2(tTR_rAcc[2 * p], tTR_rAcc[2 * p + 1]) for p in range(16)])
    cute.arch.fence_view_async_tmem_load()
    with cute.arch.elect_one():
        acc_pipeline.consumer_release(acc_consumer_state)
    acc_consumer_state.advance()

    # Per-row sequence facts of rows r0 .. r0 + 15 (sequence boundaries inside
    # the chunk advance s).
    # Per-row flags of rows r0 .. r0 + 15: bit 0 = conv output stored here,
    # bit 1 = pre-conv row saved for the fix-up / conv state (sequence
    # boundaries inside the chunk advance s).
    flags = []
    for i in cutlass.range_constexpr(16):
        t = r0 + i
        while b2i(t >= end) * b2i(s + 1 < nseq) != 0:
            s = s + 1
            start = end
            end = ldg_s32(qsl_addr + Int64(s + 1) * 4)
        pos = t - start
        valid = b2i(t < m_rows) * b2i(t < end)
        fix = b2i(pos < 3)
        edge = fix | b2i(pos >= end - start - 3)
        if cutlass.const_expr(i < 3):
            fix = fix | b2i(rc == 0)
            edge = edge | b2i(rc == 0)
        if cutlass.const_expr(i >= 13):
            edge = edge | b2i(rc == 7)
        flags.append(valid * (1 - fix) + 2 * valid * edge)

    gbase = smem_x + g * (128 * _PITCH * 4)
    wrow = gbase + (trow % 128) * (_PITCH * 4)
    cq = cp // 4
    cr = (cp % 4) * 4
    ssq = [Float32(0.0) for _ in range(16)]
    st_on = Int32(1)
    if cutlass.const_expr(skel >= 2):
        st_on = b2i(nseq < 0)
    for ni in cutlass.range_constexpr(epi_n):
        xs = xs_all[ni]
        cute.arch.barrier(barrier_id=bar, number_of_threads=128)
        for q in cutlass.range_constexpr(4):
            sts_v4(wrow + (((trow % 4) ^ q) * 16), Int32(1), *xs[4 * q : 4 * q + 4])
        cute.arch.barrier(barrier_id=bar, number_of_threads=128)
        if cutlass.const_expr(skel == 1):
            continue
        if cutlass.const_expr(skel == 3):
            if gt >= 64:
                _skel3_math(
                    gbase, cq, cr, rc, out_addr, w_addr,
                    tTR_cC[(None, None, None, g, ni)][0][1] + 2 * cp,
                    Int64(r0), p_dim, st_on,
                )  # fmt: skip
            continue
        col0 = tTR_cC[(None, None, None, g, ni)][0][1]
        col = col0 + 2 * cp
        part = col0 // p_dim
        cc = col - part * p_dim
        wa = ldg_f32x4(w_addr + Int64(col) * 16)
        wb = ldg_f32x4(w_addr + Int64(col + 1) * 16)
        wv = [wa[0], wb[0], wa[1], wb[1], wa[2], wb[2], wa[3], wb[3]]
        # Window: rows r0 - 3 .. r0 - 1 (clamped into the block; those
        # outputs are block-start fix-up rows when rc == 0).
        hw = []
        for j in cutlass.range_constexpr(3):
            rr = rc * 16 - 3 + j
            rr = rr + 3 * b2i(rc == 0)
            hw.append(unpack_bf16x2(lds_u32(gbase + rr * (_PITCH * 4) + ((cq ^ (rr % 4)) * 16) + cr)))
        obase = out_addr + (
            (Int64(part) * Int64(m_rows) + Int64(r0)) * Int64(p_dim) + Int64(cc)
        ) * 2
        sbase = side_addr + Int64(col) * 2
        for i in cutlass.range_constexpr(16):
            t = r0 + i
            rr = rc * 16 + i
            x = lds_u32(gbase + rr * (_PITCH * 4) + ((cq ^ (rr % 4)) * 16) + cr)
            fl = flags[i]
            sq_in = ssq[i]
            if cutlass.const_expr(norms):
                sq_in = self32(b2i(col0 % 128 == 0), Float32(0.0), ssq[i])
            y, ssq[i] = conv_row(
                (hw[0][0], hw[0][1], hw[1][0], hw[1][1], hw[2][0], hw[2][1]),
                x, wv, sq_in,
            )  # fmt: skip
            stg_u32(obase + Int64(i) * Int64(p_dim) * 2, (fl % 2) * st_on, y)
            # Pre-conv rows for the fix-up / conv state (rare rows).
            if cutlass.const_expr(skel == 0):
                if fl >= 2:
                    es = find_seq(qsl_addr, nseq, top, nsteps, t)
                    e0 = ldg_s32(qsl_addr + Int64(es) * 4)
                    eln = ldg_s32(qsl_addr + Int64(es + 1) * 4) - e0
                    pos = t - e0
                    side_seq = Int64(n_blk) * 6 + Int64(es) * 6
                    tb = Int64(t // 128) * 6
                    rb = t % 128
                    stg_u32(sbase + (tb + Int64(rb)) * Int64(n_cols) * 2, b2i(rb < 3), x)
                    stg_u32(sbase + (tb + Int64(rb - 122)) * Int64(n_cols) * 2, b2i(rb >= 125), x)
                    stg_u32(sbase + (side_seq + Int64(pos)) * Int64(n_cols) * 2, b2i(pos < 3), x)
                    stg_u32(
                        sbase + (side_seq + Int64(pos - eln + 6)) * Int64(n_cols) * 2,
                        b2i(pos >= eln - 3),
                        x,
                    )
            hw = [hw[1], hw[2], unpack_bf16x2(x)]
        if cutlass.const_expr(norms):
            # Head end: reduce the 16 column-pair lanes, lane cp == 0 stores.
            if b2i((col0 + 32) % 128 == 0) * b2i(part < 2) != 0:
                h = p_dim // 128
                for i in cutlass.range_constexpr(16):
                    v = ssq[i]
                    v = v + shfl_xor_f32(v, 8)
                    v = v + shfl_xor_f32(v, 4)
                    v = v + shfl_xor_f32(v, 2)
                    v = v + shfl_xor_f32(v, 1)
                    t = r0 + i
                    ok = b2i(cp == 0) * (flags[i] % 2)
                    naddr = norm_addr + (
                        (Int64(part) * Int64(m_rows) + Int64(t)) * Int64(h)
                        + Int64(cc // 128)
                    ) * 4
                    stg_f32(naddr, ok, rsqrt_eps(v))
    return acc_consumer_state


# Branch-free conv + SiLU of one column pair (stock arithmetic): $0 bf16x2 out,
# $1 running max of the d = 1 + exp(-acc) bit patterns (d > 2^126 needs the
# exact div.full path: ``conv_row``); $2..$9 rows t-3..t as FP32 (lo, hi);
# $10..$17 FP32 weights (tap0 lo, tap0 hi, ..); $18 running max in;
# $19/$20 the constant pairs (-log2 e) and 1.0.
_CONV_FAST = """{
.reg .b32 i0, i1, im;
.reg .b64 X0, X1, X2, X3, W0, W1, W2, W3, A, E, R, NL, ONE;
.reg .f32 e0, e1, r0, r1;
mov.b64 X0, {$2, $3};
mov.b64 X1, {$4, $5};
mov.b64 X2, {$6, $7};
mov.b64 X3, {$8, $9};
mov.b64 W0, {$10, $11};
mov.b64 W1, {$12, $13};
mov.b64 W2, {$14, $15};
mov.b64 W3, {$16, $17};
mov.b64 NL, {$19, $19};
mov.b64 ONE, {$20, $20};
mov.b64 A, 0;
fma.rn.f32x2 A, X0, W0, A;
fma.rn.f32x2 A, X1, W1, A;
fma.rn.f32x2 A, X2, W2, A;
fma.rn.f32x2 A, X3, W3, A;
mul.rn.f32x2 E, A, NL;
mov.b64 {e0, e1}, E;
ex2.approx.ftz.f32 e0, e0;
ex2.approx.ftz.f32 e1, e1;
mov.b64 E, {e0, e1};
add.rn.f32x2 E, E, ONE;
mov.b64 {e0, e1}, E;
rcp.approx.ftz.f32 r0, e0;
rcp.approx.ftz.f32 r1, e1;
mov.b64 R, {r0, r1};
mul.rn.f32x2 R, A, R;
mov.b64 {r0, r1}, R;
cvt.rn.bf16x2.f32 $0, r1, r0;
mov.b32 i0, e0;
mov.b32 i1, e1;
max.s32 im, i0, i1;
max.s32 $1, im, $18;
}"""


@dsl_user_op
def conv_fast(h, w, gm, nl, one, *, loc=None, ip=None):
    """h: 8 FP32 taps (rows t-3..t, lo/hi); w: 8 FP32; returns (y, gm)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [_v(v, Float32, loc, ip) for v in h]
        + [_v(v, Float32, loc, ip) for v in w]
        + [_v(gm, Int32, loc, ip), _v(nl, Float32, loc, ip), _v(one, Float32, loc, ip)],
        _CONV_FAST,
        "=r,=r," + ",".join(["f"] * 16 + ["r", "f", "f"]),
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM,
    )
    y = Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip))
    g = Int32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip))
    return y, g


@dsl_user_op
def stg_u32_all(addr: Int64, v: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [_v(addr, Int64, loc, ip), _v(v, Uint32, loc, ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM,
    )


# conv_fast with packed operands: $0 bf16x2 out, $1 running max of the d bit
# patterns; $2..$5 rows t-3..t (f32x2), $6..$9 taps 0..3 (f32x2), $10 running
# max in, $11 (-log2 e) pair, $12 1.0 pair.
_CONV_FAST2 = """{
.reg .b32 i0, i1, im;
.reg .b64 A, E, R;
.reg .f32 e0, e1, r0, r1;
mov.b64 A, 0;
fma.rn.f32x2 A, $2, $6, A;
fma.rn.f32x2 A, $3, $7, A;
fma.rn.f32x2 A, $4, $8, A;
fma.rn.f32x2 A, $5, $9, A;
mul.rn.f32x2 E, A, $11;
mov.b64 {e0, e1}, E;
ex2.approx.ftz.f32 e0, e0;
ex2.approx.ftz.f32 e1, e1;
mov.b64 E, {e0, e1};
add.rn.f32x2 E, E, $12;
mov.b64 {e0, e1}, E;
rcp.approx.ftz.f32 r0, e0;
rcp.approx.ftz.f32 r1, e1;
mov.b64 R, {r0, r1};
mul.rn.f32x2 R, A, R;
mov.b64 {r0, r1}, R;
cvt.rn.bf16x2.f32 $0, r1, r0;
mov.b32 i0, e0;
mov.b32 i1, e1;
max.s32 im, i0, i1;
max.s32 $1, im, $10;
}"""
# Same arithmetic with scalar add / mul after the MUFU ops (no pair repacking).
_CONV_FAST3 = """{
.reg .b32 i0, i1;
.reg .b64 A, E;
.reg .f32 a0, a1, e0, e1, d0, d1, r0, r1;
mov.b64 A, 0;
fma.rn.f32x2 A, $2, $6, A;
fma.rn.f32x2 A, $3, $7, A;
fma.rn.f32x2 A, $4, $8, A;
fma.rn.f32x2 A, $5, $9, A;
mul.rn.f32x2 E, A, $11;
mov.b64 {e0, e1}, E;
mov.b64 {a0, a1}, A;
ex2.approx.ftz.f32 e0, e0;
ex2.approx.ftz.f32 e1, e1;
add.f32 d0, e0, 0f3F800000;
add.f32 d1, e1, 0f3F800000;
rcp.approx.ftz.f32 r0, d0;
rcp.approx.ftz.f32 r1, d1;
mul.rn.f32 r0, a0, r0;
mul.rn.f32 r1, a1, r1;
cvt.rn.bf16x2.f32 $0, r1, r0;
mov.b32 i0, d0;
mov.b32 i1, d1;
max.s32 i0, i0, i1;
max.s32 $1, i0, $10;
}"""


@dsl_user_op
def conv_fast2(x3, x2, x1, x0, w, gm, nl, one, scalar=False, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [_v(v, Int64, loc, ip) for v in (x3, x2, x1, x0)]
        + [_v(v, Int64, loc, ip) for v in w]
        + [_v(gm, Int32, loc, ip), _v(nl, Int64, loc, ip), _v(one, Int64, loc, ip)],
        _CONV_FAST3 if scalar else _CONV_FAST2,
        "=r,=r," + ",".join(["l"] * 8 + ["r", "l", "l"]),
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM,
    )
    y = Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip))
    g = Int32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip))
    return y, g


@dsl_user_op
def unpack_b64(x: Uint32, *, loc=None, ip=None) -> Int64:
    """bf16x2 -> f32x2 (lo, hi) in one 64-bit register."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [_v(x, Uint32, loc, ip)],
            "{ .reg .b32 l, h; shl.b32 l, $1, 16; and.b32 h, $1, -65536; "
            "mov.b64 $0, {l, h}; }",
            "=l,r",
            has_side_effects=False,
            loc=loc,
            ip=ip,
            **_ASM,
        )
    )


@dsl_user_op
def pack_f32x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [_v(a, Float32, loc, ip), _v(b, Float32, loc, ip)],
            "mov.b64 $0, {$1, $2};",
            "=l,f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
            **_ASM,
        )
    )


@dsl_user_op
def setmaxnreg(inc: bool, n: int, *, loc=None, ip=None) -> None:
    """Warpgroup register reconfiguration (all four warps of the warpgroup)."""
    llvm.inline_asm(
        None,
        [],
        f"setmaxnreg.{'inc' if inc else 'dec'}.sync.aligned.u32 {n};",
        "",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM,
    )


@dsl_user_op
def ldg_b64x2(addr: Int64, *, loc=None, ip=None):
    """Two aligned f32x2 pairs (16 B, read-only path)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i64(), T.i64()]),
        [_v(addr, Int64, loc, ip)],
        "ld.global.nc.v2.b64 {$0, $1}, [$2];",
        "=l,=l,l",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM,
    )
    return (
        Int64(llvm.extractvalue(T.i64(), res, [0], loc=loc, ip=ip)),
        Int64(llvm.extractvalue(T.i64(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def split_b64(v: Int64, *, loc=None, ip=None):
    """f32x2 pair -> (lo, hi)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [_v(v, Int64, loc, ip)],
        "mov.b64 {$0, $1}, $2;",
        "=f,=f,l",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


#: (-log2 e, -log2 e) and (1.0, 1.0) as f32x2 bit patterns (signed int64).
_NL2 = 0xBFB8AA3BBFB8AA3B - (1 << 64)
_ONE2 = 0x3F8000003F800000


#: d bit pattern above which ``div.full`` scales (2^126): the exact path.
_D_EXACT = 0x7E800000


@cute.jit
def _t2_chunk(
    rc: Int32,
    r0: Int32,
    gbase: Int32,
    cq: Int32,
    cr: Int32,
    w0: Int64,
    w1: Int64,
    w2: Int64,
    w3: Int64,
    nl2: Int64,
    one2: Int64,
    out_addr: Int64,
    part: Int32,
    m_rows: Int32,
    cc: Int32,
    row_end: Int32,
    p_dim: cutlass.Constexpr,
    scalar: cutlass.Constexpr,
    probe: cutlass.Constexpr = 0,
):
    """Conv + SiLU of rows r0 .. r0 + 15 (chunk rc of the staged subtile) for
    one column pair: loads, 16 branch-free chains, stores; the exact path
    when some d > 2^126 or the chunk crosses ``row_end``."""
    wp = [w0, w1, w2, w3]
    # Row rc * 16 + i: lb[i % 4] + 64 i (rc * 16 % 4 == 0, XOR swizzle by row).
    lb = [gbase + rc * (16 * _PITCH * 4) + ((cq ^ j) * 16) + cr for j in range(4)]
    hr = []
    for j in cutlass.range_constexpr(3):
        rr = rc * 16 - 3 + j
        rr = rr + 3 * b2i(rc == 0)
        hr.append(lds_u32(gbase + rr * (_PITCH * 4) + ((cq ^ (rr % 4)) * 16) + cr))
    xr = [lds_u32(lb[i % 4] + i * (_PITCH * 4)) for i in range(16)]
    if cutlass.const_expr(probe == 1):
        # Microbench probe: no shared-memory reads (register stand-ins).
        hr = [Uint32(cr * (j + 3) + rc) for j in range(3)]
        xr = [Uint32(cr * (i + 7) + rc) for i in range(16)]
    hp = [unpack_b64(v) for v in hr]
    ys = []
    gm = Int32(0)
    for i in cutlass.range_constexpr(16):
        xp = unpack_b64(xr[i])
        y, gm = conv_fast2(hp[0], hp[1], hp[2], xp, wp, gm, nl2, one2, scalar)
        ys.append(y)
        hp = [hp[1], hp[2], xp]
    obase = out_addr + (
        (Int64(part) * Int64(m_rows) + Int64(r0)) * Int64(p_dim) + Int64(cc)
    ) * 2
    if cutlass.const_expr(probe == 2):
        # Microbench probe: stores issued but predicated off (no memory).
        for fi in cutlass.range_constexpr(16):
            stg_u32(obase + fi * p_dim * 2, b2i(r0 < 0), ys[fi])
    elif b2i(gm <= _D_EXACT) * b2i(r0 + 16 <= row_end) != 0:
        for fi in cutlass.range_constexpr(16):
            stg_u32_all(obase + fi * p_dim * 2, ys[fi])
    else:
        wv = []
        for wj in wp:
            wv.extend(split_b64(wj))
        hx = [unpack_bf16x2(v) for v in hr]
        for xi in cutlass.range_constexpr(16):
            y2, s2 = conv_row(
                (hx[0][0], hx[0][1], hx[1][0], hx[1][1], hx[2][0], hx[2][1]),
                xr[xi], wv, Float32(0.0),
            )  # fmt: skip
            stg_u32(obase + Int64(xi) * Int64(p_dim) * 2, b2i(r0 + xi < row_end), y2)
            hx = [hx[1], hx[2], unpack_bf16x2(xr[xi])]


@cute.jit
def _t2_chunk_loop(
    rc: Int32,
    r0: Int32,
    gbase: Int32,
    cq: Int32,
    cr: Int32,
    w0: Int64,
    w1: Int64,
    w2: Int64,
    w3: Int64,
    nl2: Int64,
    one2: Int64,
    out_addr: Int64,
    part: Int32,
    m_rows: Int32,
    cc: Int32,
    row_end: Int32,
    p_dim: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
):
    """``_t2_chunk`` as a rolled row loop (small hot code: the SM-wide fast
    I-cache holds ~8-16 KiB). Rows are stored as computed; a chunk with some
    d > 2^126 is recomputed on the exact path and overwritten."""
    wp = [w0, w1, w2, w3]
    rbase = gbase + rc * (16 * _PITCH * 4) + cr
    rr0 = rc * 16 - 3 + 3 * b2i(rc == 0)
    h0 = unpack_b64(lds_u32(gbase + rr0 * (_PITCH * 4) + ((cq ^ (rr0 % 4)) * 16) + cr))
    rr1 = rr0 + 1
    h1 = unpack_b64(lds_u32(gbase + rr1 * (_PITCH * 4) + ((cq ^ (rr1 % 4)) * 16) + cr))
    rr2 = rr0 + 2
    h2 = unpack_b64(lds_u32(gbase + rr2 * (_PITCH * 4) + ((cq ^ (rr2 % 4)) * 16) + cr))
    obase = out_addr + (
        (Int64(part) * Int64(m_rows) + Int64(r0)) * Int64(p_dim) + Int64(cc)
    ) * 2
    gm = Int32(0)
    for i in cutlass.range(16, unroll=unroll):
        x = lds_u32(rbase + i * (_PITCH * 4) + ((cq ^ (i % 4)) * 16))
        xp = unpack_b64(x)
        y, gm = conv_fast2(h0, h1, h2, xp, wp, gm, nl2, one2, False)
        stg_u32(obase + Int64(i) * (p_dim * 2), b2i(r0 + i < row_end), y)
        h0 = h1
        h1 = h2
        h2 = xp
    if gm > _D_EXACT:
        wv = []
        for wj in wp:
            wv.extend(split_b64(wj))
        hx = []
        for j in cutlass.range_constexpr(3):
            rr = rr0 + j
            hx.append(unpack_bf16x2(lds_u32(gbase + rr * (_PITCH * 4) + ((cq ^ (rr % 4)) * 16) + cr)))
        for xi in cutlass.range_constexpr(16):
            xv = lds_u32(rbase + xi * (_PITCH * 4) + ((cq ^ (xi % 4)) * 16))
            y2, s2 = conv_row(
                (hx[0][0], hx[0][1], hx[1][0], hx[1][1], hx[2][0], hx[2][1]),
                xv, wv, Float32(0.0),
            )  # fmt: skip
            stg_u32(obase + Int64(xi) * Int64(p_dim) * 2, b2i(r0 + xi < row_end), y2)
            hx = [hx[1], hx[2], unpack_bf16x2(xv)]


@cute.jit
def epilogue_kda_conv_t2(
    gemm_kernel,
    tidx: Int32,
    tCtAcc_base: cute.Tensor,
    tCgC_base: cute.Tensor,
    tCcC_base: cute.Tensor,
    epi_tile: cute.Tile,
    mma_tile_coord_mnl: tuple,
    m_rows: Int32,
    n_cols: Int32,
    conv: tuple,
    smem_x: Int32,
    acc_consumer_state: pipeline.PipelineState,
    acc_pipeline: pipeline.PipelineAsync,
    dgs0_st=None,
) -> pipeline.PipelineState:
    """``epilogue_kda_conv_t`` without inverse norms, branch-free per 16-row
    chunk: the drain thread saves its pre-conv row when it is a block or
    sequence edge row; the column pass loads all 19 rows, runs the 16 conv +
    SiLU chains, and stores them unpredicated unless the chunk crosses the
    end of the rows or some d > 2^126 (then the exact per-row path). Rows
    0-2 of blocks and sequences are stored too and rewritten by the fix-up.

    With ``ckpt_addr`` (int32 checkpoint offset per sequence, 0 = none) the
    drain thread also saves the three pre-conv rows before each checkpoint
    (offset >= 3) to ``halo_addr`` rows ``3 s + j``: the window the prefix-cache
    checkpoint exporter stores as that checkpoint's conv state."""
    out_addr, w_addr, side_addr, qsl_addr, norm_addr, nseq, top, nsteps = conv[:8]
    ckpt_addr, halo_addr = conv[8], conv[9]
    g = tidx // 128
    gt = tidx % 128
    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
        gemm_kernel, gt, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
    )
    thr_copy_t2r = tiled_copy_t2r.get_slice(gt)
    tTR_cC = thr_copy_t2r.partition_D(
        cute.flat_divide(transform_partitioned_tensor_layout(tCcC_base), epi_tile)
    )[(None, None, None, None, None, *mma_tile_coord_mnl)]
    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

    epi_m = cute.size(tTR_tAcc.shape, mode=[3])
    epi_n = cute.size(tTR_tAcc.shape, mode=[4])
    ncol = cute.size(tTR_rAcc)
    if cutlass.const_expr(epi_m != 2 or ncol != 32):
        raise ValueError(f"conv epilogue: need 2 M blocks x 32 columns, got {epi_m}/{ncol}")
    # Compile-time plane width: immediate store offsets.
    p_dim = getattr(gemm_kernel, "conv_p_dim", None) or n_cols // 3
    n_blk = (m_rows + 127) // 128
    bar = 3 + g

    trow = tTR_cC[(None, None, None, g, 0)][0][0]
    blk0 = trow - trow % 128
    cp = gt % 16
    rc = gt // 16
    r0 = blk0 + rc * 16
    total_end = ldg_s32(qsl_addr + Int64(nseq) * 4)
    row_end = total_end - (total_end - m_rows) * b2i(m_rows < total_end)

    # This drain row's sequence (side rows for the fix-up / conv state).
    s_t = find_seq(qsl_addr, nseq, top, nsteps, trow)
    st0 = ldg_s32(qsl_addr + Int64(s_t) * 4)
    sln = ldg_s32(qsl_addr + Int64(s_t + 1) * 4) - st0
    pos = trow - st0
    rb = trow % 128
    # Checkpoint halo row (branch-free: without checkpoints the offset load
    # reads qsl and the predicate is 0).
    ck_on = b2i(ckpt_addr != Int64(0))
    ck = ldg_s32(qsl_addr + Int64(ck_on) * (ckpt_addr - qsl_addr) + Int64(s_t) * 4)
    hj = pos - ck + 3
    pk = ck_on * b2i(ck >= 3) * b2i(hj >= 0) * b2i(hj < 3)

    acc_pipeline.consumer_wait(acc_consumer_state)
    if cutlass.const_expr(dgs0_st is not None):
        dgs0.st64lo(dgs0_st[0], dgs0.clk(), dgs0_st[1])
    xs_all = []
    for ni in cutlass.range_constexpr(epi_n):
        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, g, ni)], tTR_rAcc)
        xs_all.append([pack_bf16x2(tTR_rAcc[2 * p], tTR_rAcc[2 * p + 1]) for p in range(16)])
    cute.arch.fence_view_async_tmem_load()
    with cute.arch.elect_one():
        acc_pipeline.consumer_release(acc_consumer_state)
    acc_consumer_state.advance()
    if cutlass.const_expr(dgs0_st is not None):
        dgs0.st64lo(dgs0_st[0] + 4, dgs0.clk(), dgs0_st[1])

    pb0 = b2i(rb < 3)
    pb1 = b2i(rb >= 125)
    ps0 = b2i(pos < 3)
    ps1 = b2i(pos >= sln - 3)
    if b2i(trow < row_end) * (pb0 | pb1 | ps0 | ps1 | pk) != 0:
        tb = Int64(trow // 128) * 6
        side_seq = Int64(n_blk) * 6 + Int64(s_t) * 6
        dst = (
            side_addr + (tb + Int64(rb)) * Int64(n_cols) * 2,
            side_addr + (tb + Int64(rb - 122)) * Int64(n_cols) * 2,
            side_addr + (side_seq + Int64(pos)) * Int64(n_cols) * 2,
            side_addr + (side_seq + Int64(pos - sln + 6)) * Int64(n_cols) * 2,
            halo_addr + (Int64(s_t) * 3 + Int64(hj)) * Int64(n_cols) * 2,
        )
        prd = (pb0, pb1, ps0, ps1, pk)
        for sni in cutlass.range_constexpr(epi_n):
            sxs = xs_all[sni]
            sc0 = tTR_cC[(None, None, None, g, sni)][0][1]
            for sd in cutlass.range_constexpr(5):
                sbase = dst[sd] + Int64(sc0) * 2
                for sq in cutlass.range_constexpr(4):
                    stg_v4(sbase + sq * 16, prd[sd], *sxs[4 * sq : 4 * sq + 4])

    gbase = smem_x + g * (128 * _PITCH * 4)
    wrow = gbase + (trow % 128) * (_PITCH * 4)
    cq = cp // 4
    cr = (cp % 4) * 4
    nl2 = Int64(_NL2)
    one2 = Int64(_ONE2)
    # Conv weights in column-pair layout (KdaConvGemm.conv_w_pairs): 32 B per
    # column pair = taps 0..3 as (col, col + 1) f32x2 pairs.
    col_n = tTR_cC[(None, None, None, g, 0)][0][1] + 2 * cp
    wl = ldg_b64x2(w_addr + Int64(col_n) * 16)
    wh = ldg_b64x2(w_addr + Int64(col_n) * 16 + 16)
    for ni in cutlass.range_constexpr(epi_n):
        xs = xs_all[ni]
        wp = [wl[0], wl[1], wh[0], wh[1]]
        col0 = tTR_cC[(None, None, None, g, ni)][0][1]
        if cutlass.const_expr(ni + 1 < epi_n):
            col_n = tTR_cC[(None, None, None, g, ni + 1)][0][1] + 2 * cp
            wl = ldg_b64x2(w_addr + Int64(col_n) * 16)
            wh = ldg_b64x2(w_addr + Int64(col_n) * 16 + 16)
        cute.arch.barrier(barrier_id=bar, number_of_threads=128)
        for q in cutlass.range_constexpr(4):
            sts_v4(wrow + (((trow % 4) ^ q) * 16), Int32(1), *xs[4 * q : 4 * q + 4])
        cute.arch.barrier(barrier_id=bar, number_of_threads=128)
        col = col0 + 2 * cp
        part = col0 // p_dim
        cc = col - part * p_dim
        args = (gbase, cq, cr, wp[0], wp[1], wp[2], wp[3], nl2, one2, out_addr,
                part, m_rows, cc, row_end, p_dim,
                getattr(gemm_kernel, "conv_scalar_silu", False),
                getattr(gemm_kernel, "conv_probe", 0))  # fmt: skip
        if cutlass.const_expr(getattr(gemm_kernel, "conv_loop_unroll", 0) > 0):
            _t2_chunk_loop(rc, r0, *args[:15], gemm_kernel.conv_loop_unroll)
        elif cutlass.const_expr(getattr(gemm_kernel, "conv_math_smsp23", False)):
            # Microbench: column pass only on the SMSP 2/3 warps (two chunks).
            if gt >= 64:
                _t2_chunk(rc, r0, *args)
                _t2_chunk(rc - 4, r0 - 64, *args)
        else:
            _t2_chunk(rc, r0, *args)
    if cutlass.const_expr(dgs0_st is not None):
        dgs0.st64lo(dgs0_st[0] + 8, dgs0.clk(), dgs0_st[1])
    return acc_consumer_state
