# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gated output RMSNorm + MXFP8 quantization of one staged BF16 O tile inside
the persistent KDA kernel's epilogue (the o_proj input, written directly).

Per element the arithmetic of ``o_norm_mx_kernel`` (ops/flashkda_split
smxx/o_norm_mx.cu, itself the expressions of the stock Triton gated norm):
sum of squares of the BF16-rounded output in 16-element parts summed in
element order, parts combined by the same butterfly; ``rstd`` with div.full /
sqrt.approx; sigmoid gate with ex2.approx.ftz + rcp.approx.ftz (div.full when
1 + 2^t exceeds 2^126); ``((x * rstd) * w) * sigmoid`` rounded to BF16; UE8M0
block scale of the 32-element block maximum; E4M3 satfinite. Each thread owns
one token row and one 32-column MX block of the head (4 threads per token).
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

_ASM = dict(is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


def _v(x, cls, loc, ip):
    return cls(x).ir_value(loc=loc, ip=ip)


def _struct(n, ty):
    return llvm.StructType.get_literal([ty] * n)


@dsl_user_op
def ldg_nc_v4(addr: Int64, *, loc=None, ip=None):
    """Four 32-bit words from global memory (read-only path, no L1 allocate)."""
    res = llvm.inline_asm(
        _struct(4, T.i32()),
        [_v(addr, Int64, loc, ip)],
        "ld.global.nc.L1::no_allocate.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def lds_v4(saddr: Int32, *, loc=None, ip=None):
    """Four 32-bit words from shared memory (32-bit shared-window address)."""
    res = llvm.inline_asm(
        _struct(4, T.i32()),
        [_v(saddr, Int32, loc, ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
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
def lds_f32x4(saddr: Int32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        _struct(4, T.f32()),
        [_v(saddr, Int32, loc, ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM,
    )
    return [
        Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def sts_f32(saddr: Int32, val: Float32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [_v(saddr, Int32, loc, ip), _v(val, Float32, loc, ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM,
    )


@dsl_user_op
def ldg_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_v(addr, Int64, loc, ip)],
            "ld.global.nc.f32 $0, [$1];",
            "=f,l",
            has_side_effects=True,
            loc=loc,
            ip=ip,
            **_ASM,
        )
    )


@dsl_user_op
def stg_v4(addr: Int64, w0, w1, w2, w3, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [_v(addr, Int64, loc, ip)] + [_v(w, Uint32, loc, ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM,
    )


@dsl_user_op
def stg_u8(addr: Int64, val: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [_v(addr, Int64, loc, ip), _v(val, Uint32, loc, ip)],
        "{\n.reg .b16 b;\ncvt.u16.u32 b, $1;\nst.global.u8 [$0], b;\n}",
        "l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM,
    )


@dsl_user_op
def prefetch_l2(addr: Int64, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [_v(addr, Int64, loc, ip)],
        "prefetch.global.L2 [$0];",
        "l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM,
    )


# Sequential sum of squares of 16 BF16 values (8 words, element order), as
# o_norm_mx's ``ss = (ss + x0^2) + x1^2 ...``: BF16 squares are exact in FP32, so
# the fused form rounds once per addition like the separate multiply + add.
_SUMSQ16 = "{\n.reg .f32 l<8>, h<8>, s;\n" + "".join(
    f"shl.b32 l{i}, ${i + 1}, 16;\nand.b32 h{i}, ${i + 1}, -65536;\n" for i in range(8)
) + "mul.rn.ftz.f32 s, l0, l0;\nfma.rn.ftz.f32 s, h0, h0, s;\n" + "".join(
    f"fma.rn.ftz.f32 s, l{i}, l{i}, s;\nfma.rn.ftz.f32 s, h{i}, h{i}, s;\n"
    for i in range(1, 8)
) + "mov.f32 $0, s;\n}"


@dsl_user_op
def sumsq16(words, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_v(w, Uint32, loc, ip) for w in words],
            _SUMSQ16,
            "=f," + ",".join(["r"] * 8),
            has_side_effects=False,
            loc=loc,
            ip=ip,
            **_ASM,
        )
    )


@dsl_user_op
def rstd_of(ss: Float32, eps: Float32, *, loc=None, ip=None) -> Float32:
    """div.full(1, sqrt.approx(div.full(ss, 128) + eps)) (norm_mx.cuh order)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_v(ss, Float32, loc, ip), _v(eps, Float32, loc, ip)],
            "{\n.reg .f32 a, b;\ndiv.full.f32 a, $1, 0f43000000;\n"
            "add.rn.ftz.f32 a, a, $2;\nsqrt.approx.f32 b, a;\n"
            "div.full.f32 $0, 0f3F800000, b;\n}",
            "=f,f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
            **_ASM,
        )
    )


@dsl_user_op
def gate_fast_ok(words, *, loc=None, ip=None) -> Int32:
    """1 when every BF16 gate value is >= -86 (then 1 + 2^(-g log2 e) < 2^126
    for every element and rcp.approx equals div.full)."""
    body = "{\n.reg .b32 m, lo, hi;\n.reg .f32 fl, fh, fm;\n.reg .pred p;\n"
    body += "min.bf16x2 m, $1, $2;\n"
    body += "".join(f"min.bf16x2 m, m, ${i};\n" for i in range(3, 17))
    body += (
        "shl.b32 lo, m, 16;\nand.b32 hi, m, -65536;\nmov.b32 fl, lo;\nmov.b32 fh, hi;\n"
        "min.f32 fm, fl, fh;\nsetp.ge.f32 p, fm, 0fC2AC0000;\nselp.u32 $0, 1, 0, p;\n}"
    )
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [_v(w, Uint32, loc, ip) for w in words],
            body,
            "=r," + ",".join(["r"] * 16),
            has_side_effects=False,
            loc=loc,
            ip=ip,
            **_ASM,
        )
    )


_PAIR_HEAD = """{
.reg .b32 xl, xh, gl, gh, yp, ya;
.reg .b64 X, G, E, Dd, V, W, S, R, CN, C1;
.reg .f32 t0, t1, e0, e1, d0, d1, r0, r1, v0, v1, dm, cn, c1;
.reg .pred p;
mov.f32 cn, 0fBFB8AA3B;
mov.f32 c1, 0f3F800000;
mov.b64 CN, {cn, cn};
mov.b64 C1, {c1, c1};
shl.b32 xl, $2, 16;
and.b32 xh, $2, -65536;
shl.b32 gl, $3, 16;
and.b32 gh, $3, -65536;
mov.b64 G, {gl, gh};
mul.rn.ftz.f32x2 G, G, CN;
mov.b64 {t0, t1}, G;
ex2.approx.ftz.f32 e0, t0;
ex2.approx.ftz.f32 e1, t1;
mov.b64 E, {e0, e1};
add.rn.ftz.f32x2 Dd, C1, E;
mov.b64 {d0, d1}, Dd;
rcp.approx.ftz.f32 r0, d0;
rcp.approx.ftz.f32 r1, d1;
"""
_PAIR_SLOW = """max.f32 dm, d0, d1;
setp.gt.f32 p, dm, 0f7E800000;
@p div.full.f32 r0, 0f3F800000, d0;
@p div.full.f32 r1, 0f3F800000, d1;
"""
_PAIR_TAIL = """mov.b64 X, {xl, xh};
mov.b64 R, {$5, $5};
mul.rn.ftz.f32x2 V, X, R;
mov.b64 W, {$6, $7};
mul.rn.ftz.f32x2 V, V, W;
mov.b64 S, {r0, r1};
mul.rn.ftz.f32x2 V, V, S;
mov.b64 {v0, v1}, V;
cvt.rn.bf16x2.f32 yp, v1, v0;
and.b32 ya, yp, 2147450879;
max.bf16x2 $1, $4, ya;
mov.b32 $0, yp;
}"""


@dsl_user_op
def onorm_pair(x, g, amax, rstd, w0, w1, slow: bool = False, *, loc=None, ip=None):
    """One element pair: returns (bf16x2 of ((x rstd) w) sigmoid(g), running
    max of |y| as bf16x2)."""
    res = llvm.inline_asm(
        _struct(2, T.i32()),
        [
            _v(x, Uint32, loc, ip),
            _v(g, Uint32, loc, ip),
            _v(amax, Uint32, loc, ip),
            _v(rstd, Float32, loc, ip),
            _v(w0, Float32, loc, ip),
            _v(w1, Float32, loc, ip),
        ],
        _PAIR_HEAD + (_PAIR_SLOW if slow else "") + _PAIR_TAIL,
        "=r,=r,r,r,r,f,f,f",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def ue8m0_of(amax2: Uint32, *, loc=None, ip=None):
    """(e, inv) of norm_mx::ue8m0 for the block maximum (max of both halves)."""
    res = llvm.inline_asm(
        _struct(2, T.i32()),
        [_v(amax2, Uint32, loc, ip)],
        """{
.reg .b32 lo, hi, bits, ex, mant, bump, e, ie;
.reg .f32 fl, fh, am, nm, inv;
.reg .pred pz, pm, pn, pe;
shl.b32 lo, $2, 16;
and.b32 hi, $2, -65536;
mov.b32 fl, lo;
mov.b32 fh, hi;
max.f32 am, fl, fh;
mul.rn.ftz.f32 nm, am, 0f3B124925;
mov.b32 bits, nm;
bfe.u32 ex, bits, 23, 8;
and.b32 mant, bits, 8388607;
setp.ne.u32 pm, mant, 0;
selp.u32 bump, 1, 0, pm;
setp.eq.u32 pz, ex, 0;
setp.le.u32 pn, mant, 4194304;
and.pred pz, pz, pn;
@pz mov.u32 bump, 0;
add.u32 e, ex, bump;
min.u32 e, e, 254;
setp.le.f32 pn, nm, 0f00000000;
@pn mov.u32 e, 0;
sub.u32 ie, 254, e;
shl.b32 ie, ie, 23;
setp.eq.u32 pe, e, 0;
@pe mov.u32 ie, 0;
mov.b32 $0, e;
mov.b32 $1, ie;
}""",
        "=r,=r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Float32(
            llvm.bitcast(
                T.f32(), llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip),
                loc=loc, ip=ip,
            )
        ),
    )


@dsl_user_op
def quant4(y0: Uint32, y1: Uint32, inv: Float32, *, loc=None, ip=None) -> Uint32:
    """Four BF16 values (two words) times the block's 2^-e as E4M3 bytes."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [_v(y0, Uint32, loc, ip), _v(y1, Uint32, loc, ip), _v(inv, Float32, loc, ip)],
            """{
.reg .b32 a, b, c, d;
.reg .b64 A, B, I;
.reg .f32 a0, a1, b0, b1;
.reg .b16 qa, qb;
shl.b32 a, $1, 16;
and.b32 b, $1, -65536;
shl.b32 c, $2, 16;
and.b32 d, $2, -65536;
mov.b64 I, {$3, $3};
mov.b64 A, {a, b};
mov.b64 B, {c, d};
mul.rn.ftz.f32x2 A, A, I;
mul.rn.ftz.f32x2 B, B, I;
mov.b64 {a0, a1}, A;
mov.b64 {b0, b1}, B;
cvt.rn.satfinite.e4m3x2.f32 qa, a1, a0;
cvt.rn.satfinite.e4m3x2.f32 qb, b1, b0;
mov.b32 $0, {qa, qb};
}""",
            "=r,r,r,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
            **_ASM,
        )
    )


@dsl_user_op
def ss_bfly(sa: Float32, sb: Float32, *, loc=None, ip=None) -> Float32:
    """Row sum of squares: (s_2q + s_2q+1) then the xor-1 / xor-2 butterfly
    over the row's 4 lanes (o_norm_mx's 8-part butterfly, same pairing)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [_v(sa, Float32, loc, ip), _v(sb, Float32, loc, ip)],
            "{\n.reg .f32 s, t;\nadd.rn.ftz.f32 s, $1, $2;\n"
            "shfl.sync.bfly.b32 t, s, 1, 31, -1;\nadd.rn.ftz.f32 s, s, t;\n"
            "shfl.sync.bfly.b32 t, s, 2, 31, -1;\nadd.rn.ftz.f32 $0, s, t;\n}",
            "=f,f,f",
            has_side_effects=True,
            loc=loc,
            ip=ip,
            **_ASM,
        )
    )


@cute.jit
def onorm_tile(
    so_tile: Int32,
    s_w: Int32,
    etx: Int32,
    tok0: Int32,
    nvalid: Int32,
    hidx: Int32,
    og_addr: Int64,
    og_ld: Int64,
    oq_addr: Int64,
    oq_ld: Int64,
    osf_addr: Int64,
    osf_cols: Int32,
    eps: Float32,
    pf_tok0: Int32,
    pf_n: Int32,
):
    """Normalize + quantize one staged [32 tok, 128] BF16 tile (SW128 halves at
    so_tile, +4 KiB) of head ``hidx``; rows >= nvalid are computed but not
    stored. Prefetches the next tile's gate rows (pf_n > 0) into L2."""
    r = etx >> 2
    qq = etx & 3
    rr = cutlass.min(r, nvalid - 1)
    tok = tok0 + rr
    gaddr = og_addr + Int64(tok) * og_ld + Int64(hidx * 256 + qq * 64)
    gw = (
        ldg_nc_v4(gaddr)
        + ldg_nc_v4(gaddr + 16)
        + ldg_nc_v4(gaddr + 32)
        + ldg_nc_v4(gaddr + 48)
    )
    base = so_tile + (qq >> 1) * 4096 + r * 128
    sw = r & 7
    xw = []
    for j in cutlass.range_constexpr(4):
        xw = xw + lds_v4(base + ((((qq & 1) * 4 + j) ^ sw) << 4))
    rstd = rstd_of(ss_bfly(sumsq16(xw[0:8]), sumsq16(xw[8:16])), eps)
    yv = cute.make_rmem_tensor((16,), cutlass.Uint32)
    am = cute.make_rmem_tensor((1,), cutlass.Uint32)
    am[0] = Uint32(0)
    wbase = s_w + qq * 128
    if gate_fast_ok(gw) != 0:
        for k in cutlass.range_constexpr(8):
            wv = lds_f32x4(wbase + 16 * k)
            for u in cutlass.range_constexpr(2):
                y, a = onorm_pair(
                    xw[2 * k + u], gw[2 * k + u], am[0], rstd, wv[2 * u], wv[2 * u + 1]
                )
                yv[2 * k + u] = y
                am[0] = a
    else:
        for k in cutlass.range_constexpr(8):
            wv = lds_f32x4(wbase + 16 * k)
            for u in cutlass.range_constexpr(2):
                y, a = onorm_pair(
                    xw[2 * k + u],
                    gw[2 * k + u],
                    am[0],
                    rstd,
                    wv[2 * u],
                    wv[2 * u + 1],
                    slow=True,
                )
                yv[2 * k + u] = y
                am[0] = a
    e, inv = ue8m0_of(am[0])
    q = []
    for k in cutlass.range_constexpr(8):
        q = q + [quant4(yv[2 * k], yv[2 * k + 1], inv)]
    if r < nvalid:
        qaddr = oq_addr + Int64(tok) * oq_ld + Int64(hidx * 128 + qq * 32)
        stg_v4(qaddr, q[0], q[1], q[2], q[3])
        stg_v4(qaddr + 16, q[4], q[5], q[6], q[7])
        soff = hidx * 512 + (tok & 31) * 16 + ((tok & 127) >> 5) * 4 + qq
        stg_u8(osf_addr + Int64(tok >> 7) * Int64(osf_cols * 128) + Int64(soff), e)
    if pf_n > 0:
        if qq < 2:
            prow = pf_tok0 + cutlass.min(r, pf_n - 1)
            prefetch_l2(og_addr + Int64(prow) * og_ld + Int64(hidx * 256 + qq * 128))
