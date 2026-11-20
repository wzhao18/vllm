# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-expert SiTU + MXFP8 epilogue of the packed prefill MoE GEMM.

The packed GEMM computes ``[latent_down | shared gate_up]`` of the layer input.
Its shared gate_up weight rows are interleaved in 128-row chunks (``[gate
c..c+127 | up c..c+127]`` per 256-wide N tile), so every epilogue thread holds the
gate and up accumulators of the same output columns: epilogue subtile ``ni`` is
the gate of subtile ``ni + n/2``. Per 32 output columns of a row the thread

1. rounds gate and up to BF16 (the stock GEMM output),
2. applies ``mxfp8_act`` mode 4 (the routed experts' MegaMoE FC1 SiTU form:
   ``ex2.approx`` / ``rcp.approx`` on the same operation order) and rounds to
   BF16,
3. quantizes the 32 values exactly as ``quant8_store`` / ``situ4_quant_store``
   (UE8M0 rounded up from ``amax * fp32(1/448)``, saturating E4M3),

and stores the 32 E4M3 bytes plus one scale byte straight into the
``Mxfp8Buffer`` the packed down/up GEMM reads. So the bytes equal
``mxfp8_act.situ_and_mul_quantize_fast`` of the BF16 GEMM output whenever the
accumulators are equal (the GEMM's own summation order may differ from
cuBLASLt).
"""

import re

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Float32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Constexpr, T, dsl_user_op
from cutlass.utils.gemm.sm100 import (
    epilogue_tmem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

_ONE = "0f3F800000"
_NEG_ONE = "0fBF800000"
_CLAMP = "0f4E6E6B28"  # 1.0e9f
_INV_448 = "0f3B124925"  # fp32(1 / 448)


def _situ_pair_ptx(a_pow: int, clamp_up: bool) -> str:
    """PTX of ``situ4_pair`` (mxfp8_act.py) on two FP32 accumulator pairs.

    Operands: $0 bf16x2 result, $1 max |result|; $2/$3 gate, $4/$5 up
    accumulators; $6 k_b, $7 k_a, $8 k_c, $9 bb.
    """
    regs = (
        ".reg .b32 gp, up, hp, t0, t1;\n"
        ".reg .f32 g<2>, u<2>, xb<2>, eb<2>, ea<2>, xa<2>, xc<2>, ec<2>, n<2>, "
        "m<2>, num<2>, d<2>, e<2>, f<2>, den<2>, rc<2>, r<2>, h<2>;\n"
    )
    body = [
        "cvt.rn.bf16x2.f32 gp, $3, $2;",
        "cvt.rn.bf16x2.f32 up, $5, $4;",
        "shl.b32 t0, gp, 16;",
        "and.b32 t1, gp, 0xFFFF0000;",
        "mov.b32 g0, t0;",
        "mov.b32 g1, t1;",
        "shl.b32 t0, up, 16;",
        "and.b32 t1, up, 0xFFFF0000;",
        "mov.b32 u0, t0;",
        "mov.b32 u1, t1;",
    ]
    for i in range(2):
        body += [
            f"mul.rn.f32 xb{i}, g{i}, $6;",
            f"ex2.approx.ftz.f32 eb{i}, xb{i};",
            f"min.f32 eb{i}, eb{i}, {_CLAMP};",
        ]
        if a_pow > 0:
            body.append(f"mov.f32 ea{i}, eb{i};")
            body += [f"mul.rn.f32 ea{i}, ea{i}, eb{i};"] * (a_pow - 1)
        else:
            body += [
                f"mul.rn.f32 xa{i}, g{i}, $7;",
                f"ex2.approx.ftz.f32 ea{i}, xa{i};",
            ]
        body.append(f"fma.rn.f32 n{i}, eb{i}, {_NEG_ONE}, {_ONE};")
        if clamp_up:
            body += [
                f"mul.rn.f32 xc{i}, u{i}, $8;",
                f"ex2.approx.ftz.f32 ec{i}, xc{i};",
                f"min.f32 ec{i}, ec{i}, {_CLAMP};",
                f"fma.rn.f32 m{i}, ec{i}, {_NEG_ONE}, {_ONE};",
                f"mul.rn.f32 num{i}, n{i}, m{i};",
                f"mul.rn.f32 num{i}, num{i}, $9;",
                f"add.rn.f32 d{i}, ea{i}, {_ONE};",
                f"add.rn.f32 e{i}, eb{i}, {_ONE};",
                f"add.rn.f32 f{i}, ec{i}, {_ONE};",
                f"mul.rn.f32 den{i}, d{i}, e{i};",
                f"mul.rn.f32 den{i}, den{i}, f{i};",
            ]
        else:
            body += [
                f"mul.rn.f32 num{i}, n{i}, $9;",
                f"mul.rn.f32 num{i}, num{i}, u{i};",
                f"add.rn.f32 d{i}, ea{i}, {_ONE};",
                f"add.rn.f32 e{i}, eb{i}, {_ONE};",
                f"mul.rn.f32 den{i}, d{i}, e{i};",
            ]
        body += [
            f"rcp.approx.ftz.f32 rc{i}, den{i};",
            f"mul.rn.f32 r{i}, num{i}, rc{i};",
        ]
    body += [
        "cvt.rn.bf16x2.f32 hp, r1, r0;",
        "mov.b32 $0, hp;",
        "shl.b32 t0, hp, 16;",
        "and.b32 t1, hp, 0xFFFF0000;",
        "and.b32 t0, t0, 0x7FFFFFFF;",
        "and.b32 t1, t1, 0x7FFFFFFF;",
        "mov.b32 h0, t0;",
        "mov.b32 h1, t1;",
        "max.f32 $1, h0, h1;",
    ]
    return "{\n" + regs + "\n".join(body) + "\n}"


def _situ_pair_tanh_ptx(clamp_up: bool) -> str:
    """SiTU via MUFU tanh: beta*tanh(g/beta) * sigmoid(g) * (lb*tanh(u/lb) | u),
    sigmoid(g) = 0.5 + 0.5*tanh(g/2) (approximate; not bit-equal to the stock
    ex2/rcp form). Same operands as ``_situ_pair_packed_ptx``."""
    b = [
        ".reg .b32 gp, up, t0, t1, hp, kb, kc;",
        ".reg .f32 g<2>, u<2>, a<2>, c<2>, q<2>, r<2>, h<2>;",
        "mov.b32 gp, $2;",
        "mov.b32 up, $3;",
        "shl.b32 t0, gp, 16;", "and.b32 t1, gp, 0xFFFF0000;", "mov.b32 g0, t0;", "mov.b32 g1, t1;",
        "shl.b32 t0, up, 16;", "and.b32 t1, up, 0xFFFF0000;", "mov.b32 u0, t0;", "mov.b32 u1, t1;",
        "mul.rn.f32 kb, $4, 0fBEB17218;",  # k_b * (-ln2 / 2) = 1 / beta
        "mul.rn.f32 kc, $6, 0fBEB17218;",  # k_c * (-ln2 / 2) = 1 / linear_beta
    ]
    for i in range(2):
        b += [
            f"mul.rn.f32 a{i}, g{i}, kb;",
            f"tanh.approx.f32 a{i}, a{i};",
            f"mul.rn.f32 q{i}, g{i}, 0f3F000000;",
            f"tanh.approx.f32 q{i}, q{i};",
            f"fma.rn.f32 q{i}, q{i}, 0f3F000000, 0f3F000000;",
        ]
        if clamp_up:
            b += [
                f"mul.rn.f32 c{i}, u{i}, kc;",
                f"tanh.approx.f32 c{i}, c{i};",
                f"mul.rn.f32 r{i}, a{i}, c{i};",
                f"mul.rn.f32 r{i}, r{i}, $7;",
            ]
        else:
            b += [f"mul.rn.f32 r{i}, a{i}, $7;", f"mul.rn.f32 r{i}, r{i}, u{i};"]
        b.append(f"mul.rn.f32 r{i}, r{i}, q{i};")
    b += [
        "cvt.rn.bf16x2.f32 hp, r1, r0;",
        "mov.b32 $0, hp;",
        "shl.b32 t0, hp, 16;", "and.b32 t1, hp, 0xFFFF0000;",
        "and.b32 t0, t0, 0x7FFFFFFF;", "and.b32 t1, t1, 0x7FFFFFFF;",
        "mov.b32 h0, t0;", "mov.b32 h1, t1;",
        "max.f32 $1, h0, h1;",
    ]
    return "{\n" + "\n".join(b) + "\n}"


def _situ_pair_packed_ptx(a_pow: int, clamp_up: bool, probe: int = 0) -> str:
    """``_situ_pair_ptx`` on gate/up values already rounded to BF16 and packed
    (bf16x2, element 0 in the low half). Operands: $0 bf16x2 result, $1 max
    |result|; $2 gate, $3 up bf16x2; $4 k_b, $5 k_a, $6 k_c, $7 bb."""
    src = _situ_pair_ptx(a_pow, clamp_up)
    src = src.replace("cvt.rn.bf16x2.f32 gp, $3, $2;", "mov.b32 gp, $2;")
    src = src.replace("cvt.rn.bf16x2.f32 up, $5, $4;", "mov.b32 up, $3;")
    remap = {"$6": "$4", "$7": "$5", "$8": "$6", "$9": "$7"}
    src = re.sub(r"\$[6-9]", lambda m: remap[m.group(0)], src)
    if probe == 2:
        return _situ_pair_tanh_ptx(clamp_up)
    if probe == 3:
        # Microbench probe only (wrong values): every FP32 op -> one integer op.
        src = src.replace("0f", "0x")
        src = re.sub(r"(mul|add)\.rn\.f32 (\w+), (\w+), (\w+);", r"xor.b32 \2, \3, \4;", src)
        src = re.sub(r"fma\.rn\.f32 (\w+), (\w+), (\w+), (\w+);", r"add.u32 \1, \2, \3;", src)
        src = re.sub(r"(ex2|rcp)\.approx\.ftz\.f32 (\w+), (\w+);", r"not.b32 \2, \3;", src)
        src = re.sub(r"min\.f32 (\w+), (\w+), (\w+);", r"min.u32 \1, \2, \3;", src)
        src = src.replace("max.f32 $1, h0, h1;", "max.u32 h0, h0, h1;\nmov.b32 $1, h0;")
        src = src.replace(".reg .f32", ".reg .b32")
    if probe == 1:
        # Microbench probe only (wrong values): MUFU ops -> FMUL.
        src = re.sub(r"(ex2|rcp)\.approx\.ftz\.f32 (\w+), (\w+);", r"mul.rn.f32 \2, \3, 0f3F000000;", src)
    return src


@dsl_user_op
def warp_yield(*, loc=None, ip=None) -> None:
    """``nanosleep 0``: lets the scheduler pick another warp (MMA/TMA)."""
    llvm.inline_asm(
        None, [], "nanosleep.u32 0;", "", has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def situ_pair_packed(
    gp: Uint32,
    up: Uint32,
    k_b: Float32,
    k_a: Float32,
    k_c: Float32,
    bb: Float32,
    a_pow: int,
    clamp_up: bool,
    probe: int = 0,
    *,
    loc=None,
    ip=None,
) -> tuple[Uint32, Float32]:
    """SiTU of two BF16 (gate, up) pairs: (bf16x2 bits, max |value|)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [Uint32(gp).ir_value(loc=loc, ip=ip), Uint32(up).ir_value(loc=loc, ip=ip)]
        + [Float32(v).ir_value(loc=loc, ip=ip) for v in (k_b, k_a, k_c, bb)],
        _situ_pair_packed_ptx(a_pow, clamp_up, probe),
        "=r,=f,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    h = llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)
    amax = llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)
    return Uint32(h), Float32(amax)


def _situ_pair_x2_ptx(a_pow: int, clamp_up: bool) -> str:
    """``_situ_pair_packed_ptx`` with the FP32 add/mul/fma on f32x2 pairs (same
    IEEE round-to-nearest per lane, so bit-identical) and the max |value| kept
    packed: $0 bf16x2 result, $1 bf16x2 |result|; $2 gate, $3 up bf16x2;
    $4 k_b, $5 k_a, $6 k_c, $7 bb."""
    b = [
        ".reg .b32 t0, t1, x0, x1, y0, y1, z0, z1, hp;",
        ".reg .b64 G, U, K2, ONE2, NEG2, XB, EB, EA, N, XC, EC, M, NUM, D, E, F, DEN, RC, R;",
        "mov.b32 t0, " + _ONE + ";",
        "mov.b64 ONE2, {t0, t0};",
        "mov.b32 t0, " + _NEG_ONE + ";",
        "mov.b64 NEG2, {t0, t0};",
        "shl.b32 t0, $2, 16;",
        "and.b32 t1, $2, 0xFFFF0000;",
        "mov.b64 G, {t0, t1};",
        "shl.b32 t0, $3, 16;",
        "and.b32 t1, $3, 0xFFFF0000;",
        "mov.b64 U, {t0, t1};",
        "mov.b32 t0, $4;",
        "mov.b64 K2, {t0, t0};",
        "mul.rn.f32x2 XB, G, K2;",
        "mov.b64 {x0, x1}, XB;",
        "ex2.approx.ftz.f32 y0, x0;",
        "ex2.approx.ftz.f32 y1, x1;",
        f"min.f32 y0, y0, {_CLAMP};",
        f"min.f32 y1, y1, {_CLAMP};",
        "mov.b64 EB, {y0, y1};",
    ]
    if a_pow > 0:
        b.append("mov.b64 EA, EB;")
        b += ["mul.rn.f32x2 EA, EA, EB;"] * (a_pow - 1)
    else:
        b += [
            "mov.b32 t0, $5;",
            "mov.b64 K2, {t0, t0};",
            "mul.rn.f32x2 XB, G, K2;",
            "mov.b64 {x0, x1}, XB;",
            "ex2.approx.ftz.f32 z0, x0;",
            "ex2.approx.ftz.f32 z1, x1;",
            "mov.b64 EA, {z0, z1};",
        ]
    b.append("fma.rn.f32x2 N, EB, NEG2, ONE2;")
    if clamp_up:
        b += [
            "mov.b32 t0, $6;",
            "mov.b64 K2, {t0, t0};",
            "mul.rn.f32x2 XC, U, K2;",
            "mov.b64 {x0, x1}, XC;",
            "ex2.approx.ftz.f32 y0, x0;",
            "ex2.approx.ftz.f32 y1, x1;",
            f"min.f32 y0, y0, {_CLAMP};",
            f"min.f32 y1, y1, {_CLAMP};",
            "mov.b64 EC, {y0, y1};",
            "fma.rn.f32x2 M, EC, NEG2, ONE2;",
            "mul.rn.f32x2 NUM, N, M;",
            "mov.b32 t0, $7;",
            "mov.b64 K2, {t0, t0};",
            "mul.rn.f32x2 NUM, NUM, K2;",
            "add.rn.f32x2 D, EA, ONE2;",
            "add.rn.f32x2 E, EB, ONE2;",
            "add.rn.f32x2 F, EC, ONE2;",
            "mul.rn.f32x2 DEN, D, E;",
            "mul.rn.f32x2 DEN, DEN, F;",
        ]
    else:
        b += [
            "mov.b32 t0, $7;",
            "mov.b64 K2, {t0, t0};",
            "mul.rn.f32x2 NUM, N, K2;",
            "mul.rn.f32x2 NUM, NUM, U;",
            "add.rn.f32x2 D, EA, ONE2;",
            "add.rn.f32x2 E, EB, ONE2;",
            "mul.rn.f32x2 DEN, D, E;",
        ]
    b += [
        "mov.b64 {x0, x1}, DEN;",
        "rcp.approx.ftz.f32 y0, x0;",
        "rcp.approx.ftz.f32 y1, x1;",
        "mov.b64 RC, {y0, y1};",
        "mul.rn.f32x2 R, NUM, RC;",
        "mov.b64 {x0, x1}, R;",
        "cvt.rn.bf16x2.f32 hp, x1, x0;",
        "mov.b32 $0, hp;",
        "abs.bf16x2 $1, hp;",
    ]
    return "{\n" + "\n".join(b) + "\n}"


@dsl_user_op
def situ_pair_x2(
    gp: Uint32,
    up: Uint32,
    k_b: Float32,
    k_a: Float32,
    k_c: Float32,
    bb: Float32,
    a_pow: int,
    clamp_up: bool,
    *,
    loc=None,
    ip=None,
) -> tuple[Uint32, Uint32]:
    """``situ_pair_packed`` on f32x2: (bf16x2 bits, bf16x2 |bits|)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(gp).ir_value(loc=loc, ip=ip), Uint32(up).ir_value(loc=loc, ip=ip)]
        + [Float32(v).ir_value(loc=loc, ip=ip) for v in (k_b, k_a, k_c, bb)],
        _situ_pair_x2_ptx(a_pow, clamp_up),
        "=r,=r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    h = llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)
    a = llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)
    return Uint32(h), Uint32(a)


@dsl_user_op
def bf16x2_max(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f32_of_bf16x2_max(a: Uint32, *, loc=None, ip=None) -> Float32:
    """max of the two (non-negative) BF16 halves as FP32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(a).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b32 t0, t1;\nshl.b32 t0, $1, 16;\nand.b32 t1, $1, 0xFFFF0000;\n"
            "max.f32 $0, t0, t1;\n}",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _drain_math_probe(frags, half, n_blk, rQ32, situ_k, a_pow, clamp_up):
    """Probe body (traced inline): the drain epilogue's SiTU + quant math."""
    k_b, k_a, k_c, bb = situ_k
    x = Uint32(0)
    for ni in range(half):
        rG = frags[ni]
        rU = frags[ni + half]
        for vi in range(n_blk):
            hs = []
            amax = Float32(0.0)
            for p in range(16):
                h, am = situ_pair_packed(
                    rG[vi * 16 + p], rU[vi * 16 + p], k_b, k_a, k_c, bb, a_pow, clamp_up
                )
                hs.append(h)
                amax = fmax_f32(amax, am)
            inv = inv_scale_of_ue8m0(ue8m0_of_amax(amax))
            for q in range(8):
                x = x ^ e4m3x4_of_bf16x4(hs[2 * q], hs[2 * q + 1], inv)
    return x


def _tree_max_bf16x2(vals: list) -> Uint32:
    """Pairwise max tree (short dependency chains), traced at compile time."""
    if len(vals) == 1:
        return vals[0]
    h = len(vals) // 2
    return bf16x2_max(_tree_max_bf16x2(vals[:h]), _tree_max_bf16x2(vals[h:]))


@dsl_user_op
def e4m3x4_of_bf16x4_x2(
    h01: Uint32, h23: Uint32, inv: Float32, *, loc=None, ip=None
) -> Uint32:
    """``e4m3x4_of_bf16x4`` with the scale products on f32x2 (bit-identical)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(h01).ir_value(loc=loc, ip=ip),
                Uint32(h23).ir_value(loc=loc, ip=ip),
                Float32(inv).ir_value(loc=loc, ip=ip),
            ],
            "{\n"
            ".reg .b32 t0, t1, v0, v1, v2, v3, s;\n"
            ".reg .b64 A, B, S;\n"
            ".reg .b16 p0, p1;\n"
            "mov.b32 s, $3;\n"
            "mov.b64 S, {s, s};\n"
            "shl.b32 t0, $1, 16;\n"
            "and.b32 t1, $1, 0xFFFF0000;\n"
            "mov.b64 A, {t0, t1};\n"
            "shl.b32 t0, $2, 16;\n"
            "and.b32 t1, $2, 0xFFFF0000;\n"
            "mov.b64 B, {t0, t1};\n"
            "mul.rn.f32x2 A, A, S;\n"
            "mul.rn.f32x2 B, B, S;\n"
            "mov.b64 {v0, v1}, A;\n"
            "mov.b64 {v2, v3}, B;\n"
            "cvt.rn.satfinite.e4m3x2.f32 p0, v1, v0;\n"
            "cvt.rn.satfinite.e4m3x2.f32 p1, v3, v2;\n"
            "mov.b32 $0, {p0, p1};\n"
            "}",
            "=r,r,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def situ_pair_bf16(
    g0: Float32,
    g1: Float32,
    u0: Float32,
    u1: Float32,
    k_b: Float32,
    k_a: Float32,
    k_c: Float32,
    bb: Float32,
    a_pow: int,
    clamp_up: bool,
    *,
    loc=None,
    ip=None,
) -> tuple[Uint32, Float32]:
    """SiTU of two (gate, up) accumulator pairs: (bf16x2 bits, max |value|)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [
            Float32(v).ir_value(loc=loc, ip=ip)
            for v in (g0, g1, u0, u1, k_b, k_a, k_c, bb)
        ],
        _situ_pair_ptx(a_pow, clamp_up),
        "=r,=f,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    h = llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)
    amax = llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)
    return Uint32(h), Float32(amax)


@dsl_user_op
def fmax_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ue8m0_of_amax(amax: Float32, *, loc=None, ip=None) -> Uint32:
    """``float_to_ue8m0(amax * (1.0f / 448.0f))`` of mxfp8_act.py."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(amax).ir_value(loc=loc, ip=ip)],
            "{\n"
            ".reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;\n"
            ".reg .u32 bits, exp_biased, mantissa, bump, result;\n"
            ".reg .f32 v;\n"
            f"mul.rn.f32 v, $1, {_INV_448};\n"
            "setp.le.f32 p_zero, v, 0f00000000;\n"
            "mov.b32 bits, v;\n"
            "shr.b32 exp_biased, bits, 23;\n"
            "and.b32 exp_biased, exp_biased, 255;\n"
            "and.b32 mantissa, bits, 0x7FFFFF;\n"
            "setp.ne.u32 p_has_mant, mantissa, 0;\n"
            "selp.u32 bump, 1, 0, p_has_mant;\n"
            "setp.eq.u32 p_exp_zero, exp_biased, 0;\n"
            "setp.le.u32 p_tiny_sub, mantissa, 0x400000;\n"
            "and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;\n"
            "@p_tiny_sub mov.u32 bump, 0;\n"
            "add.u32 result, exp_biased, bump;\n"
            "setp.gt.u32 p_ovf, result, 254;\n"
            "selp.u32 result, 254, result, p_ovf;\n"
            "selp.u32 $0, 0, result, p_zero;\n"
            "}",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def inv_scale_of_ue8m0(e: Uint32, *, loc=None, ip=None) -> Float32:
    """``ue8m0_to_inv_scale`` of mxfp8_act.py."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(e).ir_value(loc=loc, ip=ip)],
            "{\n"
            ".reg .s32 ne;\n"
            ".reg .b32 fb;\n"
            ".reg .pred pz;\n"
            "setp.eq.u32 pz, $1, 0;\n"
            "sub.s32 ne, 254, $1;\n"
            "max.s32 ne, ne, 0;\n"
            "shl.b32 fb, ne, 23;\n"
            "mov.b32 $0, fb;\n"
            "@pz mov.b32 $0, 0;\n"
            "}",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def e4m3x4_of_bf16x4(
    h01: Uint32, h23: Uint32, inv: Float32, *, loc=None, ip=None
) -> Uint32:
    """Four BF16 values times ``inv`` as saturating E4M3 bytes (``fp8x2``)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(h01).ir_value(loc=loc, ip=ip),
                Uint32(h23).ir_value(loc=loc, ip=ip),
                Float32(inv).ir_value(loc=loc, ip=ip),
            ],
            "{\n"
            ".reg .b32 t0, t1, t2, t3;\n"
            ".reg .f32 v0, v1, v2, v3;\n"
            ".reg .b16 p0, p1;\n"
            "shl.b32 t0, $1, 16;\n"
            "and.b32 t1, $1, 0xFFFF0000;\n"
            "shl.b32 t2, $2, 16;\n"
            "and.b32 t3, $2, 0xFFFF0000;\n"
            "mov.b32 v0, t0;\n"
            "mov.b32 v1, t1;\n"
            "mov.b32 v2, t2;\n"
            "mov.b32 v3, t3;\n"
            "mul.rn.f32 v0, v0, $3;\n"
            "mul.rn.f32 v1, v1, $3;\n"
            "mul.rn.f32 v2, v2, $3;\n"
            "mul.rn.f32 v3, v3, $3;\n"
            "cvt.rn.satfinite.e4m3x2.f32 p0, v1, v0;\n"
            "cvt.rn.satfinite.e4m3x2.f32 p1, v3, v2;\n"
            "mov.b32 $0, {p0, p1};\n"
            "}",
            "=r,r,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def epilogue_situ_mxfp8(
    gemm_kernel,
    epi_tidx: cutlass.Int32,
    tCtAcc_base: cute.Tensor,
    tCgQ_base: cute.Tensor,
    tCgSF_base: cute.Tensor,
    tCcQ_base: cute.Tensor,
    epi_tile: cute.Tile,
    situ_k: tuple,
    a_pow: Constexpr,
    clamp_up: Constexpr,
    mma_tile_coord_mnl: tuple,
    m_rows: cutlass.Int32,
    acc_consumer_state: pipeline.PipelineState,
    acc_pipeline: pipeline.PipelineAsync,
) -> pipeline.PipelineState:
    """One shared tile: SiTU + MXFP8 of the gate/up accumulator halves, stored
    directly (32 E4M3 bytes and one scale byte per thread and 32 columns)."""
    k_b, k_a, k_c, bb = situ_k
    tCgQ = transform_partitioned_tensor_layout(tCgQ_base)
    tCgSF = transform_partitioned_tensor_layout(tCgSF_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)

    tiled_copy_t2r, tTR_tAcc_base, tTR_rG = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, tCtAcc, tCgQ, epi_tile, gemm_kernel.use_2cta_instrs
    )
    thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
    tTR_gQ = thr_copy_t2r.partition_D(cute.flat_divide(tCgQ, epi_tile))[
        (None, None, None, None, None, *mma_tile_coord_mnl)
    ]
    tTR_gSF = thr_copy_t2r.partition_D(cute.flat_divide(tCgSF, epi_tile))[
        (None, None, None, None, None, *mma_tile_coord_mnl)
    ]
    tTR_cQ = thr_copy_t2r.partition_D(
        cute.flat_divide(transform_partitioned_tensor_layout(tCcQ_base), epi_tile)
    )[(None, None, None, None, None, *mma_tile_coord_mnl)]
    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

    tTR_rU = cute.make_rmem_tensor(tTR_rG.layout, cutlass.Float32)
    rQ = cute.make_rmem_tensor(tTR_rG.layout, cutlass.Float8E4M3FN)
    rQ32 = cute.recast_tensor(rQ, cutlass.Uint32)
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Float8E4M3FN, num_bits_per_copy=128
    )

    acc_pipeline.consumer_wait(acc_consumer_state)

    epi_m = cute.size(tTR_tAcc.shape, mode=[3])
    epi_n = cute.size(tTR_tAcc.shape, mode=[4])
    half = epi_n // 2
    n_blk = cute.size(tTR_rG) // 32
    for mi in cutlass.range_constexpr(epi_m):
        for ni in cutlass.range_constexpr(half):
            cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, mi, ni)], tTR_rG)
            cute.copy(
                tiled_copy_t2r, tTR_tAcc[(None, None, None, mi, ni + half)], tTR_rU
            )
            if cutlass.const_expr(mi == epi_m - 1 and ni == half - 1):
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
            row = tTR_cQ[(None, None, None, mi, ni)][0][0]
            sf_frg = cute.logical_divide(
                tTR_gSF[(None, None, None, mi, ni)], cute.make_layout(32)
            )
            for vi in cutlass.range_constexpr(n_blk):
                hs = []
                amax = Float32(0.0)
                for p in cutlass.range_constexpr(16):
                    i = vi * 32 + 2 * p
                    h, am = situ_pair_bf16(
                        tTR_rG[i], tTR_rG[i + 1], tTR_rU[i], tTR_rU[i + 1],
                        k_b, k_a, k_c, bb, a_pow, clamp_up,
                    )  # fmt: skip
                    hs.append(h)
                    amax = fmax_f32(amax, am)
                e = ue8m0_of_amax(amax)
                inv = inv_scale_of_ue8m0(e)
                for q in cutlass.range_constexpr(8):
                    rQ32[vi * 8 + q] = e4m3x4_of_bf16x4(hs[2 * q], hs[2 * q + 1], inv)
                if row < (m_rows + 127) // 128 * 128:
                    sf_frg[0, vi] = e.to(cutlass.Uint8)
            if row < m_rows:
                cute.copy(store_atom, rQ, tTR_gQ[(None, None, None, mi, ni)])
    return acc_consumer_state


@cute.jit
def epilogue_situ_mxfp8_drain(
    gemm_kernel,
    tidx: cutlass.Int32,
    tCtAcc_base: cute.Tensor,
    tCgQ_base: cute.Tensor,
    tCgSF_base: cute.Tensor,
    tCcQ_base: cute.Tensor,
    epi_tile: cute.Tile,
    situ_k: tuple,
    a_pow: Constexpr,
    clamp_up: Constexpr,
    mma_tile_coord_mnl: tuple,
    m_rows: cutlass.Int32,
    acc_consumer_state: pipeline.PipelineState,
    acc_pipeline: pipeline.PipelineAsync,
    skel: Constexpr = 0,
    yield_mode: Constexpr = 0,
    x2: Constexpr = False,
    tanh: Constexpr = False,
) -> pipeline.PipelineState:
    """Drain-and-release form of ``epilogue_situ_mxfp8`` for the 1-stage
    B-reuse tile (8 epilogue warps; warp group g owns M block g): every thread
    loads its TMEM row of all gate/up subtiles, rounds them to BF16 (the stock
    GEMM output) into registers and hands TMEM back to the MMA warp, then runs
    the same SiTU + MXFP8 math and stores while the next tile's mainloop
    runs. Bytes equal ``epilogue_situ_mxfp8``."""
    k_b, k_a, k_c, bb = situ_k
    g = tidx // 128
    gt = tidx % 128
    tCgQ = transform_partitioned_tensor_layout(tCgQ_base)
    tCgSF = transform_partitioned_tensor_layout(tCgSF_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)

    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
        gemm_kernel, gt, tCtAcc, tCgQ, epi_tile, gemm_kernel.use_2cta_instrs
    )
    thr_copy_t2r = tiled_copy_t2r.get_slice(gt)
    tTR_gQ = thr_copy_t2r.partition_D(cute.flat_divide(tCgQ, epi_tile))[
        (None, None, None, None, None, *mma_tile_coord_mnl)
    ]
    tTR_gSF = thr_copy_t2r.partition_D(cute.flat_divide(tCgSF, epi_tile))[
        (None, None, None, None, None, *mma_tile_coord_mnl)
    ]
    tTR_cQ = thr_copy_t2r.partition_D(
        cute.flat_divide(transform_partitioned_tensor_layout(tCcQ_base), epi_tile)
    )[(None, None, None, None, None, *mma_tile_coord_mnl)]
    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]
    epi_n = cute.size(tTR_tAcc.shape, mode=[4])
    half = epi_n // 2
    n_blk = cute.size(tTR_rAcc) // 32

    acc_pipeline.consumer_wait(acc_consumer_state)
    frags = []
    for ni in cutlass.range_constexpr(epi_n):
        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, g, ni)], tTR_rAcc)
        rB = cute.make_rmem_tensor(tTR_rAcc.shape, cutlass.BFloat16)
        rB.store(tTR_rAcc.load().to(cutlass.BFloat16))
        frags.append(cute.recast_tensor(rB, Uint32))
    cute.arch.fence_view_async_tmem_load()
    with cute.arch.elect_one():
        acc_pipeline.consumer_release(acc_consumer_state)
    acc_consumer_state.advance()

    rQ = cute.make_rmem_tensor(tTR_rAcc.layout, cutlass.Float8E4M3FN)
    rQ32 = cute.recast_tensor(rQ, cutlass.Uint32)
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Float8E4M3FN, num_bits_per_copy=128
    )
    if cutlass.const_expr(skel == 1):
        # Skeleton: keep the drained values live, no math, no stores.
        acc = Uint32(0)
        for f in frags:
            for i in cutlass.range_constexpr(cute.size(f)):
                acc = acc ^ f[i]
        if m_rows < 0:
            if acc == Uint32(0x12345677):
                rQ32[0] = acc
                cute.copy(store_atom, rQ, tTR_gQ[(None, None, None, g, 0)])
        return acc_consumer_state
    if cutlass.const_expr(skel == 6):
        # Probe: same math instruction count, issued only by the warps of
        # SMSPs 2 and 3 (twice each); SMSPs 0/1 (MMA, TMA warps) get none.
        if (tidx // 32) % 4 >= 2:
            x = _drain_math_probe(frags, half, n_blk, rQ32, situ_k, a_pow, clamp_up)
            x = x ^ _drain_math_probe(frags, half, n_blk, rQ32, situ_k, a_pow, clamp_up)
            rQ32[0] = x
        if m_rows < 0:
            cute.copy(store_atom, rQ, tTR_gQ[(None, None, None, g, 0)])
        return acc_consumer_state
    for ni in cutlass.range_constexpr(half // 2 if skel == 4 else half):
        rG = frags[ni]
        rU = frags[ni + half]
        row = tTR_cQ[(None, None, None, g, ni)][0][0]
        sf_frg = cute.logical_divide(
            tTR_gSF[(None, None, None, g, ni)], cute.make_layout(32)
        )
        for vi in cutlass.range_constexpr(n_blk):
            hs = []
            if cutlass.const_expr(x2):
                am2 = []
                for p in cutlass.range_constexpr(16):
                    h, a2 = situ_pair_x2(
                        rG[vi * 16 + p], rU[vi * 16 + p],
                        k_b, k_a, k_c, bb, a_pow, clamp_up,
                    )  # fmt: skip
                    hs.append(h)
                    am2.append(a2)
                amax = f32_of_bf16x2_max(_tree_max_bf16x2(am2))
            else:
                amax = Float32(0.0)
                for p in cutlass.range_constexpr(16):
                    h, am = situ_pair_packed(
                        rG[vi * 16 + p], rU[vi * 16 + p],
                        k_b, k_a, k_c, bb, a_pow, clamp_up, 1 if skel == 3 else 3 if skel == 5 else 2 if tanh else 0,
                    )  # fmt: skip
                    hs.append(h)
                    amax = fmax_f32(amax, am)
            e = ue8m0_of_amax(amax)
            inv = inv_scale_of_ue8m0(e)
            for q in cutlass.range_constexpr(8):
                if cutlass.const_expr(x2):
                    rQ32[vi * 8 + q] = e4m3x4_of_bf16x4_x2(hs[2 * q], hs[2 * q + 1], inv)
                else:
                    rQ32[vi * 8 + q] = e4m3x4_of_bf16x4(hs[2 * q], hs[2 * q + 1], inv)
            if row < (m_rows + 127) // 128 * 128:
                sf_frg[0, vi] = e.to(cutlass.Uint8)
            if cutlass.const_expr(yield_mode == 1):
                warp_yield()
        if cutlass.const_expr(yield_mode == 2):
            warp_yield()
        if cutlass.const_expr(skel >= 2):
            if m_rows < 0:
                cute.copy(store_atom, rQ, tTR_gQ[(None, None, None, g, ni)])
        elif row < m_rows:
            cute.copy(store_atom, rQ, tTR_gQ[(None, None, None, g, ni)])
    return acc_consumer_state
