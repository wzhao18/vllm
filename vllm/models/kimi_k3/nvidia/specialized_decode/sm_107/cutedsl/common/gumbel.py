# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The stock Gumbel-max noise of ``vllm/v1/worker/gpu/sample/gumbel.py`` (fp32 path),
bitwise, for the vocabulary-sharded samplers.

Stock draws, for request seed ``s``, position ``p`` and vocabulary index ``t``:
``r = murmur3_hash32(s, p, t)``, ``u = murmur3_uniform32``'s conversion of ``r``,
``noise = -log(-_log1p_neg_stable(u))``, and samples ``argmax(logits / T + noise)``
(drafting adds ``_DRAFT_NOISE_SALT`` to ``p``; temperature 0: plain argmax). The PTX
below is the container Triton's (3.8.0) lowering of that code, instruction for
instruction: integer murmur3, the exact uniform conversion, the degree-8 series with
``fma.rn.f32``, libdevice ``__nv_logf`` (ftz flavour, inlined: range reduction, the
``fma.rn.ftz`` polynomial, the inf / NaN and zero fixes, the denormal rescale of the
second log) and, for the temperature, the SASS of Triton's ``div.full.f32``. Checked
bitwise against the stock ``_gumbel_sample_kernel``'s scores (drafting and residual
draws, every vocabulary block, temperatures 0, 1 and random).
"""

import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, Uint64

from .primitives import asm, float_bits

DRAFT_NOISE_SALT = 1 << 30  # stock _DRAFT_NOISE_SALT

_MIX = (
    "mul.lo.u32 k,{key},0xCC9E2D51;"
    "shf.l.wrap.b32 k,k,k,15;"
    "mul.lo.u32 k,k,0x1B873593;"
    "xor.b32 h,h,k;"
    "shf.l.wrap.b32 h,h,h,13;"
    "mad.lo.u32 h,h,5,0xE6546B64;"
)


@asm(Uint32, pure=True)
def murmur_prefix(seed_lo, seed_hi, pos_lo):
    """murmur3_hash32's state after the seed and position words (before the index):
    ``mix(mix(mix(0, seed & 0xFFFFFFFF), seed >> 32), pos & 0xFFFFFFFF)``."""
    code = (
        "{.reg .b32 h,k; mov.b32 h,0;"
        + _MIX.format(key="$1")
        + _MIX.format(key="$2")
        + _MIX.format(key="$3")
        + "mov.b32 $0,h;}"
    )
    return code, "=r,r,r,r", (Uint32(seed_lo), Uint32(seed_hi), Uint32(pos_lo))


# __nv_logf (ftz) of register x (> 0 normal or rescaled) into L; e0 the exponent
# offset register (0 or -23). Clobbers ei, mb, e, i, f, q, pf.
_LOGF = (
    "add.s32 ei,{x},-1059760811;"
    "and.b32 ei,ei,-8388608;"
    "sub.s32 mb,{x},ei;"
    "cvt.rn.f32.s32 e,ei;"
    "fma.rn.ftz.f32 i,e,0f34000000,{e0};"
    "add.f32 f,mb,0fBF800000;"
    "fma.rn.ftz.f32 q,0fBE055027,f,0f3E1039F6;"
    "fma.rn.ftz.f32 q,q,f,0fBDF8CDCC;"
    "fma.rn.ftz.f32 q,q,f,0f3E0F2955;"
    "fma.rn.ftz.f32 q,q,f,0fBE2AD8B9;"
    "fma.rn.ftz.f32 q,q,f,0f3E4CED0B;"
    "fma.rn.ftz.f32 q,q,f,0fBE7FFF22;"
    "fma.rn.ftz.f32 q,q,f,0f3EAAAA78;"
    "fma.rn.ftz.f32 q,q,f,0fBF000000;"
    "mul.f32 q,f,q;"
    "fma.rn.ftz.f32 q,q,f,f;"
    "fma.rn.ftz.f32 {out},i,0f3F317218,q;"
    "setp.lt.u32 pf,{x},2139095040;"
    "@!pf fma.rn.ftz.f32 {out},{x},0f7F800000,0f7F800000;"
)


@asm(Float32, pure=True)
def gumbel_log(prefix, index):
    """``log(-_log1p_neg_stable(u))`` for the stock draw at vocabulary ``index`` from a
    ``murmur_prefix`` state: the Gumbel noise is its negation."""
    code = (
        "{.reg .b32 h,k,r,hi,lo,a,b,u,p,s,d,ld,x,y,z,ea,ei,mb,e,i,f,q,L;"
        ".reg .pred pf,lt,sm,zr;"
        "mov.b32 h,$1;"
        + _MIX.format(key="$2")
        # fmix32(h ^ 16)
        + "xor.b32 h,h,16;"
        "shr.u32 r,h,16; xor.b32 h,h,r;"
        "mul.lo.u32 h,h,0x85EBCA6B;"
        "shr.u32 r,h,13; xor.b32 h,h,r;"
        "mul.lo.u32 h,h,0xC2B2AE35;"
        "shr.u32 r,h,16; xor.b32 h,h,r;"
        # murmur3_uniform32: hi16 * 2^-16 + (lo16 + 0.5) * 2^-32 (exact products)
        "shr.u32 hi,h,16; and.b32 lo,h,65535;"
        "cvt.rn.f32.u32 a,hi; cvt.rn.f32.u32 b,lo;"
        "add.f32 b,b,0f3F000000;"
        "mul.f32 b,b,0f2F800000;"
        "fma.rn.f32 u,a,0f37800000,b;"
        # _log1p_neg_stable(u)
        "fma.rn.f32 p,u,0f3E000000,0f3E124925;"
        "fma.rn.f32 p,u,p,0f3E2AAAAB;"
        "fma.rn.f32 p,u,p,0f3E4CCCCD;"
        "fma.rn.f32 p,u,p,0f3E800000;"
        "fma.rn.f32 p,u,p,0f3EAAAAAB;"
        "fma.rn.f32 p,u,p,0f3F000000;"
        "fma.rn.f32 p,u,p,0f3F800000;"
        "neg.f32 s,u;"
        "mul.f32 s,p,s;"
        "sub.f32 d,0f3F800000,u;"
        "max.f32 d,d,0f33800000;"
        + _LOGF.format(x="d", e0="0f00000000", out="ld")
        + "setp.lt.f32 lt,u,0f3E800000;"
        "selp.f32 x,s,ld,lt;"
        # log(-x), denormal -x rescaled by 2^23
        "neg.f32 y,x;"
        "setp.gt.f32 sm,x,0f80800000;"
        "mul.f32 z,x,0fCB000000;"
        "selp.f32 z,z,y,sm;"
        "selp.f32 ea,0fC1B80000,0f00000000,sm;"
        + _LOGF.format(x="z", e0="ea", out="L")
        + "setp.eq.f32 zr,z,0f00000000;"
        "selp.f32 L,0fFF800000,L,zr;"
        "mov.b32 $0,L;}"
    )
    return code, "=f,r,r", (Uint32(prefix), Uint32(index))


@asm(Float32, pure=True)
def div_full(a, b):
    """Triton's fp32 ``/`` as the stock sampler runs it: its PTX ``div.full.f32``
    assembled by Triton's ptxas (container SASS): both operands scaled by 1/4 when
    ``|b| > 2^126`` and by 2^24 when ``|b|`` is below the smallest normal, then
    ``a * MUFU.RCP(b)``, rounded on its own (``.rn``: no contraction with the add of
    the noise, as in the stock SASS)."""
    code = (
        "{.reg .b32 x,y,a,r; .reg .pred big,small;"
        "mov.b32 x,$1; mov.b32 y,$2; abs.f32 a,$2;"
        "setp.gt.f32 big,a,0f7E800000;"
        "setp.lt.f32 small,a,0f00800000;"
        "@big mul.rn.f32 x,x,0f3E800000; @big mul.rn.f32 y,y,0f3E800000;"
        "@small mul.rn.f32 x,x,0f4B800000; @small mul.rn.f32 y,y,0f4B800000;"
        "rcp.approx.ftz.f32 r,y;"
        "mul.rn.f32 $0,x,r;}"
    )
    return code, "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def sub_f32(a, b):
    """``a + (-b)`` as stock's ``logits + gumbel_noise`` (``.rn``: one rounding, never
    contracted with a preceding multiply)."""
    return "sub.rn.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@cute.jit
def ordered32(value: Float32):
    """Monotone u32 image of an FP32 value (-0 as +0, as an FP32 compare finds them
    equal; NaN above everything)."""
    bits = Uint32(float_bits(value))
    if bits == Uint32(0x80000000):
        bits = Uint32(0)
    result = bits | Uint32(0x80000000)
    if (bits & Uint32(0x80000000)) != Uint32(0):
        result = ~bits
    if (bits & Uint32(0x7FFFFFFF)) > Uint32(0x7F800000):
        result = Uint32(0xFFFFFFFF)
    return result


@cute.jit
def sampled_key(value: Float32, index):
    """(FP32 value, lowest index wins) as one u64 below 2^63 with no zero word:
    ``ordered32(value) << 31 | (2^31 - 1 - index)``. The stock samplers' argmax
    (``tl.max(..., return_indices=True)`` then the lowest block) keeps the lowest index
    among equal maxima."""
    low = Uint64(Uint32(0x7FFFFFFF) - Uint32(index))
    return (Uint64(ordered32(value)) << 31) | low


@cute.jit
def sampled_token(key: Uint64):
    return Int32(Uint32(0x7FFFFFFF) - Uint32(key & Uint64(0x7FFFFFFF)))


@cute.jit
def drafted_value(logit: Float32, temperature: Float32, prefix, index):
    """The stock drafter's Gumbel-max score of one BF16-valued logit: ``logit / T +
    noise`` (``T`` != 0), or the logit itself at temperature 0."""
    value = logit
    if temperature != Float32(0.0):
        value = sub_f32(div_full(logit, temperature), gumbel_log(prefix, index))
    return value


@cute.jit
def request_prefix(seed: Int64, position: Int64):
    """``murmur_prefix`` of a request's seed and a (salted) position."""
    return murmur_prefix(
        Uint32(seed & Int64(0xFFFFFFFF)),
        Uint32((seed >> 32) & Int64(0xFFFFFFFF)),
        Uint32(position & Int64(0xFFFFFFFF)),
    )


__all__ = [
    "DRAFT_NOISE_SALT",
    "div_full",
    "drafted_value",
    "gumbel_log",
    "murmur_prefix",
    "ordered32",
    "request_prefix",
    "sampled_key",
    "sampled_token",
    "sub_f32",
]
