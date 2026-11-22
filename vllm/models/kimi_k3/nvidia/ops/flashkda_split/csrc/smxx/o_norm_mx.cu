// Kimi-K3: the KDA output gate norm (FusedRMSNormGated, sigmoid, D = 128) of
// every (token, head) row written as the MXFP8 o_proj input, at the HBM rate.
// Per row the expressions and instructions of ops/kda_norm.py
// _kda_rmsnorm_gated_mxfp8_kernel (norm_mx.cuh). 8 lanes per row, 16 columns
// per lane (two lanes per MX block), 4 rows per warp step, kSteps steps of
// loads in flight. The sum of squares is a lane butterfly (another order than
// Triton's tl.sum: stock tolerance). Element pairs run in packed FP32
// (mul/add.rn.f32x2, cvt.rn.bf16x2.f32, max.bf16x2): bitwise the scalar ops.
// Head counts that are a multiple of 32 take o_norm_tma_kernel instead: the
// same arithmetic (bitwise), 4 lanes per row (one MX block each) fed from
// shared memory by a TMA ring, so the HBM loads stay in flight while the
// warps compute (F16K DEP16 rank: 115.5 -> 95.9 us on VR200).
#include <algorithm>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>

#include "norm_mx.cuh"
#include "k31_spin_guard.cuh"  // bounded mbarrier waits
K31SG_DEFINE_SET_RANK(o_norm_mx)  // rank of the timeout line (binding: set_spin_rank)

namespace {

constexpr int kSteps = 2;  // 4-row warp steps with loads in flight

__device__ __forceinline__ uint4 ld_nc_v4(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) {
  return __uint_as_float(w & 0xffff0000u);
}

// r / heads for 0 <= r < 2^31: (r * m) >> s with s = 31 + ceil(log2 heads),
// m = ceil(2^s / heads) < 2^32 (error < 2^-ceil(log2 heads) <= 1 / heads).
__device__ __forceinline__ int div_heads(int r, uint32_t m, int s) {
  return int((uint64_t(uint32_t(r)) * m) >> s);
}

// 4 CTAs (32 warps) per SM, 64 registers: occupancy-bound on VR200 (F16K 122 -> 115.7 us;
// 3/5/6/8 CTAs per SM or more rows per warp step are slower).
__global__ void __launch_bounds__(256, 4) o_norm_mx_kernel(
    const __nv_bfloat16* __restrict__ x, int rows, int heads, uint32_t hmul, int hshift,
    NormMxArgs const a) {
  const int lane = threadIdx.x % 32;
  const int sub = lane / 8;   // row of the warp step
  const int part = lane % 8;  // columns 16 part + [0, 16)
  // 32-bit row arithmetic (rows < 2^31): a 64-bit division per row costs more
  // than the row.
  const int warp = int((blockIdx.x * blockDim.x + threadIdx.x) / 32);
  const int nwarps = int(gridDim.x * blockDim.x / 32);
  float w[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = a.w[part * 16 + i];
  const auto* g = static_cast<const __nv_bfloat16*>(a.g);
  for (int r0 = warp * 4 * kSteps; r0 < rows; r0 += nwarps * 4 * kSteps) {
    uint4 xv[kSteps][2], gv[kSteps][2];
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
      const int r = min(r0 + 4 * s + sub, rows - 1);
      const int tok = div_heads(r, hmul, hshift), head = r - tok * heads;
      const __nv_bfloat16* xp = x + int64_t(r) * 128 + part * 16;
      const __nv_bfloat16* gp = g + int64_t(tok) * a.g_ld + head * 128 + part * 16;
      xv[s][0] = ld_nc_v4(xp);
      xv[s][1] = ld_nc_v4(xp + 8);
      gv[s][0] = ld_nc_v4(gp);
      gv[s][1] = ld_nc_v4(gp + 8);
    }
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
      const int r = r0 + 4 * s + sub;
      // Element pairs in packed FP32 (mul/add .rn.f32x2 round each lane as the
      // scalar op: bitwise the scalar expressions, half the FMA-pipe issue).
      float2 xf[8];
      const uint32_t xw[8] = {xv[s][0].x, xv[s][0].y, xv[s][0].z, xv[s][0].w,
                              xv[s][1].x, xv[s][1].y, xv[s][1].z, xv[s][1].w};
      const uint32_t gw[8] = {gv[s][0].x, gv[s][0].y, gv[s][0].z, gv[s][0].w,
                              gv[s][1].x, gv[s][1].y, gv[s][1].z, gv[s][1].w};
#pragma unroll
      for (int i = 0; i < 8; ++i) xf[i] = make_float2(lo(xw[i]), hi(xw[i]));
      float ss = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {  // same left-to-right sum as the scalar loop
        const float2 sq = __fmul2_rn(xf[i], xf[i]);
        ss = __fadd_rn(__fadd_rn(ss, sq.x), sq.y);
      }
#pragma unroll
      for (int o = 1; o <= 4; o <<= 1)
        ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
      const float rstd = norm_mx::div_full(
          1.0f, norm_mx::sqrt_approx(
                    __fadd_rn(norm_mx::div_full(ss, 128.0f), a.eps)));
      const float2 rstd2 = make_float2(rstd, rstd);
      float2 y[8];
      __nv_bfloat162 amax2 = __floats2bfloat162_rn(0.0f, 0.0f);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float2 t = __fmul2_rn(make_float2(lo(gw[i]), hi(gw[i])),
                                    make_float2(-1.4426950216293335f, -1.4426950216293335f));
        // ex2 .ftz: a flushed input or output only changes ex2 below 2^-126,
        // where 1 + ex2 is 1.0 either way. 1 / d for d in [1, 2^126] is the
        // reciprocal div.full forms (verified bitwise over every such d);
        // larger d take div.full itself.
        float ex, ey, rx, ry;
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(ex) : "f"(t.x));
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(ey) : "f"(t.y));
        const float2 d = __fadd2_rn(make_float2(1.0f, 1.0f), make_float2(ex, ey));
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(rx) : "f"(d.x));
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(ry) : "f"(d.y));
        if (fmaxf(d.x, d.y) > 0x1p126f) {
          rx = norm_mx::div_full(1.0f, d.x);
          ry = norm_mx::div_full(1.0f, d.y);
        }
        const float2 sg = make_float2(rx, ry);
        float2 v = __fmul2_rn(__fmul2_rn(xf[i], rstd2), make_float2(w[2 * i], w[2 * i + 1]));
        v = __fmul2_rn(v, sg);
        const __nv_bfloat162 b = __float22bfloat162_rn(v);  // cvt.rn.bf16x2.f32
        amax2 = __hmax2(amax2, __habs2(b));                   // exact on BF16 values
        const uint32_t bw = *reinterpret_cast<const uint32_t*>(&b);
        y[i] = make_float2(lo(bw), hi(bw));
      }
      float amax = fmaxf(__low2float(amax2), __high2float(amax2));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
      float inv;
      const uint32_t e = norm_mx::ue8m0(amax, inv);
      // The row's 4 block scales are 4 consecutive bytes of the swizzled
      // layout (col % 4): one 32-bit store from the row's first lane.
      uint32_t e4 = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        e4 |= __shfl_sync(0xffffffffu, e, (lane & ~7) + 2 * j) << (8 * j);
      if (r >= rows) continue;
      uint32_t q[4];
      const float2 inv2 = make_float2(inv, inv);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 u = __fmul2_rn(y[2 * i], inv2), v = __fmul2_rn(y[2 * i + 1], inv2);
        q[i] = uint32_t(norm_mx::fp8x2(u.x, u.y)) | (uint32_t(norm_mx::fp8x2(v.x, v.y)) << 16);
      }
      const int tok = div_heads(r, hmul, hshift), head = r - tok * heads;
      *reinterpret_cast<uint4*>(static_cast<uint8_t*>(a.q) + int64_t(tok) * a.q_ld +
                                head * 128 + part * 16) =
          make_uint4(q[0], q[1], q[2], q[3]);
      if (part == 0) {
        const int off = head * 512 + (tok % 32) * 16 + ((tok % 128) / 32) * 4 +
                        (tok / 128) * (128 * a.sf_cols);
        *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(a.sf) + off) = e4;
      }
    }
  }
}


// ---- TMA-fed variant (heads % 32 == 0) ----
__device__ __forceinline__ float sumsq16(const uint32_t* w) {
  float s;
  asm("{\n.reg .f32 l<8>, h<8>, s;\n"
      "shl.b32 l0, %1, 16;\nand.b32 h0, %1, -65536;\n"
      "shl.b32 l1, %2, 16;\nand.b32 h1, %2, -65536;\n"
      "shl.b32 l2, %3, 16;\nand.b32 h2, %3, -65536;\n"
      "shl.b32 l3, %4, 16;\nand.b32 h3, %4, -65536;\n"
      "shl.b32 l4, %5, 16;\nand.b32 h4, %5, -65536;\n"
      "shl.b32 l5, %6, 16;\nand.b32 h5, %6, -65536;\n"
      "shl.b32 l6, %7, 16;\nand.b32 h6, %7, -65536;\n"
      "shl.b32 l7, %8, 16;\nand.b32 h7, %8, -65536;\n"
      "mul.rn.ftz.f32 s, l0, l0;\nfma.rn.ftz.f32 s, h0, h0, s;\n"
      "fma.rn.ftz.f32 s, l1, l1, s;\nfma.rn.ftz.f32 s, h1, h1, s;\n"
      "fma.rn.ftz.f32 s, l2, l2, s;\nfma.rn.ftz.f32 s, h2, h2, s;\n"
      "fma.rn.ftz.f32 s, l3, l3, s;\nfma.rn.ftz.f32 s, h3, h3, s;\n"
      "fma.rn.ftz.f32 s, l4, l4, s;\nfma.rn.ftz.f32 s, h4, h4, s;\n"
      "fma.rn.ftz.f32 s, l5, l5, s;\nfma.rn.ftz.f32 s, h5, h5, s;\n"
      "fma.rn.ftz.f32 s, l6, l6, s;\nfma.rn.ftz.f32 s, h6, h6, s;\n"
      "fma.rn.ftz.f32 s, l7, l7, s;\nfma.rn.ftz.f32 s, h7, h7, s;\n"
      "mov.f32 %0, s;\n}"
      : "=f"(s)
      : "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]), "r"(w[4]), "r"(w[5]), "r"(w[6]),
        "r"(w[7]));
  return s;
}

__device__ __forceinline__ float ss_bfly(float a, float b) {
  float r;
  asm volatile(
      "{\n.reg .f32 s, t;\nadd.rn.ftz.f32 s, %1, %2;\n"
      "shfl.sync.bfly.b32 t, s, 1, 31, -1;\nadd.rn.ftz.f32 s, s, t;\n"
      "shfl.sync.bfly.b32 t, s, 2, 31, -1;\nadd.rn.ftz.f32 %0, s, t;\n}"
      : "=f"(r)
      : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float rstd_of(float ss, float eps) {
  float r;
  asm("{\n.reg .f32 a, b;\ndiv.full.f32 a, %1, 0f43000000;\n"
      "add.rn.ftz.f32 a, a, %2;\nsqrt.approx.f32 b, a;\n"
      "div.full.f32 %0, 0f3F800000, b;\n}"
      : "=f"(r)
      : "f"(ss), "f"(eps));
  return r;
}

// 1 when every BF16 gate value of the 16 words is >= -86
__device__ __forceinline__ bool gate_fast_ok(const uint32_t* g) {
  uint32_t m = g[0];
#pragma unroll
  for (int i = 1; i < 16; ++i)
    asm("min.bf16x2 %0, %0, %1;" : "+r"(m) : "r"(g[i]));
  const float lo = __uint_as_float(m << 16), hi = __uint_as_float(m & 0xffff0000u);
  return fminf(lo, hi) >= -86.0f;
}

#define PAIR_HEAD                                                              \
  "{\n.reg .b32 xl, xh, gl, gh, yp, ya;\n"                                      \
  ".reg .b64 X, G, E, Dd, V, W, S, R, CN, C1;\n"                                 \
  ".reg .f32 t0, t1, e0, e1, d0, d1, r0, r1, v0, v1, dm, cn, c1;\n"              \
  ".reg .pred p;\n"                                                              \
  "mov.f32 cn, 0fBFB8AA3B;\nmov.f32 c1, 0f3F800000;\n"                          \
  "mov.b64 CN, {cn, cn};\nmov.b64 C1, {c1, c1};\n"                              \
  "shl.b32 xl, %2, 16;\nand.b32 xh, %2, -65536;\n"                              \
  "shl.b32 gl, %3, 16;\nand.b32 gh, %3, -65536;\n"                              \
  "mov.b64 G, {gl, gh};\nmul.rn.ftz.f32x2 G, G, CN;\nmov.b64 {t0, t1}, G;\n"   \
  "ex2.approx.ftz.f32 e0, t0;\nex2.approx.ftz.f32 e1, t1;\n"                    \
  "mov.b64 E, {e0, e1};\nadd.rn.ftz.f32x2 Dd, C1, E;\nmov.b64 {d0, d1}, Dd;\n" \
  "rcp.approx.ftz.f32 r0, d0;\nrcp.approx.ftz.f32 r1, d1;\n"
#define PAIR_SLOW                                                              \
  "max.f32 dm, d0, d1;\nsetp.gt.f32 p, dm, 0f7E800000;\n"                       \
  "@p div.full.f32 r0, 0f3F800000, d0;\n@p div.full.f32 r1, 0f3F800000, d1;\n"
#define PAIR_TAIL                                                              \
  "mov.b64 X, {xl, xh};\nmov.b64 R, {%5, %5};\nmul.rn.ftz.f32x2 V, X, R;\n"    \
  "mov.b64 W, {%6, %7};\nmul.rn.ftz.f32x2 V, V, W;\n"                           \
  "mov.b64 S, {r0, r1};\nmul.rn.ftz.f32x2 V, V, S;\n"                           \
  "mov.b64 {v0, v1}, V;\ncvt.rn.bf16x2.f32 yp, v1, v0;\n"                       \
  "and.b32 ya, yp, 2147450879;\nmax.bf16x2 %1, %4, ya;\nmov.b32 %0, yp;\n}"

template <bool kSlow>
__device__ __forceinline__ uint32_t pair(uint32_t x, uint32_t g, uint32_t& amax, float rstd,
                                         float w0, float w1) {
  uint32_t y, am;
  if constexpr (kSlow) {
    asm(PAIR_HEAD PAIR_SLOW PAIR_TAIL
        : "=r"(y), "=r"(am)
        : "r"(x), "r"(g), "r"(amax), "f"(rstd), "f"(w0), "f"(w1));
  } else {
    asm(PAIR_HEAD PAIR_TAIL
        : "=r"(y), "=r"(am)
        : "r"(x), "r"(g), "r"(amax), "f"(rstd), "f"(w0), "f"(w1));
  }
  amax = am;
  return y;
}

__device__ __forceinline__ uint32_t ue8m0(uint32_t amax2, float& inv) {
  const float am = fmaxf(__uint_as_float(amax2 << 16), __uint_as_float(amax2 & 0xffff0000u));
  const float nm = __fmul_rn(am, 0.002232142857142857f);
  const uint32_t bits = __float_as_uint(nm);
  const uint32_t ex = (bits >> 23) & 255u;
  const uint32_t mant = bits & 0x7FFFFFu;
  uint32_t bump = mant != 0 ? 1u : 0u;
  if (ex == 0 && mant <= 0x400000u) bump = 0;
  uint32_t e = min(ex + bump, 254u);
  if (nm <= 0.0f) e = 0;
  inv = e == 0 ? 0.0f : __uint_as_float((254u - e) << 23);
  return e;
}

__device__ __forceinline__ uint32_t quant4(uint32_t y0, uint32_t y1, float inv) {
  uint32_t r;
  asm("{\n.reg .b32 a, b, c, d;\n.reg .b64 A, B, I;\n.reg .f32 a0, a1, b0, b1;\n"
      ".reg .b16 qa, qb;\n"
      "shl.b32 a, %1, 16;\nand.b32 b, %1, -65536;\nshl.b32 c, %2, 16;\nand.b32 d, %2, -65536;\n"
      "mov.b64 I, {%3, %3};\nmov.b64 A, {a, b};\nmov.b64 B, {c, d};\n"
      "mul.rn.ftz.f32x2 A, A, I;\nmul.rn.ftz.f32x2 B, B, I;\n"
      "mov.b64 {a0, a1}, A;\nmov.b64 {b0, b1}, B;\n"
      "cvt.rn.satfinite.e4m3x2.f32 qa, a1, a0;\ncvt.rn.satfinite.e4m3x2.f32 qb, b1, b0;\n"
      "mov.b32 %0, {qa, qb};\n}"
      : "=r"(r)
      : "r"(y0), "r"(y1), "f"(inv));
  return r;
}


constexpr int kStage = 16384;      // x 8 KiB + gate 8 KiB

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return uint32_t(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t b, uint32_t n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory");
}
// bounded (k31_spin_guard.cuh): site = the wait's line
#define mbar_wait(b, parity) k31sg::mbar_wait((b), (parity), "flashkda_o_norm_mx", __LINE__)
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* tm, int c0, int c1,
                                      uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tma3d(uint32_t dst, const CUtensorMap* tm, int c0, int c1,
                                      int c2, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(c2),
      "r"(bar) : "memory");
}
__device__ __forceinline__ uint4 lds_v4(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}

constexpr int kConsumerWarps = 16;
constexpr int kNS = 12;  // 16 KiB stages: 192 KiB of loads in flight per SM

__global__ void __launch_bounds__(32 * (kConsumerWarps + 1), 1) o_norm_tma_kernel(
    const __grid_constant__ CUtensorMap tm_x, const __grid_constant__ CUtensorMap tm_g,
    int nblk, int heads, NormMxArgs const a) {
  k31sg::begin_call((long long)(nblk));  // timeout lines: call index and size
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  __shared__ __align__(8) uint64_t full[kNS], empty[kNS];
  constexpr int kG = kConsumerWarps / 4;
  static_assert(kNS % kG == 0, "slot ownership");
  __shared__ __align__(16) float sw[128];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x < 128) sw[threadIdx.x] = a.w[threadIdx.x];
  if (threadIdx.x == 0) {
    for (int i = 0; i < kNS; ++i) {
      mbar_init(smem_u32(&full[i]), 1);
      mbar_init(smem_u32(&empty[i]), 4);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int b0 = int((int64_t(blockIdx.x) * nblk) / gridDim.x);
  const int b1 = int((int64_t(blockIdx.x + 1) * nblk) / gridDim.x);
  const int nb = b1 - b0;
  const uint32_t sbase = smem_u32(smem);
  if (warp == kConsumerWarps) {
    if (lane == 0) {
      for (int i = 0; i < nb; ++i) {
        const int s = i % kNS;
        if (i >= kNS) mbar_wait(smem_u32(&empty[s]), ((i / kNS) - 1) & 1);
        const uint32_t fb = smem_u32(&full[s]);
        mbar_expect_tx(fb, kStage);
        const int r0 = (b0 + i) * 32;
        const int tok = r0 / heads, h0 = r0 - tok * heads;
        const uint32_t st = sbase + s * kStage;
        tma2d(st, &tm_x, 0, r0, fb);
        tma2d(st + 4096, &tm_x, 64, r0, fb);
        tma3d(st + 8192, &tm_g, 0, h0, tok, fb);
        tma3d(st + 12288, &tm_g, 64, h0, tok, fb);
      }
    }
    return;
  }
  const int grp = warp / 4, wg = warp % 4;
  const int ph = lane >> 3, li = lane & 7;
  const int row = wg * 8 + (li & 3) + 4 * (ph & 1);
  const int qb = (li >> 2) + 2 * (ph >> 1);
  const float* wq = sw + qb * 32;
  const uint32_t roff = uint32_t(row) * 128 + uint32_t(qb >> 1) * 4096;
  const int sw7 = row & 7;
  for (int i = grp; i < nb; i += kG) {
    const int s = i % kNS;
    mbar_wait(smem_u32(&full[s]), (i / kNS) & 1);
    const uint32_t st = sbase + s * kStage + roff;
    uint32_t xw[16], gw[16];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t c = uint32_t((((qb & 1) * 4 + j) ^ sw7) << 4);
      const uint4 u = lds_v4(st + c);
      xw[4 * j] = u.x; xw[4 * j + 1] = u.y; xw[4 * j + 2] = u.z; xw[4 * j + 3] = u.w;
      const uint4 v = lds_v4(st + 8192 + c);
      gw[4 * j] = v.x; gw[4 * j + 1] = v.y; gw[4 * j + 2] = v.z; gw[4 * j + 3] = v.w;
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(smem_u32(&empty[s]));
    float sa = sumsq16(xw), sb = sumsq16(xw + 8);
    float ss;
    asm volatile(
        "{\n.reg .f32 s, t;\nadd.rn.ftz.f32 s, %1, %2;\n"
        "shfl.sync.bfly.b32 t, s, 4, 31, -1;\nadd.rn.ftz.f32 s, s, t;\n"
        "shfl.sync.bfly.b32 t, s, 16, 31, -1;\nadd.rn.ftz.f32 %0, s, t;\n}"
        : "=f"(ss) : "f"(sa), "f"(sb));
    const float rstd = rstd_of(ss, a.eps);
    uint32_t y[16], amax = 0;
    if (gate_fast_ok(gw)) {
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        const float2 w2 = *reinterpret_cast<const float2*>(wq + 2 * k);
        y[k] = pair<false>(xw[k], gw[k], amax, rstd, w2.x, w2.y);
      }
    } else {
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        const float2 w2 = *reinterpret_cast<const float2*>(wq + 2 * k);
        y[k] = pair<true>(xw[k], gw[k], amax, rstd, w2.x, w2.y);
      }
    }
    float inv;
    const uint32_t e = ue8m0(amax, inv);
    uint32_t q[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) q[k] = quant4(y[2 * k], y[2 * k + 1], inv);
    const int r = (b0 + i) * 32 + row;
    const int tok = r / heads, head = r - tok * heads;
    uint8_t* qp = static_cast<uint8_t*>(a.q) + int64_t(tok) * a.q_ld + head * 128 + qb * 32;
    *reinterpret_cast<uint4*>(qp) = make_uint4(q[0], q[1], q[2], q[3]);
    *reinterpret_cast<uint4*>(qp + 16) = make_uint4(q[4], q[5], q[6], q[7]);
    const int off = head * 512 + (tok % 32) * 16 + ((tok % 128) / 32) * 4 +
                    (tok / 128) * (128 * a.sf_cols) + qb;
    static_cast<uint8_t*>(a.sf)[off] = uint8_t(e);
  }
}


// 32-row blocks never straddle a token (heads % 32 == 0); TMA needs 16 B
// aligned bases and row strides.
bool launch_o_norm_tma(const void* x, int64_t tokens, int heads, NormMxArgs const& a,
                       int num_sms, cudaStream_t stream) {
  static PFN_cuTensorMapEncodeTiled_v12000 encode = nullptr;
  if (encode == nullptr) {
    cudaDriverEntryPointQueryResult qr;
    void* p = nullptr;
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr) !=
            cudaSuccess || qr != cudaDriverEntryPointSuccess)
      return false;
    encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  CUtensorMap tx, tg;
  {
    cuuint64_t dims[2] = {128, cuuint64_t(tokens * heads)};
    cuuint64_t strides[1] = {256};
    cuuint32_t box[2] = {64, 32}, es[2] = {1, 1};
    if (encode(&tx, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(x), dims, strides, box,
               es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return false;
  }
  {
    cuuint64_t dims[3] = {128, cuuint64_t(heads), cuuint64_t(tokens)};
    cuuint64_t strides[2] = {256, cuuint64_t(a.g_ld) * 2};
    cuuint32_t box[3] = {64, 32, 1}, es[3] = {1, 1, 1};
    if (encode(&tg, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(a.g), dims, strides,
               box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return false;
  }
  const int smem = kNS * kStage + 1024;
  if (cudaFuncSetAttribute(o_norm_tma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem) != cudaSuccess)
    return false;
  const int nblk = int(tokens * heads / 32);
  o_norm_tma_kernel<<<dim3(unsigned(std::min(num_sms, nblk))), dim3(32 * (kConsumerWarps + 1)),
                      smem, stream>>>(tx, tg, nblk, heads, a);
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace

bool launch_o_norm_mx(const void* x, int64_t tokens, int heads,
                      NormMxArgs const& a, int num_sms, cudaStream_t stream) {
  const int64_t rows = tokens * heads;
  if (rows == 0) return true;
  if (heads % 32 == 0 && a.g_ld % 8 == 0 &&
      (reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(a.g)) % 16 == 0)
    return launch_o_norm_tma(x, tokens, heads, a, num_sms, stream);
  int l = 0;
  while ((int64_t(1) << l) < heads) ++l;
  const int hshift = 31 + l;
  const uint32_t hmul = uint32_t(((uint64_t(1) << hshift) + uint64_t(heads) - 1) / uint64_t(heads));
  const int64_t warps_needed = (rows + 4 * kSteps - 1) / (4 * kSteps);
  const int64_t blocks =
      std::min<int64_t>((warps_needed + 7) / 8, int64_t(num_sms) * 8);
  o_norm_mx_kernel<<<dim3(unsigned(blocks)), dim3(256), 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), int(rows), heads, hmul, hshift, a);
  return cudaGetLastError() == cudaSuccess;
}
