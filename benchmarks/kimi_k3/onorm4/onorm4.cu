// Experimental gated o_norm + MXFP8 variants (microbench only): 4 lanes per
// (token, head) row, each lane owns one 32-column MX block. Same per-element
// arithmetic as smxx/o_norm_mx.cu (bitwise), fewer instructions per element.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

struct Args {
  const void* g;
  int64_t g_ld;
  const float* w;
  float eps;
  void* q;
  int64_t q_ld;
  void* sf;
  int sf_cols;
};

__device__ __forceinline__ uint4 ld_nc_v4(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ void bulk_prefetch_l2(const void* p, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}

__device__ __forceinline__ int div_heads(int r, uint32_t m, int s) {
  return int((uint64_t(uint32_t(r)) * m) >> s);
}

// sequential sum of squares of 16 BF16 values (8 words) in element order
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

// kSteps rows per lane per loop iteration (loads of all steps issued first).
template <int kSteps, int kMinBlocks, int kAhead>
__global__ void __launch_bounds__(256, kMinBlocks) o_norm4_kernel(
    const __nv_bfloat16* __restrict__ x, int rows, int heads, uint32_t hmul, int hshift,
    Args const a) {
  __shared__ __align__(16) float sw[128];
  sw[threadIdx.x % 128] = a.w[threadIdx.x % 128];
  __syncthreads();
  const int lane = threadIdx.x % 32;
  const int sub = lane / 4;  // row of the warp step (8 rows)
  const int qb = lane % 4;   // MX block of the row
  const int warp = int((blockIdx.x * blockDim.x + threadIdx.x) / 32);
  const int nwarps = int(gridDim.x * blockDim.x / 32);
  const auto* g = static_cast<const __nv_bfloat16*>(a.g);
  const float* wq = sw + qb * 32;
  const int stride = nwarps * 8 * kSteps;
  if (kAhead > 0 && lane == 0) {
    for (int d = 0; d < kAhead; ++d) {
      const int rp = warp * 8 * kSteps + d * stride;
      if (rp + 8 * kSteps <= rows) {
        const int tp = div_heads(rp, hmul, hshift), hp = rp - tp * heads;
        bulk_prefetch_l2(x + int64_t(rp) * 128, 8 * kSteps * 256);
        bulk_prefetch_l2(g + int64_t(tp) * a.g_ld + hp * 128, 8 * kSteps * 256);
      }
    }
  }
  for (int r0 = warp * 8 * kSteps; r0 < rows; r0 += stride) {
    if (kAhead > 0 && lane == 0) {
      const int rp = r0 + kAhead * stride;
      if (rp + 8 * kSteps <= rows) {
        const int tp = div_heads(rp, hmul, hshift), hp = rp - tp * heads;
        bulk_prefetch_l2(x + int64_t(rp) * 128, 8 * kSteps * 256);
        bulk_prefetch_l2(g + int64_t(tp) * a.g_ld + hp * 128, 8 * kSteps * 256);
      }
    }
    uint32_t xw[kSteps][16], gw[kSteps][16];
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
      const int r = min(r0 + 8 * s + sub, rows - 1);
      const int tok = div_heads(r, hmul, hshift), head = r - tok * heads;
      const __nv_bfloat16* xp = x + int64_t(r) * 128 + qb * 32;
      const __nv_bfloat16* gp = g + int64_t(tok) * a.g_ld + head * 128 + qb * 32;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint4 u = ld_nc_v4(xp + 8 * j);
        xw[s][4 * j] = u.x; xw[s][4 * j + 1] = u.y; xw[s][4 * j + 2] = u.z; xw[s][4 * j + 3] = u.w;
        const uint4 v = ld_nc_v4(gp + 8 * j);
        gw[s][4 * j] = v.x; gw[s][4 * j + 1] = v.y; gw[s][4 * j + 2] = v.z; gw[s][4 * j + 3] = v.w;
      }
    }
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
      const int r = r0 + 8 * s + sub;
      const float rstd = rstd_of(ss_bfly(sumsq16(xw[s]), sumsq16(xw[s] + 8)), a.eps);
      uint32_t y[16], amax = 0;
      if (gate_fast_ok(gw[s])) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const float2 w2 = *reinterpret_cast<const float2*>(wq + 2 * i);
          y[i] = pair<false>(xw[s][i], gw[s][i], amax, rstd, w2.x, w2.y);
        }
      } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const float2 w2 = *reinterpret_cast<const float2*>(wq + 2 * i);
          y[i] = pair<true>(xw[s][i], gw[s][i], amax, rstd, w2.x, w2.y);
        }
      }
      float inv;
      const uint32_t e = ue8m0(amax, inv);
      uint32_t q[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) q[k] = quant4(y[2 * k], y[2 * k + 1], inv);
      if (r < rows) {
        const int tok = div_heads(r, hmul, hshift), head = r - tok * heads;
        uint8_t* qp = static_cast<uint8_t*>(a.q) + int64_t(tok) * a.q_ld + head * 128 + qb * 32;
        *reinterpret_cast<uint4*>(qp) = make_uint4(q[0], q[1], q[2], q[3]);
        *reinterpret_cast<uint4*>(qp + 16) = make_uint4(q[4], q[5], q[6], q[7]);
        const int off = head * 512 + (tok % 32) * 16 + ((tok % 128) / 32) * 4 +
                        (tok / 128) * (128 * a.sf_cols) + qb;
        static_cast<uint8_t*>(a.sf)[off] = uint8_t(e);
      }
    }
  }
}

template <int kSteps, int kMinBlocks, int kAhead = 0>
void launch(const at::Tensor& x, int64_t rows, int heads, uint32_t hmul, int hshift,
            const Args& a, int occ_blocks) {
  int nsm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, x.get_device());
  const int64_t warps_needed = (rows + 8 * kSteps - 1) / (8 * kSteps);
  const int64_t blocks = std::min<int64_t>((warps_needed + 7) / 8, int64_t(nsm) * occ_blocks);
  o_norm4_kernel<kSteps, kMinBlocks, kAhead><<<dim3(unsigned(blocks)), dim3(256), 0,
                                       c10::cuda::getCurrentCUDAStream().stream()>>>(
      static_cast<const __nv_bfloat16*>(x.data_ptr()), int(rows), heads, hmul, hshift, a);
}


// ---------------------------------------------------------------------------
// TMA pipeline variant: persistent CTA per SM, one producer warp streams
// 32-row blocks (x [32 x 128] and the gate rows of the same 32 heads, SW128
// boxes of 64 columns) into a ring of stages; 16 consumer warps (4 groups of
// 4 warps, one 32-row stage per group) normalize + quantize from shared memory.
// Lane map per warp (8 rows): phase p = lane / 8, i = lane % 8,
// row = (i & 3) + 4 (p & 1), MX block q = (i >> 2) + 2 (p >> 1): conflict-free
// SW128 reads; a row's 4 lanes are xor-4 / xor-16 partners (same butterfly
// pairing as o_norm_mx).
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
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n}" ::"r"(b), "r"(parity) : "memory");
}
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

template <int kConsumerWarps, int kNS, int kCtas, int kProbe = 0>
__global__ void __launch_bounds__(32 * (kConsumerWarps + 1), kCtas) o_norm_tma_kernel(
    const __grid_constant__ CUtensorMap tm_x, const __grid_constant__ CUtensorMap tm_g,
    int nblk, int heads, Args const a) {
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
    if (kProbe != 2) {
      *reinterpret_cast<uint4*>(qp) = make_uint4(q[0], q[1], q[2], q[3]);
      *reinterpret_cast<uint4*>(qp + 16) = make_uint4(q[4], q[5], q[6], q[7]);
    } else if ((q[0] ^ q[7]) == 0x12345678u) {
      qp[0] = 1;
    }
    const int off = head * 512 + (tok % 32) * 16 + ((tok % 128) / 32) * 4 +
                    (tok / 128) * (128 * a.sf_cols) + qb;
    if (kProbe != 1) static_cast<uint8_t*>(a.sf)[off] = uint8_t(e);
    else if (e == 0x99u) static_cast<uint8_t*>(a.sf)[off] = uint8_t(e);
  }
}

PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult qr;
    void* p = nullptr;
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr) ==
                    cudaSuccess && qr == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

template <int kConsumerWarps, int kNS, int kCtas, int kProbe = 0>
void launch_tma(const at::Tensor& x, int64_t T, int heads, const at::Tensor& g, const Args& a) {
  CUtensorMap tx, tg;
  {
    cuuint64_t dims[2] = {128, cuuint64_t(T * heads)};
    cuuint64_t strides[1] = {256};
    cuuint32_t box[2] = {64, 32}, es[2] = {1, 1};
    TORCH_CHECK(encode_fn()(&tx, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(), dims, strides,
                            box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "tm x");
  }
  {
    cuuint64_t dims[3] = {128, cuuint64_t(heads), cuuint64_t(T)};
    cuuint64_t strides[2] = {256, cuuint64_t(g.stride(0)) * 2};
    cuuint32_t box[3] = {64, 32, 1}, es[3] = {1, 1, 1};
    TORCH_CHECK(encode_fn()(&tg, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, g.data_ptr(), dims, strides,
                            box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "tm g");
  }
  int nsm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, x.get_device());
  const int smem = kNS * kStage + 1024;
  static bool attr = false;
  auto kern = o_norm_tma_kernel<kConsumerWarps, kNS, kCtas, kProbe>;
  if (!attr) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr = true;
  }
  const int nblk = int(T * heads / 32);
  kern<<<dim3(unsigned(nsm * kCtas)), dim3(32 * (kConsumerWarps + 1)), smem,
                      c10::cuda::getCurrentCUDAStream().stream()>>>(tx, tg, nblk, heads, a);
}

}  // namespace

void run(at::Tensor x, at::Tensor g, at::Tensor w, double eps, at::Tensor q, at::Tensor sf,
         int64_t variant) {
  const int64_t H = x.size(x.dim() - 2), P = H * 128, T = x.numel() / P, rows = T * H;
  Args a{};
  a.g = g.data_ptr();
  a.g_ld = g.stride(0);
  a.w = w.data_ptr<float>();
  a.eps = float(eps);
  a.q = q.data_ptr();
  a.q_ld = q.stride(0);
  a.sf = sf.data_ptr();
  a.sf_cols = int(P / 32);
  int l = 0;
  while ((int64_t(1) << l) < H) ++l;
  const int hshift = 31 + l;
  const uint32_t hmul = uint32_t(((uint64_t(1) << hshift) + uint64_t(H) - 1) / uint64_t(H));
  switch (variant) {
    case 0: launch<1, 4>(x, rows, int(H), hmul, hshift, a, 4); break;
    case 1: launch<2, 2>(x, rows, int(H), hmul, hshift, a, 2); break;
    case 2: launch<1, 3>(x, rows, int(H), hmul, hshift, a, 3); break;
    case 3: launch<2, 3>(x, rows, int(H), hmul, hshift, a, 3); break;
    case 4: launch<1, 4>(x, rows, int(H), hmul, hshift, a, 8); break;
    case 5: launch<2, 2, 2>(x, rows, int(H), hmul, hshift, a, 2); break;
    case 6: launch<1, 4, 4>(x, rows, int(H), hmul, hshift, a, 4); break;
    case 7: launch<2, 2, 4>(x, rows, int(H), hmul, hshift, a, 2); break;
    case 8: launch<2, 2, 1>(x, rows, int(H), hmul, hshift, a, 2); break;
    case 9: launch_tma<16, 12, 1>(x, T, int(H), g, a); break;
    case 10: launch_tma<24, 12, 1>(x, T, int(H), g, a); break;
    case 11: launch_tma<8, 6, 2>(x, T, int(H), g, a); break;
    case 12: launch_tma<12, 6, 2>(x, T, int(H), g, a); break;
    case 13: launch_tma<28, 14, 1>(x, T, int(H), g, a); break;
    case 14: launch_tma<16, 8, 1>(x, T, int(H), g, a); break;
    case 15: launch_tma<16, 12, 1, 1>(x, T, int(H), g, a); break;
    case 16: launch_tma<16, 12, 1, 2>(x, T, int(H), g, a); break;
    default: TORCH_CHECK(false, "variant");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
