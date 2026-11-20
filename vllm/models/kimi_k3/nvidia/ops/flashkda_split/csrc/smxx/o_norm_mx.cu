// Kimi-K3: the KDA output gate norm (FusedRMSNormGated, sigmoid, D = 128) of
// every (token, head) row written as the MXFP8 o_proj input, at the HBM rate.
// Per row the expressions and instructions of ops/kda_norm.py
// _kda_rmsnorm_gated_mxfp8_kernel (norm_mx.cuh). 8 lanes per row, 16 columns
// per lane (two lanes per MX block), 4 rows per warp step, kSteps steps of
// loads in flight. The sum of squares is a lane butterfly (another order than
// Triton's tl.sum: stock tolerance). Element pairs run in packed FP32
// (mul/add.rn.f32x2, cvt.rn.bf16x2.f32, max.bf16x2): bitwise the scalar ops.
#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "norm_mx.cuh"

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

__global__ void __launch_bounds__(256) o_norm_mx_kernel(
    const __nv_bfloat16* __restrict__ x, int rows, int heads, NormMxArgs const a) {
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
      const int tok = r / heads, head = r - tok * heads;
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
        float ex, ey;
        asm("ex2.approx.f32 %0, %1;" : "=f"(ex) : "f"(t.x));
        asm("ex2.approx.f32 %0, %1;" : "=f"(ey) : "f"(t.y));
        const float2 d = __fadd2_rn(make_float2(1.0f, 1.0f), make_float2(ex, ey));
        const float2 sg = make_float2(norm_mx::div_full(1.0f, d.x), norm_mx::div_full(1.0f, d.y));
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
      const int tok = r / heads, head = r - tok * heads;
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

}  // namespace

void launch_o_norm_mx(const void* x, int64_t tokens, int heads,
                      NormMxArgs const& a, int num_sms, cudaStream_t stream) {
  const int64_t rows = tokens * heads;
  if (rows == 0) return;
  const int64_t warps_needed = (rows + 4 * kSteps - 1) / (4 * kSteps);
  const int64_t blocks =
      std::min<int64_t>((warps_needed + 7) / 8, int64_t(num_sms) * 8);
  o_norm_mx_kernel<<<dim3(unsigned(blocks)), dim3(256), 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), int(rows), heads, a);
}
