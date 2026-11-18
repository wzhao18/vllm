// Kimi-K3: the KDA output gate norm (FusedRMSNormGated, sigmoid, D = 128) of
// every (token, head) row written as the MXFP8 o_proj input, at the HBM rate.
// Per row the expressions and instructions of ops/kda_norm.py
// _kda_rmsnorm_gated_mxfp8_kernel (norm_mx.cuh); one warp per row, four
// columns per lane, ROWS rows in flight per warp. The sum of squares is a
// warp butterfly (another order than Triton's tl.sum: stock tolerance).
#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "norm_mx.cuh"

namespace {

constexpr int kRows = 8;  // rows in flight per warp

__device__ __forceinline__ uint2 ld_nc_v2(const void* p) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];"
               : "=r"(v.x), "=r"(v.y)
               : "l"(p));
  return v;
}

__device__ __forceinline__ void unpack4(uint2 v, float (&f)[4]) {
  f[0] = __uint_as_float(v.x << 16);
  f[1] = __uint_as_float(v.x & 0xffff0000u);
  f[2] = __uint_as_float(v.y << 16);
  f[3] = __uint_as_float(v.y & 0xffff0000u);
}

__global__ void __launch_bounds__(256) o_norm_mx_kernel(
    const __nv_bfloat16* __restrict__ x, int rows, int heads,
    NormMxArgs const a) {
  const int lane = threadIdx.x % 32;
  // 32-bit row arithmetic (rows < 2^31; a 64-bit division per row costs more
  // than the row itself).
  const int warp = int((blockIdx.x * blockDim.x + threadIdx.x) / 32);
  const int nwarps = int(gridDim.x * blockDim.x / 32);
  float w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) w[i] = a.w[lane * 4 + i];
  for (int r0 = warp * kRows; r0 < rows; r0 += nwarps * kRows) {
    uint2 xv[kRows], gv[kRows];
#pragma unroll
    for (int j = 0; j < kRows; ++j) {
      const int r = min(r0 + j, rows - 1);
      const int tok = r / heads, head = r - tok * heads;
      xv[j] = ld_nc_v2(x + int64_t(r) * 128 + lane * 4);
      gv[j] = ld_nc_v2(static_cast<const __nv_bfloat16*>(a.g) +
                       int64_t(tok) * a.g_ld + head * 128 + lane * 4);
    }
#pragma unroll
    for (int j = 0; j < kRows; ++j) {
      const int r = r0 + j;
      if (r >= rows) break;
      float xf[4], gf[4];
      unpack4(xv[j], xf);
      unpack4(gv[j], gf);
      float ss = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) ss = __fadd_rn(ss, __fmul_rn(xf[i], xf[i]));
#pragma unroll
      for (int o = 16; o >= 1; o >>= 1)
        ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
      const float rstd = norm_mx::div_full(
          1.0f, norm_mx::sqrt_approx(
                    __fadd_rn(norm_mx::div_full(ss, 128.0f), a.eps)));
      float y[4], amax = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float v = __fmul_rn(__fmul_rn(xf[i], rstd), w[i]);
        v = __fmul_rn(v, norm_mx::sigmoid_triton(gf[i]));
        y[i] = __bfloat162float(__float2bfloat16_rn(v));
        amax = fmaxf(amax, fabsf(y[i]));
      }
#pragma unroll
      for (int o = 4; o >= 1; o >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
      float inv;
      const uint32_t e = norm_mx::ue8m0(amax, inv);
      const uint32_t q =
          uint32_t(norm_mx::fp8x2(__fmul_rn(y[0], inv), __fmul_rn(y[1], inv))) |
          (uint32_t(norm_mx::fp8x2(__fmul_rn(y[2], inv), __fmul_rn(y[3], inv)))
           << 16);
      const int tok = r / heads, head = r - tok * heads;
      *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(a.q) + int64_t(tok) * a.q_ld +
                                   head * 128 + lane * 4) = q;
      if (lane % 8 == 0) {
        const int col = head * 4 + lane / 8;
        const int off = (col % 4) + (col / 4) * 512 + (tok % 32) * 16 +
                            ((tok % 128) / 32) * 4 +
                        (tok / 128) * (128 * a.sf_cols);
        static_cast<uint8_t*>(a.sf)[off] = uint8_t(e);
      }
    }
  }
}

}  // namespace

void launch_o_norm_mx(const void* x, int64_t tokens, int heads,
                      NormMxArgs const& a, int num_sms, cudaStream_t stream) {
  const int64_t rows = tokens * heads;
  if (rows == 0) return;
  const int64_t warps_needed = (rows + kRows - 1) / kRows;
  const int64_t blocks =
      std::min<int64_t>((warps_needed + 7) / 8, int64_t(num_sms) * 8);
  o_norm_mx_kernel<<<dim3(unsigned(blocks)), dim3(256), 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), int(rows), heads, a);
}
