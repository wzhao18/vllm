// Kimi-K3: the KDA output gate norm (FusedRMSNormGated, sigmoid, D = 128) of
// every (token, head) row written as the MXFP8 o_proj input, at the HBM rate.
// Per row the expressions and instructions of ops/kda_norm.py
// _kda_rmsnorm_gated_mxfp8_kernel (norm_mx.cuh). 8 lanes per row, 16 columns
// per lane (two lanes per MX block), 4 rows per warp step, kSteps steps of
// loads in flight. The sum of squares is a lane butterfly (another order than
// Triton's tl.sum: stock tolerance).
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
      float xf[16];
      const uint32_t xw[8] = {xv[s][0].x, xv[s][0].y, xv[s][0].z, xv[s][0].w,
                              xv[s][1].x, xv[s][1].y, xv[s][1].z, xv[s][1].w};
      const uint32_t gw[8] = {gv[s][0].x, gv[s][0].y, gv[s][0].z, gv[s][0].w,
                              gv[s][1].x, gv[s][1].y, gv[s][1].z, gv[s][1].w};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        xf[2 * i] = lo(xw[i]);
        xf[2 * i + 1] = hi(xw[i]);
      }
      float ss = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) ss = __fadd_rn(ss, __fmul_rn(xf[i], xf[i]));
#pragma unroll
      for (int o = 1; o <= 4; o <<= 1)
        ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
      const float rstd = norm_mx::div_full(
          1.0f, norm_mx::sqrt_approx(
                    __fadd_rn(norm_mx::div_full(ss, 128.0f), a.eps)));
      float y[16], amax = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const float gf = (i % 2 == 0) ? lo(gw[i / 2]) : hi(gw[i / 2]);
        float v = __fmul_rn(__fmul_rn(xf[i], rstd), w[i]);
        v = __fmul_rn(v, norm_mx::sigmoid_triton(gf));
        y[i] = __bfloat162float(__float2bfloat16_rn(v));
        amax = fmaxf(amax, fabsf(y[i]));
      }
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
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        q[i] = uint32_t(norm_mx::fp8x2(__fmul_rn(y[4 * i], inv),
                                       __fmul_rn(y[4 * i + 1], inv))) |
               (uint32_t(norm_mx::fp8x2(__fmul_rn(y[4 * i + 2], inv),
                                        __fmul_rn(y[4 * i + 3], inv)))
                << 16);
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


// ---- v2: token rows staged in shared memory by bulk async copies --------------------------
// One CTA per SM walks tokens t = blockIdx.x, + gridDim.x, ... A producer lane copies the
// token's H x 128 o rows and its H x 128 gate columns (two contiguous rows of the inputs)
// into a kOStages-deep shared-memory ring with one transaction barrier per stage; 8
// consumer warps (two 4-row steps each at H = 96) normalize the H rows from shared memory with the v1 lane map and
// arithmetic, except: the products x * rstd * w * sigmoid and the E4M3 scaling run as
// packed f32x2 multiplies (same per-lane rounding), and sigmoid's 1 / (1 + e) is
// rcp.approx (equal to div.full for 1 + e <= 2^126; a row with a larger 1 + e is redone
// with div.full). Bit for bit the v1 output.
constexpr int kOStages = 4;
constexpr int kOConsumerWarps = 12;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n}" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}

__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}

__device__ __forceinline__ float rcp_approx(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// One 4-row warp step of token t: head h0 + lane / 8 of the staged stage.
__device__ __forceinline__ void o_norm_step(const uint8_t* stage, uint32_t row_bytes, int h0, int heads, int t,
                                            int sub, int part, int lane, const float (&w)[16], NormMxArgs const& a) {
    const int head = h0 + sub;
    const int hh = min(head, heads - 1);
    const uint4* xs = reinterpret_cast<const uint4*>(stage + hh * 256 + part * 32);
    const uint4* gs = reinterpret_cast<const uint4*>(stage + row_bytes + hh * 256 + part * 32);
    const uint4 xv0 = xs[0], xv1 = xs[1], gv0 = gs[0], gv1 = gs[1];
    const uint32_t xw[8] = {xv0.x, xv0.y, xv0.z, xv0.w, xv1.x, xv1.y, xv1.z, xv1.w};
    const uint32_t gw[8] = {gv0.x, gv0.y, gv0.z, gv0.w, gv1.x, gv1.y, gv1.z, gv1.w};
    float xf[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      xf[2 * i] = lo(xw[i]);
      xf[2 * i + 1] = hi(xw[i]);
    }
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; i += 2) {
      const float2 sq = __fmul2_rn(make_float2(xf[i], xf[i + 1]), make_float2(xf[i], xf[i + 1]));
      ss = __fadd_rn(__fadd_rn(ss, sq.x), sq.y);
    }
#pragma unroll
    for (int o = 1; o <= 4; o <<= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
    const float rstd =
        norm_mx::div_full(1.0f, norm_mx::sqrt_approx(__fadd_rn(norm_mx::div_full(ss, 128.0f), a.eps)));
    // sigmoid(g) = 1 / (1 + ex2(g * -log2 e)): the reciprocal, the division where 1 + e > 2^126
    float sg[16], dmax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; i += 2) {
      const float2 tt = __fmul2_rn(make_float2(lo(gw[i / 2]), hi(gw[i / 2])),
                                   make_float2(-1.44269502162933349609375f, -1.44269502162933349609375f));
      float e0, e1;
      asm("ex2.approx.f32 %0, %1;" : "=f"(e0) : "f"(tt.x));
      asm("ex2.approx.f32 %0, %1;" : "=f"(e1) : "f"(tt.y));
      const float2 d = __fadd2_rn(make_float2(1.0f, 1.0f), make_float2(e0, e1));
      dmax = fmaxf(dmax, fmaxf(d.x, d.y));
      sg[i] = rcp_approx(d.x);
      sg[i + 1] = rcp_approx(d.y);
    }
    if (__any_sync(0xffffffffu, dmax > 0x1p126f)) {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const float gf = (i % 2 == 0) ? lo(gw[i / 2]) : hi(gw[i / 2]);
        sg[i] = norm_mx::sigmoid_triton(gf);
      }
    }
    float y[16], amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; i += 2) {
      float2 v = __fmul2_rn(make_float2(xf[i], xf[i + 1]), make_float2(rstd, rstd));
      v = __fmul2_rn(v, make_float2(w[i], w[i + 1]));
      v = __fmul2_rn(v, make_float2(sg[i], sg[i + 1]));
      const __nv_bfloat162 vb = __floats2bfloat162_rn(v.x, v.y);
      y[i] = __low2float(vb);
      y[i + 1] = __high2float(vb);
      amax = fmaxf(amax, fmaxf(fabsf(y[i]), fabsf(y[i + 1])));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    float inv;
    const uint32_t e = norm_mx::ue8m0(amax, inv);
    uint32_t e4 = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) e4 |= __shfl_sync(0xffffffffu, e, (lane & ~7) + 2 * j) << (8 * j);
    if (head < heads) {
      uint32_t q[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 p0 = __fmul2_rn(make_float2(y[4 * i], y[4 * i + 1]), make_float2(inv, inv));
        const float2 p1 = __fmul2_rn(make_float2(y[4 * i + 2], y[4 * i + 3]), make_float2(inv, inv));
        q[i] = uint32_t(norm_mx::fp8x2(p0.x, p0.y)) | (uint32_t(norm_mx::fp8x2(p1.x, p1.y)) << 16);
      }
      *reinterpret_cast<uint4*>(static_cast<uint8_t*>(a.q) + int64_t(t) * a.q_ld + head * 128 + part * 16) =
          make_uint4(q[0], q[1], q[2], q[3]);
      if (part == 0) {
        const int off = head * 512 + (t % 32) * 16 + ((t % 128) / 32) * 4 + (t / 128) * (128 * a.sf_cols);
        *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(a.sf) + off) = e4;
      }
    }
}

__global__ void __launch_bounds__((kOConsumerWarps + 1) * 32, 1)
    o_norm_mx_tma_kernel(const __nv_bfloat16* __restrict__ x, int tokens, int heads, NormMxArgs const a) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t row_bytes = uint32_t(heads) * 128 * 2;
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + kOStages * 2 * row_bytes);
  uint64_t* empty = full + kOStages;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) {
    for (int s = 0; s < kOStages; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], kOConsumerWarps * 32);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const auto* g = static_cast<const __nv_bfloat16*>(a.g);
  if (warp == kOConsumerWarps) {
    if (lane == 0) {
      int it = 0;
      for (int t = blockIdx.x; t < tokens; t += gridDim.x, ++it) {
        const int s = it % kOStages;
        if (it >= kOStages) mbar_wait(&empty[s], ((it / kOStages) - 1) & 1);
        uint8_t* stage = smem + s * 2 * row_bytes;
        mbar_expect_tx(&full[s], 2 * row_bytes);
        bulk_g2s(stage, x + int64_t(t) * heads * 128, row_bytes, &full[s]);
        bulk_g2s(stage + row_bytes, g + int64_t(t) * a.g_ld, row_bytes, &full[s]);
      }
    }
    return;
  }
  const int sub = lane / 8;   // row of the warp step
  const int part = lane % 8;  // columns 16 part + [0, 16)
  float w[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = a.w[part * 16 + i];
  int it = 0;
  for (int t = blockIdx.x; t < tokens; t += gridDim.x, ++it) {
    const int s = it % kOStages;
    mbar_wait(&full[s], (it / kOStages) & 1);
    const uint8_t* stage = smem + s * 2 * row_bytes;
    for (int h0 = warp * 4; h0 < heads; h0 += 2 * kOConsumerWarps * 4) {
      o_norm_step(stage, row_bytes, h0, heads, t, sub, part, lane, w, a);
      if (h0 + kOConsumerWarps * 4 < heads)
        o_norm_step(stage, row_bytes, h0 + kOConsumerWarps * 4, heads, t, sub, part, lane, w, a);
    }
    mbar_arrive(&empty[s]);
  }
}

}  // namespace

// The staged kernel's shared memory: kOStages x (o row + gate row) + barriers.
static int o_norm_tma_smem(int heads) { return kOStages * 2 * heads * 128 * 2 + 2 * kOStages * 8; }

void launch_o_norm_mx_ldg(const void* x, int64_t tokens, int heads, NormMxArgs const& a, int num_sms,
                          cudaStream_t stream);

void launch_o_norm_mx(const void* x, int64_t tokens, int heads, NormMxArgs const& a, int num_sms,
                      cudaStream_t stream) {
  if (tokens == 0) return;
  const int smem = o_norm_tma_smem(heads);
  if (smem > 232448) {  // more heads than 4 staged token rows fit: the direct-load kernel
    launch_o_norm_mx_ldg(x, tokens, heads, a, num_sms, stream);
    return;
  }
  static int configured = 0;
  if (configured < smem) {
    cudaFuncSetAttribute(o_norm_mx_tma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = smem;
  }
  const int blocks = int(std::min<int64_t>(tokens, num_sms));
  o_norm_mx_tma_kernel<<<dim3(unsigned(blocks)), dim3((kOConsumerWarps + 1) * 32), smem, stream>>>(
      static_cast<const __nv_bfloat16*>(x), int(tokens), heads, a);
}

// v1 (direct loads), kept as the parity reference of the staged kernel.
void launch_o_norm_mx_ldg(const void* x, int64_t tokens, int heads,
                      NormMxArgs const& a, int num_sms, cudaStream_t stream) {
  const int64_t rows = tokens * heads;
  if (rows == 0) return;
  const int64_t warps_needed = (rows + 4 * kSteps - 1) / (4 * kSteps);
  const int64_t blocks =
      std::min<int64_t>((warps_needed + 7) / 8, int64_t(num_sms) * 8);
  o_norm_mx_kernel<<<dim3(unsigned(blocks)), dim3(256), 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), int(rows), heads, a);
}
