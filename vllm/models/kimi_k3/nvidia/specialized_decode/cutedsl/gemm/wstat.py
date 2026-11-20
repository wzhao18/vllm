# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-stationary decode GEMM: the weight slice of every CTA is loaded into shared
memory BEFORE the PDL wait, while the predecessor (the SP all-gather / reduce, a
poll-bound kernel with almost no shared memory) still runs.

``out[T, N] = x[T, K] @ W[N, K]^T``: BF16 in, FP32 accumulation, BF16 out (one
rounding; the split-K partials are summed in rank order, so only the summation order
differs from stock's GEMM). Layout (from GEMM-EX2's ``prewait_gemm``):

* swap-AB tcgen05 ``kind::f16``: 128 weight rows are the MMA's M, the T tokens its N
  (TMEM accumulator [128 lanes x T columns]);
* split-K over the ``C`` CTAs of a cluster (uneven K-block ranges when C does not divide
  K / 64); CTA r owns output rows [r * 128 / C, ...) of its row tile, its peers push their
  FP32 partials into its shared memory (DSMEM, 32-column chunks) and it sums them in
  rank order;
* a DEEP weight ring (``NWS`` 16 KiB slots, 2D TMA boxes of the row-major weight with
  the 128-byte swizzle: no packed copy) decoupled from a short activation ring (``NXS``
  slots): the weight producer warps never execute ``griddepcontrol.wait``, so up to
  ``NWS`` x 16 KiB of each CTA's weight slice is on chip when the wait returns; only the
  activation producer waits.
"""

import hashlib
import os
import threading

import torch

_CPP_SRC = r"""
#include <torch/extension.h>
void wstat_gemm(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& w, int64_t n, int64_t C,
                int64_t NWS, int64_t NXS, int64_t NW, int64_t XL, int64_t KB, int64_t sleep_ns);
int64_t wstat_smem_optin();
void wstat_host(int64_t iters, int64_t ctas, torch::Tensor& sink);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

struct P {
  int N, K, T, C, NWS, NXS, NW, XL, KB, RT, KBT, packed, sleep_ns;
  unsigned idesc, tcols, x_off, w_off;
};

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint32_t b, unsigned c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(c));
}
__device__ __forceinline__ void mb_wait(uint32_t b, unsigned ph) {
  asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W;}"
               :: "r"(b), "r"(ph) : "memory");
}
// Waits that can span the predecessor's whole run (weight producers on a full ring, the MMA warp on
// the first activation stage) back off with nanosleep so the co-resident predecessor keeps its issue slots.
__device__ __forceinline__ void mb_wait_bo(uint32_t b, unsigned ph, unsigned ns) {
  uint32_t ok;
  asm volatile("{.reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; selp.u32 %0, 1, 0, P;}"
               : "=r"(ok) : "r"(b), "r"(ph) : "memory");
  while (!ok) {
    if (ns) __nanosleep(ns);
    asm volatile("{.reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; selp.u32 %0, 1, 0, P;}"
                 : "=r"(ok) : "r"(b), "r"(ph) : "memory");
  }
}
__device__ __forceinline__ uint32_t uni(uint32_t v) { return __shfl_sync(0xffffffffu, v, 0); }
__device__ __forceinline__ uint64_t sdesc(uint32_t saddr) {  // K-major SW128, SBO 1024 B, 15-bit address
  return (uint64_t)((saddr >> 4) & 0x7FFF) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) |
         ((uint64_t)1 << 46) | ((uint64_t)2 << 61);
}
// elected lane: expect_tx + 2D / 3D TMA box into this CTA's shared memory
__device__ __forceinline__ void tma2d_e(uint32_t dst, const CUtensorMap* m, int c0, int c1, uint32_t b,
                                        unsigned tx) {
  asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.and.b32 Q, %5, 0, P; "
               "@Q mbarrier.arrive.expect_tx.shared::cta.b64 _, [%4], %5; "
               "@P cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
               "[%0], [%1, {%2, %3}], [%4];}"
               :: "r"(dst), "l"(m), "r"(c0), "r"(c1), "r"(b), "r"(tx) : "memory");
}
__device__ __forceinline__ void tma3d_e(uint32_t dst, const CUtensorMap* m, int c0, int c1, int c2,
                                        uint32_t b, unsigned tx) {
  asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.and.b32 Q, %6, 0, P; "
               "@Q mbarrier.arrive.expect_tx.shared::cta.b64 _, [%5], %6; "
               "@P cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
               "[%0], [%1, {%2, %3, %4}], [%5];}"
               :: "r"(dst), "l"(m), "r"(c0), "r"(c1), "r"(c2), "r"(b), "r"(tx) : "memory");
}
__device__ __forceinline__ void bulk_e(uint32_t dst, const void* src, unsigned n, uint32_t b) {
  asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff; @P mbarrier.arrive.expect_tx.shared::cta.b64 _, [%3], %2; "
               "@P cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];}"
               :: "r"(dst), "l"(src), "r"(n), "r"(b) : "memory");
}
__device__ __forceinline__ void mma_e(uint32_t d, uint64_t a, uint64_t b, unsigned id, unsigned acc) {
  asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; "
               "@P tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, Q;}"
               :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
__device__ __forceinline__ void commit_e(uint32_t mb) {
  asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff; "
               "@P tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];}"
               :: "r"(mb) : "memory");
}
__device__ __forceinline__ void ld8nw(uint32_t ta, float* v) {
  uint32_t r[8];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
                 "=r"(r[7]) : "r"(ta));
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = __uint_as_float(r[i]);
}
__device__ __forceinline__ unsigned crank() {
  unsigned r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
}
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.acquire;" ::: "memory"); }

// warps 0-3 epilogue (TMEM lanes 32w..), warp 4 TMEM alloc + MMA, warps 5..5+XL-1 X producers (wait on
// the predecessor), then NW W producers (never wait). T tokens (MMA N), C = cluster K-split; one stage =
// KB 64-wide K blocks (W: KB 2D boxes on one barrier, X: one 3D box).
template <int T, int CC, int KB>
__global__ void __launch_bounds__(512, 1) wstat_k(const __grid_constant__ CUtensorMap xmap,
                                                  const __grid_constant__ CUtensorMap wmap,
                                                  const uint8_t* __restrict__ wp,
                                                  __nv_bfloat16* __restrict__ y, const P p) {
  constexpr unsigned WB1 = 16384, XB1 = T * 128, WB = WB1 * KB, XB = XB1 * KB;
  constexpr int RPR = (128 + CC - 1) / CC, RS = T + 4;
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = (uint8_t*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  __shared__ __align__(8) uint64_t wfull[16], wempty[16], xfull[8], xempty[8], accf;
  __shared__ uint32_t tbase;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned cx = CC > 1 ? crank() : 0;
  const int rt = blockIdx.y;
  const int kb0 = (int)(cx * p.KBT / CC), kb1 = (int)((cx + 1) * p.KBT / CC);
  const int nst = (kb1 - kb0 + KB - 1) / KB;  // the last stage may hold fewer blocks
  if (threadIdx.x == 0) {
    for (int s = 0; s < p.NWS; ++s) { mb_init(sa(&wfull[s]), 1); mb_init(sa(&wempty[s]), 1); }
    for (int s = 0; s < p.NXS; ++s) { mb_init(sa(&xfull[s]), 1); mb_init(sa(&xempty[s]), 1); }
    mb_init(sa(&accf), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (CC > 1) cl_arrive();  // peers write this CTA's receive rows only after it started
  const int ww0 = 5 + p.XL;
  if (warp >= ww0) {  // W producers: never wait on the predecessor
    for (int i = warp - ww0; i < nst; i += p.NW) {
      const int s = i % p.NWS, b0 = kb0 + i * KB, nb = min(KB, kb1 - b0);
      if (i >= p.NWS) mb_wait_bo(sa(&wempty[s]), ((i / p.NWS) - 1) & 1, p.sleep_ns);
      const uint32_t dst = uni(sa(sm + p.w_off + (size_t)s * WB)), bar = uni(sa(&wfull[s]));
      if (p.packed)  // the tile image: one contiguous bulk copy per stage
        bulk_e(dst, wp + ((size_t)rt * p.KBT + b0) * WB1, nb * WB1, bar);
      else
        for (int j = 0; j < nb; ++j)
          tma2d_e(dst + j * WB1, &wmap, (b0 + j) * 64, rt * 128, bar, j == 0 ? nb * WB1 : 0u);
    }
  } else if (warp >= 5) {  // X producers after the dependency wait: stage i -> warp 5 + i % XL
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int i = warp - 5; i < nst; i += p.XL) {
      const int s = i % p.NXS, b0 = kb0 + i * KB, nb = min(KB, kb1 - b0);
      if (i >= p.NXS) mb_wait(sa(&xempty[s]), ((i / p.NXS) - 1) & 1);
      const uint32_t dst = uni(sa(sm + p.x_off + (size_t)s * XB)), bar = uni(sa(&xfull[s]));
      for (int j = 0; j < nb; ++j)
        tma3d_e(dst + j * XB1, &xmap, 0, 0, b0 + j, bar, j == 0 ? nb * XB1 : 0u);
    }
  } else if (warp == 4) {  // TMEM alloc + MMA issue
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(sa(&tbase)), "r"(p.tcols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncwarp();
    asm volatile("bar.arrive 1, 160;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t tm = uni(*(volatile uint32_t*)&tbase);
    for (int i = 0; i < nst; ++i) {
      const int sw = i % p.NWS, sx = i % p.NXS;
      mb_wait(sa(&wfull[sw]), (i / p.NWS) & 1);
      mb_wait_bo(sa(&xfull[sx]), (i / p.NXS) & 1, i == 0 ? p.sleep_ns : 0u);
      asm volatile("tcgen05.fence::after_thread_sync;");
      // warp-uniform descriptor bases once per stage, constant offsets per MMA (no R2UR waterfall)
      const uint64_t ad = sdesc(uni(sa(sm + p.w_off + (size_t)sw * WB)));
      const uint64_t bd = sdesc(uni(sa(sm + p.x_off + (size_t)sx * XB)));
      const int nb = min(KB, kb1 - (kb0 + i * KB));
#pragma unroll
      for (int j = 0; j < KB; ++j) {
        if (j < nb) {
#pragma unroll
          for (int k = 0; k < 4; ++k)
            mma_e(tm, ad + (uint64_t)((j * WB1 + k * 32) >> 4), bd + (uint64_t)((j * XB1 + k * 32) >> 4),
                  p.idesc, (i > 0 || j > 0 || k > 0) ? 1u : 0u);
        }
      }
      commit_e(uni(sa(&wempty[sw])));
      commit_e(uni(sa(&xempty[sx])));
    }
    commit_e(uni(sa(&accf)));
  }
  // ---- epilogue: TMEM -> (own rows: local, other rows: the owner's DSMEM) -> rank-ordered sum ----
  float* recv = (float*)sm;  // [C src][RPR][RS], in the low (non-oversized) region
  if (warp < 4) {
    asm volatile("bar.sync 1, 160;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t tm = tbase;
    mb_wait(sa(&accf), 0);
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (threadIdx.x == 0) asm volatile("griddepcontrol.launch_dependents;");
    const int r = warp * 32 + lane;
    const uint32_t ta = tm + ((uint32_t)(warp * 32) << 16);
    if (CC > 1) cl_wait();
    const unsigned d = r / RPR;
    const int jr = r % RPR;
    const uint32_t dst = sa(&recv[((size_t)cx * RPR + jr) * RS]);
    uint32_t rdst;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(dst), "r"(d));
#pragma unroll 1
    for (int c0 = 0; c0 < T; c0 += 32) {
      float v[32];
#pragma unroll
      for (int c = 0; c < 32; c += 8) ld8nw(ta + c0 + c, v + c);
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
      for (int q = 0; q < 32; q += 4)
        asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};"
                     :: "r"(rdst + (c0 + q) * 4), "f"(v[q]), "f"(v[q + 1]), "f"(v[q + 2]), "f"(v[q + 3])
                     : "memory");
    }
  } else if (CC > 1) {
    cl_wait();
  }
  cluster_sync();
  if (warp < 4) {
    constexpr int G = T / 4;
    for (int e = threadIdx.x; e < RPR * G; e += 128) {
      const int jr = e % RPR, g = e / RPR;
      const int lr = (int)cx * RPR + jr, row = rt * 128 + lr;
      if (lr >= 128 || row >= p.N) continue;
      float4 a = *(const float4*)&recv[(size_t)jr * RS + 4 * g];
#pragma unroll
      for (int src = 1; src < CC; ++src) {
        const float4 b = *(const float4*)&recv[((size_t)src * RPR + jr) * RS + 4 * g];
        a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
      }
      y[(size_t)(4 * g) * p.N + row] = __float2bfloat16_rn(a.x);
      y[(size_t)(4 * g + 1) * p.N + row] = __float2bfloat16_rn(a.y);
      y[(size_t)(4 * g + 2) * p.N + row] = __float2bfloat16_rn(a.z);
      y[(size_t)(4 * g + 3) * p.N + row] = __float2bfloat16_rn(a.w);
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (warp == 4)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tbase), "r"(p.tcols)
                 : "memory");
}

namespace {
typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                             const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                             CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeFn encode_fn() {
  static EncodeFn f = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess && p,
                "wstat_gemm: cuTensorMapEncodeTiled not found");
    return (EncodeFn)p;
  }();
  return f;
}
typedef void (*KF)(CUtensorMap, CUtensorMap, const uint8_t*, __nv_bfloat16*, P);
KF pick(int T, int C, int KB) {
#define C_(t, kb) if (T == t && KB == kb) return C == 1 ? wstat_k<t, 1, kb> : C == 2 ? wstat_k<t, 2, kb> : \
    C == 4 ? wstat_k<t, 4, kb> : C == 6 ? wstat_k<t, 6, kb> : nullptr;
  C_(32, 1) C_(32, 2) C_(32, 4) C_(64, 1) C_(64, 2) C_(64, 4) C_(96, 1) C_(96, 2) C_(128, 1) C_(128, 2)
#undef C_
  return nullptr;
}
int optin() {  // the oversized shared-memory mode's per-CTA limit, minus the static barriers
  int dev, v;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&v, cudaDevAttrOversizedSharedMemoryPerBlock, dev);
  return v - 1024;
}
}  // namespace

int64_t wstat_smem_optin() { return optin(); }

// Bench host standing in for the SP all-gather / reduce: 128 threads, ~2 KB static smem, triggers
// launch_dependents at entry, then spins (dependent FMAs) after its own dependency wait. Carve-out 100
// so a dependent GEMM with a large shared-memory ring can be co-resident (hw29b).
__global__ void __launch_bounds__(128) wstat_host_k(int iters, float* sink) {
  __shared__ float pad[512];
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  float v = threadIdx.x * 1e-7f;
  for (int i = 0; i < iters; ++i) v = v * 0.999999f + 1e-9f;
  pad[threadIdx.x] = v;
  __syncthreads();
  if (pad[(threadIdx.x + 1) & 127] == -1.0f) sink[blockIdx.x] = v;
}

void wstat_host(int64_t iters, int64_t ctas, torch::Tensor& sink) {
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(wstat_host_k, cudaFuncAttributePreferredSharedMemoryCarveout, 100) ==
                    cudaSuccess, "wstat_host: carve-out");
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)ctas); cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = 0;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, wstat_host_k, (int)iters, sink.data_ptr<float>()) == cudaSuccess,
              "wstat_host: launch");
}

void wstat_gemm(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& w, int64_t n, int64_t C,
                int64_t NWS, int64_t NXS, int64_t NW, int64_t XL, int64_t KB, int64_t sleep_ns) {
  // w: [N, K] row-major (2D TMA boxes) or the tile image [ceil(N / 128), K / 64, 128 x 64] of
  // prewait_gemm.pack_weight (one bulk copy per stage); n: the output features.
  const int T = (int)x.size(0), K = (int)x.size(1);
  const bool packed = w.dim() != 2;
  const int N = packed ? (int)n : (int)w.size(0);
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 &&
                  (packed ? w.is_contiguous() && w.numel() == (int64_t)((N + 127) / 128) * 128 * K
                          : w.dim() == 2 && w.stride(1) == 1 && w.size(1) == K && (w.stride(0) * 2) % 16 == 0),
              "wstat_gemm: x [T, K] / w [N, K] or its tile image, bf16");
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.size(0) == T && out.size(1) == N,
              "wstat_gemm: out [T, N] bf16");
  TORCH_CHECK(K % 64 == 0 && K / 64 >= C * KB && NWS >= 2 && NWS <= 16 && NXS >= 2 && NXS <= 8 && NW >= 1 &&
                  XL >= 1 && 5 + NW + XL <= 16 && KB >= 1 && KB <= 4,
              "wstat_gemm: config");
  KF kf = pick(T, (int)C, (int)KB);
  TORCH_CHECK(kf, "wstat_gemm: no kernel for T=", T, " C=", C);
  P p = {};
  p.N = N; p.K = K; p.T = T; p.C = (int)C; p.NWS = (int)NWS; p.NXS = (int)NXS; p.NW = (int)NW;
  p.XL = (int)XL; p.KB = (int)KB; p.packed = packed; p.sleep_ns = (int)sleep_ns;
  p.RT = (N + 127) / 128; p.KBT = K / 64;
  p.idesc = (1u << 4) | (1u << 7) | (1u << 10) | ((unsigned)(T >> 3) << 17) | ((unsigned)(128 >> 4) << 24);
  unsigned tc = 32; while (tc < (unsigned)T) tc <<= 1; p.tcols = tc;
  const int rpr = (128 + (int)C - 1) / (int)C;
  p.x_off = (unsigned)((((size_t)C * rpr * (T + 4) * 4) + 1023) & ~(size_t)1023);
  p.w_off = p.x_off + (unsigned)NXS * (unsigned)KB * (unsigned)T * 128u;
  const size_t smem = (size_t)p.w_off + (size_t)NWS * (size_t)KB * 16384u + 1024;
  const int oi = optin();
  TORCH_CHECK(smem <= (size_t)oi, "wstat_gemm: smem ", smem, " > ", oi);
  static thread_local KF done[16] = {};
  static thread_local size_t done_smem[16] = {};
  bool seen = false;
  for (int i = 0; i < 16; ++i) seen |= done[i] == kf && done_smem[i] >= smem;
  if (!seen) {
    // Above the 227 KiB opt-in the oversized mode (function + launch attribute) is the opt-in; the
    // max-dynamic attribute only takes sizes up to the regular limit (hwdb hw1).
    int dev, regular;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&regular, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    const cudaError_t e = smem + 1024 > (size_t)regular
        ? cudaFuncSetAttribute(kf, cudaFuncAttributeSharedMemoryMode, cudaSharedMemoryModeAllowOversizedSharedMemory)
        : cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    TORCH_CHECK(e == cudaSuccess, "wstat_gemm: smem attribute (", smem, " B): ", cudaGetErrorString(e));
    for (int i = 0; i < 16; ++i) if (!done[i] || done[i] == kf) { done[i] = kf; done_smem[i] = smem; break; }
  }
  CUtensorMap xm, wm = {};
  {
    cuuint64_t dims[3] = {64, (cuuint64_t)T, (cuuint64_t)K / 64};
    cuuint64_t strides[2] = {(cuuint64_t)K * 2, 128};
    cuuint32_t box[3] = {64, (cuuint32_t)T, 1}, es[3] = {1, 1, 1};
    TORCH_CHECK(encode_fn()(&xm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)x.data_ptr(), dims, strides, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS,
                "wstat_gemm: x tensor map");
  }
  if (!packed) {
    cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)N};
    cuuint64_t strides[1] = {(cuuint64_t)w.stride(0) * 2};
    cuuint32_t box[2] = {64, 128}, es[2] = {1, 1};
    TORCH_CHECK(encode_fn()(&wm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)w.data_ptr(), dims, strides, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS,
                "wstat_gemm: w tensor map");
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)C, p.RT);
  cfg.blockDim = dim3((5 + (unsigned)XL + (unsigned)NW) * 32);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[3];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = (unsigned)C; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = 1;
  at[2].id = cudaLaunchAttributeSharedMemoryMode;
  at[2].val.sharedMemoryMode = cudaSharedMemoryModeAllowOversizedSharedMemory;
  cfg.attrs = at; cfg.numAttrs = 3;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kf, xm, wm, (const uint8_t*)w.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
                                 p) == cudaSuccess,
              "wstat_gemm: launch");
}
"""

_mod = None
_lock = threading.Lock()


def _module():
    global _mod
    if _mod is not None:
        return _mod
    with _lock:
        if _mod is None:
            from torch.utils.cpp_extension import load_inline

            import vllm.envs as envs

            h = hashlib.sha256((_CUDA_SRC + _CPP_SRC).encode())
            h.update(f"{torch.__version__} {torch.version.cuda}".encode())
            tag = h.hexdigest()[:16]
            build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_wstat", tag)
            os.makedirs(build, exist_ok=True)
            _mod = load_inline(
                name=f"k3_wstat_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["wstat_gemm", "wstat_smem_optin", "wstat_host"],
                extra_cuda_cflags=[
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-gencode",
                    "arch=compute_107a,code=sm_107a",
                ],
                build_directory=build,
                verbose=False,
            )
    return _mod


def gemm(out: torch.Tensor, x: torch.Tensor, w: torch.Tensor, cfg) -> torch.Tensor:
    """``out[T, N] = x @ w^T`` on config ``cfg = (C, NWS, NXS, NW, XL, KB)``: cluster K-split,
    weight / activation ring slots, weight / activation producer warps, K blocks per stage."""
    c, nws, nxs, nw, xl, kb, *rest = cfg
    _module().wstat_gemm(out, x, w, out.shape[1], c, nws, nxs, nw, xl, kb, rest[0] if rest else 0)
    return out
