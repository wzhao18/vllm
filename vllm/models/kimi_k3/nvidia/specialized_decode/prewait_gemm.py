# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-streaming decode GEMM that issues its weight loads BEFORE the PDL wait.

``out[T, N] = x[T, K] @ W[N, K]^T`` (BF16 in, FP32 accumulation, BF16 or FP32 out) for the
skinny decode shapes (T = 32..96 tokens). Design (GEMM-EX2):

* swap-AB tcgen05 ``kind::f16``: weights are the M = 128 operand, tokens the N = T operand,
  the accumulator lives in TMEM;
* the weight is re-packed once at load time into the exact 128-byte-swizzled shared-memory
  image of its [128 rows x 64 K] tiles, so one pipeline stage is ONE contiguous 1D bulk copy
  (16-32 KiB) and several producer warps keep several copies in flight;
* weight producer warps never execute ``griddepcontrol.wait``: the weights stream while the
  predecessor kernel drains; only the activation producer (3D TMA box per stage) waits;
* split-K over the ``C`` CTAs of a cluster: CTA r owns output rows [r*128/C, (r+1)*128/C) of its
  row tile, the peers push their FP32 partials into its shared memory (DSMEM) and the owner sums
  them in rank order (deterministic; one rounding at the end, as stock's FP32 GEMM).
"""

import hashlib
import os
import threading

import torch

_CPP_SRC = r"""
#include <torch/extension.h>
void prewait_gemm(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& wp, int64_t n, int64_t C, int64_t KB,
                  int64_t NS, int64_t NW, int64_t XL, int64_t pace, int64_t cr, int64_t tile);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
struct P {
  int N, K, T, C, KB, NS, NW, RT, nst, pdl, xlanes, out32;
  int TL;  // live tokens (<= T): the TMA zero-fills the tile's rows past them and the epilogue stores none
  unsigned WB, XB, SB, recv_off, idesc, nm, tcols;
  long long pace;  // clocks between a weight warp's pre-wait stages (0: issue them at once)
};

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint32_t b, unsigned c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(c)); }
__device__ __forceinline__ void mb_expect(uint32_t b, unsigned tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(tx) : "memory");
}
__device__ __forceinline__ void mb_wait(uint32_t b, unsigned ph) {
  asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W;}" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, unsigned n, uint32_t b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(n), "r"(b) : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* m, int c0, int c1, uint32_t b) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(m), "r"(c0), "r"(c1), "r"(b) : "memory");
}
__device__ __forceinline__ void tma3d(uint32_t dst, const CUtensorMap* m, int c0, int c1, int c2, uint32_t b) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               :: "r"(dst), "l"(m), "r"(c0), "r"(c1), "r"(c2), "r"(b) : "memory");
}
__device__ __forceinline__ uint64_t sdesc(uint32_t saddr) {  // K-major SW128, SBO 1024 B, sm100 version bit
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | ((uint64_t)1 << 46) | ((uint64_t)2 << 61);
}
__device__ __forceinline__ void mma_bf16(uint32_t d, uint64_t a, uint64_t b, unsigned id, unsigned acc) {
  asm volatile("{.reg .pred Q; setp.ne.b32 Q, %4, 0; tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, Q;}"
               :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
__device__ __forceinline__ void commit(uint32_t mb) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mb) : "memory");
}
__device__ __forceinline__ void ld8(uint32_t ta, float* v) {
  uint32_t r[8];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(ta));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = __uint_as_float(r[i]);
}
__device__ __forceinline__ void bulk_e(uint32_t dst, const void* src, unsigned n, uint32_t b) {
  asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff; @P mbarrier.arrive.expect_tx.shared::cta.b64 _, [%3], %2; "
               "@P cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];}"
               :: "r"(dst), "l"(src), "r"(n), "r"(b) : "memory");
}
__device__ __forceinline__ void tma3d_e(uint32_t dst, const CUtensorMap* m, int c0, int c1, int c2, uint32_t b, unsigned tx) {
  asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.and.b32 Q, %6, 0, P; @Q mbarrier.arrive.expect_tx.shared::cta.b64 _, [%5], %6; "
               "@P cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];}"
               :: "r"(dst), "l"(m), "r"(c0), "r"(c1), "r"(c2), "r"(b), "r"(tx) : "memory");
}
__device__ __forceinline__ void mma_e(uint32_t d, uint64_t a, uint64_t b, unsigned id, unsigned acc) {
  asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; @P tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, Q;}"
               :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
__device__ __forceinline__ void commit_e(uint32_t mb) {
  asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff; @P tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];}" :: "r"(mb) : "memory");
}
__device__ __forceinline__ void ld8nw(uint32_t ta, float* v) {
  uint32_t r[8];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(ta));
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = __uint_as_float(r[i]);
}
__device__ __forceinline__ unsigned crank() { unsigned r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
}
__device__ __forceinline__ void st_remote(uint32_t local, unsigned rank, float v) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(local), "r"(rank));
  asm volatile("st.shared::cluster.f32 [%0], %1;" :: "r"(r), "f"(v) : "memory");
}

__device__ __forceinline__ void stout(void* y, size_t i, float v, int o32) {
  if (o32) ((float*)y)[i] = v; else ((__nv_bfloat16*)y)[i] = __float2bfloat16_rn(v);
}
__device__ __forceinline__ uint32_t uni(uint32_t v) { return __shfl_sync(0xffffffffu, v, 0); }
__device__ __forceinline__ void st_remote_v4(uint32_t local, unsigned rank, float a, float b, float c, float d) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(local), "r"(rank));
  asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};" :: "r"(r), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.acquire;" ::: "memory"); }
__device__ __forceinline__ bool mb_done(uint32_t b) {  // phase 0 of ``b`` completed
  uint32_t r;
  asm volatile("{.reg .pred P; mbarrier.test_wait.parity.shared::cta.b64 P, [%1], 0; selp.u32 %0, 1, 0, P;}" : "=r"(r) : "r"(b) : "memory");
  return r != 0;
}
__device__ __forceinline__ long long clk() { long long t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory"); return t; }

// warps 0-3 epilogue (TMEM lanes 32w..), warp 4 MMA (+TMEM alloc), warps 5..5+XL-1 X producers, then NW W producers.
// Template: T tokens (N of the MMA, split into NM = ceil(T/256) MMAs), KB 64-wide K blocks per stage, CC split-K CTAs
// per cluster, CR co-resident form: <= 256 threads at <= 64 registers (launch bound), so the CTA fits beside the
// latency-bound predecessor's CTAs (SpReduceAttnRes: 128 threads x 76 registers, up to 4 per SM) and its weight
// stages land during the predecessor instead of after it.
template <int T, int KB, int CC, int CR>
__global__ void __launch_bounds__(CR ? 256 : 512, CR ? 4 : 1) wsg_k(const __grid_constant__ CUtensorMap xmap,
                                                                     const uint8_t* __restrict__ wp, void* __restrict__ yv,
                                                                     const P p) {
  constexpr int NM = (T + 255) / 256, NR = T / NM;
  constexpr unsigned WB = KB * 16384, XB = KB * T * 128, SB = WB + XB;
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = (uint8_t*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  __shared__ __align__(8) uint64_t full[8], empty[8], accf, dep, rbar;
  __shared__ uint32_t tbase;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned cx = CC > 1 ? crank() : 0;
  const int rt = blockIdx.y;
  constexpr int rpr = (128 + CC - 1) / CC, RS = T + 4;  // owner rows per CTA, padded partial row (floats)
  if (threadIdx.x == 0) {
    for (int s = 0; s < p.NS; ++s) { mb_init(sa(&full[s]), 2); mb_init(sa(&empty[s]), 1); }
    mb_init(sa(&accf), 1);
    mb_init(sa(&dep), 1);
    if (CR && CC > 1) {  // the peers' partials of my rows arrive as bulk copies that complete_tx on rbar
      mb_init(sa(&rbar), 1);
      mb_expect(sa(&rbar), (unsigned)((CC - 1) * rpr * RS * 4));
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (CC > 1) cl_arrive();  // peers' recv buffers are written only after every CTA of the cluster started
  const int kb_per_cta = p.K / 64 / CC;
  const int kb0 = cx * kb_per_cta;
  const int wx0 = 5, ww0 = 5 + p.xlanes;
  if (warp >= ww0) {  // W producers: never wait on the predecessor
    const int wi = warp - ww0;
    const uint8_t* src = wp + ((size_t)rt * (p.K / 64) + kb0) * 16384;
    // Paced pre-wait issue: until the activation warp has passed the PDL wait (``dep``), stage i
    // goes out no earlier than t0 + i * pace, so the stream does not burst into the predecessor.
    const long long t0 = p.pace ? __shfl_sync(0xffffffffu, clk(), 0) : 0;
    bool free_issue = p.pace == 0;
    for (int i = wi; i < p.nst; i += p.NW) {
      const int s = i % p.NS;
      if (i >= p.NS) mb_wait(sa(&empty[s]), ((i / p.NS) - 1) & 1);
      while (!free_issue) {  // warp-uniform decision (lane 0's view): bulk_e elects within the full warp
        const unsigned d = mb_done(sa(&dep)) ? 1u : 0u;
        const unsigned go = __shfl_sync(0xffffffffu, d | (clk() >= t0 + (long long)i * p.pace ? 2u : 0u), 0);
        if (go) { free_issue = go & 1u; break; }
        __nanosleep(64);
      }
      bulk_e(uni(sa(sm + (size_t)s * SB)), src + (size_t)i * WB, WB, uni(sa(&full[s])));
    }
  } else if (warp >= wx0) {  // X producers after the dependency wait: stage i -> warp i % xlanes
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (warp == wx0 && lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(&dep)) : "memory");
    for (int i = warp - wx0; i < p.nst; i += p.xlanes) {
      const int s = i % p.NS;
      if (i >= p.NS) mb_wait(sa(&empty[s]), ((i / p.NS) - 1) & 1);
      const uint32_t xb = uni(sa(sm + (size_t)s * SB + WB));
#pragma unroll
      for (int m = 0; m < NM; ++m) tma3d_e(xb + m * KB * NR * 128, &xmap, 0, m * NR, kb0 + i * KB, uni(sa(&full[s])), m == 0 ? XB : 0);
    }
  } else if (warp == 4) {  // TMEM alloc + MMA issuer (warp-uniform, elected, constant descriptor offsets)
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(sa(&tbase)), "r"(p.tcols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncwarp();
    asm volatile("bar.arrive 1, 160;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t tm = uni(*(volatile uint32_t*)&tbase);
    for (int i = 0; i < p.nst; ++i) {
      const int s = i % p.NS;
      mb_wait(sa(&full[s]), (i / p.NS) & 1);
      asm volatile("tcgen05.fence::after_thread_sync;");
      const uint32_t ab = uni(sa(sm + (size_t)s * SB));
      const uint64_t ad = sdesc(ab), bd = sdesc(ab + WB);
#pragma unroll
      for (int kb = 0; kb < KB; ++kb)
#pragma unroll
        for (int k = 0; k < 4; ++k)
#pragma unroll
          for (int m = 0; m < NM; ++m)
            mma_e(tm + m * NR, ad + (uint64_t)((kb * 16384 + k * 32) >> 4), bd + (uint64_t)(((m * KB + kb) * NR * 128 + k * 32) >> 4),
                  p.idesc, (i > 0 || kb > 0 || k > 0) ? 1u : 0u);
      commit_e(uni(sa(&empty[s])));
    }
    commit_e(uni(sa(&accf)));
  }
  // ---- epilogue ----
  float* recv = (float*)(sm + p.recv_off);  // [C src][rpr][RS] (CR: [C - 1 peers][rpr][RS])
  float* stg = (float*)sm;  // CR: [C dst][rpr][RS] staging of this CTA's partials (aliases the drained ring)
  if (warp < 4) {
    asm volatile("bar.sync 1, 160;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t tm = tbase;
    mb_wait(sa(&accf), 0);
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (p.pdl && threadIdx.x == 0) asm volatile("griddepcontrol.launch_dependents;");
    const int r = warp * 32 + lane;
    const uint32_t ta = tm + ((uint32_t)(warp * 32) << 16);
    if (CC == 1) {
      const int row = rt * 128 + r;
#pragma unroll 4
      for (int c = 0; c < T; c += 8) {
        float v[8]; ld8(ta + c, v);
        if (row < p.N)
#pragma unroll
          for (int j = 0; j < 8; ++j)
            if (c + j < p.TL) stout(yv, (size_t)(c + j) * p.N + row, v[j], p.out32);
      }
    } else if (CR) {
      // Stage the row in local shared memory (this CTA's MMAs are done, the ring is free), then ONE bulk
      // DSMEM copy per warp (32 rows) to the owner of those rows; the owner's rbar counts the bytes.
      const unsigned d = r / rpr; const int jr = r % rpr;
      float* mine = stg + ((size_t)d * rpr + jr) * RS;
#pragma unroll 1
      for (int c0 = 0; c0 < T; c0 += 16) {
        float v[16];
        ld8nw(ta + c0, v); ld8nw(ta + c0 + 8, v + 8);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
        for (int q = 0; q < 16; q += 4) *(float4*)&mine[c0 + q] = make_float4(v[q], v[q + 1], v[q + 2], v[q + 3]);
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncwarp();
      cl_wait();  // every peer started (its rbar holds the expected bytes)
      if (d != cx && lane == 0) {  // rpr >= 32: a warp's rows have one owner
        const int j0 = (warp * 32) % rpr;
        const unsigned idx = cx < d ? cx : cx - 1;  // my slot among the owner's peers (rank order)
        uint32_t dr, br;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(dr) : "r"(sa(recv + ((size_t)idx * rpr + j0) * RS)), "r"(d));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(br) : "r"(sa(&rbar)), "r"(d));
        asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(dr), "r"(sa(stg + ((size_t)d * rpr + j0) * RS)), "r"((unsigned)(32 * RS * 4)), "r"(br) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      }
    } else {
      cl_wait();
      const unsigned d = r / rpr; const int jr = r % rpr;
      const uint32_t dst = sa(&recv[((size_t)cx * rpr + jr) * RS]);
      uint32_t rdst; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(dst), "r"(d));
      const bool own = d == cx;
      float v[T];
#pragma unroll
      for (int c = 0; c < T; c += 8) ld8nw(ta + c, v + c);
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
      for (int q = 0; q < T; q += 4) {
        if (own) *(float4*)&recv[((size_t)cx * rpr + jr) * RS + q] = make_float4(v[q], v[q + 1], v[q + 2], v[q + 3]);
        else asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};" :: "r"(rdst + q * 4), "f"(v[q]), "f"(v[q + 1]), "f"(v[q + 2]), "f"(v[q + 3]) : "memory");
      }
    }
  } else if (CC > 1) {
    cl_wait();
  }
  if (CC > 1) {
    if (!CR) cluster_sync();
    if (warp < 4) {
      constexpr int G = T / 4;
      if (CR) {  // own rows staged by the other epilogue warps; the peers' rows counted on rbar
        asm volatile("bar.sync 2, 128;" ::: "memory");
        mb_wait(sa(&rbar), 0);
      }
#pragma unroll
      for (int e = threadIdx.x; e < rpr * G; e += 128) {
        const int jr = e % rpr, g = e / rpr;
        const int lr = cx * rpr + jr, row = rt * 128 + lr;
        if (lr >= 128) continue;
        float4 a;
#pragma unroll
        for (int src = 0; src < CC; ++src) {  // rank order (deterministic, as the DSMEM-store form)
          const float* part = !CR ? recv + (size_t)src * rpr * RS
                                  : (src == (int)cx ? stg + (size_t)cx * rpr * RS
                                                    : recv + (size_t)(src < (int)cx ? src : src - 1) * rpr * RS);
          const float4 b = *(const float4*)&part[(size_t)jr * RS + 4 * g];
          if (src == 0) a = b;
          else { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
        }
        if (row < p.N) {  // tokens 4 g .. 4 g + 3 (the tile's rows past TL are zero-filled, not stored)
          const int t0 = 4 * g;
          if (t0 < p.TL) stout(yv, (size_t)t0 * p.N + row, a.x, p.out32);
          if (t0 + 1 < p.TL) stout(yv, (size_t)(t0 + 1) * p.N + row, a.y, p.out32);
          if (t0 + 2 < p.TL) stout(yv, (size_t)(t0 + 2) * p.N + row, a.z, p.out32);
          if (t0 + 3 < p.TL) stout(yv, (size_t)(t0 + 3) * p.N + row, a.w, p.out32);
        }
      }
    }
  }
  if (CR && CC > 1 && warp < 4 && lane == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (warp == 4) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tbase), "r"(p.tcols) : "memory");
}


namespace {
typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                             const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                             CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeFn encode_fn() {
  static EncodeFn f = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess && p,
                "prewait_gemm: cuTensorMapEncodeTiled not found");
    return (EncodeFn)p;
  }();
  return f;
}
typedef void (*KF)(CUtensorMap, const uint8_t*, void*, P);
KF pick(int T, int KB, int C, int CR) {
  if (CR) {
#define R_(t) if (T == t && KB == 1 && C == 4) return wsg_k<t, 1, 4, 1>;
    R_(32) R_(64) R_(96)
#undef R_
    return nullptr;
  }
#define C_(t, kb) if (T == t && KB == kb) return C == 1 ? wsg_k<t, kb, 1, 0> : C == 2 ? wsg_k<t, kb, 2, 0> : C == 4 ? wsg_k<t, kb, 4, 0> : nullptr;
  C_(32, 1) C_(32, 2) C_(64, 1) C_(64, 2) C_(96, 1) C_(96, 2)
#undef C_
  return nullptr;
}
// Co-resident form budget: beside up to 4 SpReduceAttnRes CTAs (128 threads x 76 registers, 2 KiB shared memory
// each) an SM has 24K registers and ~215 KiB of shared memory left.
constexpr int kCoresRegs = 64, kCoresThreads = 256;
constexpr size_t kCoresSmem = 215 * 1024;
}  // namespace

// out[T, N] (bf16 or fp32) = x[T, K] (bf16) @ W[N, K]^T with W pre-packed by pack_weight (SW128 tile image).
void prewait_gemm(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& wp, int64_t n, int64_t C, int64_t KB,
                  int64_t NS, int64_t NW, int64_t XL, int64_t pace, int64_t cr, int64_t tile) {
  // ``tile``: the kernel's token tile T (a compiled instance); x holds TL <= T live tokens.
  const int TL = (int)x.size(0), T = (int)tile, K = (int)x.size(1), N = (int)n;
  TORCH_CHECK(TL >= 1 && TL <= T, "prewait_gemm: ", TL, " tokens exceed the tile ", T);
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kBFloat16, "prewait_gemm: x bf16 contiguous");
  TORCH_CHECK(out.is_contiguous() && out.size(0) == TL && out.size(1) == N, "prewait_gemm: out shape");
  const bool o32 = out.scalar_type() == at::kFloat;
  TORCH_CHECK(o32 || out.scalar_type() == at::kBFloat16, "prewait_gemm: out bf16/fp32");
  const int RT = (N + 127) / 128;
  TORCH_CHECK(wp.numel() * wp.element_size() == (int64_t)RT * 128 * K * 2, "prewait_gemm: packed weight size");
  TORCH_CHECK(K % (64 * C * KB) == 0 && NW <= NS && XL <= NS && NS <= 8, "prewait_gemm: config");
  KF kf = pick(T, (int)KB, (int)C, (int)cr);
  TORCH_CHECK(kf, "prewait_gemm: no kernel for T=", T, " KB=", KB, " C=", C, " co-resident=", cr);
  TORCH_CHECK(pace >= 0, "prewait_gemm: pace");
  P p = {};
  p.N = N; p.K = K; p.T = T; p.TL = TL; p.C = (int)C; p.KB = (int)KB; p.NS = (int)NS; p.NW = (int)NW; p.RT = RT;
  p.nst = K / 64 / (int)C / (int)KB; p.pdl = 1; p.xlanes = (int)XL; p.out32 = o32; p.pace = pace;
  p.nm = (T + 255) / 256;
  p.WB = KB * 16384; p.XB = KB * T * 128; p.SB = p.WB + p.XB;
  p.recv_off = NS * p.SB;
  p.idesc = (1u << 4) | (1u << 7) | (1u << 10) | ((unsigned)((T / p.nm) >> 3) << 17) | ((unsigned)(128 >> 4) << 24);
  unsigned tc = 32; while (tc < (unsigned)T) tc <<= 1; p.tcols = tc;
  const size_t rbytes = C > 1 ? (size_t)(cr ? C - 1 : C) * (T + 4) * ((128 + C - 1) / C) * 4 : 0;
  TORCH_CHECK(!cr || (size_t)NS * p.SB >= (size_t)C * (T + 4) * ((128 + C - 1) / C) * 4, "prewait_gemm: staging > ring");
  const size_t smem = (size_t)NS * p.SB + rbytes + 1024;
  int dev; cudaGetDevice(&dev);
  int optin; cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(smem <= (size_t)optin - 1024, "prewait_gemm: smem ", smem);
  const int threads = (5 + (int)XL + (int)NW) * 32;
  if (cr) {  // the co-resident form must fit beside the predecessor's CTAs: raise, never fall back
    static thread_local KF checked[8] = {};
    bool ok = false;
    for (KF f : checked) ok |= f == kf;
    if (!ok) {
      cudaFuncAttributes fa;
      TORCH_CHECK(cudaFuncGetAttributes(&fa, kf) == cudaSuccess, "prewait_gemm: attributes");
      TORCH_CHECK(fa.numRegs <= kCoresRegs, "prewait_gemm: co-resident CTA uses ", fa.numRegs, " registers > ",
                  kCoresRegs);
      for (KF& f : checked) if (!f) { f = kf; break; }
    }
    TORCH_CHECK(threads <= kCoresThreads && smem <= kCoresSmem, "prewait_gemm: co-resident CTA too large: ", threads,
                " threads, ", smem, " B shared memory");
  }
  static thread_local KF done[8] = {};
  bool seen = false;
  for (KF f : done) seen |= f == kf;
  if (!seen) {
    TORCH_CHECK(cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - 1024) == cudaSuccess, "attr");
    for (KF& f : done) if (!f) { f = kf; break; }
  }
  CUtensorMap xm;
  {
    const int rows = T / p.nm;
    // Global extent = the live tokens: the box (the tile's rows) past them is zero-filled and still counts
    // its full bytes toward the stage's transaction, so the pipeline is the full tile's.
    cuuint64_t dims[3] = {64, (cuuint64_t)TL, (cuuint64_t)K / 64}; cuuint64_t strides[2] = {(cuuint64_t)K * 2, 128};
    cuuint32_t box[3] = {64, (cuuint32_t)rows, (cuuint32_t)KB}; cuuint32_t es[3] = {1, 1, 1};
    TORCH_CHECK(encode_fn()(&xm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)x.data_ptr(), dims, strides, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "prewait_gemm: tensor map");
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)C, RT); cfg.blockDim = dim3(threads); cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = (unsigned)C;
  at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = 2;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kf, xm, (const uint8_t*)wp.data_ptr(), out.data_ptr(), p) == cudaSuccess,
              "prewait_gemm: launch");
}
"""

_mod = None
_lock = threading.Lock()

# Per-T launch config (C split-K CTAs per cluster, KB 64-wide K blocks per stage, NS stages,
# NW weight-producer warps, XL activation-producer warps, 0 clocks between the pre-wait
# weight stages of a CTA (0: all at once), CR co-resident form) for the [4480, 7168]
# router|latent concat, chosen in-step on VR200 (D3-32 NVFP4 slice; fastab gemmex2-ab3: the
# standalone-best n32 (4, 1, 8, 4, 4) was +8.7 us/step slower in-step than (4, 2, 5, 4, 2)).
# n64 / n96 run the co-resident form: 8 warps at <= 64 registers and <= 215 KiB of shared
# memory, so the CTAs land beside SpReduceAttnRes (launched with the full shared-memory
# carve-out) and stage NS weight stages before the reduce ends.
CONFIGS = {32: (4, 2, 5, 4, 2, 0, 0), 64: (4, 1, 7, 2, 1, 1440, 1), 96: (4, 1, 6, 2, 1, 1440, 1)}


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
            build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_prewait_gemm", tag)
            os.makedirs(build, exist_ok=True)
            _mod = load_inline(
                name=f"k3_prewait_gemm_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["prewait_gemm"],
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


def tile(n_tokens: int) -> int | None:
    """The token tile (a ``CONFIGS`` key) a call of ``n_tokens`` live tokens runs: the
    smallest one holding them. The tile's rows past the live tokens are zero-filled by
    the activation TMA and never stored, so each live row's FP32 sums (and its one
    rounding) are the tile's, bit for bit."""
    return next((t for t in sorted(CONFIGS) if n_tokens <= t), None)


def supported(n_tokens: int, k: int) -> bool:
    t = tile(n_tokens)
    if t is None:
        return False
    cfg = CONFIGS[t]
    return k % (64 * cfg[0] * cfg[1]) == 0


def pack_weight(w: torch.Tensor) -> torch.Tensor:
    """[N, K] BF16 row-major -> the kernel's tile image [ceil(N/128), K/64, 128, 64] with the
    128-byte swizzle (16-byte chunk c of row r stored at chunk c ^ (r % 8)); rows >= N are zero."""
    n, k = w.shape
    assert w.dtype == torch.bfloat16 and k % 64 == 0
    rt = (n + 127) // 128
    if rt * 128 != n:
        w = torch.cat([w, w.new_zeros(rt * 128 - n, k)])
    t = w.reshape(rt, 128, k // 64, 8, 8).permute(0, 2, 1, 3, 4)  # [rt, kb, r, c, e]
    r = torch.arange(128, device=w.device)[:, None]
    pc = torch.arange(8, device=w.device)[None, :]
    return t[:, :, r, pc ^ (r & 7), :].contiguous()


def gemm(out: torch.Tensor, x: torch.Tensor, packed: torch.Tensor, n: int,
         config: tuple | None = None, tile_tokens: int | None = None) -> torch.Tensor:
    """out[T, n] = x @ W^T for the W ``packed`` came from (``pack_weight``); ``x`` may
    hold fewer tokens than the tile (``tile``) it runs on."""
    t = tile_tokens or tile(x.shape[0])
    if t is None:
        raise ValueError(f"prewait_gemm: no tile for {x.shape[0]} tokens")
    c, kb, ns, nw, xl, pace, cr = config or CONFIGS[t]
    _module().prewait_gemm(out, x, packed, n, c, kb, ns, nw, xl, pace, cr, t)
    return out
