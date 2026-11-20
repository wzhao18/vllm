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
                  int64_t NS, int64_t NW, int64_t XL, int64_t pace, int64_t cr, int64_t nt, int64_t nsx);
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
  unsigned WB, XB, SB, recv_off, idesc, nm, tcols;
  long long pace;  // clocks between a weight warp's pre-wait stages (0: issue them at once)
  int NSX;         // activation ring stages (TMEM-staged form)
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
          for (int j = 0; j < 8; ++j) stout(yv, (size_t)(c + j) * p.N + row, v[j], p.out32);
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
        if (row < p.N) {
          stout(yv, (size_t)(4 * g) * p.N + row, a.x, p.out32);
          stout(yv, (size_t)(4 * g + 1) * p.N + row, a.y, p.out32);
          stout(yv, (size_t)(4 * g + 2) * p.N + row, a.z, p.out32);
          stout(yv, (size_t)(4 * g + 3) * p.N + row, a.w, p.out32);
        }
      }
    }
  }
  if (CR && CC > 1 && warp < 4 && lane == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (warp == 4) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tbase), "r"(p.tcols) : "memory");
}

__device__ __forceinline__ void tcp_e(uint32_t taddr, uint64_t d) {  // smem (descriptor) -> TMEM, 128 lanes x 8 columns
  asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff; @P tcgen05.cp.cta_group::1.128x256b [%0], %1;}" :: "r"(taddr), "l"(d) : "memory");
}
__device__ __forceinline__ void mma_ta_e(uint32_t d, uint32_t a, uint64_t b, unsigned id, unsigned acc) {  // A from TMEM
  asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, Q;}"
               :: "r"(d), "r"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}

// TMEM-staged co-resident form: separate weight (NS x 16 KiB) and activation (NSX x T x 128 B) rings; the first NT
// weight stages are copied from their ring slot into tensor memory (tcgen05.cp) as they land, before the PDL wait, so
// NT + NS weight stages are on chip when the predecessor ends; their MMAs read A from tensor memory (same K order and
// FP32 accumulation as the shared-memory form). Warps: 0-3 epilogue, 4 TMEM + MMA, 5 activations, 6-7 weights.
template <int T, int CC, int NT>
__global__ void __launch_bounds__(256, 4) wst_k(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ wp,
                                                void* __restrict__ yv, const P p) {
  constexpr unsigned WB = 16384, XB = T * 128;
  constexpr int ACOL = T <= 32 ? 32 : T <= 64 ? 64 : 128;  // accumulator columns; A stages follow
  static_assert(T <= 128 && ACOL + 32 * NT <= 512 && CC > 1, "wst_k: TMEM / split");
  constexpr int rpr = (128 + CC - 1) / CC, RS = T + 4;
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = (uint8_t*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  uint8_t* xr = sm + (size_t)p.NS * WB;
  __shared__ __align__(8) uint64_t wfull[16], wempty[16], xfull[16], xempty[16], accf, dep, rbar;
  __shared__ uint32_t tbase;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned cx = crank();
  const int rt = blockIdx.y;
  if (threadIdx.x == 0) {
    for (int s = 0; s < p.NS; ++s) { mb_init(sa(&wfull[s]), 1); mb_init(sa(&wempty[s]), 1); }
    for (int s = 0; s < p.NSX; ++s) { mb_init(sa(&xfull[s]), 1); mb_init(sa(&xempty[s]), 1); }
    mb_init(sa(&accf), 1);
    mb_init(sa(&dep), 1);
    mb_init(sa(&rbar), 1);
    mb_expect(sa(&rbar), (unsigned)((CC - 1) * rpr * RS * 4));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  cl_arrive();
  const int kb0 = cx * (p.K / 64 / CC);
  if (warp >= 5 + p.xlanes) {  // weight producers: never wait on the predecessor (paced until it ends)
    const int wi = warp - 5 - p.xlanes;
    const uint8_t* src = wp + ((size_t)rt * (p.K / 64) + kb0) * WB;
    const long long t0 = p.pace ? __shfl_sync(0xffffffffu, clk(), 0) : 0;
    bool free_issue = p.pace == 0;
    for (int i = wi; i < p.nst; i += p.NW) {
      const int s = i % p.NS;
      if (i >= p.NS) mb_wait(sa(&wempty[s]), ((i / p.NS) - 1) & 1);
      while (!free_issue) {
        const unsigned d = mb_done(sa(&dep)) ? 1u : 0u;
        const unsigned go = __shfl_sync(0xffffffffu, d | (clk() >= t0 + (long long)i * p.pace ? 2u : 0u), 0);
        if (go) { free_issue = go & 1u; break; }
        __nanosleep(64);
      }
      bulk_e(uni(sa(sm + (size_t)s * WB)), src + (size_t)i * WB, WB, uni(sa(&wfull[s])));
    }
  } else if (warp >= 5) {  // activation producers after the dependency wait
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (warp == 5 && lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(&dep)) : "memory");
    for (int i = warp - 5; i < p.nst; i += p.xlanes) {
      const int s = i % p.NSX;
      if (i >= p.NSX) mb_wait(sa(&xempty[s]), ((i / p.NSX) - 1) & 1);
      tma3d_e(uni(sa(xr + (size_t)s * XB)), &xmap, 0, 0, kb0 + i, uni(sa(&xfull[s])), XB);
    }
  } else if (warp == 4) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(sa(&tbase)), "r"(512) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncwarp();
    asm volatile("bar.arrive 1, 160;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t tm = uni(*(volatile uint32_t*)&tbase);
    for (int j = 0; j < NT; ++j) {  // pre-wait: weight stage j -> TMEM columns ACOL + 32 j (frees its ring slot)
      const int s = j % p.NS;
      mb_wait(sa(&wfull[s]), (j / p.NS) & 1);
      asm volatile("tcgen05.fence::after_thread_sync;");
      const uint32_t wb = uni(sa(sm + (size_t)s * WB));
#pragma unroll
      for (int k = 0; k < 4; ++k) tcp_e(tm + ACOL + 32 * j + 8 * k, sdesc(wb + 32 * k));
      commit_e(uni(sa(&wempty[s])));
    }
    for (int i = 0; i < p.nst; ++i) {
      const int sx = i % p.NSX, s = i % p.NS;
      mb_wait(sa(&xfull[sx]), (i / p.NSX) & 1);
      if (i >= NT) mb_wait(sa(&wfull[s]), (i / p.NS) & 1);
      asm volatile("tcgen05.fence::after_thread_sync;");
      const uint64_t bd = sdesc(uni(sa(xr + (size_t)sx * XB)));
      if (i < NT) {
#pragma unroll
        for (int k = 0; k < 4; ++k)
          mma_ta_e(tm, tm + ACOL + 32 * i + 8 * k, bd + (uint64_t)((k * 32) >> 4), p.idesc, (i > 0 || k > 0) ? 1u : 0u);
      } else {
        const uint64_t ad = sdesc(uni(sa(sm + (size_t)s * WB)));
#pragma unroll
        for (int k = 0; k < 4; ++k)
          mma_e(tm, ad + (uint64_t)((k * 32) >> 4), bd + (uint64_t)((k * 32) >> 4), p.idesc, (i > 0 || k > 0) ? 1u : 0u);
      }
      commit_e(uni(sa(&xempty[sx])));
      if (i >= NT) commit_e(uni(sa(&wempty[s])));
    }
    commit_e(uni(sa(&accf)));
  }
  // ---- epilogue: stage partial rows, one bulk DSMEM copy per warp to the rows' owner, rank-ordered sum ----
  float* recv = (float*)(sm + p.recv_off);  // [C - 1 peers][rpr][RS]
  float* stg = (float*)sm;                  // [C dst][rpr][RS] (aliases the drained rings)
  if (warp < 4) {
    asm volatile("bar.sync 1, 160;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t tm = tbase;
    mb_wait(sa(&accf), 0);
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (p.pdl && threadIdx.x == 0) asm volatile("griddepcontrol.launch_dependents;");
    const int r = warp * 32 + lane;
    const uint32_t ta = tm + ((uint32_t)(warp * 32) << 16);
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
    cl_wait();
    if (d != cx && lane == 0) {
      const int j0 = (warp * 32) % rpr;
      const unsigned idx = cx < d ? cx : cx - 1;
      uint32_t dr, br;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(dr) : "r"(sa(recv + ((size_t)idx * rpr + j0) * RS)), "r"(d));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(br) : "r"(sa(&rbar)), "r"(d));
      asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(dr), "r"(sa(stg + ((size_t)d * rpr + j0) * RS)), "r"((unsigned)(32 * RS * 4)), "r"(br) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
    asm volatile("bar.sync 2, 128;" ::: "memory");
    mb_wait(sa(&rbar), 0);
    constexpr int G = T / 4;
#pragma unroll
    for (int e = threadIdx.x; e < rpr * G; e += 128) {
      const int jr2 = e % rpr, g = e / rpr;
      const int lr = cx * rpr + jr2, row = rt * 128 + lr;
      if (lr >= 128) continue;
      float4 a;
#pragma unroll
      for (int src = 0; src < CC; ++src) {
        const float* part = src == (int)cx ? stg + (size_t)cx * rpr * RS : recv + (size_t)(src < (int)cx ? src : src - 1) * rpr * RS;
        const float4 b = *(const float4*)&part[(size_t)jr2 * RS + 4 * g];
        if (src == 0) a = b;
        else { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
      }
      if (row < p.N) {
        stout(yv, (size_t)(4 * g) * p.N + row, a.x, p.out32);
        stout(yv, (size_t)(4 * g + 1) * p.N + row, a.y, p.out32);
        stout(yv, (size_t)(4 * g + 2) * p.N + row, a.z, p.out32);
        stout(yv, (size_t)(4 * g + 3) * p.N + row, a.w, p.out32);
      }
    }
    if (lane == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
  } else {
    cl_wait();
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (warp == 4) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tbase), "r"(512) : "memory");
}

// Full pre-wait weight residency: split-K over a 6-CTA cluster (35 x 6 = 210 CTAs, one per SM beside the
// predecessor); a CTA's whole weight slice (18-19 K-blocks) is on chip before the PDL wait: its last NT K-blocks in
// tensor memory (copied through pool slots with tcgen05.cp), its first nsw K-blocks in pool slots. After the wait the
// pool becomes the activation ring: each MMA frees its activation slot and (first nsw K-blocks) its weight slot, so the
// activation loads in flight grow from P - nsw to P. The (slot, use) schedule of every load is computed once per CTA.
// MMAs run in K order (A from the pool, then from TMEM), the 6 partials are summed in rank order by their owner.
template <int T, int NT>
__global__ void __launch_bounds__(256, 4) wsf_k(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ wp,
                                                void* __restrict__ yv, const P p) {
  constexpr int CC = 6, MAXP = 16, MAXS = 24;
  constexpr unsigned WB = 16384, XB = T * 128;
  constexpr int ACOL = T <= 32 ? 32 : T <= 64 ? 64 : 128;
  static_assert(T <= 128 && ACOL + 32 * NT <= 512 && XB <= WB, "wsf_k: TMEM / slot");
  constexpr int RPR = (128 + CC - 1) / CC, RS = T + 4;  // owner rows (the last owner has 128 - 5 * RPR)
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = (uint8_t*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  __shared__ __align__(8) uint64_t full[MAXP], empty[MAXP], accf, dep, rbar;
  __shared__ uint32_t tbase;
  __shared__ uint8_t tm_slot[MAXS], tm_use[MAXS], rs_use[MAXS], x_slot[MAXS], x_use[MAXS];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int cx = (int)crank();
  const int rt = blockIdx.y;
  const int kbt = p.K / 64, kbase = kbt / CC, kext = kbt % CC;
  const int nst = kbase + (cx < kext ? 1 : 0), kb0 = cx * kbase + min(cx, kext), nsw = nst - NT;
  const int np = p.NS;
  const int rows_me = min(RPR, 128 - cx * RPR);
  if (threadIdx.x == 0) {
    for (int s = 0; s < np; ++s) { mb_init(sa(&full[s]), 1); mb_init(sa(&empty[s]), 1); }
    mb_init(sa(&accf), 1);
    mb_init(sa(&dep), 1);
    mb_init(sa(&rbar), 1);
    mb_expect(sa(&rbar), (unsigned)((CC - 1) * rows_me * RS * 4));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    uint8_t use[MAXP], q[2 * MAXS + MAXP];
    for (int s = 0; s < np; ++s) use[s] = 0;
    for (int j = 0; j < NT; ++j) { const int s = j % np; tm_slot[j] = (uint8_t)s; tm_use[j] = use[s]++; }
    for (int m = 0; m < nsw; ++m) rs_use[m] = use[m]++;
    int qh = 0, qt = 0;
    for (int s = nsw; s < np; ++s) q[qt++] = (uint8_t)s;
    for (int k = 0; k < nst; ++k) {  // activation k takes the slot freed earliest (by the MMA that frees it)
      const int s = q[qh++];
      x_slot[k] = (uint8_t)s; x_use[k] = use[s]++;
      q[qt++] = (uint8_t)s;
      if (k < nsw) q[qt++] = (uint8_t)k;
    }
  }
  __syncthreads();
  cl_arrive();
  const uint8_t* wsrc = wp + ((size_t)rt * kbt + kb0) * WB;
  if (warp >= 6) {  // warp 6: TMEM-bound weight K-blocks nsw.., warp 7: pool-resident K-blocks 0..nsw-1
    const bool tmw = warp == 6;
    const int n = tmw ? NT : nsw;
    const long long t0 = p.pace ? __shfl_sync(0xffffffffu, clk(), 0) : 0;
    bool free_issue = p.pace == 0;
    for (int i = 0; i < n; ++i) {
      const int s = tmw ? tm_slot[i] : i, u = tmw ? tm_use[i] : rs_use[i];
      if (u >= 1) mb_wait(sa(&empty[s]), (u - 1) & 1);
      while (!free_issue) {
        const unsigned d = mb_done(sa(&dep)) ? 1u : 0u;
        const unsigned go = __shfl_sync(0xffffffffu, d | (clk() >= t0 + (long long)i * p.pace ? 2u : 0u), 0);
        if (go) { free_issue = go & 1u; break; }
        __nanosleep(64);
      }
      bulk_e(uni(sa(sm + (size_t)s * WB)), wsrc + (size_t)(tmw ? nsw + i : i) * WB, WB, uni(sa(&full[s])));
    }
  } else if (warp == 5) {  // activations after the dependency wait, into the slot schedule
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(&dep)) : "memory");
    for (int k = 0; k < nst; ++k) {
      const int s = x_slot[k], u = x_use[k];
      if (u >= 1) mb_wait(sa(&empty[s]), (u - 1) & 1);
      tma3d_e(uni(sa(sm + (size_t)s * WB)), &xmap, 0, 0, kb0 + k, uni(sa(&full[s])), XB);
    }
  } else if (warp == 4) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(sa(&tbase)), "r"(512) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncwarp();
    asm volatile("bar.arrive 1, 160;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t tm = uni(*(volatile uint32_t*)&tbase);
    for (int j = 0; j < NT; ++j) {  // pre-wait: K-block nsw + j -> TMEM columns ACOL + 32 j
      const int s = tm_slot[j];
      mb_wait(sa(&full[s]), tm_use[j] & 1);
      asm volatile("tcgen05.fence::after_thread_sync;");
      const uint32_t wb = uni(sa(sm + (size_t)s * WB));
#pragma unroll
      for (int k = 0; k < 4; ++k) tcp_e(tm + ACOL + 32 * j + 8 * k, sdesc(wb + 32 * k));
      commit_e(uni(sa(&empty[s])));
    }
    for (int k = 0; k < nst; ++k) {
      const int sx = x_slot[k];
      mb_wait(sa(&full[sx]), x_use[k] & 1);
      if (k < nsw) mb_wait(sa(&full[k]), rs_use[k] & 1);
      asm volatile("tcgen05.fence::after_thread_sync;");
      const uint64_t bd = sdesc(uni(sa(sm + (size_t)sx * WB)));
      if (k >= nsw) {
#pragma unroll
        for (int q = 0; q < 4; ++q)
          mma_ta_e(tm, tm + ACOL + 32 * (k - nsw) + 8 * q, bd + (uint64_t)((q * 32) >> 4), p.idesc, (k > 0 || q > 0) ? 1u : 0u);
      } else {
        const uint64_t ad = sdesc(uni(sa(sm + (size_t)k * WB)));
#pragma unroll
        for (int q = 0; q < 4; ++q)
          mma_e(tm, ad + (uint64_t)((q * 32) >> 4), bd + (uint64_t)((q * 32) >> 4), p.idesc, (k > 0 || q > 0) ? 1u : 0u);
      }
      commit_e(uni(sa(&empty[sx])));
      if (k < nsw) commit_e(uni(sa(&empty[k])));
    }
    commit_e(uni(sa(&accf)));
  }
  // ---- epilogue: stage rows, bulk DSMEM copy per (warp, owner) segment, rank-ordered sum by the owner ----
  float* recv = (float*)(sm + p.recv_off);  // [C - 1 peers][RPR][RS]
  float* stg = (float*)sm;                  // [C owners][RPR][RS] (aliases the drained pool)
  if (warp < 4) {
    asm volatile("bar.sync 1, 160;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t tm = tbase;
    mb_wait(sa(&accf), 0);
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (p.pdl && threadIdx.x == 0) asm volatile("griddepcontrol.launch_dependents;");
    const int r = warp * 32 + lane;
    const uint32_t ta = tm + ((uint32_t)(warp * 32) << 16);
    float* mine = stg + ((size_t)(r / RPR) * RPR + r % RPR) * RS;
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
    cl_wait();
    if (lane == 0) {
      for (int o = (warp * 32) / RPR; o <= (warp * 32 + 31) / RPR && o < CC; ++o) {
        if (o == cx) continue;
        const int r0 = max(warp * 32, o * RPR), r1 = min(warp * 32 + 32, min(o * RPR + RPR, 128));
        if (r1 <= r0) continue;
        const int jr0 = r0 - o * RPR;
        const unsigned idx = cx < o ? cx : cx - 1;
        uint32_t dr, br;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(dr) : "r"(sa(recv + ((size_t)idx * RPR + jr0) * RS)), "r"(o));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(br) : "r"(sa(&rbar)), "r"(o));
        asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(dr), "r"(sa(stg + ((size_t)o * RPR + jr0) * RS)), "r"((unsigned)((r1 - r0) * RS * 4)), "r"(br) : "memory");
      }
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
    asm volatile("bar.sync 2, 128;" ::: "memory");
    mb_wait(sa(&rbar), 0);
    constexpr int G = T / 4;
    for (int e = threadIdx.x; e < rows_me * G; e += 128) {
      const int jr = e % rows_me, g = e / rows_me;
      const int row = rt * 128 + cx * RPR + jr;
      float4 a;
#pragma unroll
      for (int src = 0; src < CC; ++src) {
        const float* part = src == cx ? stg + (size_t)cx * RPR * RS : recv + (size_t)(src < cx ? src : src - 1) * RPR * RS;
        const float4 b = *(const float4*)&part[(size_t)jr * RS + 4 * g];
        if (src == 0) a = b;
        else { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
      }
      if (row < p.N) {
        stout(yv, (size_t)(4 * g) * p.N + row, a.x, p.out32);
        stout(yv, (size_t)(4 * g + 1) * p.N + row, a.y, p.out32);
        stout(yv, (size_t)(4 * g + 2) * p.N + row, a.z, p.out32);
        stout(yv, (size_t)(4 * g + 3) * p.N + row, a.w, p.out32);
      }
    }
    if (lane == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
  } else {
    cl_wait();
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (warp == 4) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tbase), "r"(512) : "memory");
}

__device__ __forceinline__ uint32_t mapa_u32(uint32_t a, unsigned rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r;
}
__device__ __forceinline__ void tma3_2e(uint32_t dst, const CUtensorMap* m, int c0, int c1, int c2, uint32_t mbar_cl,
                                        uint32_t mbar_local, unsigned tx) {  // completes on the pair leader's barrier
  asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.and.b32 Q, %7, 0, P; "
               "@Q mbarrier.arrive.expect_tx.shared::cta.b64 _, [%6], %7; "
               "@P cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];}"
               :: "r"(dst), "l"(m), "r"(c0), "r"(c1), "r"(c2), "r"(mbar_cl), "r"(mbar_local), "r"(tx) : "memory");
}
__device__ __forceinline__ void mma2_e(uint32_t d, uint64_t a, uint64_t b, unsigned id, unsigned acc) {
  asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; @P tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, Q;}"
               :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
__device__ __forceinline__ void commit2_e(uint32_t mb, unsigned short mask) {
  asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff; @P tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;}"
               :: "r"(mb), "h"(mask) : "memory");
}

// cta_group::2 pair form of the co-resident front: cluster 8 = 4 K-split CTAs x 2 pair members. A pair computes 256
// weight rows (each CTA its own 128) x T tokens with M256 MMAs issued by the leader; each CTA loads HALF of the tokens'
// activations (activation bytes per CTA halve vs the M128 form). A CTA's [128 rows x T] split-K partial is exchanged
// with the 3 CTAs of the same pair rank (bulk DSMEM, owner sums in K-split order). Warps: 0-3 epilogue, 4 TMEM + MMA
// (leader), 5 activations, 6-7 weights.
template <int T>
__global__ void __launch_bounds__(256, 4) wsp_k(const __grid_constant__ CUtensorMap xmap, const __grid_constant__ CUtensorMap wmap,
                                                void* __restrict__ yv, const P p) {
  constexpr int KS = 4, rpr = 32, RS = T + 4, TH = T / 2;
  constexpr unsigned WB = 16384, XB = TH * 128, SB = WB + XB;
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = (uint8_t*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  __shared__ __align__(8) uint64_t full[16], empty[16], accf, dep, rbar;
  __shared__ uint32_t tbase;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned rank = crank(), pr = rank & 1u, ks = rank >> 1, lead = rank & ~1u;
  const bool leader = pr == 0;
  const unsigned short pmask = (unsigned short)(3u << lead);
  const int rt = 2 * blockIdx.y + (int)pr;
  const int kbt = p.K / 64, nst = kbt / KS, kb0 = (int)ks * nst;
  if (threadIdx.x == 0) {
    for (int s = 0; s < p.NS; ++s) { mb_init(sa(&full[s]), 2); mb_init(sa(&empty[s]), 1); }
    mb_init(sa(&accf), 1);
    mb_init(sa(&dep), 1);
    mb_init(sa(&rbar), 1);
    mb_expect(sa(&rbar), (unsigned)((KS - 1) * rpr * RS * 4));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  cluster_sync();  // the pair's loads complete on the leader's barriers; peers' bulk copies target rbar
  const uint32_t full_cl = uni(mapa_u32(sa(&full[0]), lead)), full_l = uni(sa(&full[0]));
  if (warp >= 6) {  // weights (never wait on the predecessor; paced until it ends)
    const int wi = warp - 6;
    const long long t0 = p.pace ? __shfl_sync(0xffffffffu, clk(), 0) : 0;
    bool free_issue = p.pace == 0;
    for (int i = wi; i < nst; i += 2) {
      const int s = i % p.NS;
      if (i >= p.NS) mb_wait(sa(&empty[s]), ((i / p.NS) - 1) & 1);
      while (!free_issue) {
        const unsigned d = mb_done(sa(&dep)) ? 1u : 0u;
        const unsigned go = __shfl_sync(0xffffffffu, d | (clk() >= t0 + (long long)i * p.pace ? 2u : 0u), 0);
        if (go) { free_issue = go & 1u; break; }
        __nanosleep(64);
      }
      tma3_2e(uni(sa(sm + (size_t)s * SB)), &wmap, 0, 0, rt * kbt + kb0 + i, full_cl + 8 * s, full_l + 8 * s, leader ? 2 * WB : 0);
    }
  } else if (warp == 5) {  // this CTA's half of the tokens, after the dependency wait
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(&dep)) : "memory");
    for (int i = 0; i < nst; ++i) {
      const int s = i % p.NS;
      if (i >= p.NS) mb_wait(sa(&empty[s]), ((i / p.NS) - 1) & 1);
      tma3_2e(uni(sa(sm + (size_t)s * SB + WB)), &xmap, 0, (int)pr * TH, kb0 + i, full_cl + 8 * s, full_l + 8 * s, leader ? 2 * XB : 0);
    }
  } else if (warp == 4) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(sa(&tbase)), "r"(p.tcols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncwarp();
    asm volatile("bar.arrive 1, 160;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t tm = uni(*(volatile uint32_t*)&tbase);
    if (leader) {
      for (int i = 0; i < nst; ++i) {
        const int s = i % p.NS;
        mb_wait(sa(&full[s]), (i / p.NS) & 1);
        asm volatile("tcgen05.fence::after_thread_sync;");
        const uint32_t ab = uni(sa(sm + (size_t)s * SB));
        const uint64_t ad = sdesc(ab), bd = sdesc(ab + WB);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          mma2_e(tm, ad + (uint64_t)((k * 32) >> 4), bd + (uint64_t)((k * 32) >> 4), p.idesc, (i > 0 || k > 0) ? 1u : 0u);
        commit2_e(uni(sa(&empty[s])), pmask);
      }
      commit2_e(uni(sa(&accf)), pmask);
    }
  }
  // ---- epilogue: this CTA's 128 rows; owner of rows [32 w, 32 w + 32) is K-split w of the same pair rank ----
  float* recv = (float*)(sm + p.recv_off);  // [KS - 1 peers][rpr][RS]
  float* stg = (float*)sm;                  // [KS owners][rpr][RS] (aliases the drained ring)
  if (warp < 4) {
    asm volatile("bar.sync 1, 160;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t tm = tbase;
    mb_wait(sa(&accf), 0);
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (p.pdl && threadIdx.x == 0) asm volatile("griddepcontrol.launch_dependents;");
    const int r = warp * 32 + lane;
    const uint32_t ta = tm + ((uint32_t)(warp * 32) << 16);
    float* mine = stg + ((size_t)warp * rpr + lane) * RS;
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
    if (warp != (int)ks && lane == 0) {
      const unsigned idx = ks < (unsigned)warp ? ks : ks - 1;
      const unsigned orank = 2u * warp + pr;
      asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(mapa_u32(sa(recv + (size_t)idx * rpr * RS), orank)), "r"(sa(stg + (size_t)warp * rpr * RS)),
                      "r"((unsigned)(rpr * RS * 4)), "r"(mapa_u32(sa(&rbar), orank)) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
    asm volatile("bar.sync 2, 128;" ::: "memory");
    mb_wait(sa(&rbar), 0);
    constexpr int G = T / 4;
    for (int e = threadIdx.x; e < rpr * G; e += 128) {
      const int jr = e % rpr, g = e / rpr;
      const int row = rt * 128 + (int)ks * rpr + jr;
      float4 a;
#pragma unroll
      for (int src = 0; src < KS; ++src) {
        const float* part = src == (int)ks ? stg + (size_t)ks * rpr * RS : recv + (size_t)(src < (int)ks ? src : src - 1) * rpr * RS;
        const float4 b = *(const float4*)&part[(size_t)jr * RS + 4 * g];
        if (src == 0) a = b;
        else { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
      }
      if (row < p.N) {
        stout(yv, (size_t)(4 * g) * p.N + row, a.x, p.out32);
        stout(yv, (size_t)(4 * g + 1) * p.N + row, a.y, p.out32);
        stout(yv, (size_t)(4 * g + 2) * p.N + row, a.z, p.out32);
        stout(yv, (size_t)(4 * g + 3) * p.N + row, a.w, p.out32);
      }
    }
    (void)r;
    if (lane == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (warp == 4) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tbase), "r"(p.tcols) : "memory");
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
typedef void (*KF2)(CUtensorMap, CUtensorMap, void*, P);
KF pick(int T, int KB, int C, int CR, int NT) {
  if (CR == 2) {  // full pre-wait weight residency, 6-CTA split-K cluster
    if (KB != 1 || C != 6) return nullptr;
    return T == 96 && NT == 12 ? wsf_k<96, 12> : T == 64 && NT == 14 ? wsf_k<64, 14> : T == 32 && NT == 15 ? wsf_k<32, 15> : nullptr;
  }
  if (NT) {
    if (KB != 1 || C != 4 || !CR) return nullptr;
    return T == 96 && NT == 12 ? wst_k<96, 4, 12> : T == 64 && NT == 14 ? wst_k<64, 4, 14> : T == 32 && NT == 15 ? wst_k<32, 4, 15> : nullptr;
  }
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
                  int64_t NS, int64_t NW, int64_t XL, int64_t pace, int64_t cr, int64_t nt, int64_t nsx) {
  const int T = (int)x.size(0), K = (int)x.size(1), N = (int)n;
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kBFloat16, "prewait_gemm: x bf16 contiguous");
  TORCH_CHECK(out.is_contiguous() && out.size(0) == T && out.size(1) == N, "prewait_gemm: out shape");
  const bool o32 = out.scalar_type() == at::kFloat;
  TORCH_CHECK(o32 || out.scalar_type() == at::kBFloat16, "prewait_gemm: out bf16/fp32");
  const int RT = (N + 127) / 128;
  TORCH_CHECK(wp.numel() * wp.element_size() >= (int64_t)RT * 128 * K * 2, "prewait_gemm: packed weight size");
  if (cr == 3) {  // cta_group::2 pair form (M256 per pair, half the tokens' activations per CTA)
    TORCH_CHECK(C == 4 && KB == 1 && NW == 2 && XL == 1 && (T == 32 || T == 64 || T == 96) && K % 256 == 0 && NS <= 16,
                "prewait_gemm: pair config");
    const int RTp = (RT + 1) / 2 * 2;
    TORCH_CHECK(wp.numel() * wp.element_size() >= (int64_t)RTp * 128 * K * 2, "prewait_gemm: pair form needs 256-row padding");
    P p = {};
    p.N = N; p.K = K; p.T = T; p.C = 4; p.KB = 1; p.NS = (int)NS; p.NW = 2; p.RT = RTp; p.pdl = 1; p.xlanes = 1;
    p.out32 = o32; p.pace = pace;
    const unsigned WB = 16384, XB = (unsigned)(T / 2) * 128, SB = WB + XB;
    p.recv_off = (unsigned)NS * SB;
    p.idesc = (1u << 4) | (1u << 7) | (1u << 10) | ((unsigned)(T >> 3) << 17) | ((unsigned)(256 >> 4) << 24);
    unsigned tc = 32; while (tc < (unsigned)T) tc <<= 1; p.tcols = tc;
    const size_t smem = (size_t)NS * SB + (size_t)3 * 32 * (T + 4) * 4 + 1024;
    TORCH_CHECK((size_t)NS * SB >= (size_t)4 * 32 * (T + 4) * 4 && smem <= kCoresSmem, "prewait_gemm: pair smem ", smem);
    KF2 kf = T == 96 ? wsp_k<96> : T == 64 ? wsp_k<64> : wsp_k<32>;
    static thread_local KF2 seen2[4] = {};
    bool ok = false;
    for (KF2 f : seen2) ok |= f == kf;
    if (!ok) {
      int dev2; cudaGetDevice(&dev2);
      int optin2; cudaDeviceGetAttribute(&optin2, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev2);
      TORCH_CHECK(cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, optin2 - 1024) == cudaSuccess, "attr");
      cudaFuncAttributes fa;
      TORCH_CHECK(cudaFuncGetAttributes(&fa, kf) == cudaSuccess && fa.numRegs <= kCoresRegs,
                  "prewait_gemm: pair CTA registers ", fa.numRegs, " > ", kCoresRegs);
      for (KF2& f : seen2) if (!f) { f = kf; break; }
    }
    CUtensorMap xm, wm;
    {
      cuuint64_t dims[3] = {64, (cuuint64_t)T, (cuuint64_t)K / 64}; cuuint64_t strides[2] = {(cuuint64_t)K * 2, 128};
      cuuint32_t box[3] = {64, (cuuint32_t)(T / 2), 1}; cuuint32_t es[3] = {1, 1, 1};
      TORCH_CHECK(encode_fn()(&xm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)x.data_ptr(), dims, strides, box, es,
                              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "prewait_gemm: x map");
      cuuint64_t wd[3] = {64, 128, (cuuint64_t)RTp * (K / 64)}; cuuint64_t wsd[2] = {128, 16384};
      cuuint32_t wbox[3] = {64, 128, 1};
      TORCH_CHECK(encode_fn()(&wm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)wp.data_ptr(), wd, wsd, wbox, es,
                              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "prewait_gemm: w map");
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(8, RTp / 2); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = smem;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 8;
    at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[1].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = 2;
    TORCH_CHECK(cudaLaunchKernelEx(&cfg, kf, xm, wm, out.data_ptr(), p) == cudaSuccess, "prewait_gemm: pair launch");
    return;
  }
  TORCH_CHECK((cr == 2 ? K % 64 == 0 && K / 64 / C > nt : K % (64 * C * KB) == 0) && NW <= NS && XL <= NS && NS <= (nt ? 16 : 8), "prewait_gemm: config");
  TORCH_CHECK(!nt || cr == 2 || (NW + XL == 3 && nsx >= 2 && nsx <= 16 && nt < K / 64 / C), "prewait_gemm: TMEM-staged config");
  KF kf = pick(T, (int)KB, (int)C, (int)cr, (int)nt);
  TORCH_CHECK(kf, "prewait_gemm: no kernel for T=", T, " KB=", KB, " C=", C, " co-resident=", cr);
  TORCH_CHECK(pace >= 0, "prewait_gemm: pace");
  P p = {};
  p.N = N; p.K = K; p.T = T; p.C = (int)C; p.KB = (int)KB; p.NS = (int)NS; p.NW = (int)NW; p.RT = RT;
  p.nst = K / 64 / (int)C / (int)KB; p.pdl = 1; p.xlanes = (int)XL; p.out32 = o32; p.pace = pace;
  p.nm = (T + 255) / 256;
  p.WB = KB * 16384; p.XB = KB * T * 128; p.SB = p.WB + p.XB;
  p.recv_off = cr == 2 ? NS * p.WB : nt ? NS * p.WB + nsx * p.XB : NS * p.SB; p.NSX = (int)nsx;
  TORCH_CHECK(cr != 2 || (NS > K / 64 / C + 1 - nt && NS <= 16), "prewait_gemm: full-residency pool too small");
  p.idesc = (1u << 4) | (1u << 7) | (1u << 10) | ((unsigned)((T / p.nm) >> 3) << 17) | ((unsigned)(128 >> 4) << 24);
  unsigned tc = 32; while (tc < (unsigned)T) tc <<= 1; p.tcols = tc;
  const size_t rbytes = C > 1 ? (size_t)(cr ? C - 1 : C) * (T + 4) * ((128 + C - 1) / C) * 4 : 0;
  TORCH_CHECK(!cr || (size_t)p.recv_off >= (size_t)C * (T + 4) * ((128 + C - 1) / C) * 4, "prewait_gemm: staging > ring");
  const size_t smem = (size_t)p.recv_off + rbytes + 1024;
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
    cuuint64_t dims[3] = {64, (cuuint64_t)T, (cuuint64_t)K / 64}; cuuint64_t strides[2] = {(cuuint64_t)K * 2, 128};
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
# TMEM-staged co-resident form (NT weight stages copied to tensor memory before the PDL wait, NSX
# activation stages): (C, KB, NS, NW, XL, PACE, CR, NT, NSX).
TMEM_CONFIGS = {32: (4, 1, 8, 2, 1, 960, 1, 15, 8), 64: (4, 1, 6, 2, 1, 960, 1, 14, 8),
                96: (4, 1, 6, 2, 1, 960, 1, 12, 6)}
# Full pre-wait weight residency (CR 2): 6-CTA split-K cluster (210 CTAs), NT K-blocks per CTA in TMEM, the rest in a
# pool of NS 16 KiB slots that turns into the activation ring after the PDL wait.
FULL_CONFIGS = {32: (6, 1, 12, 2, 1, 0, 2, 15, 0), 64: (6, 1, 11, 2, 1, 0, 2, 14, 0), 96: (6, 1, 10, 2, 1, 0, 2, 12, 0)}
# cta_group::2 pair form (CR 3): 8-CTA clusters (2 pair members x 4 K-splits), NS ring stages of 16 KiB weights + half
# the tokens' activations.
PAIR_CONFIGS = {32: (4, 1, 11, 2, 1, 1440, 3, 0, 0), 64: (4, 1, 9, 2, 1, 1440, 3, 0, 0), 96: (4, 1, 8, 2, 1, 1440, 3, 0, 0)}
CONFIGS[64], CONFIGS[96] = PAIR_CONFIGS[64], PAIR_CONFIGS[96]  # EXPERIMENT: pair form in-step


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


def supported(n_tokens: int, k: int) -> bool:
    cfg = CONFIGS.get(n_tokens)
    return cfg is not None and k % (64 * cfg[0] * cfg[1]) == 0


def pack_weight(w: torch.Tensor) -> torch.Tensor:
    """[N, K] BF16 row-major -> the kernel's tile image [ceil(N/128), K/64, 128, 64] with the
    128-byte swizzle (16-byte chunk c of row r stored at chunk c ^ (r % 8)); rows >= N are zero."""
    n, k = w.shape
    assert w.dtype == torch.bfloat16 and k % 64 == 0
    rt = (n + 255) // 256 * 2  # whole 256-row pairs (the pair form reads both tiles of a pair)
    if rt * 128 != n:
        w = torch.cat([w, w.new_zeros(rt * 128 - n, k)])
    t = w.reshape(rt, 128, k // 64, 8, 8).permute(0, 2, 1, 3, 4)  # [rt, kb, r, c, e]
    r = torch.arange(128, device=w.device)[:, None]
    pc = torch.arange(8, device=w.device)[None, :]
    return t[:, :, r, pc ^ (r & 7), :].contiguous()


def gemm(out: torch.Tensor, x: torch.Tensor, packed: torch.Tensor, n: int,
         config: tuple | None = None) -> torch.Tensor:
    """out[T, n] = x @ W^T for the W ``packed`` came from (``pack_weight``)."""
    cfg = tuple(config or CONFIGS[x.shape[0]])
    c, kb, ns, nw, xl, pace, cr, nt, nsx = cfg + (0, 0)[: 9 - len(cfg)]
    _module().prewait_gemm(out, x, packed, n, c, kb, ns, nw, xl, pace, cr, nt, nsx)
    return out
