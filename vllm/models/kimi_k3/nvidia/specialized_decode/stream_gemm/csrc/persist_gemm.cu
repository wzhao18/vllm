// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Persistent multi-phase weight-streaming decode GEMM chain (G-PERSIST).
// R <= 256 token rows, BF16 operands, FP32 accumulation, one BF16 rounding.
//
// Design (speed of light for a weight-stream-bound GEMM):
//  * swap-AB: 128 weight rows are the MMA M operand, the R tokens the N operand
//    (one M128 x N=BN x K16 tcgen05.mma per 16 K), accumulator in tensor memory;
//  * one CTA per SM, the whole shared memory a TMA ring of (16 KB weight box +
//    BN x 128 B token box) stages;
//  * stream-K: the tiles x K-blocks sequence is cut into equal chunks, one per CTA,
//    so every SM streams the same number of weight bytes (a chunk spans <= 2 tiles);
//  * split tiles reduce through L2: every contributor stores its FP32 partial,
//    arrives on the tile counter, and then sums its share of the token rows over all
//    contributors in contributor order (deterministic, one rounding);
//  * PDL: the producer requests the first ring's weight boxes before
//    griddepcontrol.wait (weights do not depend on the previous kernel), the token
//    boxes after it; launch_dependents fires at entry.
#include <cuda.h>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>

namespace k31sg {

constexpr int kThreads = 192;  // warp 0 TMA, warp 1 MMA + TMEM, warps 2-5 epilogue
constexpr int kWBox = 128 * 128;  // 128 weight rows x 64 BF16 (128 B), bytes
constexpr int kMaxStages = 16;

struct Params {
  __nv_bfloat16* out;
  float* ws;
  unsigned* cnt;
  unsigned long long* stamps;  // [grid][8] globaltimer ns (benchmarks) or null
  int n, k, r, bn;
  int tiles, kb, chunk, total, stages;
  int pf;  // weight boxes requested into L2 ahead of the ring (0: the whole chunk at entry)
  long long ldo;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
               "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{ .reg .pred p; WAIT_%=: "
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; "
      "@!p bra WAIT_%=; }" ::"r"(bar),
      "r"(parity)
      : "memory");
}

__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int c0,
                                       int c1, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void tma_2d_evict_first(uint32_t dst, const CUtensorMap* map,
                                                   int c0, int c1, uint32_t bar) {
  asm volatile(
      "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      ".L2::cache_hint [%0], [%1, {%2, %3}], [%4], pol; }" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void tma_prefetch_2d(const CUtensorMap* map, int c0, int c1) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" ::"l"(
                   reinterpret_cast<uint64_t>(map)),
               "r"(c0), "r"(c1)
               : "memory");
}

// K-major operand in a 128-byte-swizzled TMA box: rows of 128 B, 8-row atoms of
// 1024 B (SBO), descriptor version 1, layout SWIZZLE_128B.
__device__ __forceinline__ uint64_t smem_desc(uint32_t addr) {
  return static_cast<uint64_t>((addr & 0x3FFFF) >> 4) | (1ull << 16) | (64ull << 32) |
         (1ull << 46) | (2ull << 61);
}

__device__ __forceinline__ void mma_bf16(uint32_t tmem, uint64_t a, uint64_t b,
                                         uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %4, 0; "
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }" ::"r"(tmem),
      "l"(a), "l"(b), "r"(idesc), "r"(acc));
}

__device__ __forceinline__ void mma_commit(uint32_t bar) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(
          bar)
      : "memory");
}

__device__ __forceinline__ void tmem_ld16(uint32_t taddr, float (&v)[16]) {
  uint32_t r[16];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,"
      "%12,%13,%14,%15}, [%16];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]),
        "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]),
        "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = __uint_as_float(r[i]);
}

__device__ __forceinline__ void named_sync(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}

__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define STAMP(i) \
  do { \

// ---------------------------------------------------------------------------------
// Persistent layer-section GEMM chain (G-PERSIST): one launch runs up to kMaxPh dense
// GEMM phases out_p[R, N_p] = x_p[R, K_p] @ W_p[N_p, K_p]^T, x_{p+1} = out_p (or
// SiTU-and-mul(out_p) for an "act" phase whose weight rows are gate/up interleaved per
// 128-row tile: rows 0-63 gate, 64-127 up of the same 64 output columns).
// Phases are separated by a grid-wide flag barrier instead of a kernel boundary. Before
// waiting for phase p's input, every CTA has already put phase p's first ring of weight
// boxes in flight into shared memory (and the rest of its weight chunk into L2), i.e.
// the next phase's weights stream while the current phase computes and reduces.
// ---------------------------------------------------------------------------------
constexpr int kMaxPh = 3;

struct PMaps {
  CUtensorMap tw[kMaxPh];
  CUtensorMap tx[kMaxPh];
};

struct PPhase {
  __nv_bfloat16* out;
  unsigned* cnt;  // split-tile counters (monotone generations)
  int n, k, tiles, kb, chunk, total, act;
  long long ldo;
};

struct PParams {
  PPhase ph[kMaxPh];
  int nph, r, bn, stages, pf;
  float* ws;
  unsigned* bar;  // [0] phase barrier counter, [1] exit counter (self-resetting)
  unsigned long long* stamps;  // [grid][8] or null
};

#define PSTAMP(i) \
  do { \
    if (P.stamps) P.stamps[blockIdx.x * 8 + (i)] = globaltimer(); \
  } while (0)

__device__ __forceinline__ float situ(float g) { return g / (1.f + __expf(-g)); }

__global__ void __launch_bounds__(kThreads, 1)
    persist_gemm_bf16(const __grid_constant__ PMaps M, const PParams P) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* ring = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int xbytes = P.bn * 128;
  const int stage_bytes = kWBox + xbytes;
  uint64_t* bars = reinterpret_cast<uint64_t*>(ring + P.stages * stage_bytes);
  // full[kMaxStages], empty[kMaxStages], acc[2 slots][2 segs]
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(bars + 2 * kMaxStages + 4);
  unsigned* targets = tmem_slot + 1;

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int b = blockIdx.x, G = gridDim.x;
  const uint32_t ring_s = smem_u32(ring);
  const uint32_t full0 = smem_u32(bars), empty0 = smem_u32(bars + kMaxStages),
                 acc0 = smem_u32(bars + 2 * kMaxStages);

  if (threadIdx.x == 0) asm volatile("griddepcontrol.launch_dependents;");
  if (threadIdx.x == 0) {
    PSTAMP(0);
    for (int s = 0; s < P.stages; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, 1);
    }
    for (int a = 0; a < 4; ++a) mbar_init(acc0 + 8 * a, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // two accumulator slots (phase parity) x two segments x bn columns
  const int need = 4 * P.bn;
  const int cols = need <= 32 ? 32 : need <= 64 ? 64 : need <= 128 ? 128
                   : need <= 256 ? 256 : 512;
  if (warp == 1) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
            smem_u32(tmem_slot)),
        "r"(cols));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t tmem = *tmem_slot;

  if (warp == 0) {
    if (lane == 0) {
      int it = 0;  // global ring sequence over all phases
      for (int ph = 0; ph < P.nph; ++ph) {
        const PPhase& Q = P.ph[ph];
        const CUtensorMap* tw = &M.tw[ph];
        const CUtensorMap* tx = &M.tx[ph];
        const int begin = b * Q.chunk, end = min(Q.total, begin + Q.chunk);
        const int n_kb = max(0, end - begin);
        const int pre = min(n_kb, P.stages);
        // phase ph's first ring of weights (waits only for ring slots, not for x)
        for (int i = 0; i < pre; ++i) {
          const int s = (it + i) % P.stages;
          const uint32_t par = ((it + i) / P.stages) & 1;
          mbar_wait(empty0 + 8 * s, par ^ 1);
          const int g = begin + i;
          mbar_expect_tx(full0 + 8 * s, stage_bytes);
          tma_2d_evict_first(ring_s + s * stage_bytes, tw, (g % Q.kb) * 64,
                             (g / Q.kb) * 128, full0 + 8 * s);
        }
        const int pf_end = P.pf > 0 ? min(n_kb, pre + P.pf) : n_kb;
        for (int i = pre; i < pf_end; ++i) {
          const int g = begin + i;
          tma_prefetch_2d(tw, (g % Q.kb) * 64, (g / Q.kb) * 128);
        }
        // phase input dependency
        if (ph == 0) {
          asm volatile("griddepcontrol.wait;" ::: "memory");
        } else {
          const unsigned target = static_cast<unsigned>(ph) * G;
          while (static_cast<int>(ld_acquire(P.bar) - target) < 0) {
          }
          asm volatile("fence.proxy.async.global;" ::: "memory");
        }
        if (ph == P.nph - 1) PSTAMP(1);
        for (int i = 0; i < pre; ++i) {
          const int s = (it + i) % P.stages;
          const int g = begin + i;
          tma_2d(ring_s + s * stage_bytes + kWBox, tx, (g % Q.kb) * 64, 0, full0 + 8 * s);
        }
        for (int i = pre; i < n_kb; ++i) {
          if (P.pf > 0 && i + P.pf < n_kb) {
            const int gp = begin + i + P.pf;
            tma_prefetch_2d(tw, (gp % Q.kb) * 64, (gp / Q.kb) * 128);
          }
          const int s = (it + i) % P.stages;
          const uint32_t par = ((it + i) / P.stages) & 1;
          mbar_wait(empty0 + 8 * s, par ^ 1);
          const int g = begin + i;
          const uint32_t st = ring_s + s * stage_bytes;
          mbar_expect_tx(full0 + 8 * s, stage_bytes);
          tma_2d_evict_first(st, tw, (g % Q.kb) * 64, (g / Q.kb) * 128, full0 + 8 * s);
          tma_2d(st + kWBox, tx, (g % Q.kb) * 64, 0, full0 + 8 * s);
        }
        it += n_kb;
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) |
                             (static_cast<uint32_t>(P.bn >> 3) << 17) | (8u << 24);
      int it = 0;
      for (int ph = 0; ph < P.nph; ++ph) {
        const PPhase& Q = P.ph[ph];
        const int begin = b * Q.chunk, end = min(Q.total, begin + Q.chunk);
        const int n_kb = max(0, end - begin);
        const int split0 = (begin / Q.kb + 1) * Q.kb;
        const int slot = ph & 1;
        for (int i = 0; i < n_kb; ++i) {
          const int s = (it + i) % P.stages;
          const uint32_t par = ((it + i) / P.stages) & 1;
          const int g = begin + i;
          const int seg = g >= split0 ? 1 : 0;
          const bool first = (g == begin) || (g == split0);
          mbar_wait(full0 + 8 * s, par);
          if (ph == P.nph - 1 && i + 1 == n_kb) PSTAMP(7);
          asm volatile("tcgen05.fence::after_thread_sync;");
          const uint32_t st = ring_s + s * stage_bytes;
          const uint64_t a = smem_desc(st), bd = smem_desc(st + kWBox);
          const uint32_t d = tmem + (slot * 2 + seg) * P.bn;
#pragma unroll
          for (int kk = 0; kk < 4; ++kk)
            mma_bf16(d, a + 2 * kk, bd + 2 * kk, idesc, (first && kk == 0) ? 0u : 1u);
          mma_commit(empty0 + 8 * s);
          if (g + 1 == end || g + 1 == split0) mma_commit(acc0 + 8 * (slot * 2 + seg));
        }
        it += n_kb;
      }
    }
  } else {
    const int q = warp & 3;
    const int row = q * 32 + lane;
    const int et = threadIdx.x - 64;  // 0..127
    for (int ph = 0; ph < P.nph; ++ph) {
      const PPhase& Q = P.ph[ph];
      const int begin = b * Q.chunk, end = min(Q.total, begin + Q.chunk);
      const int slot = ph & 1;
      const uint32_t apar = (ph >> 1) & 1;
      if (begin < end) {
        const int tile0 = begin / Q.kb;
        const int split0 = (tile0 + 1) * Q.kb;
        const int nseg = end > split0 ? 2 : 1;
        bool split_seg[2] = {false, false};
        for (int seg = 0; seg < nseg; ++seg) {
          const int tile = tile0 + seg;
          const int c_first = (tile * Q.kb) / Q.chunk;
          const int c_last = ((tile + 1) * Q.kb - 1) / Q.chunk;
          mbar_wait(acc0 + 8 * (slot * 2 + seg), apar);
          asm volatile("tcgen05.fence::after_thread_sync;");
          const int n = tile * 128 + row;
          const uint32_t ta =
              tmem + (static_cast<uint32_t>(q * 32) << 16) + (slot * 2 + seg) * P.bn;
          if (c_first == c_last) {  // (act phases require split tiles: host check)
            for (int c = 0; c < P.bn; c += 16) {
              float v[16];
              tmem_ld16(ta + c, v);
              if (n < Q.n) {
#pragma unroll
                for (int j = 0; j < 16; ++j)
                  if (c + j < P.r) Q.out[(c + j) * Q.ldo + n] = __float2bfloat16_rn(v[j]);
              }
            }
          } else {
            split_seg[seg] = true;
            float* part = P.ws + (static_cast<long long>(b * 2 + seg) * P.bn) * 128;
            for (int c = 0; c < P.bn; c += 16) {
              float v[16];
              tmem_ld16(ta + c, v);
#pragma unroll
              for (int j = 0; j < 16; ++j)
                if (c + j < P.r) __stcg(part + (c + j) * 128 + row, v[j]);
            }
          }
        }
        if (split_seg[0] || split_seg[1]) {
          named_sync(1, 128);
          if (et == 0) {
            __threadfence();
            for (int seg = 0; seg < nseg; ++seg) {
              if (!split_seg[seg]) continue;
              const int tile = tile0 + seg;
              const int c_first = (tile * Q.kb) / Q.chunk;
              const int c_last = ((tile + 1) * Q.kb - 1) / Q.chunk;
              const unsigned cn = c_last - c_first + 1;
              const unsigned old = atomicAdd(Q.cnt + tile, 1u);
              targets[seg] = (old / cn + 1) * cn;
            }
          }
          named_sync(1, 128);
          for (int seg = 0; seg < nseg; ++seg) {
            if (!split_seg[seg]) continue;
            const int tile = tile0 + seg;
            const int c_first = (tile * Q.kb) / Q.chunk;
            const int c_last = ((tile + 1) * Q.kb - 1) / Q.chunk;
            const int cn = c_last - c_first + 1;
            if (et == 0)
              while (static_cast<int>(ld_acquire(Q.cnt + tile) - targets[seg]) < 0) {
              }
            named_sync(1, 128);
            const int qi = b - c_first;
            const int t0 = (qi * P.r) / cn, t1 = ((qi + 1) * P.r) / cn;
            // float4 f = token t0 + f / fpt, rows 4 (f % fpt); act: gate rows 0-63
            // with their up rows 64-127 in the same thread.
            const int fpt = Q.act ? 16 : 32;
            const int nf = (t1 - t0) * fpt;
            for (int f = et; f < nf; f += 128) {
              const int t = t0 + f / fpt, r4 = (f % fpt) * 4;
              float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
              float4 acu = make_float4(0.f, 0.f, 0.f, 0.f);
              for (int c0 = c_first; c0 <= c_last; c0 += 4) {
                float4 v[4], u[4];
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                  const int c = c0 + w;
                  if (c <= c_last) {
                    const int sc = tile - (c * Q.chunk) / Q.kb;
                    const float* base =
                        P.ws + (static_cast<long long>(c * 2 + sc) * P.bn + t) * 128;
                    v[w] = __ldcg(reinterpret_cast<const float4*>(base + r4));
                    if (Q.act) u[w] = __ldcg(reinterpret_cast<const float4*>(base + 64 + r4));
                  }
                }
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                  if (c0 + w <= c_last) {
                    acc.x += v[w].x; acc.y += v[w].y; acc.z += v[w].z; acc.w += v[w].w;
                    if (Q.act) {
                      acu.x += u[w].x; acu.y += u[w].y; acu.z += u[w].z; acu.w += u[w].w;
                    }
                  }
                }
              }
              float o[4] = {acc.x, acc.y, acc.z, acc.w};
              int n0 = tile * 128 + r4;
              if (Q.act) {
                // stock: gate_up rounded to BF16, SiTU(gate) * up, one BF16 rounding
                const float uu[4] = {acu.x, acu.y, acu.z, acu.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                  const float g = __bfloat162float(__float2bfloat16_rn(o[j]));
                  const float up = __bfloat162float(__float2bfloat16_rn(uu[j]));
                  o[j] = situ(g) * up;
                }
                n0 = tile * 64 + r4;
              }
              if (n0 < (Q.act ? Q.n / 2 : Q.n)) {
                __nv_bfloat162 lo = __floats2bfloat162_rn(o[0], o[1]);
                __nv_bfloat162 hi = __floats2bfloat162_rn(o[2], o[3]);
                uint2 pk;
                pk.x = *reinterpret_cast<uint32_t*>(&lo);
                pk.y = *reinterpret_cast<uint32_t*>(&hi);
                *reinterpret_cast<uint2*>(Q.out + t * Q.ldo + n0) = pk;
              }
            }
          }
        }
      }
      // phase ph's outputs of this CTA are stored: arrive on the grid barrier
      named_sync(1, 128);
      if (et == 0) {
        if (ph == P.nph - 1) PSTAMP(5);
        if (ph + 1 < P.nph) {
          __threadfence();
          asm volatile("fence.proxy.async.global;" ::: "memory");
          atomicAdd(P.bar, 1u);
        }
      }
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (warp == 1) {
    asm volatile("tcgen05.fence::after_thread_sync;");
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem),
                 "r"(cols));
  }
  // self-reset of the phase barrier: the last CTA out zeroes both counters
  if (threadIdx.x == 0 && P.nph > 1) {
    __threadfence();
    const unsigned old = atomicAdd(P.bar + 1, 1u);
    if (old == static_cast<unsigned>(G) - 1) {
      atomicExch(P.bar, 0u);
      atomicExch(P.bar + 1, 0u);
    }
  }
}

static void pmake_map(CUtensorMap* m, const void* ptr, int rows, int k, int box_rows) {
  cuuint64_t dims[2] = {static_cast<cuuint64_t>(k), static_cast<cuuint64_t>(rows)};
  cuuint64_t strides[1] = {static_cast<cuuint64_t>(k) * 2};
  cuuint32_t box[2] = {64, static_cast<cuuint32_t>(box_rows)};
  cuuint32_t es[2] = {1, 1};
  CUresult rc = cuTensorMapEncodeTiled(
      m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box,
      es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(rc == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", static_cast<int>(rc));
}

int psmem_bytes(int bn, int stages) {
  return 1024 + stages * (kWBox + bn * 128) + (2 * kMaxStages + 4) * 8 + 16;
}

// x: [R, K0] BF16. ws_list[p]: [N_p, K_p] BF16 (act phases: gate/up interleaved per
// 128-row tile). outs[p]: [R, N_p] (act: [R, N_p / 2]); out_p is x of phase p + 1.
// cnts[p]: uint32 tile counters. bar: uint32[2] zeros (self-resetting).
void prun(std::vector<torch::Tensor> w, torch::Tensor x, std::vector<torch::Tensor> outs,
          std::vector<int64_t> act, torch::Tensor ws, std::vector<torch::Tensor> cnts,
          torch::Tensor bar, int64_t grid, int64_t stages, int64_t pf,
          torch::Tensor stamps) {
  const int nph = w.size();
  TORCH_CHECK(nph >= 1 && nph <= kMaxPh && outs.size() == w.size() && act.size() == w.size()
              && cnts.size() == w.size());
  const int r = x.size(0);
  TORCH_CHECK(r >= 1 && r <= 96, "1 <= R <= 96 (4 x bn TMEM columns)");
  TORCH_CHECK(stages >= 2 && stages <= kMaxStages);
  const int bn = (r + 15) / 16 * 16;
  PMaps maps;
  PParams P{};
  P.nph = nph;
  P.r = r;
  P.bn = bn;
  P.stages = stages;
  P.pf = pf;
  P.ws = ws.data_ptr<float>();
  P.bar = reinterpret_cast<unsigned*>(bar.data_ptr());
  P.stamps = stamps.numel() ? reinterpret_cast<unsigned long long*>(stamps.data_ptr())
                            : nullptr;
  torch::Tensor xin = x;
  for (int p = 0; p < nph; ++p) {
    const int n = w[p].size(0), k = w[p].size(1);
    TORCH_CHECK(w[p].dtype() == torch::kBFloat16 && w[p].is_contiguous());
    TORCH_CHECK(xin.size(1) == k && xin.size(0) == r && xin.is_contiguous(), "phase ", p,
                " input");
    TORCH_CHECK(k % 64 == 0 && n % 128 == 0, "K % 64, N % 128");
    const int nout = act[p] ? n / 2 : n;
    TORCH_CHECK(outs[p].size(0) == r && outs[p].size(1) == nout && outs[p].is_contiguous());
    const int tiles = n / 128, kb = k / 64, total = tiles * kb;
    const int chunk = (total + grid - 1) / grid;
    TORCH_CHECK(chunk <= kb, "phase ", p, ": chunk spans <= 2 tiles");
    TORCH_CHECK(!act[p] || chunk < kb, "act phase needs every tile split");
    TORCH_CHECK(cnts[p].numel() >= tiles);
    pmake_map(&maps.tw[p], w[p].data_ptr(), n, k, 128);
    pmake_map(&maps.tx[p], xin.data_ptr(), r, k, bn);
    P.ph[p] = PPhase{reinterpret_cast<__nv_bfloat16*>(outs[p].data_ptr()),
                     reinterpret_cast<unsigned*>(cnts[p].data_ptr()),
                     n, k, tiles, kb, chunk, total, static_cast<int>(act[p]),
                     outs[p].stride(0)};
    xin = outs[p];
  }
  TORCH_CHECK(ws.numel() >= grid * 2 * bn * 128);
  const int smem = psmem_bytes(bn, stages);
  static int configured = 0;
  if (smem > configured) {
    TORCH_CHECK(cudaFuncSetAttribute(persist_gemm_bf16,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     smem) == cudaSuccess, "smem attribute ", smem);
    configured = smem;
  }
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(static_cast<unsigned>(grid));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, persist_gemm_bf16, maps, P) == cudaSuccess,
              "launch: ", cudaGetErrorString(cudaGetLastError()));
}

}  // namespace k31sg

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prun", &k31sg::prun);
  m.def("psmem_bytes", &k31sg::psmem_bytes);
}
