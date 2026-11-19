// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Weight-streaming decode GEMM (Rubin, tcgen05): out[R, N] = x[R, K] @ W[N, K]^T for
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
    if (p.stamps) p.stamps[blockIdx.x * 8 + (i)] = globaltimer(); \
  } while (0)

__global__ void __launch_bounds__(kThreads, 1)
    stream_gemm_bf16(const __grid_constant__ CUtensorMap tw,
                     const __grid_constant__ CUtensorMap tx, const Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  // 1024-aligned ring: stage s = [W box 16 KB][x box BN x 128 B]
  uint8_t* ring = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int xbytes = p.bn * 128;
  const int stage_bytes = kWBox + xbytes;
  uint64_t* bars = reinterpret_cast<uint64_t*>(ring + p.stages * stage_bytes);
  // full[stages], empty[stages], acc[2]
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(bars + 2 * kMaxStages + 2);
  unsigned* targets = tmem_slot + 1;  // [2] counter targets

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int b = blockIdx.x;
  const int begin = b * p.chunk;
  const int end = min(p.total, begin + p.chunk);

  if (threadIdx.x == 0) asm volatile("griddepcontrol.launch_dependents;");
  if (threadIdx.x == 0) {
    STAMP(0);
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    if (p.stamps) p.stamps[blockIdx.x * 8 + 6] = smid;
  }
  if (begin >= end) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    return;
  }
  const int tile0 = begin / p.kb;
  const int split0 = (tile0 + 1) * p.kb;  // first K-block of tile0 + 1
  const int nseg = end > split0 ? 2 : 1;
  const uint32_t ring_s = smem_u32(ring);
  const uint32_t full0 = smem_u32(bars), empty0 = smem_u32(bars + kMaxStages),
                 acc0 = smem_u32(bars + 2 * kMaxStages);

  if (threadIdx.x == 0) {
    for (int s = 0; s < p.stages; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, 1);
    }
    mbar_init(acc0, 1);
    mbar_init(acc0 + 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  const int cols = 2 * p.bn <= 32 ? 32 : 2 * p.bn <= 64 ? 64 : 2 * p.bn <= 128 ? 128
                   : 2 * p.bn <= 256 ? 256 : 512;
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
  const int n_kb = end - begin;

  if (warp == 0) {
    if (lane == 0) {
      // The first ring's weight boxes into shared memory and the rest of the weight
      // chunk into L2, all before the PDL wait; tokens after it.
      const int pre = min(n_kb, p.stages);
      for (int i = 0; i < pre; ++i) {
        const int g = begin + i;
        const uint32_t st = ring_s + i * stage_bytes;
        mbar_expect_tx(full0 + 8 * i, stage_bytes);
        tma_2d_evict_first(st, &tw, (g % p.kb) * 64, (g / p.kb) * 128, full0 + 8 * i);
      }
      const int pf_end = p.pf > 0 ? min(n_kb, pre + p.pf) : n_kb;
      for (int i = pre; i < pf_end; ++i) {
        const int g = begin + i;
        tma_prefetch_2d(&tw, (g % p.kb) * 64, (g / p.kb) * 128);
      }
      asm volatile("griddepcontrol.wait;" ::: "memory");
      STAMP(1);
      for (int i = 0; i < pre; ++i) {
        const int g = begin + i;
        tma_2d(ring_s + i * stage_bytes + kWBox, &tx, (g % p.kb) * 64, 0, full0 + 8 * i);
      }
      for (int i = pre; i < n_kb; ++i) {
        if (p.pf > 0 && i + p.pf < n_kb) {
          const int gp = begin + i + p.pf;
          tma_prefetch_2d(&tw, (gp % p.kb) * 64, (gp / p.kb) * 128);
        }
        const int s = i % p.stages;
        const uint32_t ph = (i / p.stages) & 1;
        mbar_wait(empty0 + 8 * s, ph ^ 1);
        const int g = begin + i;
        const uint32_t st = ring_s + s * stage_bytes;
        mbar_expect_tx(full0 + 8 * s, stage_bytes);
        tma_2d_evict_first(st, &tw, (g % p.kb) * 64, (g / p.kb) * 128, full0 + 8 * s);
        tma_2d(st + kWBox, &tx, (g % p.kb) * 64, 0, full0 + 8 * s);
      }
    }
  } else if (warp == 1) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (lane == 0) {
      const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) |
                             (static_cast<uint32_t>(p.bn >> 3) << 17) | (8u << 24);
      for (int i = 0; i < n_kb; ++i) {
        const int s = i % p.stages;
        const uint32_t ph = (i / p.stages) & 1;
        const int g = begin + i;
        const int seg = g >= split0 ? 1 : 0;
        const bool first = (g == begin) || (g == split0);
        mbar_wait(full0 + 8 * s, ph);
        if (i + 1 == n_kb) STAMP(7);  // last stage's operands landed
        asm volatile("tcgen05.fence::after_thread_sync;");
        const uint32_t st = ring_s + s * stage_bytes;
        const uint64_t a = smem_desc(st), bd = smem_desc(st + kWBox);
        const uint32_t d = tmem + seg * p.bn;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
          mma_bf16(d, a + 2 * kk, bd + 2 * kk, idesc, (first && kk == 0) ? 0u : 1u);
        mma_commit(empty0 + 8 * s);
        if (g + 1 == end || g + 1 == split0) mma_commit(acc0 + 8 * seg);
      }
    }
  } else {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    // Epilogue: thread = one weight row (TMEM lane) of the tile.
    const int q = warp & 3;
    const int row = q * 32 + lane;
    const int et = threadIdx.x - 64;  // 0..127
    bool split_seg[2] = {false, false};
    for (int seg = 0; seg < nseg; ++seg) {
      const int tile = tile0 + seg;
      const int c_first = (tile * p.kb) / p.chunk;
      const int c_last = ((tile + 1) * p.kb - 1) / p.chunk;
      mbar_wait(acc0 + 8 * seg, 0);
      if (et == 0) STAMP(2);
      asm volatile("tcgen05.fence::after_thread_sync;");
      const int n = tile * 128 + row;
      const uint32_t ta = tmem + (static_cast<uint32_t>(q * 32) << 16) + seg * p.bn;
      if (c_first == c_last) {
        for (int c = 0; c < p.bn; c += 16) {
          float v[16];
          tmem_ld16(ta + c, v);
          if (n < p.n) {
#pragma unroll
            for (int j = 0; j < 16; ++j)
              if (c + j < p.r) p.out[(c + j) * p.ldo + n] = __float2bfloat16_rn(v[j]);
          }
        }
      } else {
        split_seg[seg] = true;
        float* part = p.ws + (static_cast<long long>(b * 2 + seg) * p.bn) * 128;
        for (int c = 0; c < p.bn; c += 16) {
          float v[16];
          tmem_ld16(ta + c, v);
#pragma unroll
          for (int j = 0; j < 16; ++j)
            if (c + j < p.r) __stcg(part + (c + j) * 128 + row, v[j]);
        }
      }
    }
    if (split_seg[0] || split_seg[1]) {
      named_sync(1, 128);
      if (et == 0) STAMP(3);
      if (et == 0) {
        __threadfence();  // cumulative over the CTA's partial stores (bar.sync above)
        for (int seg = 0; seg < nseg; ++seg) {
          if (!split_seg[seg]) continue;
          const int tile = tile0 + seg;
          const int c_first = (tile * p.kb) / p.chunk;
          const int c_last = ((tile + 1) * p.kb - 1) / p.chunk;
          const unsigned cn = c_last - c_first + 1;
          const unsigned old = atomicAdd(p.cnt + tile, 1u);
          targets[seg] = (old / cn + 1) * cn;
        }
      }
      named_sync(1, 128);
      for (int seg = 0; seg < nseg; ++seg) {
        if (!split_seg[seg]) continue;
        const int tile = tile0 + seg;
        const int c_first = (tile * p.kb) / p.chunk;
        const int c_last = ((tile + 1) * p.kb - 1) / p.chunk;
        const int cn = c_last - c_first + 1;
        if (et == 0)
          while (static_cast<int>(ld_acquire(p.cnt + tile) - targets[seg]) < 0) {
          }
        named_sync(1, 128);
        if (et == 0) STAMP(4);
        const int qi = b - c_first;
        const int t0 = (qi * p.r) / cn, t1 = ((qi + 1) * p.r) / cn;
        // float4 f of the share = token t0 + f / 32, rows 4 (f % 32) .. +3; every
        // contributor's float4 is requested before the in-order sum (8 at a time).
        const int nf = (t1 - t0) * 32;
        for (int f = et; f < nf; f += 128) {
          const int t = t0 + (f >> 5), r4 = (f & 31) * 4;
          const int n0 = tile * 128 + r4;
          float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
          for (int c0 = c_first; c0 <= c_last; c0 += 8) {
            float4 v[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
              const int c = c0 + u;
              if (c <= c_last) {
                const int sc = tile - (c * p.chunk) / p.kb;
                v[u] = __ldcg(reinterpret_cast<const float4*>(
                    p.ws + (static_cast<long long>(c * 2 + sc) * p.bn + t) * 128 + r4));
              }
            }
#pragma unroll
            for (int u = 0; u < 8; ++u) {
              if (c0 + u <= c_last) {
                acc.x += v[u].x;
                acc.y += v[u].y;
                acc.z += v[u].z;
                acc.w += v[u].w;
              }
            }
          }
          if (n0 < p.n) {
            __nv_bfloat162 lo = __floats2bfloat162_rn(acc.x, acc.y);
            __nv_bfloat162 hi = __floats2bfloat162_rn(acc.z, acc.w);
            uint2 pk;
            pk.x = *reinterpret_cast<uint32_t*>(&lo);
            pk.y = *reinterpret_cast<uint32_t*>(&hi);
            *reinterpret_cast<uint2*>(p.out + t * p.ldo + n0) = pk;
          }
        }
      }
    }
  }
  if (threadIdx.x == 64) STAMP(5);
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (warp == 1) {
    asm volatile("tcgen05.fence::after_thread_sync;");
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem),
                 "r"(cols));
  }
}

static void make_map(CUtensorMap* m, const void* ptr, int rows, int k, int box_rows) {
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

int smem_bytes(int bn, int stages) {
  return 1024 + stages * (kWBox + bn * 128) + (2 * kMaxStages + 2) * 8 + 16;
}

// out[R, N] = x[R, K] @ w[N, K]^T. ws: FP32 workspace >= grid x 2 x bn x 128;
// cnt: uint32 tile counters (zero at first use, owned by this weight's call site).
void run(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws,
         torch::Tensor cnt, int64_t grid, int64_t stages,
         torch::Tensor stamps, int64_t pf) {
  TORCH_CHECK(w.dtype() == torch::kBFloat16 && x.dtype() == torch::kBFloat16 &&
              out.dtype() == torch::kBFloat16);
  TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && out.stride(1) == 1);
  const int n = w.size(0), k = w.size(1), r = x.size(0);
  TORCH_CHECK(x.size(1) == k && out.size(0) == r && out.size(1) == n);
  TORCH_CHECK(k % 64 == 0 && n % 4 == 0 && r >= 1 && r <= 256, "K % 64, N % 4, 1 <= R <= 256");
  TORCH_CHECK(stages >= 2 && stages <= kMaxStages);
  const int bn = (r + 15) / 16 * 16;
  const int tiles = (n + 127) / 128;
  const int kb = k / 64;
  const int total = tiles * kb;
  const int chunk = (total + grid - 1) / grid;
  TORCH_CHECK(chunk <= kb, "a CTA chunk must span <= 2 tiles (tiles <= grid)");
  TORCH_CHECK(ws.numel() >= grid * 2 * bn * 128 && cnt.numel() >= tiles);
  CUtensorMap tw, tx;
  make_map(&tw, w.data_ptr(), n, k, 128);
  make_map(&tx, x.data_ptr(), r, k, bn);
  Params p{reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
           ws.data_ptr<float>(),
           reinterpret_cast<unsigned*>(cnt.data_ptr()),
           stamps.numel() ? reinterpret_cast<unsigned long long*>(stamps.data_ptr()) : nullptr,
           n, k, r, bn, tiles, kb, chunk, total, static_cast<int>(stages),
           static_cast<int>(pf),
           out.stride(0)};
  const int smem = smem_bytes(bn, stages);
  static int configured = 0;
  if (smem > configured) {
    TORCH_CHECK(cudaFuncSetAttribute(stream_gemm_bf16,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     smem) == cudaSuccess,
                "smem attribute ", smem);
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
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, stream_gemm_bf16, tw, tx, p) == cudaSuccess,
              "launch: ", cudaGetErrorString(cudaGetLastError()));
}

}  // namespace k31sg

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &k31sg::run);
  m.def("smem_bytes", &k31sg::smem_bytes);
}
