// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Cluster split-K weight-streaming decode GEMM (Rubin, tcgen05), BF16:
//   out[R, N] = x[R, K] @ W[N, K]^T, R <= 256 token rows, FP32 accumulation, one rounding.
//
// * swap-AB: a 128-row weight tile is the MMA M operand, the R tokens the N operand;
// * one cluster of C CTAs per weight tile, CTA rank r streams the K blocks
//   [r kb / C, (r + 1) kb / C) of that tile: the K split never leaves the cluster;
// * die-local: the weight's tiles are homed per die (w0: tiles [0, T0) on die 0, w1:
//   tiles [T0, T) on die 1); the cluster leader reads %smid, claims the next tile of its
//   own die's queue (the other die's once its own is empty) and broadcasts it over
//   DSMEM, so every CTA streams memory homed on its own die;
// * several TMA issuing warps (stage i is issued by producer i % P, each on its own
//   SM-subpartition queue): one issuer cannot keep enough bytes in flight;
// * PDL: weight boxes for the whole ring before griddepcontrol.wait, tokens after;
// * reduce: each CTA leaves its FP32 partial [bn][128] in its own shared memory; after
//   a cluster barrier rank r sums its token rows over ranks 0..C-1 in rank order
//   (deterministic) from DSMEM, rounds once and stores BF16.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>

namespace k31cg {

constexpr int kThreads = 192;  // w0 TMA, w1 MMA + TMEM, w2-5 epilogue (w2, w3 also TMA)
constexpr int kWBox = 128 * 128;
constexpr int kMaxStages = 16;

struct Params {
  __nv_bfloat16* out;
  unsigned* cnt;  // [0] die-0 claims, [1] die-1 claims, [2] finished clusters
  const int* sm_die;  // die of every %smid (null: everything is die 0)
  unsigned long long* stamps;  // [grid][8] globaltimer ns or null
  int n, k, r, bn;
  int tiles, t0, kb, csize, stages, skb, nprod;
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
__device__ __forceinline__ void tma_2d_ef(uint32_t dst, const CUtensorMap* map, int c0,
                                          int c1, uint32_t bar) {
  asm volatile(
      "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      ".L2::cache_hint [%0], [%1, {%2, %3}], [%4], pol; }" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar)
      : "memory");
}
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
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;"
               ::: "memory");
}
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t o;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(addr), "r"(rank));
  return o;
}
__device__ __forceinline__ void st_cluster_u32(uint32_t addr, uint32_t v) {
  asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(addr), "r"(v) : "memory");
}
__device__ __forceinline__ float4 ld_cluster_f4(uint32_t addr) {
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "r"(addr)
               : "memory");
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
    cluster_gemm_bf16(const __grid_constant__ CUtensorMap tw0,
                      const __grid_constant__ CUtensorMap tw1,
                      const __grid_constant__ CUtensorMap tx, const Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* ring = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int xbytes = p.bn * 128;
  const int stage_bytes = p.skb * (kWBox + xbytes);
  const int part_bytes = p.bn * 128 * 4;
  const int ring_bytes = max(p.stages * stage_bytes, part_bytes);
  uint64_t* bars = reinterpret_cast<uint64_t*>(ring + ring_bytes);
  uint32_t* slot = reinterpret_cast<uint32_t*>(bars + 2 * kMaxStages + 1);  // [0] tmem [1] tile
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t rank = cluster_rank();
  const uint32_t ring_s = smem_u32(ring);
  const uint32_t full0 = smem_u32(bars), empty0 = smem_u32(bars + kMaxStages),
                 acc0 = smem_u32(bars + 2 * kMaxStages);

  if (threadIdx.x == 0) {
    asm volatile("griddepcontrol.launch_dependents;");
    STAMP(0);
    for (int s = 0; s < p.stages; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, 1);
    }
    mbar_init(acc0, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (rank == 0) {
      // Claim a tile of this die's queue, else of the other die's (each of the T
      // clusters gets exactly one tile), and broadcast it to the cluster.
      unsigned smid;
      asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
      const int die = p.sm_die ? p.sm_die[smid] : 0;
      const int t1n = p.tiles - p.t0;
      int tile;
      const unsigned a = atomicAdd(p.cnt + die, 1u);
      if (static_cast<int>(a) < (die ? t1n : p.t0)) {
        tile = die ? p.t0 + static_cast<int>(a) : static_cast<int>(a);
      } else {
        const unsigned b = atomicAdd(p.cnt + (1 - die), 1u);
        tile = die ? static_cast<int>(b) : p.t0 + static_cast<int>(b);
      }
      if (p.stamps) p.stamps[blockIdx.x * 8 + 6] = (static_cast<unsigned long long>(die) << 32) | tile;
      slot[1] = tile;
    }
  }
  const int cols = p.bn <= 32 ? 32 : p.bn <= 64 ? 64 : p.bn <= 128 ? 128 : 256;
  if (warp == 1) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
            smem_u32(slot)),
        "r"(cols));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  cluster_sync();  // barrier init + the leader's tile visible cluster-wide
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t tmem = slot[0];
  uint32_t tile_u;
  asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(tile_u) : "r"(mapa(smem_u32(slot + 1), 0)) : "memory");
  const int tile = static_cast<int>(tile_u);
  const bool hi = tile >= p.t0;
  const CUtensorMap* tw = hi ? &tw1 : &tw0;
  const int wrow = (hi ? tile - p.t0 : tile) * 128;
  const int kb_lo = (static_cast<int>(rank) * p.kb) / p.csize;
  const int kb_hi = ((static_cast<int>(rank) + 1) * p.kb) / p.csize;
  const int nst = (kb_hi - kb_lo + p.skb - 1) / p.skb;  // ring iterations

  // Producers: warps 0, 2, 3 (distinct SM subpartitions), lane 0 each.
  int prod = -1;
  if (lane == 0) prod = warp == 0 ? 0 : warp == 2 ? 1 : warp == 3 ? 2 : -1;
  if (prod >= p.nprod) prod = -1;
  if (prod >= 0) {
    const int pre = min(nst, p.stages);
    for (int it = prod; it < pre; it += p.nprod) {
      const int k0 = kb_lo + it * p.skb, nk = min(p.skb, kb_hi - k0);
      const uint32_t st = ring_s + it * stage_bytes;
      mbar_expect_tx(full0 + 8 * it, nk * (kWBox + xbytes));
      for (int j = 0; j < nk; ++j)
        tma_2d_ef(st + j * kWBox, tw, (k0 + j) * 64, wrow, full0 + 8 * it);
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (prod == 0) STAMP(1);
    for (int it = prod; it < pre; it += p.nprod) {
      const int k0 = kb_lo + it * p.skb, nk = min(p.skb, kb_hi - k0);
      const uint32_t st = ring_s + it * stage_bytes + p.skb * kWBox;
      for (int j = 0; j < nk; ++j) tma_2d(st + j * xbytes, &tx, (k0 + j) * 64, 0, full0 + 8 * it);
    }
    for (int it = pre + ((prod - pre) % p.nprod + p.nprod) % p.nprod; it < nst; it += p.nprod) {
      const int s = it % p.stages;
      mbar_wait(empty0 + 8 * s, ((it / p.stages) & 1) ^ 1);
      const int k0 = kb_lo + it * p.skb, nk = min(p.skb, kb_hi - k0);
      const uint32_t st = ring_s + s * stage_bytes;
      mbar_expect_tx(full0 + 8 * s, nk * (kWBox + xbytes));
      for (int j = 0; j < nk; ++j)
        tma_2d_ef(st + j * kWBox, tw, (k0 + j) * 64, wrow, full0 + 8 * s);
      for (int j = 0; j < nk; ++j)
        tma_2d(st + p.skb * kWBox + j * xbytes, &tx, (k0 + j) * 64, 0, full0 + 8 * s);
    }
  } else if (warp == 1 && lane == 0) {
    const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) |
                           (static_cast<uint32_t>(p.bn >> 3) << 17) | (8u << 24);
    for (int it = 0; it < nst; ++it) {
      const int s = it % p.stages;
      mbar_wait(full0 + 8 * s, (it / p.stages) & 1);
      if (it + 1 == nst) STAMP(7);
      asm volatile("tcgen05.fence::after_thread_sync;");
      const int nk = min(p.skb, kb_hi - (kb_lo + it * p.skb));
      const uint32_t st = ring_s + s * stage_bytes;
      for (int j = 0; j < nk; ++j) {
        const uint64_t a = smem_desc(st + j * kWBox);
        const uint64_t bd = smem_desc(st + p.skb * kWBox + j * xbytes);
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
          mma_bf16(tmem, a + 2 * kk, bd + 2 * kk, idesc,
                   (it == 0 && j == 0 && kk == 0) ? 0u : 1u);
      }
      mma_commit(empty0 + 8 * s);
    }
    mma_commit(acc0);
  }
  __syncwarp();
  asm volatile("griddepcontrol.wait;" ::: "memory");  // before any global write
  if (warp >= 2) {
    // TMEM -> own shared memory partial [bn][128] FP32 (thread = weight row).
    const int q = warp & 3, row = q * 32 + lane;
    mbar_wait(acc0, 0);
    if (threadIdx.x == 64) STAMP(2);
    asm volatile("tcgen05.fence::after_thread_sync;");
    float* part = reinterpret_cast<float*>(ring);
    const uint32_t ta = tmem + (static_cast<uint32_t>(q * 32) << 16);
    for (int c = 0; c < p.bn; c += 16) {
      float v[16];
      tmem_ld16(ta + c, v);
#pragma unroll
      for (int j = 0; j < 16; ++j) part[(c + j) * 128 + row] = v[j];
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  cluster_sync();  // every rank's partial is in its shared memory
  if (threadIdx.x == 64) STAMP(3);
  {
    // Rank r: token rows [r R / C, (r + 1) R / C), summed over ranks in order.
    const int t0 = (static_cast<int>(rank) * p.r) / p.csize;
    const int t1 = ((static_cast<int>(rank) + 1) * p.r) / p.csize;
    const int nf = (t1 - t0) * 32;
    const uint32_t base = ring_s;
    for (int f = threadIdx.x; f < nf; f += kThreads) {
      const int t = t0 + (f >> 5), r4 = (f & 31) * 4;
      const uint32_t off = base + (t * 128 + r4) * 4;
      float4 acc = ld_cluster_f4(mapa(off, 0));
      for (int c = 1; c < p.csize; ++c) {
        const float4 v = ld_cluster_f4(mapa(off, c));
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
      }
      const int n0 = tile * 128 + r4;
      if (n0 < p.n) {
        __nv_bfloat162 lo = __floats2bfloat162_rn(acc.x, acc.y);
        __nv_bfloat162 hi2 = __floats2bfloat162_rn(acc.z, acc.w);
        uint2 pk;
        pk.x = *reinterpret_cast<uint32_t*>(&lo);
        pk.y = *reinterpret_cast<uint32_t*>(&hi2);
        *reinterpret_cast<uint2*>(p.out + t * p.ldo + n0) = pk;
      }
    }
  }
  cluster_sync();  // peers are done reading this CTA's partial
  if (threadIdx.x == 64) STAMP(5);
  if (threadIdx.x == 0 && rank == 0) {
    const unsigned d = atomicAdd(p.cnt + 2, 1u);
    if (static_cast<int>(d) == p.tiles - 1) {  // last cluster: reset for the next call
      p.cnt[0] = 0;
      p.cnt[1] = 0;
      p.cnt[2] = 0;
    }
  }
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

int smem_bytes(int bn, int stages, int skb) {
  const int ring = std::max(stages * skb * (kWBox + bn * 128), bn * 128 * 4);
  return 1024 + ring + (2 * kMaxStages + 1) * 8 + 16;
}

// w0: rows [0, t0 * 128) of the logical weight (die 0 tiles), w1: the rest (die 1
// tiles; may alias w0 when t0 == tiles). cnt: uint32 [3], zero at first use, owned
// by this call site. sm_die: int32 per smid or empty.
void run(torch::Tensor w0, torch::Tensor w1, int64_t t0, torch::Tensor x,
         torch::Tensor out, torch::Tensor cnt, torch::Tensor sm_die, int64_t csize,
         int64_t stages, int64_t skb, int64_t nprod, torch::Tensor stamps) {
  TORCH_CHECK(w0.dtype() == torch::kBFloat16 && x.dtype() == torch::kBFloat16 &&
              out.dtype() == torch::kBFloat16);
  TORCH_CHECK(w0.is_contiguous() && w1.is_contiguous() && x.is_contiguous() &&
              out.stride(1) == 1);
  const int k = w0.size(1), r = x.size(0), n = out.size(1);
  const int tiles = (n + 127) / 128;
  TORCH_CHECK(t0 >= 0 && t0 <= tiles);
  TORCH_CHECK(t0 == tiles || w0.size(0) == t0 * 128, "w0 rows = t0 x 128");
  TORCH_CHECK(t0 == tiles || w1.size(0) == n - t0 * 128, "w1 rows = N - t0 x 128");
  TORCH_CHECK(x.size(1) == k && out.size(0) == r && w1.size(1) == k);
  TORCH_CHECK(k % 64 == 0 && n % 4 == 0 && r >= 1 && r <= 256);
  TORCH_CHECK(stages >= 2 && stages <= kMaxStages && skb >= 1 && skb <= 4);
  TORCH_CHECK(nprod >= 1 && nprod <= 3 && csize >= 1 && csize <= 16);
  const int bn = (r + 15) / 16 * 16;
  const int kb = k / 64;
  TORCH_CHECK(kb >= csize && r >= csize);
  CUtensorMap tw0, tw1, tx;
  make_map(&tw0, w0.data_ptr(), w0.size(0), k, 128);
  make_map(&tw1, w1.data_ptr(), w1.size(0), k, 128);
  make_map(&tx, x.data_ptr(), r, k, bn);
  Params p{reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
           reinterpret_cast<unsigned*>(cnt.data_ptr()),
           sm_die.numel() ? sm_die.data_ptr<int>() : nullptr,
           stamps.numel() ? reinterpret_cast<unsigned long long*>(stamps.data_ptr()) : nullptr,
           n, k, r, bn, tiles, static_cast<int>(t0), kb, static_cast<int>(csize),
           static_cast<int>(stages), static_cast<int>(skb), static_cast<int>(nprod),
           out.stride(0)};
  const int smem = smem_bytes(bn, stages, skb);
  static int configured = 0;
  if (smem > configured) {
    TORCH_CHECK(cudaFuncSetAttribute(cluster_gemm_bf16,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     smem) == cudaSuccess, "smem attribute ", smem);
    TORCH_CHECK(cudaFuncSetAttribute(cluster_gemm_bf16,
                                     cudaFuncAttributeNonPortableClusterSizeAllowed,
                                     1) == cudaSuccess);
    configured = smem;
  }
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(static_cast<unsigned>(tiles * csize));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  at[1].id = cudaLaunchAttributeClusterDimension;
  at[1].val.clusterDim.x = static_cast<unsigned>(csize);
  at[1].val.clusterDim.y = 1;
  at[1].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 2;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, cluster_gemm_bf16, tw0, tw1, tx, p) == cudaSuccess,
              "launch: ", cudaGetErrorString(cudaGetLastError()));
}

// Co-resident clusters of csize CTAs at this smem size (cudaOccupancyMaxActiveClusters).
int max_clusters(int64_t csize, int64_t smem) {
  cudaFuncSetAttribute(cluster_gemm_bf16, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(smem));
  cudaFuncSetAttribute(cluster_gemm_bf16, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(static_cast<unsigned>(csize * 64));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = static_cast<size_t>(smem);
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = static_cast<unsigned>(csize);
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  int n = -1;
  cudaError_t e = cudaOccupancyMaxActiveClusters(&n, cluster_gemm_bf16, &cfg);
  return e == cudaSuccess ? n : -static_cast<int>(e);
}

}  // namespace k31cg

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &k31cg::run);
  m.def("smem_bytes", &k31cg::smem_bytes);
  m.def("max_clusters", &k31cg::max_clusters);
}
