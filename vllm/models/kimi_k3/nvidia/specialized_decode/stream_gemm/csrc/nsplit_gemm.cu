// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// N-partitioned weight-stationary decode GEMM (Rubin, tcgen05), no cross-SM reduce:
//   out[R, N] = x[R, K] @ W[N, K]^T, R <= 128 token rows, FP32 accumulation, one rounding.
//
// * CTA t owns output columns [t Nc, (t + 1) Nc) over the full K: no split, no reduce;
// * tokens are the MMA M operand (M = 128, rows >= R zero-filled by TMA), the Nc weight
//   rows the N operand (Nc % 16 == 0);
// * PDL: the CTA's weight slice is TMA-loaded into shared memory BEFORE
//   griddepcontrol.wait (as much as the ring holds; the rest streams behind it);
//   after the wait only the activation K-blocks are loaded;
// * NA TMEM accumulators interleave the K-blocks (no single serial accumulation chain),
//   summed in the epilogue in a fixed order;
// * dtype: BF16 (kind::f16) or FP8 E4M3 (kind::f8f6f4, unscaled: timing proxy of the
//   MXFP8 operand stream).
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstdint>

namespace k31ns {

constexpr int kThreads = 192;  // w0 W-TMA, w1 MMA, w2-5 epilogue (w2, w3 W-TMA; w5 x-TMA)
constexpr int kMaxW = 64;
constexpr int kMaxX = 8;
constexpr int kXBox = 128 * 128;  // 128 token rows x 128 B of K

struct Params {
  __nv_bfloat16* out;
  unsigned long long* stamps;  // [grid][8] or null
  int n, r, nc, kb, wg, wst, xg, xst, na, nprod, nxp, fp8, rotq, mode, xpriv;
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
__device__ __forceinline__ void tma_3d(uint32_t dst, const CUtensorMap* map, int c0,
                                       int c1, int c2, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tma_3d_ef(uint32_t dst, const CUtensorMap* map, int c0,
                                          int c1, int c2, uint32_t bar) {
  asm volatile(
      "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      ".L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], pol; }" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(bar)
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
// K-major, 128 B swizzle, 8-row groups of 1024 B.
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
__device__ __forceinline__ void mma_fp8(uint32_t tmem, uint64_t a, uint64_t b,
                                        uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %4, 0; "
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p; }" ::"r"(tmem),
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
__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define STAMP(i) \
  do { \
    if (p.stamps) p.stamps[blockIdx.x * 8 + (i)] = globaltimer(); \
  } while (0)

__host__ __device__ inline int wstage_bytes(int nc, int wg) { return wg * nc * 128; }

__global__ void __launch_bounds__(kThreads, 1)
    nsplit_gemm(const __grid_constant__ CUtensorMap tw, const __grid_constant__ CUtensorMap tx,
                const Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* base = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int wsb = wstage_bytes(p.nc, p.wg);
  const int wring = p.wst * wsb;
  const int xsb = p.xg * kXBox;
  uint8_t* xring = base;
  uint8_t* wbase = base + p.xst * xsb;
  uint64_t* bars = reinterpret_cast<uint64_t*>(wbase + wring);
  uint32_t* slot = reinterpret_cast<uint32_t*>(bars + 2 * kMaxW + 2 * kMaxX + 1);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t w_s = smem_u32(wbase), x_s = smem_u32(xring);
  const uint32_t fw0 = smem_u32(bars), ew0 = fw0 + 8 * kMaxW, fx0 = ew0 + 8 * kMaxW,
                 ex0 = fx0 + 8 * kMaxX, acc0 = ex0 + 8 * kMaxX;
  const int n0 = blockIdx.x * p.nc;

  if (threadIdx.x == 0) {
    asm volatile("griddepcontrol.launch_dependents;");
    STAMP(0);
    for (int s = 0; s < p.wst; ++s) {
      mbar_init(fw0 + 8 * s, 1);
      mbar_init(ew0 + 8 * s, 1);
    }
    for (int s = 0; s < p.xst; ++s) {
      mbar_init(fx0 + 8 * s, 1);
      mbar_init(ex0 + 8 * s, 1);
    }
    mbar_init(acc0, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  int need = p.na * p.nc;
  const int cols = need <= 32 ? 32 : need <= 64 ? 64 : need <= 128 ? 128 : need <= 256 ? 256 : 512;
  if (warp == 1) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
            smem_u32(slot)),
        "r"(cols));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t tmem = slot[0];
  const int kel = p.fp8 ? 128 : 64;  // K elements per 128 B block
  const int nws = (p.kb + p.wg - 1) / p.wg;
  const int nxs = (p.kb + p.xg - 1) / p.xg;
  // K rotation (rotq = K-block quantum, 0 = off): CTAs start at different K blocks so
  // they do not all read the same activation lines from L2 at the same time.
  const int rotb = p.rotq ? ((blockIdx.x * (p.kb / p.rotq) / gridDim.x) * p.rotq) % p.kb : 0;

  // W issuers: lanes 0, 8, 16 of warp 0 (the slice is mostly pre-staged); x issuers:
  // lane 0 of warps 2..5 (one per SM subpartition: TMA issue rate scales with warps).
  int prod = -1;
  if (warp == 0 && (lane & 7) == 0 && lane < 8 * p.nprod) prod = lane >> 3;
  if (prod >= 0) {
    // Weight stages never wait for the predecessor grid.
    for (int it = 0; it < nws; ++it) {
      const int s = it % p.wst;
      if (s % p.nprod != prod) continue;
      if (it >= p.wst) mbar_wait(ew0 + 8 * s, ((it / p.wst) & 1) ^ 1);
      mbar_expect_tx(fw0 + 8 * s, wsb);
      tma_3d_ef(w_s + s * wsb, &tw, 0, n0, (it * p.wg + rotb) % p.kb, fw0 + 8 * s);
    }
  } else if (warp >= 2 && lane == 0 && warp - 2 < p.nxp) {
    const int xid = warp - 2;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (xid == 0) STAMP(1);
    for (int g = 0; g < nxs; ++g) {
      const int s = g % p.xst;
      if (s % p.nxp != xid) continue;
      if (g >= p.xst) mbar_wait(ex0 + 8 * s, ((g / p.xst) & 1) ^ 1);
      mbar_expect_tx(fx0 + 8 * s, xsb);
      tma_3d(x_s + s * xsb, &tx, 0, p.xpriv * blockIdx.x, (g * p.xg + rotb) % p.kb, fx0 + 8 * s);
    }
  } else if (warp == 1 && lane == 0) {
    const uint32_t idesc = p.fp8 ? ((1u << 4) | (static_cast<uint32_t>(p.nc >> 3) << 17) |
                                    (8u << 24))
                                 : ((1u << 4) | (1u << 7) | (1u << 10) |
                                    (static_cast<uint32_t>(p.nc >> 3) << 17) | (8u << 24));
    for (int kbi = 0; kbi < p.kb; ++kbi) {
      const int iw = kbi / p.wg, jw = kbi % p.wg, ix = kbi / p.xg, jx = kbi % p.xg;
      const int sw = iw % p.wst, sx = ix % p.xst;
      if (jx == 0) mbar_wait(fx0 + 8 * sx, (ix / p.xst) & 1);
      if (jw == 0) mbar_wait(fw0 + 8 * sw, (iw / p.wst) & 1);
      if (kbi + 1 == p.kb) STAMP(7);
      if (kbi == 0) STAMP(4);
      if (kbi == p.kb / 2) STAMP(5);
      asm volatile("tcgen05.fence::after_thread_sync;");
      const uint64_t a = smem_desc(x_s + sx * xsb + jx * kXBox);
      const uint64_t b = smem_desc(w_s + sw * wsb + jw * p.nc * 128);
      const int ai = kbi % p.na;
      const uint32_t t = tmem + static_cast<uint32_t>(ai * p.nc);
      const uint32_t first = kbi < p.na ? 1u : 0u;
      if (p.mode & 1) {
      } else if (p.fp8) {
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
          mma_fp8(t, a + 2 * kk, b + 2 * kk, idesc, (first && kk == 0) ? 0u : 1u);
      } else {
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
          mma_bf16(t, a + 2 * kk, b + 2 * kk, idesc, (first && kk == 0) ? 0u : 1u);
      }
      if (jw == p.wg - 1 || kbi + 1 == p.kb) mma_commit(ew0 + 8 * sw);
      if (jx == p.xg - 1 || kbi + 1 == p.kb) mma_commit(ex0 + 8 * sx);
    }
    mma_commit(acc0);
  }
  __syncwarp();
  asm volatile("griddepcontrol.wait;" ::: "memory");  // before any global write
  if (warp >= 2) {
    const int q = warp & 3, row = q * 32 + lane;  // TMEM lane = token row
    mbar_wait(acc0, 0);
    if (threadIdx.x == 64) STAMP(2);
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t ta = tmem + (static_cast<uint32_t>(q * 32) << 16);
    const int na = min(p.na, p.kb);
    for (int c = 0; c < p.nc; c += 16) {
      float acc[16], v[16];
      tmem_ld16(ta + c, acc);
      for (int a = 1; a < na; ++a) {
        tmem_ld16(ta + a * p.nc + c, v);
#pragma unroll
        for (int j = 0; j < 16; ++j) acc[j] += v[j];
      }
      if (row < p.r && n0 + c < p.n) {
        __nv_bfloat16* o = p.out + row * p.ldo + n0 + c;
        if (n0 + c + 16 <= p.n) {
          uint32_t pk[8];
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            __nv_bfloat162 h = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
            pk[j] = *reinterpret_cast<uint32_t*>(&h);
          }
          reinterpret_cast<uint4*>(o)[0] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
          reinterpret_cast<uint4*>(o)[1] = make_uint4(pk[4], pk[5], pk[6], pk[7]);
        } else {
          for (int j = 0; j < 16 && n0 + c + j < p.n; ++j) o[j] = __float2bfloat16_rn(acc[j]);
        }
      }
    }
    if (threadIdx.x == 64) STAMP(3);
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (warp == 1) {
    asm volatile("tcgen05.fence::after_thread_sync;");
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem),
                 "r"(cols));
  }
}

// 3D view [kb][rows][128 B]: box = depth K-blocks of box_rows x 128 B (one TMA op).
static void make_map(CUtensorMap* m, const void* ptr, int rows, int kbytes, int box_rows,
                     int depth, bool fp8) {
  const int esz = fp8 ? 1 : 2;
  cuuint64_t dims[3] = {static_cast<cuuint64_t>(128 / esz), static_cast<cuuint64_t>(rows),
                        static_cast<cuuint64_t>(kbytes / 128)};
  cuuint64_t strides[2] = {static_cast<cuuint64_t>(kbytes), 128};
  cuuint32_t box[3] = {static_cast<cuuint32_t>(128 / esz), static_cast<cuuint32_t>(box_rows),
                       static_cast<cuuint32_t>(depth)};
  cuuint32_t es[3] = {1, 1, 1};
  CUresult rc = cuTensorMapEncodeTiled(
      m, fp8 ? CU_TENSOR_MAP_DATA_TYPE_UINT8 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
      const_cast<void*>(ptr), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(rc == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", static_cast<int>(rc));
}

int smem_bytes(int nc, int wg, int wst, int xg, int xst) {
  return 1024 + wst * wstage_bytes(nc, wg) + xst * xg * kXBox + (2 * kMaxW + 2 * kMaxX + 1) * 8 +
         16;
}
// Operand rings must stay below 256 KiB (14-bit smem descriptor address).
int operand_bytes(int nc, int wg, int wst, int xg, int xst) {
  return 1024 + wst * wstage_bytes(nc, wg) + xst * xg * kXBox;
}

static void set_attrs(int smem) {
  if (smem > 232448) {
    TORCH_CHECK(cudaFuncSetAttribute(nsplit_gemm, cudaFuncAttributeSharedMemoryMode,
                                     cudaSharedMemoryModeAllowOversizedSharedMemory) ==
                    cudaSuccess,
                "oversized shared memory mode");
  }
  if (cudaFuncSetAttribute(nsplit_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) !=
      cudaSuccess) {
    (void)cudaGetLastError();
    TORCH_CHECK(smem > 232448 &&
                    cudaFuncSetAttribute(nsplit_gemm,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         232448) == cudaSuccess,
                "smem attribute ", smem);
  }
}

// w [N, K] (bf16 or float8_e4m3fn), x [R, K] same dtype, out [R, N] bf16.
void run(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t nc, int64_t wg,
         int64_t wst, int64_t xg, int64_t xst, int64_t na, int64_t nprod, int64_t nxp,
         int64_t rotq, int64_t mode, int64_t xpriv, torch::Tensor stamps) {
  const bool fp8 = w.scalar_type() == torch::kFloat8_e4m3fn;
  TORCH_CHECK(fp8 || w.dtype() == torch::kBFloat16);
  TORCH_CHECK(x.scalar_type() == w.scalar_type() && out.dtype() == torch::kBFloat16);
  TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && out.stride(1) == 1);
  const int k = w.size(1), r = out.size(0), n = out.size(1);
  const int kbytes = k * (fp8 ? 1 : 2);
  const int xrows = x.size(0);
  TORCH_CHECK(kbytes % 128 == 0 && r >= 1 && r <= 128 && w.size(0) == n && x.size(1) == k);
  TORCH_CHECK(nc % 16 == 0 && nc >= 16 && nc <= 256 && n % 8 == 0);
  TORCH_CHECK(wst >= 1 && wst <= kMaxW && xst >= 1 && xst <= kMaxX && wg >= 1 && xg >= 1);
  TORCH_CHECK(nprod >= 1 && nprod <= 3 && nxp >= 1 && nxp <= 4 && na >= 1 && na * nc <= 512);
  TORCH_CHECK(operand_bytes(nc, wg, wst, xg, xst) <= 262144, "operand rings > 256 KiB");
  TORCH_CHECK(wg <= 256 && xg <= 256);
  TORCH_CHECK(rotq == 0 || ((kbytes / 128) % rotq == 0 && rotq % wg == 0 && rotq % xg == 0),
              "rotation quantum must divide kb and be a multiple of wg and xg");
  const int kb = kbytes / 128;
  CUtensorMap tw, tx;
  make_map(&tw, w.data_ptr(), n, kbytes, static_cast<int>(nc), static_cast<int>(wg), fp8);
  make_map(&tx, x.data_ptr(), xrows, kbytes, 128, static_cast<int>(xg), fp8);
  Params p{reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
           stamps.numel() ? reinterpret_cast<unsigned long long*>(stamps.data_ptr()) : nullptr,
           n, r, static_cast<int>(nc), kb, static_cast<int>(wg), static_cast<int>(wst),
           static_cast<int>(xg), static_cast<int>(xst), static_cast<int>(na),
           static_cast<int>(nprod), static_cast<int>(nxp), fp8 ? 1 : 0, static_cast<int>(rotq),
           static_cast<int>(mode), static_cast<int>(xpriv), out.stride(0)};
  const int smem = smem_bytes(nc, wg, wst, xg, xst);
  static int configured = 0;
  if (smem > configured) {
    set_attrs(smem);
    configured = smem;
  }
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(static_cast<unsigned>((n + nc - 1) / nc));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  int na_ = 1;
  if (smem > 232448) {
    at[1].id = cudaLaunchAttributeSharedMemoryMode;
    at[1].val.sharedMemoryMode = cudaSharedMemoryModeAllowOversizedSharedMemory;
    na_ = 2;
  }
  cfg.attrs = at;
  cfg.numAttrs = na_;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, nsplit_gemm, tw, tx, p) == cudaSuccess,
              "launch: ", cudaGetErrorString(cudaGetLastError()));
}

}  // namespace k31ns

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &k31ns::run);
  m.def("smem_bytes", &k31ns::smem_bytes);
  m.def("operand_bytes", &k31ns::operand_bytes);
}
