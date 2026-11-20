#!/usr/bin/env python3
"""P-GEMM R5 probe: shared-expert SiTU + MXFP8 quant with a bulk-copy (TMA 1D) smem ring.

situ4_row_kernel (production, one row per CTA) streams ~8 TB/s at M16384 d6144: every warp alternates a
short load phase with a ~2 us MUFU/FMA phase, and register prefetch (R4) halves occupancy (slower).
Here one thread per CTA bulk-copies whole rows (gate 2d bytes | up 2d bytes) into an S-stage smem ring
(mbarrier complete_tx), so HBM reads stay in flight independent of registers; the 192 consumer threads
compute exactly the production per-element math. Compared bitwise with mxfp8_act (production).
usage: pgemm_situ_tma_bench.py
"""
import json
import os

import torch

from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

KSRC = r"""
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int n) {
#if __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(n));
#else
  __trap();
#endif
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t tx) {
#if __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(tx)
               : "memory");
#else
  __trap();
#endif
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
#if __CUDA_ARCH__ >= 900
  asm volatile(
      "{\n .reg .pred p;\n LAB_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra LAB_WAIT;\n}\n" ::"r"(smem_u32(b)), "r"(parity) : "memory");
#else
  __trap();
#endif
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
#if __CUDA_ARCH__ >= 900
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)) : "memory");
#else
  __trap();
#endif
}

template <int A_POW, bool CLAMP, int S, bool ROT>
__global__ void __launch_bounds__(1024) situ4_tma_kernel(
    uint8_t* __restrict__ q, uint8_t* __restrict__ sf, const uint16_t* __restrict__ x,
    const int64_t rows, const int64_t rows_pad, const int d, const int64_t x_stride,
    const int64_t q_stride, const int sf_col_base, const int padded_cols, const float beta,
    const float linear_beta) {
  extern __shared__ __align__(128) uint8_t ring[];
  __shared__ __align__(8) uint64_t full[S];
  const float log2e = 1.4426950408889634f;
  const float kb = -2.0f * log2e / beta;
  const float kc = CLAMP ? -2.0f * log2e / linear_beta : 0.0f;
  const float bbs = CLAMP ? beta * linear_beta : beta;
  const float2 k_a = make_float2(-log2e, -log2e), k_b = make_float2(kb, kb);
  const float2 k_c = make_float2(kc, kc), bb = make_float2(bbs, bbs);
  const int blk = threadIdx.x;
  const uint32_t row_bytes = (uint32_t)d * 2u;  // gate (or up) half of one row
  if (blk == 0) {
    for (int s = 0; s < S; ++s) mbar_init(&full[s], 1);
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
  }
  __syncthreads();
  if (blk == 0) {
    for (int s = 0; s < S; ++s) {
      const int64_t r = blockIdx.x + (int64_t)s * gridDim.x;
      if (r < rows) {
        uint8_t* st = ring + (size_t)s * 2 * row_bytes;
        mbar_expect_tx(&full[s], 2 * row_bytes);
        bulk_g2s(st, x + r * x_stride, row_bytes, &full[s]);
        bulk_g2s(st + row_bytes, x + r * x_stride + d, row_bytes, &full[s]);
      }
    }
  }
  const int rot = ROT ? ((blk >> 1) & 3) : 0;
  int k = 0;
  for (int64_t row = blockIdx.x; row < rows_pad; row += gridDim.x, ++k) {
    const int64_t sf_off = swizzled_sf_offset(row, sf_col_base + blk, padded_cols);
    if (row >= rows) {
      sf[sf_off] = 0;
      continue;
    }
    const int s = k % S;
    mbar_wait(&full[s], (uint32_t)((k / S) & 1));
    const uint8_t* st = ring + (size_t)s * 2 * row_bytes;
    const uint4* gp = reinterpret_cast<const uint4*>(st + blk * 64);
    const uint4* up = reinterpret_cast<const uint4*>(st + row_bytes + blk * 64);
    uint4 a[4], b[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int j = (i + rot) & 3;
      a[i] = gp[j];
      b[i] = up[j];
    }
    __syncthreads();  // stage s fully read: refill it with row k + S
    if (blk == 0) {
      const int64_t nr = row + (int64_t)S * gridDim.x;
      if (nr < rows) {
        uint8_t* dst = ring + (size_t)s * 2 * row_bytes;
        mbar_expect_tx(&full[s], 2 * row_bytes);
        bulk_g2s(dst, x + nr * x_stride, row_bytes, &full[s]);
        bulk_g2s(dst + row_bytes, x + nr * x_stride + d, row_bytes, &full[s]);
      }
    }
    uint32_t h[16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      h[4 * i + 0] = situ4_pair_ct<A_POW, CLAMP>(a[i].x, b[i].x, k_a, k_b, k_c, bb);
      h[4 * i + 1] = situ4_pair_ct<A_POW, CLAMP>(a[i].y, b[i].y, k_a, k_b, k_c, bb);
      h[4 * i + 2] = situ4_pair_ct<A_POW, CLAMP>(a[i].z, b[i].z, k_a, k_b, k_c, bb);
      h[4 * i + 3] = situ4_pair_ct<A_POW, CLAMP>(a[i].w, b[i].w, k_a, k_b, k_c, bb);
    }
    __nv_bfloat162 m = __habs2(*reinterpret_cast<const __nv_bfloat162*>(&h[0]));
#pragma unroll
    for (int i = 1; i < 16; ++i)
      m = __hmax2(m, __habs2(*reinterpret_cast<const __nv_bfloat162*>(&h[i])));
    const float amax = fmaxf(__low2float(m), __high2float(m));
    const uint32_t e = float_to_ue8m0(amax * (1.0f / 448.0f));
    const float inv = ue8m0_to_inv_scale(e);
    const float2 inv2 = make_float2(inv, inv);
    uint8_t* qrow = q + row * q_stride + blk * 32;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 v0 = f2mul(
          __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&h[4 * i])), inv2);
      const float2 v1 = f2mul(
          __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&h[4 * i + 1])), inv2);
      const float2 v2 = f2mul(
          __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&h[4 * i + 2])), inv2);
      const float2 v3 = f2mul(
          __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&h[4 * i + 3])), inv2);
      uint2 w;
      w.x = (uint32_t)fp8x2(v0.x, v0.y) | ((uint32_t)fp8x2(v1.x, v1.y) << 16);
      w.y = (uint32_t)fp8x2(v2.x, v2.y) | ((uint32_t)fp8x2(v3.x, v3.y) << 16);
      *reinterpret_cast<uint2*>(qrow + ((i + rot) & 3) * 8) = w;
    }
    sf[sf_off] = (uint8_t)e;
  }
}

void situ_tma(torch::Tensor& q, torch::Tensor& sf, const torch::Tensor& x, int64_t d,
              int64_t sf_col_base, int64_t padded_cols, double beta, double linear_beta,
              int64_t stages, int64_t rot, int64_t grid_override) {
  const int64_t rows = x.size(0);
  const int64_t rows_pad = (rows + 127) / 128 * 128;
  TORCH_CHECK(beta == 4.0 && linear_beta == 25.0, "probe instantiates A_POW 2 + clamp only");
  auto stream = c10::cuda::getCurrentCUDAStream();
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device());
  const size_t smem = (size_t)stages * 4 * d;
#define TMA_LAUNCH(S, R)                                                                   \
  {                                                                                        \
    auto kern = situ4_tma_kernel<2, true, S, R>;                                           \
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);    \
    int occ = 0;                                                                           \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, (int)(d / 32), smem);        \
    int64_t grid = std::min<int64_t>(rows_pad, (int64_t)sms * std::max(occ, 1));           \
    if (grid_override > 0) grid = std::min<int64_t>(grid, grid_override);                  \
    kern<<<(unsigned)grid, (unsigned)(d / 32), smem, stream>>>(                            \
        reinterpret_cast<uint8_t*>(q.data_ptr()), reinterpret_cast<uint8_t*>(sf.data_ptr()), \
        reinterpret_cast<const uint16_t*>(x.data_ptr()), rows, rows_pad, (int)d,           \
        x.stride(0), q.stride(0), (int)sf_col_base, (int)padded_cols, (float)beta,         \
        (float)linear_beta);                                                               \
  }
  if (stages == 1 && rot) TMA_LAUNCH(1, true)
  else if (stages == 2 && rot) TMA_LAUNCH(2, true)
  else if (stages == 3 && rot) TMA_LAUNCH(3, true)
  else if (stages == 4 && rot) TMA_LAUNCH(4, true)
  else if (stages == 2) TMA_LAUNCH(2, false)
  else TORCH_CHECK(false, "bad stages");
#undef TMA_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

CPP = """
void situ_tma(torch::Tensor& q, torch::Tensor& sf, const torch::Tensor& x, int64_t d,
              int64_t sf_col_base, int64_t padded_cols, double beta, double linear_beta,
              int64_t stages, int64_t rot, int64_t grid_override);
"""


def main():
    from torch.utils.cpp_extension import load_inline

    build = os.path.join(os.environ.get("VLLM_CACHE_ROOT", "/tmp"), "pg_situ_tma_probe")
    os.makedirs(build, exist_ok=True)
    mod = load_inline(name="pg_situ_tma_probe", cpp_sources=CPP,
                      cuda_sources=mxfp8_act._CUDA_SRC + KSRC, functions=["situ_tma"],
                      extra_cuda_cflags=["-O3"], build_directory=build, verbose=False)
    inter = 6144
    for rows, off in [(16384, 0), (8192, 0), (16384, 3584), (4100, 0), (37, 0)]:
        torch.manual_seed(rows + off)
        src = (torch.randn(rows, off + 2 * inter, device="cuda") * 2).to(torch.bfloat16)
        x = src[:, off:]
        ref = mxfp8_act.Mxfp8Buffer(rows, inter, x.device, x.dtype)
        mxfp8_act.situ_and_mul_quantize_fast(x, 4.0, 25.0, out=ref, col=0)
        ra = ref.activation()
        rq, rs = ra.data.view(torch.uint8).clone(), ra.scale.view(torch.uint8).clone()

        def timeit(f):
            f()
            torch.cuda.synchronize()
            ts = []
            for _ in range(7):
                s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s0.record()
                for _ in range(20):
                    f()
                e0.record()
                torch.cuda.synchronize()
                ts.append(s0.elapsed_time(e0) * 1000 / 20)
            return round(sorted(ts)[3], 2)

        prod_us = timeit(lambda: mxfp8_act.situ_and_mul_quantize_fast(x, 4.0, 25.0, out=ref, col=0))
        r = {"rows": rows, "off": off, "prod_us": prod_us}
        for stages, rot, grid in [(1, 1, 0), (2, 1, 0), (3, 1, 0), (4, 1, 0), (2, 0, 0)]:
            buf = mxfp8_act.Mxfp8Buffer(rows, inter, x.device, x.dtype)
            buf.data.view(torch.uint8).fill_(0x5A)
            buf.scale.fill_(0x5A)
            qv, sv = buf.window(0, inter), buf.scale

            def f():
                mod.situ_tma(qv, sv, x, inter, 0, buf.padded_cols, 4.0, 25.0, stages, rot, grid)

            try:
                us = timeit(f)
                qa = buf.activation()
                same = bool(torch.equal(qa.data.view(torch.uint8), rq)
                            and torch.equal(qa.scale.view(torch.uint8), rs))
            except Exception as e:  # noqa: BLE001
                us, same = None, repr(e)[:120]
            r[f"S{stages}r{rot}"] = [us, same]
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
