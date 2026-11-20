# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D-MOEFRONT M4a: swap-AB TS tcgen05 MMA for the decode MoE front with the WEIGHT as the
A operand held in TMEM (loaded by tcgen05.cp from SW128 K-major smem) and the tokens x as
the B operand in SW128 K-major smem: D[128 W rows, n] += W[128, K] . x[n, K]^T.

Checks the result against torch (FP32 accumulate) and times the MMA sweep per CTA on all
SMs (clock64 from first issue to commit completion) for n = 32/64/96 and K = nkb x 64.

python ts_mma_probe.py [--nkb 7] [--reps 50]
"""

import argparse
import os

import torch
from torch.utils.cpp_extension import load_inline

_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void bar_init(uint32_t b, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(c)); }
__device__ __forceinline__ void bar_expect(uint32_t b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void bar_wait(uint32_t b, uint32_t par) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}" :: "r"(b), "r"(par) : "memory"); }
__device__ __forceinline__ void tma(uint32_t dst, const void* src, uint32_t bytes, uint32_t b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory"); }
// K-major SWIZZLE_128B descriptor (8-row core groups 1024 B apart), as moe_front.
__device__ __forceinline__ uint64_t sdesc(uint32_t a) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 46) | (2ull << 61); }

// smem: [bars 1024][W image nkb x 16 KiB][x image nkb x n x 128 B]
__global__ void __launch_bounds__(128, 1) ts_mma(const char* wimg, const char* ximg, float* out, int n, int nkb, int reps,
                                                  long long* clk, int mode) {
  extern __shared__ __align__(1024) char sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t bw = sa(sm), bx = bw + 8, bc = bw + 16, bd = bw + 24;
  uint32_t* tptr = (uint32_t*)(sm + 64);
  char* ws = sm + 1024; char* xs = ws + nkb * 16384;
  const int cta = blockIdx.x;
  const char* wsrc = wimg + (long long)(mode ? 0 : cta) * nkb * 16384;
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(sa(tptr)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  if (tid == 0) { bar_init(bw, 1); bar_init(bx, 1); bar_init(bc, 1); bar_init(bd, 1); asm volatile("fence.mbarrier_init.release.cluster;"); }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t tb = *tptr, tw = tb + 256;   // D at columns [0, n), W at [256, 256 + nkb * 32)
  const uint32_t xbytes = (uint32_t)n * 128;
  if (tid == 0) {
    bar_expect(bw, nkb * 16384); tma(sa(ws), wsrc, nkb * 16384, bw);
    bar_expect(bx, nkb * xbytes); tma(sa(xs), ximg, nkb * xbytes, bx);
    bar_wait(bw, 0);
    for (int kb = 0; kb < nkb; ++kb)
#pragma unroll
      for (int kk = 0; kk < 4; ++kk)
        asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;" :: "r"(tw + kb * 32 + kk * 8), "l"(sdesc(sa(ws) + kb * 16384 + kk * 32)) : "memory");
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bc) : "memory");
    bar_wait(bc, 0);
    bar_wait(bx, 0);
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  // kind::f16, BF16 A/B, FP32 D, A from TMEM, B K-major, M = 128, N = n
  const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(n >> 3) << 17) | (8u << 24);
  long long c0 = 0, c1 = 0;
  if (warp == 0) {
    for (int r = 0; r < reps; ++r) {
      if (r == 1) c0 = clock64();
      for (int kb = 0; kb < nkb; ++kb) {
        const uint64_t b0 = sdesc(sa(xs) + kb * xbytes);
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const uint32_t en = (kb | kk) ? 1u : 0u;
          asm volatile("{ .reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; "
                       "@P tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, Q; }"
                       :: "r"(tb), "r"(tw + kb * 32 + kk * 8), "l"(b0 + 2 * kk), "r"(idesc), "r"(en));
        }
      }
      asm volatile("{ .reg .pred P; elect.sync _|P, 0xffffffff; "
                   "@P tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0]; }" :: "r"(bd) : "memory");
      bar_wait(bd, r & 1);
    }
    c1 = clock64();
  }
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  __syncthreads();
  // D lane = W row (warp w owns lanes 32w..32w+31), column = token
  const uint32_t ta = tb + ((uint32_t)(warp * 32) << 16);
  for (int c = 0; c < n; c += 16) {
    uint32_t v[16];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
                   "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15])
                 : "r"(ta + c));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    if (cta == 0)
      for (int j = 0; j < 16; ++j) out[(warp * 32 + lane) * n + c + j] = __uint_as_float(v[j]);
  }
  if (tid == 0 && clk) clk[cta] = reps > 1 ? (c1 - c0) / (reps - 1) : 0;
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tb) : "memory");
}

void run(torch::Tensor wimg, torch::Tensor ximg, torch::Tensor out, int64_t n, int64_t nkb, int64_t reps, torch::Tensor clk,
         int64_t grid, int64_t mode) {
  const size_t smem = 1024 + nkb * 16384 + nkb * n * 128;
  TORCH_CHECK(smem <= 232448 && n % 16 == 0 && n <= 256 && nkb <= 8);
  TORCH_CHECK(cudaFuncSetAttribute(ts_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess);
  ts_mma<<<grid, 128, smem, c10::cuda::getCurrentCUDAStream()>>>((const char*)wimg.data_ptr(), (const char*)ximg.data_ptr(),
      out.data_ptr<float>(), n, nkb, reps, clk.numel() ? reinterpret_cast<long long*>(clk.data_ptr()) : nullptr, mode);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
"""


def _module():
    root = os.environ.get("VLLM_CACHE_ROOT", "/tmp")
    build = os.path.join(root, "k3_ts_mma_probe")
    os.makedirs(build, exist_ok=True)
    return load_inline(
        name="k3_ts_mma_probe",
        cpp_sources="void run(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, int64_t, int64_t);",
        cuda_sources=_SRC, functions=["run"],
        extra_cuda_cflags=["-O3", "-gencode", "arch=compute_100f,code=sm_100f"],
        build_directory=build, verbose=False,
    )  # fmt: skip


def sw128_image(t: torch.Tensor) -> torch.Tensor:
    """[R, K] bf16 (R % 8 == 0, K % 64 == 0) -> K-major SWIZZLE_128B smem image as bytes:
    K-block kb (64 elements) of all R rows is one R x 128 B region; 16 B chunk c of row r is
    stored at chunk c ^ (r % 8)."""
    r, k = t.shape
    blk = t.view(r, k // 64, 8, 8).permute(1, 0, 2, 3).contiguous()  # [kb, r, chunk, 8 elems]
    idx = torch.arange(8, device=t.device).view(1, 8) ^ (torch.arange(r, device=t.device) % 8).view(r, 1)
    out = torch.empty_like(blk)
    out.scatter_(2, idx.view(1, r, 8, 1).expand_as(blk), blk)
    return out.contiguous().view(-1).view(torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nkb", type=int, default=7)
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    m = _module()
    dev = torch.device("cuda")
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    k = a.nkb * 64
    torch.manual_seed(0)
    for n in (32, 64, 96):
        w = (torch.randn(128, k, device=dev) * 0.05).to(torch.bfloat16)
        x = torch.randn(n, k, device=dev).to(torch.bfloat16)
        ref = w.float() @ x.float().t()
        out = torch.zeros(128, n, device=dev)
        clk = torch.zeros(nsm, dtype=torch.int64, device=dev)
        m.run(sw128_image(w), sw128_image(x), out, n, a.nkb, 1, clk, 1, 0)
        torch.cuda.synchronize()
        err = (out - ref).abs().max().item()
        scale = ref.abs().max().item()
        ok = err <= 1e-3 * max(scale, 1.0)
        # timing: every SM runs reps sweeps of nkb x 4 MMAs (M128 N n K16)
        m.run(sw128_image(w), sw128_image(x), out, n, a.nkb, a.reps, clk, nsm, 1)
        torch.cuda.synchronize()
        cyc = clk.float()
        per = cyc / (a.nkb * 4)
        print(f"n{n} K{k}: max|err| {err:.3e} (ref max {scale:.2f}) {'PASS' if ok else 'FAIL'} | "
              f"clk per sweep median {cyc.median().item():.0f} max {cyc.max().item():.0f} = "
              f"{per.median().item():.1f} clk per M128xN{n}xK16 MMA (FLOP floor {2 * 128 * n * 16 / 8192:.0f})")


if __name__ == "__main__":
    main()
