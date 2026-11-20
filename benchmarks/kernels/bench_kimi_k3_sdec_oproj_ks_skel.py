# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D-ATTNRES S-RS skeleton: KDA o-projection with the SP reduce-scatter pushed as owner
blocks finish (K split 3 over 3-CTA clusters, DSMEM reduce-scatter, NVLink row pushes).

Data movement only (no MMA, no sums): per CTA (feature block f of 56 x 128 hidden
columns, K slice s of 3 x 1024): the W slice (128 KB) lands before the timed window (the
real kernel loads it before griddepcontrol.wait); then per owner block b (remote owners
first, own last): the x block (96 tokens x 1024 B, 8 x 12 KB 1D bulk loads), the FP32
partial thirds to the 2 peer K-slice CTAs (2 x 16 KB DSMEM bulk copies, mbarrier
complete_tx, 2 receive slots with credit barriers), and the owned third (32 tokens x 128
columns BF16) pushed to the owner GPU's receive rows (32 x 256 B bulk stores, Lamport
slot layout [rank][n][7168]). One process, 4 GPUs, P2P; a host-flag gate starts all GPUs
together. Prints per-GPU window (first CTA start after W -> last CTA end) per variant.

    python benchmarks/kernels/bench_kimi_k3_sdec_oproj_ks_skel.py --rows 384 --iters 20
"""

import argparse
import hashlib
import os

import torch

_CPP = r"""
#include <torch/extension.h>
void setup_peers(int64_t ngpu);
void launch(int64_t dev, int64_t w, int64_t x, std::vector<int64_t> peers, int64_t rank, int64_t n,
            int64_t mode, int64_t ts, int64_t flag, int64_t stream);
"""

_CU = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#define NF 56
#define KS 3
#define WB (128 * 1024)
#define XCH (12 * 1024)
#define XNCH 8
#define XB (XCH * XNCH)
#define PART (16 * 1024)
#define STG (8 * 1024)
#define SMEM (WB + 2 * XCH + 2 * 2 * PART + STG + 1024)
struct Prm { const char* w; const char* x; char* peers[4]; int rank, n, mode; unsigned long long* ts;
             const volatile int* flag; };
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ unsigned long long gt() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void bar_init(uint64_t* b, int c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(b)), "r"(c) : "memory"); }
__device__ __forceinline__ void expect(uint64_t* b, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(b)), "r"(bytes) : "memory"); }
__device__ __forceinline__ void bwait(uint64_t* b, uint32_t ph) {
  uint32_t ok = 0;
  while (!ok) asm volatile("{ .reg .pred q; mbarrier.try_wait.parity.shared::cta.b64 q, [%1], %2; selp.u32 %0, 1, 0, q; }"
                           : "=r"(ok) : "r"(sa(b)), "r"(ph) : "memory"); }
__device__ __forceinline__ void gload(void* dst, const void* src, int bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(b)) : "memory"); }
__device__ __forceinline__ uint32_t mapa(uint32_t a, int cta) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(cta)); return r; }
__device__ __forceinline__ void dsm_copy(uint32_t dst_cl, const void* src, int bytes, uint32_t bar_cl) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst_cl), "r"(sa(src)), "r"(bytes), "r"(bar_cl) : "memory"); }
__device__ __forceinline__ void remote_arrive(uint32_t bar_cl) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(bar_cl) : "memory"); }
__device__ __forceinline__ void gstore(char* dst, const void* src, int bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(dst), "r"(sa(src)), "r"(bytes) : "memory"); }
__device__ __forceinline__ unsigned cl_rank() { unsigned r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }

// mode bits: 1 x stream, 2 DSMEM reduce-scatter, 4 NVLink pushes
__global__ void __cluster_dims__(KS, 1, 1) __launch_bounds__(128, 1) skel(Prm p) {
  extern __shared__ __align__(1024) char sm[];
  char* W = sm; char* X = W + WB; char* R = X + 2 * XCH;  // X: 2-slot ring of 12 KB; R: 2 slots x 2 peers x 16 KB
  char* S = R + 4 * PART;
  uint64_t* bars = (uint64_t*)(S + STG);  // 0 w, 1-2 x slots, 3-4 recv slot, 5-6 credit slot
  volatile int* landed = (volatile int*)(bars + 8);
  const int s = (int)cl_rank(), f = blockIdx.y, n = p.n;
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int i = 0; i < 7; ++i) bar_init(bars + i, i >= 5 ? KS - 1 : 1);
    *landed = (p.mode & 1) ? 0 : 4;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  if (tid == 0) {
    const unsigned long long tg = gt();
    while (*p.flag == 0 && gt() - tg < 2000000000ull) {}
    expect(bars + 0, WB);
    for (int c = 0; c * XCH < WB; ++c) {
      int bytes = (WB - c * XCH) < XCH ? (WB - c * XCH) : XCH;
      gload(W + c * XCH, p.w + ((size_t)(f * KS + s)) * WB + c * XCH, bytes, bars + 0);
    }
    bwait(bars + 0, 0);
  }
  __syncthreads();
  unsigned long long t0 = gt();
  int order[4];
  for (int i = 0; i < 4; ++i) order[i] = (p.rank + 1 + i) & 3;  // remote owners first, own last
  if (tid == 0 && (p.mode & 1)) {  // x producer: 32 chunks of 12 KB through 2 slots
    for (int k = 0; k < 4 * XNCH; ++k) {
      const int slot = k & 1, b = k / XNCH, c = k % XNCH;
      if (k >= 2) bwait(bars + 1 + slot, ((k - 2) >> 1) & 1);
      if (k >= 2 && ((k - 2) % XNCH) == XNCH - 1) *landed = (k - 2) / XNCH + 1;
      expect(bars + 1 + slot, XCH);
      gload(X + slot * XCH, p.x + ((size_t)(order[b] * KS + s)) * XB + c * XCH, XCH, bars + 1 + slot);
    }
    for (int k = 4 * XNCH; k < 4 * XNCH + 2; ++k) {
      const int slot = k & 1;
      bwait(bars + 1 + slot, ((k - 2) >> 1) & 1);
      if (((k - 2) % XNCH) == XNCH - 1) *landed = (k - 2) / XNCH + 1;
    }
  }
  if (tid == 32) {  // reduce-scatter + push of each finished owner block
    for (int b = 0; b < 4; ++b) {
      const int o = order[b], slot = b & 1;
      while (*landed < b + 1) {}
      if (p.mode & 2) {
        if (b >= 2) bwait(bars + 5 + slot, ((b - 2) >> 1) & 1);
        expect(bars + 3 + slot, 2 * PART);
        for (int d = 1; d < KS; ++d) {
          const int peer = (s + d) % KS, idx = (s - peer + KS) % KS - 1;
          uint32_t dst = mapa(sa(R + (slot * 2 + idx) * PART), peer);
          uint32_t pb = mapa(sa(bars + 3 + slot), peer);
          dsm_copy(dst, W + d * PART, PART, pb);
        }
        bwait(bars + 3 + slot, (b >> 1) & 1);
        for (int d = 1; d < KS; ++d) remote_arrive(mapa(sa(bars + 5 + slot), (s + d) % KS));
      }
      if (p.mode & 4) {
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        char* base = p.peers[o];
        const int tok0 = s * (n / KS);
        for (int j = 0; j < n / KS; ++j)
          gstore(base + ((size_t)(p.rank * n + tok0 + j) * 7168 + f * 128) * 2, S + (j % 32) * 256, 256);
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      }
    }
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  unsigned long long t1 = gt();
  if (tid == 0) { atomicMin(p.ts + 0, t0); atomicMax(p.ts + 1, t1); }
}

void setup_peers(int64_t ngpu) {
  for (int g = 0; g < ngpu; ++g) { cudaSetDevice(g);
    for (int q = 0; q < ngpu; ++q) if (q != g) { cudaError_t e = cudaDeviceEnablePeerAccess(q, 0);
      if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) TORCH_CHECK(false, cudaGetErrorString(e)); }
    int optin = 0; cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, g);
    cudaError_t e = cudaFuncSetAttribute(skel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    if (g == 0) printf("smem optin %d need %d set %s\n", optin, (int)SMEM, cudaGetErrorString(e));
    TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
  }
}

void launch(int64_t dev, int64_t w, int64_t x, std::vector<int64_t> peers, int64_t rank, int64_t n,
            int64_t mode, int64_t ts, int64_t flag, int64_t stream) {
  cudaSetDevice((int)dev);
  Prm p; p.w = (const char*)w; p.x = (const char*)x;
  for (int i = 0; i < 4; ++i) p.peers[i] = (char*)peers[i];
  p.rank = (int)rank; p.n = (int)n; p.mode = (int)mode; p.ts = (unsigned long long*)ts; p.flag = (const volatile int*)flag;
  skel<<<dim3(KS, NF, 1), 128, SMEM, (cudaStream_t)stream>>>(p);
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
}
"""


def _module():
    from torch.utils.cpp_extension import load_inline

    tag = hashlib.sha256((_CPP + _CU).encode()).hexdigest()[:16]
    root = os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
    build = os.path.join(root, "k3_oproj_ks_skel", tag)
    os.makedirs(build, exist_ok=True)
    return load_inline(
        name=f"k3_oproj_ks_skel_{tag}", cpp_sources=_CPP, cuda_sources=_CU,
        functions=["setup_peers", "launch"],
        extra_cuda_cflags=["-O3", "-gencode", "arch=compute_100f,code=sm_100f"],
        build_directory=build,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="384")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--modes", default="7,1,2,4,3,5,6")
    args = ap.parse_args()
    mod = _module()
    ng = 4
    mod.setup_peers(ng)
    names = {1: "x", 2: "dsmem", 4: "push"}
    for R in [int(r) for r in args.rows.split(",")]:
        n = R // 4
        assert n % 3 == 0, "owner block must split in 3 K-slice thirds"
        ws, xs, recv, ts, st = [], [], [], [], []
        for g in range(ng):
            d = torch.device("cuda", g)
            ws.append(torch.randint(0, 255, (56 * 3 * 128 * 1024,), dtype=torch.uint8, device=d))
            xs.append(torch.randint(0, 255, (4 * 3 * 96 * 1024,), dtype=torch.uint8, device=d))
            recv.append(torch.zeros(4 * n * 7168, dtype=torch.bfloat16, device=d))
            ts.append(torch.zeros(2, dtype=torch.int64, device=d))
            st.append(torch.cuda.Stream(device=d))
        peers = [r.data_ptr() for r in recv]
        flag = torch.zeros(1, dtype=torch.int32).pin_memory()
        fptr = flag.data_ptr()
        for mode in [int(m) for m in args.modes.split(",")]:
            res = [[] for _ in range(ng)]
            for it in range(args.iters + 3):
                flag[0] = 0
                for g in range(ng):
                    with torch.cuda.device(g):
                        ts[g][0] = (1 << 62)
                        ts[g][1] = 0
                for g in range(ng):
                    torch.cuda.synchronize(g)
                for g in range(ng):
                    mod.launch(g, ws[g].data_ptr(), xs[g].data_ptr(), peers, g, n, mode,
                               ts[g].data_ptr(), fptr, st[g].cuda_stream)
                flag[0] = 1
                for g in range(ng):
                    torch.cuda.synchronize(g)
                if it >= 3:
                    for g in range(ng):
                        t = ts[g].cpu()
                        res[g].append((int(t[1]) - int(t[0])) / 1e3)
            med = [sorted(r)[len(r) // 2] for r in res]
            label = "+".join(v for k, v in names.items() if mode & k) or "none"
            print(f"R{R} n{n} mode {mode} ({label}): window us per GPU "
                  + " ".join(f"{m:6.2f}" for m in med) + f" | max {max(med):6.2f}", flush=True)


if __name__ == "__main__":
    main()
