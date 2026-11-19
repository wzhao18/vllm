# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA decode core SOL skeleton: data movement only (no math), same bytes and sync as
the deployed core at the D3-32 per-GPU shape (24 KDA heads per rank, 64 KiB FP32 state
page per (request, head) item).

Per item: the state page (64 KiB) goes to registers with 256-bit loads, the item's
small inputs (projection rows, convolution window, pending records) are read, the
page is written back in place with 256-bit stores and the small outputs (o rows,
records, window) are written. One persistent grid, MINB CTAs per SM, items claimed
one ahead. PDL: each layer's state loads are issued before griddepcontrol.wait (the
state does not depend on the predecessor), dependents are triggered at the start.

Arms (--arms):
  rmw<NT>x<MINB>   arm (a), in-place R+W every step (hw39 register form)
  lazy<NT>x<MINB>  arm (b), lazy fold: state read-only, plus RING bytes of pending
                   records per item, the page written back every F-th step only
                   (items staggered: item i writes when i % F == step % F)
  ro<NT>x<MINB>    read-only reference (state + small inputs, no state write)

Placement (--place): striped (cudaMalloc) or local (2 MiB chunks alternating dies,
die-matched claim with steal, dies.py SM map).

usage: python kda_skel.py --n 32 64 96 128 --arms rmw512x2 lazy512x2 --place local
"""

import argparse
import os
import sys

import torch
from torch.utils.cpp_extension import load_inline

sys.path.insert(0, os.getcwd())

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import (  # noqa: E402
    dies,
)

PG = 65536
HEADS = 24

CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

constexpr int PG = 65536;
struct Args {
  char* state; int P;
  const char* sin; int sin_b;
  char* sout; int sout_b;
  const char* ring; int ring_b;
  const int* dom; int* ctr; int mode, n0, n1;
  int wmod, step;
  float* sink;
};

__device__ __forceinline__ unsigned smid() {
  unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }

// mode 0: one counter; mode 2: die-matched (own die list first, then steal).
// die d's k-th page: chunk 2 * (k / 32) + d (32 pages per 2 MiB chunk).
__device__ __forceinline__ int claim(const Args& a, unsigned d0) {
  if (a.mode != 2) { int k = atomicAdd(a.ctr, 1); return k < a.P ? k : -1; }
  for (int t = 0; t < 2; ++t) {
    int d = (d0 + t) & 1, nd = d ? a.n1 : a.n0;
    if (*(volatile int*)(a.ctr + 1 + d) >= nd) continue;
    int k = atomicAdd(a.ctr + 1 + d, 1);
    if (k < nd) return (2 * (k / 32) + d) * 32 + (k % 32);
  }
  return -1;
}

#define LDV8(r, p) asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" \
  : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "l"(p))
#define STV8(p, r) asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" \
  :: "l"(p), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]) : "memory")

template <int NT, int MINB>
__global__ void __launch_bounds__(NT, MINB) skel_k(Args a) {
  constexpr int U = PG / (NT * 32);
  __shared__ int s_pg[2];
  const int tid = threadIdx.x;
  unsigned dom = a.dom[smid()] & 1;
  if (tid == 0) s_pg[0] = claim(a, dom);
  __syncthreads();
  int pg = s_pg[0], it = 0;
  unsigned v[U][8];
  auto ld = [&](int p) {
    const char* b = a.state + (size_t)p * PG;
#pragma unroll
    for (int u = 0; u < U; ++u) LDV8(v[u], b + (size_t)(u * NT + tid) * 32);
  };
  if (pg >= 0) ld(pg);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  unsigned acc = 0;
  while (pg >= 0) {
    if (tid == 0) s_pg[(it + 1) & 1] = claim(a, dom);
    for (int o = tid * 32; o < a.sin_b; o += NT * 32) {
      unsigned r[8]; LDV8(r, a.sin + (size_t)pg * a.sin_b + o); acc ^= r[0] ^ r[7]; }
    for (int o = tid * 32; o < a.ring_b; o += NT * 32) {
      unsigned r[8]; LDV8(r, a.ring + (size_t)pg * a.ring_b + o); acc ^= r[0] ^ r[7]; }
    __syncthreads();
    int nx = s_pg[(it + 1) & 1];
    bool wr = a.wmod == 1 || (a.wmod > 0 && pg % a.wmod == a.step);
    if (wr) {
      char* b = a.state + (size_t)pg * PG;
#pragma unroll
      for (int u = 0; u < U; ++u) {
#pragma unroll
        for (int j = 0; j < 8; ++j) v[u][j] = __float_as_uint(__uint_as_float(v[u][j]) + 1.0f);
        STV8(b + (size_t)(u * NT + tid) * 32, v[u]);
      }
    } else {
#pragma unroll
      for (int u = 0; u < U; ++u) acc ^= v[u][0] ^ v[u][5];
    }
    for (int o = tid * 32; o < a.sout_b; o += NT * 32) {
      unsigned r[8] = {acc, acc, acc, acc, acc, acc, acc, acc};
      STV8(a.sout + (size_t)pg * a.sout_b + o, r); }
    if (nx >= 0) ld(nx);
    pg = nx; ++it;
  }
  if (acc == 0x12345678u) a.sink[0] = 1.f;
}

__device__ __forceinline__ unsigned sa(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void wait_par(unsigned b, unsigned ph) {
  asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W;}"
               :: "r"(b), "r"(ph) : "memory"); }
__device__ __forceinline__ void arrive(unsigned b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }

// TMA ring: S slots x 64 KiB, warp 0 lane 0 claims + issues (OPS bulk copies per page),
// NC consumer threads read their 128 B of the page from smem, write it back by STG.
// Scan math of one item on this thread's U rows x 8 keys (rows u * NC / 16 + c / 16,
// keys 8 (c % 16) + [0, 8)): fold S = fma(S0, D, sum_t u_t w_t) (4 pending tokens),
// then acc[row][m] += S x_m over the 8 scan vectors (FFMA2), the 16 lanes sharing
// the rows reduce by a transposed butterfly. Operands: D[128] W[4][128]
// X[128 keys][8] Ur[4][128 rows] (constant in the skeleton).
template <int NC, int U>
__device__ __forceinline__ unsigned scan_math(unsigned (&v)[U][8], const float* ops, int c) {
  const float* D = ops; const float* W = ops + 128; const float* X = ops + 640;
  const float* Ur = ops + 1664;
  const int kb = (c & 15) * 8;
  float2 acc[U][4];
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int m = 0; m < 4; ++m) acc[u][m] = make_float2(0.f, 0.f);
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int row = (u * NC + c) / 16;
    float ut[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) ut[t] = Ur[t * 128 + row];
#pragma unroll
    for (int e = 0; e < 8; e += 2) {
      const int k = kb + e;
      float2 fc = make_float2(-0.f, -0.f);
#pragma unroll
      for (int t = 3; t >= 0; --t)
        fc = __ffma2_rn(make_float2(ut[t], ut[t]), *(const float2*)(W + t * 128 + k), fc);
      float2 s = make_float2(__uint_as_float(v[u][e]), __uint_as_float(v[u][e + 1]));
      float2 n = __ffma2_rn(s, *(const float2*)(D + k), fc);
      v[u][e] = __float_as_uint(n.x); v[u][e + 1] = __float_as_uint(n.y);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float nv = h ? n.y : n.x;
        const float4* xp = (const float4*)(X + (k + h) * 8);
        float4 x0 = xp[0], x1 = xp[1];
        acc[u][0] = __ffma2_rn(make_float2(nv, nv), make_float2(x0.x, x0.y), acc[u][0]);
        acc[u][1] = __ffma2_rn(make_float2(nv, nv), make_float2(x0.z, x0.w), acc[u][1]);
        acc[u][2] = __ffma2_rn(make_float2(nv, nv), make_float2(x1.x, x1.y), acc[u][2]);
        acc[u][3] = __ffma2_rn(make_float2(nv, nv), make_float2(x1.z, x1.w), acc[u][3]);
      }
    }
  }
  // transposed butterfly over the 16 lanes: U*8 values -> U*8/16 per lane
  float vals[U * 8];
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int m = 0; m < 4; ++m) { vals[u * 8 + 2 * m] = acc[u][m].x; vals[u * 8 + 2 * m + 1] = acc[u][m].y; }
  int n = U * 8;
#pragma unroll
  for (int off = 8; off >= 1; off >>= 1) {
    const bool up = (c & off) != 0;
    const int half = n / 2;
#pragma unroll
    for (int i = 0; i < U * 4; ++i) {
      if (i < half) {
        float send = up ? vals[i] : vals[i + half];
        float keep = up ? vals[i + half] : vals[i];
        vals[i] = keep + __shfl_xor_sync(0xffffffffu, send, off);
      }
    }
    n = half;
  }
  unsigned r = 0;
#pragma unroll
  for (int i = 0; i < 2; ++i) r ^= __float_as_uint(vals[i]);
  return r;
}

template <int NC, int S, int OPS, int MATH = 0>
__global__ void __launch_bounds__(NC + 32, 1) tma_k(Args a) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) unsigned long long full[S], empty[S];
  __shared__ int s_item[S];
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < S; ++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&full[s])));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(&empty[s])), "r"(NC / 32));
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid < 32) {
    if (tid == 0) {
      unsigned dom = a.dom[smid()] & 1;
      bool waited = false;
      for (int j = 0;; ++j) {
        int s = j % S;
        if (j >= S) wait_par(sa(&empty[s]), ((j / S) - 1) & 1);
        int pg = claim(a, dom);
        s_item[s] = pg;
        if (pg < 0) { arrive(sa(&full[s])); break; }
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     :: "r"(sa(&full[s])), "r"(PG) : "memory");
        for (int o = 0; o < OPS; ++o)
          asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                       :: "r"(sa(smem + (size_t)s * PG + o * (PG / OPS))),
                          "l"(a.state + (size_t)pg * PG + o * (PG / OPS)), "r"(PG / OPS),
                          "r"(sa(&full[s])) : "memory");
        if (!waited && j == S - 1) {
          asm volatile("griddepcontrol.launch_dependents;");
          waited = true;
        }
      }
    }
    return;
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int c = tid - 32;
  unsigned acc = 0;
  for (int j = 0;; ++j) {
    int s = j % S;
    wait_par(sa(&full[s]), (j / S) & 1);
    int pg = s_item[s];
    if (pg < 0) break;
    for (int o = c * 32; o < a.sin_b; o += NC * 32) {
      unsigned r[8]; LDV8(r, a.sin + (size_t)pg * a.sin_b + o); acc ^= r[0] ^ r[7]; }
    for (int o = c * 32; o < a.ring_b; o += NC * 32) {
      unsigned r[8]; LDV8(r, a.ring + (size_t)pg * a.ring_b + o); acc ^= r[0] ^ r[7]; }
    bool wr = a.wmod == 1 || (a.wmod > 0 && pg % a.wmod == a.step);
    constexpr int U = PG / (NC * 32);
    unsigned v[U][8];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const uint4* p = (const uint4*)(smem + (size_t)s * PG + (size_t)(u * NC + c) * 32);
      uint4 x = p[0], y = p[1];
      v[u][0] = x.x; v[u][1] = x.y; v[u][2] = x.z; v[u][3] = x.w;
      v[u][4] = y.x; v[u][5] = y.y; v[u][6] = y.z; v[u][7] = y.w;
    }
    __syncwarp();
    if ((c & 31) == 0) arrive(sa(&empty[s]));
    if (MATH) acc ^= scan_math<NC, U>(v, (const float*)(smem + (size_t)S * PG), c);
    if (wr) {
      char* b = a.state + (size_t)pg * PG;
#pragma unroll
      for (int u = 0; u < U; ++u) {
#pragma unroll
        for (int k = 0; k < 8; ++k) if (!MATH) v[u][k] = __float_as_uint(__uint_as_float(v[u][k]) + 1.0f);
        STV8(b + (size_t)(u * NC + c) * 32, v[u]);
      }
    } else {
#pragma unroll
      for (int u = 0; u < U; ++u) acc ^= v[u][0] ^ v[u][5];
    }
    for (int o = c * 32; o < a.sout_b; o += NC * 32) {
      unsigned r[8] = {acc, acc, acc, acc, acc, acc, acc, acc};
      STV8(a.sout + (size_t)pg * a.sout_b + o, r); }
  }
  if (acc == 0x12345678u) a.sink[0] = 1.f;
}

template <int NC, int S, int OPS, int MATH = 0>
static void go_tma(const Args& a, cudaStream_t st) {
  int dev; cudaGetDevice(&dev);
  int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  size_t sm = (size_t)S * PG + (MATH ? 8704 : 0);
  cudaFuncSetAttribute(tma_k<NC, S, OPS, MATH>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(nsm); cfg.blockDim = dim3(NC + 32); cfg.stream = st;
  cfg.dynamicSmemBytes = sm;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, tma_k<NC, S, OPS, MATH>, a) == cudaSuccess, "launch tma");
}

template <int NT, int MINB>
static void go(const Args& a, cudaStream_t s) {
  int dev; cudaGetDevice(&dev);
  int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(nsm * MINB); cfg.blockDim = dim3(NT); cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, skel_k<NT, MINB>, a) == cudaSuccess, "launch");
}

void launch(int64_t form, int64_t state, int64_t P, int64_t sin, int64_t sin_b,
            int64_t sout, int64_t sout_b, int64_t ring, int64_t ring_b, int64_t dom,
            int64_t ctr, int64_t mode, int64_t n0, int64_t n1, int64_t wmod,
            int64_t step, int64_t sink) {
  Args a{(char*)state, (int)P, (const char*)sin, (int)sin_b, (char*)sout, (int)sout_b,
         (const char*)ring, (int)ring_b, (const int*)dom, (int*)ctr, (int)mode,
         (int)n0, (int)n1, (int)wmod, (int)step, (float*)sink};
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  switch (form) {
    case 5121: go<512, 1>(a, s); break;
    case 5122: go<512, 2>(a, s); break;
    case 2564: go<256, 4>(a, s); break;
    case 2562: go<256, 2>(a, s); break;
    case 70531: go_tma<512, 3, 1>(a, s); break;
    case 70534: go_tma<512, 3, 4>(a, s); break;
    case 70231: go_tma<256, 3, 1>(a, s); break;
    case 70234: go_tma<256, 3, 4>(a, s); break;
    case 70541: go_tma<512, 4, 1>(a, s); break;
    case 70542: go_tma<512, 4, 2>(a, s); break;
    case 70241: go_tma<256, 4, 1>(a, s); break;
    case 170531: go_tma<512, 3, 1, 1>(a, s); break;
    case 170541: go_tma<512, 4, 1, 1>(a, s); break;
    case 170241: go_tma<256, 4, 1, 1>(a, s); break;
    default: TORCH_CHECK(false, "form");
  }
}

std::vector<int64_t> attrs(int64_t form) {
  cudaFuncAttributes f;
  switch (form) {
    case 5121: cudaFuncGetAttributes(&f, skel_k<512, 1>); break;
    case 5122: cudaFuncGetAttributes(&f, skel_k<512, 2>); break;
    case 2564: cudaFuncGetAttributes(&f, skel_k<256, 4>); break;
    default: cudaFuncGetAttributes(&f, skel_k<256, 2>); break;
  }
  return {f.numRegs, (int64_t)f.localSizeBytes};
}
"""

CPP = """
void launch(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
            int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
std::vector<int64_t> attrs(int64_t);
"""


def _module():
    return load_inline(
        name="k31_kda_skel_v3",
        cpp_sources=CPP,
        cuda_sources=CUDA,
        functions=["launch", "attrs"],
        extra_cuda_cflags=["-O3", "-gencode", "arch=compute_100f,code=sm_100f"],
        verbose=False,
    )


def parse_arm(arm):
    for kind in ("rmw", "lazy", "ro"):
        if arm.startswith(kind):
            form = arm[len(kind):]
            if form.startswith("t"):  # t<NC>s<S>o<OPS>: TMA ring, NC consumers
                nc, ops = form[1:].rstrip("m").split("o")
                nc, slots = (nc.split("s") + ["3"])[:2]
                code = (5 if nc == "512" else 2) * 100 + 10 * int(slots) + int(ops)
                return kind, 70000 + code + (100000 if form.endswith("m") else 0)
            nt, minb = form.split("x")
            return kind, int(nt) * 10 + int(minb)
    raise ValueError(arm)


def run(mod, n, arm, place, args, dom):
    kind, form = parse_arm(arm)
    P = n * HEADS
    L = args.layers
    ppad = (P + 63) // 64 * 64  # even chunk count: every layer starts on die 0
    dev = torch.device("cuda")
    if place == "local":
        backing = dies.localized_empty(L * ppad * PG, [0, 1], dev)
        mode = 2
    else:
        backing = torch.empty(L * ppad * PG, dtype=torch.uint8, device=dev)
        mode = 0
    st = backing.view(torch.float32)
    st.zero_()
    per_die = [sum(1 for p in range(P) if (p // 32) % 2 == d) for d in (0, 1)]
    n0, n1 = per_die
    sin = torch.zeros(L, P, args.sin, dtype=torch.uint8, device=dev)
    sout = torch.empty(L, P, max(args.sout, 32), dtype=torch.uint8, device=dev)
    ring_b = args.ring if kind == "lazy" else 0
    ring = torch.zeros(L, P, max(ring_b, 32), dtype=torch.uint8, device=dev)
    ctr = torch.zeros(L, 4, dtype=torch.int32, device=dev)
    sink = torch.zeros(4, dtype=torch.float32, device=dev)
    wmod = {"rmw": 1, "lazy": args.fold, "ro": 0}[kind]

    def body():
        ctr.zero_()
        for li in range(L):
            mod.launch(
                form,
                backing.data_ptr() + li * ppad * PG,
                P,
                sin[li].data_ptr(),
                args.sin,
                sout[li].data_ptr(),
                args.sout,
                ring[li].data_ptr(),
                ring_b,
                dom.data_ptr(),
                ctr[li].data_ptr(),
                mode,
                n0,
                n1,
                wmod,
                li % max(wmod, 1),
                sink.data_ptr(),
            )

    body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(args.iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    us = e0.elapsed_time(e1) * 1e3 / args.iters / L
    # every written page was incremented once per run: check layer 0 (first P pages)
    runs = 1 + 3 + args.iters
    s0 = st[: P * PG // 4].view(P, PG // 4)
    ok = True
    if kind == "rmw":
        ok = bool((s0 == float(runs)).all())
    elif kind == "lazy":
        exp = torch.tensor(
            [float(runs) if p % wmod == 0 else 0.0 for p in range(P)], device=dev
        )
        ok = bool((s0 == exp[:, None]).all())
    elif kind == "ro":
        ok = bool((s0 == 0).all())
    rb = P * (PG + args.sin + ring_b)
    wb = P * args.sout + (P * PG if kind == "rmw" else (P * PG // wmod if wmod else 0))
    tbs = (rb + wb) / us / 1e6
    print(
        f"{arm:12s} {place:7s} n{n:<4d} {us:7.2f} us/layer  {tbs:6.2f} TB/s "
        f"(R {rb / 2**20:.0f} MiB W {wb / 2**20:.0f} MiB) ok={ok}",
        flush=True,
    )
    del backing, st, sin, sout, ring
    torch.cuda.empty_cache()
    return us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[32, 64, 96, 128])
    ap.add_argument("--arms", nargs="+", default=["rmw512x2", "lazy512x2"])
    ap.add_argument("--place", nargs="+", default=["local", "striped"])
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--sin", type=int, default=8192, help="small input bytes/item")
    ap.add_argument("--sout", type=int, default=6144, help="small output bytes/item")
    ap.add_argument("--ring", type=int, default=8192, help="lazy ring bytes/item")
    ap.add_argument("--fold", type=int, default=4, help="lazy fold period F")
    args = ap.parse_args()
    mod = _module()
    for arm in args.arms:
        _, form = parse_arm(arm)
        if form >= 70000:
            continue
        regs, local = mod.attrs(form)
        print(f"{arm}: regs {regs} local {local} B", flush=True)
    dom = dies.sm_domain_table(torch.device("cuda"))
    print(f"SMs per die: {int((dom == 0).sum())} / {int((dom == 1).sum())}", flush=True)
    for n in args.n:
        for place in args.place:
            for arm in args.arms:
                run(mod, n, arm, place, args, dom)


if __name__ == "__main__":
    main()
