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
#include <c10/cuda/CUDAStream.h>
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
__device__ __forceinline__ float2 ffma2(float2 a, float2 b, float2 c) {
  float2 d;
  asm("{.reg .b64 ra, rb, rc, rd;\n mov.b64 ra, {%2, %3};\n mov.b64 rb, {%4, %5};\n"
      " mov.b64 rc, {%6, %7};\n fma.rn.f32x2 rd, ra, rb, rc;\n mov.b64 {%0, %1}, rd;}"
      : "=f"(d.x), "=f"(d.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y), "f"(c.x), "f"(c.y));
  return d;
}

// Scan math of one item on this thread's U rows x 8 keys (rows u * NC / 16 + c / 16,
// keys 8 (c % 16) + [0, 8)): fold S = fma(S0, D, sum_t u_t w_t) (4 pending tokens),
// then acc[row][m] += S x_m over the 8 scan vectors (FFMA2), the 16 lanes sharing
// the rows reduce by a transposed butterfly. Operands: D[128] W[4][128]
// X[128 keys][8] Ur[4][128 rows] (constant in the skeleton).
template <int M>
__device__ __forceinline__ float2 fm(float2 a, float2 b, float2 c) {
  if (M == 2) return make_float2(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y));
  return ffma2(a, b, c);
}


__device__ __forceinline__ void tm_st16(unsigned taddr, const float* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
               :: "r"(taddr), "f"(r[0]), "f"(r[1]), "f"(r[2]), "f"(r[3]), "f"(r[4]), "f"(r[5]), "f"(r[6]), "f"(r[7]), "f"(r[8]), "f"(r[9]), "f"(r[10]), "f"(r[11]), "f"(r[12]), "f"(r[13]), "f"(r[14]), "f"(r[15]) : "memory");
}
__device__ __forceinline__ void tm_ld16(unsigned taddr, float* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
               : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]), "=f"(r[5]), "=f"(r[6]), "=f"(r[7]), "=f"(r[8]), "=f"(r[9]), "=f"(r[10]), "=f"(r[11]), "=f"(r[12]), "=f"(r[13]), "=f"(r[14]), "=f"(r[15]) : "r"(taddr) : "memory");
}
__device__ __forceinline__ void tm_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tm_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tm_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tm_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
// The storing warp's lane l (key lane l % 16): per key pair i, the 26 operand floats the
// M=1 scan reads by LDS (w[0..3] float2, d float2, x[h][0..1] float4), padded to 32 columns.
__device__ __forceinline__ void tm_store_ops(unsigned taddr, const float* ops, int l) {
  const float* D = ops; const float* W = ops + 128; const float* X = ops + 640;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int p = i * 16 + l;
    float r[32];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const float2 w = *(const float2*)(W + (t * 64 + p) * 2);
      r[2 * t] = w.x; r[2 * t + 1] = w.y;
    }
    const float2 d = *(const float2*)(D + p * 2);
    r[8] = d.x; r[9] = d.y;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float4* xp = (const float4*)X + ((2 * i + h) * 2) * 16 + l;
      const float4 a = xp[0], b = xp[16];
      r[10 + 8 * h] = a.x; r[11 + 8 * h] = a.y; r[12 + 8 * h] = a.z; r[13 + 8 * h] = a.w;
      r[14 + 8 * h] = b.x; r[15 + 8 * h] = b.y; r[16 + 8 * h] = b.z; r[17 + 8 * h] = b.w;
    }
#pragma unroll
    for (int k = 26; k < 32; ++k) r[k] = 0.f;
    tm_st16(taddr + 32 * i, r);
    tm_st16(taddr + 32 * i + 16, r + 16);
  }
  tm_wait_st();
}

template <int NC, int U, int M>
__device__ __forceinline__ unsigned scan_math(unsigned (&v)[U][8], const float* ops, int c,
                                              unsigned taddr = 0) {
  const float* D = ops; const float* W = ops + 128; const float* X = ops + 640;
  const float* Ur = ops + 1664;
  const int l = c & 15;  // operands swizzled by lane: conflict-free shared loads
  float2 acc[U][4];
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int m = 0; m < 4; ++m) acc[u][m] = make_float2(0.f, 0.f);
  float ut[U][4];
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int t = 0; t < 4; ++t) ut[u][t] = Ur[t * 128 + (u * NC + c) / 16];
#pragma unroll
  // M 5/7 (diagnostic): every key pair reuses pair 0's operands (operand LDS / 4);
  // M 6/7: no butterfly (the partials are folded locally).
  constexpr bool kReuse = M == 5 || M == 7, kNoBfly = M == 6 || M == 7;
  for (int e = 0; e < 8; e += 2) {
    const int p = (kReuse ? 0 : (e >> 1) * 16) + l;
    float2 w[4];
    float2 d;
    float4 x[2][2];
    if constexpr (M == 8) {  // operands from tensor memory (written by the quarter's storer)
      float r[32];
      tm_ld16(taddr + 32 * (e >> 1), r);
      tm_ld16(taddr + 32 * (e >> 1) + 16, r + 16);
      tm_wait_ld();
#pragma unroll
      for (int t = 0; t < 4; ++t) w[t] = make_float2(r[2 * t], r[2 * t + 1]);
      d = make_float2(r[8], r[9]);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        x[h][0] = make_float4(r[10 + 8 * h], r[11 + 8 * h], r[12 + 8 * h], r[13 + 8 * h]);
        x[h][1] = make_float4(r[14 + 8 * h], r[15 + 8 * h], r[16 + 8 * h], r[17 + 8 * h]);
      }
    } else {
#pragma unroll
      for (int t = 0; t < 4; ++t) w[t] = *(const float2*)(W + (t * 64 + p) * 2);
      d = *(const float2*)(D + p * 2);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float4* xp = (const float4*)X + (((kReuse ? 0 : e) + h) * 2) * 16 + l;
        if constexpr (M == 9) {
          float2 q[4];
#pragma unroll
          for (int i = 0; i < 4; ++i)
            asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(q[i].x), "=f"(q[i].y)
                         : "r"(sa((const float2*)(xp + 16 * (i >> 1)) + (i & 1))));
          x[h][0] = make_float4(q[0].x, q[0].y, q[1].x, q[1].y);
          x[h][1] = make_float4(q[2].x, q[2].y, q[3].x, q[3].y);
        } else {
          x[h][0] = xp[0]; x[h][1] = xp[16];
        }
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      float2 fc = make_float2(-0.f, -0.f);
#pragma unroll
      for (int t = 3; t >= 0; --t) fc = fm<M>(make_float2(ut[u][t], ut[u][t]), w[t], fc);
      float2 s = make_float2(__uint_as_float(v[u][e]), __uint_as_float(v[u][e + 1]));
      float2 n = fm<M>(s, d, fc);
      v[u][e] = __float_as_uint(n.x); v[u][e + 1] = __float_as_uint(n.y);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float nv = h ? n.y : n.x;
        const float2 nn = make_float2(nv, nv);
        acc[u][0] = fm<M>(nn, make_float2(x[h][0].x, x[h][0].y), acc[u][0]);
        acc[u][1] = fm<M>(nn, make_float2(x[h][0].z, x[h][0].w), acc[u][1]);
        acc[u][2] = fm<M>(nn, make_float2(x[h][1].x, x[h][1].y), acc[u][2]);
        acc[u][3] = fm<M>(nn, make_float2(x[h][1].z, x[h][1].w), acc[u][3]);
      }
    }
  }
  // transposed butterfly over the 16 lanes: U*8 values -> U*8/16 per lane
  float vals[U * 8];
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int m = 0; m < 4; ++m) { vals[u * 8 + 2 * m] = acc[u][m].x; vals[u * 8 + 2 * m + 1] = acc[u][m].y; }
  if constexpr (kNoBfly) {
    unsigned r = 0;
#pragma unroll
    for (int i = 0; i < U * 8; ++i) r ^= __float_as_uint(vals[i]);
    return r;
  }
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

// MATH 3: lazy-fold verify (state read-only; the ring's rank-16 term on the value side).
// Row-octet layout: thread c owns rows 2 (c / 8) + {0, 1} and keys 4 l + 32 i + [0, 4)
// (l = c % 8, i < 4): every shared load of the page is one 128 B wavefront per 8 lanes.
// Dots: FFMA2 over key pairs (state pair x operand pair, both consecutive registers, no
// packing moves), pair collapse, 8-lane transposed butterfly (16 -> 2 values per lane),
// then the ring term u_t[row] c_t[m] over 16 records. Operands: X[i][j][q][l] float4
// (x_{2kp}[2q], x_{2kp+1}[2q], x_{2kp}[2q+1], x_{2kp+1}[2q+1]) = 4 KiB, C[4][8] float4.
__device__ __forceinline__ unsigned lazy_verify(const unsigned char* slot, const float* ops,
                                                const float* ring, int l, int r0, bool arr,
                                                unsigned bar, char* wb) {
  float2 st[2][8];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float4 q = *(const float4*)(slot + (size_t)(r0 + h) * 512 + i * 128 + l * 16);
      st[h][2 * i] = make_float2(q.x, q.y);
      st[h][2 * i + 1] = make_float2(q.z, q.w);
    }
  __syncwarp();
  if (arr && (threadIdx.x & 31) == 0) arrive(bar);
  const float4* X = (const float4*)ops;
  float2 acc[2][8];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int m = 0; m < 8; ++m) acc[h][m] = make_float2(0.f, 0.f);
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float4 x[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) x[q] = X[((i * 2 + j) * 4 + q) * 8 + l];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float2 sv = st[h][2 * i + j];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          acc[h][2 * q] = ffma2(sv, make_float2(x[q].x, x[q].y), acc[h][2 * q]);
          acc[h][2 * q + 1] = ffma2(sv, make_float2(x[q].z, x[q].w), acc[h][2 * q + 1]);
        }
      }
    }
  float vals[16];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int m = 0; m < 8; ++m) vals[h * 8 + m] = acc[h][m].x + acc[h][m].y;
  int n = 16;
#pragma unroll
  for (int off = 4; off >= 1; off >>= 1) {
    const bool up = (l & off) != 0;
    const int half = n / 2;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (i < half) {
        float send = up ? vals[i] : vals[i + half];
        float keep = up ? vals[i + half] : vals[i];
        vals[i] = keep + __shfl_xor_sync(0xffffffffu, send, off);
      }
    }
    n = half;
  }
  // ring term: this lane's 2 values (row r0 + bit 2 of l, vectors 2 (l % 4) + {0, 1})
  const int row = r0 + ((l >> 2) & 1);
  const float4* U = (const float4*)(ring + row * 16);
  const float4* C = (const float4*)(ops + 1024) + (l & 3) * 8;
#pragma unroll
  for (int t4 = 0; t4 < 4; ++t4) {
    const float4 u = U[t4];
    const float4 c0 = C[2 * t4], c1 = C[2 * t4 + 1];  // (m0, m1) for records 4 t4 + {0..3}
    vals[0] = fmaf(u.x, c0.x, vals[0]); vals[1] = fmaf(u.x, c0.y, vals[1]);
    vals[0] = fmaf(u.y, c0.z, vals[0]); vals[1] = fmaf(u.y, c0.w, vals[1]);
    vals[0] = fmaf(u.z, c1.x, vals[0]); vals[1] = fmaf(u.z, c1.y, vals[1]);
    vals[0] = fmaf(u.w, c1.z, vals[0]); vals[1] = fmaf(u.w, c1.w, vals[1]);
  }
  if (wb) {  // fold step (every F-th): write the page back (skeleton: unchanged values)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int i = 0; i < 4; ++i)
        *(float4*)(wb + (size_t)(r0 + h) * 512 + i * 128 + l * 16) =
            make_float4(st[h][2 * i].x, st[h][2 * i].y, st[h][2 * i + 1].x, st[h][2 * i + 1].y);
  }
  return __float_as_uint(vals[0]) ^ __float_as_uint(vals[1]);
}

// MATH 4 (M1b): lazy-fold verify, KEY-SPANNING warps. Lane t of warp w owns keys 4 t + [0, 4)
// of rows 8 w + [0, 8) (4 passes of 2 rows): every operand load is distinct per lane (no
// broadcast replay over the 8-lane phases of a 128-bit shared load). Operands per lane:
// X[t][key 4][m 8] (32 floats, 8 x LDS.128, rotated by lane: conflict-free) kept in
// registers for all passes; dots as FFMA2 over vector pairs with the state value
// broadcast (no pair collapse); ring term before the reduction: lane t adds record
// e = t / 2 for vectors 4 (t % 2) + [0, 4): u_e[row] (ring [row pair][e][2]) x C[e][m].
// Then a 32-lane transposed butterfly per pass (16 -> 1 value per lane pair).
__device__ __forceinline__ unsigned lazy_verify_ks(const unsigned char* slot, const float* ops,
                                                   const float* ring, int w, unsigned bar,
                                                   char* wb) {
  const int t = threadIdx.x & 31;
  float st[8][4];  // [row 8 w + p][key 4 t + k]
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const float4 q = *(const float4*)(slot + (size_t)(8 * w + p) * 512 + t * 16);
    st[p][0] = q.x; st[p][1] = q.y; st[p][2] = q.z; st[p][3] = q.w;
  }
  __syncwarp();
  if (t == 0) arrive(bar);
  float2 x[4][4];  // [key][m pair]
  const float4* X = (const float4*)ops + t * 8;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float4 v = X[(j + t) & 7];  // rotated: physical (j + t) % 8 holds logical j
    x[j >> 1][(j & 1) * 2] = make_float2(v.x, v.y);
    x[j >> 1][(j & 1) * 2 + 1] = make_float2(v.z, v.w);
  }
  const float4 cr = *((const float4*)(ops + 1024) + t);  // C[e = t / 2][4 (t % 2) + 0..3]
  unsigned r = 0;
#pragma unroll
  for (int ps = 0; ps < 4; ++ps) {
    float2 acc[2][4];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[h][q] = make_float2(0.f, 0.f);
#pragma unroll
    for (int k = 0; k < 4; ++k)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float sv = st[2 * ps + h][k];
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[h][q] = ffma2(make_float2(sv, sv), x[k][q], acc[h][q]);
      }
    float vals[16];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int q = 0; q < 4; ++q) { vals[h * 8 + 2 * q] = acc[h][q].x; vals[h * 8 + 2 * q + 1] = acc[h][q].y; }
    const float2 u = *((const float2*)ring + (4 * w + ps) * 16 + (t >> 1));  // rows 8w+2ps+{0,1}
    const bool odd = (t & 1) != 0;  // static register indices only (no local memory)
    const float crv[4] = {cr.x, cr.y, cr.z, cr.w};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float uh = h ? u.y : u.x, ue = odd ? 0.f : uh, uo = odd ? uh : 0.f;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        vals[h * 8 + k] = fmaf(ue, crv[k], vals[h * 8 + k]);
        vals[h * 8 + 4 + k] = fmaf(uo, crv[k], vals[h * 8 + 4 + k]);
      }
    }
    int n = 16;
#pragma unroll
    for (int off = 16; off >= 2; off >>= 1) {
      const bool up = (t & off) != 0;
      const int half = n / 2;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (i < half) {
          float send = up ? vals[i] : vals[i + half];
          float keep = up ? vals[i + half] : vals[i];
          vals[i] = keep + __shfl_xor_sync(0xffffffffu, send, off);
        }
      }
      n = half;
    }
    vals[0] += __shfl_xor_sync(0xffffffffu, vals[0], 1);
    r ^= __float_as_uint(vals[0]);
    if (wb) {  // fold step (every F-th): this pass's 2 rows back (skeleton: unchanged)
#pragma unroll
      for (int h = 0; h < 2; ++h)
        *(float4*)(wb + (size_t)(8 * w + 2 * ps + h) * 512 + t * 16) =
            make_float4(st[2 * ps + h][0], st[2 * ps + h][1], st[2 * ps + h][2], st[2 * ps + h][3]);
    }
  }
  return r;
}

// G alternating consumer groups of NC / G threads: group g takes items j = g (mod G); a
// thread covers PG / (NC / G) bytes of the page in passes of 4 chunks (32 values), the
// slot is released after the last pass's copy (math of one group overlaps the other's
// copy / stores).
template <int NC, int S, int OPS, int MATH = 0, int G = 1>
__global__ void __launch_bounds__(NC + 32, 1) tma_k(Args a) {
  constexpr int NG = NC / G;
  const int SB = a.sin_b + a.ring_b;  // small inputs ride on the slot barrier
  const size_t SL = PG + SB;
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) unsigned long long full[S], empty[S];
  __shared__ int s_item[S];
  __shared__ unsigned s_taddr;
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < S; ++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&full[s])));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(&empty[s])), "r"(NG / 32));
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
        if (pg < 0) {
          arrive(sa(&full[s]));
          for (int t = 1; t < G; ++t) {  // every group sees the end
            int jj = j + t, ss = jj % S;
            if (jj >= S) wait_par(sa(&empty[ss]), ((jj / S) - 1) & 1);
            s_item[ss] = -1;
            arrive(sa(&full[ss]));
          }
          break;
        }
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     :: "r"(sa(&full[s])), "r"(PG + SB) : "memory");
        for (int o = 0; o < OPS; ++o)
          asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                       :: "r"(sa(smem + (size_t)s * SL + o * (PG / OPS))),
                          "l"(a.state + (size_t)pg * PG + o * (PG / OPS)), "r"(PG / OPS),
                          "r"(sa(&full[s])) : "memory");
        if (SB) {
          char* d = (char*)smem + (size_t)s * SL + PG;
          if (a.sin_b) asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(sa(d)), "l"(a.sin + (size_t)pg * a.sin_b), "r"(a.sin_b), "r"(sa(&full[s])) : "memory");
          if (a.ring_b) asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(sa(d + a.sin_b)), "l"(a.ring + (size_t)pg * a.ring_b), "r"(a.ring_b), "r"(sa(&full[s])) : "memory");
        }
        if (!waited && j == S - 1) {
          asm volatile("griddepcontrol.launch_dependents;");
          waited = true;
        }
      }
    }
    return;
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int cg = tid - 32, grp = cg / NG, c = cg % NG;
  unsigned tq = 0;  // this warp's TMEM lane quarter base (MATH 8)
  if constexpr (MATH == 8) {
    if (tid < 64) {
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;"
                   :: "r"(sa(&s_taddr)) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    tm_fence_before();
    asm volatile("bar.sync 1, %0;" :: "r"(NC) : "memory");
    tm_fence_after();
    tq = s_taddr + ((32u * ((tid >> 5) & 3)) << 16);
  }
  unsigned acc = 0;
  for (int j = grp;; j += G) {
    int s = j % S;
    wait_par(sa(&full[s]), (j / S) & 1);
    int pg = s_item[s];
    if (pg < 0) break;
    for (int o = c * 16; o < SB; o += NG * 16) {
      uint4 r = *(const uint4*)(smem + (size_t)s * SL + PG + o); acc ^= r.x ^ r.w; }
    bool wr = a.wmod == 1 || (a.wmod > 0 && pg % a.wmod == a.step);
    if constexpr (MATH == 4) {
      acc ^= lazy_verify_ks(smem + (size_t)s * SL, (const float*)(smem + (size_t)S * SL),
                            (const float*)(smem + (size_t)s * SL + PG + a.sin_b), c >> 5,
                            sa(&empty[s]), wr ? a.state + (size_t)pg * PG : nullptr);
    } else if constexpr (MATH == 3) {
      constexpr int NPV = 128 / (2 * (NG / 8));  // row-pair passes per thread (G2: 2)
#pragma unroll 1
      for (int ps = 0; ps < NPV; ++ps)
        acc ^= lazy_verify(smem + (size_t)s * SL, (const float*)(smem + (size_t)S * SL),
                           (const float*)(smem + (size_t)s * SL + PG + a.sin_b), c & 7,
                           2 * NPV * (c >> 3) + 2 * ps, ps == NPV - 1, sa(&empty[s]),
                           wr ? a.state + (size_t)pg * PG : nullptr);
    } else {
    constexpr int UT = PG / (NG * 32), U = UT < 4 ? UT : 4, NP = UT / U;
#pragma unroll 1
    for (int ps = 0; ps < NP; ++ps) {
      unsigned v[U][8];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const uint4* p = (const uint4*)(smem + (size_t)s * SL + (size_t)((ps * U + u) * NG + c) * 32);
        uint4 x = p[0], y = p[1];
        v[u][0] = x.x; v[u][1] = x.y; v[u][2] = x.z; v[u][3] = x.w;
        v[u][4] = y.x; v[u][5] = y.y; v[u][6] = y.z; v[u][7] = y.w;
      }
      if (ps == NP - 1) {
        __syncwarp();
        if ((c & 31) == 0) arrive(sa(&empty[s]));
      }
      if (MATH == 1 || MATH == 2 || (MATH >= 5 && MATH <= 7) || MATH == 9) acc ^= scan_math<NG, U, MATH>(v, (const float*)(smem + (size_t)S * SL), c);
      if constexpr (MATH == 8) {
        const unsigned col = (j & 1) * 128;
        const int hw = tid >> 5;
        if (hw >= 4 && hw < 8) tm_store_ops(tq + col, (const float*)(smem + (size_t)S * SL), tid & 15);
        tm_fence_before();
        asm volatile("bar.sync %0, 128;" :: "r"(2 + (hw & 3)) : "memory");
        tm_fence_after();
        acc ^= scan_math<NG, U, 8>(v, (const float*)(smem + (size_t)S * SL), c, tq + col);
      }
      if (wr) {
        char* b = a.state + (size_t)pg * PG;
#pragma unroll
        for (int u = 0; u < U; ++u) {
#pragma unroll
          for (int k = 0; k < 8; ++k) if (!MATH) v[u][k] = __float_as_uint(__uint_as_float(v[u][k]) + 1.0f);
          STV8(b + (size_t)((ps * U + u) * NG + c) * 32, v[u]);
        }
      } else {
#pragma unroll
        for (int u = 0; u < U; ++u) acc ^= v[u][0] ^ v[u][5];
      }
    }
    }
    for (int o = c * 32; o < a.sout_b; o += NG * 32) {
      unsigned r[8] = {acc, acc, acc, acc, acc, acc, acc, acc};
      STV8(a.sout + (size_t)pg * a.sout_b + o, r); }
  }
  if constexpr (MATH == 8) {
    tm_fence_before();
    asm volatile("bar.sync 1, %0;" :: "r"(NC) : "memory");
    tm_fence_after();
    if (tid < 64) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(s_taddr) : "memory");
  }
  if (acc == 0x12345678u) a.sink[0] = 1.f;
}

template <int NC, int S, int OPS, int MATH = 0, int G = 1>
static void go_tma(const Args& a, cudaStream_t st) {
  int dev; cudaGetDevice(&dev);
  int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  size_t sm = (size_t)S * (PG + a.sin_b + a.ring_b) + (MATH ? 8704 : 0);
  cudaFuncSetAttribute(tma_k<NC, S, OPS, MATH, G>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(nsm); cfg.blockDim = dim3(NC + 32); cfg.stream = st;
  cfg.dynamicSmemBytes = sm;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, tma_k<NC, S, OPS, MATH, G>, a);
  int optin = 0; cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(e == cudaSuccess, "launch tma: ", cudaGetErrorString(e), " smem ", sm, " optin ", optin);
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
  cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();
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
    case 270531: go_tma<512, 3, 1, 2>(a, s); break;
    case 170541: go_tma<512, 4, 1, 1>(a, s); break;
    case 170241: go_tma<256, 4, 1, 1>(a, s); break;
    case 1070531: go_tma<512, 3, 1, 0, 2>(a, s); break;
    case 1070541: go_tma<512, 4, 1, 0, 2>(a, s); break;
    case 1170531: go_tma<512, 3, 1, 1, 2>(a, s); break;
    case 1170541: go_tma<512, 4, 1, 1, 2>(a, s); break;
    case 1270541: go_tma<512, 4, 1, 2, 2>(a, s); break;
    case 370531: go_tma<512, 3, 1, 3>(a, s); break;
    case 370541: go_tma<512, 4, 1, 3>(a, s); break;
    case 1370531: go_tma<512, 3, 1, 3, 2>(a, s); break;
    case 470531: go_tma<512, 3, 1, 4>(a, s); break;
    case 570531: go_tma<512, 3, 1, 5>(a, s); break;
    case 670531: go_tma<512, 3, 1, 6>(a, s); break;
    case 770531: go_tma<512, 3, 1, 7>(a, s); break;
    case 870531: go_tma<512, 3, 1, 8>(a, s); break;
    case 970531: go_tma<512, 3, 1, 9>(a, s); break;
    case 570541: go_tma<512, 4, 1, 5>(a, s); break;
    default: TORCH_CHECK(false, "form");
  }
}

std::vector<int64_t> attrs(int64_t form) {
  cudaFuncAttributes f;
  switch (form) {
    case 5121: cudaFuncGetAttributes(&f, skel_k<512, 1>); break;
    case 5122: cudaFuncGetAttributes(&f, skel_k<512, 2>); break;
    case 2564: cudaFuncGetAttributes(&f, skel_k<256, 4>); break;
    case 70531: cudaFuncGetAttributes(&f, tma_k<512, 3, 1, 0>); break;
    case 170531: cudaFuncGetAttributes(&f, tma_k<512, 3, 1, 1>); break;
    case 270531: cudaFuncGetAttributes(&f, tma_k<512, 3, 1, 2>); break;
    case 1170541: cudaFuncGetAttributes(&f, tma_k<512, 4, 1, 1, 2>); break;
    case 1170531: cudaFuncGetAttributes(&f, tma_k<512, 3, 1, 1, 2>); break;
    case 170541: cudaFuncGetAttributes(&f, tma_k<512, 4, 1, 1>); break;
    case 370531: cudaFuncGetAttributes(&f, tma_k<512, 3, 1, 3>); break;
    case 1370531: cudaFuncGetAttributes(&f, tma_k<512, 3, 1, 3, 2>); break;
    case 470531: cudaFuncGetAttributes(&f, tma_k<512, 3, 1, 4>); break;
    case 570531: cudaFuncGetAttributes(&f, tma_k<512, 3, 1, 5>); break;
    case 670531: cudaFuncGetAttributes(&f, tma_k<512, 3, 1, 6>); break;
    case 770531: cudaFuncGetAttributes(&f, tma_k<512, 3, 1, 7>); break;
    case 870531: cudaFuncGetAttributes(&f, tma_k<512, 3, 1, 8>); break;
    case 970531: cudaFuncGetAttributes(&f, tma_k<512, 3, 1, 9>); break;
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
        name="k31_kda_skel_v20",
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
            if form.startswith("t"):  # t<NC>s<S>o<OPS>[m|f][g]: TMA ring, NC consumers
                grp = 1000000 if form.endswith("g") else 0  # g: 2 alternating groups
                form = form.rstrip("g")
                nc, ops = form[1:].rstrip("mfvwpbqtx").split("o")
                nc, slots = (nc.split("s") + ["3"])[:2]
                code = (5 if nc == "512" else 2) * 100 + 10 * int(slots) + int(ops)
                return kind, grp + 70000 + code + {"m": 100000, "f": 200000, "v": 300000, "w": 400000, "p": 500000, "b": 600000, "q": 700000, "t": 800000, "x": 900000}.get(form[-1], 0)
            nt, minb = form.split("x")
            return kind, int(nt) * 10 + int(minb)
    raise ValueError(arm)


def run(mod, n, arm, place, args, dom):
    kind, form = parse_arm(arm)
    P = n * HEADS
    L = args.layers
    ppad = (P + 63) // 64 * 64  # even chunk count: every layer starts on die 0
    dev = torch.device("cuda", 0)
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
        regs, local = mod.attrs(form)
        print(f"{arm}: regs {regs} local {local} B", flush=True)
    dom = dies.sm_domain_table(torch.device("cuda", 0))
    print(f"SMs per die: {int((dom == 0).sum())} / {int((dom == 1).sum())}", flush=True)
    for n in args.n:
        for place in args.place:
            for arm in args.arms:
                try:
                    run(mod, n, arm, place, args, dom)
                except RuntimeError as error:
                    print(f"{arm} n{n}: {str(error).splitlines()[0][:160]}", flush=True)
                    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
