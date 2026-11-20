# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KdaRelay: KdaCore's state pass (deferred fold + verify scan, solve, gated norm)
on prepared operand blocks, as a release-fast page stream with the per-key operands
relayed through tensor memory.

Roles (one persistent CTA per SM, a static head-major item range each):
  warp 0      page producer: the item's 64 KiB FP32 state page by one bulk copy into
              a 3-slot ring (issued before the PDL wait: the state does not depend on
              the predecessor);
  warps 1-16  consumers, thread (warp w, lane 8 a + b) = rows 8 w + 2 a + {0, 1} x keys
              4 b + 32 j + [0, 4) (j < 4): the page slot is released as soon as the
              rows are in registers; fold S = S D + sum_t u_t w_t and the 8 scan dots
              (FFMA2) read their per-key operands from tensor memory; an 8-lane
              butterfly, a 4-lane gather of each row's 8 dots, the token solve, the
              records, the BF16 outputs + sums of squares to a 2-deep finish ring,
              the folded rows back to the cache (128-bit stores);
  warp 17     copier: the per-key part of the operand blocks (2 slots)
              and the per-key part to tensor memory, 32 tcgen05.cp.128x256b with a
              zero stride-dimension offset (lane L receives key group L % 8),
              double-buffered;
  warps 18-19 finishers (tokens 0-1 / 2-3): gated RMSNorm from the warps' sums of
              squares, then KdaFlow's BF16 or MXFP8 store_output (warp-collective per
              32 channels); warp 19 also loads the rest of each block (v, u, gates,
              couplings, betas; 3 slots, freed once the consumers and finishers read it).

Operand block (``OPB_FLOATS`` FP32 words per item, head-major item rows): the per-key
part [32 copies][2 halves][8 key groups][4] (key pair q of key group b = 32 columns
[x_0..7 of key k0 | x_0..7 of key k1 | D_k0 D_k1 | w_t(k0) w_t(k1) for t < 4 | 6
unused], k0 = 4 b + 32 (q // 2) + 2 (q % 2)), then v [T][K], the pending corrections
u [T][K], the output gates (BF16 [T][K]), the 16 token couplings and the T betas.
``from_core`` builds it from KdaCore's block.

Math: KdaCore's (fold, dots, solve, gated norm, store_output) with another FP32
reduction order of the dot products and of the norm's sum of squares."""

import hashlib
import os
import threading

import torch

from . import core

T, K = 4, 128
OPB_X = 2048
OPB_V, OPB_U = OPB_X, OPB_X + T * K
OPB_G = OPB_U + T * K
OPB_CE = OPB_G + T * K // 2
OPB_BETA = OPB_CE + 16
OPB_FLOATS = OPB_BETA + T  # 3348 (13392 bytes, a multiple of 16)

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {
constexpr int T = 4, K = 128, PG = K * K * 4;
constexpr int OPB_FLOATS = 3348, OPB_BYTES = OPB_FLOATS * 4;
constexpr int OPB_V = 2048, OPB_U = OPB_V + T * K, OPB_G = OPB_U + T * K;
constexpr int OPB_CE = OPB_G + T * K / 2, OPB_BETA = OPB_CE + 16;
constexpr int S = 3, NX = 2, NR = 3, NCW = 16, NC = NCW * 32;
constexpr int THREADS = 32 + NC + 96;
constexpr int X_BYTES = 8192, R_BYTES = OPB_BYTES - X_BYTES;  // per-key part / the rest
constexpr int RV = OPB_V - 2048, RU = OPB_U - 2048, RG = OPB_G - 2048, RCE = OPB_CE - 2048;
constexpr int RBETA = OPB_BETA - 2048;
constexpr int FIN_BYTES = T * K * 2 + NCW * T * 4;  // BF16 outputs + per-warp sums of squares
constexpr int MAX_ITEMS = 32;  // items per CTA (static head-major ranges)
constexpr int X_AT = S * PG, R_AT = X_AT + NX * X_BYTES, FIN_AT = R_AT + NR * R_BYTES;
constexpr int SMEM = FIN_AT + 2 * FIN_BYTES;

struct Params {
  float* state; long long slot_stride;  // FP32 words
  const float* opb;
  const int* slots; long long slots_stride;
  const int* accepted; const int* rec; const int* pend;
  float* records; long long rec_stride, head_stride;  // [rec, head, T, K]
  __nv_bfloat16* out; long long out_stride;
  unsigned char* mx_data; unsigned char* mx_scale; long long mx_stride; int mx_width;
  const float* norm; float eps;
  int heads, requests, mx;
  int* trace;  // benchmarks: per (CTA < 4, item, consumer warp) phase cycles
  const int* order;  // die-sorted items (die 0 first) or null: static head-major ranges
  const int* smdie;  // per SM: die | rank among the die's SMs << 1
  int nd0, nd1, ns0, ns1;  // items / SMs per die
};

struct Item { int item, head, request, slot, rec, pending; };

__device__ __forceinline__ Item item_info(const Params& p, int item) {
  Item it;
  it.item = item;
  it.head = item / p.requests;
  it.request = item - it.head * p.requests;
  const int offset = p.accepted[it.request] - 1;
  it.slot = p.slots[(long long)it.request * p.slots_stride + offset];
  it.rec = p.rec[it.request];
  int pending = it.rec >= 0 ? p.pend[it.rec] : 0;
  it.pending = pending > T ? T : (pending < 0 ? 0 : pending);
  return it;
}

__device__ __forceinline__ unsigned sa(const void* q) {
  return (unsigned)__cvta_generic_to_shared(q); }
__device__ __forceinline__ void wait_par(unsigned b, unsigned ph) {
  asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W;}"
               :: "r"(b), "r"(ph) : "memory"); }
__device__ __forceinline__ void arrive(unsigned b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void expect_load(unsigned dst, const void* src, unsigned bytes, unsigned bar) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ float2 ffma2(float2 a, float2 b, float2 c) {
  float2 d;
  asm("{.reg .b64 ra, rb, rc, rd;\n mov.b64 ra, {%2, %3};\n mov.b64 rb, {%4, %5};\n"
      " mov.b64 rc, {%6, %7};\n fma.rn.f32x2 rd, ra, rb, rc;\n mov.b64 {%0, %1}, rd;}"
      : "=f"(d.x), "=f"(d.y) : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y), "f"(c.x), "f"(c.y));
  return d;
}
__device__ __forceinline__ void tm_ld16(unsigned t, float* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
               : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]), "=f"(r[5]), "=f"(r[6]), "=f"(r[7]),
                 "=f"(r[8]), "=f"(r[9]), "=f"(r[10]), "=f"(r[11]), "=f"(r[12]), "=f"(r[13]), "=f"(r[14]), "=f"(r[15])
               : "r"(t) : "memory");
}
__device__ __forceinline__ void tm_ld8(unsigned t, float* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]), "=f"(r[5]), "=f"(r[6]), "=f"(r[7])
               : "r"(t) : "memory");
}
__device__ __forceinline__ void tm_ld2(unsigned t, float* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];" : "=f"(r[0]), "=f"(r[1]) : "r"(t) : "memory");
}
__device__ __forceinline__ void tm_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tm_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tm_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
// K-major SWIZZLE_NONE source: LBO 128 B, SBO 0 (the 8-row groups alias), version 1.
__device__ __forceinline__ unsigned long long cp_desc(unsigned saddr) {
  return ((unsigned long long)(1u << 14) << 32) | ((saddr >> 4) & 0x3FFFu) | ((128u >> 4) << 16);
}
__device__ __forceinline__ void tm_cp(unsigned taddr, unsigned long long desc) {
  asm volatile("{.reg .pred p; elect.sync _|p, 0xffffffff;\n"
               "@p tcgen05.cp.cta_group::1.128x256b [%0], %1;}" :: "r"(taddr), "l"(desc) : "memory");
}
__device__ __forceinline__ void tm_commit(unsigned bar) {
  asm volatile("{.reg .pred p; elect.sync _|p, 0xffffffff;\n"
               "@p tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];}"
               :: "r"(bar) : "memory");
}
__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float triton_exp(float x) {
  float r; asm("{ .reg .f32 t; mul.f32 t,%1,0f3FB8AA3B; ex2.approx.ftz.f32 %0,t; }" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("{ .reg .f32 a,b,r; .reg .pred big; mov.f32 a,%1; mov.f32 b,%2; "
      "setp.gt.f32 big,b,0f7E800000; @big mul.f32 a,a,0f3E800000; "
      "@big mul.f32 b,b,0f3E800000; rcp.approx.ftz.f32 r,b; mul.f32 %0,a,r; }" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float rcp_approx(float x) { float r; asm("rcp.approx.ftz.f32 %0,%1;" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ float sqrt_approx(float x) { float r; asm("sqrt.approx.ftz.f32 %0,%1;" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ unsigned ue8m0(float v) {
  unsigned r;
  asm("{ .reg .pred pz, pm, pe, pt, po; .reg .u32 b, e, m, u, r; "
      "setp.le.f32 pz, %1, 0f00000000; mov.b32 b, %1; shr.b32 e, b, 23; "
      "and.b32 e, e, 255; and.b32 m, b, 0x7FFFFF; setp.ne.u32 pm, m, 0; "
      "selp.u32 u, 1, 0, pm; setp.eq.u32 pe, e, 0; setp.le.u32 pt, m, 0x400000; "
      "and.pred pt, pe, pt; @pt mov.u32 u, 0; add.u32 r, e, u; "
      "setp.gt.u32 po, r, 254; selp.u32 r, 254, r, po; selp.u32 %0, 0, r, pz; }" : "=r"(r) : "f"(v));
  return r;
}
__device__ __forceinline__ unsigned inverse_scale_bits(unsigned e) {
  unsigned r;
  asm("{ .reg .pred p; .reg .s32 n; sub.s32 n, 254, %1; max.s32 n, n, 0; "
      "shl.b32 n, n, 23; setp.eq.u32 p, %1, 0; selp.b32 %0, 0, n, p; }" : "=r"(r) : "r"(e));
  return r;
}
__device__ __forceinline__ unsigned pack_fp8(float a, float b, float c, float d) {
  unsigned r;
  asm("{ .reg .b16 lo,hi; cvt.rn.satfinite.e4m3x2.f32 lo,%2,%1; "
      "cvt.rn.satfinite.e4m3x2.f32 hi,%4,%3; mov.b32 %0,{lo,hi}; }" : "=r"(r) : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
}

template <bool TRACE>
__global__ void __launch_bounds__(THREADS, 1) relay_k(Params p) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) unsigned long long full[S], empty[S], xfull[NX], xdone[NX], rfull[NR], rfree[NR];
  __shared__ __align__(8) unsigned long long ofull[2], oempty[2], ffull[2], ffree[2];
  __shared__ unsigned s_taddr;
  __shared__ Item s_items[MAX_ITEMS];
  const int tid = threadIdx.x;
  const long long total = (long long)p.heads * p.requests;
  int first = (int)(total * blockIdx.x / gridDim.x);
  int count = (int)(total * (blockIdx.x + 1) / gridDim.x) - first;  // <= MAX_ITEMS
  if (p.order) {  // this SM's share of its die's items (one CTA per SM)
    unsigned sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    const int e = p.smdie[sm], d = e & 1, r = e >> 1;
    const int nd = d ? p.nd1 : p.nd0, ns = d ? p.ns1 : p.ns0, base = d ? p.nd0 : 0;
    first = base + (int)((long long)nd * r / ns);
    count = base + (int)((long long)nd * (r + 1) / ns) - first;
  }
  if (tid == 0) {
    for (int s = 0; s < S; ++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&full[s])));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(&empty[s])), "r"(NCW));
    }
    for (int x = 0; x < NX; ++x) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&xfull[x])));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&xdone[x])));
    }
    for (int r = 0; r < NR; ++r) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&rfull[r])));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(&rfree[r])), "r"(NCW + 2));
    }
    for (int b = 0; b < 2; ++b) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&ofull[b])));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(&oempty[b])), "r"(NCW));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(&ffull[b])), "r"(NCW));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" :: "r"(sa(&ffree[b])));
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (tid >= 32 && tid < 64) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(sa(&s_taddr)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  if (tid >= 64 && tid < 64 + count)
    s_items[tid - 64] = item_info(p, p.order ? p.order[first + tid - 64] : first + tid - 64);
  tm_fence_before();
  __syncthreads();
  tm_fence_after();

  if (tid < 32) {  // ---------------------------------------------------- page producer
    if (tid == 0) {
      for (int j = 0; j < count; ++j) {
        const int s = j % S;
        if (j >= S) wait_par(sa(&empty[s]), ((j / S) - 1) & 1);
        const Item it = s_items[j];
        const float* page = p.state + (long long)it.slot * p.slot_stride + (long long)it.head * K * K;
        expect_load(sa(smem + (size_t)s * PG), page, PG, sa(&full[s]));
      }
    }
    return;
  }
  if (tid >= 32 + NC + 32) {  // ------------------------------------- finishers A, B
    // Finisher A: tokens 0, 1; B: tokens 2, 3 and the R parts (item i + NR once every
    // reader of item i's slot is done). Per (token, 32-channel block): the gated RMSNorm
    // from the consumers' sums of squares, then KdaFlow's store_output (warp-collective).
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const int fw = (tid - 32 - NC - 32) >> 5, lane = tid & 31;
    auto load_r = [&](int jj) {
      if (lane == 0)
        expect_load(sa(smem + R_AT + (size_t)(jj % NR) * R_BYTES),
                    p.opb + (long long)s_items[jj].item * OPB_FLOATS + 2048, R_BYTES, sa(&rfull[jj % NR]));
    };
    if (fw == 1)
      for (int jj = 0; jj < NR && jj < count; ++jj) load_r(jj);
    float nw[4];
#pragma unroll
    for (int blk = 0; blk < 4; ++blk) nw[blk] = p.norm[32 * blk + lane];
    for (int i = 0; i < count; ++i) {
      const int fi = i & 1, ri = i % NR;
      const Item fit = s_items[i];
      wait_par(sa(&ffull[fi]), (i >> 1) & 1);
      wait_par(sa(&rfull[ri]), (i / NR) & 1);
      const unsigned char* fb = smem + FIN_AT + fi * FIN_BYTES;
      const float* sums = (const float*)(fb + T * K * 2);
      const __nv_bfloat16* gates = (const __nv_bfloat16*)((const float*)(smem + R_AT + (size_t)ri * R_BYTES) + RG);
      // 8 independent (token, 32-channel block) outputs per lane, batched stage by stage
      // (KdaFlow store_output numerics; the batches give the chains ILP 8).
      float x[8];
#pragma unroll
      for (int tt = 0; tt < 2; ++tt) {
        const int t = 2 * fw + tt;
        float sumsq = 0.f;
#pragma unroll
        for (int w = 0; w < NCW; ++w) sumsq += sums[w * T + t];
        const float rstd = rcp_approx(sqrt_approx(sumsq * (1.0f / 128) + p.eps));
#pragma unroll
        for (int blk = 0; blk < 4; ++blk) {
          const int c = 32 * blk + lane;
          const float pg = __bfloat162float(gates[t * K + c]);
          const float sig = div_full(1.0f, 1.0f + triton_exp(-pg));
          const float result = __bfloat162float(((const __nv_bfloat16*)fb)[t * K + c]) * rstd * nw[blk] * sig;
          x[tt * 4 + blk] = __bfloat162float(__float2bfloat16_rn(fit.slot > 0 ? result : 0.f));
        }
      }
      if (!p.mx) {
#pragma unroll
        for (int k = 0; k < 8; ++k)
          p.out[(long long)(fit.request * T + 2 * fw + (k >> 2)) * p.out_stride + fit.head * K + 32 * (k & 3) + lane] =
              __float2bfloat16_rn(x[k]);
      } else {
        unsigned amax[8], e[8];
#pragma unroll
        for (int k = 0; k < 8; ++k)
          asm("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(amax[k]) : "r"(__float_as_uint(x[k]) & 0x7FFFFFFFu));
#pragma unroll
        for (int k = 0; k < 8; ++k) e[k] = ue8m0(__uint_as_float(amax[k]) * __uint_as_float(0x3B124925u));
        float q[8], q1[8], q2[8], q3[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = x[k] * __uint_as_float(inverse_scale_bits(e[k]));
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          q1[k] = __shfl_down_sync(0xffffffffu, q[k], 1);
          q2[k] = __shfl_down_sync(0xffffffffu, q[k], 2);
          q3[k] = __shfl_down_sync(0xffffffffu, q[k], 3);
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int row = fit.request * T + 2 * fw + (k >> 2), column = fit.head * K + 32 * (k & 3) + lane;
          unsigned word = pack_fp8(q[k], q1[k], q2[k], q3[k]);
          if (word == 0x80000000u) word = 0;
          if ((lane & 3) == 0) *(unsigned*)(p.mx_data + (long long)row * p.mx_stride + column) = word;
          if (lane == 0) {
            const long long blk = column / 32, r = row;
            const long long at = (blk & 3) + (blk >> 2) * 512 + (r & 31) * 16 + ((r & 127) >> 5) * 4
                                 + (r >> 7) * (128LL * (p.mx_width / 32));
            p.mx_scale[at] = (unsigned char)e[k];
          }
        }
      }
      __syncwarp();
      if (lane == 0) {
        arrive(sa(&ffree[fi]));
        arrive(sa(&rfree[ri]));
      }
      if (fw == 1 && i + NR < count) {
        wait_par(sa(&rfree[ri]), (i / NR) & 1);
        load_r(i + NR);
      }
    }
    return;
  }
  if (tid >= 32 + NC) {  // ------------------------------------------------- copier
    asm volatile("griddepcontrol.wait;" ::: "memory");  // the blocks come from the prep
    if (s_taddr != 0) __trap();
    const int lane = tid & 31;
    auto load_x = [&](int jj) {  // the per-key part of item jj's block
      if (lane == 0)
        expect_load(sa(smem + X_AT + (size_t)(jj % NX) * X_BYTES), p.opb + (long long)s_items[jj].item * OPB_FLOATS,
                    X_BYTES, sa(&xfull[jj % NX]));
    };
    if (count > 0) load_x(0);
    for (int j = 0; j < count; ++j) {
      const int x = j % NX, b = j & 1;
      if (j + 1 < count) {  // item j + 1's per-key part, once copy j - 1 has read its slot
        if (j >= 1) wait_par(sa(&xdone[(j + 1) % NX]), ((j - 1) / NX) & 1);
        load_x(j + 1);
      }
      wait_par(sa(&xfull[x]), (j / NX) & 1);
      if (j >= 2) wait_par(sa(&oempty[b]), ((j >> 1) - 1) & 1);
      tm_fence_after();
      const unsigned src = sa(smem + X_AT + (size_t)x * X_BYTES), dst = 256u * b;
#pragma unroll
      for (int k = 0; k < 32; ++k) tm_cp(dst + 8 * k, cp_desc(src + 256 * k));
      tm_commit(sa(&ofull[b]));
      tm_commit(sa(&xdone[x]));
    }
    return;
  }
  // ------------------------------------------------------------------------ consumers
  asm volatile("griddepcontrol.wait;" ::: "memory");  // records: the prep reads the old ones
  const int c = tid - 32, cw = c >> 5, l = c & 31, b = l & 7, q4 = b & 3, hh = (b >> 2) & 1;
  const int r0 = 8 * cw + 2 * (l >> 3);
  const unsigned tq = (32u * ((tid >> 5) & 3)) << 16;
  for (int j = 0; j < count; ++j) {
    const int s = j % S, rs = j % NR, bq = j & 1, f = j & 1;
    const Item it = s_items[j];
    long long t0 = 0;
    int* trc = nullptr;
    if (TRACE && blockIdx.x < 4 && l == 0) { trc = p.trace + ((blockIdx.x * MAX_ITEMS + j) * NCW + cw) * 10; t0 = clock64(); }
#define STAMP(i) if (TRACE && trc) { const long long now = clock64(); trc[i] = (int)(now - t0); t0 = now; }
    if (TRACE && blockIdx.x == 0 && c == 0 && (j == 0 || j == count - 1)) {
      unsigned long long gt; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt));
      long long* ck = (long long*)(p.trace + 4 * MAX_ITEMS * NCW * 10) + (j == 0 ? 0 : 2);
      ck[0] = clock64(); ck[1] = (long long)gt;
    }
    wait_par(sa(&full[s]), (j / S) & 1);
    STAMP(0)
    float4 v[2][4];
    const unsigned char* slot = smem + (size_t)s * PG;
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int q = 0; q < 4; ++q) v[h][q] = *(const float4*)(slot + (size_t)(r0 + h) * 512 + q * 128 + b * 16);
    __syncwarp();
    if (l == 0) arrive(sa(&empty[s]));
    STAMP(1)
    wait_par(sa(&rfull[rs]), (j / NR) & 1);
    const float* ob = (const float*)(smem + R_AT + (size_t)rs * R_BYTES);
    float ut[2][4];
#pragma unroll
    for (int t = 0; t < T; ++t) {
      const float2 u = *(const float2*)(ob + RU + t * K + r0);
      ut[0][t] = u.x; ut[1][t] = u.y;
    }
    STAMP(2)
    wait_par(sa(&ofull[bq]), (j >> 1) & 1);
    STAMP(3)
    tm_fence_after();
    const unsigned tb = tq + 256u * bq;
    float2 acc[2][4];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int m = 0; m < 4; ++m) acc[h][m] = make_float2(0.f, 0.f);
    float A[10], B[16];
    tm_ld8(tb + 16, A);
    tm_ld2(tb + 24, A + 8);
    tm_wait_ld();
    tm_ld16(tb, B);
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const int jq = q >> 1;
      float2 n[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float2 fc = make_float2(-0.f, -0.f);
#pragma unroll
        for (int t = 3; t >= 0; --t)
          fc = ffma2(make_float2(ut[h][t], ut[h][t]), make_float2(A[2 + 2 * t], A[3 + 2 * t]), fc);
        const float2 st = (q & 1) ? make_float2(v[h][jq].z, v[h][jq].w) : make_float2(v[h][jq].x, v[h][jq].y);
        n[h] = ffma2(st, make_float2(A[0], A[1]), fc);
        if (q & 1) { v[h][jq].z = n[h].x; v[h][jq].w = n[h].y; } else { v[h][jq].x = n[h].x; v[h][jq].y = n[h].y; }
      }
      tm_wait_ld();  // B of pair q
      if (q < 7) {
        tm_ld8(tb + 32 * (q + 1) + 16, A);
        tm_ld2(tb + 32 * (q + 1) + 24, A + 8);
      }
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          const float nv = kk ? n[h].y : n[h].x;
#pragma unroll
          for (int m = 0; m < 4; ++m)
            acc[h][m] = ffma2(make_float2(nv, nv), make_float2(B[8 * kk + 2 * m], B[8 * kk + 2 * m + 1]), acc[h][m]);
        }
      tm_wait_ld();  // A of pair q + 1
      if (q < 7) tm_ld16(tb + 32 * (q + 1), B);
    }
    tm_fence_before();
    __syncwarp();
    if (l == 0) arrive(sa(&oempty[bq]));
    if (it.slot > 0 && it.pending > 0) {
      float* g = p.state + (long long)it.slot * p.slot_stride + (long long)it.head * K * K;
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int q = 0; q < 4; ++q) *(float4*)(g + (r0 + h) * K + q * 32 + b * 4) = v[h][q];
    }
    const int row = r0 + hh;
    float ce[16], vt[T], beta[T];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float4 c4 = *(const float4*)(ob + RCE + 4 * i);
      ce[4 * i] = c4.x; ce[4 * i + 1] = c4.y; ce[4 * i + 2] = c4.z; ce[4 * i + 3] = c4.w;
    }
    {
      const float4 b4 = *(const float4*)(ob + RBETA);
      beta[0] = b4.x; beta[1] = b4.y; beta[2] = b4.z; beta[3] = b4.w;
    }
#pragma unroll
    for (int t = 0; t < T; ++t) vt[t] = ob[RV + t * K + row];
    STAMP(4)
    // 8-lane transposed butterfly: lane b keeps dots 2 q4 + {0, 1} of row r0 + hh
    float vals[16];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int m = 0; m < 4; ++m) { vals[h * 8 + 2 * m] = acc[h][m].x; vals[h * 8 + 2 * m + 1] = acc[h][m].y; }
    int nn = 16;
#pragma unroll
    for (int off = 4; off >= 1; off >>= 1) {
      const bool up = (l & off) != 0;
      const int half = nn / 2;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (i < half) {
          const float send = up ? vals[i] : vals[i + half];
          const float keep = up ? vals[i + half] : vals[i];
          vals[i] = keep + __shfl_xor_sync(0xffffffffu, send, off);
        }
      }
      nn = half;
    }
    // the row's 8 dots on all 4 lanes of the row
    const float e0 = __shfl_xor_sync(0xffffffffu, vals[0], 1), e1 = __shfl_xor_sync(0xffffffffu, vals[1], 1);
    const bool odd = (q4 & 1) != 0, hi = (q4 & 2) != 0;
    const float f0 = odd ? e0 : vals[0], f1 = odd ? e1 : vals[1], f2 = odd ? vals[0] : e0, f3 = odd ? vals[1] : e1;
    const float g0 = __shfl_xor_sync(0xffffffffu, f0, 2), g1 = __shfl_xor_sync(0xffffffffu, f1, 2);
    const float g2 = __shfl_xor_sync(0xffffffffu, f2, 2), g3 = __shfl_xor_sync(0xffffffffu, f3, 2);
    const float D[8] = {hi ? g0 : f0, hi ? g1 : f1, hi ? g2 : f2, hi ? g3 : f3,
                        hi ? f0 : g0, hi ? f1 : g1, hi ? f2 : g2, hi ? f3 : g3};
    // the token solve (KdaFlow.solve): dot_t = A_t + sum_s<t c_ts u_s, u_t = (v_t - dot_t) beta_t,
    // out_t = B_t + sum_s<=t e_ts u_s (BF16-rounded)
    float u[T], po[T];
    int at = 0;
#pragma unroll
    for (int t = 0; t < T; ++t) {
      float dot = D[t];
#pragma unroll
      for (int s2 = 0; s2 < t; ++s2) dot = fmaf(ce[at + s2], u[s2], dot);
      u[t] = (vt[t] - dot) * beta[t];
      at += t + (t + 1);
    }
    at = 0;
#pragma unroll
    for (int t = 0; t < T; ++t) {
      float o2 = D[T + t];
#pragma unroll
      for (int s2 = 0; s2 <= t; ++s2) o2 = fmaf(ce[at + t + s2], u[s2], o2);
      po[t] = bf16_round(o2);
      at += t + (t + 1);
    }
    float mine_u = u[0], mine_o = po[0];
#pragma unroll
    for (int t = 1; t < T; ++t) {
      mine_u = q4 == t ? u[t] : mine_u;
      mine_o = q4 == t ? po[t] : mine_o;
    }
    if (it.slot > 0 && it.rec >= 0)
      p.records[(long long)it.rec * p.rec_stride + (long long)it.head * p.head_stride + q4 * K + row] = mine_u;
    STAMP(5)
    if (j >= 2) wait_par(sa(&ffree[f]), ((j >> 1) - 1) & 1);
    unsigned char* fb = smem + FIN_AT + f * FIN_BYTES;
    ((__nv_bfloat16*)fb)[q4 * K + row] = __float2bfloat16_rn(mine_o);
    float sq = mine_o * mine_o;
    sq += __shfl_xor_sync(0xffffffffu, sq, 4);
    sq += __shfl_xor_sync(0xffffffffu, sq, 8);
    sq += __shfl_xor_sync(0xffffffffu, sq, 16);
    if (l < 4) ((float*)(fb + T * K * 2))[cw * T + q4] = sq;
    __syncwarp();
    if (l == 0) {
      arrive(sa(&ffull[f]));
      arrive(sa(&rfree[rs]));
    }
    STAMP(6)
#undef STAMP
  }
  tm_fence_before();
  asm volatile("bar.sync 1, %0;" :: "r"(NC) : "memory");
  tm_fence_after();
  if (tid < 64) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(s_taddr) : "memory");
}
}  // namespace

void relay(int64_t state, int64_t slot_stride, int64_t opb, int64_t slots, int64_t slots_stride,
           int64_t accepted, int64_t rec, int64_t pend, int64_t records, int64_t rec_stride,
           int64_t head_stride, int64_t out, int64_t out_stride, int64_t mx_data, int64_t mx_scale,
           int64_t mx_stride, int64_t mx_width, int64_t norm, double eps, int64_t heads,
           int64_t requests, int64_t ctas, int64_t trace, int64_t order, int64_t smdie,
           int64_t nd0, int64_t nd1, int64_t ns0, int64_t ns1) {
  TORCH_CHECK((heads * requests + ctas - 1) / ctas <= MAX_ITEMS, "KdaRelay: more than ",
              MAX_ITEMS, " items per CTA");
  TORCH_CHECK(!order || ((nd0 + ns0 - 1) / ns0 <= MAX_ITEMS && (nd1 + ns1 - 1) / ns1 <= MAX_ITEMS),
              "KdaRelay: more than ", MAX_ITEMS, " items per CTA on one die");
  Params p{(float*)state, slot_stride, (const float*)opb, (const int*)slots, slots_stride,
           (const int*)accepted, (const int*)rec, (const int*)pend, (float*)records, rec_stride,
           head_stride, (__nv_bfloat16*)out, out_stride, (unsigned char*)mx_data,
           (unsigned char*)mx_scale, mx_stride, (int)mx_width, (const float*)norm, (float)eps,
           (int)heads, (int)requests, mx_data != 0, (int*)trace, (const int*)order,
           (const int*)smdie, (int)nd0, (int)nd1, (int)ns0, (int)ns1};
  static bool configured = false;
  if (!configured) {
    for (const void* k : {(const void*)relay_k<false>, (const void*)relay_k<true>})
      TORCH_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess,
                  "KdaRelay: ", SMEM, " bytes of shared memory");
    configured = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)ctas);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = c10::cuda::getCurrentCUDAStream().stream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  const cudaError_t e = trace ? cudaLaunchKernelEx(&cfg, relay_k<true>, p) : cudaLaunchKernelEx(&cfg, relay_k<false>, p);
  TORCH_CHECK(e == cudaSuccess, "KdaRelay launch: ", cudaGetErrorString(e));
}

std::vector<int64_t> relay_attrs() {
  cudaFuncAttributes f;
  cudaFuncGetAttributes(&f, relay_k<false>);
  return {f.numRegs, (int64_t)f.localSizeBytes, SMEM};
}
"""

_CPP_SRC = r"""
#include <vector>
void relay(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
           int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
           double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
           int64_t, int64_t);
std::vector<int64_t> relay_attrs();
"""

_lock = threading.Lock()
_mod = None


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
            build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_kda_relay", tag)
            os.makedirs(build, exist_ok=True)
            _mod = load_inline(
                name=f"k3_kda_relay_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["relay", "relay_attrs"],
                extra_cuda_cflags=["-O3", "-gencode", "arch=compute_100f,code=sm_100f"],
                build_directory=build,
                verbose=False,
            )
    return _mod


def _core_index():
    """For each relay block word, the KdaCore block word it copies (unused words: 0)."""
    index = torch.zeros(OPB_FLOATS, dtype=torch.long)
    for b in range(8):
        for q in range(8):
            k0 = 4 * b + 32 * (q // 2) + 2 * (q % 2)
            cols = [0] * 32
            for kk in range(2):
                key = k0 + kk
                chunk, lane_in, e = key // 32, (key // 4) % 8, key % 4
                for h in range(2):
                    for i in range(4):
                        pos = (((chunk * 4 + e) * 2 + h) * 8 + lane_in) * 4 + i
                        cols[8 * kk + 4 * h + i] = core.X_AT + pos
                cols[16 + kk] = core.D_AT + key
                for t in range(T):
                    cols[18 + 2 * t + kk] = core.W_AT + t * K + key
            for col in range(26):
                cp, rem = divmod(32 * q + col, 8)
                index[cp * 64 + (rem // 4) * 32 + b * 4 + rem % 4] = cols[col]
    index[OPB_V : OPB_V + T * K] = torch.arange(core.V_AT, core.V_AT + T * K)
    index[OPB_U : OPB_U + T * K] = torch.arange(core.U_AT, core.U_AT + T * K)
    index[OPB_G : OPB_G + T * K // 2] = torch.arange(core.G_AT, core.G_AT + T * K // 2)
    index[OPB_CE : OPB_CE + 16] = torch.arange(core.CE_AT, core.CE_AT + 16)
    index[OPB_BETA : OPB_BETA + T] = torch.arange(core.BETA_AT, core.BETA_AT + T)
    return index


_INDEX = {}


def from_core(core_blocks, out):
    """KdaCore's operand blocks [items, core.OPB_FLOATS] -> relay blocks in ``out``."""
    device = core_blocks.device
    if device not in _INDEX:
        _INDEX[device] = _core_index().to(device)
    torch.index_select(core_blocks, 1, _INDEX[device], out=out)
    return out


class KdaRelay:
    """The relay state pass of one step shape: ``launch(tensors)`` with SpKdaCore's
    operand tuple (no scalars), the MXFP8 output pair when ``mx``, then the blocks."""

    def __init__(self, heads, tokens, eps, mx, device):
        if tokens != T:
            raise ValueError(f"KdaRelay runs q{T} only")
        self.heads, self.eps, self.mx = heads, float(eps), mx
        self.ctas = torch.cuda.get_device_properties(device).multi_processor_count
        self.trace = None  # benchmarks: an int32 buffer of phase cycles (see TRACE)
        self.die_order = False  # benchmarks: die-matched items (static metadata)
        self._dies = {}
        _module()

    def die_args(self, state, slots, accepted, requests):
        """(order, SM table, items and SMs per die) of a die-matched item order when the
        state lives in alternating-die 2 MiB chunks; zeros (static ranges) otherwise.
        The order is built once per metadata tensor pair (benchmarks: static steps)."""
        from ..common import dies

        parity = dies.alternating_parity(state) if self.die_order else None
        if parity is None:
            return (0, 0, 0, 0, 0, 0)
        key = (state.data_ptr(), slots.data_ptr(), accepted.data_ptr(), requests)
        if key not in self._dies:
            req = torch.arange(requests, device=state.device)
            slot = slots[req, accepted[:requests].long() - 1].long()
            head = torch.arange(self.heads, device=state.device)
            word = slot[None, :] * state.stride(0) + head[:, None] * (K * K)
            addr = state.data_ptr() + 4 * word.flatten()  # item = head * requests + request
            die = ((addr >> 21) ^ parity) & 1
            order = torch.argsort(die, stable=True).to(torch.int32)
            nd1 = int(die.sum())
            domains = dies.sm_domains(state.device.index)
            ranks = [sum(1 for e in domains[:i] if e == d) for i, d in enumerate(domains)]
            table = torch.tensor([(d & 1) | (r << 1) for d, r in zip(domains, ranks)],
                                 dtype=torch.int32, device=state.device)  # fmt: skip
            ns1 = sum(1 for d in domains if d & 1)
            self._dies[key] = (order, table, order.numel() - nd1, nd1, len(domains) - ns1, ns1)
        order, table, *counts = self._dies[key]
        return (order.data_ptr(), table.data_ptr(), *counts)

    def launch(self, tensors):
        slots, state, out, norm = tensors[8], tensors[9], tensors[10], tensors[12]
        accepted, records, rec, pend = tensors[14], tensors[15], tensors[17], tensors[18]
        opb = tensors[-1]
        requests = out.shape[0] // T
        assert state.dtype == torch.float32 and state.stride(1) == K * K
        assert opb.shape[1] == OPB_FLOATS and opb.is_contiguous()
        assert records.stride(3) == 1 and records.stride(2) == K
        if self.mx:
            data, scale = tensors[19], tensors[20]
            mx = (data.data_ptr(), scale.data_ptr(), data.stride(0), out.shape[1])
            bf16 = (0, 0)
        else:
            mx = (0, 0, 0, 0)
            bf16 = (out.data_ptr(), out.stride(0))
        _module().relay(
            state.data_ptr(), state.stride(0), opb.data_ptr(), slots.data_ptr(),
            slots.stride(0), accepted.data_ptr(), rec.data_ptr(), pend.data_ptr(),
            records.data_ptr(), records.stride(0), records.stride(1), *bf16, *mx,
            norm.data_ptr(), self.eps, self.heads, requests, self.ctas,
            0 if self.trace is None else self.trace.data_ptr(),
            *self.die_args(state, slots, accepted, requests),
        )  # fmt: skip
        return out
