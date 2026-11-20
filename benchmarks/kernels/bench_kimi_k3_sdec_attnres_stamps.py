# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP decode (D3-32, TP4): CALIBRATED skeleton of the SP layer boundary,
with the AttnRes math (SOL protocol: the skeleton configured as today's kernels must
reproduce today's measured kernel time; the new design is measured the same way).

Per layer the chain is the four boundary GEMMs (weight-streaming stand-ins with a
GEMM-sized shared-memory footprint, so one CTA per SM as a real mainloop) and the
exchange:

``today``  A o_proj | RED | B latent_down | C latent_up | GATH | D kda in_proj
           RED  = SpReduceAttnRes: one 7-CTA cluster per shard row, 128 threads x 8
                  columns; pre() saved-block stats (DSMEM exchange), PDL wait, push
                  the o-proj partial rows to the 4 owners (Lamport), poll the 4 slots,
                  FP32 sum in rank order, re-arm, post() (second DSMEM exchange),
                  prefix + output rows, re-arm the consumed gather rows.
           GATH = SpAttnResGather: pre(), PDL wait, post(delta = MLP row), multicast
                  the normed row into every rank's gather box, poll the 4 rows.
           GEMMs weights only, nothing before the PDL wait (cuBLAS / nvjet form).
``fz``     no exchange kernel; the exchange lives in the GEMMs (row-owner clusters of
           ``rs`` CTAs, DSMEM stats):
           A  pre-wait weights, epilogue pushes the partial rows to the owners
           B  pre-wait weights + pre() of the owned row, PDL wait, owner poll/sum/
              re-arm/post(), activation-ready grid flag, rest of the weights
           C  pre-wait weights + pre(), weights, MLP rows out, row-complete flag,
              owners post() + multicast into the gather boxes
           D  pre-wait weights, poll the gathered rows, grid flag, rest of the weights
``w``      the four GEMMs weights only (exposed = chain - w at the same pre-wait KB).

Kernel time = per launch, last CTA end - first CTA start (globaltimer), averaged over
layers (median over replays) - the in-step profiler's kernel time.

usage: torchrun --nproc-per-node 4 bench_kimi_k3_sdec_sp_boundary_cal.py
       [--rows 128,256,384] [--pre-kb 0,96,192] [--rs 2] [--nb 8]
"""

import argparse
import inspect
import hashlib
import os

import torch
import torch.distributed as dist

H = 7168
WORDS = H // 2
RMAX = 384
NMAX = RMAX // 4
NBMAX = 16
W_BYTES = {
    "A": 7168 * 3072 * 2,
    "B": 3584 * 7168 * 2,
    "C": 7168 * 3584 * 2,
    "D": 12448 * 7168 * 2,
}
KNAMES = ["A", "B", "C", "D", "RED", "GATH"]

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <stdint.h>

#define SENT 0xFFFFFFFFu
#define HID 7168
#define WORDS 3584
#define SEGS 56
#define NBMAX 16
#ifndef NB
#define NB 8
#endif

struct Prm {
  int early, do_w, do_x, rank, R, n, par, nmax, rmax, chunks, cs, layer, G;
  const uint4* w; long long w16, pre16;
  unsigned long long peer[4];
  uint32_t* box; unsigned long long mc;
  const uint32_t* blocks;   // [nmax][NBMAX][WORDS]
  uint32_t* prefix;         // [nmax][WORDS]
  const uint32_t* normw; const uint32_t* query; const uint32_t* outnorm;  // [WORDS]
  const uint32_t* partial;  // [rmax][WORDS]
  uint32_t* act;            // [nmax][WORDS]  post() output (latent_down input)
  uint32_t* mlp;            // [nmax][WORDS]  latent_up output (gather-side delta)
  unsigned int* ctr;        // [layers][4]
  unsigned long long* ts;   // [layers][6][8] (tflush)
  uint32_t* sink;
  // co-runner (successors of RED / GATH only): 0 off, 1 TMA stream, 2 FMA spin
  int co, co_ns, co_nst; long long co_bytes, co_win, co_bufb; const char* co_buf;
};

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ uint32_t stream_w(const uint4* w, long long b, long long e) {
  uint32_t acc = 0;
  const long long step = (long long)blockDim.x * 8;
  for (long long i = b + threadIdx.x; i < e; i += step) {
    uint4 v[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      long long j = i + (long long)u * blockDim.x;
      v[u] = j < e ? __ldcs(w + j) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < 8; ++u) acc ^= v[u].x ^ v[u].w;
  }
  return acc;
}
static int g_co = 0, g_co_ns = 0;
static long long g_co_bytes = 0, g_co_win = 0, g_co_bufb = 0;
static const char* g_co_buf = nullptr;
// ---- co-runner: a pre-wait successor beside the exchange (warp 0 waits, thread 32 streams)
#define CO_CHUNK 16384
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_wait(uint32_t bar, uint32_t ph) {
  uint32_t ok = 0;
  while (!ok)
    asm volatile("{ .reg .pred q; mbarrier.try_wait.parity.shared::cta.b64 q, [%1], %2; selp.u32 %0, 1, 0, q; }"
                 : "=r"(ok) : "r"(bar), "r"(ph) : "memory");
}
// TMA ring of `nst` 16 KB stages; chunk i is issued no earlier than t0 + i * ns (pacing);
// *done counts completed chunks. L2 mode (win > 0): all CTAs cycle through one window.
__device__ void co_stream(const Prm& p, char* ring, uint64_t* bar, int nst, volatile int* done) {
  const long long total = p.co_bytes / CO_CHUNK;
  const long long nwin = (p.co_win > 0 ? p.co_win : p.co_bufb) / CO_CHUNK;
  const long long base = p.co_win > 0 ? (long long)blockIdx.x * 151
                                      : ((long long)p.layer * gridDim.x + blockIdx.x) * total;
  const unsigned long long t0 = gtime();
  for (long long i = 0; i < total; ++i) {
    const int s = (int)(i % nst);
    if (i >= nst) { mb_wait(sa(bar + s), (uint32_t)((i / nst - 1) & 1)); *done = (int)(i - nst + 1); }
    if (p.co_ns) while (gtime() < t0 + (unsigned long long)i * p.co_ns) {}
    const char* src = p.co_buf + ((base + i) % nwin) * CO_CHUNK;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(bar + s)), "r"(CO_CHUNK) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(sa(ring + (long long)s * CO_CHUNK)), "l"(src), "r"(CO_CHUNK), "r"(sa(bar + s)) : "memory");
  }
  for (long long i = total > nst ? total - nst : 0; i < total; ++i) {
    mb_wait(sa(bar + (int)(i % nst)), (uint32_t)((i / nst) & 1)); *done = (int)(i + 1);
  }
}
__device__ __forceinline__ uint4 ld_vol(const uint32_t* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_vol(uint32_t* p, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
               :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void st_mc(unsigned long long p, uint4 v) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
               :: "l"(p), "f"(__uint_as_float(v.x)), "f"(__uint_as_float(v.y)),
                  "f"(__uint_as_float(v.z)), "f"(__uint_as_float(v.w)) : "memory");
}
__device__ __forceinline__ bool armed(uint4 v) {
  return v.x == SENT || v.y == SENT || v.z == SENT || v.w == SENT;
}
__device__ __forceinline__ uint32_t posz(uint32_t w) {  // -0 halves -> +0
  if ((w & 0xFFFFu) == 0x8000u) w &= 0xFFFF0000u;
  if ((w & 0xFFFF0000u) == 0x80000000u) w &= 0x0000FFFFu;
  return w;
}
__device__ __forceinline__ uint4 posz4(uint4 v) {
  return make_uint4(posz(v.x), posz(v.y), posz(v.z), posz(v.w));
}
__device__ __forceinline__ void unpack(uint4 v, float* o) {
  uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) { o[2 * j] = __uint_as_float(w[j] << 16); o[2 * j + 1] = __uint_as_float(w[j] & 0xFFFF0000u); }
}
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint4 pack8(const float* f) {
  return make_uint4(pack2(f[0], f[1]), pack2(f[2], f[3]), pack2(f[4], f[5]), pack2(f[6], f[7]));
}
__device__ __forceinline__ float dot8(const float* a, const float* b) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) s = fmaf(a[i], b[i], s);
  return s;
}
__device__ __forceinline__ float rstd(float ss, float eps) { return rsqrtf(fmaf(ss, 1.f / 7168.f, eps)); }
__device__ __forceinline__ uint32_t cl_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void cl_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }
__device__ __forceinline__ void cl_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void st_remote(float* local, uint32_t peer, float v) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(local), r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(peer));
  asm volatile("st.shared::cluster.f32 [%0], %1;" :: "r"(r), "f"(v) : "memory");
}

struct Xs { float wsum[16 * 16]; float x[3 * 8 * 16]; unsigned long long bar[3]; };

// One-sided DSMEM exchange (as ClusterAttnRes): every CTA pushes its partials into each
// peer's slot with st.async completing on the peer's round mbarrier (expect_tx armed
// at init); before the cluster's start barrier.
__device__ __forceinline__ void xs_init(Xs& s, int cs, int cnt0, int cnt1) {
  if (threadIdx.x == 0 && cs > 1) {
    const int cnt[3] = {cnt0, cnt1, 1};
    for (int r = 0; r < 3; ++r) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&s.bar[r]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(b) : "memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(cs * cnt[r] * 4) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
}
__device__ __forceinline__ void push_async(float* local, unsigned long long* bar, uint32_t peer, float v) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(local), b = (uint32_t)__cvta_generic_to_shared(bar), ra, rb;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(a), "r"(peer));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rb) : "r"(b), "r"(peer));
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
               :: "r"(ra), "r"(__float_as_uint(v)), "r"(rb) : "memory");
}
__device__ __forceinline__ void mb_wait(unsigned long long* bar) {
  uint32_t b = (uint32_t)__cvta_generic_to_shared(bar), done = 0;
  while (!done) {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                 : "=r"(done) : "r"(b) : "memory");
  }
}

// Cluster-sum CNT per-thread partials (round r); every thread gets the totals.
template <int CNT>
__device__ __forceinline__ void exchange(float* vals, float* tot, Xs& s, int r, int cs) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
  for (int i = 0; i < CNT; ++i) {
    float x = vals[i];
#pragma unroll
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xFFFFFFFFu, x, o);
    if (lane == 0) s.wsum[warp * 16 + i] = x;
  }
  __syncthreads();
  const uint32_t me = cs > 1 ? cl_rank() : 0;
  if (threadIdx.x < CNT) {
    float v = 0.f;
    for (int w = 0; w < nw; ++w) v += s.wsum[w * 16 + threadIdx.x];
    float* slot = &s.x[(r * 8 + me) * 16 + threadIdx.x];
    if (cs > 1) { for (int p = 0; p < cs; ++p) push_async(slot, &s.bar[r], p, v); } else { *slot = v; }
  }
  if (cs > 1) mb_wait(&s.bar[r]); else __syncthreads();
#pragma unroll
  for (int i = 0; i < CNT; ++i) {
    float t = 0.f;
    for (int p = 0; p < cs; ++p) t += s.x[(r * 8 + p) * 16 + i];
    tot[i] = t;
  }
}

struct St { float mix[8]; float mx, es; float wq[8]; };

// pre(): saved-block statistics + softmax mixture of this thread's 8 columns.
__device__ __forceinline__ void attn_pre(const Prm& p, int row, int col, bool act, St& st, Xs& s) {
  float v[NB][8];
  float vals[2 * NB];
  if (act) {
    float nw[8], qw[8];
    unpack(*reinterpret_cast<const uint4*>(p.normw + col / 2), nw);
    unpack(*reinterpret_cast<const uint4*>(p.query + col / 2), qw);
#pragma unroll
    for (int i = 0; i < 8; ++i) st.wq[i] = nw[i] * qw[i];
#pragma unroll
    for (int b = 0; b < NB; ++b)
      unpack(*reinterpret_cast<const uint4*>(p.blocks + ((long long)row * NBMAX + b) * WORDS + col / 2), v[b]);
#pragma unroll
    for (int b = 0; b < NB; ++b) { vals[b] = dot8(v[b], v[b]); vals[NB + b] = dot8(v[b], st.wq); }
  } else {
#pragma unroll
    for (int b = 0; b < 2 * NB; ++b) vals[b] = 0.f;
#pragma unroll
    for (int b = 0; b < NB; ++b)
#pragma unroll
      for (int i = 0; i < 8; ++i) v[b][i] = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) st.wq[i] = 0.f;
  }
  float tot[2 * NB];
  exchange<2 * NB>(vals, tot, s, 0, p.cs);
  float lg[NB];
  float mx = -1e30f;
#pragma unroll
  for (int b = 0; b < NB; ++b) { lg[b] = tot[NB + b] * rstd(tot[b], 1e-6f); mx = fmaxf(mx, lg[b]); }
  float es = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) st.mix[i] = 0.f;
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    float e = __expf(lg[b] - mx);
    es += e;
#pragma unroll
    for (int i = 0; i < 8; ++i) st.mix[i] = fmaf(e, v[b][i], st.mix[i]);
  }
  st.mx = mx; st.es = es;
}

// post(): add the delta to the prefix row, stats exchange, normed output words.
__device__ __forceinline__ uint4 attn_post(const Prm& p, int row, int col, bool act, const float* cur,
                                           const St& st, Xs& s) {
  float nw[8], g[8], old[8], nv[8];
  uint4 nwords = make_uint4(0, 0, 0, 0);
  float vals[4] = {0.f, 0.f, 0.f, 0.f};
  if (act) {
    unpack(*reinterpret_cast<const uint4*>(p.outnorm + col / 2), g);
    unpack(*reinterpret_cast<const uint4*>(p.prefix + (long long)row * WORDS + col / 2), old);
#pragma unroll
    for (int i = 0; i < 8; ++i) nw[i] = old[i] + cur[i];
    nwords = pack8(nw);
    unpack(nwords, nv);
    vals[0] = dot8(nv, nv); vals[1] = dot8(nv, st.wq); vals[2] = dot8(st.mix, nv); vals[3] = dot8(st.mix, st.mix);
  }
  float tot[4];
  exchange<4>(vals, tot, s, 1, p.cs);
  if (!act) return make_uint4(0, 0, 0, 0);
  *reinterpret_cast<uint4*>(p.prefix + (long long)row * WORDS + col / 2) = nwords;
  float ss = tot[0], logit = tot[1] * rstd(ss, 1e-6f);
  float top = fmaxf(st.mx, logit), alpha = __expf(st.mx - top), ep = __expf(logit - top);
  float den = fmaf(alpha, st.es, ep);
  float quad = fmaf(alpha * alpha, tot[3], fmaf(2.f * alpha * ep, tot[2], ep * ep * ss));
  float scale = rsqrtf(fmaf(quad, 1.f / 7168.f, 1e-6f * den * den));
  float res[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) res[i] = fmaf(alpha * scale, st.mix[i], ep * scale * nv[i]) * g[i];
  return posz4(pack8(res));
}

__device__ __forceinline__ long long recv_ix(const Prm& p, int par, int src, int lrow) {
  return ((long long)(par * 4 + src) * p.nmax + lrow) * WORDS;
}
__device__ __forceinline__ long long gather_ix(const Prm& p, int par, int row) {
  return (long long)2 * 4 * p.nmax * WORDS + ((long long)par * p.rmax + row) * WORDS;
}
// Lamport poll of 4 words at once: every pass issues the loads of all still-armed
// words back to back (one round trip per pass, not one per word).
__device__ __forceinline__ void poll4(const uint32_t* const* q, uint4* got) {
  bool ok[4] = {false, false, false, false};
  bool all;
  do {
    all = true;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (!ok[i]) got[i] = ld_vol(q[i]);
#pragma unroll
    for (int i = 0; i < 4; ++i) { ok[i] = ok[i] || !armed(got[i]); all = all && ok[i]; }
  } while (!all);
}
// Sum the 4 TP slots of shard row lrow (these 8 columns) in rank order, re-arm them.
__device__ __forceinline__ void owner_sum(const Prm& p, int lrow, int col, float* cur) {
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const uint4 s4 = make_uint4(SENT, SENT, SENT, SENT);
  uint4 got[4];
  const uint32_t* q[4];
#pragma unroll
  for (int src = 0; src < 4; ++src) q[src] = p.box + recv_ix(p, p.par, src, lrow) + col / 2;
  poll4(q, got);
#pragma unroll
  for (int src = 0; src < 4; ++src) {
    st_vol(p.box + recv_ix(p, p.par, src, lrow) + col / 2, s4);
    float f[8];
    unpack(got[src], f);
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] += f[i];
  }
  unpack(posz4(pack8(acc)), cur);  // one BF16 rounding, no -0 (all-reduce boundary)
}
#ifndef STAMPS
#define STAMPS 1
#endif
__device__ __forceinline__ unsigned long long tnow() {
  if (!STAMPS) return 0;
  __syncthreads(); return gtime();
}
// per (layer, kernel) slots: [min start, min wait-done, max wait-done, max end,
// sum(pre-wait part), sum(wait-done -> exchange done), sum(exchange done -> end), CTAs]
__device__ __forceinline__ void tflush(const Prm& p, int k, unsigned long long a, unsigned long long b,
                                       unsigned long long c, unsigned long long d) {
  unsigned long long e = tnow();
  if (threadIdx.x == 0) {
    unsigned long long* s = p.ts + (p.layer * 6 + k) * 8;
    atomicMin(s + 0, a); atomicMin(s + 1, c); atomicMax(s + 2, c); atomicMax(s + 3, e);
    if (!(p.co && (k == 1 || k == 3))) { atomicAdd(s + 4, b - a); atomicAdd(s + 5, d - c); }
    atomicAdd(s + 6, e - d); atomicAdd(s + 7, 1ull);
  }
}
__device__ __forceinline__ void grid_flag(const Prm& p, int slot, int G) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    unsigned int* c = p.ctr + p.layer * 4 + slot;
    unsigned int old = atomicAdd(c, 1u);
    unsigned int target = (old / G + 1) * G;
    while (*(volatile unsigned int*)c < target) {}
    __threadfence();
  }
  __syncthreads();
}

// ---- today's exchange kernels: grid (7, n), 7-CTA cluster per shard row, 128 threads
template <int KIND>
__global__ void __launch_bounds__(128, 4) today_k(Prm p) {
  __shared__ Xs s;
  const unsigned long long ta = tnow();
  const int part = blockIdx.x, token = blockIdx.y, n = p.n;
  const int col = (part * 128 + threadIdx.x) * 8;
  xs_init(s, 7, 2 * NB, 4);
  cl_arrive_relaxed();
  asm volatile("griddepcontrol.launch_dependents;");
  St st;
  cl_wait();
  attn_pre(p, token, col, true, st, s);
  const unsigned long long tb = tnow();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const unsigned long long tc = tnow();
  unsigned long long td;
  float cur[8];
  if (KIND == 4) {  // RED: push my partial rows o n + token to owner o, then the owner sum
#pragma unroll
    for (int o = 0; o < 4; ++o) {
      uint4 v = posz4(*reinterpret_cast<const uint4*>(p.partial + (long long)(o * n + token) * WORDS + col / 2));
      st_vol(reinterpret_cast<uint32_t*>(p.peer[o]) + recv_ix(p, p.par, p.rank, token) + col / 2, v);
    }
    owner_sum(p, token, col, cur);
    td = tnow();
    uint4 out = attn_post(p, token, col, true, cur, st, s);
    *reinterpret_cast<uint4*>(p.act + (long long)token * WORDS + col / 2) = out;
    const uint4 s4 = make_uint4(SENT, SENT, SENT, SENT);
#pragma unroll
    for (int o = 0; o < 4; ++o) st_vol(p.box + gather_ix(p, p.par ^ 1, o * n + token) + col / 2, s4);
  } else {  // GATH: delta = MLP row, multicast the normed row, poll every rank's row
    unpack(*reinterpret_cast<const uint4*>(p.mlp + (long long)token * WORDS + col / 2), cur);
    uint4 out = attn_post(p, token, col, true, cur, st, s);
    st_mc(p.mc + (unsigned long long)(gather_ix(p, p.par, p.rank * n + token) + col / 2) * 4, out);
    td = tnow();
    uint32_t acc = 0;
    const uint32_t* q[4];
    uint4 got[4];
#pragma unroll
    for (int o = 0; o < 4; ++o) q[o] = p.box + gather_ix(p, p.par, o * n + token) + col / 2;
    poll4(q, got);
#pragma unroll
    for (int o = 0; o < 4; ++o) acc ^= got[o].x;
    if (acc == 0x12345678u) p.sink[0] = acc;
  }
  tflush(p, KIND, ta, tb, tc, td);
}

// ---- GEMM stand-ins (one CTA per SM via dynamic smem), exchange fused when do_x
template <int KIND>
__global__ void __launch_bounds__(512, 1) gemm_k(Prm p) {
  __shared__ Xs s;
  extern __shared__ uint4 ring[];
  const unsigned long long ta = tnow();
  const int G = gridDim.x, g = blockIdx.x, cs = p.cs;
  const int ncl = G / cs, cl = g / cs, crank = g % cs;
  const int R = p.R, n = p.n, par = p.par;
  uint32_t acc = 0;
  long long per = (p.w16 + G - 1) / G;
  long long wb = (long long)g * per, we = min(p.w16, wb + per);
  if (wb > we) wb = we;
  long long wm = min(we, wb + p.pre16);
  xs_init(s, cs, 2 * NB, 4);
  if (cs > 1) cl_arrive_relaxed();
  if (p.do_w) acc ^= stream_w(p.w, wb, wm);
  // row-owner cluster cl owns shard row cl (one row per cluster: n <= G / cs)
  const bool own = p.do_x && cl < n && (KIND == 1 || KIND == 2);
  const int gpc = 896 / cs;  // 8-column groups per CTA
  const bool act = threadIdx.x < gpc;
  const int col = (crank * gpc + (act ? threadIdx.x : 0)) * 8;
  St st;
  if (cs > 1) cl_wait();
  if (own) attn_pre(p, cl, col, act, st, s);
  unsigned long long cw = 0;
  if ((KIND == 1 || KIND == 3) && p.co) {
    __shared__ __align__(8) uint64_t cobar[12];
    __shared__ volatile int codone;
    __shared__ unsigned long long cot;
    __shared__ int cokb;
    if (threadIdx.x == 32) {
      for (int i = 0; i < 12; ++i) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(cobar + i)) : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      codone = 0;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      asm volatile("griddepcontrol.wait;" ::: "memory");
      if (threadIdx.x == 0) { cot = gtime(); cokb = codone * (CO_CHUNK / 1024); }
    } else if (p.co == 1) {
      if (threadIdx.x == 32) co_stream(p, reinterpret_cast<char*>(ring), cobar, p.co_nst, &codone);
    } else if (p.co == 2) {
      float x = threadIdx.x * 1e-3f;
      const unsigned long long t0 = gtime();
      while (gtime() < t0 + (unsigned long long)p.co_ns) {
#pragma unroll
        for (int u = 0; u < 64; ++u) x = fmaf(x, 1.0001f, 0.5f);
      }
      if (x == 1234.5f) p.sink[0] = 1;
    }
    __syncthreads();
    cw = cot;
    if (threadIdx.x == 0) {
      unsigned long long* sl = p.ts + (p.layer * 6 + KIND) * 8;
      atomicAdd(sl + 4, (unsigned long long)cokb);
      atomicAdd(sl + 5, (unsigned long long)(cot > ta + 300 ? 1 : 0));
    }
  }
  const unsigned long long tb = tnow();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const unsigned long long tc = tnow();
  unsigned long long td = tc;
  if (p.early) asm volatile("griddepcontrol.launch_dependents;");
  if (KIND == 0) {
    if (p.do_x) {  // re-arm gather[par ^ 1] (consumed by the previous layer's D)
      uint32_t* gb = p.box + gather_ix(p, par ^ 1, 0);
      long long tot = (long long)R * (WORDS / 4);
      uint4 s4 = make_uint4(SENT, SENT, SENT, SENT);
      for (long long i = (long long)g * blockDim.x + threadIdx.x; i < tot; i += (long long)G * blockDim.x)
        reinterpret_cast<uint4*>(gb)[i] = s4;
    }
    const int lane = threadIdx.x & 15;
    const long long grp = ((long long)g * blockDim.x + threadIdx.x) >> 4;
    const long long ngrp = ((long long)G * blockDim.x) >> 4;
    for (int c = 0; c < p.chunks; ++c) {  // output tiles finish in `chunks` waves
      long long cb = wm + (we - wm) * c / p.chunks, ce = wm + (we - wm) * (c + 1) / p.chunks;
      if (p.do_w) acc ^= stream_w(p.w, cb, ce);
      if (!p.do_x) continue;
      long long ub = (long long)R * SEGS * c / p.chunks, ue = (long long)R * SEGS * (c + 1) / p.chunks;
      for (long long u = ub + grp; u < ue; u += ngrp) {
        int seg = (int)(u / R), row = (int)(u % R);
        int owner = row / n, lrow = row % n;
        long long cw = seg * 64 + lane * 4;
        uint4 v = posz4(*reinterpret_cast<const uint4*>(p.partial + (long long)row * WORDS + cw));
        st_vol(reinterpret_cast<uint32_t*>(p.peer[owner]) + recv_ix(p, par, p.rank, lrow) + cw, v);
      }
    }
  } else if (KIND == 1) {
    if (p.do_x) {
      if (cl < n) {
        float cur[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (act) owner_sum(p, cl, col, cur);
        uint4 out = attn_post(p, cl, col, act, cur, st, s);
        if (act) *reinterpret_cast<uint4*>(p.act + (long long)cl * WORDS + col / 2) = out;
      }
      grid_flag(p, 0, G);  // the latent_down mainloop needs every owned row
      td = tnow();
    }
    if (p.do_w) acc ^= stream_w(p.w, wm, we);
  } else if (KIND == 2) {
    if (p.do_w) acc ^= stream_w(p.w, wm, we);
    td = tnow();
    if (p.do_x) {
      // this CTA's output tiles of the MLP rows (n x 7168), then the row-complete flag
      long long tot = (long long)n * (WORDS / 4);
      for (long long i = (long long)g * blockDim.x + threadIdx.x; i < tot; i += (long long)G * blockDim.x) {
        uint4 v = *reinterpret_cast<const uint4*>(p.partial + i * 4);
        v.x ^= (acc & 0x10000u);
        reinterpret_cast<uint4*>(p.mlp)[i] = v;
      }
      grid_flag(p, 1, G);
      if (cl < n) {
        float cur[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (act) unpack(*reinterpret_cast<const uint4*>(p.mlp + (long long)cl * WORDS + col / 2), cur);
        uint4 out = attn_post(p, cl, col, act, cur, st, s);
        if (act) st_mc(p.mc + (unsigned long long)(gather_ix(p, par, p.rank * n + cl) + col / 2) * 4, out);
      }
    }
  } else {
    if (p.do_x) {  // poll the gathered rows, then a local grid flag
      long long tot = (long long)R * (WORDS / 4);
      const uint32_t* gb = p.box + gather_ix(p, par, 0);
      const long long st = (long long)G * blockDim.x;
      for (long long i = (long long)g * blockDim.x + threadIdx.x; i < tot; i += 4 * st) {
        const uint32_t* q[4];
        uint4 got[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) q[u] = gb + min(i + u * st, tot - 1) * 4;
        poll4(q, got);
#pragma unroll
        for (int u = 0; u < 4; ++u) acc ^= got[u].x;
      }
      grid_flag(p, 2, G);
      td = tnow();
    }
    if (p.do_w) acc ^= stream_w(p.w, wm, we);
  }
  if (acc == 0x12345678u) { p.sink[g] = acc; ring[threadIdx.x] = make_uint4(acc, 0, 0, 0); }
  tflush(p, KIND, ta, tb, cw ? cw : tc, td);
}

void launch(int64_t kind, int64_t do_w, int64_t do_x, int64_t rank, int64_t R, int64_t parity, int64_t layer,
            int64_t w, int64_t wbytes, int64_t pre_bytes, std::vector<int64_t> peers, int64_t box,
            int64_t mc, std::vector<int64_t> bufs, int64_t grid, int64_t chunks, int64_t cs, int64_t smem, int64_t early) {
  Prm p;
  p.early = early; p.do_w = do_w; p.do_x = do_x; p.rank = rank; p.R = R; p.n = R / 4; p.par = parity; p.layer = layer;
  p.nmax = 96; p.rmax = 384; p.chunks = chunks; p.cs = kind >= 4 ? 7 : cs; p.G = grid;
  p.w = reinterpret_cast<const uint4*>(w); p.w16 = wbytes / 16; p.pre16 = pre_bytes / 16;
  for (int i = 0; i < 4; ++i) p.peer[i] = peers[i];
  p.box = reinterpret_cast<uint32_t*>(box); p.mc = mc;
  p.blocks = reinterpret_cast<const uint32_t*>(bufs[0]); p.prefix = reinterpret_cast<uint32_t*>(bufs[1]);
  p.normw = reinterpret_cast<const uint32_t*>(bufs[2]); p.query = reinterpret_cast<const uint32_t*>(bufs[3]);
  p.outnorm = reinterpret_cast<const uint32_t*>(bufs[4]); p.partial = reinterpret_cast<const uint32_t*>(bufs[5]);
  p.act = reinterpret_cast<uint32_t*>(bufs[6]); p.mlp = reinterpret_cast<uint32_t*>(bufs[7]);
  p.ctr = reinterpret_cast<unsigned int*>(bufs[8]);
  p.ts = reinterpret_cast<unsigned long long*>(bufs[9]);
  p.sink = reinterpret_cast<uint32_t*>(bufs[10]);
  const bool succ = kind == 1 || kind == 3;
  p.co = succ && g_co != 3 ? g_co : 0; p.co_ns = g_co_ns; p.co_bytes = g_co_bytes;
  p.co_win = g_co_win; p.co_buf = g_co_buf; p.co_bufb = g_co_bufb; p.co_nst = (int)(smem / CO_CHUNK < 12 ? smem / CO_CHUNK : 12);
  cudaLaunchConfig_t cfg = {};
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = (succ && g_co == 3) ? 0 : 1;
  attr[1].id = cudaLaunchAttributeClusterDimension;
  cfg.attrs = attr;
  void* fn;
  if (kind >= 4) {
    cfg.gridDim = dim3(7, R / 4); cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = 0;
    attr[1].val.clusterDim.x = 7; attr[1].val.clusterDim.y = 1; attr[1].val.clusterDim.z = 1;
    cfg.numAttrs = 2;
    fn = kind == 4 ? (void*)today_k<4> : (void*)today_k<5>;
    static bool once = false;
    if (!once) {
      C10_CUDA_CHECK(cudaFuncSetAttribute((void*)today_k<4>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      C10_CUDA_CHECK(cudaFuncSetAttribute((void*)today_k<5>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      once = true;
    }
  } else {
    cfg.gridDim = dim3(grid); cfg.blockDim = dim3(p.co ? 128 : 512); cfg.dynamicSmemBytes = smem;
    attr[1].val.clusterDim.x = cs; attr[1].val.clusterDim.y = 1; attr[1].val.clusterDim.z = 1;
    cfg.numAttrs = cs > 1 ? 2 : 1;
    void* fns[4] = {(void*)gemm_k<0>, (void*)gemm_k<1>, (void*)gemm_k<2>, (void*)gemm_k<3>};
    fn = fns[kind];
    static int64_t set_smem[4] = {-1, -1, -1, -1};
    if (set_smem[kind] != smem) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
      set_smem[kind] = smem;
    }
  }
  void* args[1] = {&p};
  C10_CUDA_CHECK(cudaLaunchKernelExC(&cfg, fn, args));
}

void set_co(int64_t co, int64_t ns, int64_t bytes, int64_t win, int64_t buf, int64_t bufb) {
  g_co = (int)co; g_co_ns = (int)ns; g_co_bytes = bytes; g_co_win = win;
  g_co_buf = reinterpret_cast<const char*>(buf); g_co_bufb = bufb;
}

int64_t max_clusters(int64_t cs, int64_t smem) {
  void* fn = (void*)gemm_k<1>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(cs * 512); cfg.blockDim = dim3(512); cfg.dynamicSmemBytes = smem;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = cs; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr; cfg.numAttrs = 1;
  int n = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, fn, &cfg));
  return n;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
void launch(int64_t kind, int64_t do_w, int64_t do_x, int64_t rank, int64_t R, int64_t parity, int64_t layer,
            int64_t w, int64_t wbytes, int64_t pre_bytes, std::vector<int64_t> peers, int64_t box,
            int64_t mc, std::vector<int64_t> bufs, int64_t grid, int64_t chunks, int64_t cs, int64_t smem, int64_t early);
int64_t max_clusters(int64_t cs, int64_t smem);
void set_co(int64_t co, int64_t ns, int64_t bytes, int64_t win, int64_t buf, int64_t bufb);
"""


def _module(nb, stamps=1):
    from torch.utils.cpp_extension import load_inline

    src = f"#define NB {nb}\n#define STAMPS {stamps}\n" + _CUDA_SRC
    tag = hashlib.sha256((src + _CPP_SRC).encode()).hexdigest()[:16]
    root = os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
    build = os.path.join(root, "k3_sp_boundary_cal", tag)
    os.makedirs(build, exist_ok=True)
    return load_inline(
        name=f"k3_sp_boundary_cal_{tag}",
        cpp_sources=_CPP_SRC,
        cuda_sources=src,
        functions=["launch", "max_clusters", "set_co"],
        extra_cuda_cflags=["-O3", "-gencode", "arch=compute_100f,code=sm_100f",
                           "-Xptxas", "-v"],
        build_directory=build,
    )


def _bf16_words(shape, dev, scale=1.0):
    t = (torch.randn(*shape, device=dev) * scale).to(torch.bfloat16)
    return t


def _real_phases(rst, a, n, layers):
    """Per real kernel (RED, GATH), us averaged over layers 1..: kernel span (last CTA
    mark6 - first mark0), producer stand-in end -> first/last wait release (mark2),
    and per-CTA mean phase deltas between consecutive marks (0..6)."""
    x = rst[:, :, : n * 7].cpu().double()
    out = {}
    for k, name, prod in ((0, "RED", 0), (1, "GATH", 2)):
        y = x[1:, k]  # skip layer 0 (no predecessor overlap)
        m0, m6 = y[:, :, 0], y[:, :, 6]
        span = ((m6.max(1).values - m0.min(1).values) / 1e3).mean().item()
        pend = a[1:, prod, 3]
        w = y[:, :, 2]
        rel_first = ((w.min(1).values - pend) / 1e3).mean().item()
        rel_last = ((w.max(1).values - pend) / 1e3).mean().item()
        start_rel = ((m0.min(1).values - pend) / 1e3).mean().item()
        end_rel = ((m6.max(1).values - pend) / 1e3).mean().item()
        ph = []
        live = [j for j in range(7) if not (y[:, :, j] == 0).all()]
        live.sort(key=lambda j: y[:, :, j].mean().item())
        for prev, j in zip(live, live[1:]):
            d = (y[:, :, j] - y[:, :, prev]) / 1e3
            ph.append(f"{prev}->{j} {d.mean().item():.2f}/{d.max(1).values.mean().item():.2f}")
        out[name] = (f"span {span:.2f} start {start_rel:+.2f} wait1st {rel_first:+.2f} "
                     f"waitlast {rel_last:+.2f} end {end_rel:+.2f} (vs producer end) | "
                     + " ".join(ph))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="128,256,384")
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--pre-kb", default="0,96,192")
    ap.add_argument("--chunks", default="2")
    ap.add_argument("--rs", default="2")
    ap.add_argument("--nb", type=int, default=8)
    ap.add_argument("--stamps", type=int, default=1)
    ap.add_argument("--mx", type=int, default=0)
    ap.add_argument("--poll-sleep", type=int, default=0)
    ap.add_argument("--counted", type=int, default=-1)
    ap.add_argument("--smem-kb", type=int, default=200)
    ap.add_argument("--modes", default="w,wl,today,real,fz")
    # co-runner beside RED / GATH (real mode): park | none | fma:NS | l2:NS | hbm:NS
    # (NS = pacing in ns per 16 KB chunk per CTA, 0 = full rate; fma: spin NS)
    ap.add_argument("--co", default="")
    ap.add_argument("--co-kb", type=int, default=1024)
    ap.add_argument("--carve", type=int, default=100)
    args = ap.parse_args()
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4
    dev = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(dev)
    group = dist.group.WORLD
    import torch.distributed._symmetric_memory as symm_mem

    try:
        symm_mem.enable_symm_mem_for_group(group.group_name)
    except (AttributeError, ImportError):
        pass
    if rank == 0:
        _module(args.nb, args.stamps)
    dist.barrier()
    mod = _module(args.nb, args.stamps)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    words = 2 * 4 * NMAX * WORDS + 2 * RMAX * WORDS
    box = symm_mem.empty((words,), dtype=torch.int32, device=dev)
    box.fill_(-1)
    handle = symm_mem.rendezvous(box, group.group_name)
    peers = [int(x) for x in handle.buffer_ptrs]
    mc = int(handle.multicast_ptr or 0)
    assert mc, "no multicast object"
    ncopy = 2
    co_buf = torch.empty(768 << 20, dtype=torch.uint8, device=dev).random_(0, 255)
    wts = {k: [torch.empty(v, dtype=torch.uint8, device=dev).random_(0, 255)
               for _ in range(ncopy)] for k, v in W_BYTES.items()}
    blocks = _bf16_words((NMAX, NBMAX, H), dev)
    prefix = _bf16_words((NMAX, H), dev)
    normw = _bf16_words((H,), dev, 0.1) + 1
    query = _bf16_words((H,), dev, 0.1)
    outnorm = _bf16_words((H,), dev, 0.1) + 1
    partial = _bf16_words((RMAX, H), dev)
    act = torch.empty(NMAX, H, dtype=torch.bfloat16, device=dev)
    mlp = _bf16_words((NMAX, H), dev)
    ctr = torch.zeros(args.layers * 4, dtype=torch.int32, device=dev)
    ts = torch.zeros(args.layers * 6 * 8, dtype=torch.int64, device=dev)
    ts0 = torch.zeros(args.layers, 6, 8, dtype=torch.int64, device=dev)
    ts0[:, :, 0:2] = -1  # 0xFF..: atomicMin slots
    ts0 = ts0.view(-1)
    sink = torch.zeros(sms, dtype=torch.int32, device=dev)
    bufs = [x.data_ptr() for x in (blocks, prefix, normw, query, outnorm, partial, act, mlp,
                                   ctr, ts, sink)]
    prefix0 = prefix.clone()
    real_blocks = blocks[:, : args.nb].contiguous()
    real_cache = {}
    # real-kernel phase stamps: [layer, RED/GATH, CTA (token * 7 + part), 8 slots]
    rst = torch.zeros(args.layers, 2, NMAX * 7, 8, dtype=torch.int64, device=dev)

    def real_bound(R, unicast=False, link=False, bulk=False):
        """Today's SpBoundary (SpReduceAttnRes + SpAttnResGather), same nb / shapes."""
        key = (R, unicast, link, bulk)
        if key not in real_cache:
            from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.attn_res import (
                sp_fused,
            )
            from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import (
                sp_comm,
            )

            sp_fused.POLL_SLEEP_NS = args.poll_sleep
            if "comm" not in real_cache:
                real_cache["comm"] = sp_comm.SpComm(group, dev, world, rank, rows_cap=RMAX)
                if args.counted >= 0:  # A/B of the counted MXFP8 all-gather
                    real_cache["comm"].counted &= bool(args.counted)
            w3 = (normw, query, outnorm)
            extra = {}  # only the options this tree's SpBoundary has (dev branches)
            if args.stamps and "stamps" in inspect.signature(sp_fused.SpBoundary).parameters:
                extra["stamps"] = True
            if unicast:
                extra["unicast"] = True
            if bulk:
                extra["bulk"] = True
            real_cache[key] = sp_fused.SpBoundary(
                real_cache["comm"], (w3, (1e-6, 1e-6)), (w3, (1e-6, 1e-6)), args.nb, -1,
                NMAX, {R: True}, gather_mx=bool(args.mx), **extra,
            )  # fmt: skip
            if args.carve >= 0:
                from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
                    prepare_shared_carveout,
                )

                b_ = real_cache[key]
                for ck in list(b_.codes):
                    b_.codes[ck] = prepare_shared_carveout(b_.codes[ck], dev.index, args.carve)
            if link:  # the reduce computes the (same-layer stand-in) gather's pre()
                real_cache[key].link(real_cache[key], 0)
        return real_cache[key]
    torch.cuda.synchronize()
    dist.barrier()
    smem = args.smem_kb * 1024
    res, stats = {}, {}
    for R in [int(r) for r in args.rows.split(",")]:
        cfgs = []
        for pre in [int(x) for x in args.pre_kb.split(",")]:
            for mode in args.modes.split(","):
                if mode in ("today", "wl", "real", "realu", "realk", "realb", "realkb") and pre != 0:
                    continue
                for ch in ([int(x) for x in args.chunks.split(",")] if mode == "fz" else [1]):
                    for rs in ([int(x) for x in args.rs.split(",")] if mode == "fz" else [2]):
                        cos = ([c for c in args.co.split(",") if c] or [None]) if mode == "real" else [None]
                        for co in cos:
                            cfgs.append((pre, mode, ch, rs, co))
        for pre, mode, ch, rs, co in cfgs:
            kind_, _, arg_ = (co or "off").partition(":")
            ns_ = int(arg_ or 0)
            cmode = {"off": 0, "park": 0, "none": 3, "fma": 2, "l2": 1, "hbm": 1}[kind_]
            mod.set_co(cmode, ns_, args.co_kb * 1024, (32 << 20) if kind_ == "l2" else 0,
                       co_buf.data_ptr(), co_buf.numel())
            grid = rs * min(sms // rs, mod.max_clusters(rs, smem)) if rs > 1 else sms
            assert R // 4 <= grid // rs, "one shard row per owner cluster"
            if rank == 0 and R == int(args.rows.split(",")[0]) and mode == "fz":
                print(f"sms {sms} rs {rs} grid {grid}", flush=True)
            if mode == "today":
                seq = [(0, 1, 0), (4, 0, 1), (1, 1, 0), (2, 1, 0), (5, 0, 1), (3, 1, 0)]
            else:
                seq = [(k, 1, int(mode == "fz")) for k in range(4)]
            early = int(mode in ("fz", "w"))
            n = R // 4
            bound = (real_bound(R, mode == "realu", mode in ("realk", "realkb"),
                                mode in ("realb", "realkb"))
                     if mode in ("real", "realu", "realk", "realb", "realkb") else None)

            def chain():
                ts.copy_(ts0)
                for i in range(args.layers):
                    par = i & 1
                    for kind, dw, dx in seq:
                        k = "ABCD"[kind] if kind < 4 else "A"
                        w = wts[k][i % ncopy]
                        mod.launch(kind, dw, dx, rank, R, par, i, w.data_ptr(), W_BYTES[k],
                                   pre * 1024, peers, box.data_ptr(), mc, bufs, grid, ch, rs,
                                   smem, early)
                        if bound is not None and kind == 0:  # today's SpReduceAttnRes
                            bound.stamp_ptr = rst[i, 0].data_ptr()
                            bound.reduce(prefix[:n], real_blocks[:n], act[:n], par,
                                         partial[:R])
                        if bound is not None and kind == 2:  # today's SpAttnResGather
                            bound.stamp_ptr = rst[i, 1].data_ptr()
                            bound.gather(mlp[:n], prefix[:n], real_blocks[:n], par)

            box.fill_(-1)
            prefix.copy_(prefix0)
            torch.cuda.synchronize()
            dist.barrier()
            chain()
            torch.cuda.synchronize()
            dist.barrier()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                chain()
            g.replay()
            torch.cuda.synchronize()
            dist.barrier()
            times, kst = [], []
            kinds = [s[0] for s in seq]
            for _ in range(args.replays):
                prefix.copy_(prefix0)
                dist.barrier()
                torch.cuda.synchronize()
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                g.replay()
                e.record()
                e.synchronize()
                times.append(s.elapsed_time(e) * 1e3 / args.layers)
                a = ts.view(args.layers, 6, 8).cpu().double()
                ends = a[:, :, 3]
                # per kernel: span, eff (end - last wait-done), tail (end - previous kernel end),
                # CTA-mean pre-wait / exchange / rest parts (us)
                st = {}
                order = [(i, k) for i in range(args.layers) for k in kinds]
                for k in kinds:
                    span = (a[:, k, 3] - a[:, k, 0]).mean() / 1e3
                    eff = (a[:, k, 3] - a[:, k, 2]).mean() / 1e3
                    tails = [ends[i, k] - ends[order[j - 1][0], order[j - 1][1]]
                             for j, (i, kk) in enumerate(order) if kk == k and j > 0]
                    tail = sum(tails) / len(tails) / 1e3
                    cnt = a[:, k, 7].clamp(min=1)
                    parts = [(a[:, k, c] / cnt).mean() / 1e3 for c in (4, 5, 6)]
                    st[k] = [float(span), float(eff), float(tail)] + [float(x) for x in parts]
                # exchange tail to the successor's release: B/D first wait-done - A/C last end;
                # co-runner KB streamed before its release, CTAs landed early (per layer)
                co_st = [float((a[:, 1, 1] - a[:, 0, 3]).mean() / 1e3),
                         float((a[:, 3, 1] - a[:, 2, 3]).mean() / 1e3),
                         float(a[:, 1, 4].mean()), float(a[:, 3, 4].mean()),
                         float(a[:, 1, 5].mean()), float(a[:, 3, 5].mean())]
                st["co"] = co_st if mode == "real" and len(kinds) == 4 else [0.0] * 6
                kst.append(st)
                if bound is not None and args.stamps and getattr(bound, "stamps", False):
                    kst[-1]["rs"] = _real_phases(rst, a, n, args.layers)
            order_t = sorted(range(len(times)), key=lambda j: times[j])
            mid = order_t[len(order_t) // 2]
            flat = [times[mid]] + [v for k in kinds for v in kst[mid][k]] + kst[mid]["co"]
            t = torch.tensor(flat, device=dev, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            res[(R, pre, mode, ch, rs, co)] = t[0].item()
            if bound is not None and 'rs' in kst[mid] and rank == 0:
                print(f"R{R} REAL STAMPS (rank0, median replay): {kst[mid]['rs']}", flush=True)
            vals = t[1:].tolist()
            stats[(R, pre, mode, ch, rs, co)] = {k: vals[6 * j:6 * j + 6] for j, k in enumerate(kinds)}
            stats[(R, pre, mode, ch, rs, co)]["co"] = vals[6 * len(kinds):]
            del g
        if rank == 0:
            for pre, mode, ch, rs, co in cfgs:
                key = (R, pre, mode, ch, rs, co)
                t = res[key]
                base = "wl" if mode in ("today", "real", "realu", "realk", "realb", "realkb") else "w"
                w = res.get((R, pre, base, 1, 2, None))
                ex = f" exposed {t - w:.2f}" if mode in ("today", "fz", "real", "realu", "realk", "realb", "realkb") and w else ""
                print(f"R{R} r{R // 16} pre{pre}K {mode} c{ch} rs{rs}: {t:.2f} us/layer{ex}",
                      flush=True)
                if co is not None:
                    c = stats[key]["co"]
                    print(f"    CO {co}: RED rel {c[0]:6.2f} GATH rel {c[1]:6.2f} us | co KB in window"
                          f" RED {c[2]:8.0f} GATH {c[3]:8.0f} | early CTAs RED {c[4]:5.0f} GATH {c[5]:5.0f}",
                          flush=True)
                for k, v in stats[key].items():
                    if k == "co":
                        continue
                    print(f"    {KNAMES[k]:4s} span {v[0]:6.2f} eff {v[1]:6.2f} tail {v[2]:6.2f}"
                          f" | CTA pre {v[3]:5.2f} xch {v[4]:5.2f} rest {v[5]:5.2f}", flush=True)
                if mode == "today":
                    st = stats[key]
                    print(f"    RED+GATH eff {st[4][1] + st[5][1]:.2f} tail {st[4][2] + st[5][2]:.2f}"
                          f" span {st[4][0] + st[5][0]:.2f}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
