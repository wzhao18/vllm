# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D-MOEFRONT data-movement skeleton of the decode MoE front (router|latent_down,
64.2 MB BF16 weight [4480 x 7168]) behind a PDL producer, one GPU, CUDA graphs.

Per layer: producer (7 x n CTAs x 128 thr, ~2 KB smem, triggers dependents at
entry, busy T us, writes x = h [n, 7168] bf16) -> front skeleton (ncta CTAs,
1 CTA/SM). Each CTA owns a contiguous W slice (pre-tiled) of Bc bytes:
  pre-wait : RES bytes TMA'd into smem (resident), the first NS ring stages, and
             (warm=1) the rest TMA'd through a scratch slot so it is L2-resident.
  post-wait: x K-slice (n x Kt bf16) LDG -> smem; remainder streamed through an
             NS x S ring by I issuer warps; math proxy (consumer warps read every
             W byte and n/128 x-bytes per W byte from smem, the SS swap-AB MMA
             operand traffic); partial Mt x n fp32 -> smem -> one TMA bulk
             reduce-add into the tile output; per-tile counter, the last CTA of a
             tile converts the tile to bf16 (stage proxy) and re-zeroes it.
Reported: per-layer (graph(prod+front) - graph(prod)) / L, i.e. exposed time
beyond the producer, plus per-phase stamps relative to the producer end.

python skel_moe_front.py --n 32,64,96 --cfg name:ncta:ksplit:res_kb:S_kb:NS:I:warm:x:math:red ...
"""

import argparse
import os

import torch
from torch.utils.cpp_extension import load_inline

KDIM, MDIM = 7168, 4480
WBYTES = KDIM * MDIM * 2

_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void bar_init(uint64_t* b, int c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(b)), "r"(c)); }
__device__ __forceinline__ void bar_expect(uint64_t* b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(b)), "r"(tx) : "memory"); }
__device__ __forceinline__ void bar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(b)) : "memory"); }
__device__ __forceinline__ void bar_wait(uint64_t* b, uint32_t par) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}"
               :: "r"(sa(b)), "r"(par) : "memory"); }
__device__ __forceinline__ void tma(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(b)) : "memory"); }
// copy [off, off+len) of src into dst in <=chunk pieces on barrier b (expect already set)
__device__ __forceinline__ void tma_n(char* dst, const char* src, uint32_t len, uint32_t chunk, uint64_t* b) {
  for (uint32_t o = 0; o < len; o += chunk) tma(dst + o, src + o, min(chunk, len - o), b); }

struct P {
  const char* w; long long bc;            // this layer's W, bytes per CTA
  const __nv_bfloat16* x; const char* xblk; int n, ksplit, kt, mt;
  int res, S, NS, I, warm, xmode, math, red, trig, chunk, C, fw, ptot;
  int* xcnt; int* ep;
  int tm, smem, tsb;                        // tsb: TMEM-fill staging block bytes (multiple of 16 KiB)                             // tm: W bytes held in TMEM (A operand of the TS form); smem: dynamic bytes                      // fw: producer x-ready counter (per layer) + per-CTA epoch
  float* out; int* cnt; __nv_bfloat16* outb;
  unsigned long long* st;                  // [ncta][8] stamps or null
};

// smem: [bars 1024][res][ring NS*S][x (>= S if warm)]
__global__ void __launch_bounds__(256, 1) front_skel(P p) {
  extern __shared__ __align__(1024) char sm[];
  uint64_t* bres = (uint64_t*)sm; uint64_t* bwarm = bres + 1;
  uint64_t* full = bres + 2; uint64_t* empty = full + 32;
  uint64_t* bx = empty + 32; volatile int* flag = (volatile int*)(bx + 1);
  char* res = sm + 1024; char* ring = res + p.res; char* xs = ring + (long long)p.NS * p.S;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, cta = blockIdx.x;
  const long long bc = p.bc;
  const char* w0 = p.w + (long long)cta * bc;
  const char* w = w0 + p.tm;               // [tm bytes -> TMEM][res -> smem][ring / remainder]
  const long long rem = bc - p.tm - p.res; const int nch = p.S ? (int)((rem + p.S - 1) / p.S) : 0;
  const int ncons = 8 - p.I;  // consumer warps
  unsigned long long t0 = gtime();
  uint64_t* tf = (uint64_t*)(sm + 576); uint64_t* tc = tf + 16; uint32_t* tmaddr = (uint32_t*)(sm + 960);
  const int NSt = min(16, (p.smem - 1024) / p.tsb);
  if (p.tm && warp == 7) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(sa(tmaddr)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    if (lane < NSt) { bar_init(tf + lane, 1); bar_init(tc + lane, 1); }
  }
  if (tid == 0) {
    bar_init(bres, 1); bar_init(bwarm, 1); bar_init(bx, 1);
    for (int s = 0; s < p.NS; ++s) { bar_init(full + s, 1); bar_init(empty + s, ncons); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (p.trig == 1) asm volatile("griddepcontrol.launch_dependents;");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t tbase = *tmaddr;
  if (p.tm) {
    // phase A (pre-wait): W[0, tm) -> TMEM through all of smem as 16 KiB staging blocks, one issuing lane per block
    // (per-issuer bulk ops serialize, hw51); 4 x tcgen05.cp 128x256b per block (SW128 K-major, 32 B K-steps).
    if (warp == 7 && lane < NSt) {
      const int nst = (p.tm + p.tsb - 1) / p.tsb; char* buf = sm + 1024 + lane * p.tsb;
      int u = 0;
      for (int i = lane; i < nst; i += NSt, ++u) {
        if (u) bar_wait(tc + lane, (u - 1) & 1);
        const uint32_t len = (uint32_t)min(p.tsb, p.tm - i * p.tsb);
        bar_expect(tf + lane, len); tma(buf, w0 + (long long)i * p.tsb, len, tf + lane);
        bar_wait(tf + lane, u & 1);
        for (uint32_t j = 0; j < len / 16384; ++j) {
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const uint32_t a = sa(buf) + j * 16384 + 32 * k;
            const uint64_t d = (uint64_t)((a & 0x3FFFF) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 46) | (2ull << 61);
            const uint32_t col = (uint32_t)(((i * p.tsb / 16384 + j) * 4 + k) * 8);
            asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;" :: "r"(tbase + col), "l"(d) : "memory");
          }
        }
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(sa(tc + lane)) : "memory");
      }
      if (u) bar_wait(tc + lane, (u - 1) & 1);
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
  }
  const uint32_t ck = (uint32_t)p.chunk;
  // ---- pre-wait: resident W (ops spread over all 8 warps), first ring stages, L2 warm of the rest ----
  const long long wtot = p.warm ? max(0LL, rem - (long long)p.NS * p.S) : 0;
  if (tid == 0) {
    if (p.res) bar_expect(bres, p.res);
    if (p.warm) { if (wtot) bar_expect(bwarm, (uint32_t)wtot); else bar_arrive(bwarm); }
  }
  __syncthreads();
  if (lane == 0 && p.res) {
    const int nop = (p.res + ck - 1) / ck;
    for (int i = warp; i < nop; i += 8) tma(res + (long long)i * ck, w + (long long)i * ck, min(ck, (uint32_t)p.res - i * ck), bres);
  }
  if (lane == 0 && warp < p.I) {
    for (int c = warp; c < min(nch, p.NS); c += p.I) {  // slot c owned by issuer c % I
      uint32_t len = (uint32_t)min((long long)p.S, rem - (long long)c * p.S);
      bar_expect(full + c, len); tma_n(ring + (long long)c * p.S, w + p.res + (long long)c * p.S, len, ck, full + c);
    }
  }
  if (p.warm) {   // consumer warps' lane 0 warm the remainder past the ring into L2 through the scratch slot
    const long long wb = (long long)p.NS * p.S, tot = wtot;
    const uint32_t sc = min((uint32_t)p.S, ck);
    const int k = (int)((tot + sc - 1) / sc), nw = 8 - p.I;
    if (warp >= p.I && lane == 0)
      for (int i = warp - p.I; i < k; i += nw) tma(xs, w + p.res + wb + (long long)i * sc, (uint32_t)min((long long)sc, tot - (long long)i * sc), bwarm);
  }
  const int L0 = 32 * p.I;               // leader = first consumer thread
  unsigned long long tres = 0;
  if (p.fw) {   // flag wait: only the x consumers wait, on the producer's per-CTA release counter (no grid-end resume)
    if (warp >= p.I) {
      if (tid == L0) {
        const int e = p.ep[cta]; p.ep[cta] = e + 1;
        const int tgt = (e + 1) * p.ptot; int c;
        do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(c) : "l"(p.xcnt) : "memory"); } while (c < tgt);
      }
      asm volatile("bar.sync 3, %0;" :: "r"(32 * (8 - p.I)));
      asm volatile("fence.proxy.async.global;" ::: "memory");   // generic-proxy x writes -> TMA reads
    }
  } else {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  unsigned long long tw = gtime();
  if (p.trig == 2 && tid == 0) asm volatile("griddepcontrol.launch_dependents;");
  // ---- post-wait ----
  float acc = 0.f; float acc2[4] = {0.f, 0.f, 0.f, 0.f};
  unsigned long long tx = 0, tm = 0;
  if (warp < p.I) {                      // issuers: refill the ring
    if (lane == 0) {
      for (int c = p.NS; c < nch; ++c) { if ((c % p.NS) % p.I != warp) continue;  // one issuer per slot: in-order phases
        int s = c % p.NS, u = c / p.NS;
        bar_wait(empty + s, (u - 1) & 1);
        uint32_t len = (uint32_t)min((long long)p.S, rem - (long long)c * p.S);
        bar_expect(full + s, len); tma_n(ring + (long long)s * p.S, w + p.res + (long long)c * p.S, len, ck, full + s);
      }
    }
  } else {                               // consumers: x slice, then W (resident, ring)
    const int ct = tid - 32 * p.I, nct = 32 * ncons;
    if (p.res) { bar_wait(bres, 0); if (tid == L0) tres = gtime(); }
    if (p.warm) bar_wait(bwarm, 0);
    if (p.xmode == 2) {   // producer wrote x K-blocked: this CTA's slice is one contiguous bulk op
      // x slice split over up to 4 consumer warps (one op each: per-issuer ops serialize, hw51)
      const uint32_t xbytes = p.n * p.kt * 2, nw = min(4, 8 - p.I), xp = ((xbytes + nw - 1) / nw + 15) / 16 * 16;
      const char* xsrc = p.xblk + (long long)(cta % p.ksplit) * xbytes;
      if (tid == L0) bar_expect(bx, xbytes);
      asm volatile("bar.sync 1, %0;" :: "r"(nct));
      const uint32_t wi = (uint32_t)(warp - p.I);
      if (lane == 0 && wi < nw && wi * xp < xbytes) tma(xs + wi * xp, xsrc + wi * xp, min(xp, xbytes - wi * xp), bx);
      bar_wait(bx, 0);
    } else if (p.xmode == 1) {
      const int k0 = (cta % p.ksplit) * p.kt, kv = p.kt / 8;
      const uint4* xg = (const uint4*)p.x; uint4* xd = (uint4*)xs;
      const int tot = p.n * kv;
      for (int i0 = ct; i0 < tot; i0 += nct * 8) {
        uint4 v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          int i = i0 + j * nct, r = i / kv, c = i - r * kv;
          v[j] = (i < tot && k0 / 8 + c < KD8) ? __ldcg(xg + (long long)r * KD8 + k0 / 8 + c) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) if (i0 + j * nct < tot) xd[i0 + j * nct] = v[j];
      }
      asm volatile("bar.sync 1, %0;" :: "r"(nct));
    }
    tx = gtime();
    const int xb = p.n * p.kt * 2;       // x bytes in smem
    long long xo = 0; const int rep = (p.mt + 127) / 128;
    auto proxy = [&](const char* buf, uint32_t len) {
      const uint4* b4 = (const uint4*)buf;
      const uint32_t n16 = len / 16;
      for (uint32_t i0 = ct; i0 < n16; i0 += nct * 8) {
        uint4 v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = (i0 + j * nct < n16) ? b4[i0 + j * nct] : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc2[j & 3] += __int_as_float(v[j].x ^ v[j].w);
      }
      if (p.xmode) {   // swap-AB SS: per 4 KiB of A (M128 K16) the MMA reads n*32 B of B, rep times per K
        long long xl = (long long)len * p.n / 128;
        const uint4* x4 = (const uint4*)xs;
        const int x16 = (int)(xl / 16), xm16 = xb / 16, xo16 = (int)((xo / 16) % xm16);
        for (int i0 = ct; i0 < x16; i0 += nct * 8) {
          uint4 v[8];
#pragma unroll
          for (int j = 0; j < 8; ++j) { int q = xo16 + i0 + j * nct; q = q >= xm16 ? q - xm16 : q; v[j] = (i0 + j * nct < x16) ? x4[q] : make_uint4(0, 0, 0, 0); }
#pragma unroll
          for (int j = 0; j < 8; ++j) acc2[j & 3] += __int_as_float(v[j].y);
        }
        xo += xl;
      }
    };
    if (p.math && p.res) proxy(res, p.res);
    for (int c = 0; c < nch; ++c) {
      int s = c % p.NS, u = c / p.NS;
      bar_wait(full + s, u & 1);
      if (p.math) proxy(ring + (long long)s * p.S, (uint32_t)min((long long)p.S, rem - (long long)c * p.S));
      __syncwarp();
      if (lane == 0) bar_arrive(empty + s);
    }
    tm = gtime();
    acc += acc2[0] + acc2[1] + acc2[2] + acc2[3];
  }
  __syncthreads();
  unsigned long long tr = tm, te = tm;
  if (p.red == 1) {
    // partial tile [mt x n] fp32 -> smem (reuse res+ring) -> one bulk reduce-add
    const int pe = p.mt * p.n; float* ps = (float*)res;
    for (int i = tid; i < pe; i += 256) ps[i] = (i == tid) ? acc : 0.f;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    const int tile = cta / p.ksplit;
    if (tid == L0) {
      asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;"
                   :: "l"(p.out + (long long)tile * pe), "r"(sa(ps)), "r"(pe * 4) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      asm volatile("fence.proxy.async.global;" ::: "memory");
      __threadfence();
      int old = atomicAdd(p.cnt + tile, 1);
      *flag = (old == p.ksplit - 1);
      tr = gtime();
    }
    __syncthreads();
    if (*flag) {   // last CTA of this tile: stage proxy (fp32 -> bf16) and re-zero
      __threadfence();
      float4* o4 = (float4*)(p.out + (long long)tile * pe);
      __nv_bfloat162* ob = (__nv_bfloat162*)(p.outb + (long long)tile * pe);
      for (int i0 = tid; i0 < pe / 4; i0 += 256 * 8) {
        float4 v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = (i0 + j * 256 < pe / 4) ? __ldcg(o4 + i0 + j * 256) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          int i = i0 + j * 256;
          if (i < pe / 4) {
            ob[2 * i] = __floats2bfloat162_rn(v[j].x, v[j].y); ob[2 * i + 1] = __floats2bfloat162_rn(v[j].z, v[j].w);
            __stcg(o4 + i, make_float4(0.f, 0.f, 0.f, 0.f));
          }
        }
      }
      if (tid == 0) p.cnt[tile] = 0;
    }
    __syncthreads();
    te = gtime();
  }
  if (p.red == 2) {
    // partial -> smem -> one bulk store to part[cta]; per-tile arrival counter (monotonic); every CTA of the tile
    // waits for all ksplit partials, then sums its 1/ksplit share of the tile's rows and stages bf16
    const int pe = p.mt * p.n; float* ps = (float*)res;
    for (int i = tid; i < pe; i += 256) ps[i] = (i == tid) ? acc : 0.f;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    const int tile = cta / p.ksplit, kidx = cta % p.ksplit;
    if (tid == L0) {
      asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                   :: "l"(p.out + (long long)cta * pe), "r"(sa(ps)), "r"(pe * 4) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      asm volatile("fence.proxy.async.global;" ::: "memory");
      __threadfence();
      int old = atomicAdd(p.cnt + tile, 1), target = (old / p.ksplit + 1) * p.ksplit, c;
      do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(c) : "l"(p.cnt + tile) : "memory"); } while (c < target);
      tr = gtime();
    }
    __syncthreads();
    const int q4 = pe / 4, per = (q4 + p.ksplit - 1) / p.ksplit, b0 = kidx * per, b1 = min(q4, b0 + per);
    const float4* pt = (const float4*)(p.out + (long long)tile * p.ksplit * pe);
    __nv_bfloat162* ob = (__nv_bfloat162*)(p.outb + (long long)tile * pe);
    for (int i = b0 + tid; i < b1; i += 256) {
      float4 v[16];
#pragma unroll
      for (int k = 0; k < 16; ++k) v[k] = k < p.ksplit ? __ldcg(pt + (long long)k * q4 + i) : make_float4(0.f, 0.f, 0.f, 0.f);
      float4 t = v[0];
#pragma unroll
      for (int k = 1; k < 16; ++k) { t.x += v[k].x; t.y += v[k].y; t.z += v[k].z; t.w += v[k].w; }
      ob[2 * i] = __floats2bfloat162_rn(t.x, t.y); ob[2 * i + 1] = __floats2bfloat162_rn(t.z, t.w);
    }
    __syncthreads();
    te = gtime();
  }
  if (p.red == 4) {   // deferred reduce: plain bulk store of the partial; the next kernel (consumer) sums the ksplit partials
    const int pe = p.mt * p.n; float* ps = (float*)res;
    for (int i = tid; i < pe; i += 256) ps[i] = (i == tid) ? acc : 0.f;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid == L0) {
      asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                   :: "l"(p.out + (long long)cta * pe), "r"(sa(ps)), "r"(pe * 4) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
      tr = gtime();
    }
    te = gtime();
  }
  if (p.red == 3) {
    // cluster of C CTAs along K (consecutive k-splits of one tile): partial -> own smem; cluster barrier; CTA rank r sums
    // its 1/C row share of the C partials over DSMEM; if C == ksplit it stages bf16 directly, else it red.add's its share
    // into the fp32 tile and the last CTA of the tile stages + re-zeroes it.
    const int pe = p.mt * p.n; float* ps = (float*)res;
    for (int i = tid; i < pe; i += 256) ps[i] = (i == tid) ? acc : 0.f;
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (tid == L0) tr = gtime();
    uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    const int tile = cta / p.ksplit, q4 = pe / 4, per = (q4 + p.C - 1) / p.C, b0 = r * per, b1 = min(q4, b0 + per);
    const uint32_t ps_s = sa(ps);
    uint32_t rb[32];
    for (int k = 0; k < p.C; ++k) asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rb[k]) : "r"(ps_s), "r"(k));
    float4* o4 = (float4*)(p.out + (long long)tile * pe);
    __nv_bfloat162* ob = (__nv_bfloat162*)(p.outb + (long long)tile * pe);
    for (int i = b0 + tid; i < b1; i += 256) {
      float4 t = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 8
      for (int k = 0; k < p.C; ++k) {
        float4 v;
        asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];"
                     : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(rb[k] + 16u * i) : "memory");
        t.x += v.x; t.y += v.y; t.z += v.z; t.w += v.w;
      }
      if (p.C == p.ksplit) { ob[2 * i] = __floats2bfloat162_rn(t.x, t.y); ob[2 * i + 1] = __floats2bfloat162_rn(t.z, t.w); }
      else asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(o4 + i), "f"(t.x), "f"(t.y), "f"(t.z), "f"(t.w) : "memory");
    }
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (p.C < p.ksplit) {
      __threadfence();
      __syncthreads();
      if (tid == 0) { int old = atomicAdd(p.cnt + tile, 1); *flag = (old == p.ksplit - 1); }
      __syncthreads();
      if (*flag) {
        __threadfence();
        for (int i = tid; i < q4; i += 256) {
          float4 v = __ldcg(o4 + i);
          ob[2 * i] = __floats2bfloat162_rn(v.x, v.y); ob[2 * i + 1] = __floats2bfloat162_rn(v.z, v.w);
          __stcg(o4 + i, make_float4(0.f, 0.f, 0.f, 0.f));
        }
        if (tid == 0) p.cnt[tile] = 0;
      }
    }
    te = gtime();
  }
  if (p.st && tid == L0) {
    unsigned long long* s = p.st + (long long)cta * 8;
    s[0] = t0; s[1] = tw; s[2] = tres; s[3] = tx; s[4] = tm; s[5] = tr; s[6] = te; s[7] = gtime();
  }
  if (p.tm) {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    if (warp == 7) {
      __syncwarp();
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tbase) : "memory");
    }
  }
  if (acc == 12345.f) p.out[0] = acc;   // keep the proxy live
}

__global__ void __launch_bounds__(128) prod_kernel(const __nv_bfloat16* src, __nv_bfloat16* dst,
    int n, long long busy_ns, unsigned long long* end_stamp, __nv_bfloat16* blk, int kt, int* xc,
    const float4* rd, int rdk, int rdq) {
  __shared__ float pad[448];
  const unsigned long long te0 = gtime();
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  unsigned long long t0 = gtime(), t = t0;
  if (end_stamp && threadIdx.x == 0) { atomicMin(end_stamp + 1, te0); atomicMin(end_stamp + 2, t0); }
  if (rd) {   // consumer proxy: sum rdk partial copies of rdq float4 each (the deferred split-K reduce), spread over the grid
    const int nt = gridDim.x * gridDim.y * 128, me = (blockIdx.y * gridDim.x + blockIdx.x) * 128 + threadIdx.x;
    float s = 0.f;
    for (int i = me; i < rdq; i += nt) {
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int k = 0; k < rdk; ++k) { float4 v = __ldcg(rd + (long long)k * rdq + i); a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w; }
      s += a.x + a.y + a.z + a.w;
    }
    pad[threadIdx.x] = s;
    if (s == 12345.f) dst[0] = __float2bfloat16(s);
    t0 = gtime(); t = t0;
  }
  pad[threadIdx.x] = 0.f;
  while ((long long)(t - t0) < busy_ns) t = gtime();
  const int row = blockIdx.y, col = (blockIdx.x * 128 + threadIdx.x) * 8;
  if (row < n && col < 7168) {
    uint4 v = *reinterpret_cast<const uint4*>(src + (long long)row * 7168 + col);
    *reinterpret_cast<uint4*>(dst + (long long)row * 7168 + col) = v;
    if (kt) { int b = col / kt; *reinterpret_cast<uint4*>(blk + ((long long)b * n + row) * kt + col - b * kt) = v; }
  }
  if (xc) { __syncthreads(); if (threadIdx.x == 0) { __threadfence(); atomicAdd(xc, 1); } }
  if (end_stamp && threadIdx.x == 0) atomicMax(end_stamp, gtime() + (unsigned long long)pad[0]);
}

#define KCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { cudaGetLastError(); \
  TORCH_CHECK(false, #x, ": ", cudaGetErrorString(e_)); } } while (0)
static int g_pmode = 0;  // producer shared-memory mode launch attribute (0 = default, 3 allow / 4 prefer oversized)
static void pdl_launch(const void* fn, dim3 g, dim3 b, size_t smem, void** args, int mode = 0, int cl = 1) {
  cudaLaunchConfig_t cfg{}; cfg.gridDim = g; cfg.blockDim = b; cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[3]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1; cfg.attrs = at; cfg.numAttrs = 1;
  if (smem > 232448 || mode) {
#if CUDART_VERSION < 13040
    TORCH_CHECK(false, "oversized smem needs CUDA >= 13.4");
#endif
#if CUDART_VERSION >= 13040
    at[1].id = cudaLaunchAttributeSharedMemoryMode;
    at[1].val.sharedMemoryMode = mode == 4 ? cudaSharedMemoryModePreferOversizedSharedMemory
                                           : cudaSharedMemoryModeAllowOversizedSharedMemory;
    cfg.numAttrs = 2;
#endif
  }
  if (cl > 1) {
    at[cfg.numAttrs].id = cudaLaunchAttributeClusterDimension;
    at[cfg.numAttrs].val.clusterDim.x = cl; at[cfg.numAttrs].val.clusterDim.y = 1; at[cfg.numAttrs].val.clusterDim.z = 1;
    cfg.numAttrs++;
  }
  KCK(cudaLaunchKernelExC(&cfg, fn, args));
}

void set_pmode(int64_t m) { g_pmode = (int)m; }
void prod(torch::Tensor src, torch::Tensor dst, int64_t busy_ns, torch::Tensor end, torch::Tensor blk, int64_t kt_, torch::Tensor xcnt, torch::Tensor rd, int64_t rdk) {
  static bool set = false;
  if (!set) { KCK(cudaFuncSetAttribute(prod_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100)); set = true; }
  const __nv_bfloat16* s = (const __nv_bfloat16*)src.data_ptr(); __nv_bfloat16* d = (__nv_bfloat16*)dst.data_ptr();
  int n = src.size(0); long long b = busy_ns;
  unsigned long long* e = end.numel() ? (unsigned long long*)end.data_ptr() : nullptr;
  __nv_bfloat16* bk = (__nv_bfloat16*)blk.data_ptr(); int kt = kt_;
  int* xc = xcnt.numel() ? (int*)xcnt.data_ptr() : nullptr;
  const float4* rdp = rd.numel() ? (const float4*)rd.data_ptr() : nullptr; int rk = (int)rdk; int rq = rdp ? (int)(rd.numel() / 4 / rdk) : 0;
  void* args[] = {&s, &d, &n, &b, &e, &bk, &kt, &xc, &rdp, &rk, &rq};
  pdl_launch((const void*)prod_kernel, dim3(7, n), dim3(128), 0, args, g_pmode);
}

void front(torch::Tensor w, int64_t ncta, int64_t bc, torch::Tensor x, torch::Tensor xblk, int64_t ksplit, int64_t kt, int64_t mt,
           int64_t res, int64_t S, int64_t NS, int64_t I, int64_t warm, int64_t xmode, int64_t math, int64_t red,
           torch::Tensor out, torch::Tensor cnt, torch::Tensor outb, torch::Tensor st, int64_t trig, int64_t chunk, int64_t cl,
           int64_t fw, torch::Tensor xcnt, torch::Tensor ep, int64_t tm, int64_t tsb) {
  P p; p.tm = (int)tm; p.tsb = (int)tsb; p.fw = fw; p.ptot = 7 * x.size(0); p.xcnt = fw ? (int*)xcnt.data_ptr() : nullptr; p.ep = fw ? (int*)ep.data_ptr() : nullptr;
  p.C = cl > 1 ? cl : 1; p.trig = trig; p.chunk = chunk; p.w = (const char*)w.data_ptr(); p.bc = bc; p.x = (const __nv_bfloat16*)x.data_ptr(); p.xblk = (const char*)xblk.data_ptr(); p.n = x.size(0);
  p.ksplit = ksplit; p.kt = kt; p.mt = mt; p.res = res; p.S = S; p.NS = NS; p.I = I; p.warm = warm;
  p.xmode = xmode; p.math = math; p.red = red; p.out = (float*)out.data_ptr(); p.cnt = (int*)cnt.data_ptr();
  p.outb = (__nv_bfloat16*)outb.data_ptr(); p.st = st.numel() ? (unsigned long long*)st.data_ptr() : nullptr;
  long long xb = xmode ? (long long)p.n * kt * 2 : 0; if (warm && xb < S) xb = S;
  size_t smem = 1024 + res + NS * S + xb;
  TORCH_CHECK(smem <= 327680, "smem ", smem);
  p.smem = (int)smem;
  TORCH_CHECK(tm % 16384 == 0 && tm <= 262144 && tsb % 16384 == 0 && (!tm || (long long)(smem - 1024) >= tsb), "tm");
#if CUDART_VERSION >= 13040
  if (smem > 232448) KCK(cudaFuncSetAttribute(front_skel, cudaFuncAttributeSharedMemoryMode, cudaSharedMemoryModeAllowOversizedSharedMemory));
#endif
  TORCH_CHECK(!red || (long long)mt * p.n * 4 <= res + NS * S + xb, "partial does not fit");
  if (smem <= 232448) KCK(cudaFuncSetAttribute(front_skel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  TORCH_CHECK(p.red != 3 || (cl >= 1 && cl <= 32 && ksplit % cl == 0 && ncta % cl == 0), "cluster must divide ksplit");
  static int occ_cl = -1, occ_smem = -1;
  if (cl > 1) KCK(cudaFuncSetAttribute(front_skel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  if (occ_cl != cl || occ_smem != (int)smem) {   // report max co-resident clusters once per shape (GPC packing)
    occ_cl = cl; occ_smem = (int)smem;
    cudaLaunchConfig_t c{}; c.gridDim = dim3(ncta); c.blockDim = dim3(256); c.dynamicSmemBytes = smem;
    cudaLaunchAttribute a[2]; int na = 0;
    a[na].id = cudaLaunchAttributeClusterDimension; a[na].val.clusterDim.x = cl; a[na].val.clusterDim.y = 1; a[na].val.clusterDim.z = 1; ++na;
#if CUDART_VERSION >= 13040
    if (smem > 232448) { a[na].id = cudaLaunchAttributeSharedMemoryMode; a[na].val.sharedMemoryMode = cudaSharedMemoryModeAllowOversizedSharedMemory; ++na; }
#endif
    c.attrs = a; c.numAttrs = na; int nc = -1;
    cudaError_t e = cudaOccupancyMaxActiveClusters(&nc, (const void*)front_skel, &c);
    fprintf(stderr, "    occupancy: cluster %d smem %d -> max active clusters %d (%s), need %d\n", (int)cl, (int)smem, nc,
           cudaGetErrorString(e), (int)(ncta / cl));
    cudaGetLastError();
  }
  void* args[] = {&p};
  pdl_launch((const void*)front_skel, dim3(ncta), dim3(256), smem, args, 0, (int)cl);
}
"""


def _mod():
    root = os.environ.get("VLLM_CACHE_ROOT", "/tmp")
    build = os.path.join(root, "k3_skel_moe_front")
    os.makedirs(build, exist_ok=True)
    return load_inline(
        name="k3_skel_moe_front",
        cpp_sources="void set_pmode(int64_t);\nvoid prod(torch::Tensor, torch::Tensor, int64_t, torch::Tensor, torch::Tensor, int64_t, torch::Tensor, torch::Tensor, int64_t);\n"
        "void front(torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t,"
        " int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
        " int64_t, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t, int64_t);",
        cuda_sources="#define KD8 896\n" + _SRC, functions=["set_pmode", "prod", "front"],
        extra_cuda_cflags=["-O3", "-gencode", "arch=compute_100f,code=sm_100f"],
        build_directory=build, verbose=False,
    )  # fmt: skip


def _time(g, iters=20, reps=5):
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(iters):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1e3 / iters)
    return sorted(out)[len(out) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="32,64,96")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--busy-us", type=float, default=4.3)
    ap.add_argument("--smem-kb", type=int, default=197)
    ap.add_argument("--cfg", nargs="+", required=True)
    ap.add_argument("--trig", type=int, default=0, help="front PDL trigger: 0 none (exit), 1 entry, 2 after wait")
    ap.add_argument("--prod-mode", type=int, default=0, help="producer smem mode attr: 0, 3 allow, 4 prefer oversized")
    ap.add_argument("--chunk-kb", type=int, default=32, help="max bytes per bulk op")
    a = ap.parse_args()
    m = _mod()
    m.set_pmode(a.prod_mode)
    dev = torch.device("cuda")
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    L = a.layers
    W = [torch.randint(-100, 100, (WBYTES // 2,), dtype=torch.int16, device=dev) for _ in range(L)]
    src = torch.randn(96, KDIM, device=dev).to(torch.bfloat16)
    print(f"prod-mode {a.prod_mode}  SMs {nsm}  layers {L}  W {WBYTES / 1e6:.1f} MB/layer  busy {a.busy_us} us")
    empty = torch.empty(0, dtype=torch.int64, device=dev)
    for n in [int(v) for v in a.n.split(",")]:
        x = torch.zeros(L, n, KDIM, dtype=torch.bfloat16, device=dev)
        xblk = torch.zeros(L, n * 7680, dtype=torch.bfloat16, device=dev)
        ends = torch.zeros(L, 3, dtype=torch.int64, device=dev)  # producer: max end, min entry, min post-wait

        def reset_ends():
            ends[:, 0] = 0
            ends[:, 1:] = -1

        def graph(fn):
            fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            g.replay()
            torch.cuda.synchronize()
            return g

        def prods(stamp=False):
            for l in range(L):
                m.prod(src[:n], x[l], int(a.busy_us * 1000), ends[l] if stamp else empty, xblk[l], 0, empty, empty, 1)

        tp = _time(graph(prods))
        print(f"n{n}: producer-only {tp / L:.2f} us/layer")
        for spec in a.cfg:
            f = spec.split(":")
            name = f[0]
            ncta, ksplit, res_kb, s_kb, ns, iss, warm, xm, math, red = (int(v) for v in f[1:11])
            trig = int(f[11]) if len(f) > 11 else a.trig
            chunk = (int(f[12]) if len(f) > 12 else a.chunk_kb) * 1024
            cl = int(f[13]) if len(f) > 13 else 1
            fw = int(f[14]) if len(f) > 14 else 0
            tm_kb = int(f[15]) if len(f) > 15 else 0
            if tm_kb < 0:   # all TMEM except n fp32 accumulator columns (one M128 tile at a time)
                tm_kb = (512 - -(-n // 32) * 32) // 2
            xcnt = torch.zeros(L, dtype=torch.int32, device=dev)
            ep = torch.zeros(L, ncta, dtype=torch.int32, device=dev)
            ncta = min(ncta, nsm)
            ntile = ncta // ksplit
            ncta = ntile * ksplit
            bc = -(-WBYTES // ncta // 16) * 16
            kt = -(-KDIM // ksplit // 8) * 8
            mt = -(-MDIM // ntile)
            S = s_kb * 1024
            tm = min(tm_kb * 1024, bc // 16384 * 16384) if len(f) > 15 else 0
            tsb = (int(f[16]) if len(f) > 16 else 16) * 1024
            xb = n * kt * 2 if xm else 0
            if warm:
                xb = max(xb, S)
            if ns < 0:
                ns = min(32, (a.smem_kb * 1024 - 1024 - xb) // S)
            res = (a.smem_kb * 1024 - 1024 - ns * S - xb) // 1024 * 1024 if res_kb < 0 else res_kb * 1024
            res = min(res, bc - tm)
            out = torch.zeros(ncta * mt * n, dtype=torch.float32, device=dev)
            outb = torch.empty(ntile * mt * n, dtype=torch.bfloat16, device=dev)
            cnt = torch.zeros(ntile, dtype=torch.int32, device=dev)
            st = torch.zeros(L, ncta, 8, dtype=torch.int64, device=dev)
            wpad = [w for w in W]
            # W slices: CTA c reads [c*bc, (c+1)*bc); bc*ncta may exceed WBYTES by < 16*ncta: clamp via a padded view
            need = bc * ncta // 2
            wpad = [torch.cat([w, w[: need - w.numel()]]) if need > w.numel() else w for w in W]

            def run(stamp=False):
                for l in range(L):
                    m.prod(src[:n], x[l], int(a.busy_us * 1000), ends[l] if stamp else empty, xblk[l], kt if xm == 2 else 0,
                           xcnt[l : l + 1] if fw else empty, out if red == 4 else empty, ksplit)  # fmt: skip
                    m.front(wpad[l], ncta, bc, x[l], xblk[l], ksplit, kt, mt, res, S, ns, iss, warm, xm, math, red,
                            out, cnt, outb, st[l] if stamp else empty, trig, chunk, cl, fw, xcnt[l : l + 1], ep[l], tm, tsb)  # fmt: skip

            try:
                tf = _time(graph(run))
            except Exception as e:  # noqa: BLE001
                print(f"  {name}: FAILED {str(e)[:200]}")
                continue
            reset_ends()
            st.zero_()
            g = graph(lambda: run(True))
            reset_ends()
            st.zero_()
            g.replay()
            torch.cuda.synchronize()
            # phases relative to the producer end, layers 2..L-1, median over CTAs (p90 in brackets)
            E = ends[2 : L - 1, 0].view(-1, 1).double()
            S8 = st[2 : L - 1].double()
            ph = {}
            for i, k in [(0, "entry"), (2, "res"), (1, "wait"), (3, "x"), (4, "W"), (5, "red"), (6, "epi"), (7, "end")]:
                v = S8[:, :, i]
                ok = v > 0
                if ok.any():
                    d = ((v - E) / 1e3)[ok]
                    ph[k] = (d.median().item(), d.quantile(0.9).item(), d.max().item())
            # next producer: first entry / first post-wait vs this front's last CTA end (gap root cause)
            lastend = st[2 : L - 1, :, 7].double().max(dim=1).values
            nent = ends[3:L, 1].double()
            nwt = ends[3:L, 2].double()
            ph["nxtEnt"] = (((nent - lastend) / 1e3).median().item(),) * 3
            ph["nxtWait"] = (((nwt - lastend) / 1e3).median().item(),) * 3
            ph["nxtBusyEnd"] = (((ends[3:L, 0].double() - lastend) / 1e3).median().item(),) * 3
            s = " ".join(f"{k} {m_:.2f}/{p:.2f}/{mx:.2f}" for k, (m_, p, mx) in ph.items())
            print(f"  {name}: red {red} tsb {tsb // 1024}K tm {tm // 1024}K fw {fw} C {cl} trig {trig} op {chunk // 1024}K smem {(1024 + res + ns * S + xb) // 1024}K ncta {ncta} ({ntile}x{ksplit}) Bc {bc // 1024} KB res {res // 1024} KB "
                  f"pre-wait {(min(res, bc) + min(ns * S, bc - res)) * ncta / 1e6:.1f} MB "
                  f"| exposed {(tf - tp) / L:.2f} us/layer | {s}")  # fmt: skip


if __name__ == "__main__":
    main()
