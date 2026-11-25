// m1_single.cu: boundary-chain design M1 probes on ONE GPU (probe code, not production).
//   a  : local hand-off chain in a cooperative 1-CTA/SM persistent kernel. Every phase each CTA writes an N-byte tile,
//        then consumes tiles of its group (data-as-flag tags in the last u32 of every 16 B, or the hw56B fenced flag),
//        named barrier, next phase. Per-phase increment = the cost of one dependent L hand-off (bodies empty).
//        Reference arms in the same binary: the same pattern as a chain of kernels with PDL (and without PDL).
//        Optional L2/HBM thrash: one extra warp per CTA streams a 2 GiB buffer during the whole run.
//   k  : K-reduce payload (KDA / boundary-chain phase 1): FP32 partial tiles R x 128 summed over a CS8 / CS16 cluster
//        through DSMEM (reduce-scatter, one or two cluster barriers per phase). The L2-plane form is probe a with
//        group = 8/16, chunk read, N = R*512*4/3 (tag per 16 B).
//   c  : wait-then-run: CTA 0 publishes a tagged tile W us after every consumer finished; the other CTAs poll it
//        (plain / nanosleep / spin_tick-like guard) and then run a tail-like MIO chain or an FFMA chain. Cycles per chain.
//   d  : phase-P staging: a CTA issues TMA bulk loads of a 192 KiB die-local weight slice, then polls a tag that lands
//        W us later; body = read the slice from SMEM. Arms: warm (slice already in SMEM), staged, cold (issue after flag).
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100f,code=sm_100f -lcuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s; cuGetErrorString(r_, &s); \
  printf("CU error %s at %s:%d\n", s, __FILE__, __LINE__); exit(1); } } while (0)
#define RK(x) do { cudaError_t r_ = (x); if (r_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", \
  cudaGetErrorString(r_), __FILE__, __LINE__); exit(1); } } while (0)

__device__ unsigned char g_dom[1024];
__device__ __forceinline__ unsigned smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void st4r(void* p, uint4 v) {
  asm volatile("st.relaxed.gpu.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void st4w(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ uint4 ld4r(const void* p) {
  uint4 v; asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld4w(const void* p) {
  uint4 v; asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned ld_acq(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rel(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void red_rel(unsigned* p, unsigned v) { asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void nbar(int n) { asm volatile("bar.sync 1, %0;" :: "r"(n) : "memory"); }
#define SPIN_CHECK(s) do { if (++(s) > (1ll << 31)) asm volatile("trap;"); } while (0)

static void* alloc_dom(int dev, int dom, size_t bytes) {
  CUmemAllocationProp p = {}; p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
  p.location.localized.deviceId = (unsigned char)dev; p.location.localized.localityDomainId = (unsigned char)dom;
  size_t g = 0; CK(cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)); bytes = (bytes + g - 1) / g * g;
  CUmemGenericAllocationHandle h; CK(cuMemCreate(&h, bytes, &p, 0)); CUdeviceptr va; CK(cuMemAddressReserve(&va, bytes, g, 0, 0));
  CK(cuMemMap(va, bytes, 0, h, 0)); CUmemAccessDesc a = {}; a.location.type = CU_MEM_LOCATION_TYPE_DEVICE; a.location.id = dev;
  a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; CK(cuMemSetAccess(va, bytes, &a, 1)); return (void*)va;
}
// die map (hw56 method): per SM, pointer-chase latency to each die-local buffer; the faster one is the SM's die.
__global__ void lat_k(const unsigned long long* b0, const unsigned long long* b1, size_t words, unsigned* out) {
  if (threadIdx.x) return;
  const unsigned long long* bs[2] = {b0, b1};
  for (int d = 0; d < 2; ++d) {
    size_t idx = (size_t)(blockIdx.x * 7919ull * 1031ull + d * 977) % words;
    unsigned long long acc = 0; long long t0 = clock64();
    for (int i = 0; i < 32; ++i) {
      unsigned long long v; asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(v) : "l"(bs[d] + idx));
      acc += v; idx = (idx + 131088ull + (v & 1)) % words;
    }
    out[4 * blockIdx.x + 1 + d] = (unsigned)((clock64() - t0) / 32);
    if (acc == 12345) out[0] = 1;
  }
  out[4 * blockIdx.x] = smid();
}

// ------------------------------------------------------------------------------------------------ probe a
struct LA {
  unsigned char* box; unsigned* flags; const uint4* thr; size_t thr_n16; unsigned long long* thr_bytes;
  int N, G, rd, mode, K, thrash; unsigned long long* out;
};
// mode 0 = data-as-flag (relaxed v4 stores, tag = phase in .w of every 16 B; consumer polls each 16 B it needs)
// mode 1 = fenced flag (hw56B: weak stores, named barrier, st.release.gpu flag; consumer ld.acquire flag, then loads)
// rd 0 = full: read the whole tile of every OTHER group member; rd 1 = chunk: read my 1/gs chunk of every member's tile
__global__ void __launch_bounds__(288, 1) la_chain(LA a) {
  __shared__ volatile int stop;
  const int c = blockIdx.x, n = gridDim.x, tid = threadIdx.x, W = 256;
  if (tid == 0) stop = 0;
  __syncthreads();
  if (tid >= W) {  // weight-stream stand-in: lane 0 keeps 4 x 32 KiB TMA bulk loads in flight into the (unused) dynamic smem
    if (!a.thrash || tid != W) return;
    extern __shared__ __align__(1024) unsigned char smth[];
    __shared__ __align__(8) unsigned long long tb[4];
    const size_t per = (a.thr_n16 * 16 / n) / 32768 * 32768; const unsigned char* base = (const unsigned char*)a.thr + per * c;
    for (int k = 0; k < 4; ++k) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&tb[k])) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    unsigned ph = 0; size_t off = 0; unsigned long long ops = 0; int inflight = 0;
    auto issue = [&](int k) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(&tb[k])), "r"(32768) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(sa(smth + k * 32768)), "l"(base + off), "r"(32768), "r"(sa(&tb[k])) : "memory");
      off += 32768; if (off + 32768 > per) off = 0; ++ops;
    };
    for (int k = 0; k < 4; ++k) issue(k);
    inflight = 4;
    while (true) {
      for (int k = 0; k < 4; ++k) {
        unsigned done = 0;
        while (!done) asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(done) : "r"(sa(&tb[k])), "r"((ph >> k) & 1u) : "memory");
        ph ^= 1u << k;
        if (!stop) issue(k); else --inflight;
      }
      if (inflight <= 0) break;
    }
    atomicAdd(a.thr_bytes, ops * 32768ull);
    return;
  }
  const int g0 = (c / a.G) * a.G, gs = min(a.G, n - g0), me = c - g0, n16 = a.N / 16;
  const int ch = a.rd ? n16 / gs : n16;              // v4 per member read
  const int nm = a.rd ? gs : gs - 1;                 // members read
  const int tot = ch * nm;
  unsigned acc = 0;
  if (tid == 0) a.out[(size_t)c * (a.K + 1)] = gtime();
  for (int p = 1; p <= a.K; ++p) {
    const int buf = p & 1;
    unsigned char* mine = a.box + ((size_t)buf * n + c) * a.N;
    if (a.mode != 1) {
      for (int i = tid; i < n16; i += W) st4r(mine + (size_t)i * 16, make_uint4(i, c, 3u * p, (unsigned)p));
    } else {
      for (int i = tid; i < n16; i += W) st4w(mine + (size_t)i * 16, make_uint4(i, c, 3u * p, (unsigned)p));
      nbar(W);
      if (tid == 0) st_rel(a.flags + buf * n + c, (unsigned)p);
    }
    if (a.mode == 1) {  // fenced flag: one acquire poll per producer, spread over the threads, then batched loads
      for (int mi = tid; mi < nm; mi += W) {
        int m = mi; if (!a.rd && m >= me) ++m;
        long long sp = 0; while (ld_acq(a.flags + buf * n + g0 + m) != (unsigned)p) SPIN_CHECK(sp);
      }
      nbar(W);
    }
    if (a.mode == 2) {  // data-as-flag, one item at a time (v1 form)
      for (int it = tid; it < tot; it += W) {
        const int mi = it / ch, vi = it - mi * ch;
        int m = mi; if (!a.rd && m >= me) ++m;
        const unsigned char* s1 = a.box + ((size_t)buf * n + g0 + m) * a.N + (size_t)((a.rd ? me * ch : 0) + vi) * 16;
        uint4 v; long long sp = 0;
        do { v = ld4r(s1); SPIN_CHECK(sp); } while (v.w != (unsigned)p);
        acc += v.x ^ v.z;
      }
    } else
    for (int it0 = tid; it0 < tot; it0 += 8 * W) {  // 8 independent 16 B loads in flight per thread, then re-poll stragglers
      uint4 v[8]; const unsigned char* sp8[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int it = it0 + k * W; sp8[k] = nullptr;
        if (it < tot) {
          const int mi = it / ch, vi = it - mi * ch;
          int m = mi; if (!a.rd && m >= me) ++m;
          sp8[k] = a.box + ((size_t)buf * n + g0 + m) * a.N + (size_t)((a.rd ? me * ch : 0) + vi) * 16;
          v[k] = a.mode == 0 ? ld4r(sp8[k]) : ld4w(sp8[k]);
        }
      }
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        if (!sp8[k]) continue;
        long long sp = 0;
        if (a.mode == 0) while (v[k].w != (unsigned)p) { v[k] = ld4r(sp8[k]); SPIN_CHECK(sp); }
        acc += v[k].x ^ v[k].z;
      }
    }
    nbar(W);
    if (tid == 0) a.out[(size_t)c * (a.K + 1) + p] = gtime();
  }
  if (tid == 0) stop = 1;
  if (acc == 0x12345678u) a.out[1] = 0;
}
// PDL / plain chain reference: one launch per phase, same per-CTA write + read pattern, no polling.
__global__ void __launch_bounds__(256, 1) la_kern(LA a, int p) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int c = blockIdx.x, n = gridDim.x, tid = threadIdx.x, W = 256;
  const int g0 = (c / a.G) * a.G, gs = min(a.G, n - g0), me = c - g0, n16 = a.N / 16;
  const int ch = a.rd ? n16 / gs : n16, nm = a.rd ? gs : gs - 1, tot = ch * nm;
  unsigned acc = 0;
  if (p > 1) {
    const int buf = (p - 1) & 1;
    for (int it0 = tid; it0 < tot; it0 += 8 * W) {
      uint4 v[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int it = it0 + k * W; v[k] = make_uint4(0, 0, 0, 0);
        if (it < tot) {
          const int mi = it / ch, vi = it - mi * ch;
          int m = mi; if (!a.rd && m >= me) ++m;
          v[k] = ld4w(a.box + ((size_t)buf * n + g0 + m) * a.N + (size_t)((a.rd ? me * ch : 0) + vi) * 16);
        }
      }
#pragma unroll
      for (int k = 0; k < 8; ++k) acc += v[k].x ^ v[k].z;
    }
  }
  unsigned char* mine = a.box + ((size_t)(p & 1) * n + c) * a.N;
  for (int i = tid; i < n16; i += W) st4w(mine + (size_t)i * 16, make_uint4(i, c, acc, (unsigned)p));
  __syncthreads();
  if (tid == 0) a.out[(size_t)c * (a.K + 1) + p] = gtime();
}

// ------------------------------------------------------------------------------------------------ probe k (DSMEM)
__global__ void __launch_bounds__(256, 1) kred(int R, int K, int dbl, unsigned long long* out) {
  extern __shared__ __align__(16) float smf[];
  unsigned CS, rank; asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(CS)); asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const int tid = threadIdx.x, nf4 = R * 128 / 4, ch = nf4 / CS;
  float acc = 0;
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  if (tid == 0) out[(size_t)blockIdx.x * (K + 1)] = gtime();
  for (int p = 1; p <= K; ++p) {
    float4* buf = (float4*)smf + (dbl ? (p & 1) * nf4 : 0);
    for (int i = tid; i < nf4; i += 256) buf[i] = make_float4((float)p, (float)i, (float)rank, 1.f);
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
    for (int it0 = tid; it0 < ch * (int)CS; it0 += 8 * 256) {  // 8 remote loads in flight per thread
      float4 x[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int it = it0 + k * 256; x[k] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (it < ch * (int)CS) {
          const int m = it / ch, i = rank * ch + (it - m * ch);
          unsigned ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(sa(buf + i)), "r"(m));
          asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(x[k].x), "=f"(x[k].y), "=f"(x[k].z), "=f"(x[k].w) : "r"(ra) : "memory");
        }
      }
#pragma unroll
      for (int k = 0; k < 8; ++k) acc += x[k].x + x[k].y + x[k].z + x[k].w;
    }
    if (!dbl) asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
    else __syncthreads();
    if (tid == 0) out[(size_t)blockIdx.x * (K + 1) + p] = gtime();
  }
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  if (acc == 1.2345f) out[0] = 0;
}

// ------------------------------------------------------------------------------------------------ probe c
struct PW { uint4* tile; unsigned* done; int I; long long Wns; int pform, chain, len; unsigned long long* tpub; unsigned long long* rec; };
__device__ __noinline__ void guard_timeout() {
  printf("K31_SPIN_TIMEOUT kernel=m1 site=m1_single.cu\n");
  const unsigned long long t = gtime();
  while (gtime() - t < 1000000000ull) asm volatile("nanosleep.u32 1000;");
  asm volatile("trap;");
}
__device__ __forceinline__ uint4 poll_tile(const uint4* p, unsigned tag, int pform) {
  uint4 v; long long sp = 0;
  if (pform == 0) {
    do { v = ld4r(p); SPIN_CHECK(sp); } while (v.w != tag);
  } else if (pform == 1) {
    while (true) { v = ld4r(p); if (v.w == tag) break; asm volatile("nanosleep.u32 64;"); SPIN_CHECK(sp); }
  } else {  // spin_tick shape (k31_spin_guard._tick, SPIN_POLLS=1024): poll count, %globaltimer every 1024 polls, cold printf/trap
    unsigned nn = 0; unsigned long long t0 = 0;
    while (true) {
      v = ld4r(p); if (v.w == tag) break;
      nn += 1;
      if ((nn & 1023u) == 0) {
        const unsigned long long t = gtime();
        if (t0 == 0) t0 = t; else if (t - t0 > 10000000000ull) guard_timeout();
      }
    }
  }
  return v;
}
__device__ __forceinline__ float run_chain(int chain, int len, float v, float* ws, int lane) {
  if (chain == 0) {  // tail-like MIO chain: SHFL x2, STS, LDS, FFMA per step (WaveMoeTail exchange_squares/scale shape)
#pragma unroll 1
    for (int k = 0; k < 24 * len; ++k) {
      v += __shfl_xor_sync(0xffffffffu, v, 1);
      v += __shfl_xor_sync(0xffffffffu, v, 2);
      ws[lane] = v; __syncwarp();
      v = fmaf(ws[lane ^ 5], 1.0001f, v * 0.5f); __syncwarp();
    }
  } else {
#pragma unroll 1
    for (int k = 0; k < 64 * len; ++k) {
#pragma unroll
      for (int j = 0; j < 8; ++j) v = fmaf(v, 1.0000001f, 0.5f);
    }
  }
  return v;
}
__global__ void __launch_bounds__(256, 1) pw_kern(PW a) {
  extern __shared__ __align__(16) float smp[];
  const int c = blockIdx.x, n = gridDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float v = (float)tid;
  for (int it = 1; it <= a.I; ++it) {
    if (c == 0) {
      if (tid == 0) {
        long long sp = 0;
        while (ld_acq(a.done) < (unsigned)((it - 1) * (n - 1))) SPIN_CHECK(sp);
        if (a.Wns > 0) { const unsigned long long t = gtime() + a.Wns; while (gtime() < t) {} }
        a.tpub[it] = gtime();
      }
      __syncthreads();
      st4r(a.tile + tid, make_uint4(it, tid, 0, (unsigned)it));
      continue;
    }
    if (a.Wns < 0) {  // pre-landed control: ~-Wns ns of FFMA before the poll, so the tile has landed already
      const unsigned long long t = gtime() - a.Wns;
      while (gtime() < t) { for (int j = 0; j < 32; ++j) v = fmaf(v, 0.9999999f, 0.25f); }
    }
    const unsigned long long tpoll = gtime();
    uint4 x = poll_tile(a.tile + tid, (unsigned)it, a.pform);
    v += (float)(x.y & 1);
    __syncthreads();
    unsigned long long t0 = 0; long long c0 = 0;
    if (tid == 0) { t0 = gtime(); c0 = clock64(); }
    v = run_chain(a.chain, a.len, v, smp + warp * 64, lane);
    __syncthreads();
    if (tid == 0) {
      const long long c1 = clock64(); const unsigned long long t1 = gtime();
      unsigned long long* r = a.rec + ((size_t)it * n + c) * 4;
      r[0] = tpoll; r[1] = t0; r[2] = t1 - t0; r[3] = (unsigned long long)(c1 - c0);
      red_rel(a.done, 1);
    }
  }
  if (v == 1.2345f) a.rec[0] = 0;
}

// ------------------------------------------------------------------------------------------------ probe d
constexpr int ST_BYTES = 196608, ST_OP = 32768;
struct STP { const unsigned char* w0; const unsigned char* w1; uint4* tile; unsigned* done; int I, arm, nrot; long long Wns; unsigned long long* tpub; unsigned long long* rec; };
__device__ __forceinline__ void tma_slice(unsigned char* sm, const unsigned char* src, unsigned long long* mbar) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(mbar)), "r"(ST_BYTES) : "memory");
  for (int o = 0; o < ST_BYTES; o += ST_OP)
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(sa(sm + o)), "l"(src + o), "r"(ST_OP), "r"(sa(mbar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long* mbar, unsigned ph) {
  unsigned done = 0;
  while (!done) asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(done) : "r"(sa(mbar)), "r"(ph) : "memory");
}
// arm 0 warm (slice resident in SMEM from before the loop), 1 staged (TMA issued before the poll), 2 cold (TMA after the flag)
__global__ void __launch_bounds__(256, 1) st_kern(STP a) {
  extern __shared__ __align__(1024) unsigned char sms[];
  __shared__ __align__(8) unsigned long long mbar;
  const int c = blockIdx.x, n = gridDim.x, tid = threadIdx.x;
  const unsigned char* wb = (g_dom[smid() & 1023] & 1) ? a.w1 : a.w0;
  auto src = [&](int it) { return wb + ((size_t)(it % a.nrot) * n + c) * ST_BYTES; };
  unsigned ph = 0;
  if (tid == 0) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&mbar)) : "memory"); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  __syncthreads();
  if (c != 0 && a.arm == 0) { if (tid == 0) tma_slice(sms, src(0), &mbar); mbar_wait(&mbar, ph); ph ^= 1; }
  float v = (float)tid;
  for (int it = 1; it <= a.I; ++it) {
    if (c == 0) {
      if (tid == 0) {
        long long sp = 0;
        while (ld_acq(a.done) < (unsigned)((it - 1) * (n - 1))) SPIN_CHECK(sp);
        if (a.Wns > 0) { const unsigned long long t = gtime() + a.Wns; while (gtime() < t) {} }
        a.tpub[it] = gtime();
      }
      __syncthreads();
      st4r(a.tile + tid, make_uint4(it, tid, 0, (unsigned)it));
      continue;
    }
    unsigned long long t_iss = 0, t_land = 0, t_flag = 0, t_rdy = 0, t_end = 0;
    if (tid == 0) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      t_iss = gtime();
      if (a.arm == 1) tma_slice(sms, src(it), &mbar);
      red_rel(a.done, 1);
    }
    if (a.arm == 1 && tid == 224) { mbar_wait(&mbar, ph); t_land = gtime(); a.rec[((size_t)it * n + c) * 6 + 5] = t_land; }
    uint4 x = poll_tile(a.tile + tid, (unsigned)it, 0);
    v += (float)(x.y & 1);
    __syncthreads();
    if (tid == 0) { t_flag = gtime(); if (a.arm == 2) tma_slice(sms, src(it), &mbar); }
    if (a.arm != 0) { mbar_wait(&mbar, ph); ph ^= 1; }
    __syncthreads();
    if (tid == 0) t_rdy = gtime();
    const uint4* s4 = (const uint4*)sms;
#pragma unroll 4
    for (int i = tid; i < ST_BYTES / 16; i += 256) { uint4 w = s4[i]; v = fmaf(__uint_as_float(w.x & 0x3f800000u), v, __uint_as_float(w.y & 0x3f800000u)); v = fmaf(v, 0.999f, __uint_as_float(w.z & 0x3f800000u)); }
    __syncthreads();
    if (tid == 0) {
      t_end = gtime();
      unsigned long long* r = a.rec + ((size_t)it * n + c) * 6;
      r[0] = t_iss; r[1] = t_flag; r[2] = t_rdy; r[3] = t_end; r[4] = a.tpub[it];
    }
  }
  if (v == 1.2345f) a.rec[0] = 0;
}

// ------------------------------------------------------------------------------------------------ host
static double pct(std::vector<double> v, double q) { if (v.empty()) return -1; std::sort(v.begin(), v.end()); return v[(size_t)(q * (v.size() - 1))]; }
static int g_nsm;
static unsigned char* g_d0; static unsigned char* g_d1;

static void coop(const void* f, int grid, int block, size_t smem, void** args) {
  RK(cudaLaunchCooperativeKernel(f, dim3(grid), dim3(block), args, smem, 0));
  RK(cudaDeviceSynchronize());
}
static const size_t SMEM_1PER = 150 * 1024;  // forces 1 CTA per SM

// per-CTA slope of phase stamps over phases [K0, K]; returns median and max over CTAs (us per phase)
static void slope(const std::vector<unsigned long long>& h, int ncta, int K, int K0, double& med, double& mx) {
  std::vector<double> s;
  for (int c = 0; c < ncta; ++c) {
    const unsigned long long* r = &h[(size_t)c * (K + 1)];
    s.push_back((double)(long long)(r[K] - r[K0]) / (K - K0) / 1e3);
  }
  med = pct(s, .5); mx = pct(s, 1.0);
}

static void probe_a(int reps) {
  const int n = g_nsm, K = 400, K0 = 50;
  unsigned char* box; RK(cudaMalloc(&box, (size_t)2 * n * 360 * 1024));
  unsigned* flags; RK(cudaMalloc(&flags, 2 * n * 4));
  const size_t thrB = 2ull << 30; uint4* thr; RK(cudaMalloc(&thr, thrB)); RK(cudaMemset(thr, 1, thrB));
  unsigned long long* tb; RK(cudaMalloc(&tb, 8));
  unsigned long long* out; RK(cudaMalloc(&out, (size_t)n * (K + 1) * 8));
  std::vector<unsigned long long> h((size_t)n * (K + 1));
  RK(cudaFuncSetAttribute(la_chain, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_1PER));
  RK(cudaFuncSetAttribute(la_kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_1PER));
  printf("== M1a local hand-off chain: %d CTAs (1/SM), K=%d phases (slope over %d..%d), %d reps; us per phase = median CTA / max CTA\n", n, K, K0, K, reps);
  printf("A form       N_B   G   rd thrash | per-phase med (rep spread)  max | ingress_B/CTA | thrash TB/s\n");
  struct Arm { int N, G, rd; };
  // per-B x-slice bytes per CTA (rows x 7168 x 2 / 212, rows = 8B): B1 541 .. B64 34.6 KB, rounded to 16 B
  std::vector<Arm> arms = {
      {544, 2, 0}, {1088, 2, 0}, {2176, 2, 0}, {4336, 2, 0}, {8656, 2, 0}, {17312, 2, 0}, {34624, 2, 0},
      {544, 8, 0}, {2176, 8, 0}, {544, 16, 0},       // fan-in 7 / 15 full tiles (o-proj reads 6 head slots)
      {4336, 8, 1}, {34624, 8, 1}, {34624, 16, 1},   // reduce-scatter within 8 / 16
      {4336, 0, 1}, {34624, 0, 1}};                  // all-to-all transpose over all CTAs (combine)
  for (int mode : {2, 0, 1})
    for (int thrash = 0; thrash < 2; ++thrash)
      for (auto ar : arms) {
        LA a; a.box = box; a.flags = flags; a.thr = thr; a.thr_n16 = thrB / 16; a.thr_bytes = tb; a.N = ar.N; a.G = ar.G ? ar.G : n;
        a.rd = ar.rd; a.mode = mode; a.K = K; a.thrash = thrash; a.out = out;
        std::vector<double> meds, mxs, bws;
        for (int r = 0; r < reps; ++r) {
          RK(cudaMemset(box, 0, (size_t)2 * n * ar.N)); RK(cudaMemset(flags, 0, 2 * n * 4)); RK(cudaMemset(tb, 0, 8));
          void* args[] = {&a};
          cudaEvent_t e0, e1; RK(cudaEventCreate(&e0)); RK(cudaEventCreate(&e1)); RK(cudaEventRecord(e0));
          coop((const void*)la_chain, n, 288, SMEM_1PER, args);
          RK(cudaEventRecord(e1)); RK(cudaEventSynchronize(e1)); float ms; RK(cudaEventElapsedTime(&ms, e0, e1));
          RK(cudaMemcpy(h.data(), out, h.size() * 8, cudaMemcpyDeviceToHost));
          double md, mx; slope(h, n, K, K0, md, mx); meds.push_back(md); mxs.push_back(mx);
          unsigned long long hb; RK(cudaMemcpy(&hb, tb, 8, cudaMemcpyDeviceToHost)); bws.push_back(hb / (ms * 1e-3) / 1e12);
        }
        const int gs = std::min(a.G, n), ing = ar.rd ? ar.N : ar.N * (gs - 1);
        printf("A %-9s %6d %3d %d  %d     | %6.3f (%5.3f-%5.3f)  %6.3f | %8d | %5.2f\n", mode == 2 ? "daf-1" : mode ? "flag" : "daf-b8", ar.N, gs, ar.rd, thrash,
               pct(meds, .5), pct(meds, 0), pct(meds, 1), pct(mxs, .5), ing, thrash ? pct(bws, .5) : 0.0);
        fflush(stdout);
      }
  // kernel-chain reference: K launches (PDL / plain) in one graph, same write + read pattern
  printf("== M1a reference: kernel per phase (%d CTAs, 150 KiB smem -> 1/SM), graph of K=%d launches; us per edge = slope of CTA exit stamps\n", n, 200);
  const int KK = 200;
  for (int pdl = 1; pdl >= 0; --pdl)
    for (auto ar : std::vector<Arm>{{544, 2, 0}, {4336, 2, 0}, {34624, 2, 0}, {2176, 8, 0}, {34624, 8, 1}, {34624, 0, 1}}) {
      LA a; a.box = box; a.flags = flags; a.thr = thr; a.thr_n16 = 0; a.thr_bytes = tb; a.N = ar.N; a.G = ar.G ? ar.G : n;
      a.rd = ar.rd; a.mode = 2; a.K = KK; a.thrash = 0; a.out = out;
      cudaStream_t st; RK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
      RK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
      for (int p = 1; p <= KK; ++p) {
        cudaLaunchConfig_t cfg = {}; cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = pdl;
        cfg.gridDim = dim3(n); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = SMEM_1PER; cfg.stream = st; cfg.attrs = at; cfg.numAttrs = 1;
        RK(cudaLaunchKernelEx(&cfg, la_kern, a, p));
      }
      cudaGraph_t g; RK(cudaStreamEndCapture(st, &g)); cudaGraphExec_t ge; RK(cudaGraphInstantiate(&ge, g, 0));
      std::vector<double> meds, mxs;
      for (int r = 0; r < reps + 1; ++r) {
        RK(cudaGraphLaunch(ge, st)); RK(cudaStreamSynchronize(st));
        if (r == 0) continue;
        RK(cudaMemcpy(h.data(), out, (size_t)n * (KK + 1) * 8, cudaMemcpyDeviceToHost));
        // per launch: last CTA exit
        std::vector<double> lastx(KK + 1, 0);
        for (int p = 1; p <= KK; ++p) { unsigned long long m = 0; for (int c = 0; c < n; ++c) m = std::max(m, h[(size_t)c * (KK + 1) + p]); lastx[p] = (double)m; }
        meds.push_back((lastx[KK] - lastx[20]) / (KK - 20) / 1e3);
      }
      printf("A %-9s %6d %3d %d  0     | %6.3f (%5.3f-%5.3f)  per edge (last-exit slope)\n", pdl ? "pdl-kern" : "plain-kern", ar.N, std::min(a.G, n), ar.rd,
             pct(meds, .5), pct(meds, 0), pct(meds, 1));
      fflush(stdout);
      RK(cudaGraphExecDestroy(ge)); RK(cudaGraphDestroy(g)); RK(cudaStreamDestroy(st));
    }
  RK(cudaFree(box)); RK(cudaFree(flags)); RK(cudaFree(thr)); RK(cudaFree(tb)); RK(cudaFree(out));
}

static void probe_k(int reps) {
  const int K = 200, K0 = 20;
  RK(cudaFuncSetAttribute(kred, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  RK(cudaFuncSetAttribute(kred, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024));
  unsigned long long* out; RK(cudaMalloc(&out, (size_t)256 * (K + 1) * 8));
  std::vector<unsigned long long> h((size_t)256 * (K + 1));
  printf("== M1k DSMEM K-reduce (FP32 partial R x 128 per CTA, reduce-scatter over the cluster), K=%d phases; us per phase med CTA / max CTA\n", K);
  for (int CS : {8, 16})
    for (int R : {8, 64, 128, 256, 512}) {
      const int tile = R * 512; const int dbl = 2 * tile <= 200 * 1024 ? 1 : 0;
      if (tile > 200 * 1024) { printf("K dsmem CS%-2d R%-4d tile %6d B: > 200 KiB smem (needs oversized mode), skipped\n", CS, R, tile); continue; }
      const size_t smem = std::max((size_t)(dbl ? 2 : 1) * tile, SMEM_1PER);
      cudaLaunchConfig_t cfg = {}; cudaLaunchAttribute at[1];
      at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = CS; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
      cfg.gridDim = dim3(CS); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = smem; cfg.attrs = at; cfg.numAttrs = 1;
      int ncl = 0; RK(cudaOccupancyMaxActiveClusters(&ncl, (void*)kred, &cfg));
      cfg.gridDim = dim3(ncl * CS);
      std::vector<double> meds, mxs;
      for (int r = 0; r < reps; ++r) {
        RK(cudaLaunchKernelEx(&cfg, kred, R, K, dbl, out)); RK(cudaDeviceSynchronize());
        RK(cudaMemcpy(h.data(), out, (size_t)ncl * CS * (K + 1) * 8, cudaMemcpyDeviceToHost));
        double md, mx; slope(h, ncl * CS, K, K0, md, mx); meds.push_back(md); mxs.push_back(mx);
      }
      printf("K dsmem CS%-2d R%-4d tile %6d B %s %2d clusters | %6.3f (%5.3f-%5.3f) %6.3f\n", CS, R, tile, dbl ? "dbuf 1 barrier " : "1buf 2 barriers", ncl,
             pct(meds, .5), pct(meds, 0), pct(meds, 1), pct(mxs, .5));
      fflush(stdout);
    }
  RK(cudaFree(out));
  // L2-plane form: probe a with group 8/16, chunk read, tag per 16 B
  const int n = g_nsm, KA = 200, KA0 = 20;
  unsigned char* box; RK(cudaMalloc(&box, (size_t)2 * n * 360 * 1024));
  unsigned long long* o2; RK(cudaMalloc(&o2, (size_t)n * (KA + 1) * 8)); std::vector<unsigned long long> h2((size_t)n * (KA + 1));
  RK(cudaFuncSetAttribute(la_chain, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_1PER));
  for (int G : {8, 16})
    for (int R : {8, 64, 128, 256, 512}) {
      int N = (R * 512 * 4 / 3 + 15) / 16 * 16; N = (N + G * 16 - 1) / (G * 16) * (G * 16);
      if ((size_t)N > 360 * 1024) continue;
      LA a = {}; a.box = box; a.N = N; a.G = G; a.rd = 1; a.mode = 0; a.K = KA; a.thrash = 0; a.out = o2;
      std::vector<double> meds, mxs;
      for (int r = 0; r < reps; ++r) {
        RK(cudaMemset(box, 0, (size_t)2 * n * N)); void* args[] = {&a};
        coop((const void*)la_chain, n, 288, SMEM_1PER, args);
        RK(cudaMemcpy(h2.data(), o2, h2.size() * 8, cudaMemcpyDeviceToHost));
        double md, mx; slope(h2, n, KA, KA0, md, mx); meds.push_back(md); mxs.push_back(mx);
      }
      printf("K l2plane G%-2d R%-4d tile %6d B (tagged)       %3d CTAs | %6.3f (%5.3f-%5.3f) %6.3f\n", G, R, N, n, pct(meds, .5), pct(meds, 0), pct(meds, 1), pct(mxs, .5));
      fflush(stdout);
    }
  RK(cudaFree(box)); RK(cudaFree(o2));
}

static void probe_c(int reps) {
  const int n = g_nsm, I = 120, I0 = 20;
  uint4* tile; RK(cudaMalloc(&tile, 4096)); unsigned* done; RK(cudaMalloc(&done, 4));
  unsigned long long* tpub; RK(cudaMalloc(&tpub, (I + 1) * 8)); unsigned long long* rec; RK(cudaMalloc(&rec, (size_t)(I + 1) * n * 32));
  std::vector<unsigned long long> hr((size_t)(I + 1) * n * 4), hp(I + 1);
  RK(cudaFuncSetAttribute(pw_kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_1PER));
  printf("== M1c wait-then-run: %d CTAs (CTA 0 publishes), %d iterations (stats over %d..%d x %d CTAs), %d reps\n", n, I, I0, I, n - 1, reps);
  printf("C chain poll      W_ns  | wait med us | chain ns p10/p50/p90 | cycles p50 (rep spread) | cyc/ns | ratio vs pre-landed\n");
  const char* pn[] = {"plain", "nsleep64", "guardlike"};
  const char* env_len = getenv("M1C_LENS"); std::vector<int> lens = {1, 4};
  if (env_len && env_len[0] == '1') lens = {1};
  for (int len : lens)
  for (int chain = 0; chain < 2; ++chain)
    for (int pform = 0; pform < 3; ++pform) {
      if (len > 1 && pform == 2) continue;
      double base = -1;
      std::vector<long long> Ws = {-5000ll, 0ll, 1000ll, 3000ll, 5000ll, 10000ll, 20000ll};
      if (len > 1) Ws = {-5000ll, 0ll, 5000ll, 20000ll};
      for (long long W : Ws) {
        std::vector<double> ns, cy, wt, cyrep;
        for (int r = 0; r < reps; ++r) {
          RK(cudaMemset(tile, 0, 4096)); RK(cudaMemset(done, 0, 4));
          PW a; a.tile = tile; a.done = done; a.I = I; a.Wns = W; a.pform = pform; a.chain = chain; a.len = len; a.tpub = tpub; a.rec = rec;
          void* args[] = {&a}; coop((const void*)pw_kern, n, 256, SMEM_1PER, args);
          RK(cudaMemcpy(hr.data(), rec, hr.size() * 8, cudaMemcpyDeviceToHost));
          std::vector<double> cyr;
          for (int it = I0; it <= I; ++it)
            for (int c = 1; c < n; ++c) {
              const unsigned long long* x = &hr[((size_t)it * n + c) * 4];
              ns.push_back((double)x[2]); cy.push_back((double)x[3]); cyr.push_back((double)x[3]); wt.push_back((double)(long long)(x[1] - x[0]) / 1e3);
            }
          cyrep.push_back(pct(cyr, .5));
        }
        const double c50 = pct(cy, .5);
        if (W < 0) base = c50;
        printf("C %-5s x%d %-9s %6lld | %6.2f | %6.0f %6.0f %6.0f | %7.0f (%6.0f-%6.0f) | %5.2f | %5.2f | excess ns p50 %5.0f\n", chain ? "ffma" : "mio", len, pn[pform], W, pct(wt, .5),
               pct(ns, .1), pct(ns, .5), pct(ns, .9), c50, pct(cyrep, 0), pct(cyrep, 1), c50 / pct(ns, .5), base > 0 ? c50 / base : 1.0,
               base > 0 ? (c50 - base) / (c50 / pct(ns, .5)) : 0.0);
        fflush(stdout);
      }
    }
  RK(cudaFree(tile)); RK(cudaFree(done)); RK(cudaFree(tpub)); RK(cudaFree(rec));
}

static void probe_d(int reps) {
  const int n = g_nsm, I = 80, I0 = 10, NROT = 40;
  uint4* tile; RK(cudaMalloc(&tile, 4096)); unsigned* done; RK(cudaMalloc(&done, 4));
  unsigned long long* tpub; RK(cudaMalloc(&tpub, (I + 1) * 8)); unsigned long long* rec; RK(cudaMalloc(&rec, (size_t)(I + 1) * n * 48));
  std::vector<unsigned long long> hr((size_t)(I + 1) * n * 6);
  RK(cudaFuncSetAttribute(st_kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ST_BYTES));
  printf("== M1d phase-P staging: %d CTAs, %d KiB die-local slice per CTA (%d x %d KiB TMA bulk ops, 1 issuer), %.1f MB per iteration, %d reps\n",
         n, ST_BYTES >> 10, ST_BYTES / ST_OP, ST_OP >> 10, (double)ST_BYTES * (n - 1) / 1e6, reps);
  printf("D arm     W_ns  | flag-issue us | landed-issue us p50/p90/max | post-flag us p50/p90 (rep spread) | ready-flag p50/p90 | body p50\n");
  const char* an[] = {"warm", "staged", "cold"};
  for (long long W : {1000ll, 2500ll, 5000ll, 10000ll})
    for (int arm = 0; arm < 3; ++arm) {
      std::vector<double> fi, li, pf, rf, bd, pfrep;
      for (int r = 0; r < reps; ++r) {
        RK(cudaMemset(tile, 0, 4096)); RK(cudaMemset(done, 0, 4)); RK(cudaMemset(rec, 0, hr.size() * 8));
        STP a; a.w0 = g_d0; a.w1 = g_d1; a.tile = tile; a.done = done; a.I = I; a.arm = arm; a.nrot = NROT; a.Wns = W; a.tpub = tpub; a.rec = rec;
        void* args[] = {&a}; coop((const void*)st_kern, n, 256, ST_BYTES, args);
        RK(cudaMemcpy(hr.data(), rec, hr.size() * 8, cudaMemcpyDeviceToHost));
        std::vector<double> pr;
        for (int it = I0; it <= I; ++it)
          for (int c = 1; c < n; ++c) {
            const unsigned long long* x = &hr[((size_t)it * n + c) * 6];
            const double iss = (double)x[0];
            fi.push_back(((double)x[1] - iss) / 1e3);
            if (arm == 1) li.push_back(((double)x[5] - iss) / 1e3);
            pf.push_back(((double)x[3] - (double)x[1]) / 1e3); pr.push_back(pf.back());
            rf.push_back(((double)x[2] - (double)x[1]) / 1e3); bd.push_back(((double)x[3] - (double)x[2]) / 1e3);
          }
        pfrep.push_back(pct(pr, .5));
      }
      printf("D %-6s %6lld | %6.2f | %6.2f %6.2f %6.2f | %6.2f %6.2f (%5.2f-%5.2f) | %6.2f %6.2f | %5.2f\n", an[arm], W, pct(fi, .5),
             pct(li, .5), pct(li, .9), pct(li, 1), pct(pf, .5), pct(pf, .9), pct(pfrep, 0), pct(pfrep, 1), pct(rf, .5), pct(rf, .9), pct(bd, .5));
      fflush(stdout);
    }
  RK(cudaFree(tile)); RK(cudaFree(done)); RK(cudaFree(tpub)); RK(cudaFree(rec));
}

int main(int argc, char** argv) {
  const std::string which = argc > 1 ? argv[1] : "akcd"; const int reps = argc > 2 ? atoi(argv[2]) : 3;
  RK(cudaSetDevice(0)); RK(cudaFree(0));
  RK(cudaDeviceGetAttribute(&g_nsm, cudaDevAttrMultiProcessorCount, 0));
  int clk = 0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
  printf("m1_single: %d SMs, clock attr %d kHz, probes '%s', reps %d\n", g_nsm, clk, which.c_str(), reps);
  const size_t PB = 2ull << 30;
  g_d0 = (unsigned char*)alloc_dom(0, 0, PB); g_d1 = (unsigned char*)alloc_dom(0, 1, PB);
  RK(cudaMemset(g_d0, 0x3c, PB)); RK(cudaMemset(g_d1, 0x3c, PB));
  {
    int nb = 8 * g_nsm; unsigned* lo; RK(cudaMalloc(&lo, nb * 16)); RK(cudaMemset(lo, 0, nb * 16));
    lat_k<<<nb, 32>>>((const unsigned long long*)g_d0, (const unsigned long long*)g_d1, (1ull << 30) / 8, lo); RK(cudaDeviceSynchronize());
    std::vector<unsigned> hl(nb * 4); RK(cudaMemcpy(hl.data(), lo, nb * 16, cudaMemcpyDeviceToHost));
    std::vector<double> a0(1024, 0), a1(1024, 0); std::vector<int> cnt(1024, 0);
    for (int b = 0; b < nb; ++b) { unsigned s = hl[4 * b]; if (s >= 1024) continue; a0[s] += hl[4 * b + 1]; a1[s] += hl[4 * b + 2]; cnt[s]++; }
    std::vector<unsigned char> dom(1024, 0); int nd1 = 0;
    for (int s = 0; s < 1024; ++s) if (cnt[s]) { dom[s] = a0[s] / cnt[s] < a1[s] / cnt[s] ? 0 : 1; nd1 += dom[s]; }
    RK(cudaMemcpyToSymbol(g_dom, dom.data(), 1024));
    printf("die map: %d SMs on die 1\n", nd1);
  }
  if (which.find('a') != std::string::npos) probe_a(reps);
  if (which.find('k') != std::string::npos) probe_k(reps);
  if (which.find('c') != std::string::npos) probe_c(reps);
  if (which.find('d') != std::string::npos) probe_d(reps);
  printf("DONE\n");
  return 0;
}
