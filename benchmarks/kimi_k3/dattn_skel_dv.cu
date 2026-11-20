// dattn_skel_dv.cu (D-ATTN M4, from hwdb hw24_fabric_ops.cu; part 'v' = DcpValue skeleton). hw24 (4 GPUs, one process): counted fabric puts as a drop-in for the decode NVLink exchanges, ALL ranks
// exchanging concurrently at the deployed D3-32 per-DP-rank shapes (TP4 = DCP4 = SP4, width 7168 BF16):
//   AG  SP all-gather: every rank sends its n-row shard (n = 32/64/96 at r8/r16/r24) to all 4 gather buffers.
//       L  production Lamport: multimem.st (NVLS) of 16 B words, receivers poll all R rows for the sentinel, re-arm.
//       M  counted multimem fabric put (multicast LE) from smem staging; ONE thread per GPU waits the local counter.
//       MN same as M without the source mbarrier wait (grid exit orders completion).
//       U  3 counted unicast puts + plain local copy of the own shard.
//   RS  SP reduce-scatter: every rank sends 3 x n rows of its BF16 partial to the row owners; owner sums FP32 in rank order.
//       L  production Lamport: st.relaxed.sys v4 into the owner's slot, owner polls + sums + re-arms.
//       C  counted unicast puts into the owner's slot; every owner CTA waits the local counter, then sums.
//   A2A small all-to-all (MoE size exchange / top-k merge messages): M bytes from every rank to every peer, 1 CTA.
//       L  pointer stores of epoch-tagged words, receiver polls the words.  U 3 counted unicast puts.  M 1 multimem put.
//   DCP DcpValue-like push: U bytes to each of 3 owners in 32 KiB units from G CTAs, owner waits for completion.
//       P  16 B pointer stores + per-CTA fence + red.release flag.  T TMA bulk store (cp.async.bulk) to the peer VA +
//       flag.  C counted unicast fabric puts.
// Timing: per GPU a CUDA graph of N dependent kernels (one exchange each), all 4 GPUs replayed together; us/op = max GPU.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s = "?"; cuGetErrorString(r_, &s); \
  printf("CU error %d %s at %s:%d (%s)\n", (int)r_, s, __FILE__, __LINE__, #x); exit(1); } } while (0)
#define RK(x) do { cudaError_t r_ = (x); if (r_ != cudaSuccess) { printf("CUDA error %s at %s:%d (%s)\n", \
  cudaGetErrorString(r_), __FILE__, __LINE__, #x); exit(1); } } while (0)
typedef unsigned long long u64;
typedef unsigned int LeId;
struct LeProp { int type; union { CUdevice device; unsigned numDevices; } u; u64 size; unsigned ipcHandleTypes; unsigned flags; unsigned pad[8]; };
typedef CUresult (*pReserve)(LeId*, unsigned);
typedef CUresult (*pCreate)(LeId, const LeProp*);
typedef CUresult (*pAddDev)(LeId, CUdevice);
typedef CUresult (*pQuery)(LeId, unsigned, int*);
typedef CUresult (*pBindMem)(LeId, CUdevice, u64, CUmemGenericAllocationHandle, u64, u64, u64);
template <class T> static T sym(const char* n) {
  void* f = nullptr; CUdriverProcAddressQueryResult q; CK(cuGetProcAddress(n, &f, 13050, CU_GET_PROC_ADDRESS_DEFAULT, &q));
  if (!f) { printf("missing %s\n", n); exit(1); } return (T)f;
}
constexpr int NR = 4, WID = 7168, ROWB = WID * 2, THR = 256, RMAX = 1024;
constexpr u64 AGB = (u64)RMAX * ROWB;              // one gather parity (R rows)
constexpr u64 SMAX = AGB / NR;                      // max shard bytes
constexpr u64 DAG = 1ull << 20, DRS = DAG + 3 * AGB, DA2A = DRS + 2 * NR * SMAX, DDCP = DA2A + (1ull << 20);
constexpr u64 UDCP = 2560ull << 10;                 // DcpValue bytes per owner (~7 MiB per rank / 3 owners)
constexpr u64 BUFSZ = 128ull << 20;
constexpr u64 C_AGU = 0, C_RS = 256, C_A2AU = 512, C_DCP = 768, C_DCPF = 1024, C_AGM = 0, C_A2AM = 256;  // counter offsets
constexpr unsigned SENT = 0x80000000u;
constexpr int STAGE = 65536;                        // smem staging bytes per CTA

struct Ctx {
  int me; LeId uni[NR]; LeId mc;
  unsigned char* B[NR];  // unicast-LE buffers (pointer-mapped on every GPU)
  unsigned char* MB;     // my multicast-LE replica
  unsigned char* NBmc;   // NVLS multicast VA (Lamport AG)
  unsigned char* NB;     // my NVLS replica
  unsigned char* src;    // local source: x shard / BF16 partial [R, W] / DCP payload
  unsigned char* out;    // RS output shard
  u64* exp;              // [16][4096] per-(arm, CTA) expected counter values
};

__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ u64 gclk() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u64 ld_acq(const void* p) { u64 v; asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void trap_timeout(u64 t0) { if (gclk() - t0 > 4000000000ull) { printf("TIMEOUT\n"); asm volatile("trap;"); } }
__device__ __forceinline__ void waitge(const void* p, u64 want) { u64 t = gclk(); while (ld_acq(p) < want) trap_timeout(t); }
__device__ __forceinline__ void fabric_acquire() { asm volatile("fence.proxy.generic::fabric.alias.acquire.sys;" ::: "memory"); }
__device__ __forceinline__ void bar_init(unsigned b) {
  asm volatile("mbarrier.init.layout::v1.shared::cta.b64 [%0], 1;" :: "r"(b) : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void put_u(unsigned b, LeId le, u64 off, u64 cnt, unsigned src, unsigned S) {
  asm volatile("fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.b128 [%0, %1, %2], [%3], %4, [%5];"
               :: "r"(le), "l"(off), "l"(cnt), "r"(src), "r"(S), "r"(b) : "memory");
}
__device__ __forceinline__ void put_mc(unsigned b, LeId le, u64 off, u64 cnt, unsigned src, unsigned S) {
  asm volatile("fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.b128 [%0, %1, %2], [%3], %4, [%5];"
               :: "r"(le), "l"(off), "l"(cnt), "r"(src), "r"(S), "r"(b) : "memory");
}
__device__ __forceinline__ void submit_expect(unsigned b, unsigned bytes) {
  asm volatile("fabric.submit;" ::: "memory");
  asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes / 16) : "memory");
}
__device__ unsigned errs_dev;
__device__ __forceinline__ void bar_wait(unsigned b, unsigned ph) {
  unsigned done = 0, rep = 0; u64 t = gclk();
  while (!done) {
    trap_timeout(t);
    asm volatile("{.reg .pred P, Q; mbarrier.try_wait.parity.phase_type::primary.relaxed.cta.shared::cta.b64 P|Q, [%2], %3; selp.u32 %0, 1, 0, P; selp.u32 %1, 1, 0, Q;}"
                 : "=r"(done), "=r"(rep) : "r"(b), "r"(ph) : "memory");
  }
  if (rep) atomicAdd(&errs_dev, 1u);
}
__device__ __forceinline__ uint4 ldv(const void* p) { uint4 v; asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }
__device__ __forceinline__ uint4 ldvv(const void* p) { uint4 v; asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_sys(void* p, uint4 v) { asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ void st_mm(void* p, uint4 v) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(p), "f"(__uint_as_float(v.x)), "f"(__uint_as_float(v.y)), "f"(__uint_as_float(v.z)), "f"(__uint_as_float(v.w)) : "memory");
}
__device__ __forceinline__ bool armed(uint4 v) { return v.x == SENT || v.y == SENT || v.z == SENT || v.w == SENT; }
__device__ __forceinline__ uint4 san(uint4 v) { v.x = v.x == SENT ? 0 : v.x; v.y = v.y == SENT ? 0 : v.y; v.z = v.z == SENT ? 0 : v.z; v.w = v.w == SENT ? 0 : v.w; return v; }
// stage [g, g+len) of global into smem (all threads), make it visible to the async proxy
__device__ __forceinline__ void stage(unsigned char* sm, const unsigned char* g, unsigned len) {
  for (unsigned o = threadIdx.x * 16; o < len; o += blockDim.x * 16) *(uint4*)(sm + o) = ldv(g + o);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
}
__device__ __forceinline__ u64 next_exp(u64* e, u64 add) { u64 v = *e + add; *e = v; return v; }

// ---------------------------------------------------------------- AG
// mode 0 L (Lamport NVLS, 3 parities), 1 M (multimem counted put), 2 MN (no source wait), 3 U (unicast counted puts)
__global__ void ag_k(Ctx x, int mode, int n, int par, u64 T, int pdl = 0) {
  extern __shared__ __align__(128) unsigned char sm[];
  __shared__ __align__(8) u64 bar;
  const u64 S = (u64)n * ROWB, R = NR * (u64)n;
  const int G = gridDim.x, cta = blockIdx.x;
  if (pdl) { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); asm volatile("griddepcontrol.wait;" ::: "memory"); }
  if (mode == 4) {  // local copy only: the kernel floor without any exchange
    for (u64 c = ((u64)cta * THR + threadIdx.x) * 16; c < S; c += (u64)G * THR * 16) *(uint4*)(x.B[x.me] + DAG + par * AGB + x.me * S + c) = ldv(x.src + c);
    return;
  }
  if (mode == 5) {  // wait only (split exchange: the put was issued by an earlier kernel)
    if (cta == 0 && threadIdx.x == 0) { waitge(x.MB + C_AGM, next_exp(&x.exp[1 * 4096], T)); fabric_acquire(); }
    return;
  }
  if (mode == 0) {
    unsigned char* mc = x.NBmc + par * AGB + x.me * S;
    for (u64 c = ((u64)cta * THR + threadIdx.x) * 16; c < S; c += (u64)G * THR * 16) st_mm(mc + c, san(ldv(x.src + c)));
    unsigned char* g = x.NB + par * AGB;
    for (u64 c = ((u64)cta * THR + threadIdx.x) * 16; c < R * ROWB; c += (u64)G * THR * 16) {
      u64 t = gclk(); while (armed(ldvv(g + c))) trap_timeout(t);
    }
    unsigned char* old = x.NB + ((par + 2) % 3) * AGB;  // previous iteration's parity: consumed, re-arm
    uint4 s4 = make_uint4(SENT, SENT, SENT, SENT);
    for (u64 c = ((u64)cta * THR + threadIdx.x) * 16; c < R * ROWB; c += (u64)G * THR * 16) *(uint4*)(old + c) = s4;
    return;
  }
  u64 chunk = (S / G + 15) / 16 * 16, beg = cta * chunk, end = min(S, beg + chunk);
  if (threadIdx.x == 0) bar_init(sa(&bar));
  __syncthreads();
  unsigned ph = 0;
  for (u64 base = beg; base < end; base += STAGE) {
    unsigned len = (unsigned)min((u64)STAGE, end - base);
    stage(sm, x.src + base, len);
    if (mode == 3) for (unsigned o = threadIdx.x * 16; o < len; o += THR * 16)
      *(uint4*)(x.B[x.me] + DAG + par * AGB + x.me * S + base + o) = *(uint4*)(sm + o);
    if (threadIdx.x == 0) {
      unsigned tx = 0;
      for (unsigned o = 0; o < len; o += 32768) {
        unsigned s = min(32768u, len - o); u64 dst = DAG + par * AGB + x.me * S + base + o;
        if (mode == 3) { for (int d = 1; d < NR; ++d) put_u(sa(&bar), x.uni[(x.me + d) % NR], dst, C_AGU, sa(sm) + o, s); tx += 3 * s; }
        else { put_mc(sa(&bar), x.mc, dst, C_AGM, sa(sm) + o, s); tx += s; }
      }
      submit_expect(sa(&bar), tx);
      if (mode != 2 || base + STAGE < end) bar_wait(sa(&bar), ph);
      ph ^= 1;
    }
    __syncthreads();
  }
  if (mode == 6) return;
  if (cta == 0 && threadIdx.x == 0) {
    const void* ctr = mode == 3 ? (const void*)(x.B[x.me] + C_AGU) : (const void*)(x.MB + C_AGM);
    waitge(ctr, next_exp(&x.exp[(mode == 2 ? 1 : mode) * 4096], T));
    fabric_acquire();
  }
}
// ---------------------------------------------------------------- RS
// mode 0 L (Lamport pointer push, 2 parities), 1 C (counted unicast puts). slot s of parity p: DRS + (p*NR + s)*S.
__device__ __forceinline__ void sum_store(Ctx& x, const uint4* v, unsigned char* dst) {
  uint4 r; unsigned* rr = (unsigned*)&r;
  for (int w = 0; w < 4; ++w) {
    float lo = 0.f, hi = 0.f;
    for (int s = 0; s < NR; ++s) { unsigned u = ((const unsigned*)&v[s])[w]; lo += __uint_as_float(u << 16); hi += __uint_as_float(u & 0xffff0000u); }
    __nv_bfloat16 a = __float2bfloat16_rn(lo), b = __float2bfloat16_rn(hi);
    rr[w] = (unsigned)__bfloat16_as_ushort(a) | ((unsigned)__bfloat16_as_ushort(b) << 16);
  }
  *(uint4*)dst = san(r);
}
__global__ void rs_k(Ctx x, int mode, int n, int par) {
  extern __shared__ __align__(128) unsigned char sm[];
  __shared__ __align__(8) u64 bar;
  const u64 S = (u64)n * ROWB;
  const int G = gridDim.x, cta = blockIdx.x;
  unsigned char* myslots = x.B[x.me] + DRS + (u64)par * NR * S;
  if (mode == 0) {
    for (u64 c = ((u64)cta * THR + threadIdx.x) * 16; c < 3 * S; c += (u64)G * THR * 16) {
      int o = (x.me + 1 + (int)(c / S)) % NR; u64 w = c % S;
      st_sys(x.B[o] + DRS + ((u64)par * NR + x.me) * S + w, san(ldv(x.src + (u64)o * S + w)));
    }
    for (u64 c = ((u64)cta * THR + threadIdx.x) * 16; c < S; c += (u64)G * THR * 16) {
      uint4 v[NR];
      for (int s = 0; s < NR; ++s) {
        if (s == x.me) { v[s] = san(ldv(x.src + (u64)x.me * S + c)); continue; }
        u64 t = gclk(); do { v[s] = ldvv(myslots + s * S + c); trap_timeout(t); } while (armed(v[s]));
        *(uint4*)(myslots + s * S + c) = make_uint4(SENT, SENT, SENT, SENT);
      }
      sum_store(x, v, x.out + c);
    }
    return;
  }
  u64 tot = 3 * S, chunk = (tot / G + 15) / 16 * 16, beg = cta * chunk, end = min(tot, beg + chunk);
  if (threadIdx.x == 0) bar_init(sa(&bar));
  __syncthreads();
  unsigned ph = 0;
  for (u64 base = beg; base < end; ) {
    int k = (int)(base / S); u64 w = base % S;
    unsigned len = (unsigned)min(min((u64)STAGE, end - base), S - w);  // stay inside one owner
    int o = (x.me + 1 + k) % NR;
    stage(sm, x.src + (u64)o * S + w, len);
    if (threadIdx.x == 0) {
      for (unsigned q = 0; q < len; q += 32768) put_u(sa(&bar), x.uni[o], DRS + ((u64)par * NR + x.me) * S + w + q, C_RS, sa(sm) + q, min(32768u, len - q));
      submit_expect(sa(&bar), len);
      bar_wait(sa(&bar), ph); ph ^= 1;
    }
    __syncthreads();
    base += len;
  }
  if (threadIdx.x == 0) { waitge(x.B[x.me] + C_RS, next_exp(&x.exp[4 * 4096 + cta], 3 * S)); fabric_acquire(); }
  __syncthreads();
  for (u64 c = ((u64)cta * THR + threadIdx.x) * 16; c < S; c += (u64)G * THR * 16) {
    uint4 v[NR];
    for (int s = 0; s < NR; ++s) v[s] = s == x.me ? ldv(x.src + (u64)x.me * S + c) : ldv(myslots + s * S + c);
    sum_store(x, v, x.out + c);
  }
}
// ---------------------------------------------------------------- A2A (1 CTA x THR threads), M bytes per peer
// mode 0 L epoch-tagged pointer words, 1 U counted unicast, 2 M one multimem put into slot [me] of every replica
__global__ void a2a_k(Ctx x, int mode, int M, int par, u64 T) {
  __shared__ __align__(128) unsigned char sm[4096];
  __shared__ __align__(8) u64 bar;
  if (mode == 0) {
    u64 e = x.exp[5 * 4096] + 1;  // read by every thread before the update below
    __syncthreads();
    int words = M / 16;
    for (int i = threadIdx.x; i < 3 * words; i += THR) {
      int o = (x.me + 1 + i / words) % NR;
      unsigned v = (unsigned)e;
      st_sys(x.B[o] + DA2A + ((u64)par * NR + x.me) * 4096 + (i % words) * 16, make_uint4(v, v, v, v));
    }
    for (int i = threadIdx.x; i < 3 * words; i += THR) {
      int s = (x.me + 1 + i / words) % NR;
      const unsigned char* p = x.B[x.me] + DA2A + ((u64)par * NR + s) * 4096 + (i % words) * 16;
      u64 t = gclk(); while (ldvv(p).x != (unsigned)e) trap_timeout(t);
    }
    __syncthreads();
    if (threadIdx.x == 0) x.exp[5 * 4096] = e;
    return;
  }
  for (int o = threadIdx.x * 16; o < M; o += THR * 16) *(uint4*)(sm + o) = make_uint4(x.me, 1, 2, 3);
  if (threadIdx.x == 0) bar_init(sa(&bar));
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (threadIdx.x) return;
  if (mode == 1) { for (int d = 1; d < NR; ++d) put_u(sa(&bar), x.uni[(x.me + d) % NR], DA2A + ((u64)par * NR + x.me) * 4096, C_A2AU, sa(sm), M); submit_expect(sa(&bar), 3 * M); }
  else { put_mc(sa(&bar), x.mc, DA2A + ((u64)par * NR + x.me) * 4096, C_A2AM, sa(sm), M); submit_expect(sa(&bar), M); }
  waitge(mode == 1 ? (const void*)(x.B[x.me] + C_A2AU) : (const void*)(x.MB + C_A2AM), next_exp(&x.exp[(5 + mode) * 4096], T));
  fabric_acquire();
  bar_wait(sa(&bar), 0);
}
// ---------------------------------------------------------------- DCP push: UDCP bytes to each of 3 owners, 32 KiB units
// mode 0 P pointer v4 stores + flag, 1 T TMA bulk store to peer VA + flag, 2 C counted unicast puts
__global__ void dcp_k(Ctx x, int mode, int par, unsigned UNIT) {
  extern __shared__ __align__(128) unsigned char sm[];
  __shared__ __align__(8) u64 bar[2];
  const int G = gridDim.x, cta = blockIdx.x; const int units = (int)(UDCP / UNIT);
  if (threadIdx.x == 0) { bar_init(sa(&bar[0])); bar_init(sa(&bar[1])); }
  __syncthreads();
  unsigned ph[2] = {0, 0}; int k = 0;
  for (int u = cta; u < 3 * units; u += G, ++k) {
    int o = (x.me + 1 + u / units) % NR; u64 w = (u64)(u % units) * UNIT;
    unsigned char* dst = x.B[o] + DDCP + ((u64)par * NR + x.me) * UDCP + w;
    const unsigned char* s = x.src + (u64)o * UDCP + w;
    if (mode == 0) { for (unsigned q = threadIdx.x * 16; q < UNIT; q += THR * 16) st_sys(dst + q, ldv(s + q)); continue; }
    int b = k & 1; unsigned char* buf = sm + b * UNIT;
    if (mode >= 2 && k >= 2 && threadIdx.x == 0) { bar_wait(sa(&bar[b]), ph[b]); ph[b] ^= 1; }
    if (mode == 1 && threadIdx.x == 0) asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
    __syncthreads();
    if (mode != 3) stage(buf, s, UNIT); else __syncthreads();
    if (threadIdx.x == 0) {
      if (mode == 1) {
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(dst), "r"(sa(buf)), "r"(UNIT) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      } else { put_u(sa(&bar[b]), x.uni[o], DDCP + ((u64)par * NR + x.me) * UDCP + w, C_DCP, sa(buf), UNIT); submit_expect(sa(&bar[b]), UNIT); }
    }
  }
  if (mode >= 2) {
    if (threadIdx.x == 0) for (int b = 0; b < 2; ++b) if (k > b) bar_wait(sa(&bar[b]), ph[b]);
    if (cta == 0 && threadIdx.x == 0) { waitge(x.B[x.me] + C_DCP, next_exp(&x.exp[12 * 4096], 3 * UDCP)); fabric_acquire(); }
    return;
  }
  if (mode == 1 && threadIdx.x == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  __syncthreads();
  if (threadIdx.x < 3) {
    asm volatile("fence.acq_rel.sys;" ::: "memory");
    asm volatile("red.release.sys.global.add.u64 [%0], %1;" :: "l"(x.B[(x.me + 1 + threadIdx.x) % NR] + C_DCPF), "l"(1ull) : "memory");
  }
  if (cta == 0 && threadIdx.x == 0) waitge(x.B[x.me] + C_DCPF, next_exp(&x.exp[10 * 4096], 3ull * G));
}
// ---------------------------------------------------------------- NVLS switch-reduction RS / AR (part c)
// partial [R, W] BF16 in this rank's NVLS replica, parity par at NB + par*AGB; flag u32 at NB + 2*AGB (multicast red).
// mode 0 RS: barrier + multimem.ld_reduce.acc::f32 of my n rows -> out
// mode 1 AR one-shot: barrier + ld_reduce of all R rows -> outar
// mode 2 AR two-shot: barrier + ld_reduce my rows -> smem -> counted multimem put to every rank (MB), wait counter
__device__ __forceinline__ uint4 ldred(const void* p) {
  uint4 v; asm volatile("multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__global__ void nvrs_k(Ctx x, int mode, int n, int par, unsigned char* outar) {
  extern __shared__ __align__(128) unsigned char sm[];
  __shared__ __align__(8) u64 bar;
  const u64 S = (u64)n * ROWB; const int G = gridDim.x, cta = blockIdx.x;
  if (cta == 0 && threadIdx.x == 0)
    asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;" :: "l"(x.NBmc + 2 * AGB), "r"(1u) : "memory");
  if (threadIdx.x == 0) {
    u64 want = next_exp(&x.exp[13 * 4096 + cta], NR); u64 t = gclk();
    for (;;) { unsigned v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(x.NB + 2 * AGB) : "memory"); if (v >= (unsigned)want) break; trap_timeout(t); }
    if (mode == 2) bar_init(sa(&bar));
  }
  __syncthreads();
  const unsigned char* mc = x.NBmc + par * AGB;
  if (mode == 0 || mode == 3 || mode == 4) {  // 3/4: 8 ld_reduce in flight per thread (own rows / all rows)
    const u64 off = mode == 4 ? 0 : x.me * S, tot = mode == 4 ? NR * S : S; unsigned char* o = mode == 4 ? outar : x.out;
    if (mode == 0) { for (u64 c = ((u64)cta * THR + threadIdx.x) * 16; c < S; c += (u64)G * THR * 16) *(uint4*)(x.out + c) = ldred(mc + x.me * S + c); return; }
    const u64 step = (u64)G * THR * 16;
    for (u64 c0 = ((u64)cta * THR + threadIdx.x) * 16; c0 < tot; c0 += 8 * step) {
      uint4 v[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) if (c0 + j * step < tot) v[j] = ldred(mc + off + c0 + j * step);
#pragma unroll
      for (int j = 0; j < 8; ++j) if (c0 + j * step < tot) *(uint4*)(o + c0 + j * step) = v[j];
    }
    return;
  }
  if (mode == 1) {
    for (u64 c = ((u64)cta * THR + threadIdx.x) * 16; c < NR * S; c += (u64)G * THR * 16) *(uint4*)(outar + c) = ldred(mc + c);
    return;
  }
  u64 chunk = (S / G + 15) / 16 * 16, beg = cta * chunk, end = min(S, beg + chunk); unsigned ph = 0;
  for (u64 base = beg; base < end; base += STAGE) {
    unsigned len = (unsigned)min((u64)STAGE, end - base);
    for (unsigned o = threadIdx.x * 16; o < len; o += THR * 16) *(uint4*)(sm + o) = ldred(mc + x.me * S + base + o);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
      for (unsigned o = 0; o < len; o += 32768) put_mc(sa(&bar), x.mc, DAG + par * AGB + x.me * S + base + o, C_AGM, sa(sm) + o, min(32768u, len - o));
      submit_expect(sa(&bar), len); bar_wait(sa(&bar), ph); ph ^= 1;
    }
    __syncthreads();
  }
  if (cta == 0 && threadIdx.x == 0) { waitge(x.MB + C_AGM, next_exp(&x.exp[1 * 4096], NR * S)); fabric_acquire(); }
}
// ---------------------------------------------------------------- MegaMoE-style size exchange (part d), 1 CTA
// every rank sends its [E] int32 expert-count row to every rank (row slot [me]) and bulk-reduces it into every rank's
// totals [E]. mode 0 production: cp.async.bulk row + cp.reduce.async.bulk.add.u32 totals per peer, wait_group 0,
// fence.acq_rel.sys, red.relaxed.sys flag per peer; receiver waits flag. mode 1: counted fabric put row + counted
// fabric try_red add.u32 totals per peer, local copy for self; receiver waits its byte counter.
__global__ void sx_k(Ctx x, int mode, unsigned rowb) {
  __shared__ __align__(128) unsigned char sm[8192];
  __shared__ __align__(8) u64 bar;
  const u64 ROWS = DA2A + 65536, TOT = ROWS + NR * 8192, FLAG = C_A2AU + 4096;
  for (unsigned o = threadIdx.x * 4; o < rowb; o += THR * 4) *(unsigned*)(sm + o) = x.me + 1 + o;
  if (threadIdx.x == 0) bar_init(sa(&bar));
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (mode == 0) {
    if (threadIdx.x == 0) {
      for (int d = 0; d < NR; ++d) {
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(x.B[d] + ROWS + x.me * 8192), "r"(sa(sm)), "r"(rowb) : "memory");
        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%0], [%1], %2;" :: "l"(x.B[d] + TOT), "r"(sa(sm)), "r"(rowb) : "memory");
      }
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) asm volatile("fence.acq_rel.sys;" ::: "memory");
    __syncthreads();
    if (threadIdx.x < NR) asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;" :: "l"(x.B[threadIdx.x] + FLAG), "r"(1u) : "memory");
    if (threadIdx.x == 0) {
      u64 want = next_exp(&x.exp[14 * 4096], NR); u64 t = gclk();
      for (;;) { unsigned v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(x.B[x.me] + FLAG) : "memory"); if (v >= (unsigned)want) break; trap_timeout(t); }
    }
    return;
  }
  for (unsigned o = threadIdx.x * 4; o < rowb; o += THR * 4) {  // self: plain row copy + atomic totals
    *(unsigned*)(x.B[x.me] + ROWS + x.me * 8192 + o) = *(unsigned*)(sm + o);
    atomicAdd((unsigned*)(x.B[x.me] + TOT + o), *(unsigned*)(sm + o));
  }
  if (threadIdx.x == 0) {
    for (int k = 1; k < NR; ++k) {
      int d = (x.me + k) % NR;
      put_u(sa(&bar), x.uni[d], ROWS + x.me * 8192, C_A2AU, sa(sm), rowb);
      asm volatile("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.u32 [%0, %1, %2], [%3], %4, [%5];"
                   :: "r"(x.uni[d]), "l"(TOT), "l"(C_A2AU), "r"(sa(sm)), "r"(rowb), "r"(sa(&bar)) : "memory");
    }
    submit_expect(sa(&bar), 2 * (NR - 1) * rowb);
    waitge(x.B[x.me] + C_A2AU, next_exp(&x.exp[15 * 4096], 2ull * (NR - 1) * rowb));
    fabric_acquire();
    bar_wait(sa(&bar), 0);
  }
}
__global__ void spin_k(u64 ns) { if (threadIdx.x == 0) { u64 t = gclk(); while (gclk() - t < ns) {} } }
__global__ void fill_k(unsigned* p, u64 words, unsigned v) { for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < words; i += (u64)gridDim.x * blockDim.x) p[i] = v; }
__global__ void init_src_k(unsigned short* p, u64 n, int me) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x)
    p[i] = __bfloat16_as_ushort(__float2bfloat16((float)((int)((i * 7 + me * 3 + (i >> 13)) % 17) - 8)));
}

// ---------------------------------------------------------------- D-ATTN M4: DcpValue skeleton, owner-PULL form
// Phase S (latB > 0): every CTA streams its share of latB local bytes (latent partial O + W_UV stand-in) with LDG.
// Grid arrive (counter in my LE buffer); the last CTA publishes "my outgoing slots are ready" with one
// red.release.sys per peer (C_PULL). Every CTA waits for 3 peers, then pulls this owner's slot (UDCP bytes) from each
// of the 3 peers with 1D bulk copies (NW issuer warps per CTA, DP slots per warp, UNIT bytes per op) into smem; the
// warp merges the landed slot in registers (+ own local slot for the first source) and writes UNIT output bytes.
constexpr u64 C_PULL = 1280, C_GD = 1536;
__global__ void __launch_bounds__(512, 1) dvp_k(Ctx x, int par, unsigned UNIT, int DP, const unsigned char* lat, u64 latB, int pull) {
  extern __shared__ __align__(1024) unsigned char sm[];
  __shared__ __align__(8) u64 bar[16 * 8];
  const int G = gridDim.x, cta = blockIdx.x, w = threadIdx.x >> 5, ln = threadIdx.x & 31, NW = blockDim.x >> 5;
  float acc = 0.f;
  if (ln == 0) for (int j = 0; j < DP; ++j) bar_init(sa(&bar[w * 8 + j]));
  if (latB) {
    for (u64 c = ((u64)cta * blockDim.x + threadIdx.x) * 16; c < latB; c += (u64)G * blockDim.x * 16) { uint4 v = ldv(lat + c); acc += __uint_as_float(v.x ^ v.w); }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    u64 e = next_exp(&x.exp[14 * 4096 + cta], (u64)G);
    unsigned v = atomicAdd((unsigned*)(x.B[x.me] + C_GD), 1u) + 1u;
    if (v == (unsigned)e) {
      asm volatile("fence.acq_rel.sys;" ::: "memory");
      for (int p = 1; p < NR; ++p) asm volatile("red.release.sys.global.add.u64 [%0], %1;" :: "l"(x.B[(x.me + p) % NR] + C_PULL), "l"(1ull) : "memory");
    }
    if (pull) { waitge(x.B[x.me] + C_PULL, next_exp(&x.exp[15 * 4096 + cta], (u64)(NR - 1))); asm volatile("fence.proxy.async.global;" ::: "memory"); }
  }
  __syncthreads();
  if (!pull) { if (acc == 1.2345f) x.out[0] = 1; return; }
  const int units = (int)(UDCP / UNIT), items = (NR - 1) * units, gw = cta * NW + w, GW = G * NW;
  const int cnt = gw < items ? (items - gw + GW - 1) / GW : 0;
  unsigned char* slot0 = sm + (size_t)w * DP * UNIT;
  auto src_of = [&](int n, int& s, u64& off) { int it = gw + n * GW; s = (x.me + 1 + it / units) % NR; off = (u64)(it % units) * UNIT; };
  auto issue = [&](int n) {
    int s; u64 off; src_of(n, s, off); unsigned b = sa(&bar[w * 8 + n % DP]);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(UNIT) : "memory");
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(sa(slot0 + (size_t)(n % DP) * UNIT)), "l"(x.B[s] + DDCP + ((u64)par * NR + x.me) * UDCP + off), "r"(UNIT), "r"(b) : "memory");
  };
  if (ln == 0) for (int n = 0; n < DP && n < cnt; ++n) issue(n);
  for (int n = 0; n < cnt; ++n) {
    unsigned b = sa(&bar[w * 8 + n % DP]);
    asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W;}" :: "r"(b), "r"((unsigned)((n / DP) & 1)) : "memory");
    int s; u64 off; src_of(n, s, off);
    const unsigned char* sl = slot0 + (size_t)(n % DP) * UNIT;
    const bool first = s == (x.me + 1) % NR;
    for (unsigned o = ln * 16; o < UNIT; o += 512) {
      uint4 v = *(const uint4*)(sl + o);
      if (first) { uint4 m = ldv(x.B[x.me] + DDCP + ((u64)par * NR + x.me) * UDCP + off + o); v.x ^= m.x; *(uint4*)(x.out + off + o) = v; }
      acc += __uint_as_float(v.y);
    }
    __syncwarp();
    if (ln == 0 && n + DP < cnt) issue(n + DP);
  }
  if (acc == 1.2345f) x.out[1] = 1;
}

int main(int argc, char** argv) {
  int ndev = 0; RK(cudaGetDeviceCount(&ndev)); if (ndev < NR) { printf("need 4 GPUs\n"); return 1; }
  CUdevice dv[NR]; int sms = 0;
  for (int d = 0; d < NR; ++d) { RK(cudaSetDevice(d)); RK(cudaFree(0)); CK(cuDeviceGet(&dv[d], d)); }
  RK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  auto reserve = sym<pReserve>("cuLogicalEndpointIdReserve"); auto create = sym<pCreate>("cuLogicalEndpointCreate");
  auto adddev = sym<pAddDev>("cuLogicalEndpointAddDevice"); auto query = sym<pQuery>("cuLogicalEndpointQuery");
  auto bindmem = sym<pBindMem>("cuLogicalEndpointBindMem");
  auto mk = [&](int d, CUmemGenericAllocationHandle& h, CUdeviceptr& va, bool fabric) {
    CUmemAllocationProp p = {}; p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.location.type = CU_MEM_LOCATION_TYPE_DEVICE; p.location.id = d;
    if (fabric) p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC; size_t g = 0; CK(cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    CK(cuMemCreate(&h, BUFSZ, &p, 0)); CK(cuMemAddressReserve(&va, BUFSZ, g, 0, 0)); CK(cuMemMap(va, BUFSZ, 0, h, 0));
    CUmemAccessDesc a[NR] = {}; for (int e = 0; e < NR; ++e) { a[e].location.type = CU_MEM_LOCATION_TYPE_DEVICE; a[e].location.id = e; a[e].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; }
    CK(cuMemSetAccess(va, BUFSZ, a, NR)); RK(cudaSetDevice(d)); RK(cudaMemset((void*)va, 0, BUFSZ)); RK(cudaDeviceSynchronize());
  };
  CUmemGenericAllocationHandle hu[NR], hm[NR]; CUdeviceptr vu[NR], vm[NR]; LeId le[NR], mcle;
  for (int d = 0; d < NR; ++d) {
    mk(d, hu[d], vu[d], true); mk(d, hm[d], vm[d], true);
    LeProp q; memset(&q, 0, sizeof(q)); q.type = 1; q.u.device = dv[d]; q.size = BUFSZ; q.ipcHandleTypes = 1; q.flags = 1;
    CK(reserve(&le[d], 1)); CK(create(le[d], &q)); int st = 0; for (int t = 0; t < 1000000 && !st; ++t) CK(query(le[d], 1, &st));
    RK(cudaSetDevice(d)); CK(bindmem(le[d], dv[d], 0, hu[d], 0, BUFSZ, 0));
  }
  {
    LeProp q; memset(&q, 0, sizeof(q)); q.type = 2; q.u.numDevices = NR; q.size = BUFSZ; q.ipcHandleTypes = 1; q.flags = 1;
    CK(reserve(&mcle, 1)); CK(create(mcle, &q)); for (int d = 0; d < NR; ++d) CK(adddev(mcle, dv[d]));
    int st = 0; for (int t = 0; t < 1000000 && !st; ++t) query(mcle, 1, &st);
    for (int d = 0; d < NR; ++d) { RK(cudaSetDevice(d)); CK(bindmem(mcle, dv[d], 0, hm[d], 0, BUFSZ, 0)); }
  }
  // NVLS multicast object for the Lamport AG (3 parities)
  CUmulticastObjectProp mp = {}; mp.numDevices = NR; mp.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
  size_t gran = 0; CK(cuMulticastGetGranularity(&gran, &mp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
  size_t NBSZ = (3 * AGB + gran - 1) / gran * gran; mp.size = NBSZ;
  CUmemGenericAllocationHandle mco; CK(cuMulticastCreate(&mco, &mp));
  for (int d = 0; d < NR; ++d) CK(cuMulticastAddDevice(mco, dv[d]));
  CUdeviceptr nb[NR];
  CUmemAccessDesc acc[NR] = {}; for (int e = 0; e < NR; ++e) { acc[e].location.type = CU_MEM_LOCATION_TYPE_DEVICE; acc[e].location.id = e; acc[e].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; }
  for (int d = 0; d < NR; ++d) {
    RK(cudaSetDevice(d));
    CUmemAllocationProp ap = {}; ap.type = CU_MEM_ALLOCATION_TYPE_PINNED; ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE; ap.location.id = d;
    CUmemGenericAllocationHandle h; CK(cuMemCreate(&h, NBSZ, &ap, 0)); CK(cuMulticastBindMem(mco, 0, h, 0, NBSZ, 0));
    CK(cuMemAddressReserve(&nb[d], NBSZ, gran, 0, 0)); CK(cuMemMap(nb[d], NBSZ, 0, h, 0)); CK(cuMemSetAccess(nb[d], NBSZ, acc, NR));
  }
  CUdeviceptr nbmc; CK(cuMemAddressReserve(&nbmc, NBSZ, gran, 0, 0)); CK(cuMemMap(nbmc, NBSZ, 0, mco, 0)); CK(cuMemSetAccess(nbmc, NBSZ, acc, NR));
  Ctx X[NR]; cudaStream_t st[NR]; cudaEvent_t e0[NR], e1[NR];
  for (int d = 0; d < NR; ++d) {
    RK(cudaSetDevice(d)); Ctx& x = X[d]; x.me = d; x.mc = mcle;
    for (int e = 0; e < NR; ++e) { x.uni[e] = le[e]; x.B[e] = (unsigned char*)vu[e]; }
    x.MB = (unsigned char*)vm[d]; x.NBmc = (unsigned char*)nbmc; x.NB = (unsigned char*)nb[d];
    RK(cudaMalloc(&x.src, 16ull << 20)); RK(cudaMalloc(&x.out, SMAX)); RK(cudaMalloc(&x.exp, 16 * 4096 * 8)); RK(cudaMemset(x.exp, 0, 16 * 4096 * 8));
    init_src_k<<<256, 256>>>((unsigned short*)x.src, (16ull << 20) / 2, d);
    fill_k<<<512, 256>>>((unsigned*)nb[d], NBSZ / 4, SENT);
    fill_k<<<512, 256>>>((unsigned*)(vu[d] + DRS), 2 * NR * SMAX / 4, SENT);
    RK(cudaFuncSetAttribute(ag_k, cudaFuncAttributeMaxDynamicSharedMemorySize, STAGE));
    RK(cudaFuncSetAttribute(rs_k, cudaFuncAttributeMaxDynamicSharedMemorySize, STAGE));
    RK(cudaFuncSetAttribute(dcp_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
    RK(cudaStreamCreateWithFlags(&st[d], cudaStreamNonBlocking)); RK(cudaEventCreate(&e0[d])); RK(cudaEventCreate(&e1[d]));
    RK(cudaDeviceSynchronize());
  }
  printf("sms %d  LE uni %u %u %u %u mc %u  NVLS %zu B\n", sms, le[0], le[1], le[2], le[3], mcle, NBSZ); fflush(stdout);
  // calibration: does a multimem counted put count the sender's own replica?
  {
    for (int d = 0; d < NR; ++d) { RK(cudaSetDevice(d)); a2a_k<<<1, THR, 0, st[d]>>>(X[d], 2, 256, 0, 0); }
    for (int d = 0; d < NR; ++d) { RK(cudaSetDevice(d)); RK(cudaStreamSynchronize(st[d])); }
  }
  u64 cal[NR]; for (int d = 0; d < NR; ++d) { RK(cudaSetDevice(d)); RK(cudaMemcpy(&cal[d], (void*)(vm[d] + C_A2AM), 8, cudaMemcpyDeviceToHost)); }
  printf("calibration: multimem counted put 256 B from each of 4 ranks -> counters %llu %llu %llu %llu (1024 = own replica counted)\n", cal[0], cal[1], cal[2], cal[3]);
  const bool selfmc = cal[0] >= 1024; fflush(stdout);
  // reset counters/expected after calibration
  for (int d = 0; d < NR; ++d) { RK(cudaSetDevice(d)); RK(cudaMemset((void*)vm[d], 0, 65536)); RK(cudaMemset(X[d].exp, 0, 16 * 4096 * 8)); RK(cudaDeviceSynchronize()); }
  const int N = 96;
  const bool partc = argc > 1 && argv[1][0] == 'c';
  auto run = [&](const char* nm, auto launch) -> double {
    for (int d = 0; d < NR; ++d) { RK(cudaSetDevice(d)); RK(cudaMemset((void*)vu[d], 0, 65536)); RK(cudaMemset((void*)vm[d], 0, 65536));
      if (partc) RK(cudaMemset((void*)(nb[d] + 2 * AGB), 0, 4096));
      RK(cudaMemset(X[d].exp, 0, 16 * 4096 * 8)); fill_k<<<512, 256>>>((unsigned*)(vu[d] + DRS), 2 * NR * SMAX / 4, SENT); RK(cudaDeviceSynchronize()); }  // all GPUs idle: counters restart at 0, Lamport RS slots re-armed
    cudaGraphExec_t ge[NR];
    for (int d = 0; d < NR; ++d) {
      RK(cudaSetDevice(d)); cudaGraph_t g; RK(cudaStreamBeginCapture(st[d], cudaStreamCaptureModeGlobal));
      for (int i = 0; i < N; ++i) launch(d, i);
      RK(cudaStreamEndCapture(st[d], &g)); RK(cudaGraphInstantiate(&ge[d], g, 0)); RK(cudaGraphDestroy(g));
    }
    double best = 1e30;
    for (int r = 0; r < 7; ++r) {
      for (int d = 0; d < NR; ++d) { RK(cudaSetDevice(d)); RK(cudaEventRecord(e0[d], st[d])); RK(cudaGraphLaunch(ge[d], st[d])); RK(cudaEventRecord(e1[d], st[d])); }
      double mx = 0;
      for (int d = 0; d < NR; ++d) { RK(cudaSetDevice(d)); RK(cudaEventSynchronize(e1[d])); float ms; RK(cudaEventElapsedTime(&ms, e0[d], e1[d])); mx = std::max(mx, (double)ms); }
      if (r >= 2) best = std::min(best, mx);
    }
    for (int d = 0; d < NR; ++d) { RK(cudaSetDevice(d)); RK(cudaGraphExecDestroy(ge[d])); }
    double us = best * 1e3 / N; unsigned errs = 0; RK(cudaSetDevice(0)); RK(cudaMemcpyFromSymbol(&errs, errs_dev, 4));
    printf("%-58s %8.2f us/op%s\n", nm, us, errs ? "  FABRIC-ERR-REPORTS" : ""); fflush(stdout);
    return us;
  };
  char nm[160];
  std::vector<unsigned short> h(SMAX / 2), ref(SMAX / 2);
  auto host_val = [](int me, u64 i) { return (float)((int)((i * 7 + me * 3 + (i >> 13)) % 17) - 8); };
  if (argc > 1 && argv[1][0] == 'd') {
    for (unsigned rowb : {512u, 3584u, 8192u}) for (int mode : {0, 1}) {
      snprintf(nm, sizeof nm, "D size exchange %5u B/peer %s", rowb, mode ? "counted fabric put + try_red" : "bulk copy + bulk reduce + fence + flag (production)");
      run(nm, [&](int d, int i) { sx_k<<<1, THR, 0, st[d]>>>(X[d], mode, rowb); });
    }
    printf("DONE\n");
    return 0;
  }
  if (argc > 1 && argv[1][0] == 'v') {  // D-ATTN M4 DcpValue skeleton
    const u64 LATB = 50300000ull;  // 37.7 MB latent partial O (384 x 96 x 512 BF16) + 12.6 MB W_UV (96 x 512 x 128 BF16)
    unsigned char* lat[NR];
    for (int d = 0; d < NR; ++d) { RK(cudaSetDevice(d)); RK(cudaMalloc(&lat[d], 64ull << 20)); RK(cudaMemset(lat[d], 3, 64ull << 20));
      RK(cudaFuncSetAttribute(dvp_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024)); RK(cudaDeviceSynchronize()); }
    run("V push ref: counted unicast puts 32K G=sms (hw24 form)", [&](int d, int i) { dcp_k<<<sms, THR, 65536, st[d]>>>(X[d], 2, i % 2, 32768u); });
    run("V src phase only (50.3 MB LDG) + publish, no pull", [&](int d, int i) { dvp_k<<<sms, 256, 0, st[d]>>>(X[d], i % 2, 8192u, 1, lat[d], LATB, 0); });
    run("V flag round only (no src, no pull bytes: UNIT pull of 0 items)", [&](int d, int i) { dvp_k<<<sms, 256, 0, st[d]>>>(X[d], i % 2, 8192u, 1, lat[d], 0, 0); });
    struct Cf { unsigned U; int NW, DP, G; };
    const Cf cfs[] = {{8192, 4, 2, 0}, {8192, 8, 2, 0}, {8192, 16, 1, 0}, {16384, 4, 2, 0}, {16384, 8, 1, 0}, {16384, 4, 3, 0},
                      {32768, 4, 1, 0}, {32768, 2, 3, 0}, {16384, 8, 1, 128}, {4096, 16, 2, 0}};
    for (u64 lb : {0ull, LATB}) for (const Cf& c : cfs) {
      int G = c.G ? c.G : sms;
      snprintf(nm, sizeof nm, "V pull%s U%-5u NW%-2d DP%d G%-3d", lb ? "+src" : "    ", c.U, c.NW, c.DP, G);
      run(nm, [&](int d, int i) { dvp_k<<<G, c.NW * 32, (size_t)c.NW * c.DP * c.U, st[d]>>>(X[d], i % 2, c.U, c.DP, lat[d], lb, 1); });
    }
    printf("DONE\n");
    return 0;
  }
  if (partc) {
    unsigned char* outar[NR];
    for (int d = 0; d < NR; ++d) {
      RK(cudaSetDevice(d)); RK(cudaMalloc(&outar[d], 16ull << 20)); RK(cudaFuncSetAttribute(nvrs_k, cudaFuncAttributeMaxDynamicSharedMemorySize, STAGE));
      for (int p = 0; p < 2; ++p) RK(cudaMemcpy((void*)(nb[d] + p * AGB), X[d].src, AGB, cudaMemcpyDeviceToDevice));
      RK(cudaDeviceSynchronize());
    }
    for (int n : {8, 32, 72, 96}) {
      const u64 S = (u64)n * ROWB;
      for (int G : {128, sms}) {
        snprintf(nm, sizeof nm, "C RS n%-3d G%-3d lamport push+poll (production)", n, G); run(nm, [&](int d, int i) { rs_k<<<G, THR, 0, st[d]>>>(X[d], 0, n, i % 2); });
        for (int mode : {0, 3, 4, 2}) {
          const char* mn[] = {"RS NVLS barrier + ld_reduce.acc::f32 own rows", "AR one-shot: barrier + ld_reduce all rows", "AR two-shot: ld_reduce own + counted mc put", "RS NVLS ld_reduce x8 in flight", "AR one-shot ld_reduce x8 in flight"};
          snprintf(nm, sizeof nm, "C %s n%-3d G%-3d", mn[mode], n, G);
          run(nm, [&](int d, int i) { nvrs_k<<<G, THR, mode == 2 ? STAGE : 0, st[d]>>>(X[d], mode, n, i % 2, outar[d]); });
          int bad = 0; RK(cudaSetDevice(2));
          if (mode == 0 || mode == 3) { RK(cudaMemcpy(h.data(), X[2].out, S, cudaMemcpyDeviceToHost));
            for (u64 i = 0; i < S / 2; ++i) { float v = 0; for (int r = 0; r < NR; ++r) v += host_val(r, 2 * S / 2 + i); if (h[i] != __bfloat16_as_ushort(__float2bfloat16(v))) ++bad; } }
          if (mode == 1 || mode == 4) { std::vector<unsigned short> hh(2 * S); RK(cudaMemcpy(hh.data(), outar[2], NR * S, cudaMemcpyDeviceToHost));
            for (u64 i = 0; i < 2 * S; ++i) { float v = 0; for (int r = 0; r < NR; ++r) v += host_val(r, i); if (hh[i] != __bfloat16_as_ushort(__float2bfloat16(v))) ++bad; } }
          if (mode == 2) { RK(cudaMemcpy(h.data(), (void*)(vm[2] + DAG + ((N - 1) % 2) * AGB + 1 * S), S, cudaMemcpyDeviceToHost));
            for (u64 i = 0; i < S / 2; ++i) { float v = 0; for (int r = 0; r < NR; ++r) v += host_val(r, 1 * S / 2 + i); if (h[i] != __bfloat16_as_ushort(__float2bfloat16(v))) ++bad; } }
          if (bad) printf("   VERIFY mode %d: %d mismatches\n", mode, bad);
        }
      }
    }
    printf("DONE\n");
    return 0;
  }
  if (argc > 1 && argv[1][0] == 'b') {
    for (int d = 0; d < NR; ++d) { RK(cudaSetDevice(d)); RK(cudaFuncSetAttribute(dcp_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 131072)); }
    auto agx = [&](int d, int G, int mode, int n, int par, u64 T, int pdl) {
      cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(G); cfg.blockDim = dim3(THR); cfg.dynamicSmemBytes = (mode == 0 || mode >= 4 && mode != 6) ? 0 : STAGE; cfg.stream = st[d];
      cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
      cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
      RK(cudaLaunchKernelEx(&cfg, ag_k, X[d], mode, n, par, T, pdl));
    };
    for (int n : {8, 96}) {
      const u64 S = (u64)n * ROWB; const u64 T = selfmc ? 4 * S : 3 * S; const int G = 128;
      snprintf(nm, sizeof nm, "B AG n%-3d local copy only (kernel floor)", n); run(nm, [&](int d, int i) { agx(d, G, 4, n, i % 2, T, 0); });
      snprintf(nm, sizeof nm, "B AG n%-3d local copy only + PDL", n); run(nm, [&](int d, int i) { agx(d, G, 4, n, i % 2, T, 1); });
      snprintf(nm, sizeof nm, "B AG n%-3d lamport", n); run(nm, [&](int d, int i) { agx(d, G, 0, n, i % 3, T, 0); });
      snprintf(nm, sizeof nm, "B AG n%-3d lamport + PDL", n); run(nm, [&](int d, int i) { agx(d, G, 0, n, i % 3, T, 1); });
      snprintf(nm, sizeof nm, "B AG n%-3d counted mc put+wait", n); run(nm, [&](int d, int i) { agx(d, G, 1, n, i % 2, T, 0); });
      snprintf(nm, sizeof nm, "B AG n%-3d counted mc put+wait + PDL", n); run(nm, [&](int d, int i) { agx(d, G, 1, n, i % 2, T, 1); });
      for (u64 sp : {0ull, 3000ull, 6000ull}) {
        snprintf(nm, sizeof nm, "B n%-3d spin %llu ns alone", n, sp); run(nm, [&](int d, int i) { spin_k<<<1, 32, 0, st[d]>>>(sp); });
        snprintf(nm, sizeof nm, "B n%-3d AG(put+wait) -> spin %llu", n, sp); run(nm, [&](int d, int i) { agx(d, G, 1, n, i % 2, T, 0); spin_k<<<1, 32, 0, st[d]>>>(sp); });
        snprintf(nm, sizeof nm, "B n%-3d split: put -> spin %llu -> wait", n, sp);
        run(nm, [&](int d, int i) { agx(d, G, 6, n, i % 2, T, 0); spin_k<<<1, 32, 0, st[d]>>>(sp); agx(d, 1, 5, n, i % 2, T, 0); });
      }
    }
    for (unsigned U : {16384u, 32768u, 65536u}) for (int G : {64, 128, sms}) for (int mode : {2, 3}) {
      snprintf(nm, sizeof nm, "B DCP counted unit %3u KiB G%-3d %s", U / 1024, G, mode == 2 ? "staged from global" : "no staging (push only)");
      run(nm, [&](int d, int i) { dcp_k<<<G, THR, 2 * U, st[d]>>>(X[d], mode, i % 2, U); });
    }
    printf("DONE\n");
    return 0;
  }
  // AG
  for (int n : {8, 32, 64, 96, 256}) {
    const u64 S = (u64)n * ROWB;
    for (int G : {32, 64, 128, sms}) {
      for (int mode : {0, 1, 2, 3}) {
        if (mode == 0 && G < 64) continue;
        if (mode != 0 && (S + G - 1) / G > (u64)STAGE * 4) continue;
        u64 T = mode == 3 ? 3 * S : (selfmc ? 4 * S : 3 * S);
        const char* mn[] = {"L lamport multimem.st+poll", "M counted multimem put", "MN counted mc put, no src wait", "U counted unicast puts"};
        snprintf(nm, sizeof nm, "AG n%-3d (r%d) G%-3d %s", n, n / 4, G, mn[mode]);
        run(nm, [&](int d, int i) { ag_k<<<G, THR, mode ? STAGE : 0, st[d]>>>(X[d], mode, n, mode == 0 ? i % 3 : i % 2, T); });
        if (mode != 0) {  // verify last iteration's gather on GPU 1 (parity (N-1)%2), rank 3's shard
          RK(cudaSetDevice(1)); const unsigned char* base = (mode == 3 ? (unsigned char*)vu[1] : (unsigned char*)vm[1]) + DAG + ((N - 1) % 2) * AGB + 3 * S;
          RK(cudaMemcpy(h.data(), base, S, cudaMemcpyDeviceToHost)); int bad = 0;
          for (u64 i = 0; i < S / 2; ++i) { __nv_bfloat16 b = __float2bfloat16(host_val(3, i)); if (h[i] != __bfloat16_as_ushort(b)) ++bad; }
          if (bad) printf("   VERIFY AG mode %d: %d mismatches\n", mode, bad);
        }
      }
    }
  }
  // RS
  for (int n : {8, 32, 64, 96, 256}) {
    const u64 S = (u64)n * ROWB;
    for (int G : {64, 128, sms}) for (int mode : {0, 1}) {
      snprintf(nm, sizeof nm, "RS n%-3d (r%d) G%-3d %s", n, n / 4, G, mode ? "C counted unicast puts" : "L lamport push+poll");
      run(nm, [&](int d, int i) { rs_k<<<G, THR, mode ? STAGE : 0, st[d]>>>(X[d], mode, n, i % 2); });
      RK(cudaSetDevice(2)); RK(cudaMemcpy(h.data(), X[2].out, S, cudaMemcpyDeviceToHost)); int bad = 0;
      for (u64 i = 0; i < S / 2; ++i) { float s = 0; for (int r = 0; r < NR; ++r) s += host_val(r, 2 * S / 2 + i); if (h[i] != __bfloat16_as_ushort(__float2bfloat16(s))) ++bad; }
      if (bad) printf("   VERIFY RS mode %d: %d mismatches\n", mode, bad);
    }
  }
  // A2A
  for (int M : {256, 4096}) for (int mode : {0, 1, 2}) {
    u64 T = mode == 1 ? 3ull * M : (selfmc ? 4ull * M : 3ull * M);
    const char* mn[] = {"L epoch-tagged pointer words", "U 3 counted unicast puts", "M 1 counted multimem put"};
    snprintf(nm, sizeof nm, "A2A %4d B/peer %s", M, mn[mode]);
    run(nm, [&](int d, int i) { a2a_k<<<1, THR, 0, st[d]>>>(X[d], mode, M, i % 2, T); });
  }
  // DCP push
  for (int G : {32, 64, 128, sms}) for (int mode : {0, 1, 2}) {
    const char* mn[] = {"P pointer v4 stores + flag", "T TMA bulk store to peer + flag", "C counted unicast puts"};
    snprintf(nm, sizeof nm, "DCP %.2f MiB/owner x3 G%-3d %s", UDCP / 1048576.0, G, mn[mode]);
    run(nm, [&](int d, int i) { dcp_k<<<G, THR, 65536, st[d]>>>(X[d], mode, i % 2, 32768u); });
  }
  printf("DONE\n");
  return 0;
}
