// Dual-chain sync SKELETON of a persistent prefill KDA core (P-KDA2 SOL protocol).
// Data movement + sync only, no math: same bytes and the same per-step tcgen05/TMEM/mbarrier edges as the
// WY+TF4 BT32 chain step, for NCH (1|2) interleaved (seq, head) chains per SM.
//   per chain step t (32 tokens):
//     loader (prep warp 0, elected): TMA q,k,v,g tiles [32 x 128] bf16 (32 KiB) -> raw ring (2 stages)
//     prep WG (4 warps): raw -> operand image (W^T, Kd fp32, Qd, V^T, B': 42 KiB), proxy fence, IMG_FULL
//     chain group (PW x 4 warps): wait FIN(t-1); state pass (tcgen05.ld S fp32 -> st INP bf16 + st S decayed);
//        bar; issuer: MMA3v (SS, u^T) | MMA1 (TS f16 A=INP, D=U) | MMA4 (TS tf32 A=U, D=S) -> commit FIN;
//        wait OFREE(t-1); OUT (TS f16 A=INP + TS tf32 A=U) -> commit OFIN + IMG_FREE
//     epilogue WG (4 warps): wait OFIN; tcgen05.ld OUT 32 cols; OFREE; bf16 st.shared; TMA store [32 x 128] bf16
// Placement: mode 0 = striped cudaMalloc [T][96][128]; mode 1 = die-local (per-die VMM [T][48][128], CTA works on
// heads of its own die, die = (smid >> 1) & 1).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CU(x) do { CUresult e = (x); if (e != CUDA_SUCCESS) { const char* s; cuGetErrorString(e, &s); printf("CU %s @%d: %s\n", #x, __LINE__, s); exit(1); } } while (0)

constexpr int T_TOK = 16384, HD = 128, BT = 32;
constexpr int RAW_B = 4 * BT * HD * 2;      // 32 KiB: q,k,v,g tiles
constexpr int IMG_B = 42 * 1024;            // W^T 8K | Kd fp32 16K | Qd 8K | V^T 8K | B' 2K
constexpr int OUT_B = BT * HD * 2;          // 8 KiB
constexpr int NST = 2;
constexpr int CH_SMEM = NST * RAW_B + NST * IMG_B + OUT_B;

struct Maps { CUtensorMap in[2][4]; CUtensorMap out[2]; };   // [die][q,k,v,g], [die]

__device__ __forceinline__ uint64_t sdesc(uint32_t saddr) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | ((uint64_t)1 << 46) | ((uint64_t)2 << 61);
}
__device__ __forceinline__ void mwait(uint32_t mb, uint32_t ph) {
  uint32_t ok = 0; long long t0 = clock64();
  while (true) {
    asm volatile("{.reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; selp.u32 %0, 1, 0, P;}" : "=r"(ok) : "r"(mb), "r"(ph) : "memory");
    if (ok) return;
    if (clock64() - t0 > 20000000000LL) { printf("TIMEOUT mb %x ph %u blk %d tid %d\n", mb, ph, blockIdx.x, threadIdx.x); __trap(); }
  }
}
__device__ __forceinline__ void arrive(uint32_t mb) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mb) : "memory"); }
__device__ __forceinline__ void minit(uint32_t mb, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mb), "r"(n) : "memory"); }
__device__ __forceinline__ void nbar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void commit_el(uint32_t mb) {
  asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff; @P tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];}" :: "r"(mb) : "memory");
}
__device__ __forceinline__ void mma_ss_f16(uint32_t d, uint64_t a, uint64_t b, unsigned idesc, uint32_t acc) {
  asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; @P tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, Q;}"
               :: "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc) : "memory");
}
__device__ __forceinline__ void mma_ts_f16(uint32_t d, uint32_t a, uint64_t b, unsigned idesc, uint32_t acc) {
  asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, Q;}"
               :: "r"(d), "r"(a), "l"(b), "r"(idesc), "r"(acc) : "memory");
}
__device__ __forceinline__ void mma_ts_tf32(uint32_t d, uint32_t a, uint64_t b, unsigned idesc, uint32_t acc) {
  asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; @P tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, Q;}"
               :: "r"(d), "r"(a), "l"(b), "r"(idesc), "r"(acc) : "memory");
}
#define LD32(r, addr) asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];" \
  : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),"=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]), \
    "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),"=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31]) : "r"(addr) : "memory")
#define ST32(addr, r) asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" \
  :: "r"(addr), "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),"r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]), \
    "r"(r[16]),"r"(r[17]),"r"(r[18]),"r"(r[19]),"r"(r[20]),"r"(r[21]),"r"(r[22]),"r"(r[23]),"r"(r[24]),"r"(r[25]),"r"(r[26]),"r"(r[27]),"r"(r[28]),"r"(r[29]),"r"(r[30]),"r"(r[31]) : "memory")
#define ST16(addr, r) asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" \
  :: "r"(addr), "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),"r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]) : "memory")

#define LD16(r, addr) asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];" \
  : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),"=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]) : "r"(addr) : "memory")
#define ST8(addr, r) asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" \
  :: "r"(addr), "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]) : "memory")

struct Args { int nch, steps, mode, flags, pfill, efill, bgfill; unsigned id_f16_n32, id_tf32_n128, id_tf32_n32, id_f16_n64; unsigned* ctr; long long* stamps; };
// math fill (calibration): n iterations x 32 independent-chain FFMA (+~2 loop instr) per warp per step, result kept live
__device__ __forceinline__ float mfill(int n, float x) {
  float a[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) a[j] = x + j;
#pragma unroll 1
  for (int i = 0; i < n; ++i) {
#pragma unroll
    for (int u = 0; u < 4; ++u)
#pragma unroll
      for (int j = 0; j < 8; ++j) a[j] = fmaf(a[j], 1.0001f, 0.5f);
  }
  return ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
}

static unsigned mk(int dfmt, int afmt, int bfmt, int N, int M) {
  return ((unsigned)dfmt << 4) | ((unsigned)afmt << 7) | ((unsigned)bfmt << 10) | ((unsigned)(N >> 3) << 17) | ((unsigned)(M >> 4) << 24);
}
static void* alloc_dom(size_t bytes, int die, int dev) {
  CUmemAllocationProp p = {};
  p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  p.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
  p.location.localized.deviceId = dev;
  p.location.localized.localityDomainId = die;
  size_t g; CU(cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  size_t sz = (bytes + g - 1) / g * g;
  CUmemGenericAllocationHandle h; CU(cuMemCreate(&h, sz, &p, 0));
  CUdeviceptr va; CU(cuMemAddressReserve(&va, sz, g, 0, 0));
  CU(cuMemMap(va, sz, 0, h, 0));
  CUmemAccessDesc a = {}; a.location.type = CU_MEM_LOCATION_TYPE_DEVICE; a.location.id = dev; a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU(cuMemSetAccess(va, sz, &a, 1));
  return (void*)va;
}
static void mkmap(CUtensorMap* m, void* p, int H) {
  cuuint64_t dims[3] = {HD, (cuuint64_t)H, T_TOK};
  cuuint64_t str[2] = {HD * 2, (cuuint64_t)H * HD * 2};
  cuuint32_t box[3] = {HD, 1, BT}, es[3] = {1, 1, 1};
  CU(cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, p, dims, str, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// ===== skel2c: calibration variant (1 chain/SM, BT32) with fip's role layout =====
// fip: compute warps 0-3, epilogue 4-7, MMA/load 8-11, five 4-warp prep instances 12-31.
// Here: chain group warps 0..4*PW-1, prep instance 0 = warps 8-11, epilogue 12-15, prep instances 1..NPI-1 = warps
// 16.., loader = warp 4 (PW=1) or warp 28 (PW=2, NPI <= 4). Chunk t is prepared by instance t % NPI into image slot
// t % NPI (own IMG_FULL/IMG_FREE), raw input ring of NR slots (RAW_FULL tx / RAW_FREE 128 arrivals).
// Paced math: pfill iterations of mfill per prep warp per chunk, efill per epilogue warp per chunk, cfill per chain warp.
constexpr int NR = 2, MAXI = 5;
struct CArgs { int steps, mode, flags, npi, pfill, efill, cfill; unsigned id_f16_n32, id_tf32_n128, id_tf32_n32; long long* stamps; };

template <int PW>
__global__ void __launch_bounds__(1024, 1) skelc(const __grid_constant__ Maps maps, CArgs A) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ uint32_t tmem_base;
  __shared__ __align__(8) uint64_t mbar[32];
  __shared__ int raw_turn;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t s0 = ((uint32_t)__cvta_generic_to_shared(smem) + 127) & ~127u;
  unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  const int die = (smid >> 1) & 1;
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"((uint32_t)__cvta_generic_to_shared(&tmem_base)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  auto MB = [&](int i) { return (uint32_t)__cvta_generic_to_shared(&mbar[i]); };
  // 0,1 RAW_FULL | 2,3 RAW_FREE | 4..8 IMG_FULL | 9..13 IMG_FREE | 14 FIN | 15 OFIN | 16 OFREE
  if (threadIdx.x == 0) {
    for (int r = 0; r < NR; ++r) { minit(MB(r), 1); minit(MB(2 + r), 128); }
    for (int i = 0; i < MAXI; ++i) { minit(MB(4 + i), 128); minit(MB(9 + i), 1); }
    minit(MB(14), 1); minit(MB(15), 1); minit(MB(16), 128);
    raw_turn = 0;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t tbase = tmem_base;
  const int NPI = A.npi;
  const int head = A.mode == 1 ? (blockIdx.x % 48) : (blockIdx.x % 96);
  const int mdie = A.mode == 1 ? die : 0;
  const int seq = (blockIdx.x / 96) % 3;
  auto tok0 = [&](int t) { return seq * 5461 + (t * BT) % 5440; };
  const uint32_t img = s0, raw = s0 + MAXI * IMG_B, osm = raw + NR * RAW_B;
  const uint32_t cb = tbase;                        // S 0..127 | INP 128..191 | U 192..223 | OUT 224..255
  const uint32_t lane_off = (uint32_t)((warp & 3) * 32) << 16;
  const int steps = A.steps;
  const int loader_warp = PW == 1 ? 4 : 28;
  int inst = -1, pt = 0;
  if (warp >= 8 && warp < 12) { inst = 0; pt = threadIdx.x - 256; }
  else if (warp >= 16 && warp != loader_warp) { int i = 1 + (warp - 16) / 4; if (i < NPI) { inst = i; pt = threadIdx.x - (512 + (i - 1) * 128); } }
  if (warp < 4 * PW) {
    // ---------------- chain group ----------------
    long long ph[3] = {0, 0, 0};
    const int part = warp >> 2, ncol = 128 / PW;
    for (int t = 0; t < steps; ++t) {
      const int s = t % NPI; const uint32_t sp = (t / NPI) & 1;
      long long c0 = clock64();
      if (t > 0) mwait(MB(14), (t - 1) & 1);
      asm volatile("tcgen05.fence::after_thread_sync;");
      const uint32_t im = img + s * IMG_B;
      if (warp == 0 && !(A.flags & 8)) {
        mwait(MB(4 + s), sp);
        asm volatile("tcgen05.fence::after_thread_sync;");
        for (int kk = 0; kk < 2; ++kk) mma_ss_f16(cb + 192, sdesc(im + 32768) + kk * 2, sdesc(im + 40960) + kk * 2, A.id_f16_n32, kk);
      }
      long long c1 = clock64();
      if (!(A.flags & 4)) {
#pragma unroll 1
        for (int j = 0; j < ncol; j += 16) {
          uint32_t r[16], p[8];
          const uint32_t col = part * ncol + j;
          LD16(r, cb + lane_off + col);
          asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
          for (int e = 0; e < 8; ++e)
            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(p[e]) : "f"(__uint_as_float(r[2 * e + 1])), "f"(__uint_as_float(r[2 * e])));
          ST8(cb + lane_off + 128 + col / 2, p);
#pragma unroll
          for (int e = 0; e < 16; ++e) r[e] = __float_as_uint(__uint_as_float(r[e]) * 0.99f);
          ST16(cb + lane_off + col, r);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
      }
      if (A.cfill) { float f = mfill(A.cfill, (float)lane); if (f == 1234.5f) A.stamps[1] = 1; }
      asm volatile("tcgen05.fence::before_thread_sync;");
      nbar(1, 128 * PW);
      asm volatile("tcgen05.fence::after_thread_sync;");
      long long c2 = clock64();
      if (warp == 0 && !(A.flags & 8)) {
        uint64_t dW = sdesc(im), dKd = sdesc(im + 8192), dQ = sdesc(im + 24576);
        for (int kk = 0; kk < 8; ++kk) mma_ts_f16(cb + 192, cb + 128 + kk * 8, dW + kk * 2, A.id_f16_n32, 1);
        for (int kk = 0; kk < 4; ++kk) mma_ts_tf32(cb, cb + 192 + kk * 8, dKd + kk * 2, A.id_tf32_n128, 1);
        if (t > 0) mwait(MB(16), (t - 1) & 1);
        for (int kk = 0; kk < 8; ++kk) mma_ts_f16(cb + 224, cb + 128 + kk * 8, dQ + kk * 2, A.id_f16_n32, kk);
        commit_el(MB(14));
        for (int kk = 0; kk < 4; ++kk) mma_ts_tf32(cb + 224, cb + 192 + kk * 8, dKd + kk * 2, A.id_tf32_n32, 1);
        commit_el(MB(15));
        commit_el(MB(9 + s));
      } else if (warp == 0) {
        mwait(MB(4 + s), sp);
        if (lane == 0) { arrive(MB(14)); if (t > 0) mwait(MB(16), (t - 1) & 1); arrive(MB(15)); arrive(MB(9 + s)); }
        __syncwarp();
      }
      long long c3 = clock64();
      ph[0] += c1 - c0; ph[1] += c2 - c1; ph[2] += c3 - c2;
    }
    if (warp == 0 && lane == 0) { long long* o = A.stamps + blockIdx.x * 8; o[0] = ph[0]; o[1] = ph[1]; o[2] = ph[2]; o[3] = die; }
  } else if (warp == loader_warp) {
    if (lane == 0) {
      const CUtensorMap* mp = &maps.in[mdie][0];
      for (int t = 0; t < steps; ++t) {
        const int r = t % NR;
        if (t >= NR) mwait(MB(2 + r), ((t / NR) - 1) & 1);
        if ((A.flags & 1) && t >= NR) { arrive(MB(r)); continue; }
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(MB(r)), "r"(RAW_B) : "memory");
        for (int i = 0; i < 4; ++i)
          asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                       :: "r"(raw + r * RAW_B + i * 8192), "l"((uint64_t)(mp + i)), "r"(0), "r"(head), "r"(tok0(t)), "r"(MB(r)) : "memory");
      }
    }
  } else if (inst >= 0) {
    // ---------------- prep instance ----------------
    const uint32_t is = img + inst * IMG_B;
    for (int t = inst; t < steps; t += NPI) {
      const int k = t / NPI, r = t % NR;
      // raw-ring parity is only unambiguous once chunk t-1 has passed its RAW_FULL wait (instances run out of order)
      if (lane == 0) { int v; do { asm volatile("ld.volatile.shared.b32 %0, [%1];" : "=r"(v) : "r"((uint32_t)__cvta_generic_to_shared(&raw_turn))); } while (v < t); }
      __syncwarp();
      mwait(MB(r), (t / NR) & 1);
      if (pt == 0) asm volatile("st.volatile.shared.b32 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&raw_turn)), "r"(t + 1) : "memory");
      mwait(MB(9 + inst), (k & 1) ^ 1);
      const uint32_t rs = raw + r * RAW_B;
#pragma unroll 3
      for (int i = 0; i < ((A.flags & 16) ? 0 : IMG_B / 16 / 128); ++i) {
        const int idx = pt + 128 * i;
        uint32_t a, b, cc, d;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(cc), "=r"(d) : "r"(rs + (idx & 2047) * 16));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(is + idx * 16), "r"(a), "r"(b), "r"(cc), "r"(d) : "memory");
      }
      arrive(MB(2 + r));
      if (A.pfill) { float f = mfill(A.pfill, (float)pt); if (f == 1234.5f) asm volatile("st.shared.f32 [%0], %1;" :: "r"(is), "f"(f) : "memory"); }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      arrive(MB(4 + inst));
    }
  } else if (warp >= 12 && warp < 16) {
    // ---------------- epilogue WG ----------------
    const int et = threadIdx.x - 384;
    const CUtensorMap* mo = &maps.out[mdie];
    for (int t = 0; t < steps; ++t) {
      mwait(MB(15), t & 1);
      asm volatile("tcgen05.fence::after_thread_sync;");
      uint32_t r[32];
      if (!(A.flags & 8)) { LD32(r, cb + lane_off + 224); asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
      else for (int e = 0; e < 32; ++e) r[e] = e;
      asm volatile("tcgen05.fence::before_thread_sync;");
      arrive(MB(16));
      if (A.efill) { float f = mfill(A.efill, __uint_as_float(r[0])); r[1] = __float_as_uint(__uint_as_float(r[1]) + f); }
      if (!(A.flags & 2)) {
        if (et == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        nbar(3, 128);
#pragma unroll
        for (int e = 0; e < 32; ++e) {
          unsigned short h; asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(__uint_as_float(r[e])));
          asm volatile("st.shared.b16 [%0], %1;" :: "r"(osm + e * 256 + et * 2), "h"(h) : "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        nbar(3, 128);
        if (et == 0) {
          asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%1, %2, %3}], [%4];"
                       :: "l"((uint64_t)mo), "r"(0), "r"(head), "r"(tok0(t)), "r"(osm) : "memory");
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
      }
    }
    if (et == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tbase) : "memory");
}

int main(int argc, char** argv) {
  // usage: skel2c <mode 0|1> <pw 1|2> <npi 1..5> <steps> <flags> <iters> [pfill efill cfill]
  int mode = atoi(argv[1]), pw = atoi(argv[2]), npi = atoi(argv[3]), steps = atoi(argv[4]), flags = atoi(argv[5]);
  int iters = argc > 6 ? atoi(argv[6]) : 5;
  if (npi < 1 || npi > MAXI || (pw == 2 && npi > 4)) { printf("bad npi\n"); return 1; }
  CK(cudaSetDevice(0)); CK(cudaFree(0));
  int dev = 0, nsm, maxsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  CK(cudaDeviceGetAttribute(&maxsm, cudaDevAttrOversizedSharedMemoryPerBlock, dev));
  static Maps maps;
  size_t full = (size_t)T_TOK * 96 * HD * 2, half = full / 2;
  if (mode == 0) {
    void* p[5]; for (int i = 0; i < 5; ++i) { CK(cudaMalloc(&p[i], full)); CK(cudaMemset(p[i], 0, full)); }
    for (int d = 0; d < 2; ++d) { for (int i = 0; i < 4; ++i) mkmap(&maps.in[d][i], p[i], 96); mkmap(&maps.out[d], p[4], 96); }
  } else {
    for (int d = 0; d < 2; ++d)
      for (int i = 0; i < 5; ++i) {
        void* p = alloc_dom(half, d, dev); CK(cudaMemset(p, 0, half));
        if (i < 4) mkmap(&maps.in[d][i], p, 48); else mkmap(&maps.out[d], p, 48);
      }
  }
  size_t smem = (size_t)MAXI * IMG_B + NR * RAW_B + OUT_B + 128;
  CArgs A; A.steps = steps; A.mode = mode; A.flags = flags; A.npi = npi;
  A.pfill = argc > 7 ? atoi(argv[7]) : 0; A.efill = argc > 8 ? atoi(argv[8]) : 0; A.cfill = argc > 9 ? atoi(argv[9]) : 0;
  A.id_f16_n32 = mk(1, 1, 1, 32, 128); A.id_tf32_n128 = mk(1, 2, 2, 128, 128); A.id_tf32_n32 = mk(1, 2, 2, 32, 128);
  CK(cudaMalloc(&A.stamps, (size_t)nsm * 8 * 8));
  auto kern = pw == 1 ? skelc<1> : skelc<2>;
  CUfunction cf; CK(cudaGetFuncBySymbol((cudaFunction_t*)&cf, (const void*)kern));
  CU(cuFuncSetAttribute(cf, CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE, CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY));
  cudaLaunchConfig_t lc = {}; cudaLaunchAttribute la[1];
  la[0].id = cudaLaunchAttributeSharedMemoryMode; la[0].val.sharedMemoryMode = cudaSharedMemoryModeAllowOversizedSharedMemory;
  lc.gridDim = dim3(nsm); lc.blockDim = dim3(1024); lc.dynamicSmemBytes = smem; lc.attrs = la; lc.numAttrs = 1;
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  std::vector<float> ts;
  for (int it = 0; it < iters + 1; ++it) {
    CK(cudaEventRecord(e0));
    CK(cudaLaunchKernelEx(&lc, kern, maps, A));
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (it) ts.push_back(ms * 1000.f);
  }
  std::sort(ts.begin(), ts.end());
  std::vector<long long> st((size_t)nsm * 8);
  CK(cudaMemcpy(st.data(), A.stamps, st.size() * 8, cudaMemcpyDeviceToHost));
  double fin = 0, pass = 0, iss = 0;
  for (int b = 0; b < nsm; ++b) { fin += st[b * 8]; pass += st[b * 8 + 1]; iss += st[b * 8 + 2]; }
  double tot = (fin + pass + iss) / nsm / steps;
  printf("RESULT mode %d pw %d npi %d steps %d flags %d pfill %d efill %d cfill %d: kernel p50 %.1f us min %.1f | step clk waitFIN %.0f pass %.0f issue %.0f total %.0f | eff clk %.2f GHz | smem %zu/%d\n",
         mode, pw, npi, steps, flags, A.pfill, A.efill, A.cfill, ts[ts.size() / 2], ts[0], fin / nsm / steps, pass / nsm / steps,
         iss / nsm / steps, tot, tot * steps / (ts[ts.size() / 2] * 1e3), smem, maxsm);
  return 0;
}
