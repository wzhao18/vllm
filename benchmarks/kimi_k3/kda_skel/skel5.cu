// M3'' skeleton (P-KDA2; skel4 + flag 256 U double buffer, flag 512 packed out stores, flag-8 deadlock fix,
//   per-step PASSED/UFIN barriers so no waiter can miss a phase). Base text of skel4:
// M2' skeleton (P-KDA2): ONE chain per SM, BT64 steps, bf16 state image INP (stock operand precision), two MMA issuers:
//   issuer A (recurrence): MMA1u U = INP x W^T (f16 TS N64, 8 x K16) | MMA3v U += V^T B' (f16 SS N64, 4) -> UFIN |
//                          MMA4 S += U x Kd (tf32 TS N128, 8 x K8) -> FIN
//   issuer B (output, off the recurrence): MMA1o OUT[b] = INP x Qd^T (f16 TS N64, 8) | UFIN | OUT[b] += U x Mqk (tf32 N64, 8) -> OFIN[b]
//   chain group (8 warps): FIN(t-1); pass: ld S -> st INP[t&1] (bf16) + st S*decay; PASSED
//   prep WG (4 warps): raw (TMA, 3 x 32-token half stages) -> operand images (2 stages, 104 KB); epilogue WG: OUT[b] -> bf16 -> TMA
//   loader warp: TMA q,k,v,g. TMEM: S 0..127 | INP0 128 | INP1 192 | U 256 | OUT0 320 | OUT1 384.
// flags: 1024 stock-precision U legs: chain group converts U fp32 -> bf16 in place (stock casts v_new to bf16 before
//   h += k^T v_new and o += A v_new), MMA4 / OUT intra become kind::f16 K16 (4 instr each, half the tf32 tensor time)
// flags: 256 U double buffer (U1 at TMEM col 448; issuer A drops its OFIN(t-1) wait) | 512 packed (b32) out stores
// flags: 2 no out stores | 4 no pass | 8 no MMAs | 16 ideal prep (no image writes) | 64 no HBM loads
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CU(x) do { CUresult e = (x); if (e != CUDA_SUCCESS) { const char* s; cuGetErrorString(e, &s); printf("CU %s @%d: %s\n", #x, __LINE__, s); exit(1); } } while (0)

constexpr int T_TOK = 16384, HD = 128, BT = 64, NH = 96, NTHR = 576;
constexpr int WT_B = 16384, QD_B = 16384, VT_B = 16384, BP_B = 8192, KD_B = 32768, MQ_B = 16384;
constexpr int IMG_B = WT_B + QD_B + VT_B + BP_B + KD_B + MQ_B;     // 104 KiB
constexpr int RAWH_B = 4 * 32 * HD * 2;                            // 32 KiB: q,k,v,g x 32 tokens
constexpr int NRAW = 3, OUT_B = BT * HD * 2;
constexpr int SMEM = 2 * IMG_B + NRAW * RAWH_B + OUT_B + 1024;

struct Maps { CUtensorMap in[4]; CUtensorMap out; };
struct Args { int steps, flags, pfill, efill; unsigned id_f16_n64, id_f16_n128, id_tf32_n128, id_tf32_n64; unsigned* ctr; long long* stamps; };

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
#define LD16(r, addr) asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];" \
  : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),"=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]) : "r"(addr) : "memory")
#define ST16(addr, r) asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" \
  :: "r"(addr), "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),"r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]) : "memory")
#define ST8(addr, r) asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" \
  :: "r"(addr), "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]) : "memory")

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

__global__ void __launch_bounds__(NTHR, 1) skel(const __grid_constant__ Maps maps, Args A) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ uint32_t tmem_base;
  __shared__ int s_rank;
  // 0-2 RAW_FULL | 3-5 RAW_FREE (128) | 6,7 IMG_FULL (128) | 8,9 IMG_FREE (2 commits) | 10 PASSED (256) | 11 UFIN | 12 FIN |
  // 13,14 OFIN | 15,16 OFREE (128)
  __shared__ __align__(8) uint64_t mbar[21];  // + 19,20 UCVT (flag 1024)  // + 17 UFIN1, 18 PASSED1
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t s0 = ((uint32_t)__cvta_generic_to_shared(smem) + 1023) & ~1023u;
  if (threadIdx.x == 0) s_rank = atomicAdd(A.ctr, 1);
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"((uint32_t)__cvta_generic_to_shared(&tmem_base)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  auto MB = [&](int i) { return (uint32_t)__cvta_generic_to_shared(&mbar[i]); };
  if (threadIdx.x == 0) {
    for (int i = 0; i < 3; ++i) { minit(MB(i), 1); minit(MB(3 + i), 128); }
    for (int i = 0; i < 2; ++i) { minit(MB(6 + i), 128); minit(MB(8 + i), 2); minit(MB(13 + i), 1); minit(MB(15 + i), 128); }
    minit(MB(10), 256); minit(MB(11), 1); minit(MB(12), 1); minit(MB(17), 1); minit(MB(18), 256); minit(MB(19), 256); minit(MB(20), 256);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t tb = tmem_base;
  const int head = s_rank % NH, seq = (s_rank / NH) % 3;
  auto tok0 = [&](int t) { return seq * 5461 + (t * BT) % 5440; };
  const uint32_t raw = s0 + 2 * IMG_B, osm = raw + NRAW * RAWH_B;
  const uint32_t lane_off = (uint32_t)((warp & 3) * 32) << 16;
  const uint32_t S = tb;
  const int steps = A.steps, fl = A.flags;
  const bool udb = fl & 256, ub16 = fl & 1024;
  auto UB = [&](int t) { return tb + ((udb && (t & 1)) ? 448u : 256u); };
  auto PASSED = [&](int t) { return (t & 1) ? MB(18) : MB(10); };
  auto UFIN = [&](int t) { return (t & 1) ? MB(17) : MB(11); };
  auto UCVT = [&](int t) { return (t & 1) ? MB(20) : MB(19); };
  // U fp32 (64 cols) -> bf16 packed in place (cols 0..31); 8 chain warps: quadrant warp&3, half part = warp>>2
  auto ucvt = [&](uint32_t U, int t) {
    if (!(fl & 8)) {
      mwait(UFIN(t), (t >> 1) & 1);
      asm volatile("tcgen05.fence::after_thread_sync;");
      uint32_t r[32], p[16];
      LD32(r, U + lane_off + (warp >> 2) * 32);
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
      for (int e = 0; e < 16; ++e)
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(p[e]) : "f"(__uint_as_float(r[2 * e + 1])), "f"(__uint_as_float(r[2 * e])));
      nbar(4, 256);                                                    // all fp32 reads done before packed writes
      ST16(U + lane_off + (warp >> 2) * 16, p);
      asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
      asm volatile("tcgen05.fence::before_thread_sync;");
    }
    arrive(UCVT(t));
  };
  if (warp < 8) {
    // ---------------- chain group; warp 0 = issuer A ----------------
    long long ph[3] = {0, 0, 0}, qs[3] = {0, 0, 0};
    const int part = warp >> 2;
    for (int t = 0; t < steps; ++t) {
      const int b = t & 1, s = t & 1;
      const uint32_t INP = tb + 128 + b * 64, img = s0 + s * IMG_B;
      long long c0 = clock64();
      if (t > 0) mwait(MB(12), (t - 1) & 1);                       // FIN(t-1): S final
      if (t > 1) mwait(MB(13 + b), ((t - 2) >> 1) & 1);            // OFIN(t-2): INP[b] free
      asm volatile("tcgen05.fence::after_thread_sync;");
      long long c1 = clock64();
      if (!(fl & 4)) {
#pragma unroll 1
        for (int j = 0; j < 64; j += 16) {
          uint32_t r[16], p[8];
          const uint32_t col = part * 64 + j;
          LD16(r, S + lane_off + col);
          asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
          for (int e = 0; e < 8; ++e)
            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(p[e]) : "f"(__uint_as_float(r[2 * e + 1])), "f"(__uint_as_float(r[2 * e])));
          ST8(INP + lane_off + col / 2, p);
#pragma unroll
          for (int e = 0; e < 16; ++e) r[e] = __float_as_uint(__uint_as_float(r[e]) * 0.99f);
          ST16(S + lane_off + col, r);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
      }
      asm volatile("tcgen05.fence::before_thread_sync;");
      arrive(PASSED(t));                                              // PASSED(t)
      long long c2 = clock64();
      if (warp == 0) {
        const uint32_t U = UB(t);
        long long q0 = clock64();
        mwait(PASSED(t), (t >> 1) & 1);
        long long q1 = clock64();
        mwait(MB(6 + s), (t >> 1) & 1);                              // IMG_FULL[s]
        long long q2 = clock64();
        qs[0] += q1 - q0; qs[1] += q2 - q1;
        // single U: OFIN(t-1) = issuer B done reading U(t-1). U double buffer: U[t&1] was last read by B at t-2,
        // and OFIN(t-2) is already waited by the whole chain group above.
        if (t > 0 && !udb) mwait(MB(13 + (b ^ 1)), ((t - 1) >> 1) & 1);
        qs[2] += clock64() - q2;
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (!(fl & 8)) {
          const uint64_t dW = sdesc(img), dV = sdesc(img + WT_B + QD_B), dB = sdesc(img + WT_B + QD_B + VT_B);
          const uint64_t dK = sdesc(img + WT_B + QD_B + VT_B + BP_B);
#pragma unroll
          for (int kk = 0; kk < 8; ++kk) mma_ts_f16(U, INP + kk * 8, dW + kk * 2, A.id_f16_n64, kk);
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) mma_ss_f16(U, dV + kk * 2, dB + kk * 2, A.id_f16_n64, 1);
          commit_el(UFIN(t));                                         // UFIN(t)
          if (ub16) {
            ucvt(U, t);
            mwait(UCVT(t), (t >> 1) & 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) mma_ts_f16(S, U + kk * 8, dK + kk * 2, A.id_f16_n128, 1);
          } else {
#pragma unroll
          for (int kk = 0; kk < 8; ++kk) mma_ts_tf32(S, U + kk * 8, dK + kk * 2, A.id_tf32_n128, 1);
          }
          commit_el(MB(12));                                          // FIN(t)
          commit_el(MB(8 + s));                                       // IMG_FREE[s] (1 of 2)
        } else { if (lane == 0) { arrive(UFIN(t)); arrive(MB(12)); arrive(MB(8 + s)); } if (ub16) ucvt(UB(t), t); }
        __syncwarp();
      } else if (ub16) ucvt(UB(t), t);
      long long c3 = clock64();
      ph[0] += c1 - c0; ph[1] += c2 - c1; ph[2] += c3 - c2;
    }
    if (warp == 0 && lane == 0) { long long* o = A.stamps + blockIdx.x * 8; o[0] = ph[0]; o[1] = ph[1]; o[2] = ph[2]; o[3] = qs[0]; o[4] = qs[1]; o[5] = qs[2]; }
  } else if (warp < 12) {
    // ---------------- prep WG ----------------
    const int pt = threadIdx.x - 256;
    uint32_t acc = 0; long long pw[2] = {0, 0};
    for (int t = 0; t < steps; ++t) {
      const int s = t & 1;
      const uint32_t img = s0 + s * IMG_B;
      const int j0 = 2 * t, j1 = 2 * t + 1;
      long long p0 = clock64();
      mwait(MB(j0 % 3), (j0 / 3) & 1); mwait(MB(j1 % 3), (j1 / 3) & 1);   // RAW_FULL
      long long p1 = clock64();
      if (t > 1) mwait(MB(8 + s), ((t - 2) >> 1) & 1);                    // IMG_FREE[s] from step t-2
      pw[0] += p1 - p0; pw[1] += clock64() - p1;
      if (A.pfill) acc += __float_as_uint(mfill(A.pfill, (float)(acc + pt)));
      if (!(fl & 16)) {
        const uint32_t r0 = raw + (j0 % 3) * RAWH_B, r1 = raw + (j1 % 3) * RAWH_B;
#pragma unroll 4
        for (int i = 0; i < IMG_B / 16 / 128; ++i) {
          const int idx = pt + 128 * i;
          uint32_t a, bb, cc, d;
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(bb), "=r"(cc), "=r"(d) : "r"(((i & 1) ? r1 : r0) + (idx & 2047) * 16));
          asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(img + idx * 16), "r"(a), "r"(bb ^ acc), "r"(cc), "r"(d) : "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      }
      arrive(MB(6 + s));                                                  // IMG_FULL[s]
      arrive(MB(3 + j0 % 3)); arrive(MB(3 + j1 % 3));                     // RAW_FREE
    }
    if (pt == 0) { long long* o = A.stamps + blockIdx.x * 8; o[6] = pw[0]; o[7] = pw[1]; }
  } else if (warp < 16) {
    // ---------------- epilogue WG ----------------
    const int et = threadIdx.x - 384;
    for (int t = 0; t < steps; ++t) {
      const int b = t & 1;
      const uint32_t OUT = tb + 320 + b * 64;
      mwait(MB(13 + b), (t >> 1) & 1);                                    // OFIN(t)
      asm volatile("tcgen05.fence::after_thread_sync;");
      uint32_t p[32];
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        uint32_t r[32];
        if (!(fl & 8)) { LD32(r, OUT + lane_off + hf * 32); asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
        else {
#pragma unroll
          for (int e = 0; e < 32; ++e) r[e] = e + t;
        }
#pragma unroll
        for (int e = 0; e < 16; ++e)
          asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(p[hf * 16 + e]) : "f"(__uint_as_float(r[2 * e + 1])), "f"(__uint_as_float(r[2 * e])));
      }
      asm volatile("tcgen05.fence::before_thread_sync;");
      arrive(MB(15 + b));                                                 // OFREE[b]
      if (A.efill) { float f = mfill(A.efill, __uint_as_float(p[0])); p[1] ^= __float_as_uint(f); }
      if (!(fl & 2)) {
        if (et == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        nbar(3, 128);
        if (fl & 512) {
          // pair lanes (d, d+1): even lane keeps token 2e as (d, d+1), odd lane keeps token 2e+1 as (d, d+1): one b32 store each
          const bool odd = et & 1;
          const uint32_t base = osm + (et & ~1) * 2 + (odd ? 256 : 0);
#pragma unroll
          for (int e = 0; e < 32; ++e) {
            const uint32_t give = odd ? (p[e] & 0xffffu) : (p[e] >> 16);
            const uint32_t got = __shfl_xor_sync(0xffffffffu, give, 1);
            const uint32_t w = odd ? __byte_perm(got, p[e], 0x7610) : __byte_perm(p[e], got, 0x5410);
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(base + (2 * e) * 256), "r"(w) : "memory");
          }
        } else {
#pragma unroll
        for (int e = 0; e < 32; ++e) {
          asm volatile("st.shared.b16 [%0], %1;" :: "r"(osm + (2 * e) * 256 + et * 2), "h"((unsigned short)(p[e] & 0xffff)) : "memory");
          asm volatile("st.shared.b16 [%0], %1;" :: "r"(osm + (2 * e + 1) * 256 + et * 2), "h"((unsigned short)(p[e] >> 16)) : "memory");
        }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        nbar(3, 128);
        if (et == 0) {
          asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%1, %2, %3}], [%4];"
                       :: "l"((uint64_t)&maps.out), "r"(0), "r"(head), "r"(tok0(t)), "r"(osm) : "memory");
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
      }
    }
    if (et == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  } else if (warp == 16) {
    // ---------------- issuer B: output legs ----------------
    for (int t = 0; t < steps; ++t) {
      const int b = t & 1, s = t & 1;
      const uint32_t INP = tb + 128 + b * 64, OUT = tb + 320 + b * 64, img = s0 + s * IMG_B;
      const uint32_t U = UB(t);
      mwait(PASSED(t), (t >> 1) & 1);                                     // PASSED(t): INP[b] written
      mwait(MB(6 + s), (t >> 1) & 1);                                     // IMG_FULL[s]
      if (t > 1) mwait(MB(15 + b), ((t - 2) >> 1) & 1);                   // OFREE[b] (step t-2)
      asm volatile("tcgen05.fence::after_thread_sync;");
      if (!(fl & 8)) {
        const uint64_t dQ = sdesc(img + WT_B), dM = sdesc(img + WT_B + QD_B + VT_B + BP_B + KD_B);
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) mma_ts_f16(OUT, INP + kk * 8, dQ + kk * 2, A.id_f16_n64, kk);
        if (ub16) mwait(UCVT(t), (t >> 1) & 1); else mwait(UFIN(t), (t >> 1) & 1);   // U(t) ready
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (ub16) {
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) mma_ts_f16(OUT, U + kk * 8, dM + kk * 2, A.id_f16_n64, 1);
        } else {
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) mma_ts_tf32(OUT, U + kk * 8, dM + kk * 2, A.id_tf32_n64, 1);
        }
        commit_el(MB(13 + b));                                            // OFIN[b]
        commit_el(MB(8 + s));                                             // IMG_FREE[s] (2 of 2)
      } else {
        if (ub16) mwait(UCVT(t), (t >> 1) & 1); else mwait(UFIN(t), (t >> 1) & 1);
        if (lane == 0) { arrive(MB(13 + b)); arrive(MB(8 + s)); }
        __syncwarp();
      }
    }
  } else {
    // ---------------- loader warp: TMA q,k,v,g in 32-token halves ----------------
    // whole warp loops (a lone lane looping while its warp sits in bar.sync deadlocked v1); lane 0 issues
    for (int j = 0; j < 2 * steps; ++j) {
      const int r = j % 3;
      if (j >= 3) mwait(MB(3 + r), ((j - 3) / 3) & 1);                   // RAW_FREE[r]
      if (lane == 0) {
        if ((fl & 64) && j >= 3) arrive(MB(r));
        else {
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(MB(r)), "r"(RAWH_B) : "memory");
          const int tk = tok0(j >> 1) + (j & 1) * 32;
          for (int i = 0; i < 4; ++i)
            asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                         :: "r"(raw + r * RAWH_B + i * 8192), "l"((uint64_t)&maps.in[i]), "r"(0), "r"(head), "r"(tk), "r"(MB(r)) : "memory");
        }
      }
      __syncwarp();
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tb) : "memory");
}

static unsigned mk(int dfmt, int afmt, int bfmt, int N, int M) {
  return ((unsigned)dfmt << 4) | ((unsigned)afmt << 7) | ((unsigned)bfmt << 10) | ((unsigned)(N >> 3) << 17) | ((unsigned)(M >> 4) << 24);
}
static void mkmap(CUtensorMap* m, void* p, int box_t) {
  cuuint64_t dims[3] = {HD, (cuuint64_t)NH, T_TOK}, str[2] = {HD * 2, (cuuint64_t)NH * HD * 2};
  cuuint32_t box[3] = {HD, 1, (cuuint32_t)box_t}, es[3] = {1, 1, 1};
  CU(cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, p, dims, str, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

int main(int argc, char** argv) {
  // usage: skel5 <steps> <flags> [iters] [pfill] [efill]
  int steps = atoi(argv[1]), flags = atoi(argv[2]), iters = argc > 3 ? atoi(argv[3]) : 5;
  CK(cudaSetDevice(0)); CK(cudaFree(0));
  int nsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  size_t full = (size_t)T_TOK * NH * HD * 2;
  static Maps maps;
  for (int i = 0; i < 5; ++i) { void* p; CK(cudaMalloc(&p, full)); CK(cudaMemset(p, 0, full)); if (i < 4) mkmap(&maps.in[i], p, 32); else mkmap(&maps.out, p, BT); }
  Args A = {}; A.steps = steps; A.flags = flags;
  A.pfill = argc > 4 ? atoi(argv[4]) : 0; A.efill = argc > 5 ? atoi(argv[5]) : 0;
  A.id_f16_n64 = mk(1, 1, 1, 64, 128); A.id_f16_n128 = mk(1, 1, 1, 128, 128); A.id_tf32_n128 = mk(1, 2, 2, 128, 128); A.id_tf32_n64 = mk(1, 2, 2, 64, 128);
  CK(cudaMalloc(&A.ctr, 8)); CK(cudaMalloc(&A.stamps, (size_t)nsm * 8 * 8));
  CUfunction cf; CK(cudaGetFuncBySymbol((cudaFunction_t*)&cf, (const void*)skel));
  CU(cuFuncSetAttribute(cf, CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE, CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY));
  cudaLaunchConfig_t lc = {}; cudaLaunchAttribute la[1];
  la[0].id = cudaLaunchAttributeSharedMemoryMode; la[0].val.sharedMemoryMode = cudaSharedMemoryModeAllowOversizedSharedMemory;
  lc.gridDim = dim3(nsm); lc.blockDim = dim3(NTHR); lc.dynamicSmemBytes = SMEM; lc.attrs = la; lc.numAttrs = 1;
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  std::vector<float> ts;
  for (int it = 0; it < iters + 1; ++it) {
    CK(cudaMemset(A.ctr, 0, 8));
    CK(cudaEventRecord(e0));
    CK(cudaLaunchKernelEx(&lc, skel, maps, A));
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (it) ts.push_back(ms * 1000.f);
  }
  std::sort(ts.begin(), ts.end());
  std::vector<long long> st((size_t)nsm * 8);
  CK(cudaMemcpy(st.data(), A.stamps, st.size() * 8, cudaMemcpyDeviceToHost));
  double a0 = 0, a1 = 0, a2 = 0;
  double w[5] = {0, 0, 0, 0, 0};
  for (int b = 0; b < nsm; ++b) { a0 += st[b * 8]; a1 += st[b * 8 + 1]; a2 += st[b * 8 + 2]; for (int k = 0; k < 5; ++k) w[k] += st[b * 8 + 3 + k]; }
  for (int k = 0; k < 5; ++k) w[k] /= (double)nsm * steps;
  a0 /= (double)nsm * steps; a1 /= (double)nsm * steps; a2 /= (double)nsm * steps;
  double bytes = (double)nsm * steps * (4 * BT * HD * 2 + OUT_B), tot = a0 + a1 + a2;
  printf("RESULT5 steps %d flags %d pfill %d efill %d smem %d: kernel p50 %.1f us min %.1f | per step clk: waitFIN %.0f pass %.0f issueA %.0f total %.0f | eff clk %.2f GHz | HBM %.2f GB %.2f TB/s\n",
         steps, flags, A.pfill, A.efill, SMEM, ts[ts.size() / 2], ts[0], a0, a1, a2, tot, tot * steps / (ts[ts.size() / 2] * 1e3),
         bytes / 1e9, bytes / (ts[ts.size() / 2] * 1e-6) / 1e12);
  printf("STAMPS5 flags %d: issuerA waits PASSED %.0f IMG_FULL %.0f OFIN %.0f | prep waits RAW_FULL %.0f IMG_FREE %.0f clk/step\n", flags, w[0], w[1], w[2], w[3], w[4]);
  return 0;
}
