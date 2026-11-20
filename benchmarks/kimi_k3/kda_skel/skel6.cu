// P-KDA2 M5 skeleton skel6.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CU(x) do { CUresult e = (x); if (e != CUDA_SUCCESS) { const char* s; cuGetErrorString(e, &s); printf("CU %s @%d: %s\n", #x, __LINE__, s); exit(1); } } while (0)

constexpr int T_TOK = 16384, HD = 128, BT = 64, NH = 96, NTHR = 672;
// per-chain image (bf16 operands, stock precision): W^T 16K | Qd 16K | V^T 16K | B' 8K | Kd 16K | Mqk 8K = 80 KiB
constexpr int O_WT = 0, O_QD = 16384, O_VT = 32768, O_BP = 49152, O_KD = 57344, O_MQ = 73728, IMG_B = 81920;
constexpr int INP_B = 32768;                                        // bf16 S image (128 d_v x 128 k), SW-xor rows
constexpr int RAWQ_B = 4 * 16 * HD * 2;                             // 16 KiB: q,k,v,g x 16 tokens
constexpr int NRAW = 5, OUT_B = BT * HD * 2;
constexpr int SMEM = 2 * IMG_B + 2 * INP_B + NRAW * RAWQ_B + OUT_B + 1024;   // 321 KiB
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


// M5 skeleton (P-KDA2): TWO independent chains per SM (two heads), BT64 steps, one MMA issuer per chain, latency of
// one chain hidden by the other. TMEM per chain c: S 128 | U 64 | OUT 64 at c*256 (512 cols). INP (bf16 S image)
// lives in smem (MMA1u/MMA1o are SS). bf16 U legs (stock: v_new cast to bf16 before h += k^T v and o += A v).
// warps 0-7 chain groups (c = warp>>2, issuer = first warp), 8-15 prep groups, 16-19 shared epilogue, 20 loader.
// flags: 2048 loader issues from 4 lanes (one tensor each)
// flags: 2 no out stores | 4 no pass | 8 no MMAs | 16 ideal prep (no image writes) | 64 no HBM loads
__global__ void __launch_bounds__(NTHR, 1) skel(const __grid_constant__ Maps maps, Args A) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ uint32_t tmem_base;
  __shared__ int s_rank;
  // 0-4 RAW_FULL | 5-9 RAW_FREE(128) | per chain c at 10 + 8c: +0 IMG_FULL(128) +1 IMG_FREE(1) +2 PASSED(128) +3 UFIN(1)
  //   +4 UCVT(128) +5 FIN(1) +6 OFIN(1) +7 OFREE(128)
  __shared__ __align__(8) uint64_t mbar[26];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t s0 = ((uint32_t)__cvta_generic_to_shared(smem) + 1023) & ~1023u;
  if (threadIdx.x == 0) s_rank = atomicAdd(A.ctr, 1);
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"((uint32_t)__cvta_generic_to_shared(&tmem_base)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  auto MB = [&](int i) { return (uint32_t)__cvta_generic_to_shared(&mbar[i]); };
  if (threadIdx.x == 0) {
    for (int i = 0; i < NRAW; ++i) { minit(MB(i), 1); minit(MB(5 + i), 128); }
    for (int c = 0; c < 2; ++c) {
      const int b = 10 + 8 * c;
      minit(MB(b + 0), 128); minit(MB(b + 1), 1); minit(MB(b + 2), 128); minit(MB(b + 3), 1);
      minit(MB(b + 4), 128); minit(MB(b + 5), 1); minit(MB(b + 6), 1); minit(MB(b + 7), 128);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t tb = tmem_base;
  const int steps = A.steps, fl = A.flags;
  const int seq = (s_rank * 2 / NH) % 3;
  auto headc = [&](int c) { return (s_rank * 2 + c) % NH; };
  auto tok0 = [&](int t) { return seq * 5461 + (t * BT) % 5440; };
  const uint32_t img0 = s0, inp0 = s0 + 2 * IMG_B, raw = inp0 + 2 * INP_B, osm = raw + NRAW * RAWQ_B;
  const uint32_t lane_off = (uint32_t)((warp & 3) * 32) << 16;
  auto B_ = [&](int c, int k) { return MB(10 + 8 * c + k); };
  if (warp < 8) {
    // ---------------- chain group c (4 warps, one per TMEM lane quadrant); first warp = MMA issuer ----------------
    const int c = warp >> 2, row = (warp & 3) * 32 + lane;
    const uint32_t S = tb + c * 256, U = S + 128, OUT = S + 192, INP = inp0 + c * INP_B, img = img0 + c * IMG_B;
    long long ph[3] = {0, 0, 0};
    for (int t = 0; t < steps; ++t) {
      const uint32_t par = t & 1;
      long long c0 = clock64();
      if (t > 0) mwait(B_(c, 5), par ^ 1);                              // FIN(t-1): S final, INP/U free
      asm volatile("tcgen05.fence::after_thread_sync;");
      long long c1 = clock64();
      if (!(fl & 4)) {
#pragma unroll 1
        for (int j = 0; j < 128; j += 16) {
          uint32_t r[16], p[8];
          LD16(r, S + lane_off + j);
          asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
          for (int e = 0; e < 8; ++e)
            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(p[e]) : "f"(__uint_as_float(r[2 * e + 1])), "f"(__uint_as_float(r[2 * e])));
          const int cb = j >> 3;                                        // 16 B chunk index (8 bf16)
          asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(INP + row * 256 + ((cb ^ (row & 7)) * 16)), "r"(p[0]), "r"(p[1]), "r"(p[2]), "r"(p[3]) : "memory");
          asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(INP + row * 256 + (((cb + 1) ^ (row & 7)) * 16)), "r"(p[4]), "r"(p[5]), "r"(p[6]), "r"(p[7]) : "memory");
#pragma unroll
          for (int e = 0; e < 16; ++e) r[e] = __float_as_uint(__uint_as_float(r[e]) * 0.99f);
          ST16(S + lane_off + j, r);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      }
      asm volatile("tcgen05.fence::before_thread_sync;");
      arrive(B_(c, 2));                                                 // PASSED(t)
      long long c2 = clock64();
      const bool iss = (warp & 3) == 0;
      if (iss) {
        mwait(B_(c, 2), par);
        mwait(B_(c, 0), par);                                           // IMG_FULL(t)
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (!(fl & 8)) {
          const uint64_t dI = sdesc(INP), dW = sdesc(img + O_WT), dQ = sdesc(img + O_QD), dV = sdesc(img + O_VT), dB = sdesc(img + O_BP);
#pragma unroll
          for (int kk = 0; kk < 8; ++kk) mma_ss_f16(U, dI + kk * 2, dW + kk * 2, A.id_f16_n64, kk);
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) mma_ss_f16(U, dV + kk * 2, dB + kk * 2, A.id_f16_n64, 1);
          commit_el(B_(c, 3));                                          // UFIN(t)
          if (t > 0) mwait(B_(c, 7), par ^ 1);                          // OFREE(t-1)
          asm volatile("tcgen05.fence::after_thread_sync;");
#pragma unroll
          for (int kk = 0; kk < 8; ++kk) mma_ss_f16(OUT, dI + kk * 2, dQ + kk * 2, A.id_f16_n64, kk);
        } else {
          if (lane == 0) arrive(B_(c, 3));
          if (t > 0) mwait(B_(c, 7), par ^ 1);
        }
        __syncwarp();
      }
      // U fp32 -> bf16 in place (each warp owns its 32 lanes: no cross-warp hazard)
      if (!(fl & 8)) {
        mwait(B_(c, 3), par);
        asm volatile("tcgen05.fence::after_thread_sync;");
        uint32_t r[32], p[16];
        LD32(r, U + lane_off);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
        for (int e = 0; e < 16; ++e)
          asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(p[e]) : "f"(__uint_as_float(r[2 * e + 1])), "f"(__uint_as_float(r[2 * e])));
        LD32(r, U + lane_off + 32);                                     // cols 32..63 (cols 0..31 already in registers)
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        ST16(U + lane_off, p);                                          // packed cols 0..15
#pragma unroll
        for (int e = 0; e < 16; ++e)
          asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(p[e]) : "f"(__uint_as_float(r[2 * e + 1])), "f"(__uint_as_float(r[2 * e])));
        ST16(U + lane_off + 16, p);                                     // packed cols 16..31
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::before_thread_sync;");
      }
      arrive(B_(c, 4));                                                 // UCVT(t)
      if (iss) {
        mwait(B_(c, 4), par);
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (!(fl & 8)) {
          const uint64_t dK = sdesc(img + O_KD), dM = sdesc(img + O_MQ);
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) mma_ts_f16(S, U + kk * 8, dK + kk * 2, A.id_f16_n128, 1);
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) mma_ts_f16(OUT, U + kk * 8, dM + kk * 2, A.id_f16_n64, 1);
          commit_el(B_(c, 5)); commit_el(B_(c, 6)); commit_el(B_(c, 1));   // FIN, OFIN, IMG_FREE
        } else if (lane == 0) { arrive(B_(c, 5)); arrive(B_(c, 6)); arrive(B_(c, 1)); }
        __syncwarp();
      }
      long long c3 = clock64();
      ph[0] += c1 - c0; ph[1] += c2 - c1; ph[2] += c3 - c2;
    }
    if (warp == 0 && lane == 0) { long long* o = A.stamps + blockIdx.x * 4; o[0] = ph[0]; o[1] = ph[1]; o[2] = ph[2]; }
  } else if (warp < 16) {
    // ---------------- prep group c: raw (16-token slots, shared ring) -> image (single stage per chain) ----------------
    const int c = (warp - 8) >> 2, pt = threadIdx.x - 256 - c * 128;
    const uint32_t img = img0 + c * IMG_B;
    uint32_t acc = 0;
    for (int t = 0; t < steps; ++t) {
      if (t > 0) mwait(B_(c, 1), (t - 1) & 1);                          // IMG_FREE(t-1)
      if (A.pfill) acc += __float_as_uint(mfill(A.pfill, (float)(acc + pt)));
      for (int qq = 0; qq < 4; ++qq) {
        const int j = (t * 2 + c) * 4 + qq, r = j % NRAW;
        mwait(MB(r), (j / NRAW) & 1);                                   // RAW_FULL
        if (!(fl & 16)) {
          const uint32_t rs = raw + r * RAWQ_B;
#pragma unroll 2
          for (int i = 0; i < IMG_B / 4 / 16 / 128; ++i) {              // 10 x 16 B per thread per slot
            const int idx = pt + 128 * i;
            uint32_t a, bb, cc, d;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(bb), "=r"(cc), "=r"(d) : "r"(rs + (idx & 1023) * 16));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(img + (qq * (IMG_B / 4)) + idx * 16), "r"(a), "r"(bb ^ acc), "r"(cc), "r"(d) : "memory");
          }
        }
        arrive(MB(5 + r));                                              // RAW_FREE
      }
      if (!(fl & 16)) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      arrive(B_(c, 0));                                                 // IMG_FULL(t)
    }
  } else if (warp < 20) {
    // ---------------- shared epilogue: OUT(c) -> bf16 -> TMA store, alternating chains ----------------
    const int et = threadIdx.x - 512;
    for (int t = 0; t < steps; ++t) for (int c = 0; c < 2; ++c) {
      const uint32_t OUT = tb + c * 256 + 192;
      mwait(B_(c, 6), t & 1);                                           // OFIN(t)
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
      arrive(B_(c, 7));                                                 // OFREE(t)
      if (A.efill) { float f = mfill(A.efill, __uint_as_float(p[0])); p[1] ^= __float_as_uint(f); }
      if (!(fl & 2)) {
        if (et == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        nbar(3, 128);
        const bool odd = et & 1;
        const uint32_t base = osm + (et & ~1) * 2 + (odd ? 256 : 0);
#pragma unroll
        for (int e = 0; e < 32; ++e) {
          const uint32_t give = odd ? (p[e] & 0xffffu) : (p[e] >> 16);
          const uint32_t got = __shfl_xor_sync(0xffffffffu, give, 1);
          const uint32_t w = odd ? __byte_perm(got, p[e], 0x7610) : __byte_perm(p[e], got, 0x5410);
          asm volatile("st.shared.b32 [%0], %1;" :: "r"(base + (2 * e) * 256), "r"(w) : "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        nbar(3, 128);
        if (et == 0) {
          asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%1, %2, %3}], [%4];"
                       :: "l"((uint64_t)&maps.out), "r"(0), "r"(headc(c)), "r"(tok0(t)), "r"(osm) : "memory");
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
      }
    }
    if (et == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  } else {
    // ---------------- loader warp: q,k,v,g 16-token slots, chains interleaved per step ----------------
    for (int j = 0; j < steps * 8; ++j) {
      const int r = j % NRAW, t = j >> 3, c = (j >> 2) & 1, qq = j & 3;
      if (j >= NRAW) mwait(MB(5 + r), ((j - NRAW) / NRAW) & 1);         // RAW_FREE
      if (fl & 2048) {
        // 4 issuing lanes (one tensor each): hw51b/04-tma: one issuing thread caps a ring at ~0.4 us per op
        if ((fl & 64) && j >= NRAW) { if (lane == 0) arrive(MB(r)); }
        else {
          if (lane == 0) asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(MB(r)), "r"(RAWQ_B) : "memory");
          __syncwarp();
          if (lane < 4) {
            const int tk = tok0(t) + qq * 16;
            asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                         :: "r"(raw + r * RAWQ_B + lane * 4096), "l"((uint64_t)&maps.in[lane]), "r"(0), "r"(headc(c)), "r"(tk), "r"(MB(r)) : "memory");
          }
        }
      } else
      if (lane == 0) {
        if ((fl & 64) && j >= NRAW) arrive(MB(r));
        else {
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(MB(r)), "r"(RAWQ_B) : "memory");
          const int tk = tok0(t) + qq * 16;
          for (int i = 0; i < 4; ++i)
            asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                         :: "r"(raw + r * RAWQ_B + i * 4096), "l"((uint64_t)&maps.in[i]), "r"(0), "r"(headc(c)), "r"(tk), "r"(MB(r)) : "memory");
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
  // usage: skel6 <steps per chain> <flags> [iters] [pfill] [efill]
  int steps = atoi(argv[1]), flags = atoi(argv[2]), iters = argc > 3 ? atoi(argv[3]) : 5;
  CK(cudaSetDevice(0)); CK(cudaFree(0));
  int nsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  size_t full = (size_t)T_TOK * NH * HD * 2;
  static Maps maps;
  for (int i = 0; i < 5; ++i) { void* p; CK(cudaMalloc(&p, full)); CK(cudaMemset(p, 0, full)); if (i < 4) mkmap(&maps.in[i], p, 16); else mkmap(&maps.out, p, BT); }
  Args A = {}; A.steps = steps; A.flags = flags;
  A.pfill = argc > 4 ? atoi(argv[4]) : 0; A.efill = argc > 5 ? atoi(argv[5]) : 0;
  A.id_f16_n64 = mk(1, 1, 1, 64, 128); A.id_f16_n128 = mk(1, 1, 1, 128, 128); A.id_tf32_n128 = mk(1, 2, 2, 128, 128); A.id_tf32_n64 = mk(1, 2, 2, 64, 128);
  CK(cudaMalloc(&A.ctr, 8)); CK(cudaMalloc(&A.stamps, (size_t)nsm * 4 * 8));
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
  std::vector<long long> st((size_t)nsm * 4);
  CK(cudaMemcpy(st.data(), A.stamps, st.size() * 8, cudaMemcpyDeviceToHost));
  double a0 = 0, a1 = 0, a2 = 0;
  for (int b = 0; b < nsm; ++b) { a0 += st[b * 4]; a1 += st[b * 4 + 1]; a2 += st[b * 4 + 2]; }
  a0 /= (double)nsm * steps; a1 /= (double)nsm * steps; a2 /= (double)nsm * steps;
  double bytes = (double)nsm * 2 * steps * (4 * BT * HD * 2 + OUT_B), tot = a0 + a1 + a2;
  printf("RESULT6 2ch steps/chain %d flags %d pfill %d efill %d smem %d: kernel p50 %.1f us min %.1f | per step clk: waitFIN %.0f pass %.0f issueA %.0f total %.0f | eff clk %.2f GHz | HBM %.2f GB %.2f TB/s\n",
         steps, flags, A.pfill, A.efill, SMEM, ts[ts.size() / 2], ts[0], a0, a1, a2, tot, tot * steps / (ts[ts.size() / 2] * 1e3),
         bytes / 1e9, bytes / (ts[ts.size() / 2] * 1e-6) / 1e12);
  return 0;
}
