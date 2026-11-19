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

struct Args { int nch, steps, mode, flags; unsigned id_f16_n32, id_tf32_n128, id_tf32_n32, id_f16_n64; unsigned* ctr; long long* stamps; };
// flags: 1 = no HBM loads (prep reuses resident raw), 2 = no out stores, 4 = no state pass, 8 = no MMAs (chain = pass only),
//        16 = no prep image copy (ideal prep: handoff barriers only), 32 = fused N64 MMA1+OUT-inter leg, FIN right after MMA4

template <int PW>
__global__ void __launch_bounds__(1024, 1) skel(const __grid_constant__ Maps maps, Args A) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ uint32_t tmem_base;
  __shared__ int s_rank;
  __shared__ __align__(8) uint64_t mbar[2][16];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t s0 = ((uint32_t)__cvta_generic_to_shared(smem) + 127) & ~127u;
  unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  const int die = (smid >> 1) & 1;
  if (threadIdx.x == 0) s_rank = atomicAdd(&A.ctr[die], 1);
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"((uint32_t)__cvta_generic_to_shared(&tmem_base)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  // barriers per chain: 0,1 RAW_FULL[s] | 2,3 IMG_FULL[s] (128) | 4,5 IMG_FREE[s] (commit) | 6 FIN | 7 OFIN | 8 OFREE (128)
  if (threadIdx.x < 2) {
    int c = threadIdx.x;
    for (int s = 0; s < 2; ++s) {
      minit((uint32_t)__cvta_generic_to_shared(&mbar[c][0 + s]), 1);
      minit((uint32_t)__cvta_generic_to_shared(&mbar[c][2 + s]), 128);
      minit((uint32_t)__cvta_generic_to_shared(&mbar[c][4 + s]), 1);
    }
    minit((uint32_t)__cvta_generic_to_shared(&mbar[c][6]), 1);
    minit((uint32_t)__cvta_generic_to_shared(&mbar[c][7]), 1);
    minit((uint32_t)__cvta_generic_to_shared(&mbar[c][8]), 128);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t tbase = tmem_base;
  // warp roles: chain c uses warps [c*16, c*16+16): 0..4*PW-1 chain group, 8..11 prep, 12..15 epilogue
  const int c = warp >> 4, w = warp & 15;
  if (c >= A.nch) goto done;
  {
    const int k = s_rank * A.nch + c;
    const int hl = k % 48, seq = (k / 48) % 3;
    const int head = A.mode == 1 ? hl : die * 48 + hl;
    const int mdie = A.mode == 1 ? die : 0;
    auto tok0 = [&](int t) { return seq * 5461 + (t * BT) % 5440; };
    // UMMA smem descriptors carry a 14-bit address: every MMA operand (the images) sits in the low 256 KiB;
    // TMA-only buffers (raw ring, out tile) go above.
    const uint32_t img = s0 + c * NST * IMG_B, raw = s0 + 2 * NST * IMG_B + c * NST * RAW_B;
    const uint32_t osm = s0 + 2 * NST * (IMG_B + RAW_B) + c * OUT_B;
    auto MB = [&](int i) { return (uint32_t)__cvta_generic_to_shared(&mbar[c][i]); };
    const uint32_t cb = tbase + c * 256;          // S 0..127 | INP 128..191 | U 192..223 | OUT 224..255
    const uint32_t lane_off = (uint32_t)((warp & 3) * 32) << 16;
    const int steps = A.steps;
    if (PW == 2) { if (w < 8) asm volatile("setmaxnreg.inc.sync.aligned.u32 72;"); else if (w < 12) asm volatile("setmaxnreg.dec.sync.aligned.u32 48;"); }
    if (w < 4 * PW) {
      // ---------------- chain group ----------------
      long long ph[4] = {0, 0, 0, 0};
      const int part = w >> 2;            // PW=2: part 0 -> S cols 0..63, part 1 -> 64..127
      const int ncol = 128 / PW;
      for (int t = 0; t < steps; ++t) {
        const int s = t & 1; const uint32_t sp = (t >> 1) & 1;
        long long c0 = clock64();
        if (t > 0) mwait(MB(6), (t - 1) & 1);                 // FIN(t-1): MMA4 done, S current
        asm volatile("tcgen05.fence::after_thread_sync;");
        const uint32_t im = img + s * IMG_B;
        if (w == 0 && !(A.flags & 8)) {
          mwait(MB(2 + s), sp);                                   // IMG_FULL[s]
          asm volatile("tcgen05.fence::after_thread_sync;");
          // MMA3v: u^T = V^T B'  (SS f16, M128 N32 K32), overlaps the state pass
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
        asm volatile("tcgen05.fence::before_thread_sync;");
        nbar(1 + c, 128 * PW);
        asm volatile("tcgen05.fence::after_thread_sync;");
        long long c2 = clock64();
        if (w == 0 && !(A.flags & 8) && (A.flags & 32)) {
          // fused: [U | OUT] += INP [W^T | Qd] (one N64 TS f16 leg, W^T and Qd adjacent), MMA4, FIN; then OUT intra
          uint64_t dWQ = sdesc(im), dKd = sdesc(im + 16384);
          if (t > 0) mwait(MB(8), (t - 1) & 1);                   // OFREE(t-1): epilogue read + zeroed OUT
          for (int kk = 0; kk < 8; ++kk) mma_ts_f16(cb + 192, cb + 128 + kk * 8, dWQ + kk * 2, A.id_f16_n64, 1);
          for (int kk = 0; kk < 4; ++kk) mma_ts_tf32(cb, cb + 192 + kk * 8, dKd + kk * 2, A.id_tf32_n128, 1);
          commit_el(MB(6));
          for (int kk = 0; kk < 4; ++kk) mma_ts_tf32(cb + 224, cb + 192 + kk * 8, dKd + kk * 2, A.id_tf32_n32, 1);
          commit_el(MB(7));
          commit_el(MB(4 + s));
        } else if (w == 0 && !(A.flags & 8)) {
          uint64_t dW = sdesc(im), dKd = sdesc(im + 8192), dQ = sdesc(im + 24576);
          // MMA1: U += INP W^T (TS f16, A=INP 64 cols bf16 = K128)
          for (int kk = 0; kk < 8; ++kk) mma_ts_f16(cb + 192, cb + 128 + kk * 8, dW + kk * 2, A.id_f16_n32, 1);
          // MMA4: S += vnew^T Kd (TS tf32, A = U fp32 K32, N128)
          for (int kk = 0; kk < 4; ++kk) mma_ts_tf32(cb, cb + 192 + kk * 8, dKd + kk * 2, A.id_tf32_n128, 1);
          // OUT inter leg reads INP: it must finish before the next pass overwrites INP -> FIN covers it
          if (t > 0) mwait(MB(8), (t - 1) & 1);                   // OFREE(t-1)
          for (int kk = 0; kk < 8; ++kk) mma_ts_f16(cb + 224, cb + 128 + kk * 8, dQ + kk * 2, A.id_f16_n32, kk);
          commit_el(MB(6));
          for (int kk = 0; kk < 4; ++kk) mma_ts_tf32(cb + 224, cb + 192 + kk * 8, dKd + kk * 2, A.id_tf32_n32, 1);
          commit_el(MB(7));
          commit_el(MB(4 + s));
        } else if (w == 0) {
          mwait(MB(2 + s), sp);
          if (lane == 0) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(MB(6)) : "memory");
                           if (t > 0) mwait(MB(8), (t - 1) & 1);
                           arrive(MB(7)); arrive(MB(4 + s)); }
          __syncwarp();
        }
        long long c3 = clock64();
        ph[0] += c1 - c0; ph[1] += c2 - c1; ph[2] += c3 - c2;
      }
      if (w == 0 && lane == 0) {
        long long* o = A.stamps + (blockIdx.x * 2 + c) * 8;
        o[0] = ph[0]; o[1] = ph[1]; o[2] = ph[2]; o[3] = die;
      }
    } else if (w >= 8 && w < 12) {
      // ---------------- prep WG (+ loader thread) ----------------
      const int pt = threadIdx.x - (c * 512 + 256);
      const CUtensorMap* mp = &maps.in[mdie][0];
      auto load = [&](int t) {
        const int s = t & 1; const uint32_t mb = MB(s);
        if (A.flags & 1) { if (t < 2) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(RAW_B) : "memory");
                             for (int i = 0; i < 4; ++i) asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                               :: "r"(raw + s * RAW_B + i * 8192), "l"((uint64_t)(mp + i)), "r"(0), "r"(head), "r"(tok0(t)), "r"(mb) : "memory"); }
                           else arrive(mb);
                           return; }
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(RAW_B) : "memory");
        for (int i = 0; i < 4; ++i)
          asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                       :: "r"(raw + s * RAW_B + i * 8192), "l"((uint64_t)(mp + i)), "r"(0), "r"(head), "r"(tok0(t)), "r"(mb) : "memory");
      };
      if (pt == 0) load(0);
      for (int t = 0; t < steps; ++t) {
        const int s = t & 1; const uint32_t sp = (t >> 1) & 1;
        if (pt == 0 && t + 1 < steps) load(t + 1);       // stage (t+1)&1 was released by the prep bar of t-1
        mwait(MB(s), sp);                                 // RAW_FULL[s]
        mwait(MB(4 + s), sp ^ 1);                         // IMG_FREE[s]
        const uint32_t rs = raw + s * RAW_B, is = img + s * IMG_B;
#pragma unroll 3
        for (int i = 0; i < ((A.flags & 16) ? 0 : IMG_B / 16 / 128); ++i) {
          const int idx = pt + 128 * i;
          uint32_t a, b, cc, d;
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(cc), "=r"(d) : "r"(rs + (idx & 2047) * 16));
          asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(is + idx * 16), "r"(a), "r"(b), "r"(cc), "r"(d) : "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        arrive(MB(2 + s));                                // IMG_FULL[s]
        nbar(5 + c, 128);                                 // all prep threads done reading raw[s]
      }
    } else if (w >= 12) {
      // ---------------- epilogue WG ----------------
      const int et = threadIdx.x - (c * 512 + 384);
      const CUtensorMap* mo = &maps.out[mdie];
      for (int t = 0; t < steps; ++t) {
        mwait(MB(7), t & 1);                              // OFIN(t)
        asm volatile("tcgen05.fence::after_thread_sync;");
        uint32_t r[32];
        if (!(A.flags & 8)) { LD32(r, cb + lane_off + 224); asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
        else for (int e = 0; e < 32; ++e) r[e] = e;
        if ((A.flags & 40) == 32) {                       // fused leg accumulates: zero OUT for the next step
          uint32_t z[16];
#pragma unroll
          for (int e = 0; e < 16; ++e) z[e] = 0;
          ST16(cb + lane_off + 224, z); ST16(cb + lane_off + 240, z);
          asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        asm volatile("tcgen05.fence::before_thread_sync;");
        arrive(MB(8));                                    // OFREE(t)
        if (!(A.flags & 2)) {
          if (et == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
          nbar(3 + c, 128);
#pragma unroll
          for (int e = 0; e < 32; ++e) {
            unsigned short h; asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(__uint_as_float(r[e])));
            asm volatile("st.shared.b16 [%0], %1;" :: "r"(osm + e * 256 + et * 2), "h"(h) : "memory");
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          nbar(3 + c, 128);
          if (et == 0) {
            asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%1, %2, %3}], [%4];"
                         :: "l"((uint64_t)mo), "r"(0), "r"(head), "r"(tok0(t)), "r"(osm) : "memory");
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
          }
        }
      }
      if (et == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
  }
done:
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tbase) : "memory");
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

int main(int argc, char** argv) {
  // usage: skel2 <mode 0|1> <nch 1|2> <pw 1|2> <steps> <flags> [iters]
  int mode = atoi(argv[1]), nch = atoi(argv[2]), pw = atoi(argv[3]), steps = atoi(argv[4]), flags = atoi(argv[5]);
  int iters = argc > 6 ? atoi(argv[6]) : 5;
  CK(cudaSetDevice(0)); CK(cudaFree(0));
  int dev = 0, nsm, maxsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  CK(cudaDeviceGetAttribute(&maxsm, cudaDevAttrOversizedSharedMemoryPerBlock, dev));
  static Maps maps;
  std::vector<void*> bufs;
  size_t full = (size_t)T_TOK * 96 * HD * 2, half = full / 2;
  if (mode == 0) {
    void* p[5]; for (int i = 0; i < 5; ++i) { CK(cudaMalloc(&p[i], full)); CK(cudaMemset(p[i], 0, full)); }
    for (int d = 0; d < 2; ++d) { for (int i = 0; i < 4; ++i) mkmap(&maps.in[d][i], p[i], 96); mkmap(&maps.out[d], p[4], 96); }
  } else {
    for (int d = 0; d < 2; ++d) {
      for (int i = 0; i < 5; ++i) {
        void* p = alloc_dom(half, d, dev); CK(cudaMemset(p, 0, half));
        int ord = -9; cuPointerGetAttribute(&ord, CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL, (CUdeviceptr)p);
        if (i == 0) printf("die %d alloc ordinal %d\n", d, ord);
        if (i < 4) mkmap(&maps.in[d][i], p, 48); else mkmap(&maps.out[d], p, 48);
      }
    }
  }
  size_t smem = (size_t)2 * CH_SMEM + 128;
  printf("SMs %d maxsmem %d need %zu\n", nsm, maxsm, smem);
  Args A; A.nch = nch; A.steps = steps; A.mode = mode; A.flags = flags;
  A.id_f16_n32 = mk(1, 1, 1, 32, 128); A.id_tf32_n128 = mk(1, 2, 2, 128, 128); A.id_tf32_n32 = mk(1, 2, 2, 32, 128); A.id_f16_n64 = mk(1, 1, 1, 64, 128);
  CK(cudaMalloc(&A.ctr, 8)); CK(cudaMalloc(&A.stamps, (size_t)nsm * 16 * 8));
  auto kern = pw == 1 ? skel<1> : skel<2>;
  CUfunction cf; CK(cudaGetFuncBySymbol((cudaFunction_t*)&cf, (const void*)kern));
  CU(cuFuncSetAttribute(cf, CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE, CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY));
  cudaLaunchConfig_t lc = {}; cudaLaunchAttribute la[1];
  la[0].id = cudaLaunchAttributeSharedMemoryMode; la[0].val.sharedMemoryMode = cudaSharedMemoryModeAllowOversizedSharedMemory;
  lc.gridDim = dim3(nsm); lc.blockDim = dim3(1024); lc.dynamicSmemBytes = smem; lc.attrs = la; lc.numAttrs = 1;
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  std::vector<float> ts;
  for (int it = 0; it < iters + 1; ++it) {
    CK(cudaMemset(A.ctr, 0, 8));
    CK(cudaEventRecord(e0));
    CK(cudaLaunchKernelEx(&lc, kern, maps, A));
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (it) ts.push_back(ms * 1000.f);
  }
  std::sort(ts.begin(), ts.end());
  std::vector<long long> st((size_t)nsm * 16);
  CK(cudaMemcpy(st.data(), A.stamps, st.size() * 8, cudaMemcpyDeviceToHost));
  double fin = 0, pass = 0, iss = 0; int n = 0, nd[2] = {0, 0};
  for (int b = 0; b < nsm; ++b) for (int c = 0; c < nch; ++c) { long long* o = &st[(b * 2 + c) * 8]; fin += o[0]; pass += o[1]; iss += o[2]; ++n; if (c == 0) nd[o[3] & 1]++; }
  double tot = (fin + pass + iss) / n / steps;
  double bytes = (double)nsm * nch * steps * (RAW_B + OUT_B);
  printf("RESULT mode %d nch %d pw %d steps %d flags %d: kernel p50 %.1f us min %.1f | per step clk: waitFIN %.0f pass %.0f issue %.0f total %.0f | HBM %.2f GB %.2f TB/s | dies %d/%d\n",
         mode, nch, pw, steps, flags, ts[ts.size() / 2], ts[0], fin / n / steps, pass / n / steps, iss / n / steps, tot,
         bytes / 1e9, bytes / (ts[ts.size() / 2] * 1e-6) / 1e12, nd[0], nd[1]);
  return 0;
}
