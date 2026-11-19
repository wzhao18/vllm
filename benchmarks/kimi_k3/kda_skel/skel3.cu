// BT64 dual-chain SKELETON of the persistent prefill KDA core (P-KDA2 M2/M3/M4): data movement + sync + every
// tcgen05 instruction of the BT64 chain step, no SIMT math (optional calibrated FFMA fill), 2 chains per SM.
//   FP32 state S (128 dv lanes x 128 dk cols) is the kind::tf32 A operand straight from TMEM: no bf16 image, so per
//   chain TMEM = S 0..127 | U 128..191 | OUT 192..255 and two chains fill 512 columns.
//   per chain step t (64 tokens):
//     prep WG (4 warps): LDG q,k,v,g [64 x 128] bf16 (64 KiB, no smem staging); WQ slot (W^T|Qd fp32 64K, V^T bf16 16K,
//        B' bf16 8K) after WQ_FREE(t-1) -> WQ_FULL(t); KM slot (Kd fp32 32K, Mqk fp32 16K) after KM_FREE(t-1) -> KM_FULL
//     issuer (chain warp 0): WQ_FULL, OFREE(t-1); MMA1 [U|OUT] = S x [W^T|Qd] (tf32 TS N128, 16 x K8);
//        MMA3v U += V^T B' (f16 SS N64, 4 x K16); commit UFIN, WQ_FREE
//     chain group (8 warps): UFIN; decay pass on S (tcgen05.ld 16 cols, scale, tcgen05.st), fence, bar
//     issuer: KM_FULL; MMA4 S += U x Kd (tf32 TS N128, 8 x K8); OUT += U x Mqk (tf32 TS N64, 8 x K8); commit OFIN, KM_FREE
//     epilogue WG: OFIN; tcgen05.ld OUT 64 cols; OFREE; bf16 st.shared; TMA store [64 x 128] bf16
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

constexpr int T_TOK = 16384, HD = 128, BT = 64, NH = 96;
constexpr int WQ_B = 64 * 1024, VT_B = 16 * 1024, BP_B = 8 * 1024, KD_B = 32 * 1024, MQ_B = 16 * 1024;
constexpr int SLOTW = WQ_B + VT_B + BP_B, SLOTK = KD_B + MQ_B, IMG_B = SLOTW + SLOTK;   // 136 KiB per chain
constexpr int OUT_B = BT * HD * 2;                                                     // 16 KiB
constexpr int SMEM = 2 * IMG_B + 2 * OUT_B + 1024;

struct Args { int nch, steps, flags, pfill, efill, descw; unsigned id_tf32_n128, id_tf32_n64, id_f16_n64;
              const uint8_t* in[4]; unsigned* ctr; long long* stamps; };

__device__ __forceinline__ uint64_t sdesc(uint32_t saddr, int w) {
  return (uint64_t)((saddr >> 4) & (w ? 0x7FFF : 0x3FFF)) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | ((uint64_t)1 << 46) | ((uint64_t)2 << 61);
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

__global__ void __launch_bounds__(1024, 1) skel(const __grid_constant__ CUtensorMap omap, Args A) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ uint32_t tmem_base;
  __shared__ int s_rank;
  __shared__ __align__(8) uint64_t mbar[2][16];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t s0 = ((uint32_t)__cvta_generic_to_shared(smem) + 1023) & ~1023u;
  if (threadIdx.x == 0) s_rank = atomicAdd(A.ctr, 1);
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"((uint32_t)__cvta_generic_to_shared(&tmem_base)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  // per chain: 0 WQ_FULL (128) | 1 WQ_FREE (commit) | 2 KM_FULL (128) | 3 KM_FREE (commit) | 4 UFIN | 5 OFIN | 6 OFREE (128)
  if (threadIdx.x < 2) {
    int c = threadIdx.x;
    minit((uint32_t)__cvta_generic_to_shared(&mbar[c][0]), 128);
    minit((uint32_t)__cvta_generic_to_shared(&mbar[c][1]), 1);
    minit((uint32_t)__cvta_generic_to_shared(&mbar[c][2]), 128);
    minit((uint32_t)__cvta_generic_to_shared(&mbar[c][3]), 1);
    minit((uint32_t)__cvta_generic_to_shared(&mbar[c][4]), 1);
    minit((uint32_t)__cvta_generic_to_shared(&mbar[c][5]), 1);
    minit((uint32_t)__cvta_generic_to_shared(&mbar[c][6]), 128);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t tbase = tmem_base;
  const int c = warp >> 4, w = warp & 15;
  if (c >= A.nch) goto done;
  {
    const int k = s_rank * A.nch + c;
    const int head = k % NH, seq = (k / NH) % 3;
    auto tok0 = [&](int t) { return seq * 5461 + (t * BT) % 5440; };
    const uint32_t img = s0 + c * IMG_B, osm = s0 + 2 * IMG_B + c * OUT_B;
    const uint32_t wq = img, vt = img + WQ_B, bp = vt + VT_B, kd = img + SLOTW, mq = kd + KD_B;
    auto MB = [&](int i) { return (uint32_t)__cvta_generic_to_shared(&mbar[c][i]); };
    const uint32_t cb = tbase + c * 256;          // S 0..127 | U 128..191 | OUT 192..255
    const uint32_t lane_off = (uint32_t)((warp & 3) * 32) << 16;
    const int steps = A.steps, fl = A.flags, dw = A.descw;
    if (w >= 8 && w < 12) asm volatile("setmaxnreg.dec.sync.aligned.u32 56;"); else if (w >= 12) asm volatile("setmaxnreg.inc.sync.aligned.u32 72;");
    if (w < 8) {
      // ---------------- chain group (issuer = warp 0) ----------------
      long long ph[3] = {0, 0, 0};
      const int part = w >> 2;
      if (w == 0 && !(fl & 8)) {                     // prologue: MMA1/MMA3v of step 0
        mwait(MB(0), 0);
        asm volatile("tcgen05.fence::after_thread_sync;");
        for (int kk = 0; kk < 16; ++kk) mma_ts_tf32(cb + 128, cb + kk * 8, sdesc(wq, dw) + kk * 2, A.id_tf32_n128, kk);
        for (int kk = 0; kk < 4; ++kk) mma_ss_f16(cb + 128, sdesc(vt, dw) + kk * 2, sdesc(bp, dw) + kk * 2, A.id_f16_n64, 1);
        commit_el(MB(4)); commit_el(MB(1));
      }
      for (int t = 0; t < steps; ++t) {
        long long c0 = clock64();
        if (!(fl & 8)) mwait(MB(4), t & 1);            // UFIN(t): MMA1(t) has read S, U(t) final
        else if (w == 0) mwait(MB(0), t & 1);
        asm volatile("tcgen05.fence::after_thread_sync;");
        long long c1 = clock64();
        if (!(fl & 4)) {
#pragma unroll 1
          for (int j = 0; j < 64; j += 16) {
            uint32_t r[16];
            const uint32_t col = part * 64 + j;
            LD16(r, cb + lane_off + col);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
            for (int e = 0; e < 16; ++e) r[e] = __float_as_uint(__uint_as_float(r[e]) * 0.99f);
            ST16(cb + lane_off + col, r);
          }
          asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        asm volatile("tcgen05.fence::before_thread_sync;");
        nbar(1 + c, 256);
        asm volatile("tcgen05.fence::after_thread_sync;");
        long long c2 = clock64();
        if (w == 0) {
          if (!(fl & 8)) {
            mwait(MB(2), t & 1);                         // KM_FULL(t)
            asm volatile("tcgen05.fence::after_thread_sync;");
            for (int kk = 0; kk < 8; ++kk) mma_ts_tf32(cb, cb + 128 + kk * 8, sdesc(kd, dw) + kk * 2, A.id_tf32_n128, 1);
            for (int kk = 0; kk < 8; ++kk) mma_ts_tf32(cb + 192, cb + 128 + kk * 8, sdesc(mq, dw) + kk * 2, A.id_tf32_n64, 1);
            commit_el(MB(5)); commit_el(MB(3));
            if (t + 1 < steps) {
              mwait(MB(0), (t + 1) & 1);                 // WQ_FULL(t+1)
              mwait(MB(6), t & 1);                       // OFREE(t): epilogue has read OUT(t)
              asm volatile("tcgen05.fence::after_thread_sync;");
              for (int kk = 0; kk < 16; ++kk) mma_ts_tf32(cb + 128, cb + kk * 8, sdesc(wq, dw) + kk * 2, A.id_tf32_n128, kk);
              for (int kk = 0; kk < 4; ++kk) mma_ss_f16(cb + 128, sdesc(vt, dw) + kk * 2, sdesc(bp, dw) + kk * 2, A.id_f16_n64, 1);
              commit_el(MB(4)); commit_el(MB(1));
            }
          } else if (lane == 0) {
            mwait(MB(2), t & 1);
            arrive(MB(5)); arrive(MB(3)); arrive(MB(1));
            if (t > 0 || true) mwait(MB(6), t & 1);
          }
          __syncwarp();
        }
        long long c3 = clock64();
        ph[0] += c1 - c0; ph[1] += c2 - c1; ph[2] += c3 - c2;
      }
      if (w == 0 && lane == 0) {
        long long* o = A.stamps + (blockIdx.x * 2 + c) * 4;
        o[0] = ph[0]; o[1] = ph[1]; o[2] = ph[2];
      }
    } else if (w < 12) {
      // ---------------- prep WG: LDG raw, operand images ----------------
      const int pt = threadIdx.x - (c * 512 + 256);
      uint32_t acc = 0;
      for (int t = 0; t < steps; ++t) {
        const int tk = tok0(t);
        uint4 v[8];
        if (!(fl & 64)) {
#pragma unroll 1
          for (int m = 0; m < 32; m += 8) {
#pragma unroll
            for (int u = 0; u < 8; ++u) {
              const int idx = pt + 128 * (m + u), i = idx >> 10, r = (idx >> 4) & 63, j = idx & 15;
              const uint8_t* g = A.in[i] + ((size_t)(tk + r) * NH + head) * 256 + j * 16;
              asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(g));
            }
#pragma unroll
            for (int u = 0; u < 8; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
          }
        } else {
#pragma unroll
          for (int u = 0; u < 8; ++u) v[u] = make_uint4(acc, u, t, pt);
        }
        if (A.pfill) acc += __float_as_uint(mfill(A.pfill, (float)acc));
        if (t > 0) mwait(MB(1), (t - 1) & 1);           // WQ_FREE(t-1)
        if (!(fl & 16)) {
#pragma unroll 4
          for (int i = 0; i < SLOTW / 16 / 128; ++i) {
            const uint4 x = v[i & 7];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wq + (pt + 128 * i) * 16), "r"(x.x), "r"(x.y), "r"(x.z), "r"(x.w ^ acc) : "memory");
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        arrive(MB(0));                                   // WQ_FULL(t)
        if (t > 0) mwait(MB(3), (t - 1) & 1);           // KM_FREE(t-1)
        if (!(fl & 16)) {
#pragma unroll 4
          for (int i = 0; i < SLOTK / 16 / 128; ++i) {
            const uint4 x = v[i & 7];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(kd + (pt + 128 * i) * 16), "r"(x.x), "r"(x.y), "r"(x.z), "r"(x.w) : "memory");
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        arrive(MB(2));                                   // KM_FULL(t)
      }
      if (acc == 0x12345678u) asm volatile("st.shared.u32 [%0], %1;" :: "r"(osm), "r"(acc) : "memory");
    } else {
      // ---------------- epilogue WG ----------------
      const int et = threadIdx.x - (c * 512 + 384);
      for (int t = 0; t < steps; ++t) {
        mwait(MB(5), t & 1);                             // OFIN(t)
        asm volatile("tcgen05.fence::after_thread_sync;");
        uint32_t p[32];
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          uint32_t r[32];
          if (!(fl & 8)) {
            LD32(r, cb + lane_off + 192 + hf * 32);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
          } else {
#pragma unroll
            for (int e = 0; e < 32; ++e) r[e] = e + t;
          }
#pragma unroll
          for (int e = 0; e < 16; ++e)
            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(p[hf * 16 + e]) : "f"(__uint_as_float(r[2 * e + 1])), "f"(__uint_as_float(r[2 * e])));
        }
        asm volatile("tcgen05.fence::before_thread_sync;");
        arrive(MB(6));                                   // OFREE(t)
        if (A.efill) { float f = mfill(A.efill, __uint_as_float(p[0])); p[1] ^= __float_as_uint(f); }
        if (!(fl & 2)) {
          if (et == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
          nbar(3 + c, 128);
#pragma unroll
          for (int e = 0; e < 32; ++e) {
            asm volatile("st.shared.b16 [%0], %1;" :: "r"(osm + (2 * e) * 256 + et * 2), "h"((unsigned short)(p[e] & 0xffff)) : "memory");
            asm volatile("st.shared.b16 [%0], %1;" :: "r"(osm + (2 * e + 1) * 256 + et * 2), "h"((unsigned short)(p[e] >> 16)) : "memory");
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          nbar(3 + c, 128);
          if (et == 0) {
            asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%1, %2, %3}], [%4];"
                         :: "l"((uint64_t)&omap), "r"(0), "r"(head), "r"(tok0(t)), "r"(osm) : "memory");
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

int main(int argc, char** argv) {
  // usage: skel3 <nch 1|2> <steps> <flags> [iters] [pfill] [efill] [descw]
  int nch = atoi(argv[1]), steps = atoi(argv[2]), flags = atoi(argv[3]);
  int iters = argc > 4 ? atoi(argv[4]) : 5;
  CK(cudaSetDevice(0)); CK(cudaFree(0));
  int nsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  size_t full = (size_t)T_TOK * NH * HD * 2;
  Args A = {};
  for (int i = 0; i < 4; ++i) { void* p; CK(cudaMalloc(&p, full)); CK(cudaMemset(p, 0, full)); A.in[i] = (const uint8_t*)p; }
  void* po; CK(cudaMalloc(&po, full));
  static CUtensorMap omap;
  cuuint64_t dims[3] = {HD, (cuuint64_t)NH, T_TOK}, str[2] = {HD * 2, (cuuint64_t)NH * HD * 2};
  cuuint32_t box[3] = {HD, 1, BT}, es[3] = {1, 1, 1};
  CU(cuTensorMapEncodeTiled(&omap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, po, dims, str, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  A.nch = nch; A.steps = steps; A.flags = flags;
  A.pfill = argc > 5 ? atoi(argv[5]) : 0; A.efill = argc > 6 ? atoi(argv[6]) : 0; A.descw = argc > 7 ? atoi(argv[7]) : 1;
  A.id_tf32_n128 = mk(1, 2, 2, 128, 128); A.id_tf32_n64 = mk(1, 2, 2, 64, 128); A.id_f16_n64 = mk(1, 1, 1, 64, 128);
  CK(cudaMalloc(&A.ctr, 8)); CK(cudaMalloc(&A.stamps, (size_t)nsm * 8 * 8));
  CUfunction cf; CK(cudaGetFuncBySymbol((cudaFunction_t*)&cf, (const void*)skel));
  CU(cuFuncSetAttribute(cf, CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE, CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY));
  cudaLaunchConfig_t lc = {}; cudaLaunchAttribute la[1];
  la[0].id = cudaLaunchAttributeSharedMemoryMode; la[0].val.sharedMemoryMode = cudaSharedMemoryModeAllowOversizedSharedMemory;
  lc.gridDim = dim3(nsm); lc.blockDim = dim3(1024); lc.dynamicSmemBytes = SMEM; lc.attrs = la; lc.numAttrs = 1;
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  std::vector<float> ts;
  for (int it = 0; it < iters + 1; ++it) {
    CK(cudaMemset(A.ctr, 0, 8));
    CK(cudaEventRecord(e0));
    CK(cudaLaunchKernelEx(&lc, skel, omap, A));
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (it) ts.push_back(ms * 1000.f);
  }
  std::sort(ts.begin(), ts.end());
  std::vector<long long> st((size_t)nsm * 8);
  CK(cudaMemcpy(st.data(), A.stamps, st.size() * 8, cudaMemcpyDeviceToHost));
  double a0 = 0, a1 = 0, a2 = 0; int n = 0;
  for (int b = 0; b < nsm; ++b) for (int cc = 0; cc < nch; ++cc) { long long* o = &st[(b * 2 + cc) * 4]; a0 += o[0]; a1 += o[1]; a2 += o[2]; ++n; }
  double bytes = (double)nsm * nch * steps * (4 * BT * HD * 2 + OUT_B);
  printf("RESULT3 nch %d steps %d flags %d pfill %d efill %d descw %d smem %d: kernel p50 %.1f us min %.1f | per step clk: waitUFIN %.0f pass %.0f issue %.0f total %.0f | HBM %.2f GB %.2f TB/s\n",
         nch, steps, flags, A.pfill, A.efill, A.descw, SMEM, ts[ts.size() / 2], ts[0], a0 / n / steps, a1 / n / steps, a2 / n / steps,
         (a0 + a1 + a2) / n / steps, bytes / 1e9, bytes / (ts[ts.size() / 2] * 1e-6) / 1e12);
  return 0;
}
