// SPDX-License-Identifier: Apache-2.0
// EXPERIMENT ONLY (never for merge): MM-SOL A0 = TMEM layout gate (prefill/designs/megamoe-sol.md sec 4, "A0").
// One CTA pair (cta_group::2, exclusive 576-column TMEM) runs two routed FC1-like tiles of one unit with the skel's
// own core (mmsol_core.cuh): pieces a = N208 (double-buffered accumulators at columns 0 / 208) and b = N96 (column
// 416, single-buffered: tile 1's b MMAs wait for tile 0's b drain), one single-buffered SF image at columns 512-575
// (SFA 16 + SFB(a) 32 + SFB(b) 16), SF copied per K256 stage with tcgen05.cp 32x128b.warpx4, operands by cta_group::2
// TMA (SW128). Data are exact (E2M1 values, UE4M3 scales in {1, 2}): every partial sum is exactly representable in FP32,
// so the MMA result must equal the float64 reference BITWISE whatever the hardware summation order is.
// Build: nvcc -O3 -std=c++17 -arch=sm_107a mmsol_a0.cu -lcuda -o a0 ; run: ./a0 [stages=14] [ns=4]
#include "mmsol_core.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

using namespace mm;
constexpr int SLOT_A = 16384, SLOT_SFA = 2048, SLOT_B = 152 * 128, SLOT_SFB = 12 * 512;
constexpr int SLOT = SLOT_A + SLOT_SFA + SLOT_B + SLOT_SFB;  // 44032 = 43 KiB, 1024-aligned
static_assert(SLOT % 1024 == 0, "slot alignment");

struct A0Args {
  CUtensorMap tmW, tmX, tmSFA, tmSFB;
  float* out;  // [2 tiles][2 pieces][2 cta][128][208]
  int S, NS, neg;  // neg = 1: negative control (SFB(a) atoms swapped) must FAIL
};

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(192, 1) a0_kernel(const __grid_constant__ A0Args g) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t full[8], empty[8], accfull[2], bempty;
  __shared__ uint32_t tmem_base;
  uint8_t* ring = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  const uint32_t cr = cta_rank();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) {
    for (int i = 0; i < g.NS; ++i) { bar_init(sa(&full[i]), 1); bar_init(sa(&empty[i]), 1); }
    bar_init(sa(&accfull[0]), 1); bar_init(sa(&accfull[1]), 1);
    bar_init(sa(&bempty), 8);  // 4 epilogue warps x 2 CTAs
    bar_fence_init();
  }
  if (warp == 1) tmem_alloc576(sa(&tmem_base));
  tc_fence_before();
  cluster_sync();
  tc_fence_after();
  const uint32_t tb = tmem_base;
  const int S = g.S, NS = g.NS;
  if (warp == 0) {  // TMA issuer (both CTAs); leader expects both CTAs' bytes
    if (lane == 0) {
      prefetch_tmap(&g.tmW); prefetch_tmap(&g.tmX); prefetch_tmap(&g.tmSFA); prefetch_tmap(&g.tmSFB);
      int it = 0;
      for (int t = 0; t < 2; ++t)
        for (int s = 0; s < S; ++s, ++it) {
          const int slot = it % NS, round = it / NS;
          if (round > 0) bar_wait(sa(&empty[slot]), (round - 1) & 1);
          const uint32_t lb = mapa(sa(&full[slot]), 0);
          if (cr == 0) bar_expect(sa(&full[slot]), 2 * SLOT);
          uint8_t* base = ring + (size_t)slot * SLOT;
          tma2d_cg2(sa(base), &g.tmW, s * 128, t * 256 + cr * 128, lb);
          tma2d_cg2(sa(base + SLOT_A), &g.tmSFA, 0, ((t * 2 + cr) * S + s) * 4, lb);
          tma2d_cg2(sa(base + SLOT_A + SLOT_SFA), &g.tmX, s * 128, cr * 152, lb);
          tma2d_cg2(sa(base + SLOT_A + SLOT_SFA + SLOT_B), &g.tmSFB, 0, s * 12, lb);
        }
    }
  } else if (warp == 1) {  // MMA issuer: leader CTA only
    if (cr == 0) {
      const uint32_t ida = idesc_nvf4(NA), idb = idesc_nvf4(NB);
      int it = 0;
      for (int t = 0; t < 2; ++t) {
        const uint32_t acc_a = tb + (t ? COL_A1 : COL_A0), acc_b = tb + COL_B;
        for (int s = 0; s < S; ++s, ++it) {
          const int slot = it % NS, round = it / NS;
          bar_wait(sa(&full[slot]), round & 1);
          if (t == 1 && s == 0) bar_wait(sa(&bempty), 0);  // b turnaround: tile 0's b drained by both CTAs' epilogues
          tc_fence_after();
          uint8_t* base = ring + (size_t)slot * SLOT;
          const uint32_t pa = sa(base), psfa = sa(base + SLOT_A), pb = sa(base + SLOT_A + SLOT_SFA);
          const uint32_t psfb = pb + SLOT_B;
          for (int j = 0; j < 4; ++j) sf_cp(tb + COL_SFA + 4 * j, desc_sf(psfa + 512 * j));
          for (int j = 0; j < 8; ++j) sf_cp(tb + COL_SFBA + 4 * (g.neg ? (j ^ 1) : j), desc_sf(psfb + 512 * j));
          for (int j = 0; j < 4; ++j) sf_cp(tb + COL_SFBB + 4 * j, desc_sf(psfb + 4096 + 512 * j));
          const uint64_t da = desc_sw128(pa), dba = desc_sw128(pb), dbb = desc_sw128(pb + 104 * 128);
          for (int i = 0; i < 2; ++i) {
            const uint32_t acc = (s > 0 || i > 0);
            mma_nvf4(acc_a, da + 4 * i, dba + 4 * i, ida, tb + COL_SFA + 8 * i, tb + COL_SFBA + 16 * i, acc);
            mma_nvf4(acc_b, da + 4 * i, dbb + 4 * i, idb, tb + COL_SFA + 8 * i, tb + COL_SFBB + 8 * i, acc);
          }
          commit_mc(sa(&empty[slot]), 3);
        }
        commit_mc(sa(&accfull[t]), 3);
      }
    }
  } else {  // epilogue warps 2..5: lane quadrant q = warp % 4
    const int q = warp % 4;
    for (int t = 0; t < 2; ++t) {
      bar_wait(sa(&accfull[t]), 0);
      tc_fence_after();
      for (int p = 0; p < 2; ++p) {
        const int N = p ? NB : NA, col = p ? COL_B : (t ? COL_A1 : COL_A0);
        float* o = g.out + ((((size_t)t * 2 + p) * 2 + cr) * 128 + q * 32 + lane) * 208;
        for (int c0 = 0; c0 < N; c0 += 16) {
          uint32_t r[16];
          MM_LD16(tb + ((uint32_t)(q * 32) << 16) + col + c0, r);
          tmem_ld_wait();
          for (int k = 0; k < 16; ++k) o[c0 + k] = __uint_as_float(r[k]);
        }
      }
      tc_fence_before();
      __syncwarp();
      if (t == 0 && lane == 0) bar_arrive_cluster(mapa(sa(&bempty), 0));
    }
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  if (warp == 1) { tc_fence_after(); tmem_dealloc576(tb); }
}

// ---------------- host ----------------
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_; cuGetErrorString(r_, &s_); \
  printf("CU error %s at %s:%d\n", s_, __FILE__, __LINE__); exit(1); } } while (0)
#define RK(x) do { cudaError_t r_ = (x); if (r_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", \
  cudaGetErrorString(r_), __FILE__, __LINE__); exit(1); } } while (0)

static CUtensorMap tmap(void* p, CUtensorMapDataType dt, int esz, uint64_t d0, uint64_t d1, uint32_t b0, uint32_t b1,
                        CUtensorMapSwizzle sw) {
  CUtensorMap m; cuuint64_t dims[2] = {d0, d1}; cuuint64_t strides[1] = {d0 * esz};
  cuuint32_t box[2] = {b0, b1}; cuuint32_t es[2] = {1, 1};
  CK(cuTensorMapEncodeTiled(&m, dt, 2, p, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}
static const float E2M1[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -.5f, -1, -1.5f, -2, -3, -4, -6};

int main(int argc, char** argv) {
  const int S = argc > 1 ? atoi(argv[1]) : 14, NS = argc > 2 ? atoi(argv[2]) : 4, NEG = argc > 3 ? atoi(argv[3]) : 0;
  const int K = S * 256, KB = K / 16;
  srand(1234);
  // logical data: W [512 rows][K] (2 tiles x 256), X [304 tokens][K] (a: 0..207, b: 208..303)
  std::vector<uint8_t> wn((size_t)512 * K), xn((size_t)304 * K), wsf((size_t)512 * KB), xsf((size_t)304 * KB);
  for (auto& v : wn) v = rand() & 15;
  for (auto& v : xn) v = rand() & 15;
  for (auto& v : wsf) v = (rand() & 1) ? 0x40 : 0x38;  // UE4M3 2.0 / 1.0
  for (auto& v : xsf) v = (rand() & 1) ? 0x40 : 0x38;
  auto sfv = [](uint8_t b) { return b == 0x40 ? 2.0 : 1.0; };
  // global images
  std::vector<uint8_t> gW((size_t)512 * K / 2), gX((size_t)304 * K / 2);
  for (int r = 0; r < 512; ++r) for (int k = 0; k < K; k += 2) gW[(size_t)r * K / 2 + k / 2] = wn[(size_t)r * K + k] | (wn[(size_t)r * K + k + 1] << 4);
  auto xrow = [](int c, int j) { return j < 104 ? c * 104 + j : 208 + c * 48 + (j - 104); };  // slot row -> token
  for (int c = 0; c < 2; ++c) for (int j = 0; j < 152; ++j) {
    const int n = xrow(c, j);
    for (int k = 0; k < K; k += 2) gX[(size_t)(c * 152 + j) * K / 2 + k / 2] = xn[(size_t)n * K + k] | (xn[(size_t)n * K + k + 1] << 4);
  }
  // SFA chunks: [t][c][s][j=0..3][512]: byte (r%32)*16 + (r/32)*4 + kk = sf(row t*256+c*128+r, kblock s*16+j*4+kk)
  std::vector<uint8_t> gSFA((size_t)2 * 2 * S * 4 * 512, 0);
  for (int t = 0; t < 2; ++t) for (int c = 0; c < 2; ++c) for (int s = 0; s < S; ++s) for (int j = 0; j < 4; ++j)
    for (int r = 0; r < 128; ++r) for (int kk = 0; kk < 4; ++kk)
      gSFA[((((size_t)t * 2 + c) * S + s) * 4 + j) * 512 + (r % 32) * 16 + (r / 32) * 4 + kk] = wsf[(size_t)(t * 256 + c * 128 + r) * KB + s * 16 + j * 4 + kk];
  // SFB chunks per stage: a: q=0..7 -> (i = q/4, (atom, kc) = [(0,0),(1,0),(0,1),(1,1)][q%4] with kc += 2i); b: q=0..3 -> kc = q
  std::vector<uint8_t> gSFB((size_t)S * 12 * 512, 0);
  for (int s = 0; s < S; ++s) {
    for (int q = 0; q < 8; ++q) {
      const int i = q / 4, atom = (q % 4) & 1, kc = ((q % 4) >> 1) + 2 * i;
      for (int m = 0; m < 128; ++m) { const int n = atom * 128 + m; if (n >= 208) continue;
        for (int kk = 0; kk < 4; ++kk) gSFB[((size_t)s * 12 + q) * 512 + (m % 32) * 16 + (m / 32) * 4 + kk] = xsf[(size_t)n * KB + s * 16 + kc * 4 + kk]; }
    }
    for (int q = 0; q < 4; ++q)
      for (int m = 0; m < 96; ++m) for (int kk = 0; kk < 4; ++kk)
        gSFB[((size_t)s * 12 + 8 + q) * 512 + (m % 32) * 16 + (m / 32) * 4 + kk] = xsf[(size_t)(208 + m) * KB + s * 16 + q * 4 + kk];
  }
  uint8_t *dW, *dX, *dSFA, *dSFB; float* dout;
  RK(cudaMalloc(&dW, gW.size())); RK(cudaMalloc(&dX, gX.size())); RK(cudaMalloc(&dSFA, gSFA.size())); RK(cudaMalloc(&dSFB, gSFB.size()));
  const size_t nout = (size_t)2 * 2 * 2 * 128 * 208; RK(cudaMalloc(&dout, nout * 4)); RK(cudaMemset(dout, 0xFF, nout * 4));
  RK(cudaMemcpy(dW, gW.data(), gW.size(), cudaMemcpyHostToDevice)); RK(cudaMemcpy(dX, gX.data(), gX.size(), cudaMemcpyHostToDevice));
  RK(cudaMemcpy(dSFA, gSFA.data(), gSFA.size(), cudaMemcpyHostToDevice)); RK(cudaMemcpy(dSFB, gSFB.data(), gSFB.size(), cudaMemcpyHostToDevice));
  A0Args a;
  a.tmW = tmap(dW, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, K / 2, 512, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B);
  a.tmX = tmap(dX, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, K / 2, 304, 128, 152, CU_TENSOR_MAP_SWIZZLE_128B);
  a.tmSFA = tmap(dSFA, CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, 128, (uint64_t)2 * 2 * S * 4, 128, 4, CU_TENSOR_MAP_SWIZZLE_NONE);
  a.tmSFB = tmap(dSFB, CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, 128, (uint64_t)S * 12, 128, 12, CU_TENSOR_MAP_SWIZZLE_NONE);
  a.out = dout; a.S = S; a.NS = NS; a.neg = NEG;
  const int smem = NS * SLOT + 1024;
  RK(cudaFuncSetAttribute(a0_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  a0_kernel<<<2, 192, smem>>>(a);
  RK(cudaGetLastError()); RK(cudaDeviceSynchronize());
  std::vector<float> out(nout); RK(cudaMemcpy(out.data(), dout, nout * 4, cudaMemcpyDeviceToHost));
  // reference + bitwise compare
  long bad = 0, tot = 0; double maxabs = 0;
  for (int t = 0; t < 2; ++t) for (int p = 0; p < 2; ++p) for (int c = 0; c < 2; ++c) for (int r = 0; r < 128; ++r) {
    const int row = t * 256 + c * 128 + r, N = p ? 96 : 208;
    for (int n = 0; n < N; ++n) {
      const int tok = p ? 208 + n : n; double acc = 0;
      for (int k = 0; k < K; ++k)
        acc += (double)E2M1[wn[(size_t)row * K + k]] * sfv(wsf[(size_t)row * KB + k / 16]) * E2M1[xn[(size_t)tok * K + k]] * sfv(xsf[(size_t)tok * KB + k / 16]);
      const float ref = (float)acc, got = out[((((size_t)t * 2 + p) * 2 + c) * 128 + r) * 208 + n];
      ++tot; maxabs = fmax(maxabs, fabs(acc));
      if (memcmp(&ref, &got, 4) != 0) { if (bad < 12) printf("MISMATCH t%d %s cta%d row%d n%d: got %.4f ref %.4f\n", t, p ? "b" : "a", c, r, n, got, ref); ++bad; }
    }
  }
  printf("A0%s S=%d NS=%d K=%d: %ld / %ld mismatches (max |ref| %.0f, exact in FP32: %s) -> %s\n", NEG ? " NEGATIVE-CONTROL" : "", S, NS, K, bad, tot, maxabs,
         maxabs < 4194304.0 ? "yes" : "NO", bad ? "FAIL" : "PASS (bitwise)");
  return bad ? 2 : 0;
}
