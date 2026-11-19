// Throughput probe: cvt.rn.relu.bf16x2.f32 (F2FP) vs fma.rn.bf16x2 (HFMA2) on one SM.
// Every chain feeds its own result back (no hoisting), 8 chains per thread.
// Build: nvcc -arch=sm_107a -O3 -o probe probe_k31_f2fp.cu ; run: ./probe
#include <cstdio>
#include <cstdint>

__device__ __forceinline__ uint32_t f2fp(float a, float b) {
  uint32_t d;
  asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

template <int MODE>
__global__ void probe(uint32_t* out, long long* clk, int iters, uint32_t seed) {
  uint32_t v[8], s[8];
  for (int c = 0; c < 8; ++c) { v[c] = seed * (threadIdx.x + c) | 0x3f00u; s[c] = v[c] ^ 0x1234u; }
  uint32_t w = seed | 0x3f803f80u;
  __syncthreads();
  long long t0 = clock64();
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      if (MODE == 0) {  // F2FP only
        v[c] = f2fp(__uint_as_float(v[c]), __uint_as_float(s[c]));
      } else if (MODE == 1) {  // HFMA2 only
        s[c] = hfma2(s[c], w, s[c]);
      } else if (MODE == 2) {  // independent F2FP and HFMA2 chains, 1:1
        v[c] = f2fp(__uint_as_float(v[c]), __uint_as_float(v[c]));
        s[c] = hfma2(s[c], w, s[c]);
      } else {  // coupled like the indexer epilogue: s += relu_bf16(v) * w
        uint32_t r = f2fp(__uint_as_float(v[c]), __uint_as_float(s[c]));
        s[c] = hfma2(r, w, s[c]);
        v[c] ^= s[c];
      }
    }
  }
  __syncthreads();
  long long t1 = clock64();
  uint32_t acc = 0;
  for (int c = 0; c < 8; ++c) acc ^= v[c] ^ s[c];
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
  if (threadIdx.x == 0) clk[blockIdx.x] = t1 - t0;
}

template <int MODE>
void run(const char* name, int threads, int instr_per_chain_step) {
  uint32_t* out; long long* clk;
  cudaMalloc(&out, threads * 4 * 4); cudaMalloc(&clk, 8 * 4);
  int iters = 4096;
  probe<MODE><<<1, threads>>>(out, clk, iters, 7);
  probe<MODE><<<1, threads>>>(out, clk, iters, 7);
  cudaDeviceSynchronize();
  long long c; cudaMemcpy(&c, clk, 8, cudaMemcpyDeviceToHost);
  double winstr = double(iters) * 8 * instr_per_chain_step * (threads / 32);
  printf("%-28s threads %4d: %.3f warp-instr/clk/SM (%.3f per SMSP)\n", name, threads, winstr / c,
         winstr / c / 4);
  cudaFree(out); cudaFree(clk);
}

int main() {
  for (int t : {256, 512, 1024}) {
    run<0>("f2fp only", t, 1);
    run<1>("hfma2 only", t, 1);
    run<2>("f2fp+hfma2 independent", t, 2);
    run<3>("f2fp->hfma2 coupled(+lop)", t, 3);
  }
  return 0;
}
