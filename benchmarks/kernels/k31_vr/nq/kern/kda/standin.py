"""PDL stand-in kernels for 1-GPU chain benches (vendored from $W/b1/kda/tools/standin.py, b1r2-kda track;
changes: extension name, arch from K31KDA_STANDIN_ARCH).

standin(src, dst, spin_ns, marks, slot, pdl, ctas): triggers dependents at entry, waits
for its predecessor grid (griddepcontrol.wait), spins ``spin_ns`` on %globaltimer (the
AttnRes hop wait), copies ``src`` -> ``dst`` (the attention input rows), and the last
CTA records the globaltimer of its x store in marks[slot]. ``ctas`` CTAs of 128 threads.
mark(marks, slot, pdl): one thread; waits for the predecessor grid, records the time.
"""
import os

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

__global__ void standin_k(const uint4* __restrict__ src, uint4* __restrict__ dst, long n16,
                          long long spin, unsigned long long* marks, int slot, int* done) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  unsigned long long t0 = gtime();
  while ((long long)(gtime() - t0) < spin) {}
  for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (long)gridDim.x * blockDim.x)
    dst[i] = src[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    int prev = atomicAdd(done, 1);
    if (prev == gridDim.x - 1) { marks[slot] = gtime(); *done = 0; }
  }
}

__global__ void mark_k(unsigned long long* marks, int slot) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  marks[slot] = gtime();
}

static void launch(void* fn, int grid, int block, void** args, bool pdl, size_t smem) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(block); cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr; cfg.numAttrs = pdl ? 1 : 0;
  TORCH_CHECK(cudaLaunchKernelExC(&cfg, fn, args) == cudaSuccess);
}

void standin(torch::Tensor src, torch::Tensor dst, int64_t spin, torch::Tensor marks, int64_t slot,
             bool pdl, int64_t ctas, torch::Tensor done) {
  const uint4* s = (const uint4*)src.data_ptr(); uint4* d = (uint4*)dst.data_ptr();
  long n16 = src.numel() * src.element_size() / 16; long long sp = spin;
  unsigned long long* m = (unsigned long long*)marks.data_ptr(); int sl = slot; int* dn = (int*)done.data_ptr();
  void* args[] = {&s, &d, &n16, &sp, &m, &sl, &dn};
  launch((void*)standin_k, ctas, 128, args, pdl, 0);
}

void mark(torch::Tensor marks, int64_t slot, bool pdl) {
  unsigned long long* m = (unsigned long long*)marks.data_ptr(); int sl = slot;
  void* args[] = {&m, &sl};
  launch((void*)mark_k, 1, 32, args, pdl, 0);
}
"""

DECL = """
void standin(torch::Tensor src, torch::Tensor dst, int64_t spin, torch::Tensor marks, int64_t slot,
             bool pdl, int64_t ctas, torch::Tensor done);
void mark(torch::Tensor marks, int64_t slot, bool pdl);
"""

_mod = None


def mod():
    global _mod
    if _mod is None:
        _mod = load_inline(
            "nqk_kda_standin", cpp_sources=DECL, cuda_sources=SRC,
            functions=["standin", "mark"],
            extra_cuda_cflags=["-O3", "-std=c++20", os.environ.get(
                "K31KDA_STANDIN_ARCH", "-gencode=arch=compute_107a,code=sm_107a")],
            verbose=False,
        )
    return _mod
