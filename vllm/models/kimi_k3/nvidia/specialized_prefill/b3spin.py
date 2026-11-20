# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DIAGNOSTIC (spin branch, never landed): a zero-work kernel with the B3
footprint (grid [2, 4, 22] of (2, 4) clusters, 352 threads, the opt-in max
dynamic smem per CTA so nothing else co-resides) whose first warp sleeps on
%globaltimer for a fixed time and whose other warps exit. It measures the
critical-path footprint S of the shared gate_up family (07:44 SOL audit,
Experiment A) without its energy."""

import hashlib
import os
import threading

import torch

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void b3spin_k(long long ns) {
  extern __shared__ char smem[];
  if (threadIdx.x >= 32) return;
  if (threadIdx.x == 0) smem[0] = 0;
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do {
    __nanosleep(500);
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  } while ((long long)(t - t0) < ns);
}

int64_t b3spin_smem() {
  int dev = 0, v = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  return v;
}

void b3spin(int64_t ns, int64_t smem) {
  static int64_t set = -1;
  if (set != smem) {
    TORCH_CHECK(cudaFuncSetAttribute(b3spin_k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem) == cudaSuccess, "b3spin smem attr");
    TORCH_CHECK(cudaFuncSetAttribute(b3spin_k, cudaFuncAttributeNonPortableClusterSizeAllowed,
                                     1) == cudaSuccess, "b3spin cluster attr");
    set = smem;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(2, 4, 22);
  cfg.blockDim = dim3(352, 1, 1);
  cfg.dynamicSmemBytes = (size_t)smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 4;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, b3spin_k, (long long)ns) == cudaSuccess,
              "b3spin launch");
}
"""
_CPP_SRC = "int64_t b3spin_smem(); void b3spin(int64_t ns, int64_t smem);"
_mod = None
_smem = 0
_lock = threading.Lock()


def _module():
    global _mod, _smem
    if _mod is not None:
        return _mod
    with _lock:
        if _mod is None:
            from torch.utils.cpp_extension import load_inline

            import vllm.envs as envs

            tag = hashlib.sha256(_CUDA_SRC.encode()).hexdigest()[:16]
            build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_b3spin", tag)
            os.makedirs(build, exist_ok=True)
            _mod = load_inline(
                name=f"k3_b3spin_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["b3spin", "b3spin_smem"],
                extra_cuda_cflags=["-O3"],
                build_directory=build,
                verbose=False,
            )
            _smem = int(_mod.b3spin_smem()) - 1024
    return _mod


def spin_for_rows(rows: int) -> None:
    """Sleep B3's in-step op-keyed time on B3's footprint (F16K 327.7 us,
    F8K 206.1 us, linear in between)."""
    us = 327.7 if rows >= 16384 else 206.1 * rows / 8192
    _module().b3spin(int(us * 1000), _smem)
