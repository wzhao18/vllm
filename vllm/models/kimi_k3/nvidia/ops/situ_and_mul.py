# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vectorized SiTU-and-mul for the Kimi K3 shared experts.

Bitwise equal to ``torch.ops._C.situ_and_mul`` (csrc/activation_kernels.cu) for
every input, subnormals and NaNs included: the per-element code is that
kernel's, compiled the same way (nvcc without fast math: precise ``tanhf``,
IEEE division, no flush-to-zero) and rounded with the same ``c10::BFloat16``
conversion. Only the work layout differs: the CUDA op runs one block per row
with scalar 2-byte accesses (~10x off its memory floor at prefill sizes); this
kernel gives each thread 8 consecutive columns with 16-byte loads and stores.

Built on first use per source hash (torch JIT extension, cached under
``VLLM_CACHE_ROOT/k3_situ``).
"""

import hashlib
import os
import threading

import torch

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

// Verbatim from csrc/activation_kernels.cu (situ_tanh / situ_activation).
__device__ __forceinline__ float situ_tanh(float x) { return tanhf(x); }

__device__ __forceinline__ float situ_activation(float g, float u, float beta,
                                                 float linear_beta,
                                                 bool clamp_up, float inv_beta,
                                                 float inv_linear_beta) {
  const float gate_out =
      (0.5f * beta) * situ_tanh(g * inv_beta) * (1.0f + situ_tanh(g * 0.5f));
  const float up_out =
      clamp_up ? linear_beta * situ_tanh(u * inv_linear_beta) : u;
  return gate_out * up_out;
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t b) {
  return (float)c10::BFloat16(b, c10::BFloat16::from_bits());
}

// SiTU by table (>= SITU_LUT_MIN_ROWS rows): the inputs are BF16, so gate_out(g)
// and up_out(u) of situ_activation take at most 2^16 values each. Each CTA
// builds both for |x| in [2^-16, 2^16) (either sign) in shared memory with the
// exact situ_activation expressions; zeros, subnormals, huge values, Inf and
// NaN take situ_activation directly. gate_out * up_out is the same FP32
// product, so the result is bitwise equal; 3 precise tanhf per element
// (MUFU-bound) become 2 shared-memory loads.
constexpr int SITU_LUT_E0 = 111;
constexpr int SITU_LUT_NE = 32;
constexpr int SITU_LUT_N = 2 * SITU_LUT_NE * 128;
constexpr int64_t SITU_LUT_MIN_ROWS = 1024;

__device__ __forceinline__ int situ_lut_index(uint32_t b) {
  const uint32_t e = ((b >> 7) & 0xffu) - (uint32_t)SITU_LUT_E0;
  return e < (uint32_t)SITU_LUT_NE
             ? (int)(((b >> 15) << 12) | (e << 7) | (b & 0x7fu))
             : -1;
}

__global__ void __launch_bounds__(512)
    situ_and_mul_lut_kernel(uint16_t* __restrict__ out,
                            const uint16_t* __restrict__ input,
                            const int64_t rows, const int d,
                            const int64_t in_stride, const int64_t out_stride,
                            const float beta, const float linear_beta) {
  extern __shared__ float lut[];
  const bool clamp_up = linear_beta > 0.0f;
  const float inv_beta = 1.0f / beta;
  const float inv_linear_beta = clamp_up ? 1.0f / linear_beta : 0.0f;
  for (int i = threadIdx.x; i < SITU_LUT_N; i += blockDim.x) {
    const uint32_t bits = ((uint32_t)(i >> 12) << 15) |
                          ((uint32_t)(((i >> 7) & (SITU_LUT_NE - 1)) +
                                      SITU_LUT_E0) << 7) |
                          (uint32_t)(i & 0x7f);
    const float x = bf16_bits_to_float((uint16_t)bits);
    lut[i] = (0.5f * beta) * situ_tanh(x * inv_beta) *
             (1.0f + situ_tanh(x * 0.5f));
    lut[SITU_LUT_N + i] =
        clamp_up ? linear_beta * situ_tanh(x * inv_linear_beta) : x;
  }
  __syncthreads();
  constexpr int U = 4;  // 16-byte loads in flight per thread (x2: gate, up)
  const int vecs_per_row = d / 8;
  const int64_t total = rows * vecs_per_row;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t v0 = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; v0 < total;
       v0 += stride * U) {
    uint4 graw[U], uraw[U];
    int64_t oo[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int64_t v = v0 + u * stride;
      if (v < total) {
        const int64_t row = v / vecs_per_row;
        const int col = (int)(v - row * vecs_per_row) * 8;
        const uint16_t* gp = input + row * in_stride + col;
        graw[u] = __ldg(reinterpret_cast<const uint4*>(gp));
        uraw[u] = __ldg(reinterpret_cast<const uint4*>(gp + d));
        oo[u] = row * out_stride + col;
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (v0 + u * stride < total) {
        const uint16_t* g16 = reinterpret_cast<const uint16_t*>(&graw[u]);
        const uint16_t* u16 = reinterpret_cast<const uint16_t*>(&uraw[u]);
        uint4 oraw;
        uint16_t* o16 = reinterpret_cast<uint16_t*>(&oraw);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int ig = situ_lut_index(g16[i]);
          const int iu = situ_lut_index(u16[i]);
          float r;
          if (ig >= 0 && iu >= 0) {
            r = lut[ig] * lut[SITU_LUT_N + iu];
          } else {
            r = situ_activation(bf16_bits_to_float(g16[i]),
                                bf16_bits_to_float(u16[i]), beta, linear_beta,
                                clamp_up, inv_beta, inv_linear_beta);
          }
          o16[i] = c10::BFloat16(r).x;
        }
        *reinterpret_cast<uint4*>(out + oo[u]) = oraw;
      }
    }
  }
}

// 8 columns per thread: one 16-byte load of gate, one of up, one store.
__global__ void situ_and_mul_vec8_kernel(uint16_t* __restrict__ out,
                                         const uint16_t* __restrict__ input,
                                         const int64_t rows, const int d,
                                         const int64_t in_stride,
                                         const int64_t out_stride,
                                         const float beta,
                                         const float linear_beta) {
  const bool clamp_up = linear_beta > 0.0f;
  const float inv_beta = 1.0f / beta;
  const float inv_linear_beta = clamp_up ? 1.0f / linear_beta : 0.0f;
  const int vecs_per_row = d / 8;
  const int64_t total = rows * vecs_per_row;
  for (int64_t v = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; v < total;
       v += (int64_t)gridDim.x * blockDim.x) {
    const int64_t row = v / vecs_per_row;
    const int col = (int)(v - row * vecs_per_row) * 8;
    const uint16_t* gp = input + row * in_stride + col;
    const uint4 graw = __ldg(reinterpret_cast<const uint4*>(gp));
    const uint4 uraw = __ldg(reinterpret_cast<const uint4*>(gp + d));
    const uint16_t* g16 = reinterpret_cast<const uint16_t*>(&graw);
    const uint16_t* u16 = reinterpret_cast<const uint16_t*>(&uraw);
    uint4 oraw;
    uint16_t* o16 = reinterpret_cast<uint16_t*>(&oraw);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float r = situ_activation(bf16_bits_to_float(g16[i]),
                                      bf16_bits_to_float(u16[i]), beta,
                                      linear_beta, clamp_up, inv_beta,
                                      inv_linear_beta);
      o16[i] = c10::BFloat16(r).x;
    }
    *reinterpret_cast<uint4*>(out + row * out_stride + col) = oraw;
  }
}

}  // namespace

void situ_and_mul_vec8(torch::Tensor& out, const torch::Tensor& input,
                       double beta, double linear_beta) {
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 &&
              out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(input.dim() == 2 && out.dim() == 2 &&
              input.stride(1) == 1 && out.stride(1) == 1);
  const int64_t rows = input.size(0);
  const int d = (int)(input.size(1) / 2);
  TORCH_CHECK(d % 8 == 0 && input.stride(0) % 8 == 0 && out.stride(0) % 8 == 0);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0);
  if (rows == 0) return;
  if (rows >= SITU_LUT_MIN_ROWS) {
    const int threads = 512;
    const size_t smem = 2 * SITU_LUT_N * sizeof(float);
    static int blocks = [&] {
      TORCH_CHECK(cudaFuncSetAttribute(situ_and_mul_lut_kernel,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       (int)smem) == cudaSuccess);
      int per_sm = 0, dev = 0, sms = 0;
      TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess);
      TORCH_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                         dev) == cudaSuccess);
      TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                      &per_sm, situ_and_mul_lut_kernel, threads, smem) ==
                  cudaSuccess && per_sm > 0);
      return per_sm * sms;
    }();
    situ_and_mul_lut_kernel<<<blocks, threads, smem,
                              c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<uint16_t*>(out.data_ptr()),
        reinterpret_cast<const uint16_t*>(input.data_ptr()), rows, d,
        input.stride(0), out.stride(0), (float)beta, (float)linear_beta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  const int threads = 256;
  const int64_t total = rows * (d / 8);
  const int64_t max_blocks = 4096;
  const int blocks =
      (int)std::min<int64_t>((total + threads - 1) / threads, max_blocks);
  situ_and_mul_vec8_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint16_t*>(out.data_ptr()),
      reinterpret_cast<const uint16_t*>(input.data_ptr()), rows, d,
      input.stride(0), out.stride(0), (float)beta, (float)linear_beta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_CPP_SRC = (
    "void situ_and_mul_vec8(torch::Tensor& out, const torch::Tensor& input, "
    "double beta, double linear_beta);"
)

_lock = threading.Lock()
_mod = None


def _module():
    global _mod
    if _mod is not None:
        return _mod
    with _lock:
        if _mod is None:
            from torch.utils.cpp_extension import load_inline

            import vllm.envs as envs

            h = hashlib.sha256((_CUDA_SRC + _CPP_SRC).encode())
            h.update(f"{torch.__version__} {torch.version.cuda}".encode())
            tag = h.hexdigest()[:16]
            build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_situ", tag)
            os.makedirs(build, exist_ok=True)
            # No --use_fast_math: precise tanhf, IEEE division, subnormals kept,
            # as csrc/activation_kernels.cu is built.
            _mod = load_inline(
                name=f"k3_situ_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["situ_and_mul_vec8"],
                extra_cuda_cflags=["-O3"],
                build_directory=build,
                verbose=False,
            )
    return _mod


def situ_and_mul(
    x: torch.Tensor, beta: float, linear_beta: float | None
) -> torch.Tensor:
    """``x``: [..., 2 * d] BF16 (last dim contiguous) -> [..., d]."""
    d = x.shape[-1] // 2
    x2 = x.reshape(-1, x.shape[-1])
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    if x2.shape[0] == 0:
        return out
    _module().situ_and_mul_vec8(
        out.view(-1, d),
        x2,
        float(beta),
        -1.0 if linear_beta is None else float(linear_beta),
    )
    return out


def warmup() -> None:
    """Build / load the extension before serving (first use compiles it)."""
    _module()
