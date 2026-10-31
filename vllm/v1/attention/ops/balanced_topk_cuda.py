# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA implementation of the length-balanced exact BF16 top-k.

Same algorithm and output contract as ``balanced_topk`` (see
``balanced_topk.py``), written in CUDA for warp-level compaction:

1. ``sample_kernel`` (one CTA per row): candidate threshold ``t0`` on the
   order-preserving 16-bit key from a strided sample (exact for rows of at
   most ``SAMPLES`` keys), by two 8-bit radix passes over shared histograms;
   ``t0`` aims at ``2 * topk`` keys above it (at most ``TARGET``).
2. ``filter_kernel`` (persistent grid over the flattened (row, 4096-key
   block) list of the rows longer than ``CAP``): 16-byte loads; each warp
   compacts its keys above ``t0`` (and equal to ``t0``) with a shuffle scan
   and one 64-bit atomic per warp reserves both lists' slots.
3. ``select_kernel`` (one 1024-thread CTA per row): the exact top-k of the
   candidates held in registers, by two 8-bit radix passes; rows the sample
   misjudged run the same selection streamed over the whole row.

Built on first use per source hash (torch JIT extension, cached under
``VLLM_CACHE_ROOT/balanced_topk``), like the Kimi K3 SiTU kernel.
"""

import hashlib
import os
import threading

import torch

CAP = 16384
TARGET = 8192
SAMPLES = 4096
BLOCK = 4096

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>

namespace {

constexpr int CAP = 16384;
constexpr int SAMPLES = 4096;
constexpr int BLOCK = 4096;          // keys per filter work item
constexpr int FILTER_THREADS = 256;  // 16 keys per thread
constexpr int SELECT_THREADS = 1024; // 16 candidates per thread
constexpr int SAMPLE_THREADS = 256;  // 16 samples per thread
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ int row_len(const int* __restrict__ seq_lens, int r,
                                       int next_n, int width, bool per_row) {
  int n;
  if (per_row) {
    n = seq_lens[r];
  } else {
    const int q = r / next_n;
    n = seq_lens[q] - next_n + 1 + (r - q * next_n);
  }
  return min(max(n, 0), width);
}

// Order-preserving 16-bit key of BF16 bits.
__device__ __forceinline__ uint32_t key16(uint32_t u) {
  return (u & 0x8000u) ? (u ^ 0xFFFFu) : (u | 0x8000u);
}

// Warp 0: given hist[256], the largest bin b with count(bins >= b) >= need,
// and count(bins > b), into out[0], out[1]. Requires 1 <= need <= total.
__device__ __forceinline__ void top_bin(const int* hist, int need, int* out) {
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    int c[8];
    int s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      c[j] = hist[255 - (lane * 8 + j)];
      s += c[j];
    }
    int incl = s;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(FULL, incl, o);
      if (lane >= o) incl += y;
    }
    int run = incl - s;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (run < need && run + c[j] >= need) {
        out[0] = 255 - (lane * 8 + j);
        out[1] = run;
      }
      run += c[j];
    }
  }
}

// Largest 16-bit key t with count(valid keys >= t) >= need (1 <= need <=
// count(valid)); also writes count(valid keys > t) to *above. Keys of
// invalid entries are 0xFFFFFFFF. All threads of the CTA must call it.
template <int PER>
__device__ uint32_t kth_key(const uint32_t (&k)[PER], int need, int* hist,
                            int* sh, int* above) {
  for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < PER; ++i)
    if (k[i] != 0xFFFFFFFFu) atomicAdd(&hist[k[i] >> 8], 1);
  __syncthreads();
  top_bin(hist, need, sh);
  __syncthreads();
  const uint32_t b1 = sh[0];
  const int above1 = sh[1];
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < PER; ++i)
    if (k[i] != 0xFFFFFFFFu && (k[i] >> 8) == b1) atomicAdd(&hist[k[i] & 0xFF], 1);
  __syncthreads();
  top_bin(hist, need - above1, sh + 2);
  __syncthreads();
  const uint32_t t = (b1 << 8) | (uint32_t)sh[2];
  *above = above1 + sh[3];
  __syncthreads();
  return t;
}

__global__ void sample_kernel(const uint16_t* __restrict__ logits, int64_t stride,
                              const int* __restrict__ seq_lens, int next_n,
                              int width, int topk, int target, bool per_row,
                              int* __restrict__ thresh,
                              unsigned long long* __restrict__ counters) {
  __shared__ int hist[256];
  __shared__ int sh[4];
  const int r = blockIdx.x;
  const int n = row_len(seq_lens, r, next_n, width, per_row);
  if (threadIdx.x == 0) counters[r] = 0ull;
  if (n <= topk) return;  // uniform per CTA
  const uint16_t* row = logits + (int64_t)r * stride;
  constexpr int PER = SAMPLES / SAMPLE_THREADS;
  uint32_t k[PER];
  int need;
  if (n <= SAMPLES) {
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int c = threadIdx.x + i * SAMPLE_THREADS;
      k[i] = c < n ? key16(row[c]) : 0xFFFFFFFFu;
    }
    need = topk;
  } else {
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int64_t j = threadIdx.x + i * SAMPLE_THREADS;
      k[i] = key16(row[(2 * j + 1) * n / (2 * SAMPLES)]);
    }
    need = min(max((int)(((int64_t)target * SAMPLES + n - 1) / n), 1), SAMPLES);
  }
  int above;
  const uint32_t t = kth_key<PER>(k, need, hist, sh, &above);
  if (threadIdx.x == 0) thresh[r] = (int)t;
}

// Rows longer than CAP, as a flattened (row, block) work list.
__global__ void filter_kernel(const uint16_t* __restrict__ logits, int64_t stride,
                              const int* __restrict__ seq_lens, int next_n,
                              int width, int num_rows, bool per_row,
                              const int* __restrict__ thresh,
                              unsigned long long* __restrict__ counters,
                              unsigned long long* __restrict__ cand,
                              int* __restrict__ ties, int tie_cap,
                              bool vec_ok) {
  extern __shared__ int prefix[];  // num_rows + 1
  // Blocks per row in parallel, then warp 0 scans them 32 rows at a time.
  for (int r = threadIdx.x; r < num_rows; r += blockDim.x) {
    const int n = row_len(seq_lens, r, next_n, width, per_row);
    prefix[r] = n > CAP ? (n + BLOCK - 1) / BLOCK : 0;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    int carry = 0;
    for (int r0 = 0; r0 < num_rows; r0 += 32) {
      const int r = r0 + lane;
      const int c = r < num_rows ? prefix[r] : 0;
      int incl = c;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(FULL, incl, o);
        if (lane >= o) incl += y;
      }
      if (r < num_rows) prefix[r] = carry + incl - c;
      carry += __shfl_sync(FULL, incl, 31);
    }
    if (lane == 0) prefix[num_rows] = carry;
  }
  __syncthreads();
  const int total = prefix[num_rows];
  const int lane = threadIdx.x & 31;
  constexpr int PER = BLOCK / FILTER_THREADS;  // 16
  for (int w = blockIdx.x; w < total; w += gridDim.x) {
    // Row of work item w: last r with prefix[r] <= w.
    int lo = 0, hi = num_rows - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (prefix[mid] <= w) lo = mid; else hi = mid - 1;
    }
    const int r = lo;
    const int n = row_len(seq_lens, r, next_n, width, per_row);
    const uint32_t t0 = (uint32_t)thresh[r];
    const uint16_t* row = logits + (int64_t)r * stride;
    const int col0 = (w - prefix[r]) * BLOCK + threadIdx.x * PER;
    uint16_t v[PER];
    if (vec_ok && col0 + PER <= n) {
      const uint4* p = reinterpret_cast<const uint4*>(row + col0);
      const uint4 a = __ldg(p), b = __ldg(p + 1);
      const uint16_t* pa = reinterpret_cast<const uint16_t*>(&a);
      const uint16_t* pb = reinterpret_cast<const uint16_t*>(&b);
#pragma unroll
      for (int i = 0; i < 8; ++i) { v[i] = pa[i]; v[8 + i] = pb[i]; }
    } else {
#pragma unroll
      for (int i = 0; i < PER; ++i) v[i] = col0 + i < n ? row[col0 + i] : 0;
    }
    uint32_t gt = 0, eq = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      if (col0 + i < n) {
        const uint32_t k = key16(v[i]);
        gt |= (uint32_t)(k > t0) << i;
        eq |= (uint32_t)(k == t0) << i;
      }
    }
    // Packed (eq << 32 | gt) counts, warp-inclusive scan.
    const unsigned long long mine =
        ((unsigned long long)__popc(eq) << 32) | (unsigned long long)__popc(gt);
    unsigned long long incl = mine;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const unsigned long long y = __shfl_up_sync(FULL, incl, o);
      if (lane >= o) incl += y;
    }
    const unsigned long long warp_total = __shfl_sync(FULL, incl, 31);
    if (warp_total == 0ull) continue;
    unsigned long long base = 0;
    if (lane == 31) base = atomicAdd(counters + r, warp_total);
    base = __shfl_sync(FULL, base, 31);
    const unsigned long long excl = base + incl - mine;
    int pos_gt = (int)(excl & 0xFFFFFFFFull);
    int pos_eq = (int)(excl >> 32);
    unsigned long long* crow = cand + (int64_t)r * CAP;
    int* trow = ties + (int64_t)r * tie_cap;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      if ((gt >> i) & 1u) {
        if (pos_gt < CAP)
          crow[pos_gt] = ((unsigned long long)key16(v[i]) << 32) |
                         (unsigned long long)(uint32_t)(col0 + i);
        ++pos_gt;
      } else if ((eq >> i) & 1u) {
        if (pos_eq < tie_cap) trow[pos_eq] = col0 + i;
        ++pos_eq;
      }
    }
  }
}

// Writes a selected column: keys above the k-th go to [0, num_gt) and the
// first `ties` keys equal to it to [num_gt, topk), slots taken in arrival
// order (shared counters).
__device__ __forceinline__ void emit(int* out_row, int col, uint32_t k, uint32_t t,
                                     int num_gt, int ties, int* sh_gt, int* sh_eq) {
  if (k > t && k != 0xFFFFFFFFu) {
    out_row[atomicAdd(sh_gt, 1)] = col;
  } else if (k == t) {
    const int q = atomicAdd(sh_eq, 1);
    if (q < ties) out_row[num_gt + q] = col;
  }
}

__global__ void __launch_bounds__(SELECT_THREADS, 1)
select_kernel(const uint16_t* __restrict__ logits, int64_t stride,
              const int* __restrict__ seq_lens, int next_n, int width, int topk,
              bool per_row, const int* __restrict__ thresh,
              const unsigned long long* __restrict__ counters,
              const unsigned long long* __restrict__ cand,
              const int* __restrict__ ties, int tie_cap,
              int* __restrict__ out, int64_t out_stride) {
  __shared__ int hist[256];
  __shared__ int sh[8];
  const int r = blockIdx.x;
  const int n = row_len(seq_lens, r, next_n, width, per_row);
  int* out_row = out + (int64_t)r * out_stride;
  if (n <= topk) {
    for (int i = threadIdx.x; i < topk; i += SELECT_THREADS)
      out_row[i] = i < n ? i : -1;
    return;
  }
  const uint32_t t0 = (uint32_t)thresh[r];
  const uint16_t* row = logits + (int64_t)r * stride;
  constexpr int PER = CAP / SELECT_THREADS;  // 16
  uint32_t k[PER];
  int col[PER];
  int num_gt, num_eq;
  const bool direct = n <= CAP;
  if (direct) {
    int g = 0, e = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int c = threadIdx.x + i * SELECT_THREADS;
      col[i] = c;
      k[i] = c < n ? key16(row[c]) : 0xFFFFFFFFu;
      g += (k[i] != 0xFFFFFFFFu && k[i] > t0);
      e += (k[i] == t0);
    }
    if (threadIdx.x == 0) { sh[4] = 0; sh[5] = 0; }
    __syncthreads();
    // Block sums (warp reduce + shared atomics).
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      g += __shfl_down_sync(FULL, g, o);
      e += __shfl_down_sync(FULL, e, o);
    }
    if ((threadIdx.x & 31) == 0) { atomicAdd(&sh[4], g); atomicAdd(&sh[5], e); }
    __syncthreads();
    num_gt = sh[4];
    num_eq = sh[5];
    __syncthreads();
  } else {
    const unsigned long long c = counters[r];
    num_gt = (int)(c & 0xFFFFFFFFull);
    num_eq = (int)(c >> 32);
    const unsigned long long* crow = cand + (int64_t)r * CAP;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int j = threadIdx.x + i * SELECT_THREADS;
      if (j < num_gt && j < CAP) {
        const unsigned long long p = crow[j];
        k[i] = (uint32_t)(p >> 32);
        col[i] = (int)(uint32_t)(p & 0xFFFFFFFFull);
      } else {
        k[i] = 0xFFFFFFFFu;
        col[i] = -1;
      }
    }
  }
  if (threadIdx.x == 0) { sh[6] = 0; sh[7] = 0; }
  if (num_gt >= topk && (direct || num_gt <= CAP)) {
    // The k-th largest among the keys above t0.
#pragma unroll
    for (int i = 0; i < PER; ++i)
      if (k[i] <= t0) k[i] = 0xFFFFFFFFu;
    int above;
    const uint32_t t = kth_key<PER>(k, topk, hist, sh, &above);
#pragma unroll
    for (int i = 0; i < PER; ++i)
      emit(out_row, col[i], k[i], t, above, topk - above, &sh[6], &sh[7]);
  } else if (num_gt < topk && num_gt + num_eq >= topk) {
    // Everything above t0, then topk - num_gt keys equal to t0.
    __syncthreads();
    const int need_ties = topk - num_gt;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      if (k[i] != 0xFFFFFFFFu && k[i] > t0) out_row[atomicAdd(&sh[6], 1)] = col[i];
      if (direct && k[i] == t0) {
        const int q = atomicAdd(&sh[7], 1);
        if (q < need_ties) out_row[num_gt + q] = col[i];
      }
    }
    if (!direct) {
      const int* trow = ties + (int64_t)r * tie_cap;
      for (int i = threadIdx.x; i < need_ties; i += SELECT_THREADS)
        out_row[num_gt + i] = trow[i];
    }
  } else {
    // The sample misjudged t0: exact selection streamed over the whole row.
    for (int i = threadIdx.x; i < 256; i += SELECT_THREADS) hist[i] = 0;
    __syncthreads();
    for (int c = threadIdx.x; c < n; c += SELECT_THREADS)
      atomicAdd(&hist[key16(row[c]) >> 8], 1);
    __syncthreads();
    top_bin(hist, topk, sh);
    __syncthreads();
    const uint32_t b1 = sh[0];
    const int above1 = sh[1];
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += SELECT_THREADS) hist[i] = 0;
    __syncthreads();
    for (int c = threadIdx.x; c < n; c += SELECT_THREADS) {
      const uint32_t kc = key16(row[c]);
      if ((kc >> 8) == b1) atomicAdd(&hist[kc & 0xFF], 1);
    }
    __syncthreads();
    top_bin(hist, topk - above1, sh + 2);
    __syncthreads();
    const uint32_t t = (b1 << 8) | (uint32_t)sh[2];
    const int above = above1 + sh[3];
    for (int c = threadIdx.x; c < n; c += SELECT_THREADS)
      emit(out_row, c, key16(row[c]), t, above, topk - above, &sh[6], &sh[7]);
  }
}

}  // namespace

void balanced_topk_cuda(torch::Tensor logits, torch::Tensor seq_lens, int64_t next_n,
                        torch::Tensor out, int64_t topk, int64_t target,
                        torch::Tensor thresh,
                        torch::Tensor counters, torch::Tensor cand,
                        torch::Tensor ties, int64_t filter_ctas) {
  const int num_rows = logits.size(0);
  const int width = logits.size(1);
  if (num_rows == 0) return;
  const bool per_row = seq_lens.numel() == num_rows;
  const int64_t stride = logits.stride(0);
  const uint16_t* lp = reinterpret_cast<const uint16_t*>(logits.data_ptr());
  const bool vec_ok = (stride % 8 == 0) && (reinterpret_cast<uintptr_t>(lp) % 16 == 0);
  const int tie_cap = ties.numel() / num_rows;
  auto stream = c10::cuda::getCurrentCUDAStream();
  auto* ctr = reinterpret_cast<unsigned long long*>(counters.data_ptr());
  auto* cnd = reinterpret_cast<unsigned long long*>(cand.data_ptr());
  sample_kernel<<<num_rows, SAMPLE_THREADS, 0, stream>>>(
      lp, stride, seq_lens.data_ptr<int>(), (int)next_n, width, (int)topk,
      (int)target, per_row, thresh.data_ptr<int>(), ctr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  filter_kernel<<<(int)filter_ctas, FILTER_THREADS, (num_rows + 1) * sizeof(int), stream>>>(
      lp, stride, seq_lens.data_ptr<int>(), (int)next_n, width, num_rows, per_row,
      thresh.data_ptr<int>(), ctr, cnd, ties.data_ptr<int>(), tie_cap, vec_ok);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  select_kernel<<<num_rows, SELECT_THREADS, 0, stream>>>(
      lp, stride, seq_lens.data_ptr<int>(), (int)next_n, width, (int)topk, per_row,
      thresh.data_ptr<int>(), ctr, cnd, ties.data_ptr<int>(), tie_cap,
      out.data_ptr<int>(), out.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("balanced_topk_cuda", &balanced_topk_cuda);
}
"""

_lock = threading.Lock()
_mod = None


def _module():
    global _mod
    if _mod is None:
        with _lock:
            if _mod is None:
                from torch.utils.cpp_extension import load_inline

                from vllm import envs

                digest = hashlib.sha256(_CUDA_SRC.encode()).hexdigest()[:16]
                build_dir = os.path.join(envs.VLLM_CACHE_ROOT, "balanced_topk", digest)
                os.makedirs(build_dir, exist_ok=True)
                _mod = load_inline(
                    name=f"vllm_balanced_topk_{digest}",
                    cpp_sources="",
                    cuda_sources=_CUDA_SRC,
                    extra_cuda_cflags=["-O3", "-lineinfo"],
                    build_directory=build_dir,
                    verbose=False,
                )
    return _mod


def warmup() -> None:
    """Build / load the extension before serving (first use compiles it)."""
    _module()


def balanced_topk_cuda(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    next_n: int,
    topk_indices: torch.Tensor,
    topk: int,
    ctas_per_sm: int = 8,
) -> None:
    """``balanced_topk`` (same arguments and contract), CUDA kernels."""
    num_rows = logits.shape[0]
    if num_rows == 0:
        return
    assert logits.dtype == torch.bfloat16 and logits.stride(1) == 1
    assert topk_indices.dtype == torch.int32 and topk_indices.stride(1) == 1
    assert seq_lens.dtype == torch.int32 and 0 < topk <= TARGET // 2
    if seq_lens.numel() != num_rows:
        assert seq_lens.numel() * next_n == num_rows
    dev = logits.device
    tie_cap = 1 << (topk - 1).bit_length()
    cand = torch.empty(num_rows * CAP, dtype=torch.int64, device=dev)
    # counters first: 64-bit (gt | eq << 32) words need 8-byte alignment.
    scratch = torch.empty(num_rows * (tie_cap + 3), dtype=torch.int32, device=dev)
    counters, thresh, ties = scratch.split(
        [2 * num_rows, num_rows, num_rows * tie_cap]
    )
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    _module().balanced_topk_cuda(
        logits,
        seq_lens.contiguous(),
        next_n,
        topk_indices,
        topk,
        # Twice top-k expected keys above t0: the sample's estimate misses
        # top-k (the exact streamed fallback) only many standard deviations
        # out, and the filter and select handle half the candidates of
        # TARGET (DEP16 q8 owner points, DEP4 8-layer step ABBA on VR200:
        # -2.3 / -4.0 us per call at 128 / 224 rows of ~105K / ~190K keys).
        min(2 * topk, TARGET),
        thresh,
        counters,
        cand,
        ties,
        ctas_per_sm * sms,
    )
