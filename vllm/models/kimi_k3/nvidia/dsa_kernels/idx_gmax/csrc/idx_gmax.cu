// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Kimi K3.1 DSA decode under DCP: the indexer's paged MQA logits (DeepGEMM's SM100 kernel,
// mqa_logits_gmax.cuh) launched with the fused top-k producer outputs:
//   gmax[row][g]   = max order-preserving 16-bit key of logits [8g, 8g + 8) of the row
//                    (0 when the group holds no valid non-NaN key), written for every group
//                    that starts below the row's context length;
//   nan_count[row] += number of NaN logits in [0, context length) (the consumer re-zeroes it).
// The logits are the same values and layout as deep_gemm.fp8_fp4_paged_mqa_logits (bf16,
// row stride aligned to 1536 elements); host-side TMA descriptors mirror DeepGEMM's runtime
// (csrc/jit_kernels/impls/sm100_mqa_logits.hpp + runtime_utils.hpp @ 5cdb1457).

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "mqa_logits_gmax.cuh"

namespace k31gmax_host {

constexpr int kHeads = 32, kHeadDim = 128, kPage = 64, kBlockQ = 4, kUmmaN = 128;
constexpr int kSplitKV = 384, kSplitsPerChunk = 8, kSpecThreads = 128, kMathThreads = 384, kTmemStages = 3;

template <bool kFP4>
struct Cfg {
  static constexpr int kQStages = kFP4 ? 2 : 3;
  static constexpr int kKVStages = kFP4 ? 8 : 3;
  using qk_t = std::conditional_t<kFP4, cutlass::float_e2m1_t, cutlass::float_e4m3_t>;
  using Storage = deep_gemm::layout::MQALogitsSharedStorage<kHeads, kHeadDim, kBlockQ, kSplitKV, kUmmaN, kQStages,
                                                            kKVStages, kTmemStages, qk_t>;
  static constexpr auto kernel =
      &deep_gemm::k31gmax::paged_mqa_logits_gmax<kHeads, kHeadDim, kPage, kBlockQ, kUmmaN, kQStages, kKVStages,
                                                kTmemStages, kSplitKV, kSplitsPerChunk, kSpecThreads, kMathThreads, qk_t>;
};

using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                              const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                              CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeFn encode_fn() {
  static EncodeFn fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess &&
                    q == cudaDriverEntryPointSuccess && p != nullptr,
                "idx_gmax: cuTensorMapEncodeTiled unavailable");
    return reinterpret_cast<EncodeFn>(p);
  }();
  return fn;
}

CUtensorMapSwizzle swz(int bytes) {
  switch (bytes) {
    case 0: return CU_TENSOR_MAP_SWIZZLE_NONE;
    case 32: return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64: return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
  }
  TORCH_CHECK(false, "idx_gmax: swizzle ", bytes);
}

// dims / box in elements of `dt`, strides (outer dims) in bytes
CUtensorMap tma(void* ptr, CUtensorMapDataType dt, int rank, const uint64_t* dims, const uint64_t* strides,
                const uint32_t* box, int swizzle) {
  TORCH_CHECK(reinterpret_cast<uintptr_t>(ptr) % 16 == 0, "idx_gmax: TMA base not 16-byte aligned");
  for (int i = 0; i + 1 < rank; ++i) TORCH_CHECK(strides[i] % 16 == 0, "idx_gmax: TMA stride ", strides[i]);
  CUtensorMap m;
  cuuint64_t d[3], s[2];
  cuuint32_t b[3], e[3] = {1, 1, 1};
  for (int i = 0; i < rank; ++i) { d[i] = dims[i]; b[i] = box[i]; }
  for (int i = 0; i + 1 < rank; ++i) s[i] = strides[i];
  const CUresult r = encode_fn()(&m, dt, rank, ptr, d, s, b, e, CU_TENSOR_MAP_INTERLEAVE_NONE, swz(swizzle),
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "idx_gmax: cuTensorMapEncodeTiled failed ", static_cast<int>(r));
  return m;
}

template <bool kFP4>
void launch(const torch::Tensor& q, const torch::Tensor& q_sf, const torch::Tensor& kv, const torch::Tensor& w,
            const torch::Tensor& ctx, const torch::Tensor& bt, const torch::Tensor& idx, const torch::Tensor& sched,
            torch::Tensor& logits, torch::Tensor& gmax, torch::Tensor& nanc) {
  using C = Cfg<kFP4>;
  const int B = static_cast<int>(q.size(0));
  const int qk_bytes = kFP4 ? kHeadDim / 2 : kHeadDim;  // bytes per (token, head) / per KV token
  const int sw = qk_bytes;                              // TMA swizzle: one token row
  const auto qdt = kFP4 ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B : CU_TENSOR_MAP_DATA_TYPE_UINT8;
  // fp4 packed smem: inner box = 2 x swizzle bytes (4-bit elements); fp8: swizzle bytes
  const uint32_t inner_box = kFP4 ? 2 * sw : sw;
  // Q [B * H, D]
  const uint64_t qd[2] = {(uint64_t)kHeadDim, (uint64_t)B * kHeads};
  const uint64_t qs[1] = {(uint64_t)q.stride(2) * q.element_size()};
  const uint32_t qb[2] = {inner_box, (uint32_t)(kBlockQ * kHeads)};
  const CUtensorMap tq = tma(q.data_ptr(), qdt, 2, qd, qs, qb, sw);
  // KV [blocks, page, D] inside the fused cache (D bytes then the page's 4-byte scales)
  const int nb = static_cast<int>(kv.size(0));
  const uint64_t kv_stride = (uint64_t)kv.stride(0);  // bytes (uint8 cache)
  const uint64_t kd[3] = {(uint64_t)kHeadDim, (uint64_t)kPage, (uint64_t)nb};
  const uint64_t ks[2] = {(uint64_t)qk_bytes, kv_stride};
  const uint32_t kb[3] = {inner_box, (uint32_t)kPage, 1u};
  const CUtensorMap tkv = tma(kv.data_ptr(), qdt, 3, kd, ks, kb, sw);
  const uint64_t sd[2] = {(uint64_t)kPage, (uint64_t)nb};
  const uint64_t ss[1] = {kv_stride};
  const uint32_t sb[2] = {(uint32_t)kPage, 1u};
  const CUtensorMap tsfkv = tma(static_cast<uint8_t*>(kv.data_ptr()) + kPage * qk_bytes, CU_TENSOR_MAP_DATA_TYPE_INT32,
                                2, sd, ss, sb, 0);
  // SF Q [B, H] int32
  const uint64_t fd[2] = {(uint64_t)kHeads, (uint64_t)B};
  const uint64_t fs[1] = {(uint64_t)q_sf.stride(1) * 4};
  const uint32_t fb[2] = {(uint32_t)kHeads, (uint32_t)kBlockQ};
  const CUtensorMap tsfq = tma(q_sf.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_INT32, 2, fd, fs, fb, 0);
  // weights [B, H] bf16 (rows padded to 16 bytes)
  const uint64_t wd[2] = {(uint64_t)kHeads, (uint64_t)B};
  const uint64_t ws[1] = {(uint64_t)w.stride(0) * 2};
  const uint32_t wb[2] = {(uint32_t)kHeads, (uint32_t)kBlockQ};
  const CUtensorMap tw = tma(w.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, wd, ws, wb, 0);

  const int smem = static_cast<int>(sizeof(typename C::Storage));
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(C::kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess,
                "idx_gmax: smem attribute");
    attr = true;
  }
  const int num_sms = static_cast<int>(sched.size(0)) - 1;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(num_sms, 1, 1);
  cfg.blockDim = dim3(kSpecThreads + kMathThreads, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  const cudaError_t e = cudaLaunchKernelEx(
      &cfg, C::kernel, (uint32_t)B, (uint32_t)logits.stride(0), (uint32_t)bt.stride(0),
      reinterpret_cast<const uint32_t*>(ctx.data_ptr<int>()), reinterpret_cast<nv_bfloat16*>(logits.data_ptr()),
      reinterpret_cast<uint16_t*>(gmax.data_ptr()), (uint32_t)gmax.stride(0), nanc.data_ptr<int>(),
      reinterpret_cast<const uint32_t*>(bt.data_ptr<int>()), reinterpret_cast<const uint32_t*>(idx.data_ptr<int>()),
      reinterpret_cast<const uint32_t*>(sched.data_ptr<int>()), tq, tsfq, tkv, tsfkv, tw);
  TORCH_CHECK(e == cudaSuccess, "idx_gmax: launch failed: ", cudaGetErrorString(e));
}

// q: [B, 1, H, D bytes] (fp4 packed as int8/uint8, or fp8 e4m3), q_sf: [B, 1, H] int32 (UE8M0 x4),
// kv: fused paged cache [blocks, page, 1, D bytes + 4] uint8, w: [B, H] bf16, ctx: [B, 1] int32,
// bt: [B, max_blocks] int32, idx: [B] int32, sched: [num_sms + 1, 2] int32 (DeepGEMM metadata);
// logits: [B, S] bf16 (S % 1536 == 0), gmax: [B, G] int16 (G >= S / 8, 16-byte rows), nanc: [>= B] int32.
void paged_logits_gmax(torch::Tensor q, torch::Tensor q_sf, torch::Tensor kv, torch::Tensor w, torch::Tensor ctx,
                       torch::Tensor bt, torch::Tensor idx, torch::Tensor sched, torch::Tensor logits,
                       torch::Tensor gmax, torch::Tensor nanc, bool fp4) {
  TORCH_CHECK(q.dim() == 4 && q.size(1) == 1 && q.size(2) == kHeads, "idx_gmax: q [B, 1, 32, D]");
  TORCH_CHECK(q.size(3) == (fp4 ? kHeadDim / 2 : kHeadDim) && q.stride(3) == 1 && q.element_size() == 1,
              "idx_gmax: q head dim");
  TORCH_CHECK(kv.scalar_type() == torch::kUInt8 && kv.dim() == 4 && kv.size(1) == kPage && kv.size(2) == 1 &&
                  kv.size(3) == (fp4 ? kHeadDim / 2 : kHeadDim) + 4 && kv.stride(1) == kv.size(3) && kv.stride(3) == 1,
              "idx_gmax: fused kv cache");
  TORCH_CHECK(q_sf.scalar_type() == torch::kInt32 && w.scalar_type() == torch::kBFloat16 && w.stride(1) == 1,
              "idx_gmax: q_sf int32 / weights bf16");
  TORCH_CHECK(ctx.dim() == 2 && ctx.size(1) == 1 && ctx.is_contiguous() && bt.stride(1) == 1 && idx.is_contiguous(),
              "idx_gmax: ctx / block table / indices");
  TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.stride(1) == 1 && logits.stride(0) % 1536 == 0,
              "idx_gmax: logits");
  TORCH_CHECK(gmax.element_size() == 2 && gmax.stride(1) == 1 && gmax.stride(0) * 8 >= logits.stride(0) &&
                  gmax.stride(0) % 8 == 0,
              "idx_gmax: gmax");
  TORCH_CHECK(nanc.scalar_type() == torch::kInt32 && nanc.numel() >= q.size(0), "idx_gmax: nan count");
  if (fp4) launch<true>(q, q_sf, kv, w, ctx, bt, idx, sched, logits, gmax, nanc);
  else launch<false>(q, q_sf, kv, w, ctx, bt, idx, sched, logits, gmax, nanc);
}

}  // namespace k31gmax_host

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_logits_gmax", &k31gmax_host::paged_logits_gmax);
}
