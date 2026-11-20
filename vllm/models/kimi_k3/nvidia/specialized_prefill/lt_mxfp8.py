# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 prefill GEMMs on pinned cuBLASLt algorithms.

``Y[M, N] = X[M, K] W[N, K]^T`` (E4M3 operands, UE8M0 block-32 scales in the
128x4-swizzled layout, BF16 out) through ``cublasLtMatmul`` with the descriptors
``torch._scaled_mm`` builds for MXFP8 (column-major ``D(N x M) = W^T X``, A = W
transposed, VEC32_UE8M0 scale mode on both operands, FP32 compute) and an
algorithm replayed from its serialized ``cublasLtMatmulAlgo_t``. Same math as
``torch._scaled_mm`` (FP32 accumulation of the block-scaled products, one BF16
rounding); a different algorithm may sum K in another order.

Pins are cuBLASLt-version specific; a pin is served only when it passes
``cublasLtMatmulAlgoCheck`` at the call's shape (checked once per row count).
"""

import hashlib
import os
import threading

import torch

_CPP_SRC = r"""
#include <torch/extension.h>
int64_t lt_version();
torch::Tensor lt_heuristic(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& sx,
                           const torch::Tensor& sw, int64_t count);
torch::Tensor lt_attrs(const torch::Tensor& algo);
int64_t lt_check(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& sx,
                 const torch::Tensor& sw, const torch::Tensor& algo);
void lt_gemm(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& w,
             const torch::Tensor& sx, const torch::Tensor& sw, const torch::Tensor& algo,
             torch::Tensor& workspace);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <cstring>
#include <vector>

#define LT_CHECK(expr)                                                        \
  do {                                                                        \
    cublasStatus_t s_ = (expr);                                               \
    TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, #expr " failed: ", int(s_));     \
  } while (0)

namespace {
cublasLtHandle_t handle() {
  static cublasLtHandle_t h = [] {
    cublasLtHandle_t x;
    LT_CHECK(cublasLtCreate(&x));
    return x;
  }();
  return h;
}

struct Desc {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  Desc(int64_t m, int64_t n, int64_t k, int64_t ldx, int64_t ldw, int64_t ldo,
       const void* sx, const void* sw) {
    LT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t t = CUBLAS_OP_T, nn = CUBLAS_OP_N;
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof nn));
    cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof mode));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof mode));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sw, sizeof sw));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sx, sizeof sx));
    LT_CHECK(cublasLtMatrixLayoutCreate(&a, CUDA_R_8F_E4M3, k, n, ldw));
    LT_CHECK(cublasLtMatrixLayoutCreate(&b, CUDA_R_8F_E4M3, k, m, ldx));
    LT_CHECK(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, n, m, ldo));
  }
  ~Desc() {
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (op) cublasLtMatmulDescDestroy(op);
  }
};

void check_args(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& sx,
                const torch::Tensor& sw) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && sx.is_cuda() && sw.is_cuda(), "CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::kFloat8_e4m3fn && w.scalar_type() == at::kFloat8_e4m3fn,
              "E4M3 operands");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "x [M, K], w [N, K]");
  TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1, "K-major operands");
  TORCH_CHECK(x.size(1) % 32 == 0, "K % 32 == 0");
}

void check_algo(const torch::Tensor& algo) {
  TORCH_CHECK(algo.scalar_type() == at::kByte && algo.numel() == (int64_t)sizeof(cublasLtMatmulAlgo_t)
              && algo.device().is_cpu() && algo.is_contiguous(), "a 64-byte CPU algorithm descriptor");
}

cublasLtMatmulAlgo_t as_algo(const torch::Tensor& algo) {
  cublasLtMatmulAlgo_t a;
  std::memcpy(&a, algo.data_ptr(), sizeof a);
  return a;
}

Desc make_desc(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& sx,
               const torch::Tensor& sw, int64_t ldo) {
  return Desc(x.size(0), w.size(0), x.size(1), x.stride(0), w.stride(0), ldo, sx.data_ptr(),
              sw.data_ptr());
}
}  // namespace

int64_t lt_version() { return (int64_t)cublasLtGetVersion(); }

torch::Tensor lt_heuristic(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& sx,
                           const torch::Tensor& sw, int64_t count) {
  check_args(x, w, sx, sw);
  Desc d(x.size(0), w.size(0), x.size(1), x.stride(0), w.stride(0), w.size(0), sx.data_ptr(),
         sw.data_ptr());
  cublasLtMatmulPreference_t pref;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws = 64ull << 20;
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws,
                                                sizeof ws));
  std::vector<cublasLtMatmulHeuristicResult_t> r(count);
  int got = 0;
  cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(handle(), d.op, d.a, d.b, d.c, d.c, pref,
                                                    (int)count, r.data(), &got);
  cublasLtMatmulPreferenceDestroy(pref);
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "cublasLtMatmulAlgoGetHeuristic failed: ", int(s));
  auto out = torch::zeros({got, (int64_t)sizeof(cublasLtMatmulAlgo_t)}, torch::kUInt8);
  for (int i = 0; i < got; ++i)
    std::memcpy(out.data_ptr<uint8_t>() + i * sizeof(cublasLtMatmulAlgo_t), &r[i].algo,
                sizeof(cublasLtMatmulAlgo_t));
  return out;
}

torch::Tensor lt_attrs(const torch::Tensor& algo) {
  check_algo(algo);
  cublasLtMatmulAlgo_t a = as_algo(algo);
  const cublasLtMatmulAlgoConfigAttributes_t ids[] = {
      CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_STAGES_ID,
      CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
      CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID};
  const int n = sizeof(ids) / sizeof(ids[0]);
  auto out = torch::full({n}, -1, torch::kInt64);
  for (int i = 0; i < n; ++i) {
    uint64_t v = 0;
    size_t written = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(&a, ids[i], &v, sizeof v, &written) ==
        CUBLAS_STATUS_SUCCESS && written <= sizeof v) {
      uint64_t mask = written >= 8 ? ~0ull : ((1ull << (8 * written)) - 1);
      out.data_ptr<int64_t>()[i] = (int64_t)(v & mask);
    }
  }
  return out;
}

int64_t lt_check(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& sx,
                 const torch::Tensor& sw, const torch::Tensor& algo) {
  check_args(x, w, sx, sw);
  check_algo(algo);
  Desc d = make_desc(x, w, sx, sw, w.size(0));
  cublasLtMatmulAlgo_t a = as_algo(algo);
  cublasLtMatmulHeuristicResult_t r;
  std::memset(&r, 0, sizeof r);
  cublasStatus_t s = cublasLtMatmulAlgoCheck(handle(), d.op, d.a, d.b, d.c, d.c, &a, &r);
  if (s != CUBLAS_STATUS_SUCCESS || r.state != CUBLAS_STATUS_SUCCESS) return -1;
  return (int64_t)r.workspaceSize;
}

void lt_gemm(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& w,
             const torch::Tensor& sx, const torch::Tensor& sw, const torch::Tensor& algo,
             torch::Tensor& workspace) {
  check_args(x, w, sx, sw);
  check_algo(algo);
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.size(0) == x.size(0)
              && out.size(1) == w.size(0) && out.stride(1) == 1, "out [M, N] BF16 row-major");
  TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous(), "a CUDA workspace");
  Desc d = make_desc(x, w, sx, sw, out.stride(0));
  cublasLtMatmulAlgo_t a = as_algo(algo);
  float alpha = 1.f, beta = 0.f;
  auto stream = c10::cuda::getCurrentCUDAStream(x.device().index());
  LT_CHECK(cublasLtMatmul(handle(), d.op, &alpha, w.data_ptr(), d.a, x.data_ptr(), d.b, &beta,
                          out.data_ptr(), d.c, out.data_ptr(), d.c, &a, workspace.data_ptr(),
                          (size_t)workspace.numel() * workspace.element_size(), stream.stream()));
}
"""

ATTR_NAMES = ("algo_id", "tile", "stages", "splitk", "swizzle", "cluster", "inner")

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
            build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_lt_mxfp8", tag)
            os.makedirs(build, exist_ok=True)
            _mod = load_inline(
                name=f"k3_lt_mxfp8_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["lt_version", "lt_heuristic", "lt_attrs", "lt_check", "lt_gemm"],
                extra_cuda_cflags=["-O3"],
                extra_ldflags=["-lcublasLt"],
                build_directory=build,
                verbose=False,
            )
    return _mod


def version() -> int:
    return int(_module().lt_version())


def algo_from_hex(hexstr: str) -> torch.Tensor:
    if len(hexstr) != 128:
        raise ValueError(f"cuBLASLt algorithm descriptor of {len(hexstr)} hex digits")
    return torch.frombuffer(bytearray(bytes.fromhex(hexstr)), dtype=torch.uint8).clone()


def algo_to_hex(algo: torch.Tensor) -> str:
    return bytes(algo.contiguous().tolist()).hex()


def heuristic(x, w, sx, sw, count: int = 64) -> list[torch.Tensor]:
    """cuBLASLt's ranked algorithms for this MXFP8 GEMM (x [M, K], w [N, K])."""
    t = _module().lt_heuristic(x, w, sx, sw, count)
    return [t[i].clone() for i in range(t.shape[0])]


def attrs(algo: torch.Tensor) -> dict[str, int]:
    return dict(zip(ATTR_NAMES, _module().lt_attrs(algo).tolist()))


def check(x, w, sx, sw, algo: torch.Tensor) -> int:
    """Workspace bytes the algorithm needs at this shape, or -1 if it cannot run it."""
    return int(_module().lt_check(x, w, sx, sw, algo))


def gemm(out, x, w, sx, sw, algo: torch.Tensor, workspace: torch.Tensor) -> torch.Tensor:
    _module().lt_gemm(out, x, w, sx, sw, algo, workspace)
    return out
