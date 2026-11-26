# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Power-aware cuBLASLt tactics for the prefill dense GEMMs (Rubin / sm_107).

The prefill forward runs under the GPU power cap (clock-event reason 0x4 in every
sample of the DEP16 F16K / F8K slice), so a GEMM's cost in the step is its energy,
not its burst time: J per call = MMA energy + per-cycle energy x cycles + operand
delivery energy x L2->SM bytes. cuBLASLt's heuristic ranks algorithms by burst
time; at the large prefill row counts the cuBLASLt tactic with the fewest joules
per call (exhaustive search, sustained at the cap, P-GEMM ``ltpow``) is pinned per
weight shape here. Same library, same math: FP32 accumulation over K in order per
output tile (no split-K pins), one rounding to the output dtype.

``out[R, N] = x[R, K] @ W[N, K]^T`` in cuBLASLt's column-major view
``D(N x R) = W^T x``, BF16 or MXFP8 (E4M3 + VEC32 UE8M0 block scales, the
F8_128x4 swizzled layout ``torch._scaled_mm`` takes). Plans are built per weight
shape; the row count of a call only changes the B / D layouts (cached per row
count, each checked once with ``cublasLtMatmulAlgoCheck``).

The pins are cuBLASLt 13.8 algorithm descriptors: a process whose cuBLASLt is
another minor release keeps the heuristic algorithm (capability check at plan
build, logged once).
"""

import hashlib
import os
import threading

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

logger = init_logger(__name__)

# The pins' cuBLASLt minor release (13.8.x).
CUBLASLT_MINOR = 1308

# Calls with fewer rows keep the stock dispatch (decode / small chunks).
MIN_ROWS = 4096

# (dtype, N, K) -> cuBLASLt algorithm descriptor (8 uint64 words, hex). Searched
# at 16384 rows (F16K per GPU) with uniform-random operands, J/call sustained at
# the 2.3 kW cap vs the heuristic pick (tile 222, cluster 2x2: nvjet tst_224x192):
PINS: dict[tuple[str, int, int], str] = {
    # dense MLP gate_up: 13.84 -> 13.11 J/call, burst 5972 -> 5775 us
    ("bf16", 67584, 7168): (
        "00000201000000420000000100000023000000000000000000050000000000000001"
        "fef0ab0900010000000e0000000e00000044000e000e0000000000000000"
    ),
    # DSA o_proj: 2.512 -> 2.449 J/call, burst 1095 -> 1037 us
    ("bf16", 7168, 12288): (
        "0000023a000000420000000100000023000000000000000000050000000000020001"
        "fef0000000010000000e0000000e00000044000e000e0000000000000000"
    ),
    # dense MLP down: 6.988 -> 6.594 J/call, burst 3040 -> 2905 us
    ("bf16", 7168, 33792): (
        "0000023a000000420000000100000023000000000000000000050000000000020001"
        "fef0ab0900010000000e0000000e00000044000e000e0000000000000000"
    ),
    # latent-MoE down_proj: 0.740 -> 0.721 J/call, burst 337 -> 321 us
    ("bf16", 3584, 7168): (
        "000001b3000000420000000100000023000000000000000000030000000000010001"
        "fef0000000010000000e0000000e00000044000e000e0000000000000000"
    ),
    # Tile 192x128 (cuBLASLt tile 31, cluster 2x2) for the K = 7168 BF16 projections
    # that stay BF16 on the MXFP4 checkpoint. Sustained 100-call blocks at the power
    # cap, cold weights, A B B A vs the heuristic pick (per-call interleaved in
    # brackets); output bitwise equal to F.linear at every searched row count:
    # KDA in_proj_qkvgfab (TP8): 16384 rows 0.874 (0.885), 32768 rows 0.993 (0.952)
    ("bf16", 6288, 7168): (
        "0000001f000000420000000100000023000000000000000000050000000000000001"
        "fef1083d00010000000e0000000e00000044000e000e0000000000000000"
    ),
    # shared expert gate_up (replicated, SP rows): 2048 rows 0.838 (0.914),
    # 4096 rows 0.892 (0.930)
    ("bf16", 12288, 7168): (
        "0000001f000000420000000100000023000000000000000000050000000000000001"
        "fef1083d00010000000e0000000e00000044000e000e0000000000000000"
    ),
    # DSA fused_qkv_a_proj: 16384 rows 0.904 (0.932), 32768 rows 0.936 (0.936)
    ("bf16", 2112, 7168): (
        "0000001f000000420000000100000023000000000000000000050000000000000001"
        "fef1083d00010000000e0000000e00000044000e000e0000000000000000"
    ),
    # DSA g_proj (TP8): 16384 rows 0.850 (0.930), 32768 rows 0.955 (0.963)
    ("bf16", 1536, 7168): (
        "0000001f000000420000000100000023000000000000000000050000000000000001"
        "fef1083d00010000000e0000000e00000044000e000e0000000000000000"
    ),
}

# Pins searched down to fewer rows than MIN_ROWS (the replicated shared expert runs
# on one rank's SP rows: 2048 at 16384 batched tokens over TP8).
PIN_MIN_ROWS: dict[tuple[str, int, int], int] = {
    ("bf16", 12288, 7168): 2048,
}


def min_rows(dtype: str, n: int, k: int) -> int:
    """Fewest rows a call of an [N, K] ``dtype`` weight runs on its pinned tactic."""
    return PIN_MIN_ROWS.get((dtype, n, k), MIN_ROWS)


_CPP_SRC = r"""
#include <torch/extension.h>
int64_t lt_version();
int64_t lt_plan(int64_t n, int64_t k, bool mx, const std::string& hex);
std::vector<int64_t> lt_info(int64_t id, int64_t rows);
void lt_run(int64_t id, const torch::Tensor& w, const torch::Tensor& x, torch::Tensor& out,
            const c10::optional<torch::Tensor>& wsf, const c10::optional<torch::Tensor>& xsf);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <map>
#include <mutex>
#include <vector>

#define LTCHECK(x) do { cublasStatus_t s_ = (x); TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, \
    "gemm_tactics: " #x " failed: ", (int)s_); } while (0)

namespace {
struct Rows {
  cublasLtMatrixLayout_t b, d;
};
struct Plan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a;
  cublasLtMatmulAlgo_t algo;
  bool mx;
  int64_t n, k;
  std::map<int64_t, Rows> rows;
};
std::vector<Plan> g_plans;
std::mutex g_mu;
cublasLtHandle_t g_handles[64] = {};
cublasLtHandle_t handle_for(int device) {
  TORCH_CHECK(device >= 0 && device < 64, "gemm_tactics: device ", device);
  if (!g_handles[device]) LTCHECK(cublasLtCreate(&g_handles[device]));
  return g_handles[device];
}
void* placeholder() {
  static void* p = nullptr;
  if (!p) C10_CUDA_CHECK(cudaMalloc(&p, 4096));
  return p;
}
int32_t cfg_attr(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoConfigAttributes_t at, int bytes) {
  uint64_t v = 0; size_t w = 0;
  if (cublasLtMatmulAlgoConfigGetAttribute(&a, at, &v, bytes, &w) != CUBLAS_STATUS_SUCCESS) return -999;
  return (int32_t)(bytes == 2 ? (uint16_t)v : (uint32_t)v);
}
// The B / D layouts of `rows` rows, checked against the pinned algorithm once.
Rows& rows_for(Plan& p, int64_t rows) {
  auto it = p.rows.find(rows);
  if (it != p.rows.end()) return it->second;
  Rows r{};
  cudaDataType_t ta = p.mx ? CUDA_R_8F_E4M3 : CUDA_R_16BF;
  LTCHECK(cublasLtMatrixLayoutCreate(&r.b, ta, p.k, rows, p.k));
  LTCHECK(cublasLtMatrixLayoutCreate(&r.d, CUDA_R_16BF, p.n, rows, p.n));
  if (p.mx) {  // AlgoCheck validates block-scaled descriptors with scale pointers set
    const void* ph = placeholder();
    LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ph, sizeof ph));
    LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &ph, sizeof ph));
  }
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cublasLtMatmulHeuristicResult_t h{};
  cublasStatus_t st = cublasLtMatmulAlgoCheck(handle_for(device), p.op, p.a, r.b, r.d, r.d, &p.algo, &h);
  if (st != CUBLAS_STATUS_SUCCESS || h.state != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatrixLayoutDestroy(r.b); cublasLtMatrixLayoutDestroy(r.d);
    TORCH_CHECK(false, "gemm_tactics: cublasLtMatmulAlgoCheck rejected the pinned algorithm for [",
                p.n, ", ", p.k, "] x ", rows, " rows (status ", (int)st, ")");
  }
  // Pins run without workspace: calls on concurrent streams share nothing.
  if (h.workspaceSize) {
    cublasLtMatrixLayoutDestroy(r.b); cublasLtMatrixLayoutDestroy(r.d);
    TORCH_CHECK(false, "gemm_tactics: the pinned algorithm for [", p.n, ", ", p.k, "] x ", rows,
                " rows needs ", h.workspaceSize, " workspace bytes; pins must need none");
  }
  return p.rows.emplace(rows, r).first->second;
}
}  // namespace

int64_t lt_version() { return (int64_t)cublasLtGetVersion(); }

int64_t lt_plan(int64_t n, int64_t k, bool mx, const std::string& hex) {
  TORCH_CHECK(hex.size() == 128, "gemm_tactics: descriptor hex must be 128 characters");
  std::lock_guard<std::mutex> lock(g_mu);
  Plan p{};
  for (int i = 0; i < 8; ++i) p.algo.data[i] = std::stoull(hex.substr(16 * i, 16), nullptr, 16);
  LTCHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t T = CUBLAS_OP_T, N = CUBLAS_OP_N;
  LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
  LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &N, sizeof N));
  if (mx) {
    int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof mode));
    LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof mode));
  }
  LTCHECK(cublasLtMatrixLayoutCreate(&p.a, mx ? CUDA_R_8F_E4M3 : CUDA_R_16BF, k, n, k));
  p.mx = mx; p.n = n; p.k = k;
  g_plans.push_back(std::move(p));
  return (int64_t)g_plans.size() - 1;
}

// [algo id, tile, stages, split-K, cluster shape] (checked at `rows` rows).
std::vector<int64_t> lt_info(int64_t id, int64_t rows) {
  std::lock_guard<std::mutex> lock(g_mu);
  TORCH_CHECK(id >= 0 && id < (int64_t)g_plans.size(), "gemm_tactics: no plan ", id);
  Plan& p = g_plans[id];
  rows_for(p, rows);
  return {cfg_attr(p.algo, CUBLASLT_ALGO_CONFIG_ID, 4),
          cfg_attr(p.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, 4),
          cfg_attr(p.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, 4),
          cfg_attr(p.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 4),
          cfg_attr(p.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, 2)};
}

void lt_run(int64_t id, const torch::Tensor& w, const torch::Tensor& x, torch::Tensor& out,
            const c10::optional<torch::Tensor>& wsf, const c10::optional<torch::Tensor>& xsf) {
  std::lock_guard<std::mutex> lock(g_mu);
  TORCH_CHECK(id >= 0 && id < (int64_t)g_plans.size(), "gemm_tactics: no plan ", id);
  Plan& p = g_plans[id];
  const int64_t rows = x.size(0);
  TORCH_CHECK(w.dim() == 2 && w.size(0) == p.n && w.size(1) == p.k && w.stride(1) == 1 &&
              w.stride(0) == p.k, "gemm_tactics: weight must be packed [N, K]");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == p.k && x.stride(1) == 1 && x.stride(0) == p.k,
              "gemm_tactics: x must be packed [rows, K]");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == rows && out.size(1) == p.n && out.is_contiguous() &&
              out.scalar_type() == at::kBFloat16, "gemm_tactics: out must be packed BF16 [rows, N]");
  TORCH_CHECK(w.scalar_type() == x.scalar_type() &&
              w.scalar_type() == (p.mx ? at::kFloat8_e4m3fn : at::kBFloat16), "gemm_tactics: operand dtype");
  Rows& r = rows_for(p, rows);
  if (p.mx) {
    TORCH_CHECK(wsf.has_value() && xsf.has_value(), "gemm_tactics: MXFP8 needs both scale tensors");
    const void* a = wsf->data_ptr(); const void* b = xsf->data_ptr();
    LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a, sizeof a));
    LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b, sizeof b));
  }
  const c10::cuda::CUDAGuard guard(out.device());
  float alpha = 1.f, beta = 0.f;
  LTCHECK(cublasLtMatmul(handle_for(out.get_device()), p.op, &alpha, w.data_ptr(), p.a, x.data_ptr(), r.b,
                         &beta, out.data_ptr(), r.d, out.data_ptr(), r.d, &p.algo, nullptr, 0,
                         c10::cuda::getCurrentCUDAStream()));
}
"""

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
            build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_pf_tactics", tag)
            os.makedirs(build, exist_ok=True)
            _mod = load_inline(
                name=f"k3_pf_tactics_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["lt_version", "lt_plan", "lt_info", "lt_run"],
                extra_cuda_cflags=["-O3"],
                extra_ldflags=["-lcublasLt"],
                build_directory=build,
                verbose=False,
            )
    return _mod


_PLANS: dict[tuple[str, int, int], int] = {}


def _supported() -> bool:
    version = int(_module().lt_version())
    if version // 100 == CUBLASLT_MINOR:
        return True
    logger.warning_once(
        "Kimi K3 prefill GEMM tactics: pinned for cuBLASLt 13.8, loaded %d; "
        "prefill GEMMs keep cuBLASLt's heuristic algorithm.",
        version,
    )
    return False


def plan(dtype: str, n: int, k: int) -> int | None:
    """Plan id of the pinned tactic of an [N, K] ``dtype`` weight, or None."""
    key = (dtype, n, k)
    if key not in PINS:
        return None
    if key not in _PLANS:
        if not _supported():
            return None
        _PLANS[key] = int(_module().lt_plan(n, k, dtype == "mxfp8", PINS[key]))
    return _PLANS[key]


def mm(
    plan_id: int,
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor | None = None,
    w_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x[R, K] @ w[N, K]^T`` -> BF16 [R, N] on the pinned tactic."""
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    _module().lt_run(plan_id, w, x, out, w_scale, x_scale)
    return out


class _PinnedBf16Method(UnquantizedLinearMethod):
    """A BF16 linear on its pinned tactic at >= ``min_rows`` rows."""

    plan_id: int
    min_rows: int

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rows = x.numel() // x.shape[-1]
        if rows >= self.min_rows and bias is None and x.dtype == torch.bfloat16:
            x2 = x.reshape(rows, x.shape[-1]).contiguous()
            out = mm(self.plan_id, x2, layer.weight)
            return out.view(*x.shape[:-1], out.shape[-1])
        return super().apply(layer, x, bias)


def enable(model: torch.nn.Module) -> int:
    """Put every BF16 linear whose weight shape has a pin on its pinned tactic
    (prefill row counts only). Returns the number switched."""
    switched = 0
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        weight = getattr(module, "weight", None)
        if type(method) is not UnquantizedLinearMethod or not isinstance(
            weight, torch.Tensor
        ):
            continue
        if weight.dtype != torch.bfloat16 or weight.dim() != 2:
            continue
        if not weight.is_contiguous():
            continue
        plan_id = plan("bf16", weight.shape[0], weight.shape[1])
        if plan_id is None:
            continue
        pinned = object.__new__(_PinnedBf16Method)
        pinned.__dict__.update(method.__dict__)
        pinned.plan_id = plan_id
        pinned.min_rows = min_rows("bf16", weight.shape[0], weight.shape[1])
        module.quant_method = pinned
        switched += 1
    return switched
