# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``ltpin``: a cuBLASLt matmul on a pinned algorithm per (weight shape, rows).

``out[R, N] = x[R, K] @ W[N, K]^T`` (BF16 or MXFP8 W8A8 operands, FP32 accumulation,
one rounding to the output dtype; ``beta``: ``out += x @ W^T`` in place, stock's
beta-add ``residual.addmm_``), on the cuBLASLt nvjet kernel an exhaustive VR200
search measured fastest at the SP decode's per-GPU shapes (GEMMLIB/cublaslt p1: cold
weights, CUDA-graph replay, two GPUs). cuBLASLt's heuristic picks slower tiles at
these small-row shapes (up to 1.6x: latent_up n64 10.0 -> 6.4 us); the pinned
algorithm is the same library and the same math (FP32 sum over K, order per tile).

The pins are cuBLASLt 13.8 algorithm descriptors (``cublasLtMatmulAlgo_t``, 64 bytes),
valid for that library release only: building a plan checks the loaded
cuBLASLt is the ``CUBLASLT_VERSION`` minor release (13.8.x) and that ``cublasLtMatmulAlgoCheck`` accepts the
descriptor for the plan's layouts, and raises ``Unsupported`` otherwise (no silent
fallback to the heuristic). Re-search the pins (GEMMLIB ``ltsearch.cu``) on a
cuBLAS upgrade.

Plans and split-K workspaces are created at build (before graph capture); a call
sets the per-layer pointers and enqueues one ``cublasLtMatmul`` on the current
stream (graph-capturable)."""

import hashlib
import os
import threading

import torch

from ...shapes import Unsupported

# The pins' cuBLASLt release (13.8.x: searched on 13.8.0 = 130800; torch's process
# loads the 13.8.1 patch, 130801, which accepts the same descriptors under
# cublasLtMatmulAlgoCheck). Another minor release raises: re-search.
CUBLASLT_VERSION = 130800

# Pinned descriptors (cuBLASLt 13.8.0.4, VR200 sm_107, GEMMLIB/cublaslt p1 replay:
# mean of two GPUs' medians, cold us per call, pinned vs cuBLAS heuristic) per
# (shape name, measured rows): (descriptor hex = the 8 uint64 words, workspace
# bytes). Shape names: ``SHAPES``.
PINS: dict[tuple[str, int], tuple[str, int]] = {
    # dsa_in BF16 5344 x 7168: 10.09 vs 10.34 / 12.60 vs 13.99 / 15.85 vs 15.87
    ("dsa_in", 128): ("0000000f000000420000000100000023000000000000000000060000000000020001fef0000000010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    ("dsa_in", 256): ("00000011000000420000000100000023000000000000000000060000000000010001fef0000000010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    ("dsa_in", 384): ("00000074000000420000000100000023000000000000000000060000000000030001fef0ab0900010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    # dsa_o BF16 7168 x 3072: 6.16 vs 7.30 / 7.89 vs 9.28 / 9.11 vs 9.50
    ("dsa_o", 128): ("00000086000000420000000100000023000000000000000000060000000000020001fef0000000010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    ("dsa_o", 256): ("000000c1000000420000000100000023000000000000000000060000000000020001fef0000000010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    ("dsa_o", 384): ("000000d6000000420000000100000023000000000000000000050000000000000001fef0000000010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    # q_b BF16 4608 x 1536: 3.62 vs 3.76 / 3.95 vs 4.24 / 4.80 vs 5.26
    ("q_b", 128): ("0000005c000000420000000100000023000000000000000000030000000000000001fef0000000010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    ("q_b", 256): ("0000001b000000420000000100000023000000000000000000050000000000000001fef0000000010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    ("q_b", 384): ("000000b7000000420000000100000023000000000000000000050000000000020001fef0000000010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    # wq_b BF16 4096 x 1536: 3.51 vs 3.80 / 3.86 vs 4.40 / 4.31 vs 4.37
    ("wq_b", 128): ("00000050000000420000000100000023000000000000000000050000000000020001fef0000000010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    ("wq_b", 256): ("00000086000000420000000100000023000000000000000000050000000000020001fef0000000010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    ("wq_b", 384): ("00000011000000420000000100000023000000000000000000030000000000010001fef0000000010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    # latent_down BF16 3584 x 7168 (n96 only: n32 / n64 keep sdec split-K): 9.22 vs 9.55
    ("latent_down", 96): ("00000086000000420000000100000023000000000000000000050000000000020001fef0000000010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    # latent_up BF16 7168 x 3584, beta = 1 in place: 6.28 vs 7.37 / 6.36 vs 10.00 /
    # 6.79 vs 7.73
    ("latent_up", 32): ("00000050000000420000000100000023000000000000000000020000000000020001fef0000000010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    ("latent_up", 64): ("00000050000000420000000100000023000000000000000000020000000000000001fef0000000010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    ("latent_up", 96): ("00000086000000420000000100000023000000000000000000060000000000020001fef0000000010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    # wq_b at 512 rows (cuBLASLt 13.8.1 ltsearch, cold; faster than the heuristic)
    ("wq_b", 512): ("000000d0000000420000000100000023000000000000000000060000000000020001fef1ab0b00010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    # TP16 in-projections (BF16; KDA 3216 x 7168, DSA 3040 x 7168) at 512 rows, above
    # the split-K tactics (cuBLASLt 13.8.1 ltsearch, cold; faster than the heuristic).
    # At 256 rows the heuristic's pick is the search's best (no pin).
    ("kda_in_tp16", 512): ("00000012000000420000000100000023000000000000000000050000000000020001fef1000000010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    ("dsa_in_tp16", 512): ("00000011000000420000000100000023000000000000000000030000000000030001fef1ab0a00010000000e0000000e00000044000e000e0000000000000000", 0),  # noqa: E501
    # router | latent down BF16 4480 x 7168, FP32 output (D-GEMM rl32 search, 13.8,
    # cold): 8.28 / 9.16 / 9.67 vs heuristic 9.13 / 10.74 / 9.92
    ("router_latent", 32): ("0000000d000000420000000300000023000000000000000200030000000000030001fef0000000010000000e0000000e00000044000000000000000000000000", 1720336),  # noqa: E501
    ("router_latent", 64): ("0000003b000000420000000100000023000000000000000000020000000000020001fef0000000010000000e0000000e00000044000000000000000000000000", 0),  # noqa: E501
    ("router_latent", 96): ("0000005c000000420000000100000023000000000000000000050000000000020001fef0000000010000000e0000000e00000044000000000000000000000000", 0),  # noqa: E501
    # kda_in MXFP8 12448 x 7168 (R128 only): 9.98 vs 12.56 (fi_cute 10.96)
    ("kda_in", 128): ("00000014000000420000000100000024000000000000000000030000000000030001fef0000000010000001c0000001c00000044000e000e0000000000000000", 0),  # noqa: E501
    # shared_gu MXFP8 12288 x 7168: 9.58 vs 11.76 / 9.67 vs 10.60 / 9.73 vs 12.12
    ("shared_gu", 32): ("00000014000000420000000100000024000000000000000000030000000000030001fef0ab0c00010000001c0000001c00000044000e000e0000000000000000", 0),  # noqa: E501
    ("shared_gu", 64): ("00000014000000420000000100000024000000000000000000030000000000030001fef0000000010000001c0000001c00000044000e000e0000000000000000", 0),  # noqa: E501
    ("shared_gu", 96): ("00000014000000420000000100000024000000000000000000030000000000030001fef0ab0c00010000001c0000001c00000044000e000e0000000000000000", 0),  # noqa: E501
    # shared_down MXFP8 7168 x 6144: 7.36 vs 7.68 / 7.46 vs 7.73 / 7.51 vs 7.61
    ("shared_down", 32): ("00000014000000420000000100000024000000000000000000030000000000030001fef0000000010000001c0000001c00000044000e000e0000000000000000", 0),  # noqa: E501
    ("shared_down", 64): ("00000014000000420000000100000024000000000000000000030000000000010001fef0000000010000001c0000001c00000044000e000e0000000000000000", 0),  # noqa: E501
    ("shared_down", 96): ("00000014000000420000000100000024000000000000000000030000000000010001fef0000000010000001c0000001c00000044000e000e0000000000000000", 0),  # noqa: E501
}  # fmt: skip

# Shape name per (N, K, operand dtype, FP32 output, beta-add).
SHAPES: dict[tuple, str] = {
    (5344, 7168, "bf16", False, False): "dsa_in",
    (7168, 3072, "bf16", False, False): "dsa_o",
    (4608, 1536, "bf16", False, False): "q_b",
    (4096, 1536, "bf16", False, False): "wq_b",
    (3216, 7168, "bf16", False, False): "kda_in_tp16",
    (3040, 7168, "bf16", False, False): "dsa_in_tp16",
    (3584, 7168, "bf16", False, False): "latent_down",
    (7168, 3584, "bf16", False, True): "latent_up",
    (4480, 7168, "bf16", True, False): "router_latent",
    (12448, 7168, "mxfp8", False, False): "kda_in",
    (12288, 7168, "mxfp8", False, False): "shared_gu",
    (7168, 6144, "mxfp8", False, False): "shared_down",
}


def shape_name(n, k, dtype, out32=False, beta=False):
    return SHAPES.get((n, k, dtype, bool(out32), bool(beta)))


def pinned_rows(name) -> tuple[int, ...]:
    return tuple(sorted(r for (s, r) in PINS if s == name))


def pin_for(name, rows):
    """The pin measured at the nearest row count of ``name`` (ties: the larger)."""
    measured = pinned_rows(name)
    if not measured:
        return None
    best = min(measured, key=lambda m: (abs(m - rows), -m))
    return best, PINS[(name, best)]


_CPP_SRC = r"""
#include <torch/extension.h>
int64_t lt_version();
std::vector<int64_t> lt_plan(int64_t n, int64_t k, int64_t rows, bool mx, bool out32,
                             bool beta, const std::string& hex);
void lt_run(int64_t id, const torch::Tensor& w, const torch::Tensor& x, torch::Tensor& out,
            const c10::optional<torch::Tensor>& ws, const c10::optional<torch::Tensor>& wsf,
            const c10::optional<torch::Tensor>& xsf);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <mutex>
#include <vector>

#define LTCHECK(x) do { cublasStatus_t s_ = (x); TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, \
    "ltpin: " #x " failed: ", (int)s_); } while (0)

namespace {
struct Plan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, d;
  cublasLtMatmulAlgo_t algo;
  size_t ws;
  bool mx, beta;
  int64_t n, k, rows;
  at::ScalarType out;
};
std::vector<Plan> g_plans;
std::mutex g_mu;
// One cuBLASLt handle per device (ATen's handle header needs cuSPARSE headers the
// image lacks); cuBLASLt handles are not bound to a stream.
cublasLtHandle_t g_handles[64] = {};
cublasLtHandle_t handle_for(int device) {
  TORCH_CHECK(device >= 0 && device < 64, "ltpin: device ", device);
  if (!g_handles[device]) {
    TORCH_CHECK(cublasLtCreate(&g_handles[device]) == CUBLAS_STATUS_SUCCESS, "ltpin: cublasLtCreate");
  }
  return g_handles[device];
}
}  // namespace

int64_t lt_version() { return (int64_t)cublasLtGetVersion(); }

static int32_t cfg_attr(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoConfigAttributes_t at,
                        int bytes) {
  uint64_t v = 0; size_t w = 0;
  if (cublasLtMatmulAlgoConfigGetAttribute(&a, at, &v, bytes, &w) != CUBLAS_STATUS_SUCCESS)
    return -999;
  return (int32_t)(bytes == 2 ? (uint16_t)v : (uint32_t)v);
}

// Returns [plan id, workspace bytes, algo id, tile, stages, split-K, cluster shape].
std::vector<int64_t> lt_plan(int64_t n, int64_t k, int64_t rows, bool mx, bool out32,
                             bool beta, const std::string& hex) {
  TORCH_CHECK(hex.size() == 128, "ltpin: descriptor hex must be 128 characters");
  std::lock_guard<std::mutex> lock(g_mu);
  Plan p{};
  for (int i = 0; i < 8; ++i)
    p.algo.data[i] = std::stoull(hex.substr(16 * i, 16), nullptr, 16);
  cudaDataType_t ta = mx ? CUDA_R_8F_E4M3 : CUDA_R_16BF;
  cudaDataType_t tc = out32 ? CUDA_R_32F : CUDA_R_16BF;
  LTCHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t T = CUBLAS_OP_T, N = CUBLAS_OP_N;
  LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
  LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &N, sizeof N));
  if (mx) {
    int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof mode));
    LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof mode));
    // AlgoCheck validates block-scaled descriptors with their scale pointers set:
    // plan with a static aligned placeholder; every run sets the real pointers.
    static void* placeholder = nullptr;
    if (!placeholder) C10_CUDA_CHECK(cudaMalloc(&placeholder, 4096));
    const void* ph = placeholder;
    LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ph, sizeof ph));
    LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &ph, sizeof ph));
  }
  // Column-major view of out[R, N] = x[R, K] @ W[N, K]^T: D(N x R) = W^T(N x K) x(K x R).
  LTCHECK(cublasLtMatrixLayoutCreate(&p.a, ta, k, n, k));
  LTCHECK(cublasLtMatrixLayoutCreate(&p.b, ta, k, rows, k));
  LTCHECK(cublasLtMatrixLayoutCreate(&p.d, tc, n, rows, n));
  cublasLtMatmulHeuristicResult_t r{};
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cublasLtHandle_t h = handle_for(device);
  cublasStatus_t st = cublasLtMatmulAlgoCheck(h, p.op, p.a, p.b, p.d, p.d, &p.algo, &r);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulDescDestroy(p.op); cublasLtMatrixLayoutDestroy(p.a);
    cublasLtMatrixLayoutDestroy(p.b); cublasLtMatrixLayoutDestroy(p.d);
    TORCH_CHECK(false, "ltpin: cublasLtMatmulAlgoCheck rejected the pinned descriptor (status ",
                (int)st, ") for [", n, ", ", k, "] x ", rows, " rows");
  }
  p.ws = r.workspaceSize; p.mx = mx; p.beta = beta; p.n = n; p.k = k; p.rows = rows;
  p.out = out32 ? at::kFloat : at::kBFloat16;
  g_plans.push_back(p);
  return {(int64_t)g_plans.size() - 1, (int64_t)p.ws,
          cfg_attr(p.algo, CUBLASLT_ALGO_CONFIG_ID, 4),
          cfg_attr(p.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, 4),
          cfg_attr(p.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, 4),
          cfg_attr(p.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 4),
          cfg_attr(p.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, 2)};
}

static void check_packed(const torch::Tensor& t, int64_t rows, int64_t cols, const char* name) {
  TORCH_CHECK(t.dim() == 2 && t.size(0) == rows && t.size(1) == cols && t.stride(1) == 1 &&
              t.stride(0) == cols, "ltpin: ", name, " must be packed [", rows, ", ", cols, "]");
  TORCH_CHECK(((uintptr_t)t.data_ptr()) % 16 == 0, "ltpin: ", name, " not 16 B aligned");
}

void lt_run(int64_t id, const torch::Tensor& w, const torch::Tensor& x, torch::Tensor& out,
            const c10::optional<torch::Tensor>& ws, const c10::optional<torch::Tensor>& wsf,
            const c10::optional<torch::Tensor>& xsf) {
  TORCH_CHECK(id >= 0 && id < (int64_t)g_plans.size(), "ltpin: no plan ", id);
  Plan& p = g_plans[id];
  check_packed(w, p.n, p.k, "weight");
  check_packed(x, p.rows, p.k, "x");
  check_packed(out, p.rows, p.n, "out");
  TORCH_CHECK(out.scalar_type() == p.out, "ltpin: output dtype");
  TORCH_CHECK(w.scalar_type() == x.scalar_type() &&
              w.scalar_type() == (p.mx ? at::kFloat8_e4m3fn : at::kBFloat16), "ltpin: operand dtype");
  void* wsp = nullptr;
  if (p.ws) {
    TORCH_CHECK(ws.has_value() && ws->is_cuda() && (size_t)ws->nbytes() >= p.ws,
                "ltpin: plan needs ", p.ws, " workspace bytes");
    wsp = ws->data_ptr();
  }
  if (p.mx) {
    TORCH_CHECK(wsf.has_value() && xsf.has_value(), "ltpin: MXFP8 needs both scale tensors");
    const void* a = wsf->data_ptr(); const void* b = xsf->data_ptr();
    LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a, sizeof a));
    LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b, sizeof b));
  }
  const c10::cuda::CUDAGuard guard(out.device());
  float alpha = 1.f, beta = p.beta ? 1.f : 0.f;
  LTCHECK(cublasLtMatmul(handle_for(out.get_device()), p.op, &alpha, w.data_ptr(), p.a,
                         x.data_ptr(), p.b, &beta, out.data_ptr(), p.d, out.data_ptr(), p.d,
                         &p.algo, wsp, p.ws, c10::cuda::getCurrentCUDAStream()));
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
            build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_ltpin", tag)
            os.makedirs(build, exist_ok=True)
            _mod = load_inline(
                name=f"k3_ltpin_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["lt_version", "lt_plan", "lt_run"],
                extra_cuda_cflags=["-O3"],
                extra_ldflags=["-lcublasLt"],
                build_directory=build,
                verbose=False,
            )
    return _mod


def version() -> int:
    return int(_module().lt_version())


_PLANS: dict[tuple, tuple[int, int, int]] = {}
_WORKSPACES: dict[tuple, torch.Tensor] = {}


def _workspace(name: str, nbytes: int, device) -> torch.Tensor | None:
    """One split-K workspace per (shape name, device): the layers of one shape run one
    after another on one stream; different shapes may run concurrently."""
    if not nbytes:
        return None
    key = (name, torch.device(device))
    ws = _WORKSPACES.get(key)
    if ws is None or ws.numel() < nbytes:
        ws = _WORKSPACES[key] = torch.empty(nbytes, dtype=torch.uint8, device=device)
    return ws


class LtPin:
    """The pinned cuBLASLt GEMM of one [N, K] weight for each prepared R (module
    docstring). ``dtype``: "bf16" or "mxfp8" operands; ``out32``: FP32 output;
    ``beta``: in-place beta-add into ``out``. Raises ``Unsupported`` at build when a
    row has no pin, the cuBLASLt version differs or the descriptor fails the check."""

    def __init__(self, family, n, k, rows, *, dtype="bf16", out32=False, beta=False,
                 device=None):  # fmt: skip
        self.family = family
        self.name = shape_name(n, k, dtype, out32, beta)
        if self.name is None:
            raise Unsupported(
                family, None, f"ltpin: no pinned shape [{n}, {k}] {dtype} "
                f"out32={out32} beta={beta}",
            )  # fmt: skip
        found = version()
        if found // 100 != CUBLASLT_VERSION // 100:
            raise Unsupported(
                family, None, f"ltpin: pins are for cuBLASLt {CUBLASLT_VERSION}, "
                f"loaded {found} (re-search GEMMLIB ltsearch)",
            )  # fmt: skip
        self.n, self.k, self.mx, self.beta = n, k, dtype == "mxfp8", beta
        mod = _module()
        self.plans: dict[int, tuple[int, torch.Tensor | None]] = {}
        self.pinned: dict[int, int] = {}
        for r in sorted(set(rows)):
            found = pin_for(self.name, r)
            if found is None:
                raise Unsupported(family, r, f"ltpin: no pin for {self.name}")
            measured, (hexdesc, _) = found
            key = (self.name, r, n, k, self.mx, out32, beta)
            plan = _PLANS.get(key)
            if plan is None:
                try:
                    out = mod.lt_plan(n, k, r, self.mx, out32, beta, hexdesc)
                except RuntimeError as e:  # re-raised with the shape: no fallback
                    raise Unsupported(family, r, f"ltpin {self.name}: {e}") from e
                plan = _PLANS[key] = (int(out[0]), int(out[1]), int(out[2]))
            pid, nbytes, algo_id = plan
            if algo_id != 66:
                raise Unsupported(family, r, f"ltpin {self.name}: algo id {algo_id}")
            self.plans[r] = (pid, _workspace(self.name, nbytes, device))
            self.pinned[r] = measured

    def run(self, w, x, out, w_scale=None, x_scale=None):
        rows = x.shape[0]
        plan = self.plans.get(rows)
        if plan is None:
            raise Unsupported(self.family, rows, "ltpin: row count not prepared")
        pid, ws = plan
        _module().lt_run(pid, w, x, out, ws, w_scale, x_scale)
        return out


def warmup() -> None:
    _module()
