# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 GEMMs on pinned cuBLASLt algorithms (the specialized decode's drafter
projections at row counts where cuBLAS's heuristic pick is slow).

``Y[M, N] = X[M, K] W[N, K]^T`` through ``cublasLtMatmul`` with the descriptors
F.linear uses for BF16 (FP32 compute and scale, column-major ``D(N x M) = W^T X``) and
an algorithm replayed from its serialized ``cublasLtMatmulAlgo_t`` (the 64-byte
descriptor an exhaustive cuBLASLt search over the public algorithm-configuration API
picked at that shape). The same math as F.linear (FP32 accumulation; a split-K
algorithm sums in another order, one BF16 rounding). Pins are cuBLASLt-version
specific: ``PinnedLinear`` serves only rows whose pin was searched with the loaded
cuBLASLt version and passes ``cublasLtMatmulAlgoCheck`` (the rest stay on F.linear,
logged by the caller).
"""

import hashlib
import os
import threading

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_CPP_SRC = r"""
#include <torch/extension.h>
int64_t lt_version();
int64_t lt_check(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& algo);
void lt_gemm(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& w,
             const torch::Tensor& algo, torch::Tensor& workspace);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <cstring>

#define LT_CHECK(expr)                                                        \
  do {                                                                        \
    cublasStatus_t s_ = (expr);                                               \
    TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, #expr " failed: ", int(s_));     \
  } while (0)

namespace {
// The image's ATen/cuda/CUDAContext.h needs cusparse.h (absent): an own handle.
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
  Desc(int64_t m, int64_t n, int64_t k, int64_t ldx, int64_t ldo) {
    LT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t t = CUBLAS_OP_T, nn = CUBLAS_OP_N;
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof nn));
    LT_CHECK(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, k, n, k));
    LT_CHECK(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, m, ldx));
    LT_CHECK(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, n, m, ldo));
  }
  ~Desc() {
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (op) cublasLtMatmulDescDestroy(op);
  }
};

void check_args(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& algo) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.device() == w.device(), "CUDA tensors on one device");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16, "BF16 operands");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "x [M, K], w [N, K]");
  TORCH_CHECK(x.stride(1) == 1 && w.is_contiguous(), "row-major operands");
  TORCH_CHECK(algo.scalar_type() == at::kByte && algo.numel() == (int64_t)sizeof(cublasLtMatmulAlgo_t)
              && algo.device().is_cpu() && algo.is_contiguous(), "a 64-byte CPU algorithm descriptor");
}

cublasLtMatmulAlgo_t as_algo(const torch::Tensor& algo) {
  cublasLtMatmulAlgo_t a;
  std::memcpy(&a, algo.data_ptr(), sizeof a);
  return a;
}
}  // namespace

int64_t lt_version() { return (int64_t)cublasLtGetVersion(); }

int64_t lt_check(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& algo) {
  check_args(x, w, algo);
  Desc d(x.size(0), w.size(0), x.size(1), x.stride(0), w.size(0));
  cublasLtMatmulAlgo_t a = as_algo(algo);
  cublasLtMatmulHeuristicResult_t r;
  std::memset(&r, 0, sizeof r);
  cublasStatus_t s = cublasLtMatmulAlgoCheck(handle(), d.op, d.a, d.b,
                                             d.c, d.c, &a, &r);
  if (s != CUBLAS_STATUS_SUCCESS || r.state != CUBLAS_STATUS_SUCCESS) return -1;
  return (int64_t)r.workspaceSize;
}

void lt_gemm(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& w,
             const torch::Tensor& algo, torch::Tensor& workspace) {
  check_args(x, w, algo);
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.size(0) == x.size(0)
              && out.size(1) == w.size(0) && out.stride(1) == 1, "out [M, N] BF16 row-major");
  TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous(), "a CUDA workspace");
  Desc d(x.size(0), w.size(0), x.size(1), x.stride(0), out.stride(0));
  cublasLtMatmulAlgo_t a = as_algo(algo);
  float alpha = 1.f, beta = 0.f;
  auto stream = c10::cuda::getCurrentCUDAStream(x.device().index());
  LT_CHECK(cublasLtMatmul(handle(), d.op, &alpha, w.data_ptr(), d.a,
                          x.data_ptr(), d.b, &beta, out.data_ptr(), d.c, out.data_ptr(), d.c, &a,
                          workspace.data_ptr(), (size_t)workspace.numel() * workspace.element_size(),
                          stream.stream()));
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
            build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_lt_pin", tag)
            os.makedirs(build, exist_ok=True)
            _mod = load_inline(
                name=f"k3_lt_pin_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["lt_version", "lt_check", "lt_gemm"],
                extra_cuda_cflags=["-O3"],
                extra_ldflags=["-lcublasLt"],
                build_directory=build,
                verbose=False,
            )
    return _mod


def version() -> int:
    return int(_module().lt_version())


def _algo(hexstr: str) -> torch.Tensor:
    """The descriptor from its hex form: ``data[0..7]`` (uint64), each as 16 hex
    digits, most significant first (ltsearch.cu ``hexalgo``)."""
    if len(hexstr) != 128:
        raise ValueError(f"cuBLASLt algorithm descriptor of {len(hexstr)} hex digits")
    words = [int(hexstr[16 * i : 16 * i + 16], 16) for i in range(8)]
    raw = b"".join(w.to_bytes(8, "little") for w in words)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()


# Pinned algorithms by cuBLASLt version, then (N, K), then rows: the 64-byte
# descriptor (hex, as GEMMLIB ltsearch.cu serializes it). Release DSpark drafter TP4
# (C512's SP decode role, D3-32): q rows = 3 x requests (k3), context rows = 4 x
# requests; 1 VR200 GPU, cold weights, us per call (heuristic default -> pin),
# runs/D-FUSE/lt/lt1.jsonl.
PINS: dict[int, dict[tuple[int, int], dict[int, str]]] = {}
PINS[130800] = {
    # dc_fc 128 16.6 -> 16.0; dc_fc 256 22.7 -> 19.0; dc_fc 384 30.1 -> 26.8
    (1792, 35840): {
        128: "00000146000000420000000500000023000000000000000200030000000000010001fef0ab0900010000000e0000000e00000044000e000e0000000000000000",
        256: "0000001e000000420000000800000023000000000000000000050000000000020001fef0ab0900010000000e0000000e00000044000e000e0000000000000000",
        384: "000000f7000000420000000500000023000000000000000000060000000000020001fef0ab0900010000000e0000000e00000044000e000e0000000000000000",
    },
    # dq_qkv 192 9.0 -> 8.6; dq_qkv 288 9.6 -> 9.0
    (2048, 7168): {
        192: "0000013b000000420000000100000023000000000000000000030000000000030001fef0ab0900010000000e0000000e00000044000e000e0000000000000000",
        288: "0000000f0000004200000001000000230000000000000000000b0000000000030001fef0000000010000000e0000000e00000044000e000e0000000000000000",
    },
    # dc_kv 256 9.5 -> 9.2; dc_kv 384 11.0 -> 9.9
    (2560, 7168): {
        256: "0000000f000000420000000100000023000000000000000000070000000000020001fef0ab0900010000000e0000000e00000044000e000e0000000000000000",
        384: "0000001a0000004200000001000000230000000000000000000b0000000000010001fef0ab0900010000000e0000000e00000044000e000e0000000000000000",
    },
    # dq_o 96 5.3 -> 4.0; dq_o 192 5.1 -> 4.8
    (7168, 1536): {
        96: "00000086000000420000000100000023000000000000000000050000000000020001fef0000000010000000e0000000e00000044000e000e0000000000000000",
        192: "000000c1000000420000000100000023000000000000000000060000000000020001fef0000000010000000e0000000e00000044000e000e0000000000000000",
    },
    # dq_down 96 7.4 -> 6.7; dq_down 192 8.4 -> 8.2
    (7168, 3584): {
        96: "00000086000000420000000100000023000000000000000000060000000000020001fef0000000010000000e0000000e00000044000e000e0000000000000000",
        192: "000000c1000000420000000100000023000000000000000000060000000000020001fef0000000010000000e0000000e00000044000e000e0000000000000000",
    },
    # dq_gu 96 12.9 -> 11.1; dq_gu 192 16.6 -> 13.5; dq_gu 288 30.1 -> 17.1
    (7168, 7168): {
        96: "00000086000000420000000100000023000000000000000000060000000000000001fef0000000010000000e0000000e00000044000e000e0000000000000000",
        192: "000000c1000000420000000100000023000000000000000000080000000000020001fef0000000010000000e0000000e00000044000e000e0000000000000000",
        288: "000000dc000000420000000100000023000000000000000000050000000000020001fef0000000010000000e0000000e00000044000e000e0000000000000000",
    },
    # dq_lm 96 69.5 -> 46.3; dq_lm 192 76.1 -> 53.2; dq_lm 288 106.1 -> 75.9
    (40960, 7168): {
        96: "000000d4000000420000000100000023000000000000000000020000000000020001fef0000000010000000e0000000e00000044000e000e0000000000000000",
        192: "0000023a000000420000000100000023000000000000000000060000000000000001fef0000000010000000e0000000e00000044000e000e0000000000000000",
        288: "000000d8000000420000000100000023000000000000000000050000000000000001fef0000000010000000e0000000e00000044000e000e0000000000000000",
    },
}


def pins_for(shape) -> dict[int, str]:
    """The current cuBLASLt version's pins of (N, K) (rows -> descriptor hex)."""
    return PINS.get(version(), {}).get(tuple(shape), {})


class PinnedLinear:
    """``weight`` [N, K] BF16 on pinned algorithms for the rows in ``pins`` that pass
    the algorithm check here: ``kernels`` (served rows), ``run(x, out)``."""

    def __init__(self, weight: torch.Tensor, pins: dict[int, str]):
        mod = _module()
        self.weight = weight
        self.kernels: dict[int, torch.Tensor] = {}
        self.rejected: list[int] = []
        need = 0
        for rows, hexstr in sorted(pins.items()):
            algo = _algo(hexstr)
            probe = torch.empty(
                (rows, weight.shape[1]), dtype=weight.dtype, device=weight.device
            )
            size = int(mod.lt_check(probe, weight, algo))
            if size < 0:
                self.rejected.append(rows)
                continue
            self.kernels[rows] = algo
            need = max(need, size)
        self.workspace = torch.empty(
            max(need, 1), dtype=torch.uint8, device=weight.device
        )

    def takes(self, x: torch.Tensor) -> bool:
        return (
            x.dim() == 2
            and x.shape[0] in self.kernels
            and x.stride(1) == 1
            and x.dtype == self.weight.dtype
        )

    def run(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        _module().lt_gemm(out, x, self.weight, self.kernels[x.shape[0]], self.workspace)
        return out


def pinned(weight: torch.Tensor, rows) -> "PinnedLinear | None":
    """A ``PinnedLinear`` of ``weight`` for those of ``rows`` with a pin (or None)."""
    table = pins_for(weight.shape)
    pins = {r: table[r] for r in rows if r in table}
    if not pins:
        return None
    linear = PinnedLinear(weight, pins)
    if linear.rejected:
        logger.warning_once(
            "Kimi K3 sdec: cuBLASLt pins of (N, K) %s rejected by the algorithm "
            "check at rows %s (they run F.linear)",
            tuple(weight.shape),
            linear.rejected,
        )
    return linear if linear.kernels else None


class WithPins:
    """A prepared projection (``kernels`` / ``run(x, out)`` / ``policies``, or
    None) plus a ``PinnedLinear`` for rows it does not serve."""

    def __init__(self, prepared, pins: PinnedLinear):
        self.prepared, self.pins = prepared, pins
        self.policies = prepared.policies if prepared is not None else {}
        self.kernels = dict(pins.kernels)
        if prepared is not None:
            self.kernels.update(prepared.kernels)

    def run(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if self.prepared is not None and x.shape[0] in self.prepared.kernels:
            return self.prepared.run(x, out)
        return self.pins.run(x, out)


def with_pins(prepared, weight: torch.Tensor, rows):
    """``prepared`` (a split-K projection of ``weight`` or None) extended by the
    pins of the ``rows`` it does not serve."""
    served = set(prepared.kernels) if prepared is not None else set()
    pins = pinned(weight, [r for r in rows if r not in served])
    return prepared if pins is None else WithPins(prepared, pins)
