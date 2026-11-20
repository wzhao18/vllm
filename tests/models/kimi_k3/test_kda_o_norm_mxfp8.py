# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KDA prefill gated o_norm + MXFP8 kernel (specialized_prefill/kda.py)
against the stock Triton kernel it replaces (ops/kda_norm.py)."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(100),
    reason="the FlashKDA split build targets SM10x",
)

D = 128


@pytest.mark.parametrize(
    "tokens,heads", [(1, 4), (300, 8), (2051, 96), (1, 32), (130, 64), (4097, 96)]
)
def test_o_norm_mxfp8(tokens, heads):
    """specialized_prefill.kda.o_norm_mxfp8 against kda_norm.rmsnorm_gated_mxfp8:
    same FP8 codes and scales up to rare one-step flips (the sum of squares
    runs in another order)."""
    from vllm.models.kimi_k3.nvidia.mxfp8 import unswizzle_mxfp8_scale
    from vllm.models.kimi_k3.nvidia.ops import kda_norm
    from vllm.models.kimi_k3.nvidia.ops.mxfp8_act import Mxfp8Buffer
    from vllm.models.kimi_k3.nvidia.specialized_prefill.kda import o_norm_mxfp8

    torch.manual_seed(0)
    T, P = tokens, heads * D
    x = (torch.randn(1, T, heads, D, device="cuda") * 3).to(torch.bfloat16)
    x[0, 0, 0] = 0  # an all-zero row
    gate = (torch.randn(T, P + 64, device="cuda") * 2).to(torch.bfloat16)[:, :P]
    w = torch.rand(D, device="cuda") + 0.5
    ref = Mxfp8Buffer(T, P, "cuda", torch.bfloat16)
    kda_norm.rmsnorm_gated_mxfp8(x, gate.view(T, heads, D), w, 1e-6, out=ref)
    out = Mxfp8Buffer(T, P, "cuda", torch.bfloat16)
    o_norm_mxfp8(x, gate, w, 1e-6, out)
    torch.accelerator.synchronize()
    rows = ((T + 127) // 128) * 128
    su = unswizzle_mxfp8_scale(out.scale, rows, P)[:T]
    su_ref = unswizzle_mxfp8_scale(ref.scale, rows, P)[:T]
    assert (su != su_ref).float().mean().item() <= 1e-3
    a = out.data.view(torch.uint8).to(torch.int16)
    b = ref.data.view(torch.uint8).to(torch.int16)
    assert ((a - b).abs() > 1).float().mean().item() <= 1e-4
    assert (a != b).float().mean().item() <= 2e-3


_RCP_SRC = r"""
#include <torch/extension.h>
__global__ void k(uint32_t lo, uint32_t n, unsigned long long* bad) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const float d = __uint_as_float(lo + i);
    float a, b;
    asm("div.full.f32 %0, %1, %2;" : "=f"(a) : "f"(1.0f), "f"(d));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(d));
    if (__float_as_uint(a) != __float_as_uint(b)) atomicAdd(bad, 1ull);
  }
}
torch::Tensor run(int64_t lo, int64_t n) {
  auto bad = torch::zeros({1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  k<<<4096, 256>>>(uint32_t(lo), uint32_t(n), reinterpret_cast<unsigned long long*>(bad.data_ptr()));
  return bad;
}
"""


def test_sigmoid_reciprocal_matches_div_full():
    """The kernel's sigmoid takes rcp.approx.ftz for 1 / d, d in [1, 2^126]
    (larger d use div.full): bitwise div.full.f32 on every float there."""
    from torch.utils.cpp_extension import load_inline

    mod = load_inline(
        "k3_onorm_rcp_check",
        cpp_sources="torch::Tensor run(int64_t lo, int64_t n);",
        cuda_sources=_RCP_SRC,
        functions=["run"],
    )
    lo, hi = 0x3F800000, 0x7E800000  # 1.0, 2^126 (inclusive)
    assert int(mod.run(lo, hi - lo + 1).item()) == 0
