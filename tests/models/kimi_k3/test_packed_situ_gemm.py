# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The packed prefill MoE front GEMM with the shared SiTU + MXFP8 epilogue
(specialized_prefill/sm107_gemm) writes the same latent BF16 output and the same
MXFP8 activation bytes as the unfused path (cuBLASLt packed GEMM, then
``mxfp8_act.situ_and_mul_quantize_fast``).

Inputs are small integers with power-of-two scales, so every accumulator is
exact in FP32 whatever the summation order: the comparison is bitwise.
"""

import pytest
import torch

from vllm.platforms import current_platform

N_LAT, INTER, K = 3584, 6144, 7168
N = N_LAT + 2 * INTER

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)),
    reason="sm_107 CuTe DSL kernel",
)


def test_interleave_roundtrip():
    from vllm.models.kimi_k3.nvidia.model import packed_situ_gemm_deinterleave
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        packed_situ_gemm as psg,
    )

    n_lat, n_sh, k = 256, 1024, 128
    w = torch.randn(n_lat + n_sh, k)
    scale = torch.arange((n_lat + n_sh) * k // 32, dtype=torch.int32)
    w_i, s_i = w.clone(), scale.clone()
    psg.interleave_shared_rows(w_i, s_i, n_lat, k)
    x = torch.randn(8, k)
    torch.testing.assert_close(
        packed_situ_gemm_deinterleave((x @ w_i.T)[:, n_lat:]), (x @ w.T)[:, n_lat:]
    )
    assert torch.equal(w_i[:n_lat], w[:n_lat])
    group = 128 * k // 32
    assert torch.equal(
        s_i.view(-1, group)[n_lat // 128 :][1::2], scale.view(-1, group)[-n_sh // 256 :]
    )


@pytest.mark.parametrize("m,beta,linear_beta", [(4100, 4.0, 25.0), (2048, 3.0, 10.0)])
def test_packed_situ_gemm_bitwise(m, beta, linear_beta):
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        packed_situ_gemm as psg,
    )

    dev = torch.device("cuda")
    torch.manual_seed(m)
    x = torch.randint(-1, 2, (m, K), device=dev).float().to(torch.float8_e4m3fn)
    w = torch.randint(-1, 2, (N, K), device=dev).float().to(torch.float8_e4m3fn)
    sx = torch.full(
        (((m + 127) // 128) * 128 * K // 32,), 127, dtype=torch.uint8, device=dev
    )
    sw = torch.randint(119, 123, (N * K // 32,), dtype=torch.uint8, device=dev)
    y = torch._scaled_mm(
        x,
        w.t(),
        scale_a=sx.view(torch.float8_e8m0fnu),
        scale_b=sw.view(torch.float8_e8m0fnu),
        out_dtype=torch.bfloat16,
    )
    ref = mxfp8_act.Mxfp8Buffer(m, INTER + N_LAT, dev)
    ref.data.zero_()
    ref.scale.zero_()
    mxfp8_act.situ_and_mul_quantize_fast(y[:, N_LAT:], beta, linear_beta, out=ref)

    psg.interleave_shared_rows(w, sw, N_LAT, K)
    act = mxfp8_act.Mxfp8Buffer(m, INTER + N_LAT, dev)
    act.data.zero_()
    act.scale.zero_()
    lat = torch.empty(m, N_LAT, device=dev, dtype=torch.bfloat16)
    psg.PackedSituGemm(w, sw, N_LAT, beta, linear_beta)(x, sx, act, lat)

    assert torch.equal(lat.view(torch.int16), y[:, :N_LAT].view(torch.int16))
    assert torch.equal(act.data.view(torch.uint8), ref.data.view(torch.uint8))
    groups = ((m + 127) // 128, -1, 512)
    assert torch.equal(
        act.scale.view(groups)[:, : INTER // 128], ref.scale.view(groups)[:, : INTER // 128]
    )
