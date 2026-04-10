# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.quantization.utils import fp8_utils, int8_utils


@pytest.mark.parametrize(
    "shape", [(31, 128), (32, 128), (63, 256), (64, 256), (16, 512)]
)
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("tma_aligned", [False, True])
@pytest.mark.parametrize("scale_ue8m0", [False, True])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_per_token_group_quant_fp8(
    shape, column_major: bool, tma_aligned: bool, scale_ue8m0: bool, group_size: int
):
    device = "cuda"

    torch.manual_seed(42)
    num_tokens, hidden_dim = shape

    x = torch.randn((num_tokens, hidden_dim), device=device, dtype=torch.bfloat16) * 8

    # cuda path
    out_q, scale = fp8_utils.per_token_group_quant_fp8(
        x,
        group_size,
        column_major_scales=column_major,
        tma_aligned_scales=tma_aligned,
        use_ue8m0=scale_ue8m0,
    )

    # triton ref
    with patch("vllm.platforms.current_platform.is_cuda", return_value=False):
        ref_q, ref_s = fp8_utils.per_token_group_quant_fp8(
            x,
            group_size,
            column_major_scales=column_major,
            use_ue8m0=scale_ue8m0,
        )

    assert torch.allclose(out_q.float(), ref_q.float(), atol=0.15, rtol=0.15)
    assert torch.allclose(scale, ref_s, atol=0.01, rtol=0.01)


@pytest.mark.parametrize(
    "num_tokens,hidden_dim",
    [
        # No padding: mn=4 (mult of 4), groups_per_row=56 (mult of 4)
        (4, 7168),
        # MN padding only: mn=1, tma_aligned_mn=4
        (1, 7168),
        # MN padding only: mn=3, tma_aligned_mn=4
        (3, 7168),
        # K padding only: groups_per_row=5 (5%4=1)
        (4, 640),
        # K padding only: groups_per_row=6 (6%4=2)
        (4, 768),
        # Both MN and K padding
        (1, 384),
        (3, 640),
        # Larger shapes
        (64, 7168),
        (128, 14336),
    ],
)
@pytest.mark.parametrize("group_size", [128])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_per_token_group_quant_fp8_packed(num_tokens, hidden_dim, group_size):
    """Test the packed DeepGEMM quantization kernel against the Triton
    reference (row-major, UE8M0 scales)."""
    device = "cuda"
    torch.manual_seed(42)

    x = torch.randn((num_tokens, hidden_dim), device=device, dtype=torch.bfloat16) * 8

    # Packed CUDA kernel under test
    out_q, out_s_packed = fp8_utils.per_token_group_quant_fp8_packed_for_deepgemm(
        x,
        group_size=group_size,
        use_ue8m0=True,
    )

    # Triton reference (row-major float32 scales, UE8M0)
    with patch("vllm.platforms.current_platform.is_cuda", return_value=False):
        ref_q, ref_s = fp8_utils.per_token_group_quant_fp8(
            x,
            group_size,
            use_ue8m0=True,
        )

    # Quantized values must match.
    assert torch.equal(out_q, ref_q), "Quantized output mismatch"

    # Verify packed scales against reference float scales.
    # ref_s is row-major float32 with shape [mn, groups_per_row].
    # out_s_packed is int32 with 4 UE8M0 exponent bytes per word.
    mn = num_tokens
    groups_per_row = hidden_dim // group_size
    k_num_packed = (groups_per_row + 3) // 4
    tma_aligned_mn = ((mn + 3) // 4) * 4

    # Extract reference exponents from float scales.
    # UE8M0 scale = 2^(exponent - 127), so the IEEE exponent byte encodes it.
    ref_s_flat = ref_s.reshape(mn, groups_per_row)
    ref_exponents = (ref_s_flat.view(torch.int32) >> 23) & 0xFF

    # Read packed buffer as raw bytes.
    packed_bytes = out_s_packed.view(torch.int32).cpu()
    # The physical layout is column-major: stride (1, tma_aligned_mn).
    # packed_bytes has shape [mn, k_num_packed].
    for row in range(mn):
        for g in range(groups_per_row):
            pack_col = g // 4
            pos = g % 4
            word = packed_bytes[row, pack_col].item()
            byte_val = (word >> (pos * 8)) & 0xFF
            expected = ref_exponents[row, g].item()
            assert byte_val == expected, (
                f"Scale mismatch at row={row}, group={g}: "
                f"packed={byte_val}, expected={expected}"
            )

    # Verify padding bytes are zero.
    # MN padding: rows mn..tma_aligned_mn-1 in each column.
    raw = out_s_packed.cpu().contiguous()
    # Physical buffer is k_num_packed * tma_aligned_mn int32s.
    # Stride is (1, tma_aligned_mn) so column c starts at int32 offset
    # c * tma_aligned_mn.
    raw_int32 = torch.zeros(k_num_packed, tma_aligned_mn, dtype=torch.int32)
    for c in range(k_num_packed):
        for r in range(tma_aligned_mn):
            # physical offset = c * tma_aligned_mn + r
            raw_int32[c, r] = raw.view(-1)[c * tma_aligned_mn + r]

    # Check MN padding rows
    for c in range(k_num_packed):
        for r in range(mn, tma_aligned_mn):
            assert raw_int32[c, r].item() == 0, (
                f"MN padding not zero at col={c}, row={r}: "
                f"0x{raw_int32[c, r].item():08x}"
            )

    # Check K padding bytes within valid MN rows
    padded_groups_per_row = k_num_packed * 4
    if padded_groups_per_row > groups_per_row:
        for r in range(mn):
            for g in range(groups_per_row, padded_groups_per_row):
                pack_col = g // 4
                pos = g % 4
                word = raw_int32[pack_col, r].item()
                byte_val = (word >> (pos * 8)) & 0xFF
                assert byte_val == 0, (
                    f"K padding not zero at row={r}, group={g}: {byte_val}"
                )


@pytest.mark.parametrize("shape", [(32, 128), (64, 256), (16, 512)])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_per_token_group_quant_int8(shape, group_size: int):
    device = "cuda"

    torch.manual_seed(42)
    num_tokens, hidden_dim = shape

    x = torch.randn((num_tokens, hidden_dim), device=device, dtype=torch.bfloat16) * 8

    # cuda path
    out_q, scale = int8_utils.per_token_group_quant_int8(
        x,
        group_size,
    )

    # triton ref
    with patch("vllm.platforms.current_platform.is_cuda", return_value=False):
        ref_q, ref_s = int8_utils.per_token_group_quant_int8(
            x,
            group_size,
        )

    assert torch.allclose(out_q.float(), ref_q.float(), atol=0.15, rtol=0.15)
    assert torch.allclose(scale, ref_s, atol=0.01, rtol=0.01)
