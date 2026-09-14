# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Storage-only NVFP4 cache helpers for dense MLA.

The persistent cache stores each 576-element latent-K/RoPE row as 288 packed
E2M1 bytes followed by 36 E4M3 per-16 scale factors. Decode expands only the
pages referenced by the current batch into the FP8 layout consumed by the
TokenSpeed MLA kernel. This is intentionally distinct from the
``nvfp4_ds_mla`` sparse-MLA wire format.
"""

import torch

from vllm.triton_utils import tl, triton

NVFP4_MLA_HEAD_SIZE = 576
NVFP4_MLA_PACKED_BYTES = NVFP4_MLA_HEAD_SIZE // 2
NVFP4_MLA_SCALE_BYTES = NVFP4_MLA_HEAD_SIZE // 16
NVFP4_MLA_ENTRY_BYTES = NVFP4_MLA_PACKED_BYTES + NVFP4_MLA_SCALE_BYTES

_HEAD_SIZE_TL = tl.constexpr(NVFP4_MLA_HEAD_SIZE)
_PACKED_BYTES_TL = tl.constexpr(NVFP4_MLA_PACKED_BYTES)
_ENTRY_BYTES_TL = tl.constexpr(NVFP4_MLA_ENTRY_BYTES)


@triton.jit
def _e2m1_code(value):
    """Round a normalized float to an E2M1 nibble (RNE at midpoints)."""
    magnitude = tl.abs(value)
    code = tl.where(
        magnitude <= 0.25,
        0,
        tl.where(
            magnitude < 0.75,
            1,
            tl.where(
                magnitude <= 1.25,
                2,
                tl.where(
                    magnitude < 1.75,
                    3,
                    tl.where(
                        magnitude <= 2.5,
                        4,
                        tl.where(
                            magnitude < 3.5,
                            5,
                            tl.where(magnitude <= 5.0, 6, 7),
                        ),
                    ),
                ),
            ),
        ),
    )
    return code | tl.where(value < 0, 8, 0)


@triton.jit
def _e2m1_value(nibble):
    magnitude_code = nibble & 0x7
    magnitude = tl.where(
        magnitude_code == 0,
        0.0,
        tl.where(
            magnitude_code == 1,
            0.5,
            tl.where(
                magnitude_code == 2,
                1.0,
                tl.where(
                    magnitude_code == 3,
                    1.5,
                    tl.where(
                        magnitude_code == 4,
                        2.0,
                        tl.where(
                            magnitude_code == 5,
                            3.0,
                            tl.where(magnitude_code == 6, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return tl.where((nibble & 0x8) != 0, -magnitude, magnitude)


@triton.jit
def _store_nvfp4_mla_kernel(
    kv_c,
    k_pe,
    cache_u8,
    cache_fp8,
    slot_mapping,
    k_scale,
    num_tokens,
    kv_c_stride,
    k_pe_stride,
    cache_entry_stride,
    KV_LORA_RANK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    slot = tl.load(slot_mapping + token, mask=token < num_tokens, other=-1).to(tl.int64)
    valid_slot = (token < num_tokens) & (slot >= 0)
    pairs = tl.arange(0, 8)
    even_dims = group * 16 + pairs * 2
    odd_dims = even_dims + 1

    even_nope = tl.load(
        kv_c + token * kv_c_stride + even_dims,
        mask=valid_slot & (even_dims < KV_LORA_RANK),
        other=0.0,
    ).to(tl.float32)
    odd_nope = tl.load(
        kv_c + token * kv_c_stride + odd_dims,
        mask=valid_slot & (odd_dims < KV_LORA_RANK),
        other=0.0,
    ).to(tl.float32)
    even_rope_dims = even_dims - KV_LORA_RANK
    odd_rope_dims = odd_dims - KV_LORA_RANK
    even_rope = tl.load(
        k_pe + token * k_pe_stride + even_rope_dims,
        mask=valid_slot & (even_dims >= KV_LORA_RANK) & (even_rope_dims < ROPE_DIM),
        other=0.0,
    ).to(tl.float32)
    odd_rope = tl.load(
        k_pe + token * k_pe_stride + odd_rope_dims,
        mask=valid_slot & (odd_dims >= KV_LORA_RANK) & (odd_rope_dims < ROPE_DIM),
        other=0.0,
    ).to(tl.float32)
    even = tl.where(even_dims < KV_LORA_RANK, even_nope, even_rope)
    odd = tl.where(odd_dims < KV_LORA_RANK, odd_nope, odd_rope)

    scale = tl.load(k_scale).to(tl.float32)
    even /= scale
    odd /= scale
    amax = tl.maximum(tl.max(tl.abs(even), axis=0), tl.max(tl.abs(odd), axis=0))
    block_scale = tl.minimum(tl.maximum(amax / 6.0, 0.001953125), 448.0)
    block_scale_fp8 = block_scale.to(tl.float8e4nv)
    inverse_scale = 1.0 / block_scale_fp8.to(tl.float32)
    even_code = _e2m1_code(even * inverse_scale).to(tl.uint8)
    odd_code = _e2m1_code(odd * inverse_scale).to(tl.uint8)
    packed = even_code | (odd_code << 4)

    cache_base = slot * cache_entry_stride
    tl.store(
        cache_u8 + cache_base + group * 8 + pairs,
        packed,
        mask=valid_slot,
    )
    tl.store(
        cache_fp8 + cache_base + _PACKED_BYTES_TL + group,
        block_scale_fp8,
        mask=valid_slot,
    )


@triton.jit
def _stage_nvfp4_mla_as_fp8_kernel(
    src_u8,
    src_fp8,
    dst_fp8,
    page_ids,
    block_size: tl.constexpr,
    src_block_stride: tl.constexpr,
    src_token_stride: tl.constexpr,
    dst_block_stride: tl.constexpr,
    dst_token_stride: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_ref = tl.program_id(0)
    dim_block = tl.program_id(1)
    page_ref = token_ref // block_size
    token_offset = token_ref % block_size
    source_page = tl.load(page_ids + page_ref).to(tl.int64)
    dims = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dims < _HEAD_SIZE_TL
    # The launch grid contains exactly one block_size-wide region per page
    # reference. Negative entries are padding and must become zero-filled
    # staging pages.
    valid_page = source_page >= 0
    src_base = source_page * src_block_stride + token_offset * src_token_stride
    packed = tl.load(
        src_u8 + src_base + dims // 2,
        mask=valid_page & dim_mask,
        other=0,
    ).to(tl.int32)
    nibble = (packed >> ((dims & 1) * 4)) & 0xF
    block_scale = tl.load(
        src_fp8 + src_base + _PACKED_BYTES_TL + dims // 16,
        mask=valid_page & dim_mask,
        other=0.0,
    ).to(tl.float32)
    value = _e2m1_value(nibble) * block_scale
    value = tl.maximum(tl.minimum(value, 448.0), -448.0)
    value = tl.where(valid_page, value, 0.0)
    dst_base = page_ref * dst_block_stride + token_offset * dst_token_stride
    # Padded -1 references are remapped to real staging pages, so zero them.
    tl.store(dst_fp8 + dst_base + dims, value, mask=dim_mask)


@triton.jit
def _gather_nvfp4_mla_as_bf16_kernel(
    src_u8,
    src_fp8,
    dst,
    block_table,
    workspace_starts,
    k_scale,
    num_reqs,
    total_tokens,
    block_size: tl.constexpr,
    block_table_stride: tl.constexpr,
    src_block_stride: tl.constexpr,
    src_token_stride: tl.constexpr,
    dst_token_stride: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    out_token = tl.program_id(0)
    dim_block = tl.program_id(1)
    req = 0
    for candidate in tl.range(1, num_reqs):
        start = tl.load(workspace_starts + candidate)
        req = tl.where(start <= out_token, candidate, req)
    request_start = tl.load(workspace_starts + req)
    token_offset = out_token - request_start
    logical_page = token_offset // block_size
    offset_in_page = token_offset % block_size
    physical_page = tl.load(block_table + req * block_table_stride + logical_page).to(
        tl.int64
    )
    dims = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (out_token < total_tokens) & (dims < _HEAD_SIZE_TL)
    src_base = physical_page * src_block_stride + offset_in_page * src_token_stride
    packed = tl.load(src_u8 + src_base + dims // 2, mask=mask, other=0).to(tl.int32)
    nibble = (packed >> ((dims & 1) * 4)) & 0xF
    block_scale = tl.load(
        src_fp8 + src_base + _PACKED_BYTES_TL + dims // 16,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(k_scale).to(tl.float32)
    value = _e2m1_value(nibble) * block_scale * scale
    tl.store(dst + out_token * dst_token_stride + dims, value, mask=mask)


def store_nvfp4_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
) -> None:
    """Quantize and scatter dense MLA latent rows into the NVFP4 cache."""
    assert kv_c.dtype == torch.bfloat16 and k_pe.dtype == torch.bfloat16
    assert kv_c.shape[-1] == 512 and k_pe.shape[-1] == 64
    assert cache.dtype == torch.uint8 and cache.shape[-1] == NVFP4_MLA_ENTRY_BYTES
    assert kv_c.is_cuda and cache.is_cuda and slot_mapping.is_cuda and k_scale.is_cuda
    num_tokens = slot_mapping.numel()
    _store_nvfp4_mla_kernel[(num_tokens, NVFP4_MLA_SCALE_BYTES)](
        kv_c,
        k_pe,
        cache,
        cache.view(torch.float8_e4m3fn),
        slot_mapping,
        k_scale,
        num_tokens,
        kv_c.stride(0),
        k_pe.stride(0),
        cache.stride(-2),
        KV_LORA_RANK=512,
        ROPE_DIM=64,
        num_warps=1,
    )


def stage_nvfp4_mla_as_fp8(
    src_cache: torch.Tensor,
    block_table: torch.Tensor,
    dst_cache: torch.Tensor,
) -> None:
    """Expand referenced NVFP4 pages into TokenSpeed's FP8 cache layout."""
    assert src_cache.dtype == torch.uint8
    assert src_cache.shape[-1] == NVFP4_MLA_ENTRY_BYTES
    assert dst_cache.dtype == torch.float8_e4m3fn
    assert dst_cache.shape == (
        block_table.numel(),
        src_cache.shape[1],
        NVFP4_MLA_HEAD_SIZE,
    )
    assert src_cache.is_cuda and dst_cache.is_cuda and block_table.is_cuda
    page_ids = block_table.reshape(-1)
    grid = (
        page_ids.numel() * src_cache.shape[1],
        triton.cdiv(NVFP4_MLA_HEAD_SIZE, 128),
    )
    _stage_nvfp4_mla_as_fp8_kernel[grid](
        src_cache,
        src_cache.view(torch.float8_e4m3fn),
        dst_cache,
        page_ids,
        src_cache.shape[1],
        src_cache.stride(0),
        src_cache.stride(1),
        dst_cache.stride(0),
        dst_cache.stride(1),
        BLOCK_D=128,
        num_warps=4,
    )


def gather_nvfp4_mla_as_bf16(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    workspace_starts: torch.Tensor,
    batch_size: int,
    k_scale: torch.Tensor,
) -> None:
    """Gather a chunked-prefix cache into the dense BF16 MLA workspace."""
    assert src_cache.dtype == torch.uint8
    assert src_cache.shape[-1] == NVFP4_MLA_ENTRY_BYTES
    assert dst.dtype == torch.bfloat16 and dst.shape[-1] == NVFP4_MLA_HEAD_SIZE
    assert batch_size > 0
    grid = (dst.shape[0], triton.cdiv(NVFP4_MLA_HEAD_SIZE, 128))
    _gather_nvfp4_mla_as_bf16_kernel[grid](
        src_cache,
        src_cache.view(torch.float8_e4m3fn),
        dst,
        block_table,
        workspace_starts,
        k_scale,
        batch_size,
        dst.shape[0],
        src_cache.shape[1],
        block_table.stride(0),
        src_cache.stride(0),
        src_cache.stride(1),
        dst.stride(0),
        BLOCK_D=128,
        num_warps=4,
    )
