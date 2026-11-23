# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 prefill MLA KV-cache insert (per-tensor FP8 cache, unit scale).

The stock ``concat_and_cache_mla`` kernel runs one CTA per token with one
2-byte load and one 1-byte store per thread: at a 16K-token prefill it moves
~28 MB in ~70 us (~0.4 TB/s). This kernel writes the same bytes (each BF16
value converted to E4M3 with round-to-nearest and finite saturation, exactly
``fp8::scaled_convert(x, 1.0f)``) with 16-byte vector rows of 16 tokens per
program, so the insert runs at HBM rate.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _mla_fp8_insert_kernel(
    kv_c,
    k_pe,
    cache,
    slots,
    num_tokens,
    kv_c_stride,
    k_pe_stride,
    block_stride,
    entry_stride,
    block_size,
    RANK: tl.constexpr,
    PE: tl.constexpr,
    BT: tl.constexpr,
):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    in_range = t < num_tokens
    slot = tl.load(slots + t, mask=in_range, other=-1)
    ok = in_range & (slot >= 0)
    slot = tl.where(ok, slot, 0)
    base = (slot // block_size) * block_stride + (slot % block_size) * entry_stride
    t64 = t.to(tl.int64)
    c = tl.arange(0, RANK)
    x = tl.load(kv_c + t64[:, None] * kv_c_stride + c[None, :], mask=ok[:, None])
    tl.store(
        cache + base[:, None] + c[None, :], x.to(tl.float8e4nv), mask=ok[:, None]
    )
    p = tl.arange(0, PE)
    y = tl.load(k_pe + t64[:, None] * k_pe_stride + p[None, :], mask=ok[:, None])
    tl.store(
        cache + base[:, None] + RANK + p[None, :],
        y.to(tl.float8e4nv),
        mask=ok[:, None],
    )


def prefill_dcp_supported(
    vllm_config, kv_cache_dtype: str, kv_lora_rank: int, rope_dim: int
) -> bool:
    """Whether a FLASHINFER_MLA_SPARSE layer whose prefill MQA runs off the
    specialized prefill sparse MLA (DCP > 1 or ShardQ: TP8+DCP8) still writes
    its MLA KV rows here: a P/D prefill worker (KV producer) on Rubin (sm_107)
    with the per-tensor FP8 cache and D 512 + 64. The rows are DCP-local slot
    maps; ``insert`` skips slots < 0 (tokens another DCP rank owns), as
    ``concat_and_cache_mla`` does."""
    from vllm.platforms import current_platform

    kv_transfer = getattr(vllm_config, "kv_transfer_config", None)
    return (
        kv_transfer is not None
        and bool(kv_transfer.is_kv_producer)
        and kv_cache_dtype in ("fp8", "fp8_e4m3")
        and (kv_lora_rank, rope_dim) == (512, 64)
        and current_platform.is_cuda()
        and current_platform.is_device_capability(107)
    )


def applicable(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    skip_scale: bool,
) -> bool:
    """BF16 ``kv_c [T, 512]`` and ``k_pe [T, 64]`` (unit inner stride), a
    ``[blocks, block_size, 576]`` per-tensor E4M3 cache with unit scale and
    int64 slots."""
    if kv_c.dim() != 2 or k_pe.dim() != 2 or kv_cache.dim() != 3:
        return False
    rank, pe = kv_c.shape[1], k_pe.shape[1]
    return (
        kv_cache_dtype in ("fp8", "fp8_e4m3")
        and skip_scale
        and kv_c.dtype == k_pe.dtype == torch.bfloat16
        and kv_cache.dtype in (torch.uint8, torch.float8_e4m3fn)
        and (rank, pe) == (512, 64)
        and kv_cache.shape[2] == rank + pe
        and kv_cache.stride(2) == 1
        and kv_c.stride(1) == 1
        and k_pe.stride(1) == 1
        and k_pe.shape[0] == kv_c.shape[0]
        and slot_mapping.dtype == torch.int64
        and slot_mapping.dim() == 1
        and slot_mapping.is_contiguous()
        and slot_mapping.numel() == kv_c.shape[0]
    )


def insert(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """``concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, "fp8", scale=1,
    skip_scale=True)``, bitwise; slots < 0 are skipped."""
    num_tokens = kv_c.shape[0]
    if num_tokens == 0:
        return
    cache = kv_cache.view(torch.float8_e4m3fn)
    bt = 16
    _mla_fp8_insert_kernel[(triton.cdiv(num_tokens, bt),)](
        kv_c,
        k_pe,
        cache,
        slot_mapping,
        num_tokens,
        kv_c.stride(0),
        k_pe.stride(0),
        cache.stride(0),
        cache.stride(1),
        cache.shape[1],
        RANK=kv_c.shape[1],
        PE=k_pe.shape[1],
        BT=bt,
        num_warps=4,
    )
