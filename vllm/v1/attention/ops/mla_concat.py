# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-preserving MLA key concatenation for broadcast and strided inputs."""

import torch

from vllm.triton_utils import tl, triton

# Conservative eager crossover measured on GB300, including output allocation.
# The caller restricts this optimization to SM103; other architectures retain
# the existing copy path until their crossover has been measured.
MIN_TRITON_CONCAT_BYTES = 64 * 1024 * 1024


def can_use_triton_concat(
    k: torch.Tensor, k_nope: torch.Tensor, k_pe: torch.Tensor
) -> bool:
    return (
        k.numel() * k.element_size() >= MIN_TRITON_CONCAT_BYTES
        and k.ndim == k_nope.ndim == 3
        and k_pe.ndim <= 3
        and k.is_cuda
        and k.dtype in (torch.bfloat16, torch.float16, torch.float8_e4m3fn)
        and k.dtype == k_nope.dtype == k_pe.dtype
        and k.device == k_nope.device == k_pe.device
        and k_nope.shape[-1] > 0
        and k_pe.shape[-1] > 0
    )


@triton.jit
def _concat_mla_k_kernel(
    K_NOPE,
    K_PE,
    K,
    numel,
    H: tl.constexpr,
    DN: tl.constexpr,
    DR: tl.constexpr,
    N0: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    R2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    dim = index % (DN + DR)
    head = index // (DN + DR) % H
    token = index // ((DN + DR) * H)
    valid = index < numel
    nope = tl.load(
        K_NOPE + token * N0 + head * N1 + dim * N2, valid & (dim < DN), other=0
    )
    rope = tl.load(
        K_PE + token * R0 + head * R1 + (dim - DN) * R2, valid & (dim >= DN), other=0
    )
    tl.store(K + index, tl.where(dim < DN, nope, rope), valid)


def concat_mla_k(k: torch.Tensor, k_nope: torch.Tensor, k_pe: torch.Tensor) -> None:
    """Copy strided/broadcast keys into a fresh contiguous output.

    Integer views preserve every input bit, including FP8 NaN payloads, without
    a floating-point conversion. This does not modify the projected KV storage
    shared by k_nope and the caller's V view.
    """
    assert k.is_contiguous()
    assert k.shape == (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1])
    rope = k_pe.expand(*k_nope.shape[:-1], k_pe.shape[-1])
    bits = torch.uint8 if k.element_size() == 1 else torch.int16
    _concat_mla_k_kernel[(triton.cdiv(k.numel(), 1024),)](
        k_nope.view(bits),
        rope.view(bits),
        k.view(bits),
        k.numel(),
        k.shape[1],
        k_nope.shape[-1],
        rope.shape[-1],
        *k_nope.stride(),
        *rope.stride(),
        BLOCK=1024,
    )
