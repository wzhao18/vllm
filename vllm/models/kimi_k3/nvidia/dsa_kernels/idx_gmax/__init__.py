# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA decode: the indexer's paged MQA logits with the fused top-k producer.

DeepGEMM's SM100 paged MQA logits kernel (vendored core, csrc/mqa_logits_gmax.cuh) that also
writes, per row, the max order-preserving key of every aligned group of 8 logits (``gmax``)
and the row's NaN count. The DCP top-k (dcp_exchange topk_v2) derives its candidate
threshold from ``gmax`` and reads only the groups that can hold a candidate, instead of
copying and sampling the whole logits row.
"""

import hashlib
import os
from pathlib import Path

import torch

_EXT = None
SPLIT_STRIDE = 1536  # DeepGEMM SM100 logits row-stride alignment (lcm(384, 512) elements)


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load

    import vllm.third_party.deep_gemm as dg

    inc = Path(dg.__file__).parent / "include"
    src = Path(__file__).parent / "csrc" / "idx_gmax.cu"
    h = hashlib.sha1()
    for f in sorted(src.parent.iterdir()):
        if f.suffix in (".cu", ".cuh"):
            h.update(f.read_bytes())
    digest = h.hexdigest()[:10]
    _EXT = load(
        name=f"k31_idx_gmax_{digest}",
        sources=[str(src)],
        extra_include_paths=[str(inc)],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_100f,code=sm_100f",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--ptxas-options=--register-usage-level=10",
            "--diag-suppress=39,161,174,177,186,940",
            "-Xcompiler=-fconcepts",
        ],
        extra_cflags=["-O3", "-std=c++20"],
    )
    return _EXT


def logits_cols(max_model_len: int) -> int:
    return (max_model_len + SPLIT_STRIDE - 1) // SPLIT_STRIDE * SPLIT_STRIDE


def gmax_cols(max_model_len: int) -> int:
    return logits_cols(max_model_len) // 8


def paged_logits_gmax(
    q: tuple[torch.Tensor, torch.Tensor],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    indices: torch.Tensor,
    gmax: torch.Tensor,
    nan_count: torch.Tensor,
) -> torch.Tensor:
    """deep_gemm.fp8_fp4_paged_mqa_logits (SM100 varlen) plus gmax / NaN counts.

    ``gmax``: int16 [>= rows, >= gmax_cols(max_model_len)] (16-byte rows); ``nan_count``:
    int32 [>= rows], zero on entry (the top-k consumer re-zeroes it). Returns the bf16
    logits [rows, max_model_len] (row stride logits_cols(max_model_len)).
    """
    qv, qs = q
    fp4 = qv.shape[-1] == 64  # packed MXFP4 (head dim 128) vs FP8 (128 bytes)
    if qv.dtype not in (torch.int8, torch.uint8):
        qv = qv.view(torch.uint8)
    rows = qv.shape[0]
    if block_table.stride(-1) != 1:
        block_table = block_table.clone(memory_format=torch.contiguous_format)
    cols = logits_cols(max_model_len)
    logits = torch.empty((rows, cols), dtype=torch.bfloat16, device=qv.device)
    _load_ext().paged_logits_gmax(
        qv,
        qs,
        kv_cache,
        weights,
        context_lens,
        block_table,
        indices,
        schedule_metadata,
        logits,
        gmax,
        nan_count,
        fp4,
    )
    return logits[:, :max_model_len]
