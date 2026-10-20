# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the CuTe KDA RecoverSSM verify and state commit."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    architecture,
    dynamic_fake,
    dynamic_view,
    stream,
)

from .cutedsl.kda_commit import KdaCommit
from .cutedsl.kda_verify import KdaChunkedVerify


@cache
def _compile(heads: int, window: int, arch: str):
    kernel = KdaChunkedVerify(heads, window)
    bf16, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    fakes = (
        dynamic_fake(2, bf16),  # qkv
        dynamic_fake(2, bf16),  # raw_g
        dynamic_fake(2, bf16),  # raw_beta
        dynamic_fake(1, f32),  # a_log
        dynamic_fake(1, f32),  # dt_bias
        dynamic_fake(4, bf16),  # state
        dynamic_fake(4, f32),  # corrections
        dynamic_fake(4, bf16),  # key / gate records
        dynamic_fake(2, bf16),  # out
        dynamic_fake(1, i32),  # query_start
        dynamic_fake(1, i32),  # slots
    )
    return cute.compile(
        kernel,
        *fakes,
        cutlass.Float32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options=f"--gpu-arch={arch}",
    )


def kda_cute_verify(
    mixed_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    kg_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """``kda_recoverssm_verify`` on 2-D views: ``mixed_qkv`` [T, 3 * H * 128] (the
    convolution output), ``raw_g`` [T, H * 128], ``raw_beta`` [T, H], ``out``
    [T, H * 128]."""
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    heads = A_log.numel()
    code = _compile(heads, spec_query_len, architecture(mixed_qkv.device))
    tensors = (
        mixed_qkv,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        checkpoint_state,
        correction_cache,
        kg_cache,
        out,
        query_start_loc,
        state_indices,
    )
    code(
        *map(dynamic_view, tensors),
        cutlass.Float32(lower_bound),
        cutlass.Int32(NULL_BLOCK_ID),
        stream(mixed_qkv.device),
    )
    return out


@cache
def _compile_commit(heads: int, window: int, arch: str, param_dtype: torch.dtype):
    kernel = KdaCommit(heads, window)
    i32, i64 = torch.int32, torch.int64
    fakes = (
        dynamic_fake(1, i64),  # state base addresses
        dynamic_fake(1, i64),  # correction base addresses
        dynamic_fake(1, i64),  # key / gate base addresses
        dynamic_fake(1, i64),  # state slot strides
        dynamic_fake(1, i64),  # correction slot strides
        dynamic_fake(1, i64),  # key / gate slot strides
        dynamic_fake(2, param_dtype),  # a_log
        dynamic_fake(3, param_dtype),  # dt_bias
        dynamic_fake(1, i32),  # source slots
        dynamic_fake(1, i32),  # commit lengths
        dynamic_fake(1, i32),  # destination slots
    )
    return cute.compile(
        kernel,
        *fakes,
        cutlass.Float32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options=f"--gpu-arch={arch}",
    )


def kda_cute_commit(context, state_indices: torch.Tensor, batch: int) -> None:
    """``KDARecoverSSMCommitContext.state_commit``: the state fold of a prepared
    commit plan (non-aligned mode)."""
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    state = context.checkpoints[0]
    _, heads, value_dim, key_dim = state.shape
    window = context.spec_query_len
    assert (value_dim, key_dim) == (128, 128) and context.lower_bound is not None
    assert state.stride()[1:] == (value_dim * key_dim, key_dim, 1)
    assert context.correction_caches[0].stride()[1:] == (
        window * value_dim,
        value_dim,
        1,
    )
    assert context.kg_caches[0].stride()[1:] == (window * 2 * key_dim, 2 * key_dim, 1)
    assert context.A_log.dtype == context.dt_bias.dtype
    code = _compile_commit(
        heads, window, architecture(state.device), context.A_log.dtype
    )
    tensors = (
        context.state_base_addrs,
        context.correction_cache_base_addrs,
        context.kg_cache_base_addrs,
        context.state_block_strides,
        context.correction_cache_block_strides,
        context.kg_cache_block_strides,
        context.A_log,
        context.dt_bias,
        state_indices[:batch],
        context.commit_lens[:batch],
        context.final_state_indices[:batch],
    )
    code(
        *map(dynamic_view, tensors),
        cutlass.Float32(context.lower_bound),
        cutlass.Int32(NULL_BLOCK_ID),
        stream(state.device),
    )


def dep_kda_verify(
    layer,
    mixed_qkv,
    raw_g,
    raw_beta,
    checkpoint_state,
    records,
    query_start_loc,
    state_indices,
    out,
):
    """``KimiK3DeltaAttention.dep_verify``: the RecoverSSM verify of ``layer`` on its
    stock views (``raw_g`` [1, T, H, K], ``raw_beta`` [1, T, H], ``out`` [1, T, H, K]
    or None); returns ``out``."""
    tokens = mixed_qkv.shape[0]
    if out is None:
        out = mixed_qkv.new_empty(1, tokens, layer.local_num_heads, layer.head_dim)
    kda_cute_verify(
        mixed_qkv,
        raw_g.view(tokens, -1),
        raw_beta.view(tokens, -1),
        layer.A_log,
        layer.dt_bias,
        layer.gate_lower_bound,
        checkpoint_state,
        records[0],
        records[1],
        query_start_loc,
        state_indices,
        layer.spec_query_len,
        out.view(tokens, -1),
    )
    return out


def install_dep_kda(layer) -> None:
    """Route ``layer``'s RecoverSSM verify through the CuTe kernel (compiled here,
    outside graph capture)."""
    assert layer.A_log.dtype == layer.dt_bias.dtype == torch.float32
    _compile(
        layer.local_num_heads, layer.spec_query_len, architecture(layer.A_log.device)
    )
    layer.dep_verify = dep_kda_verify


def install_state_commit(context) -> None:
    """Fold RecoverSSM commits with the CuTe kernel (compiled here, outside graph
    capture)."""
    state = context.checkpoints[0]
    _compile_commit(
        state.shape[1],
        context.spec_query_len,
        architecture(state.device),
        context.A_log.dtype,
    )
    context.state_commit = kda_cute_commit
