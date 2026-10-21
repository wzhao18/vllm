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
    fake,
    stream,
)

from .cutedsl.kda_commit import KdaCommit
from .cutedsl.kda_verify import DEFAULT_STAGES, DEFAULT_WARPS, KdaChunkedVerify


@cache
def _compile_variant(
    heads: int,
    window: int,
    arch: str,
    stages: int = DEFAULT_STAGES,
    conv: bool = False,
    norm_dtype: torch.dtype | None = None,
    warps: int = DEFAULT_WARPS,
):
    kernel = KdaChunkedVerify(
        heads, window, stages, conv, norm=norm_dtype is not None, warps=warps
    )
    bf16, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    fakes = (
        dynamic_fake(2, bf16),  # qkv (raw projection when conv)
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
    if conv:
        fakes += (
            dynamic_fake(3, bf16),  # conv state [slots, Q + 2, 3HK]
            dynamic_fake(2, bf16),  # conv weight [3HK, 4]
            dynamic_fake(1, i32),  # accepted tokens
        )
    scalars = (cutlass.Float32(0), cutlass.Int32(0))
    if norm_dtype is not None:
        fakes += (
            dynamic_fake(2, bf16),  # g_proj
            dynamic_fake(1, norm_dtype),  # norm weight
            dynamic_fake(2, cutlass.Uint8),  # E4M3 values (bytes)
            fake((cute.sym_int(),), (1,), cutlass.Uint8, align=1),  # scales
        )
        scalars += (cutlass.Float32(0),)  # norm eps
    return cute.compile(
        kernel,
        *fakes,
        *scalars,
        make_fake_stream(),
        options=f"--gpu-arch={arch}",
    )


def _compile(heads: int, window: int, arch: str):
    """Compile the verify variants a decode step launches (plain, and the tcgen05 one
    with the fused convolution), outside graph capture; returns the plain one."""
    _compile_tc(heads, window, arch, _tc_ctas(torch.cuda.current_device()))
    return _compile_variant(heads, window, arch)


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
    conv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    norm: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]
    | None = None,
) -> torch.Tensor:
    """``kda_recoverssm_verify`` on 2-D views: ``mixed_qkv`` [T, 3 * H * 128] (the
    convolution output), ``raw_g`` [T, H * 128], ``raw_beta`` [T, H], ``out``
    [T, H * 128]. With ``conv`` = (conv state [slots, Q + 2, 3 * H * 128] token
    major, conv weight [3 * H * 128, 4], accepted tokens [B]), ``mixed_qkv`` is the
    projection and the kernel applies ``causal_conv1d_update`` (SiLU, no bias,
    speculative rolling) first. With ``norm`` = (g_proj [T, H * 128], norm weight
    [128], E4M3 values [T, H * 128], swizzled scales, eps), the output is KDA's
    gated RMSNorm of the result in MXFP8 (``out`` is not written)."""
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    heads = A_log.numel()
    device = mixed_qkv.device
    if conv is not None and norm is None:
        # The fused convolution runs on the tcgen05 verify (faster at every batch).
        code = _compile_tc(
            heads, spec_query_len, architecture(device), _tc_ctas(device)
        )
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
        ) + conv
        code(
            *map(dynamic_view, tensors),
            cutlass.Float32(lower_bound),
            cutlass.Int32(NULL_BLOCK_ID),
            stream(device),
        )
        return out
    code = _compile_variant(
        heads,
        spec_query_len,
        architecture(mixed_qkv.device),
        conv=conv is not None,
        norm_dtype=None if norm is None else norm[1].dtype,
    )
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
    ) + (conv or ())
    views = list(map(dynamic_view, tensors))
    scalars = [cutlass.Float32(lower_bound), cutlass.Int32(NULL_BLOCK_ID)]
    if norm is not None:
        g_proj, weight, values, scales, eps = norm
        views += [
            dynamic_view(g_proj),
            dynamic_view(weight),
            dynamic_view(values.view(torch.uint8)),
            dynamic_view(scales.view(torch.uint8), align=1),
        ]
        scalars.append(cutlass.Float32(eps))
    code(*views, *scalars, stream(mixed_qkv.device))
    return out


_compiled_commits: set = set()


@cache
def _compile_commit(heads: int, window: int, arch: str, param_dtype: torch.dtype):
    _compiled_commits.add((heads, window, arch, param_dtype))
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


def compile_state_commit(module, window: int) -> None:
    """Compile the RecoverSSM state commit for ``module``'s KDA layers (outside
    graph capture)."""
    _compile_commit(
        module.local_num_heads,
        window,
        architecture(module.A_log.device),
        module.A_log.dtype,
    )


def bind_state_commit(context) -> bool:
    """Fold the RecoverSSM commits of ``context`` with the CuTe kernel (compiled by
    ``compile_state_commit``). Returns whether it is bound."""
    if context.state_commit is not None:
        return True
    state = context.checkpoints[0]
    key = (
        state.shape[1],
        context.spec_query_len,
        architecture(state.device),
        context.A_log.dtype,
    )
    if key not in _compiled_commits:
        return False  # never compile under capture: keep the stock commit
    context.state_commit = kda_cute_commit
    return True


class KdaOutProj:
    """KDA's gated output norm, MXFP8 quantization and ``o_proj`` for
    decode-sized batches: one fused norm + quantize kernel, then the stream-K GEMM
    (``KimiK3DeltaAttention.dep_out_proj``)."""

    def __init__(self, layer):
        from .linear import fp8_rows

        rows = fp8_rows(layer.o_proj)
        assert rows is not None and layer.o_norm.activation == "sigmoid"
        self.weight, self.scale = rows
        self.norm = layer.o_norm

    @staticmethod
    def supports(num_tokens: int) -> bool:
        from .gemm import MAX_TILE_N

        return 0 < num_tokens <= MAX_TILE_N

    def __call__(self, core_attn_out: torch.Tensor, gate: torch.Tensor):
        from .elementwise import gated_rms_norm_mxfp8
        from .gemm import streamk_mxfp8_gemm

        xq, xs = gated_rms_norm_mxfp8(
            core_attn_out, gate, self.norm.weight, self.norm.eps
        )
        return streamk_mxfp8_gemm(xq, xs, self.weight, self.scale)


class MlaOutProj:
    """MLA's sigmoid output gate, MXFP8 quantization and ``o_proj`` for
    decode-sized batches (``MultiHeadLatentAttention.dep_out_proj``)."""

    def __init__(self, layer):
        from .linear import fp8_rows

        rows = fp8_rows(layer.o_proj)
        assert rows is not None
        self.weight, self.scale = rows

    supports = staticmethod(KdaOutProj.supports)

    def __call__(self, attn_out: torch.Tensor, gate: torch.Tensor):
        from .elementwise import sigmoid_gate_mxfp8
        from .gemm import streamk_mxfp8_gemm

        xq, xs = sigmoid_gate_mxfp8(attn_out, gate)
        return streamk_mxfp8_gemm(xq, xs, self.weight, self.scale)


@cache
def _compile_tc(heads: int, window: int, arch: str, ctas: int):
    """The tcgen05 verify with the fused convolution (``cutedsl/kda_verify_tc.py``)."""
    from .cutedsl.kda_verify_tc import KdaTcVerify

    bf16, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    fakes = (
        dynamic_fake(2, bf16),  # raw qkv projection
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
        dynamic_fake(3, bf16),  # conv state [slots, Q + 2, 3HK]
        dynamic_fake(2, bf16),  # conv weight [3HK, 4]
        dynamic_fake(1, i32),  # accepted tokens
    )
    return cute.compile(
        KdaTcVerify(heads, window, conv=True, ctas=ctas),
        *fakes,
        cutlass.Float32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options=f"--gpu-arch={arch}",
    )


def _tc_ctas(device) -> int:
    from .cutedsl.kda_verify_tc import CTAS_PER_SM

    return CTAS_PER_SM * torch.cuda.get_device_properties(device).multi_processor_count
