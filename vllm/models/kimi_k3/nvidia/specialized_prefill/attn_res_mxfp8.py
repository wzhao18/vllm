# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AttnRes with the next projection's MXFP8 input quantization fused in.

Under the nvfp4_modelopt Kimi K3.1 checkpoint the KDA ``in_proj_qkvgfab`` and
the shared experts' ``gate_up_proj`` are online MXFP8 and read the output of
the layer's AttnRes (pre-attention and post-attention respectively). Without
this kernel every such call re-reads the BF16 AttnRes output in a standalone
``mxfp8_quantize`` pass (16384 x 7168 at DEP16 F16K: 161 extra passes per
step). ``attn_res_mxfp8`` writes the MXFP8 activation (E4M3 values + F8_128x4
swizzled E8M0 scales) from the AttnRes output-RMSNorm epilogue, and the BF16
output only when another consumer needs it (the MoE router / latent down
projection; the KDA in_proj is the only consumer of its layer's AttnRes).

Numerics: the kernel is ``csrc/kimi_k3/attn_res_kernel.cu`` (same source, same
``--use_fast_math`` build) plus the quantization of the BF16-rounded output
with the bit-exact ``mxfp8_quantize`` primitives of ``ops/mxfp8_act.py``, so
the result equals ``mxfp8_quantize(kimi_k3_attn_res(...))`` bit for bit.
"""

import hashlib
import os
import threading

import torch

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.platforms import current_platform

from . import attn_res_dielocal

HIDDEN = 7168
MXFP8_BLOCK = 32
_SRC = os.path.join(os.path.dirname(__file__), "attn_res_mxfp8.cu")
_lock = threading.Lock()
_mods: dict[int, object] = {}


def _module(chunk_depth: int = 2):
    mod = _mods.get(chunk_depth)
    if mod is not None:
        return mod
    with _lock:
        mod = _mods.get(chunk_depth)
        if mod is None:
            from torch.utils.cpp_extension import load

            import vllm.envs as envs

            h = hashlib.sha256()
            with open(_SRC, "rb") as fh:
                h.update(fh.read())
            h.update(f"{torch.__version__} {torch.version.cuda} {chunk_depth}".encode())
            tag = h.hexdigest()[:16]
            build = os.path.join(envs.VLLM_CACHE_ROOT, "k31pf_attn_res_mxfp8", tag)
            os.makedirs(build, exist_ok=True)
            # Same flags as the native kernel (CMakeLists.txt KIMI_K3_ATTN_RES).
            mod = load(
                name=f"k31pf_attn_res_mxfp8_{tag}",
                sources=[_SRC],
                extra_cflags=["-O3"],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    # The native build does not set torch's no-conversion macros.
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    f"-DK31PF_CHUNK_DEPTH={chunk_depth}",
                    "-gencode",
                    "arch=compute_100f,code=sm_100f",
                ],
                build_directory=build,
                verbose=False,
            )
            _mods[chunk_depth] = mod
    return mod


_PF_SRC = os.path.join(os.path.dirname(__file__), "attn_res_pf.cu")
_pf_mod: list[object] = []


def _pf_module():
    """The two-tokens-in-flight AttnRes (attn_res_pf.cu), the production
    kernel of :func:`attn_res_mxfp8`."""
    if _pf_mod:
        return _pf_mod[0]
    with _lock:
        if not _pf_mod:
            from torch.utils.cpp_extension import load

            import vllm.envs as envs

            h = hashlib.sha256()
            with open(_PF_SRC, "rb") as fh:
                h.update(fh.read())
            h.update(f"{torch.__version__} {torch.version.cuda}".encode())
            tag = h.hexdigest()[:16]
            build = os.path.join(envs.VLLM_CACHE_ROOT, "k31pf_attn_res_pf", tag)
            os.makedirs(build, exist_ok=True)
            _pf_mod.append(
                load(
                    name=f"k31pf_attn_res_pf_{tag}",
                    sources=[_PF_SRC],
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=[
                        "-O3",
                        "--use_fast_math",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-gencode",
                        "arch=compute_100f,code=sm_100f",
                    ],
                    build_directory=build,
                    verbose=False,
                )
            )
    return _pf_mod[0]


def supported(prefix: torch.Tensor, num_blocks: int, has_output_norm: bool) -> bool:
    """Capability of :func:`attn_res_mxfp8` (the native AttnRes contract)."""
    return (
        has_output_norm
        and prefix.dim() == 2
        and prefix.shape[1] == HIDDEN
        and prefix.dtype == torch.bfloat16
        and prefix.stride(0) == HIDDEN
        and 0 <= num_blocks <= 8
        and current_platform.is_device_capability_family(100)
    )


# Per-shape policy: prefill chunks (the decode steps keep the native AttnRes;
# the specialized decode backend has its own AttnRes path).
MIN_ROWS = 1024


def use_for(prefix: torch.Tensor) -> bool:
    """Policy + capability: run the fused kernel for this AttnRes call."""
    return prefix.shape[0] >= MIN_ROWS and supported(prefix, 0, True)


def ring_depth(num_blocks: int) -> int:
    """Per-shape row-load ring depth (measured at 16384 / 8192 rows on VR:
    depth 3 wins from 6 source blocks, depth 2 below)."""
    return 3 if num_blocks >= 6 else 2


def warmup() -> None:
    _pf_module()


def attn_res_mxfp8(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
    bf16_out: bool,
    chunk_depth: int | None = None,
    prefix_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, QuantizedActivation]:
    """``kimi_k3_attn_res`` (output RMSNorm applied) as MXFP8, and as BF16
    when ``bf16_out``. Returns ``(bf16 or None, QuantizedActivation)``."""
    rows = prefix.shape[0]
    padded_cols = HIDDEN // MXFP8_BLOCK
    q = torch.empty((rows, HIDDEN), dtype=torch.float8_e4m3fn, device=prefix.device)
    sf = torch.empty(
        (max(rows, 1) + 127) // 128 * 128 * padded_cols,
        dtype=torch.uint8,
        device=prefix.device,
    )
    out = torch.empty_like(prefix) if bf16_out else None
    if chunk_depth is None:
        _pf_module().attn_res_pf(
            prefix,
            delta,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            out,
            q.view(torch.uint8),
            sf,
            num_blocks,
            block_write_idx,
            float(eps),
            float(output_norm_eps),
            *attn_res_dielocal.kernel_args(blocks),
            prefix_out,
        )
        return out, QuantizedActivation(
            data=q,
            scale=sf,
            orig_dtype=prefix.dtype,
            orig_shape=torch.Size((rows, HIDDEN)),
            quant_key=kMxfp8Dynamic,
        )
    # Explicit ring depth: the previous one-token-per-SM kernel (bench / test
    # reference only).
    _module(chunk_depth).attn_res_mxfp8(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        q.view(torch.uint8),
        sf,
        num_blocks,
        block_write_idx,
        float(eps),
        float(output_norm_eps),
    )
    qa = QuantizedActivation(
        data=q,
        scale=sf,
        orig_dtype=prefix.dtype,
        orig_shape=torch.Size((rows, HIDDEN)),
        quant_key=kMxfp8Dynamic,
    )
    return out, qa


def attn_res_bf16(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
    prefix_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """The native kimi_k3_attn_res contract (BF16 output, optional output
    RMSNorm) on the prefill kernel: bit-identical results; die-local block
    reads when blocks is the die-local buffer."""
    out = torch.empty_like(prefix)
    _pf_module().attn_res_pf(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        None,
        None,
        num_blocks,
        block_write_idx,
        float(eps),
        float(output_norm_eps),
        *attn_res_dielocal.kernel_args(blocks),
        prefix_out,
    )
    return out
