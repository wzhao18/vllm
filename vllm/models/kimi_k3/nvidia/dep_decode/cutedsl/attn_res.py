# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The cluster AttnRes of the latency decode with an MXFP8 copy of its output.

The normalized output feeds an MXFP8 projection (KDA in-proj, MLA q/KV-A, the MoE
front GEMM), which would otherwise quantize it in a separate kernel. Each thread
owns 8 columns of a token row, so one 32-column MX block spans 4 adjacent lanes:
after the stock epilogue stores the BF16 row, each thread re-reads its 8 values,
the 4 lanes reduce the block maximum, and the E4M3 values and UE8M0 scale (in the
F8_128x4 swizzled layout the GEMMs read) are stored. The numerics are those of the
stock MXFP8 quantization of the BF16 output.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Uint8
from cutlass.cute.runtime import make_fake_stream

from ...specialized_decode.cutedsl.attn_res.kernel import ClusterAttnRes
from ...specialized_decode.cutedsl.attn_res.prepared import PreparedAttnRes
from ...specialized_decode.cutedsl.common.host import (
    architecture,
    dynamic_fake,
    dynamic_view,
    fake,
    stream,
)
from ...specialized_decode.cutedsl.common.primitives import (
    at,
    bf16,
    load_words,
    mx_exponent,
    mx_rescale,
    pack_fp8,
    store_words,
)

HIDDEN = 7168
# Swizzled scale tiles along K: 4 blocks of 32 columns each.
K_TILES = HIDDEN // 32 // 4


def scale_numel(rows: int) -> int:
    return -(-rows // 128) * 128 * HIDDEN // 32


class ClusterAttnResMxfp8(ClusterAttnRes):
    """``ClusterAttnRes`` whose extra arguments are the MXFP8 output (E4M3 values
    [T, 7168] and swizzled scales), then the optional aux prefix stream."""

    @cute.jit
    def post(
        self, state, bufs, delta, rows, output, eps: Float32, out_eps: Float32, ids, aux
    ):
        ClusterAttnRes.post(
            self, state, bufs, delta, rows, output, eps, out_eps, ids, aux[2:]
        )
        values, scales = aux[0], aux[1]
        rank, token, tid = ids
        col = (rank * 128 + tid) * 8
        # This thread's own BF16 stores, read back in program order.
        words = load_words(at(output, token, col), count=4, mutable=True)
        x = [bf16(words[i // 2], i % 2) for i in range(8)]
        amax = abs(x[0])
        for i in cutlass.range_constexpr(1, 8):
            amax = cute.arch.fmax(amax, abs(x[i]))
        for shift in cutlass.range_constexpr(2):
            amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, 1 << shift))
        exponent = mx_exponent(amax)
        scale = mx_rescale(exponent)
        packed = (
            pack_fp8((x[0] * scale, x[1] * scale, x[2] * scale, x[3] * scale)),
            pack_fp8((x[4] * scale, x[5] * scale, x[6] * scale, x[7] * scale)),
        )
        store_words(at(values, token, col), packed)
        if tid % 4 == 0:
            block = col // 32
            offset = (
                (((token // 128) * K_TILES + block // 4) * 32 + token % 32) * 4
                + (token // 32) % 4
            ) * 4 + block % 4
            scales[offset] = Uint8(exponent)


@cache
def compile_mxfp8_consumer(nb, delta, write, final, arch, aux=False):
    row, weight, eps = dynamic_fake(2), fake((HIDDEN,), (1,)), cutlass.Float32(1e-6)
    kernel = ClusterAttnResMxfp8(nb, slots=0, has_delta=delta, write=write, final=final)
    values = dynamic_fake(2, cutlass.Float8E4M3FN)
    scales = fake((cute.sym_int(),), (1,), Uint8, align=1)
    args = (row, row, dynamic_fake(3), weight, weight, weight, row, eps, eps)
    args += (values, scales) + (row,) * aux
    return cute.compile(kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}")


class Mxfp8AttnRes(PreparedAttnRes):
    """A ``PreparedAttnRes`` (tensor delta input only) that also writes the MXFP8
    copy of its output."""

    def _compile(self, slots, cap, aux=False):
        assert slots == 0, "the DEP decode has no mailbox input"
        arch = architecture(self.device)
        return compile_mxfp8_consumer(*self.key, arch, aux)

    def run(self, delta, prefix, blocks, out, values, scales, aux=None):
        code = self.code if aux is None else self.aux[0]
        rows = (prefix, blocks, out, values) + (() if aux is None else (aux,))
        prefix, blocks, view, values, *aux = map(dynamic_view, rows)
        scales = dynamic_view(scales, align=1)
        args = *self.views, view, *self.eps, values, scales, *aux
        code(dynamic_view(delta), prefix, blocks, *args, stream(self.device))
        return out


def mxfp8_buffers(rows: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 values and swizzled scales of an MXFP8 [rows, 7168] activation."""
    values = torch.empty(rows, HIDDEN, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty(scale_numel(rows), dtype=torch.uint8, device=device)
    return values, scales
