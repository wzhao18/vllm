# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The K3.1 DSA query of every gathered head, computed on every rank (SP decode:
TP4 = DCP4, 24 of 96 heads per rank; ``dsa.sp``).

Under sequence parallelism every rank of the TP group projects all R gathered rows
(the SP block's in-projection), so each rank already holds every row's q_a. The
sparse decode of this rank's KV shard needs the FP8 query of all 96 heads (128 padded
for trtllm-gen), which MlaQuery builds as 24 heads per rank and multicasts into every
rank's q mailbox (DCP q publish), read back after QueryWait polls the other ranks'
publishes. LocalQuery instead projects all 96 heads here, from the gathered weights:

  q_b       q_c [R, 1536] @ W_qb[96 x 192, 1536]^T   (``gemm.decode`` tactic per R)
  W_UK      bmm(q_nope [96, R, 128], W_UK_T [96, 128, 512]) -> [96, R, 512] BF16
  pack      cat(latent, q_pe) * (1 / q_scale), clamped, E4M3 RN -> [R, 128, 576]
            (heads 96..127 stay zero)
  KV insert RMSNorm(kv_a) and concat_and_cache_mla of this rank's slots (stock ops)

No NVLink bytes and no cross-rank wait: the publish (R x 24 x 576 B to each of 3
peers per layer), the QueryWait poll and DcpValue's q mailbox re-arm all drop out, for
4x the q_b / W_UK FLOPs and their weights (+52 MB per layer on each rank).

Precision is the stock decode's: the same q_c rows (QueryNorm), BF16 q_b output with
FP32 accumulation (K order may differ), the stock BF16 W_UK bmm, and the stock decode
query quantization (``flashinfer_mla_sparse.decode_query_quant_pad``: x * (1 / scale),
clamped to +-448, round to nearest); the KV row is the stock insert's."""

import torch

from vllm.triton_utils import tl, triton

from ...shapes import Unsupported
from ..gemm.decode import DecodeProjection
from ..gemm.sp_policy import sp_projection_tactics

QK_NOPE, QK_ROPE, LATENT = 128, 64, 512
QK = QK_NOPE + QK_ROPE  # 192 Q_B rows per head
ROW = LATENT + QK_ROPE  # 576 FP8 bytes per query head


@triton.jit
def _pack_query_kernel(
    latent_ptr,  # [H, R, 512] BF16
    q_ptr,  # [R, H * 192] BF16 (q_pe at columns 128..191 of each head)
    scale_ptr,
    out_ptr,  # [R, OUT_HEADS, 576] FP8
    rows,
    q_stride,
    HEADS: tl.constexpr,
    OUT_HEADS: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
):
    """One row's HEADS_PER_PROGRAM heads of cat(latent, q_pe) to E4M3 with the
    static per-tensor scale (the stock decode query quantization's arithmetic)."""
    row = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * HEADS_PER_PROGRAM + tl.arange(0, HEADS_PER_PROGRAM)
    inv_scale = tl.fdiv(1.0, tl.load(scale_ptr).to(tl.float32), ieee_rounding=True)
    fp8_max: tl.constexpr = 448.0
    out_row = out_ptr + (row * OUT_HEADS + heads[:, None]) * 576
    cols = tl.arange(0, 512)[None, :]
    x = tl.load(latent_ptr + (heads[:, None] * rows + row) * 512 + cols).to(tl.float32)
    x = tl.minimum(tl.maximum(x * inv_scale, -fp8_max), fp8_max)
    tl.store(out_row + cols, x.to(out_ptr.dtype.element_ty))
    cols = tl.arange(0, 64)[None, :]
    x = tl.load(q_ptr + row * q_stride + heads[:, None] * 192 + 128 + cols)
    x = tl.minimum(tl.maximum(x.to(tl.float32) * inv_scale, -fp8_max), fp8_max)
    tl.store(out_row + 512 + cols, x.to(out_ptr.dtype.element_ty))


def pack_query(latent, q, scale, out):
    """``latent`` [H, R, 512] BF16, ``q`` [R, H * 192] BF16 -> ``out`` [R, P, 576]
    E4M3 heads 0..H-1 (P >= H padded heads, untouched)."""
    heads, rows, _ = latent.shape
    assert heads % 8 == 0 and out.shape[1] >= heads and q.stride(1) == 1
    _pack_query_kernel[(rows, heads // 8)](
        latent,
        q,
        scale,
        out,
        rows,
        q.stride(0),
        HEADS=heads,
        OUT_HEADS=out.shape[1],
        HEADS_PER_PROGRAM=8,
        num_warps=4,
    )
    return out


def gather_rank_major(weight, size, group):
    """Every rank's ``weight`` [H, ...] -> [size * H, ...], block d = rank d's
    (collective over ``group``; ``size`` 1: the weight itself)."""
    weight = weight.contiguous()
    if size == 1:
        return weight
    out = weight.new_empty((size * weight.shape[0], *weight.shape[1:]))
    torch.distributed.all_gather_into_tensor(out, weight, group=group)
    return out


class LocalBuffers:
    """The rows every SP DSA layer's LocalQuery shares (they run one at a time; a
    layer's side stream starts after the previous layer's sparse decode in stream
    order): q [cap, G x 192], latent [G, R, 512] (contiguous per R), kv_c [cap, 512] BF16 and the FP8
    query [cap, q_heads, 576] (padding heads zero)."""

    def __init__(self, device, cap, gathered, q_heads):
        bf16 = dict(dtype=torch.bfloat16, device=device)
        self.q = torch.empty((cap, gathered * QK), **bf16)
        self.gathered = gathered
        self.flat = torch.empty(gathered * cap * LATENT, **bf16)  # [G, R, 512] per R
        self.kv = torch.empty((cap, LATENT), **bf16)
        self.query = torch.zeros(
            (cap, q_heads, ROW), dtype=torch.float8_e4m3fn, device=device
        )


_BUFFERS: dict = {}


def local_buffers(device, cap, gathered, q_heads) -> LocalBuffers:
    key = device, cap, gathered, q_heads
    if key not in _BUFFERS:
        _BUFFERS[key] = LocalBuffers(device, cap, gathered, q_heads)
    return _BUFFERS[key]


class LocalQuery:
    """One SP DSA layer's query of all ``size`` x H gathered heads on this rank
    (module docstring). Built collectively over the TP (= DCP) ``group`` (the q_b
    and W_UK gathers), in layer order on every rank. ``rows``: the row counts it
    serves (its q_b tactics)."""

    def __init__(self, attention, group, size, rows, ctx, q_heads):
        wrapper = attention.mla_attn
        layer = wrapper.mla_attn
        heads = layer.W_UK_T.shape[0]
        qb = wrapper.q_b_proj.weight
        if qb.dtype != torch.bfloat16 or tuple(qb.shape) != (heads * QK, 1536):
            raise Unsupported(
                "dsa", None, f"LocalQuery: q_b weight {tuple(qb.shape)} {qb.dtype}"
            )
        uk = layer.W_UK_T  # [H, 128, 512]
        if tuple(uk.shape) != (heads, QK_NOPE, LATENT):
            raise Unsupported("dsa", None, f"LocalQuery: W_UK_T {tuple(uk.shape)}")
        self.heads, self.gathered = heads, size * heads
        self.rows = tuple(sorted(set(rows)))
        self.qb = gather_rank_major(qb, size, group)  # [G * 192, 1536]
        self.uk = gather_rank_major(uk, size, group)  # [G, 128, 512]
        n, k = self.qb.shape
        tactics = sp_projection_tactics("q_b", n, k, self.rows, ctx)
        self.projection = DecodeProjection(self.qb, self.rows, tactics, family="q_b")
        device = qb.device
        self.buffers = local_buffers(device, max(self.rows), self.gathered, q_heads)
        self.projection.check_operands(
            torch.empty((max(self.rows), k), dtype=qb.dtype, device=device),
            self.buffers.q,
        )
        self.q_scale = layer._q_scale.float().reshape(1).to(device)
        self.k_scale = layer._k_scale.float().reshape(1).to(device)
        norm = wrapper.kv_a_layernorm
        self.kv_norm, self.kv_eps = norm.weight, float(norm.variance_epsilon)

    @property
    def query(self):
        return self.buffers.query

    def run(self, qc, projected, cache, slots):
        """``qc`` [R, 1536] the normalized q_a rows, ``projected`` [R, pitch] the
        packed in-projection rows -> the FP8 query rows ``query[:R]`` of every
        gathered head, and this rank's KV rows inserted at ``slots``."""
        from vllm import _custom_ops as ops

        rows, b = qc.shape[0], self.buffers
        q = self.projection.run(qc, b.q[:rows])
        nope = q.view(rows, self.gathered, QK)[..., :QK_NOPE].transpose(0, 1)
        latent = b.flat[: self.gathered * rows * LATENT].view(self.gathered, rows, -1)
        torch.bmm(nope, self.uk, out=latent)
        pack_query(latent, q, self.q_scale, b.query[:rows])
        kv = b.kv[:rows]
        ops.rms_norm(kv, projected[:, 1536:2048], self.kv_norm, self.kv_eps)
        k_pe = projected[:, 2048:2112]
        ops.concat_and_cache_mla(kv, k_pe, cache, slots[:rows], "fp8", self.k_scale)
        return b.query[:rows]
