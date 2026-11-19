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
  W_UK      q_nope [R, 96, 128] W_UK_T [96, 128, 512] -> BF16 latent, then
  + pack    cat(latent, q_pe) * (1 / q_scale), clamped, E4M3 RN -> [R, 128, 576]
            (heads 96..127 stay zero): one tcgen05 kernel (dsa.query_uk), the
            BF16 latent never leaves the SM
  KV insert RMSNorm(kv_a) and concat_and_cache_mla of this rank's slots (stock ops)

No NVLink bytes and no cross-rank wait: the publish (R x 24 x 576 B to each of 3
peers per layer), the QueryWait poll and DcpValue's q mailbox re-arm all drop out, for
4x the q_b / W_UK FLOPs and their weights (+52 MB per layer on each rank).

Precision is the stock decode's: the same q_c rows (QueryNorm), BF16 q_b output with
FP32 accumulation (K order may differ), the stock BF16 W_UK bmm, and the stock decode
query quantization (``flashinfer_mla_sparse.decode_query_quant_pad``: x * (1 / scale),
clamped to +-448, round to nearest); the KV row is the stock insert's."""

import torch

from ...shapes import Unsupported
from ..gemm.decode import DecodeProjection
from ..gemm.sp_policy import sp_projection_tactics
from .query_uk import query_uk

QK_NOPE, QK_ROPE, LATENT = 128, 64, 512
QK = QK_NOPE + QK_ROPE  # 192 Q_B rows per head
ROW = LATENT + QK_ROPE  # 576 FP8 bytes per query head


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
    order): q [cap, G x 192], kv_c [cap, 512] BF16 and the FP8
    query [cap, q_heads, 576] (padding heads zero)."""

    def __init__(self, device, cap, gathered, q_heads):
        bf16 = dict(dtype=torch.bfloat16, device=device)
        self.q = torch.empty((cap, gathered * QK), **bf16)
        self.gathered = gathered
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
        # [G, 512, 128]: W_UK_T K-major (query_uk's B operand)
        self.uk_t = gather_rank_major(uk, size, group).transpose(1, 2).contiguous()
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

    def run(self, qc, projected, cache, slots, insert=True):
        """``qc`` [R, 1536] the normalized q_a rows, ``projected`` [R, pitch] the
        packed in-projection rows -> the FP8 query rows ``query[:R]`` of every
        gathered head, and (``insert``; else QueryNorm did it) this rank's KV rows
        inserted at ``slots``."""
        from vllm import _custom_ops as ops

        rows, b = qc.shape[0], self.buffers
        q = self.projection.run(qc, b.q[:rows])
        query_uk(q, self.uk_t, self.q_scale, b.query[:rows])
        if not insert:
            return b.query[:rows]
        kv = b.kv[:rows]
        ops.rms_norm(kv, projected[:, 1536:2048], self.kv_norm, self.kv_eps)
        k_pe = projected[:, 2048:2112]
        ops.concat_and_cache_mla(kv, k_pe, cache, slots[:rows], "fp8", self.k_scale)
        return b.query[:rows]
