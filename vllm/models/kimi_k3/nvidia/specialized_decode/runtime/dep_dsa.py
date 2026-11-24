# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSA layer of the SP decode under DSA-DP (``sp_attention.SpStep.run_dsa_dp``, the
attention of ``ops.dsa_dp.run_segment``): the stock ``KimiK3DSAAttention`` forward
(``MultiHeadLatentAttentionWrapper.forward`` on the decode rows of this card's DEP
segment, all 96 heads) with its projections packed
as the stock fused DSA forward packs them (``KimiK3DSAAttention.pack_projections``,
which needs an fp8_ds_mla cache and so does not apply to this per-tensor FP8 one):

  in-projection   [q_a | kv_a | wk | weights_proj]  one BF16 GEMM  7168 -> 2272
                  (``DecodeProjection``: warp / cluster split-K, FP32 sum, one
                  rounding; VR200 1 GPU cold, us at 6 / 12 / 24 / 36 / 48 rows vs
                  cuBLAS + its split-K reduce: 4.6 / 5.3 / 5.9 / 7.6 / 7.7 vs 6.9 /
                  7.6 / 8.2 / 8.8 / 8.7)
  query           [q_b | wq_index]                  one BF16 GEMM  1536 -> 22528
  output gate     g_proj                            on the layer's aux stream (``GATE``)
  front-end       q_a / kv_a RMSNorm (one kernel, ``fused_q_kv_rmsnorm``), the indexer
                  key norm + quant + insert (``_fused_k_insert``), the indexer score and
                  top-k, the MLA attention (``MLAAttention``: KV insert, W_UK, the
                  query's concat + FP8 quant + head pad in one kernel where the
                  backend fuses it, sparse core, W_UV)
  output          sigmoid gate (one kernel: ``prefill_glue.gate_sigmoid_mul``, bitwise
                  the stock ``attn * gate.sigmoid()``, also below its prefill row
                  policy), o_proj

Every op is the stock op the wrapper forward calls, with the same inputs; the packed
GEMMs compute the same products (cuBLAS, another tile / split-K choice: FP32
reduction order only). The linears become views of the packed weights (as
``pack_projections``), so a weight reload refills them in place and the stock forward
of non-decode steps reads the same storage. 96 x 192 + 32 x 128 query columns, 2272
in-projection columns: no extra weight memory.

Per decode step this reads the DSA-DP weights of all 96 heads (q_b, kv_b, g_proj,
o_proj four times the TP4 shard: about 3.1x the TP4 DSA layer's bytes); the DSA layer
issues no collective."""

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.common.ops import fused_q_kv_rmsnorm
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel

from ...ops import prefill_glue
from ..cutedsl.gemm.die_local import attach_die_local, run_die_local
from ..shapes import Unsupported

# Where the output-gate GEMM (all heads: 176 MB of BF16 weight) runs: "late" on the
# layer's aux stream once the query projection is issued, concurrent with the
# latency-bound indexer / sparse core (not with the two bandwidth-bound projections);
# "early" on the aux stream from the layer start (as the stock fused DSA forward);
# "packed" in the in-projection GEMM. VR200, 24-layer model, single node, step us
# (b4 / b8 / b24 / b32 at 131k), early vs packed: 5609 / 6598 / 9500 / 10056 vs
# 5661 / 6656 / 9479 / 10096.
GATE = "late"


def _bf16(linear, where):
    method = getattr(linear, "quant_method", None)
    weight = getattr(linear, "weight", None)
    if not isinstance(method, UnquantizedLinearMethod) or weight is None:
        raise Unsupported("dsa_dep", None, f"{where}: not an unquantized linear")
    if weight.dtype != torch.bfloat16 or weight.dim() != 2:
        raise Unsupported("dsa_dep", None, f"{where}: weight {weight.dtype}")
    if getattr(linear, "bias", None) is not None:
        raise Unsupported("dsa_dep", None, f"{where}: a bias")
    return weight


def pack_linears(linears, where):
    """One packed [sum N, K] BF16 weight of ``linears`` (same K), each linear's
    weight becoming a view of its rows (``pack_projections``' packing)."""
    weights = [_bf16(linear, where) for linear in linears]
    k = weights[0].shape[1]
    if any(w.shape[1] != k for w in weights):
        raise Unsupported(
            "dsa_dep", None, f"{where}: K {[w.shape[1] for w in weights]}"
        )
    packed = torch.cat([w.data for w in weights])
    for linear, view in zip(linears, packed.split([w.shape[0] for w in weights])):
        linear.weight.data = view
    return packed


def _fused_gate_ok(attn, gate):
    """``prefill_glue.use_gate_sigmoid_mul``'s capability checks without its prefill
    row policy (decode rows: one kernel instead of sigmoid + mul)."""
    return (
        attn.dim() == 2
        and gate.shape == attn.shape
        and attn.dtype == gate.dtype == torch.bfloat16
        and attn.stride(1) == 1
        and gate.stride(1) == 1
        and attn.shape[1] % 8 == 0
        and attn.stride(0) % 8 == 0
        and gate.stride(0) % 8 == 0
        and attn.data_ptr() % 16 == 0
        and gate.data_ptr() % 16 == 0
        and attn.is_cuda
    )


class DepDsa:
    """The decode forward of one DSA-DP DSA layer (module docstring). ``attention``:
    the stock KimiK3DSAAttention, built with all heads (``local_full_heads``)."""

    def __init__(self, attention, rows=(), ctx=None, scratch=None):
        """``rows``: the segment row counts the in-projection serves on
        ``DecodeProjection`` (``ctx``: the SP ``shapes.Context``; ``scratch(name,
        shape)``: a shared BF16 buffer); without them it runs on cuBLAS."""
        wrapper, indexer = attention.mla_attn, attention.dsa_indexer
        where = f"DSA layer {wrapper.prefix}"
        checks = (
            (not getattr(attention, "local_full_heads", False), "not all heads"),
            (attention.q_lora_rank is None, "no q-LoRA"),
            (wrapper.rotary_emb is not None, "a rotary MLA (NoPE only)"),
            (indexer.index_rope_dim > 0, "a rotary indexer"),
            (getattr(wrapper, "g_proj", None) is None, "no output gate"),
            (wrapper.dcp_q_replicate, "a DCP-replicated query"),
            (wrapper.skip_topk or not wrapper.is_sparse, "no top-k of its own"),
            (indexer.dcp_world_size != 1, "a DCP indexer"),
            (attention.gemm_rs_ar is not None, "a fused GEMM-RS o_proj"),
            (attention._packed is not None, "the stock packed projections"),
            (attention.o_proj.reduce_results, "a reducing o_proj"),
        )
        cause = next((c for failed, c in checks if failed), None)
        if cause is not None:
            raise Unsupported("dsa_dep", None, f"{where}: {cause}")
        self.attention, self.wrapper, self.indexer = attention, wrapper, indexer
        if GATE not in ("late", "early", "packed"):
            raise Unsupported("dsa_dep", None, f"gate placement {GATE!r}")
        front = [wrapper.fused_qkv_a_proj, indexer.wkv_index]
        if GATE == "packed":
            front.append(wrapper.g_proj)
        self.in_weight = pack_linears(front, where)
        self.q_weight = pack_linears([wrapper.q_b_proj, indexer.wq_index], where)
        _bf16(wrapper.g_proj, where)
        _bf16(wrapper.o_proj, where)
        self.heads = attention.num_local_heads
        self.in_split = [
            attention.q_lora_rank,
            attention.kv_lora_rank,
            attention.qk_rope_head_dim,
            indexer.index_head_dim,
            indexer.index_num_heads,
        ]
        if GATE == "packed":
            self.in_split.append(wrapper.g_proj.weight.shape[0])
        if sum(self.in_split) != self.in_weight.shape[0]:
            raise Unsupported(
                "dsa_dep", None,
                f"{where}: in-projection rows {self.in_weight.shape[0]} != "
                f"{self.in_split}",
            )  # fmt: skip
        self.q_split = [
            self.heads * attention.qk_head_dim,
            indexer.index_num_heads * indexer.index_head_dim,
        ]
        if sum(self.q_split) != self.q_weight.shape[0]:
            raise Unsupported(
                "dsa_dep", None,
                f"{where}: query rows {self.q_weight.shape[0]} != {self.q_split}",
            )  # fmt: skip
        self.aux = attention.aux_stream if GATE != "packed" else None
        self.events = (None, None)
        if self.aux is not None:
            self.events = (torch.cuda.Event(), torch.cuda.Event())
        # One RMSNorm kernel for q_a and kv_a (their eps are equal, as the stock
        # fused DSA forward requires).
        self.fused_norm = (
            wrapper.q_a_layernorm.variance_epsilon
            == wrapper.kv_a_layernorm.variance_epsilon
        )
        self.in_proj = self.in_out = None
        if rows and ctx is not None and GATE != "packed":
            from ..cutedsl.gemm.decode import DecodeProjection
            from ..cutedsl.gemm.sp_policy import sp_projection_tactics

            rows = tuple(sorted(set(rows)))
            n, k = self.in_weight.shape
            with torch.accelerator.device_index(self.in_weight.device.index):
                tactics = sp_projection_tactics("dsa_in", n, k, rows, ctx)
            self.in_proj = DecodeProjection(
                self.in_weight, rows, tactics, family="dsa_dep_in"
            )
            self.in_out = scratch("dsa_dep.in", (rows[-1], n))
            probe = self.in_weight.new_empty((rows[-1], k))
            self.in_proj.check_operands(probe, self.in_out)
            # The all-head o_proj / g_proj (176 MB each), placed on the dies at
            # construction (die_local.place_layer_on_dies): die-local GEMMs.
            attach_die_local(wrapper.o_proj, rows[-1])
            attach_die_local(wrapper.g_proj, rows[-1])

    def __call__(self, hidden_states, positions):
        """``attn(hidden_states=..., positions=...)`` as ``dsa_dp.run_segment`` calls
        the stock module."""
        return self.run(hidden_states, positions)

    def run(self, x, positions):
        """``x`` [n, 7168] BF16: the segment's rows (``_pre_attn_norm`` output);
        returns the layer output [n, 7168] (the stock forward's, packed GEMMs)."""
        wrapper = self.wrapper
        gate_fn = lambda: self.linear(wrapper.g_proj, x)  # noqa: E731
        if GATE == "packed":
            parts = self.project(x)
            attn_out, gate = self.attend(x, *parts[:6]), parts[6]
        elif GATE == "early":
            attn_out, gate = maybe_execute_in_parallel(
                lambda: self.attend(x, *self.project(x)[:6]), gate_fn, *self.events,
                self.aux,
            )  # fmt: skip
        else:
            parts = self.project(x)
            attn_out, gate = maybe_execute_in_parallel(
                lambda: self.attend(x, *parts[:6]), gate_fn, *self.events, self.aux
            )
        return self.linear(wrapper.o_proj, self.gated(attn_out, gate))

    @staticmethod
    def linear(linear, x):
        """``linear(x)``: its die-local GEMM where it has one serving ``x``."""
        out = run_die_local(linear, x)
        return out if out is not None else linear(x)[0]

    def gated(self, attn_out, gate):
        """``attn_out * gate.sigmoid()`` (the wrapper's ``gate_sigmoid_mul``), in
        one kernel where its capability holds at any row count."""
        mul = self.wrapper.gate_sigmoid_mul
        if mul is prefill_glue.eager_gate_sigmoid_mul and _fused_gate_ok(
            attn_out, gate
        ):
            return prefill_glue.gate_sigmoid_mul(attn_out, gate)
        if mul is not None:
            return mul(attn_out, gate)
        return attn_out * gate.sigmoid()

    def project(self, x):
        """The packed projections: (q [n, H, 192], kv_c_normed, k_pe, index_q,
        index_k, index_weights, gate or None)."""
        wrapper, indexer, attention = self.wrapper, self.indexer, self.attention
        if self.in_proj is not None:
            projected = self.in_proj.run(x.contiguous(), self.in_out[: x.shape[0]])
        else:
            projected = F.linear(x, self.in_weight)
        parts = projected.split(self.in_split, dim=-1)
        q_c, kv_c, k_pe, index_k, index_weights = parts[:5]
        if self.fused_norm:
            q_c, kv_c_normed = fused_q_kv_rmsnorm(
                q_c,
                kv_c,
                wrapper.q_a_layernorm.weight.data,
                wrapper.kv_a_layernorm.weight.data,
                wrapper.q_a_layernorm.variance_epsilon,
            )
        else:
            q_c = wrapper.q_a_layernorm(q_c)
            kv_c_normed = wrapper.kv_a_layernorm(kv_c)
        q, index_q = F.linear(q_c, self.q_weight).split(self.q_split, dim=-1)
        q = q.view(-1, self.heads, attention.qk_head_dim)
        index_q = index_q.view(-1, indexer.index_num_heads, indexer.index_head_dim)
        gate = parts[5] if GATE == "packed" else None
        return q, kv_c_normed, k_pe, index_q, index_k, index_weights, gate

    def attend(self, x, q, kv_c_normed, k_pe, index_q, index_k, index_weights):
        """The indexer (key insert, score, top-k) and the MLA attention: the wrapper
        forward's attention output (no gate, no o_proj)."""
        wrapper, indexer = self.wrapper, self.indexer
        # The indexer forward on the packed projections (index_rope_dim 0).
        if indexer.use_fused_k_insert:
            indexer._fused_k_insert(index_k, x.shape[0])
            index_k = None
        else:
            index_k = indexer.k_index_norm(index_k)
        indexer.score(x, index_q, index_weights, index_k)
        layer = wrapper.mla_attn
        layer.impl.record_logical_topk_ready()
        return layer(
            q,
            kv_c_normed,
            k_pe.unsqueeze(1),
            output_shape=(x.shape[0], self.heads * self.attention.v_head_dim),
            q_dcp_replicated=None,
        )
