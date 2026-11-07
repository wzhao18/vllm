# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer stage of a K3.1 DSA layer at stock Kimi K3 precision: the FP32 indexer
logits (DeepGEMM's paged MQA logits with the FP32 head weights DsaIndex folds, or
the fused scorer of indexer.py) and the exact top-k on them.

  TP   ``run``: on a pure-decode step of uniform rows the DeepGEMM FP32 logits and
       IndexerSelect ("ids") into the model's shared top-k buffer (the fused TP
       indexer, indexer.FusedIndexer, serves the FlashInfer FP8 steps itself);
       any other step the stock decode op (its key insert skipped: DsaIndex
       inserted the key), FP32 as on main.
  DCP  ``merge_rows``: stock's FP32 rule (no one-round #63 merge: stock gates it
       on BF16 logits): the FP32 logits of this rank's keys (the fused DCP
       scorer, or DeepGEMM), every rank's full local top-K, the all-gather and
       the exact merge straight to this rank's physical KV rows and counts
       (dcp_topk.DcpTopk). DcpExchange (``VLLM_K31_DCP_EXCHANGE``) selects on
       BF16 logits only: a layer it is registered for raises."""

import torch

from vllm.utils.torch_utils import _encode_layer_name

from .index import DIM, fp4_query_views, is_fp4_cache
from .indexer import LOGITS_DTYPE, dcp_interleave


def stock_logits_dtype():
    """Stock Kimi K3's indexer logits dtype on main: the sparse_attn_indexer op's
    default (KimiK3DSAIndexer.score passes none), FP32 on every device."""
    return LOGITS_DTYPE


class IndexerStage:
    """``indexer``: the layer's KimiK3DSAIndexer; ``rows``: the most decode rows
    of a step (sizes the DCP top-k's outputs; 0: no DCP top-k); ``impl``: the MLA
    layer's (DcpExchange's counts)."""

    def __init__(self, indexer, rows=0, impl=None):
        self.indexer = indexer
        self.impl = impl
        self.k_cache_name = _encode_layer_name(indexer.k_cache.prefix)
        self.max_model_len = indexer.max_model_len
        self.max_total = indexer.max_total_seq_len
        self.topk = indexer.topk_tokens
        # MXFP4 keys and query (indexer_kv_dtype="mxfp4", kimi_k3_indexer_uses_fp4).
        self.fp4 = indexer.use_fp4_cache
        device = indexer.topk_indices_buffer.device
        # --- DCP top-k (FP32: every rank's local top-K, merged) --------------
        # Each rank's candidates per row: its full local top-K (stock's FP32
        # rule); 0: no DCP top-k (the caller runs the op and the filter).
        world = indexer.dcp_world_size
        self.local_topk = self.topk if rows and world > 1 else 0
        self.dcp = None
        # Optional [>= rows, K] int32: the merge also writes the global ids (tests).
        self.dcp_ids: torch.Tensor | None = None
        if self.local_topk:
            from .dcp_topk import DcpTopk

            self.dcp = DcpTopk(
                self.topk,
                world,
                indexer.dcp_rank,
                lambda: dcp_interleave(indexer),
                device,
                rows,
            )
            shape = (rows, self.topk)
            self.dcp_rows = torch.full(shape, -1, dtype=torch.int32, device=device)
            self.dcp_valid = torch.zeros(rows, dtype=torch.int32, device=device)
        # The fused DCP scorer (PreparedDsa.prepare_cache; None: DeepGEMM).
        self.fused = None
        # TP: the select of DeepGEMM's logits into the top-k buffer, per rows.
        self.tp_rows = rows if world == 1 else 0
        self.dummy = torch.zeros(1, dtype=torch.int32, device=device)
        self.tp_valid = torch.zeros(max(rows, 1), dtype=torch.int32, device=device)
        self.ends = torch.zeros(max(rows, 1), dtype=torch.int32, device=device)
        # -----------------------------------------------------------------------

    def query(self, q_fp8, rows, batch):
        """The stock op's (query, its scales or None) of the first ``rows`` rows of a
        ``batch``-row step: FP8 [rows, 32, 128] (scales folded into the head
        weights), or MXFP4 E2M1 [rows, 32, 64] and UE8M0 x4 [rows, 32] (contiguous
        views of the query DsaIndex wrote, index.fp4_query_views)."""
        if not self.fp4:
            return q_fp8[:rows].view(torch.float8_e4m3fn), None
        return fp4_query_views(q_fp8, batch, rows)

    def decode_step(self, batch):
        """(the indexer's decode metadata, rows) of a pure-decode step of uniform
        rows within ``batch`` (the same on every rank), else (None, 0)."""
        from vllm.forward_context import get_forward_context

        index_md = get_forward_context().attn_metadata[self.indexer.k_cache.prefix]
        decode, rows = index_md.decode, index_md.num_decode_tokens
        if (
            decode is None
            or index_md.num_prefills
            or not 0 < rows <= batch
            or decode.requires_padding
            or rows % decode.decode_lens.shape[0]
        ):
            return None, 0
        return decode, rows

    def run(self, projected, q_fp8, head_w):
        """Top-k of the current rows into the shared buffer: TP on a pure-decode
        step the DeepGEMM FP32 logits and IndexerSelect, else the stock op (FP32,
        as main)."""
        from ...dsa_kernels.hooks import run_hooked_topk

        indexer, batch = self.indexer, projected.shape[0]
        assert is_fp4_cache(indexer.k_cache.kv_cache) == self.fp4
        assert head_w.dtype == torch.float32, head_w.dtype
        # DcpExchange's top-k raises on FP32 logits (BF16 only).
        hooked = run_hooked_topk(
            indexer,
            lambda decode, rows: self.decode_logits(q_fp8, head_w, decode, rows, batch),
        )
        if hooked is not None:
            return hooked
        decode, rows = self.decode_step(batch) if self.tp_rows else (None, 0)
        if decode is not None and batch <= self.tp_rows:
            return self.select_ids(q_fp8, head_w, decode, rows, batch)
        query, q_scale = self.query(q_fp8, batch, batch)
        return torch.ops.vllm.sparse_attn_indexer(
            projected,
            self.k_cache_name,
            indexer.k_cache.kv_cache,
            query,
            q_scale,
            None,
            head_w[:batch],
            DIM,
            indexer.scale_fmt,
            self.topk,
            DIM,
            self.max_model_len,
            self.max_total,
            indexer.topk_indices_buffer,
            True,  # the key is inserted by DsaIndex
            False,
            False,
            _encode_layer_name(""),
            self.fp4,
            indexer.dcp_rank,
            indexer.dcp_world_size,
            indexer.cp_kv_cache_interleave_size,
            topk_backend=indexer.topk_backend,
            dcp_local_topk=indexer.dcp_local_topk,
            logits_dtype=stock_logits_dtype(),
        )

    def select_ids(self, q_fp8, head_w, decode, rows, batch):
        """TP: the exact top-k of the DeepGEMM FP32 logits of the step's ``rows``
        rows into the top-k buffer (key indices, -1 past each row's count; rows
        past them attend nothing)."""
        from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk

        from ..common.host import stream, to_cute
        from .indexer import compile_select

        buffer = self.indexer.topk_indices_buffer
        assert buffer.dtype == torch.int32 and buffer.stride(0) == self.topk
        logits = self.decode_logits(q_fp8, head_w, decode, rows, batch)
        reqs = decode.decode_lens.shape[0]
        ends = SparseIndexerTopk._row_ends(decode.seq_lens[:reqs], rows // reqs, rows)
        row_ends = self.padded_ends(ends, batch)
        if logits.stride(1) != 1 or logits.stride(0) % 4 or logits.data_ptr() % 16:
            raise ValueError(f"indexer top-k: 16-byte aligned rows, {logits.stride()}")
        # Every row of the step (rows past the requests: no keys, -1 rows).
        select = compile_select(batch, 1, self.topk, "ids", buffer.device.index, 1)
        out = buffer[:batch]
        select(
            _i64(logits.data_ptr()),
            _i32(logits.stride(0)),
            to_cute(self.dummy, 4),
            to_cute(row_ends, 4),
            _i64(0),
            _i32(0),
            _i32(0),
            _i32(0),
            to_cute(out, 4),
            to_cute(self.tp_valid[:batch], 4),
            to_cute(out, 4),
            stream(buffer.device),
        )
        return buffer

    def padded_ends(self, ends, batch):
        """``ends`` [rows] extended to ``batch`` rows with zeros (rows past the
        step's requests: no keys)."""
        rows = ends.shape[0]
        if rows == batch:
            return ends
        self.ends[:rows].copy_(ends)
        self.ends[rows:batch].zero_()
        return self.ends[:batch]

    # --- DCP top-k ---------------------------------------------------------------
    def warm(self, block_size, rows=(), tp=True):
        """Compile ``merge_rows``' (DCP) and, ``tp``, ``run``'s (TP: where the fused
        indexer does not serve the layer) select kernels for steps of ``rows`` rows
        now, before any CUDA graph capture (a first compile there invalidates it).
        ``block_size``: unused (any MLA block size)."""
        from .indexer import compile_select

        for r in rows:
            if self.dcp is not None:
                self.dcp.warm(r, self.dcp_ids is not None)
            if self.tp_rows and tp:
                device = self.indexer.topk_indices_buffer.device.index
                compile_select(r, 1, self.topk, "ids", device, 1)

    def merge_source(self, batch):
        """Which top-k ``merge_rows`` gives this step of ``batch`` rows, decided from
        the step metadata alone (the same on every rank): ("merge", rows) on a
        pure-decode step of uniform rows (the FP32 DCP top-k), else (None, 0): the
        caller runs the op and the filter. Raises where DcpExchange is registered
        for the layer: its top-k selects BF16 logits only, stock K3's are FP32."""
        from ...dsa_kernels.hooks import DCP_DECODE_TOPK_HOOKS

        if not self.local_topk:
            return None, 0
        indexer = self.indexer
        if indexer.k_cache.prefix in DCP_DECODE_TOPK_HOOKS:
            raise RuntimeError(
                "Kimi K3.1 specialized DSA: DcpExchange (VLLM_K31_DCP_EXCHANGE) "
                "selects BF16 logits; stock Kimi K3's indexer logits are FP32 "
                "(unset it)"
            )
        decode, rows = self.decode_step(batch)
        if decode is None or decode.global_seq_lens is None:
            return None, 0
        if rows > self.dcp_rows.shape[0]:
            return None, 0
        return "merge", rows

    def merge_rows(self, q_fp8, head_w, metadata, stride_rows, batch, q=1):
        """DCP: this rank's flat KV rows of the exact global top-k of the first
        ``batch`` rows (``q`` per request; compacted to a prefix in its local key
        order) and their counts, as ``dcp_stage.local_slots`` of the stock op's
        global ids; ``metadata``: the MLA layer's. None when the step does not fit
        (the caller then runs the op and the filter). One all-gather over the DCP
        group, so every rank takes the same branch."""
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk

        source, rows = self.merge_source(batch)
        if source is None:
            return None
        assert head_w.dtype == torch.float32, head_w.dtype
        index_md = get_forward_context().attn_metadata[self.indexer.k_cache.prefix]
        decode = index_md.decode
        slots, valid = self.dcp_rows[:batch], self.dcp_valid[:batch]
        fused = self.fused
        # Every row of the step (rows past the requests: no keys, so no rows).
        if fused is not None and decode.seq_lens.numel() == rows:
            logits, ends = fused.run_dcp(q_fp8, head_w, decode, batch, q)
        else:
            logits = self.decode_logits(q_fp8, head_w, decode, rows, batch)
            reqs = decode.decode_lens.shape[0]
            seq_lens = decode.seq_lens[:reqs]
            ends = SparseIndexerTopk._row_ends(seq_lens, rows // reqs, rows)
            ends = self.padded_ends(ends, batch)
        ids = None if self.dcp_ids is None else self.dcp_ids[:batch]
        self.dcp.run(
            logits,
            ends,
            metadata.req_id_per_token,
            metadata.block_table,
            metadata.block_size,
            stride_rows,
            slots,
            valid,
            ids,
        )
        return slots, valid

    def decode_logits(self, q_fp8, head_w, decode, rows, batch):
        """The FP32 paged MQA logits of the first ``rows`` rows of a ``batch``-row
        step, as the stock op's decode branch computes them on main (uniform rows;
        FP8 or MXFP4 cache and query; FP32 head weights reduce in FP32)."""
        from vllm.model_executor.layers.sparse_attn_indexer import (
            kv_cache_as_quant_view,
        )
        from vllm.utils.deep_gemm import (
            fp8_fp4_paged_mqa_logits,
            mqa_logits_mx_inputs,
        )

        logits_dtype = stock_logits_dtype()
        reqs = decode.decode_lens.shape[0]
        query, q_scale = self.query(q_fp8, rows, batch)
        query = query.view(reqs, rows // reqs, *query.shape[1:])
        if q_scale is not None:  # DeepGEMM tags packed FP4 as int8 (kPackedFP4)
            query = query.view(torch.int8)
            q_scale = q_scale.view(reqs, rows // reqs, -1)
        q_scale, weights = mqa_logits_mx_inputs(
            query, q_scale, head_w[:rows], logits_dtype
        )
        return fp8_fp4_paged_mqa_logits(
            (query, q_scale),
            kv_cache_as_quant_view(self.indexer.k_cache.kv_cache, DIM, self.fp4),
            weights,
            decode.seq_lens[:reqs],
            decode.block_table,
            decode.schedule_metadata,
            max_model_len=self.max_model_len,
            logits_dtype=logits_dtype,
            indices=decode.indices,
        )

    # -------------------------------------------------------------------------------


def _i64(value):
    from cutlass import Int64

    return Int64(value)


def _i32(value):
    from cutlass import Int32

    return Int32(value)


def check_index_cache(indexer, fused_unsupported=()):
    """At bind time: whether a DSA layer's indexer cache holds MXFP4 keys; raises
    unless every writer and reader of it takes the bound cache's format. DsaIndex
    and the index fronts compile for the bound cache, while IndexerStage, WideDsa's
    indexer core and the stock op read ``use_fp4_cache``'s format and the fused
    decode indexer FP8 keys only: on MXFP4 it must refuse every shape
    (``fused_unsupported``: its reasons per shape, None where it would run)."""
    cache = indexer.k_cache.kv_cache
    fp4 = is_fp4_cache(cache)
    if fp4 != indexer.use_fp4_cache or (fp4 and not all(fused_unsupported)):
        raise ValueError(
            f"Kimi K3.1 specialized DSA: the indexer cache {tuple(cache.shape)} holds "
            f"{'MXFP4' if fp4 else 'FP8'} keys, the indexer reads use_fp4_cache="
            f"{indexer.use_fp4_cache} and the fused decode indexer (FP8 keys) takes "
            f"{list(fused_unsupported).count(None)} of its shapes"
        )
    return fp4
