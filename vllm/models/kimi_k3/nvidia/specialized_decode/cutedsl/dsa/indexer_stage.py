# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer stage of a K3.1 DSA layer: the stock decode scoring and top-k
(sparse_attn_indexer, its key insert skipped: DsaIndex inserted the key) of the FP8
or MXFP4 (index.py) indexer query and head weights into the model's shared top-k
buffer. Under DCP the op merges the ranks' local top-k into global token ids.

Under DCP on the FlashInfer core, when the layer's indexer has the one-round DCP
top-k merge (``dcp_local_topk``, Inferact/mum#63), ``merge_rows`` replaces the op
and the local-slot filter: it scores the rows (DeepGEMM paged MQA logits, bf16)
and merges them straight into this rank's compacted physical KV rows and counts.
With the fused DCP indexer (``fused``, indexer.py) the scoring is its scorer (the
same logits layout) and, packed, its select also takes the local top-N (the
merge's B' rows), so the merge only all-gathers and selects.

With Wei's DcpExchange on (dsa_kernels.dcp_exchange: built by policy on decode),
``merge_rows`` gives the steps it admits to its exact top-k instead: the same
DeepGEMM logits, its KV slots and counts in place of the merge's (no DeepSelect,
pack or all-gather). The merge stays built for the steps it declines."""

from functools import cache

import torch

from vllm.utils.torch_utils import _encode_layer_name

from .index import DIM, fp4_query_views, is_fp4_cache


class IndexerStage:
    """``indexer``: the layer's KimiK3DSAIndexer; ``rows``: the most decode rows
    of a step (sizes the DCP merge's outputs; 0: no direct DCP merge); ``impl``:
    the MLA layer's, whose ``dcp_local_topk_counts`` DcpExchange sets."""

    def __init__(self, indexer, rows=0, impl=None):
        self.indexer = indexer
        self.impl = impl
        self.k_cache_name = _encode_layer_name(indexer.k_cache.prefix)
        self.max_model_len = indexer.max_model_len
        self.max_total = indexer.max_total_seq_len
        self.topk = indexer.topk_tokens
        # MXFP4 keys and query (indexer_kv_dtype="mxfp4", kimi_k3_indexer_uses_fp4).
        self.fp4 = indexer.use_fp4_cache
        # --- DCP top-k merge (#63): its caller-owned outputs -------------------
        # N, this rank's candidate count; 0 runs the op and the local-slot filter.
        self.local_topk = indexer.dcp_local_topk if rows else 0
        # Optional [>= rows, K] int32: the merge also writes the global ids (tests).
        self.dcp_ids: torch.Tensor | None = None
        if self.local_topk:
            device = indexer.topk_indices_buffer.device
            # K wide: the exact merge gives a rank up to K of a row's keys (a
            # saturated row: more than its N local candidates).
            shape = (rows, self.topk)
            self.dcp_rows = torch.full(shape, -1, dtype=torch.int32, device=device)
            self.dcp_valid = torch.zeros(rows, dtype=torch.int32, device=device)
        # The fused DCP indexer (PreparedDsa.prepare_cache; None: DeepGEMM).
        self.fused = None
        # -----------------------------------------------------------------------

    def query(self, q_fp8, rows, batch):
        """The stock op's (query, its scales or None) of the first ``rows`` rows of a
        ``batch``-row step: FP8 [rows, 32, 128] (scales folded into the head
        weights), or MXFP4 E2M1 [rows, 32, 64] and UE8M0 x4 [rows, 32] (contiguous
        views of the query DsaIndex wrote, index.fp4_query_views)."""
        if not self.fp4:
            return q_fp8[:rows].view(torch.float8_e4m3fn), None
        return fp4_query_views(q_fp8, batch, rows)

    def run(self, projected, q_fp8, head_w):
        """Top-k of the current rows into the shared buffer (stock decode scoring)."""
        indexer, batch = self.indexer, projected.shape[0]
        assert is_fp4_cache(indexer.k_cache.kv_cache) == self.fp4
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
            # Interleave 1: the exact one-round DCP merge (gated in the op).
            dcp_local_topk=indexer.dcp_local_topk,
        )

    # --- DCP top-k merge (#63) ---------------------------------------------------
    def warm(self, block_size):
        """Compile ``merge_rows``' merge kernels for MLA blocks of ``block_size``
        tokens now (nothing without the direct DCP merge): see warm_merge_rows."""
        if self.local_topk:
            args = self.topk, self.local_topk, self.indexer.dcp_world_size
            warm_merge_rows(*args, self.dcp_ids is not None, block_size)

    def merge_source(self, batch):
        """Which top-k ``merge_rows`` gives this step of ``batch`` rows, decided from
        the step metadata alone (the same on every rank): ("exchange", rows) where
        DcpExchange's top-k admits the step, ("merge", rows) where the #63 merge fits
        it (pure decode, uniform rows, interleave 1), else (None, 0): the caller
        runs the op and the filter."""
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.sparse_attn_indexer import (
            DCP_DECODE_TOPK_HOOKS,
        )
        from vllm.v1.attention.ops.dcp_topk_merge import dcp_topk_merge_fits

        if not self.local_topk:
            return None, 0
        indexer = self.indexer
        index_md = get_forward_context().attn_metadata[indexer.k_cache.prefix]
        decode, rows = index_md.decode, index_md.num_decode_tokens
        if (
            decode is None
            or decode.global_seq_lens is None
            or index_md.num_prefills
            or not 0 < rows <= batch
            or decode.requires_padding
            or rows % decode.decode_lens.shape[0]
        ):
            return None, 0
        hook = DCP_DECODE_TOPK_HOOKS.get(indexer.k_cache.prefix)
        if hook is not None and self.impl is not None and hook[0](rows):
            return "exchange", rows
        if indexer.cp_kv_cache_interleave_size == 1 and dcp_topk_merge_fits(
            rows, self.topk, self.local_topk
        ):
            return "merge", rows
        return None, 0

    def merge_rows(self, q_fp8, head_w, metadata, stride_rows, batch, q=1):
        """DCP: this rank's flat KV rows of the merged global top-k of the first
        ``batch`` rows (``q`` per request; compacted to a prefix, at most N per
        row) and their counts, as ``dcp_stage.local_slots`` of the op's global
        ids; ``metadata``: the MLA layer's. None when the step does not fit the
        merge (the caller then runs the op and the filter). One all-gather over
        the DCP group, so every rank takes the same branch. The logits must hold
        no NaN: DeepSelect then returns out-of-range ids, on which the merge's
        pack faults; no per-step check here (finite query, head weights and
        cached keys keep them finite)."""
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.indexer_topk import (
            SparseIndexerTopk,
            is_deep_select_supported,
        )
        from vllm.model_executor.layers.sparse_attn_indexer import (
            DCP_DECODE_TOPK_HOOKS,
        )
        from vllm.utils.math_utils import cdiv
        from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
            dcp_topk_merge_rows,
        )

        source, rows = self.merge_source(batch)
        if source is None:
            return None
        indexer, n = self.indexer, self.local_topk
        index_md = get_forward_context().attn_metadata[indexer.k_cache.prefix]
        decode = index_md.decode
        if source == "exchange":
            run = DCP_DECODE_TOPK_HOOKS[indexer.k_cache.prefix][1]
            return self.exchange_rows(run, q_fp8, head_w, decode, rows, batch)
        slots, valid = self.dcp_rows[:batch], self.dcp_valid[:batch]
        ids = None if self.dcp_ids is None else self.dcp_ids[:rows, : self.topk]
        # As the stock hook: interleave 1 bounds the local lengths.
        max_local_len = cdiv(self.max_model_len, indexer.dcp_world_size)
        merge = dict(
            local_topk=n,
            topk=self.topk,
            req_id=metadata.req_id_per_token,
            block_table=metadata.block_table,
            block_size=metadata.block_size,
            stride_rows=stride_rows,
            dcp_rank=indexer.dcp_rank,
            rows_out=slots[:rows],
            valid_out=valid[:rows],
            ids_out=ids,
        )
        fused = self.fused
        if fused is not None and decode.seq_lens.numel() == rows:
            # The logits stay valid until the merge returns (stream order): a
            # saturated rank re-reads its row in the exact merge's second round.
            packed, logits, ends = fused.run_dcp(q_fp8, head_w, decode, batch, q)
            dcp_topk_merge_rows(
                logits=logits[:rows],
                row_ends=ends[:rows],
                packed=None if packed is None else packed[:rows],
                max_local_len=max_local_len,
                **merge,
            )
            if rows < batch:  # rows past the requests attend nothing
                slots[rows:].fill_(-1)
                valid[rows:].zero_()
            return slots, valid
        logits = self.decode_logits(q_fp8, head_w, decode, rows, batch)
        # SM100 DeepGEMM emits bf16 at a 1 KiB row stride (DeepSelect's input).
        assert logits.dtype == torch.bfloat16 and is_deep_select_supported(logits, n)
        reqs = decode.decode_lens.shape[0]
        seq_lens = decode.seq_lens[:reqs]
        row_ends = SparseIndexerTopk._row_ends(seq_lens, rows // reqs, rows)
        dcp_topk_merge_rows(
            logits=logits,
            row_ends=row_ends,
            max_local_len=max_local_len,
            **merge,
        )
        if rows < batch:  # rows past the requests attend nothing
            slots[rows:].fill_(-1)
            valid[rows:].zero_()
        return slots, valid

    def decode_logits(self, q_fp8, head_w, decode, rows, batch):
        """The bf16 paged MQA logits of the first ``rows`` rows of a ``batch``-row
        step, as the stock op's decode branch computes them (uniform rows; FP8 or
        MXFP4 cache and query)."""
        from vllm.model_executor.layers.sparse_attn_indexer import (
            kv_cache_as_quant_view,
        )
        from vllm.utils.deep_gemm import (
            fp8_fp4_paged_mqa_logits,
            mqa_logits_mx_inputs,
        )

        reqs = decode.decode_lens.shape[0]
        query, q_scale = self.query(q_fp8, rows, batch)
        query = query.view(reqs, rows // reqs, *query.shape[1:])
        if q_scale is not None:  # DeepGEMM tags packed FP4 as int8 (kPackedFP4)
            query = query.view(torch.int8)
            q_scale = q_scale.view(reqs, rows // reqs, -1)
        q_scale, weights = mqa_logits_mx_inputs(query, q_scale, head_w[:rows])
        return fp8_fp4_paged_mqa_logits(
            (query, q_scale),
            kv_cache_as_quant_view(self.indexer.k_cache.kv_cache, DIM, self.fp4),
            weights,
            decode.seq_lens[:reqs],
            decode.block_table,
            decode.schedule_metadata,
            max_model_len=self.max_model_len,
            indices=decode.indices,
        )

    def exchange_rows(self, run, q_fp8, head_w, decode, rows, batch):
        """``merge_rows`` of a DcpExchange step: ``run`` (its indexer hook) writes
        this rank's KV slots of the exact global top-k of the same logits and the
        MLA impl's ``dcp_local_topk_counts`` gives their counts."""
        slots, valid = self.dcp_rows[:batch], self.dcp_valid[:batch]
        logits = self.decode_logits(q_fp8, head_w, decode, rows, batch)
        run(logits, decode, slots[:rows])
        counts = self.impl.dcp_local_topk_counts(rows)
        assert counts is not None, "DcpExchange admitted a step it did not serve"
        if rows == batch:
            return slots, counts
        valid[:rows].copy_(counts)
        slots[rows:].fill_(-1)  # rows past the requests attend nothing
        valid[rows:].zero_()
        return slots, valid

    # -------------------------------------------------------------------------------


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


@cache
def warm_merge_rows(topk, local_topk, world_size, emit_ids, block_size):
    """Compile the #63 merge's kernels as ``merge_rows`` launches them (physical rows
    of ``block_size``-token blocks; global ids too with ``emit_ids``) now, once per
    process. The specialized decode runs only inside CUDA graph captures, where a
    first compile invalidates the capture (CUDA error 901), and it is built after
    the runner's JIT warmup registry closes, whose warmup compiles only the stock
    indexer's variant (global ids, no rows)."""
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        register_dcp_topk_merge_warmup,
    )

    registry = JitWarmupRegistry(None)
    with registry.activate():
        register_dcp_topk_merge_warmup(
            topk, local_topk, world_size, emit_ids=emit_ids, block_size=block_size
        )
    registry.warmup()
