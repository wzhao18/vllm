# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer stage of a K3.1 DSA layer at stock Kimi K3 precision: the FP32 indexer
logits of sm_107's own scorer (indexer.IndexerScore: FP8 or MXFP4 keys, FP32 head
weights, the FP32 head reduction of DeepGEMM's paged MQA logits) and the exact
top-k on them. No stock op runs here: a step this stage does not serve raises.

  TP   ``run``: the scorer's FP32 logits and IndexerSelect ("compact_ids") into
       the model's shared top-k buffer (the FlashMLA sparse decode's key ids;
       on FlashInfer, PreparedDsa's FusedIndexer gives the physical rows itself).
  DCP  ``merge_rows``: stock's FP32 rule (no one-round #63 merge: stock gates it
       on BF16 logits): the FP32 logits of this rank's keys (the scorer, in
       DeepGEMM's logits layout), every rank's full local top-K, the peer-store
       exchange of each rank's top-N (dcp_round.py, exact) and the merge straight
       to this rank's physical KV rows and counts (dcp_topk.DcpTopk).

Both serve the steps the specialized decode runs (runtime.decode.decode_metadata):
pure decode or uniform verify, the same on every rank."""

import torch

from ...shapes import refuse
from .indexer import dcp_interleave


def check_head_weights(head_w: torch.Tensor):
    """The indexer head weights are FP32 at stock K3 precision (never rounded)."""
    if head_w.dtype != torch.float32:
        raise TypeError(
            "Kimi K3.1 specialized DSA: the indexer head weights are FP32 (stock "
            f"Kimi K3 precision), got {head_w.dtype}"
        )


class IndexerStage:
    """``indexer``: the layer's KimiK3DSAIndexer; ``rows``: the most decode rows
    of a step (sizes the DCP top-k's outputs). ``fused``: the layer's scorer
    (indexer.FusedIndexer: DCP, this rank's keys in DeepGEMM's layout; TP, the
    request-compact logits and the top-k ids), set by the layer's prepare_cache."""

    def __init__(self, indexer, rows):
        if rows <= 0:
            raise ValueError(f"Kimi K3.1 specialized DSA: indexer stage rows {rows}")
        self.indexer = indexer
        self.topk = indexer.topk_tokens
        # MXFP4 keys and query (indexer_kv_dtype="mxfp4", kimi_k3_indexer_uses_fp4).
        self.fp4 = indexer.use_fp4_cache
        device = indexer.topk_indices_buffer.device
        # --- DCP top-k (FP32: every rank's local top-K, merged) --------------
        world = indexer.dcp_world_size
        self.local_topk = self.topk if world > 1 else 0
        self.dcp = None
        # Optional [>= rows, K] int32: the merge also writes the global ids (tests).
        self.dcp_ids: torch.Tensor | None = None
        self.max_rows = rows
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
        self.fused = None
        # -----------------------------------------------------------------------

    def decode_step(self, batch, q=1):
        """(the indexer's decode metadata, rows) of this pure-decode or uniform
        verify step of ``batch`` graph rows (``q`` per request). Raises on any
        other step: the specialized decode runs only those (decode_metadata)."""
        from vllm.forward_context import get_forward_context

        index_md = get_forward_context().attn_metadata[self.indexer.k_cache.prefix]
        decode, rows = index_md.decode, index_md.num_decode_tokens
        cause = None
        if decode is None or index_md.num_prefills:
            cause = "a step with prefill rows"
        elif not 0 < rows <= batch or rows % q:
            cause = f"{rows} decode rows in a {batch}-row q{q} step"
        elif decode.requires_padding or rows % decode.decode_lens.shape[0]:
            cause = "non-uniform decode lengths"
        elif rows > self.max_rows:
            cause = f"{rows} rows above the stage's {self.max_rows}"
        elif decode.seq_lens.numel() != rows:
            cause = f"{decode.seq_lens.numel()} per-row lengths for {rows} rows"
        if cause is not None:
            refuse("DSA indexer stage", cause, batch)
        return decode, rows

    def run(self, q_fp8, head_w, batch, q=1):
        """TP: the exact top-k of the step's rows (FP32 logits of the scorer) into
        the indexer's top-k buffer (key indices, -1 past each row's count; rows
        past the requests attend nothing) -> the buffer."""
        check_head_weights(head_w)
        if self.local_topk or self.fused is None or self.fused.pitch:
            raise RuntimeError("Kimi K3.1 specialized DSA: the TP top-k is not built")
        decode, _ = self.decode_step(batch, q)
        buffer = self.indexer.topk_indices_buffer
        self.fused.run_ids(q_fp8, head_w, decode, batch, q, buffer)
        return buffer

    # --- DCP top-k ---------------------------------------------------------------
    def warm(self, rows=()):
        """Compile the DCP top-k's select and merge for steps of ``rows`` rows now,
        before any CUDA graph capture (a first compile there invalidates it)."""
        for r in rows:
            if self.dcp is not None:
                self.dcp.warm(r, self.dcp_ids is not None)

    def merge_rows(self, q_fp8, head_w, metadata, stride_rows, batch, q=1):
        """DCP: this rank's flat KV rows of the exact global top-k of the first
        ``batch`` rows (``q`` per request; compacted to a prefix in its local key
        order) and their counts, as ``dcp_stage.local_slots`` of the stock op's
        global ids; ``metadata``: the MLA layer's. One peer-store exchange over
        the DCP group: every rank runs it on every step."""
        check_head_weights(head_w)
        if not self.local_topk or self.fused is None or not self.fused.pitch:
            raise RuntimeError("Kimi K3.1 specialized DSA: the DCP top-k is not built")
        decode, _ = self.decode_step(batch, q)
        slots, valid = self.dcp_rows[:batch], self.dcp_valid[:batch]
        # Every row of the step (rows past the requests: no keys, so no rows).
        logits, ends = self.fused.run_dcp(q_fp8, head_w, decode, batch, q)
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

    # -------------------------------------------------------------------------------


def check_index_cache(indexer):
    """At bind time: whether a DSA layer's indexer cache holds MXFP4 keys; raises
    unless every writer and reader of it takes the bound cache's format (DsaIndex
    and the index fronts compile for the bound cache; the scorer reads
    ``use_fp4_cache``'s format)."""
    from .index import is_fp4_cache

    cache = indexer.k_cache.kv_cache
    fp4 = is_fp4_cache(cache)
    if fp4 != indexer.use_fp4_cache:
        raise ValueError(
            f"Kimi K3.1 specialized DSA: the indexer cache {tuple(cache.shape)} holds "
            f"{'MXFP4' if fp4 else 'FP8'} keys, the indexer reads use_fp4_cache="
            f"{indexer.use_fp4_cache}"
        )
    return fp4
