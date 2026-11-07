# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DcpExchange top-k hooks of the SM107 specialized decode.

On the source branch these lived in stock code: ``sparse_attn_indexer``'s
``DCP_DECODE_TOPK_HOOKS`` (the stock indexer op ran the hook in place of its decode
top-k) and ``SparseMLACommonImpl.dcp_local_topk_counts`` (the stock sparse MLA decode
read the hook's slot counts). Here the stock indexer op and the stock sparse MLA
decode stay unchanged: they never see the exchange, so every stock step keeps the
stock DCP top-k (global ids, then the local-slot filter). Only the specialized decode
consults these registries, and every place where it used to call the stock op on a
step the exchange admits runs the exchange itself (``run_hooked_topk``), so the slots
it later reads always come from the exchange.
"""

from collections.abc import Callable
from typing import Any

import torch

# Indexer k_cache prefix -> (active(rows) -> bool, run(logits, decode, out)).
# ``run`` writes this rank's selected KV slots (not global ids) into ``out`` and their
# counts into the exchange's buffer; ``active`` decides from the step metadata alone,
# so the indexer side and the attention side always agree.
DCP_DECODE_TOPK_HOOKS: dict[str, tuple[Callable, Callable]] = {}

# id(sparse MLA impl) -> counts(rows) -> Tensor | None: the slot counts of the step's
# exchanged top-k (None: the step did not take the exchange).
_LOCAL_TOPK_COUNTS: dict[int, Callable[[int], torch.Tensor | None]] = {}


def set_local_topk_counts(impl: Any, counts: Callable) -> None:
    _LOCAL_TOPK_COUNTS[id(impl)] = counts


def local_topk_counts(impl: Any) -> Callable[[int], torch.Tensor | None] | None:
    """The exchange's counts callable of ``impl``'s layer, or None (no exchange)."""
    if impl is None:
        return None
    return _LOCAL_TOPK_COUNTS.get(id(impl))


def hooked_topk_rows(prefix: str) -> int:
    """Decode rows of the current step whose top-k the exchange takes for the
    indexer layer ``prefix`` (0: the stock op runs). Same predicate as the source
    branch's stock op: the hook's ``active`` on the pure-decode rows of the step."""
    hook = DCP_DECODE_TOPK_HOOKS.get(prefix)
    if hook is None:
        return 0
    from vllm.forward_context import get_forward_context

    md_all = get_forward_context().attn_metadata
    index_md = md_all.get(prefix) if isinstance(md_all, dict) else None
    if index_md is None or getattr(index_md, "decode", None) is None:
        return 0
    rows = index_md.num_decode_tokens
    if rows <= 0 or index_md.num_prefills != 0 or not hook[0](rows):
        return 0
    return rows


def run_hooked_topk(indexer, logits_fn: Callable[[Any, int], torch.Tensor]):
    """Where the source branch's stock indexer op handed the step to the exchange:
    run the exchange on the step's bf16 decode logits (``logits_fn(decode, rows)``)
    into the indexer's top-k buffer and return the buffer; None when the exchange
    does not take the step (the caller runs the stock op)."""
    prefix = indexer.k_cache.prefix
    rows = hooked_topk_rows(prefix)
    if not rows:
        return None
    from vllm.forward_context import get_forward_context

    decode = get_forward_context().attn_metadata[prefix].decode
    logits = logits_fn(decode, rows)
    buffer = indexer.topk_indices_buffer
    DCP_DECODE_TOPK_HOOKS[prefix][1](logits, decode, buffer[:rows])
    return buffer
