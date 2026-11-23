# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile DeepGEMM's paged-MQA-logits scheduling metadata kernel for every row
count a serving step can build it for.

``DeepseekV32IndexerMetadataBuilder`` builds the DSA indexer's decode schedule
with DeepGEMM ``get_paged_mqa_logits_metadata`` on every step that has decode
rows. DeepGEMM JIT-compiles that kernel per thread-block size, which it picks
from the step's number of Q rows (on SM100: <= 512, <= 2048, more), so the
first step with rows in a new class compiles it in the serving loop unless a
startup step had rows in that class. The warmup builds the metadata (into a
scratch tensor) at each row count of a doubling ladder, plus each power of two
+ 1, up to the most decode rows a step can hold: compile only; no state,
cache or output of the builders is touched.
"""

from collections.abc import Iterable
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def metadata_row_counts(max_rows: int) -> list[int]:
    """Row counts that reach every DeepGEMM thread-block class up to
    ``max_rows``: 1, each power of two and power of two + 1 below it, and
    ``max_rows`` itself."""
    if max_rows < 1:
        return []
    counts = {1, max_rows}
    power = 1
    while power <= max_rows:
        counts.add(power)
        if power + 1 <= max_rows:
            counts.add(power + 1)
        power *= 2
    return sorted(counts)


def max_decode_rows(vllm_config: Any, decode_threshold: int) -> int:
    """The most indexer decode rows (requests x query rows) one step can hold."""
    scheduler = vllm_config.scheduler_config
    return min(
        scheduler.max_num_batched_tokens,
        scheduler.max_num_seqs * max(1, decode_threshold),
    )


def _indexer_builders(attn_groups: Iterable[Iterable[Any]]) -> list[Any]:
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )

    builders = []
    for groups in attn_groups:
        for group in groups:
            for builder in getattr(group, "metadata_builders", ()):
                if isinstance(builder, DeepseekV32IndexerMetadataBuilder):
                    builders.append(builder)
    return builders


def warmup_paged_mqa_logits_metadata(
    attn_groups: Iterable[Iterable[Any]], vllm_config: Any
) -> int:
    """Build the paged MQA logits metadata of each distinct indexer builder
    (block size, SM count, decode query length) at every row count of
    ``metadata_row_counts``, as one-row (``next_n`` 1) and decode-query-length
    rows. Returns the number of metadata builds."""
    from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata

    calls = 0
    seen: set[tuple[int, int, int]] = set()
    for builder in _indexer_builders(attn_groups):
        block = int(builder.kv_cache_spec.num_states)
        next_n = int(builder.decode_threshold)
        key = (block, int(builder.num_sms), next_n)
        if key in seen:
            continue
        seen.add(key)
        max_rows = max_decode_rows(vllm_config, next_n)
        for query_len in sorted({1, next_n}):
            for rows in metadata_row_counts(max_rows):
                batch = max(1, rows // query_len)
                context_lens = torch.full(
                    (batch, query_len), block, dtype=torch.int32, device=builder.device
                )
                get_paged_mqa_logits_metadata(context_lens, block, builder.num_sms)
                calls += 1
    if calls:
        torch.accelerator.synchronize()
        logger.info(
            "DSA indexer metadata warmup: %d DeepGEMM paged MQA logits metadata "
            "builds for %d builder configuration(s) (block, SMs, next_n): %s.",
            calls,
            len(seen),
            sorted(seen),
        )
    return calls
