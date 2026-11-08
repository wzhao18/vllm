# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Histogram-exchange DCP top-k merge for DSA prefill chunks.

The kernels live in ``hist_merge.py`` (imports CuTe DSL); this module only
holds the configuration helper so model construction stays import-light.

Selection: the stock vLLM setting ``VLLM_DSA_DCP_PREFILL_HIST_MERGE`` (vllm/envs.py,
default off), honoured with its stock meaning: the stock sparse_attn_indexer runs this
merge when the indexer passes ``dcp_prefill_merge_tie_words > 0``. Not a Kimi K3
feature switch; no serving recipe sets it.
"""

from typing import Any

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

# One CTA thread per histogram bin; each round histograms one key byte.
NUM_BINS = 256


def dcp_prefill_merge_tie_words(vllm_config: Any, topk: int) -> int:
    """int32 words per row of the tie bitmap, or 0 when the merge is off
    (``VLLM_DSA_DCP_PREFILL_HIST_MERGE`` unset) or unavailable (no DCP, PCP, a
    KV cache interleave other than 1, not SM100, no CuTe DSL, a top-k that is
    not a multiple of ``NUM_BINS``, or a local row longer than the bitmap
    covers).

    The tie rounds bin local token indices into ``NUM_BINS`` bins of
    ``32 * words`` positions, so ``words`` covers the longest local row; one
    CTA thread clears each word, so at most ``NUM_BINS`` words (2M local
    tokens, 16M at DCP8).
    """
    import vllm.envs as envs
    from vllm.v1.attention.ops.dcp import cp_kv_cache_interleave_one

    parallel_config = vllm_config.parallel_config
    world_size = parallel_config.decode_context_parallel_size
    if (
        not envs.VLLM_DSA_DCP_PREFILL_HIST_MERGE
        or world_size <= 1
        or parallel_config.prefill_context_parallel_size > 1
        or not cp_kv_cache_interleave_one(vllm_config)
        or not current_platform.is_cuda()
        or not current_platform.is_device_capability_family(100)
        or not has_cutedsl()
        or topk % NUM_BINS != 0
    ):
        return 0
    max_local_len = -(-vllm_config.model_config.max_model_len // world_size)
    words = -(-max_local_len // (NUM_BINS * 32))
    return words if words <= NUM_BINS else 0
