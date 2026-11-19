# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 prefill indexer on sm_107: one prefill chunk's exact top-k through the
selective path of ``indexer`` (sampled thresholds, candidate-publishing MXFP4
score, exact candidate select), called from the stock sparse_attn_indexer prefill
loop for single-rank (no DCP / PCP / ShardQ) MXFP4 chunks.

Per step (cached on ``chunk.logits_inputs``): one host read of the chunk's row
bounds -> requests, their 128-aligned key layout and the SelectivePlan. Per layer:
each request's keys gathered to its aligned base, the selective top-k, and one
host read of the flagged-row count; a chunk with flagged rows (threshold missed:
fewer candidates than k, or more than CAP) returns False and the caller runs the
stock path for it, which is exact.
"""

import torch

from vllm import _custom_ops as ops
from vllm.models.kimi_k3.nvidia.specialized_prefill import indexer as pidx
from vllm.platforms import current_platform

_KEY = "kimi_k3_pidx"


def applicable(
    use_fp4_cache: bool,
    topk_tokens: int,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    dcp_world_size: int,
    has_shardq: bool,
    has_candidate_blocks: bool,
    chunk,
) -> bool:
    return (
        use_fp4_cache
        and topk_tokens == 2048
        and q_scale is not None
        and q_quant.shape[1] == pidx.HEADS
        and dcp_world_size == 1
        and not has_shardq
        and not has_candidate_blocks
        and chunk.pcp_deinterleave_idx is None
        and chunk.token_end > chunk.token_start
        and current_platform.is_device_capability((10, 7))
    )


def _step_layout(chunk, device):
    cached = chunk.logits_inputs.get(_KEY)
    if cached is not None:
        return cached
    ks = chunk.cu_seqlen_ks.to("cpu", non_blocking=True)
    ke = chunk.cu_seqlen_ke.to("cpu", non_blocking=True)
    cu = chunk.local_cu_seq_lens.to("cpu", non_blocking=True)
    torch.cuda.current_stream(device).synchronize()
    ks, ke, cu = ks.tolist(), ke.tolist(), cu.tolist()
    reqs = []  # (request index in chunk, rows, last row's keys, key count)
    starts = {s: b for b, s in enumerate(cu[:-1])}
    i = 0
    while i < len(ks):
        j = i
        while j + 1 < len(ks) and ks[j + 1] == ks[i]:
            j += 1
        b = starts[ks[i]]
        reqs.append((b, j - i + 1, ke[j] - ks[i], cu[b + 1] - cu[b]))
        i = j + 1
    bases, nk = [], 0
    for _, _, _, n in reqs:
        bases.append(nk)
        nk += (n + pidx.TILE - 1) // pidx.TILE * pidx.TILE
    sp = pidx.SelectivePlan(
        [r[1] for r in reqs], [r[2] for r in reqs], bases, device
    )
    cu_one = [
        torch.tensor([0, r[3]], dtype=torch.int32, device=device) for r in reqs
    ]
    cached = (reqs, bases, max(nk, pidx.TILE), sp, cu_one)
    chunk.logits_inputs[_KEY] = cached
    return cached


def chunk_topk(
    chunk,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    weights_bf16: torch.Tensor,
    topk_out: torch.Tensor,
) -> bool:
    """Exact top-k of the chunk's rows into ``topk_out`` (request-local key
    indices, unordered, -1 padded). False: rows were flagged, run the stock path."""
    dev = q_quant.device
    reqs, bases, nk, sp, cu_one = _step_layout(chunk, dev)
    keys = torch.empty((nk, 64), dtype=torch.uint8, device=dev)
    scales = torch.empty((nk, 4), dtype=torch.uint8, device=dev)
    for (b, _, _, n), base, cu in zip(reqs, bases, cu_one):
        ops.cp_gather_indexer_k_quant_cache(
            kv_cache,
            keys[base : base + n],
            scales[base : base + n],
            chunk.block_table[b : b + 1],
            cu,
        )
    rows = slice(chunk.token_start, chunk.token_end)
    q = q_quant[rows].contiguous().view(torch.uint8)
    out = torch.empty((sp.t, 2048), dtype=torch.int32, device=dev)
    row_bin, _ = pidx.selective_topk(
        sp,
        q,
        q_scale[rows].contiguous(),
        keys,
        scales.view(torch.int32).view(nk),
        weights_bf16.contiguous(),
        out,
    )
    if int(row_bin[1 : 3 * sp.t : 3].count_nonzero()) != 0:
        return False
    topk_out.copy_(out)
    return True
