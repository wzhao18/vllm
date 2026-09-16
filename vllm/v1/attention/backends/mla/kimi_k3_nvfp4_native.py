# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native NVFP4 decode attention for Kimi-K3 MLA.

The persistent page is opaque and page-major: packed K data, K block scales,
then the PV-oriented scale view. Attention consumes those buffers directly;
it never materializes a context-sized FP8 or BF16 cache.

The Triton kernels and launch structure originate from TensorRT-LLM MR 10576.
This adapter deliberately keeps only the non-prepacked-V path: a 128-wide V
tile is required on SM103 and preserves the 392-byte/token cache footprint.
"""

import os
from typing import Any

import torch

from vllm.triton_utils import triton

FP4_BLOCK_SIZE = 16
FP4_MLA_TOKENS_PER_BLOCK = 128
FP4_MLA_K_RESIDUAL_DIM = 64
FP4_MLA_Q_RESIDUAL_DIM = 64
FP4_MLA_P_GLOBAL_SCALE = 448.0 * 6.0
FP4_MLA_Q_GLOBAL_SCALE = FP4_MLA_P_GLOBAL_SCALE / 400.0
FP4_MLA_KV_GLOBAL_SCALE = FP4_MLA_P_GLOBAL_SCALE / 30.0
FP4_MLA_Q_LOGICAL_DIM = 640
FP4_MLA_Q_PACKED_DIM = FP4_MLA_Q_LOGICAL_DIM // 2
FP4_MLA_STORAGE_DIM = 640
FP4_MLA_DATA_BYTES_PER_TOKEN = FP4_MLA_STORAGE_DIM // 2
FP4_MLA_K_SCALE_BYTES_PER_TOKEN = FP4_MLA_STORAGE_DIM // FP4_BLOCK_SIZE
FP4_MLA_V_SCALE_BYTES_PER_TOKEN = 32
FP4_MLA_BYTES_PER_TOKEN = (
    FP4_MLA_DATA_BYTES_PER_TOKEN
    + FP4_MLA_K_SCALE_BYTES_PER_TOKEN
    + FP4_MLA_V_SCALE_BYTES_PER_TOKEN
)
HP_BLOCK_SIZE = 16
FP4_MLA_Q_PREFIX_DIM = 512
FP4_MLA_Q_PREFIX_BLOCK_DIM = 256
FP4_MLA_Q1_PREFIX_BLOCK_DIM = 512


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    return None if not value else int(value)


def _get_sm_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _select_triton_block_v(
    num_queries: int, *, prefer_prepacked_v: bool = False
) -> int:
    override = _env_int("VLLM_KIMI_K3_NVFP4_BLOCK_V")
    if override is not None:
        return override
    capability = torch.cuda.get_device_capability()
    # The 32-wide inline transpose issues a misaligned access on GB300/SM103.
    if capability == (10, 3):
        return 128
    return 128 if prefer_prepacked_v or num_queries >= 64 else 32


def _triton_prepack_v_enabled() -> bool:
    return False


def _infer_assume_full_pages(metadata: Any, max_pages: int, page_size: int) -> bool:
    # The masked path is safe for ragged vLLM block tables and DSpark queries.
    return False


def _ensure_workspace_tensor(
    owner: Any,
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    value = getattr(owner, name, None)
    required = int(torch.tensor(shape).prod().item())
    if value is None or value.dtype != dtype or value.numel() < required:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Kimi-K3 NVFP4 workspace {name} was not warmed up before capture."
            )
        value = torch.empty(shape, dtype=dtype, device=device)
        setattr(owner, name, value)
    return value.view(-1)[:required].view(shape)


def _get_triton_v_packed_cache(*args, **kwargs) -> None:
    return None


def _triton_can_prepack_v(*args, **kwargs) -> bool:
    return False


def _update_triton_v_packed_cache(*args, **kwargs) -> None:
    raise AssertionError("the Kimi-K3 NVFP4 vLLM path never pre-packs V")


def _run_triton_attention_decode(
    *,
    metadata: Any,
    layer_idx: int,
    local_layer: int,
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_sf: torch.Tensor,
    global_scale: torch.Tensor,
    src_page_ids: torch.Tensor,
    kv_lens: torch.Tensor,
    p_fp4: torch.Tensor,
    p_sf: torch.Tensor,
    max_scores: torch.Tensor,
    denom: torch.Tensor,
    output: torch.Tensor,
    num_queries: int,
    num_heads: int,
    head_dim: int,
    kv_lora_rank: int,
    q_residual_dim: int,
    query_len_per_seq: int,
    max_pages: int,
    sm_scale: float,
    q_global_scale: torch.Tensor,
) -> None:
    """Dispatch the ``triton`` FP4 MLA decode pipeline.

    Mirrors the four-stage layout used by ``fp4_mla_cutile.py``
    (page-stats with packed P -> reduce-stats -> prob-scale -> PV) but
    routes through the self-contained kernels in
    ``fp4_mla_triton.py``. Threads through the constexpr assume flags,
    TMA descriptors, occupancy/num-warps launch meta, and pipelined PV loop.
    """
    from .kimi_k3_nvfp4_triton import (
        _fp4_mla_attention_group_reduce_stats_kernel as _attn_group_reduce_stats_kernel,
    )
    from .kimi_k3_nvfp4_triton import (
        _fp4_mla_attention_page_stats_kernel as _attn_page_stats_kernel,
    )
    from .kimi_k3_nvfp4_triton import (
        _fp4_mla_attention_prob_scale_kernel as _attn_prob_scale_kernel,
    )
    from .kimi_k3_nvfp4_triton import _fp4_mla_attention_pv_kernel as _attn_pv_kernel
    from .kimi_k3_nvfp4_triton import (
        _fp4_mla_attention_pv_prepacked_v_kernel as _attn_pv_prepacked_v_kernel,
    )
    from .kimi_k3_nvfp4_triton import (
        _fp4_mla_attention_pv_reduce_kernel as _attn_pv_reduce_kernel,
    )
    from .kimi_k3_nvfp4_triton import (
        _fp4_mla_attention_reduce_stats_kernel as _attn_reduce_stats_kernel,
    )

    block_h = 128
    block_t = metadata.page_size
    # Adaptive BLOCK_V: the fallback PV path uses a finer V split at small batch
    # on B200 (~148 SMs). PV grid = num_queries * num_head_blocks(1) *
    # (kv_lora_rank / BLOCK_V). We want >= ~2*num_SMs programs so that >1 CTA
    # lands per SM and hides the L1TEX scoreboard stalls. Empirically (sweep):
    #   bs<=32 -> BLOCK_V=32; bs>=64 -> BLOCK_V=128.
    # (BLOCK_V=16 is rejected by the V TMA descriptor min-stride requirement.)
    # With prepacked V, BLOCK_V=128 avoids reloading the same P tile four times
    # and matches the cutile prepacked-V tile shape.
    block_v = _select_triton_block_v(
        num_queries, prefer_prepacked_v=_triton_prepack_v_enabled()
    )
    q_storage_head_dim = head_dim + q_residual_dim
    # The virtual GEMM tail evaluates QK + Q_r K + Q K_r in one reduction.
    # Q and Q_r still occupy the 640-channel interleaved physical Q buffer;
    # the final Q term reuses Q's main tail groups while K_r comes from the
    # contiguous 64-channel tail of the primary paged KV cache.
    q_head_dim = head_dim + q_residual_dim + FP4_MLA_K_RESIDUAL_DIM
    # BLOCK_K = 512 aligns the K-window with the 512-channel non-residual prefix.
    block_k = 512
    full_block_end = (q_head_dim // block_k) * block_k
    tail_k = q_head_dim - full_block_end
    tail_block_k = 1 << (tail_k - 1).bit_length() if tail_k > 0 else block_k
    q_sf_per_token = q_storage_head_dim // FP4_BLOCK_SIZE
    k_sf_per_token = (head_dim + FP4_MLA_K_RESIDUAL_DIM) // FP4_BLOCK_SIZE
    sf_per_page = metadata.page_size // FP4_BLOCK_SIZE
    num_head_blocks = triton.cdiv(num_heads, block_h)

    assume_full_heads = num_heads % block_h == 0
    assume_full_v = kv_lora_rank % block_v == 0
    # Match the cutile path: only mark pages "full" when we can prove every
    # generation sequence has the same number of cached tokens AND
    # query_len_per_seq == 1 (so the kv_len adjustment is a no-op).
    assume_full_pages = (
        _infer_assume_full_pages(metadata, max_pages, metadata.page_size)
        and query_len_per_seq == 1
    )
    # Leave validity checks on. Matches cutile's default and is correctness-
    # safe. The perfect-shape PV fast path (tl.ext.make_view + load_view_tko)
    # remains gated off — when measured on the TileIR backend (ENABLE_TILE=1)
    # it was net-slower on the bench, so the cost of enabling it isn't worth
    # the win on the FP4 MLA shapes we care about.
    assume_valid_pages = False
    num_gen_seqs = num_queries // query_len_per_seq
    if (
        not assume_valid_pages
        and assume_full_pages
        and src_page_ids.numel() == num_gen_seqs * max_pages
    ):
        assume_valid_pages = True
    # cutile checks only `make_tensor_descriptor`; on the nvt backend the
    # presence of TMA descriptors implies `tl.ext.make_view` is available too.
    tma_default = torch.cuda.get_device_capability() != (10, 3)
    use_tma_data_load = hasattr(triton.language, "make_tensor_descriptor") and (
        os.getenv("VLLM_KIMI_K3_NVFP4_USE_TMA", "1" if tma_default else "0")
        not in ("0", "false", "no", "off")
    )

    # Install the device-side scratch allocator on every call. Triton stores
    # the allocator in a ContextVar (triton.runtime._allocation), so a single
    # process-wide install is not visible from worker threads / asyncio tasks
    # that run with a different Context — the kernel launch would then hit the
    # default NullAllocator and raise. Matches the cutile path.
    if use_tma_data_load:

        def _tma_alloc(size: int, alignment: int, stream):
            return torch.empty(size, device=q_fp4.device, dtype=torch.int8)

        triton.set_allocator(_tma_alloc)

    # cutile-equivalent launch meta. occupancy=2 lets two CTAs land per SM
    # which improves wave-tail efficiency at the bs=32 hot point.
    # NOTE: num_stages=2 (instead of the Triton 3.6 default of 3) sidesteps
    # the TritonGPUAutomaticWarpSpecialization + NVWSInsertTmemAref pass that
    # ICEs on the page_stats kernel under Triton 3.6.0 / sm_100.
    launch_meta = {"occupancy": 2}
    # The matmul kernels (page-stats QK and PV) are register-limited: at the
    # Triton default of num_warps=4 the [BLOCK_H, BLOCK_T] epilogue spills the
    # register file down to ~2 CTAs/SM (12.5% occupancy), so there are too few
    # warps to hide the QK/PV load latency (ncu: ~0.3 eligible warps/scheduler).
    # Spreading the tile epilogue over num_warps=8 halves the per-thread
    # register need and roughly doubles resident warps. Matches the cutile
    # ("nvt") backend, which launches page-stats at num_warps=8. Both are
    # overridable for tuning.
    sm_count = _get_sm_count(q_fp4.device)
    # page-stats num_warps: the full-pages fast path (uniform q_len==1 decode)
    # benefits from num_warps=8 (more warps hide the QK load latency); the
    # masked path (q_len>1 / ragged lengths) carries extra per-thread state and
    # measured markedly faster at num_warps=4 (e.g. bs256 q_len4: 131->95ms).
    page_stats_num_warps = _env_int("TRTLLM_FP4_MLA_PAGE_STATS_NUM_WARPS")
    if page_stats_num_warps is None:
        page_stats_num_warps = 8 if assume_full_pages else 4
    page_stats_launch_meta = {"occupancy": 2, "num_warps": page_stats_num_warps}
    # PV benefits from num_warps=8 across shapes measured.
    pv_num_warps = _env_int("TRTLLM_FP4_MLA_PV_NUM_WARPS") or 8
    pv_launch_meta = {"occupancy": 2, "num_warps": pv_num_warps}
    # PV loop pipelining. With TMA loads, num_stages>=2 lets the next page's
    # loads overlap with the current MMA via mbarrier. The PV report shows
    # long_scoreboard=4.5 cycles avg on V loads at PV_LOOP_STAGES=2; bumping the
    # depth pays off when the grid is small enough that occupancy can absorb
    # the extra in-flight tile state — i.e. medium batch / large max_pages.
    # Larger pipelines hurt at small batch (more live state, fewer dim blocks).
    pv_loop_stages = 2 if num_queries <= 16 or max_pages <= 4 else 3

    # Page-stats kernel: per (query, head_block, page) program, does QK,
    # softmax stats, and packs probs into FP4 with the per-page local-max
    # scaling trick. The page-max correction is applied later by
    # prob_scale_kernel via p_sf in-place rescaling.
    page_stats_shape = (num_queries, max_pages, num_heads)
    page_max = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_page_max_buf",
        page_stats_shape,
        dtype=torch.float32,
        device=q_fp4.device,
    )
    page_sum = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_page_sum_buf",
        page_stats_shape,
        dtype=torch.float32,
        device=q_fp4.device,
    )

    pack_prob_in_page_stats = True
    _attn_page_stats_kernel[(num_queries, num_head_blocks, max_pages)](
        page_max,
        page_sum,
        p_fp4,
        p_sf,
        q_fp4,
        q_sf,
        kv_cache,
        sf_cache,
        global_scale,
        q_global_scale,
        src_page_ids,
        metadata.paged_kv_indptr_decode,
        kv_lens,
        src_page_ids.shape[0],
        kv_cache.shape[0],
        q_fp4.stride(0),
        q_fp4.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(2),
        kv_cache.stride(4),
        sf_cache.stride(0),
        page_max.stride(0),
        page_max.stride(1),
        p_fp4.stride(0),
        p_fp4.stride(1),
        p_fp4.shape[0],
        q_fp4.shape[0],
        sm_scale,
        NUM_HEADS=num_heads,
        Q_HEAD_D=q_head_dim,
        Q_STORAGE_HEAD_D=q_storage_head_dim,
        K_HEAD_D=head_dim,
        Q_RESIDUAL_D=q_residual_dim,
        K_RESIDUAL_D=FP4_MLA_K_RESIDUAL_DIM,
        PAGE_SIZE=metadata.page_size,
        FP4_BLOCK=FP4_BLOCK_SIZE,
        Q_SF_PER_TOKEN=q_sf_per_token,
        K_SF_PER_TOKEN=k_sf_per_token,
        SF_PER_PAGE=sf_per_page,
        P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
        QUERY_LEN_PER_SEQ=query_len_per_seq,
        MAX_PAGES=max_pages,
        BLOCK_H=block_h,
        BLOCK_T=block_t,
        BLOCK_K=block_k,
        FULL_BLOCK_END=full_block_end,
        TAIL_BLOCK_K=tail_block_k,
        USE_TMA_DATA_LOAD=use_tma_data_load,
        PACK_PROBS=pack_prob_in_page_stats,
        ASSUME_FULL_HEADS=assume_full_heads,
        ASSUME_FULL_PAGES=assume_full_pages,
        ASSUME_VALID_PAGES=assume_valid_pages,
        **page_stats_launch_meta,
    )
    # Two-level softmax-stats reduction. The single-level reduce launched only
    # (num_queries * num_head_blocks) CTAs, each serially walking all max_pages
    # twice -- at small batch that handful of CTAs left the GPU almost idle and
    # the reduce cost more than the QK matmul. Level 1 parallelizes the page
    # reduction across a page-group axis (online-softmax partials, pipelined);
    # level 2 reuses the existing reduce kernel to fold the few groups into the
    # global (max, denom). When the (query, head) grid already fills the GPU the
    # group count collapses to 1 and this degenerates to the original reduce.
    seqhead_ctas = num_queries * num_head_blocks
    # Aim for ~3 waves of level-1 CTAs so page loads have enough memory-level
    # parallelism to hide latency, while keeping the group count small enough
    # that the level-2 combine loop stays short.
    target_l1_ctas = 3 * sm_count
    num_reduce_groups = _ceil_div(target_l1_ctas, max(seqhead_ctas, 1))
    num_reduce_groups = max(1, min(num_reduce_groups, max_pages, 64))
    # The grouped (two-level) reduce needs an auxiliary workspace, and
    # _ensure_workspace_tensor can only (re)allocate it outside CUDA graph
    # capture. If a warmup forward did not already size that workspace (e.g. the
    # warmup batch took the single-level path), fall back to the single-level
    # reduce during capture so we never allocate mid-capture. The single-level
    # reduce is numerically identical (it just launches fewer CTAs).
    if num_reduce_groups > 1 and torch.cuda.is_current_stream_capturing():
        gmax = getattr(metadata, "_fp4_mla_attention_group_max_buf", None)
        gsum = getattr(metadata, "_fp4_mla_attention_group_sum_buf", None)
        groups_ready = (
            gmax is not None
            and gsum is not None
            and gmax.shape[0] >= num_queries
            and gmax.shape[1] >= num_reduce_groups
            and gmax.shape[2] >= num_heads
            and gsum.shape[0] >= num_queries
            and gsum.shape[1] >= num_reduce_groups
            and gsum.shape[2] >= num_heads
        )
        if not groups_ready:
            num_reduce_groups = 1
    if num_reduce_groups <= 1:
        _attn_reduce_stats_kernel[(num_queries, num_head_blocks)](
            max_scores,
            denom,
            page_max,
            page_sum,
            max_pages,
            max_scores.stride(0),
            page_max.stride(0),
            page_max.stride(1),
            NUM_HEADS=num_heads,
            MAX_PAGES=max_pages,
            BLOCK_H=block_h,
            **launch_meta,
        )
    else:
        group_pages = _ceil_div(max_pages, num_reduce_groups)
        num_reduce_groups = _ceil_div(max_pages, group_pages)
        group_max = _ensure_workspace_tensor(
            metadata,
            "_fp4_mla_attention_group_max_buf",
            (num_queries, num_reduce_groups, num_heads),
            dtype=torch.float32,
            device=q_fp4.device,
        )
        group_sum = _ensure_workspace_tensor(
            metadata,
            "_fp4_mla_attention_group_sum_buf",
            (num_queries, num_reduce_groups, num_heads),
            dtype=torch.float32,
            device=q_fp4.device,
        )
        _attn_group_reduce_stats_kernel[
            (num_queries, num_head_blocks, num_reduce_groups)
        ](
            group_max,
            group_sum,
            page_max,
            page_sum,
            max_pages,
            group_max.stride(0),
            group_max.stride(1),
            page_max.stride(0),
            page_max.stride(1),
            NUM_HEADS=num_heads,
            GROUP_PAGES=group_pages,
            BLOCK_H=block_h,
            PIPELINE_STAGES=min(group_pages, 4),
            **launch_meta,
        )
        _attn_reduce_stats_kernel[(num_queries, num_head_blocks)](
            max_scores,
            denom,
            group_max,
            group_sum,
            num_reduce_groups,
            max_scores.stride(0),
            group_max.stride(0),
            group_max.stride(1),
            NUM_HEADS=num_heads,
            MAX_PAGES=num_reduce_groups,
            BLOCK_H=block_h,
            **launch_meta,
        )
    _attn_prob_scale_kernel[(num_queries, num_head_blocks, max_pages)](
        p_sf,
        max_scores,
        denom,
        page_max,
        metadata.paged_kv_indptr_decode,
        kv_lens,
        src_page_ids.shape[0],
        max_scores.stride(0),
        page_max.stride(0),
        page_max.stride(1),
        NUM_HEADS=num_heads,
        PAGE_SIZE=metadata.page_size,
        SF_PER_PAGE=sf_per_page,
        QUERY_LEN_PER_SEQ=query_len_per_seq,
        MAX_PAGES=max_pages,
        BLOCK_H=block_h,
        ASSUME_FULL_HEADS=assume_full_heads,
        ASSUME_FULL_PAGES=assume_full_pages,
        ASSUME_VALID_PAGES=assume_valid_pages,
        **launch_meta,
    )
    num_dim_blocks = triton.cdiv(kv_lora_rank, block_v)
    v_packed = _get_triton_v_packed_cache(
        metadata,
        layer_idx,
        kv_cache,
        v_head_dim=kv_lora_rank,
        page_size=metadata.page_size,
        block_v=block_v,
        local_layer=local_layer,
        v_sf=v_sf,
        page_ids=src_page_ids,
    )
    if (
        v_packed is None
        and _triton_can_prepack_v(kv_lora_rank, metadata.page_size, block_v)
        and not torch.cuda.is_current_stream_capturing()
    ):
        v_packed = _update_triton_v_packed_cache(
            metadata,
            layer_idx,
            kv_cache,
            src_page_ids,
            v_head_dim=kv_lora_rank,
            page_size=metadata.page_size,
            block_v=block_v,
            local_layer=local_layer,
            v_sf=v_sf,
        )
    use_triton_v_packed_cache = v_packed is not None

    # PV page split: partition the page range across additional programs and
    # reduce in a follow-up kernel. ncu showed PV at waves/SM=0.49 for bs=32 —
    # PV is L1-bandwidth bound, so raising in-flight CTAs is the lever.
    # BLOCK_V is bounded below by the 16-byte TMA descriptor min-stride.
    # PV page split: ncu shows that with the current shape (bs=32, max_pages=256)
    # the PV kernel is L1-cache-throughput bound (long_scoreboard=4.5 cycles
    # avg, L1 global LD hit-rate <40%). Increasing the program count via page
    # splitting reduced waves/SM idle time but did NOT improve wall-time at
    # current shapes — the per-CTA L1 thrash is the limit. Gate the split off
    # by default; re-enable only for very small grids where occupancy is the
    # bottleneck rather than per-CTA L1 pressure.
    page_split = 1
    base_grid = num_queries * num_head_blocks * num_dim_blocks
    if max_pages >= 16 and base_grid < 148:
        for p in (8, 4, 2):
            if max_pages % p == 0 and max_pages // p >= 16 and base_grid * p <= 148 * 4:
                page_split = p
                break
    # The page-split PV path needs a partial-output workspace, which
    # _ensure_workspace_tensor can only (re)allocate outside CUDA graph capture.
    # Fall back to the unsplit PV (numerically identical) during capture unless a
    # warmup forward already sized that workspace, so capture never allocates.
    if page_split > 1 and torch.cuda.is_current_stream_capturing():
        pbuf = getattr(metadata, "_fp4_mla_attention_pv_partial_buf", None)
        partial_ready = (
            pbuf is not None
            and pbuf.shape[0] >= num_queries
            and pbuf.shape[1] >= page_split
            and pbuf.shape[2] >= num_heads
            and pbuf.shape[3] >= kv_lora_rank
        )
        if not partial_ready:
            page_split = 1
    if page_split > 1:
        pages_per_split = max_pages // page_split
        partial_out = _ensure_workspace_tensor(
            metadata,
            "_fp4_mla_attention_pv_partial_buf",
            (num_queries, page_split, num_heads, kv_lora_rank),
            dtype=torch.float32,
            device=q_fp4.device,
        )
        if use_triton_v_packed_cache:
            _attn_pv_prepacked_v_kernel[
                (num_queries, num_head_blocks, num_dim_blocks * page_split)
            ](
                output,
                p_fp4,
                p_sf,
                v_packed,
                v_sf,
                global_scale,
                src_page_ids,
                metadata.paged_kv_indptr_decode,
                kv_lens,
                src_page_ids.shape[0],
                kv_cache.shape[0],
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.shape[0] * output.shape[1],
                p_fp4.stride(0),
                p_fp4.stride(1),
                p_fp4.shape[0],
                v_sf.stride(0),
                NUM_HEADS=num_heads,
                V_HEAD_D=kv_lora_rank,
                PAGE_SIZE=metadata.page_size,
                FP4_BLOCK=FP4_BLOCK_SIZE,
                SF_PER_PAGE=sf_per_page,
                QUERY_LEN_PER_SEQ=query_len_per_seq,
                MAX_PAGES=max_pages,
                P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
                BLOCK_H=block_h,
                BLOCK_V=block_v,
                USE_TMA_P_LOAD=use_tma_data_load
                and assume_full_heads
                and assume_valid_pages,
                USE_TMA_OUT_STORE=use_tma_data_load
                and assume_full_heads
                and assume_full_v,
                PV_LOOP_STAGES=pv_loop_stages,
                ASSUME_FULL_HEADS=assume_full_heads,
                ASSUME_FULL_PAGES=assume_full_pages,
                ASSUME_FULL_V=assume_full_v,
                ASSUME_VALID_PAGES=assume_valid_pages,
                PAGE_SPLIT=page_split,
                PAGES_PER_SPLIT=pages_per_split,
                PARTIAL_OUT=True,
                partial_out_ptr=partial_out,
                partial_s0=partial_out.stride(0),
                partial_s1=partial_out.stride(1),
                partial_s2=partial_out.stride(2),
                partial_s3=partial_out.stride(3),
                **pv_launch_meta,
            )
        else:
            _attn_pv_kernel[
                (num_queries, num_head_blocks, num_dim_blocks * page_split)
            ](
                output,
                p_fp4,
                p_sf,
                kv_cache,
                kv_cache,
                v_sf,
                global_scale,
                src_page_ids,
                metadata.paged_kv_indptr_decode,
                kv_lens,
                src_page_ids.shape[0],
                kv_cache.shape[0],
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.shape[0] * output.shape[1],
                p_fp4.stride(0),
                p_fp4.stride(1),
                p_fp4.shape[0],
                kv_cache.stride(0),
                kv_cache.stride(2),
                kv_cache.stride(4),
                v_sf.stride(0),
                NUM_HEADS=num_heads,
                V_HEAD_D=kv_lora_rank,
                PAGE_SIZE=metadata.page_size,
                FP4_BLOCK=FP4_BLOCK_SIZE,
                SF_PER_PAGE=sf_per_page,
                QUERY_LEN_PER_SEQ=query_len_per_seq,
                MAX_PAGES=max_pages,
                P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
                BLOCK_H=block_h,
                BLOCK_V=block_v,
                USE_TMA_P_LOAD=use_tma_data_load
                and assume_full_heads
                and assume_valid_pages,
                USE_TMA_V_LOAD=use_tma_data_load and kv_lora_rank % block_v == 0,
                USE_PREPACKED_V=False,
                PV_LOOP_STAGES=pv_loop_stages,
                ASSUME_FULL_HEADS=assume_full_heads,
                ASSUME_FULL_PAGES=assume_full_pages,
                ASSUME_FULL_V=assume_full_v,
                ASSUME_VALID_PAGES=assume_valid_pages,
                PAGE_SPLIT=page_split,
                PAGES_PER_SPLIT=pages_per_split,
                PARTIAL_OUT=True,
                partial_out_ptr=partial_out,
                partial_s0=partial_out.stride(0),
                partial_s1=partial_out.stride(1),
                partial_s2=partial_out.stride(2),
                partial_s3=partial_out.stride(3),
                **pv_launch_meta,
            )
        _attn_pv_reduce_kernel[(num_queries, num_head_blocks, num_dim_blocks)](
            output,
            partial_out,
            global_scale,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            partial_out.stride(0),
            partial_out.stride(1),
            partial_out.stride(2),
            partial_out.stride(3),
            NUM_HEADS=num_heads,
            V_HEAD_D=kv_lora_rank,
            PAGE_SPLIT=page_split,
            P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
            BLOCK_H=block_h,
            BLOCK_V=block_v,
            ASSUME_FULL_HEADS=assume_full_heads,
            ASSUME_FULL_V=assume_full_v,
            **launch_meta,
        )
    else:
        if use_triton_v_packed_cache:
            _attn_pv_prepacked_v_kernel[(num_queries, num_head_blocks, num_dim_blocks)](
                output,
                p_fp4,
                p_sf,
                v_packed,
                v_sf,
                global_scale,
                src_page_ids,
                metadata.paged_kv_indptr_decode,
                kv_lens,
                src_page_ids.shape[0],
                kv_cache.shape[0],
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.shape[0] * output.shape[1],
                p_fp4.stride(0),
                p_fp4.stride(1),
                p_fp4.shape[0],
                v_sf.stride(0),
                NUM_HEADS=num_heads,
                V_HEAD_D=kv_lora_rank,
                PAGE_SIZE=metadata.page_size,
                FP4_BLOCK=FP4_BLOCK_SIZE,
                SF_PER_PAGE=sf_per_page,
                QUERY_LEN_PER_SEQ=query_len_per_seq,
                MAX_PAGES=max_pages,
                P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
                BLOCK_H=block_h,
                BLOCK_V=block_v,
                USE_TMA_P_LOAD=use_tma_data_load
                and assume_full_heads
                and assume_valid_pages,
                USE_TMA_OUT_STORE=use_tma_data_load
                and assume_full_heads
                and assume_full_v,
                PV_LOOP_STAGES=pv_loop_stages,
                ASSUME_FULL_HEADS=assume_full_heads,
                ASSUME_FULL_PAGES=assume_full_pages,
                ASSUME_FULL_V=assume_full_v,
                ASSUME_VALID_PAGES=assume_valid_pages,
                **pv_launch_meta,
            )
        else:
            _attn_pv_kernel[(num_queries, num_head_blocks, num_dim_blocks)](
                output,
                p_fp4,
                p_sf,
                kv_cache,
                kv_cache,
                v_sf,
                global_scale,
                src_page_ids,
                metadata.paged_kv_indptr_decode,
                kv_lens,
                src_page_ids.shape[0],
                kv_cache.shape[0],
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.shape[0] * output.shape[1],
                p_fp4.stride(0),
                p_fp4.stride(1),
                p_fp4.shape[0],
                kv_cache.stride(0),
                kv_cache.stride(2),
                kv_cache.stride(4),
                v_sf.stride(0),
                NUM_HEADS=num_heads,
                V_HEAD_D=kv_lora_rank,
                PAGE_SIZE=metadata.page_size,
                FP4_BLOCK=FP4_BLOCK_SIZE,
                SF_PER_PAGE=sf_per_page,
                QUERY_LEN_PER_SEQ=query_len_per_seq,
                MAX_PAGES=max_pages,
                P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
                BLOCK_H=block_h,
                BLOCK_V=block_v,
                USE_TMA_P_LOAD=use_tma_data_load
                and assume_full_heads
                and assume_valid_pages,
                USE_TMA_V_LOAD=use_tma_data_load and kv_lora_rank % block_v == 0,
                USE_PREPACKED_V=False,
                PV_LOOP_STAGES=pv_loop_stages,
                ASSUME_FULL_HEADS=assume_full_heads,
                ASSUME_FULL_PAGES=assume_full_pages,
                ASSUME_FULL_V=assume_full_v,
                ASSUME_VALID_PAGES=assume_valid_pages,
                **pv_launch_meta,
            )


def _swizzled_scale_size(rows: int, cols: int) -> int:
    scale_cols = _ceil_div(cols, FP4_BLOCK_SIZE)
    return _ceil_div(rows, 128) * _ceil_div(scale_cols, 4) * 32 * 16


def quantize_kimi_k3_nvfp4_query(
    owner: Any, q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 Q to the 640-channel residual-aware native layout."""
    from .kimi_k3_nvfp4_cache_kernels import _fp4_mla_quantize_q_kernel

    if q.dtype != torch.bfloat16 or q.shape[-1] != 576:
        raise ValueError(
            "Kimi-K3 NVFP4 query quantization requires contiguous BF16 "
            f"[..., 576] input, got {q.dtype} {tuple(q.shape)}."
        )
    q_2d = q.contiguous().view(-1, 576)
    rows = q_2d.shape[0]
    q_fp4 = _ensure_workspace_tensor(
        owner,
        "_kimi_k3_nvfp4_q",
        (rows, FP4_MLA_Q_PACKED_DIM),
        dtype=torch.uint8,
        device=q.device,
    )
    q_sf = _ensure_workspace_tensor(
        owner,
        "_kimi_k3_nvfp4_q_sf",
        (_swizzled_scale_size(rows, FP4_MLA_Q_LOGICAL_DIM),),
        dtype=torch.float8_e4m3fn,
        device=q.device,
    )
    global_scale = getattr(owner, "_kimi_k3_nvfp4_q_global_scale", None)
    if global_scale is None:
        global_scale = torch.tensor(
            [FP4_MLA_Q_GLOBAL_SCALE], dtype=torch.float32, device=q.device
        )
        owner._kimi_k3_nvfp4_q_global_scale = global_scale
    _fp4_mla_quantize_q_kernel[(rows, 36)](
        q_2d,
        q_fp4,
        q_sf,
        global_scale,
        rows,
        q_2d.stride(0),
        q_2d.stride(1),
        Q_SF_COLS=FP4_MLA_Q_LOGICAL_DIM // FP4_BLOCK_SIZE,
    )
    return q_fp4, q_sf


def update_kimi_k3_nvfp4_decode_cache(
    owner: Any,
    *,
    latent: torch.Tensor,
    q_pe: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    state_indices: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor | None,
    query_len_per_seq: int,
    rotary_cos_sin: torch.Tensor | None,
    max_state_slots: int,
    max_rewind: int,
) -> torch.Tensor:
    """Update native cache tiles and return the optionally rotated Q tail."""
    from .kimi_k3_nvfp4_cache_kernels import (
        _fp4_mla_generation_fused_qk_rope_cache_update_kernel,
    )

    if latent.dtype != torch.bfloat16 or latent.shape[-1] != 576:
        raise ValueError("Kimi-K3 NVFP4 cache update requires BF16 [tokens, 576].")
    num_tokens = latent.shape[0]
    if num_tokens % query_len_per_seq:
        raise ValueError("Kimi-K3 NVFP4 decode tokens must be request-uniform.")
    num_seqs = num_tokens // query_len_per_seq
    active_table = block_table[:num_seqs]
    max_pages = active_table.shape[1]
    page_ids = active_table.contiguous().view(-1)
    hp_page_ids = (
        state_indices[:num_seqs, None].expand(num_seqs, max_pages).contiguous().view(-1)
    )
    indptr = _ensure_workspace_tensor(
        owner,
        "_kimi_k3_nvfp4_update_indptr",
        (num_seqs + 1,),
        dtype=torch.int32,
        device=latent.device,
    )
    torch.arange(
        0,
        (num_seqs + 1) * max_pages,
        max_pages,
        out=indptr,
        dtype=torch.int32,
        device=latent.device,
    )
    gen_lens = _ensure_workspace_tensor(
        owner,
        "_kimi_k3_nvfp4_gen_lens",
        (num_seqs,),
        dtype=torch.int32,
        device=latent.device,
    )
    local_slots = slot_mapping[:num_tokens].view(num_seqs, query_len_per_seq) >= 0
    torch.sum(local_slots, dim=1, dtype=torch.int32, out=gen_lens)
    gen_offsets = _ensure_workspace_tensor(
        owner,
        "_kimi_k3_nvfp4_gen_offsets",
        (num_seqs,),
        dtype=torch.int64,
        device=latent.device,
    )
    torch.argmax(local_slots.to(torch.int32), dim=1, out=gen_offsets)

    hp_pool_size = HP_BLOCK_SIZE + max_rewind
    hp_pool = _ensure_workspace_tensor(
        owner,
        "_kimi_k3_nvfp4_hp_pool",
        (max_state_slots, 1, hp_pool_size * 576),
        dtype=torch.bfloat16,
        device=latent.device,
    )
    kv, k_sf, v_sf_2d = split_kimi_k3_nvfp4_cache(cache)
    # The page allocation is byte-addressed, but the Triton cache-update kernel
    # stores E4M3 scale values.  Reinterpret the scale regions rather than
    # letting Triton numerically cast those values to uint8 on store.
    k_sf = k_sf.view(torch.float8_e4m3fn)
    v_sf = v_sf_2d.view(torch.float8_e4m3fn).unsqueeze(0)
    global_scale = getattr(owner, "_kimi_k3_nvfp4_kv_global_scale", None)
    if global_scale is None:
        global_scale = torch.tensor(
            [FP4_MLA_KV_GLOBAL_SCALE], dtype=torch.float32, device=latent.device
        )
        owner._kimi_k3_nvfp4_kv_global_scale = global_scale
    q_global_scale = getattr(owner, "_kimi_k3_nvfp4_q_global_scale", None)
    if q_global_scale is None:
        q_global_scale = torch.tensor(
            [FP4_MLA_Q_GLOBAL_SCALE], dtype=torch.float32, device=latent.device
        )
        owner._kimi_k3_nvfp4_q_global_scale = q_global_scale

    apply_rope = rotary_cos_sin is not None
    q_rope_out = torch.empty_like(q_pe) if apply_rope else q_pe
    q_full_dummy = latent.unsqueeze(1).expand(-1, q_pe.shape[1], -1)
    q_fp4_dummy = torch.empty((1, 1), dtype=torch.uint8, device=latent.device)
    q_sf_dummy = torch.empty((1,), dtype=torch.float8_e4m3fn, device=latent.device)
    if apply_rope and positions is None:
        raise ValueError("Kimi-K3 NVFP4 RoPE cache updates require token positions.")
    rope_positions = positions if positions is not None else gen_lens
    rotary_table = rotary_cos_sin if apply_rope else global_scale
    max_gen_len = query_len_per_seq
    block_q_heads = 32
    q_head_blocks = _ceil_div(q_pe.shape[1], block_q_heads)
    q_prefix_block_dim = (
        FP4_MLA_Q1_PREFIX_BLOCK_DIM if max_gen_len == 1 else FP4_MLA_Q_PREFIX_BLOCK_DIM
    )
    q_prefix_blocks = FP4_MLA_Q_PREFIX_DIM // q_prefix_block_dim
    q_work_blocks = q_prefix_blocks + 1
    num_dim_blocks = 576 // FP4_BLOCK_SIZE
    max_gen_tiles = _ceil_div(max_gen_len + FP4_BLOCK_SIZE - 1, FP4_BLOCK_SIZE)
    kv_work_blocks = 512 // FP4_BLOCK_SIZE + 1 if max_gen_len == 1 else num_dim_blocks
    grid = (
        num_seqs,
        max(
            kv_work_blocks,
            max_gen_len * q_head_blocks * q_work_blocks if apply_rope else 0,
        ),
    )
    _fp4_mla_generation_fused_qk_rope_cache_update_kernel[grid](
        kv,
        k_sf,
        v_sf,
        kv,
        hp_pool,
        latent,
        global_scale,
        q_global_scale,
        rotary_table,
        q_pe,
        q_rope_out,
        q_full_dummy,
        q_fp4_dummy,
        q_sf_dummy,
        rope_positions,
        seq_lens[:num_seqs],
        gen_lens,
        gen_offsets,
        page_ids,
        hp_page_ids,
        indptr,
        page_ids.numel(),
        hp_page_ids.numel(),
        indptr.numel(),
        kv.shape[0],
        hp_pool.shape[0],
        1,
        0,
        0,
        FP4_MLA_TOKENS_PER_BLOCK,
        kv.stride(0),
        kv.stride(2),
        kv.stride(4),
        k_sf.stride(0),
        hp_pool.stride(0),
        hp_pool.stride(1),
        v_sf.stride(0),
        v_sf.stride(1),
        0,
        0,
        q_pe.stride(0),
        q_pe.stride(1),
        q_pe.stride(2),
        q_rope_out.stride(0),
        q_rope_out.stride(1),
        q_rope_out.stride(2),
        HEAD_D=576,
        V_HEAD_D=512,
        HP_BLOCK=HP_BLOCK_SIZE,
        HP_POOL_SIZE=hp_pool_size,
        FP4_BLOCK=FP4_BLOCK_SIZE,
        SF_PER_TOKEN=FP4_MLA_K_SCALE_BYTES_PER_TOKEN,
        SF_PER_PAGE=FP4_MLA_TOKENS_PER_BLOCK // FP4_BLOCK_SIZE,
        K_RESIDUAL_D=FP4_MLA_K_RESIDUAL_DIM,
        STORE_K_RESIDUAL=True,
        FUSE_ROPE_CACHE_STORE=True,
        APPLY_ROPE=apply_rope,
        WRITE_V_PACKED=False,
        MAX_GEN_TILES=max_gen_tiles,
        ROPE_DIM=64,
        ROPE_PAIR_BLOCK=32,
        NUM_DIM_BLOCKS=num_dim_blocks,
        NUM_Q_HEADS=q_pe.shape[1],
        Q_HEAD_BLOCKS=max(q_head_blocks, 1),
        BLOCK_Q_HEADS=block_q_heads,
        Q_PREFIX_D=FP4_MLA_Q_PREFIX_DIM,
        Q_PREFIX_BLOCK_D=q_prefix_block_dim,
        Q_PREFIX_BLOCKS=q_prefix_blocks,
        Q_PREFIX_BLOCKS_PER_PROGRAM=1,
        Q_WORK_BLOCKS=q_work_blocks,
        Q_SF_COLS=FP4_MLA_Q_LOGICAL_DIM // FP4_BLOCK_SIZE,
        WRITE_Q=False,
        Q1_KV_BLOCKS_PER_PROGRAM=1,
        USE_EXTERNAL_ROPE_POSITIONS=positions is not None,
        QUERY_STRIDE=query_len_per_seq,
        PROCESS_Q=apply_rope,
        maxnreg=56,
    )
    return q_rope_out if apply_rope else q_pe


def update_kimi_k3_nvfp4_prefill_cache(
    owner: Any,
    *,
    latent: torch.Tensor,
    query: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    rotary_cos_sin: torch.Tensor | None,
    max_state_slots: int,
    max_rewind: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize prefill latents directly into vLLM's native FP4 pages.

    ``slot_mapping`` is authoritative for physical cache placement.  This is
    important under DCP, where cache positions are local while ``positions``
    must remain global for Q/K RoPE.
    """
    from .kimi_k3_nvfp4_cache_kernels import (
        _fp4_mla_context_cache_update_kernel,
    )

    if latent.dtype != torch.bfloat16 or latent.ndim != 2 or latent.shape[1] != 576:
        raise ValueError("Kimi-K3 NVFP4 prefill requires BF16 [tokens, 576] latents.")
    if query.dtype != torch.bfloat16 or query.ndim != 3:
        raise ValueError("Kimi-K3 NVFP4 prefill requires a BF16 3D query tensor.")
    num_tokens = latent.shape[0]
    if query.shape[0] != num_tokens or positions.numel() != num_tokens:
        raise ValueError("Kimi-K3 NVFP4 prefill tensors must have matching tokens.")
    num_contexts = query_start_loc.numel() - 1
    token_ids = _ensure_workspace_tensor(
        owner,
        "_kimi_k3_nvfp4_prefill_token_ids",
        (num_tokens,),
        dtype=torch.int32,
        device=latent.device,
    )
    torch.arange(num_tokens, out=token_ids, dtype=torch.int32, device=latent.device)
    batch_indices = _ensure_workspace_tensor(
        owner,
        "_kimi_k3_nvfp4_prefill_batch_indices",
        (num_tokens,),
        dtype=torch.int32,
        device=latent.device,
    )
    torch.bucketize(
        token_ids,
        query_start_loc[1:],
        out_int32=True,
        right=True,
        out=batch_indices,
    )

    hp_pool_size = HP_BLOCK_SIZE + max_rewind
    hp_pool = _ensure_workspace_tensor(
        owner,
        "_kimi_k3_nvfp4_hp_pool",
        (max_state_slots, 1, hp_pool_size * 576),
        dtype=torch.bfloat16,
        device=latent.device,
    )
    kv, k_sf, v_sf_2d = split_kimi_k3_nvfp4_cache(cache)
    k_sf = k_sf.view(torch.float8_e4m3fn)
    v_sf = v_sf_2d.view(torch.float8_e4m3fn).unsqueeze(0)
    global_scale = getattr(owner, "_kimi_k3_nvfp4_kv_global_scale", None)
    if global_scale is None:
        global_scale = torch.tensor(
            [FP4_MLA_KV_GLOBAL_SCALE], dtype=torch.float32, device=latent.device
        )
        owner._kimi_k3_nvfp4_kv_global_scale = global_scale

    indptr = _ensure_workspace_tensor(
        owner,
        "_kimi_k3_nvfp4_prefill_indptr",
        (num_contexts + 1,),
        dtype=torch.int32,
        device=latent.device,
    )
    torch.arange(
        num_contexts + 1,
        out=indptr,
        dtype=torch.int32,
        device=latent.device,
    )
    dummy = indptr
    apply_rope = rotary_cos_sin is not None
    rotary_table = rotary_cos_sin if apply_rope else global_scale
    num_dim_blocks = 576 // FP4_BLOCK_SIZE
    block_q_heads = 16
    q_head_blocks = _ceil_div(query.shape[1], block_q_heads) if apply_rope else 0
    _fp4_mla_context_cache_update_kernel[(num_tokens, num_dim_blocks + q_head_blocks)](
        kv,
        k_sf,
        v_sf,
        kv,
        latent,
        query,
        global_scale,
        rotary_table,
        hp_pool,
        dummy,
        batch_indices,
        positions,
        slot_mapping,
        state_indices[:num_contexts],
        dummy,
        indptr,
        dummy.numel(),
        indptr.numel(),
        num_tokens,
        kv.shape[0],
        1,
        num_contexts,
        hp_pool.shape[0],
        0,
        num_tokens,
        0,
        0,
        FP4_MLA_TOKENS_PER_BLOCK,
        kv.stride(0),
        kv.stride(2),
        kv.stride(4),
        k_sf.stride(0),
        latent.stride(0),
        latent.stride(1),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        v_sf.stride(0),
        v_sf.stride(1),
        0,
        0,
        hp_pool.stride(0),
        hp_pool.stride(1),
        HEAD_D=576,
        V_HEAD_D=512,
        HP_BLOCK=FP4_BLOCK_SIZE,
        HP_POOL_SIZE=hp_pool_size,
        FP4_BLOCK=FP4_BLOCK_SIZE,
        SF_PER_TOKEN=FP4_MLA_K_SCALE_BYTES_PER_TOKEN,
        SF_PER_PAGE=FP4_MLA_TOKENS_PER_BLOCK // FP4_BLOCK_SIZE,
        K_RESIDUAL_D=FP4_MLA_K_RESIDUAL_DIM,
        STORE_K_RESIDUAL=True,
        ROPE_DIM=64,
        APPLY_K_ROPE=apply_rope,
        APPLY_Q_ROPE=apply_rope,
        NUM_DIM_BLOCKS=num_dim_blocks,
        NUM_Q_HEADS=query.shape[1],
        Q_NOPE_DIM=query.shape[2] - 64,
        BLOCK_Q_HEADS=block_q_heads,
        POOL_HEAD_D=576,
        STORE_HP_TAIL=True,
        WRITE_V_PACKED=False,
        USE_SLOT_MAPPING=True,
    )
    return latent, query


def split_kimi_k3_nvfp4_cache(
    cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expose native page regions without copying the vLLM allocation."""
    if cache.shape[0] == 0:
        raise ValueError("Kimi-K3 NVFP4 cache must contain at least one page.")
    raw = cache.view(torch.uint8)
    page_bytes = FP4_MLA_TOKENS_PER_BLOCK * FP4_MLA_BYTES_PER_TOKEN
    if raw.stride(0) < page_bytes:
        raise ValueError(
            f"Kimi-K3 NVFP4 page stride {raw.stride(0)} is smaller than "
            f"the native {page_bytes}-byte page."
        )
    base = raw.storage_offset()
    page_stride = raw.stride(0)
    num_pages = raw.shape[0]
    data_bytes = FP4_MLA_TOKENS_PER_BLOCK * FP4_MLA_DATA_BYTES_PER_TOKEN
    k_scale_bytes = FP4_MLA_TOKENS_PER_BLOCK * FP4_MLA_K_SCALE_BYTES_PER_TOKEN
    kv = torch.as_strided(
        raw,
        (num_pages, 1, FP4_MLA_TOKENS_PER_BLOCK, 1, FP4_MLA_DATA_BYTES_PER_TOKEN),
        (
            page_stride,
            data_bytes,
            FP4_MLA_DATA_BYTES_PER_TOKEN,
            FP4_MLA_DATA_BYTES_PER_TOKEN,
            1,
        ),
        base,
    )
    k_sf = torch.as_strided(
        raw,
        (
            num_pages,
            FP4_MLA_TOKENS_PER_BLOCK,
            FP4_MLA_K_SCALE_BYTES_PER_TOKEN,
        ),
        (page_stride, FP4_MLA_K_SCALE_BYTES_PER_TOKEN, 1),
        base + data_bytes,
    )
    v_sf = torch.as_strided(
        raw,
        (
            num_pages,
            FP4_MLA_TOKENS_PER_BLOCK * FP4_MLA_V_SCALE_BYTES_PER_TOKEN,
        ),
        (
            page_stride,
            1,
        ),
        base + data_bytes + k_scale_bytes,
    )
    return kv, k_sf, v_sf


def run_kimi_k3_nvfp4_attention(
    owner: Any,
    *,
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    query_len_per_seq: int,
    sm_scale: float,
) -> None:
    """Run native FP4 QK/PV attention from a vLLM opaque cache page."""
    if cache.shape[2] != FP4_MLA_TOKENS_PER_BLOCK:
        raise ValueError(
            "Kimi-K3 NVFP4 attention requires a 128-token kernel page, got "
            f"{cache.shape[2]}."
        )
    num_queries, num_heads = output.shape[:2]
    if num_queries % query_len_per_seq:
        raise ValueError("NVFP4 query rows must be uniform across requests.")
    num_seqs = num_queries // query_len_per_seq
    active_table = block_table[:num_seqs]
    max_pages = active_table.shape[1]
    page_ids = active_table.contiguous().view(-1)
    indptr = _ensure_workspace_tensor(
        owner,
        "_kimi_k3_nvfp4_indptr",
        (num_seqs + 1,),
        dtype=torch.int32,
        device=output.device,
    )
    torch.arange(
        0,
        (num_seqs + 1) * max_pages,
        max_pages,
        out=indptr,
        dtype=torch.int32,
        device=output.device,
    )
    owner.page_size = FP4_MLA_TOKENS_PER_BLOCK
    owner.paged_kv_indptr_decode = indptr
    kv, k_sf, v_sf = split_kimi_k3_nvfp4_cache(cache)

    total_p_rows = num_queries * max_pages * num_heads
    p_fp4 = _ensure_workspace_tensor(
        owner,
        "_kimi_k3_nvfp4_p",
        (max(total_p_rows, 1), FP4_MLA_TOKENS_PER_BLOCK // 2),
        dtype=torch.uint8,
        device=output.device,
    )[:total_p_rows]
    p_sf = _ensure_workspace_tensor(
        owner,
        "_kimi_k3_nvfp4_p_sf",
        (max(_swizzled_scale_size(total_p_rows, FP4_MLA_TOKENS_PER_BLOCK), 1),),
        dtype=torch.float8_e4m3fn,
        device=output.device,
    )
    stats_shape = (num_queries, num_heads)
    max_scores = _ensure_workspace_tensor(
        owner,
        "_kimi_k3_nvfp4_max",
        stats_shape,
        dtype=torch.float32,
        device=output.device,
    )
    denom = _ensure_workspace_tensor(
        owner,
        "_kimi_k3_nvfp4_denom",
        stats_shape,
        dtype=torch.float32,
        device=output.device,
    )
    global_scale = getattr(owner, "_kimi_k3_nvfp4_kv_global_scale", None)
    if global_scale is None:
        global_scale = torch.tensor(
            [FP4_MLA_KV_GLOBAL_SCALE], dtype=torch.float32, device=output.device
        )
        owner._kimi_k3_nvfp4_kv_global_scale = global_scale
    q_global_scale = getattr(owner, "_kimi_k3_nvfp4_q_global_scale", None)
    if q_global_scale is None:
        q_global_scale = torch.tensor(
            [FP4_MLA_Q_GLOBAL_SCALE], dtype=torch.float32, device=output.device
        )
        owner._kimi_k3_nvfp4_q_global_scale = q_global_scale

    _run_triton_attention_decode(
        metadata=owner,
        layer_idx=0,
        local_layer=0,
        q_fp4=q_fp4,
        q_sf=q_sf.contiguous().view(-1),
        kv_cache=kv,
        sf_cache=k_sf.view(torch.float8_e4m3fn),
        v_sf=v_sf.view(torch.float8_e4m3fn),
        global_scale=global_scale,
        src_page_ids=page_ids,
        kv_lens=seq_lens[:num_seqs],
        p_fp4=p_fp4,
        p_sf=p_sf,
        max_scores=max_scores,
        denom=denom,
        output=output,
        num_queries=num_queries,
        num_heads=num_heads,
        head_dim=576,
        kv_lora_rank=512,
        q_residual_dim=FP4_MLA_Q_RESIDUAL_DIM,
        query_len_per_seq=query_len_per_seq,
        max_pages=max_pages,
        sm_scale=float(sm_scale),
        q_global_scale=q_global_scale,
    )
