# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3's KDA side of a speculative decode step's input prep in one kernel.

Before every verify step the hybrid model state runs, as separate launches: the
Mamba "align" preprocess (advance each request's running state block, record the
pre-copy source, reset the accepted count on a block crossing), the gather of each
row's accepted count, the aligned state indices of every Mamba group, and each KDA
metadata builder's staging of its CUDA-graph buffers (state indices, query start
locations, accepted counts). On a steady spec-decode step (every request verifies
its drafts, FULL graph) they are one row-parallel kernel here, with the same values:
``KdaStepPrep.run`` replays ``preprocess_mamba_align_fused_kernel``, the gather in
``MambaHybridModelState.gather_num_accepted_tokens``,
``get_aligned_state_indices_multi_group_kernel`` and ``stage_spec_decode_metadata``.
The state migration itself (``precopy_mamba_align_fused_kernel``) stays a separate
launch: it only copies on a block crossing and then needs many CTAs.

The model state takes the runner's step prep over for such a step
(``defers_step_prep``), so the same launch also replays, with the same values, the
runner's ``expand_idx_mapping``, ``prepare_pos_seq_lens`` and
``combine_sampled_and_draft_tokens`` (in the row's program; positions and sequence
lengths are recomputed where other programs need them, not read back),
``_gather_block_tables_kernel`` (every KV cache group's batch-order block table,
padded rows zeroed; in the row's program, whose KDA work reads the gathered rows
after a CTA barrier), ``_compute_slot_mappings_kernel`` (one program per (group,
request) and per (group, padding chunk) through ``max_num_tokens``) and
``_dcp_local_seq_lens_kernel`` (row 0).
"""

from dataclasses import dataclass
from functools import partial

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID
from vllm.v1.worker.gpu.buffer_utils import _load_ptr

GATHER_BLOCK = tl.constexpr(1024)  # the runner's block-table gather width
TOKEN_BLOCK = tl.constexpr(128)  # the runner's slot-mapping token block
PAD_BLOCK = tl.constexpr(1024)  # the runner's slot-mapping padding block
PAD_SLOT = tl.constexpr(PAD_SLOT_ID)


@triton.jit(
    do_not_specialize=["num_reqs", "num_rows", "max_num_tokens", "num_tokens_padded"]
)
def _kda_step_prep_kernel(
    # Mamba align preprocess, indexed by request slot (idx_mapping).
    idx_mapping_ptr,  # [num_reqs]
    state_idx_ptr,  # [max_num_reqs] running state block column (advanced here)
    num_computed_ptr,  # [max_num_reqs]
    query_start_loc_ptr,  # [num_rows + 1]
    num_accepted_ptr,  # [max_num_reqs] (reset to 1 on a block crossing)
    src_col_ptr,  # [max_num_reqs] pre-copy source column
    src_off_ptr,  # [max_num_reqs] pre-copy accepted-token bias
    # Per batch row.
    num_accepted_out_ptr,  # [num_rows] gathered accepted counts
    seq_lens_ptr,  # [max_num_reqs] (written: prepare_pos_seq_lens)
    # Aligned state indices of every Mamba group.
    block_table_ptrs_ptr,  # int64 [NUM_GROUPS] (batch-order input block tables)
    block_table_stride,
    aligned_ptr,  # [NUM_GROUPS, max_num_reqs, NUM_SLOTS]
    aligned_stride_group,
    aligned_stride_row,
    # KDA builders' graph buffers.
    staged_state_ptrs_ptr,  # int64 [NUM_STAGED] -> [batch, STAGED_SLOTS] int32
    staged_state_stride,
    staged_qsl_ptrs_ptr,  # int64 [NUM_STAGED] -> [batch + 1] int32
    staged_accepted_ptrs_ptr,  # int64 [NUM_STAGED] -> [batch] int32
    staged_group_ptr,  # int32 [NUM_STAGED]: Mamba group index of each builder
    num_reqs,  # real requests (all spec decodes)
    num_rows,  # rows incl. CUDA-graph padding (the builders' batch size)
    # The runner's tables: BlockTables' request-indexed and batch-order tables of
    # every KV cache group, their slot mappings, the DCP local lengths.
    src_tables_ptr,  # uint64 [TABLE_GROUPS]
    dst_tables_ptr,  # uint64 [TABLE_GROUPS]
    table_strides_ptr,  # int64 [TABLE_GROUPS] (= a row's block capacity)
    num_blocks_ptr,  # int32 [TABLE_GROUPS, max_num_reqs]
    num_blocks_stride,
    block_sizes_ptr,  # int32 [TABLE_GROUPS]
    mapping_enabled_ptr,  # bool [TABLE_GROUPS]
    dcp_sharded_ptr,  # bool [TABLE_GROUPS]
    positions_ptr,  # int64 [num_tokens]
    slot_mappings_ptr,  # int64 [TABLE_GROUPS, max_num_tokens]
    slot_mappings_stride,
    max_num_tokens,
    dcp_out_ptr,  # int32 [max_num_reqs]
    max_num_reqs,
    cp_rank,
    pad_programs,  # slot-mapping padding programs per group
    is_padding_ptr,  # bool [max_num_tokens] (IS_PADDING)
    num_tokens_padded,
    # The runner's step inputs: expand_idx_mapping, prepare_pos_seq_lens and
    # combine_sampled_and_draft_tokens.
    cu_num_logits_ptr,  # [num_reqs + 1]
    expanded_idx_ptr,  # int64 [num_logits]
    expanded_pos_ptr,  # int32 [num_logits]
    logits_indices_ptr,  # int64 [num_logits]
    input_ids_ptr,  # [max_num_tokens]
    last_sampled_ptr,  # [max_num_reqs]
    prefill_len_ptr,  # [max_num_reqs]
    draft_tokens_ptr,  # [max_num_reqs, num_speculative_steps]
    draft_tokens_stride,
    # An attention group's kernel block table (KERNEL_GROUP >= 0: the KV cache group
    # of AttentionGroup.map_to_kernel_block_table, ubatch 0).
    kernel_table_ptr,  # int32 [max_num_reqs, kernel_table_stride]
    kernel_table_stride,
    EXPAND_BLOCK: tl.constexpr,
    NUM_NEW_SAMPLED: tl.constexpr,
    REQ_BLOCK: tl.constexpr,
    TABLE_GROUPS: tl.constexpr,
    IS_PADDING: tl.constexpr,  # also prepare_inputs' is_padding fills
    KERNEL_GROUP: tl.constexpr,  # -1: none
    KERNEL_BLOCKS: tl.constexpr,  # kernel blocks per block
    KERNEL_MULT: tl.constexpr,  # kernel block j of block b is b * KERNEL_MULT + j
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    STAGED_SLOTS: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
    NUM_STAGED: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0)  # 0 .. num_rows (the last one stages qsl[num_rows])
    if row > num_rows:
        q = row - num_rows - 1
        if q < TABLE_GROUPS * num_rows:
            # _compute_slot_mappings_kernel's request programs: (group, row).
            _slot_mappings_row(
                q % num_rows, q // num_rows, num_reqs, idx_mapping_ptr,
                query_start_loc_ptr, num_computed_ptr, src_tables_ptr,
                table_strides_ptr, block_sizes_ptr, mapping_enabled_ptr,
                dcp_sharded_ptr, slot_mappings_ptr, slot_mappings_stride, cp_rank,
                CP_SIZE, CP_INTERLEAVE,
            )  # fmt: skip
        else:
            # Its padding programs: (group, PAD_BLOCK chunk).
            q -= TABLE_GROUPS * num_rows
            group, chunk = q // pad_programs, q % pad_programs
            actual = tl.load(query_start_loc_ptr + num_reqs)
            offset = chunk * PAD_BLOCK + tl.arange(0, PAD_BLOCK)
            mask = (offset >= actual) & (offset < max_num_tokens)
            pad = slot_mappings_ptr + group * slot_mappings_stride + offset
            tl.store(pad, PAD_SLOT, mask=mask)
            if IS_PADDING:  # noqa: SIM102
                if group == 0:  # is_padding[:num_tokens_padded]: token >= actual
                    padded = offset < num_tokens_padded
                    tl.store(is_padding_ptr + offset, offset >= actual, mask=padded)
        return
    # A row program's loads, all issued before its first phase: in serving the step
    # prep runs right after the verify graph with a cold L2, so each level of
    # dependent loads costs an HBM round trip (the block counts and prefill lengths
    # are device copies: ``StepInputs``). Two levels remain: the row's request, then
    # what it indexes; the pointer tables load beside the first.
    groups = tl.arange(0, triton.next_power_of_2(TABLE_GROUPS))
    in_groups = groups < TABLE_GROUPS
    src_tables = tl.load(src_tables_ptr + groups, mask=in_groups, other=0)
    dst_tables = tl.load(dst_tables_ptr + groups, mask=in_groups, other=0)
    strides = tl.load(table_strides_ptr + groups, mask=in_groups, other=0)
    mamba = tl.arange(0, triton.next_power_of_2(NUM_GROUPS))
    aligned_tables = tl.load(
        block_table_ptrs_ptr + mamba, mask=mamba < NUM_GROUPS, other=0
    )
    staged = tl.arange(0, triton.next_power_of_2(NUM_STAGED))
    in_staged = staged < NUM_STAGED
    staged_states = tl.load(staged_state_ptrs_ptr + staged, mask=in_staged, other=0)
    staged_qsls = tl.load(staged_qsl_ptrs_ptr + staged, mask=in_staged, other=0)
    staged_counts = tl.load(staged_accepted_ptrs_ptr + staged, mask=in_staged, other=0)
    staged_groups = tl.load(staged_group_ptr + staged, mask=in_staged, other=0)
    is_real = row < num_reqs
    is_row = row < num_rows
    req = tl.load(idx_mapping_ptr + row, mask=is_real, other=0)
    start = tl.load(query_start_loc_ptr + row, mask=is_real, other=0)
    end = tl.load(query_start_loc_ptr + row + 1, mask=is_real, other=0)
    query_start = tl.load(query_start_loc_ptr + tl.minimum(row, num_reqs))
    logits_start_idx = tl.load(cu_num_logits_ptr + row, mask=is_real, other=0)
    logits_end_idx = tl.load(cu_num_logits_ptr + row + 1, mask=is_real, other=0)
    counts = tl.load(
        num_blocks_ptr + groups * num_blocks_stride + req,
        mask=in_groups & is_real,
        other=0,
    )
    prefill_len = tl.load(prefill_len_ptr + req, mask=is_real, other=0)
    num_computed = tl.load(num_computed_ptr + req, mask=is_real, other=0)
    state_idx = tl.load(state_idx_ptr + req, mask=is_real, other=0)
    num_accepted = tl.load(num_accepted_ptr + req, mask=is_real, other=0)
    last_sampled = tl.load(last_sampled_ptr + req, mask=is_real, other=0)
    num_logits = logits_end_idx - logits_start_idx
    e = tl.arange(0, EXPAND_BLOCK)
    in_drafts = is_real & (e < num_logits - NUM_NEW_SAMPLED)
    drafts = tl.load(
        draft_tokens_ptr + req * draft_tokens_stride + e, mask=in_drafts, other=0
    )

    _gather_rows(
        row, num_reqs, num_rows, req, counts, groups, src_tables, dst_tables,
        strides, kernel_table_ptr, kernel_table_stride, TABLE_GROUPS, KERNEL_GROUP,
        KERNEL_BLOCKS, KERNEL_MULT,
    )  # fmt: skip
    # The KDA work below reads this row's gathered tables.
    tl.debug_barrier()
    if CP_SIZE != 1:  # noqa: SIM102
        if row == num_rows:  # (the program without a row's gather)
            _dcp_local_seq_lens(
                dcp_out_ptr, idx_mapping_ptr, num_computed_ptr, query_start_loc_ptr,
                num_reqs, max_num_reqs, cp_rank, CP_SIZE, CP_INTERLEAVE, REQ_BLOCK,
            )  # fmt: skip
    # prepare_pos_seq_lens's value for the row (0 past the real requests).
    row_seq_len = _step_inputs_row(
        row, num_reqs, max_num_reqs, req, num_computed, start, end,
        logits_start_idx, num_logits, prefill_len, last_sampled, drafts, in_drafts,
        positions_ptr, seq_lens_ptr, expanded_idx_ptr, expanded_pos_ptr,
        logits_indices_ptr, input_ids_ptr, EXPAND_BLOCK, NUM_NEW_SAMPLED, REQ_BLOCK,
    )  # fmt: skip

    # preprocess_mamba_align_fused_kernel, then the gather of the (reset) count.
    accepted = tl.zeros((), tl.int32) + 1
    if is_real:
        tl.store(src_col_ptr + req, state_idx)
        tl.store(src_off_ptr + req, tl.maximum(num_accepted - 1, 0))
        computed_after = num_computed + end - start
        new_state_idx = (computed_after + MAMBA_BLOCK_SIZE - 1) // MAMBA_BLOCK_SIZE - 1
        tl.store(state_idx_ptr + req, new_state_idx)
        should_reset = (state_idx >= 0) & (state_idx != new_state_idx)
        tl.store(num_accepted_ptr + req, 1, mask=should_reset)
        accepted = tl.where(should_reset, 1, num_accepted)
    tl.store(num_accepted_out_ptr + row, accepted, mask=is_row)

    # get_aligned_state_indices_multi_group_kernel for this row.
    slots = tl.arange(0, BLOCK_SLOTS)
    seq_len = tl.where(is_row, row_seq_len, 1)
    first_slot = tl.maximum((seq_len - 1) // CACHE_BLOCK_SIZE, 0)
    for g in tl.static_range(NUM_GROUPS):
        table = _pick_ptr(aligned_tables, mamba == g)
        indices = tl.load(
            table + row * block_table_stride + first_slot + slots,
            mask=is_row & (slots < NUM_SLOTS),
        )
        tl.store(
            aligned_ptr + g * aligned_stride_group + row * aligned_stride_row + slots,
            indices,
            mask=is_row & (slots < NUM_SLOTS),
        )

    # stage_spec_decode_metadata of every KDA builder, with num_spec_decodes =
    # num_reqs and batch_size = num_rows. The state indices are re-read from the
    # group's block table (the aligned values above, without a read-back).
    for k in tl.static_range(NUM_STAGED):
        state_out = _pick_ptr(staged_states, staged == k)
        qsl_out = _pick_ptr(staged_qsls, staged == k)
        accepted_out = _pick_ptr(staged_counts, staged == k)
        group = tl.sum(tl.where(staged == k, staged_groups, 0))
        table = _pick_ptr(aligned_tables, mamba == group)
        state = tl.load(
            table + row * block_table_stride + first_slot + slots,
            mask=is_real & (slots < STAGED_SLOTS),
            other=NULL_STATE_ID,
        )
        tl.store(
            state_out + row * staged_state_stride + slots,
            state,
            mask=is_row & (slots < STAGED_SLOTS),
        )
        tl.store(accepted_out + row, accepted, mask=is_row)
        tl.store(qsl_out + row, query_start)


@triton.jit
def _pick_ptr(addresses, select):
    """The int32 pointer whose address ``select`` picks from a small vector."""
    address = tl.sum(tl.where(select, addresses, 0))
    return address.to(tl.pointer_type(tl.int32))


@triton.jit
def _gather_rows(
    row, num_reqs, num_rows, req, counts, groups, src_tables, dst_tables, strides,
    kernel_table_ptr, kernel_table_stride, TABLE_GROUPS: tl.constexpr,
    KERNEL_GROUP: tl.constexpr, KERNEL_BLOCKS: tl.constexpr, KERNEL_MULT: tl.constexpr,
):  # fmt: skip
    """_gather_block_tables_kernel for batch row ``row`` of every group (request
    ``req``; the groups' block ``counts``, tables and row strides as loaded by the
    caller). A real row's gathered blocks of KERNEL_GROUP also go through
    AttentionGroup.map_to_kernel_block_table into the kernel block table.
    (The mapping also covers the rest of the row's width and the padded rows, which
    no kernel reads: those entries are left as they are.)"""
    if row < num_rows:
        for g in tl.static_range(TABLE_GROUPS):
            stride = tl.sum(tl.where(groups == g, strides, 0))
            dst_row = _pick_ptr(dst_tables, groups == g) + row * stride
            if row >= num_reqs:  # padded rows are zeroed
                for i in tl.range(0, stride, GATHER_BLOCK):
                    pad = i + tl.arange(0, GATHER_BLOCK)
                    tl.store(dst_row + pad, 0, mask=pad < stride)
            else:
                num_blocks = tl.sum(tl.where(groups == g, counts, 0))
                src_row = _pick_ptr(src_tables, groups == g) + req * stride
                for j in tl.range(0, num_blocks, GATHER_BLOCK):
                    offset = j + tl.arange(0, GATHER_BLOCK)
                    ids = tl.load(src_row + offset, mask=offset < num_blocks)
                    tl.store(dst_row + offset, ids, mask=offset < num_blocks)
                if g == KERNEL_GROUP:
                    out = kernel_table_ptr + row * kernel_table_stride
                    entries = num_blocks * KERNEL_BLOCKS
                    for j in tl.range(0, entries, GATHER_BLOCK):
                        entry = j + tl.arange(0, GATHER_BLOCK)
                        block = tl.load(
                            src_row + entry // KERNEL_BLOCKS, mask=entry < entries
                        )
                        kernel_block = block * KERNEL_MULT + entry % KERNEL_BLOCKS
                        tl.store(out + entry, kernel_block, mask=entry < entries)


@triton.jit
def _slot_mappings_row(
    row, g, num_reqs, idx_mapping_ptr, query_start_loc_ptr, num_computed_ptr,
    src_tables_ptr, table_strides_ptr, block_sizes_ptr, mapping_enabled_ptr,
    dcp_sharded_ptr, slot_mappings_ptr, slot_mappings_stride, cp_rank,
    CP_SIZE: tl.constexpr, CP_INTERLEAVE: tl.constexpr,
):  # fmt: skip
    """_compute_slot_mappings_kernel for request row ``row`` of group ``g``, with
    the positions as prepare_pos_seq_lens writes them (computed here)."""
    if row < num_reqs:
        req = tl.load(idx_mapping_ptr + row)
        is_real_req = req >= 0
        start = tl.load(query_start_loc_ptr + row)
        end = tl.load(query_start_loc_ptr + row + 1)
        table = _load_ptr(src_tables_ptr + g, tl.int32)
        stride = tl.load(table_strides_ptr + g)
        kv_block_size = tl.load(block_sizes_ptr + g)
        mapping_enabled = tl.load(mapping_enabled_ptr + g)
        out = slot_mappings_ptr + g * slot_mappings_stride
        for i in range(start, end, TOKEN_BLOCK):
            offset = i + tl.arange(0, TOKEN_BLOCK)
            num_computed = tl.load(num_computed_ptr + req)
            positions = tl.where(offset < end, num_computed + offset - start, 0).to(
                tl.int64
            )  # as loaded from the int64 positions
            if CP_SIZE == 1:
                local_positions = positions
                is_local = True
            else:
                sharded = tl.load(dcp_sharded_ptr + g)
                virtual_block_size = kv_block_size * CP_SIZE
                virtual_block_indices = positions // virtual_block_size
                virtual_block_offsets = positions % virtual_block_size
                is_local = virtual_block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
                rounds = virtual_block_offsets // (CP_INTERLEAVE * CP_SIZE)
                remainder = virtual_block_offsets % CP_INTERLEAVE
                local_offsets = rounds * CP_INTERLEAVE + remainder
                local_positions = virtual_block_indices * kv_block_size + (
                    local_offsets
                )
                local_positions = tl.where(sharded, local_positions, positions)
                is_local = ~sharded | is_local
            block_indices = tl.where(
                mapping_enabled, local_positions // kv_block_size, 0
            )
            block_offsets = local_positions % kv_block_size
            block_numbers = tl.load(
                table + req * stride + block_indices,
                mask=is_local & is_real_req,
                other=0,
            )
            slot_ids = block_numbers * kv_block_size + block_offsets
            if CP_SIZE != 1:
                slot_ids = tl.where(is_local, slot_ids, PAD_SLOT)
            slot_ids = tl.where(mapping_enabled & is_real_req, slot_ids, PAD_SLOT)
            tl.store(out + offset, slot_ids, mask=offset < end)


@triton.jit
def _dcp_local_seq_lens(
    out_ptr, idx_mapping_ptr, num_computed_ptr, query_start_loc_ptr, num_reqs,
    max_num_reqs, cp_rank, CP_SIZE: tl.constexpr, CP_INTERLEAVE: tl.constexpr,
    DCP_BLOCK: tl.constexpr,
):  # fmt: skip
    """_dcp_local_seq_lens_kernel over all ``max_num_reqs`` entries, with the
    sequence lengths as prepare_pos_seq_lens writes them (computed here)."""
    block = tl.arange(0, DCP_BLOCK)
    real = block < num_reqs
    req = tl.load(idx_mapping_ptr + block, mask=real, other=0)
    num_computed = tl.load(num_computed_ptr + req, mask=real, other=0)
    start = tl.load(query_start_loc_ptr + block, mask=real, other=0)
    end = tl.load(query_start_loc_ptr + block + 1, mask=real, other=0)
    seq_lens = num_computed + end - start
    rounds = seq_lens // (CP_SIZE * CP_INTERLEAVE)
    remainder = seq_lens % (CP_SIZE * CP_INTERLEAVE)
    remainder = tl.maximum(remainder - cp_rank * CP_INTERLEAVE, 0)
    remainder = tl.minimum(remainder, CP_INTERLEAVE)
    local_seq_lens = rounds * CP_INTERLEAVE + remainder
    local_seq_lens = tl.where(block < num_reqs, local_seq_lens, 0)
    tl.store(out_ptr + block, local_seq_lens, mask=block < max_num_reqs)


@triton.jit
def _step_inputs_row(
    row, num_reqs, max_num_reqs, req, num_computed, start, end, logits_start_idx,
    num_logits, prefill_len, last_sampled, drafts, in_drafts, positions_ptr,
    seq_lens_ptr, expanded_idx_ptr, expanded_pos_ptr, logits_indices_ptr,
    input_ids_ptr, EXPAND_BLOCK: tl.constexpr, NUM_NEW_SAMPLED: tl.constexpr,
    REQ_BLOCK: tl.constexpr,
):  # fmt: skip
    """expand_idx_mapping, prepare_pos_seq_lens and combine_sampled_and_draft_tokens
    for request row ``row`` (its request's values as loaded by the caller; the draft
    tokens under ``in_drafts``); row 0 also zeroes seq_lens[num_reqs:max_num_reqs].
    Returns the row's sequence length (0 past the real requests)."""
    if row == 0:
        pad = tl.arange(0, REQ_BLOCK)
        tl.store(seq_lens_ptr + pad, 0, mask=(pad >= num_reqs) & (pad < max_num_reqs))
    seq_len = tl.full((), 0, tl.int32)
    if row < num_reqs:
        query_len = end - start
        seq_len = num_computed + query_len
        tl.store(seq_lens_ptr + row, seq_len)
        for i in tl.range(0, query_len, GATHER_BLOCK):
            block = i + tl.arange(0, GATHER_BLOCK)
            pos = num_computed + block
            tl.store(positions_ptr + start + block, pos, mask=block < query_len)
        num_draft_tokens = num_logits - NUM_NEW_SAMPLED
        e = tl.arange(0, EXPAND_BLOCK)
        in_logits = e < num_logits
        tl.store(expanded_idx_ptr + logits_start_idx + e, req, mask=in_logits)
        tl.store(expanded_pos_ptr + logits_start_idx + e, e, mask=in_logits)
        logits_start = end - num_logits
        tl.store(
            logits_indices_ptr + logits_start_idx + e, logits_start + e, mask=in_logits
        )
        if seq_len > prefill_len:
            first_logit_seq_pos = seq_len - num_logits
            if NUM_NEW_SAMPLED > 0 and first_logit_seq_pos >= prefill_len:
                tl.store(input_ids_ptr + logits_start, last_sampled)
            if num_draft_tokens > 0:
                tl.store(
                    input_ids_ptr + end - num_draft_tokens + e, drafts, mask=in_drafts
                )
    return seq_len


@dataclass
class StepInputs:
    """What the runner's skipped launches (``defers_step_prep``) would write and what
    they read: ``expand_idx_mapping``'s and ``combine_sampled_and_draft_tokens``'
    outputs (allocated by the runner), the input buffers they and
    ``prepare_pos_seq_lens`` fill, the runner's ``BlockTables`` (gathered tables, slot
    mappings), the DCP local lengths (None without DCP) and the request states."""

    block_tables: object  # vllm.v1.worker.gpu.block_table.BlockTables
    dcp_local_seq_lens: torch.Tensor | None  # the whole [max_num_reqs] buffer
    cu_num_logits: torch.Tensor
    expanded_idx_mapping: torch.Tensor
    expanded_local_pos: torch.Tensor
    logits_indices: torch.Tensor
    input_ids: torch.Tensor
    positions: torch.Tensor  # int64
    seq_lens: torch.Tensor  # the whole [max_num_reqs] buffer
    last_sampled_tokens: torch.Tensor
    # The block tables' per-group block counts [groups, max_num_reqs] and the
    # requests' prefill lengths: device copies (``ModelState.stage_step_inputs``).
    num_blocks: torch.Tensor
    prefill_len: torch.Tensor
    draft_tokens: torch.Tensor  # [max_num_reqs, num_speculative_steps]
    num_new_sampled: int
    max_expand_len: int  # the decode query length
    # prepare_inputs' is_padding (None: the runner does not fill it), through
    # num_tokens_padded.
    is_padding: torch.Tensor | None = None
    num_tokens_padded: int = 0


class KdaStepPrep:
    """One launch of the KDA side of a steady spec-decode step's prep (see the module
    docstring). Built once the model state knows its KDA builders and Mamba context;
    ``builders``: [(Mamba group index, KimiK3KDAMetadataBuilder)]; ``kernel_groups``:
    the attention groups whose kernel block tables the launch also maps (their
    ``kernel_block_table_prestaged`` is the caller's to set)."""

    def __init__(self, ctx, builders, mamba_block_size, device, kernel_groups=()):
        self.ctx = ctx
        self.builders = [b for _, b in builders]
        self.mamba_block_size = mamba_block_size
        staged = [b.spec_state_indices_tensor for b in self.builders]
        assert len({t.stride(0) for t in staged}) == 1
        self.staged_slots = staged[0].shape[1]
        self.num_slots = ctx.aligned_state_indices.shape[2]
        assert self.staged_slots <= self.num_slots
        self.capacity = min(t.shape[0] for t in staged)

        def ptrs(tensors):
            return torch.tensor(
                [t.data_ptr() for t in tensors], dtype=torch.int64, device=device
            )

        self.staged_state_ptrs = ptrs(staged)
        self.staged_qsl_ptrs = ptrs([b.spec_query_start_loc for b in self.builders])
        self.staged_accepted_ptrs = ptrs([b.num_accepted_tokens for b in self.builders])
        self.staged_group = torch.tensor(
            [g for g, _ in builders], dtype=torch.int32, device=device
        )
        self.num_accepted_out = torch.empty(
            ctx.aligned_state_indices.shape[1], dtype=torch.int32, device=device
        )
        # At most one kernel block table (any more: none, the groups map their own).
        self.kernel_groups = list(kernel_groups) if len(kernel_groups) == 1 else []
        self.kernel_meta = dict(KERNEL_GROUP=-1, KERNEL_BLOCKS=1, KERNEL_MULT=1)
        self.kernel_table = self.num_accepted_out  # (never written without a group)
        for g in self.kernel_groups:
            per = g.kv_cache_spec.block_size // g.kernel_block_size
            self.kernel_meta = dict(
                KERNEL_GROUP=g.kv_cache_group_id,
                KERNEL_BLOCKS=per,
                KERNEL_MULT=g.kernel_block_stride or per,
            )
            self.kernel_table = g.kernel_block_table[0]

    def run(
        self,
        input_batch,
        num_rows,
        num_computed_tokens,
        state_idx,
        num_accepted,
        src_col,
        src_off,
        inputs,
        compile_only=False,
    ):
        """Fills ``inputs`` (StepInputs) and returns (num_accepted [num_rows], the
        aligned state indices [groups, num_rows, slots]). ``compile_only``: compile the
        kernel for these arguments' specialization without launching it (nothing is
        written; returns None)."""
        ctx = self.ctx
        aligned = ctx.aligned_state_indices
        bt = inputs.block_tables
        slots = bt.slot_mappings
        max_num_reqs = bt.max_num_reqs
        assert inputs.seq_lens.shape[0] == max_num_reqs
        pad_programs = triton.cdiv(slots.shape[1], int(PAD_BLOCK))
        grid = num_rows + 1 + bt.num_kv_cache_groups * (num_rows + pad_programs)
        dcp = inputs.dcp_local_seq_lens
        launch = _kda_step_prep_kernel[(grid,)]
        if compile_only:
            launch = partial(_kda_step_prep_kernel.warmup, grid=(grid,))
        launch(
            input_batch.idx_mapping,
            state_idx,
            num_computed_tokens,
            input_batch.query_start_loc,
            num_accepted,
            src_col,
            src_off,
            self.num_accepted_out,
            inputs.seq_lens,
            ctx.aligned_index_block_table_ptrs,
            ctx.block_table_stride_req,
            aligned,
            aligned.stride(0),
            aligned.stride(1),
            self.staged_state_ptrs,
            self.builders[0].spec_state_indices_tensor.stride(0),
            self.staged_qsl_ptrs,
            self.staged_accepted_ptrs,
            self.staged_group,
            input_batch.num_reqs,
            num_rows,
            bt.block_table_ptrs,
            bt.input_block_table_ptrs,
            bt.block_table_strides,
            inputs.num_blocks,
            inputs.num_blocks.stride(0),
            bt.block_sizes_tensor,
            bt.slot_mapping_enabled,
            bt.dcp_sharded,
            inputs.positions,
            slots,
            slots.stride(0),
            slots.shape[1],
            dcp if dcp is not None else slots,  # (never read without DCP)
            max_num_reqs,
            bt.cp_rank,
            pad_programs,
            inputs.is_padding if inputs.is_padding is not None else slots,
            inputs.num_tokens_padded,
            inputs.cu_num_logits,
            inputs.expanded_idx_mapping,
            inputs.expanded_local_pos,
            inputs.logits_indices,
            inputs.input_ids,
            inputs.last_sampled_tokens,
            inputs.prefill_len,
            inputs.draft_tokens,
            inputs.draft_tokens.stride(0),
            self.kernel_table,
            self.kernel_table.stride(0),
            EXPAND_BLOCK=triton.next_power_of_2(inputs.max_expand_len),
            NUM_NEW_SAMPLED=inputs.num_new_sampled,
            REQ_BLOCK=triton.next_power_of_2(max_num_reqs),
            TABLE_GROUPS=bt.num_kv_cache_groups,
            IS_PADDING=inputs.is_padding is not None,
            **self.kernel_meta,
            CP_SIZE=bt.cp_size,
            CP_INTERLEAVE=bt.cp_interleave,
            MAMBA_BLOCK_SIZE=self.mamba_block_size,
            CACHE_BLOCK_SIZE=ctx.block_size,
            NUM_GROUPS=ctx.num_groups,
            NUM_SLOTS=self.num_slots,
            STAGED_SLOTS=self.staged_slots,
            BLOCK_SLOTS=triton.next_power_of_2(self.num_slots),
            NUM_STAGED=len(self.builders),
            NULL_STATE_ID=NULL_BLOCK_ID,
            launch_pdl=current_platform.is_arch_support_pdl(),
            num_warps=4,
        )
        if compile_only:
            return None
        return self.num_accepted_out[:num_rows], aligned[:, :num_rows]
