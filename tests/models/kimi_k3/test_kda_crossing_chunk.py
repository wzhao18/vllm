# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashKDA crossing chunk: an intermediate prefill chunk resuming mid block p
runs past the block boundary B into block p + 1 and exports the state at B into
column p from inside the step (``get_mamba_crossing_checkpoint_position``),
instead of stopping at B.

Replays the worker order on the real kernels (align pre-copy p -> running
column, conv1d prefill, FlashKDA with the in-kernel checkpoint, the checkpoint
exporter) with the columns ``compute_mamba_prefill_checkpoints`` plans, and
requires, bit for bit, against the block-ended split ``[s, B)`` then ``[B, e)``:
column p's recurrent and conv state (= the state at B), the running column's
state at e, and every output row. A prefix hit restoring from column p then
continues identically too.
"""

import pytest
import torch

from tests.models.kimi_k3.test_kda_mid_block_checkpoint import (
    ALIGNMENT,
    DEVICE,
    HASH,
    LOWER_BOUND,
    MAMBA_BLOCK,
    NUM_SLOTS,
    STATE_DTYPE,
    D,
    H,
    _conv_cache,
    _Inputs,
)
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointMetadata,
    _gather_checkpoint_state_indices,
    compute_mamba_prefill_checkpoints,
)
from vllm.model_executor.layers.mamba.kda_checkpoint import (
    FlashKDAPrefillCheckpointExporter,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.models.kimi_k3.nvidia.kda import _flashkda_prefill, is_flashkda_supported
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="FlashKDA requires a CUDA device."
)

K_SLOT, R_SLOT, A_SLOT = 3, 5, 6  # column p, running column p + 1, aligned run


def _forward(inputs, start, end, conv_state, recurrent_state, slot, checkpoint=None):
    """One prefill row ``[start, end)`` on ``slot`` in the KDA layer's order:
    conv -> FlashKDA -> checkpoint export -> final state. Returns the output."""
    n = end - start
    cu_seqlens = torch.tensor([0, n], dtype=torch.int32, device=DEVICE)
    raw = inputs.raw_qkv[start:end]
    has_initial = start > 0
    qkv = causal_conv1d_fn(
        raw.transpose(0, 1),
        inputs.conv_weight,
        None,
        conv_state,
        cu_seqlens,
        cache_indices=torch.tensor([slot], dtype=torch.int32, device=DEVICE),
        has_initial_state=torch.tensor([has_initial], dtype=torch.bool, device=DEVICE),
    ).transpose(0, 1)
    q, k, v = (x.reshape(1, n, H, D) for x in qkv.split(H * D, dim=-1))
    initial_state = (
        recurrent_state[slot : slot + 1].clone()
        if has_initial
        else torch.zeros_like(recurrent_state[:1])
    )
    out = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    checkpoint_state = torch.empty_like(initial_state)
    workspace = torch.empty(
        torch.ops._flashkda_C.get_workspace_size(n, H, 1),
        dtype=torch.uint8,
        device=DEVICE,
    )
    _flashkda_prefill(
        q=q,
        k=k,
        v=v,
        g=inputs.raw_g[:, start:end],
        beta=inputs.raw_beta[:, start:end],
        A_log=inputs.A_log,
        dt_bias=inputs.dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        out=out,
        final_state=final_state,
        workspace=workspace,
        checkpoint_state=checkpoint_state if checkpoint is not None else None,
        checkpoint_offsets=(
            checkpoint.checkpoint_offsets if checkpoint is not None else None
        ),
    )
    if checkpoint is not None:
        FlashKDAPrefillCheckpointExporter().export(
            checkpoint,
            raw_qkv=raw,
            conv_state=conv_state,
            recurrent_checkpoint=checkpoint_state,
            recurrent_state=recurrent_state,
            cu_seqlens=cu_seqlens,
        )
    recurrent_state[slot] = final_state[0]
    return out


def _caches():
    return _conv_cache(), torch.zeros(
        NUM_SLOTS, H, D, D, dtype=STATE_DTYPE, device=DEVICE
    )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        # Mid block 1 -> past B = 2 * MAMBA_BLOCK, ends mid block 2.
        (MAMBA_BLOCK + MAMBA_BLOCK // 2, 2 * MAMBA_BLOCK + 700),
        # Just past the boundary.
        (MAMBA_BLOCK + 128, 2 * MAMBA_BLOCK + 16),
        # Starts one hash block before B, ends exactly on the next boundary.
        (2 * MAMBA_BLOCK - HASH, 3 * MAMBA_BLOCK),
        # Long run into the next block.
        (MAMBA_BLOCK + 896, 2 * MAMBA_BLOCK + 1900),
    ],
)
@torch.inference_mode()
def test_crossing_chunk_matches_the_block_ended_split(start: int, end: int):
    if not is_flashkda_supported(D, torch.bfloat16, STATE_DTYPE, LOWER_BOUND):
        pytest.skip("FlashKDA prefill is not supported on this platform")
    import vllm._flashkda_C  # noqa: F401

    prompt_len = end + 3 * MAMBA_BLOCK  # an intermediate chunk
    inputs = _Inputs(prompt_len + 512)
    boundary = (start // MAMBA_BLOCK + 1) * MAMBA_BLOCK
    initial_col = (start - 1) // MAMBA_BLOCK
    running_col = -(-end // MAMBA_BLOCK) - 1
    assert running_col == initial_col + 1

    # The planner the KDA metadata builder runs, with the V2 model state's
    # prefill end: the crossed boundary into the column the chunk resumed from.
    offsets, cols = compute_mamba_prefill_checkpoints(
        [end],
        [end - start],
        hash_block_size=HASH,
        mamba_block_size=MAMBA_BLOCK,
        checkpoint_alignment=ALIGNMENT,
        drop_eagle_block=False,
        cache_hit_alignment_tokens=HASH,
        first_chunk=[False],
        prefill_ends=[prompt_len],
    )
    assert offsets == [boundary - start] and cols == [initial_col]
    block_table = torch.full(
        (1, running_col + 1), NULL_BLOCK_ID, dtype=torch.int32, device=DEVICE
    )
    block_table[0, initial_col] = K_SLOT
    block_table[0, running_col] = R_SLOT
    checkpoint = MambaPrefillCheckpointMetadata(
        torch.tensor(offsets, dtype=torch.int32, device=DEVICE),
        _gather_checkpoint_state_indices(
            block_table,
            torch.zeros(1, dtype=torch.int64, device=DEVICE),
            torch.tensor(cols, dtype=torch.int64, device=DEVICE),
        ),
    )
    assert checkpoint.state_indices.tolist() == [K_SLOT]

    # Crossing: state@start in column p, align pre-copy p -> running column,
    # one pass with the boundary export into column p.
    conv_x, rec_x = _caches()
    _forward(inputs, 0, start, conv_x, rec_x, K_SLOT)
    for i in range(NUM_SLOTS):
        if i not in (K_SLOT,):
            rec_x[i].fill_(float(i))
            conv_x[i].fill_(float(i))
    untouched = [i for i in range(NUM_SLOTS) if i not in (K_SLOT, R_SLOT)]
    others = (rec_x[untouched].clone(), conv_x[untouched].clone())
    rec_x[R_SLOT] = rec_x[K_SLOT]
    conv_x[R_SLOT] = conv_x[K_SLOT]
    out_x = _forward(inputs, start, end, conv_x, rec_x, R_SLOT, checkpoint)

    # Block-ended split: [start, B) on one slot (state@B), then [B, end).
    conv_a, rec_a = _caches()
    _forward(inputs, 0, start, conv_a, rec_a, A_SLOT)
    out_a1 = _forward(inputs, start, boundary, conv_a, rec_a, A_SLOT)
    state_b = (rec_a[A_SLOT].clone(), conv_a[A_SLOT].clone())
    out_a2 = _forward(inputs, boundary, end, conv_a, rec_a, A_SLOT)

    # Column p holds exactly the block-ended chunk's state at B.
    diff = (rec_x[K_SLOT] - state_b[0]).abs().max().item()
    assert torch.equal(rec_x[K_SLOT], state_b[0]), f"state@B max abs diff {diff}"
    assert torch.equal(conv_x[K_SLOT], state_b[1])
    # Running column and outputs: identical to the split.
    diff = (rec_x[R_SLOT] - rec_a[A_SLOT]).abs().max().item()
    assert torch.equal(rec_x[R_SLOT], rec_a[A_SLOT]), f"state@e max abs diff {diff}"
    assert torch.equal(conv_x[R_SLOT], conv_a[A_SLOT])
    assert torch.equal(out_x, torch.cat([out_a1, out_a2], dim=1))
    # No other slot is written.
    assert torch.equal(rec_x[untouched], others[0])
    assert torch.equal(conv_x[untouched], others[1])

    # UT3 (GPU-local prefix hit on block p): a request restoring from column p
    # continues bit for bit like one restoring from the split's state@B.
    conv_h, rec_h = _caches()
    rec_h[K_SLOT], conv_h[K_SLOT] = rec_x[K_SLOT], conv_x[K_SLOT]
    conv_s, rec_s = _caches()
    rec_s[K_SLOT], conv_s[K_SLOT] = state_b
    nxt = min(boundary + MAMBA_BLOCK, prompt_len)
    out_h = _forward(inputs, boundary, nxt, conv_h, rec_h, K_SLOT)
    out_s = _forward(inputs, boundary, nxt, conv_s, rec_s, K_SLOT)
    assert torch.equal(out_h, out_s)
    assert torch.equal(rec_h[K_SLOT], rec_s[K_SLOT])
