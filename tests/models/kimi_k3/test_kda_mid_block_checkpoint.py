# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashKDA prefill checkpoint for a query that resumes mid Mamba block.

A final chunk that resumes from a sub-block prefix hit at ``s`` exports the
prompt checkpoint ``c`` into the block-table column of the state it resumes
from, ``K = (s - 1) // B``. The worker first copies that column into the
running column ``R`` (align pre-copy), the forward reads and writes only ``R``,
and the exporter then overwrites ``K``.

This replays that order on the real kernels (conv1d prefill, FlashKDA with an
in-kernel checkpoint, the checkpoint exporter) with the columns
``compute_mamba_prefill_checkpoints`` plans, and requires column ``K`` to hold
exactly, bit for bit, the recurrent and conv state a from-scratch prefill of
``[0, c)`` leaves.
"""

import pytest
import torch

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

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="FlashKDA requires a CUDA device."
)

# Serving geometry: 2048-token Mamba blocks, 128-token hash blocks, FlashKDA
# checkpoint alignment 16, fp32 recurrent state.
MAMBA_BLOCK = 2048
HASH = 128
ALIGNMENT = 16
H, D = 2, 128
CONV_DIM = 3 * H * D
CONV_WIDTH = 4
LOWER_BOUND = -5.0
STATE_DTYPE = torch.float32
NUM_SLOTS = 8
# Slot 0 is NULL_BLOCK_ID: the conv kernel skips a row whose cache index is the
# null block, so no real state may live there.
SCRATCH_SLOT, K_SLOT, R_SLOT = 1, 3, 5


class _Inputs:
    def __init__(self, num_tokens: int):
        torch.manual_seed(0)
        self.raw_qkv = torch.randn(
            num_tokens, CONV_DIM, dtype=torch.bfloat16, device=DEVICE
        )
        self.conv_weight = 0.5 * torch.randn(
            CONV_DIM, CONV_WIDTH, dtype=torch.bfloat16, device=DEVICE
        )
        self.raw_g = torch.randn(
            1, num_tokens, H, D, dtype=torch.bfloat16, device=DEVICE
        )
        self.raw_beta = torch.randn(
            1, num_tokens, H, dtype=torch.bfloat16, device=DEVICE
        )
        self.A_log = 0.5 * torch.randn(H, dtype=torch.float32, device=DEVICE)
        self.dt_bias = 0.1 * torch.randn(H, D, dtype=torch.float32, device=DEVICE)


def _conv_cache() -> torch.Tensor:
    # The serving "DS" layout: (dim, width - 1) per slot.
    return torch.zeros(
        NUM_SLOTS, CONV_DIM, CONV_WIDTH - 1, dtype=torch.bfloat16, device=DEVICE
    )


def _forward(
    inputs: _Inputs,
    start: int,
    end: int,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    slot: int,
    has_initial_state: bool,
    checkpoint: MambaPrefillCheckpointMetadata | None = None,
) -> None:
    """One prefill row over tokens ``[start, end)`` on cache slot ``slot``,
    in the KDA layer's order: conv -> FlashKDA -> checkpoint export -> final
    state."""
    n = end - start
    cu_seqlens = torch.tensor([0, n], dtype=torch.int32, device=DEVICE)
    raw = inputs.raw_qkv[start:end]
    qkv = causal_conv1d_fn(
        raw.transpose(0, 1),
        inputs.conv_weight,
        None,
        conv_state,
        cu_seqlens,
        cache_indices=torch.tensor([slot], dtype=torch.int32, device=DEVICE),
        has_initial_state=torch.tensor(
            [has_initial_state], dtype=torch.bool, device=DEVICE
        ),
    ).transpose(0, 1)
    q, k, v = (x.reshape(1, n, H, D) for x in qkv.split(H * D, dim=-1))
    initial_state = (
        recurrent_state[slot : slot + 1].clone()
        if has_initial_state
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


def _from_scratch(inputs: _Inputs, end: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Recurrent and conv state a fresh prefill of ``[0, end)`` leaves."""
    conv_state = _conv_cache()
    recurrent_state = torch.zeros(
        NUM_SLOTS, H, D, D, dtype=STATE_DTYPE, device=DEVICE
    )
    _forward(
        inputs,
        0,
        end,
        conv_state,
        recurrent_state,
        SCRATCH_SLOT,
        has_initial_state=False,
    )
    return recurrent_state[SCRATCH_SLOT].clone(), conv_state[SCRATCH_SLOT].clone()


@pytest.mark.parametrize(
    "prompt_len",
    [
        # Resume at s = 3072 (column K = 1). Ends in the next block (R = 2):
        # the case that used to take three steps.
        2 * MAMBA_BLOCK + 300,
        # Ends two blocks later (R = 3).
        3 * MAMBA_BLOCK + 300,
    ],
)
@torch.inference_mode()
def test_mid_block_resume_exports_from_scratch_checkpoint(prompt_len: int):
    if not is_flashkda_supported(D, torch.bfloat16, STATE_DTYPE, LOWER_BOUND):
        pytest.skip("FlashKDA prefill is not supported on this platform")
    import vllm._flashkda_C  # noqa: F401

    start = MAMBA_BLOCK + MAMBA_BLOCK // 2
    inputs = _Inputs(prompt_len)
    checkpoint_position = (prompt_len - 1) // HASH * HASH

    # The planner the KDA metadata builder runs.
    offsets, cols = compute_mamba_prefill_checkpoints(
        [prompt_len],
        [prompt_len - start],
        hash_block_size=HASH,
        mamba_block_size=MAMBA_BLOCK,
        checkpoint_alignment=ALIGNMENT,
        drop_eagle_block=False,
        cache_hit_alignment_tokens=HASH,
    )
    initial_col = (start - 1) // MAMBA_BLOCK
    running_col = -(-prompt_len // MAMBA_BLOCK) - 1
    assert offsets == [checkpoint_position - start]
    assert cols == [initial_col]

    # The request's block-table row: the resumed state's column K and the
    # running column R; the scheduler nulls the columns below and between.
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

    # Column K holds the hit's state@s (the CoW copy of a partial hit).
    conv_state = _conv_cache()
    recurrent_state = torch.zeros(
        NUM_SLOTS, H, D, D, dtype=STATE_DTYPE, device=DEVICE
    )
    _forward(inputs, 0, start, conv_state, recurrent_state, K_SLOT, False)
    untouched = [i for i in range(NUM_SLOTS) if i not in (K_SLOT, R_SLOT)]
    for i in untouched:
        recurrent_state[i].fill_(float(i))
        conv_state[i].fill_(float(i))
    others = (recurrent_state[untouched].clone(), conv_state[untouched].clone())

    # Align pre-copy K -> R, then the forward on R with the checkpoint export.
    recurrent_state[R_SLOT] = recurrent_state[K_SLOT]
    conv_state[R_SLOT] = conv_state[K_SLOT]
    _forward(
        inputs, start, prompt_len, conv_state, recurrent_state, R_SLOT, True, checkpoint
    )

    expected_recurrent, expected_conv = _from_scratch(inputs, checkpoint_position)
    max_diff = (recurrent_state[K_SLOT] - expected_recurrent).abs().max().item()
    assert torch.equal(recurrent_state[K_SLOT], expected_recurrent), (
        f"checkpoint recurrent state differs from a prefill to "
        f"{checkpoint_position} (max abs diff {max_diff})"
    )
    assert torch.equal(conv_state[K_SLOT], expected_conv)

    # The running slot finishes the prompt; no other slot is written.
    final_recurrent, final_conv = _from_scratch(inputs, prompt_len)
    torch.testing.assert_close(recurrent_state[R_SLOT], final_recurrent)
    assert torch.equal(conv_state[R_SLOT], final_conv)
    assert torch.equal(recurrent_state[untouched], others[0])
    assert torch.equal(conv_state[untouched], others[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
