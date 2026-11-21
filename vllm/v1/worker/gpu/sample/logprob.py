# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor

# Upper bound on the topk kernel's per-iteration gather width.
_MAX_TOPK_BLOCK = 1024


@triton.jit
def _topk_log_softmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk_ids_stride,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=0.0)
        # NOTE(woosuk): Make sure that logits and all following operations use FP32.
        logits = logits.to(tl.float32)
        e = tl.exp(logits - max_val)
        e = tl.where(block < vocab_size, e, 0.0)
        se += tl.sum(e)
    lse = tl.log(se)

    for j in range(0, topk, TOPK_BLOCK_SIZE):
        k_offset = j + tl.arange(0, TOPK_BLOCK_SIZE)
        k_mask = k_offset < topk
        topk_ids = tl.load(
            topk_ids_ptr + req_idx * topk_ids_stride + k_offset, mask=k_mask, other=0
        )
        logits = tl.load(row_ptr + topk_ids, mask=k_mask)
        logits = logits.to(tl.float32)
        o = logits - max_val - lse
        tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


@triton.jit
def _ranks_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id)

    n = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        n += tl.sum((logits >= x).to(tl.int32))
    tl.store(output_ptr + req_idx, n)


def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    # NOTE(woosuk): To save GPU memory, we do not materialize the full
    # [batch_size, vocab_size] logprobs tensor. The kernel computes
    # max + logsumexp per row and only emits logprobs at `token_ids`.
    return _topk_log_softmax(logits, token_ids)


def _topk_log_softmax(
    logits: torch.Tensor, token_ids: torch.Tensor, warmup: bool = False
) -> torch.Tensor:
    """Launch `_topk_log_softmax_kernel`, or with `warmup` only compile it."""
    batch_size, vocab_size = logits.shape
    token_ids = token_ids.to(torch.int64)
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    args = (
        logprobs,
        logits,
        logits.stride(0),
        token_ids,
        token_ids.stride(0),
        num_logprobs,
        vocab_size,
    )
    kwargs = dict(
        BLOCK_SIZE=1024,
        TOPK_BLOCK_SIZE=_topk_block_size(num_logprobs),
    )
    if warmup:
        _topk_log_softmax_kernel.warmup(*args, grid=(batch_size,), **kwargs)
    else:
        _topk_log_softmax_kernel[(batch_size,)](*args, **kwargs)
    return logprobs


def _topk_block_size(num_logprobs: int) -> int:
    # Cap the kernel's per-iteration width so very large num_logprobs requests
    # stream the gather in bounded-size chunks, avoiding excessive mem use.
    return min(triton.next_power_of_2(num_logprobs), _MAX_TOPK_BLOCK)


def token_logprobs_warmup_counts(max_count: int, odd_only: bool = False) -> list[int]:
    """One token-id count (columns of `compute_token_logprobs`'s `token_ids`)
    per `_topk_log_softmax_kernel` compile variant among counts 1..max_count.

    A count's variant is its TOPK_BLOCK_SIZE plus Triton's specialization of
    the `topk` and `topk_ids_stride` ints, which both equal the count (the
    token-id rows are contiguous): equal to 1, divisible by 16, or neither.
    `odd_only` keeps the variants an odd count reaches.
    """
    first: dict[tuple[int, bool, bool], int] = {}
    for n in range(1, max_count + 1, 2 if odd_only else 1):
        first.setdefault((_topk_block_size(n), n == 1, n % 16 == 0), n)
    return list(first.values())


def warmup_token_logprobs(
    logits: torch.Tensor, max_count: int, max_unaligned_count: int = 0
) -> None:
    """Compile, without launching, `compute_token_logprobs` for logits laid out
    like `logits` (dtype, row stride, vocab, alignment) and every token-id
    count 1..max_count. Counts up to `max_unaligned_count` are also compiled
    with the token-id rows at an odd int64 offset: scoring a chunk of a
    request's [rows, count] prompt_logprob_token_ids slices it at any row, which
    leaves the pointer 8-byte aligned when the count is odd.
    """
    for unaligned, counts in (
        (False, token_logprobs_warmup_counts(max_count)),
        (True, token_logprobs_warmup_counts(max_unaligned_count, odd_only=True)),
    ):
        for n in counts:
            ids = torch.zeros(
                1 + int(unaligned) + n, dtype=torch.int64, device=logits.device
            )
            ids = ids[int(unaligned) :][:n].view(1, n)
            _topk_log_softmax(logits, ids, warmup=True)


def compute_topk_scores(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | torch.Tensor | None = None,
    logprob_token_ids_state: "LogprobTokenIdsState | None" = None,
    expanded_idx_mapping: torch.Tensor | None = None,
    max_per_req_token_ids: int = 0,
    logits_mode: bool = False,
) -> LogprobsTensors:
    assert num_logprobs >= 0
    batch_size, vocab_size = logits.shape

    if max_per_req_token_ids == 0:
        # Fast path: no request asked for custom logprob_token_ids.
        logprob_token_ids = sampled_token_ids.unsqueeze(-1)
        if num_logprobs > 0:
            topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
            logprob_token_ids = torch.cat((logprob_token_ids, topk_indices), dim=1)
        if logits_mode:
            scores = logits.gather(-1, logprob_token_ids).to(torch.float32)
        else:
            scores = compute_token_logprobs(logits, logprob_token_ids)
    else:
        # Some requests specified logprob_token_ids. Build the [batch_size,
        # 1 + max_cols] token_ids matrix and validity mask on the GPU via a
        # single triton kernel, overriding the topk columns with per-request
        # tokens where applicable.
        assert logprob_token_ids_state is not None
        assert expanded_idx_mapping is not None

        if num_logprobs > 0:
            topk_token_ids = torch.topk(logits, num_logprobs, dim=-1).indices
            topk_token_ids = topk_token_ids.to(torch.int32)
        else:
            # This tensor just used as an int32 pointer, data not accessed.
            topk_token_ids = logprob_token_ids_state.token_ids.gpu

        num_cols = max(num_logprobs, max_per_req_token_ids)
        logprob_token_ids, valid_mask = _fill_logprob_token_ids(
            sampled_token_ids,
            topk_token_ids,
            num_logprobs,
            num_cols,
            expanded_idx_mapping,
            logprob_token_ids_state,
        )
        if logits_mode:
            scores = logits.gather(-1, logprob_token_ids).to(torch.float32)
        else:
            scores = compute_token_logprobs(logits, logprob_token_ids)
        scores = scores.masked_fill(~valid_mask, float("-inf"))

    token_ranks = torch.empty(batch_size, dtype=torch.int64, device=logits.device)
    _ranks_kernel[(batch_size,)](
        token_ranks,
        logits,
        logits.stride(0),
        sampled_token_ids,
        vocab_size,
        BLOCK_SIZE=8192,  # type: ignore
    )
    is_tensor = isinstance(cu_num_logits, torch.Tensor)
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=scores,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=None if is_tensor else cu_num_logits,
        cu_num_generated_tokens_tensor=cu_num_logits if is_tensor else None,
    )


# The ints are not specialized, so the compile variants are PADDED_COLS only
# (`warmup_logprob_token_ids` compiles them at startup).
@triton.jit(
    do_not_specialize=[
        "out_token_ids_stride",
        "out_valid_mask_stride",
        "topk_indices_stride",
        "num_topk",
    ]
)
def _fill_logprob_token_ids_kernel(
    # [batch_size, 1 + num_cols]
    out_token_ids_ptr,
    out_token_ids_stride,
    # [batch_size, 1 + num_cols]
    out_valid_mask_ptr,
    out_valid_mask_stride,
    sampled_token_ids_ptr,  # [batch_size]
    topk_indices_ptr,  # [batch_size, num_topk] (unused when num_topk == 0)
    topk_indices_stride,
    expanded_idx_mapping_ptr,  # [batch_size] -> req_state_idx
    num_per_req_token_ids_ptr,  # [max_num_reqs]
    per_req_token_ids_ptr,  # [max_num_reqs, MAX_LOGPROB_TOKEN_IDS]
    per_req_token_ids_stride,
    num_topk,
    PADDED_COLS: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    # Column 0: always the sampled token, always valid.
    sampled = tl.load(sampled_token_ids_ptr + batch_idx)
    tl.store(out_token_ids_ptr + batch_idx * out_token_ids_stride, sampled)
    tl.store(out_valid_mask_ptr + batch_idx * out_valid_mask_stride, 1)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + batch_idx)
    num_custom = tl.load(num_per_req_token_ids_ptr + req_state_idx)

    col = tl.arange(0, PADDED_COLS)
    tid_base = out_token_ids_ptr + batch_idx * out_token_ids_stride + 1
    mask_base = out_valid_mask_ptr + batch_idx * out_valid_mask_stride + 1

    if num_custom > 0:
        # Override topk with per-request custom tokens.
        src = per_req_token_ids_ptr + req_state_idx * per_req_token_ids_stride
        valid = col < num_custom
    else:
        # Fill with topk indices (no-op when num_topk == 0).
        src = topk_indices_ptr + batch_idx * topk_indices_stride
        valid = col < num_topk

    tokens = tl.load(src + col, mask=valid, other=0).to(tl.int64)
    tl.store(tid_base + col, tokens, mask=valid)
    tl.store(mask_base + col, tl.full([PADDED_COLS], 1, tl.int1), mask=valid)


def _fill_logprob_token_ids(
    sampled_token_ids: torch.Tensor,
    topk_token_ids: torch.Tensor,
    num_logprobs: int,
    num_cols: int,
    expanded_idx_mapping: torch.Tensor,
    logprob_token_ids_state: "LogprobTokenIdsState",
    warmup: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch `_fill_logprob_token_ids_kernel`, or with `warmup` only compile it."""
    batch_size = sampled_token_ids.shape[0]
    logprob_token_ids = sampled_token_ids.new_zeros((batch_size, 1 + num_cols))
    valid_mask = torch.zeros_like(logprob_token_ids, dtype=torch.bool)
    args = (
        logprob_token_ids,
        logprob_token_ids.stride(0),
        valid_mask,
        valid_mask.stride(0),
        sampled_token_ids,
        topk_token_ids,
        topk_token_ids.stride(0),
        expanded_idx_mapping,
        logprob_token_ids_state.num_token_ids.gpu,
        logprob_token_ids_state.token_ids.gpu,
        logprob_token_ids_state.token_ids.gpu.stride(0),
        num_logprobs,
    )
    padded_cols = triton.next_power_of_2(num_cols)
    if warmup:
        _fill_logprob_token_ids_kernel.warmup(
            *args, grid=(batch_size,), PADDED_COLS=padded_cols
        )
    else:
        _fill_logprob_token_ids_kernel[(batch_size,)](*args, PADDED_COLS=padded_cols)
    return logprob_token_ids, valid_mask


def warmup_logprob_token_ids(
    logprob_token_ids_state: "LogprobTokenIdsState",
    sampled_token_ids: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    max_num_cols: int,
) -> None:
    """Compile, without launching, the fill of requests' `logprob_token_ids` for
    every column count 1..max_num_cols (one variant per PADDED_COLS), with
    sampled ids and request mapping of the dtypes the sampler passes."""
    topk_token_ids = torch.zeros(
        1, max_num_cols, dtype=torch.int32, device=sampled_token_ids.device
    )
    for padded_cols in sorted(
        {triton.next_power_of_2(n) for n in range(1, max_num_cols + 1)}
    ):
        num_cols = min(padded_cols, max_num_cols)
        _fill_logprob_token_ids(
            sampled_token_ids[:1],
            topk_token_ids[:, :num_cols],
            num_cols,
            num_cols,
            expanded_idx_mapping[:1],
            logprob_token_ids_state,
            warmup=True,
        )


class LogprobTokenIdsState:
    """Per-request override of which token ids' logprobs to return.

    See `SamplingParams.logprob_token_ids`.
    """

    def __init__(self, max_num_reqs: int, device: torch.device):
        self.max_num_reqs = max_num_reqs
        self.num_token_ids = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.token_ids = StagedWriteTensor(
            (max_num_reqs, MAX_LOGPROB_TOKEN_IDS),
            dtype=torch.int32,
            device=device,
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        token_ids = sampling_params.logprob_token_ids
        if not token_ids:
            self.num_token_ids.np[req_idx] = 0
            return
        n = len(token_ids)
        if n > MAX_LOGPROB_TOKEN_IDS:
            raise ValueError(
                f"Too many logprob_token_ids: {n}. The max is {MAX_LOGPROB_TOKEN_IDS}."
            )
        self.num_token_ids.np[req_idx] = n
        self.token_ids.stage_write(req_idx, 0, token_ids)

    def apply_staged_writes(self) -> None:
        self.num_token_ids.copy_to_uva()
        self.token_ids.apply_write()

    def max_num_token_ids(self, idx_mapping_np: np.ndarray) -> int:
        return int(self.num_token_ids.np[idx_mapping_np].max(initial=0))
