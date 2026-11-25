# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The deferred-verify KDA record protocol outside the specialized decode's record
kernels (``ops/kda_records.py``, ``vllm.v1.worker.mamba_utils`` record format):

* the stock Triton verify in the record format (``materialize`` + ``records=True``)
  gives the stock verify's outputs bit for bit over multi-step sequences with partial
  drafts and rejections, and every accepted state it leaves (folded on read) is the
  stock checkpoint bit for bit;
* the align copies (``precopy_mamba_align_fused_kernel`` into the old slot 1 = the
  record page, ``postprocess_mamba_fused_kernel`` in place into slot 0) fold a tagged
  slot to the stock checkpoint bit for bit and copy an untagged (stock) slot.

One CUDA GPU, Triton only."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from vllm.models.kimi_k3.nvidia.ops.kda_records import materialize  # noqa: E402
from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (  # noqa: E402
    fused_recurrent_kda,
)
from vllm.v1.worker import mamba_utils as mu  # noqa: E402

DEVICE = torch.device("cuda", 0)
H, Q = 2, 8
LOWER_BOUND = -5.0


def bits(tensor):
    return tensor.view({2: torch.int16, 4: torch.int32}[tensor.element_size()])


def assert_bitwise(name, got, want):
    diff = bits(got) != bits(want)
    assert not diff.any(), f"{name}: {int(diff.sum())} of {diff.numel()} differ"


def inputs(lens, generator):
    rows = sum(lens)

    def randn(*shape, scale=1.0, dtype=torch.bfloat16):
        x = torch.randn(*shape, generator=generator, device=DEVICE) * scale
        return x.to(dtype)

    q, k, v = (randn(1, rows, H, 128) for _ in range(3))
    g = randn(1, rows, H, 128)
    beta = randn(1, rows, H)
    starts = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    return q, k, v, g, beta, starts.to(DEVICE)


def run(state, slots, accepted, lens, tensors, weights, records):
    q, k, v, g, beta, starts = tensors
    a_log, dt_bias = weights
    if records:
        materialize(state, slots, accepted, len(lens))
    out, _ = fused_recurrent_kda(
        q=q, k=k, v=v, raw_g=g, raw_beta=beta, A_log=a_log, dt_bias=dt_bias,
        lower_bound=LOWER_BOUND, initial_state=state, cu_seqlens=starts,
        ssm_state_indices=slots, num_accepted_tokens=accepted, records=records,
    )  # fmt: skip
    return out


def accepted_state(state, slots, a):
    """Every request's state after its accepted token a - 1, in either format (the
    stock checkpoint at slot a - 1, or folded from slot 0 + slot 1's records)."""
    view = state.clone()
    n = slots.shape[0]
    accepted = torch.full((n,), a, dtype=torch.int32, device=DEVICE)
    materialize(view, slots, accepted, n)
    return view[slots[:, 0].long()]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stock_verify_record_format_matches_stock(seed):
    n, steps = 5, 6
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    gc = torch.Generator().manual_seed(seed)
    weights = (
        torch.randn(H, generator=g, device=DEVICE) * 0.5,
        torch.randn(H * 128, generator=g, device=DEVICE) * 0.5,
    )
    slots_total = n * Q + 1
    base = torch.randn(slots_total, H, 128, 128, generator=g, device=DEVICE) * 0.1
    stock, record = base.clone(), base.clone()
    order = torch.randperm(slots_total - 1, generator=gc)[: n * Q] + 1
    slots = order.view(n, Q).to(torch.int32).to(DEVICE)
    accepted = torch.ones(n, dtype=torch.int32, device=DEVICE)
    for step in range(steps):
        lens = (torch.randint(1, Q + 1, (n,), generator=gc)).tolist()
        tensors = inputs(lens, g)
        want = run(stock, slots, accepted, lens, tensors, weights, records=False)
        got = run(record, slots, accepted, lens, tensors, weights, records=True)
        assert_bitwise(f"step {step} outputs", got, want)
        for a in range(1, max(lens) + 1):
            live = [i for i in range(n) if lens[i] >= a]
            rows = torch.tensor(live, device=DEVICE)
            assert_bitwise(
                f"step {step} accepted state {a}",
                accepted_state(record, slots, a)[rows],
                stock[slots[rows, a - 1].long()],
            )
        nxt = [int(torch.randint(1, lens[i] + 1, (1,), generator=gc)) for i in range(n)]
        accepted = torch.tensor(nxt, dtype=torch.int32, device=DEVICE)


def align_meta(state):
    i64 = dict(dtype=torch.int64, device=DEVICE)
    i32 = dict(dtype=torch.int32, device=DEVICE)
    return (
        torch.tensor([state.data_ptr()], **i64),
        torch.tensor([state.stride(0) * 4], **i64),
        torch.tensor([4], **i32),
        torch.tensor([state[0].numel()], **i64),
        torch.zeros(1, **i32),  # temporal
        torch.zeros(1, **i32),  # group 0
        torch.zeros(1, **i32),
        torch.zeros(1, **i64),
    ), torch.tensor([H], **i32)


@pytest.mark.parametrize("records", [True, False])
def test_align_copies_fold(records):
    """Requests whose last verify accepted 1..8 tokens: the pre-copy into the old slot
    1 (the record page) and the align save in place into slot 0 leave the stock
    checkpoint of the accepted token, from a record-format (tagged) or a stock slot."""
    n = Q
    g = torch.Generator(device=DEVICE).manual_seed(7)
    weights = (
        torch.randn(H, generator=g, device=DEVICE) * 0.5,
        torch.randn(H * 128, generator=g, device=DEVICE) * 0.5,
    )
    cols = Q + 1
    slots_total = n * cols + 1
    base = torch.randn(slots_total, H, 128, 128, generator=g, device=DEVICE) * 0.1
    table = (torch.randperm(slots_total - 1, generator=torch.Generator().manual_seed(3))
             [: n * cols] + 1).view(n, cols).to(torch.int32).to(DEVICE)  # fmt: skip
    slots = table[:, :Q].contiguous()
    ones = torch.ones(n, dtype=torch.int32, device=DEVICE)
    lens = [Q] * n
    tensors = inputs(lens, g)
    stock = base.clone()
    run(stock, slots, ones, lens, tensors, weights, records=False)
    state = base.clone()
    run(state, slots, ones, lens, tensors, weights, records=records)
    bias = torch.arange(n, dtype=torch.int32, device=DEVICE)  # request i accepted i + 1
    want = stock[table[torch.arange(n), bias.long()].long()]
    meta, heads = align_meta(state)
    ptrs = torch.tensor([table.data_ptr()], dtype=torch.int64, device=DEVICE)
    extra = dict(state_record_heads_ptr=heads, HAS_RECORDS=True, num_warps=8)
    idx = torch.arange(n, dtype=torch.int32, device=DEVICE)

    pre = state.clone()
    meta, heads = align_meta(pre)
    mu._launch_align_kernel(
        mu.precopy_mamba_align_fused_kernel, n, 1,
        (ones, torch.zeros_like(ones), bias, ptrs, table.stride(0), *meta, idx, n),
        dict(COPY_BLOCK_SIZE=1024, CONV_STATE_DIM_FIRST=False, HAS_IDX_MAPPING=True,
             launch_pdl=False, **extra),
    )  # fmt: skip
    assert_bitwise("pre-copy into slot 1", pre[table[:, 1].long()], want)

    post = state.clone()
    meta, heads = align_meta(post)
    bs = 64
    computed = (bs - bias + bias).to(torch.int32)  # running = bs - bias, aligned = bs
    out = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    mu._launch_align_kernel(
        mu.postprocess_mamba_fused_kernel, n, 1,
        (bias + 1, torch.zeros_like(ones), None, computed, None, ptrs, table.stride(0),
         *meta, out, idx, n),
        dict(block_size=bs, COPY_BLOCK_SIZE=1024, CONV_STATE_DIM_FIRST=False,
             HAS_IDX_MAPPING=True, PRECOMPUTED_NEW_COMPUTED=True, launch_pdl=False,
             **extra),
    )  # fmt: skip
    assert_bitwise("align save into slot 0", post[table[:, 0].long()], want)
    assert out.eq(1).all()
