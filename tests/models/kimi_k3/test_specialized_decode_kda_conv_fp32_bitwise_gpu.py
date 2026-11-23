# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCRIT r125: the SP KDA record cores on an FP32 conv state are BITWISE the BF16-conv
run of the same step (the conv window holds BF16 in-projection inputs whatever its
dtype), for every record core (KdaSplit n2, KdaWide n9, KdaFlow n33 at q2 / q4,
KdaWide at q3 / q5..q8) and verify length q2..q8:

* the verify outputs, the recurrent checkpoints, the key / gate records and the
  corrections: bitwise equal between FP32 and BF16 conv;
* the conv window after the verify and after the stock Triton commit (rollback of
  rejected tokens): FP32 window == BF16 window .float(), bitwise;
* the invariant: every FP32 conv-state value is BF16-representable
  (state == state.bfloat16().float()) after the verify and after the commit.

(The FP32-conv run against the stock FP32-conv reference: test_specialized_decode_
kda_conv_fp32_gpu, the same tolerances as the BF16-conv tests.)"""

import pytest
import torch

from tests.models.kimi_k3 import test_specialized_decode_sp_kda_gpu as kda
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.sp import (
    SpKdaCore,
    record_kind,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm.context import (
    KDARecoverSSMCommitContext,
)

CASES = [
    pytest.param(q, n, id=f"q{q}-n{n}")
    for q in (2, 3, 4, 5, 6, 8)
    for n in (2, 9, 33)
]


def run(q, n, conv_dtype, monkeypatch):
    """One verify step + the stock Triton commit at ``conv_dtype``; returns the
    tensors to compare."""
    monkeypatch.setattr(kda, "Q", q)
    monkeypatch.setattr(kda, "CONV_DTYPE", conv_dtype)
    rows, slots = q * n, 2 * n + 2
    attention = kda.layer()
    conv, state = kda.caches(slots)  # BF16-valued window at either dtype
    projected, metadata = kda.step(n, slots)
    core = SpKdaCore(attention, (rows,), records="triton", ctx=kda.context())
    core.prepare_caches([conv, state])
    out = torch.empty(rows, kda.DIM, dtype=torch.bfloat16, device=kda.DEVICE)
    core.run(projected, out, caches=[conv, state], metadata=metadata)
    torch.cuda.synchronize()
    after_verify = (conv.clone(), state.clone())
    records = attention.recoverssm_records
    g = torch.Generator().manual_seed(3)
    accepted = torch.randint(1, q + 1, (n,), generator=g).to(torch.int32)
    accepted = accepted.to(kda.DEVICE)
    starts = torch.arange(0, rows + 1, q, dtype=torch.int32, device=kda.DEVICE)
    checkpoints = metadata.spec_state_indices_tensor[:, 0].contiguous()
    attention.kv_cache = (conv, state)
    commit = KDARecoverSSMCommitContext.create(
        [attention], spec_query_len=q, max_num_reqs=kda.MAX_REQS, backend="triton"
    )
    commit.commit(accepted, checkpoints, starts)
    torch.cuda.synchronize()
    return dict(
        kind=core.kinds[rows],
        out=out,
        verify_conv=after_verify[0],
        verify_state=after_verify[1],
        key_gate=records.key_gate[:n].clone(),
        correction=records.correction[:n].clone(),
        commit_conv=conv,
        commit_state=state,
    )


@pytest.mark.parametrize("q,n", CASES)
def test_fp32_conv_bitwise_bf16_conv(q, n, monkeypatch):
    assert kda.is_conv_state_dim_first()
    bf16 = run(q, n, torch.bfloat16, monkeypatch)
    fp32 = run(q, n, torch.float32, monkeypatch)
    assert fp32["kind"] == bf16["kind"] == record_kind(kda.HEADS, n, q), (
        fp32["kind"], bf16["kind"],
    )  # fmt: skip
    for name in ("out", "verify_state", "key_gate", "correction", "commit_state"):
        kda.assert_bitwise(f"{fp32['kind']} {name}", fp32[name], bf16[name])
    for name in ("verify_conv", "commit_conv"):
        assert fp32[name].dtype == torch.float32
        kda.assert_bitwise(
            f"{fp32['kind']} {name} (fp32 vs bf16.float())",
            fp32[name], bf16[name].float(),
        )  # fmt: skip
        kda.assert_bitwise(
            f"{fp32['kind']} {name} BF16-representable",
            fp32[name], fp32[name].bfloat16().float(),
        )  # fmt: skip
