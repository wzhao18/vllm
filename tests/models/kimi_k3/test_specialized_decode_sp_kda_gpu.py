# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 SP decode KDA core (``cutedsl.kda.sp.SpKdaCore``, C512: 24 heads per rank,
FP32 state, q4, RecoverSSM records) vs the stock RecoverSSM verify of the same layer:
``KimiK3DeltaAttention._recoverssm_verify`` (Triton backend: ``causal_conv1d_update``
-> ``kda_recoverssm_verify``) then ``o_norm`` (``FusedRMSNormGated``), F_B by
``F.linear``.

Per request count n (one per record core: KdaSplit, KdaWS, KdaWide, KdaStream):

* the FP32 checkpoint is left unchanged and the conv window equals stock bit for bit;
* the recorded keys equal stock bit for bit, the recorded raw gates within one BF16
  ulp (F_B: the core's MMA vs cuBLAS), the corrections within FP32 reduction order;
* the o_norm-gated output within BF16 reduction-order tolerance;
* the stock Triton commit of either verify's records (random accepted counts) gives
  the same committed state (within the corrections' FP32 order);
* "deferred" mode writes the same records at each request's record slot.

One Rubin GPU; mostly CuTe compiles on a cold cache."""

import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("needs a Rubin GPU (compute capability 10.7)", allow_module_level=True)

os.environ.setdefault("VLLM_SSM_CONV_STATE_LAYOUT", "DS")  # C512's conv layout

from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    is_conv_state_dim_first,
)
from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention  # noqa: E402
from vllm.models.kimi_k3.nvidia.ops.recoverssm import (  # noqa: E402
    KDARecoverSSMCommitContext,
    KDARecoverSSMRecords,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.sp import (  # noqa: E402
    SpKdaCore,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Context  # noqa: E402
from vllm.third_party.flash_linear_attention.ops.kda import (  # noqa: E402
    FusedRMSNormGated,
)

DEVICE = torch.device("cuda", 0)
HEADS, DIM, Q = 24, 3072, 4  # C512: 96 KDA heads over TP4, DSpark q4
PITCH = 4 * DIM + 128 + HEADS + 8  # Q | K | V | gate | F_A | beta | pad (16)
MAX_REQS = 128
LOWER_BOUND = -5.0
# One request count per record core (sp.RECORD_KINDS at 24 heads).
REQUESTS = [
    pytest.param(1, id="split-n1"),
    pytest.param(5, id="split-n5"),
    pytest.param(8, id="ws-n8"),
    pytest.param(16, id="wide-n16"),
    pytest.param(32, id="stream-n32"),
    pytest.param(128, id="ws-n128"),
]

Module = SimpleNamespace


def context():
    sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    return Context(
        arch="rubin", tp=4, rank=0, dcp=4, sms=sms, q=Q, max_rows=512,
        max_context=1 << 20, bf16_state=False, dp=4, ep=16,
        sequence_parallel=True, heads_per_rank=HEADS,
    )  # fmt: skip


def layer(seed=0):
    """The fields the stock verify and ``SpKdaCore`` read from a TP4 KDA layer
    (stock dtypes: FP32 conv weight, A_log, dt bias; BF16 F_B and norm weight)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    bf16 = torch.bfloat16

    def randn(*shape, dtype=torch.float32, scale=1.0):
        return (torch.randn(*shape, generator=g, device=DEVICE) * scale).to(dtype)

    norm = FusedRMSNormGated(128, eps=1e-5, activation="sigmoid").to(DEVICE)
    with torch.no_grad():
        norm.weight.copy_(1 + randn(128, scale=0.1))
    return Module(
        prefix="model.layers.0.self_attn",
        head_dim=128,
        conv_size=4,
        spec_query_len=Q,
        local_num_heads=HEADS,
        local_projection_size=DIM,
        num_heads=4 * HEADS,
        conv1d=Module(weight=randn(3 * DIM, 1, 4, scale=0.5), bias=None),
        dt_bias=randn(DIM, scale=0.5),
        A_log=randn(HEADS, scale=0.5),
        f_b_proj=Module(weight=randn(DIM, 128, dtype=bf16, scale=128**-0.5)),
        in_proj_qkvgfab=Module(weight=torch.empty(PITCH, 8, device="meta")),
        o_norm=norm,
        gate_lower_bound=LOWER_BOUND,
        get_state_dtype=lambda: (torch.bfloat16, torch.float32),
        recoverssm_backend="triton",
        recoverssm_verify=None,
        recoverssm_records=KDARecoverSSMRecords.allocate(
            MAX_REQS, HEADS, Q, 128, torch.bfloat16, DEVICE
        ),
    )


def caches(slots, seed=1):
    """The DS conv window [slot, 3 dim, q + 2] (BF16) and the FP32 state."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    conv = torch.randn(slots, 3 * DIM, Q + 2, generator=g, device=DEVICE)
    state = torch.randn(slots, HEADS, 128, 128, generator=g, device=DEVICE) * 0.1
    return conv.to(torch.bfloat16), state


def step(n, slots, seed=2):
    """Projected rows [4 n, PITCH] and the uniform-verify metadata (one checkpoint
    slot per request, accepted 1: RecoverSSM)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    projected = torch.randn(Q * n, PITCH, generator=g, device=DEVICE)
    projected = projected.to(torch.bfloat16)
    order = torch.randperm(slots - 1, generator=torch.Generator().manual_seed(seed))
    checkpoints = (order[:n] + 1).to(torch.int32).view(n, 1).to(DEVICE)
    metadata = Module(
        spec_state_indices_tensor=checkpoints,
        num_accepted_tokens=torch.ones(n, dtype=torch.int32, device=DEVICE),
        num_spec_decodes=n,
    )
    return projected, metadata


def stock(attention, kv, projected, metadata):
    """The stock layer's RecoverSSM verify + o_norm on ``projected``."""
    rows = projected.shape[0]
    mixed = projected[:, : 3 * DIM]
    g2 = projected[:, 3 * DIM : 4 * DIM].reshape(1, rows, HEADS, 128)
    f_a = projected[:, 4 * DIM : 4 * DIM + 128]
    beta = projected[:, 4 * DIM + 128 : 4 * DIM + 128 + HEADS][None]
    g1 = torch.nn.functional.linear(f_a, attention.f_b_proj.weight)
    g1 = g1.view(1, rows, HEADS, 128)
    conv_weight = attention.conv1d.weight.view(3 * DIM, 4)
    starts = torch.arange(0, rows + 1, Q, dtype=torch.int32, device=DEVICE)
    attention.kv_cache = kv
    core = KimiK3DeltaAttention._recoverssm_verify(
        attention, metadata, mixed, g1, beta, conv_weight, starts, None
    )
    return attention.o_norm(core, g2).reshape(rows, DIM)


def bits(tensor):
    return tensor.view({2: torch.int16, 4: torch.int32}[tensor.element_size()])


def assert_bitwise(name, got, want):
    diff = bits(got) != bits(want)
    assert not diff.any(), f"{name}: {int(diff.sum())} of {diff.numel()} differ"


@pytest.mark.parametrize("n", REQUESTS)
def test_record_core_matches_stock_verify(n):
    assert is_conv_state_dim_first()
    rows, slots = Q * n, 2 * n + 2
    attention = layer()
    ref_attention = layer()  # its own records
    kv = caches(slots)
    ref_kv = tuple(t.clone() for t in kv)
    projected, metadata = step(n, slots)

    core = SpKdaCore(attention, (rows,), records="triton", ctx=context())
    core.prepare_caches(list(kv))
    out = torch.empty(rows, DIM, dtype=torch.bfloat16, device=DEVICE)
    core.run(projected, out, caches=list(kv), metadata=metadata)
    want = stock(ref_attention, ref_kv, projected, metadata)
    torch.cuda.synchronize()

    assert_bitwise("checkpoint", kv[1], ref_kv[1])
    assert_bitwise("conv window", kv[0], ref_kv[0])
    got_rec, want_rec = attention.recoverssm_records, ref_attention.recoverssm_records
    got_kg, want_kg = got_rec.key_gate[:n], want_rec.key_gate[:n]
    assert_bitwise("keys", got_kg[..., :128], want_kg[..., :128])
    torch.testing.assert_close(
        got_kg[..., 128:].float(),
        want_kg[..., 128:].float(),
        rtol=2**-7, atol=1e-6,
    )  # fmt: skip
    torch.testing.assert_close(
        got_rec.correction[:n], want_rec.correction[:n], rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(out.float(), want.float(), rtol=2e-2, atol=2e-2)

    # The stock Triton commit of each verify's records: the same committed state.
    g = torch.Generator().manual_seed(3)
    accepted = torch.randint(1, Q + 1, (n,), generator=g).to(torch.int32).to(DEVICE)
    starts = torch.arange(0, rows + 1, Q, dtype=torch.int32, device=DEVICE)
    checkpoints = metadata.spec_state_indices_tensor[:, 0].contiguous()
    for owner, cache in ((attention, kv), (ref_attention, ref_kv)):
        owner.kv_cache = cache
        commit = KDARecoverSSMCommitContext.create(
            [owner], spec_query_len=Q, max_num_reqs=MAX_REQS, backend="triton"
        )
        commit.commit(accepted, checkpoints, starts)
    torch.cuda.synchronize()
    torch.testing.assert_close(kv[1], ref_kv[1], rtol=1e-4, atol=1e-5)
    assert_bitwise("committed conv window", kv[0], ref_kv[0])


DEFERRED = [pytest.param(5, id="split-n5"), pytest.param(16, id="wide-n16")]


@pytest.mark.parametrize("n", DEFERRED)
def test_deferred_records_at_record_slots(n):
    """Deferred mode: the same records, at ``record_rows[b]`` (a slot permutation; the
    pending counts are zero, so the flush the core calls has nothing to fold)."""
    rows, slots = Q * n, 2 * n + 2
    triton_attention, deferred_attention = layer(), layer()
    deferred_attention.deferred_recoverssm = True
    kv = caches(slots)
    kv_deferred = tuple(t.clone() for t in kv)
    projected, metadata = step(n, slots)
    record_rows = torch.randperm(MAX_REQS, generator=torch.Generator().manual_seed(4))
    record_rows = record_rows[:n].to(torch.int32).to(DEVICE)
    flushes = []
    metadata_deferred = Module(
        **vars(metadata),
        recoverssm_deferred=Module(
            record_rows=record_rows,
            pending=torch.zeros(MAX_REQS, dtype=torch.int32, device=DEVICE),
            flush=lambda: flushes.append(1),
        ),
    )
    out = torch.empty(rows, DIM, dtype=torch.bfloat16, device=DEVICE)
    out_deferred = torch.empty_like(out)
    plain = SpKdaCore(triton_attention, (rows,), records="triton", ctx=context())
    plain.prepare_caches(list(kv))
    plain.run(projected, out, caches=list(kv), metadata=metadata)
    core = SpKdaCore(deferred_attention, (rows,), records="deferred", ctx=context())
    core.prepare_caches(list(kv_deferred))
    caches_deferred = list(kv_deferred)
    core.run(
        projected, out_deferred, caches=caches_deferred, metadata=metadata_deferred
    )
    torch.cuda.synchronize()
    assert flushes == [1]
    assert_bitwise("output", out_deferred, out)
    assert_bitwise("conv window", kv_deferred[0], kv[0])
    want = triton_attention.recoverssm_records
    got = deferred_attention.recoverssm_records
    index = record_rows.long()
    assert_bitwise("corrections", got.correction[index], want.correction[:n])
    assert_bitwise("key / gate", got.key_gate[index], want.key_gate[:n])
