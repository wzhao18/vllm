# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 SP (C512: TP4 = DCP4 = SP4) decode projections on Rubin (compute
capability 10.7): ``DecodeProjection`` with ``sp_policy``'s tactics at the TP4
per-rank shapes (DSA packed in-projection 5344 x 7168, KDA in-projection 12448 x 7168,
q_b 4608 x 1536, o_proj 7168 x 3072) vs an FP32 reference rounded once to BF16:

* every tactic kind (warp split-K, cluster split-K, cuBLAS) at R in {4, 8, 16, 20, 32,
  64, 68, 128, 324, 512}: within one BF16 rounding of the reference (the split-K
  kinds reorder the FP32 sum); cuBLAS bitwise equal to ``torch.mm``, the stock GEMM;
* a captured graph replays every R with fresh inputs;
* the policy covers every captured R (4-512) and refuses non-SP contexts; operands
  the selected tactics cannot take raise at build (no runtime fallback).

One GPU; about a minute of CuTe compiles on a cold cache."""

import pytest
import torch

pytest.importorskip("cutlass")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("needs a Rubin GPU (compute capability 10.7)", allow_module_level=True)

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.decode import (  # noqa: E402
    DecodeProjection,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sp_policy import (  # noqa: E402
    sp_projection_tactics,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import (  # noqa: E402
    Context,
    Path,
    Unsupported,
)

DEVICE = torch.device("cuda", 0)
SHAPES = {
    "dsa_in": (5344, 7168),
    "kda_in": (12448, 7168),
    "q_b": (4608, 1536),
    "o_proj": (7168, 3072),
}
CAPTURED = tuple(range(4, 513, 4))
ROWS = (4, 8, 16, 20, 32, 64, 68, 128, 324, 512)


def context(**overrides):
    torch.cuda.set_device(DEVICE)
    sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    fields = dict(
        arch="rubin", tp=4, rank=0, dcp=4, sms=sms, q=4, max_rows=512,
        max_context=131072, bf16_state=False, dp=4, ep=16, sequence_parallel=True,
        heads_per_rank=24,
    )  # fmt: skip
    return Context(**(fields | overrides))


def operands(n, k, rows, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(rows, k, device=DEVICE, generator=g).bfloat16()
    w = (torch.randn(n, k, device=DEVICE, generator=g) * 0.02).bfloat16()
    return x, w


def check_close(out, x, w):
    ref = (x.float() @ w.float().t()).bfloat16().float()
    err = (out.float() - ref).abs()
    bound = ref.abs() * 2**-7 + 2**-12
    assert bool((err <= bound).all()), f"max err {err.max().item()} over the BF16 bound"


@pytest.mark.parametrize("kind", list(SHAPES))
def test_policy_covers_captured(kind):
    n, k = SHAPES[kind]
    tactics = sp_projection_tactics(kind, n, k, CAPTURED, context())
    assert set(tactics) == set(CAPTURED)
    kinds = {r: t[0] for r, t in tactics.items()}
    assert {kinds[r] for r in CAPTURED if r <= 16} == {"warp"}
    assert {kinds[r] for r in CAPTURED if r > 64} == {"cublas"}


def test_policy_refuses_outside_sp():
    with pytest.raises(Unsupported, match="TP4"):
        sp_projection_tactics("dsa_in", 5344, 7168, CAPTURED, context(tp=8))
    with pytest.raises(Unsupported, match="TP4"):
        ctx = context(sequence_parallel=False)
        sp_projection_tactics("dsa_in", 5344, 7168, CAPTURED, ctx)


@pytest.mark.parametrize("kind", list(SHAPES))
def test_projection_matches_reference(kind):
    n, k = SHAPES[kind]
    x, w = operands(n, k, max(ROWS), seed=len(kind))
    tactics = sp_projection_tactics(kind, n, k, ROWS, context())
    projection = DecodeProjection(w, ROWS, tactics, family=kind)
    out = torch.empty(max(ROWS), n, device=DEVICE, dtype=torch.bfloat16)
    projection.check_operands(x, out)
    for r in ROWS:
        out.fill_(float("nan"))
        projection.run(x[:r], out[:r])
        check_close(out[:r], x[:r], w)
        if projection.kinds[r] == "cublas":
            assert torch.equal(out[:r], torch.mm(x[:r], w.t()))
        assert projection.choice(r).path in (Path.GENERIC, Path.TUNED)


def test_projection_graph_replay():
    n, k = SHAPES["dsa_in"]
    x, w = operands(n, k, max(ROWS), seed=7)
    projection = DecodeProjection(
        w, ROWS, sp_projection_tactics("dsa_in", n, k, ROWS, context())
    )
    out = torch.empty(max(ROWS), n, device=DEVICE, dtype=torch.bfloat16)
    for r in ROWS:  # warm (compiles and cuBLAS workspaces) before capture
        projection.run(x[:r], out[:r])
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for r in ROWS:
            projection.run(x[:r], out[:r])
    for seed in (1, 2):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        x.copy_(torch.randn(x.shape, device=DEVICE, generator=g))
        graph.replay()
        check_close(out[: max(ROWS)], x, w)  # the last (largest) R wrote every row


def test_misaligned_operands_raise():
    n, k = SHAPES["dsa_in"]
    x, w = operands(n, k, 65, seed=3)
    rows = (4, 64)
    projection = DecodeProjection(
        w, rows, sp_projection_tactics("dsa_in", n, k, rows, context())
    )
    out = torch.empty(65, n, device=DEVICE, dtype=torch.bfloat16)
    flat = torch.empty(64 * k + 8, device=DEVICE, dtype=torch.bfloat16)
    shifted = flat[8:].view(64, k)  # 16 B off a 32 B boundary
    with pytest.raises(Unsupported, match="aligned"):
        projection.check_operands(shifted, out[:64])
    with pytest.raises(Unsupported, match="packed"):
        projection.check_operands(x[:64, : k // 2], out[:64])
    with pytest.raises(Unsupported, match="no tactic"):
        DecodeProjection(w, (4, 8), {4: ("cublas", None)})
