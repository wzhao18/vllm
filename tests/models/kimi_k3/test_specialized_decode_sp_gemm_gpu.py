# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 SP (C512: TP4 = DCP4 = SP4) decode projections on Rubin (compute
capability 10.7): ``DecodeProjection`` with ``sp_policy``'s tactics at the TP4
per-rank shapes (DSA packed in-projection 5344 x 7168, KDA in-projection 12448 x 7168,
q_b 4608 x 1536, o_proj 7168 x 3072, indexer wq_b 4096 x 1536; the MegaMoE layer's
latent down 3584 x 7168 / up 7168 x 3584 and shared gate_up 12288 x 7168 / down
7168 x 6144 on shard rows) vs an FP32 reference rounded once to BF16:

* every tactic kind (warp split-K, cluster split-K, cuBLAS, cuBLASLt) at R in {4, 8,
  16, 20, 32, 64, 68, 128, 324, 512} (the MLP families at their shard rows, the
  split-K 80 / 96-column tiles full and partial among them): within
  one BF16 rounding of the reference (the split-K kinds reorder the FP32 sum);
  cuBLAS bitwise equal to ``torch.mm``, the stock GEMM, and cuBLASLt to ``torch.mm``
  under torch's cuBLASLt preference; the beta-add (latent up) bitwise equal to
  stock's ``residual.addmm_``;
* a captured graph replays every R with fresh inputs;
* ``choice`` reports TUNED exactly where the family's own table (main or side
  stream) names the tactic;
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
    blas_library,
    warp_tactic,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sp_policy import (  # noqa: E402
    measured,
    sp_projection_tactics,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.warp_splitk import (  # noqa: E402
    WarpSplitKTactic,
    smem_limits,
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
    "wq_b": (4096, 1536),
    "latent_down": (3584, 7168),
    "shared_gu": (12288, 7168),
    "shared_down": (7168, 6144),
    "latent_up": (7168, 3584),
    "shared_gu_side": (12288, 7168),
    "shared_down_side": (7168, 6144),
}
MLP = ("latent_down", "shared_gu", "shared_down", "latent_up", "shared_gu_side",
       "shared_down_side")  # fmt: skip
CAPTURED = tuple(range(4, 513, 4))
ROWS = (4, 8, 16, 20, 32, 64, 68, 84, 128, 324, 512)
# Shard rows n = R / 4 of the MLP families (every kind of their tables).
SHARD_CAPTURED = tuple(range(1, 129))
# 72 / 80: the split-K 80-column tile partial / full; 88: a partial 96-column tile.
SHARD_ROWS = (1, 4, 16, 20, 32, 50, 64, 72, 80, 88, 96, 128)


def rows_of(kind, captured=False):
    if kind in MLP:
        return SHARD_CAPTURED if captured else SHARD_ROWS
    return CAPTURED if captured else ROWS


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
    captured = rows_of(kind, captured=True)
    tactics = sp_projection_tactics(kind, n, k, captured, context())
    assert set(tactics) == set(captured)
    kinds = {r: t[0] for r, t in tactics.items()}
    assert set(kinds.values()) <= {"warp", "split_k", "cublas", "lt"}
    if kind == "latent_up" or kind.endswith("_side"):  # torch kinds only
        assert set(kinds.values()) <= {"cublas", "lt"}
    if kind == "q_b":  # no table: the GENERIC rule
        assert {kinds[r] for r in captured if r <= 16} == {"warp"}
        assert {kinds[r] for r in captured if r > 64} == {"cublas"}
    if kind == "kda_in":  # warp loses to cuBLAS on the 178 MB weight (28.9 vs 15.5 us)
        assert kinds[16] == "cublas" and kinds[32] == "lt"
    if kind == "wq_b":  # no 64-column split-K tiles above 64 rows
        assert {kinds[r] for r in captured if r > 64} <= {"cublas", "lt"}


def test_measured_is_per_family():
    """TUNED labels follow the family's own table, not any table with the shape."""
    down = SHAPES["shared_down"]
    split = ("split_k", (64, 64, 1, 7))
    assert measured("shared_down", *down, 50, split)
    assert not measured("shared_down", *down, 50, ("lt", None))  # the side table's
    assert measured("shared_down_side", *down, 50, ("lt", None))
    assert not measured("shared_down_side", *down, 50, split)
    assert not measured("shared_down", *down, 50, ("split_k", (64, 32, 1, 9)))
    sms = context().sms
    sm_bytes = smem_limits(DEVICE.index).sm
    latent = SHAPES["latent_down"]
    warp = ("warp", warp_tactic(4, *latent, sms, sm_bytes))
    assert measured("latent_down", *latent, 4, warp, sms, sm_bytes)
    other = ("warp", WarpSplitKTactic(16, 8, 128, 2, 2))
    assert other != warp
    assert not measured("latent_down", *latent, 4, other, sms, sm_bytes)
    q_b = SHAPES["q_b"]  # no table: the GENERIC warp rule is never TUNED
    assert not measured("q_b", *q_b, 8, ("warp", warp_tactic(8, *q_b, sms, sm_bytes)))


def test_policy_refuses_outside_sp():
    with pytest.raises(Unsupported, match="TP4"):
        sp_projection_tactics("dsa_in", 5344, 7168, CAPTURED, context(tp=8))
    with pytest.raises(Unsupported, match="TP4"):
        ctx = context(sequence_parallel=False)
        sp_projection_tactics("dsa_in", 5344, 7168, CAPTURED, ctx)


@pytest.mark.parametrize("kind", list(SHAPES))
def test_projection_matches_reference(kind):
    n, k = SHAPES[kind]
    rows = rows_of(kind)
    x, w = operands(n, k, max(rows), seed=len(kind))
    tactics = sp_projection_tactics(kind, n, k, rows, context())
    projection = DecodeProjection(w, rows, tactics, family=kind)
    out = torch.empty(max(rows), n, device=DEVICE, dtype=torch.bfloat16)
    projection.check_operands(x, out)
    for r in rows:
        kind_r = projection.kinds[r]
        if kind == "latent_up":  # stock's beta-add, in place into the residual
            residual = torch.randn(r, n, device=DEVICE).bfloat16()
            want = residual.clone()
            with blas_library("cublaslt" if kind_r == "lt" else None):
                want.addmm_(x[:r], w.t())
            got = projection.run(x[:r], residual=residual)
            assert got is residual and torch.equal(got, want), r
            continue
        out.fill_(float("nan"))
        projection.run(x[:r], out[:r])
        check_close(out[:r], x[:r], w)
        if kind_r in ("cublas", "lt"):
            with blas_library("cublaslt" if kind_r == "lt" else None):
                assert torch.equal(out[:r], torch.mm(x[:r], w.t())), (r, kind_r)
        assert projection.choice(r).path in (Path.GENERIC, Path.TUNED)


def test_residual_needs_a_torch_kind():
    n, k = SHAPES["latent_down"]
    x, w = operands(n, k, 4, seed=11)
    projection = DecodeProjection(
        w, (4,), sp_projection_tactics("latent_down", n, k, (4,), context())
    )
    assert projection.kinds[4] == "warp"
    with pytest.raises(Unsupported, match="no residual epilogue"):
        projection.run(x, residual=torch.zeros(4, n, device=DEVICE).bfloat16())


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
