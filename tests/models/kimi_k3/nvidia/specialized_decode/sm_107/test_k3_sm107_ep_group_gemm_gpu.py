# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 replicated EP_GROUP decode projections on Rubin (compute capability 10.7):
``DecodeProjection`` with ``sp_policy``'s tactics (re-keyed by (G, DCP, N, K)) at the
per-rank shapes of the TP4 (24 heads) and TP8 (12 heads) attention groups under
expert parallelism, the rows the replicated step captures (R = n q <= 64, q 2 / 4 /
8), vs an FP32 reference rounded once to BF16:

* every family: the DSA packed in-projection (5344 / 3808 x 7168), the KDA
  in-projection (12448 / 6288 x 7168) and the o_proj partial (7168 x 3072 / 1536),
  whose rows ``group_ep`` hands to stock's reduce-scatter;
* every tactic kind the GENERIC rule picks there (warp split-K up to 16 rows, the
  cluster split-K at 17-64): within one BF16 rounding of the reference (the split-K
  kinds reorder the FP32 sum);
* the policy covers every captured R of each (G, DCP, q) and keys a measured row by
  its group: a TUNED entry of another (G, DCP) is not this group's;
* a captured graph replays every R with fresh inputs.

One GPU; about a minute of CuTe compiles on a cold cache."""

import pytest
import torch

pytest.importorskip("cutlass")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("needs a Rubin GPU (compute capability 10.7)", allow_module_level=True)

from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.gemm import (  # noqa: E402
    sp_policy,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.gemm.decode import (  # noqa: E402
    DecodeProjection,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.gemm.sp_policy import (  # noqa: E402
    sp_projection_tactics,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.shapes import (  # noqa: E402
    Context,
    Path,
)
from vllm.platforms import current_platform  # noqa: E402

_SM107_ONLY = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)),
    reason="the SM107 specialized decode runs on compute capability 10.7 only",
)
pytestmark = _SM107_ONLY

DEVICE = torch.device("cuda", 0)
WIDTH = 7168


def shapes_of(heads):
    """The per-rank [N, K] weights of one attention group at ``heads`` per rank."""
    dsa = 2112 + heads * 128 + 128 + 32
    kda = -(-(4 * heads * 128 + 128 + heads) // 16) * 16
    return {
        "dsa_in": (dsa, WIDTH),
        "kda_in": (kda, WIDTH),
        "o_proj": (WIDTH, heads * 128),
    }


# (G, DCP, heads per rank): TP4 at DCP1 (the replicated step's TP4 group), TP8 at
# DCP 1 and 8.
GROUPS = [
    pytest.param(4, 1, 24, id="tp4-dcp1-h24"),
    pytest.param(8, 1, 12, id="tp8-dcp1-h12"),
    pytest.param(8, 8, 12, id="tp8-dcp8-h12"),
]


def context(g, dcp, heads, q, **overrides):
    torch.cuda.set_device(DEVICE)
    sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    fields = dict(
        arch="rubin", tp=g, rank=0, dcp=dcp, sms=sms, q=q, max_rows=64,
        max_context=131072, bf16_state=False, dp=2, ep=2 * g,
        heads_per_rank=heads, ep_group=True,
    )  # fmt: skip
    return Context(**(fields | overrides))


def rows_of(q):
    return tuple(range(q, 65, q))


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


@pytest.mark.parametrize("q", [2, 4, 8])
@pytest.mark.parametrize("g,dcp,heads", GROUPS)
def test_policy_covers_captured(g, dcp, heads, q):
    """Every captured R of the group gets a structurally valid tactic: warp up to 16
    rows, split-K at 17-64 (the replicated step stops at 64)."""
    for kind, (n, k) in shapes_of(heads).items():
        rows = rows_of(q)
        tactics = sp_projection_tactics(kind, n, k, rows, context(g, dcp, heads, q))
        assert set(tactics) == set(rows), kind
        kinds = {r: t[0] for r, t in tactics.items()}
        assert {kinds[r] for r in rows if r <= 16} == {"warp"}, kind
        assert {kinds[r] for r in rows if 16 < r <= 64} <= {"split_k"}, kind


@pytest.mark.parametrize("g,dcp,heads", GROUPS)
def test_projection_matches_reference(g, dcp, heads):
    """Each family at q4's rows (4..64) and q8's odd-request rows, within one BF16
    rounding of the FP32 reference; the coverage choice GENERIC (no table)."""
    rows = tuple(sorted({*rows_of(4), 8, 24, 40, 56}))
    ctx = context(g, dcp, heads, 4)
    for seed, (kind, (n, k)) in enumerate(shapes_of(heads).items()):
        x, w = operands(n, k, max(rows), seed=seed + heads)
        tactics = sp_projection_tactics(kind, n, k, rows, ctx)
        projection = DecodeProjection(w, rows, tactics, family=kind, group=(g, dcp))
        out = torch.empty(max(rows), n, device=DEVICE, dtype=torch.bfloat16)
        projection.check_operands(x, out)
        for r in rows:
            out.fill_(float("nan"))
            projection.run(x[:r], out[:r])
            check_close(out[:r], x[:r], w)
            assert projection.choice(r).path is Path.GENERIC


def test_measured_rows_are_keyed_by_group(monkeypatch):
    """A TUNED entry belongs to its (G, DCP): TP8 = DCP8's 12-head in-projection
    entry is TUNED there and GENERIC at TP8 DCP1 (same weight, other overlap)."""
    n, k = shapes_of(12)["dsa_in"]
    monkeypatch.setitem(sp_policy.TUNED_KIND, (8, 8, n, k), {"*": "8:cublas"})
    rows = (8, 16)
    tuned = sp_projection_tactics("dsa_in", n, k, rows, context(8, 8, 12, 4))
    plain = sp_projection_tactics("dsa_in", n, k, rows, context(8, 1, 12, 4))
    assert tuned[8] == ("cublas", None) and plain[8][0] == "warp"
    x, w = operands(n, k, 16, seed=5)
    on = DecodeProjection(w, rows, tuned, family="dsa_in", group=(8, 8))
    off = DecodeProjection(w, rows, plain, family="dsa_in", group=(8, 1))
    assert on.choice(8).path is Path.TUNED and on.choice(16).path is Path.GENERIC
    assert off.choice(8).path is Path.GENERIC
    out = torch.empty(16, n, device=DEVICE, dtype=torch.bfloat16)
    on.run(x[:8], out[:8])
    assert torch.equal(out[:8], torch.mm(x[:8], w.t()))  # cuBLAS: the stock GEMM


@pytest.mark.parametrize("g,dcp,heads", GROUPS)
def test_oproj_graph_replay(g, dcp, heads):
    """The o_proj partial (the replicated step's reduce-scatter input) replays in a
    CUDA graph at every q4 row with fresh inputs."""
    n, k = shapes_of(heads)["o_proj"]
    rows = rows_of(4)
    x, w = operands(n, k, max(rows), seed=7)
    tactics = sp_projection_tactics("o_proj", n, k, rows, context(g, dcp, heads, 4))
    projection = DecodeProjection(w, rows, tactics, family="oproj", group=(g, dcp))
    out = torch.empty(max(rows), n, device=DEVICE, dtype=torch.bfloat16)
    for r in rows:  # warm (compiles) before capture
        projection.run(x[:r], out[:r])
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for r in rows:
            projection.run(x[:r], out[:r])
    for seed in (1, 2):
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        x.copy_(torch.randn(x.shape, device=DEVICE, generator=gen))
        graph.replay()
        check_close(out[: max(rows)], x, w)  # the last (largest) R wrote every row
