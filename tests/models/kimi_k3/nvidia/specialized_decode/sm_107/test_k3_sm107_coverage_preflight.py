# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from the source branch's tests/models/kimi_k3/test_specialized_decode_coverage_preflight.py for the SM107
# specialized decode (vllm/models/kimi_k3/nvidia/specialized_decode/sm_107).
"""Coverage preflight of the Kimi K3 specialized decode (CPU only, seconds, no GPU).

NQ-NOFALLBACK: on the target topology (Rubin VR200 = SM107, TP16 = DCP16, DSpark
verify q4 / q8, the new drafter: 96 heads, 6 per rank, Markov rank 512) the build
refuses (``shapes.Uncovered``) every captured shape a family does not cover, so an
uncovered shape is only found when a 16-GPU job starts. This preflight runs the same
coverage decisions on CPU, against a faked VR200 (212 SMs, SM107 shared memory), for

- every cfg2bench point (c1, c8, c48-1p1d, c48-2p1d, c64, c160, c256: the decode
  role's max-num-seqs, capture cap and capture list, as ``cfg2bench.build`` makes
  them), its 'hi' rows (rows above the production limit, as the n4 harness ran them),
  and
- an arbitrary deployment: every verify row 1..1024 (q4 and q8) captured, max-num-seqs
  raised to match (``cfg2bench.build --rows`` of every row),

and asserts that no build-time site refuses, that every captured row runs the
specialized step, and that every kernel whose CTAs wait on other ranks' same-kernel
CTAs (cross-rank Lamport polls inside one grid) fits one wave of the device. Each
failure names the site (file), the rows and why the limit exists.

What CPU cannot know is each kernel's occupancy (registers, the compiled shared
memory): ``VR200_BLOCKS_PER_SM`` records the measured one per kernel, with its
provenance. Everything else runs the code's own tables, planners and grid formulas.
"""

import importlib.util
from contextlib import nullcontext
from pathlib import Path as FilePath
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from vllm.platforms import current_platform  # noqa: E402

# The SM107 specialized decode (specialized_decode/sm_107) serves VR200 only.
_SM107_ONLY = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)),
    reason="the SM107 specialized decode runs on compute capability 10.7 only",
)
pytestmark = _SM107_ONLY

ROOT = FilePath(__file__).resolve().parents[3]
# The checkout's harness, else the copy the cluster runs (a staged overlay ships only
# vllm/ and tests/).
CFG2BENCH = next(
    (
        path
        for path in (
            ROOT / "benchmarks/kernels/k31_vr/nq/cfg2bench.py",
            FilePath(
                "/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/nqbench/tools/"
                "cfg2bench.py"
            ),
        )
        if path.exists()
    ),
    ROOT / "benchmarks/kernels/k31_vr/nq/cfg2bench.py",
)

# ---------------------------------------------------------------- the faked VR200

VR200_SMS = 212
SMEM_SM107 = 334848  # opt-in shared memory per CTA (markov_wide: SM107's opt-in)
# (SMs, opt-in shared memory per CTA, shared memory per SM): markov.device_limits
LIMITS_VR200 = (VR200_SMS, SMEM_SM107, SMEM_SM107 + 1024)
# Rubin clusters of the wide DSA MlaQuery resident at once (dsa.wide
# check_query_residency: cuOccupancyMaxActiveClusters, 8-CTA clusters, measured 22).
VR200_QUERY_CLUSTERS = 22
# cuOccupancyMaxActiveBlocksPerMultiprocessor of the kernels whose whole grid must be
# co-resident (their CTAs poll what other ranks' CTAs of the same grid publish):
# - CtxGather (shape-dynamic, 256 threads): 1. Measured on VR200 by the n4 nq bench at
#   f5b39b1: PreparedContext kept 208 context rows and dropped 216 (= 1 x 212 SMs).
#   Above it the persistent CtxGather(ctas=resident) runs (one wave).
# - ArNormRows (1024 threads, push sites): at most 2 by threads (2048 per SM); its
#   registers are not known on CPU: 1, the only occupancy guaranteed without a
#   measurement (PreparedDraftAr reads the real one and goes persistent above it).
# - DcpValue (512 threads, 103 registers): 1 (mla.dcp: "one 512-thread CTA per SM").
VR200_BLOCKS_PER_SM = {"CtxGather": 1, "ArNormRows": 1, "DcpValue": 1}

DRAFT_HEADS = 6  # the new drafter: 96 heads / TP16
DRAFT_RANK = 512  # its Markov rank
VOCAB = 163840
HIDDEN = 7168

# ------------------------------------------------------------ the cfg2bench points

# The decode role of nqbench/cfg/nq-<point>-xdr.json (the snapshot cfg2bench reads):
# K, max-num-seqs, max-cudagraph-capture-size and the capture list.
Q8_SIZES = (
    "7,8,14,16,21,24,28,32,35,40,42,48,49,56,63,64,70,72,77,80,84,88,91,96,98,104,105,"
    "112,119,120,126,128,133,136,140,144,147,152,154,160,161,168,175,176,182,184,189,"
    "192,196,200,203,208,210,216,217,224,232,240,248,256"
)
Q4_SIZES_256 = (
    "6,8,15,16,18,20,21,24,30,32,39,40,45,48,54,56,63,64,66,68,69,72,75,76,78,80,81,84,"
    "87,88,90,92,93,96,100,102,104,108,112,114,116,120,124,126,128,132,136,138,144,150,"
    "152,156,160,162,168,174,176,180,184,186,192,200,208,216,224,232,240,248,256"
)
Q4_SIZES_512 = (
    "6,8,15,16,18,20,21,24,30,32,39,40,45,48,54,56,63,64,66,68,69,72,75,76,78,80,81,84,"
    "87,88,90,92,93,96,100,102,104,108,112,114,116,120,124,126,128,132,136,138,144,150,"
    "152,156,160,162,168,174,176,180,184,186,192,200,204,208,216,224,228,232,240,248,"
    "252,256,264,272,276,288,300,304,312,320,324,336,348,352,360,368,372,384,400,416,"
    "432,448,464,480,496,512"
)
POINTS = {
    "c1": (7, 32, 256, Q8_SIZES),
    "c8": (7, 32, 256, Q8_SIZES),
    "c48-1p1d": (7, 32, 256, Q8_SIZES),
    "c64": (7, 32, 256, Q8_SIZES),
    "c48-2p1d": (3, 64, 256, Q4_SIZES_256),
    "c160": (3, 128, 512, Q4_SIZES_512),
    "c256": (3, 128, 512, Q4_SIZES_512),
}
# The rows the n4 bench ran above the production limit ('hi': c64-hi, c256-hi).
HI_ROWS = {7: (512, 1024), 3: (768, 1024)}
TOP_ROWS = 1024


def load_cfg2bench():
    spec = importlib.util.spec_from_file_location("nq_cfg2bench", CFG2BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def point_doc(k, mns, cap, sizes):
    import json

    comp = {"cudagraph_capture_sizes": [int(s) for s in sizes.split(",")]}
    args = {
        "tensor-parallel-size": 16,
        "decode-context-parallel-size": 16,
        "speculative-config": json.dumps(
            {"method": "dspark", "num_speculative_tokens": k}
        ),
        "compilation-config": json.dumps(comp),
        "max-num-seqs": mns,
        "max-cudagraph-capture-size": cap,
    }
    return {"roles": {"decode": {"args": args}}}


def deployments():
    """(name, q, max-num-seqs, capture cap, capture sizes) of every point, its 'hi'
    rows, and the arbitrary deployment of each q (every verify row up to 1024)."""
    cfg2bench = load_cfg2bench()
    out = []
    for name, (k, mns, cap, sizes) in POINTS.items():
        doc = point_doc(k, mns, cap, sizes)
        q = k + 1
        variants = {
            name: [],
            f"{name}-hi": list(HI_ROWS[k]),
            f"{name}-any": list(range(q, TOP_ROWS + 1, q)),
        }
        for label, rows in variants.items():
            built = cfg2bench.build(doc, "ls", "new", rows)
            sizes_ = tuple(built["capture_sizes"])
            out.append((label, q, built["mns"], built["max_capture"], sizes_))
    return out


DEPLOYMENTS = deployments() if CFG2BENCH.exists() else []
IDS = [d[0] for d in DEPLOYMENTS]


def test_deployments_found():
    """The preflight must not pass by covering nothing: without the harness the
    deployment-parametrized checks would all be skipped."""
    assert DEPLOYMENTS, f"no deployments: {CFG2BENCH} not found"


def test_points_match_the_snapshots():
    """The embedded points are the nqbench/cfg snapshots' decode roles (where the
    snapshots are reachable: the cluster's Lustre)."""
    import json

    cfg2bench = load_cfg2bench()
    checked = 0
    for name, (k, mns, cap, sizes) in POINTS.items():
        path = FilePath(cfg2bench.NB) / "cfg" / f"nq-{name}-xdr.json"
        if not path.exists():
            continue
        args = json.loads(path.read_text())["roles"]["decode"]["args"]
        spec = json.loads(args["speculative-config"])
        comp = json.loads(args.get("compilation-config") or "{}")
        assert spec["num_speculative_tokens"] == k, name
        assert int(args["max-num-seqs"]) == mns, name
        assert int(args.get("max-cudagraph-capture-size") or 0) == cap, name
        listed = sorted(set(comp.get("cudagraph_capture_sizes") or ()))
        assert listed == [int(s) for s in sizes.split(",")], name
        checked += 1
    if not checked:
        pytest.skip("nqbench/cfg snapshots not reachable")


# ----------------------------------------------------------------------- helpers


class Gaps:
    """Every uncovered (site, rows, why), reported together."""

    def __init__(self, name):
        self.name, self.items = name, []

    def add(self, site, rows, why):
        rows = tuple(rows)
        if rows:
            self.items.append(f"{site}: rows {rows}: {why}")

    def check(self):
        assert not self.items, f"{self.name}: uncovered\n  " + "\n  ".join(self.items)


def resident(kernel, grid):
    """Whether ``grid`` CTAs of ``kernel`` are co-resident on the faked VR200."""
    return grid <= VR200_BLOCKS_PER_SM[kernel] * VR200_SMS


def fake_resident_ctas(compiled, device_index, threads):
    """common.host.resident_ctas on the faked VR200."""
    return VR200_BLOCKS_PER_SM[compiled[0]] * VR200_SMS


def fake_check_residency(compiled, device_index, grid, threads, what):
    """common.host.check_residency on the faked VR200 (``compiled``: the fake
    compile's (kernel name, ...))."""
    if not resident(compiled[0], grid):
        per = VR200_BLOCKS_PER_SM[compiled[0]]
        raise RuntimeError(
            f"{what}: {grid} CTAs cannot be co-resident ({per} per SM x "
            f"{VR200_SMS} SMs)"
        )


def draft_config(sizes, cap, mns):
    compilation = SimpleNamespace(
        cudagraph_capture_sizes=list(sizes), max_cudagraph_capture_size=cap
    )
    return SimpleNamespace(
        compilation_config=compilation,
        scheduler_config=SimpleNamespace(max_num_seqs=mns),
    )


def drafter_rows(q, mns, cap, sizes):
    """(draft batches, query rows, context rows) as runtime.draft builds them."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime import draft

    steps = q - 1
    batches = draft.draft_batches(draft_config(sizes, cap, mns), steps)
    return batches, [b * steps for b in batches], [b * q for b in batches]


def prepared_context(monkeypatch, rows):
    """PreparedContext.__init__ for the wide context ``rows`` on the faked VR200: the
    real row split and residency decision, with fake compiles, symmetric memory and
    collectives (meta tensors)."""
    import torch

    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.draft import context

    monkeypatch.setattr(
        context,
        "compile_gather",
        lambda rows, shard, device, polled=False, ctas=None: (
            "CtxGather",
            rows,
            polled,
            ctas,
        ),
    )
    monkeypatch.setattr(context, "check_residency", fake_check_residency)
    monkeypatch.setattr(context, "resident_ctas", fake_resident_ctas)
    meta = lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device="meta")  # noqa: E731
    handle = SimpleNamespace(multicast_ptr=1)
    monkeypatch.setattr(
        context,
        "symm_mem",
        SimpleNamespace(empty=meta, rendezvous=lambda buffer, group: handle),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *a, **k: None)
    monkeypatch.setattr(context, "to_cute", lambda *a, **k: None)
    monkeypatch.setattr(context, "dynamic_view", lambda *a, **k: None)
    bf16 = torch.bfloat16
    weight = torch.empty((HIDDEN, 5 * HIDDEN), dtype=bf16, device="meta")
    norm = SimpleNamespace(
        weight=torch.empty(HIDDEN, dtype=bf16, device="meta"), variance_epsilon=1e-6
    )
    model = SimpleNamespace(
        context_proj=SimpleNamespace(weight=weight), context_norm=norm
    )
    return context.PreparedContext(model, None, 16, 0, rows)


def prepared_ar(monkeypatch, rows, two_shot, wide):
    """PreparedDraftAr.__init__ for the drafter all-reduce ``rows`` on the faked VR200:
    the real row split and the wide push sites' residency decision, with fake
    compiles, two-shot boxes, symmetric memory and collectives (meta tensors)."""
    import torch

    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.draft import (
        allreduce,
    )

    def rows_site(rows, sources, tp, cap, device, trigger=False, push=False, ctas=0):
        return ("ArNormRows", rows, sources, push, ctas)

    monkeypatch.setattr(allreduce, "compile_rows_site", rows_site)
    monkeypatch.setattr(allreduce, "compile_two_shot", lambda *a, **k: ("TwoShot",))
    monkeypatch.setattr(allreduce, "compile_ar_norm", lambda *a, **k: ("ArNorm",))
    monkeypatch.setattr(allreduce, "compile_ring_gather", lambda *a, **k: None)
    monkeypatch.setattr(allreduce, "architecture", lambda device: "sm_107a")
    monkeypatch.setattr(allreduce, "check_residency", fake_check_residency)
    monkeypatch.setattr(allreduce, "resident_ctas", fake_resident_ctas)
    box = SimpleNamespace(views=lambda parity: (None, None, None, 0))
    monkeypatch.setattr(allreduce, "TwoShot", lambda *a, **k: box)
    monkeypatch.setattr(torch.accelerator, "device_index", lambda i: nullcontext())
    meta = lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device="meta")  # noqa: E731
    handle = SimpleNamespace(multicast_ptr=1)
    monkeypatch.setattr(
        allreduce,
        "symm_mem",
        SimpleNamespace(empty=meta, rendezvous=lambda buffer, group: handle),
    )
    monkeypatch.setattr(allreduce, "to_cute", lambda *a, **k: None)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (10, 7))
    real_empty = torch.empty
    monkeypatch.setattr(
        allreduce.torch,
        "empty",
        lambda *a, **k: real_empty(*a, **{**k, "device": "meta"}),
    )
    device = SimpleNamespace(index=0)
    return allreduce.PreparedDraftAr(
        None, device, 16, 0, rows, VOCAB, two_shot, (), (), (), wide
    )


# ------------------------------------------------------------------ the drafter


@pytest.mark.parametrize("deployment", DEPLOYMENTS, ids=IDS)
def test_drafter_build_covers_every_captured_graph(monkeypatch, deployment):
    """runtime.draft.DraftFastPath.build's coverage decisions (strict on the target
    topology: each one is a ``refuse``) at every drafter graph the deployment
    captures: Markov sampling (FUSED_MARKOV, supports, the MarkovWide plan), the
    context step (FAST_CONTEXT, PreparedContext's co-resident row gather), the
    all-reduce sites (FAST_AR, PreparedDraftAr's rows) and the attention (FAST_DCP,
    DraftDcp.uncovered_rows, the drafter MlaQuery tile)."""
    pytest.importorskip("cutlass")
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.draft import (
        allreduce,
        attention,
        context,
        markov,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime import draft
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.shapes import target_topology

    name, q, mns, cap, sizes = deployment
    steps = q - 1
    assert target_topology(True, 16, 16, q)  # strict: every gap below refuses
    batches, query_rows, context_rows = drafter_rows(q, mns, cap, sizes)
    assert batches, name
    gaps = Gaps(name)
    top = max(batches)
    # Markov sampling (runtime/draft.py: "DSpark drafter Markov sampling")
    table = draft.batch_table(draft.closed(draft.FUSED_MARKOV.get(16, ""), top))
    gaps.add(
        "draft.py FUSED_MARKOV", [b for b in batches if not table.get(b)],
        "draft batches without a fused Markov entry",
    )  # fmt: skip
    shape = (VOCAB, DRAFT_RANK)
    if not markov.supports(shape, shape, 16, LIMITS_VR200):
        gaps.add("markov.supports", batches, "rank / vocab split / co-residency")
    prepared: Any = object.__new__(markov.PreparedMarkov)
    prepared.tp, prepared.markov_rank, prepared.limits = 16, DRAFT_RANK, LIMITS_VR200
    prepared.vocab_rank, prepared.plans = VOCAB // 16, {}
    prepared.wide_box = object() if top > markov.MAX_WIDTH else None
    for sampled in (False, True):
        gaps.add(
            f"markov.PreparedMarkov.covers (sampled={sampled})",
            [b for b in batches if not prepared.covers(b, sampled)],
            "no MarkovWide launch fits SM107 shared memory at Markov rank 512",
        )
    # Context step (runtime/draft.py:582 / :603, cutedsl/draft/context.py)
    spec = draft.closed(draft.FAST_CONTEXT.get(16, ""), top * q)
    gaps.add(
        "draft.py FAST_CONTEXT",
        [r for r in context_rows if not draft.covered(spec, r, steps)],
        "context rows the table does not list (stock replicated context step)",
    )
    wide = [r for r in context_rows if r > context.CAP]
    if wide:
        ctx = prepared_context(monkeypatch, wide)
        assert set(ctx.wide) == set(wide), name
        gaps.add(
            "context.py PreparedContext.gathers",
            [r for r in wide if r not in ctx.gathers],
            "wide context rows without a row gather",
        )
        # One CTA per row while that grid is co-resident, the persistent one-wave
        # gather above (its grid checked co-resident by check_residency).
        limit = VR200_BLOCKS_PER_SM["CtxGather"] * VR200_SMS
        for r in wide:
            kernel = ctx.gathers.get(r)
            grid = r if kernel is None or kernel[3] is None else kernel[3]
            if kernel is not None and kernel[3] is None and r > limit:
                gaps.add(
                    "context.py CtxGather (one CTA per row)", [r],
                    f"{r} CTAs not co-resident ({limit}) and not persistent",
                )  # fmt: skip
            if grid > limit:
                gaps.add("context.py CtxGather grid", [r], f"{grid} CTAs > {limit}")
    # All-reduce sites (runtime/draft.py:631 / :652, cutedsl/draft/allreduce.py)
    top_rows = top * steps
    ar = draft.closed(draft.FAST_AR.get(16, ""), top_rows)
    ar_rows = [r for r in query_rows if draft.covered(ar, r, steps)]
    gaps.add(
        "draft.py FAST_AR", sorted(set(query_rows) - set(ar_rows)),
        "query rows without Lamport sites",
    )  # fmt: skip
    wide_ar = {r for r in ar_rows if draft.covered(ar, r, steps) == 3}
    cap_ar = allreduce.WIDE_CAP or max(ar_rows, default=0)
    served = {
        r
        for r in ar_rows
        if r <= min(allreduce.CAP, allreduce.TWO_CAP) or (r in wide_ar and r <= cap_ar)
    }  # (PreparedDraftAr.rows)
    gaps.add(
        "allreduce.py PreparedDraftAr.rows (draft.py:652)",
        sorted(set(query_rows) - served),
        "query rows above the 64-row boxes that are not ArNormRows rows",
    )
    # ArNormRows push sites: each CTA pushes its partial row's chunks, reduces the
    # chunks this rank owns from every rank's same-row CTA, then polls the gathered
    # row (cross-rank polls inside one grid): one CTA per row while co-resident, the
    # persistent one-wave site above (PreparedDraftAr.persistent_site).
    two = {r for r in ar_rows if draft.covered(ar, r, steps) == 2}
    pa = prepared_ar(monkeypatch, ar_rows, two, wide_ar)
    gaps.add(
        "allreduce.py PreparedDraftAr.rows (built)",
        sorted(set(query_rows) - set(pa.rows)),
        "query rows the built PreparedDraftAr does not serve",
    )
    for r in sorted(pa.wide_rows):
        _, rows, _, push, ctas = pa.reduce[r]
        grid = ctas or rows
        if not push or not resident("ArNormRows", grid):
            gaps.add(
                "allreduce.py ArNormRows push site", [r],
                f"grid {grid} not co-resident (persistent ctas={ctas})",
            )  # fmt: skip
    # Attention (runtime/draft.py:721-:772, cutedsl/draft/attention.py)
    dcp_spec = draft.closed(draft.FAST_DCP.get(16, ""), top_rows)
    dcp_rows = [r for r in query_rows if draft.covered(dcp_spec, r, steps)]
    gaps.add(
        "draft.py FAST_DCP", sorted(set(query_rows) - set(dcp_rows)),
        "query rows on the stock DCP attention",
    )  # fmt: skip
    if not attention.query_heads(DRAFT_HEADS):
        gaps.add("attention.query_heads", query_rows, "no drafter MlaQuery tile")
    dropped = attention.uncovered_rows(dcp_rows, steps, 64, 16, VR200_SMS)
    for why in sorted(set(dropped.values())):
        gaps.add(
            "attention.uncovered_rows",
            [r for r, w in dropped.items() if w == why],
            why,
        )
    gaps.check()


@pytest.mark.parametrize("q", [4, 8])
def test_drafter_dcp_value_one_wave(q):
    """DraftDcp's wide rows: compile_value's DcpValue (6 heads, DCP16, the drafter's
    q = draft length, validity box; fold 0 or an LSE split) within one wave of the
    VR200's SMs at every drafter query row up to 1024 (mla.dcp.compile_value's
    check_residency drops the row otherwise: DraftDcp.dropped -> draft.py:772)."""
    pytest.importorskip("cutlass")
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.mla.dcp import DcpValue

    steps = q - 1
    gaps = Gaps(f"q{q}")
    for fold in (0, 2, 4, 8, 16, 32):
        over = []
        for rows in range(65 // steps * steps + steps, TOP_ROWS + 1, steps):
            value = DcpValue(
                DRAFT_HEADS, 16, rows, VR200_SMS, steps, fold, lse_split=fold > 0,
                valid_box=True,
            )  # fmt: skip
            grid = value.grid[0] * value.grid[1] * value.grid[2]
            if not resident("DcpValue", grid):
                over.append(rows)
        gaps.add(f"mla/dcp.py DcpValue fold {fold}", over, "grid above one wave")
    gaps.check()


# ------------------------------------------------------------------- the target


class Built(Exception):
    """Stops SpecializedDecode.__init__ once the capacity and wide rows are set."""


@pytest.mark.parametrize("deployment", DEPLOYMENTS, ids=IDS)
def test_target_step_covers_every_captured_row(monkeypatch, deployment):
    """The target decode at every captured verify row: SpecializedDecode's prepared
    batches + wide rows hold every captured row (decode.py:578, the whole-step stock
    forward), and each family's ``choose`` admits every wide row (decode.py:509-517,
    check_wide: seams S1 / S2-DSA / S3 / S4)."""
    pytest.importorskip("cutlass")
    import torch

    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.wide import (
        choose as dsa_choose,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.kda.prepared import (
        STOCK_ROWS as KDA_STOCK_ROWS,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.kda.prepared import (
        choose as kda_choose,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.moe.prepared import (
        choose as moe_choose,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.oproj.prepared import (  # noqa: E501
        choose as oproj_choose,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime import DecodeConfig
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime import (
        decode as decode_module,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.shapes import (
        Context,
        Path,
        Shape,
        captured_rows,
        rows_policy,
    )

    name, q, mns, cap, sizes = deployment
    captured = captured_rows(sizes, q, min(mns * q, cap))
    group = SimpleNamespace(world_size=16, rank_in_group=0, device_group=None)
    platform = SimpleNamespace(current_device=lambda: "cuda:0")
    device = SimpleNamespace(major=10, minor=7, multi_processor_count=VR200_SMS)
    monkeypatch.setattr(decode_module, "get_tp_group", lambda: group)
    monkeypatch.setattr(decode_module, "current_platform", platform)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: device)
    monkeypatch.setattr(decode_module, "Context", MagicMock(side_effect=Built))
    text = SimpleNamespace(dsa_config=SimpleNamespace())
    backbone = SimpleNamespace(config=text, use_sequence_parallel=False)
    config = DecodeConfig(
        max_context=1 << 20, mla_core="auto", dsa_core="auto", dcp=16, q=q,
        capture_sizes=tuple(sizes), captured=captured,
    )  # fmt: skip
    decode: Any = object.__new__(decode_module.SpecializedDecode)
    with pytest.raises(Built):
        decode_module.SpecializedDecode.__init__(decode, backbone, config)
    gaps = Gaps(name)
    gaps.add(
        "decode.py wide_gate", captured if decode.wide_off else (), str(decode.wide_off)
    )
    gaps.add(
        "decode.py plan_coverage (:578)",
        [m for m in captured if m not in decode.batches or m % q],
        "captured rows on the whole-step stock forward",
    )
    if captured and decode.capacity < max(captured):
        gaps.add(
            "decode.py capacity", [max(captured)], f"buffers hold {decode.capacity}"
        )
    ctx = Context(
        arch="rubin", tp=16, rank=0, dcp=16, sms=VR200_SMS, q=q, max_rows=max(captured),
        max_context=1 << 20, bf16_state=False,
    )  # fmt: skip
    wide = [Shape(m, q) for m in decode.wide]
    gaps.add(
        "kda/prepared.py STOCK_ROWS (seam S1)",
        [s.rows for s in wide if rows_policy(KDA_STOCK_ROWS, s, default=0)],
        "stock KDA core",
    )
    gaps.add(
        "kda/prepared.py choose", [s.rows for s in wide if kda_choose(s, ctx) is None],
        "no KDA path",
    )  # fmt: skip
    gaps.add(
        "dsa/wide.py DSA_WIDE (seam S2-DSA)",
        [s.rows for s in wide if dsa_choose(s, ctx) is None],
        "stock DSA attention",
    )
    gaps.add(
        "oproj/prepared.py WIDE_TWOSHOT (seam S3)",
        [s.rows for s in wide if oproj_choose(s, ctx).path is Path.STOCK],
        "stock o-projection + all-reduce",
    )
    gaps.add(
        "moe/prepared.py WIDE_TAIL (seam S4)",
        [s.rows for s in wide if moe_choose(s, ctx).path is Path.STOCK],
        "stock MoE layer",
    )
    gaps.check()


@pytest.mark.parametrize("q", [4, 8])
def test_target_dsa_wide_grids_resident(q):
    """WideDsa at every wide verify row up to 1024: its MlaQuery's polling clusters
    (dsa/wide.py check_query_residency: the whole grid up to WIDE_MAX_ROWS, one row
    tile's under tile_poll) within the VR200's 22 resident clusters, and both DcpValue
    schedules ("proj" and the GB300-schedule latent merge) within one wave."""
    pytest.importorskip("cutlass")
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa import dcp_value
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.wide import (
        WIDE_MAX_ROWS,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.mla.dcp import (
        GB300_SMS,
        DcpValue,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.mla.query import MlaQuery

    gaps = Gaps(f"q{q}")
    query, proj, latent = [], [], []
    for batch in range(64 + q, TOP_ROWS + 1, q):
        mla_query = MlaQuery(
            6, batch, dsa=True, fp8=True, dcp=16, tile_poll=batch > WIDE_MAX_ROWS
        )
        grid = mla_query.grid
        need = grid[1] * (1 if mla_query.tile_poll else grid[2])
        if need > VR200_QUERY_CLUSTERS:
            query.append(batch)
        g = dcp_value.DcpValue(6, 16, batch, VR200_SMS, partials="proj").grid
        if not resident("DcpValue", g[0] * g[1] * g[2]):
            proj.append(batch)
        g = DcpValue(6, 16, batch, GB300_SMS, q, resident_sms=VR200_SMS).grid
        if not resident("DcpValue", g[0] * g[1] * g[2]):
            latent.append(batch)
    gaps.add("dsa/wide.py check_query_residency", query, "MlaQuery clusters")
    gaps.add("dsa/wide.py check_proj_residency", proj, "DcpValue proj grid")
    gaps.add("dsa/wide.py check_value_residency", latent, "DcpValue latent grid")
    gaps.check()
