# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP specialized decode on CPU (C512 decode role: TP4 = DCP4 over DP4, EP16
MegaMoE, q4; ``runtime.sp_attention``): the startup of the real ``SpStep`` (and of
``SpecializedDecode`` under ``sequence_parallel``) over stand-in providers, for every
captured R = 4..512 (n = 1..128 requests per DP group, 0.25..32 per GPU):

* the coverage table: every R has the nine specialized families (ag, dsa_in, dsa,
  kda_in, kda, oproj_rs, moe_dense: the MegaMoE layers' dense GEMMs on the n = R / 4
  shard rows, sp_boundary: the fused AttnRes + SP AG / RS, final_gather), every other family is a declared stock module, no R is missing or
  stock, one log line per R;
* the collective build order (SpComm, the DSA DcpBox and scratch, RecoverSSM, then
  the blocks in layer order) and the mailbox parities (SP blocks by ordinal, DSA
  cores by DSA ordinal);
* no silent fallback: each injected unsupported layer, row set, provider gap or
  drafter table raises ``shapes.Unsupported`` naming the family, the R and the cause;
* ``plan`` serves every captured R in a FULL capture and raises outside them;
* ``SpecializedDecode`` delegates plan / run / prepare_kv / reload to the SP step and
  never builds the TP8 / TP16 providers.

The providers are stand-ins with the contract's interfaces (no collective, no kernel):
their own build-time checks run in the GPU tests (test_specialized_decode_sp_*_gpu.py,
tests/distributed/test_kimi_k3_sdec_sp_*.py)."""

import contextlib
import itertools
import math
import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass")

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode import shapes  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import (  # noqa: E402
    DecodeConfig,
    decode,
    draft,
    sp_attention,
)

ROWS = tuple(range(4, 513, 4))  # C512: max_num_seqs 128 x q4, 128 FULL graphs
Q, TP, HEADS, WIDTH = 4, 4, 24, 7168
DSA_PITCH = 2112 + HEADS * 128 + 128 + 32  # [q_a | kv_a | gate | wk | wv]: 5344
KDA_PITCH = 4 * HEADS * 128 + 128 + HEADS + 8  # [q | k | v | g | f_a | beta | pad]
# A K3.1-like layer stack: a dense first layer, then MoE; KDA and DSA interleaved,
# an odd count of SP blocks (the step's last and next step's first share a parity).
KINDS = ("kda", "kda", "dsa", "kda", "dsa", "kda", "kda")
SPECIALIZED = {"ag", "dsa_in", "dsa", "kda_in", "kda", "oproj_rs", "moe_dense"}
SPECIALIZED |= {"sp_boundary", "final_gather"}  # fused AttnRes + SP AG / RS, final AG
SPECIALIZED |= {"dense"}  # the layer-0 dense MLP (SpDenseMlp) at <= 128 shard rows
DECLARED = {"embed", "attn_res", "moe", "lm_head"}
DECLARED |= {"sampler", "drafter", "verify"}


def meta(*shape):
    return torch.empty(shape, dtype=torch.bfloat16, device="meta")


class Events(list):
    def add(self, *event):
        self.append(event)


class Provider:
    """A stand-in provider: ``choice`` GENERIC at the rows it prepared, raising
    ``Unsupported`` (with the int R, as the providers do) elsewhere."""

    family = "provider"

    def __init__(self, rows, gaps=()):
        self.rows = frozenset(rows) - frozenset(gaps)

    def choice(self, r):
        if r not in self.rows:
            raise shapes.Unsupported(self.family, r, "not prepared")
        return shapes.Choice(shapes.Path.GENERIC, note=f"stand-in {self.family}")


class Fakes:
    """The providers ``sp_attention`` builds, recording the build order; ``gaps``:
    family -> rows its stand-in does not serve; ``pitch``: family -> a core pitch
    that differs from the in-projection's."""

    def __init__(self, events, gaps=None, pitch=None, raises=None):
        self.events, self.gaps = events, gaps or {}
        self.pitch, self.raises = pitch or {}, raises or {}
        fakes = self

        class Comm(Provider):
            family = "ag"

            def __init__(self, group, device, tp, rank, *, width, rows_cap):
                super().__init__(range(tp, rows_cap + 1, tp), fakes.gaps.get("ag", ()))
                fakes.events.add("SpComm", tp, rank, width, rows_cap)
                self.tp, self.rank, self.width, self.parities = tp, rank, width, 2
                self.gather = [meta(rows_cap, width) for _ in range(2)]

            def nbytes(self):
                return 3 * self.gather[0].numel() * 2

        class FinalGather:
            def __init__(self, group, device, tp, rank, *, width, rows_cap):
                fakes.events.add("SpFinalGather", tp, rank, width, rows_cap)
                self.width, self.rows_cap, self.prepared = width, rows_cap, []
                self.pushes = []

            def prepare(self, rows, cols=(0,), widths=None):
                if widths is None:
                    self.prepared.append(rows)

            def prepare_push(self, rows, col, width):
                assert 0 < col < self.width and width < self.width, (col, width)
                self.pushes.append((rows, col))

        def boundary(decoder, comm, block, rows, mlp=None, delta_parts=()):
            assert block.oproj.comm is comm and tuple(rows) == tuple(ROWS), rows
            assert delta_parts == (), delta_parts  # stand-in MLPs have no split partials
            assert (mlp is None) == (decoder.layer_idx == 0), decoder.layer_idx
            fakes.events.add("SpBoundary", decoder.layer_idx, block.parity)
            return SimpleNamespace(layer=decoder.layer_idx, comm=comm)

        class Projection(Provider):
            def __init__(self, weight, rows, tactics, *, family):
                self.family = family
                assert set(tactics) == set(rows), family
                super().__init__(rows, fakes.gaps.get(family, ()))
                self.weight, self.operands = weight, []
                fakes.events.add("DecodeProjection", family, tuple(weight.shape))

            def check_operands(self, x, out):
                assert x.shape[1] == self.weight.shape[1], (x.shape, self.weight.shape)
                assert out.shape[1] == self.weight.shape[0], (out.shape, self.weight)
                self.operands.append((tuple(x.shape), tuple(out.shape)))

        class DsaCore(Provider):
            family = "dsa"

            def __init__(self, attention, rows, scratch, *, box, parity, ctx):
                if "dsa" in fakes.raises:
                    raise fakes.raises["dsa"]
                super().__init__(rows, fakes.gaps.get("dsa", ()))
                assert box == "box" and scratch == "scratch"
                self.attention, self.parity, self.box = attention, parity, Box()
                self.pitch = fakes.pitch.get("dsa", DSA_PITCH)
                self.caches = []
                fakes.events.add("SpDsaCore", attention.index, parity)

            def prepare_cache(self, kv):
                self.caches.append(kv)
                return self

        class KdaCore(Provider):
            family = "kda"

            def __init__(self, attention, rows, *, records, ctx, mx_out=False):
                if "kda" in fakes.raises:
                    raise fakes.raises["kda"]
                super().__init__(rows, fakes.gaps.get("kda", ()))
                assert records == "triton", records
                self.attention = attention
                self.pitch = fakes.pitch.get(
                    "kda", attention.in_proj_qkvgfab.weight.shape[0]
                )
                self.caches, self.reloads = [], 0
                fakes.events.add("SpKdaCore", attention.index)

            def prepare_caches(self, caches):
                self.caches.append(caches)
                return self

            def process_weights_after_loading(self):
                self.reloads += 1

        class Oproj(Provider):
            family = "oproj_rs"

            def __init__(self, weight, comm, rows, ctx):
                super().__init__(rows, fakes.gaps.get("oproj_rs", ()))
                assert tuple(weight.shape) == (WIDTH, HEADS * 128)
                self.comm = comm
                fakes.events.add("SpOproj")

        class Mlp:
            def __init__(self, mlp, rows, ctx, index):
                assert mlp.use_mega_moe and ctx.sequence_parallel
                assert tuple(rows) == tuple(sorted({r // TP for r in ROWS})), rows  # n = R / 4
                gaps = fakes.gaps.get("moe_dense", ())
                self.providers = {}
                for f in ("latent_down", "shared_gu", "shared_down", "latent_up"):
                    self.providers[f] = Provider(rows, gaps)
                    self.providers[f].family = f
                self.parts = {}  # SpMlp.parts: no latent up split-partial rows
                fakes.events.add("SpMlp", index)

            def choices(self, n):
                return {f: p.choice(n) for f, p in self.providers.items()}

        class Dense(Provider):
            family = "dense"

            def __init__(self, mlp, rows, device, index):
                assert index == 0 and not mlp.use_mega_moe, index
                assert tuple(rows) == tuple(sorted({r // TP for r in ROWS})), rows  # n = R / 4
                super().__init__(rows, fakes.gaps.get("dense", ()))

            def serves(self, n):
                return n in self.rows

        def tactics(family, n, k, rows, ctx):
            assert ctx.sequence_parallel and ctx.tp == TP, ctx
            return dict.fromkeys(rows, ("cublas", None))

        def box(group, device, rank, *, heads, cap, interleave):
            fakes.events.add("sp_dsa_box", heads, cap, interleave)
            return "box"

        def scratch(device, cap):
            fakes.events.add("sp_dsa_scratch", cap)
            return "scratch"

        def recoverssm(layers, ctx, *, defer=True):
            fakes.events.add("prepare_recoverssm", defer)
            return "triton"

        def bf16(linear):
            if linear.weight.dtype != torch.bfloat16:
                raise ValueError(f"needs BF16 projections, got {linear.weight.dtype}")
            return linear.weight

        self.patch = dict(
            SpComm=Comm,
            SpFinalGather=FinalGather,
            sp_boundary=boundary,
            DecodeProjection=Projection,
            SpDsaCore=DsaCore,
            SpKdaCore=KdaCore,
            SpOproj=Oproj,
            SpMlp=Mlp,
            SpDenseMlp=Dense,
            sp_projection_tactics=tactics,
            sp_dsa_box=box,
            sp_dsa_scratch=scratch,
            prepare_recoverssm=recoverssm,
            bf16_weight=bf16,
            pack_dsa_projection=lambda attention: attention.packed,
        )


class Box:
    def __init__(self):
        self.gathered = 0

    def gather_heads(self, w_uv):
        """Every rank's heads of ``w_uv`` (TP4), as DsaBox.gather_heads."""
        self.gathered += 1
        return torch.empty((TP * w_uv.shape[0], *w_uv.shape[1:]), device="meta")


def linear(*shape, dtype=torch.bfloat16, reduce=False):
    return SimpleNamespace(
        weight=torch.empty(shape, dtype=dtype, device="meta"), reduce_results=reduce
    )


def attention(kind, index):
    """A stock DSA / KDA module's fields the SP step reads."""
    if kind == "kda":
        return SimpleNamespace(
            index=index,
            in_proj_qkvgfab=linear(KDA_PITCH, WIDTH),
            o_proj=linear(WIDTH, HEADS * 128),
            kv_cache=[meta(2, 3 * HEADS * 128, Q + 2), meta(2, HEADS, 128, 128)],
        )
    layer = SimpleNamespace(
        layer_name=f"model.layers.{index}.self_attn.mla_attn",
        kv_cache=meta(8, 64, 576),
        W_UV=meta(HEADS, 512, 128),
        W_UV_dcp_gathered=[meta(TP * HEADS, 512, 128)],
    )
    indexer = SimpleNamespace(k_cache=SimpleNamespace(prefix=f"idx{index}"))
    return SimpleNamespace(
        index=index,
        packed=meta(DSA_PITCH, WIDTH),
        dsa_indexer=indexer,
        mla_attn=SimpleNamespace(mla_attn=layer),
        o_proj=linear(WIDTH, HEADS * 128),
    )


def decoder(kind, index, *, tokens=128):
    moe = index > 0
    mlp = SimpleNamespace(
        use_mega_moe=moe,
        experts=SimpleNamespace(max_num_tokens=tokens),
    )
    return SimpleNamespace(
        layer_idx=index,
        self_attn=attention(kind, index),
        use_sequence_parallel=True,
        is_moe_layer=moe,
        mlp=mlp,
    )


def context(**overrides):
    fields = dict(
        arch="rubin", tp=TP, rank=0, dcp=TP, sms=148, q=Q, max_rows=512,
        max_context=1 << 20, bf16_state=False, dp=4, ep=16,
        sequence_parallel=True, heads_per_rank=HEADS,
    )  # fmt: skip
    fields.update(overrides)
    return shapes.Context(**fields)


def vllm_config(method="synthetic"):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        speculative_config=SimpleNamespace(rejection_sample_method=method),
    )


@pytest.fixture
def fakes(monkeypatch):
    """Patch ``sp_attention``'s providers with recording stand-ins; returns a
    function (gaps, pitch, raises, method) -> (Fakes, Events)."""

    def install(gaps=None, pitch=None, raises=None, method="synthetic"):
        events = Events()
        made = Fakes(events, gaps, pitch, raises)
        for name, value in made.patch.items():
            monkeypatch.setattr(sp_attention, name, value)
        config = vllm_config(method)
        monkeypatch.setattr(sp_attention, "get_current_vllm_config", lambda: config)
        monkeypatch.setattr(
            torch.accelerator, "device_index", lambda index: contextlib.nullcontext()
        )
        # the early aux gather's side stream (meta device: no CUDA stream)
        monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: SimpleNamespace())
        return made, events

    return install


def runtime(layers=None, rows=ROWS, ctx=None, **backbone):
    stock = dict(
        use_sequence_parallel=True,
        use_attn_res=True,
        aux_hidden_state_layers=(2, 5),
        embed_tokens=None,  # no vocab shard: SpEmbedScatter declines (stock embedding)
        _aux_attn_res_stream=False,
        start_layer=0,
    )
    stock.update(backbone)
    if layers is None:
        layers = [decoder(kind, i) for i, kind in enumerate(KINDS)]
    return SimpleNamespace(
        context=ctx or context(),
        device=torch.device("meta"),
        backbone=SimpleNamespace(**stock),
        captured=rows,
        layers=tuple(layers),
        group="tp-group",
    )


def test_coverage_every_captured_row(fakes):
    """Every R = 4..512 specialized: the six SP families from the blocks and the
    mailboxes, moe_dense from the SpMlps, the rest declared stock modules; one log
    line per R."""
    fakes()
    step = sp_attention.SpStep(runtime())
    table = step.coverage()
    assert step.rows == ROWS and sorted(table.plan) == list(ROWS)
    table.assert_complete(ROWS)
    for r in ROWS:
        choices = table.plan[r]
        assert set(choices) == SPECIALIZED | DECLARED, (r, sorted(choices))
        for family in SPECIALIZED:
            assert choices[family].path in (shapes.Path.GENERIC, shapes.Path.TUNED)
        for family in DECLARED:
            assert choices[family].path is shapes.Path.DECLARED, (r, family)
    lines = table.lines()
    assert len(lines) == len(ROWS)
    assert lines[0].startswith("M4(q4): ") and lines[-1].startswith("M512(q4): ")


def test_every_block_serves_every_row(fakes):
    """Each layer's in-projection, core and o_proj prepared every captured R; the
    shard cap is 128 rows; the in-projection accepted both gather parities."""
    fakes()
    step = sp_attention.SpStep(runtime())
    assert step.shard_cap == 128 and step.records == "triton"
    assert sorted(step.blocks) == list(range(len(KINDS)))
    for index, block in step.blocks.items():
        assert block.kind == KINDS[index]
        for provider in (block.projection, block.core, block.oproj):
            assert provider.rows == frozenset(ROWS), (index, provider.family)
        pitch = DSA_PITCH if block.kind == "dsa" else KDA_PITCH
        assert block.projection.operands == [((512, WIDTH), (512, pitch))] * 2
        assert block.oproj.comm is step.comm and block.comm is step.comm


def test_collective_build_order_and_parities(fakes):
    """Collective allocations in the same order on every rank: SpComm, the final
    all-gather box ([hidden | 2 aux] rows), the DSA box and scratch, RecoverSSM
    (deferred), then each layer's block in layer order, then each layer's fused
    boundaries. SP mailbox parity by block ordinal, DSA box parity by DSA ordinal."""
    _, events = fakes()
    step = sp_attention.SpStep(runtime())
    names = [e[0] for e in events]
    assert names[:5] == [
        "SpComm", "SpFinalGather", "sp_dsa_box", "sp_dsa_scratch", "prepare_recoverssm"
    ]
    assert events[0] == ("SpComm", TP, 0, WIDTH, 512)
    assert events[1] == ("SpFinalGather", TP, 0, 3 * WIDTH, 512)
    assert step.final_gather.prepared == list(ROWS)
    # prefix-only capture: aux layers 2 and 5 push their columns early, per row
    assert sorted(step.final_gather.pushes) == sorted(
        (r, WIDTH * (1 + k)) for r in ROWS for k in range(2)
    )
    assert events[2] == ("sp_dsa_box", HEADS, 512, 1)
    assert events[4] == ("prepare_recoverssm", True)  # deferred (the cores fold)
    cores = [e for e in events if e[0] in ("SpDsaCore", "SpKdaCore")]
    assert [e[1] for e in cores] == list(range(len(KINDS)))
    dsa = [e[2] for e in cores if e[0] == "SpDsaCore"]
    assert dsa == [i % 2 for i in range(len(dsa))]
    # Per block: in-projection, core, o_proj (none built ahead of its layer); then
    # each MegaMoE layer's SpMlp (no collective) in layer order.
    # then each layer's fused boundaries (they take the SpMlp's MXFP8 input).
    mlps = [e for e in events if e[0] == "SpMlp"]
    assert mlps == [("SpMlp", i) for i in range(1, len(KINDS))]
    bounds = [e for e in events if e[0] == "SpBoundary"]
    assert bounds == [("SpBoundary", i, i % 2) for i in range(len(KINDS))]
    tail = len(mlps) + len(bounds)
    assert names[-tail:] == ["SpMlp"] * len(mlps) + ["SpBoundary"] * len(bounds)
    assert sorted(step.bounds) == list(range(len(KINDS)))
    per_block = [e[0] for e in events[5:] if e[0] not in ("SpMlp", "SpBoundary")]
    expected = list(
        itertools.chain.from_iterable(
            ("DecodeProjection", "SpDsaCore" if k == "dsa" else "SpKdaCore", "SpOproj")
            for k in KINDS
        )
    )
    assert per_block == expected
    assert [b.parity for b in step.blocks.values()] == [
        i % 2 for i in range(len(KINDS))
    ]


def test_blocks_share_scratch(fakes):
    """One projected scratch per kind (its pitch) and one attention output scratch
    (24 x 128) for all blocks: they run one at a time."""
    fakes()
    step = sp_attention.SpStep(runtime())
    blocks = list(step.blocks.values())
    for kind, pitch in (("dsa", DSA_PITCH), ("kda", KDA_PITCH)):
        mine = [b for b in blocks if b.kind == kind]
        assert all(b.projected is mine[0].projected for b in mine)
        assert tuple(mine[0].projected.shape) == (512, pitch)
    assert all(b.attn_out is blocks[0].attn_out for b in blocks)
    assert tuple(blocks[0].attn_out.shape) == (512, HEADS * 128)
    assert step.dense and step.moe


def test_verify_declared_only_under_synthetic_rejection(fakes):
    """Synthetic rejection (C512): verify is a declared stock module; otherwise the
    table names VerifyArgmax (which covers TP4 or raises when the model builds it)."""
    fakes(method="probabilistic")
    table = sp_attention.SpStep(runtime()).coverage()
    table.assert_complete(ROWS)
    for r in (4, 324, 512):
        assert table.plan[r]["verify"].path is shapes.Path.GENERIC
        assert "VerifyArgmax" in table.plan[r]["verify"].note


def set_layer(layers, index, **fields):
    for name, value in fields.items():
        target = layers[index]
        *path, last = name.split(".")
        for part in path:
            target = getattr(target, part)
        setattr(target, last, value)
    return layers


def stack():
    return [decoder(kind, i) for i, kind in enumerate(KINDS)]


BUILD_FAILURES = {
    # injected stock layer / topology -> the cause the startup names
    "megamoe_tokens": (
        dict(layers=set_layer(stack(), 3, **{"mlp.experts.max_num_tokens": 64})),
        "moe", r"layer 3: MegaMoE max_num_tokens 64 < 128 shard rows",
    ),
    "not_megamoe": (
        dict(layers=set_layer(stack(), 2, **{"mlp.use_mega_moe": False})),
        "moe", r"layer 2: not MegaMoE on the SP shard",
    ),
    "reducing_oproj": (
        dict(layers=set_layer(stack(), 5, **{"self_attn.o_proj.reduce_results": True})),
        "oproj_rs", r"layer 5: a reducing stock o_proj",
    ),
    "not_sp_layer": (
        dict(layers=set_layer(stack(), 1, use_sequence_parallel=False)),
        "sp_step", r"layer 1: not a sequence-parallel layer",
    ),
    "mla_layer": (
        dict(layers=set_layer(stack(), 2, self_attn=SimpleNamespace(o_proj=None))),
        "mla", r"layer 2: mla attention \(DSA / KDA only\)",
    ),
    "quantized_projection": (
        dict(
            layers=set_layer(
                stack(), 3, **{"self_attn.in_proj_qkvgfab": linear(
                    KDA_PITCH, WIDTH, dtype=torch.float8_e4m3fn
                )},
            )
        ),
        "kda", r"layer 3: needs BF16 projections",
    ),
    "half_requests": (
        dict(rows=(4, 6, 8)), "sp_step", r"M6\(q4\): not whole q4 requests"
    ),
    "no_graphs": (dict(rows=()), "sp_step", "no captured FULL decode graphs"),
    "dcp2": (
        dict(ctx=context(dcp=2)), "sp_step", r"TP4 / DCP2, sequence parallel True"
    ),
    "not_sp": (
        dict(ctx=context(sequence_parallel=False)), "sp_step", "sequence parallel False"
    ),
    "backbone_not_sp": (
        dict(use_sequence_parallel=False),
        "sp_step", "the backbone is not sequence-parallel",
    ),
    "no_attn_res": (
        dict(use_attn_res=False), "attn_res", "the SP step runs AttnRes models only"
    ),
}  # fmt: skip


@pytest.mark.parametrize("case", list(BUILD_FAILURES))
def test_startup_raises_naming_the_cause(fakes, case):
    """No silent fallback: an injected layer or topology the SP step cannot serve
    raises at build with the family, the shape and the cause."""
    fakes()
    kwargs, family, cause = BUILD_FAILURES[case]
    with pytest.raises(shapes.Unsupported, match=cause) as error:
        sp_attention.SpStep(runtime(**kwargs))
    assert error.value.family == family, str(error.value)


@pytest.mark.parametrize("family", ["ag", "dsa_in", "dsa", "kda_in", "kda", "oproj_rs"])
@pytest.mark.parametrize("gap", [4, 324, 512])
def test_a_provider_gap_raises_naming_the_row(fakes, family, gap):
    """A provider that does not serve one captured R fails the startup coverage at
    that R (never a stock R)."""
    fakes(gaps={family: (gap,)})
    with pytest.raises(shapes.Unsupported, match=rf"{family} at {gap}: not prepared"):
        sp_attention.SpStep(runtime())


@pytest.mark.parametrize("gap", [1, 81, 128])
def test_an_mlp_gemm_gap_raises_naming_the_shard_row(fakes, gap):
    """The MegaMoE layers' dense GEMMs serve the n = R / 4 shard rows: a shard row
    they do not serve fails the startup coverage (never a stock MLP)."""
    fakes(gaps={"moe_dense": (gap,)})
    with pytest.raises(shapes.Unsupported, match=rf"latent_down at {gap}: not"):
        sp_attention.SpStep(runtime())


@pytest.mark.parametrize("family", ["dsa", "kda"])
def test_core_build_failures_propagate(fakes, family):
    """A core that refuses the layer at build (e.g. residency, layout) stops the
    startup with its own cause; a core pitch other than the in-projection's too."""
    cause = shapes.Unsupported(family, shapes.Shape(324, Q), "grid not co-resident")
    fakes(raises={family: cause})
    with pytest.raises(shapes.Unsupported, match="grid not co-resident"):
        sp_attention.SpStep(runtime())
    fakes(pitch={family: 5000})
    pitch = DSA_PITCH if family == "dsa" else KDA_PITCH
    with pytest.raises(
        shapes.Unsupported, match=f"core pitch 5000 != in-projection N {pitch}"
    ):
        sp_attention.SpStep(runtime())


def test_drafter_table_with_a_tp4_entry_raises(fakes, monkeypatch):
    """The DSpark drafter is declared stock at TP4 but for its attention
    (SP_DRAFTER_TABLES: FAST_DCP): another drafter table gaining a TP4 entry (a
    specialized drafter piece) stops the startup instead of silently contradicting
    the declaration."""
    fakes()
    tables = sp_attention.check_stock_drafter(TP)
    assert tables and all(name.isupper() for name in tables)
    assert sp_attention.SP_DRAFTER_TABLES == {"FAST_DCP"} and TP in draft.FAST_DCP
    name = next(n for n in tables if n not in sp_attention.SP_DRAFTER_TABLES)
    monkeypatch.setattr(draft, name, {**getattr(draft, name), TP: "x"})
    with pytest.raises(shapes.Unsupported, match=rf"\['{name}'\] have TP4 entries"):
        sp_attention.SpStep(runtime())


def test_drafter_attention_without_its_tp4_entry_raises(fakes, monkeypatch):
    """FAST_DCP without its TP4 entry would leave the drafter's attention stock under
    a declaration that says it is specialized: refused at startup."""
    fakes()
    fast_dcp = {k: v for k, v in draft.FAST_DCP.items() if k != TP}
    monkeypatch.setattr(draft, "FAST_DCP", fast_dcp)
    with pytest.raises(shapes.Unsupported, match=r"\['FAST_DCP'\] have no TP4 entry"):
        sp_attention.SpStep(runtime())


def metadata(kind, rows):
    """A layer's uniform q4 verify metadata at ``rows`` (decode.is_uniform_verify)."""
    n = rows // Q
    if kind == "kda":
        return SimpleNamespace(
            num_prefills=0, num_decodes=0, num_spec_decodes=n,
            num_spec_decode_tokens=rows,
            spec_state_indices_tensor=torch.empty((n, 1), device="meta"),
        )  # fmt: skip
    return SimpleNamespace(num_prefills=0, num_decodes=n, num_decode_tokens=rows)


def capture(monkeypatch, step, rows, mode=CUDAGraphMode.FULL, kinds=KINDS):
    by_layer = {}
    for d in step.layers:
        attention = d.self_attn
        if hasattr(attention, "dsa_indexer"):
            by_layer[attention.mla_attn.mla_attn.layer_name] = metadata("dsa", rows)
        else:
            attention.prefix = f"model.layers.{d.layer_idx}.self_attn"
            by_layer[attention.prefix] = metadata("kda", rows)
    context = SimpleNamespace(
        cudagraph_runtime_mode=mode, attn_metadata=by_layer, slot_mapping={"s": 0}
    )
    monkeypatch.setattr(
        sp_attention.torch.cuda, "is_current_stream_capturing", lambda: True
    )
    monkeypatch.setattr(sp_attention, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(sp_attention, "get_forward_context", lambda: context)
    return SimpleNamespace(shape=(rows,))


def test_plan_serves_every_captured_row(fakes, monkeypatch):
    """A FULL capture at every captured R plans the SP step (q4, one metadata per
    layer); one outside them, or PIECEWISE / eager, never captures the stock forward
    silently: it raises, or (not a FULL decode capture) runs stock eagerly."""
    fakes()
    step = sp_attention.SpStep(runtime())
    for r in ROWS:
        plan = step.plan(capture(monkeypatch, step, r), None)
        assert plan.q == Q and len(plan.metadata) == len(KINDS), r
        assert all(md is not None for md in plan.metadata)
    with pytest.raises(shapes.Unsupported, match="outside the captured rows 4-512"):
        step.plan(capture(monkeypatch, step, 516), None)
    positions = capture(monkeypatch, step, 64, mode=CUDAGraphMode.PIECEWISE)
    assert step.plan(positions, None) is None
    with pytest.raises(shapes.Unsupported, match="pipeline-parallel"):
        step.plan(capture(monkeypatch, step, 64), object())
    # One layer's metadata not a uniform q4 verify (a mixed step at a FULL size).
    positions = capture(monkeypatch, step, 64)
    context = sp_attention.get_forward_context()
    kda = next(d for d in step.layers if not hasattr(d.self_attn, "dsa_indexer"))
    context.attn_metadata[kda.self_attn.prefix].num_decodes = 1
    with pytest.raises(shapes.Unsupported, match="lack uniform q4 verify"):
        step.plan(positions, None)


@pytest.fixture
def sp_decode(fakes, monkeypatch):
    """SpecializedDecode built with ``sequence_parallel`` over the stand-ins (the
    TP group and device properties of a C512 rank)."""
    made, events = fakes()
    properties = SimpleNamespace(major=10, minor=7, multi_processor_count=148)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: properties)
    group = SimpleNamespace(world_size=TP, rank_in_group=1, device_group="tp-group")
    monkeypatch.setattr(decode, "get_tp_group", lambda: group)
    platform = SimpleNamespace(current_device=lambda: "meta")
    monkeypatch.setattr(decode, "current_platform", platform)
    layers = [decoder(kind, i) for i, kind in enumerate(KINDS)]
    backbone = SimpleNamespace(
        layers=layers, start_layer=0, end_layer=len(layers),
        aux_hidden_state_layers=(2, 5), config=SimpleNamespace(num_attention_heads=96),
        use_sequence_parallel=True, use_attn_res=True, _aux_attn_res_stream=False,
        embed_tokens=None,
    )  # fmt: skip
    config = DecodeConfig(
        max_context=1 << 20, mla_core="auto", dsa_core="auto", dcp=TP, q=Q,
        capture_sizes=ROWS, captured=ROWS, sequence_parallel=True, dp=4, ep=16,
        replayssm=True,
    )  # fmt: skip
    built = decode.SpecializedDecode(backbone, config)
    return built, made, events


def test_specialized_decode_builds_only_the_sp_step(sp_decode):
    """Under SP the decode builds the SP step (rank 1 of TP4, 24 heads per rank) and
    no TP8 / TP16 provider; building them anyway raises."""
    built, _, events = sp_decode
    assert isinstance(built.sp, sp_attention.SpStep)
    assert decode.SpecializedDecode.sp is None  # (the TP8 / TP16 default)
    ctx = built.context
    assert (ctx.tp, ctx.rank, ctx.dcp, ctx.dp, ctx.ep) == (TP, 1, TP, 4, 16)
    assert ctx.sequence_parallel and ctx.heads_per_rank == HEADS and ctx.q == Q
    assert ctx.arch == "rubin" and ctx.max_rows == 512
    assert built.coverage is built.sp.coverage()
    built.coverage.assert_complete(ROWS)
    assert events[0][:3] == ("SpComm", TP, 1)
    assert events[1][:3] == ("SpFinalGather", TP, 1)
    for name in ("attention", "feed_forward", "prefetch", "moe_batches"):
        assert not hasattr(built, name), name
    with pytest.raises(RuntimeError, match="TP providers built under SP"):
        built.build_providers()


def test_specialized_decode_delegates_to_the_sp_step(sp_decode, monkeypatch):
    """Plan / prepare_kv / process_weights_after_loading go to the SP step: every
    core compiled for its layer's caches, the KDA copies refilled and the DSA W_UV
    gather redone in place, in layer order."""
    built, _, _ = sp_decode
    step = built.sp
    built.prepare_kv()
    for d in step.layers:
        core = step.blocks[d.layer_idx].core
        if core.family == "dsa":
            assert core.caches == [d.self_attn.mla_attn.mla_attn.kv_cache]
        else:
            assert core.caches == [d.self_attn.kv_cache]
    built.process_weights_after_loading()
    for block in step.blocks.values():
        if block.kind == "kda":
            assert block.core.reloads == 1
        else:
            assert block.core.box.gathered == 1
    plan = built.plan(capture(monkeypatch, step, 128), None)
    assert plan.q == Q and len(plan.metadata) == len(KINDS)
    with pytest.raises(shapes.Unsupported, match="outside the captured rows"):
        built.plan(capture(monkeypatch, step, 520), None)


@pytest.mark.parametrize("q", [2, 3, 5, 6, 7, 8])
def test_any_q_on_aligned_rows(fakes, monkeypatch, q):
    """DSpark k = 1..7 (q2..q8): the SP step over rows every lcm(q, 4) (whole q
    requests split evenly over TP4) up to 512 is fully specialized and plans every
    captured R from uniform q verify metadata; an unaligned R raises at build."""
    step_rows = math.lcm(q, TP)
    rows = tuple(range(step_rows, 512 + 1, step_rows))
    monkeypatch.setattr(sys.modules[__name__], "Q", q)
    monkeypatch.setattr(sys.modules[__name__], "ROWS", rows)
    fakes()
    step = sp_attention.SpStep(runtime(rows=rows, ctx=context(q=q)))
    assert step.q == q and step.rows == rows
    assert step.shard_cap == rows[-1] // TP
    table = step.coverage()
    table.assert_complete(rows)
    for r in rows:
        assert set(table.plan[r]) == SPECIALIZED | DECLARED, r
        plan = step.plan(capture(monkeypatch, step, r), None)
        assert plan.q == q and len(plan.metadata) == len(KINDS), r
    assert table.lines()[0].startswith(f"M{rows[0]}(q{q}): ")
    bad = rows + (rows[-1] + q,) if (rows[-1] + q) % TP else None
    if bad is not None:
        with pytest.raises(shapes.Unsupported, match="do not split over TP4"):
            sp_attention.SpStep(runtime(rows=bad, ctx=context(q=q)))
