# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 replicated EP_GROUP specialized decode on CPU (``runtime.group_ep``:
TP4 / TP8 / TP16 attention groups, DCP 1 or TP, expert-parallel MegaMoE, DSpark
q 2..8, at most 64 rows per group): the startup of the real ``GroupStep`` (and of
``SpecializedDecode`` in the EP_GROUP mode without sequence parallelism) over
stand-in providers:

* the coverage / declared-seam table: every captured R has the specialized families
  (layout, embed, attn_res, dsa_in, dsa, kda_in, kda, oproj), every other family is
  a declared stock module or collective of the step's ``allowed`` set (the
  embedding's TP all-reduce, the o_proj reduce-scatter + all-gather, the MLP shard
  gather, the trtllm-gen sparse MLA kernel at DCP1 only, the MegaMoE / dense MLPs on
  stock's row shard, LM head, sampler, drafter, verify under synthetic rejection),
  no R missing or stock, no required boundary missing; a stock per-layer AttnRes or
  embedding declaration is refused;
* the DSA core by DCP (DCP1: ``TpDsaCore``; DCP = TP: ``SpDsaCore`` on one DcpBox of
  the group's heads, parities by DSA ordinal), the KDA core in "none" mode, the
  projections' (G, DCP) key;
* no silent fallback: each injected unsupported layer, row set or context raises
  ``shapes.Unsupported`` naming the family, the R and the cause (more than 64 rows,
  MegaMoE capacity, a TP-sharded shared expert, a fused GEMM-RS o_proj, RecoverSSM
  records, ...);
* ``plan`` serves every captured R and raises outside them;
* ``SpecializedDecode`` dispatches EP_GROUP without sequence parallelism to the group
  step and delegates its hooks.

The providers are stand-ins with the contract's interfaces (no collective, no
kernel): their kernels run in the GPU tests (test_k3_sm107_ep_group_gemm_gpu.py,
test_k3_sm107_ep_group_kda_gpu.py, tests/distributed/
test_kimi_k3_sm107_sdec_ep_group_step.py). The load-time refusals of the deployment
(KV dtype, rows, speculation, MegaMoE records) are test_k3_sm107_topology.py's."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass")

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107 import shapes  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime import (  # noqa: E402
    DecodeConfig,
    Topology,
    decode,
    group_ep,
)

WIDTH = 7168
# A K3.1-like layer stack: a dense first layer, then MoE; KDA and DSA interleaved
# (an odd DSA count: the DcpBox parity of the step's last DSA layer and the next
# step's first coincide, separated by the stock collectives between them).
KINDS = ("kda", "kda", "dsa", "kda", "dsa", "kda", "dsa")
SPECIALIZED = {"layout", "embed", "attn_res", "dsa_in", "dsa", "kda_in", "kda"}
SPECIALIZED |= {"oproj"}
DECLARED = {"embed_reduce", "oproj_reduce", "mlp_gather", "moe", "dense"}
DECLARED |= {"lm_head", "sampler", "drafter", "verify"}


def declared(dcp):
    """The declared families at ``dcp``: the trtllm-gen sparse MLA only at DCP1."""
    return DECLARED | ({"sparse_mla"} if dcp == 1 else set())
# (G, DCP, q): every group size at DCP 1 and = G, q 2 / 4 / 8.
TOPOLOGIES = [
    (4, 1, 4),
    (8, 1, 4),
    (8, 8, 4),
    (8, 8, 8),
    (16, 1, 2),
    (16, 16, 4),
]


def rows_of(q):
    """Every captured R = n q up to 64 rows (n = 1..64 / q requests per group)."""
    return tuple(range(q, 65, q))


def pitches(g):
    heads = 96 // g
    dsa = 2112 + heads * 128 + 128 + 32
    kda = -(-(4 * heads * 128 + 128 + heads) // 16) * 16
    return heads, dsa, kda


def meta(*shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="meta")


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
    """The providers ``group_ep`` builds, recording the build order."""

    def __init__(self, events, g, gaps=None, records="none"):
        self.events, self.gaps = events, gaps or {}
        heads, dsa_pitch, kda_pitch = pitches(g)
        fakes = self

        class Projection(Provider):
            def __init__(self, weight, rows, tactics, *, family, group):
                self.family = family
                assert set(tactics) == set(rows), family
                super().__init__(rows, fakes.gaps.get(family, ()))
                self.weight, self.group, self.operands = weight, group, []
                fakes.events.add("DecodeProjection", family, tuple(weight.shape), group)

            def check_operands(self, x, out):
                assert x.shape[1] == self.weight.shape[1], (x.shape, self.weight.shape)
                assert out.shape[1] == self.weight.shape[0], (out.shape, self.weight)
                self.operands.append((tuple(x.shape), tuple(out.shape)))

        class DsaCore(Provider):
            family = "dsa"

            def __init__(self, attention, rows, scratch, *, box, parity, ctx):
                super().__init__(rows, fakes.gaps.get("dsa", ()))
                assert box == "box" and scratch == "scratch"
                assert ctx.ep_group and not ctx.sequence_parallel
                self.attention, self.parity, self.box = attention, parity, Box(g)
                self.pitch, self.caches = dsa_pitch, []
                fakes.events.add("SpDsaCore", attention.index, parity)

            def prepare_cache(self, kv):
                self.caches.append(kv)
                return self

        class TpDsa(Provider):
            family = "dsa"

            def __init__(self, attention, rows, step):
                super().__init__(rows, fakes.gaps.get("dsa", ()))
                self.attention, self.pitch, self.caches = attention, dsa_pitch, []
                fakes.events.add("TpDsaCore", attention.index)

            def prepare_cache(self, kv):
                self.caches.append(kv)
                return self

        class KdaCore(Provider):
            family = "kda"

            def __init__(self, attention, rows, *, records, ctx):
                super().__init__(rows, fakes.gaps.get("kda", ()))
                assert records == "none", records
                assert ctx.ep_group and ctx.heads_per_rank == heads
                self.attention = attention
                self.pitch = attention.in_proj_qkvgfab.weight.shape[0]
                self.caches, self.reloads = [], 0
                fakes.events.add("SpKdaCore", attention.index)

            def prepare_caches(self, caches):
                self.caches.append(caches)
                return self

            def process_weights_after_loading(self):
                self.reloads += 1

        def tactics(family, n, k, rows, ctx):
            assert ctx.ep_group and ctx.tp == g, ctx
            return dict.fromkeys(rows, ("cublas", None))

        def box(group, device, rank, *, heads, cap, interleave):
            fakes.events.add("sp_dsa_box", heads, cap, interleave)
            return "box"

        def scratch(device, cap):
            fakes.events.add("sp_dsa_scratch", cap)
            return "scratch"

        def recoverssm(layers, ctx, *, defer=True):
            fakes.events.add("prepare_recoverssm", defer)
            return records

        def bf16(linear):
            if linear.weight.dtype != torch.bfloat16:
                raise ValueError(f"needs BF16 projections, got {linear.weight.dtype}")
            return linear.weight

        class AttnRes:
            """``PreparedAttnRes`` stand-in: records each site and its aux bind."""

            def __init__(self, site, index, has_delta=True):
                self.site, self.index, self.has_delta = site, index, has_delta
                self.aux = False
                fakes.events.add("AttnRes", site, index, has_delta)

            @classmethod
            def for_target(cls, target, *, final=False, has_delta=True, **_):
                if final:
                    return cls("final", None)
                return cls("before", target.layer_idx, has_delta)

            @classmethod
            def for_post(cls, decoder):
                return cls("post", decoder.layer_idx)

            def bind_aux(self):
                self.aux = True

        def embedding(table, tp, rank):
            assert tp == g, (tp, g)
            fakes.events.add("PreparedEmbedding", tp, rank)
            return SimpleNamespace(table=table, tp=tp, rank=rank)

        self.patch = dict(
            PreparedAttnRes=AttnRes,
            PreparedEmbedding=embedding,
            DecodeProjection=Projection,
            SpDsaCore=DsaCore,
            TpDsaCore=TpDsa,
            SpKdaCore=KdaCore,
            sp_projection_tactics=tactics,
            sp_dsa_box=box,
            sp_dsa_scratch=scratch,
            prepare_recoverssm=recoverssm,
            bf16_weight=bf16,
            pack_dsa_projection=lambda attention: attention.packed,
        )


class Box:
    def __init__(self, g):
        self.g, self.gathered = g, 0

    def gather_heads(self, w_uv):
        """Every rank's heads of ``w_uv``, as DsaBox.gather_heads."""
        self.gathered += 1
        return torch.empty((self.g * w_uv.shape[0], *w_uv.shape[1:]), device="meta")


def linear(*shape, dtype=torch.bfloat16, reduce=False):
    return SimpleNamespace(
        weight=torch.empty(shape, dtype=dtype, device="meta"), reduce_results=reduce
    )


def attention(kind, index, g):
    heads, dsa_pitch, kda_pitch = pitches(g)
    if kind == "kda":
        return SimpleNamespace(
            index=index,
            in_proj_qkvgfab=linear(kda_pitch, WIDTH),
            o_proj=linear(WIDTH, heads * 128),
            gemm_rs_ar=None,
            kv_cache=[meta(2, 3 * heads * 128, 6), meta(2, heads, 128, 128)],
        )
    layer = SimpleNamespace(
        layer_name=f"model.layers.{index}.self_attn.mla_attn",
        kv_cache=meta(8, 64, 576),
        W_UV=meta(heads, 512, 128),
        W_UV_dcp_gathered=[meta(g * heads, 512, 128)],
    )
    indexer = SimpleNamespace(k_cache=SimpleNamespace(prefix=f"idx{index}"))
    return SimpleNamespace(
        index=index,
        packed=meta(dsa_pitch, WIDTH),
        dsa_indexer=indexer,
        mla_attn=SimpleNamespace(mla_attn=layer),
        o_proj=linear(WIDTH, heads * 128),
    )


def decoder(kind, index, g, *, tokens=64):
    moe = index > 0
    mlp = SimpleNamespace(
        use_mega_moe=moe,
        moe_token_shard=0,
        experts=SimpleNamespace(max_num_tokens=tokens),
        shared_experts=SimpleNamespace(shard_sequence_parallel=False),
        shard_sequence_parallel=False,
        gemm_rs_ar=None,
    )
    return SimpleNamespace(
        layer_idx=index,
        self_attn=attention(kind, index, g),
        use_sequence_parallel=True,
        is_moe_layer=moe,
        mlp=mlp,
    )


def context(g, dcp, q, **overrides):
    fields = dict(
        arch="rubin", tp=g, rank=0, dcp=dcp, sms=212, q=q, max_rows=64,
        max_context=1 << 20, bf16_state=False, dp=2, ep=2 * g,
        heads_per_rank=96 // g, ep_group=True,
    )  # fmt: skip
    fields.update(overrides)
    return shapes.Context(**fields)


def vllm_config(method="synthetic", backend="deep_gemm_mega_moe"):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        speculative_config=SimpleNamespace(rejection_sample_method=method),
        kernel_config=SimpleNamespace(moe_backend=backend),
    )


@pytest.fixture
def fakes(monkeypatch):
    """Patch ``group_ep``'s providers with recording stand-ins; returns a function
    (g, gaps, records, method) -> (Fakes, Events)."""

    def install(g, gaps=None, records="none", method="synthetic"):
        events = Events()
        made = Fakes(events, g, gaps, records)
        for name, value in made.patch.items():
            monkeypatch.setattr(group_ep, name, value)
        config = vllm_config(method)
        monkeypatch.setattr(group_ep, "get_current_vllm_config", lambda: config)
        monkeypatch.setattr(
            torch.accelerator, "device_index", lambda index: contextlib.nullcontext()
        )
        return made, events

    return install


def runtime(g, dcp, q, layers=None, rows=None, ctx=None, **backbone):
    stock = dict(
        use_sequence_parallel=True, use_attn_res=True, aux_hidden_state_layers=(2, 5),
        num_attn_res_blocks=8, embed_tokens=SimpleNamespace(),
        _aux_attn_res_stream=False,
    )  # fmt: skip
    stock.update(backbone)
    if layers is None:
        layers = [decoder(kind, i, g) for i, kind in enumerate(KINDS)]
    return SimpleNamespace(
        context=ctx or context(g, dcp, q),
        device=torch.device("meta"),
        backbone=SimpleNamespace(**stock),
        captured=rows_of(q) if rows is None else rows,
        layers=tuple(layers),
        group="tp-group",
    )


@pytest.mark.parametrize("g,dcp,q", TOPOLOGIES)
def test_coverage_every_captured_row(fakes, g, dcp, q):
    """Every captured R specialized: the layout, the embedding gather, AttnRes and
    the block families, the rest declared stock modules / collectives of the step's
    allowed set (sparse MLA at DCP1 only); one log line per R."""
    fakes(g)
    step = group_ep.GroupStep(runtime(g, dcp, q))
    rows = rows_of(q)
    table = step.coverage()
    assert step.rows == rows and sorted(table.plan) == list(rows)
    table.assert_complete(rows, step.allowed)
    expected = declared(dcp)
    for r in rows:
        choices = table.plan[r]
        assert set(choices) == SPECIALIZED | expected, (r, sorted(choices))
        for family in SPECIALIZED:
            assert choices[family].path in (shapes.Path.GENERIC, shapes.Path.TUNED)
        for family in expected:
            assert choices[family].path is shapes.Path.DECLARED, (r, family)
        assert choices["layout"].tactic == "replicated"
        assert f"{-(-r // g)}-row shard" in choices["layout"].note
    assert expected <= step.allowed <= group_ep.GROUP_DECLARED
    assert ("sparse_mla" in step.allowed) == (dcp == 1)
    for stock in ("attn_res", "embed", "final_gather", "final_attn_res"):
        assert stock not in group_ep.GROUP_DECLARED, stock
    assert len(table.lines()) == len(rows)


def test_specialized_attn_res_sites_and_embedding(fakes):
    """Every AttnRes site is a PreparedAttnRes (before each layer, the first
    without a delta; after each attention; the final one), the aux layers' sites
    store the prefix stream, and the embedding is the vocab-shard gather."""
    _, events = fakes(8)
    step = group_ep.GroupStep(runtime(8, 8, 4))
    sites = [e[1:] for e in events if e[0] == "AttnRes"]
    layers = range(len(KINDS))
    assert sites == (
        [("before", i, i != 0) for i in layers]
        + [("post", i, True) for i in layers]
        + [("final", None, True)]
    )
    assert [i for i, s in step.before.items() if s.aux] == [2, 5]
    assert not step.final.aux and not any(s.aux for s in step.post.values())
    assert ("PreparedEmbedding", 8, 0) in events
    assert tuple(step.prefix.shape) == (64, WIDTH)
    assert tuple(step.res_blocks.shape) == (64, 8, WIDTH)


def test_aux_final_layer_binds_final_attn_res(fakes):
    fakes(8)
    step = group_ep.GroupStep(
        runtime(8, 8, 4, aux_hidden_state_layers=(0, len(KINDS)))
    )
    assert step.before[0].aux and step.final.aux


def test_aux_attn_res_stream_refused(fakes):
    """The AttnRes-stream aux mode is not what the specialized sites store."""
    fakes(8)
    with pytest.raises(shapes.Unsupported, match="AttnRes-stream aux") as error:
        group_ep.GroupStep(runtime(8, 8, 4, _aux_attn_res_stream=True))
    assert error.value.family == "attn_res"


def test_missing_boundary_raises(fakes, monkeypatch):
    """A captured R missing a required boundary (here: no MLP gather declared)
    raises at build, even though every planned family is allowed."""
    fakes(8)
    original = group_ep.GroupStep.declared

    def without_gather(self):
        table = original(self)
        table.pop("mlp_gather")
        return table

    monkeypatch.setattr(group_ep.GroupStep, "declared", without_gather)
    with pytest.raises(shapes.Unsupported, match="without a provider: mlp_gather"):
        group_ep.GroupStep(runtime(8, 8, 4))


def test_stock_attn_res_declaration_refused():
    """A per-layer AttnRes declared stock fails assert_complete (not allowed)."""
    table = shapes.Coverage(4)
    table.specialize(4, {"layout": shapes.Choice(shapes.Path.GENERIC)})
    table.declare(4, "attn_res", "stock attn_res")
    with pytest.raises(RuntimeError, match="attn_res=.* is not a declared stock"):
        table.assert_complete((4,), group_ep.GROUP_DECLARED)


def test_declared_notes_name_each_seam(fakes):
    """The declared table names the stock collective and module of each seam."""
    fakes(8)
    step = group_ep.GroupStep(runtime(8, 8, 4))
    choices = step.coverage().plan[64]
    assert "sp_reduce_scatter" in choices["oproj_reduce"].note
    assert "sp_all_gather" in choices["oproj_reduce"].note
    assert "sp_shard" in choices["moe"].note and "EP16" in choices["moe"].note
    assert "deep_gemm_mega_moe" in choices["moe"].note
    assert "PENDING SIGN-OFF" in choices["moe"].note
    assert "sp_all_gather" in choices["mlp_gather"].note
    assert "all-reduce" in choices["embed_reduce"].note
    assert "sparse_mla" not in choices  # DCP = G: inside the specialized DSA chain
    assert "TP8" in choices["drafter"].note
    fakes(8)
    dcp1 = group_ep.GroupStep(runtime(8, 1, 4)).coverage().plan[64]
    assert "trtllm-gen" in dcp1["sparse_mla"].note and "DCP1" in dcp1["sparse_mla"].note


def test_verify_declared_only_under_synthetic_rejection(fakes):
    fakes(16, method="probabilistic")
    step = group_ep.GroupStep(runtime(16, 16, 4))
    table = step.coverage()
    table.assert_complete(rows_of(4), step.allowed)
    for r in (4, 64):
        assert table.plan[r]["verify"].path is shapes.Path.GENERIC
        assert "VerifyArgmax" in table.plan[r]["verify"].note


@pytest.mark.parametrize("g,dcp", [(4, 1), (8, 1), (8, 8), (16, 16)])
def test_dsa_core_by_dcp_and_build_order(fakes, g, dcp):
    """DCP1: TpDsaCore, no DcpBox; DCP = G: one DcpBox of the group's heads per rank
    and the rows cap, then SpDsaCore per DSA layer with alternating parities. The
    KDA core in "none" mode; per block in-projection, core, o_proj, in layer order;
    the projections keyed by (G, DCP)."""
    _, events = fakes(g)
    step = group_ep.GroupStep(runtime(g, dcp, 4))
    names = [e[0] for e in events]
    if dcp == 1:
        assert "sp_dsa_box" not in names and "SpDsaCore" not in names
        assert names[0] == "prepare_recoverssm"
    else:
        assert names[:3] == ["sp_dsa_box", "sp_dsa_scratch", "prepare_recoverssm"]
        assert events[0] == ("sp_dsa_box", 96 // g, 64, 1)
        dsa = [e[2] for e in events if e[0] == "SpDsaCore"]
        assert dsa == [i % 2 for i in range(len(dsa))]
    assert step.records == "none" and step.shard_cap == -(-64 // g)
    core = "TpDsaCore" if dcp == 1 else "SpDsaCore"
    later = ("prepare_recoverssm", "AttnRes", "PreparedEmbedding")
    per_block = [e[0] for e in events if e[0] not in later][(0 if dcp == 1 else 2) :]
    expected = []
    for kind in KINDS:
        expected += ["DecodeProjection", "DecodeProjection"]
        expected.append(core if kind == "dsa" else "SpKdaCore")
    assert per_block == expected
    projections = [e for e in events if e[0] == "DecodeProjection"]
    assert {e[3] for e in projections} == {(g, dcp)}
    heads, dsa_pitch, kda_pitch = pitches(g)
    shapes_by_family = {e[1]: e[2] for e in projections}
    assert shapes_by_family == {
        "dsa_in": (dsa_pitch, WIDTH),
        "kda_in": (kda_pitch, WIDTH),
        "oproj": (WIDTH, heads * 128),
    }


def test_blocks_share_scratch(fakes):
    fakes(8)
    step = group_ep.GroupStep(runtime(8, 8, 4))
    blocks = list(step.blocks.values())
    heads, dsa_pitch, kda_pitch = pitches(8)
    for kind, pitch in (("dsa", dsa_pitch), ("kda", kda_pitch)):
        mine = [b for b in blocks if b.kind == kind]
        assert all(b.projected is mine[0].projected for b in mine)
        assert tuple(mine[0].projected.shape) == (64, pitch)
    assert all(b.attn_out is blocks[0].attn_out for b in blocks)
    assert all(b.partial is blocks[0].partial for b in blocks)
    assert tuple(blocks[0].attn_out.shape) == (64, heads * 128)
    assert tuple(blocks[0].partial.shape) == (64, WIDTH)
    assert step.dense and step.moe


def set_layer(layers, index, **fields):
    for name, value in fields.items():
        target = layers[index]
        *path, last = name.split(".")
        for part in path:
            target = getattr(target, part)
        setattr(target, last, value)
    return layers


def stack(g=8):
    return [decoder(kind, i, g) for i, kind in enumerate(KINDS)]


BUILD_FAILURES = {
    # injected stock layer / topology -> (runtime kwargs, family, cause)
    "rows_above_64": (
        dict(rows=(4, 64, 68)), "group_step",
        r"M68\(q4\): more than 64 rows per attention group \(the wide rows need W4\)",
    ),
    "half_requests": (
        dict(rows=(4, 6, 8)), "group_step", r"M6\(q4\): not whole q4 requests"
    ),
    "no_graphs": (dict(rows=()), "group_step", "no captured FULL decode graphs"),
    "megamoe_tokens": (
        dict(layers=set_layer(stack(), 3, **{"mlp.experts.max_num_tokens": 4})),
        "moe", r"layer 3: MegaMoE max_num_tokens 4 < 8 shard rows",
    ),
    "moe_token_shard": (
        dict(layers=set_layer(stack(), 4, **{"mlp.moe_token_shard": 4})),
        "moe", r"layer 4: not MegaMoE on the stock row shard",
    ),
    "sharded_shared_expert": (
        dict(layers=set_layer(
            stack(), 2, **{"mlp.shared_experts.shard_sequence_parallel": True}
        )),
        "moe", r"layer 2: a TP-sharded shared expert",
    ),
    "sharded_dense": (
        dict(layers=set_layer(stack(), 0, **{"mlp.shard_sequence_parallel": True})),
        "dense", r"layer 0: a TP-sharded dense MLP",
    ),
    "gemm_rs_oproj": (
        dict(layers=set_layer(stack(), 1, **{"self_attn.gemm_rs_ar": object()})),
        "oproj", r"layer 1: a fused GEMM-RS / AR o_proj",
    ),
    "reducing_oproj": (
        dict(layers=set_layer(stack(), 5, **{"self_attn.o_proj.reduce_results": True})),
        "oproj", r"layer 5: a reducing stock o_proj",
    ),
    "not_sp_layer": (
        dict(layers=set_layer(stack(), 1, use_sequence_parallel=False)),
        "group_step", r"layer 1: not a sequence-parallel layer",
    ),
    "mla_layer": (
        dict(layers=set_layer(stack(), 2, self_attn=SimpleNamespace(o_proj=None))),
        "mla", r"layer 2: mla attention \(DSA / KDA only\)",
    ),
    "quantized_projection": (
        dict(layers=set_layer(
            stack(), 3,
            **{"self_attn.in_proj_qkvgfab": linear(
                1, WIDTH, dtype=torch.float8_e4m3fn
            )},
        )),
        "kda", r"layer 3: needs BF16 projections",
    ),
    "sp_context": (
        dict(ctx=context(8, 8, 4, sequence_parallel=True)), "group_step",
        "sequence parallel True",
    ),
    "tp2": (dict(ctx=context(2, 2, 4)), "group_step", r"TP2 / DCP2"),
    "dcp_between": (dict(ctx=context(16, 8, 4)), "group_step", r"TP16 / DCP8"),
    "backbone_not_sp": (
        dict(use_sequence_parallel=False), "group_step",
        "the backbone is not sequence-parallel",
    ),
    "no_attn_res": (
        dict(use_attn_res=False), "attn_res", "the group step runs AttnRes models"
    ),
}  # fmt: skip


@pytest.mark.parametrize("case", list(BUILD_FAILURES))
def test_startup_raises_naming_the_cause(fakes, case):
    """No silent fallback: an injected layer, row set or context the group step
    cannot serve raises at build with the family, the shape and the cause."""
    fakes(8)
    kwargs, family, cause = BUILD_FAILURES[case]
    with pytest.raises(shapes.Unsupported, match=cause) as error:
        group_ep.GroupStep(runtime(8, 8, 4, **kwargs))
    assert error.value.family == family, str(error.value)


def test_recoverssm_records_raise(fakes):
    """ReplaySSM / RecoverSSM records (refused at load: W7) left on the layers are
    never read by the plain KDA core: the build raises."""
    fakes(8, records="triton")
    with pytest.raises(shapes.Unsupported, match=r"RecoverSSM records \(triton\)"):
        group_ep.GroupStep(runtime(8, 8, 4))


@pytest.mark.parametrize("family", ["dsa_in", "dsa", "kda_in", "kda", "oproj"])
@pytest.mark.parametrize("gap", [4, 36, 64])
def test_a_provider_gap_raises_naming_the_row(fakes, family, gap):
    fakes(8, gaps={family: (gap,)})
    with pytest.raises(shapes.Unsupported, match=rf"{family} at {gap}: not prepared"):
        group_ep.GroupStep(runtime(8, 8, 4))


def metadata(kind, rows, q):
    n = rows // q
    if kind == "kda":
        return SimpleNamespace(
            num_prefills=0, num_decodes=0, num_spec_decodes=n,
            num_spec_decode_tokens=rows,
            spec_state_indices_tensor=torch.empty((n, 1), device="meta"),
        )  # fmt: skip
    return SimpleNamespace(num_prefills=0, num_decodes=n, num_decode_tokens=rows)


def capture(monkeypatch, step, rows, mode=CUDAGraphMode.FULL):
    by_layer = {}
    for d in step.layers:
        attention = d.self_attn
        if hasattr(attention, "dsa_indexer"):
            by_layer[attention.mla_attn.mla_attn.layer_name] = metadata(
                "dsa", rows, step.q
            )
        else:
            attention.prefix = f"model.layers.{d.layer_idx}.self_attn"
            by_layer[attention.prefix] = metadata("kda", rows, step.q)
    context_ = SimpleNamespace(
        cudagraph_runtime_mode=mode, attn_metadata=by_layer, slot_mapping={"s": 0}
    )
    monkeypatch.setattr(
        group_ep.torch.cuda, "is_current_stream_capturing", lambda: True
    )
    monkeypatch.setattr(group_ep, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(group_ep, "get_forward_context", lambda: context_)
    return SimpleNamespace(shape=(rows,))


def test_plan_serves_every_captured_row(fakes, monkeypatch):
    """A FULL capture at every captured R plans the group step; one outside them
    raises; PIECEWISE / eager steps run stock eagerly (the same MegaMoE collective
    per MoE layer as the step)."""
    fakes(8)
    step = group_ep.GroupStep(runtime(8, 8, 8))
    for r in rows_of(8):
        plan = step.plan(capture(monkeypatch, step, r), None)
        assert plan.q == 8 and len(plan.metadata) == len(KINDS), r
    with pytest.raises(shapes.Unsupported, match="outside the captured rows 8-64"):
        step.plan(capture(monkeypatch, step, 72), None)
    positions = capture(monkeypatch, step, 64, mode=CUDAGraphMode.PIECEWISE)
    assert step.plan(positions, None) is None
    with pytest.raises(shapes.Unsupported, match="pipeline-parallel"):
        step.plan(capture(monkeypatch, step, 64), object())
    monkeypatch.setattr(
        group_ep.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    assert step.plan(SimpleNamespace(shape=(64,)), None) is None


@pytest.fixture
def group_decode(fakes, monkeypatch):
    """SpecializedDecode built in the EP_GROUP mode without sequence parallelism
    (TP8 = DCP8 x DP2, EP16): rank 3 of the group."""
    made, events = fakes(8)
    properties = SimpleNamespace(major=10, minor=7, multi_processor_count=212)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: properties)
    group = SimpleNamespace(world_size=8, rank_in_group=3, device_group="tp-group")
    monkeypatch.setattr(decode, "get_tp_group", lambda: group)
    platform = SimpleNamespace(current_device=lambda: "meta")
    monkeypatch.setattr(decode, "current_platform", platform)
    layers = stack(8)
    backbone = SimpleNamespace(
        layers=layers, start_layer=0, end_layer=len(layers),
        aux_hidden_state_layers=(2, 5), config=SimpleNamespace(num_attention_heads=96),
        use_sequence_parallel=True, use_attn_res=True, num_attn_res_blocks=8,
        embed_tokens=SimpleNamespace(), _aux_attn_res_stream=False,
    )  # fmt: skip
    topology = Topology(
        g=8, dcp=8, dp=2, ep=16, moe_backend="deep_gemm_mega_moe", q_set=(4,),
        stock_sp=True,
    )  # fmt: skip
    config = DecodeConfig(
        max_context=1 << 20, mla_core="auto", dsa_core="auto", dcp=8, q=4,
        capture_sizes=rows_of(4), captured=rows_of(4), dp=2, ep=16,
        topology=topology,
    )  # fmt: skip
    return decode.SpecializedDecode(backbone, config), made, events


def test_specialized_decode_builds_only_the_group_step(group_decode):
    built, _, _ = group_decode
    assert isinstance(built.sp, group_ep.GroupStep)
    ctx = built.context
    assert (ctx.tp, ctx.rank, ctx.dcp, ctx.dp, ctx.ep) == (8, 3, 8, 2, 16)
    assert ctx.ep_group and not ctx.sequence_parallel and ctx.ep_group_step
    assert ctx.heads_per_rank == 12 and ctx.q == 4 and ctx.max_rows == 64
    assert built.coverage is built.sp.coverage()
    built.coverage.assert_complete(rows_of(4), group_ep.GROUP_DECLARED)
    for name in ("attention", "feed_forward", "prefetch", "moe_batches"):
        assert not hasattr(built, name), name
    with pytest.raises(RuntimeError, match="TP providers built under SP / EP_GROUP"):
        built.build_providers()


def test_specialized_decode_delegates_to_the_group_step(group_decode, monkeypatch):
    built, _, _ = group_decode
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
    plan = built.plan(capture(monkeypatch, step, 32), None)
    assert plan.q == 4 and len(plan.metadata) == len(KINDS)
