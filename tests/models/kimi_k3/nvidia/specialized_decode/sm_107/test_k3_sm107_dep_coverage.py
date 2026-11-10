# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DEP specialized decode on CPU (``runtime.dep``: TP1 = DCP1 attention,
96 heads per rank, expert-parallel MegaMoE over DP16 / DP32, DSpark q 2..8): the
startup of the real ``DepStep`` (and of ``SpecializedDecode`` in the DEP mode) over
stand-in providers:

* the coverage / declared-seam table: every captured R has the specialized families
  (layout, embed, attn_res, dsa_in, dsa, kda_in, kda, oproj), every other family is
  one of ``DEP_DECLARED`` (the stock MoE / dense modules, the trtllm-gen sparse MLA,
  LM head, sampler, verify, drafter); no collective is declared (TP1);
* the providers: ``TpDsaCore`` for every DSA layer (no DcpBox), the KDA core in
  "none" mode at 96 heads, the projections keyed (1, 1) at the DEP shapes;
* no silent fallback: each injected unsupported layer, row set or context raises
  ``shapes.Unsupported`` naming the family, the R and the cause;
* no 64-row cap: rows up to 512 requests x q build;
* ``plan`` serves every captured R and raises outside them;
* ``run`` calls the stock MLP once per layer on every row, runs no SP collective and
  leaves ``forward_context.is_padding`` untouched (stock DEP's mask);
* ``SpecializedDecode`` dispatches DEP to the DEP step and delegates its hooks.

The providers are stand-ins (no kernel): their kernels run in the GPU tests
(test_k3_sm107_dep_kernels_gpu.py, tests/distributed/
test_kimi_k3_sm107_sdec_dep_step.py). The load-time refusals of the deployment are
test_k3_sm107_topology.py's."""

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
    dep,
    group_ep,
)

WIDTH = 7168
HEADS = 96
# A K3.1-like layer stack: a dense first layer, then MoE; KDA and DSA interleaved.
KINDS = ("kda", "kda", "dsa", "kda", "dsa", "kda", "dsa")
SPECIALIZED = {"layout", "embed", "attn_res", "dsa_in", "dsa", "kda_in", "kda"}
SPECIALIZED |= {"oproj"}
DECLARED = {"moe", "dense", "sparse_mla", "lm_head", "sampler", "verify", "drafter"}
# Per-rank weights at TP1 (plan section 0.9).
DSA_PITCH = 2112 + HEADS * 128 + 128 + 32  # 14560
KDA_PITCH = -(-(4 * HEADS * 128 + 128 + HEADS) // 16) * 16  # 49376


def rows_of(q, requests=64):
    """Every captured R = n q, n = 1..requests per rank."""
    return tuple(range(q, requests * q + 1, q))


def meta(*shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="meta")


class Events(list):
    def add(self, *event):
        self.append(event)


class Provider:
    """A stand-in provider: ``choice`` GENERIC at the rows it prepared, raising
    ``Unsupported`` (with the int R) elsewhere; ``run`` records the call."""

    family = "provider"

    def __init__(self, rows, events, gaps=()):
        self.rows = frozenset(rows) - frozenset(gaps)
        self.events = events

    def choice(self, r):
        if r not in self.rows:
            raise shapes.Unsupported(self.family, r, "not prepared")
        return shapes.Choice(shapes.Path.GENERIC, note=f"stand-in {self.family}")


class Fakes:
    """The providers ``dep`` / ``group_ep`` build, recording the build order."""

    def __init__(self, events, gaps=None, records="none"):
        self.events, self.gaps = events, gaps or {}
        fakes = self

        class Projection(Provider):
            def __init__(self, weight, rows, tactics, *, family, group):
                self.family = family
                assert set(tactics) == set(rows), family
                super().__init__(rows, fakes.events, fakes.gaps.get(family, ()))
                self.weight, self.group = weight, group
                fakes.events.add("DecodeProjection", family, tuple(weight.shape), group)

            def check_operands(self, x, out):
                assert x.shape[1] == self.weight.shape[1], (x.shape, self.weight.shape)
                assert out.shape[1] == self.weight.shape[0], (out.shape, self.weight)

            def run(self, x, out):
                assert x.shape[0] == out.shape[0] and x.shape[0] in self.rows
                fakes.events.add("run", self.family, x.shape[0])
                return out

        class TpDsa(Provider):
            family = "dsa"

            def __init__(self, attention, rows, step):
                super().__init__(rows, fakes.events, fakes.gaps.get("dsa", ()))
                assert step.ctx.dep and (step.ctx.tp, step.ctx.dcp) == (1, 1)
                self.attention, self.pitch, self.caches = attention, DSA_PITCH, []
                fakes.events.add("TpDsaCore", attention.index)

            def prepare_cache(self, kv):
                self.caches.append(kv)
                return self

            def run(self, projected, out, **kwargs):
                fakes.events.add("run", "dsa", projected.shape[0])
                return out

        class KdaCore(Provider):
            family = "kda"

            def __init__(self, attention, rows, *, records, ctx):
                super().__init__(rows, fakes.events, fakes.gaps.get("kda", ()))
                assert records == "none", records
                assert ctx.dep and ctx.ep_mode_step and not ctx.ep_group_step
                assert ctx.heads_per_rank == HEADS
                self.attention = attention
                self.pitch = attention.in_proj_qkvgfab.weight.shape[0]
                self.caches, self.reloads = [], 0
                fakes.events.add("SpKdaCore", attention.index)

            def prepare_caches(self, caches):
                self.caches.append(caches)
                return self

            def process_weights_after_loading(self):
                self.reloads += 1

            def run(self, projected, out, **kwargs):
                fakes.events.add("run", "kda", projected.shape[0])
                return out

        def tactics(family, n, k, rows, ctx):
            assert ctx.dep and (ctx.tp, ctx.dcp) == (1, 1), ctx
            return dict.fromkeys(rows, ("cublas", None))

        def recoverssm(layers, ctx, *, defer=True):
            fakes.events.add("prepare_recoverssm", defer)
            return records

        def bf16(linear):
            if linear.weight.dtype != torch.bfloat16:
                raise ValueError(f"needs BF16 projections, got {linear.weight.dtype}")
            return linear.weight

        class AttnRes:
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

            def run(self, delta, prefix, blocks, out, aux=None):
                assert delta.shape[0] == prefix.shape[0] == out.shape[0]
                fakes.events.add("run", f"attn_res.{self.site}", delta.shape[0])
                return out

        def embedding(table, tp, rank):
            assert (tp, rank) == (1, 0), (tp, rank)
            fakes.events.add("PreparedEmbedding", tp, rank)
            return SimpleNamespace(run=lambda ids, out: out)

        def unexpected(name):
            def fail(*args, **kwargs):
                raise AssertionError(f"DEP built / ran {name}")

            return fail

        self.group_patch = dict(
            PreparedAttnRes=AttnRes,
            DecodeProjection=Projection,
            TpDsaCore=TpDsa,
            SpKdaCore=KdaCore,
            sp_projection_tactics=tactics,
            bf16_weight=bf16,
            pack_dsa_projection=lambda attention: attention.packed,
            SpDsaCore=unexpected("SpDsaCore"),
            sp_dsa_box=unexpected("sp_dsa_box"),
            sp_dsa_scratch=unexpected("sp_dsa_scratch"),
            sp_all_gather=unexpected("sp_all_gather"),
            sp_reduce_scatter=unexpected("sp_reduce_scatter"),
            sp_shard=unexpected("sp_shard"),
            sp_padding_mask=unexpected("sp_padding_mask"),
        )
        self.dep_patch = dict(
            PreparedEmbedding=embedding, prepare_recoverssm=recoverssm
        )


def linear(*shape, dtype=torch.bfloat16, reduce=True):
    return SimpleNamespace(
        weight=torch.empty(shape, dtype=dtype, device="meta"), reduce_results=reduce
    )


def attention(kind, index):
    if kind == "kda":
        return SimpleNamespace(
            index=index,
            in_proj_qkvgfab=linear(KDA_PITCH, WIDTH),
            o_proj=linear(WIDTH, HEADS * 128),
            gemm_rs_ar=None,
            kv_cache=[meta(2, 3 * HEADS * 128, 6), meta(2, HEADS, 128, 128)],
        )
    layer = SimpleNamespace(
        layer_name=f"model.layers.{index}.self_attn.mla_attn",
        kv_cache=meta(8, 64, 576),
        W_UV=meta(HEADS, 512, 128),
    )
    indexer = SimpleNamespace(k_cache=SimpleNamespace(prefix=f"idx{index}"))
    return SimpleNamespace(
        index=index,
        packed=meta(DSA_PITCH, WIDTH),
        dsa_indexer=indexer,
        mla_attn=SimpleNamespace(mla_attn=layer),
        o_proj=linear(WIDTH, HEADS * 128),
    )


class Mlp(SimpleNamespace):
    """The stock MoE / dense MLP stand-in: records the rows of each call."""

    def __call__(self, x):
        self.calls.append(x.shape[0])
        return torch.empty_like(x)


def decoder(kind, index, *, tokens=8192):
    moe = index > 0
    mlp = Mlp(
        use_mega_moe=moe,
        moe_token_shard=0,
        experts=SimpleNamespace(max_num_tokens=tokens),
        shared_experts=SimpleNamespace(shard_sequence_parallel=False),
        shard_sequence_parallel=False,
        gemm_rs_ar=None,
        calls=[],
    )
    return SimpleNamespace(
        layer_idx=index,
        self_attn=attention(kind, index),
        use_sequence_parallel=False,
        is_moe_layer=moe,
        mlp=mlp,
    )


def context(q, **overrides):
    fields = dict(
        arch="rubin", tp=1, rank=0, dcp=1, sms=212, q=q, max_rows=64 * q,
        max_context=1 << 20, bf16_state=False, dp=16, ep=16,
        heads_per_rank=HEADS, dep=True,
    )  # fmt: skip
    fields.update(overrides)
    return shapes.Context(**fields)


def vllm_config(method="probabilistic", backend="deep_gemm_mega_moe"):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        speculative_config=SimpleNamespace(rejection_sample_method=method),
        kernel_config=SimpleNamespace(moe_backend=backend),
    )


@pytest.fixture
def fakes(monkeypatch):
    """Patch ``dep`` / ``group_ep``'s providers with recording stand-ins; returns a
    function (gaps, records, method) -> (Fakes, Events)."""

    def install(gaps=None, records="none", method="probabilistic"):
        events = Events()
        made = Fakes(events, gaps, records)
        for name, value in made.group_patch.items():
            monkeypatch.setattr(group_ep, name, value)
        for name, value in made.dep_patch.items():
            monkeypatch.setattr(dep, name, value)
        config = vllm_config(method)
        monkeypatch.setattr(group_ep, "get_current_vllm_config", lambda: config)
        monkeypatch.setattr(dep, "get_current_vllm_config", lambda: config)
        monkeypatch.setattr(
            torch.accelerator, "device_index", lambda index: contextlib.nullcontext()
        )
        return made, events

    return install


def runtime(q, layers=None, rows=None, ctx=None, **backbone):
    stock = dict(
        use_sequence_parallel=False, use_attn_res=True, aux_hidden_state_layers=(2, 5),
        num_attn_res_blocks=8, embed_tokens=SimpleNamespace(),
        _aux_attn_res_stream=False,
        collect_remote_aux_hidden_states=lambda tensors: [],
    )  # fmt: skip
    stock.update(backbone)
    if layers is None:
        layers = [decoder(kind, i) for i, kind in enumerate(KINDS)]
    return SimpleNamespace(
        context=ctx or context(q),
        device=torch.device("meta"),
        backbone=SimpleNamespace(**stock),
        captured=rows_of(q) if rows is None else rows,
        layers=tuple(layers),
        group="tp-group",
    )


@pytest.mark.parametrize("q", [2, 4, 8])
def test_coverage_every_captured_row(fakes, q):
    """Every captured R specialized: the layout, the embedding, AttnRes and the
    block families; the rest exactly ``DEP_DECLARED``; one log line per R."""
    fakes()
    step = dep.DepStep(runtime(q))
    rows = rows_of(q)
    table = step.coverage()
    assert step.rows == rows and sorted(table.plan) == list(rows)
    table.assert_complete(rows, step.allowed)
    for r in rows:
        choices = table.plan[r]
        assert set(choices) == SPECIALIZED | DECLARED, (r, sorted(choices))
        for family in SPECIALIZED:
            assert choices[family].path in (shapes.Path.GENERIC, shapes.Path.TUNED)
        for family in DECLARED:
            assert choices[family].path is shapes.Path.DECLARED, (r, family)
        assert choices["layout"].tactic == "local"
    assert step.allowed == dep.DEP_DECLARED == frozenset(DECLARED)
    for collective in ("embed_reduce", "oproj_reduce", "mlp_gather", "final_gather"):
        assert collective not in dep.DEP_DECLARED, collective
    for stock in ("attn_res", "embed"):
        assert stock not in dep.DEP_DECLARED, stock
    assert len(table.lines()) == len(rows)


def test_declared_notes_name_each_seam(fakes):
    fakes()
    choices = dep.DepStep(runtime(4)).coverage().plan[64]
    assert "EP16" in choices["moe"].note and "deep_gemm_mega_moe" in choices["moe"].note
    assert "PENDING SIGN-OFF" in choices["moe"].note
    assert "trtllm-gen" in choices["sparse_mla"].note
    assert "128" in choices["sparse_mla"].note
    assert "B1 / B2" in choices["verify"].note
    assert "W10" in choices["drafter"].note
    assert "163840" in choices["lm_head"].note


def test_verify_declared_at_every_method(fakes):
    """DEP phase A declares the stock verify under real rejection too."""
    for method in ("synthetic", "probabilistic", "greedy"):
        fakes(method=method)
        step = dep.DepStep(runtime(4))
        table = step.coverage()
        table.assert_complete(rows_of(4), step.allowed)
        assert table.plan[4]["verify"].path is shapes.Path.DECLARED


def test_providers_and_build_order(fakes):
    """TpDsaCore for every DSA layer (no DcpBox), the KDA core in "none" mode; per
    block in-projection, o_proj, core, in layer order; projections keyed (1, 1) at
    the DEP shapes; the embedding at TP1."""
    _, events = fakes()
    step = dep.DepStep(runtime(4))
    names = [e[0] for e in events]
    assert names[0] == "prepare_recoverssm" and events[0] == ("prepare_recoverssm",
                                                               False)  # fmt: skip
    assert step.records == "none" and step.dsa_box is None
    later = ("prepare_recoverssm", "AttnRes", "PreparedEmbedding")
    per_block = [e[0] for e in events if e[0] not in later]
    expected = []
    for kind in KINDS:
        expected += ["DecodeProjection", "DecodeProjection"]
        expected.append("TpDsaCore" if kind == "dsa" else "SpKdaCore")
    assert per_block == expected
    projections = [e for e in events if e[0] == "DecodeProjection"]
    assert {e[3] for e in projections} == {(1, 1)}
    assert {e[1]: e[2] for e in projections} == {
        "dsa_in": (DSA_PITCH, WIDTH),
        "kda_in": (KDA_PITCH, WIDTH),
        "oproj": (WIDTH, HEADS * 128),
    }
    assert (DSA_PITCH, KDA_PITCH) == (14560, 49376)
    assert ("PreparedEmbedding", 1, 0) in events
    assert step.dense and step.moe


def test_no_64_row_cap(fakes):
    """Rows are capped by the load gate's 512 requests only: 512 x q4 builds."""
    fakes()
    rows = rows_of(4, 512)
    step = dep.DepStep(runtime(4, rows=rows))
    step.coverage().assert_complete(rows, step.allowed)
    assert step.rows[-1] == 2048 and tuple(step.prefix.shape) == (2048, WIDTH)


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
    # injected stock layer / topology -> (runtime kwargs, family, cause)
    "half_requests": (
        dict(rows=(4, 6, 8)), "dep_step", r"M6\(q4\): not whole q4 requests"
    ),
    "no_graphs": (dict(rows=()), "dep_step", "no captured FULL decode graphs"),
    "megamoe_tokens": (
        dict(layers=set_layer(stack(), 3, **{"mlp.experts.max_num_tokens": 128})),
        "moe", r"layer 3: MegaMoE max_num_tokens 128 < 256 rows",
    ),
    "moe_token_shard": (
        dict(layers=set_layer(stack(), 4, **{"mlp.moe_token_shard": 4})),
        "moe", r"layer 4: not MegaMoE on the rank's rows",
    ),
    "not_mega_moe": (
        dict(layers=set_layer(stack(), 4, **{"mlp.use_mega_moe": False})),
        "moe", r"layer 4: not MegaMoE",
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
    "gemm_rs_dense": (
        dict(layers=set_layer(stack(), 0, **{"mlp.gemm_rs_ar": object()})),
        "dense", r"layer 0: a fused GEMM-RS / AR dense MLP",
    ),
    "gemm_rs_oproj": (
        dict(layers=set_layer(stack(), 1, **{"self_attn.gemm_rs_ar": object()})),
        "oproj", r"layer 1: a fused GEMM-RS / AR o_proj",
    ),
    "sp_layer": (
        dict(layers=set_layer(stack(), 1, use_sequence_parallel=True)),
        "dep_step", r"layer 1: a sequence-parallel layer",
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
    "group_context": (
        dict(ctx=context(4, dep=False, ep_group=True)), "dep_step",
        "the DEP step only",
    ),
    "sp_context": (
        dict(ctx=context(4, sequence_parallel=True)), "dep_step",
        "the DEP step only",
    ),
    "tp8_context": (
        dict(ctx=context(4, tp=8, heads_per_rank=12)), "dep_step", r"TP8 / DCP1"
    ),
    "heads_12": (dict(ctx=context(4, heads_per_rank=12)), "dep_step", "12 heads"),
    "backbone_sp": (
        dict(use_sequence_parallel=True), "dep_step",
        "the backbone is sequence-parallel",
    ),
    "no_attn_res": (
        dict(use_attn_res=False), "attn_res", "the DEP step runs AttnRes models"
    ),
    "aux_stream": (
        dict(_aux_attn_res_stream=True), "attn_res", "AttnRes-stream aux"
    ),
}  # fmt: skip


@pytest.mark.parametrize("case", list(BUILD_FAILURES))
def test_startup_raises_naming_the_cause(fakes, case):
    """No silent fallback: an injected layer, row set or context the DEP step cannot
    serve raises at build with the family, the shape and the cause."""
    fakes()
    kwargs, family, cause = BUILD_FAILURES[case]
    with pytest.raises(shapes.Unsupported, match=cause) as error:
        dep.DepStep(runtime(4, **kwargs))
    assert error.value.family == family, str(error.value)


def test_recoverssm_records_raise(fakes):
    fakes(records="triton")
    with pytest.raises(shapes.Unsupported, match=r"RecoverSSM records \(triton\)"):
        dep.DepStep(runtime(4))


@pytest.mark.parametrize("family", ["dsa_in", "dsa", "kda_in", "kda", "oproj"])
@pytest.mark.parametrize("gap", [4, 128, 256])
def test_a_provider_gap_raises_naming_the_row(fakes, family, gap):
    fakes(gaps={family: (gap,)})
    with pytest.raises(shapes.Unsupported, match=rf"{family} at {gap}: not prepared"):
        dep.DepStep(runtime(4))


def test_missing_boundary_raises(fakes, monkeypatch):
    """A captured R missing a required boundary (no drafter declared) raises."""
    fakes()
    original = dep.DepStep.declared

    def without_drafter(self):
        table = original(self)
        table.pop("drafter")
        return table

    monkeypatch.setattr(dep.DepStep, "declared", without_drafter)
    with pytest.raises(shapes.Unsupported, match="without a provider: drafter"):
        dep.DepStep(runtime(4))


def metadata(kind, rows, q):
    n = rows // q
    if kind == "kda":
        return SimpleNamespace(
            num_prefills=0, num_decodes=0, num_spec_decodes=n,
            num_spec_decode_tokens=rows,
            spec_state_indices_tensor=torch.empty((n, 1), device="meta"),
        )  # fmt: skip
    return SimpleNamespace(num_prefills=0, num_decodes=n, num_decode_tokens=rows)


def capture(monkeypatch, step, rows, mode=CUDAGraphMode.FULL, padding="mask"):
    by_layer = {}
    for d in step.layers:
        attention = d.self_attn
        if hasattr(attention, "dsa_indexer"):
            by_layer[attention.mla_attn.mla_attn.layer_name] = metadata(
                "dsa", rows, step.q
            )
            by_layer[attention.dsa_indexer.k_cache.prefix] = SimpleNamespace(
                slot_mapping="index-slots"
            )
        else:
            attention.prefix = f"model.layers.{d.layer_idx}.self_attn"
            by_layer[attention.prefix] = metadata("kda", rows, step.q)
    slots = {
        d.self_attn.mla_attn.mla_attn.layer_name: "slots"
        for d in step.layers
        if hasattr(d.self_attn, "dsa_indexer")
    }
    context_ = SimpleNamespace(
        cudagraph_runtime_mode=mode, attn_metadata=by_layer, slot_mapping=slots,
        is_padding=padding,
    )  # fmt: skip
    for module in (group_ep, dep):
        monkeypatch.setattr(module, "get_forward_context", lambda: context_)
    monkeypatch.setattr(
        group_ep.torch.cuda, "is_current_stream_capturing", lambda: True
    )
    monkeypatch.setattr(group_ep, "is_forward_context_available", lambda: True)
    return SimpleNamespace(shape=(rows,)), context_


def test_plan_serves_every_captured_row(fakes, monkeypatch):
    """A FULL capture at every captured R plans the DEP step; one outside them
    raises (family dep_step); PIECEWISE / eager steps run stock eagerly (the same
    MegaMoE collective per MoE layer as the step)."""
    fakes()
    step = dep.DepStep(runtime(8, rows=rows_of(8, 16)))
    for r in rows_of(8, 16):
        plan = step.plan(capture(monkeypatch, step, r)[0], None)
        assert plan.q == 8 and len(plan.metadata) == len(KINDS), r
    outside = "outside the captured rows 8-128"
    with pytest.raises(shapes.Unsupported, match=outside) as error:
        step.plan(capture(monkeypatch, step, 136)[0], None)
    assert error.value.family == "dep_step"
    positions = capture(monkeypatch, step, 64, mode=CUDAGraphMode.PIECEWISE)[0]
    assert step.plan(positions, None) is None
    with pytest.raises(shapes.Unsupported, match="pipeline-parallel"):
        step.plan(capture(monkeypatch, step, 64)[0], object())
    monkeypatch.setattr(
        group_ep.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    assert step.plan(SimpleNamespace(shape=(64,)), None) is None


@pytest.mark.parametrize("rows", [4, 64, 256])
def test_run_composition(fakes, monkeypatch, rows):
    """A planned step: the embedding, then per layer the pre AttnRes, in-projection,
    core, o_proj, post AttnRes and the stock MLP once on every row; the final
    AttnRes; no SP collective (the stand-ins raise) and ``is_padding`` untouched."""
    _, events = fakes()
    step = dep.DepStep(runtime(4))
    positions, context_ = capture(monkeypatch, step, rows)
    plan = step.plan(positions, None)
    monkeypatch.setattr(
        decode, "caches_of", lambda attention: [attention.mla_attn.mla_attn.kv_cache]
        if hasattr(attention, "dsa_indexer") else list(attention.kv_cache)
    )  # fmt: skip
    events.clear()
    ids = torch.zeros(rows, dtype=torch.int64, device="meta")
    out, streams = step.run(plan, ids, None)
    assert tuple(out.shape) == (rows, WIDTH) and len(streams) == 2
    assert context_.is_padding == "mask"
    for d in step.layers:
        assert d.mlp.calls == [rows], (d.layer_idx, d.mlp.calls)
    runs = [e[1:] for e in events if e[0] == "run"]
    expected = []
    for kind in KINDS:
        expected += [("attn_res.before", rows), (f"{kind}_in", rows), (kind, rows)]
        expected += [("oproj", rows), ("attn_res.post", rows)]
    expected.append(("attn_res.final", rows))
    assert runs == expected


@pytest.fixture
def dep_decode(fakes, monkeypatch):
    """SpecializedDecode built in the DEP mode (TP1 x DP16, EP16)."""
    made, events = fakes()
    properties = SimpleNamespace(major=10, minor=7, multi_processor_count=212)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: properties)
    group = SimpleNamespace(world_size=1, rank_in_group=0, device_group="tp-group")
    monkeypatch.setattr(decode, "get_tp_group", lambda: group)
    platform = SimpleNamespace(current_device=lambda: "meta")
    monkeypatch.setattr(decode, "current_platform", platform)
    layers = stack()
    backbone = SimpleNamespace(
        layers=layers, start_layer=0, end_layer=len(layers),
        aux_hidden_state_layers=(2, 5), config=SimpleNamespace(num_attention_heads=96),
        use_sequence_parallel=False, use_attn_res=True, num_attn_res_blocks=8,
        embed_tokens=SimpleNamespace(), _aux_attn_res_stream=False,
        collect_remote_aux_hidden_states=lambda tensors: [],
    )  # fmt: skip
    topology = Topology(
        g=1, dcp=1, dp=16, ep=16, moe_backend="deep_gemm_mega_moe", q_set=(4,),
        stock_sp=False, moe_tokens=8192,
    )  # fmt: skip
    config = DecodeConfig(
        max_context=1 << 20, mla_core="auto", dsa_core="auto", dcp=1, q=4,
        capture_sizes=rows_of(4), captured=rows_of(4), dp=16, ep=16,
        topology=topology,
    )  # fmt: skip
    return decode.SpecializedDecode(backbone, config), made, events


def test_specialized_decode_builds_only_the_dep_step(dep_decode):
    built, _, _ = dep_decode
    assert isinstance(built.sp, dep.DepStep)
    ctx = built.context
    assert (ctx.tp, ctx.rank, ctx.dcp, ctx.dp, ctx.ep) == (1, 0, 1, 16, 16)
    assert ctx.dep and ctx.ep_mode_step and not ctx.ep_group_step
    assert ctx.heads_per_rank == 96 and ctx.q == 4 and ctx.max_rows == 256
    assert built.coverage is built.sp.coverage()
    built.coverage.assert_complete(rows_of(4), dep.DEP_DECLARED)
    for name in ("attention", "feed_forward", "prefetch", "moe_batches"):
        assert not hasattr(built, name), name
    with pytest.raises(RuntimeError, match="TP providers built under"):
        built.build_providers()


def test_specialized_decode_delegates_to_the_dep_step(dep_decode, monkeypatch):
    built, _, _ = dep_decode
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
    plan = built.plan(capture(monkeypatch, step, 32)[0], None)
    assert plan.q == 4 and len(plan.metadata) == len(KINDS)
