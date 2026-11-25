# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1's release DSpark drafter pieces on Rubin GPUs (``runtime.draft``
GqaDraftFastPath; see test_dspark_gqa_fastpath.py for the CPU checks).

1 GPU (pytest): every split-K tactic of the GQA tables at every row of its bin
against the stock GEMM (F.linear, cuBLAS), and the release model's context step
(``K3GqaDSparkModel.combine``: hidden_norm(fc(aux)) with the stock kernels) against
the stock Qwen3 DSpark context step (fc, then the stored-row RMSNorm).

Torchrun (one process per GPU, Rubin, multicast memory; the drafter forward at the
TP16 per-rank geometry, the exchange over the local GPUs): ``GqaSites`` (the
release drafter's forward over PreparedDraftAr's Lamport all-reduce + norm sites,
with the GQA producers: ring o_proj (7168, 384) / down (7168, 896) at one-shot rows,
two-shot pushing producers + SiLU-fused gate/up at 9-64 rows, ArNormRows above) in a
CUDA graph against the stock math (F.linear partials, FP32 rank-order sums, BF16
residual adds, RMSNorm), with a stand-in attention core (the attention kernel is
cutedsl.draft.gqa_*):

    torchrun --nproc-per-node 2 tests/models/kimi_k3/test_dspark_gqa_fastpath_gpu.py
    torchrun --nproc-per-node 4 ... (or --nnodes 4 --nproc-per-node 4: 16 ranks)
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass")

HIDDEN = 7168
EPS = 1e-5


def _rubin():
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (10, 7)


def _rubin_world():
    if not _rubin() or torch.cuda.device_count() < 2:
        return 0
    return torch.cuda.device_count()


# ------------------------------------------------------------------- 1 GPU


def _tabled(tp):
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft

    out = []
    for shape, tactic in draft.GQA_PROJECTIONS.get(tp, {}).items():
        out.append((shape, tactic, range(1, draft.PROJECTION_ROWS + 1)))
    low = draft.PROJECTION_ROWS
    for top, table in draft.GQA_PROJECTIONS_WIDE.get(tp, ()):
        for shape, tactic in table.items():
            out.append((shape, tactic, range(low + 1, top + 1)))
        low = top
    return out


@pytest.mark.skipif(not _rubin(), reason="Rubin (SM107) only")
@pytest.mark.parametrize("tp", [16, 8, 4])
def test_gqa_tactics_match_stock_gemm(tp):
    """Every GQA split-K tactic at every row count of its bin (the drafter rows it
    can serve) within the FP32 split-sum reorder of the stock GEMM."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (
        PreparedProjection,
    )

    device = torch.device("cuda", 0)
    gen = torch.Generator(device=device).manual_seed(5)
    bad = []
    for shape, tactic, rows in _tabled(tp):
        weight = (torch.randn(shape, generator=gen, device=device) * 0.02).bfloat16()
        sample = sorted({rows[0], rows[-1], *rows[:: max(1, len(rows) // 5)]})
        prepared = PreparedProjection(weight, {r: tactic for r in sample}).prepare()
        for r in sample:
            x = torch.randn((r, shape[1]), generator=gen, device=device).bfloat16()
            out = torch.empty((r, shape[0]), dtype=torch.bfloat16, device=device)
            prepared.run(x, out)
            ref = F.linear(x.float(), weight.float())
            err = (out.float() - ref).abs().max().item()
            scale = ref.abs().max().item()
            if not err <= 2e-2 * scale:
                bad.append((shape, tactic, r, err, scale))
    torch.accelerator.synchronize(device)
    assert not bad, bad[:5]


@pytest.mark.skipif(not _rubin(), reason="Rubin (SM107) only")
def test_combine_normalizes_as_stock():
    """``K3GqaDSparkModel.combine``'s stock path (fc, then ops.rms_norm with
    hidden_norm) equals the stock Qwen3 DSpark context step's (fc in
    combine_hidden_states, the same rms_norm in precompute_and_store_context_kv)."""
    import vllm._custom_ops as ops
    from vllm.models.kimi_k3.nvidia.dspark_gqa import K3GqaDSparkModel

    device = torch.device("cuda", 0)
    gen = torch.Generator(device=device).manual_seed(11)
    fc = torch.randn((HIDDEN, 5 * HIDDEN), generator=gen, device=device) * 0.01
    fc = fc.bfloat16()
    gamma = (torch.rand(HIDDEN, generator=gen, device=device) + 0.5).bfloat16()
    inner = SimpleNamespace(
        _fast_context=None,
        fc=lambda x: F.linear(x, fc),
        hidden_norm=SimpleNamespace(
            weight=torch.nn.Parameter(gamma, requires_grad=False),
            variance_epsilon=EPS,
        ),
    )
    inner.fc.params_dtype = torch.bfloat16
    inner.normalize = lambda rows: K3GqaDSparkModel.normalize(inner, rows)
    for rows in (8, 56, 640):
        aux = torch.randn((rows, 5 * HIDDEN), generator=gen, device=device)
        aux = aux.bfloat16()
        combined = K3GqaDSparkModel.combine(inner, aux)
        projected = F.linear(aux, fc)
        stock = torch.empty_like(projected)
        ops.rms_norm(stock, projected, gamma, EPS)
        assert torch.equal(combined, stock), rows


# ---------------------------------------------------------------- torchrun


@pytest.mark.parametrize(
    "part", ["sites", "sites4", "vpmarkov", "vpmarkov1", "build", "gather"]
)
def test_gpu_torchrun(part):
    """The torchrun checks below on every local GPU (skipped without 2+ Rubin
    GPUs)."""
    import subprocess

    world = _rubin_world()
    if not world:
        pytest.skip("needs 2+ Rubin GPUs (multicast memory)")
    command = [sys.executable, "-m", "torch.distributed.run"]
    command += [f"--nproc-per-node={world}", __file__, part]
    subprocess.run(command, check=True, timeout=3600)


class _Linear(torch.nn.Module):
    """A BF16 linear with vLLM's (out, bias) call and a swappable quant method."""

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.bias = None
        self.quant_method = SimpleNamespace(
            apply=lambda layer, x, bias=None: F.linear(x, layer.weight)
        )

    def forward(self, x):
        return self.quant_method.apply(self, x), None


class _Norm:
    def __init__(self, weight):
        self.weight, self.variance_epsilon = weight, EPS


def _standin(qkv, q_size):
    """The stand-in attention core: [rows, q_size] from the qkv rows."""
    return (qkv[:, :q_size].float() * 0.5 + qkv[:, -q_size:].float()).bfloat16()


class _Mlp:
    """Qwen2MLP's surface: gate_up_proj, act_fn (SiluAndMul), down_proj."""

    def __init__(self, gate_up, down, act_fn):
        self.gate_up_proj, self.down_proj = _Linear(gate_up), _Linear(down)
        self.act_fn = act_fn

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)[0]))[0]


class _Layer:
    """The K3GqaDSparkDecoderLayer surface GqaSites / PreparedDraftAr read, at the
    release drafter's per-rank GQA geometry of ``tp`` (runtime.draft.gqa_shapes), with
    the stand-in attention core."""

    def __init__(self, index, gen, device, act_fn, tp=16):
        from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft

        def w(shape):
            n, k = shape
            return (torch.randn((n, k), generator=gen, device=device) * 0.02).bfloat16()

        def g():
            return (torch.rand(HIDDEN, generator=gen, device=device) + 0.5).bfloat16()

        shapes = draft.gqa_shapes(tp)
        self.layer_idx = index
        self.q_size = shapes["o_proj"][1]
        self.input_layernorm, self.post_attention_layernorm = _Norm(g()), _Norm(g())
        self.self_attn = SimpleNamespace(
            qkv_proj=_Linear(w(shapes["qkv"])),
            o_proj=_Linear(w(shapes["o_proj"])),
            attention_sink_bias=None,
            v_scale=None,
        )
        self.mlp = _Mlp(w(shapes["gate_up"]), w(shapes["down"]), act_fn)

    def attention_core(self, positions, hidden):
        qkv, _ = self.self_attn.qkv_proj(hidden)
        return _standin(qkv, self.q_size)

    def attention(self, positions, hidden):
        return self.self_attn.o_proj(self.attention_core(positions, hidden))[0]


def _rms(x, weight, dtype=torch.bfloat16):
    """RMSNorm of ``x`` (FP32 statistics), rounded once to ``dtype``."""
    xf = x.float()
    scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return (xf * scale * weight.float()).to(dtype)


def _all_sum(partial, world, dtype=torch.bfloat16):
    """The FP32 rank-order sum of every rank's partial, rounded once to ``dtype``."""
    parts = [torch.empty_like(partial) for _ in range(world)]
    torch.distributed.all_gather(parts, partial.contiguous())
    total = parts[0].float()
    for other in parts[1:]:
        total = total + other.float()
    return total.to(dtype)


def _reference(model, input_ids, embedding, world, dtype=torch.bfloat16):
    """The stock drafter forward's math (sums and roundings as FlashInfer's fused
    all-reduce + RMSNorm), every rank's partials exchanged with all_gather;
    ``dtype`` FP32: the exact forward (no BF16 rounding between the ops; the BF16
    weights and the stand-in core's inputs aside)."""
    residual = embedding[input_ids.long()].to(dtype)
    layers = model.layers
    hidden = _rms(residual, layers[0].input_layernorm.weight, dtype)

    def linear(x, weight):
        return F.linear(x.to(dtype), weight.to(dtype))

    for index, layer in enumerate(layers):
        if index > 0:
            residual = _all_sum(hidden, world, dtype).float() + residual.float()
            residual = residual.to(dtype)
            hidden = _rms(residual, layer.input_layernorm.weight, dtype)
        qkv = linear(hidden, layer.self_attn.qkv_proj.weight)
        attn = _standin(qkv, layer.q_size).to(dtype)
        if dtype == torch.float32:  # (the stand-in's own BF16 rounding is not stock's)
            attn = qkv[:, : layer.q_size] * 0.5 + qkv[:, -layer.q_size :]
        partial = linear(attn, layer.self_attn.o_proj.weight)
        residual = _all_sum(partial, world, dtype).float() + residual.float()
        residual = residual.to(dtype)
        normed = _rms(residual, layer.post_attention_layernorm.weight, dtype)
        gate_up = linear(normed, layer.mlp.gate_up_proj.weight)
        if dtype == torch.float32:
            half = gate_up.shape[-1] // 2
            act = F.silu(gate_up[:, :half]) * gate_up[:, half:]
        else:
            act = layer.mlp.act_fn(gate_up)
        hidden = linear(act, layer.mlp.down_proj.weight)
    residual = (_all_sum(hidden, world, dtype).float() + residual.float()).to(dtype)
    return _rms(residual, model.norm.weight, dtype)


def _rel(out, exact):
    """Mean relative distance of ``out`` to the exact forward."""
    diff = (out.float() - exact).abs().mean()
    return (diff / exact.abs().mean()).item()


SITE_ROWS = (6, 7, 9, 14, 21, 28, 56, 63, 70, 84, 112, 128, 224)


def _sites_main(rank, world, device, group, tp=16):
    """GqaSites (GqaDraftFastPath's site build with the TP``tp`` GQA tables, at the
    TP``tp`` per-rank geometry, over the local world) in a CUDA graph vs the stock
    math, at one-shot (6, 7: ring producers, polled sites), two-shot (9-128: pushing
    producers, SiLU-fused gate/up; 65-128 over the wide box; stored partials where
    the tables have none) and wide (224: ArNormRows over stored partials) rows;
    three replays bitwise equal."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft.allreduce import (
        PreparedDraftAr,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft

    # The TP``tp`` tables at the local world (the per-rank shapes are TP``tp``'s).
    for name in dir(draft):
        table = getattr(draft, name)
        if name.startswith("GQA_") and isinstance(table, dict):
            if tp in table:
                table[world] = table[tp]
            else:
                table.pop(world, None)
    from vllm.config import CompilationConfig

    # The CUDA silu_and_mul kernel (as the served drafter's), not a torch.compiled
    # native op (which cannot compile inside the capture).
    config = VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))
    with set_current_vllm_config(config):
        act = SiluAndMul()
    gen = torch.Generator(device=device).manual_seed(100 + rank)
    layers = [_Layer(i, gen, device, act, tp) for i in range(5)]
    shared = torch.Generator(device=device).manual_seed(7)
    vocab = 512 * world
    embedding = (
        torch.randn((vocab, HIDDEN), generator=shared, device=device)
    ).bfloat16()
    final = (torch.rand(HIDDEN, generator=shared, device=device) + 0.5).bfloat16()
    shard = vocab // world
    model = SimpleNamespace(
        layers=layers,
        norm=_Norm(final),
        embed_tokens=SimpleNamespace(
            weight=embedding[rank * shard : (rank + 1) * shard].contiguous()
        ),
        has_separate_mask_embedding=False,
    )
    rows = list(SITE_ROWS)
    steps, top = 7, max(SITE_ROWS)
    policy = lambda table, r: draft.covered(  # noqa: E731
        draft.closed(table.get(world, ""), top), r, steps
    )
    # The split-K projections of the GQA tables on the layers' linears.
    wrapper = SimpleNamespace(model=SimpleNamespace(layers=layers))
    draft.install_gqa_projections(wrapper, world, rows, [], rank)
    ar = PreparedDraftAr(
        group, device, world, rank, rows, vocab,
        {r for r in rows if policy(draft.GQA_FAST_AR, r) == 2},
        {r for r in rows if policy(draft.GQA_GATED_AR, r)},
        {r for r in rows if policy(draft.GQA_TWO_SHOT_EARLY, r)},
        {r for r in rows if policy(draft.GQA_VALID_AR, r)},
        {r for r in rows if policy(draft.GQA_FAST_AR, r) == 3},
    )  # fmt: skip
    draft.GqaDraftFastPath.prepare_producers(ar, layers, world, policy)
    sites = draft.GqaSites(ar)
    if rank == 0:
        print(
            f"sites rows {ar.rows}: polled {sorted(ar.polled)}, two-shot "
            f"{sorted(ar.two_rows)}, wide {sorted(ar.wide_rows)}, producers "
            f"{len(ar.producers)} ring / {len(ar.two_producers)} two-shot",
            flush=True,
        )
    failures = 0
    for r in rows:
        ids_gen = torch.Generator(device=device).manual_seed(1000 + r)
        input_ids = torch.randint(
            0, vocab, (r,), generator=ids_gen, device=device, dtype=torch.int32
        )
        positions = torch.zeros(r, dtype=torch.int64, device=device)
        torch.distributed.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = sites.run(model, input_ids, positions)
        outs = []
        for _ in range(3):
            torch.accelerator.synchronize(device)
            torch.distributed.barrier()
            graph.replay()
            torch.accelerator.synchronize(device)
            outs.append(out.clone())
        stock = _reference(model, input_ids, embedding, world)
        exact = _reference(model, input_ids, embedding, world, torch.float32)
        path = (
            ("polled" if r in ar.polled else "stored")
            + (" two-shot" if r in ar.two_rows else "")
            + (" wide" if r in ar.wide_rows else "")
        )
        # The specialized forward is as close to the exact forward as the stock
        # BF16 one (the same roundings; its split-K / producer GEMMs reorder FP32
        # sums), and close to the stock one.
        fast_err, stock_err = _rel(outs[0], exact), _rel(stock, exact)
        to_stock = _rel(outs[0], stock.float())
        ok = fast_err <= 1.25 * stock_err + 1e-4 and to_stock < 2e-2
        if not all(torch.equal(outs[0], o) for o in outs[1:]):
            print(f"rank {rank}: rows {r} ({path}): replays differ", flush=True)
            failures += 1
        if not ok:
            failures += 1
        if rank == 0 or not ok:
            print(
                f"rank {rank}: rows {r} ({path}): vs exact {fast_err:.2e} (stock "
                f"{stock_err:.2e}), vs stock {to_stock:.2e} {'ok' if ok else 'FAIL'}",
                flush=True,
            )
        del graph
    return failures


def _stock_markov(w1, w2, base, anchors, steps, sampling=None, shards=1):
    """The stock sequential Markov loop (DSparkSpeculator._sample_sequential): logits
    = base + F.linear(markov_w1[prev], markov_w2) (BF16), then torch.argmax or the
    stock ``gumbel_sample`` (drafting salt, position - 1, the cache column).
    ``shards``: the bias GEMM per vocabulary shard (each shard's cuBLAS columns as a
    TP rank computes them; 1: the stock full-vocabulary GEMM)."""
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    prev, out = anchors.long(), []
    cols = torch.arange(steps, dtype=torch.int32, device=base.device)
    width = w2.shape[0] // shards
    for i in range(steps):
        emb = w1[prev]
        bias = torch.cat(
            [F.linear(emb, w2[s * width : (s + 1) * width]) for s in range(shards)],
            dim=-1,
        )
        logits = base[:, i] + bias
        if sampling is None:
            prev = torch.argmax(logits, dim=-1)
        else:
            temperature, seeds, slots, positions, cache = sampling
            batch = anchors.shape[0]
            prev = gumbel_sample(
                logits,
                slots.view(batch, steps)[:, i].contiguous(),
                temperature,
                seeds,
                positions.view(batch, steps)[:, i] - 1,
                apply_temperature=True,
                is_drafting=True,
                logits_cache=cache,
                logits_cache_col=cols[i],
            )
        out.append(prev)
    return torch.stack(out, dim=1)


def _vp_markov_main(rank, world, device, group, steps=3):
    """VocabParallelMarkov (the TP4 drafter's Markov sampling, here over the local
    world) against the stock loop at Markov rank 512 and the real vocabulary: the
    drafts bitwise (greedy and sampled; a fifth of the requests at temperature 0) and,
    sampled, this rank's columns of every live request's cache rows bitwise (no
    other column written), eager and in a CUDA graph."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft

    vocab, rank_dim = 163840, 512
    gen = torch.Generator(device=device).manual_seed(4242)
    w1 = torch.randn((vocab, rank_dim), generator=gen, device=device) * 0.05
    w1 = w1.bfloat16()
    w2 = torch.randn((vocab, rank_dim), generator=gen, device=device) * 0.05
    w2 = w2.bfloat16()

    def gather(x):
        out = torch.empty((world * x.shape[0],), dtype=x.dtype, device=x.device)
        torch.distributed.all_gather_into_tensor(out, x, group=group)
        return out

    # The NCCL all-gather of the keys (explicit gather) and the peer mailbox; one
    # draft position (DSpark k1) runs the gather only (GqaDraftFastPath.vp_markov:
    # the mailbox needs >= 2 positions).
    variants = [
        ("gather", draft.VocabParallelMarkov(w1, w2, world, rank, steps, gather)),
    ]
    if steps > 1:
        variants.append(
            (
                "mailbox",
                draft.VocabParallelMarkov(
                    w1, w2, world, rank, steps, group=group, cap=128
                ),
            )
        )
    failures = 0
    for variant, markov in variants:
        lo, width = rank * markov.vocab_rank, markov.vocab_rank
        for batch in (2, 7, 50, 128):
            for sampled in (False, True):
                markov.prepare(batch, steps, steps, sampled)
                base = torch.randn((batch, steps, vocab), generator=gen, device=device)
                base = base.bfloat16()
                anchors = torch.randint(
                    0, vocab, (batch,), generator=gen, device=device, dtype=torch.int32
                )
                max_reqs = batch + 3
                temperature = torch.rand((max_reqs,), generator=gen, device=device) + 0.5
                temperature[::5] = 0.0
                seeds = torch.randint(
                    0, 1 << 62, (max_reqs,), generator=gen, device=device,
                    dtype=torch.int64,
                )  # fmt: skip
                slots = torch.randperm(max_reqs, generator=gen, device=device)[:batch]
                slots = slots.repeat_interleave(steps).contiguous()
                start = torch.randint(
                    1, 1 << 20, (batch,), generator=gen, device=device, dtype=torch.int64
                )
                positions = start[:, None] + torch.arange(steps, device=device)
                positions = positions.reshape(-1).contiguous()
                expected_cache = sampling = expected_sampling = None
                if sampled:
                    cache = torch.zeros(
                        (max_reqs, steps, vocab), dtype=torch.bfloat16, device=device
                    )
                    expected_cache = cache.clone()
                    sampling = (temperature, seeds, slots, positions, cache)
                    expected_sampling = (
                        temperature,
                        seeds,
                        slots,
                        positions,
                        expected_cache,
                    )
                # The stock math with each rank's bias columns as its shard GEMM gives
                # them (bitwise target), and the stock full-vocabulary GEMM (the drafts
                # may differ only through cuBLAS's per-shape accumulation order: logged).
                expected = _stock_markov(
                    w1, w2, base, anchors, steps, expected_sampling, shards=world
                )
                stock_cache = None if not sampled else torch.zeros_like(expected_cache)
                stock_sampling = None if not sampled else (
                    temperature, seeds, slots, positions, stock_cache
                )  # fmt: skip
                stock = _stock_markov(w1, w2, base, anchors, steps, stock_sampling)
                if rank == 0:
                    n = (stock != expected).sum().item()
                    print(
                        f"B{batch} sampled={sampled}: per-shard vs full-vocabulary bias "
                        f"GEMM drafts differ at {n} of {stock.numel()}",
                        flush=True,
                    )
                local = base[:, :, lo : lo + width].reshape(batch * steps, width)
                local = local.contiguous()
                for mode in ("eager", "graph"):
                    draft_out = torch.zeros(
                        (batch, steps), dtype=torch.int64, device=device
                    )
                    if sampled:
                        sampling[4].zero_()
                    torch.distributed.barrier()
                    if mode == "eager":
                        markov.run(local, anchors, draft_out, sampling)
                    else:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            markov.run(local, anchors, draft_out, sampling)
                        torch.accelerator.synchronize(device)
                        torch.distributed.barrier()
                        graph.replay()
                    torch.accelerator.synchronize(device)
                    label = f"B{batch} {'sampled' if sampled else 'greedy'} {mode}"
                    if not torch.equal(draft_out, expected):
                        n = (draft_out != expected).sum().item()
                        print(f"rank {rank}: {label}: {n} drafts differ", flush=True)
                        failures += 1
                    if sampled:
                        cache, live = sampling[4], slots[::steps]
                        cols = slice(lo, lo + width)
                        if not torch.equal(
                            cache[live, :, cols], expected_cache[live, :, cols]
                        ):
                            print(f"rank {rank}: {label}: cached logits differ", flush=True)
                            failures += 1
                        rest = cache.clone()
                        rest[live, :, cols] = 0
                        if rest.any():
                            print(
                                f"rank {rank}: {label}: cache written elsewhere", flush=True
                            )
                            failures += 1
                    if rank == 0:
                        print(f"VocabParallelMarkov {variant} {label}: checked", flush=True)
    return failures


def _build_main(rank, world, device, group):
    """``GqaDraftFastPath.build`` end to end over the local world (non-strict: TP2),
    on stand-in layers at the TP16 per-rank geometry with the release Markov head
    shapes and a vocabulary-parallel LM head: the TP16 tables at the local world
    (the context step dropped: CtxGather needs TP >= 4; Markov sampling on
    VocabParallelMarkov), then ``sampled_markov_draft``'s path
    (``DraftFastPath.greedy_markov``: the LM-head GEMV / shard GEMM and the Markov
    kernels) in a CUDA graph, against the stock probabilistic loop on the full
    logits."""
    from vllm.config import (
        CompilationConfig,
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft

    for name in dir(draft):
        table = getattr(draft, name)
        if name.startswith("GQA_") and isinstance(table, dict) and 16 in table:
            table[world] = table[16]
    draft.GQA_FAST_CONTEXT.pop(world, None)
    draft.GQA_FUSED_MARKOV.pop(world, None)
    draft.GQA_VP_MARKOV[world] = "1-:1"
    steps, vocab, sizes = 7, 163840, [7, 14, 21, 56]
    config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["all"]),
        parallel_config=ParallelConfig(tensor_parallel_size=world),
    )
    failures = 0
    with set_current_vllm_config(config):
        init_distributed_environment(world, rank, local_rank=device.index)
        initialize_model_parallel(tensor_model_parallel_size=world)
        act = SiluAndMul()
        gen = torch.Generator(device=device).manual_seed(300 + rank)
        shared = torch.Generator(device=device).manual_seed(301)
        layers = [_Layer(i, gen, device, act) for i in range(5)]
        with torch.device(device):
            lm_head = ParallelLMHead(vocab, HIDDEN).bfloat16()
        lm_head.weight.data.normal_(0.0, 0.02, generator=gen)
        markov = [
            torch.randn((vocab, 512), generator=shared, device=device).bfloat16() * 0.05
            for _ in range(2)
        ]
        embedding = torch.randn((vocab, HIDDEN), generator=shared, device=device)
        shard = vocab // world
        inner = SimpleNamespace(
            layers=layers,
            norm=_Norm((torch.rand(HIDDEN, device=device) + 0.5).bfloat16()),
            fc=SimpleNamespace(weight=torch.zeros((HIDDEN, 2), device=device)),
            hidden_norm=_Norm(torch.ones(HIDDEN, device=device).bfloat16()),
            markov_head=SimpleNamespace(
                markov_w1=SimpleNamespace(weight=markov[0]),
                markov_w2=SimpleNamespace(weight=markov[1]),
            ),
            has_separate_mask_embedding=False,
            _fused_kv_weight=(
                torch.randn((640, HIDDEN), generator=gen, device=device) * 0.02
            ).bfloat16(),
            embed_tokens=SimpleNamespace(
                weight=embedding[rank * shard : (rank + 1) * shard].bfloat16()
            ),
            _fast_ar=None,
            _fast_context=None,
            _fast_context_kv=None,
        )
        model = SimpleNamespace(
            model=inner,
            config=SimpleNamespace(
                hidden_size=HIDDEN,
                sample_from_anchor=True,
                vocab_size=vocab,
                draft_vocab_size=vocab,
            ),
            logits_processor=LogitsProcessor(vocab),
            lm_head=lm_head,
        )
        vllm_config = SimpleNamespace(
            additional_config={"decode_backend": "latency_specialized"},
            speculative_config=SimpleNamespace(
                method="dspark",
                num_speculative_tokens=steps,
                draft_sample_method="probabilistic",
            ),
            compilation_config=SimpleNamespace(
                cudagraph_capture_sizes=sizes, max_cudagraph_capture_size=max(sizes)
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=8),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=world,
                decode_context_parallel_size=1,
                pipeline_parallel_size=1,
                enable_expert_parallel=False,
            ),
            model_config=SimpleNamespace(use_fp64_gumbel=False),
            watermark_config=None,
        )
        # The stand-in layers have no stock attention modules for the fused GQA
        # attention's installer (covered by test_dspark_gqa_attention_int.py): this
        # non-strict build keeps the stand-in attention core.
        install = draft.install_gqa_attention_state
        draft.install_gqa_attention_state = lambda *args, **kwargs: None
        try:
            fast = draft.GqaDraftFastPath.build(vllm_config, model)
        finally:
            draft.install_gqa_attention_state = install
        built = (
            fast is not None
            and isinstance(fast.markov, draft.VocabParallelMarkov)
            and sorted(fast.batches) == [1, 2, 3, 8]
            and isinstance(inner._fast_ar, draft.GqaSites)
            and fast.attention is None
        )
        if rank == 0:
            print(
                f"build: markov {type(fast.markov).__name__}, batches "
                f"{sorted(fast.batches)}, sites {inner._fast_ar.rows}, LM-head GEMV "
                f"{sorted(fast.lm_batches)} {'ok' if built else 'FAIL'}",
                flush=True,
            )
        failures += not built
        for batch in (1, 8):
            rows = batch * steps
            hidden = torch.randn((rows, HIDDEN), generator=shared, device=device)
            hidden = hidden.bfloat16()
            anchors = torch.randint(
                0, vocab, (batch,), generator=shared, device=device, dtype=torch.int32
            )
            temperature = torch.rand(batch, generator=shared, device=device) + 0.5
            seeds = torch.arange(batch, dtype=torch.int64, device=device) * 7919
            slots = torch.arange(batch, device=device).repeat_interleave(steps)
            positions = torch.arange(rows, dtype=torch.int64, device=device) + 11
            cache = torch.zeros(
                (batch, steps, vocab), dtype=torch.bfloat16, device=device
            )
            out = torch.zeros((batch, steps), dtype=torch.int64, device=device)
            sampling = (temperature, seeds, slots, positions, cache)
            assert not fast.greedy_markov(model, hidden, anchors, out, sampling)
            torch.distributed.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ran = fast.greedy_markov(model, hidden, anchors, out, sampling)
            torch.accelerator.synchronize(device)
            torch.distributed.barrier()
            graph.replay()
            torch.accelerator.synchronize(device)
            base = model.logits_processor._get_logits(hidden, lm_head, None)
            expected_cache = torch.zeros_like(cache)
            expected = _stock_markov(
                markov[0], markov[1], base.view(batch, steps, vocab), anchors, steps,
                (temperature, seeds, slots, positions, expected_cache), shards=world,
            )  # fmt: skip
            match = (out == expected).float().mean().item()
            ok = ran and match >= 0.9
            failures += not ok
            if rank == 0 or not ok:
                print(
                    f"rank {rank}: build B{batch}: fused path {ran}, drafts equal to "
                    f"the stock loop on cuBLAS logits {match:.3f} "
                    f"{'ok' if ok else 'FAIL'}",
                    flush=True,
                )
            del graph
    return failures


def _gather_main(rank, world, device, group):
    """The context step's mailbox-publishing shard GEMM (``GatherProjection``, as
    ``gqa_context_projection`` installs it) at the TP16 fc shard (448, 35840) with
    the GQA tables' tactics (N tiles of 8 and 16 rows, 8-32 rows): every rank's rows
    land in its columns of every rank's row mailbox, bitwise the split-K
    projection's (sanitized: no -0.0), the other columns untouched."""
    import torch.distributed._symmetric_memory as symm_mem

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import context
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (
        PreparedProjection,
    )

    hidden, shard = context.HIDDEN, 448
    gen = torch.Generator(device=device).manual_seed(900 + rank)
    weight = torch.randn((shard, 5 * hidden), generator=gen, device=device) * 0.01
    weight = weight.bfloat16()
    failures = 0
    cases = {
        8: (64, 8, 14, 10),
        12: (64, 16, 14, 8),
        16: (64, 16, 14, 8),
        24: (64, 16, 10, 8),
        32: (64, 16, 10, 8),
    }
    box = symm_mem.empty((64 * hidden,), dtype=torch.bfloat16, device=device)
    mc = int(symm_mem.rendezvous(box, group).multicast_ptr or 0)
    assert mc, "multicast memory"
    ring = context.GatherProjection(weight, cases).prepare()
    plain = PreparedProjection(weight, cases).prepare()
    shape_rows = torch.empty((64, shard), dtype=torch.bfloat16, device=device)
    for rows, tactic in cases.items():
        box.view(torch.int32).fill_(-0x80000000)
        torch.accelerator.synchronize(device)
        torch.distributed.barrier()
        x = torch.randn((rows, 5 * hidden), generator=gen, device=device).bfloat16()
        ring.run_ring(x, shape_rows[:rows], mc + rank * shard * 2)
        expected = torch.empty((rows, shard), dtype=torch.bfloat16, device=device)
        plain.run(x, expected)
        expected = expected + 0.0  # (-0.0 published as +0.0)
        torch.accelerator.synchronize(device)
        torch.distributed.barrier()
        landed = box.view(64, hidden)[:rows]
        parts = [torch.empty_like(expected) for _ in range(world)]
        torch.distributed.all_gather(parts, expected)
        ok = all(
            torch.equal(landed[:, r * shard : (r + 1) * shard], parts[r])
            for r in range(world)
        )
        rest = landed[:, world * shard :].view(torch.int32)
        ok = ok and bool((rest == -0x80000000).all())
        failures += not ok
        if rank == 0 or not ok:
            print(
                f"rank {rank}: GatherProjection rows {rows} tactic {tactic}: "
                f"{'ok' if ok else 'FAIL'}",
                flush=True,
            )
    return failures


def _gpu_main(parts):
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    device = torch.device("cuda", torch.cuda.current_device())
    group = dist.group.WORLD
    failures = 0
    if "sites" in parts:
        failures += _sites_main(rank, world, device, group)
    if "sites4" in parts:
        failures += _sites_main(rank, world, device, group, tp=4)
    if "vpmarkov" in parts:
        failures += _vp_markov_main(rank, world, device, group)
    if "vpmarkov1" in parts:
        failures += _vp_markov_main(rank, world, device, group, steps=1)
    if "build" in parts:
        failures += _build_main(rank, world, device, group)
    if "gather" in parts:
        failures += _gather_main(rank, world, device, group)
    dist.barrier()
    dist.destroy_process_group()
    return failures


if __name__ == "__main__":
    # torchrun ... test_dspark_gqa_fastpath_gpu.py [sites] [sites4] [vpmarkov]
    # [build] [gather] (default: all)
    parts = sys.argv[1:] or ("sites", "sites4", "vpmarkov", "build", "gather")
    sys.exit(1 if _gpu_main(parts) else 0)
