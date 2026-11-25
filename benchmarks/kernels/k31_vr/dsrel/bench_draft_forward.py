# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The release DSpark drafter's forward (5 GQA layers at the TP16 per-rank geometry:
qkv (512, 7168), o_proj (7168, 384), gate_up (1792, 7168), down (7168, 896)) over
the local TP group, stock vs specialized, in CUDA graphs (torchrun, one process per
GPU; Rubin):

  stock: the K3GqaDSparkModel stock forward: the vocabulary-parallel embedding
         (local gather + all-reduce), cuBLAS linears, o_proj / down_proj partials
         all-reduced by FlashInfer's fused all-reduce + residual + RMSNorm
         (``fused_allreduce_rms_norm``), the final norm;
  spec:  ``runtime.draft.GqaSites`` (GqaDraftFastPath's build with the TP16 GQA
         tables): the ring embedding site, split-K projections, ring / two-shot
         o_proj and down producers, SiLU-fused gate/up, the Lamport all-reduce +
         norm sites (one-shot / two-shot / ArNormRows).

Both with the same stand-in attention core (the attention kernel is
cutedsl.draft.gqa_*'s, timed by its own bench), so the difference is the
projections + all-reduce chain. Per row count: the median replay time, the max over
ranks. At fewer than 16 ranks the exchanges are cheaper than at TP16 (indicative).

    torchrun --nproc-per-node 2 bench_draft_forward.py --out DIR [--rows 7,14,...]
    torchrun --nnodes 4 --nproc-per-node 4 ... (16 ranks)
"""

import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F

HIDDEN = 7168
EPS = 1e-5
ROWS = (6, 7, 9, 14, 21, 28, 35, 42, 49, 56, 63, 70, 84, 112, 168, 224, 336, 448)


def log(msg):
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Linear(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.bias = None
        self.quant_method = type(
            "Base",
            (),
            {"apply": lambda s, layer, x, bias=None: F.linear(x, layer.weight)},
        )()

    def forward(self, x):
        return self.quant_method.apply(self, x), None


class Mlp:
    def __init__(self, gate_up, down, act_fn):
        self.gate_up_proj, self.down_proj, self.act_fn = (
            Linear(gate_up),
            Linear(down),
            act_fn,
        )

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)[0]))[0]


class Layer:
    def __init__(self, index, gen, device, act_fn, norm_cls):
        def w(n, k):
            return (torch.randn((n, k), generator=gen, device=device) * 0.02).bfloat16()

        def g():
            norm = norm_cls(HIDDEN, eps=EPS).to(device)
            norm.weight.data.copy_(
                torch.rand(HIDDEN, generator=gen, device=device) + 0.5
            )
            return norm

        self.layer_idx, self.q_size = index, 384
        self.input_layernorm, self.post_attention_layernorm = g(), g()
        self.self_attn = type("Attn", (), {})()
        self.self_attn.qkv_proj = Linear(w(512, HIDDEN))
        self.self_attn.o_proj = Linear(w(HIDDEN, self.q_size))
        self.mlp = Mlp(w(1792, HIDDEN), w(HIDDEN, 896), act_fn)

    def attention_core(self, positions, hidden):
        qkv, _ = self.self_attn.qkv_proj(hidden)
        return qkv[:, : self.q_size].contiguous()

    def attention(self, positions, hidden):
        return self.self_attn.o_proj(self.attention_core(positions, hidden))[0]


def stock_forward(model, input_ids, rank, shard):
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.models.common.ops.fused_allreduce_rms_norm import (
        fused_allreduce_rms_norm,
    )

    local = input_ids.long() - rank * shard
    mask = (local >= 0) & (local < shard)
    emb = F.embedding(local.clamp(0, shard - 1), model.embed_tokens.weight)
    emb = tensor_model_parallel_all_reduce(emb * mask.unsqueeze(-1).to(emb.dtype))
    residual = emb
    hidden = model.layers[0].input_layernorm(emb)
    for index, layer in enumerate(model.layers):
        if index > 0:
            hidden, residual = fused_allreduce_rms_norm(
                hidden, residual, layer.input_layernorm
            )
        attn = layer.attention(None, hidden)
        hidden, residual = fused_allreduce_rms_norm(
            attn, residual, layer.post_attention_layernorm
        )
        hidden = layer.mlp(hidden)
    out, _ = fused_allreduce_rms_norm(hidden, residual, model.norm)
    return out


def time_graph(fn, device, iters=40, flush=None):
    fn()  # eager warmup (the FlashInfer all-reduce workspace is created lazily)
    torch.accelerator.synchronize(device)
    torch.distributed.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(iters):
        torch.accelerator.synchronize(device)
        if flush is not None:  # --flush-mb: every weight cold (the target forward's place)
            flush.zero_()
            torch.accelerator.synchronize(device)
        torch.distributed.barrier()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    times.sort()
    median = torch.tensor([times[len(times) // 2]], device=device)
    torch.distributed.all_reduce(median, op=torch.distributed.ReduceOp.MAX)
    del graph
    return median.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=",".join(map(str, ROWS)))
    # weight-exposure probe: --flush-mb N memsets N MB before every replay (cold
    # weights); --shared-weights gives the 5 layers layer 0's weights (51 MB/rank,
    # L2-resident across replays: hot)
    ap.add_argument("--flush-mb", type=int, default=0)
    ap.add_argument("--shared-weights", action="store_true")
    ap.add_argument("--spec-only", action="store_true")
    args = ap.parse_args()
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft.allreduce import (
        PreparedDraftAr,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world))
    rows = [int(r) for r in args.rows.split(",")]
    with set_current_vllm_config(config):
        init_distributed_environment(world, rank, local_rank=local, backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=world)
        group = get_tp_group().device_group
        for name in dir(draft):
            table = getattr(draft, name)
            if name.startswith("GQA_") and isinstance(table, dict) and 16 in table:
                table[world] = table[16]
        act = SiluAndMul()
        gen = torch.Generator(device=device).manual_seed(100 + rank)
        layers = [Layer(i, gen, device, act, RMSNorm) for i in range(5)]
        if args.shared_weights:
            for layer in layers[1:]:
                for mod, src in (
                    (layer.self_attn.qkv_proj, layers[0].self_attn.qkv_proj),
                    (layer.self_attn.o_proj, layers[0].self_attn.o_proj),
                    (layer.mlp.gate_up_proj, layers[0].mlp.gate_up_proj),
                    (layer.mlp.down_proj, layers[0].mlp.down_proj),
                ):
                    mod.weight = src.weight
        flush = None
        if args.flush_mb:
            flush = torch.empty(args.flush_mb << 20, dtype=torch.uint8, device=device)
        shared = torch.Generator(device=device).manual_seed(7)
        vocab = 1024 * world
        table = torch.randn((vocab, HIDDEN), generator=shared, device=device).bfloat16()
        shard = vocab // world
        final = RMSNorm(HIDDEN, eps=EPS).to(device)
        model = type("Model", (), {})()
        model.layers, model.norm = layers, final
        model.has_separate_mask_embedding = False
        model.embed_tokens = type("Embed", (), {})()
        model.embed_tokens.weight = table[
            rank * shard : (rank + 1) * shard
        ].contiguous()
        wrapper = type("Wrapper", (), {})()
        wrapper.model = type("Inner", (), {})()
        wrapper.model.layers = layers
        draft.install_gqa_projections(wrapper, world, rows, [], rank)
        top = max(rows)
        policy = lambda t, r: draft.covered(  # noqa: E731
            draft.closed(t.get(world, ""), top), r, 7
        )
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
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"forward_w{world}.csv")
        out_rows = []
        for r in rows:
            ids = torch.randint(0, vocab, (r,), generator=shared, device=device,
                                dtype=torch.int32)  # fmt: skip
            stock = float("nan")
            if not args.spec_only:
                stock = time_graph(
                    lambda ids=ids: stock_forward(model, ids, rank, shard), device,
                    flush=flush,
                )
            spec = time_graph(
                lambda ids=ids: sites.run(model, ids, None), device, flush=flush
            )
            kind = (
                ("polled" if r in ar.polled else "stored")
                + (" two-shot" if r in ar.two_rows else "")
                + (" wide" if r in ar.wide_rows else "")
            )
            out_rows.append((r, kind, stock, spec, spec / stock))
            log(f"rows {r:>4} ({kind}): stock {stock:8.1f} us, specialized "
                f"{spec:8.1f} us ({spec / stock:.3f})")  # fmt: skip
        if rank == 0:
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["rows", "sites", "stock_us", "spec_us", "ratio"])
                w.writerows(out_rows)
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
