# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Kimi K3.1 DSA layer chain on one node, as the specialized decode runs it.

torchrun --nproc-per-node 4 (TP = DCP = W = 4, one node): every rank builds a real
KimiK3DSAAttention (``--heads`` per rank, default 24: 96 gathered heads as at DCP16;
MXFP4 indexer cache, per-tensor FP8 MLA cache; index top-k ``--topk``, default 512:
every rank's share of a row's selection is 512 / W = 128 rows, a DCP16 rank's share
of 2048) and the specialized decode's PreparedDsa with the real q / partial-row
mailboxes and the #63 exact top-k merge, through the layer harness of
tests/models/kimi_k3/test_specialized_decode_dsa.py (run_layer_step's setup). Each
rank holds 1/W of every context; the contexts are ``--local`` x W tokens, so every
rank scores and attends the same number of local KV rows as a DCP16 rank at
``--local`` x 16 (default 8250 = 132K / 16). The layer's query overlap, sparse core,
partials and merge are the code's policies at the rows (front + DSA sparse core at
1-16 rows, fork + FlashInfer trtllm-gen + "proj" from 17).

One CUDA graph of ``--layers`` DSA layers alternating the two mailbox parities
(consecutive DSA layers), launched as in the step (PDL edges inside each layer, the
side / index streams joined per layer). Reports per rows: us per layer (median of
replays, max over ranks) and, with ``--prof DIR``, a CUPTI trace of 3 replays per
rank (DIR/<tag>_M<rows>.rank<r>.trace.json) for the per-kernel in-layer timeline
(lanes tooling: dsa_chain_split.py).

Differences from a DCP16 rank, by construction (defaults): the per-rank head work of
MlaQuery (q_b, W_UK, the q publish) and DcpValue (W_UV, gate) is 4x (24 heads, not
6; the q publish and the value merge move the same head-rows per rank: 24 x 3 peers
vs 6 x 15); the merge and select run toward a top-512 over W x N candidates (not a
top-2048 over 16 x 192); the exchange crosses W = 4 ranks of one node, not 16 over 4
nodes (its 16-rank latency is priced from the hwdb). The sparse core / FlashInfer
decode, the indexer front, logits and local select are a DCP16 rank's.
"""

import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
)


def init(world, rank, local, topk):
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.v1.worker.workspace import init_workspace_manager

    from tests.models.kimi_k3 import test_specialized_decode_dsa as dsa

    torch.accelerator.set_device_index(local)
    with set_current_vllm_config(dsa.layer_config("fi-trtllm", 1, world)):
        init_distributed_environment(
            world_size=world, rank=rank, local_rank=local,
            distributed_init_method="env://",
        )  # fmt: skip
        initialize_model_parallel(
            tensor_model_parallel_size=world, decode_context_model_parallel_size=world
        )
        init_workspace_manager(torch.device("cuda", local))
        from vllm.v1.attention.ops.dcp_topk_merge import init_dcp_topk_merge

        wide = dsa.layer_config("fi-trtllm", 8, world)
        with set_current_vllm_config(wide):
            n = init_dcp_topk_merge(wide, topk)
    return n


def patch(mp, layer_len, heads, topk):
    """The harness on an MXFP4 indexer cache (test_specialized_decode_dsa_fp4_index's
    mxfp4_layers), ``heads`` per rank, an index top-k of ``topk``, contexts up to
    ``layer_len``."""
    from tests.models.kimi_k3 import test_specialized_decode_dsa as dsa
    from tests.models.kimi_k3 import test_specialized_decode_dsa_fp4_index as f4
    from vllm.models.kimi_k3.nvidia.ops import dsa_indexer
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import (
        mxfp4_k_norm_quant_insert,
    )

    from vllm.transformers_utils.configs import kimi_linear

    base = kimi_linear.KimiLinearConfig

    class TopkConfig(base):  # dsa_layer's model config with this index top-k
        def __init__(self, *a, **kw):
            if kw.get("dsa_config"):
                kw["dsa_config"] = dict(kw["dsa_config"], index_topk=topk)
            super().__init__(*a, **kw)

    mp.setattr(kimi_linear, "KimiLinearConfig", TopkConfig)
    fp8_config = dsa.layer_config

    def layer_config(core, q, world=1):
        config = fp8_config(core, q, world)
        config.attention_config.indexer_kv_dtype = "mxfp4"
        config.model_config.hf_config.index_topk = topk
        text = getattr(config.model_config.hf_config, "text_config", None)
        if text is not None and hasattr(text, "index_topk"):
            text.index_topk = topk
        return config

    mp.setattr(dsa, "HEADS", heads)
    mp.setattr(dsa, "LAYER_LEN", layer_len)
    mp.setattr(dsa, "layer_config", layer_config)
    mp.setattr(dsa, "INDEX_ROW", f4.FP4_ROW)
    mp.setattr(dsa_indexer, "indexer_k_norm_quant_insert", mxfp4_k_norm_quant_insert)


def build(batch, q, world, local_len, topk):
    """run_layer_step's setup (no stock forward, no checks): the layer, its caches
    with every rank's shard of ``batch // q`` contexts of ``local_len * world``
    tokens, the packed projection rows, and two PreparedDsa (mailbox parities)."""
    from types import SimpleNamespace

    from vllm import _custom_ops as ops
    from vllm.config import set_current_vllm_config
    from vllm.distributed import get_dcp_group
    from vllm.forward_context import create_forward_context, override_forward_context
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core.decode import DsacDecode
    from vllm.models.kimi_k3.nvidia.ops import dsa_indexer
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.dcp_value import (
        DSA_PARTIALS,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.prepared import (
        PreparedDsa,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.providers import (
        dcp_box,
        pack_dsa_projection,
    )
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.v1.worker.block_table import get_block_table_width

    from tests.models.kimi_k3 import test_specialized_decode_dsa as dsa

    core = "fi-auto"
    cache_dtype, _, _ = dsa.LAYER_CORES[core]
    config = dsa.layer_config(core, q, world)
    rank = get_dcp_group().rank_in_group
    unit_rows = dsa.UNIT
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    bf = torch.bfloat16
    with set_current_vllm_config(config):
        topk_buffer = torch.full((batch, topk), -1, dtype=torch.int32, device=dev)
        attn = dsa.dsa_layer(config, topk_buffer, dsa.HEADS * world)
        layer, indexer = attn.mla_attn.mla_attn, attn.dsa_indexer
        # The model's W_UV is a latent-contiguous view of kv_b_proj (mla.py's
        # absorption), which DcpValue reads; the harness's stock layer keeps a dense copy.
        uv = layer.W_UV.detach()
        layer.W_UV.data = uv.transpose(1, 2).contiguous().transpose(1, 2)
        packed = pack_dsa_projection(attn)
        spec = layer.get_kv_cache_spec(config)
        builder = layer.attn_backend.get_builder_cls()(
            spec, [layer.layer_name], config, dev
        )
        index_spec = indexer.k_cache.get_kv_cache_spec(config)
        index_builder = DeepseekV32IndexerMetadataBuilder(
            index_spec,
            [indexer.k_cache.prefix],
            config,
            dev,
            block_table_width=get_block_table_width(
                index_spec.max_num_blocks_per_req(config, dsa.LAYER_LEN), unit_rows
            ),
        )
        reqs = batch // q
        gen = torch.Generator(device=dev).manual_seed(batch + q)
        contexts = [local_len * world - 3 * r for r in range(reqs)]
        local = [(c + q - rank + world - 1) // world for c in contexts]
        mla, mla_table = dsa.paged_rows(local, unit_rows, dsa.FP8_ROW, unit_rows, gen)
        index, index_table = dsa.paged_rows(
            local, unit_rows, dsa.INDEX_ROW, dsa.GAP, gen
        )
        old = [range(c) for c in contexts]
        old_mla = dsa.owned_slots(mla_table, old, world, rank)
        old_index = dsa.owned_slots(index_table, old, world, rank)
        n = old_mla.numel()
        unit = torch.ones(1, dtype=torch.float32, device=dev)
        for lo in range(0, n, 1 << 15):  # (chunks: the writers' scratch)
            sl = slice(lo, min(n, lo + (1 << 15)))
            m = sl.stop - sl.start
            kv_c = torch.randn(m, 512, device=dev, dtype=bf, generator=gen)
            k_pe = torch.randn(m, 64, device=dev, dtype=bf, generator=gen)
            ops.concat_and_cache_mla(kv_c, k_pe, mla, old_mla[sl], cache_dtype, unit)
            keys = torch.randn(m, 128, device=dev, dtype=bf, generator=gen)
            scratch = torch.empty(m, 2048, dtype=torch.int32, device=dev)
            norm = torch.ones(128, dtype=bf, device=dev)
            args = (norm, torch.zeros_like(norm), dsa.EPS, old_index[sl], index)
            dsa_indexer.indexer_k_norm_quant_insert(keys, *args, scratch, m)
        new = [range(c, c + q) for c in contexts]
        new_mla = dsa.owned_slots(mla_table, new, world, rank)
        new_index = dsa.owned_slots(index_table, new, world, rank)
        hidden = torch.randn(batch, dsa.LAYER_HIDDEN, device=dev, dtype=bf, generator=gen)
        names = layer.layer_name, indexer.k_cache.prefix
        layer.kv_cache, indexer.k_cache.kv_cache = mla, index
        mla_common = dsa.common_metadata(contexts, q, mla_table, new_mla, world, rank)
        index_common = dsa.common_metadata(
            contexts, q, index_table, new_index, world, rank
        )
        md = builder.build(0, mla_common)
        index_md = index_builder.build(0, index_common)
        context = override_forward_context(
            create_forward_context(
                dict(zip(names, (md, index_md))),
                config,
                slot_mapping=dict(zip(names, (new_mla, index_md.slot_mapping))),
            )
        )
        projected = torch.cat(
            [
                attn.fused_qkv_a_proj(hidden)[0],
                attn.g_proj(hidden)[0],
                indexer.wkv_index(hidden)[0],
            ],
            dim=-1,
        )
        cap = max(64, batch)
        scratch_rows = (
            torch.zeros(cap, 64, 576, device=dev, dtype=bf),
            torch.empty(cap, 1536, device=dev, dtype=bf),
            torch.empty(cap, 32 * 128, device=dev, dtype=bf),
            torch.empty(cap, 32, 128, device=dev, dtype=torch.uint8),
            torch.empty(cap, 32, device=dev, dtype=bf),
            torch.empty(cap, 64, 512, device=dev, dtype=bf),
        )
        indexer.dsac = DsacDecode(dev, dsa.HEADS * world)  # the served policy
        indexer.dsac_layer = 0
        runtime = SimpleNamespace(
            dcp=world, tp=world, device=dev, dcp_box=None, dsa_dcp_box=None
        )
        box = dcp_box(runtime, dsa.HEADS, dsa=True, flashinfer=True)
        policy = dict(queries=(q,), core="auto", dcp=box, partials=DSA_PARTIALS)
        layers = []
        for parity in (0, 1):
            p = PreparedDsa(
                attn, {batch: packed.shape[0]}, scratch_rows, **policy, parity=parity
            )
            layers.append(p.prepare_cache(mla))
        out = torch.empty(batch, dsa.HEADS * 128, device=dev, dtype=bf)
        p0 = layers[0]
        info = lambda: dict(  # noqa: E731  (inside the forward context)
            overlap=p0.overlaps[batch, q],
            core=p0.coverage_note(batch, q),
            local_topk=p0.indexer_stage.local_topk,
            merge=str(p0.indexer_stage.merge_source(batch)),
            local_kv=local[0],
        )

        def run(prepared):
            prepared.run(
                projected,
                out,
                cache=mla,
                slots=new_mla,
                index_slots=index_md.slot_mapping,
                metadata=md,
                q=q,
            )

    return SimpleNamespace(
        config=config, context=context, layers=layers, run=run, info=info
    )


DBG = {}


def patch_dsac_dbg(mp):
    """DsaCore.run with its per-CTA timestamps (dsa_core.cu ``dbg``: [CTAs, 24] int64;
    every layer overwrites them, so they hold the graph's last layer)."""
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core import ops

    run = ops.DsaCore.run

    def run_dbg(self, q_box, sel, cnt, kv, **kw):
        key = kw["rows"]
        if key not in DBG:
            DBG[key] = torch.zeros(key * 8 * 8, 24, dtype=torch.int64, device=sel.device)
        kw["dbg"] = DBG[key]
        return run(self, q_box, sel, cnt, kv, **kw)

    mp.setattr(ops.DsaCore, "run", run_dbg)


def dsac_phases(rows):
    """Per-CTA phases of the last layer's DSA sparse core (us, mean / max over CTAs)."""
    d = DBG.get(rows)
    if d is None:
        return None
    d = d[d[:, 0] > 0].double()
    if d.numel() == 0:
        return None
    t0 = d[:, 0:1]
    names = {1: "q_polled", 5: "pdl_wait", 2: "slots", 3: "loop", 6: "pre_cluster",
             7: "cluster_sync", 8: "lse", 9: "merged", 4: "end"}  # fmt: skip
    out = {"ctas": int(d.shape[0]), "start_spread_us": float((t0.max() - t0.min()) / 1e3)}
    for i, n in names.items():
        col = d[:, i]
        ok = col > 0
        if ok.any():
            rel = (col[ok] - d[ok, 0]) / 1e3
            out[n] = [round(float(rel.mean()), 3), round(float(rel.max()), 3)]
    return out


def bench(case, args, rank):
    from vllm.config import set_current_vllm_config
    from vllm.distributed import get_dcp_group
    from vllm.distributed.parallel_state import graph_capture

    group = get_dcp_group()
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    with set_current_vllm_config(case.config), case.context:
        case.info = case.info()
        for i in range(4):  # eager warm-up (compiles), both parities
            case.run(case.layers[i % 2])
        torch.accelerator.synchronize()
        group.barrier()
        g = torch.cuda.CUDAGraph()
        with graph_capture(dev) as ctx, torch.cuda.graph(g, stream=ctx.stream):
            for i in range(args.layers):
                case.run(case.layers[i % 2])
        for _ in range(3):
            g.replay()
        torch.accelerator.synchronize()
        times = []
        for _ in range(args.reps):
            group.barrier()
            torch.accelerator.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            e1.synchronize()
            times.append(e0.elapsed_time(e1) * 1000 / args.layers)
        trace = None
        if args.prof:
            from torch.profiler import ProfilerActivity, profile

            group.barrier()
            torch.accelerator.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(3):
                    group.barrier()
                    g.replay()
                torch.accelerator.synchronize()
            os.makedirs(args.prof, exist_ok=True)
            trace = os.path.join(
                args.prof, f"{args.tag}_M{args.cur_rows}.rank{rank}.trace.json"
            )
            prof.export_chrome_trace(trace)
        del g
    return statistics.median(times), trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="8,16,32,64")
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--local", type=int, default=8250)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--topk", type=int, default=512)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--prof", default="")
    ap.add_argument("--tag", default="chain")
    ap.add_argument("--out", default="")
    ap.add_argument("--dsac-dbg", action="store_true")
    args = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    import pytest

    mp = pytest.MonkeyPatch()
    patch(mp, args.local * world + 4096, args.heads, args.topk)
    if args.dsac_dbg:
        patch_dsac_dbg(mp)
    n = init(world, rank, local_rank, args.topk)
    from vllm.distributed import get_dcp_group

    group = get_dcp_group()
    res = []
    for rows in [int(x) for x in args.rows.split(",")]:
        args.cur_rows = rows
        case = build(rows, args.q, world, args.local, args.topk)
        us, trace = bench(case, args, rank)
        all_us = group.all_gather(
            torch.tensor([us], device="cuda"), dim=0
        ).tolist()
        rec = dict(rows=rows, q=args.q, world=world, local_topk_n=n,
                   us_per_layer_max=max(all_us), us_per_layer=all_us,
                   layers=args.layers, dsac=dsac_phases(rows),
                   **case.info)  # fmt: skip
        if rank == 0:
            print("REC " + json.dumps(rec), flush=True)
            res.append(rec)
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        del case
        torch.accelerator.synchronize()
        group.barrier()
    mp.undo()


if __name__ == "__main__":
    main()
