# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The drafter's DCP attention (cutedsl.draft.attention DraftDcp: Lamport q mailbox,
the drafter cores, DcpValue's LSE merge + W_UV) against the stock drafter DCP path
(tokenspeed over the gathered heads, cp_lse_ag_out_rs = LSE all-gather +
correct_attn_out + BF16 reduce-scatter, W_UV bmm; tests/models/kimi_k3/
test_specialized_decode_draft_nq.py ``_stock_dcp``), per MLA layer, FP8 KV pages of
64 tokens, DL3 (q4) and DL7 (q8) drafter rows.

Two modes:
  --single (1 GPU): the stock tokenspeed core alone at the TP16 = DCP16 per-rank
      shape: 96 gathered heads causal (new drafter) vs 64 non-causal (old drafter: the
      previous reference), cp_world 16, every request with ``local`` KV tokens on
      this rank (the specialized core is measured inside the merge below).
  torchrun --nproc-per-node 4 (default): DraftDcp.merge vs the stock path at DCP4,
      configs (heads per rank, causal):
        new-core  24, causal   96 gathered heads: the production core shape and the
                               production (rows, 96) table keys; the head-owner
                               exchange / W_UV moves 4x the production heads
        new-xchg   6, causal   the production per-rank heads (exchange, W_UV) but 24
                               gathered heads in the core and (rows, 24) table keys
        old-core  16, non-causal / old-xchg 4, non-causal: the 64-head drafter
      Both arms start from the gathered q (the specialized arm: the q rows copied
      into its mailbox, as the stock arm's q is already all-gathered; the copy is
      timed alone and subtracted, ``copy_us``). Every request has ``local`` KV
      tokens on every rank (total = local x DCP).

Timing: CUDA graph of alternating layers (parity 0 / 1) replayed by every rank
together; a capture failure on any rank falls back to eager on all (``mode``).
"""

import argparse
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Deadline, Table, guarded, log, time_graph  # noqa: E402

PAGE, Q_SCALE, K_SCALE = 64, 1.0, 0.5
SOFTMAX_SCALE = (128 + 64) ** -0.5
DL3_B = [1, 2, 3, 4, 5, 8, 12, 16, 21, 32, 48, 64, 96, 128, 192, 256]
DL7_B = [1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 24, 32, 48, 64, 96, 128]
CONFIGS = {
    "new-core": (24, True),
    "new-xchg": (6, True),
    "old-core": (16, False),
    "old-xchg": (4, False),
}


def kv_inputs(torch, reqs, steps, world, heads, local, gen, device):
    """FP8 paged KV of ``reqs`` requests with ``local`` tokens on this rank (pages
    shuffled), the gathered FP8 q rows [rows, heads, 576] and the decode metadata
    (total = local x world: interleave 1, every rank the same local length)."""
    pages = max(1, -(-local // PAGE))
    blocks = reqs * pages + 1
    order = torch.randperm(blocks - 1, generator=gen) + 1
    table = order[: reqs * pages].view(reqs, pages).to(torch.int32).to(device)
    # q and the KV pool from a device generator seeded off ``gen``: identical on every
    # rank when ``gen`` is (the stock arm's q is all-gathered, the same on every rank)
    dgen = torch.Generator(device=device).manual_seed(
        int(torch.randint(0, 1 << 62, (1,), generator=gen))
    )
    pool = torch.randn((PAGE * 64, 576), device=device, generator=dgen) * 0.5
    pool = pool.to(torch.float8_e4m3fn)
    cache = torch.empty((blocks, PAGE, 576), dtype=torch.uint8, device=device)
    flat = cache.view(-1, 576)
    src = pool.view(torch.uint8)
    for i in range(0, flat.shape[0], src.shape[0]):
        n = min(src.shape[0], flat.shape[0] - i)
        flat[i : i + n].copy_(src[:n])
    rows = reqs * steps
    q = torch.randn((rows, heads, 576), device=device, generator=dgen) * 0.5
    q = q.to(torch.float8_e4m3fn)
    q = q.view(torch.uint8)
    q[q == 0x80] = 0  # (-0 -> +0: no chunk starts with the validity sentinel)
    i32 = dict(dtype=torch.int32, device=device)
    total = local * world
    metadata = SimpleNamespace(
        decode=SimpleNamespace(
            block_table=table,
            seq_lens=torch.full((reqs,), local, **i32),
            dcp_tot_seq_lens=torch.full((reqs,), total, **i32),
        ),
        num_decodes=reqs, num_decode_tokens=rows, num_prefills=0, max_seq_len=total,
    )  # fmt: skip
    return cache.view(torch.float8_e4m3fn), q, metadata


def tokenspeed(torch, q, cache, metadata, causal, world, rank, workspace=None):
    """Stock TokenspeedMLAImpl.forward_mqa under DCP (return_lse)."""
    from tokenspeed_mla import get_num_sm, tokenspeed_mla_decode

    decode = metadata.decode
    reqs = decode.seq_lens.shape[0]
    rows, heads = q.shape[0], q.shape[1]
    q = q.view(torch.float8_e4m3fn).view(reqs, rows // reqs, heads, 576)
    if workspace is None:
        size = get_num_sm(q.device) * heads * 8 * 513 * 4
        workspace = torch.empty(size, dtype=torch.int8, device=q.device)
    o, lse = tokenspeed_mla_decode(
        query=q, kv_cache=cache, workspace_buffer=workspace, kv_lora_rank=512,
        qk_rope_head_dim=64, block_tables=decode.block_table, seq_lens=decode.seq_lens,
        max_seq_len=metadata.max_seq_len,
        softmax_scale=SOFTMAX_SCALE * Q_SCALE * K_SCALE, output_scale=K_SCALE,
        enable_pdl=False, return_lse=True, causal_mask=causal,
        causal_seqs=decode.dcp_tot_seq_lens, cp_world=world, cp_rank=rank,
        cp_interleave_size=1, min_split_kv=1,
    )  # fmt: skip
    return o.view(rows, heads, 512), lse.reshape(rows, heads)


def single(args):
    import torch

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    # Reference rows (new geometry vs old geometry, not specialized vs stock): spec /
    # stock stay empty so they never enter a WORSE list; the times are in new96_us /
    # old64_us / new_over_old (summarize.py lists dcp_single/ apart).
    tab = Table(os.path.join(args.out, "dcp_core_single.csv"),
                extra_cols=["steps", "batch", "local", "new96_us", "old64_us",
                            "new_over_old"])  # fmt: skip
    deadline = Deadline(args.budget)
    gen = torch.Generator().manual_seed(5)
    from tokenspeed_mla import get_num_sm

    for local in args.locals:
        for steps, bl in ((3, DL3_B), (7, DL7_B)):
            for b in bl:
                if deadline.out():
                    continue
                res = {}
                for heads, causal in ((96, True), (64, False)):
                    def case(heads=heads, causal=causal):
                        cache, q, md = kv_inputs(torch, b, steps, 16, heads, local, gen, dev)
                        ws = torch.empty(get_num_sm(dev) * heads * 8 * 513 * 4,
                                         dtype=torch.int8, device=dev)  # fmt: skip
                        fn = lambda: tokenspeed(torch, q, cache, md, causal, 16, 0, ws)  # noqa: E731
                        return time_graph([fn], calls=16)

                    res[heads] = guarded(log, f"tokenspeed h{heads} K{steps} B{b} L{local}",
                                         case)  # fmt: skip
                # the new drafter core (96 heads causal) relative to the previous
                # drafter's (64 heads non-causal): a reference, not a WORSE candidate
                new, old = res[96], res[64]
                tab.add(f"ts_core_new96_vs_old64_K{steps}_L{local}", b * steps, None,
                        None, "tokenspeed", "reference: 96h causal vs 64h non-causal",
                        f"B{b}", steps=steps, batch=b, local=local, new96_us=new,
                        old64_us=old,
                        new_over_old=new / old if new and old else None)  # fmt: skip
                torch.cuda.empty_cache()
    tab.finish("single-rank tokenspeed core: 96-head causal vs 64-head non-causal")


class DraftAttention:
    """What DraftDcp reads of a drafter MLA layer (cf. the GPU test's
    ``_DraftAttention``): stock tokenspeed as ``impl.forward_mqa``."""

    def __init__(self, torch, heads, device, cache, world, rank, gen):
        self.num_local_heads = heads
        # W_UV [H, 512, 128] latent-contiguous, as the model's (split_kv_b_proj's .T
        # view, transposed): DcpValue's 16 B W_UV loads need it (mla.dcp compile_value)
        w = torch.randn((heads * 128, 512), generator=gen) * 0.05
        self.W_UV = w.bfloat16().to(device).T.view(512, heads, 128).transpose(0, 1)
        self.scale = SOFTMAX_SCALE
        self._q_scale_float, self._k_scale_float = Q_SCALE, K_SCALE
        self.rotary_emb = None
        self.kv_cache = cache
        self.impl = SimpleNamespace(
            forward_mqa=lambda q, kv, meta, layer: tokenspeed(
                torch, q, kv, meta, meta.causal, world, rank
            )
        )

    def _attn_read_kv_cache(self):
        return self.kv_cache


def dcp_path(shared, rows):
    for name, attr in (("fork", "forks"), ("rows", "row_modes"), ("wide", "wide_cores"),
                       ("long_polled", "long_polled"), ("polled", "polled"),
                       ("fold", "folds")):  # fmt: skip
        if rows in getattr(shared, attr, ()):
            return name
    return "gathered+stock-core"


def distributed(args):
    import torch
    import torch.distributed as dist

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft.attention import (
        DraftDcp,
    )
    from vllm.v1.attention.ops.dcp import correct_attn_out, mask_dcp_empty_shards_

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())
    group = dist.group.WORLD
    lead = rank == 0
    say = log if lead else (lambda m: None)
    deadline = Deadline(args.budget)
    tab = Table(os.path.join(args.out, "dcp.csv") if lead else None, echo=lead,
                extra_cols=["config", "steps", "batch", "local", "heads_rank",
                            "gathered", "path", "mode", "copy_us"])  # fmt: skip

    def agree(ok):
        t = torch.tensor([1 if ok else 0], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())

    def timed(fns, label):
        """(us, mode): eager warmup together, then a graph capture agreed by every
        rank (else eager on all)."""
        for f in fns:
            f()
        torch.cuda.synchronize()
        if not agree(True):
            raise RuntimeError("warmup failed on another rank")
        g = None
        try:
            loops = max(1, 16 // len(fns))
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                for _ in range(loops):
                    for f in fns:
                        f()
            ok = True
        except Exception as e:  # noqa: BLE001
            log(f"rank {rank}: {label}: graph capture failed: {str(e)[:200]}")
            ok, g = False, None
        if agree(ok):
            dist.barrier()
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            dist.barrier()
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(8):
                g.replay()
            b.record()
            b.synchronize()
            return a.elapsed_time(b) * 1e3 / (8 * loops * len(fns)), "graph"
        dist.barrier()
        torch.cuda.synchronize()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        iters = 20
        a.record()
        for _ in range(iters):
            for f in fns:
                f()
        b.record()
        b.synchronize()
        return a.elapsed_time(b) * 1e3 / (iters * len(fns)), "eager"

    def stock_layer(q, cache, md, causal, w_uv, out_rows):
        """The stock drafter DCP attention of one layer (cf. the test's _stock_dcp)."""
        o, lse = tokenspeed(torch, q, cache, md, causal, world, rank)
        decode = md.decode
        reqs = decode.seq_lens.shape[0]
        rows, heads = o.shape[0], o.shape[1]
        starts = torch.arange(reqs + 1, dtype=torch.int32, device=o.device)
        lse = lse.contiguous()
        mask_dcp_empty_shards_(lse, decode.seq_lens, starts * (rows // reqs))
        o = torch.where(torch.isinf(lse)[..., None], torch.zeros_like(o), o)
        lses = torch.empty((world, rows, heads), dtype=lse.dtype, device=o.device)
        dist.all_gather_into_tensor(lses, lse, group=group)
        out, _ = correct_attn_out(o, lses, rank, None, is_lse_base_on_e=False)
        per = heads // world
        mine = torch.empty((rows, per, 512), dtype=out.dtype, device=o.device)
        chunks = out.view(rows, world, per, 512).transpose(0, 1).contiguous()
        dist.reduce_scatter_tensor(mine, chunks, group=group)
        torch.bmm(mine.transpose(0, 1), w_uv, out=out_rows.view(rows, per, 128).transpose(0, 1))

    configs = [c for c in args.configs.split(",") if c]
    for cname in configs:
        heads, causal = CONFIGS[cname]
        gathered = heads * world
        for steps, bl in ((3, DL3_B), (7, DL7_B)):
            rows_list = [steps * b for b in bl]
            gen = torch.Generator().manual_seed(77 + steps + rank * 1000)
            shared_gen = torch.Generator().manual_seed(4242 + steps)
            t0 = time.time()

            def build():
                cache0, _, _ = kv_inputs(torch, 1, steps, world, gathered, 64,
                                         shared_gen, dev)  # fmt: skip
                layers = [DraftAttention(torch, heads, dev, cache0, world, rank, gen)
                          for _ in (0, 1)]  # fmt: skip
                shared = DraftDcp(layers, group, world, rank, rows_list, steps, 1 << 20,
                                  1, causal)  # fmt: skip
                return layers, shared

            built = guarded(log, f"DraftDcp {cname} DL{steps}", build)
            if not agree(built is not None):
                say(f"skip {cname} DL{steps}: DraftDcp build failed on some rank")
                continue
            layers, shared = built
            say(f"DraftDcp {cname} DL{steps}: built in {time.time() - t0:.0f}s, "
                f"dropped {shared.dropped}")  # fmt: skip
            for local in args.locals:
                for rows in rows_list:
                    t = torch.tensor([0 if deadline.out() else 1], device=dev)
                    dist.all_reduce(t, op=dist.ReduceOp.MIN)
                    if not t.item():
                        say(f"budget out: skip {cname} DL{steps} rows {rows} L{local}")
                        continue
                    reqs = rows // steps
                    path = dcp_path(shared, rows)
                    labels = f"DL{steps}B{reqs}"
                    if rows in shared.dropped:
                        if lead:
                            tab.add(f"dcp_{cname}_K{steps}_L{local}", rows, None, None,
                                    "stock", f"dropped: {shared.dropped[rows]}", labels,
                                    config=cname, steps=steps, batch=reqs, local=local,
                                    heads_rank=heads, gathered=gathered)  # fmt: skip
                        continue

                    def setup(rows=rows, reqs=reqs, local=local):
                        cache, q, md = kv_inputs(torch, reqs, steps, world, gathered,
                                                 local, shared_gen, dev)  # fmt: skip
                        md.causal = causal
                        for layer in layers:
                            layer.kv_cache = cache
                            layer.draft_dcp.bind(cache)
                        padded = shared.row_modes.get(rows, (rows,))[0]
                        outs = [torch.empty((padded, heads * 128), dtype=torch.bfloat16,
                                            device=dev) for _ in layers]  # fmt: skip
                        rows_mode = rows in shared.row_modes

                        def mailbox_of(dcp):
                            if rows in shared.wide_cores:
                                mb = shared.wide_box.mailbox(True)
                            else:
                                mb = shared.box.mailbox(rows in shared.valid)
                            return mb[dcp.parity, :rows, :gathered]

                        boxes = [mailbox_of(layer.draft_dcp) for layer in layers]
                        return cache, q, md, outs, rows_mode, boxes

                    st = guarded(log, f"{cname} DL{steps} rows {rows} L{local} setup",
                                 setup)  # fmt: skip
                    if not agree(st is not None):
                        say(f"{cname} rows {rows}: setup failed on some rank")
                        continue

                    def one(rows=rows, st=st):
                        cache, q, md, outs, rows_mode, boxes = st

                        def spec(i):
                            layer = layers[i]
                            boxes[i].copy_(q)
                            layer.draft_dcp.merge(layer, md, rows, outs[i],
                                                  rows_mode=rows_mode)  # fmt: skip

                        spec_us, mode = timed([lambda: spec(0), lambda: spec(1)],
                                              f"{cname} rows {rows}")  # fmt: skip
                        scratch = torch.empty_like(q)
                        copy_us = time_graph([lambda: scratch.copy_(q)], calls=32)
                        stock_out = torch.empty((rows, heads * 128), dtype=torch.bfloat16,
                                                device=dev)  # fmt: skip
                        stock_us, smode = timed(
                            [lambda: stock_layer(q, cache, md, causal, layers[0].W_UV,
                                                 stock_out)], f"stock rows {rows}")  # fmt: skip
                        return spec_us - copy_us, stock_us, mode + "/" + smode, copy_us

                    r = guarded(log, f"{cname} DL{steps} rows {rows} L{local}", one)
                    if not agree(r is not None):
                        say(f"{cname} rows {rows}: failed on some rank")
                        if lead:
                            tab.add(f"dcp_{cname}_K{steps}_L{local}", rows, None, None,
                                    path, "FAILED", labels, config=cname, steps=steps,
                                    batch=reqs, local=local, heads_rank=heads,
                                    gathered=gathered, path=path)  # fmt: skip
                        continue
                    if lead:
                        tab.add(f"dcp_{cname}_K{steps}_L{local}", rows, r[0], r[1], path,
                                "", labels, config=cname, steps=steps, batch=reqs,
                                local=local, heads_rank=heads, gathered=gathered,
                                path=path, mode=r[2], copy_us=r[3])  # fmt: skip
                    torch.cuda.empty_cache()
            del layers, shared, built
            torch.cuda.empty_cache()
    if lead:
        tab.finish("DraftDcp merge vs stock drafter DCP path (per layer)")
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--configs", default="new-core,new-xchg,old-core,old-xchg")
    ap.add_argument("--locals", default="560,4096",
                    help="local KV tokens per request per rank (code comments' "
                         "'local 560' / 'local 4096')")  # fmt: skip
    ap.add_argument("--budget", type=float, default=0)
    args = ap.parse_args()
    args.locals = [int(x) for x in args.locals.split(",")]
    os.makedirs(args.out, exist_ok=True)
    if args.single:
        single(args)
    else:
        distributed(args)


if __name__ == "__main__":
    main()
