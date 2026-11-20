# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode, SP layer boundaries fused with the AttnRes
(``cutedsl.attn_res.sp_fused``) and the final multicast gather
(``cutedsl.common.sp_comm.SpFinalGather``), on 4 (or 2) ranks against the stock
chain they replace:

  stock: attn_res(pre) -> SpComm.all_gather -> SpComm.reduce_scatter(push the
         partial) -> attn_res(post)          (the SP step before this change)
  fused: SpAttnResGather -> SpReduceAttnRes(push)

per layer kind of the 93-layer AttnRes pattern (block size 12: first layer without
delta, block-write layers, 1..8 saved blocks) and per shard rows (r1 / r8 / r24 of
D3-32):

- the reduce-scatter sum, the updated / new prefix rows and the saved blocks are
  stock's (FP32 rank-order sums, one rounding; zero signs aside);
- the gathered rows and the post-AttnRes output are within two BF16 ulps of the row's
  largest magnitude of the stock op (the AttnRes summation order differs:
  ClusterAttnRes);
- every mailbox is re-armed after each boundary pair;
- the MXFP8 forms (an MXFP8 in-projection's all-gather, ``gather_mx``; the MXFP8
  shared gate_up input of the post-AttnRes, ``out_mx``) are ``mxfp8_act.quantize``
  of the fused BF16 rows (FP8 values equal, scales bitwise), and MXFP8 GEMMs over the
  gathered box equal those over the quantized reference;
- the signalled MXFP8 all-gather (``signal_rows``: no poll, per-chunk row counters)
  into a scribbled box, read by a gated pre GEMM with no host sync, equals the same
  GEMM ungated over the reference; the counters hold R and the reduce zeroes them.

``SpFinalGather`` returns stock ``sp_all_gather``'s rows (sanitized words) and
re-arms its box.
"""

import os
import sys
import time
import zlib

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

WIDTH = 7168
CAP = 512
BLOCK = 12
NBLOCKS = 8
SENTINEL = -0x80000000
# n rows per rank: r1 / r8 / r24 of D3-32 and the rows_cap (n_max: the full grid of
# 7-CTA clusters co-resident at the 93-layer 8-block AttnRes)
SHARDS = (4, 32, 96, 128)
# first, plain, block write, after one, 8 blocks
LAYERS = tuple(
    int(x) for x in os.environ.get("SP_FUSED_LAYERS", "0,1,12,13,92").split(",")
)


def _cdiv(a, b):
    return -(-a // b)


def _gen(*key):
    g = torch.Generator()
    g.manual_seed(zlib.crc32(repr(key).encode()))
    return g


def _rand(shape, key, device, scale=1.0):
    x = (torch.randn(shape, generator=_gen(*key)) * scale).bfloat16()
    words = x.view(torch.int32)
    words[words == SENTINEL] = 0
    return x.to(device)


def _weights(layer, site, device):
    g = _gen("w", layer, site)
    norm = (1 + 0.1 * torch.randn(WIDTH, generator=g)).bfloat16().to(device)
    query = (0.05 * torch.randn(WIDTH, generator=g)).bfloat16().to(device)
    out = (1 + 0.1 * torch.randn(WIDTH, generator=g)).bfloat16().to(device)
    return norm, query, out


def _ulp(x):
    """BF16 ulp of |x| (FP32 tensor)."""
    e = torch.floor(torch.log2(x.abs().clamp_min(2.0**-100)))
    return torch.exp2(e - 7)


def _close(a, b, ulps):
    """|a - b| <= ulps BF16 ulps of the row's largest magnitude (the AttnRes mixes and
    normalizes whole rows in FP32; a different summation order moves an element by a
    fraction of the row scale's rounding, also where sources cancel)."""
    scale = torch.maximum(a.abs(), b.abs()).amax(dim=-1, keepdim=True)
    bound = ulps * _ulp(scale) + 2.0**-60
    return bool(((a - b).abs() <= bound).all()), ((a - b).abs() / bound).max().item()


def _armed(comm, parity, rows):
    g = comm.gather[parity, :rows].view(torch.int32)
    r = comm.recv[parity].view(torch.int32)
    s = comm.box[parity, comm.rows_cap : -1].view(torch.int32)
    ready = comm.box[parity, -1].view(torch.int32)  # the signalled chunk counters
    ok = bool((g == SENTINEL).all()) and bool((r == SENTINEL).all())
    return ok and bool((s == -1).all()) and bool((ready == 0).all())


def _scale_index(rows, device):
    """F8_128x4 byte offsets of rows [0, rows) x 224 scale blocks (7168 / 32)."""
    r = torch.arange(rows, device=device).view(-1, 1)
    b = torch.arange(WIDTH // 32, device=device).view(1, -1)
    off = (b & 3) + (b >> 2) * 512 + (r & 31) * 16 + ((r & 127) >> 5) * 4
    return (off + (r >> 7) * 128 * (WIDTH // 32)).reshape(-1)


def _quant(mxfp8_act, x):
    out = mxfp8_act.Mxfp8Buffer(x.shape[0], x.shape[1], x.device)
    mxfp8_act.quantize(x, out=out)
    return out


def _same_mx(got, want, rows):
    """FP8 values equal (a -0 may be +0) and the rows' scale bytes bitwise."""
    data = torch.equal(got.data[:rows].float(), want.data[:rows].float())
    index = _scale_index(rows, want.scale.device)
    return data and torch.equal(got.scale[index], want.scale[index])


def _worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local = int(env["LOCAL_RANK"])
    device = torch.device(f"cuda:{local}")
    torch.accelerator.set_device_index(local)
    dist.init_process_group(backend="nccl")
    try:
        from vllm.models.kimi_k3.nvidia.ops.attn_res import attn_res
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.attn_res import (
            sp_fused,
        )
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import (
            sp_comm,
        )

        rank, world = dist.get_rank(), dist.get_world_size()
        group = dist.group.WORLD
        try:
            from torch.distributed._symmetric_memory import enable_symm_mem_for_group

            enable_symm_mem_for_group(group.group_name)
        except (ImportError, AttributeError):
            pass
        from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import mxfp8

        comm = sp_comm.SpComm(group, device, world, rank, rows_cap=CAP)
        eps = (1e-6, 1e-6)
        scratch = mxfp8.Mxfp8Input(max(SHARDS), WIDTH, device)
        # An MXFP8 weight [256, 7168] for the GEMM check (F8_128x4 scales).
        wq = mxfp8_act.quantize(_rand((256, WIDTH), ("w",), device))
        gemm = mxfp8.Mxfp8Projection(
            (wq.data, wq.scale), [n * world for n in SHARDS], family="kda_in",
            tactics={n * world: "scaled_mm" for n in SHARDS},
        )  # fmt: skip
        # The gated pre GEMM of a signalled all-gather (``signal_rows``).
        gated = mxfp8.Mxfp8Projection(
            (wq.data, wq.scale), [n * world for n in SHARDS], family="kda_in",
            tactics={n * world: "pre" for n in SHARDS}, gated=True,
        )  # fmt: skip
        worst = {}
        for n in SHARDS:
            rows = n * world
            blocks0 = _rand((n, NBLOCKS, WIDTH), ("b", rows, rank), device)
            prefix0 = _rand((n, WIDTH), ("p", rows, rank), device)
            delta = _rand((n, WIDTH), ("d", rows, rank), device, 0.1)
            partial = _rand((rows, WIDTH), ("o", rows, rank), device, 0.05)
            out = torch.empty((n, WIDTH), dtype=torch.bfloat16, device=device)
            for layer in LAYERS:
                parity = layer % 2
                prev = _cdiv(layer, BLOCK)
                write = layer % BLOCK == 0
                pre_w, post_w = _weights(layer, 0, device), _weights(layer, 1, device)
                d = None if layer == 0 else delta
                bound = sp_fused.SpBoundary(
                    comm, (pre_w, eps), (post_w, eps), prev,
                    layer // BLOCK if write else -1, max(SHARDS), {rows: True},
                    first=layer == 0,
                )  # fmt: skip
                res = {}
                for mode in ("stock", "fused"):
                    prefix, blocks = prefix0.clone(), blocks0.clone()
                    dist.barrier()
                    if os.environ.get("SP_FUSED_TRACE"):
                        print(f"rank{rank} R{rows} L{layer} {mode}", flush=True)
                    if mode == "stock":
                        x = attn_res(
                            prefix, d, blocks, *pre_w, num_blocks=prev,
                            block_write_idx=layer // BLOCK if write else -1,
                            eps=eps[0], output_norm_eps=eps[1],
                        )  # fmt: skip
                        g = comm.all_gather(x, parity).clone()
                        if os.environ.get("SP_FUSED_TRACE"):
                            torch.accelerator.synchronize(device)
                            print(f"rank{rank} stock gathered", flush=True)
                        h = comm.reduce_scatter(out, parity, rows, partial)
                        if write:
                            prefix, pd = h.clone(), None
                        else:
                            pd = h
                        y = attn_res(
                            prefix, pd, blocks, *post_w,
                            num_blocks=prev + int(write), block_write_idx=-1,
                            eps=eps[0], output_norm_eps=eps[1],
                        )  # fmt: skip
                    else:
                        g = bound.gather(d, prefix, blocks, parity).clone()
                        if os.environ.get("SP_FUSED_TRACE"):
                            torch.accelerator.synchronize(device)
                            print(f"rank{rank} gathered", flush=True)
                        y = bound.reduce(prefix, blocks, out, parity, partial).clone()
                    torch.accelerator.synchronize(device)
                    assert _armed(comm, parity, rows), (mode, rows, layer)
                    res[mode] = [t.float() for t in (g, y, prefix, blocks)]
                    if mode == "fused":
                        res["fused_bf16"] = (g, y)
                if rank == 0:
                    print(f"R{rows} L{layer} done", flush=True)
                # The MXFP8 forms of the same boundaries.
                mx = sp_fused.SpBoundary(
                    comm, (pre_w, eps), (post_w, eps), prev,
                    layer // BLOCK if write else -1, max(SHARDS), {rows: True},
                    first=layer == 0, gather_mx=True, out_mx=scratch,
                )  # fmt: skip
                signalled = sp_fused.SpBoundary(
                    comm, (pre_w, eps), (post_w, eps), prev,
                    layer // BLOCK if write else -1, max(SHARDS), {rows: True},
                    first=layer == 0, gather_mx=True, out_mx=scratch,
                    signal_rows={rows},
                )  # fmt: skip
                prefix, blocks = prefix0.clone(), blocks0.clone()
                dist.barrier()
                gv = mx.gather(d, prefix, blocks, parity)
                torch.accelerator.synchronize(device)
                ref = _quant(mxfp8_act, res["fused_bf16"][0])
                assert gv.rows == rows and _same_mx(gv, ref, rows), ("mx gather", rows)
                for kind in ("scaled_mm", "fi_cute"):
                    a = torch.empty((rows, 256), dtype=torch.bfloat16, device=device)
                    b = torch.empty_like(a)
                    gemm.kinds[rows] = kind
                    gemm.run(gv, a)
                    gemm.run(ref, b)
                    assert torch.equal(a, b), ("mx gemm", kind, rows)
                # The signalled form (gated in-projection, ``signal_rows``): the box
                # scribbled first, so a GEMM that did not wait would read sentinels.
                comm.gather[parity, :rows].view(torch.int32).fill_(SENTINEL)
                comm.box[parity, comm.rows_cap : -1].view(torch.int32).fill_(-1)
                prefix, blocks = prefix0.clone(), blocks0.clone()
                torch.accelerator.synchronize(device)
                dist.barrier()
                gs = signalled.gather(d, prefix, blocks, parity)
                a = torch.empty((rows, 256), dtype=torch.bfloat16, device=device)
                gated.run(gs, a)  # waits per chunk, no host sync in between
                b = torch.empty_like(a)
                gated.run(ref, b)  # the same GEMM, ungated, on the reference
                torch.accelerator.synchronize(device)
                assert _same_mx(gs.view, ref, rows), ("signalled gather", rows)
                assert torch.equal(a, b), ("gated gemm", rows, layer)
                ready = comm.box[parity, -1].view(torch.int32)[: 7 * 32 : 32]
                assert bool((ready == rows).all()), ("ready counts", rows, ready)
                y = mx.reduce(prefix, blocks, out, parity, partial).clone()
                torch.accelerator.synchronize(device)
                assert _armed(comm, parity, rows), ("mx", rows, layer)
                assert torch.equal(y, res["fused_bf16"][1]), ("mx post", rows, layer)
                ref = _quant(mxfp8_act, y)
                assert _same_mx(scratch.view(n), ref, n), ("mx out", rows, layer)
                if d is not None:
                    # A beta-add GEMM's FP32 split partials + residual (SplitDelta, the
                    # latent up's form): bitwise the gather of the BF16 delta stock's
                    # beta-add forms (split order; BF16 GEMM output, then + residual).
                    pv = _rand((3, n, WIDTH), ("q", rows, rank), device, 0.05).float()
                    resid = _rand((n, WIDTH), ("r", rows, rank), device, 0.1)
                    gsum = ((pv[0] + pv[1]) + pv[2]).bfloat16().float()
                    want = (gsum + resid.float()).bfloat16()
                    pb = sp_fused.SpBoundary(
                        comm, (pre_w, eps), (post_w, eps), prev,
                        layer // BLOCK if write else -1, max(SHARDS), {rows: True},
                        delta_parts=(3,),
                    )  # fmt: skip
                    got = []
                    for dd in (want, sp_fused.SplitDelta(pv.reshape(3 * n, WIDTH),
                                                         resid, 3)):  # fmt: skip
                        prefix, blocks = prefix0.clone(), blocks0.clone()
                        dist.barrier()
                        g = pb.gather(dd, prefix, blocks, parity).clone()
                        y = pb.reduce(prefix, blocks, out, parity, partial).clone()
                        torch.accelerator.synchronize(device)
                        assert _armed(comm, parity, rows), ("parts", rows, layer)
                        got.append((g, y, prefix, blocks))
                    for a, b in zip(*got):
                        assert torch.equal(a, b), ("parts", rows, layer)
                s, f = res["stock"], res["fused"]
                assert torch.equal(s[2], f[2]), ("prefix", rows, layer)
                assert torch.equal(s[3], f[3]), ("blocks", rows, layer)
                for name, i in (("gather", 0), ("post", 1)):
                    ok, ratio = _close(s[i], f[i], 2)
                    worst[name] = max(worst.get(name, 0.0), ratio)
                    assert ok, (name, rows, layer, ratio)
        if rank == 0:
            print("boundaries ok; final gather", flush=True)
        # The final multicast gather: stock rows (sanitized), box re-armed.
        width = 3 * WIDTH
        final = sp_comm.SpFinalGather(
            group, device, world, rank, width=width, rows_cap=CAP
        )
        for n in SHARDS:
            rows = n * world
            x = _rand((n, width), ("f", rows, rank), device)
            x.view(torch.int32)[:, 7::101] = SENTINEL  # (+0, -0): sanitized to 0
            got = final.run(x)
            torch.accelerator.synchronize(device)
            want = torch.cat([_rand((n, width), ("f", rows, r), device)
                              for r in range(world)])
            want.view(torch.int32)[:, 7::101] = 0
            assert torch.equal(got.view(torch.int32), want.view(torch.int32)), rows
            # Every rank done with this gather before the re-arm check, and checked
            # before any rank multicasts the next one into the boxes.
            dist.barrier()
            assert bool((final.box.view(torch.int32) == SENTINEL).all()), rows
            dist.barrier()
            # The same gather with column block 1 pushed early from a side stream
            # (SpStep's early aux) and blocks 0 and 2 with the final kernel.
            parts = [x[:, i * WIDTH : (i + 1) * WIDTH].contiguous() for i in range(3)]
            side = torch.cuda.Stream(device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                final.push(parts[1], WIDTH)
            torch.cuda.current_stream(device).wait_stream(side)
            got = final.run(None, cols=((parts[0], 0), (parts[2], 2 * WIDTH)))
            torch.accelerator.synchronize(device)
            assert torch.equal(got.view(torch.int32), want.view(torch.int32)), rows
            dist.barrier()
            assert bool((final.box.view(torch.int32) == SENTINEL).all()), rows
            dist.barrier()
        if rank == 0:
            print("worst ulp ratios", worst)
        dist.barrier()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)  # (a peer may wait in a collective: never hang the test)
    dist.destroy_process_group()


def _run(world):
    port = str(get_open_port())
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(world):
        env = {
            "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world),
            "LOCAL_WORLD_SIZE": str(world), "MASTER_ADDR": "localhost",
            "MASTER_PORT": port,
        }  # fmt: skip
        p = ctx.Process(target=_worker, args=(env,))
        p.start()
        procs.append(p)
    # A failed rank exits at once (os._exit); its peers may spin on device waits
    # for it, so stop them as soon as any rank fails (or at the deadline).
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline and any(p.is_alive() for p in procs):
        if any(p.exitcode not in (None, 0) for p in procs):
            break
        time.sleep(0.5)
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join()
    assert [p.exitcode for p in procs] == [0] * world


def _multicast() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        return _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)
    except Exception:
        return False


@pytest.mark.skipif(not _multicast(), reason="needs NVLink multicast")
@pytest.mark.parametrize("world", [2, 4])
def test_sp_fused_boundaries(world):
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    _run(world)
