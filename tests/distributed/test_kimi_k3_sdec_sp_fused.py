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
- the gathered rows and the post-AttnRes output are within the BF16 rounding of the
  stock op (the AttnRes summation order differs: ClusterAttnRes);
- every mailbox is re-armed after each boundary pair.

``SpFinalGather`` returns stock ``sp_all_gather``'s rows (sanitized words) and
re-arms its box.
"""

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
ROWS = (16, 128, 384)  # n = 4 / 32 / 96 per rank at TP4
LAYERS = (0, 1, 5, 12, 13, 50, 92)


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
    """|a - b| <= ulps BF16 ulps of the larger magnitude, elementwise."""
    bound = ulps * _ulp(torch.maximum(a.abs(), b.abs())) + 2.0**-60
    return bool(((a - b).abs() <= bound).all()), ((a - b).abs() / bound).max().item()


def _armed(comm, parity, rows):
    g = comm.gather[parity, :rows].view(torch.int32)
    r = comm.recv[parity].view(torch.int32)
    return bool((g == SENTINEL).all()) and bool((r == SENTINEL).all())


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
        comm = sp_comm.SpComm(group, device, world, rank, rows_cap=CAP)
        eps = (1e-6, 1e-6)
        worst = {}
        for rows in ROWS:
            n = rows // world
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
                    layer // BLOCK if write else -1, n, {rows: True},
                )  # fmt: skip
                res = {}
                for mode in ("stock", "fused"):
                    prefix, blocks = prefix0.clone(), blocks0.clone()
                    dist.barrier()
                    if mode == "stock":
                        x = attn_res(
                            prefix, d, blocks, *pre_w, num_blocks=prev,
                            block_write_idx=layer // BLOCK if write else -1,
                            eps=eps[0], output_norm_eps=eps[1],
                        )  # fmt: skip
                        g = comm.all_gather(x, parity).clone()
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
                        y = bound.reduce(prefix, blocks, out, parity, partial).clone()
                    torch.accelerator.synchronize(device)
                    assert _armed(comm, parity, rows), (mode, rows, layer)
                    res[mode] = [t.float() for t in (g, y, prefix, blocks)]
                s, f = res["stock"], res["fused"]
                assert torch.equal(s[2], f[2]), ("prefix", rows, layer)
                assert torch.equal(s[3], f[3]), ("blocks", rows, layer)
                for name, i in (("gather", 0), ("post", 1)):
                    ok, ratio = _close(s[i], f[i], 2)
                    worst[name] = max(worst.get(name, 0.0), ratio)
                    assert ok, (name, rows, layer, ratio)
        # The final multicast gather: stock rows (sanitized), box re-armed.
        width = 3 * WIDTH
        final = sp_comm.SpFinalGather(
            group, device, world, rank, width=width, rows_cap=CAP
        )
        for rows in ROWS:
            n = rows // world
            x = _rand((n, width), ("f", rows, rank), device)
            x.view(torch.int32)[:, 7::101] = SENTINEL  # (+0, -0): sanitized to 0
            got = final.run(x)
            torch.accelerator.synchronize(device)
            want = torch.cat([_rand((n, width), ("f", rows, r), device)
                              for r in range(world)])
            want.view(torch.int32)[:, 7::101] = 0
            assert torch.equal(got.view(torch.int32), want.view(torch.int32)), rows
            assert bool((final.box.view(torch.int32) == SENTINEL).all()), rows
        if rank == 0:
            print("worst ulp ratios", worst)
        dist.barrier()
    finally:
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
    deadline = time.monotonic() + 600
    for p in procs:
        p.join(timeout=max(0.0, deadline - time.monotonic()))
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
@pytest.mark.parametrize("world", [4])
def test_sp_fused_boundaries(world):
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    _run(world)
