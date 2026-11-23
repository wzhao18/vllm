# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP specialized decode under DSA-DP (TP4 at DCP1, ``runtime.sp_attention``
``SpStep.run_dsa_dp``), on 4 ranks: the step's layer chain with the stock DSA-DP owner
segment (``ops.dsa_dp.run_segment``: Lamport push in -> the layer on the owner's
DEP_PERM segment -> Lamport push out) between fused SP boundaries
(``attn_res.sp_fused.SpBoundary`` on ``SpComm``):

  ours: KDA-like layers through SpBoundary.gather / reduce (SpComm mailboxes, parity
        by ordinal over the SP blocks only), DSA layers through stock
        ``attn_res`` -> ``dsa_dp.run_segment`` -> stock ``attn_res``
  ref : the same chain with each DSA layer's owner exchange replaced by an all-gather
        of the shard rows and the layer on every row (``run_segment``'s math with
        no exchange)

over several steps with random, unbalanced owners (DEP_PERM padding requests
included) and shard rows 1 / 8 / 24 requests per owner at q4. The layer stand-ins
are row-wise (a row's output depends on that row and its position only), so ``ours``
must equal ``ref`` bit for bit; every SpComm mailbox and both DSA-DP Lamport buffers
must be re-armed after each step (the two kinds of mailboxes interleave without
colliding).
"""

import os
import sys
import time
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

WIDTH, Q, WORLD = 7168, 4, 4
CAP = 512
BLOCK, NBLOCKS = 12, 8
SENTINEL = -0x80000000
SHARDS = (4, 32, 96)  # n rows per rank: 1 / 8 / 24 requests per owner at q4
# (layer index, kind): an odd number of SP blocks, DSA layers between them, a
# block-write layer (12) among each kind.
LAYERS = ((1, "kda"), (2, "kda"), (3, "dsa"), (12, "kda"), (15, "dsa"), (16, "kda"))
STEPS = 3


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


def _armed(comm, parity, rows):
    g = comm.gather[parity, :rows].view(torch.int32)
    r = comm.recv[parity].view(torch.int32)
    s = comm.box[parity, comm.rows_cap :].view(torch.int32)
    ok = bool((g == SENTINEL).all()) and bool((r == SENTINEL).all())
    return ok and bool((s == -1).all())


def _owners(rows, step, world):
    """Owner card of each real request of a ``rows``-row step: a random number of
    real requests, unbalanced, each owner at most ``per`` (DEP_PERM pads the rest)."""
    per = rows // (world * Q)
    rng = np.random.default_rng(zlib.crc32(repr(("own", rows, step)).encode()))
    counts = rng.integers(0, per + 1, size=world)
    counts[rng.integers(0, world)] = per  # one owner full
    owners = np.repeat(np.arange(world), counts)
    rng.shuffle(owners)
    return owners


def _worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local = int(env["LOCAL_RANK"])
    device = torch.device(f"cuda:{local}")
    torch.accelerator.set_device_index(local)
    dist.init_process_group(backend="nccl")
    try:
        import vllm.distributed as vdist
        import vllm.forward_context as vfc
        from vllm.models.kimi_k3.nvidia.ops import dsa_dp
        from vllm.models.kimi_k3.nvidia.ops.attn_res import attn_res
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.attn_res import (
            sp_fused,
        )
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import (
            sp_comm,
        )
        from vllm.v1.attention.ops import shardq

        rank, world = dist.get_rank(), dist.get_world_size()
        group = dist.group.WORLD
        try:
            from torch.distributed._symmetric_memory import enable_symm_mem_for_group

            enable_symm_mem_for_group(group.group_name)
        except (ImportError, AttributeError):
            pass
        # dsa_dp's process facts: the TP group is this world.
        dsa_dp._tp = lambda: (world, rank)
        vdist.get_tp_group = lambda: SimpleNamespace(device_group=group)
        context = SimpleNamespace()
        vfc.get_forward_context = lambda: context
        dsa_dp.DEP_PERM.ensure(CAP, Q, device)

        comm = sp_comm.SpComm(group, device, world, rank, rows_cap=CAP)
        eps = (1e-6, 1e-6)
        scale = {i: _rand((WIDTH,), ("a", i), device, 0.5) for i, _ in LAYERS}
        sp_blocks = [i for i, kind in LAYERS if kind == "kda"]
        parity = {i: o % 2 for o, i in enumerate(sp_blocks)}

        def attn(i):
            def layer(hidden_states, positions):  # row-wise stand-in
                x = hidden_states.float() * scale[i].float()
                return (x + positions[:, None].float() * 1e-3).bfloat16()

            return layer

        for n in SHARDS:
            rows = n * world
            positions = torch.arange(rows, device=device, dtype=torch.int64) * 7 % 4093
            bounds = {}
            for i, kind in LAYERS:
                if kind != "kda":
                    continue
                write = i % BLOCK == 0
                bounds[i] = sp_fused.SpBoundary(
                    comm, (_weights(i, 0, device), eps), (_weights(i, 1, device), eps),
                    -(-i // BLOCK), i // BLOCK if write else -1, max(SHARDS),
                    {rows: True},
                )  # fmt: skip
            for step in range(STEPS):
                owners = _owners(rows, step, world)
                res = {}
                for mode in ("ref", "ours"):
                    dsa_dp.DEP_PERM.set(owners, rows)
                    context.__dict__.clear()  # a new forward: new exchange plan
                    torch.accelerator.synchronize(device)
                    dist.barrier()
                    prefix = _rand((n, WIDTH), ("p", rows, rank), device)
                    blocks = _rand((n, NBLOCKS, WIDTH), ("b", rows, rank), device)
                    hidden = _rand((n, WIDTH), ("d", rows, rank, step), device, 0.1)
                    out = torch.empty((n, WIDTH), dtype=torch.bfloat16, device=device)
                    for i, kind in LAYERS:
                        prev = -(-i // BLOCK)
                        write = i % BLOCK == 0
                        if kind == "kda":
                            bound, p = bounds[i], parity[i]
                            g = bound.gather(hidden, prefix, blocks, p)
                            partial = (g[:rows].float() * scale[i].float() * 0.25)
                            partial = partial.bfloat16()
                            hidden = bound.reduce(prefix, blocks, out, p, partial)
                            hidden = hidden.clone()
                        else:  # the stock DSA-DP layer (SpStep.run_dsa_dp)
                            pre_w = _weights(i, 0, device)
                            post_w = _weights(i, 1, device)
                            x = attn_res(
                                prefix, hidden, blocks, *pre_w, num_blocks=prev,
                                block_write_idx=i // BLOCK if write else -1,
                                eps=eps[0], output_norm_eps=eps[1],
                            )  # fmt: skip
                            if mode == "ours":
                                x = dsa_dp.run_segment(attn(i), positions, x)
                            else:
                                full = torch.empty(
                                    (rows, WIDTH), dtype=x.dtype, device=device
                                )
                                dist.all_gather_into_tensor(full, x.contiguous())
                                x = attn(i)(full, positions)[rank * n : (rank + 1) * n]
                            if write:
                                prefix, pd = x.clone(), None
                            else:
                                pd = x
                            hidden = attn_res(
                                prefix, pd, blocks, *post_w,
                                num_blocks=prev + int(write), block_write_idx=-1,
                                eps=eps[0], output_norm_eps=eps[1],
                            )  # fmt: skip
                        hidden = (hidden.float() * 0.5).bfloat16()  # the MLP stand-in
                    torch.accelerator.synchronize(device)
                    dist.barrier()  # (the step's final gather)
                    for p in (0, 1):
                        assert _armed(comm, p, rows), (mode, rows, step, p)
                    for name in ("dsadp_dep_in", "dsadp_dep_out"):
                        key = (group.group_name, name, torch.bfloat16)
                        buf = shardq._buffers.get(key)
                        if buf is not None:
                            view = buf.storage.view(torch.int16)
                            assert bool((view == -32768).all()), (name, rows, step)
                    res[mode] = (hidden, prefix, blocks)
                for a, b in zip(res["ref"], res["ours"]):
                    assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (
                        rows,
                        step,
                    )
                if rank == 0:
                    counts = np.bincount(owners, minlength=world)
                    print(f"R{rows} step {step} owners {counts} ok", flush=True)
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
def test_sp_dsadp_chain_matches_gathered_reference():
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"needs {WORLD} GPUs")
    _run(WORLD)
