# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode, sequence parallelism (C512 SP attention blocks): the
TP group's SP mailboxes (``cutedsl.common.sp_comm.SpComm``) and the o-projection
fused with its reduce-scatter (``cutedsl.oproj.sp.SpOproj``).

On 2 or 4 ranks (the deployment is TP4; the mailboxes are generic in TP):

- ``all_gather``: every rank gathers every rank's shard in rank order, bit for bit,
  with stock's Lamport sanitizing (a 0x80000000 word, BF16 (+0, -0), becomes 0);
- ``SpOproj`` "cublas": the shard equals stock's o-projection + reduce-scatter bit
  for bit (F.linear partials, summed in FP32 in rank order, rounded once,
  sanitized: ``mnnvl_lamport_reduce_scatter``);
- ``SpOproj`` "warp" (fused split-K mma.sync, pushes from the GEMM) and "gemm" with
  a cuBLASLt partial (the TP4 o_proj table's ``lt`` rows): within the BF16
  rounding of the partials of the FP64 reference (summation order differs); "gemm"
  with a cuBLAS partial (every R at TP2, where the TP4 table does not apply) is
  stock's bit for bit;
- the mailboxes are re-armed after every block (gather rows [0, R) and every recv
  slot hold the sentinel), across alternating parities and row counts, and in a
  captured AG -> o-projection -> AG -> o-projection graph replayed between eager
  steps of other row counts.
"""

import time
import zlib

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

K = 24 * 128  # attention output per rank (24 heads x 128)
WIDTH = 7168
CAP = 512
SENTINEL = -0x80000000
# Token rows (multiples of 4 and 2): both tactics' ranges, the R <= 64 / > 64 seam,
# the warp tactic's largest rows, a cuBLASLt row of the TP4 o_proj table (324) and
# the deployment's largest graph.
ROWS = (4, 8, 12, 64, 68, 128, 324, 512, 8)
WARP_ROWS = (4, 8, 12, 64, 68, 128)
GRAPH_ROWS, REPLAYS = 64, 3


def _has_multicast_support() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        return _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)
    except Exception:
        return False


def _generator(*key):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(zlib.crc32(repr(key).encode()))  # the same on every rank
    return gen


def _shard(step, rank, n, device):
    """Rank ``rank``'s normed shard [n, WIDTH] at ``step``, with some (+0, -0) BF16
    pairs (the Lamport sentinel word) and some (-0, +0) pairs."""
    x = torch.randn(n, WIDTH, generator=_generator("x", step, rank)).bfloat16()
    words = x.view(torch.int32).view(n, WIDTH // 2)
    words[:, 3::97] = SENTINEL  # bits 0x80000000: low +0, high -0
    words[:, 5::89] = 0x8000  # low -0, high +0 (not the sentinel)
    return x.to(device)


def _sanitized(x):
    words = x.view(torch.int32).clone()
    words[words == SENTINEL] = 0
    return words.view(torch.bfloat16)


def _inputs(step, rank, rows, device):
    attn = torch.randn(rows, K, generator=_generator("a", step, rank)).bfloat16()
    return attn.to(device)


def _weight(rank, device):
    w = torch.randn(WIDTH, K, generator=_generator("w", rank)) * K**-0.5
    return w.bfloat16().to(device)


def _reference(step, world, rank, rows, device, weights):
    """(stock shard, FP64 shard, partial scale): stock = sanitize(bf16(sum in rank
    order of F.linear partials in FP32)) of this rank's rows."""
    n = rows // world
    mine = slice(rank * n, (rank + 1) * n)
    total, exact, scale = None, None, 0.0
    for src in range(world):
        attn = _inputs(step, src, rows, device)
        partial = _sanitized(torch.nn.functional.linear(attn, weights[src]))[mine]
        total = partial.float() if total is None else total + partial.float()
        product = attn.double() @ weights[src].double().t()
        exact = product[mine] if exact is None else exact + product[mine]
        scale = max(scale, product.abs().max().item())
    return _sanitized(total.bfloat16()), exact, scale


def _check_armed(comm, parity, rows):
    gather = comm.gather[parity, :rows].view(torch.int32)
    assert bool((gather == SENTINEL).all()), ("gather not re-armed", parity, rows)
    recv = comm.recv[parity].view(torch.int32)
    assert bool((recv == SENTINEL).all()), ("recv not re-armed", parity, rows)


def _stock_bits(tactic) -> bool:
    """Whether an ``SpOproj.choice`` tactic computes its partial with stock's cuBLAS
    GEMM (bit-exact shard); cuBLASLt may pick another kernel than ``F.linear``."""
    return tactic in ("cublas", ("gemm", "cublas"))


def _step(comm, oproj, step, rank, world, rows, parity, out, weights, bitwise):
    device = comm.device
    n = rows // world
    gathered = comm.all_gather(_shard(step, rank, n, device), parity)
    torch.accelerator.synchronize(device)
    expected = torch.cat([_shard(step, r, n, device) for r in range(world)])
    same = gathered.view(torch.int32) == _sanitized(expected).view(torch.int32)
    assert bool(same.all()), ("all-gather", step, rows)
    shard = oproj.run(_inputs(step, rank, rows, device), out[:n], parity)
    torch.accelerator.synchronize(device)
    _check_armed(comm, parity, rows)
    stock, exact, scale = _reference(step, world, rank, rows, device, weights)
    if bitwise:
        assert bool((shard.view(torch.int16) == stock.view(torch.int16)).all()), (
            "cublas shard != stock", step, rows,
        )  # fmt: skip
    # Each partial within half a BF16 ulp of its exact value, then one rounding.
    error = (shard.double() - exact).abs().max().item()
    assert error <= (2 * world + 1) * scale * 2**-8, ("shard", step, rows, error)


def _worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import (
            sp_comm,
        )
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj import sp
        from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Context

        rank, world = dist.get_rank(), dist.get_world_size()
        group = dist.group.WORLD
        try:  # (older torch: symmetric memory is enabled per group)
            from torch.distributed._symmetric_memory import enable_symm_mem_for_group

            enable_symm_mem_for_group(group.group_name)
        except (ImportError, AttributeError):
            pass
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        ctx = Context(
            arch="rubin", tp=world, rank=rank, dcp=world, sms=sms, q=4,
            max_rows=CAP, max_context=1 << 17, bf16_state=False, dp=1, ep=world,
            sequence_parallel=True, heads_per_rank=K // 128,
        )  # fmt: skip
        comm = sp_comm.SpComm(group, device, world, rank, rows_cap=CAP)
        weights = [_weight(r, device) for r in range(world)]
        tables = {
            "cublas": "1-:cublas",
            "warp": f"1-{sp.WARP_ROWS}:warp",
            "default": sp.SP_OPROJ[4],
        }
        out = torch.empty((CAP // world, WIDTH), dtype=torch.bfloat16, device=device)
        step = 0
        for name, table in tables.items():
            sp.SP_OPROJ[world] = table
            rows = WARP_ROWS if name == "warp" else ROWS
            oproj = sp.SpOproj(weights[rank], comm, sorted(set(rows)), ctx)
            for r in rows:
                _step(
                    comm, oproj, step, rank, world, r, step % 2, out, weights,
                    bitwise=_stock_bits(oproj.choice(r).tactic),
                )  # fmt: skip
                step += 1
        # A captured two-block step (parities 0, 1), replayed between eager steps.
        sp.SP_OPROJ[world] = tables["default"]
        oproj = sp.SpOproj(weights[rank], comm, (GRAPH_ROWS, 324), ctx)
        n = GRAPH_ROWS // world
        shards = [_shard(0, rank, n, device) for _ in range(2)]
        attns = [_inputs(0, rank, GRAPH_ROWS, device) for _ in range(2)]
        outs = [torch.empty_like(out[:n]) for _ in range(2)]

        def block():
            for p in range(2):
                comm.all_gather(shards[p], p)
                oproj.run(attns[p], outs[p], p)

        # Consecutive blocks of one parity need a TP collective between them (in the
        # model: the stock final sp_all_gather between steps).
        dist.barrier()
        block()  # (warm: every rank, the same launches)
        torch.accelerator.synchronize(device)
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            block()
        for replay in range(REPLAYS):
            base = 1000 + 2 * replay
            for p in range(2):
                shards[p].copy_(_shard(base + p, rank, n, device))
                attns[p].copy_(_inputs(base + p, rank, GRAPH_ROWS, device))
            dist.barrier()  # (the previous eager step may have used parity 0)
            graph.replay()
            torch.accelerator.synchronize(device)
            for p in range(2):
                _check_armed(comm, p, GRAPH_ROWS)
                stock, exact, scale = _reference(
                    base + p, world, rank, GRAPH_ROWS, device, weights
                )
                error = (outs[p].double() - exact).abs().max().item()
                assert error <= (2 * world + 1) * scale * 2**-8, ("graph", replay, p)
            _step(
                comm, oproj, 2000 + replay, rank, world, 324, replay % 2, out,
                weights, bitwise=_stock_bits(oproj.choice(324).tactic),
            )  # fmt: skip
        dist.barrier()
    finally:
        dist.destroy_process_group()


# The online-MXFP8 o-projection: an "mxfp8" row (library GEMM + RS push) and the
# OprojPushMx rows (the GEMM pushes to the owners), each twice (both parities).
MX_ROWS = (64, 256, 324, 384)


def _mx(x):
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    out = mxfp8_act.Mxfp8Buffer(x.shape[0], x.shape[1], x.device)
    mxfp8_act.quantize(x, out=out)
    return out


def _mx_stock(step, world, rank, rows, device, wqs):
    """(stock shard, partial scale): sanitize(bf16(rank-order FP32 sum of the stock
    MXFP8 linear's BF16 partials)) of this rank's rows."""
    n = rows // world
    mine = slice(rank * n, (rank + 1) * n)
    total, scale = None, 0.0
    for src in range(world):
        xq = _mx(_inputs(step, src, rows, device))
        partial = torch._scaled_mm(
            xq.data, wqs[src].data.t(), scale_a=xq.scale.view(torch.float8_e8m0fnu),
            scale_b=wqs[src].scale.view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16,
        )  # fmt: skip
        scale = max(scale, partial.float().abs().max().item())
        partial = _sanitized(partial)[mine].float()
        total = partial if total is None else total + partial
    return _sanitized(total.bfloat16()), scale


def _mx_check(shard, step, world, rank, rows, device, wqs, bitwise):
    stock, scale = _mx_stock(step, world, rank, rows, device, wqs)
    same = shard.view(torch.int16) == stock.view(torch.int16)
    if bitwise:
        assert bool(same.all()), ("mxfp8 shard != stock", step, rows)
    # Every partial within half a BF16 ulp of its FP32-accumulated value (stock's and
    # this one's), then the owner's FP32 sum and one rounding.
    error = (shard.float() - stock.float()).abs().max().item()
    assert error <= (3 * world + 1) * scale * 2**-8, ("mx shard", step, rows, error)
    assert same.float().mean().item() > 0.99, ("mx shard bits", step, rows)


def _mx_worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import (
            sp_comm,
        )
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import mxfp8
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj import sp
        from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Context

        rank, world = dist.get_rank(), dist.get_world_size()
        group = dist.group.WORLD
        try:  # (older torch: symmetric memory is enabled per group)
            from torch.distributed._symmetric_memory import enable_symm_mem_for_group

            enable_symm_mem_for_group(group.group_name)
        except (ImportError, AttributeError):
            pass
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        ctx = Context(
            arch="rubin", tp=world, rank=rank, dcp=world, sms=sms, q=4,
            max_rows=CAP, max_context=1 << 17, bf16_state=False, dp=1, ep=world,
            sequence_parallel=True, heads_per_rank=K // 128,
        )  # fmt: skip
        comm = sp_comm.SpComm(group, device, world, rank, rows_cap=CAP)
        wqs = [_mx(_weight(r, device)) for r in range(world)]
        projection = mxfp8.Mxfp8Projection(
            (wqs[rank].data, wqs[rank].scale), MX_ROWS, family="o_proj",
            tactics=dict.fromkeys(MX_ROWS, "scaled_mm"),
        )  # fmt: skip
        oproj = sp.SpOproj(projection, comm, MX_ROWS, ctx)
        tactics = {r: oproj.choice(r).tactic for r in MX_ROWS}
        assert tactics[324] == tactics[384] == "mx_push" and not oproj.pushes(324), tactics
        out = torch.empty((CAP // world, WIDTH), dtype=torch.bfloat16, device=device)
        for step, rows in enumerate(MX_ROWS * 2):
            n = rows // world
            xq = _mx(_inputs(step, rank, rows, device))
            shard = oproj.run(xq, out[:n], step % 2)
            torch.accelerator.synchronize(device)
            _check_armed(comm, step % 2, rows)
            bitwise = tactics[rows] == ("mxfp8", "scaled_mm")
            _mx_check(shard, step, world, rank, rows, device, wqs, bitwise)
        # A captured two-block step (parities 0, 1) at the largest rows, replayed.
        rows = MX_ROWS[-1]
        n = rows // world
        xqs = [_mx(_inputs(0, rank, rows, device)) for _ in range(2)]
        outs = [torch.empty_like(out[:n]) for _ in range(2)]

        def block():
            for p in range(2):
                oproj.run(xqs[p], outs[p], p)

        dist.barrier()
        block()  # (warm: every rank, the same launches)
        torch.accelerator.synchronize(device)
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            block()
        for replay in range(REPLAYS):
            base = 3000 + 2 * replay
            for p in range(2):
                fresh = _mx(_inputs(base + p, rank, rows, device))
                xqs[p].data.copy_(fresh.data)
                xqs[p].scale.copy_(fresh.scale)
            dist.barrier()
            graph.replay()
            torch.accelerator.synchronize(device)
            for p in range(2):
                _check_armed(comm, p, rows)
                _mx_check(outs[p], base + p, world, rank, rows, device, wqs, False)
        dist.barrier()
    finally:
        dist.destroy_process_group()


VOCAB_SHARD, VOCAB_PAD = 640, 40  # rows per rank; the last rank's padding rows


def _embed_worker(env: dict[str, str]) -> None:
    """SpEmbedScatter: this rank's shard of table[ids] (zeros for ids >= vocab),
    sanitized as the stock all-reduce's x + 0 (-0 -> +0), bit for bit; re-armed."""
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    try:
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
        vocab = world * VOCAB_SHARD - VOCAB_PAD
        table = torch.randn(world * VOCAB_SHARD, WIDTH, generator=_generator("e"))
        table = table.bfloat16()
        words = table.view(torch.int32)
        words[:, 7::101] = SENTINEL  # (+0, -0) pairs
        words[:, 9::83] = 0x8000  # (-0, +0) pairs
        table = table.to(device)
        weight = table[rank * VOCAB_SHARD : (rank + 1) * VOCAB_SHARD].contiguous()
        for step, rows in enumerate((8, 128, 384, 4, 512)):
            gen = _generator("ids", step)
            ids = torch.randint(0, world * VOCAB_SHARD, (rows,), generator=gen)
            ids[:4] = torch.tensor([0, vocab - 1, vocab, world * VOCAB_SHARD - 1])
            ids = ids.to(torch.int32 if step % 2 else torch.int64).to(device)
            n = rows // world
            out = torch.full((n, WIDTH), 7.0, dtype=torch.bfloat16, device=device)
            dist.barrier()
            comm.embed_scatter(ids, weight, rank * VOCAB_SHARD, vocab, out, 1)
            torch.accelerator.synchronize(device)
            mine = ids[rank * n : (rank + 1) * n].long()
            want = torch.where(
                (mine < vocab)[:, None], table[mine], torch.zeros_like(table[mine])
            )
            want = (want.float() + 0.0).bfloat16()  # -0 + 0 = +0
            assert torch.equal(out.view(torch.int16), want.view(torch.int16)), rows
            _check_armed(comm, 1, rows)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _distributed_run(fn, world_size: int) -> None:
    port = str(get_open_port())
    ctx = mp.get_context("spawn")
    processes = []
    for rank in range(world_size):
        env = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": port,
        }
        process = ctx.Process(target=fn, args=(env,))
        processes.append(process)
        process.start()
    # One deadline for the group: a hung rank fails the test, never CI.
    deadline = time.monotonic() + 300
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
    assert [process.exitcode for process in processes] == [0] * world_size


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2 or not _has_multicast_support(),
    reason="Need 2 GPUs with symmetric-memory multicast.",
)
def test_sp_comm_and_oproj_rs():
    """SpComm.all_gather and SpOproj (cublas, warp, default table) on 4 ranks when
    there are 4 GPUs (the C512 TP group), else 2: bitwise all-gather, stock-exact
    cublas reduce-scatter, FP64-bounded warp and cuBLASLt partials, re-arming across
    parities and row counts (4..512), and a captured two-block step replayed."""
    pytest.importorskip("cutlass")
    world = 4 if torch.accelerator.device_count() >= 4 else 2
    _distributed_run(_worker, world_size=world)


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2 or not _has_multicast_support(),
    reason="Need 2 GPUs with symmetric-memory multicast.",
)
def test_sp_embed_scatter():
    """SpComm.embed_scatter (the SP decode's input embedding as the reduce-scatter
    of the vocab-parallel gather) on 4 ranks (else 2): bitwise table rows, zeros past
    the vocabulary, -0 as +0, the mailboxes re-armed."""
    pytest.importorskip("cutlass")
    world = 4 if torch.accelerator.device_count() >= 4 else 2
    _distributed_run(_embed_worker, world_size=world)


@pytest.mark.skipif(
    torch.accelerator.device_count() < 4 or not _has_multicast_support(),
    reason="Need 4 GPUs with symmetric-memory multicast.",
)
def test_sp_oproj_rs_mxfp8_push():
    """SpOproj on an online-MXFP8 o-projection (NVFP4 checkpoint KDA layers) on 4
    ranks: the library GEMM + RS push rows bitwise stock, the OprojPushMx rows (the
    GEMM pushes to the row owners, the owner kernel push=False) within the GEMM
    summation-order bound of stock, mailboxes re-armed, and a captured two-parity step
    replayed."""
    pytest.importorskip("cutlass")
    _distributed_run(_mx_worker, world_size=4)
