# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode, TP8 = DCP8 wide verify rows (> 64 token rows,
Rubin): WideDsa's kernels (``dsa.wide.compile_core``) over 8 ranks on two 4-GPU
nodes of one NVLink domain, 12 DSA heads per rank, 96 gathered heads padded to
trtllm-gen's 128 (FlashInfer sparse, per-tensor FP8 query, "proj" partials), the
widened DSA box of the deployment's largest graph (1536 rows, partials False, as
``DcpBox.widened`` builds it for WideDsa).

Per (q, rows) case (``CASES``: q2-q8, 65-256 across the tile edges, 384, 768 and the
1536-row capacity anchor; requests of zero total length stand for the padded rows of
a FULL graph):

- the query: the fork mode's publish-only MlaQuery and QueryWait (every wide row
  forks at TP8 as at TP16): every rank's query rows and KV row bitwise equal to the
  query without DCP (``query_stage.compile_query``), and the gathered mailbox
  bitwise equal to every rank's rows (padded heads zero);
- the "proj" DcpValue (W_UV bmm of the 96 gathered heads, value-row exchange, LSE
  merge of this rank's heads, gate) against the FP32 reference of the stock
  combine (an FP32 reorder and one BF16 rounding: tolerance), and the q mailbox
  re-armed for the next layer;
- the chain (query, QueryWait, DcpValue) replayed from one CUDA graph: bitwise equal
  to its eager run on every replay.

Launch: one pytest per node over two nodes (``srun -N2 --ntasks-per-node=1``), with
MASTER_ADDR (node 0) and MASTER_PORT set; each pytest spawns its node's 4 ranks
(global rank = SLURM_NODEID x 4 + local). Skipped elsewhere."""

import os
import time

import pytest
import torch
import torch.distributed as dist

LOCAL, NODES = 4, 2
WORLD = LOCAL * NODES
HEADS, Q_HEADS, CAP = 12, 128, 1536
V, INDEX_DIM, INDEX_HEADS = 128, 128, 32
GATE = HEADS * V
PITCH = 2112 + GATE + INDEX_DIM + INDEX_HEADS
UNIT, FP8_ROW, INDEX_ROW = 64, 576, 132
EPS = 1e-6
# (q, rows): every q from 2 to 8, the first wide row past 64 (65, 66, 68, 72) and
# both sides of the 128 / 192 / 256 tile edges (129, 130, 132, 194, 196), mid rows,
# and the 1536-row capacity anchor (the widened box's every row).
CASES = (
    (2, 66), (2, 130), (2, 194), (3, 129), (4, 68), (4, 128), (4, 132), (4, 192),
    (4, 196), (4, 256), (4, 768), (5, 65), (6, 384), (7, 196), (8, 72), (8, 256),
    (8, 1536),
)  # fmt: skip
TIMEOUT = 600


def _launch():
    """(node, master address, port) of a two-node launch, else None."""
    if os.environ.get("SLURM_NNODES") != str(NODES):
        return None
    if "SLURM_NODEID" not in os.environ or "MASTER_ADDR" not in os.environ:
        return None
    port = int(os.environ.get("MASTER_PORT", "29571"))
    return int(os.environ["SLURM_NODEID"]), os.environ["MASTER_ADDR"], port


def _rubin_node() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < LOCAL:
        return False
    if {torch.cuda.get_device_capability(i) for i in range(LOCAL)} != {(10, 7)}:
        return False
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        return _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)
    except Exception:
        return False


pytestmark = [
    pytest.mark.distributed(num_gpus=WORLD),
    pytest.mark.skipif(
        _launch() is None or not _rubin_node(),
        reason="needs 2 nodes x 4 Rubin (SM107) GPUs of one NVLink domain, "
        "launched with srun -N2 and MASTER_ADDR",
    ),
]


def _inputs(batch, seed):
    """Packed projection rows, layer weights, per-tensor FP8 MLA / indexer caches and
    slots of one rank's DSA layer at 12 heads."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    bf, dev = torch.bfloat16, "cuda"

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=dev) * scale).to(bf)

    weights = (
        (torch.rand(1536, generator=gen, device=dev) + 0.5).to(bf),
        (torch.rand(512, generator=gen, device=dev) + 0.5).to(bf),
        randn(HEADS * 192, 1536, scale=1 / 40),
        randn(HEADS, 128, 512, scale=1 / 12),
        randn(HEADS, 128, 512, scale=1 / 16).transpose(1, 2),
    )
    blocks = -(-(batch + 1) // UNIT) + 1
    mla = torch.zeros(blocks, UNIT, FP8_ROW, dtype=torch.uint8, device=dev)
    index = torch.zeros(blocks, UNIT, INDEX_ROW, dtype=torch.uint8, device=dev)
    slots = torch.randperm(blocks * UNIT, generator=gen, device=dev)[:batch]
    slots[1::7] = -1  # other ranks' tokens write nothing
    return randn(batch, PITCH), weights, mla, index, slots


def _case(box, size, rank, local, q, batch, parity):
    import cutlass

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        fake,
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import wide
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.dcp_stage import (
        publish_args,
        run_value,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.query_stage import (  # noqa: E501
        compile_query,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.r1 import DsaBox

    dev, bf = box.q.device, torch.bfloat16
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    s = stream(dev)
    projected, weights, mla, index, slots = _inputs(batch, 100 + rank)
    specs = tuple(layout(w) for w in (*weights, mla, index))
    mode = wide.wide_overlap(batch)
    assert mode == "fork" and wide.query_publishes_only(batch, size, mode)
    query_kernel, _, value = wide.compile_core(
        batch, HEADS, PITCH, specs, local, q, size, Q_HEADS, sms, fp8=True,
        lse_base_e=False, overlap=mode, partials="proj", cap=box.cap,
    )  # fmt: skip
    wait = wide.compile_query_wait(
        batch, HEADS, size, Q_HEADS, sms, box.cap, True, overlap=mode
    )
    unit = torch.ones(1, device=dev, dtype=torch.float32)

    def query_args(cache, query, q_c):
        args = (to_cute(projected), *(to_cute(w) for w in weights[:4]))
        args += (to_cute(cache), to_cute(slots, 8), to_cute(unit, 4), to_cute(unit, 4))
        return (*args, cutlass.Float32(EPS), to_cute(query), to_cute(q_c))

    # The q gather: publish-only MlaQuery (no q_c: QueryNorm writes it), QueryWait.
    query = torch.zeros(batch, HEADS, 576, device=dev, dtype=torch.uint8)
    q_c = torch.full((batch, 1536), -7.0, device=dev, dtype=bf)
    tp_mla = mla.clone()
    publish = publish_args(box, parity)
    torch.accelerator.synchronize()
    dist.barrier()
    query_kernel(*query_args(mla, query, q_c), *publish, s)
    wait(publish[0], s)
    torch.accelerator.synchronize()
    # The query rows and KV row as the query without DCP writes them (bitwise).
    fakes = tuple(
        fake(shape, stride, dtype, 4 if dtype == torch.float32 else 16)
        for shape, stride, dtype in (layout(w) for w in (*weights[:4], tp_mla))
    )
    tp_kernel = compile_query(batch, HEADS, PITCH, fakes, local, fp8=True)
    tp_query = torch.zeros_like(query)
    tp_kernel(*query_args(tp_mla, tp_query, torch.zeros_like(q_c)), s)
    torch.accelerator.synchronize()
    assert torch.equal(query, tp_query) and torch.equal(mla, tp_mla)
    assert (q_c == -7.0).all()  # (publish-only: q_c untouched)
    queries = [torch.empty_like(query) for _ in range(size)]
    dist.all_gather(queries, query)
    gathered = box.q[parity, :batch]
    assert torch.equal(gathered[:, : size * HEADS], torch.cat(queries, 1))
    assert (gathered[:, size * HEADS :] == 0).all()

    # The "proj" value over every rank's padded-head latent rows and base-2 LSEs.
    gen = torch.Generator(device=dev).manual_seed(rank)
    reqs = batch // q
    latent = torch.randn(batch, Q_HEADS, 512, generator=gen, device=dev).to(bf)
    lse = torch.randn(batch, Q_HEADS, generator=gen, device=dev) * 3
    valid = torch.randint(1, 100, (batch,), generator=gen, device=dev).int()
    row = torch.arange(batch, device=dev)
    valid[(row + rank) % 5 == 0] = 0  # this rank owns none of the row's keys
    shared = torch.Generator(device=dev).manual_seed(7)  # the same on every rank
    totals = torch.randint(0, 3, (reqs,), generator=shared, device=dev).int()
    dsa_box = DsaBox(box, (batch,), "proj", fp8=True, verify_q=q)
    w_uv = weights[4]
    w_uv_gathered = dsa_box.gather_heads(w_uv)
    out = torch.zeros(batch, GATE, device=dev, dtype=bf)
    args = value, dsa_box, parity, publish, HEADS, (latent, lse, valid)
    run_value(*args, totals, None, projected, out, w_uv_gathered)
    torch.accelerator.synchronize()
    dist.barrier()
    assert (box.q[parity, :batch, : size * HEADS].view(torch.int32) == -1).all()

    tensors = latent, lse, valid
    everything = [torch.empty_like(t) for t in tensors for _ in range(size)]
    for i, t in enumerate(tensors):
        dist.all_gather(everything[i * size : (i + 1) * size], t)
    mine = slice(rank * HEADS, (rank + 1) * HEADS)
    all_rows = torch.stack(everything[:size])[:, :, mine]
    all_lse = torch.stack(everything[size : 2 * size])[:, :, mine]
    all_valid = torch.stack(everything[2 * size :])
    live = (all_valid > 0).unsqueeze(-1).expand_as(all_lse)
    lse2 = torch.where(live, all_lse, torch.full_like(all_lse, -3.0e38))
    weight = torch.exp2(lse2 - lse2.amax(0))
    weight = torch.where(live, weight, torch.zeros_like(weight))
    weight = weight / weight.sum(0).clamp_min(1e-30)
    merged = (weight.unsqueeze(-1) * all_rows.float()).sum(0)
    value_ref = torch.einsum("bhk,hkv->bhv", merged, w_uv.float())
    gate = projected[:, 2112 : 2112 + GATE].float().view(batch, HEADS, 128)
    ref = (value_ref * torch.sigmoid(gate)).reshape(batch, GATE)
    ref[(totals.repeat_interleave(q) == 0)] = 0
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    eager = out.clone()

    # The served steps replay the chain (query publish, QueryWait, DcpValue) from
    # one CUDA graph: bitwise the eager chain, the q mailbox re-armed each time.
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream(dev)
    side.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(side), torch.cuda.graph(
        graph, stream=side, capture_error_mode="thread_local"
    ):
        query_kernel(*query_args(mla, query, q_c), *publish, stream(dev))
        wait(publish[0], stream(dev))
        run_value(*args, totals, None, projected, out, w_uv_gathered)
    dist.barrier()
    for replay in range(3):
        out.zero_()
        graph.replay()
        torch.accelerator.synchronize()
        dist.barrier()
        assert torch.equal(out, eager), (q, batch, replay)
        armed = box.q[parity, :batch, : size * HEADS].view(torch.int32) == -1
        bad = (~armed).nonzero()
        assert armed.all(), (q, batch, replay, rank, bad.shape[0], bad[:4].tolist())
        # (no rank publishes the next replay's query into a mailbox still checked)
        torch.accelerator.synchronize()
        dist.barrier()
    del graph


def _worker(local, node, addr, port, result):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import DcpBox
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        trtllm_gen_query_heads,
    )

    rank = node * LOCAL + local
    torch.accelerator.set_device_index(local)
    dev = torch.device("cuda", local)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{addr}:{port}",
        rank=rank,
        world_size=WORLD,
        device_id=dev,
    )
    group = dist.group.WORLD
    try:  # (older torch: symmetric memory is enabled per group)
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        enable_symm_mem_for_group(group.group_name)
    except (ImportError, AttributeError):
        pass
    assert trtllm_gen_query_heads(WORLD * HEADS) == Q_HEADS
    # The model's DSA box, then its widened box as WideDsa takes it (no partials).
    base = DcpBox(group, dev, WORLD, rank, HEADS, q_heads=Q_HEADS)
    box = base.widened(CAP, partials=False)
    assert box.q.shape == (2, CAP, Q_HEADS, 576) and box.partial is None
    for i, (q, batch) in enumerate(CASES):
        start = time.monotonic()
        _case(box, WORLD, rank, local, q, batch, i % 2)
        if rank == 0:
            print(f"TP8 wide DSA q{q} M{batch}: ok {time.monotonic() - start:.1f} s")
    dist.barrier()
    dist.destroy_process_group()
    result.put((rank, "ok"))


def test_tp8_wide_dsa_dcp8():
    """WideDsa's query gather, "proj" DcpValue and graph replay at DCP8 / 12 heads
    over two nodes (module docstring); a hang fails within the deadline."""
    pytest.importorskip("cutlass")
    import torch.multiprocessing as mp

    node, addr, port = _launch()
    ctx = mp.get_context("spawn")
    result = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(i, node, addr, port, result))
        for i in range(LOCAL)
    ]
    for p in procs:
        p.start()
    deadline = time.monotonic() + TIMEOUT
    done = []
    while len(done) < LOCAL:
        if time.monotonic() > deadline:
            for p in procs:
                p.kill()
            pytest.fail(f"TP8 wide DSA: no result in {TIMEOUT} s (a hang?)")
        if any(p.exitcode not in (None, 0) for p in procs):
            for p in procs:
                p.kill()
            pytest.fail("TP8 wide DSA: a rank failed (see its traceback)")
        try:
            done.append(result.get(timeout=5))
        except Exception:
            continue
    for p in procs:
        p.join(timeout=60)
    assert sorted(r for r, _ in done) == [node * LOCAL + i for i in range(LOCAL)]
