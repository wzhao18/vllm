# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from the source branch's tests/distributed/test_kimi_k3_sdec_sp_dsa.py for the SM107
# specialized decode (vllm/models/kimi_k3/nvidia/specialized_decode/sm_107).
"""Kimi K3.1 specialized decode under sequence parallelism (C512: TP4 = DCP4, 24 DSA
heads per rank, q4), 4 ranks over one node's NVLink multicast: the SP DSA box
(dsa.sp.sp_dsa_box: FP8 q mailbox [2, 512, 128, 576], no BF16 partials), the
24-head MlaQuery's publish-only q gather polled by QueryWait (every rank's box holds
every rank's query rows bit for bit; padded heads zero; query, q_c-free and KV row as
the query without DCP), and the "proj" DcpValue (W_UV bmm of the 96 gathered heads,
value-row exchange, LSE merge of this rank's heads, gate) against an FP32 reference,
the q mailbox re-armed for the next layer. Rows 4, 64, 324 and 512 (1, 16, 81 and
128 requests per DP group); ``rank1_late`` starts rank 1's query ~0.1 s late."""

import time

import pytest
import torch
import torch.distributed as dist

from vllm.utils.network_utils import get_open_port
from vllm.platforms import current_platform  # noqa: E402

# The SM107 specialized decode (specialized_decode/sm_107) serves VR200 only.
_SM107_ONLY = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)),
    reason="the SM107 specialized decode runs on compute capability 10.7 only",
)
pytestmark = _SM107_ONLY

HEADS, Q_HEADS, CAP, Q = 24, 128, 512, 4
V, INDEX_DIM, INDEX_HEADS = 128, 128, 32
GATE = HEADS * V
PITCH = 2112 + GATE + INDEX_DIM + INDEX_HEADS
UNIT, FP8_ROW, INDEX_ROW = 64, 576, 132
EPS = 1e-6
CASES = (4, 64, 324, 512)


def _has_multicast_support() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        return _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)
    except Exception:
        return False


def _inputs(batch, seed):
    """Packed projection rows, layer weights, per-tensor FP8 MLA / indexer caches and
    slots of one rank's DSA layer at 24 heads."""
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


def _case(box, size, rank, batch, parity, skew):
    import cutlass

    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.common.host import (
        fake,
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa import sp
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.dcp_stage import (
        publish_args,
        run_value,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.overlap import (
        overlap_for,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.prepared import (
        compile_core,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.query_stage import (  # noqa: E501
        compile_query,
        compile_wait,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.r1 import DsaBox

    dev, bf = box.q.device, torch.bfloat16
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    s = stream(dev)
    projected, weights, mla, index, slots = _inputs(batch, 100 + rank)
    specs = tuple(layout(w) for w in (*weights, mla, index))
    mode = overlap_for(batch, sp.SP_QUERY_OVERLAP, True, True)
    assert mode in sp.SP_MODES
    query_kernel, _, value = compile_core(
        batch, HEADS, PITCH, specs, rank, Q, size, Q_HEADS, sms, fp8=True,
        lse_base_e=False, overlap=mode, partials="proj", cap=box.cap,
    )  # fmt: skip
    wait = compile_wait(batch, HEADS, size, Q_HEADS, True, box.cap)
    unit = torch.ones(1, device=dev, dtype=torch.float32)

    def query_args(cache, query, q_c):
        args = (to_cute(projected), *(to_cute(w) for w in weights[:4]))
        args += (to_cute(cache), to_cute(slots, 8), to_cute(unit, 4), to_cute(unit, 4))
        return (*args, cutlass.Float32(EPS), to_cute(query), to_cute(q_c))

    # The q gather: publish-only MlaQuery, then QueryWait.
    query = torch.zeros(batch, HEADS, 576, device=dev, dtype=torch.uint8)
    q_c = torch.full((batch, 1536), -7.0, device=dev, dtype=bf)
    tp_mla = mla.clone()
    publish = publish_args(box, parity)
    torch.accelerator.synchronize()
    dist.barrier()
    if skew and rank == 1:
        torch.cuda._sleep(skew)
    query_kernel(*query_args(mla, query, q_c), *publish, s)
    wait(publish[0], s)
    torch.accelerator.synchronize()
    # The query rows and KV row as the query without DCP writes them (bitwise).
    fakes = tuple(
        fake(shape, stride, dtype, 4 if dtype == torch.float32 else 16)
        for shape, stride, dtype in (layout(w) for w in (*weights[:4], tp_mla))
    )
    tp_kernel = compile_query(batch, HEADS, PITCH, fakes, rank, fp8=True)
    tp_query = torch.zeros_like(query)
    tp_q_c = torch.zeros_like(q_c)
    tp_kernel(*query_args(tp_mla, tp_query, tp_q_c), s)
    torch.accelerator.synchronize()
    assert torch.equal(query, tp_query) and torch.equal(mla, tp_mla)
    queries = [torch.empty_like(query) for _ in range(size)]
    dist.all_gather(queries, query)
    gathered = box.q[parity, :batch]
    assert torch.equal(gathered[:, : size * HEADS], torch.cat(queries, 1))
    assert (gathered[:, size * HEADS :] == 0).all()

    # The "proj" value over every rank's padded-head latent rows and base-2 LSEs.
    gen = torch.Generator(device=dev).manual_seed(rank)
    reqs = batch // Q
    latent = torch.randn(batch, Q_HEADS, 512, generator=gen, device=dev).to(bf)
    lse = torch.randn(batch, Q_HEADS, generator=gen, device=dev) * 3
    valid = torch.randint(1, 100, (batch,), generator=gen, device=dev).int()
    row = torch.arange(batch, device=dev)
    valid[(row + rank) % 5 == 0] = 0  # this rank owns none of the row's keys
    shared = torch.Generator(device=dev).manual_seed(7)  # the same on every rank
    totals = torch.randint(0, 3, (reqs,), generator=shared, device=dev).int()
    dsa_box = DsaBox(box, (batch,), "proj", fp8=True, verify_q=Q)
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
    ref[(totals.repeat_interleave(Q) == 0)] = 0
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def _worker(rank, size, port, skew):
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa import sp

    torch.accelerator.set_device_index(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=size,
        device_id=dev,
    )
    group = dist.group.WORLD
    try:  # (older torch: symmetric memory is enabled per group)
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        enable_symm_mem_for_group(group.group_name)
    except (ImportError, AttributeError):
        pass
    box = sp.sp_dsa_box(group, dev, rank, heads=HEADS, cap=CAP, interleave=1)
    assert box.q.shape == (2, CAP, Q_HEADS, 576) and box.partial is None
    for i, batch in enumerate(CASES):
        _case(box, size, rank, batch, i % 2, skew)
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 4 or not _has_multicast_support(),
    reason="Need 4 GPUs with symmetric-memory multicast.",
)
@pytest.mark.parametrize("skew", [0, 200_000_000], ids=["aligned", "rank1_late"])
def test_sp_dsa_dcp4_mailboxes(skew):
    """The SP DSA q gather and "proj" value exchange at DCP4 / 24 heads (see the
    module docstring); a hang fails within the deadline."""
    pytest.importorskip("cutlass")
    import torch.multiprocessing as mp

    args = (4, get_open_port(), skew)
    context = mp.start_processes(
        _worker, args=args, nprocs=4, join=False, start_method="spawn"
    )
    deadline = time.monotonic() + 900
    while not context.join(timeout=10):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.kill()
            pytest.fail("SP DSA DCP4 mailboxes: no progress in 15 min (a hang?)")
