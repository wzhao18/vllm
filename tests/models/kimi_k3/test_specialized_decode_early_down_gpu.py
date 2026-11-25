# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode, Rubin TP16 verify merged batches (``front.EARLY_DOWN_WIDE``):
``SharedDown`` (writers) launched on a side stream BEFORE the merged front
(``MoeFrontTmem64Ready``), its h load polling the front's gate/up CTAs' count. Back to
back over fresh inputs (no sync between calls; the main stream joins the side stream
after each, as the MoE tail does): the front box and h bitwise equal
``MoeFrontTmem64``'s, the mailbox rows bitwise equal the plain SharedDown's on that h,
within FP32 noise of h @ down^T, and the count re-armed (2 GPUs: the front box is
multicast memory)."""

import pytest
import torch

pytest.importorskip("cutlass")

K, TP, CAP = 7168, 16, 64
VALID, SPLIT, SHARD, HIDDEN, BOX_ROWS = 56, 64, 224, 384, 512
CASES = [(16, 8), (32, 24), (64, 4), (64, 24)]  # (rows, SharedDown SMs)


def _rubin():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _rubin(), reason="Rubin / Blackwell GPU required")


def test_early_down_table():
    """Only merged batches on the TMEM front, TP16 verify; writers = gate/up CTAs."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import front
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.front_tmem import (
        MoeFrontTmem64Ready,
    )

    table = front.batch_table(front.EARLY_DOWN_WIDE[16][1])
    merged = front.batch_table(f"{front.MERGED[16]},{front.MERGED_WIDE[16]}")
    tmem = front.batch_table(front.FRONT_TMEM_WIDE[SPLIT + SHARD])
    assert all(merged.get(b) and tmem.get(b) and n > 0 for b, n in table.items())
    assert not front.EARLY_DOWN_WIDE[16][0]
    kernel = MoeFrontTmem64Ready(valid=VALID, split_rows=SPLIT, shard=SHARD, hidden=HIDDEN)
    assert kernel.writers == front.front_tmem_writers(HIDDEN) == 48


def test_early_down_torchrun():
    """The torchrun check below on 2 local GPUs (skipped without them)."""
    import subprocess
    import sys

    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs (multicast memory)")
    command = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node=2"]
    subprocess.run([*command, __file__], check=True, timeout=900)


def _mailbox_rows(boxes, parity, rank, rows):
    shard = K // TP
    first = (parity * TP + rank) * CAP
    return torch.cat([b.view(-1, shard)[first : first + rows] for b in boxes], dim=1)


def _main():
    import os

    import cutlass as c
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    from cutlass import Int32

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (
        matrix_view,
        weight_map,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import front
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.shared import (
        compile_down,
        weight_maps,
    )

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())
    box = symm_mem.empty((1, BOX_ROWS, front.WIDTH), dtype=torch.bfloat16, device="cuda")
    mc = int(symm_mem.rendezvous(box, dist.group.WORLD).multicast_ptr or 0)
    assert mc, "multicast memory"
    plain_front = front.compile_front_tmem(SPLIT, VALID, SHARD, HIDDEN, dev.index)
    ready_front = front.compile_front_tmem(SPLIT, VALID, SHARD, HIDDEN, dev.index, True)
    writers = front.front_tmem_writers(HIDDEN)
    side = torch.cuda.Stream(dev)
    failures = 0
    for rows, sms in CASES:
        gen = torch.Generator(device="cuda").manual_seed(1000 * rows + sms + rank)
        packed = torch.randn(SPLIT + SHARD, K, generator=gen, device="cuda") * 0.02
        packed[VALID:SPLIT] = 0
        packed = packed.bfloat16()
        gate_up = (torch.randn(2 * HIDDEN, K, generator=gen, device="cuda") * 0.02)
        gate_up = gate_up.bfloat16()
        down = (torch.randn(K, HIDDEN, generator=gen, device="cuda") * 0.02).bfloat16()
        # the descriptor tensors stay referenced: a CuTe view / pointer does not hold them
        front_maps = (weight_map(packed, 128), weight_map(gate_up, 64))
        ptrs = tuple(c.Int64(m.data_ptr()) for m in front_maps)
        down_maps = weight_maps(gate_up, down)
        maps = to_cute(down_maps, 64)
        plain_down = compile_down(TP, rows, dev.index, sms)
        early_down = compile_down(TP, rows, dev.index, sms, writers=writers)
        count = torch.zeros(2, dtype=torch.int32, device=dev)
        calls = 4
        xs = [torch.randn(rows, K, generator=gen, device="cuda").bfloat16() for _ in range(calls)]

        def boxes():
            return [torch.full((2 * TP * CAP * (K // TP),), float("nan"), device=dev)
                    .bfloat16() for _ in range(TP)]  # fmt: skip

        def front_args(x, h):
            args = (matrix_view(packed), matrix_view(x), to_cute(h), to_cute(box))
            return args + (c.Int64(mc), Int32(rank * VALID), Int32(rank * SHARD),
                           c.Float32(4.0), c.Float32(25.0))  # fmt: skip

        # reference: plain front, then plain SharedDown on its h (synchronized)
        refs = []
        for x in xs:
            h = torch.zeros(64, HIDDEN, dtype=torch.bfloat16, device=dev)
            box.view(torch.int32).fill_(-0x80000000)
            torch.cuda.synchronize()
            dist.barrier()
            plain_front(*front_args(x, h), *ptrs, stream(dev))
            out = boxes()
            peers = torch.tensor([b.data_ptr() for b in out], dtype=torch.int64, device=dev)
            plain_down(to_cute(h), maps, to_cute(peers), Int32(1), Int32(rank), Int32(CAP),
                       stream(dev))  # fmt: skip
            torch.cuda.synchronize()
            dist.barrier()
            refs.append((box[0, :rows].clone(), h[:rows].clone(), out))
        # early: SharedDown first on the side stream, then the counting front; back to
        # back (only the per-call join, no host sync); the box is copied per call
        # the counting front launched once first: its lazy module load waits for the
        # device, i.e. would wait for a SharedDown already polling (``fork_first``)
        warm = torch.zeros(64, HIDDEN, dtype=torch.bfloat16, device=dev)
        ready_front(*front_args(xs[0], warm), *ptrs, c.Int64(count.data_ptr()), stream(dev))
        torch.cuda.synchronize()
        assert count.tolist() == [writers, 0], count.tolist()
        count.zero_()
        hs = [torch.zeros(64, HIDDEN, dtype=torch.bfloat16, device=dev) for _ in xs]
        outs = [boxes() for _ in xs]
        got_box = []
        main = torch.cuda.current_stream(dev)
        box.view(torch.int32).fill_(-0x80000000)
        torch.cuda.synchronize()
        dist.barrier()
        for i, x in enumerate(xs):
            begin, done = torch.cuda.Event(), torch.cuda.Event()
            begin.record(main)
            peers = torch.tensor([b.data_ptr() for b in outs[i]], dtype=torch.int64,
                                 device=dev)  # fmt: skip
            with torch.cuda.stream(side):
                side.wait_event(begin)
                early_down(to_cute(hs[i]), maps, to_cute(peers), Int32(1), Int32(rank),
                           Int32(CAP), to_cute(count), stream(dev))  # fmt: skip
                done.record(side)
            ready_front(*front_args(x, hs[i]), *ptrs, c.Int64(count.data_ptr()), stream(dev))
            main.wait_event(done)
            got_box.append(box[0, :rows].clone())  # stream-ordered after the front
        torch.cuda.synchronize()
        dist.barrier()
        ok = int(count.abs().sum()) == 0
        own = (slice(rank * SHARD, (rank + 1) * SHARD),  # latent columns (BF16)
               slice(3584 + 2 * rank * VALID, 3584 + 2 * (rank + 1) * VALID))  # logits
        for i in range(calls):
            ref_box, ref_h, ref_out = refs[i]
            a = _mailbox_rows(outs[i], 1, rank, rows)
            b = _mailbox_rows(ref_out, 1, rank, rows)
            exact = ref_h.float() @ down.float().t()
            # this rank's own front-box columns (the peer's front of another call may
            # have multicast its columns meanwhile: no barrier between the calls)
            for cols in own:
                ok &= torch.equal(got_box[i][:, cols].view(torch.int16),
                                  ref_box[:, cols].view(torch.int16))  # fmt: skip
            ok &= torch.equal(hs[i][:rows].view(torch.int16), ref_h.view(torch.int16))
            ok &= not a.isnan().any() and torch.equal(a.view(torch.int16), b.view(torch.int16))
            ok &= bool(torch.allclose(a.float(), exact, rtol=1.6e-2, atol=1e-3))
        print(f"rank {rank}: rows {rows} SharedDown {sms} SMs: {'ok' if ok else 'FAIL'} "
              f"count {count.tolist()}", flush=True)  # fmt: skip
        failures += not ok
    dist.barrier()
    dist.destroy_process_group()
    return failures


if __name__ == "__main__":
    import sys

    sys.exit(1 if _main() else 0)
