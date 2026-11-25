# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode, ``SplitKTmemGemm64`` with separate reduce mailboxes
(``separate``: the owners' FP32 mailboxes past the A / x slots, no cluster barrier
between the MMAs and the DSMEM pushes): the MoE front kernels (``MoeFrontTmem64`` at
16 / 32 / 64 rows, ``MoeFrontRows`` N64 at 128 rows; 8 owners) publish bitwise the
same front box and h as the same kernels with the mailboxes in the A / x slots behind
that barrier (2 GPUs: the front box is multicast memory); which tactics take separate
mailboxes (only where they fit below the lone CTA's unchanged shared-memory request);
the 4-owner projections keep the barrier form (FP64 bound, deterministic)."""

import pytest
import torch

pytest.importorskip("cutlass")

K = 7168
TACTIC = (128, 64, 8, 14, 14)  # the TMEM split-K decode tactic at N64 (front, in-proj)


def _rubin():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _rubin(), reason="Rubin / Blackwell GPU required")


def _run(weight, x, separate):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import (
        projection,
        splitk_tmem64,
    )

    cls = splitk_tmem64.SplitKTmemGemm64
    init = cls.__init__

    def barrier_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.separate = False  # the mailboxes in the A / x slots, behind the barrier

    projection.compile_projection_tmem.cache_clear()
    if not separate:
        cls.__init__ = barrier_init
    try:
        prepared = projection.PreparedProjection(weight, {x.shape[0]: TACTIC}).prepare()
    finally:
        cls.__init__ = init
        projection.compile_projection_tmem.cache_clear()
    outs = []
    for _ in range(3):
        out = torch.full((x.shape[0], weight.shape[0]), float("nan"), device=x.device,
                         dtype=torch.bfloat16)  # fmt: skip
        prepared.run(x, out)
        torch.cuda.synchronize()
        outs.append(out)
    return outs


@pytest.mark.parametrize("rows", [16, 32, 64])
@pytest.mark.parametrize("n", [1024, 3040])
def test_projection_mailboxes_unchanged(rows, n):
    """The 4-owner projections' mailboxes (32-row shares, 64 KB at N64) do not fit
    beside the A / x slots: they keep the barrier form; still within FP32
    accumulation noise of FP64 and deterministic over calls."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.splitk import (
        SplitKTactic,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.splitk_tmem64 import (  # noqa: E501
        SplitKTmemGemm64,
    )

    assert not SplitKTmemGemm64(tactic=SplitKTactic(*TACTIC[:4]), k=K).separate
    gen = torch.Generator(device="cuda").manual_seed(1000 * rows + n)
    weight = (torch.randn(n, K, device="cuda", generator=gen) * 0.02).bfloat16()
    x = torch.randn(rows, K, device="cuda", generator=gen).bfloat16()
    got = _run(weight, x, separate=True)
    assert torch.equal(got[0].view(torch.int16), got[2].view(torch.int16))
    exact = x.double() @ weight.double().T
    err = (got[0].double() - exact).abs().max() / exact.abs().max()
    assert err < 1e-2, err  # BF16 output rounding


def test_separate_only_below_the_lone_request():
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.splitk import (
        SplitKTactic,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.splitk_tmem64 import (  # noqa: E501
        ALONE_SMEM,
        SplitKTmemGemm64,
    )

    # the MoE front's 8 owners (16-row shares): 32 KB of mailboxes at N64 fit below
    # the lone request (separate), 64 KB at N128 do not
    front = SplitKTmemGemm64(tactic=SplitKTactic(128, 64, 8, 14), k=K, owners=8)
    assert front.separate and front.smem_bytes == ALONE_SMEM  # the request unchanged
    wide = SplitKTmemGemm64(tactic=SplitKTactic(128, 128, 8, 8), k=K, owners=8)
    assert not wide.separate
    lone = SplitKTmemGemm64(
        tactic=SplitKTactic(128, 64, 8, 14), k=K, owners=8, alone=False
    )
    assert not lone.separate


# --------------------------------------------------------------------------------------
# The MoE front kernels (8 owners: every K split sums a 16-row share) on 2 GPUs: their
# front box needs NVLink multicast memory (torchrun --nproc-per-node 2 <this file>).
VALID, SPLIT, SHARD, HIDDEN, BOX_ROWS = 56, 64, 224, 384, 512
FRONT_CASES = [
    (16, "tmem", 64), (32, "tmem", 64), (64, "tmem", 64), (128, "rows", 64),
    (256, "rows", 128),
]


def test_front_separate_mailboxes_torchrun():
    """The torchrun check below on 2 local GPUs (skipped without them)."""
    import subprocess
    import sys

    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs (multicast memory)")
    command = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node=2"]
    subprocess.run([*command, __file__], check=True, timeout=900)


def _front(kind, rows, n, separate, dev, front_first=True):
    """``front_first`` False: ``MoeFrontRows`` clusters in grid order (the base
    ``cluster_tile``) instead of the front tiles first."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import splitk_tmem64
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        front,
        front_rows,
        front_tmem,
    )

    cls = splitk_tmem64.SplitKTmemGemm64
    init = cls.__init__
    rows_cls = front_tmem.MoeFrontRows
    order = rows_cls.__dict__["cluster_tile"]

    def barrier_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.separate = False

    front.compile_front_tmem.cache_clear()
    front_rows.compile_front_rows.cache_clear()
    if not separate:
        cls.__init__ = barrier_init
    if not front_first:
        rows_cls.cluster_tile = cls.cluster_tile
    try:
        if kind == "tmem":
            code = front.compile_front_tmem(SPLIT, VALID, SHARD, HIDDEN, dev)
        else:
            code = front_rows.compile_front_rows(
                VALID, SPLIT, SHARD, HIDDEN, BOX_ROWS, dev, n
            )
    finally:
        cls.__init__ = init
        rows_cls.cluster_tile = order
        front.compile_front_tmem.cache_clear()
        front_rows.compile_front_rows.cache_clear()
    return code


def _front_main():
    import os

    import cutlass as c
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    from cutlass import Int32

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        dynamic_view,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (
        matrix_view,
        weight_map,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.front import WIDTH

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.cuda.current_device()
    box = symm_mem.empty((1, BOX_ROWS, WIDTH), dtype=torch.bfloat16, device="cuda")
    mc = int(symm_mem.rendezvous(box, dist.group.WORLD).multicast_ptr or 0)
    assert mc, "multicast memory"
    failures = 0
    for rows, kind, n in FRONT_CASES:
        gen = torch.Generator(device="cuda").manual_seed(1000 * rows + rank)
        packed = torch.randn(SPLIT + SHARD, K, generator=gen, device="cuda") * 0.02
        packed[VALID:SPLIT] = 0
        packed = packed.bfloat16()
        gate_up = (torch.randn(2 * HIDDEN, K, generator=gen, device="cuda") * 0.02)
        gate_up = gate_up.bfloat16()
        x = torch.randn(rows, K, generator=gen, device="cuda").bfloat16()
        maps = (weight_map(packed, 128), weight_map(gate_up, 64))
        ptrs = tuple(c.Int64(m.data_ptr()) for m in maps)
        outs = []
        # separate, barrier, separate again, clusters in grid order: all bitwise
        for separate, first in ((True, True), (False, True), (True, True), (True, False)):
            code = _front(kind, rows, n, separate, dev, first)
            h_rows = rows if kind == "rows" else 64  # MoeFrontTmem64: a 64-row h
            h = torch.zeros(h_rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
            box.view(torch.int32).fill_(-0x80000000)
            torch.cuda.synchronize()
            dist.barrier()
            hv = dynamic_view(h) if kind == "rows" else to_cute(h)
            args = (matrix_view(packed), matrix_view(x), hv, to_cute(box), c.Int64(mc))
            args += (Int32(rank * VALID), Int32(rank * SHARD), c.Float32(1.0))
            args += (c.Float32(1.0),)
            code(*args, *ptrs, stream(x.device))
            torch.cuda.synchronize()
            dist.barrier()
            outs.append((box[0, :rows].clone(), h[:rows]))
        ok = all(
            torch.equal(a[0].view(torch.int16), b[0].view(torch.int16))
            and torch.equal(a[1].view(torch.int16), b[1].view(torch.int16))
            for a, b in zip(outs[1:], outs[:1] * 3)
        )
        print(f"rank {rank}: {kind} rows {rows} n{n}: {'ok' if ok else 'FAIL'}",
              flush=True)  # fmt: skip
        failures += not ok
    dist.barrier()
    dist.destroy_process_group()
    return failures


if __name__ == "__main__":
    import sys

    sys.exit(1 if _front_main() else 0)
