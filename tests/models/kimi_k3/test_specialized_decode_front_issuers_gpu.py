# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode, TP16 MoE front kernels (``MoeFrontTmem64`` at 16-64
rows, ``MoeFrontRows`` above 64) with the pre-wait weight stream on two TMA issuing
warps: the published router logits and latent-down columns and the shared expert's
h are bitwise those of the one-issuer kernels (only the weights' TMA issue order
changes), and the logits / latent are within FP32-accumulation noise of an FP64
reference.

    pytest tests/models/kimi_k3/test_specialized_decode_front_issuers_gpu.py
    (runs: torchrun --nproc-per-node 2 <this file>; the front box needs NVLink
    multicast memory, i.e. 2+ GPUs)"""

import sys

import pytest
import torch

pytest.importorskip("cutlass")

import cutlass as c  # noqa: E402
from cutlass import Int32  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (  # noqa: E402
    dynamic_view,
    stream,
    to_cute,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (  # noqa: E402
    matrix_view,
    weight_map,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.front import (  # noqa: E402
    WIDTH,
    compile_front_tmem,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.front_rows import (  # noqa: E402
    compile_front_rows,
)

K, VALID, SPLIT, SHARD, HIDDEN = 7168, 56, 64, 224, 384  # TP16 front + shared shard
BOX_ROWS = 512
SENTINEL = -0x80000000
# (rows, kernel, n, hidden): MoeFrontTmem64 (merged) at 16-64 rows; MoeFrontRows n64 /
# n128 merged and n128 front only above 64 (the decode policy's shapes).
CASES = [
    (16, "tmem", 64, HIDDEN),
    (32, "tmem", 64, HIDDEN),
    (64, "tmem", 64, HIDDEN),
    (128, "rows", 64, HIDDEN),
    (256, "rows", 128, HIDDEN),
    (512, "rows", 128, 0),
]


def _world():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 10:
        return 0
    return min(torch.cuda.device_count(), 2) if torch.cuda.device_count() >= 2 else 0


def test_front_two_issuers_torchrun():
    """The torchrun check below on 2 local GPUs (skipped without them)."""
    import subprocess

    world = _world()
    if not world:
        pytest.skip("needs 2 Rubin / Blackwell GPUs (multicast memory)")
    command = [sys.executable, "-m", "torch.distributed.run"]
    command += [f"--nproc-per-node={world}", __file__]
    subprocess.run(command, check=True, timeout=900)


def _run(code, rows_kernel, packed, gate_up, x, box, mc, rank, hidden):
    """One front launch into the rank's mailbox columns; returns (box rows, h)."""
    import torch.distributed as dist

    h = torch.zeros(x.shape[0], max(hidden, 1), dtype=torch.bfloat16, device=x.device)
    box.view(torch.int32).fill_(SENTINEL)
    torch.accelerator.synchronize()
    dist.barrier()
    maps = (weight_map(packed, 128), weight_map(gate_up, 64))
    ptrs = tuple(c.Int64(m.data_ptr()) for m in maps)
    hv = dynamic_view(h) if rows_kernel else to_cute(h)
    args = (matrix_view(packed), matrix_view(x), hv, to_cute(box), c.Int64(mc))
    args += (Int32(rank * VALID), Int32(rank * SHARD), c.Float32(1.0), c.Float32(1.0))
    code(*args, *ptrs, stream(x.device))
    torch.accelerator.synchronize()
    dist.barrier()
    return box[0, : x.shape[0]].clone(), h


def _case(rank, world, rows, kind, n, hidden, box, mc):
    import torch.distributed as dist

    gen = torch.Generator(device="cuda").manual_seed(1000 * rows + rank)
    packed = torch.randn(SPLIT + SHARD, K, generator=gen, device="cuda") * 0.02
    packed[VALID:SPLIT] = 0  # the router's pad rows
    packed = packed.bfloat16()
    gate_up = (torch.randn(2 * HIDDEN, K, generator=gen, device="cuda") * 0.02).bfloat16()
    x = torch.randn(rows, K, generator=torch.Generator(device="cuda").manual_seed(rows),
                    device="cuda").bfloat16()  # fmt: skip
    dev = x.device.index
    outs = []
    for issuers in (1, 2):
        if kind == "tmem":
            code = compile_front_tmem(SPLIT, VALID, SHARD, hidden, dev, issuers)
        else:
            code = compile_front_rows(
                VALID, SPLIT, SHARD, hidden, BOX_ROWS, dev, n, issuers
            )
        outs.append(_run(code, kind == "rows", packed, gate_up, x, box, mc, rank, hidden))
    (box1, h1), (box2, h2) = outs
    ok = torch.equal(box1.view(torch.int16), box2.view(torch.int16))
    if hidden:
        ok = ok and torch.equal(h1.view(torch.int16), h2.view(torch.int16))
    # Every rank's columns against its FP64 reference (logits: FP32 words at BF16
    # columns 3584 + 2 r; latent: BF16 columns [r * SHARD, + SHARD)).
    ref = (x.double() @ packed.double().T).float()
    refs = [torch.empty_like(ref) for _ in range(world)]
    dist.all_gather(refs, ref)
    words = box2.view(torch.float32)
    for r, part in enumerate(refs):
        part = part.double()
        scale = part.abs().max().item()
        logits = words[:, 1792 + r * VALID : 1792 + (r + 1) * VALID].double()
        ok = ok and (logits - part[:, :VALID]).abs().max().item() / scale < 2e-5
        latent = box2[:, r * SHARD : (r + 1) * SHARD].double()
        ok = ok and (latent - part[:, SPLIT:]).abs().max().item() / scale < 1e-2
    unused = box2.view(torch.int32)[:, world * SHARD // 2 : 1792]
    ok = ok and bool((unused == SENTINEL).all())
    print(f"rank {rank}: rows {rows} {kind} n{n} hidden {hidden}: "
          f"{'ok' if ok else 'FAIL'}", flush=True)  # fmt: skip
    return not ok


def _main():
    import os

    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    box = symm_mem.empty((1, BOX_ROWS, WIDTH), dtype=torch.bfloat16, device="cuda")
    mc = int(symm_mem.rendezvous(box, dist.group.WORLD).multicast_ptr or 0)
    assert mc, "multicast memory"
    failures = sum(_case(rank, world, *case, box, mc) for case in CASES)
    dist.barrier()
    dist.destroy_process_group()
    return failures


if __name__ == "__main__":
    sys.exit(1 if _main() else 0)
