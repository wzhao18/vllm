# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode, TP16 MoE front kernels (``MoeFrontTmem64`` at 16-64
rows, ``MoeFrontRows`` at any rows) on ``SplitKTmemGemm64``'s reduce-scatter, which
sends only the token quads that hold real tokens of an N tile across DSMEM: for M
real tokens, the published router logits and latent-down columns and the shared
expert's h are bitwise those of the same kernel run on the tokens zero-padded to
whole N tiles (every quad sent), the box rows past M stay untouched, and the
logits / latent are within FP32-accumulation noise of an FP64 reference.

    pytest tests/models/kimi_k3/test_specialized_decode_front_live_tokens_gpu.py
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
# (real tokens, kernel, n, hidden, padded tokens): the decode shapes (16 / 32 rows on
# MoeFrontTmem64) and MoeFrontRows tiles with 20 (a partial quad), 8 and 72 live tokens.
CASES = [
    (16, "tmem", 64, HIDDEN, 64),
    (32, "tmem", 64, HIDDEN, 64),
    (20, "rows", 64, HIDDEN, 64),
    (72, "rows", 64, HIDDEN, 128),
    (136, "rows", 128, HIDDEN, 256),
    (200, "rows", 128, 0, 256),
]


def _world():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 10:
        return 0
    return 2 if torch.cuda.device_count() >= 2 else 0


def test_front_live_tokens_torchrun():
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
    return box[0].clone(), h


def _case(rank, world, rows, kind, n, hidden, padded, box, mc):
    import torch.distributed as dist

    gen = torch.Generator(device="cuda").manual_seed(1000 * rows + rank)
    packed = torch.randn(SPLIT + SHARD, K, generator=gen, device="cuda") * 0.02
    packed[VALID:SPLIT] = 0  # the router's pad rows
    packed = packed.bfloat16()
    gate_up = (torch.randn(2 * HIDDEN, K, generator=gen, device="cuda") * 0.02).bfloat16()
    x = torch.randn(rows, K, generator=torch.Generator(device="cuda").manual_seed(rows),
                    device="cuda").bfloat16()  # fmt: skip
    x_pad = torch.zeros(padded, K, dtype=torch.bfloat16, device="cuda")
    x_pad[:rows] = x
    dev = x.device.index
    if kind == "tmem":
        code = compile_front_tmem(SPLIT, VALID, SHARD, hidden, dev)
    else:
        code = compile_front_rows(VALID, SPLIT, SHARD, hidden, BOX_ROWS, dev, n)
    live, h_live = _run(code, kind == "rows", packed, gate_up, x, box, mc, rank, hidden)
    full, h_full = _run(code, kind == "rows", packed, gate_up, x_pad, box, mc, rank,
                        hidden)  # fmt: skip
    ok = torch.equal(live[:rows].view(torch.int16), full[:rows].view(torch.int16))
    if hidden:
        ok = ok and torch.equal(h_live.view(torch.int16), h_full[:rows].view(torch.int16))
    ok = ok and bool((live[rows:].view(torch.int32) == SENTINEL).all())
    # Every rank's columns against its FP64 reference (logits: FP32 words at BF16
    # columns 3584 + 2 r; latent: BF16 columns [r * SHARD, + SHARD)).
    ref = (x.double() @ packed.double().T).float()
    refs = [torch.empty_like(ref) for _ in range(world)]
    dist.all_gather(refs, ref)
    words = live[:rows].view(torch.float32)
    for r, part in enumerate(refs):
        part = part.double()
        scale = part.abs().max().item()
        logits = words[:, 1792 + r * VALID : 1792 + (r + 1) * VALID].double()
        ok = ok and (logits - part[:, :VALID]).abs().max().item() / scale < 2e-5
        latent = live[:rows, r * SHARD : (r + 1) * SHARD].double()
        ok = ok and (latent - part[:, SPLIT:]).abs().max().item() / scale < 1e-2
    print(f"rank {rank}: rows {rows} (padded {padded}) {kind} n{n} hidden {hidden}: "
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
