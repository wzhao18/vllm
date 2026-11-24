# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode, TP16 MoE front kernels (``MoeFrontTmem64`` at 16-64
rows, ``MoeFrontRows`` above 64) with the pre-wait weight stream on two TMA issuing
warps: the published router logits and latent-down columns and the shared expert's
h are bitwise those of the one-issuer kernels (only the weights' TMA issue order
changes), and the logits / latent are within FP32-accumulation noise of an FP64
reference."""

import pytest
import torch

pytest.importorskip("cutlass")

import cutlass as c  # noqa: E402
import torch.distributed as dist  # noqa: E402
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

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Rubin / Blackwell GPU required",
)

K, VALID, SPLIT, SHARD, HIDDEN = 7168, 56, 64, 224, 384  # TP16 front + shared shard
BOX_ROWS = 512
SENTINEL = -0x80000000


@pytest.fixture(scope="module")
def box():
    """A one-rank front mailbox [1, BOX_ROWS, WIDTH] and its multicast address."""
    import torch.distributed._symmetric_memory as symm_mem

    if not dist.is_initialized():
        dist.init_process_group(
            "nccl", init_method="tcp://127.0.0.1:29517", rank=0, world_size=1
        )
    buffer = symm_mem.empty((1, BOX_ROWS, WIDTH), dtype=torch.bfloat16, device="cuda")
    mc = int(symm_mem.rendezvous(buffer, dist.group.WORLD).multicast_ptr or 0)
    if not mc:
        pytest.skip("no NVLink multicast object on this device")
    return buffer, mc


def _weights(seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    packed = torch.randn(SPLIT + SHARD, K, generator=gen, device="cuda") * 0.02
    packed[VALID:SPLIT] = 0  # the router's pad rows
    gate_up = torch.randn(2 * HIDDEN, K, generator=gen, device="cuda") * 0.02
    return packed.bfloat16(), gate_up.bfloat16(), gen


def _run(code, rows_kernel, packed, gate_up, x, box, mc, hidden):
    buffer, h = box, torch.zeros(x.shape[0], max(hidden, 1), dtype=torch.bfloat16)
    h = h.cuda()[:, :hidden] if hidden else h.cuda()
    buffer.view(torch.int32).fill_(SENTINEL)
    maps = (weight_map(packed, 128), weight_map(gate_up, 64))
    ptrs = tuple(c.Int64(m.data_ptr()) for m in maps)
    hv = dynamic_view(h) if rows_kernel else to_cute(h)
    args = (matrix_view(packed), matrix_view(x), hv, to_cute(buffer), c.Int64(mc))
    args += (Int32(0), Int32(0), c.Float32(1.0), c.Float32(1.0), *ptrs)
    code(*args, stream(x.device))
    torch.accelerator.synchronize()
    m = x.shape[0]
    published = buffer[0, :m].clone()
    return published, h.clone()


# (rows, kernel): MoeFrontTmem64 (merged) at 16-64; MoeFrontRows n64 / n128 merged and
# n128 front only above 64 (the decode policy's shapes).
CASES = [
    (16, "tmem", 64, HIDDEN),
    (64, "tmem", 64, HIDDEN),
    (128, "rows", 64, HIDDEN),
    (256, "rows", 128, HIDDEN),
    (512, "rows", 128, 0),
]


@pytest.mark.parametrize("rows,kind,n,hidden", CASES)
def test_two_issuers_bitwise(box, rows, kind, n, hidden):
    buffer, mc = box
    packed, gate_up, gen = _weights(rows)
    x = torch.randn(rows, K, generator=gen, device="cuda").bfloat16()
    dev = x.device.index
    outs = []
    for issuers in (1, 2):
        if kind == "tmem":
            code = compile_front_tmem(SPLIT, VALID, SHARD, hidden, dev, issuers)
        else:
            code = compile_front_rows(
                VALID, SPLIT, SHARD, hidden, BOX_ROWS, dev, n, issuers
            )
        outs.append(_run(code, kind == "rows", packed, gate_up, x, buffer, mc, hidden))
    (box1, h1), (box2, h2) = outs
    assert torch.equal(box1.view(torch.int16), box2.view(torch.int16))
    if hidden:
        assert torch.equal(h1.view(torch.int16), h2.view(torch.int16))
    # FP64 reference: router logits (FP32 words at columns 3584 + 2 r) and latent.
    ref = x.double() @ packed.double().T
    logits = box2.view(torch.float32)[:, 1792 : 1792 + VALID].double()
    scale = ref.abs().max().item()
    assert (logits - ref[:, :VALID]).abs().max().item() / scale < 2e-5
    latent = box2[:, :SHARD].double()
    assert (latent - ref[:, SPLIT:]).abs().max().item() / scale < 1e-2
    untouched = box2.view(torch.int32)[:, SHARD // 2 : 1792]
    assert bool((untouched == SENTINEL).all())
