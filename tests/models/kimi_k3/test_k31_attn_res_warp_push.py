# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ClusterAttnRes(warp_push=True) (per-warp pushes and totals after the delta lands) is
bitwise the barrier-exchange form, at the plain and the final boundary."""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
    pytest.skip("needs a CUDA device with thread-block clusters", allow_module_level=True)

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import make_fake_stream  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.attn_res.kernel import (  # noqa: E402
    ClusterAttnRes,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (  # noqa: E402
    architecture,
    dynamic_fake,
    dynamic_view,
    fake,
    stream,
    to_cute,
)

H = 7168


def _compile(nb, final, warp_push, device):
    row, weight, eps = dynamic_fake(2), fake((H,), (1,)), cutlass.Float32(1e-6)
    kernel = ClusterAttnRes(
        nb, slots=0, final=final, stores_first=True, spread_push=True,
        warp_push=warp_push,
    )  # fmt: skip
    args = (row, row, dynamic_fake(3), weight, weight, weight, row, eps, eps)
    arch = architecture(device)
    return cute.compile(kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}")


@pytest.mark.parametrize("nb,final", [(0, False), (2, False), (7, False), (4, True)])
@pytest.mark.parametrize("rows", [1, 8, 64])
def test_warp_push_bitwise(nb, final, rows):
    device = torch.device("cuda", 0)
    gen = torch.Generator(device=device).manual_seed(rows * 31 + nb)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=device) * scale).bfloat16()

    weights = [1 + 0.1 * rnd(H), rnd(H, scale=H**-0.5), 1 + 0.1 * rnd(H)]
    views = [to_cute(w) for w in weights]
    prefix0, blocks, delta = rnd(rows, H), rnd(rows, 8, H), rnd(rows, H, scale=0.01)
    eps = cutlass.Float32(1e-6)
    results = []
    for warp_push in (False, True):
        code = _compile(nb, final, warp_push, device)
        prefix, out = prefix0.clone(), torch.empty_like(prefix0)
        for _ in range(3):  # chained: each call feeds the next one's prefix
            code(
                dynamic_view(delta), dynamic_view(prefix), dynamic_view(blocks),
                *views, dynamic_view(out), eps, eps, stream(device),
            )  # fmt: skip
        torch.cuda.synchronize()
        results.append((out.clone(), prefix.clone()))
    for a, b in zip(results[0], results[1]):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
