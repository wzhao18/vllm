# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode: the fused SP boundary kernels
(``cutedsl.attn_res.sp_fused``) build co-resident at the deployed D3-32 shape of
the full 93-layer model (TP4, rows_cap 512 -> n_max 128 shard rows per rank, up to 8
saved AttnRes blocks), where every CTA of the grid polls rows that other ranks' CTAs
store in the same kernel.

Regression: without a launch bound the 8-block reduce compiled to 104 registers = 4
CTAs per SM, 848 < 896 CTAs on 212 SMs, and model init raised Unsupported on every
rank (the 24-layer view, at most 2 saved blocks, fit). The numerics of the bounded
codes are covered by tests/distributed/test_kimi_k3_sdec_sp_fused.py at n_max 128.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10,
    reason="Blackwell / Rubin specialized decode kernels",
)

TP, CAP = 4, 512
N_MAX = CAP // TP
NB = 8  # 93 layers, AttnRes block size 12: prev_valid_blocks up to 8


def _spill(code, device):
    import cuda.bindings.driver as cuda

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        _checked,
        resolve_kernel,
    )

    # compile_reduce returns the compiled kernel wrapped with its launch attributes
    # (prepare_shared_carveout): resolve the kernel underneath.
    _, function = resolve_kernel(getattr(code, "compiled", code), device)
    local = cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
    return _checked(cuda.cuFuncGetAttribute(local, function))[0]


# (kind, has_delta, push / write, mx): the 8-block variants of the deployed layers
# (93 layers: layer 92 is a plain layer after 8 blocks; layer 84 writes block 7, so
# its reduce sees 8).
CASES = [
    ("gather", True, -1, False),
    ("gather", True, -1, True),
    ("reduce", True, True, False),
    ("reduce", False, True, False),
    ("reduce", True, False, True),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(map(str, c)))
def test_sp_fused_co_resident_93_layers(case):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.attn_res import (
        sp_fused,
    )

    kind, has_delta, flag, mx = case
    device = torch.cuda.current_device()
    if kind == "gather":
        code = sp_fused.compile_gather(
            NB, has_delta, flag, TP, CAP, N_MAX, device, mx
        )
        per_sm, sms, registers, _ = sp_fused.RESIDENCY["gather", NB, mx]
    else:
        code = sp_fused.compile_reduce(
            NB, has_delta, flag, TP, CAP, N_MAX, device, mx, mx
        )
        per_sm, sms, registers, _ = sp_fused.RESIDENCY["reduce", NB, mx, mx]
    assert sp_fused.CLUSTER * N_MAX <= per_sm * sms
    assert registers * sp_fused.THREADS * per_sm <= 64 * 1024
    print(f"{case}: {registers} registers, {per_sm} CTAs/SM x {sms}, "
          f"{_spill(code, device)} B local")  # fmt: skip
