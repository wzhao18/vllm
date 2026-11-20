# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registers, static shared memory and CTAs per SM of the drafter window kernels
(GqaWindowUmma, GqaWindowStream) at TP4 q3, page 4096, one GPU (development check of
the two-CTAs-per-SM residency the stream relies on)."""

import cuda.bindings.driver as cuda
import torch

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    _checked,
    resolve_kernel,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.resident import (
    per_sm_fit,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
    gqa_window_stream as gs,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
    gqa_window_umma as gu,
)


def report(name, compiled, smem):
    _, function = resolve_kernel(compiled, 0)
    a = cuda.CUfunction_attribute
    (regs,) = _checked(cuda.cuFuncGetAttribute(a.CU_FUNC_ATTRIBUTE_NUM_REGS, function))
    (static,) = _checked(
        cuda.cuFuncGetAttribute(a.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function)
    )
    (local,) = _checked(
        cuda.cuFuncGetAttribute(a.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function)
    )
    carve = a.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
    _checked(cuda.cuFuncSetAttribute(function, carve, 100))
    occ = [
        _checked(cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(function, 160, s))[0]
        for s in smem
    ]
    fit = per_sm_fit(compiled, 0, 160, gs.smem_bytes(6, 3))
    print(f"{name}: regs {regs} static smem {static} local {local} fit {fit} | API at "
          + " ".join(f"{s}B:{o}" for s, o in zip(smem, occ)), flush=True)


def main():
    torch.cuda.set_device(0)
    smem = (100000, 113000, 115632, 116000)
    u = gu.compile_window_umma(6, 4, 3, 4096, 6, 2048, None, 1 << 18, 0)
    report("umma R6", u, smem)
    for pmax in (2, 5):
        s = gs.compile_window_stream(6, 4, 3, 4096, 6, 2048, None, 1 << 18, 0, pmax)
        report(f"stream R6 pmax {pmax}", s, smem)


if __name__ == "__main__":
    main()
