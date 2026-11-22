# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the runtime spin-guard bounds (int64.3 item 1): write guard_config.VALUES into the module globals
that guard_global.read_global_u64 created in a compiled CuTe kernel (cuLibraryGetGlobal + cuMemcpyHtoD). Called by
cute_cache after every compile or cache load of a Kimi K3 kernel, OUTSIDE the trace; guard_config is imported here
(function scope) and nowhere on the trace path, so its values stay out of every compile-cache key."""

from __future__ import annotations


def patch_compiled(compiled) -> int:
    """Patch every guard global ``compiled`` defines; returns how many (0: no guard globals or no library)."""
    lib = getattr(compiled, "library", None)
    if lib is None:
        return 0
    import cuda.bindings.driver as cuda
    import torch

    from vllm.models.kimi_k3.nvidia import guard_config

    n = 0
    for name, value in guard_config.VALUES.items():
        err, dptr, size = cuda.cuLibraryGetGlobal(lib, name.encode())
        if err != cuda.CUresult.CUDA_SUCCESS:
            continue  # this kernel has no such guard
        if size != 8:
            raise RuntimeError(f"guard_patch: {name} is {size} bytes")
        host = torch.tensor([int(value)], dtype=torch.int64)
        (err,) = cuda.cuMemcpyHtoD(dptr, host.data_ptr(), 8)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"guard_patch: patching {name} failed: {err}")
        n += 1
    return n
