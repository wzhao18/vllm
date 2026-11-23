# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime spin-guard bound (int64.3 item 1): a CuTe kernel reads a module global with its built-in default until
the host patches it (guard_patch.patch_compiled), then the patched value, on the next launch."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@cuda
def test_kernel_reads_the_host_patched_guard_bound():
    import cutlass.cute as cute
    from cutlass import Int64
    from cutlass.cute.runtime import from_dlpack

    from vllm.models.kimi_k3.nvidia import guard_config, guard_patch
    from vllm.models.kimi_k3.nvidia.guard_global import read_global_u64

    builtin = 111_000_000_000

    @cute.kernel
    def k(out: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        if tid == 0:
            out[0] = Int64(read_global_u64("k31_guard_ns", builtin))

    @cute.jit
    def launch(out: cute.Tensor):
        k(out).launch(grid=(1, 1, 1), block=(32, 1, 1))

    out = torch.zeros(1, dtype=torch.int64, device="cuda")
    t = from_dlpack(out)
    compiled = cute.compile(launch, t)
    compiled(t)
    torch.cuda.synchronize()
    assert int(out[0]) == builtin
    assert guard_patch.patch_compiled(compiled) == 1  # k31_guard_ns only (no MegaMoE global in this kernel)
    out.zero_()
    compiled(t)
    torch.cuda.synchronize()
    assert int(out[0]) == guard_config.GUARD_NS
