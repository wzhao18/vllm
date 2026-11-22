# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""k31_spin_guard.cuh (bounded spin waits of the Kimi K3.1 CUDA kernels): a wait that
completes is unchanged, a wait that never completes traps after the guard with one
K31_SPIN_TIMEOUT line naming the kernel and the wait site."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="needs SM10x (mbarrier try_wait, the kernels' arch)",
)

_HDR_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "vllm", "models", "kimi_k3", "nvidia"
)
_HDR_DIR = os.path.abspath(_HDR_DIR)

_CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "k31_spin_guard.cuh"
K31SG_DEFINE_SET_RANK(test)

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// mode 0: the barrier completes (one arrive) -> out = 7; mode 1: it never does.
__global__ void k31_spin_guard_test(int* out, int mode) {
  k31sg::begin_call(3);
  __shared__ __align__(8) uint64_t bar;
  if (threadIdx.x == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&bar)));
  __syncthreads();
  if (threadIdx.x == 0 && mode == 0)
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(&bar)) : "memory");
  k31sg::mbar_wait(smem_u32(&bar), 0u, "k31_spin_guard_test", __LINE__);
  if (threadIdx.x == 0) out[0] = 7;
}

// flag poll across CTAs: CTA 1 publishes, CTA 0 polls with k31sg::Spin
__global__ void k31_spin_guard_flag(int* flag, int* out) {
  if (blockIdx.x == 1) {
    if (threadIdx.x == 0) atomicExch(flag, 1);
    return;
  }
  if (threadIdx.x == 0) {
    k31sg::Spin s;
    int v;
    while ((v = atomicAdd(flag, 0)) == 0) s.check("k31_spin_guard_flag", __LINE__, v, 1);
    out[0] = v + 6;
  }
}

void setup(int64_t rank, int64_t diag) { k31sg_set_rank_test((int)rank, diag); }
void run(torch::Tensor out, int64_t mode) {
  k31_spin_guard_test<<<1, 64, 0, c10::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int>(),
                                                                      (int)mode);
}
void run_flag(torch::Tensor flag, torch::Tensor out) {
  k31_spin_guard_flag<<<2, 32, 0, c10::cuda::getCurrentCUDAStream()>>>(flag.data_ptr<int>(),
                                                                       out.data_ptr<int>());
}
"""

_CPP = (
    "void setup(int64_t rank, int64_t diag); void run(torch::Tensor out, int64_t mode); "
    "void run_flag(torch::Tensor flag, torch::Tensor out);"
)


def _load(guard_ns: int):
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name=f"k31_spin_guard_test_{guard_ns}",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=["setup", "run", "run_flag"],
        extra_include_paths=[_HDR_DIR],
        extra_cuda_cflags=["-O3", f"-DK31_SPIN_GUARD_NS={guard_ns}ull",
                           "-gencode", "arch=compute_100f,code=sm_100f"],
        verbose=False,
    )


def test_completed_waits_unchanged():
    m = _load(500_000_000)
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    m.run(out, 0)
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    out2 = torch.zeros(1, dtype=torch.int32, device="cuda")
    m.run_flag(flag, out2)
    torch.cuda.synchronize()
    assert out.item() == 7 and out2.item() == 7


def test_stuck_wait_traps_naming_kernel():
    # a fresh process: the trap ends its CUDA context
    code = textwrap.dedent(
        f"""
        import sys, torch
        sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
        import test_k31_spin_guard as t
        from vllm.models.kimi_k3.nvidia import spin_record
        rec = spin_record.get(5)
        m = t._load(500_000_000)
        m.setup(5, rec.address + spin_record.RECORD_WORDS * 4)  # record 1
        out = torch.zeros(1, dtype=torch.int32, device="cuda")
        m.run(out, 1)
        try:
            torch.cuda.synchronize()
            print("NO_TRAP")
        except Exception:
            print("RECORD", rec.check(), flush=True)
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=600)
    text = r.stdout + r.stderr
    assert "NO_TRAP" not in text, text[-2000:]
    # the host-mapped record survives the trap and names the kernel, the site and the
    # rank (read either by the watcher thread or by the explicit check)
    assert "K31_SPIN_RECORD source=pidx kernel=k31_spin_guard_test site=" in text, \
        text[-2000:]
    assert " rank=5 call=1 size=3 " in text, text[-2000:]
