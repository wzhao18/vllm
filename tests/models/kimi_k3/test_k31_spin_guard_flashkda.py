# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashKDA's mbarrier wait (k64tc::wait_wd, ops/flashkda_split/csrc/smxx/k64_tc.cu) under the
shared spin guard (DCRIT r33): it used to give up after 2^20 polls and RETURN, letting the
kernel continue with an incomplete stage; now a wait that never completes traps after the
guard and the host record (spin_record.py) names the kernel. The real k64_tc.cu translation
unit is compiled into the test extension, so the test exercises the production wait_wd."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="needs SM10x (mbarrier try_wait, the kernel's arch)",
)

_NV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "vllm", "models", "kimi_k3", "nvidia")
)
_SMXX = os.path.join(_NV, "ops", "flashkda_split", "csrc", "smxx")

_CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "k64_tc.cu"

__device__ __forceinline__ uint32_t k64t_smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// mode 0: the barrier completes (one arrive) -> out = 7; mode 1: it never does.
__global__ void flashkda_wait_wd_test(int* out, long long* dbg, int mode) {
  k31sg::begin_call(11);
  __shared__ __align__(8) uint64_t bar;
  if (threadIdx.x == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(k64t_smem_u32(&bar)));
  __syncthreads();
  if (threadIdx.x == 0 && mode == 0)
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(k64t_smem_u32(&bar)) : "memory");
  k64tc::wait_wd(k64t_smem_u32(&bar), 0u, 5, 0, dbg);
  if (threadIdx.x == 0) out[0] = 7;
}

void setup(int64_t rank, int64_t diag) { k31sg_set_rank_k64_tc((int)rank, diag); }
void run(torch::Tensor out, torch::Tensor dbg, int64_t mode) {
  flashkda_wait_wd_test<<<1, 64, 0, c10::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<int>(), reinterpret_cast<long long*>(dbg.data_ptr<int64_t>()), (int)mode);
}
"""

_CPP = "void setup(int64_t rank, int64_t diag); void run(torch::Tensor out, torch::Tensor dbg, int64_t mode);"


def _load(guard_ns: int):
    import flashinfer
    from torch.utils.cpp_extension import load_inline

    # the include paths and flags of the production build (ops/kda_segscan.load_extension)
    cutlass = os.path.join(os.path.dirname(flashinfer.__file__), "data", "cutlass", "include")
    return load_inline(
        name=f"k31_flashkda_wait_wd_test_{guard_ns}",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=["setup", "run"],
        extra_include_paths=[_NV, _SMXX, os.path.dirname(_SMXX), cutlass],
        extra_cuda_cflags=[
            "-O3", f"-DK31_SPIN_GUARD_NS={guard_ns}ull",
            "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-gencode", "arch=compute_100f,code=sm_100f",
        ],
        verbose=False,
    )


def test_flashkda_wait_wd_completed_wait_unchanged():
    m = _load(500_000_000)
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    dbg = torch.zeros(4096, dtype=torch.int64, device="cuda")
    m.run(out, dbg, 0)
    torch.cuda.synchronize()
    assert out.item() == 7
    assert int(dbg.abs().sum()) == 0  # the old watchdog's dbg report is gone for good waits


def test_flashkda_wait_wd_stuck_wait_traps_with_host_record():
    # a fresh process: the trap ends its CUDA context
    code = textwrap.dedent(
        f"""
        import sys, torch
        sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
        import test_k31_spin_guard_flashkda as t
        from vllm.models.kimi_k3.nvidia import spin_record
        rec = spin_record.get(6)
        m = t._load(500_000_000)
        m.setup(6, rec.address + spin_record.RECORD_WORDS * 4)  # record 1
        out = torch.zeros(1, dtype=torch.int32, device="cuda")
        dbg = torch.zeros(4096, dtype=torch.int64, device="cuda")
        m.run(out, dbg, 1)
        try:
            torch.cuda.synchronize()
            print("NO_TRAP", out.item())
        except Exception as e:
            print("TRAPPED", type(e).__name__, flush=True)
            print("RECORD", rec.check(), flush=True)
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    text = r.stdout + r.stderr
    assert "NO_TRAP" not in text, text[-2000:]  # the old wait_wd returned and the kernel went on
    assert "TRAPPED" in text, text[-2000:]
    # the host-mapped record survives the trap: kernel, site (= barrier id 5), rank, call, size
    assert "K31_SPIN_RECORD source=pidx kernel=flashkda_k64_tc site=5 rank=6 call=1 size=11 " in text, text[-2000:]
