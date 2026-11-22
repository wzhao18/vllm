# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""k31_spin_guard.py (bounded spin waits of the Kimi K3.1 CuTe DSL kernels), through the
decode mailbox poll: a poll that completes returns the same words, a poll that never
completes traps after the guard with a K31_SPIN_TIMEOUT line whose site is the poll's
CALLER (this file), not the mailbox helper."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="needs SM10x (the decode kernels' arch)",
)

SENTINEL = -(2**31)  # 0x80000000, the mailbox's unpublished word


def _compile():
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.mailbox import (
        poll,
    )

    class Poll:
        @cute.jit
        def __call__(self, box: cute.Tensor, out: cute.Tensor):
            self.kernel(box, out).launch(grid=[1, 1, 1], block=[32, 1, 1])

        @cute.kernel
        def kernel(self, box: cute.Tensor, out: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            if tidx == 0:
                got = poll(box.iterator, 8, 1)  # POLL_SITE
                for k in cutlass.range_constexpr(4):
                    out[k] = cutlass.Int32(got[0, k])

    def run(box: torch.Tensor, out: torch.Tensor):
        b = from_dlpack(box.view(torch.bfloat16), assumed_align=16)
        o = from_dlpack(out, assumed_align=16)
        cute.compile(Poll(), b, o)(b, o)

    return run


def _compile_mbar():
    """primitives.wait on a CTA mbarrier that completes (arrive) or never does."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
        wait,
    )

    class Wait:
        def __init__(self, arrive):
            self.arrive = arrive

        @cute.jit
        def __call__(self, out: cute.Tensor):
            self.kernel(out).launch(grid=[1, 1, 1], block=[32, 1, 1])

        @cute.kernel
        def kernel(self, out: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            smem = cutlass.utils.SmemAllocator()
            bar = smem.allocate_array(cutlass.Int64, 1)
            if tidx == 0:
                cute.arch.mbarrier_init(bar, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            if cutlass.const_expr(self.arrive):
                if tidx == 0:
                    cute.arch.mbarrier_arrive(bar)
            wait(bar, 0, scope="cta")  # WAIT_SITE
            if tidx == 0:
                out[0] = cutlass.Int32(7)

    def run(out: torch.Tensor, arrive: bool):
        o = from_dlpack(out, assumed_align=16)
        cute.compile(Wait(arrive), o)(o)

    return run


def _line(tag: str) -> int:
    with open(__file__) as f:
        return next(i for i, l in enumerate(f, 1) if tag in l and "(" in l.split("#")[0])


def _poll_line() -> int:
    return _line("# POLL_SITE")


def test_completed_mbar_wait_unchanged():
    run = _compile_mbar()
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    run(out, True)
    torch.cuda.synchronize()
    assert out.item() == 7


def test_completed_poll_unchanged():
    run = _compile()
    box = torch.tensor([5, 6, 7, 8], dtype=torch.int32, device="cuda")
    out = torch.zeros(4, dtype=torch.int32, device="cuda")
    run(box, out)
    torch.cuda.synchronize()
    assert out.tolist() == [5, 6, 7, 8]


def _stuck(body: str) -> str:
    """Run ``body`` (after a 0.5 s guard setup) in a fresh process (the trap ends its
    CUDA context); returns its output."""
    code = textwrap.dedent(
        f"""
        import sys, torch
        sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
        from vllm.models.kimi_k3.nvidia import k31_spin_guard as g
        g.SPIN_GUARD_NS = 500_000_000
        g.SPIN_FLUSH_NS = 200_000_000
        import test_k31_spin_guard_dsl as t
        """
    ) + textwrap.dedent(body) + textwrap.dedent(
        """
        try:
            torch.cuda.synchronize()
            print("NO_TRAP")
        except Exception as e:
            print("TRAPPED", type(e).__name__, flush=True)
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=900)
    return r.stdout + r.stderr


def test_stuck_mbar_wait_traps_naming_site():
    text = _stuck(
        """
        run = t._compile_mbar()
        run(torch.zeros(1, dtype=torch.int32, device="cuda"), False)
        """
    )
    assert "NO_TRAP" not in text, text[-3000:]
    site = f"site=kimi_k3/test_k31_spin_guard_dsl.py:{_line('# WAIT_SITE')} "
    assert site in text, text[-3000:]


def test_stuck_poll_traps_naming_site():
    # a fresh process (the trap ends its CUDA context) with a 0.5 s guard
    code = textwrap.dedent(
        f"""
        import sys, torch
        sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
        from vllm.models.kimi_k3.nvidia import k31_spin_guard as g
        g.SPIN_GUARD_NS = 500_000_000
        g.SPIN_FLUSH_NS = 200_000_000
        import test_k31_spin_guard_dsl as t
        run = t._compile()
        box = torch.full((4,), t.SENTINEL, dtype=torch.int32, device="cuda")
        out = torch.zeros(4, dtype=torch.int32, device="cuda")
        run(box, out)
        try:
            torch.cuda.synchronize()
            print("NO_TRAP")
        except Exception as e:
            print("TRAPPED", type(e).__name__, flush=True)
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=900)
    text = r.stdout + r.stderr
    assert "NO_TRAP" not in text, text[-3000:]
    site = f"site=kimi_k3/test_k31_spin_guard_dsl.py:{_poll_line()} "
    assert "K31_SPIN_TIMEOUT kernel=kimi_k3/test_k31_spin_guard_dsl " in text, text[-3000:]
    assert site in text, text[-3000:]
