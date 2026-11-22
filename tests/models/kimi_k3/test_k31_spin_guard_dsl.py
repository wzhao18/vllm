# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""k31_spin_guard.py (bounded spin waits of the Kimi K3.1 CuTe DSL kernels) on a Lamport
poll (per thread, and warp-vote): a poll that completes returns the same words, a poll
that never completes traps after the guard with a K31_SPIN_TIMEOUT line naming the
poll loop's source line in this file."""

import os
import re
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


def _compile(warp: bool = False):
    """A Lamport poll of one 16-byte fragment (thread 0, or a whole warp with a vote
    exit) bounded by spin_tick, as the guarded kernels' polls are."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from vllm.models.kimi_k3.nvidia.k31_spin_guard import spin_start, spin_tick
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
        load_words,
        published,
    )

    class Poll:
        @cute.jit
        def __call__(self, box: cute.Tensor, out: cute.Tensor):
            self.kernel(box, out).launch(grid=[1, 1, 1], block=[32, 1, 1])

        @cute.kernel
        def kernel(self, box: cute.Tensor, out: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            held = cute.make_rmem_tensor(4, cutlass.Uint32)
            got = load_words(box.iterator, count=4, volatile=True)
            for k in cutlass.range_constexpr(4):
                held[k] = got[k]
            n, t0 = spin_start()
            if cutlass.const_expr(warp):
                while not cute.arch.vote_all_sync(published(held[k] for k in range(4))):
                    got = load_words(box.iterator, count=4, volatile=True)
                    for k in cutlass.range_constexpr(4):
                        held[k] = got[k]
                    n, t0 = spin_tick(n, t0)  # WARP_SITE
            else:
                while not published(held[k] for k in range(4)):
                    got = load_words(box.iterator, count=4, volatile=True)
                    for k in cutlass.range_constexpr(4):
                        held[k] = got[k]
                    n, t0 = spin_tick(n, t0)  # POLL_SITE
            if tidx == 0:
                for k in cutlass.range_constexpr(4):
                    out[k] = cutlass.Int32(held[k])

    def run(box: torch.Tensor, out: torch.Tensor):
        b = from_dlpack(box.view(torch.bfloat16), assumed_align=16)
        o = from_dlpack(out, assumed_align=16)
        cute.compile(Poll(), b, o)(b, o)

    return run


def _line(tag: str) -> int:
    with open(__file__) as f:
        return next(i for i, l in enumerate(f, 1) if tag in l and "(" in l.split("#")[0])


def _poll_line() -> int:
    return _line("# POLL_SITE")


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


def test_stuck_poll_warp_traps_naming_site():
    text = _stuck(
        """
        run = t._compile(warp=True)
        box = torch.full((4,), t.SENTINEL, dtype=torch.int32, device="cuda")
        run(box, torch.zeros(4, dtype=torch.int32, device="cuda"))
        """
    )
    assert "NO_TRAP" not in text, text[-3000:]
    site = f"site=kimi_k3/test_k31_spin_guard_dsl.py:{_line('# WARP_SITE')}"
    assert re.search(re.escape(site) + r"(?!\d)", text), text[-3000:]


def test_completed_poll_warp_unchanged():
    run = _compile(warp=True)
    box = torch.tensor([5, 6, 7, 8], dtype=torch.int32, device="cuda")
    out = torch.zeros(4, dtype=torch.int32, device="cuda")
    run(box, out)
    torch.cuda.synchronize()
    assert out.tolist() == [5, 6, 7, 8]


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
    site = f"site=kimi_k3/test_k31_spin_guard_dsl.py:{_poll_line()}"
    assert "K31_SPIN_TIMEOUT kernel=kimi_k3/test_k31_spin_guard_dsl " in text, \
        text[-3000:]
    assert re.search(re.escape(site) + r"(?!\d)", text), text[-3000:]
