# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device side of the counted fabric windows (``specialized_decode.fabric``):
PTX 9.3 ``fabric.*`` operations with counted completion (sm_100+, assembled by the
13.4 ptxas) for the decode NVLink exchanges.

A put copies ``size`` bytes (multiple of 16, 16 B aligned) from shared memory into
window ``[endpoint, offset)`` of a peer (unicast LE id ``base + rank``) or of every
rank (``put_multimem`` through the multicast LE) and adds ``size`` to the 8-byte
counter at window byte ``counter`` on the destination. Protocol of one exchange:

    sender (one thread)                         receiver
    stage payload in smem; source_fence()       ...
    put(...) [x n]; submit_expect(bar, total)   target = advance(expected, bytes)
    wait(bar, phase) before reusing the smem    wait_counter(own counter, target)
                                                (includes the fabric acquire fence)
                                                read the data with plain loads

- The mbarrier must be initialized with ``barrier_init`` (layout::v1, count 1); one
  arrive.expect_tx per batch of puts (``submit_expect``, bytes summed over the batch,
  a multimem put counts once). ``wait`` returns nonzero when the fabric reported an
  error for the batch.
- Grid exit orders completion of submitted fabric operations before the next task
  in the stream, but the smem source must stay valid until ``wait`` returns.
- Counters only grow; the receiver keeps its running target (``advance`` on a global
  int64 slot, e.g. ``FabricWindow.expected[counter, cta]``), so replays need no
  reset. A multimem put also lands on (and counts at) the sender's own replica.
- Readers of fabric-written data use plain loads after ``wait_counter`` (it issues
  ``fence.proxy.generic::fabric.alias.acquire.sys``, the CUDA guide's acquire).
"""

import cutlass.cute as cute
from cutlass import Int64, Uint32, Uint64

from vllm.models.kimi_k3.nvidia.k31_spin_guard import (
    register_poll_helper,
    spin_start,
    spin_tick,
)

from .primitives import asm

register_poll_helper(__file__)  # timeouts report the caller's wait site

_PUT = (
    "fabric.try_put.async{mm}.shared::cta.mbarrier::complete_tx::16B"
    ".mbarrier::report::fabric.counted::bytes.relaxed.sys.b128 [$0,$1,$2],[$3],$4,[$5];"
)


@asm()
def barrier_init(barrier):
    """layout::v1 mbarrier (count 1) for fabric completion; visible to the async
    proxy. One thread."""
    code = "mbarrier.init.layout::v1.shared::cta.b64 [$0],1; fence.proxy.async.shared::cta;"
    return code, "r,~{memory}", (barrier,)


@asm()
def source_fence():
    """Make this thread's smem stores visible to the fabric (async-proxy) source
    read; every writing thread, then a CTA barrier, before the put."""
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


@asm()
def put(barrier, endpoint, offset, counter, source, size):
    """Counted unicast put of ``size`` bytes at smem ``source`` to window byte
    ``offset`` of ``endpoint``, counted at window byte ``counter`` there."""
    code = _PUT.format(mm="")
    operands = (Uint32(endpoint), Int64(offset), Int64(counter), source, Uint32(size), barrier)
    return code, "r,l,l,r,r,r,~{memory}", operands


@asm()
def put_multimem(barrier, endpoint, offset, counter, source, size):
    """Counted multimem put through multicast ``endpoint``: one copy to every
    member's replica (the sender's included), each counting at its ``counter``."""
    code = _PUT.format(mm=".multimem")
    operands = (Uint32(endpoint), Int64(offset), Int64(counter), source, Uint32(size), barrier)
    return code, "r,l,l,r,r,r,~{memory}", operands


_RED_TYPES = ("u32", "u64", "f16", "bf16", "f32", "f64")


@asm()
def red_add(barrier, endpoint, offset, counter, source, size, dtype="u32"):
    """Counted ``fabric.try_red ... add.<dtype>``: element-wise add of ``size`` bytes
    at smem ``source`` into window byte ``offset`` of ``endpoint`` (order across
    senders unspecified: exact for integers), counted like ``put``."""
    assert dtype in _RED_TYPES, dtype
    code = (
        "fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B"
        ".mbarrier::report::fabric.counted::bytes.relaxed.sys.add."
        f"{dtype} [$0,$1,$2],[$3],$4,[$5];"
    )
    operands = (Uint32(endpoint), Int64(offset), Int64(counter), source, Uint32(size), barrier)
    return code, "r,l,l,r,r,r,~{memory}", operands


@asm()
def submit_expect(barrier, size):
    """Submit this thread's queued puts and arrive on ``barrier`` expecting ``size``
    bytes of source reads (multiple of 16; a multimem put counts once)."""
    code = (
        "{ .reg .b32 t; shr.u32 t,$1,4; fabric.submit; "
        "mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _,[$0],t; }"
    )
    return code, "r,r,~{memory}", (barrier, Uint32(size))


@asm(Uint32)
def _try_wait(barrier, phase):
    code = (
        "{ .reg .pred p, q; mbarrier.try_wait.parity.phase_type::primary.relaxed.cta"
        ".shared::cta.b64 p|q,[$1],$2; selp.u32 $0,1,0,p; @q or.b32 $0,$0,2; }"
    )
    return code, "=r,r,r,~{memory}", (barrier, Uint32(phase))


@cute.jit
def wait(barrier, phase) -> Uint32:
    """Wait for ``phase`` of ``barrier`` (the batch's source reads and completion);
    1 when the fabric reported an error for it, else 0."""
    state = _try_wait(barrier, phase)
    while state & Uint32(1) == Uint32(0):
        state = _try_wait(barrier, phase)
    return state >> Uint32(1)


@asm(Uint64)
def load_counter(address):
    """System-scope acquire load of the counter at global ``address`` (Int64)."""
    return "ld.acquire.sys.global.u64 $0,[$1];", "=l,l,~{memory}", (Int64(address),)


@asm()
def fabric_acquire():
    """Order later generic loads after the fabric writes the counter observed."""
    return "fence.proxy.generic::fabric.alias.acquire.sys;", "~{memory}", ()


@cute.jit
def wait_counter(address, target):
    """Spin until the counter at ``address`` reaches ``target`` (Uint64), then the
    fabric acquire fence: the counted bytes are readable with plain loads."""
    value = load_counter(address)
    n, t0 = spin_start()
    while value < Uint64(target):
        value = load_counter(address)
        n, t0 = spin_tick(n, t0, observed=value, target=target)
    fabric_acquire()


@asm(Uint64)
def advance(slot, size):
    """``slot`` (Int64 global address of a running target) += ``size``; returns the
    new target. Only the thread that waits calls it."""
    code = (
        "{ .reg .u64 v; ld.global.u64 v,[$1]; add.u64 $0,v,$2; st.global.u64 [$1],$0; }"
    )
    return code, "=l,l,l,~{memory}", (Int64(slot), Uint64(size))
