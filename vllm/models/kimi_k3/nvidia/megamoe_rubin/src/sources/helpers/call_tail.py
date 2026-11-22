"""vLLM-local (BUG-E2E-3, VENDOR.md S8): previous-call guard for the token-comm router.

The router of call k+1 adds its sizes / ready counts into every peer's tail-reset
state right after entry. Nothing ordered those remote writes after a peer's call-k
tail had reset that state, so a router running ahead (PDL chains let the stream move
past a still-running tail) could have its add erased and the peer then waited forever.
Same protocol as src_gp's ``GenphaseTail`` (one done signal per rank and call into a
sense-reversing two-slot counter keyed by a per-call epoch): ``reset_tail`` ends with
``signal_done`` after its barrier, and the next call's router waits for every rank's
done signal of the previous call before its first remote write. Ordering only.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64

from .device_workspace import DeviceWorkspace
from .ptx_helpers import red_add_relaxed_sys_s32
from .spin_guard import globaltimer_ns, spin_guard


class CallTail:
    def __init__(self, *, world_size: int, epoch_region: str, signal_region: str) -> None:
        self.world_size = world_size
        self.epoch_region = epoch_region
        self.signal_region = signal_region

    def register_device_workspace(self, workspace: DeviceWorkspace) -> None:
        workspace.register(
            self.epoch_region, cutlass.Int32, (1,), buffer_space="local", byte_alignment=16,
            reset="zero_on_first_allocate",
        )
        workspace.register(
            self.signal_region, cutlass.Int32, (2,), buffer_space="shared", byte_alignment=16,
            reset="zero_on_first_allocate",
        )

    @cute.jit
    def read_epoch(self, workspace: DeviceWorkspace) -> Int32:
        return cute.arch.load(workspace.ptr(self.epoch_region), Int32, sem="relaxed", scope="gpu")

    @cute.jit
    def signal_done(self, workspace: DeviceWorkspace, peer_rank_ptr_mapper, epoch: Int32) -> None:
        """Single thread: this rank has reset its tail state for call ``epoch``."""
        status = epoch & Int32(3)
        signal_phase = status & Int32(1)
        signal_delta = Int32(1)
        if (status >> Int32(1)) != Int32(0):
            signal_delta = Int32(-1)
        cute.arch.fence_acq_rel_gpu()
        cute.arch.store(workspace.ptr(self.epoch_region), epoch + Int32(1), sem="relaxed", scope="gpu")
        cute.arch.fence_acq_rel_sys()
        signal_address = workspace.ptr(self.signal_region).toint()
        for destination_rank in cutlass.range_constexpr(self.world_size):
            red_add_relaxed_sys_s32(
                peer_rank_ptr_mapper.map(signal_address, Int32(destination_rank), Int64(signal_phase * Int32(4))),
                signal_delta,
            )

    @cute.jit
    def wait_previous_done(self, workspace: DeviceWorkspace) -> None:
        """Single thread: spin until every rank has signaled the previous call (epoch 0:
        the zero-initialized slot already meets the target)."""
        previous = self.read_epoch(workspace) - Int32(1)
        status = previous & Int32(3)
        signal_phase = status & Int32(1)
        signal_target = Int32(self.world_size)
        if (status >> Int32(1)) != Int32(0):
            signal_target = Int32(0)
        local_signal = workspace.ptr(self.signal_region) + signal_phase
        spin_start_ns = globaltimer_ns()
        while cute.arch.load(local_signal, Int32, sem="acquire", scope="sys") != signal_target:
            spin_guard(spin_start_ns, 60, local_signal, signal_target)
