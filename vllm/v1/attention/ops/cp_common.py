# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared symmetric-memory infrastructure for context-parallel attention."""

from __future__ import annotations

import atexit
import functools
import os
import sys
import threading
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)

try:
    import torch.distributed._symmetric_memory as symm_mem

    symm_mem_available = True
except ImportError:
    symm_mem = None  # type: ignore[assignment]
    symm_mem_available = False


@functools.cache
def _symm_mem_spans_group(group: GroupCoordinator) -> bool:
    """Probe whether the group has NVLS symmetric memory."""
    if not symm_mem_available:
        return False
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        device = torch.device("cuda", torch.accelerator.current_device_index())
        if not _SymmetricMemory.has_multicast_support(DeviceType.CUDA, device.index):
            return False
        probe = symm_mem.empty(8, dtype=torch.uint8, device=device)
        probe.zero_()
        torch.accelerator.synchronize()
        handle = symm_mem.rendezvous(probe, group.device_group.group_name)
        spans = handle is not None and handle.multicast_ptr != 0
    except Exception as error:
        logger.debug("Direct CP symmetric-memory probe failed: %s", error)
        return False
    logger.debug_once(
        "Direct CP symmetric memory across %d ranks: %s",
        group.world_size,
        "available" if spans else "unavailable",
    )
    return spans


def direct_cp_enabled(
    group: GroupCoordinator,
    dtype: torch.dtype,
    use_direct: bool | None,
    supported_dtypes: tuple[torch.dtype, ...] | None = None,
) -> bool:
    if use_direct is not None:
        return use_direct
    return (
        symm_mem_available
        and current_platform.is_cuda()
        and (supported_dtypes is None or dtype in supported_dtypes)
        and (
            all(in_the_same_node_as(group.cpu_group, source_rank=0))
            or _symm_mem_spans_group(group)
        )
    )


def direct_cp_multicast_enabled(
    group: GroupCoordinator,
    dtype: torch.dtype,
    use_direct: bool | None,
    supported_dtypes: tuple[torch.dtype, ...] | None = None,
) -> bool:
    return direct_cp_enabled(
        group, dtype, use_direct, supported_dtypes
    ) and _symm_mem_spans_group(group)


# Startup arrival barrier.
#
# The direct CP collectives wait for their peers on the device with a bounded
# spin, and nothing on the host orders the ranks before a launch. During startup
# one rank can lag its peers by minutes (a JIT build or a cubin load under a
# cross-node file lock at a kernel's first use), long enough to exhaust the
# peers' device wait. So the first direct collective of each workspace after
# ``request_arrival_barrier()`` (start-up, then once per CUDA graph capture size)
# first waits for every rank of its group on a host barrier. In steady state the
# generation never changes: one integer compare per eager call and nothing in a
# graph replay.
_arrival_generation = 1


def request_arrival_barrier() -> None:
    """Make the next direct CP collective of every workspace start with a host
    barrier on its group. Call it collectively (every rank, same point), e.g.
    before warming up each CUDA graph capture size."""
    global _arrival_generation
    _arrival_generation += 1


def _cpu_group_of(device_group: ProcessGroup) -> ProcessGroup | None:
    """The CPU (gloo) group of the vLLM GroupCoordinator owning ``device_group``."""
    from vllm.distributed import parallel_state

    for ref in list(parallel_state._groups.values()):
        coordinator = ref()
        if coordinator is not None and coordinator.device_group is device_group:
            return coordinator.cpu_group
    return None


# Device fault watch.
#
# A direct CP collective whose device wait expires prints "direct DCP ...
# timeout source=<rank> ..." and traps. The trap faults this rank's CUDA context,
# but the host only learns of it at its next CUDA call, and a host already
# blocked in a CPU collective (graph-capture bookkeeping) never makes one, while
# the peers spin forever in collectives that wait for the dead rank: the engine
# hangs instead of failing (DCRIT r148). A watcher thread polls the context
# state with a stream query and aborts the worker on a sticky error, which the
# executor turns into an engine failure.
#
# The query must never touch a stream that is being captured: a stream query on
# a capturing stream fails with CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED and
# invalidates the capture. The watcher therefore owns a stream created with the
# driver API (a torch.cuda.Stream() comes from torch's 32-stream round-robin
# pool and is shared with whoever draws the same slot, e.g. torch.cuda.graph's
# default capture stream or vLLM's current_stream(); 'vllm bench forward' drew
# the same slot and every rank aborted at its first capture). Its thread runs
# in relaxed capture mode, so its calls are never "unsafe" against a concurrent
# global-mode capture on another thread, and it checks the stream's capture
# status before querying it.
DEVICE_FAULT_EXIT_CODE = 87

# Results that do not mean the context faulted. The capture family is per-call
# (an operation met a stream capture) and never sticky: a real fault keeps being
# returned by every later call, so it is still caught at the next poll.
_HEALTHY_RESULTS = frozenset({"CUDA_SUCCESS", "CUDA_ERROR_NOT_READY"})
_CAPTURE_RESULTS = frozenset(
    {
        "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED",
        "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED",
        "CUDA_ERROR_STREAM_CAPTURE_MERGE",
        "CUDA_ERROR_STREAM_CAPTURE_UNMATCHED",
        "CUDA_ERROR_STREAM_CAPTURE_UNJOINED",
        "CUDA_ERROR_STREAM_CAPTURE_ISOLATION",
        "CUDA_ERROR_STREAM_CAPTURE_IMPLICIT",
        "CUDA_ERROR_CAPTURED_EVENT",
        "CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD",
    }
)


def _classify_cuda_result(error: Any) -> str:
    """'healthy', 'capture' (a capture was in the way; not a fault) or 'fault'."""
    name = error.name
    if name in _HEALTHY_RESULTS:
        return "healthy"
    if name in _CAPTURE_RESULTS:
        return "capture"
    return "fault"


class _DeviceFaultWatch:
    POLL_S = 1.0

    def __init__(self, device_index: int) -> None:
        # Driver API only: the thin calls below never initialize a CUDA
        # runtime, so the watcher thread cannot stall the main thread (a
        # cuda-python runtime init on the thread's first call can hold the GIL
        # for seconds, enough start-up skew to trip the peers' device waits).
        from cuda.bindings import driver

        self.driver = driver
        self.device_index = device_index
        self.stream: Any = None
        self.polls = 0
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._setup_error: str | None = None
        self._thread = threading.Thread(
            target=self._watch, name="direct-cp-fault-watch", daemon=True
        )
        self._thread.start()
        # All driver calls run on the watcher thread (after it switched to
        # relaxed capture mode), so creating the watch is safe even while this
        # thread is capturing.
        self._ready.wait()
        if self._setup_error is not None:
            raise RuntimeError(
                f"Device fault watch for device {device_index} could not start: "
                f"{self._setup_error}"
            )
        atexit.register(self.close)

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=5.0)

    def _setup(self) -> None:
        driver = self.driver

        def call(name: str, *args: Any) -> Any:
            error, *values = getattr(driver, name)(*args)
            if error != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"{name}: {error.name}")
            return values[0] if values else None

        # Relaxed first: no call of this thread is then prohibited by (or can
        # invalidate) a global-mode capture running on another thread.
        call(
            "cuThreadExchangeStreamCaptureMode",
            driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED,
        )
        call("cuInit", 0)
        device = call("cuDeviceGet", self.device_index)
        context = call("cuDevicePrimaryCtxRetain", device)
        self._device = device
        call("cuCtxSetCurrent", context)
        self.stream = call(
            "cuStreamCreate", driver.CUstream_flags.CU_STREAM_NON_BLOCKING
        )

    def query(self) -> Any:
        """CUDA_SUCCESS while the context is healthy, else its sticky error."""
        driver = self.driver
        error, status = driver.cuStreamIsCapturing(self.stream)
        if error != driver.CUresult.CUDA_SUCCESS:
            return error
        if status != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
            # Unreachable for a private stream; never query a capturing stream
            # (the query would invalidate the capture).
            return driver.CUresult.CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED
        (error,) = driver.cuStreamQuery(self.stream)
        return error

    def check(self) -> bool:
        self.polls += 1
        error = self.query()
        kind = _classify_cuda_result(error)
        if kind == "healthy":
            return False
        if kind == "capture":
            logger.warning_once(
                "Device fault watch (device %d): stream query met a stream "
                "capture (%s); not a fault, polling continues.",
                self.device_index,
                error.name,
            )
            return False
        self._abort(
            f"CUDA context of this worker (device {self.device_index}) faulted: "
            f"{error.name}. A device trap or fault, e.g. a direct DCP collective "
            "spin timeout ('direct DCP ... timeout source=<rank>' on stdout names "
            "the late source rank). Aborting the worker so the engine fails "
            "instead of hanging."
        )
        return True

    def _abort(self, message: str) -> None:
        logger.critical("%s", message)
        print(message, file=sys.stderr, flush=True)
        os._exit(DEVICE_FAULT_EXIT_CODE)

    def _watch(self) -> None:
        try:
            self._setup()
        except Exception as e:  # reported by __init__ on the creating thread
            self._setup_error = repr(e)
            return
        finally:
            self._ready.set()
        while not self._stop.wait(self.POLL_S):
            if self.check():
                return
        self.driver.cuStreamDestroy(self.stream)
        self.driver.cuDevicePrimaryCtxRelease(self._device)


_device_fault_watches: dict[int, _DeviceFaultWatch] = {}
_device_fault_lock = threading.Lock()


def watch_device_faults(device: torch.device) -> None:
    """Abort this process when the CUDA context of ``device`` faults (once per
    device; no-op off CUDA)."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    index = torch.cuda.current_device() if device.index is None else device.index
    with _device_fault_lock:
        if index not in _device_fault_watches:
            _device_fault_watches[index] = _DeviceFaultWatch(index)


class DirectCPWorkspace:
    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        num_ubatches: int,
    ) -> None:
        self.group = group
        self.world_size = group.size()
        self.rank = group.rank()
        self.device = torch.device(device)
        self.num_ubatches = num_ubatches
        self.epoch = torch.zeros(num_ubatches, dtype=torch.int64, device=self.device)
        self._allocations: list[tuple[torch.Tensor, Any, list[torch.Tensor]]] = []
        # Host barrier group for arrive(); None skips the barrier (a standalone
        # group outside vLLM's parallel state). Tests may assign a gloo group.
        self.arrival_group = _cpu_group_of(group)
        self._arrived_generation = 0

    def arrive(self) -> None:
        """Start-up arrival barrier: call before launching a spin collective."""
        if self._arrived_generation == _arrival_generation:
            return
        self._arrived_generation = _arrival_generation
        if self.arrival_group is not None and self.world_size > 1:
            dist.barrier(group=self.arrival_group)

    def _allocate(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        storage = symm_mem.empty(shape, device=self.device, dtype=dtype)
        storage.zero_()
        torch.accelerator.synchronize()
        handle = symm_mem.rendezvous(storage, self.group.group_name)
        assert handle is not None, "CP symmetric memory rendezvous returned None"
        handle.barrier()
        views = [
            handle.get_buffer(peer, list(shape), dtype, 0)
            for peer in range(self.world_size)
        ]
        self.device = storage.device
        peer_ptrs = torch.tensor(
            [
                [view[ubatch].data_ptr() for view in views]
                for ubatch in range(self.num_ubatches)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._allocations.append((storage, handle, views))
        return storage, peer_ptrs

    def _multicast_ptrs(self, storage: torch.Tensor) -> list[int]:
        disabled = [0] * self.num_ubatches
        for allocated, handle, _ in self._allocations:
            if allocated is storage:
                break
        else:
            return disabled
        try:
            from torch._C._autograd import DeviceType
            from torch._C._distributed_c10d import _SymmetricMemory

            if not _SymmetricMemory.has_multicast_support(
                DeviceType.CUDA, storage.device.index
            ):
                return disabled
            multicast_base = handle.multicast_ptr
        except Exception:
            return disabled
        if not multicast_base:
            return disabled
        storage_base = storage.data_ptr()
        return [
            multicast_base + (storage[ubatch].data_ptr() - storage_base)
            for ubatch in range(self.num_ubatches)
        ]
