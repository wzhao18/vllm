# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Counted fabric windows: CUDA 13.5 Compute Fabric Transport (logical endpoints)
with counted completion, for the decode NVLink exchanges.

A ``FabricWindow`` is one FABRIC-handle VMM buffer per rank of a process group,
bound to a unicast logical endpoint (LE) with counted operations, plus (optionally)
a multicast LE over the group with one replica per rank. Device code
(``cutedsl.common.fabric``) writes into a peer's window with
``fabric.try_put ... counted::bytes``: the payload and its completion travel as one
fabric operation, and the receiver waits on an 8-byte byte counter in its OWN window
(no flag round, no producer fence, no Lamport sentinel to poll or re-arm). A
``.multimem`` put through the multicast LE lands one copy on every rank.

Layout of every window (unicast and multicast replica alike): ``COUNTERS`` 8-byte
counters, 256 B apart, in [0, ``DATA``); data from byte ``DATA``. Counters only
grow (a put adds its byte count): a kernel waits for ``counter >= expected`` where
``expected`` is a running total it keeps itself (``expected`` below: one int64 per
counter and CTA, so CUDA-graph replays need no host bookkeeping).

Endpoint ids are per process: every rank reserves ``world`` consecutive ids and
keeps rank r's endpoint at ``base + r`` (its own created, the others imported), so a
kernel addresses rank r as ``base + r`` from one Int32 argument.

Setup follows the CUDA programming guide (Compute Fabric Transport): unicast LEs are
created, bound and exported by their owner, then imported by the peers; the
multicast LE is created and exported by rank 0, imported by the others, every rank
adds its own device and binds its own replica, and nobody issues an operation until
every replica is bound (a group barrier). The driver entry points are resolved with
``cuGetProcAddress`` (CUDA 13.5 driver API; the image's Python bindings predate
them). Capability is checked once (``supported``); a group without it raises
``FabricUnavailable`` at construction (no silent fallback).
"""

import ctypes
import functools
import time
import weakref

import cuda.bindings.driver as cuda
import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)

COUNTER_STRIDE = 256  # counters must be 256 B aligned
COUNTERS = 256
DATA = COUNTERS * COUNTER_STRIDE  # first data byte of every window
GRANULE = 2 << 20  # endpoint bind alignment on VR200 (cuLogicalEndpointGetLimits)
MAX_CTAS = 1024  # expected-counter slots per counter

# CUDA 13.5 driver enums (cuda.h): device attributes and logical-endpoint constants.
_ATTR_LE_UNICAST = 153
_ATTR_LE_MULTICAST = 154
_ATTR_LE_COUNTED_OPS = 155
_ATTR_LE_HANDLE_TYPES = 158
_LE_TYPE_UNICAST, _LE_TYPE_MULTICAST = 1, 2
_LE_HANDLE_FABRIC = 1
_LE_FLAG_COUNTED_OPS = 1
_LE_QUERY_READY = 1
_HANDLE_BYTES = 256  # exported handle buffer (the driver writes 64 B)
_DRIVER_VERSION = 13050


class FabricUnavailable(RuntimeError):
    """The device, driver or container cannot run counted fabric operations."""


class _LeProp(ctypes.Structure):
    # CUlogicalEndpointProp: {type; union {CUdevice device; unsigned numDevices};
    # size; ipcHandleTypes; flags; reserved} (the union layout is the accepted one).
    _fields_ = [
        ("type", ctypes.c_int),
        ("target", ctypes.c_int),
        ("size", ctypes.c_uint64),
        ("ipc_handle_types", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 8),
    ]


_U32, _U64, _INT, _P = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p
_PROTOTYPES = {
    "cuLogicalEndpointIdReserve": (ctypes.POINTER(_U32), _U32),
    "cuLogicalEndpointCreate": (_U32, ctypes.POINTER(_LeProp)),
    "cuLogicalEndpointAddDevice": (_U32, _INT),
    "cuLogicalEndpointQuery": (_U32, _U32, ctypes.POINTER(_INT)),
    "cuLogicalEndpointBindMem": (_U32, _INT, _U64, _U64, _U64, _U64, _U64),
    "cuLogicalEndpointExport": (_P, _U32, _INT),
    "cuLogicalEndpointImport": (_U32, _P, _INT),
    "cuLogicalEndpointDestroy": (_U32,),
}


def _checked(result, operation):
    status, *values = result
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {status}")
    return values[0] if len(values) == 1 else tuple(values)


@functools.cache
def _driver():
    calls = {}
    flags = cuda.CUdriverProcAddress_flags.CU_GET_PROC_ADDRESS_DEFAULT
    for name, args in _PROTOTYPES.items():
        status, address, _ = cuda.cuGetProcAddress(
            name.encode(), _DRIVER_VERSION, flags
        )
        if status != cuda.CUresult.CUDA_SUCCESS or not int(address):
            raise FabricUnavailable(f"driver has no {name} (needs a CUDA 13.5 driver)")
        calls[name] = ctypes.CFUNCTYPE(_INT, *args)(int(address))
    return calls


def _call(name, *args):
    status = _driver()[name](*args)
    if status != 0:
        raise RuntimeError(f"{name} failed: CUresult {status}")


@functools.cache
def supported(device: int) -> bool:
    """Unicast + multicast logical endpoints with counted operations and FABRIC
    handles on ``device``, and the driver entry points present."""
    values = []
    for attr in (
        _ATTR_LE_UNICAST, _ATTR_LE_MULTICAST, _ATTR_LE_COUNTED_OPS,
        _ATTR_LE_HANDLE_TYPES,
    ):  # fmt: skip
        status, value = cuda.cuDeviceGetAttribute(attr, device)
        values.append(value if status == cuda.CUresult.CUDA_SUCCESS else 0)
    if not (values[0] and values[1] and values[2] and values[3] & _LE_HANDLE_FABRIC):
        return False
    try:
        _driver()
    except FabricUnavailable:
        return False
    return True


def _wait_ready(endpoint: int, what: str, timeout: float = 30.0) -> None:
    ready, deadline = _INT(0), time.monotonic() + timeout
    while True:
        _call("cuLogicalEndpointQuery", endpoint, _LE_QUERY_READY, ctypes.byref(ready))
        if ready.value:
            return
        if time.monotonic() > deadline:
            raise RuntimeError(f"logical endpoint {endpoint} ({what}) never ready")
        time.sleep(1e-4)


class _FabricMemory:
    """``nbytes`` (2 MiB multiple) of FABRIC-handle device memory mapped for this
    device only; ``__cuda_array_interface__`` (uint8) for ``torch.as_tensor``."""

    def __init__(self, nbytes: int, device: int):
        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device
        prop.requestedHandleTypes = (
            cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        )
        self.nbytes = nbytes
        self.handle = _checked(cuda.cuMemCreate(nbytes, prop, 0), "cuMemCreate")
        self.base = int(
            _checked(cuda.cuMemAddressReserve(nbytes, GRANULE, 0, 0), "reserve")
        )
        _checked(cuda.cuMemMap(self.base, nbytes, 0, self.handle, 0), "cuMemMap")
        access = cuda.CUmemAccessDesc()
        access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = device
        access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _checked(cuda.cuMemSetAccess(self.base, nbytes, [access], 1), "access")
        weakref.finalize(self, _FabricMemory._release, self.base, nbytes, self.handle)

    @staticmethod
    def _release(base, nbytes, handle):
        cuda.cuMemUnmap(base, nbytes)
        cuda.cuMemRelease(handle)
        cuda.cuMemAddressFree(base, nbytes)

    @property
    def __cuda_array_interface__(self):
        return dict(
            shape=(self.nbytes,), typestr="|u1", data=(self.base, False), version=3
        )


class FabricWindow:
    """Counted fabric window of ``nbytes`` data bytes on every rank of ``group``
    (collective). Attributes:

    - ``base``: Int32 kernel argument; rank r's unicast endpoint id is ``base + r``.
    - ``multicast``: the multicast endpoint id (``-1`` without ``multicast=True``).
    - ``local`` / ``mc_local``: uint8 views of this rank's unicast window and its
      multicast replica ([DATA + nbytes] each; counters first).
    - ``expected``: int64 [COUNTERS, MAX_CTAS] zeros, for kernels' running targets.

    Byte offsets passed to puts are window offsets (``DATA + ...``); counter i sits
    at byte ``i * COUNTER_STRIDE`` (``counter(i)``)."""

    def __init__(self, group, nbytes: int, multicast: bool = True, device=None):
        if group is None:
            group = dist.group.WORLD
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        device = torch.device(device if device is not None else "cuda")
        index = device.index if device.index is not None else torch.cuda.current_device()
        self.device = torch.device("cuda", index)
        total = -(-(DATA + nbytes) // GRANULE) * GRANULE
        self.nbytes = nbytes
        self._endpoints: list[int] = []
        with torch.accelerator.device_index(index):
            torch.cuda.current_stream(self.device)  # primary context current
            ok = torch.tensor([int(supported(index))], device=self.device)
            dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
            if not ok.item():
                raise FabricUnavailable(
                    "counted fabric operations are unavailable on some rank of the "
                    "group (driver, device or IMEX channel)"
                )
            dev = int(_checked(cuda.cuDeviceGet(index), "cuDeviceGet"))
            self._memory = _FabricMemory(total, index)
            self.local = torch.as_tensor(self._memory, device=self.device)
            self.local.zero_()
            first = _U32(0)
            _call("cuLogicalEndpointIdReserve", ctypes.byref(first), self.world)
            self.base = int(first.value)
            own = self.base + self.rank
            self._create(own, _LE_TYPE_UNICAST, dev, total)
            _call("cuLogicalEndpointBindMem", own, dev, 0, int(self._memory.handle),
                  0, total, 0)  # fmt: skip
            handles = self._exchange(self._export(own))
            for r, handle in enumerate(handles):
                if r != self.rank:
                    self._import(self.base + r, handle)
            self.multicast, self.mc_local = -1, None
            if multicast:
                self._build_multicast(dev, total)
            torch.cuda.synchronize(self.device)
            dist.barrier(group=group)  # every endpoint bound everywhere before use
            self.expected = torch.zeros(
                (COUNTERS, MAX_CTAS), dtype=torch.int64, device=self.device
            )

    @staticmethod
    def counter(i: int) -> int:
        """Window byte offset of counter ``i``."""
        assert 0 <= i < COUNTERS
        return i * COUNTER_STRIDE

    def _create(self, endpoint, kind, target, size):
        prop = _LeProp(kind, target, size, _LE_HANDLE_FABRIC, _LE_FLAG_COUNTED_OPS)
        _call("cuLogicalEndpointCreate", endpoint, ctypes.byref(prop))
        self._endpoints.append(endpoint)
        if kind == _LE_TYPE_UNICAST:
            _wait_ready(endpoint, "unicast")

    def _export(self, endpoint) -> bytes:
        buffer = ctypes.create_string_buffer(_HANDLE_BYTES)
        _call("cuLogicalEndpointExport", buffer, endpoint, _LE_HANDLE_FABRIC)
        return buffer.raw

    def _import(self, endpoint, handle: bytes, ready=True):
        buffer = ctypes.create_string_buffer(handle, _HANDLE_BYTES)
        _call("cuLogicalEndpointImport", endpoint, buffer, _LE_HANDLE_FABRIC)
        self._endpoints.append(endpoint)
        if ready:
            _wait_ready(endpoint, "imported")

    def _exchange(self, handle: bytes) -> list[bytes]:
        handles: list = [None] * self.world
        dist.all_gather_object(handles, handle, group=self.group)
        return handles

    def _build_multicast(self, dev, total):
        mc = _U32(0)
        _call("cuLogicalEndpointIdReserve", ctypes.byref(mc), 1)
        self.multicast = int(mc.value)
        handle = [None]
        if self.rank == 0:
            self._create(self.multicast, _LE_TYPE_MULTICAST, self.world, total)
            handle[0] = self._export(self.multicast)
        source = dist.get_global_rank(self.group, 0)
        dist.broadcast_object_list(handle, src=source, group=self.group)
        if self.rank != 0:
            self._import(self.multicast, handle[0], ready=False)
        _call("cuLogicalEndpointAddDevice", self.multicast, dev)
        _wait_ready(self.multicast, "multicast (group incomplete?)")
        self._mc_memory = _FabricMemory(total, self.device.index)
        self.mc_local = torch.as_tensor(self._mc_memory, device=self.device)
        self.mc_local.zero_()
        _call("cuLogicalEndpointBindMem", self.multicast, dev, 0,
              int(self._mc_memory.handle), 0, total, 0)  # fmt: skip

    def close(self) -> None:
        """Destroy the endpoints (collective use only: every rank closes)."""
        for endpoint in reversed(self._endpoints):
            _driver()["cuLogicalEndpointDestroy"](endpoint)
        self._endpoints.clear()
