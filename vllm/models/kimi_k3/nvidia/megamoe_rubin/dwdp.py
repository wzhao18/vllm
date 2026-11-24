# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-engine expert DWDP for the Rubin MegaMoE (EP x DWDP2).

Two engines with the same EP layout (e.g. the two DEP4 engines of an 8-GPU prefill
instance) share their routed experts: EP rank ``r`` of each engine holds one half of
its local experts (its *home* half) and copies the other half from EP rank ``r`` of the
partner engine before each MoE layer. The engines never synchronize: the copies are
one-sided copy-engine reads of the partner's fabric memory, so each engine keeps its
own DEP scheduling and an EP group of 4 instead of 8.

Per MoE layer, each kernel tensor (fc1/fc2 weight and block scale, experts on dim 0) is
one fixed VA range: the home half maps this engine's fabric memory (exported to the
partner), the other half maps staging slot ``ordinal % depth``. The pointers never
change, so bound launch arguments and CUDA graphs see ordinary tensors, and nothing is
merged. A leading pad aligns the half boundary of the block-scale planes to 2 MiB.

Locality domains: when the die-local MegaMoE streams are available, half ``p`` (experts
``[0, E/2)`` or ``[E/2, E)``) is homed on die ``p``, where its experts are read, and
the copies land there. The engine owning half ``p`` alternates layer by layer, so each
engine's home memory (and the partner's reads of it) is spread over both dies.

Schedule: the next layer's halves are copied right after a MoE kernel is enqueued, so
the copies overlap the next attention and stay off the EP dispatch/combine NVLink
traffic. The first MoE layer stays resident. Weights are fixed after loading: EPLB is
not supported.

Layout config (``ParallelConfig.dwdp``, set by ``configure`` at model init):
``dwdp="experts"`` enables DWDP of the experts and of the non-expert weights
(``dense_dwdp``); ``dwdp_dir`` is shared by both engines (fresh per run), and they
pair and take engine indices through it. ``dwdp="dense"`` enables only the
non-expert DWDP, inside one engine (e.g. DEP8).
"""

from __future__ import annotations

import atexit
import datetime
import json
import os

import cuda.bindings.driver as cuda
import torch

from vllm.logger import init_logger

from .symm_arena import aligned

logger = init_logger(__name__)

# Set by ``configure`` from ``ParallelConfig.dwdp`` / ``dwdp_dir``.
LOCAL = False  # dwdp="dense": one engine, non-expert DWDP only
DIR = ""  # dwdp="experts": the pairing directory
ENABLED = False  # expert DWDP across two engines


def configure(parallel_config) -> None:
    """Set this process's DWDP layout from the parallel config (model init)."""
    global LOCAL, DIR, ENABLED
    mode = getattr(parallel_config, "dwdp", "none")
    LOCAL = mode == "dense"
    DIR = (parallel_config.dwdp_dir or "") if mode == "experts" else ""
    ENABLED = bool(DIR)


def dense_enabled() -> bool:
    """Non-expert DWDP (``dense_dwdp``) runs: dwdp="dense" or "experts"."""
    return ENABLED or LOCAL
DEPTH = 2  # expert staging slots: copies start one layer ahead
N_STREAMS = 4  # copy-engine streams
_FABRIC = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
_TIMEOUT = datetime.timedelta(seconds=1800)


def _dies():
    from ..specialized_decode.cutedsl.common import dies  # imports cutlass

    return dies


def create(nbytes: int, device: int, die: int | None = None):
    """Fabric-exportable physical memory of ``nbytes`` (a multiple of 2 MiB), homed on
    locality domain ``die`` when given."""
    dies = _dies()
    if die is None:
        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device
    else:
        prop = dies._prop(device, die)
    prop.requestedHandleTypes = _FABRIC
    return dies._checked(cuda.cuMemCreate(nbytes, prop, 0), "DWDP create")


def release(handle) -> None:
    _dies()._checked(cuda.cuMemRelease(handle), "DWDP release")


def export(handle) -> bytes:
    result = cuda.cuMemExportToShareableHandle(handle, _FABRIC, 0)
    return bytes(_dies()._checked(result, "DWDP export").data)


class VaRange:
    """A reserved VA range whose 2 MiB chunks map physical memory; ``tensor()`` is a
    uint8 view that keeps the range alive. Ranges live for the process."""

    def __init__(self, nbytes: int, device: int):
        self.device = device
        self.nbytes = aligned(nbytes, _dies().CHUNK)
        self.base = int(
            _dies()._checked(
                cuda.cuMemAddressReserve(self.nbytes, _dies().CHUNK, 0, 0),
                "DWDP reserve",
            )
        )

    def map(self, offset: int, size: int, handle, writable: bool = True) -> None:
        checked = _dies()._checked
        checked(cuda.cuMemMap(self.base + offset, size, 0, handle, 0), "DWDP map")
        access = cuda.CUmemAccessDesc()
        access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = self.device
        access.flags = (
            cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            if writable
            else cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
        )
        checked(
            cuda.cuMemSetAccess(self.base + offset, size, [access], 1), "DWDP access"
        )

    def unmap(self, offset: int, size: int) -> None:
        _dies()._checked(cuda.cuMemUnmap(self.base + offset, size), "DWDP unmap")

    def map_imported(self, offset: int, size: int, data: bytes) -> None:
        """Map (read-only) the partner memory whose fabric handle is ``data``."""
        fh = cuda.CUmemFabricHandle()
        fh.data = data
        handle = _dies()._checked(
            cuda.cuMemImportFromShareableHandle(fh.getPtr(), _FABRIC), "DWDP import"
        )
        self.map(offset, size, handle, writable=False)

    @property
    def __cuda_array_interface__(self):
        return dict(
            shape=(self.nbytes,), typestr="|u1", data=(self.base, False), version=3
        )

    def tensor(self) -> torch.Tensor:
        return torch.as_tensor(self, device=torch.device("cuda", self.device))


def storage_bytes(t: torch.Tensor) -> torch.Tensor:
    """The contiguous storage bytes behind ``t`` (also behind permuted kernel views)."""
    n = t.numel() * t.element_size()
    return torch.empty(0, dtype=torch.uint8, device=t.device).set_(
        t.untyped_storage(), t.storage_offset() * t.element_size(), (n,), (1,)
    )


class Store:
    """A FileStore in ``dwdp_dir`` shared by the members of one DWDP group."""

    def __init__(self, name: str, members: int, me: int):
        self.store = torch.distributed.FileStore(os.path.join(DIR, name), members)
        self.members, self.me = members, me

    def exchange(
        self, mine: dict[str, bytes], tag: str = "handles"
    ) -> list[dict[str, bytes]]:
        """Publish this member's fabric handles under ``tag`` (one exchange per tag);
        returns every member's."""
        payload = json.dumps({k: v.hex() for k, v in mine.items()})
        self.store.set(f"{tag}/{self.me}", payload)
        keys = [f"{tag}/{m}" for m in range(self.members)]
        self.store.wait(keys, _TIMEOUT)
        return [
            {k: bytes.fromhex(v) for k, v in json.loads(self.store.get(key)).items()}
            for key in keys
        ]

    def barrier(self, tag: str) -> None:
        self.store.set(f"{tag}/{self.me}", b"1")
        self.store.wait([f"{tag}/{m}" for m in range(self.members)], _TIMEOUT)


class GroupStore:
    """``Store`` over the members of one engine's process group."""

    def __init__(self, group):
        self.group = group

    def exchange(
        self, mine: dict[str, bytes], tag: str = "handles"
    ) -> list[dict[str, bytes]]:
        del tag  # a collective: calls pair up in order
        out: list = [None] * self.group.world_size
        torch.distributed.all_gather_object(out, mine, group=self.group.cpu_group)
        return out

    def barrier(self, tag: str) -> None:
        self.group.barrier()


def copy_streams() -> tuple[list[torch.cuda.Stream], list]:
    streams = [torch.cuda.Stream() for _ in range(N_STREAMS)]
    return streams, [cuda.CUstream(s.cuda_stream) for s in streams]


class Copies:
    """One layer's copies (``(dst, src)`` uint8 pairs) as raw copy-engine memcpys, each
    pair cut into one contiguous piece per copy stream; events are preallocated."""

    def __init__(self, pairs: list[tuple[torch.Tensor, torch.Tensor]], n_streams: int):
        self.jobs: list[tuple[int, int, int, int]] = []  # (dst, src, bytes, stream)
        for dst, src in pairs:
            piece = aligned(-(-dst.numel() // n_streams), 4096)
            for i, o in enumerate(range(0, dst.numel(), piece)):
                n = min(piece, dst.numel() - o)
                self.jobs.append((dst.data_ptr() + o, src.data_ptr() + o, n, i))
        self.go = torch.cuda.Event()
        self.done = [torch.cuda.Event() for _ in range(n_streams)]
        self.in_flight = False

    def fetch(self, streams: list[torch.cuda.Stream], handles: list) -> None:
        """Copy on ``streams`` after the work already on the current stream."""
        if self.in_flight or not self.jobs:
            return
        self.go.record()
        for s in streams:
            s.wait_event(self.go)
        for dst, src, n, i in self.jobs:
            _dies()._checked(cuda.cuMemcpyAsync(dst, src, n, handles[i]), "DWDP copy")
        for s, ev in zip(streams, self.done):
            ev.record(s)
        self.in_flight = True

    def wait(self) -> None:
        if self.in_flight:
            main = torch.cuda.current_stream()
            for ev in self.done:
                main.wait_event(ev)
            self.in_flight = False


class _Layer:
    def __init__(self, ordinal: int, experts: int, home: int | None):
        self.ordinal = ordinal
        self.experts = experts  # local experts of the layer (both halves)
        self.home = home  # the half this engine owns; None: the resident first layer
        # (dst, key, size, data offset) of the partner's half, mapped at finalize
        self.pending: list[tuple] = []
        self.copies: Copies | None = None


class ExpertDwdp:
    def __init__(self):
        from vllm.config import get_current_vllm_config
        from vllm.distributed import get_ep_group

        from .die_local import die_sm_mask32

        if get_current_vllm_config().parallel_config.enable_eplb:
            raise ValueError("Expert DWDP (dwdp=\"experts\") does not support EPLB.")
        g = get_ep_group()
        os.makedirs(DIR, exist_ok=True)
        index = None
        if g.rank_in_group == 0:  # engines take indices in arrival order
            engines = torch.distributed.FileStore(os.path.join(DIR, "engines"), 2)
            index = engines.add("engines", 1) - 1
        self.index = g.broadcast_object(index, src=0)
        if self.index not in (0, 1):
            raise RuntimeError(f"dwdp_dir={DIR} is used by more than two engines.")
        self.device = torch.accelerator.current_device_index()
        # Each expert half belongs on its die when the die-local streams run.
        self.die_placed = die_sm_mask32(self.device) is not None
        self.streams, self.stream_handles = copy_streams()
        self.ep_rank = g.rank_in_group
        self.store = Store(f"experts_ep{self.ep_rank}", 2, self.index)
        self.order: list[_Layer] = []
        self.slots: dict[tuple[int, str], object] = {}
        self.exports: dict[str, bytes] = {}

    # -- loading ---------------------------------------------------------------------

    def claim(self, num_local_experts: int) -> tuple[_Layer, int, int]:
        """At construction of one MoE layer (module order, identical on every rank):
        its DWDP layer and the local experts ``[lo, hi)`` this rank holds, which are
        the only ones it allocates and loads. The first layer stays resident (all
        experts); every other layer keeps the half this engine owns,
        ``index ^ (ordinal & 1)``."""
        from .die_local import expert_split

        ordinal = len(self.order)
        if ordinal == 0:
            layer = _Layer(0, num_local_experts, None)
            self.order.append(layer)
            return layer, 0, num_local_experts
        half = expert_split(num_local_experts)
        if 2 * half != num_local_experts:
            raise ValueError(
                f"Expert DWDP needs an even local expert count, got {num_local_experts}."
            )
        home = self.index ^ (ordinal & 1)
        layer = _Layer(ordinal, num_local_experts, home)
        self.order.append(layer)
        return layer, home * half, (home + 1) * half

    def register(self, module, weights):
        """Replace one MoE layer's kernel weights (``RubinMoeWeights``) with tensors
        assembled from home memory and a staging slot. ``weights`` holds the experts
        ``claim`` gave this rank: all of them on the first layer, which stays
        resident (die-placed when the die streams run), else the home half."""
        from .die_local import place_experts

        layer = module._dwdp_layer
        assert self.order[layer.ordinal] is layer, "register: unclaimed layer"
        if layer.home is None:
            return place_experts(weights) if self.die_placed else weights
        tensors = (*weights.fc1, *weights.fc2)
        new = [self._assemble(layer, name, t) for name, t in zip("ABCD", tensors)]
        return weights._replace(fc1=tuple(new[:2]), fc2=tuple(new[2:]))

    def _assemble(self, layer: _Layer, name: str, t: torch.Tensor) -> torch.Tensor:
        """One kernel tensor as [half 0 | half 1] in a fixed VA range from the home
        half ``t`` (experts on dim 0); half ``p`` is owned by engine ``p`` on even
        layers, by the other engine on odd ones."""
        if 2 * t.shape[0] != layer.experts:
            raise ValueError(
                f"expert DWDP: layer {layer.ordinal} {name} has {t.shape[0]} experts, "
                f"expected the home half of {layer.experts}."
            )
        src = storage_bytes(t)
        hb = src.numel()
        chunk = _dies().CHUNK
        pad = (-hb) % chunk  # the half boundary lands on a chunk
        total = aligned(pad + 2 * hb, chunk)
        rng = VaRange(total, self.device)
        full = rng.tensor()[pad : pad + 2 * hb]
        spans = ((0, pad + hb, pad), (pad + hb, total, 0))  # (va0, va1, data offset)
        for p, (va0, va1, off) in enumerate(spans):
            die = p if self.die_placed else None
            owner = p ^ (layer.ordinal & 1)
            key = f"{layer.ordinal}/{name}/{p}"
            if owner == self.index:
                assert p == layer.home
                handle = create(va1 - va0, self.device, die)
                self.exports[key] = export(handle)
                rng.map(va0, va1 - va0, handle)
                full[p * hb : (p + 1) * hb].copy_(src)
            else:
                slot = (layer.ordinal % DEPTH, f"{name}/{p}")
                if slot not in self.slots:
                    self.slots[slot] = create(va1 - va0, self.device, die)
                rng.map(va0, va1 - va0, self.slots[slot])
                layer.pending.append((full[p * hb : (p + 1) * hb], key, va1 - va0, off))
        shape = (layer.experts, *t.shape[1:])
        return full.view(t.dtype).as_strided(shape, t.stride())

    def finalize(self) -> None:
        """After every layer registered (on both engines): map the partner's halves.
        Runs at load."""
        partner = self.store.exchange(self.exports)[1 - self.index]
        self.exports.clear()
        for layer in self.order:
            pairs = []
            for dst, key, size, off in layer.pending:
                rng = VaRange(size, self.device)
                rng.map_imported(0, size, partner[key])
                pairs.append((dst, rng.tensor()[off : off + dst.numel()]))
            layer.pending.clear()
            layer.copies = Copies(pairs, len(self.streams))
        self.store.barrier("ready")
        atexit.register(self._leave_together)
        logger.info(
            "DWDP experts: engine %d, EP rank %d, %d MoE layers, die-placed %s",
            self.index,
            self.ep_rank,
            len(self.order),
            self.die_placed,
        )

    def _leave_together(self) -> None:
        # The partner may read our home memory until it is done too.
        try:
            torch.accelerator.synchronize()
            self.store.barrier("exit")
        except Exception:  # noqa: BLE001 - best effort at interpreter exit
            pass

    # -- per step --------------------------------------------------------------------

    def _fetch(self, layer: _Layer) -> None:
        assert layer.copies is not None, "fetch before finalize"
        layer.copies.fetch(self.streams, self.stream_handles)

    def before_experts(self, module) -> None:
        """Before a MoE layer's kernel: wait for its copied half."""
        copies = module._dwdp_layer.copies
        if copies is not None:
            copies.wait()

    def after_experts(self, module) -> None:
        """After a MoE layer's kernel was enqueued: copy the layers up to ``DEPTH - 1``
        ahead (all of them after the first layer). Their slots were last read by MoE
        kernels already enqueued, and the copies overlap the next attention."""
        o = module._dwdp_layer.ordinal
        for layer in self.order[1 if o == 0 else o + DEPTH - 1 : o + DEPTH]:
            self._fetch(layer)


_D: ExpertDwdp | None = None


def get() -> ExpertDwdp | None:
    global _D
    if ENABLED and _D is None:
        _D = ExpertDwdp()
    return _D
