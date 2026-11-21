# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DWDP of Kimi K3's non-expert weights.

Members: every EP rank of the two engines that share experts (e.g. 2 x 4 = 8), or of
this engine alone with ``dwdp="dense"`` (e.g. DEP8). Per
decoder layer after the first, every weight storage of at least ``MIN_BYTES`` behind a
parameter of the layer (attention and KDA projections, shared expert, latent
projections, router, and the packed MXFP8 projections of ``pack_projections`` with
every view of them; the routed experts live in the MegaMoE) is packed into one byte
blob, cut into one 2 MiB-aligned shard per member. Each layer's blob is one fixed VA
range: this member's shard maps its own fabric memory, the others map staging slot
``ordinal % depth``. Every tensor object of the layer that views a moved storage
(parameters, the packed holders, the fused prefill GEMMs' weight views) is re-pointed
in place (``Tensor.set_``) at the same offset and strides inside that range, so its
pointer never changes afterwards. The old storage must then be unreferenced: a moved
storage that something outside the walk still holds raises.

A layer's shards are copied (one copy-engine read per member, on the expert DWDP
streams if any) when the layer ``depth - 1`` before it starts; the slot they overwrite
was last read by a layer that has fully finished, including its shared expert's side
stream. Installed at load, after the projections are packed (``ParallelConfig.dwdp``).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.multiprocessing.reductions import StorageWeakRef

from vllm.logger import init_logger

from .megamoe_rubin import dwdp as edwdp
from .megamoe_rubin.symm_arena import aligned

logger = init_logger(__name__)

DEPTH = 3  # staging slots: copies start two layers ahead
MIN_BYTES = 1 << 20


class DenseDwdp:
    def __init__(self, layers: list[nn.Module], start: int, x: edwdp.ExpertDwdp | None):
        from vllm.distributed import get_ep_group

        g = get_ep_group()
        self.start = start
        if x is not None:  # both engines' EP ranks
            n, me = 2 * g.world_size, x.index * g.world_size + g.rank_in_group
            self.streams, self.handles = x.streams, x.stream_handles
        else:  # this engine's EP ranks
            n, me = g.world_size, g.rank_in_group
            self.streams, self.handles = edwdp.copy_streams()
        chunk = edwdp._dies().CHUNK
        device = torch.accelerator.current_device_index()

        plans, resident = _plan_layers(layers)
        plans = [(items, aligned(-(-off // n), chunk)) for items, off in plans]
        # The fabric memory comes from the driver, while the parameters it replaces
        # were allocated through PyTorch's caching allocator: install one layer at a
        # time and hand each layer's replaced bytes back to the driver before the
        # next layer's home shard is created, so the full non-expert copy and the
        # shards are never held together. The staging slots are created last, from
        # the memory the replaced layers returned.
        torch.accelerator.empty_cache()
        exports: dict[str, bytes] = {}
        blobs = []
        for ordinal in range(1, len(plans) + 1):
            items, shard = plans[ordinal - 1]
            plans[ordinal - 1] = (None, shard)  # drop the old storages with `items`
            rng = edwdp.VaRange(shard * n, device)
            home = edwdp.create(shard, device)
            exports[str(ordinal)] = edwdp.export(home)
            rng.map(me * shard, shard, home)
            blob = rng.tensor()
            lo, hi = me * shard, (me + 1) * shard
            mine = blob[lo:hi]
            mine.zero_()
            refs = []
            for storage, views, off in items:  # this member's rows of the storages
                src = _storage_u8(storage)
                a, b = max(off, lo), min(off + storage.nbytes(), hi)
                if a < b:
                    mine[a - lo : b - lo].copy_(src[a - off : b - off])
                del src
            target = blob.untyped_storage()
            base = blob.storage_offset()
            with torch.no_grad():
                for storage, views, off in items:
                    for t in views:
                        _repoint(t, target, (base + off) // t.element_size())
                    refs.append(
                        (StorageWeakRef(storage), storage.nbytes(),
                         tuple(views[0].shape), views[0].dtype)
                    )  # fmt: skip
            del items, storage, views
            torch.cuda.synchronize(device)
            leaked = [r for r in refs if not r[0].expired()]
            del refs
            if leaked:
                raise RuntimeError(
                    f"Dense DWDP layer {ordinal}: {len(leaked)} moved weight storage(s) "
                    "are still referenced outside the layer's modules (e.g. "
                    f"{[(nb, shape, str(dt)) for _, nb, shape, dt in leaked[:4]]}); "
                    "their old copy would stay resident."
                )
            blobs.append((rng, blob, shard))
            torch.accelerator.empty_cache()
        # slots[(k, m, shard)]: staging memory for member m's shard in slot k (a
        # mapping covers a whole allocation, so each shard size has its own)
        slots: dict[tuple[int, int, int], object] = {}
        for ordinal, (rng, _, shard) in enumerate(blobs, start=1):
            for m in range(n):
                if m != me:
                    key = (ordinal % DEPTH, m, shard)
                    if key not in slots:
                        slots[key] = edwdp.create(shard, device)
                    rng.map(m * shard, shard, slots[key])

        store = edwdp.Store("dense", n, me) if x is not None else edwdp.GroupStore(g)
        handles = store.exchange(exports)
        # copies[i]: layer start + i (the first layer is resident: no copies)
        self.copies = [edwdp.Copies([], len(self.streams))]
        for ordinal, (_, blob, shard) in enumerate(blobs, start=1):
            pairs = []
            for m in range(n):
                if m != me:
                    r = edwdp.VaRange(shard, device)
                    r.map_imported(0, shard, handles[m][str(ordinal)])
                    pairs.append((blob[m * shard : (m + 1) * shard], r.tensor()))
            self.copies.append(edwdp.Copies(pairs, len(self.streams)))
        store.barrier("ready")
        moved = sum(shard * (n - 1) for *_, shard in blobs) / max(len(blobs), 1)
        logger.info(
            "Dense DWDP: member %d/%d, %d layers, %.2f GB/layer copied, %.2f GiB "
            "of weight storages shared across layers resident",
            me,
            n,
            len(blobs),
            moved / 1e9,
            resident / 2**30,
        )

    def _fetch(self, i: int) -> None:
        if i < len(self.copies):
            self.copies[i].fetch(self.streams, self.handles)

    def layer_start(self, layer_idx: int) -> None:
        """Before a decoder layer: wait for its shards, then copy those of the layer
        ``depth - 1`` ahead (and, at the first layer, the ones in between)."""
        i = layer_idx - self.start
        if i == 0:
            for j in range(1, DEPTH - 1):
                self._fetch(j)
        self.copies[i].wait()
        self._fetch(i + DEPTH - 1)


def _repoint(t: torch.Tensor, target: torch.UntypedStorage, start: int) -> None:
    """Make the tensor object ``t`` view ``target`` from element ``start`` (the moved
    storage's first element), keeping its offset, sizes and strides. ``set_`` keeps
    the object, so every holder of ``t`` follows; tensors made by ``.data`` /
    ``.detach()`` refuse ``set_`` and take the same view through ``.data``."""
    args = (target, start + t.storage_offset(), t.size(), t.stride())
    try:
        t.set_(*args)
    except RuntimeError:
        t.data = torch.empty(0, dtype=t.dtype, device=t.device).set_(*args)
    if t.untyped_storage().data_ptr() != target.data_ptr():
        raise RuntimeError("Dense DWDP: a weight view could not be re-pointed.")


def _storage_u8(storage: torch.UntypedStorage) -> torch.Tensor:
    """All bytes of ``storage`` as a flat uint8 tensor."""
    return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )


def _walk(obj, out: list[torch.Tensor], seen: set[int], stop: set[int], depth=0):
    """Every tensor object reachable from ``obj`` through module attributes,
    containers and vLLM objects, without entering ``stop`` (other decoder layers,
    the MegaMoE experts: their weights are the expert DWDP's or stay resident)."""
    if depth > 12 or id(obj) in seen or id(obj) in stop:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        out.append(obj)
        return
    if isinstance(obj, dict):
        vals = list(obj.values())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        vals = list(obj)
    elif isinstance(obj, nn.Module) or (
        type(obj).__module__.startswith("vllm.") and hasattr(obj, "__dict__")
    ):
        vals = list(vars(obj).values())
    else:
        return
    for v in vals:
        _walk(v, out, seen, stop, depth + 1)


def _plan_layers(layers: list[nn.Module]):
    """Per layer after the first: ``(items, shard)`` with ``items`` =
    ``[(storage, tensor objects viewing it, blob offset)]``."""
    from .megamoe_rubin.experts import KimiK3MegaMoEExperts

    experts = {
        id(m) for layer in layers for m in layer.modules()
        if isinstance(m, KimiK3MegaMoEExperts)
    }  # fmt: skip
    found = []
    owner: dict[int, int] = {}
    for i, layer in enumerate(layers):
        stop = {id(other) for other in layers if other is not layer} | experts
        tensors: list[torch.Tensor] = []
        _walk(layer, tensors, set(), stop)
        groups: dict[int, list] = {}
        for t in tensors:
            if t.device.type != "cuda":
                continue
            st = t.untyped_storage()
            g = groups.setdefault(st.data_ptr(), [st, [], False])
            if all(t is not v for v in g[1]):
                g[1].append(t)
            g[2] |= isinstance(t, nn.Parameter)
        for key, (st, views, has_param) in groups.items():
            if has_param and st.nbytes() >= MIN_BYTES:
                owner[key] = -1 if key in owner else i
        found.append(groups)
    plans, resident = [], 0
    for key, i in owner.items():
        if i <= 0:  # the first layer, or shared with another layer
            for groups in found:
                if key in groups:
                    resident += groups[key][0].nbytes() if i == -1 else 0
                    break
    for i, groups in enumerate(found):
        if i == 0:
            continue
        items, off = [], 0
        for key, (st, views, has_param) in groups.items():
            if owner.get(key) != i:
                continue  # small, not a weight, or shared with another layer
            items.append((st, views, off))
            off = aligned(off + st.nbytes())
        plans.append((items, off))
    return plans, resident


_D: DenseDwdp | None = None


def install(layers: nn.ModuleList, start: int, end: int) -> None:
    """Shard the non-expert weights of ``layers[start:end]`` (the first stays
    resident), at load after the expert DWDP (if any) finalized."""
    global _D
    if edwdp.dense_enabled() and _D is None:
        _D = DenseDwdp(list(layers[start:end]), start, edwdp.get())


def get() -> DenseDwdp | None:
    return _D
