# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DWDP of Kimi K3's non-expert weights.

Members: every EP rank of the two engines that share experts (e.g. 2 x 4 = 8), or of
this engine alone with ``dwdp="dense"`` (e.g. DEP8). Per decoder layer after the
first, every weight storage of at least ``MIN_BYTES`` behind a parameter of the layer
(attention and KDA projections, shared expert, latent projections, router, in any
dtype or quantization: weights and their scales alike; the routed experts live in
the MegaMoE) is packed into one byte blob, cut into one 2 MiB-aligned shard per
member. Each layer's blob is one fixed VA range: this member's shard maps its own
fabric memory, the others map staging slot ``ordinal % depth``. Every tensor object
of the layer that views a moved storage (parameters, packed holders, the fused
prefill GEMMs' weight views) is re-pointed in place (``Tensor.set_``) at the same
offset and strides inside that range, so its pointer never changes afterwards. The
old storage must then be unreferenced: a moved storage that something outside the
walk still holds raises.

A layer's shards are copied (one copy-engine read per member, on the expert DWDP
streams if any) when the layer ``depth - 1`` before it starts; the slot they overwrite
was last read by a layer that has fully finished, including its shared expert's side
stream.

Selection is by storage only, so a new checkpoint format needs no change here. The
weights are sharded at construction (``shard_layer``, right after each decoder layer
is built), so the full non-expert copy is never allocated next to the experts. The
loader writes every member's rows through the parameters: this member's land in its
shard, the others in a staging slot (``loaded`` then fills those with NaN bytes).
Post-load processing needs whole tensors (scale swizzles, row interleaves, packing,
kernel layouts), so it runs per layer in a load window (``install``), exactly as a
forward sees the layer: the layer's full blob is fetched into its staging slot, and
the modules whose processing reads a moved storage (``defer_weight_processing``:
vLLM's ``process_weights_after_loading`` skips them) and the model's per-layer
transforms run on it. A transform may rewrite a moved storage in place (every member
computes the same bytes, so each home shard stays right) or replace it: a
replacement that fits goes back into its construction region, any other new storage
goes into the layer's load-time blob of the same layout. Peak memory is one layer.
Linears whose processing merges several of them into a new storage
(``dwdp_resident_linears``: the DSA packed groups, MLA's ``kv_b_proj``) stay whole
until their layer's window, so no dead region is copied every forward.

The vocabulary tables (embedding, LM head; ``shard_table``) are sharded the same way
but read in place: from ``install`` on, the other members' shards map those members'
memory, as the embedding gathers few rows and the prefill LM head reads its weight
once per step. Until then they map one shared load-time stand-in.

Shared-expert selector (``configure_shared``; a P/D prefill worker under sequence
parallelism, TP > 1, whose shared experts are replicated): the members are the EP
ranks, and per decoder layer only its shared expert's storages are sharded (the
other weights are TP-sharded, i.e. differ between members, or stay resident).
Every module owning a selected parameter must be a replicated linear (``tp_size``
1): any other raises, so a TP-sharded storage is never selected. A layer's copies
start at the previous-but-one layer's MegaMoE launch (``moe_start``), under the
MegaMoE kernel, instead of at its start, where the attention's SP all-gather runs.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.multiprocessing.reductions import StorageWeakRef

from vllm.logger import init_logger
from vllm.model_executor.layers.attention import is_deferred_attention_layer
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

from .megamoe_rubin import dwdp as edwdp
from .megamoe_rubin.symm_arena import aligned

logger = init_logger(__name__)

DEPTH = 3  # staging slots: copies start two layers ahead
MIN_BYTES = 1 << 20
DEVICE_TYPE = "cuda"  # storages on other devices (host copies) are not moved

# Shared-expert selector (configure_shared): decoder layer -> the module whose
# storages are sharded (its replicated shared expert) or None; None = whole layers.
_SELECT: Callable[[nn.Module], nn.Module | None] | None = None


def configure_shared(select: Callable[[nn.Module], nn.Module | None] | None) -> None:
    """Model init: shard only what ``select`` returns per decoder layer (its
    replicated shared expert), or nothing extra (None). Exclusive with
    ``ParallelConfig.dwdp`` (whole layers, DEP engines only)."""
    global _SELECT
    if select is not None and edwdp.dense_enabled():
        raise ValueError(
            "Shared-expert DWDP (TP > 1 SP prefill worker) and dwdp="
            "'dense'/'experts' (DEP engines) are exclusive."
        )
    _SELECT = select


def _enabled() -> bool:
    return edwdp.dense_enabled() or _SELECT is not None


def _root(layer: nn.Module, layer_idx: int) -> nn.Module | None:
    """The module of decoder layer ``layer_idx`` whose storages are sharded: the
    whole layer, or under the shared-expert selector its replicated shared expert
    (None: nothing)."""
    if _SELECT is None:
        return layer
    root = _SELECT(layer)
    if root is not None:
        _check_replicated(root, layer_idx)
    return root


def _check_replicated(root: nn.Module, layer_idx: int) -> None:
    """Every module of ``root`` owning a parameter of at least ``MIN_BYTES`` (or not
    yet materialized) is a replicated linear: byte-identical on every member, so
    any member's shard of it is everyone's. A TP-sharded storage raises."""
    from vllm.model_executor.layers.linear import LinearBase

    for name, module in root.named_modules():
        owned = [
            p for p in module.parameters(recurse=False)
            if p.is_meta or p.untyped_storage().nbytes() >= MIN_BYTES
        ]  # fmt: skip
        if owned and not (
            isinstance(module, LinearBase) and getattr(module, "tp_size", None) == 1
        ):
            raise RuntimeError(
                f"Shared-expert DWDP, layer {layer_idx}: {name or type(root).__name__} "
                f"({type(module).__name__}, tp_size "
                f"{getattr(module, 'tp_size', None)}) is not a replicated linear; "
                "only weights identical on every EP rank can be sharded."
            )


class DenseDwdp:
    def __init__(
        self,
        layers: list[nn.Module],
        start: int,
        x: edwdp.ExpertDwdp | None,
        built: _Built | None,
        prepare: Callable[[nn.Module], None],
        dtype: torch.dtype,
    ):
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
        store = edwdp.Store("dense", n, me) if x is not None else edwdp.GroupStore(g)

        built_layers = {} if built is None else built.layers
        if built is not None:
            assert built.start == start and (built.n, built.me) == (n, me)
        # Every member has loaded its home shards: the construction blobs' handles
        # (the vocabulary tables map the other members' memory from here on).
        c_handles = store.exchange(
            {} if built is None else built.exports, "construction"
        )
        if built is not None:
            built.exports.clear()
            built.map_tables(c_handles)
        c_pairs = {
            idx: _peer_pairs(blob, shard, n, me, c_handles, f"c{idx}", device)
            for idx, (blob, shard, _) in built_layers.items()
        }
        stops = _layer_stops(layers)
        # what each layer shards: itself, or its shared expert (shared-expert selector)
        roots = [_root(layer, start + i) for i, layer in enumerate(layers)]
        self.issue_at_moe = _SELECT is not None
        self._ordinal: int | None = None
        self._eager_issue = False
        shared = _shared_storages(
            [r for r in roots if r is not None],
            [s for r, s in zip(roots, stops) if r is not None],
        )
        skip = {b.untyped_storage().data_ptr() for b, _, _ in built_layers.values()}
        # Per layer, in order: its load window (module docstring), then its storages
        # that are not in its construction blob go to a load-time blob of the same
        # layout. The fabric memory comes from the driver, while the storages it
        # replaces were allocated through PyTorch's caching allocator: each layer's
        # replaced bytes go back to the driver before the next layer's home shard is
        # created; the load-time staging slots are created last, from that memory.
        torch.accelerator.empty_cache()
        exports: dict[str, bytes] = {}
        blobs: dict[int, tuple] = {}  # ordinal -> (range, blob, shard)
        rehomed = dead = 0
        for ordinal, layer in enumerate(layers):
            idx = start + ordinal
            if idx in c_pairs:
                copies = edwdp.Copies(c_pairs[idx], len(self.streams))
                copies.fetch(self.streams, self.handles)
                copies.wait()
                torch.cuda.synchronize(device)
                # Every member holds the layer as loaded before any transforms
                # its home shard in place.
                store.barrier(f"window{idx}")
            _process_deferred(layer, stops[ordinal], dtype)
            prepare(layer)
            torch.cuda.synchronize(device)
            root = roots[ordinal]
            if ordinal == 0 or root is None:  # the first layer stays resident
                continue
            if idx in built_layers:
                r, d = _rehome(root, built_layers[idx], stops[ordinal])
                rehomed, dead = rehomed + r, dead + d
            items, off = _plan_layer(root, stops[ordinal], skip, shared)
            shard = _shard_bytes(items, off, n, chunk)
            if shard:
                blobs[ordinal] = self._move(items, shard, n, me, device, ordinal)
                exports[str(ordinal)] = edwdp.export(blobs[ordinal][3])
            del items
            torch.accelerator.empty_cache()
        # slots[(k, m, shard)]: staging memory for member m's shard in slot k (a
        # mapping covers a whole allocation, so each shard size has its own)
        slots: dict[tuple[int, int, int], object] = {}
        for ordinal, (rng, _, shard, _) in blobs.items():
            for m in range(n):
                if m != me:
                    key = (ordinal % DEPTH, m, shard)
                    if key not in slots:
                        slots[key] = edwdp.create(shard, device)
                    rng.map(m * shard, shard, slots[key])

        handles = store.exchange(exports, "load")
        # copies[i]: layer start + i (the first layer is resident: no copies), from
        # its construction blob and its load-time blob
        self.copies = [edwdp.Copies([], len(self.streams))]
        moved = 0
        for ordinal in range(1, len(layers)):
            pairs = list(c_pairs.get(start + ordinal, []))
            if ordinal in blobs:
                _, blob, shard, _ = blobs[ordinal]
                pairs += _peer_pairs(blob, shard, n, me, handles, str(ordinal), device)
            moved += sum(dst.numel() for dst, _ in pairs)
            self.copies.append(edwdp.Copies(pairs, len(self.streams)))
        store.barrier("ready")
        resident = sum(st.nbytes() for st in shared.values())
        free, total = torch.cuda.mem_get_info(device)
        logger.info(
            "Dense DWDP%s: member %d/%d, %d layers (%d sharded at construction, "
            "%d with a load-time blob), %.2f GB/layer copied, %d replaced weight "
            "storages back in their construction shards, %.2f GiB of construction "
            "shards left unused, %.2f GiB of weight storages shared across layers "
            "resident; torch allocated %.2f GiB, device free %.2f / %.2f GiB",
            " (shared-expert selector, copies at the MegaMoE launch)"
            if self.issue_at_moe
            else "",
            me,
            n,
            len(layers),
            len(built_layers),
            len(blobs),
            moved / max(len(layers) - 1, 1) / 1e9,
            rehomed,
            dead / 2**30,
            resident / 2**30,
            torch.cuda.memory_allocated(device) / 2**30,
            free / 2**30,
            total / 2**30,
        )

    @staticmethod
    def _move(items, shard: int, n: int, me: int, device: int, ordinal: int):
        """Move ``items`` (one layer's storages) into a new blob whose shard ``me``
        is this member's fabric memory (the others are mapped later); returns
        ``(range, blob, shard, home)``."""
        rng = edwdp.VaRange(shard * n, device)
        home = edwdp.create(shard, device)
        rng.map(me * shard, shard, home)
        blob = rng.tensor()
        lo, hi = me * shard, (me + 1) * shard
        mine = blob[lo:hi]
        mine.zero_()
        refs = []
        storage = views = None
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
        items.clear()
        del storage, views
        torch.cuda.synchronize(device)
        leaked = [r for r in refs if not r[0].expired()]
        if leaked:
            raise RuntimeError(
                f"Dense DWDP layer {ordinal}: {len(leaked)} moved weight storage(s) "
                "are still referenced outside the layer's modules (e.g. "
                f"{[(nb, shape, str(dt)) for _, nb, shape, dt in leaked[:4]]}); "
                "their old copy would stay resident."
            )
        return rng, blob, shard, home

    def _fetch(self, i: int) -> None:
        if i < len(self.copies):
            self.copies[i].fetch(self.streams, self.handles)

    def layer_start(self, layer_idx: int) -> None:
        """Before a decoder layer: wait for its shards, then copy those of the layer
        ``depth - 1`` ahead (and, at the first layer, the ones in between). With
        ``issue_at_moe`` those copies start at this layer's MegaMoE launch
        (``moe_start``) instead; here only the next layer's, if the previous layer
        launched no MegaMoE (a no-op otherwise: they are in flight).

        Under a breakable CUDA graph capture (graph segments end at every attention
        op, which runs eagerly) a copy may not span two segments: the wait and the
        copies of the layer ``depth - 1`` ahead then run in an eager break of their
        own at the layer start, replayed every step."""
        capture = _breakable_capture()
        self._eager_issue = capture is not None
        if capture is not None:
            capture.add_eager(lambda: self._start(layer_idx, at_moe=False))
        else:
            self._start(layer_idx, self.issue_at_moe)

    def _start(self, layer_idx: int, at_moe: bool) -> None:
        i = layer_idx - self.start
        self._ordinal = i
        if i == 0:
            for j in range(1, DEPTH - 1):
                self._fetch(j)
        self.copies[i].wait()
        self._fetch(i + DEPTH - (2 if at_moe and i else 1))

    def moe_start(self) -> None:
        """At the current layer's MegaMoE launch (main stream, before its stager):
        copy the shards of the layer ``depth - 1`` ahead. Their slot was last read by
        the previous layer, whose shared-expert side stream has joined the main
        stream before this point; the copies run under the MegaMoE kernel, away from
        the next layer's SP all-gather. A no-op without ``issue_at_moe``, outside a
        forward, under a breakable capture (``layer_start`` issued them) and when
        already in flight (sub-call launches)."""
        if self.issue_at_moe and self._ordinal is not None and not self._eager_issue:
            self._fetch(self._ordinal + DEPTH - 1)


def _breakable_capture():
    """The active breakable CUDA graph capture that ends a segment at each eager
    op (not a FULL-mode capture, whose graph spans the whole forward), or None."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    capture = BreakableCUDAGraphCapture.current()
    if capture is None or not capture._capturing:
        return None
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if (
        is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
    ):
        return None
    return capture


def _peer_pairs(blob, shard: int, n: int, me: int, handles, key: str, device: int):
    """``(dst, src)`` copies of the other members' shards of ``blob`` (their home
    memory, ``handles[m][key]``) into its staging part."""
    pairs = []
    for m in range(n):
        if m != me:
            r = edwdp.VaRange(shard, device)
            r.map_imported(0, shard, handles[m][key])
            pairs.append((blob[m * shard : (m + 1) * shard], r.tensor()))
    return pairs


def _shard_bytes(items: list, off: int, n: int, chunk: int) -> int:
    """One member's shard of a layer's load-time blob of ``off`` bytes; 0 when the
    layer has nothing left to shard (every weight was sharded at construction), so
    no blob is created."""
    return aligned(-(-off // n), chunk) if items else 0


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
        base = obj._base  # a view's base holds the old storage too
        if base is not None:
            _walk(base, out, seen, stop, depth + 1)
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


def _layer_stops(layers: list[nn.Module]) -> list[set[int]]:
    """Per layer: the module ids its walk must not enter (the other decoder layers,
    the MegaMoE experts: their weights are the expert DWDP's or stay resident)."""
    from .megamoe_rubin.experts import KimiK3MegaMoEExperts

    experts = {
        id(m) for layer in layers for m in layer.modules()
        if isinstance(m, KimiK3MegaMoEExperts)
    }  # fmt: skip
    ids = {id(layer) for layer in layers}
    return [(ids - {id(layer)}) | experts for layer in layers]


def _groups(layer: nn.Module, stop: set[int], skip=frozenset()) -> dict[int, list]:
    """The CUDA storages the layer's tensor objects view, outside ``skip``:
    ``{data_ptr: [storage, [views], has a parameter]}``."""
    tensors: list[torch.Tensor] = []
    _walk(layer, tensors, set(), stop)
    groups: dict[int, list] = {}
    for t in tensors:
        if t.device.type != DEVICE_TYPE:
            continue
        st = t.untyped_storage()
        if st.data_ptr() in skip:
            continue
        g = groups.setdefault(st.data_ptr(), [st, [], False])
        if all(t is not v for v in g[1]):
            g[1].append(t)
        g[2] |= isinstance(t, nn.Parameter)
    return groups


def _shared_storages(
    layers: list[nn.Module], stops: list[set[int]]
) -> dict[int, torch.UntypedStorage]:
    """Weight storages of at least ``MIN_BYTES`` that several layers view (before
    the load windows): they stay resident."""
    seen: dict[int, torch.UntypedStorage] = {}
    shared: dict[int, torch.UntypedStorage] = {}
    for layer, stop in zip(layers, stops):
        for key, (st, _, has_param) in _groups(layer, stop).items():
            if has_param and st.nbytes() >= MIN_BYTES:
                if key in seen:
                    shared[key] = st
                seen[key] = st
    return shared


def _plan_layer(layer: nn.Module, stop: set[int], skip: set[int], shared: dict):
    """``(items, bytes)``: the layer's weight storages of at least ``MIN_BYTES``
    outside the construction blobs (``skip``) and ``shared``, each with the tensor
    objects viewing it and its offset in the layer's load-time blob."""
    items, off = [], 0
    for key, (st, views, has_param) in _groups(layer, stop, skip).items():
        if has_param and st.nbytes() >= MIN_BYTES and key not in shared:
            items.append((st, views, off))
            off = aligned(off + st.nbytes())
    return items, off


def _byte_range(t: torch.Tensor) -> tuple[int, int]:
    """The bytes of its storage ``t`` spans."""
    size = t.element_size()
    lo = t.storage_offset() * size
    if t.numel() == 0:
        return lo, lo
    extent = sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1
    return lo, lo + extent * size


def _rehome(layer: nn.Module, entry: tuple, stop: set[int]) -> tuple[int, int]:
    """After the layer's load window: a construction-sharded parameter whose
    transform moved it to a new storage (e.g. a die-localized or re-swizzled copy)
    goes back into its construction region when it fits and nothing else views
    that region any more; every tensor object viewing the new storage is
    re-pointed there. Returns (storages moved back, bytes of regions left
    unused)."""
    blob, _, regions = entry
    target = blob.untyped_storage()
    base = blob.storage_offset()
    groups = _groups(layer, stop)
    in_use = [_byte_range(t) for t in groups.get(target.data_ptr(), [None, []])[1]]
    moved: dict[int, list] = {}  # new storage -> its regions
    new = None
    for p, off, nbytes in regions:
        if p.untyped_storage().data_ptr() != target.data_ptr():
            new = moved.setdefault(p.untyped_storage().data_ptr(), [None, []])
            new[0] = p.untyped_storage()
            new[1].append((off, nbytes))
    rehomed = dead = 0
    refs = []
    del new
    for key in list(moved):
        st, spans = moved.pop(key)
        lo, nbytes = spans[0][0] + base, spans[0][1]
        free = not any(a < lo + nbytes and lo < b for a, b in in_use)
        if len(spans) != 1 or st.nbytes() > nbytes or not free or key not in groups:
            dead += sum(nb for _, nb in spans) if free else 0
            del st
            continue
        views = groups.pop(key)[1]
        _storage_u8(target)[lo : lo + st.nbytes()].copy_(_storage_u8(st))
        with torch.no_grad():
            for t in views:
                _repoint(t, target, lo // t.element_size())
        refs.append(StorageWeakRef(st))
        del st, views
        rehomed += 1
    del groups, moved
    if any(not ref.expired() for ref in refs):
        raise RuntimeError(
            "Dense DWDP: a transformed weight storage moved back into its "
            "construction shard is still referenced outside the layer."
        )
    return rehomed, dead


def _modules(layer: nn.Module, stop: set[int]):
    """``layer.modules()`` (each module once, also when several parents hold it)
    without the subtrees of the modules in ``stop``."""
    todo, seen = [layer], set()
    while todo:
        module = todo.pop()
        if id(module) in stop or id(module) in seen:
            continue
        seen.add(id(module))
        yield module
        todo.extend(reversed(list(module.children())))


def _defer(layer: nn.Module, stop: set[int], blob: int) -> None:
    """Mark the modules whose post-load processing reads a storage that moved into
    the construction blob (data pointer ``blob``): vLLM's
    ``process_weights_after_loading`` skips them (``defer_weight_processing``) and
    ``DenseDwdp`` runs it in the layer's load window, on the whole layer."""

    def reads(module: nn.Module, recurse: bool) -> bool:
        return any(
            p.untyped_storage().data_ptr() == blob
            for p in module.parameters(recurse=recurse)
            if not p.is_meta
        )

    for module in _modules(layer, stop):
        method = getattr(module, "quant_method", None)
        quantized = isinstance(method, QuantizeMethodBase)
        if (quantized and reads(module, False)) or (
            is_deferred_attention_layer(module) and reads(module, True)
        ):
            module.defer_weight_processing = True


def _process_deferred(layer: nn.Module, stop: set[int], dtype: torch.dtype) -> None:
    """In the layer's load window: the post-load processing ``_defer`` held back,
    in vLLM's order (quantization methods, then attention-like modules at the model
    dtype)."""
    from vllm.model_executor.utils import is_weights_pre_processed

    deferred = [
        m for m in _modules(layer, stop) if getattr(m, "defer_weight_processing", False)
    ]
    if deferred and is_weights_pre_processed():
        raise RuntimeError("Dense DWDP does not support pre-processed weights.")
    for module in deferred:
        method = getattr(module, "quant_method", None)
        if isinstance(method, QuantizeMethodBase):
            method.process_weights_after_loading(module)
            if hasattr(module, "update_param_tp_status"):
                module.update_param_tp_status()
    for module in deferred:
        if is_deferred_attention_layer(module):
            module.process_weights_after_loading(dtype)
    for module in deferred:
        module.defer_weight_processing = False


class _Built:
    """The construction-time shards (module docstring): per decoder layer after the
    first, its weight storages in one blob laid out as ``DenseDwdp``'s
    (this member's shard is fabric memory, the others map staging slot
    ``ordinal % DEPTH``), each tensor object viewing them re-pointed in place."""

    def __init__(self) -> None:
        from vllm.distributed import get_ep_group

        g = get_ep_group()
        x = edwdp.get()
        if x is not None:  # both engines' EP ranks
            self.n, self.me = 2 * g.world_size, x.index * g.world_size + g.rank_in_group
        else:  # this engine's EP ranks
            self.n, self.me = g.world_size, g.rank_in_group
        self.device = torch.accelerator.current_device_index()
        self.start: int | None = None  # the first decoder layer (resident)
        self.slots: dict[tuple[int, int, int], object] = {}
        self.exports: dict[str, bytes] = {}
        # layer index -> (blob, shard bytes, [(parameter, region offset, bytes)])
        self.layers: dict[int, tuple[torch.Tensor, int, list]] = {}
        self.resident = 0  # dwdp_resident_linears bytes kept whole until install
        self.fabric = 0  # fabric bytes created: home shards and staging slots
        # vocabulary tables: (key, range, shard, parameter, data_ptr); their load-time
        # stand-in per shard size
        self.tables: list[tuple] = []
        self.stand_ins: dict[int, object] = {}

    def poison(self) -> None:
        """``loaded``: NaN bytes in every construction blob's staging part."""
        for blob, shard, _ in self.layers.values():
            for m in range(self.n):
                if m != self.me:
                    blob[m * shard : (m + 1) * shard].fill_(0xFF)

    def shard_table(self, param: torch.Tensor) -> None:
        """A vocabulary table read in place (module docstring)."""
        if param.is_meta or any(
            param.untyped_storage().data_ptr() == p for *_, p in self.tables
        ):
            return  # materialized at load, or tied to a table already sharded
        n, me = self.n, self.me
        shard = aligned(-(-param.nbytes // n), edwdp._dies().CHUNK)
        key = f"t{len(self.tables)}"
        rng = edwdp.VaRange(shard * n, self.device)
        home = edwdp.create(shard, self.device)
        self.exports[key] = edwdp.export(home)
        self.fabric += shard
        if shard not in self.stand_ins:
            self.stand_ins[shard] = edwdp.create(shard, self.device)
            self.fabric += shard
        for m in range(n):
            rng.map(m * shard, shard, home if m == me else self.stand_ins[shard])
        blob = rng.tensor()
        lo, hi = me * shard, (me + 1) * shard
        blob[lo:hi].zero_()
        src = _storage_u8(param.untyped_storage())
        a, b = lo, min(src.numel(), hi)
        if a < b:
            blob[a:b].copy_(src[a:b])
        del src
        old = StorageWeakRef(param.untyped_storage())
        with torch.no_grad():
            _repoint(
                param,
                blob.untyped_storage(),
                blob.storage_offset() // param.element_size(),
            )
        if not old.expired():
            raise RuntimeError(
                "Dense DWDP: a vocabulary table is still referenced outside its module."
            )
        self.tables.append((key, rng, shard, param, param.data_ptr()))
        torch.accelerator.empty_cache()

    def map_tables(self, handles: list[dict[str, bytes]]) -> None:
        """At install: the other members' table shards map their memory."""
        for key, rng, shard, param, ptr in self.tables:
            if param.data_ptr() != ptr:
                raise RuntimeError(
                    f"Dense DWDP: table {key} was replaced after loading."
                )
            for m in range(self.n):
                if m != self.me:
                    rng.unmap(m * shard, shard)
                    rng.map_imported(m * shard, shard, handles[m][key])
        for mem in self.stand_ins.values():
            edwdp.release(mem)
        self.stand_ins.clear()

    def _eligible(
        self, layer: nn.Module, stop: set[int]
    ) -> dict[int, torch.UntypedStorage]:
        """Storages of at least ``MIN_BYTES`` behind the layer's parameters (any
        module, dtype or quantization; not the MegaMoE experts in ``stop``) that no
        ``dwdp_resident_linears`` module's parameter shares."""
        keep = {
            id(m)
            for sub in layer.modules()
            for m in getattr(sub, "dwdp_resident_linears", lambda: [])()
        }
        eligible: dict[int, torch.UntypedStorage] = {}
        shared: set[int] = set()
        for module in _modules(layer, stop):
            for p in module.parameters(recurse=False):
                if p.is_meta:  # materialized at load (online quantization)
                    continue
                st = p.untyped_storage()
                if st.nbytes() < MIN_BYTES:
                    continue
                if id(module) in keep:
                    shared.add(st.data_ptr())
                    self.resident += st.nbytes()
                else:
                    eligible[st.data_ptr()] = st
        return {k: st for k, st in eligible.items() if k not in shared}

    def shard(self, layer: nn.Module, layer_idx: int, stop: set[int]) -> None:
        assert self.start is not None
        eligible = self._eligible(layer, stop)
        if not eligible:
            return
        tensors: list[torch.Tensor] = []
        _walk(layer, tensors, set(), stop)
        views: dict[int, list[torch.Tensor]] = {key: [] for key in eligible}
        for t in tensors:
            key = None if t.is_meta else t.untyped_storage().data_ptr()
            if key in views and all(t is not v for v in views[key]):
                views[key].append(t)
        items, off = [], 0
        for key, st in eligible.items():
            items.append((st, views[key], off))
            off = aligned(off + st.nbytes())
        del eligible, views, tensors
        n, me = self.n, self.me
        shard = aligned(-(-off // n), edwdp._dies().CHUNK)
        ordinal = layer_idx - self.start
        rng = edwdp.VaRange(shard * n, self.device)
        home = edwdp.create(shard, self.device)
        self.exports[f"c{layer_idx}"] = edwdp.export(home)
        self.fabric += shard
        for m in range(n):
            if m == me:
                rng.map(m * shard, shard, home)
                continue
            key = (ordinal % DEPTH, m, shard)
            if key not in self.slots:
                self.slots[key] = edwdp.create(shard, self.device)
                self.fabric += shard
            rng.map(m * shard, shard, self.slots[key])
        blob = rng.tensor()
        lo, hi = me * shard, (me + 1) * shard
        mine = blob[lo:hi]
        mine.zero_()
        target, base = blob.untyped_storage(), blob.storage_offset()
        regions, refs = [], []
        with torch.no_grad():
            for st, ts, off in items:
                # this member's rows keep what construction wrote (e.g. padding)
                src = _storage_u8(st)
                a, b = max(off, lo), min(off + st.nbytes(), hi)
                if a < b:
                    mine[a - lo : b - lo].copy_(src[a - off : b - off])
                del src
                for t in ts:
                    _repoint(t, target, (base + off) // t.element_size())
                    if isinstance(t, nn.Parameter):
                        regions.append((t, off, st.nbytes()))
                refs.append((StorageWeakRef(st), st.nbytes()))
        del items, st, ts
        leaked = [nb for ref, nb in refs if not ref.expired()]
        if leaked:
            raise RuntimeError(
                f"Dense DWDP construction, layer {layer_idx}: {len(leaked)} "
                f"weight storage(s) ({leaked[:4]} bytes) are still referenced outside "
                "the layer's modules; their full copy would stay resident."
            )
        _defer(layer, stop, target.data_ptr())
        self.layers[layer_idx] = (blob, shard, regions)
        # Hand the replaced bytes back (also from the cumem pool) before the next
        # layer allocates its own.
        torch.accelerator.empty_cache()


_B: _Built | None = None
_D: DenseDwdp | None = None


def shard_layer(layer: nn.Module, layer_idx: int) -> None:
    """At construction, right after decoder layer ``layer_idx`` is built (in layer
    order): the first layer stays resident, the others' BF16 linear weights are
    sharded (``ParallelConfig.dwdp``)."""
    global _B
    if not _enabled():
        return
    if _B is None:
        _B = _Built()
    if _B.start is None:
        _B.start = layer_idx
        return
    root = _root(layer, layer_idx)
    if root is None:
        return
    from .megamoe_rubin.experts import KimiK3MegaMoEExperts

    stop = {id(m) for m in layer.modules() if isinstance(m, KimiK3MegaMoEExperts)}
    _B.shard(root, layer_idx, stop)


def shard_table(param: torch.Tensor) -> None:
    """At construction: the embedding or LM head weight (``ParallelConfig.dwdp``)."""
    global _B
    if not edwdp.dense_enabled():
        return
    if _B is None:
        _B = _Built()
    _B.shard_table(param)


def log_construction() -> None:
    """After the decoder layers are built: what construction sharded and the memory
    the process holds."""
    if _B is None:
        return
    usage = None
    try:
        from vllm.device_allocator.cumem import CuMemAllocator

        if CuMemAllocator.instance is not None:
            usage = CuMemAllocator.instance.get_current_usage()
    except ImportError:
        pass
    free, total = torch.cuda.mem_get_info(_B.device)
    logger.info(
        "Dense DWDP construction%s: member %d/%d, %d layers and %d vocabulary tables "
        "sharded, %.2f GiB fabric (home shards + staging slots + table stand-in), "
        "%.2f GiB dwdp_resident_linears weights resident until install; "
        "torch allocated %.2f / reserved %.2f GiB, cumem pool %s GiB, device free "
        "%.2f / %.2f GiB",
        " (shared-expert selector: replicated shared experts only)"
        if _SELECT is not None
        else "",
        _B.me,
        _B.n,
        len(_B.layers),
        len(_B.tables),
        _B.fabric / 2**30,
        _B.resident / 2**30,
        torch.cuda.memory_allocated(_B.device) / 2**30,
        torch.cuda.memory_reserved(_B.device) / 2**30,
        "n/a" if usage is None else f"{usage / 2**30:.2f}",
        free / 2**30,
        total / 2**30,
    )


def install(
    layers: nn.ModuleList,
    start: int,
    end: int,
    prepare: Callable[[nn.Module], None],
    dtype: torch.dtype,
) -> None:
    """At load, after vLLM's post-load processing and the expert DWDP (if any):
    ``prepare`` (the model's per-layer weight transforms) on every decoder layer of
    ``layers[start:end]`` in order. Under the non-expert DWDP each layer runs in
    its load window and its storages are then sharded (the first stays
    resident)."""
    global _D
    if not _enabled():
        for layer in layers[start:end]:
            prepare(layer)
    elif _D is None:
        _D = DenseDwdp(list(layers[start:end]), start, edwdp.get(), _B, prepare, dtype)


def loaded() -> None:
    """After the checkpoint is loaded, before any post-load processing: the
    staging parts of the construction blobs (the other members' rows, which the
    loader wrote there and later layers overwrote) are filled with 0xFF (NaN in
    BF16, E4M3 and E8M0), so a transform that reads a sharded weight outside its
    layer's load window yields NaNs, not plausible values."""
    if _B is not None:
        _B.poison()


def get() -> DenseDwdp | None:
    return _D
