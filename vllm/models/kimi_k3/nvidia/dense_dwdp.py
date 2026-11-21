# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DWDP of Kimi K3's non-expert weights.

Members: every EP rank of the two engines that share experts (e.g. 2 x 4 = 8), or of
this engine alone with ``dwdp="dense"`` (e.g. DEP8). Per
decoder layer after the first, the parameters of at least ``MIN_BYTES`` that own their
storage (attention and KDA projections, shared expert, latent projections, router; the
routed experts live in the MegaMoE) are packed into one byte blob, cut into one
2 MiB-aligned shard per member. Each layer's blob is one fixed VA range: this member's
shard maps its own fabric memory, the others map staging slot ``ordinal % depth``. The
parameters become views of that range, so their pointers never change. Parameters that
are views of a packed projection (``pack_projections``) stay resident.

A layer's shards are copied (one copy-engine read per member, on the expert DWDP
streams if any) when the layer ``depth - 1`` before it starts; the slot they overwrite
was last read by a layer that has fully finished, including its shared expert's side
stream. Installed at load, after the projections are packed (``ParallelConfig.dwdp``).
"""

from __future__ import annotations

import torch
from torch import nn

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

        plans, resident = [], 0
        for layer in layers[1:]:
            items, off = [], 0
            for p in layer.parameters():
                if p.nbytes < MIN_BYTES:
                    continue
                owned = (  # not a view of a packed projection
                    p.storage_offset() == 0 and p.untyped_storage().nbytes() == p.nbytes
                )
                if not (p.is_contiguous() and owned):
                    resident += p.nbytes
                    continue
                items.append((p, off))
                off = aligned(off + p.nbytes)
            plans.append((items, aligned(-(-off // n), chunk)))
        # slots[(k, m, shard)]: staging memory for member m's shard in slot k (a
        # mapping covers a whole allocation, so each shard size has its own)
        slots: dict[tuple[int, int, int], object] = {}
        exports: dict[str, bytes] = {}
        blobs = []
        for ordinal, (items, shard) in enumerate(plans, start=1):
            rng = edwdp.VaRange(shard * n, device)
            home = edwdp.create(shard, device)
            exports[str(ordinal)] = edwdp.export(home)
            for m in range(n):
                key = (ordinal % DEPTH, m, shard)
                if m != me and key not in slots:
                    slots[key] = edwdp.create(shard, device)
                rng.map(m * shard, shard, home if m == me else slots[key])
            blob = rng.tensor()
            lo, hi = me * shard, (me + 1) * shard
            mine = blob[lo:hi]
            mine.zero_()
            for p, off in items:  # this member's rows of the packed parameters
                a, b = max(off, lo), min(off + p.nbytes, hi)
                if a < b:
                    mine[a - lo : b - lo].copy_(
                        edwdp.storage_bytes(p.data)[a - off : b - off]
                    )
            for p, off in items:
                p.data = blob[off : off + p.nbytes].view(p.dtype).view(p.shape)
            blobs.append((blob, shard))
        torch.accelerator.empty_cache()

        store = edwdp.Store("dense", n, me) if x is not None else edwdp.GroupStore(g)
        handles = store.exchange(exports)
        # copies[i]: layer start + i (the first layer is resident: no copies)
        self.copies = [edwdp.Copies([], len(self.streams))]
        for ordinal, (blob, shard) in enumerate(blobs, start=1):
            pairs = []
            for m in range(n):
                if m != me:
                    r = edwdp.VaRange(shard, device)
                    r.map_imported(0, shard, handles[m][str(ordinal)])
                    pairs.append((blob[m * shard : (m + 1) * shard], r.tensor()))
            self.copies.append(edwdp.Copies(pairs, len(self.streams)))
        store.barrier("ready")
        moved = sum(shard * (n - 1) for _, shard in plans) / len(plans)
        logger.info(
            "Dense DWDP: member %d/%d, %d layers, %.2f GB/layer copied, %.2f GiB of "
            "packed projections resident",
            me,
            n,
            len(plans),
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


_D: DenseDwdp | None = None


def install(layers: nn.ModuleList, start: int, end: int) -> None:
    """Shard the non-expert weights of ``layers[start:end]`` (the first stays
    resident), at load after the expert DWDP (if any) finalized."""
    global _D
    if edwdp.dense_enabled() and _D is None:
        _D = DenseDwdp(list(layers[start:end]), start, edwdp.get())


def get() -> DenseDwdp | None:
    return _D
