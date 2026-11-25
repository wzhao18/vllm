# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact NVFP4 routed-expert block scales, expanded one MoE layer ahead.

NVFP4 experts whose per-16 E4M3 block scales come in equal pairs (an exact cast of
MXFP4: the Kimi K3.1 ModelOpt export, or this backend's MXFP4 -> NVFP4 conversion)
hold twice the scale bytes they need (K3.1 at EP8: 20.0 GiB per GPU for 10.0). In the
kernel's 128x4-atom swizzle (``weights.to_blocked_batched``) the two per-16 scales of
one per-32 block are adjacent bytes (even, odd) of the flat plane, padding included,
so such a plane is kept as its even bytes and expanded by doubling every byte: the
expanded plane is the kernel's bytes, bit for bit. Kernel, NVFP4 numerics
(activations included) and the per-expert global scales are unchanged.

Layout: each MoE layer keeps one compact buffer ``[FC1 even bytes | FC2 even
bytes]``; MoE layer ``o``'s FC1 / FC2 scale tensors view staging slot ``o % 2``
(``[FC1 plane | FC2 plane]``, fixed addresses: bound launch arguments and CUDA graphs
see ordinary tensors), filled by one launch. The slots use the default allocation
(not die-placed).

Schedule (every forward expands every layer, as a captured graph does): layer 0
expands on the current stream before its kernel; right after layer ``o``'s kernel is
enqueued, layer ``o + 1`` expands on a side stream, after that kernel (its slot was
last read by layer ``o - 1``, so the expansion overlaps the next layer's attention,
not a MegaMoE kernel); the current stream waits for it before layer ``o + 1``'s
kernel.

Policy (``additional_config["megamoe_compact_scales"]``): ``"auto"`` (default)
compacts on P/D prefill workers only (``pd_prefill_worker``: KV-bound, and a
prefill step is long against the expansion), never on a decode instance: off for a
``kv_consumer`` and for the specialized decode backend (``decode_backend:
latency_specialized``, also when launched ``kv_both``), where ~17 us per layer is a
few percent of a decode step. ``true`` / ``false`` force it (``true`` is not
recommended for decode; a ``kv_both`` instance decoding on the stock backend
should pass ``false``). Every MoE layer on
every EP rank compacts, or none (EPLB sizes one buffer per expert tensor from the
first layer and moves rows between ranks): an unequal pair anywhere keeps the full
planes, logged. Not under expert DWDP (its copies move the full planes).
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

CONFIG_KEY = "megamoe_compact_scales"

# One block of 4-byte source words per program, the whole grid at once. Measured on
# VR200 (K3.1 EP8 layer, 117 -> 233 MB): 30.6 us alone (11.4 TB/s); beside 1.46 ms of
# compute-bound BF16 GEMMs on the main stream it adds 19 us, while a persistent grid
# capped at 32-212 programs adds 374-482 us (its resident CTAs hold SMs that the
# GEMMs' waves need for the whole expansion).
_BLOCK_WORDS = 1024


def wanted(vllm_config, prefill_worker: bool) -> bool:
    """Whether the server asks for compact scale planes (``CONFIG_KEY``)."""
    extra = getattr(vllm_config, "additional_config", None)
    extra = extra if isinstance(extra, dict) else {}
    value = extra.get(CONFIG_KEY, "auto")
    if value == "auto":
        kv = getattr(vllm_config, "kv_transfer_config", None)
        return (
            prefill_worker
            and getattr(kv, "kv_role", None) != "kv_consumer"
            and extra.get("decode_backend", "auto") != "latency_specialized"
        )
    if isinstance(value, bool):
        return value
    raise ValueError(
        f'additional_config["{CONFIG_KEY}"] must be "auto", true or false, got '
        f"{value!r}."
    )


def unequal_pairs(plane: torch.Tensor) -> int:
    """Byte pairs (even, odd) of an ``(E, n)`` scale plane that differ."""
    pairs = plane.view(torch.uint8).view(plane.shape[0], -1, 2)
    return int((pairs[..., 0] != pairs[..., 1]).sum())


@triton.jit
def _expand_kernel(src_ptr, dst_ptr, words, BLOCK: tl.constexpr):
    """Dst word i (8 bytes) = src word i (4 bytes) with every byte doubled."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < words
    x = tl.load(src_ptr + offs, mask=mask).to(tl.uint32, bitcast=True).to(tl.uint64)
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF
    x = x | (x << 8)
    tl.store(dst_ptr + offs, x.to(tl.int64, bitcast=True), mask=mask)


def expand(compact: torch.Tensor, out: torch.Tensor) -> None:
    """Every byte of the flat uint8 ``compact`` twice into ``out``, on the current
    stream (Triton on CUDA, torch elsewhere)."""
    n = compact.numel()
    contiguous = compact.is_contiguous() and out.is_contiguous()
    if out.numel() != 2 * n or n % 4 or not contiguous:
        raise ValueError(f"cannot expand {n} bytes into {out.numel()}.")
    if not compact.is_cuda:
        out.view(n, 2).copy_(compact.view(n, 1).expand(n, 2))
        return
    words = n // 4
    _expand_kernel[(triton.cdiv(words, _BLOCK_WORDS),)](
        compact.view(torch.int32), out.view(torch.int64), words, BLOCK=_BLOCK_WORDS
    )


class ScaleStaging:
    """Every MoE layer's compact planes (module order), the two staging slots and the
    per-step expansion state."""

    def __init__(
        self,
        compact: list[torch.Tensor],
        legs: list[tuple[torch.Tensor, torch.Tensor]],
        slots: list[torch.Tensor],
    ):
        self.compact = compact  # flat uint8 per layer
        self.legs = legs  # (FC1, FC2) (E, n/2) views of each compact buffer
        self.slots = slots
        self.held = [-1, -1]  # layer each slot holds (expanded or pending)
        self.pending: int | None = None
        self.stream = None
        if slots[0].is_cuda:
            # High priority: the ~30 us expansion must get SMs at the next wave boundary
            # of the main stream's kernels instead of queueing behind them (measured on the
            # TP8+DCP8 prefill critical path: 92 x ~30 us serialized before MegaMoE).
            self.stream = torch.cuda.Stream(slots[0].device, priority=-1)
            self.go, self.done = torch.cuda.Event(), torch.cuda.Event()

    def _expand(self, o: int) -> None:
        expand(self.compact[o], self.slots[o % 2])
        self.held[o % 2] = o

    def before_kernel(self, o: int) -> None:
        """Before layer ``o``'s kernel: its planes are in its slot, ordered before
        the current stream's next work."""
        if self.pending == o:
            torch.cuda.current_stream().wait_event(self.done)
            self.pending = None
        elif self.held[o % 2] != o:
            self._expand(o)

    def after_kernel(self, o: int) -> None:
        """After layer ``o``'s kernel is enqueued: expand layer ``o + 1``; after the
        last layer, forget the slots (the next forward expands every layer)."""
        nxt = o + 1
        if nxt == len(self.compact):
            self.held = [-1, -1]
            return
        if self.pending == nxt or self.held[nxt % 2] == nxt:
            return
        assert self.pending is None, f"layer {self.pending} was expanded, never used"
        if self.stream is None:
            self._expand(nxt)
            return
        self.go.record()
        self.stream.wait_event(self.go)
        with torch.cuda.stream(self.stream):
            self._expand(nxt)
        self.done.record(self.stream)
        self.pending = nxt


def install(modules: list, cpu_group) -> ScaleStaging | None:
    """Compact the scale planes of every NVFP4 MoE layer in ``modules`` (module
    order, the kernel weights prepared), or of none. Collective over the EP group;
    before the runtime warmup."""
    from . import dwdp

    if not modules or not modules[0].compact_scales:
        return None
    if any(m.kernel_weights.quant_kind != "nvfp4" for m in modules):
        return None
    if dwdp.ENABLED:
        logger.info_once(
            "Rubin MegaMoE: full NVFP4 scale planes (expert DWDP copies them)."
        )
        return None
    planes = [(m.kernel_weights.fc1[1], m.kernel_weights.fc2[1]) for m in modules]
    shapes = {(a.shape, b.shape) for a, b in planes}
    if len(shapes) != 1:
        raise AssertionError(f"Rubin MegaMoE layers disagree on scale shapes {shapes}.")
    bad = [(o, leg, unequal_pairs(p)) for o, pair in enumerate(planes)
           for leg, p in zip(("FC1", "FC2"), pair)]  # fmt: skip
    bad = [b for b in bad if b[2]]
    flag = torch.tensor([len(bad)], dtype=torch.int64)
    torch.distributed.all_reduce(
        flag, op=torch.distributed.ReduceOp.MAX, group=cpu_group
    )
    if int(flag):
        logger.warning(
            "Rubin MegaMoE: full NVFP4 scale planes: per-16 scale pairs differ on "
            "some EP rank (here: %s).",
            ", ".join(f"MoE layer {o} {leg} {n} pairs" for o, leg, n in bad[:4])
            or "none",
        )
        return None

    (e, n1), (_, n2) = planes[0][0].shape, planes[0][1].shape
    del planes
    device = modules[0].kernel_weights.fc1[1].device
    slots = [
        torch.empty(e * (n1 + n2), dtype=torch.uint8, device=device) for _ in range(2)
    ]
    compact, legs = [], []
    for o, module in enumerate(modules):
        w = module.kernel_weights
        buf = torch.empty(e * (n1 + n2) // 2, dtype=torch.uint8, device=device)
        c1 = buf[: e * n1 // 2].view(e, n1 // 2)
        c2 = buf[e * n1 // 2 :].view(e, n2 // 2)
        for dst, plane in ((c1, w.fc1[1]), (c2, w.fc2[1])):
            dst.copy_(plane.view(torch.uint8).view(e, -1, 2)[..., 0])
        del plane
        slot = slots[o % 2]
        e4m3 = torch.float8_e4m3fn
        module.kernel_weights = w._replace(
            fc1=(w.fc1[0], slot[: e * n1].view(e4m3).view(e, n1)),
            fc2=(w.fc2[0], slot[e * n1 :].view(e4m3).view(e, n2)),
        )
        del w  # the full planes
        compact.append(buf)
        legs.append((c1, c2))
    staging = ScaleStaging(compact, legs, slots)
    for o, module in enumerate(modules):
        module.scale_staging, module.moe_ordinal = staging, o
    # Valid planes in both slots for the warmup launches.
    for o in range(min(2, len(modules))):
        staging._expand(o)
    staging.held = [-1, -1]
    if device.type == "cuda":
        torch.accelerator.empty_cache()
    full = len(modules) * e * (n1 + n2)
    logger.info_once(
        "Rubin MegaMoE: compact NVFP4 scale planes on %d MoE layers: %.2f -> %.2f GiB "
        "per GPU (incl. 2 staging slots of %.0f MiB), expanded one layer ahead.",
        len(modules),
        full / 2**30,
        (full / 2 + 2 * e * (n1 + n2)) / 2**30,
        e * (n1 + n2) / 2**20,
    )
    return staging
