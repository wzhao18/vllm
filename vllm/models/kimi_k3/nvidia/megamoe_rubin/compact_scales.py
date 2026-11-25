# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact NVFP4 routed-expert block scales, read by the MegaMoE kernel itself.

NVFP4 experts whose per-16 E4M3 block scales come in equal pairs (an exact cast of
MXFP4: the Kimi K3.1 ModelOpt export, or this backend's MXFP4 -> NVFP4 conversion)
hold twice the scale bytes they need (K3.1 at EP8: 19.8 GiB per GPU for 9.9). In the
kernel's 128x4-atom swizzle (``weights.to_blocked_batched``) the two per-16 scales of
one per-32 block are adjacent bytes (even, odd) of the flat plane, padding included,
so such a plane is kept as its even bytes (``RubinMoeWeights.compact_sf``). The
refreshed inference kernel (src_gp, the prefill dedup variants) reads them directly:
each CTA bulk-loads its compact bytes of a k tile into the upper half of the stage's
SFA slot and its MMA warp doubles every byte in place before the SMEM -> TMEM copy,
so the MMA sees the full planes' bytes bit for bit (``compact_weight_sf`` in the
mainloop). Kernel math, NVFP4 numerics (activations included) and the per-expert
global scales are unchanged; no expansion kernel, no staging slots, half the scale
bytes read per layer.

Policy (``additional_config["megamoe_compact_scales"]``): ``"auto"`` (default)
compacts on P/D prefill workers only (``pd_prefill_worker``: KV-bound), never on a
decode instance: off for a ``kv_consumer`` and for the specialized decode backend
(``decode_backend: latency_specialized``, also when launched ``kv_both``). ``true`` /
``false`` force it. Compaction needs every bucket on a kernel that reads compact
planes (``session.reads_compact_scales``; the GenPhase and a368c5f66 inference kernels
do not); otherwise the full planes stay, logged. Every MoE layer on every EP rank
compacts, or none (EPLB sizes one buffer per expert tensor from the first layer and
moves rows between ranks): an unequal pair anywhere keeps the full planes, logged.
Not under expert DWDP (its copies move the full planes).
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

from .config import DEDUP_VARIANTS

logger = init_logger(__name__)

CONFIG_KEY = "megamoe_compact_scales"


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


def compact(plane: torch.Tensor, out: torch.Tensor) -> None:
    """The even bytes of an ``(E, n)`` scale plane into ``(E, n / 2)`` ``out``."""
    pairs = plane.view(torch.uint8).view(plane.shape[0], -1, 2)
    out.view(torch.uint8).copy_(pairs[..., 0])


def expand(compact_plane: torch.Tensor) -> torch.Tensor:
    """Every byte of an ``(E, n / 2)`` compact plane twice: the full ``(E, n)`` plane
    (uint8; tests and debugging)."""
    c = compact_plane.view(torch.uint8)
    return c.unsqueeze(-1).expand(*c.shape, 2).reshape(c.shape[0], -1)


def _reads_compact(module) -> bool:
    """Every bucket on the kernel that reads compact planes (the prefill inference
    kernel, 2-CTA M256 tiles, no fused shared expert: session.reads_compact_scales)."""
    return all(
        b.variant in DEDUP_VARIANTS
        and getattr(b, "shared", None) is None
        and b.knobs.mma_tiler_mnk[0] == 256
        for b in module.buckets
    )


def install(modules: list, cpu_group) -> bool:
    """Compact the scale planes of every NVFP4 MoE layer in ``modules`` (module
    order, the kernel weights prepared), or of none; returns whether it did.
    Collective over the EP group; before the runtime warmup (which compiles the
    kernel for the compact planes)."""
    from . import dwdp

    if not modules or not modules[0].compact_scales:
        return False
    if any(m.kernel_weights.quant_kind != "nvfp4" for m in modules):
        return False
    if dwdp.ENABLED:
        logger.info_once(
            "Rubin MegaMoE: full NVFP4 scale planes (expert DWDP copies them)."
        )
        return False
    if not all(_reads_compact(m) for m in modules):
        buckets = ", ".join(f"{b.variant} {b.max_tokens}" for b in modules[0].buckets)
        logger.info_once(
            "Rubin MegaMoE: full NVFP4 scale planes (buckets %s: only the prefill "
            "inference kernel reads compact planes).",
            buckets,
        )
        return False
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
        return False

    (e, n1), (_, n2) = planes[0][0].shape, planes[0][1].shape
    del planes
    device = modules[0].kernel_weights.fc1[1].device
    e4m3 = torch.float8_e4m3fn
    for module in modules:
        w = module.kernel_weights
        # One buffer per layer [FC1 even bytes | FC2 even bytes]; EPLB moves rows of
        # each (E, n / 2) view (get_expert_weights).
        buf = torch.empty(e * (n1 + n2) // 2, dtype=torch.uint8, device=device)
        c1 = buf[: e * n1 // 2].view(e4m3).view(e, n1 // 2)
        c2 = buf[e * n1 // 2 :].view(e4m3).view(e, n2 // 2)
        compact(w.fc1[1], c1)
        compact(w.fc2[1], c2)
        module.kernel_weights = w._replace(
            fc1=(w.fc1[0], c1), fc2=(w.fc2[0], c2), compact_sf=True
        )
        del w  # the full planes
    if device.type == "cuda":
        torch.accelerator.empty_cache()
    full = len(modules) * e * (n1 + n2)
    logger.info_once(
        "Rubin MegaMoE: compact NVFP4 scale planes on %d MoE layers, read by the "
        "kernel: %.2f -> %.2f GiB per GPU.",
        len(modules),
        full / 2**30,
        full / 2 / 2**30,
    )
    return True
