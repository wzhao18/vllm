# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-local routed-expert streaming for the Rubin MegaMoE on decode servers.

Rubin is two dies, each with its own HBM. Default allocations interleave the dies
every 4 KiB, so a die-oblivious kernel pulls about half of every weight byte over the
die-to-die link: a full-chip read streams at ~13.9 TB/s against ~17.1 TB/s when every
SM reads its own die (ledger row 10, measured on VR200). At C512 decode the MegaMoE is
a pure weight stream (every local expert is touched by ~15 routes), so this ceiling
is its runtime.

Policy (automatic, no flag): on 2-die devices with a trusted SM -> die map, decode
servers whose MegaMoE runs the phase-interleaved inference schedule on one cluster
shape (the shared-expert overlap cap) compile die-local expert streams
(``DieSplitPhaseInterleavedFc12Scheduler``) and home each layer's local experts
``[0, E/2)`` on die 0 and ``[E/2, E)`` on die 1 (``place_experts``). Same bytes,
shapes and strides (EPLB moves rows as before), same tiles and accumulation order:
bitwise-identical outputs. P/D prefill workers keep the stock schedule.

Capability fallback (with a warning, never silent): single-die devices, no trusted
die map, or a die map that streaming contradicts -> die-oblivious; a layer whose
placement cannot be verified keeps its default allocation (the die-local schedule is
still exact on it).
"""

from __future__ import annotations

import functools

import torch

from vllm.logger import init_logger

from .weights import RubinMoeWeights

logger = init_logger(__name__)

# Chunks verified per placed tensor, measurement passes, chunks allowed unclear after
# them (as ``specialized_decode.cutedsl.moe.experts_runtime.place``).
PLACE_SAMPLES, PLACE_PASSES, PLACE_UNCLEAR = 8, 5, 1


def _dies():
    from ..specialized_decode.cutedsl.common import dies

    return dies


@functools.cache
def die_sm_mask32(device: int) -> tuple[int, ...] | None:
    """The die-1 SM bit mask (32-bit words, bit ``s``: SM ``s`` on die 1) of
    ``device``, or None when die-local streaming is not available there."""
    dies = _dies()
    if dies.domain_count(device) < 2:
        return None
    try:
        table = tuple(dies.sm_domains(device))
    except RuntimeError as exc:
        logger.warning(
            "Rubin MegaMoE: no trusted SM -> die map on GPU %d (%s); die-oblivious "
            "routed experts.", device, exc,
        )  # fmt: skip
        return None
    wrong = dies.stream_check(device)
    if wrong:
        logger.warning(
            "Rubin MegaMoE: die map and streaming disagree on SMs %s of GPU %d; "
            "die-oblivious routed experts.", wrong, device,
        )  # fmt: skip
        return None
    words = [0] * (-(-len(table) // 32))
    for sm, die in enumerate(table):
        if die == 1:
            words[sm // 32] |= 1 << (sm % 32)
    return tuple(words)


def current_die_sm_mask32() -> tuple[int, ...] | None:
    """``die_sm_mask32`` of the current CUDA device (None without CUDA)."""
    if not torch.cuda.is_available():
        return None
    return die_sm_mask32(torch.accelerator.current_device_index())


def expert_split(num_local_experts: int) -> int:
    """Local experts ``[0, split)`` are homed on die 0, the rest on die 1 (the
    scheduler's ``die_expert_split``)."""
    return num_local_experts // 2


def chunk_domains(nbytes: int, split_byte: int) -> list[int]:
    """Home die of each 2 MiB chunk of an ``nbytes`` tensor whose bytes below
    ``split_byte`` belong to die 0 (a straddling chunk goes to its majority)."""
    chunk = _dies().CHUNK
    chunks = -(-nbytes // chunk)
    return [0 if k * chunk + chunk // 2 <= split_byte else 1 for k in range(chunks)]


def _place(storage: torch.Tensor, split: int) -> torch.Tensor | None:
    """A copy of the contiguous expert-major ``storage`` (E, ...) homed per expert
    half, verified by measurement; None when the homes cannot be confirmed."""
    dies = _dies()
    experts = storage.shape[0]
    nbytes = storage.numel() * storage.element_size()
    split_byte = nbytes // experts * split
    domains = chunk_domains(nbytes, split_byte)
    raw = dies.localized_empty(nbytes, domains, storage.device)
    copy = raw.view(storage.dtype).view(storage.shape)
    copy.copy_(storage)
    chunks = len(domains)
    sample = tuple(sorted({chunks * i // PLACE_SAMPLES for i in range(PLACE_SAMPLES)}))
    wrong, sms = sample, 4
    for attempt in range(PLACE_PASSES):
        homes = dies.chunk_homes(copy, wrong, sms)
        wrong = tuple(c for c, home in zip(wrong, homes) if home != domains[c])
        if not wrong:
            break
        sms = 8 if attempt < 2 else 16
    if len(wrong) > PLACE_UNCLEAR:
        logger.warning(
            "Rubin MegaMoE: die-localized expert chunks %s of %s not homed as "
            "requested; default allocation kept for this tensor.", wrong, sample,
        )  # fmt: skip
        return None
    return copy


def place_experts(weights: RubinMoeWeights) -> RubinMoeWeights:
    """``weights`` with every expert-major tensor (FC1/FC2 codes and scales) homed
    per die half (``expert_split``); per-expert scalars stay where they are. The
    caller drops the originals and empties the allocator cache."""
    fc1_codes = weights.fc1[0].permute(0, 2, 1)  # K-major physical (E, N, K/2)
    fc2_codes = weights.fc2[0].permute(0, 2, 1)
    tensors = (fc1_codes, weights.fc1[1], fc2_codes, weights.fc2[1])
    for t in tensors:
        if not t.is_contiguous():
            raise AssertionError(
                f"Rubin MegaMoE: expert tensor not contiguous {tuple(t.shape)} "
                f"{tuple(t.stride())}"
            )
    split = expert_split(fc1_codes.shape[0])
    placed = [_place(t, split) for t in tensors]
    placed = [p if p is not None else t for p, t in zip(placed, tensors)]
    return weights._replace(
        fc1=(placed[0].permute(0, 2, 1), placed[1]),
        fc2=(placed[2].permute(0, 2, 1), placed[3]),
    )
