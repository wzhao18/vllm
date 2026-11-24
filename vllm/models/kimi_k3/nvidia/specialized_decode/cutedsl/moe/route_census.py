# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MEASUREMENT BUILD ONLY (not for merge): report-only census of the routed experts'
routes per MoE layer call. After every routed-experts call, a few graph-capturable ops
reduce the call's top-k ids to (distinct experts, 8-token units, 16-token units,
experts with more than 8 tokens, token-expert pairs) into a persistent per-(layer,
rows) int32[5]; the bench's run_case reports them per case (``report``). The values
are overwritten every call (graph replays of one case route identically)."""

import json
import os

import torch

_STATS: dict = {}  # (layer ordinal, rows) -> int32[5] device tensor
_ORDER: dict = {}  # id(experts object) -> layer ordinal (creation order)


def record(owner, ids, experts=896):
    """``ids`` [M, top_k] (or flat) int32/int64 expert ids of one routed call."""
    key = id(owner)
    if key not in _ORDER:
        _ORDER[key] = len(_ORDER)
    flat = ids.reshape(-1)
    rows = flat.numel() // 16
    slot = (_ORDER[key], rows)
    stats = _STATS.get(slot)
    if stats is None:
        stats = torch.zeros(5, dtype=torch.int32, device=ids.device)
        _STATS[slot] = stats
    valid = (flat >= 0) & (flat < experts)
    index = torch.where(valid, flat, torch.zeros_like(flat)).long()
    hist = torch.zeros(experts, dtype=torch.int32, device=ids.device)
    hist.scatter_add_(0, index, valid.to(torch.int32))
    values = torch.stack(
        (
            (hist > 0).sum(),
            ((hist + 7) // 8).sum(),
            ((hist + 15) // 16).sum(),
            (hist > 8).sum(),
            hist.sum(),
        )
    ).to(torch.int32)
    stats.copy_(values)


def report(case, rows, out_dir=None, rank=0):
    """The census of ``rows``-row calls: per-layer lists + mean/min/max; logged and,
    with ``out_dir``, written to ``<out_dir>/census.<case>.rank<rank>.json``."""
    slots = sorted(k for k in _STATS if k[1] == rows)
    if not slots:
        print(f"[L6CENSUS] {case} rows {rows}: no routed-experts calls", flush=True)
        return None
    values = torch.stack([_STATS[k] for k in slots]).cpu().tolist()
    names = ("distinct", "units8", "units16", "over8", "pairs")
    per = {n: [v[i] for v in values] for i, n in enumerate(names)}
    summary = {
        n: {
            "mean": sum(x) / len(x),
            "min": min(x),
            "max": max(x),
        }
        for n, x in per.items()
    }
    result = {"case": case, "rows": rows, "layers": len(slots), "summary": summary}
    result["per_layer"] = per
    print(
        f"[L6CENSUS] {case} rows {rows} layers {len(slots)} "
        + json.dumps(summary, sort_keys=True),
        flush=True,
    )
    if out_dir:
        path = os.path.join(out_dir, f"census.{case}.rank{rank}.json")
        with open(path, "w") as f:
            json.dump(result, f)
    return result
