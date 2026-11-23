# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TEMPORARY DIAGNOSTIC (DCPA2A, never merge): count decode rows per query-gather path.

Device counters (captured into CUDA graphs, so replays count) per site: [calls, rows].
Host histogram: (site, rows, capturing) -> host calls. SIGUSR1 writes a report file.
"""

import collections
import os
import signal
import socket

import torch

SITES = ("direct_q_gather", "draft_sdec_qgather", "sp_sdec_attend", "sp_sdec_run",
         "mla_stock_dcp")
_dev: dict[str, torch.Tensor] = {}
_host: collections.Counter = collections.Counter()
_OUT = "/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/dcpa2a/qgcount"


def ensure(device) -> None:
    if _dev:
        return
    for s in SITES:
        _dev[s] = torch.zeros(2, dtype=torch.int64, device=device)
    try:
        signal.signal(signal.SIGUSR1, lambda *_: report())
    except ValueError:
        pass


def bump(site: str, rows: int) -> None:
    capturing = torch.cuda.is_current_stream_capturing()
    _host[(site, int(rows), capturing)] += 1
    t = _dev.get(site)
    if t is not None:
        t[0].add_(1)
        t[1].add_(int(rows))


_TRIGGER = _OUT + "/TRIGGER"
_seen: set = set()


def maybe_report() -> None:
    """Per step (main thread): write a report for each new trigger file."""
    try:
        tag = open(_TRIGGER).read().strip()
    except OSError:
        return
    if tag and tag not in _seen and _dev:
        _seen.add(tag)
        report(tag)


def report(tag: str = "sig") -> None:
    os.makedirs(_OUT, exist_ok=True)
    lines = []
    for s, t in _dev.items():
        c, r = t.tolist()
        lines.append(f"DEV {s} calls={c} rows={r}")
    for (s, rows, cap), n in sorted(_host.items()):
        lines.append(f"HOST {s} rows={rows} capturing={cap} n={n}")
    path = f"{_OUT}/{tag}-{socket.gethostname()}-{os.getpid()}.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
