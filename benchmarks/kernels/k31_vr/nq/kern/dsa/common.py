# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers of the nq DSA kernel microbenchmarks (kern/dsa).

Kimi K3.1 specialized decode, DSA layers, Rubin VR200 (SM107), TP16 = DCP16 decode
(6 query heads per rank, 96 gathered, FlashInfer trtllm-gen's 128-padded rows),
per-tensor fp8 MLA KV, verify q = 4 (K = 3) and q = 8 (K = 7).

Every bench writes one JSON line per measured (bench, case, q, rows, arm) to its
--out file (rank 0 for the torchrun ones) and resumes from its progress file
(--progress, default <out>.progress): a finished (case, q, rows) key is skipped, and
a key begun and never finished is recorded as arm "*", status "hang" (the run that
began it died or was killed by the task's watchdog) and skipped. summarize.py turns
the files into the tables, the WORSE list and the CSV.

Record fields: bench, case, q, rows, arm, role ("spec": the code's policy at that
row, "stock": the stock op it replaces, "tactic": an alternative specialized tactic,
"ref": the previous configuration (nvfp4_ds_mla), "alt": another stock kernel,
"part": a component), us (net per call, max over ranks for torchrun benches),
per_rank_us, raw_us / sub_us (the call with its restore copy, the restore alone),
status ("ok", "fail", "hang", "skip"), err, note.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NQ = os.path.dirname(os.path.dirname(HERE))  # benchmarks/kernels/k31_vr/nq

# DCP16 rank geometry of a K3.1 DSA layer.
DCP, HPR, GATHERED, Q_HEADS = 16, 6, 96, 128
TOPK, INTERLEAVE = 2048, 896
PITCH = 2112 + HPR * 128 + 160  # packed in-projection row [q_a|kv_a|gate|wk|wv]

# Verify rows in scope (multiples of q), including above the old 512 / 768 caps.
ROWS = {
    4: [4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384, 400, 448,
        512, 560, 576, 640, 768, 1024, 1536, 2048],
    8: [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384, 400, 448, 512, 560,
        576, 640, 768, 1024, 1536, 2048],
}  # fmt: skip


# Rows measured first (a run the deadline cuts still covers the range and the rows
# above the old 512 / 768 caps); the others follow in ascending order.
PRIORITY = (8, 16, 64, 128, 256, 512, 1024, 2048, 32, 96, 768, 400, 560, 576, 1536,
            4, 24, 48, 192, 384, 640)  # fmt: skip


def by_priority(cases):
    """(rows, q) pairs: PRIORITY rows first (q4 and q8 interleaved), then the rest."""
    rank = {r: i for i, r in enumerate(PRIORITY)}
    return sorted(cases, key=lambda c: (rank.get(c[0], len(rank)), c[0], c[1]))


def rows_for(q, lo=1, hi=1 << 30, only=None):
    rows = [r for r in ROWS[q] if lo <= r <= hi]
    if only:
        rows = [r for r in rows if r in only]
    return rows


def parse_ints(text):
    return [int(x) for x in text.split(",") if x.strip()] if text else []


def ctx_vectors():
    """The nq AgentX context vectors (a copy beside this file; else nq/'s)."""
    for path in (os.path.join(HERE, "decode_ctx_vectors.json"),
                 os.path.join(NQ, "decode_ctx_vectors.json")):  # fmt: skip
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError("decode_ctx_vectors.json")


def ctx_for(vecs, batch, q):
    """AgentX per-request contexts of ``batch`` requests (bench_dcp_exchange's
    packing: larger buckets repeated past the largest recorded one)."""
    group = vecs["q8_C64"] if q == 8 else vecs["q4_C48_C160_C256"]
    if str(batch) in group:
        return list(group[str(batch)])
    out: list = []
    while len(out) < batch:
        for b in sorted((int(k) for k in group), reverse=True):
            while len(out) + b <= batch:
                out += group[str(b)]
            if len(out) == batch:
                break
        if len(out) < batch:  # (a bucket of 1 always exists)
            out += group["1"]
    return out[:batch]


def local_len(global_len, rank, world=DCP, interleave=INTERLEAVE):
    full = global_len // (world * interleave)
    rem = global_len - full * world * interleave
    extra = (rem - rank * interleave).clamp(0, interleave)
    return full * interleave + extra


def row_contexts(ctxs, q, device):
    """[B] global contexts -> [B * q] per-row global lengths (verify rows)."""
    c = torch.tensor(ctxs, dtype=torch.int64, device=device)
    rows = c[:, None] - q + 1 + torch.arange(q, device=device)[None, :]
    return rows.reshape(-1)


def balanced_counts(rows, gen, device, lo=96, hi=155):
    """A DCP16 rank's share of a balanced top-k 2048 (96-155 entries per row: the
    measured spread of real indexer logits)."""
    return torch.randint(lo, hi + 1, (rows,), generator=gen, device=device).int()


def graph_time(fn, inner, reps, barrier=None, warm=3):
    """Median per-call us over ``reps`` replays of a CUDA graph of ``inner`` calls."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            fn()
    times = []
    for _ in range(reps):
        if barrier is not None:
            barrier()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        graph.replay()
        e1.record()
        e1.synchronize()
        times.append(e0.elapsed_time(e1) * 1000 / inner)
    del graph
    return statistics.median(times)


class Alternating:
    """A no-argument call alternating the DCP mailbox parity (consecutive layers)."""

    def __init__(self, call):
        self.call, self.parity = call, 0

    def __call__(self):
        self.call(self.parity)
        self.parity ^= 1


class Recorder:
    """JSONL results with resume (see the module docstring). ``rank0``: whether this
    process writes (torchrun: rank 0 only; every rank reads the same files at start,
    so every rank takes the same decisions). The progress file holds a "begin" line
    per key and an "end" line once the key's arms all ran (``finish``)."""

    def __init__(self, out, progress, bench, rank0=True):
        self.out, self.progress, self.bench, self.rank0 = out, progress, bench, rank0
        self.begun, self.ended = set(), set()
        if progress and os.path.exists(progress):
            with open(progress) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    (self.ended if rec.get("end") else self.begun).add(self.key(rec))
        self.records: list = []

    @staticmethod
    def key(rec):
        return (rec.get("bench"), rec.get("case"), rec.get("q"), rec.get("rows"))

    def _mark(self, case, q, rows, **extra):
        if self.rank0 and self.progress:
            line = dict(bench=self.bench, case=case, q=q, rows=rows, t=time.time())
            line.update(extra)
            with open(self.progress, "a") as f:
                f.write(json.dumps(line) + "\n")

    def should_run(self, case, q, rows):
        """False for a finished key, and for a key begun by an earlier run that never
        finished it (recorded as arm "*", status "hang": the arms after its last
        record did not run); else marks the key begun."""
        key = (self.bench, case, q, rows)
        if key in self.ended:
            return False
        if key in self.begun:
            self.emit(case, q, rows, "*", None, status="hang",
                      err="an earlier run began this key and died or was killed "
                      "(the arms after its last record)")  # fmt: skip
            self.finish(case, q, rows)
            return False
        self._mark(case, q, rows)
        self.begun.add(key)
        return True

    def finish(self, case, q, rows):
        self._mark(case, q, rows, end=True)
        self.ended.add((self.bench, case, q, rows))

    def emit(self, case, q, rows, arm, us, role="part", status="ok", **extra):
        rec = dict(bench=self.bench, case=case, q=q, rows=rows, arm=arm, role=role,
                   us=None if us is None else round(float(us), 3), status=status)  # fmt: skip
        rec.update({k: v for k, v in extra.items() if v is not None})
        self.records.append(rec)
        if self.rank0:
            line = json.dumps(rec)
            print("REC " + line, flush=True)
            with open(self.out, "a") as f:
                f.write(line + "\n")
        return rec


def error_text(exc):
    text = f"{type(exc).__name__}: {exc}"
    return text[:400]


def log_exc(what, exc):
    print(f"[{what}] FAILED: {error_text(exc)}", file=sys.stderr, flush=True)
    traceback.print_exc(limit=6)


STICKY = ("CUDA error", "CUDA_ERROR", "cudaError", "illegal", "unspecified launch failure",
          "misaligned address", "device-side assert", "uncorrectable ECC")  # fmt: skip


def is_sticky(exc):
    """Whether ``exc`` looks like a CUDA error that poisons the context (every later
    call in this process would fail too)."""
    text = f"{type(exc).__name__}: {exc}"
    return any(s in text for s in STICKY)


def exit_if_sticky(exc, what=""):
    """On a sticky CUDA error: flush and leave the process at once (exit 3; no
    ``finish`` of the key in flight, no further collectives). The task's resume then
    records the key as "hang" (the arms after its last record) and continues in a
    fresh process with a fresh CUDA context; torchrun tears down the other ranks."""
    if not is_sticky(exc):
        return
    print(f"[{what}] sticky CUDA error, exiting for a fresh process: {error_text(exc)}",
          file=sys.stderr, flush=True)  # fmt: skip
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(3)


def device_info(device):
    p = torch.cuda.get_device_properties(device)
    return dict(name=p.name, sms=p.multi_processor_count, cc=f"{p.major}.{p.minor}",
                mem_gb=round(p.total_memory / 2**30, 1))  # fmt: skip
