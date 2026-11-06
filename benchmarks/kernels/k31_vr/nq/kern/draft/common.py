# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared pieces of the drafter kernel benches (nq/kern/draft): drafter geometry,
row lists, CUDA-graph timing, and the result table / CSV / WORSE list.

Pure-python parts (row lists, Table, band summaries) import without torch, so
``python3 common.py`` self-checks them on a laptop."""

import csv
import math
import os
import sys
import time
import traceback

# ----------------------------------------------------------------- geometry
# Per-rank (N, K) of the drafter linears at TP16 (DSpark: hidden 7168, q_lora 1536,
# kv_lora 512 + rope 64, intermediate 14336, 5 layers, vocab 163840).
VOCAB = 163840
TP = 16
NEW_HEADS, OLD_HEADS = 96, 64  # kimi-k3.1-pro-dspark-preview / kimi-k3-dspark
NEW_RANK, OLD_RANK = 512, 256  # Markov rank
# name -> ((N, K), row kind, drafters): "q" query rows (B x K), "c" context rows
# (B x (K + 1)); drafters: which drafter has that shape ("new", "old", "both").
LINEARS = {
    "fused_qkv_a": ((2112, 7168), "q", "both"),
    "q_b": ((NEW_HEADS * 192 // TP, 1536), "q", "new"),  # (1152, 1536)
    "o_proj": ((7168, NEW_HEADS * 128 // TP), "q", "new"),  # (7168, 768)
    "gate_up": ((2 * 14336 // TP, 7168), "q", "both"),  # (1792, 7168)
    "down": ((7168, 14336 // TP), "q", "both"),  # (7168, 896)
    "context_kv": ((5 * 576, 7168), "c", "both"),  # (2880, 7168)
    "ctx_shard": ((7168 // TP, 5 * 7168), "c", "both"),  # (448, 35840)
    "q_b_old": ((OLD_HEADS * 192 // TP, 1536), "q", "old"),  # (768, 1536)
    "o_proj_old": ((7168, OLD_HEADS * 128 // TP), "q", "old"),  # (7168, 512)
}
LM_HEAD = (VOCAB // TP, 7168)  # (10240, 7168)
# The stock op a specialized GEMM replaces when it is not the same-shape cuBLAS GEMM:
# the context step's sharded projection replaces the replicated context_proj.
STOCK_SHAPE = {"ctx_shard": (7168, 5 * 7168)}

# Requests per drafter graph in scope (B=1..128 and the open rows above the old caps:
# up to 2048 rows at q8 B256 context).
BATCHES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 32, 40, 48, 56, 64,
           80, 96, 112, 128, 160, 192, 256]  # fmt: skip
STEPS = {"q4": 3, "q8": 7}  # draft length K per verify width q


def rows_of(kind, batches=BATCHES, steps=STEPS):
    """{rows: [labels]} of a row kind over (q, B)."""
    out = {}
    for q, k in steps.items():
        for b in batches:
            r = b * k if kind == "q" else b * (k + 1)
            out.setdefault(r, []).append(f"{q}B{b}")
    return dict(sorted(out.items()))


def band(rows):
    """The split-K tables' row bands (runtime.draft PROJECTION_ROWS / WIDE bins)."""
    for top, name in ((8, "1-8"), (16, "9-16"), (32, "17-32"), (64, "33-64"),
                      (167, "65-167"), (168, "168"), (512, "169-512")):  # fmt: skip
        if rows <= top:
            return name
    return "513+"


# ------------------------------------------------------------------- output
class Table:
    """Rows of (case, rows, labels, specialized us, stock us, chosen, extra) -> stdout,
    a CSV flushed per row, and the WORSE list (specialized / stock > 1)."""

    COLS = ["case", "rows", "labels", "spec_us", "stock_us", "ratio", "chosen", "note"]

    def __init__(self, path, extra_cols=(), echo=True, resume=False):
        """``resume``: an existing CSV with the same header is kept and appended to;
        its rows are loaded (``done``: the (case, rows) already measured) so a run
        restarted after a sticky CUDA error (os._exit, see ``guarded``) skips them."""
        self.path, self.echo = path, echo
        self.cols = self.COLS + list(extra_cols)
        self.rows = []
        self.done = set()
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            mode = "w"
            if resume and os.path.exists(path):
                with open(path, newline="") as fh:
                    old = list(csv.reader(fh))
                if old and old[0] == self.cols:
                    for vals in old[1:]:
                        rec = dict(zip(self.cols, vals))
                        for c in ("spec_us", "stock_us", "ratio"):
                            rec[c] = _num(rec.get(c))
                        self.rows.append(rec)
                        self.done.add((rec["case"], int(rec["rows"])))
                    mode = "a"
            self.fh = open(path, mode, newline="")
            self.writer = csv.writer(self.fh)
            if mode == "w":
                self.writer.writerow(self.cols)
            self.fh.flush()
        else:
            self.fh = self.writer = None
        if echo:
            print(self.header(), flush=True)

    def header(self):
        return (f"{'case':<34} {'rows':>5} {'spec_us':>9} {'stock_us':>9} "
                f"{'ratio':>6}  chosen / note")  # fmt: skip

    def add(self, case, rows, spec, stock, chosen="", note="", labels="", **extra):
        ratio = spec / stock if spec and stock and stock > 0 else float("nan")
        rec = dict(case=case, rows=rows, labels=labels, spec_us=spec, stock_us=stock,
                   ratio=ratio, chosen=chosen, note=note, **extra)  # fmt: skip
        self.rows.append(rec)
        if self.writer:
            self.writer.writerow(
                [_fmt(rec.get(c, ""), 4 if c == "ratio" else 3) for c in self.cols]
            )
            self.fh.flush()
        if self.echo:
            more = " ".join(f"{k}={_fmt(v)}" for k, v in extra.items())
            print(f"{case:<34} {rows:>5} {_fmt(spec):>9} {_fmt(stock):>9} "
                  f"{_fmt(ratio, 3):>6}  {chosen} {note} {more}".rstrip(),
                  flush=True)  # fmt: skip
        return rec

    def worse(self, limit=1.0):
        return [r for r in self.rows
                if isinstance(r["ratio"], float) and not math.isnan(r["ratio"])
                and r["ratio"] > limit]  # fmt: skip

    def finish(self, title=""):
        bad = self.worse()
        print(f"\n==== WORSE ({title}): specialized / stock > 1.0: {len(bad)} of "
              f"{len(self.rows)} rows", flush=True)  # fmt: skip
        for r in bad:
            print(f"WORSE {r['case']:<34} rows {r['rows']:>5} spec {_fmt(r['spec_us'])} "
                  f"stock {_fmt(r['stock_us'])} ratio {_fmt(r['ratio'], 3)} "
                  f"({r['chosen']}) {r['labels']}", flush=True)  # fmt: skip
        if self.fh:
            self.fh.close()
            self.fh = None
        return bad


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt(v, digits=2):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.{digits}f}"
    return "" if v is None else str(v)


# Exit status of a process whose CUDA context is poisoned (sticky error): the caller
# (inner.sh) restarts the bench, which resumes from its CSVs.
STICKY_EXIT = 75
_STICKY = ("illegal memory access", "unspecified launch failure", "misaligned address",
           "illegal instruction", "hardware stack error", "uncorrectable ecc",
           "an illegal", "device-side assert", "launch timed out",
           "cuda_error_illegal", "cudaerrorillegal")  # fmt: skip


def cuda_poisoned(exc=None):
    """True when the CUDA context can no longer run work: a sticky error (every later
    call fails with it), seen in the exception text or on a device synchronize."""
    if exc is not None and any(k in str(exc).lower() for k in _STICKY):
        return True
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        return any(k in str(e).lower() for k in _STICKY) or "cuda" in str(e).lower()
    return False


def die_if_poisoned(log, what, exc=None):
    """After a failure: if the CUDA context is poisoned, flush and os._exit(STICKY_EXIT)
    (no atexit / NCCL teardown, which would hang on the dead context). Every later case
    of this process would otherwise fail and fill the CSV with MISSING rows."""
    if cuda_poisoned(exc):
        log(f"STICKY CUDA ERROR after {what}: exiting {STICKY_EXIT} (restart resumes)")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(STICKY_EXIT)


def guarded(log, what, fn, *args, **kwargs):
    """Run ``fn``; on failure log the traceback tail and return None (a failing case
    never stops the bench) -- unless the failure left a sticky CUDA error, then the
    process exits (``die_if_poisoned``)."""
    t0 = time.time()
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        tail = traceback.format_exc().strip().splitlines()[-6:]
        log(f"FAILED {what} after {time.time() - t0:.1f}s: {type(e).__name__}: "
            f"{str(e)[:300]} | " + " / ".join(tail))  # fmt: skip
        die_if_poisoned(log, what, e)
        return None


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class Deadline:
    """Soft time budget: cases are skipped (logged) once it runs out, so the phase
    ends with its tables written before the hard timeout of task.sh."""

    def __init__(self, seconds):
        self.end = time.time() + seconds if seconds else None

    def left(self):
        return float("inf") if self.end is None else self.end - time.time()

    def out(self):
        return self.left() <= 0


# ------------------------------------------------------------------- timing
def time_graph(fns, calls=64, reps=5, between=None):
    """Median us per call of the zero-arg callables ``fns`` (one per rotating
    weight copy; > 2x L2 so every call streams its weight), replayed in one CUDA
    graph of about ``calls`` calls. ``between``: a callable captured between calls
    (its own time is not subtracted here)."""
    import torch

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
            if between:
                between()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    n = len(fns)
    loops = max(1, calls // n)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(loops):
            for f in fns:
                f()
                if between:
                    between()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(4):
            g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / (4 * loops * n))
    ts.sort()
    del g
    return ts[len(ts) // 2]


def time_graph_agreed(fns, agree, barrier, calls=64, reps=5, between=None):
    """time_graph for kernels that exchange across ranks: warmup together, then a
    capture whose success every rank agrees on (``agree(ok)`` -> all ok) before any
    rank replays, so a one-sided failure raises on every rank instead of leaving the
    others spinning in a cross-rank replay."""
    import torch

    ok = True
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for f in fns:
                f()
                if between:
                    between()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        ok = False
        agree(False)
        raise
    if not agree(ok):
        raise RuntimeError("warmup failed on another rank")
    n = len(fns)
    loops = max(1, calls // n)
    g = None
    err = None
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(loops):
                for f in fns:
                    f()
                    if between:
                        between()
    except Exception as e:  # noqa: BLE001
        err, g = e, None
    if not agree(err is None):
        if err is not None:
            raise err
        raise RuntimeError("graph capture failed on another rank")
    barrier()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        barrier()
        torch.cuda.synchronize()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(4):
            g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / (4 * loops * n))
    ts.sort()
    del g
    return ts[len(ts) // 2]


def time_eager(fn, iters=50, warmup=5, sync=None):
    """Median-free mean us per call of ``fn`` launched eagerly (host overhead
    included); ``sync``: a callable run before the timed loop (a barrier)."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if sync:
        sync()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1e3 / iters


def copies_for(nbytes, target=640 << 20, lo=2, hi=128):
    """Rotating copies so that the weights' total exceeds ``target`` bytes."""
    return max(lo, min(hi, math.ceil(target / max(1, nbytes))))


# ------------------------------------------------------------- self-check
def _selfcheck():
    q = rows_of("q")
    c = rows_of("c")
    assert 3 in q and 7 in q and 768 in q and 1792 in q, sorted(q)
    assert max(c) == 2048 and 4 in c and 8 in c, sorted(c)
    assert band(7) == "1-8" and band(14) == "9-16" and band(168) == "168"
    assert band(2048) == "513+"
    for name, ((n, k), kind, _) in LINEARS.items():
        assert n > 0 and k % 128 == 0, name
    assert LINEARS["q_b"][0] == (1152, 1536) and LINEARS["o_proj"][0] == (7168, 768)
    assert LINEARS["q_b_old"][0] == (768, 1536)
    assert LINEARS["o_proj_old"][0] == (7168, 512)
    t = Table(None, echo=False)
    t.add("x", 7, 2.0, 1.0)
    t.add("y", 7, 1.0, 2.0)
    t.add("z", 7, None, 2.0)
    assert [r["case"] for r in t.worse()] == ["x"]
    assert guarded(lambda m: None, "boom", lambda: 1 / 0) is None
    assert not cuda_poisoned(ValueError("bad shape"))
    assert cuda_poisoned(RuntimeError("CUDA error: an illegal memory access was encountered"))
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.csv")
        t = Table(p, extra_cols=["band"], echo=False)
        t.add("a", 8, 2.0, 1.0, band="1-8")
        t.fh.close()
        t = Table(p, extra_cols=["band"], echo=False, resume=True)
        assert ("a", 8) in t.done and len(t.worse()) == 1, t.rows
        t.add("b", 16, 1.0, 2.0, band="9-16")
        t.finish()
        with open(p) as fh:
            assert len(fh.read().strip().splitlines()) == 3
    print(f"common.py self-check OK: {len(q)} query row counts {min(q)}-{max(q)}, "
          f"{len(c)} context row counts {min(c)}-{max(c)}")


if __name__ == "__main__":
    _selfcheck()
