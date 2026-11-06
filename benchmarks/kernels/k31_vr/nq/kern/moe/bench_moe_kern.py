# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode, MoE group: kernel microbenchmark of the wide-row
(65..2048 token rows) MoE / o-projection / AttnRes kernels against the stock op each one
replaces, at TP16 = DCP16 per-rank shapes on one Rubin VR200 node (1 or 4 GPUs).

Sections (one CSV per section and q, rows appended as cases finish):

  experts  (1 GPU)  ``experts_runtime.WideExperts`` (routing plan pre-written, as behind
           FrontRows) vs the stock routed experts (seam S5: FlashInfer TRT-LLM-Gen
           MXFP4 x MXFP8 ``trtllm_fp4_block_scale_routed_moe``, finalize, MXFP8 input
           pre-quantized). Variants: plan writer in front, stream width 8 vs 16,
           FC2 pool zone 12 vs 48, and above WIDE_MAX = 768 rows the chunk tactic
           (NQ-TUNE 6: equal ``wide_chunks`` vs 768 + remainder). Extra stock arms:
           with the MXFP8 activation quant, and the monolithic kernel (routing inside).
  attnres  (1 GPU)  ``attn_res.PreparedAttnRes`` (ClusterAttnRes, plain delta input =
           head / dense / final sites; ring input = post-tail site, Lamport ring
           pre-filled outside the timed region) vs the stock AttnRes op
           (``ops.attn_res``, the native kimi_k3_attn_res kernel). Variants: tokens per
           cluster 1..4 (NQ-TUNE 10 ROWS_TOKENS); spec vs stock is NQ-TUNE 9
           (STOCK_ROWS). nb = 2 / 4 / 8 saved blocks, final site.
  front    (torchrun, 4 real ranks + 12 emulated)  ``front_rows.FrontRows`` front GEMM +
           routing (+ the shared expert on the side stream as PreparedMoE forks it) vs
           the stock router GEMM (FP32 out) + replicated latent-down + grouped top-k
           (+ the TP-sharded shared expert). Variants: FRONT_N_ROWS tile 64 / 96 / 128
           (NQ-TUNE 7).
  layer    (torchrun, 4 + 12 emulated)  FrontRows + shared expert + WideExperts on the
           routing's plan (``PreparedMoE.run_front_rows`` without the tail) vs the stock
           layer up to its tail: router + latent-down + shared expert + MXFP8 quant +
           monolithic TRT-LLM MoE (routing inside; falls back to grouped top-k + routed).
           Variants: SHARED_FORK_ROWS 1 / 2 (NQ-TUNE 8, 1-GPU-chain proxy).
  tail     (torchrun, 4 + 12 emulated)  ``tail_rows.PreparedTailRows`` (PushShared on
           the side stream, LatentRows two-shot, NormRows, up projection, OutRows) vs
           the stock tail (NCCL all-reduce of the latent partial, RMSNorm, replicated
           up GEMM, all-reduce of the shared partial, add) over the 4 real ranks.
           Variants: NORM_POLLED on / off, UP_ROWS fused (UpTmem64) / cuBLAS, latent
           chunk 2 / 4 / 8. EXPERTS_PUSH needs the 16-rank experts -> not measured.
  oproj    (torchrun, 4 + 12 emulated)  o-projection ``OprojUmmaRows`` (WIDE_TWOSHOT,
           producer reduces) + the reduced-gather consumer with its AttnRes vs stock seam
           S3 (cuBLAS o_proj, NCCL all-reduce over the 4 real ranks, stock AttnRes op).
           Variants: WIDE_CHUNK 64 / 80 / 96 / 128 rows per chunk (NQ-TUNE 5).

"4 real ranks + 12 emulated": every kernel is compiled for TP16 (16 owners / slots);
ranks 0-3 are real GPUs exchanging through symmetric / multicast memory; the
Lamport slots that ranks 4-15 would fill (receive slots of this rank's owned columns,
gathered columns of owners 4-15, RS slots, front-box columns) are pre-filled before
every timed replay (outside the timed region), and stores to emulated owners land in a
local dummy buffer. The stock arms' collectives run over the 4 real ranks only, so
they under-state 16-rank communication; read 4-rank ratios as kernel-level proxies.

Timing: each measured region is captured once as a CUDA graph and replayed; one
replay per sample, CUDA events around the replay only, median over samples; torchrun
sections take the max over ranks per sample. A case that fails logs and continues; a
case that hangs is killed by a watchdog (the process exits), journaled as HANG and
skipped on the rerun (``--resume``, which task.sh does); a sticky CUDA error
(illegal address, ...) journals its case as FAIL and exits 18 so the rerun starts in a
fresh context. experts: pass 1 = stock arms + production for every row count, pass 2
= the variants; each pass takes rows > 768 first. Production's launch plan: at the
SHARED_DOWN_SMS rows (65-96) the experts' grid leaves SharedDown's SMs and the
front / layer shared down runs SharedDown (pushing into the tail's RS slots).

Usage::

  python bench_moe_kern.py experts --out DIR [--rows 72,128,...] [--routes uniform,req4]
  python bench_moe_kern.py attnres --out DIR
  torchrun --nproc-per-node 4 bench_moe_kern.py {front,layer,tail,oproj} --out DIR
  python bench_moe_kern.py summary --out DIR       # table, WORSE list, all.csv
  python bench_moe_kern.py selftest                # pure-python checks (no torch)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path

H, LATENT, EXPERTS, TOPK = 7168, 3584, 896, 16
TP = 16
BETA, LINEAR_BETA = 4.0, 25.0
WIDE_MAX = 768  # experts_wide.WIDE_MAX (checked against the module at run time)
ROWS = (72, 96, 128, 192, 256, 384, 512, 640, 768, 832, 1024, 1280, 1536, 2048)
ROWS_DIST = (72, 128, 256, 384, 512, 768, 1024, 1536, 2048)
CSV_FIELDS = (
    "section", "case", "q", "rows", "variant", "prod", "spec_us", "stock_us", "ratio",
    "tactic", "stock_arm", "prev_us", "note",
)  # fmt: skip

# Previous numbers recorded in code comments (us per call), the "previous performance"
# reference. The MoE precision (MXFP4 weights x MXFP8 activations, BF16 linears) did
# not change in this round; FP32 KDA state / the new drafter do not touch these kernels.
PREV = {
    # experts_runtime.WIDE / WIDE_POOL: width 16, 1 GPU, real q4 routes, us per call
    ("experts", "routed"): {
        72: 38.68, 96: 44.34, 128: 50.54, 160: 56.32, 192: 65.7, 224: 65.69,
        256: 73.2, 384: 87.0, 512: 104.9, 576: 115.0, 640: 121.9, 704: 131.9,
        768: 136.0,
    },
    # front_rows.FRONT_N_ROWS: front + routing, 1 GPU, real weights (n 64 at 384,
    # n 128 at 512)
    ("front", "route"): {384: 9.0, 512: 10.7},
    # tail_rows.NORM_POLLED: 4 ranks + 12 emulated, side push, us per tail call
    # (production setting: wait <= 192, polled >= 193); latent_chunk: M512 4 ranks
    ("tail", "tail"): {
        96: 12.37, 128: 12.52, 160: 13.60, 192: 14.67, 224: 14.87, 256: 15.59,
        512: 29.1,
    },
    # attn_res.SHARDED comment: full-rows ClusterAttnRes at 256 rows (TP16 geometry)
    ("attnres", "plain-nb2"): {256: 5.4},
    ("attnres", "plain-nb4"): {256: 5.2},
    ("attnres", "plain-nb8"): {256: 6.7},
}  # fmt: skip


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def parse_rows(spec, default):
    if not spec:
        return tuple(default)
    rows = set()
    for item in str(spec).split(","):
        lo, _, hi = item.partition("-")
        lo_i = int(lo)
        hi_i = int(hi) if hi else lo_i
        step = 64 if hi else 1
        rows.update(range(lo_i, hi_i + 1, step))
    return tuple(sorted(rows))


def maxrem_chunks(rows, size=WIDE_MAX):
    """NQ-TUNE 6 alternative to ``wide_chunks``: launches of ``size`` rows, then the
    remainder; a remainder of 64 rows or fewer (the wide engines start above 64) takes
    64-row blocks from the previous launch. Every launch starts on a 64-row block."""
    if rows <= size:
        return ((0, rows),)
    sizes = [size] * (rows // size)
    rem = rows - sum(sizes)
    if rem:
        sizes.append(rem)
    if len(sizes) > 1 and sizes[-1] <= 64:
        move = 64 * -(-(65 - sizes[-1]) // 64)
        sizes[-2] -= move
        sizes[-1] += move
    out, start = [], 0
    for n in sizes:
        out.append((start, n))
        start += n
    return tuple(out)


def equal_chunks(rows, size=WIDE_MAX):
    """Copy of ``experts_runtime.wide_chunks`` (pure python, for the self-test)."""
    if rows <= size:
        return ((0, rows),)
    count = -(-rows // size)
    step = (-(-rows // count) + 63) // 64 * 64
    return tuple((s, min(step, rows - s)) for s in range(0, rows, step))


# --------------------------------------------------------------------------------------
# Results: CSV rows, journal (resume / hang detection), summary


class Results:
    def __init__(self, out, section, tag, rank=0):
        self.out, self.section, self.rank = Path(out), section, rank
        self.out.mkdir(parents=True, exist_ok=True)
        self.csv = self.out / f"{section}{tag}.csv"
        self.journal = self.out / f"journal-{section}{tag}.txt"
        self.done = set()
        if self.journal.exists():
            for line in self.journal.read_text().splitlines():
                kind, _, key = line.partition(" ")
                if kind in ("START", "END", "HANG", "FAIL"):
                    self.done.add(key)
        self.hung = set()
        if self.journal.exists():
            starts, ends = set(), set()
            for line in self.journal.read_text().splitlines():
                kind, _, key = line.partition(" ")
                (starts if kind == "START" else ends).add(key)
            self.hung = starts - ends
        if rank == 0:
            for key in sorted(self.hung):  # a previous attempt died inside the case
                self._journal("HANG", key)
                self.add(dict(case=key, variant="-", note="HANG (previous attempt)"))

    def _journal(self, kind, key):
        if self.rank == 0:
            with open(self.journal, "a") as f:
                f.write(f"{kind} {key}\n")

    def start(self, key):
        self._journal("START", key)

    def end(self, key, ok=True):
        self._journal("END" if ok else "FAIL", key)
        if not ok:
            self._journal("END", key)

    def add(self, row):
        if self.rank != 0:
            return
        row = {k: row.get(k, "") for k in CSV_FIELDS} | {"section": self.section}
        for k in ("spec_us", "stock_us", "ratio", "prev_us"):
            if isinstance(row[k], float):
                row[k] = f"{row[k]:.3f}"
        new = not self.csv.exists()
        with open(self.csv, "a", newline="") as f:
            w = csv.DictWriter(f, CSV_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        log(
            f"ROW {row['section']} {row['case']} q{row['q']} M{row['rows']} "
            f"{row['variant']}{'*' if row['prod'] == '1' else ''} spec {row['spec_us']} "
            f"stock {row['stock_us']} ratio {row['ratio']} tactic {row['tactic']} "
            f"{row['note']}"
        )


def _f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize(out):
    """Every section CSV in ``out``: per (section, case, q, rows) the production row
    (spec, stock, ratio, tactic, prev) and the best variant; WORSE (production ratio
    > 1.0), TUNE (a variant beats production by > 2 %), FAIL/HANG lines; all.csv."""
    out = Path(out)
    rows = []
    for path in sorted(out.glob("*.csv")):
        if path.name == "all.csv":
            continue
        with open(path) as f:
            rows += list(csv.DictReader(f))
    with open(out / "all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    groups = {}
    for r in rows:
        key = (r["section"], r["case"], r["q"], r["rows"])
        groups.setdefault(key, []).append(r)
    lines, worse, tune, fails = [], [], [], []

    def order(k):
        return (k[0], k[1], k[2], int(k[3]) if str(k[3]).isdigit() else 0)

    head = (
        f"{'section':8} {'case':16} {'q':>2} {'rows':>5} {'spec_us':>9} "
        f"{'stock_us':>9} {'ratio':>6} {'prev_us':>8}  tactic | best variant"
    )
    lines.append(head)
    lines.append("-" * len(head))
    for key in sorted(groups, key=order):
        rs = groups[key]
        for r in rs:
            if "HANG" in r["note"] or "FAIL" in r["note"]:
                fails.append(f"{key[0]} {key[1]} q{key[2]} M{key[3]} {r['variant']}: "
                             f"{r['note'][:160]}")  # fmt: skip
        prod = [r for r in rs if r["prod"] == "1" and _f(r["spec_us"])]
        timed = [r for r in rs if _f(r["spec_us"])]
        if not prod and not timed:
            continue
        p = prod[0] if prod else None
        best = min(timed, key=lambda r: _f(r["spec_us"])) if timed else None
        bstr = ""
        if best is not None and p is not None and best is not p:
            gain = 100 * (1 - _f(best["spec_us"]) / _f(p["spec_us"]))
            bstr = f"{best['variant']} {_f(best['spec_us']):.2f} ({gain:+.1f}%)"
            if gain > 2.0:
                tune.append(f"{key[0]} {key[1]} q{key[2]} M{key[3]}: {best['variant']} "
                            f"{_f(best['spec_us']):.2f} vs prod {p['variant']} "
                            f"{_f(p['spec_us']):.2f} us ({gain:+.1f}%)")  # fmt: skip
        if p is None:
            lines.append(f"{key[0]:8} {key[1]:16} {key[2]:>2} {key[3]:>5}  (no "
                         f"production row) | {bstr}")  # fmt: skip
            continue
        ratio = _f(p["ratio"])
        lines.append(
            f"{key[0]:8} {key[1]:16} {key[2]:>2} {key[3]:>5} {_f(p['spec_us']):9.2f} "
            f"{p['stock_us'] or '-':>9} {p['ratio'] or '-':>6} {p['prev_us'] or '-':>8}"
            f"  {p['tactic']} | {bstr}"
        )
        if ratio is not None and ratio > 1.0:
            worse.append(f"{key[0]} {key[1]} q{key[2]} M{key[3]}: spec "
                         f"{_f(p['spec_us']):.2f} vs stock {p['stock_us']} "
                         f"({p['stock_arm']}) ratio {ratio:.3f} [{p['tactic']}]")  # fmt: skip
    text = "\n".join(lines)
    text += f"\n\nWORSE (production ratio > 1.0): {len(worse)}\n" + "\n".join(worse)
    text += f"\n\nTUNE (a variant beats production by > 2%): {len(tune)}\n"
    text += "\n".join(tune)
    text += f"\n\nFAIL / HANG: {len(fails)}\n" + "\n".join(fails) + "\n"
    (out / "SUMMARY.txt").write_text(text)
    print(text)
    return worse, tune, fails


# --------------------------------------------------------------------------------------
# GPU helpers (torch imported lazily: summary / selftest run without it)


class Watchdog:
    """Exits the process if a case runs past ``seconds`` (a spinning Lamport kernel
    cannot be interrupted); the journal then shows START without END = HANG."""

    def __init__(self, seconds):
        self.seconds, self.timer = seconds, None

    @contextlib.contextmanager
    def case(self, key):
        def fire():
            log(f"WATCHDOG: case {key} exceeded {self.seconds}s, exiting")
            sys.stdout.flush()
            os._exit(17)

        self.timer = threading.Timer(self.seconds, fire)
        self.timer.daemon = True
        self.timer.start()
        try:
            yield
        finally:
            self.timer.cancel()


STICKY_EXIT = 18  # inner.sh's run_section restarts the section; the journal resumes


def cuda_poisoned():
    """True if the CUDA context holds a sticky error (illegal address, ...): every later
    kernel / sync in this process fails, so the case must end the process."""
    import torch

    try:
        torch.cuda.synchronize()
        return False
    except Exception as error:  # noqa: BLE001
        log(f"sticky CUDA error: {type(error).__name__}: {error}")
        return True


def raise_if_poisoned(error):
    """In a variant's ``except``: abort the case (re-raise) on a sticky CUDA error
    instead of recording every later variant as FAIL in a dead context."""
    if cuda_poisoned():
        raise error


def exit_sticky(res, key, rows=""):
    """Journal ``key`` as FAIL (done: the restart skips it) and exit for a fresh
    CUDA context."""
    res.add(dict(case=key, rows=rows, variant="-",
                 note=f"FAIL sticky CUDA error; process exits {STICKY_EXIT}"))  # fmt: skip
    res.end(key, False)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(STICKY_EXIT)


class Dist:
    """torchrun context: vLLM TP group over the real ranks (the owner group the
    specialized boxes rendezvous on), a gloo CPU group for barriers."""

    def __init__(self):
        import torch
        import torch.distributed as dist

        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import get_tp_group
        from vllm.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
            set_custom_all_reduce,
        )

        self.rank = int(os.environ.get("RANK", 0))
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        local = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device("cuda", local)
        torch.accelerator.set_device_index(self.device)
        self._cfg = set_current_vllm_config(VllmConfig())
        self._cfg.__enter__()
        init_distributed_environment()
        set_custom_all_reduce(False)
        initialize_model_parallel(tensor_model_parallel_size=self.world)
        group = get_tp_group()
        self.group, self.cpu = group.device_group, group.cpu_group
        self.dist = dist
        if self.world > TP:
            raise SystemExit("at most 16 real ranks")

    def barrier(self):
        self.dist.barrier(group=self.cpu)

    def all_ok(self, ok):
        import torch

        flag = torch.tensor([0 if ok else 1], dtype=torch.int32)
        self.dist.all_reduce(flag, group=self.cpu)
        return int(flag.item()) == 0


def time_region(run, *, prefill=None, dist=None, warm=3, samples=25, graph=True):
    """Median us of one ``run()`` per sample (CUDA graph replay; eager fallback).
    ``prefill``: eager, before every replay, outside the timed region; ``dist``: a
    barrier before every replay and the max over ranks per sample. Returns (us, how)."""
    import torch

    def ready():
        if prefill is not None:
            prefill()
        if dist is not None:
            torch.cuda.synchronize()
            dist.barrier()

    for _ in range(warm):
        ready()
        run()
    torch.cuda.synchronize()
    how, g = "graph", None
    if graph:
        try:
            if dist is not None:
                dist.barrier()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run()
            torch.cuda.synchronize()
        except Exception as error:  # noqa: BLE001
            log(f"graph capture failed ({type(error).__name__}: {error}); eager timing")
            g, how = None, "eager"
            with contextlib.suppress(Exception):
                torch.cuda.synchronize()
        if dist is not None and not dist.all_ok(g is not None):  # every rank alike
            g, how = None, "eager"
    else:
        how = "eager"
    if g is not None:  # one replay after capture (warm the graph's launch path)
        ready()
        g.replay()
        torch.cuda.synchronize()
    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(True)
    for _ in range(samples):
        ready()
        start.record()
        if g is not None:
            g.replay()
        else:
            run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    if dist is not None:
        t = torch.tensor(times, dtype=torch.float64, device=dist.device)
        dist.dist.all_reduce(t, op=dist.dist.ReduceOp.MAX, group=dist.group)
        times = t.tolist()
    del g
    return statistics.median(times), how


@contextlib.contextmanager
def patched(module, **values):
    """Temporarily set module attributes (policy tables) for one variant."""
    old = {k: getattr(module, k) for k in values}
    try:
        for k, v in values.items():
            setattr(module, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


def beside_factory(device):
    """PreparedMoE.beside: ``work`` on a side stream forked from the current stream,
    joined after the block (the yielded join may be called earlier)."""
    import torch

    side = torch.cuda.Stream(device=device)
    begin, done = torch.cuda.Event(), torch.cuda.Event()

    @contextlib.contextmanager
    def beside(work):
        main = torch.cuda.current_stream(device)
        begin.record(main)
        with torch.cuda.stream(side):
            side.wait_event(begin)
            work()
            done.record(side)
        yield lambda: main.wait_event(done)
        main.wait_event(done)

    return beside


def moe_weights(device, seed=11):
    """One TP16 MoE layer's routed MXFP4 shards (bytes; scales in a sane E8M0 range):
    w13 [896, 512, 1792], s13 [896, 512, 112], w2 [896, 3584, 128], s2 [896, 3584, 8].
    The specialized and the stock arms read the same bytes (layouts differ: timing
    only; correctness is the GPU tests' job)."""
    import torch

    gen = torch.Generator(device=device).manual_seed(seed)
    u8 = torch.uint8

    def b(shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, dtype=u8, device=device, generator=gen)

    return (
        b((EXPERTS, 512, 1792)),
        b((EXPERTS, 512, 112), 118, 127),
        b((EXPERTS, LATENT, 128)),
        b((EXPERTS, LATENT, 8), 118, 127),
    )


def place_weights(weights):
    """``experts_runtime.place`` on the routed shards (die-alternating re-home + SM die
    map, as production) or () when the device has no trusted die map."""
    from types import SimpleNamespace

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_runtime as er,
    )

    w13, s13, w2, s2 = weights
    ns = SimpleNamespace(w13_weight=w13, w2_weight=w2)
    try:
        dies = er.place(ns)
    except Exception as error:  # noqa: BLE001
        log(f"place() failed ({error}); die-oblivious experts")
        return weights, ()
    return (ns.w13_weight, s13, ns.w2_weight, s2), dies


def make_routes(rows, mode, device, seed):
    """[rows, 16] int32 ids, BF16 weights (sum 1), BF16 latent. ``mode``: "uniform"
    (16 distinct experts per token), or "req<q>": a request's q consecutive tokens
    share a 16-expert base and each slot is replaced by another expert with p 0.25
    (verify tokens of one request route alike, as real DSpark verify routes do)."""
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    ids = np.empty((rows, TOPK), dtype=np.int32)
    q = int(mode[3:]) if mode.startswith("req") else 1
    for r0 in range(0, rows, q):
        base = rng.choice(EXPERTS, TOPK, replace=False)
        for t in range(r0, min(rows, r0 + q)):
            if q == 1:
                ids[t] = rng.choice(EXPERTS, TOPK, replace=False)
                continue
            row = base.copy()
            for s in range(TOPK):
                if rng.random() < 0.25:
                    new = int(rng.integers(EXPERTS))
                    while new in row:
                        new = int(rng.integers(EXPERTS))
                    row[s] = new
            ids[t] = row
    gen = torch.Generator(device=device).manual_seed(seed)
    w = torch.rand(rows, TOPK, device=device, generator=gen)
    w = (w / w.sum(1, keepdim=True)).bfloat16()
    latent = (torch.randn(rows, LATENT, device=device, generator=gen) * 0.5).bfloat16()
    return torch.tensor(ids, device=device), w, latent


class StockExpertsFI:
    """The stock routed experts' kernels, called directly: FlashInfer TRT-LLM-Gen
    MXFP4 x MXFP8 (as ``TrtLlmMxfp4ExpertsModular`` / ``...Monolithic`` call them for
    the K3 MoE: SiTU (beta 4, linear beta 25), I = 256 per rank, 896 experts)."""

    def __init__(self, weights, device, autotune=True):
        import torch

        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.quantization.utils import flashinfer_utils as fu

        self.w13, s13, self.w2, s2 = weights
        self.s13 = s13.view(torch.float8_e4m3fn)
        self.s2 = s2.view(torch.float8_e4m3fn)
        self.device, self.autotune = device, autotune
        situ = getattr(fu, "has_flashinfer_situ_activation", lambda: False)()
        act = MoEActivation.SITU if situ else MoEActivation.SILU
        self.act = fu.activation_to_flashinfer_int(act)
        self.note = "" if situ else "no FI SiTU: SiLU"
        full = lambda v: torch.full((EXPERTS,), v, dtype=torch.float32, device=device)  # noqa: E731
        self.alpha, self.beta = (full(BETA), full(LINEAR_BETA)) if situ else (None, None)
        self.bias = torch.zeros(EXPERTS, dtype=torch.bfloat16, device=device)
        self.tuned = set()

    def quant(self, latent):
        import torch

        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize,
        )

        x, s = mxfp8_e4m3_quantize(latent)
        return x, s.view(torch.float8_e4m3fn)

    def common(self):
        return dict(
            gemm1_weights=self.w13, gemm1_weights_scale=self.s13, gemm1_bias=None,
            gemm1_alpha=self.alpha, gemm1_beta=self.beta, gemm1_clamp_limit=None,
            gemm2_weights=self.w2, gemm2_weights_scale=self.s2, gemm2_bias=None,
            output1_scale_scalar=None, output1_scale_gate_scalar=None,
            output2_scale_scalar=None, num_experts=EXPERTS, top_k=TOPK,
            intermediate_size=256, local_expert_offset=0, local_num_experts=EXPERTS,
            activation_type=self.act, tune_max_num_tokens=8192,
        )  # fmt: skip

    def routed(self, x, s, ids, w, out):
        from flashinfer import trtllm_fp4_block_scale_routed_moe

        return trtllm_fp4_block_scale_routed_moe(
            topk_ids=(ids, w), routing_bias=None, hidden_states=x,
            hidden_states_scale=s, n_group=None, topk_group=None,
            routed_scaling_factor=None, routing_method_type=1, do_finalize=True,
            enable_pdl=True, output=out, **self.common(),
        )  # fmt: skip

    def mono(self, x, s, logits, out):
        from flashinfer import trtllm_fp4_block_scale_moe

        return trtllm_fp4_block_scale_moe(
            routing_logits=logits, routing_bias=self.bias, hidden_states=x,
            hidden_states_scale=s, n_group=1, topk_group=1, routed_scaling_factor=1.0,
            routing_method_type=2, do_finalize=True, output=out, **self.common(),
        )  # fmt: skip

    def tune(self, key, fn):
        """FlashInfer's autotuner once per (arm, rows) outside capture (production
        runs it at startup unless disabled)."""
        if self.autotune and key not in self.tuned:
            try:
                from flashinfer.autotuner import autotune

                with autotune(True):
                    fn()
            except Exception as error:  # noqa: BLE001
                log(f"stock autotune {key} failed: {error}")
            self.tuned.add(key)
        fn()


# --------------------------------------------------------------------------------------
# Section: experts (1 GPU)


def _experts_impl(args):
    import torch

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.policy import (
        batch_table,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_runtime as er,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_wide as ew,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        front_rows as fr,
    )

    assert ew.WIDE_MAX == WIDE_MAX, ew.WIDE_MAX
    dev = torch.device("cuda", 0)
    torch.accelerator.set_device_index(dev)
    rows_all = parse_rows(args.rows, ROWS)
    res = Results(args.out, "experts", args.tag)
    dog = Watchdog(args.case_timeout)
    raw = moe_weights(dev)
    stock = StockExpertsFI(raw, dev, autotune=not args.no_stock_autotune)
    weights, dies = place_weights(raw)  # die-placed in place (same bytes, both arms)
    log(f"experts: dies {'placed' if dies else 'oblivious'}; rows {rows_all}")
    buffers = er.scratch(TP, dev, rows=max(rows_all))
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    prod_width = batch_table(er.WIDE[TP])
    prod_pool = batch_table(er.WIDE_POOL[TP])
    # production's grid at SharedDown rows (moe.prepared: the experts' grid leaves
    # front_rows.SHARED_DOWN_SMS SMs to the merged rows' shared down)
    down_sms = {
        r: n for r in rows_all
        if fr.rows_policy(fr.MERGED_ROWS, fr.Shape(r, 4), 0)
        and (n := fr.rows_policy(fr.SHARED_DOWN_SMS, fr.Shape(r, 4), 0))
    }  # fmt: skip
    modes = [m for m in args.routes.split(",") if m]
    # pass 1: stock arms + production for every row count, pass 2: the variants; each
    # pass takes the chunked rows (> WIDE_MAX, NQ-TUNE 6) first, so the budget cuts
    # the least important cases.
    order = [r for r in rows_all if r > WIDE_MAX] + [r for r in rows_all if r <= WIDE_MAX]
    passes = ("prod",) if args.quick else ("prod", "var")
    for phase, mode in [(p, m) for p in passes for m in modes]:
        q = mode[3:] if mode.startswith("req") else "-"
        for rows in order:
            if q != "-" and rows % int(q):
                continue
            key = f"routed:{mode}:M{rows}" + ("" if phase == "prod" else ":var")
            if key in res.done:
                continue
            res.start(key)
            ok = True
            with dog.case(key):
                try:
                    ok = _experts_case(
                        args, res, er, ew, stock, weights, dies, buffers, rows, mode,
                        q, prod_width, prod_pool, stream, to_cute, sms, dev, phase,
                        down_sms.get(rows, 0),
                    )  # fmt: skip
                except Exception as error:  # noqa: BLE001
                    ok = False
                    res.add(dict(case="routed", q=q, rows=rows, variant="*",
                                 note=f"FAIL {type(error).__name__}: {error}"[:300]))  # fmt: skip
                    traceback.print_exc()
                if cuda_poisoned():
                    exit_sticky(res, key, rows)
            res.end(key, ok)
            torch.cuda.empty_cache()


def _stock_routed_from_csv(res, q, rows):
    """Pass 2: the stock routed time pass 1 recorded for (q, rows) (this process or a
    previous attempt), or None."""
    if not res.csv.exists():
        return None
    with open(res.csv) as f:
        for r in csv.DictReader(f):
            if (r["variant"] == "stock:routed" and r["q"] == str(q)
                    and r["rows"] == str(rows) and _f(r["stock_us"])):  # fmt: skip
                return _f(r["stock_us"])
    return None


def _experts_case(
    args, res, er, ew, stock, weights, dies, buffers, rows, mode, q, prod_width,
    prod_pool, stream, to_cute, sms, dev, phase="prod", down_sms=0,
):  # fmt: skip
    """``phase`` "prod": the stock arms and the production variant; "var": the other
    variants against pass 1's stock time. ``down_sms``: SMs production's SharedDown
    takes from the experts' grid at these rows (0: none)."""
    import torch

    seed = 1000 + rows + (0 if mode == "uniform" else 7 * int(mode[3:]))
    ids, topk_w, latent = make_routes(rows, mode, dev, seed)
    distinct = int(torch.unique(ids).numel())
    note0 = f"routes {mode}, {distinct} distinct experts"
    out = torch.empty(rows, LATENT, dtype=torch.bfloat16, device=dev)
    stock_us, stock_arm, stock_note = None, "", ""
    arms = {}
    if phase == "var":
        stock_us = _stock_routed_from_csv(res, q, rows)
        stock_arm = "fi-routed(prequant)" if stock_us else ""
        stock_note = "" if stock_us else "no stock routed time from pass 1"
        return _experts_variants(
            args, res, er, weights, dies, buffers, rows, q, prod_width, prod_pool,
            stream, to_cute, sms, dev, phase, down_sms, ids, topk_w, latent, note0,
            stock_us, stock_arm, stock_note,
        )  # fmt: skip
    # ---- stock arms
    x8, s8 = stock.quant(latent)
    try:
        stock.tune(("routed", rows), lambda: stock.routed(x8, s8, ids, topk_w, out))
        arms["routed"] = time_region(
            lambda: stock.routed(x8, s8, ids, topk_w, out), samples=args.samples
        )
    except Exception as error:  # noqa: BLE001
        stock_note = f"stock routed FAIL {type(error).__name__}: {error}"[:200]
        traceback.print_exc()
        raise_if_poisoned(error)
    try:
        def routed_q():
            xq, sq = stock.quant(latent)
            stock.routed(xq, sq, ids, topk_w, out)

        arms["routed+quant"] = time_region(routed_q, samples=args.samples)
    except Exception as error:  # noqa: BLE001
        log(f"stock routed+quant FAIL {error}")
        raise_if_poisoned(error)
    if args.stock_mono:
        try:
            logits = torch.randn(rows, EXPERTS, device=dev, dtype=torch.float32)
            stock.tune(("mono", rows), lambda: stock.mono(x8, s8, logits, out))
            arms["mono"] = time_region(
                lambda: stock.mono(x8, s8, logits, out), samples=args.samples
            )
        except Exception as error:  # noqa: BLE001
            log(f"stock mono FAIL {type(error).__name__}: {error}")
            raise_if_poisoned(error)
    if "routed" in arms:
        stock_us, stock_arm = arms["routed"][0], "fi-routed(prequant)"
    for name, (us, how) in arms.items():
        res.add(dict(case="routed", q=q, rows=rows, variant=f"stock:{name}",
                     prod="0", stock_us=us, tactic=how, stock_arm=f"fi-{name}",
                     note=note0 + (f"; {stock.note}" if stock.note else "")))  # fmt: skip
    return _experts_variants(
        args, res, er, weights, dies, buffers, rows, q, prod_width, prod_pool,
        stream, to_cute, sms, dev, phase, down_sms, ids, topk_w, latent, note0,
        stock_us, stock_arm, stock_note,
    )  # fmt: skip


def _experts_variants(
    args, res, er, weights, dies, buffers, rows, q, prod_width, prod_pool, stream,
    to_cute, sms, dev, phase, down_sms, ids, topk_w, latent, note0, stock_us,
    stock_arm, stock_note,
):  # fmt: skip
    import torch

    # ---- specialized variants (pass 1: production; pass 2: the rest)
    width = prod_width.get(rows) or prod_width.get(min(rows, WIDE_MAX))
    pool = prod_pool.get(rows, 12)
    variants = [("prod", {}, True, True)]
    if phase == "var":
        variants = []
        variants.append(("planwriter", {}, False, False))
        alt_w = 8 if width == 16 else 16
        if rows <= WIDE_MAX:
            variants.append((f"w{alt_w}", dict(WIDE={TP: f"65-:{alt_w}"}), True, False))
        if rows > 256:
            alt_p = 12 if pool == 48 else 48
            variants.append(
                (f"pool{alt_p}", dict(WIDE_POOL={TP: f"65-:{alt_p}"}), True, False)
            )
        if rows > WIDE_MAX:
            variants.append(("chunks768+rem", dict(wide_chunks=maxrem_chunks), True,
                             False))  # fmt: skip
    prev = PREV[("experts", "routed")].get(rows)
    for name, table, planned, is_prod in variants:
        try:
            with patched(er, **table):
                chunks = er.wide_chunks(rows)
                experts = er.PersistentExperts(*weights, dies=dies, q=4)
                if down_sms:  # production: SharedDown's SMs leave the experts' grid
                    experts.grids[rows] = min(experts.grid_for(rows), sms - down_sms)
                partial = torch.empty(rows, LATENT, dtype=torch.bfloat16, device=dev)
                wide = er.WideExperts(experts, buffers, {rows: partial}, planned=planned)
            if planned:  # the routing's plan, written once (as FrontRows' routing)
                writer = er.compile_wide_plan(rows, dev.index)
                writer(to_cute(latent, 16), er.tensor(buffers["plan"]), stream(dev))
                torch.cuda.synchronize()
            us, how = time_region(
                lambda: wide.run(latent, topk_w, ids), samples=args.samples
            )
            ratio = us / stock_us if stock_us else ""
            w = experts.wide.get(rows) or "-"
            p = experts.wide_pool.get(rows, 12)
            tactic = (
                f"width {w} pool ({p},6) launches "
                f"{'/'.join(str(n) for _, n in chunks)} grid {experts.grid_for(rows)}"
                f"{f' (SharedDown {down_sms} SMs)' if down_sms else ''}"
                f"{' planned' if planned else ' plan-writer'} [{how}]"
            )
            res.add(dict(case="routed", q=q, rows=rows, variant=name,
                         prod="1" if is_prod else "0", spec_us=us, stock_us=stock_us
                         or "", ratio=ratio, tactic=tactic, stock_arm=stock_arm,
                         prev_us=prev if is_prod and prev else "",
                         note=(note0 + ("; " + stock_note if stock_note else ""))))  # fmt: skip
            del wide, experts
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            res.add(dict(case="routed", q=q, rows=rows, variant=name,
                         prod="1" if is_prod else "0",
                         note=f"FAIL {type(error).__name__}: {error}"[:300]))  # fmt: skip
            raise_if_poisoned(error)
    return True


# --------------------------------------------------------------------------------------
# Section: attnres (1 GPU)


def _attnres_impl(args):
    import torch

    import vllm._custom_ops  # noqa: F401  (torch.ops._C)
    from vllm.models.kimi_k3.nvidia.ops.attn_res import attn_res
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.attn_res import (
        prepared as ar,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        to_cute,
    )

    dev = torch.device("cuda", 0)
    torch.accelerator.set_device_index(dev)
    rows_all = parse_rows(args.rows, ROWS)
    res = Results(args.out, "attnres", args.tag)
    dog = Watchdog(args.case_timeout)
    cap = max(rows_all)
    gen = torch.Generator(device=dev).manual_seed(5)
    bf = dict(dtype=torch.bfloat16, device=dev)
    nblocks = 8
    eps = 1e-6
    weights = (
        (1 + 0.1 * torch.randn(H, generator=gen, device=dev)).bfloat16(),
        (torch.randn(H, generator=gen, device=dev) / H**0.5).bfloat16(),
        (1 + 0.1 * torch.randn(H, generator=gen, device=dev)).bfloat16(),
    )
    prefix0 = torch.randn(cap, H, generator=gen, device=dev).bfloat16()
    blocks = torch.randn(cap, nblocks, H, generator=gen, device=dev).bfloat16()
    delta = (0.01 * torch.randn(cap, H, generator=gen, device=dev)).bfloat16()
    ring = torch.empty(1, cap, H, **bf)
    small = torch.empty(1, 64, H, **bf)
    for t in (ring, small):
        t.view(torch.int32).fill_(-0x80000000)
    out = torch.empty(cap, H, **bf)
    # (nb, final): the ring (post-tail) and plain (head / dense / final) sites
    cases = [(2, False), (4, False), (8, False), (8, True)]
    for nb, final in cases:
        key0 = f"nb{nb}{'-final' if final else ''}"
        try:
            pa = ar.PreparedAttnRes(weights, (nb, True, -1, final), (eps, eps))
            pa.bind_ring(small)
            pa.bind_ring_rows(ring)
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            res.add(dict(case=key0, variant="*", note=f"FAIL prepare: {error}"[:300]))
            if cuda_poisoned():  # (not journaled: the restart retries the prepare)
                sys.stdout.flush()
                os._exit(STICKY_EXIT)
            continue
        for rows in rows_all:
            key = f"{key0}:M{rows}"
            if key in res.done:
                continue
            res.start(key)
            with dog.case(key):
                try:
                    ok = _attnres_case(args, res, ar, pa, attn_res, to_cute, rows, nb,
                                       final, weights, prefix0, blocks, delta, ring,
                                       out, eps, key0)  # fmt: skip
                except Exception as error:  # noqa: BLE001
                    ok = False
                    traceback.print_exc()
                    res.add(dict(case=key0, rows=rows, variant="*",
                                 note=f"FAIL {type(error).__name__}: {error}"[:300]))  # fmt: skip
                if cuda_poisoned():
                    exit_sticky(res, key, rows)
            res.end(key, ok)


def _attnres_case(args, res, ar, pa, attn_res, to_cute, rows, nb, final, weights,
                  prefix0, blocks, delta, ring, out, eps, key0):  # fmt: skip
    import torch

    prefix = prefix0[:rows].clone()
    b, d, o = blocks[:rows], delta[:rows], out[:rows]
    norm, query, onorm = weights
    stock_us = None
    normed = torch.empty_like(o)
    stock_arm = "ops.attn_res" + ("+rms_norm(final norm)" if final else "")
    try:
        def stock():
            h = attn_res(prefix, d, b, norm, query, None if final else onorm,
                         num_blocks=nb, block_write_idx=-1, eps=eps,
                         output_norm_eps=0.0 if final else eps)  # fmt: skip
            if final:  # the stock model applies its final RMSNorm (``self.norm``)
                torch.ops._C.rms_norm(normed, h, onorm, eps)  # after AttnRes; the
                # specialized final kernel fuses it (output weight = model.norm)

        stock_us, how = time_region(stock, samples=args.samples)
        res.add(dict(case=f"plain-{key0}", rows=rows, variant="stock", prod="0",
                     stock_us=stock_us, tactic=how, stock_arm=stock_arm))  # fmt: skip
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        res.add(dict(case=f"plain-{key0}", rows=rows, variant="stock",
                     note=f"FAIL {error}"[:300]))  # fmt: skip
        raise_if_poisoned(error)
    prod_tokens = ar.rows_tokens(rows)
    stock_rows = bool(ar.rows_policy(ar.STOCK_ROWS, ar.Shape(rows), default=0))
    pattern = d.contiguous()

    def fill_ring():
        ring[0, :rows].copy_(pattern)

    for site in ("plain", "ring"):
        case = f"{site}-{key0}"
        prev = PREV.get(("attnres", f"plain-nb{nb}"), {}).get(rows) if (
            site == "plain" and not final) else None  # fmt: skip
        for tokens in (1, 2, 3, 4):
            if tokens > 1 and args.quick and tokens != 2:
                continue
            if tokens > 1 and site == "plain":  # token-batched clusters: ring site only
                continue
            is_prod = tokens == (prod_tokens if site == "ring" else 1)
            try:
                slots, cap = (1, ring.shape[1]) if site == "ring" else (0, 0)
                code = pa._compile(slots, cap, tokens=tokens)
                source = to_cute(ring) if site == "ring" else None

                def spec():
                    if site == "ring":
                        pa._launch(code, source, prefix, b, o)
                    else:
                        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (  # noqa: E501
                            dynamic_view,
                        )

                        pa._launch(code, dynamic_view(d), prefix, b, o)

                us, how = time_region(
                    spec, prefill=fill_ring if site == "ring" else None,
                    samples=args.samples,
                )  # fmt: skip
                tactic = f"ClusterAttnRes tokens/cluster {tokens} [{how}]"
                if site == "plain" and is_prod and stock_rows:
                    tactic += " (STOCK_ROWS: production runs stock here)"
                res.add(dict(case=case, rows=rows, variant=f"t{tokens}",
                             prod="1" if is_prod else "0", spec_us=us,
                             stock_us=stock_us or "",
                             ratio=us / stock_us if stock_us else "",
                             tactic=tactic, stock_arm=stock_arm + "(plain delta)",
                             prev_us=prev if is_prod and prev else ""))  # fmt: skip
            except Exception as error:  # noqa: BLE001
                if tokens == 1:
                    traceback.print_exc()
                res.add(dict(case=case, rows=rows, variant=f"t{tokens}",
                             prod="1" if is_prod else "0",
                             note=f"FAIL {type(error).__name__}: {error}"[:300]))  # fmt: skip
                raise_if_poisoned(error)
    return True


# --------------------------------------------------------------------------------------
# Emulation helpers (4 real ranks + 12 emulated)


def emulate_peers(real_ptrs, dummy_ptr, parities, size):
    """[parities, 16] int64 owner bases: the real ranks' symmetric buffers, then a
    local dummy buffer for owners >= world (stores to emulated owners land there)."""
    world = len(real_ptrs)
    bases = [
        [int(real_ptrs[o]) + p * size if o < world else dummy_ptr + p * size
         for o in range(TP)]
        for p in range(parities)
    ]  # fmt: skip
    return bases


def patch_twoshot(box, world, device):
    """TwoShot built over the real ranks with TP16 slots: owner bases for all 16."""
    import torch

    dummy = torch.empty_like(box.recv)
    size = box.recv[0].numel() * 2
    ptrs = box.peers[:, :world][0].tolist()  # parity 0 bases of the real ranks
    bases = emulate_peers(ptrs, dummy.data_ptr(), box.parities, size)
    box.peers = torch.tensor(bases, dtype=torch.int64, device=device)
    box._emulation_dummy = dummy
    return box


def section_dist(args):
    import torch

    d = Dist()
    tag = args.tag
    res = Results(args.out, args.section, tag, rank=d.rank)
    dog = Watchdog(args.case_timeout)
    fn = {"front": _front_impl, "layer": _front_impl, "tail": _tail_impl,
          "oproj": _oproj_impl}[args.section]  # fmt: skip
    log(f"rank {d.rank}/{d.world} {args.section}: {d.world} real + "
        f"{TP - d.world} emulated ranks")  # fmt: skip
    try:
        fn(args, d, res, dog)
    finally:
        torch.cuda.synchronize()
        d.barrier()


def run_case(d, res, dog, key, body):
    """One distributed case: journaled, watchdog-guarded, the ok flag agreed by every
    rank (a rank that failed alone does not desync the next case)."""
    if key in res.done:
        return
    res.start(key)
    ok = True
    with dog.case(key):
        try:
            body()
        except Exception as error:  # noqa: BLE001
            ok = False
            traceback.print_exc()
            log(f"rank {d.rank} case {key} FAIL {type(error).__name__}: {error}")
        alive = not cuda_poisoned()
        ok = d.all_ok(ok)
        if not d.all_ok(alive):  # (gloo: works with a dead CUDA context) every rank
            exit_sticky(res, key)
    res.end(key, ok)


# --------------------------------------------------------------------------------------
# Sections: front / layer


def _front_impl(args, d, res, dog):
    from types import SimpleNamespace

    import torch

    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_runtime as er,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        front_rows as fr,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        prepared as mp,
    )

    dev, world, rank = d.device, d.world, d.rank
    layer = args.section == "layer"
    rows_all = parse_rows(args.rows, ROWS_DIST)
    q = int(args.q)
    rows_all = tuple(r for r in rows_all if r % q == 0)
    cap = max(rows_all)
    gen = torch.Generator(device=dev).manual_seed(100 + rank)
    bf = dict(dtype=torch.bfloat16, device=dev)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=dev) * scale).bfloat16()

    shared_h = 6144 // TP
    gate_up, down = rnd(2 * shared_h, H, scale=H**-0.5), rnd(H, shared_h, scale=0.05)
    router_full = rnd(EXPERTS, H, scale=H**-0.5)
    down_full = rnd(LATENT, H, scale=H**-0.5)
    bias = torch.zeros(EXPERTS, dtype=torch.float32, device=dev)
    moe = SimpleNamespace(
        shared_experts=SimpleNamespace(
            gate_up_proj=SimpleNamespace(weight=gate_up),
            down_proj=SimpleNamespace(weight=down),
        ),
        gate=SimpleNamespace(e_score_correction_bias=bias, weight=router_full),
        routed_scaling_factor=1.0,
        moe_renormalize=True,
    )
    valid, shard = EXPERTS // TP, LATENT // TP
    router = router_full[rank * valid : (rank + 1) * valid]
    packed = torch.cat(
        (router, router.new_zeros((-valid % 16, H)),
         down_full[rank * shard : (rank + 1) * shard])
    ).contiguous()  # fmt: skip
    state = mp.MoEScratch(TP, dev, rows_all, owner=(d.group, dev, TP, rank))
    box = state.front_rows
    # SHARED_DOWN_SMS rows (production: SharedDown pushes the merged rows' shared down
    # straight into the tail's RS slots, ``PreparedMoE.run_front_rows``): the push
    # arguments of ``PreparedTailRows.push_args`` (parity 0) over the wide tail's box,
    # emulated owners' slots in a local dummy buffer (as the tail section).
    push_args = None
    if state.rows is not None:
        import cutlass

        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
            to_cute,
        )

        tbox = state.rows
        rs_dummy = torch.empty_like(tbox.rs.buffer)
        real = list(tbox.rs.handle.buffer_ptrs)
        tbox.rs_peers = torch.tensor(
            [int(real[o]) if o < world else rs_dummy.data_ptr() for o in range(TP)],
            dtype=torch.int64, device=dev,
        )  # fmt: skip
        tbox._emulation_dummy = rs_dummy
        push_args = (to_cute(tbox.rs_peers, 16), cutlass.Int32(0), cutlass.Int32(rank),
                     cutlass.Int32(tbox.cap))  # fmt: skip
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    # emulated ranks' front-box columns: latent [shard*world, 3584), logit words
    lat_pat = rnd(cap, LATENT - shard * world, scale=0.5)
    logit_pat = (
        torch.randn(cap, valid * (TP - world), generator=gen, device=dev)
        .contiguous()
        .view(torch.bfloat16)
    )

    def prefill_box(b):
        buf = box.buffer
        buf[0, :b, shard * world : LATENT].copy_(lat_pat[:b])
        buf[0, :b, LATENT + 2 * valid * world :].copy_(logit_pat[:b])

    x_all = rnd(cap, H, scale=1.0)
    beside = beside_factory(dev)
    experts = stock = None
    if layer:
        raw = moe_weights(dev, seed=11 + rank)
        weights, dies = place_weights(raw)
        stock = StockExpertsFI(weights, dev, autotune=not args.no_stock_autotune)
        experts = er.PersistentExperts(*weights, dies=dies, q=q)
    wides = {}

    def wide_for(b):
        """WideExperts for ``b`` rows on the routing's plan (prepared per row count,
        inside its case: a failing compile fails that case only)."""
        if b not in wides:
            wides[b] = er.WideExperts(
                experts, state.experts, {b: state.latent_partial}, planned=True
            )
        return wides[b]

    side = torch.cuda.Stream(device=dev)
    ev0, ev1 = torch.cuda.Event(), torch.cuda.Event()

    def stock_front(x, with_shared):
        """Stock K3 MoE front: GateLinear (FP32 out) on the main stream beside the
        replicated latent-down (aux stream at <= 256 tokens), grouped top-k (layer:
        inside the monolithic kernel instead); shared expert (TP-sharded KimiMLP)."""
        main = torch.cuda.current_stream(dev)
        b = x.shape[0]
        if b <= 256:
            ev0.record(main)
            with torch.cuda.stream(side):
                side.wait_event(ev0)
                lat = torch.mm(x, down_full.t())
                ev1.record(side)
            logits = torch.mm(x, router_full.t(), out_dtype=torch.float32)
            main.wait_event(ev1)
        else:
            logits = torch.mm(x, router_full.t(), out_dtype=torch.float32)
            lat = torch.mm(x, down_full.t())
        routes = None
        if not layer:
            routes = ops.grouped_topk(logits, 1, 1, TOPK, True, 1.0, bias, 1)
        shared = None
        if with_shared:
            gu = torch.mm(x, gate_up.t())
            h = torch.empty(b, shared_h, **bf)
            torch.ops._C.situ_and_mul(h, gu, BETA, LINEAR_BETA)
            shared = torch.mm(h, down.t())
        return lat, logits, routes, shared

    out_stock = torch.empty(cap, LATENT, **bf)

    def stock_layer(x):
        lat, logits, _, _ = stock_front(x, True)
        xq, sq = stock.quant(lat)
        b = x.shape[0]
        try_mono = getattr(stock_layer, "mono", True)
        if try_mono:
            stock.mono(xq, sq, logits, out_stock[:b])
        else:
            w, ids = ops.grouped_topk(logits, 1, 1, TOPK, True, 1.0, bias, 1)
            stock.routed(xq, sq, ids, w.bfloat16(), out_stock[:b])

    n_variants = [64, 96, 128]
    fork_variants = [1, 2]
    for rows in rows_all:
        x = x_all[:rows]
        shape = fr.Shape(rows, q)
        prod_n = fr.rows_policy(fr.FRONT_N_ROWS, shape, 64)
        stock_box = {}

        def stock_case():
            if layer:
                b = rows
                xq_lat = torch.mm(x, down_full.t())
                xq, sq = stock.quant(xq_lat)
                logits = torch.mm(x, router_full.t(), out_dtype=torch.float32)
                try:
                    stock.tune(("mono", b), lambda: stock.mono(xq, sq, logits,
                                                               out_stock[:b]))  # fmt: skip
                    stock_layer.mono = True
                    arm = "gate+down+shared+mxfp8+fi-mono"
                except Exception as error:  # noqa: BLE001
                    log(f"stock mono failed ({error}); grouped top-k + routed")
                    raise_if_poisoned(error)
                    stock_layer.mono = False
                    w, ids = ops.grouped_topk(logits, 1, 1, TOPK, True, 1.0, bias, 1)
                    stock.tune(("routed", b), lambda: stock.routed(
                        xq, sq, ids, w.bfloat16(), out_stock[:b]))  # fmt: skip
                    arm = "gate+down+shared+mxfp8+topk+fi-routed"
                us, how = time_region(lambda: stock_layer(x), dist=d,
                                      samples=args.samples)  # fmt: skip
            else:
                us, how = time_region(lambda: stock_front(x, False), dist=d,
                                      samples=args.samples)  # fmt: skip
                us2, _ = time_region(lambda: stock_front(x, True), dist=d,
                                     samples=args.samples)  # fmt: skip
                stock_box["shared"] = us2
                arm = "gate(fp32)+latent-down+grouped_topk"
            stock_box["us"], stock_box["arm"], stock_box["how"] = us, arm, how
            case = "layer" if layer else "route"
            res.add(dict(case=case, q=q, rows=rows, variant="stock", prod="0",
                         stock_us=us, tactic=how, stock_arm=arm))  # fmt: skip
            if not layer:
                res.add(dict(case="route+shared", q=q, rows=rows, variant="stock",
                             prod="0", stock_us=stock_box["shared"], tactic=how,
                             stock_arm=arm + "+shared"))  # fmt: skip

        run_case(d, res, dog, f"stock:M{rows}", stock_case)
        if layer:
            variants = [("prod", {}, True)]
            if not fr.rows_policy(fr.MERGED_ROWS, shape, 0):  # fork: unmerged rows
                for f in fork_variants:
                    variants.append((f"fork{f}", dict(SHARED_FORK_ROWS={"*": f"65-:{f}"}),
                                     False))  # fmt: skip
        else:
            variants = [("prod", {}, True)] + [
                (f"n{n}", dict(FRONT_N_ROWS={"*": f"65-:{n}"}), False)
                for n in n_variants if n != prod_n
            ]  # fmt: skip
        for name, table, is_prod in variants:

            def spec_case(name=name, table=table, is_prod=is_prod):
                with patched(fr, **table):
                    front = fr.FrontRows(
                        moe, packed, box, state.front, [rows], BETA, LINEAR_BETA,
                        plan=state.plan, q=q,
                    )  # fmt: skip
                shared_out = state.shared_partial[:rows]
                merged = front.merged_batch(rows)
                n = fr.rows_policy(fr.FRONT_N_ROWS, shape, 64) if not table.get(
                    "FRONT_N_ROWS") else int(name[1:])  # fmt: skip
                fork = front.fork.get(rows, 0)
                early = front.early.get(rows, 0)
                down_n = front.down_sms.get(rows, 0) if push_args is not None else 0
                if layer and front.down_sms.get(rows):
                    # production (moe.prepared): the experts' grid leaves SharedDown's SMs
                    experts.grids[rows] = min(
                        experts.grid_for(rows), sms - front.down_sms[rows]
                    )  # (idempotent across variants)
                tactic = (f"n{n} merged {int(merged)} early {early} fork {fork} "
                          f"{f'SharedDown {down_n} SMs ' if down_n else ''}"
                          f"(world {world}+{TP - world} emulated)")  # fmt: skip
                if front.down_sms.get(rows) and not down_n:
                    tactic += " [cuBLAS down: no tail box, production runs SharedDown]"

                def shared_part():
                    """The shared expert as production runs it beside the experts."""
                    if down_n:  # SharedDown: the down rows pushed to the RS slots
                        front.push_down(x, push_args)
                    else:
                        front.shared_expert(x, shared_out)

                def route_only():
                    front.route(x)

                def with_shared():
                    if merged or fork:
                        front.route(x)
                        with beside(shared_part):
                            pass
                    else:
                        with beside(shared_part):
                            front.route(x)

                wide = wide_for(rows) if layer else None

                def layer_run():
                    def shared():
                        shared_part()

                    if early:
                        front.front(x)
                        with beside(shared) as join:
                            ids, w, lat = front.route_only(x)
                            if early in (2, 4):
                                join()
                            wide.run(lat, w, ids)
                        return
                    if merged or fork == 2:
                        ids, w, lat = front.route(x)
                    elif fork == 1:
                        front.front(x)
                    with beside(shared):
                        if fork == 1:
                            ids, w, lat = front.route_only(x)
                        elif not merged and not fork:
                            ids, w, lat = front.route(x)
                        wide.run(lat, w, ids)

                pre = lambda: prefill_box(rows)  # noqa: E731
                regions = (
                    [("layer", layer_run)] if layer
                    else [("route", route_only), ("route+shared", with_shared)]
                )  # fmt: skip
                for case, fn in regions:
                    us, how = time_region(fn, prefill=pre, dist=d, samples=args.samples)
                    ref = stock_box.get("shared" if case == "route+shared" else "us")
                    prev = PREV.get(("front", case), {}).get(rows) if (
                        is_prod and not layer) else None  # fmt: skip
                    res.add(dict(case=case, q=q, rows=rows, variant=name,
                                 prod="1" if is_prod else "0", spec_us=us,
                                 stock_us=ref or "", ratio=us / ref if ref else "",
                                 tactic=f"{tactic} [{how}]",
                                 stock_arm=stock_box.get("arm", ""),
                                 prev_us=prev or ""))  # fmt: skip

            run_case(d, res, dog, f"spec:{name}:M{rows}", spec_case)


# --------------------------------------------------------------------------------------
# Section: tail


def _tail_impl(args, d, res, dog):
    import torch
    import torch.nn.functional as F

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.policy import (
        batch_table,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        tail_rows as tr,
    )

    dev, world, rank = d.device, d.world, d.rank
    rows_all = parse_rows(args.rows, ROWS_DIST)
    cap = max(rows_all)
    gen = torch.Generator(device=dev).manual_seed(200 + rank)
    bf = dict(dtype=torch.bfloat16, device=dev)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=dev) * scale).bfloat16()

    box = tr.TailRowsBox(d.group, dev, TP, rank, cap)
    patch_twoshot(box.latent, world, dev)
    rs_dummy = torch.empty_like(box.rs.buffer)
    real = list(box.rs.handle.buffer_ptrs)
    box.rs_peers = torch.tensor(
        [int(real[o]) if o < world else rs_dummy.data_ptr() for o in range(TP)],
        dtype=torch.int64, device=dev,
    )  # fmt: skip
    up = rnd(H, LATENT, scale=LATENT**-0.5)
    gamma = (1 + 0.1 * torch.randn(LATENT, generator=gen, device=dev)).bfloat16()
    eps = 1e-6
    partial_all = rnd(cap, LATENT, scale=0.05)
    shared_all = rnd(cap, H, scale=0.05)
    width, hs = LATENT // TP, H // TP
    recv_pat = rnd(TP - world, cap, width, scale=0.05)
    gather_pat = rnd(cap, LATENT - width * world, scale=0.5)
    rs_pat = rnd(TP - world, cap, hs, scale=0.05)
    parity = 0

    def prefill(b):
        box.latent.recv[parity, world:, :b].copy_(recv_pat[:, :b])
        box.latent.gather[parity, 0, :b, width * world :].copy_(gather_pat[:b])
        box.rs.buffer[parity, world:, :b].copy_(rs_pat[:, :b])

    beside = beside_factory(dev)
    lat_ws, sh_ws = torch.empty(cap, LATENT, **bf), torch.empty(cap, H, **bf)

    for rows in rows_all:
        b = rows
        partial, shared = partial_all[:b], shared_all[:b]
        stock_box = {}

        def stock_case():
            def stock():
                lw, sw = lat_ws[:b], sh_ws[:b]
                lw.copy_(partial)
                d.dist.all_reduce(lw, group=d.group)
                xn = F.rms_norm(lw, (LATENT,), gamma, eps)
                proj = torch.mm(xn, up.t())
                sw.copy_(shared)
                d.dist.all_reduce(sw, group=d.group)
                return proj.add_(sw)

            us, how = time_region(stock, dist=d, samples=args.samples)
            stock_box["us"] = us
            res.add(dict(case="tail", rows=rows, variant="stock", prod="0",
                         stock_us=us, tactic=how,
                         stock_arm=f"nccl-AR x2 ({world} ranks)+rms+up"))  # fmt: skip

        run_case(d, res, dog, f"stock:M{rows}", stock_case)
        polled = tr.norm_polled(TP, rows)
        up_fused = bool(batch_table(tr.UP_ROWS.get(TP, "")).get(rows))
        prod_chunk = tr.latent_chunk(rows)
        variants = [("prod", {}, None, True)]
        variants.append(("norm-" + ("wait" if polled else "polled"),
                         dict(NORM_POLLED={TP: "" if polled else "65-:1"}), None,
                         False))  # fmt: skip
        variants.append(("up-" + ("cublas" if up_fused else "uptmem"),
                         dict(UP_ROWS={TP: "" if up_fused else "65-:1"}), None, False))
        for c in (2, 4, 8):
            if c != prod_chunk and not args.quick:
                variants.append((f"chunk{c}", {}, c, False))
        prev = PREV[("tail", "tail")].get(rows)
        for name, table, chunk, is_prod in variants:

            def spec_case(name=name, table=table, chunk=chunk, is_prod=is_prod):
                tr.compile_rows.cache_clear()  # (keyed without the policy tables)
                with patched(tr, **table):
                    tail = tr.PreparedTailRows(
                        TP, rank, up, gamma, eps, [b], box, 0, chunk=chunk,
                        experts_push=(),
                    )  # fmt: skip
                    np_ = tr.norm_polled(TP, b)
                fused = b in tail.fused_rows
                tactic = (f"latent chunk {chunk or tr.latent_chunk(b)} norm "
                          f"{'polled' if np_ else 'wait'} up "
                          f"{'uptmem' if fused else 'cublas'} side push "
                          f"(world {world}+{TP - world} emulated)")  # fmt: skip
                if b <= WIDE_MAX and batch_table(tr.EXPERTS_PUSH.get(TP, "")).get(b):
                    tactic += (" (proxy: production EXPERTS_PUSH, the experts push and "
                               "LatentRows only reduces; unpushed LatentRows here)")

                def run():
                    with beside(lambda: tail.push(shared)) as join:
                        tail.run(partial, b, join)

                us, how = time_region(run, prefill=lambda: prefill(b), dist=d,
                                      samples=args.samples)  # fmt: skip
                ref = stock_box.get("us")
                res.add(dict(case="tail", rows=rows, variant=name,
                             prod="1" if is_prod else "0", spec_us=us,
                             stock_us=ref or "", ratio=us / ref if ref else "",
                             tactic=f"{tactic} [{how}]",
                             stock_arm=f"nccl-AR x2 ({world} ranks)+rms+up",
                             prev_us=prev if is_prod and prev else ""))  # fmt: skip

            run_case(d, res, dog, f"spec:{name}:M{rows}", spec_case)
        tr.compile_rows.cache_clear()


# --------------------------------------------------------------------------------------
# Section: oproj


def _oproj_impl(args, d, res, dog):
    import cutlass
    import torch

    from vllm.models.kimi_k3.nvidia.ops.attn_res import attn_res
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        dynamic_view,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.twoshot import (
        TwoShot,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (
        weight_map,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj import (
        prepared as op,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Shape

    dev, world, rank = d.device, d.world, d.rank
    rows_all = parse_rows(args.rows, ROWS_DIST)
    cap = -(-max(rows_all) // 128) * 128
    k = 96 * 128 // TP  # o_proj input columns per rank (96 heads x 128, TP16)
    nb, nblocks, eps = 4, 8, 1e-6
    gen = torch.Generator(device=dev).manual_seed(300 + rank)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=dev) * scale).bfloat16()

    weight = rnd(H, k, scale=k**-0.5)
    x_buffer = rnd(cap, k, scale=1.0)
    prefix0, blocks = rnd(cap, H), rnd(cap, nblocks, H)
    out = torch.empty(cap, H, dtype=torch.bfloat16, device=dev)
    norm = (1 + 0.1 * torch.randn(H, generator=gen, device=dev)).bfloat16()
    query = rnd(H, scale=H**-0.5)
    onorm = (1 + 0.1 * torch.randn(H, generator=gen, device=dev)).bfloat16()
    views = tuple(to_cute(w.reshape(-1)) for w in (norm, query, onorm))
    epsv = (cutlass.Float32(eps), cutlass.Float32(eps))
    box = TwoShot(d.group, dev, TP, rank, H, cap, parities=2)
    patch_twoshot(box, world, dev)
    parity = 0
    width = H // TP
    recv_pat = rnd(TP - world, cap, width, scale=0.05)
    gather_pat = rnd(cap, H - width * world, scale=0.05)

    def prefill(b):
        box.recv[parity, world:, :b].copy_(recv_pat[:, :b])
        box.gather[parity, 0, :b, width * world :].copy_(gather_pat[:b])

    red = torch.empty(cap, H, dtype=torch.bfloat16, device=dev)
    for rows in rows_all:
        b = rows
        prefix = prefix0[:b].clone()
        x = x_buffer[:b]
        stock_box = {}

        def stock_case():
            def stock():
                torch.mm(x, weight.t(), out=red[:b])
                d.dist.all_reduce(red[:b], group=d.group)
                attn_res(prefix, red[:b], blocks[:b], norm, query, onorm,
                         num_blocks=nb, block_write_idx=-1, eps=eps,
                         output_norm_eps=eps)  # fmt: skip

            us, how = time_region(stock, dist=d, samples=args.samples)
            stock_box["us"] = us
            res.add(dict(case="oproj+attnres", rows=rows, variant="stock", prod="0",
                         stock_us=us, tactic=how,
                         stock_arm=f"cublas+nccl-AR({world})+ops.attn_res"))  # fmt: skip

        run_case(d, res, dog, f"stock:M{rows}", stock_case)
        prod_chunk = op.wide_chunk(TP, Shape(b))
        chunks = sorted({prod_chunk, 64, 96, 128} | ({80} if b <= 80 else set()))
        for chunk in chunks:
            is_prod = chunk == prod_chunk
            name = f"chunk{chunk}"

            def spec_case(chunk=chunk, is_prod=is_prod, name=name):
                shape = (k, b, TP, nb, True, False, False, cap, True)
                producer = op.compile_oproj("producer", *shape, chunk=chunk)
                consumer = op.compile_oproj(
                    "consumer", *shape, False, False, False, False, True
                )
                tile = min(-(-b // chunk) * chunk, cap)
                tma = weight_map(x_buffer[:tile], chunk)
                recv, peers, gather, gather_mc = box.views(parity)
                recv, peers, gather = to_cute(recv), to_cute(peers), to_cute(gather)
                head = (recv, peers, cutlass.Int32(rank), gather,
                        cutlass.Int64(gather_mc), cutlass.Int64(tma.data_ptr()))  # fmt: skip
                tail = (recv, cutlass.Int64(gather_mc), cutlass.Int32(rank))
                wview = to_cute(weight)

                def run():
                    s = stream(dev)
                    rows_ = (dynamic_view(prefix), dynamic_view(blocks[:b]), *views)
                    rows_ += (dynamic_view(out[:b]), None, None, *epsv)
                    producer(to_cute(x), wview, *head, s)
                    consumer(gather, *rows_, *tail, s)

                us, how = time_region(run, prefill=lambda: prefill(b), dist=d,
                                      samples=args.samples)  # fmt: skip
                ref = stock_box.get("us")
                res.add(dict(case="oproj+attnres", rows=rows, variant=name,
                             prod="1" if is_prod else "0", spec_us=us,
                             stock_us=ref or "", ratio=us / ref if ref else "",
                             tactic=f"OprojUmmaRows chunk {chunk} producer-reduce, "
                             f"k {k} nb {nb} (world {world}+{TP - world} emulated) "
                             f"[{how}]", stock_arm=f"cublas+nccl-AR({world})+attn_res"))  # fmt: skip
                del tma

            run_case(d, res, dog, f"spec:{name}:M{rows}", spec_case)


# --------------------------------------------------------------------------------------
# Self-test (pure python)


def selftest():
    for rows in range(65, 2049):
        for fn in (equal_chunks, maxrem_chunks):
            ch = fn(rows)
            assert sum(n for _, n in ch) == rows, (fn.__name__, rows, ch)
            assert all(s % 64 == 0 for s, _ in ch), (fn.__name__, rows, ch)
            assert all(n <= WIDE_MAX for _, n in ch), (fn.__name__, rows, ch)
            if len(ch) > 1:
                assert all(n > 64 for _, n in ch), (fn.__name__, rows, ch)
            starts = [s for s, _ in ch]
            assert starts == [sum(n for _, n in ch[:i]) for i in range(len(ch))]
    assert maxrem_chunks(832) == ((0, 704), (704, 128)), maxrem_chunks(832)
    assert maxrem_chunks(2048) == ((0, 768), (768, 768), (1536, 512))
    assert equal_chunks(2048) == ((0, 704), (704, 704), (1408, 640))
    assert parse_rows("72,128-256", ()) == (72, 128, 192, 256)
    assert parse_rows("", ROWS) == ROWS
    assert all(r % 8 == 0 for r in ROWS) and all(r % 8 == 0 for r in ROWS_DIST)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        r = Results(tmp, "experts", "")
        r.start("a")
        r.add(dict(case="routed", q="-", rows=128, variant="prod", prod="1",
                   spec_us=50.0, stock_us=40.0, ratio=1.25, tactic="w16"))  # fmt: skip
        r.add(dict(case="routed", q="-", rows=128, variant="w8", prod="0",
                   spec_us=45.0, stock_us=40.0, ratio=1.125, tactic="w8"))  # fmt: skip
        r.end("a")
        r.start("b")  # dies: HANG on the next attempt
        rs = Results(tmp, "experts", "-req4")  # (pass 2 reads pass 1's stock time)
        rs.add(dict(case="routed", q="4", rows=832, variant="stock:routed", prod="0",
                    stock_us=77.5, tactic="graph"))  # fmt: skip
        assert _stock_routed_from_csv(rs, "4", 832) == 77.5
        assert _stock_routed_from_csv(rs, "4", 1024) is None
        r2 = Results(tmp, "experts", "")
        assert "b" in r2.done and "b" in r2.hung and "a" not in r2.hung
        worse, tune, fails = summarize(tmp)
        assert len(worse) == 1 and len(tune) == 1 and len(fails) == 1, (
            worse, tune, fails)  # fmt: skip
    print("selftest OK")


# --------------------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("section", choices=("experts", "attnres", "front", "layer", "tail",
                                       "oproj", "summary", "selftest"))  # fmt: skip
    p.add_argument("--out", default="out")
    p.add_argument("--rows", default="", help="e.g. 72,128,1024 or 128-2048 (step 64)")
    p.add_argument("--q", default="4", help="front/layer: verify length (rows divisible by q)")
    p.add_argument("--routes", default="uniform,req4,req8",
                   help="experts: uniform | req<q> (request-local routes)")
    p.add_argument("--samples", type=int, default=25)
    p.add_argument("--case-timeout", type=int, default=900)
    p.add_argument("--tag", default="", help="CSV / journal name suffix")
    p.add_argument("--quick", action="store_true", help="fewer variants")
    p.add_argument("--no-stock-autotune", action="store_true")
    p.add_argument("--stock-mono", type=int, default=1)
    args = p.parse_args()
    if args.section == "selftest":
        return selftest()
    if args.section == "summary":
        summarize(args.out)
        return
    if args.section == "experts":
        return _experts_impl(args)
    if args.section == "attnres":
        return _attnres_impl(args)
    return section_dist(args)


if __name__ == "__main__":
    main()
