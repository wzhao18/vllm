# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drafter projection GEMMs at the TP16 per-rank shapes (1 GPU, Rubin SM107):
the specialized decode's split-K tactic tables (runtime.draft PROJECTIONS /
PROJECTIONS_WIDE / DRAFT_168 / DRAFT_96; cutedsl.gemm.projection.PreparedProjection,
exactly as ``runtime.draft.projection`` builds it) and the drafter LM-head shard
(cutedsl.verify.lmhead: GEMV at 1-7 rows, the evict-first wide GEMM at 10-64 rows)
against stock cuBLAS (torch.nn.functional.linear: the unquantized BF16 linear method),
at every drafter row count in scope (query rows B x K, context rows B x (K + 1), q4
K = 3 and q8 K = 7, B = 1..256: up to 1792 query / 2048 context rows).

Weights cold: rotating weight copies (> 640 MB per case) in one CUDA graph (so each
call streams its weight from HBM, as in the 5-layer drafter step). The split-K
kernels launch with PDL, back to back in the graph as in the drafter chain.

Part 1 ("chosen"): per (linear, rows): the production choice (the table's tactic, or
cuBLAS where the table has none: ratio 1 by construction, both timed) vs cuBLAS.
ctx_shard's stock is the replicated context_proj GEMM (7168, 35840) it replaces (the
CtxGather exchange is not included). Old-drafter shapes (q_b_old, o_proj_old) give
the previous-geometry reference.
Part 2 ("sweep", NQ-TUNE 4): candidate split-K tactics (M tile 64, N tile 8-64,
every split dividing the K tiles, 4 / 8 / 14 stages; the tensor-memory tactics of the
tables at 33-512 rows) at every row count up to --sweep-max rows; per (linear, row
band) the tactic with the best geometric-mean time over the band's rows.

Resume: every measurement is appended to ``raw.csv`` (linear, rows, arm, us) and the
arm about to run is written to ``inflight.txt``. A sticky CUDA error ends the process
with common.STICKY_EXIT; restarted with the same --out, the bench reloads raw.csv and
the tables' CSVs, records the in-flight arm as crashed (never retried), and continues.

Usage: python bench_draft_gemm.py --out DIR [--part chosen,sweep] [--budget S]
"""

import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    LINEARS,
    LM_HEAD,
    STOCK_SHAPE,
    Deadline,
    Table,
    band,
    copies_for,
    guarded,
    log,
    rows_of,
    time_graph,
)

TMEM_TACTICS = ((128, 64, 6, 16, 14), (128, 64, 6, 14, 14))
# A sweep tactic is skipped for the rest of a K after this many failures at distinct
# row counts (a failure specific to one row count does not hide the others).
FAIL_LIMIT = 3


class Raw:
    """raw.csv (linear, rows, arm, us; us empty = failed) + inflight.txt: the resume
    state of one --out dir."""

    def __init__(self, out):
        self.path = os.path.join(out, "raw.csv")
        self.inflight = os.path.join(out, "inflight.txt")
        self.recs = []
        if os.path.exists(self.path):
            with open(self.path, newline="") as fh:
                for r in csv.DictReader(fh):
                    us = float(r["us"]) if r["us"] else None
                    self.recs.append((r["linear"], int(r["rows"]), r["arm"], us))
        new = not os.path.exists(self.path)
        self.fh = open(self.path, "a", newline="")
        self.w = csv.writer(self.fh)
        if new:
            self.w.writerow(["linear", "rows", "arm", "us"])
            self.fh.flush()
        if os.path.exists(self.inflight):
            with open(self.inflight) as fh:
                parts = fh.read().strip().split(",", 2)
            if len(parts) == 3:
                log(f"resume: {parts} was in flight at the crash: recorded as failed")
                self.add(parts[0], int(parts[1]), parts[2], None)
            os.remove(self.inflight)

    def add(self, linear, rows, arm, us):
        self.recs.append((linear, rows, arm, us))
        self.w.writerow([linear, rows, arm, "" if us is None else f"{us:.4f}"])
        self.fh.flush()

    def get(self, linear, rows, arm):
        """(found, us) of the latest record."""
        for n, r, a, us in reversed(self.recs):
            if (n, r, a) == (linear, rows, arm):
                return True, us
        return False, None

    def run(self, linear, rows, arm, fn, *args):
        """The recorded result, else ``guarded(fn)`` with the in-flight marker set."""
        found, us = self.get(linear, rows, arm)
        if found:
            return us
        with open(self.inflight, "w") as fh:
            fh.write(f"{linear},{rows},{arm}")
        us = guarded(log, f"{linear} r{rows} {arm}", fn, *args)
        os.remove(self.inflight)
        self.add(linear, rows, arm, us)
        return us


def tname(t):
    return "splitk" + ".".join(map(str, t))


def fused_note(draft, name, rows):
    """Where production does not run the plain table GEMM at these rows (the bench
    times the GEMM core only)."""
    shape = LINEARS.get(name, ((0, 0),))[0]
    if rows <= draft.PROJECTION_ROWS and shape in draft.RING_PROJECTIONS.get(16, {}):
        return "prod: ring producer (RING_PROJECTIONS) - GEMM core only"
    if name in ("o_proj", "down", "o_proj_old") and draft.batch_table(
        draft.TWO_SHOT_PRODUCERS.get(16, "")
    ).get(rows):
        return "prod: two-shot producer (TWO_SHOT_PRODUCERS) - GEMM core only"
    if name == "gate_up" and draft.batch_table(draft.TWO_SHOT_SILU.get(16, "")).get(rows):
        return "prod: SiLU-fused gate/up (TWO_SHOT_SILU) - GEMM core only"
    return ""


def candidates(shape, rows, table_tactics):
    """Split-K tactics to sweep for (N, K) at ``rows`` (deduplicated)."""
    n_out, k = shape
    k_tiles = k // 128
    if rows <= 8:
        ntiles = (8,)
    elif rows <= 16:
        ntiles = (8, 16)
    elif rows <= 32:
        ntiles = (16, 32)
    elif rows <= 64:
        ntiles = (32, 64)
    else:
        ntiles = (64,)
    out = []
    for n in ntiles:
        for split in (1, 2, 3, 4, 6, 7, 8, 12, 14, 16):
            if k_tiles % split:
                continue
            per = k_tiles // split
            for stages in sorted({min(per, s) for s in (4, 8, 14)}):
                if stages * (64 + n) * 256 > 200 * 1024:  # (A + B stage bytes)
                    continue
                out.append((64, n, split, stages))
    if k == 7168 and 33 <= rows <= 64:  # (SplitKTmemGemm64: one 64-row x tile)
        out += list(TMEM_TACTICS)
    for t in table_tactics:
        if len(t) == 4 and k % (128 * t[2]) == 0:
            out.append(tuple(t))
        elif len(t) > 4 and k == 7168 and 33 <= rows <= 64:
            out.append(tuple(t))
    seen, result = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


class Weights:
    """Rotating cold copies of one (N, K) BF16 weight."""

    def __init__(self, shape, device, copies=None, gen=None):
        import torch

        n, k = shape
        count = copies or copies_for(n * k * 2)
        self.ws = [
            (torch.randn((n, k), device=device, generator=gen) * 0.02).bfloat16()
            for _ in range(count)
        ]
        self.shape = shape


def table_tactics(draft, shape):
    """Every tactic any of runtime.draft's tables lists for (N, K)."""
    out = []
    t = draft.PROJECTIONS.get(16, {}).get(shape)
    if t:
        out.append(t)
    for _, tab in draft.PROJECTIONS_WIDE.get(16, ()):
        if shape in tab:
            out.append(tab[shape])
    return out


def chosen_tactic(draft, shape, rows):
    """The production tactic at ``rows`` (runtime.draft.projection's lookup)."""
    if rows <= draft.PROJECTION_ROWS:
        return draft.PROJECTIONS.get(16, {}).get(shape)
    bins = draft.PROJECTIONS_WIDE.get(16, ())
    tab = next((t for top, t in bins if rows <= top), {})
    return tab.get(shape)


def run_cublas(weights, x, out):
    import torch.nn.functional as F

    fns = [lambda w=w: F.linear(x, w) for w in weights.ws]
    return time_graph(fns)


def run_tactic(weights, x, out, tactic, max_copies=None):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (
        PreparedProjection,
    )

    rows = x.shape[0]
    ws = weights.ws[:max_copies] if max_copies else weights.ws
    preps = [PreparedProjection(w, {rows: tactic}).prepare() for w in ws]
    fns = [lambda p=p: p.run(x, out) for p in preps]
    return time_graph(fns)


def lm_choice(draft, lmhead, rows):
    """runtime.draft FAST_LM_HEAD (GEMV, <= 8 rows) / WIDE_LM_HEAD rows at TP16."""
    if rows <= lmhead.TOKENS and draft.batch_table(draft.FAST_LM_HEAD[16]).get(rows):
        return "gemv"
    if rows in lmhead.WIDE_ROWS and draft.batch_table(draft.WIDE_LM_HEAD[16]).get(rows):
        return "wide"
    return None


def run_lm(weights, x, kind):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.verify import lmhead

    if kind == "gemv":
        preps = [lmhead.PreparedLmHeadLogits(w, w.device) for w in weights.ws]
        outs = [p.run(x) for p in preps]
        fns = [lambda p=p, o=o: p.run(x, o) for p, o in zip(preps, outs)]
    else:
        preps = [lmhead.PreparedWideLmHead(w, w.device) for w in weights.ws]
        fns = [lambda p=p: p.run(x) for p in preps]
    return time_graph(fns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="chosen,sweep")
    ap.add_argument("--linears", default=",".join([*LINEARS, "lm_head"]))
    ap.add_argument("--sweep-max", type=int, default=512)
    ap.add_argument("--sweep-copies", type=int, default=48)
    ap.add_argument("--budget", type=float, default=0, help="soft seconds (0: none)")
    args = ap.parse_args()
    import torch

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.verify import lmhead
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev)
    log(f"device {props.name} SMs {props.multi_processor_count} cc {props.major}.{props.minor}")
    gen = torch.Generator(device=dev).manual_seed(0)
    deadline = Deadline(args.budget)
    names = [n for n in args.linears.split(",") if n]
    parts = args.part.split(",")
    specs = dict(LINEARS)
    specs["lm_head"] = (LM_HEAD, "q", "both")

    # ---------------------------------------------------------------- chosen
    sweep_results = {}  # (name, rows) -> {tactic: us}
    cublas = {}  # (name, rows) -> us
    raw = Raw(args.out)
    for n, r, arm, us in raw.recs:  # resume: what an earlier (crashed) run measured
        if us is None:
            continue
        if arm == "cublas":
            cublas[n, r] = us
        elif arm.startswith("splitk"):
            sweep_results.setdefault((n, r), {})[
                tuple(int(v) for v in arm[6:].split("."))
            ] = us
    if "chosen" in parts:
        tab = Table(os.path.join(args.out, "gemm_chosen.csv"), extra_cols=["band"],
                    resume=True)  # fmt: skip
        for name in names:
            shape, kind, which = specs[name]
            w = guarded(log, f"{name} weights", Weights, shape, dev, gen=gen)
            stock_w = None
            if name in STOCK_SHAPE:
                stock_w = guarded(log, f"{name} stock weights", Weights,
                                  STOCK_SHAPE[name], dev, gen=gen)  # fmt: skip
            if w is None:
                continue
            for rows, labels in rows_of(kind).items():
                if (name, rows) in tab.done:
                    continue
                if deadline.out():
                    log(f"budget out: skip {name} rows {rows}")
                    continue
                x = (torch.randn((rows, shape[1]), device=dev, generator=gen)).bfloat16()
                out = torch.empty((rows, shape[0]), dtype=torch.bfloat16, device=dev)
                base = raw.run(name, rows, "cublas", run_cublas, w, x, out)
                if base is not None:
                    cublas[name, rows] = base
                stock, note = base, "cuBLAS same shape"
                if stock_w is not None:
                    stock = raw.run(name, rows, "stock", run_cublas, stock_w, x, None)
                    note = f"stock = replicated {STOCK_SHAPE[name]} cuBLAS"
                if name == "lm_head":
                    kind_lm = lm_choice(draft, lmhead, rows)
                    if kind_lm:
                        spec = raw.run(name, rows, f"lmhead.{kind_lm}", run_lm, w, x,
                                       kind_lm)  # fmt: skip
                        chosen = f"lmhead.{kind_lm}"
                    else:
                        spec, chosen = base, "cuBLAS"
                else:
                    tactic = chosen_tactic(draft, shape, rows)
                    if tactic is not None:
                        tactic = tuple(tactic)
                        spec = raw.run(name, rows, tname(tactic), run_tactic, w, x,
                                       out, tactic)  # fmt: skip
                        chosen = tname(tactic)
                        if spec is not None:
                            sweep_results.setdefault((name, rows), {})[tactic] = spec
                        fused = fused_note(draft, name, rows)
                        if fused:
                            note = f"{note}; {fused}"
                    else:
                        spec, chosen = base, "cuBLAS"
                tab.add(name, rows, spec, stock, chosen, note, " ".join(labels),
                        band=band(rows))  # fmt: skip
            del w, stock_w
            torch.cuda.empty_cache()
        tab.finish("chosen tactic vs stock")

    # ----------------------------------------------------------------- sweep
    if "sweep" in parts:
        tab = Table(os.path.join(args.out, "gemm_sweep.csv"),
                    extra_cols=["band", "best_tactic", "best_us", "cublas_us"],
                    resume=True)  # fmt: skip
        failed = {}  # (tactic, K) -> row counts it failed at
        for n, r, arm, us in raw.recs:
            if us is None and arm.startswith("splitk") and n in specs:
                t = tuple(int(v) for v in arm[6:].split("."))
                failed.setdefault((t, specs[n][0][1]), set()).add(r)
        for name in names:
            shape, kind, _ = specs[name]
            w = guarded(log, f"{name} weights", Weights, shape, dev, gen=gen)
            if w is None:
                continue
            listed = table_tactics(draft, shape)
            if name == "lm_head":
                listed.append(lmhead.WIDE_TACTIC)
            for rows, labels in rows_of(kind).items():
                if rows > args.sweep_max or (name, rows) in tab.done:
                    continue
                if deadline.out():
                    log(f"budget out: skip sweep {name} rows {rows}")
                    continue
                x = (torch.randn((rows, shape[1]), device=dev, generator=gen)).bfloat16()
                out = torch.empty((rows, shape[0]), dtype=torch.bfloat16, device=dev)
                base = cublas.get((name, rows))
                if base is None:
                    base = raw.run(name, rows, "cublas", run_cublas, w, x, out)
                    if base is not None:
                        cublas[name, rows] = base
                res = sweep_results.setdefault((name, rows), {})
                t0 = time.time()
                for t in candidates(shape, rows, listed):
                    t = tuple(t)
                    fails = failed.get((t, shape[1]), set())
                    if t in res or rows in fails or len(fails) >= FAIL_LIMIT:
                        continue
                    us = raw.run(name, rows, tname(t), run_tactic, w, x, out, t,
                                 args.sweep_copies)  # fmt: skip
                    if us is None:
                        failed.setdefault((t, shape[1]), set()).add(rows)
                    else:
                        res[t] = us
                if not res:
                    continue
                best = min(res, key=res.get)
                chosen = chosen_tactic(draft, shape, rows) if name != "lm_head" else None
                chosen = tuple(chosen) if chosen else None
                spec = res.get(chosen) if chosen else base
                tab.add(f"{name}", rows, spec, base,
                        tname(chosen) if chosen else
                        ("lmhead: see gemm_chosen" if name == "lm_head" else "cuBLAS"),
                        f"{len(res)} tactics {time.time() - t0:.0f}s", " ".join(labels),
                        band=band(rows), best_tactic=".".join(map(str, best)),
                        best_us=res[best], cublas_us=base)  # fmt: skip
            del w
            torch.cuda.empty_cache()
        tab.finish("sweep: production choice vs cuBLAS")
        # Per (linear, band): the tactic with the best geometric mean over the band's
        # rows (only tactics measured at every row of the band), vs cuBLAS.
        path = os.path.join(args.out, "gemm_tune_bands.csv")
        with open(path, "w") as fh:
            fh.write("linear,band,rows,best_tactic,geo_us,cublas_geo_us,ratio,"
                     "table_tactic,table_geo_us\n")  # fmt: skip
            print("\n==== TUNE per (linear, band): best tactic (geo-mean over rows) vs cuBLAS")
            for name in names:
                bands = {}
                for (n, rows), res in sweep_results.items():
                    if n == name and res and cublas.get((n, rows)):
                        bands.setdefault(band(rows), []).append(rows)
                for b, rws in bands.items():
                    common_t = set.intersection(
                        *(set(sweep_results[name, r]) for r in rws)
                    )
                    if not common_t:
                        continue

                    def geo(vals):
                        return math.exp(sum(math.log(v) for v in vals) / len(vals))

                    scores = {t: geo([sweep_results[name, r][t] for r in rws])
                              for t in common_t}  # fmt: skip
                    best = min(scores, key=scores.get)
                    cub = geo([cublas[name, r] for r in rws])
                    shape = specs[name][0]
                    ct = chosen_tactic(draft, shape, max(rws)) if name != "lm_head" else None
                    ctg = scores.get(tuple(ct)) if ct else None
                    line = (f"{name},{b},{' '.join(map(str, sorted(rws)))},"
                            f"{'.'.join(map(str, best))},{scores[best]:.2f},{cub:.2f},"
                            f"{scores[best] / cub:.3f},"
                            f"{'.'.join(map(str, ct)) if ct else 'cuBLAS'},"
                            f"{'' if ctg is None else f'{ctg:.2f}'}")  # fmt: skip
                    fh.write(line + "\n")
                    print("TUNE " + line, flush=True)
    log("gemm bench done")


if __name__ == "__main__":
    main()
