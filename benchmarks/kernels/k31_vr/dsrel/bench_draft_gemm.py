# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The release DSpark drafter's projection GEMMs at their per-rank shapes (1 GPU,
Rubin SM107): stock cuBLAS (torch.nn.functional.linear, the unquantized BF16 linear
method) against the specialized decode's split-K projections
(cutedsl.gemm.projection.PreparedProjection, as ``runtime.draft.projection`` builds
them), at the drafter row counts of the chosen deployments (query rows B x K,
context rows B x (K + 1)), and the preview drafter's matching pieces for reference.

Inferact/Kimi-K3.1-Pro-DSpark per rank (hidden 7168, 96 / 16 heads of 64, FFN 14336,
5 layers, fc 35840 -> 7168):

    linear     TP16           TP8            TP4
    qkv        (512, 7168)    (1024, 7168)   (2048, 7168)
    o_proj     (7168, 384)    (7168, 768)    (7168, 1536)
    gate_up    (1792, 7168)   (3584, 7168)   (7168, 7168)
    down       (7168, 896)    (7168, 1792)   (7168, 3584)
    ctx_shard  (448, 35840)   (896, 35840)   (1792, 35840)
    ctx_kv     (640, 7168)    (1280, 7168)   (2560, 7168)

Weights cold: rotating copies (> 640 MB per case) replayed in one CUDA graph, so every
call streams its weight from HBM as in the 5-layer drafter step. ``--part chosen``
times the production choice (``runtime.draft`` GQA tables) vs cuBLAS at every row
count; ``--part sweep`` times split-K candidates (M tile 64, N tile 8-64, every split
dividing the K tiles; the tensor-memory tactics at 33-64 rows) at rows up to
``--sweep-max`` and prints, per (linear, band), the best tactic by geometric mean.

Usage: python bench_draft_gemm.py --out DIR --tp 16 [--part chosen,sweep]
           [--linears qkv,o_proj,...] [--budget S]
"""

import argparse
import csv
import math
import os
import sys
import time
import traceback

HIDDEN, FFN, HEADS, KV_HEADS, HEAD_DIM, LAYERS, AUX = 7168, 14336, 96, 16, 64, 5, 5


def shapes(tp):
    """{linear: ((N, K), row kind)} of the release drafter at ``tp`` ("q": query
    rows, "c": context rows)."""
    q = HEADS // tp * HEAD_DIM
    kv = max(1, KV_HEADS // tp) * HEAD_DIM
    return {
        "qkv": ((q + 2 * kv, HIDDEN), "q"),
        "o_proj": ((HIDDEN, q), "q"),
        "gate_up": ((2 * FFN // tp, HIDDEN), "q"),
        "down": ((HIDDEN, FFN // tp), "q"),
        "ctx_shard": ((HIDDEN // tp, AUX * HIDDEN), "c"),
        "ctx_kv": ((LAYERS * 2 * kv, HIDDEN), "c"),
    }


# The preview drafter's (kimi-k3.1-pro-dspark-preview, MLA) TP16 pieces the release
# replaces (runtime.draft PROJECTIONS / DRAFT_96), timed at the same rows.
PREVIEW16 = {
    "pv_qkv_a": ((2112, HIDDEN), "q"),
    "pv_q_b": ((1152, 1536), "q"),
    "pv_o_proj": ((HIDDEN, 768), "q"),
    "pv_ctx_kv": ((2880, HIDDEN), "c"),
}

# Drafter batches of the chosen deployments (runtime.draft.draft_batches on their
# decode capture lists): k7 points B1-B64, k3 points B2-B160 (TP16), C512 B2-B128.
K7_BATCHES = list(range(1, 65))
K3_BATCHES = list(range(2, 161))
C512_BATCHES = list(range(2, 129))


def rows_of(kind, tp):
    """Sorted drafter row counts of ``kind`` at ``tp``."""
    plans = [(3, C512_BATCHES)] if tp == 4 else [(7, K7_BATCHES), (3, K3_BATCHES)]
    out = set()
    for k, batches in plans:
        for b in batches:
            out.add(b * k if kind == "q" else b * (k + 1))
    return sorted(out)


def band(rows):
    for top, name in ((8, "1-8"), (16, "9-16"), (32, "17-32"), (64, "33-64"),
                      (128, "65-128"), (256, "129-256"), (640, "257-640")):  # fmt: skip
        if rows <= top:
            return name
    return "641+"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def time_graph(fns, calls=64, reps=5):
    """Median us per call of ``fns`` (one per rotating weight copy) replayed in one
    CUDA graph of about ``calls`` calls."""
    import torch

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    n = len(fns)
    loops = max(1, calls // n)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(loops):
            for f in fns:
                f()
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


def copies_for(nbytes, target=640 << 20, lo=2, hi=96):
    return max(lo, min(hi, math.ceil(target / max(1, nbytes))))


TMEM = ((128, 64, 6, 16, 14), (128, 64, 6, 14, 14))
# Distinct row counts of one K a tactic may fail at before the sweep skips it.
FAIL_LIMIT = 2


def candidates(shape, rows):
    """Split-K tactics (M tile 64; N tile by rows; splits dividing the K tiles; 4 / 8
    / 14 stages within shared memory) for (N, K) at ``rows``."""
    _, k = shape
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
        for split in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 28):
            if k_tiles % split:
                continue
            per = k_tiles // split
            for stages in sorted({min(per, s) for s in (3, 4, 6, 8, 10, 14)}):
                if stages * (64 + n) * 256 > 200 * 1024:
                    continue
                out.append((64, n, split, stages))
    if k == HIDDEN and 33 <= rows <= 64:
        out += list(TMEM)
    if k == HIDDEN and 65 <= rows <= 128:
        # SplitKTmemGemm64's wider N tiles (80 / 96 / 128 activation rows)
        n = next(t for t in (80, 96, 128) if rows <= t)
        tiles = {80: 13, 96: 13, 128: 12}[n]
        out += [(128, n, split, slots, tiles) for split in (4, 6, 7, 8)
                for slots in (8, 12, 16)]  # fmt: skip
    return list(dict.fromkeys(out))


# Exit status after a sticky CUDA error (the context can run nothing more): the
# wrapper (run_gemm.sh) restarts the bench, which resumes from raw.csv and records
# the arm that was in flight as failed (never retried).
STICKY_EXIT = 75


def cuda_poisoned():
    import torch

    try:
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        return True
    return False


class Raw:
    """raw.csv (linear, tp, rows, arm, us): every measurement, reloaded on restart;
    inflight.txt: the arm being measured (a crash records it as failed)."""

    def __init__(self, out):
        self.path = os.path.join(out, "raw.csv")
        self.inflight = os.path.join(out, "inflight.txt")
        self.recs = {}
        if os.path.exists(self.path):
            with open(self.path, newline="") as fh:
                for r in csv.DictReader(fh):
                    key = (r["linear"], int(r["tp"]), int(r["rows"]), r["arm"])
                    self.recs[key] = float(r["us"]) if r["us"] else None
        new = not os.path.exists(self.path)
        self.fh = open(self.path, "a", newline="")  # noqa: SIM115 (kept open)
        self.w = csv.writer(self.fh)
        if new:
            self.w.writerow(["linear", "tp", "rows", "arm", "us"])
            self.fh.flush()
        if os.path.exists(self.inflight):
            with open(self.inflight) as fh:
                parts = fh.read().strip().split(",")
            if len(parts) == 4:
                key = (parts[0], int(parts[1]), int(parts[2]), parts[3])
                log(f"resume: {key} crashed the previous process: recorded failed")
                self.add(key, None)
            os.remove(self.inflight)

    def add(self, key, us):
        self.recs[key] = us
        self.w.writerow([*key, "" if us is None else f"{us:.4f}"])
        self.fh.flush()

    def run(self, key, fn, *args):
        if key in self.recs:
            return self.recs[key]
        with open(self.inflight, "w") as fh:
            fh.write(",".join(map(str, key)))
        try:
            us = fn(*args)
        except Exception as e:  # noqa: BLE001
            tail = traceback.format_exc().strip().splitlines()[-3:]
            log(
                f"FAILED {key}: {type(e).__name__}: {str(e)[:200]} | {' / '.join(tail)}"
            )
            if cuda_poisoned():
                log(f"sticky CUDA error after {key}: exit {STICKY_EXIT} (resume)")
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(STICKY_EXIT)
            us = None
        os.remove(self.inflight)
        self.add(key, us)
        return us


def weights(shape, device, gen, copies=None):
    import torch

    n, k = shape
    count = copies or copies_for(n * k * 2)
    return [
        (torch.randn((n, k), device=device, generator=gen) * 0.02).bfloat16()
        for _ in range(count)
    ]


def run_cublas(ws, x):
    import torch.nn.functional as F

    return time_graph([lambda w=w: F.linear(x, w) for w in ws])


def run_tactic(ws, x, out, tactic):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (
        PreparedProjection,
    )

    rows = x.shape[0]
    preps = [PreparedProjection(w, {rows: tactic}).prepare() for w in ws]
    return time_graph([lambda p=p: p.run(x, out) for p in preps])


def preview_tactic(draft, shape, rows):
    """The preview drafter's TP16 split-K tactic (runtime.draft PROJECTIONS /
    PROJECTIONS_WIDE, as ``runtime.draft.projection`` looks it up)."""
    if rows <= draft.PROJECTION_ROWS:
        return draft.PROJECTIONS.get(16, {}).get(shape)
    bins = draft.PROJECTIONS_WIDE.get(16, ())
    return next((t for top, t in bins if rows <= top), {}).get(shape)


def tname(t):
    return "splitk" + ".".join(map(str, t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=16)
    ap.add_argument("--part", default="chosen,sweep")
    ap.add_argument("--linears", default="")
    ap.add_argument("--preview", action="store_true", help="also the preview pieces")
    ap.add_argument("--sweep-max", type=int, default=64)
    ap.add_argument("--sweep-rows", default="", help="explicit sweep rows (csv)")
    ap.add_argument("--sweep-copies", type=int, default=24)
    ap.add_argument("--budget", type=float, default=0)
    args = ap.parse_args()
    import torch

    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev)
    cc = f"{props.major}.{props.minor}"
    log(f"device {props.name} SMs {props.multi_processor_count} cc {cc}")
    gen = torch.Generator(device=dev).manual_seed(0)
    # (the hold's keepalive yields within ~2 s of another CUDA context)
    time.sleep(3)
    end = time.time() + args.budget if args.budget else float("inf")
    specs = dict(shapes(args.tp))
    if args.preview and args.tp == 16:
        specs.update(PREVIEW16)
    names = [n for n in (args.linears.split(",") if args.linears else specs) if n]
    raw = Raw(args.out)
    parts = args.part.split(",")
    chosen_path = os.path.join(args.out, f"chosen_tp{args.tp}.csv")
    with open(chosen_path, "w", newline="") as fh:
        cw = csv.writer(fh)
        cw.writerow(["linear", "tp", "shape", "rows", "band", "chosen", "spec_us",
                     "cublas_us", "ratio"])  # fmt: skip
        for name in names:
            shape, kind = specs[name]
            ws = weights(shape, dev, gen)
            rows_list = rows_of(kind, args.tp)
            if "chosen" in parts:
                for rows in rows_list:
                    if time.time() > end:
                        break
                    x = torch.randn((rows, shape[1]), device=dev, generator=gen)
                    x = x.bfloat16()
                    out = torch.empty(
                        (rows, shape[0]), dtype=torch.bfloat16, device=dev
                    )
                    key = (name, args.tp, rows)
                    base = raw.run((*key, "cublas"), run_cublas, ws, x)
                    if name.startswith("pv_"):
                        tactic = preview_tactic(draft, shape, rows)
                    else:
                        tactic = draft.gqa_projection_tactic(shape, args.tp, rows)
                    spec, chosen = base, "cuBLAS"
                    if tactic is not None:
                        tactic = tuple(tactic)
                        chosen = tname(tactic)
                        spec = raw.run((*key, chosen), run_tactic, ws, x, out, tactic)
                    ratio = spec / base if spec and base else float("nan")
                    shape_name = "x".join(map(str, shape))
                    cw.writerow([name, args.tp, shape_name, rows, band(rows), chosen,
                                 spec, base, f"{ratio:.4f}"])  # fmt: skip
                    fh.flush()
                    spec_us = spec if spec is None else round(spec, 2)
                    base_us = base if base is None else round(base, 2)
                    print(f"CHOSEN {name:<10} {shape} rows {rows:>4} {chosen:<24} "
                          f"{spec_us} vs cuBLAS {base_us} ratio {ratio:.3f}",
                          flush=True)  # fmt: skip
            if "sweep" in parts:
                if args.sweep_rows:
                    sweep = [int(r) for r in args.sweep_rows.split(",")]
                else:
                    sweep = [r for r in rows_list if r <= args.sweep_max]
                swept = {}
                few = ws[: args.sweep_copies]
                for rows in sweep:
                    if time.time() > end:
                        log(f"budget out: {name} rows {rows}")
                        break
                    x = torch.randn((rows, shape[1]), device=dev, generator=gen)
                    x = x.bfloat16()
                    out = torch.empty(
                        (rows, shape[0]), dtype=torch.bfloat16, device=dev
                    )
                    key = (name, args.tp, rows)
                    base = raw.run((*key, "cublas"), run_cublas, ws, x)
                    res = {}
                    for t in candidates(shape, rows):
                        if len(t) == 4 and shape[1] % (128 * t[2]):
                            continue
                        # A tactic that failed (or crashed) at FAIL_LIMIT row counts of
                        # this K is skipped for the rest of it.
                        fails = {
                            k[2] for k, v in raw.recs.items()
                            if v is None and k[3] == tname(t)
                            and specs.get(k[0], ((0, 0),))[0][1] == shape[1]
                        }  # fmt: skip
                        if len(fails) >= FAIL_LIMIT and rows not in fails:
                            continue
                        us = raw.run((*key, tname(t)), run_tactic, few, x, out, t)
                        if us is not None:
                            res[t] = us
                    swept[rows] = (base, res)
                    if res:
                        best = min(res, key=res.get)
                        print(f"SWEEP {name:<10} {shape} rows {rows:>4} best "
                              f"{tname(best)} {res[best]:.2f} vs cuBLAS {base:.2f} "
                              f"({res[best] / base:.3f}; {len(res)} tactics)",
                              flush=True)  # fmt: skip
                bands = {}
                for rows, (base, res) in swept.items():
                    if res and base:
                        bands.setdefault(band(rows), []).append(rows)
                for b, rws in bands.items():
                    common = set.intersection(*(set(swept[r][1]) for r in rws))
                    if not common:
                        continue

                    def geo(vals):
                        return math.exp(sum(math.log(v) for v in vals) / len(vals))

                    scores = {t: geo([swept[r][1][t] for r in rws]) for t in common}
                    best = min(scores, key=scores.get)
                    cub = geo([swept[r][0] for r in rws])
                    print(
                        f"TUNE {name},tp{args.tp},{shape},{b},rows "
                        f"{min(rws)}-{max(rws)},{tname(best)},{scores[best]:.2f},"
                        f"cublas {cub:.2f},{scores[best] / cub:.3f}",
                        flush=True,
                    )
            del ws
            torch.cuda.empty_cache()
    log("done")


if __name__ == "__main__":
    main()
