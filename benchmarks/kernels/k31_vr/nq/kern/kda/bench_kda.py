#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""nq kernel microbench, group kda: Kimi K3.1 specialized decode KDA kernels on Rubin (SM107), TP16 per-rank shapes
(6 KDA heads x 128, packed in-projection pitch 3216, hidden 7168), FP32 recurrent state (the nq serving precision)
with the BF16 state (the previous serving precision, every earlier number in the code comments) as reference.

Sections (1 GPU each; cases x rows below):
  a  KdaHead family vs the in-projection -> KdaSplit pair vs stock, from x (the attention input) to the KDA core
     output: q1 B8 (KdaHeadWide), q8 M8 (KdaHeadVerify), q8 M16 and q4 M8 (KdaHeadMulti); plain and polled
     output; FP32 and BF16 state. Modes: iso (69 layers back to back, graph us/layer) and chain (each layer behind
     a PDL AttnRes stand-in, standin.py: the head's / pair's done time minus the x store, as b1/kda/tools/kda_chain.py;
     plus chain_graph = graph us/layer incl. the stand-in). Optional --early: NQ-TUNE 3 early-tile re-sweep.
     pair  = production tactic of the in-projection (providers.projection_policies) + PreparedKda.run
     stock = cuBLAS torch.mm in-projection + seam S1 (providers.Attention.stock_kda -> the stock
             KimiK3DeltaAttention._forward: causal_conv1d_update + fused_recurrent_kda (spec) / packed decode (q1)
             + gated RMSNorm, DS conv layout, RecoverSSM off: what the stock layer runs at these shapes)
  b  plain verify core from the projected rows (``run_wide``'s KDA core): KdaSplit / KdaVerifyWide / KdaVerifyHalves /
     KdaVerifyStream3 (BF16 state, q4 only) / KdaSplit with the 1.8.3.4 schedule (B > 64), forced through
     ``prepared.wide_kind`` (the kernel the production table picks is marked prod), vs stock seam S1 on the same
     projected rows. q4 requests 1..512, q8 1..256, FP32 and BF16 state. Mode iso: 69 layers (fewer above the
     --mem-gb budget), each with its own cold cache slots, graph us/layer (the method of the VERIFY_WIDE numbers).
  c  in-projection GEMM: cuBLAS (stock) vs cuBLASLt (INPROJ_LT) vs the production split-K / tensor-memory tactic
     where one exists, rows 8..2048, pitches 3216 (KDA), 2880 (MLA), 3040 (DSA); 8 cold weight copies, graphs of 16
     calls (the INPROJ_LT method); bitwise cuBLAS == cuBLASLt per row.

Output: one CSV row per (case, rows, mode, arm) in <out>/<tag>.csv (appended as measured), a table row per (case,
rows, mode) in the log; ``merge`` writes <out>/ALL.csv, <out>/SUMMARY.csv and <out>/SUMMARY.txt (WORSE = production
pick slower than stock, RETUNE = another arm beats the production pick by > 2 %, MISSING / errors, FP32-vs-BF16 state,
previous numbers from the code comments). One failing arm or case is logged and the run continues.

  bench_kda.py run --sections a|b|c [--shard i/n] --out DIR --tag NAME [--modes iso,chain] [--early q1=13]
  bench_kda.py merge --out DIR
  bench_kda.py selftest            (pure python: case lists, previous-number tables, summary logic)
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import sys
import time
import traceback
from collections import OrderedDict
from contextlib import contextmanager

HERE = os.path.dirname(os.path.abspath(__file__))
HEADS, HDIM, DIM, HIDDEN = 6, 128, 768, 7168  # TP16: 6 of 96 KDA heads per rank
PITCH = 3216  # Q | K | V | gate | F_A 128 | beta 6 | pad
FA_ROWS = PITCH - 4 * DIM
KDA_LAYERS = 69
RUBIN = (10, 7)

# ---------------------------------------------------------------------------------------------------------------- cases
HEAD_SHAPES = [  # (rows, q, name)
    (8, 1, "q1-B8-head"),
    (8, 8, "q8-M8-headverify"),
    (16, 8, "q8-M16-headmulti"),
    (8, 4, "q4-M8-headmulti"),
]
Q4_REQUESTS = [1, 2, 4, 6, 8, 12, 14, 16, 24, 32, 33, 34, 36, 40, 48, 56, 64, 65, 80, 96, 128, 160, 192, 256, 320,
               384, 448, 512]
Q8_REQUESTS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 65, 96, 128, 160, 192, 256]
STATES = ("fp32", "bf16")
GEMM_PITCHES = (3216, 2880, 3040)
# early-tile sweep values (NQ-TUNE 3): production q1 HEAD_TILE early 17, VERIFY_EARLY_LOADS 21, MULTI_TILE early 21
EARLY_SWEEP = {"q1": (9, 13, 21), "verify": (15, 18, 24), "multi": (15, 18, 24)}


def gemm_rows():
    return list(range(8, 129, 8)) + list(range(132, 1025, 4)) + list(range(1040, 2049, 16))


def verify_cases(quick=False):
    """Section b cases (q, requests, state), heavy and light interleaved so shards balance."""
    if quick:
        return [(4, 8, "fp32"), (4, 64, "bf16"), (8, 16, "fp32")]
    cases = []
    for state in STATES:
        cases += [(4, b, state) for b in Q4_REQUESTS] + [(8, b, state) for b in Q8_REQUESTS]
    # sort by rows, then deal big/small alternately
    cases.sort(key=lambda c: c[0] * c[1])
    out, lo, hi = [], 0, len(cases) - 1
    while lo <= hi:
        out.append(cases[hi])
        if lo != hi:
            out.append(cases[lo])
        lo, hi = lo + 1, hi - 1
    return out


def verify_arms(q, requests, state):
    """(arm, wide-kind value, schedule patch) of section b; the production pick is marked at run time."""
    arms = [("kdasplit", 0, None), ("wide", 1, None), ("halves", 2, None)]
    if state == "bf16" and q <= 4:
        arms.append(("stream3", 4, None))
    if requests > 64:
        arms.append(("kdasplit1834", 0, "65-:1.8.3.4"))
    return arms


def arm_wanted(args, arm):
    """--arms / --skip-arms (section b): only these / not these arms are timed (kdasplit1834 runs in its own process
    with its own CuTe cache dir, and the shards skip it)."""
    if arm in (getattr(args, "skip_arms", None) or ()):
        return False
    return not getattr(args, "arms", None) or arm in args.arms


def shard(items, spec):
    if not spec:
        return list(items)
    i, n = map(int, spec.split("/"))
    return [c for k, c in enumerate(items) if k % n == i]


# ------------------------------------------------------------------------------------------- previous numbers (comments)
# Section b, BF16 state, 1-GPU verify alone, 69 layers, cold slots, us per layer (kda/prepared.py VERIFY_WIDE and
# VERIFY_Q comments; the VERIFY_Q run where both give a value).
PREV_VERIFY = {}
for _q, _rows in ((8, {16: (9.60, 10.62, 8.93), 24: (16.88, 11.10, 11.98), 32: (18.13, 11.21, 12.13),
                       48: (24.76, 21.51, 19.75), 64: (27.44, 22.21, 23.73)}),
                  (4, {16: (6.27, 6.40, 5.57), 24: (10.57, 6.82, 7.27), 32: (11.93, 7.10, 7.49),
                       48: (16.76, 13.58, 12.58), 64: (18.31, 14.10, 14.80)})):
    for _b, (_s, _w, _h) in _rows.items():
        PREV_VERIFY[_q, _b, "kdasplit"] = _s
        PREV_VERIFY[_q, _b, "wide"] = _w
        PREV_VERIFY[_q, _b, "halves"] = _h
for _b, (_w, _h, _s3) in {33: (6.27, 7.08, 6.70), 36: (10.92, 11.21, 8.94), 48: (11.66, 12.38, 9.27),
                          64: (12.28, 13.88, 10.29), 96: (18.65, None, 16.26), 128: (25.03, None, 19.44),
                          192: (37.22, None, 28.51), 256: (49.13, None, 37.25)}.items():
    PREV_VERIFY[4, _b, "wide"] = _w
    if _h is not None:
        PREV_VERIFY[4, _b, "halves"] = _h
    PREV_VERIFY[4, _b, "stream3"] = _s3
for _b, _v in {96: 36.38, 128: 47.60, 192: 69.85}.items():
    PREV_VERIFY[4, _b, "kdasplit"] = _v
# Section c (providers.py INPROJ_LT / PROJECTION_GENERIC comments): (pitch, lo, hi, cuBLAS us, cuBLASLt us)
PREV_LT_BANDS = [
    (3216, 260, 288, 10.8, 10.4), (3216, 292, 336, 11.9, 11.1), (3216, 388, 448, 13.0, 12.4),
    (3216, 452, 464, 13.6, 12.5), (3216, 468, 480, 13.0, 12.4), (3216, 516, 576, 15.2, 13.9),
    (2880, 292, 320, 11.6, 11.0), (2880, 324, 384, 16.3, 11.85), (2880, 388, 512, 12.6, 12.1),
    (2880, 516, 576, 19.9, 13.6),
]
PREV_GEMM = {(3216, 128, "cublas"): 9.30, (3216, 256, "cublas"): 9.82, (3216, 512, "cublas"): 13.06,
             (2880, 128, "cublas"): 10.51, (2880, 256, "cublas"): 9.76, (2880, 512, "cublas"): 12.92,
             (3216, 128, "splitk"): 8.73, (2880, 128, "splitk"): 8.47}
# Section a (chain, BF16 state; providers.py HEAD_BATCHES / HEAD_MULTI): deltas, not absolute us
PREV_HEAD_NOTES = {"q1-B8-head": "BF16 chain: head -0.36 us/layer vs pair",
                   "q8-M16-headmulti": "BF16 chain: pair 12.40 -> head 9.09 us (o-proj PDL release)"}


def prev_us(section, q, requests, rows, arm, pitch=None):
    if section == "b":
        return PREV_VERIFY.get((q, requests, arm))
    if section == "c":
        if (pitch, rows, arm) in PREV_GEMM:
            return PREV_GEMM[pitch, rows, arm]
        for p, lo, hi, cb, lt in PREV_LT_BANDS:
            if p == pitch and lo <= rows <= hi and arm in ("cublas", "cublaslt"):
                return cb if arm == "cublas" else lt
    return None


# ------------------------------------------------------------------------------------------------------------ CSV/summary
FIELDS = ["section", "case", "state", "q", "requests", "rows", "mode", "arm", "us", "prod", "stock", "relerr",
          "prev_bf16_us", "layers", "note"]
GROUP = ("section", "case", "state", "q", "requests", "rows", "mode")


class Sink:
    def __init__(self, path):
        self.path = path
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self.rows = []
        if not new:  # a restarted process (sticky CUDA error, rc STICKY_RC): keep and skip what is measured
            with open(path, newline="") as f:
                self.rows = list(csv.DictReader(f))
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, FIELDS)
        if new:
            self.w.writeheader()
        self.done = {case_key(r) for r in self.rows}

    def add(self, **row):
        row = {k: ("" if row.get(k) is None else row.get(k)) for k in FIELDS}
        for k in ("us", "relerr", "prev_bf16_us"):
            if isinstance(row[k], float):
                row[k] = f"{row[k]:.4g}" if k == "relerr" else f"{row[k]:.3f}"
        self.w.writerow(row)
        self.f.flush()
        self.rows.append(row)
        return row


def case_key(r):
    """One measured case (all arms / modes): resume skips a case with any row, the crashed one included."""
    return tuple(str(r.get(k, "")) for k in ("section", "case", "state", "q", "requests", "rows"))


def is_eager(r):
    return "eager" in str(r.get("note", "")).split()


def _f(v):
    try:
        x = float(v)
        return x if x > 0 else None
    except (TypeError, ValueError):
        return None


def summarize(rows):
    groups = OrderedDict()
    for r in rows:
        groups.setdefault(tuple(r[k] for k in GROUP), []).append(r)
    out = []
    for key, rs in groups.items():
        ok = [r for r in rs if _f(r["us"])]
        errs = [f"{r['arm']}: {r['note']}" for r in rs if not _f(r["us"])]
        prod = [r for r in ok if str(r["prod"]) == "1"]
        stock = [r for r in ok if str(r["stock"]) == "1"]
        cands = [r for r in ok if str(r["stock"]) != "1"]
        best = min(cands, key=lambda r: _f(r["us"])) if cands else None
        p = min(prod, key=lambda r: _f(r["us"])) if prod else None
        s = stock[0] if stock else None
        d = dict(zip(GROUP, key))
        d.update(prod_arm=p["arm"] if p else "", prod_us=_f(p["us"]) if p else None,
                 stock_us=_f(s["us"]) if s else None, best_arm=best["arm"] if best else "",
                 best_us=_f(best["us"]) if best else None, errors="; ".join(errs),
                 prev=p["prev_bf16_us"] if p else "", arms=" ".join(f"{r['arm']}={_f(r['us']):.2f}" for r in ok))
        d["eager"] = " ".join(sorted({r["arm"] for r in ok if is_eager(r)}))
        d["ratio"] = d["prod_us"] / d["stock_us"] if d["prod_us"] and d["stock_us"] else None
        d["best_vs_prod"] = d["best_us"] / d["prod_us"] if d["best_us"] and d["prod_us"] else None
        if not prod and any(str(r["prod"]) == "1" for r in rs):
            d["prod_arm"] = next(r["arm"] for r in rs if str(r["prod"]) == "1") + "(FAILED)"
        out.append(d)
    return out


def fmt_row(d):
    f = lambda v, w=8: (f"{v:{w}.2f}" if isinstance(v, float) else f"{'-':>{w}}")
    ratio = f"{d['ratio']:.3f}" if d["ratio"] else "  -  "
    flag = " WORSE" if d["ratio"] and d["ratio"] > 1.0 else ""
    if d["best_vs_prod"] and d["best_vs_prod"] < 0.98 and d["best_arm"] != d["prod_arm"]:
        flag += f" RETUNE->{d['best_arm']}"
    if d.get("eager"):
        flag += f" EAGER({d['eager']})"
    return (f"{d['section']} {d['case']:<20} {d['state']:<5} q{d['q']:<2} B{d['requests']:<4} M{d['rows']:<5} "
            f"{d['mode']:<11} spec {f(d['prod_us'])} stock {f(d['stock_us'])} ratio {ratio:>6} "
            f"tactic {d['prod_arm'] or '-':<14} best {d['best_arm'] or '-':<14}{f(d['best_us'])}"
            f"{' prev(bf16) ' + str(d['prev']) if d['prev'] else ''}{flag}"
            f"{'  ERR ' + d['errors'][:160] if d['errors'] else ''}")


def report(rows, out_txt=None, out_csv=None):
    summ = summarize(rows)
    lines = ["# kda nq kernel bench summary: spec = production pick, ratio = spec / stock (> 1.0: WORSE)", ""]
    lines += [fmt_row(d) for d in summ]
    worse = [d for d in summ if d["ratio"] and d["ratio"] > 1.0]
    retune = [d for d in summ if d["best_vs_prod"] and d["best_vs_prod"] < 0.98 and d["best_arm"] != d["prod_arm"]]
    missing = [d for d in summ if d["prod_us"] is None or d["stock_us"] is None]
    eager = [d for d in summ if d["eager"]]
    lines += ["", f"WORSE ({len(worse)}): production pick slower than stock"] + ["  " + fmt_row(d) for d in worse]
    lines += ["", f"RETUNE ({len(retune)}): another arm beats the production pick by > 2 %"]
    lines += ["  " + fmt_row(d) for d in retune]
    lines += ["", f"MISSING ({len(missing)}): no production or no stock time"] + ["  " + fmt_row(d) for d in missing]
    lines += ["", f"EAGER ({len(eager)}): graph capture failed for these arms, timed eagerly (launch overhead "
              "included): the ratio is not comparable"] + ["  " + fmt_row(d) for d in eager]
    # FP32 vs BF16 state of the production pick (same section / case / q / rows / mode)
    by = {}
    for d in summ:
        by[(d["section"], d["case"], d["q"], d["requests"], d["rows"], d["mode"], d["state"])] = d
    lines += ["", "FP32 / BF16 state (production pick):"]
    for k, d in by.items():
        if k[-1] != "fp32":
            continue
        b = by.get(k[:-1] + ("bf16",))
        if b and d["prod_us"] and b["prod_us"]:
            lines.append(f"  {k[0]} {k[1]:<20} q{k[2]:<2} B{k[3]:<4} M{k[4]:<5} {k[5]:<11} fp32 {d['prod_arm']:<12} "
                         f"{d['prod_us']:8.2f}  bf16 {b['prod_arm']:<12} {b['prod_us']:8.2f}  x{d['prod_us'] / b['prod_us']:.2f}")
    text = "\n".join(lines) + "\n"
    if out_txt:
        with open(out_txt, "w") as f:
            f.write(text)
    if out_csv:
        keys = list(GROUP) + ["prod_arm", "prod_us", "stock_us", "ratio", "best_arm", "best_us", "best_vs_prod",
                              "prev", "arms", "eager", "errors"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, keys, extrasaction="ignore")
            w.writeheader()
            for d in summ:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in d.items()})
    return text, worse, retune, missing


def merge(out):
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "*.csv"))):
        if os.path.basename(p) in ("ALL.csv", "SUMMARY.csv"):
            continue
        with open(p) as f:
            rows += list(csv.DictReader(f))
    with open(os.path.join(out, "ALL.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    text, worse, retune, missing = report(rows, os.path.join(out, "SUMMARY.txt"), os.path.join(out, "SUMMARY.csv"))
    print(text)
    print(f"merged {len(rows)} rows: WORSE {len(worse)} RETUNE {len(retune)} MISSING {len(missing)}")


# ================================================================================================================== GPU
torch = None  # imported in gpu_setup


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gpu_setup():
    global torch
    import torch as _torch

    torch = _torch
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    cap = torch.cuda.get_device_capability(dev)
    props = torch.cuda.get_device_properties(dev)
    import vllm

    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    log(f"device {props.name} cap {cap} sms {props.multi_processor_count} mem {props.total_memory / 2**30:.0f} GiB; "
        f"torch {torch.__version__}; vllm from {os.path.dirname(vllm.__file__)}; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}; "
        f"conv layout DS={is_conv_state_dim_first()}")
    if cap != RUBIN:
        log(f"WARNING: not Rubin (10.7): {cap}; the device tables (Rubin rows) will not apply as in production")
    if not is_conv_state_dim_first():
        log("WARNING: VLLM_SSM_CONV_STATE_LAYOUT is not DS: production (nq configs) runs the DS conv layout")
    return dev, props.multi_processor_count


STICKY_RC = 86  # run_inner.sh restarts the process on this rc; the Sink resumes after the crashed case


def check_cuda_alive(sink, dev, where):
    """After each case: a sticky CUDA error (illegal address, trap, ...) poisons the context and every later case in
    this process would record an ERROR. Record it on the case and exit, so run_inner.sh restarts a fresh process that
    resumes with the next case."""
    try:
        torch.cuda.synchronize()
        torch.ones(1, device=dev).add_(1).item()
        return
    except Exception as e:  # noqa: BLE001
        msg = f"STICKY CUDA error after {where}: {type(e).__name__}: {str(e)[:200]}"
    log(msg + f"; exiting {STICKY_RC} for a restart")
    try:
        sec, case, state, q, requests, rows = where
        sink.add(section=sec, case=case, state=state, q=q, requests=requests, rows=rows, mode="iso", arm="case",
                 note=f"ERROR {msg}")
        sink.f.flush()
        os.fsync(sink.f.fileno())
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(STICKY_RC)


def graph_time(body, ncalls, reps, marks=None, nmarks=0):
    """(median us per call, [per-rep us], median mark delta us or None): body captured in one CUDA graph; falls back
    to an eager loop when capture fails (note 'eager')."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    note = "graph"
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        run = g.replay
    except Exception as e:  # noqa: BLE001
        log(f"  graph capture failed ({type(e).__name__}: {str(e)[:200]}); timing eager")
        torch.cuda.synchronize()
        g, run, note = None, body, "eager"
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    times, deltas = [], []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        run()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000 / ncalls)
        if marks is not None:
            m = marks[: 2 * nmarks].view(nmarks, 2).cpu().tolist()
            deltas += [(m[i][1] - m[i][0]) / 1000 for i in range(1, nmarks)]
    del g
    return statistics.median(times), note, (statistics.median(deltas) if deltas else None)


def relerr(got, want):
    g, w = got.float(), want.float()
    if not torch.isfinite(g).all():
        return float("nan")
    return ((g - w).norm() / w.norm().clamp_min(1e-30)).item()


class Kda:
    """Imports of the code under test (overlay), and the stock layer stand-in."""

    def __init__(self, dev, sms, stock_norm="auto"):
        self.stock_norm = stock_norm
        from vllm.models.kimi_k3.nvidia import kda as kda_mod
        from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
        from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadata
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.policy import batch_table
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import PreparedProjection
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import head as head_mod
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import prepared
        from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import providers
        from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Path, Shape

        self.dev, self.sms = dev, sms
        self.kda_mod, self.KDA, self.Meta = kda_mod, KimiK3DeltaAttention, KimiK3KDAMetadata
        self.batch_table, self.PreparedProjection = batch_table, PreparedProjection
        self.head, self.prepared, self.providers, self.Path, self.Shape = head_mod, prepared, providers, Path, Shape
        self.ctx = {"m": None}
        kda_mod.get_forward_context = lambda: _Ctx(self.ctx["m"])
        self.o_norm = self._o_norm()

    def _o_norm(self):
        from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

        def build():
            m = FusedRMSNormGated(HDIM, activation="sigmoid")
            m.weight.data = (1 + 0.1 * torch.randn(HDIM, device=self.dev)).to(torch.bfloat16)
            return m

        try:
            try:
                from vllm.config import VllmConfig, set_current_vllm_config

                with set_current_vllm_config(VllmConfig()):
                    m = build()
            except Exception:  # noqa: BLE001
                m = build()
        except Exception as e:  # noqa: BLE001
            m = None
            error = e
        if m is not None:
            # The stock layer's norm path follows CustomOp dispatch: the nq configs set no custom_ops, so under
            # VLLM_COMPILE the op is disabled (forward_native, maybe torch.compile'd); enabled, it runs forward_cuda
            # and kda_norm's tiled kernel. --stock-norm pins one; auto keeps VllmConfig()'s dispatch (logged).
            if self.stock_norm == "cuda":
                m._forward_method = m.forward_cuda
            elif self.stock_norm == "native":
                m._forward_method = m.forward_native
            fwd = getattr(getattr(m, "_forward_method", None), "__name__", "?")
            log(f"stock o_norm dispatch: {fwd} (--stock-norm {self.stock_norm})")
            try:
                from vllm.models.kimi_k3.nvidia.ops import kda_norm

                probe = torch.empty(1, 8, HEADS, HDIM, device=self.dev, dtype=torch.bfloat16)
                gate = torch.empty(8, HEADS, HDIM, device=self.dev, dtype=torch.bfloat16)
                log(f"stock o_norm: FusedRMSNormGated, kda_norm tiles path {kda_norm.supported(probe, gate, m)}")
            except Exception as e:  # noqa: BLE001
                log(f"stock o_norm probe failed (informational): {e}")
            return m
        try:
            raise error
        except Exception as e:  # noqa: BLE001
            log(f"WARNING: FusedRMSNormGated did not build ({type(e).__name__}: {e}); stock norm through kda_norm "
                "directly (the path the stock layer takes on CUDA)")
            from types import SimpleNamespace

            from vllm.models.kimi_k3.nvidia.ops import kda_norm

            kda_norm.supported = lambda *a, **k: True
            w = (1 + 0.1 * torch.randn(HDIM, device=self.dev)).to(torch.bfloat16)
            return SimpleNamespace(weight=w, eps=1e-5, bias=None, activation="sigmoid")

    def attention(self, seed=0):
        """The fields PreparedKda and the stock layer read (stock dtypes: FP32 conv weight [3 dim, 1, 4], A_log,
        dt bias; BF16 F_B projection, BF16 norm weight)."""
        from types import SimpleNamespace as NS

        g = torch.Generator(device=self.dev).manual_seed(seed)
        r = lambda *s, dtype=torch.float32, scale=1.0: (torch.randn(*s, generator=g, device=self.dev) * scale).to(dtype)
        return NS(conv1d=NS(weight=r(3 * DIM, 1, 4, scale=0.5), bias=None), dt_bias=r(DIM, scale=0.5),
                  A_log=r(HEADS, scale=0.5), f_b_proj=NS(weight=r(DIM, HDIM, dtype=torch.bfloat16, scale=HDIM ** -0.5)),
                  o_norm=self.o_norm, gate_lower_bound=-5.0)

    def stock_layer(self, attn, caches):
        """The stock KimiK3DeltaAttention fields seam S1 (``Attention.stock_kda``) and ``_forward`` read, with the
        backends the stock layer resolves at TP16 / DS layout / spec decode: kda_spec_decode_backend native
        (FlashInfer's packed path needs the SD layout), kda_decode_backend triton (6 heads, num_spec > 0)."""
        from types import SimpleNamespace as NS

        import torch.nn.functional as F

        fb = attn.f_b_proj.weight
        layer = NS(prefix="nqk.kda", kv_cache=caches, kda_spec_decode_backend="native", kda_decode_backend="triton",
                   decode_conv1d_weight=None, decode_norm_weight=None, conv1d=attn.conv1d,
                   local_projection_size=DIM, local_num_heads=HEADS, head_dim=HDIM, use_recoverssm=False,
                   A_log=attn.A_log, dt_bias=attn.dt_bias, gate_lower_bound=attn.gate_lower_bound,
                   kda_prefill_backend="triton", o_norm=attn.o_norm)
        layer.f_b_proj = lambda t: (F.linear(t, fb), None)
        layer._forward = lambda **kw: self.KDA._forward(layer, **kw)
        return layer

    def metadata(self, requests, q, slots, accepted):
        rows = requests * q
        if q == 1:
            return self.Meta(num_prefills=0, num_prefill_tokens=0, num_decodes=rows, num_decode_tokens=rows,
                             num_spec_decodes=0, num_spec_decode_tokens=0, num_actual_tokens=rows,
                             non_spec_state_indices_tensor=slots,
                             non_spec_query_start_loc=torch.arange(rows + 1, dtype=torch.int32, device=self.dev))
        return self.Meta(num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0,
                         num_spec_decodes=requests, num_spec_decode_tokens=rows, num_actual_tokens=rows,
                         spec_query_start_loc=torch.arange(0, rows + 1, q, dtype=torch.int32, device=self.dev),
                         spec_state_indices_tensor=slots, num_accepted_tokens=accepted)

    def stock_kda(self, layer, projected, out):
        self.providers.Attention.stock_kda(layer, projected, out)

    def caches(self, state_dtype, slots, taps, seed):
        g = torch.Generator(device=self.dev).manual_seed(seed)
        conv = torch.randn(slots, 3 * DIM, taps, generator=g, device=self.dev).to(torch.bfloat16)
        state = (torch.randn(slots, HEADS, HDIM, HDIM, generator=g, device=self.dev) * 0.1).to(state_dtype)
        return conv, state

    def spec_inputs(self, requests, q, slots, seed=2):
        g = torch.Generator(device="cpu").manual_seed(seed)
        if q == 1:
            order = torch.randperm(slots - 1, generator=g)[:requests] + 1
            return order.to(torch.int32).to(self.dev), None
        order = torch.randperm(slots - 1, generator=g)[: requests * q] + 1
        table = order.view(requests, q).to(torch.int32).to(self.dev)
        accepted = torch.randint(1, q + 1, (requests,), generator=g).to(torch.int32).to(self.dev)
        return table, accepted

    def prod_verify(self, requests, q, state):
        """The production KDA core at a wide verify shape: ('stock', note) on seam S1 or the plain verify kernel."""
        from types import SimpleNamespace as NS

        ctx = NS(arch="rubin", tp=16, bf16_state=state == "bf16")
        choice = self.prepared.choose(self.Shape(requests * q, q), ctx)
        if choice.path is self.Path.STOCK:
            return "stock", str(choice.note)
        return choice.note.split()[0], str(choice.note)


class _Ctx:
    def __init__(self, m):
        self.attn_metadata = {"nqk.kda": m}


def n_copies(per_copy_bytes, mem_gb, cap=KDA_LAYERS):
    return max(2, min(cap, int(mem_gb * 2**30 // max(per_copy_bytes, 1))))


# ------------------------------------------------------------------------------------------------------------ section b
@contextmanager
def forced_kind(prep, kind, sched):
    orig_kind, orig_sched = prep.wide_kind, prep.DEVICE_SPEC_SCHEDULE
    prep.executable.cache_clear()
    prep.wide_kind = lambda *a, **k: kind
    if sched:
        table = {cap: dict(v) for cap, v in orig_sched.items()}
        table.setdefault(RUBIN, {})[HEADS] = ",".join(filter(None, (table.get(RUBIN, {}).get(HEADS, ""), sched)))
        prep.DEVICE_SPEC_SCHEDULE = table
    try:
        yield
    finally:
        prep.wide_kind, prep.DEVICE_SPEC_SCHEDULE = orig_kind, orig_sched
        prep.executable.cache_clear()


def run_verify_case(k, sink, q, requests, state, args):
    prep = k.prepared
    sdt = torch.float32 if state == "fp32" else torch.bfloat16
    rows, slots, taps = requests * q, requests * q + 1, q + 2
    per_copy = slots * (HEADS * HDIM * HDIM * (4 if state == "fp32" else 2) + 3 * DIM * taps * 2)
    L = n_copies(per_copy, args.mem_gb)
    case = f"verify-q{q}"
    base = dict(section="b", case=case, state=state, q=q, requests=requests, rows=rows, mode="iso", layers=L)
    prod, prod_note = k.prod_verify(requests, q, state)
    attn = k.attention()
    copies = [k.caches(sdt, slots, taps, seed=10 + i) for i in range(L)]
    g = torch.Generator(device=k.dev).manual_seed(4)
    projected = torch.randn(rows, PITCH, generator=g, device=k.dev).to(torch.bfloat16)
    table, accepted = k.spec_inputs(requests, q, slots)
    k.ctx["m"] = k.metadata(requests, q, table, accepted)
    out = torch.empty(rows, DIM, dtype=torch.bfloat16, device=k.dev)
    ref = None
    # stock (seam S1); with --arms not naming stock only its reference output (the relerr of the other arms)
    try:
        check = tuple(t.clone() for t in copies[0])
        ref = torch.empty_like(out)
        k.stock_kda(k.stock_layer(attn, check), projected, ref)
        del check
        if arm_wanted(args, "stock"):
            layers = [k.stock_layer(attn, c) for c in copies]

            def body():
                for lay in layers:
                    k.stock_kda(lay, projected, out)

            us, note, _ = graph_time(body, L, args.reps)
            sink.add(**base, arm="stock", us=us, stock=1, prod=int(prod == "stock"), note=f"S1 native {note}")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        if arm_wanted(args, "stock"):
            sink.add(**base, arm="stock", stock=1, note=f"ERROR {type(e).__name__}: {str(e)[:200]}")
    for arm, kind, sched in verify_arms(q, requests, state):
        if not arm_wanted(args, arm):
            continue
        try:
            with forced_kind(prep, kind, sched):
                core = prep.PreparedKda(attn, {(requests, q): (PITCH, 0)}, True)
                core.prepare_caches(copies[0])
                bf16 = state == "bf16"
                schedule = prep.schedule_of(HEADS, requests, 0, q, RUBIN, bf16)
                kobj = prep.make_kernel((HEADS, requests, 0, False, schedule, q, bf16), RUBIN)
                # the requested schedule and what the kernel object resolved it to (lanes / occupancy / cluster):
                # a 1.8.3.4 arm whose built object shows the default schedule did not get its schedule
                built = "/".join(str(getattr(kobj, a, "?")) for a in ("lanes_per_row", "occupancy", "cluster"))
                kname = f"{type(kobj).__name__} sched {'.'.join(map(str, schedule))} built lanes/occ/cluster {built}"
            err = None
            if ref is not None:
                check = tuple(t.clone() for t in copies[0])
                got = torch.empty_like(out)
                core.run(projected, got, caches=check, slots=table, accepted=accepted)
                torch.cuda.synchronize()
                err = relerr(got, ref)
                del check, got

            def body():
                for c in copies:
                    core.run(projected, out, caches=c, slots=table, accepted=accepted)

            us, note, _ = graph_time(body, L, args.reps)
            sink.add(**base, arm=arm, us=us, prod=int(arm == prod), relerr=err,
                     prev_bf16_us=prev_us("b", q, requests, rows, arm),
                     note=f"{note} {kname} {prod_note if arm == prod else ''}".strip())
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            sink.add(**base, arm=arm, prod=int(arm == prod), note=f"ERROR {type(e).__name__}: {str(e)[:200]}")
    del copies, projected
    torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------------------ section a
@contextmanager
def early_override(k, spec):
    """--early q1=N | verify=N | multi=N: the head's A tiles loaded before the PDL wait (NQ-TUNE 3)."""
    hv = __import__(k.head.__name__.rsplit(".", 1)[0] + ".head_verify", fromlist=["x"])
    hm = __import__(k.head.__name__.rsplit(".", 1)[0] + ".head_multi", fromlist=["x"])
    saved = (k.head.HEAD_TILE, hv.VERIFY_EARLY_LOADS, dict(hm.MULTI_TILE))
    tile = k.head.HEAD_TILE
    if spec:
        which, val = spec.split("=")
        val = int(val)
        if which == "q1":
            tile = (*k.head.HEAD_TILE[:3], val)
        elif which == "verify":
            hv.VERIFY_EARLY_LOADS = val
        elif which == "multi":
            for key, (st, tm, _) in list(hm.MULTI_TILE.items()):
                hm.MULTI_TILE[key] = (st, tm, val)
        for f in (hv.compile_head_verify, hm.compile_head_multi, k.head.head_constructs, k.head.compile_head):
            if hasattr(f, "cache_clear"):
                f.cache_clear()
    try:
        yield tile
    finally:
        hv.VERIFY_EARLY_LOADS = saved[1]
        hm.MULTI_TILE.clear()
        hm.MULTI_TILE.update(saved[2])


def head_case_applies(spec, q, rows):
    if not spec:
        return True
    which = spec.split("=")[0]
    return {"q1": q == 1, "verify": q > 1 and rows == q, "multi": q > 1 and rows != q}[which]


def run_head_case(k, sink, rows, q, name, state, args, standin_mod):
    H = k.head
    sdt = torch.float32 if state == "fp32" else torch.bfloat16
    requests, slots, taps = rows // q, 4 * rows + 1, q + 2
    L = KDA_LAYERS
    polled_rows = H.polled_out_batches(16, k.dev)
    sweep = args.early
    attn = k.attention()
    g = torch.Generator(device=k.dev).manual_seed(3)
    Ws = [(torch.randn(PITCH, HIDDEN, generator=g, device=k.dev) * HIDDEN ** -0.5).to(torch.bfloat16)
          for _ in range(L)]
    copies = [k.caches(sdt, slots, taps, seed=20 + i) for i in range(L)]
    xs = [torch.randn(rows, HIDDEN, generator=g, device=k.dev).to(torch.bfloat16) for _ in range(8)]
    x = xs[0].clone()
    table, accepted = k.spec_inputs(requests, q, slots)
    k.ctx["m"] = k.metadata(requests, q, table, accepted)
    key = rows if q == 1 else (requests, q)
    core = k.prepared.PreparedKda(attn, {key: (PITCH, 0)}, True).prepare_caches(copies[0])
    out = torch.empty(rows, DIM, dtype=torch.bfloat16, device=k.dev)
    projected = torch.empty(rows, PITCH, dtype=torch.bfloat16, device=k.dev)
    # a sweep process must not compile the production early tiles (head_supported -> head_compiles for an FP32
    # state) into its fresh CuTe cache before early_override: if the cache key ignores the swept constant, the @tag
    # arms would load the production binary. Sweep arms are never the production pick, so the constructor check is
    # enough there.
    supported = (H.head_constructs(HEADS, sdt, False, HIDDEN, rows, q) if sweep else
                 H.head_supported(HEADS, sdt, False, HIDDEN, rows, q, core=core, fa_width=FA_ROWS,
                                  cache_tokens=q, tp=16))
    prod = ("head_polled" if rows in polled_rows else "head") if supported else "pair"
    tag = f"@{sweep.replace('=', '')}" if sweep else ""
    arms = []  # (arm, per-layer callable(i, x), prod?, stock?, build-note)
    built = {}
    ref = None

    def run_check(fn):
        caches = tuple(t.clone() for t in copies[0])
        got = torch.empty_like(out)
        fn(0, x, caches, got)
        torch.cuda.synchronize()
        return got

    # stock: cuBLAS in-projection + seam S1
    if not args.heads_only:
        try:
            layers = [k.stock_layer(attn, c) for c in copies]

            def stock(i, xx, caches=None, o=None):
                lay = layers[i] if caches is None else k.stock_layer(attn, caches)
                torch.mm(xx, Ws[i].t(), out=projected)
                k.stock_kda(lay, projected, out if o is None else o)

            ref = run_check(stock)
            arms.append(("stock", stock, False, True, "cuBLAS + S1"))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            built["stock"] = f"ERROR {type(e).__name__}: {str(e)[:200]}"
        # the pair: production in-projection tactic + PreparedKda
        try:
            pol = k.providers.projection_policies(16, PITCH, [rows], k.sms, q)
            projs = [k.PreparedProjection(w, pol).prepare() for w in Ws]

            def pair(i, xx, caches=None, o=None):
                projs[i].run(xx, projected)
                core.run(projected, out if o is None else o, caches=copies[i] if caches is None else caches,
                         slots=table, accepted=accepted)

            arms.append(("pair", pair, prod == "pair", False, f"inproj {pol.get(rows)}"))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            built["pair"] = f"ERROR {type(e).__name__}: {str(e)[:200]}"
    # heads (plain, and polled where production polls)
    variants = [False] + ([True] if rows in polled_rows else [])
    for polled in variants:
        arm = ("head_polled" if polled else "head") + tag
        try:
            with early_override(k, sweep) as tile:
                side = H.KdaSide(k.dev)
                counters = torch.zeros((L, 2), dtype=torch.int32, device=k.dev)
                fa_all = torch.empty((L, 64, FA_ROWS), dtype=torch.bfloat16, device=k.dev)
                batches = (8,) if (q == 1 or polled) else ()
                verify = () if q == 1 else ((rows, q),)
                heads = [H.PreparedKdaHead(Ws[i], core, batches, side, counters[i], fa_all[i], tile=tile,
                                           polled=(8,) if polled else (), verify=verify).prepare_caches(copies[i])
                         for i in range(L)]
            bufs = None
            if polled:
                bufs = torch.empty((L, 8, DIM), dtype=torch.bfloat16, device=k.dev)
                bufs.view(torch.int16).fill_(-1)

            def headrun(i, xx, caches=None, o=None, heads=heads, bufs=bufs, side=side):
                c = copies[i] if caches is None else caches
                dst, rearm = (out if o is None else o), None
                if bufs is not None:
                    dst, rearm = bufs[i][:rows], bufs[i - 1]
                if q == 1:
                    heads[i].run(xx, dst, caches=c, slots=table, rearm=rearm)
                else:
                    heads[i].run_verify(xx, dst, caches=c, slots=table, accepted=accepted, rearm=rearm)
                if o is not None:
                    side.join()
                    if bufs is not None:
                        o.copy_(dst)

            headrun.side = side
            arms.append((arm, headrun, arm == prod, False, f"tile {tile}"))  # sweep arms (@tag) never prod
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            built[arm] = f"ERROR {type(e).__name__}: {str(e)[:200]}"
    for arm, note in built.items():
        for mode in args.modes:
            sink.add(section="a", case=name, state=state, q=q, requests=requests, rows=rows, mode=mode, arm=arm,
                     prod=int(arm == prod), stock=int(arm == "stock"), layers=L, note=note)

    marks = torch.zeros(2 * L + 4, dtype=torch.int64, device=k.dev)
    done = torch.zeros(1, dtype=torch.int32, device=k.dev)
    spin = int(args.spin_us * 1000)
    for arm, fn, is_prod, is_stock, bnote in arms:
        err = None
        if ref is not None and not is_stock:
            try:
                err = relerr(run_check(fn), ref)
            except Exception as e:  # noqa: BLE001
                log(f"  check {arm} failed: {e}")
        side = getattr(fn, "side", None)
        for mode in args.modes:
            base = dict(section="a", case=name, state=state, q=q, requests=requests, rows=rows, arm=arm,
                        prod=int(is_prod), stock=int(is_stock), relerr=err, layers=L)
            try:
                if mode == "iso":
                    def body():
                        for i in range(L):
                            fn(i, x)
                        if side is not None:
                            side.join()

                    us, note, _ = graph_time(body, L, args.reps)
                    sink.add(**base, mode="iso", us=us, note=f"{note} {bnote}")
                else:
                    if standin_mod is None:
                        raise RuntimeError("stand-in extension unavailable")

                    def body():
                        for i in range(L):
                            standin_mod.standin(xs[i % len(xs)], x, spin, marks, 2 * i, False, 56, done)
                            fn(i, x)
                            standin_mod.mark(marks, 2 * i + 1, True)
                        if side is not None:
                            side.join()

                    us, note, delta = graph_time(body, L, args.reps, marks, L)
                    hint = PREV_HEAD_NOTES.get(name, "")
                    sink.add(**base, mode="chain", us=delta, note=f"{note} done-x_stored spin {args.spin_us} {bnote}"
                             + (f"; prev {hint}" if hint and is_prod else ""))
                    sink.add(**base, mode="chain_graph", us=us, note=f"{note} graph/layer incl stand-in {bnote}")
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                sink.add(**base, mode=mode, note=f"ERROR {type(e).__name__}: {str(e)[:200]}")
    del Ws, copies, arms
    torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------------------ section c
def run_gemm(k, sink, args):
    prov = k.providers
    rows_list = [8, 64, 264, 1024] if args.quick else gemm_rows()
    pitches = (3216,) if args.quick else GEMM_PITCHES
    for pitch in pitches:
        g = torch.Generator(device=k.dev).manual_seed(pitch)
        Ws = [(torch.randn(pitch, HIDDEN, generator=g, device=k.dev) * 0.02).to(torch.bfloat16) for _ in range(8)]
        lt = k.batch_table(prov.INPROJ_LT.get(pitch, "") if k.sms > 152 else "")
        case = {3216: "inproj-kda3216", 2880: "inproj-mla2880", 3040: "inproj-dsa3040"}[pitch]
        for rows in rows_list:
            where = ("c", case, "bf16", "", "", str(rows))
            if where in sink.done:
                continue
            if time.time() > args.deadline:
                sink.add(section="c", case=case, state="bf16", q="", requests="", rows=rows, mode="iso",
                         arm="cublas", stock=1, note="SKIPPED deadline")
                continue
            x = torch.randn(rows, HIDDEN, generator=g, device=k.dev).to(torch.bfloat16)
            out = torch.empty(rows, pitch, dtype=torch.bfloat16, device=k.dev)
            try:
                pol = prov.projection_policies(16, pitch, [rows], k.sms, 1)
            except Exception as e:  # noqa: BLE001
                pol, _ = {}, log(f"projection_policies({pitch}, {rows}) failed: {e}")
            prod = "splitk" if rows in pol else ("cublaslt" if lt.get(rows) else "cublas")
            base = dict(section="c", case=case, state="bf16", q="", requests="", rows=rows, mode="iso", layers=8)
            bit = ""
            try:
                with prov.blas_library("cublaslt"):
                    a = torch.mm(x, Ws[0].t())
                b = torch.mm(x, Ws[0].t())
                torch.cuda.synchronize()
                bit = f"lt==cublas bitwise {int(torch.equal(a, b))}"
            except Exception as e:  # noqa: BLE001
                bit = f"bitwise check failed {e}"
            for arm in ("cublas", "cublaslt", "splitk"):
                if arm == "splitk" and rows not in pol:
                    continue
                try:
                    if arm == "splitk":
                        projs = [k.PreparedProjection(w, pol).prepare() for w in Ws]

                        def body():
                            for j in range(16):
                                projs[j % 8].run(x, out)
                    else:
                        lib = "cublaslt" if arm == "cublaslt" else None

                        def body(lib=lib):
                            with prov.blas_library(lib):
                                for j in range(16):
                                    torch.mm(x, Ws[j % 8].t(), out=out)

                    us, note, _ = graph_time(body, 16, args.reps)
                    sink.add(**base, arm=arm, us=us, prod=int(arm == prod), stock=int(arm == "cublas"),
                             prev_bf16_us=prev_us("c", None, None, rows, arm, pitch),
                             note=f"{note} {bit} {('tactic ' + str(pol[rows])) if arm == 'splitk' else ''}".strip())
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    sink.add(**base, arm=arm, prod=int(arm == prod), stock=int(arm == "cublas"),
                             note=f"ERROR {type(e).__name__}: {str(e)[:200]}")
            for d in summarize([r for r in sink.rows if r["section"] == "c" and r["case"] == case
                                and str(r["rows"]) == str(rows)]):
                log(fmt_row(d))
            check_cuda_alive(sink, k.dev, where)
        del Ws
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------------------------ run
def run(args):
    args.deadline = time.time() + args.deadline_s
    args.modes = args.modes.split(",")
    args.arms = [a for a in args.arms.split(",") if a]
    args.skip_arms = [a for a in args.skip_arms.split(",") if a]
    os.makedirs(args.out, exist_ok=True)
    sink = Sink(os.path.join(args.out, f"{args.tag}.csv"))
    if sink.done:
        log(f"resume {args.tag}: {len(sink.done)} cases already recorded are skipped")
    dev, sms = gpu_setup()
    k = Kda(dev, sms, args.stock_norm)
    sections = args.sections.split(",")
    log(f"run {args.tag}: sections {sections} shard {args.shard or '-'} modes {args.modes} early {args.early or '-'}"
        f" reps {args.reps} mem {args.mem_gb} GB deadline in {args.deadline_s} s")
    with torch.inference_mode():
        if "c" in sections:
            run_gemm(k, sink, args)
        if "a" in sections:
            standin_mod = None
            if "chain" in args.modes:
                try:
                    sys.path.insert(0, HERE)
                    import standin

                    standin_mod = standin.mod()
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    log(f"stand-in extension failed ({e}): chain mode rows will be errors")
            shapes = HEAD_SHAPES[:1] if args.quick else HEAD_SHAPES
            states = ("fp32",) if args.quick else STATES
            cases = [(r, q, n, s) for r, q, n in shapes for s in states if head_case_applies(args.early, q, r)]
            for rows, q, name, state in shard(cases, args.shard):
                where = ("a", name, state, str(q), str(rows // q), str(rows))
                if where in sink.done:
                    continue
                if time.time() > args.deadline:
                    sink.add(section="a", case=name, state=state, q=q, requests=rows // q, rows=rows, mode="iso",
                             arm="stock", stock=1, note="SKIPPED deadline")
                    continue
                log(f"a {name} {state}")
                try:
                    run_head_case(k, sink, rows, q, name, state, args, standin_mod)
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    sink.add(section="a", case=name, state=state, q=q, requests=rows // q, rows=rows, mode="iso",
                             arm="case", note=f"ERROR {type(e).__name__}: {str(e)[:200]}")
                check_cuda_alive(sink, k.dev, where)
                torch.cuda.empty_cache()
                for d in summarize([r for r in sink.rows if r["section"] == "a" and r["case"] == name
                                    and r["state"] == state]):
                    log(fmt_row(d))
        if "b" in sections:
            for q, requests, state in shard(verify_cases(args.quick), args.shard):
                where = ("b", f"verify-q{q}", state, str(q), str(requests), str(q * requests))
                if where in sink.done or not any(arm_wanted(args, a) for a in
                                                 ["stock"] + [x[0] for x in verify_arms(q, requests, state)]):
                    continue
                if time.time() > args.deadline:
                    sink.add(section="b", case=f"verify-q{q}", state=state, q=q, requests=requests,
                             rows=q * requests, mode="iso", arm="stock", stock=1, note="SKIPPED deadline")
                    continue
                log(f"b q{q} B{requests} {state}")
                try:
                    run_verify_case(k, sink, q, requests, state, args)
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    sink.add(section="b", case=f"verify-q{q}", state=state, q=q, requests=requests,
                             rows=q * requests, mode="iso", arm="case", note=f"ERROR {type(e).__name__}: {str(e)[:200]}")
                check_cuda_alive(sink, k.dev, where)
                torch.cuda.empty_cache()
                for d in summarize([r for r in sink.rows if r["section"] == "b" and str(r["q"]) == str(q)
                                    and str(r["requests"]) == str(requests) and r["state"] == state]):
                    log(fmt_row(d))
    text, worse, retune, missing = report(sink.rows)
    log(f"done {args.tag}: {len(sink.rows)} rows, WORSE {len(worse)}, RETUNE {len(retune)}, MISSING {len(missing)}")
    print(text, flush=True)


# ------------------------------------------------------------------------------------------------------------- selftest
def selftest():
    vc = verify_cases()
    assert len(vc) == 2 * (len(Q4_REQUESTS) + len(Q8_REQUESTS)) and len(set(vc)) == len(vc)
    parts = [shard(vc, f"{i}/3") for i in range(3)]
    assert sorted(sum(parts, [])) == sorted(vc)
    rows = [sum(q * b for q, b, _ in p) for p in parts]
    assert max(rows) < 1.5 * min(rows), rows
    assert max(q * b for q, b, _ in vc) == 2048 and {q for q, _, _ in vc} == {4, 8}
    assert [a for a, _, _ in verify_arms(4, 128, "bf16")] == ["kdasplit", "wide", "halves", "stream3", "kdasplit1834"]
    assert [a for a, _, _ in verify_arms(8, 16, "bf16")] == ["kdasplit", "wide", "halves"]
    gr = gemm_rows()
    assert gr[0] == 8 and gr[-1] == 2048 and all(r % 4 == 0 for r in gr) and 772 in gr and 1024 in gr
    assert prev_us("b", 4, 64, 256, "stream3") == 10.29 and prev_us("b", 8, 16, 128, "halves") == 8.93
    assert prev_us("c", None, None, 300, "cublaslt", 3216) == 11.1 and prev_us("c", None, None, 128, "splitk", 3216)
    assert head_case_applies("verify=18", 8, 8) and not head_case_applies("verify=18", 8, 16)
    assert head_case_applies("multi=18", 4, 8) and head_case_applies("q1=13", 1, 8)
    assert n_copies(805 * 2**20, 40) == 50 and n_copies(10, 40) == KDA_LAYERS and n_copies(10**12, 1) == 2
    import tempfile

    d = tempfile.mkdtemp()
    s = Sink(os.path.join(d, "t.csv"))
    b = dict(section="b", case="verify-q4", state="fp32", q=4, requests=64, rows=256, mode="iso")
    s.add(**b, arm="stock", us=20.0, stock=1)
    s.add(**b, arm="wide", us=22.0, prod=1)
    s.add(**b, arm="kdasplit", us=15.0)
    s.add(**b, arm="halves", note="ERROR x")
    b2 = dict(b, state="bf16")
    s.add(**b2, arm="stock", us=18.0, stock=1)
    s.add(**b2, arm="stream3", us=10.0, prod=1)
    s.add(**dict(b, requests=32, rows=128), arm="stock", us=9.0, stock=1, note="eager")
    s.add(**dict(b, requests=32, rows=128), arm="wide", us=8.0, prod=1, note="graph x")
    s.f.close()
    summ = summarize(s.rows)
    assert summ[0]["ratio"] == 1.1 and not summ[0]["eager"] and summ[0]["best_arm"] == "kdasplit"
    assert "halves" in summ[0]["errors"] and summ[2]["eager"] == "stock", summ
    s_re = Sink(os.path.join(d, "t.csv"))  # resume: existing rows load, cases are done
    assert len(s_re.rows) == 8 and ("b", "verify-q4", "fp32", "4", "64", "256") in s_re.done, s_re.done
    s_re.f.close()
    ns = argparse.Namespace(arms=["kdasplit1834"])
    assert arm_wanted(ns, "kdasplit1834") and not arm_wanted(ns, "stock") and arm_wanted(argparse.Namespace(arms=[]), "x")
    ns = argparse.Namespace(arms=[], skip_arms=["kdasplit1834"])
    assert not arm_wanted(ns, "kdasplit1834") and arm_wanted(ns, "stock")
    s2 = Sink(os.path.join(d, "u.csv"))
    s2.add(section="c", case="inproj-kda3216", state="bf16", q="", requests="", rows=264, mode="iso", arm="cublas",
           us=10.0, stock=1)
    s2.f.close()
    merge(d)
    txt = open(os.path.join(d, "SUMMARY.txt")).read()
    assert "WORSE (1)" in txt and "RETUNE (1)" in txt and "MISSING (1)" in txt and "x2.20" in txt, txt
    assert "EAGER (1)" in txt and "EAGER(stock)" in txt, txt
    print("selftest ok:", len(vc), "verify cases,", len(gr), "gemm rows x", len(GEMM_PITCHES), "pitches,",
          len(HEAD_SHAPES) * len(STATES), "head cases")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--sections", default="a,b,c")
    r.add_argument("--shard", default="", help="i/n: this process's share of the section's cases")
    r.add_argument("--out", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--modes", default="iso,chain", help="section a: iso and/or chain")
    r.add_argument("--early", default="", help="section a early-tile override: q1=N | verify=N | multi=N")
    r.add_argument("--heads-only", action="store_true", help="section a: head arms only (early sweeps)")
    r.add_argument("--arms", default="", help="section b: only these arms (comma list; stock = time stock too)")
    r.add_argument("--skip-arms", default="", help="section b: not these arms (comma list)")
    r.add_argument("--spin-us", type=float, default=6.0, help="chain stand-in spin (the AttnRes hop wait)")
    r.add_argument("--reps", type=int, default=20)
    r.add_argument("--mem-gb", type=float, default=40.0, help="section b: cache-copy budget")
    r.add_argument("--deadline-s", type=float, default=8400.0, help="skip cases that start after this")
    r.add_argument("--quick", action="store_true")
    r.add_argument("--stock-norm", default="auto", choices=("auto", "cuda", "native"),
                   help="stock gated-norm dispatch (auto: CustomOp dispatch under VllmConfig())")
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    sub.add_parser("selftest")
    args = ap.parse_args()
    if args.cmd == "selftest":
        return selftest()
    if args.cmd == "merge":
        return merge(args.out)
    return run(args)


if __name__ == "__main__":
    main()
