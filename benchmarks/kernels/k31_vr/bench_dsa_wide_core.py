# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wide-row DSA sparse MLA core on one VR200 GPU (per-tensor FP8 MLA KV):
FlashInfer trtllm-gen (the production chain) vs the candidate wide core
(specialized_decode/cutedsl/dsa/wide_core.py), e.g.

  python benchmarks/kernels/k31_vr/bench_dsa_wide_core.py time --arms trtllm,wide --out t.jsonl
  python benchmarks/kernels/k31_vr/bench_dsa_wide_core.py probe --arms trtllm,wide --out p.jsonl


Production rows of the K3.1 specialized decode's DSA sparse core (WideDsa.dcp_core /
SpDsa: FlashInfer trtllm-gen over the gathered q mailbox, 96 real heads padded to
128):
  dcp16  TP16 = DCP16 (C160 / C256 q4, C48 / C64 q8): ~96-155 local top-k rows / row
  dcp4   C512 SP (TP4 = DCP4 per DP group): ~440-580 local top-k rows / row
Rows of one request (q of them) draw their selections from one shared pool (the
consecutive verify tokens' top-k overlap), slots over 4M FP8 KV rows (2.3 GB).

Modes
  time   per arm: "cold" = graph of inner x (L2 flush + call) minus inner x flush,
         "warm" = graph of inner back-to-back calls; achieved bytes / BW
  probe  trtllm-gen numerics vs emulated variants (P scale, l from P or P8, rescale
         threshold) -> which math the stock kernel computes
  ncu    one call per case (for ncu -k regex)
Arms: trtllm (the stock chain: trtllm-gen + _finish_dcp_partials), and "wide" (the
specialized core, if importable)."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys

import torch

H, QH, D, NOPE = 96, 128, 576, 512
TOPK = 2048
NKV = 1 << 22
LOG2E = 1.4426950408889634


def counts_for(kind, rows, gen, dev):
    lo, hi = (96, 155) if kind == "dcp16" else (440, 580)
    return torch.randint(lo, hi + 1, (rows,), generator=gen, device=dev).int()


def make_sel(kind, rows, q, cnt, gen, dev, nkv=None):
    """[rows, TOPK] int32 flat KV rows (compacted prefix of cnt), -1 after."""
    nkv = NKV if nkv is None else nkv
    reqs = rows // q
    pool_n = int(1.35 * (155 if kind == "dcp16" else 580))
    pools = torch.randint(0, nkv, (reqs, pool_n), generator=gen, device=dev)
    sel = torch.full((rows, TOPK), -1, dtype=torch.int32, device=dev)
    # each row: a random permutation of its request's pool, first cnt entries
    keys = torch.rand(rows, pool_n, generator=gen, device=dev)
    order = keys.argsort(dim=1)
    rowpool = pools.repeat_interleave(q, dim=0)
    picked = torch.gather(rowpool, 1, order)
    # (duplicates within a pool are possible but rare; keep them unique per row)
    picked, _ = picked.sort(dim=1)
    dup = torch.zeros_like(picked, dtype=torch.bool)
    dup[:, 1:] = picked[:, 1:] == picked[:, :-1]
    picked = picked.masked_fill(dup, -1)
    # shuffle again so -1 holes scatter, then compact valid to the front
    picked = torch.gather(picked, 1, torch.rand(rows, pool_n, generator=gen,
                                                device=dev).argsort(dim=1))
    valid = picked >= 0
    pos = valid.int().cumsum(1) - 1
    take = valid & (pos < cnt[:, None])
    out_cnt = take.sum(1).int()
    r = torch.arange(rows, device=dev)[:, None].expand_as(picked)
    sel[r[take], pos[take]] = picked[take].int()
    return sel, out_cnt


class Bench:
    def __init__(self, dev, maxrows):
        self.dev = dev
        gen = torch.Generator(device=dev).manual_seed(7)
        # random E4M3 bytes without NaN (bit 6 cleared: |x| < 2)
        self.kv8 = (torch.randint(0, 256, (NKV, D), generator=gen, device=dev,
                                  dtype=torch.uint8) & 0xBF)
        self.q = torch.zeros(maxrows, QH, D, dtype=torch.uint8, device=dev)
        self.q[:, :H] = torch.randint(0, 256, (maxrows, H, D), generator=gen,
                                      device=dev, dtype=torch.uint8) & 0xBF
        self.k_scale, self.q_scale = 0.02, 0.03
        self.sm_scale = 1.0 / math.sqrt(192)
        self.bmm1 = self.sm_scale * self.q_scale * self.k_scale
        self.bmm2 = self.k_scale
        self.flush = torch.empty(1 << 28, dtype=torch.uint8, device=dev)  # 256 MB

    def trtllm(self, rows, sel, cnt, kv8=None):
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fims

        dev = self.dev
        kv8 = self.kv8 if kv8 is None else kv8
        fims._reserve_kv_counters(dev, max(rows, 64), QH)
        pages = kv8.view(torch.float8_e4m3fn).view(-1, 64, D)
        out = torch.empty(rows, 1, QH, NOPE, dtype=torch.bfloat16, device=dev)
        lse = torch.empty(rows, QH, dtype=torch.float32, device=dev)
        need = fims._required_workspace_bytes(16, 6, max(rows, 2048))
        workspace = fims._get_workspace_buffer(dev, need)
        query = self.q[:rows].view(torch.float8_e4m3fn).unsqueeze(1)
        tables = sel.unsqueeze(1)
        extra = fims._kv_counters(rows, QH, dev)
        finish = fims._finish_dcp_partials

        def call():
            trtllm_batch_decode_with_kv_cache_mla(
                query=query, kv_cache=pages.unsqueeze(1), workspace_buffer=workspace,
                qk_nope_head_dim=128, kv_lora_rank=512, qk_rope_head_dim=64,
                block_tables=tables, seq_lens=cnt, max_seq_len=TOPK, out=out,
                bmm1_scale=self.bmm1, bmm2_scale=self.bmm2, sparse_mla_top_k=TOPK,
                return_lse=True, lse=lse, **extra)
            finish(out[:, 0, :H], lse[:, :H], cnt)

        return call, out[:, 0], lse

    def trtllm_mf(self, rows, sel, cnt, kv8=None):
        """The chain as WideDsa.dcp_core ran it before (+ CompareEq + masked_fill)."""
        fn, out, lse = self.trtllm(rows, sel, cnt, kv8)
        real = lse[:, :H]

        def call():
            fn()
            real.masked_fill((cnt == 0).view(-1, 1), float("-inf"))

        return call, out, lse

    def wide(self, rows, sel, cnt, kv8=None):
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import wide_core

        kv8 = self.kv8 if kv8 is None else kv8
        out = torch.empty(rows, QH, NOPE, dtype=torch.bfloat16, device=self.dev)
        lse = torch.empty(rows, QH, dtype=torch.float32, device=self.dev)
        core = wide_core.WideCore(self.dev)

        def call():
            core.run(self.q[:rows], sel, cnt, kv8, self.bmm1, self.bmm2, out, lse, H)

        return call, out, lse

    def graph(self, fn, inner, flush):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(inner):
                if flush:
                    self.flush.add_(1)
                if fn is not None:
                    fn()
        return g

    def time_graph(self, g, inner, reps):
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(2_000_000)
            e0.record()
            g.replay()
            e1.record()
            e1.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000 / inner)
        return statistics.median(ts)

    def time(self, fn, inner=10, reps=15):
        noop = lambda: None  # noqa: E731
        gf = self.graph(noop, inner, True)
        flush_us = self.time_graph(gf, inner, reps)
        gc = self.graph(fn, inner, True)
        cold = self.time_graph(gc, inner, reps) - flush_us
        gw = self.graph(fn, inner, False)
        warm = self.time_graph(gw, inner, reps)
        del gf, gc, gw
        return cold, warm


def bytes_of(rows, sel, cnt, q):
    """Useful bytes (96 heads): q, KV (per row, and unique per request), out, lse,
    slots; and what the 128-head trtllm-gen moves."""
    tot_kv = int(cnt.long().sum()) * D
    uniq = 0
    s = sel.view(rows // q, q * TOPK)
    for r in range(s.shape[0]):
        u = torch.unique(s[r])
        uniq += int((u >= 0).sum())
    uniq *= D
    qb, ob, lb = rows * H * D, rows * H * NOPE * 2, rows * H * 4
    slots = int(cnt.long().sum()) * 4
    return dict(q=qb, kv_rows=tot_kv, kv_unique=uniq, out=ob, lse=lb, slots=slots,
                useful=qb + uniq + ob + lb + slots,
                trtllm=rows * QH * D + uniq + rows * QH * NOPE * 2 + rows * QH * 4 + slots)


CASES = {
    "dcp16": {4: [64, 128, 256, 384, 512, 768, 1024], 8: [64, 128, 256, 512]},
    "dcp4": {4: [64, 128, 256, 512]},
}


def cmd_time(args, dev):
    b = Bench(dev, 1024)
    arms = args.arms.split(",")
    with open(args.out, "a") as f:
        for kind, byq in CASES.items():
            if args.kinds and kind not in args.kinds.split(","):
                continue
            for q, rowlist in byq.items():
                for rows in rowlist:
                    gen = torch.Generator(device=dev).manual_seed(rows * 7 + q)
                    cnt = counts_for(kind, rows, gen, dev)
                    sel, cnt = make_sel(kind, rows, q, cnt, gen, dev)
                    by = bytes_of(rows, sel, cnt, q)
                    for arm in arms:
                        try:
                            fn = getattr(b, arm)(rows, sel, cnt)[0]
                            cold, warm = b.time(fn)
                            rec = dict(kind=kind, q=q, rows=rows, arm=arm, nkv=NKV, cold_us=round(cold, 2),
                                       warm_us=round(warm, 2), mean_cnt=round(float(cnt.float().mean()), 1),
                                       bytes=by,
                                       useful_TBps_cold=round(by["useful"] / cold / 1e6, 2),
                                       sol_us=round(by["useful"] / 13.82e6, 2))
                        except Exception as e:  # noqa: BLE001
                            rec = dict(kind=kind, q=q, rows=rows, arm=arm, err=repr(e)[:400])
                        print("REC", json.dumps(rec), flush=True)
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                    torch.cuda.empty_cache()


def e4m3(x):
    return x.to(torch.float8_e4m3fn).float()


def emulate(q8, k8, n, bmm1, bmm2, pscale_log2, T, l_from_p8, tile=128):
    """One row: q8 [H, 576] e4m3 (float), k8 [n, 576] (float). Online softmax over
    tiles of ``tile`` keys in base 2; P = e4m3(2^(s - m_anchor) * 2^pscale_log2);
    anchor moves only when the tile max exceeds it by > T. Returns O (fp32) [H, 512]
    and base-2 LSE [H]."""
    s = (q8 @ k8.T) * (bmm1 * LOG2E)  # [H, n] base 2
    Hh = q8.shape[0]
    m = torch.full((Hh,), -float("inf"), device=q8.device)
    l = torch.zeros(Hh, device=q8.device)
    acc = torch.zeros(Hh, NOPE, device=q8.device)
    for t0 in range(0, n, tile):
        st = s[:, t0:t0 + tile]
        tm = st.amax(1)
        new = torch.where(tm > m + T, tm, m)
        alpha = torch.where(torch.isinf(m), torch.zeros_like(m), torch.exp2(m - new))
        acc = acc * alpha[:, None]
        l = l * alpha
        p = torch.exp2(st - new[:, None])
        p8 = e4m3(p * 2.0 ** pscale_log2)
        l = l + (p8.sum(1) / 2.0 ** pscale_log2 if l_from_p8 else p.sum(1))
        acc = acc + p8 @ k8[t0:t0 + tile, :NOPE]
        m = new
    o = acc * (bmm2 / 2.0 ** pscale_log2) / l[:, None]
    return o, m + torch.log2(l)


def cmd_probe(args, dev):
    gen = torch.Generator(device=dev).manual_seed(3)
    nkv = 1 << 16
    kv = torch.randn(nkv, D, generator=gen, device=dev)
    k_scale = float(kv.abs().max()) / 448.0
    kv8 = (kv / k_scale).to(torch.float8_e4m3fn)
    b = Bench.__new__(Bench)
    b.dev = dev
    rows = 64
    qf = torch.randn(rows, H, D, generator=gen, device=dev) * 2.0
    q_scale = float(qf.abs().max()) / 448.0
    q8 = (qf / q_scale).to(torch.float8_e4m3fn)
    b.q = torch.zeros(rows, QH, D, dtype=torch.uint8, device=dev)
    b.q[:, :H] = q8.view(torch.uint8)
    b.k_scale, b.q_scale = k_scale, q_scale
    b.sm_scale = 1.0 / math.sqrt(192)
    b.bmm1 = b.sm_scale * q_scale * k_scale
    b.bmm2 = k_scale
    counts = [1, 7, 64, 100, 127, 128, 129, 155, 200, 256, 300, 512, 580, 1000, 2048] * 5
    counts = counts[:rows - 1] + [0]
    cnt = torch.tensor(counts, dtype=torch.int32, device=dev)
    sel = torch.full((rows, TOPK), -1, dtype=torch.int32, device=dev)
    for r, n in enumerate(counts):
        if n:
            sel[r, :n] = torch.randperm(nkv, generator=gen, device=dev)[:n].int()
    arms = {"trtllm": b.trtllm}
    if args.arms and "wide" in args.arms:
        arms["wide"] = b.wide
    outs = {}
    for name, mk in arms.items():
        try:
            fn, out, lse = mk(rows, sel, cnt, kv8.view(torch.uint8))
            fn()
            torch.cuda.synchronize()
            outs[name] = (out[:, :H].float().clone(), lse[:, :H].clone())
        except Exception as e:  # noqa: BLE001
            print("PROBE arm", name, "failed", repr(e)[:500], flush=True)
    kf = kv8.float()
    qq = q8.float()
    variants = [(math.log2(448.0) - T, T, lp) for T in (0.0, 8.0) for lp in (False, True)]
    variants += [(8.0, 0.0, lp) for lp in (False, True)]
    res = []
    for r, n in enumerate(counts):
        if n == 0:
            for name, (o, l) in outs.items():
                print(f"PROBE row {r} n 0 {name}: max|o| {float(o[r].abs().max()):.3g} "
                      f"lse {float(l[r].max()):.4g}", flush=True)
            continue
        k8 = kf[sel[r, :n].long()]
        exact_s = (qq[r] @ k8.T) * b.bmm1
        exact_o = torch.softmax(exact_s, -1) @ (k8[:, :NOPE] * k_scale)
        line = dict(row=r, n=n)
        for name, (o, l) in outs.items():
            ob = o[r].to(torch.bfloat16)
            line[f"{name}_err_exact"] = float((o[r] - exact_o).abs().max() / exact_o.abs().max())
            best = None
            for ps, T, lp in variants:
                eo, el = emulate(qq[r], k8, n, b.bmm1, b.bmm2, ps, T, lp)
                eb = eo.to(torch.bfloat16)
                eq = float((eb == ob).float().mean())
                ulp = float(((eb.float() - ob.float()).abs() /
                             eb.float().abs().clamp_min(1e-30)).max())
                dl = float((el - l[r]).abs().max())
                key = f"ps{ps:.2f}_T{T:g}_{'lp8' if lp else 'lp32'}"
                line[f"{name}:{key}"] = (round(eq, 4), round(ulp, 5), round(dl, 6))
                if best is None or eq > best[1]:
                    best = (key, eq)
            line[f"{name}_best"] = best
        if "wide" in outs and "trtllm" in outs:
            a = outs["trtllm"][0][r].to(torch.bfloat16)
            w = outs["wide"][0][r].to(torch.bfloat16)
            line["wide_vs_trtllm_eq"] = float((a == w).float().mean())
            line["wide_vs_trtllm_rel"] = float((a.float() - w.float()).abs().max() /
                                               a.float().abs().max())
            line["wide_vs_trtllm_lse"] = float((outs["trtllm"][1][r] - outs["wide"][1][r]).abs().max())
        res.append(line)
        print("PROBE", json.dumps(line), flush=True)
    with open(args.out, "a") as f:
        for line in res:
            f.write(json.dumps(line) + "\n")


def cmd_ncu(args, dev):
    b = Bench(dev, 1024)
    for spec in args.cases.split(";"):
        kind, q, rows = spec.split(",")
        q, rows = int(q), int(rows)
        gen = torch.Generator(device=dev).manual_seed(rows * 7 + q)
        cnt = counts_for(kind, rows, gen, dev)
        sel, cnt = make_sel(kind, rows, q, cnt, gen, dev)
        for arm in args.arms.split(","):
            fn = getattr(b, arm)(rows, sel, cnt)[0]
            torch.cuda.synchronize()
            print(f"NCU case {kind} q{q} M{rows} {arm} mean_cnt {float(cnt.float().mean()):.1f}",
                  json.dumps(bytes_of(rows, sel, cnt, q)), flush=True)
            fn()
            torch.cuda.synchronize()


def cmd_dbg(args, dev):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import wide_core

    b = Bench(dev, 1024)
    core = wide_core.WideCore(dev)
    for spec in args.cases.split(";"):
        kind, q, rows = spec.split(",")
        q, rows = int(q), int(rows)
        gen = torch.Generator(device=dev).manual_seed(rows * 7 + q)
        cnt = counts_for(kind, rows, gen, dev)
        sel, cnt = make_sel(kind, rows, q, cnt, gen, dev)
        out = torch.empty(rows, QH, NOPE, dtype=torch.bfloat16, device=dev)
        lse = torch.empty(rows, QH, dtype=torch.float32, device=dev)
        for cta in (0, 101):
            for rep in range(3):
                dbg = torch.zeros(32 * 8, dtype=torch.int64, device=dev)
                b.flush.add_(1)
                core.run(b.q[:rows], sel, cnt, b.kv8, b.bmm1, b.bmm2, out, lse, H, dbg=dbg,
                         dbg_cta=cta)
                torch.cuda.synchronize()
            t = dbg.view(32, 8).cpu()
            t0 = int(t[0, 0])
            print(f"DBG {kind} q{q} M{rows} cta {cta}: per item (us from first item start): "
                  "start, data(mma thread), qk, softmax, pv, epilogue, -, Q issued", flush=True)
            for k in range(32):
                if int(t[k, 0]) == 0 and int(t[k, 6]) == 0:
                    break
                v = [(int(x) - t0) / 1000 if int(x) else -1 for x in t[k, :8]]
                print("  item", k, " ".join(f"{x:7.2f}" for x in v), flush=True)


def main():
    global NKV
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("time", "probe", "ncu", "dbg"))
    ap.add_argument("--out", default="/dev/null")
    ap.add_argument("--arms", default="trtllm")
    ap.add_argument("--kinds", default="")
    ap.add_argument("--cases", default="dcp16,4,512")
    ap.add_argument("--nkv", type=int, default=NKV, help="FP8 KV rows (slots range)")
    args = ap.parse_args()
    NKV = args.nkv
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    print("device", torch.cuda.get_device_name(dev), torch.cuda.get_device_capability(dev),
          flush=True)
    {"time": cmd_time, "probe": cmd_probe, "ncu": cmd_ncu, "dbg": cmd_dbg}[args.mode](args, dev)
    print("bench done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
