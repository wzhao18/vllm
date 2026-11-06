# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA sparse core of one DCP16 rank on one GPU (SM107), per-tensor fp8 MLA KV.

The rank's sparse MLA over its local KV shard for every gathered query head (96 of
the 128-padded q mailbox rows), CUDA graphs of --inner calls, median of --reps:

  trtllm_fp8   (stock; also what WideDsa.dcp_core and PreparedDsa call at fp8) the
               FlashInfer trtllm-gen sparse MLA decode over the flat fp8 KV rows,
               128 query heads, LSE out, then the stock _finish_dcp_partials
               (empty-row identity) = "the trtllm-gen chain"
  dsac_<v>     dsa_kernels.dsa_core.DsaCore (the VLLM_K31_DSA_CORE path) variant v
               (1x16 1x32 2x16s2 4x16s2 8x16s4 u), pushing O + base-2 LSE into the
               16 head owners' DcpBox slots (16 owner boxes on this GPU: local
               stores, not NVLink; the multi-rank merge is bench_value.py's
               "pushed" arms), at rows <= 64 (the DcpBox cap); the code's policy
               (dsa_core.decode.VARIANT by q, ROWS) is the "spec" arm
  prims_fp8    (alt) FlashInfer prims_ts sparse MLA on the same fp8 rows
  prims_nvfp4  (ref: the previous nvfp4_ds_mla configuration) prims_ts on 352 B
               nvfp4_ds_mla rows (random bytes, finite) of the same slots

Count patterns per row (the rank's local entries of a top-k 2048, slots over 1M KV
rows): "bal" 96-155 per row; "x<N>": the q rows of request 0 hold N entries (one
rank owns most of a row's selection), the others bal. --part small: rows <= 64,
every pattern; --part wide: rows >= 96 (the WideDsa rows), "bal".

Previous numbers (dsa_core/decode.py, Round 3 Step 0a, CUPTI us per call, chain /
1x16 / 2x16s2 / 4x16s2 / 8x16s4, q8): M8 bal 13.47 / 9.48 / 9.45 / 8.73 / 10.43,
M16 bal 14.34 / 9.65 / 9.71 / 16.25 / 17.66, M8 x2048 17.17 / 78.56 / 43.72 /
49.28 / 47.25 (events over graphs here, so a few % above CUPTI).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import DCP, GATHERED, HPR, Q_HEADS, TOPK, Recorder  # noqa: E402

NKV = 1 << 20
CAP = 64
VARIANTS = ("1x16", "1x32", "2x16s2", "4x16s2", "8x16s4", "u")
PATTERNS = ("bal", "x256", "x384", "x600", "x2048")


class CoreBench:
    def __init__(self, dev, rec, args):
        self.dev, self.rec, self.args = dev, rec, args
        gen = torch.Generator(device=dev).manual_seed(11)
        kv = torch.randn(NKV, 576, generator=gen, device=dev) * 0.5
        self.k_scale = float(kv.abs().max()) / 448.0
        self.kv8 = (kv / self.k_scale).to(torch.float8_e4m3fn)
        del kv
        q = torch.randn(4096, GATHERED, 576, generator=gen, device=dev)
        self.q_scale = float(q.abs().max()) / 448.0
        q8 = (q / self.q_scale).to(torch.float8_e4m3fn)
        del q
        # the q mailbox rows (FP8 bytes, 128 padded heads; parity 2 x rows)
        self.q_rows = torch.zeros(4096, Q_HEADS, 576, dtype=torch.uint8, device=dev)
        self.q_rows[:, :GATHERED] = q8.view(torch.uint8)
        del q8
        self.sm_scale = 1.0 / math.sqrt(192)
        self.bmm1 = self.sm_scale * self.q_scale * self.k_scale
        self.bmm2 = self.k_scale
        self.nvfp4 = None
        self.core = None
        self.core_err = None

    # --- inputs -------------------------------------------------------------------
    def case(self, rows, q, pattern):
        gen = torch.Generator(device=self.dev).manual_seed(rows * 31 + q + len(pattern))
        cnt = common.balanced_counts(rows, gen, self.dev)
        if pattern.startswith("x"):
            cnt[:q] = int(pattern[1:])
        sel = torch.randint(0, NKV, (rows, TOPK), generator=gen, device=self.dev)
        sel = sel.int()
        sel[torch.arange(TOPK, device=self.dev)[None, :] >= cnt[:, None].long()] = -1
        return sel.contiguous(), cnt.contiguous()

    # --- arms ---------------------------------------------------------------------
    def trtllm(self, rows, sel, cnt):
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fims

        dev = self.dev
        fims._reserve_kv_counters(dev, max(rows, 64), Q_HEADS)
        pages = self.kv8.view(-1, 64, 576)
        out = torch.empty(rows, 1, Q_HEADS, 512, dtype=torch.bfloat16, device=dev)
        lse = torch.empty(rows, Q_HEADS, dtype=torch.float32, device=dev)
        # production size (FlashInferMLASparseTRTLLMMetadataBuilder under DCP): the
        # base plus the softmax-stats slab of 128 heads x 256 slots per row (LSE out)
        need = fims._required_workspace_bytes(DCP, HPR, max(rows, 2048))
        workspace = fims._get_workspace_buffer(dev, need)
        query = self.q_rows[:rows].view(torch.float8_e4m3fn).unsqueeze(1)
        tables = sel.unsqueeze(1)
        extra = fims._kv_counters(rows, Q_HEADS, dev)
        finish = getattr(fims, "_finish_dcp_partials", None)

        def call():
            trtllm_batch_decode_with_kv_cache_mla(
                query=query,
                kv_cache=pages.unsqueeze(1),
                workspace_buffer=workspace,
                qk_nope_head_dim=128,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                block_tables=tables,
                seq_lens=cnt,
                max_seq_len=TOPK,
                out=out,
                bmm1_scale=self.bmm1,
                bmm2_scale=self.bmm2,
                sparse_mla_top_k=TOPK,
                return_lse=True,
                lse=lse,
                **extra,
            )
            if finish is not None:
                finish(out[:, 0, :GATHERED], lse[:, :GATHERED], cnt)

        return call

    def prims(self, rows, sel, cnt, nvfp4):
        from flashinfer.attention.prims_ts import (
            BatchSparseMLADecodePagedTSWrapper,
            get_prims_ts_sparse_mla_decode_workspace_size,
        )

        from vllm.utils.flashinfer import prims_ts_sparse_mla_schedules_native_heads

        dev = self.dev
        heads = GATHERED if prims_ts_sparse_mla_schedules_native_heads() else Q_HEADS
        fmt = "nvfp4.fp8rope" if nvfp4 else "native"
        kwargs = dict(max_topk=TOPK, max_q_len=1, packed_query=True,
                      q_dtype=torch.float8_e4m3fn, qk_rope_head_dim=64,
                      kv_cache_format=fmt, return_lse=True)  # fmt: skip
        size = get_prims_ts_sparse_mla_decode_workspace_size(dev, rows, heads, **kwargs)
        workspace = torch.empty(size, dtype=torch.uint8, device=dev)
        plan = BatchSparseMLADecodePagedTSWrapper(workspace)
        plan.plan(dev, rows, heads, **kwargs)
        indptr = torch.arange(rows + 1, dtype=torch.int32, device=dev)
        query = self.q_rows[:rows, :heads].contiguous().view(torch.float8_e4m3fn)
        if nvfp4:
            if self.nvfp4 is None:
                gen = torch.Generator(device=dev).manual_seed(5)
                self.nvfp4 = torch.randint(0, 64, (NKV, 352), generator=gen, device=dev,
                                           dtype=torch.uint8)  # (finite E2M1 / E4M3)
            kv, descale = self.nvfp4, 1.0
        else:
            kv, descale = self.kv8, self.k_scale
        out = torch.empty(rows, heads, 512, dtype=torch.bfloat16, device=dev)
        lse = torch.empty(rows, heads, dtype=torch.float32, device=dev)
        scale = self.sm_scale * self.q_scale * descale

        def call():
            plan.run(query, kv, sel, cnt, qo_indptr=indptr, softmax_scale=scale,
                     output_scale=descale, out=out, lse=lse, validate=False)  # fmt: skip

        return call

    def dsac(self, rows, sel, cnt, variant, lse_wide):
        from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core.decode import (
            parse_variant,
        )
        from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core.ops import DsaCore

        dev = self.dev
        if self.core is None:
            self.core = DsaCore(dev, cap=CAP, heads=GATHERED, max_splits=8)
            # 16 owners' boxes [2, DCP, CAP, HPR, 512] / [2, DCP, CAP, HPR]
            shape = (DCP, 2, DCP, CAP, HPR)
            self.part = torch.zeros(*shape, 512, dtype=torch.bfloat16, device=dev)
            self.lse = torch.zeros(shape, dtype=torch.float32, device=dev)
            self.peers = torch.tensor([p.data_ptr() for p in self.part], device=dev)
            self.lpeers = torch.tensor([v.data_ptr() for v in self.lse], device=dev)
            self.q_box = self.q_rows[: 2 * CAP].view(2, CAP, Q_HEADS, 576)
        splits, heads, split_min = parse_variant(variant)
        core = self.core

        def call(parity):
            core.run(self.q_box[parity], sel, cnt, self.kv8.view(torch.uint8),
                     bmm1_scale=self.bmm1, bmm2_scale=self.bmm2,
                     part_peers=self.peers, lse_peers=self.lpeers, box_cap=CAP,
                     hpr=HPR, parity=parity, rank=5, rows=rows, splits=splits,
                     split_min=split_min, heads_per_cta=heads or 32, umma=heads == 0,
                     lse_wide=lse_wide, pdl=True)  # fmt: skip

        return common.Alternating(call)

    # --- policy -------------------------------------------------------------------
    @staticmethod
    def policy(rows, q):
        """(variant name or None, spread) of the code's DSA core policy at these rows
        (as if VLLM_K31_DSA_CORE=1; default off: the served op is trtllm_fp8)."""
        from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core import decode

        if rows > decode.ROWS:
            return None, False
        table = decode.rows_table(decode.VARIANT.get(q, "").replace(";", ","))
        spread = decode.rows_table(decode.SPREAD.replace(";", ",")).get(rows)
        return table.get(rows), bool(spread)

    def time(self, fn):
        return common.graph_time(fn, self.args.inner, self.args.reps)

    def run_case(self, rows, q, pattern):
        rec = self.rec
        if not rec.should_run(pattern, q, rows):
            return
        try:
            self.run_arms(rows, q, pattern)
        finally:
            rec.finish(pattern, q, rows)

    def run_arms(self, rows, q, pattern):
        rec, a = self.rec, self.args
        sel, cnt = self.case(rows, q, pattern)
        extra = dict(max_count=int(cnt.max()), mean_count=round(float(cnt.float().mean()), 1))
        try:
            pol, spread = self.policy(rows, q)
        except Exception as e:  # noqa: BLE001
            common.log_exc("policy", e)
            pol, spread = None, False
        arms = [("trtllm_fp8", "stock", lambda: self.trtllm(rows, sel, cnt))]
        if not a.no_prims:
            arms.append(("prims_fp8", "alt", lambda: self.prims(rows, sel, cnt, False)))
            arms.append(("prims_nvfp4", "ref", lambda: self.prims(rows, sel, cnt, True)))
        if rows <= CAP and not a.no_dsac:
            for v in VARIANTS:
                role = "spec" if v == pol else "tactic"
                arms.append((f"dsac_{v}", role,
                             lambda v=v: self.dsac(rows, sel, cnt, v, spread)))  # fmt: skip
        spec_arm = f"dsac_{pol}" if (pol and rows <= CAP and not a.no_dsac) else "trtllm_fp8"
        note = "VLLM_K31_DSA_CORE policy" if pol else "policy: trtllm-gen (no core)"
        for name, role, build in arms:
            try:
                fn = build()
                us = self.time(fn)
                status, err = "ok", None
                if name.startswith("dsac") and self.core is not None:
                    torch.cuda.synchronize()
                    code = int(self.core.err.item())
                    if code:
                        status, err = "fail", f"DsaCore error word {code}"
                        self.core.err.zero_()
                rec.emit(pattern, q, rows, name, us, role=role, status=status, err=err,
                         policy=name == spec_arm, note=note, **extra)  # fmt: skip
            except Exception as e:  # noqa: BLE001
                common.log_exc(f"core {pattern} q{q} M{rows} {name}", e)
                rec.emit(pattern, q, rows, name, None, role=role, status="fail",
                         err=common.error_text(e), policy=name == spec_arm)  # fmt: skip
                common.exit_if_sticky(e, f"core {pattern} q{q} M{rows} {name}")
                try:
                    torch.cuda.synchronize()
                except Exception as e2:  # noqa: BLE001
                    common.log_exc(f"core sync after {name}", e2)
                    common.exit_if_sticky(e2, f"core sync after {name}")
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress", default=None)
    ap.add_argument("--part", choices=("small", "wide"), default="small")
    ap.add_argument("--qs", default="4,8")
    ap.add_argument("--rows", default="", help="subset of the in-scope rows")
    ap.add_argument("--patterns", default=",".join(PATTERNS))
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--no-prims", action="store_true")
    ap.add_argument("--no-dsac", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    print("core bench on", common.device_info(dev), flush=True)
    rec = Recorder(args.out, args.progress or args.out + ".progress", "core")
    bench = CoreBench(dev, rec, args)
    only = set(common.parse_ints(args.rows))
    cases: list = []
    for q in common.parse_ints(args.qs):
        if args.part == "small":
            cases += [(r, q) for r in common.rows_for(q, hi=CAP, only=only)]
        else:
            cases += [(r, q) for r in common.rows_for(q, lo=CAP + 1, only=only)]
    patterns = [p for p in args.patterns.split(",") if p] if args.part == "small" else ["bal"]
    for rows, q in common.by_priority(cases):
        for pattern in patterns:
            try:
                bench.run_case(rows, q, pattern)
            except Exception as e:  # noqa: BLE001
                common.log_exc(f"core case {pattern} q{q} M{rows}", e)
                common.exit_if_sticky(e, f"core case {pattern} q{q} M{rows}")
    print("core bench done", flush=True)


if __name__ == "__main__":
    main()
