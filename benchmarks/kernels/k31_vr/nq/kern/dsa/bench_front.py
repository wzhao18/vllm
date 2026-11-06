# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The query front of a K3.1 DSA layer at WideDsa's rows (> 64) on one node: 4 real
ranks (torchrun --nproc-per-node 4, NVLink multicast) + 12 emulated ones, DCP16 rank
shapes (6 heads, the 128-row FP8 q mailbox), per-tensor fp8 MLA KV.

  mlaquery      (spec) WideDsa's MlaQuery as compile_core builds it (dsa, fp8,
                dcp 16, q_heads 128; tile poll above WIDE_MAX_ROWS): q_a RMSNorm,
                q_b, W_UK absorption, FP8 query, kv_a RMSNorm + FP8 KV insert of this
                rank's tokens (1 / 16 of the rows), the multicast publish into every
                rank's q mailbox and the gathered-mailbox poll. The emulated ranks'
                rows of the mailbox stay published; the 4 real ranks' rows are
                re-armed after every call (as DcpValue does), so the poll waits for
                the real peers. Net of the re-arm: (call + re-arm) - (re-arm alone).
  mlaquery_nvfp4 (ref, the previous nvfp4_ds_mla configuration) the same with the
                nvfp4_ds_mla KV insert (352 B rows)
  stock_front   (stock, APPROXIMATE) seam S2-DSA's query path: q_a / kv_a RMSNorm,
                q_b GEMM of the rank's 6 heads, W_UK bmm, FP8 cast,
                concat_and_cache_mla "fp8", and the VR recipe's direct DCP q gather
                (DirectDCPQGatherWorkspace, padded to 128 heads) over the 4 real
                ranks at 24 heads per rank (the 96 gathered rows; 72 of 90 remote
                head rows of DCP16)

Not here: the indexer front (WideDsa run_index / DsaIndex vs the stock wq_index +
fused key insert) and the indexer core / sparse core, which both paths share.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import DCP, GATHERED, HPR, PITCH, Q_HEADS, Recorder  # noqa: E402

FRONT_ROWS = (96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
EPS = 1e-6


class FrontBench:
    def __init__(self, args, group, dev, rank, world):
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import DcpBox

        self.args, self.group, self.dev = args, group, dev
        self.rank, self.world = rank, world
        self.sms = torch.cuda.get_device_properties(dev).multi_processor_count
        bf = torch.bfloat16
        gen = torch.Generator(device=dev).manual_seed(40 + rank)

        def randn(*shape, scale=1.0):
            return (torch.randn(*shape, generator=gen, device=dev) * scale).to(bf)

        self.weights = (
            (torch.rand(1536, generator=gen, device=dev) + 0.5).to(bf),
            (torch.rand(512, generator=gen, device=dev) + 0.5).to(bf),
            randn(HPR * 192, 1536, scale=1 / 40),
            randn(HPR, 128, 512, scale=1 / 12),
            randn(HPR, 128, 512, scale=1 / 16).transpose(1, 2),
        )
        rows = args.max_rows
        self.blocks = -(-rows // 64) + 2
        self.kv = torch.zeros(self.blocks, 64, 576, dtype=torch.uint8, device=dev)
        self.kv4 = torch.zeros(self.blocks, 64, 352, dtype=torch.uint8, device=dev)
        self.index = torch.zeros(self.blocks, 64, 132, dtype=torch.uint8, device=dev)
        self.query = torch.zeros(rows, HPR, 576, dtype=torch.uint8, device=dev)
        self.qc = torch.zeros(rows, 1536, dtype=bf, device=dev)
        self.unit = torch.ones(1, dtype=torch.float32, device=dev)
        self.projected = randn(rows, PITCH)
        # this rank's tokens: 1 / DCP of the rows insert their KV row, others -1
        slots = torch.randperm(self.blocks * 64, generator=gen, device=dev)[:rows]
        keep = (torch.arange(rows, device=dev) % DCP) == rank
        self.slots = torch.where(keep, slots, torch.full_like(slots, -1)).long()
        self.box = DcpBox(group, dev, DCP, rank, HPR, cap=rows, interleave=1,
                          q_heads=Q_HEADS, partials=False)  # fmt: skip
        self.real = slice(0, HPR * world)
        self.box.q[:, :, HPR * world : GATHERED].zero_()  # the emulated ranks' rows
        torch.cuda.synchronize()
        dist.barrier()
        self.qg = None

    def specs(self, nvfp4):
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
            layout,
        )

        kv = self.kv4 if nvfp4 else self.kv
        return tuple(layout(t) for t in (*self.weights, kv, self.index))

    def compile_query(self, batch, nvfp4):
        """wide.compile_core's query kernel (dsa, per-tensor FP8, DCP 16, q_heads 128,
        the box's cap; tile poll above WIDE_MAX_ROWS; its residency check and shared
        carve-out), without its index and value kernels."""
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_stream

        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
            fake,
            prepare_shared_carveout,
        )
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import wide
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.query import (
            MlaQuery,
        )

        specs = self.specs(nvfp4)
        qnorm, kvnorm, qb, uk, _, kv, _ = (
            fake(shape, stride, dtype, 4 if dtype == torch.float32 else 16)
            for shape, stride, dtype in specs
        )
        scalar = fake((1,), (1,), torch.float32, 4)
        query = fake((batch, HPR, 576), (HPR * 576, 576, 1), torch.uint8)
        qc = fake((batch, 1536), (1536, 1))
        projected = fake((batch, PITCH), (PITCH, 1))
        slots = fake((batch,), (1,), torch.int64, 8)
        mla_query = MlaQuery(HPR, batch, q_b_cp_async=False, dsa=True, fp8=True,
                             nvfp4=nvfp4, dcp=DCP, q_heads=Q_HEADS,
                             tile_poll=batch > wide.WIDE_MAX_ROWS)  # fmt: skip
        q_args = (projected, qnorm, kvnorm, qb, uk, kv, slots, scalar, scalar,
                  cutlass.Float32(1e-6))  # fmt: skip
        mailbox = fake((self.box.cap, Q_HEADS, 576), None, torch.uint8)
        publish = (mailbox, cutlass.Int64(0), cutlass.Int32(0))
        device = self.dev.index
        with torch.accelerator.device_index(device):
            kernel = cute.compile(mla_query, *q_args, query, qc, *publish,
                                  make_fake_stream())  # fmt: skip
            if batch > 64:
                wide.check_query_residency(kernel, mla_query, device)
            if batch > 1:
                kernel = prepare_shared_carveout(kernel, device, 100)
        return kernel

    def rearm(self, parity, rows):
        self.box.q[parity, :rows, self.real].fill_(0xFF)

    def arm_query(self, rows, q, nvfp4=False):
        import cutlass

        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
            stream,
            to_cute,
        )
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import wide

        try:  # the query kernel alone, as compile_core builds it (1 compile, not 3)
            mq = self.compile_query(rows, nvfp4)
        except Exception as e:  # noqa: BLE001
            common.log_exc(f"query-only compile M{rows}: falling back to compile_core", e)
            args = (rows, HPR, PITCH, self.specs(nvfp4), self.dev.index, q, DCP,
                    Q_HEADS, self.sms, "proj")  # fmt: skip
            mq, _, _ = wide.compile_core(*args, fp8=True, cap=self.box.cap, nvfp4=nvfp4)
        kv = self.kv4 if nvfp4 else self.kv
        views = tuple(to_cute(w, 4 if w.dtype == torch.float32 else 16)
                      for w in self.weights[:4])  # fmt: skip
        head = (to_cute(self.projected[:rows]), *views, to_cute(kv.view(torch.uint8)),
                to_cute(self.slots[:rows], 8), to_cute(self.unit, 4),
                to_cute(self.unit, 4), cutlass.Float32(EPS))  # fmt: skip
        outs = to_cute(self.query[:rows]), to_cute(self.qc[:rows])
        mailboxes = [to_cute(self.box.q[p].view(torch.uint8)) for p in (0, 1)]
        rank = cutlass.Int32(self.rank)

        def call(parity):
            mq(*head, *outs, mailboxes[parity], self.box.mc(parity), rank,
               stream(self.dev))  # fmt: skip

        def rearm(parity):
            self.rearm(parity, rows)

        return call, rearm

    def arm_stock(self, rows, q):
        from vllm import _custom_ops as ops
        from vllm.v1.attention.ops.dcp import DirectDCPQGatherWorkspace

        dev, bf, fp8 = self.dev, torch.bfloat16, torch.float8_e4m3fn
        heads_g = GATHERED // self.world
        if self.qg is None:
            self.qg = DirectDCPQGatherWorkspace(self.group, dev, self.args.max_rows,
                                                heads_g, 576, dtype=fp8,
                                                padded_num_heads=Q_HEADS)  # fmt: skip
        qg = self.qg
        qnorm, kvnorm, qb_w, uk, _ = self.weights
        x = self.projected[:rows]
        q_c, kv_c, k_pe = x[:, :1536], x[:, 1536:2048], x[:, 2048:2112]
        qn = torch.empty(rows, 1536, dtype=bf, device=dev)
        kvn = torch.empty(rows, 512, dtype=bf, device=dev)
        qabs = torch.empty(rows, HPR, 576, dtype=bf, device=dev)
        qloc = torch.zeros(rows, heads_g, 576, dtype=fp8, device=dev)
        k_scale = torch.ones(1, dtype=torch.float32, device=dev)
        cache = self.kv  # (uint8 rows, as the stock fp8 cache and the tests)
        slots = self.slots[:rows]

        def call(parity):
            ops.rms_norm(qn, q_c, qnorm, EPS)
            ops.rms_norm(kvn, kv_c, kvnorm, EPS)
            qb = torch.mm(qn, qb_w.t()).view(rows, HPR, 192)
            torch.bmm(qb[..., :128].transpose(0, 1), uk,
                      out=qabs[..., :512].transpose(0, 1))  # fmt: skip
            qabs[..., 512:] = qb[..., 128:]
            qloc[:, :HPR].copy_(qabs)
            ops.concat_and_cache_mla(kvn, k_pe, cache, slots, "fp8", k_scale)
            qg.gather(qloc)

        return call, None

    # --- timing (as bench_value) ----------------------------------------------------
    def agree(self, ok):
        flag = torch.tensor([1 if ok else 0], device=self.dev)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def gather(self, us):
        t = torch.tensor([-1.0 if us is None else us], device=self.dev)
        out = [torch.zeros_like(t) for _ in range(self.world)]
        dist.all_gather(out, t)
        return [float(v) for v in out]

    def timed(self, call, restore):
        a, barrier = self.args, dist.barrier
        if restore is None:
            us = common.graph_time(common.Alternating(call), a.inner, a.reps, barrier)
            return us, us, 0.0

        def both(parity):
            call(parity)
            restore(parity)

        raw = common.graph_time(common.Alternating(both), a.inner, a.reps, barrier)
        sub = common.graph_time(common.Alternating(restore), a.inner, a.reps, barrier)
        return raw - sub, raw, sub

    def measure(self, rec, q, rows, name, role, build, policy, note):
        torch.cuda.synchronize()
        dist.barrier()
        # the real ranks' rows armed, the emulated ones published
        self.box.q[:, :, self.real].fill_(0xFF)
        torch.cuda.synchronize()
        dist.barrier()
        built, err = None, None
        try:
            built = build()
        except Exception as e:  # noqa: BLE001
            common.log_exc(f"front build q{q} M{rows} {name}", e)
            common.exit_if_sticky(e, f"front build q{q} M{rows} {name}")
            err = common.error_text(e)
        if not self.agree(built is not None):
            rec.emit("agentx", q, rows, name, None, role=role, status="fail",
                     err=err or "failed on another rank", policy=policy, note=note)  # fmt: skip
            return
        try:
            us, raw, sub = self.timed(*built)
        except Exception as e:  # noqa: BLE001
            common.log_exc(f"front time q{q} M{rows} {name}", e)
            common.exit_if_sticky(e, f"front time q{q} M{rows} {name}")
            us = raw = sub = None
            err = common.error_text(e)
        per = self.gather(us)
        ok = all(v >= 0 for v in per)
        rec.emit("agentx", q, rows, name, max(per) if ok else None, role=role,
                 status="ok" if ok else "fail", err=err, policy=policy,
                 per_rank_us=[round(v, 3) for v in per],
                 raw_us=None if raw is None else round(raw, 3),
                 sub_us=None if sub is None else round(sub, 3), note=note)  # fmt: skip

    def run(self, rec, q, rows):
        if not rec.should_run("agentx", q, rows):
            return
        try:
            self.run_arms(rec, q, rows)
        finally:
            rec.finish("agentx", q, rows)

    def run_arms(self, rec, q, rows):
        self.measure(rec, q, rows, "mlaquery", "spec",
                     lambda: self.arm_query(rows, q), True, "WideDsa MlaQuery (fp8)")  # fmt: skip
        if not self.args.no_nvfp4:
            self.measure(rec, q, rows, "mlaquery_nvfp4", "ref",
                         lambda: self.arm_query(rows, q, nvfp4=True), False,
                         "MlaQuery, nvfp4_ds_mla KV insert")  # fmt: skip
        if not self.args.no_stock:
            self.measure(rec, q, rows, "stock_front", "stock",
                         lambda: self.arm_stock(rows, q), False,
                         f"APPROX: q gather over {self.world} ranks x "
                         f"{GATHERED // self.world} heads")  # fmt: skip
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress", default=None)
    ap.add_argument("--qs", default="4,8")
    ap.add_argument("--rows", default="", help="subset of the front rows")
    ap.add_argument("--max-rows", type=int, default=2048)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--no-nvfp4", action="store_true")
    ap.add_argument("--no-stock", action="store_true")
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    if rank == 0:
        print(f"front bench: {world} real ranks + {DCP - world} emulated on",
              common.device_info(dev), flush=True)  # fmt: skip
    rec = Recorder(args.out, args.progress or args.out + ".progress", "front",
                   rank0=rank == 0)  # fmt: skip
    dist.barrier()
    only = set(common.parse_ints(args.rows)) or set(FRONT_ROWS)
    cases = sorted({(r, q) for q in common.parse_ints(args.qs)
                    for r in common.rows_for(q, lo=65, hi=args.max_rows, only=only)})  # fmt: skip
    args.max_rows = max([r for r, _ in cases], default=128)
    bench = FrontBench(args, dist.group.WORLD, dev, rank, world)
    for rows, q in common.by_priority(cases):
        try:
            bench.run(rec, q, rows)
        except Exception as e:  # noqa: BLE001
            common.log_exc(f"front case q{q} M{rows}", e)
            common.exit_if_sticky(e, f"front case q{q} M{rows}")
    dist.barrier()
    if rank == 0:
        print("front bench done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
