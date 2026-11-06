# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DCP value stage of a K3.1 DSA layer at DCP16 rank shapes on one node: 4 real
ranks (torchrun --nproc-per-node 4, NVLink peer stores) + 12 emulated ones.

Every kernel is compiled for DCP 16 (6 heads per rank, 96 gathered heads, the FP8 q
mailbox of 128 padded rows, the sparse decode's natural-log LSE). The mailboxes
[2, 16, CAP, 6, W] are this script's own symmetric buffers ("EmuMail"): owner o < 4
is real rank o's mailbox; a push to an emulated owner o >= 4 lands over NVLink in a
sink buffer on real rank o % 4 (so 12 of 16 owners are remote, 15 in production),
and the 12 emulated sources of this rank's own mailbox are restored from a snapshot
before every call (the merge re-arms what it reads). Times are net of that restore:
(restore + kernel) - (restore alone), CUDA graphs alternating the two parities,
median of --reps replays, max over the 4 ranks.

Arms per (q, rows), rows 4 .. 2048 (verify rows, AgentX contexts for lengths and
per-row valid counts):

  bf16        "bf16" partials: rows <= 64 PreparedDsa's DcpValue (dsa.dcp_stage,
              the 64-row box), rows > 64 WideDsa's (MR !8 mla.dcp.DcpValue on the
              widened box, GB300 schedule: unstrided up to 560 rows, above it the
              merging CTAs stride the row tiles, G = SMs // 6 tile CTAs per head)
  bf16_g<G>   the same with the stride forced to G tile CTAs per head
              (``resident_sms`` = 6 G; mla.dcp NQ-TUNE "strided vs unstrided at
              401-560 rows"), rows 400 .. 2048
  proj        "proj" partials: the W_UV bmm over the 96 gathered heads, then
              dsa.dcp_value.DcpValue(preprojected) through the "proj" mailbox
  pushed      (part) DcpValue(pushed) after the DSA sparse core (dcp_stage
              compile_core_value: merge + W_UV + gate only; every source
              restored), rows <= 64; pushed_spread: the Rubin spread merge
              (dsa_core.decode.SPREAD)
  stock_a2a   (stock, APPROXIMATE) the S2-DSA output path: DirectDCPA2AWorkspace
              lse_reduce over the 4 real ranks with 24 heads per rank (the 96
              gathered heads; ~0.8x the DCP16 NVLink bytes and 4 sources instead
              of 16), then the W_UV bmm of this rank's 6 heads and the eager gate
              (sigmoid, mul)

"spec" = the code's partials policy at the rows (dsa.dcp_value.DSA_PARTIALS via
partials_for: bf16 at 1-16, proj from 17 on).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import DCP, GATHERED, HPR, PITCH, Q_HEADS, TOPK, Recorder  # noqa: E402

CAP = 64  # PreparedDsa's DSA DcpBox rows
PAIR_SENT_I32 = 0x80000000 - (1 << 32)


def sanitized_bf16(shape, gen, dev, scale=1.0):
    """Random BF16 without negative-zero halves (a pushed pair is never PAIR_SENT)."""
    x = (torch.randn(shape, generator=gen, device=dev) * scale).to(torch.bfloat16)
    halves = x.view(torch.int16)
    halves.masked_fill_(halves == -0x8000, 0x0001)
    return x


class EmuMail:
    """A DcpBox-layout partials mailbox of DCP 16 sources over ``world`` real ranks:
    partial [2, 16, cap, 6, width] BF16, LSE [2, 16, cap, 6] FP32 (sentinel-filled),
    peers of the 16 owners (real ranks' mailboxes, emulated owners' sinks on real
    rank o % world) and the snapshot the emulated sources restore from."""

    def __init__(self, group, dev, world, rank, cap, width, fmt):
        import torch.distributed._symmetric_memory as symm_mem

        from vllm.models.kimi_k3.nvidia.specialized_decode.symm import peer_rendezvous

        assert DCP % world == 0, (DCP, world)
        self.format, self.cap, self.size, self.q_heads = fmt, cap, DCP, Q_HEADS
        self.world, self.rank = world, rank
        shape = (2, DCP, cap, HPR)

        def pair():
            part = symm_mem.empty((*shape, width), dtype=torch.bfloat16, device=dev)
            lse = symm_mem.empty(shape, dtype=torch.float32, device=dev)
            handles = [peer_rendezvous(t, group) for t in (part, lse)]
            return part, lse, handles

        self.partial, self.lse, (hp, hl) = pair()
        sinks = [pair() for _ in range(DCP // world - 1)]
        ptrs, lptrs = [], []
        for o in range(DCP):
            if o < world:
                ptrs.append(hp.buffer_ptrs[o])
                lptrs.append(hl.buffer_ptrs[o])
            else:
                _, _, (sp, sl) = sinks[o // world - 1]
                ptrs.append(sp.buffer_ptrs[o % world])
                lptrs.append(sl.buffer_ptrs[o % world])
        self.sinks = sinks  # (kept alive)
        i64 = dict(dtype=torch.int64, device=dev)
        self.peers, self.lse_peers = torch.tensor(ptrs, **i64), torch.tensor(lptrs, **i64)
        gen = torch.Generator(device=dev).manual_seed(77 + rank)
        self.snap = sanitized_bf16((*shape, width), gen, dev, 0.5)
        self.snap_lse = 4 * torch.rand(shape, generator=gen, device=dev) - 2
        self.arm()
        torch.cuda.synchronize()
        dist.barrier()

    def arm(self):
        self.partial.view(torch.int32).fill_(PAIR_SENT_I32)
        self.lse.view(torch.int32).fill_(-1)

    def restore(self, parity, rows, lo):
        """Sources lo.. of this rank's mailbox, rows [0, rows), from the snapshot."""
        self.partial[parity, lo:, :rows].copy_(self.snap[parity, lo:, :rows])
        if lo == 0:  # (pushed: every source; the spread LSE box layout differs)
            self.lse[parity].copy_(self.snap_lse[parity])
        else:
            self.lse[parity, lo:, :rows].copy_(self.snap_lse[parity, lo:, :rows])


class Shim:
    """What dcp_stage.run_value / run_core_value read of a DsaBox."""

    def __init__(self, mail):
        self.mail, self.size, self.q_heads, self.cap = mail, DCP, Q_HEADS, mail.cap

    def mailbox(self, batch):
        assert batch <= self.cap
        return self.mail


class ValueBench:
    def __init__(self, args, group, dev, rank, world):
        self.args, self.group, self.dev = args, group, dev
        self.rank, self.world = rank, world
        self.sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.vecs = common.ctx_vectors()
        gen = torch.Generator(device=dev).manual_seed(5 + rank)
        w_uv = torch.randn(HPR, 128, 512, generator=gen, device=dev) / 23
        self.w_uv = w_uv.bfloat16().transpose(1, 2)  # [6, 512, 128] latent-contiguous
        g = torch.randn(GATHERED, 128, 512, generator=gen, device=dev) / 23
        self.gathered = g.bfloat16().transpose(1, 2)  # gather_heads' layout
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
            layout,
        )

        self.spec = layout(self.w_uv)
        self.wide_cap = args.wide_cap
        self.mails: dict = {}
        self.qboxes: dict = {}
        self.a2a = None

    def mail(self, cap, fmt):
        key = (cap, fmt)
        if key not in self.mails:  # (collective: every rank in the same order)
            width = 128 if fmt == "proj" else 512
            args = self.group, self.dev, self.world, self.rank, cap, width, fmt
            self.mails[key] = EmuMail(*args)
        return self.mails[key]

    def qbox(self, cap):
        if cap not in self.qboxes:
            self.qboxes[cap] = torch.zeros(2, cap, Q_HEADS, 576, dtype=torch.uint8,
                                           device=self.dev)  # fmt: skip
        return self.qboxes[cap]

    # --- inputs ---------------------------------------------------------------------
    def inputs(self, rows, q):
        dev, rank = self.dev, self.rank
        ctxs = common.ctx_for(self.vecs, rows // q, q)
        g_rows = common.row_contexts(ctxs, q, dev)
        local = common.local_len(g_rows, rank)
        valid = torch.minimum(local, (g_rows.clamp(max=TOPK) // DCP).clamp(min=1))
        gen = torch.Generator(device=dev).manual_seed(rows + q + rank)
        latent = torch.randn(rows, Q_HEADS, 512, generator=gen, device=dev).bfloat16()
        lse = 4 * torch.rand(rows, Q_HEADS, generator=gen, device=dev)
        lse = lse.masked_fill((valid == 0).view(-1, 1), float("-inf"))
        projected = torch.randn(rows, PITCH, generator=gen, device=dev).bfloat16()
        out = torch.empty(rows, HPR * 128, device=dev, dtype=torch.bfloat16)
        lengths = torch.tensor(ctxs, dtype=torch.int32, device=dev)
        core = latent, lse, valid.int().contiguous()
        return dict(core=core, lengths=lengths, projected=projected, out=out,
                    mean_ctx=float(sum(ctxs)) / len(ctxs))  # fmt: skip

    # --- arms: each returns (call(parity), restore(parity)) ---------------------------
    def arm_prepared(self, rows, q, x, fmt):
        """dcp_stage's DcpValue (PreparedDsa at <= 64 rows; "proj" at any rows)."""
        import cutlass

        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
            to_cute,
        )
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.dcp_stage import (
            compile_dcp_value,
            run_value,
        )

        cap = CAP if rows <= CAP else self.wide_cap
        mail, qbox = self.mail(cap, fmt), self.qbox(cap)
        args = HPR, DCP, rows, self.spec, PITCH, self.dev.index, self.sms, q, Q_HEADS
        kernel = compile_dcp_value(*args, True, True, fmt, cap)
        if fmt == "proj" and rows > CAP:
            # as production compile_core: the merging CTAs poll the rows every rank's
            # projecting CTAs push, so the whole grid must be resident (else a hang)
            from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import wide

            wide.check_proj_residency(kernel, HPR, DCP, rows, self.sms, self.dev.index)
        shim, uv = Shim(mail), to_cute(self.w_uv)
        w_uv = self.gathered if fmt == "proj" else None
        rank = cutlass.Int32(self.rank)
        mailboxes = [to_cute(qbox[p].view(torch.uint8)) for p in (0, 1)]

        def call(parity):
            publish = mailboxes[parity], None, rank
            run_value(kernel, shim, parity, publish, HPR, x["core"], x["lengths"], uv,
                      x["projected"], x["out"], w_uv)  # fmt: skip

        def restore(parity):
            mail.restore(parity, rows, self.world)

        return call, restore

    def wide_value(self, rows, q, resident=0):
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import (
            GB300_SMS,
            DcpValue,
        )

        sms = min(self.sms, GB300_SMS)
        value = DcpValue(HPR, DCP, rows, sms, q, q_chunks=36, q_heads=Q_HEADS,
                         lse_scale=math.log2(math.e),
                         resident_sms=resident or self.sms)  # fmt: skip
        return value

    def arm_wide_bf16(self, rows, q, x, resident=0):
        """WideDsa's "bf16" DcpValue (mla.dcp, GB300 schedule) on the widened box."""
        import cutlass

        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
            stream,
            to_cute,
        )
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import (
            compile_value,
        )

        cap = self.wide_cap
        mail, qbox = self.mail(cap, "bf16"), self.qbox(cap)
        kw = dict(dsa_heads=Q_HEADS, dsa_bf16=False, lse_base_e=True)
        if resident:
            kw["resident_sms"] = resident
        kernel = compile_value(HPR, DCP, rows, cap, self.spec, PITCH, self.dev.index,
                               self.sms, q, **kw)  # fmt: skip
        latent, lse, _ = x["core"]
        rows_view = latent.as_strided((rows, 1, GATHERED, 512), (Q_HEADS * 512, 0, 512, 1))
        lse_view = lse.as_strided((rows, 1, GATHERED), (Q_HEADS, 0, 1))
        lens = to_cute(x["lengths"], 4)
        head = (to_cute(rows_view), to_cute(lse_view, 4), lens)
        boxes = [(to_cute(mail.partial[p]), to_cute(mail.lse[p], 4)) for p in (0, 1)]
        peers = to_cute(mail.peers, 8), to_cute(mail.lse_peers, 8)
        mailboxes = [to_cute(qbox[p].view(torch.uint8)) for p in (0, 1)]
        tail = (to_cute(self.w_uv), to_cute(x["projected"][:, 2112:]), lens,
                to_cute(x["out"]))  # fmt: skip
        rank = cutlass.Int32(self.rank)

        def call(parity):
            args = (*head, *boxes[parity], *peers, mailboxes[parity], *tail)
            kernel(*args, cutlass.Int32(parity), rank, stream(self.dev))

        def restore(parity):
            mail.restore(parity, rows, self.world)

        return call, restore

    def arm_pushed(self, rows, q, x, spread):
        """DcpValue(pushed) after the DSA sparse core (every source restored)."""
        import cutlass

        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
            to_cute,
        )
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.dcp_stage import (
            compile_core_value,
            run_core_value,
        )

        mail, qbox = self.mail(CAP, "bf16"), self.qbox(CAP)
        args = HPR, DCP, rows, self.spec, PITCH, self.dev.index, self.sms, q, Q_HEADS
        kernel = compile_core_value(*args, cap=CAP, spread=spread)
        shim, uv = Shim(mail), to_cute(self.w_uv)
        latent, lse, _ = x["core"]
        buffers = latent.view(-1), lse.view(-1)
        rank = cutlass.Int32(self.rank)
        mailboxes = [to_cute(qbox[p].view(torch.uint8)) for p in (0, 1)]

        def call(parity):
            publish = mailboxes[parity], None, rank
            run_core_value(kernel, shim, parity, publish, HPR, buffers, x["lengths"],
                           uv, x["projected"], x["out"])  # fmt: skip

        def restore(parity):
            mail.restore(parity, rows, 0)

        return call, restore

    def arm_stock(self, rows, q, x):
        """S2-DSA's output path, approximated on 4 ranks (see the module docstring)."""
        from vllm.v1.attention.ops.dcp import DirectDCPA2AWorkspace

        if self.a2a is None:
            heads = GATHERED // self.world
            self.a2a = DirectDCPA2AWorkspace(self.group, self.dev, self.args.max_rows,
                                             heads, 512)  # fmt: skip
        a2a = self.a2a
        latent, lse, _ = x["core"]
        o_c = latent[:, :GATHERED].contiguous()
        l_c = lse[:, :GATHERED].contiguous()
        values = torch.empty(rows, HPR, 128, dtype=torch.bfloat16, device=self.dev)
        gate = x["projected"][:, 2112 : 2112 + HPR * 128]
        out = x["out"]

        def call(parity):
            merged = a2a.lse_reduce(o_c, l_c, True)
            torch.bmm(merged[:, :HPR].transpose(0, 1), self.w_uv,
                      out=values.transpose(0, 1))  # fmt: skip
            torch.mul(values.view(rows, HPR * 128), gate.sigmoid(), out=out)

        def a2a_only(parity):
            a2a.lse_reduce(o_c, l_c, True)

        return call, None, a2a_only

    # --- timing -------------------------------------------------------------------
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
        a = self.args
        barrier = dist.barrier
        if restore is None:
            us = common.graph_time(common.Alternating(call), a.inner, a.reps, barrier)
            return us, us, 0.0

        def both(parity):
            restore(parity)
            call(parity)

        raw = common.graph_time(common.Alternating(both), a.inner, a.reps, barrier)
        sub = common.graph_time(common.Alternating(restore), a.inner, a.reps, barrier)
        return raw - sub, raw, sub

    def measure(self, rec, case, q, rows, name, role, build, policy, note=None,
                extra=None):  # fmt: skip
        built, err = None, None
        # Every mailbox re-armed between arms (no stale word lets a merge pass
        # early); the previous arm's last collective ordered every rank's pushes.
        torch.cuda.synchronize()
        dist.barrier()
        for mail in self.mails.values():
            mail.arm()
        for box in self.qboxes.values():
            box.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        try:
            built = build()
        except Exception as e:  # noqa: BLE001
            common.log_exc(f"value build q{q} M{rows} {name}", e)
            common.exit_if_sticky(e, f"value build q{q} M{rows} {name}")
            err = common.error_text(e)
        if not self.agree(built is not None):
            rec.emit(case, q, rows, name, None, role=role, status="fail",
                     err=err or "failed on another rank", policy=policy, note=note)  # fmt: skip
            return None
        call, restore = built[0], built[1]
        try:
            us, raw, sub = self.timed(call, restore)
            parts = {}
            if len(built) > 2:  # the stock collective alone
                parts["a2a_only_us"] = round(self.timed(built[2], None)[0], 3)
        except Exception as e:  # noqa: BLE001 (a rank that fails here may hang the others)
            common.log_exc(f"value time q{q} M{rows} {name}", e)
            common.exit_if_sticky(e, f"value time q{q} M{rows} {name}")
            us, raw, sub, parts = None, None, None, {}
            err = common.error_text(e)
        per = self.gather(us)
        ok = all(v >= 0 for v in per)
        rec.emit(case, q, rows, name, max(per) if ok else None, role=role,
                 status="ok" if ok else "fail", err=err, policy=policy,
                 per_rank_us=[round(v, 3) for v in per],
                 raw_us=None if raw is None else round(raw, 3),
                 sub_us=None if sub is None else round(sub, 3), note=note,
                 **parts, **(extra or {}))  # fmt: skip
        return max(per) if ok else None

    def run(self, rec, q, rows):
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.dcp_value import (
            DSA_PARTIALS,
            partials_for,
        )

        case = "agentx"
        if not rec.should_run(case, q, rows):
            return
        try:
            self.run_arms(rec, case, q, rows, partials_for(DSA_PARTIALS, rows))
        finally:
            rec.finish(case, q, rows)

    def run_arms(self, rec, case, q, rows, policy):
        x = self.inputs(rows, q)
        extra = dict(mean_ctx=round(x["mean_ctx"]))
        bf16_note = "PreparedDsa dcp_stage bf16" if rows <= CAP else "WideDsa mla.dcp bf16"
        if rows > CAP:
            value = self.wide_value(rows, q)
            bf16_note += f" grid {list(value.grid)} tile_stride {value.tile_stride}"
        build = (lambda: self.arm_prepared(rows, q, x, "bf16")) if rows <= CAP else (
            lambda: self.arm_wide_bf16(rows, q, x))
        self.measure(rec, case, q, rows, "bf16", "spec" if policy == "bf16" else "tactic",
                     build, policy == "bf16", bf16_note, extra)  # fmt: skip
        self.measure(rec, case, q, rows, "proj", "spec" if policy == "proj" else "tactic",
                     lambda: self.arm_prepared(rows, q, x, "proj"), policy == "proj",
                     "dcp_stage proj (bmm + DcpValue preprojected)", extra)  # fmt: skip
        if rows > CAP and rows >= self.args.stride_from:
            tiles = -(-rows // 16)
            default = self.wide_value(rows, q)
            g_default = default.tile_stride or default.grid[1]
            forced = {-(-tiles // 2), -(-tiles * 3 // 4)} if g_default == tiles else {18}
            for g in sorted(forced):
                if g >= g_default or g < 4:
                    continue
                v = self.wide_value(rows, q, resident=HPR * g)
                if v.tile_stride != g:
                    continue
                self.measure(rec, case, q, rows, f"bf16_g{g}", "tactic",
                             lambda g=g: self.arm_wide_bf16(rows, q, x, HPR * g), False,
                             f"forced stride: grid {list(v.grid)}", extra)  # fmt: skip
        if rows <= CAP and not self.args.no_pushed:
            from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core import decode

            spread_on = bool(decode.rows_table(decode.SPREAD.replace(";", ",")).get(rows))
            for spread in sorted({spread_on} | ({False, True} if rows in (8, 16) else set())):
                name = "pushed_spread" if spread else "pushed"
                self.measure(rec, case, q, rows, name, "part",
                             lambda s=spread: self.arm_pushed(rows, q, x, s), False,
                             "after the DSA sparse core (merge only)", extra)  # fmt: skip
        if not self.args.no_stock:
            self.measure(rec, case, q, rows, "stock_a2a", "stock",
                         lambda: self.arm_stock(rows, q, x), False,
                         f"APPROX: A2A over {self.world} ranks x "
                         f"{GATHERED // self.world} heads + bmm(6) + gate", extra)  # fmt: skip
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress", default=None)
    ap.add_argument("--qs", default="4,8")
    ap.add_argument("--rows", default="", help="subset of the in-scope rows")
    ap.add_argument("--max-rows", type=int, default=2048)
    ap.add_argument("--stride-from", type=int, default=400)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--no-pushed", action="store_true")
    ap.add_argument("--no-stock", action="store_true")
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    if rank == 0:
        print(f"value bench: {world} real ranks + {DCP - world} emulated on",
              common.device_info(dev), flush=True)  # fmt: skip
    rec = Recorder(args.out, args.progress or args.out + ".progress", "value",
                   rank0=rank == 0)  # fmt: skip
    dist.barrier()
    only = set(common.parse_ints(args.rows))
    qs = common.parse_ints(args.qs)
    cases = sorted({(r, q) for q in qs for r in common.rows_for(q, hi=args.max_rows,
                                                               only=only)})  # fmt: skip
    args.wide_cap = max([r for r, _ in cases if r > CAP], default=CAP + 1)
    bench = ValueBench(args, dist.group.WORLD, dev, rank, world)
    for rows, q in common.by_priority(cases):
        try:
            bench.run(rec, q, rows)
        except Exception as e:  # noqa: BLE001
            common.log_exc(f"value case q{q} M{rows}", e)
            common.exit_if_sticky(e, f"value case q{q} M{rows}")
    dist.barrier()
    if rank == 0:
        print("value bench done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
