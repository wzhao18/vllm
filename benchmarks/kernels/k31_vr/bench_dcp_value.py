# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DCP value stage of a Kimi K3.1 DSA layer, as served ("prod") and with #108's
"proj" partials (r1.auto_partials), for bench_dcp_exchange.py --paths value.

Per (B requests, q) bucket of T = B * q rows, one DCP16 rank's 6 heads of 96
gathered (FlashInfer's 128-padded rows and natural-log LSEs, random; per-row valid
counts from the bucket's AgentX contexts), every arm in CUDA graphs whose calls
alternate the two mailbox parities as consecutive DSA layers do:

  prod       T <= 64: dcp_stage.run_value "bf16" on the 64-row DSA DcpBox
             (PreparedDsa); T > 64: MR !8's DcpValue on the widened box (WideDsa)
  proj       run_value "proj": the W_UV bmm over the gathered heads (gather_heads'
             layout), then DcpValue(preprojected) through the box's DcpMailbox
  proj_contig  the same with a contiguous gathered W_UV
  prod_again   prod once more: the A-vs-A' spread

Extras: the bmm alone in both layouts, the "proj" grid, and the relative
difference of the proj rows to the prod rows (the two roundings of one merge).
"""

import os

import cutlass
import torch
import torch.distributed as dist

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    layout,
    stream,
    to_cute,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import dcp_value
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.dcp_stage import (
    compile_dcp_value,
    publish_args,
    run_value,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.r1 import DsaBox
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla import dcp as mla_dcp

Q_HEADS = 128  # FlashInfer's padded gathered heads (trtllm_gen_query_heads(96))
CAP = 64  # the DSA DcpBox's rows (providers.dcp_box)


class ValueBench:
    """The boxes, gathered W_UV and kernels of the value arms on this rank (the
    collectives run in the same order on every rank)."""

    def __init__(self, group, dev, world, rank, heads, interleave, wide_cap):
        self.dev, self.world, self.rank, self.heads = dev, world, rank, heads
        self.sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.pitch = 2112 + heads * 128 + 160
        box = mla_dcp.DcpBox(
            group, dev, world, rank, heads, interleave=interleave, q_heads=Q_HEADS
        )
        self.boxes = {CAP: box}
        if wide_cap > CAP:
            self.boxes[wide_cap] = box.widened(wide_cap)
        self.prod = {cap: DsaBox(b, partials="bf16") for cap, b in self.boxes.items()}
        self.proj = {cap: DsaBox(b, partials="proj") for cap, b in self.boxes.items()}
        gen = torch.Generator(device=dev).manual_seed(5 + rank)
        w_uv = torch.randn(heads, 128, 512, device=dev, generator=gen) / 23
        self.w_uv = w_uv.bfloat16().transpose(1, 2)  # latent-contiguous
        self.gathered = dcp_value.gather_heads(self.w_uv, world, group)
        self.contig = self.gathered.contiguous()
        self.spec = layout(self.w_uv)

    def cap(self, rows):
        return CAP if rows <= CAP else max(self.boxes)

    def inputs(self, rows, q, lengths, valid):
        gen = torch.Generator(device=self.dev).manual_seed(rows + q + self.rank)
        latent = torch.randn(rows, Q_HEADS, 512, generator=gen, device=self.dev)
        lse = 4 * torch.rand(rows, Q_HEADS, generator=gen, device=self.dev)
        # A row with no local key: LSE -inf (WideDsa.dcp_core's mask)
        lse = lse.masked_fill((valid == 0).view(-1, 1), float("-inf"))
        projected = torch.randn(rows, self.pitch, generator=gen, device=self.dev)
        out = torch.empty(rows, self.heads * 128, device=self.dev).bfloat16()
        core = latent.bfloat16(), lse, valid.int()
        return core, lengths.int(), projected.bfloat16(), out

    def prod_fn(self, rows, q, core, lengths, projected, out):
        """The served value stage: PreparedDsa's up to 64 rows, WideDsa's above."""
        cap, box = self.cap(rows), self.boxes[self.cap(rows)]
        args = self.heads, self.world, rows, self.spec, self.pitch, self.dev.index
        uv = to_cute(self.w_uv)
        if rows <= CAP:
            args += (self.sms, q, Q_HEADS, True, True, "bf16", cap)
            kernel, dsa = compile_dcp_value(*args), self.prod[cap]

            def call(parity):
                publish = publish_args(box, parity)
                run_value(kernel, dsa, parity, publish, self.heads, core, lengths, uv,
                          projected, out)  # fmt: skip

            return call
        kernel = mla_dcp.compile_value(
            self.heads, self.world, rows, cap, self.spec, self.pitch, self.dev.index,
            self.sms, q, dsa_heads=Q_HEADS, dsa_bf16=False, lse_base_e=True,
        )  # fmt: skip
        latent, lse, _ = core
        gathered = self.heads * self.world
        rows_view = latent.as_strided(
            (rows, 1, gathered, 512), (Q_HEADS * 512, 0, 512, 1)
        )
        lse_view = lse.as_strided((rows, 1, gathered), (Q_HEADS, 0, 1))
        lens = to_cute(lengths, 4)

        def call(parity):
            mailbox = to_cute(box.q[parity].view(torch.uint8))
            boxes = to_cute(box.partial[parity]), to_cute(box.lse[parity], 4)
            peers = to_cute(box.peers, 8), to_cute(box.lse_peers, 8)
            v_args = (to_cute(rows_view), to_cute(lse_view, 4), lens, *boxes, *peers)
            v_args += (mailbox, uv, to_cute(projected[:, 2112:]), lens, to_cute(out))
            ids = cutlass.Int32(parity), cutlass.Int32(self.rank)
            kernel(*v_args, *ids, stream(self.dev))

        return call

    def proj_fn(self, rows, q, core, lengths, projected, out, w_uv):
        """The W_UV bmm, then DcpValue(preprojected) through the "proj" DcpMailbox."""
        cap = self.cap(rows)
        args = self.heads, self.world, rows, self.spec, self.pitch, self.dev.index
        args += (self.sms, q, Q_HEADS, True, True, "proj", cap)
        kernel, dsa, box = compile_dcp_value(*args), self.proj[cap], self.boxes[cap]
        uv = to_cute(w_uv)

        def call(parity):
            publish = publish_args(box, parity)
            run_value(kernel, dsa, parity, publish, self.heads, core, lengths, uv,
                      projected, out, w_uv)  # fmt: skip

        return call

    def bmm_fn(self, rows, core, w_uv):
        gathered = self.heads * self.world
        latent = core[0][:, :gathered]
        values = torch.empty(gathered, rows, 128, device=self.dev).bfloat16()
        return lambda: torch.bmm(latent.transpose(0, 1), w_uv, out=values)

    def grid(self, rows):
        return dcp_value.DcpValue(
            self.heads, self.world, rows, self.sms, partials=dcp_value.PROJECTED
        ).grid


def alternating(call):
    """A no-argument function whose calls alternate the parity (consecutive layers)."""
    state = [0]

    def fn():
        call(state[0])
        state[0] ^= 1

    return fn


def run_value_paths(bench, rows, q, lengths, valid, graph_time, emit, shape, args):
    """Times the value arms of one bucket (graph_time / emit of bench_dcp_exchange)."""
    core, lengths, projected, out = bench.inputs(rows, q, lengths, valid)
    prod = alternating(bench.prod_fn(rows, q, core, lengths, projected, out))
    # One eager call per arm: the proj rows against the prod rows.
    prod()
    torch.accelerator.synchronize()
    reference = out.float().clone()
    arms = {"value_prod": prod}
    for name, w_uv in (
        ("value_proj", bench.gathered),
        ("value_proj_contig", bench.contig),
    ):
        arms[name] = alternating(
            bench.proj_fn(rows, q, core, lengths, projected, out, w_uv)
        )
    arms["value_proj"]()
    torch.accelerator.synchronize()
    diff = (out.float() - reference).norm() / reference.norm().clamp(min=1e-30)
    extras = {
        "value_proj": dict(
            rel_diff_vs_prod=float(diff),
            grid=list(bench.grid(rows)),
            bmm_us=graph_time(bench.bmm_fn(rows, core, bench.gathered), *args),
        ),
        "value_proj_contig": dict(
            bmm_us=graph_time(bench.bmm_fn(rows, core, bench.contig), *args)
        ),
    }
    for name, fn in (*arms.items(), ("value_prod_again", prod)):
        emit(shape, name, graph_time(fn, *args), extras.get(name))
    dist.barrier()


def wide_cap(rows):
    """The widened box of the widest bucket (VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS-like)."""
    return max((r for r in rows if r > CAP), default=CAP)


if __name__ == "__main__":
    raise SystemExit(
        "run bench_dcp_exchange.py --paths value (" + os.path.basename(__file__) + ")"
    )
