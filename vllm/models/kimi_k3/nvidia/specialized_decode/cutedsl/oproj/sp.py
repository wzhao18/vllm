# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 sequence-parallel o-projection fused with its reduce-scatter (C512 SP decode,
``runtime.sp_attention``): out_shard [n, 7168] = rows [rank n, (rank + 1) n) of
sum over the TP ranks of bf16(attn_out @ W^T), attn_out [R, 3072] (24 heads x 128),
W this rank's row-parallel K shard [7168, 3072], n = R / TP.

Each rank's partial is FP32-accumulated and rounded once to BF16 (as stock's
o-projection), then pushed 16 bytes at a time into its row owner's receive slot of
``common.sp_comm.SpComm``; the owner sums the TP slots in FP32 in rank order and
rounds once (``SpReduceScatter``: stock ``mnnvl_lamport_reduce_scatter``
precision). Tactics per token rows (``SP_OPROJ``, ``shapes.rows_policy``):

- "warp": ``SpOprojWarp``, the push fused into the GEMM: 56 clusters of S = 8 CTAs.
  Each CTA keeps 128 weight rows x 3072 / S K (its K slice) in registers (loaded
  before the PDL wait), copies x's K slice to shared memory, runs m16n8k16 BF16
  MMAs with FP32 accumulation, and leaves its FP32 [tokens, 128] partial in shared
  memory; after a cluster barrier every CTA sums 1 / S of the tile over the S CTAs
  in split order (DSMEM, FP32), rounds once and pushes the chunks to their row
  owners. Then ``SpReduceScatter(push=False)``. x and partial tiles fit shared
  memory up to WARP_ROWS rows.
- "cublas": stock's o-projection GEMM (``torch.mm``: the same cuBLAS call as the
  stock ``F.linear``, so the same partial bits) into ``SpComm.partial``, then
  ``SpReduceScatter(push=True)`` (pushes, owner reduce). Given stock's partials the
  shard is stock's bit for bit. A declared tactic of this provider, not a stock
  fallback: the communication is this provider's.

Both re-arm the all-gather rows of their parity (``SpReduceScatter``).

NQ-TUNE: ``SP_OPROJ`` is GENERIC (no VR200 measurement yet): "warp" where its
weight streaming should be bandwidth-bound (R <= 64), "cublas" above, where
mma.sync falls behind tcgen05. To measure per R (4 GPUs, 24 heads): the pair vs
stock o-projection + ``mnnvl_lamport_reduce_scatter``; S (4 / 8), features per CTA;
a tcgen05 streaming producer for R > 64 (``OprojUmma``-style ring, tokens on N,
pushes to row owners) to replace "cublas"; polling the owner slots in the GEMM's
dependents before their griddepcontrol.wait."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Constexpr, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ...shapes import Choice, Context, Path, Shape, Unsupported, rows_policy
from ..common.host import fake, stream, to_cute
from ..common.primitives import (
    add_rn,
    alloc,
    asm,
    at,
    launch,
    load_vector,
    mma_into,
    pack_pair,
    shared_address,
    smem_words,
    store_relaxed,
    store_shared_words,
    thread,
)
from ..common.sp_comm import SpComm, positive_zero32

WIDTH = 7168  # output features (the hidden size)
SPLITS = 8  # CTAs per cluster: K slices
FEATURES = 128  # output features per CTA (8 warps x 16)
WARP_ROWS = 128  # rows whose x / partial tiles fit one CTA's shared memory
# Tactic per TP on token rows (GENERIC, module docstring: NQ-TUNE).
SP_OPROJ = {4: "4-64:warp,65-:cublas"}


@asm((Float32,) * 4)
def load_cluster_f32x4(address, rank):
    """Four FP32 at this CTA's shared ``address`` in cluster CTA ``rank``'s window."""
    code = "{ .reg .b32 a; mapa.shared::cluster.u32 a,$4,$5; "
    code += "ld.shared::cluster.v4.f32 {$0,$1,$2,$3},[a]; }"
    return code, "=f,=f,=f,=f,r,r,~{memory}", (Uint32(address), Int32(rank))


class SpOprojWarp:
    """The fused "warp" o-projection producer (module docstring). Grid 56 x S CTAs in
    clusters of S along x (CTA = feature block x S + split), 256 threads (warp w:
    features 16 w .. 16 w + 15 of the block). Args: x [R, k], the weight [7168, k],
    the local receive view [TP, cap / TP, 7168] (layout only), the owners' receive
    bases [TP], rank."""

    def __init__(self, k, tp, splits=SPLITS):
        self.k, self.tp, self.splits = k, tp, splits
        self.slice = k // splits  # K per CTA
        self.kblocks = self.slice // 32  # 32 K (4 lanes x 8) per register block
        self.x_stride = self.slice // 8 + 4  # 16-byte chunks per x row (+4: banks)
        self.red_stride = FEATURES + 4  # FP32 per partial row (+4: banks)
        assert k % (32 * splits) == 0 and WIDTH % FEATURES == 0

    def smem_bytes(self, rows):
        tokens = -(-rows // 8) * 8
        return tokens * (self.x_stride * 16 + self.red_stride * 4) + 1024

    @cute.jit
    def __call__(self, *args):
        grid = WIDTH // FEATURES * self.splits
        launch(self.kernel, args, grid, 256, self.splits)

    @cute.kernel
    def kernel(self, args):
        x, weight, recv, peers, rank = args
        M: Constexpr = x.shape[0]
        NT: Constexpr = (M + 7) // 8
        S: Constexpr = self.splits
        tid, warp, lane, cta = thread()
        group, part = lane // 4, lane % 4
        split = cute.arch.block_idx_in_cluster()
        block = cta // S
        smem = cutlass.utils.SmemAllocator()
        xs = alloc(smem, Uint32, NT * 8 * self.x_stride * 4)
        red = alloc(smem, Float32, (NT * 8, self.red_stride))
        # Weights do not depend on the previous kernel: rows (feature) f and f + 8,
        # K = split slice + 32 j + 8 part .. + 8 (x below uses the same K order).
        feature = block * FEATURES + warp * 16 + group
        k0 = split * self.slice + part * 8
        wr = cute.make_rmem_tensor((self.kblocks, 2, 4), Uint32)
        for j in cutlass.range_constexpr(self.kblocks):
            for r in cutlass.range_constexpr(2):
                src = at(weight, feature + 8 * r, k0 + j * 32)
                words = load_vector(src, cache="weight")
                for v in cutlass.range_constexpr(4):
                    wr[j, r, v] = words[v]
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        per: Constexpr = self.slice // 8
        for c in range(tid, M * per, 256):
            token, chunk = c // per, c % per
            words = load_vector(at(x, token, split * self.slice + chunk * 8))
            dst = xs.iterator + (token * self.x_stride + chunk) * 4
            store_shared_words(dst, words)
        cute.arch.sync_threads()
        acc = cute.make_rmem_tensor((NT, 4), Float32)
        acc.fill(0)
        for j in cutlass.range_constexpr(self.kblocks):
            for nt in cutlass.range_constexpr(NT):
                xv = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                if nt * 8 + group < M:
                    row = nt * 8 + group
                    ptr = xs.iterator + (row * self.x_stride + j * 4 + part) * 4
                    xv = smem_words(ptr)
                for half in cutlass.range_constexpr(2):
                    a = (wr[j, 0, 2 * half], wr[j, 1, 2 * half])
                    a += (wr[j, 0, 2 * half + 1], wr[j, 1, 2 * half + 1])
                    b = (xv[2 * half], xv[2 * half + 1])
                    mma_into(acc[nt, None], a, b, bf16=True)
        # C fragment v: feature group + 8 (v // 2), token 2 part + v % 2 of tile nt.
        for nt in cutlass.range_constexpr(NT):
            for v in cutlass.range_constexpr(4):
                token = nt * 8 + 2 * part + v % 2
                red[token, warp * 16 + group + (v // 2) * 8] = acc[nt, v]
        cute.arch.cluster_arrive()  # (release: the partial tiles are visible)
        cute.arch.cluster_wait()
        # Items (token, 8-feature chunk) of the block, 1 / S per CTA: the S slices
        # summed in split order, rounded once, pushed to the token's row owner.
        n: Constexpr = M // self.tp
        items: Constexpr = M * (FEATURES // 8)
        for i in cutlass.range_constexpr(-(-items // (S * 256))):
            item = (i * S + split) * 256 + tid
            if item < items:
                token, chunk = item // (FEATURES // 8), item % (FEATURES // 8)
                values = cute.make_rmem_tensor(8, Float32)
                for h in cutlass.range_constexpr(2):
                    address = shared_address(at(red, token, chunk * 8 + h * 4))
                    sums = load_cluster_f32x4(address, 0)
                    for src in cutlass.range_constexpr(1, S):
                        more = load_cluster_f32x4(address, src)
                        sums = tuple(add_rn(s, m) for s, m in zip(sums, more))
                    for v in cutlass.range_constexpr(4):
                        values[h * 4 + v] = sums[v]
                words = tuple(
                    positive_zero32(
                        pack_pair(values[2 * j], values[2 * j + 1], policy=None)
                    )
                    for j in range(4)
                )
                owner = token // n
                column = block * FEATURES + chunk * 8
                offset = cute.crd2idx((rank, token - owner * n, column), recv.layout)
                store_relaxed(Int64(peers[owner]) + Int64(offset) * 2, words)
        cute.arch.cluster_arrive()  # no CTA leaves while peers read its tile
        cute.arch.cluster_wait()


@cache
def compile_warp(rows, k, tp, cap, device):
    kernel = SpOprojWarp(k, tp)
    with torch.accelerator.device_index(device):
        x, w = fake((rows, k), (k, 1)), fake((WIDTH, k), (k, 1))
        recv, peers = fake((tp, cap // tp, WIDTH)), fake((tp,), None, torch.int64)
        args = (x, w, recv, peers, Int32(0))
        return cute.compile(kernel, *args, make_fake_stream())


def sp_tactic(tp: int, shape: Shape) -> str:
    """The o-projection tactic at ``shape`` (``SP_OPROJ``); Unsupported if none."""
    try:
        return rows_policy(SP_OPROJ.get(tp, ""), shape, name="SP_OPROJ")
    except KeyError as e:
        raise Unsupported("oproj_rs", shape, f"no SP_OPROJ entry at TP{tp}") from e


class SpOproj:
    """One SP decoder layer's o-projection + reduce-scatter (module docstring).
    ``run(attn_out [R, k], out_shard [R / TP, 7168], parity)``; also re-arms
    ``comm.gather[parity][:R]``. Every R of ``rows`` is compiled (and its polling
    kernels residency-checked) at build; nothing falls back at run."""

    def __init__(self, weight: torch.Tensor, comm: SpComm, rows, ctx: Context):
        self.comm, self.ctx = comm, ctx
        k = weight.shape[1] if weight.ndim == 2 else 0
        if not ctx.sequence_parallel or ctx.tp != comm.tp:
            raise Unsupported(
                "oproj_rs", None, f"SP o-projection at TP{ctx.tp} (comm TP{comm.tp}, "
                f"sequence parallel {ctx.sequence_parallel})",
            )  # fmt: skip
        if (
            weight.dtype != torch.bfloat16
            or weight.shape != (comm.width, k)
            or weight.stride() != (k, 1)
            or weight.data_ptr() % 16
            or comm.width != WIDTH
        ):
            raise Unsupported(
                "oproj_rs", None, f"weight {tuple(weight.shape)} {weight.dtype} "
                f"stride {weight.stride()} (want [{WIDTH}, k] BF16 row-major)",
            )  # fmt: skip
        if ctx.heads_per_rank and k != ctx.heads_per_rank * 128:
            raise Unsupported(
                "oproj_rs", None, f"K {k} != {ctx.heads_per_rank} heads x 128"
            )
        self.weight, self.k = weight, k
        self.wview = to_cute(weight)
        self.tactics: dict[int, str] = {}
        self.codes: dict[int, object] = {}
        index = weight.device.index
        for r in rows:
            shape = Shape(r, ctx.q)
            comm.shard(r)  # raises for rows the mailbox cannot carry
            tactic = sp_tactic(ctx.tp, shape)
            if tactic == "warp":
                if r > WARP_ROWS or k % (32 * SPLITS):
                    raise Unsupported(
                        "oproj_rs", shape, f"warp tactic needs rows <= {WARP_ROWS}, "
                        f"K % {32 * SPLITS} == 0 (K {k})",
                    )  # fmt: skip
                self.codes[r] = compile_warp(r, k, comm.tp, comm.rows_cap, index)
            elif tactic != "cublas":
                raise Unsupported("oproj_rs", shape, f"unknown tactic {tactic!r}")
            # the owner half of either tactic and the all-gather (residency-checked)
            comm.reduce_scatter_code(r, push=tactic == "cublas")
            comm.prepare(r)
            self.tactics[r] = tactic

    def choice(self, rows: int) -> Choice:
        tactic = self.tactics.get(rows)
        if tactic is None:
            raise Unsupported("oproj_rs", rows, "not prepared")
        note = f"sp oproj {tactic}" + (f" S{SPLITS}" if tactic == "warp" else "")
        return Choice(Path.GENERIC, tactic, note=note + " (NQ-TUNE)")

    def run(
        self, attn_out: torch.Tensor, out_shard: torch.Tensor, parity: int
    ) -> torch.Tensor:
        rows = attn_out.shape[0]
        tactic = self.tactics.get(rows)
        if (
            tactic is None
            or attn_out.dtype != torch.bfloat16
            or attn_out.shape[1] != self.k
            or attn_out.stride() != (self.k, 1)
        ):
            raise Unsupported(
                "oproj_rs", rows, f"attn_out {tuple(attn_out.shape)} stride "
                f"{attn_out.stride()} {attn_out.dtype} (prepared: [R, {self.k}] BF16)",
            )  # fmt: skip
        comm = self.comm
        if tactic == "cublas":
            partial = comm.partial[:rows]
            torch.mm(attn_out, self.weight.t(), out=partial)
            return comm.reduce_scatter(out_shard, parity, rows, partial)
        _, _, recv, peers = comm.views(parity)
        self.codes[rows](
            to_cute(attn_out), self.wview, recv, peers, Int32(comm.rank),
            stream(self.weight.device),
        )  # fmt: skip
        return comm.reduce_scatter(out_shard, parity, rows)
