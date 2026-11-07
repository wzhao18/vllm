# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared first dense MLP, sharing projection and SiTU decode arithmetic."""

from functools import cache

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64
from cutlass.cute.runtime import make_fake_stream

from ..common.host import architecture, dynamic_fake, dynamic_view, stream
from ..common.policy import batch_table
from ..common.primitives import launch, pack_pair, store_words, unpack_bf16
from ..gemm.projection import PreparedProjection
from ..gemm.splitk import SplitKTactic
from ..gemm.splitk_ring import (
    FitProjection,
    RingProjection,
    RingSplitKGemm,
    _fake,
    fit_smem,
)
from .shared import situ


class DenseSitu:
    def __init__(self, hidden, padded):
        self.hidden, self.padded = hidden, padded

    @cute.jit
    def __call__(self, *args):
        grid = (cute.ceil_div(self.padded, 256), args[0].shape[0], 1)
        launch(self.kernel, args, grid, 256)

    @cute.kernel
    def kernel(self, args):
        x, out, beta, linear = args[0:4]
        tile, row, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        col = tile * 256 + tid
        cute.arch.griddepcontrol_wait()
        if col < self.padded:
            value = cutlass.Float32(0)
            if col < self.hidden:
                gate = cutlass.Float32(x[row, col])
                up = cutlass.Float32(x[row, self.hidden + col])
                value = situ(gate, up, beta, linear, native=True)
            out[row, col] = cutlass.BFloat16(value)
        cute.arch.griddepcontrol_launch_dependents()


@cache
def _compile_activation(hidden, padded, arch):
    args = (dynamic_fake(2), dynamic_fake(2), *map(cutlass.Float32, (4, 25)))
    kernel = DenseSitu(hidden, padded)
    return cute.compile(kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}")


class SituSplitKGemm(RingSplitKGemm):
    """Gate/up GEMM over row tiles of 32 gate rows then the matching 32 up rows
    (``fuse_gate_up``) whose owner rounds both halves to BF16 (as the unfused GEMM
    stores them), applies ``situ`` as ``DenseSitu`` does and stores the BF16 hidden
    rows at ``ring`` (the [B, padded] hidden rows' address); launched with the shared
    memory it uses (``FitSplitKGemm``)."""

    def __init__(self, *, tactic, beta, linear, padded):
        super().__init__(tactic=tactic)
        assert self.cta_m == 64 and self.cta_n * 16 == self.epilog_threads
        self.beta, self.linear, self.padded = beta, linear, padded
        self.fit = fit_smem(tactic) + self.cta_m * self.cta_n * 4

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, bias: cute.Tensor,
        a_map: cutlass.Int64, ring: cutlass.Int64, stream: _cuda.CUstream,
    ):  # fmt: skip
        self.kernel(a, b, c, bias, a_map, ring).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.fit),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.jit
    def publish(
        self, staged: cute.Tensor, mC: cute.Tensor, ring: cutlass.Int64,
        bidx: cutlass.Int32, n_idx: cutlass.Int32, epi_tid: cutlass.Int32,
    ):  # fmt: skip
        """Each epilogue thread: two adjacent hidden columns of one token."""
        cute.arch.barrier(barrier_id=1, number_of_threads=self.epilog_threads)
        token = epi_tid // 16
        column = (epi_tid - token * 16) * 2
        n = n_idx * self.cta_n + token
        if n < mC.layout.shape[1]:
            pair = (staged[column, token], staged[column + 1, token])
            gate = unpack_bf16(pack_pair(*pair, policy=None))
            pair = (staged[column + 32, token], staged[column + 33, token])
            up = unpack_bf16(pack_pair(*pair, policy=None))
            beta, linear = Float32(self.beta), Float32(self.linear)
            low = situ(gate[0], up[0], beta, linear, native=True)
            high = situ(gate[1], up[1], beta, linear, native=True)
            offset = Int64(n) * self.padded + Int64(bidx * 32 + column)
            store_words(ring + offset * 2, (pack_pair(low, high, policy=None),))


def fuse_gate_up(gate_up):
    """[2H, K] gate-then-up rows -> 64-row tiles of 32 gate rows then their up rows."""
    rows, k = gate_up.shape
    tiles = rows // 64
    pairs = gate_up.detach().view(2, tiles, 32, k).transpose(0, 1)
    return pairs.reshape(rows, k).contiguous()


@cache
def compile_situ_gate(tactic, beta, linear, padded, arch):
    kernel = SituSplitKGemm(
        tactic=SplitKTactic(*tactic), beta=beta, linear=linear, padded=padded
    )
    y = _fake(0)
    args = (_fake(1), _fake(1), y, y, Int64(0), Int64(0), make_fake_stream())
    return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")


class SituProjection(RingProjection):
    """The fused gate/up + SiTU (``run_ring(x, gates, hidden_address)``)."""

    def __init__(self, weight, policies, beta, linear, padded):
        super().__init__(fuse_gate_up(weight), policies)
        self.situ_args = beta, linear, padded

    def prepare(self):
        arch = architecture(self.device)
        with torch.accelerator.device_index(self.device.index):
            for batch, tactic in self.policies.items():
                code = compile_situ_gate(tactic, *self.situ_args, arch)
                self.kernels[batch] = code
                self.extra[batch] = self.weight_map(tactic[0])
        return self


# Rubin (compute capability 10.7) TP16 dense MLP tactics per batch (1-GPU, synthetic
# TP16 weights, CUDA graphs, chained behind an o-proj-like / SiTU-like PDL predecessor,
# cold L2, B8 us/call): gate/up 64.8.4.8 launched with the shared memory it uses
# (~156 KiB, ``FitSplitKGemm``) instead of 227 KiB 7.50 -> 5.19 (B4 7.48 -> 5.17; 9+
# stages lose it); the ring down projection 64.8.1.10 (112 CTAs) instead of 128.8.1.6
# (56) 5.06 -> 4.43. Both keep the split: bitwise.
RUBIN_FIT_GATE = {2112: "1-8:1"}
# Rubin TP16 batches whose gate/up GEMM applies SiTU in its epilogue
# (``SituSplitKGemm``, same tactic and launch as the fit gate/up; bitwise) instead of a
# ``DenseSitu`` launch.
RUBIN_FUSED_SITU = {2112: "1-8:1"}
RUBIN_RING_DOWN = {2112: "1-8:64.8.1.10"}


def rubin_table(table, hidden, device):
    if torch.cuda.get_device_capability(device) != (10, 7):
        return {}
    return batch_table(table.get(hidden, ""))


class PreparedDenseMLP:
    """Dense MLP; TP16 pads down K 2112 -> 2176 and the activation zero-fills it.

    With ``ring`` (``ring_batches``, Rubin TP16) the down projection publishes its BF16
    partial rows into slot ``rank`` of every rank's [TP, cap, 7168] Lamport mailbox
    (``ring`` = that slot's multicast address) and the next AttnRes sums the TP slots
    in rank order: a one-shot all-reduce fused into its consumer (``run(consume=)``)."""

    def __init__(self, mlp, batches, reduce=None, ring=None, ring_batches=(), rows=64):
        """``rows``: the largest batch (the activation buffers' rows)."""
        gate_up, down = mlp.gate_up_proj.weight, mlp.down_proj.weight
        self.device, self.reduce = gate_up.device, reduce
        self.hidden = hidden = gate_up.shape[0] // 2
        self.padded = padded = (hidden + 127) // 128 * 128
        linear = -1 if mlp.act_fn.linear_beta is None else mlp.act_fn.linear_beta
        self.scalars = tuple(map(cutlass.Float32, (mlp.act_fn.beta, linear)))
        if padded != hidden:
            down = torch.nn.functional.pad(down.detach(), (0, padded - hidden))
        self.gates = gate_up.new_empty((rows, 2 * hidden))
        # The fused SiTU never writes the zero padding columns.
        self.hidden_rows = gate_up.new_zeros((rows, padded))
        # Batches above 64 rows run the 64-row tactics (generic, untuned): the
        # shape-dynamic split-K GEMMs repeat their N tile over the rows.
        tile = {b: next((n for n in (8, 16, 32, 64) if b <= n), 64) for b in batches}
        up_tiles = {8: (64, 8, 4, 8), 16: (64, 16, 2, 8), 32: (64, 32, 2, 8)}
        up_tiles[64] = (64, 64, 2, 5)
        down_tiles = {8: (128, 8, 1, 6), 16: (128, 16, 1, 6), 32: (128, 32, 1, 5)}
        down_tiles[64] = (64, 64, 1, 5)
        policy = lambda tiles: {b: tiles[n] for b, n in tile.items()}
        gate_tiles = policy(up_tiles)
        fit = rubin_table(RUBIN_FIT_GATE, hidden, self.device)
        fit_tiles = {b: t for b, t in gate_tiles.items() if fit.get(b)}
        plain = {b: t for b, t in gate_tiles.items() if b not in fit_tiles}
        self.gate = PreparedProjection(gate_up, plain).prepare()
        if fit_tiles:
            # One provider: the fit kernels share the weight and its maps.
            self.gate_fit = fitted = FitProjection(gate_up, fit_tiles).prepare()
            self.gate.kernels.update(fitted.kernels)
            self.gate.extra.update(fitted.extra)
        fused = rubin_table(RUBIN_FUSED_SITU, hidden, self.device)
        fused = {b: t for b, t in fit_tiles.items() if fused.get(b)}
        self.situ_gate = None
        if fused:
            beta, linear_beta = float(mlp.act_fn.beta), float(linear)
            args = (gate_up, fused, beta, linear_beta, padded)
            self.situ_gate = SituProjection(*args).prepare()
        self.down = PreparedProjection(down, policy(down_tiles))
        self.down.prepare()
        self.ring, self.ring_batches = ring, frozenset()
        if ring is not None:
            ring_down = rubin_table(RUBIN_RING_DOWN, hidden, self.device)
            ring_tiles = {
                b: ring_down.get(b, down_tiles[tile[b]]) for b in ring_batches
            }
            # The ring's AttnRes reads the prefix (written by this layer's o-projection)
            # before its PDL wait: the down GEMM triggers only after its weight warp
            # has refilled a stage, i.e. after its own wait, so K tiles must exceed the
            # stages.
            for m, n, split, stages in ring_tiles.values():
                if padded // 128 // split <= stages:
                    raise ValueError(f"ring down tactic {(m, n, split, stages)}")
            self.ring_down = RingProjection(down, ring_tiles).prepare()
            self.ring_batches = frozenset(ring_batches)
        self.activation = _compile_activation(hidden, padded, architecture(self.device))

    def run(self, x, out, *, consume=None):
        gate, hidden = self.gates[: x.shape[0]], self.hidden_rows[: x.shape[0]]
        situ_gate = self.situ_gate
        if situ_gate is not None and x.shape[0] in situ_gate.kernels:
            situ_gate.run_ring(x, gate, hidden.data_ptr())
        else:
            self.gate.run(x, gate)
            views = (dynamic_view(gate), dynamic_view(hidden))
            self.activation(*views, *self.scalars, stream(self.device))
        if consume is not None:
            if x.shape[0] not in self.ring_batches:
                raise ValueError(f"B{x.shape[0]}: no ring for the dense MLP")
            self.ring_down.run_ring(hidden, out, self.ring)
            return consume(out)
        self.down.run(hidden, out)
        if self.reduce is not None:
            reduced = self.reduce(out)
            if reduced.data_ptr() != out.data_ptr():
                out.copy_(reduced)
        return out
