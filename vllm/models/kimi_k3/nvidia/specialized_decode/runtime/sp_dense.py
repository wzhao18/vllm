# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The layer-0 dense MLP on this rank's SP shard (replicated weights, ``KimiMLP``
under sequence parallel) through ``gemm.dense_flow.DenseFlow``: the stock
gate_up GEMM + SituAndMul + down GEMM in ONE die-local weight stream. Every rank
streams the whole 1.45 GB weight a step for its few shard rows, so the layer is
bound by the per-die HBM read rate: the weights are re-homed per die in place
(``place_rows``: same bytes and layout, so the stock module, which eager /
profiling / warmup steps still run, reads the same values), half the weight
blocks on each die, and each SM streams only rows homed on its die.

Numerics: FP32 accumulation and one BF16 rounding per GEMM, SiTU on the
BF16-rounded gate / up as the stock op (only the FP32 summation order differs);
deterministic."""

import torch

from vllm.logger import init_logger

from ..cutedsl.common import dies
from ..cutedsl.common.host import stream as cute_stream
from ..cutedsl.gemm import dense_flow as dfl
from ..cutedsl.gemm import dense_stream as dsm
from ..shapes import Choice, Path, Unsupported

logger = init_logger(__name__)

PLACE_SAMPLES, PLACE_PASSES = 8, 5
MAX_ROWS = dfl.MAX_ROWS  # shard rows per GPU (q x requests per DP group / TP)


def dense_cause(mlp):
    """Why ``mlp`` (the layer-0 ``KimiMLP``) cannot run here, or None."""
    act = getattr(mlp, "act_fn", None)
    if getattr(mlp, "shard_sequence_parallel", False):
        return "TP-sharded under sequence parallel (gate_up / down not replicated)"
    if getattr(mlp, "gemm_rs_ar", None) is not None:
        return "a GEMM + RS / AR down projection"
    if not (hasattr(act, "beta") and hasattr(act, "linear_beta")):
        return f"activation {type(act).__name__} (SiTU only)"
    gate_up, down = mlp.gate_up_proj, mlp.down_proj
    for lin in (gate_up, down):
        if getattr(lin, "bias", None) is not None:
            return "biased projection"
        if lin.weight.dtype != torch.bfloat16 or not lin.weight.is_contiguous():
            return f"{lin.weight.dtype} / non-contiguous weight (BF16 only)"
        if getattr(lin, "tp_size", 1) != 1:
            return "TP-sharded projection"
    hidden, inter = down.weight.shape
    if tuple(gate_up.weight.shape) != (2 * inter, hidden):
        return f"gate_up {tuple(gate_up.weight.shape)} vs down {(hidden, inter)}"
    if not (dsm.supports(16, inter, hidden, True) and dsm.supports(16, hidden, inter)):
        return f"hidden {hidden} / intermediate {inter} (multiples of 256 / 64)"
    return None


def die_map(device):
    """(die of each %smid (-1: none), SMs of die 0, SMs of die 1); one die (every SM
    on die 0) without a trusted two-die map."""
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    sms = torch.cuda.get_device_properties(index).multi_processor_count
    one = ((0,) * sms, sms, 0)
    if dies.domain_count(index) < 2:
        return one
    try:
        domains = tuple(dies.sm_domains(index))
    except RuntimeError as exc:
        logger.warning_once("Kimi K3 dense MLP: no trusted SM -> die map (%s)", exc)
        return one
    wrong = dies.stream_check(index)
    if wrong:
        logger.warning_once(
            "Kimi K3 dense MLP: die map and streaming disagree on SMs %s", wrong
        )
        return one
    return domains, domains.count(0), domains.count(1)


def place_rows(weight, die_of_row):
    """Re-home the contiguous 2-D ``weight`` parameter in place: 2 MiB chunk k on the
    die of its middle row (``die_of_row``); same bytes, layout and values, the
    original released. False if the homes could not be confirmed (kept placed:
    correct, only slower)."""
    w = weight.data
    row_bytes = w.shape[1] * w.element_size()
    nbytes = w.numel() * w.element_size()
    chunks = -(-nbytes // dies.CHUNK)
    domains = [
        die_of_row(min(w.shape[0] - 1, (k * dies.CHUNK + dies.CHUNK // 2) // row_bytes))
        for k in range(chunks)
    ]
    raw = dies.localized_empty(nbytes, domains, w.device)
    copy = raw.view(w.dtype).view(w.shape)
    copy.copy_(w)
    sample = tuple(sorted({chunks * i // PLACE_SAMPLES for i in range(PLACE_SAMPLES)}))
    wrong, sms = sample, 4
    for attempt in range(PLACE_PASSES):
        homes = dies.chunk_homes(copy, wrong, sms)
        wrong = tuple(c for c, home in zip(wrong, homes) if home != domains[c])
        if not wrong:
            break
        sms = 8 if attempt < 2 else 16
    weight.data = copy
    torch.accelerator.empty_cache()  # the original
    return not wrong


class SpDenseMlp:
    """``run(h [n, H])`` -> [n, H] BF16 (stock ``mlp(h)``, the down GEMM's FP32
    summation order aside: gate_up sums in K order, h bitwise == stock) for every
    captured shard row count ``n`` in ``rows``: one ``dense_flow.DenseFlow`` launch
    (gate_up + SiTU + down in one die-local weight stream)."""

    def __init__(self, mlp, rows, device, index):
        cause = dense_cause(mlp)
        if cause is not None:
            raise Unsupported("dense", None, f"layer {index}: {cause}")
        gate_up, down = mlp.gate_up_proj.weight, mlp.down_proj.weight
        hidden, inter = down.shape
        self.hidden, self.inter = hidden, inter
        self.act = mlp.act_fn
        beta = float(self.act.beta)
        linear = -1.0 if self.act.linear_beta is None else float(self.act.linear_beta)
        self.rows = tuple(sorted(set(rows)))
        if not self.rows or self.rows[-1] > MAX_ROWS:
            raise Unsupported("dense", None, f"layer {index}: shard rows {self.rows}")
        if not all(dfl.supports(n, hidden, inter) for n in self.rows):
            raise Unsupported("dense", None, f"layer {index}: {hidden} x {inter}")
        self.domains, self.s0, self.s1 = die_map(device)
        two = self.s1 > 0
        nbA0 = dfl.die_blocks(inter // 64, self.s1)
        nbB0 = dfl.die_blocks(hidden // 128, self.s1)
        if two:  # a gate_up block's gate and up rows (64 each) and its down rows per die
            placed = place_rows(gate_up, lambda row: int((row % inter) // 64 >= nbA0))
            placed &= place_rows(down, lambda row: int(row // 128 >= nbB0))
            if not placed:
                logger.warning_once(
                    "Kimi K3 dense MLP: die-local weight chunks not confirmed"
                )
        top = self.rows[-1]
        self.h = torch.empty(top, inter, dtype=torch.bfloat16, device=device)
        self.y = torch.empty(top, hidden, dtype=torch.bfloat16, device=device)
        self.sync = torch.zeros(dfl.SYNC_WORDS, dtype=torch.int32, device=device)
        self.weights = (gate_up, down)
        self.kernels, ws_bytes = {}, 0
        for bucket in sorted({-(-n // 16) * 16 for n in self.rows}):
            op = dfl.DenseFlow(bucket, hidden, inter, beta, linear, nbA0=nbA0, nbB0=nbB0,
                               two_dies=two)  # fmt: skip
            ws_bytes = max(ws_bytes, op.ws_bytes(self.s0 + self.s1))
            self.kernels[bucket] = (op, dfl.flow_tables(op, self.domains, None, device,
                                                        whole=True))
        self.ws = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)
        for bucket, (op, tables) in list(self.kernels.items()):
            x = torch.zeros(bucket, hidden, dtype=torch.bfloat16, device=device)
            fn, _ = dfl.compile_dense_flow(op, gate_up, x, down, self.h, self.y[:bucket],
                                           self.ws, self.sync, *tables, self.s0, self.s1,
                                           cute_stream(device))  # fmt: skip
            self.kernels[bucket] = (fn, tables)
        logger.info_once(
            "Kimi K3 dense MLP (layer %d): DenseFlow gate_up + SiTU + down in one grid, "
            "%s, shard rows %d-%d (%d kernels)", index,
            f"die-local ({self.s0} + {self.s1} SMs)" if two else "one die",
            self.rows[0], self.rows[-1], len(self.kernels),
            scope="local",
        )  # fmt: skip

    def serves(self, n):
        return n in self.rows

    def choice(self, n):
        if n not in self.rows:
            raise Unsupported("dense", n, "shard rows not prepared")
        return Choice(Path.GENERIC, "flow", note="DenseFlow die-local one-grid MLP (sp_dense)")

    def run(self, x):
        n = x.shape[0]
        fn, tables = self.kernels[-(-n // 16) * 16]
        gate_up, down = self.weights
        y = self.y[:n]
        fn(dfl.ds.as_cute(gate_up, k_major=True), dfl.ds.as_cute(x, k_major=True),
           dfl.ds.as_cute(down, k_major=True), dfl.ds.as_cute(self.h, k_major=True),
           dfl.ds.as_cute(y, rows=True), dfl.ds.as_cute(self.ws), dfl.ds.as_cute(self.sync),
           *(dfl.ds.as_cute(t) for t in tables),
           *(dfl.Int32(v) for v in (self.s0, self.s1, n)), cute_stream(x.device))  # fmt: skip
        return y
