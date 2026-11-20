# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The layer-0 dense MLP on this rank's SP shard (replicated weights, ``KimiMLP``
under sequence parallel) through ``gemm.dense_stream.DenseStream``: the stock
gate_up GEMM + SituAndMul fused into one die-local weight stream, then the down
GEMM. Every rank streams the whole 1.45 GB weight a step for its few shard rows,
so the layer is bound by the per-die HBM read rate: the weights are re-homed per
die in place (``place_rows``: same bytes and layout, so the stock module, which
eager / profiling / warmup steps still run, reads the same values) and each SM
streams only rows homed on its die.

Numerics: FP32 accumulation and one BF16 rounding per GEMM, SiTU on the
BF16-rounded gate / up as the stock op (only the FP32 summation order differs);
deterministic."""

import torch

from vllm.logger import init_logger

from ..cutedsl.common import dies
from ..cutedsl.common.host import stream as cute_stream
from ..cutedsl.gemm import dense_stream as dsm
from ..shapes import Choice, Path, Unsupported

logger = init_logger(__name__)

PLACE_SAMPLES, PLACE_PASSES = 8, 5
# The most shard rows DenseStream serves; above, the stock dense MLP (cuBLAS) on the
# same (re-homed) weights. In-step ABBA (fastab dd-ds1, D3-32 slice, Hecate): rows 32
# -10.9 us/step; rows 64 / 96 +15.6 / +15.7 (gate_up 111-119 us in-step vs 90-98
# standalone: the predecessor's tail overlap and the end-of-stream SiTU).
STREAM_ROWS = 32


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
    """(int32 [1024] die of each %smid, SMs of die 0, SMs of die 1); one die (every
    SM on die 0) without a trusted two-die map."""
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    sms = torch.cuda.get_device_properties(index).multi_processor_count
    table = torch.zeros(1024, dtype=torch.int32, device=device)
    if dies.domain_count(index) < 2:
        return table, sms, 0
    try:
        domains = tuple(dies.sm_domains(index))
    except RuntimeError as exc:
        logger.warning_once("Kimi K3 dense MLP: no trusted SM -> die map (%s)", exc)
        return table, sms, 0
    wrong = dies.stream_check(index)
    if wrong:
        logger.warning_once(
            "Kimi K3 dense MLP: die map and streaming disagree on SMs %s", wrong
        )
        return table, sms, 0
    table[: len(domains)] = torch.tensor(domains, dtype=torch.int32, device=device)
    return table, domains.count(0), domains.count(1)


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
    """``run(h [n, H])`` -> [n, H] BF16 (stock ``mlp(h)``, the GEMMs' FP32 summation
    order aside) for every captured shard row count ``n`` in ``rows``."""

    def __init__(self, mlp, rows, device, index):
        cause = dense_cause(mlp)
        if cause is not None:
            raise Unsupported("dense", None, f"layer {index}: {cause}")
        gate_up, down = mlp.gate_up_proj.weight, mlp.down_proj.weight
        hidden, inter = down.shape
        self.hidden, self.inter = hidden, inter
        act = mlp.act_fn
        beta = float(act.beta)
        linear = -1.0 if act.linear_beta is None else float(act.linear_beta)
        self.rows = tuple(n for n in sorted(set(rows)) if n <= STREAM_ROWS)
        if not self.rows:
            raise Unsupported("dense", None, f"layer {index}: no shard rows <= {STREAM_ROWS}")
        self.table, self.s0, self.s1 = die_map(device)
        two = self.s1 > 0
        nb0_gu = dsm.split_blocks(inter // 128, self.s0, self.s1)
        nb0_d = dsm.split_blocks(hidden // 128, self.s0, self.s1)
        if two:
            half_gu, half_d = nb0_gu * 128, nb0_d * 128
            placed = place_rows(gate_up, lambda row: int(row % inter >= half_gu))
            placed &= place_rows(down, lambda row: int(row >= half_d))
            if not placed:
                logger.warning_once(
                    "Kimi K3 dense MLP: die-local weight chunks not confirmed"
                )
        slots = self.s0 + self.s1
        top = self.rows[-1]
        self.h = torch.empty(top, inter, dtype=torch.bfloat16, device=device)
        self.y = torch.empty(top, hidden, dtype=torch.bfloat16, device=device)
        self.sync_gu = torch.zeros(dsm.SYNC_WORDS, dtype=torch.int32, device=device)
        self.sync_d = torch.zeros(dsm.SYNC_WORDS, dtype=torch.int32, device=device)
        self.kernels = {}
        ws_bytes = 0
        for bucket in sorted({-(-n // 16) * 16 for n in self.rows}):
            op_gu = dsm.DenseStream(bucket, inter, hidden, situ_mode=True, beta=beta,
                                    linear_beta=linear, two_dies=two, nb0=nb0_gu)  # fmt: skip
            op_d = dsm.DenseStream(bucket, hidden, inter, two_dies=two, nb0=nb0_d)
            ws_bytes = max(ws_bytes, op_gu.ws_bytes(slots), op_d.ws_bytes(slots))
            self.kernels[bucket] = (op_gu, op_d)
        self.ws = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)
        self.weights = (gate_up, down)
        for bucket, (op_gu, op_d) in self.kernels.items():
            x = torch.zeros(bucket, hidden, dtype=torch.bfloat16, device=device)
            self.kernels[bucket] = (
                self._compile(op_gu, gate_up, x, self.h[:bucket], self.sync_gu),
                self._compile(op_d, down, self.h[:bucket], self.y[:bucket], self.sync_d),
            )
        logger.info_once(
            "Kimi K3 dense MLP (layer %d): DenseStream gate_up+SiTU / down, %s, "
            "shard rows %d-%d (%d kernels)", index,
            f"die-local ({self.s0} + {self.s1} SMs)" if two else "one die",
            self.rows[0], self.rows[-1], 2 * len(self.kernels),
            scope="local",
        )  # fmt: skip

    def _compile(self, op, w, x, y, sync):
        dev = w.device
        fn, _ = dsm.compile_dense_stream(op, w, x, y, self.ws, sync, self.table,
                                         self.s0, self.s1, cute_stream(dev))  # fmt: skip
        return fn

    def serves(self, n):
        return n in self.rows

    def choice(self, n):
        if n > STREAM_ROWS:
            return Choice(
                Path.STOCK, note=f"stock dense MLP above {STREAM_ROWS} shard rows (sp_dense)"
            )
        if n not in self.rows:
            raise Unsupported("dense", n, "shard rows not prepared")
        return Choice(Path.GENERIC, "stream", note="DenseStream die-local (sp_dense)")

    def run(self, x):
        n = x.shape[0]
        fn_gu, fn_d = self.kernels[-(-n // 16) * 16]
        gate_up, down = self.weights
        h, y = self.h[:n], self.y[:n]
        stream = cute_stream(x.device)
        s0, s1, rows = (dsm.Int32(v) for v in (self.s0, self.s1, n))
        fn_gu(dsm.as_cute(gate_up, k_major=True), dsm.as_cute(x, k_major=True),
              dsm.as_cute(h, rows=True), dsm.as_cute(self.ws), dsm.as_cute(self.sync_gu),
              dsm.as_cute(self.table), s0, s1, rows, stream)  # fmt: skip
        fn_d(dsm.as_cute(down, k_major=True), dsm.as_cute(h, k_major=True),
             dsm.as_cute(y, rows=True), dsm.as_cute(self.ws), dsm.as_cute(self.sync_d),
             dsm.as_cute(self.table), s0, s1, rows, stream)  # fmt: skip
        return y
