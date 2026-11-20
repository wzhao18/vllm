# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 projections of the SP decode (NVFP4 checkpoint ``modelopt_mixed``: the KDA
in / o projections and the shared experts are online MXFP8, W8A8 with 32-wide E8M0
blocks; ledger item B4).

``Mxfp8Projection`` is ``decode.DecodeProjection`` for an MXFP8 weight: ``out[R, N] =
bf16(dequant(xq[R, K]) @ dequant(wq[N, K])^T)`` with FP32 accumulation and one BF16
rounding, from an MXFP8 activation (FP8 E4M3 values + F8_128x4-swizzled E8M0 scales,
FlashInfer ``mxfp8_quantize`` semantics). The weight is the stock linear's own storage
(no copy): FP8 rows [N, K] and the F8_128x4 scales of ``KimiK3Mxfp8Kernel`` /
FlashInfer's MXFP8 kernels. A BF16 input is quantized first by
``ops.mxfp8_act.quantize`` into a capacity-sized scratch (bitwise FlashInfer's
quantization, which stock's online-MXFP8 linear runs on its BF16 input); producers
that already wrote the MXFP8 activation (``situ_and_mul_quantize``) pass it instead.

Kinds (every kind sums K in FP32 and rounds once to BF16; the order differs):
  "fi_cute"     FlashInfer CuTe DSL MXFP8 GEMM (stock's choice at <= 16 rows)
  "fi_cutlass"  FlashInfer CUTLASS MXFP8 GEMM
  "scaled_mm"   cuBLASLt ``torch._scaled_mm`` (stock's choice above 16 rows)
  "dense_mx8"   ``dense_mx8.DenseMx8``: 2-CTA M256 over all R tokens, K split in two,
                weights loaded before the PDL wait

The per-R kind is GENERIC = stock's own per-call rule (``nvidia.mxfp8.use_cute_dsl``:
the same kernel as the stock linear, so the stock bits) unless ``MX_TUNED`` names a
measured VR200 winner for the weight's [N, K] at that R. A row count without a kind,
or an operand a kind cannot take, raises ``shapes.Unsupported`` at build; nothing falls
back at run."""

import torch

from ...shapes import Choice, Path, Shape, Unsupported, rows_policy

MX_KINDS = ("fi_cute", "fi_cutlass", "scaled_mm", "dense_mx8")
BLOCK = 32  # MXFP8 block along K
DENSE_MAX_ROWS = 384  # dense_mx8: three 128-token MMA blocks in tensor memory

# TUNED (VR200, D3-32 per-GPU shapes, 2026-11-18 D-GEMM bench1: 1 GPU, a CUDA graph of
# 8-64 back-to-back calls each on its own cold weight copy, pre-quantized input; us per
# call fi_cute / scaled_mm (fi_cutlass 2-6x slower everywhere, trtllm-gen slower above
# 16 rows)): per MXFP8 weight [N, K] the kind per rows; rows outside keep GENERIC.
#   kda_in 12448 x 7168  R16 9.55 / 16.20, R128 10.96 / 12.40, R256 11.59 / 13.50,
#                        R384 16.73 / 14.63
#   shared_gu 12288 x 7168  n4 9.30 / 14.09, n32 10.01 / 11.98, n64 10.12 / 12.17,
#                           n96 12.92 / 12.23
#   shared_down 7168 x 6144  n4 6.09 / 9.85, n32 7.64 / 7.68, n64 7.75 / 8.93,
#                            n96 8.44 / 7.75
#   o_proj 7168 x 3072  R16 4.92 / 4.95, R128 5.84 / 5.13, R256 6.33 / 6.03, R384
#                       7.54 / 6.26 (= stock's rule: no entry)
MX_TUNED: dict[tuple[int, int], dict] = {
    (12448, 7168): {"*": "1-320:fi_cute,321-:scaled_mm"},
    (12288, 7168): {"*": "1-80:fi_cute,81-:scaled_mm"},
    (7168, 6144): {"*": "1-80:fi_cute,81-:scaled_mm"},
}  # fmt: skip


def stock_kind(n: int, k: int, rows: int) -> str:
    """Stock's online-MXFP8 kernel at ``rows`` (``KimiK3Mxfp8Kernel``)."""
    from ....mxfp8 import use_cute_dsl

    return "fi_cute" if use_cute_dsl(n, k, rows) else "scaled_mm"


def mxfp8_tactics(n: int, k: int, rows) -> dict[int, str]:
    """The kind per R: ``MX_TUNED``'s where it names one, stock's rule otherwise."""
    out = {}
    for r in sorted(set(rows)):
        kind = rows_policy(MX_TUNED.get((n, k), ""), Shape(r, 1), default=None)
        out[r] = kind if kind is not None else stock_kind(n, k, r)
    return out


def mxfp8_weight(linear, family: str):
    """The stock online-MXFP8 linear's weight as (FP8 [N, K] row-major view, swizzled
    E8M0 scales as uint8): its own storage. Raises unless the linear is online MXFP8
    on a kernel whose layout is known (CuTe DSL "kn" view or row-major "nk")."""
    from ....mxfp8 import _kernel_layout, is_mxfp8_linear

    where = getattr(linear, "prefix", type(linear).__name__)
    if not is_mxfp8_linear(linear):
        raise Unsupported(
            family, None, f"{where}: {type(linear.quant_method).__name__} is not "
            "online MXFP8",
        )  # fmt: skip
    layout = _kernel_layout(linear)
    weight = linear.weight.data
    if layout == "kn":
        weight = weight.t()
    if layout is None or weight.dtype != torch.float8_e4m3fn or weight.dim() != 2:
        raise Unsupported(
            family, None, f"{where}: MXFP8 weight layout {layout} {weight.dtype}"
        )
    n, k = weight.shape
    if weight.stride() != (k, 1) or k % (4 * BLOCK):
        raise Unsupported(
            family, None, f"{where}: MXFP8 weight [{n}, {k}] stride {weight.stride()}"
        )
    scale = linear.weight_scale.data.view(torch.uint8).reshape(-1)
    want = -(-n // 128) * 128 * (k // BLOCK)
    if scale.numel() != want:
        raise Unsupported(
            family, None, f"{where}: {scale.numel()} scale bytes (F8_128x4: {want})"
        )
    if getattr(linear, "bias", None) is not None:
        raise Unsupported(family, None, f"{where}: a linear with bias")
    return weight, scale


def projection_rows(linear) -> int:
    """N of a linear's [N, K] weight: the BF16 one, or an online-MXFP8 one in either
    of its kernels' layouts."""
    from ....mxfp8 import _kernel_layout, is_mxfp8_linear

    if is_mxfp8_linear(linear) and _kernel_layout(linear) == "kn":
        return linear.weight.shape[1]
    return linear.weight.shape[0]


class Mxfp8Input:
    """A capacity-rows MXFP8 activation scratch [cap, K] whose first R rows are the
    MXFP8 input of an R-row call (F8_128x4 scales: whole 128-row tiles, so the first
    ceil(R / 128) tiles of the capacity layout are exactly an R-row layout)."""

    def __init__(self, cap: int, k: int, device):
        from ....ops import mxfp8_act

        if not mxfp8_act.supports(k):
            raise Unsupported("mxfp8", None, f"MXFP8 activation K {k}")
        self.buf = mxfp8_act.Mxfp8Buffer(cap, k, device)
        self.k = k
        self._views: dict[int, object] = {}

    def view(self, rows: int):
        view = self._views.get(rows)
        if view is None:
            from ....ops import mxfp8_act

            if rows > self.buf.rows:
                raise Unsupported("mxfp8", rows, f"> capacity {self.buf.rows} rows")
            view = object.__new__(mxfp8_act.Mxfp8Buffer)
            view.rows, view.k, view.padded_cols = rows, self.k, self.buf.padded_cols
            view.data = self.buf.data[:rows]
            view.scale = self.buf.scale[: -(-rows // 128) * 128 * view.padded_cols]
            view.orig_dtype = self.buf.orig_dtype
            self._views[rows] = view
        return view

    def quantize(self, x: torch.Tensor):
        """BF16 [R, K] -> this scratch's R-row MXFP8 view (bitwise FlashInfer's
        ``mxfp8_quantize``)."""
        from ....ops import mxfp8_act

        view = self.view(x.shape[0])
        mxfp8_act.quantize(x, out=view)
        return view


_INPUTS: dict[tuple, Mxfp8Input] = {}


def shared_input(family: str, cap: int, k: int, device) -> Mxfp8Input:
    """One MXFP8 input scratch per (family, K, device), shared by every layer's
    projection of that family (layers run one after another; the largest capacity
    wins), instead of one per layer."""
    key = (family, k, torch.device(device))
    scratch = _INPUTS.get(key)
    if scratch is None or scratch.buf.rows < cap:
        scratch = _INPUTS[key] = Mxfp8Input(cap, k, device)
    return scratch


class Mxfp8Projection:
    """``out[R, N] = x @ W^T`` for an MXFP8 weight (module docstring) for each
    prepared R with its kind. ``run(x, out)``: ``x`` BF16 [R, K] (quantized into the
    scratch) or an ``mxfp8_act.Mxfp8Buffer`` view of R rows; ``out`` BF16 [R, N]
    packed rows."""

    def __init__(self, linear_or_weight, rows, *, family: str, tactics=None,
                 cap: int | None = None):  # fmt: skip
        self.family = family
        if isinstance(linear_or_weight, tuple):
            weight, scale = linear_or_weight
        else:
            weight, scale = mxfp8_weight(linear_or_weight, family)
        self.weight, self.scale = weight, scale
        n, k = weight.shape
        self.n, self.k = n, k
        self.rows = tuple(sorted(set(rows)))
        if not self.rows or self.rows[0] < 1:
            raise Unsupported(family, None, f"rows {rows!r}")
        self.kinds = dict(tactics) if tactics is not None else mxfp8_tactics(
            n, k, self.rows
        )
        for r in self.rows:
            kind = self.kinds.get(r)
            if kind not in MX_KINDS:
                raise Unsupported(family, r, f"MXFP8 kind {kind!r} (kinds {MX_KINDS})")
        self.scale_e8 = scale.view(torch.float8_e8m0fnu)
        self.input = shared_input(family, cap or self.rows[-1], k, weight.device)
        for r in self.rows:
            if self.kinds[r] == "dense_mx8" and (r > DENSE_MAX_ROWS or k % 256):
                raise Unsupported(family, r, f"dense_mx8 needs R <= {DENSE_MAX_ROWS}, K % 256")
        self._dense = {
            r: _dense_engine(n, k, r, weight.device)
            for r in self.rows if self.kinds[r] == "dense_mx8"
        }  # fmt: skip

    def _gemm(self, kind, xq, xs, out):
        import flashinfer

        if kind == "dense_mx8":
            self._dense[xq.shape[0]].run(self.weight, self.scale, xq, xs, out)
            return out
        if kind == "scaled_mm":
            return torch._scaled_mm(
                xq, self.weight.t(), scale_a=xs.view(torch.float8_e8m0fnu),
                scale_b=self.scale_e8, out_dtype=torch.bfloat16, out=out,
            )  # fmt: skip
        backend = "cute-dsl" if kind == "fi_cute" else "cutlass"
        return flashinfer.mm_mxfp8(
            xq, self.weight.t(), xs, self.scale, out=out, out_dtype=torch.bfloat16,
            backend=backend,
        )  # fmt: skip

    def run(self, x, out: torch.Tensor) -> torch.Tensor:
        rows = x.shape[0] if isinstance(x, torch.Tensor) else x.rows
        kind = self.kinds.get(rows)
        if kind is None:
            raise Unsupported(self.family, rows, f"not prepared ({self.prepared()})")
        q = self.input.quantize(x) if isinstance(x, torch.Tensor) else x
        if q.rows != rows or q.k != self.k:
            raise Unsupported(self.family, rows, f"MXFP8 input [{q.rows}, {q.k}]")
        result = self._gemm(kind, q.data, q.scale, out)
        if result.data_ptr() != out.data_ptr():  # (never: both kinds take out=)
            raise Unsupported(self.family, rows, f"{kind} did not write out=")
        return out

    def check_operands(self, x: torch.Tensor, out: torch.Tensor) -> None:
        n, k = self.n, self.k
        problems = []
        if x.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
            problems.append(f"dtypes {x.dtype} / {out.dtype} (BF16)")
        if x.dim() != 2 or x.shape[1] != k or x.stride() != (k, 1):
            problems.append(f"x {tuple(x.shape)} stride {x.stride()}: packed [R, {k}]")
        if out.dim() != 2 or out.shape[1] != n or out.stride() != (n, 1):
            problems.append(f"out {tuple(out.shape)} stride {out.stride()}")
        if out.data_ptr() % 16:
            problems.append("out not 16 B aligned")
        if problems:
            raise Unsupported(self.family, None, "; ".join(problems))

    def prepared(self):
        return f"{len(self.rows)} prepared R in {self.rows[0]}-{self.rows[-1]}"

    def engine(self, rows):
        return self.kinds[rows]

    def choice(self, rows: int) -> Choice:
        if rows not in self.kinds:
            raise Unsupported(self.family, rows, f"not prepared ({self.prepared()})")
        kind = self.kinds[rows]
        tuned = rows_policy(MX_TUNED.get((self.n, self.k), ""), Shape(rows, 1),
                            default=None) == kind  # fmt: skip
        path = Path.TUNED if tuned else Path.GENERIC
        return Choice(path, tactic=("mxfp8", kind), note=f"mxfp8 {kind} {self.n}x{self.k}")


class _DenseEngine:
    """One compiled ``DenseMx8`` shape and its split-K scratch (shared by every
    projection of the same shape on the device: layers run one after another)."""

    def __init__(self, n, k, rows, device):
        from ..common.host import stream
        from .dense_mx8 import DenseMx8

        self.gemm = DenseMx8(n, k, rows, n)
        words, tickets = self.gemm.scratch_words()
        self.part = torch.empty(words, dtype=torch.float32, device=device)
        self.tickets = torch.zeros(tickets, dtype=torch.int32, device=device)
        self.device = device
        self._stream = stream
        import cutlass.cute as cute

        any_ = self.part  # compile-time only: the pointers' types, not their values
        args = self._ptrs(any_, any_, any_, any_, any_)
        self.fn = cute.compile(self.gemm, *args, stream(device))

    def _ptrs(self, w, ws, xq, xs, out):
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_ptr

        g = cute.AddressSpace.gmem
        f8, e8 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
        return (
            make_ptr(f8, w.data_ptr(), g, assumed_align=16),
            make_ptr(e8, ws.data_ptr(), g, assumed_align=16),
            make_ptr(f8, xq.data_ptr(), g, assumed_align=16),
            make_ptr(e8, xs.data_ptr(), g, assumed_align=16),
            make_ptr(cutlass.BFloat16, out.data_ptr(), g, assumed_align=16),
            make_ptr(cutlass.Float32, self.part.data_ptr(), g, assumed_align=16),
            make_ptr(cutlass.Int32, self.tickets.data_ptr(), g, assumed_align=16),
        )

    def run(self, w, ws, xq, xs, out):
        self.fn(*self._ptrs(w, ws, xq, xs, out), self._stream(self.device))


_ENGINES: dict[tuple, _DenseEngine] = {}


def _dense_engine(n, k, rows, device) -> _DenseEngine:
    key = (n, k, rows, torch.device(device))
    engine = _ENGINES.get(key)
    if engine is None:
        engine = _ENGINES[key] = _DenseEngine(n, k, rows, device)
    return engine
