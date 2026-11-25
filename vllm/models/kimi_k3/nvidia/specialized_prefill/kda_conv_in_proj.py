# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill KDA in_proj with the short conv in the GEMM epilogue.

On a prefill worker with an SM107 GPU, the online-MXFP8 KDA in_proj
(q|k|v|g|f_a|beta rows) and FP32 conv weights, prefill-only batches of at
least ``MIN_ROWS`` rows run the q|k|v rows through one sm_107 GEMM whose
epilogue applies the width-4 causal conv + SiLU and updates the conv state
(``sm107_gemm.kda_conv_gemm``: bit for bit the in_proj GEMM followed by
``ops/kda_conv.qkv_conv``), so neither the BF16 q|k|v GEMM output nor the
separate conv pass exist. The g|f_a|beta rows keep a cuBLASLt MXFP8 GEMM on a
view of the same weight. Batches with decode or speculative tokens, and
smaller batches, keep the stock in_proj + conv.
"""

from typing import NamedTuple

import torch

from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

MIN_ROWS = 2048
#: 512x256 B-reuse tile, 2x4 cluster (A multicast across N), drain epilogue.
TACTIC = ((512, 256), (2, 4), 256, 4, "m", 0, 0, "")
#: Per-shape tactics, keyed by the q|k|v width N = 3 P of a TP-sharded layer,
#: where TACTIC's cluster columns (tile N x cluster N = 1024) do not tile N.
#: TP8 (P 1536, N 4608): 4x1 cluster, "n" raster (pads M, which the epilogue
#: bounds). Measured on VR200 (pgemm_conv_bench --mode conv, K 7168, 8 equal
#: sequences, fused GEMM + fix-up vs cuBLASLt + qkv_conv): 32768 rows 258.5 vs
#: 277.4 us, 16384 rows 142.9 vs 152.3 us; 2x2 "m" 268.5 / 148.6, 2x1 "m"
#: 274.8 / 152.8, 4x1 "m" 282.9 / 158.0, 2x2 "n" 262.1 / 145.2.
SHAPE_TACTICS = {
    4608: ((512, 256), (4, 1), 256, 8, "n", 0, 0, ""),
}


def tactic_for(n: int):
    """The tactic of an ``n``-column q|k|v problem (before ``shape_tactic``)."""
    return SHAPE_TACTICS.get(n, TACTIC)


def _qkv_rows(lin) -> torch.Tensor:
    """The online-MXFP8 in_proj weight as K-major [N, K] FP8 rows."""
    from vllm.models.kimi_k3.nvidia.mxfp8 import _kernel_layout

    w = lin.weight.data
    return w.t() if _kernel_layout(lin) == "kn" else w


def unsupported_reason(layer) -> str | None:
    """None when the layer can run the fused GEMM: SM107, online-MXFP8 in_proj
    with K-major [N, K] rows (either kernel layout), FP32 width-4 conv
    weights, 256-multiple planes that the shape's tactic (``tactic_for``)
    clusters tile exactly (``kda_conv_gemm.shape_tactic``); else why not.
    Raises ValueError when only the tactic is missing for the shape."""
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability((10, 7)):
        return "not SM107"
    from vllm.models.kimi_k3.nvidia.mxfp8 import _kernel_layout, is_mxfp8_linear

    lin = layer.in_proj_qkvgfab
    if not is_mxfp8_linear(lin) or _kernel_layout(lin) is None:
        return "in_proj is not an online-MXFP8 [N, K] / [K, N] linear"
    if not _qkv_rows(lin).is_contiguous():
        return "in_proj rows are not K-major"
    w = layer.conv1d.weight
    if w.dtype != torch.float32 or w.shape[-1] != 4:
        return f"conv weight {w.dtype} {tuple(w.shape)} (FP32 width 4 only)"
    if layer.local_projection_size % 256 or lin.input_size_per_partition % 128:
        return "plane / K size"
    from .sm107_gemm.kda_conv_gemm import shape_tactic

    # A supported layer whose q|k|v width no tactic tiles is a missing
    # SHAPE_TACTICS entry, not a capability limit: raise at load (no silent
    # fallback to the stock in_proj + conv).
    n = 3 * layer.local_projection_size
    try:
        shape_tactic(n, tactic_for(n))
    except ValueError as e:
        raise ValueError(
            f"prefill KDA conv GEMM: no tactic tiles q|k|v width {n} "
            f"(add a SHAPE_TACTICS entry): {e}"
        ) from e
    return None


class PrefillKdaConvInProj:
    """Fused q|k|v in_proj + conv of one KDA layer (``unsupported_reason`` None)."""

    def __init__(self, layer) -> None:
        self.layer = layer
        self.p = layer.local_projection_size
        self._gemm = None

    def prepare(self, max_num_seqs: int) -> None:
        """After weight loading, before any forward: views of the q|k|v and
        g|f_a|beta rows and their F8_128x4 scales; compile the fused GEMM and
        every conv fix-up variant of a step of up to ``max_num_seqs``
        sequences."""
        if self._gemm is not None:
            return
        from .sm107_gemm.kda_conv_gemm import KdaConvGemm

        lin = self.layer.in_proj_qkvgfab
        w = _qkv_rows(lin)
        n, k = w.shape
        nq = 3 * self.p
        # F8_128x4 scales are 128-row blocks of K / 32 bytes: q|k|v (a whole
        # number of blocks) is a prefix of the flat scale tensor.
        s = lin.weight_scale.data.view(torch.uint8).reshape(-1)
        cut = nq * (k // 32)
        self.w_rest, self.s_rest = w[nq:], s[cut:]
        # g | f_a | beta columns of the rest GEMM; TP-sharded in_proj rows may
        # end in padding (layer.in_proj_padding, e.g. 4 rows at TP8), which the
        # caller's split must not see.
        self.rest_cols = (
            self.layer.local_projection_size
            + self.layer.head_dim
            + self.layer.local_num_heads
        )
        cw = self.layer.conv1d.weight
        self.conv_w = cw.view(cw.shape[0], cw.shape[-1])
        self._gemm = KdaConvGemm(
            w[:nq], s[:cut], self.conv_w, self.p, tactic=tactic_for(nq)
        )
        dev = w.device
        rows = MIN_ROWS
        x = torch.zeros((rows, k), dtype=torch.float8_e4m3fn, device=dev)
        sx = torch.full((rows * k // 32,), 127, dtype=torch.uint8, device=dev)
        qsl = torch.tensor([0, rows], dtype=torch.int32, device=dev)
        lines = torch.zeros(1, dtype=torch.int32, device=dev)
        init = torch.zeros(1, dtype=torch.bool, device=dev)
        st = torch.zeros((1, nq, 3), dtype=torch.bfloat16, device=dev)
        self._gemm(x, sx, st, qsl, lines, init, null_block_id=NULL_BLOCK_ID)
        # Overlapped SP all-gather (ops/sp_all_gather.py): the grid of the "ag"
        # form leaves SMs idle for the pushers; compile it now (every tile local:
        # no waits, no pushers) and tell the gather.
        idle = overlap_idle_sms(dev, self._gemm.tactic)
        if idle > 0:
            from vllm.models.kimi_k3.nvidia.ops import sp_all_gather as sp_ag

            from .sm107_gemm.kda_conv_gemm import ag_tactic, cluster_rows

            ctr = torch.zeros(64, dtype=torch.int32, device=dev)
            self._gemm(
                x, sx, st, qsl, lines, init, null_block_id=NULL_BLOCK_ID,
                ag=_LocalAg(ctr.data_ptr(), rows // cluster_rows(ag_tactic(self._gemm.tactic))),
            )  # fmt: skip
            torch.cuda.synchronize(dev)
            sp_ag.enable_overlap(idle, cluster_rows(ag_tactic(self._gemm.tactic)))
        from .sm107_gemm import conv_fixup

        conv_fixup.warm(
            self.p,
            self.conv_w,
            self.layer.get_state_dtype()[0],
            max_num_seqs,
            NULL_BLOCK_ID,
        )

    def fused(self, qa, ag=None):
        """``((3, T, P) conv + SiLU q|k|v planes, (T, rest) g|f_a|beta,
        checkpoint rows)`` for an eager prefill-only batch of at least
        ``MIN_ROWS`` rows, else None (stock in_proj + conv).

        A step that exports prefill checkpoints (prefix-cache snapshots) stores
        the raw q|k|v rows before each checkpoint offset as its conv state. The
        fused GEMM never materializes the raw rows, so its epilogue saves those
        windows (``CheckpointRows``); the exporter reads them in place of the
        in_proj output, bit for bit the same rows."""
        gemm = self._gemm
        assert gemm is not None, "PrefillKdaConvInProj.prepare() not run"
        rows = qa.data.shape[0]
        if rows < MIN_ROWS or torch.cuda.is_current_stream_capturing():
            # Not inside a CUDA graph: the GEMM's sequence count (grid and
            # epilogue scan), the has_initial_state tensor and the conv state
            # rows come from this step's metadata, which a graph would freeze
            # at capture (a piecewise graph replayed for a later prefill read
            # and wrote conv state rows of the capture batch: illegal address
            # on the first prefix-hit prefill). The stock in_proj is captured
            # instead and the conv runs in the eager KDA break (_forward).
            return None
        md = get_forward_context().attn_metadata
        if not isinstance(md, dict):
            return None
        m = md.get(self.layer.prefix)
        if (
            m is None
            or m.num_prefills == 0
            or m.num_decodes != 0
            or m.num_spec_decodes != 0
        ):
            return None
        qsl = m.non_spec_query_start_loc
        nseq = qsl.numel() - 1
        ckpt = m.checkpoint
        if ckpt is not None and (
            not gemm.supports_checkpoint_halo
            or ckpt.checkpoint_offsets.numel() != nseq
            or ckpt.checkpoint_offsets.dtype != torch.int32
            or not ckpt.checkpoint_offsets.is_contiguous()
        ):
            # The exporter needs the raw rows the fused GEMM does not keep.
            return None
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )

        conv_state = self.layer.kv_cache[0]
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        if conv_state.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"KDA conv GEMM: conv state dtype {conv_state.dtype}")
        x_scale = qa.scale.view(torch.uint8).reshape(-1)
        ckpt_rows = None
        halo = None
        if ckpt is not None:
            halo = torch.empty(
                (nseq, 3, 3 * self.p), dtype=torch.bfloat16, device=qa.data.device
            )
            ckpt_rows = CheckpointRows(halo.view(3 * nseq, 3 * self.p), _halo_qsl(ckpt))
        planes, _ = gemm(
            qa.data, x_scale, conv_state, qsl,
            m.non_spec_state_indices_tensor, m.has_initial_state,
            null_block_id=NULL_BLOCK_ID,
            checkpoint_offsets=ckpt.checkpoint_offsets if ckpt is not None else None,
            halo=halo,
            ag=ag,
        )  # fmt: skip
        rest = torch._scaled_mm(
            qa.data,
            self.w_rest.t(),
            scale_a=qa.scale.view(torch.float8_e8m0fnu),
            scale_b=self.s_rest.view(torch.float8_e8m0fnu),
            out_dtype=torch.bfloat16,
        )
        if rest.shape[-1] != self.rest_cols:
            rest = rest[:, : self.rest_cols]  # drop the in_proj padding rows
        return planes, rest, ckpt_rows


class _LocalAg(NamedTuple):
    """An "ag" GEMM launch whose rows are all local (compile / warm-up)."""

    ctr_ptr: int
    hi: int
    target: int = 0
    rot: int = 0
    lo: int = 0

    def launch_pushers(self) -> None:
        pass


def overlap_idle_sms(device, tactic) -> int:
    """SMs the "ag" form of the Q|K|V GEMM leaves idle: whole clusters of its
    shape fill only part of the GPU (TP8 tactic, 4x1 clusters: 184 of 212);
    the pusher grid of the overlapped SP all-gather runs there. 0 when the
    grid covers every SM (then the gather stays stock)."""
    import cutlass.utils as cutlass_utils

    from .sm107_gemm.kda_conv_gemm import ag_tactic

    cl = ag_tactic(tactic)[1]
    nsm = torch.cuda.get_device_properties(device).multi_processor_count
    mac = cutlass_utils.HardwareInfo().get_max_active_clusters(cl[0] * cl[1])
    return max(0, nsm - mac * cl[0] * cl[1])


class CheckpointRows(NamedTuple):
    """The pre-conv rows a checkpoint exporter reads, for the fused in_proj:
    ``raw`` holds sequence ``s``'s three rows before its checkpoint offset at
    rows ``3 s .. 3 s + 2``; ``qsl`` is the per-sequence start the exporter
    adds the offset to (``3 s + 3 - offset``), so it reads exactly those."""

    raw: torch.Tensor
    qsl: torch.Tensor


#: (checkpoint metadata, its halo starts): one per step, shared by the layers.
_HALO_QSL: list = [None, None]


def _halo_qsl(ckpt) -> torch.Tensor:
    if _HALO_QSL[0] is not ckpt:
        offsets = ckpt.checkpoint_offsets
        n = offsets.numel()
        if ckpt.offsets is not None:
            starts = async_tensor_h2d(
                [3 * s + 3 - o for s, o in enumerate(ckpt.offsets)],
                dtype=torch.int32,
                device=offsets.device,
            )
        else:
            starts = (
                torch.arange(3, 3 * n + 3, 3, dtype=torch.int32, device=offsets.device)
                - offsets
            )
        _HALO_QSL[0], _HALO_QSL[1] = ckpt, starts
    return _HALO_QSL[1]


def contiguous_rows(x: torch.Tensor) -> torch.Tensor:
    """``x.contiguous()`` of a row-strided 2-D slice, copied as 16-byte
    elements when rows and strides allow it (byte-identical; the generic
    strided copy moves 2-byte elements: 7.7 vs ~1.5 us for the KDA beta
    columns at F16K)."""
    if x.is_contiguous():
        return x
    nbytes = x.size(-1) * x.element_size()
    if (
        x.dim() == 2
        and x.stride(-1) == 1
        and nbytes % 16 == 0
        and (x.stride(0) * x.element_size()) % 16 == 0
        and x.data_ptr() % 16 == 0
    ):
        return x.view(torch.complex128).contiguous().view(x.dtype)
    return x.contiguous()

