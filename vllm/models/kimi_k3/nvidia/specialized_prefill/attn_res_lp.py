# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 prefill AttnRes with precomputed block logits.

Every AttnRes call c mixes the completed block rows v_b and the prefix source
u with softmax weights over s = (v . w_c) * rsqrt(mean(v^2) + eps). For a
block row both factors are fixed once the block is written: its RMS factor
(stored by the writing call) and its dot with the static per-call query w_c.
So when block b is written, one tensor-core GEMM computes v_b . w_c for every
later call of the forward; the per-call kernel (attn_res_lp.cu) then reads n
scalars per token instead of reducing every block row again, and streams the
block rows as a plain weighted sum. Per call only the prefix source needs a
row reduction, so the per-source compute no longer grows with depth.

Exactness: the query is the stock kernel's, w_c = ftz(ftz(norm_w) * ftz(qk_w))
in FP32 (flush-to-zero, as --use_fast_math does there). Scaled by 2^32
(exact), it is an exact sum of BF16 terms (1-2 for BF16 weights, 3 for any
normal FP32; checked at plan build, never approximated), so the GEMM's
BF16 x BF16 products are exact and accumulate in FP32 like the stock kernel
(only the summation order differs); the kernel scales the logit back by
2^-32 (exact).

Lifecycle: the model calls :func:`begin` with the forward's block buffer at
the start of a prefill forward; every AttnRes call of that forward then goes
through :func:`attn_res` with its call index (layer l: 2 l pre-attention,
2 l + 1 post-attention; the final AttnRes: 2 * num_layers).
"""

import hashlib
import os
import threading

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.models.kimi_k3.nvidia.ops.stable_build import (
    SPIN_GUARD,
    spin_guard_bytes,
    spin_rank,
    spin_record_ptr,
    stable_copy,
)

from . import attn_res_dielocal, attn_res_mxfp8

logger = init_logger(__name__)

HIDDEN = 7168
MAX_BLOCKS = 8
MXFP8_BLOCK = 32
# Per-warp row ring depth and token groups per CTA, measured on the 93-layer
# F16K die-local call sequence (VR200): G3 d2/d3/d4 27.71/27.21/28.0 ms,
# G4 d2/d3 25.85/25.97 ms, G5 d2 28.2 and G6 d2 33.3 ms (register spills).
RING_DEPTH = 2
# Per-shape ring depth (93-layer F16K die-local sequence, per-call A/B of
# depth 2 vs 3): MXFP8-only outputs with >= 5 blocks run 2-3 % faster with a
# 3-row ring; every other shape 0-8 % slower.
RING_DEPTH_Q_DEEP = 3
Q_DEEP_MIN_BLOCKS = 5
GROUPS = 4
VEC = 8  # BF16 per lane chunk (4 warps per token group; 4 -> 8 warps)
# Floor of the tokens a group claims ahead of the one it processes (the
# kernel raises it to cover the ring's run-ahead in rows): the shortest
# run-ahead balances the call's end best (93-layer F16K die-local sequence:
# 3 -> 25.38 ms, 5 -> 25.63 ms).
CLAIM_AHEAD = 2
PDL = 1
_SRC = os.path.join(os.path.dirname(__file__), "attn_res_lp.cu")
_lock = threading.Lock()
_mod: dict[tuple, object] = {}


def _module(
    probe: int = 0,
    vec: int = VEC,
    groups: int = GROUPS,
    claim_ahead: int = CLAIM_AHEAD,
    pdl: int = 1,
):
    """``probe`` != 0 or ``pdl`` == 0: a bench-only build of the kernel."""
    key = (probe, vec, groups, claim_ahead, pdl)
    if key in _mod:
        return _mod[key]
    with _lock:
        if key not in _mod:
            from torch.utils.cpp_extension import load

            import vllm.envs as envs

            h = hashlib.sha256()
            with open(_SRC, "rb") as fh:
                h.update(fh.read())
            h.update(spin_guard_bytes())
            h.update(f"{torch.__version__} {torch.version.cuda}".encode())
            h.update(f"{probe} {vec} {groups} {claim_ahead} {pdl}".encode())
            tag = h.hexdigest()[:16]
            build = os.path.join(
                envs.VLLM_CACHE_ROOT, "k31pf_attn_res_lp", tag + "-s"
            )
            os.makedirs(build, exist_ok=True)
            name = os.path.basename(_SRC)
            src = stable_copy(build, os.path.dirname(_SRC), [name], extra=[SPIN_GUARD])
            _mod[key] = (
                load(
                    name=f"k31pf_attn_res_lp_{tag}",
                    sources=[os.path.join(src, name)],
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=[
                        "-O3",
                        f"-DK31LP_PROBE={probe}",
                        f"-DK31LP_VEC={vec}",
                        f"-DK31LP_G={groups}",
                        f"-DK31LP_QA={claim_ahead}",
                        f"-DK31LP_PDL={pdl}",
                        "--use_fast_math",
                        "--expt-relaxed-constexpr",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-gencode",
                        "arch=compute_100f,code=sm_100f",
                    ],
                    build_directory=build,
                    verbose=False,
                )
            )
            # k31_spin_guard.cuh
            _mod[key].set_spin_rank(spin_rank(), spin_record_ptr())
    return _mod[key]


use_for = attn_res_mxfp8.use_for

# The block-logit GEMMs run on the query scaled by 2^QUERY_SCALE_EXP (exact):
# the real checkpoint's normal queries reach down to 2^-126, whose BF16 low
# term would be subnormal (inexact); scaled, every query of hf-1d82032 is an
# exact sum of 1-2 BF16 terms. The kernel scales the logit back (exact).
QUERY_SCALE_EXP = 32
QUERY_SCALE = 2.0**QUERY_SCALE_EXP
MAX_TERMS = 3


def _ftz(x: torch.Tensor) -> torch.Tensor:
    """Flush FP32 subnormals to zero: the stock AttnRes kernel builds its query
    q = rms_w * res_w under --use_fast_math (csrc/kimi_k3/attn_res_kernel.cu),
    so subnormal weights and subnormal products are zero there."""
    return torch.where(x.abs() < torch.finfo(torch.float32).tiny, torch.zeros_like(x), x)


def _bf16_terms(x: torch.Tensor) -> list[torch.Tensor]:
    """BF16 tensors whose FP32 sum is exactly ``x`` (FP32), fewest first: one
    to three terms (8 significant bits each) for any normal FP32 value."""
    terms = []
    r = x
    while True:
        t = r.bfloat16()
        terms.append(t)
        r = r - t.float()
        if not bool(r.any()):
            return terms
        if len(terms) == MAX_TERMS:
            bad = int((r != 0).sum())
            raise ValueError(
                f"attn_res_lp: {bad} AttnRes query values are not an exact sum of "
                f"{MAX_TERMS} BF16 terms"
            )


class Plan:
    """Static per-model tables: the FP32 query of every AttnRes call (stock
    flush-to-zero semantics) and, per block, the BF16 term rows of the scaled
    query of the calls after its write (``terms[k][c]``: term k of call c)."""

    def __init__(
        self,
        weights: list[tuple[torch.Tensor, torch.Tensor]],
        eps: list[float],
        block_size: int,
        num_blocks: int,
    ):
        if len(set(eps)) != 1:
            raise ValueError(f"attn_res_lp: AttnRes eps differ across calls {set(eps)}")
        if num_blocks > MAX_BLOCKS:
            raise ValueError(f"attn_res_lp: {num_blocks} blocks > {MAX_BLOCKS}")
        self.eps = eps[0]
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_calls = len(weights)
        w32 = torch.stack(
            [_ftz(_ftz(nw.float()) * _ftz(qw.float())) for nw, qw in weights]
        ).contiguous()
        self.w32 = w32
        self.terms = _bf16_terms(w32 * QUERY_SCALE)
        # Block b is written by call 2 * block_size * b (pre-attention of
        # layer block_size * b) and read by every later call.
        self.first_call = [2 * block_size * b + 1 for b in range(num_blocks)]
        self.wcat = []
        for b in range(num_blocks):
            c0 = self.first_call[b]
            self.wcat.append(torch.cat([t[c0:] for t in self.terms]).contiguous())

    def calls_after(self, b: int) -> int:
        return self.num_calls - self.first_call[b]


class _Forward:
    def __init__(self, plan: Plan, blocks: torch.Tensor):
        self.plan = plan
        self.blocks = blocks
        self.ptr = blocks.data_ptr()
        self.tokens = blocks.shape[0]
        self.rstd = torch.empty(
            (MAX_BLOCKS, self.tokens), dtype=torch.float32, device=blocks.device
        )
        self.logits: list[torch.Tensor | None] = [None] * plan.num_blocks


# Planned forwards in flight, keyed by their block buffer (one per forward;
# a forward is released by its final AttnRes call).
_states: dict[int, _Forward] = {}
_claims: dict[int, torch.Tensor] = {}


def _schedule_args(blocks: torch.Tensor):
    """(sm_die, claim): die-aware token claims when ``blocks`` is the
    die-local block buffer, else plain claims (sm_die None)."""
    sm_die, claim = attn_res_dielocal.kernel_args(blocks)
    if claim is not None:
        return sm_die, claim
    dev = blocks.device.index
    claim = _claims.get(dev)
    if claim is None:
        claim = _claims[dev] = torch.zeros(4, dtype=torch.int32, device=blocks.device)
    return None, claim


def begin(plan: Plan, blocks: torch.Tensor) -> None:
    """Start a prefill forward whose AttnRes block buffer is ``blocks``."""
    while len(_states) >= 2:  # forwards that never reached their final call
        del _states[next(iter(_states))]
    _states[blocks.data_ptr()] = _Forward(plan, blocks)
    logger.info_once(
        "Kimi K3.1 prefill AttnRes: precomputed block logits (attn_res_lp, "
        "%d calls, %d blocks, die-aware claims: %s)",
        plan.num_calls, plan.num_blocks,
        attn_res_dielocal.kernel_args(blocks)[0] is not None,
    )


def active(blocks: torch.Tensor, prefix: torch.Tensor) -> bool:
    """A planned prefill forward owns this AttnRes call's block buffer."""
    st = _states.get(blocks.data_ptr())
    return st is not None and st.tokens == prefix.shape[0]


def _forward_for(blocks: torch.Tensor, rows: int) -> _Forward:
    st = _states.get(blocks.data_ptr())
    if st is None or st.tokens != rows:
        raise RuntimeError(
            "attn_res_lp: AttnRes call outside the prefill forward it was "
            "planned for (begin() not called with this block buffer)"
        )
    return st


def attn_res(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    call: int,
    num_blocks: int,
    block_write_idx: int,
    output_norm_weight: torch.Tensor | None,
    output_norm_eps: float,
    bf16_out: bool,
    quant_out: bool,
    prefix_out: torch.Tensor | None = None,
    local_out: bool = True,
    _probe: int = 0,
    _vec: int | None = None,
    _groups: int | None = None,
    _stamps: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, QuantizedActivation | None]:
    """AttnRes call ``call`` of the current prefill forward. Returns the BF16
    output (``bf16_out``) and the MXFP8 activation (``quant_out``)."""
    rows = prefix.shape[0]
    st = _forward_for(blocks, rows)
    plan = st.plan
    lp = []
    for i in range(num_blocks):
        lg = st.logits[i]
        if lg is None:
            raise RuntimeError(f"attn_res_lp: block {i} read before its write")
        row = call - plan.first_call[i]
        n_after = plan.calls_after(i)
        if not 0 <= row < n_after:
            raise RuntimeError(f"attn_res_lp: call {call} has no logits of block {i}")
        base = lg.data_ptr()
        lp += [base + (k * n_after + row) * rows * 4 for k in range(len(plan.terms))]
    lptr = torch.tensor(lp or [0], dtype=torch.int64)
    # local_out: outputs consumed within the layer (every call but the final
    # one, whose output leaves the model) go to die-local rotating buffers,
    # written by the SMs of their token's die.
    local_out = local_out and call != plan.num_calls - 1
    out = q = sf = None
    if bf16_out:
        if local_out:
            out = attn_res_dielocal.out_buffer(blocks, "bf16", rows)
        if out is None:
            out = torch.empty_like(prefix)
    if quant_out:
        if local_out:
            q = attn_res_dielocal.out_buffer(blocks, "fp8", rows)
        if q is None:
            q = torch.empty((rows, HIDDEN), dtype=torch.float8_e4m3fn, device=prefix.device)
        sf = torch.empty(
            (max(rows, 1) + 127) // 128 * 128 * (HIDDEN // MXFP8_BLOCK),
            dtype=torch.uint8,
            device=prefix.device,
        )
        if rows % 128:
            # Scale rows past `rows` (last 128-row tile) are zero, as
            # mxfp8_quantize writes them; the kernel fills the real rows.
            sf.view(-1, 128 * (HIDDEN // MXFP8_BLOCK))[-1].zero_()
    _module(
        _probe,
        VEC if _vec is None else _vec,
        GROUPS if _groups is None else _groups,
        CLAIM_AHEAD,
        PDL,
    ).attn_res_lp(
        prefix,
        prefix_out,
        delta,
        blocks,
        plan.w32[call],
        output_norm_weight,
        lptr,
        len(plan.terms),
        1.0 / QUERY_SCALE,
        st.rstd,
        num_blocks,
        block_write_idx,
        float(plan.eps),
        float(output_norm_eps),
        out,
        None if q is None else q.view(torch.uint8),
        sf,
        RING_DEPTH_Q_DEEP
        if quant_out and not bf16_out and num_blocks >= Q_DEEP_MIN_BLOCKS
        else RING_DEPTH,
        *_schedule_args(blocks),
        _stamps,
    )
    if call == plan.num_calls - 1:
        # The final AttnRes is the forward's last reader of the plan state.
        del _states[st.ptr]
    elif block_write_idx >= 0 and plan.calls_after(block_write_idx) > 0:
        # 2^32 v_b . w_c for every later call c: [terms * calls_after, rows]
        # FP32 (the rows of term 0, then of term 1, ...).
        st.logits[block_write_idx] = torch.mm(
            plan.wcat[block_write_idx],
            blocks[:, block_write_idx].t(),
            out_dtype=torch.float32,
        )
    qa = None
    if quant_out:
        qa = QuantizedActivation(
            data=q,
            scale=sf,
            orig_dtype=prefix.dtype,
            orig_shape=torch.Size((rows, HIDDEN)),
            quant_key=kMxfp8Dynamic,
        )
    return out, qa
