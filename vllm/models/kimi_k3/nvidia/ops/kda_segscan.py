# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Segment-parallel FlashKDA prefill for long sequences.

FlashKDA's recurrence kernel (K2) is serial over 16-token chunks: one CTA pair per
head walks the whole sequence, so a 16k-token prefill at 12 local heads keeps 24
CTAs busy for ~0.5 ms per layer while the rest of the GPU idles. The state update
is affine in the state. In FlashKDA's [V, K] state layout each row evolves as
x <- x A_c^T + b_c, so over a segment p of a sequence

    X_end = X_start @ M_p + C_p,
    M_p = K2(v = 0, X_start = I),   C_p = K2(v, X_start = 0).

Every long sequence is cut into segments (multiples of the 16-token chunk, so K1's
tiles are unchanged) that run as pseudo-sequences of one varlen launch:

  K1  once over the segmented cu_seqlens (token-parallel; identical tiles);
  A   K2 twice: C_p (real v, zero initial state) and M_p (v = 0, identity);
  B   a per-sequence fp32 scan X_{p+1} = X_p @ M_p + C_p from the real initial state;
  C   K2 from X_p over every segment -> outputs, final state, checkpoint.

All states between phases are fp32: the split build's StateFP32 path reads and
writes the fp32 register state directly (FlashKDA staged it through bf16), so
M_p, C_p and X_p are fp32 partials. The served state dtype is restored at the
end. It is the same arithmetic per chunk as one long K2
call, reassociated at segment boundaries (fp32). M_p is a product of contractions
(singular values < 1), so boundary rounding is damped, not amplified.

The K1 / K2 split is FlashKDA itself with a `mode` argument
(ops/flashkda_split, vllm-project/FlashKDA 17a037d, MIT), JIT-built once per
source hash into VLLM_CACHE_ROOT and reused across restarts.
Policy (``wanted``): on wherever prefills run locally; a P/D decode-only instance
(KV consumer, not producer) prefills only to recompute a failed KV load and keeps the
plain FlashKDA prefill (no segment buffers or extension build there).
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d

# Segment only steps whose longest sequence has at least this many tokens.
_MIN_TOKENS = 6144
# Pseudo-sequences per launch: one wave of V-split CTA pairs (2 x 12 heads x 8
# = 192 of 212 SMs on Rubin). Segmenting triples the K2 work (two phase-A
# passes + the output pass), so it pays only when the longest chain shrinks by
# well over 3x: the plan requires the longest segment to be at most a fifth of
# the longest sequence.
_MAX_PSEUDO = 8
_MIN_SEG = 256
_CHUNK = 16
_D = 128
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flashkda_split")


def wanted(vllm_config) -> bool:
    """Whether the instance builds the segment-parallel prefill (module docstring)."""
    kv_transfer = getattr(vllm_config, "kv_transfer_config", None)
    decode_only = (
        kv_transfer is not None
        and kv_transfer.is_kv_consumer
        and not kv_transfer.is_kv_producer
    )
    return not decode_only


# --- the K1 / K2-split FlashKDA build ----------------------------------------

_lock = threading.Lock()
_loaded = False


def _sources() -> list[str]:
    out = []
    for root, _, files in os.walk(_SRC):
        out += [os.path.join(root, f) for f in files]
    return sorted(out)


def load_extension() -> None:
    """Build (first use per source hash) and load torch.ops.k3_flashkda_split."""
    global _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        import flashinfer
        from torch.utils.cpp_extension import load

        import vllm.envs as envs

        cutlass = os.path.join(
            os.path.dirname(flashinfer.__file__), "data", "cutlass", "include"
        )
        h = hashlib.sha256()
        for f in _sources():
            h.update(os.path.relpath(f, _SRC).encode())
            with open(f, "rb") as fh:
                h.update(fh.read())
        with open(os.path.join(cutlass, "cutlass", "version.h"), "rb") as fh:
            h.update(fh.read())
        h.update(f"{torch.__version__} {torch.version.cuda}".encode())
        build = os.path.join(
            envs.VLLM_CACHE_ROOT, "k3_flashkda_split", h.hexdigest()[:16]
        )
        os.makedirs(build, exist_ok=True)
        flags = [
            "-O3",
            "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
            "-DUSE_CUDA",
            "-UPy_LIMITED_API",
        ]
        cuda = flags + [
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-gencode",
            "arch=compute_100f,code=sm_100f",
            "--threads",
            "8",
        ]
        csrc = os.path.join(_SRC, "csrc")
        load(
            name="k3_flashkda_split",
            sources=[
                os.path.join(csrc, "binding.cpp"),
                os.path.join(csrc, "flash_kda.cpp"),
                os.path.join(csrc, "smxx", "fwd_launch.cu"),
            ],
            extra_include_paths=[csrc, cutlass],
            extra_cflags=flags,
            extra_cuda_cflags=cuda,
            build_directory=build,
            is_python_module=False,
            verbose=False,
        )
        _loaded = True


def workspace_size(tokens: int, heads: int, sequences: int) -> int:
    """FlashKDA's workspace bound (csrc/flash_kda.cpp get_workspace_size)."""
    total_tiles = (tokens + _CHUNK - 1) // _CHUNK + sequences
    per_tile = 3 * _CHUNK * _D * 2 + _D * 4 + 2 * _CHUNK * _CHUNK * 2
    return heads * total_tiles * per_tile


# --- persistent buffers ------------------------------------------------------


class SegBuffers:
    """fp32 phase states for up to _MAX_PSEUDO pseudo-sequences, their identity
    initial states, and a zero v; allocated once per (tokens, heads, device) at
    layer init. ~5 x 6.3 MB + 6.3 MB + 50 MB at 16384 tokens and 12 heads. The
    FlashKDA workspace is the layer's own (sized for max_num_seqs sequences)
    and phase A writes its discarded outputs into the output buffer, which
    phase C then overwrites."""

    def __init__(self, tokens: int, heads: int, device):
        f32 = torch.float32
        n = _MAX_PSEUDO
        self.n = n
        shape = (n, heads, _D, _D)
        self.C = torch.empty(shape, dtype=f32, device=device)
        self.M = torch.empty(shape, dtype=f32, device=device)
        self.X = torch.empty(shape, dtype=f32, device=device)
        self.F = torch.empty(shape, dtype=f32, device=device)
        self.K = torch.empty(
            shape, dtype=f32, device=device
        )  # in-kernel checkpoint states
        self.eye = torch.eye(_D, dtype=f32, device=device).expand(shape).contiguous()
        self.zero_v = torch.zeros(
            (1, tokens, heads, _D), dtype=torch.bfloat16, device=device
        )
        self.tokens = tokens


_BUFFERS: dict[tuple, SegBuffers] = {}


def init_buffers(tokens: int, heads: int, device) -> SegBuffers:
    # Build / load the extension at model init (cached on disk), not at the first
    # long prefill.
    load_extension()
    key = (tokens, heads, str(torch.device(device)))
    b = _BUFFERS.get(key)
    if b is None:
        b = _BUFFERS[key] = SegBuffers(tokens, heads, device)
    return b


# --- plan --------------------------------------------------------------------


@dataclass
class SegPlan:
    num_seqs: int
    num_pseudo: int
    cu: torch.Tensor  # [P + 1] int32, pseudo-sequence cumulative tokens
    first: torch.Tensor  # [N] int32, first pseudo index of each sequence
    count: torch.Tensor  # [N] int32, pseudo count of each sequence
    last: torch.Tensor  # [N] int64, last pseudo index of each sequence
    # Checkpoint states (sequence rows of the [N, H, D, D] output), by source:
    # a segment boundary (the scan's X), the sequence end (F), or the in-kernel
    # export of an unsplit pseudo-sequence (K2's checkpoint at a local offset).
    ck_x: tuple[torch.Tensor, torch.Tensor] | None  # (sequence rows, pseudo rows of X)
    ck_f: tuple[torch.Tensor, torch.Tensor] | None  # (sequence rows, pseudo rows of F)
    ck_k: tuple[torch.Tensor, torch.Tensor] | None  # (sequence rows, pseudo rows of K)
    kernel_ckpt_offsets: torch.Tensor | None  # [P] int32 local offsets (0 = none)


_PLANS: dict[tuple, SegPlan | None] = {}


def _segment_counts(lens: tuple[int, ...]) -> list[int] | None:
    """Pseudo-sequences per sequence: the extra segments go, one at a time, to
    the sequence with the longest current segment. None if it does not pay."""
    n = len(lens)
    if n == 0 or n >= _MAX_PSEUDO or max(lens) < _MIN_TOKENS:
        return None
    parts = [1] * n
    for _ in range(_MAX_PSEUDO - n):
        i = max(range(n), key=lambda j: lens[j] / parts[j])
        if lens[i] / (parts[i] + 1) < _MIN_SEG:
            break
        parts[i] += 1
    return parts


def _split(a: int, b: int, k: int) -> list[int]:
    """Boundaries a = x0 < x1 < ... = b in at most k chunk-aligned pieces."""
    if b <= a:
        return [a, b]
    piece = max(_CHUNK, -(-(b - a) // k // _CHUNK) * _CHUNK)
    out = list(range(a, b, piece)) + [b]
    return out


def make_plan(
    query_start_loc_cpu: torch.Tensor,
    heads: int,
    device,
    checkpoint_offsets: list[int] | None = None,
) -> SegPlan | None:
    """Segment plan for one step's prefill sequences, or None to run FlashKDA as
    one launch. Host-only (CPU query lengths, host checkpoint offsets); cached.

    A sequence's checkpoint offset becomes a segment boundary whenever the
    sequence is split, so its checkpoint state is the scan's X there (no
    in-kernel export); at the sequence end it is the final state."""
    lens = tuple(int(x) for x in query_start_loc_cpu.diff().tolist())
    offs = (
        tuple(int(x) for x in checkpoint_offsets)
        if checkpoint_offsets
        else (0,) * len(lens)
    )
    key = (lens, offs, heads, str(device))
    if key in _PLANS:
        return _PLANS[key]
    plan = None
    parts = _segment_counts(lens) if len(offs) == len(lens) else None
    if parts is not None:
        bounds_all = []
        for i, n in enumerate(lens):
            c = offs[i]
            if 0 < c < n and parts[i] >= 2:
                k1 = min(max(1, round(parts[i] * c / n)), parts[i] - 1)
                bnd = _split(0, c, k1)[:-1] + _split(c, n, parts[i] - k1)
            else:
                bnd = _split(0, n, parts[i])
            bounds_all.append(bnd)
        longest_seg = max(
            max(b[j + 1] - b[j] for j in range(len(b) - 1)) for b in bounds_all
        )
        num_pseudo = sum(len(b) - 1 for b in bounds_all)
        # three K2 passes over chains a fifth as long (or less), in one wave;
        # at 4x a mixed batch (12k + 3k + 1.4k) gained only 1%
        if 5 * longest_seg <= max(lens) and num_pseudo <= _MAX_PSEUDO:
            ck_x: tuple[list[int], list[int]]
            ck_f: tuple[list[int], list[int]]
            ck_k: tuple[list[int], list[int]]
            k_off: list[int]
            cu, first, count, ck_x, ck_f, ck_k, k_off = (
                [0],
                [],
                [],
                ([], []),
                ([], []),
                ([], []),
                [],
            )
            for i, (n, bnd) in enumerate(zip(lens, bounds_all)):
                first.append(len(k_off))
                count.append(len(bnd) - 1)
                c = offs[i]
                for j in range(len(bnd) - 1):
                    p = len(k_off)
                    cu.append(cu[-1] + bnd[j + 1] - bnd[j])
                    local = 0
                    if 0 < c < n and bnd[j] == c:
                        ck_x[0].append(i), ck_x[1].append(p)  # type: ignore[func-returns-value]
                    elif 0 < c < n and bnd[j] < c < bnd[j + 1]:
                        local = c - bnd[j]  # unsplit sequence: in-kernel export
                        ck_k[0].append(i), ck_k[1].append(p)  # type: ignore[func-returns-value]
                    k_off.append(local)
                if c > 0 and c == n:
                    ck_f[0].append(i), ck_f[1].append(first[-1] + count[-1] - 1)  # type: ignore[func-returns-value]
            last = [f + k - 1 for f, k in zip(first, count)]
            h2d = lambda x, dt: async_tensor_h2d(x, dtype=dt, device=device)  # noqa: E731
            pair = (
                lambda x: (h2d(x[0], torch.int64), h2d(x[1], torch.int64))
                if x[0]
                else None
            )  # noqa: E731
            plan = SegPlan(
                num_seqs=len(lens),
                num_pseudo=len(k_off),
                cu=h2d(cu, torch.int32),
                first=h2d(first, torch.int32),
                count=h2d(count, torch.int32),
                last=h2d(last, torch.int64),
                ck_x=pair(ck_x),
                ck_f=pair(ck_f),
                ck_k=pair(ck_k),
                kernel_ckpt_offsets=h2d(k_off, torch.int32) if ck_k[0] else None,
            )
    if len(_PLANS) > 512:
        _PLANS.clear()
    _PLANS[key] = plan
    return plan


# --- phase B: per-sequence fp32 scan -----------------------------------------


@triton.jit
def _chain_scan_kernel(
    init_ptr,  # [N, H, D, D] initial states (state dtype)
    m_ptr,  # [P, H, D, D] fp32
    c_ptr,  # [P, H, D, D] fp32
    x_ptr,  # [P, H, D, D] fp32 out: each pseudo's initial state
    first_ptr,
    count_ptr,
    H: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
):
    # One program per (head, BV rows, sequence): rows of the [V, K] state are
    # independent. Each step reads the previous rows back from X (L2) in BK
    # column blocks: x_{p+1} = x_p @ M_p + C_p, fp32 (ieee) throughout.
    h = tl.program_id(0)
    rv = tl.program_id(1) * BV + tl.arange(0, BV)
    s = tl.program_id(2)
    first = tl.load(first_ptr + s).to(tl.int64)
    cnt = tl.load(count_ptr + s)
    rn = tl.arange(0, D)
    hdd = H * D * D
    row = h * D * D + rv[:, None] * D
    x0 = tl.load(init_ptr + s.to(tl.int64) * hdd + row + rn[None, :]).to(tl.float32)
    tl.store(x_ptr + first * hdd + row + rn[None, :], x0)
    for j in range(cnt - 1):
        tl.debug_barrier()
        p = first + j
        acc = tl.load(c_ptr + p * hdd + row + rn[None, :])
        mb = m_ptr + p * hdd + h * D * D
        for kb in range(0, D, BK):
            rk = kb + tl.arange(0, BK)
            xa = tl.load(x_ptr + p * hdd + row + rk[None, :])
            mm = tl.load(mb + rk[:, None] * D + rn[None, :])
            acc += tl.dot(xa, mm, input_precision="ieee")
        tl.store(x_ptr + (p + 1) * hdd + row + rn[None, :], acc)


def chain_scan(initial_state, M, C, X, plan: SegPlan) -> None:
    heads = initial_state.shape[1]
    assert initial_state.is_contiguous()
    _chain_scan_kernel[(heads, _D // 16, plan.num_seqs)](
        initial_state,
        M,
        C,
        X,
        plan.first,
        plan.count,
        H=heads,
        D=_D,
        BV=16,
        BK=64,
        num_warps=4,
        num_stages=2,
    )


# --- the segmented prefill ---------------------------------------------------


def _beta_view(beta: torch.Tensor) -> torch.Tensor:
    """[1, T, H] view of a contiguous [H, T] copy, so FlashKDA's internal
    transpose / transpose / contiguous is a no-op in all four launches."""
    bt = beta[0].t().contiguous()
    return bt.view(bt.shape[0], 1, bt.shape[1]).permute(1, 2, 0)


def segmented_prefill(
    *,
    plan: SegPlan,
    bufs: SegBuffers,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    # [N, H, D, D] state dtype (zeros where no prior state)
    initial_state: torch.Tensor,
    out: torch.Tensor,  # [1, T, H, D] contiguous
    final_state: torch.Tensor,  # [N, H, D, D] state dtype (written)
    # FlashKDA workspace for >= _MAX_PSEUDO sequences of T tokens
    workspace: torch.Tensor,
    checkpoint_state: torch.Tensor | None = None,  # [N, H, D, D] state dtype (written)
) -> torch.Tensor:
    """Writes out, final_state and (rows with a checkpoint) checkpoint_state."""
    load_extension()
    op = torch.ops.k3_flashkda_split.fwd
    tokens = q.shape[1]
    assert tokens <= bufs.tokens and plan.num_pseudo <= bufs.n
    n = plan.num_pseudo
    q, k, v, g = q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous()
    A_log = A_log.contiguous()
    dt_bias = dt_bias.view(-1, q.shape[-1]).contiguous()
    bv = _beta_view(beta)
    scale = q.shape[-1] ** -0.5
    ws = workspace
    assert ws.numel() >= workspace_size(tokens, q.shape[2], n) and out.is_contiguous()
    C, M, X, F, K = bufs.C[:n], bufs.M[:n], bufs.X[:n], bufs.F[:n], bufs.K[:n]
    scratch = out  # phase A outputs are discarded; phase C rewrites every token
    # K1 once (token-parallel; the segmented tiles equal the unsegmented ones)
    op(
        q,
        k,
        v,
        g,
        bv,
        scale,
        scratch,
        ws,
        A_log,
        dt_bias,
        lower_bound,
        None,
        None,
        plan.cu,
        None,
        None,
        1,
    )
    # A: C_p from a zero state, M_p from the identity with v = 0
    op(
        q,
        k,
        v,
        g,
        bv,
        scale,
        scratch,
        ws,
        A_log,
        dt_bias,
        lower_bound,
        None,
        C,
        plan.cu,
        None,
        None,
        2,
    )
    op(
        q,
        k,
        bufs.zero_v[:, :tokens],
        g,
        bv,
        scale,
        scratch,
        ws,
        A_log,
        dt_bias,
        lower_bound,
        bufs.eye[:n],
        M,
        plan.cu,
        None,
        None,
        2,
    )
    # B: each sequence's segment initial states, fp32
    chain_scan(initial_state.contiguous(), M, C, X, plan)
    # C: every segment from its true initial state
    kernel_ckpt = checkpoint_state is not None and plan.kernel_ckpt_offsets is not None
    op(
        q,
        k,
        v,
        g,
        bv,
        scale,
        out,
        ws,
        A_log,
        dt_bias,
        lower_bound,
        X,
        F,
        plan.cu,
        K if kernel_ckpt else None,
        plan.kernel_ckpt_offsets if kernel_ckpt else None,
        2,
    )
    final_state.copy_(F.index_select(0, plan.last))
    if checkpoint_state is not None:
        for src, rows in (
            (X, plan.ck_x),
            (F, plan.ck_f),
            (K if kernel_ckpt else None, plan.ck_k),
        ):
            if src is not None and rows is not None:
                checkpoint_state.index_copy_(
                    0, rows[0], src.index_select(0, rows[1]).to(checkpoint_state.dtype)
                )
    return out
