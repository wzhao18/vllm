# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DEP decode's MoE router gate: FP32 logits of the BF16 gate on a few CTAs.

The router runs on a side stream beside the MoE block's front GEMM -> activation ->
shared-down chain, which streams weights at full bandwidth. The stock gate (cuBLAS
split-K plus a separate reduction kernel) spreads over most SMs and slows that chain
more than its own bytes justify. This gate splits the 896 experts into 64-row tiles
and K in two halves (28 CTAs): each CTA streams its weight slice with TMA, and the
last arriver of a tile sums the two halves in a fixed order and re-arms the tile's
counter for the next launch (graph replays need no reset).

It is used only for token counts where its logits equal the stock gate's bitwise
(``ExactGate`` checks every count against ``moe.gate`` before graph capture), so the
routing is identical to stock.
"""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

BLOCK_N, SPLIT, BLOCK_K, NUM_STAGES, NUM_WARPS = 64, 2, 128, 4, 8


@triton.jit(do_not_specialize=["M"])
def _gate_kernel(
    x_desc,
    w_desc,
    out_ptr,
    partial_ptr,
    counter_ptr,
    M,
    N,
    K,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n0 = pid_n * BLOCK_N
    k_per = K // SPLIT
    k0 = pid_k * k_per
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k0, k0 + k_per, BLOCK_K):
        xt = x_desc.load([0, k])
        wt = w_desc.load([n0, k])
        acc = tl.dot(xt, wt.T, acc)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    base = partial_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(base + pid_k * BLOCK_M * N, acc, mask=mask)
    arrived = tl.atomic_add(counter_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
    if arrived == SPLIT - 1:
        total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for s in range(SPLIT):
            total += tl.load(
                base + s * BLOCK_M * N, mask=mask, other=0.0, cache_modifier=".cg"
            )
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :], total, mask=mask
        )
        tl.atomic_xchg(counter_ptr + pid_n, 0)


def _block_m(tokens: int) -> int:
    return max(64, triton.next_power_of_2(tokens))


class _Scratch:
    """Partials and tile counters shared by every layer's gate on a device: the
    gates run one after another on the router stream."""

    _by_key: dict = {}

    def __init__(self, n: int, max_tokens: int, device: torch.device):
        tiles = triton.cdiv(n, BLOCK_N)
        self.partials = torch.empty(
            (SPLIT, _block_m(max_tokens), n), dtype=torch.float32, device=device
        )
        self.counters = torch.zeros(tiles, dtype=torch.int32, device=device)

    @classmethod
    def get(cls, n: int, max_tokens: int, device: torch.device) -> "_Scratch":
        key = (n, _block_m(max_tokens), device)
        if key not in cls._by_key:
            cls._by_key[key] = cls(n, max_tokens, device)
        return cls._by_key[key]


def few_cta_gate(
    weight: torch.Tensor, x: torch.Tensor, scratch: _Scratch
) -> torch.Tensor:
    """FP32 ``x @ weight.T`` for BF16 ``x`` [M, K] and ``weight`` [N, K]."""
    from triton.tools.tensor_descriptor import TensorDescriptor

    m, k = x.shape
    n = weight.shape[0]
    block_m = _block_m(m)
    stages = max(2, min(NUM_STAGES, 200_000 // ((block_m + BLOCK_N) * BLOCK_K * 2)))
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    x_desc = TensorDescriptor.from_tensor(x, [block_m, BLOCK_K])
    w_desc = TensorDescriptor.from_tensor(weight, [BLOCK_N, BLOCK_K])
    _gate_kernel[(triton.cdiv(n, BLOCK_N), SPLIT)](
        x_desc,
        w_desc,
        out,
        scratch.partials,
        scratch.counters,
        m,
        n,
        k,
        out.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT=SPLIT,
        num_warps=NUM_WARPS,
        num_stages=stages,
    )
    return out


def _ranges(values: list[int]) -> str:
    """``[1, 2, 3, 7]`` -> ``"1-3, 7"``."""
    spans: list[list[int]] = []
    for v in values:
        if spans and v == spans[-1][1] + 1:
            spans[-1][1] = v
        else:
            spans.append([v, v])
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in spans)


class ExactGate:
    """The few-CTA gate for one layer's ``GateLinear``, at the token counts where it
    reproduces the stock gate bitwise; ``None`` from ``__call__`` means use stock."""

    # (weight shape, device) -> token counts verified equal to the stock gate.
    _exact: dict = {}

    def __init__(self, gate: torch.nn.Module, max_tokens: int):
        weight = gate.weight
        self.usable = (
            weight.dtype == torch.bfloat16
            and weight.is_contiguous()
            and weight.shape[1] % (SPLIT * BLOCK_K) == 0
            and getattr(gate, "out_dtype", None) == torch.float32
        )
        self.weight = weight
        self.max_tokens = max_tokens
        if not self.usable:
            self.tokens: frozenset[int] = frozenset()
            return
        n = weight.shape[0]
        self.scratch = _Scratch.get(n, max_tokens, weight.device)
        key = (tuple(weight.shape), weight.device)
        if key not in self._exact:
            self._exact[key] = self._verify(gate)
        self.tokens = self._exact[key]

    def _verify(self, gate: torch.nn.Module) -> frozenset[int]:
        """Token counts whose logits equal the stock gate's on random inputs (this
        layer's weight); also compiles every variant before graph capture."""
        generator = torch.Generator(device=self.weight.device).manual_seed(0)
        exact = set()
        for m in range(1, self.max_tokens + 1):
            same = True
            for scale in (0.5, 4.0):
                x = (
                    torch.randn(
                        (m, self.weight.shape[1]),
                        generator=generator,
                        device=self.weight.device,
                        dtype=torch.float32,
                    )
                    * scale
                ).to(torch.bfloat16)
                ref = gate(x)
                ref = ref[0] if isinstance(ref, tuple) else ref
                got = few_cta_gate(self.weight, x, self.scratch)
                same &= ref.dtype == torch.float32 and torch.equal(ref, got)
            if same:
                exact.add(m)
        torch.cuda.synchronize(self.weight.device)
        missing = [m for m in range(1, self.max_tokens + 1) if m not in exact]
        logger.info_once(
            "DEP decode router gate: few-CTA kernel for %d of %d token counts "
            "(stock gate for %s)",
            len(exact),
            self.max_tokens,
            _ranges(missing) or "none",
        )
        return frozenset(exact)

    def __call__(self, x: torch.Tensor) -> torch.Tensor | None:
        if x.shape[0] not in self.tokens:
            return None
        return few_cta_gate(self.weight, x, self.scratch)
