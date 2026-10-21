# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the DEP decode stream-K MXFP8 GEMM (``cutedsl/streamk_gemm.py``).

Weights keep the online-MXFP8 layout vLLM already builds for FlashInfer: FP8
``[N, K]`` and UE8M0 scales in the 128x4 swizzled layout (rows padded to 128).
Activations come from ``mxfp8_e4m3_quantize(..., is_sf_swizzled_layout=True)``.
"""

from functools import cache

import torch

from .cutedsl.streamk_gemm import ROW_TILE, SUPPORTED_TILE_N

MAX_TILE_N = max(SUPPORTED_TILE_N)
# Fewer K blocks per CTA than this and the per-CTA fixed cost dominates.
MIN_BLOCKS_PER_CTA = 4
# K blocks each cluster split-K peer leaves to the leader so that its hand-over
# overlaps the leader's last blocks.
PEER_DEFICIT = 5
K_BLOCK = 128


def default_num_ctas(row_tiles: int, k_blocks: int, sms: int) -> int:
    """Whole row tiles per CTA: a split tile's reduction (about 3 us of exposed
    synchronization) costs more than idle SMs unless the GEMM has very few row
    tiles, which then split each tile over two CTAs (three below 16 tiles)."""
    tiles_per_cta = -(-row_tiles // sms)
    ctas = -(-row_tiles // tiles_per_cta)
    if 4 * ctas >= sms:
        return ctas
    parts = 2 if row_tiles >= 16 else 3
    return min(
        sms, row_tiles * parts, max(1, row_tiles * k_blocks // MIN_BLOCKS_PER_CTA)
    )


def default_ksplit(row_tiles: int, k_blocks: int, sms: int, tile_n: int) -> int:
    """CTAs per row tile in a cluster split-K: two when whole tiles would leave
    about half the SMs idle, the tile has at most 96 tokens (the peer buffer fits)
    and K is long enough for the hand-over to hide (40+ blocks)."""
    if 2 * row_tiles > sms or 4 * row_tiles < sms:
        return 1
    return 2 if tile_n <= 96 and k_blocks >= 40 else 1


def tile_n_for(tokens: int) -> int:
    for tile_n in SUPPORTED_TILE_N:
        if tokens <= tile_n:
            return tile_n
    raise ValueError(
        f"DEP stream-K GEMM supports up to {MAX_TILE_N} tokens, got {tokens}"
    )


@cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _compile(
    tile_n: int,
    residual: bool,
    b_stages: int = 4,
    ablate: int = 0,
    k_tile: int = 128,
    ksplit: int = 1,
    peer_deficit: int = 0,
):
    """The compiled kernel of a variant (one cache entry however it is named)."""
    return _compile_variant(
        tile_n, residual, b_stages, ablate, k_tile, ksplit, peer_deficit
    )


@cache
def _compile_variant(
    tile_n: int,
    residual: bool,
    b_stages: int,
    ablate: int,
    k_tile: int,
    ksplit: int,
    peer_deficit: int,
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    from .cutedsl.streamk_gemm import DepMxfp8StreamKGemm

    kernel = DepMxfp8StreamKGemm(
        tile_n, residual, b_stages, ablate, k_tile, ksplit, peer_deficit
    )
    sym_n, sym_k, sym_m = cute.sym_int(), cute.sym_int(), cute.sym_int()
    fp8 = cutlass.Float8E4M3FN

    def fake(dtype, shape, align=16):
        return cute.runtime.make_fake_compact_tensor(
            dtype, shape, stride_order=(1, 0), assumed_align=align
        )

    def fake1d(dtype):
        return cute.runtime.make_fake_compact_tensor(
            dtype, (cute.sym_int(),), assumed_align=16
        )

    return cute.compile(
        kernel.wrapper,
        fake(fp8, (sym_n, sym_k)),
        fake(fp8, (sym_m, sym_k)),
        fake(cutlass.BFloat16, (sym_m, sym_n)),
        fake(cutlass.BFloat16, (sym_m, sym_n)),
        fake1d(cutlass.Float32),
        fake1d(cutlass.Int32),
        1,
        1,
        1,
        make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16),
        make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16),
        cutlass.Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )


def compile_all() -> None:
    """Compile every variant ``streamk_mxfp8_gemm`` launches by default (call
    before CUDA graph capture): each token tile, with and without a residual,
    plus the cluster split-K variants of tiles up to 96 tokens."""
    for tile_n in SUPPORTED_TILE_N:
        for residual in (False, True):
            _compile(tile_n, residual)
            if tile_n <= 96:
                _compile(tile_n, residual, ksplit=2, peer_deficit=PEER_DEFICIT)


class StreamKScratch:
    """Workspace and tile counters shared by every stream-K GEMM on a device.

    Launches on one stream never overlap their reductions: each epilogue starts
    behind the PDL wait, after the previous grid completed, and the last arriver
    of every split tile re-zeroes its accumulator and counter.
    """

    _per_device: dict[int, "StreamKScratch"] = {}

    def __init__(self, device: torch.device):
        sms = _num_sms(device.index)
        # One FP32 slot per CTA (a CTA helps at most one split tile: the one its
        # run opens with) and a counter per split tile, indexed by its owner CTA.
        self.workspace = torch.zeros(
            sms * ROW_TILE * MAX_TILE_N, dtype=torch.float32, device=device
        )
        self.counters = torch.zeros(sms, dtype=torch.int32, device=device)

    @classmethod
    def get(cls, device: torch.device) -> "StreamKScratch":
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        scratch = cls._per_device.get(index)
        if scratch is None:
            scratch = cls._per_device[index] = cls(torch.device("cuda", index))
        return scratch


def streamk_mxfp8_gemm(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    num_ctas: int | None = None,
    b_stages: int = 4,
    ablate: int = 0,
    k_tile: int = 128,
    ksplit: int | None = None,
    peer_deficit: int | None = None,
) -> torch.Tensor:
    """``out[M, N] = x[M, K] @ W[N, K]^T (+ residual)`` in BF16.

    ``x_scale`` / ``weight_scale`` are swizzled UE8M0 scales (rows padded to 128).
    """
    m, k = x_fp8.shape
    n, k_w = weight_fp8.shape
    assert k == k_w and k % k_tile == 0, (k, k_w, k_tile)
    assert x_fp8.is_contiguous() and weight_fp8.is_contiguous()
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=x_fp8.device)
    row_tiles = (n + ROW_TILE - 1) // ROW_TILE
    sms = _num_sms(x_fp8.device.index)
    if ksplit is None:
        ksplit = (
            1
            if num_ctas is not None
            else default_ksplit(row_tiles, k // k_tile, sms, tile_n_for(m))
        )
    if ksplit > 1:
        num_ctas = row_tiles * ksplit
    if num_ctas is None:
        num_ctas = default_num_ctas(
            row_tiles, k // k_tile, _num_sms(x_fp8.device.index)
        )
    scratch = StreamKScratch.get(x_fp8.device)
    compiled = _compile(
        tile_n_for(m),
        residual is not None,
        b_stages,
        ablate,
        k_tile,
        ksplit,
        (min(PEER_DEFICIT, k // k_tile // 4) if peer_deficit is None else peer_deficit)
        if ksplit > 1
        else 0,
    )
    compiled(
        weight_fp8,
        x_fp8,
        out,
        residual if residual is not None else out,
        scratch.workspace,
        scratch.counters,
        row_tiles,
        (m + 127) // 128,
        (k // 32 + 3) // 4,
        weight_scale.data_ptr(),
        x_scale.data_ptr(),
        num_ctas,
    )
    return out
