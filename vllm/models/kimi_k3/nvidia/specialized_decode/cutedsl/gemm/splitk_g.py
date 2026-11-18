# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-streaming split-K projection with the K reduction through L2 (``SplitKGemmG``).

``SplitKGemm`` (FlashInfer's split-K core) reduces its cluster's K partials through a
shared-memory mailbox on rank 0: (split - 1) x 128 x N-tile FP32 that is allocated
beside the operand ring, so at the SP decode's 32-128 token columns a split above 2
leaves room for only one or two ring stages (too few weight bytes in flight to stream
at HBM rate; the D-GEMM bench measured 5-7 TB/s for the 32-96-row SP shapes), and one
rank sums every partial.

Here every CTA keeps the whole shared memory for the ring (weights requested before
the PDL wait, as ``SplitKGemm``), and the epilogue:

1. loads its FP32 accumulator row from tensor memory (``tcgen05.ld`` 32x32b: epilogue
   thread t <-> weight row t of the 128-row tile, column j <-> token j);
2. writes it to a global FP32 workspace ``[cluster][rank][column][row]`` (coalesced,
   L2-resident: split x 128 x N-tile x 4 B per cluster);
3. one cluster barrier (``barrier.cluster`` arrive.release / wait.acquire: the cluster
   is co-scheduled, so the wait cannot deadlock);
4. rank r owns the token columns j with j % split == r: it sums the split partials in
   rank order, ((p0 + p1) + p2) + ..., rounds once to BF16 and stores them.

So no separate reduce kernel, no shared-memory mailbox, and the reduction is spread
over all ranks. FP32 accumulation within a rank is the MMA's; the split partials are
summed in rank order (an FP32 reorder vs stock's cuBLAS, as every split-K tactic).

The workspace is reused by consecutive calls: a call's epilogue runs after its
activations land, i.e. after ``griddepcontrol.wait``, which waits for the whole
previous grid (whose epilogue last read the workspace)."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32
from cutlass.cute import experimental as cute_ext

from ..common.primitives import asm, load_tmem
from .splitk import CTA_K, SplitKGemm, SplitKTactic, _fence_tcgen05, _smem

__all__ = ["SplitKGemmG", "workspace_floats", "G_N_TILES"]

# Token columns per CTA (the MMA N; one accumulator of N FP32 TMEM columns).
G_N_TILES = (8, 16, 32, 48, 64, 80, 96, 112, 128)
MAX_SPLIT = 8  # portable cluster size


def workspace_floats(n_rows, tokens, tactic):
    """FP32 workspace elements of one call of ``tactic`` on an [n_rows, K] weight with
    ``tokens`` activation rows."""
    m_tile, n_tile, split = tactic[:3]
    clusters = -(-n_rows // m_tile) * -(-tokens // n_tile)
    return clusters * split * n_tile * m_tile


@asm()
def _cluster_sync_release_acquire():
    code = "barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;"
    return code, "~{memory}", ()


def _load_columns(address, n):
    """``n`` FP32 TMEM columns of this lane (power-of-two ``tcgen05.ld`` pieces)."""
    values, column, rest = (), 0, n
    while rest:
        count = 1 << (rest.bit_length() - 1)
        count = min(count, 64)
        part = load_tmem(address + column, count)
        values += part if count > 1 else (part,)
        column, rest = column + count, rest - count
    return values


class SplitKGemmG(SplitKGemm):
    """``SplitKGemm``'s mainloop with the L2 reduction epilogue (module docstring).
    Tactic (128, N tile, split, stages); ``__call__(a, b, c, workspace, a_map, stream)``
    with ``workspace`` a flat FP32 tensor of at least ``workspace_floats`` elements."""

    def __init__(self, *, tactic: SplitKTactic):
        assert tactic.mma_m == 128 and tactic.mma_n in G_N_TILES
        assert 1 <= tactic.split_k <= MAX_SPLIT
        super().__init__(tactic=tactic)
        self.mailbox_elements = 0
        self.expected_transaction_bytes = 0

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mW: cute.Tensor,
        a_map: cutlass.Int64,
    ):  # fmt: skip
        """A (M, K, L) and B (N, K, L) K-major, C (M, N, L) M-major; W FP32 flat."""
        stages = self.num_ab_stage
        ab_dtype = mA.element_type
        majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group,
            self.mma_tiler_mn,
        )  # fmt: skip

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        shape = (tiled_mma, mnk_tiler, ab_dtype, stages)
        sA = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_a(*shape))
        sB = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_b(*shape))
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        bar_full = _smem(cutlass.Int64, stages, 8).iterator
        bar_empty = _smem(cutlass.Int64, stages, 8).iterator
        bar_mma_epilog = _smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _smem(cutlass.Int32, 1, 4).iterator

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count
        tiles = (k_tile_start, k_tile_count)
        if warp_idx == 0:
            self.weight_warp(bar_full, bar_empty, sA, a_map, bidx * self.cta_m, *tiles)
        elif warp_idx == 1:
            b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx))
            b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
            self.activation_warp(bar_full, bar_empty, b_tile, sB, b_map, *tiles)
        elif warp_idx == 2:
            k_blocks = self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2])
            self.mma_warp(
                bar_full, bar_empty, bar_mma_epilog, bar_tmem_alloc, tiled_mma, sA, sB,
                tmem_base_ptr, acc_layout, k_blocks, k_tile_count,
            )  # fmt: skip
        elif warp_idx >= 4:
            cluster = bidx * cute.ceil_div(mC.layout.shape[1], self.cta_n) + n_idx
            self.reduce_epilogue(
                bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, mC, mW, bidx, n_idx,
                cluster, split_rank, cute.arch.thread_idx()[0] - 128,
            )  # fmt: skip
        if warp_idx < 4:
            _cluster_sync_release_acquire()

    @cute.experimental.jit
    def reduce_epilogue(
        self, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, mC: cute.Tensor,
        mW: cute.Tensor, bidx: Int32, n_idx: Int32, cluster: Int32, rank: Int32,
        row: Int32,
    ):  # fmt: skip
        n_tile, split = self.cta_n, self.split_k
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _fence_tcgen05()
        base = tmem_base_ptr[0]
        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _fence_tcgen05()
        # this thread's TMEM lane quadrant (warp % 4) -> lane bits 31:16
        address = base + ((row // 32) * 32 << 16)
        values = _load_columns(address, n_tile)
        cute.arch.fence_view_async_tmem_load()
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)  # TMEM may be released

        # partials: W[cluster][rank][column][row]
        mine = (cluster * split + rank) * n_tile * 128
        for j in cutlass.range_constexpr(n_tile):
            mW[mine + j * 128 + row] = values[j]
        _cluster_sync_release_acquire()

        m = bidx * self.cta_m + row
        rows_w, tokens = mC.layout.shape[0], mC.layout.shape[1]
        first = cluster * split * n_tile * 128
        for jj in cutlass.range_constexpr((n_tile + split - 1) // split):
            j = jj * split + rank
            token = n_idx * n_tile + j
            if j < n_tile:
                if token < tokens:
                    if m < rows_w:
                        acc = Float32(0.0)
                        for r in cutlass.range_constexpr(split):
                            acc = acc + mW[first + (r * n_tile + j) * 128 + row]
                        mC[m, token, 0] = acc.to(mC.element_type)


def check_tactic(tactic, k):
    """Why ``tactic`` cannot serve reduction length ``k`` (None: it can)."""
    if len(tactic) != 4:
        return f"G tactic {tactic!r}: (128, N tile, split, stages)"
    m, n, split, stages = tactic
    if m != 128 or n not in G_N_TILES or not 1 <= split <= MAX_SPLIT or stages < 1:
        return f"G tactic {tactic!r}: M 128, N in {G_N_TILES}, split <= {MAX_SPLIT}"
    if k % (CTA_K * split):
        return f"G tactic {tactic!r}: K {k} not a multiple of {CTA_K} x {split}"
    return None


_WORKSPACE: dict = {}


def workspace(device, floats):
    """The per-device FP32 workspace, grown (before capture) to floats."""
    import torch

    ws = _WORKSPACE.get(device)
    if ws is None or ws.numel() < floats:
        ws = _WORKSPACE[device] = torch.empty(floats, dtype=torch.float32, device=device)
    return ws


def compile_g(tactic, arch):
    from functools import cache  # noqa: F401  (code cache below)

    return _compile_g(tuple(tactic), arch)


def _fake(leading, dtype=cutlass.BFloat16):
    from cutlass.cute.runtime import make_fake_tensor

    shape = tuple(cute.sym_int() for _ in range(3))
    stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
    return make_fake_tensor(dtype, shape, stride, assumed_align=16)


_CODES: dict = {}


def _compile_g(tactic, arch):
    code = _CODES.get((tactic, arch))
    if code is None:
        from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

        kernel = SplitKGemmG(tactic=SplitKTactic(*tactic))
        w = make_fake_tensor(Float32, (cute.sym_int(),), (1,), assumed_align=16)
        args = (_fake(1), _fake(1), _fake(0), w, cutlass.Int64(0), make_fake_stream())
        code = _CODES[(tactic, arch)] = cute.experimental.compile(
            kernel, *args, options=f"--gpu-arch={arch}"
        )
    return code


class PreparedG:
    """SplitKGemmG for a BF16 [N, K] weight with one tactic per token-row count:
    run(x [R, K], out [R, N])."""

    def __init__(self, weight, tactics):
        from ..common.host import architecture, to_cute
        from .projection import weight_map

        n, k = weight.shape
        for r, t in tactics.items():
            problem = check_tactic(t, k)
            if problem is not None:
                raise ValueError(f"{r} rows: {problem}")
        self.weight, self.tactics = weight, dict(tactics)
        arch = architecture(weight.device)
        self.codes = {r: _compile_g(tuple(t), arch) for r, t in self.tactics.items()}
        self.maps = {}
        floats = max(workspace_floats(n, r, t) for r, t in self.tactics.items())
        self.ws = workspace(weight.device, floats)
        self.ws_view = to_cute(self.ws).mark_layout_dynamic(leading_dim=0)
        self.weight_view = None
        from .projection import matrix_view

        self.weight_view = matrix_view(weight)
        self._map = weight_map
        for t in self.tactics.values():
            if t[0] not in self.maps:
                self.maps[t[0]] = weight_map(weight, t[0])

    def run(self, x, out):
        from ..common.host import stream as current_stream
        from .projection import matrix_view

        r = x.shape[0]
        t = self.tactics[r]
        if t[0] not in self.maps:
            self.maps[t[0]] = self._map(self.weight, t[0])
        code = self.codes[r]
        y = matrix_view(out, output=True)
        code(self.weight_view, matrix_view(x), y, self.ws_view,
             cutlass.Int64(self.maps[t[0]].data_ptr()), current_stream(self.weight.device))
        return out
