# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DEP decode's MLA core: FP8 latent attention over the paged FP8 cache with a
device-planned split-KV work list.

The kernel is TokenSpeed's FP8 MLA decode (vendored in ``cutedsl/mla``): packed
query rows (96 heads x q tokens in 128-row tiles), one 2-CTA cluster per (query
tile, request, KV split), FP16 partials merged by its reduction kernel. TokenSpeed
sizes the split grid by a host split capacity, so every request launches as many
split clusters as the largest one, and clusters past a request's split count only
write empty partials. Here a planner kernel picks each request's split count from
its KV length (whole waves of the GPU's clusters, at most ``capacity`` splits) and
lists the planned clusters; one wave of persistent clusters walks the list, so no
split cluster runs empty however short or unequal the lengths are. Nothing is read
on the host: plans are CUDA-graph safe.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from vllm.triton_utils import tl, triton

from .cutedsl.mla.helpers import get_mla_decode_arch, select_mla_decode_tilers
from .cutedsl.mla.kernel import BlackwellMultiHeadLatentAttentionForwardFP8

LATENT, ROPE = 512, 64
KV_TILE = 128  # KV tokens per MMA tile
CAPACITY = 8  # most splits of one request (sizes the partial workspace)


@triton.jit
def _plan_work_list_kernel(
    seq_lens_ptr,
    counts_ptr,
    work_ptr,
    num_reqs,
    q_tiles,
    clusters,
    capacity,
    min_tiles,
    overhead_tiles,
    list_len,
    KV_TILE: tl.constexpr,
    MAX_WAVES: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_G: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    """Per-request split counts, then the work list.

    For each W = 1 .. MAX_WAVES waves of ``clusters`` (rows of a [W, B] search):
    the counts ceil(T_b / c) (T_b: KV tiles, at most ``capacity``) for the smallest
    chunk c >= ``min_tiles`` whose clusters fit W * clusters, as the kernel's
    nonempty splits; the estimated time is waves used x (longest chunk +
    ``overhead_tiles``). The fastest W wins (ties: fewer waves). Entry g of the
    list is ``(q_tile * B + b) << 8 | split`` (request-major), then -1."""
    offs = tl.arange(0, BLOCK_B)
    mask = offs < num_reqs
    lens = tl.load(seq_lens_ptr + offs, mask, other=0)
    tiles_i = tl.where(mask, tl.maximum((lens + KV_TILE - 1) // KV_TILE, 1), 0)
    tiles = tiles_i.to(tl.float32)[None, :]
    waves_w = tl.arange(0, BLOCK_W) + 1
    budget = (waves_w * clusters).to(tl.float32)
    low = tl.zeros([BLOCK_W], tl.float32) + tl.maximum(min_tiles.to(tl.float32), 1.0)
    high = tl.maximum(tl.zeros([BLOCK_W], tl.float32) + tl.max(tiles, 1), low)
    for _ in tl.static_range(SEARCH_STEPS):
        mid = 0.5 * (low + high)
        n = tl.minimum(tl.ceil(tiles / mid[:, None]), capacity)
        used = tl.sum(tl.where(mask[None, :], n, 0.0), 1) * q_tiles
        fits = used <= budget
        high = tl.where(fits, mid, high)
        low = tl.where(fits, low, mid)
    n = tl.maximum(tl.minimum(tl.ceil(tiles / high[:, None]), capacity), 1.0)
    n = n.to(tl.int32)
    per = (tiles_i[None, :] + n - 1) // n  # the kernel's chunk, in KV tiles
    n = tl.where(mask[None, :], (tiles_i[None, :] + per - 1) // per, 0)
    chunk = tl.max(tl.where(mask[None, :], per, 0), 1)
    used = tl.sum(n, 1) * q_tiles
    estimate = ((used + clusters - 1) // clusters) * (chunk + overhead_tiles)
    estimate = tl.where(waves_w <= MAX_WAVES, estimate, 1 << 30)
    best = tl.argmin(estimate, 0)
    rows = tl.arange(0, BLOCK_W)
    best_n = tl.sum(tl.where(rows[:, None] == best, n, 0), 0)
    tl.store(counts_ptr + offs, best_n, mask)

    work = best_n * q_tiles
    ends = tl.cumsum(work, 0)
    starts = ends - work
    g = tl.arange(0, BLOCK_G)
    in_req = (g[:, None] >= starts[None, :]) & (g[:, None] < ends[None, :])
    in_req = in_req & mask[None, :]
    req = tl.sum(tl.where(in_req, offs[None, :], 0), 1)
    start = tl.sum(tl.where(in_req, starts[None, :], 0), 1)
    valid = tl.sum(in_req.to(tl.int32), 1) > 0
    local = g - start
    split = local // q_tiles
    q_tile = local - split * q_tiles
    entry = ((q_tile * num_reqs + req) << 8) | split
    tl.store(work_ptr + g, tl.where(valid, entry, -1), g < list_len)


@cache
def _compile(heads, q, page_size, capacity, arch_cc, use_pdl):
    """The FP8 split-KV MLA decode with a work list (packed query rows, causal
    among the q verify rows, FP16 partials, BF16 latent output)."""
    qk_tiler, pv_tiler = select_mla_decode_tilers(
        heads, q, is_fp8=True, compute_capability=arch_cc
    )
    assert qk_tiler[0] == 128, "the work list targets the 2-CTA M128 kernel"
    hardware = cutlass.utils.HardwareInfo()
    kernel = BlackwellMultiHeadLatentAttentionForwardFP8(
        compute_capability=arch_cc,
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=qk_tiler,
        mma_pv_tiler_mn=pv_tiler,
        max_active_clusters=hardware.get_max_active_clusters(2),
        page_size=page_size,
        skip_correction_threshold=8.0,
        is_persistent=False,
        is_var_seq=True,
        is_var_split_kv=True,
        fold_sq_factor=1,
        is_causal=True,
        num_heads=heads,
        seq_len_q=q,
        window_left=-1,
        cp_world=1,
        cp_rank=0,
        cp_interleave_size=1,
        reducer_d_tiles=1,
        pack_q=True,
        reducer_max_splits=max(4, 1 << (capacity - 1).bit_length()),
        partial_fp16=True,
        work_list=True,
    )
    fp8, sym = cutlass.Float8E4M3FN, cute.sym_int
    make, compact = cute.runtime.make_fake_tensor, cute.runtime.make_fake_compact_tensor
    batch, seq_q, num_heads = sym(), sym(), sym()
    latent, rope = sym(divisibility=16), sym(divisibility=16)
    pages, page_len, page_count = sym(), sym(), sym()

    def query(dim):
        return make(
            fp8,
            (batch, seq_q, num_heads, dim),
            stride=(sym(), sym(), sym(), 1),
            assumed_align=16,
        )

    def cache_view(dim):
        return make(
            fp8, (pages, page_len, dim), stride=(sym(), sym(), 1), assumed_align=16
        )

    lengths = compact(Int32, (batch,), assumed_align=4)
    args = (
        kernel,
        query(latent),
        query(rope),
        cache_view(latent),
        cache_view(rope),
        compact(Int32, (batch, page_count), stride_order=(1, 0), assumed_align=4),
        compact(
            cutlass.BFloat16,
            (batch, seq_q, num_heads, latent),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        ),
        None,  # LSE
        compact(cutlass.Int8, (sym(),), assumed_align=32),
        Int32(capacity),
        lengths,  # cache lengths
        lengths,  # causal bound (no DCP)
        compact(Int32, (batch,), assumed_align=4),  # split counts
        Float32(1.0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        use_pdl,
        None,  # local visible lengths
        compact(Int32, (sym(),), assumed_align=4),  # work list
    )
    arch = get_mla_decode_arch(arch_cc)
    return cute.compile(
        *args, options=f"--enable-tvm-ffi --opt-level 2 --gpu-arch={arch}a"
    )


class WorkListMla:
    """FP8 MLA decode of up to ``max_tokens`` query rows (``q`` per request) for a
    deployment's MLA layers: one plan per step (``plan``), shared by every layer."""

    def __init__(
        self,
        heads: int,
        q: int,
        max_tokens: int,
        device: torch.device,
        *,
        max_waves: int = 8,
        min_split_tokens: int = 1024,
        overhead_tiles: int = 6,
        capacity: int = CAPACITY,
        page_size: int = 64,
        use_pdl: bool = False,
    ):
        self.heads, self.q, self.capacity, self.page_size = (
            heads,
            q,
            capacity,
            page_size,
        )
        self.use_pdl = use_pdl
        self.arch = torch.cuda.get_device_capability(device)
        max_requests = -(-max_tokens // q)
        # HardwareInfo queries the current CUDA context: make sure there is one.
        torch.empty(0, device=device)
        # Query rows are packed into 128-row tiles, one 2-CTA cluster each.
        self.q_tiles = -(-heads * q // 128)
        self.clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(2)
        self.max_waves = max_waves
        self.min_tiles = max(1, min_split_tokens // KV_TILE)
        self.overhead_tiles = overhead_tiles
        # FP32-sized partials (the kernel's bound; FP16 partials use part of it).
        rows = max_requests * self.q_tiles * 128
        workspace = rows * capacity * (LATENT + 1) * 4
        self.workspace = torch.empty(workspace, dtype=torch.int8, device=device)
        self.counts = torch.ones(max_requests, dtype=torch.int32, device=device)
        # Planned clusters, then a wave of empty entries that ends every walk.
        self.list_len = (max_waves + 1) * self.clusters
        self.work = torch.full((self.list_len,), -1, dtype=torch.int32, device=device)
        self.max_requests = max_requests

    def plan(self, seq_lens: torch.Tensor) -> None:
        """Plan this step's split counts and work list (every MLA layer reuses it)."""
        requests = seq_lens.numel()
        assert requests <= self.max_requests
        _plan_work_list_kernel[(1,)](
            seq_lens,
            self.counts,
            self.work,
            requests,
            self.q_tiles,
            self.clusters,
            self.capacity,
            self.min_tiles,
            self.overhead_tiles,
            self.list_len,
            KV_TILE=KV_TILE,
            MAX_WAVES=self.max_waves,
            BLOCK_W=triton.next_power_of_2(self.max_waves),
            BLOCK_B=triton.next_power_of_2(max(16, requests)),
            BLOCK_G=triton.next_power_of_2(self.list_len),
            SEARCH_STEPS=20,
        )

    def decode(
        self,
        query: torch.Tensor,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        softmax_scale: float,
        output_scale: float,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """BF16 latent [T, H, 512] of FP8 ``query`` [T, H, 576] (``q`` rows per
        request, causal among them) over the FP8 ``cache`` [blocks, page, 576]."""
        import tvm_ffi

        requests = seq_lens.numel()
        query = query.view(requests, self.q, self.heads, LATENT + ROPE)
        if out is None:
            out = query.new_empty(
                (requests, self.q, self.heads, LATENT), dtype=torch.bfloat16
            )
        else:
            out = out.view(requests, self.q, self.heads, LATENT)
        cache = cache.view(torch.float8_e4m3fn)
        kernel = _compile(
            self.heads, self.q, self.page_size, self.capacity, self.arch, self.use_pdl
        )
        with tvm_ffi.use_torch_stream():
            kernel(
                query[..., :LATENT],
                query[..., LATENT:],
                cache[:, :, :LATENT],
                cache[:, :, LATENT:],
                block_table,
                out,
                None,
                self.workspace,
                Int32(self.capacity),
                seq_lens,
                seq_lens,
                self.counts[:requests],
                Float32(softmax_scale),
                Float32(output_scale),
                None,
                self.work,
            )
        return out.view(-1, self.heads, LATENT)
