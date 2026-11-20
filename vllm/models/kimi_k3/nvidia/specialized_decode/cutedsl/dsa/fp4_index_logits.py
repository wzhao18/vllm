# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 decode indexer logits with one key stream per request.

The stock SM100 path (DeepGEMM ``fp8_fp4_paged_mqa_logits``, varlen form) scores
every verify row as its own sequence, so each key block is read once per row: q
times per request (4x at k = 3). Here the q rows of a request are scored together:
the MMA tile is 128 keys x (q x 32 heads), one pass over the request's keys
(fp4_mqa_logits_kernel, vendored FlashInfer CuTe DSL kernel). Same math as the
stock op: per head ReLU of the block-scaled E2M1 dot product (FP32 accumulate),
FP32 head-weighted sum, BF16 logits; the head sum runs in another order.

Every row of a request is scored up to the longest local length of its rows (the
request's last row); the caller masks each row by its own end (``row_ends``), as
it does for the stock logits, whose columns past a row's end are unspecified too.
"""

from functools import cache

import torch

from vllm.triton_utils import tl, triton

BLOCK_KV = 128  # the kernel's compute tile (two 64-token pages)
SPLIT_KV = 256  # its schedule unit: one tile per math warpgroup
STORE = 256  # it writes logit columns in multiples of this
HEADS = 32
HALF_D = 64  # E2M1 bytes of a 128-dim row
PAGE = 64  # indexer page tokens


@triton.jit
def _prep_kernel(lens_ptr, sl_ptr, cum_ptr, sched_ptr, hw_ptr, w_ptr, reqs, n_w,
                 Q: tl.constexpr, NUM_SMS: tl.constexpr, SPLIT: tl.constexpr,
                 BLOCK_B: tl.constexpr, LOG_B: tl.constexpr, BLOCK_C: tl.constexpr,
                 BLOCK_W: tl.constexpr):
    """Program 0: per-request length (max of its Q row lengths) and the kernel's CTA
    schedule (FlashInfer/DeepGEMM PagedMQALogitsScheduler: the requests' SPLIT-token
    splits dealt evenly over NUM_SMS CTAs; row i = (request, split) where CTA i
    starts, (reqs, 0) past the end; the request by binary search over the inclusive
    split sums in ``cum_ptr``). Other programs: head weights BF16 -> FP32."""
    pid = tl.program_id(0)
    if pid == 0:
        b = tl.arange(0, BLOCK_B)
        m = b < reqs
        sl = tl.zeros([BLOCK_B], dtype=tl.int32)
        for j in tl.static_range(Q):
            sl = tl.maximum(sl, tl.load(lens_ptr + b * Q + j, mask=m, other=0))
        tl.store(sl_ptr + b, sl, mask=m)
        splits = tl.where(m, (sl + SPLIT - 1) // SPLIT, 0)
        tl.store(cum_ptr + b, tl.cumsum(splits, 0), mask=m)
        total = tl.sum(splits, 0)
        tl.debug_barrier()
        q_div = total // NUM_SMS
        r_mod = total % NUM_SMS
        i = tl.arange(0, BLOCK_C)
        tgt = i * q_div + tl.minimum(i, r_mod)
        lo = tl.zeros([BLOCK_C], dtype=tl.int32)
        hi = tl.zeros([BLOCK_C], dtype=tl.int32) + reqs
        for _ in tl.static_range(LOG_B + 1):  # first request whose sum exceeds tgt
            act = lo < hi
            mid = (lo + hi) // 2
            c = tl.load(cum_ptr + mid, mask=act, other=0)
            up = act & (c <= tgt)
            lo = tl.where(up, mid + 1, lo)
            hi = tl.where(act & (c > tgt), mid, hi)
        before = tl.load(cum_ptr + lo - 1, mask=lo > 0, other=0)
        done = lo >= reqs
        mi = i <= NUM_SMS
        tl.store(sched_ptr + 2 * i, tl.where(done, reqs, lo), mask=mi)
        tl.store(sched_ptr + 2 * i + 1, tl.where(done, 0, tgt - before), mask=mi)
    else:
        o = (pid - 1) * BLOCK_W + tl.arange(0, BLOCK_W)
        mw = o < n_w
        tl.store(w_ptr + o, tl.load(hw_ptr + o, mask=mw).to(tl.float32), mask=mw)


def _arch(device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}a"


@cache
def _compiled(q: int, num_sms: int, arch: str, epi_math: bool = True, sf_skip: bool = False):
    import cutlass
    import cutlass.cute as cute

    from .fp4_mqa_logits_kernel import FP4MQALogitsKernel

    n = q * HEADS
    kernel = FP4MQALogitsKernel(
        block_kv=BLOCK_KV,
        phys_block_kv=PAGE,
        num_heads=HEADS,
        head_dim=2 * HALF_D,
        next_n=q,
        num_next_n_atoms=1,
        num_sms=num_sms,
        num_epi_subtiles=1,
        num_umma_stages=2,
        epi_math=epi_math,
        epi_dtype=cutlass.Float32,
        output_dtype=cutlass.BFloat16,
        remove_online_sf_transpose=sf_skip,
        arch=arch,
    )
    sym = cute.sym_int
    kv = cute.runtime.make_fake_tensor(
        cutlass.Uint8, (sym(), PAGE * (HALF_D + 4)), stride=(cute.sym_int64(), 1))
    q_t = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (n, HALF_D, sym()), stride_order=(1, 0, 2))
    sf = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (n, sym()), stride_order=(0, 1))
    w = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (n, sym()), stride_order=(0, 1))
    logits = cute.runtime.make_fake_tensor(
        cutlass.BFloat16, (sym(), sym()), stride=(cute.sym_int64(), 1))
    batch = sym()
    bt = cute.runtime.make_fake_tensor(
        cutlass.Int32, (batch, sym()), stride=(cute.sym_int64(), 1), assumed_align=4)
    cl = cute.runtime.make_fake_tensor(cutlass.Int32, (batch,), stride=(sym(),), assumed_align=4)
    sm = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sym(), 2), stride_order=(1, 0))
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel, kv, q_t, sf, w, logits, bt, cl, sm, cutlass.Int32(1),
                        cutlass.Int32(1), stream,
                        options=f"--gpu-arch {arch} --enable-tvm-ffi")


class Fp4IndexLogits:
    """Per-layer buffers of the request-batched scoring (``max_rows``: the most decode
    rows of a step; ``q``: verify rows per request)."""

    def __init__(self, max_rows: int, q: int, device):
        assert 1 <= q <= 4, q
        self.q = q
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.max_rows = max_rows
        # Step buffers come from the caller's stream allocation on every call (under
        # CUDA graphs: the capturing graph's pool), so no tensor outlives a graph.
        self.seq_lens = self.cum = self.schedule = self.weights = None
        self.arch = _arch(device)
        self.epi_math = True  # benchmarks only: False skips the head-sum math
        self.sf_skip = False  # benchmarks only: True skips the KV scale transpose (wrong math)

    def warm(self):
        _compiled(self.q, self.num_sms, self.arch)

    def __call__(self, query, q_scale, head_w, kv_cache, row_lens, block_table, max_len):
        """BF16 logits [rows, >= max_len] (row stride a multiple of 1 KiB): ``query``
        E2M1 [rows, 32, 64] and ``q_scale`` UE8M0 x4 [rows, 32] (fp4_query_views),
        ``head_w`` [rows, 32] BF16, ``kv_cache`` the indexer cache [pages, 64, 68]
        (page stride free), ``row_lens`` [rows] or [rows, 1] the rows' local lengths,
        ``block_table`` [rows, pages] (row r of request b: r = b * q + t; the request
        reads row b * q's)."""
        q, rows = self.q, query.shape[0]
        assert rows % q == 0, (rows, q)
        assert query.is_contiguous() and q_scale.is_contiguous() and head_w.shape[1] == HEADS
        reqs = rows // q
        lens = row_lens.reshape(rows)
        assert lens.dtype == torch.int32 and lens.is_contiguous()
        self.prepare(head_w, lens, rows, reqs)
        return self.launch(query, q_scale, kv_cache, block_table, max_len, rows, reqs)

    def prepare(self, head_w, lens, rows, reqs):
        """The per-request lengths, the CTA schedule and the FP32 head weights."""
        dev = head_w.device
        self.seq_lens = torch.empty(reqs, dtype=torch.int32, device=dev)
        self.cum = torch.empty(reqs, dtype=torch.int32, device=dev)
        self.schedule = torch.empty(self.num_sms + 1, 2, dtype=torch.int32, device=dev)
        self.weights = torch.empty(rows, HEADS, dtype=torch.float32, device=dev)
        q, w = self.q, self.weights
        n_w = rows * HEADS
        block_w = 4096
        grid = (1 + triton.cdiv(n_w, block_w),)
        block_b = max(16, triton.next_power_of_2(reqs))
        _prep_kernel[grid](
            lens, self.seq_lens, self.cum, self.schedule, head_w[:rows].contiguous(), w,
            reqs, n_w, Q=q, NUM_SMS=self.num_sms, SPLIT=SPLIT_KV, BLOCK_B=block_b,
            LOG_B=block_b.bit_length() - 1,
            BLOCK_C=triton.next_power_of_2(self.num_sms + 1), BLOCK_W=block_w)

    def launch(self, query, q_scale, kv_cache, block_table, max_len, rows, reqs):
        q, w = self.q, self.weights[:rows]
        width = triton.cdiv(max_len, STORE) * STORE
        pitch = triton.cdiv(width, 512) * 512  # DeepSelect: 1 KiB BF16 row stride
        out = torch.empty(rows, pitch, dtype=torch.bfloat16, device=query.device)
        logits = out[:, :width]
        pages = kv_cache.shape[0]
        kv = torch.as_strided(kv_cache, (pages, PAGE * (HALF_D + 4)), (kv_cache.stride(0), 1))
        n = q * HEADS
        fn = _compiled(q, self.num_sms, self.arch, self.epi_math, self.sf_skip)
        fn(kv, query.view(reqs, n, HALF_D).permute(1, 2, 0),
           q_scale.view(reqs, n).permute(1, 0), w.view(reqs, n).permute(1, 0), logits,
           block_table[::q][:reqs], self.seq_lens[:reqs], self.schedule, pages, reqs)
        return out[:, :max_len]
