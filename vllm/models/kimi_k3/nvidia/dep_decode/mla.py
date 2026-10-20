# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DEP decode's MLA core: TokenSpeed's MLA decode with wave split scheduling.

TP1 attention leaves each rank its own few requests, so the uniform split budget of
the stock backend leaves most of the GPU idle on short requests and serializes long
ones. Wave scheduling splits each request's KV by length and picks the split total
that fills whole waves of the GPU's concurrent CTA clusters; one plan per step
serves every MLA layer (they share the lengths).
"""

import torch

from vllm.triton_utils import tl, triton

# Reduction scratch per query-token head tile: enough for 8 splits of every token of
# the largest planned batch (as the stock backend reserves for a split floor of 8),
# and at least the stock bound of 8 query tokens per SM.
_SPLITS_RESERVED = 8
_MAX_SPLITS = 32


@triton.jit
def _wave_fit(lengths, mask, longest, budget, max_splits, SEARCH_STEPS: tl.constexpr):
    """Split counts ceil(len / T) for the smallest chunk length T whose total fits
    ``budget``, and that total."""
    low = longest / max_splits
    high = longest
    for _ in tl.static_range(SEARCH_STEPS):
        mid = 0.5 * (low + high)
        counts = tl.minimum(tl.maximum(tl.ceil(lengths / mid), 1.0), max_splits)
        fits = tl.sum(tl.where(mask, counts, 0.0), 0) <= budget
        high = tl.where(fits, mid, high)
        low = tl.where(fits, low, mid)
    counts = tl.minimum(tl.maximum(tl.ceil(lengths / high), 1.0), max_splits)
    return counts, tl.sum(tl.where(mask, counts, 0.0), 0)


@triton.jit(do_not_specialize=["num_reqs", "slots", "query_tiles", "max_splits"])
def _plan_wave_split_counts(
    seq_lens_ptr,
    counts_ptr,
    num_reqs,
    slots,
    query_tiles,
    max_splits,
    BLOCK_SIZE: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    """Split counts for whole waves of ``slots`` concurrent clusters (one cluster
    per split and query tile). Of the fewest waves that hold one split per
    request and one more, keeps the better-filled plan (ties: fewer waves); within
    a plan, splits minimize the longest chunk."""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_reqs
    lengths = tl.load(seq_lens_ptr + offsets, mask, other=0).to(tl.float32)
    lengths = tl.where(mask, tl.maximum(lengths, 1.0), 0.0)
    longest = tl.max(lengths, 0)
    waves = (num_reqs * query_tiles + slots - 1) // slots
    counts, total = _wave_fit(
        lengths, mask, longest, (waves * slots) // query_tiles, max_splits, SEARCH_STEPS
    )
    more, more_total = _wave_fit(
        lengths,
        mask,
        longest,
        ((waves + 1) * slots) // query_tiles,
        max_splits,
        SEARCH_STEPS,
    )
    fill = total / waves
    more_fill = more_total / (waves + 1)
    counts = tl.where(more_fill > 1.03 * fill, more, counts)
    tl.store(counts_ptr + offsets, counts.to(tl.int32), mask)


class WaveMla:
    """TokenSpeed MLA decode of one deployment's MLA layers (all share the head
    count, ranks and scales layout) with per-step wave split counts."""

    def __init__(self, module, max_tokens: int):
        from tokenspeed_mla import get_num_sm

        impl = module.impl
        self.kv_lora_rank, self.rope_dim = impl.kv_lora_rank, impl.qk_rope_head_dim
        self.heads = module.num_local_heads
        device = module.W_UV.device
        self.sms = get_num_sm(device)
        # Reduction scratch (FP32 partial outputs plus LSE) and the split counts of
        # the current step. Packed queries pad head rows to 128.
        rows = max(self.sms * self.heads, max_tokens * max(self.heads, 128))
        scratch = rows * _SPLITS_RESERVED * (self.kv_lora_rank + 1)
        self.workspace = torch.empty(scratch * 4, dtype=torch.int8, device=device)
        self.counts = torch.ones(max_tokens, dtype=torch.int32, device=device)

    def tiling(self, requests: int, q: int):
        """(query tiles per request, concurrent cluster slots, split capacity)."""
        from tokenspeed_mla import select_mla_decode_tilers

        (tile_rows, _), _ = select_mla_decode_tilers(
            self.heads,
            q,
            is_fp8=True,
            compute_capability=torch.cuda.get_device_capability(self.workspace.device),
        )
        query_tiles = (q * self.heads + tile_rows - 1) // tile_rows
        slots = self.sms // (1 if tile_rows == 64 else 2)
        # Whole query tiles bound both folded and packed layouts; TokenSpeed rejects
        # reduction workspaces whose byte offsets exceed int32.
        per_split = requests * query_tiles * tile_rows * (self.kv_lora_rank + 1) * 4
        capacity = min(
            _MAX_SPLITS, min(self.workspace.numel(), (1 << 31) - 1) // per_split
        )
        assert capacity >= 1, "the MLA reduction scratch holds no split"
        return query_tiles, slots, capacity

    def plan(self, seq_lens: torch.Tensor, q: int) -> int:
        """Plan this step's split counts (once for every MLA layer); returns the
        split capacity to decode with."""
        requests = seq_lens.numel()
        query_tiles, slots, capacity = self.tiling(requests, q)
        _plan_wave_split_counts[(1,)](
            seq_lens,
            self.counts,
            requests,
            slots,
            query_tiles,
            capacity,
            BLOCK_SIZE=triton.next_power_of_2(max(128, requests)),
            SEARCH_STEPS=20,
        )
        return capacity

    def __call__(self, module, query, md, capacity: int) -> torch.Tensor:
        """Latent attention output [T, H, L] of an FP8 query [T, H, L + R]."""
        decode = md.decode
        return self.decode(
            module,
            query,
            decode.block_table,
            decode.seq_lens,
            md.max_seq_len,
            md.causal,
            capacity,
        )

    def decode(
        self, module, query, block_table, seq_lens, max_seq_len, causal, capacity
    ):
        from tokenspeed_mla import tokenspeed_mla_decode

        requests = seq_lens.numel()
        query = query.view(requests, -1, query.shape[-2], query.shape[-1])
        softmax_scale = (
            module.impl.scale * module._q_scale_float * module._k_scale_float
        )
        out = tokenspeed_mla_decode(
            query=query,
            kv_cache=module._attn_read_kv_cache(),
            workspace_buffer=self.workspace,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.rope_dim,
            block_tables=block_table,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            softmax_scale=softmax_scale,
            output_scale=module._k_scale_float,
            enable_pdl=False,
            causal_mask=causal,
            min_split_kv=1,
            split_kv_counts=self.counts[:requests],
            max_split_kv=capacity,
        )
        return out.view(-1, out.shape[-2], out.shape[-1])

    def warm_up(self, module, requests: int, q: int):
        """Compile the planner and the split-count decode for ``requests`` requests
        of ``q`` tokens: one-token requests on the first cache block of ``module``'s
        bound cache (read only)."""
        device = self.workspace.device
        seq_lens = torch.ones(requests, dtype=torch.int32, device=device)
        block_table = torch.zeros((requests, 1), dtype=torch.int32, device=device)
        query = torch.zeros(
            (requests * q, self.heads, self.kv_lora_rank + self.rope_dim),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        capacity = self.plan(seq_lens, q)
        self.decode(module, query, block_table, seq_lens, 1, True, capacity)
