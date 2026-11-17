# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The release DSpark drafter's context K/V insert (Inferact/Kimi-K3.1-Pro-DSpark:
five sliding-window GQA layers, head 64): ``gqa_attention``'s ``store_context``.

Stock (qwen3_dflash.DFlashQwen3Model.precompute_and_store_context_kv on the context
K/V projection rows, token-major [C, L, 2, H, 64]): a permute copy to
[2, L, C, H, 64]; ops.rms_norm of every layer's K rows with that layer's k_norm
weight (the grouped [L, 64] weight, eps of the first layer's q_norm); one
ops.rotary_embedding of all K rows at the context positions repeated per layer (neox,
64 rotary dims, the first layer's BF16 cos_sin_cache); then per layer FlashAttention's
do_kv_cache_update (reshape_and_cache_flash) of K and V at the context slots, slot -1
skipped. Eight launches, five of them inserts.

``ContextKV`` is one launch over every (row, layer, KV head): 8 lanes per item (lane j
holds dims 8j..8j+7 of its K and of its V), 16 items per 128-thread CTA. Each lane
loads its 16 B of K and of V, its 4 words of the layer's k_norm weight (before the PDL
wait: a constant) and of the row's cos / sin (lanes j and j + 4 hold the neox pair
d, d + 32), normalizes and rotates K, and stores K and V at the row's slot in the
layer's cache view [blocks, H, page, K | V] (block = slot // page, offset = slot %
page; the block stride is the view's own: a BLHNC row, or a slot of one under the
packed draft group). Slot -1 stores nothing.

Arithmetic (the stock kernels', so the inserted bytes are the stock bytes):
  RMSNorm (vLLM's rms_norm with one 8-thread block per 64 values): each lane an FMA
  chain of its 8 squares from 0, the 8 partial sums added as cub's warp reduction
  pairs them (the xor butterfly over the 8 lanes: the same tree), r = rsqrtf(sum / 64
  + eps) (div.rn, rsqrt.approx), y = bf16((x * r) * w);
  RoPE (vLLM's rotary_embedding, FP32 from the BF16 y and the BF16 cos / sin, as nvcc
  contracts it): bf16(fma(x1, cos, -(x2 * sin))), bf16(fma(x1, sin, x2 * cos));
  V: the projection's bits.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ..common.host import dynamic_fake, fake
from ..common.primitives import (
    asm,
    bf16,
    bf16x2,
    fma,
    launch,
    load_words,
    store_words,
)

HEAD = 64  # head_dim of the release drafter
LANES = 8  # lanes per (row, layer, KV head): 8 of the 64 values each
THREADS = 128
ITEMS = THREADS // LANES  # (row, layer, KV head) items per CTA


@asm(Float32, pure=True)
def rsqrt_approx(value):
    """rsqrtf (MUFU.RSQ), as vLLM's rms_norm kernel."""
    return "rsqrt.approx.f32 $0,$1;", "=f,f", (Float32(value),)


@asm(Float32, pure=True)
def div_rn(a, b):
    return "div.rn.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Int64)
def load_i64(pointer):
    """An L2 (``.cg``) load after the PDL wait (side-effecting: never hoisted)."""
    return "ld.global.cg.b64 $0, [$1];", "=l,l,~{memory}", (pointer,)


@cute.jit
def norm_rope(x, weights, cos, sin, eps, upper):
    """k_norm + neox RoPE of one 64-value head held by an aligned group of 8 lanes (8
    values each; ``upper``: dims 32..63). ``x``: the lane's 8 FP32 values; ``weights``,
    ``cos``, ``sin``: 4 BF16x2 words each (cos / sin of dims 8 (j % 4)..). Returns the
    lane's 4 BF16x2 words."""
    total = Float32(0)
    for e in cutlass.range_constexpr(8):
        total = fma(x[e], x[e], total)
    for offset in (1, 2, 4):
        total = total + cute.arch.shuffle_sync_bfly(total, offset)
    scale = rsqrt_approx(div_rn(total, Float32(HEAD)) + eps)
    normed = []
    for k in cutlass.range_constexpr(4):
        low = x[2 * k] * scale * bf16(weights[k], False)
        high = x[2 * k + 1] * scale * bf16(weights[k], True)
        normed.append(bf16x2(low, high))
    partner = [Uint32(cute.arch.shuffle_sync_bfly(Int32(w), 4)) for w in normed]
    out = []
    for k in cutlass.range_constexpr(4):
        pair = []
        for half in (False, True):
            mine = bf16(normed[k], half)
            other = bf16(partner[k], half)
            c, s = bf16(cos[k], half), bf16(sin[k], half)
            # lanes 0-3 (x1 = mine, x2 = other): x1 cos - x2 sin
            first = fma(mine, c, -(other * s))
            # lanes 4-7 (x1 = other, x2 = mine): x2 cos + x1 sin
            second = fma(other, s, mine * c)
            value = first
            if upper:
                value = second
            pair.append(value)
        out.append(bf16x2(pair[0], pair[1]))
    return out


class ContextKV:
    """The context step's K/V insert for ``layers`` drafter layers of ``kv_heads`` KV
    heads per rank (module notes), any context row count (row-dynamic views).

    Args: kv [C, layers x 2 x kv_heads x 64] BF16, positions [C] int64, slots [C]
    int64, cos_sin [max_position, 64] BF16, k_norm weights [layers, 64] BF16, the
    layers' cache view base addresses (``layers`` Int64), page (tokens per block), the
    views' block and head strides (elements), eps; then the stream."""

    def __init__(self, layers, kv_heads):
        self.layers, self.kv_heads = layers, kv_heads

    @cute.jit
    def __call__(self, *args):
        items = args[0].shape[0] * (self.layers * self.kv_heads)
        launch(self.kernel, args, (items + ITEMS - 1) // ITEMS, THREADS)

    @cute.kernel
    def kernel(self, args):
        kv, positions, slots, cos_sin, weights = args[0:5]
        L: cutlass.Constexpr = self.layers
        H: cutlass.Constexpr = self.kv_heads
        caches = args[5 : 5 + L]
        page, block_stride, head_stride, eps = args[5 + L : 9 + L]
        bid, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        j = tid % LANES
        items = kv.shape[0] * (L * H)
        item = bid * ITEMS + tid // LANES
        live = item < items
        # (a tail group reads the last item: the 8-lane shuffles stay converged)
        item = min(item, items - 1)
        row = item // (L * H)
        layer = (item % (L * H)) // H
        head = item % H
        # the layer's k_norm weight words: constant, read before the PDL wait
        w = load_words(weights.iterator + layer * HEAD + 8 * j, count=4)
        cute.arch.griddepcontrol_wait()
        source = kv.iterator + row * kv.layout.stride[0]
        source = source + (layer * 2 * H + head) * HEAD + 8 * j
        kw = load_words(source, count=4, mutable=True)
        vw = load_words(source + H * HEAD, count=4, mutable=True)
        position = load_i64(positions.iterator + row)
        slot = load_i64(slots.iterator + row)
        rope = cos_sin.iterator + Int32(position) * cos_sin.layout.stride[0]
        rope = rope + 8 * (j % 4)
        cos = load_words(rope, count=4)
        sin = load_words(rope + HEAD // 2, count=4)
        x = [bf16(kw[e // 2], e % 2 == 1) for e in range(8)]
        roped = norm_rope(x, w, cos, sin, eps, j >= 4)
        if live & (slot >= 0):
            base = Int64(0)
            for i in cutlass.range_constexpr(L):
                if layer == i:
                    base = caches[i]
            block = slot // page
            offset = slot - block * page
            element = block * block_stride + Int64(head) * head_stride
            element = element + offset * (2 * HEAD) + Int64(8 * j)
            address = base + element * 2
            store_words(address, tuple(roped))
            store_words(address + 2 * HEAD, tuple(vw))
        cute.arch.griddepcontrol_launch_dependents()


@cache
def compile_context_kv(layers, kv_heads, max_position, device):
    """The context K/V insert for any row count (row-dynamic views)."""
    i64 = torch.int64
    args = (
        dynamic_fake(2),  # kv [C, layers x 2 x kv_heads x 64]
        fake((cute.sym_int(),), None, i64, 8),  # positions
        fake((cute.sym_int(),), None, i64, 8),  # slots
        fake((max_position, HEAD)),  # cos_sin (BF16)
        fake((layers, HEAD)),  # k_norm weights
        *(Int64(0) for _ in range(layers)),  # the layers' cache view bases
        Int64(0),  # page (tokens per block)
        Int64(0),  # block stride, elements
        Int64(0),  # head stride, elements
        Float32(1e-5),  # eps
    )
    with torch.accelerator.device_index(device):
        return cute.compile(ContextKV(layers, kv_heads), *args, make_fake_stream())
