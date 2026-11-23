# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The release DSpark drafter's fused attention (Inferact/Kimi-K3.1-Pro-DSpark: five
sliding-window GQA layers of 96 query / 16 KV heads of 64, window 2048) behind
``runtime.draft.GqaAttention``. ``install_gqa_attention(layers, vllm_config,
query_rows, context_rows, strict)`` returns the prepared state whose

  forward(index, qkv, positions)  layer ``index``'s attention core on its qkv_proj
      rows [rows, q + 2 kv]: q / k RMSNorm, RoPE, the K / V insert of the query rows
      and the causal sliding-window attention in one GqaWindow launch (gqa_window):
      the o_proj input [rows, q_size]; None for a call it does not cover;
  store_context(kv, positions, slots)  the context step's K / V insert of every layer
      from the context K/V projection rows [C, L x 2 x kv] in one ContextKV launch
      (gqa_context_kv); False when not covered.

Both read and write the draft layers' own cache views exactly as the stock path
(FLASH_ATTN, the draft's backend on P and D; P writes the draft KV that NIXL moves to
D): [blocks, H, page, K | V] BF16 with the view's block stride, which is a whole BLHNC
row under the planner's 1024 / 4096-token draft blocks, or one of the S slots of a row
under the packed 128-token draft group (KVCacheTensor.block_stride = row / S). Slots
are never zeroed there: GqaWindow zeroes every tile position outside [window start,
seq_len) and masks each row's own window, so no stale byte reaches an output.

Metadata: as FlashAttentionImpl.forward reads it during the drafter graph's capture,
the layer's FlashAttentionMetadata from the forward context (seq_lens, block_table,
causal; the persistent buffers a replay rewrites) and the layer's slot mapping. Query
row i of a request of seq_len L sits at key L - q + i and attends to keys
[L - q + i - 2047, L - q + i] (FlashAttention's bottom-right causal window (2047, 0));
positions feed only RoPE, as in the stock path.

Checked at install (else nothing is installed: a strict deployment refuses, others keep
the stock attention): head 64, 6 query heads per KV head, 1 / 2 / 4 KV heads per rank
(TP16 / TP8 / TP4), scale 1/8, causal window 2048, one RMSNorm eps, neox RoPE over all
64 dims from one BF16 cos_sin_cache shared by the layers (the stock context step uses
the first layer's), no value scale, sinks, ALiBi or soft cap, BF16 KV ("auto"),
FlashAttention at DCP 1, every drafter row count a whole number of q-row requests.

GqaWindow compiles the cache page in and loads K / V tiles through a TMA descriptor of
each layer's view (device memory), so its kernels are prepared when the layers' caches
are bound (``bind_kv_cache``, before any capture): one per launch shape GQA_WINDOW
assigns to the deployment's drafter batches. The context insert takes the page and
strides at run time and compiles at install.
"""

import inspect
from dataclasses import dataclass, field

import torch
from cutlass import Float32, Int32, Int64

from vllm.logger import init_logger

from ...shapes import refuse
from ..common.host import dynamic_view, stream, tensor_maps, to_cute
from .gqa_context_kv import compile_context_kv
from .gqa_window import TB, compile_window
from ..common.resident import per_sm_fit
from .gqa_window_stream import (
    MAXP,
    compile_window_stream,
    grid_ctas,
    piece_bounds,
    smem_bytes,
    unit_slots,
)
from .gqa_window_umma import KB, compile_window_umma, kv_map_5d, umma_smem

logger = init_logger(__name__)

HEAD = 64  # head_dim of the release drafter
GROUP = 6  # query heads per KV head (96 / 16)
WINDOW = 2048  # sliding window (keys a query row reads, itself included)
KV_HEADS = (1, 2, 4)  # KV heads per rank: TP16, TP8, TP4

# GqaWindow launch shape by (KV heads per rank, query rows per request): bins of
# (largest drafter batch, (splits, key groups, ring stages, L2-ahead stages)), the last
# bin open. Splits = CTAs (one cluster) per (request, KV head): small batches spread
# the 2048-key window over a cluster, large ones fill the GPU with requests while the
# grid stays one wave (VR200: 212 SMs; shapes at a wave's edge are avoided). Measured
# with gqa_window a5e68ee2: 1 VR200 GPU, CUDA graph, full windows at 150K-token
# contexts, cold KV (benchmarks/kernels/k31_vr/dsrel/bench_gqa_attention_int.py
# --shape sweeps), us per layer:
#   (1, 7) TP16 k7: B1-4 4.3-4.7, B8 5.1, B12-16 6.0-6.2, B24-32 7.6-7.8, B64 11.1
#   (1, 3) TP16 k3: B1-8 3.3-4.4, B16 4.8, B28-38 6.5-6.8, B64 8.5, B128 12.1
#   (4, 3) TP4 k3 (C512): B1 3.7, B4-8 4.8-6.8, B16 8.6, B32 12.2, B50 14.7,
#          B64 19.3, B128 32.6
# TP4 k3 with GqaWindowUmma's S issuer warp (ahead bit 1; 2026-11-20, 1 VR200 GPU, same
# bench): B24 9.2 vs 12.0, B32 11.3 vs 12.3, B40-52 12.4-13.8 vs 14.5-15.2 (splits 2),
# B56-64 16.7-17.4 vs 19.2-19.7, B72-96 18.4-21.3 vs 21.1-22.2 (one CTA per unit);
# B106 23.5 vs 23.1 without it.
# Key groups 0: GqaWindowUmma (tcgen05, gqa_window_umma; ring stages of 64 keys), which
# streams the window at near HBM rate once the grid holds two CTAs on most SMs: TP4 k3
# B96 23.3 vs GqaWindow's best 29.6 (B64 20.4 vs 19.6, B32 12.9 vs 12.3: one CTA per
# (request, KV head) is a serial stage chain at small batches; above ~106 requests
# its 97 KB / 256-column CTAs exceed one wave).
# The TP8 rows (DFn decode, and prefill instances, which run the stock drafter) and
# TP4 k7 (no deployment) take shapes measured at the same CTA counts.
GQA_WINDOW = {
    (1, 7): ((4, (16, 4, 2, 0)), (8, (16, 2, 3, 0)), (16, (8, 4, 3, 0)), (32, (4, 4, 4, 0)),
             (0, (2, 4, 4, 0))),
    (1, 3): ((8, (16, 4, 2, 8)), (16, (8, 4, 3, 0)), (40, (4, 4, 3, 0)), (64, (8, 2, 3, 0)),
             (0, (4, 2, 3, 0))),
    # k5 (C128 / C160 1P1D llh, TP16, release_draft/k5): (1, 3)'s shapes. Its 30 query
    # rows per (request, KV head) fill the same two 16-row M tiles as k3's 18, so every
    # shape builds alike (GqaWindow asserts) and each batch keeps k3's CTA count and
    # wave. NQ-TUNE: --shape sweep at q5 (bench_gqa_attention_int.py) at B16-B48.
    (1, 5): ((8, (16, 4, 2, 8)), (16, (8, 4, 3, 0)), (40, (4, 4, 3, 0)), (64, (8, 2, 3, 0)),
             (0, (4, 2, 3, 0))),
    (2, 7): ((4, (8, 4, 3, 0)), (16, (4, 4, 4, 0)), (0, (2, 2, 3, 0))),
    (2, 3): ((4, (8, 4, 3, 0)), (32, (4, 2, 3, 0)), (0, (2, 2, 3, 0))),
    (4, 7): ((8, (8, 4, 3, 0)), (16, (8, 2, 3, 0)), (32, (4, 2, 3, 0)), (64, (2, 2, 3, 0)),
             (0, (1, 2, 3, 0))),
    (4, 3): ((2, (16, 4, 2, 8)), (8, (8, 4, 3, 0)), (16, (8, 2, 3, 0)), (52, (2, 0, 6, 2)),
             (96, (1, 0, 6, 2)), (106, (1, 0, 6, 0)), (0, (1, 2, 3, 0))),
}  # fmt: skip
# TP4 (the SP decode roles DC / DB) at DSpark k1, k2, k4, k5, k6: the measured shapes of
# the nearest measured draft length, per (request, KV head) row tiling (GROUP x k query
# rows): k1 / k2 / k4 / k5 (6-30 rows, within GqaWindowUmma's 32: group x q <= 32)
# take k3's, k6 (36 rows, above it: no GqaWindowUmma shape) takes k7's; as (1, 5) does
# for TP16 k5. Every shape is checked against stock per q (test_dspark_gqa_attention_int
# test_window_tp4_any_q). NQ-TUNE: --shape sweeps at TP4 k1 / k2 / k4 / k5 / k6.
# TP8 (DFn: TP8 = DCP8 decode, 2 KV heads per rank) at k1, k2, k4, k5, k6 the same
# way, from (2, 3) / (2, 7) (test_window_tp8_any_q).
for _k in (1, 2, 4, 5):
    GQA_WINDOW[(4, _k)] = GQA_WINDOW[(4, 3)]
    GQA_WINDOW[(2, _k)] = GQA_WINDOW[(2, 3)]
GQA_WINDOW[(4, 6)] = GQA_WINDOW[(4, 7)]
GQA_WINDOW[(2, 6)] = GQA_WINDOW[(2, 7)]
del _k


def window_options():
    """GqaWindow's rounding option for the stock drafter's arithmetic: the q / k norm
    output rounded to BF16 before RoPE, as vLLM's rms_norm and rotary_embedding kernels
    (the dispatch ``stock_dispatch_reason`` requires). Kernels without the option round
    it always."""
    if "round_norm" in inspect.signature(compile_window).parameters:
        return {"round_norm": True}
    return {}


def stock_dispatch_reason(vllm_config):
    """Why the stock drafter's q_norm / k_norm / RoPE would not run vLLM's C kernels
    (rms_norm through the IR priority's ``vllm_c``, the rotary_embedding custom op):
    the arithmetic the fused kernels implement (the drafter is never torch.compiled:
    K3GqaDSparkModel ignores compilation, so a native dispatch would run eager torch
    ops of other rounding points). None: it would."""
    import vllm.envs as envs

    if envs.VLLM_BATCH_INVARIANT:
        return "the batch-invariant RMSNorm"
    kernel = getattr(vllm_config, "kernel_config", None)
    priority = getattr(getattr(kernel, "ir_op_priority", None), "rms_norm", None)
    if priority and priority[0] != "vllm_c":
        return f"rms_norm IR priority {list(priority)} (vllm_c first: the C kernel)"
    compilation = getattr(vllm_config, "compilation_config", None)
    if compilation is not None and not compilation.is_custom_op_enabled(
        "rotary_embedding"
    ):
        return "the native rotary_embedding (custom op off)"
    return None


def window_shape(kv_heads, q, batch):
    """GQA_WINDOW's launch shape for ``batch`` requests (None: no table)."""
    bins = GQA_WINDOW.get((kv_heads, q))
    if bins is None:
        return None
    for top, shape in bins:
        if top == 0 or batch <= top:
            return shape
    return None


def _cache_key(cache):
    return cache.data_ptr(), tuple(cache.shape), tuple(cache.stride())


@dataclass
class _Bound:
    """One layer's bound cache view: its key, page, strides, TMA descriptor and the
    GqaWindow kernels by launch shape."""

    key: tuple
    page: int
    blocks: int
    block_stride: int
    head_stride: int
    descriptor: torch.Tensor
    kernels: dict = field(default_factory=dict)


class GqaAttentionState:
    """The prepared fused attention of the drafter's layers (module notes)."""

    def __init__(self, layers, q, query_rows, strict, device):
        self.layers, self.q, self.strict = list(layers), q, strict
        self.device = device
        first = self.layers[0]
        self.kv_heads = first.num_kv_heads
        self.heads = first.num_heads
        self.width = (self.heads + 2 * self.kv_heads) * HEAD
        self.rows = frozenset(r for r in query_rows if r % q == 0)
        batches = sorted({r // q for r in self.rows})
        self.shapes = {b: window_shape(self.kv_heads, q, b) for b in batches}
        self.stream_buffers(device)
        self.eps = Float32(first.q_norm.variance_epsilon)
        rope = first.rotary_emb
        self.cos_sin = rope.cos_sin_cache
        self.max_position = self.cos_sin.shape[0]
        # the context step's k_norm weights by layer (stock: _k_norm_weights)
        self.k_weights = torch.stack(
            [layer.k_norm.weight.data for layer in self.layers]
        ).contiguous()
        index = device.index
        self.context = compile_context_kv(
            len(self.layers), self.kv_heads, self.max_position, index
        )
        self.bound: dict[int, _Bound] = {}
        self.unbound: dict[int, str] = {}
        for i, layer in enumerate(self.layers):
            attention = layer.attn
            original = attention.bind_kv_cache

            def bind_kv_cache(kv_cache, _i=i, _attention=attention, _bind=original):
                _bind(kv_cache)
                self.bind(_i, _attention.kv_cache)

            attention.bind_kv_cache = bind_kv_cache
            cache = getattr(attention, "kv_cache", None)
            if isinstance(cache, torch.Tensor) and cache.numel() > 0:
                self.bind(i, cache)  # (already bound)

    def stream_buffers(self, device):
        """GqaWindowStream's partials [units, pieces, rows x 66] FP32 and per-unit
        counters (zero; each launch's last pieces reset theirs) for the batches it
        serves (launch shapes with key groups -1)."""
        batches = [b for b, s in self.shapes.items() if s is not None and s[1] < 0]
        self.pmax, self.partials, self.counters = 1, None, None
        # die split (launch shapes ahead 1): the resident grid, and the KV heads' home
        # dies (bit h: die of KV head h) of a die-localized cache layout
        self.resident, self.head_dies = None, None
        if any(s is not None and s[1] == 0 and s[3] & 1 for s in self.shapes.values()):
            from ..common.resident import ResidentGrid

            self.resident = ResidentGrid(device, 2)
        if not batches:
            return
        if self.resident is None and any(self.shapes[b][3] == 1 for b in batches):
            from ..common.resident import ResidentGrid

            self.resident = ResidentGrid(device, 2)
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        slots = unit_slots(WINDOW, self.q)
        for b in batches:
            units = b * self.kv_heads
            if self.shapes[b][3] == 1 and self.shapes[b][0] == 1:
                continue  # (die split, one unit per CTA: no pieces to merge)
            ctas = grid_ctas(units, slots, sms, self.shapes[b][0])
            per_cta, per_unit = piece_bounds(units, slots, ctas)
            if per_cta > MAXP:
                raise ValueError(
                    f"GqaWindowStream: B{b} spreads {per_cta} units over one CTA "
                    f"(at most {MAXP})"
                )
            self.pmax = max(self.pmax, per_unit)
        units = max(batches) * self.kv_heads
        words = units * self.pmax * GROUP * self.q * (HEAD + 2)
        self.partials = torch.empty(words, dtype=torch.float32, device=device)
        self.counters = torch.zeros(units, dtype=torch.int32, device=device)

    # ------------------------------------------------------------------ binding
    def cache_reason(self, cache):
        """Why GqaWindow / ContextKV cannot address this cache view (None: they can)."""
        H = self.kv_heads
        if not isinstance(cache, torch.Tensor) or cache.dim() != 4:
            return f"a cache view {getattr(cache, 'shape', None)} that is not 4-D"
        page = cache.shape[2]
        checks = (
            (cache.dtype != torch.bfloat16, f"a {cache.dtype} cache (BF16 only)"),
            (cache.device != self.device, f"a cache on {cache.device}"),
            (
                tuple(cache.shape[1:]) != (H, page, 2 * HEAD),
                f"a cache view {tuple(cache.shape)} that is not [blocks, {H}, page, "
                f"{2 * HEAD}]",
            ),
            (
                cache.stride(3) != 1 or cache.stride(2) != 2 * HEAD,
                f"cache strides {cache.stride()} (tokens of 128 contiguous values)",
            ),
            (
                H > 1 and cache.stride(1) != page * 2 * HEAD,
                f"cache strides {cache.stride()} (heads of one page each)",
            ),
            (page % TB, f"a {page}-token page (a multiple of {TB})"),
            (
                cache.data_ptr() % 16 or (cache.stride(0) * 2) % 16,
                "a cache view not 16 B aligned per block",
            ),
        )
        return next((why for failed, why in checks if failed), None)

    def bind(self, index, cache):
        """Layer ``index``'s cache view is bound: its TMA descriptor and the GqaWindow
        kernels of the deployment's launch shapes for its page."""
        reason = self.cache_reason(cache)
        if reason is not None:
            self.bound.pop(index, None)
            self.unbound[index] = reason
            logger.warning_once(
                "Kimi K3.1 GQA DSpark drafter attention: layer %d cache not covered "
                "(%s)",
                index,
                reason,
            )
            return
        import cuda.bindings.driver as cuda

        page, blocks = cache.shape[2], cache.shape[0]
        head_stride = page * 2 * HEAD if self.kv_heads == 1 else cache.stride(1)
        spec = (
            cache.data_ptr(),
            cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            (2 * HEAD, page, self.kv_heads, blocks),
            (4 * HEAD, head_stride * 2, cache.stride(0) * 2),
            (HEAD, TB, 1, 1),
            cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
            cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        )
        bound = _Bound(
            _cache_key(cache),
            page,
            blocks,
            cache.stride(0),
            head_stride,
            # (+ the 5-D map GqaWindowUmma loads a stage with, at descriptor + 128)
            tensor_maps(
                [spec] + ([kv_map_5d(cache, self.kv_heads, head_stride)]
                          if page % KB == 0 else []),
                self.device,
            ),  # fmt: skip
        )
        for shape in set(self.shapes.values()):
            if shape is not None:
                bound.kernels[shape] = self.window(page, shape)
        self.bound[index] = bound
        self.unbound.pop(index, None)
        if len(self.bound) == len(self.layers):
            logger.info_once(
                "Kimi K3.1 GQA DSpark drafter attention: GqaWindow on %d-token pages "
                "for drafter batches %s (launch shapes %s), ContextKV insert",
                page,
                tuple(sorted(self.shapes)),
                tuple(sorted({s for s in self.shapes.values() if s is not None})),
            )

    def window(self, page, shape):
        splits, kgroups, ring, ahead = shape
        if kgroups < 0:  # GqaWindowStream: the persistent stream over every SM
            if page % KB or page < 2 * KB:
                raise ValueError(
                    f"GqaWindowStream launch shape {shape} on {page}-token pages"
                )
            die_sms = self.resident.die_sms if ahead == 1 else None
            compiled = compile_window_stream(
                GROUP, self.kv_heads, self.q, page, ring, WINDOW, None,
                self.max_position, self.device.index, self.pmax,
                round_norm=window_options().get("round_norm", False), cap=splits,
                die_sms=die_sms,
            )  # fmt: skip
            smem = smem_bytes(ring, self.q)
            if die_sms is not None:
                self.resident.check(compiled, 160, smem, "GqaWindowStream die split")
            elif per_sm_fit(compiled, self.device.index, 160, smem) < 2:
                raise RuntimeError(
                    f"GqaWindowStream {shape}: two CTAs per SM do not fit"
                )
            return compiled
        if kgroups == 0:  # GqaWindowUmma (tcgen05): splits CTAs per (request, head)
            if page % KB:
                raise ValueError(
                    f"GqaWindowUmma launch shape {shape} on {page}-token pages"
                )
            # ahead bit 0: die split (resident grid, one unit per CTA on its KV
            # head's die); bit 1: the separate S issuer warp
            die_sms = self.resident.die_sms if ahead & 1 else None
            compiled = compile_window_umma(
                GROUP, self.kv_heads, self.q, page, ring, WINDOW, None,
                self.max_position, self.device.index,
                round_norm=window_options().get("round_norm", False), splits=splits,
                swarp=bool(ahead & 2), die_sms=die_sms,
            )  # fmt: skip
            if die_sms is not None:
                self.resident.check(
                    compiled, 192 if ahead & 2 else 160,
                    umma_smem(ring, self.q, page, WINDOW, True),
                    "GqaWindowUmma die split",
                )  # fmt: skip
            return compiled
        return compile_window(
            GROUP, self.kv_heads, self.q, page, splits, kgroups, ring, WINDOW, None,
            self.max_position, self.device.index, ahead=ahead, **window_options(),
        )  # fmt: skip

    def _bound(self, index):
        """Layer ``index``'s binding if it is the layer's current cache, else None."""
        bound = self.bound.get(index)
        if bound is None:
            return None
        cache = self.layers[index].attn.kv_cache
        return bound if _cache_key(cache) == bound.key else None

    # ------------------------------------------------------------------ calls
    def _decline(self, rows, cause):
        logger.warning_once(
            "Kimi K3.1 GQA DSpark drafter attention: %d rows not covered (%s)",
            rows,
            cause,
        )
        return None

    def forward(self, index, qkv, positions):
        """Layer ``index``'s attention core [rows, q_size], or None (not covered)."""
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

        rows, q = qkv.shape[0], self.q
        if rows not in self.rows:
            return self._decline(rows, "no prepared drafter batch")
        shape = self.shapes.get(rows // q)
        bound = self._bound(index)
        if shape is None or bound is None or shape not in bound.kernels:
            cause = self.unbound.get(index, "no GqaWindow for the bound cache")
            return self._decline(rows, cause)
        if not is_forward_context_available():
            return self._decline(rows, "no forward context")
        attention = self.layers[index].attn
        context = get_forward_context()
        metadata, slots = context.attn_metadata, context.slot_mapping
        if isinstance(metadata, list):
            metadata = metadata[0] if metadata else None
        if isinstance(metadata, dict):
            metadata = metadata.get(attention.layer_name)
        slots = slots.get(attention.layer_name) if isinstance(slots, dict) else None
        md = metadata
        checks = (
            (not isinstance(md, FlashAttentionMetadata), f"metadata {type(md)}"),
            (md is not None and md.causal is not True, "non-causal metadata"),
            (
                md is not None and md.seq_lens.shape[0] * q != rows,
                "requests x draft length != rows",
            ),
            (md is not None and md.max_query_len != q, "query rows != draft length"),
            (
                md is not None
                and (md.seq_lens.dtype != torch.int32 or md.block_table.dtype != torch.int32
                     or md.block_table.dim() != 2 or md.block_table.stride(1) != 1),
                "seq_lens / block_table not int32 [requests, blocks]",
            ),  # fmt: skip
            (
                md is not None
                and (md.mm_prefix_query_range_tensor is not None
                     or md.rswa_prefix_lens is not None),
                "prefix-LM or reference sliding-window metadata",
            ),  # fmt: skip
            (
                slots is None or slots.dtype != torch.int64 or slots.shape[0] < rows,
                "no int64 slot mapping for the rows",
            ),
            (
                positions.dtype != torch.int64 or positions.shape[0] < rows,
                f"positions {positions.dtype} {tuple(positions.shape)}",
            ),
            (
                qkv.dtype != torch.bfloat16
                or qkv.shape[1] != self.width
                or qkv.stride(1) != 1,
                f"qkv rows {tuple(qkv.shape)} {qkv.dtype}",
            ),
        )
        reason = next((why for failed, why in checks if failed), None)
        if reason is not None:
            return self._decline(rows, reason)
        layer = self.layers[index]
        cache = attention.kv_cache
        out = qkv.new_empty((rows, self.heads * HEAD))
        extra = ()
        if shape[1] < 0:
            extra = (Int64(self.partials.data_ptr()), Int64(self.counters.data_ptr()))
        if shape[1] == 0 and shape[3] & 1:
            ones = bin(self.head_dies or 0).count("1")
            per_die = max(ones, self.kv_heads - ones)
            if rows // self.q * per_die > 2 * min(self.resident.die_sms):
                raise ValueError(
                    f"GqaWindowUmma die split: {rows // self.q} requests exceed one "
                    "unit per CTA of the resident grid"
                )
        if shape[1] <= 0 and shape[3] & 1:
            if self.head_dies is None:
                raise RuntimeError(
                    "GqaWindowStream die split: no KV head -> die map for the cache"
                )
            extra += tuple(Int64(a) for a in self.resident.addresses())
            extra += (Int32(self.head_dies),)
        bound.kernels[shape](
            dynamic_view(qkv),
            to_cute(positions[:rows], 8),
            to_cute(slots[:rows], 8),
            to_cute(md.seq_lens, 4),
            dynamic_view(md.block_table, 4),
            to_cute(self.cos_sin),
            to_cute(layer.q_norm.weight.data),
            to_cute(layer.k_norm.weight.data),
            dynamic_view(out),
            Int64(bound.descriptor.data_ptr() + (128 if shape[1] <= 0 else 0)),
            Int64(cache.data_ptr()),
            Int64(bound.block_stride),
            Int64(bound.head_stride),
            Int32(bound.blocks),
            self.eps,
            *extra,
            stream(self.device),
        )
        return out

    def store_context(self, kv, positions, slots):
        """The context rows' K / V insert of every layer; False when not covered."""
        rows = kv.shape[0]
        bounds = [self._bound(i) for i in range(len(self.layers))]
        if any(b is None for b in bounds):
            self._decline(rows, "a layer's cache is not bound")
            return False
        first = bounds[0]
        checks = (
            (
                any(
                    (b.page, b.block_stride, b.head_stride)
                    != (first.page, first.block_stride, first.head_stride)
                    for b in bounds
                ),
                "layers whose cache views differ in page or strides",
            ),
            (
                kv.dtype != torch.bfloat16
                or kv.dim() != 2
                or kv.shape[1] != len(self.layers) * 2 * self.kv_heads * HEAD
                or kv.stride(1) != 1,
                f"context K/V rows {tuple(kv.shape)} {kv.dtype}",
            ),
            (
                positions.dtype != torch.int64 or positions.shape[0] < rows,
                f"context positions {positions.dtype} {tuple(positions.shape)}",
            ),
            (
                slots.dtype != torch.int64 or slots.shape[0] < rows,
                f"context slots {slots.dtype} {tuple(slots.shape)}",
            ),
        )
        reason = next((why for failed, why in checks if failed), None)
        if reason is not None:
            self._decline(rows, reason)
            return False
        if rows == 0:
            return True
        bases = [
            Int64(self.layers[i].attn.kv_cache.data_ptr())
            for i in range(len(self.layers))
        ]
        self.context(
            dynamic_view(kv),
            to_cute(positions[:rows], 8),
            to_cute(slots[:rows], 8),
            to_cute(self.cos_sin),
            to_cute(self.k_weights),
            *bases,
            Int64(first.page),
            Int64(first.block_stride),
            Int64(first.head_stride),
            self.eps,
            stream(self.device),
        )
        return True


def layer_reason(layers):
    """Why the fused attention cannot serve these drafter layers (None: it can)."""
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.attention.backend import AttentionType

    if not layers:
        return "no draft layers"
    first = layers[0]
    rope = getattr(first, "rotary_emb", None)
    cos_sin = getattr(rope, "cos_sin_cache", None)
    for i, layer in enumerate(layers):
        attention = getattr(layer, "attn", None)
        impl = getattr(attention, "impl", None)
        checks = (
            (layer.head_dim != HEAD, f"head_dim {layer.head_dim}"),
            (layer.num_kv_heads not in KV_HEADS, f"{layer.num_kv_heads} KV heads per rank"),
            (layer.num_heads != GROUP * layer.num_kv_heads, f"{layer.num_heads} query heads"),
            (
                (layer.num_heads, layer.num_kv_heads) != (first.num_heads, first.num_kv_heads),
                "layers of different head counts",
            ),
            (abs(layer.scaling - HEAD**-0.5) > 0, f"scale {layer.scaling}"),
            (getattr(layer, "v_scale", None) is not None, "a value scale"),
            (getattr(layer, "attention_sink_bias", None) is not None, "attention sinks"),
            (layer.sliding_window != WINDOW, f"sliding window {layer.sliding_window}"),
            (not getattr(layer, "causal", False), "non-causal attention"),
            (
                layer.q_norm.variance_epsilon != layer.k_norm.variance_epsilon
                or layer.q_norm.variance_epsilon != first.q_norm.variance_epsilon,
                "q / k norms of different eps",
            ),
            (
                layer.q_norm.weight.dtype != torch.bfloat16
                or layer.k_norm.weight.dtype != torch.bfloat16
                or layer.q_norm.weight.shape != (HEAD,),
                "norm weights that are not BF16 [64]",
            ),
            (
                getattr(layer, "rotary_emb", None) is not rope
                and not (
                    isinstance(cos_sin, torch.Tensor)
                    and torch.equal(layer.rotary_emb.cos_sin_cache, cos_sin)
                ),
                "layers of different RoPE tables",
            ),
            (not isinstance(impl, FlashAttentionImpl), f"attention {type(impl).__name__}"),
            (
                impl is not None
                and getattr(impl, "sliding_window", None) != (WINDOW - 1, 0),
                f"window {getattr(impl, 'sliding_window', None)}",
            ),
            (
                impl is not None
                and (impl.alibi_slopes is not None or impl.logits_soft_cap
                     or impl.sinks is not None),
                "ALiBi, soft cap or sinks",
            ),  # fmt: skip
            (impl is not None and impl.kv_cache_dtype != "auto", "a quantized KV cache"),
            (impl is not None and impl.dcp_world_size != 1, "a DCP-sharded draft cache"),
            (
                impl is not None and impl.attn_type != AttentionType.DECODER,
                "a non-decoder attention",
            ),
            (
                impl is not None and impl.kv_sharing_target_layer_name is not None,
                "a shared KV cache",
            ),
            (
                impl is not None and abs(impl.scale - HEAD**-0.5) > 0,
                f"kernel scale {getattr(impl, 'scale', None)}",
            ),
        )  # fmt: skip
        reason = next((why for failed, why in checks if failed), None)
        if reason is not None:
            return f"layer {i}: {reason}"
    checks = (
        (
            rope is None
            or not rope.is_neox_style
            or rope.head_size != HEAD
            or rope.rotary_dim != HEAD,
            "RoPE that is not neox over all 64 dims",
        ),
        (
            not isinstance(cos_sin, torch.Tensor)
            or cos_sin.dtype != torch.bfloat16
            or cos_sin.dim() != 2
            or cos_sin.shape[1] != HEAD
            or not cos_sin.is_contiguous(),
            "a RoPE table that is not BF16 [positions, 64]",
        ),
    )
    return next((why for failed, why in checks if failed), None)


def install_gqa_attention(layers, vllm_config, query_rows, context_rows, strict):
    """The fused attention state of the drafter's attention modules ``layers``
    (DFlashQwen3Attention), or None when it cannot serve them (strict: refuses).
    ``query_rows`` / ``context_rows``: the drafter graphs' row counts."""
    spec = vllm_config.speculative_config
    q = spec.num_speculative_tokens if spec is not None else 0
    reason = layer_reason(layers) or stock_dispatch_reason(vllm_config)
    if reason is None and (not q or any(r % q for r in query_rows)):
        reason = f"drafter rows {sorted(query_rows)} not whole {q}-row requests"
    if reason is None:
        max_len = vllm_config.model_config.max_model_len
        if layers[0].rotary_emb.cos_sin_cache.shape[0] < max_len:
            reason = f"a RoPE table shorter than max_model_len {max_len}"
    if reason is None:
        heads = layers[0].num_kv_heads
        missing = [r for r in query_rows if window_shape(heads, q, r // q) is None]
        if missing:
            reason = f"no GQA_WINDOW launch shape for rows {missing}"
    if reason is not None:
        if strict:
            refuse("DSpark drafter attention (fused GQA)", reason)
        logger.warning_once(
            "Kimi K3.1 GQA DSpark drafter attention keeps stock (%s)", reason
        )
        return None
    device = layers[0].q_norm.weight.device
    state = GqaAttentionState(layers, q, query_rows, strict, device)
    logger.info_once(
        "Kimi K3.1 GQA DSpark drafter attention: %d KV heads per rank, q %d, drafter "
        "rows %s, context rows %s",
        state.kv_heads,
        q,
        tuple(sorted(query_rows)),
        tuple(sorted(context_rows)),
    )
    return state
