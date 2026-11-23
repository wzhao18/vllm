# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The release DSpark drafter's fused attention (cutedsl.draft.gqa_attention:
``install_gqa_attention``'s state over GqaWindow and ContextKV) against the stock path
on one Rubin GPU, at the per-rank shapes of TP16 (6 query / 1 KV heads), TP8 (12 / 2)
and TP4 (24 / 4), k3 and k7, drafter batches 1-128.

Stock, as production runs it (decode render logs: custom_ops ['all'], IR rms_norm
priority ['vllm_c', ...]): ``dspark_gqa.stock_attention_core`` (q_norm / k_norm by
vLLM's rms_norm kernel, rotary_embedding, the Attention layer's FlashAttention
do_kv_cache_update + FlashAttentionImpl.forward over the layer's metadata and slot
mapping), and ``DFlashQwen3Model.precompute_and_store_context_kv`` itself for the
context step (its projection replaced by the given K/V rows).

Both draft-cache layouts the stock path writes: "rows" (the current planner: one draft
block per BLHNC row, 1024 tokens, 4096 at TP4, the layers' pages inside a wider row)
and "slots" (the packed draft group: 128-token blocks as equal slots of a row, block
stride row / S, never zeroed). Contexts below, at and past the window, window starts
on and off block / stage edges, random block tables (evicted pages = the null block),
stale finite bytes everywhere else in the cache (NaN in the dedicated test). Checks:
the whole cache buffer bit-identical to stock's after the inserts, and the attention
rows as close to the FP64 reference as stock's are (BF16 noise).
"""

import math
from types import MethodType, SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass")

G, HEAD, WINDOW, LAYERS = 6, 64, 2048, 5
EPS = 1e-5
MAX_POS = 1 << 20
TP_HEADS = {16: 1, 8: 2, 4: 4}
BATCHES = (1, 2, 3, 4, 7, 8, 12, 16, 28, 38, 50, 64, 96, 128)
# keys before the query rows: inside, at and past the window, at block / stage edges
CONTEXTS = (
    1, 5, 31, 32, 33, 100, 1023, 1024, 2016, 2040, 2041, 2047, 2048, 2049, 2050,
    3071, 4096, 4097, 50000, 150000, MAX_POS - 16,
)  # fmt: skip
# window starts lo = context - 2047 with lo % 32 in 26..31: the first 32-key stage holds
# 1-6 window keys and the query rows' own window starts (lo + i) cross the stage edge
WINDOW_EDGES = tuple(
    WINDOW - 1 + 32 * m + r
    for m, r in ((0, 26), (1, 27), (3, 28), (20, 29), (50, 30), (1000, 31), (4000, 26),
                 (30000, 31))
)  # fmt: skip


def _rubin():
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (10, 7)


pytestmark = pytest.mark.skipif(not _rubin(), reason="Rubin (SM107) only")


def _device():
    return torch.device("cuda", 0)


def _rope_table(device):
    inv = 1.0 / (1e4 ** (torch.arange(0, HEAD, 2, dtype=torch.float) / HEAD))
    t = torch.arange(MAX_POS, dtype=torch.float)
    freqs = torch.einsum("i,j->ij", t, inv)
    return torch.cat((freqs.cos(), freqs.sin()), -1).to(torch.bfloat16).to(device)


_ROPE = {}


def rope_table():
    if "t" not in _ROPE:
        _ROPE["t"] = _rope_table(_device())
    return _ROPE["t"]


# ------------------------------------------------------------- stock stand-ins
class _Norm:
    """vLLM's RMSNorm as production dispatches it (IR rms_norm, vllm_c impl)."""

    def __init__(self, weight):
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.variance_epsilon = EPS

    def __call__(self, x):
        import vllm._custom_ops as ops

        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        ops.rms_norm(out, x, self.weight.data, EPS)
        return out


class _Rope:
    """RotaryEmbedding.forward_cuda (ops.rotary_embedding, in place)."""

    def __init__(self, table):
        self.cos_sin_cache = table
        self.head_size = self.rotary_dim = HEAD
        self.is_neox_style = True

    def __call__(self, positions, q, k):
        import vllm._custom_ops as ops

        ops.rotary_embedding(positions, q, k, HEAD, self.cos_sin_cache, True)
        return q, k


class _Attention:
    """The Attention layer's forward (unified_kv_cache_update, then
    unified_attention_with_output) over FlashAttentionImpl, reading the forward
    context; ``bind_kv_cache`` as AttentionLayerBase's."""

    def __init__(self, name, heads, kv_heads, device):
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

        self.layer_name = name
        self.num_heads, self.num_kv_heads = heads, kv_heads
        self.impl = FlashAttentionImpl(
            heads, HEAD, HEAD**-0.5, kv_heads, None, WINDOW, "auto"
        )
        one = torch.ones((), dtype=torch.float32, device=device)
        self._q_scale = self._k_scale = self._v_scale = one
        self.kv_cache = torch.tensor([])

    def bind_kv_cache(self, kv_cache):
        self.kv_cache = kv_cache

    def __call__(self, q, k, v):
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
        metadata = context.attn_metadata[self.layer_name]
        slots = context.slot_mapping[self.layer_name]
        rows = q.shape[0]
        out = torch.empty((rows, self.num_heads * HEAD), dtype=q.dtype, device=q.device)
        query = q.view(-1, self.num_heads, HEAD)
        key = k.view(-1, self.num_kv_heads, HEAD)
        value = v.view(-1, self.num_kv_heads, HEAD)
        self.impl.do_kv_cache_update(self, key, value, self.kv_cache, slots)
        self.impl.forward(
            self, query, key, value, self.kv_cache, metadata,
            output=out.view(-1, self.num_heads, HEAD),
        )  # fmt: skip
        return out


def make_layers(kv_heads, gen):
    """Five drafter attention stand-ins (DFlashQwen3Attention's attributes and stock
    callables), random BF16 norm weights, one RoPE table."""
    device = _device()
    heads = G * kv_heads
    rope = _Rope(rope_table())
    layers = []
    for i in range(LAYERS):
        w = lambda: (torch.rand(HEAD, generator=gen, device=device) + 0.5).bfloat16()  # noqa: E731
        layers.append(
            SimpleNamespace(
                head_dim=HEAD, num_heads=heads, num_kv_heads=kv_heads,
                q_size=heads * HEAD, kv_size=kv_heads * HEAD, scaling=HEAD**-0.5,
                v_scale=None, attention_sink_bias=None, sliding_window=WINDOW,
                causal=True, q_norm=_Norm(w()), k_norm=_Norm(w()), rotary_emb=rope,
                attn=_Attention(f"draft.layers.{i}.self_attn.attn", heads, kv_heads, device),
            )
        )  # fmt: skip
    return layers


def install(layers, q, batches):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft.gqa_attention import (  # noqa: E501
        install_gqa_attention,
    )

    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=q),
        model_config=SimpleNamespace(max_model_len=MAX_POS),
    )
    rows = [b * q for b in batches]
    state = install_gqa_attention(layers, config, rows, [b * (q + 1) for b in batches], True)
    assert state is not None
    return state


# ------------------------------------------------------------- the cache
class DraftCache:
    """One backing buffer and the five layers' views [blocks, H, page, 128] in the
    ``layout`` the stock path writes; ``blocks`` block ids (0 = the null block)."""

    def __init__(self, layout, kv_heads, blocks, gen, fill="finite"):
        device = _device()
        self.layout, self.H = layout, kv_heads
        self.page = page_of(layout, kv_heads)
        layer_page = kv_heads * self.page * 2 * HEAD  # elements
        if layout == "rows":
            # a BLHNC row: 256 B of another group's bytes, then the draft layers' pages
            stride, first = LAYERS * layer_page + 128, 128
            size = blocks * stride
        else:
            # S slots per row, each the 5 layers' pages + 128 B of alignment pad
            slots = 6
            stride, first = LAYERS * layer_page + 64, 0
            rows = -(-blocks // slots)
            size = rows * slots * stride
            blocks = rows * slots
        self.blocks, self.stride = blocks, stride
        self.buffer = torch.empty(size, dtype=torch.bfloat16, device=device)
        if fill == "nan":
            self.buffer.view(torch.int16).fill_(0x7FC0)
        else:
            self.buffer.normal_(0, 30, generator=gen)  # stale finite bytes
        self.first, self.layer_page = first, layer_page

    def view(self, layer, buffer=None):
        buffer = self.buffer if buffer is None else buffer
        shape = (self.blocks, self.H, self.page, 2 * HEAD)
        strides = (self.stride, self.page * 2 * HEAD, 2 * HEAD, 1)
        offset = self.first + layer * self.layer_page
        return buffer.as_strided(shape, strides, offset)


# the planner's deployment whose draft views a KV head count gets (slots per row)
PLANNER_POINT = {1: "D_TP16_DCP16", 2: "P_TP8_DCP8", 4: "C512_TP4_DCP4"}  # 72, 48, 87
NAN_BITS = 0x7FC0


class PlannerCache:
    """The draft group's five layer views exactly as the worker builds them for the
    planned K3.1 deployment (GQADraftKVCacheGrouping, get_kv_cache_config_from_groups,
    allocate_kv_cache: 128-token draft blocks as S equal slots of a BLHNC row, block
    ID = slot ID), over a pool of whole rows; every byte stale (NaN, or finite)."""

    def __init__(self, kv_heads, blocks, gen, fill="nan"):
        from tests.models.kimi_k3.test_gqa_draft_kv_plan import (
            LAYOUT,
            POINTS,
            _config,
            _draft_group,
            _plan,
        )
        from vllm.v1.core.kv_cache_planning import (
            _get_kv_cache_bytes_per_block,
            get_kv_cache_config_from_groups,
        )
        from vllm.v1.worker.utils import allocate_kv_cache

        device = _device()
        point = PLANNER_POINT[kv_heads]
        groups = _plan(point, "release")
        draft = _draft_group(groups)
        S = draft.blocks_per_pool_block
        row_bytes = _get_kv_cache_bytes_per_block(groups, LAYOUT)
        rows = -(-blocks // S) + 1
        config = get_kv_cache_config_from_groups(
            _config(POINTS[point][1]), groups, rows * row_bytes
        )
        views = allocate_kv_cache(config, device, LAYOUT)
        self.views = [views[name] for name in draft.layer_names]
        assert len(self.views) == LAYERS
        storage = self.views[0].untyped_storage()
        self.buffer = torch.empty(0, dtype=torch.bfloat16, device=device)
        self.buffer.set_(storage, 0, (storage.nbytes() // 2,), (1,))
        if fill == "nan":
            self.buffer.view(torch.int16).fill_(NAN_BITS)
        else:
            self.buffer.normal_(0, 30, generator=gen)
        self.layout, self.H, self.page = "planner", kv_heads, 128
        self.blocks, self.slots = rows * S, S
        v = self.views[0]
        assert tuple(v.shape) == (self.blocks, kv_heads, self.page, 2 * HEAD)
        assert v.stride(0) * 2 * S == row_bytes

    def view(self, layer, buffer=None):
        buffer = self.buffer if buffer is None else buffer
        v = self.views[layer]
        return buffer.as_strided(v.shape, v.stride(), v.storage_offset())


def batch_case(B, q, page, blocks_free, gen, contexts=None, seed=None):
    """seq_lens, positions, block table (window pages only; the rest the null block),
    and the query slots of B requests (contexts drawn with ``seed``, default B)."""
    device = _device()
    pool = CONTEXTS + WINDOW_EDGES
    seed = B if seed is None else seed
    order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed))
    ctx = contexts or [pool[int(order[b % len(pool)])] for b in range(B)]
    lengths = [c + q for c in ctx]
    width = max(-(-L // page) for L in lengths) + 2
    table = torch.zeros((B, width), dtype=torch.int32)
    pick = iter(blocks_free.tolist())
    for b, L in enumerate(lengths):
        lo = max(0, L - q - (WINDOW - 1))
        for p in range(lo // page, (L - 1) // page + 1):
            table[b, p] = next(pick)
    pos = torch.cat([torch.arange(L - q, L) for L in lengths])
    req = torch.arange(B).repeat_interleave(q)
    slots = table[req, pos // page].long() * page + pos % page
    return (
        torch.tensor(lengths, dtype=torch.int32, device=device),
        pos.to(device),
        table.to(device),
        slots.to(device),
        lengths,
    )


def fill_context(cache, views, lengths, table, q, gen):
    """The context K / V of every request's window keys before its query rows (the
    previous steps' inserts), random BF16, in every layer's view."""
    page = cache.page
    for b, L in enumerate(lengths):
        lo = max(0, L - q - (WINDOW - 1))
        if L - q <= lo:
            continue
        pos = torch.arange(lo, L - q, device=table.device)
        blk = table[b, pos // page].long()
        for v in views:
            vals = torch.randn((pos.numel(), cache.H, 2 * HEAD), generator=gen,
                               device=table.device).bfloat16()  # fmt: skip
            v[blk, :, pos % page, :] = vals


def metadata(seq_lens, table, slots, q):
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    B = seq_lens.shape[0]
    return FlashAttentionMetadata(
        num_actual_tokens=B * q,
        max_query_len=q,
        query_start_loc=torch.arange(0, (B + 1) * q, q, dtype=torch.int32,
                                     device=seq_lens.device),  # fmt: skip
        max_seq_len=int(seq_lens.max()),
        seq_lens=seq_lens,
        block_table=table,
        slot_mapping=slots,
        causal=True,
    )


def forward_context(layers, md, slots):
    from vllm.forward_context import ForwardContext

    names = [layer.attn.layer_name for layer in layers]
    return ForwardContext(
        no_compile_layers={},
        attn_metadata={n: md for n in names},
        slot_mapping={n: slots for n in names},
    )


def exact_rows(layer, view, qkv, positions, seq_lens, table, q):
    """The FP64 attention rows from the inserted cache (q normalized and rotated by
    the stock kernels)."""
    rows = qkv.shape[0]
    H, heads = layer.num_kv_heads, layer.num_heads
    qn = layer.q_norm(qkv[:, : heads * HEAD].view(rows, heads, HEAD)).view(rows, -1)
    kn = layer.k_norm(qkv[:, heads * HEAD : (heads + H) * HEAD].view(rows, H, HEAD))
    layer.rotary_emb(positions, qn, kn.view(rows, -1).clone())
    B, W = seq_lens.shape[0], WINDOW - 1 + q
    page = view.shape[2]
    L = seq_lens.long()
    lo = (L - q - (WINDOW - 1)).clamp(min=0)
    keys = lo[:, None] + torch.arange(W, device=L.device)[None, :]  # [B, W]
    valid = keys < L[:, None]
    keys = torch.minimum(keys, L[:, None] - 1)
    blk = table.long().gather(1, keys // page)
    kv = view[blk, :, keys % page, :].double()  # [B, W, H, 128]
    K, V = kv[..., :HEAD], kv[..., HEAD:]
    qd = qn.view(B, q, H, G, HEAD).double()
    s = torch.einsum("bihgd,bwhd->bhigw", qd, K) * HEAD**-0.5
    qpos = L[:, None] - q + torch.arange(q, device=L.device)[None, :]  # [B, q]
    keyp = lo[:, None] + torch.arange(W, device=L.device)[None, :]
    see = valid[:, None, :] & (keyp[:, None, :] <= qpos[:, :, None])
    see &= keyp[:, None, :] >= qpos[:, :, None] - (WINDOW - 1)  # [B, q, W]
    s = s.masked_fill(~see[:, None, :, None, :], float("-inf"))
    p = torch.softmax(s, -1)
    o = torch.einsum("bhigw,bwhd->bihgd", p, V)
    return o.reshape(rows, heads * HEAD)


# ------------------------------------------------------------- context insert
def stock_context(layers, kv, positions, slots, caches):
    """DFlashQwen3Model.precompute_and_store_context_kv with its projection replaced
    by the given token-major K/V rows."""
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    C = kv.shape[0]
    H = layers[0].num_kv_heads
    attn = [
        SimpleNamespace(kv_cache=c, impl=layer.attn.impl, _k_scale=layer.attn._k_scale,
                        _v_scale=layer.attn._v_scale)  # fmt: skip
        for layer, c in zip(layers, caches)
    ]
    fake = SimpleNamespace(
        _num_attn_layers=LAYERS, _kv_size=H * HEAD, _head_dim=HEAD, _num_kv_heads=H,
        _k_norm_weights=torch.stack([la.k_norm.weight.data for la in layers]),
        _rms_norm_eps=layers[0].q_norm.variance_epsilon,
        _rope_cos_sin_cache=layers[0].rotary_emb.cos_sin_cache, _rope_head_size=HEAD,
        _rope_is_neox=True, layers=[SimpleNamespace(self_attn=la) for la in layers],
        _attn_layers=attn,
    )  # fmt: skip
    fake._project_context_kv_token_major = lambda s, n, L_, h, d: kv.view(n, L_, 2, h, d)
    fake._project_context_kv = MethodType(DFlashQwen3Model._project_context_kv, fake)
    fake._normalize_context_k = MethodType(DFlashQwen3Model._normalize_context_k, fake)
    states = torch.empty((C, 1), device=kv.device)
    DFlashQwen3Model.precompute_and_store_context_kv(fake, states, positions, slots)


@pytest.mark.parametrize("layout", ["rows", "slots"])
@pytest.mark.parametrize("tp", [16, 8, 4])
def test_context_insert_matches_stock(tp, layout):
    """ContextKV writes every layer's cache bytes exactly as the stock context step:
    the whole backing buffer bit-identical, slot -1 rows skipped."""
    device = _device()
    H = TP_HEADS[tp]
    gen = torch.Generator(device=device).manual_seed(100 + tp)
    layers = make_layers(H, gen)
    state = install(layers, 7, (1, 2, 4))
    for C in (8, 21, 64, 152, 513):
        cache = DraftCache(layout, H, -(-2 * C // page_of(layout, H)) + 3, gen)
        for i, layer in enumerate(layers):
            layer.attn.bind_kv_cache(cache.view(i))
        kv = (torch.randn((C, LAYERS * 2 * H * HEAD), generator=gen, device=device)
              * 3).bfloat16()  # fmt: skip
        pick = torch.randperm(cache.blocks * cache.page, device=device)[:C]
        slots = pick.long()
        slots[::5] = -1  # rejected / evicted rows
        positions = torch.randint(0, MAX_POS, (C,), generator=gen, device=device)
        positions[0], positions[-1] = 0, MAX_POS - 1
        before = cache.buffer.clone()
        assert state.store_context(kv, positions, slots)
        torch.accelerator.synchronize(device)
        ours = cache.buffer.clone()
        cache.buffer.copy_(before)
        stock_context(layers, kv, positions, slots, [cache.view(i) for i in range(LAYERS)])
        torch.accelerator.synchronize(device)
        same = ours.view(torch.int16) == cache.buffer.view(torch.int16)
        assert bool(same.all()), (
            f"C{C}: {int((~same).sum())} cache elements differ from stock"
        )
        changed = int((ours.view(torch.int16) != before.view(torch.int16)).sum())
        assert changed > 0
        del cache, before, ours


# ------------------------------------------------------------- attention
def page_of(layout, kv_heads):
    return 128 if layout in ("slots", "planner") else (4096 if kv_heads == 4 else 1024)


def run_stock(layers, cache, before, qkvs, positions, stale=0):
    """The stock path over a copy of ``before`` whose NaN are ``stale`` (stock
    FlashAttention multiplies masked stale V by 0): (buffer after, buffer before,
    rows per layer). Call inside the batch's forward context."""
    from vllm.models.kimi_k3.nvidia.dspark_gqa import stock_attention_core

    start = before.clone()
    start.view(torch.int16)[torch.isnan(start)] = stale
    buffer = start.clone()
    rows = []
    for i, layer in enumerate(layers):
        layer.attn.kv_cache = cache.view(i, buffer)
        rows.append(stock_attention_core(layer, qkvs[i], positions))
    return buffer, start, rows


def stock_writes(layers, cache, before, qkvs, positions, buffer, start):
    """The elements stock wrote: changed from its stale bytes, also with other stale
    bytes where they were NaN (an insert may write the value they were set to)."""
    i16 = torch.int16
    written = buffer.view(i16) != start.view(i16)
    if bool(torch.isnan(before).any()):
        other, other_start, _ = run_stock(layers, cache, before, qkvs, positions, 0x3F80)
        written |= other.view(i16) != other_start.view(i16)
    return written


def make_cache(layout, kv_heads, blocks, gen, fill="finite"):
    if layout == "planner":
        return PlannerCache(kv_heads, blocks, gen, fill)
    return DraftCache(layout, kv_heads, blocks, gen, fill)


def run_case(state, layers, layout, B, q, gen, fill="finite", contexts=None, seed=None):
    """One drafter batch through both paths: (ours, stock, exact rows per layer,
    lengths, cache elements whose insert differs from stock's). ``seed`` draws the
    contexts and blocks (default B)."""
    from vllm.forward_context import override_forward_context

    device = _device()
    H = layers[0].num_kv_heads
    width = (G * H + 2 * H) * HEAD
    page = page_of(layout, H)
    need = B * (-(-(WINDOW + q) // page) + 1) + 4
    cache = make_cache(layout, H, need, gen, fill)
    seed = B if seed is None else seed
    free = torch.randperm(cache.blocks - 1, generator=torch.Generator().manual_seed(seed))
    free = free[: need - 2] + 1
    seq_lens, positions, table, slots, lengths = batch_case(
        B, q, cache.page, free, gen, contexts, seed
    )
    if B >= 3 and fill == "finite":
        # a query row whose block is the null block: no insert, stock reads the cache
        slots[q - 1] = -1
    views = [cache.view(i) for i in range(LAYERS)]
    fill_context(cache, views, lengths, table, q, gen)
    md = metadata(seq_lens, table, slots, q)
    context = forward_context(layers, md, slots)
    before = cache.buffer.clone()
    for i, layer in enumerate(layers):
        layer.attn.bind_kv_cache(views[i])
    qkvs = [(torch.randn((B * q, width), generator=gen, device=device) * 2).bfloat16()
            for _ in layers]  # fmt: skip
    ours, stock, exact = [], [], []
    with override_forward_context(context):
        for i in range(LAYERS):
            out = state.forward(i, qkvs[i], positions)
            assert out is not None, f"layer {i} B{B} not covered"
            ours.append(out)
        stock_buffer, stock_before, stock = run_stock(
            layers, cache, before, qkvs, positions
        )
        written_stock = stock_writes(
            layers, cache, before, qkvs, positions, stock_buffer, stock_before
        )
        torch.accelerator.synchronize(device)
        for i, layer in enumerate(layers):
            exact.append(exact_rows(layer, cache.view(i, stock_buffer), qkvs[i],
                                    positions, seq_lens, table, q))  # fmt: skip
    torch.accelerator.synchronize(device)
    i16 = torch.int16
    written_ours = cache.buffer.view(i16) != before.view(i16)
    differ = int((written_stock != written_ours).sum())
    differ += int(
        (cache.buffer.view(i16)[written_stock] != stock_buffer.view(i16)[written_stock])
        .sum()
    )
    assert int(written_stock.sum()) > 0
    return ours, stock, exact, lengths, differ


def check_rows(ours, stock, exact, B, q, label):
    stats = []
    for i, (o, s, e) in enumerate(zip(ours, stock, exact)):
        o, s = o.double(), s.double()
        assert torch.isfinite(o).all(), f"{label} layer {i}: non-finite rows"
        scale = e.abs().max().item()
        err_ours = (o - e).abs().max().item()
        err_stock = (s - e).abs().max().item()
        diff = (o - s).abs().max().item()
        rel = ((o - s).abs() / (s.abs() + 1e-2 * scale)).max().item()
        stats.append((err_ours, err_stock, diff, rel, scale))
        # BF16 noise: as close to the FP64 rows as stock FlashAttention (both round P
        # and the output to BF16), and within a few BF16 ulps of stock's rows.
        bound = max(2.0 * err_stock, scale * 2.0**-8)
        assert err_ours <= bound, (label, i, err_ours, err_stock, scale)
        assert diff <= max(4.0 * err_stock, scale * 2.0**-7), (label, i, diff, err_stock)
    return stats


@pytest.mark.parametrize("layout", ["rows", "slots"])
@pytest.mark.parametrize("q", [7, 3])
@pytest.mark.parametrize("tp", [16, 8, 4])
def test_window_matches_stock(tp, q, layout):
    """GqaWindow (q / k norm, RoPE, the query rows' insert, the causal 2048 window) per
    drafter batch 1-128: cache bytes identical to stock's, rows within BF16 noise."""
    device = _device()
    H = TP_HEADS[tp]
    gen = torch.Generator(device=device).manual_seed(7 * tp + q)
    layers = make_layers(H, gen)
    state = install(layers, q, BATCHES)
    worst = []
    for B in BATCHES:
        seed = 1000 * tp + 100 * q + 10 * (layout == "slots") + B  # (per combo)
        ours, stock, exact, lengths, differ = run_case(
            state, layers, layout, B, q, gen, seed=seed
        )
        label = f"TP{tp} q{q} {layout} B{B}"
        assert differ == 0, f"{label}: {differ} cache elements differ from stock"
        stats = check_rows(ours, stock, exact, B, q, label)
        worst.append((B, max(s[0] / max(s[4], 1e-30) for s in stats),
                      max(s[1] / max(s[4], 1e-30) for s in stats),
                      max(s[3] for s in stats)))  # fmt: skip
        del ours, stock, exact
    print(f"\nTP{tp} q{q} {layout}: B, rel err vs FP64 (ours, stock), max rel vs stock")
    for B, eo, es, rel in worst:
        print(f"  B{B:4d}  {eo:.2e}  {es:.2e}  {rel:.2e}")


@pytest.mark.parametrize("q", [1, 2, 4, 5, 6])
def test_window_tp4_any_q(q):
    """TP4 (the SP decode roles) at DSpark k1 / k2 / k4 / k5 / k6: every GQA_WINDOW
    shape of the draft length (GqaWindow and GqaWindowUmma bins) matches stock, as
    test_window_matches_stock does at k3 / k7."""
    test_window_matches_stock(4, q, "slots")


@pytest.mark.parametrize("q", [1, 2, 4, 5, 6])
def test_window_tp8_any_q(q):
    """TP8 (DFn: TP8 = DCP8 decode, 2 KV heads) at DSpark k1 / k2 / k4 / k5 / k6:
    every GQA_WINDOW shape of the draft length matches stock, as at k3 / k7."""
    test_window_matches_stock(8, q, "slots")


def test_window_umma_redo(monkeypatch):
    """GqaWindowUmma's moved-metadata path (its speculated first loads drained and
    reissued, those ring slots' phases offset; forced on unmoved metadata): the
    same cache bytes and rows as stock (TP4 q3, B96 on the GqaWindowUmma shape)."""
    import functools

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
        gqa_attention as ga,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
        gqa_window_umma as gu,
    )

    compile_redo = functools.partial(gu.compile_window_umma.__wrapped__, redo=True)
    monkeypatch.setattr(ga, "compile_window_umma", compile_redo)
    device = _device()
    gen = torch.Generator(device=device).manual_seed(96)
    layers = make_layers(4, gen)
    assert ga.window_shape(4, 3, 96)[1] == 0  # (the GqaWindowUmma shape)
    state = install(layers, 3, (96,))
    ours, stock, exact, lengths, differ = run_case(state, layers, "slots", 96, 3, gen)
    assert differ == 0
    check_rows(ours, stock, exact, 96, 3, "umma redo")


@pytest.mark.parametrize("shape", [(1, 0, 6, 2), (1, 0, 6, 3)])
def test_window_umma_variants_match_stock(shape, monkeypatch):
    """GqaWindowUmma with the separate S issuer warp (ahead bit 1) and the die split
    (bit 0: resident grid, each CTA one unit of its die; heads 2-3 on die 1) at TP4
    q3, batches up to 96: cache bytes identical to stock's, rows within BF16 noise."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
        gqa_attention as ga,
    )

    monkeypatch.setitem(ga.GQA_WINDOW, (4, 3), ((0, shape),))
    device = _device()
    gen = torch.Generator(device=device).manual_seed(53)
    layers = make_layers(4, gen)
    batches = tuple(b for b in BATCHES if b <= 96)
    state = install(layers, 3, batches)
    state.head_dies = 0b1100
    for B in batches:
        ours, stock, exact, lengths, differ = run_case(
            state, layers, "rows", B, 3, gen, seed=5300 + B
        )
        label = f"umma {shape} B{B}"
        assert differ == 0, f"{label}: {differ} cache elements differ from stock"
        check_rows(ours, stock, exact, B, 3, label)
        del ours, stock, exact


STREAM = (0, -1, 6, 0)  # GqaWindowStream's launch shape (key groups -1)


@pytest.mark.parametrize("layout", ["rows", "slots"])
def test_window_stream_matches_stock(layout, monkeypatch):
    """GqaWindowStream (the persistent stream over every SM: units split into pieces,
    the pieces' partials merged by each unit's last piece) at TP4 q3, batches 1-128:
    cache bytes identical to stock's, rows within BF16 noise."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
        gqa_attention as ga,
    )

    monkeypatch.setitem(ga.GQA_WINDOW, (4, 3), ((0, STREAM),))
    device = _device()
    gen = torch.Generator(device=device).manual_seed(43)
    layers = make_layers(4, gen)
    state = install(layers, 3, BATCHES)
    assert state.counters is not None
    for B in BATCHES:
        seed = 4300 + 10 * (layout == "slots") + B
        ours, stock, exact, lengths, differ = run_case(
            state, layers, layout, B, 3, gen, seed=seed
        )
        label = f"stream {layout} B{B}"
        assert differ == 0, f"{label}: {differ} cache elements differ from stock"
        check_rows(ours, stock, exact, B, 3, label)
        assert int(state.counters.abs().sum()) == 0, f"{label}: counters not reset"
        del ours, stock, exact


def test_window_stream_redo(monkeypatch):
    """GqaWindowStream's moved-metadata path (the speculated first loads drained and
    reissued, forced on unmoved metadata): stock's cache bytes and rows (B96)."""
    import functools

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
        gqa_attention as ga,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
        gqa_window_stream as gs,
    )

    compile_redo = functools.partial(gs.compile_window_stream.__wrapped__, redo=True)
    monkeypatch.setattr(ga, "compile_window_stream", compile_redo)
    monkeypatch.setitem(ga.GQA_WINDOW, (4, 3), ((0, STREAM),))
    device = _device()
    gen = torch.Generator(device=device).manual_seed(97)
    layers = make_layers(4, gen)
    state = install(layers, 3, (96,))
    ours, stock, exact, lengths, differ = run_case(state, layers, "slots", 96, 3, gen)
    assert differ == 0
    check_rows(ours, stock, exact, 96, 3, "stream redo")


@pytest.mark.parametrize("layout", ["rows", "slots"])
def test_window_ignores_stale_nan(layout):
    """Never-zeroed blocks: NaN in every cache byte outside the requests' written keys
    (before the window, past seq_len, the rest of their pages) never reaches a row;
    the rows equal the stock rows over zeroed stale bytes."""
    device = _device()
    gen = torch.Generator(device=device).manual_seed(3)
    layers = make_layers(1, gen)
    state = install(layers, 7, (8,))
    contexts = [1, 31, 2040, 2041, 2048, 4097, 150000, 33]
    ours, stock, exact, lengths, differ = run_case(
        state, layers, layout, 8, 7, gen, fill="nan", contexts=contexts
    )
    assert differ == 0
    check_rows(ours, stock, exact, 8, 7, f"nan {layout}")


def test_install_refuses_unsupported_geometry():
    """Strict deployments refuse at install what the kernels do not implement; others
    get None (the stock attention)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft.gqa_attention import (  # noqa: E501
        install_gqa_attention,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Uncovered

    device = _device()
    gen = torch.Generator(device=device).manual_seed(4)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=7),
        model_config=SimpleNamespace(max_model_len=MAX_POS),
    )
    for attribute, value in (("sliding_window", 4096), ("v_scale", 2.0), ("causal", False)):
        layers = make_layers(1, gen)
        setattr(layers[2], attribute, value)
        assert install_gqa_attention(layers, config, [7], [8], False) is None
        with pytest.raises(Uncovered):
            install_gqa_attention(layers, config, [7], [8], True)
    layers = make_layers(1, gen)
    longer = SimpleNamespace(
        speculative_config=config.speculative_config,
        model_config=SimpleNamespace(max_model_len=2 * MAX_POS),
    )
    assert install_gqa_attention(layers, longer, [7], [8], False) is None
    # the stock norm / RoPE are not vLLM's C kernels (the arithmetic implemented)
    native_norm = SimpleNamespace(
        **vars(config),
        kernel_config=SimpleNamespace(
            ir_op_priority=SimpleNamespace(rms_norm=["native", "vllm_c"])
        ),
    )
    native_rope = SimpleNamespace(
        **vars(config),
        compilation_config=SimpleNamespace(is_custom_op_enabled=lambda op: False),
    )
    for other in (native_norm, native_rope):
        assert install_gqa_attention(layers, other, [7], [8], False) is None
        with pytest.raises(Uncovered):
            install_gqa_attention(layers, other, [7], [8], True)
    production = SimpleNamespace(
        **vars(config),
        kernel_config=SimpleNamespace(
            ir_op_priority=SimpleNamespace(rms_norm=["vllm_c", "native"])
        ),
        compilation_config=SimpleNamespace(is_custom_op_enabled=lambda op: True),
    )
    assert install_gqa_attention(layers, production, [7], [8], True) is not None
    assert math.isfinite(EPS)


# ------------------------------------------------------------- graph padding
# (KV heads per rank, q) of the GQA_WINDOW shapes a decode deployment captures:
# TP16 k7 / k3 (C1, C64, C160) and TP4 k3 / k7 (C512)
PADDED = ((1, 7), (1, 3), (4, 3), (4, 7))


def padded_graphs(kv_heads, q):
    """(graph batch, real requests) pairs: every GQA_WINDOW bin whose launch shape
    spreads a request over a cluster (16 splits at the small batches) or runs
    GqaWindowUmma, its largest
    batch (the open bin: twice the previous) padded by one request up to 16
    requests, by half above."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft.gqa_attention import (  # noqa: E501
        GQA_WINDOW,
    )

    cases, previous = [], 0
    for top, shape in GQA_WINDOW[(kv_heads, q)]:
        graph = top or 2 * previous
        previous = top
        if shape[0] > 1 or shape[1] == 0:  # (cluster or GqaWindowUmma shapes)
            cases.append((graph, graph - 1 if graph <= 16 else graph // 2))
    return cases


def pad_batch(seq_lens, positions, table, slots, q, graph, blocks, gen):
    """The runner's padded graph batch: trailing requests of seq_len 0, their query
    rows at slot -1 (position 0), their block-table rows stale block IDs."""
    device = seq_lens.device
    pad = graph - seq_lens.shape[0]
    stale = torch.randint(1, blocks, (pad, table.shape[1]), generator=gen,
                          device=device, dtype=torch.int32)  # fmt: skip
    return (
        torch.cat((seq_lens, seq_lens.new_zeros(pad))),
        torch.cat((positions, positions.new_zeros(pad * q))),
        torch.cat((table, stale)),
        torch.cat((slots, slots.new_full((pad * q,), -1))),
    )


@pytest.mark.parametrize("kv_heads,q", PADDED)
def test_window_padded_requests(kv_heads, q):
    """Graph padding: trailing requests of seq_len 0 at slot -1 over planner slot
    pages whose never-written bytes are NaN. Per cluster launch shape: no fault or
    hang; the real requests' rows and cache bytes bit-identical to the same launch
    shape without the padded requests; padded rows zero; no write for slot -1; the
    inserted bytes are stock's and the real rows as close to FP64 as stock's."""
    from vllm.forward_context import override_forward_context

    device = _device()
    H = kv_heads
    width = (G * H + 2 * H) * HEAD
    page = page_of("planner", H)
    i16 = torch.int16
    for graph, real in padded_graphs(kv_heads, q):
        seed = 10007 * kv_heads + 101 * q + 7 * graph + real
        gen = torch.Generator(device=device).manual_seed(seed)
        layers = make_layers(H, gen)
        state = install(layers, q, (graph,))
        # the real requests alone through the graph batch's launch shape (the
        # state's batch -> shape table, before any cache is bound)
        state.rows = state.rows | {real * q}
        state.shapes[real] = state.shapes[graph]
        need = real * (-(-(WINDOW + q) // page) + 1) + 4
        cache = PlannerCache(H, need, gen, "nan")
        free = torch.randperm(cache.blocks - 1, generator=torch.Generator().manual_seed(seed))
        free = free[: need - 2] + 1
        seq_lens, positions, table, slots, lengths = batch_case(
            real, q, page, free, gen, seed=seed
        )
        views = [cache.view(i) for i in range(LAYERS)]
        fill_context(cache, views, lengths, table, q, gen)
        for i, layer in enumerate(layers):
            layer.attn.bind_kv_cache(views[i])
        padded = pad_batch(seq_lens, positions, table, slots, q, graph, cache.blocks, gen)
        p_lens, p_pos, p_table, p_slots = padded
        qkvs = [(torch.randn((graph * q, width), generator=gen, device=device) * 2)
                .bfloat16() for _ in layers]  # fmt: skip
        before = cache.buffer.clone()
        label = f"H{H} q{q} graph {graph} real {real} {state.shapes[graph]}"

        md = metadata(p_lens, p_table, p_slots, q)
        with override_forward_context(forward_context(layers, md, p_slots)):
            ours_padded = [state.forward(i, qkvs[i], p_pos) for i in range(LAYERS)]
        torch.accelerator.synchronize(device)
        assert all(o is not None for o in ours_padded), label
        after_padded = cache.buffer.clone()
        cache.buffer.copy_(before)
        md = metadata(seq_lens, table, slots, q)
        with override_forward_context(forward_context(layers, md, slots)):
            ours = [state.forward(i, qkvs[i][: real * q], positions) for i in range(LAYERS)]
        torch.accelerator.synchronize(device)
        assert all(o is not None for o in ours), label
        same = after_padded.view(i16) == cache.buffer.view(i16)
        assert bool(same.all()), f"{label}: {int((~same).sum())} cache elements differ"
        for i in range(LAYERS):
            assert torch.equal(ours_padded[i][: real * q].view(i16), ours[i].view(i16)), (
                f"{label} layer {i}: real rows differ from the unpadded run"
            )
            tail = ours_padded[i][real * q :]
            assert bool(torch.isfinite(tail).all()) and not bool(tail.any()), (
                f"{label} layer {i}: padded rows not zero"
            )

        # stock over the padded batch (its stale NaN zeroed: stock multiplies masked
        # stale V by 0): the same inserted bytes, real rows within BF16 noise
        md = metadata(p_lens, p_table, p_slots, q)
        with override_forward_context(forward_context(layers, md, p_slots)):
            stock_buffer, stock_before, stock = run_stock(
                layers, cache, before, qkvs, p_pos
            )
            written_stock = stock_writes(
                layers, cache, before, qkvs, p_pos, stock_buffer, stock_before
            )
        torch.accelerator.synchronize(device)
        written_ours = after_padded.view(i16) != before.view(i16)
        assert bool((written_stock == written_ours).all()), f"{label}: writes differ"
        assert bool(
            (after_padded.view(i16)[written_ours] == stock_buffer.view(i16)[written_ours]).all()
        ), f"{label}: inserted bytes differ from stock"
        assert int(written_ours.sum()) > 0
        exact = [
            exact_rows(layer, cache.view(i, stock_buffer), qkvs[i][: real * q],
                       positions, seq_lens, table, q)
            for i, layer in enumerate(layers)
        ]  # fmt: skip
        check_rows(ours, [s[: real * q] for s in stock], exact, real, q, label)
        for i, layer in enumerate(layers):
            layer.attn.kv_cache = views[i]
        del cache, before, after_padded, stock_before, stock_buffer


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_context_insert_padded_rows(kv_heads):
    """ContextKV with trailing padded rows (slot -1, position 0) over NaN planner slot
    pages: the whole allocation bit-identical to the unpadded insert and to stock's
    (no write for slot -1, never-written bytes untouched)."""
    device = _device()
    H = kv_heads
    gen = torch.Generator(device=device).manual_seed(500 + kv_heads)
    layers = make_layers(H, gen)
    state = install(layers, 7, (1, 2, 4))
    for C, pad in ((8, 8), (21, 43), (152, 104), (513, 511)):
        cache = PlannerCache(H, -(-2 * C // 128) + 3, gen, "nan")
        for i, layer in enumerate(layers):
            layer.attn.bind_kv_cache(cache.view(i))
        kv = (torch.randn(((C + pad), LAYERS * 2 * H * HEAD), generator=gen,
                          device=device) * 3).bfloat16()  # fmt: skip
        slots = torch.randperm(cache.blocks * 128, device=device)[: C + pad].long()
        slots[C:] = -1
        slots[1:C:7] = -1  # rejected / evicted rows among the real ones
        positions = torch.randint(0, MAX_POS, (C + pad,), generator=gen, device=device)
        positions[C:] = 0
        before = cache.buffer.clone()
        assert state.store_context(kv, positions, slots)
        torch.accelerator.synchronize(device)
        padded = cache.buffer.clone()
        cache.buffer.copy_(before)
        assert state.store_context(kv[:C], positions[:C], slots[:C])
        torch.accelerator.synchronize(device)
        i16 = torch.int16
        same = padded.view(i16) == cache.buffer.view(i16)
        assert bool(same.all()), f"C{C}+{pad}: {int((~same).sum())} differ (unpadded)"
        cache.buffer.copy_(before)
        stock_context(layers, kv, positions, slots, [cache.view(i) for i in range(LAYERS)])
        torch.accelerator.synchronize(device)
        same = padded.view(i16) == cache.buffer.view(i16)
        assert bool(same.all()), f"C{C}+{pad}: {int((~same).sum())} differ (stock)"
        written = int((padded.view(i16) != before.view(i16)).sum())
        live = int((slots[:C] >= 0).sum())
        assert written == live * LAYERS * H * 2 * HEAD, (written, live)
        del cache, before, padded


@pytest.mark.parametrize("kv_heads,q", PADDED + ((2, 7), (2, 3)))
def test_window_start_stage_edges(kv_heads, q):
    """Window starts lo with lo % 32 in 26..31 (the query rows' own starts cross a
    32-key stage edge) at the 16-split small batches and a cluster batch, over NaN
    planner slot pages: inserts identical to stock's, rows as close to FP64."""
    device = _device()
    gen = torch.Generator(device=device).manual_seed(77 * kv_heads + q)
    layers = make_layers(kv_heads, gen)
    batches = (2, 4, 8, 16)
    state = install(layers, q, batches)
    for B in batches:
        seed = 31 * kv_heads + 7 * q + B
        order = torch.randperm(len(WINDOW_EDGES), generator=torch.Generator().manual_seed(seed))
        contexts = [WINDOW_EDGES[int(order[b % len(WINDOW_EDGES)])] for b in range(B)]
        ours, stock, exact, lengths, differ = run_case(
            state, layers, "planner", B, q, gen, fill="nan", contexts=contexts,
            seed=seed,
        )  # fmt: skip
        label = f"H{kv_heads} q{q} B{B} edges {state.shapes[B]}"
        assert differ == 0, f"{label}: {differ} cache elements differ from stock"
        check_rows(ours, stock, exact, B, q, label)
