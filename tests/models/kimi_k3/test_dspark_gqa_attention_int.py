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
BATCHES = (1, 2, 3, 4, 7, 8, 12, 16, 28, 38, 50, 64, 128)
# keys before the query rows: inside, at and past the window, at block / stage edges
CONTEXTS = (
    1, 5, 31, 32, 33, 100, 1023, 1024, 2016, 2040, 2041, 2047, 2048, 2049, 2050,
    3071, 4096, 4097, 50000, 150000, MAX_POS - 16,
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


def batch_case(B, q, page, blocks_free, gen, contexts=None):
    """seq_lens, positions, block table (window pages only; the rest the null block),
    and the query slots of B requests."""
    device = _device()
    order = torch.randperm(len(CONTEXTS), generator=torch.Generator().manual_seed(B))
    ctx = contexts or [CONTEXTS[int(order[b % len(CONTEXTS)])] for b in range(B)]
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
    return 128 if layout == "slots" else (4096 if kv_heads == 4 else 1024)


def run_case(state, layers, layout, B, q, gen, fill="finite", contexts=None):
    """One drafter batch through both paths: (ours, stock, exact rows per layer,
    lengths, cache elements whose insert differs from stock's)."""
    from vllm.forward_context import override_forward_context
    from vllm.models.kimi_k3.nvidia.dspark_gqa import stock_attention_core

    device = _device()
    H = layers[0].num_kv_heads
    width = (G * H + 2 * H) * HEAD
    page = page_of(layout, H)
    need = B * (-(-(WINDOW + q) // page) + 1) + 4
    cache = DraftCache(layout, H, need, gen, fill)
    free = torch.randperm(cache.blocks - 1, generator=torch.Generator().manual_seed(B))
    free = free[: need - 2] + 1
    seq_lens, positions, table, slots, lengths = batch_case(
        B, q, cache.page, free, gen, contexts
    )
    if B >= 3 and fill == "finite":
        # a query row whose block is the null block: no insert, stock reads the cache
        slots[q - 1] = -1
    views = [cache.view(i) for i in range(LAYERS)]
    fill_context(cache, views, lengths, table, q, gen)
    md = metadata(seq_lens, table, slots, q)
    context = forward_context(layers, md, slots)
    before = cache.buffer.clone()
    stock_buffer = before.clone()
    stock_before = before
    if fill == "nan":  # stock FlashAttention multiplies masked stale V by 0
        stock_buffer.view(torch.int16)[torch.isnan(stock_buffer)] = 0
        stock_before = stock_buffer.clone()
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
        for i, layer in enumerate(layers):
            layer.attn.kv_cache = cache.view(i, stock_buffer)
            stock.append(stock_attention_core(layer, qkvs[i], positions))
        torch.accelerator.synchronize(device)
        for i, layer in enumerate(layers):
            exact.append(exact_rows(layer, cache.view(i, stock_buffer), qkvs[i],
                                    positions, seq_lens, table, q))  # fmt: skip
    torch.accelerator.synchronize(device)
    i16 = torch.int16
    written_stock = stock_buffer.view(i16) != stock_before.view(i16)
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
        ours, stock, exact, lengths, differ = run_case(state, layers, layout, B, q, gen)
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
    assert math.isfinite(EPS)
