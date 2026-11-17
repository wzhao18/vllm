# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer microbench of the release DSpark drafter's fused attention
(cutedsl.draft.gqa_attention) against the stock path, one Rubin GPU, CUDA graphs.

  attention  stock: dspark_gqa.stock_attention_core (q_norm, k_norm, rotary_embedding,
             FlashAttention's reshape_and_cache_flash + FA2 sliding-window forward:
             5 kernels); fused: the state's forward (one GqaWindow launch)
  context    stock: DFlashQwen3Model.precompute_and_store_context_kv after the K/V
             projection (permute copy, grouped rms_norm, positions repeat, RoPE, five
             inserts); fused: store_context (one ContextKV launch); B x (q + 1) rows

Every request has a full window (``--ctx`` keys before its q query rows; AgentX decode
contexts are far longer than 2048). Cold KV: the graph cycles through enough disjoint
window page sets (>= 600 MB) that no call finds its K/V in L2, as in a decode step
where the target model runs between the draft layers. Median of 5 replays, us per call.

  python3 bench_gqa_attention_int.py --tp 16 --q 7 --batches 1,2,4,8,16,32
  python3 bench_gqa_attention_int.py --tp 4 --q 3 --layout slots --batches 16,64,128
  python3 bench_gqa_attention_int.py ... --shape 8/2/2/0  (GqaWindow launch shape
  splits/key groups/ring/ahead for every batch, overriding GQA_WINDOW)
"""

import argparse
from types import SimpleNamespace

import torch

G, HEAD, WINDOW, LAYERS = 6, 64, 2048, 5
EPS = 1e-5
MAX_POS = 1 << 20


class _Norm:
    def __init__(self, weight):
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.variance_epsilon = EPS

    def __call__(self, x):
        import vllm._custom_ops as ops

        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        ops.rms_norm(out, x, self.weight.data, EPS)
        return out


class _Rope:
    def __init__(self, table):
        self.cos_sin_cache = table
        self.head_size = self.rotary_dim = HEAD
        self.is_neox_style = True

    def __call__(self, positions, q, k):
        import vllm._custom_ops as ops

        ops.rotary_embedding(positions, q, k, HEAD, self.cos_sin_cache, True)
        return q, k


class _Attention:
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


def rope_table(device):
    inv = 1.0 / (1e4 ** (torch.arange(0, HEAD, 2, dtype=torch.float) / HEAD))
    t = torch.arange(MAX_POS, dtype=torch.float)
    freqs = torch.einsum("i,j->ij", t, inv)
    return torch.cat((freqs.cos(), freqs.sin()), -1).to(torch.bfloat16).to(device)


def make_layers(kv_heads, device):
    heads = G * kv_heads
    rope = _Rope(rope_table(device))
    layers = []
    for i in range(LAYERS):
        w = lambda: (torch.rand(HEAD, device=device) + 0.5).bfloat16()  # noqa: E731
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


def timed(calls, loops):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for call in calls:
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(loops):
            for call in calls:
                call()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / (loops * len(calls)))
    return sorted(times)[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--q", type=int, default=7)
    parser.add_argument("--batches", default="1,2,4,8,16,32,64")
    parser.add_argument("--layout", default="rows", choices=("rows", "slots"))
    parser.add_argument("--ctx", type=int, default=150000)
    parser.add_argument("--shape", default="")
    parser.add_argument("--arms", default="stock,fused")
    parser.add_argument("--no-context", action="store_true")
    args = parser.parse_args()

    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
    from vllm.models.kimi_k3.nvidia.dspark_gqa import stock_attention_core
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
        gqa_attention as ga,
    )
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    H = {16: 1, 8: 2, 4: 4}[args.tp]
    q = args.q
    batches = [int(b) for b in args.batches.split(",")]
    if args.shape:
        shape = tuple(int(x) for x in args.shape.split("/"))
        ga.GQA_WINDOW[(H, q)] = ((0, shape),)
    arms = args.arms.split(",")
    layers = make_layers(H, device)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=q),
        model_config=SimpleNamespace(max_model_len=MAX_POS),
    )
    state = ga.install_gqa_attention(
        layers, config, [b * q for b in batches], [b * (q + 1) for b in batches], True
    )
    page = 128 if args.layout == "slots" else (4096 if H == 4 else 1024)
    layer_page = H * page * 2 * HEAD
    stride = LAYERS * layer_page + (64 if args.layout == "slots" else 128)
    width = (G * H + 2 * H) * HEAD
    L = args.ctx + q
    lo = L - q - (WINDOW - 1)
    pages = list(range(lo // page, (L - 1) // page + 1))
    print(
        f"TP{args.tp} ({G * H}q/{H}kv) q{q} {args.layout} page {page}, contexts "
        f"{args.ctx}: us per layer call (cold KV), stock / fused / speedup"
    )
    for B in batches:
        per_call = B * H * (WINDOW - 1 + q) * 256
        copies = max(2, min(64, (600 << 20) // per_call))
        blocks = copies * B * len(pages) + 2
        buffer = torch.empty(blocks * stride, dtype=torch.bfloat16, device=device)
        buffer.normal_()
        views = [
            buffer.as_strided((blocks, H, page, 2 * HEAD),
                              (stride, page * 2 * HEAD, 2 * HEAD, 1), i * layer_page)
            for i in range(LAYERS)
        ]  # fmt: skip
        for i, layer in enumerate(layers):
            layer.attn.bind_kv_cache(views[i])
        perm = torch.randperm(blocks - 1, device=device)[: copies * B * len(pages)] + 1
        perm = perm.view(copies, B, len(pages)).int()
        seq = torch.full((B,), L, dtype=torch.int32, device=device)
        positions = torch.arange(L - q, L, device=device).repeat(B)
        cases = []
        for c in range(copies):
            table = torch.zeros((B, (L - 1) // page + 2), dtype=torch.int32, device=device)
            table[:, pages[0] : pages[-1] + 1] = perm[c]
            pos = positions.cpu()
            slots = (table.cpu()[torch.arange(B).repeat_interleave(q), pos // page].long()
                     * page + pos % page).to(device)  # fmt: skip
            md = FlashAttentionMetadata(
                num_actual_tokens=B * q, max_query_len=q,
                query_start_loc=torch.arange(0, (B + 1) * q, q, dtype=torch.int32,
                                             device=device),
                max_seq_len=L, seq_lens=seq, block_table=table, slot_mapping=slots,
                causal=True,
            )  # fmt: skip
            names = [layer.attn.layer_name for layer in layers]
            fc = ForwardContext(
                no_compile_layers={},
                attn_metadata={n: md for n in names},
                slot_mapping={n: slots for n in names},
            )
            qkv = torch.randn((B * q, width), device=device).bfloat16()
            cases.append((fc, qkv))
        layer = layers[0]

        def stock_call(fc, qkv):
            def run():
                with override_forward_context(fc):
                    stock_attention_core(layer, qkv, positions)

            return run

        def fused_call(fc, qkv):
            def run():
                with override_forward_context(fc):
                    assert state.forward(0, qkv, positions) is not None

            return run

        loops = max(1, 200 // copies)
        result = {}
        if "stock" in arms:
            result["stock"] = timed([stock_call(*c) for c in cases], loops)
        if "fused" in arms:
            result["fused"] = timed([fused_call(*c) for c in cases], loops)
        line = f"  B{B:4d} attention  "
        line += "  ".join(f"{k} {v:7.2f}" for k, v in result.items())
        if len(result) == 2:
            line += f"  x{result['stock'] / result['fused']:.2f}"
        shape = ga.window_shape(H, q, B)
        line += f"  (shape {shape}, {copies} window sets)"
        if not args.no_context:
            C = B * (q + 1)
            kv = torch.randn((C, LAYERS * 2 * H * HEAD), device=device).bfloat16()
            cpos = torch.randint(0, MAX_POS, (C,), device=device)
            cslots = torch.randperm(blocks * page, device=device)[:C].long()
            fake = SimpleNamespace(
                _num_attn_layers=LAYERS, _kv_size=H * HEAD, _head_dim=HEAD,
                _num_kv_heads=H,
                _k_norm_weights=torch.stack([la.k_norm.weight.data for la in layers]),
                _rms_norm_eps=EPS, _rope_cos_sin_cache=layer.rotary_emb.cos_sin_cache,
                _rope_head_size=HEAD, _rope_is_neox=True,
                layers=[SimpleNamespace(self_attn=la) for la in layers],
                _attn_layers=[la.attn for la in layers],
            )  # fmt: skip
            fake._project_context_kv_token_major = (
                lambda s, n, L_, h, d: kv.view(n, L_, 2, h, d)
            )
            fake._project_context_kv = DFlashQwen3Model._project_context_kv.__get__(fake)
            fake._normalize_context_k = DFlashQwen3Model._normalize_context_k.__get__(
                fake
            )
            states = torch.empty((C, 1), device=device)
            ctx = {}
            if "stock" in arms:
                ctx["stock"] = timed(
                    [lambda: DFlashQwen3Model.precompute_and_store_context_kv(
                        fake, states, cpos, cslots)], 50,
                )  # fmt: skip
            if "fused" in arms:
                ctx["fused"] = timed(
                    [lambda: state.store_context(kv, cpos, cslots)], 50
                )
            line += f"\n        context C{C:4d} "
            line += "  ".join(f"{k} {v:7.2f}" for k, v in ctx.items())
            if len(ctx) == 2:
                line += f"  x{ctx['stock'] / ctx['fused']:.2f}"
        print(line, flush=True)
        del buffer, views, cases


if __name__ == "__main__":
    main()
