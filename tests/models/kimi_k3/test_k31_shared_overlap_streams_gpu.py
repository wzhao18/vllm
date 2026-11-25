# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``KimiMoE._overlapped_forward``: the sequence-parallel MegaMoE layer with the
replicated shared expert concurrent with the routed experts, on the stock methods
bound to a stand-in layer (BF16 linears, deterministic router / experts stand-ins:
the real MegaMoE needs the EP group).

* Full-grid layer (P/D prefill workers): the routing chain on the high-priority
  stream, the shared expert on the current stream; bitwise equal to the serial
  order eagerly, inside a whole-step CUDA graph and inside a breakable CUDA graph
  with eager breaks before and after the layer (the fork and join stay in one
  segment), over repeated replays.
* SM-capped layer (decode, unified serving): never enters _overlapped_forward;
  forward()'s stock maybe_execute_in_parallel form, unchanged.
* A shared expert with a persistent kernel (GEMM-RS/AR, fused SiTU) raises.

One GPU, seconds."""

import types

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture  # noqa: E402
from vllm.model_executor.layers.linear import UnquantizedLinearMethod  # noqa: E402
from vllm.models.kimi_k3.nvidia import model as kimi  # noqa: E402

DEVICE = torch.device("cuda", 0)
HIDDEN, LATENT, INTER = 1024, 512, 768


class Linear:
    def __init__(self, n, k, g):
        self.weight = (torch.randn(n, k, device=DEVICE, generator=g) * 0.05).bfloat16()
        self.quant_method = UnquantizedLinearMethod()
        self.bias = None

    def __call__(self, x):
        return torch.nn.functional.linear(x, self.weight), None


class Norm:
    has_weight, variance_size_override, variance_epsilon = True, None, 1e-6

    def __init__(self, g):
        self.weight = 1 + 0.1 * torch.randn(LATENT, device=DEVICE, generator=g)

    def __call__(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + self.variance_epsilon) * self.weight).to(
            x.dtype
        )


def silu_and_mul(x):
    d = x.shape[-1] // 2
    return (torch.nn.functional.silu(x[..., :d].float()) * x[..., d:].float()).to(
        x.dtype
    )


class Shared(types.SimpleNamespace):
    _prefill_situ = None
    gate_up_striped = False

    def __call__(self, x):
        self.streams.append(torch.cuda.current_stream())
        return self.forward(x)


class Experts:
    """Routed experts stand-in (a deterministic latent map, long enough to overlap)."""

    def __init__(self):
        self.streams = []

    def __call__(self, routed, weights, ids):
        self.streams.append(torch.cuda.current_stream())
        out = routed
        for _ in range(8):
            out = (out.float() * 0.5 + routed.float().flip(0) * 0.5).to(routed.dtype)
        return out * weights[:, :1].to(out.dtype)


def layer(full_grid, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    shared = Shared(
        gate_up_proj=Linear(2 * INTER, HIDDEN, g),
        down_proj=Linear(HIDDEN, INTER, g),
        act_fn=silu_and_mul,
        shard_sequence_parallel=False,
        gemm_rs_ar=None,
        streams=[],
    )
    for name in ("forward", "_gate_up", "_act_mxfp8", "_act_mxfp8_ok", "mxfp8_input"):
        setattr(shared, name, types.MethodType(getattr(kimi.KimiMLP, name), shared))
    router_w = torch.randn(16, HIDDEN, device=DEVICE, generator=g).bfloat16()

    def router(h):
        probs = (h.float() @ router_w.float().t()).softmax(-1)
        return probs, probs.argmax(-1, keepdim=True)

    mlp = types.SimpleNamespace(
        use_mega_moe=True, use_latent_moe=True, _packed_moe=None,
        routed_expert_down_proj=Linear(LATENT, HIDDEN, g), shared_experts=shared,
        _shared_overlap=True, _router=router,
        routed_output_transform=kimi.KimiRoutedOutputTransform(
            Norm(g), Linear(HIDDEN, LATENT, g)
        ),
        experts=Experts(),
        _down_proj_stream=torch.cuda.Stream(),
        _down_proj_events=(torch.cuda.Event(), torch.cuda.Event()),
        _shared_stream=torch.cuda.Stream(),
        _shared_events=(torch.cuda.Event(), torch.cuda.Event()),
        _routed_stream=kimi.routed_priority_stream(),
        _routed_down_proj_stream=kimi.routed_priority_stream(1),
        _shared_full_grid=full_grid, _shared_kernels_checked=False,
        _prefill_subcalls=None,
    )  # fmt: skip
    for name in ("forward", "_maybe_overlap_router_and_down_proj",
                 "_shared_mxfp8_input", "_routed_experts", "_prefill_shared_stream",
                 "_overlapped_forward"):  # fmt: skip
        setattr(mlp, name, types.MethodType(getattr(kimi.KimiMoE, name), mlp))
    return mlp


def serial(mlp, h):
    """The same callables in the serial order on the current stream."""
    routed, weights, ids = mlp._maybe_overlap_router_and_down_proj(h, None)
    final = mlp.experts(routed, weights, ids)
    shared = mlp.shared_experts(h)
    return mlp.routed_output_transform(final, residual=shared)


def inputs(n, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(n, HIDDEN, device=DEVICE, generator=g).bfloat16()


def test_priority_stream_is_above_default():
    s = kimi.routed_priority_stream()
    assert s is kimi.routed_priority_stream()  # one per device
    assert s.priority < torch.cuda.current_stream().priority
    assert s.priority < torch.cuda.Stream().priority
    s1 = kimi.routed_priority_stream(1)  # the chain's own fork (latent down)
    assert s1 is kimi.routed_priority_stream(1) and s1 != s
    assert s1.priority == s.priority


@pytest.mark.parametrize("full_grid", [False, True])
@pytest.mark.parametrize("n", [16, 512])
def test_eager_streams_and_bitwise(full_grid, n):
    torch.cuda.set_device(DEVICE)
    mlp = layer(full_grid)
    h = inputs(n, n)
    want = serial(mlp, h)
    mlp.experts.streams.clear()
    mlp.shared_experts.streams.clear()
    main = torch.cuda.current_stream()
    got = mlp.forward(h)
    assert torch.cuda.current_stream() == main
    assert torch.equal(got, want)
    if not full_grid:
        assert mlp.experts.streams == [main]
        assert mlp.shared_experts.streams == [mlp._shared_stream]
    else:
        assert mlp.experts.streams == [mlp._routed_stream]
        assert mlp.shared_experts.streams == [main]


@pytest.mark.parametrize("full_grid", [False, True])
def test_cuda_graph_replays_bitwise(full_grid, n=256):
    torch.cuda.set_device(DEVICE)
    mlp = layer(full_grid, seed=1)
    static = inputs(n, 7)
    want = serial(mlp, static).clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mlp.forward(static)  # warm on the capture stream
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = mlp.forward(static)
    torch.cuda.current_stream().wait_stream(stream)
    for i in range(50):
        static.copy_(inputs(n, 100 + i))
        graph.replay()
        assert torch.equal(out, serial(mlp, static)), i
    static.copy_(inputs(n, 7))
    graph.replay()
    assert torch.equal(out, want)


def test_breakable_graph_keeps_overlap_bitwise(full_grid=True, n=256):
    """Production prefill steps of <= 2048 instance tokens run in breakable CUDA
    graphs (a segment ends at each eager attention op): the layer forks and joins
    inside one segment, with eager breaks before and after it."""
    torch.cuda.set_device(DEVICE)
    mlp = layer(full_grid, seed=2)
    static = inputs(n, 3)
    pre = torch.empty_like(static)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mlp.forward(static)
        cap = BreakableCUDAGraphCapture()
        with cap:
            cap.add_eager(lambda: pre.copy_(static * 2))
            out = mlp.forward(pre)
            post = out.clone()
            cap.add_eager(lambda: post.mul_(1))
        assert cap.num_eager_breaks == 2
    torch.cuda.current_stream().wait_stream(stream)
    for i in range(50):
        static.copy_(inputs(n, 200 + i))
        with torch.cuda.stream(stream):
            cap.replay()
        torch.cuda.current_stream().wait_stream(stream)
        assert torch.equal(post, serial(mlp, static * 2)), i


def test_persistent_shared_kernel_raises():
    mlp = layer(True)
    mlp.shared_experts.gemm_rs_ar = object()
    with pytest.raises(RuntimeError, match="persistent"):
        mlp.forward(inputs(512, 0))
    mlp = layer(True)
    mlp.shared_experts._prefill_situ = object()
    with pytest.raises(RuntimeError, match="persistent"):
        mlp.forward(inputs(16, 0))


def test_capped_layer_keeps_the_stock_form():
    """Decode / unified serving (SM-capped layers) never enter _overlapped_forward:
    forward() keeps maybe_execute_in_parallel (routing chain on the current stream,
    shared expert on the side stream), as before this change."""
    torch.cuda.set_device(DEVICE)
    mlp = layer(False)
    mlp._overlapped_forward = None  # calling it would raise
    h = inputs(64, 9)
    want = serial(mlp, h)
    mlp.experts.streams.clear()
    mlp.shared_experts.streams.clear()
    assert torch.equal(mlp.forward(h), want)
    assert mlp.experts.streams == [torch.cuda.current_stream()]
    assert mlp.shared_experts.streams == [mlp._shared_stream]


class StreamLinear(Linear):
    """A Linear that records the stream it runs on."""

    def __init__(self, n, k, g):
        super().__init__(n, k, g)
        self.streams = []

    def __call__(self, x):
        self.streams.append(torch.cuda.current_stream())
        return super().__call__(x)


@pytest.mark.parametrize("n", [512, 4096])
def test_latent_down_fork_keeps_chain_priority(n):
    """Full-grid layer: from _ROUTED_DOWN_PROJ_STREAM_PREFILL_TOKENS rows the latent
    down projection forks off the routing chain onto the chain's second
    high-priority stream (never the normal-priority aux stream, where it would
    compete with the concurrent shared expert while the chain joins on it); below,
    it stays serial on the chain's stream. Bitwise equal to the serial order."""
    torch.cuda.set_device(DEVICE)
    mlp = layer(True, seed=4)
    g = torch.Generator(device=DEVICE).manual_seed(5)
    mlp.routed_expert_down_proj = StreamLinear(LATENT, HIDDEN, g)
    h = inputs(n, 11)
    want = serial(mlp, h)
    mlp.routed_expert_down_proj.streams.clear()
    assert torch.equal(mlp.forward(h), want)
    fork = n >= kimi._ROUTED_DOWN_PROJ_STREAM_PREFILL_TOKENS
    expect = mlp._routed_down_proj_stream if fork else mlp._routed_stream
    assert mlp.routed_expert_down_proj.streams == [expect]
    assert expect.priority < torch.cuda.Stream().priority
