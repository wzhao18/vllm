# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``KimiMoE._overlapped_forward``: the sequence-parallel MegaMoE layer with the
replicated shared expert concurrent with the routed experts, on the stock methods
bound to a stand-in layer (BF16 linears, deterministic router / experts stand-ins:
the real MegaMoE needs the EP group).

* SM-capped bucket (decode; P/D prefill steps of at most 1024 rows per rank): the
  routing chain on the current stream, the shared expert on the side stream.
* Full-grid bucket (larger P/D prefill steps): the routing chain on the
  high-priority stream, the shared expert on the current stream.
* Both: bitwise equal to the serial order, eagerly, inside a whole-step CUDA graph
  and inside a breakable CUDA graph with eager breaks before and after the layer
  (the fork and join stay in one segment), over repeated replays.
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
CAPPED_UP_TO = 64  # rows per rank on the SM-capped bucket in this stand-in


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
    """Routed experts stand-in (a deterministic latent map, long enough to overlap)
    on an SM-capped bucket up to CAPPED_UP_TO rows, full grid above."""

    def __init__(self):
        self.streams = []

    def __call__(self, routed, weights, ids):
        self.streams.append(torch.cuda.current_stream())
        out = routed
        for _ in range(8):
            out = (out.float() * 0.5 + routed.float().flip(0) * 0.5).to(routed.dtype)
        return out * weights[:, :1].to(out.dtype)

    def shared_overlap(self, num_tokens):
        return num_tokens <= CAPPED_UP_TO


def layer(seed=0):
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


@pytest.mark.parametrize("n", [16, CAPPED_UP_TO, CAPPED_UP_TO + 1, 512])
def test_eager_streams_and_bitwise(n):
    torch.cuda.set_device(DEVICE)
    mlp = layer()
    h = inputs(n, n)
    want = serial(mlp, h)
    mlp.experts.streams.clear()
    mlp.shared_experts.streams.clear()
    main = torch.cuda.current_stream()
    got = mlp.forward(h)
    assert torch.cuda.current_stream() == main
    assert torch.equal(got, want)
    if n <= CAPPED_UP_TO:
        assert mlp.experts.streams == [main]
        assert mlp.shared_experts.streams == [mlp._shared_stream]
    else:
        assert mlp.experts.streams == [mlp._routed_stream]
        assert mlp.shared_experts.streams == [main]


@pytest.mark.parametrize("n", [CAPPED_UP_TO, 512])
def test_cuda_graph_replays_bitwise(n):
    torch.cuda.set_device(DEVICE)
    mlp = layer(seed=1)
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


@pytest.mark.parametrize("n", [CAPPED_UP_TO, 512])
def test_breakable_graph_keeps_overlap_bitwise(n):
    """Production prefill steps of <= 2048 instance tokens run in breakable CUDA
    graphs (a segment ends at each eager attention op): the layer forks and joins
    inside one segment, with eager breaks before and after it."""
    torch.cuda.set_device(DEVICE)
    mlp = layer(seed=2)
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
    mlp = layer()
    mlp.shared_experts.gemm_rs_ar = object()
    with pytest.raises(RuntimeError, match="persistent"):
        mlp.forward(inputs(512, 0))
    mlp = layer()
    mlp.shared_experts._prefill_situ = object()
    with pytest.raises(RuntimeError, match="persistent"):
        mlp.forward(inputs(16, 0))
