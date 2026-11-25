# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 NVFP4 checkpoint, TP (non-SP) specialized decode MoE providers
(``specialized_decode.cutedsl.moe.nvfp4``) against the stock layers on identical
inputs, one GPU, at a TP16 rank's shapes:

- ``test_nvfp4_experts_match_stock``: the stock routed-experts layer production
  builds for one TP16 MoE layer (``FusedMoEFactory`` with ``KimiMoE``'s arguments:
  896 experts, top 16, SiTU beta 4 / linear 25, sigmoid grouped top-k with an FP32
  correction bias; intermediate 3072 / 16 = 192 zero-padded to 256 as ``KimiMoE``
  pads it) under ``ModelOptNvFp4Config`` and moe-backend auto (FLASHINFER_TRTLLM:
  the monolithic TRT-LLM-Gen kernel, routing from the logits), random NVFP4 weights
  (E2M1, E4M3 block scales per 16, per-expert global and input scales), processed by
  the layer's ``process_weights_after_loading``. Stock = the layer's
  ``apply_monolithic`` on (latent, logits); ours = ``Nvfp4Experts.run``: bitwise,
  eager repeat and CUDA-graph replay bitwise.
- ``test_mxfp8_shared_expert_matches_stock``: stock ``KimiMLP`` (a TP16 rank's shared
  expert: gate_up [768, 7168], down [7168, 384], ModelOpt MXFP8 W8A8, SiTU, no
  reduce) after load + the SM107 dispatch kernel, vs ``Mxfp8SharedExpert.run``: the
  same kernels on the same operands, bitwise at the CuTe DSL rows; then graph replay.
- ``test_refusals``: the cause functions refuse what the providers cannot run.

Needs a Blackwell-family GPU with FlashInfer's TRT-LLM fused MoE (SiTU)."""

import os
import tempfile
from types import SimpleNamespace

import pytest

EXPERTS, TOPK, LATENT, HIDDEN = 896, 16, 3584, 7168
TP, INTER_CKPT, INTER = 16, 3072 // 16, 256
SHARED_INTER = 2 * 3072 // TP  # two shared experts, TP16 shard
BETA, LINEAR_BETA = 4.0, 25.0
PREFIX = "language_model.model.layers.1.mlp"
ROWS = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1536)


def _torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability_family(100):
        pytest.skip("Blackwell-family TRT-LLM-Gen kernels")
    return torch


class _Dist:
    """A one-rank TP group (the parallel linears and the MoE layer need one)."""

    def __init__(self, dev, config):
        import torch

        from vllm.config import set_current_vllm_config
        from vllm.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        self.tmp = tempfile.TemporaryDirectory()
        self.config = config
        with set_current_vllm_config(config), torch.device(dev):
            init_distributed_environment(
                world_size=1, rank=0, local_rank=dev.index,
                distributed_init_method=f"file://{os.path.join(self.tmp.name, 's')}",
                backend="nccl",
            )  # fmt: skip
            initialize_model_parallel(1, 1)

    def close(self):
        from vllm.distributed import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
        self.tmp.cleanup()


@pytest.fixture
def should_do_global_cleanup_after_test():
    """No per-case global cleanup (tests/conftest.py's ``cleanup_fixture``): it tore
    the module-scoped ``dist`` groups down after the first case, so the next case
    found no TP group; ``dist`` cleans up once at module end."""
    return False


@pytest.fixture(scope="module")
def dist():
    torch = _torch()
    from vllm.config import VllmConfig

    config = VllmConfig()
    config.kernel_config.moe_backend = "auto"  # production: moe-backend auto
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    device = torch.device("cuda", torch.cuda.current_device())
    d = _Dist(device, config)
    init_workspace_manager(device)
    yield d
    reset_workspace_manager()
    d.close()


def _gen(seed):
    import torch

    return torch.Generator(device="cuda").manual_seed(seed)


def _e4m3(shape, lo, hi, gen):
    """Random positive E4M3 block scales in [lo, hi]."""
    import torch

    values = torch.rand(shape, generator=gen, device="cuda") * (hi - lo) + lo
    return values.to(torch.float8_e4m3fn)


def nvfp4_layer(dev, config):
    """The stock NVFP4 routed-experts runner of one TP16 MoE layer (module
    docstring), loaded with random checkpoint values and processed."""
    import torch

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4FusedMoE,
    )
    from vllm.model_executor.layers.quantization.utils import flashinfer_utils as fu

    if not fu.has_flashinfer_situ_activation():
        pytest.skip("FlashInfer without the SiTU activation: not the production MoE")
    quant = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True, kv_cache_quant_algo=None,
        exclude_modules=[], group_size=16,
    )  # fmt: skip
    gen = _gen(11)
    with set_current_vllm_config(config), torch.device(dev):
        bias = torch.randn(EXPERTS, generator=gen, device="cuda") * 0.05
        runner = FusedMoEFactory(
            num_experts=EXPERTS, top_k=TOPK, hidden_size=LATENT,
            intermediate_size=INTER, params_dtype=torch.bfloat16,
            activation="situ", activation_situ_beta=BETA,
            activation_situ_linear_beta=LINEAR_BETA, renormalize=True,
            quant_config=quant, use_grouped_topk=True, num_expert_group=1,
            topk_group=1, prefix=f"{PREFIX}.experts", scoring_func="sigmoid",
            e_score_correction_bias=bias.float(), routed_scaling_factor=1.0,
            tp_size=1, dp_size=1, pcp_size=1, skip_padding=True,
        )  # fmt: skip
        layer = runner.routed_experts
        method = layer.quant_method
        assert isinstance(method, ModelOptNvFp4FusedMoE), type(method)
        u8 = torch.uint8
        # The checkpoint slice of a TP16 rank: rows [0, 192) gate, [INTER, INTER +
        # 192) up; w2 columns [0, 192); the padding stays zero (``KimiMoE``).
        w13 = torch.zeros_like(layer.w13_weight)
        s13 = torch.zeros_like(layer.w13_weight_scale)
        for start in (0, INTER):
            rows = slice(start, start + INTER_CKPT)
            w13[:, rows] = torch.randint(
                0, 256, w13[:, rows].shape, generator=gen, device="cuda", dtype=u8
            )
            s13[:, rows] = _e4m3(s13[:, rows].shape, 1.0, 8.0, gen)
        w2 = torch.zeros_like(layer.w2_weight)
        s2 = torch.zeros_like(layer.w2_weight_scale)
        w2[..., : INTER_CKPT // 2] = torch.randint(
            0, 256, w2[..., : INTER_CKPT // 2].shape, generator=gen, device="cuda",
            dtype=u8,
        )  # fmt: skip
        s2[..., : INTER_CKPT // 16] = _e4m3(s2[..., : INTER_CKPT // 16].shape, 1, 8, gen)
        layer.w13_weight.data.copy_(w13)
        layer.w13_weight_scale.data.copy_(s13)
        layer.w2_weight.data.copy_(w2)
        layer.w2_weight_scale.data.copy_(s2)
        g = torch.rand(EXPERTS, generator=gen, device="cuda") * 1e-3 + 5e-4
        layer.w13_weight_scale_2.data.copy_(g[:, None].expand_as(layer.w13_weight_scale_2))
        layer.w2_weight_scale_2.data.copy_(g * 2)
        layer.w13_input_scale.data.fill_(3.0 / (6 * 448))
        layer.w2_input_scale.data.fill_(2.0 / (6 * 448))
        method.process_weights_after_loading(layer)
        assert method.moe_kernel is not None
    return runner, layer


def _inputs(rows, seed):
    import torch

    gen = _gen(seed)
    latent = torch.randn(rows, LATENT, generator=gen, device="cuda").bfloat16()
    logits = torch.randn(rows, EXPERTS, generator=gen, device="cuda") * 2
    return latent, logits


def _graph(fn):
    import torch

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        torch.cuda.current_stream().synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = fn()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, out


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def test_nvfp4_experts_match_stock(dist):
    """``Nvfp4Experts`` is the stock layer's own monolithic experts call (routing in
    the kernel): bitwise the stock output, eager repeat and CUDA-graph replay bitwise.
    Also records why the routed-ids entry is not used: on the stock router's routes
    with its FP32 weights it is 2.7-3.1e-3 (rel) off the stock layer (the kernel's own
    routing hands BF16 weights; with BF16-rounded weights it is bitwise), and the
    kernel's routing and vLLM's grouped top-k op disagree on near-ties (1 row in 256 /
    1536 here)."""
    import torch

    from vllm.config import set_current_vllm_config
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.nvfp4 import (
        Nvfp4Experts,
    )

    dev = torch.device("cuda", torch.cuda.current_device())
    _, layer = nvfp4_layer(dev, dist.config)
    method = layer.quant_method
    with set_current_vllm_config(dist.config), torch.inference_mode():
        ours = Nvfp4Experts(layer)
        assert not ours.deferred(8)  # (TP1: the layer never defers)
        for rows in ROWS:
            latent, logits = _inputs(rows, 1000 + rows)
            stock = method.apply_monolithic(layer, latent, logits)
            assert isinstance(stock, torch.Tensor), type(stock)
            stock = stock.clone()
            got = ours.run(latent, logits).clone()
            repeat = ours.run(latent, logits).clone()
            torch.cuda.synchronize()
            assert torch.isfinite(got).all()
            assert torch.equal(got, stock), f"M{rows}: not the stock output"
            assert torch.equal(got, repeat), f"M{rows}: eager repeat not bitwise"
            graph, out = _graph(lambda: ours.run(latent, logits))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, got), f"M{rows}: graph replay != eager"
            latent.copy_(_inputs(rows, 7 + rows)[0])
            graph.replay()
            fresh = ours.run(latent, logits)
            torch.cuda.synchronize()
            assert torch.equal(out, fresh), f"M{rows}: graph ignores new input"
            del graph


def mxfp8_shared(dev, config):
    import torch

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
    from vllm.models.kimi_k3.nvidia.model import KimiMLP
    from vllm.models.kimi_k3.nvidia.mxfp8 import enable_kimi_k3_mxfp8_dispatch

    quant = ModelOptMxFp8Config(
        is_checkpoint_mxfp8_serialized=True, kv_cache_quant_algo=None,
        exclude_modules=[],
    )  # fmt: skip
    from vllm.utils.torch_utils import set_default_torch_dtype

    gen = _gen(23)
    with (
        set_current_vllm_config(config),
        torch.device(dev),
        set_default_torch_dtype(torch.bfloat16),
    ):
        mlp = KimiMLP(
            hidden_size=HIDDEN, intermediate_size=SHARED_INTER, hidden_act="situ",
            quant_config=quant, reduce_results=False,
            prefix=f"{PREFIX}.shared_experts", activation_situ_beta=BETA,
            activation_situ_linear_beta=LINEAR_BETA,
        )  # fmt: skip
        for linear in (mlp.gate_up_proj, mlp.down_proj):
            w = linear.weight
            values = torch.randn(w.shape, generator=gen, device="cuda") * 2
            w.data.copy_(values.to(w.dtype))
            s = linear.weight_scale
            # E8M0 exponents around 2^-8 (biased 119..122): O(1) outputs
            exps = torch.randint(119, 123, s.shape, generator=gen, device="cuda")
            s.data.view(torch.uint8).copy_(exps.to(torch.uint8))
            linear.quant_method.process_weights_after_loading(linear)
        enable_kimi_k3_mxfp8_dispatch(mlp)
    return mlp


def test_mxfp8_shared_expert_matches_stock(dist):
    import torch

    from vllm.config import set_current_vllm_config
    from vllm.models.kimi_k3.nvidia.mxfp8 import use_cute_dsl
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.nvfp4 import (
        Mxfp8SharedExpert,
        mxfp8_shared_cause,
    )

    dev = torch.device("cuda", torch.cuda.current_device())
    mlp = mxfp8_shared(dev, dist.config)
    assert mxfp8_shared_cause(mlp) is None, mxfp8_shared_cause(mlp)
    with set_current_vllm_config(dist.config), torch.inference_mode():
        ours = Mxfp8SharedExpert(mlp, ROWS)
        report = []
        for rows in ROWS:
            x = torch.randn(rows, HIDDEN, generator=_gen(rows), device="cuda")
            x = x.bfloat16()
            stock = mlp(x).clone()
            out = torch.empty_like(stock)
            got = ours.run(x, out).clone()
            torch.cuda.synchronize()
            rel = _rel(got, stock)
            bitwise = torch.equal(got, stock)
            report.append(f"M{rows} rel {rel:.2e} bitwise {bitwise}")
            # Same kernels on the same operands where stock runs CuTe DSL on both
            # GEMMs (Mxfp8Projection's GENERIC kind = stock's per-call rule).
            if use_cute_dsl(2 * SHARED_INTER, HIDDEN, rows) and use_cute_dsl(
                HIDDEN, SHARED_INTER, rows
            ):
                assert bitwise, f"M{rows}: rel {rel:.3e} ({report})"
            assert rel <= 1e-2, f"M{rows}: rel {rel:.3e} ({report})"
            graph, gout = _graph(lambda: ours.run(x, out))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(gout, got), f"M{rows}: graph replay != eager"
            del graph
        print("MXFP8 shared vs stock:", "; ".join(report))


def test_refusals():
    _torch()
    
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.nvfp4 import (
        mxfp8_shared_cause,
        nvfp4_experts_cause,
        nvfp4_moe_cause,
    )

    class _Method:
        pass

    routed = SimpleNamespace(quant_method=_Method())
    assert "not the ModelOpt NVFP4" in nvfp4_experts_cause(routed)
    assert mxfp8_shared_cause(None) == "no shared expert"
    module = SimpleNamespace(use_mega_moe=True)
    assert "MegaMoE" in nvfp4_moe_cause(module)
    module = SimpleNamespace(experts=SimpleNamespace(routed_experts=routed))
    assert "not the ModelOpt NVFP4" in nvfp4_moe_cause(module)
