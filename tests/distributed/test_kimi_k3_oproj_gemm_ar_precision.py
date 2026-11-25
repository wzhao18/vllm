# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision parity of the TP fused GEMM-AR o_proj with the stock layer.

At TP > 1 the Kimi K3.1 KDA and DSA o_proj run ``GemmRsAr.apply`` (fused GEMM +
all-reduce) on the BF16 o_norm / gate output instead of the stock
``RowParallelLinear`` forward + all-reduce. The KDA o_proj is an MXFP8
checkpoint linear (served by ``KimiK3Mxfp8Kernel``); the stock layer quantizes
its BF16 input to MXFP8 (``mxfp8_e4m3_quantize``) before the MXFP8 GEMM. The
fused op must do the same: quantize the input with the stock op and run the
MXFP8 MMA, i.e. neither a BF16 GEMM (higher precision) nor anything lower. The
DSA o_proj is BF16 in the checkpoint: BF16 GEMM on both routes.

Per rank, identical inputs: fused vs stock layer vs an exact float64 reference
of the stock math (MXFP8-dequantized input and weight). A BF16-input GEMM of
the same weight (what a fused o_proj that skipped the input quantization would
compute) is far from that reference, so the check separates the two.
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

# K3.1 o_proj at TP8: K per rank 12 heads x 128 (KDA) / 8 heads x 128 (DSA v),
# N = hidden 7168. Rows: one 128-row tile, an odd count, a 16K-token chunk share.
_N = 7168
_K_KDA = 1536
_K_DSA = 1024
_ROWS = (128, 1000, 4096)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return ((a - b).square().mean().sqrt() / b.square().mean().sqrt()).item()


def _dequant(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """MXFP8 [R, K] E4M3 codes + row-major [R, K/32] E8M0 scales -> float64."""
    scale = torch.exp2(sf.view(torch.uint8).double() - 127.0)
    return q.double() * scale.repeat_interleave(32, dim=1)


def _mxfp8_linear(k_total: int, w: torch.Tensor, device):
    """A row-parallel MXFP8 linear on the production kernel (KimiK3Mxfp8Kernel,
    as the K3.1 MXFP8 checkpoint o_proj), weights from ``w`` (this rank's BF16
    [N, K] shard)."""
    from vllm.config.quantization import QuantizationConfigArgs
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
    )
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )
    from vllm.models.kimi_k3.nvidia.mxfp8 import KimiK3Mxfp8Kernel

    with torch.device(device):
        linear = RowParallelLinear(
            k_total,
            _N,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=OnlineQuantizationConfig(
                QuantizationConfigArgs(linear="mxfp8")
            ),
            prefix="model.layers.0.self_attn.o_proj",
        )
    method = linear.quant_method
    assert type(method.kernel) is FlashInferCutedslMxfp8LinearKernel
    method.kernel = KimiK3Mxfp8Kernel(method.kernel.config)
    return linear


def _bf16_linear(k_total: int, device):
    from vllm.model_executor.layers.linear import RowParallelLinear

    with torch.device(device):
        return RowParallelLinear(
            k_total,
            _N,
            bias=False,
            params_dtype=torch.bfloat16,
            prefix="model.layers.3.self_attn.o_proj",
        )


def _worker(local_rank: int, world_size: int, port: int) -> None:
    device = torch.device("cuda", local_rank)
    torch.accelerator.set_device_index(device)
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
        }
    )
    init_distributed_environment()
    config = VllmConfig()
    config.model_config = ModelConfig(dtype="bfloat16")
    config.kernel_config.linear_backend = "flashinfer_cutedsl"
    with set_current_vllm_config(config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        _check(local_rank, world_size, device)
    cleanup_dist_env_and_memory()


def _check(rank: int, world_size: int, device) -> None:
    from vllm.model_executor.kernels.linear.cute_dsl import gemm_rs_ar as g
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
    )
    from vllm.model_executor.layers.quantization.utils import mxfp8_utils
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )
    from vllm.models.kimi_k3.nvidia.mxfp8 import unswizzle_mxfp8_scale

    op = g.GemmRsAr(max_M=max(_ROWS), N=_N, all_reduce=True)
    gen = torch.Generator(device=device).manual_seed(100 + rank)

    # --- KDA o_proj: MXFP8 checkpoint linear -------------------------------
    w = (torch.randn(_N, _K_KDA, generator=gen, device=device) * _K_KDA**-0.5).to(
        torch.bfloat16
    )
    kda = _mxfp8_linear(_K_KDA * world_size, w, device)
    assert op.can_run(kda), "the fused o_proj must bind the MXFP8 linear"
    kda.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    kda.quant_method.process_weights_after_loading(kda)
    assert kda.weight.dtype == torch.float8_e4m3fn
    w_q = kda.weight.t()  # KimiK3Mxfp8Kernel stores a [K, N] column-major view
    assert w_q.shape == (_N, _K_KDA)
    w_dq = _dequant(w_q, unswizzle_mxfp8_scale(kda.weight_scale, _N, _K_KDA))

    # Record the activation quantization the fused op performs.
    seen = []
    orig_q = mxfp8_utils.mxfp8_e4m3_quantize

    def spy(x, *a, **k):
        seen.append(x.clone())
        return orig_q(x, *a, **k)

    for M in _ROWS:
        x = (torch.randn(M, _K_KDA, generator=gen, device=device) * 2).to(
            torch.bfloat16
        )
        stock = kda(x)[0]  # stock: MXFP8 linear (quantizes x) + all-reduce
        seen.clear()
        mxfp8_utils.mxfp8_e4m3_quantize = spy
        try:
            fused = op.apply(x, kda).clone()
        finally:
            mxfp8_utils.mxfp8_e4m3_quantize = orig_q
        torch.accelerator.synchronize(device)
        # The fused op quantizes exactly its BF16 input with the stock op.
        assert len(seen) == 1 and torch.equal(seen[0], x)

        x_q, x_sf = orig_q(x, is_sf_swizzled_layout=True)
        x_sf = unswizzle_mxfp8_scale(x_sf, M, _K_KDA)
        exact = _dequant(x_q, x_sf) @ w_dq.T  # this rank's partial, float64
        dist.all_reduce(exact)
        bf16_in = x.double() @ w_dq.T  # same weight, BF16 (unquantized) input
        dist.all_reduce(bf16_in)

        e_fused, e_stock = _rel(fused, exact), _rel(stock, exact)
        e_bf16 = _rel(bf16_in, exact)
        d = _rel(fused, stock)
        if rank == 0:
            print(
                f"KDA o_proj M={M}: fused-vs-exactMXFP8 {e_fused:.2e} "
                f"stock-vs-exactMXFP8 {e_stock:.2e} fused-vs-stock {d:.2e} "
                f"bf16input-vs-exactMXFP8 {e_bf16:.2e}",
                flush=True,
            )
        # Both are the MXFP8 math up to BF16 output rounding / reduction order.
        assert e_fused < 4e-3 and e_stock < 4e-3 and d < 4e-3
        # A BF16-input o_proj differs by the activation quantization error.
        assert e_bf16 > 5 * max(e_fused, e_stock, d)

        # Pre-quantized input (the KDA gated o_norm's MXFP8 output): the fused
        # op skips its quantize and gives the BF16-input result bitwise.
        x_q, x_sf_sw = orig_q(x, is_sf_swizzled_layout=True)
        qa = QuantizedActivation(
            data=x_q,
            scale=x_sf_sw.view(torch.uint8),
            orig_dtype=torch.bfloat16,
            orig_shape=x.shape,
            quant_key=kMxfp8Dynamic,
        )
        assert op.should_run(qa) == op.should_run(x)
        seen.clear()
        mxfp8_utils.mxfp8_e4m3_quantize = spy
        try:
            fused_q = op.apply(qa, kda).clone()
        finally:
            mxfp8_utils.mxfp8_e4m3_quantize = orig_q
        assert not seen, "a pre-quantized activation must not be re-quantized"
        assert torch.equal(fused_q, fused)

    # The KDA producer: gated o_norm written as MXFP8 (kda_norm) into the fused
    # op == the BF16 gated o_norm (in place) into the fused op, bitwise.
    from vllm.models.kimi_k3.nvidia.ops import kda_norm

    heads = _K_KDA // 128
    for M in _ROWS:
        core = (torch.randn(1, M, heads, 128, generator=gen, device=device) * 3).to(
            torch.bfloat16
        )
        gate = (torch.randn(M, heads, 128, generator=gen, device=device) * 2).to(
            torch.bfloat16
        )
        nw = torch.rand(128, generator=gen, device=device) + 0.5
        qa = kda_norm.rmsnorm_gated_mxfp8(core, gate, nw, 1e-6)
        via_mx = op.apply(qa, kda).clone()
        kda_norm.rmsnorm_gated_(core, gate, nw, 1e-6)
        via_bf16 = op.apply(core.view(M, _K_KDA), kda).clone()
        assert torch.equal(via_mx, via_bf16), f"KDA o_norm MXFP8 route M={M}"

    # --- DSA o_proj: BF16 checkpoint linear --------------------------------
    wb = (torch.randn(_N, _K_DSA, generator=gen, device=device) * _K_DSA**-0.5).to(
        torch.bfloat16
    )
    dsa = _bf16_linear(_K_DSA * world_size, device)
    assert op.can_run(dsa)
    dsa.weight.data.copy_(wb)
    for M in _ROWS:
        x = torch.randn(M, _K_DSA, generator=gen, device=device).to(torch.bfloat16)
        stock = dsa(x)[0]
        fused = op.apply(x, dsa).clone()
        exact = x.double() @ wb.double().T
        dist.all_reduce(exact)
        e_fused, e_stock = _rel(fused, exact), _rel(stock, exact)
        if rank == 0:
            print(
                f"DSA o_proj M={M}: fused-vs-exact {e_fused:.2e} "
                f"stock-vs-exact {e_stock:.2e} fused-vs-stock {_rel(fused, stock):.2e}",
                flush=True,
            )
        assert e_fused < 4e-3 and e_stock < 4e-3 and _rel(fused, stock) < 4e-3
    dist.barrier()
    del op


@pytest.mark.distributed(num_gpus=2)
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="GEMM-RS/AR requires SM100",
)
def test_oproj_gemm_ar_matches_stock_precision(monkeypatch: pytest.MonkeyPatch):
    world_size = 2
    if torch.accelerator.device_count() < world_size:
        pytest.skip("needs two GPUs")
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "1")
    try:
        mp.spawn(_worker, args=(world_size, get_open_port()), nprocs=world_size)
    finally:
        cleanup_dist_env_and_memory()
