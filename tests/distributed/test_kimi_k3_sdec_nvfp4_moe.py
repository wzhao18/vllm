# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 NVFP4 checkpoint, TP (non-SP) specialized decode MoE layer
(``specialized_decode.cutedsl.moe.nvfp4.Nvfp4MoE``) against the stock ``KimiMoE`` on
one node (4 Rubin GPUs, MoE TP4, no EP).

The stock layer is built as the checkpoint builds it: ``KimiMoE`` with K3.1's MoE
arguments (896 experts, top 16, latent 3584 with the routed RMSNorm, two shared
experts, SiTU 4 / 25, sigmoid grouped top-k with a correction bias) under the
checkpoint's ``modelopt_mixed`` recipe (routed experts NVFP4 W4A4, shared experts
MXFP8, router / latent projections BF16), moe-backend auto (FLASHINFER_TRTLLM). Random
weights: the replicated ones (router, latent down / up, norm, bias) equal on every
rank, the TP-sharded ones (experts, shared expert) per rank; processed as the loader
does (``process_weights_after_loading`` + the SM107 MXFP8 dispatch kernel).

Per row count R: stock = ``KimiMoE.forward(x)`` (all-reduced [R, 7168]); ours =
``Nvfp4MoE.run`` (stock router + latent down, the layer's NVFP4 TRT-LLM-Gen experts,
MXFP8 shared expert + RS push, the wide tail into the output ring), read from the
ring; then again with the layer's top-k finalize deferred to the tail (R <= 128). Checks: finite; ours vs stock within ``TOL`` (the experts' operands are the
same; the latent all-reduce, RMS and up projection sum in another FP32 order, and
stock's monolithic kernel routes from the logits itself); our eager repeat and our
CUDA-graph replay bitwise; every rank's output equal (one multicast ring)."""

import time
import zlib

import pytest
import torch

from vllm.utils.network_utils import get_open_port

WORLD = 4
ROWS = (1, 4, 8, 16, 32, 64, 96, 128, 256, 512)
TOL = 2e-2
DEFER_ROWS = 128  # (the stock tail op's cap)
TIMEOUT = 1800
PREFIX = "language_model.model.layers.1.mlp"
SENTINEL = -0x80000000


def _rubin_node() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        return False
    capabilities = {torch.cuda.get_device_capability(i) for i in range(WORLD)}
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        multicast = _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)
    except Exception:
        multicast = False
    return capabilities == {(10, 7)} and multicast


pytestmark = [
    pytest.mark.distributed(num_gpus=WORLD),
    pytest.mark.skipif(
        not _rubin_node(), reason="needs 4 Rubin (SM107) GPUs with NVLink multicast"
    ),
]


def _moe_config():
    """K3.1's MoE fields (hf-1d82032 config.json)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        hidden_size=7168, moe_intermediate_size=3072, num_experts=896,
        num_experts_per_token=16, moe_renormalize=True,
        routed_expert_hidden_size=3584, latent_moe_use_norm=True,
        routed_scaling_factor=1.0, use_grouped_topk=True, num_expert_group=1,
        topk_group=1, moe_router_activation_func="sigmoid", num_shared_experts=2,
        hidden_act="situ", activation_situ_beta=4.0, activation_situ_linear_beta=25.0,
        rms_norm_eps=1e-5,
    )  # fmt: skip


def _quant_config():
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )

    layers = {
        f"{PREFIX}.experts": {"quant_algo": "NVFP4", "group_size": 16},
        f"{PREFIX}.shared_experts.gate_up_proj": {"quant_algo": "MXFP8"},
        f"{PREFIX}.shared_experts.down_proj": {"quant_algo": "MXFP8"},
    }
    return ModelOptMixedPrecisionConfig.from_config(
        {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": None,
                "group_size": 16,
                "quantized_layers": layers,
            }
        }
    )


def _vllm_config():
    from vllm.config import VllmConfig

    config = VllmConfig()
    config.kernel_config.moe_backend = "auto"
    parallel = config.parallel_config
    parallel.tensor_parallel_size = parallel.world_size = WORLD
    return config


def _fill(module, rank):
    """Random values: replicated parameters equal on every rank, sharded ones per
    rank (seeded by name)."""
    replicated = ("gate.", "routed_expert_down_proj", "routed_expert_up_proj",
                  "routed_expert_norm", "e_score_correction_bias")  # fmt: skip
    for name, param in module.named_parameters():
        shared = any(key in name for key in replicated)
        seed = zlib.crc32(name.encode()) + (0 if shared else 7919 * (rank + 1))
        gen = torch.Generator(device=param.device).manual_seed(seed)
        data = param.data
        if data.dtype == torch.uint8 and "scale" not in name:  # packed E2M1
            data.copy_(torch.randint(0, 256, data.shape, generator=gen,
                                     device=data.device, dtype=torch.uint8))  # fmt: skip
        elif "experts.w" in name and data.dtype == torch.float8_e4m3fn:  # NVFP4 SF
            values = torch.rand(data.shape, generator=gen, device=data.device) * 7 + 1
            data.copy_(values.to(data.dtype))
        elif "weight_scale_2" in name:
            data.copy_(torch.rand(data.shape, generator=gen, device=data.device)
                       * 1e-3 + 5e-4)  # fmt: skip
        elif "input_scale" in name:
            data.fill_(3.0 / (6 * 448))
        elif "weight_scale" in name:  # MXFP8 E8M0 exponents around 2^-8
            exps = torch.randint(119, 123, data.shape, generator=gen,
                                 device=data.device)  # fmt: skip
            data.view(torch.uint8).copy_(exps.to(torch.uint8))
        elif data.dtype == torch.float8_e4m3fn:  # MXFP8 weight
            values = torch.randn(data.shape, generator=gen, device=data.device) * 2
            data.copy_(values.to(data.dtype))
        elif "norm" in name:
            data.copy_(1 + 0.1 * torch.randn(data.shape, generator=gen,
                                              device=data.device))  # fmt: skip
        elif "bias" in name:
            data.copy_(0.05 * torch.randn(data.shape, generator=gen, device=data.device))
        else:  # BF16 projections
            scale = data.shape[-1] ** -0.5
            data.copy_(scale * torch.randn(data.shape, generator=gen,
                                           device=data.device))  # fmt: skip


def _stock_layer(config, rank, device):
    from vllm.models.kimi_k3.nvidia.model import KimiMoE
    from vllm.models.kimi_k3.nvidia.mxfp8 import enable_kimi_k3_mxfp8_dispatch

    from vllm.utils.torch_utils import set_default_torch_dtype

    with torch.device(device), set_default_torch_dtype(torch.bfloat16):
        module = KimiMoE(
            _moe_config(), config, quant_config=_quant_config(), prefix=PREFIX,
            layer_idx=1,
        )  # fmt: skip
    _fill(module, rank)
    # (zero padding of the experts' intermediate, as KimiMoE zeroes it: none at TP4)
    for sub in module.modules():
        method = getattr(sub, "quant_method", None)
        if method is not None and hasattr(method, "process_weights_after_loading"):
            method.process_weights_after_loading(sub)
    enable_kimi_k3_mxfp8_dispatch(module)
    return module


def _read_ring(box, rows, group):
    """The wide tail's output rows once every rank's multicast landed, then re-armed
    (the role of the next AttnRes)."""
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)
    ring = box.out.buffer[0]
    out = ring[:rows].clone()
    ring[:rows].view(torch.int32).fill_(SENTINEL)
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)
    return out


def _case(config, rank, device, group, fused_up=False):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import tail_rows

    # ``fused_up``: the wide tail's up projection fused with the output ring
    # (``tail_rows.UP_ROWS``, ``up_tmem.UpTmem64``) at every row, as TP16 runs the
    # NVFP4 layers' 1-80 and 129- rows; else cuBLAS + OutRows (this TP has no entry).
    saved = dict(tail_rows.UP_ROWS)
    if fused_up:
        tail_rows.UP_ROWS[WORLD] = "1-:1"
    else:
        tail_rows.UP_ROWS.pop(WORLD, None)
    try:
        _case_body(config, rank, device, group)
    finally:
        tail_rows.UP_ROWS.clear()
        tail_rows.UP_ROWS.update(saved)


def _case_body(config, rank, device, group):
    from vllm.forward_context import set_forward_context
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.nvfp4 import (
        Nvfp4MoE,
        nvfp4_moe_cause,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.prepared import (
        MoEScratch,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.tail_rows import (
        PreparedTailRows,
    )

    module = _stock_layer(config, rank, device)
    assert nvfp4_moe_cause(module) is None, nvfp4_moe_cause(module)
    owner = (group, device, WORLD, rank)
    state = MoEScratch(WORLD, device, ROWS, owner=owner, tuned=False, rows=True)
    norm = module.routed_expert_norm
    rows = PreparedTailRows(
        WORLD, rank, module.routed_expert_up_proj.weight, norm.weight,
        norm.variance_epsilon, ROWS, state.rows, 0,
    )  # fmt: skip
    ours = Nvfp4MoE(module, state, rows, ROWS)
    report, refs = [], {}
    with torch.inference_mode(), set_forward_context(None, config):
        for r in ROWS:
            gen = torch.Generator(device=device).manual_seed(100 + r)  # same x per rank
            x = torch.randn(r, 7168, generator=gen, device=device).bfloat16()
            stock = module(x.clone()).clone()
            refs[r] = x, stock
            out = torch.empty_like(x)
            ours.run(x, out, consume=lambda o: o)
            got = _read_ring(state.rows, r, group)
            ours.run(x, out, consume=lambda o: o)
            repeat = _read_ring(state.rows, r, group)
            assert torch.isfinite(got).all() and torch.isfinite(stock).all(), r
            assert torch.equal(got, repeat), f"R{r}: eager repeat not bitwise"
            rel = ((got.float() - stock.float()).norm() / stock.float().norm()).item()
            report.append(f"R{r} {rel:.2e}")
            assert rel <= TOL, f"rank {rank} R{r}: rel {rel:.3e} ({report})"
            gathered = [torch.empty_like(got) for _ in range(WORLD)]
            torch.distributed.all_gather(gathered, got, group=group)
            assert all(torch.equal(g, got) for g in gathered), f"R{r}: ranks differ"
            # CUDA graph: capture on a side stream, replay, read the ring
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph, stream=stream):
                    ours.run(x, out, consume=lambda o: o)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            torch.distributed.barrier(group=group)
            graph.replay()
            replay = _read_ring(state.rows, r, group)
            assert torch.equal(replay, got), f"R{r}: graph replay != eager"
            del graph
    # The layer's deferred top-k finalize (the stock runner asks for it at TP8 /
    # TP16 for small batches): the experts return their unfinalized output and the
    # wide tail's LatentRows sums the top k. Against the stock references above.
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.prepared import (
        deferred_rows,
    )

    module.experts.moe_config.defer_moe_finalize(DEFER_ROWS)
    late = deferred_rows(module, ROWS)
    assert late[0] == tuple(r for r in ROWS if r <= DEFER_ROWS), late
    rows = PreparedTailRows(
        WORLD, rank, module.routed_expert_up_proj.weight, norm.weight,
        norm.variance_epsilon, ROWS, state.rows, 1, deferred=late,
    )  # fmt: skip
    ours = Nvfp4MoE(module, state, rows, ROWS)
    deferred = []
    with torch.inference_mode(), set_forward_context(None, config):
        for r in ROWS:
            x, stock = refs[r]
            out = torch.empty_like(x)
            ours.run(x, out, consume=lambda o: o)
            got = _read_ring(state.rows, r, group)
            rel = ((got.float() - stock.float()).norm() / stock.float().norm()).item()
            deferred.append(f"R{r} {rel:.2e}")
            assert rel <= TOL, f"rank {rank} deferred R{r}: rel {rel:.3e}"
    if rank == 0:
        print("NVFP4 MoE vs stock KimiMoE (rel):", "; ".join(report), flush=True)
        print("deferred finalize (rel):", "; ".join(deferred), flush=True)


def _worker(rank: int, port: int) -> None:
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.platforms import current_platform
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    current_platform.import_kernels()
    config = _vllm_config()
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=WORLD, rank=rank, local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )  # fmt: skip
        initialize_model_parallel(tensor_model_parallel_size=WORLD)
        group = get_tp_group().device_group
        try:
            from torch.distributed._symmetric_memory import enable_symm_mem_for_group

            enable_symm_mem_for_group(group.group_name)
        except (ImportError, AttributeError):
            pass
        init_workspace_manager(device)
        try:
            _case(config, rank, device, group)
            _case(config, rank, device, group, fused_up=True)
        finally:
            torch.accelerator.synchronize()
            reset_workspace_manager()
    cleanup_dist_env_and_memory()


def test_nvfp4_moe_matches_stock_layer():
    """Nvfp4MoE vs the stock KimiMoE (NVFP4 experts, MXFP8 shared) at MoE TP4."""
    pytest.importorskip("cutlass")
    import torch.multiprocessing as mp

    port = get_open_port()
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=_worker, args=(rank, port)) for rank in range(WORLD)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + TIMEOUT
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    hung = [p.pid for p in processes if p.is_alive()]
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
    assert not hung, f"ranks {hung} did not finish in {TIMEOUT} s (a hang?)"
    assert [p.exitcode for p in processes] == [0] * WORLD
