# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 NVFP4 checkpoint, TP specialized decode MoE layer at the tuned rows
(<= 64): ``Nvfp4MoE`` with ``Nvfp4Front`` and the front-mode MoeTail over the stock
experts' deferred output (``MoeTail(permuted=True)``), on 8 or 16 Rubin ranks (the
MoeTail's TP8 / TP16 geometries; MoE TP = world, no EP).

One process per rank, launched by the caller (e.g. ``srun`` over 2 / 4 nodes with
``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` / ``MASTER_ADDR`` / ``MASTER_PORT``); the
test skips unless ``WORLD_SIZE`` is 8 or 16. Per row count R:

- ours (``Nvfp4MoE.run``: front, deferred experts, MXFP8 shared, MoeTail into the
  tail box's ring) vs the stock ``KimiMoE.forward`` within ``TOL``; eager repeat and
  CUDA-graph replay bitwise; every rank's ring equal;
- parity: the permuted MoeTail's ring is bitwise the route-major front-mode MoeTail's
  (the MXFP4 layers' kernel) on the experts' rows gathered route-major (row t * 16 + k
  = permuted row map[t, k], weights as given): the same finalize FMAs in k order;
- routes: the stock router's top 16 on our logits equals it on the stock logits for
  every token whose 16th / 17th scores are apart by more than 1e-4 (clear margin).
The three tail parities rotate as over three MoE layers."""

import os

import pytest
import torch

ROWS = (1, 4, 8, 16, 32, 64)
TOL = 2e-2
DEFER_ROWS = 128
SENTINEL = -0x80000000
WORLD = int(os.environ.get("WORLD_SIZE", "0"))

pytestmark = pytest.mark.skipif(
    WORLD not in (8, 16) or "RANK" not in os.environ,
    reason="launch one process per rank on 8 or 16 Rubin GPUs (srun)",
)


def _read_ring(box, rows, group):
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)
    ring = box.out.buffer[0]
    out = ring[:rows].clone()
    ring[:rows].view(torch.int32).fill_(SENTINEL)
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)
    return out


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def test_nvfp4_moetail_matches_stock():
    from test_kimi_k3_sdec_nvfp4_moe import _vllm_config

    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.platforms import current_platform
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    rank, local = int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", "0"))
    torch.accelerator.set_device_index(local)
    device = torch.device("cuda", local)
    current_platform.import_kernels()
    config = _vllm_config()
    config.parallel_config.tensor_parallel_size = WORLD
    config.parallel_config.world_size = WORLD
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=WORLD, rank=rank, local_rank=local,
            distributed_init_method="env://",
        )  # fmt: skip
        initialize_model_parallel(tensor_model_parallel_size=WORLD)
        group = get_tp_group().device_group
        init_workspace_manager(device)
        try:
            _run(config, rank, device, group)
        finally:
            torch.accelerator.synchronize()
            reset_workspace_manager()
    cleanup_dist_env_and_memory()


def _run(config, rank, device, group):
    from test_kimi_k3_sdec_nvfp4_moe import _stock_layer

    from vllm.forward_context import set_forward_context
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        tail_runtime,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.nvfp4 import (
        Nvfp4Front,
        Nvfp4MoE,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.prepared import (
        MoEScratch,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.tail_rows import (
        PreparedTailRows,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj.prepared import (
        OprojMailbox,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.providers import (
        pack_front,
    )

    module = _stock_layer(config, rank, device)
    module.experts.moe_config.defer_moe_finalize(DEFER_ROWS)
    owner = (group, device, WORLD, rank)
    state = MoEScratch(WORLD, device, ROWS, owner=owner, tuned=False, rows=True)
    norm = module.routed_expert_norm
    up = module.routed_expert_up_proj.weight
    local_up = up.narrow(0, rank * (7168 // WORLD), 7168 // WORLD)
    rows = PreparedTailRows(
        WORLD, rank, up, norm.weight, norm.variance_epsilon, ROWS, state.rows, 0,
    )  # fmt: skip
    tail_box = tail_runtime.MoeTailBox(*owner)
    front_box = OprojMailbox(*owner, (1, 64, 3584 + 2 * 896))
    front = Nvfp4Front(
        module, pack_front(module, rank, WORLD), front_box, ROWS
    )

    def weights(ordinal, permuted):
        args = (WORLD, local_up, norm.weight, norm.variance_epsilon, ordinal, 3, ROWS)
        if permuted:
            return tail_runtime.MoeTailWeights(*args, permuted=True)
        # the route-major reference: the front-mode MoeTail at every row (the
        # MXFP4 layers' kernel below the published rows)
        published = tail_runtime.published
        tail_runtime.published = lambda *a, **k: False
        try:
            return tail_runtime.MoeTailWeights(*args)
        finally:
            tail_runtime.published = published

    layers = [
        Nvfp4MoE(
            module, state, rows, ROWS, front=front, tail=(weights(i, True), tail_box)
        )
        for i in range(3)
    ]
    reference = [weights(i, False) for i in range(3)]
    step = iter(range(10**6))
    report = []
    with torch.inference_mode(), set_forward_context(None, config):
        for r in ROWS:
            gen = torch.Generator(device=device).manual_seed(100 + r)
            x = torch.randn(r, 7168, generator=gen, device=device).bfloat16()
            stock = module(x.clone()).clone()
            out = torch.empty_like(x)
            ours = layers[next(step) % 3]
            ours.run(x, out, consume=lambda o: o)
            got = _read_ring(tail_box, r, group)
            layer = layers[next(step) % 3]
            layer.run(x, out, consume=lambda o: o)
            repeat = _read_ring(tail_box, r, group)
            assert torch.isfinite(got).all(), r
            assert torch.equal(got, repeat), f"R{r}: eager repeat not bitwise"
            rel = _rel(got, stock)
            assert rel <= TOL, f"rank {rank} R{r}: rel {rel:.3e}"
            gathered = [torch.empty_like(got) for _ in range(WORLD)]
            torch.distributed.all_gather(gathered, got, group=group)
            assert all(torch.equal(g, got) for g in gathered), f"R{r}: ranks differ"
            # parity with the route-major MoeTail on the same experts' rows
            latent, logits = front.route(x)
            partial = layer.experts.run(latent, logits)
            shared = layer.shared_out[:r]
            layer.shared.run(x, shared)
            w = partial.expert_weights[:r].contiguous()
            index = partial.expanded_idx_to_permuted_idx[:r]
            rows_p = partial.gemm2_permuted
            expanded = rows_p[index.clamp(min=0).reshape(-1)].contiguous()
            w_rm = torch.where(index >= 0, w, torch.zeros_like(w)).contiguous()
            i = next(step) % 3
            layers[i].tail.run(tail_box, rows_p, w, shared, index=index)
            perm = _read_ring(tail_box, r, group)
            reference[i].run(tail_box, expanded, w_rm, shared)
            route_major = _read_ring(tail_box, r, group)
            assert torch.equal(perm, route_major), f"R{r}: permuted != route-major"
            # clear-margin routes: the stock router on our logits vs on stock's
            ref_logits = module._maybe_overlap_router_and_down_proj(x)[1]
            bias = module.gate.e_score_correction_bias
            keys = [torch.sigmoid(t.float()) + bias for t in (logits, ref_logits)]
            top = [k.topk(17, dim=1) for k in keys]
            margin = top[1].values[:, 15] - top[1].values[:, 16]
            clear = margin > 1e-4
            ids = [t.indices[:, :16].sort(dim=1).values for t in top]
            assert torch.equal(ids[0][clear], ids[1][clear]), f"R{r}: clear routes"
            # CUDA graph
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            layer = layers[next(step) % 3]
            with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
                layer.run(x, out, consume=lambda o: o)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            torch.distributed.barrier(group=group)
            graph.replay()
            replay = _read_ring(tail_box, r, group)
            assert torch.equal(replay, got), f"R{r}: graph replay != eager"
            del graph
            report.append(f"R{r} {rel:.2e} clear {int(clear.sum())}/{r}")
    if rank == 0:
        print("NVFP4 MoeTail vs stock KimiMoE (rel):", "; ".join(report), flush=True)
