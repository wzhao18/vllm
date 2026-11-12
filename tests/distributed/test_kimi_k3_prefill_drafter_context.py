# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 prefill on a TP group (2 ranks, or 4 when there are 4 GPUs):

- the DSpark drafter's eager context step ``K3DSparkModel.combine_aux_rows``
  (each rank combines 1/tp of the rows, then one all-gather) equals the stock
  ``context_norm(context_proj(torch.cat(aux rows)))`` per row up to the GEMM's
  accumulation order (cuBLAS may pick another split-K tactic for 1/tp of the
  rows; at the deployed shapes it is bit for bit, see the lane bench), for aux states
  given as column views of one packed SP gather (read in place) and as separate
  tensors, for row ranges below the shard threshold (replicated), at it, with a
  short last shard, and starting past row 0 (a FULL graph's staged rows);
- ``sp_embed`` (vocab-shard lookup + reduce-scatter) equals the stock lookup's
  all-reduce + ``sp_shard`` bit for bit (each token has one owning shard), with a
  row count that needs SP padding, and keeps the all-reduce below its threshold.
"""

import time
import types

import pytest
import torch
import torch.multiprocessing as mp

from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

H, L, VOCAB = 1024, 5, 4099


def _worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    world = int(env["WORLD_SIZE"])
    torch.accelerator.set_device_index(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world, int(env["RANK"]), "env://", local_rank, "nccl"
        )
        initialize_model_parallel(tensor_model_parallel_size=world)
        import torch.nn.functional as F

        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )
        from vllm.models.common.ops.sequence_parallel import sp_shard
        from vllm.models.kimi_k3.nvidia import dspark_mla
        from vllm.models.kimi_k3.nvidia import model as k3_model

        gen = torch.Generator(device="cpu").manual_seed(7)  # same on every rank
        weight = (torch.randn(H, L * H, generator=gen) / (L * H) ** 0.5).bfloat16()
        weight = weight.to(device)
        norm = RMSNorm(H, eps=1e-6).to(device, torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(1 + 0.1 * torch.randn(H, generator=gen).to(device))
        stand_in = types.SimpleNamespace(
            context_proj=lambda x: F.linear(x, weight),
            context_norm=norm,
            combine_hidden_states=lambda x: norm(F.linear(x, weight)),
        )
        threshold = world * dspark_mla._CONTEXT_ROW_SHARD_MIN_ROWS
        rows = 2 * threshold + 3  # short last shard
        packed = torch.randn(rows, (1 + L) * H, generator=gen).bfloat16().to(device)
        layouts = {
            "packed": list(packed.split(H, dim=-1))[1:],
            "separate": [
                torch.randn(rows, H, generator=gen).bfloat16().to(device)
                for _ in range(L)
            ],
        }
        ranges = [(0, rows), (0, threshold), (0, threshold - 1), (5, rows), (0, 17)]
        with torch.inference_mode():
            assert dspark_mla._aux_rows(layouts["packed"], 3, 9).data_ptr() == (
                packed[3, H:].data_ptr()
            )  # read in place
            for name, aux in layouts.items():
                for start, end in ranges:
                    out = torch.full((end - start, H), float("nan"), device=device)
                    out = out.bfloat16()
                    dspark_mla.K3DSparkModel.combine_aux_rows(
                        stand_in, aux, start, end, out
                    )
                    ref = norm(
                        F.linear(torch.cat([h[start:end] for h in aux], -1), weight)
                    )
                    torch.testing.assert_close(
                        out, ref, rtol=1.6e-2, atol=1e-2, msg=str((name, start, end))
                    )

            torch.set_default_dtype(torch.bfloat16)
            with torch.device(device):
                embed = VocabParallelEmbedding(VOCAB, H)
            torch.set_default_dtype(torch.float32)
            with torch.no_grad():
                embed.weight.copy_(torch.randn(embed.weight.shape, generator=gen))
            assert embed.use_fused_embedding
            for n in (
                k3_model._SP_EMBED_RS_MIN_TOKENS + 1,  # SP padding
                k3_model._SP_EMBED_RS_MIN_TOKENS - 1,  # all-reduce + shard
            ):
                ids = torch.randint(0, VOCAB, (n,), generator=gen).to(device)
                ref = sp_shard(embed(ids))
                assert torch.equal(k3_model.sp_embed(embed, ids), ref), n


def _distributed_run(fn, world_size: int) -> None:
    port = str(get_open_port())
    ctx = mp.get_context("spawn")
    processes = []
    for rank in range(world_size):
        env = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": port,
        }
        process = ctx.Process(target=fn, args=(env,))
        processes.append(process)
        process.start()
    deadline = time.monotonic() + 300
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
    assert [process.exitcode for process in processes] == [0] * world_size


def test_prefill_drafter_context_and_sp_embed():
    if torch.accelerator.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    world = 4 if torch.accelerator.device_count() >= 4 else 2
    _distributed_run(_worker, world_size=world)
