# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 SP MXFP8 all-gather over symmetric memory
(``ops/sp_all_gather.py`` ``kimi_sp_all_gather_mxfp8``), on GPUs: each rank
quantizes its BF16 shard with the in_proj's quantizer (FlashInfer
``mxfp8_quantize``, F8_128x4 scales) and pushes MXFP8; the gathered E4M3 rows,
the scale bytes of every real row and an MXFP8 GEMM over them are bitwise
``mxfp8_quantize(all_gather(bf16))`` (the path it replaces) at 128 / 512 /
2048 rows per rank, with and without SP padding rows, between BF16 gathers of
the same forward (shared slots) and in a replayed CUDA graph. The CPU layout
and dispatch tests are ``tests/models/kimi_k3/test_sp_all_gather_mxfp8.py``.
"""

import queue

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

HIDDEN = 7168
ROWS = (128, 512, 2048)
N_OUT = 512  # GEMM check width


def _shard(step, rank, rows, device):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1000 * step + rank)
    x = torch.randn(rows, HIDDEN, generator=gen)
    # Per-row magnitudes over many binades (distinct E8M0 scales per row).
    x *= torch.exp2(torch.randint(-20, 20, (rows, 1), generator=gen).float())
    return x.bfloat16().to(device)


def _row_scales(sf, rows):
    """F8_128x4 scale bytes -> row-major [rows, HIDDEN / 32]."""
    m_tiles = -(-rows // 128)
    tiles = sf.view(torch.uint8)[: m_tiles * 128 * HIDDEN // 32]
    tiles = tiles.reshape(m_tiles, HIDDEN // 128, 32, 4, 4)
    return tiles.transpose(1, 3).reshape(m_tiles * 128, HIDDEN // 32)[:rows]


def _worker(local_rank: int, world_size: int, port: int, q: mp.Queue):
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))
    with set_current_vllm_config(config):
        device = torch.device(f"cuda:{local_rank}")
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
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        from vllm.model_executor.layers.fusion.quant_activation import (
            QuantizedActivation,
        )
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kMxfp8Dynamic,
        )
        from vllm.models.kimi_k3.nvidia.ops import sp_all_gather as sp_ag

        max_rows = max(ROWS)
        if not sp_ag.init_sp_all_gather_buffer(max_rows, HIDDEN, torch.bfloat16):
            if local_rank == 0:
                q.put("no symmetric memory over the TP group")
            return
        group = get_tp_group().device_group
        ca_comm = get_tp_group().device_communicator.ca_comm

        def custom(rows):
            return ca_comm is not None and ca_comm.should_custom_all_gather(
                torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device="meta")
            )

        def all_gather(x):
            out = x.new_empty((world_size * x.shape[0], *x.shape[1:]))
            dist.all_gather_into_tensor(out, x, group=group)
            return out

        def quantize(x):
            q8, sf = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
            return QuantizedActivation(
                data=q8,
                scale=sf,
                orig_dtype=x.dtype,
                orig_shape=x.shape,
                quant_key=kMxfp8Dynamic,
            )

        gen = torch.Generator(device="cpu").manual_seed(7)
        w = torch.randn(N_OUT, HIDDEN, generator=gen).bfloat16().to(device)
        w_q, w_sf = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=True)

        def gemm(data, sf):
            return torch._scaled_mm(
                data,
                w_q.t(),
                scale_a=sf.view(torch.float8_e8m0fnu),
                scale_b=w_sf.view(torch.float8_e8m0fnu),
                out_dtype=torch.bfloat16,
            )

        def check(x, out, num_tokens):
            """``out`` (gathered MXFP8 of ``num_tokens`` rows) against the
            BF16 gather + quantize it replaces."""
            ref_q, ref_sf = mxfp8_e4m3_quantize(
                all_gather(x)[:num_tokens], is_sf_swizzled_layout=True
            )
            assert out.data.shape == (num_tokens, HIDDEN)
            assert out.scale.numel() == ref_sf.numel(), (
                out.scale.numel(),
                ref_sf.numel(),
            )
            assert torch.equal(out.data.view(torch.uint8), ref_q.view(torch.uint8))
            assert torch.equal(
                _row_scales(out.scale, num_tokens), _row_scales(ref_sf, num_tokens)
            )
            if num_tokens % 128 == 0:  # no padding rows: every byte
                assert torch.equal(
                    out.scale.view(torch.uint8), ref_sf.view(torch.uint8)
                )
            assert torch.equal(gemm(out.data, out.scale), gemm(ref_q, ref_sf))

        # Bitwise at every size, with and without SP padding rows, and the
        # dispatch: MXFP8 exactly when the BF16 shard would take the push.
        step = 0
        for rows in ROWS:
            assert sp_ag.mxfp8_gather_rows_ok(rows, HIDDEN) == (not custom(rows))
            for num_tokens in (world_size * rows, world_size * rows - 5):
                step += 1
                x = _shard(step, local_rank, rows, device)
                sp_ag.begin_forward(1)
                out = sp_ag.kimi_sp_all_gather_mxfp8(quantize(x), num_tokens)
                check(x, out, num_tokens)

        # One forward mixing BF16 and MXFP8 gathers on the two slots: each
        # output intact until the gather two later.
        sizes = (512, 2048, 512, 2048)
        sp_ag.begin_forward(len(sizes))
        prev = None
        for i, rows in enumerate(sizes):
            x = _shard(100 + i, local_rank, rows, device)
            if i % 2:
                out = sp_ag.kimi_sp_all_gather_mxfp8(quantize(x), world_size * rows)
                check(x, out, world_size * rows)
                live = out.data.view(torch.uint8)
            else:
                live = sp_ag.kimi_sp_all_gather(x)
                assert torch.equal(live, all_gather(x))
            if prev is not None:
                assert torch.equal(prev[1], prev[0])
            prev = (live.clone(), live)

        # A captured forward of an MXFP8 and a BF16 gather, replayed.
        rows = 512
        static = _shard(200, local_rank, rows, device)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            qa_static = quantize(static)  # warm the quantizer off-capture
        torch.accelerator.synchronize()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            sp_ag.begin_forward(2)
            qa_static = quantize(static)
            g0 = sp_ag.kimi_sp_all_gather_mxfp8(qa_static, world_size * rows - 3)
            out0 = gemm(g0.data, g0.scale)
            g1 = sp_ag.kimi_sp_all_gather(static)
            sums = g1.float().sum(1)
        for rep in range(3):
            static.copy_(_shard(300 + rep, local_rank, rows, device))
            graph.replay()
            torch.accelerator.synchronize()
            ref_q, ref_sf = mxfp8_e4m3_quantize(
                all_gather(static)[: world_size * rows - 3], is_sf_swizzled_layout=True
            )
            assert torch.equal(out0, gemm(ref_q, ref_sf))
            assert torch.equal(sums, all_gather(static).float().sum(1))
        torch.accelerator.synchronize()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("world_size", [2, 4])
def test_kimi_k3_sp_all_gather_mxfp8(world_size):
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    if not current_platform.has_device_capability(100):
        pytest.skip("MXFP8 GEMM needs SM100+")
    q = mp.get_context("spawn").Queue()
    mp.spawn(_worker, args=(world_size, get_open_port(), q), nprocs=world_size)
    try:
        val = q.get(timeout=1)
    except queue.Empty:
        val = None
    finally:
        cleanup_dist_env_and_memory()
    if val is not None:
        pytest.skip(val)
