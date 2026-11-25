# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Overlapped SP all-gather of the KDA in_proj input (``ops/sp_all_gather.py``
``kimi_sp_all_gather_mxfp8_overlapped``) on GPUs: the fused sm_107 Q|K|V +
conv GEMM consumes peer shards while the pushers land them on its idle SMs.

Bitwise against the stock path (stock MXFP8 gather, then the same GEMM): the
q|k|v planes, the conv state rows and the gathered bytes, at 1024 / 2048 rows
per rank, over several forwards with BF16 gathers in between (shared slots,
arrival protocol), with one rank late by 30 us, and through ``complete`` (a
consumer that is not the fused GEMM).
"""

import queue

import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

HIDDEN = 7168
P = 1536  # TP8 q|k|v plane (N = 4608: the production tactic)
ROWS = (1024, 2048)
NSEQ = 4


def _shard(step, rank, rows, device):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(7919 * step + rank)
    x = torch.randn(rows, HIDDEN, generator=gen)
    x *= torch.exp2(torch.randint(-8, 8, (rows, 1), generator=gen).float())
    return x.bfloat16().to(device)


def _sleep_us(us):
    if us:
        torch.cuda._sleep(int(us * 2000))  # ~2 GHz cycles


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
        from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_conv_in_proj import (
            overlap_idle_sms,
            tactic_for,
        )
        from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
            KdaConvGemm,
            ag_tactic,
            cluster_rows,
        )
        from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

        try:
            assert sp_ag.init_sp_all_gather_buffer(max(ROWS), HIDDEN, torch.bfloat16)
            gen = torch.Generator(device="cpu").manual_seed(11)
            w = (torch.randn(3 * P, HIDDEN, generator=gen) * 0.02).bfloat16().to(device)
            wq, ws = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=True)
            conv_w = (torch.randn(3 * P, 4, generator=gen) * 0.5).to(device)
            gemm = KdaConvGemm(wq, ws.view(torch.uint8).reshape(-1), conv_w, P,
                               tactic=tactic_for(3 * P))  # fmt: skip
            idle = overlap_idle_sms(device, gemm.tactic)
            cl = cluster_rows(ag_tactic(gemm.tactic))
            assert idle > 0 and cl == 1024, (idle, cl)
            sp_ag.enable_overlap(idle, cl)
            sp_ag.begin_forward(10_000)

            def qa_of(x):
                data, sf = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
                return QuantizedActivation(
                    data=data, scale=sf, quant_key=kMxfp8Dynamic,
                    orig_dtype=torch.bfloat16, orig_shape=x.shape,
                )  # fmt: skip

            def conv_inputs(m):
                qsl = torch.tensor(
                    [i * m // NSEQ for i in range(NSEQ + 1)], dtype=torch.int32, device=device
                )
                lines = torch.arange(NSEQ, dtype=torch.int32, device=device)
                init = torch.tensor([i % 2 == 1 for i in range(NSEQ)], device=device)
                st = (torch.randn(NSEQ, 3 * P, 3, generator=gen) * 0.1).bfloat16().to(device)
                return qsl, lines, init, st

            step = 0
            for it in range(3):
                for rows in ROWS:
                    step += 1
                    m = rows * world_size
                    late = 30 if local_rank == (step % world_size) else 0
                    x = _shard(step, local_rank, rows, device)
                    qa = qa_of(x)
                    qsl, lines, init, st = conv_inputs(m)
                    # stock: gather, then the GEMM
                    ref_qa = sp_ag.kimi_sp_all_gather_mxfp8(qa, m)
                    ref_bytes = ref_qa.data.view(torch.uint8).clone()
                    ref_sf = ref_qa.scale.view(torch.uint8).reshape(-1)[: m * HIDDEN // 32].clone()
                    st_ref = st.clone()
                    out_ref, _ = gemm(
                        ref_qa.data, ref_qa.scale.view(torch.uint8).reshape(-1), st_ref,
                        qsl, lines, init, null_block_id=NULL_BLOCK_ID,
                    )  # fmt: skip
                    # a BF16 gather between (other slot user, arrival protocol)
                    _ = sp_ag.kimi_sp_all_gather(x)
                    # overlapped
                    _sleep_us(late)
                    oqa = sp_ag.kimi_sp_all_gather_mxfp8_overlapped(qa, m, cl)
                    assert oqa is not None
                    pending = sp_ag.take_pending(oqa)
                    assert pending is not None
                    st_ag = st.clone()
                    out, _ = gemm(
                        oqa.data, oqa.scale.view(torch.uint8).reshape(-1), st_ag,
                        qsl, lines, init, null_block_id=NULL_BLOCK_ID, ag=pending,
                    )  # fmt: skip
                    torch.accelerator.synchronize()
                    assert torch.equal(oqa.data.view(torch.uint8), ref_bytes), (it, rows)
                    sf = oqa.scale.view(torch.uint8).reshape(-1)[: m * HIDDEN // 32]
                    assert torch.equal(sf, ref_sf), (it, rows)
                    assert torch.equal(out, out_ref), (it, rows, (out != out_ref).sum())
                    assert torch.equal(st_ag, st_ref), (it, rows)
                    # complete(): a consumer that is not the fused GEMM
                    oqa2 = sp_ag.kimi_sp_all_gather_mxfp8_overlapped(qa, m, cl)
                    sp_ag.complete_pending()
                    torch.accelerator.synchronize()
                    assert torch.equal(oqa2.data.view(torch.uint8), ref_bytes), (it, rows)
            q.put(None)
        except Exception as e:  # noqa: BLE001
            import traceback

            q.put(f"rank {local_rank}: {e!r}\n{traceback.format_exc()}")
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)) or torch.cuda.device_count() < 4,
    reason="needs 4 SM107 GPUs",
)
def test_sp_ag_overlap_bitwise():
    world_size = 4
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = get_open_port()
    procs = [ctx.Process(target=_worker, args=(r, world_size, port, q)) for r in range(world_size)]
    for p in procs:
        p.start()
    errors = []
    for _ in procs:
        try:
            r = q.get(timeout=900)
        except queue.Empty:
            r = "timeout"
        if r is not None:
            errors.append(r)
    for p in procs:
        p.join(timeout=60)
    assert not errors, "\n".join(errors)
