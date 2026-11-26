# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The #63 exact DCP top-k merge inside the DSA sparse core (DsaCore.run_mic) over a
real DCP group: the fused indexer's local top-N (IndexerLocalTopk) publishes round 1
into the merge's mailbox, and the core selects this rank's keys itself (with the
exact second round of saturated rows) instead of the select kernel. Against the
published select (rescue folded) + DsaCore.run on the same logits: the same rows and
counts (in the same order) and bitwise the same pushed partials and LSEs, at the
core's row counts of B1 / B2 / B4 (8 / 16 / 32 rows: 2x16s3, 2x16s3, 1x16), on
normal, saturated (one rank; the slot bound), short and tie-heavy rows, eager and
under CUDA-graph replay; afterwards every mailbox slot is armed again and the
second-round state agrees with the select's."""

import time

import pytest
import torch
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.network_utils import get_open_port

from .test_dcp_topk_merge import (
    TOPK,
    _BLOCK,
    _block_table,
    _boost,
    _has_deep_select,
    _make_case,
    _shard,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100)
    or not has_cutedsl()
    or not _has_deep_select(),
    reason="The DCP top-k merge needs SM100, CuTe DSL and DeepSelect",
)

HEADS, QH, D, CAP = 96, 128, 576, 64
# rows -> (splits, heads per CTA, split_min): decode.VARIANT at q8
VARIANTS = {8: (2, 16, 3), 16: (2, 16, 3), 32: (1, 16, 4)}


def _lengths(rows: int, world: int, kind: str) -> list[int]:
    base = [100_000, 40_000, 9_000, 70_123, 2_049, 150_000, 6_000, world * 66_000]
    if kind == "short":
        base = [300, 1_000, 3_000, 9_000, 20_000, 120, 2_100, 4_096]
    return [base[i % len(base)] + 17 * (i // len(base)) for i in range(rows)]


def _publish(box, local, local_end, n, world):
    """IndexerLocalTopk: this rank's B' rows multicast into the round-1 mailbox."""
    import cutlass
    import cutlass.cute as cute

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.indexer import (
        IndexerLocalTopk,
    )
    from vllm.v1.attention.ops.dcp_topk_merge import packed_row_words

    rows, pitch = local.shape
    packed = torch.zeros((rows, packed_row_words(n)), dtype=torch.int32, device="cuda")
    counts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    args = (
        to_cute(local.view(torch.int16).reshape(-1)),
        to_cute(local_end, 4),
        to_cute(packed, 4),
        to_cute(counts, 4),
        to_cute(box.ordinals[:rows], 4),
        cutlass.Int64(box.box_mc),
        cutlass.Int32(box.rank),
        cutlass.Int32(box.max_rows),
        stream(torch.device("cuda")),
    )
    key = (rows, n, pitch, world)
    fn = _COMPILED.get(key)
    if fn is None:
        fn = _COMPILED[key] = cute.compile(IndexerLocalTopk(rows, n, pitch, world), *args)
    fn(*args)
    return packed


_COMPILED: dict = {}


class _Boxes:
    """This rank's pushes: one local DcpBox (partial, LSE) per head owner."""

    def __init__(self, world, hpr):
        self.part = [
            torch.zeros(2, world, CAP, hpr, 512, dtype=torch.bfloat16, device="cuda")
            for _ in range(world)
        ]
        self.lse = [
            torch.zeros(2, world, CAP, hpr, device="cuda") for _ in range(world)
        ]
        self.peers = torch.tensor([t.data_ptr() for t in self.part], device="cuda")
        self.lpeers = torch.tensor([t.data_ptr() for t in self.lse], device="cuda")

    def arm(self):
        for t in self.part:
            t.view(torch.int32).fill_(-0x80000000)
        for t in self.lse:
            t.view(torch.int32).fill_(-1)

    def snap(self):
        return [t.view(torch.int16).clone() for t in self.part] + [
            t.view(torch.int32).clone() for t in self.lse
        ]


def _check(group, ws, n) -> None:
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dsa_core.ops import DsaCore
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.indexer_stage import (  # noqa: E501
        MicInputs,
    )
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import dcp_topk_merge_rows

    world, rank = group.world_size, group.rank_in_group
    box = ws.mailbox
    hpr = HEADS // world
    core = DsaCore(torch.device("cuda"), cap=CAP, heads=HEADS)
    g = torch.Generator(device="cuda").manual_seed(7)
    q = (torch.randn(CAP, QH, D, generator=g, device="cuda") * 0.3).to(
        torch.float8_e4m3fn
    )
    q_box = q.view(torch.uint8)
    scales = (0.07, 0.9)
    boxes = _Boxes(world, hpr)
    cases = [("normal", []), ("sat1", [world - 1]), ("short", []), ("ties", [])]
    num_slots = (TOPK - 1) // n
    if num_slots >= 2 and world > 2:
        cases.append(("sat2", [0, world - 1][:num_slots]))
    for rows, variant in VARIANTS.items():
        splits, heads, split_min = variant
        for seed, (name, sat) in enumerate(cases):
            lengths = _lengths(rows, world, "short" if name == "short" else "")
            logits, ends = _make_case(
                lengths, seed=seed + 300 + rows, levels=4 if name == "ties" else 0
            )
            if sat:  # every other row saturated by the boosted ranks
                logits = _boost(logits, sat, slice(0, rows, 2), world)
            local, local_end = _shard(logits, ends, world, rank)
            req_id = torch.arange(rows, dtype=torch.int32, device="cuda")
            table = _block_table(rows, int(local_end.max()))
            nkv = int(table.max() + 1) * _BLOCK
            # FP8 E4M3 codes of magnitude < 2 (no NaN), random sign
            kv = torch.randint(0, 0x40, (nkv, D), generator=g, device="cuda").to(
                torch.uint8
            )
            kv |= torch.randint(0, 2, (nkv, D), generator=g, device="cuda").to(
                torch.uint8
            ) << 7
            rows_a = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
            valid_a = torch.full((rows,), -7, dtype=torch.int32, device="cuda")
            rows_b = torch.full_like(rows_a, -1)
            valid_b = torch.full_like(valid_a, -7)
            common = dict(
                bmm1_scale=scales[0], bmm2_scale=scales[1], part_peers=boxes.peers,
                lse_peers=boxes.lpeers, box_cap=CAP, hpr=hpr, parity=0, rank=rank,
                rows=rows, splits=splits, split_min=split_min, heads_per_cta=heads,
            )  # fmt: skip

            def arm_a():
                packed = _publish(box, local, local_end, n, world)
                dcp_topk_merge_rows(
                    logits=local, row_ends=local_end, packed=packed, published=True,
                    release=True, local_topk=n, topk=TOPK, req_id=req_id,
                    block_table=table, block_size=_BLOCK, stride_rows=_BLOCK,
                    dcp_rank=rank, rows_out=rows_a, valid_out=valid_a,
                )  # fmt: skip
                core.run(q_box, rows_a, valid_a, kv, **common)

            mic = MicInputs(
                ws=ws, logits=local, row_ends=local_end, req_id=req_id,
                block_table=table, block_size=_BLOCK, stride_rows=_BLOCK,
                rows=rows_b, valid=valid_b, topk=TOPK, local_topk=n,
            )  # fmt: skip

            def arm_b():
                _publish(box, local, local_end, n, world)
                core.run_mic(q_box, kv, mic, **common)

            what = f"{name}/rows {rows}"
            boxes.arm()
            arm_a()
            torch.accelerator.synchronize()
            core.check()
            want = boxes.snap()
            masks = ((ws.sat_word[:rows] >> 16) & 0xFFFF).tolist()
            if sat:
                assert any(masks), f"{what}: no saturated row"
            # eager, both mailbox parities
            for rep in range(2):
                boxes.arm()
                rows_b.fill_(-1)
                valid_b.fill_(-7)
                arm_b()
                torch.accelerator.synchronize()
                core.check()
                assert torch.equal(valid_b, valid_a), f"{what}/{rep}: counts"
                assert torch.equal(rows_b, rows_a), f"{what}/{rep}: rows"
                got = boxes.snap()
                for i, (x, y) in enumerate(zip(got, want)):
                    assert torch.equal(x, y), f"{what}/{rep}: box {i}"
            # CUDA graph: publish + core captured, replayed
            group.barrier()
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                arm_b()
            for rep in range(3):
                boxes.arm()
                rows_b.fill_(-1)
                graph.replay()
                torch.accelerator.synchronize()
                core.check()
                assert torch.equal(rows_b, rows_a), f"{what}/graph {rep}: rows"
                got = boxes.snap()
                for i, (x, y) in enumerate(zip(got, want)):
                    assert torch.equal(x, y), f"{what}/graph {rep}: box {i}"
            del graph
            group.barrier()
            torch.accelerator.synchronize()
            assert bool((box.box == -1).all()), f"{what}: a mailbox slot is armed"
            assert not bool(core.readers.any()), f"{what}: reader counts"
            every = group.all_gather(ws.seq2.unsqueeze(0), dim=0)
            assert all(torch.equal(every[r], every[0]) for r in range(world))
            every = group.all_gather(box.ordinals.unsqueeze(0), dim=0)
            assert all(torch.equal(every[r], every[0]) for r in range(world))


def run_worker(rank: int, world: int, local_rank: int, init_method: str) -> None:
    from types import SimpleNamespace

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        get_dcp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.attention.ops.dcp_topk_merge import (
        dcp_merge_local_topk,
        init_dcp_topk_merge,
    )
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        get_dcp_topk_rescue_workspace,
    )

    torch.accelerator.set_device_index(local_rank)
    config = VllmConfig()
    config.parallel_config.tensor_parallel_size = world
    config.parallel_config.decode_context_parallel_size = world
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method=init_method,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world, decode_context_model_parallel_size=world
        )
        try:
            group = get_dcp_group()
            n = dcp_merge_local_topk(TOPK, world)
            gate_config = SimpleNamespace(
                parallel_config=config.parallel_config,
                kernel_config=SimpleNamespace(
                    sparse_indexer_topk_backend="auto", enable_jit_warmup=False
                ),
                model_config=SimpleNamespace(max_model_len=world << 17),
                kv_transfer_config=None,
                scheduler_config=config.scheduler_config,
                compilation_config=config.compilation_config,
                speculative_config=None,
                num_speculative_tokens=0,
            )
            assert init_dcp_topk_merge(gate_config, TOPK) == n, "merge disabled"
            ws = get_dcp_topk_rescue_workspace(TOPK, n)
            assert ws is not None and ws.mailbox is not None
            assert ws.max_rows >= max(VARIANTS)
            _check(group, ws, n)
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    "world",
    [
        pytest.param(
            w,
            marks=pytest.mark.skipif(
                torch.accelerator.device_count() < w, reason=f"Need {w} GPUs."
            ),
        )
        for w in (4, 8)  # (IndexerLocalTopk: N < 1024)
    ],
)
def test_dsa_core_merge_in_core(world: int):
    init_method = f"tcp://127.0.0.1:{get_open_port()}"
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=run_worker, args=(r, world, r, init_method))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    deadline = time.monotonic() + 900
    for p in procs:
        p.join(timeout=max(0.0, deadline - time.monotonic()))
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join()
    assert [p.exitcode for p in procs] == [0] * world
