# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP pair exchanger: its Triton kernels against the torch reference (A6), the
real cache writers (A7) and two ranks over NCCL (A8), which need GPUs; and, on
CPU, its chunking, buffers, signature and partner checks, with the exchange
driven through the torch reference.
"""

import contextlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import CacheConfig
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import dcp_pair_exchange
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.dcp_pair_exchange import (
    DcpPairExchanger,
    NixlPairModeError,
    PairRegion,
    PairUnit,
    build_dcp_pair_exchanger,
    build_pair_row_table,
    pair_layout,
    pair_units,
    reference_pack,
    reference_pair_exchange,
    reference_unpack,
)
from vllm.triton_utils import HAS_TRITON
from vllm.utils import torch_utils
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import AttentionGroup, allocate_kv_cache

from .test_nixl_dcp_pair_exchange import (
    D_DCP,
    INDEX_KERNEL_ROWS,
    INDEX_VALUE_BYTES,
    P_DCP,
    _block_tables,
    _k31_dsa_specs,
    _pull_as_today,
)

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON), reason="needs CUDA and Triton"
)


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    """Many small tensor ops on a shared host: keep them off OpenMP."""
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(num_threads)


@pytest.fixture
def cpu_h2d(monkeypatch):
    """Upload unit tables to a CPU device without pinned staging, which a
    CPU-only torch cannot allocate."""
    monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)


class _Rank:
    """A rank's packed KV cache on ``device`` as the real planner lays it out
    and the worker allocates it, with the NIXL regions the worker registers
    and their row table."""

    def __init__(self, specs: dict, num_blocks: int, device="cpu"):
        self.specs = specs
        self.block_size = next(iter(specs.values())).block_size
        cache_config = CacheConfig(block_size=self.block_size)
        cache_config.kv_cache_layout = "BLHNC"
        cache_config.num_gpu_blocks_override = num_blocks
        config = SimpleNamespace(
            cache_config=cache_config,
            attention_config=SimpleNamespace(hisparse_config=None),
        )
        group = KVCacheGroupSpec(
            list(specs), UniformTypeKVCacheSpecs(self.block_size, specs)
        )
        kv_cache_config = get_kv_cache_config_from_groups(config, [group], 1)
        self.views = allocate_kv_cache(
            kv_cache_config, torch.device(device), KVCacheLayout.BLHNC
        )
        storage = next(iter(self.views.values())).untyped_storage()
        self.buf = torch.empty(0, dtype=torch.uint8, device=device).set_(storage)
        self.regions = [
            PairRegion(
                name,
                specs[name],
                view.data_ptr(),
                specs[name].page_size_bytes,
                view.stride(0) * view.element_size(),
            )
            for name, view in self.views.items()
        ]
        self.table = build_pair_row_table(
            self.regions,
            index_kernel_rows=INDEX_KERNEL_ROWS,
            index_value_bytes=INDEX_VALUE_BYTES,
        )

    def fill_random(self, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        data = torch.randint(0, 256, self.buf.shape, dtype=torch.uint8, generator=gen)
        self.buf.copy_(data)


def _rebased(table, old: torch.Tensor, new: torch.Tensor):
    """``table`` for a copy ``new`` of the buffer ``old`` it addresses."""
    return replace(table, base_addr=table.base_addr - old.data_ptr() + new.data_ptr())


class _Recording:
    """Stands in for NCCL in a first pass: keeps every round's ``send``."""

    def __init__(self):
        self.sends: list[torch.Tensor] = []

    def batch_isend_irecv(self, ops) -> None:
        self.sends.append(ops[0].tensor.clone())


class _Replaying:
    """The second pass: every round receives the partner's recorded ``send``."""

    def __init__(self, sends: list[torch.Tensor]):
        self.sends = list(sends)

    def batch_isend_irecv(self, ops) -> None:
        ops[1].tensor.copy_(self.sends.pop(0))


@contextlib.contextmanager
def _no_host_sync(device: torch.device):
    if device.type != "cuda":
        yield
        return
    mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(mode)


def _exchange_pair(make, bufs, tables, layouts, units, chunk_units):
    """Both partners' exchanges in lockstep, as over NCCL. A first pass on
    copies records each rank's sends: a round's pack reads only blocks that no
    earlier round rewrote. The second pass delivers them to the partner.

    Returns:
        Each rank's sends, round by round.

    """
    sends = []
    for buf, table, layout in zip(bufs, tables, layouts):
        copy = buf.clone()
        comm = _Recording()
        make(copy, _rebased(table, buf, copy), layout, comm, chunk_units).exchange(
            units
        )
        sends.append(comm.sends)
    exchangers = [
        make(buf, table, layout, _Replaying(sends[1 - r]), chunk_units)
        for r, (buf, table, layout) in enumerate(zip(bufs, tables, layouts))
    ]
    with _no_host_sync(bufs[0].device):
        for exchanger in exchangers:
            exchanger.exchange(units)
    return sends


def _triton_exchanger(buf, table, layout, comm, chunk_units):
    return DcpPairExchanger(
        table, layout, buf.device, chunk_units=chunk_units, communicator=comm
    )


class _TorchExchanger(DcpPairExchanger):
    """The exchanger with the torch reference in place of its kernels."""

    def __init__(self, buf, table, layout, comm, chunk_units):
        super().__init__(
            table, layout, buf.device, chunk_units=chunk_units, communicator=comm
        )
        self.buf = buf

    def _pack(self, units, rows, classes):
        block_ids = [u.block_id for u in units]
        keep, send = reference_pack(self.buf, self.table, self.layout, block_ids)
        self.keep[: len(units)] = keep
        self.send[: len(units)] = send

    def _unpack(self, units, rows, classes):
        n = len(units)
        keep, recv = self.keep[:n], self.recv[:n]
        reference_unpack(self.buf, self.table, self.layout, units, keep, recv)


# Two requests in one step: each ends on a unit without partner, and the blocks
# are out of order.
_UNITS = [
    PairUnit(0, 5, True),
    PairUnit(0, 2, False),
    PairUnit(0, 0, True),
    PairUnit(0, 6, True),
    PairUnit(0, 3, False),
]
_PAIR = [pair_layout(c, 2, 1, 1024, 1) for c in (0, 1)]


# ----------------------------------------------------------------------------
# CPU: buffers, signature, chunking and partner checks
# ----------------------------------------------------------------------------


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_draft", [0, 5])
def test_buffers_and_kernel_classes_on_k31_geometry(num_draft: int):
    """Design §3.5: H is 8,699,904 B plus 294,912 B per drafter layer, so the
    three [4, H] buffers take 104.4 MB without drafter pages and 122.1 MB with
    five. Each chunk's one launch moves every row width: 16-byte vectors,
    4-byte for scales."""
    rank = _Rank(_k31_dsa_specs(1024, 24, num_draft), num_blocks=1)
    exchanger = DcpPairExchanger(
        rank.table, pair_layout(3, 16, 8, 1024, 1), "cpu", communicator=_Recording()
    )
    half_bytes = 8_699_904 + num_draft * 294_912
    for buf in (exchanger.keep, exchanger.send, exchanger.recv):
        assert buf.shape == (4, half_bytes) and buf.dtype == torch.uint8
    assert round(3 * exchanger.keep.numel() / 1e6, 1) == (122.1 if num_draft else 104.4)
    classes = {
        (c.width, c.rows_per_subpage, c.vector_bytes, c.num_arrays)
        for c in exchanger._classes
    }
    assert classes == {
        (576, 1024, 16, 24 + num_draft),
        (128, INDEX_KERNEL_ROWS, 16, 24),
        (4, INDEX_KERNEL_ROWS, 4, 24),
    }


@pytest.mark.cpu_test
def test_signature_is_geometry_and_chunking_not_addresses():
    specs = _k31_dsa_specs(64, 2, 1)
    layouts = [pair_layout(c, 2, 1, 64, 1) for c in (0, 1)]

    def signature(table, layout=layouts[0], chunk_units=4):
        return DcpPairExchanger(
            table, layout, "cpu", chunk_units=chunk_units, communicator=_Recording()
        ).signature

    table = _Rank(specs, 2).table
    # The partner: other buffer addresses, pool size and parity.
    assert signature(_Rank(specs, 3).table, layouts[1]) == signature(table)
    assert signature(table, chunk_units=2) != signature(table)
    assert signature(_Rank(_k31_dsa_specs(64, 2, 0), 2).table) != signature(table)


@pytest.mark.cpu_test
def test_exchange_rounds_follow_metadata_order(cpu_h2d):
    """Chunks of K units in metadata order, each packed, swapped with the
    partner through the first n rows of send and recv, then unpacked."""
    log: list[tuple] = []

    class Logged(DcpPairExchanger):
        def _pack(self, units, rows, classes):
            log.append(("pack", [u.block_id for u in units], rows.tolist()))

        def _unpack(self, units, rows, classes):
            log.append(("unpack", [u.block_id for u in units], rows.tolist()))

    class Comm:
        def batch_isend_irecv(self, ops):
            log.append(tuple((op.op, op.tensor, op.group_peer) for op in ops))

    table = _Rank(_k31_dsa_specs(64, 2, 1), 2).table
    layout = pair_layout(3, 16, 8, 64, 1)
    exchanger = Logged(table, layout, "cpu", chunk_units=2, communicator=Comm())
    exchanger.exchange(_UNITS)

    expected = []
    for start in range(0, len(_UNITS), 2):
        chunk = _UNITS[start : start + 2]
        ids = [u.block_id for u in chunk]
        rows = [[u.block_id, int(u.has_partner)] for u in chunk]
        expected += [("pack", ids, rows), "swap", ("unpack", ids, rows)]
    assert [e if e[0] in ("pack", "unpack") else "swap" for e in log] == expected
    for n, swap in zip((2, 2, 1), log[1::3]):
        (send_op, send, send_peer), (recv_op, recv, recv_peer) = swap
        assert send_op is torch.distributed.isend
        assert recv_op is torch.distributed.irecv
        assert send_peer == recv_peer == 11
        assert send.data_ptr() == exchanger.send.data_ptr() and send.shape[0] == n
        assert recv.data_ptr() == exchanger.recv.data_ptr() and recv.shape[0] == n

    log.clear()
    exchanger.exchange([])
    assert not log


@pytest.mark.cpu_test
@pytest.mark.parametrize("chunk_units", [1, 2, 4, 8])
def test_chunked_exchange_matches_one_shot_reference(chunk_units: int, cpu_h2d):
    """The exchange driven through the torch reference and a lockstep fake
    NCCL equals the one-shot reference exchange, for any chunking."""
    specs = _k31_dsa_specs(64, 2, 1)
    ranks = [_Rank(specs, 7) for _ in range(2)]
    for seed, rank in enumerate(ranks):
        rank.fill_random(seed)
    expected = [rank.buf.clone() for rank in ranks]
    reference_pair_exchange(
        expected,
        [{0: _rebased(r.table, r.buf, e)} for r, e in zip(ranks, expected)],
        _UNITS,
    )
    layouts = [pair_layout(c, 2, 1, 64, 1) for c in (0, 1)]
    _exchange_pair(
        _TorchExchanger,
        [r.buf for r in ranks],
        [r.table for r in ranks],
        layouts,
        _UNITS,
        chunk_units,
    )
    for rank, want in zip(ranks, expected):
        assert torch.equal(rank.buf, want)


def _fake_dcp_group(rank=0, world_size=2, disabled=False, pynccl=True, comm=True):
    pynccl_comm = SimpleNamespace(disabled=disabled) if pynccl else None
    device_comm = SimpleNamespace(pynccl_comm=pynccl_comm) if comm else None
    return SimpleNamespace(
        device_communicator=device_comm, rank_in_group=rank, world_size=world_size
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "group",
    [
        _fake_dcp_group(comm=False),
        _fake_dcp_group(pynccl=False),
        _fake_dcp_group(disabled=True),
        _fake_dcp_group(rank=1),
        _fake_dcp_group(world_size=4),
    ],
    ids=["no_communicator", "no_pynccl", "pynccl_disabled", "rank", "world_size"],
)
def test_default_communicator_must_be_enabled_pynccl(group, monkeypatch):
    """Raw PyNccl returns silently when disabled: the exchanger refuses to
    build on it, or on a DCP group whose ranks are not the layout's."""
    table = _Rank(_k31_dsa_specs(64, 2, 1), 2).table
    layout = pair_layout(0, 2, 1, 64, 1)
    monkeypatch.setattr(dcp_pair_exchange, "get_dcp_group", lambda: group)
    with pytest.raises(NixlPairModeError):
        DcpPairExchanger(table, layout, "cpu")

    monkeypatch.setattr(dcp_pair_exchange, "get_dcp_group", _fake_dcp_group)
    exchanger = DcpPairExchanger(table, layout, "cpu")
    assert exchanger._communicator is not None


class _StampEcho:
    """A partner that answers a stamp with its mirror image, edited."""

    def __init__(self, **edits):
        self.edits = edits

    def batch_isend_irecv(self, ops) -> None:
        ours = dcp_pair_exchange._Stamp(*ops[0].tensor.tolist())
        theirs = ours._replace(
            dcp_rank=ours.partner, partner=ours.dcp_rank, parity=1 - ours.parity
        )
        ops[1].tensor.copy_(torch.tensor(theirs._replace(**self.edits)))


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "edits,problem",
    [
        (dict(chunk_units=3), "chunking differ"),
        (dict(signature=1), "chunking differ"),
        (dict(partner=9), "not each other's partner"),
        (dict(parity=0), "same half"),
        (dict(magic=0), "does not come from"),
    ],
)
def test_warm_up_stops_on_a_partner_that_does_not_pair(edits, problem, monkeypatch):
    """The stamp check runs before any geometry-sized message, names both
    sides, and is symmetric: the partner raises too instead of waiting."""
    monkeypatch.setattr(
        torch.accelerator,
        "get_memory_info",
        lambda device=None: (1, 1),
        raising=False,  # CPU-only torch builds lack it
    )
    table = _Rank(_k31_dsa_specs(64, 2, 1), 2).table
    exchanger = DcpPairExchanger(
        table,
        pair_layout(0, 16, 8, 64, 1),
        "cpu",
        communicator=_StampEcho(**edits),
    )
    with pytest.raises(
        NixlPairModeError, match=rf"DCP rank 0 .*NCCL peer, DCP rank 8 .*{problem}"
    ):
        exchanger.warm_up()

    ours = exchanger._stamp()
    theirs = ours._replace(dcp_rank=8, partner=0, parity=1)
    assert dcp_pair_exchange._pair_problem(ours, theirs) is None
    theirs = theirs._replace(**edits)
    assert dcp_pair_exchange._pair_problem(ours, theirs) == (
        dcp_pair_exchange._pair_problem(theirs, ours)
    )


@pytest.mark.cpu_test
def test_warm_up_scratch_check_wants_the_partner_rows_and_nothing_else():
    """The warm-up's check of its exchanged scratch block accepts the torch
    reference's outcome on both partners, and rejects a round that delivered
    nothing (a silent no-op NCCL) or a single stray byte anywhere."""
    table = _Rank(_k31_dsa_specs(64, 2, 1), 1).table
    fill = [dcp_pair_exchange._fill_byte(c) for c in (0, 1)]
    blocks = [
        torch.full((table.block_stride,), fill[c], dtype=torch.uint8) for c in (0, 1)
    ]
    tables = [replace(table, base_addr=block.data_ptr()) for block in blocks]
    reference_pair_exchange(blocks, [{0: t} for t in tables], [PairUnit(0, 0, True)])

    for c in (0, 1):
        layout = pair_layout(c, 2, 1, 64, 1)
        exchanger = DcpPairExchanger(
            tables[c], layout, "cpu", communicator=_Recording()
        )
        good = blocks[c].numpy()
        exchanger._check_scratch(good)
        with pytest.raises(NixlPairModeError, match="wrong bytes"):
            exchanger._check_scratch(np.full_like(good, fill[c]))
        for index in (
            np.flatnonzero(good == fill[c])[0],
            np.flatnonzero(good == fill[1 - c])[-1],
            good.size - 1,
        ):
            stray = good.copy()
            stray[index] ^= 0xFF
            with pytest.raises(
                NixlPairModeError, match=f"with 1 wrong bytes.* {index}:"
            ):
                exchanger._check_scratch(stray)


@pytest.mark.cpu_test
def test_factory_builds_without_warm_up_for_the_cuda_indexer_layout(monkeypatch):
    rank = _Rank(_k31_dsa_specs(128, 2, 1), 2)
    layout = pair_layout(0, 2, 1, 128, 1)
    monkeypatch.setattr(dcp_pair_exchange, "get_dcp_group", _fake_dcp_group)

    def build(**kwargs):
        args = dict(
            index_kernel_rows=INDEX_KERNEL_ROWS,
            index_value_bytes=INDEX_VALUE_BYTES,
            layout=layout,
            device=torch.device("cpu"),
        )
        return build_dcp_pair_exchanger(rank.regions, **{**args, **kwargs})

    exchanger = build(chunk_units=3)
    assert exchanger.table == rank.table and exchanger.chunk_units == 3
    # Kernel-page rows come from the argument, not a constant.
    rows = build(index_kernel_rows=32).table.arrays
    assert {a.rows_per_subpage for a in rows if a.kind.value != "row"} == {32}
    # Values not in 16-byte vectors, or scales other than one fp32 per row.
    for value_bytes in (120, 112):
        with pytest.raises(NixlPairModeError, match="indexer.k_cache"):
            build(index_value_bytes=value_bytes)
    monkeypatch.setattr(dcp_pair_exchange.current_platform, "is_rocm", lambda: True)
    with pytest.raises(NixlPairModeError, match="CUDA"):
        build()


def _token_table(base: int, width: int, stride: int):
    spec = MLAAttentionSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=width,
        dtype=torch.uint8,
        cache_dtype_str="fp8",
    )
    return build_pair_row_table([PairRegion("L0", spec, base, 8 * width, stride)])


@pytest.mark.cpu_test
def test_copy_width_follows_alignment():
    """Rows move in the widest copy every address, stride and width allows;
    rows that are not 4-byte aligned are refused, naming the layer."""
    comm = _Recording()
    layout = pair_layout(0, 2, 1, 8, 1)
    for base, width, stride, vector_bytes in (
        (1 << 20, 32, 512, 16),
        (1 << 20, 24, 512, 8),
        ((1 << 20) + 4, 32, 512, 4),
    ):
        table = _token_table(base, width, stride)
        exchanger = DcpPairExchanger(table, layout, "cpu", communicator=comm)
        assert [c.vector_bytes for c in exchanger._classes] == [vector_bytes]
    with pytest.raises(NixlPairModeError, match="L0"):
        DcpPairExchanger(
            _token_table((1 << 20) + 2, 32, 512), layout, "cpu", communicator=comm
        )


# ----------------------------------------------------------------------------
# A6: Triton kernels against the torch reference
# ----------------------------------------------------------------------------


def _ds_mla_specs(block_size: int, num_dsa: int, num_draft: int) -> dict:
    """K3.1's packed DSA group with fp8_ds_mla latent rows: 512 fp8 values,
    four fp32 scales and 64 bf16 rope values, 656 bytes."""
    specs = _k31_dsa_specs(block_size, num_dsa, num_draft)
    return {
        name: spec
        if spec.cache_role == SparseCacheRole.INDEXER
        else replace(spec, cache_dtype_str="fp8_ds_mla", state_content_bytes=656)
        for name, spec in specs.items()
    }


@requires_gpu
@pytest.mark.parametrize("far", [False, True], ids=["near", "far"])
@pytest.mark.parametrize("latent", ["fp8", "fp8_ds_mla"])
def test_triton_exchange_matches_torch_reference(latent: str, far: bool):
    """A6 on the real K3.1 geometry: every byte of both partners' caches,
    padding included, equals the reference exchange, and each round's send
    equals reference_pack's. ``far`` moves the units to blocks whose offset
    b * S passes 2**31 by shifting the table base down, so the int64 address
    math runs without 2 GiB of cache."""
    device = torch.device("cuda")
    specs = (
        _k31_dsa_specs(1024, 24, 5) if latent == "fp8" else _ds_mla_specs(1024, 4, 1)
    )
    ranks = [_Rank(specs, 7, device) for _ in range(2)]
    for seed, rank in enumerate(ranks):
        rank.fill_random(seed)
    initial = [rank.buf.clone() for rank in ranks]
    expected = [rank.buf.clone() for rank in ranks]
    reference_pair_exchange(
        expected,
        [{0: _rebased(r.table, r.buf, e)} for r, e in zip(ranks, expected)],
        _UNITS,
    )

    stride = ranks[0].table.block_stride
    shift = 2**31 // stride + 3 if far else 0
    assert not far or shift * stride > 2**31
    tables = [
        replace(r.table, base_addr=r.table.base_addr - shift * stride) for r in ranks
    ]
    units = [replace(u, block_id=u.block_id + shift) for u in _UNITS]
    sends = _exchange_pair(
        _triton_exchanger, [r.buf for r in ranks], tables, _PAIR, units, 2
    )
    torch.accelerator.synchronize()

    for r, rank in enumerate(ranks):
        assert torch.equal(rank.buf, expected[r]), f"rank {r}"
        table = _rebased(rank.table, rank.buf, initial[r])
        for start, send in zip(range(0, len(_UNITS), 2), sends[r]):
            ids = [u.block_id for u in _UNITS[start : start + 2]]
            _, want = reference_pack(initial[r], table, _PAIR[r], ids)
            assert torch.equal(send, want), f"rank {r} round {start // 2}"


@requires_gpu
@pytest.mark.parametrize(
    "width,lead,vector_bytes", [(576, 0, 16), (24, 0, 8), (24, 4, 4)]
)
def test_triton_exchange_with_tile_tails_and_narrow_copies(
    width: int, lead: int, vector_bytes: int
):
    """A6 where 1024-row blocks never go: 8-row blocks leave every tile a
    masked tail, and their rows move in 16-, 8- or 4-byte copies."""
    device = torch.device("cuda")
    stride = 8 * width + 16
    bufs = [
        torch.randint(
            0,
            256,
            (lead + 7 * stride,),
            dtype=torch.uint8,
            generator=torch.Generator().manual_seed(seed),
        ).to(device)
        for seed in range(2)
    ]
    tables = [_token_table(buf.data_ptr() + lead, width, stride) for buf in bufs]
    layouts = [pair_layout(c, 2, 1, 8, 1) for c in (0, 1)]
    expected = [buf.clone() for buf in bufs]
    reference_pair_exchange(
        expected,
        [{0: _rebased(t, b, e)} for t, b, e in zip(tables, bufs, expected)],
        _UNITS,
    )

    exchanger = _triton_exchanger(bufs[0], tables[0], layouts[0], _Recording(), 2)
    assert [c.vector_bytes for c in exchanger._classes] == [vector_bytes]
    _exchange_pair(_triton_exchanger, bufs, tables, layouts, _UNITS, 2)
    torch.accelerator.synchronize()
    for r in range(2):
        assert torch.equal(bufs[r], expected[r]), f"rank {r}"


# ----------------------------------------------------------------------------
# A7: real cache writers
# ----------------------------------------------------------------------------


def _cache_ops():
    from vllm import _custom_ops as ops

    cache_ops = getattr(torch.ops, "_C_cache_ops", None)
    if not all(
        hasattr(cache_ops, op)
        for op in ("concat_and_cache_mla", "indexer_k_quant_and_cache")
    ):
        pytest.skip("vllm._C cache ops are unavailable")
    return ops


def _slot_mapping(rank, dcp_size, block_ids, num_tokens, device) -> torch.Tensor:
    """Slots of tokens [0, num_tokens) on a DCP rank at interleave 1, from the
    V2 runner's block-table kernel; -1 for the other ranks' tokens."""
    from vllm.v1.worker.gpu.block_table import BlockTables

    tables = BlockTables(
        block_sizes=[1024],
        max_num_reqs=1,
        max_num_batched_tokens=num_tokens,
        max_num_blocks_per_group=[len(block_ids)],
        device=device,
        cp_size=dcp_size,
        cp_rank=rank,
        cp_interleave=1,
    )
    tables.append_block_ids(0, (list(block_ids),), overwrite=True)
    tables.apply_staged_writes()
    slots = tables.compute_slot_mappings(
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.tensor([0, num_tokens], dtype=torch.int32, device=device),
        torch.arange(num_tokens, dtype=torch.int64, device=device),
        num_tokens_padded=num_tokens,
    )
    return slots[0].clone()


def _write_tokens(ops, rank: _Rank, inputs: dict, slots: torch.Tensor) -> None:
    """Write each layer's rows of the slotted tokens through the real cache
    writers: latent and drafter pages by concat_and_cache_mla, indexer pages
    by indexer_k_quant_and_cache into their kernel pages, as the runner binds
    them."""
    num_tokens = slots.shape[0]
    scale = torch.ones(1, dtype=torch.float32, device=slots.device)
    for name, spec in rank.specs.items():
        view = rank.views[name]
        if spec.cache_role != SparseCacheRole.INDEXER:
            kv_c, k_pe = inputs[name]
            ops.concat_and_cache_mla(
                kv_c[:num_tokens], k_pe[:num_tokens], view[:, 0], slots, "fp8", scale
            )
            continue
        group = AttentionGroup(
            backend=DeepseekV32IndexerBackend,
            layer_names=[name],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
            kernel_block_size=INDEX_KERNEL_ROWS,
            maps_slot_mapping=True,
        )
        pages = group.map_kv_cache_to_kernel_blocks(view).squeeze(1)
        group.kernel_slot_mapping = torch.empty_like(slots)[None]
        page_slots = group.map_to_kernel_slot_mapping(slots)
        ops.indexer_k_quant_and_cache(
            inputs[name][:num_tokens], pages, page_slots, INDEX_VALUE_BYTES, "ue8m0"
        )


@requires_gpu
@pytest.mark.parametrize(
    "num_tokens,num_cached",
    [(3 * 16384 + 5, 0), (3 * 16384 + 5, 1), (5 * 8192 + 1, 0)],
    ids=["odd_p_blocks", "d_prefix_hit", "even_p_blocks"],
)
def test_real_writers_place_and_exchange_like_direct_writes(
    num_tokens: int, num_cached: int
):
    """A7: KV, indexer keys and drafter rows written by the real writers at the
    interleave-1 slot mappings of 8 P ranks, then pulled as today and exchanged
    by the Triton exchanger, equal what the same writers put in 16 D ranks
    directly: the whole caches, byte for byte. The only test whose ground truth
    does not come from this design's own tables."""
    ops = _cache_ops()
    device = torch.device("cuda")
    specs = _k31_dsa_specs(1024, 2, 1)
    gen = torch.Generator(device=device).manual_seed(num_tokens)
    inputs = {}
    for name, spec in specs.items():
        if spec.cache_role == SparseCacheRole.INDEXER:
            shape = (num_tokens, INDEX_VALUE_BYTES)
            inputs[name] = torch.randn(
                shape, dtype=torch.bfloat16, device=device, generator=gen
            )
        else:
            inputs[name] = tuple(
                torch.randn(
                    (num_tokens, dim),
                    dtype=torch.bfloat16,
                    device=device,
                    generator=gen,
                )
                for dim in (512, 64)
            )
    num_d = cdiv(num_tokens, D_DCP * 1024)
    num_p = cdiv(num_tokens, P_DCP * 1024)
    d_ids, p_ids = _block_tables(np.random.default_rng(num_tokens), num_d, num_p)

    p_ranks = [_Rank(specs, num_p + 2, device) for _ in range(P_DCP)]
    for p, rank in enumerate(p_ranks):
        slots = _slot_mapping(p, P_DCP, p_ids, num_tokens, device)
        _write_tokens(ops, rank, inputs, slots)
    direct = [_Rank(specs, num_d + 2, device) for _ in range(D_DCP)]
    pulled = [_Rank(specs, num_d + 2, device) for _ in range(D_DCP)]
    cached_tokens = min(num_tokens, num_cached * D_DCP * 1024)
    for r in range(D_DCP):
        _write_tokens(
            ops, direct[r], inputs, _slot_mapping(r, D_DCP, d_ids, num_tokens, device)
        )
        if cached_tokens:
            slots = _slot_mapping(r, D_DCP, d_ids, cached_tokens, device)
            _write_tokens(ops, pulled[r], inputs, slots)
    # The caches compared below start as zeros: make sure the writers filled
    # every layer, and every token's latent row on exactly one D rank.
    for name in specs:
        assert all(rank.views[name].count_nonzero() for rank in direct), name
    latent = next(
        n for n, s in specs.items() if s.cache_role != SparseCacheRole.INDEXER
    )
    rows = [rank.views[latent][:, 0].flatten(0, 1).any(-1).sum() for rank in direct]
    assert sum(int(n) for n in rows) == num_tokens

    _pull_as_today(pulled, d_ids, num_cached, p_ranks, p_ids)
    units = pair_units([d_ids[num_cached:]], [num_cached], [p_ids], groups=[0])
    for p in range(P_DCP):
        pair = [pulled[p], pulled[p + P_DCP]]
        _exchange_pair(
            _triton_exchanger,
            [r.buf for r in pair],
            [r.table for r in pair],
            [pair_layout(r, D_DCP, P_DCP, 1024, 1) for r in (p, p + P_DCP)],
            units,
            2,
        )
    torch.accelerator.synchronize()
    for r in range(D_DCP):
        assert torch.equal(pulled[r].buf, direct[r].buf), f"D rank {r}"


# ----------------------------------------------------------------------------
# A8: two ranks over NCCL
# ----------------------------------------------------------------------------


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2 or not HAS_TRITON,
    reason="needs 2 GPUs and Triton",
)
def test_two_ranks_over_nccl(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_two_rank_worker, args=(f"file://{tmp_path / 'init'}",), nprocs=2)


def _two_rank_worker(rank: int, init_method: str) -> None:
    from tests.utils import ensure_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    with ensure_current_vllm_config():
        init_distributed_environment(
            world_size=2,
            rank=rank,
            local_rank=rank,
            distributed_init_method=init_method,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=2, decode_context_model_parallel_size=2
        )
        try:
            _check_two_ranks(rank, torch.device("cuda", rank))
        finally:
            cleanup_dist_env_and_memory()


def _check_two_ranks(rank: int, device: torch.device) -> None:
    """The DCP group's exchange equals the reference; a partner with other
    chunking fails both warm-ups, each naming both ranks; and with PyNccl
    disabled the exchanger refuses to build, while one already built raises
    instead of exchanging nothing."""
    from vllm.distributed import get_dcp_group

    specs = _k31_dsa_specs(1024, 2, 1)
    # Both partners' caches, filled from per-rank seeds; this rank owns one.
    ranks = [_Rank(specs, 7, device) for _ in range(2)]
    for seed, replica in enumerate(ranks):
        replica.fill_random(seed)
    expected = [replica.buf.clone() for replica in ranks]
    reference_pair_exchange(
        expected,
        [{0: _rebased(r.table, r.buf, e)} for r, e in zip(ranks, expected)],
        _UNITS,
    )
    own, layout = ranks[rank], _PAIR[rank]

    exchanger = build_dcp_pair_exchanger(
        own.regions,
        index_kernel_rows=INDEX_KERNEL_ROWS,
        index_value_bytes=INDEX_VALUE_BYTES,
        layout=layout,
        device=device,
        chunk_units=2,
    )
    before = own.buf.clone()
    exchanger.warm_up()
    # The warm-up exchanges a scratch block, never the KV cache.
    assert torch.equal(own.buf, before)
    exchanger.exchange(_UNITS)
    exchanger.record_event().synchronize()
    assert torch.equal(own.buf, expected[rank])

    skewed = DcpPairExchanger(own.table, layout, device, chunk_units=2 + rank)
    with pytest.raises(
        NixlPairModeError,
        match=rf"DCP rank {rank} .*NCCL peer, DCP rank {1 - rank} .*chunking differ",
    ):
        skewed.warm_up()

    pynccl = get_dcp_group().device_communicator.pynccl_comm
    pynccl.disabled = True
    try:
        with pytest.raises(NixlPairModeError, match="disabled"):
            DcpPairExchanger(own.table, layout, device)
        with pytest.raises(ValueError, match="PyNCCL"):
            exchanger.exchange(_UNITS)
    finally:
        pynccl.disabled = False
