# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where the NIXL KV fingerprint is taken (design §4, gate A9), on CPU.

With VLLM_NIXL_KV_FINGERPRINT on, the schedulers send each request's
fingerprint inputs: P's when it finishes a request a decoder will pull, D's
when a pull ends, with its pair-exchange plan. D's workers log right after the
exchange, with the request's own block table and prefix hit; P's before
reporting the send, which they hold back until then. A P DCP8 -> D DCP16 pull
runs through the real NIXL workers on the pair exchange worker tests' byte
moving transport, and the comparator script checks the records. Also covered:
the exchange's GPU time in the iterlog, producers ignoring dcp_pair_exchange,
the startup region log, and nothing changing with the fingerprint off.
"""

import json
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from tests.v1.kv_connector.unit import test_nixl_kv_fingerprint as fingerprint_tests
from tests.v1.kv_connector.unit import (
    test_nixl_pair_exchange_scheduler as scheduler_tests,
)
from tests.v1.kv_connector.unit import test_nixl_pair_exchange_worker as worker_tests
from tests.v1.kv_connector.unit.test_nixl_dcp_pair_exchange import (
    _canonical_tokens,
    _row_tags,
)
from tests.v1.kv_connector.unit.utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    base_worker,
    dcp_pair_exchange,
    page_zeroer,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.dcp_pair_exchange import (
    PairUnit,
    pair_units,
    reference_pair_exchange,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.kv_fingerprint import (
    FINGERPRINT_DIR_ENV,
    FINGERPRINT_ENV,
    KVFingerprintLog,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    KVFingerprintReq,
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.metrics import iterlog
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test

P_DCP, D_DCP = worker_tests.P_DCP, worker_tests.D_DCP
SMALL = dict(block_size=64, num_dsa=2, num_kda=3)
PAIR = {"dcp_pair_exchange": True}


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(num_threads)


@pytest.fixture
def k31_env(monkeypatch):
    """Kimi-K3.1's KDA state groups and DS conv layout, as in the recipe."""
    from vllm.model_executor.layers.mamba import mamba_utils

    monkeypatch.setenv("VLLM_PACKED_KV_STATE_GROUPS", "2")
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
    mamba_utils.get_conv_state_layout.cache_clear()
    yield
    mamba_utils.get_conv_state_layout.cache_clear()


@pytest.fixture
def kvfp_dir(monkeypatch, tmp_path) -> Path:
    """VLLM_NIXL_KV_FINGERPRINT on, logging to a fresh directory."""
    out_dir = tmp_path / "kvfp"
    monkeypatch.setenv(FINGERPRINT_ENV, "1")
    monkeypatch.setenv(FINGERPRINT_DIR_ENV, str(out_dir))
    return out_dir


@pytest.fixture
def kvfp_off(monkeypatch, tmp_path) -> Path:
    """VLLM_NIXL_KV_FINGERPRINT off, with a directory it must not write."""
    out_dir = tmp_path / "kvfp"
    monkeypatch.delenv(FINGERPRINT_ENV, raising=False)
    monkeypatch.setenv(FINGERPRINT_DIR_ENV, str(out_dir))
    return out_dir


class _EagerExchanger(worker_tests._FakeExchanger):
    """Swaps both partners' rows at the first one's call, as NCCL has done by
    the time either rank's stream reads the blocks; the partner's call then
    only checks its units."""

    def exchange(self, units) -> None:
        units = list(units)
        fabric = self.fabric
        fabric.log.append(("exchange", self.layout.dcp_rank, units))
        swapped = fabric.swapped.pop(self.layout.dcp_rank, None)
        if swapped is not None:
            assert swapped == units, "partners must exchange the same units"
            return
        other = fabric.exchangers[self.layout.partner]
        pair = sorted((self, other), key=lambda e: e.layout.parity)
        reference_pair_exchange(
            [e.buf for e in pair],
            [{u.group: e.table for u in units} for e in pair],
            units,
        )
        fabric.swapped[self.layout.partner] = units


class _EagerFabric(worker_tests._Fabric):
    def __init__(self, memory, log):
        super().__init__(memory, log)
        self.exchangers: dict[int, _EagerExchanger] = {}
        self.swapped: dict[int, list[PairUnit]] = {}

    def build(self, regions, **kwargs):
        built = super().build(regions, **kwargs)
        exchanger = _EagerExchanger(self, built.buf, built.table, built.layout)
        self.exchangers[built.layout.dcp_rank] = exchanger
        return exchanger


@pytest.fixture
def fabric(monkeypatch) -> _EagerFabric:
    fabric = _EagerFabric(worker_tests._Memory(), [])
    monkeypatch.setattr(
        dcp_pair_exchange, "build_dcp_pair_exchanger", fabric.build, raising=False
    )
    # The page zeroer stages block ids in pinned memory, which CPU hosts lack.
    monkeypatch.setattr(
        page_zeroer,
        "async_tensor_h2d",
        lambda data, device, dtype: torch.tensor(data, dtype=dtype, device=device),
    )
    return fabric


class _BoundRank(worker_tests._Rank):
    """A rank whose every DSA layer (latent, indexer and drafter pages) is
    bound to the view its backend reads, as the runner binds them before the
    connector registers the caches."""

    def _start_worker(self, fabric, layers, extra):
        bound = {
            name: SimpleNamespace(kv_cache=self.views[name].squeeze(1))
            for name in self.dsa_layers
            if name not in layers
        }
        return super()._start_worker(fabric, {**layers, **bound}, extra)


# ----------------------------------------------------------------------------
# P and D records of a pull, checked by the comparator
# ----------------------------------------------------------------------------


class _Pull:
    """A request's P DCP8 -> D DCP16 pull at interleave 1 on the real NIXL
    workers: P ranks hold its rows, D ranks hold their rows of its first
    ``num_cached`` D blocks (the prefix hit) and pull the rest.

    ``read_hit`` is the hit the reads and the plan count from: the true one,
    or another to model a scheduler bug that both would share.
    """

    def __init__(
        self,
        fabric: _EagerFabric,
        num_tokens: int,
        num_cached: int,
        *,
        num_draft: int = 1,
        read_hit: int | None = None,
    ):
        self.num_tokens, self.num_cached = num_tokens, num_cached
        block_size = SMALL["block_size"]
        num_p = cdiv(num_tokens, P_DCP * block_size)
        num_d = cdiv(num_tokens, D_DCP * block_size) + 1  # A lookahead block.
        rng = np.random.default_rng(num_tokens)
        d_pool = rng.permutation(num_d + 3).tolist()
        p_pool = rng.permutation(num_p + 3).tolist()
        self.d_ids, d_kda = d_pool[:num_d], d_pool[num_d : num_d + 2]
        self.p_ids, p_kda = p_pool[:num_p], p_pool[num_p : num_p + 2]
        geometry = dict(SMALL, num_draft=num_draft)

        self.prefillers = {
            p: _BoundRank(fabric, P_DCP, p, num_p + 3, **geometry) for p in range(P_DCP)
        }
        arrays = self.prefillers[0].arrays()
        for p, rank in self.prefillers.items():
            tokens = _canonical_tokens(p, P_DCP, block_size, num_p)
            held = (tokens < num_tokens)[..., None]
            for a, array in enumerate(arrays):
                valid = _row_tags(tokens, a, array[2])
                junk = _row_tags(tokens, a + len(arrays), array[2])
                rank.write_rows(array, self.p_ids, np.where(held, valid, junk))
        self.decoders = [
            _BoundRank(fabric, D_DCP, r, num_d + 3, extra_config=PAIR, **geometry)
            for r in range(D_DCP)
        ]
        for r, rank in enumerate(self.decoders):
            tokens = _canonical_tokens(r, D_DCP, block_size, num_cached)
            for a, array in enumerate(arrays):
                rank.write_rows(
                    array, self.d_ids[:num_cached], _row_tags(tokens, a, array[2])
                )
            worker_tests._handshake(rank, self.prefillers)

        rank0 = self.decoders[0]
        self.d_table = worker_tests._per_group(rank0, self.d_ids, [[b] for b in d_kda])
        self.remote = worker_tests._per_group(rank0, self.p_ids, [[b] for b in p_kda])
        self.local = worker_tests._per_group(
            rank0, self.d_ids[num_cached:], [[b] for b in d_kda]
        )
        hit = num_cached if read_hit is None else read_hit
        self.computed = worker_tests._per_group(rank0, hit, [0, 0])
        self.dsa_group = rank0.dsa_group

    def p_logs_its_kv(self) -> None:
        """The step after P's request_finished, on every P rank."""
        for rank in self.prefillers.values():
            metadata = NixlConnectorMetadata()
            metadata.reqs_to_send = {"req-P": 1e9}
            metadata.fingerprint_sends = {
                "req-P": KVFingerprintReq("req-P", self.remote, self.num_tokens)
            }
            connector = worker_tests._facade(rank.worker)
            connector.bind_connector_metadata(metadata)
            connector.start_load_kv(None)

    def d_pulls_and_logs(self) -> list[PairUnit]:
        """Every D rank pulls; in the next step each exchanges the plan and
        logs the request with its own block table and true prefix hit."""
        for rank in self.decoders:
            rank.worker.start_load_kv(
                worker_tests._recv_metadata(
                    "req", self.local, self.computed, self.remote, self.num_tokens
                )
            )
            results = rank.worker.get_transfer_results()
            assert results.finished_recving == {"req"}
            assert not results.failed_recving
        units = pair_units(
            self.local, self.computed, self.remote, groups=[self.dsa_group]
        )
        for rank in self.decoders:
            step = NixlConnectorMetadata()
            step.reqs_to_pair_exchange = {"req": units}
            step.fingerprint_pulls = {
                "req": KVFingerprintReq(
                    "req-P",
                    self.d_table,
                    self.num_tokens,
                    first_block=self.num_cached,
                )
            }
            connector = worker_tests._facade(rank.worker)
            connector.finalize_loaded_kv(step)
            connector.bind_connector_metadata(step)
        return units

    def relog_d(self, out_dir: Path, first_block: int, table) -> None:
        """Log D's ranks again into ``out_dir``, as they are now."""
        for rank in self.decoders:
            log = rank.worker._fingerprint_log
            rank.worker._fingerprint_log = KVFingerprintLog(
                str(out_dir), "D", log.dcp_rank, log.grid, log.engine_id
            )
            rank.worker._log_fingerprints(
                {
                    "req": KVFingerprintReq(
                        "req-P", table, self.num_tokens, first_block=first_block
                    )
                }
            )
            rank.worker._fingerprint_log = log


def _records(out_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for path in sorted(out_dir.glob("kvfp-*.jsonl"))
        for line in path.read_text().splitlines()
    ]


def _copy_p_records(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.glob("kvfp-P-*.jsonl"):
        shutil.copy(path, dst / path.name)


def test_pull_records_on_p_and_d_pass_the_comparator(
    k31_env, fabric, kvfp_dir, tmp_path, capsys, monkeypatch
):
    """A9 on CPU: an odd P block count (the last D block half filled) and a D
    prefix hit. P logs every block it holds, D its blocks from the hit on,
    right after the exchange: the comparator passes, then catches one flipped
    byte of an exchanged drafter row on one D rank and names its cell."""
    compare = fingerprint_tests._comparator(monkeypatch)
    num_tokens, m = 2100, 1
    pull = _Pull(fabric, num_tokens, m)
    assert len(pull.p_ids) % 2 == 1
    pull.p_logs_its_kv()
    units = pull.d_pulls_and_logs()
    assert [u.has_partner for u in units] == [True, False]

    records = _records(kvfp_dir)
    p_recs = [r for r in records if r["side"] == "P"]
    d_recs = [r for r in records if r["side"] == "D"]
    assert sorted(r["rank"] for r in p_recs) == list(range(P_DCP))
    assert sorted(r["rank"] for r in d_recs) == list(range(D_DCP))
    for rec in records:
        assert rec["req"] == "req-P" and rec["num_tokens"] == num_tokens
        assert rec["grid"] == [P_DCP, D_DCP, SMALL["block_size"]]
        assert rec["extra"]["group"] == pull.dsa_group
    # P: the blocks request_finished gets. D: its own table from the hit on,
    # lookahead block included, not the plan's units.
    assert all(r["block_ids"] == pull.p_ids and r["first_block"] == 0 for r in p_recs)
    assert all(r["block_ids"] == pull.d_ids[m:] for r in d_recs)
    assert all(r["first_block"] == m for r in d_recs)
    assert [u.block_id for u in units] != pull.d_ids[m:]
    assert all(r["extra"]["local_req"] == "req" for r in d_recs)
    layers = {e["layer"]: e["draft"] for e in d_recs[0]["fp"]}
    assert layers.keys() == set(pull.decoders[0].dsa_layers)
    assert sum(layers.values()) == 1

    assert compare.main([str(kvfp_dir)]) == 0
    out = capsys.readouterr().out
    assert "1 checked, 0 P only" in out and "0 MISMATCH" in out
    assert "0 missing on P, 0 missing on D" in out

    # D rank 5, its block j = 1, row 40 of 64: half 1, from P block 3.
    rank = pull.decoders[5]
    (draft,) = (n for n in rank.dsa_layers if n.startswith("draft."))
    rank.views[draft][pull.d_ids[1], 0, 40, 7] ^= 0x10
    bad = tmp_path / "bad"
    _copy_p_records(kvfp_dir, bad)
    pull.relog_d(bad, m, pull.d_table)
    report = tmp_path / "report.json"
    assert compare.main([str(bad), "--json", str(report)]) == 1
    (x,) = json.loads(report.read_text())["mismatches"]
    assert (x["layer"], x["field"], x["j"], x["rho"], x["h"]) == (draft, "kv", 1, 5, 1)
    assert (x["d_block"], x["d_rows"]) == (pull.d_ids[1], [32, 64])
    assert (x["p_rank"], x["p_block_index"], x["p_block"]) == (5, 3, pull.p_ids[3])


def test_a_hit_shared_by_the_reads_and_the_plan_is_caught(
    k31_env, fabric, kvfp_dir, tmp_path, monkeypatch
):
    """The reads and the plan take the same wrong prefix hit (0 for a true 1),
    so the exchange leaves each block in D order but one chunk early. D logs
    with the hit its token counts give and its own table: every pulled cell
    mismatches. Logged with the plan's view instead (its blocks, its hit), the
    run would pass."""
    compare = fingerprint_tests._comparator(monkeypatch)
    pull = _Pull(fabric, 2100, num_cached=1, read_hit=0)
    pull.p_logs_its_kv()
    pull.d_pulls_and_logs()
    report = tmp_path / "report.json"
    assert compare.main([str(kvfp_dir), "--json", str(report)]) == 1
    mismatches = json.loads(report.read_text())["mismatches"]
    assert {x["j"] for x in mismatches} == {1, 2}

    plan_view = tmp_path / "plan_view"
    _copy_p_records(kvfp_dir, plan_view)
    pull.relog_d(plan_view, 0, pull.local)
    assert compare.main([str(plan_view)]) == 0


@pytest.mark.parametrize(
    ("num_tokens", "num_cached"),
    [(2048, 0), (1025, 0), (2100, 2)],
    ids=["even-P-blocks,L-on-a-D-block", "one-token-in-the-last-D-block", "m=J-1"],
)
def test_pull_records_match_at_tail_boundaries(
    k31_env, fabric, kvfp_dir, capsys, monkeypatch, num_tokens, num_cached
):
    """Every partner pair, a last D block holding one token of D rank 0 only,
    and a hit covering all but the last D block: P and D log the same cells
    below L, and nothing past it."""
    compare = fingerprint_tests._comparator(monkeypatch)
    pull = _Pull(fabric, num_tokens, num_cached)
    pull.p_logs_its_kv()
    pull.d_pulls_and_logs()
    assert compare.main([str(kvfp_dir)]) == 0
    out = capsys.readouterr().out
    assert "1 checked, 0 P only" in out and "0 MISMATCH" in out
    assert "0 missing on P, 0 missing on D, 0 unexpected on D" in out


def test_a_pull_without_a_plan_is_missing_on_d(
    k31_env, fabric, kvfp_dir, tmp_path, capsys, monkeypatch
):
    """A pull that ended without its plan being emitted is logged without
    cells: the comparator reports its cells missing on D rather than listing
    the request as P only, as it does for a request D never logged."""
    compare = fingerprint_tests._comparator(monkeypatch)
    pull = _Pull(fabric, 1500, num_cached=0, num_draft=0)
    pull.p_logs_its_kv()
    for rank in pull.decoders:
        step = NixlConnectorMetadata()
        step.fingerprint_pulls = {
            "req": KVFingerprintReq("req-P", pull.d_table, 1500, exchanged=False)
        }
        worker_tests._facade(rank.worker).finalize_loaded_kv(step)
    d_recs = [r for r in _records(kvfp_dir) if r["side"] == "D"]
    assert len(d_recs) == D_DCP
    for rec in d_recs:
        assert rec["cells"] == [] and rec["fp"] == [] and rec["block_ids"] == []
        assert rec["extra"]["exchanged"] is False
    assert not any(entry[0] == "exchange" for entry in fabric.log)
    assert compare.main([str(kvfp_dir)]) == 1
    out = capsys.readouterr().out
    assert "1 checked, 0 P only" in out and "0 missing on P" in out

    p_only = tmp_path / "p_only"
    _copy_p_records(kvfp_dir, p_only)
    assert compare.main([str(p_only)]) == 1  # Nothing checked.
    assert "0 checked, 1 P only" in capsys.readouterr().out


# ----------------------------------------------------------------------------
# P holds a request's finished_sending until its record is written
# ----------------------------------------------------------------------------


def _p_step(worker, *, notified: tuple[str, ...] = (), **fields):
    """One P worker step: D's read notifications (each from its only reader)
    arrive, the scheduler's metadata is bound and loaded, and the step's
    results are returned."""
    metadata = NixlConnectorMetadata()
    for name, value in fields.items():
        setattr(metadata, name, value)
    connector = worker_tests._facade(worker)
    connector.bind_connector_metadata(metadata)
    connector.start_load_kv(None)
    worker.nixl_wrapper.get_new_notifs = lambda: {
        "D": [f"{req_id}:1".encode() for req_id in notified]
    }
    return worker.get_transfer_results()


def _send(req_id: str, block_ids) -> dict:
    return dict(
        reqs_to_send={req_id: 1e9},
        fingerprint_sends={req_id: KVFingerprintReq(req_id, block_ids, 600)},
    )


def test_p_holds_a_send_until_its_record_is_written(k31_env, fabric, kvfp_dir):
    """D can read and notify before the metadata announcing the send reaches
    P's worker (P lagging under async scheduling). The worker holds the send
    until that metadata's step has logged the request, then reports it, and
    does not start a lease for the read that already ended. A send announced
    first is reported as soon as D's notification arrives."""
    rank = _BoundRank(fabric, P_DCP, 2, 6, **SMALL)
    worker = rank.worker
    assert worker._fingerprint_log.side == "P"
    blocks = worker_tests._per_group(rank, [1, 4], [[0], [5]])

    assert not _p_step(worker, reqs_in_batch={"early", "late"}).finished_sending
    held = _p_step(worker, notified=("early",))
    assert not held.finished_sending and worker._fingerprint_held == {"early"}
    assert not _records(kvfp_dir)
    # A heartbeat for it extends nothing and releases nothing.
    worker._handle_heartbeat("early")
    assert "early" not in worker._reqs_to_send

    released = _p_step(worker, **_send("early", blocks))
    assert released.finished_sending == {"early"}
    assert "early" not in worker._reqs_to_send
    (record,) = _records(kvfp_dir)
    assert record["req"] == "early" and record["block_ids"] == [1, 4]
    assert not worker._fingerprint_held and not worker._fingerprint_decided

    # Announced first: its lease runs until D's notification reports it.
    assert not _p_step(worker, **_send("late", blocks)).finished_sending
    assert "late" in worker._reqs_to_send
    assert _p_step(worker, notified=("late",)).finished_sending == {"late"}
    assert len(_records(kvfp_dir)) == 2
    assert not worker._fingerprint_decided


def test_an_expired_lease_is_reported_once_logged(k31_env, fabric, kvfp_dir):
    """A send whose lease runs out is logged by then (its lease starts in the
    step that logs it), so it is reported at expiry, not held."""
    rank = _BoundRank(fabric, P_DCP, 0, 6, **SMALL)
    _p_step(rank.worker, reqs_in_batch={"x"})
    send = _send("x", worker_tests._per_group(rank, [2, 5], [[0], [3]]))
    send["reqs_to_send"] = {"x": 0.0}  # Already expired.
    assert _p_step(rank.worker, **send).finished_sending == {"x"}
    assert len(_records(kvfp_dir)) == 1


def test_a_failed_record_does_not_hold_the_send(k31_env, fabric, kvfp_dir):
    """A P record that cannot be taken (here: more tokens than its blocks
    hold) is reported, not raised, and its send is reported as usual: the
    comparator finds its cells missing on P, and P never keeps the blocks."""
    rank = _BoundRank(fabric, P_DCP, 1, 6, **SMALL)
    _p_step(rank.worker, reqs_in_batch={"x"})
    send = _send("x", worker_tests._per_group(rank, [1, 4], [[0], [5]]))
    send["fingerprint_sends"]["x"].num_tokens = 10_000
    logger = MagicMock()
    with patch.object(base_worker, "logger", logger):
        assert not _p_step(rank.worker, **send).finished_sending
    logger.exception.assert_called_once()
    assert not _records(kvfp_dir)
    assert _p_step(rank.worker, notified=("x",)).finished_sending == {"x"}
    assert not rank.worker._fingerprint_held
    assert not rank.worker._fingerprint_decided


def test_p_scheduler_keeps_the_lease_while_the_send_is_held(kvfp_dir):
    """The P scheduler answers D's lease queries from the sends it has not
    seen finish: a send the workers hold stays leased, so D never discards a
    read because of the hold, and the blocks stay allocated."""
    scheduler = create_scheduler(create_vllm_config(kv_role="kv_producer"))
    nixl = scheduler.connector.connector_scheduler
    nixl._hold_send_lease("x")
    nixl.update_connector_output(KVConnectorOutput())
    assert nixl._query_send_leases(["x"]) == [True]
    nixl.update_connector_output(KVConnectorOutput(finished_sending={"x"}))
    assert nixl._query_send_leases(["x"]) == [False]


# ----------------------------------------------------------------------------
# Schedulers: what they send, and when
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("kda_group", [None, 0], ids=["attention", "kda+attention"])
def test_d_scheduler_sends_each_pull_with_its_plan(kvfp_dir, kda_group):
    """Two turns, the second hitting the m = 2 D blocks the first cached. Each
    pull's fingerprint inputs go with its plan, once: P's request id, the
    request's whole block table from the KV cache manager, the tokens D loads
    and the hit in blocks. With K3.1's state group first, the table is indexed
    by KV cache group and D loads all but the backoff token."""
    engine = scheduler_tests._Engine(
        scheduler_tests._make_scheduler(kda_group=kda_group)
    )
    m, d_block = 2, scheduler_tests.D_BLOCK
    attn, backoff = (0, 0) if kda_group is None else (1, 1)
    turn1 = scheduler_tests._pull_request(
        1, m * d_block + 5 + backoff, kda_group=kda_group, common_prefix_len=m * d_block
    )
    engine.add(turn1)
    engine.step()
    engine.step(finished={turn1.request_id})
    output = engine.step()
    meta = output.kv_connector_metadata
    assert meta.fingerprint_pulls == {
        turn1.request_id: KVFingerprintReq(
            "prefill-1",
            engine.scheduler.kv_cache_manager.get_block_ids(turn1.request_id),
            m * d_block + 5,
        )
    }
    assert meta.reqs_to_pair_exchange.keys() == {turn1.request_id}

    turn2 = scheduler_tests._pull_request(
        2, 4 * d_block + 3 + backoff, kda_group=kda_group, common_prefix_len=m * d_block
    )
    engine.add(turn2)
    engine.step()
    table = engine.scheduler.kv_cache_manager.get_block_ids(turn2.request_id)
    pull_ended = engine.step(finished={turn2.request_id})
    assert not pull_ended.kv_connector_metadata.fingerprint_pulls
    meta = engine.step().kv_connector_metadata
    assert meta.fingerprint_pulls == {
        turn2.request_id: KVFingerprintReq(
            "prefill-2", table, 4 * d_block + 3, first_block=m
        )
    }
    # D logs table[attn][m:]: the blocks the plan exchanges, found without it.
    units = meta.reqs_to_pair_exchange[turn2.request_id]
    assert {u.group for u in units} == {attn}
    assert [u.block_id for u in units] == table[attn][m : m + len(units)]
    assert meta.reqs_to_pair_exchange.keys() == {turn2.request_id}
    assert not meta.fingerprint_sends
    # As the executor's scheduler-output queue sends it to every rank.
    assert pickle.loads(pickle.dumps(meta)).fingerprint_pulls == meta.fingerprint_pulls
    assert not engine.nixl._fingerprint_pulls
    assert not engine.nixl._fingerprint_pulls_ready


@pytest.mark.parametrize("fate", ["no_plan", "failed", "aborted", "aborted_ready"])
def test_d_scheduler_on_a_pull_that_is_not_exchanged(kvfp_dir, fate):
    """A pull that ended without a plan to emit is sent unexchanged, so D logs
    it without cells. A failed pull (recomputed) or a finished request is sent
    nothing: P's record alone marks it P only."""
    engine = scheduler_tests._Engine(scheduler_tests._make_scheduler())
    request = scheduler_tests._pull_request(1, 3 * scheduler_tests.P_BLOCK)
    req_id = request.request_id
    engine.add(request)
    engine.step()
    if fate == "no_plan":
        del engine.nixl._pair_plans[req_id]
    elif fate == "aborted":
        engine.abort(req_id)
    engine.step(
        finished=None if fate == "failed" else {req_id},
        failed={req_id} if fate == "failed" else None,
    )
    if fate == "aborted_ready":
        engine.abort(req_id)
    sent = engine.step().kv_connector_metadata
    if fate == "no_plan":
        (pulled,) = sent.fingerprint_pulls.values()
        assert not pulled.exchanged and pulled.req_id == "prefill-1"
    else:
        assert not sent.fingerprint_pulls
    assert not sent.reqs_to_pair_exchange
    assert not engine.nixl._fingerprint_pulls
    assert not engine.nixl._fingerprint_pulls_ready


def _p_engine(kv_role: str = "kv_producer"):
    scheduler = create_scheduler(create_vllm_config(kv_role=kv_role))
    request = create_request(request_id=1, num_tokens=40, do_remote_decode=True)
    scheduler.add_request(request)
    return scheduler, request


def test_p_scheduler_sends_each_finished_request_with_its_lease(kvfp_dir):
    """P sends a request's fingerprint inputs with its reqs_to_send, in the
    step after request_finished: the block ids that request_finished gets
    and the tokens P computed."""
    scheduler, request = _p_engine()
    output = scheduler.schedule()
    blocks = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    scheduler.update_from_output(output, create_model_runner_output([request]))
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    meta = scheduler.schedule().kv_connector_metadata
    assert meta.reqs_to_send.keys() == {request.request_id}
    assert meta.fingerprint_sends == {
        request.request_id: KVFingerprintReq(
            request.request_id, blocks, request.num_computed_tokens
        )
    }
    assert not scheduler.schedule().kv_connector_metadata.fingerprint_sends


# ----------------------------------------------------------------------------
# Off by default; producers ignore the pair opt-in
# ----------------------------------------------------------------------------


def test_fingerprint_off_changes_nothing(k31_env, fabric, kvfp_off):
    """Off, the schedulers send nothing and keep nothing, the workers open no
    log, report sends at once, and write no file."""
    engine = scheduler_tests._Engine(scheduler_tests._make_scheduler())
    assert not engine.nixl._fingerprint
    request = scheduler_tests._pull_request(1, 3 * scheduler_tests.P_BLOCK)
    engine.add(request)
    outputs = [engine.step(), engine.step(finished={request.request_id})]
    outputs += [engine.step(), engine.step(eos=True)]
    assert any(o.kv_connector_metadata.reqs_to_pair_exchange for o in outputs)
    assert not any(o.kv_connector_metadata.fingerprint_pulls for o in outputs)
    assert not engine.nixl._fingerprint_pulls

    scheduler, p_request = _p_engine()
    output = scheduler.schedule()
    scheduler.update_from_output(output, create_model_runner_output([p_request]))
    meta = scheduler.schedule().kv_connector_metadata
    assert meta.reqs_to_send and not meta.fingerprint_sends

    p_worker = _BoundRank(fabric, P_DCP, 1, 6, **SMALL).worker
    d_worker = _BoundRank(fabric, D_DCP, 1, 6, extra_config=PAIR, **SMALL).worker
    assert p_worker._fingerprint_log is None and d_worker._fingerprint_log is None
    _p_step(p_worker, reqs_in_batch={"x"})
    assert _p_step(p_worker, notified=("x",)).finished_sending == {"x"}
    # A stub worker, as A5 builds them: fingerprint fields touch nothing.
    stub = object.__new__(base_worker.NixlBaseConnectorWorker)
    step = NixlConnectorMetadata()
    step.fingerprint_pulls = {"r": KVFingerprintReq("p", ([1],), 16)}
    step.fingerprint_sends = {"s": KVFingerprintReq("s", ([1],), 16)}
    stub.finalize_loaded_kv(step)
    stub.log_send_fingerprints(step)
    assert vars(stub) == {}
    assert not kvfp_off.exists()


@pytest.mark.parametrize("kv_role", ["kv_producer", "kv_both"])
def test_producers_ignore_the_pair_opt_in(k31_env, fabric, monkeypatch, kv_role):
    """dcp_pair_exchange fixes up a decoder's pulls. A producer's workers warn
    and build no exchanger, and its scheduler plans nothing, so the two never
    disagree."""

    def nixl_scheduler(role: str) -> NixlPullConnectorScheduler:
        config = create_vllm_config(kv_role=role, kv_connector_extra_config=PAIR)
        return create_scheduler(config).connector.connector_scheduler

    assert not nixl_scheduler(kv_role)._pair_mode
    assert nixl_scheduler("kv_consumer")._pair_mode

    config_cls = worker_tests.KVTransferConfig
    monkeypatch.setattr(
        worker_tests,
        "KVTransferConfig",
        lambda **kwargs: config_cls(**{**kwargs, "kv_role": kv_role}),
    )
    logger = MagicMock()
    with patch.object(base_worker, "logger", logger):
        worker = _BoundRank(fabric, P_DCP, 0, 6, extra_config=PAIR, **SMALL).worker
    assert worker._pair_exchanger is None and not worker._dcp_pair_exchange
    assert not fabric.builds
    (warning,) = [
        c for c in logger.warning.call_args_list if "dcp_pair_exchange" in c.args[0]
    ]
    assert warning.args[1] == kv_role


# ----------------------------------------------------------------------------
# Startup log and iterlog
# ----------------------------------------------------------------------------


def test_startup_logs_the_regions_and_row_table(k31_env, fabric):
    """Once at startup: the NIXL regions per transfer group, the drafter
    layers and the row table, for stage 6 to check against the model."""
    logger = MagicMock()
    with patch.object(base_worker, "logger", logger):
        rank = _BoundRank(fabric, D_DCP, 3, 6, extra_config=PAIR, num_draft=2, **SMALL)
    worker = rank.worker
    (call,) = [
        c for c in logger.info.call_args_list if "pair exchange, DCP rank" in c.args[0]
    ]
    (
        _,
        dcp_rank,
        num_regions,
        per_group,
        num_draft,
        group,
        num_pages,
        draft_pages,
        block_size,
        stride,
        arrays,
        half_bytes,
        chunk_units,
    ) = call.args
    table = worker._pair_exchanger.table
    assert (dcp_rank, num_regions) == (3, worker.num_regions)
    assert sum(per_group.values()) == worker.num_regions
    assert per_group[rank.dsa_group] == len(rank.dsa_layers) == 6
    assert (num_draft, draft_pages) == (2, 2)
    assert (group, num_pages) == (rank.dsa_group, 6)
    assert (block_size, stride) == (64, table.block_stride)
    assert arrays == "4 x row 576 B, 2 x index_value 128 B, 2 x index_scale 4 B"
    # Half of a block's 64 rows, over those arrays' widths.
    assert half_bytes == 32 * (4 * 576 + 2 * 128 + 2 * 4) == table.half_bytes
    assert chunk_units == 4


class _FakeTimingEvent:
    def __init__(self, clock: list):
        self.clock, self.at, self.done = clock, 0.0, False

    def record(self) -> None:
        self.at = self.clock[0]

    def query(self) -> bool:
        return self.done

    def elapsed_time(self, end: "_FakeTimingEvent") -> float:
        assert self.done and end.done
        return end.at - self.at

    def synchronize(self) -> None:
        raise AssertionError("the exchange timing must never wait on the GPU")


def test_exchange_gpu_time_goes_to_the_iterlog(monkeypatch, tmp_path):
    """Each step with units writes one X record: the exchange's GPU time from
    events around it, read once they complete (never waited on), with the
    step's request and unit counts."""
    monkeypatch.setattr(iterlog, "ITERLOG_DIR", str(tmp_path))
    clock, events = [0.0], []

    def new_event():
        events.append(_FakeTimingEvent(clock))
        return events[-1]

    times = base_worker._PairExchangeTimes(5, {"dcp_rank": 5}, new_event)
    log: list = []
    worker = object.__new__(base_worker.NixlBaseConnectorWorker)
    worker._pair_exchanger = SimpleNamespace(
        exchange=lambda units: clock.__setitem__(0, clock[0] + 0.5 * len(units)),
        record_event=lambda: worker_tests._FakeEvent(log, "guard"),
    )
    worker._pair_group = 2
    worker._pair_times = times

    for blocks in ([3, 4], [7]):
        step = NixlConnectorMetadata()
        step.reqs_to_pair_exchange = {f"r{b}": [PairUnit(2, b, True)] for b in blocks}
        worker.run_pair_exchanges(step)
    assert len(times._inflight) == 2
    times.poll()
    assert len(times._inflight) == 2  # Neither has completed: nothing waits.
    for event in events[:2]:
        event.done = True
    times.poll()
    times._writer.close()
    path = Path(times._writer.path)
    assert path.name.startswith("nixl-pair-dcp5-")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0][0] == "H"
    assert lines[0][1]["fields"] == base_worker._PairExchangeTimes.FIELDS
    (record,) = [line for line in lines if line[0] == "X"]
    k, _, n_reqs, n_units, g_exchange = record[1:]
    assert (k, n_reqs, n_units, g_exchange) == (0, 2, 2, 1.0)
    assert len(times._inflight) == 1


def test_iterlog_ring_overflow_writes_untimed_records(monkeypatch, tmp_path):
    """With every event pair in flight, a step is written untimed rather than
    reusing events still in use."""
    monkeypatch.setattr(iterlog, "ITERLOG_DIR", str(tmp_path))
    monkeypatch.setattr(base_worker._PairExchangeTimes, "RING", 3)
    times = base_worker._PairExchangeTimes(0, {}, lambda: _FakeTimingEvent([0.0]))
    for k in range(3):
        times.begin()
        times.end(1, k + 1)
    times._writer.close()
    (record,) = [
        json.loads(line)
        for line in Path(times._writer.path).read_text().splitlines()
        if line.startswith('["X"')
    ]
    assert record[1] == 2 and record[3:] == [1, 3, -1.0]
    assert len(times._inflight) == 2


def test_no_exchange_timing_without_cuda_caches(k31_env, fabric, monkeypatch, tmp_path):
    """The exchange is timed only with CUDA events on CUDA caches; CPU caches
    (these tests) build no timer even with the iterlog on."""
    monkeypatch.setattr(iterlog, "ITERLOG_DIR", str(tmp_path))
    worker = _BoundRank(fabric, D_DCP, 0, 6, extra_config=PAIR, **SMALL).worker
    assert worker._pair_exchanger is not None and worker._pair_times is None
