# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests: the rows of the DCP path's direct workspaces
(``get_dcp_workspace_max_num_tokens``) and what happens past them.

Under DCP ShardQ the DCP path keeps decode rows, few-token steps and FULL
decode graphs only (PIECEWISE graphs run attention eagerly on the real rows),
so the bound no longer follows the largest CUDA graph capture size: a
16384-token capture used to size the direct workspaces, the trtllm-gen
softmax-stats slab and the profile's DCP-path claim for 16384 rows (~11 GiB per
GPU at Kimi K3.1 TP8+DCP8). Every other instance (decode-only P/D consumers,
DCP without ShardQ) keeps the old bound. A DCP ShardQ layer runs larger
DCP-path passes in row chunks of the bound, and its manager raises past the
direct workspaces instead of silently taking the NCCL fallback.
"""

import functools
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fi
from vllm.v1.attention.ops import dcp
from vllm.v1.attention.ops.dcp_shardq.context import dcp_fallback_max_tokens


def _config(
    *,
    tp: int,
    dcp_size: int,
    role: str | None,
    mns: int,
    k: int,
    mnbt: int,
    capture: int,
    shard_q: bool = True,
    mode: CUDAGraphMode = CUDAGraphMode.FULL_AND_PIECEWISE,
    min_tokens: int = 1024,
):
    kv = None
    if role is not None:
        kv = SimpleNamespace(
            kv_role=role,
            is_kv_consumer=role in ("kv_consumer", "kv_both"),
            is_kv_producer=role in ("kv_producer", "kv_both"),
            is_kv_transfer_instance=True,
            has_connector=lambda name: name == "NixlConnector",
        )
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            decode_context_parallel_size=dcp_size,
            prefill_context_parallel_size=1,
            use_ubatching=False,
            cp_kv_cache_interleave_size=1,
            dcp_kv_cache_interleave_size=1,
            _allow_auto_resolve_cp_interleave_size=False,
        ),
        attention_config=SimpleNamespace(
            sparse_mla_shard_q=shard_q,
            hisparse_config=None,
            sparse_mla_shard_q_min_tokens=min_tokens,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=mode, max_cudagraph_capture_size=capture
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=mnbt, max_num_seqs=mns),
        kv_transfer_config=kv,
        num_speculative_tokens=k,
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=k, parallel_drafting=False)
            if k
            else None
        ),
        use_v2_model_runner=True,
    )


def _old_bound(cfg) -> int:
    """The bound before: min(MNBT, max(mns x (1 + k), max capture size))."""
    sc = cfg.scheduler_config
    return min(
        sc.max_num_batched_tokens,
        max(
            sc.max_num_seqs * (1 + cfg.num_speculative_tokens),
            cfg.compilation_config.max_cudagraph_capture_size,
        ),
    )


# name -> (config, ShardQ, old bound, new bound)
CONFIGS = {
    # Kimi K3.1 TP8+DCP8 ShardQ P/D prefill, P6 arm A: max capture 2048.
    "p6_arm_a_prefill": (
        _config(
            tp=8,
            dcp_size=8,
            role="kv_producer",
            mns=32,
            k=7,
            mnbt=16384,
            capture=2048,
        ),
        True,
        2048,
        1279,
    ),
    # P6 arm B: captures up to the token budget.
    "p6_arm_b_prefill": (
        _config(
            tp=8,
            dcp_size=8,
            role="kv_producer",
            mns=32,
            k=7,
            mnbt=16384,
            capture=16384,
        ),
        True,
        16384,
        1279,
    ),
    # P6 arm C8 (prediction in P6mem.md section 2).
    "p6_arm_c8_prefill": (
        _config(
            tp=8,
            dcp_size=8,
            role="kv_producer",
            mns=32,
            k=7,
            mnbt=16384,
            capture=8192,
        ),
        True,
        8192,
        1279,
    ),
    # Production decode-c64.render.yaml prefill side (kv_both MultiConnector).
    "prod_c64_tp8dcp8_prefill_kv_both": (
        _config(
            tp=8,
            dcp_size=8,
            role="kv_both",
            mns=32,
            k=7,
            mnbt=16384,
            capture=2048,
        ),
        True,
        2048,
        1279,
    ),
    # Production TP16+DCP16 decode, decode-c64.render.yaml (C64 / C16).
    "prod_dcp16_decode_c64": (
        _config(
            tp=16,
            dcp_size=16,
            role="kv_consumer",
            mns=64,
            k=7,
            mnbt=8192,
            capture=512,
        ),
        False,
        512,
        512,
    ),
    # Production TP16+DCP16 decode, decode-c1.render.yaml.
    "prod_dcp16_decode_c1": (
        _config(
            tp=16,
            dcp_size=16,
            role="kv_consumer",
            mns=32,
            k=7,
            mnbt=8192,
            capture=256,
        ),
        False,
        256,
        256,
    ),
    # TP8+DCP8 decode (P/D consumer), k7 with FULL spec-verify graphs.
    "tp8dcp8_decode_k7": (
        _config(
            tp=8,
            dcp_size=8,
            role="kv_consumer",
            mns=64,
            k=7,
            mnbt=8192,
            capture=512,
        ),
        False,
        512,
        512,
    ),
    # TP8+DCP8 decode at 256 seqs per 8 GPUs, k3, MNBT = 5 x mns.
    "tp8dcp8_decode_c256_k3": (
        _config(
            tp=8,
            dcp_size=8,
            role="kv_consumer",
            mns=256,
            k=3,
            mnbt=1280,
            capture=1024,
        ),
        False,
        1024,
        1024,
    ),
    # DCP without ShardQ (flag off): unchanged.
    "tp8dcp8_prefill_no_shardq": (
        _config(
            tp=8,
            dcp_size=8,
            role="kv_producer",
            mns=32,
            k=7,
            mnbt=16384,
            capture=16384,
            shard_q=False,
        ),
        False,
        16384,
        16384,
    ),
    # FULL mixed graphs pad prefill steps on the DCP path: no ShardQ, unchanged.
    "tp8dcp8_full_mixed_graphs": (
        _config(
            tp=8,
            dcp_size=8,
            role="kv_producer",
            mns=32,
            k=7,
            mnbt=16384,
            capture=2048,
            mode=CUDAGraphMode.FULL,
        ),
        False,
        2048,
        2048,
    ),
    # Colocated (no P/D) ShardQ instance that also decodes with FULL graphs.
    "tp8dcp8_colocated_shardq": (
        _config(tp=8, dcp_size=8, role=None, mns=32, k=7, mnbt=16384, capture=2048),
        True,
        2048,
        1279,
    ),
    # Few-token rows above the token budget: never above MNBT.
    "shardq_small_budget": (
        _config(
            tp=8,
            dcp_size=8,
            role="kv_producer",
            mns=32,
            k=7,
            mnbt=1024,
            capture=1024,
        ),
        True,
        1024,
        1024,
    ),
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_workspace_rows(name):
    cfg, shardq, old, new = CONFIGS[name]
    assert dcp.dcp_shardq_configured(cfg) is shardq
    assert _old_bound(cfg) == old
    rows = dcp.get_dcp_workspace_max_num_tokens(cfg)
    assert rows == new
    sc = cfg.scheduler_config
    decode_rows = sc.max_num_seqs * (1 + cfg.num_speculative_tokens)
    # Every decode / spec-verify step, and with it every FULL decode graph
    # (at most max_num_seqs x (1 + k) rows), fits.
    assert rows >= min(sc.max_num_batched_tokens, decode_rows)
    if not shardq:
        # Decode-only and non-ShardQ instances keep their bound.
        assert rows == old
        return
    # The few-token steps the planner leaves on the DCP path fit as well, so
    # every ShardQ DCP-path bound is this one.
    decode_query_len = 1 + cfg.num_speculative_tokens
    few = sc.max_num_seqs * decode_query_len + 1024 - 1
    assert dcp.dcp_shardq_few_token_rows(cfg, decode_query_len) == few
    assert rows == min(sc.max_num_batched_tokens, max(decode_rows, few))
    assert dcp_fallback_max_tokens(cfg, decode_query_len) == rows
    assert fi.trtllm_gen_lse_max_rows(cfg, True) == rows
    assert fi.trtllm_gen_lse_slab_rows(cfg, True) == rows
    assert fi.dcp_path_max_rows(cfg, True) == rows


@pytest.mark.parametrize(
    "name",
    [
        n
        for n, (cfg, _, _, _) in CONFIGS.items()
        if cfg.kv_transfer_config is not None
        and cfg.kv_transfer_config.kv_role == "kv_consumer"
    ],
)
def test_decode_only_bounds_unchanged(name):
    """The decode-only slab and chunk rows still derive from the old bound."""
    cfg, _, old, _ = CONFIGS[name]
    assert dcp.dcp_path_rows_bound(cfg) == old
    assert fi.dcp_path_max_rows(cfg, False) == old
    assert fi.trtllm_gen_lse_slab_rows(cfg, False) == min(
        cfg.scheduler_config.max_num_batched_tokens,
        max(old, fi._TRTLLM_GEN_LSE_MIN_SLAB_ROWS),
    )


def test_p6_bytes_recovered():
    """P6mem.md section 1: 514,816 B per row of persistent DCP workspaces
    (LSE slab 256 KiB, A2A receive 192 KiB, Q gather 54 KiB, A2A LSE 768 B) and
    268,800 B per row of the profile's DCP-path claim."""
    per_row = 256 * 1024 + 192 * 1024 + 54 * 1024 + 768 + 268_800
    gib = 1 << 30
    rows = {n: (CONFIGS[n][2], CONFIGS[n][3]) for n in CONFIGS}
    a_old, a_new = rows["p6_arm_a_prefill"]
    b_old, b_new = rows["p6_arm_b_prefill"]
    assert round((b_old - b_new) * per_row / gib, 1) == 11.0
    assert round((a_old - a_new) * per_row / gib, 2) == 0.56


# The DCP manager past its direct workspaces.

_DIRECT_ROWS = 4


def _manager_config():
    config = MagicMock()
    config.parallel_config.num_ubatches = 1
    config.parallel_config.dcp_comm_backend = "a2a"
    config.scheduler_config.max_num_batched_tokens = 16
    config.scheduler_config.max_num_seqs = _DIRECT_ROWS
    config.num_speculative_tokens = 0
    config.speculative_config = None
    config.compilation_config.max_cudagraph_capture_size = 0
    return config


def _manager(monkeypatch, strict: bool):
    group = MagicMock(world_size=2)
    group.all_gather.side_effect = lambda q, dim: q
    monkeypatch.setattr(dcp, "get_dcp_group", lambda: group)
    direct_a2a = MagicMock(max_num_tokens=_DIRECT_ROWS)
    direct_a2a.lse_reduce.side_effect = lambda out, lse, *args: out * 2
    direct_query = MagicMock(max_num_tokens=_DIRECT_ROWS)
    direct_query.gather.side_effect = lambda q: q + 1
    monkeypatch.setattr(
        dcp, "get_direct_dcp_a2a_workspace", MagicMock(return_value=direct_a2a)
    )
    monkeypatch.setattr(
        dcp, "get_direct_dcp_q_gather_workspace", MagicMock(return_value=direct_query)
    )
    nccl_combine = MagicMock(side_effect=lambda out, lse, **kwargs: out * 3)
    monkeypatch.setattr(dcp, "dcp_a2a_lse_reduce", nccl_combine)
    manager = dcp.MLADCPManager(
        vllm_config=_manager_config(),
        device=torch.device("cpu"),
        num_heads=2,
        query_head_dim=8,
        output_head_dim=4,
        query_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        padded_num_heads=None,
        is_lse_base_on_e=True,
        use_pcp=False,
    )
    assert manager.max_num_tokens == _DIRECT_ROWS
    assert manager.direct_a2a is direct_a2a
    assert not manager.strict_direct_rows
    manager.strict_direct_rows = strict
    return manager, group, nccl_combine


def _combine(manager, rows: int):
    return manager.combine(
        torch.ones(rows, 4, 4),
        torch.zeros(rows, 4),
        seq_lens=torch.ones(1, dtype=torch.int32),
        query_start_loc=torch.tensor([0, rows], dtype=torch.int32),
    )


@pytest.mark.parametrize("strict", [False, True])
def test_direct_rows_fit(monkeypatch, strict):
    manager, group, nccl_combine = _manager(monkeypatch, strict)
    assert torch.equal(_combine(manager, _DIRECT_ROWS), torch.full((4, 4, 4), 2.0))
    q = torch.zeros(_DIRECT_ROWS, 2, 8)
    assert torch.equal(manager.query_gather(q), q + 1)
    nccl_combine.assert_not_called()
    group.all_gather.assert_not_called()


def test_overflow_takes_nccl_without_shardq(monkeypatch):
    """A non-ShardQ layer keeps the NCCL fallback past the direct workspaces."""
    manager, group, nccl_combine = _manager(monkeypatch, strict=False)
    assert torch.equal(_combine(manager, _DIRECT_ROWS + 1), torch.full((5, 4, 4), 3.0))
    nccl_combine.assert_called_once()
    q = torch.zeros(_DIRECT_ROWS + 1, 2, 8)
    assert torch.equal(manager.query_gather(q), q)
    group.all_gather.assert_called_once()


def test_overflow_raises_under_shardq(monkeypatch):
    """A DCP ShardQ layer's pass past the direct workspaces is a bug: no silent
    NCCL fallback."""
    manager, group, nccl_combine = _manager(monkeypatch, strict=True)
    with pytest.raises(RuntimeError, match="5-row DCP combine exceeds the 4-row"):
        _combine(manager, _DIRECT_ROWS + 1)
    with pytest.raises(RuntimeError, match="5-row DCP query gather exceeds"):
        manager.query_gather(torch.zeros(_DIRECT_ROWS + 1, 2, 8))
    nccl_combine.assert_not_called()
    group.all_gather.assert_not_called()


def test_uninitialized_manager_is_not_strict():
    assert dcp.MLADCPManager.strict_direct_rows is False
    assert object.__new__(dcp.MLADCPManager).strict_direct_rows is False


# MLAAttention: chunk rows and the ShardQ flag.


def _layer(impl_rows, manager, exchange=None):
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

    layer = object.__new__(MLAAttention)
    layer.__dict__.update(
        impl=SimpleNamespace(dcp_path_max_rows=impl_rows, dcp_world_size=8),
        dcp_manager=manager,
        use_pcp=False,
        dcp_decode_o_exchange=exchange,
    )
    return layer


def _strict_manager(rows: int, strict: bool = True):
    return SimpleNamespace(max_num_tokens=rows, strict_direct_rows=strict)


@pytest.mark.parametrize(
    "impl_rows, manager, num_mqa, cap",
    [
        # trtllm-gen ShardQ: the impl's rows are the workspace's (P6 arms).
        (1279, _strict_manager(1279), 30000, 1279),
        (1279, _strict_manager(1279), 1279, None),
        # A backend without its own bound (prims_ts) chunks at the workspace.
        (None, _strict_manager(1279), 3000, 1279),
        (None, _strict_manager(1279), 1000, None),
        # Never above the direct workspace.
        (2048, _strict_manager(1279), 2000, 1279),
        # Non-ShardQ layers: the impl's rows only (unchanged).
        (None, _strict_manager(1279, strict=False), 3000, None),
        (2048, _strict_manager(1024, strict=False), 3000, 2048),
        (None, None, 3000, None),
    ],
)
def test_row_cap(impl_rows, manager, num_mqa, cap):
    layer = _layer(impl_rows, manager)
    metadata = SimpleNamespace(num_prefills=1)
    assert layer._dcp_mqa_row_cap(metadata, None, num_mqa, 0, num_mqa + 10) == cap
    # ShardQ steps and KV-gathered prefill rows never chunk here.
    assert layer._dcp_mqa_row_cap(metadata, object(), num_mqa, 0, num_mqa) is None
    assert layer._dcp_mqa_row_cap(metadata, None, num_mqa, 5, num_mqa) is None


def test_decode_exchange_keeps_whole_step_unless_past_the_workspace():
    metadata = SimpleNamespace(num_prefills=0)
    exchange = object()
    loose = _layer(1024, _strict_manager(1024, strict=False), exchange)
    assert loose._dcp_mqa_row_cap(metadata, None, 3000, 0, 3000) is None
    strict = _layer(1279, _strict_manager(1279), exchange)
    assert strict._dcp_mqa_row_cap(metadata, None, 3000, 0, 3000) == 1279
    assert strict._dcp_mqa_row_cap(metadata, None, 1000, 0, 1000) is None


def test_row_chunks_stay_in_the_direct_workspace(monkeypatch):
    """A strict manager's chunked pass: every gather and combine fits the
    direct workspace, and the output is the single pass's (rows are
    independent)."""
    manager, group, nccl_combine = _manager(monkeypatch, strict=True)
    gathers, combines = [], []
    manager.query_gather = functools.partial(
        lambda f, q: gathers.append(q.shape[0]) or f(q), manager.query_gather
    )
    combine = manager.combine
    manager.combine = lambda out, lse, **kw: combines.append(out.shape[0]) or combine(
        out, lse, **kw
    )
    calls = []

    def forward_mqa(q, kv_cache, metadata, layer, *, row_offset, num_step_rows):
        calls.append((row_offset, q.shape[0], num_step_rows))
        return q[..., :4] * 5, torch.zeros(q.shape[0], 2)

    layer = _layer(None, manager)
    layer.impl.forward_mqa = forward_mqa
    num_rows = 3 * _DIRECT_ROWS + 1
    layer.__dict__.update(
        _dcp_combine_requests=lambda metadata, decode_rows_only: (
            torch.ones(1, dtype=torch.int32),
            torch.tensor([0, num_rows], dtype=torch.int32),
        ),
        _v_up_proj=lambda x, out: out.copy_(x),
    )
    cap = layer._dcp_mqa_row_cap(
        SimpleNamespace(num_prefills=1), None, num_rows, 0, num_rows
    )
    assert cap == _DIRECT_ROWS
    q = torch.arange(num_rows * 2 * 8, dtype=torch.float32).view(num_rows, 2, 8)
    output = torch.zeros(num_rows, 2, 4)
    layer._dcp_mqa_in_row_chunks(
        q,
        None,
        SimpleNamespace(),
        output,
        cap,
        gather_query=True,
        decode_rows_only=False,
    )
    assert gathers == combines == [4, 4, 4, 1]
    assert calls == [(0, 4, 13), (4, 4, 13), (8, 4, 13), (12, 1, 13)]
    # gather adds 1, the kernel scales by 5, the direct combine doubles.
    assert torch.equal(output, (q[..., :4] + 1) * 5 * 2)
    nccl_combine.assert_not_called()
    group.all_gather.assert_not_called()


@pytest.mark.parametrize("use_shardq", [False, True])
def test_process_weights_sets_strict_rows(use_shardq):
    """The flag follows the layer's final ShardQ decision (a model may opt the
    layer in after construction, as Kimi K3.1's DSA layer does)."""
    from vllm.model_executor.layers.attention import mla_attention

    layer = _layer(1279, _strict_manager(1279, strict=False))
    layer.__dict__.update(
        impl=MagicMock(),
        use_shardq=use_shardq,
        is_amx_bmm_enabled=True,
        kv_b_proj=SimpleNamespace(weight=torch.empty(2)),
    )
    with patch.object(mla_attention, "replace_parameter"):
        layer.process_weights_after_loading(torch.bfloat16)
    assert layer.dcp_manager.strict_direct_rows is use_shardq
