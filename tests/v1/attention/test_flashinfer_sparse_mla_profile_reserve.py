# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests: the trtllm-gen sparse MLA path's memory-profile reservation.

The profile run skips attention, and the metadata builders that used to allocate
trtllm-gen's workspace are created with the KV cache, after the profile. The
layers' profile pass (FlashInferMLASparseImpl.reserve_workspace) therefore
allocates the workspace and KV counters at their final size and claims one full
step's attention working set.

Under DCP ShardQ the DCP softmax-stats slab covers the DCP path's
dcp_fallback_max_tokens rows; dense-MHA steps, whose MQA rows can exceed them,
launch in slab-sized row chunks.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fi

# Kimi K3.1 C512 prefill instance: 96 heads, TP4 = DCP4 (24 heads per rank),
# ShardQ, max_num_batched_tokens 32768, max_num_seqs 64, top-k 2048, fp8 cache.
_HEADS = 24
_TP = 4
_MNBT = 32768
_MNS = 64
_TOPK = 2048
_DCP_FALLBACK_TOKENS = 2048


def _impl(use_shardq: bool, dcp: int) -> fi.FlashInferMLASparseImpl:
    impl = object.__new__(fi.FlashInferMLASparseImpl)
    impl.__dict__.update(
        use_prims_ts=False,
        use_shardq=use_shardq,
        num_heads=_HEADS,
        dcp_world_size=dcp,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        kv_cache_dtype="fp8",
        _topk_indices_buffer=SimpleNamespace(shape=(_MNBT, _TOPK)),
    )
    return impl


def _config():
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=_MNBT, max_num_seqs=_MNS
        ),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )


def _reserve(impl):
    buffers, claims = [], []
    with (
        patch.object(fi, "get_current_vllm_config", return_value=_config()),
        patch.object(fi, "get_tensor_model_parallel_world_size", return_value=_TP),
        patch.object(
            fi,
            "reserve_trtllm_gen_buffers",
            side_effect=lambda cfg, dev, dcp, dcp_shardq: buffers.append(
                (dcp, dcp_shardq)
            ),
        ),
        patch.object(fi.torch.accelerator, "current_device_index", return_value=0),
        patch(
            "vllm.v1.attention.ops.dcp_shardq.context.dcp_fallback_max_tokens",
            return_value=_DCP_FALLBACK_TOKENS,
        ),
        patch(
            "vllm.v1.attention.backends.mla.indexer."
            "DeepseekV32IndexerMetadataBuilder.get_decode_threshold",
            return_value=4,
        ),
        patch.object(fi.torch, "empty", side_effect=lambda n, **_: claims.append(n)),
    ):
        impl.reserve_workspace()
    return buffers, claims


def test_shardq_dcp_reserves_workspace_and_step_working_set():
    buffers, claims = _reserve(_impl(use_shardq=True, dcp=4))
    # The workspace (with the DCP ShardQ softmax-stats slab) and the KV counters.
    assert buffers == [(4, True)]
    # Layer query: bf16 (T, 24, 576) and its FP8 quantization.
    layer_query = _MNBT * _HEADS * 576 * 3
    # ShardQ slice: cdiv(T, 4) + 2 * 64 rows with 96 heads padded to 128:
    # bf16 output and int32 physical top-k.
    shardq = (_MNBT // _TP + 2 * _MNS) * (128 * 512 * 2 + _TOPK * 4)
    # DCP-path step (at most the fallback tokens): output, top-k, the
    # gathered (96 heads) and padded (128) fp8 query, and the fp32 LSE.
    dcp = _DCP_FALLBACK_TOKENS * (
        128 * 512 * 2 + _TOPK * 4 + (96 + 128) * 576 + 128 * 4
    )
    assert shardq > dcp
    assert claims == [layer_query + shardq]
    # 2.34 GiB: the measured n32k attention working set was 2.33 GiB
    # (q 0.844 + fp8 q 0.422 + output 1.0 + physical top-k 0.062).
    assert abs(claims[0] / 2**30 - 2.345) < 0.01


def test_dcp_without_shardq_claims_full_step_with_lse():
    buffers, claims = _reserve(_impl(use_shardq=False, dcp=4))
    assert buffers == [(4, False)]
    layer_query = _MNBT * _HEADS * 576 * 3
    dcp = _MNBT * (128 * 512 * 2 + _TOPK * 4 + (96 + 128) * 576 + 128 * 4)
    assert claims == [layer_query + dcp]


def _buffers_config():
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=_MNBT),
        model_config=SimpleNamespace(get_num_attention_heads=lambda pc: _HEADS),
        parallel_config=None,
    )


def _patch_fallback_rows():
    return (
        patch(
            "vllm.v1.attention.ops.dcp_shardq.context.dcp_fallback_max_tokens",
            return_value=_DCP_FALLBACK_TOKENS,
        ),
        patch(
            "vllm.v1.attention.backends.mla.indexer."
            "DeepseekV32IndexerMetadataBuilder.get_decode_threshold",
            return_value=4,
        ),
    )


def _allocate(monkeypatch, calls):
    """Run reserve_trtllm_gen_buffers for each (dcp, dcp_shardq) in ``calls``
    on a fresh workspace; returns the workspace allocations in bytes."""
    monkeypatch.delenv("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", raising=False)
    allocations, counters = [], []

    def zeros(n, **_):
        allocations.append(n)
        return SimpleNamespace(numel=lambda: n)

    monkeypatch.setattr(fi, "_fi_sparse_workspace", None)
    fallback, threshold = _patch_fallback_rows()
    with (
        fallback,
        threshold,
        patch.object(fi.torch, "zeros", side_effect=zeros),
        patch.object(
            fi, "_reserve_kv_counters", side_effect=lambda *a: counters.append(a[1:])
        ),
    ):
        for dcp, dcp_shardq in calls:
            fi.reserve_trtllm_gen_buffers(
                _buffers_config(), torch.device("cpu"), dcp, dcp_shardq
            )
    monkeypatch.setattr(fi, "_fi_sparse_workspace", None)
    # The KV counters cover every launch, ShardQ slices included.
    assert counters == [(_MNBT, 128)] * len(calls)
    return allocations


def test_reserve_trtllm_gen_buffers_is_allocated_once(monkeypatch):
    """The profile pass sizes the workspace; the builders' later call reuses it."""
    allocations = _allocate(monkeypatch, [(4, False), (4, False)])
    required = fi._required_workspace_bytes(4, _HEADS, _MNBT)
    assert allocations == [required]
    # 8 B x 128 padded heads x 32768 rows x 256 slots + 1 MiB + 394 MiB.
    assert required == 8 * 128 * _MNBT * 256 + (1 + 394) * 2**20


def test_dcp_shardq_slab_covers_fallback_rows(monkeypatch):
    """Under DCP ShardQ the slab covers the DCP path's dcp_fallback_max_tokens
    rows: 0.89 GiB instead of 8.39 GiB at Kimi K3.1 C512 prefill."""
    allocations = _allocate(monkeypatch, [(4, True), (4, True)])
    required = fi._required_workspace_bytes(4, _HEADS, _DCP_FALLBACK_TOKENS)
    assert allocations == [required]
    assert required == 8 * 128 * _DCP_FALLBACK_TOKENS * 256 + (1 + 394) * 2**20
    full = fi._required_workspace_bytes(4, _HEADS, _MNBT)
    assert round(required / 2**30, 2) == 0.89
    assert round((full - required) / 2**30, 2) == 7.5


def test_builder_without_shardq_grows_the_profiled_slab(monkeypatch):
    """A builder that keeps whole prefill steps on the DCP path (e.g. no packed
    cache layout for DCP ShardQ) needs the full slab: it grows (with a warning)."""
    allocations = _allocate(monkeypatch, [(4, True), (4, False)])
    assert allocations == [
        fi._required_workspace_bytes(4, _HEADS, _DCP_FALLBACK_TOKENS),
        fi._required_workspace_bytes(4, _HEADS, _MNBT),
    ]


def test_lse_max_rows():
    def cfg(max_tokens):
        return SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens)
        )

    fallback, threshold = _patch_fallback_rows()
    with fallback, threshold:
        assert fi.trtllm_gen_lse_max_rows(cfg(_MNBT), dcp_shardq=False) == _MNBT
        assert (
            fi.trtllm_gen_lse_max_rows(cfg(_MNBT), dcp_shardq=True)
            == _DCP_FALLBACK_TOKENS
        )
    fallback, threshold = _patch_fallback_rows()
    with fallback, threshold:
        assert fi.trtllm_gen_lse_max_rows(cfg(512), dcp_shardq=True) == 512


@pytest.mark.parametrize("capture_size, rows", [(2048, 2048), (0, 1279)])
def test_dcp_fallback_rows_at_c512_prefill(capture_size, rows):
    """dcp_fallback_max_tokens (unpatched) at the Kimi K3.1 C512 prefill config
    (max_num_seqs 64, 3 speculative tokens, sparse_mla_shard_q_min_tokens 1024,
    max_cudagraph_capture_size 2048): the larger of 64 x 4 + 1024 - 1 = 1279
    few-token rows and the 2048-token padded (FULL decode graph) steps."""
    from vllm.v1.attention.ops.dcp_shardq.context import dcp_fallback_max_tokens

    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=_MNBT, max_num_seqs=_MNS
        ),
        attention_config=SimpleNamespace(sparse_mla_shard_q_min_tokens=1024),
        num_speculative_tokens=3,
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=capture_size),
    )
    with patch("vllm.v1.attention.ops.dcp.max_decode_query_len", return_value=4):
        assert dcp_fallback_max_tokens(cfg, 4) == rows


@pytest.mark.parametrize("rows", [1, 255, 2048, 2303, 32768])
def test_lse_rows_capacity_inverts_the_workspace_size(rows):
    required = fi._required_workspace_bytes(4, _HEADS, rows)
    assert fi.trtllm_gen_lse_rows_capacity(required, 128) == rows
    assert fi.trtllm_gen_lse_rows_capacity(required - 1, 128) == rows - 1
    # No slab at all (no DCP): no LSE row fits.
    base = fi._DEFAULT_WORKSPACE_BUFFER_SIZE
    assert fi.trtllm_gen_lse_rows_capacity(base, 128) == 0


class _Workspace:
    def __init__(self, nbytes: int):
        self.nbytes = nbytes

    def numel(self) -> int:
        return self.nbytes


def _mqa(rows: int, slab_rows: int, return_lse: bool | None):
    """Runs FlashInferMLASparseImpl._forward_mqa_kernel on CPU with the
    trtllm-gen launch faked, for ``rows`` query rows of 32 heads x DCP4 (128
    gathered heads, no padding) and a workspace whose slab holds ``slab_rows``
    rows; returns the launches."""
    import flashinfer.decode

    impl = object.__new__(fi.FlashInferMLASparseImpl)
    impl.__dict__.update(
        use_prims_ts=False,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        is_nope_mla=False,
        need_to_return_lse_for_decode=True,
        index_group=None,
        _workspace_buffer=_Workspace(fi._required_workspace_bytes(4, 32, slab_rows)),
        bmm1_scale=1.0,
        bmm2_scale=1.0,
    )
    launches = []
    q = torch.zeros(rows, 128, 576, dtype=torch.bfloat16)
    kv_cache = torch.zeros(4, 64, 576, dtype=torch.bfloat16)
    topk = torch.zeros(rows, _TOPK, dtype=torch.int32)
    seq_lens = torch.ones(rows, dtype=torch.int32)
    with (
        patch.object(fi, "_fi_sparse_kv_counters", None),
        patch.object(fi, "_finish_dcp_partials"),
        patch.object(
            flashinfer.decode,
            "trtllm_batch_decode_with_kv_cache_mla",
            side_effect=lambda **kw: launches.append(kw),
        ),
    ):
        impl._forward_mqa_kernel(
            q,
            kv_cache,
            topk,
            seq_lens,
            layer=None,
            block_size=64,
            is_decode=True,
            return_lse=return_lse,
        )
    return launches


def test_lse_launch_within_slab_runs():
    launches = _mqa(rows=16, slab_rows=16, return_lse=None)
    assert len(launches) == 1 and launches[0]["return_lse"]
    assert launches[0]["lse"].shape == (16, 128)


def test_lse_launch_beyond_slab_runs_in_slab_chunks():
    """A dense-MHA step's MQA rows can exceed the slab: they launch in
    slab-sized row chunks, each writing its rows of the output and LSE."""
    launches = _mqa(rows=40, slab_rows=16, return_lse=None)
    assert [len(kw["query"]) for kw in launches] == [16, 16, 8]
    assert all(kw["return_lse"] for kw in launches)
    out, lse = launches[0]["out"], launches[0]["lse"]
    for i, kw in enumerate(launches):
        assert kw["lse"].shape == (len(kw["query"]), 128)
        assert kw["lse"].data_ptr() == lse.data_ptr() + i * 16 * lse.stride(0) * 4
        assert kw["out"].data_ptr() == out.data_ptr() + i * 16 * out.stride(0) * 2
        assert kw["seq_lens"].shape == (len(kw["query"]),)
        assert kw["block_tables"].shape[0] == len(kw["query"])


def test_lse_launch_without_slab_raises():
    """A workspace without a softmax-stats slab (not sized for DCP) cannot
    run any LSE launch."""
    with pytest.raises(RuntimeError, match="no softmax-stats slab"):
        _mqa(rows=1, slab_rows=0, return_lse=None)


def test_launch_without_lse_ignores_slab():
    """ShardQ token slices (return_lse=False) use no slab, at any row count."""
    launches = _mqa(rows=17, slab_rows=16, return_lse=False)
    assert len(launches) == 1 and not launches[0]["return_lse"]
