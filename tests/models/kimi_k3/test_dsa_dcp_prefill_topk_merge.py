# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 histogram-exchange DCP prefill top-k merge against the generic
merge's selector.

Every DCP rank is simulated on one GPU: rank ``r`` owns global token ``g`` iff
``g % W == r`` (interleave 1). Each round's all-gather is replaced by stacking
every rank's send rows.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia.dsa_kernels import dcp_prefill_topk_merge
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

needs_cutedsl = pytest.mark.skipif(
    not current_platform.is_cuda() or not has_cutedsl(),
    reason="The DCP prefill top-k merge needs CUDA and CuTe DSL",
)


class _FakeDCPGroup:
    """``all_gather(dim=0)`` replays the recorded per-round concatenations."""

    def __init__(self, rank: int, rounds: list[torch.Tensor]) -> None:
        self.rank_in_group = rank
        self.world_size = rounds[0].shape[0]
        self.rounds = iter(rounds)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == 0
        gathered = next(self.rounds).view(input_.dtype)
        assert gathered.shape[1:] == input_.shape
        return gathered.reshape(-1, *input_.shape[1:]).clone()


def _make_rank_inputs(
    world: int, topk: int, rows: int, max_local: int, seed: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-rank bf16 logits with heavy score ties (and negatives) and each
    rank's local top-K columns in random order. Row ``i`` holds a causal-like
    share of the local tokens, so some rows have fewer than ``topk``
    candidates in total and one rank holds no tokens at all."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits, topks = [], []
    for rank in range(world):
        if rank == world - 1:
            # No local tokens in this chunk: empty FP32 logits as the op
            # allocates them, every candidate -1.
            logits.append(torch.empty((rows, 0), device="cuda"))
            topks.append(torch.full((rows, topk), -1, dtype=torch.int32, device="cuda"))
            continue
        width = max_local + 7 * rank
        lens = torch.linspace(0, width, rows, device="cuda").long()
        lens[rows // 2] = width
        # Few distinct values -> long runs of tied scores across ranks.
        values = torch.randint(-12, 12, (rows, width), generator=gen, device="cuda")
        row_logits = (values.float() * 0.25).to(torch.bfloat16)
        cols = torch.arange(width, device="cuda")
        masked = row_logits.float().masked_fill(cols >= lens[:, None], float("-inf"))
        # Any local top-K is a valid input; break local ties randomly.
        noise = torch.rand((rows, width), generator=gen, device="cuda") * 1e-3
        order = (masked + noise).argsort(dim=1, descending=True)[:, :topk]
        valid = torch.gather(masked, 1, order) > float("-inf")
        local = torch.full((rows, topk), -1, dtype=torch.int32, device="cuda")
        local[:, : order.shape[1]] = torch.where(valid, order, -1)
        perm = torch.randperm(topk, generator=gen, device="cuda")
        logits.append(row_logits)
        topks.append(local[:, perm].contiguous())
    return logits, topks


def _reference(logits, topks, topk: int) -> list[torch.Tensor]:
    """Each rank's share of the generic merge: the pack + CuTe stable top-k
    selector over every rank's (score, global id) candidates."""
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )

    world = len(topks)
    packed = []
    for rank, (rank_logits, local) in enumerate(zip(logits, topks)):
        pairs = torch.empty((*local.shape, 2), device="cuda")
        if rank_logits.shape[1] == 0:
            pairs[..., 0] = float("-inf")
            pairs[..., 1] = -1
        else:
            pack_dcp_topk_candidates_cutedsl(
                rank_logits, local, pairs, rank, world, 1, None
            )
        packed.append(pairs)
    merged = stable_topk_from_gathered_candidates_cutedsl(
        torch.cat(packed, dim=1).contiguous(), topk
    )
    return [torch.where(merged % world == rank, merged, -1) for rank in range(world)]


def _simulate(logits, topks, tie_words: int):
    """Run the five stages for every rank with stacked all-gathers. Returns
    each rank's output and the gathered tensors of each round."""
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_prefill_topk_merge.hist_merge import (  # noqa: E501
        _DCP_PREFILL_MERGE_KERNEL as kernel,
    )

    world = len(topks)
    rows, topk = topks[0].shape
    outs = [local.clone() for local in topks]
    keys = [torch.empty_like(local) for local in topks]
    meta = [torch.empty((rows, 8), dtype=torch.int32, device="cuda") for _ in topks]
    hist = [torch.empty((rows, 256), dtype=torch.int16, device="cuda") for _ in topks]
    bits = [
        torch.empty((rows, tie_words), dtype=torch.int32, device="cuda") for _ in topks
    ]
    for r in range(world):
        rank_logits = logits[r]
        if rank_logits.shape[1] == 0:
            rank_logits = torch.empty((rows, 0), dtype=torch.bfloat16, device="cuda")
        kernel.keys(
            rank_logits,
            outs[r],
            keys[r],
            hist[r],
            world_size=world,
            tie_words=tie_words,
        )
    rounds = [torch.stack(hist)]
    for r in range(world):
        kernel.refine(rounds[-1], keys[r], meta[r], hist[r], tie_words=tie_words)
    rounds.append(torch.stack(hist))
    for r in range(world):
        kernel.tie_hist(
            rounds[-1], meta[r], keys[r], outs[r], hist[r], tie_words=tie_words
        )
    rounds.append(torch.stack(hist))
    for r in range(world):
        kernel.tie_bits(rounds[-1], meta[r], keys[r], outs[r], bits[r])
    rounds.append(torch.stack(bits))
    for r in range(world):
        kernel.select(rounds[-1], meta[r], keys[r], outs[r], r)
    return outs, rounds


def _assert_rank_share(out: torch.Tensor, expected: torch.Tensor) -> None:
    """``out`` holds the expected ids as a prefix (any order), then -1."""
    counts = (expected >= 0).sum(dim=1)
    assert torch.equal((out >= 0).sum(dim=1), counts)
    cols = torch.arange(out.shape[1], device=out.device)
    assert torch.equal(out >= 0, cols[None, :] < counts[:, None])
    assert torch.equal(out.sort(dim=1).values, expected.sort(dim=1).values)


@needs_cutedsl
@pytest.mark.parametrize(
    "world,topk,max_local,tie_words",
    [(4, 512, 700, 1), (8, 2048, 4000, 1), (8, 2048, 9000, 2), (2, 1024, 600, 16)],
)
def test_hist_merge_matches_generic_merge(world, topk, max_local, tie_words):
    """Every rank keeps exactly its share of the generic merge's global top-K
    (bf16 score desc, global id asc), ties included."""
    assert max_local + 7 * world <= 256 * 32 * tie_words
    logits, topks = _make_rank_inputs(world, topk, 96, max_local, seed=world + topk)
    expected = _reference(logits, topks, topk)
    outs, _ = _simulate(logits, topks, tie_words)
    for out, rank_expected in zip(outs, expected):
        _assert_rank_share(out, rank_expected)


def _boundary_inputs(world: int, topk: int, case: str):
    """Every rank holds ``topk`` candidates (locals ``0..topk-1``) and the
    global ``topk``-th candidate ends a histogram bin exactly, with the next
    bin non-empty:

    - ``high_byte``: ``counts[r]`` candidates of rank ``r`` score 2.0 (they
      sum to ``topk``), the rest 1.0, a different key high byte;
    - ``low_byte``: 2.5 over 2.0, the same high byte;
    - ``tie_bin``: every score ties, so the kept ties are locals below
      ``topk / world`` on every rank: whole tie-index bins.
    """
    gen = torch.Generator(device="cuda").manual_seed(world)
    counts = [300, 212, 0, 512, 256, 256, 256, 256][:world]
    counts[-1] += topk - sum(counts)
    hi, lo = {"high_byte": (2.0, 1.0), "low_byte": (2.5, 2.0)}.get(case, (2.0, 2.0))
    logits, topks = [], []
    for rank in range(world):
        row = torch.full((topk,), lo, device="cuda")
        top = torch.randperm(topk, generator=gen, device="cuda")[: counts[rank]]
        row[top] = hi
        logits.append(row.to(torch.bfloat16)[None].repeat(2, 1))
        local = torch.randperm(topk, generator=gen, device="cuda").to(torch.int32)
        topks.append(local[None].repeat(2, 1))
    return logits, topks


@needs_cutedsl
@pytest.mark.parametrize("case", ["high_byte", "low_byte", "tie_bin"])
@pytest.mark.parametrize("world,topk", [(2, 512), (8, 2048)])
def test_hist_merge_boundary_on_a_bin_end(world: int, topk: int, case: str):
    """The running count reaches K exactly at the end of a histogram bin: only
    that bin may claim the boundary, not the next one."""
    logits, topks = _boundary_inputs(world, topk, case)
    expected = _reference(logits, topks, topk)
    outs, _ = _simulate(logits, topks, tie_words=1)
    for rank, (out, rank_expected) in enumerate(zip(outs, expected)):
        _assert_rank_share(out, rank_expected)
        if case == "tie_bin":
            kept = torch.arange(topk // world, device="cuda") * world + rank
            assert torch.equal(rank_expected[0].sort().values[-kept.numel() :], kept)


@needs_cutedsl
def test_hist_merge_driver_matches_stages():
    """``dcp_prefill_topk_merge`` (in place, through ``all_gather``) gives the
    staged result on every rank, the empty-shard rank included."""
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_prefill_topk_merge.hist_merge import (  # noqa: E501
        dcp_prefill_topk_merge,
    )

    world, topk, tie_words = 4, 512, 1
    logits, topks = _make_rank_inputs(world, topk, 40, 900, seed=3)
    outs, rounds = _simulate(logits, topks, tie_words)
    for rank in range(world):
        topk_indices = topks[rank].clone()
        group = _FakeDCPGroup(rank, rounds)
        dcp_prefill_topk_merge(logits[rank], topk_indices, tie_words, group=group)
        assert torch.equal(topk_indices, outs[rank])


@needs_cutedsl
def test_hist_merge_warmup_keys_cover_every_stage():
    from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_prefill_topk_merge.hist_merge import (  # noqa: E501
        DCPPrefillMergeKernel,
    )

    keys = DCPPrefillMergeKernel().get_warmup_keys(
        topk=2048, world_size=8, tie_words=16
    )
    assert sorted(key.stage for key in keys) == sorted(
        ("keys", "refine", "tie_hist", "tie_bits", "select")
    )


@pytest.mark.parametrize(
    "tie_words,world,interleave,has_row_starts,expected",
    [
        (16, 8, 1, False, True),
        (0, 8, 1, False, False),
        # A rank that skips the DCP merge (world 1) must not enter the
        # exchange's collectives.
        (16, 1, 1, False, False),
        (16, 8, 2, False, False),
        (16, 8, 64, False, False),
        (16, 8, 1, True, False),
    ],
)
def test_prefill_hist_merge_dispatch(
    tie_words, world, interleave, has_row_starts, expected
):
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _uses_dcp_prefill_hist_merge,
    )

    row_starts = torch.zeros(4, dtype=torch.int32) if has_row_starts else None
    assert (
        _uses_dcp_prefill_hist_merge(tie_words, world, interleave, row_starts)
        is expected
    )


def test_tie_words_cover_the_longest_local_row(monkeypatch):
    platform = SimpleNamespace(
        is_cuda=lambda: True, is_device_capability_family=lambda family: True
    )
    monkeypatch.setattr(dcp_prefill_topk_merge, "current_platform", platform)
    monkeypatch.setattr(dcp_prefill_topk_merge, "has_cutedsl", lambda: True)
    monkeypatch.setenv("VLLM_DSA_DCP_PREFILL_HIST_MERGE", "1")

    def config(dcp: int, max_model_len: int, pcp: int = 1, interleave: int = 1):
        return SimpleNamespace(
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=dcp,
                prefill_context_parallel_size=pcp,
                cp_kv_cache_interleave_size=interleave,
            ),
            model_config=SimpleNamespace(max_model_len=max_model_len),
        )

    tie_words = dcp_prefill_topk_merge.dcp_prefill_merge_tie_words
    # Interleave 1 only: the exchange decodes ids as local * W + rank.
    assert tie_words(config(8, 1 << 20, interleave=1024), 2048) == 0
    # 1M tokens at DCP8: 131072 local tokens = 256 bins x 512 positions.
    assert tie_words(config(8, 1 << 20), 2048) == 16
    assert tie_words(config(8, (1 << 20) + 1), 2048) == 17
    assert tie_words(config(8, 4096), 2048) == 1
    assert tie_words(config(1, 1 << 20), 2048) == 0
    assert tie_words(config(8, 1 << 20, pcp=2), 2048) == 0
    assert tie_words(config(8, 1 << 20), 2000) == 0
    # One thread clears each bitmap word: at most 256 words (2M local tokens).
    assert tie_words(config(8, 1 << 24), 2048) == 256
    assert tie_words(config(8, (1 << 24) + 8 * 8192), 2048) == 0


def test_hist_merge_is_off_by_default(monkeypatch):
    """The histogram merge is a fallback for steps that do not take DCP
    ShardQ: without VLLM_DSA_DCP_PREFILL_HIST_MERGE the indexer keeps the
    generic merge (and registers no compiles)."""
    platform = SimpleNamespace(
        is_cuda=lambda: True, is_device_capability_family=lambda family: True
    )
    monkeypatch.setattr(dcp_prefill_topk_merge, "current_platform", platform)
    monkeypatch.setattr(dcp_prefill_topk_merge, "has_cutedsl", lambda: True)
    monkeypatch.delenv("VLLM_DSA_DCP_PREFILL_HIST_MERGE", raising=False)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=8,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        model_config=SimpleNamespace(max_model_len=1 << 20),
    )
    assert dcp_prefill_topk_merge.dcp_prefill_merge_tie_words(config, 2048) == 0
    monkeypatch.setenv("VLLM_DSA_DCP_PREFILL_HIST_MERGE", "1")
    assert dcp_prefill_topk_merge.dcp_prefill_merge_tie_words(config, 2048) == 16
