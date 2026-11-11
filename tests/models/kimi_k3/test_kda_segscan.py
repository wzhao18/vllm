# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Segment-parallel FlashKDA prefill (ops/kda_segscan.py).

The policy tests are host-only: the head-aware length threshold, the segment
split, make_plan's go / no-go at the served shapes, and the narrow V-split
dispatch with a mocked SM count. The GPU tests compare the segmented prefill
and the narrow V-split launch against plain FlashKDA: the V-split launch is
bitwise stock (outputs, final and checkpoint states, fp32 states rounded through
bf16 as stock); the segmented prefill is bitwise on its first segment, and
bitwise overall when the gate forgets the state within a segment.
"""

import pytest
import torch

from vllm.models.kimi_k3.nvidia.ops import kda_segscan
from vllm.platforms import current_platform

RUBIN_SMS = 212


def _cu(*lens: int) -> torch.Tensor:
    return torch.tensor([0, *lens], dtype=torch.int32).cumsum(0, dtype=torch.int32)


@pytest.fixture(autouse=True)
def _fresh_plan_cache():
    kda_segscan._PLANS.clear()
    yield
    kda_segscan._PLANS.clear()


# --- policy (CPU) ------------------------------------------------------------


@pytest.mark.parametrize(
    ("heads", "tokens"),
    [(1, 3072), (8, 3072), (12, 3072), (13, 5120), (24, 5120), (25, 6144)],
)
def test_min_tokens_by_heads(heads: int, tokens: int):
    assert kda_segscan._min_tokens(heads) == tokens


def test_segment_counts_threshold_is_head_aware():
    # 4096 tokens: past the 12-head threshold, short of the 24-head one.
    assert kda_segscan._segment_counts((4096,), 12) == [8]
    assert kda_segscan._segment_counts((4096,), 24) is None
    assert kda_segscan._segment_counts((3071,), 12) is None
    assert kda_segscan._segment_counts((5120,), 24) == [8]


def test_segment_counts_shapes():
    assert kda_segscan._segment_counts((), 12) is None
    # as many sequences as pseudo-sequences: nothing left to split
    assert kda_segscan._segment_counts((8192,) * 8, 12) is None
    assert kda_segscan._segment_counts((16384,), 12) == [8]
    assert kda_segscan._segment_counts((8192, 8192), 12) == [4, 4]
    # extra segments go to the longest current segment; short ones stay whole
    assert kda_segscan._segment_counts((12288, 3072, 1400), 12) == [5, 2, 1]


def test_make_plan_2x8192_h12_declines():
    # 4 x 2048 segments per sequence: the chain shrinks only 4x (< 5x).
    assert kda_segscan.make_plan(_cu(8192, 8192), 12, "cpu") is None


def test_make_plan_1x16384_h12_eight_pieces():
    plan = kda_segscan.make_plan(_cu(16384), 12, "cpu")
    assert plan is not None
    assert plan.num_seqs == 1 and plan.num_pseudo == 8
    assert plan.cu.tolist() == list(range(0, 16384 + 1, 2048))
    assert plan.first.tolist() == [0] and plan.count.tolist() == [8]
    assert plan.last.tolist() == [7]
    assert plan.ck_x is None and plan.ck_f is None and plan.ck_k is None
    assert plan.kernel_ckpt_offsets is None


def test_make_plan_c512_5x3480_h24_declines():
    assert kda_segscan.make_plan(_cu(*([3480] * 5)), 24, "cpu") is None


def test_make_plan_checkpoint_offset_is_a_boundary():
    plan = kda_segscan.make_plan(_cu(16384), 12, "cpu", [4096])
    assert plan is not None and plan.num_pseudo == 8
    cu = plan.cu.tolist()
    assert cu == [0, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384]
    assert plan.ck_x is not None
    assert plan.ck_x[0].tolist() == [0]
    assert plan.ck_x[1].tolist() == [cu.index(4096)]
    assert plan.ck_k is None and plan.kernel_ckpt_offsets is None


def test_make_plan_is_cached():
    a = kda_segscan.make_plan(_cu(16384), 12, "cpu")
    assert kda_segscan.make_plan(_cu(16384), 12, "cpu") is a


@pytest.mark.parametrize(
    ("heads", "sequences", "fits"),
    [
        (12, 2, True),  # C256 TP8: 12 x 8 x 2 = 192 <= 212
        (12, 1, True),
        (24, 1, True),  # 192
        (12, 3, False),  # 288
        (24, 5, False),  # C512: 24 x 8 x 5 = 960
    ],
)
def test_narrow_vsplit_fits(monkeypatch, heads: int, sequences: int, fits: bool):
    monkeypatch.setattr(kda_segscan, "_num_sms", lambda index: RUBIN_SMS)
    device = torch.device("cuda", 0)
    assert kda_segscan.narrow_vsplit_fits(heads, sequences, 128, device) is fits


# --- numerics (GPU) ----------------------------------------------------------


def _split_build_supported() -> bool:
    # ops/flashkda_split builds for sm_100f only.
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ):
        return False
    try:
        import flashinfer  # noqa: F401
        import vllm._flashkda_C  # noqa: F401
    except ImportError:
        return False
    return True


requires_split_build = pytest.mark.skipif(
    not _split_build_supported(),
    reason="needs CUDA SM10x, FlashKDA (vllm._flashkda_C) and FlashInfer's CUTLASS",
)


def _inputs(
    lens: list[int],
    heads: int,
    state_dtype: torch.dtype,
    seed: int = 7,
    slow_decay: bool = False,
):
    """Random KDA inputs. ``slow_decay`` makes the gate ~1 (g = -60) with unit
    q / k, so the initial state survives the whole sequence and its rounding at
    the kernel boundary shows in the outputs and the final state."""
    torch.manual_seed(seed)
    T, H, D = sum(lens), heads, 128
    dev = "cuda"
    q, k, v, g = (
        torch.randn(1, T, H, D, dtype=torch.bfloat16, device=dev) for _ in range(4)
    )
    if slow_decay:
        q = torch.nn.functional.normalize(q.float(), dim=-1).to(torch.bfloat16)
        k = torch.nn.functional.normalize(k.float(), dim=-1).to(torch.bfloat16)
        g = torch.full_like(g, -60.0)
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=torch.randn(1, T, H, dtype=torch.bfloat16, device=dev),
        A_log=torch.randn(H, dtype=torch.float32, device=dev) * 0.5,
        dt_bias=torch.randn(H, D, dtype=torch.float32, device=dev) * 0.1,
        lower_bound=-3.0,
        initial_state=(
            torch.randn(len(lens), H, D, D, dtype=torch.float32, device=dev) * 0.1
        ).to(state_dtype),
        cu_seqlens=_cu(*lens).to(dev),
    )


def _workspace(tokens: int, heads: int, sequences: int) -> torch.Tensor:
    return torch.empty(
        torch.ops._flashkda_C.get_workspace_size(tokens, heads, sequences),
        dtype=torch.uint8,
        device="cuda",
    )


def _plain_flashkda(x: dict, checkpoint_offsets: torch.Tensor | None = None):
    """Stock FlashKDA build (vllm._flashkda_C), one launch."""
    _, T, H, D = x["q"].shape
    N = x["initial_state"].shape[0]
    out = torch.empty_like(x["v"])
    final = torch.empty_like(x["initial_state"])
    ckpt = torch.empty_like(final) if checkpoint_offsets is not None else None
    torch.ops._flashkda_C.fwd(
        x["q"].contiguous(),
        x["k"].contiguous(),
        x["v"].contiguous(),
        x["g"].contiguous(),
        x["beta"],
        D**-0.5,
        out,
        _workspace(T, H, N),
        x["A_log"].contiguous(),
        x["dt_bias"].view(-1, D).contiguous(),
        x["lower_bound"],
        x["initial_state"].contiguous(),
        final,
        x["cu_seqlens"],
        ckpt,
        checkpoint_offsets,
    )
    return out, final, ckpt


def _split_flashkda(
    x: dict, mode: int, checkpoint_offsets: torch.Tensor | None = None
):
    """The in-tree split build (ops/flashkda_split), one K1 + K2 launch."""
    kda_segscan.load_extension()
    _, T, H, D = x["q"].shape
    N = x["initial_state"].shape[0]
    out = torch.empty_like(x["v"])
    final = torch.empty_like(x["initial_state"])
    ckpt = torch.empty_like(final) if checkpoint_offsets is not None else None
    torch.ops.k3_flashkda_split.fwd(
        x["q"].contiguous(),
        x["k"].contiguous(),
        x["v"].contiguous(),
        x["g"].contiguous(),
        x["beta"],
        D**-0.5,
        out,
        _workspace(T, H, N),
        x["A_log"].contiguous(),
        x["dt_bias"].view(-1, D).contiguous(),
        x["lower_bound"],
        x["initial_state"].contiguous(),
        final,
        x["cu_seqlens"],
        ckpt,
        checkpoint_offsets,
        mode,
    )
    torch.accelerator.synchronize()
    return out, final, ckpt


def _close(name: str, ref: torch.Tensor, got: torch.Tensor, tol: float) -> None:
    ref, got = ref.float(), got.float()
    err = (ref - got).abs().max().item()
    scale = ref.abs().max().item()
    assert torch.isfinite(got).all(), name
    assert err <= tol * max(scale, 1.0), f"{name}: max err {err} (scale {scale})"


def _bf16_representable(state: torch.Tensor) -> bool:
    return torch.equal(state.to(torch.bfloat16).to(state.dtype), state)


@requires_split_build
@pytest.mark.parametrize("slow_decay", [False, True], ids=["decay", "slow_decay"])
@pytest.mark.parametrize(
    "state_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
)
@pytest.mark.parametrize("checkpoint", [None, 1024], ids=["no_ckpt", "ckpt1024"])
@torch.inference_mode()
def test_segmented_prefill_matches_flashkda(
    state_dtype: torch.dtype, checkpoint: int | None, slow_decay: bool
):
    # 3072 tokens at 2 heads segments by the shipped policy (8 pieces).
    lens, H = [3072], 2
    x = _inputs(lens, H, state_dtype, slow_decay=slow_decay)
    offs = [checkpoint] if checkpoint is not None else None
    plan = kda_segscan.make_plan(x["cu_seqlens"].cpu(), H, "cuda", offs)
    assert plan is not None and plan.num_pseudo == 8

    ckpt_offsets = (
        torch.tensor(offs, dtype=torch.int32, device="cuda") if offs else None
    )
    ref_out, ref_final, ref_ckpt = _plain_flashkda(x, ckpt_offsets)

    T = sum(lens)
    bufs = kda_segscan.init_buffers(T, H, torch.device("cuda"))
    out = torch.empty_like(x["v"])
    final = torch.empty_like(x["initial_state"])
    ckpt = torch.empty_like(final) if offs else None
    kda_segscan.segmented_prefill(
        plan=plan,
        bufs=bufs,
        q=x["q"],
        k=x["k"],
        v=x["v"],
        g=x["g"],
        beta=x["beta"],
        A_log=x["A_log"],
        dt_bias=x["dt_bias"],
        lower_bound=x["lower_bound"],
        initial_state=x["initial_state"],
        out=out,
        final_state=final,
        workspace=_workspace(T, H, kda_segscan._MAX_PSEUDO),
        checkpoint_state=ckpt,
    )
    torch.accelerator.synchronize()

    # Segment 0 runs FlashKDA's chunks from FlashKDA's (bf16-rounded) initial
    # state: bitwise. States leave through bf16, as FlashKDA's.
    seg0 = int(plan.cu[1])
    torch.testing.assert_close(out[:, :seg0], ref_out[:, :seg0], rtol=0, atol=0)
    assert _bf16_representable(final)
    assert ckpt is None or _bf16_representable(ckpt)
    if not slow_decay:
        # The gate forgets each segment's initial state within the segment, so
        # the scan's X_p reach nothing: bitwise FlashKDA.
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
        assert torch.equal(final, ref_final)
        if ckpt is not None:
            assert torch.equal(ckpt, ref_ckpt)
    else:
        # The state survives every segment: later segments see the scan's
        # bf16-operand rounding of the M_p / C_p partials instead of the true
        # state's (kda_segscan docstring; measured max err 7e-3 x scale).
        _close("out", ref_out, out, 0.02)
        _close("final_state", ref_final, final, 0.02)
        if ckpt is not None:
            _close("checkpoint", ref_ckpt, ckpt, 0.02)


@requires_split_build
@pytest.mark.parametrize("slow_decay", [False, True], ids=["decay", "slow_decay"])
@pytest.mark.parametrize("checkpoint", [False, True], ids=["no_ckpt", "ckpt"])
@pytest.mark.parametrize(
    "state_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
)
@torch.inference_mode()
def test_narrow_vsplit_matches_flashkda(
    state_dtype: torch.dtype, checkpoint: bool, slow_decay: bool
):
    # Two sequences at 2 heads: 2 x 8 x 2 = 32 blocks, one wave anywhere.
    lens, H = [700, 333], 2
    x = _inputs(lens, H, state_dtype, seed=3, slow_decay=slow_decay)
    assert kda_segscan.narrow_vsplit_fits(H, len(lens), 128, x["q"].device)
    offs = (
        torch.tensor([512, 208], dtype=torch.int32, device="cuda")
        if checkpoint
        else None
    )
    ref_out, ref_final, ref_ckpt = _plain_flashkda(x, offs)
    out, final, ckpt = _split_flashkda(x, 3, offs)
    # Same per-column arithmetic as the stock kernel, and the same state I/O:
    # an fp32 initial state is read through bf16 and the final / checkpoint
    # states are written through bf16 (round to nearest even), so outputs and
    # states are bitwise those of stock _flashkda_C.
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    assert torch.equal(final, ref_final)
    if checkpoint:
        assert torch.equal(ckpt, ref_ckpt)


@requires_split_build
@torch.inference_mode()
def test_exact_state_mode_keeps_fp32():
    # Mode bit 2 (segment partials only) moves the fp32 state across the
    # kernel boundary exactly; the default mode rounds it through bf16. With
    # the fast-decaying gate the initial state does not reach the final state,
    # so the exact final state rounds to the stock one.
    lens, H = [700, 333], 2
    x = _inputs(lens, H, torch.float32, seed=3)
    _, ref_final, _ = _plain_flashkda(x)
    _, exact_final, _ = _split_flashkda(x, 3 | 4)
    assert _bf16_representable(ref_final)
    assert (exact_final.to(torch.bfloat16).float() != exact_final).float().mean() > 0.5
    assert torch.equal(exact_final.to(torch.bfloat16).float(), ref_final)
