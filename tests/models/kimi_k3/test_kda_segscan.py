# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Segment-parallel FlashKDA prefill (ops/kda_segscan.py).

The policy tests are host-only: the cost-model planner (segment split, go /
no-go at the served shapes, pseudo-sequence cap) and the V-split dispatch with
a mocked SM count. The GPU tests compare the segmented prefill and the 16- and
32-column V-split launches against plain FlashKDA: the V-split launch is
bitwise stock (outputs, final and checkpoint states, fp32 states rounded through
bf16 as stock); the segmented prefill is bitwise on its first segment, and
bitwise overall when the gate forgets the state within a segment. The
output-free transition passes give bitwise the states of the output passes.
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


def _plan(lens, heads, offs=None, max_parts=None):
    if max_parts is None:
        max_parts = kda_segscan.max_pseudo(heads, RUBIN_SMS, 64)
    return kda_segscan.make_plan(
        _cu(*lens), heads, "cpu", offs, sms=RUBIN_SMS, max_parts=max_parts
    )


@pytest.mark.parametrize(
    ("heads", "max_seqs", "n"),
    [(12, 64, 13), (24, 64, 6), (12, 8, 8), (96, 64, 1)],
)
def test_max_pseudo(heads: int, max_seqs: int, n: int):
    # one wave of 32-column transition blocks: 4 x heads x P <= 3 x 212
    assert kda_segscan.max_pseudo(heads, RUBIN_SMS, max_seqs) == n


def test_segment_counts_shapes():
    assert kda_segscan._segment_counts((), 8) is None
    # as many sequences as pseudo-sequences: nothing left to split
    assert kda_segscan._segment_counts((8192,) * 8, 8) is None
    assert kda_segscan._segment_counts((16384,), 8) == [8]
    assert kda_segscan._segment_counts((8192, 8192), 8) == [4, 4]
    # extra segments go to the longest current segment; short ones stay whole
    assert kda_segscan._segment_counts((12288, 3072, 1400), 8) == [5, 2, 1]
    # segments stay >= _MIN_SEG tokens
    assert kda_segscan._segment_counts((600,), 8) == [2]


def test_step_model_slicing():
    sms = RUBIN_SMS
    assert kda_segscan._step_us(12, 2, sms) == kda_segscan._STEP_VD16_US  # 192
    assert kda_segscan._step_us(24, 5, sms) == kda_segscan._STEP_VD32_US[2]  # 480
    assert kda_segscan._step_us(12, 8, sms) == kda_segscan._STEP_VD32_US[1]  # 384
    assert kda_segscan._step_us(24, 10, sms) > kda_segscan._STEP_FULL_US  # 2 waves


def test_make_plan_c256_2x8192_h12_segments():
    # 192 chains on 212 SMs: the transition passes have room for 13 pseudo-
    # sequences, and the projected plan beats the plain 512-chunk chain.
    plan = _plan([8192, 8192], 12)
    assert plan is not None and plan.num_seqs == 2
    assert 2 < plan.num_pseudo <= 13
    assert plan.count.tolist()[0] == plan.count.tolist()[1]
    assert plan.cu.tolist()[-1] == 16384


def test_make_plan_c512_shapes_decline():
    # 24 heads: the plain launch already fills the GPU (32-column blocks).
    assert _plan([3480] * 5, 24) is None
    assert _plan([5461] * 6, 24) is None


def test_make_plan_short_rows_decline():
    assert _plan([300, 200], 12) is None


def test_make_plan_respects_max_parts():
    assert _plan([16384], 12, max_parts=1) is None
    plan = _plan([16384], 12, max_parts=4)
    assert plan is not None and plan.num_pseudo <= 4


def test_make_plan_checkpoint_offset_is_a_boundary():
    plan = _plan([16384], 12, [4096], max_parts=8)
    assert plan is not None and plan.num_pseudo == 8
    cu = plan.cu.tolist()
    assert cu == [0, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384]
    assert plan.ck_x is not None
    assert plan.ck_x[0].tolist() == [0]
    assert plan.ck_x[1].tolist() == [cu.index(4096)]
    assert plan.ck_k is None and plan.kernel_ckpt_offsets is None


def test_make_plan_is_cached():
    a = _plan([16384], 12)
    assert _plan([16384], 12) is a


@pytest.mark.parametrize(
    ("heads", "sequences", "fits"),
    [
        (12, 2, True),  # C256 TP8: 16 columns, 12 x 8 x 2 = 192 <= 212
        (24, 5, True),  # C512 TP4: 32 columns, 24 x 4 x 5 = 480 <= 3 x 212
        (24, 6, True),  # 576
        (24, 7, False),  # 672
        (96, 2, False),
    ],
)
def test_split_build_fits(monkeypatch, heads: int, sequences: int, fits: bool):
    assert kda_segscan.split_build_fits(heads, sequences, 128, RUBIN_SMS) is fits
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
    lens, H = [3072], 2
    x = _inputs(lens, H, state_dtype, slow_decay=slow_decay)
    offs = [checkpoint] if checkpoint is not None else None
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    # 3072 tokens at 2 heads: the planner splits the chain (8 pieces at most)
    plan = kda_segscan.make_plan(
        x["cu_seqlens"].cpu(), H, "cuda", offs, sms=sms, max_parts=8
    )
    assert plan is not None and 2 <= plan.num_pseudo <= 8

    ckpt_offsets = (
        torch.tensor(offs, dtype=torch.int32, device="cuda") if offs else None
    )
    ref_out, ref_final, ref_ckpt = _plain_flashkda(x, ckpt_offsets)

    T = sum(lens)
    bufs = kda_segscan.SegBuffers(T, H, torch.device("cuda"), 8, sms)
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
        workspace=_workspace(T, H, 8),
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


def _split_fwd(x: dict, initial, final, mode: int, v=None, out=None):
    _, T, H, D = x["q"].shape
    out = torch.empty_like(x["v"]) if out is None else out
    torch.ops.k3_flashkda_split.fwd(
        x["q"].contiguous(),
        x["k"].contiguous(),
        (x["v"] if v is None else v).contiguous(),
        x["g"].contiguous(),
        x["beta"],
        D**-0.5,
        out,
        x["ws"],
        x["A_log"].contiguous(),
        x["dt_bias"].view(-1, D).contiguous(),
        x["lower_bound"],
        initial,
        final,
        x["cu_seqlens"],
        None,
        None,
        mode,
    )
    return out


@requires_split_build
@torch.inference_mode()
def test_transition_passes_match_output_passes():
    # The transition passes (exact fp32 state I/O, mode 2 | 4, plus 8: no
    # outputs, and 16: v = 0 in-kernel) run the same state arithmetic as the
    # exact-state output pass: bitwise-identical states, and the output buffer
    # is left untouched.
    kda_segscan.load_extension()
    lens, H = [700, 333, 1024], 2
    x = _inputs(lens, H, torch.float32, seed=5)
    T, N = sum(lens), len(lens)
    x["ws"] = _workspace(T, H, N)
    _split_fwd(x, None, None, 1)  # K1
    eye = torch.eye(128, device="cuda").expand(N, H, 128, 128).contiguous()

    c_ref = torch.empty_like(x["initial_state"])
    _split_fwd(x, None, c_ref, 2 | 4)
    c = torch.empty_like(c_ref)
    sentinel = torch.full_like(x["v"], 3.0)
    out = sentinel.clone()
    _split_fwd(x, None, c, 2 | 4 | 8, out=out)
    torch.testing.assert_close(c, c_ref, rtol=0, atol=0)
    assert torch.equal(out, sentinel)

    m_ref = torch.empty_like(c_ref)
    _split_fwd(x, eye, m_ref, 2 | 4, v=torch.zeros_like(x["v"]))
    m = torch.empty_like(c_ref)
    _split_fwd(x, eye, m, 2 | 4 | 8 | 16)
    torch.testing.assert_close(m, m_ref, rtol=0, atol=0)

    with pytest.raises(RuntimeError):
        _split_fwd(x, None, None, 8)  # output-free without the K2 bit
    with pytest.raises(RuntimeError):
        _split_fwd(x, None, c, 2 | 8)  # output-free without exact state I/O


@requires_split_build
@torch.inference_mode()
def test_vd32_launch_matches_flashkda():
    # More 16-column blocks than SMs, within three 32-column blocks per SM:
    # the 32-column slicing; same per-column arithmetic as stock FlashKDA.
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    H = sms // 16 + 1
    lens = [300, 517]
    assert H * 8 * len(lens) > sms and H * 4 * len(lens) <= 3 * sms
    x = _inputs(lens, H, torch.float32, seed=11)
    assert kda_segscan.narrow_vsplit_fits(H, len(lens), 128, x["q"].device)
    ref_out, ref_final, _ = _plain_flashkda(x)
    kda_segscan.load_extension()
    T = sum(lens)
    x["ws"] = _workspace(T, H, len(lens))
    final = torch.empty_like(x["initial_state"])
    out = _split_fwd(x, x["initial_state"].contiguous(), final, 3)
    torch.accelerator.synchronize()
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    # the fp32 state crosses the kernel through bf16, as the stock build
    assert torch.equal(final, ref_final)
