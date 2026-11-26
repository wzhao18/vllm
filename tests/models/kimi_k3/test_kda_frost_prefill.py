# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KDA prefill core on cuDNN FROST (specialized_prefill/kda.py) against
FlashKDA, the core it replaces: same output and final state up to BF16
operand rounding order."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(100),
    reason="cuDNN FROST KDA targets SM10x",
)

D = 128


def _rel_rms(a, b):
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


@pytest.mark.parametrize("lens,heads", [([37], 4), ([300, 1, 1029], 8), ([2051, 777], 96)])
def test_frost_prefill_matches_flashkda(lens, heads):
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

    assert sp_kda.frost_supported(torch.device("cuda"))
    import vllm._flashkda_C  # noqa: F401

    torch.manual_seed(0)
    T, N, H = sum(lens), len(lens), heads
    bf = torch.bfloat16
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    q, k, g = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
    v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
    beta = torch.randn(T, H + 7, device="cuda").to(bf)[:, :H].unsqueeze(0)  # strided
    A_log = torch.randn(H, device="cuda") * 0.5
    dt_bias = torch.randn(H * D, device="cuda") * 0.1
    h0 = torch.randn(N, H, D, D, device="cuda") * 0.1
    out = torch.empty(1, T, H, D, dtype=bf, device="cuda")
    fs = torch.empty_like(h0)
    ws = torch.empty(torch.ops._flashkda_C.get_workspace_size(T, H, N), dtype=torch.uint8, device="cuda")
    torch.ops._flashkda_C.fwd(
        q, k, v, g, beta.contiguous(), D**-0.5, out, ws, A_log, dt_bias.view(H, D),
        -5.0, h0, fs, cu, None, None,
    )
    o, state = sp_kda.frost_prefill(q, k, v, g, beta, A_log, dt_bias, -5.0, h0, cu)
    torch.accelerator.synchronize()
    assert o.shape == out.shape and state.shape == fs.shape
    assert not torch.isnan(o).any()
    assert _rel_rms(o, out) < 2e-2
    assert _rel_rms(state, fs) < 2e-2


@pytest.mark.parametrize(
    "lens,heads",
    [
        ([37], 8),
        ([300, 1, 1029], 8),
        ([2051, 777, 64], 96),
        ([5461] * 3, 96),
        # TP shards of the 96-head KDA whose beta rows are not 16 B multiples
        # (TP8: 12 heads; the beta TMA path read wrong rows, BUG-DCP8-3).
        ([300, 1, 1029], 12),
        ([2048] * 8, 12),
        ([37], 4),
        ([300, 1], 20),
    ],
)
def test_persistent_prefill_matches_flashkda(lens, heads):
    """The persistent core (LPT chain pieces with FP32 state handoffs; the
    3 x 5461 x 96-head shape splits chains across CTAs) against FlashKDA."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

    if not sp_kda.persistent_supported(torch.device("cuda")):
        pytest.skip("CuTe DSL not installed")
    import vllm._flashkda_C  # noqa: F401

    torch.manual_seed(0)
    T, N, H = sum(lens), len(lens), heads
    bf = torch.bfloat16
    offs = [0, *torch.tensor(lens).cumsum(0).tolist()]
    cu_cpu = torch.tensor(offs, dtype=torch.int32)
    cu = cu_cpu.cuda()
    q, k, g = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
    v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
    beta = torch.randn(T, H + 7, device="cuda").to(bf)[:, :H].unsqueeze(0)  # strided
    A_log = torch.randn(H, device="cuda") * 0.5
    dt_bias = torch.randn(H * D, device="cuda") * 0.1
    h0 = torch.randn(N, H, D, D, device="cuda") * 0.1
    out = torch.empty(1, T, H, D, dtype=bf, device="cuda")
    fs = torch.empty_like(h0)
    ws = torch.empty(torch.ops._flashkda_C.get_workspace_size(T, H, N), dtype=torch.uint8, device="cuda")
    torch.ops._flashkda_C.fwd(
        q, k, v, g, beta.contiguous(), D**-0.5, out, ws, A_log, dt_bias.view(H, D),
        -5.0, h0, fs, cu, None, None,
    )
    h0_ref = h0.clone()
    o, state = sp_kda.persistent_prefill(
        q, k, v, g, beta, A_log, dt_bias, -5.0, h0, cu, cu_cpu
    )
    torch.accelerator.synchronize()
    assert torch.equal(h0, h0_ref)
    assert o.shape == out.shape and state.shape == fs.shape
    assert not torch.isnan(o).any()
    assert _rel_rms(o, out) < 2e-2
    assert _rel_rms(state, fs) < 2e-2


@pytest.mark.parametrize("lens,heads", [([2048] * 8, 12), ([300, 1, 1029], 12)])
def test_persistent_prefill_beta_view_bitwise(lens, heads):
    """The KDA layer hands the persistent core its beta as a column view of
    the fused prefill in_proj output (row stride 1680 at the TP8 shard: q|k|v
    planes aside, g | f_a | beta | pad) instead of a dense copy: output and
    final state bitwise equal to the dense beta."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

    if not sp_kda.persistent_supported(torch.device("cuda")):
        pytest.skip("CuTe DSL not installed")
    torch.manual_seed(3)
    T, N, H = sum(lens), len(lens), heads
    bf = torch.bfloat16
    offs = [0, *torch.tensor(lens).cumsum(0).tolist()]
    cu_cpu = torch.tensor(offs, dtype=torch.int32)
    cu = cu_cpu.cuda()
    q, k, g = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
    v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
    rest = torch.randn(T, H * D + D + H + 4, device="cuda").to(bf)
    beta_view = rest[:, H * D + D : H * D + D + H].unsqueeze(0)
    assert not beta_view.is_contiguous()
    A_log = torch.randn(H, device="cuda") * 0.5
    dt_bias = torch.randn(H * D, device="cuda") * 0.1
    h0 = torch.randn(N, H, D, D, device="cuda") * 0.1
    o_ref, fs_ref = sp_kda.persistent_prefill(
        q, k, v, g, beta_view.contiguous(), A_log, dt_bias, -5.0, h0, cu, cu_cpu
    )
    o, fs = sp_kda.persistent_prefill(
        q, k, v, g, beta_view, A_log, dt_bias, -5.0, h0, cu, cu_cpu
    )
    torch.accelerator.synchronize()
    assert torch.equal(o, o_ref)
    assert torch.equal(fs, fs_ref)


@pytest.mark.parametrize(
    "lens,heads", [([300, 1, 1029], 8), ([5461] * 3, 96), ([300, 1, 1029], 12)]
)
def test_persistent_prefill_state_in_place(lens, heads):
    """In-place state I/O on a padded state cache (slot stride > H*D*D,
    scattered slots, one sequence without an initial state) is bitwise equal
    to the gathered-state path with a row scatter."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

    if not sp_kda.persistent_supported(torch.device("cuda")):
        pytest.skip("CuTe DSL not installed")
    torch.manual_seed(1)
    T, N, H = sum(lens), len(lens), heads
    bf = torch.bfloat16
    offs = [0, *torch.tensor(lens).cumsum(0).tolist()]
    cu_cpu = torch.tensor(offs, dtype=torch.int32)
    cu = cu_cpu.cuda()
    q, k, g = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
    v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
    beta = torch.randn(1, T, H, device="cuda").to(bf)
    A_log = torch.randn(H, device="cuda") * 0.5
    dt_bias = torch.randn(H * D, device="cuda") * 0.1
    # Production layout: the hybrid cache's per-slot page stride (83897856
    # elements) is not a multiple of D * D, and slot offsets pass 2**31
    # elements, so the kernel needs 64-bit, non-block-aligned slot offsets.
    sstride = 83897856
    slots = 2 * N + 1 + (2**31 // sstride)
    flat = torch.empty(slots * sstride, device="cuda")
    cache = flat.as_strided((slots, H, D, D), (sstride, D * D, D, 1))
    idx = torch.randperm(slots - 1, device="cuda")[: N - 1].to(torch.int32)
    idx = torch.cat((idx, torch.tensor([slots - 1], device="cuda", dtype=torch.int32)))
    cache[idx.long()] = torch.randn(N, H, D, D, device="cuda") * 0.1
    has = torch.ones(N, dtype=torch.bool, device="cuda")
    has[N // 2] = False
    h0 = cache[idx.long()].clone()
    h0[~has] = 0
    o_ref, fs_ref = sp_kda.persistent_prefill(
        q, k, v, g, beta, A_log, dt_bias, -5.0, h0, cu, cu_cpu
    )
    pages = flat.view(slots, sstride)
    pages[idx.long(), H * D * D :] = 7.0  # the rest of each touched page
    o, fs = sp_kda.persistent_prefill(
        q, k, v, g, beta, A_log, dt_bias, -5.0, None, cu, cu_cpu,
        state_cache=cache, state_indices=idx, has_initial_state=has,
    )
    torch.accelerator.synchronize()
    assert fs is None
    assert torch.equal(o, o_ref)
    assert torch.equal(cache[idx.long()], fs_ref)
    assert bool((pages[idx.long(), H * D * D :] == 7.0).all())


def test_frost_strided_beta():
    """frost_prefill passes beta as a row-strided view of the projection:
    FROST must read it exactly like a contiguous copy."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

    assert sp_kda.frost_supported(torch.device("cuda"))
    torch.manual_seed(1)
    T, H, bf = 300, 8, torch.bfloat16
    cu = torch.tensor([0, 100, 300], dtype=torch.int32, device="cuda")
    q, k, v, g = (torch.randn(T, H, D, device="cuda").to(bf) for _ in range(4))
    beta = torch.randn(T, H + 7, device="cuda").to(bf)[:, :H]
    kw = dict(scale=D**-0.5, initial_state=torch.zeros(2, H, D, D, device="cuda"),
              output_final_state=True, use_qk_l2norm_in_kernel=True,
              use_beta_sigmoid_in_kernel=True, safe_gate=True, gate_lower_bound=-5.0,
              a_log=torch.zeros(H, device="cuda"), dt_bias=torch.zeros(H, D, device="cuda"),
              plan_name="kda_frost")
    ref = sp_kda._frost_fn(q, k, v, g, beta.contiguous(), cu, **kw)[0]
    got = sp_kda._frost_fn(q, k, v, g, beta, cu, **kw)[0]
    assert torch.equal(got, ref)


def test_frost_requires_cudnn_frontend_133(monkeypatch):
    """An older cuDNN frontend on SM10x is an install error, not a silent
    FlashKDA fallback."""
    import cudnn

    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

    monkeypatch.setattr(sp_kda, "_frost_fn", None)
    monkeypatch.setattr(cudnn, "__version__", "1.32.1")
    with pytest.raises(RuntimeError, match="older than 1.33"):
        sp_kda.frost_supported(torch.device("cuda"))


@pytest.mark.parametrize("heads", [12, 96])
def test_persistent_warm_covers_serving_beta_layouts(heads):
    """The layer-init warm-up (warm_persistent) compiles the beta TMA variant
    serving launches: TMA at H % 8 == 0, per-element loads at the TP8 shard's
    12 heads (BUG-DCP8-3), also for misaligned beta views. A serving step must
    not JIT (a mid-serving compile on one rank stalls its TP/DCP peers)."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

    if not sp_kda.persistent_supported(torch.device("cuda")):
        pytest.skip("CuTe DSL not installed")
    if not hasattr(sp_kda, "warm_persistent"):
        pytest.skip("no persistent-core warm-up in this tree")
    from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent import (
        forward as kp,
    )

    torch.manual_seed(0)
    H, bf = heads, torch.bfloat16
    sp_kda.warm_persistent(H, torch.device("cuda"), -5.0)
    compiled = set(kp._COMPILED)
    for lens in ([300, 1, 1029], [2048], [256, 256]):
        T, N = sum(lens), len(lens)
        offs = [0, *torch.tensor(lens).cumsum(0).tolist()]
        cu_cpu = torch.tensor(offs, dtype=torch.int32)
        q, k, g = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
        v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
        for beta in (
            torch.randn(1, T, H, device="cuda").to(bf),
            # a view 4 bytes past an aligned base (not a 16-byte multiple)
            torch.randn(T * H + 2, device="cuda").to(bf)[2:].view(1, T, H),
        ):
            sp_kda.persistent_prefill(
                q, k, v, g, beta, torch.randn(H, device="cuda") * 0.5,
                torch.randn(H * D, device="cuda") * 0.1, -5.0,
                torch.randn(N, H, D, D, device="cuda") * 0.1, cu_cpu.cuda(), cu_cpu,
            )
    torch.accelerator.synchronize()
    new = set(kp._COMPILED) - compiled
    assert not new, f"{len(new)} persistent-core variants compiled after warm-up"


@pytest.mark.parametrize("lens,heads", [([300, 1, 1029], 8), ([5461] * 3, 96)])
def test_persistent_prefill_step_memo_bitwise(lens, heads):
    """Several KDA layers of one step through the per-step memo (plan and
    state-row table built by the first layer, reused by the others) give
    the same bits as each layer building its own, on separate state caches."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

    if not sp_kda.persistent_supported(torch.device("cuda")):
        pytest.skip("CuTe DSL not installed")
    torch.manual_seed(2)
    T, N, H = sum(lens), len(lens), heads
    bf = torch.bfloat16
    offs = [0, *torch.tensor(lens).cumsum(0).tolist()]
    cu_cpu = torch.tensor(offs, dtype=torch.int32)
    cu = cu_cpu.cuda()
    slots = 2 * N + 3
    idx = torch.randperm(slots, device="cuda")[:N].to(torch.int32)
    has = torch.ones(N, dtype=torch.bool, device="cuda")
    has[0] = False
    layers = []
    for _ in range(3):
        q, k, g = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
        v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
        beta = torch.randn(1, T, H, device="cuda").to(bf)
        A_log = torch.randn(H, device="cuda") * 0.5
        dt_bias = torch.randn(H * D, device="cuda") * 0.1
        init = torch.randn(slots, H, D, D, device="cuda") * 0.1
        layers.append((q, k, v, g, beta, A_log, dt_bias, init))

    def run(memo_obj):
        outs, caches = [], []
        for q, k, v, g, beta, A_log, dt_bias, init in layers:
            cache = init.clone()
            o, _ = sp_kda.persistent_prefill(
                q, k, v, g, beta, A_log, dt_bias, -5.0, None, cu, cu_cpu,
                state_cache=cache, state_indices=idx, has_initial_state=has,
                step_memo=(
                    sp_kda.step_memo_for(memo_obj) if memo_obj is not None else None
                ),
            )
            outs.append(o)
            caches.append(cache)
        torch.accelerator.synchronize()
        return outs, caches

    class _Step:
        pass

    ref_o, ref_c = run(None)
    o, c = run(_Step())
    for a, b in zip(ref_o, o):
        assert torch.equal(a, b)
    for a, b in zip(ref_c, c):
        assert torch.equal(a, b)
