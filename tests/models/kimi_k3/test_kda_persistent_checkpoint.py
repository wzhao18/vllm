# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill checkpoints on the persistent KDA core (specialized_prefill/kda.py).

A checkpoint is a sequence boundary of the persistent schedule: the prefix
[0, c) runs as a sequence of its own and exports its FP32 state, the suffix
[c, L) runs seeded with it. So, bitwise:
  (i)   checkpoint state == the persistent final state of the prefix alone;
  (ii)  prefix outputs == that run's outputs; suffix outputs and the final
        state == a persistent run of the suffix seeded with (i);
  (iii) sequences without a checkpoint == the batch run without checkpoints,
        and with every cut on the 32-token chunk grid the whole batch is;
and against FlashKDA with the same checkpoints, within the persistent core's
existing tolerance (test_kda_frost_prefill)."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(100),
    reason="persistent KDA targets SM10x",
)

D = 128


def _rel_rms(a, b):
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


def _inputs(lens, H, seed):
    torch.manual_seed(seed)
    T = sum(lens)
    bf = torch.bfloat16
    q, k, g = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
    v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
    beta = torch.randn(1, T, H, device="cuda").to(bf)
    A_log = torch.randn(H, device="cuda") * 0.5
    dt_bias = torch.randn(H * D, device="cuda") * 0.1
    h0 = torch.randn(len(lens), H, D, D, device="cuda") * 0.1
    return q, k, v, g, beta, A_log, dt_bias, h0


def _run(sp_kda, x, segs, h0, has=None, cuts=None, ckpt=None, memo=None):
    """Persistent prefill over token windows ``segs`` [(start, len)] of the
    inputs, packed in order, with initial states ``h0`` (in-place state cache).
    Returns (outputs per seg, final states [n, H, D, D])."""
    q, k, v, g, beta, A_log, dt_bias = x
    H = q.shape[2]
    idx_tok = torch.cat([torch.arange(s, s + n, device="cuda") for s, n in segs])
    sel = lambda t: t[:, idx_tok].contiguous()
    offs = [0]
    for _, n in segs:
        offs.append(offs[-1] + n)
    cu_cpu = torch.tensor(offs, dtype=torch.int32)
    n = len(segs)
    cache = torch.zeros(2 * n + 3, H, D, D, device="cuda")
    slot = torch.arange(n, device="cuda", dtype=torch.int32) * 2 + 1
    cache[slot.long()] = h0
    if has is None:
        has = torch.ones(n, dtype=torch.bool, device="cuda")
    o, fs = sp_kda.persistent_prefill(
        sel(q), sel(k), sel(v), sel(g), sel(beta), A_log, dt_bias, -5.0, None,
        cu_cpu.cuda(), cu_cpu, state_cache=cache, state_indices=slot,
        has_initial_state=has, step_memo=memo,
        checkpoint_offsets=cuts, checkpoint_out=ckpt,
    )
    assert fs is None
    torch.accelerator.synchronize()
    outs = [o[:, offs[i] : offs[i + 1]] for i in range(n)]
    return outs, cache[slot.long()].clone()


CASES = [
    # (lens, cuts, heads): cuts 16 mod 32, 32-aligned, tiny prefix, near the
    # end, uncut neighbours; a long chain the planner splits; many sequences.
    ([1000, 4096, 777], [496, 2048, 0], 12),
    ([1000, 4096, 777], [16, 4080, 0], 12),
    ([2051, 777, 64], [1040, 0, 48], 96),
    ([5461] * 3, [2720, 5456, 16], 96),
    ([16384], [8208], 96),
    ([16384], [8192], 96),
    ([1024] * 16, [512, 0, 1008, 16, 0, 0, 784, 0, 0, 0, 1000 // 16 * 16, 0, 0, 0, 0, 528], 12),
]


@pytest.mark.parametrize("lens,cuts,heads", CASES)
def test_checkpoint_bitwise_self_consistent(lens, cuts, heads):
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

    if not sp_kda.persistent_supported(torch.device("cuda")):
        pytest.skip("CuTe DSL not installed")
    H = heads
    q, k, v, g, beta, A_log, dt_bias, h0 = _inputs(lens, H, 0)
    x = (q, k, v, g, beta, A_log, dt_bias)
    starts = [0]
    for n in lens:
        starts.append(starts[-1] + n)
    segs = [(starts[i], n) for i, n in enumerate(lens)]
    has = torch.ones(len(lens), dtype=torch.bool, device="cuda")
    has[0] = False  # one sequence starts from a zero state
    h0 = h0 * has.view(-1, 1, 1, 1)
    ckpt = torch.full((len(lens) + 2, H, D, D), float("nan"), device="cuda")
    outs, fin = _run(sp_kda, x, segs, h0, has, cuts, ckpt, memo={})

    # (i) prefix alone -> checkpoint state, prefix outputs
    cut_ids = [i for i, c in enumerate(cuts) if c]
    pre_outs, pre_fin = _run(
        sp_kda, x, [(starts[i], cuts[i]) for i in cut_ids], h0[cut_ids], has[cut_ids]
    )
    for j, i in enumerate(cut_ids):
        assert torch.equal(ckpt[i], pre_fin[j]), f"checkpoint state of seq {i}"
        assert torch.equal(outs[i][:, : cuts[i]], pre_outs[j]), f"prefix out {i}"
    # (ii) suffix seeded with the checkpoint state
    suf_outs, suf_fin = _run(
        sp_kda, x, [(starts[i] + cuts[i], lens[i] - cuts[i]) for i in cut_ids],
        pre_fin,
    )
    for j, i in enumerate(cut_ids):
        assert torch.equal(outs[i][:, cuts[i] :], suf_outs[j]), f"suffix out {i}"
        assert torch.equal(fin[i], suf_fin[j]), f"final state {i}"
    # (iii) uncut sequences == the batch without checkpoints
    ref_outs, ref_fin = _run(sp_kda, x, segs, h0, has)
    for i, c in enumerate(cuts):
        if not c or c % 32 == 0:
            assert torch.equal(outs[i], ref_outs[i]), f"out {i} vs uncut run"
            assert torch.equal(fin[i], ref_fin[i]), f"final {i} vs uncut run"
    # rows without a checkpoint are untouched
    for i, c in enumerate(cuts):
        if not c:
            assert torch.isnan(ckpt[i]).all()


@pytest.mark.parametrize("lens,cuts,heads", CASES[:5])
def test_checkpoint_matches_flashkda(lens, cuts, heads):
    """Outputs, final and checkpoint states against FlashKDA's checkpoint
    route (the one this replaces), within the persistent core's tolerance."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

    if not sp_kda.persistent_supported(torch.device("cuda")):
        pytest.skip("CuTe DSL not installed")
    import vllm._flashkda_C  # noqa: F401

    H, N, T = heads, len(lens), sum(lens)
    q, k, v, g, beta, A_log, dt_bias, h0 = _inputs(lens, H, 1)
    starts = [0]
    for n in lens:
        starts.append(starts[-1] + n)
    cu = torch.tensor(starts, dtype=torch.int32, device="cuda")
    out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda")
    fs = torch.empty_like(h0)
    cs = torch.zeros_like(h0)
    ws = torch.empty(
        torch.ops._flashkda_C.get_workspace_size(T, H, N), dtype=torch.uint8, device="cuda"
    )
    torch.ops._flashkda_C.fwd(
        q, k, v, g, beta, D**-0.5, out, ws, A_log, dt_bias.view(H, D), -5.0,
        h0, fs, cu, cs, torch.tensor(cuts, dtype=torch.int32, device="cuda"),
    )
    ckpt = torch.zeros(N + 1, H, D, D, device="cuda")
    outs, fin = _run(
        sp_kda, (q, k, v, g, beta, A_log, dt_bias),
        [(starts[i], n) for i, n in enumerate(lens)], h0, None, cuts, ckpt,
    )
    o = torch.cat(outs, dim=1)
    assert not torch.isnan(o).any()
    assert _rel_rms(o, out) < 2e-2
    assert _rel_rms(fin, fs) < 2e-2
    for i, c in enumerate(cuts):
        if c:
            assert _rel_rms(ckpt[i], cs[i]) < 2e-2, f"checkpoint {i}"


def test_cut_schedule_handoffs():
    """Every cut schedule covers each sequence exactly once, every handoff
    buffer has one producer and one consumer, and a cut's consumer runs right
    after its producer in the same slot."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent import (
        forward as kp,
    )

    for lens, cuts, H in CASES + [([5461] * 3, [0, 0, 0], 96)]:
        items, nbuf = kp._piece_items(lens, H, 212)
        out, plens, pseudo_of, ntot = kp._cut_items(items, lens, cuts, H, nbuf)
        cover = {}
        prod, cons = {}, {}
        for slot in out:
            for pos, (c, t0, tn, src, dst) in enumerate(slot):
                cover.setdefault(c, []).append((t0, tn))
                if dst >= 0:
                    prod[dst] = (id(slot), pos)
                if src >= 0:
                    cons[src] = (id(slot), pos)
        assert set(prod) == set(cons)
        for b, (sl, pos) in cons.items():
            if b < len(lens) * H:  # cut buffer: same slot, right after
                assert prod[b] == (sl, pos - 1)
        for c, pieces in cover.items():
            p, h = divmod(c, H)
            pieces.sort()
            t = 0
            for t0, tn in pieces:
                assert t0 == t and tn > 0
                t += tn
            assert t == plens[p]
        assert ntot >= len(lens) * H
