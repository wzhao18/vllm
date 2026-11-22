# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Crossing chunks on the segment-parallel FlashKDA route (ops/kda_segscan.py).

The segmented prefill splits each sequence into segments chosen from the whole
step's batch, so the FP32 state the block-ended path stores at a boundary
already depends on the co-scheduled requests (order-level differences, same
math). A crossing chunk exports the state at its crossed boundary B as the
sequence's checkpoint (a segment boundary when the sequence is split). This
measures the block-ended path's own envelope over >= 20 random co-batches --
state at B, final state, outputs, against the single-launch reference -- and
requires the crossing chunk to stay inside it over >= 20 other co-batches.
"""

import random

import pytest
import torch

from vllm.models.kimi_k3.nvidia.ops import kda_segscan
from vllm.platforms import current_platform

H, D = 8, 128
BLOCK = 2048
START, END = BLOCK + BLOCK // 2, 2 * BLOCK + 700  # crosses B = 2 * BLOCK
B = (START // BLOCK + 1) * BLOCK
NUM_BATCHES = 20


def _supported() -> bool:
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


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="needs CUDA SM10x, FlashKDA (vllm._flashkda_C) and FlashInfer's CUTLASS",
)


def _tokens(n: int, seed: int) -> dict:
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, device="cuda")  # noqa: E731
    return dict(
        q=r(1, n, H, D).to(torch.bfloat16),
        k=r(1, n, H, D).to(torch.bfloat16),
        v=r(1, n, H, D).to(torch.bfloat16),
        g=r(1, n, H, D).to(torch.bfloat16),
        beta=r(1, n, H).to(torch.bfloat16),
    )


A_LOG = None
DT_BIAS = None


def _params():
    global A_LOG, DT_BIAS
    if A_LOG is None:
        g = torch.Generator(device="cuda").manual_seed(1)
        A_LOG = torch.randn(H, generator=g, device="cuda") * 0.5
        DT_BIAS = torch.randn(H, D, generator=g, device="cuda") * 0.1
    return A_LOG, DT_BIAS


def _run(seqs: list[tuple[dict, torch.Tensor, int]]):
    """One prefill step over ``seqs`` = [(tokens, initial state, checkpoint
    offset or 0)], on the route the KDA layer takes: segmented when the planner
    splits, else one FlashKDA launch. Returns per-sequence (out, final, ckpt)."""
    lens = [t["q"].shape[1] for t, _, _ in seqs]
    cu = torch.tensor([0, *lens], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    cat = {k: torch.cat([t[k] for t, _, _ in seqs], dim=1) for k in seqs[0][0]}
    init = torch.stack([s for _, s, _ in seqs])
    offs = [o for _, _, o in seqs]
    A_log, dt_bias = _params()
    T, N = sum(lens), len(seqs)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    max_parts = kda_segscan.max_pseudo(H, sms, 64)
    plan = kda_segscan.make_plan(cu, H, "cuda", offs, sms=sms, max_parts=max_parts)
    out = torch.empty_like(cat["v"])
    final = torch.empty_like(init)
    ckpt = torch.empty_like(init)
    ws_n = max(N, max_parts)
    workspace = torch.empty(
        torch.ops._flashkda_C.get_workspace_size(T, H, ws_n),
        dtype=torch.uint8,
        device="cuda",
    )
    if plan is not None and plan.num_seqs == N:
        bufs = kda_segscan.SegBuffers(T, H, torch.device("cuda"), max_parts, sms)
        kda_segscan.segmented_prefill(
            plan=plan,
            bufs=bufs,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=-3.0,
            initial_state=init,
            out=out,
            final_state=final,
            workspace=workspace,
            checkpoint_state=ckpt,
            **cat,
        )
    else:
        torch.ops._flashkda_C.fwd(
            cat["q"].contiguous(),
            cat["k"].contiguous(),
            cat["v"].contiguous(),
            cat["g"].contiguous(),
            cat["beta"],
            D**-0.5,
            out,
            workspace,
            A_log.contiguous(),
            dt_bias.view(-1, D).contiguous(),
            -3.0,
            init.contiguous(),
            final,
            cu.cuda(),
            ckpt,
            torch.tensor(offs, dtype=torch.int32, device="cuda"),
        )
    torch.accelerator.synchronize()
    res = []
    for i in range(N):
        res.append((out[:, cu[i] : cu[i + 1]].clone(), final[i].clone(), ckpt[i].clone()))
    return res, plan is not None


def _fillers(rng: random.Random, seed: int):
    n = rng.randint(1, 5)
    return [
        (_tokens(rng.choice([256, 512, 1024, 2048, 4096, 8192]), seed * 10 + j),
         torch.randn(H, D, D, device="cuda") * 0.1, 0)
        for j in range(n)
    ]


def _with(rng: random.Random, seed: int, seq):
    others = _fillers(rng, seed)
    pos = rng.randint(0, len(others))
    batch = others[:pos] + [seq] + others[pos:]
    res, split = _run(batch)
    return res[pos], split


@torch.inference_mode()
def test_crossing_stays_inside_the_block_ended_envelope():
    import vllm._flashkda_C  # noqa: F401

    kda_segscan.load_extension()
    x = _tokens(END - START, 12345)
    s0 = torch.randn(H, D, D, device="cuda", generator=torch.Generator(
        device="cuda").manual_seed(3)) * 0.1
    x1 = {k: v[:, : B - START] for k, v in x.items()}
    x2 = {k: v[:, B - START :] for k, v in x.items()}

    # Single-launch reference (alone in its step: one FlashKDA launch).
    (ref,), _ = _run([(x, s0, B - START)])
    ref_out, ref_final, ref_b = ref
    (a1,), _ = _run([(x1, s0, 0)])
    (a2,), _ = _run([(x2, a1[1], 0)])
    # Single launch: the crossing chunk is the block-ended split, bit for bit.
    assert torch.equal(ref_b, a1[1]) and torch.equal(ref_final, a2[1])
    assert torch.equal(ref_out, torch.cat([a1[0], a2[0]], dim=1))

    def dev(a, b):
        d = (a.float() - b.float()).abs()
        return d.max().item(), int((d > 0).sum())

    rng = random.Random(0)
    env = {"b": 0.0, "final": 0.0, "out": 0.0}
    splits = 0
    for i in range(NUM_BATCHES):
        (o1, f1, _), sp1 = _with(rng, 100 + i, (x1, s0, 0))
        (o2, f2, _), sp2 = _with(rng, 200 + i, (x2, f1, 0))
        splits += sp1 + sp2
        env["b"] = max(env["b"], dev(f1, ref_b)[0])
        env["final"] = max(env["final"], dev(f2, ref_final)[0])
        env["out"] = max(env["out"], dev(torch.cat([o1, o2], 1), ref_out)[0])
    worst = {"b": (0.0, 0), "final": (0.0, 0), "out": (0.0, 0)}
    xsplits = 0
    for i in range(NUM_BATCHES):
        (o, f, c), sp = _with(rng, 300 + i, (x, s0, B - START))
        xsplits += sp
        for k, (a, r) in {"b": (c, ref_b), "final": (f, ref_final), "out": (o, ref_out)}.items():
            m, nz = dev(a, r)
            worst[k] = max(worst[k], (m, nz))
    print(f"\nblock-ended envelope {env} ({splits} segmented runs); crossing worst "
          f"(max |delta|, #diff elems) {worst} ({xsplits} segmented runs)")
    assert splits > 0 and xsplits > 0, "no co-batch took the segmented route"
    for k in env:
        assert worst[k][0] <= env[k], f"{k}: crossing {worst[k][0]} > envelope {env[k]}"
