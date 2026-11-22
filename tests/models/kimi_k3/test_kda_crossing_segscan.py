# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Crossing chunks on the segment-parallel FlashKDA route (ops/kda_segscan.py).

The segmented prefill splits each sequence into segments chosen from the whole
step's batch, so the FP32 state the block-ended path stores at a boundary
already depends on the co-scheduled requests (order-level differences, same
math). A crossing chunk exports the state at its crossed boundary B as the
sequence's checkpoint (a segment boundary when the sequence is split). This
measures stock's own envelope over >= 20 random co-batches -- the block-ended
split (state at B, final state, outputs) and the same span run as one stock
chunk, without an export or as a prompt's last chunk exporting its tail
checkpoint in-step, against the single-launch reference -- and requires
the crossing chunk to stay inside it over >= 20 other co-batches.
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


def _tokens(n: int, seed: int, slow_decay: bool = False) -> dict:
    """``slow_decay``: gate ~1 with unit q / k, so the initial state survives
    every segment and the segment scan's rounding shows (kda_segscan tests)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=gen, device="cuda")

    q, k = r(1, n, H, D), r(1, n, H, D)
    g = r(1, n, H, D)
    if slow_decay:
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        g = torch.full_like(g, -60.0)
    return dict(
        q=q.to(torch.bfloat16),
        k=k.to(torch.bfloat16),
        v=r(1, n, H, D).to(torch.bfloat16),
        g=g.to(torch.bfloat16),
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


@pytest.mark.parametrize("slow_decay", [False, True], ids=["decay", "slow_decay"])
@torch.inference_mode()
def test_crossing_stays_inside_the_block_ended_envelope(slow_decay: bool):
    import vllm._flashkda_C  # noqa: F401

    kda_segscan.load_extension()
    x = _tokens(END - START, 12345, slow_decay)
    s0 = torch.randn(H, D, D, device="cuda", generator=torch.Generator(
        device="cuda").manual_seed(3)) * 0.1
    x1 = {k: v[:, : B - START] for k, v in x.items()}
    x2 = {k: v[:, B - START :] for k, v in x.items()}

    # Single-launch reference (alone in its step: one FlashKDA launch).
    (ref,), _ = _run([(x, s0, B - START)])
    ref_out, ref_final, ref_b = ref
    (a1,), _ = _run([(x1, s0, 0)])
    (a2,), _ = _run([(x2, a1[1], 0)])
    # Single launch: the state at B is the block-ended split's, bit for bit.
    assert torch.equal(ref_b, a1[1])
    if not slow_decay:
        # The gate forgets the state within a block: everything is bitwise.
        assert torch.equal(ref_final, a2[1])
        assert torch.equal(ref_out, torch.cat([a1[0], a2[0]], dim=1))
    # Under a gate ~1 the split re-enters FlashKDA at B from its stored state
    # (FlashKDA's states leave through bf16), which the crossing chunk does
    # not: a chunk-boundary rounding stock chunking already places wherever
    # the budget cuts. The envelope below measures it.

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
        # Stock also runs this whole span as one chunk (no export) whenever a
        # request reaches it block-aligned: its segmentation varies the same way.
        # ... and, as a prompt's last chunk, with its tail checkpoint exported
        # in-step (a segment boundary when split), at a hash-aligned position.
        tail = (END - 1) // 128 * 128 - START - 128 * rng.randint(0, 8)
        for off in (0, tail):
            (o3, f3, _), sp3 = _with(rng, 400 + i + 50 * (off > 0), (x, s0, off))
            splits += sp3
            env["final"] = max(env["final"], dev(f3, ref_final)[0])
            env["out"] = max(env["out"], dev(o3, ref_out)[0])
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
    if not slow_decay:
        # Realistic gates: stock's envelope is 0 and the crossing chunk is
        # bitwise in every co-batch.
        for k in env:
            assert worst[k][0] <= env[k], f"{k}: crossing {worst[k][0]} > {env[k]}"
        return
    # Gate ~1 (adversarial): the segment scan's bf16-operand rounding reaches
    # every later segment (kda_segscan); the crossing chunk's segmentation
    # differs from the split's, so it lands in the same rounding regime as
    # stock, bounded like the segmented route itself (test_kda_segscan:
    # 0.02 x scale vs plain FlashKDA).
    scale = {"b": ref_b, "final": ref_final, "out": ref_out}
    for k in env:
        bound = max(env[k], 0.02 * max(scale[k].float().abs().max().item(), 1.0))
        assert worst[k][0] <= bound, f"{k}: crossing {worst[k][0]} > {bound}"
