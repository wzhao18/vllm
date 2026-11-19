# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 prefill sparse MLA (specialized_prefill, sm_107) against an FP32
reference over the same FP8 query / cache (the kernel's only rounding beyond the
reference is the stock E4M3 P), and against the stock trtllm-gen kernel."""

import math

import pytest
import torch

from vllm.platforms import current_platform

if not (current_platform.is_cuda() and current_platform.has_device_capability(107)):
    pytest.skip("needs sm_107 (Rubin)", allow_module_level=True)

from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla import (  # noqa: E402
    sparse_mla_prefill,
    sparse_mla_prefill_v1,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h2 import (  # noqa: E402
    sparse_mla_prefill_h2,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h4 import (  # noqa: E402
    sparse_mla_prefill_h4,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h5 import (  # noqa: E402
    sparse_mla_prefill_h5,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h6 import (  # noqa: E402
    sparse_mla_prefill_h6,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_sol import (  # noqa: E402
    sparse_mla_prefill_sol,
)

H, ROW, L = 96, 576, 512
E4M3 = torch.float8_e4m3fn


def _problem(lengths, pool_pages, width, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    t = len(lengths)
    kv = (torch.randn(pool_pages * 64, ROW, device="cuda", generator=g) * 0.8).to(E4M3)
    q8 = (torch.randn(t, H, ROW, device="cuda", generator=g) * 0.8).to(E4M3)
    slots = torch.full((t, width), -1, dtype=torch.int32, device="cuda")
    for i, n in enumerate(lengths):
        if n:
            sel = torch.randperm(pool_pages * 64, device="cuda", generator=g)[:n]
            slots[i, :n] = sel.to(torch.int32)
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    return q8, kv, slots, lens


def _reference(q8, kv, slots, lens, scale, out_scale):
    t = q8.shape[0]
    out = torch.zeros(t, H, L, device="cuda")
    kvf = kv.float()
    for i in range(t):
        n = int(lens[i])
        if n == 0:
            continue
        k = kvf[slots[i, :n].long()]  # [n, 576]
        s = q8[i].float() @ k.T * scale  # [96, n]
        p = torch.softmax(s, dim=-1)
        out[i] = p @ k[:, :L] * out_scale
    return out


@pytest.mark.parametrize(
    "loaders,pipelined,tma,regs,stages",
    [(3, False, False, (), 2), (3, True, False, (), 2), (7, False, True, (), 2),
     (7, True, True, (184, 72), 2), (7, True, True, (184, 72), 3)],
)
def test_matches_fp32_reference(loaders, pipelined, tma, regs, stages):
    lengths = [1, 2, 127, 128, 129, 300, 1000, 2047, 2048, 2048, 64, 255, 256, 2000]
    lengths = lengths * 23  # more rows than SMs: persistent CTAs take several
    q8, kv, slots, lens = _problem(lengths, pool_pages=600, width=2048, seed=0)
    scale, out_scale = 192**-0.5 * 0.7, 1.3
    out = sparse_mla_prefill_v1(q8, kv, 64, slots, lens, scale, out_scale,
                             loaders=loaders, pipelined=pipelined, tma=tma,
                             regs=regs, stages=stages)
    torch.cuda.synchronize()
    ref = _reference(q8, kv, slots, lens, scale, out_scale)
    err = (out.float() - ref).abs()
    rel = err.norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-6)
    assert torch.isfinite(out.float()).all()
    # E4M3 P (as the stock kernel) bounds the error; measured ~0.5-1% mean
    assert rel.mean().item() < 0.02, rel.mean().item()
    assert rel.max().item() < 0.08, rel.max().item()


def test_matches_stock_trtllm_kernel():
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    lengths = [2048] * 200 + [517] * 40
    q8, kv, slots, lens = _problem(lengths, pool_pages=1500, width=2048, seed=1)
    scale, out_scale = 192**-0.5, 1.0
    out = sparse_mla_prefill(q8, kv, 64, slots, lens, scale, out_scale)
    sol = sparse_mla_prefill_sol(q8, kv, slots, lens, scale, out_scale)
    h2 = sparse_mla_prefill_h2(q8, kv, slots, lens, scale, out_scale)
    t = q8.shape[0]
    qp = torch.zeros(t, 128, ROW, dtype=E4M3, device="cuda")
    qp[:, :H] = q8
    ref = torch.empty(t, 128, L, dtype=torch.bfloat16, device="cuda")
    ws = torch.zeros(1 << 28, dtype=torch.int8, device="cuda")
    trtllm_batch_decode_with_kv_cache_mla(
        query=qp.unsqueeze(1), kv_cache=kv.view(-1, 64, ROW).unsqueeze(1),
        workspace_buffer=ws, qk_nope_head_dim=128, kv_lora_rank=L, qk_rope_head_dim=64,
        block_tables=slots.unsqueeze(1), seq_lens=lens, max_seq_len=2048,
        out=ref.unsqueeze(1), bmm1_scale=scale, bmm2_scale=out_scale,
        sparse_mla_top_k=2048,
    )
    torch.cuda.synchronize()
    # both round P to E4M3 (differently scaled): ours must be no less precise
    exact = _reference(q8, kv, slots, lens, scale, out_scale)

    def err(x):
        d = (x.float() - exact).norm(dim=-1) / exact.norm(dim=-1).clamp(min=1e-6)
        return d.mean().item()

    ours, stock, ours_sol = err(out), err(ref[:, :H]), err(sol)
    assert not math.isnan(ours) and not math.isnan(ours_sol)
    assert ours <= 1.25 * stock + 1e-3, (ours, stock)
    assert ours_sol <= 1.25 * stock + 1e-3, (ours_sol, stock)
    ours_h2 = err(h2)
    assert ours_h2 <= 1.25 * stock + 1e-3, (ours_h2, stock)
    ours_h4 = err(sparse_mla_prefill_h4(q8, kv, slots, lens, scale, out_scale))
    assert ours_h4 <= 1.25 * stock + 1e-3, (ours_h4, stock)
    ours_h5 = err(sparse_mla_prefill_h5(q8, kv, slots, lens, scale, out_scale))
    assert ours_h5 <= 1.25 * stock + 1e-3, (ours_h5, stock)
    ours_h6 = err(sparse_mla_prefill_h6(q8, kv, slots, lens, scale, out_scale))
    assert ours_h6 <= 1.25 * stock + 1e-3, (ours_h6, stock)


@pytest.mark.parametrize("kernel", ["sol", "h2", "h4", "h4_tma", "h5", "h5_tma", "h5_split", "h5_stm", "h5_pp", "h6"])
@pytest.mark.parametrize("scale_mul", [0.7, 3.0])
def test_sol_matches_fp32_reference(scale_mul, kernel):
    """Gather-floor kernel; scale_mul 3 drives the lazy-max slow path often."""
    lengths = [1, 2, 127, 128, 129, 300, 1000, 2047, 2048, 2048, 64, 255, 256, 2000, 0]
    lengths = lengths * 23  # more rows than SMs: persistent CTAs take several
    q8, kv, slots, lens = _problem(lengths, pool_pages=600, width=2048, seed=0)
    scale, out_scale = 192**-0.5 * scale_mul, 1.3
    fn = {"sol": sparse_mla_prefill_sol, "h2": sparse_mla_prefill_h2,
          "h4": sparse_mla_prefill_h4, "h5": sparse_mla_prefill_h5,
          "h6": sparse_mla_prefill_h6}[kernel.split("_")[0]]
    knobs = {"tma": (0, 0, 1, 1), "split": (0, 0, 0, 0, 0, 0, 1),
             "stm": (0, 0, 2, 0, 0, 0, 1), "pp": (0, 0, 2, 0, 0, 0, 1, 1),
             "pp2": (0, 0, 2, 0, 0, 0, 1, 1, 0, 1)}.get(
        kernel.split("_")[-1], ())
    out = (fn(q8, kv, slots, lens, scale, out_scale, knobs=knobs) if knobs
           else fn(q8, kv, slots, lens, scale, out_scale))
    torch.cuda.synchronize()
    ref = _reference(q8, kv, slots, lens, scale, out_scale)
    err = (out.float() - ref).abs()
    rel = err.norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-6)
    assert torch.isfinite(out.float()).all()
    zero = lens == 0
    assert (out[zero].float() == 0).all()
    rel = rel[~zero]
    assert rel.mean().item() < 0.02, rel.mean().item()
    assert rel.max().item() < 0.08, rel.max().item()
