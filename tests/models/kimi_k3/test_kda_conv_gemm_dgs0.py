# SPDX-License-Identifier: Apache-2.0
"""DG-SOL probe S0 (EXPERIMENT ONLY): the stamped KDA conv GEMM is bitwise the
stock result at the TP8 production shape, and its stamp records are sane.
Also prints a 1-GPU stamp summary (stage, boundary, conv pass, tail) and the
stamps-on vs stamps-off kernel time (CUDA events)."""

import os

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)), reason="sm_107 only"
)

PART, K = 1536, 7168


def _operands(m, n, k, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randint(-2, 3, (m, k), device=dev, generator=g).float()
    w = torch.randint(-2, 3, (n, k), device=dev, generator=g).float()
    sx = torch.randint(118, 122, (((m + 127) // 128) * 128 * k // 32,), device=dev,
                       generator=g).to(torch.uint8)  # fmt: skip
    sw = torch.randint(117, 121, (n * k // 32,), device=dev, generator=g).to(torch.uint8)
    return x.to(torch.float8_e4m3fn), sx, w.to(torch.float8_e4m3fn), sw


@pytest.mark.parametrize("m", [16384, 32768])
def test_dgs0_stamped_bitwise_and_sane(m):
    from vllm.models.kimi_k3.nvidia.ops import kda_conv
    from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_conv_in_proj import tactic_for
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import dgs0
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        KdaConvGemm,
    )

    dev = torch.device("cuda")
    n = 3 * PART
    xq, xs, wq, ws = _operands(m, n, K, dev, 0)
    proj = torch._scaled_mm(
        xq, wq.t(),
        scale_a=xs.view(torch.float8_e8m0fnu), scale_b=ws.view(torch.float8_e8m0fnu),
        out_dtype=torch.bfloat16,
    )  # fmt: skip
    g = torch.Generator(device=dev).manual_seed(1)
    cw = (torch.randn(n, 4, device=dev, generator=g) * 0.5).contiguous()
    nseq = 8
    qsl = torch.arange(0, m + 1, m // nseq, dtype=torch.int32, device=dev)
    lines = torch.arange(nseq, dtype=torch.int32, device=dev) + 1
    has_init = torch.arange(nseq, device=dev) % 2 == 1
    state0 = torch.randn(nseq + 2, n, 4, device=dev, generator=g)
    state0 = state0.to(torch.bfloat16).to(torch.float32)[:, :, :3]
    st_ref = state0.clone()
    ref = kda_conv.qkv_conv(proj, cw, st_ref, qsl, lines, has_init, PART)

    gemm = KdaConvGemm(wq, ws, cw, PART, tactic=tactic_for(n))
    rec = torch.zeros((dgs0.MAXCTA, dgs0.REC), dtype=torch.int32, device=dev)
    outs = {}
    for name, stamps in (("off", None), ("on", rec)):
        st = state0.clone()
        out, _ = gemm(xq, xs, st, qsl, lines, has_init, null_block_id=NULL_BLOCK_ID,
                      stamps=stamps)  # fmt: skip
        torch.cuda.synchronize()
        assert torch.equal(out, ref), name
        assert torch.equal(st, st_ref), name
        outs[name] = out
    r = rec.cpu().numpy().astype("int64") & 0xFFFFFFFF
    live = r[:, dgs0.W_ENTRY_GT] != 0
    nlive = int(live.sum())
    assert nlive > 0
    lead = live & (r[:, dgs0.W_LEADER] == 1)
    assert int(lead.sum()) * 2 == nlive
    ktiles = r[live, dgs0.W_KTILES]
    assert (ktiles == K // 128).all()
    # every CTA ran the same number of MMA and epilogue tiles
    assert (r[live, dgs0.W_MMA_TILES] == r[live, dgs0.W_EPI_TILES]).all()

    # timing: stamps off vs on (CUDA events, 30 calls each, ABAB)
    st = state0.clone()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    res = {"off": [], "on": []}
    for _ in range(3):
        for name, stamps in (("off", None), ("on", rec), ("on", rec), ("off", None)):
            ev[0].record()
            for _ in range(10):
                gemm(xq, xs, st, qsl, lines, has_init, null_block_id=NULL_BLOCK_ID,
                     stamps=stamps)  # fmt: skip
            ev[1].record()
            torch.cuda.synchronize()
            res[name].append(ev[0].elapsed_time(ev[1]) * 100.0)  # us per call
    print(f"DGS0 1GPU m={m} ctas={nlive} tiles/cta={sorted(set(r[live, dgs0.W_MMA_TILES]))} "
          f"call_us off={min(res['off']):.1f} on={min(res['on']):.1f} (incl. conv_fixup)")
    out_dir = os.environ.get("DGS0_DIR")
    if out_dir:
        import numpy as np

        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, f"onegpu_m{m}.rank0.npz"),
                            L0=rec.cpu().numpy())  # fmt: skip
