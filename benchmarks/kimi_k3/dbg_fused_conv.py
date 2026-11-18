# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug: where the fused-conv prefill differs from qkv_conv + FlashKDA."""

import sys

import torch

sys.path.insert(0, sys.argv[1] + "/tests/models/kimi_k3")
import test_kda_prefill_fused_conv as t  # noqa: E402

from vllm.models.kimi_k3.nvidia.ops import kda_segscan  # noqa: E402
from vllm.models.kimi_k3.nvidia.ops.kda_conv import qkv_conv  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_prefill.kda import (  # noqa: E402
    fused_conv_prefill,
)

D = 128
for seqlens, heads in [([300, 40], 8), ([5461, 5461, 5462], 8), ([3, 64, 1000], 96)]:
    kda_segscan.load_extension()
    (mixed, beta, g, w, cs, ci, hi, cu, A_log, dt_bias, init) = t._inputs(
        seqlens, heads, torch.bfloat16
    )
    T, N, P = sum(seqlens), len(seqlens), heads * D
    ws = torch.empty(kda_segscan.workspace_size(T, heads, N), dtype=torch.uint8,
                     device="cuda")
    cs_ref = cs.clone()
    q, k, v = (x.view(1, T, heads, D) for x in qkv_conv(mixed, w, cs_ref, cu, ci, hi, P))
    out_ref = torch.empty(1, T, heads, D, dtype=torch.bfloat16, device="cuda")
    fs_ref = torch.empty_like(init)
    torch.ops.k3_flashkda_split.fwd(q, k, v, g, beta, D**-0.5, out_ref, ws, A_log,
                                    dt_bias.view(heads, D), -5.0, init, fs_ref, cu,
                                    None, None, 3)
    cs_new = cs.clone()
    out = torch.empty_like(out_ref)
    fs = torch.empty_like(init)
    vb = torch.empty_like(out_ref)
    fused_conv_prefill(mixed_qkv=mixed, conv_weight=w, conv_state=cs_new,
                       cache_indices=ci, has_initial_state=hi, null_block_id=-1,
                       g=g, beta=beta, A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
                       initial_state=init, cu_seqlens=cu, out=out, final_state=fs,
                       workspace=ws, v_buf=vb)
    torch.accelerator.synchronize()
    dv = (vb != v)[0]
    print(seqlens, heads, "v mismatches", int(dv.sum()), "of", dv.numel())
    if dv.any():
        idx = dv.nonzero()
        print("  first", idx[:5].tolist(), "rows hit", idx[:, 0].unique()[:20].tolist())
        print("  heads hit", idx[:, 1].unique()[:20].tolist())
    if dv.any():
        hz = torch.zeros_like(hi)
        cz = cs.clone()
        vz = qkv_conv(mixed, w, cz, cu, ci, hz, P)[2].view(1, T, heads, D)
        h1 = int(idx[0, 1])
        print("  head", h1, "rows0-2 == zero-history conv:",
              torch.equal(vb[0, :3, h1], vz[0, :3, h1]),
              "fused", vb[0, 0, h1, :4].tolist(), "ref", v[0, 0, h1, :4].tolist(),
              "zero-hist", vz[0, 0, h1, :4].tolist())
    print("  state eq", torch.equal(cs_new, cs_ref), "fs eq", torch.equal(fs, fs_ref),
          "out eq", torch.equal(out, out_ref),
          "out max diff", (out.float() - out_ref.float()).abs().max().item())
