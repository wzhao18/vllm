#!/bin/bash
# o_norm + MXFP8: unit test vs the stock Triton kernel, and its F16K / F8K time.
# usage: run_onorm.sh <overlay> <outdir>
OV=$1; OUT=$2; mkdir -p $OUT; cd $OV
timeout 600 python3 -m pytest -q -x --noconftest tests/models/kimi_k3/test_kda_o_norm_mxfp8.py 2>&1 | tail -n 3
timeout 600 python3 -u - <<'EOF'
import torch
from vllm.models.kimi_k3.nvidia.ops.mxfp8_act import Mxfp8Buffer
from vllm.models.kimi_k3.nvidia.specialized_prefill.kda import o_norm_mxfp8
H, D = 96, 128
torch.manual_seed(0)
for T in (16384, 8192):
    P = H * D
    x = (torch.randn(1, T, H, D, device="cuda") * 3).to(torch.bfloat16)
    proj = (torch.randn(T, 49376, device="cuda") * 2).to(torch.bfloat16)
    g = proj[:, 3 * P:4 * P]
    w = torch.rand(D, device="cuda") + 0.5
    out = Mxfp8Buffer(T, P, "cuda", torch.bfloat16)
    for _ in range(3):
        o_norm_mxfp8(x, g, w, 1e-6, out)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(20):
        o_norm_mxfp8(x, g, w, 1e-6, out)
    e1.record()
    torch.cuda.synchronize()
    us = e0.elapsed_time(e1) / 20 * 1000
    gb = (2 * T * P * 2 + T * P + T * P // 32) / 1e9
    print(f"o_norm_mxfp8 T={T} H={H}: {us:.1f} us ({gb / us * 1e3:.2f} TB/s); "
          f"q sum {out.data.view(torch.uint8).float().sum().item():.6e}", flush=True)
EOF
