"""NVFP4 FC2-input quantization error vs activation magnitude (unit global scale, the
K3.1 NVFP4 export's input_scale = 1.0) and with a power-of-two norm constant.
Run in the image with cwd = overlay: python3 fc2_underflow_probe.py"""
import torch

from vllm.models.kimi_k3.nvidia.megamoe_rubin import weights as W

torch.manual_seed(0)
dev = torch.device("cuda", 0)
n, inter = 4096, 3072
base = torch.randn(n, inter, device=dev)
# heavy-ish tail like SiTU(gate) * up: product of two gaussians
base = base * torch.randn(n, inter, device=dev)
base = base / base.pow(2).mean().sqrt()


def qerr(x, norm):
    q, sf = W.quantize_nvfp4_block16(x, norm)
    d = W.unpack_fp4_to_f32(q) * W.e4m3_to_f32(sf).repeat_interleave(16, dim=-1)
    d = d / norm
    zero_blocks = float((sf == 0).float().mean())
    return float((d - x).norm() / x.norm()), zero_blocks


print("rms        rel_err(norm=1) zero_blk(norm=1)  best_pow2_norm rel_err(best)")
for e in range(0, -16, -1):
    rms = 2.0**e
    x = base * rms
    r1, z1 = qerr(x, 1.0)
    amax = float(x.abs().amax())
    k = int(torch.floor(torch.log2(torch.tensor(6.0 * 448.0 / amax))))
    rb, _ = qerr(x, 2.0**k)
    print(f"2^{e:<4d} {rms:9.2e}  {r1:10.4f}  {z1:10.4f}        2^{k:<3d}        {rb:.4f}")
