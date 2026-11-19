import torch
from vllm.models.kimi_k3.nvidia.ops.mxfp8_act import Mxfp8Buffer
from vllm.models.kimi_k3.nvidia.specialized_prefill.kda import o_norm_mxfp8
H, D, T = 96, 128, 16384
torch.manual_seed(0)
P = H * D
x = (torch.randn(1, T, H, D, device="cuda") * 3).to(torch.bfloat16)
proj = (torch.randn(T, 49376, device="cuda") * 2).to(torch.bfloat16)
g = proj[:, 3 * P:4 * P]
w = torch.rand(D, device="cuda") + 0.5
out = Mxfp8Buffer(T, P, "cuda", torch.bfloat16)
for _ in range(3):
    o_norm_mxfp8(x, g, w, 1e-6, out)
torch.cuda.synchronize()
print("done", out.data.view(torch.uint8).float().sum().item())
