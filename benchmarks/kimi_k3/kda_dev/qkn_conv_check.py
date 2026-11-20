"""qkv_conv(normalize_qk=True) vs l2norm of qkv_conv(normalize_qk=False) (q/k planes), v plane equal."""
import torch

from vllm.models.kimi_k3.nvidia.ops import kda_conv

torch.manual_seed(0)
for T, H, lens in ((4096, 8, (4096,)), (6000, 96, (100, 2900, 3000))):
    part = H * 128
    x = torch.randn(T, 3 * part, device="cuda").to(torch.bfloat16)
    w = torch.randn(3 * part, 4, device="cuda") * 0.5
    N = len(lens)
    qsl = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    ci = torch.arange(N, dtype=torch.int32, device="cuda")
    his = torch.zeros(N, dtype=torch.bool, device="cuda")
    outs = []
    for norm in (False, True):
        st = torch.zeros(N, 3, 3 * part, device="cuda").transpose(1, 2)  # (lines, 3C, 3) strided
        outs.append(kda_conv.qkv_conv(x, w, st, qsl, ci, his, part, None, normalize_qk=norm))
    raw, nrm = outs
    ref = raw[:2].float().view(2, T, H, 128)
    ref = (ref * torch.rsqrt(ref.square().sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
    d = (nrm[:2].view(2, T, H, 128).float() - ref.float()).abs()
    ulp = (d / ref.float().abs().clamp_min(1e-3)).max().item()
    print(f"qkn_conv T={T} H={H} qk maxabs {d.max().item():.3e} maxrel {ulp:.3e} "
          f"v_equal {torch.equal(nrm[2], raw[2])} norms {nrm[:2].float().view(2, T, H, 128).square().sum(-1).sqrt().mean().item():.4f}")
