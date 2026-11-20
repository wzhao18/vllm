# P-KDA2 dev: compile + run one persistent-KDA route once (for cubin dumps)
import json, sys
import torch
import vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent.forward as fwd_mod
from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda
route, dev = sys.argv[1], json.loads(sys.argv[2] if len(sys.argv) > 2 else "{}")
fwd_mod._DEV = dev
D, H, lens = 128, 96, [5461, 5461, 5462]
T, N = sum(lens), len(lens)
cu_cpu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32); cu = cu_cpu.cuda()
bf = torch.bfloat16
q, k, v = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
beta = torch.randn(1, T, H, device="cuda").to(bf); A_log = torch.randn(H, device="cuda") * 0.5
dt = torch.randn(H * D, device="cuda") * 0.1; s0 = torch.randn(N, H, D, D, device="cuda") * 0.1
fa = torch.randn(T, D, device="cuda").to(bf); w = (torch.randn(H * D, D, device="cuda") * D**-0.5).to(bf)
g = torch.matmul(fa, w.t()).view(1, T, H, D)
if route == "g":
    sp_kda.persistent_prefill(q, k, v, g, beta, A_log, dt, -5.0, s0, cu, cu_cpu)
else:
    sp_kda.persistent_prefill(q, k, v, None, beta, A_log, dt, -5.0, s0, cu, cu_cpu, f_a=fa, f_b_weight=w)
torch.cuda.synchronize(); print("compiled", route, dev)
