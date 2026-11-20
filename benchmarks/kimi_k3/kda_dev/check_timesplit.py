"""PA time-split check: two microbatches (each request cut at a chunk-aligned
point, mb2 seeded with mb1's final FP32 state) vs one unsplit call.

  python3 benchmarks/kimi_k3/kda_dev/check_timesplit.py 5461,5461,5462 2720,2720,2752
"""
import sys

import torch

from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

D, H = 128, 96


def call(q, k, v, g, beta, A_log, dt_bias, lens, cache):
    offs = [0, *torch.tensor(lens).cumsum(0).tolist()]
    cu_cpu = torch.tensor(offs, dtype=torch.int32)
    return sp_kda.persistent_prefill(
        q, k, v, g, beta, A_log, dt_bias, -5.0, cache, cu_cpu.cuda(), cu_cpu
    )


def main():
    lens = [int(x) for x in sys.argv[1].split(",")]
    cut = [int(x) for x in sys.argv[2].split(",")]
    torch.manual_seed(0)
    T, N = sum(lens), len(lens)
    bf = torch.bfloat16
    q, k, g = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
    v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
    beta = torch.randn(1, T, H, device="cuda").to(bf)
    A_log = torch.randn(H, device="cuda") * 0.5
    dt_bias = torch.randn(H * D, device="cuda") * 0.1
    cache0 = torch.randn(N, H, D, D, device="cuda") * 0.1
    o_full, s_full = call(q, k, v, g, beta, A_log, dt_bias, lens, cache0.clone())
    starts = [0, *torch.tensor(lens).cumsum(0).tolist()[:-1]]
    i1 = torch.cat([torch.arange(s, s + c) for s, c in zip(starts, cut)]).cuda()
    i2 = torch.cat(
        [torch.arange(s + c, s + n) for s, c, n in zip(starts, cut, lens)]
    ).cuda()
    sel = lambda x, idx: x[:, idx].contiguous()  # noqa: E731
    args1 = [sel(x, i1) for x in (q, k, v, g, beta)]
    args2 = [sel(x, i2) for x in (q, k, v, g, beta)]
    o1, s1 = call(*args1, A_log, dt_bias, cut, cache0.clone())
    rest = [n - c for n, c in zip(lens, cut)]
    o2, s2 = call(*args2, A_log, dt_bias, rest, s1.clone().float())
    o_ts = torch.empty_like(o_full)
    o_ts[:, i1] = o1
    o_ts[:, i2] = o2
    od = (o_ts.float() - o_full.float()).abs().max().item()
    sd = (s2.float() - s_full.float()).abs().max().item()
    print(
        f"timesplit lens={sys.argv[1]} cut={sys.argv[2]} out_bitequal="
        f"{torch.equal(o_ts, o_full)} state_bitequal={torch.equal(s2, s_full)} "
        f"out_maxdiff={od:.3e} state_maxdiff={sd:.3e} "
        f"out_absmax={o_full.float().abs().max().item():.3e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
