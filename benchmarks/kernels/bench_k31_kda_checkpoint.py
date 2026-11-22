# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer KDA prefill core time on checkpoint steps: the old route
(gathered states + FlashKDA with checkpoints + final-state row scatter) vs the
persistent core with checkpoint cuts, plus the persistent core without
checkpoints as the floor. Shapes: served DEP16 c512 rank steps (chunk lens),
one prompt-end checkpoint per finishing chunk.

  python benchmarks/kernels/bench_k31_kda_checkpoint.py --heads 96 \
      --steps <rep*.steps.jsonl> [--n 12]
"""

import argparse
import json

import torch

D = 128


def _time(fn, iters=20):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    for _ in range(iters):
        fn()
    ev[1].record()
    torch.accelerator.synchronize()
    return ev[0].elapsed_time(ev[1]) * 1e3 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=96)
    ap.add_argument("--steps", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from vllm.models.kimi_k3.nvidia import kda as nkda
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda
    import vllm._flashkda_C  # noqa: F401

    H = a.heads
    shapes = []
    for line in open(a.steps):
        r = json.loads(line)
        for rank in r["ranks"]:
            lens = [int(q) for q, _ in rank if q > 0]
            if len(lens) >= 1 and sum(lens) >= 512:
                shapes.append(lens)
    # spread over sizes
    shapes.sort(key=sum)
    pick = [shapes[i * (len(shapes) - 1) // max(a.n - 1, 1)] for i in range(a.n)]
    rows = []
    torch.manual_seed(0)
    for lens in pick:
        N, T = len(lens), sum(lens)
        # one checkpoint on the longest chunk, 16-aligned, ~1K before its end
        cuts = [0] * N
        i = max(range(N), key=lambda j: lens[j])
        if lens[i] > 64:
            cuts[i] = max(16, (lens[i] - 1024) // 16 * 16)
        bf = torch.bfloat16
        q, k, g = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
        v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
        beta = torch.randn(1, T, H, device="cuda").to(bf)
        A_log = torch.randn(H, device="cuda") * 0.5
        dt_bias = torch.randn(H * D, device="cuda") * 0.1
        offs = [0]
        for n in lens:
            offs.append(offs[-1] + n)
        cu_cpu = torch.tensor(offs, dtype=torch.int32)
        cu = cu_cpu.cuda()
        slots = 2 * N + 2
        cache = torch.randn(slots, H, D, D, device="cuda") * 0.1
        idx = (torch.arange(N, device="cuda", dtype=torch.int32) * 2 + 1)
        has = torch.ones(N, dtype=torch.bool, device="cuda")
        out = torch.empty(1, T, H, D, dtype=bf, device="cuda")
        fs = torch.empty(N, H, D, D, device="cuda")
        cs = torch.empty(N + 1, H, D, D, device="cuda")
        ws = torch.empty(
            torch.ops._flashkda_C.get_workspace_size(T, H, N), dtype=torch.uint8,
            device="cuda",
        )
        cuts_t = torch.tensor(cuts, dtype=torch.int32, device="cuda")

        def old():
            h0 = nkda.gather_initial_states(cache, idx, has)
            nkda._flashkda_prefill(
                q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias,
                lower_bound=-5.0, initial_state=h0, cu_seqlens=cu, out=out,
                final_state=fs, workspace=ws, checkpoint_state=cs[:N],
                checkpoint_offsets=cuts_t,
            )
            cache[idx.long()] = fs

        memo: dict = {}

        def new():
            sp_kda.persistent_prefill(
                q, k, v, g, beta, A_log, dt_bias, -5.0, None, cu, cu_cpu,
                state_cache=cache, state_indices=idx, has_initial_state=has,
                out=out, step_memo=memo, checkpoint_offsets=cuts, checkpoint_out=cs,
            )

        memo0: dict = {}

        def floor():
            sp_kda.persistent_prefill(
                q, k, v, g, beta, A_log, dt_bias, -5.0, None, cu, cu_cpu,
                state_cache=cache, state_indices=idx, has_initial_state=has,
                out=out, step_memo=memo0,
            )

        t_old, t_new, t_floor = _time(old), _time(new), _time(floor)
        row = dict(lens=lens, cuts=cuts, T=T, old_us=round(t_old, 1),
                   new_us=round(t_new, 1), nockpt_us=round(t_floor, 1))
        rows.append(row)
        print(json.dumps(row), flush=True)
    to = sum(r["old_us"] for r in rows)
    tn = sum(r["new_us"] for r in rows)
    tf = sum(r["nockpt_us"] for r in rows)
    print(f"TOTAL old {to:.0f} us new {tn:.0f} us no-ckpt {tf:.0f} us "
          f"(new/old {tn / to:.3f}, new/no-ckpt {tn / tf:.3f})")
    if a.out:
        json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
