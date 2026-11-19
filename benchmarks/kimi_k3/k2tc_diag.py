# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dev diagnostic: tcgen05 KDA recurrence (k3_flashkda_split mode bit 5) vs the
in-tree FlashKDA K2 (same K1 workspace). Prints error stats per kr-descriptor
variant (mode bit 6) and per-kernel times at the deployed DEP16 F16K shape.

usage: python k2tc_diag.py [--modes 35,99] [--time]
"""

import argparse

import torch


def inputs(lens, H, seed=7, slow_decay=False):
    torch.manual_seed(seed)
    T, D = sum(lens), 128
    dev = "cuda"
    q, k, v, g = (
        torch.randn(1, T, H, D, dtype=torch.bfloat16, device=dev) for _ in range(4)
    )
    if slow_decay:
        q = torch.nn.functional.normalize(q.float(), dim=-1).to(torch.bfloat16)
        k = torch.nn.functional.normalize(k.float(), dim=-1).to(torch.bfloat16)
        g = torch.full_like(g, -60.0)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32,
                      device=dev)
    return dict(
        q=q, k=k, v=v, g=g,
        beta=torch.randn(1, T, H, dtype=torch.bfloat16, device=dev),
        A_log=torch.randn(H, dtype=torch.float32, device=dev) * 0.5,
        dt_bias=torch.randn(H, D, dtype=torch.float32, device=dev) * 0.1,
        lower_bound=-3.0,
        initial_state=torch.randn(len(lens), H, D, D, dtype=torch.float32,
                                  device=dev) * 0.1,
        cu_seqlens=cu,
    )


def run(x, mode, ws=None, sync=True):
    _, T, H, D = x["q"].shape
    N = x["initial_state"].shape[0]
    out = torch.zeros_like(x["v"])
    final = torch.zeros_like(x["initial_state"])
    if ws is None:
        ws = torch.empty(torch.ops.k3_flashkda_split.get_workspace_size(T, H, N),
                         dtype=torch.uint8, device="cuda")
    torch.ops.k3_flashkda_split.fwd(
        x["q"], x["k"], x["v"], x["g"], x["beta"], D**-0.5, out, ws,
        x["A_log"], x["dt_bias"], x["lower_bound"], x["initial_state"], final,
        x["cu_seqlens"], None, None, mode)
    if sync:
        torch.cuda.synchronize()
    return out, final


def stats(name, ref, got):
    ref, got = ref.float(), got.float()
    d = (ref - got).abs()
    fin = torch.isfinite(got).all().item()
    rel = d.max().item() / max(ref.abs().max().item(), 1e-30)
    neq = (d > 0).float().mean().item()
    print(f"    {name}: finite={fin} max_abs={d.max().item():.3e} rel={rel:.3e} "
          f"frac_ne={neq:.4f} ref_max={ref.abs().max().item():.3e}")
    return fin and rel < 2e-2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="35,99")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--stamps", default="")
    ap.add_argument("--wall", default="", help="modes to time end to end")
    args = ap.parse_args()
    from vllm.models.kimi_k3.nvidia.ops import kda_segscan

    kda_segscan.load_extension()
    modes = [int(m) for m in args.modes.split(",")]
    for lens, H, slow in [([16], 1, False), ([40, 23], 2, False),
                          ([300, 17, 129], 8, False), ([200, 64], 4, True),
                          ([5461, 5461, 5462], 96, False)]:
        x = inputs(lens, H, slow_decay=slow)
        ref_o, ref_s = run(x, 3)
        print(f"lens={lens} H={H} slow={slow}")
        for m in modes:
            o, s = run(x, m)
            ok1 = stats(f"mode {m} out  ", ref_o, o)
            ok2 = stats(f"mode {m} state", ref_s, s)
            print(f"    mode {m}: {'OK' if ok1 and ok2 else 'MISMATCH'}")
    for m in [int(x) for x in args.wall.split(",") if x]:
        if m & 512:
            for lens_s, H_s in [([40, 23], 2), ([300, 17, 129], 8)]:
                xs = inputs(lens_s, H_s)
                mc_s = max((n + 15) // 16 for n in lens_s)
                T_s = sum(lens_s)
                ws_s = torch.zeros(
                    torch.ops.k3_flashkda_split.get_workspace_size(T_s, H_s, len(lens_s)),
                    dtype=torch.uint8, device="cuda")
                r_o, r_s = run(xs, 3)
                o_s, s_s = run(xs, m | (mc_s << 16), ws_s)
                ok = stats(f"small {lens_s} mode {m} out", r_o, o_s) and stats(
                    f"small {lens_s} mode {m} state", r_s, s_s)
                print(f"    small {lens_s} mode {m}: {'OK' if ok else 'MISMATCH'}",
                      flush=True)
        lens = [5461, 5461, 5462]
        x = inputs(lens, 96)
        T = sum(lens)
        mc = max((n + 15) // 16 for n in lens)
        mm = m | (mc << 16) if m & 512 else m
        ws = torch.zeros(torch.ops.k3_flashkda_split.get_workspace_size(T, 96, 3),
                         dtype=torch.uint8, device="cuda")
        if m == 0:
            def f():
                out = torch.empty_like(x["v"])
                fin = torch.empty_like(x["initial_state"])
                torch.ops._flashkda_C.fwd(
                    x["q"], x["k"], x["v"], x["g"], x["beta"], 128**-0.5, out, ws,
                    x["A_log"], x["dt_bias"], x["lower_bound"], x["initial_state"],
                    fin, x["cu_seqlens"], None, None)
        else:
            def f():
                run(x, mm, ws, sync=False)
        ref_o, ref_s = run(x, 3)
        if m:
            o, s_ = run(x, mm, ws)
            ok = stats(f"wall mode {m} out", ref_o, o) and stats(
                f"wall mode {m} state", ref_s, s_)
            print(f"    wall mode {m}: {'OK' if ok else 'MISMATCH'}")
        import vllm._flashkda_C  # noqa: F401
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(args.iters):
            f()
        e1.record()
        torch.cuda.synchronize()
        print(f"WALL mode {m} ({'stock _flashkda_C' if m == 0 else mm}): "
              f"{e0.elapsed_time(e1) * 1000 / args.iters:.1f} us per K1+K2 call",
              flush=True)
    for m in [int(x) for x in args.stamps.split(",") if x]:
        x = inputs([5461, 5461, 5462], 96)
        run(x, m)
        _, fin = run(x, m | 256)
        st = fin[2, 0].reshape(-1)[:512].view(torch.int64).reshape(32, 8).cpu()
        d = st - st[:, :1]
        per = (st[1:, 0] - st[:-1, 0]).float().mean().item()
        print(f"mode {m} stamps (clk rel. to full-wait; math: d1 U_rdy d3 s_rdy end; "
              f"mma: s_rdy d1_commit), chunk period {per:.0f} clk")
        for r in range(0, 32, 8):
            print("   ", d[r].tolist())
        print("    mean", d[:, 1:].float().mean(0).round().tolist())
    if not args.time:
        return
    from torch.profiler import ProfilerActivity, profile

    lens = [5461, 5461, 5462]
    x = inputs(lens, 96)
    T = sum(lens)
    ws = torch.empty(torch.ops.k3_flashkda_split.get_workspace_size(T, 96, 3),
                     dtype=torch.uint8, device="cuda")
    for m in [3] + modes:
        for _ in range(3):
            run(x, m, ws)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                run(x, m, ws)
        acc = {}
        for e in prof.events():
            if e.device_type.name == "CUDA" and e.device_time_total > 0:
                acc[e.name] = acc.get(e.name, 0.0) + e.device_time_total
        print(f"mode {m} F16K:")
        for n, us in sorted(acc.items(), key=lambda kv: -kv[1]):
            print(f"   {us / args.iters:9.1f} us  {n[:90]}")


if __name__ == "__main__":
    main()
