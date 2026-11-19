"""kp64 (chunk-64 WY prepare for kc64 v3, no conv) dev checks.

--check  images vs the torch prep64 + images() reference, then kp64 -> kc64 v3
         vs stock FlashKDA chunk-16 on the same post-conv inputs
--time   F16K (3 x 5461, H 96) kp64 time
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k64_diag as k64  # noqa: E402
import kc64_diag as kc  # noqa: E402

D, C64 = 128, 64
V2 = False  # --v2: prep_chunk standalone (null_block_id -2)
bf = torch.bfloat16


def inputs(lens, H, seed=5):
    torch.manual_seed(seed)
    T, dev = sum(lens), "cuda"
    q, k, v, g = (torch.randn(1, T, H, D, dtype=bf, device=dev) for _ in range(4))
    beta = torch.randn(1, T, H, dtype=bf, device=dev)
    A_log = torch.randn(H, device=dev) * 0.5
    dt_bias = torch.randn(H, D, device=dev) * 0.1
    return q, k, v, g, beta, A_log, dt_bias


def kp64(q, k, v, g, beta, A_log, dt_bias, lb, scale, lens, dbg=None, x=None, out=None):
    T, H = q.shape[1], q.shape[2]
    dev = q.device
    C = sum((n + C64 - 1) // C64 for n in lens)
    x = x if x is not None else torch.cat([q[0].reshape(T, H * D), k[0].reshape(T, H * D), v[0].reshape(T, H * D)], 1)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
    e = torch.empty(0, device=dev)
    ei = torch.empty(0, dtype=torch.int32, device=dev)
    eb = torch.empty(0, dtype=torch.bool, device=dev)
    out = out if out is not None else dict(wq_img3=torch.zeros(H, C, 16, 128, 8, dtype=bf, device=dev),
               kr_img=torch.zeros(H, C, 8, 128, 8, dtype=bf, device=dev),
               mqk_img=torch.zeros(H, C, 8, 64, 8, dtype=bf, device=dev),
               gt=torch.zeros(H, C, D, device=dev),
               u_img3=torch.zeros(H, C, 8, 128, 8, dtype=bf, device=dev), C=C)
    dbg = torch.zeros(1, dtype=torch.int64, device=dev) if dbg is None else dbg
    torch.ops.k3_flashkda_split.prep64_tc(
        x, e, e, ei, eb, -2 if V2 else -1, g[0], beta[0], A_log, dt_bias, lb, scale, cu,
        out["wq_img3"], out["kr_img"], torch.empty(0, dtype=bf, device=dev), out["mqk_img"],
        out["gt"], out["u_img3"], torch.empty(0, dtype=bf, device=dev), dbg)
    return out, cu, x


def check(lens, H):
    lb, scale = -5.0, D ** -0.5
    q, k, v, g, beta, A_log, dt_bias = inputs(lens, H)
    T, N = sum(lens), len(lens)
    img, cu, _ = kp64(q, k, v, g, beta, A_log, dt_bias, lb, scale, lens)
    torch.cuda.synchronize()
    p = k64.prep64(q, k, g, beta, A_log, dt_bias, lb, lens, scale)
    ref = kc.images(p, v[0], lens, H)
    print(f"kp64 images lens={lens} H={H}: " + " | ".join(
        f"{n} {kc.err(img[n], ref[n])}" for n in ("wq_img3", "u_img3", "kr_img", "mqk_img", "gt")), flush=True)
    s0 = (torch.randn(N, H, D, D, device="cuda") * 0.1).to(bf).float()
    kc.V3 = True
    o, s, dbg = kc.run(img, s0, cu, T, H)
    torch.cuda.synchronize()
    if dbg[0].item() != 0:
        print(f"WATCHDOG {dbg[0].item():#x}")
    out_ref = torch.zeros_like(v)
    fin_ref = torch.zeros_like(s0)
    ws = torch.empty(torch.ops.k3_flashkda_split.get_workspace_size(T, H, N), dtype=torch.uint8, device="cuda")
    torch.ops.k3_flashkda_split.fwd(q, k, v, g, beta, scale, out_ref, ws, A_log, dt_bias, lb, s0,
                                    fin_ref, cu, None, None, 3)
    print(f"kp64 + kc64 v3 vs FlashKDA chunk-16 lens={lens} H={H}: out {kc.err(o, out_ref[0])} | "
          f"state {kc.err(s, fin_ref)} | nan {torch.isnan(o).any().item()}", flush=True)


def time_it(lens, H, iters=20):
    lb, scale = -5.0, D ** -0.5
    q, k, v, g, beta, A_log, dt_bias = inputs(lens, H)
    img, _, x = kp64(q, k, v, g, beta, A_log, dt_bias, lb, scale, lens)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        kp64(q, k, v, g, beta, A_log, dt_bias, lb, scale, lens, x=x, out=img)
    e1.record()
    torch.cuda.synchronize()
    print(f"kp64 time lens={lens} H={H}: {e0.elapsed_time(e1) * 1e3 / iters:.1f} us/call", flush=True)


def phases(lens, H):
    lb, scale = -5.0, D ** -0.5
    q, k, v, g, beta, A_log, dt_bias = inputs(lens, H)
    C = sum((n + C64 - 1) // C64 for n in lens)
    dbg = torch.zeros(2 * C * 16, dtype=torch.int64, device="cuda")
    img, _, x = kp64(q, k, v, g, beta, A_log, dt_bias, lb, scale, lens, dbg=dbg)
    kp64(q, k, v, g, beta, A_log, dt_bias, lb, scale, lens, dbg=dbg, x=x, out=img)
    torch.cuda.synchronize()
    st = dbg.view(2 * C, 16)[:C].cpu().double()
    names = [(0, 1, "load+l2norm"), (1, 2, "gate"), (2, 3, "cumsum"), (3, 4, "images+operands"),
             (4, 5, "A/Mq mma"), (5, 6, "INV diag"), (6, 12, "INV offdiag"), (12, 13, "INVb"),
             (13, 14, "w|u mma+store"), (0, 14, "total")]
    print("kp64 phases (ns, mean over head-0 CTAs): " + ", ".join(
        f"{n} {(st[:, b] - st[:, a]).mean().item():.0f}" for a, b, n in names), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--phases", action="store_true")
    ap.add_argument("--v2", action="store_true")
    a = ap.parse_args()
    global V2
    V2 = a.v2
    from vllm.models.kimi_k3.nvidia.ops import kda_segscan

    kda_segscan.load_extension()
    if a.check:
        check([130, 64, 257], 4)
        check([700, 1000], 8)
    if a.time:
        time_it([5461, 5461, 5462], 96)
    if a.phases:
        phases([5461, 5461, 5462], 96)


if __name__ == "__main__":
    main()
