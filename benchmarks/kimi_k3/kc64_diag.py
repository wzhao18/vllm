"""kc64_rec (chunk-64 KDA state chain, FP32 state in tensor memory) dev checks.

--check   small shapes: kernel vs torch emulation of the same chain (bf16 at the
          same points) and vs stock FlashKDA on the same post-conv inputs
--time    F16K (3 x 5461, H 96) kernel time + per-CTA spans (1- vs 2-CTA SMs)
--stamps  chain-step anatomy (issuer clock stamps, CTA 0)
"""

import argparse
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k64_diag as k64  # noqa: E402

C64, D = 64, 128
bf = torch.bfloat16
V3 = False  # --v3: all-.ts chain kernel (flags bit 4), u image [8 tg][128 v][8 t]


def err(a, b):
    a, b = a.float(), b.float()
    return f"rel-rms {((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)).item():.3e}"


def chunk_rows(x, lens, H):
    """(T, H, D) -> (H, C, 64, D) zero padded per sequence."""
    nch = [(n + C64 - 1) // C64 for n in lens]
    out = torch.zeros(H, sum(nch), C64, x.shape[-1], dtype=x.dtype, device=x.device)
    tok = ci = 0
    for n, kk in zip(lens, nch):
        for c in range(kk):
            valid = min(C64, n - c * C64)
            out[:, ci, :valid] = x[tok:tok + valid].transpose(0, 1)
            tok += valid
            ci += 1
    return out


def images(p, v, lens, H):
    """FLA-form images from the prep64 reference: w = INV (beta kd), u = INV (beta v)."""
    C = p["C"]
    beta = chunk_rows(p["beta"].t().contiguous()[..., None], lens, H)[..., 0].float()  # (H,C,64)
    vc = chunk_rows(v, lens, H).float()
    inv = p["inv"].float()
    w = (inv @ (beta[..., None] * p["kd"].float())).to(bf)
    u = (inv @ (beta[..., None] * vc)).to(bf)
    x = dict(w=w, u=u, qd=p["qd"], kr=p["kr"], mqk=p["mqk"], gt=p["gt"], C=C)
    x["wq_img"] = (torch.cat([w, p["qd"]], 2).view(H, C, 2 * C64, 16, 8).permute(0, 1, 3, 2, 4).contiguous())
    x["u_img"] = u.contiguous()
    x["wq_img3"] = (torch.cat([p["qd"], w], 2).view(H, C, 2 * C64, 16, 8).permute(0, 1, 3, 2, 4).contiguous())
    x["u_img3"] = u.view(H, C, 8, 8, D).permute(0, 1, 2, 4, 3).contiguous()
    x["kr_img"] = p["kr"].transpose(-1, -2).reshape(H, C, D, 8, 8).permute(0, 1, 3, 2, 4).contiguous()
    x["mqk_img"] = p["mqk"].view(H, C, C64, 8, 8).permute(0, 1, 3, 2, 4).contiguous()
    return x


def emulate(x, s0, lens, H):
    """Torch chain with the kernel's rounding points; s0 (N,H,V,K)."""
    outs, fins = [], []
    ci = 0
    for n_i, n in enumerate(lens):
        S = s0[n_i].transpose(-1, -2).float().clone()  # (H, K, V)
        o_seq = []
        for c in range((n + C64 - 1) // C64):
            valid = min(C64, n - c * C64)
            Sb = S.to(bf).float()
            W = x["w"][:, ci].float() @ Sb
            vnew = (x["u"][:, ci].float() - W).to(bf).float()
            O = x["qd"][:, ci].float() @ Sb + x["mqk"][:, ci].float() @ vnew
            o_seq.append(O.to(bf)[:, :valid])
            S = x["gt"][:, ci][..., None] * S + x["kr"][:, ci].float().transpose(-1, -2) @ vnew
            ci += 1
        outs.append(torch.cat(o_seq, 1).transpose(0, 1))
        fins.append(S.transpose(-1, -2))
    return torch.cat(outs, 0), torch.stack(fins)


def run(x, s0, cu, T, H, flags=0, dbg=None, out=None, fin=None):
    N = cu.numel() - 1
    out = torch.empty(T, H, D, dtype=bf, device="cuda") if out is None else out
    fin = torch.empty(N, H, D, D, device="cuda") if fin is None else fin
    dbg = torch.zeros(1024 + 4 * H * N + 64, dtype=torch.int64, device="cuda") if dbg is None else dbg
    if V3:
        flags |= 16
    torch.ops.k3_flashkda_split.kc64_rec(x["wq_img3" if V3 else "wq_img"], x["u_img3" if V3 else "u_img"], x["kr_img"], x["mqk_img"], x["gt"],
                                         s0, fin, out, cu, flags, dbg)
    return out, fin, dbg


def check(lens, H):
    torch.manual_seed(5)
    T, dev = sum(lens), "cuda"
    q, k, v, g = (torch.randn(1, T, H, D, dtype=bf, device=dev) for _ in range(4))
    beta = torch.randn(1, T, H, dtype=bf, device=dev)
    A_log = torch.randn(H, device=dev) * 0.5
    dt_bias = torch.randn(H, D, device=dev) * 0.1
    lb, scale = -5.0, D ** -0.5
    N = len(lens)
    s0 = (torch.randn(N, H, D, D, device=dev) * 0.1).to(bf).float()
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
    p = k64.prep64(q, k, g, beta, A_log, dt_bias, lb, lens, scale)
    x = images(p, v[0], lens, H)
    o, s, dbg = run(x, s0, cu, T, H)
    torch.cuda.synchronize()
    if dbg[0].item() != 0:
        print(f"WATCHDOG {dbg[0].item():#x}")
    oe, se = emulate(x, s0, lens, H)
    print(f"check lens={lens} H={H} kernel vs emulation: out {err(o, oe)} | state {err(s, se)} | "
          f"nan {torch.isnan(o).any().item()}", flush=True)
    out_ref = torch.zeros_like(v)
    fin_ref = torch.zeros_like(s0)
    ws = torch.empty(torch.ops.k3_flashkda_split.get_workspace_size(T, H, N), dtype=torch.uint8, device=dev)
    torch.ops.k3_flashkda_split.fwd(q, k, v, g, beta, scale, out_ref, ws, A_log, dt_bias, lb, s0,
                                    fin_ref, cu, None, None, 3)
    print(f"check lens={lens} H={H} kernel vs FlashKDA chunk-16: out {err(o, out_ref[0])} | "
          f"state {err(s, fin_ref)} | emulation vs FlashKDA out {err(oe, out_ref[0])}", flush=True)


def rand_images(lens, H, dev="cuda"):
    C = sum((n + C64 - 1) // C64 for n in lens)
    r = lambda *s: (torch.randn(*s, device=dev) * 0.05).to(bf)  # noqa: E731
    return dict(wq_img=r(H, C, 16, 128, 8), wq_img3=r(H, C, 16, 128, 8), u_img=r(H, C, 64, 128), u_img3=r(H, C, 8, 128, 8), kr_img=r(H, C, 8, 128, 8),
                mqk_img=r(H, C, 8, 64, 8), gt=torch.rand(H, C, D, device=dev) * 0.5 + 0.5, C=C)


def time_it(lens, H, iters=20, flags=0):
    T = sum(lens)
    N = len(lens)
    x = rand_images(lens, H)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    s0 = torch.zeros(N, H, D, D, device="cuda")
    out = torch.empty(T, H, D, dtype=bf, device="cuda")
    fin = torch.empty_like(s0)
    dbg = torch.zeros(1024 + 4 * H * N + 64, dtype=torch.int64, device="cuda")
    for _ in range(3):
        run(x, s0, cu, T, H, flags, dbg, out, fin)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        run(x, s0, cu, T, H, flags, dbg, out, fin)
    e1.record()
    torch.cuda.synchronize()
    print(f"time flags={flags} lens={lens} H={H}: {e0.elapsed_time(e1) * 1e3 / iters:.1f} us/call (events)", flush=True)
    dbg.zero_()
    run(x, s0, cu, T, H, 8 | flags, dbg, out, fin)
    torch.cuda.synchronize()
    if dbg[0].item() != 0:
        print(f"WATCHDOG {dbg[0].item():#x}")
    cta = dbg[1024:1024 + 4 * H * N].view(-1, 4).cpu()
    t0 = cta[:, 0].min().item()
    per_sm = defaultdict(list)
    for st, en, sm, _ in cta.tolist():
        per_sm[sm].append(((st - t0) / 1e3, (en - t0) / 1e3))
    one = [v[0][1] - v[0][0] for v in per_sm.values() if len(v) == 1]
    two = [max(e for _, e in v) - min(s for s, _ in v) for v in per_sm.values() if len(v) == 2]
    span = (cta[:, 1].max().item() - t0) / 1e3
    print(f"  span {span:.1f} us; SMs used {len(per_sm)}; 1-CTA SMs {len(one)} mean {sum(one) / max(1, len(one)):.1f} us; "
          f"2-CTA SMs {len(two)} mean {sum(two) / max(1, len(two)):.1f} us; "
          f">2: {sum(1 for v in per_sm.values() if len(v) > 2)}", flush=True)


def stamps(lens, H, flags=1):
    T = sum(lens)
    x = rand_images(lens, H)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    s0 = torch.zeros(len(lens), H, D, D, device="cuda")
    _, _, dbg = run(x, s0, cu, T, H, flags)
    torch.cuda.synchronize()
    st = dbg[16:16 + 96 * 8].view(96, 8).cpu()
    n = min(80, (lens[-1] + 63) // 64 - 1)
    step = [(st[c + 1, 0] - st[c, 0]).item() for c in range(4, n)]
    seg = lambda a, b: sum((st[c, b] - st[c, a]).item() for c in range(4, n)) / (n - 4)  # noqa: E731
    ps = dbg[16 + 96 * 8:16 + 96 * 8 + 48 * 4].view(48, 4).cpu()
    m = min(n, 47)
    pseg = lambda a, b: sum((ps[c, b] - ps[c, a]).item() for c in range(4, m)) / (m - 4)  # noqa: E731
    print(f"state pass CTA0: start->LDTM done {sum((ps[c, 0] - st[c, 6]).item() for c in range(4, m)) / (m - 4):.0f}, "
          f"cvt+STS half {pseg(0, 1):.0f}, ->before wait::st {pseg(1, 2):.0f}, wait::st {pseg(2, 3):.0f}, "
          f"proxy fence+arrive {sum((st[c, 7] - ps[c, 3]).item() for c in range(4, m)) / (m - 4):.0f}", flush=True)
    print(f"stamps CTA0: chain step {sum(step) / len(step):.0f} clk; wait->MMA1 {seg(0, 1):.0f}, "
          f"MMA1->MMA2/3 issue (W read + vnew) {seg(1, 2):.0f}, MMA2/3 issue {seg(2, 3):.0f}; "
          f"MMA1 issue->vnew start {seg(1, 4):.0f}, vnew {seg(4, 5):.0f}, MMA2/3 issue->state start "
          f"{seg(3, 6):.0f}, state pass {seg(6, 7):.0f}, state end->next MMA1 {sum((st[c + 1, 1] - st[c, 7]).item() for c in range(4, n)) / (n - 4):.0f}", flush=True)


def stamps3(lens, H, flags=1):
    T = sum(lens)
    x = rand_images(lens, H)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    s0 = torch.zeros(len(lens), H, D, D, device="cuda")
    _, _, dbg = run(x, s0, cu, T, H, flags)
    torch.cuda.synchronize()
    if dbg[0].item() != 0:
        print(f"WATCHDOG {dbg[0].item():#x}")
    st = dbg[16:16 + 96 * 8].view(96, 8).cpu()
    n = min(80, (lens[-1] + 63) // 64 - 1)
    step = sum((st[c + 1, 0] - st[c, 0]).item() for c in range(4, n)) / (n - 4)
    seg = lambda a, b: sum((st[c, b] - st[c, a]).item() for c in range(4, n)) / (n - 4)  # noqa: E731
    wait_s = sum((st[c + 1, 0] - st[c, 7]).item() for c in range(4, n)) / (n - 4)
    print(f"v3 stamps CTA0: chain step {step:.0f} clk; P: stage wait {seg(0, 1):.0f}, LDTM+cvt+STTM+wait {seg(1, 2):.0f}, "
          f"bar.sync {seg(2, 3):.0f}, o_empty wait {seg(3, 4):.0f}; MMA1 issue+decay+wait W {seg(4, 5):.0f}, vnew+sync {seg(5, 6):.0f}, "
          f"MMA2 issue {seg(6, 7):.0f}, wait S {wait_s:.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--stamps", action="store_true")
    ap.add_argument("--heads", type=int, default=96)
    ap.add_argument("--v3", action="store_true")
    a = ap.parse_args()
    global V3
    V3 = a.v3
    from vllm.models.kimi_k3.nvidia.ops import kda_segscan

    kda_segscan.load_extension()
    if a.check:
        check([130, 64, 257], 4)
        check([700, 1000], 8)
    F16K = [5461, 5461, 5462]
    if a.time:
        time_it(F16K, a.heads)
        time_it(F16K, a.heads, flags=2)
        time_it([5461], 96)
        if V3:
            time_it(F16K, a.heads, flags=32)
    if a.stamps:
        if V3:
            stamps3(F16K, a.heads)
            stamps3(F16K, a.heads, 3)
            stamps3(F16K, a.heads, 33)
        else:
            stamps(F16K, a.heads)
            stamps(F16K, a.heads, 3)


if __name__ == "__main__":
    main()
