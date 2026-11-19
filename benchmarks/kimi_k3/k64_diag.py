# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dev diagnostic: chunk-64 tcgen05 KDA recurrence (k3_flashkda_split.k64_rec)
vs a torch emulation with the same bf16 rounding points, and its time at the
deployed DEP16 F16K shape (3 x 5461 tokens, 96 heads): full chunk-64 image
stream (HBM) and image reuse (compute-only), plus per-chunk clock stamps.

usage: python k64_diag.py [--check] [--time] [--stamps]
"""

import argparse

import torch

D, C64 = 128, 64


def make(lens, H, seed=3, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    T, N = sum(lens), len(lens)
    nch = [(n + C64 - 1) // C64 for n in lens]
    C = sum(nch)

    def rn(*s, sc=1.0):
        return torch.randn(*s, generator=g, device=dev) * sc

    kd, qd, kr = rn(H, C, C64, D, sc=0.08), rn(H, C, C64, D, sc=0.08), rn(H, C, C64, D, sc=0.08)
    eye = torch.eye(C64, device=dev)
    inv = eye + torch.tril(rn(H, C, C64, C64, sc=0.05), -1)
    mqk = torch.tril(rn(H, C, C64, C64, sc=0.1))
    gt = 0.7 + 0.3 * torch.rand(H, C, D, generator=g, device=dev)
    # zero the rows past each sequence end (last partial chunk)
    ci = 0
    for n, k in zip(lens, nch):
        valid = n - (k - 1) * C64
        for x in (kd, qd, kr, inv, mqk):
            x[:, ci + k - 1, valid:] = 0
        ci += k
    bf = torch.bfloat16
    x = dict(
        kd=kd.to(bf), qd=qd.to(bf), kr=kr.to(bf), inv=inv.to(bf), mqk=mqk.to(bf),
        gt=gt.contiguous(),
        beta=torch.rand(H, T, generator=g, device=dev).to(bf),
        v=rn(T, H, D).to(bf),
        s0=rn(N, H, D, D, sc=0.1),
        cu=torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev),
        lens=lens, nch=nch, C=C, H=H, T=T, N=N,
    )
    x["kdq_img"] = (torch.cat([x["kd"], x["qd"]], 2).view(H, C, 2 * C64, 16, 8)
                    .permute(0, 1, 3, 2, 4).contiguous())
    x["kr_img"] = (x["kr"].transpose(-1, -2).reshape(H, C, D, 8, 8)
                   .permute(0, 1, 3, 2, 4).contiguous())
    x["inv_img"] = x["inv"].view(H, C, C64, 8, 8).permute(0, 1, 3, 2, 4).contiguous()
    x["mqk_img"] = x["mqk"].view(H, C, C64, 8, 8).permute(0, 1, 3, 2, 4).contiguous()
    x["v_img"] = v_image(x["v"], lens, H)
    return x


def v_image(v, lens, H):
    """(T, H, 128) -> per chunk [8 t-groups][128 v][8 t] (zero past each end)."""
    C = sum((n + C64 - 1) // C64 for n in lens)
    vc = torch.zeros(H, C, C64, D, dtype=v.dtype, device=v.device)
    tok, ci = 0, 0
    for n in lens:
        for c in range((n + C64 - 1) // C64):
            valid = min(C64, n - c * C64)
            vc[:, ci, :valid] = v[tok:tok + valid].permute(1, 0, 2)
            tok += valid
            ci += 1
    return vc.view(H, C, 8, 8, D).permute(0, 1, 2, 4, 3).contiguous()


def run(x, flags=0, dbg=None, bufs=None):
    if bufs is None:
        bufs = (torch.zeros(x["T"], x["H"], D, dtype=torch.bfloat16, device="cuda"),
                torch.zeros_like(x["s0"]))
    out, fin = bufs
    check_hang = dbg is None
    if dbg is None:
        dbg = torch.zeros(1, dtype=torch.int64, device="cuda")
    torch.ops.k3_flashkda_split.k64_rec(
        x["kdq_img"], x["kr_img"], x["inv_img"], x["mqk_img"], x["gt"],
        x["beta"], x["v_img"], x["s0"], fin, out, x["cu"], flags, dbg)
    if check_hang and (h := dbg[0].item()) < 0:
        h &= (1 << 63) - 1
        print(f"HANG: block ({(h >> 24) & 0xFFFF},{h >> 40}) warp {(h >> 16) & 0xFF} "
              f"barrier {(h >> 10) & 0x3F} chunk {h & 1023}", flush=True)
    return out, fin


def ref(x):
    H = x["H"]
    out = torch.zeros(x["T"], H, D, device="cuda")
    fin = torch.zeros_like(x["s0"])
    f = lambda t: t.float()  # noqa: E731
    bfr = lambda t: t.to(torch.bfloat16).float()  # noqa: E731
    ci, tok = 0, 0
    for s, (n, k) in enumerate(zip(x["lens"], x["nch"])):
        S = x["s0"][s].clone()  # (H, v, k)
        for c in range(k):
            valid = min(C64, n - c * C64)
            cc = ci + c
            Sb = bfr(S)
            W = Sb @ f(x["kd"][:, cc]).transpose(1, 2)  # (H, v, t)
            Oq = Sb @ f(x["qd"][:, cc]).transpose(1, 2)
            vv = torch.zeros(H, D, C64, device="cuda")
            vv[:, :, :valid] = f(x["v"][tok:tok + valid]).permute(1, 2, 0)
            bt = torch.zeros(H, 1, C64, device="cuda")
            bt[:, 0, :valid] = f(x["beta"][:, tok:tok + valid])
            xx = bfr(bfr(vv - bfr(W)) * bt)
            U = bfr(xx @ f(x["inv"][:, cc]).transpose(1, 2))
            P = U @ f(x["kr"][:, cc])
            o = Oq + U @ f(x["mqk"][:, cc]).transpose(1, 2)
            out[tok:tok + valid] = o[:, :, :valid].permute(2, 0, 1)
            S = torch.addcmul(P, S, x["gt"][:, cc].unsqueeze(1))
            tok += valid
        fin[s] = S
        ci += k
    return out.to(torch.bfloat16), fin


def prep64(q, k, g, beta, A_log, dt_bias, lower_bound, lens, scale):
    """Chunk-64 prepare from post-conv q, k, raw gate g and beta, in FlashKDA
    K1's rounding style (bf16 normalized q / k, bf16 decay factors, fp32 L,
    bf16 INV / Mqk); the cross-sub-chunk decays use pairwise exponents, so no
    factor overflows at 64 tokens."""
    _, T, H, _ = q.shape
    f = lambda t: t.float()  # noqa: E731
    bf = torch.bfloat16
    qn = (f(q[0]) * torch.rsqrt((f(q[0]) ** 2).sum(-1, keepdim=True) + 1e-6)).to(bf)
    kn = (f(k[0]) * torch.rsqrt((f(k[0]) ** 2).sum(-1, keepdim=True) + 1e-6)).to(bf)
    gs = lower_bound * 1.4426950408889634
    gv = gs * torch.sigmoid(A_log.exp()[None, :, None] * (f(g[0]) + dt_bias[None]))
    bt = torch.sigmoid(f(beta[0])).to(bf)  # (T, H)
    nch = [(n + C64 - 1) // C64 for n in lens]
    C = sum(nch)
    kd = torch.zeros(H, C, C64, D, dtype=bf, device=q.device)
    qd, kr = torch.zeros_like(kd), torch.zeros_like(kd)
    inv = torch.zeros(H, C, C64, C64, dtype=bf, device=q.device)
    mqk = torch.zeros_like(inv)
    gt = torch.zeros(H, C, D, device=q.device)
    eye = torch.eye(C64, device=q.device)
    tok, ci = 0, 0
    for n, kk in zip(lens, nch):
        for c in range(kk):
            valid = min(C64, n - c * C64)
            sl = slice(tok, tok + valid)
            gc = gv[sl].cumsum(0).permute(1, 0, 2)  # (H, t, D)
            gT = gc[:, -1:, :]
            K = f(kn[sl]).permute(1, 0, 2)
            Q = f(qn[sl]).permute(1, 0, 2)
            e = torch.exp2(gc).to(bf).float()
            kd[:, ci, :valid] = (K * e).to(bf)
            qd[:, ci, :valid] = ((Q * e).to(bf).float() * float(torch.tensor(scale).to(bf))).to(bf)
            kr[:, ci, :valid] = (K * torch.exp2(gT - gc).to(bf).float()).to(bf)
            gt[:, ci] = torch.exp2(gT[:, 0])
            dg = torch.exp2((gc[:, :, None, :] - gc[:, None, :, :]).clamp(max=0))  # (H,i,j,D)
            A = torch.einsum("hid,hjd,hijd->hij", K, K, dg)
            Mq = torch.einsum("hid,hjd,hijd->hij", Q * scale, K, dg)
            b = f(bt[sl]).permute(1, 0)  # (H, t)
            Lm = torch.tril(A, -1) * b[:, :, None]
            I = eye[:valid, :valid]
            inv[:, ci, :valid, :valid] = torch.linalg.solve_triangular(
                I + Lm, I.expand(H, valid, valid).contiguous(), upper=False).to(bf)
            mqk[:, ci, :valid, :valid] = torch.tril(Mq).to(bf)
            tok += valid
            ci += 1
    return dict(kd=kd, qd=qd, kr=kr, inv=inv, mqk=mqk, gt=gt.contiguous(),
                beta=bt.t().contiguous(), C=C, nch=nch)


def e2e(lens, H):
    """k64_rec on the chunk-64 prepare vs stock FlashKDA (chunk 16) on the
    same post-conv inputs."""
    torch.manual_seed(11)
    T, dev, bf = sum(lens), "cuda", torch.bfloat16
    q, k, v, g = (torch.randn(1, T, H, D, dtype=bf, device=dev) for _ in range(4))
    beta = torch.randn(1, T, H, dtype=bf, device=dev)
    A_log = torch.randn(H, device=dev) * 0.5
    dt_bias = torch.randn(H, D, device=dev) * 0.1
    lb, scale = -3.0, D ** -0.5
    s0 = (torch.randn(len(lens), H, D, D, device=dev) * 0.1).to(bf).float()
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
    out_ref = torch.zeros_like(v)
    fin_ref = torch.zeros_like(s0)
    N = len(lens)
    ws = torch.empty(torch.ops.k3_flashkda_split.get_workspace_size(T, H, N),
                     dtype=torch.uint8, device=dev)
    torch.ops.k3_flashkda_split.fwd(q, k, v, g, beta, scale, out_ref, ws, A_log, dt_bias,
                                    lb, s0, fin_ref, cu, None, None, 3)
    p = prep64(q, k, g, beta, A_log, dt_bias, lb, lens, scale)
    x = dict(p, v=v[0].contiguous(), s0=s0, cu=cu, T=T, H=H, N=N)
    Hh, C = H, p["C"]
    x["kdq_img"] = (torch.cat([p["kd"], p["qd"]], 2).view(Hh, C, 2 * C64, 16, 8)
                    .permute(0, 1, 3, 2, 4).contiguous())
    x["kr_img"] = (p["kr"].transpose(-1, -2).reshape(Hh, C, D, 8, 8)
                   .permute(0, 1, 3, 2, 4).contiguous())
    x["inv_img"] = p["inv"].view(Hh, C, C64, 8, 8).permute(0, 1, 3, 2, 4).contiguous()
    x["mqk_img"] = p["mqk"].view(Hh, C, C64, 8, 8).permute(0, 1, 3, 2, 4).contiguous()
    x["v_img"] = v_image(x["v"], lens, H)
    o, s = run(x)
    torch.cuda.synchronize()
    print(f"e2e lens={lens} H={H} chunk-64 vs FlashKDA: out {err(o, out_ref[0])} | "
          f"state {err(s, fin_ref)}", flush=True)


def prep_kernel(q, k, g, beta, A_log, dt_bias, lb, lens, scale, cu):
    """k3_flashkda_split.prep64 (chunk-64 images from post-conv q / k, raw g, beta)."""
    _, T, H, _ = q.shape
    C = sum((n + C64 - 1) // C64 for n in lens)
    bf, dev = torch.bfloat16, q.device
    o = dict(kdq_img=torch.empty(H, C, 16, 2 * C64, 8, dtype=bf, device=dev),
             kr_img=torch.empty(H, C, 8, D, 8, dtype=bf, device=dev),
             inv_img=torch.empty(H, C, 8, C64, 8, dtype=bf, device=dev),
             mqk_img=torch.empty(H, C, 8, C64, 8, dtype=bf, device=dev),
             gt=torch.empty(H, C, D, device=dev),
             beta=torch.empty(H, T, dtype=bf, device=dev))
    torch.ops.k3_flashkda_split.prep64(
        q[0], k[0], g[0], beta[0], A_log, dt_bias, lb, scale, cu,
        o["kdq_img"], o["kr_img"], o["inv_img"], o["mqk_img"], o["gt"], o["beta"])
    return o


def prep_check(lens, H):
    """prep64 kernel images vs the torch prep64, and k64_rec on them vs FlashKDA."""
    torch.manual_seed(5)
    T, dev, bf = sum(lens), "cuda", torch.bfloat16
    q, k, v, g = (torch.randn(1, T, H, D, dtype=bf, device=dev) for _ in range(4))
    beta = torch.randn(1, T, H, dtype=bf, device=dev)
    A_log = torch.randn(H, device=dev) * 0.5
    dt_bias = torch.randn(H, D, device=dev) * 0.1
    lb, scale = -5.0, D ** -0.5
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
    o = prep_kernel(q, k, g, beta, A_log, dt_bias, lb, lens, scale, cu)
    p = prep64(q, k, g, beta, A_log, dt_bias, lb, lens, scale)
    C = p["C"]
    want = dict(
        kdq_img=torch.cat([p["kd"], p["qd"]], 2).view(H, C, 2 * C64, 16, 8).permute(0, 1, 3, 2, 4),
        kr_img=p["kr"].transpose(-1, -2).reshape(H, C, D, 8, 8).permute(0, 1, 3, 2, 4),
        inv_img=p["inv"].view(H, C, C64, 8, 8).permute(0, 1, 3, 2, 4),
        mqk_img=p["mqk"].view(H, C, C64, 8, 8).permute(0, 1, 3, 2, 4),
        gt=p["gt"], beta=p["beta"])
    torch.cuda.synchronize()
    print(f"prep lens={lens} H={H}: " + " | ".join(
        f"{n} {err(o[n], want[n].contiguous())}" for n in want), flush=True)
    N = len(lens)
    s0 = (torch.randn(N, H, D, D, device=dev) * 0.1).to(bf).float()
    out_ref, fin_ref = torch.zeros_like(v), torch.zeros_like(s0)
    ws = torch.empty(torch.ops.k3_flashkda_split.get_workspace_size(T, H, N), dtype=torch.uint8, device=dev)
    torch.ops.k3_flashkda_split.fwd(q, k, v, g, beta, scale, out_ref, ws, A_log, dt_bias,
                                    lb, s0, fin_ref, cu, None, None, 3)
    x = dict(o, v_img=v_image(v[0].contiguous(), lens, H), s0=s0, cu=cu, T=T, H=H, N=N)
    out, fin = run(x)
    torch.cuda.synchronize()
    print(f"  prep64 kernel + k64_rec vs FlashKDA: out {err(out, out_ref[0])} | state {err(fin, fin_ref)}",
          flush=True)


def prep_time():
    lens, H = [5461, 5461, 5461], 96
    T, dev, bf = sum(lens), "cuda", torch.bfloat16
    q, k, g = (torch.randn(1, T, H, D, dtype=bf, device=dev) for _ in range(3))
    beta = torch.randn(1, T, H, dtype=bf, device=dev)
    A_log, dt_bias = torch.zeros(H, device=dev), torch.zeros(H, D, device=dev)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
    for _ in range(2):
        prep_kernel(q, k, g, beta, A_log, dt_bias, -5.0, lens, D ** -0.5, cu)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(5):
        prep_kernel(q, k, g, beta, A_log, dt_bias, -5.0, lens, D ** -0.5, cu)
    e1.record()
    torch.cuda.synchronize()
    print(f"F16K prep64 kernel (SIMT dev stage): {e0.elapsed_time(e1) / 5 * 1000:.1f} us", flush=True)


def preptc_inputs(lens, H, seed=7, dev="cuda"):
    torch.manual_seed(seed)
    T, N, bf = sum(lens), len(lens), torch.bfloat16
    C3 = 3 * H * D
    x = torch.randn(T, C3, dtype=bf, device=dev)
    w = torch.randn(C3, 4, device=dev) * 0.5
    st = torch.randn(N + 1, C3, 3, device=dev)  # line 0 unused
    ci = torch.arange(1, N + 1, dtype=torch.int32, device=dev)
    hi = torch.tensor([i % 2 == 0 for i in range(N)], device=dev)
    g = torch.randn(T, H, D, dtype=bf, device=dev)
    beta = torch.randn(T, H, dtype=bf, device=dev)
    A_log = torch.randn(H, device=dev) * 0.5
    dt_bias = torch.randn(H, D, device=dev) * 0.1
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
    return dict(x=x, w=w, st=st, ci=ci, hi=hi, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias,
                cu=cu, T=T, N=N, H=H, lb=-5.0, scale=D ** -0.5)


def preptc_run(a, st, lens, out=None, dbg=None):
    H, T = a["H"], a["T"]
    C = sum((n + C64 - 1) // C64 for n in lens)
    bf, dev = torch.bfloat16, a["x"].device
    o = out or dict(kdq_img=torch.empty(H, C, 16, 2 * C64, 8, dtype=bf, device=dev),
                    kr_img=torch.empty(H, C, 8, D, 8, dtype=bf, device=dev),
                    inv_img=torch.empty(H, C, 8, C64, 8, dtype=bf, device=dev),
                    mqk_img=torch.empty(H, C, 8, C64, 8, dtype=bf, device=dev),
                    gt=torch.empty(H, C, D, device=dev),
                    v_img=torch.empty(H, C, 8, D, 8, dtype=bf, device=dev),
                    beta=torch.empty(H, T, dtype=bf, device=dev))
    torch.ops.k3_flashkda_split.prep64_tc(
        a["x"], a["w"], st, a["ci"], a["hi"], -1, a["g"], a["beta"], a["A_log"], a["dt_bias"],
        a["lb"], a["scale"], a["cu"], o["kdq_img"], o["kr_img"], o["inv_img"], o["mqk_img"],
        o["gt"], o["v_img"], o["beta"],
        dbg if dbg is not None else torch.zeros(1, dtype=torch.int64, device=dev))
    return o


def preptc_check(lens, H):
    """prep64_tc (conv fused) vs stock qkv_conv + torch prep64; k64_rec on its
    images vs FlashKDA on the stock conv output."""
    from vllm.models.kimi_k3.nvidia.ops import kda_conv
    a = preptc_inputs(lens, H)
    T, N = a["T"], a["N"]
    st_ref, st_tc = a["st"].clone(), a["st"].clone()
    qkv = kda_conv.qkv_conv(a["x"], a["w"], st_ref, a["cu"], a["ci"], a["hi"], H * D)
    q, k, v = (t.view(1, T, H, D) for t in qkv.unbind(0))
    o = preptc_run(a, st_tc, lens)
    p = prep64(q, k, a["g"][None], a["beta"][None], a["A_log"], a["dt_bias"], a["lb"], lens, a["scale"])
    C = p["C"]
    want = dict(
        kdq_img=torch.cat([p["kd"], p["qd"]], 2).view(H, C, 2 * C64, 16, 8).permute(0, 1, 3, 2, 4),
        kr_img=p["kr"].transpose(-1, -2).reshape(H, C, D, 8, 8).permute(0, 1, 3, 2, 4),
        inv_img=p["inv"].view(H, C, C64, 8, 8).permute(0, 1, 3, 2, 4),
        mqk_img=p["mqk"].view(H, C, C64, 8, 8).permute(0, 1, 3, 2, 4),
        gt=p["gt"], beta=p["beta"])
    torch.cuda.synchronize()
    vb = (o["v_img"] == v_image(v[0].contiguous(), lens, H)).all().item()
    sb = (st_tc == st_ref).all().item()
    print(f"preptc lens={lens} H={H}: v bitwise {vb} conv-state bitwise {sb} | " + " | ".join(
        f"{n} {err(o[n], want[n].contiguous())}" for n in want), flush=True)
    s0 = (torch.randn(N, H, D, D, device="cuda") * 0.1).to(torch.bfloat16).float()
    out_ref, fin_ref = torch.zeros_like(v), torch.zeros_like(s0)
    ws = torch.empty(torch.ops.k3_flashkda_split.get_workspace_size(T, H, N), dtype=torch.uint8,
                     device="cuda")
    torch.ops.k3_flashkda_split.fwd(q, k, v, a["g"][None], a["beta"][None], a["scale"], out_ref, ws,
                                    a["A_log"], a["dt_bias"], a["lb"], s0, fin_ref, a["cu"], None,
                                    None, 3)
    x = dict(o, s0=s0, cu=a["cu"], T=T, H=H, N=N)
    out, fin = run(x)
    torch.cuda.synchronize()
    print(f"  prep64_tc + k64_rec vs qkv_conv + FlashKDA: out {err(out, out_ref[0])} | "
          f"state {err(fin, fin_ref)}", flush=True)


def preptc_time():
    """F16K: prep64_tc (conv fused), k64_rec, and the stock conv for reference."""
    from vllm.models.kimi_k3.nvidia.ops import kda_conv
    lens, H = [5461, 5461, 5461], 96
    a = preptc_inputs(lens, H)
    st = a["st"].clone()
    o = preptc_run(a, st, lens)
    N = a["N"]
    s0 = torch.zeros(N, H, D, D, device="cuda")
    x = dict(o, s0=s0, cu=a["cu"], T=a["T"], H=H, N=N)
    bufs = (torch.zeros(a["T"], H, D, dtype=torch.bfloat16, device="cuda"), torch.zeros_like(s0))
    dbg = torch.zeros(1, dtype=torch.int64, device="cuda")

    def t(fn, n=10):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(n):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / n * 1000

    tp = t(lambda: preptc_run(a, st, lens, o))
    tr = t(lambda: run(x, 0, dbg, bufs))
    tc = t(lambda: kda_conv.qkv_conv(a["x"], a["w"], st, a["cu"], a["ci"], a["hi"], H * D))
    tb = t(lambda: (preptc_run(a, st, lens, o), run(x, 0, dbg, bufs)))
    C = o["gt"].shape[1]
    dbg = torch.zeros(2 * C * 16, dtype=torch.int64, device="cuda")
    preptc_run(a, st, lens, o, dbg)
    torch.cuda.synchronize()
    d = dbg.view(2 * C, 16).cpu()
    span = (d[:, 14].max() - d[:, 0].min()).item() / 1000
    names = {15: "tma", 1: "conv+norm", 2: "gate", 3: "scan", 4: "images", 5: "mma", 6: "inv-diag",
             12: "inv-blocks", 14: "out"}
    prev, parts = 0, []
    for k in (15, 1, 2, 3, 4, 5, 6, 12, 14):
        parts.append(f"{names[k]} {(d[:, k] - d[:, prev]).float().median().item() / 1000:.2f}")
        prev = k
    dur = (d[:, 14] - d[:, 0]).float() / 1000
    print(f"prep64_tc CTA phases (median us, heads 0-1): " + ", ".join(parts) +
          f"; CTA dur med {dur.median().item():.2f} max {dur.max().item():.2f}; span {span:.1f}", flush=True)
    print(f"F16K 3x5461 H=96: prep64_tc {tp:.1f} us | k64_rec {tr:.1f} | prep64_tc + k64_rec {tb:.1f} "
          f"| stock qkv_conv alone {tc:.1f}", flush=True)


def err(a, b):
    a, b = a.float(), b.float()
    return f"max|d| {(a - b).abs().max().item():.3e} (max|ref| {b.abs().max().item():.3e}, " \
           f"rel-rms {((a - b).norm() / b.norm().clamp_min(1e-30)).item():.3e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--stamps", action="store_true")
    ap.add_argument("--e2e", action="store_true")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--preptc", action="store_true")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    from vllm.models.kimi_k3.nvidia.ops import kda_segscan

    kda_segscan.load_extension()
    if a.check:
        for lens, H in (([64], 1), ([40, 23], 2), ([300, 17, 129], 4), ([1000, 777], 8), ([700, 700, 700], 96)):
            x = make(lens, H)
            o, s = run(x)
            torch.cuda.synchronize()
            ro, rs = ref(x)
            print(f"check lens={lens} H={H}: out {err(o, ro)} | state {err(s, rs)}", flush=True)
    if a.prep:
        for lens, H in (([64], 1), ([300, 17, 129], 4), ([1000, 777], 8)):
            prep_check(lens, H)
        prep_time()
    if a.preptc:
        for lens, H in (([64], 2), ([2], 2), ([300, 17, 129], 4), ([1000, 777, 5], 8)):
            preptc_check(lens, H)
        preptc_time()
    if a.e2e:
        for lens, H in (([300, 17, 129], 4), ([1000, 777], 8)):
            e2e(lens, H)
    if a.time or a.stamps:
        lens, H = [5461, 5461, 5461], 96
        x = make(lens, H)
        for flags, tag in ((0, "full image stream"), (2, "image reuse (compute only)")):
            if not a.time:
                break
            bufs = (torch.zeros(x["T"], x["H"], D, dtype=torch.bfloat16, device="cuda"),
                    torch.zeros_like(x["s0"]))
            for _ in range(3):
                run(x, flags, None, bufs)
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(10):
                run(x, flags, torch.zeros(1, dtype=torch.int64, device="cuda"), bufs)
            e1.record()
            torch.cuda.synchronize()
            print(f"F16K 3x5461 H=96 k64_rec {tag}: {e0.elapsed_time(e1) / 10 * 1000:.1f} us", flush=True)
        if a.stamps:
            nct = 96 * 3
            dbg = torch.zeros(1024 + 4 * nct, dtype=torch.int64, device="cuda")
            for fl in (8, 8 | 2):
                run(x, fl, dbg)
                torch.cuda.synchronize()
                t = dbg[1024:].view(nct, 4).cpu()
                t0 = t[:, 0].min().item()
                st, en, sm = t[:, 0] - t0, t[:, 1] - t0, t[:, 2]
                dur = (en - st).float() / 1000
                per_sm = torch.bincount(sm, minlength=1)
                late = (st > 2000).sum().item()
                print(f"cta timers flags={fl}: span {en.max().item() / 1000:.1f} us, cta dur us "
                      f"min {dur.min().item():.1f} med {dur.median().item():.1f} max {dur.max().item():.1f}; "
                      f"SMs used {(per_sm > 0).sum().item()}, max CTAs/SM {per_sm.max().item()}, "
                      f"CTAs starting >2us late {late}", flush=True)
                two = per_sm[sm] >= 2
                print(f"  dur us on 2-CTA SMs med {dur[two].median().item() if two.any() else 0:.1f}, "
                      f"on 1-CTA SMs med {dur[~two].median().item() if (~two).any() else 0:.1f}", flush=True)
            dbg.zero_()
            run(x, 1 | 2, dbg)
            torch.cuda.synchronize()
            d = dbg[:1024].view(64, 16).cpu()
            print(f"CTA0 clk chunks 0..63: {(d[63, 6] - d[0, 0]).item()}")
            per = [(d[c + 1, 0] - d[c, 0]).item() for c in range(20, 60)]
            print(f"chunk period clk: median {sorted(per)[len(per) // 2]} min {min(per)} max {max(per)}")
            names = ["start", "W ready", "x->MMA2", "U ready", "MMA3/4 issued", "state done",
                     "out(c-1) staged + Oq(c)", "state half 0", "d3+gt ready", "S LD0 back", "S round0 done"]
            for c in (40, 41):
                print(f"chunk {c}: " + ", ".join(
                    f"{names[i]} +{(d[c, i] - d[c, 0]).item()}" for i in (1, 2, 3, 4, 8, 9, 10, 7, 5, 6)))


if __name__ == "__main__":
    main()
