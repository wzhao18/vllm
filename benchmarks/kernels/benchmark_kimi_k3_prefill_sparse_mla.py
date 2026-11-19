# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 prefill sparse MLA block at DEP16 (TP1, 96 heads per GPU), one GPU.

Times the stock DCP-1 chain of one DSA layer at a prefill chunk of F fresh tokens
(3 requests at L78K context) kernel by kernel, as forward_impl runs it:
  w_uk   bmm (N,B,P)x(N,P,L) into the token-major [B,N,576] query + RoPE copy
  quant  scaled_fp8_quant (static per-tensor) of the [B,N*576] query
  pad    _pad_query_heads 96 -> 128 (trtllm-gen runs 128 heads)
  attn   trtllm-gen sparse MLA (FP8 Q, FP8 KV 576 B rows, top-2048)
  w_uv   bmm (N,B,L)x(N,L,V) from the head-padded output view
  gate   attn * sigmoid(gate) (prefill_glue.eager_gate_sigmoid_mul)
  insert concat_and_cache_mla (FP8 per-tensor) of the F new KV rows
and, where the installed FlashInfer has it, the prims_ts sparse MLA at 96 and 128
heads. Each layer instance uses its own KV pool / indices (rotated over --layers
copies) so the attention does not run from a warm L2.

usage: python benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 [--json out]
"""

import argparse
import json
import math
import statistics

import torch

H, P, L, R, V = 96, 128, 512, 64, 128
TOPK = 2048


def make_layer(F, ctx, nreq, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    per = F // nreq
    fresh = [per] * (nreq - 1) + [F - per * (nreq - 1)]
    page = 64
    seq_total = [ctx + f for f in fresh]
    pages = [math.ceil(s / page) for s in seq_total]
    perm = torch.randperm(sum(pages), device=dev, generator=g)  # paged, shuffled
    kv = (torch.randn(sum(pages), page, 576, device=dev, generator=g) * 0.5).to(
        torch.float8_e4m3fn
    )
    idx = torch.empty(F, TOPK, dtype=torch.int32, device=dev)
    lidx = torch.empty(F, TOPK, dtype=torch.int32, device=dev)
    req_ids = torch.empty(F, dtype=torch.int32, device=dev)
    btab = torch.zeros(nreq, max(pages), dtype=torch.int32, device=dev)
    row = 0
    base = 0
    for r in range(nreq):
        bt = perm[base : base + pages[r]].to(torch.int64)
        btab[r, : pages[r]] = bt.to(torch.int32)
        req_ids[row : row + fresh[r]] = r
        base += pages[r]
        n = fresh[r]
        pos = ctx + torch.arange(n, device=dev)  # causal: key <= pos
        for c0 in range(0, n, 512):
            c1 = min(n, c0 + 512)
            lim = pos[c0:c1] + 1
            keys = torch.rand(c1 - c0, ctx + n, device=dev, generator=g)
            keys = keys.masked_fill(
                torch.arange(ctx + n, device=dev)[None] >= lim[:, None], -1.0
            )
            logical = keys.topk(TOPK, dim=1).indices.sort(dim=1).values
            phys = bt[logical // page] * page + logical % page
            idx[row + c0 : row + c1] = phys.to(torch.int32)
            lidx[row + c0 : row + c1] = logical.to(torch.int32)
        row += n
    idx.logical = (lidx, req_ids, btab)  # request-relative top-k for h8L / remap
    return kv, idx


def time_stage(fn, iters):
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        s[i].record()
        fn(i)
        e[i].record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) * 1e3 for a, b in zip(s, e))


def stamp_report(vr, dev):
    """Median clock deltas per tile from the kernel's clock64 stamps (8 CTAs, 64 tiles):
    ev 0 loader issue start, 1 KV landed (MMA warp), 2 S ready (softmax), 3 P ready,
    4 PV issue, 5 O ready (epilogue start, last tile of a row), 6 O drained."""
    stamps = torch.zeros(8 * 64 * 8 + 64 * 12 * 4, dtype=torch.int64, device=dev)
    vr(0, stamps=stamps)
    torch.cuda.synchronize()
    w = stamps[8 * 64 * 8:].view(64, 12, 4).cpu()
    if bool((w != 0).any()):
        ref = w[:, 0, 0:1].clone()
        rel = (w - ref[:, None, :]).masked_fill(w == 0, -999999)
        # per tile u, relative to softmax warp 0 S loaded: warps 0-7 [S loaded, barrier,
        # P published, slow]; 8 MMA [at KV wait, QK issued, PV issued]; 9-11 gather warps done
        print("warps", {u: rel[u].tolist() for u in range(4, 10)}, flush=True)
        print("slow_path_tiles", [u for u in range(64) if bool((w[u, :8, 3] != 0).any())],
              flush=True)
    x = stamps[:8 * 64 * 8].view(8, 64, 8).cpu().double()
    rep = {}

    def med(d):
        d = d[torch.isfinite(d) & (d > 0)]
        return round(float(d.median()), 0) if d.numel() else None

    sl = slice(2, 64)
    rep["land_minus_issue"] = med(x[:, sl, 1] - x[:, sl, 0])
    rep["issued_minus_issue"] = med(x[:, sl, 7] - x[:, sl, 0])
    rep["land_minus_issued"] = med(x[:, sl, 1] - x[:, sl, 7])
    rep["qk_S_ready_minus_land"] = med(x[:, sl, 2] - x[:, sl, 1])
    rep["softmax"] = med(x[:, sl, 3] - x[:, sl, 2])
    rep["pv_issue_minus_P"] = med(x[:, sl, 4] - x[:, sl, 3])
    rep["tile_period_land"] = med(x[:, 3:64, 1] - x[:, 2:63, 1])
    rep["tile_period_S"] = med(x[:, 3:64, 2] - x[:, 2:63, 2])
    rep["issue_period"] = med(x[:, 3:64, 0] - x[:, 2:63, 0])
    rep["S_ready_minus_prevP"] = med(x[:, 3:64, 2] - x[:, 2:63, 3])
    e5 = x[:, :, 5]
    e6 = x[:, :, 6]
    rep["epilogue"] = med(e6[e5 > 0] - e5[e5 > 0])
    rep["cta0_tiles_0_20"] = x[0, :20, :5].sub(x[0, 0, 0]).long().tolist()
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--ctx", type=int, default=72900)
    ap.add_argument("--h8-ns", type=int, default=0, help="h8 KV stages (diagnostic, <= 6)")
    ap.add_argument("--reqs", type=int, default=3)
    ap.add_argument("--h8l-knobs", default="", help="h8 logical-mode knob lists")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--loaders", default="3,5")
    ap.add_argument("--qmag", type=float, default=1.0)
    ap.add_argument("--h2-knobs", default="", help="no_store:no_mma list")
    ap.add_argument("--kernels", default="sol,h2",
                    help="specialized kernels to time (sol, h2, h4, h5)")
    ap.add_argument("--no-base", action="store_true", help="skip the h4/h5 default knobs")
    ap.add_argument("--h4-knobs", default="",
                    help="h4/h5 knob list no_store:no_gather:epi_tma:hint:no_math:no_mma")
    ap.add_argument("--sol-knobs", default="",
                    help="comma list of gather_warps:no_gather:no_store diagnostics")
    ap.add_argument("--stamps", action="store_true",
                    help="also run each specialized variant once with clock64 stamps")
    a = ap.parse_args()
    if a.h8_ns:
        import vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 as _h8

        _h8.NS = a.h8_ns
    dev = torch.device("cuda")
    torch.manual_seed(0)
    F = a.tokens
    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.ops import prefill_glue
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        _pad_query_heads,
    )

    layers = [make_layer(F, a.ctx, a.reqs, dev, 17 + i) for i in range(a.layers)]
    q_in = [torch.randn(F, H, P + R, device=dev, dtype=torch.bfloat16) * a.qmag
            for _ in range(2)]
    W_UK_T = torch.randn(H, P, L, device=dev, dtype=torch.bfloat16) * 0.05
    W_UV = torch.randn(H, L, V, device=dev, dtype=torch.bfloat16) * 0.05
    gate = torch.randn(F, H * V, device=dev, dtype=torch.bfloat16)
    q_scale = torch.tensor([0.05], device=dev, dtype=torch.float32)
    k_scale = torch.tensor([1.0], device=dev, dtype=torch.float32)
    ws = torch.zeros(1 << 30, dtype=torch.int8, device=dev)
    seq = torch.full((F,), TOPK, dtype=torch.int32, device=dev)
    from flashinfer.utils import (
        get_device_sm_count,
        get_trtllm_gen_multi_ctas_kv_counter_bytes,
    )

    counters = torch.zeros(
        max(16, get_trtllm_gen_multi_ctas_kv_counter_bytes(F, 128, get_device_sm_count(dev))),
        dtype=torch.uint8,
        device=dev,
    )
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    res = {"tokens": F, "ctx": a.ctx, "reqs": a.reqs, "dev": torch.cuda.get_device_name()}
    st = {}

    qcat = torch.empty(F, H, L + R, device=dev, dtype=torch.bfloat16)

    def w_uk(i):
        q = q_in[i % 2]
        qn = q[..., :P].transpose(0, 1)
        torch.bmm(qn, W_UK_T, out=qcat[..., :L].transpose(0, 1))
        qcat[..., L:].copy_(q[..., P:])

    st["w_uk"] = time_stage(w_uk, a.iters)
    qf = torch.empty(F, H * (L + R), device=dev, dtype=torch.float8_e4m3fn)

    def quant(i):
        ops.scaled_fp8_quant(qcat.view(F, -1), q_scale, output=qf)

    st["quant"] = time_stage(quant, a.iters)
    qf3 = qf.view(F, H, L + R)
    qpad = _pad_query_heads(qf3, 128)
    st["pad"] = time_stage(lambda i: _pad_query_heads(qf3, 128), a.iters)
    # Fused path (this branch): bmm into a [B,N,L] buffer, then one kernel for
    # concat + static FP8 quant + head pad (decode_query_quant_pad).
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        decode_query_quant_pad,
    )

    qln = torch.empty(F, H, L, device=dev, dtype=torch.bfloat16)

    def w_uk_only(i):
        q = q_in[i % 2]
        torch.bmm(q[..., :P].transpose(0, 1), W_UK_T, out=qln.transpose(0, 1))

    st["fused_w_uk"] = time_stage(w_uk_only, a.iters)
    st["fused_quant_pad"] = time_stage(
        lambda i: decode_query_quant_pad(qln, q_in[i % 2][..., P:], q_scale, 128),
        a.iters,
    )
    st["fused_quant_nopad"] = time_stage(
        lambda i: decode_query_quant_pad(qln, q_in[i % 2][..., P:], q_scale, H),
        a.iters,
    )
    out = torch.empty(F, 128, L, device=dev, dtype=torch.bfloat16)

    def attn(i):
        kv, idx = layers[i % len(layers)]
        for s0 in range(0, F, 65535):
            s1 = min(F, s0 + 65535)
            trtllm_batch_decode_with_kv_cache_mla(
                query=qpad[s0:s1].unsqueeze(1),
                kv_cache=kv.unsqueeze(1),
                workspace_buffer=ws,
                qk_nope_head_dim=128,
                kv_lora_rank=L,
                qk_rope_head_dim=R,
                block_tables=idx[s0:s1].unsqueeze(1),
                seq_lens=seq[s0:s1],
                max_seq_len=TOPK,
                out=out[s0:s1].unsqueeze(1),
                bmm1_scale=0.0722,
                bmm2_scale=1.0,
                sparse_mla_top_k=TOPK,
                multi_ctas_kv_counter_buffer=counters,
            )

    attn(0)
    torch.cuda.synchronize()
    res["attn_finite"] = bool(torch.isfinite(out[:, :H].float()).all())
    st["attn_trtllm"] = time_stage(attn, a.iters)
    # specialized_prefill sparse MLA (sm_107, native 96 heads, no padded query)
    try:
        from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla import (
            sparse_mla_prefill_v1 as sparse_mla_prefill,
        )

        vr_out = torch.empty(F, H, L, device=dev, dtype=torch.bfloat16)
        from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_sol import (
            sparse_mla_prefill_sol,
        )

        kernels = a.kernels.split(",")
        if "sol" in kernels:
            def vs(i, stamps=None):
                kv, idx = layers[i % len(layers)]
                sparse_mla_prefill_sol(qf3, kv.view(-1, 576), idx, seq, 0.0722, 1.0,
                                       out=vr_out, stamps=stamps)

            vs(0)
            attn(0)
            torch.cuda.synchronize()
            d = (vr_out.float() - out[:, :H].float()).norm(dim=-1)
            res["sol_rel_vs_trtllm"] = (
                d / out[:, :H].float().norm(dim=-1).clamp(min=1e-6)).mean().item()
            st["attn_sol"] = time_stage(vs, a.iters)
            print("sol", res["sol_rel_vs_trtllm"], st["attn_sol"], flush=True)
            if a.stamps:
                res["stamps_sol"] = stamp_report(vs, dev)
                print("sol", res["stamps_sol"], flush=True)
        if "h2" in kernels:
            from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h2 import (
                sparse_mla_prefill_h2,
            )

            def vh(i, stamps=None):
                kv, idx = layers[i % len(layers)]
                sparse_mla_prefill_h2(qf3, kv.view(-1, 576), idx, seq, 0.0722, 1.0,
                                      out=vr_out, stamps=stamps)

            vh(0)
            attn(0)
            torch.cuda.synchronize()
            d = (vr_out.float() - out[:, :H].float()).norm(dim=-1)
            res["h2_rel_vs_trtllm"] = (
                d / out[:, :H].float().norm(dim=-1).clamp(min=1e-6)).mean().item()
            st["attn_h2"] = time_stage(vh, a.iters)
            print("h2", res["h2_rel_vs_trtllm"], st["attn_h2"], flush=True)
            if a.stamps:
                res["stamps_h2"] = stamp_report(vh, dev)
                print("h2", res["stamps_h2"], flush=True)
            for kn in [k for k in a.h2_knobs.split(",") if k]:
                knobs = tuple(int(v) for v in kn.split(":"))

                def vhk(i, stamps=None, knobs=knobs):
                    kv, idx = layers[i % len(layers)]
                    sparse_mla_prefill_h2(qf3, kv.view(-1, 576), idx, seq, 0.0722, 1.0,
                                          out=vr_out, stamps=stamps, knobs=knobs)

                vhk(0)
                st[f"attn_h2_k{kn}"] = time_stage(vhk, a.iters)
                print("h2 knobs", kn, st[f"attn_h2_k{kn}"], flush=True)
                if a.stamps:
                    r = stamp_report(vhk, dev)
                    print("h2 knobs", kn, {k: v for k, v in r.items() if k != "cta0_tiles_0_20"},
                          flush=True)
        import importlib

        if "h8L" in kernels or "remap" in kernels:
            from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import (
                sparse_mla_prefill_h8_logical,
            )
            from vllm.v1.attention.backends.mla.sparse_utils import (
                triton_convert_req_index_to_global_index,
            )

            def vl(i, knobs=()):
                kv, idx = layers[i % len(layers)]
                lidx, rq, btab = idx.logical
                sparse_mla_prefill_h8_logical(qf3, kv.view(-1, 576), lidx, rq, btab, 64, 64,
                                              0.0722, 1.0, out=vr_out, knobs=knobs,
                                              lengths=seq)

            def vremap(i):
                kv, idx = layers[i % len(layers)]
                lidx, rq, btab = idx.logical
                triton_convert_req_index_to_global_index(
                    rq, btab, lidx, BLOCK_SIZE=64, BLOCK_STRIDE_ROWS=64, NUM_TOPK_TOKENS=TOPK,
                    return_valid_counts=True)

            if "h8L" in kernels:
                from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import (
                    sparse_mla_prefill_h8,
                )
                kv0, idx0 = layers[0]
                ref8 = sparse_mla_prefill_h8(qf3, kv0.view(-1, 576), idx0, seq, 0.0722, 1.0)
                vl(0)
                torch.cuda.synchronize()
                res["h8L_equal_h8"] = bool(torch.equal(vr_out, ref8))
                st["attn_h8L"] = time_stage(vl, a.iters)
                print("h8L equal", res["h8L_equal_h8"], st["attn_h8L"], flush=True)
                for kn in [k for k in a.h8l_knobs.split(",") if k]:
                    knobs = tuple(int(v) for v in kn.split(":"))
                    st[f"attn_h8L_{kn}"] = time_stage(lambda i, k=knobs: vl(i, k), a.iters)
                    print("h8L", kn, st[f"attn_h8L_{kn}"], flush=True)
            if "remap" in kernels:
                st["remap"] = time_stage(vremap, a.iters)
                print("remap", st["remap"], flush=True)

        for kname in [k for k in ("h4", "h5", "h6", "h7", "h8") if k in kernels]:
            kfn = getattr(importlib.import_module(
                f"vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_{kname}"),
                f"sparse_mla_prefill_{kname}")

            for kn in ([] if a.no_base else ["base"]) + [k for k in a.h4_knobs.split(",") if k]:
                knobs = () if kn == "base" else tuple(int(v) for v in kn.split(":"))

                def v4(i, stamps=None, knobs=knobs, kfn=kfn):
                    kv, idx = layers[i % len(layers)]
                    kfn(qf3, kv.view(-1, 576), idx, seq, 0.0722, 1.0,
                        out=vr_out, stamps=stamps, knobs=knobs)

                v4(0)
                attn(0)
                torch.cuda.synchronize()
                d = (vr_out.float() - out[:, :H].float()).norm(dim=-1)
                res[f"{kname}_{kn}_rel_vs_trtllm"] = (
                    d / out[:, :H].float().norm(dim=-1).clamp(min=1e-6)).mean().item()
                st[f"attn_{kname}_{kn}"] = time_stage(v4, a.iters)
                print(kname, kn, res[f"{kname}_{kn}_rel_vs_trtllm"],
                      st[f"attn_{kname}_{kn}"], flush=True)
                if a.stamps:
                    res[f"stamps_{kname}_{kn}"] = stamp_report(v4, dev)
                    print(kname, kn, res[f"stamps_{kname}_{kn}"], flush=True)
                if (len(knobs) > 8 and knobs[8]) or (kname in ("h6", "h8") and len(knobs) > 4
                                                       and knobs[4]):
                    sb = torch.zeros(256, dtype=torch.int32, device=dev)
                    v4(0, stamps=sb)
                    torch.cuda.synchronize()
                    x = sb.view(8, 32).cpu().long()
                    rel = ((x - x[:, :1]) & 0xFFFFFFFF)
                    rel = torch.where(rel > 0x7FFFFFFF, rel - (1 << 32), rel)
                    rel = rel.masked_fill(x == 0, -999999)
                    med = rel.median(dim=0).values.tolist()
                    per = ((x[1:, 0] - x[:-1, 0]) & 0xFFFFFFFF).median().item()
                    res[f"sstamp_{kname}_{kn}"] = {"median_rel_slot": med, "period": per,
                                                   "raw": x.tolist()}
                    print(kname, kn, "sstamp slots rel. to softmax w0 wait start: "
                          "0-7 sm w0 (wait, S ready, S loaded, max stored, max exch, Pbuf "
                          "free, P written, fenced), 8-15 sm w4, 16 QK at KV wait, 17 QK "
                          "issued, 18 PV OREADY passed, 19 PV issued, 20 QK KVFULL passed (h8) / out PFULL passed, "
                          "21 out OREADY arrived, 22 gather KVEMPTY passed, 23 gather "
                          "issued, 24 QK SFREE passed, 25 QK at SFREE wait, 26 PV at "
                          "OREADY wait", med, "period", per, flush=True)
        for kn in [k for k in a.sol_knobs.split(",") if k]:
            knobs = tuple(int(v) for v in kn.split(":"))

            def vk(i, stamps=None, knobs=knobs):
                kv, idx = layers[i % len(layers)]
                sparse_mla_prefill_sol(qf3, kv.view(-1, 576), idx, seq, 0.0722, 1.0,
                                       out=vr_out, stamps=stamps, knobs=knobs)

            vk(0)
            st[f"attn_sol_k{kn}"] = time_stage(vk, a.iters)
            print("sol knobs", kn, st[f"attn_sol_k{kn}"], flush=True)
            if a.stamps:
                res[f"stamps_sol_k{kn}"] = stamp_report(vk, dev)
                print("sol knobs", kn, {k: v for k, v in res[f"stamps_sol_k{kn}"].items()
                                        if k != "cta0_tiles_0_20"}, flush=True)
        for var in [v for v in a.loaders.split(",") if v]:
            f6 = (var.split(":") + ["0"] * 5)[:6]
            nl, pp, tm, rg, sh, ns = (int(v) for v in f6)
            ns = ns or 2
            tag = f"l{nl}p{pp}t{tm}r{rg}s{sh}n{ns}"
            regs = (((184, 72) if nl <= 7 else (168, 40)) if rg else ())
            shift = ((7, 1.0), (4, 4.5), (0, 8.0))[sh]

            def vr(i, nl=nl, pp=pp, tm=tm, regs=regs, shift=shift, ns=ns, stamps=None):
                kv, idx = layers[i % len(layers)]
                sparse_mla_prefill(qf3, kv.view(-1, 576), 64, idx, seq, 0.0722, 1.0,
                                   out=vr_out, loaders=nl, pipelined=bool(pp),
                                   tma=bool(tm), regs=regs, shift=shift, stages=ns,
                                   stamps=stamps)

            vr(0)
            attn(0)  # the stock output of the same layer
            torch.cuda.synchronize()
            d = (vr_out.float() - out[:, :H].float()).norm(dim=-1)
            rel = (d / out[:, :H].float().norm(dim=-1).clamp(min=1e-6)).mean().item()
            res[f"vr_{tag}_rel_vs_trtllm"] = rel
            st[f"attn_vr_{tag}"] = time_stage(vr, a.iters)
            if a.stamps:
                res[f"stamps_{tag}"] = stamp_report(vr, dev)
                print(tag, res[f"stamps_{tag}"], flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        res["vr_error"] = repr(e)[:400]
    o = torch.empty(F, H * V, device=dev, dtype=torch.bfloat16)

    def w_uv(i):
        x = out[:, :H].transpose(0, 1)
        torch.bmm(x, W_UV, out=o.view(F, H, V).transpose(0, 1))

    st["w_uv"] = time_stage(w_uv, a.iters)
    st["gate"] = time_stage(lambda i: prefill_glue.eager_gate_sigmoid_mul(o, gate), a.iters)
    kvc = torch.randn(F, L, device=dev, dtype=torch.bfloat16)
    kpe = torch.randn(F, R, device=dev, dtype=torch.bfloat16)
    cache = torch.zeros(4096, 64, 576, device=dev, dtype=torch.uint8)
    slots = torch.randperm(4096 * 64, device=dev)[:F].to(torch.int64)
    st["insert"] = time_stage(
        lambda i: ops.concat_and_cache_mla(kvc, kpe, cache, slots, "fp8", k_scale),
        a.iters,
    )

    # prims_ts (FlashInfer task-scheduled sparse MLA), if present
    try:
        from flashinfer.attention.prims_ts import (
            BatchSparseMLADecodePagedTSWrapper,
            get_prims_ts_sparse_mla_decode_workspace_size,
        )

        for heads in (96, 128):
            rows = 4096
            kw = dict(
                max_topk=TOPK,
                max_q_len=1,
                packed_query=True,
                q_dtype=torch.float8_e4m3fn,
                qk_rope_head_dim=64,
                kv_cache_format="native",
                return_lse=False,
            )
            size = get_prims_ts_sparse_mla_decode_workspace_size(dev, rows, heads, **kw)
            pws = torch.empty(size, dtype=torch.uint8, device=dev)
            plan = BatchSparseMLADecodePagedTSWrapper(pws)
            plan.plan(dev, rows, heads, **kw)
            qo = torch.arange(rows + 1, dtype=torch.int32, device=dev)
            qh = qpad[:, :heads].contiguous()
            po = torch.empty(F, heads, L, device=dev, dtype=torch.bfloat16)

            def pts(i, qh=qh, po=po, plan=plan, qo=qo):
                kv, idx = layers[i % len(layers)]
                rowsv = kv.view(-1, 576)
                for s0 in range(0, F, rows):
                    s1 = min(F, s0 + rows)
                    plan.run(
                        qh[s0:s1],
                        rowsv,
                        idx[s0:s1],
                        seq[s0:s1],
                        qo_indptr=qo,
                        softmax_scale=0.0722,
                        output_scale=1.0,
                        out=po[s0:s1],
                        lse=None,
                        validate=False,
                    )

            pts(0)
            torch.cuda.synchronize()
            st[f"attn_prims_ts_h{heads}"] = time_stage(pts, a.iters)
            d = (po[:, :H].float() - out[:, :H].float()).abs().max().item()
            res[f"prims_ts_h{heads}_maxdiff_vs_trtllm"] = d
    except Exception as e:  # noqa: BLE001
        res["prims_ts_error"] = repr(e)[:300]

    res["stages_us"] = {k: round(v, 1) for k, v in st.items()}
    fl = 2.0 * F * H * TOPK * (576 + 512)
    res["attn_tflops_real_heads"] = {
        k: round(fl / v / 1e6, 0) for k, v in st.items() if k.startswith("attn")
    }
    stock = ["w_uk", "quant", "pad", "attn_trtllm", "w_uv", "gate", "insert"]
    res["stock_chain_us"] = round(sum(st[k] for k in stock), 1)
    fused = ["fused_w_uk", "fused_quant_pad", "attn_trtllm", "w_uv", "gate", "insert"]
    res["fused_query_chain_us"] = round(sum(st[k] for k in fused), 1)
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
