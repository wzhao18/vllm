"""ops1 attention-side bench: one family per GPU (shard i of 4 -> FAMS[i]); every in-scope case of that family.

idx   : DSA indexer, MXFP4 Q x MXFP4 keys (68 B/key) + exact top-2048 (DeepSelect on the BF16 logits).
        prefill rows (requests_in_chunk set): DeepGEMM fp8_fp4_mqa_logits (cost-balanced schedule) on 3 requests
        (prefix 72900, rows spread over the 5456 fresh tokens -> mean keys/row 75.6K), as sparse_attn_indexer.
        decode rows: DeepGEMM fp8_fp4_paged_mqa_logits (page 64, distinct pages per request) over q=4 rows/request
        and the case's keys/row (DCP modes: the local key shard; the cross-rank merge is not timed).
mla   : FP8 Q x FP8 KV (576 B rows) sparse MLA via FlashInfer trtllm-gen (trtllm_batch_decode_with_kv_cache_mla,
        the FLASHINFER_MLA_SPARSE non-DCP path), 96 heads padded to 128, 2048 (or the case's) keys/row; DCP partial
        cases with return_lse (production DCP uses prims_ts: noted).
kda   : prefill: FlashKDA (kda._flashkda_prefill: in-tree split build when it fits, else stock) and the segment-parallel
        prefill (kda_segscan) when its planner accepts; 3 x 5456 tokens, FP32 state. decode: sdec PreparedKda core
        (conv + recurrence + norm) at (B requests, q=4), FP32 state.
ares  : AttnRes mix (torch.ops._C.kimi_k3_attn_res via ops.attn_res.attn_res) at rows x {1..8} sources; layer mean
        over l=1..93 with sources = ceil(l/12) (as the model). pre = no delta; post = delta + block write.
usage: attn_bench.py cases.json out.jsonl i/n
"""
import json
import math
import sys
import traceback
from types import SimpleNamespace

import torch

from mbc import Out, env_info, log, time_us

FAMS = ["idx", "mla", "kda", "ares"]
FAMILY = {"idx": "dsa_indexer_topk", "mla": "sparse_mla", "kda": "kda", "ares": "attnres"}
i, n = map(int, sys.argv[3].split("/"))
fams = [f for j, f in enumerate(FAMS) if j % n == i]
allcases = json.load(open(sys.argv[1]))
out = Out(sys.argv[2])
dev = torch.device("cuda")
out.row(kind="env", fams=fams, **env_info())


def emit(c, measured, parts, method, notes=""):
    out.row(kind="case", case=c["id"], family=c["family"], ops=c["ops"], topologies=c["topo_ids"],
            ckpts=c["ckpts"], shape=c["shape"], parts={k: round(v, 3) for k, v in parts.items()},
            measured_us=round(measured, 3), method=method, t_sol_us=c["t_sol_us"], t_ach_us=c["t_ach_us"],
            notes=notes)


def fail(c, e):
    traceback.print_exc()
    out.row(kind="case", case=c["id"], family=c["family"], shape=c["shape"], error=repr(e)[:400])


# ---------------------------------------------------------------- indexer
def idx_prefill(c):
    from vllm.model_executor.layers.indexer_topk import deep_select_topk
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, get_mqa_logits_metadata, mqa_logits_mx_inputs
    s = c["shape"]
    R, H, D, topk = s["rows"], s["heads"], s["head_dim"], s["topk"]
    nreq, prefix, fresh = int(s["requests_in_chunk"]), int(s["prefix"]), int(round(s["fresh"]))
    per = [R // nreq + (1 if j < R % nreq else 0) for j in range(nreq)]
    ks, ke, off = [], [], 0
    for j in range(nreq):
        pos = torch.linspace(0, fresh - 1, per[j]).round().long()
        ks.append(torch.full((per[j],), off, dtype=torch.long))
        ke.append(off + prefix + pos + 1)
        off += prefix + fresh
    ks = torch.cat(ks).int().to(dev)
    ke = torch.cat(ke).int().to(dev)
    nk = off
    q = torch.randint(-128, 127, (R, H, D // 2), dtype=torch.int8, device=dev)
    qs = torch.full((R, H), 0x7F7F7F7F, dtype=torch.int32, device=dev)
    k = torch.randint(-128, 127, (nk, D // 2), dtype=torch.int8, device=dev)
    kscale = torch.full((nk,), 0x7F7F7F7F, dtype=torch.int32, device=dev)
    w = torch.randn(R, H, device=dev) * 0.1
    qs2, w2 = mqa_logits_mx_inputs(q, qs, w)
    maxk = int((ke - ks).max())
    meta = get_mqa_logits_metadata(ks, ke, nk, H)
    f_logits = lambda: fp8_fp4_mqa_logits((q, qs2), (k, kscale), w2, ks, ke, maxk, schedule_meta=meta)  # noqa: E731
    logits = f_logits()
    ends = (ke - ks).int()
    outi = torch.empty(R, topk, dtype=torch.int32, device=dev)
    t_l, m = time_us(f_logits)
    t_l0, _ = time_us(lambda: fp8_fp4_mqa_logits((q, qs2), (k, kscale), w2, ks, ke, maxk))
    t_k, _ = time_us(lambda: deep_select_topk(logits, topk, end=ends, output_idx=outi))
    mean_keys = float(ends.float().mean())
    notes = (f"logits {tuple(logits.shape)} {logits.dtype}; mean keys/row {mean_keys:.0f}; one call over all rows "
             f"(production splits by the logits budget); logits w/o schedule {t_l0:.1f} us; idx_q_quant not timed")
    emit(c, t_l + t_k, {"logits": t_l, "topk": t_k, "logits_nosched": t_l0}, m, notes)


def idx_decode(c):
    from vllm.model_executor.layers.indexer_topk import deep_select_topk
    from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view
    from vllm.utils.deep_gemm import (fp8_fp4_paged_mqa_logits, get_num_sms, get_paged_mqa_logits_metadata,
                                      mqa_logits_mx_inputs)
    s = c["shape"]
    R, H, D, topk = s["rows"], s["heads"], s["head_dim"], s["topk"]
    q_per = 4 if R >= 4 else R
    reqs = max(1, R // q_per)
    K = int(round(s["mean_keys_per_row"]))
    UNIT = 64
    pages = -(-(K + q_per) // UNIT)
    fb = D // 2 + 4
    index = torch.randint(0, 255, (reqs * pages + 2, UNIT, fb), dtype=torch.uint8, device=dev)
    index[..., D // 2:] = 127
    table = (torch.arange(reqs * pages, dtype=torch.int32, device=dev).view(reqs, pages)
             .repeat_interleave(q_per, 0).contiguous())
    ctx = torch.tensor([K + 1 + j for _ in range(reqs) for j in range(q_per)], dtype=torch.int32,
                       device=dev).view(-1, 1)
    rows = reqs * q_per
    idx = torch.arange(rows, dtype=torch.int32, device=dev)
    sched = get_paged_mqa_logits_metadata(ctx, UNIT, get_num_sms(), idx)
    query = torch.randint(-128, 127, (rows, 1, H, D // 2), dtype=torch.int8, device=dev)
    qsc = torch.full((rows, 1, H), 0x7F7F7F7F, dtype=torch.int32, device=dev)
    w = torch.randn(rows, H, device=dev) * 0.1
    qs2, w2 = mqa_logits_mx_inputs(query, qsc, w)
    kv = kv_cache_as_quant_view(index, D, True)
    width = -(-(K + q_per + 1) // 256) * 256
    f = lambda: fp8_fp4_paged_mqa_logits((query, qs2), kv, w2, ctx, table, sched, width, indices=idx)  # noqa: E731
    logits = f()
    ends = ctx.view(-1)
    outi = torch.empty(rows, topk, dtype=torch.int32, device=dev)
    t_l, m = time_us(f)
    t_k, _ = time_us(lambda: deep_select_topk(logits, topk, end=ends, output_idx=outi))
    t_lw, _ = time_us(f, cold=False)
    t_kw, _ = time_us(lambda: deep_select_topk(logits, topk, end=ends, output_idx=outi), cold=False)
    notes = (f"paged (stock DeepGEMM) logits {tuple(logits.shape)} {logits.dtype}; {reqs} req x q{q_per}, "
             f"{K} keys/row; stock kernels, not the sdec fused indexer; DCP merge not timed")
    emit(c, t_l + t_k, {"logits": t_l, "topk": t_k, "logits_warm": t_lw, "topk_warm": t_kw}, m, notes)


# ---------------------------------------------------------------- sparse MLA
_WS = None


def mla_case(c):
    global _WS
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    s = c["shape"]
    T, H, keys = s["rows"], s["heads"], s["keys_per_row"]
    Hp = 128 if 64 < H < 128 else H
    dcp = "DCP" in s["mode"]
    if _WS is None:
        _WS = torch.zeros(1 << 30, dtype=torch.int8, device=dev)
    try:
        from flashinfer.utils import get_device_sm_count, get_trtllm_gen_multi_ctas_kv_counter_bytes
        nbytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(T, Hp, get_device_sm_count(dev))
        counters = torch.zeros(max(nbytes, 16), dtype=torch.uint8, device=dev)
    except Exception as e:  # noqa: BLE001
        log("kv counters", repr(e)[:150])
        counters = None
    # KV pool: decode rows = B requests x q4, 135K context each; prefill: 3 x 78.4K
    if T >= 2048 and not dcp:
        pool = 3 * 78356
    else:
        pool = max(1, T // 4) * (135000 if not dcp else max(keys * 4, 8192))
    pool = min(pool, 6_000_000)
    pool = -(-pool // 64) * 64
    kv = (torch.randn(pool // 64, 1, 64, 576, device=dev) * 0.3).to(torch.float8_e4m3fn)
    q = (torch.randn(T, 1, Hp, 576, device=dev) * 0.3).to(torch.float8_e4m3fn)
    idx = torch.randint(0, pool, (T, 1, keys), dtype=torch.int32, device=dev)
    seq = torch.full((T,), keys, dtype=torch.int32, device=dev)
    o = torch.empty(T, 1, Hp, 512, dtype=torch.bfloat16, device=dev)
    lse = torch.empty(T, Hp, dtype=torch.float32, device=dev) if dcp else None
    kw = dict(lse=lse, return_lse=True) if dcp else {}
    if counters is not None:
        kw["multi_ctas_kv_counter_buffer"] = counters

    def f():
        trtllm_batch_decode_with_kv_cache_mla(
            query=q, kv_cache=kv, workspace_buffer=_WS, qk_nope_head_dim=128, kv_lora_rank=512,
            qk_rope_head_dim=64, block_tables=idx, seq_lens=seq, max_seq_len=keys, out=o,
            bmm1_scale=0.0722, bmm2_scale=1.0, sparse_mla_top_k=keys, **kw)
    f()
    torch.cuda.synchronize()
    ok = bool(torch.isfinite(o.float()).all())
    t, m = time_us(f)
    tw, _ = time_us(f, cold=False)
    flops = 2 * T * Hp * keys * (576 + 512)
    notes = (f"trtllm-gen sparse MLA fp8, heads {H}->{Hp} padded, pool {pool} rows, finite={ok}; "
             f"{flops / t / 1e6:.0f} TFLOPS (padded heads); prologue/gate_mul not timed"
             + ("; DCP case: return_lse, production DCP uses prims_ts" if dcp else ""))
    emit(c, t, {"sparse_mla": t, "sparse_mla_warm": tw}, m, notes + f"; kv counters passed={counters is not None}")


# ---------------------------------------------------------------- KDA
def kda_prefill(c):
    from vllm.models.kimi_k3.nvidia import kda as K
    from vllm.models.kimi_k3.nvidia.ops import kda_segscan
    import vllm._flashkda_C  # noqa: F401
    s = c["shape"]
    H, D = s["heads"], 128
    nseq = int(s["nseq"])
    lens = [int(round(s["fresh_per_seq"]))] * nseq
    T = sum(lens)
    q, k, v, g = (torch.randn(1, T, H, D, dtype=torch.bfloat16, device=dev) for _ in range(4))
    beta = torch.randn(1, T, H, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.5
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=dev) * 0.1
    init = torch.randn(nseq, H, D, D, dtype=torch.float32, device=dev) * 0.1
    cu = torch.tensor([0, *lens], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    cud = cu.to(dev)
    o = torch.empty_like(v)
    fin = torch.empty_like(init)
    ws = torch.empty(torch.ops._flashkda_C.get_workspace_size(T, H, nseq), dtype=torch.uint8, device=dev)
    parts, notes = {}, []
    split = kda_segscan.narrow_vsplit_fits(H, nseq, D, dev)
    f1 = lambda: K._flashkda_prefill(q, k, v, g, beta, A_log, dt_bias, -5.0, init, cud, o, fin, ws)  # noqa: E731
    parts["flashkda"], m = time_us(f1)
    notes.append(f"flashkda path={'split(vsplit)' if split else 'stock'}")
    try:
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        bufs = kda_segscan.init_buffers(T, H, dev, max(nseq, 8))
        if bufs is None:
            notes.append("segscan: init_buffers None")
        else:
            plan = kda_segscan.make_plan(cu, H, dev, None, sms=bufs.sms, max_parts=bufs.n)
            if plan is None:
                notes.append("segscan: planner declines (production uses flashkda)")
            else:
                ws2 = torch.empty(max(ws.numel(), torch.ops._flashkda_C.get_workspace_size(T, H, bufs.n)),
                                  dtype=torch.uint8, device=dev)
                f2 = lambda: kda_segscan.segmented_prefill(  # noqa: E731
                    plan=plan, bufs=bufs, q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias,
                    lower_bound=-5.0, initial_state=init, out=o, final_state=fin, workspace=ws2,
                    checkpoint_state=None)
                parts["segscan"], _ = time_us(f2)
                notes.append(f"segscan pseudo={plan.num_pseudo} (sms {sms})")
    except Exception as e:  # noqa: BLE001
        notes.append("segscan err " + repr(e)[:150])
    best = min(parts.values())
    notes.append("core recurrence only: conv / gate_beta_norm / o_norm / checkpoint export not timed")
    emit(c, best, parts, m, "; ".join(notes))


def kda_decode(c):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import prepared
    s = c["shape"]
    B, qn, H = s["B"], s["q"], s["heads"]
    DIM = H * 128
    pitch = -(-(4 * DIM + 128 + H) // 16) * 16
    gen = torch.Generator(device=dev).manual_seed(0)

    def randn(*shape, dtype=torch.float32, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=dev) * scale).to(dtype)
    M = SimpleNamespace
    att = M(conv1d=M(weight=randn(3 * DIM, 1, 4, scale=0.5), bias=None), dt_bias=randn(DIM, scale=0.5),
            A_log=randn(H, scale=0.5), f_b_proj=M(weight=randn(DIM, 128, dtype=torch.bfloat16, scale=128**-0.5)),
            o_norm=M(weight=1 + randn(128, scale=0.1), eps=1e-5), gate_lower_bound=-5.0)
    rows = B * qn
    core = prepared.PreparedKda(att, {(B, qn): (pitch, 0)}, True)
    slots = 2 * rows + 1
    conv = torch.randn(slots, 3 * DIM, qn + 2, generator=gen, device=dev).to(torch.bfloat16)
    state = torch.randn(slots, H, 128, 128, generator=gen, device=dev) * 0.1
    cachesv = (conv, state.float())
    core.prepare_caches(cachesv)
    order = torch.randperm(slots - 1, generator=torch.Generator().manual_seed(2))[:rows] + 1
    slot_table = order.view(B, qn).to(torch.int32).to(dev)
    accepted = torch.full((B,), qn, dtype=torch.int32, device=dev)
    projected = torch.randn(rows, pitch, generator=gen, device=dev).to(torch.bfloat16) * 0.5
    o = torch.empty(rows, DIM, dtype=torch.bfloat16, device=dev)
    f = lambda: core.run(projected, o, caches=cachesv, slots=slot_table, accepted=accepted)  # noqa: E731
    t, m = time_us(f)
    tw, _ = time_us(f, cold=False)
    sb = rows * 0 + B * H * 128 * 128 * 4 * 2
    emit(c, t, {"kda_core": t, "kda_core_warm": tw}, m,
         f"sdec PreparedKda (conv+recurrence+norm), FP32 state, all {qn} accepted; state r+w {sb / 1e6:.1f} MB "
         f"-> {sb / t / 1e6:.2f} TB/s; conv state BF16 (test layout)")


# ---------------------------------------------------------------- AttnRes
def ares_case(c):
    from vllm.models.kimi_k3.nvidia.ops.attn_res import attn_res
    s = c["shape"]
    R = max(1, math.ceil(s["rows"]))
    op = s["op"]
    Hd = 7168
    prefix = torch.randn(R, Hd, device=dev, dtype=torch.bfloat16)
    delta = torch.randn(R, Hd, device=dev, dtype=torch.bfloat16) * 0.01
    blocks = torch.randn(R, 9, Hd, device=dev, dtype=torch.bfloat16)
    nw = torch.ones(Hd, device=dev, dtype=torch.bfloat16)
    qk = torch.randn(Hd, device=dev, dtype=torch.bfloat16) / Hd**0.5
    onw = 1 + 0.1 * torch.randn(Hd, device=dev, dtype=torch.bfloat16)
    per = {}
    for nb in range(1, 9):
        if op == "L.attnres_post":
            f = lambda nb=nb: attn_res(prefix, delta, blocks, nw, qk, onw, nb, nb, 1e-5, 1e-5)  # noqa: E731
        elif op == "F.attnres_final":
            f = lambda nb=nb: attn_res(prefix, delta, blocks, nw, qk, None, nb, -1, 1e-5, 1e-5)  # noqa: E731
        else:
            f = lambda nb=nb: attn_res(prefix, None, blocks, nw, qk, onw, nb, -1, 1e-5, 1e-5)  # noqa: E731
        per[nb], m = time_us(f)
        if nb in (4, 8):
            per[f"warm{nb}"], _ = time_us(f, cold=False)
    if op == "F.attnres_final":
        t = per[8]
    else:
        t = sum(per[min(8, math.ceil(l / 12))] for l in range(1, 94)) / 93
    emit(c, t, {(f"nb{k}" if isinstance(k, int) else k): v for k, v in per.items()}, m,
         f"{op}: layer mean over l=1..93, sources ceil(l/12); torch.ops._C.kimi_k3_attn_res"
         f" available={hasattr(torch.ops._C, 'kimi_k3_attn_res')}")


for fam in fams:
    cs = [c for c in allcases if c["family"] == FAMILY[fam]]
    log("family", fam, len(cs), "cases")
    for c in cs:
        s = c["shape"]
        try:
            if fam == "idx":
                (idx_prefill if "requests_in_chunk" in s else idx_decode)(c)
            elif fam == "mla":
                mla_case(c)
            elif fam == "kda":
                (kda_prefill if s["phase"] == "prefill" else kda_decode)(c)
            elif fam == "ares":
                if s["op"] in ("L.attnres_pre", "L.attnres_post", "F.attnres_final"):
                    ares_case(c)
                else:
                    out.row(kind="skip", case=c["id"], family=c["family"], shape=s, why="op not benched")
        except Exception as e:  # noqa: BLE001
            fail(c, e)
        torch.cuda.empty_cache()
log("attn shard done")
