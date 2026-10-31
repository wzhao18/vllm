"""DSpark drafter dense-MLA decode on one TP1 (DEP) rank: the current TokenSpeed SM100 FP8 path (uniform floor
min_split_kv + W6 M5 length-proportional splits) vs the SM107 M256 kernel (4-CTA clusters, CLC) with BF16 output.

usage: python3 bench_draft_attn.py --shapes <bench shapes.json> [--cases q8_r16_agx,...] [--qlen 8] [--heads 64]
Timing: CUDA graph of N back-to-back launches, median of samples (KV is GBs, so no L2 reuse across launches)."""
import argparse, inspect, json, math, statistics
import torch
import tokenspeed_mla
from tokenspeed_mla import mla_decode as md

D_LAT, D_ROPE, PAGE = 512, 64, 64
# tokenspeed-mla 2.0.2 (image) needs enable_packed_q=True; the W6 overlay build always packs
_SM107_EXTRA = ({"enable_packed_q": True}
                if "enable_packed_q" in inspect.signature(md._sm107_fp8_out_decode).parameters else {})


def rand_fp8(shape, dev, g):
    """E4M3 bytes of magnitude ~2^-5..2^0 with random sign, without an FP32 temporary (KV is GBs)."""
    b = torch.randint(0x18, 0x38, shape, dtype=torch.uint8, device=dev, generator=g)
    b |= (torch.randint(0, 2, shape, dtype=torch.uint8, device=dev, generator=g) << 7)
    return b.view(torch.float8_e4m3fn)


def make(ctxs, qlen, heads, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    B = len(ctxs)
    pages = [math.ceil(c / PAGE) for c in ctxs]
    tot = sum(pages)
    kv = rand_fp8((tot, PAGE, D_LAT + D_ROPE), dev, g)
    maxp = max(pages)
    bt = torch.zeros(B, maxp, dtype=torch.int32, device=dev)
    off = 0
    perm = torch.randperm(tot, device=dev, generator=g).to(torch.int32)  # scattered pages, like a real pool
    for i, p in enumerate(pages):
        bt[i, :p] = perm[off:off + p]; off += p
    q = rand_fp8((B, qlen, heads, D_LAT + D_ROPE), dev, g)
    seq = torch.tensor(ctxs, dtype=torch.int32, device=dev)
    return q, kv, bt, seq


def current(q, kv, bt, seq, maxlen, ws, min_split, max_split, out):
    """The TokenspeedMLAImpl call (tokenspeed_mla.py) under the W6 M5 recipe."""
    B, qlen, H, _ = q.shape
    kw = {"min_split_kv": min_split}
    if max_split > 1:
        cap = min(max_split, -(-maxlen // 128))
        if cap > max(min_split, 1):
            counts = tokenspeed_mla.tokenspeed_mla_var_split_counts(seq, cap, H, qlen, q.dtype,
                                                                    min_split * B if min_split > 1 else 0)
            kw = {"max_split_kv": cap, "block_split_kvs": counts}
    return tokenspeed_mla.tokenspeed_mla_decode(
        query=q, kv_cache=kv, workspace_buffer=ws, kv_lora_rank=D_LAT, qk_rope_head_dim=D_ROPE, block_tables=bt,
        seq_lens=seq, max_seq_len=maxlen, softmax_scale=0.05, output_scale=1.0, out=out, enable_pdl=False,
        causal_mask=False, **kw)


def sm107(q, kv, bt, seq, maxlen, ws, min_split, out):
    return md._sm107_fp8_out_decode(
        q_latent=q[..., :D_LAT], q_rope=q[..., D_LAT:], c_latent=kv[..., :D_LAT], c_rope=kv[..., D_LAT:],
        block_tables=bt, seq_lens=seq, workspace_buffer=ws, out=out, page_size=PAGE, max_seq_len=maxlen,
        softmax_scale=0.05, output_scale=1.0, is_var_seq=True, causal_mask=False, window_left=-1, enable_pdl=False,
        return_lse=False, causal_seqs=None, cp_world=1, cp_rank=0, local_visible_lens=None, min_split_kv=min_split,
        compute_capability=torch.cuda.get_device_capability(q.device), **_SM107_EXTRA)


def reference(q, kv, bt, seq, i, out_scale=1.0, sm_scale=0.05):
    """FP32 non-causal MLA for request i: softmax(q . k * scale) . v, v = latent part of the cache."""
    n = int(seq[i]); pages = -(-n // PAGE)
    kvi = kv[bt[i, :pages].long()].reshape(-1, D_LAT + D_ROPE)[:n].float()
    qi = q[i].float().reshape(-1, D_LAT + D_ROPE)                     # [qlen*H, 576]
    s = (qi @ kvi.T) * sm_scale
    p = torch.softmax(s, dim=-1)
    return (p @ kvi[:, :D_LAT] * out_scale).reshape(q.shape[1], q.shape[2], D_LAT)


def errs(o, ref):
    d = (o.float() - ref)
    return dict(maxrel=(d.abs().max() / ref.abs().max()).item(), rmsrel=(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item())


def time_graph(fn, n=5, samples=7):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(samples):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / n)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True); ap.add_argument("--cases", default="")
    ap.add_argument("--qlen", default="")  # default: the case's query_len
    ap.add_argument("--heads", type=int, default=64); ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--min-split", type=int, default=16); ap.add_argument("--max-split", type=int, default=32)
    ap.add_argument("--sm107-min-split", default="1"); ap.add_argument("--json")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    cases = json.load(open(a.shapes))["cases"]
    want = [c for c in a.cases.split(",") if c]
    res = []
    ws = torch.empty(3 << 30, dtype=torch.int8, device=dev)
    for c in cases:
        if want and c["name"] not in want:
            continue
        ctxs = [r["kv_len"] for r in c["requests"]]
        for qlen in ([int(x) for x in a.qlen.split(",")] if a.qlen else [c["requests"][0]["query_len"]]):
            q, kv, bt, seq = make(ctxs, qlen, a.heads, dev)
            maxlen = max(ctxs)
            o1 = torch.empty(len(ctxs), qlen, a.heads, D_LAT, dtype=torch.bfloat16, device=dev)
            o2 = torch.empty_like(o1)
            t_cur = time_graph(lambda: current(q, kv, bt, seq, maxlen, ws, a.min_split, a.max_split, o1))
            row = dict(case=c["name"], B=len(ctxs), qlen=qlen, heads=a.heads, mtok=sum(ctxs) / 1e6,
                       cur_us=t_cur)
            flops = 2 * a.heads * qlen * sum(ctxs) * (D_LAT + D_ROPE + D_LAT)
            row["cur_pf"] = flops / t_cur / 1e9
            for ms in [int(x) for x in a.sm107_min_split.split(",")]:
                try:
                    t = time_graph(lambda: sm107(q, kv, bt, seq, maxlen, ws, ms, o2))
                    current(q, kv, bt, seq, maxlen, ws, a.min_split, a.max_split, o1)
                    sm107(q, kv, bt, seq, maxlen, ws, ms, o2)
                    torch.cuda.synchronize()
                    err = ((o1.float() - o2.float()).abs().max() / o1.float().abs().max().clamp_min(1e-6)).item()
                    row[f"sm107_s{ms}_us"] = t; row[f"sm107_s{ms}_pf"] = flops / t / 1e9; row[f"sm107_s{ms}_relerr"] = err
                    if a.check:
                        for i in (0, len(ctxs) - 1):
                            ref = reference(q, kv, bt, seq, i)
                            row[f"ref{i}_cur"] = errs(o1[i], ref); row[f"ref{i}_sm107_s{ms}"] = errs(o2[i], ref)
                            del ref
                except Exception as e:  # noqa: BLE001
                    row[f"sm107_s{ms}_err"] = repr(e)[:200]
            row["per_step_layers"] = a.layers
            print(json.dumps(row), flush=True)
            res.append(row)
            del q, kv, bt, seq, o1, o2
            torch.cuda.empty_cache()
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
