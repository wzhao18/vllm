"""DSA decode top-k on one TP1 (DEP) rank at the owner shapes: rows = requests x q (next_n = q), BF16 logits
[rows, max ctx], per-row lengths of the AgentX contexts; exact top-2048 vs torch.topk.

usage: python3 bench_topk.py --shapes <bench shapes.json> [--cases ...] [--backends balanced,cooperative,persistent]"""
import argparse, json, statistics
import torch

TOPK = 2048


def time_graph(fn, n=10, samples=7):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(samples):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1e3 / n)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True); ap.add_argument("--cases", default="")
    ap.add_argument("--backends", default="balanced,cooperative,persistent"); ap.add_argument("--json")
    a = ap.parse_args()
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk
    from vllm.v1.worker.workspace import init_workspace_manager
    dev = torch.device("cuda")
    try:
        init_workspace_manager(dev)
    except Exception:  # noqa: BLE001
        pass
    want = [c for c in a.cases.split(",") if c]
    res = []
    with set_current_vllm_config(VllmConfig()):
        for c in json.load(open(a.shapes))["cases"]:
            if want and c["name"] not in want:
                continue
            q = c["requests"][0]["query_len"]
            ctx = [r["kv_len"] for r in c["requests"]]
            rows, width = len(ctx) * q, max(ctx)
            width = (width + 127) // 128 * 128
            g = torch.Generator(device=dev).manual_seed(0)
            logits = torch.randn(rows, width, device=dev, generator=g, dtype=torch.bfloat16)
            seq = torch.tensor(ctx, dtype=torch.int32, device=dev)
            ends = (seq.view(-1, 1) - q + 1 + torch.arange(q, device=dev, dtype=torch.int32)).reshape(-1)
            # reference: exact top-k of each row's valid prefix (as a set; ties broken arbitrarily)
            ref = []
            for i in range(rows):
                ref.append(torch.topk(logits[i, :ends[i]].float(), min(TOPK, int(ends[i]))).values.min().item())
            for be in a.backends.split(","):
                op = SparseIndexerTopk(be)
                out = torch.empty(rows, TOPK, dtype=torch.int32, device=dev)
                fn = lambda: op(logits, seq, q, out, TOPK, width)  # noqa: E731
                try:
                    t = time_graph(fn)
                    fn(); torch.cuda.synchronize()
                    ok = True
                    for i in range(0, rows, max(1, rows // 16)):
                        idx = out[i][out[i] >= 0].long()
                        if idx.numel() != min(TOPK, int(ends[i])) or (idx >= ends[i]).any() or \
                                logits[i, idx].float().min().item() < ref[i] or idx.unique().numel() != idx.numel():
                            ok = False
                    row = dict(case=c["name"], backend=be, rows=rows, mtok=sum(ctx) / 1e6, us=t, exact=ok)
                except Exception as e:  # noqa: BLE001
                    row = dict(case=c["name"], backend=be, err=repr(e)[:300])
                print(json.dumps(row), flush=True); res.append(row)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
