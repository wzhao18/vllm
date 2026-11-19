"""G-PERSIST ncu probe: cuBLAS chained GEMMs (graph) at the shared / latent shapes and the
persistent 2-phase chain, one NVTX range per case, for ncu --graph-profiling node."""
import torch
import torch.nn.functional as F

import pg_bench as pb


def graph(fn_list):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fn_list:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fn_list:
            f()
    return g


def main():
    dev = "cuda"
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    ext = pb.build()
    ws = torch.empty(sms * 2 * 128 * 128, dtype=torch.float32, device=dev)
    no = torch.empty(0, dtype=torch.int64, device=dev)
    for name, phases in pb.CHAINS:
        copies = 4
        W = [[torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.03 for n, k, _ in phases]
             for _ in range(copies)]
        Wp = [[pb.interleave_gu(w[: n // 2], w[n // 2:]) if a else w
               for w, (n, k, a) in zip(ws_, phases)] for ws_ in W]
        acts = [a for _, _, a in phases]
        for r in (32, 96):
            x = torch.randn(r, phases[0][1], device=dev, dtype=torch.bfloat16)
            st = pb.stages_for(ext, (r + 15) // 16 * 16)
            tout = [[torch.empty(r, n, device=dev, dtype=torch.bfloat16) for n, k, a in phases]
                    for _ in range(copies)]
            outs = [[torch.empty(r, n // 2 if a else n, device=dev, dtype=torch.bfloat16)
                     for n, k, a in phases] for _ in range(copies)]
            cnts = [[torch.zeros(n // 128, dtype=torch.int32, device=dev) for n, k, a in phases]
                    for _ in range(copies)]
            bars = [torch.zeros(2, dtype=torch.int32, device=dev) for _ in range(copies)]

            def tchain(c):
                h = x
                for p, (n, k, a) in enumerate(phases):
                    torch.mm(h, W[c][p].t(), out=tout[c][p])
                    h = tout[c][p]
                    if a:
                        h = F.silu(h[:, : n // 2]) * h[:, n // 2:]

            def pchain(c):
                ext.prun(Wp[c], x, outs[c], acts, ws, cnts[c], bars[c], sms, st, 12, no)

            for tag, fn in (("torch", tchain), ("persist", pchain)):
                g = graph([lambda c=c: fn(c) for c in range(copies)])
                for _ in range(2):
                    g.replay()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(f"{name}_r{r}_{tag}")
                for _ in range(3):
                    g.replay()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    main()
