"""P-MLA debug: h8 logical mode on small cases, one subprocess per case with a timeout
(finds hangs). Usage: python3 pmla_dbg_logical.py [case]"""
import os
import subprocess
import sys

PHYS_ONLY = bool(os.environ.get("PMLA_PHYS_ONLY"))  # debug script only

CASES = {
    "one_full": ([2048], 1),
    "sm_full": ([2048] * 148, 1),
    "two_per_cta": ([2048] * 296, 1),
    "short1": ([1] * 4, 1),
    "zero": ([0, 2048, 0, 5], 1),
    "mixed_small": ([2048, 1, 2047, 129, 0, 2048, 64, 1000] * 4, 1),
    "full_short": ([2048, 1] * 148, 1),
    "short_full": ([1, 2048] * 148, 1),
    "zero_full": ([0, 2048] * 148, 1),
    "full3": ([2048] * 444, 1),
    "short3": ([1] * 444, 1),
    "mixed_many": ([2048, 1, 2047, 129, 0, 2048, 64, 1000] * 40, 1),
    "mixed_many_bs32": ([2048, 1, 2047, 129, 0, 2048, 64, 1000] * 40, 0),
}


def run(case):
    import torch

    from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import (
        sparse_mla_prefill_h8,
    )

    lengths, bs64 = CASES[case]
    block_size = 64 if bs64 else 32
    g = torch.Generator(device="cuda").manual_seed(1)
    stride, blocks, reqs, width, ROW, H = 64, 900, 3, 2048, 576, 96
    ctx_blocks = [400, 120, 300]
    kv = (torch.randn(blocks * stride, ROW, device="cuda", generator=g) * 0.8).to(
        torch.float8_e4m3fn)
    bt = torch.zeros((reqs, 512), dtype=torch.int32, device="cuda")
    perm = torch.randperm(blocks, device="cuda", generator=g).to(torch.int32)
    start = 0
    for r, nb in enumerate(ctx_blocks):
        bt[r, :nb] = perm[start:start + nb]
        start += nb
    lengths = list(lengths)
    t = len(lengths)
    req_ids = torch.arange(t, device="cuda", dtype=torch.int32) % reqs
    topk = torch.full((t, width), -1, dtype=torch.int32, device="cuda")
    for i, n in enumerate(lengths):
        n = min(n, ctx_blocks[i % reqs] * block_size)
        lengths[i] = n
        if n:
            topk[i, :n] = torch.randperm(ctx_blocks[i % reqs] * block_size, device="cuda",
                                         generator=g)[:n].to(torch.int32)
    q8 = (torch.randn(t, H, ROW, device="cuda", generator=g) * 0.8).to(torch.float8_e4m3fn)
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    valid = topk >= 0
    blk = bt[req_ids.long()[:, None], (topk.clamp(min=0) // block_size).long()]
    slots = torch.where(valid, blk * stride + topk.clamp(min=0) % block_size, -1)
    ref = sparse_mla_prefill_h8(q8, kv, slots.to(torch.int32), lens, 0.2, 1.3)
    torch.cuda.synchronize()
    print(case, "physical ok", flush=True)
    if PHYS_ONLY:
        return
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import (
        sparse_mla_prefill_h8_logical,
    )
    out = sparse_mla_prefill_h8_logical(q8, kv, topk, req_ids, bt, block_size, stride, 0.2,
                                        1.3)
    torch.cuda.synchronize()
    print(case, "logical equal", bool(torch.equal(out, ref)), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        import time
        t0 = time.time()
        run(sys.argv[1])
        print(sys.argv[1], "secs", round(time.time() - t0), flush=True)
    else:
        for c in CASES:
            try:
                r = subprocess.run([sys.executable, __file__, c], timeout=240,
                                   capture_output=True, text=True)
                tail = [ln for ln in (r.stdout + r.stderr).strip().splitlines()
                        if c in ln or "Error" in ln][-3:]
                print(c, "rc", r.returncode, "|", " / ".join(tail)[-300:], flush=True)
            except subprocess.TimeoutExpired as e:
                out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
                print(c, "TIMEOUT (hang)", [ln for ln in out.splitlines() if c in ln][-1:],
                      flush=True)
