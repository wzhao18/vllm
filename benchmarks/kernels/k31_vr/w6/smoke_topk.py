# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quick compile + correctness smoke for balanced_topk (prints the full
Triton compile-error chain)."""
import traceback

import torch

from vllm.v1.attention.ops.balanced_topk import balanced_topk

try:
    torch.manual_seed(0)
    lens = [5, 3000, 20000, 400000]
    logits = torch.randn(len(lens), 400000, device="cuda").to(torch.bfloat16)
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda").view(-1, 1)
    out = torch.empty(len(lens), 2048, dtype=torch.int32, device="cuda")
    balanced_topk(logits, seq, 1, out, 2048)
    torch.cuda.synchronize()
    for r, n in enumerate(lens):
        k = min(n, 2048)
        sel = out[r, :k].long()
        ok = sel.unique().numel() == k and (out[r, k:] == -1).all().item()
        if n > 2048:
            ok = ok and torch.equal(logits[r, sel].float().sort().values,
                                    logits[r, :n].float().topk(k).values.sort().values)
        print("row", r, n, "ok" if ok else "BAD")
except Exception as e:
    traceback.print_exc()
    c = e
    while c is not None:
        print("CAUSE:", type(c).__name__, str(c)[:3000])
        c = c.__cause__ or c.__context__
