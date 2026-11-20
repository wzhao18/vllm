# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase stamps of the drafter's ArNormRows all-reduce + norm sites (wide rows) inside
bench_draft_forward.py's specialized forward (torchrun, one process per GPU):
every compiled ArNormRows gets its own int64 [CTAs, 8] %globaltimer region (entry,
after the PDL wait, after the pushes, after the owner reduces, end); after the run
rank 0 prints per site: CTAs, span (first entry -> last end), first wait-exit -> last
end, and per-CTA medians of wait / push / reduce / finish (poll + norm).

  torchrun --nproc-per-node 4 bench_ar_rows_stamps.py --out DIR --rows 288
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_draft_forward as bench  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (  # noqa: E402
    allreduce as ar,
)

REGION = 1024 * 8
SITES = []


def _patch(buf):
    orig = ar.compile_rows_site.__wrapped__

    def patched(rows, sources, tp, cap, device, trigger=False, push=False, ctas=None):
        k = len(SITES)
        ar.ROWS_STAMPS = buf.data_ptr() + k * REGION * 8
        try:
            out = orig(rows, sources, tp, cap, device, trigger, push, ctas)
        finally:
            ar.ROWS_STAMPS = 0
        SITES.append((rows, sources, tp, trigger, push, ctas))
        return out

    ar.compile_rows_site = patched


def _report(buf):
    s = buf.view(-1, REGION // 8, 8).cpu().double()
    for k, (rows, sources, tp, trigger, push, ctas) in enumerate(SITES):
        x = s[k]
        x = x[x[:, 0] > 0]
        if x.shape[0] == 0:
            print(f"site {k} rows {rows} src {sources} push {push} ctas {ctas}: not run")
            continue
        t0 = x[:, 0].min()
        end = x[:, 4]
        span = (end.max() - t0) / 1e3
        post = (end.max() - x[:, 1].min()) / 1e3

        def med(a, b):
            v = x[:, b] - x[:, a]
            v = v[(x[:, a] > 0) & (x[:, b] > 0)]
            return float(v.median()) / 1e3 if v.numel() else float("nan")

        fin_from = 3 if push else 1
        print(
            f"site {k} rows {rows} src {sources} tp {tp} push {push} trig {trigger} "
            f"ctas {ctas}: CTAs {x.shape[0]} span {span:.2f} us, wait-exit->end "
            f"{post:.2f} | median wait {med(0, 1):.2f} push {med(1, 2):.2f} "
            f"reduce {med(2, 3):.2f} finish {med(fin_from, 4):.2f} | start spread "
            f"{(x[:, 0].max() - t0) / 1e3:.2f} wait-exit spread "
            f"{(x[:, 1].max() - x[:, 1].min()) / 1e3:.2f}",
            flush=True,
        )


def main():
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    buf = torch.zeros(64 * REGION, dtype=torch.int64, device="cuda")
    _patch(buf)
    bench.main()
    torch.cuda.synchronize()
    if int(os.environ.get("RANK", 0)) == 0:
        _report(buf)


if __name__ == "__main__":
    main()
