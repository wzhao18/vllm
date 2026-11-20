#!/usr/bin/env python3
"""P-GEMM R8: one profiled call of one ledger row (pgemm_r8_ledger.make) for ncu.

    ncu --profile-from-start off ... python3 pgemm_r8_ncu.py --row <name> [--m 16384]

Warm-up calls run outside the profiled window; exactly one call runs inside
cudaProfilerStart / Stop, so the report holds that call's kernels only.
"""

import argparse

import torch
from pgemm_r8_ledger import make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--m", type=int, default=16384)
    a = ap.parse_args()
    fn = make(a.row, a.m, torch.device("cuda"))
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
