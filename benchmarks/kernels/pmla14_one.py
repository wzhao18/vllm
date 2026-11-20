# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA-2CTA round 5: one h8 variant vs the base kernel (bitwise), case by case with a
progress line before each launch, so a hanging variant (run under `timeout`) shows the
case that hangs.
Usage: python3 pmla14_one.py <knob list, e.g. 0:0:...:3> <case> [<case> ...]
cases: p300s p149s p2048 p640s (physical, pmla10_check lists), l64 l32 (logical)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmla10_check import logical_case, run_l  # noqa: E402
from pmla_cpa2_check import inputs, run  # noqa: E402

CASES = {"p300s": (300, True, 1), "p149s": (149, True, 2), "p2048": (2048, False, 3),
         "p640s": (640, True, 4)}


def main():
    k = tuple(int(v) for v in sys.argv[1].split(":"))
    for case in sys.argv[2:]:
        if case in CASES:
            t, short, seed = CASES[case]
            inp = inputs(t, 9000, seed=seed, short=short)
            ref = run(*inp, ())
            torch.cuda.synchronize()
            print(f"{case}: base done, launching {k}", flush=True)
            o = run(*inp, k)
        else:
            bs = int(case[1:])
            c = logical_case([2048, 1, 2047, 129, 0, 2048, 64, 1000] * 40, [400, 120, 300],
                             block_size=bs, blocks=900)
            ref = run_l(c, ())
            torch.cuda.synchronize()
            print(f"{case}: base done, launching {k}", flush=True)
            o = run_l(c, k)
        torch.cuda.synchronize()
        d = (o.float() - ref.float()).abs().max().item()
        print(f"{case} knobs={k}: equal={torch.equal(o, ref)} maxdiff={d:.3g}", flush=True)


if __name__ == "__main__":
    main()
