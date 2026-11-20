# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA split producer (cpa 8: rows 0-47 by cp.async, rows 48-63 by TMA gather4) vs the
production cp.async form: bitwise on physical mixed lists, then F16K L78K ABBA of full /
gather-only (no math, no MMA) / no gather. usage: python pmla_split.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmla10_check import abba, inputs, run  # noqa: E402

SPLIT = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8)
GONLY = (0, 0, 1, 1)
NOG = (0, 1)


def main():
    for t, short, seed in ((300, True, 1), (149, True, 2), (2048, False, 3), (640, True, 4)):
        inp = inputs(t, 9000, seed=seed, short=short)
        ref = run(*inp, ())
        o = run(*inp, SPLIT)
        torch.cuda.synchronize()
        print(f"phys t={t} short={short} split: equal={torch.equal(o, ref)}", flush=True)
    inp = inputs(16384, 78 * 1024, seed=5)
    arms = [(), SPLIT, GONLY, GONLY + (0,) * 6 + (8,), NOG]
    abba(lambda a: run(*inp, a), arms, "phys F16K L78K")


if __name__ == "__main__":
    main()
