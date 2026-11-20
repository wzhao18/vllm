# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA-2CTA: run sparse_mla_prefill_h8 at the F16K L78K shape a few times (for ncu / clock
measurement). Usage: python3 pmla10_ncu.py [knobs e.g. 0:0:0:0] [--n N]"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmla_cpa2_check import inputs, run  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    knobs = tuple(int(v) for v in args[0].split(":")) if args else ()
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 3
    inp = inputs(16384, 78 * 1024, seed=5)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for i in range(n):
        ev[0].record()
        run(*inp, knobs)
        ev[1].record()
        torch.cuda.synchronize()
        print(f"h8 knobs={knobs} iter {i}: {ev[0].elapsed_time(ev[1]) * 1000:.1f} us", flush=True)


if __name__ == "__main__":
    main()
