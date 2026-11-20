"""Variant generator: shorten the stage-ring lookahead by one step (timing probe
for the step model P = max(chain step, W / L)).

The compute warpgroup additionally waits MB_QK of the NEXT chunk's stage
before starting chunk t, so prep must finish one chunk earlier (L -> L - 1).
Numerics unchanged (bitwise): only an extra wait.

usage: mklook.py IN OUT
"""
import sys

src = open(sys.argv[1]).read()
old = (
    "            for t in cutlass.range(t_tiles):  # noqa: B007\n"
    "                cute.arch.mbarrier_wait(mb + MB_QK + csc, p_qk)\n"
)
new = old + (
    "                if (t + 1 < t_tiles) | (kx + 1 < k1 - k0):\n"
    "                    csx = csc + 1\n"
    "                    pxx = p_qk\n"
    "                    if csx == STAGES:\n"
    "                        csx = 0\n"
    "                        pxx = p_qk ^ 1\n"
    "                    cute.arch.mbarrier_wait(mb + MB_QK + csx, pxx)\n"
)
assert src.count(old) == 1
src = src.replace(old, new)
open(sys.argv[2], "w").write(src)
print("ok look")
