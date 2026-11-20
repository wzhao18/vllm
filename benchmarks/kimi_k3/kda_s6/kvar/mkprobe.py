"""Make PREP_ONLY probe variants.

usage: mkprobe.py SRC DST VALUE
If SRC lacks the PREP_ONLY probe edits, they are applied (same edits as the
kernel on wzhao/k31ws-p-kda2-s6-d1); then PREP_ONLY is set to VALUE."""
import re
import sys

src, dst, val = sys.argv[1], sys.argv[2], int(sys.argv[3])
s = open(src).read()
if "PREP_ONLY = " not in s:
    reps = [
        ("MB_GRAM = 71\n", "MB_GRAM = 71\nPREP_ONLY = 0\n"),
        ("    if warp < 4:\n        if cutlass.const_expr(REG_MODE_ == 1):",
         "    if warp < (0 if PREP_ONLY else 4):\n        if cutlass.const_expr(REG_MODE_ == 1):"),
        ("    if (warp >= 4) & (warp < 8):\n        if cutlass.const_expr(REG_MODE_ == 2):",
         "    if (warp >= 4) & (warp < (4 if PREP_ONLY else 8)):\n"
         "        if cutlass.const_expr(REG_MODE_ == 2):"),
        ("        if warp == mma_warp:\n            if cutlass.const_expr(True):\n"
         "                csm = cutlass.Int32(0)\n                pm_oe = cutlass.Int32(1)",
         "        if warp == mma_warp:\n            if cutlass.const_expr(PREP_ONLY == 1):\n"
         "                csm = cutlass.Int32(0)\n                pm_qk = cutlass.Int32(0)\n"
         "                for kx in cutlass.range(k1 - k0):\n"
         "                    seq_len = cute.arch.make_warp_uniform(sptn[k0 + kx])\n"
         "                    for _t in cutlass.range((seq_len + C - 1) // C):\n"
         "                        cute.arch.mbarrier_wait(mb + MB_QK + csm, pm_qk)\n"
         "                        with cute.arch.elect_one():\n"
         "                            cute.arch.mbarrier_arrive(mb + MB_RFREE + csm)\n"
         "                            cute.arch.mbarrier_arrive(mb + MB_SFREE + csm)\n"
         "                        csm += 1\n                        if csm == STAGES:\n"
         "                            csm = 0\n                            pm_qk ^= 1\n"
         "            else:\n                csm = cutlass.Int32(0)\n                pm_oe = cutlass.Int32(1)"),
        ("        if warp == 10:\n            csl = cutlass.Int32(0)",
         "        if warp == (-1 if PREP_ONLY else 10):\n            csl = cutlass.Int32(0)"),
    ]
    for a, b in reps:
        assert s.count(a) == 1, a[:60]
        s = s.replace(a, b)
s = re.sub(r"^PREP_ONLY = \d+$", f"PREP_ONLY = {val}", s, count=1, flags=re.M)
open(dst, "w").write(s)
print(dst, "PREP_ONLY =", val)
