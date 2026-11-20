"""Prep register / unroll probe variants of a PREP_ONLY kernel file.

usage: mkreg.py SRC DST REGS UNROLL
REGS: prep setmaxnreg target (48 = production dealloc; > 64 uses an increase,
which only fits when compute/epilogue are idle, i.e. PREP_ONLY probes).
UNROLL: 1 unrolls the gate (8-row) and decoration (row) loops."""
import sys

src, dst, regs, unroll = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
s = open(src).read()
a = "        cute.arch.warpgroup_reg_dealloc(48)\n        inst = (warp - 12) >> 2\n"
assert s.count(a) == 1
op = "warpgroup_reg_alloc" if regs > 64 else "warpgroup_reg_dealloc"
s = s.replace(a, a.replace("warpgroup_reg_dealloc(48)", f"{op}({regs})"))
if unroll:
    for old in ("for rp in cutlass.range(C // 8):", "for wp in cutlass.range(4):"):
        assert s.count(old) == 1, old
        s = s.replace(old, old.replace("cutlass.range(", "cutlass.range_constexpr("))
open(dst, "w").write(s)
print(dst, op, regs, "unroll" if unroll else "")
