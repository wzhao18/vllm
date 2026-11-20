"""Variant generator: mbarrier waits that sleep in hardware (long suspend hint).

usage: mkquiet.py IN OUT {prep|all}
The polling probe (n4P: +2.3..5.9% for 4 extra polling warps) says re-polling
mbarrier waits tax the shared issue/MIO path.  try_wait.parity with a large
suspendTimeHint parks the thread until the phase completes (no retry loop).
'prep' converts only the prep-instance waits (prep has slack); 'all' converts
every mbarrier wait in the kernel.
"""
import sys

src = open(sys.argv[1]).read()
mode = sys.argv[2 + 1] if len(sys.argv) > 3 else "prep"

helper = '''

@dsl_user_op
def _mbar_wait_quiet(mbar, phase, *, loc=None, ip=None):
    """Parity wait parked in hardware (long suspend hint), no re-poll loop."""
    addr = llvm.ptrtoint(T.i32(), mbar.to_llvm_ptr(loc=loc, ip=ip), loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [addr, cutlass.Int32(phase).ir_value(loc=loc, ip=ip)],
        "{\\n.reg .pred P1;\\nLAB_WAIT_${:uid}:\\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1, 10000000;\\n"
        "@!P1 bra LAB_WAIT_${:uid};\\n}",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
'''
anchor = "\n\n@dsl_user_op\ndef _load_state_f32x8("
assert src.count(anchor) == 1
src = src.replace(anchor, helper + anchor)

lines = src.split("\n")
i_prep = next(i for i, l in enumerate(lines) if l.startswith("    if warp >= 12:"))
n = 0
for i, l in enumerate(lines):
    if "cute.arch.mbarrier_wait(" in l and (mode == "all" or i > i_prep):
        lines[i] = l.replace("cute.arch.mbarrier_wait(", "_mbar_wait_quiet(")
        n += 1
open(sys.argv[2], "w").write("\n".join(lines))
print("converted", n, "waits")
