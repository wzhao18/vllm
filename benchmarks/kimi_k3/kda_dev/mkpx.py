"""PX probe generator (timing only, numerically wrong): prep supply x N.

usage: mkpx.py IN OUT N
Each prep instance runs its full prep on every N-th lap of its stage only; on
the other laps it runs only the stage protocol (RFREE/SFREE wait, instance
barrier, QK arrive) and the chain consumes the stale stage. This emulates a
prep-export helper CTA that supplies the other laps' stage images, without
the import cost. Upper bound of the PX design for chain-bound shapes.
"""
import sys

src = open(sys.argv[1]).read()
n = int(sys.argv[3])

head = "                    it = phase_base + phase_it\n"
end = "                        pp_qkraw ^= 1\n"
i = src.index(head) + len(head)
assert src.count(head) == 1
j = src.index(end, i) + len(end)
body = src[i:j]
ind = "\n".join(("    " + ln) if ln.strip() else ln for ln in body.split("\n"))
skip = f"""                    if (it % {n}) != 0:
                        if plw == 0:
                            cute.arch.mbarrier_wait(mb + MB_RFREE + inst, pp_rfree)
                            cute.arch.mbarrier_wait(mb + MB_SFREE + inst, pp_sfree)
                        ibar.arrive_and_wait()
                        if plw == 0:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(mb + MB_QK + inst)
                        pp_rfree ^= 1
                        pp_sfree ^= 1
                    else:
"""
src = src[:i] + skip + ind + src[j:]
open(sys.argv[2], "w").write(src)
print("ok px", n)
