"""P1 variant: one epilogue warp polls MB_OFIN, the other three sleep on a
named barrier and then pass their own (already complete) mbarrier wait, so
each warp still acquires the MMA result through the mbarrier. Bitwise math.

usage: mkp1.py SRC DST
MB_OFIN is the top retry site of M1 (152 TRYWAIT+PHASECHK retries per BT32
chunk from the four epilogue warps).
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
reps = [
    ("        ep_bar = pipeline.NamedBarrier(barrier_id=6, num_threads=128)\n",
     "        ep_bar = pipeline.NamedBarrier(barrier_id=6, num_threads=128)\n"
     "        ofin_bar = pipeline.NamedBarrier(barrier_id=13, num_threads=128)\n", 1),
    ("                    if cutlass.const_expr(GATE2_ == 1):\n"
     "                        cute.arch.mbarrier_wait(mb + MB_OFIN + cse, pe_fin)\n"
     "                    cute.copy(o16_cp, out_thr.partition_S(t_out), r_o)\n",
     "                    if cutlass.const_expr(GATE2_ == 1):\n"
     "                        if warp == 4:\n"
     "                            cute.arch.mbarrier_wait(mb + MB_OFIN + cse, pe_fin)\n"
     "                        ofin_bar.arrive_and_wait()\n"
     "                        if warp != 4:\n"
     "                            cute.arch.mbarrier_wait(mb + MB_OFIN + cse, pe_fin)\n"
     "                    cute.copy(o16_cp, out_thr.partition_S(t_out), r_o)\n", 1),
]
for a, b, n in reps:
    c = s.count(a)
    if c != n:
        sys.exit(f"pattern count {c} != {n}: {a[:70]!r}")
    s = s.replace(a, b)
open(dst, "w").write(s)
print("wrote", dst)
