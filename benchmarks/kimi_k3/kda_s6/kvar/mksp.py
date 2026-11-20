"""Variant generator (P-KDA2-S6 r9 E2-S): prep instance publishes MB_QK per warp (split publish) and the raw g/k
TMA issue + RFREE wait moves from instance warp 0 (the solve/combine warp) to warp 3.

Today every instance warp waits at the chunk's final ibar for warp 0's solve + off-diagonal combine before warp 0
arrives MB_QK; the next chunk's raw g/k TMA is issued by warp 0 only after that. With MB_QK counting 4 warp
arrivals (each warp fences its async-proxy smem writes, then one lane arrives), warps 1-3 leave the chunk as soon as
their own restore is done and warp 3 issues the next chunk's g/k TMA while warp 0 still solves. Math is unchanged
(bitwise vs the input kernel expected).

usage: mksp.py IN OUT
"""
import sys

src = open(sys.argv[1]).read()

old_init = "cute.arch.mbarrier_init(mb + MB_QK + i, 1)"
assert src.count(old_init) == 1, src.count(old_init)
src = src.replace(old_init, "cute.arch.mbarrier_init(mb + MB_QK + i, 4)")

pub_old = """                    cute.arch.fence_proxy("async.shared", space="cta")
                    ibar.arrive_and_wait()
                    if plw == 0:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(mb + MB_QK + stg)"""
pub_new = """                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mb + MB_QK + stg)"""
assert src.count(pub_old) == 1, src.count(pub_old)
src = src.replace(pub_old, pub_new)

tma_old = """                    if tma:
                        if plw == 0:
                            cute.arch.mbarrier_wait(mb + MB_RFREE + stg, pp_rfree)"""
tma_new = """                    if tma:
                        if plw == 3:
                            cute.arch.mbarrier_wait(mb + MB_RFREE + stg, pp_rfree)"""
assert src.count(tma_old) == 1, src.count(tma_old)
src = src.replace(tma_old, tma_new)
open(sys.argv[2], "w").write(src)
print("mksp ok", sys.argv[2])
