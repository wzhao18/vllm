# SPDX-License-Identifier: Apache-2.0
"""EXPERIMENT ONLY: 1-GPU smoke of mmsol_skel (world = 1: no peers, pushes land locally). Checks the kernel runs
without hangs / faults at n = 2048 / 4096 and prints the per-call time and the per-unit stamps (standalone; NOT an
in-step number). Run: python3 -m vllm.models.kimi_k3.nvidia.megamoe_rubin.mmsol_skel_solo [iters]"""
import sys
import time

import numpy as np
import torch

try:
    from . import mmsol_skel as S
except ImportError:
    import mmsol_skel as S


def main():
    it = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    dev = torch.device("cuda", 0)
    g = torch.Generator(device=dev).manual_seed(0)
    def rnd(nb):
        t = torch.empty(nb, dtype=torch.uint8, device=dev)
        t.random_(0, 256, generator=g)
        return t
    w1 = rnd(E1 := S.E * 6144 * 1792); w2 = rnd(S.E * 3584 * 1536)
    sfw = torch.full((S.E * (6144 * 224 + 3584 * 192),), 0x38, dtype=torch.uint8, device=dev)
    pre = rnd(448 << 20); order = rnd(2048 << 20); fc1 = rnd(128 << 20); sfx = torch.full((64 << 20,), 0x38, dtype=torch.uint8, device=dev)
    sew1 = rnd(12288 * 7168); sew2 = rnd(7168 * 6144); sex1 = rnd(4096 * 7168); x = rnd(4096 * 1792)
    ctr = torch.zeros(1024, dtype=torch.int32, device=dev)
    sk = S.MmsolSkel(None, 0, 1, ctr_ptrs=[ctr.data_ptr()], offsets=[0])
    p = lambda t: (t.data_ptr(), t.numel())
    regions = dict(W1=p(w1), W2=p(w2), SFW=p(sfw), X1=p(pre), X2=p(fc1), SFX=p(sfx), SEW1=p(sew1), SEW2=p(sew2),
                   SFSE=p(sfw), SEX1=p(sex1), ORDER=p(order), PRE=p(pre), FC1OUT=p(fc1), XSRC=p(x))
    for n in (2048, 4096):
        sh = S.shape(n)
        print("shape", sh, flush=True)
        sk.launch(regions, n); torch.cuda.synchronize()
        print(f"n={n}: first call OK", flush=True)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(it):
            e0.record(); sk.launch(regions, n); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
        slot = (sk.calls - 1) % S._RING
        st = sk.stamps[slot].cpu().numpy(); tt = sk.tst[slot].cpu().numpy()
        ent = st[:, 0]; t0 = ent.min()
        fm = st[:, 3][st[:, 3] > 0]; lm = st[:, 4][st[:, 4] > 0]
        ghz = np.median((st[:, 11] - st[:, 10]) / np.maximum(st[:, 7] - st[:, 0], 1))
        stage = []
        for r in tt[:, :64].reshape(-1, 16, 4):
            for k, dt, dc, idx in r:
                if k == 0 and dt == 0: continue
                kind, S_ = int(k) & 255, int(k) >> 8
                if S_ > 1 and dt > 0: stage.append((kind, dc / (S_ - 1), dt / (S_ - 1)))
        stage = np.array(stage) if stage else np.zeros((0, 3))
        print(f"n={n}: call us median {np.median(ts):.1f} min {np.min(ts):.1f} | first MMA +{(fm.min()-t0)/1e3:.1f} us, "
              f"last MMA max +{(lm.max()-t0)/1e3:.1f} (mean {(lm.mean()-t0)/1e3:.1f}) | exit max +{(st[:,7].max()-t0)/1e3:.1f} | "
              f"clock {ghz:.2f} GHz", flush=True)
        for kind, nm in ((1, "FC1"), (2, "FC2"), (0, "SEgu"), (3, "SEdn")):
            s = stage[stage[:, 0] == kind] if len(stage) else stage
            if len(s): print(f"   stage {nm}: {np.median(s[:,1]):.0f} clk / {np.median(s[:,2]):.0f} ns (n={len(s)})")
        ep = tt[:, 64:].reshape(-1, 4); ep = ep[ep[:, 0] >= 0x100]
        for kind, nm in ((1, "FC1"), (2, "FC2"), (0, "SEgu"), (3, "SEdn")):
            e_ = ep[(ep[:, 0] & 7) == kind]
            if len(e_): print(f"   epi {nm}: wait accfull {np.median(e_[:,1])/1e3:.1f} us, body {np.median(e_[:,2])/1e3:.1f} us (p90 {np.percentile(e_[:,2],90)/1e3:.1f}) n={len(e_)}")
    print("SOLO DONE", flush=True)


if __name__ == "__main__":
    main()
