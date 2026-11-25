# SOL M1 probes (e) + (g) for designs/dsa-layer.md (NOT production code): Lamport multicast rounds on the
# W GPUs of one node. torchrun --nproc-per-node 4 m1lam.py
# Sizes are the 16-rank busiest-receiver ingress (KB) of the design; on W GPUs each of the W-1 peers sends
# ingress / (W-1) so the receiver ingress matches (sender egress is (16-1)/(W-1) x the 16-rank one: conservative).
import json
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from m1dsa import ext, stats

CASES = {  # B: (top-k scores ingress KB, q publish ingress KB) from designs/dsa-layer.md section 1
    1: (46, 415), 2: (92, 829), 4: (184, 1659), 8: (369, 3318), 16: (737, 6636), 32: (1475, 13271),
}
REPS = int(os.environ.get("M1_REPS", "40"))


def main():
    dist.init_process_group("nccl")
    rank, W = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    E = ext()
    grp = dist.group.WORLD
    maxw = max(1, int(max(v[1] for v in CASES.values()) * 1000 / (W - 1) / 4)) + 1024
    maxt = max(1, int(max(v[0] for v in CASES.values()) * 1000 / (W - 1) / 4)) + 1024
    tk = symm_mem.empty(W * maxt, dtype=torch.int32, device="cuda"); tk.fill_(-1)
    qb = symm_mem.empty(W * maxw, dtype=torch.int32, device="cuda"); qb.fill_(-1)
    bar = symm_mem.empty(64, dtype=torch.int32, device="cuda"); bar.fill_(-1)
    ht, hq, hb = (symm_mem.rendezvous(t, grp.group_name) for t in (tk, qb, bar))
    mc = [int(getattr(h, "multicast_ptr", 0) or 0) for h in (ht, hq, hb)]
    if rank == 0:
        print(f"LAM W {W} mc {[hex(x) for x in mc]}", flush=True)
    assert all(mc), "no multicast"
    epoch = 1
    torch.cuda.synchronize(); dist.barrier()
    for B, (tkk, qk) in CASES.items():
        tkw = (int(tkk * 1000 / (W - 1)) // 16) * 4  # u32 words per sender, multiple of 4
        qw = (int(qk * 1000 / (W - 1)) // 16) * 4
        n_tk = min(128, 32 * B)
        n_q = min(212 - n_tk, max(24, 24 * B))
        for mode_name, mode, lead in (("tk", 1, 0), ("q", 2, 0), ("tk+q", 3, 0), ("q_lead1us+tk", 3, 1000)):
            st = torch.zeros((n_tk + n_q, 4), dtype=torch.int64, device="cuda")
            tk_rt, q_rt = [], []
            for it in range(REPS + 3):
                dist.barrier(); torch.cuda.synchronize()
                epoch += 1
                E.lam_run(mc[0], tk.data_ptr(), tkw, mc[1], qb.data_ptr(), qw, mc[2], bar.data_ptr(), W, rank,
                          n_tk, n_q, mode, lead, epoch, st)
                torch.cuda.synchronize()
                if it < 3:
                    continue
                s = st.cpu()
                t0 = int(s[:, 0].min())
                if mode & 1:
                    tk_rt.append((int(s[:n_tk, 2].max()) - t0) / 1e3 - (lead / 1e3 if mode == 3 and lead else 0))
                if mode & 2:
                    q_rt.append((int(s[n_tk:, 2].max()) - t0) / 1e3)
            # gather per-rank medians
            out = torch.tensor([stats(tk_rt)["med"] if tk_rt else -1, stats(q_rt)["med"] if q_rt else -1,
                                stats(tk_rt)["p90"] if tk_rt else -1], device="cuda", dtype=torch.float64)
            allr = [torch.zeros_like(out) for _ in range(W)]
            dist.all_gather(allr, out)
            if rank == 0:
                a = torch.stack(allr).cpu()
                print("REC " + json.dumps(dict(probe="lam", W=W, B=B, mode=mode_name, tk_KB16=tkk, q_KB16=qk,
                                                tk_bytes_per_sender=tkw * 4, q_bytes_per_sender=qw * 4, n_tk=n_tk,
                                                n_q=n_q, q_lead_us=lead / 1e3,
                                                tk_round_us_med_by_rank=[round(x, 3) for x in a[:, 0].tolist()],
                                                q_round_us_med_by_rank=[round(x, 3) for x in a[:, 1].tolist()],
                                                tk_round_us_p90_by_rank=[round(x, 3) for x in a[:, 2].tolist()])),
                      flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
