
## 2026-11-18 19:55 PST coordinator
- Old Hecate runner2 queue: the 5 pending tasks (w3fi FI-MegaMoE MXFP4 A/B and spgemm ab3) were superseded by the NVFP4 campaign and moved to k31vr/queue/superseded/ (not deleted), so the old Hecate holds (r2*, fe8) do not spend time on them. The queue/pending dir is now empty: when those holds start, runner2 idles; adopt them into the k31ws pool (rename) and make sure the k31ws tools can srun --overlap into them.
