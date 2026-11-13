# K3.1 Vera Rubin (VR200) pareto configs (srt-slurm)

srt-slurm renders used on Hecate (batch-spx, RoCE) for the K3.1 VR AgentX pareto and its gsm8k
accuracy gates, for code tag `k31vr-freeze-20261112b` (= af10ccc2c4 + MLA KV-capture fix f0e6a6a8ab).

- `<point>.yaml`: AgentX benchmark (DURATION 3600, synthetic acceptance as deployed).
- `gsm8k/<point>.yaml`: eval-only twin of the same deployment (real block rejection,
  FlashInfer autotune off) for the gsm8k accuracy gate.

| point | topology |
|---|---|
| c1, c8, c48-1p1d, c64 | 1P1D, 6 nodes: prefill TP8+DCP8, decode TP16+DCP16 |
| c160, c256 | 2P1D, 8 nodes: 2x prefill TP8+DCP8, decode TP16+DCP16 |
| c512 | 8 nodes: prefill DP4xTP4 (DCP4), decode DP4xTP4 SP |

The vLLM code is the Lustre tree `hti/p2/integ-af10ccc2c4` on PYTHONPATH (patched in place with the
fix; `mla_attention.py` md5 cd4070d5c87bfce0a885f01f2b28dd6a). The render names and the `vllm:`
identity string still say `integaf10frz` / af10ccc2c4 because the renders predate the fix; paths
are Hecate Lustre paths.
