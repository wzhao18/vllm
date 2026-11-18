# D-MOE (and INFRA): how to measure EP32 MegaMoE before the 8-node hold starts (2026-11-18 user question)

Decompose the per-GPU EP32 work into parts that one node reproduces exactly, and calibrate the rest once on 8 nodes.

1. **Expert compute and weight streaming: exact on 1 node.**
   - At EP32 each GPU holds 28 experts and receives about T x top-k / 32 x 32 = T_rank x top-k routes.
   - Run EP4 with the expert count set to 112 (28 per GPU) and the same tokens per rank. That gives the same expert weights per GPU, routes per expert and GEMM shapes.
   - Also run EP1 on one GPU with 28 local experts and the per-GPU route count. This isolates fc1/act/fc2 from dispatch and combine.
   - Blocker: specialized decode hard-codes the 896-expert MoE front (runtime/__init__.py:168 plus the MoE front kernels). Make it read the expert count from the model config inside specialized_decode, so production behaviour is unchanged at 896. INFRA owns the bench side; D-MOE owns the sdec change.
2. **Dispatch and combine: approximate on 1 node, then bounded.**
   - NVL72 is one flat NVLink switch domain, so per-GPU link bandwidth is the same intra-node and across nodes.
   - What differs at EP32 vs EP4:
     - bytes leaving each GPU: 31/32 of routes vs 3/4, about 1.3x more at EP32;
     - number of peers and flags: 31 vs 3, which means more, smaller messages and more completion signals;
     - lock-step skew across 32 ranks.
   - On 1 node, measure the EP4 dispatch/combine at EP32-equivalent bytes per GPU: pad the payload so 31/32 of the bytes are remote. Then add the peer-count latency term from a symmetric-memory microbench with 31 destinations. A single GPU can post to 3 real peers only, so model the extra fan-out with the measured per-message cost.
3. **Anchor and calibrate on real EP32.**
   - Existing Hecate rows: perf-model/microbench/results/moe_ep32 (FI EP32 standalone T sweep on 8 nodes) and the wave2 D3-32 step times.
   - The queued 8-node hold (*-k31ws.hold8): run the same kernel at EP32 once (standalone T4-T128 + in-step D3-32 profile). Then derive one correction factor (emulated/real) per T bucket and record it in your status.
   - After that, iterate on 1 node and re-check on 8 nodes only at milestones.
4. **Bring prepared scripts to the 8-node hold:** standalone EP32 sweep, in-step D3-32 profile, and the NVFP4 variant. These must be ready before it starts.

## 2026-11-18 13:50 PST coordinator: TOP PRIORITY
The NVFP4 decode baseline for the whole campaign is blocked on you: the NVFP4 MegaMoE must run at the decode buckets on the 1-node slice (EP4 emulating 28 experts/GPU) and at EP32. Get the vendored FI a368c5f66 path (or knob records) runnable for NVFP4 decode first, mark landed-ready, and tell INFRA (findings/D-MOE.md). Kernel optimization of MegaMoE comes after.
