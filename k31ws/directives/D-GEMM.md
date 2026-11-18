# D-GEMM directive (2026-11-18, from user question)

The user asked whether D-GEMM optimizes the MXFP8 projection GEMMs of the NVFP4 checkpoint. It does. Make that visible on the tracker:
- Report NVFP4-checkpoint numbers as the primary target. Your status shape/basis must say "NVFP4 checkpoint".
- Split current/achievable by precision in gap_analysis: MXFP8 (KDA in/o_proj, shared gate_up/down, incl. activation quant) vs BF16 (DSA/MLA projections, router, latent, f_b).
- Track the MXFP8 family's gap closed separately in the work log.
- The MXFP4-tree (all-BF16) numbers are secondary context only.
