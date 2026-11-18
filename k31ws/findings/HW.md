# HW findings (public-safe: features only, no internal numbers; details in the private hwdb)

- 2026-11-18 Mixed MXFP8-activation x MXFP4-weight block-scaled MMA runs at the FP8 tensor rate on VR200; only FP4 x FP4 (NVFP4 or MXFP4 W4A4) reaches the FP4 rate. The NVFP4 checkpoint's routed experts can use it; the in-tree W4A8 MegaMoE path cannot. Matters for compute-bound prefill MoE, not for weight-bound decode.
- 2026-11-18 The Rubin double-K block-scaled MMA forms are selected by instruction-descriptor bits and only exist in sm_107a builds; family/sm_100f builds run the Blackwell form at half the block-scaled rate. Library MXFP8 and NVFP4 GEMMs we measured must already use the native form (their measured rate exceeds the Blackwell-form ceiling).
- 2026-11-18 Model observation: decode expert rows assume die-local HBM bandwidth (a non-localized MoE kernel cannot reach model achievable); dense decode GEMM rows assume the default striped rate (die-localizing dense weights can beat model achievable).
- 2026-11-18 Model observation: the prefill indexer-score row is costed near the Blackwell-form FP8 rate; a native sm_107a kernel (FP4 x FP4 or mixed at the FP8 rate) can go below model achievable. Flagged to P-IDX.
- 2026-11-18 Counted TMA stores signal completion at the destination (byte counter), removing the producer system fence + flag; payload and counter must share one large page or the count is silently skipped.
- 2026-11-18 Cluster launch control works on sm_107a, but for small decode work static/one-shot persistent scheduling was faster in internal measurements.
