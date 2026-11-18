# D-KDA findings (decode KDA core, FP32 state)

- 2026-11-18 Production coverage (rhea1n spgemm-prof agentx.log): records mode "deferred"; KdaWide (folds the pending commit itself) at n <= 70 requests per DP group, KdaWS above (does NOT fold: the step's first KDA core flushes every layer's pending commit with a separate state read + write pass, then KdaWS reads the state again).
- 2026-11-18 Model target per layer (rhea1n/model_rows.json, kda_core = conv + gate_beta_norm + core_decode + o_norm): B32 8.76 us, B64 17.5 us, B96 26.3 us; memory bound at the measured full-chip HBM rate on 64 KB state read + 64 KB write per (request, head) item. In-step measured (rhea1n_d3x32 93L): 21.5 / 38.5 / 49.2 us per layer.
- 2026-11-18 Model note (not an error yet): FP32 recurrence + fold is ~0.8 MFLOP per item, ~55-60% of the memory time at the model's HBM rate; the model's max(compute, memory) assumes perfect overlap of the FP32 pipe with the state stream.
