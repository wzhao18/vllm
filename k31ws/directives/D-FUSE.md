
## 2026-11-18 18:55 PST coordinator
- INFRA reports the NVFP4 slice graph-vs-eager rel rms jumped from 0.0 to 6.4e-2 after the D-FUSE merge (MXFP4 view 2e-4). Prove it is benign. Run the NVFP4 decode slice with and without D-FUSE on identical inputs (eager and graph), compare per-layer outputs, and show where the difference first appears. Rounding-order differences are fine only if the fused path matches its stock reference within the stated <=1 ulp at the layer boundary. Post the result in findings before further D-FUSE work.
