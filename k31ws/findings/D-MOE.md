# D-MOE findings
- 2026-11-18 12:40 FI a368c5f66 (EP28/32 build, superset of 6e438852b) staged as a WS_PYTHONPATH tree: rhea $K/runs/D-MOE/fi-a368c (tag python+csrc+include over the image FI data). P-MOE can use the same tree; one vendoring of a368c5f66 into vllm/models/kimi_k3 should serve decode (EP32) and prefill (EP16).
