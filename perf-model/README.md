# perf-model: Kimi K3.1 on VR200, speed-of-light sharding model

First-principles speed-of-light (SOL) model of every Kimi K3.1-Pro op (DSpark draft included) under each prefill and decode sharding
strategy (TP+DCP with ShardQ, hybrid DP x (TP+DCP), DEP, TP with DSA-DP, and new per-layer-type mappings), including the communication each
topology adds. Prefill and decode are modeled separately: prefill by fresh tokens/s per GPU, decode by step time vs requests per GPU within
KV capacity.

Work in progress (2026-11-17): inputs (op inventory, topology/communication mechanics, VR200 constants, workload/KV constants) are being
collected; the model code, outputs and the breakdown page (k31vr/sol.html) follow. Tracker: k31vr/index.html (workstream W10).
