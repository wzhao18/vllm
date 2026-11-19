
## 2026-11-18 17:25 PST coordinator
- INFRA did not merge LocalQuery (72fdf53350): tests/distributed/test_kimi_k3_sdec_sp_layers.py::test_sp_dsa_block_matches_stock_layer[mxfp4-merge] fails on all 4 ranks at R68 (_check_q_codes). Fix it (the crossover at 65 rows makes R68 take the local path), re-run the distributed test on 4 GPUs, re-mark landed-ready.
