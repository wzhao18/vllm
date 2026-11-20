# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DIAG: print the B / SFB smem stage layout of the swap_ab kda_in "pre" tactic
(check_x layout question; not for merge)."""

import torch

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sm107 import prewait

torch.zeros(1, device="cuda")
for tactic in (((256, 192), (2, 1), True, 256), ((256, 128), (2, 1), True, 256)):
    try:
        prewait._compile(tactic, True, 4, 0, check=True)
        print(tactic, "compiled")
    except Exception as e:  # noqa: BLE001
        print(tactic, type(e).__name__, str(e)[:900])
torch.cuda.synchronize()
