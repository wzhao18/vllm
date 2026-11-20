# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA: h8 launches for ncu (F4K L78K). usage: pmla_ncu.py [knobs a:b:c]"""
import sys

import torch
from pmla_h8g_ab import case

from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import (
    sparse_mla_prefill_h8 as k8)

kn = tuple(int(x) for x in sys.argv[1].split(":")) if len(sys.argv) > 1 else ()
args = case(4096, 78000, 0.15)
out = k8(*args, 0.07, 1.0, unit_rows=64, knobs=kn)
for _ in range(2):
    k8(*args, 0.07, 1.0, unit_rows=64, out=out, knobs=kn)
torch.cuda.synchronize()
