# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode: the per-rows tactic tables above 64 shard rows (the
decode step-up at Bmax 65, DDSA 2026-11-22) change only tiles, never precision or
summation order, and leave every row <= 64 on its previous tactic:

* router | latent down BF16 -> FP32 split-K (``sp_mlp.ROUTER_LATENT``): the new tactic
  at every row 65-96 is bitwise the previous table's (same split, same K order);
* the MXFP8 tables (shared expert, kda_in) are unchanged: their cold-chain wins (sm_107
  "pre" swap tiles) gave step time back in the step A/B (side-stream / in-step effects)
  and are re-tuned by in-step replay separately;
* the tables at rows <= 64 (R <= 256) are the previous ones;
* split-K shared memory: every table tactic fits the launch's dynamic shared memory;
  an over-limit tactic (64.80.4.5, which faulted with an illegal address) is refused by
  ``split_k_problem`` and by ``PreparedProjection.prepare`` instead of reaching a launch.

One GPU, about a minute."""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("needs a Rubin GPU (compute capability 10.7)", allow_module_level=True)

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.policy import (
    batch_table,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import (
    mxfp8,
    sp_policy,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.decode import (
    split_k_problem,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (
    PreparedProjection,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.splitk import (
    splitk_smem_bytes,
    splitk_smem_capacity,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import sp_mlp

DEVICE = torch.device("cuda", 0)
STEP_UP = range(65, 97)
# The tables before the change (int64.2 15c281dc32)
OLD_ROUTER_LATENT = "1-40:64.32.2.9,41-72:64.64.2.6,73-96:64.96.2.5"
OLD_MX_TUNED = "1-80:fi_cute,81-:scaled_mm"
OLD_KDA_IN_PRE = {
    (1, 128): ((128, 64), (1, 1), False, 128),
    (129, 256): ((256, 128), (2, 1), False, 256),
}


def _split_tactics():
    """Every 4-field split-K tactic named by the SP decode tables."""
    specs = [sp_mlp.ROUTER_LATENT]
    for name in ("GENERIC_SPLIT", "TUNED_SPLIT"):
        table = getattr(sp_policy, name, {})
        specs.extend(table.values() if name == "TUNED_SPLIT" else [table])
    out = set()
    for spec in specs:
        for entry in spec.values() if isinstance(spec, dict) else [spec]:
            for v in batch_table(entry).values():
                if isinstance(v, tuple) and len(v) == 4:
                    out.add(v)
    return sorted(out)


def test_split_k_tables_fit_shared_memory():
    tactics = _split_tactics()
    assert (64, 80, 2, 5) in tactics
    cap = splitk_smem_capacity()
    for t in tactics:
        assert splitk_smem_bytes(t) <= cap, t
        assert split_k_problem(t, 128 * t[2] * 8) is None, t  # a K every split divides


def test_over_limit_split_k_refused():
    bad = (64, 80, 4, 5)  # 245,896 B used vs 232,448 B launched: an illegal address
    assert splitk_smem_bytes(bad) > splitk_smem_capacity()
    assert "shared memory" in split_k_problem(bad, 128 * 4 * 8)
    w = torch.zeros(4480, 7168, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="shared memory"):
        PreparedProjection(w, {66: bad}, output_dtype=torch.float32).prepare()


def test_tables_unchanged_at_or_below_64():
    new, old = batch_table(sp_mlp.ROUTER_LATENT["*"]), batch_table(OLD_ROUTER_LATENT)
    assert all(new.get(r) == old.get(r) for r in range(1, 65))
    assert sp_mlp.prewait_rows(64) and not sp_mlp.prewait_rows(32)
    for nk in ((12288, 7168), (7168, 6144)):
        new = batch_table(mxfp8.MX_TUNED[nk]["*"])
        assert all(new.get(r) == batch_table(OLD_MX_TUNED).get(r) for r in range(1, 65))
        assert mxfp8.pre_tactic(*nk, 64) is None or new.get(64) != "pre"
    for (lo, hi), tactic in OLD_KDA_IN_PRE.items():
        for r in (lo, hi):
            assert mxfp8.pre_tactic(12448, 7168, r)[0] == tactic


def test_router_latent_step_up_rows_bitwise():
    torch.manual_seed(0)
    n, k = 4480, 7168
    w = (torch.randn(n, k, device=DEVICE) * 0.02).bfloat16()
    new, old = batch_table(sp_mlp.ROUTER_LATENT["*"]), batch_table(OLD_ROUTER_LATENT)
    rows = list(STEP_UP)
    a = PreparedProjection(
        w, {r: new[r] for r in rows}, output_dtype=torch.float32
    ).prepare()
    b = PreparedProjection(
        w, {r: old[r] for r in rows}, output_dtype=torch.float32
    ).prepare()
    for r in rows:
        x = torch.randn(r, k, device=DEVICE).bfloat16()
        ya = torch.empty(r, n, device=DEVICE, dtype=torch.float32)
        yb = torch.empty_like(ya)
        a.run(x, ya)
        b.run(x, yb)
        assert torch.equal(ya, yb), (r, new[r], old[r])


def test_mxfp8_tables_unchanged():
    for nk in ((12288, 7168), (7168, 6144)):
        assert mxfp8.MX_TUNED[nk]["*"] == OLD_MX_TUNED
        assert nk not in mxfp8.PRE_TACTICS
    assert mxfp8.pre_tactic(12448, 7168, 264)[0] == ((256, 192), (2, 1), True, 256)
