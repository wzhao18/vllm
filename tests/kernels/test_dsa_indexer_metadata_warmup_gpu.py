# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSA indexer metadata warmup's row counts against the installed
DeepGEMM: probe, on this GPU, which row counts compile a new
``get_paged_mqa_logits_metadata`` variant (a new entry in a fresh DeepGEMM
cache dir), then check ``metadata_row_counts`` reaches every variant. Not the
SM100 source comment: the binary's own classes on this architecture."""

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from vllm.model_executor.warmup.dsa_indexer_metadata_warmup import (
    metadata_row_counts,
)

MAX_ROWS = 4608  # past the largest SM100 class edge (2048) with margin

_PROBE = textwrap.dedent(
    """
    import json, os, sys
    import torch
    from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata
    from vllm.utils.platform_utils import num_compute_units

    cache = os.path.join(os.environ["DG_JIT_CACHE_DIR"], "cache")
    def variants():
        if not os.path.isdir(cache):
            return 0
        return sum("paged_mqa_logits_metadata" in n for n in os.listdir(cache))
    max_rows, next_n = int(sys.argv[1]), int(sys.argv[2])
    sms = num_compute_units(0)
    edges, before = [], variants()
    for rows in range(next_n, max_rows + 1, next_n):
        lens = torch.full((rows // next_n, next_n), 64, dtype=torch.int32,
                          device="cuda")
        get_paged_mqa_logits_metadata(lens, 64, sms)
        torch.cuda.synchronize()
        now = variants()
        if now > before:
            edges.append(rows)
            before = now
    print(json.dumps(edges))
    """
)


def _variant_edges(tmp_path, next_n: int) -> list[int]:
    """Row counts at which the installed DeepGEMM compiled a new metadata
    variant (the first is the first call), probed in a fresh process and
    cache dir."""
    env = dict(os.environ, DG_JIT_CACHE_DIR=str(tmp_path / f"dg{next_n}"))
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, str(MAX_ROWS), str(next_n)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
# next_n: the served verify lengths q = k + 1 (k1 / k3 / k5 / k7) and q1.
@pytest.mark.parametrize("next_n", [1, 2, 4, 6, 8])
def test_row_counts_reach_every_installed_metadata_variant(tmp_path, next_n):
    from vllm.utils.import_utils import has_deep_gemm

    if not has_deep_gemm():
        pytest.skip("DeepGEMM not installed")
    edges = _variant_edges(tmp_path, next_n)
    assert edges, "the first metadata call compiled no variant"
    print(f"sm_{torch.cuda.get_device_capability()} next_n={next_n} edges {edges}")

    def variant(rows: int) -> int:
        return sum(rows >= e for e in edges)

    for max_rows in (512, 640, 1024, 2048, 2560, MAX_ROWS):
        reachable = {variant(r) for r in range(next_n, max_rows + 1, next_n)}
        # As the warmup builds them: batch = rows // next_n, next_n rows each.
        built = {
            variant(max(1, r // next_n) * next_n) for r in metadata_row_counts(max_rows)
        }
        assert built >= reachable, (max_rows, edges, sorted(built), sorted(reachable))
