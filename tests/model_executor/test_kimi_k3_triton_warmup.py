# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 prefill glue warmup runs against the layer's real conv cache in
both VLLM_SSM_CONV_STATE_LAYOUT layouts (the default SD cache used to be split
along state_len and tripped causal_conv1d_fn's shape assert)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba import mamba_utils
from vllm.model_executor.warmup import kimi_k3_triton_warmup as kw
from vllm.platforms import current_platform

# Per-rank KDA shapes (head_dim 128, kernel 4), a few cache lines.
H, D, K, LINES = 4, 128, 4, 3
P = H * D


def _conv_cache(layout: str, device) -> torch.Tensor:
    shape = (LINES, 3 * P, K - 1) if layout == "DS" else (LINES, K - 1, 3 * P)
    return torch.zeros(shape, dtype=torch.bfloat16, device=device)


@pytest.mark.parametrize("layout", ["SD", "DS"])
def test_conv_state_dim_view(monkeypatch, layout):
    monkeypatch.setattr(mamba_utils, "get_conv_state_layout", lambda: layout)
    cs = _conv_cache(layout, "cpu")
    view = kw._conv_state_dim_view(cs)
    assert view.shape == (LINES, 3 * P, K - 1)
    assert view.data_ptr() == cs.data_ptr()  # a view of the cache, no copy
    for part in view.split(P, dim=-2):
        assert part.shape == (LINES, P, K - 1)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA is required")
@pytest.mark.parametrize("layout", ["SD", "DS"])
def test_warm_prefill_glue_runs(monkeypatch, layout):
    monkeypatch.setattr(mamba_utils, "get_conv_state_layout", lambda: layout)
    device = torch.device("cuda")
    conv_state = _conv_cache(layout, device)
    rec_state = torch.zeros((LINES, H, D, D), dtype=torch.float32, device=device)
    conv_state.fill_(1.0)
    layer = SimpleNamespace(
        kv_cache=[conv_state, rec_state],
        local_projection_size=P,
        local_num_heads=H,
        head_dim=D,
        # KDA pads the projection rows to 16 elements (kda.py in_proj_padding).
        in_proj_padding=-(4 * P + D + H) % 16,
        conv1d=SimpleNamespace(
            weight=torch.zeros((3 * P, 1, K), dtype=torch.bfloat16, device=device)
        ),
        o_norm=SimpleNamespace(
            weight=torch.ones(D, dtype=torch.bfloat16, device=device), eps=1e-5
        ),
        _checkpoint_exporter=None,
        _segscan_bufs=None,
    )
    worker = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))
    kw._warm_prefill_glue(worker, layer)  # type: ignore[arg-type]
    # Every call targets the null cache line: the cache is left untouched.
    assert bool((conv_state == 1.0).all())
    assert not rec_state.any()
