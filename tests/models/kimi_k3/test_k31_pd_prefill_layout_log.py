# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The P/D prefill worker's startup INFO line (model.pd_prefill_moe_layout_line): it
names the MoE layout (SP+EP), the replicated shared experts' weight format (MXFP8 /
BF16) and where the KV token capacity is reported; only on P/D prefill workers in the
SP+EP layout; KimiMoE logs it once. Logging only (CPU)."""

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.kimi_k3.nvidia import model


def _config(kv_role, tp=8, dp=1, backend="kimi_k3_mega_moe"):
    kv = None
    if kv_role is not None:
        kv = SimpleNamespace(
            is_kv_producer=kv_role in ("kv_producer", "kv_both"),
            is_kv_consumer=kv_role in ("kv_consumer", "kv_both"),
        )
    return SimpleNamespace(
        kv_transfer_config=kv,
        additional_config={},
        kernel_config=SimpleNamespace(moe_backend=backend),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            pipeline_parallel_size=1,
            data_parallel_size=dp,
            tensor_parallel_size=tp,
        ),
    )


def _shared(method):
    return SimpleNamespace(gate_up_proj=SimpleNamespace(quant_method=method))


@pytest.mark.parametrize("kv_role", ["kv_both", "kv_producer"])
@pytest.mark.parametrize("mxfp8,dtype", [(True, "MXFP8"), (False, "BF16")])
def test_prefill_worker_logs_layout_dtype_and_kv(monkeypatch, kv_role, mxfp8, dtype):
    monkeypatch.setattr(model, "is_mxfp8_linear", lambda linear: mxfp8)
    line = model.pd_prefill_moe_layout_line(
        _config(kv_role), _shared(UnquantizedLinearMethod())
    )
    assert line is not None
    assert "SP+EP" in line and "TP8" in line and "EP8" in line
    assert f"shared-expert weights {dtype};" in line
    assert "KV token capacity" in line and "GPU KV cache size" in line


def test_other_quant_method_is_named(monkeypatch):
    monkeypatch.setattr(model, "is_mxfp8_linear", lambda linear: False)

    class SomeFp8Method:
        pass

    line = model.pd_prefill_moe_layout_line(
        _config("kv_both"), _shared(SomeFp8Method())
    )
    assert "shared-expert weights SomeFp8Method;" in line


@pytest.mark.parametrize(
    "cfg",
    [
        _config("kv_consumer"),  # decode
        _config(None),  # unified serving
        _config("kv_both", tp=1),  # no SP (TP1)
    ],
)
def test_no_line_off_the_pd_prefill_sp_layout(cfg, monkeypatch):
    monkeypatch.setattr(model, "is_mxfp8_linear", lambda linear: True)
    assert model.pd_prefill_moe_layout_line(cfg, _shared(None)) is None


def test_kimi_moe_logs_it_once():
    """KimiMoE.__init__ passes the line to logger.info_once (one line per process)."""
    import inspect

    src = inspect.getsource(model.KimiMoE.__init__)
    assert "pd_prefill_moe_layout_line(vllm_config, self.shared_experts)" in src
    assert "logger.info_once(layout)" in src
