# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's KDA cores on an FP32 conv state (``--mamba-cache-dtype
float32``: conv + recurrent state FP32), against the same stock references as the
BF16-conv tests, re-run here with every helper's ``CONV_DTYPE`` set to FP32:

* SP (TP4 = DCP4, 24 heads, RecoverSSM records): every record core (KdaSplit,
  KdaWide, KdaFlow) on the deferred serving path and the fold-then-verify chain at
  q2..q8, the record core vs the stock verify, the RecoverSSM chain, KdaFlow's fold;
* TP8 / TP16 (DFn / DA): the KDA heads (KdaHead / Verify / Multi), the plain verify
  kernels (LoopStock / Feed / KdaSplitStock) and the fold (KdaInProj64 -> prepared
  verify) against stock. They take an FP32 window by converting each tap to BF16 at
  the load and each stored value to FP32 (window.py); bitwise vs the BF16-conv run:
  test_specialized_decode_kda_conv_fp32_tp_gpu.

The conv window holds the BF16 in-projection inputs whatever its dtype, so every
output, checkpoint and conv window must match stock exactly as at BF16 conv (the
tests' own tolerances); a core that cannot take an FP32 conv state must raise at
build (never compute on a misread window: kimi-k3-cute-silent-arg-mismatch)."""

import pytest
import torch

from tests.models.kimi_k3 import test_specialized_decode_kda_fp32_gpu as tp_fp32
from tests.models.kimi_k3 import test_specialized_decode_kda_loop_gpu as tp_loop
from tests.models.kimi_k3 import test_specialized_decode_sp_kda_gpu as sp_kda
from tests.models.kimi_k3.test_specialized_decode_kda_fp32_gpu import (  # noqa: F401
    test_kda_head_matches_the_pair,
)
from tests.models.kimi_k3.test_specialized_decode_kda_loop_gpu import (  # noqa: F401
    test_feed_equals_loopstock,
    test_fp32_verify_rows_match_stock,
    test_loop_padding_requests,
    test_packed_and_loop_equal_wide,
)
from tests.models.kimi_k3.test_specialized_decode_kda_stock_gpu import (  # noqa: F401
    test_fp32_decode_q1_matches_stock,
    test_fp32_fold_matches_stock_and_pair,
    test_fp32_heads_match_stock,
    test_fp32_tp8_verify_matches_stock,
)
from tests.models.kimi_k3.test_specialized_decode_sp_kda_anyq_gpu import (  # noqa: F401
    test_deferred_serving_path_any_q,
    test_fold_then_verify_any_q,
    verify_length,
)
from tests.models.kimi_k3.test_specialized_decode_sp_kda_flow_gpu import (  # noqa: F401
    test_flow_fold_matches_ws_and_stock_commit,
)
from tests.models.kimi_k3.test_specialized_decode_sp_kda_gpu import (  # noqa: F401
    test_deferred_records_at_record_slots,
    test_record_core_matches_stock_verify,
)
from tests.models.kimi_k3.test_specialized_decode_sp_recoverssm_gpu import (  # noqa: F401
    test_recoverssm_chain_matches_stock,
)


@pytest.fixture(autouse=True)
def fp32_conv_state(monkeypatch):
    """Every helper's conv caches FP32 (and the layers' conv state dtype)."""
    for module in (sp_kda, tp_fp32, tp_loop):
        monkeypatch.setattr(module, "CONV_DTYPE", torch.float32)
    yield


def test_conv_state_is_fp32():
    conv, _ = sp_kda.caches(4)
    assert conv.dtype == torch.float32
    assert tp_loop.caches(4, 4)[0].dtype == torch.float32


def test_tp_kernels_check_the_conv_cache():
    """PreparedKda takes a BF16 or FP32 DS-layout window and refuses (at build,
    before any compile) another dtype or a window whose taps are not contiguous."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import prepared
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Unsupported

    heads, q, requests = 6, 4, 128
    attention = tp_loop.kda_attention(heads=heads, bias=False)
    conv, state = tp_loop.caches(requests * q + 1, q, heads=heads)
    assert conv.dtype == torch.float32

    def core():
        return prepared.PreparedKda(
            attention, {(requests, q): (tp_loop.pitch_of(heads), 0)}, True
        )

    with pytest.raises(Unsupported, match="dtype"):
        core().prepare_caches((conv.half(), state))
    strided = torch.empty_like(conv).transpose(1, 2).contiguous().transpose(1, 2)
    strided.copy_(conv)
    with pytest.raises(Unsupported, match="not contiguous"):
        core().prepare_caches((strided, state))
