# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's KDA cores on an FP32 conv state (``--mamba-cache-dtype
float32``: conv + recurrent state FP32), against the same stock references as the
BF16-conv tests, re-run here with every helper's ``CONV_DTYPE`` set to FP32:

* SP (TP4 = DCP4, 24 heads, RecoverSSM records): every record core (KdaSplit,
  KdaWide, KdaFlow) on the deferred serving path and the fold-then-verify chain at
  q2..q8, the record core vs the stock verify, the RecoverSSM chain, KdaFlow's fold;
* TP8 / TP16 (DFn / DA): the KDA heads, the fold, q1 decode, the TP8 verify, the
  wide / loop / feed verify kernels.

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
