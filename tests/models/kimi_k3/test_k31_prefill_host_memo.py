# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step host memo of the persistent KDA prefill (CPU only)."""

from vllm.models.kimi_k3.nvidia.specialized_prefill import kda


class _Meta:
    pass


def test_step_memo_is_per_metadata_object():
    kda._step_memos.clear()
    a, b = _Meta(), _Meta()
    ma = kda.step_memo_for(a)
    ma["x"] = 1
    assert kda.step_memo_for(a) is ma
    mb = kda.step_memo_for(b)
    assert mb is not ma and "x" not in mb


def test_step_memo_never_reuses_a_dead_object_entry():
    kda._step_memos.clear()
    a = _Meta()
    kda.step_memo_for(a)["plan"] = "old"
    key = id(a)
    del a
    # A new object that happens to get the same id must start empty.
    b = _Meta()
    if id(b) == key:
        assert "plan" not in kda.step_memo_for(b)
    else:
        assert kda.step_memo_for(b) == {}


def test_step_memo_bounded():
    kda._step_memos.clear()
    keep = [_Meta() for _ in range(40)]
    for m in keep:
        kda.step_memo_for(m)
    assert len(kda._step_memos) <= 16
