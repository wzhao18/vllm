# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest

from vllm.benchmarks.forward.shapes import BenchRequest, load_cases


def _write(tmp_path, data):
    p = tmp_path / "shapes.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_per_rank_case_selects_each_rank(tmp_path):
    path = _write(tmp_path, {"cases": [
        {"name": "s1", "ranks": [
            [{"query_len": 16384, "kv_len": 116384}],
            [{"query_len": 4, "kv_len": 4}],
            [{"query_len": 2000, "kv_len": 52000}, {"query_len": 100, "kv_len": 900}],
        ]},
        {"name": "plain", "requests": [{"query_len": 8, "kv_len": 8, "count": 2}]},
    ]})
    s1, plain = load_cases(path)
    assert s1.requests == (BenchRequest(16384, 116384),)  # most KV, for sizing
    assert len(s1.all_requests) == 3
    assert s1.for_rank(1, 3).requests == (BenchRequest(4, 4),)
    r2 = s1.for_rank(2, 3)
    assert r2.requests == (BenchRequest(2000, 52000), BenchRequest(100, 900))
    assert r2.per_rank == () and r2.num_tokens == 2100
    assert plain.for_rank(2, 3) is plain and len(plain.requests) == 2
    with pytest.raises(ValueError):
        s1.for_rank(0, 4)


def test_per_rank_case_rejects_empty_rank(tmp_path):
    path = _write(tmp_path, {"cases": [{"name": "x", "ranks": [[], [{"query_len": 4, "kv_len": 4}]]}]})
    with pytest.raises(ValueError):
        load_cases(path)
