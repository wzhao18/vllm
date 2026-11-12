# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLAAttention.update_kv_cache under CUDA-graph capture.

update_kv_cache may leave a DCP ShardQ step's MLA rows to forward_impl's fused
shardq_kv_insert, but only when it runs eagerly. unified_mla_kv_cache_update is
captured into the (breakable) piecewise graphs, and the V2 model runner builds
real attention metadata for every capture batch (cudagraph_utils.
prepare_inputs_to_capture), so a capture whose dummy batch plans a ShardQ step
would freeze "skip the write" into the graph and replay it for every later step
of that size, including steps without a ShardQ context, whose rows then nothing
writes (K3.1 C256: decode-classified first chunks attended all-zero KV).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import MLAAttention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"
)


class _Impl:
    def __init__(self):
        self.calls = 0

    def do_kv_cache_update(self, kv_c, k_pe, kv_cache, slot_mapping, *args):
        self.calls += 1
        # Capturable stand-in for concat_and_cache_mla (all slots valid here).
        rows = kv_cache.view(-1, kv_cache.shape[-1])
        new = torch.cat([kv_c, k_pe.reshape(k_pe.shape[0], -1)], -1)
        rows.index_copy_(0, slot_mapping, new)


def _layer():
    layer = MLAAttention.__new__(MLAAttention)
    layer.hisparse_cache = None
    layer.use_pcp = False
    layer.pcp_shard_decode_requests = False
    layer._k_scale_float = 1.0
    layer._shardq_local_slots = None
    layer.impl = _Impl()
    # A ShardQ step with one request group: forward_impl would write the rows.
    layer._shardq_fused_kv_insert = lambda *a, **k: SimpleNamespace(num_groups=1)
    return layer


def _inputs(n=64, nope=512, rope=64):
    kv_c = torch.randn(n, nope, device="cuda")
    k_pe = torch.randn(n, 1, rope, device="cuda")
    kv_cache = torch.zeros(8, 16, nope + rope, device="cuda")
    slots = torch.arange(n, device="cuda", dtype=torch.int64)
    return kv_c, k_pe, kv_cache, slots


@torch.inference_mode()
def test_eager_shardq_step_defers_to_fused_insert():
    layer = _layer()
    kv_c, k_pe, kv_cache, slots = _inputs()
    meta = SimpleNamespace(num_decode_tokens=0)
    layer.update_kv_cache(kv_c, k_pe, kv_cache, slots, meta, "auto", torch.ones(()))
    assert layer.impl.calls == 0 and layer._shardq_local_slots is slots


@torch.inference_mode()
def test_captured_update_always_writes():
    layer = _layer()
    kv_c, k_pe, kv_cache, slots = _inputs()
    meta = SimpleNamespace(num_decode_tokens=0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            scale = torch.ones((), device="cuda")
            layer.update_kv_cache(kv_c, k_pe, kv_cache, slots, meta, "auto", scale)
    # The captured graph must contain the write: replay it for a step whose rows
    # nothing else writes and check the cache holds them.
    assert layer.impl.calls == 1, (
        "the capture froze the ShardQ early return into the graph"
    )
    kv_cache.zero_()
    graph.replay()
    torch.cuda.synchronize()
    want = torch.cat([kv_c, k_pe.reshape(k_pe.shape[0], -1)], -1)
    got = kv_cache.view(-1, kv_cache.shape[-1])[: kv_c.shape[0]]
    torch.testing.assert_close(got, want)
