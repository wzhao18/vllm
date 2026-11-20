# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP decode sparse MLA on Rubin: the keys-on-M kernel (dsa.sparse_mla_km,
SP_KERNELS rows) against FlashInfer trtllm-gen, the stock decode's kernel, on the same
FP8 query (96 real of 128 mailbox heads), per-tensor FP8 KV rows, flat slots and
counts (~512 keys per row of a DCP4 shard, rows without keys included).

Both quantize P to E4M3 at 448 of the running row max over 128-key tiles, sum in FP32
and run each row in one unsplit pass at these row counts, so the latent agrees to the
BF16 rounding of the output (measured max 4.9e-4 absolute at a mean |latent| 1.6e-2,
runs/D-ATTN/sp) and the base-2 LSE to FP32 rounding; rows without keys: LSE -inf and
a zero latent on both."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the keys-on-M sparse MLA kernel needs SM100-family tcgen05 (Rubin)",
)

HEADS, REAL, ROW, WIDTH, LOCAL, CTX = 128, 96, 576, 2048, 512, 8192
BMM1, BMM2 = 0.07, 0.8


def _case(rows, gen):
    dev = "cuda"
    reqs = -(-rows // 4)
    kv = (torch.randn(reqs * CTX, ROW, device=dev, generator=gen) * 0.6).to(
        torch.float8_e4m3fn
    )
    query = (torch.randn(rows, HEADS, ROW, device=dev, generator=gen) * 0.5).to(
        torch.float8_e4m3fn
    )
    query[:, REAL:] = 0
    slots = torch.full((rows, WIDTH), -1, dtype=torch.int32, device=dev)
    valid = torch.zeros(rows, dtype=torch.int32, device=dev)
    for r in range(rows):
        n = 0 if r % 37 == 5 else LOCAL + (r * 7) % 81 - 40  # some rows without keys
        idx = torch.randperm(CTX, generator=gen, device=dev)[:n]
        slots[r, :n] = (idx + (r // 4) * CTX).int()
        valid[r] = n
    return query, kv, slots, valid


@pytest.mark.parametrize("rows", [68, 256, 384])
def test_keys_on_m_matches_trtllm_gen(rows):
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.sparse_mla_km import (  # noqa: E501
        sparse_mla_decode_km,
    )
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import _kv_counters

    gen = torch.Generator(device="cuda").manual_seed(rows)
    query, kv, slots, valid = _case(rows, gen)
    bf = torch.bfloat16
    lat_t = torch.full((rows, HEADS, 512), 7.0, dtype=bf, device="cuda")
    lse_t = torch.full((rows, HEADS), 7.0, dtype=torch.float32, device="cuda")
    lat_k, lse_k = lat_t.clone(), lse_t.clone()  # stale values: every row written
    extra = _kv_counters(rows, HEADS, query.device)
    workspace = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")
    trtllm_batch_decode_with_kv_cache_mla(
        query=query.unsqueeze(1), kv_cache=kv.view(-1, 64, ROW).unsqueeze(1),
        workspace_buffer=workspace, qk_nope_head_dim=128, kv_lora_rank=512,
        qk_rope_head_dim=64, block_tables=slots.unsqueeze(1), seq_lens=valid,
        max_seq_len=WIDTH, out=lat_t.unsqueeze(1), bmm1_scale=BMM1, bmm2_scale=BMM2,
        sparse_mla_top_k=WIDTH, return_lse=True, lse=lse_t, **extra,
    )  # fmt: skip
    sparse_mla_decode_km(
        query.view(torch.uint8), kv.view(torch.uint8), slots, valid, lat_k, lse_k,
        BMM1, BMM2,
    )  # fmt: skip
    torch.cuda.synchronize()

    keyed = valid > 0
    empty = ~keyed
    assert bool(empty.any()) and bool(keyed.any())
    got, want = lat_k[keyed, :REAL].float(), lat_t[keyed, :REAL].float()
    torch.testing.assert_close(got, want, atol=1e-3, rtol=1.6e-2)
    rel = (got - want).norm() / want.norm()
    assert float(rel) < 2e-3, float(rel)
    torch.testing.assert_close(lse_k[keyed, :REAL], lse_t[keyed, :REAL], atol=1e-4,
                               rtol=0)  # fmt: skip
    assert bool(torch.isneginf(lse_k[empty, :REAL]).all())
    assert bool((lat_k[empty, :REAL] == 0).all())
