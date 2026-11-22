# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP decode sparse MLA on Rubin: the keys-on-M kernel (dsa.sparse_mla_km)
and the latent-Q-in-TMEM kernel (dsa.sparse_mla_ts, SP_KERNELS rows) against FlashInfer
trtllm-gen, the stock decode's kernel, on the same
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


@pytest.mark.parametrize("kernel", ["km", "ts"])
@pytest.mark.parametrize("rows", [68, 256])
def test_keys_on_m_matches_trtllm_gen(rows, kernel):
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
    if kernel == "km":
        sparse_mla_decode_km(
            query.view(torch.uint8), kv.view(torch.uint8), slots, valid, lat_k, lse_k,
            BMM1, BMM2,
        )  # fmt: skip
    else:
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.sparse_mla_ts import (  # noqa: E501
            sparse_mla_decode_ts,
        )

        sparse_mla_decode_ts(
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


@pytest.mark.parametrize("rows", [256, 384])
def test_ts_last_token_staging_bitwise(rows):
    """rows > grid: a CTA's last token (one short subtile) follows a 2048-key token whose
    epilogue blocks 2-3 may still be in flight through the shared staging. Production
    grid (SM count) must match the race-free grid = rows launch (one row per CTA) bit
    for bit, latent and lse."""
    import cuda.bindings.driver as cuda
    from cutlass import Float32, Int32, Int64

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import sparse_mla_ts as ts

    gen = torch.Generator(device="cuda").manual_seed(rows)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    query, kv, slots, valid = _case(rows, gen)
    for r in range(rows):  # each CTA: earlier rows full width, the last row one subtile
        n = 17 if r + sms >= rows else WIDTH
        idx = torch.randperm(CTX, generator=gen, device="cuda")[:n]
        slots[r] = -1
        slots[r, :n] = (idx + (r // 4) * CTX).int()
        valid[r] = n

    def run(grid):
        lat = torch.zeros((rows, HEADS, 512), dtype=torch.bfloat16, device="cuda")
        lse = torch.zeros((rows, HEADS), dtype=torch.float32, device="cuda")
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        ts._compiled(grid)(
            Int64(query.data_ptr()), Int32(rows), Int64(kv.data_ptr()), Int32(64),
            Int64(slots.data_ptr()), Int32(slots.stride(0)), Int64(valid.data_ptr()),
            Int64(lat.data_ptr()), Int64(lse.data_ptr()), Float32(BMM1 * ts.LOG2E),
            Float32(BMM2), Int32(kv.shape[0]), Int64(0), Int64(0), Int64(0), Int32(1),
            Int32(1), Int32(6), stream,
        )  # fmt: skip
        return lat, lse

    ref = run(rows)
    for _ in range(20):
        got = run(sms)
        torch.cuda.synchronize()
        assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
