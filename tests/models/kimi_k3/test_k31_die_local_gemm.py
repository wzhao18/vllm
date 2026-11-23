# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The die-local BF16 decode GEMM (``specialized_decode/cutedsl/gemm/die_local``).

Host side (CPU): the cluster kernel's ticket -> tile map hands every tile out exactly
once for any split of clusters between the dies and any arrival order; the halves
placement keeps each tile's majority on its die; the cluster size choice.

GPU (a two-die part, VR200): every decode shape and output mode against an FP32
reference (error within the stock BF16 GEMM's), the relocated weight's values
unchanged, and bitwise repeatability over CUDA-graph replays.
"""

import random

import pytest
import torch

from vllm.platforms import current_platform

pytest.importorskip("cutlass")

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.die_local import (  # noqa: E402
    OUT_BF16,
    OUT_BF16_ADD,
    OUT_F32,
    DieLocalEngine,
    cluster,
    placement,
    relocate_halves,
    two_dies,
)


def _take_tile(ticket, home, tiles, split):
    """``DieLocalClusterGemm.take_tile`` after its ticket atomic: the ``ticket``-th
    cluster landing on die ``home`` takes that die's next tile, then the other die's
    tiles from the top."""
    cap0, cap1 = split, tiles - split
    home_cap = cap0 + home * (cap1 - cap0)
    other_cap = cap1 + home * (cap0 - cap1)
    tile = home * cap0 + ticket
    if ticket >= home_cap:
        tile = (1 - home) * cap0 + other_cap - 1 - (ticket - home_cap)
    return tile


def test_ticket_map_is_a_permutation():
    rng = random.Random(0)
    for tiles in range(1, 40):
        for split in range(tiles + 1):
            for on_die0 in range(tiles + 1):
                homes = [0] * on_die0 + [1] * (tiles - on_die0)
                for _ in range(3):
                    rng.shuffle(homes)
                    counts = [0, 0]
                    got = []
                    for home in homes:
                        got.append(_take_tile(counts[home], home, tiles, split))
                        counts[home] += 1
                    assert sorted(got) == list(range(tiles))
                    # clusters within their die's tile count take their die's tiles
                    for home, tile in zip(homes, got):
                        if (home == 0 and on_die0 <= split) or (
                            home == 1 and tiles - on_die0 <= tiles - split
                        ):
                            assert (tile < split) == (home == 0)


@pytest.mark.parametrize(
    "n,k",
    [
        (12448, 7168),  # KDA in-projection
        (5344, 7168),
        (7168, 3072),  # o_proj
        (4608, 1536),
        (3584, 7168),
        (7168, 3584),
        (896, 7168),
        (12288, 7168),  # shared expert gate_up
        (7168, 6144),  # shared expert down
    ],
)
def test_halves_placement_keeps_each_tile_on_its_die(n, k):
    chunk = placement.CHUNK
    domains = placement.halves_domains(n, k)
    split = placement.tile_split(n, k, 128)
    tile_bytes = 128 * k * 2
    tiles = -(-n // 128)
    for t in range(tiles):
        lo, hi = t * tile_bytes, min((t + 1) * tile_bytes, n * k * 2)
        on = [0, 0]
        for c in range(lo // chunk, (hi - 1) // chunk + 1):
            on[domains[c]] += min(hi, (c + 1) * chunk) - max(lo, c * chunk)
        die = 0 if t < split else 1
        assert on[die] >= on[1 - die], (t, on, split)
    assert abs(split - (tiles - split)) <= chunk // tile_bytes + 2


def test_cluster_split():
    cluster_split = cluster.cluster_split
    assert cluster_split(49, 106, 112) == 2
    # 28 clusters of 7 do not co-schedule on 212 SMs
    assert cluster_split(14, 106, 112, tiles=28) == 6
    assert cluster_split(28, 106, 48) == 3
    assert cluster_split(14, 106, 112, tiles=24) == 7
    assert cluster_split(4, 106, 3) == 3  # never more ranks than K blocks
    assert cluster_split(200, 106, 112) == 1


# name: (N, K, max rows, output mode, rows checked)
GPU_CASES = {
    "kda_in": (12448, 7168, 192, OUT_BF16, [1, 6, 17, 48, 64, 96, 143, 144, 150, 192]),
    "o_proj": (7168, 3072, 192, OUT_BF16, [1, 8, 31, 96, 144, 192]),
    "latent_down": (3584, 7168, 48, OUT_BF16, list(range(1, 49))),
    "latent_up": (7168, 3584, 48, OUT_BF16_ADD, [1, 2, 7, 16, 33, 36, 48]),
    "router_gate": (896, 7168, 48, OUT_F32, [1, 5, 36, 48]),
    "q_b": (4608, 1536, 192, OUT_BF16, [3, 144, 192]),
    "shared_gate_up": (12288, 7168, 48, OUT_BF16, [1, 12, 36, 48]),
    "shared_down": (7168, 6144, 48, OUT_BF16, [1, 12, 36, 48]),
}


@pytest.mark.skipif(
    not current_platform.is_cuda() or not two_dies(torch.cuda.current_device()),
    reason="needs a GPU with two locality domains (VR200)",
)
@pytest.mark.parametrize("case", list(GPU_CASES))
@torch.inference_mode()
def test_die_local_gemm_matches_reference(case):
    n, k, max_rows, mode, rows_checked = GPU_CASES[case]
    device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=device).manual_seed(0)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16, generator=gen) * 0.02
    placed = relocate_halves(w)
    assert torch.equal(placed, w), "relocation changed values"
    engine = DieLocalEngine(placed, max_rows, mode)
    out_dtype = torch.float32 if mode == OUT_F32 else torch.bfloat16
    for r in rows_checked:
        x = torch.randn(r, k, device=device, dtype=torch.bfloat16, generator=gen)
        init = None
        if mode == OUT_BF16_ADD:
            init = torch.randn(r, n, device=device, dtype=torch.bfloat16, generator=gen)
        ref = x.float() @ w.float().t()
        if init is None:
            stock = (x @ w.t()).float()
            out = torch.full((r, n), float("nan"), device=device, dtype=out_dtype)
        else:
            ref += init.float()
            stock = init.clone().addmm_(x, w.t()).float()
            out = init.clone()
        assert engine.serves(x, out)
        engine(x, out)
        torch.accelerator.synchronize()
        scale = ref.abs().max().item()
        err = (out.float() - ref).abs().max().item() / scale
        err_stock = (stock - ref).abs().max().item() / scale
        assert torch.isfinite(out.float()).all(), f"{case} R{r}: non-finite"
        assert err <= max(2 * err_stock, 1e-5), f"{case} R{r}: {err} vs {err_stock}"

        outs = [
            init.clone() if init is not None else torch.empty_like(out)
            for _ in range(20)
        ]
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), torch.cuda.graph(graph):
            for o in outs:
                engine(x, o)
        graph.replay()
        torch.accelerator.synchronize()
        assert all(torch.equal(o, out) for o in outs), f"{case} R{r}: not bitwise"
