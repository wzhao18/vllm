#!/usr/bin/env python3
"""check_gqa_window_k5.py: CPU check of gqa_window_k5.patch on an overlay that holds track K's fused GQA attention
(cutedsl/draft/gqa_attention.py + gqa_window.py, patched): every k5 drafter batch of the C128 / C160 1P1D llh decode
role (release_draft/k5 capture list, TP16: 1 KV head per rank) has a GQA_WINDOW launch shape, and every shape's
GqaWindow builds (its constructor asserts) at the draft's 1024-token blocks and the 4096-token ones. Also k3 / k7 as
controls. Run from the overlay root (dsrel/tools/gpurun.sh <tag> "" python3 check_gqa_window_k5.py)."""
from types import SimpleNamespace

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import gqa_attention as ga
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft.gqa_window import GqaWindow
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft

K5_SIZES = [
    10, 12, 24, 25, 30, 35, 36, 40, 48, 50, 60, 65, 72, 75, 80, 84, 90, 96, 102, 105, 108, 110, 114, 115, 120, 125,
    126, 130, 132, 135, 138, 140, 144, 145, 150, 155, 156, 160, 162, 168, 170, 174, 180, 186, 190, 192, 198, 200, 204,
    210, 216, 220, 228, 230, 234, 240, 250, 252, 260, 264, 270, 276, 280, 288, 290, 300, 306, 310, 312, 320, 324, 336,
    340, 342, 348, 360, 372, 378, 380, 384, 396, 400, 408, 414, 420, 432, 440, 450, 456, 460, 468, 480, 486, 500, 504,
    520, 522, 528, 540, 552, 558, 560, 576, 580, 600, 620, 624, 640, 648, 672, 696, 720, 744, 768, 792, 800, 816, 840,
    864, 888, 912, 936, 960,
]
fails = 0
for k, mns, cap, sizes in ((5, 160, 960, K5_SIZES), (3, 160, 640, [4 * b for b in range(1, 161)]),
                           (7, 64, 512, [8 * b for b in range(1, 65)])):
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_capture_sizes=sizes,
                                                             max_cudagraph_capture_size=cap),
                          scheduler_config=SimpleNamespace(max_num_seqs=mns))
    batches = draft.draft_batches(cfg, k)
    missing = [b for b in batches if ga.window_shape(1, k, b) is None]
    shapes = sorted({ga.window_shape(1, k, b) for b in batches} - {None})
    built = []
    for shape in shapes:
        splits, kgroups, ring, ahead = shape
        for page in (1024, 4096):
            try:
                GqaWindow(ga.GROUP, 1, k, page, splits, kgroups, ring, ga.WINDOW, None, ahead)
                built.append((shape, page))
            except AssertionError as e:
                print(f"k{k} shape {shape} page {page}: GqaWindow assert {e!r}")
                fails += 1
    fails += bool(missing)
    print(f"k{k}: {len(batches)} drafter batches B{batches[0]}..B{batches[-1]}; no GQA_WINDOW shape for "
          f"{missing or 'none'}; {len(built)} (shape, page) builds of {2 * len(shapes)} OK: {shapes}")
print("GQA_WINDOW_K5", "PASS" if not fails else f"FAIL ({fails})")
