#!/usr/bin/env python3
"""coverage_k5.py <k3 render config.yaml> <k5 render config.yaml>: CPU audit of what the C128 / C160 1P1D llh decode
role runs at each captured shape, k3 (frozen list, q4) vs k5 (release_draft/k5 list, q6), on the target topology
(Rubin VR200, TP16 = DCP16: NQ-NOFALLBACK, an uncovered shape refuses at build / capture).

Target (runtime/decode.py): every captured verify row's step plan. Rows <= 64 run the prepared providers, TUNED only at
q1 / q8 (decode.py ``choices``), else GENERIC, with the KDA verify kernel ``verify_kernel`` builds (FP32 recurrent state,
the deployments' mamba-ssm-cache-dtype float32) and whether a KdaHead / fold serves the row; rows above 64 run the wide
step (``wide_choices``: each family's ``choose``; STOCK = a seam, None = refused on the target).
Drafter (runtime/draft.py GqaDraftFastPath.build, the release GQA drafter): per drafter batch B, query rows B*k and
context rows B*(k+1): the GQA tables' Markov kind, context step, all-reduce site kind, LM head and split-K projection
bins. Kernel-level holds (PreparedContext / PreparedDraftAr rows, MarkovWide plans) are the coverage preflight's
(tests/models/kimi_k3/test_specialized_decode_coverage_preflight.py, deployment c160k5-prod).
Run with the vLLM tree first on PYTHONPATH and cutlass importable, e.g. on Hecate:
    bash $W/dsrel/tools/gpurun.sh <overlay tag> "" python3 coverage_k5.py <k3 config.yaml> <k5 config.yaml>
"""

import json
import re
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.attn_res.prepared import choose as attn_res_choose
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.policy import batch_table
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import markov
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.wide import choose as dsa_choose
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.prepared import choose as kda_choose
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.prepared import verify_kernel
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.prepared import choose as moe_choose
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj.prepared import choose as oproj_choose
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import DecodeConfig
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import decode as decode_module
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft, providers
from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Context, Path, Shape, captured_rows

SMS = 212  # VR200
LIMITS = (SMS, 334848, 334848 + 1024)  # markov.device_limits on VR200 (the preflight's)
VOCAB, RANK, TP = 163840, 512, 16


def decode_role(path):
    text = open(path).read()
    body = text[text.index("\n  decode:\n"):]
    arg = lambda name: re.search(rf"^      {name}: (.*)$", body, re.M).group(1)  # noqa: E731
    spec = json.loads(arg("speculative-config").strip("'"))
    sizes = json.loads(arg("compilation-config").strip("'"))["cudagraph_capture_sizes"]
    return spec["num_speculative_tokens"], int(arg("max-num-seqs")), int(arg("max-cudagraph-capture-size")), sizes


def runs(items):
    """[(key, value)] -> 'k1-k2: value; ...' over consecutive keys with equal values."""
    out, start, prev, val = [], None, None, None
    for key, v in items + [(None, object())]:
        if start is not None and (v != val or key is None):
            out.append(f"{start}-{prev}: {val}" if prev != start else f"{start}: {val}")
            start = None
        if key is None:
            break
        if start is None:
            start, val = key, v
        prev = key
    return "\n    ".join(out)


class Built(Exception):
    pass


def target(k, mns, cap, sizes):
    q = k + 1
    captured = captured_rows(sizes, q, min(mns * q, cap))
    group = SimpleNamespace(world_size=TP, rank_in_group=0, device_group=None)
    decode_module.get_tp_group = lambda: group
    decode_module.current_platform = SimpleNamespace(current_device=lambda: "cuda:0")
    device = SimpleNamespace(major=10, minor=7, multi_processor_count=SMS)
    torch.cuda.get_device_properties = lambda index: device
    real_context = decode_module.Context
    decode_module.Context = MagicMock(side_effect=Built)
    backbone = SimpleNamespace(config=SimpleNamespace(dsa_config=SimpleNamespace()), use_sequence_parallel=False)
    config = DecodeConfig(max_context=1 << 20, mla_core="auto", dsa_core="auto", dcp=16, q=q,
                          capture_sizes=tuple(sizes), captured=captured)
    decode = object.__new__(decode_module.SpecializedDecode)
    try:
        decode_module.SpecializedDecode.__init__(decode, backbone, config)
    except Built:
        pass
    decode_module.Context = real_context
    ctx = Context(arch="rubin", tp=TP, rank=0, dcp=16, sms=SMS, q=q, max_rows=max(captured),
                  max_context=1 << 20, bf16_state=False)
    stock_whole = [m for m in captured if m not in decode.batches or m % q]
    head_multi = batch_table(providers.HEAD_MULTI_FP32.get(q, ""))
    fold = batch_table(providers.KDA_FOLD64_FP32.get(TP, ""))
    rows = []
    for m in captured:
        b = m // q
        if m <= providers.PREPARED_MAX:
            path = "TUNED" if q in (1, 8) and m in providers.BATCHES else "GENERIC"
            head = "KdaHeadMulti" if head_multi.get(m) else ("fold64" if fold.get(m) and m == 8 * q else "")
            kda = head or verify_kernel(b, q, (10, 7), False, 0)
            rows.append((m, f"prepared {path} (kda {kda})"))
            continue
        fam = []
        for name, fn in (("kda", kda_choose), ("dsa", dsa_choose), ("oproj", oproj_choose), ("moe", moe_choose),
                         ("attn_res", attn_res_choose)):
            c = fn(Shape(m, q), ctx)
            # check_wide: a KDA / DSA / o-projection / MoE seam (STOCK, or no choice) refuses on the target; the
            # AttnRes stock op is a measured in-step choice
            if c is None or (c.path is Path.STOCK and name != "attn_res"):
                label = "REFUSE"
            else:
                label = c.path.name
            fam.append(f"{name}={label}" + (f"[{c.note.split(' ')[0]}]" if c is not None and name == "kda" else ""))
        embed = "ring" if m <= 256 else "stock-embed"
        rows.append((m, f"wide {' '.join(fam)} embed={embed}"))
    print(f"\n== target q{q} (k{k}): {len(captured)} captured verify rows {captured[0]}..{captured[-1]}, "
          f"capacity {decode.capacity}, wide rows {len(decode.wide)}, wide gate {decode.wide_off or 'open'}")
    print(f"  whole-step stock forward (refused on the target): {stock_whole or 'none'}")
    print("  per captured row M (B = M / q):\n    " + runs(rows))
    return captured


def drafter(k, mns, cap, sizes):
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_capture_sizes=sizes, max_cudagraph_capture_size=cap),
                          scheduler_config=SimpleNamespace(max_num_seqs=mns))
    batches = draft.draft_batches(cfg, k)
    top = max(batches)
    mk = batch_table(draft.closed(draft.GQA_FUSED_MARKOV[TP], top))
    prepared = object.__new__(markov.PreparedMarkov)
    prepared.tp, prepared.markov_rank, prepared.limits = TP, RANK, LIMITS
    prepared.vocab_rank, prepared.plans = VOCAB // TP, {}
    prepared.wide_box = object() if top > markov.MAX_WIDTH else None
    ctx_spec = draft.closed(draft.GQA_FAST_CONTEXT[TP], top * (k + 1))
    ar_spec = draft.closed(draft.GQA_FAST_AR[TP], top * k)
    lm = draft.GQA_FAST_LM_HEAD.get(TP, "")
    wide_lm = draft.GQA_WIDE_LM_HEAD.get(TP, "")
    shapes = draft.gqa_shapes(TP)
    rows, gaps = [], []
    for b in batches:
        r, c = b * k, b * (k + 1)
        kind = {1: "plain" if b <= markov.MAX_WIDTH else "MarkovWide", 2: "gated", 3: "gated+staged"}.get(mk.get(b), "NONE")
        if not prepared.covers(b, True):
            kind += "(NO PLAN)"
            gaps.append((b, "Markov plan"))
        ctx = draft.covered(ctx_spec, c, k)
        site = {1: "one-shot", 2: "two-shot", 3: "ArNormRows"}.get(draft.covered(ar_spec, r, k), "NONE")
        if site == "NONE":
            gaps.append((b, "sites"))
        if not ctx:
            gaps.append((b, "context step"))
        head = ("GEMV" if draft.covered(lm, r, k) else "evict-first" if draft.covered(wide_lm, r, k) else "cuBLAS")
        tactic = lambda shape, n: "splitK" if draft.gqa_projection_tactic(shapes[shape], TP, n) else "cuBLAS"  # noqa: E731
        proj = "/".join(tactic(s, r) for s in ("qkv", "o_proj", "gate_up", "down"))
        cproj = "/".join(tactic(s, c) for s in ("ctx_shard", "ctx_kv"))
        rows.append((b, f"markov {kind}; sites {site}; lm_head {head}; qkv/o/gate_up/down {proj}; "
                        f"context {'sharded' if ctx else 'NONE'} fc/kv {cproj}"))
    print(f"\n== release drafter k{k}: {len(batches)} drafter batches B{batches[0]}..B{top} (query rows B*{k} "
          f"{batches[0] * k}..{top * k}, context rows B*{k + 1} {batches[0] * (k + 1)}..{top * (k + 1)})")
    print(f"  strict-table gaps (refused on the target): {gaps or 'none'}")
    print("  attention: the fused GQA window kernel (track K's cutedsl.draft.gqa_attention installer) at query rows "
          f"{batches[0] * k}..{top * k} / context rows {batches[0] * (k + 1)}..{top * (k + 1)}: not in this tree")
    print("  per drafter batch B:\n    " + runs(rows))
    return batches


def main():
    for path in sys.argv[1:]:
        k, mns, cap, sizes = decode_role(path)
        print(f"\n######## {path}: k{k}, max-num-seqs {mns}, cap {cap}, {len(sizes)} capture sizes")
        target(k, mns, cap, sizes)
        drafter(k, mns, cap, sizes)


if __name__ == "__main__":
    main()
