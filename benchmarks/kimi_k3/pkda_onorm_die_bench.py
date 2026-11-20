#!/usr/bin/env python3
"""KDA prefill o_norm + MXFP8 (TMA kernel) with die-local rows vs striped.

x (the KDA core output, (T, H, 128) BF16) and the gate rows g (columns of the
(T, 12512) g|f_a|beta projection) are homed per 2048-token super-block on die
sb & 1 (specialized_prefill.attn_res_dielocal granule) or left in a default
(4 KiB die-interleaved) allocation. Arms, same data, same GPU, interleaved:
  A  striped buffers, static row ranges (production)
  B  die-homed buffers, own-die claims + steal (o_norm_mxfp8_dl)
  C  die-homed buffers, static row ranges
  D  striped buffers, own-die claims (claim overhead only)
B, C, D are checked bit for bit against A (MXFP8 data and scales).
"""

import argparse
import json
import statistics

import torch
from pgemm_conv_bench import kernel_times


def homed(shape_rows, row_bytes, device):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    nbytes = shape_rows * row_bytes
    chunk = 2 << 20
    super_bytes = 2048 * row_bytes
    domains = [((c * chunk) // super_bytes) & 1 for c in range((nbytes + chunk - 1) // chunk)]
    return dies.localized_empty(nbytes, domains, device)[:nbytes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="16383,8193")
    ap.add_argument("--heads", type=int, default=96)
    ap.add_argument("--rest", type=int, default=12512)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    from vllm.models.kimi_k3.nvidia.ops import kda_segscan
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    kda_segscan.load_extension()
    ops = torch.ops.k3_flashkda_split
    dev = torch.device("cuda")
    di = dev.index or 0
    sm_die = torch.tensor([max(d, 0) for d in dies.sm_domains(di)], dtype=torch.int8,
                          device=dev)  # fmt: skip
    claim = torch.zeros(3, dtype=torch.int32, device=dev)
    sm_die_inv = (1 - sm_die).to(torch.int8)
    H, P = args.heads, args.heads * 128
    for T in [int(v) for v in args.tokens.split(",")]:
        torch.manual_seed(0)
        x_s = torch.randn(T, H, 128, device=dev).to(torch.bfloat16)
        rest_s = torch.randn(T, args.rest, device=dev).to(torch.bfloat16)
        x_d = homed(T, P * 2, dev).view(torch.bfloat16).view(T, H, 128)
        x_d.copy_(x_s)
        rest_d = homed(T, args.rest * 2, dev).view(torch.bfloat16).view(T, args.rest)
        rest_d.copy_(rest_s)
        w = (torch.rand(128, device=dev) + 0.5).float()
        sf_n = ((T + 127) // 128) * 128 * (P // 32)
        cfg = {
            "A": (x_s, rest_s, None),
            "C": (x_d, rest_d, None),
            "B8": (x_d, rest_d, (8, 0)),
            "B32": (x_d, rest_d, (32, 0)),
            "R": (x_d, rest_d, (0, 1)),
            "Ri": (x_d, rest_d, (0, 1, True)),
            "B8i": (x_d, rest_d, (8, 0, True)),
            "D8": (x_s, rest_s, (8, 0)),
        }
        arms = list(cfg)
        outs = {k: (torch.empty(T, P, dtype=torch.float8_e4m3fn, device=dev),
                    torch.zeros(sf_n, dtype=torch.uint8, device=dev)) for k in arms}  # fmt: skip
        nd = [int((sm_die == d).sum()) for d in (0, 1)]

        def run(k):
            x, rest, die = cfg[k]
            q, sf = outs[k]
            g = rest[:, :P]
            if die is not None:
                inv = len(die) > 2
                ops.o_norm_mxfp8_dl(x, g, w, 1e-5, q, sf, sm_die_inv if inv else sm_die,
                                    claim, die[0], die[1], nd[int(inv)], nd[1 - int(inv)])  # fmt: skip
            else:
                ops.o_norm_mxfp8(x, g, w, 1e-5, q, sf)

        for k in arms:
            run(k)
        torch.cuda.synchronize()
        res = {"T": T, "H": H}
        qa, sa = outs["A"]
        for k in arms[1:]:
            q, sf = outs[k]
            res["bit_" + k] = bool(torch.equal(q.view(torch.uint8), qa.view(torch.uint8))
                                   and torch.equal(sf, sa))  # fmt: skip
        res["claim_after"] = claim.tolist()
        times = {k: [] for k in arms}
        for _ in range(args.rounds):
            for k in arms + arms[::-1]:
                times[k].append(kernel_times(lambda k=k: run(k), iters=args.iters)[0])
        for k in arms:
            res[k] = round(statistics.median(times[k]), 1)
            res[k + "-A"] = round(res[k] - statistics.median(times["A"]), 1)
        res["sm_die_count"] = [int((sm_die == d).sum()) for d in (0, 1)]
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
