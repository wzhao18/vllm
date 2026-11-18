"""mb lane prefill: Kimi K3.1 prefill step (one prefill instance) under `vllm bench forward`.

Wraps the r3pf P1 harness (bench_prefill_step.py: eager timing of the whole forward with a
rank-max, torch.profiler + per-module record_function ranges; bench_dep_step.py: file
rendezvous, layer truncation K31W0_LAYERS) with:
  * the nq_step.py MX-aware random indexer-key fill (dsrel/tools/nq_step.py patch_fill);
  * PF_MOE_QUANT=mxfp8 (default): pin the vendored Rubin MegaMoE to MXFP8 activations x
    MXFP4 weights at every tokens/rank (the release policy switches to NVFP4 W4A4 above
    512 tokens/rank, megamoe_rubin/experts.py RUBIN_MX_MAX_TOKENS_PER_RANK); the model's
    mxfp4_release prefill prices FP8 MMA. PF_MOE_QUANT=release keeps the release policy.
  * logs the vllm tree, the MegaMoE quant kind and the fast-path markers.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, "/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/dsrel/tools")

RANK0 = os.environ.get("SLURM_PROCID", "0") == "0"


def log(*a):
    if RANK0:
        print("PF:", *a, flush=True)


def pin_moe_quant():
    mode = os.environ.get("PF_MOE_QUANT", "mxfp8")
    if mode == "release":
        log("MegaMoE quant policy: release (NVFP4 above 512 tokens/rank)")
        return
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import experts

    experts.RUBIN_MX_MAX_TOKENS_PER_RANK = 1 << 30
    log(f"MegaMoE quant policy pinned: MXFP8 act x MXFP4 weights at every tokens/rank "
        f"(rubin_quant_kind(16384) = {experts.rubin_quant_kind(16384)})")


def main():
    import argparse

    from bench_dep_step import patch_truncation, rendezvous
    from bench_prefill_step import patch_eager_timer
    from nq_step import patch_fill

    ap = argparse.ArgumentParser()
    ap.add_argument("--init-file", default="")
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    rendezvous(args.init_file)
    import torch
    import vllm

    log(f"vllm {vllm.__version__} from {os.path.dirname(vllm.__file__)} torch {torch.__version__}")
    try:
        import flashinfer

        log(f"flashinfer {flashinfer.__version__} from {os.path.dirname(flashinfer.__file__)}")
    except Exception as e:  # noqa: BLE001
        log(f"flashinfer import: {e}")
    patch_truncation()
    patch_fill(True, False)
    patch_eager_timer()
    pin_moe_quant()
    from vllm.benchmarks.forward.main import add_cli_args
    from vllm.benchmarks.forward.main import main as forward_main
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    forward_main(parser.parse_args(rest))


if __name__ == "__main__":
    main()
