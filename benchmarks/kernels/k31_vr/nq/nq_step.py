"""Whole Kimi K3.1 decode step (target verify + DSpark drafter) under `vllm bench forward`, for
the nq stock-vs-specialized A/B: 16 ranks under srun (--init-file rendezvous), one process
per arm. A wrapper only: vllm/benchmarks is untouched.

Derived from $W/c1/dsacore/tools/bench_b1_step.py (= b1/r2x/tools/bench_b1_step.py, the
merge-gate / oracle harness, with the 2026-11-01 MX-aware indexer fill), minus its arm and
fault hooks, plus:

* FULL decode graphs where the specialized decode reads them. bench forward sets
  cudagraph_mode NONE and the config post-init then clears the capture sizes
  (vllm/benchmarks/forward/main.py _make_engine_args; config/vllm.py _set_cudagraph_sizes),
  so the specialized decode would refuse the deployment (runtime.unsupported: no FULL
  decode graphs) and the drafter would see no batches (draft.draft_batches() == (): stock
  drafter, ~0.9 ms at B1). Instead of re-deriving the captured rows (bench_b1_step used
  ``captured_rows(sizes, q, 1 << 20)`` and ``s // q`` drafter batches, neither the
  production rounding), the wrapper calls the real ``specialized_decode_config`` and
  ``draft_batches`` with the production compilation fields (FULL_AND_PIECEWISE, the
  config's capture sizes and max capture size) set on the config for the duration of the
  call: limit = min(max_num_seqs * q, max capture) and round_up(s, K) / K come from the
  code under test, whatever it becomes. The stock arm (decode_backend auto) is untouched.
* --overlay DIR: the code under test. DIR is a whole vllm tree (stage_overlay.sh: the base
  env vllm-jh-242fb49cd4, hard-linked, with every file of the checkout that differs from
  242fb49cd4 under vllm/ shipped over it, listed in DIR/SOURCE_REV); nq_inner.sh puts DIR
  first on PYTHONPATH in place of the base env. The wrapper checks that vllm and kimi_k3
  resolve inside DIR and rank 0 logs the revision and the shipped files.
* Random indexer keys after each case's KV zeroing (on by default; --no-fill-index): the
  bench never prefills, and an all-zero indexer cache makes the DCP top-k one 100K-token tie
  (not what serving sees). Scale word = 1.0 in the reader's format: packed UE8M0 0x7F7F7F7F
  where the MQA logits read MX scales (SM10x incl. SM107), else FP32 1.0. The MXFP4 keys of
  the nq configs (68 B/token: 64 B of e2m1 pairs + 4 UE8M0 scales) take random bytes as
  e2m1 pairs, as the C256 recipe did (merge/tools/t10/mt_n4_m10.sh). The MLA cache stays
  zero unless --fill-mla (the sparse core's cost does not depend on its values).

What the median is: one CUDA graph per case of the target forward plus the drafter's K-step
propose with every draft accepted, rank-max per replay, median of iters x samples
(capture.py). Not included in either arm: target logits / LM head, rejection or sampled
verify (incl. specialized runtime/sampled_verify.py) and CPU step overhead. The 'graph vs
eager rel rms' of the ls arm compares the specialized graph with the stock eager pass and
saturates at ~1.4 with dummy weights over 93 layers: it is not a correctness signal.
"""
import argparse
import os
import socket
import sys
import time

RANK0 = os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")) == "0"


def log(*a):
    if RANK0:
        print("NQ:", *a, flush=True)


def parse_sizes(spec):
    sizes = set()
    for item in spec.split(","):
        if item.strip():
            lo, _, hi = item.partition("-")
            sizes.update(range(int(lo), int(hi or lo) + 1))
    return tuple(sorted(sizes))


def rendezvous(init_file):
    if "RANK" in os.environ or "SLURM_PROCID" not in os.environ:
        return
    rank = int(os.environ["SLURM_PROCID"])
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=os.environ["SLURM_NTASKS"],
        LOCAL_RANK=os.environ["SLURM_LOCALID"],
    )
    if rank == 0:
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        address = f"{socket.gethostbyname(socket.gethostname())}:{port}"
        with open(init_file + ".tmp", "w") as f:
            f.write(address)
        os.rename(init_file + ".tmp", init_file)
    else:
        while not os.path.exists(init_file):
            time.sleep(0.2)
        with open(init_file) as f:
            address = f.read()
    host, port = address.split(":")
    os.environ.update(MASTER_ADDR=host, MASTER_PORT=port)


def apply_overlay(root):
    """Check that vLLM is imported from the overlay tree ``root`` (stage_overlay.sh: the
    base env vllm-jh-242fb49cd4 with every file of the checkout that differs from
    242fb49cd4 under vllm/ shipped over it; nq_inner.sh puts ``root`` first on
    PYTHONPATH in place of the base env) and log what it carries."""
    if not root:
        log("overlay: none (base env vllm-jh-242fb49cd4)")
        return
    if not os.path.isfile(os.path.join(root, "vllm", "__init__.py")):
        raise RuntimeError(
            f"overlay {root} is not a whole vllm tree (no vllm/__init__.py): restage it "
            "with stage_overlay.sh"
        )
    import vllm
    import vllm.models.kimi_k3 as k3

    real = os.path.realpath(root) + os.sep
    for mod in (vllm, k3):
        if not os.path.realpath(mod.__file__).startswith(real):
            raise RuntimeError(f"overlay not applied: {mod.__name__} from {mod.__file__}")
    rev, shipped = "", []
    p = os.path.join(root, "SOURCE_REV")
    if os.path.isfile(p):
        with open(p) as f:
            lines = f.read().splitlines()
        rev = lines[0] if lines else ""
        shipped = [x.split(None, 1)[1] for x in lines if x.startswith(("M ", "A ", "D "))]
    log(f"overlay: vllm from {os.path.dirname(vllm.__file__)} rev {rev or '?'}")
    log(f"overlay: {len(shipped)} files differ from the base under vllm/: {shipped}")


def patch_specialized_decode(sizes, max_capture):
    """Run the real specialized_decode_config / draft_batches with the production
    compilation fields (see the module docstring)."""
    import vllm.models.kimi_k3.nvidia.model as model
    import vllm.models.kimi_k3.nvidia.specialized_decode.runtime as runtime
    from vllm.config import CUDAGraphMode
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import draft

    def specialized(vllm_config):
        extra = vllm_config.additional_config
        return isinstance(extra, dict) and extra.get("decode_backend", "auto") != "auto"

    def production(fn):
        def call(vllm_config, *args):
            if not specialized(vllm_config):
                return fn(vllm_config, *args)
            c = vllm_config.compilation_config
            saved = c.cudagraph_mode, c.cudagraph_capture_sizes, c.max_cudagraph_capture_size
            c.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
            c.cudagraph_capture_sizes = list(sizes)
            c.max_cudagraph_capture_size = max_capture
            try:
                return fn(vllm_config, *args)
            finally:
                (c.cudagraph_mode, c.cudagraph_capture_sizes,
                 c.max_cudagraph_capture_size) = saved  # fmt: skip

        return call

    build = production(runtime.specialized_decode_config)

    def specialized_decode_config(vllm_config):
        config = build(vllm_config)
        if config is not None:
            rows = config.captured
            log(
                f"specialized decode: q {config.q} dcp {config.dcp} bf16_state "
                f"{config.bf16_state} captured {len(rows)} rows "
                f"{rows[:1]}..{rows[-1:]} (capture sizes {len(sizes)}, max {max_capture})"
            )
        return config

    runtime.specialized_decode_config = specialized_decode_config
    model.specialized_decode_config = specialized_decode_config
    batches = production(draft.draft_batches)

    def draft_batches(vllm_config, steps):
        out = batches(vllm_config, steps)
        if specialized(vllm_config):
            log(f"drafter fast-path candidate batches (draft_batches, K={steps}): {list(out)}")
        return out

    draft.draft_batches = draft_batches


KERNEL_PAGE = 64


def patch_fill(fill_index, fill_mla):
    """$W/c1/dsacore/tools/bench_b1_step.py patch_fill_index (2026-11-01 fix): random index
    keys per 64-key kernel page with an MX-aware 1.0 scale word; optional MLA rows."""
    if not fill_index:
        log("indexer fill: off (all-zero keys: one-tie top-k)")
        return
    import torch

    from vllm.benchmarks.forward import kv_init

    zero = kv_init.zero_kv_blocks

    @torch.inference_mode()
    def zero_kv_blocks(kv_caches, attn_groups, batch, num_blocks):
        zero(kv_caches, attn_groups, batch, num_blocks)
        from vllm.utils.deep_gemm import mqa_logits_use_mx_scales

        mx = mqa_logits_use_mx_scales()
        word = 0x7F7F7F7F if mx else 0x3F800000
        rank = int(os.environ.get("RANK", "0"))
        gen = torch.Generator(device="cuda")
        gen.manual_seed(1000 + rank)
        counts = {}
        for group in attn_groups:
            names = group.layer_names
            if not names:
                continue
            indexer = all("indexer" in n for n in names)
            if not indexer and (
                not fill_mla
                or any("indexer" in n for n in names)
                or isinstance(group.kv_cache_spec, kv_init.MambaSpec)
            ):
                continue
            ranges = batch.group_block_ranges[group.kv_cache_group_id]
            for tensor in kv_init._group_tensors(kv_caches, group, num_blocks):
                if tensor.dim() < 3:
                    continue
                dim, factor = kv_init._block_dim(tensor, num_blocks)
                assert dim == 0, tensor.shape
                for begin, end in ranges:
                    pages = tensor.narrow(0, begin * factor, (end - begin) * factor)
                    tokens = pages.shape[1]
                    flat = pages.view(torch.uint8).view(pages.shape[0], -1)
                    per_tok = flat.shape[1] // tokens
                    if indexer:
                        if tokens > KERNEL_PAGE:
                            assert tokens % KERNEL_PAGE == 0, tokens
                            flat = flat.unflatten(1, (tokens // KERNEL_PAGE, -1))
                            tokens = KERNEL_PAGE
                        dim_k = per_tok - 4
                        vals = torch.randn(
                            (*flat.shape[:-1], tokens * dim_k), device=flat.device, generator=gen
                        )
                        flat[..., : tokens * dim_k] = (
                            (vals * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
                        )
                        flat[..., tokens * dim_k :].view(torch.int32).fill_(word)
                        key = f"indexer:{per_tok}B/token+{word:#x}"
                    elif per_tok == 576:  # per-tensor FP8 latent rows (k_scale 1.0)
                        rec = flat.view(flat.shape[0], tokens, per_tok)
                        step = max(1, (1 << 22) // (tokens * per_tok))
                        for r0 in range(0, rec.shape[0], step):
                            r = rec[r0 : r0 + step]
                            v = torch.randn(r.shape, device=r.device, generator=gen) * 0.5
                            r.copy_(v.to(torch.float8_e4m3fn).view(torch.uint8))
                        key = "mla:fp8"
                    else:
                        key = f"skipped({per_tok}B/token)"
                    counts[key] = counts.get(key, 0) + pages.shape[0]
        if rank == 0:
            print(f"NQ: filled KV pages {counts} (mx_scales={mx})", flush=True)

    # NQ-TUNE: whether random FP8 bytes read as e2m1 pairs (MXFP4 keys) give the serving top-k cost: compare the
    # deep_select / DCP-exchange kernel times of a profiled case with an e2e decode iteration profile at the same B.
    kv_init.zero_kv_blocks = zero_kv_blocks
    log(f"indexer fill: random keys per 64-key page{' + MLA rows' if fill_mla else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-file", default="")
    ap.add_argument("--capture-sizes", required=True, help="production list (+ measured rows)")
    ap.add_argument("--max-capture", type=int, required=True)
    ap.add_argument("--overlay", default="")
    ap.add_argument("--no-fill-index", action="store_true")
    ap.add_argument("--fill-mla", action="store_true")
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    rendezvous(args.init_file)
    apply_overlay(args.overlay)
    sizes = parse_sizes(args.capture_sizes)
    patch_specialized_decode(sizes, args.max_capture)
    patch_fill(not args.no_fill_index, args.fill_mla)
    import torch

    import vllm

    log(f"vllm {vllm.__version__} ({os.path.dirname(vllm.__file__)}) torch {torch.__version__}")
    from vllm.benchmarks.forward.main import add_cli_args
    from vllm.benchmarks.forward.main import main as forward_main
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    forward_main(parser.parse_args(rest))


if __name__ == "__main__":
    main()
