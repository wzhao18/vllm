"""W0: Kimi K3.1 stock DEP decode step on VR under `vllm bench forward`.

One process per GPU under srun (``--init-file`` file rendezvous), DP x TP1 + EP MegaMoE.
Adapted from the K3.1 VR step harnesses (k31vr wide/int1 tools: rendezvous, layer
truncation, random indexer keys); the specialized-decode patches are dropped because the
DEP step is the stock forward.

Env knobs:
- ``K31W0_LAYERS=N``: truncate the target to its first N decoder layers (hf_overrides
  callable, which also reaches the DSpark draft config; its target_layer_ids are remapped
  below N). N=8 keeps the production 3 KDA : 1 DSA pattern (layers 4 and 8 are DSA).
- ``K31W0_FILL_INDEX=0`` disables the random indexer keys (default on). `vllm bench forward`
  never prefills, so the indexer cache is otherwise all zeros: every logit ties and the
  top-k sees one long tie, which is not what serving sees. The MLA cache stays zero (the
  sparse core's cost does not depend on its values).
"""

import argparse
import os
import socket
import time

LAYERS = int(os.environ.get("K31W0_LAYERS", "0"))
FILL_INDEX = os.environ.get("K31W0_FILL_INDEX", "1") == "1"


def rendezvous(init_file: str) -> None:
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


def truncate_layers(config):
    """hf_overrides callable: first LAYERS target layers; DSpark ids remapped below LAYERS."""
    text = getattr(config, "text_config", None)
    for c in (config, text):
        if c is None:
            continue
        if getattr(c, "model_type", "") == "k3_dspark" or hasattr(c, "target_layer_ids"):
            ids = list(getattr(c, "target_layer_ids", []) or [])
            if ids:
                n = len(ids)
                new = list(range(LAYERS - n, LAYERS)) if LAYERS >= n else ids
                c.target_layer_ids = new
                if hasattr(c, "dspark_target_layer_ids"):
                    c.dspark_target_layer_ids = new
            if hasattr(c, "target_num_hidden_layers"):
                c.target_num_hidden_layers = LAYERS
        elif getattr(c, "num_hidden_layers", 0) > LAYERS and hasattr(c, "dsa_config"):
            c.num_hidden_layers = LAYERS
    return config


def patch_truncation() -> None:
    if not LAYERS:
        return
    from vllm.engine.arg_utils import EngineArgs

    create = EngineArgs.create_engine_config

    def create_engine_config(self, *a, **k):
        self.hf_overrides = truncate_layers
        return create(self, *a, **k)

    EngineArgs.create_engine_config = create_engine_config
    print(f"K31W0: target truncated to {LAYERS} layers", flush=True)


def patch_fill_index() -> None:
    if not FILL_INDEX:
        return
    import torch

    from vllm.benchmarks.forward import kv_init

    zero = kv_init.zero_kv_blocks
    # FP32 1.0 scale word (the SM107 DeepGEMM logits read FP32 scales).
    scale = int(os.environ.get("K31W0_FILL_SCALE", "0x3F800000"), 0)

    @torch.inference_mode()
    def zero_kv_blocks(kv_caches, attn_groups, batch, num_blocks):
        zero(kv_caches, attn_groups, batch, num_blocks)
        gen = torch.Generator(device="cuda")
        gen.manual_seed(1000 + int(os.environ.get("RANK", "0")))
        filled = 0
        for group in attn_groups:
            names = group.layer_names
            if not names or not all("indexer" in n for n in names):
                continue
            ranges = batch.group_block_ranges[group.kv_cache_group_id]
            for tensor in kv_init._group_tensors(kv_caches, group, num_blocks):
                dim, factor = kv_init._block_dim(tensor, num_blocks)
                assert dim == 0, tensor.shape
                for begin, end in ranges:
                    pages = tensor.narrow(0, begin * factor, (end - begin) * factor)
                    flat = pages.view(torch.uint8).flatten(1)
                    tokens = pages.shape[1]
                    dim_k = flat.shape[1] // tokens - 4
                    vals = torch.randn(
                        (flat.shape[0], tokens * dim_k),
                        device=flat.device,
                        generator=gen,
                    )
                    flat[:, : tokens * dim_k] = (
                        (vals * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
                    )
                    flat[:, tokens * dim_k :].view(torch.int32).fill_(scale)
                    filled += flat.shape[0]
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"K31W0: filled {filled} indexer pages with random FP8 keys", flush=True)

    kv_init.zero_kv_blocks = zero_kv_blocks


def patch_capacity() -> None:
    """Log the KV pool in requests at several context lengths (the stock line only
    reports max concurrency at max_model_len, which mixes the per-request KDA state
    and drafter rows with the per-token rows) plus each group's page/blocks."""
    import copy

    from vllm.v1.core import kv_cache_utils as ku

    orig = ku.update_kv_cache_capacity

    def update_kv_cache_capacity(vllm_config, kv_cache_config):
        orig(vllm_config, kv_cache_config)
        try:
            parts = []
            for ctx in (65536, 131072, 196608, 262144, 524288, 1048576):
                cfg = copy.copy(vllm_config)
                cfg.model_config = copy.copy(vllm_config.model_config)
                cfg.model_config.max_model_len = ctx
                c = ku.get_max_concurrency_for_kv_cache_config(cfg, kv_cache_config)
                parts.append(f"{ctx // 1024}K:{c:.2f}")
            groups = []
            for g in kv_cache_config.kv_cache_groups:
                sp = g.kv_cache_spec
                inner = getattr(sp, "kv_cache_specs", None) or {}
                kinds = sorted(
                    {
                        f"{type(x).__name__}/{getattr(x, 'dtype', '?')}/{x.page_size_bytes}"
                        for x in inner.values()
                    }
                )
                groups.append(
                    f"{type(sp).__name__}(bs={getattr(sp, 'block_size', '?')},"
                    f"page={sp.page_size_bytes},layers={len(g.layer_names)}"
                    f"{',' + '|'.join(kinds) if kinds else ''})"
                )
            for t in kv_cache_config.kv_cache_tensors:
                groups.append(
                    f"tensor(size={t.size},layers={len(t.layers)},"
                    f"block_stride={t.block_stride},layer_stride={t.layer_stride},"
                    f"first={t.layers[0] if t.layers else ''})"
                )
            print(
                f"K31W0 capacity: num_blocks={kv_cache_config.num_blocks} "
                f"requests per instance at ctx {' '.join(parts)}; groups {'; '.join(groups)}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"K31W0 capacity: failed {e!r}", flush=True)

    ku.update_kv_cache_capacity = update_kv_cache_capacity
    from vllm.v1.engine import core  # imports the function by name

    core.update_kv_cache_capacity = update_kv_cache_capacity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-file", default="")
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    rendezvous(args.init_file)
    patch_truncation()
    patch_fill_index()
    patch_capacity()
    from vllm.benchmarks.forward.main import add_cli_args
    from vllm.benchmarks.forward.main import main as forward_main
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    forward_main(parser.parse_args(rest))


if __name__ == "__main__":
    main()
