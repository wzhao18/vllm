# SPDX-License-Identifier: Apache-2.0
"""M1 EXPERIMENT ONLY (never merge): dump the MegaMoE in-kernel stamp rings (src_gp/common/m1_stamps.py).

Sessions built while M1_STAMPS=1 register here; ``dump(tag)`` (called by the forward bench after each case when
M1_STAMPS_DIR is set) writes one torch file per session per rank: ctrl, header ring, record ring, layout facts.
"""

from __future__ import annotations

import os
import weakref

import torch

_SESSIONS: list = []
_ROUTES: dict = {}


def capture_routes(layer: int, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> None:
    """Eager forwards only (never inside a capture): the latest routing of each MoE layer."""
    _ROUTES[int(layer)] = (topk_ids.detach().clone(), topk_weights.detach().float().clone())


def register(session) -> None:
    _SESSIONS.append(weakref.ref(session))


def dump(tag: str) -> None:
    out_dir = os.environ.get("M1_STAMPS_DIR")
    if not out_dir:
        return
    from .src_gp.common import m1_stamps as M

    os.makedirs(out_dir, exist_ok=True)
    torch.cuda.synchronize()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    for i, ref in enumerate(_SESSIONS):
        s = ref()
        if s is None or getattr(s, "local_workspace", None) is None:
            continue
        ws = s.kernel._device_workspace
        lw = s.local_workspace

        def region(name, dtype, count):
            off = ws.offset(name)
            nbytes = count * torch.tensor([], dtype=dtype).element_size()
            return lw[off : off + nbytes].view(dtype).clone().cpu()

        blob = {
            "tag": tag,
            "rank": rank,
            "session": i,
            "kernel_name": s.name,
            "max_tokens": int(s.cfg.max_tokens_per_rank),
            "world_size": int(s.cfg.world_size),
            "ctas": int(s.kernel.token_comm.promised_launchable_sm_count),
            "slots": M.SLOTS,
            "maxcta": M.MAXCTA,
            "rec": M.REC,
            "hdr_words": M.HDR,
            "ctrl": region(M.CTRL_REGION, torch.int32, 4),
            "hdr": region(M.HDR_REGION, torch.int32, M.SLOTS * M.HDR).view(M.SLOTS, M.HDR),
            "recs": region(M.REC_REGION, torch.int64, M.SLOTS * M.MAXCTA * M.REC).view(M.SLOTS, M.MAXCTA, M.REC),
        }
        import numpy as np

        np.savez_compressed(
            os.path.join(out_dir, f"{tag}.rank{rank}.s{i}.npz"),
            **{k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in blob.items()},
        )
    if _ROUTES:
        import numpy as np

        arrays = {}
        for k, (a, b) in _ROUTES.items():
            arrays[f"ids{k}"] = a.cpu().numpy()
            arrays[f"w{k}"] = b.cpu().numpy()
        np.savez_compressed(os.path.join(out_dir, f"{tag}.rank{rank}.routes.npz"), **arrays)
        _ROUTES.clear()
