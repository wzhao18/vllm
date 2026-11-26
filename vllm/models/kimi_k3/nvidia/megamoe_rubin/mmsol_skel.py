# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXPERIMENT ONLY (never for merge): mmsol_skel, MM-SOL B2 skeleton in step (P1-IS arms S1 / S1-nopush).

prefill/designs/megamoe-sol.md sec 4. Launched right after every stock Rubin MegaMoE launch (router + main + reduce)
on all EP ranks, in every MoE layer of the 2K / 4K buckets (n = 2048 / 4096 rows per rank), on the same stream,
eager only. The stock chain still produces the step's output. Placement B and lifetime proof as the push probe
(mmsol_probe.py): writes go only to pre_reduced_activation (dispatch + combine pushes), fc1_activation (FC1-out rows)
and fc2_order_output (non-last copies, SE intermediate / output), all rewritten by the stock kernel before it reads
them; no rank pushes before every rank entered (entry count after its own reduce), a rank exits only after every
incoming push landed (done count). Counters: 320 words in the arena signal pad (reset in-kernel each call except the
monotone entry / done / window counts). Stamps: per CTA (12 int64) + sampled per-tile stage times, dumped to
K31_MMSOL_STAMPS_DIR. Env: K31_MMSOL_SKEL=1 launches; K31_MMSOL_SKEL_MODE=push (S1) | nopush (S1-nopush: the same
bytes stored to this rank's own regions, no NVLink).
"""

from __future__ import annotations

import os
import threading

import numpy as np
import torch

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # standalone solo test
    import logging

    logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PRE = "nvlink.token_comm_dedup.pre_reduced_activation"
_PRE_SF = "nvlink.token_comm_dedup.pre_reduced_activation_sf"
_ORDER = "nvlink.token_comm_dedup.fc2_order_output"
_FC1 = "nvlink.token_comm_dedup.fc1_activation"
_FC1_SF = "nvlink.token_comm_dedup.fc1_activation_sf"
_RING = 32
_NST = 12
_CTR_WORDS = 320
_SPIN_NS = 20_000_000_000
E = 112

# Set by the model's MoE layer (experiment hook) before the routed experts run: (gate_up_w, down_w, se_input) tensors.
SE_CTX: tuple | None = None

_CPP = r"""
#include <torch/extension.h>
torch::Tensor mmsol_skel_build(std::vector<int64_t>, std::vector<int64_t>);
void mmsol_skel_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                       std::vector<int64_t>, std::vector<int64_t>);
int64_t mmsol_skel_smem();
int64_t mmsol_skel_args_bytes();
void mmsol_skel_warm(int64_t);
void mmsol_skel_zero(int64_t, int64_t);
"""
_lock = threading.Lock()
_mod = None


def module():
    global _mod
    with _lock:
        if _mod is None:
            import hashlib

            from torch.utils.cpp_extension import load_inline

            core = open(os.path.join(_HERE, "mmsol_core.cuh")).read()
            src = open(os.path.join(_HERE, "mmsol_skel.cu")).read().replace('#include "mmsol_core.cuh"', core)
            abl = int(os.environ.get("K31_MMSOL_SKEL_ABL", "0"))  # solo ablations only (never in the in-step arms)
            if abl:
                src = f"#define MMSOL_ABL {abl}\n" + src
            h = hashlib.sha256((src + _CPP + torch.__version__ + str(torch.version.cuda)).encode()).hexdigest()[:16]
            root = os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
            build = os.path.join(root, "k3_mmsol_skel", h)
            os.makedirs(build, exist_ok=True)
            _mod = load_inline(
                name=f"k3_mmsol_skel_{h}", cpp_sources=_CPP, cuda_sources=src,
                functions=["mmsol_skel_build", "mmsol_skel_launch", "mmsol_skel_smem", "mmsol_skel_args_bytes",
                           "mmsol_skel_warm", "mmsol_skel_zero"],
                extra_cuda_cflags=["-O3", "-std=c++20", "-gencode", "arch=compute_107a,code=sm_107a", "-lineinfo"],
                extra_ldflags=["-lcuda"], build_directory=build, verbose=False,
            )  # fmt: skip
    return _mod


def shape(n: int) -> dict:
    """Per-call work decomposition (design sec 2) for n rows per rank at EP8 balanced routing."""
    te = n * 16 / E
    n_u = int(np.ceil(te / 320))
    t_u0 = int(np.ceil(te / n_u))
    t_u1 = int(round(te)) - t_u0 if n_u > 1 else 0
    nwin = 16 if n <= 2048 else 28
    W = E // nwin
    se_mt, se_dt = (n + 511) // 512, (n + 255) // 256
    ngu = se_mt * 48
    ntiles = ngu + E * n_u * 38 + se_dt * 28
    return dict(n=n, n_u=n_u, t_u0=t_u0, t_u1=t_u1, nwin=nwin, W=W, se_mt=se_mt, se_dt=se_dt, ngu=ngu, ntiles=ntiles)


class MmsolSkel:
    """One per RubinMegaMoE runtime (EP group). Built and module-loaded at construction."""

    def __init__(self, arena, rank: int, world: int, ctr_ptrs: list[int] | None = None, offsets: list[int] | None = None):
        self.launch_on = os.environ.get("K31_MMSOL_SKEL", "0") == "1"
        self.push = int(os.environ.get("K31_MMSOL_SKEL_MODE", "push") != "nopush")
        self.rank, self.world = rank, world
        dev = torch.device("cuda", torch.cuda.current_device())
        self.mod = module()
        self.smem = int(self.mod.mmsol_skel_smem())
        self.mod.mmsol_skel_warm(self.smem)
        self.grid = int(torch.cuda.get_device_properties(dev).multi_processor_count) // 2 * 2
        if ctr_ptrs is None:
            pads, size = list(arena.handle.signal_pad_ptrs), int(arena.handle.signal_pad_size)
            off = size - 64 - 2048
            assert off >= 0 and off + 4 * _CTR_WORDS <= size - 64
            ctr_ptrs = [p + off for p in pads]
            offsets = list(arena.offsets)
        self.ctr = int(ctr_ptrs[rank])
        self.ctr_peer = torch.tensor(ctr_ptrs, dtype=torch.int64, device=dev)
        self.offs = torch.tensor(offsets, dtype=torch.int64, device=dev)
        self.mod.mmsol_skel_zero(self.ctr, 4 * _CTR_WORDS)
        self.stamps = torch.zeros((_RING, self.grid, _NST), dtype=torch.int64, device=dev)
        self.tst = torch.zeros((_RING, self.grid, 128), dtype=torch.int64, device=dev)
        self.hdr = torch.zeros((_RING, 8), dtype=torch.int64, device=dev)
        half = _RING // 2
        self.h_hdr = torch.zeros((half, 8), dtype=torch.int64, pin_memory=True)
        self.h_st = torch.zeros((half, self.grid, _NST), dtype=torch.int64, pin_memory=True)
        self.h_ts = torch.zeros((half, self.grid, 128), dtype=torch.int64, pin_memory=True)
        self.dump_stream = torch.cuda.Stream(device=dev)
        self.dump_event = None
        self.dump_pending = False
        self.stamps_dir = os.environ.get("K31_MMSOL_STAMPS_DIR", "")
        self.calls = 0
        self.disp_hi = 0
        self.maps: dict = {}
        torch.cuda.synchronize()
        if rank == 0:
            logger.info("MMSOL skel built: launch=%s mode=%s grid=%d smem=%d KiB ctr=0x%x stamps=%s", self.launch_on,
                        "push" if self.push else "nopush", self.grid, self.smem >> 10, self.ctr, self.stamps_dir or "-")

    # ------------------------------------------------------------------------------------------------------------
    def launch(self, regions: dict, n: int) -> None:
        """regions: name -> (ptr, bytes): W1 W2 SFW X1 X2 SFX SEW1 SEW2 SFSE SEX1 ORDER PRE FC1OUT XSRC."""
        sh = shape(n)
        key = (n,) + tuple(regions[k][0] for k in ("W1", "W2", "SEW1", "SEW2", "SEX1", "ORDER", "PRE"))
        maps = self.maps.get(key)
        o_ptr, o_b = regions["ORDER"]
        se_inter = o_b // 2
        if maps is None:
            P = []
            for k in ("W1", "W2", "SFW", "X1", "X2", "SFX", "SEW1", "SEW2", "SFSE", "SEX1"):
                P += [int(regions[k][0]), int(regions[k][1])]
            P += [int(o_ptr + se_inter), int(o_b - se_inter)]
            maps = self.mod.mmsol_skel_build(P, [])
            self.maps[key] = maps
        pre_ptr, pre_b = regions["PRE"]
        slice_b = (pre_b // self.world) // 256 * 256
        disp_per_dest = n * 2080
        comb_base = (disp_per_dest + 255) // 256 * 256
        fc1_ptr, fc1_b = regions["FC1OUT"]
        x_ptr, x_b = regions["XSRC"]
        prev_lo, prev_hi = self.calls, self.disp_hi
        self.calls += 1
        if sh["nwin"] > 16:
            self.disp_hi += 1
        I = [n, sh["n_u"], sh["nwin"], sh["W"], sh["se_mt"], sh["se_dt"], sh["ntiles"], sh["ngu"], sh["t_u0"],
             sh["t_u1"], self.rank, self.world, self.push, 0, prev_lo, prev_hi, self.calls, self.grid, self.smem]
        Q = [int(fc1_ptr), int(fc1_b), int(o_ptr), int(o_b), int(se_inter), int(o_b * 3 // 4), int(pre_ptr), int(slice_b),
             int(comb_base), int(disp_per_dest), int(x_ptr), int(max(16, x_b - 8192)), self.ctr, _SPIN_NS]
        self.mod.mmsol_skel_launch(maps, self.offs, self.ctr_peer, self.stamps, self.hdr, self.tst, I, Q)
        if self.stamps_dir and self.calls % (_RING // 2) == 0:
            self._dump()

    def __call__(self, session, n: int, weights) -> None:
        if not self.launch_on or n not in (2048, 4096) or torch.cuda.is_current_stream_capturing():
            return
        ws = session.kernel._device_workspace
        names = {sp: {r.name: r for r in ws.regions(sp)} for sp in ("local", "shared", "tmp")}

        def region(name, prefer=("tmp", "local", "shared")):
            for sp in prefer:
                if name in names[sp]:
                    base = {"tmp": session.tmp_buffer, "local": session.local_workspace, "shared": session.shared_workspace}[sp]
                    if base is None:
                        continue
                    return (base.data_ptr() + ws.offset(name), ws.nbytes(name))
            return None

        pre = region(_PRE, ("shared",))
        order = region(_ORDER)
        if pre is None or order is None:
            raise RuntimeError(f"MMSOL skel: regions missing: {sorted(names['shared'])} / {sorted(names['tmp'])}")
        fc1 = region(_FC1) or (order[0] + order[1] * 7 // 8, order[1] // 8)
        sfx = region(_FC1_SF) or region(_PRE_SF, ("shared",)) or pre
        w1, w1s = weights.fc1
        w2, _ = weights.fc2
        x = session.x.view(torch.uint8).view(-1)
        se = SE_CTX
        if se is not None:
            gu, dn, sin = se
            sew1 = (gu.data_ptr(), gu.numel() * gu.element_size())
            sew2 = (dn.data_ptr(), dn.numel() * dn.element_size())
            sex1 = (sin.data_ptr(), sin.numel() * sin.element_size())
        else:
            sew1 = (w1.data_ptr(), 12288 * 7168)
            sew2 = (w1.data_ptr() + 12288 * 7168, 7168 * 6144)
            sex1 = (pre[0], n * 7168)
        regions = dict(
            W1=(w1.data_ptr(), w1.numel() * w1.element_size()), W2=(w2.data_ptr(), w2.numel() * w2.element_size()),
            SFW=(w1s.data_ptr(), w1s.numel() * w1s.element_size()), X1=pre, X2=fc1, SFX=sfx, SEW1=sew1, SEW2=sew2,
            SFSE=(w1s.data_ptr(), w1s.numel() * w1s.element_size()), SEX1=sex1, ORDER=order, PRE=pre, FC1OUT=fc1,
            XSRC=(x.data_ptr(), x.numel()),
        )  # fmt: skip
        if self.calls == 0 and self.rank == 0:
            logger.info("MMSOL skel first launch n=%d: %s", n,
                        {k: (hex(v[0]), round(v[1] / 2**20, 1)) for k, v in regions.items()})
        self.launch(regions, n)

    def _dump(self) -> None:
        half = _RING // 2
        if self.dump_pending:
            self.dump_event.synchronize()
            os.makedirs(self.stamps_dir, exist_ok=True)
            with open(os.path.join(self.stamps_dir, f"rank{self.rank}.bin"), "ab") as f:
                f.write(np.concatenate([self.h_hdr.numpy(), self.h_st.numpy().reshape(half, -1),
                                        self.h_ts.numpy().reshape(half, -1)], axis=1).tobytes())
        lo = (self.calls - half) % _RING
        self.dump_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.dump_stream):
            self.h_hdr.copy_(self.hdr[lo : lo + half], non_blocking=True)
            self.h_st.copy_(self.stamps[lo : lo + half], non_blocking=True)
            self.h_ts.copy_(self.tst[lo : lo + half], non_blocking=True)
            self.dump_event = torch.cuda.Event()
            self.dump_event.record(self.dump_stream)
        self.dump_pending = True
