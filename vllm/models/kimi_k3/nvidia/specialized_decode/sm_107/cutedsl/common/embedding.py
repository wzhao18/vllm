# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local CuTe vocabulary gather plus the ordinary framework TP all-reduce, or (Rubin
TP16 decode) the gather publishing each row from its owner rank into a one-slot
Lamport mailbox that the first AttnRes polls (no all-reduce: exactly one rank holds a
token's row, the others contribute zeros)."""

from functools import cache

import cutlass as c
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from vllm.distributed import get_tp_group

from .host import architecture, fake
from .primitives import asm, launch, load_words, multicast_words


class EmbeddingGather:
    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (7, args[0].shape[0], 1), 128, pdl=False)

    @cute.kernel
    def kernel(self, args):
        ids, weight, out, start, vocab = args[0:5]
        lane, _, _ = cute.arch.thread_idx()
        tile, token, _ = cute.arch.block_idx()
        token_id = c.Int64(ids[token])
        row, column = token_id - start, tile * 1024 + lane * 8
        values = cute.make_rmem_tensor((8,), c.BFloat16)
        values.fill(c.BFloat16(0))
        if (token_id >= 0) & (token_id < vocab) & (row >= 0) & (row < weight.shape[0]):
            source = weight.iterator + row * 7168 + column
            cute.autovec_copy(cute.make_tensor(source, cute.make_layout((8,))), values)
        destination = out.iterator + token * 7168 + column
        cute.autovec_copy(values, cute.make_tensor(destination, cute.make_layout((8,))))


@asm(c.Uint32, pure=True)
def latent_word(word):
    """BF16x2 ``word`` with negative-zero halves made positive (the mailbox sentinel;
    the all-reduce it replaces returns +0 for them too: -0 + 0 = +0)."""
    code = (
        "{ .reg .b32 bits; .reg .pred p; mov.b32 $0,$1; "
        "and.b32 bits,$0,0xffff; setp.eq.u32 p,bits,0x8000; "
        "@p and.b32 $0,$0,0xffff0000; "
        "shr.u32 bits,$0,16; setp.eq.u32 p,bits,0x8000; @p and.b32 $0,$0,0xffff; }"
    )
    return code, "=r,r", (c.Uint32(word),)


class RingEmbeddingGather:
    """``EmbeddingGather`` that multicasts each token row from the rank owning its id
    (rank 0 writes zeros for ids no shard holds) into the [cap, 7168] mailbox slot at
    multicast address ``ring``; triggers its PDL dependents at entry."""

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (7, args[0].shape[0], 1), 128, pdl=False)

    @cute.kernel
    def kernel(self, args):
        ids, weight, start, vocab, ring = args[0:5]
        cute.arch.griddepcontrol_launch_dependents()
        lane, _, _ = cute.arch.thread_idx()
        tile, token, _ = cute.arch.block_idx()
        token_id = c.Int64(ids[token])
        row, column = token_id - start, tile * 1024 + lane * 8
        invalid = (token_id < 0) | (token_id >= vocab)
        mine = (row >= 0) & (row < weight.shape[0])
        if invalid:
            if start == 0:
                zero = c.Uint32(0)
                address = ring + (c.Int64(token) * 7168 + column) * 2
                multicast_words(address, (zero, zero, zero, zero), kind="bf16x2")
        elif mine:
            source = weight.iterator + row * 7168 + column
            words = load_words(source, count=4)
            words = tuple(latent_word(w) for w in words)
            address = ring + (c.Int64(token) * 7168 + column) * 2
            multicast_words(address, words, kind="bf16x2")


@cache
def compile_ring_gather(dtype, shard, arch):
    batch = cute.sym_int32()
    args: tuple = (fake((batch,), None, dtype, dtype.itemsize), fake((shard, 7168)))
    args += (c.Int64(0), c.Int64(0), c.Int64(0))
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    options = f"--gpu-arch={arch} --enable-tvm-ffi"
    return cute.compile(RingEmbeddingGather(), *args, stream, options=options)


@cache
def compile_gather(dtype, shard, arch):
    batch = cute.sym_int32()
    args: tuple = (fake((batch,), None, dtype, dtype.itemsize), fake((shard, 7168)))
    args += (fake((batch, 7168)), c.Int64(0), c.Int64(0))
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    options = f"--gpu-arch={arch} --enable-tvm-ffi"
    return cute.compile(EmbeddingGather(), *args, stream, options=options)


class PreparedEmbedding:
    def __init__(self, embedding, tp, rank):
        weight = embedding.weight
        # The gather moves 7 x 1024 BF16 columns of contiguous rows.
        if weight.dim() != 2 or weight.shape[1] != 7168 or not weight.is_contiguous():
            raise ValueError(
                f"embedding shard must be a contiguous [V, 7168] tensor, got "
                f"{tuple(weight.shape)}"
            )
        if weight.dtype != torch.bfloat16:
            raise ValueError(f"embedding shard must be BF16, got {weight.dtype}")
        self.group, self.weight = get_tp_group(), weight.detach()
        self.start, self.vocab = rank * weight.shape[0], embedding.org_vocab_size
        with torch.accelerator.device_index(weight.device.index):
            arch, ids = architecture(weight.device), (torch.int32, torch.int64)
            self.kernels = {t: compile_gather(t, weight.shape[0], arch) for t in ids}
        self.ring_kernels, self.ring = {}, 0

    def bind_ring(self, multicast):
        """Publish into the one-slot [cap, 7168] mailbox at multicast address
        ``multicast`` (``run_ring``)."""
        weight = self.weight
        with torch.accelerator.device_index(weight.device.index):
            arch, ids = architecture(weight.device), (torch.int32, torch.int64)
            shard = weight.shape[0]
            self.ring_kernels = {t: compile_ring_gather(t, shard, arch) for t in ids}
        self.ring = multicast

    def run_ring(self, ids):
        """Rows go to the mailbox only; its consumer must run in the same step."""
        if ids.dim() != 1 or ids.dtype not in self.ring_kernels:
            raise ValueError(
                f"token ids must be 1-D int32/int64, got {ids.dtype} {tuple(ids.shape)}"
            )
        self.ring_kernels[ids.dtype](
            ids, self.weight, self.start, self.vocab, self.ring
        )

    def run(self, ids, out):
        if ids.dim() != 1 or ids.dtype not in self.kernels:
            raise ValueError(
                f"token ids must be 1-D int32/int64, got {ids.dtype} {tuple(ids.shape)}"
            )
        if out.shape != (ids.shape[0], 7168) or not out.is_contiguous():
            raise ValueError(
                f"embedding output must be contiguous [{ids.shape[0]}, 7168]"
            )
        self.kernels[ids.dtype](ids, self.weight, out, self.start, self.vocab)
        reduced = self.group.all_reduce(out)
        if reduced is not out:
            out.copy_(reduced)
        return out
