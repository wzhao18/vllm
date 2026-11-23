# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KDA conv window at the cache's dtype (``--mamba-cache-dtype``: BF16 or FP32).

The window only ever holds conv INPUTS, the BF16 in-projection rows (stock
``causal_conv1d_update`` casts x to the state dtype and stores it), so an FP32 window
holds BF16-representable values. The TP kernels compute on BF16 taps, as from a BF16
window: ``conv_tap`` converts an FP32 tap to BF16 at the load (lossless) and
``conv_put`` widens the BF16 value to FP32 at the store (exact). The conv math (FP32
weights, FP32 accumulation) is unchanged, so the outputs, state and window are bit
for bit the BF16-window run's. Both are explicit: CuTe converts a mismatched element
store implicitly or refuses it, so no load or store may rely on it. At a BF16 window
both emit exactly the original access (same PTX)."""

from cutlass import BFloat16, Float32


def conv_dtype_ok(element_type):
    """The conv window element types the TP kernels take."""
    return element_type in (BFloat16, Float32)


def conv_tap(page, *coord):
    """Tap ``coord`` of a conv window ``page`` as BF16."""
    assert conv_dtype_ok(page.element_type), page.element_type
    if page.element_type == BFloat16:
        return page[coord]
    return BFloat16(page[coord])


def conv_put(page, coord, value):
    """Store the BF16 ``value`` at ``coord`` of a conv window ``page``."""
    assert conv_dtype_ok(page.element_type), page.element_type
    if page.element_type == BFloat16:
        page[coord] = value
    else:
        page[coord] = page.element_type(value)


def check_conv_cache(conv, dim, dim_first, kernel):
    """Raise ``Unsupported`` unless ``conv`` is a conv window cache the TP kernels
    read correctly: BF16 or FP32, [slot, 3 * dim, taps] (the DS layout, ``dim_first``)
    or its transpose, each channel row's taps contiguous (DS) and every slot 16-byte
    aligned. CuTe takes a mismatched dtype or stride silently
    (kimi-k3-cute-silent-arg-mismatch), so this runs before any compile."""
    import torch

    from ...shapes import Unsupported

    def refuse(cause):
        raise Unsupported("kda", None, f"{kernel}: conv state {cause}")

    if conv.dtype not in (torch.bfloat16, torch.float32):
        refuse(f"dtype {conv.dtype} (BF16 or FP32 only)")
    if conv.dim() != 3:
        refuse(f"shape {tuple(conv.shape)} (not [slot, channels, taps])")
    channels = conv.shape[1] if dim_first else conv.shape[2]
    if channels != 3 * dim:
        refuse(f"shape {tuple(conv.shape)}: {channels} channels, not 3 x {dim}")
    if dim_first and (conv.stride(2) != 1 or conv.stride(1) != conv.shape[2]):
        refuse(f"strides {conv.stride()} (taps of a channel row not contiguous)")
    size = conv.element_size()
    aligned = conv.device.type == "meta" or conv.data_ptr() % 16 == 0
    if conv.stride(0) * size % 16 or not aligned:
        refuse(f"slot stride {conv.stride(0)} / base not 16-byte aligned")
