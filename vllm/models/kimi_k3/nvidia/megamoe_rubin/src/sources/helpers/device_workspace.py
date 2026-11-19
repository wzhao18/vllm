"""Declarative GMEM workspace shared by host layout and device access."""

import dataclasses
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64

from .utils import (
    ceil_div,
    cosize_from_shape_stride_tuples,
    flatten_shape_stride,
    is_nested_shape,
    is_power_of_two,
    ordered_stride,
    row_major_stride,
    round_up,
    validate_static_integer_tuple,
)


BufferResetAttr = Literal["data", "zero_on_first_allocate", "tail_reset"]
BufferSpace = Literal["local", "shared"]

_reset_order = {"tail_reset": 0, "zero_on_first_allocate": 1, "data": 2}


def _footprint_from_shape_stride(shape: Tuple, stride: Tuple) -> int:
    """Elements a region must own, as opposed to the ones its layout can address."""
    leaf_pairs = flatten_shape_stride(shape, stride) if shape else []
    claimed = max((size * step for size, step in leaf_pairs), default=1)
    # Layouts whose leaves overlap can address past what any single leaf tiles,
    # so the cosize stays a floor.
    return int(max(claimed, cosize_from_shape_stride_tuples(shape, stride)))


@dataclasses.dataclass(frozen=True)
class DeviceRegion:
    """One typed region in a local or symmetric GMEM workspace."""

    name: str
    dtype: Type[cutlass.Numeric]
    shape: Tuple
    buffer_space: BufferSpace
    stride: Optional[Tuple] = None
    mem_order: Optional[Tuple[int, ...]] = None
    byte_alignment: int = 16
    reset: BufferResetAttr = "data"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A workspace region needs a non-empty name.")
        if self.buffer_space not in ("local", "shared"):
            raise ValueError(f"Invalid buffer space {self.buffer_space!r}.")
        if self.reset not in _reset_order:
            raise ValueError(f"Invalid reset policy {self.reset!r}.")
        if not is_power_of_two(self.byte_alignment):
            raise ValueError(f"Region {self.name!r} alignment must be a positive power of two.")
        validate_static_integer_tuple(self.shape, field_name=f"{self.name}.shape")
        if self.stride is not None and self.mem_order is not None:
            raise ValueError(f"Region {self.name!r} accepts either stride or mem_order, not both.")
        if self.stride is None and self.mem_order is None:
            if is_nested_shape(self.shape) or len(self.shape) != 1:
                raise ValueError(f"Region {self.name!r} needs an explicit layout.")
            object.__setattr__(self, "stride", row_major_stride(self.shape))
        if is_nested_shape(self.shape) and self.mem_order is not None:
            raise ValueError(f"Nested region {self.name!r} needs an explicit stride.")
        if self.stride is not None:
            validate_static_integer_tuple(self.stride, field_name=f"{self.name}.stride")
        if self.mem_order is not None:
            validate_static_integer_tuple(self.mem_order, field_name=f"{self.name}.mem_order")
            expected = tuple(range(len(self.shape)))
            if tuple(sorted(self.mem_order)) != expected:
                raise ValueError(
                    f"Region {self.name!r} mem_order must be a permutation of {expected}, got {self.mem_order}."
                )
            resolved_stride, _ = ordered_stride(self.shape, self.mem_order)
            object.__setattr__(self, "stride", resolved_stride)


class DeviceWorkspace:
    """Single source of truth for GMEM region layout and device pointer derivation."""

    def __init__(self) -> None:
        self._registered: Dict[BufferSpace, List[DeviceRegion]] = {"local": [], "shared": []}
        self._region_by_name: Dict[str, DeviceRegion] = {}
        self._offset: Dict[str, int] = {}
        self._stride: Dict[str, Tuple] = {}
        self._cosize: Dict[str, int] = {}
        self._nbytes: Dict[str, int] = {}
        self._byte_alignment: Dict[str, int] = {}
        self._total: Dict[BufferSpace, int] = {"local": 0, "shared": 0}
        self._zero_leading: Dict[BufferSpace, int] = {"local": 0, "shared": 0}
        self._tail_leading: Dict[BufferSpace, int] = {"local": 0, "shared": 0}
        self._base: Dict[BufferSpace, Any] = {"local": None, "shared": None}
        # Parity double buffering of the tail-reset prefix (see enable_tail_parity).
        self._tail_parity_region: Optional[str] = None
        self._tail_copy_bytes: Dict[BufferSpace, int] = {"local": 0, "shared": 0}
        self._tail_parity = None
        # Optional data placement (set_data_layout): per-space byte offset of the first data region, and data
        # regions placed first with a reserved footprint.
        self._data_start: Optional[Dict[BufferSpace, int]] = None
        self._leading_data: Tuple[Tuple[str, int], ...] = ()
        self._finalized = False

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "DeviceWorkspace":
        return self

    @property
    def finalized(self) -> bool:
        return self._finalized

    def register(
        self,
        name: str,
        dtype: Type[cutlass.Numeric],
        shape: Tuple,
        *,
        buffer_space: BufferSpace,
        stride: Optional[Tuple] = None,
        mem_order: Optional[Tuple[int, ...]] = None,
        byte_alignment: int = 16,
        reset: BufferResetAttr = "data",
    ) -> None:
        if self._finalized:
            raise RuntimeError("Cannot register a region after finalize().")
        if name in self._region_by_name or any(
            region.name == name for regions in self._registered.values() for region in regions
        ):
            raise ValueError(f"Duplicate workspace region {name!r}.")
        self._registered[buffer_space].append(
            DeviceRegion(
                name=name,
                dtype=dtype,
                shape=shape,
                buffer_space=buffer_space,
                stride=stride,
                mem_order=mem_order,
                byte_alignment=byte_alignment,
                reset=reset,
            )
        )

    def enable_tail_parity(self, parity_region: str) -> None:
        """Double-buffer every ``tail_reset`` region by call parity.

        Call ``c`` uses copy ``parity(c)`` and clears copy ``1 - parity(c)``, which
        nobody reads or writes until call ``c + 1``. The parity is bit 0 of the
        ``Int32`` region ``parity_region`` (a counter advanced once per call, and
        never itself ``tail_reset``), loaded once when device members are assigned
        with ``load_tail_parity=True``. A tail-reset region pointer requested
        without a loaded parity is a trace-time error.
        """
        if self._finalized:
            raise RuntimeError("enable_tail_parity() must precede finalize().")
        self._tail_parity_region = parity_region

    def set_data_layout(
        self, data_start: Tuple[int, int], leading_data: Tuple[Tuple[str, int], ...] = ()
    ) -> None:
        """Place the data regions behind a fixed offset, some of them first with a reserved footprint.

        ``data_start`` = (local, shared) byte offset of the first ``data`` region; the reset prefix (tail-reset
        copies and zero-on-first-allocate state) must fit below it. ``leading_data`` = ((region name, reserved
        bytes), ...): these data regions come first, in this order, each occupying at least the reserved bytes.
        Several kernels can then share one allocation: each owns a private reset prefix at its own base, while
        their data regions start at one common address and the leading regions keep a common placement.
        """
        if self._finalized:
            raise RuntimeError("set_data_layout() must precede finalize().")
        local_start, shared_start = (int(v) for v in data_start)
        if local_start < 0 or shared_start < 0 or local_start % 128 or shared_start % 128:
            raise ValueError("data_start offsets must be non-negative multiples of 128 bytes.")
        self._data_start = {"local": local_start, "shared": shared_start}
        self._leading_data = tuple((str(name), int(nbytes)) for name, nbytes in leading_data)

    @property
    def tail_parity_enabled(self) -> bool:
        return self._tail_parity_region is not None

    def finalize(self) -> None:
        if self._finalized:
            raise RuntimeError("DeviceWorkspace.finalize() may only be called once.")
        if self._tail_parity_region is not None:
            parity_region = self._region_by_name.get(self._tail_parity_region) or next(
                (
                    region
                    for regions in self._registered.values()
                    for region in regions
                    if region.name == self._tail_parity_region
                ),
                None,
            )
            if parity_region is None or parity_region.reset == "tail_reset" or parity_region.dtype.width != 32:
                raise ValueError(
                    f"Tail parity region {self._tail_parity_region!r} must be a registered, "
                    "non-tail-reset Int32 region."
                )
        for buffer_space in ("local", "shared"):
            ordered = sorted(self._registered[buffer_space], key=lambda region: _reset_order[region.reset])
            leading = {name: (rank, nbytes) for rank, (name, nbytes) in enumerate(self._leading_data)}
            state = [region for region in ordered if region.reset != "data"]
            data = [region for region in ordered if region.reset == "data"]
            data = sorted(
                data, key=lambda region: leading[region.name][0] if region.name in leading else len(leading)
            )
            ordered = state + data
            cursor = 0
            for region_index, region in enumerate(ordered):
                if (
                    self._data_start is not None
                    and region.reset == "data"
                    and (region_index == 0 or ordered[region_index - 1].reset != "data")
                ):
                    if cursor > self._data_start[buffer_space]:
                        raise ValueError(
                            f"{buffer_space} reset prefix ({cursor} B) exceeds the data offset "
                            f"{self._data_start[buffer_space]} B."
                        )
                    cursor = max(cursor, self._data_start[buffer_space])
                stride = region.stride
                if stride is None:
                    raise RuntimeError(f"Region {region.name!r} stride was not resolved.")
                cosize = cosize_from_shape_stride_tuples(region.shape, stride)
                nbytes = (_footprint_from_shape_stride(region.shape, stride) * int(region.dtype.width) + 7) // 8
                if region.reset == "tail_reset" and (
                    region_index == 0 or ordered[region_index - 1].reset != "tail_reset"
                ):
                    cursor = round_up(cursor, 16)
                byte_alignment = region.byte_alignment
                cursor = round_up(cursor, byte_alignment)
                self._region_by_name[region.name] = region
                self._offset[region.name] = cursor
                self._stride[region.name] = stride
                self._cosize[region.name] = cosize
                self._nbytes[region.name] = nbytes
                self._byte_alignment[region.name] = byte_alignment
                cursor += max(nbytes, leading[region.name][1]) if region.name in leading else nbytes
                is_last_tail_reset_region = region.reset == "tail_reset" and (
                    region_index + 1 == len(ordered) or ordered[region_index + 1].reset != "tail_reset"
                )
                if is_last_tail_reset_region:
                    cursor = round_up(cursor, 16)
                    self._tail_leading[buffer_space] = cursor
                    if self._tail_parity_region is not None:
                        # Second copy right behind the first; its stride keeps every
                        # tail region's alignment.
                        copy_alignment = max(
                            [16] + [r.byte_alignment for r in ordered if r.reset == "tail_reset"]
                        )
                        copy_bytes = round_up(cursor, copy_alignment)
                        self._tail_copy_bytes[buffer_space] = copy_bytes
                        cursor = copy_bytes * 2
                if region.reset != "data":
                    self._zero_leading[buffer_space] = cursor
            self._total[buffer_space] = round_up(cursor, 16)
        self._finalized = True

    def regions(self, buffer_space: BufferSpace) -> Tuple[DeviceRegion, ...]:
        return tuple(self._registered[buffer_space])

    def region(self, name: str) -> DeviceRegion:
        self._require_finalized()
        return self._region_by_name[name]

    def offset(self, name: str) -> int:
        self._require_finalized()
        return self._offset[name]

    def stride(self, name: str) -> Tuple:
        self._require_finalized()
        return self._stride[name]

    def cosize(self, name: str) -> int:
        self._require_finalized()
        return self._cosize[name]

    def nbytes(self, name: str) -> int:
        self._require_finalized()
        return self._nbytes[name]

    def byte_alignment(self, name: str) -> int:
        self._require_finalized()
        return self._byte_alignment[name]

    def total_bytes(self, buffer_space: BufferSpace) -> int:
        self._require_finalized()
        return self._total[buffer_space]

    @property
    def local_and_shared_bytes(self) -> Tuple[int, int]:
        self._require_finalized()
        return self._total["local"], self._total["shared"]

    def zero_on_allocate_bytes(self, buffer_space: BufferSpace) -> int:
        self._require_finalized()
        return self._zero_leading[buffer_space]

    def tail_reset_bytes(self, buffer_space: BufferSpace) -> int:
        self._require_finalized()
        return self._tail_leading[buffer_space]

    @property
    def require_zero_workspace_leading_bytes(self) -> Tuple[int, int]:
        self._require_finalized()
        return self._zero_leading["local"], self._zero_leading["shared"]

    @cute.jit
    def assign_device_members(
        self,
        local_workspace: cute.Pointer,
        shared_workspace: Optional[cute.Pointer] = None,
        *,
        load_tail_parity: bool = False,
    ) -> None:
        """Bind the workspace bases; device code passes load_tail_parity=True."""
        self._base["local"] = local_workspace
        self._base["shared"] = shared_workspace
        self._tail_parity = None
        if cutlass.const_expr(load_tail_parity and self._tail_parity_region is not None):
            counter = cute.arch.load(self.ptr(self._tail_parity_region), Int32, sem="relaxed", scope="gpu")
            self._tail_parity = counter & Int32(1)

    def remove_device_members(self) -> None:
        self._base = {"local": None, "shared": None}
        self._tail_parity = None

    @cute.jit
    def _tail_copy_offset(self, buffer_space: BufferSpace, copy_select) -> Int64:
        return Int64(copy_select) * Int64(self._tail_copy_bytes[buffer_space])

    @cute.jit
    def ptr(self, name: str) -> cute.Pointer:
        region = self._region_by_name[name]
        base = self._base[region.buffer_space]
        address = base.toint() + Int64(self._offset[name])
        if cutlass.const_expr(self._tail_parity_region is not None and region.reset == "tail_reset"):
            if cutlass.const_expr(self._tail_parity is None):
                raise RuntimeError(
                    f"Tail-reset region {name!r} needs the call parity: assign device members with "
                    "load_tail_parity=True in device code."
                )
            address = address + self._tail_copy_offset(region.buffer_space, self._tail_parity)
        return cute.make_ptr(region.dtype, address, AddressSpace.gmem, assumed_align=self._byte_alignment[name])

    @cute.jit
    def tensor(self, name: str) -> cute.Tensor:
        region = self._region_by_name[name]
        return cute.make_tensor(self.ptr(name), cute.make_layout(region.shape, stride=self._stride[name]))

    @cute.jit
    def reset_tail(self, tid: Int32, total_threads: int) -> None:
        self.reset_tail_space("local", tid, total_threads)
        self.reset_tail_space("shared", tid, total_threads)

    @cute.jit
    def reset_tail_space(self, buffer_space: BufferSpace, tid: Int32, total_threads: int) -> None:
        """Clear the tail-reset prefix; with tail parity, the copy the next call uses."""
        num_vectors = self._tail_leading[buffer_space] // 16
        if cutlass.const_expr(num_vectors > 0):
            reset_base = self._base[buffer_space].toint()
            if cutlass.const_expr(self._tail_parity_region is not None):
                if cutlass.const_expr(self._tail_parity is None):
                    raise RuntimeError("reset_tail_space needs the call parity (load_tail_parity=True).")
                reset_base = reset_base + self._tail_copy_offset(buffer_space, Int32(1) - self._tail_parity)
            vectors = cute.make_tensor(
                cute.make_ptr(Int32, reset_base, AddressSpace.gmem, assumed_align=16),
                cute.make_layout((num_vectors, 4), stride=(4, 1)),
            )
            zero = cute.make_rmem_tensor((4,), Int32)
            for element in cutlass.range_constexpr(4):
                zero[element] = Int32(0)
            store_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=128)
            reset_round_count = ceil_div(num_vectors, total_threads)
            for reset_round in cutlass.range_constexpr(reset_round_count):
                vector_index = Int32(reset_round * total_threads) + tid
                if vector_index < Int32(num_vectors):
                    cute.copy(store_atom, zero, vectors[vector_index, None])

    def _require_finalized(self) -> None:
        if not self._finalized:
            raise RuntimeError("DeviceWorkspace must be finalized first.")
