# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime-patched module globals for CuTe DSL kernels (int64.3 item 1).

``read_global_u64(name, default)`` reads a 64-bit ``llvm.mlir.global`` named ``name`` in the kernel's GPU module
(created once per module with the FIXED initializer ``default``; a volatile load, so a patch is seen by the next
launch). The host patches the runtime values after every load (guard_patch.py, from cute_cache). This module must not import
guard_config: it runs inside the trace, so its imports enter the compile-cache key; the traced code only holds the
name and the fixed default, and retuning a value then changes no key."""

from __future__ import annotations

from cutlass import Uint64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

def _module_body():
    op = ir.InsertionPoint.current.block.owner
    while op is not None and op.name not in ("gpu.module", "builtin.module"):
        op = op.parent
    if op is None:
        raise RuntimeError("guard_global: no enclosing module")
    return op.regions[0].blocks[0]


@dsl_user_op
def read_global_u64(name: str, default: int, *, loc=None, ip=None) -> Uint64:
    body = _module_body()
    if not any(o.name == "llvm.mlir.global" and f"@{name}" in str(o).split("(")[0] for o in body.operations):
        with ir.InsertionPoint.at_block_begin(body):
            llvm.GlobalOp(T.i64(), name, ir.Attribute.parse("#llvm.linkage<external>"),
                          value=ir.IntegerAttr.get(T.i64(), int(default)), addr_space=1)
    ptr = llvm.AddressOfOp(llvm.PointerType.get(1), name, loc=loc, ip=ip).result
    return Uint64(llvm.LoadOp(T.i64(), ptr, volatile_=True, loc=loc, ip=ip).result)
