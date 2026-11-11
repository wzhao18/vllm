# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TensorRT-LLM's Blackwell CuTe DSL top-k kernels (vendored; see VENDORED.md).

vLLM uses the self-sampling GVR V2 engine, which ranks FP32, BF16 and FP16
logits natively (TensorRT-LLM PR #19285).
"""
