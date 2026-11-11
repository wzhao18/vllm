// Kimi-K3: torch registration for the FlashKDA K1/K2-split build (adapted from
// FlashKDA csrc/torch_api.cpp @ vllm-project/FlashKDA 17a037d, MIT, see
// LICENSE.FlashKDA). Namespace k3_flashkda_split; fwd takes `mode` (1 = K1
// prepare only, 2 = K2 recurrence only on a workspace K1 already filled, 3 =
// both, as FlashKDA; + 4 = fp32 state crosses K2's boundary exactly instead of
// rounding to bf16 as FlashKDA, for the segment-parallel prefill's partials).
#include <Python.h>

#include <torch/csrc/stable/library.h>

#include "flash_kda.h"

STABLE_TORCH_LIBRARY(k3_flashkda_split, m) {
  m.def("get_workspace_size(int T_total, int H, int N=1) -> int");
  m.def(
      "fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
      "float scale, Tensor(a!) out, Tensor(c!) workspace, Tensor A_log, "
      "Tensor dt_bias, float lower_bound, Tensor? initial_state=None, "
      "Tensor(b!)? final_state=None, Tensor? cu_seqlens=None, "
      "Tensor(d!)? checkpoint_state=None, "
      "Tensor? checkpoint_offsets=None, int mode=3) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(k3_flashkda_split, CUDA, m) {
  m.impl("fwd", TORCH_BOX(&fwd));
}

STABLE_TORCH_LIBRARY_IMPL(k3_flashkda_split, CompositeExplicitAutograd, m) {
  m.impl("get_workspace_size", TORCH_BOX(&get_workspace_size));
}

static struct PyModuleDef k3_flashkda_split_module = {
    PyModuleDef_HEAD_INIT, "k3_flashkda_split", nullptr, -1, nullptr};

PyMODINIT_FUNC PyInit_k3_flashkda_split(void) {
  return PyModule_Create(&k3_flashkda_split_module);
}
