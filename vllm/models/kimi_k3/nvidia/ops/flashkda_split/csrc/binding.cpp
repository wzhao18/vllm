// Kimi-K3: torch registration for the FlashKDA K1/K2-split build (adapted from
// FlashKDA csrc/torch_api.cpp @ vllm-project/FlashKDA 17a037d, MIT, see
// LICENSE.FlashKDA). Namespace k3_flashkda_split; fwd takes `mode` (1 = K1
// prepare only, 2 = K2 recurrence only on a workspace K1 already filled, 3 =
// both, as FlashKDA; + 4 = fp32 state crosses K2's boundary exactly instead of
// rounding to bf16 as FlashKDA, for the segment-parallel prefill's partials).
#include <Python.h>

#include <torch/csrc/stable/library.h>

#include "flash_kda.h"

// k31_spin_guard.cuh: the rank printed by a spin-timeout line, one setter per TU
void k31sg_set_rank_prep64_tc(int, long long);
void k31sg_set_rank_o_norm_mx(int, long long);
void k31sg_set_rank_fwd_launch(int, long long);
void k31sg_set_rank_k64_tc(int, long long);
void k31sg_set_rank_kc64_tc(int, long long);
static void set_spin_rank(int64_t rank, int64_t diag) {
  const int r = static_cast<int>(rank);
  k31sg_set_rank_prep64_tc(r, diag);
  k31sg_set_rank_o_norm_mx(r, diag);
  k31sg_set_rank_fwd_launch(r, diag);
  k31sg_set_rank_k64_tc(r, diag);
  k31sg_set_rank_kc64_tc(r, diag);
}

STABLE_TORCH_LIBRARY(k3_flashkda_split, m) {
  m.def("get_workspace_size(int T_total, int H, int N=1) -> int");
  m.def("set_spin_rank(int rank, int diag=0) -> ()");
  m.def(
      "fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
      "float scale, Tensor(a!) out, Tensor(c!) workspace, Tensor A_log, "
      "Tensor dt_bias, float lower_bound, Tensor? initial_state=None, "
      "Tensor(b!)? final_state=None, Tensor? cu_seqlens=None, "
      "Tensor(d!)? checkpoint_state=None, "
      "Tensor? checkpoint_offsets=None, int mode=3) -> ()");
  m.def(
      "o_norm_mxfp8(Tensor x, Tensor g, Tensor weight, float eps, Tensor(a!) q, "
      "Tensor(b!) sf) -> ()");
  m.def(
      "k64_rec(Tensor kdq, Tensor kr, Tensor inv, Tensor mqk, "
      "Tensor gt, Tensor beta, Tensor v, Tensor init_state, "
      "Tensor(a!) final_state, Tensor(b!) out, Tensor cu_seqlens, int flags, "
      "Tensor(c!) dbg) -> ()");
  m.def(
      "kc64_rec(Tensor wq, Tensor u, Tensor kr, Tensor mqk, Tensor gt, "
      "Tensor init_state, Tensor(a!) final_state, Tensor(b!) out, "
      "Tensor cu_seqlens, int flags, Tensor(c!) dbg) -> ()");
  m.def(
      "prep64(Tensor q, Tensor k, Tensor g, Tensor beta, Tensor a_log, "
      "Tensor dt_bias, float lower_bound, float scale, Tensor cu_seqlens, "
      "Tensor(a!) kdq, Tensor(b!) kr, Tensor(c!) inv, Tensor(d!) mqk, "
      "Tensor(e!) gt, Tensor(f!) beta_out) -> ()");
  m.def(
      "prep64_tc(Tensor x, Tensor conv_w, Tensor(a!) conv_state, Tensor cache_indices, "
      "Tensor has_init, int null_block_id, Tensor g, Tensor beta, Tensor a_log, "
      "Tensor dt_bias, float lower_bound, float scale, Tensor cu_seqlens, "
      "Tensor(b!) kdq, Tensor(c!) kr, Tensor(d!) inv, Tensor(e!) mqk, Tensor(f!) gt, "
      "Tensor(g!) v_img, Tensor(h!) beta_out, Tensor(i!) dbg) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(k3_flashkda_split, CUDA, m) {
  m.impl("fwd", TORCH_BOX(&fwd));
  m.impl("o_norm_mxfp8", TORCH_BOX(&o_norm_mxfp8));
  m.impl("k64_rec", TORCH_BOX(&k64_rec));
  m.impl("kc64_rec", TORCH_BOX(&kc64_rec));
  m.impl("prep64", TORCH_BOX(&prep64));
  m.impl("prep64_tc", TORCH_BOX(&prep64_tc));
}

STABLE_TORCH_LIBRARY_IMPL(k3_flashkda_split, CompositeExplicitAutograd, m) {
  m.impl("get_workspace_size", TORCH_BOX(&get_workspace_size));
  m.impl("set_spin_rank", TORCH_BOX(&set_spin_rank));
}

static struct PyModuleDef k3_flashkda_split_module = {
    PyModuleDef_HEAD_INIT, "k3_flashkda_split", nullptr, -1, nullptr};

PyMODINIT_FUNC PyInit_k3_flashkda_split(void) {
  return PyModule_Create(&k3_flashkda_split_module);
}
