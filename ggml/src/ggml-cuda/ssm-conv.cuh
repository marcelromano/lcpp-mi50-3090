#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

#ifdef GGML_EXP_COLA
#define GGML_IKCS_MAX_SNAP 8
#define GGML_IKCS_MAX_T    8

// ik_llama.cpp-style conv-state step for decode (1 <= n_t <= 8, d_conv == 4):
//   concat(state, x) -> K x (view, cpy into the cache) -> ssm_conv -> silu    as ONE kernel.
// cat: the CONCAT node (never materialized), cpys: the CPY nodes of the K snapshots, act: the SILU node (output).
// The caller (ggml_cuda_ikcs_try_fuse) has validated shapes, strides, use counts and memory ranges.
void ggml_cuda_ikcs_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * cat, const ggml_tensor * const * cpys,
                           int n_snap, const ggml_tensor * ssm, const ggml_tensor * act);
#endif // GGML_EXP_COLA
