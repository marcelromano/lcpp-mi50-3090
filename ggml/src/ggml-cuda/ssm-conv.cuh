#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

#ifdef GGML_EXP_COLA
#define GGML_IKCS_MAX_SNAP 8
#define GGML_IKCS_MAX_T    8

// rollback store of the conv state (cparams.n_rs_seq > 0, see build_conv_state_store).
//   SET_ROWS(slots, windows, slot_order) writes the Kw = min(n_t, K) overlapping 3-wide windows of
//   concat(state, x) into slots Kw-1 .. 0 of the conv cache ([TAG_RECURRENT_ROLLBACK_SPLITS]); the
//   fused kernel writes the same cells from its registers, so arange/scale/cast/views/set_rows cost nothing.
struct ggml_ikcs_rollback {
    float *       dst;     // cache view [3, K, C, n_s]
    int64_t       dst_nb1; // slot stride (elements)
    int64_t       dst_nb2; // channel stride (3 floats)
    int64_t       dst_nb3; // sequence stride
    int           K;       // n_rs_seq + 1 reserved slots
    int           Kw;      // windows written by this ubatch
};

// ik_llama.cpp-style conv-state step for decode (1 <= n_t <= 8, d_conv == 4):
//   concat(state, x) -> {K x (view, cpy into the cache) | SET_ROWS rollback store} -> ssm_conv -> silu
// as ONE kernel.
// cat: the CONCAT node (never materialized), cpys: the CPY nodes of the K snapshots (simple store),
// rb: the rollback store (nullptr on the simple one), act: the SILU node (output).
// The caller (ggml_cuda_ikcs_try_fuse) has validated shapes, strides, use counts and memory ranges.
void ggml_cuda_ikcs_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * cat, const ggml_tensor * const * cpys,
                           int n_snap, const ggml_tensor * ssm, const ggml_tensor * act,
                           const ggml_ikcs_rollback * rb = nullptr);
#endif // GGML_EXP_COLA
