#include "common.cuh"

#ifdef GGML_EXP_COLA
#define GGML_IKCS_MAX_T 8
#define GGML_IKCS_MAX_SNAP 8
void ggml_cuda_ikcs_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * cat,
        const ggml_tensor * const * cpys, int n_snap, const ggml_tensor * ssm,
        const ggml_tensor * act, bool tmp_out);
#endif

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);
