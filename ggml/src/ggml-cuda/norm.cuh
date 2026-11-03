#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#ifdef GGML_EXP_FUSE2
void ggml_cuda_op_l2_norm_x2(ggml_backend_cuda_context & ctx, ggml_tensor * dst_a, ggml_tensor * dst_b);
#endif // GGML_EXP_FUSE2

// : decode-time fusions around RMS_NORM+MUL (macros GGML_EXP_SK_ADD_RMS / GGML_EXP_SK_FFN_TAIL / GGML_EXP_SK_RMS_GATE)
#if defined(GGML_EXP_SK_ADD_RMS) || defined(GGML_EXP_SK_FFN_TAIL) || defined(GGML_EXP_SK_RMS_GATE)
#define GGML_EXP_SK_NORM_FUSE
#define GGML_EXP_SK_MAX_TOKENS 16

void ggml_cuda_op_sk_add_rms(ggml_backend_cuda_context & ctx, const ggml_tensor * add, const ggml_tensor * rms, const ggml_tensor * mul);

void ggml_cuda_op_sk_ffn_tail(ggml_backend_cuda_context & ctx, const ggml_tensor * sig, const ggml_tensor * mul,
        const ggml_tensor * add1, const ggml_tensor * add2, const ggml_tensor * rms, const ggml_tensor * mul2);

void ggml_cuda_op_sk_rms_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * mul,
        const ggml_tensor * silu, const ggml_tensor * mul2);
#endif // GGML_EXP_SK_*
