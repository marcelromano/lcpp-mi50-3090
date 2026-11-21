#pragma once

#include "common.cuh"

#ifdef GGML_EXP_COLA
// Decode-time "glue" fusions between the matmuls (macro GGML_EXP_COLA). Runtime mask, env GGML_COLA (default 255 = all on,
// 0 = all off): 1 gates (alpha/beta projections + softplus), 2 norm+q8_1, 4 conv-state chain, 8 gated norm+q8_1, 16 swiglu+q8_1,
// 32 sigmoid inside the gates kernel (without it the sigmoid node stays), 64 residual add+norm+q8_1.
//
// additions (port to sm_86):
//   GGML_COLA_GATES_BS   128|256 — launch geometry of the gates kernel (default: the mul_mat_vec_f mirror).
//   GGML_COLA_GATES_ANY  0 = the fork's rule (both projections must go through mul_mat_vec_f), 1 = always fuse,
//                        2 = fuse only where the original dispatch would NOT use mul_mat_vec_f (on sm_86 that is
//                        the tensor-core/cuBLAS path taken when n_tokens > 1).
//   GGML_COLA_LOG=1      prints `[cola] gates …` on every fused launch signature and `[mulmat] bf16 …` with the
//                        dispatch decision of every bf16 MUL_MAT with <= 8 tokens (proof of dispatch, F1d).

#define GGML_COLA_GATES  1
#define GGML_COLA_NORMQ  2
#define GGML_COLA_CONV   4
#define GGML_COLA_GNORM  8
#define GGML_COLA_SIGK   32
#define GGML_COLA_GLUQ   16
#define GGML_COLA_ADDN   64

int ggml_cuda_cola_mask();
// diagnostic logging on (env GGML_COLA_LOG=1)
bool ggml_cuda_cola_log();
// gates kernel dispatch policy (env GGML_COLA_GATES_ANY, 0..2)
int ggml_cuda_cola_gates_any();

// MUL_MAT(ssm_alpha) -> ADD(dt) -> SOFTPLUS -> MUL(a) ; MUL_MAT(ssm_beta) -> SIGMOID : one kernel (GDN decay gate + beta)
void ggml_cuda_cola_gdn_gates(ggml_backend_cuda_context & ctx,
        const ggml_tensor * mm_a, const ggml_tensor * mm_b, const ggml_tensor * dt, const ggml_tensor * a,
        const ggml_tensor * mul, const ggml_tensor * sig);
// is the (bf16, ncols, n_tokens) combination supported by the gates kernel
bool ggml_cuda_cola_gates_supported(const ggml_tensor * w, const ggml_tensor * y);


#endif
