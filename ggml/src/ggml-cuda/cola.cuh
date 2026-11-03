#pragma once

#include "common.cuh"

#ifdef GGML_EXP_COLA
// Decode-time "glue" fusions between the matmuls (macro GGML_EXP_COLA). Runtime mask, env GGML_COLA (default 255 = all on,
// 0 = all off): 1 gates (alpha/beta projections + softplus), 2 norm+q8_1, 4 conv-state chain, 8 gated norm+q8_1, 16 swiglu+q8_1,
// 32 sigmoid inside the gates kernel (without it the sigmoid node stays), 64 residual add+norm+q8_1.

#define GGML_COLA_GATES  1
#define GGML_COLA_NORMQ  2
#define GGML_COLA_CONV   4
#define GGML_COLA_GNORM  8
#define GGML_COLA_SIGK   32
#define GGML_COLA_GLUQ   16
#define GGML_COLA_ADDN   64

int ggml_cuda_cola_mask();

// MUL_MAT(ssm_alpha) -> ADD(dt) -> SOFTPLUS -> MUL(a) ; MUL_MAT(ssm_beta) -> SIGMOID : one kernel (GDN decay gate + beta)
void ggml_cuda_cola_gdn_gates(ggml_backend_cuda_context & ctx,
        const ggml_tensor * mm_a, const ggml_tensor * mm_b, const ggml_tensor * dt, const ggml_tensor * a,
        const ggml_tensor * mul, const ggml_tensor * sig);
// is the (bf16, ncols, n_tokens) combination supported by the gates kernel
bool ggml_cuda_cola_gates_supported(const ggml_tensor * w, const ggml_tensor * y);

// RMS_NORM -> MUL(w) + quantize_q8_1 of the result: writes the f32 output and the q8_1 blocks (kept in ctx.cola_q8_*)
bool ggml_cuda_cola_rms_q8_supported(const ggml_tensor * rms, const ggml_tensor * mul);
void ggml_cuda_cola_rms_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * mul);
// ADD(a, b) -> RMS_NORM -> MUL(w) + quantize_q8_1: also writes the sum (the residual stream)
void ggml_cuda_cola_add_rms_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * add, const ggml_tensor * rms, const ggml_tensor * mul);

// GDN gated norm RMS_NORM(x)*w * silu(z) + quantize_q8_1 of the result (input of ssm_out)
bool ggml_cuda_cola_gnorm_q8_supported(const ggml_tensor * rms, const ggml_tensor * w, const ggml_tensor * z, const ggml_tensor * out);
void ggml_cuda_cola_gnorm_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * w, const ggml_tensor * z,
        const ggml_tensor * out, const ggml_tensor * consumer_src1);

// SWIGLU (split gate/up) + quantize_q8_1 of the result (input of ffn_down)
bool ggml_cuda_cola_swiglu_q8_supported(const ggml_tensor * glu);
void ggml_cuda_cola_swiglu_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * glu);

// pre-quantized activations produced by ggml_cuda_cola_rms_q8 (nullptr when src1 is not the pending tensor)
const char * ggml_cuda_cola_pending_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * src1);

#endif // GGML_EXP_COLA
