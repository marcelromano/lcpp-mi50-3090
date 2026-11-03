#pragma once

// Decode/MTP-verify FlashAttention that reads the quantized KV cache directly (K q8_0, V q4_0, D=256)
// and shares every KV tile between all GQA heads x columns of one KV head.
//
// Block = one KV head x one contiguous chunk of KV tiles (split-K), 4 warps.
// Rows of the block = ncols*gqa (<= 32), padded to 16*MT for the tensor cores.
//   KQ : mma m16n8k32 s8 (Q quantized per 32-block to int8 in the kernel, K q8_0 qs used as-is),
//        int32 -> float with the q8_0 scales of Q and K.
//   PV : V tile dequantized once per block into shared f16 (transposed), mma m16n8k16 f16 -> f32.
// Partial (O, max, sum) per split go to a pool buffer; a combine kernel writes dst.

#include "common.cuh"

#if defined(GGML_EXP_CUDA_FA_KVQ_GQA)

#define KVQ_D         256
#define KVQ_T         32                          // KV positions per tile
#define KVQ_NTHREADS  128
#define KVQ_KROW      272                         // bytes of one q8_0 row of 256
#define KVQ_VROW      144                         // bytes of one q4_0 row of 256
#define KVQ_NCH_K     (KVQ_T*KVQ_KROW/16)         // 544 x 16 B
#define KVQ_NCH_V     (KVQ_T*KVQ_VROW/16)         // 288 x 16 B
#define KVQ_NCH       (KVQ_NCH_K + KVQ_NCH_V)     // 832
#define KVQ_NPF       ((KVQ_NCH + KVQ_NTHREADS - 1)/KVQ_NTHREADS) // 7
#define KVQ_VT_STRIDE (KVQ_T + 8)                 // halves per d row of the transposed V tile
#define KVQ_P_STRIDE  (KVQ_T + 8)                 // halves per row of P
#define KVQ_S_STRIDE  (KVQ_T + 1)                 // floats per row of S
#define KVQ_RMAX      32                          // max rows (ncols*gqa) per block

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
#define KVQ_AVAILABLE
#endif

static __device__ __forceinline__ void kvq_mma_s8(int (&c)[4], const int (&a)[4], const int b0, const int b1) {
#if defined(KVQ_AVAILABLE) && defined(__CUDA_ARCH__)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED(c); GGML_UNUSED(a); GGML_UNUSED(b0); GGML_UNUSED(b1);
#endif
}

static __device__ __forceinline__ void kvq_mma_f16(float (&c)[4], const uint32_t (&a)[4], const uint32_t b0, const uint32_t b1) {
#if defined(KVQ_AVAILABLE) && defined(__CUDA_ARCH__)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED(c); GGML_UNUSED(a); GGML_UNUSED(b0); GGML_UNUSED(b1);
#endif
}

// 4 consecutive bytes at a 2-byte aligned shared address
static __device__ __forceinline__ int kvq_ld_u16x2(const char * p) {
    const uint16_t * q = (const uint16_t *) p;
    return (int) ((uint32_t) q[0] | ((uint32_t) q[1] << 16));
}

template <int MT> // MT = number of 16-row m-tiles (1: rows <= 16, 2: rows <= 32)
__launch_bounds__(KVQ_NTHREADS, 2)
static __global__ void flash_attn_kvq_gqa(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const half * __restrict__ mask, float * __restrict__ part, float2 * __restrict__ meta,
        const float scale, const int ne11, const int ncols, const int gqa,
        const int64_t nbq1, const int64_t nbq2, const int64_t nbk1, const int64_t nbk2,
        const int64_t nbv1, const int64_t nbv2, const int64_t nbm1, const int tiles_per_split) {
#if defined(KVQ_AVAILABLE)
    constexpr int R    = 16*MT;
    constexpr int NT_W = MT == 1 ? 1 : 2; // KQ n-tiles (8 positions) per warp

    const int h      = blockIdx.y;
    const int split  = blockIdx.x;
    const int tid    = threadIdx.x;
    const int lane   = tid & 31;
    const int w      = tid >> 5;
    const int g      = lane >> 2;
    const int t      = lane & 3;
    const int nrows  = ncols*gqa;
    const int ntiles = ne11 / KVQ_T;
    const int tile0  = split*tiles_per_split;
    const int tile1  = min(ntiles, tile0 + tiles_per_split);

    __shared__ __align__(16) char  sK[KVQ_T*KVQ_KROW];
    __shared__ __align__(16) char  sV[KVQ_T*KVQ_VROW];
    __shared__ __align__(16) half  sVt[KVQ_D*KVQ_VT_STRIDE]; // also holds int8 Q at the start
    __shared__ __align__(16) float sS[R*KVQ_S_STRIDE];
    __shared__ __align__(16) half  sP[R*KVQ_P_STRIDE];
    __shared__ float sdQ[R*8];
    __shared__ float sAlpha[R];

    int8_t * sQ = (int8_t *) sVt;
    static_assert(R*KVQ_D <= KVQ_D*KVQ_VT_STRIDE*2, "sQ alias");

    // ---- Q: f32 -> int8 per 32-block, scale folded in (rows r = j*gqa + i) ----
    for (int idx = tid; idx < R*8; idx += KVQ_NTHREADS) {
        const int r = idx >> 3;
        const int b = idx & 7;
        float x[32];
        float amax = 0.0f;
        if (r < nrows) {
            const int j = r / gqa;
            const int i = r - j*gqa;
            const float4 * q4 = (const float4 *) (Q + j*nbq1 + (int64_t)(h*gqa + i)*nbq2) + b*8;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float4 v = q4[k];
                x[4*k+0] = v.x; x[4*k+1] = v.y; x[4*k+2] = v.z; x[4*k+3] = v.w;
            }
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                amax = fmaxf(amax, fabsf(x[k]));
            }
        } else {
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                x[k] = 0.0f;
            }
        }
        const float d  = amax / 127.0f;
        const float id = amax > 0.0f ? 127.0f/amax : 0.0f;
        int * dstq = (int *) (sQ + r*KVQ_D + b*32);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int q0 = __float2int_rn(x[4*k+0]*id);
            const int q1 = __float2int_rn(x[4*k+1]*id);
            const int q2 = __float2int_rn(x[4*k+2]*id);
            const int q3 = __float2int_rn(x[4*k+3]*id);
            dstq[k] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
        }
        sdQ[r*8 + b] = d*scale;
    }
    __syncthreads();

    const int mi = MT == 1 ? 0 : (w & 1);
    int   aQ[8][4];
    float dq0[8], dq1[8];
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        const int8_t * r0 = sQ + (mi*16 + g    )*KVQ_D + b*32 + 4*t;
        const int8_t * r1 = sQ + (mi*16 + g + 8)*KVQ_D + b*32 + 4*t;
        aQ[b][0] = *(const int *) (r0);
        aQ[b][1] = *(const int *) (r1);
        aQ[b][2] = *(const int *) (r0 + 16);
        aQ[b][3] = *(const int *) (r1 + 16);
        dq0[b] = sdQ[(mi*16 + g    )*8 + b];
        dq1[b] = sdQ[(mi*16 + g + 8)*8 + b];
    }
    __syncthreads(); // sQ aliases sVt

    float O[MT][8][4];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                O[m][n][k] = 0.0f;
            }
        }
    }

    // softmax ownership: 4 threads per row, 8 positions each
    const int rs = tid >> 2;
    const int qs = tid & 3;
    float M_run = -INFINITY;
    float l_run = 0.0f;

    const char * Kh = K + (int64_t) h*nbk2;
    const char * Vh = V + (int64_t) h*nbv2;

    uint4 pf[KVQ_NPF];
    auto issue = [&](const int tile) {
#pragma unroll
        for (int k = 0; k < KVQ_NPF; ++k) {
            const int c = tid + k*KVQ_NTHREADS;
            if (c < KVQ_NCH_K) {
                const int pos = c / 17;
                const int off = c - pos*17;
                pf[k] = *(const uint4 *) (Kh + (int64_t)(tile*KVQ_T + pos)*nbk1 + off*16);
            } else if (c < KVQ_NCH) {
                const int c2  = c - KVQ_NCH_K;
                const int pos = c2 / 9;
                const int off = c2 - pos*9;
                pf[k] = *(const uint4 *) (Vh + (int64_t)(tile*KVQ_T + pos)*nbv1 + off*16);
            }
        }
    };
    auto store = [&]() {
#pragma unroll
        for (int k = 0; k < KVQ_NPF; ++k) {
            const int c = tid + k*KVQ_NTHREADS;
            if (c < KVQ_NCH_K) {
                *(uint4 *) (sK + c*16) = pf[k];
            } else if (c < KVQ_NCH) {
                *(uint4 *) (sV + (c - KVQ_NCH_K)*16) = pf[k];
            }
        }
    };

    if (tile0 < tile1) {
        issue(tile0);
        store();
    }
    __syncthreads();

    for (int tile = tile0; tile < tile1; ++tile) {
        if (tile + 1 < tile1) {
            issue(tile + 1);
        }

        // ---- V q4_0 -> f16, transposed [d][pos]; thread = (pos pair, 32-block) ----
        {
            const int pp = tid & 15;
            const int b  = tid >> 4;
            const char * b0p = sV + (2*pp    )*KVQ_VROW + b*18;
            const char * b1p = sV + (2*pp + 1)*KVQ_VROW + b*18;
            const float d0 = __half2float(*(const half *) b0p);
            const float d1 = __half2float(*(const half *) b1p);
            const uint16_t * q0 = (const uint16_t *) (b0p + 2);
            const uint16_t * q1 = (const uint16_t *) (b1p + 2);
            half2 * col = (half2 *) (sVt + (b*32)*KVQ_VT_STRIDE + 2*pp);
#pragma unroll
            for (int k2 = 0; k2 < 8; ++k2) {
                const uint32_t x0 = q0[k2];
                const uint32_t x1 = q1[k2];
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int k = 2*k2 + e;
                    const int y0 = (x0 >> (8*e)) & 0xFF;
                    const int y1 = (x1 >> (8*e)) & 0xFF;
                    col[(k     )*(KVQ_VT_STRIDE/2)] = __floats2half2_rn(((y0 & 0xF) - 8)*d0, ((y1 & 0xF) - 8)*d1);
                    col[(k + 16)*(KVQ_VT_STRIDE/2)] = __floats2half2_rn(((y0 >>  4) - 8)*d0, ((y1 >>  4) - 8)*d1);
                }
            }
        }

        // ---- S = Q K^T (int8 tensor cores) ----
#pragma unroll
        for (int nn = 0; nn < NT_W; ++nn) {
            const int nt = MT == 1 ? w : 2*(w >> 1) + nn;
            const int pb = nt*8;
            float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const char * kr  = sK + (pb + g)*KVQ_KROW;
            const char * kd0 = sK + (pb + 2*t    )*KVQ_KROW;
            const char * kd1 = sK + (pb + 2*t + 1)*KVQ_KROW;
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                int c[4] = {0, 0, 0, 0};
                const int b0 = kvq_ld_u16x2(kr + b*34 + 2 + 4*t);
                const int b1 = kvq_ld_u16x2(kr + b*34 + 2 + 16 + 4*t);
                kvq_mma_s8(c, aQ[b], b0, b1);
                const float dk0 = __half2float(*(const half *) (kd0 + b*34));
                const float dk1 = __half2float(*(const half *) (kd1 + b*34));
                s[0] += (float) c[0] * (dq0[b]*dk0);
                s[1] += (float) c[1] * (dq0[b]*dk1);
                s[2] += (float) c[2] * (dq1[b]*dk0);
                s[3] += (float) c[3] * (dq1[b]*dk1);
            }
            float * s0 = sS + (mi*16 + g    )*KVQ_S_STRIDE + pb + 2*t;
            float * s1 = sS + (mi*16 + g + 8)*KVQ_S_STRIDE + pb + 2*t;
            s0[0] = s[0]; s0[1] = s[1];
            s1[0] = s[2]; s1[1] = s[3];
        }
        __syncthreads();

        // ---- online softmax, P -> f16 ----
        if (rs < R) {
            // warp-uniform: rows >= nrows see -inf everywhere and end with p = 0, alpha = 1
            float p[8];
            float alpha = 1.0f;
            const bool valid = rs < nrows;
            const half * mrow = (valid && mask) ? mask + (int64_t)(rs / gqa)*nbm1 + tile*KVQ_T + qs*8 : nullptr;
            float mx = -INFINITY;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                p[k] = valid ? sS[rs*KVQ_S_STRIDE + qs*8 + k] + (mrow ? __half2float(mrow[k]) : 0.0f) : -INFINITY;
                mx = fmaxf(mx, p[k]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, 2));
            const float M_new = fmaxf(M_run, mx);
            float sum = 0.0f;
            if (M_new == -INFINITY) {
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    p[k] = 0.0f;
                }
            } else {
                alpha = expf(M_run - M_new);
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    p[k] = expf(p[k] - M_new);
                    sum += p[k];
                }
            }
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
            l_run = l_run*alpha + sum;
            M_run = M_new;
            half2 * prow = (half2 *) (sP + rs*KVQ_P_STRIDE + qs*8);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                prow[k] = __floats2half2_rn(p[2*k], p[2*k + 1]);
            }
            if (qs == 0) {
                sAlpha[rs] = alpha;
            }
        }
        __syncthreads();

        // ---- O = O*alpha + P V (f16 tensor cores); warp w owns d in [64w, 64w+64) ----
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const float a0 = sAlpha[m*16 + g];
            const float a1 = sAlpha[m*16 + g + 8];
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                O[m][n][0] *= a0; O[m][n][1] *= a0;
                O[m][n][2] *= a1; O[m][n][3] *= a1;
            }
        }
#pragma unroll
        for (int ks = 0; ks < KVQ_T/16; ++ks) {
            uint32_t aP[MT][4];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                const half * p0 = sP + (m*16 + g    )*KVQ_P_STRIDE + ks*16 + 2*t;
                const half * p1 = sP + (m*16 + g + 8)*KVQ_P_STRIDE + ks*16 + 2*t;
                aP[m][0] = *(const uint32_t *) (p0);
                aP[m][1] = *(const uint32_t *) (p1);
                aP[m][2] = *(const uint32_t *) (p0 + 8);
                aP[m][3] = *(const uint32_t *) (p1 + 8);
            }
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const half * vt = sVt + (w*64 + n*8 + g)*KVQ_VT_STRIDE + ks*16 + 2*t;
                const uint32_t b0 = *(const uint32_t *) (vt);
                const uint32_t b1 = *(const uint32_t *) (vt + 8);
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    kvq_mma_f16(O[m][n], aP[m], b0, b1);
                }
            }
        }
        __syncthreads();

        if (tile + 1 < tile1) {
            store();
            __syncthreads();
        }
    }

    // ---- partial results ----
    const int64_t base = ((int64_t) split*gridDim.y + h)*KVQ_RMAX;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int r0 = m*16 + g;
        const int r1 = r0 + 8;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            const int d = w*64 + n*8 + 2*t;
            if (r0 < nrows) {
                *(float2 *) (part + (base + r0)*KVQ_D + d) = make_float2(O[m][n][0], O[m][n][1]);
            }
            if (r1 < nrows) {
                *(float2 *) (part + (base + r1)*KVQ_D + d) = make_float2(O[m][n][2], O[m][n][3]);
            }
        }
    }
    if (rs < nrows && qs == 0) {
        meta[base + rs] = make_float2(M_run, l_run);
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, part, meta, scale, ne11, ncols, gqa, nbq1, nbq2, nbk1, nbk2, nbv1, nbv2, nbm1, tiles_per_split);
    NO_DEVICE_CODE;
#endif // defined(KVQ_AVAILABLE)
}

static __global__ void flash_attn_kvq_gqa_combine(
        const float * __restrict__ part, const float2 * __restrict__ meta, float * __restrict__ dst,
        const int nsplit, const int nheads_kv, const int gqa, const int nheads_q) {
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;

    float M = -INFINITY;
    for (int s = 0; s < nsplit; ++s) {
        M = fmaxf(M, meta[((int64_t) s*nheads_kv + h)*KVQ_RMAX + r].x);
    }
    float num = 0.0f;
    float den = 0.0f;
    if (M != -INFINITY) {
        for (int s = 0; s < nsplit; ++s) {
            const int64_t i  = ((int64_t) s*nheads_kv + h)*KVQ_RMAX + r;
            const float2  ml = meta[i];
            if (ml.x == -INFINITY) {
                continue;
            }
            const float f = expf(ml.x - M);
            num += f*part[i*KVQ_D + d];
            den += f*ml.y;
        }
    }
    const int j = r / gqa;
    const int i = r - j*gqa;
    dst[((int64_t) j*nheads_q + h*gqa + i)*KVQ_D + d] = den > 0.0f ? num/den : 0.0f;
}

static int kvq_env_int(const char * name, const int def) {
    const char * s = getenv(name);
    return s && *s ? atoi(s) : def;
}

// Exact production shape only; everything else stays on the stock selector.
static bool ggml_cuda_fa_kvq_gqa_applies(const int device, const ggml_tensor * dst) {
    static const int enabled = kvq_env_int("GGML_CUDA_FA_KVQ_GQA",        1);
    static const int min_kv  = kvq_env_int("GGML_CUDA_FA_KVQ_GQA_MIN_KV", 512);
    static const int max_nb  = kvq_env_int("GGML_CUDA_FA_KVQ_GQA_MAX_NB", 4);
    if (!enabled) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE || ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sink = dst->src[4];

    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));

    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q4_0 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[0] != KVQ_D || K->ne[0] != KVQ_D || V->ne[0] != KVQ_D) {
        return false;
    }
    if (K->ne[2] <= 0 || Q->ne[2] != 6*K->ne[2] || V->ne[2] != K->ne[2]) {
        return false;
    }
    if (Q->ne[1] < 1 || Q->ne[1] > max_nb || Q->ne[1]*6 > KVQ_RMAX) {
        return false;
    }
    if (Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) {
        return false;
    }
    if (K->ne[1] != V->ne[1] || K->ne[1] % 256 != 0 || K->ne[1] < min_kv) {
        return false;
    }
    if (max_bias != 0.0f || softcap != 0.0f || sink != nullptr) {
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] ||
                 mask->ne[2] != 1 || mask->ne[3] != 1 || mask->nb[1] % sizeof(half) != 0)) {
        return false;
    }
    if (Q->nb[0] != sizeof(float) || Q->nb[1] % 16 != 0 || Q->nb[2] % 16 != 0 || (uintptr_t) Q->data % 16 != 0) {
        return false;
    }
    if (K->nb[0] != ggml_type_size(GGML_TYPE_Q8_0) || K->nb[1] % 16 != 0 || K->nb[2] % 16 != 0 || (uintptr_t) K->data % 16 != 0) {
        return false;
    }
    if (V->nb[0] != ggml_type_size(GGML_TYPE_Q4_0) || V->nb[1] % 16 != 0 || V->nb[2] % 16 != 0 || (uintptr_t) V->data % 16 != 0) {
        return false;
    }
    if (!ggml_is_contiguous(dst)) {
        return false;
    }
    return true;
}

static void ggml_cuda_fa_kvq_gqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int id        = ggml_cuda_get_device();
    const int nsm       = ggml_cuda_info().devices[id].nsm;
    const int nheads_kv = K->ne[2];
    const int gqa       = Q->ne[2] / K->ne[2];
    const int ncols     = Q->ne[1];
    const int ne11      = K->ne[1];
    const int ntiles    = ne11 / KVQ_T;

    static const int bps = kvq_env_int("GGML_CUDA_FA_KVQ_GQA_BPS", 2); // target resident blocks per SM
    int nsplit = std::max(1, std::min(ntiles, (nsm*std::max(1, bps) + nheads_kv - 1) / nheads_kv));
    const int tiles_per_split = (ntiles + nsplit - 1) / nsplit;
    nsplit = (ntiles + tiles_per_split - 1) / tiles_per_split;

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float>  part(pool, (size_t) nsplit*nheads_kv*KVQ_RMAX*KVQ_D);
    ggml_cuda_pool_alloc<float2> meta(pool, (size_t) nsplit*nheads_kv*KVQ_RMAX);

    const dim3 grid(nsplit, nheads_kv, 1);
    const dim3 block(KVQ_NTHREADS, 1, 1);
    const int64_t nbm1 = mask ? mask->nb[1] / sizeof(half) : 0;

    if (ncols*gqa <= 16) {
        flash_attn_kvq_gqa<1><<<grid, block, 0, stream>>>(
            (const char *) Q->data, (const char *) K->data, (const char *) V->data, mask ? (const half *) mask->data : nullptr,
            part.ptr, meta.ptr, scale, ne11, ncols, gqa, Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], nbm1, tiles_per_split);
    } else {
        flash_attn_kvq_gqa<2><<<grid, block, 0, stream>>>(
            (const char *) Q->data, (const char *) K->data, (const char *) V->data, mask ? (const half *) mask->data : nullptr,
            part.ptr, meta.ptr, scale, ne11, ncols, gqa, Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], nbm1, tiles_per_split);
    }
    CUDA_CHECK(cudaGetLastError());

    const dim3 grid_c(ncols*gqa, nheads_kv, 1);
    flash_attn_kvq_gqa_combine<<<grid_c, KVQ_D, 0, stream>>>(part.ptr, meta.ptr, (float *) dst->data, nsplit, nheads_kv, gqa, Q->ne[2]);
    CUDA_CHECK(cudaGetLastError());
}

#endif // defined(GGML_EXP_CUDA_FA_KVQ_GQA)
