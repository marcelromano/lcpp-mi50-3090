#include "mmvq.cuh"
#include "cola.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
#ifdef GGML_EXP_GCN_MMVQ_GEOM
        // gfx906: one wave per block for the decode/MTP-verify shapes (more blocks in flight)
        if (ncols_dst <= 4) {
            return 1;
        }
#endif
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING) {
#ifdef GGML_EXP_GCN_MMVQ_GEOM
        if (table_id == MMVQ_PARAMETERS_GCN && ncols_dst == 1 && !small_k) {
            return 2;
        }
#endif
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

#if defined(GGML_USE_HIP) && defined(GGML_EXP_IQ_LDS2)
    {
        const int t0 = threadIdx.y*blockDim.x + threadIdx.x, nt = blockDim.x*blockDim.y;
        if constexpr (type == GGML_TYPE_IQ2_XXS) { for (int i = t0; i < 256;  i += nt) iq2xxs_grid_lds[i] = iq2xxs_grid[i]; __syncthreads(); }
        if constexpr (type == GGML_TYPE_IQ2_XS)  { for (int i = t0; i < 512;  i += nt) iq2xs_grid_lds[i]  = iq2xs_grid[i];  __syncthreads(); }
        if constexpr (type == GGML_TYPE_IQ3_S)   { for (int i = t0; i < 512;  i += nt) iq3s_grid_lds[i]   = iq3s_grid[i];   __syncthreads(); }
    }
#endif
#if defined(GGML_USE_HIP) && defined(GGML_EXP_IQ_LDS)
    if constexpr (type == GGML_TYPE_IQ3_XXS) {
        for (int i = threadIdx.y*blockDim.x + threadIdx.x; i < 256; i += blockDim.x*blockDim.y) {
            iq3xxs_grid_lds[i] = iq3xxs_grid[i];
        }
        __syncthreads();
    }
#endif
#if defined(GGML_USE_HIP) && defined(GGML_EXP_MMVQ_GCN2_IQ2S_LDS)
    if constexpr (type == GGML_TYPE_IQ2_S) {
        for (int i = threadIdx.y*blockDim.x + threadIdx.x; i < 1024; i += blockDim.x*blockDim.y) {
            iq2s_grid_lds[i] = iq2s_grid[i];
        }
        __syncthreads();
    }
#endif

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

#if defined(GGML_USE_HIP) && defined(GGML_EXP_IQ_LDS2)
    {
        const int t0 = threadIdx.y*blockDim.x + threadIdx.x, nt = blockDim.x*blockDim.y;
        if constexpr (type == GGML_TYPE_IQ2_XXS) { for (int i = t0; i < 256;  i += nt) iq2xxs_grid_lds[i] = iq2xxs_grid[i]; __syncthreads(); }
        if constexpr (type == GGML_TYPE_IQ2_XS)  { for (int i = t0; i < 512;  i += nt) iq2xs_grid_lds[i]  = iq2xs_grid[i];  __syncthreads(); }
        if constexpr (type == GGML_TYPE_IQ3_S)   { for (int i = t0; i < 512;  i += nt) iq3s_grid_lds[i]   = iq3s_grid[i];   __syncthreads(); }
    }
#endif
#if defined(GGML_USE_HIP) && defined(GGML_EXP_IQ_LDS)
    if constexpr (type == GGML_TYPE_IQ3_XXS) {
        for (int i = threadIdx.y*blockDim.x + threadIdx.x; i < 256; i += blockDim.x*blockDim.y) {
            iq3xxs_grid_lds[i] = iq3xxs_grid[i];
        }
        __syncthreads();
    }
#endif
#if defined(GGML_USE_HIP) && defined(GGML_EXP_MMVQ_GCN2_IQ2S_LDS)
    if constexpr (type == GGML_TYPE_IQ2_S) {
        for (int i = threadIdx.y*blockDim.x + threadIdx.x; i < 1024; i += blockDim.x*blockDim.y) {
            iq2s_grid_lds[i] = iq2s_grid[i];
        }
        __syncthreads();
    }
#endif

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

#if defined(GGML_USE_HIP) && defined(GGML_EXP_MMVQ_GCN2)
// ---------------------------------------------------------------------------------------------
// GGML_EXP_MMVQ_GCN2: mmvq for wave64 GCN (gfx906) decode, ncols_dst <= 4 and MUL_MAT_ID MoE.
//
// * 1 wave per block (no idle second wave, no LDS/__syncthreads cross-wave reduction).
// * "Lane packing": when one row needs fewer than 64 lanes (lanes_per_row = K/qk * qi/vdr, e.g. IQ4_XS K=512 ->
//   16 lanes) the wave processes G = 64/lanes_per_row rows at the same time, one row per lane group.
// * Reduction inside a lane group with DPP row ops (quad_perm, row_half_mirror, row_mirror), ds_swizzle xor16 and
//   v_readlane instead of 6 dependent ds_bpermute (__shfl_xor). This changes the float summation order.
// * rpl serial rows per lane (shares the q8_1 loads) when a row already fills the wave.
// Without GGML_EXP_MMVQ_GCN2 nothing in this block is compiled.
// ---------------------------------------------------------------------------------------------
#ifndef GGML_EXP_MMVQ_GCN2_RPL_FULL
#define GGML_EXP_MMVQ_GCN2_RPL_FULL 2   // rows per lane when lanes_per_row == 64 (measured best for nw1: rpb 2)
#endif
#ifndef GGML_EXP_MMVQ_GCN2_RPL_PACKED
#define GGML_EXP_MMVQ_GCN2_RPL_PACKED 1 // rows per lane per group when several rows share one wave
#endif
static_assert(GGML_EXP_MMVQ_GCN2_RPL_FULL   == 1 || GGML_EXP_MMVQ_GCN2_RPL_FULL   == 2, "RPL_FULL must be 1 or 2");
static_assert(GGML_EXP_MMVQ_GCN2_RPL_PACKED == 1 || GGML_EXP_MMVQ_GCN2_RPL_PACKED == 2, "RPL_PACKED must be 1 or 2");

static constexpr __host__ __device__ bool mmvq_gcn2_supported_type(ggml_type type) {
#ifdef GGML_EXP_MMVQ_GCN2_NO_IQ3
    return type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_Q5_K;
#else
    return type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_Q5_K;
#endif
}

// log2 of the number of lanes that cooperate on one row (power of two in [8, 64])
static __host__ __device__ __forceinline__ int mmvq_gcn2_lpr_log2(const int lanes_per_blk, const int blocks_per_row) {
    const int l = lanes_per_blk * blocks_per_row;
    if (l >= 64 || (l & (l - 1)) != 0) {
        return 6;
    }
    int s = 0;
    while ((1 << s) < l) {
        ++s;
    }
    return s;
}

template <int ctrl>
static __device__ __forceinline__ float mmvq_gcn2_dpp_add(const float x) {
    return x + __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), ctrl, 0xF, 0xF, false));
}

// sum over the lpr = 1 << lpr_log2 lanes of each aligned lane group; lpr >= 8. Result is valid in lane 0 of each group.
static __device__ __forceinline__ float mmvq_gcn2_group_sum(float x, const int lpr_log2) {
    x = mmvq_gcn2_dpp_add<0xB1>(x);  // quad_perm [1,0,3,2]
    x = mmvq_gcn2_dpp_add<0x4E>(x);  // quad_perm [2,3,0,1]
    x = mmvq_gcn2_dpp_add<0x141>(x); // row_half_mirror
    if (lpr_log2 >= 4) {
        x = mmvq_gcn2_dpp_add<0x140>(x); // row_mirror
    }
    if (lpr_log2 >= 5) {
        x += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(x), 0x401F)); // lane ^ 16
    }
    if (lpr_log2 >= 6) {
        x = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x), 0)) +
            __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x), 32));
    }
    return x;
}

static constexpr __device__ vec_dot_q_cuda_t mmvq_gcn2_get_vec_dot(ggml_type type, bool lut = false) {
#if !defined(GGML_EXP_MMVQ_GCN2_NOWIDE)
    if (type == GGML_TYPE_IQ4_XS) {
        return vec_dot_iq4_xs_q8_1_gcn2;
    }
#endif // !defined(GGML_EXP_MMVQ_GCN2_NOWIDE)
#if defined(GGML_EXP_MMVQ_GCN2_IQ3V2) && !defined(GGML_EXP_MMVQ_GCN2_V2_NOVEC)
    if (type == GGML_TYPE_IQ3_XXS) { return vec_dot_iq3_xxs_q8_1_v2; }
    if (type == GGML_TYPE_IQ3_S)   { return vec_dot_iq3_s_q8_1_v2;   }
#if defined(GGML_EXP_MMVQ_DEC_IQ2LUT)
    if (type == GGML_TYPE_IQ2_XXS) { return lut ? vec_dot_iq2_xxs_q8_1_v2<true> : vec_dot_iq2_xxs_q8_1_v2<false>; }
    if (type == GGML_TYPE_IQ2_XS)  { return lut ? vec_dot_iq2_xs_q8_1_v2<true>  : vec_dot_iq2_xs_q8_1_v2<false>;  }
    if (type == GGML_TYPE_IQ2_S)   { return lut ? vec_dot_iq2_s_q8_1_v2<true>   : vec_dot_iq2_s_q8_1_v2<false>;   }
#else
    GGML_UNUSED(lut);
    if (type == GGML_TYPE_IQ2_XXS) { return vec_dot_iq2_xxs_q8_1_v2<>; }
    if (type == GGML_TYPE_IQ2_XS)  { return vec_dot_iq2_xs_q8_1_v2<>;  }
    if (type == GGML_TYPE_IQ2_S)   { return vec_dot_iq2_s_q8_1_v2<>;   }
#endif
#endif
#if defined(GGML_EXP_MMVQ_DEC_Q2K)
    if (type == GGML_TYPE_Q2_K)    { return vec_dot_q2_K_q8_1_v2;    }
#endif
    return get_vec_dot_q_cuda(type);
}

#if defined(GGML_EXP_MMVQ_GCN2_IQ3V2)
// ---------------------------------------------------------------------------------------------
// GGML_EXP_MMVQ_GCN2_IQ3V2: extend the gcn2 path to the sub-4-bit i-quants (+ Q2_K). Switches:
//   compile: -DGGML_EXP_MMVQ_GCN2_IQ3V2            master (needs GGML_EXP_MMVQ_GCN2, GGML_EXP_IQ_LDS, GGML_EXP_IQ_LDS2)
//            -DGGML_EXP_MMVQ_GCN2_V2_NOVEC         gcn2 kernels keep the original vec_dot of the i-quants (isolates the vec_dot)
//            -DGGML_EXP_MMVQ_GCN2_V2_NOWIDE_W / _NOWIDE_Y  sub-dword weight loads / dword y loads in the v2 vec_dots
//            -DGGML_EXP_MMVQ_GCN2_IQ2S_LDS         IQ2_S grid (8 KiB) in LDS (opt-in: 10 KiB per block limits occupancy)
//            -DGGML_EXP_MMVQ_GCN2_V2_RPL=1|2       rows per lane in the dedicated full-wave kernel (default 2)
//            -DGGML_EXP_MMVQ_GCN2_V2_BPC=N         persistent blocks: grid = N * #CU (0 = one tile per block, default)
//            -DGGML_EXP_MMVQ_GCN2_V2_WPE=N         amdgpu_waves_per_eu(N) on the dedicated kernel (0 = unset, default)
//   runtime: GGML_MMVQ_GCN2_V2=<mask>  bit0 IQ3_XXS, bit1 IQ3_S, bit2 IQ2_S, bit3 IQ2_XS, bit4 IQ2_XXS, bit5 Q2_K (default 0x3F)
//            GGML_MMVQ_GCN2_V2_BPC=N   overrides the compile-time BPC
//            GGML_MMVQ_GCN2=0          (existing) disables the whole gcn2 path
// For a clean A/B of IQ3_XXS keep -DGGML_EXP_MMVQ_GCN2_NO_IQ3: a type whose mask bit is 0 then takes the geometry path.
// ---------------------------------------------------------------------------------------------
#ifndef GGML_EXP_MMVQ_GCN2_V2_RPL
#define GGML_EXP_MMVQ_GCN2_V2_RPL 2
#endif
#ifndef GGML_EXP_MMVQ_GCN2_V2_BPC
#define GGML_EXP_MMVQ_GCN2_V2_BPC 0
#endif
#ifndef GGML_EXP_MMVQ_GCN2_V2_WPE
#define GGML_EXP_MMVQ_GCN2_V2_WPE 0
#endif
static_assert(GGML_EXP_MMVQ_GCN2_V2_RPL == 1 || GGML_EXP_MMVQ_GCN2_V2_RPL == 2, "V2_RPL must be 1 or 2");
#if GGML_EXP_MMVQ_GCN2_V2_WPE > 0
#define MMVQ_GCN2V2_ATTR __attribute__((amdgpu_waves_per_eu(GGML_EXP_MMVQ_GCN2_V2_WPE)))
#else
#define MMVQ_GCN2V2_ATTR
#endif

// ---------------------------------------------------------------------------------------------
// t21 (alvo 2, decode n=1): knobs GGML_EXP_MMVQ_DEC_* for the dedicated kernel (all off unless defined):
//   _ADDR   row base pointers hoisted out of the kbx loop (no v_mad_i64_i32 / 64-bit re-add per row per iteration)
//   _MUL24  scale products on v_mul_i32_i24 (vecdotq.cuh, MMVQ_V2_MUL)
//   _Q2K    Q2_K vec_dot v2 (vecdotq.cuh, vec_dot_q2_K_q8_1_v2)
//   _SGN2   conflict-free nibble sign tables (vecdotq.cuh, MMVQ_V2_SGN7/8)
//   _NW=n   n waves per block (LDS types only): tables filled once per block, 1/n of the fill per wave; with
//           GGML_EXP_MMVQ_GCN2_IQ2S_LDS the IQ2_S grid costs 10 KiB per n waves instead of per wave
// ---------------------------------------------------------------------------------------------
#ifndef GGML_EXP_MMVQ_DEC_WPE
#define GGML_EXP_MMVQ_DEC_WPE 1   // 2nd __launch_bounds__ argument = min waves per EU on HIP (amdgpu-waves-per-eu): caps VGPRs at 256/WPE
#endif
#ifndef GGML_EXP_MMVQ_DEC_NW
#define GGML_EXP_MMVQ_DEC_NW 1
#endif
static_assert(GGML_EXP_MMVQ_DEC_NW >= 1 && GGML_EXP_MMVQ_DEC_NW <= 8, "DEC_NW must be 1..8");

template <ggml_type type>
static constexpr __host__ __device__ int mmvq_dec_blk_bytes() {
    return type == GGML_TYPE_Q2_K    ? (int) sizeof(block_q2_K)    :
           type == GGML_TYPE_IQ2_XXS ? (int) sizeof(block_iq2_xxs) :
           type == GGML_TYPE_IQ2_XS  ? (int) sizeof(block_iq2_xs)  :
           type == GGML_TYPE_IQ2_S   ? (int) sizeof(block_iq2_s)   :
           type == GGML_TYPE_IQ3_XXS ? (int) sizeof(block_iq3_xxs) :
           type == GGML_TYPE_IQ3_S   ? (int) sizeof(block_iq3_s)   : 0;
}

// i-quants whose grid/sign tables live in LDS and are filled by the prologue
static constexpr __host__ __device__ bool mmvq_gcn2v2_lds_type(ggml_type type) {
    return type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_XXS ||
           type == GGML_TYPE_IQ2_XS  || type == GGML_TYPE_IQ2_S;
}

// types added to the gcn2 path by this experiment (IQ3_XXS is included even when NO_IQ3 excludes it from the base set)
static constexpr __host__ __device__ bool mmvq_gcn2v2_type(ggml_type type) {
    return mmvq_gcn2v2_lds_type(type) || type == GGML_TYPE_Q2_K;
}

// perm-LUT sign/magnitude path (GGML_EXP_MMVQ_DEC_IQ2LUT): IQ2_* and ncols_dst == 1 only
static constexpr __host__ __device__ bool mmvq_dec_lut(ggml_type type, int ncols_dst) {
#if defined(GGML_EXP_MMVQ_DEC_IQ2LUT)
    return ncols_dst == 1 && (type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S);
#else
    GGML_UNUSED(type); GGML_UNUSED(ncols_dst);
    return false;
#endif
}

// waves per block of the dedicated kernel
static constexpr __host__ __device__ int mmvq_dec_nwv(ggml_type type) {
    return mmvq_gcn2v2_lds_type(type) ? GGML_EXP_MMVQ_DEC_NW : 1;
}

static constexpr __host__ __device__ int mmvq_gcn2v2_bit(ggml_type type) {
    return type == GGML_TYPE_IQ3_XXS ? 1 : type == GGML_TYPE_IQ3_S ? 2 : type == GGML_TYPE_IQ2_S ? 4 :
           type == GGML_TYPE_IQ2_XS  ? 8 : type == GGML_TYPE_IQ2_XXS ? 16 : type == GGML_TYPE_Q2_K ? 32 : 0;
}

// Fill sign table (+ grid when with_grid) of `type` in LDS. t0 = this thread's index, nt = number of threads that cooperate.
// The caller synchronizes (wave fence for 1-wave blocks, __syncthreads otherwise).
template <ggml_type type, bool with_grid, bool lut = false>
static __device__ __forceinline__ void mmvq_gcn2v2_fill_lds(const int t0, const int nt) {
#if defined(GGML_EXP_MMVQ_DEC_SGN2)
    if constexpr (type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS ||
                  type == GGML_TYPE_IQ3_S   || type == GGML_TYPE_IQ2_S) {
        for (int i = t0; i < 32; i += nt) {
            mmvq_v2_nib_lds[i] = mmvq_v2_nib_mask(i >> 1);
        }
    }
#else
    if constexpr (type == GGML_TYPE_IQ3_XXS || (!lut && (type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS))) {
#pragma unroll
        for (int i = t0; i < 128; i += nt) {
            const uint32_t p = __popc(i) & 1;
            mmvq_v2_sgn7_lds[i] = mmvq_v2_sign_masks(i ^ (p << 7));
        }
    } else if constexpr (type == GGML_TYPE_IQ3_S || (!lut && type == GGML_TYPE_IQ2_S)) {
#pragma unroll
        for (int i = t0; i < 256; i += nt) {
            mmvq_v2_sgn8_lds[i] = mmvq_v2_sign_masks(i);
        }
    }
#endif // defined(GGML_EXP_MMVQ_DEC_SGN2)
#if defined(GGML_EXP_MMVQ_DEC_IQ2LUT)
    if constexpr (!lut) {
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        for (int i = t0; i < 256; i += nt)  { const uint2 g = ((const uint2 *) iq2xxs_grid)[i]; mmvq_v2_iq2xxs_sym[i] = make_uint2(mmvq_v2_sym4(g.x), mmvq_v2_sym4(g.y)); }
    } else if constexpr (type == GGML_TYPE_IQ2_XS) {
        for (int i = t0; i < 512; i += nt)  { const uint2 g = ((const uint2 *) iq2xs_grid)[i];  mmvq_v2_iq2xs_sym[i]  = make_uint2(mmvq_v2_sym4(g.x), mmvq_v2_sym4(g.y)); }
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        for (int i = t0; i < 1024; i += nt) { const uint2 g = ((const uint2 *) iq2s_grid)[i];   mmvq_v2_iq2s_sym[i]   = make_uint2(mmvq_v2_sym4(g.x), mmvq_v2_sym4(g.y)); }
    }
#endif
#if defined(GGML_EXP_MMVQ_DEC_IQ2LUT)
    if constexpr (lut && with_grid && (type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S)) {
        return; // the symbol table replaces the grid in LDS
    }
#endif
    if constexpr (with_grid) {
        if constexpr (type == GGML_TYPE_IQ3_XXS) {
#pragma unroll
            for (int i = t0; i < 256; i += nt) { iq3xxs_grid_lds[i] = iq3xxs_grid[i]; }
        } else if constexpr (type == GGML_TYPE_IQ3_S) {
#pragma unroll
            for (int i = t0; i < 512; i += nt) { iq3s_grid_lds[i] = iq3s_grid[i]; }
        } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
#pragma unroll
            for (int i = t0; i < 256; i += nt) { iq2xxs_grid_lds[i] = iq2xxs_grid[i]; }
        } else if constexpr (type == GGML_TYPE_IQ2_XS) {
#pragma unroll
            for (int i = t0; i < 512; i += nt) { iq2xs_grid_lds[i] = iq2xs_grid[i]; }
        }
#if defined(GGML_EXP_MMVQ_GCN2_IQ2S_LDS)
        else if constexpr (type == GGML_TYPE_IQ2_S) {
#pragma unroll
            for (int i = t0; i < 1024; i += nt) { iq2s_grid_lds[i] = iq2s_grid[i]; }
        }
#endif
    }
}

static __device__ __forceinline__ void mmvq_gcn2v2_wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
#endif // defined(GGML_EXP_MMVQ_GCN2_IQ3V2)

template <ggml_type type, int ncols_dst, bool has_fusion, int rpl>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_gcn2(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int lanes_per_blk = qi / vdr;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(lanes_per_blk >= 8 && (lanes_per_blk & (lanes_per_blk - 1)) == 0, "lanes_per_blk must be a power of two >= 8");
    static_assert(type != GGML_TYPE_NVFP4, "scale fusion not supported");

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = mmvq_gcn2_get_vec_dot(type);

#if defined(GGML_EXP_IQ_LDS)
    // IQ3_XXS grid -> LDS. One wave per block, so no __syncthreads: issue the global loads first so that they overlap the
    // ids load below, store to LDS later, wave-scope fence (LDS ops of one wave execute in order).
    uint32_t grid_regs[4] = {0, 0, 0, 0};
    if constexpr (type == GGML_TYPE_IQ3_XXS) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            grid_regs[k] = iq3xxs_grid[threadIdx.x + warp_size*k];
        }
    }
#endif // defined(GGML_EXP_IQ_LDS)

    const int lane             = threadIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    const int lpr_log2         = mmvq_gcn2_lpr_log2(lanes_per_blk, blocks_per_row_x);
    const int lpr              = 1 << lpr_log2;
    const int G                = warp_size >> lpr_log2;   // rows processed concurrently by the wave
    const int grp              = lane >> lpr_log2;
    const int lil              = lane & (lpr - 1);        // lane inside its row group
    const int row_base         = blockIdx.x * (G * rpl) + grp;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

#if defined(GGML_EXP_IQ_LDS)
    if constexpr (type == GGML_TYPE_IQ3_XXS) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            iq3xxs_grid_lds[threadIdx.x + warp_size*k] = grid_regs[k];
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }
#endif // defined(GGML_EXP_IQ_LDS)
#if defined(GGML_EXP_MMVQ_GCN2_IQ3V2)
    if constexpr (mmvq_gcn2v2_lds_type(type)) {
        // IQ3_XXS grid was filled above; everything else (sign tables, other grids) here, 1 wave: wave fence only
        mmvq_gcn2v2_fill_lds<type, type != GGML_TYPE_IQ3_XXS>(threadIdx.x, warp_size);
        mmvq_gcn2v2_wave_sync();
    }
#endif // defined(GGML_EXP_MMVQ_GCN2_IQ3V2)

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    ggml_glu_op active_glu;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
    }

    int  rows[rpl];
    int  rowc[rpl];
    bool rvalid[rpl];
#pragma unroll
    for (int r = 0; r < rpl; ++r) {
        rows[r]   = row_base + r*G;
        rvalid[r] = uint32_t(rows[r]) < nrows_x;
        rowc[r]   = rvalid[r] ? rows[r] : int(nrows_x) - 1; // tail rows recompute the last row, result is discarded
    }

    [[maybe_unused]] float x_biases[ncols_dst][rpl]    = {{0.0f}};
    [[maybe_unused]] float gate_biases[ncols_dst][rpl] = {{0.0f}};
    if constexpr (has_fusion) {
        // prefetch bias / gate bias in the lane that writes the row result
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (lil == 0) {
            if (use_bias) {
                const float * xb = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                    for (int r = 0; r < rpl; ++r) {
                        if (rvalid[r]) {
                            x_biases[j][r] = xb[j * stride_col_dst + rows[r]];
                        }
                    }
                }
            }
            if (use_gate_bias) {
                const float * gb = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                    for (int r = 0; r < rpl; ++r) {
                        if (rvalid[r]) {
                            gate_biases[j][r] = gb[j * stride_col_dst + rows[r]];
                        }
                    }
                }
            }
        }
    }

    float tmp[ncols_dst][rpl] = {{0.0f}};
    float tmp_gate[ncols_dst][rpl] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x;

    const int kbx_lane = lil / lanes_per_blk;
    const int kqs      = vdr * (lil % lanes_per_blk);
    const int bpi      = lpr / lanes_per_blk; // blocks per iteration of the lane group

    for (int kbx = kbx_lane; kbx < blocks_per_row_x; kbx += bpi) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int r = 0; r < rpl; ++r) {
                tmp[j][r] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + rowc[r]*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][r] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + rowc[r]*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst;

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int r = 0; r < rpl; ++r) {
            float result = mmvq_gcn2_group_sum(tmp[j][r], lpr_log2);
            [[maybe_unused]] float gate_value = 0.0f;
            if constexpr (has_fusion) {
                if (use_gate) {
                    gate_value = mmvq_gcn2_group_sum(tmp_gate[j][r], lpr_log2);
                }
            }

            if (lil == 0 && rvalid[r]) {
                if constexpr (has_fusion) {
                    result += x_biases[j][r];
                    if (use_gate) {
                        gate_value += gate_biases[j][r];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + rows[r]] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, active_glu, gate_bias, x_bias, tmp_gate, ids_stride);
    }
}

// MoE multi-token variant (ncols_dst > 1 with ids): one wave per token, same lane packing as above.
// Grid: (ceil(nrows_x / (G*rpl)), nchannels_dst), block (warp_size, ncols_dst).
template <ggml_type type, int rpl>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe_gcn2(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int lanes_per_blk = qi / vdr;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(lanes_per_blk >= 8 && (lanes_per_blk & (lanes_per_blk - 1)) == 0, "lanes_per_blk must be a power of two >= 8");

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = mmvq_gcn2_get_vec_dot(type);

#if defined(GGML_EXP_IQ_LDS)
    if constexpr (type == GGML_TYPE_IQ3_XXS) {
        // several waves per block here: block-wide barrier
        for (int i = threadIdx.y*blockDim.x + threadIdx.x; i < 256; i += blockDim.x*blockDim.y) {
            iq3xxs_grid_lds[i] = iq3xxs_grid[i];
        }
        __syncthreads();
    }
#endif // defined(GGML_EXP_IQ_LDS)
#if defined(GGML_EXP_MMVQ_GCN2_IQ3V2)
    if constexpr (mmvq_gcn2v2_lds_type(type)) {
        mmvq_gcn2v2_fill_lds<type, type != GGML_TYPE_IQ3_XXS>(threadIdx.y*blockDim.x + threadIdx.x, blockDim.x*blockDim.y);
        __syncthreads();
    }
#endif // defined(GGML_EXP_MMVQ_GCN2_IQ3V2)

    const uint32_t token_idx = threadIdx.y;
    const int blocks_per_row_x = ncols_x / qk;
    const int lpr_log2 = mmvq_gcn2_lpr_log2(lanes_per_blk, blocks_per_row_x);
    const int lpr      = 1 << lpr_log2;
    const int G        = warp_size >> lpr_log2;
    const int grp      = threadIdx.x >> lpr_log2;
    const int lil      = threadIdx.x & (lpr - 1);
    const int row_base = blockIdx.x * (G * rpl) + grp;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset = channel_x*stride_channel_x;

    int rows[rpl];
    int rowc[rpl];
    bool rvalid[rpl];
#pragma unroll
    for (int r = 0; r < rpl; ++r) {
        rows[r]   = row_base + r*G;
        rvalid[r] = uint32_t(rows[r]) < nrows_x;
        rowc[r]   = rvalid[r] ? rows[r] : int(nrows_x) - 1;
    }

    float tmp[rpl] = {0.0f};

    const int kbx_lane = lil / lanes_per_blk;
    const int kqs      = vdr * (lil % lanes_per_blk);
    const int bpi      = lpr / lanes_per_blk;

    for (int kbx = kbx_lane; kbx < blocks_per_row_x; kbx += bpi) {
        const int kby = kbx * (qk/QK8_1);
#pragma unroll
        for (int r = 0; r < rpl; ++r) {
            tmp[r] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + rowc[r]*stride_row_x + kbx, kqs);
        }
    }

    ggml_cuda_pdl_lc();

#pragma unroll
    for (int r = 0; r < rpl; ++r) {
        const float sum = mmvq_gcn2_group_sum(tmp[r], lpr_log2);
        if (lil == 0 && rvalid[r]) {
            dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + rows[r]] = sum;
        }
    }
}
#endif // defined(GGML_USE_HIP) && defined(GGML_EXP_MMVQ_GCN2)

#if defined(GGML_USE_HIP) && defined(GGML_EXP_MMVQ_GCN2) && defined(GGML_EXP_MMVQ_GCN2_IQ3V2)
// ---------------------------------------------------------------------------------------------
//  dedicated dense kernel for rows that fill the wave (lanes_per_row == 64, i.e. K >= 64/lanes_per_blk superblocks:
// every 27B matrix, 35B K >= 2048), ncols_dst <= 4, dense or MUL_MAT_ID with 1 token. Differences to mul_mat_vec_q_gcn2:
//  * lane geometry is constexpr (kbx0 = lane / lanes_per_blk, stride bpi = 64 / lanes_per_blk): no runtime lpr/G/grp/lil;
//  * persistent tile loop (grid = ntiles, or BPC * #CU): grid/sign-table prologue is amortized over many row tiles;
//  * optional amdgpu_waves_per_eu (GGML_EXP_MMVQ_GCN2_V2_WPE) to bound the VGPR count (rpl=2 + gate fusion = 4 live vec_dots);
//  * IQ grids and sign tables are filled by a wave-only prologue issued right after the ids loads.
// ---------------------------------------------------------------------------------------------
template <ggml_type type, int ncols_dst, bool has_fusion, int rpl>
__launch_bounds__(ggml_cuda_get_physical_warp_size()*mmvq_dec_nwv(type), GGML_EXP_MMVQ_DEC_WPE) MMVQ_GCN2V2_ATTR
static __global__ void mul_mat_vec_q_gcn2v2(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int lanes_per_blk = qi / vdr;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(lanes_per_blk >= 8 && (lanes_per_blk & (lanes_per_blk - 1)) == 0, "lanes_per_blk must be a power of two >= 8");
    constexpr int bpi = warp_size / lanes_per_blk; // superblocks per iteration of the wave

    // t21: the perm-LUT sign path halves the LDS lookups but costs VALU; it only pays for n=1 (n>=2 shares the lookups among
    // the columns and is VALU-bound), so n>=2 keeps the table path
    constexpr bool dec_lut = mmvq_dec_lut(type, ncols_dst);
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = mmvq_gcn2_get_vec_dot(type, dec_lut);

    const int lane             = threadIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    const int ntiles           = (nrows_x + rpl - 1) / rpl;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    constexpr int nwv = mmvq_dec_nwv(type);
    if constexpr (mmvq_gcn2v2_lds_type(type)) {
        if constexpr (nwv > 1) {
            mmvq_gcn2v2_fill_lds<type, true, dec_lut>(threadIdx.y*warp_size + lane, warp_size*nwv);
            __syncthreads();
        } else {
            mmvq_gcn2v2_fill_lds<type, true, dec_lut>(lane, warp_size);
            mmvq_gcn2v2_wave_sync();
        }
    }

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    ggml_glu_op active_glu;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
    }

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x;
    const int kbx_lane   = lane / lanes_per_blk;
    const int kqs        = vdr * (lane % lanes_per_blk);
    const uint32_t channel_bias = ids ? channel_x : channel_dst;

    float * dst_c = dst + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst;

    const int wave_y = nwv > 1 ? __builtin_amdgcn_readfirstlane((int) threadIdx.y) : 0; // wave-uniform: keeps rows in SGPRs
    for (int tile = blockIdx.x*nwv + wave_y; tile < ntiles; tile += gridDim.x*nwv) {
        int  rows[rpl];
        int  rowc[rpl];
        bool rvalid[rpl];
#pragma unroll
        for (int r = 0; r < rpl; ++r) {
            rows[r]   = tile*rpl + r;
            rvalid[r] = uint32_t(rows[r]) < nrows_x;
            rowc[r]   = rvalid[r] ? rows[r] : int(nrows_x) - 1; // tail row recomputes the last row, result discarded
        }
#if defined(GGML_EXP_MMVQ_DEC_ADDR)
        // t21: row base pointers once per tile; the loop then only adds kbx (the original recomputed
        // kbx_offset + rowc*stride_row_x + kbx -> v_mad_i64_i32 + 64-bit adds per row per iteration)
        // and the pointers advance by bpi blocks per iteration (vec_dot gets kbx = 0 -> constant offsets from one base)
        const char * rowp[rpl];
        [[maybe_unused]] const char * rowp_g[rpl];
#pragma unroll
        for (int r = 0; r < rpl; ++r) {
            const size_t off = ((size_t) (uint32_t) (kbx_offset + rowc[r]*stride_row_x) + (size_t) kbx_lane) * mmvq_dec_blk_bytes<type>();
            rowp[r] = (const char *) vx + off;
            if constexpr (has_fusion) {
                rowp_g[r] = (const char *) vgate + off;
            }
        }
        const int kbx0 = 0;
#endif

        [[maybe_unused]] float x_biases[ncols_dst][rpl]    = {{0.0f}};
        [[maybe_unused]] float gate_biases[ncols_dst][rpl] = {{0.0f}};
        if constexpr (has_fusion) {
            if (lane == 0) {
                if (use_bias) {
                    const float * xb = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst;
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                        for (int r = 0; r < rpl; ++r) {
                            if (rvalid[r]) {
                                x_biases[j][r] = xb[j * stride_col_dst + rows[r]];
                            }
                        }
                    }
                }
                if (use_gate_bias) {
                    const float * gb = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst;
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                        for (int r = 0; r < rpl; ++r) {
                            if (rvalid[r]) {
                                gate_biases[j][r] = gb[j * stride_col_dst + rows[r]];
                            }
                        }
                    }
                }
            }
        }

        float tmp[ncols_dst][rpl] = {{0.0f}};
        float tmp_gate[ncols_dst][rpl] = {{0.0f}};

#if defined(GGML_EXP_MMVQ_DEC_UNROLL) && GGML_EXP_MMVQ_DEC_UNROLL == 2
#pragma unroll 2
#elif defined(GGML_EXP_MMVQ_DEC_UNROLL) && GGML_EXP_MMVQ_DEC_UNROLL == 3
#pragma unroll 3
#endif
        for (int kbx = kbx_lane; kbx < blocks_per_row_x; kbx += bpi) {
            const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

#if defined(GGML_EXP_MMVQ_REUSE3)
            if constexpr (ncols_dst == 3 && !has_fusion && (type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ3_XXS)) {
#pragma unroll
                for (int r=0;r<rpl;++r) {
                    float out[3];
                    if constexpr (type == GGML_TYPE_IQ3_XXS) vec_dot_iq3_xxs_q8_1_v2_3(vx, &y[kby], kbx_offset + rowc[r]*stride_row_x + kbx, kqs, stride_col_y, out);
                    else if constexpr (type == GGML_TYPE_IQ3_S) vec_dot_iq3_s_q8_1_v2_3(vx, &y[kby], kbx_offset + rowc[r]*stride_row_x + kbx, kqs, stride_col_y, out);
                    else if constexpr (type == GGML_TYPE_IQ2_XXS) vec_dot_iq2_xxs_q8_1_v2_3(vx, &y[kby], kbx_offset + rowc[r]*stride_row_x + kbx, kqs, stride_col_y, out);
                    else if constexpr (type == GGML_TYPE_IQ2_XS) vec_dot_iq2_xs_q8_1_v2_3(vx, &y[kby], kbx_offset + rowc[r]*stride_row_x + kbx, kqs, stride_col_y, out);
                    else if constexpr (type == GGML_TYPE_IQ2_S) vec_dot_iq2_s_q8_1_v2_3(vx, &y[kby], kbx_offset + rowc[r]*stride_row_x + kbx, kqs, stride_col_y, out);
                    else if constexpr (type == GGML_TYPE_Q2_K) vec_dot_q2_K_q8_1_3(vx, &y[kby], kbx_offset + rowc[r]*stride_row_x + kbx, kqs, stride_col_y, out);
#pragma unroll
                    for(int j=0;j<3;++j) tmp[j][r] += out[j];
                }
            } else
#endif
            {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int r = 0; r < rpl; ++r) {
#if defined(GGML_EXP_MMVQ_DEC_ADDR)
                    tmp[j][r] += vec_dot_q_cuda(rowp[r], &y[j*stride_col_y + kby], kbx0, kqs);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_gate[j][r] += vec_dot_q_cuda(rowp_g[r], &y[j*stride_col_y + kby], kbx0, kqs);
                        }
                    }
#else
                    tmp[j][r] += vec_dot_q_cuda(
                        vx, &y[j*stride_col_y + kby], kbx_offset + rowc[r]*stride_row_x + kbx, kqs);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_gate[j][r] += vec_dot_q_cuda(
                                vgate, &y[j*stride_col_y + kby], kbx_offset + rowc[r]*stride_row_x + kbx, kqs);
                        }
                    }
#endif
                }
            }
            }
#if defined(GGML_EXP_MMVQ_DEC_ADDR)
#pragma unroll
            for (int r = 0; r < rpl; ++r) {
                rowp[r] += bpi * mmvq_dec_blk_bytes<type>();
                if constexpr (has_fusion) {
                    rowp_g[r] += bpi * mmvq_dec_blk_bytes<type>();
                }
            }
#endif
        }

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int r = 0; r < rpl; ++r) {
                float result = mmvq_gcn2_group_sum(tmp[j][r], 6);
                [[maybe_unused]] float gate_value = 0.0f;
                if constexpr (has_fusion) {
                    if (use_gate) {
                        gate_value = mmvq_gcn2_group_sum(tmp_gate[j][r], 6);
                    }
                }

                if (lane == 0 && rvalid[r]) {
                    if constexpr (has_fusion) {
                        result += x_biases[j][r];
                        if (use_gate) {
                            gate_value += gate_biases[j][r];
                            switch (active_glu) {
                                case GGML_GLU_OP_SWIGLU:
                                    result *= ggml_cuda_op_silu_single(gate_value);
                                    break;
                                case GGML_GLU_OP_GEGLU:
                                    result *= ggml_cuda_op_gelu_single(gate_value);
                                    break;
                                case GGML_GLU_OP_SWIGLU_OAI:
                                    result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                    break;
                                default:
                                    result = result * gate_value;
                                    break;
                            }
                        }
                    }
                    dst_c[j*stride_col_dst + rows[r]] = result;
                }
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, active_glu, gate_bias, x_bias, ids_stride, channel_bias);
    }
}

template <ggml_type type, int c_ncols_dst, int rpl>
static void mul_mat_vec_q_gcn2v2_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            ggml_cuda_kernel_launch(mul_mat_vec_q_gcn2v2<type, c_ncols_dst, true, rpl>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    ggml_cuda_kernel_launch(mul_mat_vec_q_gcn2v2<type, c_ncols_dst, false, rpl>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

static int mmvq_gcn2v2_mask() {
    static const int mask = []() {
        const char * env = getenv("GGML_MMVQ_GCN2_V2");
        return env == nullptr ? 0x3F : (int) std::strtol(env, nullptr, 0);
    }();
    return mask;
}

static int mmvq_gcn2v2_bpc() {
    static const int bpc = []() {
        const char * env = getenv("GGML_MMVQ_GCN2_V2_BPC");
        return env == nullptr ? (int) GGML_EXP_MMVQ_GCN2_V2_BPC : std::atoi(env);
    }();
    return bpc;
}
#endif // defined(GGML_USE_HIP) && defined(GGML_EXP_MMVQ_GCN2) && defined(GGML_EXP_MMVQ_GCN2_IQ3V2)

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id);
    const int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst);
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block>, launch_params,
        vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        ncols_dst, ids_stride);
}

#if defined(GGML_USE_HIP) && defined(GGML_EXP_MMVQ_GCN2)
// runtime kill switch for A/B runs with the same binary: GGML_MMVQ_GCN2=0
static bool mmvq_gcn2_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_MMVQ_GCN2");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return enabled;
}

template <ggml_type type, int c_ncols_dst, int rpl>
static void mul_mat_vec_q_gcn2_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            ggml_cuda_kernel_launch(mul_mat_vec_q_gcn2<type, c_ncols_dst, true, rpl>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    ggml_cuda_kernel_launch(mul_mat_vec_q_gcn2<type, c_ncols_dst, false, rpl>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

// Handles ncols_dst <= 4 (dense or MUL_MAT_ID with 1 token) and MUL_MAT_ID with ncols_dst > 1 (MoE kernel).
template <ggml_type type>
static void mul_mat_vec_q_gcn2_switch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const uint3 nchannels_y_fd, const uint3 channel_ratio_fd, const uint3 sample_ratio_fd,
        const int warp_size, const int ids_stride, cudaStream_t stream) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int lanes_per_blk = qi / vdr;

    const int lpr_log2 = mmvq_gcn2_lpr_log2(lanes_per_blk, ncols_x / qk);
    const int G        = warp_size >> lpr_log2;
    const int rpl      = G == 1 ? GGML_EXP_MMVQ_GCN2_RPL_FULL : GGML_EXP_MMVQ_GCN2_RPL_PACKED;
    const int64_t nblocks = (nrows_x + G*rpl - 1) / (G*rpl);

#if defined(GGML_EXP_MMVQ_GCN2_IQ3V2)
    if constexpr (mmvq_gcn2v2_type(type)) {
        if (lpr_log2 == 6 && !(ids != nullptr && ncols_dst > 1)) {
            // full-wave rows: dedicated kernel (constexpr geometry, persistent tiles)
            constexpr int rpl2 = GGML_EXP_MMVQ_GCN2_V2_RPL;
            constexpr int nwv2 = mmvq_dec_nwv(type);
            const int64_t ntiles = (nrows_x + rpl2 - 1) / rpl2;
            int64_t nb = (ntiles + nwv2 - 1) / nwv2;
            const int bpc = mmvq_gcn2v2_bpc();
            if (bpc > 0) {
                nb = std::min<int64_t>(nb, (int64_t) bpc * ggml_cuda_info().devices[ggml_cuda_get_device()].nsm);
            }
            const dim3 grid2(nb, nchannels_dst, nsamples_dst);
            const dim3 block2(warp_size, nwv2, 1);

#define MMVQ_GCN2V2_CASE(NC) \
            case NC: \
                mul_mat_vec_q_gcn2v2_launch<type, NC, rpl2>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, \
                    stride_col_dst, channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd, \
                    stride_sample_x, stride_sample_y, stride_sample_dst, grid2, block2, ids_stride, stream); \
                break;

            switch (ncols_dst) {
                MMVQ_GCN2V2_CASE(1)
                MMVQ_GCN2V2_CASE(2)
                MMVQ_GCN2V2_CASE(3)
                MMVQ_GCN2V2_CASE(4)
                default:
                    GGML_ABORT("fatal error");
                    break;
            }
#undef MMVQ_GCN2V2_CASE
            GGML_UNUSED(nchannels_y);
            return;
        }
    }
#endif // defined(GGML_EXP_MMVQ_GCN2_IQ3V2)

    if (ids != nullptr && ncols_dst > 1) {
        const dim3 block_nums(nblocks, nchannels_dst);
        const dim3 block_dims(warp_size, ncols_dst);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        if (rpl == 2) {
            ggml_cuda_kernel_launch(mul_mat_vec_q_moe_gcn2<type, 2>, launch_params,
                vx, vy, ids, dst, ncols_x, nchannels_y_fd, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, ids_stride);
        } else {
            ggml_cuda_kernel_launch(mul_mat_vec_q_moe_gcn2<type, 1>, launch_params,
                vx, vy, ids, dst, ncols_x, nchannels_y_fd, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, ids_stride);
        }
        return;
    }

    const dim3 block_nums(nblocks, nchannels_dst, nsamples_dst);
    const dim3 block_dims(warp_size, 1, 1);

#define MMVQ_GCN2_CASE(NC) \
    case NC: \
        if (rpl == 2) { \
            mul_mat_vec_q_gcn2_launch<type, NC, 2>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, \
                stride_col_dst, channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd, \
                stride_sample_x, stride_sample_y, stride_sample_dst, block_nums, block_dims, ids_stride, stream); \
        } else { \
            mul_mat_vec_q_gcn2_launch<type, NC, 1>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x, stride_col_y, \
                stride_col_dst, channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd, \
                stride_sample_x, stride_sample_y, stride_sample_dst, block_nums, block_dims, ids_stride, stream); \
        } \
        break;

    switch (ncols_dst) {
        MMVQ_GCN2_CASE(1)
        MMVQ_GCN2_CASE(2)
        MMVQ_GCN2_CASE(3)
        MMVQ_GCN2_CASE(4)
        default:
            GGML_ABORT("fatal error");
            break;
    }
#undef MMVQ_GCN2_CASE

    GGML_UNUSED(nchannels_y);
}
#endif // defined(GGML_USE_HIP) && defined(GGML_EXP_MMVQ_GCN2)

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
        constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
        constexpr int vdr                   = get_vdr_mmvq(type);
        const int     blocks_per_row_x      = ncols_x / qk;
        const int     blocks_per_iter_1warp = vdr * warp_size / qi;
        const int     nwarps                = calc_nwarps(type, c_ncols_dst, table_id);
        bool          use                   = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

#if defined(GGML_USE_HIP) && defined(GGML_EXP_MMVQ_GCN2) && defined(GGML_EXP_MMVQ_GCN2_IQ3V2)
    // : sub-4-bit i-quants and Q2_K on the gcn2 path, switchable per type at runtime (GGML_MMVQ_GCN2_V2 mask)
    if constexpr (mmvq_gcn2v2_type(type)) {
        if (GGML_CUDA_CC_IS_GCN(cc) && warp_size == 64 && (ncols_dst <= 4 || has_ids) && mmvq_gcn2_enabled() &&
            (mmvq_gcn2v2_mask() & mmvq_gcn2v2_bit(type))) {
            mul_mat_vec_q_gcn2_switch<type>(
                vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst,
                stride_row_x, stride_col_y, stride_col_dst,
                nchannels_y, nchannels_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                nchannels_y_fd, channel_ratio_fd, sample_ratio_fd,
                warp_size, ids_stride, stream);
            return;
        }
    }
#endif
#if defined(GGML_USE_HIP) && defined(GGML_EXP_MMVQ_GCN2)
    if constexpr (mmvq_gcn2_supported_type(type)) {
        if (GGML_CUDA_CC_IS_GCN(cc) && warp_size == 64 && (ncols_dst <= 4 || has_ids) && mmvq_gcn2_enabled()) {
            mul_mat_vec_q_gcn2_switch<type>(
                vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst,
                stride_row_x, stride_col_y, stride_col_dst,
                nchannels_y, nchannels_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                nchannels_y_fd, channel_ratio_fd, sample_ratio_fd,
                warp_size, ids_stride, stream);
            return;
        }
    }
#endif // defined(GGML_USE_HIP) && defined(GGML_EXP_MMVQ_GCN2)

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            constexpr int c_ncols_dst = 1;

            bool use_small_k = should_use_small_k(c_ncols_dst);

            if (use_small_k) {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id, true);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, true>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            } else {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion
#ifdef GGML_EXP_Q8_SHARE
        , const char * src1_q8_1_pre
#endif
        ) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
#ifdef GGML_EXP_Q8_SHARE
#ifdef GGML_EXP_COLA
    if (src1_q8_1_pre == nullptr && !ids) {
        src1_q8_1_pre = ggml_cuda_cola_pending_q8(ctx, src1);
    }
#endif
    ggml_cuda_pool_alloc<char> src1_q8_1_own(ctx.pool());
    const char * src1_q8_1 = src1_q8_1_pre;
    if (src1_q8_1 == nullptr) {
        src1_q8_1_own.alloc(ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1_own.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        src1_q8_1 = src1_q8_1_own.get();
    }
#else
    ggml_cuda_pool_alloc<char> src1_q8_1_pool(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1_pool.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }
    const char * src1_q8_1 = src1_q8_1_pool.get();
#endif

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1, ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);
}

#ifdef GGML_EXP_Q8_SHARE
void ggml_cuda_mul_mat_vec_q_shared_src1(ggml_backend_cuda_context & ctx, ggml_tensor * const * mm, int n) {
    GGML_ASSERT(n >= 1);
    const ggml_tensor * src1 = mm[0]->src[1];
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));

    const int64_t ne10        = src1->ne[0];
    const int64_t ne11        = src1->ne[1];
    const int64_t ne12        = src1->ne[2];
    const int64_t ne13        = src1->ne[3];
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    // same buffer size and quantize call as the non-shared path in ggml_cuda_mul_mat_vec_q
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    const char * q8_ptr = nullptr;
#ifdef GGML_EXP_COLA
    q8_ptr = ggml_cuda_cola_pending_q8(ctx, src1);
#endif
    if (q8_ptr == nullptr) {
        src1_q8_1.alloc(ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(), mm[0]->src[0]->type, ne10,
            src1->nb[1] / sizeof(float), src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float),
            ne10_padded, ne11, ne12, ne13, ctx.stream());
        q8_ptr = src1_q8_1.get();
    }

    for (int k = 0; k < n; ++k) {
        GGML_ASSERT(mm[k]->src[1] == src1);
        ggml_cuda_mul_mat_vec_q(ctx, mm[k]->src[0], src1, nullptr, mm[k], nullptr, q8_ptr);
    }
}
#endif // GGML_EXP_Q8_SHARE

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
