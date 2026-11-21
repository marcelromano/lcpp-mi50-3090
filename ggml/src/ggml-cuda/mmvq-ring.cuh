#pragma once

#include "common.cuh"
#include "cp-async.cuh"
#include "vecdotq.cuh"

#include <cstdint>

template <ggml_type type>
static constexpr __host__ __device__ bool mmvq_opt_type_supported() {
    return type == GGML_TYPE_IQ3_S ||
           type == GGML_TYPE_IQ3_XXS ||
           type == GGML_TYPE_IQ4_XS ||
           type == GGML_TYPE_Q4_K ||
           type == GGML_TYPE_Q6_K ||
           type == GGML_TYPE_IQ2_S ||
           type == GGML_TYPE_IQ2_XS ||
           type == GGML_TYPE_Q2_K;
}

static __device__ __forceinline__ int mmvq_ksign_prmt(int t) {
#if __CUDA_ARCH__ >= 200
    int r;
    asm("prmt.b32 %0, %1, 0, 0xBA98;" : "=r"(r) : "r"(t));
    return r;
#else
    return __vcmpne4(t & 0x80808080, 0);
#endif
}

static __device__ __forceinline__ int mmvq_ksign_mask_lo(uint32_t s8) {
    return mmvq_ksign_prmt((int)((s8 & 0x0Fu) * 0x10204080u));
}

static __device__ __forceinline__ int mmvq_ksign_mask_hi(uint32_t s8) {
    return mmvq_ksign_prmt((int)((s8 & 0xF0u) * 0x01020408u));
}

// Every grid entry in iq3s_grid, iq3xxs_grid, iq2s_grid and iq2xs_grid has nonzero
// positive bytes (all < 128), so the per-byte two's complement negation (~g + 1)
// never carries across a byte boundary: the single 32-bit add is exact.
static __device__ __forceinline__ int mmvq_ksign_apply(int g, int m) {
    return (g ^ m) + (m & 0x01010101);
}

template <ggml_type type>
static constexpr __host__ __device__ int mmvq_contig_vdr() {
    if constexpr (type == GGML_TYPE_IQ3_S)   return VDR_IQ3_S_Q8_1_MMVQ;
    if constexpr (type == GGML_TYPE_IQ3_XXS) return VDR_IQ3_XXS_Q8_1_MMVQ;
    if constexpr (type == GGML_TYPE_IQ4_XS)  return VDR_IQ4_XS_Q8_1_MMVQ;
    if constexpr (type == GGML_TYPE_Q4_K)    return VDR_Q4_K_Q8_1_MMVQ;
    if constexpr (type == GGML_TYPE_Q6_K)    return VDR_Q6_K_Q8_1_MMVQ;
    if constexpr (type == GGML_TYPE_IQ2_S)   return VDR_IQ2_S_Q8_1_MMVQ;
    if constexpr (type == GGML_TYPE_IQ2_XS)  return VDR_IQ2_XS_Q8_1_MMVQ;
    if constexpr (type == GGML_TYPE_Q2_K)    return VDR_Q2_K_Q8_1_MMVQ;
    return 1;
}

template <ggml_type type>
static constexpr __host__ __device__ int mmvq_opt_max_off() {
    if constexpr (type == GGML_TYPE_Q4_K) {
        return 0;
    } else if constexpr (type == GGML_TYPE_Q2_K) {
        return 12; // bs=84 (4B aligned)
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        return 8;  // bs=136 (8B aligned)
    } else {
        return 14; // bs=110,98,210,82,74 (2B aligned)
    }
}

template <ggml_type type>
static constexpr __host__ __device__ int mmvq_opt_grid_bytes() {
    if constexpr (type == GGML_TYPE_IQ3_S) {
        return 512 * (int)sizeof(uint32_t);
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        return 256 * (int)sizeof(uint32_t);
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        return 1024 * (int)sizeof(uint2);
    } else if constexpr (type == GGML_TYPE_IQ2_XS) {
        return 512 * (int)sizeof(uint2);
    } else {
        return 0;
    }
}

template <ggml_type type, int nwarps>
static constexpr __host__ __device__ int mmvq_contig_ring_sram_stride() {
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = mmvq_contig_vdr<type>();
    constexpr int blocks_per_iter = vdr * nwarps * 32 / qi;
    constexpr int raw_bytes = blocks_per_iter * ggml_cuda_type_traits<type>::bs;
    constexpr int max_off = mmvq_opt_max_off<type>();
    constexpr int nchunks = (max_off + raw_bytes + 15) / 16;
    return (nchunks % 2 == 0) ? 16 * (nchunks + 1) : 16 * nchunks;
}

template <ggml_type type, int ncols_dst, int rows_per_block, int nwarps, int nstages = 2, bool use_smem_grid = false>
static constexpr __host__ __device__ int mmvq_contig_ring_smem_bytes() {
    constexpr int grid_bytes = use_smem_grid ? mmvq_opt_grid_bytes<type>() : 0;
    return grid_bytes +
           nstages * rows_per_block * mmvq_contig_ring_sram_stride<type, nwarps>();
}

template <ggml_type type, int ncols_dst, int rows_per_block, int nwarps, int nstages = 2, bool use_smem_grid = false, int min_blk = 1, int preload = 0, bool flat_load = false>
__launch_bounds__(nwarps * 32, min_blk)
static __global__ void mul_mat_vec_q_contig_ring(
        const void * vx_ptr, const void * vy_ptr, float * dst_ptr,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint3 sample_ratio, const uint32_t stride_sample_x,
        const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {
#if defined(CP_ASYNC_AVAILABLE)
    static_assert(mmvq_opt_type_supported<type>(), "unsupported type");
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int bs  = ggml_cuda_type_traits<type>::bs;
    constexpr int vdr = mmvq_contig_vdr<type>();
    constexpr int warp_size = 32;
    constexpr int nthreads = nwarps * warp_size;
    constexpr int threads_per_blk = qi / vdr;
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;
    constexpr int sram_stride = mmvq_contig_ring_sram_stride<type, nwarps>();
    constexpr int nbytes_stage = rows_per_block * sram_stride;
    constexpr bool unaligned = (type != GGML_TYPE_Q4_K);
    constexpr int grid_bytes = use_smem_grid ? mmvq_opt_grid_bytes<type>() : 0;

    extern __shared__ __align__(16) char smem_base[];
    [[maybe_unused]] uint32_t * s_grid  = (uint32_t *)smem_base;
    [[maybe_unused]] uint2    * s_grid2 = (uint2 *)smem_base;
    char * ring_smem = smem_base + grid_bytes;

    const int tid  = (int)threadIdx.y * warp_size + (int)threadIdx.x;
    const int row0 = (int)blockIdx.x * rows_per_block;
    const int blocks_per_row_x = ncols_x / qk;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y   = channel_dst;
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    const int64_t stride_row_bytes = (int64_t)stride_row_x * bs;
    const char * GGML_CUDA_RESTRICT x =
        (const char *)vx_ptr +
        ((int64_t)sample_x * stride_sample_x + (int64_t)channel_x * stride_channel_x) * bs +
        (int64_t)row0 * stride_row_bytes;
    const block_q8_1 * GGML_CUDA_RESTRICT y =
        (const block_q8_1 *)vy_ptr +
        (int64_t)sample_y * stride_sample_y + (int64_t)channel_y * stride_channel_y;
    float * GGML_CUDA_RESTRICT dst =
        dst_ptr +
        (int64_t)sample_dst * stride_sample_dst + (int64_t)channel_dst * stride_channel_dst + row0;

    if constexpr (use_smem_grid && (type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS)) {
        constexpr int grid_int4 = (type == GGML_TYPE_IQ3_S ? 512 : 256) / 4;
        const int4 * src_grid4 = (const int4 *)(type == GGML_TYPE_IQ3_S ? iq3s_grid : iq3xxs_grid);
        int4 * dst_grid4 = (int4 *)s_grid;
        #pragma unroll
        for (int i = tid; i < grid_int4; i += nthreads) {
            dst_grid4[i] = src_grid4[i];
        }
    } else if constexpr (use_smem_grid && (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_XS)) {
        constexpr int grid_int4 = (type == GGML_TYPE_IQ2_S ? 1024 : 512) / 2;
        const int4 * src_grid4 = (const int4 *)(type == GGML_TYPE_IQ2_S ? iq2s_grid : iq2xs_grid);
        int4 * dst_grid4 = (int4 *)s_grid2;
        #pragma unroll
        for (int i = tid; i < grid_int4; i += nthreads) {
            dst_grid4[i] = src_grid4[i];
        }
    }

    int off_row[rows_per_block];
    if constexpr (unaligned) {
        #pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            const int r_clamped = min(r, (int)nrows_x - 1 - row0);
            off_row[r] = (uintptr_t)(x + (int64_t)r_clamped * stride_row_bytes) & 15;
        }
    } else {
        #pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            off_row[r] = 0;
        }
    }

    auto load_stage = [&](const int stage, const int kbx0) {
        const int nblks = min(blocks_per_iter, blocks_per_row_x - kbx0);
        const int nbytes_row = nblks * bs;
        const unsigned int dst_stage = ggml_cuda_cvta_generic_to_shared(ring_smem + stage * nbytes_stage);
        if constexpr (flat_load && !unaligned && (bs % 16 == 0)) {
            if (nblks == blocks_per_iter && row0 + rows_per_block <= (int)nrows_x) {
                constexpr int nchunks_full = (blocks_per_iter * bs) / 16;
                constexpr int total_chunks = rows_per_block * nchunks_full;
                const char * base = x + (int64_t)kbx0 * bs;
                #pragma unroll
                for (int c = tid; c < total_chunks; c += nthreads) {
                    const int r  = c / nchunks_full;
                    const int ic = c - r * nchunks_full;
                    cp_async_cg_16<preload>(dst_stage + r * sram_stride + 16 * ic,
                                            base + (int64_t)r * stride_row_bytes + 16 * ic);
                }
                return;
            }
        }
        #pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            const int r_clamped = min(r, (int)nrows_x - 1 - row0);
            const char * src = x + (int64_t)r_clamped * stride_row_bytes + (int64_t)kbx0 * bs;
            if constexpr (unaligned) {
                const int off16 = (int)((uintptr_t)src & 15);
                const int nchunks = (off16 + nbytes_row + 15) / 16;
                for (int ic = tid; ic < nchunks; ic += nthreads) {
                    cp_async_cg_16<preload>(dst_stage + r * sram_stride + 16 * ic, src - off16 + 16 * ic);
                }
            } else {
                const int nchunks = (nbytes_row + 15) / 16;
                for (int ic = tid; ic < nchunks; ic += nthreads) {
                    cp_async_cg_16<preload>(dst_stage + r * sram_stride + 16 * ic, src + 16 * ic);
                }
            }
        }
    };

    ggml_cuda_pdl_sync();

    #pragma unroll
    for (int s = 0; s < nstages - 1; ++s) {
        const int kbx0 = s * blocks_per_iter;
        if (kbx0 < blocks_per_row_x) {
            load_stage(s, kbx0);
        }
        cp_async_commit_group();
    }

    float tmp[ncols_dst][rows_per_block] = {{0.0f}};
    const int kbx_local = tid / threads_per_blk;
    const int kqs = vdr * (tid % threads_per_blk);

    for (int iter = 0, kbx0 = 0; kbx0 < blocks_per_row_x; ++iter, kbx0 += blocks_per_iter) {
        const int kbx_next = kbx0 + (nstages - 1) * blocks_per_iter;
        if (kbx_next < blocks_per_row_x) {
            load_stage((iter + nstages - 1) % nstages, kbx_next);
        }
        cp_async_commit_group();
        cp_async_wait_group<nstages - 1>();
        if constexpr (nwarps == 1) { __syncwarp(); } else { __syncthreads(); }

        const int kbx = kbx0 + kbx_local;
        if (kbx < blocks_per_row_x) {
            const char * xs = ring_smem + (iter % nstages) * nbytes_stage;
            const int kby = kbx * (qk / QK8_1);
            const int off_kbx0 = unaligned ? ((kbx0 * bs) & 15) : 0;

            if constexpr (type == GGML_TYPE_IQ3_S) {
                const int bq8_idx = kqs >> 1;
                int   uy[ncols_dst][8];
                float d8y[ncols_dst];
                #pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const block_q8_1 * bq8 = &y[(int64_t)j * stride_col_y + kby + bq8_idx];
                    #pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        uy[j][l] = get_int_b4(bq8->qs, l);
                    }
                    d8y[j] = __low2float(bq8->ds);
                }
                #pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    const int off = unaligned ? ((off_row[r] + off_kbx0) & 15) : 0;
                    const char * bx = xs + r * sram_stride + off + kbx_local * bs;
                    const block_iq3_s * bq3 = (const block_iq3_s *)bx;

                    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, kqs + 0), get_int_b2(bq3->qs, kqs + 1));
                    const uint8_t * qs = (const uint8_t *)&qs_packed;
                    const int qh = bq3->qh[bq8_idx];
                    const uint32_t signs_packed_32 = (uint32_t)get_int_b2(bq3->signs, bq8_idx);
                    const int ls = 1 + 2 * ((bq3->scales[kqs >> 2] >> ((kqs << 1) & 0x04)) & 0x0F);
                    const float d3 = __half2float(bq3->d);

                    int sumi[ncols_dst] = {0};
                    #pragma unroll
                    for (int l0 = 0; l0 < 8; l0 += 2) {
                        const uint32_t idx0 = qs[l0 + 0] | ((qh << (8 - l0)) & 0x100);
                        const uint32_t idx1 = qs[l0 + 1] | ((qh << (7 - l0)) & 0x100);
                        int grid_lo, grid_hi;
                        if constexpr (use_smem_grid) {
                            grid_lo = (int)s_grid[idx0];
                            grid_hi = (int)s_grid[idx1];
                        } else {
                            grid_lo = (int)__ldg(&iq3s_grid[idx0]);
                            grid_hi = (int)__ldg(&iq3s_grid[idx1]);
                        }
                        const uint32_t sp = signs_packed_32 >> (4 * l0);
                        const int grid_l = mmvq_ksign_apply(grid_lo, mmvq_ksign_mask_lo(sp));
                        const int grid_h = mmvq_ksign_apply(grid_hi, mmvq_ksign_mask_hi(sp));

                        #pragma unroll
                        for (int j = 0; j < ncols_dst; ++j) {
                            sumi[j] = ggml_cuda_dp4a(grid_l, uy[j][l0 + 0], sumi[j]);
                            sumi[j] = ggml_cuda_dp4a(grid_h, uy[j][l0 + 1], sumi[j]);
                        }
                    }
                    #pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        tmp[j][r] += (d3 * d8y[j]) * (float)(sumi[j] * ls);
                    }
                }
            } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
                const int bq8_idx = kqs >> 1;
                int   uy[ncols_dst][8];
                float d8y[ncols_dst];
                #pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const block_q8_1 * bq8 = &y[(int64_t)j * stride_col_y + kby + bq8_idx];
                    #pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        uy[j][l] = get_int_b4(bq8->qs, l);
                    }
                    d8y[j] = __low2float(bq8->ds);
                }
                #pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    const int off = unaligned ? ((off_row[r] + off_kbx0) & 15) : 0;
                    const char * bx = xs + r * sram_stride + off + kbx_local * bs;
                    const block_iq3_xxs * bq3 = (const block_iq3_xxs *)bx;

                    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, kqs), get_int_b2(bq3->qs, kqs + 1));
                    const uint8_t * q3 = (const uint8_t *)&q3_packed;
                    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + bq8_idx);
                    const int ls = (int)(aux32 >> 28);
                    const float d3 = __half2float(bq3->d);

                    int sumi[ncols_dst] = {0};
                    #pragma unroll
                    for (int l0 = 0; l0 < 8; l0 += 2) {
                        int grid_lo, grid_hi;
                        if constexpr (use_smem_grid) {
                            grid_lo = (int)s_grid[q3[l0 + 0]];
                            grid_hi = (int)s_grid[q3[l0 + 1]];
                        } else {
                            grid_lo = (int)__ldg(&iq3xxs_grid[q3[l0 + 0]]);
                            grid_hi = (int)__ldg(&iq3xxs_grid[q3[l0 + 1]]);
                        }
                        const uint32_t v = (uint8_t)(aux32 >> (7 * l0 / 2));
                        const uint32_t s8 = v ^ ((__popc(v) & 1) << 7);
                        const int grid_l = mmvq_ksign_apply(grid_lo, mmvq_ksign_mask_lo(s8));
                        const int grid_h = mmvq_ksign_apply(grid_hi, mmvq_ksign_mask_hi(s8));

                        #pragma unroll
                        for (int j = 0; j < ncols_dst; ++j) {
                            sumi[j] = ggml_cuda_dp4a(grid_l, uy[j][l0 + 0], sumi[j]);
                            sumi[j] = ggml_cuda_dp4a(grid_h, uy[j][l0 + 1], sumi[j]);
                        }
                    }
                    #pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        const int s_adj = (ls * sumi[j] + sumi[j] / 2) / 2;
                        tmp[j][r] += (d3 * d8y[j]) * (float)s_adj;
                    }
                }
            } else if constexpr (type == GGML_TYPE_IQ4_XS) {
                const int bq8_idx = kqs >> 2;
                int   uy[ncols_dst][8];
                float d8y[ncols_dst];
                #pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const block_q8_1 * bq8 = &y[(int64_t)j * stride_col_y + kby + bq8_idx];
                    #pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        uy[j][l] = get_int_b4(bq8->qs, l);
                    }
                    d8y[j] = __low2float(bq8->ds);
                }
                #pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    const int off = unaligned ? ((off_row[r] + off_kbx0) & 15) : 0;
                    const char * bx = xs + r * sram_stride + off + kbx_local * bs;
                    const block_iq4_xs * bq4 = (const block_iq4_xs *)bx;

                    const int ls = ((bq4->scales_l[kqs / 8] >> (kqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (kqs / 2)) & 0x03) << 4);
                    const float d4 = __half2float(bq4->d);

                    int sumi[ncols_dst] = {0};
                    #pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const int aux_q4 = get_int_b4(bq4->qs, kqs + l);
                        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
                        #pragma unroll
                        for (int j = 0; j < ncols_dst; ++j) {
                            sumi[j] = ggml_cuda_dp4a(v.x, uy[j][l + 0], sumi[j]);
                            sumi[j] = ggml_cuda_dp4a(v.y, uy[j][l + 4], sumi[j]);
                        }
                    }
                    #pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        tmp[j][r] += (d4 * d8y[j]) * (float)(sumi[j] * (ls - 32));
                    }
                }
            } else if constexpr (type == GGML_TYPE_Q4_K) {
                const int bq8_offset = QR4_K * ((kqs / 2) / (QI8_1 / 2));
                const int q4_lane = (kqs / 2) % 4;
                const int sc_j = bq8_offset / 2;
                const int jm = sc_j & 1;
                const uint32_t hi = (uint32_t)-(int32_t)(sc_j >= 2);

                int   u00[ncols_dst], u01[ncols_dst], u10[ncols_dst], u11[ncols_dst];
                float d8_0[ncols_dst], d8_1[ncols_dst], s8_0[ncols_dst], s8_1[ncols_dst];
                #pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const block_q8_1 * bq8_0 = &y[(int64_t)j * stride_col_y + kby + bq8_offset + 0];
                    const block_q8_1 * bq8_1 = &y[(int64_t)j * stride_col_y + kby + bq8_offset + 1];
                    const int * q8_0 = (const int *)bq8_0->qs + q4_lane;
                    const int * q8_1 = (const int *)bq8_1->qs + q4_lane;
                    u00[j] = q8_0[0];
                    u01[j] = q8_0[4];
                    u10[j] = q8_1[0];
                    u11[j] = q8_1[4];
                    d8_0[j] = __low2float(bq8_0->ds);
                    d8_1[j] = __low2float(bq8_1->ds);
                    s8_0[j] = d8_0[j] * (float)ggml_cuda_dp4a(0x01010101, u01[j], ggml_cuda_dp4a(0x01010101, u00[j], 0));
                    s8_1[j] = d8_1[j] * (float)ggml_cuda_dp4a(0x01010101, u11[j], ggml_cuda_dp4a(0x01010101, u10[j], 0));
                }

                #pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    const char * bx = xs + r * sram_stride + kbx_local * bs;
                    const block_q4_K * bq4 = (const block_q4_K *)bx;

                    const int * q4 = (const int *)(bq4->qs + 16 * bq8_offset + 4 * q4_lane);
                    const int v0 = q4[0];
                    const int v1 = q4[4];

                    const uint16_t * scales = (const uint16_t *)bq4->scales;
                    const uint32_t s0 = scales[jm + 0];
                    const uint32_t s2 = scales[jm + 2];
                    const uint32_t s4 = scales[jm + 4];
                    const uint16_t aux0 = (uint16_t)(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
                    const uint16_t aux1 = (uint16_t)(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
                    const float2 dm4f = __half22float2(bq4->dm);

                    const float sc0 = dm4f.x * (float)(aux0 & 0xFF);
                    const float sc1 = dm4f.x * (float)(aux0 >> 8);
                    const float m0  = dm4f.y * (float)(aux1 & 0xFF);
                    const float m1  = dm4f.y * (float)(aux1 >> 8);

                    const int v0_lo = (v0 >> 0) & 0x0F0F0F0F;
                    const int v1_lo = (v1 >> 0) & 0x0F0F0F0F;
                    const int v0_hi = (v0 >> 4) & 0x0F0F0F0F;
                    const int v1_hi = (v1 >> 4) & 0x0F0F0F0F;

                    #pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        const int dot1_0 = ggml_cuda_dp4a(v1_lo, u01[j], ggml_cuda_dp4a(v0_lo, u00[j], 0));
                        const int dot1_1 = ggml_cuda_dp4a(v1_hi, u11[j], ggml_cuda_dp4a(v0_hi, u10[j], 0));
                        tmp[j][r] += (sc0 * (d8_0[j] * (float)dot1_0) + sc1 * (d8_1[j] * (float)dot1_1))
                                   - (m0 * s8_0[j] + m1 * s8_1[j]);
                    }
                }
            } else if constexpr (type == GGML_TYPE_Q6_K) {
                const int bq8_offset = 2 * QR6_K * (kqs / (QI6_K / 2)) + (kqs % (QI6_K / 2)) / (QI6_K / 4);
                const int scale_offset = (QI6_K / 4) * (kqs / (QI6_K / 2)) + (kqs % (QI6_K / 2)) / (QI6_K / 8);
                const int vh_shift = 2 * ((kqs % (QI6_K / 2)) / (QI6_K / 4));
                const int qh_idx = (QI6_K / 4) * (kqs / (QI6_K / 2)) + kqs % (QI6_K / 4);
                const int kqs8 = kqs % QI8_1;

                int   uy[ncols_dst][4];
                float d8y[ncols_dst][2];
                int   s8_bias[ncols_dst][2];
                #pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const block_q8_1 * bq8 = &y[(int64_t)j * stride_col_y + kby + bq8_offset + 2 * i];
                        uy[j][2 * i + 0] = get_int_b4(bq8->qs, kqs8 + 0);
                        uy[j][2 * i + 1] = get_int_b4(bq8->qs, kqs8 + 1);
                        d8y[j][i] = __low2float(bq8->ds);
                        int b = ggml_cuda_dp4a(0xE0E0E0E0, uy[j][2 * i + 0], 0);
                        s8_bias[j][i] = ggml_cuda_dp4a(0xE0E0E0E0, uy[j][2 * i + 1], b);
                    }
                }

                #pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    const int off = unaligned ? ((off_row[r] + off_kbx0) & 15) : 0;
                    const char * bx = xs + r * sram_stride + off + kbx_local * bs;
                    const block_q6_K * bq6 = (const block_q6_K *)bx;

                    const int vl[2] = {get_int_b2(bq6->ql, kqs + 0), get_int_b2(bq6->ql, kqs + 1)};
                    const int vh[2] = {get_int_b2(bq6->qh, qh_idx + 0) >> vh_shift,
                                       get_int_b2(bq6->qh, qh_idx + 1) >> vh_shift};
                    const int sc0 = (int)bq6->scales[scale_offset + 0];
                    const int sc1 = (int)bq6->scales[scale_offset + 4];
                    const float d6 = __half2float(bq6->d);

                    #pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        int sumi0 = s8_bias[j][0];
                        int sumi1 = s8_bias[j][1];
                        #pragma unroll
                        for (int l = 0; l < 2; ++l) {
                            const int vil0 = (vl[l] >> 0) & 0x0F0F0F0F;
                            const int vih0 = ((vh[l] >> 0) << 4) & 0x30303030;
                            sumi0 = ggml_cuda_dp4a(vil0 | vih0, uy[j][0 + l], sumi0);

                            const int vil1 = (vl[l] >> 4) & 0x0F0F0F0F;
                            const int vih1 = ((vh[l] >> 4) << 4) & 0x30303030;
                            sumi1 = ggml_cuda_dp4a(vil1 | vih1, uy[j][2 + l], sumi1);
                        }
                        const float sumf = d8y[j][0] * (float)(sumi0 * sc0) + d8y[j][1] * (float)(sumi1 * sc1);
                        tmp[j][r] += d6 * sumf;
                    }
                }
            } else if constexpr (type == GGML_TYPE_IQ2_S) {
                const int bq8_idx = kqs >> 1;
                int   uy[ncols_dst][8];
                float d8y[ncols_dst];
                #pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const block_q8_1 * bq8 = &y[(int64_t)j * stride_col_y + kby + bq8_idx];
                    #pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        uy[j][l] = get_int_b4(bq8->qs, l);
                    }
                    d8y[j] = __low2float(bq8->ds);
                }
                #pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    const int off = unaligned ? ((off_row[r] + off_kbx0) & 15) : 0;
                    const char * bx = xs + r * sram_stride + off + kbx_local * bs;
                    const block_iq2_s * bq2 = (const block_iq2_s *)bx;

                    const int qs_packed = get_int_b2(bq2->qs, bq8_idx);
                    const uint8_t * qs = (const uint8_t *)&qs_packed;
                    const int qh = bq2->qh[bq8_idx];
                    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + bq8_idx);
                    const uint8_t * signs_packed_8 = (const uint8_t *)&signs_packed_32;
                    const int ls0 = bq2->scales[bq8_idx] & 0x0F;
                    const int ls1 = bq2->scales[bq8_idx] >> 4;
                    const float d2 = __half2float(bq2->d);

                    int sumi0[ncols_dst] = {0};
                    int sumi1[ncols_dst] = {0};
                    #pragma unroll
                    for (int l0 = 0; l0 < 8; l0 += 2) {
                        const uint32_t idx = qs[l0 / 2] | ((qh << (8 - l0)) & 0x300);
                        uint2 grid_pos;
                        if constexpr (use_smem_grid) {
                            grid_pos = s_grid2[idx];
                        } else {
                            grid_pos = __ldg((const uint2 *)iq2s_grid + idx);
                        }
                        const uint32_t sp8 = signs_packed_8[l0 / 2];
                        const int grid_l = mmvq_ksign_apply((int)grid_pos.x, mmvq_ksign_mask_lo(sp8));
                        const int grid_h = mmvq_ksign_apply((int)grid_pos.y, mmvq_ksign_mask_hi(sp8));

                        #pragma unroll
                        for (int j = 0; j < ncols_dst; ++j) {
                            if (l0 < 4) {
                                sumi0[j] = ggml_cuda_dp4a(grid_l, uy[j][l0 + 0], sumi0[j]);
                                sumi0[j] = ggml_cuda_dp4a(grid_h, uy[j][l0 + 1], sumi0[j]);
                            } else {
                                sumi1[j] = ggml_cuda_dp4a(grid_l, uy[j][l0 + 0], sumi1[j]);
                                sumi1[j] = ggml_cuda_dp4a(grid_h, uy[j][l0 + 1], sumi1[j]);
                            }
                        }
                    }
                    #pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        const int sumi = (sumi0[j] * ls0 + sumi1[j] * ls1 + (sumi0[j] + sumi1[j]) / 2) / 4;
                        tmp[j][r] += (d2 * d8y[j]) * (float)sumi;
                    }
                }
            } else if constexpr (type == GGML_TYPE_IQ2_XS) {
                const int bq8_idx = kqs >> 1;
                int   uy[ncols_dst][8];
                float d8y[ncols_dst];
                #pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const block_q8_1 * bq8 = &y[(int64_t)j * stride_col_y + kby + bq8_idx];
                    #pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        uy[j][l] = get_int_b4(bq8->qs, l);
                    }
                    d8y[j] = __low2float(bq8->ds);
                }
                #pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    const int off = unaligned ? ((off_row[r] + off_kbx0) & 15) : 0;
                    const char * bx = xs + r * sram_stride + off + kbx_local * bs;
                    const block_iq2_xs * bq2 = (const block_iq2_xs *)bx;

                    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, kqs + 0), get_int_b2(bq2->qs, kqs + 1));
                    const uint16_t * q2 = (const uint16_t *)&q2_packed;
                    const int ls0 = bq2->scales[bq8_idx] & 0x0F;
                    const int ls1 = bq2->scales[bq8_idx] >> 4;
                    const float d2 = __half2float(bq2->d);

                    int sumi0[ncols_dst] = {0};
                    int sumi1[ncols_dst] = {0};
                    #pragma unroll
                    for (int l0 = 0; l0 < 8; l0 += 2) {
                        const uint32_t idx = q2[l0 / 2] & 0x1FF;
                        uint2 grid_pos;
                        if constexpr (use_smem_grid) {
                            grid_pos = s_grid2[idx];
                        } else {
                            grid_pos = __ldg((const uint2 *)iq2xs_grid + idx);
                        }
                        const uint32_t v = q2[l0 / 2] >> 9;
                        const uint32_t s8 = v ^ ((__popc(v) & 1) << 7);
                        const int grid_l = mmvq_ksign_apply((int)grid_pos.x, mmvq_ksign_mask_lo(s8));
                        const int grid_h = mmvq_ksign_apply((int)grid_pos.y, mmvq_ksign_mask_hi(s8));

                        #pragma unroll
                        for (int j = 0; j < ncols_dst; ++j) {
                            if (l0 < 4) {
                                sumi0[j] = ggml_cuda_dp4a(grid_l, uy[j][l0 + 0], sumi0[j]);
                                sumi0[j] = ggml_cuda_dp4a(grid_h, uy[j][l0 + 1], sumi0[j]);
                            } else {
                                sumi1[j] = ggml_cuda_dp4a(grid_l, uy[j][l0 + 0], sumi1[j]);
                                sumi1[j] = ggml_cuda_dp4a(grid_h, uy[j][l0 + 1], sumi1[j]);
                            }
                        }
                    }
                    #pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        const int sumi = (sumi0[j] * ls0 + sumi1[j] * ls1 + (sumi0[j] + sumi1[j]) / 2) / 4;
                        tmp[j][r] += (d2 * d8y[j]) * (float)sumi;
                    }
                }
            } else if constexpr (type == GGML_TYPE_Q2_K) {
                const int bq8_offset = QR2_K * (kqs / QI8_1);
                const int iqs8 = kqs % QI8_1;
                const int sc_shift = 8 * (iqs8 / (QI8_1 / 2));

                int   uy[ncols_dst][8];
                float d8y[ncols_dst][4];
                float s8y[ncols_dst][4];
                #pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const block_q8_1 * bq8 = &y[(int64_t)j * stride_col_y + kby + bq8_offset + i];
                        uy[j][2 * i + 0] = get_int_b4(bq8->qs, iqs8 + 0);
                        uy[j][2 * i + 1] = get_int_b4(bq8->qs, iqs8 + 1);
                        d8y[j][i] = __low2float(bq8->ds);
                        const int sm = ggml_cuda_dp4a(0x01010101, uy[j][2 * i + 1], ggml_cuda_dp4a(0x01010101, uy[j][2 * i + 0], 0));
                        s8y[j][i] = d8y[j][i] * (float)sm;
                    }
                }

                #pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    const int off = unaligned ? ((off_row[r] + off_kbx0) & 15) : 0;
                    const char * bx = xs + r * sram_stride + off + kbx_local * bs;
                    const block_q2_K * bq2 = (const block_q2_K *)bx;

                    const int v0 = get_int_b4(bq2->qs, kqs + 0);
                    const int v1 = get_int_b4(bq2->qs, kqs + 1);
                    const int scales0 = get_int_b4(bq2->scales, 2 * (kqs / QI8_1) + 0) >> sc_shift;
                    const int scales1 = get_int_b4(bq2->scales, 2 * (kqs / QI8_1) + 1) >> sc_shift;
                    const float2 dm2f = __half22float2(bq2->dm);

                    float sumf_d[ncols_dst] = {0.0f};
                    float sumf_m[ncols_dst] = {0.0f};
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int sc = ((i < 2) ? scales0 : scales1) >> (16 * (i & 1));
                        const int sc_d = sc & 0xF;
                        const int sc_m = (sc >> 4) & 0xF;
                        const int vi0 = (v0 >> (2 * i)) & 0x03030303;
                        const int vi1 = (v1 >> (2 * i)) & 0x03030303;
                        #pragma unroll
                        for (int j = 0; j < ncols_dst; ++j) {
                            const int sumi_d = ggml_cuda_dp4a(vi1, uy[j][2 * i + 1], ggml_cuda_dp4a(vi0, uy[j][2 * i + 0], 0));
                            sumf_d[j] += d8y[j][i] * (float)(sumi_d * sc_d);
                            sumf_m[j] += s8y[j][i] * (float)sc_m;
                        }
                    }
                    #pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        tmp[j][r] += dm2f.x * sumf_d[j] - dm2f.y * sumf_m[j];
                    }
                }
            }
        }
        if constexpr (nwarps == 1) { __syncwarp(); } else { __syncthreads(); }
    }
    cp_async_wait_all();

    ggml_cuda_pdl_lc();

    #pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        #pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            tmp[j][r] = warp_reduce_sum<warp_size>(tmp[j][r]);
        }
    }

    if constexpr (nwarps == 1) {
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                #pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    if (row0 + r < (int)nrows_x) {
                        dst[(int64_t)j * stride_col_dst + r] = tmp[j][r];
                    }
                }
            }
        }
    } else {
        float * buf_sum = (float *)ring_smem;
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                #pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    buf_sum[((int)threadIdx.y * ncols_dst + j) * rows_per_block + r] = tmp[j][r];
                }
            }
        }
        __syncthreads();

        if (threadIdx.y == 0 && (int)threadIdx.x < ncols_dst * rows_per_block) {
            const int j = (int)threadIdx.x / rows_per_block;
            const int r = (int)threadIdx.x % rows_per_block;
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < nwarps; ++w) {
                acc += buf_sum[(w * ncols_dst + j) * rows_per_block + r];
            }
            if (row0 + r < (int)nrows_x) {
                dst[(int64_t)j * stride_col_dst + r] = acc;
            }
        }
    }
#else
    GGML_UNUSED_VARS(vx_ptr, vy_ptr, dst_ptr, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    NO_DEVICE_CODE;
#endif
}
