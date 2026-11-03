#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

#ifdef GGML_EXP_COLA
// ---------------------------------------------------------------------------------------------------------------
// ik_llama.cpp keeps the recurrent conv state out of the graph: ssm_conv reads the old state and the new tokens
// directly, writes the per-token snapshots itself (ssm_conv_single_seq_f32_nc4<..., save_steps>) and a second
// kernel writes the final state. Here the same idea is applied to mainline's node chain
//   CONCAT, (VIEW, CPY) x K, SSM_CONV, SILU        (4 + K kernels, 3 + K launches after view elision)
// and, being per-channel independent, it fits in a single kernel: thread (c, s) loads the 3 state values and the
// n_t new values of its channel into registers, computes the n_t outputs (same summation order as ssm_conv_f32),
// and stores the K 3-wide windows. No LDS, no __syncthreads, no cross-thread hazard.
// ---------------------------------------------------------------------------------------------------------------
struct ggml_ikcs_snap {
    float *   dst;       // [3*C, n_s] rows, row stride dst_nb1 (floats)
    int64_t   dst_nb1;
    int       s_idx;     // first column of the 3-wide window inside concat(state, x)
};

struct ggml_ikcs_args {
    const float * st;    // state rows [3, C, n_s]: st[j + c*st_nb1 + s*st_nb2]
    int64_t       st_nb1;
    int64_t       st_nb2;
    const float * x;     // new tokens: x[t*x_nb0 + c*x_nb1 + s*x_nb2]
    int64_t       x_nb0;
    int64_t       x_nb1;
    int64_t       x_nb2;
    const float * w;     // conv weights [4, C]: w[c*w_nb1 + k]
    int64_t       w_nb1;
    float *       out;   // out[c + t*out_nb1 + s*out_nb2]
    int64_t       out_nb1;
    int64_t       out_nb2;
    int           C;
    int           n_snap;
    ggml_ikcs_snap snap[GGML_IKCS_MAX_SNAP];
};

template <int NT>
static __global__ void __launch_bounds__(64) ssm_conv_state_fused_f32(const ggml_ikcs_args a) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= a.C) {
        return;
    }
    const int s = blockIdx.y;

    float x[3 + NT];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        x[j] = a.st[j + (int64_t) c * a.st_nb1 + (int64_t) s * a.st_nb2];
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        x[3 + t] = a.x[(int64_t) t * a.x_nb0 + (int64_t) c * a.x_nb1 + (int64_t) s * a.x_nb2];
    }

    float w[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        w[k] = a.w[(int64_t) c * a.w_nb1 + k];
    }

    // everything the thread needs is in registers: the outputs may alias the inputs element for element
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumf += x[t + j] * w[j];
        }
        a.out[c + (int64_t) t * a.out_nb1 + (int64_t) s * a.out_nb2] = ggml_cuda_op_silu_single(sumf);
    }

    for (int k = 0; k < a.n_snap; ++k) {
        const ggml_ikcs_snap & sn = a.snap[k];
        float * d = sn.dst + (int64_t) s * sn.dst_nb1 + (int64_t) c * 3;
#pragma unroll
        for (int e = 0; e < 3 + NT; ++e) {
            const int j = e - sn.s_idx;
            if (j >= 0 && j < 3) {
                d[j] = x[e];
            }
        }
    }
}

template <int NT>
static void ssm_conv_state_fused_launch(const ggml_ikcs_args & a, int n_s, cudaStream_t stream) {
    const dim3 block(64, 1, 1);
    const dim3 grid((a.C + 63) / 64, n_s, 1);
    ssm_conv_state_fused_f32<NT><<<grid, block, 0, stream>>>(a);
}

void ggml_cuda_ikcs_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * cat, const ggml_tensor * const * cpys,
                           int n_snap, const ggml_tensor * ssm, const ggml_tensor * act) {
    const ggml_tensor * st  = cat->src[0];  // [3, C, n_s]   (get_rows output, reshaped)
    const ggml_tensor * qkv = cat->src[1];  // [n_t, C, n_s] (transposed view of the projection)
    const ggml_tensor * w   = ssm->src[1];  // [4, C]

    GGML_ASSERT(n_snap <= GGML_IKCS_MAX_SNAP);

    ggml_ikcs_args a = {};
    const size_t f = sizeof(float);

    const int n_t = (int) qkv->ne[0];
    const int n_s = (int) cat->ne[2];

    a.st      = (const float *) st->data;
    a.st_nb1  = st->nb[1] / f;
    a.st_nb2  = st->nb[2] / f;
    a.x       = (const float *) qkv->data;
    a.x_nb0   = qkv->nb[0] / f;
    a.x_nb1   = qkv->nb[1] / f;
    a.x_nb2   = qkv->nb[2] / f;
    a.w       = (const float *) w->data;
    a.w_nb1   = w->nb[1] / f;
    a.out     = (float *) act->data;
    a.out_nb1 = act->nb[1] / f;
    a.out_nb2 = act->nb[2] / f;
    a.C       = (int) cat->ne[1];
    a.n_snap  = n_snap;
    for (int k = 0; k < n_snap; ++k) {
        const ggml_tensor * v = cpys[k]->src[0];  // view of cat: window of 3 columns
        const ggml_tensor * d = cpys[k]->src[1];  // cache view [3*C, n_s]
        a.snap[k].dst     = (float *) d->data;
        a.snap[k].dst_nb1 = d->nb[1] / f;
        a.snap[k].s_idx   = (int) (v->view_offs / f);
    }

    cudaStream_t stream = ctx.stream();
    switch (n_t) {
        case 1: ssm_conv_state_fused_launch<1>(a, n_s, stream); break;
        case 2: ssm_conv_state_fused_launch<2>(a, n_s, stream); break;
        case 3: ssm_conv_state_fused_launch<3>(a, n_s, stream); break;
        case 4: ssm_conv_state_fused_launch<4>(a, n_s, stream); break;
        case 5: ssm_conv_state_fused_launch<5>(a, n_s, stream); break;
        case 6: ssm_conv_state_fused_launch<6>(a, n_s, stream); break;
        case 7: ssm_conv_state_fused_launch<7>(a, n_s, stream); break;
        case 8: ssm_conv_state_fused_launch<8>(a, n_s, stream); break;
        default: GGML_ABORT("ikcs: n_t out of range");
    }
}
#endif // GGML_EXP_COLA
