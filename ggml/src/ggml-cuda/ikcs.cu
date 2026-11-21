#include "ssm-conv.cuh"
#include "cola.cuh"
#include "unary.cuh"
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
                           int n_snap, const ggml_tensor * ssm, const ggml_tensor * act, bool tmp_out) {
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
    void * ikcs_scratch = nullptr;
    size_t ikcs_bytes = 0;
    if (tmp_out) {
        // Context-owned scratch, reused only on the main stream. Target
        // and draft contexts must not race on a per-device singleton.
        ikcs_bytes = ggml_nbytes(act);
        if (ctx.ikcs_capacity < ikcs_bytes || ctx.ikcs_scratch == nullptr) {
            if (ctx.ikcs_scratch != nullptr) {
                CUDA_CHECK(cudaFree(ctx.ikcs_scratch));
                ctx.ikcs_scratch = nullptr;
                ctx.ikcs_capacity = 0;
            }
            CUDA_CHECK(cudaMalloc(&ctx.ikcs_scratch, ikcs_bytes));
            ctx.ikcs_capacity = ikcs_bytes;
        }
        ikcs_scratch = ctx.ikcs_scratch;
        a.out = (float *) ikcs_scratch;
    }
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
    if (tmp_out) {
        CUDA_CHECK(cudaMemcpyAsync(act->data, ikcs_scratch, ikcs_bytes, cudaMemcpyDeviceToDevice, stream));
    }
}
#endif // GGML_EXP_COLA
