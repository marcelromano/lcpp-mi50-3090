#include "cola.cuh"
#include "convert.cuh"
#include "quantize.cuh"
#include "mmvf.cuh"

#include <cstdio>

#ifdef GGML_EXP_COLA

// Decode-time "glue" fusions between the matmuls of the dense Qwen3.5/3.8 hybrid. The kernels do the same arithmetic
// as the nodes they replace (same expressions, same per-thread partial sums and block reductions, same q8_1 recipe as
// quantize_q8_1); the build uses -funsafe-math-optimizations, so results can still differ from the unfused kernels in the
// last bit (the compiler associates the expressions of each kernel on its own).

int ggml_cuda_cola_mask() {
    static const int mask = [] {
        const char * e = getenv("GGML_COLA");
        return e ? std::atoi(e) : 0xff;
    }();
    return mask;
}

static __device__ __forceinline__ float cola_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __device__ __forceinline__ float cola_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. GDN gates: MUL_MAT(ssm_alpha) + ADD(dt) + SOFTPLUS + MUL(a) and MUL_MAT(ssm_beta) [+ SIGMOID] in one launch.
//    Block r < nh computes row r of ssm_alpha, block nh + r row r of ssm_beta. The dot product replicates
//    mul_mat_vec_f<bf16, float, ncols, 128> (same per-thread column order, same two-level warp reduction), only the
//    loads of U iterations are issued before the multiply-adds (the original waits one HBM latency per iteration).
// ---------------------------------------------------------------------------------------------------------------------

// the kernel was written on gfx906 (warp 64, mul_mat_vec_f block 128). On sm_86 the warp is 32 and the mirror of
// launch_mul_mat_vec_f_cuda gives 256, which is also the geometry that makes the per-thread column order identical to
// mul_mat_vec_f there. Both sizes are instantiated; the choice is the mirror, overridable by env GGML_COLA_GATES_BS
// (128 or 256) for measurement only.

// SIGK: the beta sigmoid is applied inside the kernel. Its destination may alias the activations y (the graph allocator reuses
// them), which other blocks are still reading, so the sigmoids are parked in `scratch` and the last block to finish copies
// them to `beta` (completion counter; the counter returns to 0 for the next launch).
template <int NC, int BS, bool SIGK>
static __global__ void __launch_bounds__(BS) cola_gdn_gates_kernel(
        const nv_bfloat16 * __restrict__ wa, const nv_bfloat16 * __restrict__ wb, const float * __restrict__ y,
        const float * __restrict__ dt, const float * __restrict__ am, float * __restrict__ gate, float * beta,
        float * scratch, unsigned int * counter,
        const int ncols2, const int stride_y2, const int64_t stride_w, const int nh) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int bs        = BS;
    const int tid      = threadIdx.x;
    const int r        = blockIdx.x;
    const bool is_beta = r >= nh;
    const int h        = is_beta ? r - nh : r;

    const int * x2 = (const int *) ((is_beta ? wb : wa) + (int64_t) h * stride_w);
    const float2 * y2 = (const float2 *) y;

    __shared__ float buf_iw[warp_size];
    if (tid < warp_size) {
        buf_iw[tid] = 0.0f;
    }
    __syncthreads();

    float sumf[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        sumf[j] = 0.0f;
    }

    constexpr int U = NC <= 2 ? 10 : (NC <= 4 ? 5 : 2);
    int col2 = tid;
    for (; col2 + (U - 1)*bs < ncols2; col2 += U*bs) {
        int    xv[U];
        float2 yv[U][NC];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            xv[u] = x2[col2 + u*bs];
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                yv[u][j] = y2[j*stride_y2 + col2 + u*bs];
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const float x0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&xv[u])[0]);
            const float x1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&xv[u])[1]);
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                ggml_cuda_mad(sumf[j], x0, yv[u][j].x);
                ggml_cuda_mad(sumf[j], x1, yv[u][j].y);
            }
        }
    }
    for (; col2 < ncols2; col2 += bs) {
        const int tmpx = x2[col2];
        const float x0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]);
        const float x1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]);
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const float2 tmpy = y2[j*stride_y2 + col2];
            ggml_cuda_mad(sumf[j], x0, tmpy.x);
            ggml_cuda_mad(sumf[j], x1, tmpy.y);
        }
    }

#pragma unroll
    for (int j = 0; j < NC; ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
        buf_iw[tid/warp_size] = sumf[j];
        __syncthreads();
        if (tid < warp_size) {
            sumf[j] = buf_iw[tid];
            sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
        }
        __syncthreads();
    }

    float v = 0.0f;
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        if (tid == j) {
            v = sumf[j];
        }
    }
    if (tid < NC) {
        if (is_beta) {
            if constexpr (SIGK) {
                scratch[tid*nh + h] = cola_sigmoid(v);
            } else {
                beta[tid*nh + h] = v;   // raw projection; the (unfused) sigmoid node follows
            }
        } else {
            const float s = v + dt[h];
            gate[tid*nh + h] = cola_softplus(s) * am[h];
        }
    }
    if constexpr (SIGK) {
        __threadfence();
        __syncthreads();
        __shared__ unsigned int s_last;
        if (tid == 0) {
            s_last = atomicAdd(counter, 1u) == gridDim.x - 1 ? 1u : 0u;
        }
        __syncthreads();
        if (s_last) {
            __threadfence();
            const volatile float * vs = scratch;
            for (int e = tid; e < NC*nh; e += bs) {
                beta[e] = vs[e];
            }
            if (tid == 0) {
                atomicExch(counter, 0u);
            }
        }
    }
}

static int cola_mmvf_block_size(int64_t ncols, int warp_size, int cc) {
    // mirror of launch_mul_mat_vec_f_cuda (gfx9 pre-RDNA1: max_block_size 128; CUDA: 256)
    const int64_t max_block_size = (cc > GGML_CUDA_CC_OFFSET_AMD && cc < GGML_CUDA_CC_RDNA1) ? 128 : 256;
    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    for (int64_t bs = 2*warp_size; bs <= max_block_size; bs += warp_size) {
        const int64_t niter = (ncols + 2*bs - 1) / (2*bs);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = bs;
        }
    }
    return (int) block_size_best;
}

// launch geometry of the gates kernel: the mmvf mirror when it is one of the two instantiated sizes (0 = unsupported)
static int cola_gates_bs(int64_t ncols, int warp_size, int cc) {
    static const int forced = [] {
        const char * e = getenv("GGML_COLA_GATES_BS");
        const int v = e ? std::atoi(e) : 0;
        return (v == 128 || v == 256) ? v : 0;
    }();
    if (forced != 0) {
        return forced;
    }
    const int bs = cola_mmvf_block_size(ncols, warp_size, cc);
    return (bs == 128 || bs == 256) ? bs : 0;
}

static bool cola_log_on() {
    static const bool on = [] {
        const char * e = getenv("GGML_COLA_LOG");
        return e ? std::atoi(e) != 0 : false;
    }();
    return on;
}

bool ggml_cuda_cola_log() {
    return cola_log_on();
}

int ggml_cuda_cola_gates_any() {
    static const int mode = [] {
        const char * e = getenv("GGML_COLA_GATES_ANY");
        const int v = e ? std::atoi(e) : 0;
        return (v >= 0 && v <= 2) ? v : 0;
    }();
    return mode;
}

bool ggml_cuda_cola_gates_supported(const ggml_tensor * w, const ggml_tensor * y) {
    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const int cc = ggml_cuda_info().devices[device].cc;
    return w->type == GGML_TYPE_BF16 && y->type == GGML_TYPE_F32 && (warp_size == 64 || warp_size == 32) &&
        w->ne[2] == 1 && w->ne[3] == 1 && y->ne[2] == 1 && y->ne[3] == 1 && y->ne[1] >= 1 && y->ne[1] <= 8 &&
        w->ne[0] == y->ne[0] && (w->ne[0] % 2) == 0 && w->nb[0] == sizeof(nv_bfloat16) && (w->nb[1] % 4) == 0 &&
        y->nb[0] == sizeof(float) && (y->nb[1] % 8) == 0 &&
        cola_gates_bs(w->ne[0], warp_size, cc) != 0;
}

template <int NC, int BS>
static void cola_gates_launch_bs(const nv_bfloat16 * wa, const nv_bfloat16 * wb, const float * y, const float * dt, const float * am,
        float * gate, float * beta, float * scratch, unsigned int * counter, bool sigk, int ncols2, int stride_y2, int64_t stride_w, int nh, cudaStream_t stream) {
    if (sigk) {
        cola_gdn_gates_kernel<NC, BS, true><<<2*nh, BS, 0, stream>>>(wa, wb, y, dt, am, gate, beta, scratch, counter, ncols2, stride_y2, stride_w, nh);
    } else {
        cola_gdn_gates_kernel<NC, BS, false><<<2*nh, BS, 0, stream>>>(wa, wb, y, dt, am, gate, beta, scratch, counter, ncols2, stride_y2, stride_w, nh);
    }
}

template <int NC>
static void cola_gates_launch(int bs, const nv_bfloat16 * wa, const nv_bfloat16 * wb, const float * y, const float * dt, const float * am,
        float * gate, float * beta, float * scratch, unsigned int * counter, bool sigk, int ncols2, int stride_y2, int64_t stride_w, int nh, cudaStream_t stream) {
    if (bs == 256) {
        cola_gates_launch_bs<NC, 256>(wa, wb, y, dt, am, gate, beta, scratch, counter, sigk, ncols2, stride_y2, stride_w, nh, stream);
    } else {
        cola_gates_launch_bs<NC, 128>(wa, wb, y, dt, am, gate, beta, scratch, counter, sigk, ncols2, stride_y2, stride_w, nh, stream);
    }
}

void ggml_cuda_cola_gdn_gates(ggml_backend_cuda_context & ctx,
        const ggml_tensor * mm_a, const ggml_tensor * mm_b, const ggml_tensor * dt, const ggml_tensor * a,
        const ggml_tensor * mul, const ggml_tensor * sig) {
    // sig == nullptr: the beta sigmoid stays a separate node, beta receives the raw projection (mm_b's own output)
    const ggml_tensor * wa = mm_a->src[0];
    const ggml_tensor * wb = mm_b->src[0];
    const ggml_tensor * y  = mm_a->src[1];

    const int nh = (int) mm_a->ne[0];
    const int nc = (int) y->ne[1];
    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const int bs = cola_gates_bs(wa->ne[0], warp_size, ggml_cuda_info().devices[device].cc);
    if (cola_log_on()) {
        static int n_logged = 0;
        if (n_logged < 8) {
            ++n_logged;
            fprintf(stderr, "[cola] gates nh=%d nc=%d bs=%d warp=%d sigk=%d ncols=%d\n", nh, nc, bs, warp_size, sig ? 1 : 0, (int) wa->ne[0]);
            fflush(stderr);
        }
    }
    const int ncols2 = (int) (wa->ne[0] / 2);
    const int stride_y2 = (int) (y->nb[1] / sizeof(float) / 2);
    const int64_t stride_w = wa->nb[1] / sizeof(nv_bfloat16);
    cudaStream_t stream = ctx.stream();

    const nv_bfloat16 * wa_d = (const nv_bfloat16 *) wa->data;
    const nv_bfloat16 * wb_d = (const nv_bfloat16 *) wb->data;
    const float * y_d  = (const float *) y->data;
    const float * dt_d = (const float *) dt->data;
    const float * a_d  = (const float *) a->data;
    float * gate_d = (float *) mul->data;
    float * beta_d = (float *) (sig ? sig->data : mm_b->data);
    float * scratch_d = nullptr;
    unsigned int * counter_d = nullptr;
    if (sig) {
        if (ctx.cola_scratch == nullptr) {
            CUDA_CHECK(cudaMalloc(&ctx.cola_scratch, 4096));
            CUDA_CHECK(cudaMemsetAsync(ctx.cola_scratch, 0, 4096, stream));
        }
        counter_d = (unsigned int *) ctx.cola_scratch;
        scratch_d = (float *) ctx.cola_scratch + 64;   // nh x tokens <= 960 floats (checked by the matcher)
    }

#define COLA_GATES_CASE(N) case N: cola_gates_launch<N>(bs, wa_d, wb_d, y_d, dt_d, a_d, gate_d, beta_d, scratch_d, counter_d, sig != nullptr, ncols2, stride_y2, stride_w, nh, stream); break;
    switch (nc) {
        COLA_GATES_CASE(1) COLA_GATES_CASE(2) COLA_GATES_CASE(3) COLA_GATES_CASE(4)
        COLA_GATES_CASE(5) COLA_GATES_CASE(6) COLA_GATES_CASE(7) COLA_GATES_CASE(8)
        default: GGML_ABORT("cola gates: n_tokens out of range");
    }
#undef COLA_GATES_CASE
}


#endif
