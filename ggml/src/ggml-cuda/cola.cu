#include "cola.cuh"
#include "convert.cuh"
#include "quantize.cuh"
#include "mmvf.cuh"

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

static __device__ __forceinline__ float cola_silu(float x) {
    return x / (1.0f + expf(-x));
}

#define COLA_Q8_BUF_BYTES (1u << 20)

static void * cola_q8_buffer(ggml_backend_cuda_context & ctx) {
    if (ctx.cola_q8_buf == nullptr) {
        CUDA_CHECK(cudaMalloc(&ctx.cola_q8_buf, COLA_Q8_BUF_BYTES));
    }
    return ctx.cola_q8_buf;
}

// one q8_1 block per 32 lanes, as quantize_q8_1 does (xi: this lane's value, 0 for lanes past the end)
static __device__ __forceinline__ void cola_quant_lane(block_q8_1 * __restrict__ q, const int64_t e, const float xi, const bool valid) {
    float amax = fabsf(xi);
    float sum  = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    if (valid) {
        const float d = amax / 127.0f;
        const int8_t qv = amax == 0.0f ? 0 : roundf(xi / d);
        const int64_t ib  = e / QK8_1;
        const int     iqs = (int) (e % QK8_1);
        q[ib].qs[iqs] = qv;
        if (iqs == 0) {
            q[ib].ds = make_half2(d, sum);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. GDN gates: MUL_MAT(ssm_alpha) + ADD(dt) + SOFTPLUS + MUL(a) and MUL_MAT(ssm_beta) [+ SIGMOID] in one launch.
//    Block r < nh computes row r of ssm_alpha, block nh + r row r of ssm_beta. The dot product replicates
//    mul_mat_vec_f<bf16, float, ncols, 128> (same per-thread column order, same two-level warp reduction), only the
//    loads of U iterations are issued before the multiply-adds (the original waits one HBM latency per iteration).
// ---------------------------------------------------------------------------------------------------------------------

#define COLA_GATES_BS 128

// SIGK: the beta sigmoid is applied inside the kernel. Its destination may alias the activations y (the graph allocator reuses
// them), which other blocks are still reading, so the sigmoids are parked in `scratch` and the last block to finish copies
// them to `beta` (completion counter; the counter returns to 0 for the next launch).
template <int NC, bool SIGK>
static __global__ void __launch_bounds__(COLA_GATES_BS) cola_gdn_gates_kernel(
        const nv_bfloat16 * __restrict__ wa, const nv_bfloat16 * __restrict__ wb, const float * __restrict__ y,
        const float * __restrict__ dt, const float * __restrict__ am, float * __restrict__ gate, float * beta,
        float * scratch, unsigned int * counter,
        const int ncols2, const int stride_y2, const int64_t stride_w, const int nh) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int bs        = COLA_GATES_BS;
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
            for (int e = tid; e < NC*nh; e += COLA_GATES_BS) {
                beta[e] = vs[e];
            }
            if (tid == 0) {
                atomicExch(counter, 0u);
            }
        }
    }
}

static int cola_mmvf_block_size(int64_t ncols, int warp_size) {
    // mirror of launch_mul_mat_vec_f_cuda (gfx9: max_block_size 128)
    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    for (int64_t bs = 2*warp_size; bs <= 128; bs += warp_size) {
        const int64_t niter = (ncols + 2*bs - 1) / (2*bs);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = bs;
        }
    }
    return (int) block_size_best;
}

bool ggml_cuda_cola_gates_supported(const ggml_tensor * w, const ggml_tensor * y) {
    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    return w->type == GGML_TYPE_BF16 && y->type == GGML_TYPE_F32 && warp_size == 64 &&
        w->ne[2] == 1 && w->ne[3] == 1 && y->ne[2] == 1 && y->ne[3] == 1 && y->ne[1] >= 1 && y->ne[1] <= 8 &&
        w->ne[0] == y->ne[0] && (w->ne[0] % 2) == 0 && w->nb[0] == sizeof(nv_bfloat16) && (w->nb[1] % 4) == 0 &&
        y->nb[0] == sizeof(float) && (y->nb[1] % 8) == 0 &&
        cola_mmvf_block_size(w->ne[0], warp_size) == COLA_GATES_BS;
}

template <int NC>
static void cola_gates_launch(const nv_bfloat16 * wa, const nv_bfloat16 * wb, const float * y, const float * dt, const float * am,
        float * gate, float * beta, float * scratch, unsigned int * counter, bool sigk, int ncols2, int stride_y2, int64_t stride_w, int nh, cudaStream_t stream) {
    if (sigk) {
        cola_gdn_gates_kernel<NC, true><<<2*nh, COLA_GATES_BS, 0, stream>>>(wa, wb, y, dt, am, gate, beta, scratch, counter, ncols2, stride_y2, stride_w, nh);
    } else {
        cola_gdn_gates_kernel<NC, false><<<2*nh, COLA_GATES_BS, 0, stream>>>(wa, wb, y, dt, am, gate, beta, scratch, counter, ncols2, stride_y2, stride_w, nh);
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

#define COLA_GATES_CASE(N) case N: cola_gates_launch<N>(wa_d, wb_d, y_d, dt_d, a_d, gate_d, beta_d, scratch_d, counter_d, sig != nullptr, ncols2, stride_y2, stride_w, nh, stream); break;
    switch (nc) {
        COLA_GATES_CASE(1) COLA_GATES_CASE(2) COLA_GATES_CASE(3) COLA_GATES_CASE(4)
        COLA_GATES_CASE(5) COLA_GATES_CASE(6) COLA_GATES_CASE(7) COLA_GATES_CASE(8)
        default: GGML_ABORT("cola gates: n_tokens out of range");
    }
#undef COLA_GATES_CASE
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. [ADD(a, b) ->] RMS_NORM -> MUL(w) + quantize_q8_1. One block per row, the row in registers (NPT columns per thread);
//    block_reduce over per-thread partial sums col = tid + k*BS. Writes the f32 result (other consumers, e.g. the bf16
//    gate projections), optionally the sum (residual stream) and the q8_1 blocks into ctx.cola_q8_buf, where the following
//    mmvq nodes pick them up.
// ---------------------------------------------------------------------------------------------------------------------

template <int BS, int NPT, bool ADD>
static __global__ void __launch_bounds__(BS) cola_rms_q8_kernel(
        const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ sum_out, const float * __restrict__ w,
        float * __restrict__ dst, block_q8_1 * __restrict__ q,
        const int ncols, const int64_t s_a, const int64_t s_b, const int64_t s_sumo, const float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    a += row*s_a;
    if constexpr (ADD) {
        b       += row*s_b;
        sum_out += row*s_sumo;
    }
    dst += (int64_t) row*ncols;
    q   += (int64_t) row*(ncols/QK8_1);

    float xv[NPT];
    float wv[NPT];
    float tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < NPT; ++k) {
        const int col = tid + k*BS;
        float v = 0.0f;
        float wk = 0.0f;
        if (col < ncols) {
            v = a[col];
            if constexpr (ADD) {
                v += b[col];
                sum_out[col] = v;
            }
            wk = w[col];   // issued with the activations: the norm weights are not waited for after the reduction
            tmp += v * v;
        }
        xv[k] = v;
        wv[k] = wk;
    }

    __shared__ float s_sum[32];
    tmp = block_reduce<block_reduce_method::SUM, BS>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

#pragma unroll
    for (int k = 0; k < NPT; ++k) {
        const int col = tid + k*BS;
        float o = 0.0f;
        if (col < ncols) {
            o = scale * xv[k] * wv[k];
            dst[col] = o;
        }
        cola_quant_lane(q, (int64_t) col, o, col < ncols);
    }
}

bool ggml_cuda_cola_rms_q8_supported(const ggml_tensor * rms, const ggml_tensor * mul) {
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    const int64_t ncols = x->ne[0];
    const int64_t nrows = x->ne[1];
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32) {
        return false;
    }
    if (x->ne[2] != 1 || x->ne[3] != 1 || nrows < 1 || nrows > 8 || ncols % MATRIX_ROW_PADDING != 0 || ncols < 1024 || ncols > 8192) {
        return false;
    }
    if (x->nb[0] != sizeof(float) || x->nb[1] % sizeof(float) != 0 || !ggml_is_contiguous(mul) || !ggml_are_same_shape(mul, x) ||
        w->ne[0] != ncols || ggml_nelements(w) != ncols || !ggml_is_contiguous(w)) {
        return false;
    }
    const size_t need = (size_t) nrows * ncols / QK8_1 * sizeof(block_q8_1);
    return need <= COLA_Q8_BUF_BYTES;
}

template <int BS, int NPT, bool ADD>
static void cola_rms_q8_launch(const float * a, const float * b, float * sum_out, const float * w, float * dst, block_q8_1 * q,
        int ncols, int nrows, int64_t s_a, int64_t s_b, int64_t s_sum, float eps, cudaStream_t stream) {
    cola_rms_q8_kernel<BS, NPT, ADD><<<nrows, BS, 0, stream>>>(a, b, sum_out, w, dst, q, ncols, s_a, s_b, s_sum, eps);
}

template <int BS, bool ADD>
static void cola_rms_q8_dispatch(int ncols, const float * a, const float * b, float * sum_out, const float * w, float * dst, block_q8_1 * q,
        int nrows, int64_t s_a, int64_t s_b, int64_t s_sum, float eps, cudaStream_t stream) {
    const int npt = (ncols + BS - 1) / BS;
#define COLA_RMS_CASE(N) case N: cola_rms_q8_launch<BS, N, ADD>(a, b, sum_out, w, dst, q, ncols, nrows, s_a, s_b, s_sum, eps, stream); break;
    switch (npt) {
        COLA_RMS_CASE(1) COLA_RMS_CASE(2) COLA_RMS_CASE(3) COLA_RMS_CASE(4) COLA_RMS_CASE(5) COLA_RMS_CASE(6) COLA_RMS_CASE(7) COLA_RMS_CASE(8)
        COLA_RMS_CASE(9) COLA_RMS_CASE(10) COLA_RMS_CASE(11) COLA_RMS_CASE(12) COLA_RMS_CASE(13) COLA_RMS_CASE(14) COLA_RMS_CASE(15) COLA_RMS_CASE(16)
        COLA_RMS_CASE(17) COLA_RMS_CASE(18) COLA_RMS_CASE(19) COLA_RMS_CASE(20) COLA_RMS_CASE(21) COLA_RMS_CASE(22) COLA_RMS_CASE(23) COLA_RMS_CASE(24)
        COLA_RMS_CASE(25) COLA_RMS_CASE(26) COLA_RMS_CASE(27) COLA_RMS_CASE(28) COLA_RMS_CASE(29) COLA_RMS_CASE(30) COLA_RMS_CASE(31) COLA_RMS_CASE(32)
        default: GGML_ABORT("cola rms_q8: ncols out of range");
    }
#undef COLA_RMS_CASE
}

static int cola_rms_bs() {
    static const int bs = [] {
        const char * e = getenv("GGML_COLA_RMSBS");
        const int v = e ? std::atoi(e) : 1024;
        return (v == 256 || v == 512 || v == 1024) ? v : 1024;
    }();
    return bs;
}

template <bool ADD>
static void cola_rms_q8_run(ggml_backend_cuda_context & ctx, const ggml_tensor * a_t, const ggml_tensor * b_t, const ggml_tensor * sum_t,
        const ggml_tensor * rms, const ggml_tensor * mul) {
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    float eps = 0.0f;
    memcpy(&eps, rms->op_params, sizeof(float));
    const int ncols = (int) a_t->ne[0];
    const int nrows = (int) a_t->ne[1];
    block_q8_1 * q = (block_q8_1 *) cola_q8_buffer(ctx);
    cudaStream_t stream = ctx.stream();
    const float * a_d = (const float *) a_t->data;
    const float * b_d = ADD ? (const float *) b_t->data : nullptr;
    float * sum_d = ADD ? (float *) sum_t->data : nullptr;
    const float * w_d = (const float *) w->data;
    float * dst_d = (float *) mul->data;
    const int64_t s_a   = a_t->nb[1] / sizeof(float);
    const int64_t s_b   = ADD ? b_t->nb[1] / sizeof(float) : 0;
    const int64_t s_sum = ADD ? sum_t->nb[1] / sizeof(float) : 0;
    switch (cola_rms_bs()) {
        case 256:  cola_rms_q8_dispatch<256,  ADD>(ncols, a_d, b_d, sum_d, w_d, dst_d, q, nrows, s_a, s_b, s_sum, eps, stream); break;
        case 1024: cola_rms_q8_dispatch<1024, ADD>(ncols, a_d, b_d, sum_d, w_d, dst_d, q, nrows, s_a, s_b, s_sum, eps, stream); break;
        default:   cola_rms_q8_dispatch<512,  ADD>(ncols, a_d, b_d, sum_d, w_d, dst_d, q, nrows, s_a, s_b, s_sum, eps, stream); break;
    }
    ctx.cola_q8_src1 = mul;
    ctx.cola_q8_ne0  = ncols;
    ctx.cola_q8_ne1  = nrows;
}

void ggml_cuda_cola_rms_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * mul) {
    cola_rms_q8_run<false>(ctx, rms->src[0], nullptr, nullptr, rms, mul);
}

void ggml_cuda_cola_add_rms_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * add, const ggml_tensor * rms, const ggml_tensor * mul) {
    cola_rms_q8_run<true>(ctx, add->src[0], add->src[1], add, rms, mul);
}

const char * ggml_cuda_cola_pending_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * src1) {
    if (ctx.cola_q8_src1 == src1 && ctx.cola_q8_buf != nullptr && src1->ne[0] == ctx.cola_q8_ne0 && src1->ne[1] == ctx.cola_q8_ne1 &&
        src1->ne[2] == 1 && src1->ne[3] == 1) {
        return (const char *) ctx.cola_q8_buf;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. GDN gated norm + quantize: RMS_NORM(x) * w * silu(z) per head row (ncols = head dim, 128), the f32 result and its q8_1
//    blocks (the input of ssm_out). As rms_norm_f32<256, true> followed by unary_gated_op_kernel<silu> (silu(z) * normed)
//    and quantize_q8_1: block of 256 threads, partial sums col = tid (zero for tid >= ncols).
// ---------------------------------------------------------------------------------------------------------------------

static __global__ void __launch_bounds__(256) cola_gnorm_q8_kernel(
        const float * __restrict__ x, const float * __restrict__ z, const float * __restrict__ w, float * __restrict__ dst,
        block_q8_1 * __restrict__ q, const int ncols, const float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t base = (int64_t) row * ncols;

    float v = 0.0f, wk = 0.0f, zk = 0.0f;
    float tmp = 0.0f;
    if (tid < ncols) {
        v  = x[base + tid];
        wk = w[tid];
        zk = z[base + tid];
        tmp += v * v;
    }
    __shared__ float s_sum[32];
    tmp = block_reduce<block_reduce_method::SUM, 256>(tmp, s_sum);
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    float o = 0.0f;
    if (tid < ncols) {
        o = cola_silu(zk) * (scale * v * wk);
        dst[base + tid] = o;
    }
    cola_quant_lane(q, base + tid, o, tid < ncols);
}

bool ggml_cuda_cola_gnorm_q8_supported(const ggml_tensor * rms, const ggml_tensor * w, const ggml_tensor * z, const ggml_tensor * out) {
    const ggml_tensor * x = rms->src[0];
    const int64_t ncols = x->ne[0];
    const int64_t nel   = ggml_nelements(x);
    if (x->type != GGML_TYPE_F32 || z->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32) {
        return false;
    }
    if (ncols > 256 || ncols % QK8_1 != 0 || nel % MATRIX_ROW_PADDING != 0 || nel / MATRIX_ROW_PADDING < 1 ||
        !ggml_is_contiguous(x) || !ggml_is_contiguous(z) || !ggml_is_contiguous(out) || !ggml_are_same_shape(x, z) || !ggml_are_same_shape(x, out) ||
        w->ne[0] != ncols || ggml_nelements(w) != ncols || !ggml_is_contiguous(w)) {
        return false;
    }
    return (size_t) nel / QK8_1 * sizeof(block_q8_1) <= COLA_Q8_BUF_BYTES;
}

void ggml_cuda_cola_gnorm_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * w, const ggml_tensor * z,
        const ggml_tensor * out, const ggml_tensor * consumer_src1) {
    const ggml_tensor * x = rms->src[0];
    float eps = 0.0f;
    memcpy(&eps, rms->op_params, sizeof(float));
    const int ncols = (int) x->ne[0];
    const int rows  = (int) (ggml_nelements(x) / ncols);
    cola_gnorm_q8_kernel<<<rows, 256, 0, ctx.stream()>>>((const float *) x->data, (const float *) z->data, (const float *) w->data,
        (float *) out->data, (block_q8_1 *) cola_q8_buffer(ctx), ncols, eps);
    ctx.cola_q8_src1 = consumer_src1;
    ctx.cola_q8_ne0  = consumer_src1->ne[0];
    ctx.cola_q8_ne1  = consumer_src1->ne[1];
}

// ---------------------------------------------------------------------------------------------------------------------
// 4. SWIGLU (split) + quantize_q8_1: dst = silu(gate) * up (as unary_gated_op_kernel<silu>) and its q8_1 blocks, the input
//    of ffn_down. One thread per element like quantize_q8_1 (32-lane groups).
// ---------------------------------------------------------------------------------------------------------------------

static __global__ void __launch_bounds__(256) cola_swiglu_q8_kernel(
        const float * __restrict__ gate, const float * __restrict__ up, float * __restrict__ dst, block_q8_1 * __restrict__ q, const int64_t n) {
    const int64_t e = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;
    float v = 0.0f;
    if (e < n) {
        v = cola_silu(gate[e]) * up[e];
        dst[e] = v;
    }
    cola_quant_lane(q, e, v, e < n);
}

bool ggml_cuda_cola_swiglu_q8_supported(const ggml_tensor * glu) {
    const ggml_tensor * g = glu->src[0];
    const ggml_tensor * u = glu->src[1];
    if (!u || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || g->type != GGML_TYPE_F32 || u->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(g) || !ggml_is_contiguous(u) || !ggml_is_contiguous(glu) || !ggml_are_same_shape(g, u) || !ggml_are_same_shape(g, glu) ||
        glu->ne[2] != 1 || glu->ne[3] != 1 || glu->ne[1] > 8 || glu->ne[0] % MATRIX_ROW_PADDING != 0) {
        return false;
    }
    return (size_t) ggml_nelements(glu) / QK8_1 * sizeof(block_q8_1) <= COLA_Q8_BUF_BYTES;
}

void ggml_cuda_cola_swiglu_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * glu) {
    const int64_t n = ggml_nelements(glu);
    cola_swiglu_q8_kernel<<<(unsigned int) ((n + 255) / 256), 256, 0, ctx.stream()>>>((const float *) glu->src[0]->data, (const float *) glu->src[1]->data,
        (float *) glu->data, (block_q8_1 *) cola_q8_buffer(ctx), n);
    ctx.cola_q8_src1 = glu;
    ctx.cola_q8_ne0  = glu->ne[0];
    ctx.cola_q8_ne1  = glu->ne[1];
}

#endif // GGML_EXP_COLA
