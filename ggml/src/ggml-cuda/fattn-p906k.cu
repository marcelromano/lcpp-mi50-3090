// p906k: gfx906 (wave64) prompt-attention kernel for D=256, K q8_0, V q4_0/q8_0, GQA head groups.
// Opt-in only (GGML_HIP_FA_P906 >= 100). Production path untouched when unset.
//
// Design (one CTA = 4 waves x 64 lanes, 32 Q columns = NC1 queries x NC2 heads sharing one KV head):
//   * pre-passes: Q*scale -> int8 per 32-block (+f32 scale) in a global row of 288 B; K q8_0 -> rows of 256 int8 +
//     8 f32 scales (288 B); V -> f16 "pair-interleaved" rows (half2(V[2j][d], V[2j+1][d])).
//   * KQ: lane = KV row (64 rows/tile), wave = 8 Q columns. K row block from LDS (2x ds_read_b128), Q from
//     wave-uniform scalar loads (s_load) -> v_dot4_i32_i8 with an SGPR operand: no LDS traffic for Q.
//   * softmax per tile in f32 (online, same FATTN_KQ_MAX_OFFSET convention as the tile kernel), P stored as
//     half pairs in a wave-private LDS strip.
//   * PV: lane = 4 consecutive d, wave = the same 8 columns; v_dot2_f32_f16 over KV pairs, f32 accumulators.
//   * KV split / combine / KV_max reuse launch_fattn (parallel_blocks + flash_attn_combine_results).
#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-p906k.cuh"
#include <cstdlib>
#include <cstdio>

#ifdef GGML_USE_HIP

#define P906K_NB   64   // KV positions per tile
#define P906K_KRS  76   // LDS words per K row (72 used; 19 quads -> conflict-free b128 rows)
#define P906K_QW   72   // words per Q/K global row (64 int + 8 f32)

typedef _Float16 p906_f16x2 __attribute__((ext_vector_type(2)));

static __device__ __forceinline__ float p906_fdot2(const half2 a, const half2 b, const float c) {
    return __builtin_amdgcn_fdot2(reinterpret_cast<const p906_f16x2 &>(a), reinterpret_cast<const p906_f16x2 &>(b), c, false);
}

// ---- pre-passes ---------------------------------------------------------------------------------------------
// Q layout: row r = ((s*(ne02/NC2) + z)*ne01p + qp)*NC2 + h  (ne01p = ne01 padded to NC1; qp >= ne01 wraps like
// the tile kernel's fastmodulo). A CTA's 32 columns are 32 consecutive rows and a wave's 8 columns 8 consecutive
// rows -> one uniform base pointer. Quants: out_q[r*64 + b*8 + t]; scales: out_d[(r/8)*64 + b*8 + r%8] so that the
// 8 scales of a wave for block b are one contiguous s_load_dwordx8.
static __global__ void p906k_q_quant(const char * __restrict__ Q, int * __restrict__ out_q, float * __restrict__ out_d,
        const float scale, const int NC2, const int ql, const int ne01, const int ne01p, const int ne02, const int ne03,
        const int64_t nb01, const int64_t nb02, const int64_t nb03) {
    const int64_t gid = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t nrows = int64_t(ne01p)*ne02*ne03;
    if (gid >= nrows*8) {
        return;
    }
    const int b = gid % 8;
    const int64_t row = gid / 8;
    const int h  = row % NC2;
    const int qp = (row / NC2) % ne01p;
    const int64_t sz = row / (int64_t(NC2)*ne01p);     // s*(ne02/NC2) + z
    const int z  = sz % (ne02/NC2);
    const int s  = sz / (ne02/NC2);
    const int q  = qp % ne01;
    const int hh = z*NC2 + h;
    const float4 * src = (const float4 *) (Q + s*nb03 + hh*nb02 + q*nb01 + b*32*sizeof(float));
    float x[32];
    float amax = 0.0f;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float4 v = src[t];
        x[4*t+0] = v.x*scale; x[4*t+1] = v.y*scale; x[4*t+2] = v.z*scale; x[4*t+3] = v.w*scale;
    }
#pragma unroll
    for (int t = 0; t < 32; ++t) {
        amax = fmaxf(amax, fabsf(x[t]));
    }
    const float d  = amax / 127.0f;
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    int * dst = ql ? out_q + (row/8)*512 + (row%8)*8 : out_q + row*64;   // ql: [row/8][b][row%8][8]
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const int q0 = (int) roundf(x[4*t+0]*id) & 0xFF;
        const int q1 = (int) roundf(x[4*t+1]*id) & 0xFF;
        const int q2 = (int) roundf(x[4*t+2]*id) & 0xFF;
        const int q3 = (int) roundf(x[4*t+3]*id) & 0xFF;
        dst[(ql ? b*64 : b*8) + t] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
    }
    out_d[(row/8)*64 + b*8 + row%8] = d;
}

static __global__ void p906k_k_split(const char * __restrict__ K, int * __restrict__ out,
        const int ne11, const int ne12, const int64_t nb11, const int64_t nb12, const int64_t nb13) {
    const int i1 = blockIdx.x*32 + threadIdx.x/8;
    const int b  = threadIdx.x % 8;
    if (i1 >= ne11) {
        return;
    }
    const block_q8_0 * src = (const block_q8_0 *) (K + blockIdx.z*nb13 + blockIdx.y*nb12 + int64_t(i1)*nb11) + b;
    int * row = out + ((int64_t(blockIdx.z)*ne12 + blockIdx.y)*ne11 + i1)*P906K_QW;
    int q[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        q[t] = get_int_b2(src->qs, t);
    }
    *(int4 *) (row + b*8 + 0) = make_int4(q[0], q[1], q[2], q[3]);
    *(int4 *) (row + b*8 + 4) = make_int4(q[4], q[5], q[6], q[7]);
    ((float *) row)[64 + b] = __half2float(src->d);
}

template <ggml_type type_V>
static __device__ __forceinline__ void p906k_deq_block(const char * __restrict__ blk, float * x) {
    if constexpr (type_V == GGML_TYPE_Q4_0) {
        const block_q4_0 * bq = (const block_q4_0 *) blk;
        const float d = __half2float(bq->d);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int v = bq->qs[j];
            x[j]      = ((v & 0xF) - 8)*d;
            x[j + 16] = ((v >>  4) - 8)*d;
        }
    } else {
        const block_q8_0 * bq = (const block_q8_0 *) blk;
        const float d = __half2float(bq->d);
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            x[j] = bq->qs[j]*d;
        }
    }
}

// out: per (s, hk) ne11/2 pair-rows of 256 half2: half2(V[2j][d], V[2j+1][d])
template <ggml_type type_V>
static __global__ void p906k_v_pairs(const char * __restrict__ V, half2 * __restrict__ out,
        const int ne11, const int ne12, const int64_t nb21, const int64_t nb22, const int64_t nb23) {
    const int jp = blockIdx.x*32 + threadIdx.x/8;   // pair index
    const int b  = threadIdx.x % 8;                  // 32-wide block of d
    if (2*jp >= ne11) {
        return;
    }
    constexpr int bs = type_V == GGML_TYPE_Q4_0 ? sizeof(block_q4_0) : sizeof(block_q8_0);
    const char * base = V + blockIdx.z*nb23 + blockIdx.y*nb22 + int64_t(b)*bs;
    float x0[32], x1[32];
    p906k_deq_block<type_V>(base + int64_t(2*jp + 0)*nb21, x0);
    p906k_deq_block<type_V>(base + int64_t(2*jp + 1)*nb21, x1);
    half2 * dst = out + ((int64_t(blockIdx.z)*ne12 + blockIdx.y)*(ne11/2) + jp)*256 + b*32;
#pragma unroll
    for (int t = 0; t < 32; t += 4) {
        half2 h[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            h[u] = make_half2(x0[t + u], x1[t + u]);
        }
        *(int4 *) (dst + t) = *(const int4 *) h;
    }
}

// ---- main kernel ---------------------------------------------------------------------------------------------
template <int NC1, int NC2, int QL>
__launch_bounds__(256, 2)
static __global__ void p906k_main(
        const char * __restrict__ Q_ptr, const char * __restrict__ K_ptr, const char * __restrict__ V_ptr,
        const char * __restrict__ mask_ptr, const char * __restrict__ sinks_ptr, const int * __restrict__ KV_max,
        const int * __restrict__ KV_rows,
        float * __restrict__ dst, float2 * __restrict__ dst_meta,
        const float scale, const float max_bias, const float m0, const float m1, const uint32_t n_head_log2, const float logit_softcap,
        const int32_t ne00, const uint3 ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    static_assert(NC1*NC2 == 32, "p906k: 32 columns per CTA");
    constexpr int NB = P906K_NB;

    __shared__ __align__(16) int   Ks[NB*P906K_KRS];        // K tile (int8 rows + scales); reused for V pair rows
    __shared__ __align__(16) half  Ps[4][NB/2][8][2];        // per wave: P pairs [kv pair][col][2]

    const int lane = threadIdx.x;
    const int w    = __builtin_amdgcn_readfirstlane(threadIdx.y);
    const int tid  = threadIdx.y*64 + lane;

    const int ne01v    = ne01.z;
    const int col_Q_0  = blockIdx.x*NC1;
    const int sequence = blockIdx.z / (ne02/NC2);
    const int head0    = blockIdx.z*NC2 - sequence*ne02;
    const int gqa      = ne02/ne12;
    const int hk       = head0/gqa;

    // Q rows (uniform per wave): column jc = 8*w + c -> query j = jc/NC2, head = head0 + jc%NC2.
    // Q_ptr = int8 quants; the f32 block scales follow after nqrows*64 ints (see p906k_launch).
    const int ne01p = ((ne01v + NC1 - 1)/NC1)*NC1;
    const int64_t nqrows = int64_t(ne01p)*ne02*ne03;
    const int64_t row0 = (int64_t(blockIdx.z)*ne01p + col_Q_0)*NC2 + 8*w;
    const int   * __restrict__ Qw  = (const int   *) Q_ptr + (QL ? (row0/8)*512 : row0*64);
    const float * __restrict__ Qdw = (const float *) ((const int *) Q_ptr + nqrows*64) + (row0/8)*64;

    const char  * __restrict__ Kg = K_ptr + nb13*sequence + int64_t(nb12)*hk;     // rows of nb11 (=288) bytes
    const char  * __restrict__ Vg = V_ptr + nb23*sequence + int64_t(nb22)*hk;     // pair rows of 2*nb21 (=1024) bytes
    const half  * __restrict__ maskh = (const half *) (mask_ptr + nb33*(sequence % ne33));
    const int stride_mask = nb31 / sizeof(half);

    // queries handled by this wave: NC2 >= 8 -> one query, NC2 == 2 -> 4 queries (cols c, c+1 share)
    constexpr int QPW = 8/NC2 > 0 ? 8/NC2 : 1;
    int qmask[QPW];
#pragma unroll
    for (int jj = 0; jj < QPW; ++jj) {
        qmask[jj] = fastmodulo(col_Q_0 + (8*w)/NC2 + jj, ne01);
    }

    float O[8][4];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            O[c][t] = 0.0f;
        }
    }
    float m[8], l[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        m[c] = -FLT_MAX/2.0f;
        l[c] = 0.0f;
    }

    const int k_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;

    for (int k0 = blockIdx.y*NB; k0 < k_max; k0 += gridDim.y*NB) {
        // ---- K tile -> LDS: 64 rows x 18 int4
#pragma unroll
        for (int i0 = 0; i0 < NB*18; i0 += 256) {
            const int idx = i0 + tid;
            if (idx < NB*18) {
                const int r = idx / 18;
                const int c = idx % 18;
                const int4 v = *(const int4 *) (Kg + int64_t(k0 + r)*nb11 + c*16);
                *(int4 *) (Ks + r*P906K_KRS + c*4) = v;
            }
        }
        __syncthreads();

        // ---- KQ: lane = KV row, 8 columns per wave
        float s[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            s[c] = 0.0f;
        }
        const int * Kr = Ks + lane*P906K_KRS;
#pragma unroll 1
        for (int b = 0; b < 8; ++b) {
            const int4 k0v = *(const int4 *) (Kr + b*8 + 0);
            const int4 k1v = *(const int4 *) (Kr + b*8 + 4);
            const float kd = ((const float *) Kr)[64 + b];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const int * qb = QL ? Qw + b*64 + c*8 : Qw + c*64 + b*8;
                int acc = 0;
                acc = ggml_cuda_dp4a(k0v.x, qb[0], acc);
                acc = ggml_cuda_dp4a(k0v.y, qb[1], acc);
                acc = ggml_cuda_dp4a(k0v.z, qb[2], acc);
                acc = ggml_cuda_dp4a(k0v.w, qb[3], acc);
                acc = ggml_cuda_dp4a(k1v.x, qb[4], acc);
                acc = ggml_cuda_dp4a(k1v.y, qb[5], acc);
                acc = ggml_cuda_dp4a(k1v.z, qb[6], acc);
                acc = ggml_cuda_dp4a(k1v.w, qb[7], acc);
                const float qd = Qdw[b*8 + c];
                s[c] += (float) acc * (kd*qd);
            }
        }

        // ---- mask, online softmax (per column, wave-wide over the 64 rows)
        float mk[QPW];
#pragma unroll
        for (int jj = 0; jj < QPW; ++jj) {
            mk[jj] = __half2float(maskh[int64_t(qmask[jj])*stride_mask + k0 + lane]);
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            s[c] += mk[c / (NC2 < 8 ? NC2 : 8)];
            float mx = warp_reduce_max<64>(s[c] + FATTN_KQ_MAX_OFFSET);
            mx = fmaxf(mx, m[c]);
            const float sc = __expf(m[c] - mx);
            m[c] = mx;
            const float p = __expf(s[c] - mx);
            l[c] = l[c]*sc + p;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                O[c][t] *= sc;
            }
            Ps[w][lane/2][c][lane & 1] = __float2half(p);
        }
        __syncthreads();   // all waves done with Ks (K) and wrote their P strip

        // ---- PV in two halves of 32 KV positions (16 pair rows of 256 half2 = 16 KiB)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const char * Vt = Vg + int64_t(k0/2 + hf*16)*(2*nb21);
#pragma unroll
            for (int i0 = 0; i0 < 16*64; i0 += 256) {
                const int idx = i0 + tid;                      // int4 index in the 16 KiB half tile
                *(int4 *) (Ks + idx*4) = *(const int4 *) (Vt + int64_t(idx)*16);
            }
            __syncthreads();
            const half2 * Vs = (const half2 *) Ks;
#pragma unroll 4
            for (int jp = 0; jp < 16; ++jp) {
                const int4 vv = *(const int4 *) (Vs + jp*256 + lane*4);
                const half2 * v2 = (const half2 *) &vv;
                const int4 p0 = *(const int4 *) &Ps[w][hf*16 + jp][0][0];
                const int4 p1 = *(const int4 *) &Ps[w][hf*16 + jp][4][0];
                const half2 * pa = (const half2 *) &p0;
                const half2 * pb = (const half2 *) &p1;
#pragma unroll
                for (int t = 0; t < 4; ++t) {
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        O[c    ][t] = p906_fdot2(pa[c], v2[t], O[c    ][t]);
                        O[c + 4][t] = p906_fdot2(pb[c], v2[t], O[c + 4][t]);
                    }
                }
            }
            __syncthreads();
        }
    }

    // ---- finalize
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        l[c] = warp_reduce_sum<64>(l[c]);
    }
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        const int jc = 8*w + c;
        const int j  = jc / NC2;
        const int hh = jc % NC2;
        if (col_Q_0 + j >= ne01v) {
            continue;
        }
        const float inv = gridDim.y == 1 ? 1.0f/l[c] : 1.0f;
        const int64_t j_dst_unrolled = ((int64_t(sequence)*ne01v + col_Q_0 + j)*ne02 + head0 + hh)*gridDim.y + blockIdx.y;
        float4 o;
        o.x = O[c][0]*inv; o.y = O[c][1]*inv; o.z = O[c][2]*inv; o.w = O[c][3]*inv;
        *(float4 *) (dst + j_dst_unrolled*256 + lane*4) = o;
        if (gridDim.y != 1 && lane == 0) {
            dst_meta[j_dst_unrolled] = make_float2(m[c], l[c]);
        }
    }
    GGML_UNUSED_VARS(KV_rows, sinks_ptr, scale, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, nb01, nb02, nb03,
        ne10, ne13, ne31, ne32, nb32);
}

template <int NC1, int NC2, int QL>
static void p906k_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    cudaStream_t stream = ctx.stream();

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int64_t ne01p  = ((Q->ne[1] + NC1 - 1)/NC1)*NC1;
    const int64_t nqrows = ne01p*Q->ne[2]*Q->ne[3];        // multiple of 32 (NC1*NC2 | ne01p*ne02)
    const int64_t ne11 = K->ne[1], ne12 = K->ne[2], ne13 = K->ne[3];

    // K/V re-laid buffers live in the f16 scratch that the TILE kernel already reserves after dst
    // (ggml_cuda_flash_attn_ext_get_alloc_size): K int8 rows need 288 of the 512 B/row, V pairs exactly 512 B/row.
    // No extra VRAM versus the production path; only the (small) Q buffer comes from the pool.
    const ggml_cuda_flash_attn_ext_f16_extra_data ex = ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, true, true);
    GGML_ASSERT(ex.K != 0 && ex.V != 0 && ex.V != ex.K);
    int   * kbuf = (int   *) ex.K;
    half2 * vbuf = (half2 *) ex.V;
    ggml_cuda_pool_alloc<int> qbuf(ctx.pool(), size_t(nqrows)*P906K_QW);   // 64 int quants + 8 f32 scales per row

    p906k_q_quant<<<(unsigned) ((nqrows*8 + 255)/256), 256, 0, stream>>>(
        (const char *) Q->data, qbuf.ptr, (float *) (qbuf.ptr + nqrows*64), scale, NC2, QL, (int) Q->ne[1], (int) ne01p,
        (int) Q->ne[2], (int) Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3]);
    p906k_k_split<<<dim3((ne11 + 31)/32, ne12, ne13), 256, 0, stream>>>(
        (const char *) K->data, kbuf, (int) ne11, (int) ne12, K->nb[1], K->nb[2], K->nb[3]);
    if (V->type == GGML_TYPE_Q4_0) {
        p906k_v_pairs<GGML_TYPE_Q4_0><<<dim3((ne11/2 + 31)/32, ne12, ne13), 256, 0, stream>>>(
            (const char *) V->data, vbuf, (int) ne11, (int) ne12, V->nb[1], V->nb[2], V->nb[3]);
    } else {
        p906k_v_pairs<GGML_TYPE_Q8_0><<<dim3((ne11/2 + 31)/32, ne12, ne13), 256, 0, stream>>>(
            (const char *) V->data, vbuf, (int) ne11, (int) ne12, V->nb[1], V->nb[2], V->nb[3]);
    }
    CUDA_CHECK(cudaGetLastError());

    // launch_fattn reads data pointers and strides from the src tensors: hand it the re-laid buffers.
    ggml_tensor Qf = *Q;
    Qf.data = qbuf.ptr; Qf.view_src = nullptr; Qf.view_offs = 0;
    ggml_tensor Kf = *K;
    Kf.type = GGML_TYPE_I8; Kf.data = kbuf; Kf.view_src = nullptr; Kf.view_offs = 0;
    Kf.nb[0] = 1; Kf.nb[1] = P906K_QW*4; Kf.nb[2] = Kf.nb[1]*ne11; Kf.nb[3] = Kf.nb[2]*ne12;
    ggml_tensor Vf = *V;
    Vf.type = GGML_TYPE_F16; Vf.data = vbuf; Vf.view_src = nullptr; Vf.view_offs = 0;
    Vf.nb[0] = 2; Vf.nb[1] = 512; Vf.nb[2] = Vf.nb[1]*ne11; Vf.nb[3] = Vf.nb[2]*ne12;
    ggml_tensor d2 = *dst;
    d2.src[0] = &Qf; d2.src[1] = &Kf; d2.src[2] = &Vf;
    launch_fattn<256, NC1, NC2>(ctx, &d2, p906k_main<NC1, NC2, QL>, 4, 0, P906K_NB, false, false, false, false, 64);
}
#endif // GGML_USE_HIP

// GGML_HIP_FA_P906: unset/0 = production path; 100 = p906k (Q layout 0); 101 = p906k (Q layout 1, contiguous
// 8-column blocks for s_load_dwordx16).
static int p906k_mode() {
    static const int v = [] {
        const char * e = getenv("GGML_HIP_FA_P906");
        int r = e ? atoi(e) : 0;
        if (r != 100 && r != 101) {
            r = 0;
        }
        if (r) {
            fprintf(stderr, "[fa-p906k] enabled mode %d (gfx906 prompt attention, D=256, K q8_0, nq>=32)\n", r);
        }
        return r;
    }();
    return v;
}

bool ggml_cuda_flash_attn_ext_p906k_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    // Legacy contiguous-KV kernel: never consume upstream row-gather/sparse semantics.
    if (dst->src[5] || ggml_get_op_params_i32(dst, 4) != 0) return false;
    const int mode = p906k_mode();
    if (!mode) {
        return false;
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_GCN(cc) && !GGML_CUDA_CC_IS_CDNA(cc)) return false;   // wave64 only
    if (Q->ne[1] < 32) return false;                                          // prompt batches only: decode/MTP verify untouched
    if (K->ne[0] != 256 || V->ne[0] != 256 || !mask || dst->src[4]) return false;
    if (K->type != GGML_TYPE_Q8_0 || (V->type != GGML_TYPE_Q4_0 && V->type != GGML_TYPE_Q8_0)) return false;
    if (Q->type != GGML_TYPE_F32 || Q->nb[0] != sizeof(float) || (Q->nb[1] % 16) != 0 || (Q->nb[2] % 16) != 0 || (Q->nb[3] % 16) != 0) return false;
    if (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1) return false;
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) return false;
    if (K->ne[1] % FATTN_KQ_STRIDE != 0 || Q->ne[2] % K->ne[2] != 0) return false;
    if (K->ne[2] != V->ne[2] || K->ne[3] != V->ne[3] || K->ne[1] != V->ne[1]) return false;
    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
    if (V_is_K_view) return false;
    const int gqa = Q->ne[2] / K->ne[2];
    const int ncols2 = gqa % 8 == 0 ? 8 : (gqa % 2 == 0 && gqa % 4 != 0 ? 2 : 0);   // the two production layouts (GQA 8, GQA 6)
    if (ncols2 == 0) return false;
    static unsigned long long count = 0;
    const auto n = count++;
    const char * verbose = getenv("GGML_FA_B4_VERBOSE");
    if (verbose && atoi(verbose) && (n < 4 || n % 2000 == 0)) {
        fprintf(stderr, "[fa-b4] kernel=p906k n=%llu rows=0 mode=%d nq=%d nk=%d heads=%d/%d K=%s V=%s\n", n, mode, (int) Q->ne[1], (int) K->ne[1],
            (int) Q->ne[2], (int) K->ne[2], ggml_type_name(K->type), ggml_type_name(V->type));
    }
    const int ql = mode == 101;
    if (ncols2 == 2) { if (ql) p906k_launch<16, 2, 1>(ctx, dst); else p906k_launch<16, 2, 0>(ctx, dst); return true; }
    if (ncols2 == 8) { if (ql) p906k_launch< 4, 8, 1>(ctx, dst); else p906k_launch< 4, 8, 0>(ctx, dst); return true; }
#else
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
#endif // GGML_USE_HIP
    return false;
}
