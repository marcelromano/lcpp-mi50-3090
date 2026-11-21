//  (GGML_EXP_FA_LC): flash-attention decode kernel for long contexts, gfx906 (MI50).
//
// One block serves ALL the Q heads of a KV head (gqa NC = 4, 6 or 8), so K and V are read once per KV
// head instead of NC times (flash_attn_ext_vec launches one block per Q head and re-reads the cache NC
// times through L2). K/V are read with dwordx4 loads over the 68-byte (q8_0) / 36-byte (q4_0) pair of
// blocks, the 2-byte misalignment of the quants being absorbed by a shifted copy of Q (K) or by one
// v_alignbit per dword (V).
//
// Scope: D = 256, 1..8 tokens (decode and MTP verification; one block per token), 1 sequence, K and V in
// {q8_0, q4_0}, gqa 4/6/8, no sinks / alibi / softcap, n_kv % 256 == 0. Env: GGML_FA_LC=0 disables, GGML_FA_LC_MIN_KV
// (512), GGML_FA_LC_MAX_TOK (8), GGML_FA_LC_BPC (blocks per CU used to size the grid, 2).
// Anything else falls back to the stock path (the host function returns false).
#pragma once

#if defined(GGML_EXP_FA_LC) && defined(GGML_USE_HIP)

#include "common.cuh"
#include "fattn-common.cuh"

#define FA_LC_D  256   // head size
#define FA_LC_T  128   // KV positions per tile
#define FA_LC_NT 128   // threads per block (2 waves)
#define FA_LC_QW  24   // ints per (head, pair) row of the shifted Q table

typedef int fa_lc_v4i __attribute__((ext_vector_type(4), aligned(4)));

static __device__ __forceinline__ uint32_t fa_lc_alignbit(const uint32_t hi, const uint32_t lo, const uint32_t sh) {
#if defined(__gfx906__)
    return __builtin_amdgcn_alignbit(hi, lo, sh);
#else
    return (uint32_t) ((((uint64_t) hi << 32) | lo) >> (sh & 31));
#endif // defined(__gfx906__)
}

static __device__ __forceinline__ float fa_lc_h2f(const uint32_t bits16) {
    return __half2float(__ushort_as_half((unsigned short) (bits16 & 0xFFFFu)));
}

// Layout facts (row = 256 elements, KV heads are separate rows via nb12):
//   q8_0: 8 blocks x 34 B = 272 B = 4 pairs x 68 B;  pair = d0(2) qs0(32) d1(2) qs1(32) = 17 dwords.
//   q4_0: 8 blocks x 18 B = 144 B = 4 pairs x 36 B;  pair = d0(2) qs0(16) d1(2) qs1(16) =  9 dwords.
// K phase: the 4 lanes of a quad take one K row each, lane p owns pair p (all loads 4-byte aligned).
//   The dwords that straddle a header (d0 | qs0[0..1], qs0[30..31] | d1, ...) are multiplied with a Q
//   dword that is zero on the header bytes, so no realignment of K is needed.
// V phase: lane l of 16 owns elements 16*l .. 16*l+15 of the row; 5 dwords are loaded from the
//   4-aligned address below the quants and realigned with v_alignbit.
template <int NC, ggml_type type_K, ggml_type type_V>
__launch_bounds__(FA_LC_NT, 1)
static __global__ void flash_attn_ext_vec_lc(
        const char  * __restrict__ Q_ptr,
        const char  * __restrict__ K_ptr,
        const char  * __restrict__ V_ptr,
        const half  * __restrict__ mask_ptr,
        float       * __restrict__ dst_ptr,
        float2      * __restrict__ meta_ptr,
        const float scale,
        const int   n_kv,
        const int   nb02,
        const int64_t nb11, const int64_t nb12,
        const int64_t nb21, const int64_t nb22,
        const int   n_head, const int n_tok, const int P,
        const int64_t nb01, const int64_t nb31h) {
    ggml_cuda_pdl_lc();
#ifdef FLASH_ATTN_AVAILABLE
    constexpr int D  = FA_LC_D;
    constexpr int T  = FA_LC_T;
    constexpr int NT = FA_LC_NT;
    constexpr int QW = FA_LC_QW;
    constexpr bool K_q8 = type_K == GGML_TYPE_Q8_0;
    constexpr bool V_q8 = type_V == GGML_TYPE_Q8_0;
    constexpr int  NKD  = K_q8 ? 17 : 9;   // dwords of a K pair
    constexpr int  KPB  = K_q8 ? 68 : 36;  // bytes of a K pair
    constexpr int  NJJ  = (NC + 3) / 4;

    static_assert(NC >= 1 && NC <= 8, "bad NC");
    static_assert(type_K == GGML_TYPE_Q8_0 || type_K == GGML_TYPE_Q4_0, "bad type_K");
    static_assert(type_V == GGML_TYPE_Q8_0 || type_V == GGML_TYPE_Q4_0, "bad type_V");

    const int tid  = threadIdx.x;
    const int kvh  = blockIdx.y;
    const int tq    = blockIdx.x % n_tok;    // token of the batch (verification of MTP drafts: up to 8)
    const int chunk = blockIdx.x / n_tok;    // chunk of the KV; consecutive blocks = same chunk, other token (shares L2)
    const half * mask_t = mask_ptr ? mask_ptr + (size_t) tq*nb31h : (const half *) nullptr;

    __shared__ int8_t sQb[NC][D];                       // Q as int8 (scaled), one row per head
    __shared__ __align__(16) float sQd[NC][16];         // per 32-block: (d, sum of the scaled floats)
    __shared__ __align__(16) int   sQT[NC][4][QW];      // Q aligned to the dword grid of each K pair
    __shared__ float sS[NC][T];                         // scores of the tile
    __shared__ __align__(16) half sP[T][8];             // probabilities of the tile, [pos][head]
    __shared__ __align__(16) float sRed[8][D];          // cross-slot reduction of VKQ
    __shared__ float sScale[8];
    __shared__ float sM[8];
    __shared__ float sL[8];

    const char * Kh = K_ptr + nb12*kvh;
    const char * Vh = V_ptr + nb22*kvh;

    // ---- Q: scale, quantize to q8_1-like blocks of 32 ----
    ggml_cuda_pdl_sync();
    if (tid < NC*8) {
        const int j = tid >> 3;
        const int b = tid & 7;
        const float4 * src = (const float4 *) (Q_ptr + (size_t) nb01*tq + (size_t) nb02*(kvh*NC + j) + 128*b);
        float v[32];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 t = src[i];
            v[4*i + 0] = t.x*scale;
            v[4*i + 1] = t.y*scale;
            v[4*i + 2] = t.z*scale;
            v[4*i + 3] = t.w*scale;
        }
        float amax = 0.0f;
        float sum  = 0.0f;
#pragma unroll
        for (int e = 0; e < 32; ++e) {
            amax = fmaxf(amax, fabsf(v[e]));
            sum += v[e];
        }
        const float d = amax / 127;
#pragma unroll
        for (int e = 0; e < 32; ++e) {
            sQb[j][32*b + e] = d != 0.0f ? (int8_t) roundf(v[e] / d) : (int8_t) 0;
        }
        sQd[j][2*b + 0] = d;
        sQd[j][2*b + 1] = sum;
    }
    __syncthreads();

    // ---- Q table: for each (head, pair p, dword w of the K pair) the 4 Q bytes that meet those 4 K bytes ----
    for (int idx = tid; idx < NC*4*QW; idx += NT) {
        const int j   = idx / (4*QW);
        const int rem = idx % (4*QW);
        const int p   = rem / QW;
        const int w   = rem % QW;
        uint32_t val = 0;
        if constexpr (K_q8) {
            if (w < 17) {
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    int g = -1;
                    if (w <= 8) {
                        const int e = 4*w - 2 + m;               // element of block 2p (header bytes -> none)
                        if (e >= 0 && e < 32) {
                            g = 64*p + e;
                        }
                    } else {
                        g = 64*p + 32 + 4*(w - 9) + m;           // block 2p+1
                    }
                    const uint32_t byte = g >= 0 ? (uint32_t) (uint8_t) sQb[j][g] : 0u;
                    val |= byte << (8*m);
                }
            }
        } else {
            const bool hi = w >= 12;
            const int  wl = hi ? w - 12 : w;
            if (wl <= 8 && (w <= 8 || (w >= 12 && w <= 20))) {
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    int g = -1;
                    if (wl <= 4) {
                        const int r = 4*wl - 2 + m;              // qs byte of block 2p (header bytes -> none)
                        if (r >= 0 && r < 16) {
                            g = 64*p + r + (hi ? 16 : 0);
                        }
                    } else {
                        g = 64*p + 32 + 4*(wl - 5) + m + (hi ? 16 : 0);   // block 2p+1
                    }
                    const uint32_t byte = g >= 0 ? (uint32_t) (uint8_t) sQb[j][g] : 0u;
                    val |= byte << (8*m);
                }
            }
        }
        sQT[j][p][w] = (int) val;
    }
    __syncthreads();

    // ---- per-thread roles ----
    const int p4   = tid & 3;           // K phase: pair owned inside the quad
    const int quad = tid >> 2;          // K phase: 32 quads, 4 rows each per tile
    const int warp = tid >> 5;          // softmax: logical warp of 32
    const int lane = tid & 31;
    const int vs   = tid >> 4;          // V phase: slot 0..7 (positions vs, vs+8, ...)
    const int vl   = tid & 15;          // V phase: elements 16*vl .. 16*vl+15
    const int vb   = vl >> 1;
    const int vh   = vl & 1;

    int v_off;
    int v_doff;
    if constexpr (V_q8) {
        v_off  = 34*vb + 2 + 16*vh;
        v_doff = 34*vb;
    } else {
        v_off  = 18*vb + 2;
        v_doff = 18*vb;
    }
    const int      v_a  = v_off & ~3;
    const uint32_t v_sh = (uint32_t) (v_off & 3) * 8;

    half2 VKQ[NC][8];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            VKQ[j][e] = make_half2(0.0f, 0.0f);
        }
    }

    float m_run[NJJ];
    float l_run[NJJ];
#pragma unroll
    for (int jj = 0; jj < NJJ; ++jj) {
        m_run[jj] = -FLT_MAX/2.0f;
        l_run[jj] = 0.0f;
    }

    for (int k0 = chunk*T; k0 < n_kv; k0 += P*T) {
        // ---- K phase: scores of the tile for the NC heads ----
#pragma unroll
        for (int ii = 0; ii < 2; ++ii) {
            int a[2][NKD];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const char * kr = Kh + (size_t) (k0 + quad + 32*(2*ii + r))*nb11 + KPB*p4;
                const fa_lc_v4i v0 = *(const fa_lc_v4i *) (kr +  0);
                const fa_lc_v4i v1 = *(const fa_lc_v4i *) (kr + 16);
                a[r][0] = v0.x; a[r][1] = v0.y; a[r][2] = v0.z; a[r][3] = v0.w;
                a[r][4] = v1.x; a[r][5] = v1.y; a[r][6] = v1.z; a[r][7] = v1.w;
                if constexpr (K_q8) {
                    const fa_lc_v4i v2 = *(const fa_lc_v4i *) (kr + 32);
                    const fa_lc_v4i v3 = *(const fa_lc_v4i *) (kr + 48);
                    a[r][ 8] = v2.x; a[r][ 9] = v2.y; a[r][10] = v2.z; a[r][11] = v2.w;
                    a[r][12] = v3.x; a[r][13] = v3.y; a[r][14] = v3.z; a[r][15] = v3.w;
                    a[r][16] = *(const int *) (kr + 64);
                } else {
                    a[r][8] = *(const int *) (kr + 32);
                }
            }

#pragma unroll
            for (int j = 0; j < NC; ++j) {
                const float4 qd = *(const float4 *) &sQd[j][4*p4];
                float part[2];
                if constexpr (K_q8) {
                    const int4 t0 = *(const int4 *) &sQT[j][p4][ 0];
                    const int4 t1 = *(const int4 *) &sQT[j][p4][ 4];
                    const int4 t2 = *(const int4 *) &sQT[j][p4][ 8];
                    const int4 t3 = *(const int4 *) &sQT[j][p4][12];
                    const int  t4 = sQT[j][p4][16];
                    const int tq[17] = {t0.x, t0.y, t0.z, t0.w, t1.x, t1.y, t1.z, t1.w,
                                        t2.x, t2.y, t2.z, t2.w, t3.x, t3.y, t3.z, t3.w, t4};
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        int s0 = 0;
                        int s1 = 0;
#pragma unroll
                        for (int w = 0; w < 17; ++w) {
                            if (w <= 8) {
                                s0 = ggml_cuda_dp4a(a[r][w], tq[w], s0);
                            } else {
                                s1 = ggml_cuda_dp4a(a[r][w], tq[w], s1);
                            }
                        }
                        const float d0 = fa_lc_h2f((uint32_t) a[r][0]);
                        const float d1 = fa_lc_h2f((uint32_t) a[r][8] >> 16);
                        part[r] = d0*(qd.x*(float) s0) + d1*(qd.z*(float) s1);
                    }
                } else {
                    const int4 l0 = *(const int4 *) &sQT[j][p4][ 0];
                    const int4 l1 = *(const int4 *) &sQT[j][p4][ 4];
                    const int  l2 = sQT[j][p4][8];
                    const int4 h0 = *(const int4 *) &sQT[j][p4][12];
                    const int4 h1 = *(const int4 *) &sQT[j][p4][16];
                    const int  h2 = sQT[j][p4][20];
                    const int tl[9] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w, l2};
                    const int th[9] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w, h2};
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        int s0 = 0;
                        int s1 = 0;
#pragma unroll
                        for (int w = 0; w < 9; ++w) {
                            const int lo = a[r][w] & 0x0F0F0F0F;
                            const int hi = (a[r][w] >> 4) & 0x0F0F0F0F;
                            if (w <= 4) {
                                s0 = ggml_cuda_dp4a(lo, tl[w], s0);
                                s0 = ggml_cuda_dp4a(hi, th[w], s0);
                            } else {
                                s1 = ggml_cuda_dp4a(lo, tl[w], s1);
                                s1 = ggml_cuda_dp4a(hi, th[w], s1);
                            }
                        }
                        const float d0 = fa_lc_h2f((uint32_t) a[r][0]);
                        const float d1 = fa_lc_h2f((uint32_t) a[r][4] >> 16);
                        part[r] = d0*(qd.x*(float) s0 - 8.0f*qd.y) + d1*(qd.z*(float) s1 - 8.0f*qd.w);
                    }
                }

#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    part[r] += __shfl_xor_sync(0xFFFFFFFF, part[r], 1, 32);
                    part[r] += __shfl_xor_sync(0xFFFFFFFF, part[r], 2, 32);
                }
                if (p4 == (j & 3)) {
                    sS[j][quad + 32*(2*ii + 0)] = part[0];
                    sS[j][quad + 32*(2*ii + 1)] = part[1];
                }
            }
        }
        __syncthreads();

        // ---- softmax: logical warp w owns the heads w, w+4 ----
        {
            float mk[T/32];
#pragma unroll
            for (int i = 0; i < T/32; ++i) {
                mk[i] = mask_t ? __half2float(mask_t[k0 + lane + 32*i]) : 0.0f;
            }
#pragma unroll
            for (int jj = 0; jj < NJJ; ++jj) {
                const int j = warp + 4*jj;
                if (j < NC) {
                    float s[T/32];
                    float tmax = -INFINITY;
#pragma unroll
                    for (int i = 0; i < T/32; ++i) {
                        s[i]  = sS[j][lane + 32*i] + mk[i];
                        tmax  = fmaxf(tmax, s[i]);
                    }
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        tmax = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFF, tmax, off, 32));
                    }
                    const float m_new = fmaxf(m_run[jj], tmax + FATTN_KQ_MAX_OFFSET);
                    const float sc    = expf(m_run[jj] - m_new);
                    m_run[jj] = m_new;
                    float psum = 0.0f;
#pragma unroll
                    for (int i = 0; i < T/32; ++i) {
                        const float pv = expf(s[i] - m_new);
                        psum += pv;
                        sP[lane + 32*i][j] = __float2half(pv);
                    }
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        psum += __shfl_xor_sync(0xFFFFFFFF, psum, off, 32);
                    }
                    l_run[jj] = l_run[jj]*sc + psum;
                    if (lane == 0) {
                        sScale[j] = sc;
                    }
                }
            }
        }
        __syncthreads();

        // ---- V phase ----
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const half2 sc2 = __half2half2(__float2half(sScale[j]));
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                VKQ[j][e] *= sc2;
            }
        }

#pragma unroll 4
        for (int i = 0; i < T/8; ++i) {
            const int pos = vs + 8*i;
            const char * vr = Vh + (size_t) (k0 + pos)*nb21;

            const fa_lc_v4i w4 = *(const fa_lc_v4i *) (vr + v_a);
            uint32_t w5 = 0;
            if (v_sh != 0) {
                w5 = *(const uint32_t *) (vr + v_a + 16);
            }
            const uint32_t db = *(const unsigned short *) (vr + v_doff);

            uint32_t x[4];
            x[0] = fa_lc_alignbit((uint32_t) w4.y, (uint32_t) w4.x, v_sh);
            x[1] = fa_lc_alignbit((uint32_t) w4.z, (uint32_t) w4.y, v_sh);
            x[2] = fa_lc_alignbit((uint32_t) w4.w, (uint32_t) w4.z, v_sh);
            x[3] = fa_lc_alignbit(w5,              (uint32_t) w4.w, v_sh);

            const half2 d2 = __half2half2(__ushort_as_half((unsigned short) db));
            half2 vv[8];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                float f0;
                float f1;
                float f2;
                float f3;
                if constexpr (V_q8) {
                    f0 = (float) (int8_t) (x[q]      );
                    f1 = (float) (int8_t) (x[q] >>  8);
                    f2 = (float) (int8_t) (x[q] >> 16);
                    f3 = (float) (int8_t) (x[q] >> 24);
                } else {
                    const uint32_t n = (x[q] >> (4*vh)) & 0x0F0F0F0Fu;
                    f0 = (float) ((int) ( n        & 0xF) - 8);
                    f1 = (float) ((int) ((n >>  8) & 0xF) - 8);
                    f2 = (float) ((int) ((n >> 16) & 0xF) - 8);
                    f3 = (float) ((int) ((n >> 24) & 0xF) - 8);
                }
                vv[2*q + 0] = d2*__floats2half2_rn(f0, f1);
                vv[2*q + 1] = d2*__floats2half2_rn(f2, f3);
            }

            const half2 * pp = (const half2 *) &sP[pos][0];
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                const half2 pj2 = (j & 1) ? __high2half2(pp[j >> 1]) : __low2half2(pp[j >> 1]);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    VKQ[j][e] += vv[e]*pj2;
                }
            }
        }
    }

    // ---- epilogue: merge the 8 position slots, write partial results + (max, sum) ----
    if (lane == 0) {
#pragma unroll
        for (int jj = 0; jj < NJJ; ++jj) {
            const int j = warp + 4*jj;
            if (j < NC) {
                sM[j] = m_run[jj];
                sL[j] = l_run[jj];
            }
        }
    }

#pragma unroll
    for (int j = 0; j < NC; ++j) {
        __syncthreads();
        float * rp = &sRed[vs][16*vl];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float2 f = __half22float2(VKQ[j][e]);
            rp[2*e + 0] = f.x;
            rp[2*e + 1] = f.y;
        }
        __syncthreads();
        for (int d = tid; d < D; d += NT) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                acc += sRed[s][d];
            }
            if (P == 1) {
                acc /= sL[j];
            }
            dst_ptr[((size_t) (tq*n_head + kvh*NC + j)*P + chunk)*D + d] = acc;
        }
    }
    if (P > 1 && tid < NC) {
        meta_ptr[(size_t) (tq*n_head + kvh*NC + tid)*P + chunk] = make_float2(sM[tid], sL[tid]);
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, dst_ptr, meta_ptr, scale, n_kv, nb02, nb11, nb12, nb21, nb22, n_head, n_tok, P, nb01, nb31h);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

#ifndef FA_LC_KERNEL_ONLY

template <int NC, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_vec_lc_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int P) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();

    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);
    if (P > 1) {
        dst_tmp.alloc(P*ggml_nelements(dst));
        dst_tmp_meta.alloc(P*ggml_nrows(dst));
    }

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_tok = Q->ne[1];
    const dim3 blocks_num(P*n_tok, K->ne[2], 1);
    const dim3 block_dim(FA_LC_NT, 1, 1);

    ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num, block_dim, 0, main_stream);
    ggml_cuda_kernel_launch(flash_attn_ext_vec_lc<NC, type_K, type_V>, launch_params,
        (const char *) Q->data,
        (const char *) K->data,
        (const char *) V->data,
        mask ? (const half *) mask->data : (const half *) nullptr,
        P > 1 ? dst_tmp.ptr : (float *) dst->data,
        dst_tmp_meta.ptr,
        scale, (int) K->ne[1], (int) Q->nb[2],
        (int64_t) K->nb[1], (int64_t) K->nb[2], (int64_t) V->nb[1], (int64_t) V->nb[2],
        (int) Q->ne[2], n_tok, P, (int64_t) Q->nb[1], mask ? (int64_t) (mask->nb[1]/sizeof(half)) : (int64_t) 0);
    CUDA_CHECK(cudaGetLastError());

    if (P > 1) {
        const dim3 block_dim_combine(FA_LC_D, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = P*sizeof(float2);

        ggml_cuda_kernel_launch_params launch_params_combine = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<FA_LC_D>, launch_params_combine,
            dst_tmp.ptr, dst_tmp_meta.ptr, (float *) dst->data, P);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Returns true if the op was handled; false -> the caller runs the stock vec/tile path.
static bool ggml_cuda_flash_attn_ext_vec_lc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool env_on = []() {
        const char * e = getenv("GGML_FA_LC");
        return e == nullptr || atoi(e) != 0;
    }();
    static const int env_min_kv = []() {
        const char * e = getenv("GGML_FA_LC_MIN_KV");
        return e ? atoi(e) : 512;
    }();
    static const int env_max_tok = []() {
        const char * e = getenv("GGML_FA_LC_MAX_TOK");
        return e ? atoi(e) : 8;
    }();
    static const int env_bpc = []() {
        const char * e = getenv("GGML_FA_LC_BPC");
        return e ? atoi(e) : 2;
    }();
    if (!env_on) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    if (!GGML_CUDA_CC_IS_GCN(cc)) {
        return false;
    }

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    if (sinks || max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || Q->ne[0] != FA_LC_D || K->ne[0] != FA_LC_D || V->ne[0] != FA_LC_D) {
        return false;
    }
    if (Q->ne[1] < 1 || Q->ne[1] > env_max_tok || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 || V->ne[1] != K->ne[1]) {
        return false;
    }
    if (K->ne[1] % 256 != 0 || K->ne[1] < env_min_kv) {
        return false;
    }
    if ((K->type != GGML_TYPE_Q8_0 && K->type != GGML_TYPE_Q4_0) || (V->type != GGML_TYPE_Q8_0 && V->type != GGML_TYPE_Q4_0)) {
        return false;
    }
    if (K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type)) {
        return false;
    }
    if (K->nb[1] % 4 != 0 || K->nb[2] % 4 != 0 || V->nb[1] % 4 != 0 || V->nb[2] % 4 != 0 ||
        ((uintptr_t) K->data) % 4 != 0 || ((uintptr_t) V->data) % 4 != 0) {
        return false;
    }
    if (Q->nb[0] != sizeof(float) || Q->nb[1] % 16 != 0 || Q->nb[2] % 16 != 0 || ((uintptr_t) Q->data) % 16 != 0) {
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[1] < Q->ne[1] || mask->nb[1] % sizeof(half) != 0 || mask->nb[0] != sizeof(half) || ((uintptr_t) mask->data) % 2 != 0)) {
        return false;
    }

    const int gqa = Q->ne[2] / K->ne[2];
    if (gqa != 4 && gqa != 6 && gqa != 8) {
        return false;
    }

    // blocks in flight: env_bpc blocks per CU; each block walks tiles of 128 positions strided by P
    const int nsm    = ggml_cuda_info().devices[id].nsm;
    const int ntiles = K->ne[1] / FA_LC_T;
    const int pmax   = std::max(1, (nsm*std::max(1, env_bpc)) / ((int) K->ne[2]*(int) Q->ne[1]));
    const int tpb    = (ntiles + pmax - 1) / pmax;
    const int P      = (ntiles + tpb - 1) / tpb;

#define FA_LC_CASE(NCV, TK, TV)                                                                    \
    if (gqa == (NCV) && K->type == (TK) && V->type == (TV)) {                                      \
        ggml_cuda_flash_attn_ext_vec_lc_launch<NCV, TK, TV>(ctx, dst, P);                          \
        return true;                                                                               \
    }
#define FA_LC_CASES(NCV)                                      \
    FA_LC_CASE(NCV, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)           \
    FA_LC_CASE(NCV, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)           \
    FA_LC_CASE(NCV, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)           \
    FA_LC_CASE(NCV, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)

    FA_LC_CASES(4)
    FA_LC_CASES(6)
    FA_LC_CASES(8)

#undef FA_LC_CASES
#undef FA_LC_CASE
    return false;
}

#endif // FA_LC_KERNEL_ONLY

#endif // defined(GGML_EXP_FA_LC) && defined(GGML_USE_HIP)
