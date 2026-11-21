#include "gated_delta_net.cuh"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"

#include <cstdint>

constexpr int gdn_cols_per_warp = 4;

template <int S_v, bool KDA, bool keep_rs_t, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// Prefill variant (non-KDA): each state column is owned by LPC lanes that hold S_v/LPC state values
// each in registers. The per-token k.S and S.q dot products are in-lane FMA chains plus a log2(LPC)-step
// shuffle inside the lane group, instead of the full-warp shuffle reductions of the kernel above (which
// dominate its time). k, q, g and beta are shared by all columns of a head, so the block stages them in
// shared memory GDN_CT_TT tokens at a time. A block covers GDN_CT_COLS columns (GDN_CT_COLS*LPC threads).
// Semantics match gated_delta_net_cuda (head mapping, strides, snapshots, final state).
#define GDN_CT_TT   16
#define GDN_CT_COLS 32

// Sum over aligned groups of W <= 16 lanes with DPP moves instead of LDS-crossbar shuffles (the two per-token
// reductions sit on the recurrence's critical path). Same pairing and order as warp_reduce_sum (xor W/2 first, then
// down to xor 1), so bit-identical: xor 8 / xor 4 = row_shl + row_shr by 8 / 4 on alternating 4-lane banks, xor 2 /
// xor 1 = quad_perm [2,3,0,1] / [1,0,3,2]. gfx906, 48 heads x 320 tokens: 546 -> 422 us.
#if defined(GGML_USE_HIP) && !defined(GGML_GDN_NO_DPP)
template <int CTRL, int BANKS>
static __device__ __forceinline__ int gdn_dpp(const int old, const int v) {
    return __builtin_amdgcn_update_dpp(old, v, CTRL, 0xF, BANKS, false);
}
template <int OFF>
static __device__ __forceinline__ float gdn_xor(const float x) {
    const int v = __float_as_int(x);
    if constexpr (OFF == 8) {
        return __int_as_float(gdn_dpp<0x118, 0xC>(gdn_dpp<0x108, 0x3>(v, v), v)); // row_shl:8 banks 0-1, row_shr:8 banks 2-3
    } else if constexpr (OFF == 4) {
        return __int_as_float(gdn_dpp<0x114, 0xA>(gdn_dpp<0x104, 0x5>(v, v), v)); // row_shl:4 banks 0,2, row_shr:4 banks 1,3
    } else if constexpr (OFF == 2) {
        return __int_as_float(gdn_dpp<0x4E, 0xF>(v, v));
    } else {
        return __int_as_float(gdn_dpp<0xB1, 0xF>(v, v));
    }
}
#endif

template <int W>
static __device__ __forceinline__ float gdn_group_sum(float x) {
#if defined(GGML_USE_HIP) && !defined(GGML_GDN_NO_DPP)
    static_assert(W == 1 || W == 2 || W == 4 || W == 8 || W == 16, "group width");
    if constexpr (W >= 16) { x += gdn_xor<8>(x); }
    if constexpr (W >= 8)  { x += gdn_xor<4>(x); }
    if constexpr (W >= 4)  { x += gdn_xor<2>(x); }
    if constexpr (W >= 2)  { x += gdn_xor<1>(x); }
    return x;
#else
    return warp_reduce_sum<W>(x);
#endif
}

// KDA (per-row gate, g is [S_v, H, T, n_seqs]): exp(g) is staged per row next to k/q, the state is decayed first and
// the decayed state feeds both dot products, i.e. kv = sum_i (exp(g[i]) S[i][col]) k[i] and S = exp(g) S + k delta, as
// in gated_delta_net_cuda<.., KDA = true> (GLM-5-Next: 64 heads x 1024 tokens, gfx906: 11.6 ms there)
template <int S_v, bool keep_rs_t, int LPC, bool QKNORM = false, int COLS = GDN_CT_COLS, bool KDA = false>
__global__ void __launch_bounds__(COLS * LPC, 1)
gated_delta_net_colthread(const float * q, const float * k, const float * v, const float * g, const float * beta,
                          const float * curr_state, float * dst, float * state,
                          int64_t H, int64_t n_tokens, int64_t n_seqs,
                          int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
                          int64_t sb1, int64_t sb2, int64_t sb3,
                          const uint3 neqk1_magic, const uint3 rq3_magic,
                          float scale, int64_t state_slot_stride, int K,
                          float eps_q = 0.0f, float eps_k = 0.0f, float sc_q = 1.0f, float sc_k = 1.0f,
                          const int32_t * __restrict__ s_ids = nullptr) {
    constexpr int R       = S_v / LPC;           // state rows per lane
    constexpr int NT      = COLS * LPC;   // threads per block
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      sub      = threadIdx.x % LPC; // lane within the column group
    const int      col      = blockIdx.z * COLS + threadIdx.x / LPC;
    // lane `sub` owns the float4 row chunks sub, sub+LPC, sub+2*LPC, ... (interleaved, so the lanes of a
    // group read adjacent 16-byte pieces of the shared k/q rows instead of hitting the same bank)

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const float * q_base = q + iq3 * sq3 + iq1 * sq1;
    const float * k_base = k + iq3 * sq3 + iq1 * sq1;
    const float * v_base = v + sequence * sv3 + h_idx * sv1;
    const float * b_base = beta + sequence * sb3 + h_idx * sb1;
    const float * g_base = g    + (sequence * sb3 + h_idx * sb1) * (KDA ? S_v : 1);

    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;
    state += (sequence * H + h_idx) * S_v * S_v;
    // s_ids: the state is read straight from its recurrent-cache row (the graph's GET_ROWS gather was skipped)
    const int64_t s_row = s_ids ? (int64_t) s_ids[sequence] : (int64_t) sequence;
    const float * s_in = curr_state + s_row * H * S_v * S_v + h_idx * S_v * S_v + (int64_t) col * S_v;

    __shared__ float4 sk [GDN_CT_TT][S_v / 4];
    __shared__ float4 sqs[GDN_CT_TT][S_v / 4];
    __shared__ float4 sge[KDA ? GDN_CT_TT : 1][KDA ? S_v / 4 : 1]; // KDA: exp(g) per row
    __shared__ float  sg[GDN_CT_TT];
    __shared__ float  sb[GDN_CT_TT];
    __shared__ float  sv[GDN_CT_TT][COLS]; // this block's value columns, staged with k/q (off the per-token path)

    float s[R];
    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < R; i += 4) {
        const float4 t = *(const float4 *) (s_in + 4*(sub + LPC*(i/4)));
        s[i + 0] = t.x; s[i + 1] = t.y; s[i + 2] = t.z; s[i + 3] = t.w;
    }

    for (int64_t t0 = 0; t0 < n_tokens; t0 += GDN_CT_TT) {
        const int nt = (int) min((int64_t) GDN_CT_TT, n_tokens - t0);

        __syncthreads();
        for (int idx = threadIdx.x; idx < nt * (S_v / 4); idx += NT) {
            const int tt = idx / (S_v / 4);
            const int i4 = idx % (S_v / 4);
            sk [tt][i4] = *(const float4 *) (k_base + (t0 + tt) * sq2 + 4 * i4);
            sqs[tt][i4] = *(const float4 *) (q_base + (t0 + tt) * sq2 + 4 * i4);
            if constexpr (KDA) {
                const float4 gg = *(const float4 *) (g_base + (t0 + tt) * sb2 * S_v + 4 * i4);
                sge[tt][i4] = make_float4(expf(gg.x), expf(gg.y), expf(gg.z), expf(gg.w));
            }
        }
        for (int tt = threadIdx.x; tt < nt; tt += NT) {
            if constexpr (!KDA) {
                sg[tt] = expf(g_base[(t0 + tt) * sb2]);
            }
            sb[tt] = b_base[(t0 + tt) * sb2];
        }
        for (int idx = threadIdx.x; idx < nt * COLS; idx += NT) {
            const int tt = idx / COLS;
            const int c  = idx % COLS;
            sv[tt][c] = v_base[(t0 + tt) * sv2 + blockIdx.z * COLS + c];
        }
        __syncthreads();
        if constexpr (QKNORM) {
            // the skipped RMS_NORM -> SCALE of q and k: 32-lane groups normalize one staged 128-value row each
            static_assert(S_v == 128 && NT % 32 == 0, "q/k norm assumes 128-value rows and 32-lane groups");
            const int l32 = threadIdx.x % 32;
            for (int vec = threadIdx.x / 32; vec < 2*nt; vec += NT / 32) {
                const bool is_k = vec < nt;
                float4 & x = is_k ? sk[vec][l32] : sqs[vec - nt][l32];
                float ss = x.x*x.x + x.y*x.y + x.z*x.z + x.w*x.w;
                ss = warp_reduce_sum<32>(ss);
                const float r = rsqrtf(ss/S_v + (is_k ? eps_k : eps_q)) * (is_k ? sc_k : sc_q);
                x.x *= r; x.y *= r; x.z *= r; x.w *= r;
            }
            __syncthreads();
        }

        for (int tt = 0; tt < nt; ++tt) {
            const int64_t t    = t0 + tt;
            const float   vcol = sv[tt][threadIdx.x / LPC];
            float attn;

            if constexpr (KDA) {
                // S[i][col] *= exp(g[i]) ; kv = sum_i S[i][col] * k[i]
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
                for (int i = 0; i < R; i += 4) {
                    const float4 ee = sge[tt][sub + LPC*(i/4)];
                    const float4 kk = sk [tt][sub + LPC*(i/4)];
                    s[i + 0] *= ee.x;
                    s[i + 1] *= ee.y;
                    s[i + 2] *= ee.z;
                    s[i + 3] *= ee.w;
                    a0 += s[i + 0] * kk.x;
                    a1 += s[i + 1] * kk.y;
                    a2 += s[i + 2] * kk.z;
                    a3 += s[i + 3] * kk.w;
                }
                const float kv    = gdn_group_sum<LPC>((a0 + a1) + (a2 + a3));
                const float delta = (vcol - kv) * sb[tt];

                // S[i][col] += k[i] * delta ; attn = sum_i S[i][col] * q[i]
                float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
#pragma unroll
                for (int i = 0; i < R; i += 4) {
                    const float4 kk = sk [tt][sub + LPC*(i/4)];
                    const float4 qq = sqs[tt][sub + LPC*(i/4)];
                    s[i + 0] += kk.x * delta;
                    s[i + 1] += kk.y * delta;
                    s[i + 2] += kk.z * delta;
                    s[i + 3] += kk.w * delta;
                    b0 += s[i + 0] * qq.x;
                    b1 += s[i + 1] * qq.y;
                    b2 += s[i + 2] * qq.z;
                    b3 += s[i + 3] * qq.w;
                }
                attn = gdn_group_sum<LPC>((b0 + b1) + (b2 + b3));
            } else {
                const float   gv   = sg[tt];

                // kv = sum_i S[i][col] * k[i]  (in-lane partial over this lane's rows, then the lane group)
                // 4 independent chains (the dot products are on the per-token critical path)
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
                for (int i = 0; i < R; i += 4) {
                    const float4 kk = sk[tt][sub + LPC*(i/4)];
                    a0 += s[i + 0] * kk.x;
                    a1 += s[i + 1] * kk.y;
                    a2 += s[i + 2] * kk.z;
                    a3 += s[i + 3] * kk.w;
                }
                const float kv    = gdn_group_sum<LPC>((a0 + a1) + (a2 + a3));
                const float delta = (vcol - gv * kv) * sb[tt];

                // S[i][col] = g * S[i][col] + k[i] * delta ; attn = sum_i S[i][col] * q[i]
                float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
#pragma unroll
                for (int i = 0; i < R; i += 4) {
                    const float4 kk = sk [tt][sub + LPC*(i/4)];
                    const float4 qq = sqs[tt][sub + LPC*(i/4)];
                    s[i + 0] = gv * s[i + 0] + kk.x * delta;
                    s[i + 1] = gv * s[i + 1] + kk.y * delta;
                    s[i + 2] = gv * s[i + 2] + kk.z * delta;
                    s[i + 3] = gv * s[i + 3] + kk.w * delta;
                    b0 += s[i + 0] * qq.x;
                    b1 += s[i + 1] * qq.y;
                    b2 += s[i + 2] * qq.z;
                    b3 += s[i + 3] * qq.w;
                }
                attn = gdn_group_sum<LPC>((b0 + b1) + (b2 + b3));
            }
            if (sub == 0) {
                attn_data[t * S_v * H + col] = attn * scale;
            }

            if constexpr (keep_rs_t) {
                const int target_slot = (int) (n_tokens - 1 - t);
                if (target_slot >= 0 && target_slot < K) {
                    float * cs = state + target_slot * state_slot_stride + (int64_t) col * S_v;
#pragma unroll
                    for (int i = 0; i < R; i += 4) {
                        *(float4 *) (cs + 4*(sub + LPC*(i/4))) = make_float4(s[i + 0], s[i + 1], s[i + 2], s[i + 3]);
                    }
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
        float * cs = state + (int64_t) col * S_v;
#pragma unroll
        for (int i = 0; i < R; i += 4) {
            *(float4 *) (cs + 4*(sub + LPC*(i/4))) = make_float4(s[i + 0], s[i + 1], s[i + 2], s[i + 3]);
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // four columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
    // SMs without a CTA, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    // prefill: column-per-thread kernel (see gated_delta_net_colthread). The previous revision (K15): OPT-IN by
    // GGML_GDN_COLTHREAD=1 — default OFF keeps the production (per-token) kernel, so this env key is the
    // only difference between the `ctl` and the `A` arm.
    static const bool colthread = [] { const char * e = getenv("GGML_GDN_COLTHREAD"); return e && atoi(e) != 0; }();
    static const bool kda_colthread = [] { const char * e = getenv("GGML_GDN_KDA_COLTHREAD"); return !e || atoi(e) != 0; }();
    if (colthread && (!KDA || (kda_colthread && ((uintptr_t) g_d) % 16 == 0)) && S_v == 128 && (sq2 % 4) == 0 &&
            ((uintptr_t) q_d) % 16 == 0 && ((uintptr_t) k_d) % 16 == 0 && ((uintptr_t) s_d) % 16 == 0) {
        static const int lpc = [] {
            const char * e = getenv("GGML_GDN_LPC");
            return e ? atoi(e) : (GGML_CUDA_CC_IS_GCN(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) ? 8 : 4);
        }();
#ifdef GGML_EXP_GDN_NOGATHER
        const int32_t * colthread_ids = state_ids;
#else
        const int32_t * colthread_ids = nullptr;
#endif
        const dim3 grid(H, n_seqs, 128 / GDN_CT_COLS);
        if (lpc == 16) {
            gated_delta_net_colthread<128, keep_rs_t, 16, false, GDN_CT_COLS, KDA><<<grid, GDN_CT_COLS * 16, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                0.0f, 0.0f, 1.0f, 1.0f, colthread_ids);
        } else if (lpc == 8) {
            gated_delta_net_colthread<128, keep_rs_t, 8, false, GDN_CT_COLS, KDA><<<grid, GDN_CT_COLS * 8, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                0.0f, 0.0f, 1.0f, 1.0f, colthread_ids);
        } else if (lpc == 2) {
            gated_delta_net_colthread<128, keep_rs_t, 2, false, GDN_CT_COLS, KDA><<<grid, GDN_CT_COLS * 2, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                0.0f, 0.0f, 1.0f, 1.0f, colthread_ids);
        } else {
            gated_delta_net_colthread<128, keep_rs_t, 4, false, GDN_CT_COLS, KDA><<<grid, GDN_CT_COLS * 4, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                0.0f, 0.0f, 1.0f, 1.0f, colthread_ids);
        }
        return;
    }

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
    }
}

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    return {
        (const float *) q->data,
        (const float *) k->data,
        (const float *) v->data,
        (const float *) g->data,
        (const float *) beta->data,
        (const float *) state->data,
        (float *) dst->data,
        nullptr,
        H,
        q->ne[1],
        v->ne[2],
        v->ne[3],
        v->ne[3] / q->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)),
        (int64_t) (q->nb[2] / sizeof(float)),
        (int64_t) (q->nb[3] / sizeof(float)),
        (int64_t) (v->nb[1] / sizeof(float)),
        (int64_t) (v->nb[2] / sizeof(float)),
        (int64_t) (v->nb[3] / sizeof(float)),
        (int64_t) (beta->nb[1] / sizeof(float)),
        (int64_t) (beta->nb[2] / sizeof(float)),
        (int64_t) (beta->nb[3] / sizeof(float)),
        1.0f / sqrtf((float) S_v),
        g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) < v->ne[2] && S_v == 128 && q->ne[0] == 128,
    };
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    GGML_ASSERT(neq3 > 0 && nev3 >= neq3 && nev3 % neq3 == 0);
    GGML_ASSERT(nek3 == neq3);
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        args.state_out = state_d;
        if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
            return;
        }
    }
    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
