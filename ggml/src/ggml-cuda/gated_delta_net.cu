#include "gated_delta_net.cuh"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"
#include <cstring>

#ifdef GGML_EXP_GDN_NOGATHER
// optional src[6]: I32 cell ids; when set, src[5] is a view of the whole recurrent cache and sequence s reads cell state_ids[s]
#    define GDN_IDS_PARAM , const int32_t * state_ids
#    define GDN_IDS_ARG   , state_ids
#else
#    define GDN_IDS_PARAM
#    define GDN_IDS_ARG
#endif

template <int S_v, bool KDA, bool keep_rs_t>
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
                                     int           K GDN_IDS_PARAM) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
#ifdef GGML_EXP_GDN_NOGATHER
    const int64_t state_in_offset      = (state_ids ? (int64_t) state_ids[sequence] : (int64_t) sequence) * H * S_v * S_v + h_idx * S_v * S_v;
#else
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
#endif
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
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
            const int i = r * warp_size + lane;
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
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

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

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
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
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
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
        float scale, int64_t state_slot_stride, int K GDN_IDS_PARAM, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K GDN_IDS_ARG);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K GDN_IDS_ARG);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K GDN_IDS_ARG);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K GDN_IDS_ARG);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#ifdef GGML_EXP_GDN_DEC2
// ---- GGML_EXP_GDN_DEC2: decode / small-batch (n_tokens <= GGML_EXP_GDN_DEC2_MAXT) GDN kernel for wave64 (gfx9) ----
// S_v == 128, scalar gate (kda == false). Differences from gated_delta_net_cuda:
//  * LPC lanes per state column instead of one wave per column: each lane holds 128/LPC floats as float4 pieces
//    (lane j owns float4 j, j+LPC, ...), so one float4 load instruction moves LPC x 16 B contiguous per column;
//  * the reduction over the 128 rows is 3-4 DPP adds inside the lane group (quad_perm, row_half_mirror, row_mirror)
//    instead of 2 x 6 shfl_xor steps (ds_bpermute);
//  * 64/LPC columns per wave and GGML_EXP_GDN_DEC2_WAVES waves per block -> 1024..1536 small waves, all resident at once;
//  * the next token's k/q/v/g/beta are loaded (and exp(g) computed) before the current token's reduction chain.
// The sum order differs from the stock kernel (per-lane serial FMAs, then a DPP tree): results are not bit-identical.
#    ifndef GGML_EXP_GDN_DEC2_LPC
#        define GGML_EXP_GDN_DEC2_LPC 16 // 8 or 16
#    endif
#    ifndef GGML_EXP_GDN_DEC2_MAXT
#        define GGML_EXP_GDN_DEC2_MAXT 8
#    endif
#    ifndef GGML_EXP_GDN_DEC2_WAVES
#        define GGML_EXP_GDN_DEC2_WAVES 2
#    endif

template <int CTRL>
static __device__ __forceinline__ float gdn_dec2_dpp_add(const float x) {
#    if defined(GGML_USE_HIP) && defined(__GFX9__)
    return x + __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), CTRL, 0xF, 0xF, false));
#    else
    return x; // never launched off gfx9 (host check); keeps CUDA builds compiling
#    endif
}

template <int LPC>
static __device__ __forceinline__ float gdn_dec2_group_sum(float x) {
    static_assert(LPC == 8 || LPC == 16, "LPC must be 8 or 16");
    x = gdn_dec2_dpp_add<0xB1>(x);  // quad_perm [1,0,3,2]
    x = gdn_dec2_dpp_add<0x4E>(x);  // quad_perm [2,3,0,1]
    x = gdn_dec2_dpp_add<0x141>(x); // row_half_mirror: adds the other quad of the 8-lane group
    if constexpr (LPC == 16) {
        x = gdn_dec2_dpp_add<0x140>(x); // row_mirror: adds the other half of the 16-lane row
    }
    return x;
}

template <int NV>
struct gdn_dec2_tok {
    float4 k[NV];
    float4 q[NV];
    float  v;
    float  eg; // exp(g)
    float  beta;
};

template <int LPC, bool keep_rs_t>
__global__ void __launch_bounds__(64 * GGML_EXP_GDN_DEC2_WAVES)
gated_delta_net_dec2(const float * q, const float * k, const float * v, const float * g, const float * beta,
                     const float * curr_state, float * dst, float * state,
                     int64_t H, int64_t n_tokens, int64_t n_seqs,
                     int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
                     int64_t sb1, int64_t sb2, int64_t sb3,
                     const uint3 neqk1_magic, const uint3 rq3_magic, float scale,
                     int64_t state_slot_stride, int K, const int32_t * state_ids) {
    constexpr int S_v = 128;
    constexpr int CPW = 64 / LPC;        // columns per wave
    constexpr int NV  = S_v / (4 * LPC); // float4 per lane per column

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      sub      = lane % LPC;
    const int      col      = (blockIdx.z * blockDim.y + threadIdx.y) * CPW + lane / LPC;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t  cell      = state_ids ? (int64_t) state_ids[sequence] : (int64_t) sequence;
    const float4 * s_in      = (const float4 *) (curr_state + (cell * H + h_idx) * S_v * S_v + (int64_t) col * S_v);
    float *        s_out     = state + (sequence * H + h_idx) * S_v * S_v + (int64_t) col * S_v;
    float *        attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

    ggml_cuda_pdl_sync();

    float4 s[NV];
#pragma unroll
    for (int m = 0; m < NV; m++) {
        s[m] = s_in[sub + LPC * m];
    }

    const auto load_tok = [&](const int t, gdn_dec2_tok<NV> & o) {
        const float4 * kp = (const float4 *) (k + iq3 * sq3 + t * sq2 + iq1 * sq1);
        const float4 * qp = (const float4 *) (q + iq3 * sq3 + t * sq2 + iq1 * sq1);
#pragma unroll
        for (int m = 0; m < NV; m++) {
            o.k[m] = kp[sub + LPC * m];
            o.q[m] = qp[sub + LPC * m];
        }
        const int64_t gb = sequence * sb3 + t * sb2 + h_idx * sb1;
        o.v    = v[sequence * sv3 + t * sv2 + h_idx * sv1 + col];
        o.eg   = expf(g[gb]);
        o.beta = beta[gb];
    };

    gdn_dec2_tok<NV> cur;
    load_tok(0, cur);

    for (int t = 0; t < n_tokens; t++) {
        gdn_dec2_tok<NV> nxt;
        if (t + 1 < n_tokens) {
            load_tok(t + 1, nxt);
        }

        float kv = 0.0f;
#pragma unroll
        for (int m = 0; m < NV; m++) {
            kv += s[m].x * cur.k[m].x;
            kv += s[m].y * cur.k[m].y;
            kv += s[m].z * cur.k[m].z;
            kv += s[m].w * cur.k[m].w;
        }
        kv = gdn_dec2_group_sum<LPC>(kv);

        const float delta = (cur.v - cur.eg * kv) * cur.beta;

        float ap = 0.0f;
#pragma unroll
        for (int m = 0; m < NV; m++) {
            s[m].x = cur.eg * s[m].x + cur.k[m].x * delta;
            ap += s[m].x * cur.q[m].x;
            s[m].y = cur.eg * s[m].y + cur.k[m].y * delta;
            ap += s[m].y * cur.q[m].y;
            s[m].z = cur.eg * s[m].z + cur.k[m].z * delta;
            ap += s[m].z * cur.q[m].z;
            s[m].w = cur.eg * s[m].w + cur.k[m].w * delta;
            ap += s[m].w * cur.q[m].w;
        }
        ap = gdn_dec2_group_sum<LPC>(ap);

        if (sub == 0) {
            attn_data[col] = ap * scale;
        }
        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // same snapshot mapping as the stock kernel: slot 0 = most recent state
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float4 * so = (float4 *) (s_out + target_slot * state_slot_stride);
#pragma unroll
                for (int m = 0; m < NV; m++) {
                    so[sub + LPC * m] = s[m];
                }
            }
        }
        if (t + 1 < n_tokens) {
            cur = nxt;
        }
    }

    if constexpr (!keep_rs_t) {
        float4 * so = (float4 *) s_out;
#pragma unroll
        for (int m = 0; m < NV; m++) {
            so[sub + LPC * m] = s[m];
        }
    }
}

// returns false (nothing launched) when the DEC2 kernel does not apply; the caller then takes the stock path
static bool gdn_dec2_try_launch(
        const float * q_d, const float * k_d, const float * v_d, const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d, const int32_t * state_ids,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, bool keep_rs, cudaStream_t stream) {
    static const bool enabled = [] { const char * e = getenv("GGML_GDN_DEC2"); return !(e && e[0] == '0'); }();
    constexpr int LPC = GGML_EXP_GDN_DEC2_LPC;
    constexpr int NW  = GGML_EXP_GDN_DEC2_WAVES;
    constexpr int CPW = 64 / LPC;
    if (!enabled || S_v != 128 || n_tokens < 1 || n_tokens > GGML_EXP_GDN_DEC2_MAXT ||
        ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != 64 || (128 % (CPW * NW)) != 0) {
        return false;
    }
    const auto al16 = [](const void * p) { return ((uintptr_t) p & 15) == 0; };
    if (!al16(q_d) || !al16(k_d) || !al16(s_d) || !al16(dst_d) || !al16(state_d) ||
        (sq1 & 3) || (sq2 & 3) || (sq3 & 3) || (state_slot_stride & 3)) {
        return false;
    }

    dim3        grid_dims(H, n_seqs, 128 / (CPW * NW));
    dim3        block_dims(64, NW, 1);
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    if (keep_rs) {
        ggml_cuda_kernel_launch(gated_delta_net_dec2<LPC, true>, launch_params,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids);
    } else {
        ggml_cuda_kernel_launch(gated_delta_net_dec2<LPC, false>, launch_params,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids);
    }
    return true;
}
#endif // GGML_EXP_GDN_DEC2

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

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
#ifdef GGML_EXP_GDN_NOGATHER
    const int32_t * state_ids = dst->src[6] ? (const int32_t *) dst->src[6]->data : nullptr;
    GGML_ASSERT(!dst->src[6] || dst->src[6]->type == GGML_TYPE_I32);
    GGML_ASSERT(!dst->src[6] || n_seqs == 1); // cross-sequence reads of cells written by this kernel are not handled
#endif
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

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // Opt-in backport of upstream #29353. The recurrent-state contract promises ALL last-K snapshots,
    // unlike the donor's newer final-state-only prefill contract. Compute the
    // prefix with MMA into slot K-1, then replay the final K-1 tokens with the
    // unchanged scalar kernel. Single-sequence gating also preserves dst strides.
    static const bool mma_enabled = [] {
        const char * e = getenv("GGML_CUDA_GDN_MMA");
        return e && strcmp(e, "1") == 0;
    }();
    const int tail = keep_rs ? K - 1 : 0;
    const int64_t prefix = n_tokens - tail;
    if (mma_enabled && !kda && S_v == 128 && neq0 == 128 && n_seqs == 1 && prefix >= 64) {
#ifdef GGML_EXP_GDN_NOGATHER
        const int32_t * mma_ids = state_ids;
#else
        const int32_t * mma_ids = nullptr;
#endif
        float * prefix_state = state_d + tail * state_slot_stride;
        ggml_cuda_gdn_mma_args args = {
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, prefix_state,
            H, neqk1, prefix, n_seqs, rq3,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, scale, true, mma_ids,
        };
        if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
            static int logged = 0;
            if (logged++ < 4) fprintf(stderr, "GDN_MMA_R1 prefix=%lld tail=%d K=%d ids=%d\n",
                (long long) prefix, tail, K, mma_ids != nullptr);
            if (tail > 0) {
                // The prefix snapshot is compact, so input IDs MUST be null here.
                launch_gated_delta_net<false, true>(
                    q_d + prefix * sq2, k_d + prefix * sq2, v_d + prefix * sv2,
                    g_d + prefix * sb2, b_d + prefix * sb2, prefix_state,
                    dst_d + prefix * S_v * H, state_d,
                    S_v, H, tail, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K
#ifdef GGML_EXP_GDN_NOGATHER
                    , nullptr
#endif
                    , stream);
            }
            return;
        }
    }
#endif

#ifdef GGML_EXP_GDN_DEC2
    if (!kda) {
#    ifdef GGML_EXP_GDN_NOGATHER
        const int32_t * dec2_ids = state_ids;
#    else
        const int32_t * dec2_ids = nullptr;
#    endif
        if (gdn_dec2_try_launch(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, dec2_ids,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3,
                scale, state_slot_stride, K, keep_rs, stream)) {
            return;
        }
    }
#endif // GGML_EXP_GDN_DEC2

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K GDN_IDS_ARG, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K GDN_IDS_ARG, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K GDN_IDS_ARG, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K GDN_IDS_ARG, stream);
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
