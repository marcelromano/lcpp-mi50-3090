// third-version specializations, alongside (not replacing) upstream K/V instances.
#include "../fattn-b4.cuh"
#include "../fattn-common.cuh"

#if defined(GGML_USE_HIP)
#define GGML_EXP_FA_LC
#include "../fattn-vec-lc.cuh"
#elif !defined(GGML_USE_MUSA)
#define GGML_EXP_CUDA_FA_KVQ_GQA
#define GGML_EXP_CUDA_FA_PI8V2
#define GGML_EXP_CUDA_FA_PI8V2_QK 1
#include "../fattn-kvq-gqa.cuh"
#include "../fattn-pi8v2.cuh"
#endif

static bool fattn_b4_single_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_CUDA_NO_FA) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(ctx, dst);
    return false;
#else
    // Sparse compaction remains native. Row gather is supported only on CUDA.
    if (ggml_get_op_params_i32(dst, 4) != 0) return false;
#if defined(GGML_USE_HIP)
    if (dst->src[5]) return false;
#endif
    const char * enabled = getenv("GGML_FA_B4");
    if (!enabled || atoi(enabled) != 1) return false;
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (dst->src[0]->ne[0] != 256 || !ggml_is_contiguous(dst)) return false;
    // Only the K/V pairs used in the park: all other combinations retain native FA.
    const bool q8q4 = K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0;
    const bool q8q8 = K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0;
    const bool q4q4 = K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0;
    if (!q8q4 && !q8q8 && !q4q4) return false;
#if defined(GGML_USE_HIP)
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_GCN(cc)) return false;
    // Native q4-q4 is more accurate for q=3/6; native vec also wins q=1 at short KV.
    if (!q4q4 && (dst->src[0]->ne[1] > 1 || K->ne[1] >= 4096) &&
        ggml_cuda_flash_attn_ext_vec_lc(ctx, dst)) {
        static unsigned long long count = 0;
        const auto n = count++;
        if (const char * e = getenv("GGML_FA_B4_VERBOSE"); e && atoi(e) && (n < 4 || n % 2000 == 0)) fprintf(stderr, "[fa-b4] kernel=vec-lc n=%llu rows=0\n", n);
        return true;
    }
    return false;
#else
    // The new upstream MMA path wins small prompt batches; reserve PI8V2 for wide tiles.
    if (q8q4 && dst->src[0]->ne[1] >= 512 && ggml_cuda_fa_pi8v2_applies(ctx.device, dst)) {
        ggml_cuda_fa_pi8v2(ctx, dst);
        return true;
    }
    // q=1 native quant-KV has 15.8x less NMSE; six queries amortize only at longer KV.
    if (q8q4 && dst->src[0]->ne[1] >= 2 &&
        (dst->src[0]->ne[1] < 6 || (dst->src[5] ? dst->src[5]->ne[0] : K->ne[1]) >= 4096) && ggml_cuda_fa_kvq_gqa_applies(ctx.device, dst)) {
        ggml_cuda_fa_kvq_gqa(ctx, dst);
        return true;
    }
    return false;
#endif
#endif
}

bool ggml_cuda_fattn_b4_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(GGML_CUDA_NO_FA)
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * R = dst->src[5];
    const ggml_tensor * M = dst->src[3];
    if (R && Q->ne[3] > 1) {
        if (!ggml_is_contiguous(dst) || R->type != GGML_TYPE_I32 || !ggml_is_contiguous(R) ||
            R->ne[1] != Q->ne[3] || R->ne[2] != 1 || R->ne[3] != 1 ||
            !M || M->ne[2] != 1 || M->ne[3] != Q->ne[3]) return false;
        // All slices have the same metadata. Test eligibility on the first slice before dispatching the rest.
        for (int64_t s = 0; s < Q->ne[3]; ++s) {
            ggml_tensor q = *Q, m = *M, r = *R, out = *dst;
            q.ne[3] = m.ne[3] = r.ne[1] = out.ne[3] = 1;
            q.data = (char *) Q->data + s*Q->nb[3];
            m.data = (char *) M->data + s*M->nb[3];
            r.data = (char *) R->data + s*R->nb[1];
            out.data = (char *) dst->data + s*dst->nb[3];
            out.src[0] = &q; out.src[3] = &m; out.src[5] = &r;
            const bool applied = fattn_b4_single_try(ctx, &out);
            if (s == 0 && !applied) return false;
            GGML_ASSERT(applied);
        }
        return true;
    }
#endif
    return fattn_b4_single_try(ctx, dst);
}
