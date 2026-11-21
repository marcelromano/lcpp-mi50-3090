#include "llama-context.h"

#include "ggml.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-checkpoint-plan.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-kv-cells.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "llama-ext.h"
#include "llama-sampler.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <mutex>
#include <set>

// CPU wall instrumentation only: no extra GPU synchronization.
struct t2012_scope {
    const char * label; int64_t start; bool enabled;
    explicit t2012_scope(const char * label) : label(label), start(ggml_time_us()), enabled(getenv("T2012_TRACE")) {}
    ~t2012_scope() { const auto elapsed = ggml_time_us()-start; if (enabled && elapsed > 100) fprintf(stderr, "[t2012-time] %s us=%lld\n", label, (long long)elapsed); }
};

// instrumentação da rota de cópia do save/restore do checkpoint device (env-gated).
static bool t2022_trace() {
    static const bool enabled = getenv("T2012_TRACE") != nullptr;
    return enabled;
}

//
// llama_context
//

// check that each folded weight in the graph gets its Hadamard transform, and each latent lookup gets the inverse
// without this check, an arch that skips the transform helpers loads and computes wrong results
static void llama_verify_hadamard_graph(
        ggml_cgraph * gf,
        const llama_hadamard_rotations & rotations,
        const llama_hadamard_rotations & inverses,
        const llama_moe_cache * moe_cache) {
    // the MoE cache gives the matmul a copy of the expert weights, so the copy also needs the transform of its source
    llama_hadamard_rotations forward = rotations;
    for (const auto & [w, t] : rotations) {
        if (const ggml_tensor * cached = moe_cache ? moe_cache->get_experts(w) : nullptr) {
            forward.emplace(cached, t);
        }
    }

    auto unwrap = [](const ggml_tensor * t) {
        while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW)) {
            t = t->src[0];
        }
        return t;
    };

    std::map<const ggml_tensor *, bool> lookups; // get_rows results of latent tables

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * node = ggml_graph_node(gf, i);

        if (node->op == GGML_OP_GET_ROWS && inverses.count(node->src[0])) {
            lookups.emplace(node, false);
            continue;
        }

        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }

        if (node->op == GGML_OP_MUL_MAT && ((const int32_t *) node->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD) {
            const auto lk = lookups.find(unwrap(node->src[1]));
            if (lk != lookups.end()) {
                lk->second = true;
            }
            continue;
        }

        const auto it = forward.find(node->src[0]);
        if (it == forward.end()) {
            if (inverses.count(node->src[0])) {
                throw std::runtime_error(format(
                    "Hadamard-latent table '%s' is used as a head without a forward transform", node->src[0]->name));
            }
            continue;
        }
        const ggml_tensor * src = unwrap(node->src[1]);
        const bool transformed = src && src->op == GGML_OP_MUL_MAT &&
            ((const int32_t *) src->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD &&
            src->src[0] == it->second.rot;
        if (!transformed) {
            throw std::runtime_error(format(
                "Hadamard-folded weight '%s' is consumed without its activation transform; "
                "this graph's matmul path does not support prism.hadamard folding",
                node->src[0]->name));
        }
    }

    for (const auto & [node, ok] : lookups) {
        if (!ok) {
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                const ggml_tensor * n2 = ggml_graph_node(gf, i);
                for (int s = 0; s < GGML_MAX_SRC && n2->src[s]; ++s) {
                    if (unwrap(n2->src[s]) == node) {
                        LLAMA_LOG_WARN("%s: latent lookup '%s' consumed by op=%s name='%s' src%d hint=%d\n",
                                __func__, node->name, ggml_op_name(n2->op), n2->name, s,
                                ((const int32_t *) n2->op_params)[1]);
                    }
                }
            }
            throw std::runtime_error(format(
                "Hadamard-latent table '%s' is read without the inverse transform",
                node->src[0]->name));
        }
    }
}

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_flash_attn_kv_rows_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN_KV_ROWS,
    /*.name             =*/ "Flash Attention over K/V rows",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/ LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/ "Lightning Indexer",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/ "fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/ "fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/ "fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_top_k_mask_probe = {
    /*.op               =*/ LLM_FUSED_OP_TOP_K_MASK,
    /*.name             =*/ "fused top-k mask",
    /*.n_tokens_per_seq =*/ 1,
};

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    // MTP uses the embd input for the hidden state
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd(),
                llm_arch_supports_mixed_batch(model.arch) && params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT)) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && (!llm_arch_supports_rs_rollback(model.arch) || !hparams.causal_attn)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;

    // +1: id n_layer() taps the output of the last layer ("input" of the head)
    cparams.embeddings_layer_inp.resize(hparams.n_layer() + 1, false);
    cparams.embeddings_layer_inp_dev = false;
    cparams.embeddings_nextn_dev     = false;
    cparams.ctx_state_src            = nullptr;
    embd_layer_inp.resize(hparams.n_layer() + 1);

    cparams.ctx_type          = params.ctx_type;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.pooling_type      = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        } else if (model.arch == LLM_ARCH_DFLASH) {
            // used to read the target's layer input device buffer
            cparams.ctx_other = params.ctx_other;
        }
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        cparams.rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    // the KV cache is shared with ctx_other, so follow its flash attention setting instead of resolving it here:
    // the probe disables it when the layers and the shared KV are on different devices, which breaks a quantized V cache
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        const bool flash_attn_other = cparams.ctx_other->get_cparams().flash_attn;
        if (!cparams.auto_fa && cparams.flash_attn != flash_attn_other) {
            LLAMA_LOG_WARN("%s: flash_attn = %d does not match the context the KV cache is shared with, using %d\n",
                    __func__, cparams.flash_attn, flash_attn_other);
        }
        cparams.flash_attn = flash_attn_other;
        cparams.auto_fa    = false;
    }

    cparams.fused_lid = true;
    cparams.auto_flid = false;

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;

    // only the indexer models build this mask, skip the probe for the others
    cparams.fused_top_k_mask = true;
    cparams.auto_ftkm        = hparams.indexer_top_k > 0;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq == 0 ?
            cparams.n_outputs_max : std::min(params.n_outputs_max_per_seq, cparams.n_outputs_max);

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    cparams.op_offload     = params.op_offload;
    cparams.kv_unified     = params.kv_unified;
    cparams.moe_cache_size      = params.moe_cache_size;
    cparams.moe_cache_cpu_share = params.moe_cache_cpu_share;

    // resolved after flash attention, see resolve_fused_ops()
    {
        cparams.fused_kv_rows = false;

        const char * LLAMA_KV_ROWS_DISABLE = getenv("LLAMA_KV_ROWS_DISABLE");
        if (!LLAMA_KV_ROWS_DISABLE) {
            cparams.auto_fkvr = cparams.kv_unified;
        } else {
            LLAMA_LOG_INFO("%s: LLAMA_KV_ROWS_DISABLE set, disabling kv_rows\n", __func__);
            cparams.auto_fkvr = false;
        }
    }

    // initialized later
    cparams.pipeline_parallel = false;
    cparams.training = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max             = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx                 = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq             = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch               = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch              = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn           = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn            = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified            = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base             = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale            = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq              = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max         = %u\n",   __func__, cparams.n_outputs_max);
    LLAMA_LOG_INFO("%s: n_outputs_max_per_seq = %u\n",   __func__, cparams.n_outputs_max_per_seq);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    {
        const int64_t max_nodes = graph_max_nodes();

        gf_res_reserve = std::make_unique<llm_graph_result>(max_nodes);
        gf_res_active  = std::make_unique<llm_graph_result>(max_nodes);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // tensors of ctx_other (e.g. a shared KV cache) can be on devices that this model does not use
        if (cparams.ctx_other) {
            for (const auto & dev_other : llama_get_model(cparams.ctx_other)->devices) {
                const bool found = std::any_of(model.devices.begin(), model.devices.end(),
                        [&](const llama_device & dev) { return dev.dev == dev_other.dev; });
                if (found) {
                    continue;
                }
                ggml_backend_t backend = ggml_backend_dev_init(dev_other.dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev_other.dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
            /*.no_alloc  =*/ model.no_alloc(),
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides() &&
            cparams.moe_cache_size == 0; // not supported by the MoE cache

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        if (cparams.moe_cache_size > 0) {
            moe_cache = std::make_unique<llama_moe_cache>(model, model.no_alloc(), backend_ptrs, backend_buft, cparams.moe_cache_size, cparams.moe_cache_cpu_share);
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

// gf_cache size limit, defined next to llama_graph_cache_limit() below
static size_t llama_graph_cache_max_tokens();

llama_context::~llama_context() {
    synchronize();
    for (auto & [seq_id, pool] : device_ckpt_pool) {
        for (auto & ckpt : pool) {
            if (ckpt) {
                ckpt->wait_host();
            }
        }
    }
    pending_device_ckpt_war.clear();
    for (auto & [compute_backend, entry] : device_ckpt_streams) {
        GGML_UNUSED(compute_backend);
        if (entry.compute_ready) {
            ggml_backend_event_free(entry.compute_ready);
            entry.compute_ready = nullptr;
        }
        if (entry.copy_backend) {
            ggml_backend_free(entry.copy_backend);
            entry.copy_backend = nullptr;
        }
    }
    device_ckpt_streams.clear();

    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();

    // when training, ggml_opt allocates extra buffers through the scheduler, so the sizes no longer match the expectation
    if (getenv("T2046_DIAG") && *getenv("T2046_DIAG") != '0') {
        LLAMA_LOG_INFO("[t2046-ctx-diag] ctx_type=%d n_reused=%d slot_hits=%d slot_misses=%d pad_applied=%d pad_fallback=%d realloc_same=%d realloc_diff=%d cache_size=%zu\n",
                (int) cparams.ctx_type, n_reused, n_slot_hits, n_slot_misses, n_pad_applied, n_pad_fallback, n_reallocs_same, n_reallocs_diff, gf_cache.size());
        LLAMA_LOG_INFO("[t2052-ctx-diag] ctx_type=%d cache_limit=%zu cache_skip=%d active_ntok=%d\n",
                (int) cparams.ctx_type, llama_graph_cache_max_tokens(), n_cache_skip, gf_active_ntok);
    }

    if (!model.no_alloc() && !opt_ctx) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func = __func__;
    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq*n_seqs;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        func, node.il,
                        device_layer ? ggml_backend_dev_name(device_layer) : "none",
                        probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    // the fallback attends to the whole unified cache with a dense mask
    if (cparams.auto_fkvr) {
        cparams.fused_kv_rows = cparams.flash_attn;
        resolve(llm_fused_op_flash_attn_kv_rows_probe, cparams.fused_kv_rows);
        cparams.auto_fkvr = false;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe,  cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }

    if (cparams.auto_ftkm) {
        LLAMA_LOG_INFO("%s: resolving fused top-k mask support:\n", func);
        resolve(llm_fused_op_top_k_mask_probe, cparams.fused_top_k_mask);
        cparams.auto_ftkm = false;
    }
}

static int llama_graph_n_input_tensors(ggml_cgraph * gf) {
    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            users[node].push_back(node);
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * src = node->src[j];
            if (!src) {
                break;
            }
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                users[src].push_back(node);
            }
        }
    }

    for (const auto & [tensor, nodes] : users) {
        GGML_ASSERT(tensor->op == GGML_OP_NONE);
        for (const ggml_tensor * node : nodes) {
            LLAMA_LOG_DEBUG("%s: input tensor '%32s' [%s, ne = { %5" PRId64 ", %5" PRId64 ", %5" PRId64 ", %5" PRId64 " }] is used by node '%s' (%s)\n",
                    __func__, tensor->name, ggml_type_name(tensor->type),
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                    node->name, ggml_op_name(node->op));
        }
    }

    return (int) users.size();
}

void llama_context::sched_new(const sched_params & params) {
    // reusing the scheduler avoids reallocating its buffers (incl. the compute buffers)
    if (sched && params.can_reuse(sched_params_cur)) {
        LLAMA_LOG_DEBUG("%s: reusing scheduler: max_nodes = %zu (current max_nodes = %zu)\n",
                __func__, params.max_nodes, sched_params_cur.max_nodes);
        return;
    }

    sched.reset(ggml_backend_sched_new(
            backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), params.max_nodes, params.parallel, params.op_offload));

    ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);
    ggml_backend_sched_set_realloc_callback(sched.get(), sched_graph_realloc, this);

    sched_params_cur = params;
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes();

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    gf_cache_clear();

    gf_res_reserve->set_max_nodes(max_nodes);
    gf_res_reserve->reset();
    if (gf_res_active) {
        gf_res_active->set_max_nodes(max_nodes);
        gf_res_active->reset();
    }

    sched_new({
        /*.max_nodes  =*/ max_nodes,
        /*.parallel   =*/ cparams.pipeline_parallel,
        /*.op_offload =*/ cparams.op_offload,
    });

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    // reserve worst-case graph
    int n_splits_pp        = -1;
    int n_nodes_pp         = -1;
    int n_inputs_pp        = -1;
    int n_input_tensors_pp = -1;

    int n_splits_tg        = -1;
    int n_nodes_tg         = -1;
    int n_inputs_tg        = -1;
    int n_input_tensors_tg = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.no_alloc(), model.no_alloc() ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched_new({
                    /*.max_nodes  =*/ max_nodes,
                    /*.parallel   =*/ false,
                    /*.op_offload =*/ cparams.op_offload,
                });
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp         = ggml_graph_n_nodes(gf);
        n_inputs_pp        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_pp = this->n_input_tensors;

        graph_has_logits = get_gf_res_reserve()->get_logits() != nullptr;
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    if (cparams.training) {
        // no tg graph for training
        n_splits_tg = n_splits_pp;
        n_nodes_tg  = n_nodes_pp;
    } else {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.no_alloc());
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg         = ggml_graph_n_nodes(gf);
        n_inputs_tg        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_tg = this->n_input_tensors;
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: the worst case graph is not always reached for `n_seqs > 1`
        //       need to implement a more robust mechanism that tries a few different inputs and analyzes the results
        ggml_cgraph * gf = nullptr;
        switch (model.arch) {
            case LLM_ARCH_KIMI_LINEAR:
            case LLM_ARCH_MINIMAX_01:
                // [TAG_RESERVE_DIAG_DECAY]
                // the `inp_diag_decay` tensor size scales with `n_seq_tokens^2` which
                // makes `n_seqs == 1` use more memory for the compute graph compared to `n_seqs > 1`
                gf = graph_reserve(n_tokens, 1,      n_outputs_pp, mctx.get(), model.no_alloc());
                break;
            default:
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.no_alloc());
        };

        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.no_alloc()) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    {
        const bool diff = n_nodes_pp != n_nodes_tg || n_splits_pp != n_splits_tg ||
                          n_inputs_pp != n_inputs_tg || n_input_tensors_pp != n_input_tensors_tg;

        const auto val = [diff](int v_pp, int v_tg) -> std::string {
            return diff ? format("%d / %d", v_pp, v_tg) : format("%d", v_pp);
        };

        LLAMA_LOG_INFO("%s: graph%s: nodes = %s, splits = %s, input objects = %s, input tensors = %s\n",
                __func__,
                diff ? format(" (pp bs=%d, tg bs=%d)", n_tokens, n_seqs).c_str() : "",
                val(n_nodes_pp, n_nodes_tg).c_str(),
                val(n_splits_pp, n_splits_tg).c_str(),
                val(n_inputs_pp, n_inputs_tg).c_str(),
                val(n_input_tensors_pp, n_input_tensors_tg).c_str());
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    t2012_scope timing("synchronize");
    wait_pending_device_ckpt_war_host();
    for (auto & [compute_backend, entry] : device_ckpt_streams) {
        GGML_UNUSED(compute_backend);
        if (entry.copy_backend && entry.has_pending) {
            ggml_backend_synchronize(entry.copy_backend);
            entry.has_pending = false;
            for (auto & [seq_id, pool] : device_ckpt_pool) {
                GGML_UNUSED(seq_id);
                for (auto & ckpt : pool) {
                    if (ckpt) {
                        ckpt->clear_pending_for(entry.copy_backend);
                    }
                }
            }
        }
    }

    if (!sched) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph results to make sure that they won't be reused
        // TODO: make mctx->apply() report if a graph reserve is needed, then reset graph results only if the memory module reset the scheduler
        gf_cache_reset_results();

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    // the toggle changes the graph shape (pooled: logits only; pooling NONE: the reserve graph outputs all tokens,
    // which crashed ggml_gallocr on the first /completion), so re-reserve on every toggle
    if (cparams.embeddings != value) {
        sched_need_reserve = true;
    }

    cparams.embeddings = value;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    if (cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked) {
        // these flags change the graph shape
        sched_need_reserve = true;
    }

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;

    // the device rows are batch rows of the unmasked extraction
    if (cparams.embeddings_nextn_dev && (!value || masked)) {
        cparams.embeddings_nextn_dev = false;
        h_nextn_dev_alloc();
    }
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;

    if (cparams.embeddings_layer_inp_dev) {
        layer_inp_dev_alloc();
    }
}

void llama_context::set_embeddings_layer_inp_dev(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.embeddings_layer_inp_dev != value) {
        sched_need_reserve = true;
    }

    cparams.embeddings_layer_inp_dev = value;

    layer_inp_dev_alloc();
}

void llama_context::extract_logits(ggml_tensor * tensor, float * dst, size_t n_rows) {
    auto * backend = ggml_backend_sched_get_tensor_backend(sched.get(), tensor);
    GGML_ASSERT(backend != nullptr);
    GGML_ASSERT(dst != nullptr);
    const size_t n_vocab = model.vocab.n_tokens();

    if (!model.d2t || model.split_mode() != LLAMA_SPLIT_MODE_TENSOR) {
        ggml_backend_tensor_get_async(backend, tensor, dst, 0, n_rows*n_vocab*sizeof(float));
        return;
    }

    const size_t n_draft = model.d2t->ne[0];
    GGML_ASSERT(model.d2t->type == GGML_TYPE_I32 || model.d2t->type == GGML_TYPE_I64);
    GGML_ASSERT(n_draft > 0 && n_draft <= n_vocab);
    GGML_ASSERT(tensor->type == GGML_TYPE_F32 && tensor->ne[0] == (int64_t) n_draft);
    GGML_ASSERT(ggml_is_contiguous(tensor));
    GGML_ASSERT(ggml_nbytes(tensor) == n_rows*n_draft*sizeof(float));

    // The host scatter must wait for graph completion, including all shards of
    // a tensor-split result. No full-vocabulary tensor is allocated on the GPU.
    ggml_backend_sched_synchronize(sched.get());
    if (draft_to_target.empty()) {
        draft_to_target.resize(n_draft);
        if (model.d2t->type == GGML_TYPE_I32) {
            ggml_backend_tensor_get(model.d2t, draft_to_target.data(), 0, n_draft*sizeof(int32_t));
        } else {
            std::vector<int64_t> tmp(n_draft);
            ggml_backend_tensor_get(model.d2t, tmp.data(), 0, n_draft*sizeof(int64_t));
            for (size_t i = 0; i < n_draft; ++i) {
                draft_to_target[i] = (int32_t) tmp[i];
            }
        }
        std::vector<bool> seen(n_vocab, false);
        for (const int32_t id : draft_to_target) {
            GGML_ASSERT(id >= 0 && (size_t) id < n_vocab);
            GGML_ASSERT(!seen[id] && "d2t token IDs must be unique");
            seen[id] = true;
        }
    }
    draft_logits.resize(n_rows*n_draft);
    ggml_backend_tensor_get(tensor, draft_logits.data(), 0, draft_logits.size()*sizeof(float));
    std::fill_n(dst, n_rows*n_vocab, -INFINITY);
    for (size_t row = 0; row < n_rows; ++row) {
        for (size_t id = 0; id < n_draft; ++id) {
            dst[row*n_vocab + draft_to_target[id]] = draft_logits[row*n_draft + id];
        }
    }
}

bool llama_context::supports_embd_row() const {
    // the DFlash/DSpark decoder graphs read the device buffer
    if (model.arch != LLM_ARCH_DFLASH) {
        return false;
    }

    const ggml_tensor * src = cparams.ctx_other ? cparams.ctx_other->get_layer_inp_dev() : nullptr;
    if (src == nullptr) {
        return false;
    }

    // one of our backends must be able to read the buffer, e.g. not a draft on another GPU than the target output
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(src->buffer);
    for (const auto & backend : backends) {
        if (ggml_backend_supports_buft(backend.get(), buft)) {
            return true;
        }
    }

    return false;
}

ggml_backend_t llama_context::backend_from_dev(ggml_backend_dev_t dev) const {
    for (const auto & backend : backends) {
        if (ggml_backend_get_device(backend.get()) == dev) {
            return backend.get();
        }
    }
    return nullptr;
}

void llama_context::layer_inp_dev_alloc() {
    synchronize();

    layer_inp_dev     = nullptr;
    layer_inp_dev_dev = nullptr;
    layer_inp_backend = nullptr;
    buf_layer_inp_dev.reset();
    ctx_layer_inp_dev.reset();

    if (!cparams.embeddings_layer_inp_dev) {
        return;
    }

    int64_t n_layer_inp = 0;
    for (bool enabled : cparams.embeddings_layer_inp) {
        n_layer_inp += enabled ? 1 : 0;
    }
    if (n_layer_inp == 0) {
        return;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_layer_inp_dev.reset(ggml_init(params));

    layer_inp_dev = ggml_new_tensor_2d(ctx_layer_inp_dev.get(), GGML_TYPE_F32, n_layer_inp*model.hparams.n_embd, cparams.n_batch);
    ggml_set_name(layer_inp_dev, "layer_inp_dev");

    layer_inp_dev_dev = model.dev_output();
    auto * buft = layer_inp_dev_dev ? ggml_backend_dev_buffer_type(layer_inp_dev_dev) : ggml_backend_cpu_buffer_type();

    buf_layer_inp_dev.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_layer_inp_dev.get(), buft));
    if (!buf_layer_inp_dev) {
        // e.g. a large n_batch that does not fit next to the model, keep the host copies
        LLAMA_LOG_WARN("%s: failed to allocate the layer input device buffer, using host copies\n", __func__);
        cparams.embeddings_layer_inp_dev = false;
        layer_inp_dev     = nullptr;
        layer_inp_dev_dev = nullptr;
        ctx_layer_inp_dev.reset();
        return;
    }

    LLAMA_LOG_INFO("%s: %10s layer input buffer size = %8.2f MiB (%d layers)\n", __func__,
            ggml_backend_buffer_name(buf_layer_inp_dev.get()), ggml_backend_buffer_get_size(buf_layer_inp_dev.get())/1024.0/1024.0, (int) n_layer_inp);

    layer_inp_backend = layer_inp_dev_dev ? backend_from_dev(layer_inp_dev_dev) : nullptr;
    if (layer_inp_backend && !ev_layer_inp_written) {
        ev_layer_inp_written.reset(ggml_backend_event_new(layer_inp_dev_dev));
        ev_layer_inp_read.reset(ggml_backend_event_new(layer_inp_dev_dev));
        ev_layer_inp_read_recorded = false;
    }
}

bool llama_context::h_nextn_dev_alloc() {
    synchronize();

    auto reset = [&]() {
        h_nextn_dev     = nullptr;
        h_nextn_dev_dev = nullptr;
        h_nextn_backend = nullptr;
        buf_h_nextn_dev.reset();
        ctx_h_nextn_dev.reset();
        ev_h_nextn_written_recorded = false;
        ev_h_nextn_read_recorded    = false;
    };

    // __func__ inside the lambda would be "operator()"
    const char * func = __func__;
    auto fail = [&](const char * why) {
        LLAMA_LOG_WARN("%s: %s, extracting the h_nextn rows to host memory\n", func, why);
        cparams.embeddings_nextn_dev = false;
        reset();
        return false;
    };

    reset();

    if (!cparams.embeddings_nextn_dev) {
        return true;
    }

    h_nextn_dev_dev = model.dev_output();
    h_nextn_backend = h_nextn_dev_dev ? backend_from_dev(h_nextn_dev_dev) : nullptr;
    if (h_nextn_backend == nullptr) {
        return fail("no device backend for the model output");
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_h_nextn_dev.reset(ggml_init(params));

    h_nextn_dev = ggml_new_tensor_2d(ctx_h_nextn_dev.get(), GGML_TYPE_F32, model.hparams.n_embd_out(), cparams.n_batch + cparams.n_seq_max);
    ggml_set_name(h_nextn_dev, "h_nextn_dev");

    buf_h_nextn_dev.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_h_nextn_dev.get(), ggml_backend_dev_buffer_type(h_nextn_dev_dev)));
    if (!buf_h_nextn_dev) {
        return fail("failed to allocate the h_nextn device buffer");
    }

    if (!ev_h_nextn_written) {
        ev_h_nextn_written.reset(ggml_backend_event_new(h_nextn_dev_dev));
        ev_h_nextn_read.reset(ggml_backend_event_new(h_nextn_dev_dev));
    }
    if (!ev_h_nextn_written || !ev_h_nextn_read) {
        return fail("the device has no events");
    }

    LLAMA_LOG_INFO("%s: %10s h_nextn buffer size = %8.2f MiB\n", __func__,
            ggml_backend_buffer_name(buf_h_nextn_dev.get()), ggml_backend_buffer_get_size(buf_h_nextn_dev.get())/1024.0/1024.0);

    return true;
}

bool llama_context::set_embeddings_nextn_dev(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    // the device rows are batch rows, so the extraction must not be masked
    if (value && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked)) {
        return false;
    }

    if (cparams.embeddings_nextn_dev != value) {
        sched_need_reserve = true;
    }

    cparams.embeddings_nextn_dev = value;

    return h_nextn_dev_alloc() && cparams.embeddings_nextn_dev == value;
}

bool llama_context::get_h_nextn_rows(int64_t row0, int64_t n_rows, float * out) {
    if (h_nextn_dev == nullptr || row0 < 0 || n_rows < 0 || row0 + n_rows > h_nextn_dev->ne[1]) {
        return false;
    }
    if (ev_h_nextn_written_recorded) {
        ggml_backend_event_synchronize(ev_h_nextn_written.get());
    } else {
        synchronize();
    }
    ggml_backend_tensor_get(h_nextn_dev, out, row0*h_nextn_dev->nb[1], n_rows*h_nextn_dev->nb[1]);
    return true;
}

bool llama_context::set_h_nextn_rows(int64_t row0, int64_t n_rows, const float * data) {
    if (h_nextn_dev == nullptr || row0 < 0 || n_rows < 0 || row0 + n_rows > h_nextn_dev->ne[1]) {
        return false;
    }
    // a reader may still read these rows
    if (ev_h_nextn_read_recorded) {
        ggml_backend_event_synchronize(ev_h_nextn_read.get());
    }
    ggml_backend_tensor_set(h_nextn_dev, data, row0*h_nextn_dev->nb[1], n_rows*h_nextn_dev->nb[1]);
    return true;
}

bool llama_context::set_ctx_state_src(llama_context * ctx_src) {
    if (cparams.ctx_state_src != ctx_src) {
        sched_need_reserve = true;
    }
    cparams.ctx_state_src = ctx_src;
    return supports_state_row();
}

bool llama_context::supports_state_row() const {
    // the MTP graphs read the rows, see llm_graph_context::build_inp_embd_state()
    if (cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP) {
        return false;
    }

    // the rows replace the host state rows, so they must have the width the graph takes for those
    const ggml_tensor * src = cparams.ctx_state_src ? cparams.ctx_state_src->get_h_nextn_dev() : nullptr;
    if (src == nullptr || src->ne[0] != (int64_t) llama_batch_ext_select_n_embd_state(cparams.ctx_type, model.hparams)) {
        return false;
    }

    // one of our backends must be able to read the buffer
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(src->buffer);
    for (const auto & backend : backends) {
        if (ggml_backend_supports_buft(backend.get(), buft)) {
            return true;
        }
    }

    return false;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    // no scheduler reserve needed because graph shapes must not depend on causal_attn, a flip only rebuilds the graph

    cparams.causal_attn = value;
}

bool llama_context::get_causal_attn() const {
    return cparams.causal_attn;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

static size_t llama_graph_cache_limit() {
    static const size_t limit = [] {
        const char * env = getenv("LLAMA_GRAPH_SLOTS");
        if (!env || !*env) {
            return (size_t) 12;
        }
        int val = atoi(env);
        return (size_t) (val > 0 ? val : 1);
    }();
    return limit;
}

// only decode/verify-sized microbatches are worth caching. A prefill ubatch (up to n_ubatch,
// e.g. 512) builds a large graph that (a) evicts the small decode graphs from the LRU and (b) with
// the CUDA/HIP graph pool pays a multi-second capture/instantiation, while its reuse rate is low.
// LLAMA_GRAPH_CACHE_MAX_TOKENS=N sets the limit (default 16); 0 means no limit (previous behavior).
static size_t llama_graph_cache_max_tokens() {
    static const size_t limit = [] {
        const char * env = getenv("LLAMA_GRAPH_CACHE_MAX_TOKENS");
        if (!env || !*env) {
            return (size_t) 16;
        }
        const long val = atol(env);
        return val > 0 ? (size_t) val : (size_t) 0;
    }();
    return limit;
}

static bool llama_graph_cache_size_ok(uint32_t n_tokens) {
    const size_t limit = llama_graph_cache_max_tokens();
    return limit == 0 || (size_t) n_tokens <= limit;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    t2012_scope timing("process-ubatch-host");
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    const uint64_t cur_epoch = sched ? ggml_backend_sched_get_buffer_epoch(sched.get()) : 0;
    auto * res = get_gf_res_prev();
    auto * gf  = res->get_gf();

    // Fast path: reuse currently active graph + scheduler split/alloc state.
    const auto gparams_active = graph_params(res, ubatch, mctx, gtype);
    if (!graph_reuse_disable && gf_res_prev_active == res && active_buf_epoch == cur_epoch && res->can_reuse(gparams_active)) {
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }
        n_reused++;
    } else {
        const size_t max_slots = llama_graph_cache_limit();
        bool cache_hit = false;

        // cache-scope: com as linhas de h_nextn no device (MTP) e o KV de atencao do
        // alvo em mais de uma stream, um grafo de PREFILL estacionado no cache de grafo e
        // revivido depois no galloc compartilhado reaplicando um plano de enderecos que outro
        // grafo em voo ainda usa — o `k_set_rows` que escreve h_nextn_dev vai parar em regiao
        // morta (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION, medido nas janelas 2/4/5). Entao,
        // nesse caso, so grafo de decode/verify entra no cache (a regra da revisão anterior, automatica
        // aqui para nao mexer no 27B, que hoje roda com o cache sem limite).
        const bool prefill_cache_unsafe = [&]() {
            if (!cparams.embeddings_nextn_dev) {
                return false;
            }
            llama_memory_resource_capacity cap = {};
            return llama_memory_get_resource_capacity(llama_get_memory(this), 0, &cap, sizeof(cap)) &&
                cap.attention_known && cap.attention_stream_count > 1;
        }();
        const bool cache_size_ok = llama_graph_cache_size_ok(ubatch.n_tokens) &&
            !(prefill_cache_unsafe && ubatch.n_tokens > 16);
        if (!graph_reuse_disable && max_slots > 1 && sched && cache_size_ok) {
            for (auto & slot : gf_cache) {
                if (!slot.res || !slot.res->get_gf() || !slot.sched ||
                    !ggml_backend_sched_is_alloc(slot.sched.get()) ||
                    slot.buf_epoch != cur_epoch) {
                    continue;
                }
                const auto gparams_slot = graph_params(slot.res.get(), ubatch, mctx, gtype);
                if (slot.res->can_reuse(gparams_slot)) {
                    if (cparams.pipeline_parallel) {
                        ggml_backend_sched_synchronize(sched.get());
                    }
                    if (!ggml_backend_sched_rebind_alloc(slot.sched.get())) {
                        continue;
                    }
                    std::swap(gf_res_active, slot.res);
                    std::swap(sched, slot.sched);
                    std::swap(active_buf_epoch, slot.buf_epoch);
                    slot.last_used = ++gf_cache_tick;

                    res = gf_res_active.get();
                    gf  = res->get_gf();
                    gf_res_prev_active = res;
                    n_reused++;
                    n_slot_hits++;
                    cache_hit = true;
                    break;
                }
            }
        }

        if (!cache_hit) {
            n_slot_misses++;
            // Preserve the current active (res, sched) in the multi-slot cache if it holds a valid graph.
            // a prefill-sized graph (see llama_graph_cache_max_tokens) is never parked - it is
            // rebuilt in place, keeping the decode/verify graphs resident instead of evicting them.
            const bool cache_prev_ok = cache_size_ok && llama_graph_cache_size_ok((uint32_t) std::max(0, gf_active_ntok));
            if (!cache_prev_ok && gf_res_prev_active == res && res->get_gf() != nullptr) {
                n_cache_skip++;
            }
            if (!graph_reuse_disable && max_slots > 1 && sched && cache_prev_ok &&
                gf_res_prev_active == res && res->get_gf() != nullptr &&
                ggml_backend_sched_is_alloc(sched.get()) && active_buf_epoch == cur_epoch) {
                const size_t max_inactive = max_slots - 1;
                if (gf_cache.size() < max_inactive) {
                    gf_cache_slot parked;
                    parked.res       = std::move(gf_res_active);
                    parked.sched     = std::move(sched);
                    parked.buf_epoch = active_buf_epoch;
                    parked.last_used = ++gf_cache_tick;
                    gf_cache.push_back(std::move(parked));

                    gf_res_active = std::make_unique<llm_graph_result>(graph_max_nodes());
                    sched.reset(ggml_backend_sched_clone_shared(gf_cache.back().sched.get()));
                } else {
                    auto lru_it = std::min_element(gf_cache.begin(), gf_cache.end(),
                        [](const gf_cache_slot & a, const gf_cache_slot & b) {
                            const bool a_empty = !a.res || !a.res->get_gf();
                            const bool b_empty = !b.res || !b.res->get_gf();
                            if (a_empty != b_empty) {
                                return a_empty;
                            }
                            return a.last_used < b.last_used;
                        });
                    std::swap(gf_res_active, lru_it->res);
                    std::swap(sched, lru_it->sched);
                    lru_it->buf_epoch = active_buf_epoch;
                    lru_it->last_used = ++gf_cache_tick;
                }
                res = gf_res_active.get();
            }

            gf_res_prev_active = nullptr;
            res->reset();

            ggml_backend_sched_reset(sched.get());
            ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

            const auto gparams = graph_params(res, ubatch, mctx, gtype);
            const auto graph_start_us = ggml_time_us();
            gf = model.build_graph(gparams);

            if (!gf) {
                LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }

            if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
                ret = GGML_STATUS_ALLOC_FAILED;
                return nullptr;
            }

            const uint64_t new_epoch = ggml_backend_sched_get_buffer_epoch(sched.get());
            if (new_epoch != cur_epoch) {
                for (auto & slot : gf_cache) {
                    if (slot.res) {
                        slot.res->reset();
                    }
                    if (slot.sched) {
                        ggml_backend_sched_reset(slot.sched.get());
                    }
                    slot.buf_epoch = 0;
                }
            }
            active_buf_epoch = new_epoch;

            if (getenv("T2012_TRACE")) fprintf(stderr,"[t2012-time] graph-build-alloc us=%lld\n",(long long)(ggml_time_us()-graph_start_us));
            gf_res_prev_active = res;
        }
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        t2012_scope inputs_timing("set-inputs");
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    const auto compute_start_us = ggml_time_us();
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (getenv("T2012_TRACE")) fprintf(stderr,"[t2012-time] graph-compute-host us=%lld\n",(long long)(ggml_time_us()-compute_start_us));
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    // the graph that just ran is the one the next call may park in gf_cache
    gf_active_ntok = (int32_t) ubatch.n_tokens;

    return res;
}

int llama_context::encode(const llama_batch_ext & batch_inp) {
    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != hparams.n_embd_inp_enc()) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the encoder input %u\n",
                __func__, batch_inp.n_embd, hparams.n_embd_inp_enc());
        return -1;
    }

    if (!batch_inp.tokens.empty() && batch_inp.tokens[0].embd_row >= 0) {
        LLAMA_LOG_ERROR("%s: embd row batches are not supported by encode\n", __func__);
        return -1;
    }

    // note: encode() logits extracted via extract_logits()

    // note: during encode, we skip position continuity checks and output all tokens
    //       a decoder without memory (see decode()) outputs the marked tokens, unless it extracts embeddings
    if (!balloc->init(batch_inp, model.vocab, memory || cparams.embeddings)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    sched_reserve();

    n_queued_tokens += n_tokens;

    const uint32_t n_outputs_all = balloc->get_n_outputs();

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_outputs_all);
        return -2;
    };

    const auto & out_ids = balloc->get_out_ids();
    for (uint32_t i = 0; i < n_outputs_all; ++i) {
        output_ids[out_ids[i]] = i;
    }

    n_outputs = n_outputs_all;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits && n_outputs > 0) {
        extract_logits(t_logits, logits.data, n_outputs);
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_outputs*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_outputs*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_outputs*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_outputs*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch_ext & batch_inp) {
    t2012_scope timing("decode-total-host");
    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != batch_inp.n_embd_inp) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the decoder input %zu\n",
                __func__, batch_inp.n_embd, batch_inp.n_embd_inp);
        return -1;
    }

    const bool has_embd_row = !batch_inp.tokens.empty() && batch_inp.tokens[0].embd_row >= 0;
    if (has_embd_row && !supports_embd_row()) {
        LLAMA_LOG_ERROR("%s: this context cannot read embd rows (needs a DFlash/DSpark decoder and a layer input device buffer in ctx_other)\n", __func__);
        return -1;
    }

    const bool has_state_row = !batch_inp.tokens.empty() && batch_inp.tokens[0].state_row >= 0;
    if (has_state_row && !supports_state_row()) {
        LLAMA_LOG_ERROR("%s: this context cannot read state rows (needs an MTP context and an h_nextn device buffer in ctx_state_src)\n", __func__);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    // non-causal attention needs the whole sequence in one ubatch
    const bool embd_all     = output_all || !cparams.causal_attn;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    // embedding contexts output every token even when no token is explicitly marked as output
    if (has_samplers) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (const auto & tok : batch_inp.tokens) {
            if (!output_all && !tok.output) {
                continue;
            }

            for (auto seq_id : tok.seq_ids) {
                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    continue;
                }

                seq_output_count[seq_id]++;
                auto sampler = sampling.samplers.find(seq_id);
                if (sampler != sampling.samplers.end() &&
                        seq_output_count[seq_id] > (int32_t) cparams.n_outputs_max_per_seq) {
                    LLAMA_LOG_ERROR("%s: backend sampling supports at most %u outputs per sequence "
                            "(seq_id %d had %d)\n", __func__, cparams.n_outputs_max_per_seq,
                            seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    // Bucket-pad small MTP verify/process batches (n_outputs == 0) so the same graph/capture
    // covers variable verification lengths {1..4} -> 4, etc. Pad tokens get pos > seq max
    // so causal masking excludes them from all real queries, and are removed via seq_rm right after.
    const bool mtp_pad_enabled = []() {
        const char * env = getenv("LLAMA_MTP_BATCH_PAD");
        return !env || !*env || std::strcmp(env, "0") != 0;
    }();
    bool         pad_active    = false;
    llama_seq_id pad_seq_id    = -1;
    llama_pos    pad_pos_start = 0;
    std::unique_ptr<llama_batch_ext> batch_padded;

    if (mtp_pad_enabled &&
        cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        cparams.causal_attn &&
        !output_all &&
        !has_embd_row &&
        (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked) &&
        cparams.embeddings_layer_inp.empty() &&
        !batch_inp.tokens.empty()) {
        const uint32_t n_orig = (uint32_t) batch_inp.tokens.size();
        const uint32_t max_ub = std::min(cparams.n_ubatch, cparams.n_batch);
        static constexpr uint32_t k_buckets[] = { 4, 8, 16, 32, 64, 128, 256, 512 };
        uint32_t n_pad = 0;
        for (uint32_t b : k_buckets) {
            if (b >= n_orig && b <= max_ub) {
                n_pad = b;
                break;
            }
        }

        const auto & tok0 = batch_inp.tokens[0];
        bool eligible = n_pad > n_orig && tok0.seq_ids.size() == 1 &&
                        tok0.id != LLAMA_TOKEN_NULL && !tok0.has_embd;
        if (eligible) {
            for (const auto & t : batch_inp.tokens) {
                if (t.output || t.has_embd || t.id == LLAMA_TOKEN_NULL || t.seq_ids.size() != 1) {
                    eligible = false;
                    break;
                }
                if (!cparams.kv_unified && *t.seq_ids.begin() != *tok0.seq_ids.begin()) {
                    eligible = false;
                    break;
                }
            }
        }

        if (eligible) {
            pad_seq_id = *tok0.seq_ids.begin();
            if (pad_seq_id >= 0 && (uint32_t) pad_seq_id < n_seq_max) {
                llama_pos max_batch_pos = -1;
                for (const auto & t : batch_inp.tokens) {
                    if (t.seq_ids.count(pad_seq_id)) {
                        max_batch_pos = std::max(max_batch_pos, t.pos[0]);
                    }
                }
                const llama_pos mem_pos_max = memory->seq_pos_max(pad_seq_id);
                pad_pos_start = std::max<llama_pos>(0, std::max(mem_pos_max, max_batch_pos) + 1);

                batch_padded = std::make_unique<llama_batch_ext>(this);
                batch_padded->n_embd = batch_inp.n_embd;
                batch_padded->tokens = batch_inp.tokens;
                batch_padded->embd   = batch_inp.embd;
                batch_padded->state  = batch_inp.state;

                const size_t n_embd_st = batch_inp.n_embd_state;
                const uint32_t n_add   = n_pad - n_orig;
                for (uint32_t k = 0; k < n_add; ++k) {
                    llama_batch_ext::token ptok = tok0;
                    ptok.output = false;
                    ptok.seq_ids.clear();
                    ptok.seq_ids.insert(pad_seq_id);
                    const llama_pos ppos = pad_pos_start + (llama_pos) k;
                    ptok.pos.fill(ppos);
                    if (tok0.has_state && n_embd_st > 0 &&
                        tok0.state_off + n_embd_st <= batch_padded->state.size()) {
                        ptok.state_off = batch_padded->state.size();
                        batch_padded->state.insert(
                            batch_padded->state.end(),
                            batch_padded->state.begin() + tok0.state_off,
                            batch_padded->state.begin() + tok0.state_off + n_embd_st);
                    }
                    batch_padded->tokens.push_back(std::move(ptok));
                }
                pad_active = true;
                n_pad_applied++;
            }
        }
    }

    const llama_batch_ext & active_batch = pad_active ? *batch_padded : batch_inp;
    if (!balloc->init(active_batch, vocab, output_all)) {
        if (pad_active) {
            pad_active = false;
            n_pad_fallback++;
            batch_padded.reset();
            if (!balloc->init(batch_inp, vocab, output_all)) {
                LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
                return -1;
            }
        } else {
            LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return -1;
        }
    }

    const uint32_t n_tokens_orig = (uint32_t) batch_inp.tokens.size();
    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_orig;

    output_swaps.clear();
    embd_batch_idxs.clear();

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, embd_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (pad_active) {
                        pad_active = false;
                        n_pad_fallback++;
                        batch_padded.reset();
                        if (!balloc->init(batch_inp, vocab, output_all)) {
                            LLAMA_LOG_ERROR("%s: failed to initialize fallback batch\n", __func__);
                            return -1;
                        }
                        continue;
                    }
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    // ordering for the layer input device buffer
    llama_context * inject_src     = has_embd_row ? cparams.ctx_other : nullptr;
    ggml_backend_t  inject_backend = nullptr;
    if (inject_src) {
        inject_backend = inject_src->ev_layer_inp_written ? backend_from_dev(inject_src->layer_inp_dev_dev) : nullptr;
        if (inject_backend) {
            // wait on the host: a device-side wait would stall our queue for the whole target batch,
            // which trips the Metal command buffer timeout on long prompts
            ggml_backend_event_synchronize(inject_src->ev_layer_inp_written.get());
        } else {
            inject_src->synchronize();
        }
    }
    if (layer_inp_dev && ev_layer_inp_read && ev_layer_inp_read_recorded) {
        // do not overwrite rows the reader has not consumed yet
        ggml_backend_event_wait(layer_inp_backend, ev_layer_inp_read.get());
    }

    // ordering for the h_nextn device buffer, as for the layer input buffer
    llama_context * state_src     = has_state_row ? cparams.ctx_state_src : nullptr;
    ggml_backend_t  state_backend = nullptr;
    if (state_src) {
        state_backend = backend_from_dev(state_src->h_nextn_dev_dev);
        if (state_src->ev_h_nextn_written_recorded) {
            ggml_backend_event_synchronize(state_src->ev_h_nextn_written.get());
        } else {
            state_src->synchronize();
        }
    }
    const bool write_h_nextn = cparams.embeddings_nextn_dev && h_nextn_dev != nullptr;
    if (write_h_nextn && ev_h_nextn_read_recorded) {
        ggml_backend_event_wait(h_nextn_backend, ev_h_nextn_read.get());
    }

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        if (!res) {
            if (pad_active) {
                memory->seq_rm(pad_seq_id, pad_pos_start, -1);
            }
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        size_t d2h_logits = 0, d2h_reduced = 0, d2h_nextn = 0;
        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                extract_logits(t_logits, logits_out, n_outputs);
                d2h_logits += (size_t) n_outputs*n_vocab*sizeof(float);
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // [TAG_EXTRACT_TARGET_EMBEDDINGS]
        bool extract_all_idxs = extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            // the device buffer replaces the host extraction, see build_h_nextn_dev()
            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE && !write_h_nextn) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out();
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
                d2h_nextn += (size_t) n_rows*n_embd*sizeof(float);
                extract_all_idxs = extract_all_idxs || !masked;
            }
        }

        if (extract_all_idxs) {
            GGML_ASSERT(ubatch.data && ubatch.data->batch_idxs.size() == ubatch.n_tokens);
            GGML_ASSERT(embd_batch_idxs.size() == (size_t) n_tokens_prev);
            const auto & batch_idxs = ubatch.data->batch_idxs;
            embd_batch_idxs.insert(embd_batch_idxs.end(), batch_idxs.begin(), batch_idxs.end());
        }

        if (has_samplers) {
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched.get());
            copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     stride, n_outputs_prev, sched.get(), &sampling.logits_count);
            copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      stride, n_outputs_prev, sched.get(), &sampling.probs_count);
            copy_tensor_async_rows(res->t_candidates,     sampling.candidates, stride, n_outputs_prev, sched.get(), &sampling.candidates_count);

            for (int32_t local = 0; local < n_outputs; ++local) {
                const uint32_t row = (uint32_t) (n_outputs_prev + local);
                d2h_reduced += (sampling.logits_count[row] + sampling.probs_count[row])*sizeof(float)
                             + sampling.candidates_count[row]*sizeof(llama_token);
                if (local < (int32_t) res->t_sampled.size() && res->t_sampled[local] != nullptr) {
                    d2h_reduced += sizeof(llama_token);
                }
            }

            // Exactness guard for the opt-in top-k prefilter: CPU heap/partial_sort
            // tie ordering is not the device tie ordering. Fetch raw logits only
            // on ambiguous rows, including a tie across the k/k+1 boundary.
            const char * dev_topk_env = getenv("LLAMA_SPEC_DEVICE_TOPK");
            if (dev_topk_env && *dev_topk_env && std::strcmp(dev_topk_env, "0") != 0 && n_outputs > 0) {
                bool synced = false;
                int32_t local = 0;
                for (uint32_t i = 0; i < ubatch.n_tokens && local < n_outputs; ++i) {
                    if (!ubatch.output[i]) {
                        continue;
                    }
                    const int32_t cur_local = local++;
                    if (cur_local < (int32_t) res->t_sampled.size() && res->t_sampled[cur_local] != nullptr) {
                        continue;
                    }
                    llama_sampler * reducer = nullptr;
                    auto it = sampling.samplers.find(ubatch.seq_id[i][0]);
                    if (it != sampling.samplers.end()) {
                        reducer = it->second;
                    }
                    // A two-node [?logit-bias, (+)?top-k] chain marks our prefix.
                    if (!reducer || llama_sampler_chain_n(reducer) != 2 ||
                        std::string(llama_sampler_name(llama_sampler_chain_get(reducer, 0))) != "?logit-bias") {
                        continue;
                    }
                    const std::string second_name = llama_sampler_name(llama_sampler_chain_get(reducer, 1));
                    if (second_name != "top-k" && second_name != "+top-k") {
                        continue;
                    }
                    if (!synced) {
                        ggml_backend_sched_synchronize(sched.get());
                        synced = true;
                    }
                    const uint32_t row = (uint32_t) (n_outputs_prev + cur_local);
                    const uint32_t count = sampling.logits_count[row];
                    const float * values = sampling.logits.data + (size_t) row*stride;
                    bool ambiguous = false;
                    for (uint32_t j = 0; j < count; ++j) {
                        ambiguous |= !std::isfinite(values[j]);
                        if (j > 0) {
                            ambiguous |= values[j] == values[j - 1];
                        }
                    }
                    if (ambiguous && t_logits && logits.data) {
                        auto * backend = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
                        GGML_ASSERT(backend != nullptr);
                        ggml_backend_tensor_get_async(backend, t_logits, logits.data + (size_t) row*stride,
                                (size_t) cur_local*stride*sizeof(float), stride*sizeof(float));
                        d2h_logits += stride*sizeof(float);
                        sampling.logits_count[row] = sampling.probs_count[row] = sampling.candidates_count[row] = 0;
                    } else if (count > 1) {
                        // Drop boundary sentinel, matching the original host prefilter.
                        sampling.logits_count[row] = sampling.candidates_count[row] = count - 1;
                    }
                }
            }
        }

        if (getenv("LLAMA_D2H_COUNTS") && n_outputs > 0) {
            fprintf(stderr, "t1959 D2H rows=%d tokens=%u logits=%zu reduced=%zu nextn=%zu\n",
                n_outputs, ubatch.n_tokens, d2h_logits, d2h_reduced, d2h_nextn);
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());

    if (pad_active) {
        memory->seq_rm(pad_seq_id, pad_pos_start, -1);
    }

    if (layer_inp_dev && ev_layer_inp_written) {
        ggml_backend_event_record(ev_layer_inp_written.get(), layer_inp_backend);
    }
    if (write_h_nextn) {
        ggml_backend_event_record(ev_h_nextn_written.get(), h_nextn_backend);
        ev_h_nextn_written_recorded = true;
    }
    if (state_src) {
        if (state_backend) {
            ggml_backend_event_record(state_src->ev_h_nextn_read.get(), state_backend);
            state_src->ev_h_nextn_read_recorded = true;
        } else {
            synchronize();
        }
    }
    if (inject_src) {
        if (inject_backend) {
            ggml_backend_event_record(inject_src->ev_layer_inp_read.get(), inject_backend);
            inject_src->ev_layer_inp_read_recorded = true;
        } else {
            synchronize();
        }
    }

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = !cparams.embeddings || graph_has_logits;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled && !cparams.embeddings_layer_inp_dev) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (                                                                         backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il] && !cparams.embeddings_layer_inp_dev) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

bool llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    bool extracted = false;
    if (cparams.embeddings_layer_inp_dev) {
        return extracted;
    }
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        GGML_ASSERT(row_floats == model.hparams.n_embd);
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        // Tensor-split backends require a zero source offset.
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
        extracted = true;
    }
    return extracted;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab     = model.vocab.n_tokens();
    const uint64_t n_embd      = model.hparams.n_embd;
    const uint64_t n_embd_out  = model.hparams.n_embd_out();

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd.data[i0*n_embd_out + k], embd.data[i1*n_embd_out + k]);
            }
        }

        if (embd_nextn.size > 0 && cparams.embeddings_nextn_masked) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0*n_embd_out + k], embd_nextn.data[i1*n_embd_out + k]);
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();

    // [TAG_EXTRACT_TARGET_EMBEDDINGS]
    // Layer inputs and unmasked NextN embeddings contain all token rows, independent of logits selection.
    for (size_t i = 0; i < embd_batch_idxs.size(); ++i) {
        while (embd_batch_idxs[i] != (int32_t) i) {
            const int32_t j = embd_batch_idxs[i];
            GGML_ASSERT(j >= 0 && (size_t) j < embd_batch_idxs.size());
            if (embd_nextn.has_data() && !cparams.embeddings_nextn_masked) {
                for (size_t k = 0; k < n_embd_out; ++k) {
                    std::swap(embd_nextn.data[i*n_embd_out + k], embd_nextn.data[j*n_embd_out + k]);
                }
            }
            for (auto & layer : embd_layer_inp) {
                if (layer.has_data()) {
                    for (size_t k = 0; k < n_embd; ++k) {
                        std::swap(layer.data[i*n_embd + k], layer.data[j*n_embd + k]);
                    }
                }
            }
            std::swap(embd_batch_idxs[i], embd_batch_idxs[j]);
        }
    }
    embd_batch_idxs.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes() const {
    uint32_t res;
    if (model.arch == LLM_ARCH_KIMI_K3 || model.arch == LLM_ARCH_KIMI_K31 || model.arch == LLM_ARCH_GLM5_NEXT ||
        model.arch == LLM_ARCH_HRM_TEXT) {
        res = std::max<uint32_t>(1024u, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_METIS16 ||
        model.arch == LLM_ARCH_BAILINGMOE3 ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_CLEF ||
        model.arch == LLM_ARCH_CLOVER_MOE ||
        model.arch == LLM_ARCH_QWEN4EXP ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        model.arch == LLM_ARCH_DEEPSEEK41 ||
        model.arch == LLM_ARCH_XING4_0 ||
        (model.arch == LLM_ARCH_DFLASH && model.hparams.dsv4_hc_mult > 0) ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_01 ||
        model.arch == LLM_ARCH_MINIMAX_M3 ||
        model.arch == LLM_ARCH_HY_V4) {
        res = std::max<uint32_t>(1024u, 32u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft
        res = std::max<uint32_t>(1024u, 12u*model.n_tensors());
    } else {
        res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
        for (const auto & lora : model.loras) {
            res += lora->get_n_nodes();
        }
    }

    // every sampler builds n_outputs_max_per_seq chains, see llm_graph_context::build_sampling
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        res += llama_sampler_backend_n_nodes(sampler) * cparams.n_outputs_max_per_seq;
    }

    if (cparams.training) {
        res *= 4;
    }

    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

llm_graph_result * llama_context::get_gf_res_prev() {
    return gf_res_active.get();
}

void llama_context::gf_cache_clear() {
    gf_cache.clear();
    gf_res_prev_active = nullptr;
    active_buf_epoch   = 0;
}

void llama_context::gf_cache_reset_results() {
    if (gf_res_active) {
        gf_res_active->reset();
    }
    for (auto & slot : gf_cache) {
        if (slot.res) {
            slot.res->reset();
        }
        if (slot.sched) {
            ggml_backend_sched_reset(slot.sched.get());
        }
        slot.buf_epoch = 0;
    }
    gf_res_prev_active = nullptr;
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq) {
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // sequences with a sampler that fit in this ubatch
    std::vector<uint32_t> sampler_seqs;
    std::vector<bool> has_sampler(n_seqs, false);
    for (const auto & entry : samplers) {
        const llama_seq_id seq_id = entry.first;
        if (seq_id < 0 || (uint32_t) seq_id >= n_seqs) {
            continue;
        }

        sampler_seqs.push_back(seq_id);
        has_sampler[seq_id] = true;
    }

    uint32_t n_outputs_set = 0;

    const uint32_t n_outputs_per_seq = std::min(n_seq_tokens, n_outputs_max_per_seq);
    for (uint32_t s : sampler_seqs) {
        if (n_outputs_set >= n_outputs) {
            break;
        }

        for (uint32_t t = 0; t < n_outputs_per_seq && n_outputs_set < n_outputs; ++t) {
            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }

    // use sequences without samplers for any remaining outputs
    for (uint32_t t = 0; t < n_seq_tokens && n_outputs_set < n_outputs; ++t) {
        for (uint32_t s = 0; s < n_seqs && n_outputs_set < n_outputs; ++s) {
            if (has_sampler[s]) {
                continue;
            }

            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse old graphs, so we clear cached slots
    gf_cache_reset_results();
    gf_cache.clear();

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    ubatch_prepare_reserve(ubatch, n_outputs, sampling.samplers, cparams.n_outputs_max_per_seq);

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    auto * gf = model.build_graph(gparams);

    // check the graph before scheduling: cross-backend copies break the producer chain that the check follows
    if (!hadamard_verified && gf && (!model.hdmd.rot.empty() || !model.hdmd.inv.empty())) {
        llama_verify_hadamard_graph(gf, model.hdmd.rot, model.hdmd.inv, moe_cache.get());
        hadamard_verified = true;
    }

    this->n_input_tensors = llama_graph_n_input_tensors(gf);
    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.model       =*/ &model,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.moe_cache   =*/ moe_cache.get(),
        /*.prec_policy =*/ &model.prec_policy,
        /*.hdmd        =*/ model.hdmd.rot.empty() ? nullptr : &model.hdmd,
        /*.layer_inp_dev =*/ layer_inp_dev,
        /*.layer_inp_src =*/ supports_embd_row() ? cparams.ctx_other->get_layer_inp_dev() : nullptr,
        /*.h_nextn_dev   =*/ cparams.embeddings_nextn_dev ? h_nextn_dev : nullptr,
        /*.h_nextn_src   =*/ supports_state_row() ? cparams.ctx_state_src->get_h_nextn_dev() : nullptr,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
    };
}

void llama_context::sched_graph_realloc(ggml_backend_sched_t sched, ggml_cgraph * gf, bool backend_ids_changed, bool graph_size_changed, void * user_data) {
    // reallocating after the backend assignments changed is expected, trace only the unexpected ones
    if (backend_ids_changed) {
        return;
    }

    auto * lctx = static_cast<llama_context *>(user_data);

    if (graph_size_changed) {
        lctx->n_reallocs_diff++;
    } else {
        lctx->n_reallocs_same++;
    }

    LLAMA_LOG_INFO("%s: graph reallocation: nodes = %d (changed = %d), splits = %d, n_same = %d, n_diff = %d\n",
            __func__, ggml_graph_n_nodes(gf), graph_size_changed, ggml_backend_sched_get_n_splits(sched),
            lctx->n_reallocs_same, lctx->n_reallocs_diff);
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    copy_experts.reset();

    wait_pending_device_ckpt_war_gpu();
    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

bool llama_context::sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data) {
    auto * lctx = static_cast<llama_context *>(user_data);

    // the slot maps of the MoE cache
    if (lctx->moe_cache && lctx->moe_cache->copy(backend, src, dst, graph)) {
        return true;
    }

    auto & st = lctx->copy_experts;

    // the ids must be computed before the split starts, so only the first node of the split is considered
    if (ggml_graph_n_nodes(graph) == 0) {
        return false;
    }
    const ggml_tensor * node = ggml_graph_node(graph, 0);
    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != dst) {
        return false;
    }

    const ggml_tensor * ids = node->src[2];
    if (ggml_nelements(ids) == 0) {
        return true;
    }

    const int64_t n_expert    = src->ne[2];
    const size_t  expert_size = src->nb[2];

    if (ids != st.ids || (int64_t) st.used.size() != n_expert) {
        st.ids_data.resize(ggml_nbytes(ids)/sizeof(int32_t));
        ggml_backend_tensor_get_async(backend, ids, st.ids_data.data(), 0, ggml_nbytes(ids));
        ggml_backend_synchronize(backend);

        st.used.assign(n_expert, false);
        for (int64_t i1 = 0; i1 < ids->ne[1]; i1++) {
            for (int64_t i0 = 0; i0 < ids->ne[0]; i0++) {
                const int32_t id = st.ids_data[i1*ids->nb[1]/sizeof(int32_t) + i0*ids->nb[0]/sizeof(int32_t)];
                GGML_ASSERT(id >= 0 && id < n_expert);
                st.used[id] = true;
            }
        }

        st.ids = ids;
    }

    // group consecutive experts and copy them together
    for (int64_t first = 0; first < n_expert; ) {
        if (!st.used[first]) {
            first++;
            continue;
        }
        int64_t last = first;
        while (last + 1 < n_expert && st.used[last + 1]) {
            last++;
        }

        // the experts in the MoE cache are copied from device memory, the others are uploaded
        int64_t next = first;
        for (int64_t e = first; e <= last && lctx->moe_cache; ) {
            const int64_t n = lctx->moe_cache->copy_experts(backend, src, dst, e, last);
            if (n == 0) {
                e++;
                continue;
            }
            if (next < e) {
                ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + next*expert_size, next*expert_size, (e - next)*expert_size);
            }
            e   += n;
            next = e;
        }

        // copy a bit extra to ensure there are no NaNs in the padding of the last expert, this is necessary for MMQ in the CUDA backend
        const size_t offset  = next*expert_size;
        const size_t padding = last < n_expert - 1 ? std::min<size_t>(expert_size, 512) : 0;
        const size_t size    = (last + 1 - next)*expert_size + padding;
        if (size > 0) {
            ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + offset, offset, size);
        }

        first = last + 1;
    }

    return true;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), base_ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        if (winfos.empty()) {
            return;
        }

        std::vector<ggml_tensor *> tensors;
        std::vector<size_t> offsets;
        std::vector<size_t> sizes;
        std::vector<size_t> data_offsets;
        tensors.reserve(winfos.size());
        offsets.reserve(winfos.size());
        sizes.reserve(winfos.size());
        data_offsets.reserve(winfos.size());

        for (const auto & winfo : winfos) {
            tensors.push_back(winfo.tensor);
            offsets.push_back(winfo.offset);
            sizes.push_back(winfo.size);
            data_offsets.push_back(winfo.data_offset);
        }

        std::vector<const ggml_tensor *> const_tensors(tensors.begin(), tensors.end());
        ggml_backend_tensor_get_n(const_tensors.data(), base_ptr, offsets.data(), sizes.data(), data_offsets.data(), winfos.size());
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, (size_t) (ptr - base_ptr), size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    uint8_t * base_ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        size_t data_offset;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), base_ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        if (rinfos.empty()) {
            return;
        }

        std::vector<ggml_tensor *> tensors;
        std::vector<size_t> offsets;
        std::vector<size_t> sizes;
        std::vector<size_t> data_offsets;
        tensors.reserve(rinfos.size());
        offsets.reserve(rinfos.size());
        sizes.reserve(rinfos.size());
        data_offsets.reserve(rinfos.size());

        for (const auto & rinfo : rinfos) {
            tensors.push_back(rinfo.tensor);
            offsets.push_back(rinfo.offset);
            sizes.push_back(rinfo.size);
            data_offsets.push_back(rinfo.data_offset);
        }

        ggml_backend_tensor_set_n(tensors.data(), base_ptr, offsets.data(), sizes.data(), data_offsets.data(), rinfos.size());
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, (size_t) (ptr - base_ptr), size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void discard() override {
        rinfos.clear();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    const uint8_t * base_ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        size_t data_offset;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            GGML_ASSERT(winfo.size % ggml_type_size(winfo.tensor->type) == 0);
            const int64_t n = winfo.size/ggml_type_size(winfo.tensor->type)*ggml_blck_size(winfo.tensor->type);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            std::vector<const ggml_tensor *> srcs(mbuf_cur.org.begin(), mbuf_cur.org.end());
            std::vector<ggml_tensor *> dsts(mbuf_cur.cpy.begin(), mbuf_cur.cpy.end());
            ggml_backend_tensor_copy_n(srcs.data(), dsts.data(), mbuf_cur.org.size());
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            GGML_ASSERT(rinfo.size % ggml_type_size(rinfo.tensor->type) == 0);
            const int64_t n = rinfo.size/ggml_type_size(rinfo.tensor->type)*ggml_blck_size(rinfo.tensor->type);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            if (mbuf_cur.n_tensors == mbuf.n_tensors) {
                // an equal tensor count does not imply the same chunking, e.g. save ranges [2,1] vs restore runs [1,2]
                bool same_chunking = true;
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    if (ggml_nbytes(mbuf_cur.cpy[i]) != ggml_nbytes(mbuf.org[i])) {
                        same_chunking = false;
                        break;
                    }
                }

                if (same_chunking) {
                    // same chunking: copy 1:1 by index
                    std::vector<const ggml_tensor *> srcs(mbuf_cur.cpy.begin(), mbuf_cur.cpy.end());
                    std::vector<ggml_tensor *> dsts(mbuf.org.begin(), mbuf.org.end());
                    ggml_backend_tensor_copy_n(srcs.data(), dsts.data(), mbuf_cur.cpy.size());
                    continue;
                }
            }

            // different chunking: copy the write-side data (mbuf_cur.cpy) into the read-side targets (mbuf.org)
            // with a byte cursor. Write and read enumerate the same logical data in the same order but may chunk
            // it differently (even with an equal number of tensors), so copy across tensor boundaries rather than
            // 1:1 by index.
            const size_t total = mbuf_cur.total_size;

            ggml_init_params params_scratch = {
                /*.mem_size   =*/ 2*(mbuf_cur.cpy.size() + mbuf.org.size())*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx_scratch = ggml_init(params_scratch);

            size_t src_pos  = 0;
            size_t dst_pos  = 0;
            size_t src_j    = 0;
            size_t dst_i    = 0;
            size_t src_base = 0;
            size_t dst_base = 0;

            std::vector<const ggml_tensor *> chunk_srcs;
            std::vector<ggml_tensor *> chunk_dsts;
            chunk_srcs.reserve(mbuf_cur.cpy.size() + mbuf.org.size());
            chunk_dsts.reserve(mbuf_cur.cpy.size() + mbuf.org.size());

            while (src_pos < total) {
                const auto & src_t = mbuf_cur.cpy[src_j];
                const auto & dst_t = mbuf.org[dst_i];

                const size_t src_size = ggml_nbytes(src_t);
                const size_t dst_size = ggml_nbytes(dst_t);

                const size_t src_off  = src_pos - src_base;
                const size_t dst_off  = dst_pos - dst_base;

                const size_t n_copy = std::min(src_size - src_off, dst_size - dst_off);

                GGML_ASSERT(n_copy % ggml_type_size(src_t->type) == 0);
                const int64_t n_el = n_copy / ggml_type_size(src_t->type) * ggml_blck_size(src_t->type);

                auto * src_v = ggml_view_1d(ctx_scratch, src_t, n_el, src_off);
                ggml_backend_view_init(src_v);
                auto * dst_v = ggml_view_1d(ctx_scratch, dst_t, n_el, dst_off);
                ggml_backend_view_init(dst_v);

                chunk_srcs.push_back(src_v);
                chunk_dsts.push_back(dst_v);

                src_pos += n_copy;
                dst_pos += n_copy;

                if (src_pos - src_base == src_size) {
                    src_base = src_pos;
                    ++src_j;
                }
                if (dst_pos - dst_base == dst_size) {
                    dst_base = dst_pos;
                    ++dst_i;
                }
            }

            GGML_ASSERT(src_pos == total && dst_pos == total);
            // any tensors left unvisited hold no data
            for (size_t i = src_j; i < mbuf_cur.cpy.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf_cur.cpy[i]) == 0);
            }
            for (size_t i = dst_i; i < mbuf.org.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf.org[i]) == 0);
            }

            ggml_backend_tensor_copy_n(chunk_srcs.data(), chunk_dsts.data(), chunk_srcs.size());

            ggml_free(ctx_scratch);
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    void discard() override {
        rinfos.clear();
        buf_size = 0;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

static ggml_backend_t llama_device_copy_backend(ggml_backend_buffer_type_t buft, const std::vector<ggml_backend_t> & backends) {
    if (ggml_backend_buft_is_host(buft)) { return nullptr; }
    for (auto * backend : backends) {
        if (ggml_backend_get_device(backend) == ggml_backend_buft_get_device(buft) &&
                ggml_backend_supports_buft(backend, buft)) { return backend; }
    }
    return nullptr;
}

struct llama_device_copy_route {
    ggml_backend_t       compute_backend = nullptr;
    ggml_backend_t       copy_backend    = nullptr;
    ggml_backend_event_t compute_ready   = nullptr;
};

// save/restore do checkpoint device pelo layout físico de cada GPU.
// Uma peça = trecho contíguo no tensor simples do rank e no buffer de destino daquele rank.
struct llama_ckpt_dst_piece {
    size_t local = 0; // offset absoluto no tensor do modelo (lado do rank)
    size_t bytes = 0;
};

static bool llama_ckpt_split_of(const ggml_tensor * tensor, llama_ckpt_split & out) {
    const auto st = ggml_backend_meta_tensor_split_state(tensor);
    if (st.axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN || st.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
        return false;
    }
    out = llama_ckpt_split{};
    out.axis       = (st.axis == GGML_BACKEND_SPLIT_AXIS_NONE || st.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) ? -1 : (int) st.axis;
    out.n_ranks    = ggml_backend_meta_tensor_n_simple(tensor);
    out.n_segments = out.axis < 0 ? 0 : st.n_segments;
    for (size_t s = 0; s < 16; ++s) { out.nr[s] = st.nr[s]; }
    for (size_t i = 0; i < 16*16; ++i) { out.ne[i] = (size_t) st.ne[i]; }
    return out.n_ranks > 0;
}

// peças físicas de [offset, offset+size) do tensor; false quando o layout não é suportado
static bool llama_ckpt_pieces_of(const ggml_tensor * tensor, size_t offset, size_t size, bool all_ranks,
        std::vector<llama_ckpt_piece> & out) {
    out.clear();
    if (!ggml_backend_buffer_is_meta(tensor->buffer) || !ggml_is_contiguous(tensor)) { return false; }
    llama_ckpt_split split;
    if (!llama_ckpt_split_of(tensor, split)) { return false; }
    int64_t ne[4];
    size_t  nb[4];
    GGML_UNUSED(ne);
    GGML_UNUSED(nb);
    return llama_ckpt_plan(split, ggml_nbytes(tensor), (int) ggml_type_size(tensor->type),
            (int) ggml_blck_size(tensor->type), offset, size, all_ranks, out);
}

// backend do rank que possui o buffer `dst_buft`
static ggml_backend_t llama_ckpt_rank_backend(ggml_backend_buffer_type_t src_buft, size_t rank,
        const std::vector<ggml_backend_t> & backends, ggml_backend_buffer_type_t * dst_buft) {
    *dst_buft = ggml_backend_buft_is_meta(src_buft) ? ggml_backend_meta_buft_simple(src_buft, rank) : src_buft;
    ggml_backend_dev_t src_dev = ggml_backend_buft_get_device(src_buft);
    for (auto * b : backends) {
        if (b && ggml_backend_is_meta(b) && ggml_backend_get_device(b) == src_dev) {
            return ggml_backend_meta_simple_backend(b, rank);
        }
    }
    ggml_backend_dev_t dst_dev = ggml_backend_buft_get_device(*dst_buft);
    for (auto * b : backends) {
        if (b && ggml_backend_get_device(b) == dst_dev) { return b; }
    }
    return nullptr;
}

// one physical checkpoint budget per backend buffer type, shared by target/draft contexts.
static std::mutex llama_ckpt_mutex;
static std::vector<std::weak_ptr<llama_device_checkpoint>> & llama_ckpt_registry() {
    static std::vector<std::weak_ptr<llama_device_checkpoint>> registry;
    return registry;
}

class llama_io_write_checkpoint_device : public llama_io_write_i {
public:
    llama_io_write_checkpoint_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs,
            std::vector<llama_device_segment> * segments = nullptr, std::vector<ggml_backend_t> backends = {},
            llama_device_checkpoint * checkpoint = nullptr,
            std::vector<llama_device_copy_route> copy_routes = {},
            std::function<void()> sync_cb = {},
            std::function<void(ggml_backend_t)> war_cb = {}, bool * write_failed = nullptr)
        : ptr(p), buf_size(len), mbufs(mbufs), segments(segments), backends(std::move(backends)),
          checkpoint(checkpoint), copy_routes(std::move(copy_routes)),
          sync_cb(std::move(sync_cb)), war_cb(std::move(war_cb)), write_failed(write_failed) {
        if (segments) { segments->clear(); }
    }

    ~llama_io_write_checkpoint_device() {
        const int64_t t2022_start_us = ggml_time_us();
        std::lock_guard<std::mutex> lock(llama_ckpt_mutex);
        // layout por dispositivo (peças contíguas por GPU, D2D local) quando suportado
        if (commit_device_layout() != 0) { return; }
        emit_segments_legacy();
        if (checkpoint && checkpoint->layout != 0) { checkpoint->layout = 0; checkpoint->layout_sig = 0; }
        bool layout_matches = !mbufs.empty() && !winfos.empty();
        if (layout_matches) {
            std::map<ggml_backend_buffer_type_t, size_t> counts;
            for (const auto & winfo : winfos) {
                auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);
                auto it = mbufs.find(buft);
                if (it == mbufs.end()) {
                    layout_matches = false;
                    break;
                }
                const size_t idx = counts[buft]++;
                if (idx >= it->second.org.size()) {
                    layout_matches = false;
                    break;
                }
                const auto * org = it->second.org[idx];
                const auto * expected_src = winfo.tensor->view_src ? winfo.tensor->view_src : winfo.tensor;
                const size_t expected_offs = winfo.tensor->view_offs + winfo.offset;
                const auto * expected_data = static_cast<const uint8_t *>(winfo.tensor->data) + winfo.offset;
                if (org->type != winfo.tensor->type ||
                        org->view_src != expected_src ||
                        org->view_offs != expected_offs ||
                        org->buffer != winfo.tensor->buffer ||
                        org->data != expected_data ||
                        ggml_nbytes(org) != winfo.size) {
                    layout_matches = false;
                    break;
                }
            }
            if (layout_matches) {
                if (counts.size() != mbufs.size()) {
                    layout_matches = false;
                } else {
                    for (const auto & [buft, mbuf] : mbufs) {
                        if (!mbuf.buf || counts[buft] != mbuf.org.size()) {
                            layout_matches = false;
                            break;
                        }
                    }
                }
            }
        }

        if (!layout_matches) {
            if (checkpoint) {
                checkpoint->wait_host();
            }

            llama_memory_buffers mbufs_new;

            for (const auto & winfo : winfos) {
                auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

                mbufs_new[buft].n_tensors++;
                mbufs_new[buft].total_size += winfo.size;
            }

            for (auto & [buft, mbuf] : mbufs_new) {
                ggml_init_params params = {
                    /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                    /*.mem_buffer =*/ NULL,
                    /*.no_alloc   =*/ true,
                };

                mbuf.ctx.reset(ggml_init(params));

                mbuf.org.reserve(mbuf.n_tensors);
                mbuf.cpy.reserve(mbuf.n_tensors);
            }

            for (const auto & winfo : winfos) {
                auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

                const int64_t n = winfo.size/ggml_element_size(winfo.tensor);

                auto & mbuf = mbufs_new[buft];

                mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
                mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
            }

            // Refuse the entire replacement before allocation, mutation, or DMA. This writer
            // commits in its noexcept destructor: propagate failure explicitly, never throw here.
            if (segments && write_failed) {
                const char * env = std::getenv("LLAMA_CKPT_POOL_MIB");
                char * end = nullptr;
                const unsigned long long mib = env ? std::strtoull(env, &end, 10) : 704;
                const bool invalid = env && (!*env || *end || *env == '-' || mib > SIZE_MAX/(1024ull*1024));
                const size_t ceiling = invalid ? 0 : size_t(mib)*1024ull*1024;
                for (const auto & entry : mbufs_new) {
                    auto * buft = entry.first;
                    const size_t required = ggml_backend_alloc_ctx_tensors_from_buft_size(entry.second.ctx.get(), buft);
                    const auto cur = mbufs.find(buft);
                    const size_t old = cur != mbufs.end() && cur->second.buf ? ggml_backend_buffer_get_size(cur->second.buf.get()) : 0;
                    const size_t capacity = std::max(old, required);
                    size_t resident = 0;
                    bool refuse = invalid || capacity > ceiling;
                    for (const auto & weak : llama_ckpt_registry()) {
                        if (auto lease = weak.lock()) {
                            if (&lease->buffers == &mbufs) { continue; }
                            const auto found = lease->buffers.find(buft);
                            if (found != lease->buffers.end() && found->second.buf) {
                                const size_t bytes = ggml_backend_buffer_get_size(found->second.buf.get());
                                if (bytes > ceiling || resident > ceiling - bytes) { refuse = true; break; }
                                resident += bytes;
                            }
                        }
                    }
                    if (refuse || resident > ceiling - capacity) {
                        *write_failed = true;
                        LLAMA_LOG_WARN("checkpoint device pool ceiling: refusing allocation on %s\n", ggml_backend_buft_name(buft));
                        return;
                    }
                }
            }
            for (auto it = mbufs.begin(); it != mbufs.end();) {
                if (!mbufs_new.count(it->first)) { it = mbufs.erase(it); }
                else { ++it; }
            }
            for (auto & [buft, mbuf] : mbufs_new) {
                auto & mbuf_cur = mbufs[buft];

                bool need_alloc = false;

                need_alloc = need_alloc || (!mbuf_cur.buf);
                need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
                need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

                if (!need_alloc) {
                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        auto * org0 = mbuf_cur.org[i];
                        auto * org1 = mbuf.org[i];

                        if (!ggml_are_same_shape(org0, org1)) {
                            need_alloc = true;
                            break;
                        }

                        if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                            need_alloc = true;
                            break;
                        }
                    }
                }

                if (need_alloc) {
                    // Allocation size includes backend padding/alignment; equal payload byte
                    // counts can still require different allocations when range shapes change.
                    const size_t required = ggml_backend_alloc_ctx_tensors_from_buft_size(mbuf.ctx.get(), buft);
                    if (!mbuf_cur.buf || ggml_backend_buffer_get_size(mbuf_cur.buf.get()) < required) {
                        mbuf_cur = std::move(mbuf);

                        mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                        LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf_cur.total_size/1024.0/1024.0);
                    } else {
                        // save the old buffer and allocate the new tensors in it
                        auto buf = std::move(mbuf_cur.buf);

                        mbuf_cur = std::move(mbuf);

                        ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                        for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                            ggml_backend_view_init(mbuf_cur.org[i]);
                            ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                        }

                        mbuf_cur.buf = std::move(buf);
                    }
                }
            }
        }

        const int64_t t2022_prep_us = ggml_time_us(); // fim da preparação/alocação

        auto find_route = [&](ggml_backend_t compute_b) -> const llama_device_copy_route * {
            for (const auto & r : copy_routes) {
                if (r.compute_backend == compute_b && r.copy_backend && r.compute_ready) {
                    return &r;
                }
            }
            return nullptr;
        };

        bool compute_synced = false;
        auto ensure_compute_synced = [&]() {
            if (!compute_synced) {
                if (sync_cb) { sync_cb(); }
                compute_synced = true;
            }
        };

        std::set<ggml_backend_t> fenced_compute;
        std::map<ggml_backend_t, ggml_backend_t> async_copies; // copy_backend -> compute_backend

        size_t t2022_bytes = 0;
        for (auto & [buft, mbuf_cur] : mbufs) {
            auto * compute_backend = llama_device_copy_backend(buft, backends);
            const auto * route = (checkpoint && compute_backend) ? find_route(compute_backend) : nullptr;
            const int64_t t2022_route_us = ggml_time_us();
            t2022_bytes += mbuf_cur.total_size;
            if (route) {
                ggml_backend_dev_t dev = ggml_backend_get_device(route->copy_backend);
                for (auto & ev : checkpoint->copy_events) {
                    if (ev.dev == dev && ev.copy_backend != route->copy_backend) {
                        ev.wait_host();
                    }
                }
                if (fenced_compute.insert(compute_backend).second) {
                    ggml_backend_event_record(route->compute_ready, compute_backend);
                    ggml_backend_event_wait(route->copy_backend, route->compute_ready);
                }
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    ggml_backend_tensor_copy_async(route->copy_backend, route->copy_backend, mbuf_cur.org[i], mbuf_cur.cpy[i]);
                }
                async_copies[route->copy_backend] = compute_backend;
                LLAMA_LOG_DEBUG("ckpt_device: save buft=%s ranges=%zu bytes=%zu async=2\n",
                        ggml_backend_buft_name(buft), mbuf_cur.cpy.size(), mbuf_cur.total_size);
                if (t2022_trace()) {
                    fprintf(stderr, "[t2022-route] buft=%s ranges=%zu bytes=%zu path=stream copy_backend=%s ms=%.3f\n",
                            ggml_backend_buft_name(buft), mbuf_cur.cpy.size(), mbuf_cur.total_size,
                            ggml_backend_name(route->copy_backend), (ggml_time_us() - t2022_route_us)/1000.0);
                }
            } else {
                ensure_compute_synced();
                if (checkpoint) {
                    checkpoint->wait_host();
                }
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    if (compute_backend) { ggml_backend_tensor_copy_async(compute_backend, compute_backend, mbuf_cur.org[i], mbuf_cur.cpy[i]); }
                    else { ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]); }
                }
                if (compute_backend) { ggml_backend_synchronize(compute_backend); }
                LLAMA_LOG_DEBUG("ckpt_device: save buft=%s ranges=%zu bytes=%zu async=%d\n",
                        ggml_backend_buft_name(buft), mbuf_cur.cpy.size(), mbuf_cur.total_size, compute_backend != nullptr);
                if (t2022_trace()) {
                    fprintf(stderr, "[t2022-route] buft=%s ranges=%zu bytes=%zu path=%s copy_backend=%s ms=%.3f\n",
                            ggml_backend_buft_name(buft), mbuf_cur.cpy.size(), mbuf_cur.total_size,
                            compute_backend ? "meta-async-sync" : "plain",
                            compute_backend ? ggml_backend_name(compute_backend) : "none",
                            (ggml_time_us() - t2022_route_us)/1000.0);
                }
            }
        }

        for (const auto & [copy_backend, compute_backend] : async_copies) {
            ggml_backend_dev_t dev = ggml_backend_get_device(copy_backend);
            llama_device_copy_event * slot = nullptr;
            for (auto & ev : checkpoint->copy_events) {
                if (ev.dev == dev) {
                    slot = &ev;
                    break;
                }
            }
            if (!slot) {
                checkpoint->copy_events.emplace_back();
                slot = &checkpoint->copy_events.back();
                slot->dev = dev;
                slot->event = ggml_backend_event_new(dev);
            }
            slot->copy_backend = copy_backend;
            if (slot->event) {
                ggml_backend_event_record(slot->event, copy_backend);
                slot->pending = true;
                if (war_cb) {
                    war_cb(compute_backend);
                } else {
                    slot->wait_host();
                }
            } else {
                ggml_backend_synchronize(copy_backend);
                slot->pending = false;
            }
        }
        if (t2022_trace()) {
            fprintf(stderr, "[t2022-route] save buft_total=%zu bytes=%zu prep_ms=%.3f total_ms=%.3f\n",
                    mbufs.size(), t2022_bytes, (t2022_prep_us - t2022_start_us)/1000.0,
                    (ggml_time_us() - t2022_start_us)/1000.0);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        if (segments) { segments->push_back({nullptr, size_written, size}); }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // Each tensor range is a chunk in a byte stream, not a restore-time range boundary.
        // o segmento entra aqui como *marcador* para preservar a ordem do byte-stream (os
        // metadados são escritos entre os tensores); o destino é preenchido no destrutor, quando o
        // plano físico (legado ou por dispositivo) já está decidido.
        if (segments) {
            seg_pos.push_back(segments->size());
            segments->push_back({nullptr, 0, size});
        }
        winfos.push_back({tensor, ptr, size, offset});
    }

    // preenche os marcadores do caminho legado (byte-stream espelhado por chunk).
    void emit_segments_legacy() {
        if (!segments) { return; }
        for (size_t i = 0; i < winfos.size(); ++i) {
            auto * buft = ggml_backend_buffer_get_type(winfos[i].tensor->buffer);
            auto & seg = (*segments)[seg_pos[i]];
            seg.buft   = buft;
            seg.offset = tensor_counts[buft]++;
        }
    }

    // troca cada marcador pelas peças físicas daquele trecho, na mesma posição do stream.
    void emit_segments_device(const std::vector<std::vector<llama_ckpt_piece>> & plan) {
        if (!segments) { return; }
        std::map<ggml_backend_buffer_type_t, size_t> idx;
        std::vector<llama_device_segment> out;
        size_t next = 0;
        for (size_t s = 0; s < segments->size(); ++s) {
            if (next < seg_pos.size() && seg_pos[next] == s) {
                const auto & winfo = winfos[next];
                auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);
                for (const auto & p : plan[next]) {
                    ggml_backend_buffer_type_t db = nullptr;
                    (void) llama_ckpt_rank_backend(buft, p.rank, backends, &db);
                    out.push_back({db, idx[db]++, p.bytes});
                }
                ++next;
            } else {
                out.push_back((*segments)[s]);
            }
        }
        GGML_ASSERT(next == seg_pos.size());
        *segments = std::move(out);
    }

    // (re)constrói o armazenamento por dispositivo a partir das peças planejadas.
    // 1 = buffers prontos; 0 = recusado pelo teto do pool (write_failed marcado); -1 = não suportado.
    int rebuild_device_layout(const std::map<ggml_backend_buffer_type_t,
            std::vector<std::pair<size_t, llama_ckpt_piece>>> & groups) {
        llama_memory_buffers mbufs_new;
        for (const auto & [buft, items] : groups) {
            auto & mbuf = mbufs_new[buft];
            mbuf.n_tensors = (int) items.size();
            for (const auto & it : items) { mbuf.total_size += it.second.bytes; }
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            mbuf.ctx.reset(ggml_init(params));
            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }
        for (const auto & [buft, items] : groups) {
            auto & mbuf = mbufs_new[buft];
            for (const auto & it : items) {
                const auto & w = winfos[it.first];
                const auto & p = it.second;
                auto * simple = ggml_backend_meta_buffer_simple_tensor(w.tensor, p.rank);
                if (!simple) { return -1; }
                const size_t el = ggml_element_size(w.tensor);
                if (el == 0 || p.bytes % el != 0) { return -1; }
                const int64_t n = (int64_t) (p.bytes / el);
                auto * org = ggml_view_1d(mbuf.ctx.get(), simple, n, p.local);
                ggml_backend_view_init(org);
                auto * cpy = ggml_new_tensor_1d(mbuf.ctx.get(), w.tensor->type, n);
                if (!org || !cpy) { return -1; }
                mbuf.org.push_back(org);
                mbuf.cpy.push_back(cpy);
            }
        }
        // Teto do pool: mesma regra do caminho legado, agora por dispositivo (a GPU é a unidade
        // que realmente gasta VRAM; cada rank tem o seu orçamento).
        if (write_failed) {
            const char * env = std::getenv("LLAMA_CKPT_POOL_MIB");
            char * end = nullptr;
            const unsigned long long mib = env ? std::strtoull(env, &end, 10) : 704;
            const bool invalid = env && (!*env || *end || *env == '-' || mib > SIZE_MAX/(1024ull*1024));
            const size_t ceiling = invalid ? 0 : size_t(mib)*1024ull*1024;
            for (const auto & entry : mbufs_new) {
                auto * buft = entry.first;
                const size_t required = ggml_backend_alloc_ctx_tensors_from_buft_size(entry.second.ctx.get(), buft);
                const auto cur = mbufs.find(buft);
                const size_t old = cur != mbufs.end() && cur->second.buf ? ggml_backend_buffer_get_size(cur->second.buf.get()) : 0;
                const size_t capacity = std::max(old, required);
                size_t resident = 0;
                bool refuse = invalid || capacity > ceiling;
                for (const auto & weak : llama_ckpt_registry()) {
                    if (auto lease = weak.lock()) {
                        if (&lease->buffers == &mbufs) { continue; }
                        const auto found = lease->buffers.find(buft);
                        if (found != lease->buffers.end() && found->second.buf) {
                            const size_t bytes = ggml_backend_buffer_get_size(found->second.buf.get());
                            if (bytes > ceiling || resident > ceiling - bytes) { refuse = true; break; }
                            resident += bytes;
                        }
                    }
                }
                if (refuse || resident > ceiling - capacity) {
                    *write_failed = true;
                    LLAMA_LOG_WARN("checkpoint device pool ceiling: refusing allocation on %s\n", ggml_backend_buft_name(buft));
                    return 0;
                }
            }
        }
        for (auto & [buft, mbuf] : mbufs_new) {
            mbuf.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf.ctx.get(), buft));
            if (!mbuf.buf) { return -1; }
            LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB (checkpoint device layout)\n",
                    __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
        }
        mbufs = std::move(mbufs_new);
        return 1;
    }

    // save pelo layout físico de cada GPU (peças contíguas, D2D local, sem espelhar o
    // byte-stream do Meta — o espelho fazia metade das cópias ser peer, ~3 GB/s na MI50).
    // 0 = layout não suportado (o chamador segue no caminho legado); 1 = tratado.
    int commit_device_layout() {
        if (!checkpoint || !segments || winfos.empty()) { return 0; }

        std::vector<std::vector<llama_ckpt_piece>> plan(winfos.size());
        std::map<ggml_backend_buffer_type_t, ggml_backend_t> rank_be; // buffer destino -> backend do rank
        std::map<ggml_backend_buffer_type_t, std::vector<std::pair<size_t, llama_ckpt_piece>>> groups;
        uint64_t sig = 1469598103934665603ull;
        auto mix = [&sig](uint64_t v) { sig = (sig ^ v) * 1099511628211ull; };

        for (size_t i = 0; i < winfos.size(); ++i) {
            const auto & w = winfos[i];
            auto * buft = ggml_backend_buffer_get_type(w.tensor->buffer);
            if (!llama_ckpt_pieces_of(w.tensor, w.offset, w.size, /*all_ranks=*/true, plan[i]) || plan[i].empty()) {
                LLAMA_LOG_DEBUG("%s: chunk %zu (%s) sem plano físico suportado; usando caminho legado\n",
                        __func__, i, w.tensor->name);
                return 0;
            }
            mix((uint64_t) (uintptr_t) w.tensor);
            mix((uint64_t) w.tensor->type);
            for (const auto & p : plan[i]) {
                ggml_backend_buffer_type_t db = nullptr;
                auto * be = llama_ckpt_rank_backend(buft, p.rank, backends, &db);
                if (!db || !be) { return 0; }
                rank_be[db] = be;
                groups[db].push_back({i, p});
                mix((uint64_t) (uintptr_t) db);
                mix(p.local);
                mix(p.bytes);
            }
        }

        if (checkpoint->layout != 2 || checkpoint->layout_sig != sig) {
            const int rc = rebuild_device_layout(groups);
            if (rc < 0) { return 0; }
            if (rc == 0) { emit_segments_device(plan); return 1; } // recusado pelo teto: write_failed marcado
            checkpoint->layout     = 2;
            checkpoint->layout_sig = sig;
        }

        // segmentos no formato serializado, nas posições originais do byte-stream
        emit_segments_device(plan);

        // cópias D2D locais, assíncronas no stream de cada rank (os ranks correm em paralelo)
        if (sync_cb) { sync_cb(); }
        checkpoint->wait_host();
        std::set<ggml_backend_t> used;
        for (const auto & [buft, items] : groups) {
            auto it_be = rank_be.find(buft);
            if (it_be == rank_be.end()) { return 0; }
            auto * be = it_be->second;
            const auto & mbuf = mbufs.at(buft);
            if (mbuf.org.size() != items.size()) { return 0; }
            const int64_t t0 = ggml_time_us();
            for (size_t k = 0; k < items.size(); ++k) {
                ggml_backend_tensor_copy_async(be, be, mbuf.org[k], mbuf.cpy[k]);
            }
            used.insert(be);
            if (t2022_trace()) {
                fprintf(stderr, "[t2022-route] buft=%s pieces=%zu bytes=%zu path=device-async backend=%s ms=%.3f\n",
                        ggml_backend_buft_name(buft), items.size(), mbuf.total_size,
                        ggml_backend_name(be), (ggml_time_us() - t0)/1000.0);
            }
        }
        const int64_t t_sync0 = ggml_time_us();
        for (auto * be : used) { ggml_backend_synchronize(be); }
        const int64_t t_sync = ggml_time_us();
        checkpoint->copy_events.clear();
        if (t2022_trace()) {
            size_t pieces = 0, bytes = 0;
            for (const auto & [buft, items] : groups) {
                pieces += items.size();
                bytes  += mbufs.at(buft).total_size;
            }
            fprintf(stderr, "[t2022-route] save device-layout buffers=%zu pieces=%zu bytes=%zu sync_ms=%.3f total_ms=%.3f\n",
                    mbufs.size(), pieces, bytes, (t_sync - t_sync0)/1000.0, (ggml_time_us() - t_sync0)/1000.0);
        }
        return 1;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
    std::vector<size_t>    seg_pos; // posição do marcador de cada trecho no byte-stream

    llama_memory_buffers & mbufs;
    std::vector<llama_device_segment> * segments;
    std::map<ggml_backend_buffer_type_t, size_t> tensor_counts;
    std::vector<ggml_backend_t> backends;
    llama_device_checkpoint * checkpoint = nullptr;
    std::vector<llama_device_copy_route> copy_routes;
    std::function<void()> sync_cb;
    std::function<void(ggml_backend_t)> war_cb;
    bool * write_failed;
};

class llama_io_read_checkpoint_device : public llama_io_read_i {
public:
    llama_io_read_checkpoint_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs,
            std::vector<ggml_backend_t> backends = {}, int layout = 0, bool * restore_failed = nullptr)
        : ptr(p), buf_size(len), mbufs(mbufs), backends(std::move(backends)), layout(layout),
          restore_failed(restore_failed) {
    }

    // as cópias diferidas rodam aqui, no destrutor — nada nesta função pode lançar (um
    // throw num destrutor noexcept vira std::terminate → SIGABRT do processo inteiro, que é o
    // aborto da revisão anterior §3). Tudo é validado ANTES da primeira cópia; recusa devolve `false` e o
    // chamador publica "sem checkpoint" (o servidor reprocessa o prompt em vez de morrer).
    ~llama_io_read_checkpoint_device() {
        t2012_scope timing("restore-copies-total");
        if (discarded) return;
        const size_t n_saved = n_saved_pieces();
        bool ok = false;
        if (rinfos.empty() && !n_saved) {
            ok = buf_size == 0;
        } else if (layout == 2) {
            ok = restore_device_layout(n_saved);
        } else {
            ok = restore_compact_layout(n_saved);
        }
        if (!ok) {
            LLAMA_LOG_ERROR("%s: device checkpoint restore refused (layout=%d saved_pieces=%zu destination=%zu); "
                    "the checkpoint is unusable, the caller must restart from the prompt\n",
                    __func__, layout, n_saved, rinfos.size());
            if (restore_failed) { *restore_failed = true; }
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_read;
    }

    // Uma cópia D2D diferida (origem/destino já validados).
    struct copy_op {
        ggml_tensor * from    = nullptr;
        ggml_tensor * to      = nullptr;
        ggml_backend_t backend = nullptr; // nullptr = ggml_backend_tensor_copy síncrono
    };

    size_t n_saved_pieces() const {
        size_t n_saved = 0;
        for (const auto & entry : mbufs) { n_saved += entry.second.cpy.size(); }
        return n_saved;
    }

    // bytes de cada peça salva, por chave de armazenamento, na ordem em que foi gravada
    std::map<uintptr_t, std::vector<size_t>> saved_pieces() const {
        std::map<uintptr_t, std::vector<size_t>> saved;
        for (const auto & entry : mbufs) {
            auto & sizes = saved[(uintptr_t) entry.first];
            for (const auto * cpy : entry.second.cpy) { sizes.push_back(ggml_nbytes(cpy)); }
        }
        return saved;
    }

    // Layout legado: as peças salvas são o byte-stream lógico espelhado, então o destino consome os
    // bytes em ordem por tipo de buffer (a correspondência não é 1:1 entre views). O casamento de
    // totais é conferido com o helper puro ANTES das cópias: chave ausente é recusa, nunca exceção.
    bool restore_compact_layout(size_t n_saved) {
        std::map<uintptr_t, size_t> wanted;
        for (const auto & rinfo : rinfos) {
            wanted[(uintptr_t) ggml_backend_buffer_get_type(rinfo.tensor->buffer)] += rinfo.size;
        }
        if (!llama_ckpt_restore_totals_match(saved_pieces(), wanted)) { return false; }

        ggml_init_params params = {2*(n_saved + rinfos.size())*ggml_tensor_overhead(), nullptr, true};
        ggml_context_ptr views(ggml_init(params));
        if (!views) { return false; }

        std::vector<copy_op> ops;
        std::map<ggml_backend_buffer_type_t, std::pair<size_t, size_t>> cursors;
        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);
            const auto found = mbufs.find(buft);
            if (found == mbufs.end()) { return false; }
            const auto & mbuf = found->second;
            auto & cursor = cursors[buft];
            size_t done = 0;
            while (done < rinfo.size) {
                if (cursor.first >= mbuf.cpy.size()) { return false; }
                auto * src = mbuf.cpy[cursor.first];
                if (src->type != rinfo.tensor->type) { return false; }
                const size_t src_bytes = ggml_nbytes(src);
                if (cursor.second > src_bytes) { return false; }
                const size_t available = src_bytes - cursor.second;
                const size_t bytes = std::min(available, rinfo.size - done);
                const size_t el = ggml_type_size(src->type);
                if (bytes == 0 || el == 0 || bytes % el != 0) { return false; }
                const int64_t n = (int64_t) ((bytes/el)*ggml_blck_size(src->type));
                auto * from = ggml_view_1d(views.get(), src, n, cursor.second);
                auto * to = ggml_view_1d(views.get(), rinfo.tensor, n, rinfo.offset + done);
                if (!from || !to) { return false; }
                ggml_backend_view_init(from);
                ggml_backend_view_init(to);
                ops.push_back({from, to, llama_device_copy_backend(buft, backends)});
                done += bytes;
                cursor.second += bytes;
                if (cursor.second == src_bytes) { ++cursor.first; cursor.second = 0; }
            }
        }
        for (const auto & entry : mbufs) {
            const auto it = cursors.find(entry.first);
            const size_t first  = it == cursors.end() ? 0 : it->second.first;
            const size_t second = it == cursors.end() ? 0 : it->second.second;
            if (first != entry.second.cpy.size() || second != 0) { return false; }
        }
        if (buf_size != 0) { return false; }
        return commit(ops, n_saved);
    }

    // Layout por dispositivo: o save gravou peças contíguas no buffer de cada GPU e o
    // destino tem de produzir exatamente a mesma sequência de peças por buffer. O casamento é
    // feito pelo helper puro `llama_ckpt_restore_map` (testado em CPU), não por `map::at`.
    bool restore_device_layout(size_t n_saved) {
        struct destination_piece {
            size_t                     rinfo    = 0;
            llama_ckpt_piece           piece    = {};
            ggml_backend_t             backend  = nullptr;
            ggml_backend_buffer_type_t buft     = nullptr;
        };
        std::vector<std::vector<llama_ckpt_piece>> plans(rinfos.size());
        std::vector<destination_piece> dest_pieces;
        std::vector<std::pair<uintptr_t, size_t>> dest;
        for (size_t i = 0; i < rinfos.size(); ++i) {
            auto * buft = ggml_backend_buffer_get_type(rinfos[i].tensor->buffer);
            if (!llama_ckpt_pieces_of(rinfos[i].tensor, rinfos[i].offset, rinfos[i].size,
                        /*all_ranks=*/true, plans[i]) || plans[i].empty()) {
                return false;
            }
            for (const auto & p : plans[i]) {
                ggml_backend_buffer_type_t db = nullptr;
                auto * be = llama_ckpt_rank_backend(buft, p.rank, backends, &db);
                if (!be || !db) { return false; }
                dest_pieces.push_back({i, p, be, db});
                dest.push_back({(uintptr_t) db, p.bytes});
            }
        }
        std::vector<size_t> saved_index;
        if (!llama_ckpt_restore_map(saved_pieces(), dest, saved_index)) { return false; }

        ggml_init_params params = {2*(n_saved + rinfos.size())*ggml_tensor_overhead(), nullptr, true};
        ggml_context_ptr views(ggml_init(params));
        if (!views) { return false; }

        std::vector<copy_op> ops;
        for (size_t k = 0; k < dest_pieces.size(); ++k) {
            const auto & dp = dest_pieces[k];
            const auto it = mbufs.find(dp.buft);
            if (it == mbufs.end()) { return false; }
            if (saved_index[k] >= it->second.cpy.size()) { return false; }
            auto * from = it->second.cpy[saved_index[k]];
            if (from->type != rinfos[dp.rinfo].tensor->type) { return false; }
            auto * simple = ggml_backend_meta_buffer_simple_tensor(rinfos[dp.rinfo].tensor, dp.piece.rank);
            if (!simple) { return false; }
            auto * to = ggml_view_1d(views.get(), simple, from->ne[0], dp.piece.local);
            if (!to) { return false; }
            ggml_backend_view_init(to);
            ops.push_back({from, to, dp.backend});
        }
        if (buf_size != 0) { return false; }
        const int64_t t0 = ggml_time_us();
        if (!commit(ops, n_saved)) { return false; }
        if (t2022_trace()) {
            fprintf(stderr, "[t2022-route] restore device-layout chunks=%zu pieces=%zu ms=%.3f\n",
                    rinfos.size(), ops.size(), (ggml_time_us() - t0)/1000.0);
        }
        return true;
    }

    // Executa as cópias já validadas (assíncronas no stream de cada dispositivo) e sincroniza.
    bool commit(const std::vector<copy_op> & ops, size_t n_saved) {
        std::set<ggml_backend_t> pending;
        const int64_t sync_start = ggml_time_us();
        try {
            for (const auto & op : ops) {
                if (op.backend) {
                    ggml_backend_tensor_copy_async(op.backend, op.backend, op.from, op.to);
                    pending.insert(op.backend);
                } else {
                    ggml_backend_tensor_copy(op.from, op.to);
                }
            }
            for (auto * backend : pending) { ggml_backend_synchronize(backend); }
        } catch (const std::exception & err) {
            LLAMA_LOG_ERROR("%s: device checkpoint copies failed: %s\n", __func__, err.what());
            return false;
        }
        if (getenv("T2012_TRACE")) fprintf(stderr,"[t2012-time] restore-copy-sync us=%lld ranges=%zu saved=%zu layout=%d\n",
                (long long)(ggml_time_us()-sync_start), rinfos.size(), n_saved, layout);
        LLAMA_LOG_DEBUG("ckpt_device: restore saved_pieces=%zu destination_ranges=%zu async_backends=%zu layout=%d\n",
                n_saved, rinfos.size(), pending.size(), layout);
        return true;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
    std::vector<ggml_backend_t> backends;
    int layout = 0;
    bool * restore_failed = nullptr; // recusa da restauração (o chamador devolve 0 bytes)
    bool discarded = false;
public:
    void discard() override { rinfos.clear(); discarded = true; }
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io.discard();
        return 0;
    }
}

llama_state_seq_device * llama_context::state_seq_device_acquire(llama_seq_id seq_id, size_t limit) {
    if (!limit || seq_id < 0) { return nullptr; }
    auto & pool = device_ckpt_pool[seq_id];
    for (auto & storage : pool) {
        if (storage.use_count() == 1) { return new llama_state_seq_device{storage}; }
    }
    if (pool.size() >= limit) { return nullptr; }
    pool.push_back(std::make_shared<llama_device_checkpoint>());
    std::lock_guard<std::mutex> lock(llama_ckpt_mutex);
    auto & registry = llama_ckpt_registry();
    registry.erase(std::remove_if(registry.begin(), registry.end(), [](const auto & w) { return w.expired(); }), registry.end());
    registry.push_back(pool.back());
    return new llama_state_seq_device{pool.back()};
}

void llama_context::wait_pending_device_ckpt_war_gpu() {
    if (pending_device_ckpt_war.empty()) {
        return;
    }
    for (const auto & war : pending_device_ckpt_war) {
        auto ckpt = war.ckpt.lock();
        if (!ckpt) {
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(war.compute_backend);
        for (const auto & ev : ckpt->copy_events) {
            if (ev.pending && ev.event && ev.dev == dev) {
                ggml_backend_event_wait(war.compute_backend, ev.event);
            }
        }
    }
    pending_device_ckpt_war.clear();
}

void llama_context::wait_pending_device_ckpt_war_host() {
    if (pending_device_ckpt_war.empty()) {
        return;
    }
    for (const auto & war : pending_device_ckpt_war) {
        if (auto ckpt = war.ckpt.lock()) {
            ckpt->wait_host();
        }
    }
    pending_device_ckpt_war.clear();
}

ggml_backend_t llama_context::ensure_device_ckpt_copy_stream(ggml_backend_t compute_backend, ggml_backend_event_t * ready_event_out) {
    if (ready_event_out) { *ready_event_out = nullptr; }
    if (!compute_backend) { return nullptr; }
    static const bool async_enabled = []() {
        const char * env = std::getenv("LLAMA_CTX_CKPT_ASYNC_STREAM");
        return !env || std::strcmp(env, "0") != 0;
    }();
    if (!async_enabled) {
        return nullptr;
    }
    auto it = device_ckpt_streams.find(compute_backend);
    if (it != device_ckpt_streams.end()) {
        if (ready_event_out) { *ready_event_out = it->second.compute_ready; }
        return it->second.copy_backend;
    }
    device_ckpt_stream stream_entry{};
    ggml_backend_dev_t dev = ggml_backend_get_device(compute_backend);
    if (dev) {
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(dev, &props);
        if (props.caps.events && props.caps.async && props.type == GGML_BACKEND_DEVICE_TYPE_GPU) {
            ggml_backend_t copy_b = ggml_backend_dev_init(dev, nullptr);
            ggml_backend_event_t ev = copy_b ? ggml_backend_event_new(dev) : nullptr;
            if (copy_b && ev) {
                stream_entry.copy_backend  = copy_b;
                stream_entry.compute_ready = ev;
            } else {
                if (ev) { ggml_backend_event_free(ev); }
                if (copy_b) { ggml_backend_free(copy_b); }
            }
        }
    }
    device_ckpt_streams[compute_backend] = stream_entry;
    if (ready_event_out) { *ready_event_out = stream_entry.compute_ready; }
    return stream_entry.copy_backend;
}

llama_state_seq_device * llama_state_seq_device_acquire(llama_context * ctx, llama_seq_id seq_id, size_t limit) {
    return ctx ? ctx->state_seq_device_acquire(seq_id, limit) : nullptr;
}
void llama_state_seq_device_free(llama_state_seq_device * device) { delete device; }
size_t llama_state_seq_device_get(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id,
        llama_state_seq_flags flags, llama_state_seq_device * device) {
    if (!ctx || !device) { return 0; }
    return ctx->state_seq_get_data(seq_id, dst, size, flags | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE, device->storage.get(), true);
}
size_t llama_state_seq_device_set(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id,
        llama_state_seq_flags flags, const llama_state_seq_device * device) {
    if (!ctx || !device) { return 0; }
    return ctx->state_seq_set_data(seq_id, src, size, flags | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE, device->storage.get());
}
size_t llama_state_seq_device_to_host(const llama_state_seq_device * device, const uint8_t * src, size_t size,
        uint8_t * dst, size_t capacity) {
    if (!device) { return 0; }
    size_t total = 0;
    for (const auto & segment : device->storage->segments) { total += segment.size; }
    if (!dst) { return total; }
    if (capacity < total) { return 0; }
    t2012_scope timing("materialize-to-host");
    device->storage->wait_host();
    // Reuse the upstream batched transfer API, rather than carrying the old
    // GGML_EXP_CKPT_FAST staging backend/cache alongside the native implementation.
    std::vector<const ggml_tensor *> tensors;
    std::vector<size_t> offsets, sizes, data_offsets;
    size_t output_offset = 0;
    for (const auto & segment : device->storage->segments) {
        if (!segment.buft) {
            GGML_ASSERT(segment.offset <= size && segment.size <= size - segment.offset);
            memcpy(dst + output_offset, src + segment.offset, segment.size);
        } else {
            const auto & mbuf = device->storage->buffers.at(segment.buft);
            tensors.push_back(mbuf.cpy.at(segment.offset));
            offsets.push_back(0);
            sizes.push_back(segment.size);
            data_offsets.push_back(output_offset);
        }
        output_offset += segment.size;
    }
    if (!tensors.empty()) {
        ggml_backend_tensor_get_n(tensors.data(), dst, offsets.data(), sizes.data(), data_offsets.data(), tensors.size());
    }
    return total;
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    t2012_scope timing("save-size-metadata");
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags, llama_device_checkpoint * device, bool async_device) {
    t2012_scope timing(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE ? "save-device" : "save-host");
    if (getenv("T2012_TRACE")) fprintf(stderr,"[t2012-save] size=%zu flags=%u\n",size,(unsigned)flags);
    bool write_failed = false;
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        if (!device) {
            io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
        } else {
        std::vector<ggml_backend_t> copy_backends;
        std::vector<llama_device_copy_route> copy_routes;
        for (const auto & backend : backends) {
            ggml_backend_t compute_b = backend.get();
            copy_backends.push_back(compute_b);
            if (async_device && device) {
                ggml_backend_event_t ready_ev = nullptr;
                ggml_backend_t copy_b = ensure_device_ckpt_copy_stream(compute_b, &ready_ev);
                if (copy_b && ready_ev) {
                    copy_routes.push_back({compute_b, copy_b, ready_ev});
                }
            }
        }
        std::function<void()> sync_cb = [this]() {
            this->synchronize();
        };
        std::function<void(ggml_backend_t)> war_cb = [this, device](ggml_backend_t compute_b) {
            if (!device) { return; }
            auto weak = device->weak_from_this();
            if (!weak.expired()) {
                auto it = this->device_ckpt_streams.find(compute_b);
                if (it != this->device_ckpt_streams.end()) {
                    it->second.has_pending = true;
                }
                this->pending_device_ckpt_war.push_back({compute_b, std::move(weak)});
            } else {
                device->wait_host();
            }
        };
        io = std::make_unique<llama_io_write_checkpoint_device>(dst, size,
                device ? device->buffers : mem_storage[seq_id], device ? &device->segments : nullptr,
                std::move(copy_backends),
                async_device ? device : nullptr,
                std::move(copy_routes),
                std::move(sync_cb),
                std::move(war_cb), &write_failed);
        }
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        const size_t written = state_seq_write_data(*io, seq_id, flags);
        io.reset(); // commit deferred checkpoint copies while the error channel is still alive
        return write_failed ? 0 : written;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags, llama_device_checkpoint * device) {
    t2012_scope timing("restore-total");
    std::unique_ptr<llama_io_read_i> io;
    // a restauração do checkpoint device é adiada para o destrutor do leitor; se ela for
    // recusada (layout salvo incompatível com o destino, storage que não casa, etc.) o leitor marca
    // esta flag e este método devolve 0 — o chamador trata como "sem checkpoint" e reprocessa o
    // prompt, em vez de abortar o processo.
    bool restore_failed = false;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(device || mem_storage.find(seq_id_read) != mem_storage.end());

        wait_pending_device_ckpt_war_gpu();
        if (device) {
            for (auto & ev : device->copy_events) {
                if (ev.pending && ev.event) {
                    bool waited_gpu = false;
                    for (const auto & backend : backends) {
                        if (ggml_backend_get_device(backend.get()) == ev.dev) {
                            ggml_backend_event_wait(backend.get(), ev.event);
                            waited_gpu = true;
                        }
                    }
                    if (!waited_gpu) {
                        ev.wait_host();
                    }
                }
            }
        }

        if (!device) {
            io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
        } else {
        std::vector<ggml_backend_t> copy_backends;
        for (const auto & backend : backends) { copy_backends.push_back(backend.get()); }
        // o layout salvo (0 = legado, 2 = peças por dispositivo) tem de chegar ao leitor;
        // sem isto a restauração caía sempre no caminho legado e abortava em `mbufs.at(buft)`.
        io = std::make_unique<llama_io_read_checkpoint_device>(src, size, device->buffers,
                std::move(copy_backends), device->layout, &restore_failed);
        }
    } else {
        wait_pending_device_ckpt_war_host();
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        const size_t n = state_seq_read_data(*io, seq_id, flags);
        io.reset(); // commit das cópias diferidas do checkpoint device antes de reportar o resultado
        return restore_failed ? 0 : n;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        if (io) { io->discard(); } // upstream failure cleanup must cancel deferred tensor writes
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out, llama_state_seq_limits * limits) {
    if(limits) return state_seq_load_file_bounded(seq_id,filepath,tokens_out,n_token_capacity,n_token_count_out,limits);
    auto clear_destination = [&]() {
        // A bounded rejection before any native replacement preserves the slot.
        if (memory && (!limits || limits->mutation_started)) {
            memory->seq_rm(seq_id, -1, -1);
        }
    };
    try {
    llama_file file(filepath, "rb");
    size_t loaded_token_count = 0;

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            clear_destination();
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (tokens_out == nullptr) {
            const size_t n_token_max = (file.size() - file.tell()) / sizeof(llama_token);
            if (n_token_count > n_token_max) {
                LLAMA_LOG_ERROR("%s: token count in sequence state file exceeds the file size! %u > %zu\n", __func__, n_token_count, n_token_max);
                return 0;
            }

            *n_token_count_out = n_token_count;
            return file.tell();
        }

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            clear_destination();
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        loaded_token_count = n_token_count;
    }

    // restore the context state
    {
        const size_t offset_before_state = file.tell();
        if (offset_before_state > file.size()) {
            clear_destination();
            return 0;
        }
        const size_t state_size = file.size() - offset_before_state;
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            clear_destination();
            return 0;
        }
        if (nread > state_size ||
                nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * loaded_token_count != file.tell()) {
            LLAMA_LOG_ERROR("%s: invalid or truncated sequence state file\n", __func__);
            clear_destination();
            return 0;
        }
    }

    *n_token_count_out = loaded_token_count;
    return file.tell();
    } catch (...) {
        clear_destination();
        throw;
    }
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    const std::string tmp_path = std::string(filepath) + ".tmp";
    struct temp_file_cleanup {
        const std::string & path;
        ~temp_file_cleanup() {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    } cleanup{tmp_path};
    llama_file file(tmp_path.c_str(), "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t logical_size = file.tell();
    if (logical_size != sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes()) {
        throw std::runtime_error("inconsistent sequence state byte count");
    }
    const size_t closed_size = file.finalize();
    if (closed_size != logical_size) {
        throw std::runtime_error(format("sequence state size mismatch after close: %zu != %zu", closed_size, logical_size));
    }
    std::error_code ec;
    std::filesystem::rename(tmp_path, filepath, ec);
    if (ec) {
        throw std::runtime_error(format("failed to publish sequence state file: %s", ec.message().c_str()));
    }
    const uintmax_t published_size = std::filesystem::file_size(filepath, ec);
    if (ec || published_size != closed_size) {
        throw std::runtime_error("published sequence state file size mismatch");
    }
    return (size_t) published_size;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    t2012_scope timing("restore-metadata-enqueue");
    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    data.n_reallocs_same = n_reallocs_same;
    data.n_reallocs_diff = n_reallocs_diff;

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;

    // note: the reallocation counters are not reset - they track the lifetime totals for this context
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (moe_cache) {
        for (const auto & [buft, size] : moe_cache->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.no_alloc()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    if (cparams.flash_attn) {
        LLAMA_LOG_INFO("%s: disabling flash attention, FLASH_ATTN_EXT has no backward pass\n", __func__);
        cparams.flash_attn = false;
    }

    // gradients cannot flow through the KV cache, so the attention reads the K and V of the current ubatch directly
    if (n_ubatch == cparams.n_ctx) {
        cparams.training = true;
    } else {
        LLAMA_LOG_WARN("%s: n_ubatch (%u) != n_ctx (%u), the K and V projections will not receive gradients\n", __func__, n_ubatch, cparams.n_ctx);
    }

    // the training graph is different, need to reserve again
    sched_need_reserve = true;
    sched_reserve();

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        // TODO: use llama_batch_ext here
        {
            llama_batch_compat compat(this, batch);
            if (!balloc->init(*compat.batch_ext, model.vocab, true)) {
                LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
                return;
            }
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = get_gf_res_prev();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            // the optimizer graph is allocated outside sched, so the next decode must rebuild
            gf_cache_reset_results();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            copy_experts.reset();
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_outputs_max_per_seq       =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.moe_cache_size              =*/ 0,
        /*.moe_cache_cpu_share         =*/ 0.0f,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (model->get_split_state_ud.n_devices == 1) {
            LLAMA_LOG_WARN("%s: SPLIT_MODE_TENSOR being used for a single device is not recommended\n", __func__);
        }
    }

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4 || model->arch == LLM_ARCH_DEEPSEEK41) && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    const bool fp8_cache = params.type_k == GGML_TYPE_F8_E4M3 || params.type_v == GGML_TYPE_F8_E4M3;
    // Direct cache consumers do not use the shared attention scaling helpers.
    if (fp8_cache && (model->hparams.is_mla() ||
            model->arch == LLM_ARCH_DEEPSEEK32 ||
            model->arch == LLM_ARCH_DEEPSEEK4 ||
            model->arch == LLM_ARCH_DEEPSEEK41 ||
            model->arch == LLM_ARCH_GLM_DSA ||
            model->arch == LLM_ARCH_GLM5_NEXT ||
            model->arch == LLM_ARCH_DOTS3NOTE ||
            model->arch == LLM_ARCH_MISTRAL_LARGE4 ||
            model->arch == LLM_ARCH_HY_V4 ||
            model->arch == LLM_ARCH_MINIMAX_M3 ||
            model->arch == LLM_ARCH_QWEN4EXP ||
            model->arch == LLM_ARCH_DFLASH)) {
        LLAMA_LOG_ERROR("%s: FP8 cache is not supported for architecture '%s'\n", __func__, llm_arch_name(model->arch));
        return nullptr;
    }

    if (fp8_cache) {
        for (uint32_t il = 0; il < model->hparams.n_layer_all; ++il) {
            if (!model->hparams.has_kv(il)) {
                continue;
            }

            for (ggml_tensor * scale : { model->layers[il].k_cache_s, model->layers[il].v_cache_s }) {
                if (!scale || !scale->data) {
                    continue;
                }

                float value;
                ggml_backend_tensor_get(scale, &value, 0, sizeof(value));
                if (!std::isfinite(value) || value == 0.0f) {
                    LLAMA_LOG_ERROR("%s: FP8 cache scale '%s' must be finite and nonzero\n", __func__, ggml_get_name(scale));
                    return nullptr;
                }
            }
        }
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        (model->hparams.n_layer_nextn == 0 || model->hparams.router_layer >= 0)) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        const auto & cparams = ctx->get_cparams();

        if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN && cparams.rope_freq_scale != model->hparams.rope_freq_scale_train) {
            LLAMA_LOG_INFO("%s: custom YaRN scaling detected, re-adjusting n_ctx_train(%u)...\n", __func__, model->hparams.n_ctx_train);
            model->hparams.n_ctx_train = cparams.n_ctx_orig_yarn / cparams.rope_freq_scale;
            LLAMA_LOG_INFO("%s: n_ctx_train adjusted to %u\n", __func__, model->hparams.n_ctx_train);
        }

        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

bool llama_get_causal_attn(const llama_context * ctx) {
    return ctx->get_causal_attn();
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

bool llama_set_embeddings_nextn_dev(llama_context * ctx, bool value) {
    return ctx->set_embeddings_nextn_dev(value);
}

bool llama_get_embeddings_nextn_rows(llama_context * ctx, int64_t row0, int64_t n_rows, float * out) {
    return ctx->get_h_nextn_rows(row0, n_rows, out);
}

bool llama_set_embeddings_nextn_rows(llama_context * ctx, int64_t row0, int64_t n_rows, const float * data) {
    return ctx->set_h_nextn_rows(row0, n_rows, data);
}

bool llama_set_ctx_state_src(llama_context * ctx, llama_context * ctx_src) {
    return ctx->set_ctx_state_src(ctx_src);
}

void llama_set_embeddings_layer_inp_dev(llama_context * ctx, bool value) {
    ctx->set_embeddings_layer_inp_dev(value);
}

bool llama_supports_embd_row(const llama_context * ctx) {
    return ctx->supports_embd_row();
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

bool llama_memory_get_seq_capacity(llama_memory_t mem, llama_seq_id seq_id, llama_memory_seq_capacity * out) {
    if (!out) {
        return false;
    }
    *out = {};
    return mem && mem->get_seq_capacity(seq_id, *out);
}

bool llama_memory_get_resource_capacity(llama_memory_t mem, llama_seq_id seq_id, llama_memory_resource_capacity * out, size_t out_size) {
    if (!out) {
        return false;
    }
    if (out_size < sizeof(*out)) {
        memset(out, 0, out_size);
        return false;
    }
    *out = {};
    if (!mem) {
        return false;
    }
    llama_memory_resource_capacity result = {};
    if (!mem->get_resource_capacity(seq_id, result)) {
        return false;
    }
    *out = result;
    return true;
}

// llama state API

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}


// One engine-owned immutable capture, one open file handle. Both passes use the
// SAME native parser; there is no independent scanner or pathname reopen.
class llama_io_read_restore_snapshot : public llama_io_read_i {
    const uint8_t * ptr; size_t remaining,read_bytes=0; uint64_t tensor_bytes=0;
    bool validating;
public:
    llama_io_read_restore_snapshot(const uint8_t * data,size_t size,llama_state_seq_limits * l,bool validation)
        : llama_io_read_i(l),ptr(data),remaining(size),validating(validation) {}
    bool validation_only() const override { return validating; }
    bool destination_precleared() const override { return !validating; }
    void restore_cells(uint32_t kind,uint32_t stream,uint64_t count) override {
        // The immutable capture was charged once in the validation pass. The
        // execution pass cannot change any count, stream or payload byte.
        if(validating) llama_io_read_i::restore_cells(kind,stream,count);
    }
    void read(void * dst,size_t size) override {
        if(size>remaining) throw std::runtime_error("truncated immutable restore snapshot");
        if(size) std::memcpy(dst,ptr,size);
        ptr+=size;remaining-=size;read_bytes+=size;
    }
    void read_tensor(ggml_tensor * tensor,size_t offset,size_t size) override {
        if(size>remaining || size>limits->max_tensor_bytes) {
            limits->limit_refused=true;throw std::runtime_error("bounded tensor payload refused");
        }
        if(!validating) {
            if(size>std::numeric_limits<uint64_t>::max()-tensor_bytes) throw std::runtime_error("tensor byte overflow");
            tensor_bytes+=size;
            if(limits->before_tensor_write && !limits->before_tensor_write(limits->user,tensor_bytes))
                throw std::runtime_error("injected truncated tensor after replacement");
            ggml_backend_tensor_set(tensor,ptr,offset,size);
        }
        ptr+=size;remaining-=size;read_bytes+=size;
    }
    size_t n_bytes() override { return read_bytes; }
};

uint64_t llama_context::state_restore_file_capacity(llama_seq_id seq) const {
    if(!memory || seq<0) return 0;
    llama_memory_resource_capacity raw{};
    if(!memory->get_resource_capacity(seq,raw)) return 0;
    uint64_t result=state_restore_tensor_capacity();if(!result) return 0;
    auto sum=[&](uint64_t a,uint64_t b) {
        if(a && b>std::numeric_limits<uint64_t>::max()/a) return false;
        const uint64_t part=a*b;
        if(part>std::numeric_limits<uint64_t>::max()-result) return false;
        result+=part;return true;
    };
    // Physical tensor buffers bound all payloads. Each single-seq attention
    // cell carries pos/u32/one seq ID/optional ext; each recurrent cell pos/u32.
    // Per attention layer: K(12), V(12), optional k_idx flag/type/row(16).
    if(!sum(3,sizeof(uint32_t)) || !sum(cparams.n_ctx,sizeof(llama_token))) return 0;
    if(raw.attention_known) {
        const uint64_t streams=raw.attention_stream_count;
        if(!streams || !sum(uint64_t(raw.attention_capacity_cells)*streams,
                sizeof(llama_pos)+sizeof(uint32_t)+sizeof(llama_seq_id)+sizeof(llama_kv_cell_ext)) ||
                !sum(streams,sizeof(uint32_t)+12+uint64_t(model.hparams.n_layer())*40) || !sum(1,sizeof(uint32_t))) return 0;
    }
    if(raw.recurrent_known) {
        const uint64_t groups=raw.recurrent_rollback_groups;
        if(!groups || raw.recurrent_capacity_rows%groups ||
                !sum(raw.recurrent_capacity_rows/groups,sizeof(llama_pos)+sizeof(uint32_t)) ||
                !sum(1,sizeof(uint32_t)+8+uint64_t(model.hparams.n_layer())*24)) return 0;
    }
    return result;
}

size_t llama_context::state_seq_load_file_bounded(llama_seq_id seq,const char * path,llama_token * tokens,
        size_t capacity,size_t * count,llama_state_seq_limits * limits) {
    std::vector<uint8_t> snapshot;
    try {
        llama_file file(path,"rb");
        uint64_t bound=state_restore_file_capacity(seq);
        if(limits->max_snapshot_bytes) bound=std::min(bound,limits->max_snapshot_bytes);
        const size_t bytes=file.size();
        if(!bound || bytes>bound || bytes<3*sizeof(uint32_t)) {
            limits->limit_refused=true;throw std::runtime_error("snapshot exceeds derived host bound");
        }
        snapshot.resize(bytes);file.read_raw(snapshot.data(),bytes);
        uint32_t header[3];std::memcpy(header,snapshot.data(),sizeof(header));
        if(header[0]!=LLAMA_STATE_SEQ_MAGIC || header[1]!=LLAMA_STATE_SEQ_VERSION || header[2]>capacity ||
                uint64_t(header[2])*sizeof(llama_token)>bytes-sizeof(header)) throw std::runtime_error("invalid bounded snapshot header");
        const size_t prefix=sizeof(header)+size_t(header[2])*sizeof(llama_token);
        const size_t payload=bytes-prefix;
        size_t native_bytes = 0;
        {
            llama_io_read_restore_snapshot validation(snapshot.data()+prefix,payload,limits,true);
            native_bytes = state_seq_read_data(validation,seq,0);
            if (!native_bytes || native_bytes > payload) throw std::runtime_error("invalid bounded snapshot");
        }
        limits->mutation_started=true;++limits->clear_count;
        if(limits->before_replace) limits->before_replace(limits->user);
        if(memory && !memory->seq_rm(seq,-1,-1)) throw std::runtime_error("bounded destination replacement refused");
        {
            llama_io_read_restore_snapshot execution(snapshot.data()+prefix,payload,limits,false);
            if(state_seq_read_data(execution,seq,0)!=native_bytes) throw std::runtime_error("bounded snapshot execution mismatch");
        }
        // Keep the immutable host capture alive until all tensor writes fence.
        synchronize();
        if(header[2]) std::memcpy(tokens,snapshot.data()+sizeof(header),size_t(header[2])*sizeof(llama_token));
        *count=header[2];return prefix + native_bytes;
    } catch(...) {
        if(limits->mutation_started && memory) { ++limits->clear_count;memory->seq_rm(seq,-1,-1); }
        synchronize(); // before destroying any queued-write source buffer
        return 0;
    }
}

uint64_t llama_state_seq_restore_file_capacity(llama_context * ctx,llama_seq_id seq) {
    return ctx ? ctx->state_restore_file_capacity(seq) : 0;
}

uint64_t llama_context::state_restore_tensor_capacity() const {
    if (!memory) return 0;
    uint64_t total=0;
    for (const auto & item:memory->memory_breakdown()) {
        if (item.second>std::numeric_limits<uint64_t>::max()-total) return 0;
        total+=item.second;
    }
    return total;
}

uint64_t llama_state_seq_restore_tensor_capacity(llama_context * ctx) {
    return ctx ? ctx->state_restore_tensor_capacity() : 0;
}

size_t llama_state_seq_load_file_bounded(llama_context * ctx, const char * filepath, llama_seq_id seq,
        llama_token * tokens, size_t capacity, size_t * count, llama_state_seq_limits * limits) {
    if (!ctx || !limits || !limits->consume || seq<0 || !count || (!tokens && capacity)) return 0;
    limits->clear_count=0; limits->allocation_count=0; limits->fence_count=0;
    limits->mutation_started=false; limits->limit_refused=false;
    ctx->synchronize();
    llama_memory_resource_capacity physical{};
    if (!llama_memory_get_resource_capacity(llama_get_memory(ctx),seq,&physical,sizeof(physical)) ||
            !physical.attention_known) {
        limits->limit_refused=true;
        ctx->synchronize(); ++limits->fence_count;
        return 0; // refuse unknown / recurrent-only layout BEFORE the sub-readers (hybrid attention+recurrent is read by the native readers, charged per (kind,stream))
    }
    size_t result=0;
    try { result=ctx->state_seq_load_file(seq,filepath,tokens,capacity,count,limits); }
    catch (const std::exception & err) { LLAMA_LOG_ERROR("%s: %s\n",__func__,err.what()); }
    // Includes failed/truncated reads and all queued tensor writes/cleanup.
    ctx->synchronize();
    ++limits->fence_count;
    return result;
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

// compat: llama_batch -> llama_batch_ext -> encode/decode

int llama_context::encode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp, model.hparams.n_embd_inp_enc());
    return encode(*compat.batch_ext);
}

int llama_context::decode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp);
    return decode(*compat.batch_ext);
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
    LLAMA_LOG_INFO("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__, data.n_reallocs_same, data.n_reallocs_diff);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    switch (type) {
        case LLAMA_PROCESS_TYPE_ENCODE: return ctx->encode(*batch);
        case LLAMA_PROCESS_TYPE_DECODE: return ctx->decode(*batch);
    }
    return -1;
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}
