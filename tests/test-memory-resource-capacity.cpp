#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

static bool expect_zero(const llama_memory_resource_capacity & value) {
    return !value.attention_known && value.attention_stream_id == 0 && value.attention_stream_count == 0 &&
        value.attention_capacity_cells == 0 && value.attention_used_cells == 0 && value.attention_seq_cells == 0 &&
        value.attention_seq_exclusive_cells == 0 && !value.recurrent_known && value.recurrent_capacity_rows == 0 &&
        value.recurrent_used_rows == 0 && value.recurrent_seq_rows == 0 && value.recurrent_seq_exclusive_rows == 0 &&
        value.recurrent_rollback_groups == 0;
}

static bool get_capacity(llama_context * ctx, llama_memory_resource_capacity & out) {
    out = {};
    return llama_memory_get_resource_capacity(llama_get_memory(ctx), 0, &out, sizeof(out));
}

static bool check_short_buffer(llama_context * ctx) {
    llama_memory_resource_capacity short_out;
    memset(&short_out, 0xA5, sizeof(short_out));
    const size_t short_size = offsetof(llama_memory_resource_capacity, recurrent_known);
    if (llama_memory_get_resource_capacity(llama_get_memory(ctx), 0, &short_out, short_size)) return false;
    const auto * bytes = reinterpret_cast<const unsigned char *>(&short_out);
    for (size_t i = 0; i < short_size; ++i) if (bytes[i] != 0) return false;
    for (size_t i = short_size; i < sizeof(short_out); ++i) if (bytes[i] != 0xA5) return false;
    return true;
}

int main(int argc, char ** argv) {
    bool force_nonunified = false;
    std::vector<char *> parsed_argv;
    parsed_argv.reserve(argc);
    parsed_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--resource-nonunified") == 0) {
            force_nonunified = true;
        } else {
            parsed_argv.push_back(argv[i]);
        }
    }
    common_params params;
    params.n_predict = 1;
    params.kv_unified = !force_nonunified;
    common_init();
    if (!common_params_parse((int) parsed_argv.size(), parsed_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();
    common_init_result_ptr init = common_init_from_params(params);
    llama_model * model = init->model();
    if (!model) return 1;
    if (llama_model_is_recurrent(model) || llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s: expected a Transformer attention fixture\n", __func__);
        return 1;
    }

    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx = 512;
    cparams.n_seq_max = 2;
    cparams.n_batch = std::max(cparams.n_batch, 2u);
    cparams.n_ubatch = std::max(cparams.n_ubatch, 2u);
    auto ctx = llama_context_ptr{llama_init_from_model(model, cparams)};
    if (!ctx) return 1;

    llama_memory_resource_capacity empty = {};
    if (!get_capacity(ctx.get(), empty) || !empty.attention_known || empty.recurrent_known ||
            empty.attention_used_cells != 0 || empty.attention_seq_cells != 0 ||
            empty.attention_seq_exclusive_cells != 0 || !check_short_buffer(ctx.get())) {
        fprintf(stderr, "%s: empty attention observation failed\n", __func__);
        return 1;
    }

    const auto tokens = common_tokenize(ctx.get(), "the quick brown fox", true);
    if (tokens.size() < 2) return 1;
    llama_batch batch = llama_batch_init(2, 0, 1);
    common_batch_add(batch, tokens[0], 0, { 0 }, false);
    common_batch_add(batch, tokens[1], 0, { 1 }, true);
    if (llama_decode(ctx.get(), batch) != 0) {
        llama_batch_free(batch);
        return 1;
    }
    llama_batch_free(batch);

    llama_memory_resource_capacity seq0 = {};
    llama_memory_resource_capacity seq1 = {};
    const bool unified = cparams.kv_unified;
    if (!get_capacity(ctx.get(), seq0) ||
            !llama_memory_get_resource_capacity(llama_get_memory(ctx.get()), 1, &seq1, sizeof(seq1)) ||
            !seq0.attention_known || !seq1.attention_known || seq0.recurrent_known ||
            seq0.attention_used_cells < (unified ? 2u : 1u) || !seq0.attention_seq_cells || !seq1.attention_seq_cells) {
        fprintf(stderr, "%s: filled attention observation failed (a0=%d a1=%d streams=%u/%u ids=%u/%u used=%u/%u seq=%u/%u rec=%d)\n",
                __func__, seq0.attention_known, seq1.attention_known, seq0.attention_stream_count,
                seq1.attention_stream_count, seq0.attention_stream_id, seq1.attention_stream_id,
                seq0.attention_used_cells, seq1.attention_used_cells, seq0.attention_seq_cells,
                seq1.attention_seq_cells, seq0.recurrent_known);
        return 1;
    }

    if (unified) {
        if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 1, 0, -1)) {
            fprintf(stderr, "%s: failed to clear the destination sequence before sharing\n", __func__);
            return 1;
        }
        llama_memory_seq_cp(llama_get_memory(ctx.get()), 0, 1, 0, -1);
        llama_memory_resource_capacity shared0 = {};
        llama_memory_resource_capacity shared1 = {};
        if (!get_capacity(ctx.get(), shared0) ||
                !llama_memory_get_resource_capacity(llama_get_memory(ctx.get()), 1, &shared1, sizeof(shared1)) ||
                shared0.attention_seq_cells != shared1.attention_seq_cells ||
                shared0.attention_seq_exclusive_cells != 0 || shared1.attention_seq_exclusive_cells != 0) {
            fprintf(stderr, "%s: shared attention ownership observation failed\n", __func__);
            return 1;
        }
    } else {
        if (seq0.attention_stream_id == seq1.attention_stream_id || seq0.attention_stream_count < 2) {
            fprintf(stderr, "%s: nonunified stream mapping was not observed\n", __func__);
            return 1;
        }
        llama_memory_seq_cp(llama_get_memory(ctx.get()), 0, 1, 0, -1);
        llama_memory_resource_capacity pending = {};
        if (llama_memory_get_resource_capacity(llama_get_memory(ctx.get()), 0, &pending, sizeof(pending)) ||
                !expect_zero(pending)) {
            fprintf(stderr, "%s: pending cross-stream copy did not fail closed\n", __func__);
            return 1;
        }
    }
    fprintf(stderr, "%s: attention resource observation passed (unified=%d)\n", __func__, unified);
    return 0;
}
