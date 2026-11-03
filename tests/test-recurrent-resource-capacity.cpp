#include "ggml-backend.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"

#include <cstdio>
#include <limits>
#include <vector>

struct recurrent_test_model : llama_model {
    recurrent_test_model() : llama_model(llama_model_params{}) {}

    void load_stats(llama_model_loader &) override {}
    void load_hparams(llama_model_loader &) override {}
    void load_vocab(llama_model_loader &) override {}
    bool load_tensors(llama_model_loader &) override { return false; }
    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override { return {}; }
};

static bool snapshot_bytes(const llama_memory_recurrent & memory, std::vector<uint8_t> & out) {
    out.clear();
    for (const ggml_tensor * tensor : memory.r_l) {
        if (!tensor) continue;
        const size_t offset = out.size();
        out.resize(offset + ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, out.data() + offset, 0, ggml_nbytes(tensor));
    }
    for (const ggml_tensor * tensor : memory.s_l) {
        if (!tensor) continue;
        const size_t offset = out.size();
        out.resize(offset + ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, out.data() + offset, 0, ggml_nbytes(tensor));
    }
    return true;
}

static bool metadata_matches(const llama_memory_recurrent & memory,
        const std::vector<llama_memory_recurrent::mem_cell> & cells,
        const std::vector<uint32_t> & rs_idx, uint32_t used, int32_t rs_z) {
    if (memory.used != used || memory.rs_idx != rs_idx || memory.rs_z != rs_z || memory.cells.size() != cells.size()) {
        return false;
    }
    for (size_t i = 0; i < cells.size(); ++i) {
        const auto & a = memory.cells[i];
        const auto & b = cells[i];
        if (a.pos != b.pos || a.src != b.src || a.src0 != b.src0 || a.tail != b.tail || a.seq_id != b.seq_id) {
            return false;
        }
    }
    return true;
}

static bool check_observation(llama_memory_recurrent & memory, llama_seq_id seq,
        bool expected_known, uint32_t expected_used, uint32_t expected_seq, uint32_t expected_exclusive) {
    const auto cells = memory.cells;
    const auto rs_idx = memory.rs_idx;
    const uint32_t used = memory.used;
    const int32_t rs_z = memory.rs_z;
    std::vector<uint8_t> before;
    std::vector<uint8_t> after;
    snapshot_bytes(memory, before);

    llama_memory_resource_capacity result = {};
    const bool known = memory.get_resource_capacity(seq, result);
    snapshot_bytes(memory, after);
    if (known != expected_known || before != after || !metadata_matches(memory, cells, rs_idx, used, rs_z)) {
        fprintf(stderr, "%s: read-only/native observation invariant failed (known=%d expected=%d)\n",
                __func__, known, expected_known);
        return false;
    }
    if (!known) {
        if (result.recurrent_known || result.recurrent_capacity_rows || result.recurrent_used_rows ||
                result.recurrent_seq_rows || result.recurrent_seq_exclusive_rows || result.recurrent_rollback_groups) {
            fprintf(stderr, "%s: unknown recurrent observation was not zeroed\n", __func__);
            return false;
        }
        return true;
    }
    if (!result.recurrent_known || result.recurrent_capacity_rows != 12 ||
            result.recurrent_used_rows != expected_used || result.recurrent_seq_rows != expected_seq ||
            result.recurrent_seq_exclusive_rows != expected_exclusive || result.recurrent_rollback_groups != 3) {
        fprintf(stderr, "%s: unexpected recurrent capacity fields\n", __func__);
        return false;
    }
    return true;
}

int main() {
    recurrent_test_model model;
    model.hparams.n_layer_all = 1;
    model.hparams.n_embd = 1;
    model.hparams.ssm_d_conv = 2;
    model.hparams.ssm_d_inner = 1;
    model.hparams.ssm_n_group = 1;
    model.hparams.ssm_d_state = 1;
    llama_memory_i::layer_filter_cb filter = [](int32_t) { return true; };
    llama_memory_recurrent memory(model, GGML_TYPE_F32, GGML_TYPE_F32, false, 4, 2, 2, filter);

    if (!check_observation(memory, 0, true, 0, 0, 0)) return 1;
    if (!check_observation(memory, 2, false, 0, 0, 0)) return 1;

    memory.cells[0].pos = 0;
    memory.cells[0].seq_id.insert(0);
    memory.cells[0].tail = 0;
    memory.used = 1;
    if (!check_observation(memory, 0, true, 3, 3, 3)) return 1;

    memory.cells[0].seq_id.insert(1);
    memory.cells[1].tail = 0;
    if (!check_observation(memory, 0, true, 3, 3, 0)) return 1;

    const auto valid_cells = memory.cells;
    const auto valid_rs_idx = memory.rs_idx;
    const uint32_t valid_used = memory.used;
    const int32_t valid_rs_z = memory.rs_z;
    memory.cells[1].tail = 3;
    if (!check_observation(memory, 0, false, 0, 0, 0)) return 1;
    memory.cells = valid_cells;
    memory.rs_idx = valid_rs_idx;
    memory.used = valid_used;
    memory.rs_z = valid_rs_z;

    memory.used = 0;
    if (!check_observation(memory, 0, false, 0, 0, 0)) return 1;
    memory.used = valid_used;

    memory.rs_idx[0] = 3;
    if (!check_observation(memory, 0, false, 0, 0, 0)) return 1;
    memory.rs_idx[0] = 0;

    memory.n_rs_seq = std::numeric_limits<uint32_t>::max();
    if (!check_observation(memory, 0, false, 0, 0, 0)) return 1;
    memory.n_rs_seq = 2;

    memory.cells[2].src = 3;
    if (!check_observation(memory, 0, false, 0, 0, 0)) return 1;

    fprintf(stderr, "%s: recurrent rows/tails/rollback/overflow/orphan gates passed\n", __func__);
    return 0;
}
