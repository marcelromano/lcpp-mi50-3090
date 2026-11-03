#include "llama-memory-hybrid.h"

#include <cstdio>

static bool zero(const llama_memory_resource_capacity & out) {
    return !out.attention_known && out.attention_stream_id == 0 && out.attention_stream_count == 0 &&
        out.attention_capacity_cells == 0 && out.attention_used_cells == 0 && out.attention_seq_cells == 0 &&
        out.attention_seq_exclusive_cells == 0 && !out.recurrent_known && out.recurrent_capacity_rows == 0 &&
        out.recurrent_used_rows == 0 && out.recurrent_seq_rows == 0 && out.recurrent_seq_exclusive_rows == 0 &&
        out.recurrent_rollback_groups == 0;
}

int main() {
    llama_memory_resource_capacity attention = {};
    attention.attention_known = true;
    attention.attention_stream_id = 1;
    attention.attention_stream_count = 2;
    attention.attention_capacity_cells = 10;
    attention.attention_used_cells = 4;
    attention.attention_seq_cells = 3;
    attention.attention_seq_exclusive_cells = 2;

    llama_memory_resource_capacity recurrent = {};
    recurrent.recurrent_known = true;
    recurrent.recurrent_capacity_rows = 24;
    recurrent.recurrent_used_rows = 6;
    recurrent.recurrent_seq_rows = 3;
    recurrent.recurrent_seq_exclusive_rows = 0;
    recurrent.recurrent_rollback_groups = 3;

    llama_memory_resource_capacity out = {};
    if (!llama_memory_compose_resource_capacity(attention, recurrent, out) ||
            !out.attention_known || !out.recurrent_known || out.attention_stream_id != 1 ||
            out.attention_capacity_cells != 10 || out.attention_used_cells != 4 ||
            out.recurrent_capacity_rows != 24 || out.recurrent_used_rows != 6 || out.recurrent_rollback_groups != 3) {
        fprintf(stderr, "%s: composition of known resources failed\n", __func__);
        return 1;
    }

    recurrent.recurrent_known = false;
    out.attention_known = true;
    if (llama_memory_compose_resource_capacity(attention, recurrent, out) || !zero(out)) {
        fprintf(stderr, "%s: unknown recurrent resource exposed a partial hybrid result\n", __func__);
        return 1;
    }
    recurrent.recurrent_known = true;
    attention.attention_known = false;
    out.recurrent_known = true;
    if (llama_memory_compose_resource_capacity(attention, recurrent, out) || !zero(out)) {
        fprintf(stderr, "%s: unknown attention resource exposed a partial hybrid result\n", __func__);
        return 1;
    }
    fprintf(stderr, "%s: atomic hybrid composition gates passed\n", __func__);
    return 0;
}
