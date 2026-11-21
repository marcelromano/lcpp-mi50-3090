#include "../tools/server/kv_planner.hpp"
#include "../tools/server/kv_resource_view.hpp"
#include "llama.h"

#include <cstdio>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main(int argc, char ** argv) {
    CHECK(argc == 2);
    llama_backend_init();

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model_tgt = llama_model_load_from_file(argv[1], model_params);
    llama_model * model_dft = llama_model_load_from_file(argv[1], model_params);
    CHECK(model_tgt && model_dft);

    auto context_params = llama_context_default_params();
    context_params.n_ctx = 512;
    context_params.n_batch = 512;
    context_params.n_ubatch = 128;
    context_params.n_seq_max = 1;
    context_params.n_rs_seq = 0;
    llama_context * ctx_tgt = llama_init_from_model(model_tgt, context_params);
    llama_context * ctx_dft = llama_init_from_model(model_dft, context_params);
    CHECK(ctx_tgt && ctx_dft);

    // Build real memory occupancy only in the draft context; target remains empty.
    std::vector<llama_token> tokens(480, 19);
    for (size_t i = 0; i < tokens.size(); i += 120) {
        auto batch = llama_batch_get_one(tokens.data() + i, 120);
        CHECK(llama_decode(ctx_dft, batch) == 0);
    }
    llama_synchronize(ctx_tgt);
    llama_synchronize(ctx_dft);

    llama_memory_resource_capacity raw_tgt{}, raw_dft{};
    CHECK(llama_memory_get_resource_capacity(llama_get_memory(ctx_tgt), 0, &raw_tgt, sizeof(raw_tgt)));
    CHECK(llama_memory_get_resource_capacity(llama_get_memory(ctx_dft), 0, &raw_dft, sizeof(raw_dft)));
    std::vector<kvplan::PhysicalView> tgt, dft;
    CHECK(kvplan::normalize_resources(kvplan::Context::target, 31, raw_tgt, tgt));
    CHECK(kvplan::normalize_resources(kvplan::Context::draft, 31, raw_dft, dft));
    CHECK(tgt.size() == 1 && dft.size() == 1);
    CHECK(tgt[0].used == 0 && dft[0].used >= 400);
    CHECK(tgt[0].capacity == dft[0].capacity);

    kvplan::Snapshot s_tgt{}, s_dft{};
    s_tgt.key = tgt[0].key; s_tgt.known = true; s_tgt.capacity = tgt[0].capacity; s_tgt.used_real = tgt[0].used;
    s_dft.key = dft[0].key; s_dft.known = true; s_dft.capacity = dft[0].capacity; s_dft.used_real = dft[0].used;
    kvplan::Planner planner;
    const kvplan::Owner control{31, 0, 0};
    kvplan::Reservation control_res;
    control_res.owner = control;
    control_res.peak_by_resource.emplace(s_tgt.key, 1);
    control_res.peak_by_resource.emplace(s_dft.key, 1);
    CHECK(planner.admit({s_tgt, s_dft}, {control_res}).status == kvplan::Status::accepted);

    const kvplan::Owner pressured{31, 1, 0};
    kvplan::Reservation pressured_res;
    pressured_res.owner = pressured;
    pressured_res.peak_by_resource.emplace(s_tgt.key, 1);
    pressured_res.peak_by_resource.emplace(s_dft.key, s_dft.capacity - s_dft.used_real + 1);
    const auto denied = planner.admit({s_tgt, s_dft}, {pressured_res});
    CHECK(denied.status == kvplan::Status::rejected);
    CHECK(denied.reason == "capacity exceeded");
    // Target remains independently feasible; the AND refusal is caused by draft pressure.
    CHECK(s_tgt.used_real + 2 <= s_tgt.capacity);

    llama_free(ctx_dft);
    llama_free(ctx_tgt);
    llama_model_free(model_dft);
    llama_model_free(model_tgt);
    llama_backend_free();
    return 0;
}
