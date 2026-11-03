#include "server-context.h"
#include <cstdlib>

// CPU integration instrumentation is absent unless explicitly compiled in.
static const char * kv_admission_cpu_test_env(const char * name) {
#ifdef LLAMA_KV_ADMISSION_CPU_TEST_DRIVER
    return getenv(name);
#else
    (void) name;
    return nullptr;
#endif
}
#include "server-http.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-stream.h"
#include "server-tools.h"
#include <cpp-httplib/httplib.h>

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <clocale>
#include <exception>
#include <signal.h>
#include <thread> // for std::thread::hardware_concurrency
#include <future>
#include <chrono>

#if defined(_WIN32)
#include <windows.h>
#endif

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

#ifdef LLAMA_KV_ADMISSION_CPU_TEST_DRIVER
// One isolated binding/epoch or no-hook scenario per server process. Identical
// actor/candidate requests make each arm independently reproducible.
static int run_kv_admission_c_scenario(int port, server_context & ctx_server, const char * scenario) {
    const bool binding = std::strcmp(scenario, "binding") == 0 || std::strcmp(scenario, "omit-binding") == 0;
    const bool epoch = std::strcmp(scenario, "epoch") == 0 || std::strcmp(scenario, "omit-epoch") == 0;
    const bool hooked = std::strcmp(scenario, "control") != 0 &&
        std::strcmp(scenario, "omit-binding") != 0 && std::strcmp(scenario, "omit-epoch") != 0;
    if ((!binding && !epoch && std::strcmp(scenario, "control") != 0) ||
            !ctx_server.test_only_hybrid_draft_guard()) return 9;
    const bool b_native = ctx_server.test_only_recurrent_getter_cases();
    std::fprintf(stderr, "KV_ADMISSION_TEST gate=B-native-result scenario=%s pass=%d\n", scenario, (int)b_native);
    std::thread dispatcher([&ctx_server]() { ctx_server.start_loop(); });
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(2, 0);
    client.set_read_timeout(120, 0);
    client.set_write_timeout(10, 0);
    bool ready = false;
    for (int i = 0; i < 200; ++i) {
        auto health = client.Get("/health");
        if (health && health->status == 200) { ready = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!ready) { ctx_server.terminate(); dispatcher.join(); return 10; }
    auto snapshot = [&]() -> std::optional<json> {
        auto slots = client.Get("/slots");
        if (!slots || slots->status != 200) return std::nullopt;
        try { return json::parse(slots->get_header_value("X-KV-Admission-Test-Snapshot")); }
        catch (...) { return std::nullopt; }
    };
    auto actor_future = std::async(std::launch::async, [port]() {
        httplib::Client c("127.0.0.1", port);
        c.set_read_timeout(120, 0);
        return c.Post("/completion", R"({"prompt":"same-candidate","n_predict":200,"stream":false,"ignore_eos":true,"seed":8,"id_slot":0})", "application/json");
    });
    std::optional<json> before;
    bool actor_owned = false;
    for (int i = 0; i < 400; ++i) {
        auto s = snapshot();
        if (s && s->contains("owners") && !(*s)["owners"].empty()) {
            for (const auto & owner : (*s)["owners"]) if (owner.value("slot", UINT64_MAX) == 0) actor_owned = true;
            if (actor_owned) { before = s; break; }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    if (!actor_owned) { ctx_server.terminate(); dispatcher.join(); return 11; }
    bool fence_loop_ok = true;
    for (int i = 0; i < 3; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        auto observed = snapshot();
        bool found_actor = false;
        if (observed && before && (*observed)["launch_count"] == (*before)["launch_count"] &&
                (*observed)["ledger_size"] == (*before)["ledger_size"]) {
            for (const auto & owner : (*observed)["owners"]) {
                if (owner.value("slot", UINT64_MAX) == 0 &&
                        std::find((*before)["owners"].begin(), (*before)["owners"].end(), owner) != (*before)["owners"].end()) found_actor = true;
            }
        }
        fence_loop_ok = fence_loop_ok && found_actor;
    }
    if (hooked) setenv("LLAMA_KV_ADMISSION_TEST_FAULT", binding ? "binding" : "epoch", 1);
    auto candidate = client.Post("/completion", R"({"prompt":"same-candidate","n_predict":2,"stream":false,"ignore_eos":true,"seed":8,"id_slot":1})", "application/json");
    unsetenv("LLAMA_KV_ADMISSION_TEST_FAULT");
    const std::string expected = binding ? "invalid exclusive owner/overflow" :
        (epoch ? "exclusive owner has no commitment on resource" : "");
    bool checker = false;
    std::string actual_message;
    if (candidate) {
        try {
            const json body = json::parse(candidate->body);
            if (body.contains("error") && body["error"].is_object()) actual_message = body["error"].value("message", "");
        } catch (...) {}
    }
    const auto same_refusal_checker = [&]() {
        return candidate && candidate->status == 503 &&
            actual_message == std::string("KV admission refused: ") + expected;
    };
    if (!expected.empty()) {
        // Both hooked and omit-hook arms invoke this exact parsed-message checker.
        checker = same_refusal_checker();
    } else {
        checker = candidate && candidate->status == 200;
    }
    auto after = snapshot();
    bool owner_preserved = before && after;
    if (owner_preserved) {
        for (const auto & owner : (*before)["owners"]) {
            if (owner.value("slot", UINT64_MAX) == 0 && std::find((*after)["owners"].begin(), (*after)["owners"].end(), owner) == (*after)["owners"].end()) owner_preserved = false;
        }
        if (hooked) owner_preserved = owner_preserved && (*before)["launch_count"] == (*after)["launch_count"] && (*before)["ledger_size"] == (*after)["ledger_size"];
    }
    auto actor = actor_future.get();
    const bool actor_done = actor && actor->status == 200;
    auto final_state = snapshot();
    const bool no_leak = final_state && (*final_state)["release_failed"] == false &&
        (*final_state)["owners"].empty() && (*final_state)["ledger_size"] == 0;
    std::fprintf(stderr, "KV_ADMISSION_TEST gate=C scenario=%s candidate_status=%d actual_message=\"%s\" checker_exit=%d expected_checker_exit=%d actor_owner_preserved=%d fence_loop=%d actor_status=%d no_release_leak=%d\n",
        scenario, candidate ? candidate->status : 0, actual_message.c_str(), (int)!checker,
        (!hooked && !expected.empty()) ? 1 : 0, (int)owner_preserved, (int)fence_loop_ok, actor ? actor->status : 0, (int)no_leak);
    ctx_server.terminate();
    dispatcher.join();
    ctx_server.test_only_synchronize_contexts();
    const bool control_success = std::strcmp(scenario, "control") == 0 && checker && actor_done && owner_preserved && fence_loop_ok && no_leak;
    const bool hooked_success = hooked && checker && actor_done && owner_preserved && fence_loop_ok && no_leak;
    const bool omit_success = !hooked && !expected.empty() && candidate && candidate->status == 200 && !checker && actor_done && owner_preserved && fence_loop_ok && no_leak;
    return control_success || hooked_success || omit_success ? 0 : 12;
}

// Independent pressure/control arms use the same actor and exact public candidate.
// The sole difference is native occupancy of draft stream 1 before dispatch starts.
static int run_kv_admission_pair_driver(int port, server_context & ctx_server, const char * arm) {
    const bool pressure = std::strcmp(arm, "pressure") == 0;
    if (!pressure && std::strcmp(arm, "control") != 0) return 20;
    if (!ctx_server.test_only_mtp_hybrid_path()) return 21;
    if (pressure && !ctx_server.test_only_populate_draft_sequence(1, -1)) return 22;
    std::thread dispatcher([&ctx_server]() { ctx_server.start_loop(); });
    auto finish = [&]() {
        ctx_server.terminate();
        if (dispatcher.joinable()) dispatcher.join();
        ctx_server.test_only_synchronize_contexts();
        const bool cleared = ctx_server.test_only_clear_draft_sequence(1);
        std::fprintf(stderr, "KV_ADMISSION_TEST pair_teardown arm=%s joined=1 synchronized=1 fixture_cleared=%d\n", arm, (int)cleared);
        return cleared;
    };
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(2, 0);
    client.set_read_timeout(120, 0);
    client.set_write_timeout(10, 0);
    bool ready = false;
    for (int i = 0; i < 100; ++i) {
        auto health = client.Get("/health");
        if (health && health->status == 200) { ready = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!ready) { finish(); return 23; }
    auto snapshot = [&]() -> std::optional<json> {
        auto response = client.Get("/metrics");
        if (!response || response->status != 200) return std::nullopt;
        try { return json::parse(response->get_header_value("X-KV-Admission-Test-Snapshot")); }
        catch (...) { return std::nullopt; }
    };
    auto actor_future = std::async(std::launch::async, [port]() {
        httplib::Client c("127.0.0.1", port);
        c.set_read_timeout(120, 0);
        return c.Post("/completion", R"({"prompt":"One plus one equals","n_predict":200,"temperature":0.8,"top_k":10,"top_p":1.0,"min_p":0.0,"stream":false,"ignore_eos":true,"seed":1938,"id_slot":0})", "application/json");
    });
    std::optional<json> before;
    bool actor_active = false;
    for (int i = 0; i < 200; ++i) {
        before = snapshot();
        if (before && before->contains("owners") && before->at("owners").is_array()) {
            for (const auto & owner : before->at("owners")) {
                actor_active = owner.value("slot", UINT64_MAX) == 0 && owner.value("epoch", UINT64_MAX) != 0;
            }
        }
        if (actor_active) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!actor_active || !before) { finish(); return 24; }
    const auto candidate = client.Post("/completion", R"({"prompt":"hello","n_predict":8,"stream":true,"temperature":0.8,"top_k":10,"top_p":1.0,"min_p":0.0,"ignore_eos":true,"seed":1940,"id_slot":1})", "application/json");
    auto after_candidate = snapshot();
    bool candidate_ok = false;
    bool immutable_refusal = false;
    bool committed_control = false;
    if (pressure) {
        bool json_capacity = false;
        if (candidate && candidate->status == 503) {
            try {
                const auto body = json::parse(candidate->body);
                json_capacity = body.contains("error") && body["error"].value("type", "") == "kv_capacity_error" &&
                    body["error"].value("message", "").find("KV admission refused: capacity exceeded") != std::string::npos &&
                    candidate->get_header_value("Content-Type").find("application/json") != std::string::npos &&
                    candidate->get_header_value("Content-Type").find("text/event-stream") == std::string::npos &&
                    candidate->body.find("data:") == std::string::npos;
            } catch (...) {}
        }
        immutable_refusal = json_capacity && after_candidate &&
            after_candidate->at("owners") == before->at("owners") &&
            after_candidate->at("launch_count") == before->at("launch_count") &&
            after_candidate->at("commit_count") == before->at("commit_count") &&
            after_candidate->at("ledger_size") == before->at("ledger_size");
        candidate_ok = immutable_refusal;
    } else {
        const bool sse = candidate && candidate->status == 200 &&
            candidate->get_header_value("Content-Type").find("text/event-stream") != std::string::npos &&
            candidate->body.find("data:") != std::string::npos &&
            candidate->body.find("\"tokens_predicted\":8") != std::string::npos &&
            candidate->body.find("\"stop\":true") != std::string::npos;
        committed_control = sse && after_candidate &&
            after_candidate->value("commit_count", UINT64_MAX) == before->value("commit_count", UINT64_MAX) + 1 &&
            after_candidate->value("launch_count", UINT64_MAX) == before->value("launch_count", UINT64_MAX) + 1;
        bool candidate_owner_released = after_candidate && after_candidate->at("owners").is_array();
        if (candidate_owner_released) for (const auto & owner : after_candidate->at("owners"))
            if (owner.value("slot", UINT64_MAX) == 1) candidate_owner_released = false;
        candidate_ok = committed_control && candidate_owner_released;
    }
    auto actor = actor_future.get();
    int accepted = 0, predicted = 0;
    bool actor_ok = actor && actor->status == 200;
    if (actor_ok) try {
        const auto timings = json::parse(actor->body).at("timings");
        accepted = timings.value("draft_n_accepted", 0);
        predicted = timings.value("predicted_n", 0);
    } catch (...) { actor_ok = false; }
    auto final_snapshot = snapshot();
    const bool release_ok = final_snapshot && final_snapshot->at("owners").empty() &&
        final_snapshot->value("ledger_size", UINT64_MAX) == 0;
    const bool cleared = finish();
    std::fprintf(stderr, "KV_ADMISSION_TEST pair arm=%s candidate_status=%d candidate_ok=%d refusal_immutable=%d control_commit_launch=%d actor_status=%d actor_predicted=%d native_accepted=%d actor_ok=%d release_empty=%d cleanup=%d\n",
        arm, candidate ? candidate->status : 0, (int)candidate_ok, (int)immutable_refusal, (int)committed_control,
        actor ? actor->status : 0, predicted, accepted, (int)actor_ok, (int)release_ok, (int)cleared);
    return candidate_ok && actor_ok && predicted == 200 && accepted >= 1 && release_ok && cleared ? 0 : 25;
}

// Temporary CPU-only integration driver; only present in the isolated test build.
// It exercises public HTTP routes while the native dispatcher loop is running.
static int run_kv_admission_test_driver(int port, server_context & ctx_server) {
    // MTP support is tied to the live target/MTP context pair and its native layouts.
    if (!ctx_server.test_only_mtp_hybrid_path()) {
        std::fprintf(stderr, "KV_ADMISSION_TEST driver=FAIL gate=A reason=mtp_hybrid_context_map_discriminant\n");
        return 8;
    }
    // Prepare pressure before any request or dispatcher work can touch the draft context.
    if (!ctx_server.test_only_populate_draft_sequence(1, -1)) {
        std::fprintf(stderr, "KV_ADMISSION_TEST driver=FAIL reason=draft_only_decode_fixture_failed\n");
        return 4;
    }
    std::thread dispatcher([&ctx_server]() { ctx_server.start_loop(); });
    auto stop_and_cleanup_fixture = [&]() {
        ctx_server.terminate();
        if (dispatcher.joinable()) dispatcher.join();
        ctx_server.test_only_synchronize_contexts();
        const bool cleared = ctx_server.test_only_clear_draft_sequence(1);
        std::fprintf(stderr, "KV_ADMISSION_TEST teardown joined=1 contexts_synchronized=1 fixture_cleared=%d\n", (int) cleared);
        return cleared;
    };
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(2, 0);
    client.set_read_timeout(15, 0);
    client.set_write_timeout(10, 0);

    bool ready = false;
    for (int i = 0; i < 100; ++i) {
        auto health = client.Get("/health");
        if (health && health->status == 200) { ready = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!ready) {
        std::fprintf(stderr, "KV_ADMISSION_TEST driver=FAIL reason=http_not_ready\n");
        stop_and_cleanup_fixture();
        return 2;
    }

    auto actor_future = std::async(std::launch::async, [port]() {
        httplib::Client actor_client("127.0.0.1", port);
        actor_client.set_read_timeout(120, 0);
        return actor_client.Post("/completion", R"({"prompt":"One plus one equals","n_predict":200,"temperature":0.8,"top_k":10,"top_p":1.0,"min_p":0.0,"stream":false,"ignore_eos":true,"seed":1938,"id_slot":0})", "application/json");
    });
    auto read_slots_snapshot = [&]() -> std::optional<std::pair<json, json>> {
        auto slots = client.Get("/slots");
        if (!slots || slots->status != 200) return std::nullopt;
        try {
            json slot_data = json::parse(slots->body);
            json native_snapshot = json::parse(slots->get_header_value("X-KV-Admission-Test-Snapshot"));
            return std::make_pair(std::move(slot_data), std::move(native_snapshot));
        } catch (...) { return std::nullopt; }
    };
    auto read_metrics_snapshot = [&]() -> std::optional<json> {
        auto metrics = client.Get("/metrics");
        if (!metrics || metrics->status != 200) return std::nullopt;
        try { return json::parse(metrics->get_header_value("X-KV-Admission-Test-Snapshot")); }
        catch (...) { return std::nullopt; }
    };
    bool actor_active = false;
    std::optional<json> before_snapshot;
    for (int i = 0; i < 100; ++i) {
        const auto metrics_snapshot = read_metrics_snapshot();
        if (metrics_snapshot && metrics_snapshot->contains("owners") && (*metrics_snapshot)["owners"].is_array()) {
            for (const auto & owner : (*metrics_snapshot)["owners"]) {
                if (owner.value("slot", UINT64_MAX) == 0 && owner.value("epoch", UINT64_MAX) != 0 &&
                        owner.value("task", UINT64_MAX) != 0 && metrics_snapshot->value("launch_count", 0) > 0) {
                    actor_active = true;
                    before_snapshot = *metrics_snapshot;
                }
            }
        }
        if (actor_active) {
            const auto metrics_snapshot = read_metrics_snapshot();
            actor_active = metrics_snapshot && metrics_snapshot->at("owners") == before_snapshot->at("owners") &&
                           metrics_snapshot->at("launch_count") == before_snapshot->at("launch_count");
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!actor_active) {
        std::fprintf(stderr, "KV_ADMISSION_TEST driver=FAIL reason=prior_actor_not_observed_by_native_metrics_snapshot\n");
    }
    httplib::Result denied;
    if (actor_active) {
        denied = client.Post("/completion", R"({"prompt":"hello","n_predict":8,"stream":true,"id_slot":1})", "application/json");
    }
    bool denied_ok = denied && denied->status == 503;
    bool typed_capacity_refusal = false;
    if (denied_ok) {
        try {
            const json body = json::parse(denied->body);
            denied_ok = body.contains("error") && body["error"].value("type", "") == "kv_capacity_error" &&
                        denied->get_header_value("Content-Type").find("application/json") != std::string::npos &&
                        denied->get_header_value("Content-Type").find("text/event-stream") == std::string::npos &&
                        denied->body.find("data:") == std::string::npos;
            typed_capacity_refusal = body.contains("error") && body["error"].value("message", "").find("KV admission refused: capacity exceeded") != std::string::npos;
            denied_ok = denied_ok && typed_capacity_refusal;
        } catch (...) { denied_ok = false; }
    }
    const auto after_observation = read_slots_snapshot();
    const auto metrics_after = read_metrics_snapshot();
    const bool owner_preserved = before_snapshot && after_observation && metrics_after &&
        after_observation->second.at("owners") == before_snapshot->at("owners") &&
        metrics_after->at("owners") == before_snapshot->at("owners") &&
        after_observation->second.at("launch_count") == before_snapshot->at("launch_count") &&
        metrics_after->at("launch_count") == before_snapshot->at("launch_count") &&
        after_observation->second.at("ledger_size") == before_snapshot->at("ledger_size") &&
        !after_observation->second.at("release_failed").get<bool>();
    std::fprintf(stderr, "KV_ADMISSION_TEST gate=MTP-draft-pressure status=%d typed_json_503=%d capacity_reason=%d no_sse=%d stream_true=%d owner_exact=%d launch_unchanged=%d ledger_unchanged=%d\n",
            denied ? denied->status : 0, (int) denied_ok, (int)typed_capacity_refusal,
            (int)(denied && denied->body.find("data:") == std::string::npos), 1,
            (int) owner_preserved, (int) owner_preserved, (int)owner_preserved);

    bool fault_gates_ok = true;
    for (const char * fault : {"unknown", "binding", "epoch"}) {
        setenv("LLAMA_KV_ADMISSION_TEST_FAULT", fault, 1);
        auto response = client.Post("/completion", R"({"prompt":"hello","n_predict":8,"stream":false,"id_slot":1})", "application/json");
        const auto after_fault = read_slots_snapshot();
        const int expected_status = std::strcmp(fault, "unknown") == 0 ? 501 : 503;
        const char * expected_reason = std::strcmp(fault, "binding") == 0 ? "invalid exclusive owner/overflow" :
            (std::strcmp(fault, "epoch") == 0 ? "exclusive owner has no commitment on resource" :
             (std::strcmp(fault, "unknown") == 0 ? "unknown_memory_resource_snapshot" : "capacity exceeded"));
        const bool exact_branch_reason = response && response->body.find(std::string("KV admission refused: ") + expected_reason) != std::string::npos;
        const bool typed_refusal = response && response->status == expected_status &&
            response->get_header_value("Content-Type").find("application/json") != std::string::npos &&
            response->body.find("data:") == std::string::npos && exact_branch_reason;
        const bool fault_owner_preserved = before_snapshot && after_fault &&
            after_fault->second.at("owners") == before_snapshot->at("owners") &&
            after_fault->second.at("launch_count") == before_snapshot->at("launch_count") &&
            after_fault->second.at("ledger_size") == before_snapshot->at("ledger_size");
        std::fprintf(stderr, "KV_ADMISSION_TEST gate=C branch=%s expected=%s actual_exact=%d status=%d typed_json_refusal=%d owner_exact=%d launch_unchanged=%d ledger_unchanged=%d\n",
                fault, expected_reason, (int) exact_branch_reason, response ? response->status : 0, (int) typed_refusal,
                (int) fault_owner_preserved, (int) fault_owner_preserved, (int) fault_owner_preserved);
        fault_gates_ok = fault_gates_ok && typed_refusal && fault_owner_preserved;
    }
    unsetenv("LLAMA_KV_ADMISSION_TEST_FAULT");
    auto actor = actor_future.get();
    bool actor_ok = actor && actor->status == 200;
    int actor_tokens = 0;
    if (actor_ok) {
        try { actor_tokens = json::parse(actor->body).at("timings").value("predicted_n", 0); }
        catch (...) { actor_ok = false; }
        int accepted_native = 0;
        try { accepted_native = json::parse(actor->body).at("timings").value("draft_n_accepted", 0); }
        catch (...) { actor_ok = false; }
        actor_ok = actor_ok && actor_tokens == 200 && accepted_native >= 1;
        std::fprintf(stderr, "KV_ADMISSION_TEST gate=MTP-native-acceptance draft_n_accepted=%d required_min=1 pass=%d\n",
                accepted_native, (int)(accepted_native >= 1));
    }
    std::fprintf(stderr, "KV_ADMISSION_TEST driver=prior_actor_completed status=%d predicted_n=%d ok=%d\n",
            actor ? actor->status : 0, actor_tokens, (int) actor_ok);
    const char * omitted_hook_expected = "invalid exclusive owner/overflow";
    const bool omitted_hook_same_reason_assert = actor && actor->status == 503 &&
        actor->body.find(std::string("KV admission refused: ") + omitted_hook_expected) != std::string::npos;
    const bool omitted_hook_control = actor_ok && !omitted_hook_same_reason_assert;
    std::fprintf(stderr, "KV_ADMISSION_TEST gate=C-omit-hook branch=binding expected=%s actual=%s same_assert_result=%s pass=%d\n",
            omitted_hook_expected, actor_ok ? "accepted" : "not_accepted", omitted_hook_same_reason_assert ? "PASS_UNEXPECTED" : "FAIL_AS_REQUIRED", (int) omitted_hook_control);
    const bool control_ok = actor_ok; // first accepted native actor is the all-resources-fit completion control
    std::fprintf(stderr, "KV_ADMISSION_TEST driver=all_resources_fit_control status=%d predicted_n=%d ok=%d\n",
            actor ? actor->status : 0, actor_tokens, (int) control_ok);
    const auto after_actor_release = read_metrics_snapshot();
    const bool actor_release_empty = after_actor_release && after_actor_release->value("ledger_size", UINT64_MAX) == 0 &&
        after_actor_release->at("owners").empty() && !after_actor_release->value("release_failed", true) &&
        after_actor_release->value("launch_count", UINT64_MAX) == 1;
    std::fprintf(stderr, "KV_ADMISSION_TEST gate=MTP-release-after-completion empty_owners=%d ledger_size=%d launch_count=%" PRIu64 "\n",
            (int)actor_release_empty, after_actor_release ? (int)after_actor_release->value("ledger_size", UINT64_MAX) : -1,
            after_actor_release ? after_actor_release->value("launch_count", UINT64_MAX) : UINT64_MAX);
    const bool cleanup_ok = stop_and_cleanup_fixture();
    return actor_active && denied_ok && owner_preserved && fault_gates_ok && actor_ok && control_ok && actor_release_empty && cleanup_ok ? 0 : 7;
}
#endif // LLAMA_KV_ADMISSION_CPU_TEST_DRIVER

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

// satisfies -Wmissing-declarations (used by llama command)
int llama_server(int argc, char ** argv);

// to be used via CLI (argc / argv are used by router mode only)
int llama_server(common_params & params, int argc, char ** argv);
void llama_server_terminate();
void llama_server_terminate() {
    if (shutdown_handler) {
        shutdown_handler(0);
    }
}


// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

int llama_server(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

#ifndef _WIN32
    // Ignore SIGPIPE so the server does not crash if an MCP child exits while we are writing to its stdin
    signal(SIGPIPE, SIG_IGN);
#endif

    // own arguments required by this example
    common_params params;

    common_init();

    // start the stream session manager GC right after common init, before any HTTP route can
    // touch it. lifecycle is symmetric, stop_gc() runs in clean_up() before backend free
    server_stream_session_manager_start();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    return llama_server(params, argc, argv);
}

int llama_server(common_params & params, int argc, char ** argv) {
    bool is_run_by_cli = (argv == nullptr);

    common_models_handler models_handler;

    // note: router mode also accepts -hf remote-preset, so we need to check that first
    if (!is_run_by_cli && !params.model.hf_repo.empty()) {
        try {
            models_handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
            if (common_models_handler_is_preset_repo(models_handler)) {
                // apply the preset and start the server in router mode
                common_models_handler_apply(models_handler, params);
            }
        } catch (const std::exception & e) {
            SRV_ERR("failed to fetch model metadata: %s\n", e.what());
            return 1;
        }
    }

    // router server never loads a model and must not touch the GPU
    const bool is_router_server = params.model.path.empty()
                               && params.model.hf_repo.empty();

    // skip device enumeration so the CUDA primary context stays uncreated
    common_params_print_info(params, !is_router_server);

    if (!is_router_server) {
        // validate batch size for embeddings
        // embeddings require all tokens to be processed in a single ubatch
        // see https://github.com/ggml-org/llama.cpp/issues/12836
        if (params.embedding && params.n_batch > params.n_ubatch) {
            SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", params.n_batch, params.n_ubatch);
            SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", params.n_ubatch);
            params.n_batch = params.n_ubatch;
        }

        if (params.n_parallel < 0) {
            SRV_TRC("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

            params.n_parallel = 4;
            params.kv_unified = true;
        }
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    auto model_name = params.model.get_name();
    if (params.model_alias.empty() && !model_name.empty()) {
        params.model_alias.insert(model_name);
    }

    // note: this is guaranteed to out-live ctx_http and tools
    server_mcp mcp_mgr;

    // struct that contains llama context and inference
    server_context ctx_server;

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    //
    // Router
    //

    // register API routes
    server_child child; // only used in non-router mode
    server_routes routes(params, ctx_server);
    server_tools tools;

    std::optional<server_models_routes> models_routes{};
    if (is_router_server) {
        // setup server instances manager
        try {
            models_routes.emplace(params, argc, argv);
        } catch (const std::exception & e) {
            SRV_ERR("failed to initialize router models: %s\n", e.what());
            return 1;
        }

        // proxy handlers
        // note: routes.get_health stays the same
        routes.get_metrics                 = models_routes->proxy_get;
        routes.post_props                  = models_routes->proxy_post;
        routes.post_completions            = models_routes->proxy_post;
        routes.post_completions_oai        = models_routes->proxy_post;
        routes.post_chat_completions       = models_routes->proxy_post;
        routes.post_control                = models_routes->proxy_post;
        routes.post_responses_oai          = models_routes->proxy_post;
        routes.post_transcriptions_oai     = models_routes->proxy_post;
        routes.post_anthropic_messages     = models_routes->proxy_post;
        routes.post_anthropic_count_tokens = models_routes->proxy_post;
        routes.post_infill                 = models_routes->proxy_post;
        routes.post_embeddings             = models_routes->proxy_post;
        routes.post_embeddings_oai         = models_routes->proxy_post;
        routes.post_rerank                 = models_routes->proxy_post;
        routes.post_tokenize               = models_routes->proxy_post;
        routes.post_detokenize             = models_routes->proxy_post;
        routes.post_apply_template         = models_routes->proxy_post;
        routes.post_chat_completions_tok   = models_routes->proxy_post;
        routes.post_responses_tok_oai      = models_routes->proxy_post;
        routes.get_lora_adapters           = models_routes->proxy_get;
        routes.post_lora_adapters          = models_routes->proxy_post;
        routes.get_slots                   = models_routes->proxy_get;
        routes.post_slots                  = models_routes->proxy_post;

        // custom routes for router
        routes.get_props                   = models_routes->get_router_props;
        routes.get_models                  = models_routes->get_router_models;
        // Route slot-operation protocol calls to the exact model child selected by `model`.
        // These endpoints are registered through ctx_http below, so without these overrides
        // they would execute against the router's empty local registry.
        routes.get_slot_operations         = models_routes->proxy_get;
        routes.get_slot_operation          = models_routes->proxy_get;
        routes.post_slot_operation_ack     = models_routes->proxy_post;

        ctx_http.post("/models",               ex_wrapper(models_routes->post_router_models));
        ctx_http.post("/models/load",          ex_wrapper(models_routes->post_router_models_load));
        ctx_http.post("/models/unload",        ex_wrapper(models_routes->post_router_models_unload));
        ctx_http.get ("/models/sse",           ex_wrapper(models_routes->get_router_models_sse));
        ctx_http.del ("/models",               ex_wrapper(models_routes->del_router_models));
    }

    ctx_http.get ("/health",                   ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(routes.post_props));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions/control", ex_wrapper(routes.post_control));
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    // token counting
    ctx_http.post("/chat/completions/input_tokens",    ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/v1/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/responses/input_tokens",           ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/responses/input_tokens",        ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/messages/count_tokens",         ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(routes.post_slots));
    ctx_http.get ("/slot-operations",           ex_wrapper(routes.get_slot_operations));
    ctx_http.get ("/slot-operations/:op_id",    ex_wrapper(routes.get_slot_operation));
    ctx_http.post("/slot-operations/:op_id/ack", ex_wrapper(routes.post_slot_operation_ack));

    // resumable streaming: a child binds the local session factories, the router binds
    // proxies that resolve the owning child, see server-stream.h
    server_http_context::handler_t stream_get_h;
    server_http_context::handler_t streams_lookup_h;
    server_http_context::handler_t stream_delete_h;
    if (is_router_server) {
        stream_get_h     = models_routes->router_stream_get;
        streams_lookup_h = models_routes->router_streams_lookup;
        stream_delete_h  = models_routes->router_stream_delete;
    } else {
        stream_get_h     = server_stream_make_get_handler();
        streams_lookup_h = server_stream_make_lookup_handler();
        stream_delete_h  = server_stream_make_delete_handler();
    }
    ctx_http.get ("/v1/stream",                ex_wrapper(stream_get_h));
    ctx_http.post("/v1/streams/lookup",        ex_wrapper(streams_lookup_h));
    ctx_http.del ("/v1/stream",                ex_wrapper(stream_delete_h));

    // Google Cloud Platform (Vertex AI) compat
    ctx_http.register_gcp_compat();

    // return 403 for disabled features
    server_http_context::handler_t res_403 = [](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->status = 403;
        res->data = safe_json_to_str({
            {"error", {
                {"message", "this feature is disabled"},
                {"type", "feature_disabled"},
            }}
        });
        return res;
    };

    if (params.cors_origins == "*" && params.api_keys.empty()) {
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "CORS is set to allow all origins ('*') and no API key is set\n");
        SRV_WRN("%s", "this can be a security risk (cross-origin attacks)\n");
        SRV_WRN("%s", "more info: https://github.com/ggml-org/llama.cpp/pull/25655\n");
        SRV_WRN("%s", "-----------------\n");
    }

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    std::vector<std::string> warn_names;
    if (is_router_server) {
        warn_names.push_back("router mode");
    }

    if (params.ui_mcp_proxy) {
        ctx_http.get ("/cors-proxy",      ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy",      ex_wrapper(proxy_handler_post));
        warn_names.push_back("MCP proxy (experimental)");
    } else {
        ctx_http.get ("/cors-proxy",      ex_wrapper(res_403));
        ctx_http.post("/cors-proxy",      ex_wrapper(res_403));
    }

    try {
        mcp_mgr.start(params);
    } catch (const std::exception & e) {
        SRV_ERR("MCP starting failed: %s\n", e.what());
        return 1;
    }

    if (!params.server_tools.empty() || !mcp_mgr.empty()) {
        try {
            tools.setup(params.server_tools, mcp_mgr);
        } catch (const std::exception & e) {
            SRV_ERR("tools setup failed: %s\n", e.what());
            return 1;
        }
        ctx_http.get ("/tools",           ex_wrapper(tools.handle_get));
        ctx_http.post("/tools",           ex_wrapper(tools.handle_post));
        if (!params.server_tools.empty()) {
            warn_names.push_back("built-in tools (experimental)");
        }
        if (!mcp_mgr.empty()) {
            warn_names.push_back("MCP servers (experimental)");
        }
    } else {
        ctx_http.get ("/tools",           ex_wrapper(res_403));
        ctx_http.post("/tools",           ex_wrapper(res_403));
    }

    if (warn_names.size() > 0) {
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "the following feature(s) are enabled:\n");
        for (const auto & name : warn_names) {
            SRV_WRN("    %s\n", name.c_str());
        }
        SRV_WRN("%s", "do not expose the server to untrusted environments\n");
        SRV_WRN("%s", "-----------------\n");
    }

    //
    // Handle downloading model
    //

    if (child.is_child() && child.get_mode() == SERVER_CHILD_MODE_DOWNLOAD) {
        return child.run_download(params);
    } else if (!is_router_server && !is_run_by_cli) {
        // single-model mode (NOT spawned by router)
        // if this is invoked by CLI, model downloading should be already handled
        try {
            common_models_handler_apply(models_handler, params);
        } catch (const std::exception & e) {
            SRV_ERR("failed to download model: %s\n", e.what());
            return 1;
        }
    }

    //
    // Start the server
    //

    std::function<void()> clean_up;

    if (is_router_server) {
        SRV_INF("%s", "starting server in router mode. models will be automatically loaded on-demand\n");

        clean_up = [&models_routes, &mcp_mgr]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            // stop the session GC first, it finalizes live sessions and wakes pending readers
            server_stream_session_manager_stop();
            if (models_routes.has_value()) {
                models_routes->stopping.store(true); // maybe redundant, but just to be safe
                models_routes->models.unload_all();
            }
            mcp_mgr.shutdown();
            llama_backend_free();
        };

        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }
        ctx_http.is_ready.store(true);

        shutdown_handler = [&](int) {
            if (models_routes.has_value()) {
                // important to disconnect any SSE clients
                models_routes->stopping.store(true);
            }
            mcp_mgr.shutdown();
            ctx_http.stop();
        };

    } else {
        // setup clean up function, to be called before exit
        clean_up = [&ctx_http, &ctx_server, &mcp_mgr]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            // stop the session GC first, it finalizes live sessions and wakes pending readers
            server_stream_session_manager_stop();
            ctx_http.stop();
            ctx_server.terminate();
            mcp_mgr.shutdown();
            llama_backend_free();
        };

        // start the HTTP server before loading the model to be able to serve /health requests
        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }

        // setup communication child --> router if necessary
        if (child.is_child()) {
            ctx_server.set_state_callback([&](server_state state, json payload) {
                child.notify_to_router(server_state_to_str(state), payload);
            });
        }

        if (!ctx_server.load_model(params)) {
            clean_up();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
            SRV_ERR("%s", "exiting due to model loading error\n");
            return 1;
        }

        routes.update_meta(ctx_server);
        ctx_http.is_ready.store(true);

        SRV_INF("%s", "model loaded\n");

        shutdown_handler = [&](int) {
            mcp_mgr.shutdown();
            // this will unblock start_loop()
            ctx_server.terminate();
        };
    }

    // register signal handler if not running by CLI
    if (!is_run_by_cli) {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = signal_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
        sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    SRV_INF("listening on %s\n", ctx_http.listening_address.c_str());

    if (is_router_server) {
        if (!params.models_preset_hf.empty()) {
            SRV_WRN(      "NOTE: using preset.ini from HF repo '%s'\n", params.models_preset_hf.c_str());
            SRV_WRN("%s", "      please only use presets that you can trust! Unknown presets may be unsafe\n");
        }

        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join(); // keep the main thread alive
        }

        // when the HTTP server stops, clean up and exit
        clean_up();
    } else {
        // optionally, notify router server that this instance is ready
        std::thread monitor_thread;
        if (child.is_child()) {
            monitor_thread = child.setup(shutdown_handler);
            child.notify_to_router(server_state_to_str(SERVER_STATE_READY), routes.get_model_info());
        }

#ifdef LLAMA_KV_ADMISSION_CPU_TEST_DRIVER
        // The isolated CPU test driver calls the same dispatcher and public HTTP routes.
        const char * test_driver = kv_admission_cpu_test_env("LLAMA_KV_ADMISSION_TEST_DRIVER");
        const char * pair_arm = kv_admission_cpu_test_env("LLAMA_KV_ADMISSION_TEST_PAIR");
        const char * c_scenario = kv_admission_cpu_test_env("LLAMA_KV_ADMISSION_C_SCENARIO");
        const int test_driver_status = pair_arm
            ? run_kv_admission_pair_driver(ctx_http.port, ctx_server, pair_arm)
            : c_scenario
            ? run_kv_admission_c_scenario(ctx_http.port, ctx_server, c_scenario)
            : (test_driver ? run_kv_admission_test_driver(ctx_http.port, ctx_server)
            : (ctx_server.start_loop(), 0));
#else
        ctx_server.start_loop();
#endif

        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }

#ifdef LLAMA_KV_ADMISSION_CPU_TEST_DRIVER
        if (test_driver || pair_arm || c_scenario) return test_driver_status;
#endif

        auto * ll_ctx = ctx_server.get_llama_context();
        if (ll_ctx != nullptr) {
            common_memory_breakdown_print(ll_ctx);
        }
    }

    return 0;
}
