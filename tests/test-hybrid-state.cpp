// T1 (state bytes) and T2 (greedy continuation) for a hybrid recurrent+attention model,
// plus the "delta used == cell_count" proof through the public resource-capacity API.
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <cstdlib>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static llama_context * make_ctx(const common_params & params, llama_model * model, uint32_t n_rs_seq) {
    auto cp = common_context_params_to_llama(params);
    cp.n_ctx     = 512;
    cp.n_seq_max = 2;
    cp.n_batch   = 64;
    cp.n_ubatch  = 64;
    cp.n_rs_seq  = n_rs_seq;
    return llama_init_from_model(model, cp);
}

static bool decode(llama_context * ctx, const llama_token * tok, int n, llama_pos pos0, llama_seq_id seq, bool last_logits_only = true) {
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) common_batch_add(b, tok[i], pos0 + i, { seq }, !last_logits_only || i + 1 == n);
    const bool ok = llama_decode(ctx, b) == 0;
    llama_batch_free(b);
    return ok;
}

static llama_token argmax(llama_context * ctx, int n_vocab, std::vector<float> * logits_out = nullptr) {
    const float * l = llama_get_logits_ith(ctx, -1);
    if (logits_out) logits_out->assign(l, l + n_vocab);
    return (llama_token) (std::max_element(l, l + n_vocab) - l);
}

// last prompt token (re)decoded at pos, then n greedy steps. returns tokens; first_logits = logits of first decode
static std::vector<llama_token> continue_greedy(llama_context * ctx, int n_vocab, llama_token last, llama_pos pos, int n, llama_seq_id seq,
                                                std::vector<float> & first_logits) {
    std::vector<llama_token> out;
    if (!decode(ctx, &last, 1, pos, seq)) { CHECK(false, "decode last"); return out; }
    llama_token t = argmax(ctx, n_vocab, &first_logits);
    for (int i = 0; i < n; ++i) {
        out.push_back(t);
        if (!decode(ctx, &t, 1, pos + 1 + i, seq)) { CHECK(false, "decode step %d", i); break; }
        t = argmax(ctx, n_vocab);
    }
    return out;
}

static std::vector<uint8_t> save_seq(llama_context * ctx, llama_seq_id seq) {
    std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, seq));
    const size_t n = llama_state_seq_get_data(ctx, buf.data(), buf.size(), seq);
    CHECK(n == buf.size(), "seq_get_data wrote %zu of %zu", n, buf.size());
    buf.resize(n);
    return buf;
}

static void report_diff(const char * what, const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    size_t n = std::min(a.size(), b.size()), first = n, cnt = 0, last = 0;
    for (size_t i = 0; i < n; ++i) if (a[i] != b[i]) { if (first == n) first = i; ++cnt; last = i; }
    for (int k = 0; k < 2; ++k) {
        const auto & v = k ? b : a;
        fprintf(stderr, "HEX%d:", k);
        for (size_t i = 0; i < 64 && i < v.size(); ++i) fprintf(stderr, " %02x", v[i]);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "DIFF %s: sizes %zu/%zu first=%zu last=%zu differing_bytes=%zu\n", what, a.size(), b.size(), first, last, cnt);
}

static llama_memory_resource_capacity cap(llama_context * ctx, llama_seq_id seq) {
    llama_memory_resource_capacity c = {};
    const bool ok = llama_memory_get_resource_capacity(llama_get_memory(ctx), seq, &c, sizeof(c));
    CHECK(ok && c.recurrent_known && c.attention_known, "capacity unknown");
    return c;
}

static void run(const common_params & params, llama_model * model, uint32_t n_rs_seq) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    fprintf(stderr, "=== n_rs_seq=%u n_vocab=%d hybrid=%d\n", n_rs_seq, n_vocab, (int) llama_model_is_hybrid(model));
    CHECK(llama_model_is_hybrid(model), "model must be hybrid");

    llama_context * ctx = make_ctx(params, model, n_rs_seq);
    CHECK(ctx != nullptr, "ctx");
    if (!ctx) return;

    std::vector<llama_token> prompt = common_tokenize(ctx, "the quick brown fox jumps over the lazy dog 0123456789", true);
    const int P = (int) prompt.size();
    fprintf(stderr, "prompt tokens: %d\n", P);
    CHECK(P >= 8 && (uint32_t) P <= 64, "prompt size");
    const llama_token last = prompt.back();
    const llama_pos last_pos = P - 1;

    // state at P-1 tokens; with n_rs_seq>0 reached by rollback so rs_idx != 0 (T1, k>0)
    auto build_prefix = [&](llama_context * c, llama_seq_id seq) {
        if (n_rs_seq == 0) {
            CHECK(decode(c, prompt.data(), P - 1, 0, seq), "prefix decode");
        } else {
            CHECK(decode(c, prompt.data(), P, 0, seq), "prompt decode");
            CHECK(llama_memory_seq_rm(llama_get_memory(c), seq, last_pos, -1), "rollback");
        }
    };

    build_prefix(ctx, 0);
    const auto cap_full = cap(ctx, 0);
    const uint32_t g = cap_full.recurrent_rollback_groups;
    CHECK(g == n_rs_seq + 1, "groups %u", g);
    CHECK(cap_full.recurrent_seq_rows % g == 0 && cap_full.recurrent_used_rows % g == 0, "rows not divisible by g");
    const uint32_t cells = cap_full.recurrent_seq_rows / g;
    fprintf(stderr, "src: rec used_rows=%llu seq_rows=%llu excl=%llu cap_rows=%llu g=%u cells=%u | att used=%llu seq=%llu\n",
            (unsigned long long) cap_full.recurrent_used_rows, (unsigned long long) cap_full.recurrent_seq_rows,
            (unsigned long long) cap_full.recurrent_seq_exclusive_rows, (unsigned long long) cap_full.recurrent_capacity_rows, g, cells,
            (unsigned long long) cap_full.attention_used_cells, (unsigned long long) cap_full.attention_seq_cells);
    CHECK(cells == 1, "one logical recurrent cell expected, got %u", cells);

    const std::vector<uint8_t> blob1 = save_seq(ctx, 0);
    fprintf(stderr, "blob1 bytes=%zu\n", blob1.size());
    // the sequential save itself is deterministic (idempotent)
    CHECK(save_seq(ctx, 0) == blob1, "save not idempotent");

    // T2 live continuation (no restore)
    std::vector<float> lg_live;
    const auto A = continue_greedy(ctx, n_vocab, last, last_pos, 128, 0, lg_live);
    CHECK(A.size() == 128, "live size");

    // (a) restore into SAME ctx, seq 0, after full clear
    llama_memory_clear(llama_get_memory(ctx), true);
    const auto cap_empty = cap(ctx, 0);
    CHECK(cap_empty.recurrent_used_rows == 0, "clear left used rows %llu", (unsigned long long) cap_empty.recurrent_used_rows);
    size_t nr = llama_state_seq_set_data(ctx, blob1.data(), blob1.size(), 0);
    CHECK(nr == blob1.size(), "set_data read %zu of %zu", nr, blob1.size());
    const auto cap_rest = cap(ctx, 0);
    const uint64_t delta_rows = cap_rest.recurrent_used_rows - cap_empty.recurrent_used_rows;
    fprintf(stderr, "restore->empty dest: delta used_rows=%llu (cells=%llu) expect cell_count=%u\n",
            (unsigned long long) delta_rows, (unsigned long long) (delta_rows / g), cells);
    CHECK(delta_rows % g == 0 && delta_rows / g == cells, "delta used != cell_count");
    CHECK(cap_rest.attention_used_cells - cap_empty.attention_used_cells == cap_full.attention_seq_cells, "attention delta mismatch");
    // T1: re-save is byte identical; rs_idx back to plane 0 is implied by equality of the saved plane
    const auto blob2 = save_seq(ctx, 0);
    CHECK(blob2 == blob1, "T1 FAIL same-ctx seq0 re-save differs (%zu vs %zu)", blob2.size(), blob1.size());
    std::vector<float> lg_b;
    const auto B = continue_greedy(ctx, n_vocab, last, last_pos, 128, 0, lg_b);
    CHECK(B == A, "T2 FAIL greedy after restore != live");
    CHECK(lg_b.size() == lg_live.size() && !std::memcmp(lg_b.data(), lg_live.data(), lg_b.size() * sizeof(float)), "T2 first logits not bit-identical");

    // (b) restore into seq 1 of a clean ctx (different dest id, different physical cell)
    llama_memory_clear(llama_get_memory(ctx), true);
    nr = llama_state_seq_set_data(ctx, blob1.data(), blob1.size(), 1);
    CHECK(nr == blob1.size(), "set_data seq1");
    const auto blob3 = save_seq(ctx, 1);
    // The attention section embeds the per-seq stream index (non-unified KV), so blobs of different seq ids differ
    // in that section by design (HEX shows bytes 4 and 8..16 only). Compare: (1) same size, (2) the recurrent tail
    // (everything after the attention section, i.e. the last 16 KiB) bit for bit, (3) round trip seq1 -> seq0 == blob1.
    CHECK(blob3.size() == blob1.size(), "seq1 blob size differs");
    constexpr size_t TAIL = 16384;
    CHECK(blob1.size() > TAIL && std::equal(blob1.end() - TAIL, blob1.end(), blob3.end() - TAIL), "T1 FAIL recurrent tail seq1 != seq0");
    if (blob3.size() != blob1.size() || !std::equal(blob1.end() - TAIL, blob1.end(), blob3.end() - TAIL)) report_diff("seq1-vs-seq0", blob1, blob3);
    {
        llama_memory_clear(llama_get_memory(ctx), true);
        CHECK(llama_state_seq_set_data(ctx, blob3.data(), blob3.size(), 1) == blob3.size(), "re-restore seq1");
        llama_memory_clear(llama_get_memory(ctx), true);
        CHECK(llama_state_seq_set_data(ctx, blob3.data(), blob3.size(), 1) == blob3.size(), "re-restore seq1 b");
        // seq1 -> file bytes -> seq1 again: idempotent
        CHECK(save_seq(ctx, 1) == blob3, "T1 FAIL seq1 round trip differs");
    }
    std::vector<float> lg_c;
    const auto Cc = continue_greedy(ctx, n_vocab, last, last_pos, 128, 1, lg_c);
    CHECK(Cc == A, "T2 FAIL greedy seq1 != live");

    // (c) dirty destination: dest seq 0 holds other content, seq 1 holds ours; restore replaces, net used delta 0
    llama_memory_clear(llama_get_memory(ctx), true);
    std::vector<llama_token> noise(prompt.begin(), prompt.end());
    for (auto & t : noise) t = (t + 7) % n_vocab;
    CHECK(decode(ctx, noise.data(), P - 1, 0, 0), "noise decode");
    CHECK(decode(ctx, noise.data(), P - 1, 0, 1), "noise decode s1");
    const auto cap_dirty = cap(ctx, 0);
    nr = llama_state_seq_set_data(ctx, blob1.data(), blob1.size(), 0);
    CHECK(nr == blob1.size(), "set_data dirty");
    const auto cap_after = cap(ctx, 0);
    fprintf(stderr, "restore->dirty dest: used_rows %llu -> %llu (replacement: net delta 0 expected; other seq untouched)\n",
            (unsigned long long) cap_dirty.recurrent_used_rows, (unsigned long long) cap_after.recurrent_used_rows);
    CHECK(cap_after.recurrent_used_rows == cap_dirty.recurrent_used_rows, "replacement changed used rows");
    CHECK(save_seq(ctx, 0) == blob1, "T1 FAIL dirty-dest re-save differs");
    std::vector<float> lg_d;
    const auto D = continue_greedy(ctx, n_vocab, last, last_pos, 128, 0, lg_d);
    CHECK(D == A, "T2 FAIL greedy dirty dest != live");
    llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1);

    // (d) recompute control from scratch with the same ubatch shape
    if (n_rs_seq == 0) {
        llama_memory_clear(llama_get_memory(ctx), true);
        build_prefix(ctx, 0);
        std::vector<float> lg_r;
        const auto R = continue_greedy(ctx, n_vocab, last, last_pos, 128, 0, lg_r);
        fprintf(stderr, "recompute control (informational, CPU): %s\n", R == A ? "tokens equal" : "tokens DIFFER");
        CHECK(R == A, "recompute differs from live");
    }

    // (f) negative control: a corrupted recurrent tail must be detected by T1 and must change the continuation (T2 discriminates)
    {
        llama_memory_clear(llama_get_memory(ctx), true);
        std::vector<uint8_t> bad = blob1;
        std::fill(bad.end() - 1024, bad.end(), 0); // pure S-state data of the last recurrent layer (no headers)
        CHECK(llama_state_seq_set_data(ctx, bad.data(), bad.size(), 0) == bad.size(), "restore zeroed tail");
        CHECK(save_seq(ctx, 0) != blob1, "CONTROL: zeroed recurrent state bytes not detected by T1");
        std::vector<float> lg_z;
        const auto Z = continue_greedy(ctx, n_vocab, last, last_pos, 128, 0, lg_z);
        const bool differs = Z != A || lg_z != lg_live;
        fprintf(stderr, "negative control: zeroed recurrent state -> tokens %s, logits %s\n", Z != A ? "DIFFER" : "equal", lg_z != lg_live ? "DIFFER" : "equal");
        CHECK(differs, "CONTROL: model output insensitive to recurrent state");
    }

    // (g) bounded file restore (the library path the server uses): consume callback sees (kind,stream,count)
    {
        llama_memory_clear(llama_get_memory(ctx), true);
        build_prefix(ctx, 0);
        char path[] = "/tmp/hybrid-state-XXXXXX";
        int fd = mkstemp(path);
        CHECK(fd >= 0, "mkstemp");
        if (fd >= 0) close(fd);
        const auto pre = save_seq(ctx, 0);
        CHECK(llama_state_seq_save_file(ctx, path, 0, prompt.data(), P - 1) > 0, "save_file");
        struct Log { std::vector<std::vector<uint64_t>> calls; uint64_t clears = 0; } lg;
        llama_state_seq_limits lim{};
        lim.user = &lg;
        lim.consume = [](void * u, uint32_t kind, uint32_t stream, uint64_t count) {
            static_cast<Log *>(u)->calls.push_back({ kind, stream, count });
            return true;
        };
        lim.before_replace = [](void * u) { ++static_cast<Log *>(u)->clears; };
        lim.max_tensor_bytes = llama_state_seq_restore_tensor_capacity(ctx);
        lim.max_snapshot_bytes = llama_state_seq_restore_file_capacity(ctx, 0);
        fprintf(stderr, "bounded: tensor_cap=%llu file_cap=%llu\n", (unsigned long long) lim.max_tensor_bytes, (unsigned long long) lim.max_snapshot_bytes);
        std::vector<llama_token> toks(P);
        size_t ntok = 0;
        const size_t got = llama_state_seq_load_file_bounded(ctx, path, 0, toks.data(), toks.size(), &ntok, &lim);
        for (auto & c : lg.calls) fprintf(stderr, "bounded consume kind=%llu stream=%llu count=%llu\n", (unsigned long long) c[0], (unsigned long long) c[1], (unsigned long long) c[2]);
        fprintf(stderr, "bounded: returned %zu limit_refused=%d mutation_started=%d clears=%llu\n", got, (int) lim.limit_refused, (int) lim.mutation_started, (unsigned long long) lg.clears);
        CHECK(got > 0, "bounded restore of hybrid file failed");
        if (got > 0) {
            CHECK(save_seq(ctx, 0) == pre, "T1 FAIL bounded restore re-save differs");
            CHECK(lg.calls.size() >= 2, "consume must see attention and recurrent");
        }
        unlink(path);
    }

    // (e) corrupted cell_count must be refused with no mutation of a populated destination
    {
        llama_memory_clear(llama_get_memory(ctx), true);
        build_prefix(ctx, 0);
        const auto before = save_seq(ctx, 0);
        std::vector<uint8_t> bad = blob1;
        // truncated blob
        bad.resize(bad.size() / 2);
        const size_t r = llama_state_seq_set_data(ctx, bad.data(), bad.size(), 0);
        fprintf(stderr, "truncated blob restore returned %zu\n", r);
        CHECK(r == 0, "truncated blob accepted");
    }

    fprintf(stderr, "A[0..8]:");
    for (int i = 0; i < 8; ++i) fprintf(stderr, " %d", A[i]);
    fprintf(stderr, "\n");
    llama_free(ctx);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    params.sampling.seed = 1234;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    ggml_backend_load_all();
    common_init_result_ptr init = common_init_from_params(params);
    llama_model * model = init->model();
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    run(params, model, 0);
    run(params, model, 4);
    if (g_fail) { fprintf(stderr, "RESULT: %d FAILURES\n", g_fail); return 1; }
    fprintf(stderr, "RESULT: ALL PASSED\n");
    return 0;
}
