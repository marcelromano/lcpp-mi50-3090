// hybrid (attention + recurrent) restore under admission: (kind,stream) bridge, declared layout,
// D1 replacement credit, fail-closed refusals and T4 injected failures (synthetic hybrid memory, CPU, no gguf).
#include "ggml-backend.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid.h"
#include "llama-model.h"
#include "llama-io.h"
#include "../tools/server/kv_planner.hpp"
#include "../tools/server/kv_restore_map.hpp"
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace kvplan;
static int g_fail = 0;
#define EXPECT(x) do { if (!(x)) { ++g_fail; std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); } } while (0)
struct fixture_model : llama_model {
    fixture_model() : llama_model(llama_model_params{}) {}
    void load_stats(llama_model_loader &) override {} void load_hparams(llama_model_loader &) override {}
    void load_vocab(llama_model_loader &) override {} bool load_tensors(llama_model_loader &) override { return false; }
    void load_arch_hparams(llama_model_loader &) override {} void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override { return {}; }
};
struct Writer : llama_io_write_i {
    std::vector<uint8_t> bytes;
    void write(const void * p, size_t n) override { auto v = (const uint8_t *) p; bytes.insert(bytes.end(), v, v + n); }
    void write_tensor(ggml_tensor * t, size_t offset, size_t n) override { size_t old = bytes.size(); bytes.resize(old + n); ggml_backend_tensor_get(t, bytes.data() + old, offset, n); }
    size_t n_bytes() override { return bytes.size(); }
};
static std::vector<uint8_t> dump(llama_memory_i & m, int seq) { Writer w; m.state_write(w, seq, 0); return w.bytes; }
struct Reader : llama_io_read_i {
    const std::vector<uint8_t> & bytes; size_t position = 0; uint64_t cumulative = 0;
    Reader(const std::vector<uint8_t> & b, llama_state_seq_limits * l) : llama_io_read_i(l), bytes(b) {}
    void read(void * p, size_t n) override { if (n > bytes.size() - position) throw std::runtime_error("EOF"); std::memcpy(p, bytes.data() + position, n); position += n; }
    void read_tensor(ggml_tensor * t, size_t offset, size_t n) override {
        if (n > bytes.size() - position || n > limits->max_tensor_bytes || offset > ggml_nbytes(t) || n > ggml_nbytes(t) - offset) throw std::runtime_error("tensor EOF/bounds");
        cumulative += n;
        if (limits->before_tensor_write && !limits->before_tensor_write(limits->user, cumulative)) throw std::runtime_error("injected tensor fault");
        ggml_backend_tensor_set(t, bytes.data() + position, offset, n); position += n;
    }
    size_t n_bytes() override { return position; }
};

// Same wiring as the server: RestoreKeyMap + restore_consume + (before_replace counts clears).
struct Authority {
    Planner * planner; RestoreToken * token; RestoreKeyMap map; uint64_t unmapped = 0;
    int fail_rec_write = 0;   // inject: fail the Nth recurrent tensor write (1-based), 0 = never
    bool in_rec = false; int rec_writes = 0;
    static bool consume(void * u, uint32_t kind, uint32_t stream, uint64_t count) {
        auto & a = *static_cast<Authority *>(u);
        if (kind == 1) a.in_rec = true;
        return restore_consume(*a.planner, *a.token, a.map, kind, stream, count, a.unmapped);
    }
    static bool before_write(void * u, uint64_t) {
        auto & a = *static_cast<Authority *>(u);
        if (!a.in_rec || !a.fail_rec_write) return true;
        return ++a.rec_writes != a.fail_rec_write;
    }
};

struct Hyb {
    fixture_model model; std::unique_ptr<llama_memory_hybrid> mem;
    llama_memory_recurrent * rec() { return mem->get_mem_recr(); }
    llama_kv_cache * att() { return mem->get_mem_attn(); }
    Hyb(int attn_seq0, int attn_seq1, int pattern, bool seed_seq1 = true) {
        model.hparams.n_layer_all = 3; model.hparams.n_embd = 2;
        model.hparams.n_head_arr.fill(1); model.hparams.n_head_kv_arr.fill(1);
        model.hparams.n_embd_head_k_full = 2; model.hparams.n_embd_head_v_full = 2;
        model.hparams.ssm_d_conv = 2; model.hparams.ssm_d_inner = 1; model.hparams.ssm_n_group = 1; model.hparams.ssm_d_state = 1;
        mem = std::make_unique<llama_memory_hybrid>(model, GGML_TYPE_F32, GGML_TYPE_F32, false, 16, 1, 0, LLAMA_SWA_TYPE_NONE,
            GGML_TYPE_F32, GGML_TYPE_F32, 2, 2, 1, false, true, [](int32_t i) { return i == 0; }, [](int32_t i) { return i >= 1; });
        for (int seq = 0; seq < 2; ++seq) {
            const int n = seq ? attn_seq1 : attn_seq0;
            if (!n) continue;
            llama_batch_allocr allocator(1); auto batch = allocator.ubatch_reserve(n, 1); batch.seq_id_unq[0] = seq;
            for (int i = 0; i < n; ++i) { batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i] = &seq; }
            auto slot = att()->find_slot(batch, false); att()->apply_ubatch(slot, batch);
        }
        auto & m = *rec();
        for (int seq = 0; seq < (seed_seq1 ? 2 : 1); ++seq) { m.cells[seq].pos = 10 + seq; m.cells[seq].seq_id.insert(seq); m.cells[seq].tail = seq; }
        m.used = seed_seq1 ? 2 : 1; m.rs_idx[0] = 0;
        for (auto * t : m.r_l) if (t) { std::vector<float> v(ggml_nbytes(t) / sizeof(float)); for (size_t i = 0; i < v.size(); ++i) v[i] = float(pattern + i); ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t)); }
        for (auto * t : m.s_l) if (t) { std::vector<float> v(ggml_nbytes(t) / sizeof(float)); for (size_t i = 0; i < v.size(); ++i) v[i] = float(pattern + 100 + i); ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t)); }
    }
    std::vector<PhysicalView> view(int seq, uint64_t epoch = 7) {
        llama_memory_resource_capacity raw{}; std::vector<PhysicalView> out;
        if (!mem->get_resource_capacity(seq, raw) || !normalize_resources(Context::target, epoch, raw, out)) out.clear();
        return out;
    }
};
static const PhysicalView * pick(const std::vector<PhysicalView> & v, Kind k) { for (auto & p : v) if (p.key.kind == k) return &p; return nullptr; }
static std::vector<Snapshot> snaps_of(const std::vector<PhysicalView> & v) {
    std::vector<Snapshot> out;
    for (auto & p : v) { Snapshot s{}; s.key = p.key; s.known = true; s.capacity = p.capacity; s.used_real = p.used; out.push_back(s); }
    return out;
}
static std::vector<ResourceKey> keys_of(const std::vector<PhysicalView> & v) { std::vector<ResourceKey> k; for (auto & p : v) k.push_back(p.key); return k; }

enum class Mode { none, fail_rec_first, fail_rec_second_layer };
struct CycleOut { bool restored = false; bool limit_refused = false; uint64_t unmapped = 0, clears = 0; };

// One full restore cycle exactly like the server: snapshots + D1 credit + declared plan, bounded read,
// on failure the destination is cleared (seq_rm), fence + native release.
static CycleOut cycle(Planner & planner, Hyb & dest, const std::vector<uint8_t> & blob, uint64_t task, Mode mode, bool drop_rec_key = false) {
    CycleOut out;
    auto view = dest.view(0);
    auto snaps = snaps_of(view);
    Owner owner{7, task, 0}; std::string why;
    EXPECT(apply_restore_credit(view, owner, snaps, why));
    auto plan = planner.plan_restore(snaps, owner, keys_of(view));
    EXPECT(plan.kind == RestorePlan::Kind::ok);
    if (plan.kind != RestorePlan::Kind::ok) return out;
    Authority a{&planner, &plan.token, {}};
    for (auto & k : keys_of(view)) if (!(drop_rec_key && k.kind == Kind::recurrent_rows)) a.map.add(k);
    a.fail_rec_write = mode == Mode::fail_rec_first ? 1 : mode == Mode::fail_rec_second_layer ? 3 : 0;
    llama_state_seq_limits limits{}; limits.user = &a; limits.consume = Authority::consume; limits.before_tensor_write = Authority::before_write;
    for (const auto & b : dest.mem->memory_breakdown()) limits.max_tensor_bytes += b.second;
    try { Reader r(blob, &limits); dest.mem->state_read(r, 0, 0); out.restored = true; }
    catch (const std::exception &) { out.restored = false; }
    out.limit_refused = limits.limit_refused; out.unmapped = a.unmapped; out.clears = limits.clear_count;
    if (!out.restored && limits.mutation_started) dest.mem->seq_rm(0, -1, -1); // slot->clear_after_restore_failure
    EXPECT(planner.fence_restore(plan.token));
    EXPECT(planner.native_release(owner).status == Status::accepted);
    return out;
}

static int run(bool inject) {
    ResourceKey at{Context::target, 0, 7, Kind::attention_cells}, rc{Context::target, 0, 7, Kind::recurrent_rows};
    Hyb source(8, 0, 10, false);
    const auto blob = dump(*source.mem, 0);

    // --- limits per key + D1 credit (pool full, destination owns 1 exclusive cell) --------------------------
    {
        Hyb d(3, 2, 200); auto v = d.view(0);
        auto * r = pick(v, Kind::recurrent_rows); auto * a = pick(v, Kind::attention_cells);
        EXPECT(r && a && r->used == r->capacity && r->exclusive == 1 && r->sequence == 1);
        Planner p; auto snaps = snaps_of(v); Owner o{7, 1, 0}; std::string why;
        EXPECT(apply_restore_credit(v, o, snaps, why));
        auto plan = p.plan_restore(snaps, o, keys_of(v));
        EXPECT(plan.kind == RestorePlan::Kind::ok);
        EXPECT(plan.token.limit(rc) == 1);                                   // credit = destination's own cell, never more
        EXPECT(plan.token.limit(at) == a->capacity - a->used);               // attention takes no credit
        EXPECT(!p.consume_restore(plan.token, rc, 2));                       // limit+1 over the credit refused
        EXPECT(p.consume_restore(plan.token, rc, 1) && !p.consume_restore(plan.token, rc, 1));
        EXPECT(!p.consume_restore(plan.token, ResourceKey{Context::target, 0, 7, Kind::recurrent_rows}, 1));
        EXPECT(p.fence_restore(plan.token) && p.native_release(o).status == Status::accepted && p.size() == 0);
        // without the credit the same full pool refuses (G2 zero credit)
        Planner p2; auto s2 = snaps_of(v);
        EXPECT(p2.plan_restore(s2, o, keys_of(v)).kind == RestorePlan::Kind::refused_overcommit && p2.size() == 0);
    }
    // --- shared recurrent sequence (n_seq_id>1) refused, nothing changes -------------------------------------
    {
        Hyb d(3, 2, 200); auto & rc0 = *d.rec(); rc0.cells[1].pos = -1; rc0.cells[1].seq_id.clear(); rc0.cells[0].seq_id.insert(1); rc0.cells[1].tail = 0; rc0.used = 1;
        auto v = d.view(0); EXPECT(v.size() == 2); auto snaps = snaps_of(v); std::string why; Owner o{7, 1, 0};
        EXPECT(!apply_restore_credit(v, o, snaps, why) && why == "shared_recurrent_sequence");
    }
    // --- unknown layout: zero clear ---------------------------------------------------------------------------
    {
        Hyb d(3, 2, 200); auto v = d.view(0); Owner o{7, 1, 0};
        { Planner p; auto s = snaps_of(v); s.erase(std::remove_if(s.begin(), s.end(), [](const Snapshot & x) { return x.key.kind == Kind::recurrent_rows; }), s.end());
          EXPECT(p.plan_restore(s, o, keys_of(v)).kind == RestorePlan::Kind::refused_unsupported_layout && p.size() == 0); } // missing key
        { Planner p; auto s = snaps_of(v); Snapshot x = s.back(); x.key.context = Context::draft; s.push_back(x);
          EXPECT(p.plan_restore(s, o, keys_of(v)).kind == RestorePlan::Kind::refused_unsupported_layout && p.size() == 0); } // draft recurrent
        { Planner p; auto s = snaps_of(v); for (auto & x : s) if (x.key.kind == Kind::recurrent_rows) x.key.stream = 1;
          auto k = keys_of(v); for (auto & y : k) if (y.kind == Kind::recurrent_rows) y.stream = 1;
          EXPECT(p.plan_restore(s, o, k).kind == RestorePlan::Kind::refused_unsupported_layout && p.size() == 0); }           // stream != 0
        RestoreKeyMap m; EXPECT(!m.add(ResourceKey{Context::target, 0, 7, static_cast<Kind>(9)}));
        EXPECT(!layout_matches(v, true, false) && !layout_matches(v, false, true) && layout_matches(v, true, true));
        std::map<std::pair<Context, int>, std::vector<ResourceKey>> by_slot{{{Context::target, 0}, keys_of(v)}};
        std::vector<ResourceKey> decl;
        EXPECT(restore_declared_keys(by_slot, true, true, false, decl) && decl.size() == 2);
        EXPECT(!restore_declared_keys(by_slot, true, true, true, decl));      // draft declared but absent
        EXPECT(restore_declared_keys(by_slot, true, false, false, decl) && decl.size() == 1); // rec undeclared stays out -> planner refuses
    }
    // --- pair without key: charge refused (kind 1 mapped to nothing; never charged to attention) ------------
    {
        Hyb d(3, 2, 200); Planner p; auto before1 = dump(*d.mem, 1);
        auto o = cycle(p, d, blob, 1, Mode::none, true);
        EXPECT(!o.restored && o.limit_refused && o.unmapped == 1 && p.size() == 0 && dump(*d.mem, 1) == before1);
        auto v = d.view(0); EXPECT(pick(v, Kind::recurrent_rows)->sequence == 0 && pick(v, Kind::attention_cells)->sequence == 0);
    }
    // --- T4: injected failures, 100 cycles each ---------------------------------------------------------------
    for (Mode mode : {Mode::fail_rec_first, Mode::fail_rec_second_layer}) {
        Planner planner; uint64_t task = 100; int clean = 0, competitor = 0;
        for (int c = 0; c < 100; ++c) {
            Hyb d(3, 2, 200 + c); const auto other = dump(*d.mem, 1);
            const auto o = cycle(planner, d, blob, ++task, inject ? mode : Mode::none);
            // expectation is ALWAYS failure: with --disable-injection (negative control) this must trip
            EXPECT(!o.restored);
            auto v = d.view(0);
            const bool empty = !o.restored && v.size() == 2 && pick(v, Kind::recurrent_rows)->sequence == 0 &&
                               pick(v, Kind::attention_cells)->sequence == 0;
            if (empty && dump(*d.mem, 1) == other && planner.size() == 0) ++clean;
            // competitor admitted after release (same capacity the restore had reserved)
            auto cs = snaps_of(d.view(1)); Reservation res{Owner{7, ++task, 1}, std::nullopt, {{at, 2}, {rc, 1}}};
            auto adm = planner.admit(cs, {res});
            if (adm.status == Status::accepted) { ++competitor; EXPECT(planner.native_release(res.owner).status == Status::accepted); }
            EXPECT(planner.size() == 0);
        }
        EXPECT(clean == 100); EXPECT(competitor == 100);
        std::printf("T4 mode=%d inject=%d clean_cycles=%d competitor_admitted=%d ledger=%zu\n", int(mode), inject, clean, competitor, planner.size());
    }
    // --- success path (bridge by (kind,stream)): bytes equal, other seq untouched, delta used == cell_count ------
    {
        Hyb d(3, 2, 200); Planner p; const auto other = dump(*d.mem, 1);
        auto before = d.view(0); const uint64_t used_before = pick(before, Kind::recurrent_rows)->used;
        auto o = cycle(p, d, blob, 1, Mode::none);
        EXPECT(o.restored && o.clears >= 1 && dump(*d.mem, 0) == blob && dump(*d.mem, 1) == other && p.size() == 0);
        auto after = d.view(0); EXPECT(pick(after, Kind::recurrent_rows)->used == used_before);
        std::printf("success restored=1 bytes_equal=%d rec_used=%u->%u\n", int(dump(*d.mem, 0) == blob), unsigned(used_before), unsigned(pick(after, Kind::recurrent_rows)->used));
    }
    return g_fail ? 1 : 0;
}
int main(int argc, char ** argv) {
    const bool inject = !(argc == 2 && std::strcmp(argv[1], "--disable-injection") == 0);
    const int rc = run(inject);
    std::printf("%s fails=%d\n", rc ? "FAILED" : "PASSED", g_fail);
    return rc;
}
