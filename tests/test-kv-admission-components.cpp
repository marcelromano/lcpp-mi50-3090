#include "../common/common.h"
#include "../tools/server/kv_planner.hpp"
#include "../tools/server/kv_resource_view.hpp"

#include <cstdio>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    using namespace kvplan;

    // Scope policy shared with the actual dispatch/parser/snapshot paths.
    CHECK(common_params{}.kv_admission_scope == common_kv_admission_scope::all);
    CHECK(common_parse_kv_admission_scope("all") == common_kv_admission_scope::all);
    const auto restore_scope = common_parse_kv_admission_scope("restore");
    CHECK(common_kv_admission_plain_bypass(restore_scope));
    CHECK(!common_kv_admission_plain_bypass(common_kv_admission_scope::all));
    CHECK(common_kv_admission_snapshot_owner_valid(restore_scope, true, false));
    CHECK(!common_kv_admission_snapshot_owner_valid(common_kv_admission_scope::all, true, false));
    CHECK(!common_kv_admission_snapshot_owner_valid(restore_scope, false, true));
    bool invalid_scope = false;
    try { (void) common_parse_kv_admission_scope("RESTORE"); }
    catch (const std::invalid_argument &) { invalid_scope = true; }
    CHECK(invalid_scope);
    {
        const ResourceKey rk{Context::target, 0, 7, Kind::recurrent_rows};
        Snapshot s; s.key = rk; s.known = true; s.capacity = 2; s.used_real = 1;
        const Owner incoming_restore{7, 2, 1};
        Planner p;
        auto plan = p.plan_restore({s}, incoming_restore, {rk});
        CHECK(plan.kind == RestorePlan::Kind::ok); // active unowned neighbour occupies one real cell
        CHECK(p.consume_restore(plan.token, rk, 1));
        CHECK(!p.consume_restore(plan.token, rk, 1)); // no capacity left: fail closed
        CHECK(p.fence_restore(plan.token));
        CHECK(p.native_release(incoming_restore).status == Status::accepted);
        s.used_real = 3;
        CHECK(p.plan_restore({s}, Owner{7, 3, 1}, {rk}).kind != RestorePlan::Kind::ok);
        Reservation r; r.owner = incoming_restore; r.peak_by_resource.emplace(rk, 2);
        s.used_real = 1;
        CHECK(p.admit({s}, {r}).status == Status::rejected); // old all policy still refuses plain
    }

    // Recurrent rows are bundle counts. Each raw field must be divisible on its own.
    llama_memory_resource_capacity raw{};
    raw.recurrent_known = true;
    raw.recurrent_capacity_rows = 12;
    raw.recurrent_used_rows = 4;
    raw.recurrent_seq_rows = 4;
    raw.recurrent_seq_exclusive_rows = 2;
    raw.recurrent_rollback_groups = 2;
    std::vector<PhysicalView> views;
    CHECK(normalize_resources(Context::target, 7, raw, views));
    CHECK(views.size() == 1 && views[0].capacity == 6 && views[0].exclusive == 1);
    for (int field = 0; field < 4; ++field) {
        auto bad = raw;
        if (field == 0) ++bad.recurrent_capacity_rows;
        if (field == 1) ++bad.recurrent_used_rows;
        if (field == 2) ++bad.recurrent_seq_rows;
        if (field == 3) ++bad.recurrent_seq_exclusive_rows;
        CHECK(!normalize_resources(Context::target, 7, bad, views));
        CHECK(views.empty()); // invalid recurrent field cannot leave a partial view
    }

    // Target and draft keys retain role, stream and epoch as independent dimensions.
    llama_memory_resource_capacity attention{};
    attention.attention_known = true;
    attention.attention_stream_count = 2;
    attention.attention_stream_id = 1;
    attention.attention_capacity_cells = 32;
    CHECK(normalize_resources(Context::target, 7, attention, views));
    const auto target_key = views[0].key;
    CHECK(normalize_resources(Context::draft, 8, attention, views));
    CHECK(target_key.context == Context::target && target_key.stream == 1 && target_key.epoch == 7);
    CHECK(views[0].key.context == Context::draft && views[0].key.stream == 1 && views[0].key.epoch == 8);

    Planner planner;
    const Owner owner{7, 0, 0}; // task and slot zero are valid native identifiers.
    const ResourceKey key{Context::target, 0, 7, Kind::attention_cells};
    Snapshot snapshot;
    snapshot.key = key;
    snapshot.known = true;
    snapshot.capacity = 100;
    snapshot.used_real = 20;
    Reservation reservation;
    reservation.owner = owner;
    reservation.peak_by_resource.emplace(key, 5);
    CHECK(planner.admit({snapshot}, {reservation}).status == Status::accepted);
    snapshot.seq_exclusive_real = 20;
    snapshot.exclusive_by_owner.emplace(owner, 20);
    Reservation incoming;
    incoming.owner = Owner{7, 1, 1};
    incoming.peak_by_resource.emplace(key, 5);
    CHECK(planner.admit({snapshot}, {incoming}).status == Status::accepted);
    CHECK(planner.size() == 2); // committed owner plus incoming; oversize physical prefix is clipped.
    CHECK(planner.restore_available(snapshot) == 75); // existing credit clips to its peak; the new reservation grows by five.

    auto unknown = snapshot;
    unknown.known = false;
    Planner rejects_unknown;
    CHECK(rejects_unknown.admit({unknown}, {reservation}).status == Status::rejected);
    CHECK(rejects_unknown.size() == 0);
    auto unbound = snapshot;
    unbound.exclusive_by_owner.clear();
    unbound.exclusive_by_owner.emplace(Owner{7, 77, 0}, 1);
    unbound.seq_exclusive_real = 1;
    Planner rejects_binding;
    CHECK(rejects_binding.admit({unbound}, {reservation}).status == Status::rejected);
    CHECK(rejects_binding.size() == 0);
    auto wrong_epoch = snapshot;
    wrong_epoch.key.epoch = 8;
    wrong_epoch.exclusive_by_owner.clear();
    Planner rejects_epoch;
    auto mismatched = reservation;
    mismatched.peak_by_resource.clear();
    mismatched.peak_by_resource.emplace(wrong_epoch.key, 5);
    CHECK(rejects_epoch.admit({wrong_epoch}, {mismatched}).status == Status::rejected);
    CHECK(rejects_epoch.size() == 0);
    auto absent_resource = reservation;
    absent_resource.peak_by_resource.clear();
    absent_resource.peak_by_resource.emplace(ResourceKey{Context::draft, 0, 7, Kind::attention_cells}, 1);
    Planner rejects_resource;
    CHECK(rejects_resource.admit({snapshot}, {absent_resource}).status == Status::rejected);
    CHECK(rejects_resource.size() == 0);

    // resident credit: idle target-slot residente creditado uma vez, ligado a owner/epoch/recurso.
    {
        const ResourceKey rk{Context::target, 0, 7, Kind::attention_cells};
        const ResourceKey dk{Context::draft, 0, 7, Kind::attention_cells};
        auto mk = [&](uint64_t used) { Snapshot x; x.key = rk; x.known = true; x.capacity = 256; x.used_real = used; return x; };
        auto rsv = [&](Owner o, uint64_t peak) { Reservation r; r.owner = o; r.peak_by_resource.emplace(rk, peak); return r; };
        const Owner in{7, 5, 0};
        // 131 residentes + 132 de pico > 256 sem credito; com credito 131 cabe (limiar 252).
        { Planner p; CHECK(p.admit({mk(131)}, {rsv(in, 132)}).reason == "capacity exceeded"); CHECK(p.size() == 0); }
        { Planner p; auto s1 = mk(131); s1.resident_credit_by_incoming_owner.emplace(in, 131);
          CHECK(p.admit({s1}, {rsv(in, 132)}).status == Status::accepted && p.size() == 1); }
        { Planner p; auto s1 = mk(131); s1.resident_credit_by_incoming_owner.emplace(in, 131);
          CHECK(p.admit({s1}, {rsv(in, 257)}).status == Status::rejected); }   // capacidade real excedida
        { Planner p; auto s1 = mk(131); s1.resident_credit_by_incoming_owner.emplace(in, 131);
          CHECK(p.admit({s1}, {rsv(in, 252)}).status == Status::accepted); }    // 131-131+252 == 252 <= 256
        // negativos de binding
        { Planner p; auto s1 = mk(131); s1.resident_credit_by_incoming_owner.emplace(Owner{7, 6, 0}, 131);   // owner errado
          CHECK(p.admit({s1}, {rsv(in, 132)}).reason == "invalid incoming resident credit binding" && p.size() == 0); }
        { Planner p; auto s1 = mk(131); s1.resident_credit_by_incoming_owner.emplace(Owner{8, 5, 0}, 131);   // epoch errado
          CHECK(p.admit({s1}, {rsv(in, 132)}).status == Status::rejected && p.size() == 0); }
        { Planner p; auto s1 = mk(131); s1.resident_credit_by_incoming_owner.emplace(in, 133);                // credito > pico
          CHECK(p.admit({s1}, {rsv(in, 132)}).status == Status::rejected); }
        { Planner p; auto s1 = mk(131); s1.resident_credit_by_incoming_owner.emplace(in, 132);                // credito > fisico
          CHECK(p.admit({s1}, {rsv(in, 140)}).status == Status::rejected); }
        { Planner p; auto s1 = mk(131); s1.resident_credit_by_incoming_owner.emplace(in, 0);                  // zero nao e binding
          CHECK(p.admit({s1}, {rsv(in, 132)}).status == Status::rejected); }
        { Planner p; auto s1 = mk(131); s1.resident_credit_by_incoming_owner.emplace(in, 10);
          s1.reusable_recurrent_by_incoming_owner.emplace(in, 1);                                             // duplo credito
          CHECK(p.admit({s1}, {rsv(in, 132)}).status == Status::rejected); }
        { Planner p; auto s1 = mk(131); Reservation r = rsv(in, 132);                                         // filho: sem credito
          Reservation parent = rsv(Owner{7, 4, 1}, 4); s1.used_real = 135; s1.resident_credit_by_incoming_owner.emplace(in, 131);
          r.parent = parent.owner; auto s2 = s1; s2.capacity = 256;
          CHECK(p.admit({s2}, {parent, r}).reason == "invalid incoming resident credit binding"); }
        // dono ativo no mesmo recurso: credito do idle nao pode reaproveitar exclusivo ativo
        { Planner p; const Owner act{7, 1, 1}; CHECK(p.admit({mk(0)}, {rsv(act, 10)}).status == Status::accepted);
          auto s1 = mk(10); s1.seq_exclusive_real = 10; s1.exclusive_by_owner.emplace(act, 10);
          s1.resident_credit_by_incoming_owner.emplace(in, 5);                                                // 10+5 > 10 fisico
          CHECK(p.admit({s1}, {rsv(in, 20)}).status == Status::rejected && p.size() == 1);
          auto s2 = mk(15); s2.seq_exclusive_real = 10; s2.exclusive_by_owner.emplace(act, 10);
          s2.resident_credit_by_incoming_owner.emplace(in, 5);                                                // 10+5 <= 15
          CHECK(p.admit({s2}, {rsv(in, 20)}).status == Status::accepted && p.size() == 2); }
        // recurso target+draft em AND: sem a chave do draft no snapshot a reserva e recusada
        { Planner p; Reservation r = rsv(in, 10); r.peak_by_resource.emplace(dk, 10); auto s1 = mk(50);
          CHECK(p.admit({s1}, {r}).status == Status::rejected); }
    }
    return 0;
}
