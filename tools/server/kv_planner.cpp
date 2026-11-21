#include "kv_planner.hpp"
#include <limits>
#include <algorithm>
#include <cinttypes>
#include <set>
#include <tuple>
#ifdef LLAMA_KV_ADMISSION_CPU_TEST_DRIVER
#include <cstdio>
#endif

namespace kvplan {
namespace {
bool add(uint64_t a, uint64_t b, uint64_t &out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    out = a + b; return true;
}
Result reject(const char *why) { return {Status::rejected, why}; }
}

Result Planner::admit(const std::vector<Snapshot>& snapshots, const std::vector<Reservation>& batch) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (batch.empty()) return reject("empty batch");
    std::map<ResourceKey,const Snapshot*> by_key;
    for (const auto& s : snapshots) {
        if (!s.known || s.key.epoch == 0 || (s.key.kind != Kind::attention_cells && s.key.kind != Kind::recurrent_rows) || s.used_real > s.capacity || s.seq_exclusive_real > s.used_real)
            return reject("unknown or invalid snapshot");
        if (!by_key.emplace(s.key,&s).second) return reject("duplicate snapshot");
        uint64_t measured = 0;
        for (const auto& x : s.exclusive_by_owner) {
            if (!ledger_.count(x.first) || !add(measured,x.second,measured)) return reject("invalid exclusive owner/overflow");
            const auto& e=ledger_.at(x.first);
            if (!e.peaks.count(s.key)) return reject("exclusive owner has no commitment on resource");
            if (x.first.epoch != s.key.epoch) return reject("exclusive owner epoch mismatch");
        }
        if (measured > s.seq_exclusive_real) return reject("exclusive measurements exceed sequence total");
    }
    std::map<Owner,const Reservation*> incoming;
    for (const auto& r : batch) {
        if (!r.owner.epoch || r.peak_by_resource.empty())
            return reject("invalid owner or empty peaks");
        if (ledger_.count(r.owner) || released_.count(r.owner) || !incoming.emplace(r.owner,&r).second)
            return reject("owner already used");
        for (const auto& p : r.peak_by_resource) {
            auto s=by_key.find(p.first);
            if (s==by_key.end() || p.first.epoch!=r.owner.epoch || s->second->key.epoch!=r.owner.epoch)
                return reject("missing or mismatched resource epoch");
        }
    }
    for (const auto& [key,s] : by_key) {
        uint64_t reused=0,active=0;
        for (const auto& x : s->exclusive_by_owner) {
            if (!add(active,x.second,active)) return reject("active exclusive overflow");
        }
        for (const auto& x : s->reusable_recurrent_by_incoming_owner) {
            auto found=incoming.find(x.first);
            if (key.kind!=Kind::recurrent_rows || x.second!=1 || found==incoming.end() ||
                x.first.epoch!=key.epoch || !found->second->peak_by_resource.count(key) ||
                found->second->peak_by_resource.at(key)!=1 || s->exclusive_by_owner.count(x.first))
                return reject("invalid incoming recurrent reuse binding");
            if (!add(reused,x.second,reused)) return reject("recurrent reuse overflow");
        }
        uint64_t total;
        if (!add(active,reused,total) || total>s->seq_exclusive_real)
            return reject("recurrent reuse exceeds authentic exclusive rows");
        uint64_t resident=0;
        for (const auto& x : s->resident_credit_by_incoming_owner) {
            auto found=incoming.find(x.first);
            if (found==incoming.end() || !x.second || x.first.epoch!=key.epoch ||
                !found->second->peak_by_resource.count(key) || x.second>found->second->peak_by_resource.at(key) ||
                found->second->parent || s->exclusive_by_owner.count(x.first) ||
                s->reusable_recurrent_by_incoming_owner.count(x.first))
                return reject("invalid incoming resident credit binding");
            if (!add(resident,x.second,resident)) return reject("resident credit overflow");
        }
        uint64_t resident_total;
        if (!add(s->seq_exclusive_real,resident,resident_total) || resident_total>s->used_real)
            return reject("resident credit exceeds authentic physical cells");
    }
    for (const auto& item : incoming) if (item.second->parent) {
        if (!ledger_.count(*item.second->parent) && !incoming.count(*item.second->parent)) return reject("missing parent");
        if (item.second->parent->epoch != item.first.epoch) return reject("parent epoch mismatch");
    }
    for (const auto& [key,s] : by_key) {
        uint64_t total=s->used_real;
        for (const auto& [owner,e] : ledger_) {
            auto peak=e.peaks.find(key); if (peak==e.peaks.end()) continue;
            auto ex=s->exclusive_by_owner.find(owner);
            const uint64_t exclusive=ex==s->exclusive_by_owner.end()?0:ex->second;
            const uint64_t committed_credit=std::min(exclusive, peak->second);
            const uint64_t growth=peak->second-committed_credit;
#ifdef LLAMA_KV_ADMISSION_CPU_TEST_DRIVER
            std::fprintf(stderr, "KV_ADMISSION_TEST applied_credit owner_epoch=%" PRIu64 " owner_task=%" PRIu64
                    " owner_slot=%" PRIu64 " context=%u stream=%u kind=%u physical_exclusive=%" PRIu64
                    " committed=%" PRIu64 " applied_credit=%" PRIu64 " growth=%" PRIu64 "\n",
                    owner.epoch, owner.task, owner.slot, (unsigned) key.context, key.stream, (unsigned) key.kind,
                    exclusive, peak->second, committed_credit, growth);
#endif
            if (!add(total,growth,total)) return reject("active growth overflow");
        }
        for (const auto& [owner,r] : incoming) {
            auto peak=r->peak_by_resource.find(key);
            if (peak!=r->peak_by_resource.end()) {
                const auto reuse=s->reusable_recurrent_by_incoming_owner.find(owner);
                const auto resident=s->resident_credit_by_incoming_owner.find(owner);
                const uint64_t credit=(reuse==s->reusable_recurrent_by_incoming_owner.end()?0:reuse->second)+
                    (resident==s->resident_credit_by_incoming_owner.end()?0:resident->second);
                const uint64_t growth=peak->second>credit?peak->second-credit:0;
                if (!add(total,growth,total)) return reject("batch peak overflow");
            }
        }
        if (total>s->capacity) return reject("capacity exceeded");
    }
    Result success{Status::accepted,"batch admitted atomically"};
    auto staged=ledger_;
    for (const auto& [owner,r] : incoming) staged.emplace(owner,Entry{owner,r->parent,r->peak_by_resource});
    ledger_.swap(staged);
    return success;
}

Result Planner::native_release(const Owner& owner) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it=ledger_.find(owner);
    const bool tombstone_exists=released_.find(owner)!=released_.end();
    if (it==ledger_.end())
        return tombstone_exists ? Result{Status::already_released,"exact duplicate release"} : reject("owner identity mismatch or absent");
    if (tombstone_exists) return reject("active ledger conflicts with release tombstone");
    if (it->second.restore && !it->second.fenced) return reject("restore requires native output fence");

    // All potentially allocating work precedes the commit point. Returning this
    // local uses Result's noexcept move if NRVO is not selected.
    Result success{Status::accepted,"native release applied"};
    const auto inserted=released_.emplace(owner,true);
    if (!inserted.second) return reject("release tombstone already exists");
    ledger_.erase(it); // iterator erase is non-allocating; nothing allocates after this point.
    return success;
}


uint64_t Planner::restore_available(const Snapshot& s) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!s.known || !s.key.epoch || s.used_real>s.capacity || s.seq_exclusive_real>s.used_real) return 0;
    uint64_t measured=0;
    for (const auto& [owner,exclusive] : s.exclusive_by_owner) {
        auto e=ledger_.find(owner);
        if (e==ledger_.end() || !e->second.peaks.count(s.key) || owner.epoch!=s.key.epoch ||
            !add(measured,exclusive,measured)) return 0;
    }
    if (measured>s.seq_exclusive_real) return 0;
    uint64_t total;
    if (!active_total(s,total)) return 0;
    return total>=s.capacity?0:s.capacity-total;
}

RestorePlan Planner::plan_restore(const std::vector<Snapshot>& snapshots, const Owner& owner,
                                  const std::vector<ResourceKey>& declared) {
    std::lock_guard<std::mutex> lock(mutex_);
    using K=RestorePlan::Kind;
    RestorePlan result;
    auto refuse=[&](K kind,const char* reason) -> RestorePlan {
        RestorePlan failure; failure.kind=kind;failure.reason=reason;return failure;
    };
    if (!owner.epoch || ledger_.count(owner) || released_.count(owner))
        return refuse(K::refused_identity,"restore owner invalid or already used");
    if (snapshots.empty()) return refuse(K::refused_unknown,"empty resource set");
    // Declared layout: a hybrid model declares attention + recurrent; every declared key must
    // be present and nothing undeclared may appear. Empty = legacy behaviour (no declaration).
    if (!declared.empty()) {
        std::set<ResourceKey> want,have;
        for (auto k:declared) { k.epoch=owner.epoch; want.insert(k); }
        for (const auto& s:snapshots) have.insert(s.key);
        if (want!=have) return refuse(K::refused_unsupported_layout,"snapshot set differs from declared layout");
    }
    std::map<ResourceKey,uint64_t> limits;
    for (const auto& s:snapshots) {
        if (!s.known || !s.key.epoch) return refuse(K::refused_unknown,"unknown resource");
        if (s.key.epoch!=owner.epoch || limits.count(s.key))
            return refuse(K::refused_identity,"epoch mismatch or duplicate resource");
        // Recurrent restore remains unproved: never interpret recurrent logical CELL_COUNT as rows.
        // Recurrent CELL_COUNT is a logical cell count in pool units (no g multiplication); accepted only
        // for the target context, stream 0. Any other kind/context/stream fails closed.
        const bool rec=s.key.kind==Kind::recurrent_rows;
        if ((s.key.kind!=Kind::attention_cells && !rec) ||
            (s.key.context!=Context::target && s.key.context!=Context::draft) ||
            (rec && (s.key.context!=Context::target || s.key.stream!=0)))
            return refuse(K::refused_unsupported_layout,"unproved resource layout");
        if (s.used_real>s.capacity) return refuse(K::refused_overcommit,"physical overcommit");
        if (s.seq_exclusive_real>s.used_real || (!rec && !s.reusable_recurrent_by_incoming_owner.empty()))
            return refuse(K::refused_identity,"invalid physical measurement or replacement credit");
        // Replacement credit (D1): exclusive cells of the destination itself, only for its own owner.
        uint64_t credit=0;
        for (const auto& [o,c]:s.reusable_recurrent_by_incoming_owner) {
            if (!(o==owner)) return refuse(K::refused_identity,"replacement credit not bound to incoming owner");
            credit=c;
        }
        if (credit>s.seq_exclusive_real) return refuse(K::refused_identity,"replacement credit exceeds exclusive cells");
        uint64_t measured=0;
        for (const auto& [o,exclusive]:s.exclusive_by_owner) {
            const auto it=ledger_.find(o);
            if (o.epoch!=s.key.epoch || it==ledger_.end() || !it->second.peaks.count(s.key))
                return refuse(K::refused_identity,"exclusive measurement not bound to native owner");
            if (!add(measured,exclusive,measured)) return refuse(K::refused_overflow,"exclusive sum overflow");
        }
        if (measured>s.seq_exclusive_real) return refuse(K::refused_identity,"exclusive sum exceeds measurement");
        uint64_t total;
        if (!active_total(s,total)) return refuse(K::refused_overflow,"active growth overflow");
        if (total>s.capacity) return refuse(K::refused_overcommit,"active growth exceeds capacity");
        if (total==s.capacity && s.capacity>0 && credit==0) return refuse(K::refused_overcommit,"restore has no unreserved capacity");
        limits.emplace(s.key,s.capacity-total+credit);
    }
    // Allocate all token/result storage before the atomic ledger commit. Full
    // limits are temporary promised growth, with ZERO destination-old-state credit.
    result.kind=K::ok;result.reason="restore resources reserved atomically";
    result.token.owner_=owner;result.token.limits_=limits;
    for (const auto& [key,limit]:limits) { (void)limit;result.token.consumed_.emplace(key,0); }
    auto staged=ledger_;
    staged.emplace(owner,Entry{owner,std::nullopt,std::move(limits),true,false});
    ledger_.swap(staged);
    result.token.issuer_=this;
    return result;
}

bool Planner::consume_restore(RestoreToken& token,const ResourceKey& key,uint64_t count) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (token.issuer_!=this || token.owner_.epoch!=key.epoch) return false;
    const auto entry=ledger_.find(token.owner_);
    if (entry==ledger_.end() || !entry->second.restore || entry->second.fenced) return false;
    const auto limit=token.limits_.find(key);
    const auto consumed=token.consumed_.find(key);
    if (limit==token.limits_.end() || consumed==token.consumed_.end()) return false;
    uint64_t next;
    if (!add(consumed->second,count,next) || next>limit->second) return false;
    consumed->second=next;return true;
}

bool Planner::fence_restore(const RestoreToken& token) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (token.issuer_!=this) return false;
    const auto entry=ledger_.find(token.owner_);
    if (entry==ledger_.end() || !entry->second.restore) return false;
    entry->second.fenced=true;return true;
}
bool Planner::active_total(const Snapshot& s,uint64_t& total) const noexcept {
    total=s.used_real;
    for (const auto& [owner,e]:ledger_) {
        const auto peak=e.peaks.find(s.key);if (peak==e.peaks.end()) continue;
        const auto ex=s.exclusive_by_owner.find(owner);
        const uint64_t exclusive=ex==s.exclusive_by_owner.end()?0:ex->second;
        const uint64_t growth=peak->second>exclusive?peak->second-exclusive:0;
        if (!add(total,growth,total)) return false;
    }
    return true;
}
} // namespace kvplan
