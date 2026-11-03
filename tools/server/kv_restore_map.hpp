#pragma once
// engine-side helpers for restore of hybrid (attention + recurrent) state under KV admission.
// Header-only so the server and the CPU tests exercise the very same logic.
#include "kv_resource_view.hpp"
#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace kvplan {

// llama_state_seq_limits.consume identifies a charge as (kind, stream): kind 0 = attention cells
// (stream = destination stream), kind 1 = recurrent rows (stream 0).
inline std::pair<uint32_t, uint32_t> restore_pair(const ResourceKey & k) {
    return k.kind == Kind::recurrent_rows ? std::make_pair(1u, 0u) : std::make_pair(0u, k.stream);
}

// (kind, stream) -> ResourceKey for ONE context of the restore destination. A charge for a pair
// without a key is refused (fail-closed); no pair ever falls back to another key.
struct RestoreKeyMap {
    std::map<std::pair<uint32_t, uint32_t>, ResourceKey> keys;
    bool add(const ResourceKey & k) {
        if (k.kind != Kind::attention_cells && k.kind != Kind::recurrent_rows) return false;
        return keys.emplace(restore_pair(k), k).second;
    }
    const ResourceKey * find(uint32_t kind, uint32_t stream) const {
        const auto it = keys.find({kind, stream});
        return it == keys.end() ? nullptr : &it->second;
    }
    bool has(Kind kind) const {
        return std::any_of(keys.begin(), keys.end(), [kind](const auto & i) { return i.second.kind == kind; });
    }
};

// Charge from the loader (restore_cells -> consume). Returns false when the pair has no key or the
// planner refuses; `unmapped` counts pairs without key.
inline bool restore_consume(Planner & planner, RestoreToken & token, const RestoreKeyMap & map,
                            uint32_t kind, uint32_t stream, uint64_t count, uint64_t & unmapped) {
    const ResourceKey * key = map.find(kind, stream);
    if (!key) { ++unmapped; return false; }
    return planner.consume_restore(token, *key, count);
}

// Layout the model declares: target attention (non-recurrent or hybrid), target recurrent
// (recurrent or hybrid), draft attention only. Keys come from the slots' physical views; a
// required kind without any key, or a key of an undeclared kind, makes the layout unknown.
inline bool restore_declared_keys(const std::map<std::pair<Context, int>, std::vector<ResourceKey>> & by_slot,
                                  bool target_attention, bool target_recurrent, bool has_draft,
                                  std::vector<ResourceKey> & out) {
    out.clear();
    bool t_att = false, t_rec = false, d_att = false;
    for (const auto & item : by_slot) {
        const bool target = item.first.first == Context::target;
        for (const auto & k : item.second) {
            if (target && k.kind == Kind::attention_cells && target_attention) t_att = true;
            else if (target && k.kind == Kind::recurrent_rows && target_recurrent) t_rec = true;
            else if (!target && k.kind == Kind::attention_cells && has_draft) d_att = true;
            else continue; // undeclared: stays out of `declared`, so plan_restore refuses the mismatch
            if (std::find(out.begin(), out.end(), k) == out.end()) out.push_back(k);
        }
    }
    return (!target_attention || t_att) && (!target_recurrent || t_rec) && (!has_draft || d_att);
}

// Exact physical layout check (fence/observation): the declared kinds, each exactly once.
inline bool layout_matches(const std::vector<PhysicalView> & physical, bool expect_attention, bool expect_recurrent) {
    size_t att = 0, rec = 0;
    for (const auto & p : physical) {
        if (p.key.kind == Kind::attention_cells) ++att;
        else if (p.key.kind == Kind::recurrent_rows) ++rec;
    }
    return physical.size() == size_t(expect_attention) + size_t(expect_recurrent) &&
           att == size_t(expect_attention) && rec == size_t(expect_recurrent);
}

// D1 replacement credit. `dest` is the physical view of the IDLE destination slot (read at the restore
// boundary). The native transition removes the destination's old sequence (seq_rm) before any growth,
// so its exclusive recurrent cells are credited once, bound to the incoming owner. A recurrent sequence
// whose cells are shared with another sequence (n_seq_id>1) is refused. The credit is also added to
// seq_exclusive_real (an idle slot contributes 0 there) so Planner's own bounds (credit <=
// seq_exclusive_real <= used_real) still apply; the planner never gives more than capacity-total+credit.
inline bool apply_restore_credit(const std::vector<PhysicalView> & dest, const Owner & owner,
                                 std::vector<Snapshot> & snapshots, std::string & reason) {
    for (const auto & v : dest) {
        if (v.key.kind != Kind::recurrent_rows || v.key.context != Context::target) continue;
        if (v.sequence != v.exclusive) { reason = "shared_recurrent_sequence"; return false; }
        if (!v.exclusive) continue;
        const auto snap = std::find_if(snapshots.begin(), snapshots.end(), [&](const Snapshot & s) { return s.key == v.key; });
        if (snap == snapshots.end()) { reason = "missing_recurrent_snapshot"; return false; }
        if (snap->seq_exclusive_real + v.exclusive > snap->used_real) { reason = "credit_exceeds_used"; return false; }
        snap->seq_exclusive_real += v.exclusive;
        snap->reusable_recurrent_by_incoming_owner[owner] = v.exclusive;
    }
    return true;
}

// TEST-ONLY (called only under LLAMA_B3_LIMIT_FIXTURE): force the target restore limit of one kind to n
// by faking the physical occupancy seen by the planner (used_real = capacity - n); drops any credit.
inline bool apply_limit_fixture(std::vector<Snapshot> & snapshots, Kind kind, uint64_t n) {
    bool hit = false;
    for (auto & sn : snapshots) {
        if (sn.key.kind == kind && sn.key.context == Context::target && n <= sn.capacity) {
            sn.used_real = sn.capacity - n;
            if (sn.seq_exclusive_real > sn.used_real) sn.seq_exclusive_real = sn.used_real;
            sn.reusable_recurrent_by_incoming_owner.clear(); // forced limit is exact: no replacement credit
            hit = true;
        }
    }
    return hit;
}

} // namespace kvplan
